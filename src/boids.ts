import * as d3 from 'd3';

export {};

// ── Vector2 helpers ──────────────────────────────────────────────────────────

interface Vec2 { x: number; y: number; }

function v(x: number, y: number): Vec2 { return { x, y }; }
function vadd(a: Vec2, b: Vec2): Vec2 { return { x: a.x + b.x, y: a.y + b.y }; }
function vsub(a: Vec2, b: Vec2): Vec2 { return { x: a.x - b.x, y: a.y - b.y }; }
function vscale(a: Vec2, s: number): Vec2 { return { x: a.x * s, y: a.y * s }; }
function vlen(a: Vec2): number { return Math.sqrt(a.x * a.x + a.y * a.y); }
function vnorm(a: Vec2): Vec2 {
  const l = vlen(a);
  return l > 0 ? vscale(a, 1 / l) : { x: 0, y: 0 };
}
function vlimit(a: Vec2, max: number): Vec2 {
  const l = vlen(a);
  return l > max ? vscale(a, max / l) : a;
}
function vdist(a: Vec2, b: Vec2): number { return vlen(vsub(a, b)); }

// ── Boid ─────────────────────────────────────────────────────────────────────

interface Trail { x: number; y: number; }

class Boid {
  pos: Vec2;
  vel: Vec2;
  acc: Vec2;
  trail: Trail[];
  isPredator: boolean;

  constructor(x: number, y: number, predator = false) {
    this.pos = v(x, y);
    const angle = Math.random() * Math.PI * 2;
    const speed = 1 + Math.random() * 2;
    this.vel = v(Math.cos(angle) * speed, Math.sin(angle) * speed);
    this.acc = v(0, 0);
    this.trail = [];
    this.isPredator = predator;
  }
}

// ── Simulation params ────────────────────────────────────────────────────────

interface Params {
  numBoids: number;
  separationWeight: number;
  alignmentWeight: number;
  cohesionWeight: number;
  perceptionRadius: number;
  maxSpeed: number;
  maxForce: number;
  edgeMode: 'wrap' | 'bounce';
  predatorOn: boolean;
  trailsOn: boolean;
  colorMode: 'speed' | 'density';
  showDebug: boolean;
  simSpeed: number;
}

const defaultParams: Params = {
  numBoids: 150,
  separationWeight: 1.5,
  alignmentWeight: 1.0,
  cohesionWeight: 1.0,
  perceptionRadius: 60,
  maxSpeed: 3.5,
  maxForce: 0.15,
  edgeMode: 'wrap',
  predatorOn: false,
  trailsOn: false,
  colorMode: 'speed',
  showDebug: false,
  simSpeed: 1,
};

// ── Simulation ───────────────────────────────────────────────────────────────

let canvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;
let W: number, H: number;
let boids: Boid[] = [];
let params: Params = { ...defaultParams };
let mouse: Vec2 | null = null;
let mouseRepel = false;
let running = true;
let animId = 0;
const TRAIL_MAX = 20;
const PREDATOR_RADIUS = 100;

function initBoids(): void {
  boids = [];
  for (let i = 0; i < params.numBoids; i++) {
    boids.push(new Boid(Math.random() * W, Math.random() * H));
  }
  if (params.predatorOn) {
    boids.push(new Boid(W / 2, H / 2, true));
  }
}

function steerTowards(boid: Boid, target: Vec2, strength: number): Vec2 {
  const desired = vscale(vnorm(vsub(target, boid.pos)), params.maxSpeed);
  return vlimit(vscale(vsub(desired, boid.vel), strength), params.maxForce);
}

function applyRules(boid: Boid): void {
  if (boid.isPredator) return;

  const sep: Vec2 = v(0, 0);
  const ali: Vec2 = v(0, 0);
  const coh: Vec2 = v(0, 0);
  let sepCount = 0, aliCount = 0, cohCount = 0;

  const R = params.perceptionRadius;
  const R2 = R * R;

  boids.forEach((other) => {
    if (other === boid) return;

    // Flee predator
    if (other.isPredator) {
      const d = vdist(boid.pos, other.pos);
      if (d < PREDATOR_RADIUS && d > 0) {
        const flee = vscale(vnorm(vsub(boid.pos, other.pos)), params.maxSpeed);
        boid.acc = vadd(boid.acc, vscale(vsub(flee, boid.vel), 0.3));
      }
      return;
    }

    const dx = other.pos.x - boid.pos.x;
    const dy = other.pos.y - boid.pos.y;
    const d2 = dx * dx + dy * dy;
    if (d2 > R2 || d2 === 0) return;
    const d = Math.sqrt(d2);

    // Separation: steer away from close neighbours
    sep.x -= dx / d;
    sep.y -= dy / d;
    sepCount++;

    // Alignment: match velocity
    ali.x += other.vel.x;
    ali.y += other.vel.y;
    aliCount++;

    // Cohesion: steer towards centre of mass
    coh.x += other.pos.x;
    coh.y += other.pos.y;
    cohCount++;
  });

  if (sepCount > 0) {
    const desired = vscale(vnorm(v(sep.x / sepCount, sep.y / sepCount)), params.maxSpeed);
    const steer = vlimit(vsub(desired, boid.vel), params.maxForce);
    boid.acc = vadd(boid.acc, vscale(steer, params.separationWeight));
  }
  if (aliCount > 0) {
    const desired = vscale(vnorm(v(ali.x / aliCount, ali.y / aliCount)), params.maxSpeed);
    const steer = vlimit(vsub(desired, boid.vel), params.maxForce);
    boid.acc = vadd(boid.acc, vscale(steer, params.alignmentWeight));
  }
  if (cohCount > 0) {
    const centre = v(coh.x / cohCount, coh.y / cohCount);
    const steer = steerTowards(boid, centre, 1.0);
    boid.acc = vadd(boid.acc, vscale(steer, params.cohesionWeight));
  }

  // Mouse attraction / repulsion
  if (mouse) {
    const d = vdist(boid.pos, mouse);
    if (d < 150) {
      const dir = mouseRepel
        ? vnorm(vsub(boid.pos, mouse))
        : vnorm(vsub(mouse, boid.pos));
      const steer = vlimit(vsub(vscale(dir, params.maxSpeed), boid.vel), params.maxForce);
      boid.acc = vadd(boid.acc, vscale(steer, 2.0));
    }
  }
}

function updateBoid(boid: Boid): void {
  if (boid.isPredator) {
    // Predator chases nearest prey
    let nearest: Boid | null = null;
    let minD = Infinity;
    boids.forEach((other) => {
      if (other.isPredator) return;
      const d = vdist(boid.pos, other.pos);
      if (d < minD) { minD = d; nearest = other; }
    });
    if (nearest) {
      const steer = steerTowards(boid, (nearest as Boid).pos, 1.2);
      boid.acc = vadd(boid.acc, vscale(steer, 1.5));
    }
  }

  boid.vel = vlimit(vadd(boid.vel, boid.acc), boid.isPredator ? params.maxSpeed * 0.8 : params.maxSpeed);
  boid.acc = v(0, 0);

  // Enforce min speed
  const speed = vlen(boid.vel);
  if (speed < 0.5 && !boid.isPredator) {
    boid.vel = vscale(vnorm(boid.vel), 0.5);
  }

  // Trail
  if (params.trailsOn) {
    boid.trail.push({ x: boid.pos.x, y: boid.pos.y });
    if (boid.trail.length > TRAIL_MAX) boid.trail.shift();
  } else {
    boid.trail = [];
  }

  boid.pos = vadd(boid.pos, boid.vel);

  // Edge handling
  if (params.edgeMode === 'wrap') {
    if (boid.pos.x < 0) boid.pos.x += W;
    if (boid.pos.x > W) boid.pos.x -= W;
    if (boid.pos.y < 0) boid.pos.y += H;
    if (boid.pos.y > H) boid.pos.y -= H;
  } else {
    const margin = 20;
    const turnStrength = 0.3;
    if (boid.pos.x < margin) boid.vel.x += turnStrength;
    if (boid.pos.x > W - margin) boid.vel.x -= turnStrength;
    if (boid.pos.y < margin) boid.vel.y += turnStrength;
    if (boid.pos.y > H - margin) boid.vel.y -= turnStrength;
    boid.vel = vlimit(boid.vel, params.maxSpeed);
    boid.pos.x = Math.max(0, Math.min(W, boid.pos.x));
    boid.pos.y = Math.max(0, Math.min(H, boid.pos.y));
  }
}

// ── Color helpers ────────────────────────────────────────────────────────────

const speedScale = d3.scaleSequential(d3.interpolateTurbo).domain([0, 1]);
const densityScale = d3.scaleSequential(d3.interpolateCool).domain([0, 1]);

function boidColor(boid: Boid, density: number): string {
  if (boid.isPredator) return '#ff4444';
  if (params.colorMode === 'speed') {
    const t = Math.min(vlen(boid.vel) / params.maxSpeed, 1);
    return speedScale(t);
  } else {
    return densityScale(Math.min(density / 15, 1));
  }
}

// ── Draw ─────────────────────────────────────────────────────────────────────

function drawBoid(boid: Boid, color: string): void {
  const angle = Math.atan2(boid.vel.y, boid.vel.x);
  const size = boid.isPredator ? 12 : 7;

  ctx.save();
  ctx.translate(boid.pos.x, boid.pos.y);
  ctx.rotate(angle);

  ctx.beginPath();
  ctx.moveTo(size, 0);
  ctx.lineTo(-size * 0.6, size * 0.45);
  ctx.lineTo(-size * 0.3, 0);
  ctx.lineTo(-size * 0.6, -size * 0.45);
  ctx.closePath();

  ctx.fillStyle = color;
  ctx.fill();

  if (params.showDebug && !boid.isPredator) {
    // Perception radius
    ctx.beginPath();
    ctx.arc(0, 0, params.perceptionRadius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(88,166,255,0.12)';
    ctx.lineWidth = 0.7;
    ctx.stroke();

    // Velocity vector
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(boid.vel.x * 4, boid.vel.y * 4);
    ctx.strokeStyle = 'rgba(255,220,80,0.6)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  ctx.restore();
}

function drawTrail(boid: Boid, color: string): void {
  if (boid.trail.length < 2) return;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(boid.trail[0].x, boid.trail[0].y);
  for (let i = 1; i < boid.trail.length; i++) {
    ctx.lineTo(boid.trail[i].x, boid.trail[i].y);
  }
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.35;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();
}

// Compute local density (count of neighbours within perceptionRadius)
function computeDensities(): number[] {
  const R2 = params.perceptionRadius * params.perceptionRadius;
  return boids.map((b) => {
    let cnt = 0;
    boids.forEach((other) => {
      if (other === b || other.isPredator) return;
      const dx = b.pos.x - other.pos.x;
      const dy = b.pos.y - other.pos.y;
      if (dx * dx + dy * dy < R2) cnt++;
    });
    return cnt;
  });
}

function render(): void {
  ctx.fillStyle = 'rgba(13,17,23,0.82)';
  ctx.fillRect(0, 0, W, H);

  const densities = params.colorMode === 'density' ? computeDensities() : boids.map(() => 0);

  boids.forEach((boid, i) => {
    const color = boidColor(boid, densities[i]);
    if (params.trailsOn) drawTrail(boid, color);
    drawBoid(boid, color);
  });
}

function step(): void {
  boids.forEach((b) => applyRules(b));
  boids.forEach((b) => updateBoid(b));
}

function loop(): void {
  const steps = Math.round(params.simSpeed);
  for (let s = 0; s < steps; s++) step();
  render();
  if (running) animId = requestAnimationFrame(loop);
}

// ── UI wiring ────────────────────────────────────────────────────────────────

function getEl<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

function bindRange(id: string, key: keyof Params, labelId?: string): void {
  const el = getEl<HTMLInputElement>(id);
  el.addEventListener('input', () => {
    const val = parseFloat(el.value);
    (params as unknown as Record<string, number | string | boolean>)[key] = val;
    if (labelId) getEl(labelId).textContent = el.value;
  });
}

function bindSelect(id: string, key: keyof Params): void {
  const el = getEl<HTMLSelectElement>(id);
  el.addEventListener('change', () => {
    (params as unknown as Record<string, number | string | boolean>)[key] = el.value;
  });
}

function bindCheck(id: string, key: keyof Params, onChange?: () => void): void {
  const el = getEl<HTMLInputElement>(id);
  el.addEventListener('change', () => {
    (params as unknown as Record<string, number | string | boolean>)[key] = el.checked;
    if (onChange) onChange();
  });
}

function setupUI(): void {
  bindRange('sep-weight', 'separationWeight', 'sep-val');
  bindRange('ali-weight', 'alignmentWeight', 'ali-val');
  bindRange('coh-weight', 'cohesionWeight', 'coh-val');
  bindRange('perception', 'perceptionRadius', 'perception-val');
  bindRange('max-speed', 'maxSpeed', 'max-speed-val');
  bindRange('sim-speed', 'simSpeed', 'sim-speed-val');
  bindSelect('edge-mode', 'edgeMode');
  bindSelect('color-mode', 'colorMode');
  bindCheck('trails-on', 'trailsOn');
  bindCheck('debug-on', 'showDebug');

  const numInput = getEl<HTMLInputElement>('num-boids');
  numInput.addEventListener('change', () => {
    params.numBoids = Math.max(1, Math.min(500, parseInt(numInput.value, 10) || 150));
    numInput.value = String(params.numBoids);
    initBoids();
  });

  const predCheck = getEl<HTMLInputElement>('predator-on');
  predCheck.addEventListener('change', () => {
    params.predatorOn = predCheck.checked;
    initBoids();
  });

  // Play / Pause
  const btnPlay = getEl<HTMLButtonElement>('btn-playpause');
  btnPlay.addEventListener('click', () => {
    running = !running;
    btnPlay.textContent = running ? '⏸ Pause' : '▶ Play';
    if (running) { animId = requestAnimationFrame(loop); }
  });

  // Reset
  getEl('btn-reset').addEventListener('click', () => {
    cancelAnimationFrame(animId);
    initBoids();
    if (running) animId = requestAnimationFrame(loop);
    else render();
  });

  // Mouse
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = W / rect.width;
    const scaleY = H / rect.height;
    mouse = v((e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY);
  });
  canvas.addEventListener('mouseleave', () => { mouse = null; });
  canvas.addEventListener('mousedown', (e) => { mouseRepel = e.button === 2; });
  canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); mouseRepel = true; });
  canvas.addEventListener('mouseup', () => { mouseRepel = false; });
}

// ── Entry point ───────────────────────────────────────────────────────────────

function resize(): void {
  const wrap = getEl<HTMLDivElement>('canvas-wrap');
  W = wrap.clientWidth;
  H = wrap.clientHeight;
  canvas.width = W;
  canvas.height = H;
}

window.addEventListener('DOMContentLoaded', () => {
  canvas = getEl<HTMLCanvasElement>('boids-canvas');
  ctx = canvas.getContext('2d')!;
  resize();
  setupUI();
  initBoids();
  animId = requestAnimationFrame(loop);
});

window.addEventListener('resize', () => {
  resize();
  initBoids();
});
