/**
 * N-body gravity simulation.
 *
 * Simulates point masses under Newtonian gravity with softened forces and a
 * leapfrog (velocity-Verlet) integrator, rendered on a canvas with fading
 * trails. Several presets (random cloud, binary system, solar-system-like,
 * three-body, galaxy disk) illustrate orbital dynamics and the chaotic
 * three-body problem.
 *
 * Pure TypeScript, no external libraries.
 */

interface Body {
  x: number; y: number;
  vx: number; vy: number;
  m: number;
  color: string;
  trail: Array<[number, number]>;
}

let bodies: Body[] = [];
let running = true;
let G = 1.0;
let softening = 4;
let speed = 1;
let showTrails = true;
const MAX_TRAIL = 240;
let W = 760;
let H = 640;

const COLORS = ['#ff4d6d', '#4d9fff', '#ffd24d', '#4dffa0', '#c14dff', '#ff9f4d', '#4dffe0', '#ff4da6'];

function rand(a: number, b: number): number { return a + Math.random() * (b - a); }

function makeBody(x: number, y: number, vx: number, vy: number, m: number, i: number): Body {
  return { x, y, vx, vy, m, color: COLORS[i % COLORS.length], trail: [] };
}

function preset(name: string): void {
  bodies = [];
  const cx = W / 2, cy = H / 2;
  if (name === 'random') {
    for (let i = 0; i < 14; i++) {
      bodies.push(makeBody(rand(80, W - 80), rand(80, H - 80), rand(-0.4, 0.4), rand(-0.4, 0.4), rand(4, 18), i));
    }
  } else if (name === 'binary') {
    bodies.push(makeBody(cx - 90, cy, 0, -1.1, 120, 0));
    bodies.push(makeBody(cx + 90, cy, 0, 1.1, 120, 1));
    for (let i = 0; i < 8; i++) {
      const r = rand(180, 280), a = rand(0, 2 * Math.PI);
      const sp = Math.sqrt((G * 240) / r);
      bodies.push(makeBody(cx + r * Math.cos(a), cy + r * Math.sin(a), -sp * Math.sin(a), sp * Math.cos(a), 2, i + 2));
    }
  } else if (name === 'solar') {
    bodies.push(makeBody(cx, cy, 0, 0, 600, 0));
    for (let i = 0; i < 6; i++) {
      const r = 60 + i * 45;
      const sp = Math.sqrt((G * 600) / r);
      const a = rand(0, 2 * Math.PI);
      bodies.push(makeBody(cx + r * Math.cos(a), cy + r * Math.sin(a), -sp * Math.sin(a), sp * Math.cos(a), rand(3, 12), i + 1));
    }
  } else if (name === 'three') {
    // Figure-eight-ish three-body (sensitive / chaotic).
    bodies.push(makeBody(cx - 100, cy, 0.2, 0.3, 80, 0));
    bodies.push(makeBody(cx + 100, cy, 0.2, -0.3, 80, 1));
    bodies.push(makeBody(cx, cy, -0.4, 0, 80, 2));
  } else if (name === 'galaxy') {
    bodies.push(makeBody(cx, cy, 0, 0, 1200, 0));
    for (let i = 0; i < 60; i++) {
      const r = rand(40, 300), a = rand(0, 2 * Math.PI);
      const sp = Math.sqrt((G * 1200) / r) * rand(0.9, 1.05);
      bodies.push(makeBody(cx + r * Math.cos(a), cy + r * Math.sin(a), -sp * Math.sin(a), sp * Math.cos(a), rand(0.5, 2), i + 1));
    }
  }
}

function accel(): Array<[number, number]> {
  const n = bodies.length;
  const acc: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) acc.push([0, 0]);
  const eps2 = softening * softening;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = bodies[j].x - bodies[i].x;
      const dy = bodies[j].y - bodies[i].y;
      const r2 = dx * dx + dy * dy + eps2;
      const invr = 1 / Math.sqrt(r2);
      const invr3 = invr / r2;
      const f = G * invr3;
      acc[i][0] += f * dx * bodies[j].m;
      acc[i][1] += f * dy * bodies[j].m;
      acc[j][0] -= f * dx * bodies[i].m;
      acc[j][1] -= f * dy * bodies[i].m;
    }
  }
  return acc;
}

function step(dt: number): void {
  const a0 = accel();
  for (let i = 0; i < bodies.length; i++) {
    const b = bodies[i];
    b.x += b.vx * dt + 0.5 * a0[i][0] * dt * dt;
    b.y += b.vy * dt + 0.5 * a0[i][1] * dt * dt;
  }
  const a1 = accel();
  for (let i = 0; i < bodies.length; i++) {
    const b = bodies[i];
    b.vx += 0.5 * (a0[i][0] + a1[i][0]) * dt;
    b.vy += 0.5 * (a0[i][1] + a1[i][1]) * dt;
  }
}

function loop(ctx: CanvasRenderingContext2D): void {
  if (running) {
    const sub = 3;
    for (let s = 0; s < sub; s++) step((0.6 * speed) / sub);
    for (const b of bodies) {
      b.trail.push([b.x, b.y]);
      if (b.trail.length > MAX_TRAIL) b.trail.shift();
    }
  }

  ctx.fillStyle = '#0a0c12';
  ctx.fillRect(0, 0, W, H);

  if (showTrails) {
    for (const b of bodies) {
      if (b.trail.length < 2) continue;
      ctx.beginPath();
      for (let i = 0; i < b.trail.length; i++) {
        const [tx, ty] = b.trail[i];
        if (i === 0) ctx.moveTo(tx, ty); else ctx.lineTo(tx, ty);
      }
      ctx.strokeStyle = b.color + '44';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  for (const b of bodies) {
    const r = Math.max(2, Math.cbrt(b.m) * 1.4);
    ctx.beginPath();
    ctx.arc(b.x, b.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = b.color;
    ctx.fill();
  }

  ctx.fillStyle = '#888';
  ctx.font = '12px monospace';
  ctx.fillText('N-body gravity (velocity-Verlet) — bodies: ' + bodies.length, 12, 20);

  requestAnimationFrame(() => loop(ctx));
}

function init(): void {
  const canvas = document.getElementById('nbody-canvas') as HTMLCanvasElement;
  W = canvas.width; H = canvas.height;
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
  preset('binary');

  const sel = document.getElementById('nb-preset') as HTMLSelectElement;
  if (sel) sel.addEventListener('change', () => preset(sel.value));
  const bind = (id: string, fn: (v: number) => void) => {
    const node = document.getElementById(id) as HTMLInputElement;
    if (node) node.addEventListener('input', () => fn(parseFloat(node.value)));
  };
  bind('nb-g', v => (G = v));
  bind('nb-soft', v => (softening = v));
  bind('nb-speed', v => (speed = v));

  const play = document.getElementById('nb-play');
  if (play) play.addEventListener('click', () => {
    running = !running;
    play.textContent = running ? 'Pause' : 'Play';
  });
  const reset = document.getElementById('nb-reset');
  if (reset) reset.addEventListener('click', () => preset(sel ? sel.value : 'binary'));
  const tr = document.getElementById('nb-trails') as HTMLInputElement;
  if (tr) tr.addEventListener('change', () => (showTrails = tr.checked));

  // Click to add a body.
  canvas.addEventListener('click', e => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (W / rect.width);
    const y = (e.clientY - rect.top) * (H / rect.height);
    bodies.push(makeBody(x, y, 0, 0, 30, bodies.length));
  });

  loop(ctx);
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}

export {};
