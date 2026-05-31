import * as d3 from 'd3';

// ── Types ──────────────────────────────────────────────────────────────────────

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  bx: number;  // personal best position
  by: number;
  bf: number;  // personal best fitness
  trail: Array<{x: number; y: number}>;
}

interface PSOConfig {
  fn: string;
  n: number;
  w: number;
  c1: number;
  c2: number;
  neighborhood: 'global' | 'ring';
  maxIter: number;
}

// ── Test functions ─────────────────────────────────────────────────────────────

interface FnDef {
  label: string;
  domain: [number, number];
  fn: (x: number, y: number) => number;
}

const FUNCTIONS: Record<string, FnDef> = {
  sphere: {
    label: 'Sphere',
    domain: [-5.12, 5.12],
    fn: (x, y) => x * x + y * y,
  },
  rastrigin: {
    label: 'Rastrigin',
    domain: [-5.12, 5.12],
    fn: (x, y) =>
      20 + x * x - 10 * Math.cos(2 * Math.PI * x) +
           y * y - 10 * Math.cos(2 * Math.PI * y),
  },
  ackley: {
    label: 'Ackley',
    domain: [-5, 5],
    fn: (x, y) => {
      const a = 20, b = 0.2, c = 2 * Math.PI;
      return -a * Math.exp(-b * Math.sqrt(0.5 * (x*x + y*y)))
             - Math.exp(0.5 * (Math.cos(c*x) + Math.cos(c*y)))
             + a + Math.E;
    },
  },
  rosenbrock: {
    label: 'Rosenbrock',
    domain: [-2, 2],
    fn: (x, y) => 100 * (y - x*x) * (y - x*x) + (1 - x) * (1 - x),
  },
  schwefel: {
    label: 'Schwefel',
    domain: [-500, 500],
    fn: (x, y) =>
      418.9829 * 2 - x * Math.sin(Math.sqrt(Math.abs(x)))
                   - y * Math.sin(Math.sqrt(Math.abs(y))),
  },
  himmelblau: {
    label: 'Himmelblau',
    domain: [-5, 5],
    fn: (x, y) => {
      const a = x*x + y - 11;
      const b = x + y*y - 7;
      return a*a + b*b;
    },
  },
};

// ── PSO state ─────────────────────────────────────────────────────────────────

let particles: Particle[] = [];
let gbx = 0, gby = 0, gbf = Infinity;
let iteration = 0;
let fitnessHistory: number[] = [];
let animId: number | null = null;
let running = false;
let cfg: PSOConfig = {
  fn: 'rastrigin',
  n: 40,
  w: 0.729,
  c1: 1.494,
  c2: 1.494,
  neighborhood: 'global',
  maxIter: 300,
};

const TRAIL_LEN = 12;

// ── Canvas + SVG references ───────────────────────────────────────────────────

let mainCanvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;
let offCanvas: HTMLCanvasElement;    // heatmap cache
let offCtx: CanvasRenderingContext2D;
let CW = 600, CH = 600;

// Fitness curve SVG
let svgFit: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>;
let fitW = 300, fitH = 160;
const fitMargin = {top: 12, right: 16, bottom: 28, left: 42};

// ── Colour scale for heatmap ──────────────────────────────────────────────────

const HM_STEPS = 256;
let hmImageData: ImageData;

// ── Init / Reset ──────────────────────────────────────────────────────────────

function rnd(lo: number, hi: number) {
  return lo + Math.random() * (hi - lo);
}

function initSwarm() {
  const def = FUNCTIONS[cfg.fn];
  const [lo, hi] = def.domain;
  const vmax = (hi - lo) * 0.1;
  particles = [];
  gbf = Infinity;
  iteration = 0;
  fitnessHistory = [];

  for (let i = 0; i < cfg.n; i++) {
    const x = rnd(lo, hi);
    const y = rnd(lo, hi);
    const f = def.fn(x, y);
    const p: Particle = {
      x, y,
      vx: rnd(-vmax, vmax),
      vy: rnd(-vmax, vmax),
      bx: x, by: y, bf: f,
      trail: [],
    };
    particles.push(p);
    if (f < gbf) { gbf = f; gbx = x; gby = y; }
  }
  fitnessHistory.push(gbf);
}

function stepPSO() {
  const def = FUNCTIONS[cfg.fn];
  const [lo, hi] = def.domain;
  const vmax = (hi - lo) * 0.2;

  // For ring topology: find neighbour bests
  function ringBest(i: number): [number, number] {
    const left = (i - 1 + cfg.n) % cfg.n;
    const right = (i + 1) % cfg.n;
    let bx = particles[i].bx, by = particles[i].by, bf = particles[i].bf;
    Array.from([left, right]).forEach(j => {
      if (particles[j].bf < bf) { bf = particles[j].bf; bx = particles[j].bx; by = particles[j].by; }
    });
    return [bx, by];
  }

  particles.forEach((p, i) => {
    const r1 = Math.random(), r2 = Math.random();
    let lbx: number, lby: number;
    if (cfg.neighborhood === 'global') {
      lbx = gbx; lby = gby;
    } else {
      [lbx, lby] = ringBest(i);
    }

    p.vx = cfg.w * p.vx + cfg.c1 * r1 * (p.bx - p.x) + cfg.c2 * r2 * (lbx - p.x);
    p.vy = cfg.w * p.vy + cfg.c1 * r1 * (p.by - p.y) + cfg.c2 * r2 * (lby - p.y);

    // Clamp velocity
    const speed = Math.sqrt(p.vx*p.vx + p.vy*p.vy);
    if (speed > vmax) { p.vx *= vmax/speed; p.vy *= vmax/speed; }

    // Save trail
    p.trail.push({x: p.x, y: p.y});
    if (p.trail.length > TRAIL_LEN) p.trail.shift();

    p.x += p.vx;
    p.y += p.vy;

    // Reflect off walls
    if (p.x < lo) { p.x = lo; p.vx *= -0.5; }
    if (p.x > hi) { p.x = hi; p.vx *= -0.5; }
    if (p.y < lo) { p.y = lo; p.vy *= -0.5; }
    if (p.y > hi) { p.y = hi; p.vy *= -0.5; }

    const f = def.fn(p.x, p.y);
    if (f < p.bf) { p.bf = f; p.bx = p.x; p.by = p.y; }
    if (f < gbf)  { gbf = f; gbx = p.x; gby = p.y; }
  });

  iteration++;
  fitnessHistory.push(gbf);
}

// ── Heatmap (rendered once per function change) ───────────────────────────────

function buildHeatmap() {
  const def = FUNCTIONS[cfg.fn];
  const [lo, hi] = def.domain;
  const res = HM_STEPS;

  // Sample grid
  const vals: number[] = new Array(res * res);
  for (let row = 0; row < res; row++) {
    for (let col = 0; col < res; col++) {
      const x = lo + (col / (res - 1)) * (hi - lo);
      const y = hi - (row / (res - 1)) * (hi - lo); // y axis flipped
      vals[row * res + col] = def.fn(x, y);
    }
  }

  // Log-scale mapping for better contrast
  const minV = Math.min(...vals);
  const maxV = Math.max(...vals);
  const range = maxV - minV || 1;

  const colorScale = d3.scaleSequential(d3.interpolateInferno)
    .domain([0, 1]);

  hmImageData = offCtx.createImageData(res, res);
  for (let i = 0; i < res * res; i++) {
    const t = Math.sqrt((vals[i] - minV) / range); // sqrt for contrast
    const col = colorScale(1 - t);   // invert: minima bright
    const rgb = d3.color(col)!.rgb();
    hmImageData.data[i*4]   = rgb.r;
    hmImageData.data[i*4+1] = rgb.g;
    hmImageData.data[i*4+2] = rgb.b;
    hmImageData.data[i*4+3] = 255;
  }
  // Put at natural res then we draw scaled on main canvas
  offCanvas.width = res;
  offCanvas.height = res;
  offCtx.putImageData(hmImageData, 0, 0);
}

// ── coord helpers ─────────────────────────────────────────────────────────────

function toCanvasX(x: number): number {
  const [lo, hi] = FUNCTIONS[cfg.fn].domain;
  return ((x - lo) / (hi - lo)) * CW;
}
function toCanvasY(y: number): number {
  const [lo, hi] = FUNCTIONS[cfg.fn].domain;
  return CW - ((y - lo) / (hi - lo)) * CH;
}

// ── Draw ──────────────────────────────────────────────────────────────────────

function draw() {
  ctx.clearRect(0, 0, CW, CH);

  // Heatmap background
  ctx.drawImage(offCanvas, 0, 0, CW, CH);

  // Trails
  particles.forEach(p => {
    const n = p.trail.length;
    if (n < 2) return;
    for (let i = 1; i < n; i++) {
      const alpha = (i / n) * 0.4;
      ctx.beginPath();
      ctx.strokeStyle = `rgba(100,220,255,${alpha})`;
      ctx.lineWidth = 1;
      ctx.moveTo(toCanvasX(p.trail[i-1].x), toCanvasY(p.trail[i-1].y));
      ctx.lineTo(toCanvasX(p.trail[i].x), toCanvasY(p.trail[i].y));
      ctx.stroke();
    }
  });

  // Velocity arrows + particles
  particles.forEach(p => {
    const cx = toCanvasX(p.x);
    const cy = toCanvasY(p.y);
    const vscale = 8;
    const ex = cx + p.vx * vscale;
    const ey = cy - p.vy * vscale;  // canvas y flipped

    // Arrow
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(180,230,255,0.5)';
    ctx.lineWidth = 1;
    ctx.moveTo(cx, cy);
    ctx.lineTo(ex, ey);
    ctx.stroke();

    // Particle dot
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#38bdf8';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 0.8;
    ctx.stroke();
  });

  // Global best marker
  if (isFinite(gbf)) {
    const bx = toCanvasX(gbx);
    const by = toCanvasY(gby);
    // Pulsing ring
    const pulse = 0.5 + 0.5 * Math.sin(iteration * 0.3);
    ctx.beginPath();
    ctx.arc(bx, by, 10 + pulse * 4, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(250,204,21,${0.5 + pulse * 0.5})`;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(bx, by, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#fbbf24';
    ctx.fill();
  }

  // Iteration / best text overlay
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(4, 4, 240, 38);
  ctx.fillStyle = '#e0e0f0';
  ctx.font = '12px monospace';
  ctx.fillText(`Iter: ${iteration}   Best: ${gbf.toExponential(4)}`, 10, 18);
  ctx.fillText(`GBest pos: (${gbx.toFixed(3)}, ${gby.toFixed(3)})`, 10, 34);
}

// ── Fitness curve ─────────────────────────────────────────────────────────────

function drawFitnessCurve() {
  const iw = fitW - fitMargin.left - fitMargin.right;
  const ih = fitH - fitMargin.top - fitMargin.bottom;

  svgFit.selectAll('*').remove();

  const g = svgFit.append('g')
    .attr('transform', `translate(${fitMargin.left},${fitMargin.top})`);

  if (fitnessHistory.length < 2) return;

  const xScale = d3.scaleLinear().domain([0, fitnessHistory.length - 1]).range([0, iw]);
  const yMin = Math.min(...fitnessHistory);
  const yMax = Math.max(...fitnessHistory);
  const yScale = d3.scaleLinear().domain([yMin, yMax]).range([ih, 0]).nice();

  // Axes
  g.append('g').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(xScale).ticks(5))
    .selectAll('text').style('fill', '#aaa').style('font-size', '10px');
  g.append('g').call(d3.axisLeft(yScale).ticks(4))
    .selectAll('text').style('fill', '#aaa').style('font-size', '10px');
  g.selectAll('.domain, .tick line').style('stroke', '#555');

  // Area fill
  const area = d3.area<number>()
    .x((_, i) => xScale(i))
    .y0(ih)
    .y1(d => yScale(d));

  g.append('path')
    .datum(fitnessHistory)
    .attr('fill', 'rgba(56,189,248,0.15)')
    .attr('d', area);

  // Line
  const line = d3.line<number>()
    .x((_, i) => xScale(i))
    .y(d => yScale(d));

  g.append('path')
    .datum(fitnessHistory)
    .attr('fill', 'none')
    .attr('stroke', '#38bdf8')
    .attr('stroke-width', 1.5)
    .attr('d', line);

  // Labels
  g.append('text').attr('x', iw / 2).attr('y', ih + 24)
    .attr('text-anchor', 'middle').style('fill', '#888').style('font-size', '10px')
    .text('Iteration');
  g.append('text').attr('transform', 'rotate(-90)')
    .attr('x', -ih/2).attr('y', -32)
    .attr('text-anchor', 'middle').style('fill', '#888').style('font-size', '10px')
    .text('Best Fitness');
}

// ── Animation loop ────────────────────────────────────────────────────────────

let speedMs = 50;
let lastTime = 0;

function loop(ts: number) {
  if (!running) return;
  if (ts - lastTime >= speedMs) {
    lastTime = ts;
    stepPSO();
    draw();
    drawFitnessCurve();
    updateIterLabel();
  }
  if (iteration < cfg.maxIter) {
    animId = requestAnimationFrame(loop);
  } else {
    running = false;
    setPlayBtn(false);
  }
}

function startLoop() {
  running = true;
  setPlayBtn(true);
  animId = requestAnimationFrame(loop);
}

function pauseLoop() {
  running = false;
  setPlayBtn(false);
  if (animId !== null) { cancelAnimationFrame(animId); animId = null; }
}

function resetAll() {
  pauseLoop();
  readConfig();
  initSwarm();
  buildHeatmap();
  draw();
  drawFitnessCurve();
  updateIterLabel();
}

function stepOnce() {
  pauseLoop();
  stepPSO();
  draw();
  drawFitnessCurve();
  updateIterLabel();
}

function setPlayBtn(playing: boolean) {
  const btn = document.getElementById('btn-play') as HTMLButtonElement;
  if (btn) btn.textContent = playing ? '⏸ Pause' : '▶ Play';
}

function updateIterLabel() {
  const el = document.getElementById('iter-label');
  if (el) el.textContent = `Iter ${iteration} / ${cfg.maxIter}`;
}

// ── Read controls ─────────────────────────────────────────────────────────────

function readConfig() {
  cfg.fn = (document.getElementById('sel-fn') as HTMLSelectElement).value;
  cfg.n = parseInt((document.getElementById('inp-n') as HTMLInputElement).value, 10);
  cfg.w = parseFloat((document.getElementById('inp-w') as HTMLInputElement).value);
  cfg.c1 = parseFloat((document.getElementById('inp-c1') as HTMLInputElement).value);
  cfg.c2 = parseFloat((document.getElementById('inp-c2') as HTMLInputElement).value);
  cfg.neighborhood = (document.getElementById('sel-nbr') as HTMLSelectElement).value as 'global'|'ring';
  speedMs = parseInt((document.getElementById('inp-speed') as HTMLInputElement).value, 10);
}

// ── Entrypoint ────────────────────────────────────────────────────────────────

function main() {
  mainCanvas = document.getElementById('main-canvas') as HTMLCanvasElement;
  ctx = mainCanvas.getContext('2d')!;
  CW = mainCanvas.width;
  CH = mainCanvas.height;

  offCanvas = document.createElement('canvas');
  offCtx = offCanvas.getContext('2d')!;

  // Fitness SVG
  const fitContainer = document.getElementById('fit-chart')!;
  fitW = fitContainer.clientWidth || 320;
  fitH = 160;
  svgFit = d3.select<SVGSVGElement, unknown>('#fit-chart')
    .append('svg')
    .attr('width', fitW)
    .attr('height', fitH);

  // Button listeners
  document.getElementById('btn-play')!.addEventListener('click', () => {
    if (running) pauseLoop(); else startLoop();
  });
  document.getElementById('btn-step')!.addEventListener('click', stepOnce);
  document.getElementById('btn-reset')!.addEventListener('click', resetAll);

  // Sync slider labels
  function syncLabel(id: string, labelId: string) {
    const el = document.getElementById(id) as HTMLInputElement;
    const lb = document.getElementById(labelId)!;
    lb.textContent = el.value;
    el.addEventListener('input', () => { lb.textContent = el.value; });
  }
  syncLabel('inp-n', 'lbl-n');
  syncLabel('inp-w', 'lbl-w');
  syncLabel('inp-c1', 'lbl-c1');
  syncLabel('inp-c2', 'lbl-c2');
  syncLabel('inp-speed', 'lbl-speed');

  resetAll();
}

document.addEventListener('DOMContentLoaded', main);
