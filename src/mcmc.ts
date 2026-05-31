import * as d3 from 'd3';

// ── Types ──────────────────────────────────────────────────────────────────

type Vec2 = [number, number];

interface Sample {
  x: Vec2;
  accepted: boolean;
  proposal?: Vec2;
}

// ── Target densities (unnormalized log-density) ────────────────────────────

function logBanana(x: Vec2): number {
  const a = 1, b = 10;
  const u = x[0] / a;
  const v = x[1] - b * (u * u - 1);
  return -0.5 * (u * u + v * v);
}

function logMixtureGaussians(x: Vec2): number {
  const centers: Vec2[] = [[-2.5, -2.5], [2.5, 2.5], [-2.5, 2.5], [2.5, -2.5]];
  const sig2 = 0.7;
  let sum = 0;
  centers.forEach(c => {
    const dx = x[0] - c[0], dy = x[1] - c[1];
    sum += Math.exp(-0.5 * (dx * dx + dy * dy) / sig2);
  });
  return Math.log(sum + 1e-300);
}

function logCorrelatedGaussian(x: Vec2): number {
  // corr = 0.9
  const sig1 = 1.5, sig2 = 1.5, rho = 0.9;
  const z = (x[0] * x[0]) / (sig1 * sig1)
    - 2 * rho * x[0] * x[1] / (sig1 * sig2)
    + (x[1] * x[1]) / (sig2 * sig2);
  return -z / (2 * (1 - rho * rho));
}

function logRing(x: Vec2): number {
  const r = Math.sqrt(x[0] * x[0] + x[1] * x[1]);
  const dr = r - 3;
  return -0.5 * (dr * dr) / (0.4 * 0.4);
}

const TARGETS: Record<string, { fn: (x: Vec2) => number; label: string; range: number }> = {
  banana: { fn: logBanana, label: 'Banana (Rosenbrock)', range: 5 },
  mixture: { fn: logMixtureGaussians, label: 'Mixture of Gaussians', range: 5 },
  correlated: { fn: logCorrelatedGaussian, label: 'Correlated Gaussian', range: 5 },
  ring: { fn: logRing, label: 'Ring / Donut', range: 5 },
};

// ── RNG ────────────────────────────────────────────────────────────────────

function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ── Samplers ───────────────────────────────────────────────────────────────

function stepMH(current: Vec2, logTarget: (x: Vec2) => number, sigma: number): Sample {
  const proposal: Vec2 = [current[0] + sigma * randn(), current[1] + sigma * randn()];
  const logA = logTarget(proposal) - logTarget(current);
  const accepted = Math.log(Math.random()) < logA;
  return { x: accepted ? proposal : current, accepted, proposal };
}

function stepGibbs(current: Vec2, logTarget: (x: Vec2) => number, sigma: number): Sample {
  // Sequential coordinate-wise MH
  let state: Vec2 = [current[0], current[1]];
  let accepted = false;
  // update x0
  const prop0: Vec2 = [state[0] + sigma * randn(), state[1]];
  if (Math.log(Math.random()) < logTarget(prop0) - logTarget(state)) {
    state = prop0;
    accepted = true;
  }
  // update x1
  const prop1: Vec2 = [state[0], state[1] + sigma * randn()];
  if (Math.log(Math.random()) < logTarget(prop1) - logTarget(state)) {
    state = prop1;
    accepted = true;
  }
  return { x: state, accepted, proposal: state };
}

function stepHMC(current: Vec2, logTarget: (x: Vec2) => number, sigma: number): Sample {
  // Leapfrog HMC-lite: L steps
  const L = 10;
  const eps = sigma * 0.3;
  // gradient via finite differences
  const grad = (x: Vec2): Vec2 => {
    const h = 1e-4;
    const gx = (logTarget([x[0] + h, x[1]]) - logTarget([x[0] - h, x[1]])) / (2 * h);
    const gy = (logTarget([x[0], x[1] + h]) - logTarget([x[0], x[1] - h])) / (2 * h);
    return [gx, gy];
  };

  let q: Vec2 = [current[0], current[1]];
  let p: Vec2 = [randn(), randn()];
  const H0 = -logTarget(q) + 0.5 * (p[0] * p[0] + p[1] * p[1]);

  // half step
  let g = grad(q);
  p = [p[0] + 0.5 * eps * g[0], p[1] + 0.5 * eps * g[1]];

  for (let i = 0; i < L; i++) {
    q = [q[0] + eps * p[0], q[1] + eps * p[1]];
    g = grad(q);
    if (i < L - 1) {
      p = [p[0] + eps * g[0], p[1] + eps * g[1]];
    } else {
      p = [p[0] + 0.5 * eps * g[0], p[1] + 0.5 * eps * g[1]];
    }
  }

  const H1 = -logTarget(q) + 0.5 * (p[0] * p[0] + p[1] * p[1]);
  const accepted = Math.log(Math.random()) < H0 - H1;
  return { x: accepted ? q : current, accepted, proposal: q };
}

const SAMPLERS: Record<string, { fn: (c: Vec2, lt: (x: Vec2) => number, s: number) => Sample; label: string }> = {
  mh: { fn: stepMH, label: 'Metropolis-Hastings' },
  gibbs: { fn: stepGibbs, label: 'Gibbs Sampler' },
  hmc: { fn: stepHMC, label: 'HMC-lite' },
};

// ── State ──────────────────────────────────────────────────────────────────

let targetKey = 'banana';
let samplerKey = 'mh';
let sigma = 0.5;
let speed = 5;
let burnIn = 200;
let showHistogram = true;
let playing = false;
let animId: number | null = null;

let current: Vec2 = [0, 0];
let chainPath: Vec2[] = [];
let allSamples: Vec2[] = [];
let lastSample: Sample | null = null;
let stepCount = 0;
let acceptCount = 0;

const MAX_PATH = 200;
const HIST_BINS = 50;

// ── Layout constants ───────────────────────────────────────────────────────

const DENSITY_SIZE = 480;
const TRACE_W = 480;
const TRACE_H = 100;
const MARGIN = 40;

// ── Canvas refs ────────────────────────────────────────────────────────────

let densityCanvas: HTMLCanvasElement;
let densityCtx: CanvasRenderingContext2D;
let traceXCanvas: HTMLCanvasElement;
let traceXCtx: CanvasRenderingContext2D;
let traceYCanvas: HTMLCanvasElement;
let traceYCtx: CanvasRenderingContext2D;

// ── Heatmap cache ──────────────────────────────────────────────────────────

let heatmapCache: ImageData | null = null;
let heatmapTargetKey = '';

function buildHeatmap(range: number): ImageData {
  const size = DENSITY_SIZE;
  const img = densityCtx.createImageData(size, size);
  const logFn = TARGETS[targetKey].fn;
  let maxVal = -Infinity;
  const vals = new Float32Array(size * size);
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const wx = (px / size) * 2 * range - range;
      const wy = range - (py / size) * 2 * range;
      const v = logFn([wx, wy]);
      vals[py * size + px] = v;
      if (v > maxVal) maxVal = v;
    }
  }
  // normalize and color (dark blue → cyan → white)
  for (let i = 0; i < vals.length; i++) {
    const t = Math.max(0, Math.min(1, (vals[i] - maxVal + 12) / 12));
    const r = Math.round(t * t * 180);
    const g = Math.round(t * 200);
    const b = Math.round(80 + t * 175);
    img.data[i * 4 + 0] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  return img;
}

// ── Histogram ──────────────────────────────────────────────────────────────

function buildHistogram(range: number): Float32Array {
  const bins = HIST_BINS;
  const counts = new Float32Array(bins * bins);
  allSamples.forEach(s => {
    const bx = Math.floor(((s[0] + range) / (2 * range)) * bins);
    const by = Math.floor(((range - s[1]) / (2 * range)) * bins);
    if (bx >= 0 && bx < bins && by >= 0 && by < bins) {
      counts[by * bins + bx]++;
    }
  });
  return counts;
}

// ── World ↔ canvas coord transforms ───────────────────────────────────────

function toCanvas(wx: number, wy: number, range: number): [number, number] {
  const px = ((wx + range) / (2 * range)) * DENSITY_SIZE;
  const py = ((range - wy) / (2 * range)) * DENSITY_SIZE;
  return [px, py];
}

// ── Draw ───────────────────────────────────────────────────────────────────

function drawDensity(): void {
  const range = TARGETS[targetKey].range;
  // heatmap
  if (heatmapTargetKey !== targetKey || !heatmapCache) {
    heatmapCache = buildHeatmap(range);
    heatmapTargetKey = targetKey;
  }
  densityCtx.putImageData(heatmapCache, 0, 0);

  // histogram overlay
  if (showHistogram && allSamples.length > 10) {
    const counts = buildHistogram(range);
    const maxC = Math.max(...Array.from(counts));
    if (maxC > 0) {
      const cellSize = DENSITY_SIZE / HIST_BINS;
      for (let by = 0; by < HIST_BINS; by++) {
        for (let bx = 0; bx < HIST_BINS; bx++) {
          const c = counts[by * HIST_BINS + bx];
          if (c > 0) {
            const alpha = Math.min(1, c / maxC) * 0.5;
            densityCtx.fillStyle = `rgba(255,230,100,${alpha.toFixed(3)})`;
            densityCtx.fillRect(bx * cellSize, by * cellSize, cellSize, cellSize);
          }
        }
      }
    }
  }

  // chain path
  if (chainPath.length > 1) {
    densityCtx.lineWidth = 1;
    for (let i = 1; i < chainPath.length; i++) {
      const age = (i / chainPath.length);
      const alpha = age * 0.8;
      densityCtx.strokeStyle = `rgba(255,100,100,${alpha.toFixed(3)})`;
      densityCtx.beginPath();
      const [x0, y0] = toCanvas(chainPath[i - 1][0], chainPath[i - 1][1], range);
      const [x1, y1] = toCanvas(chainPath[i][0], chainPath[i][1], range);
      densityCtx.moveTo(x0, y0);
      densityCtx.lineTo(x1, y1);
      densityCtx.stroke();
    }
  }

  // last proposal (rejected shown differently)
  if (lastSample && lastSample.proposal) {
    const [cx, cy] = toCanvas(current[0], current[1], range);
    if (!lastSample.accepted && lastSample.proposal) {
      const [px, py] = toCanvas(lastSample.proposal[0], lastSample.proposal[1], range);
      densityCtx.strokeStyle = 'rgba(255,60,60,0.7)';
      densityCtx.lineWidth = 1;
      densityCtx.setLineDash([4, 3]);
      densityCtx.beginPath();
      densityCtx.moveTo(cx, cy);
      densityCtx.lineTo(px, py);
      densityCtx.stroke();
      densityCtx.setLineDash([]);
      // rejected proposal dot
      densityCtx.fillStyle = 'rgba(255,60,60,0.6)';
      densityCtx.beginPath();
      densityCtx.arc(px, py, 4, 0, 2 * Math.PI);
      densityCtx.fill();
    }
    // current position
    densityCtx.fillStyle = '#fff';
    densityCtx.beginPath();
    densityCtx.arc(cx, cy, 5, 0, 2 * Math.PI);
    densityCtx.fill();
    densityCtx.strokeStyle = '#7eb8f7';
    densityCtx.lineWidth = 2;
    densityCtx.stroke();
  }
}

// trace data
const MAX_TRACE = 300;
let traceX: number[] = [];
let traceY: number[] = [];

function drawTrace(
  ctx: CanvasRenderingContext2D,
  data: number[],
  color: string,
  label: string
): void {
  const W = TRACE_W, H = TRACE_H;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, W, H);

  if (data.length < 2) return;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  ctx.strokeStyle = '#333';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  data.forEach((v, i) => {
    const px = (i / (data.length - 1)) * W;
    const py = H - ((v - min) / range) * (H - 10) - 5;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();

  ctx.fillStyle = '#aaa';
  ctx.font = '11px monospace';
  ctx.fillText(label, 6, 14);
}

function drawAll(): void {
  drawDensity();
  drawTrace(traceXCtx, traceX, '#7eb8f7', `x₀  (last: ${current[0].toFixed(2)})`);
  drawTrace(traceYCtx, traceY, '#f7c07e', `x₁  (last: ${current[1].toFixed(2)})`);

  // update acceptance readout
  const rate = stepCount > 0 ? (acceptCount / stepCount * 100).toFixed(1) : '—';
  const el = document.getElementById('acc-rate');
  if (el) el.textContent = `${rate}%`;
  const stepEl = document.getElementById('step-count');
  if (stepEl) stepEl.textContent = `${stepCount}`;
  const burnEl = document.getElementById('burn-status');
  if (burnEl) burnEl.textContent = stepCount < burnIn ? `burn-in (${stepCount}/${burnIn})` : 'sampling';
}

// ── Step logic ─────────────────────────────────────────────────────────────

function doStep(): void {
  const logFn = TARGETS[targetKey].fn;
  const samplerFn = SAMPLERS[samplerKey].fn;
  const s = samplerFn(current, logFn, sigma);
  lastSample = s;
  current = s.x;
  stepCount++;
  if (s.accepted) acceptCount++;

  if (stepCount > burnIn) {
    allSamples.push([current[0], current[1]]);
    if (allSamples.length > 5000) allSamples.splice(0, allSamples.length - 5000);
  }

  chainPath.push([current[0], current[1]]);
  if (chainPath.length > MAX_PATH) chainPath.shift();

  traceX.push(current[0]);
  traceY.push(current[1]);
  if (traceX.length > MAX_TRACE) traceX.shift();
  if (traceY.length > MAX_TRACE) traceY.shift();
}

function doReset(): void {
  current = [0, 0];
  chainPath = [];
  allSamples = [];
  lastSample = null;
  stepCount = 0;
  acceptCount = 0;
  traceX = [];
  traceY = [];
  heatmapCache = null;
  drawAll();
}

// ── Animation loop ─────────────────────────────────────────────────────────

function tick(): void {
  for (let i = 0; i < speed; i++) doStep();
  drawAll();
  if (playing) animId = requestAnimationFrame(tick);
}

function play(): void {
  if (playing) return;
  playing = true;
  animId = requestAnimationFrame(tick);
}

function pause(): void {
  playing = false;
  if (animId !== null) { cancelAnimationFrame(animId); animId = null; }
}

// ── Build UI ───────────────────────────────────────────────────────────────

function buildUI(): void {
  const root = document.getElementById('app')!;
  root.innerHTML = '';

  // title
  const title = document.createElement('h1');
  title.textContent = 'MCMC Sampling Lab';
  root.appendChild(title);

  const sub = document.createElement('div');
  sub.className = 'subtitle';
  sub.textContent = 'Visualizing Markov Chain Monte Carlo over 2D target distributions';
  root.appendChild(sub);

  // main layout
  const layout = document.createElement('div');
  layout.className = 'main-layout';
  root.appendChild(layout);

  // left: density canvas + traces
  const left = document.createElement('div');
  left.className = 'left-col';
  layout.appendChild(left);

  densityCanvas = document.createElement('canvas');
  densityCanvas.width = DENSITY_SIZE;
  densityCanvas.height = DENSITY_SIZE;
  densityCanvas.className = 'density-canvas';
  left.appendChild(densityCanvas);
  densityCtx = densityCanvas.getContext('2d')!;

  const tracesLabel = document.createElement('div');
  tracesLabel.className = 'section-label';
  tracesLabel.textContent = 'Trace Plots';
  left.appendChild(tracesLabel);

  traceXCanvas = document.createElement('canvas');
  traceXCanvas.width = TRACE_W;
  traceXCanvas.height = TRACE_H;
  traceXCanvas.className = 'trace-canvas';
  left.appendChild(traceXCanvas);
  traceXCtx = traceXCanvas.getContext('2d')!;

  traceYCanvas = document.createElement('canvas');
  traceYCanvas.width = TRACE_W;
  traceYCanvas.height = TRACE_H;
  traceYCanvas.className = 'trace-canvas';
  left.appendChild(traceYCanvas);
  traceYCtx = traceYCanvas.getContext('2d')!;

  // right: controls
  const right = document.createElement('div');
  right.className = 'right-col';
  layout.appendChild(right);

  function makeSection(title: string): HTMLDivElement {
    const sec = document.createElement('div');
    sec.className = 'control-section';
    const t = document.createElement('div');
    t.className = 'control-title';
    t.textContent = title;
    sec.appendChild(t);
    right.appendChild(sec);
    return sec;
  }

  function makeLabel(text: string, forId?: string): HTMLLabelElement {
    const l = document.createElement('label');
    l.textContent = text;
    if (forId) l.htmlFor = forId;
    return l;
  }

  function makeSelect(id: string, options: Record<string, string>, val: string, onChange: (v: string) => void): HTMLSelectElement {
    const sel = document.createElement('select');
    sel.id = id;
    Object.keys(options).forEach(k => {
      const opt = document.createElement('option');
      opt.value = k;
      opt.textContent = options[k];
      if (k === val) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener('change', () => onChange(sel.value));
    return sel;
  }

  function makeSlider(
    id: string, min: number, max: number, step: number, value: number,
    onChange: (v: number) => void
  ): { row: HTMLDivElement; valSpan: HTMLSpanElement } {
    const row = document.createElement('div');
    row.className = 'slider-row';
    const inp = document.createElement('input');
    inp.type = 'range';
    inp.id = id;
    inp.min = String(min);
    inp.max = String(max);
    inp.step = String(step);
    inp.value = String(value);
    const valSpan = document.createElement('span');
    valSpan.textContent = String(value);
    inp.addEventListener('input', () => {
      valSpan.textContent = inp.value;
      onChange(parseFloat(inp.value));
    });
    row.appendChild(inp);
    row.appendChild(valSpan);
    return { row, valSpan };
  }

  // target section
  const tSec = makeSection('Target Distribution');
  const targetOpts: Record<string, string> = {};
  Object.keys(TARGETS).forEach(k => { targetOpts[k] = TARGETS[k].label; });
  const targetSel = makeSelect('target-sel', targetOpts, targetKey, v => {
    targetKey = v;
    doReset();
  });
  tSec.appendChild(targetSel);

  // sampler section
  const sSec = makeSection('Sampler');
  const samplerOpts: Record<string, string> = {};
  Object.keys(SAMPLERS).forEach(k => { samplerOpts[k] = SAMPLERS[k].label; });
  const samplerSel = makeSelect('sampler-sel', samplerOpts, samplerKey, v => {
    samplerKey = v;
    doReset();
  });
  sSec.appendChild(samplerSel);

  // parameters section
  const pSec = makeSection('Parameters');

  const sigRow = document.createElement('div');
  sigRow.className = 'param-row';
  sigRow.appendChild(makeLabel('Step size σ', 'sigma-sl'));
  const sigSlider = makeSlider('sigma-sl', 0.05, 3, 0.05, sigma, v => { sigma = v; });
  sigRow.appendChild(sigSlider.row);
  pSec.appendChild(sigRow);

  const speedRow = document.createElement('div');
  speedRow.className = 'param-row';
  speedRow.appendChild(makeLabel('Speed (steps/frame)', 'speed-sl'));
  const speedSlider = makeSlider('speed-sl', 1, 50, 1, speed, v => { speed = Math.round(v); });
  speedRow.appendChild(speedSlider.row);
  pSec.appendChild(speedRow);

  const burnRow = document.createElement('div');
  burnRow.className = 'param-row';
  burnRow.appendChild(makeLabel('Burn-in steps', 'burnin-sl'));
  const burnSlider = makeSlider('burnin-sl', 0, 1000, 50, burnIn, v => { burnIn = Math.round(v); });
  burnRow.appendChild(burnSlider.row);
  pSec.appendChild(burnRow);

  // histogram toggle
  const histRow = document.createElement('div');
  histRow.className = 'param-row toggle-row';
  const histLabel = makeLabel('Show histogram overlay');
  const histCheck = document.createElement('input');
  histCheck.type = 'checkbox';
  histCheck.checked = showHistogram;
  histCheck.addEventListener('change', () => { showHistogram = histCheck.checked; drawAll(); });
  histRow.appendChild(histCheck);
  histRow.appendChild(histLabel);
  pSec.appendChild(histRow);

  // controls section
  const cSec = makeSection('Controls');
  const btnRow = document.createElement('div');
  btnRow.className = 'btn-row';

  const playBtn = document.createElement('button');
  playBtn.textContent = '▶ Play';
  playBtn.className = 'btn btn-primary';
  playBtn.addEventListener('click', () => {
    if (playing) {
      pause();
      playBtn.textContent = '▶ Play';
    } else {
      play();
      playBtn.textContent = '⏸ Pause';
    }
  });

  const stepBtn = document.createElement('button');
  stepBtn.textContent = 'Step';
  stepBtn.className = 'btn';
  stepBtn.addEventListener('click', () => {
    if (playing) { pause(); playBtn.textContent = '▶ Play'; }
    doStep();
    drawAll();
  });

  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset';
  resetBtn.className = 'btn';
  resetBtn.addEventListener('click', () => {
    pause();
    playBtn.textContent = '▶ Play';
    doReset();
  });

  btnRow.appendChild(playBtn);
  btnRow.appendChild(stepBtn);
  btnRow.appendChild(resetBtn);
  cSec.appendChild(btnRow);

  // stats section
  const statSec = makeSection('Statistics');
  const statGrid = document.createElement('div');
  statGrid.className = 'stat-grid';

  function makeStatRow(label: string, id: string, init: string): void {
    const row = document.createElement('div');
    row.className = 'stat-row';
    const lbl = document.createElement('span');
    lbl.textContent = label;
    const val = document.createElement('span');
    val.id = id;
    val.className = 'stat-val';
    val.textContent = init;
    row.appendChild(lbl);
    row.appendChild(val);
    statGrid.appendChild(row);
  }

  makeStatRow('Acceptance Rate:', 'acc-rate', '—');
  makeStatRow('Total Steps:', 'step-count', '0');
  makeStatRow('Status:', 'burn-status', 'burn-in (0/200)');

  statSec.appendChild(statGrid);

  // legend section
  const legSec = makeSection('Legend');
  const legItems = [
    { color: '#7eb8f7', label: 'Current position' },
    { color: 'rgba(255,100,100,0.8)', label: 'Chain path (recent)' },
    { color: 'rgba(255,60,60,0.6)', label: 'Rejected proposal' },
    { color: 'rgba(255,230,100,0.7)', label: 'Sample histogram' },
  ];
  legItems.forEach(item => {
    const row = document.createElement('div');
    row.className = 'leg-row';
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = item.color;
    const lbl = document.createElement('span');
    lbl.textContent = item.label;
    row.appendChild(swatch);
    row.appendChild(lbl);
    legSec.appendChild(row);
  });

  drawAll();
}

// ── CSS injection ──────────────────────────────────────────────────────────

function injectStyles(): void {
  const style = document.createElement('style');
  style.textContent = `
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #0f1117;
      color: #e0e0e0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px 16px 40px;
    }
    h1 {
      font-size: 1.6rem;
      font-weight: 600;
      color: #7eb8f7;
      margin-bottom: 4px;
      letter-spacing: 0.5px;
    }
    .subtitle {
      font-size: 0.85rem;
      color: #888;
      margin-bottom: 20px;
      text-align: center;
    }
    .main-layout {
      display: flex;
      gap: 24px;
      align-items: flex-start;
      flex-wrap: wrap;
      justify-content: center;
      width: 100%;
      max-width: 1100px;
    }
    .left-col {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .density-canvas {
      border: 1px solid #2a2a3a;
      border-radius: 6px;
      display: block;
    }
    .section-label {
      font-size: 0.75rem;
      color: #666;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-top: 4px;
    }
    .trace-canvas {
      border: 1px solid #2a2a3a;
      border-radius: 4px;
      display: block;
    }
    .right-col {
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-width: 240px;
      max-width: 280px;
    }
    .control-section {
      background: #161820;
      border: 1px solid #2a2a3a;
      border-radius: 8px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .control-title {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #7eb8f7;
      margin-bottom: 2px;
    }
    select {
      background: #1e2030;
      color: #e0e0e0;
      border: 1px solid #3a3a5a;
      border-radius: 4px;
      padding: 5px 8px;
      font-size: 0.85rem;
      width: 100%;
      cursor: pointer;
    }
    .param-row {
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .param-row label {
      font-size: 0.8rem;
      color: #aaa;
    }
    .slider-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .slider-row input[type=range] {
      flex: 1;
      accent-color: #7eb8f7;
    }
    .slider-row span {
      font-size: 0.8rem;
      color: #7eb8f7;
      min-width: 32px;
      text-align: right;
      font-family: monospace;
    }
    .toggle-row {
      flex-direction: row;
      align-items: center;
      gap: 8px;
    }
    .toggle-row label { color: #ccc; }
    .btn-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .btn {
      background: #1e2030;
      color: #e0e0e0;
      border: 1px solid #3a3a5a;
      border-radius: 4px;
      padding: 6px 14px;
      font-size: 0.85rem;
      cursor: pointer;
      transition: background 0.15s;
    }
    .btn:hover { background: #2a2d45; }
    .btn-primary {
      background: #1a3a5c;
      border-color: #4a7ab5;
      color: #7eb8f7;
    }
    .btn-primary:hover { background: #224d7a; }
    .stat-grid {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .stat-row {
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      color: #aaa;
    }
    .stat-val {
      color: #7eb8f7;
      font-family: monospace;
    }
    .leg-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.8rem;
      color: #aaa;
    }
    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      flex-shrink: 0;
    }
  `;
  document.head.appendChild(style);
}

// ── Entry point ────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  injectStyles();
  buildUI();
});
