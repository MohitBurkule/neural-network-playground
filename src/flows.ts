export {};

// ── types ────────────────────────────────────────────────────
type Vec = number[];

// ── math helpers ─────────────────────────────────────────────
function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function logGaussian(x: number, y: number): number {
  return -0.5 * (x * x + y * y) - Math.log(2 * Math.PI);
}

function tanh(x: number): number { return Math.tanh(x); }
function sigmoid(x: number): number { return 1 / (1 + Math.exp(-x)); }
function softplus(x: number): number { return Math.log1p(Math.exp(x)); }

// ── tiny MLP (no loops over arrays, explicit 2->H->2) ────────
interface MLP {
  w1: number[][]; b1: number[];
  w2: number[][]; b2: number[];
  // grads
  gw1: number[][]; gb1: number[];
  gw2: number[][]; gb2: number[];
  // adam
  mw1: number[][]; mb1: number[];
  vw1: number[][]; vb1: number[];
  mw2: number[][]; mb2: number[];
  vw2: number[][]; vb2: number[];
}

function makeMLP(inDim: number, hidden: number, outDim: number): MLP {
  const scale1 = Math.sqrt(2 / inDim);
  const scale2 = Math.sqrt(2 / hidden);
  const zeros2d = (r: number, c: number) =>
    Array.from({ length: r }, () => new Array(c).fill(0));
  const rand2d = (r: number, c: number, s: number) =>
    Array.from({ length: r }, () =>
      Array.from({ length: c }, () => randn() * s));
  return {
    w1: rand2d(hidden, inDim, scale1), b1: new Array(hidden).fill(0),
    w2: rand2d(outDim, hidden, scale2), b2: new Array(outDim).fill(0),
    gw1: zeros2d(hidden, inDim), gb1: new Array(hidden).fill(0),
    gw2: zeros2d(outDim, hidden), gb2: new Array(outDim).fill(0),
    mw1: zeros2d(hidden, inDim), mb1: new Array(hidden).fill(0),
    vw1: zeros2d(hidden, inDim), vb1: new Array(hidden).fill(0),
    mw2: zeros2d(outDim, hidden), mb2: new Array(outDim).fill(0),
    vw2: zeros2d(outDim, hidden), vb2: new Array(outDim).fill(0),
  };
}

interface MLPFwd { h: number[]; out: number[]; }

function mlpForward(m: MLP, inp: number[]): MLPFwd {
  const h: number[] = m.w1.map((row, i) => {
    let s = m.b1[i];
    for (let j = 0; j < inp.length; j++) s += row[j] * inp[j];
    return tanh(s);
  });
  const out: number[] = m.w2.map((row, i) => {
    let s = m.b2[i];
    for (let j = 0; j < h.length; j++) s += row[j] * h[j];
    return s;
  });
  return { h, out };
}

// backward: returns dL/d(inp)
function mlpBackward(m: MLP, inp: number[], fwd: MLPFwd, dout: number[]): number[] {
  const { h } = fwd;
  // dout -> w2 grad
  for (let i = 0; i < m.w2.length; i++) {
    m.gb2[i] += dout[i];
    for (let j = 0; j < h.length; j++) m.gw2[i][j] += dout[i] * h[j];
  }
  // dh
  const dh = new Array(h.length).fill(0);
  for (let i = 0; i < m.w2.length; i++)
    for (let j = 0; j < h.length; j++) dh[j] += dout[i] * m.w2[i][j];
  // tanh back
  const dh2 = dh.map((g, j) => g * (1 - h[j] * h[j]));
  // w1 grad
  for (let i = 0; i < m.w1.length; i++) {
    m.gb1[i] += dh2[i];
    for (let j = 0; j < inp.length; j++) m.gw1[i][j] += dh2[i] * inp[j];
  }
  // dinp
  const dinp = new Array(inp.length).fill(0);
  for (let i = 0; i < m.w1.length; i++)
    for (let j = 0; j < inp.length; j++) dinp[j] += dh2[i] * m.w1[i][j];
  return dinp;
}

function adamUpdate(m: MLP, lr: number, t: number): void {
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  const bc1 = 1 - Math.pow(beta1, t);
  const bc2 = 1 - Math.pow(beta2, t);

  function update1d(
    p: number[], g: number[],
    mv: number[], vv: number[]
  ): void {
    for (let i = 0; i < p.length; i++) {
      mv[i] = beta1 * mv[i] + (1 - beta1) * g[i];
      vv[i] = beta2 * vv[i] + (1 - beta2) * g[i] * g[i];
      p[i] -= lr * (mv[i] / bc1) / (Math.sqrt(vv[i] / bc2) + eps);
      g[i] = 0;
    }
  }
  function update2d(
    p: number[][], g: number[][],
    mv: number[][], vv: number[][]
  ): void {
    for (let i = 0; i < p.length; i++) update1d(p[i], g[i], mv[i], vv[i]);
  }

  update2d(m.w1, m.gw1, m.mw1, m.vw1);
  update1d(m.b1, m.gb1, m.mb1, m.vb1);
  update2d(m.w2, m.gw2, m.mw2, m.vw2);
  update1d(m.b2, m.gb2, m.mb2, m.vb2);
}

// ── Affine Coupling Layer ─────────────────────────────────────
// x = [x0, x1]; mask alternates which half is active
// y0 = x0,   y1 = x1 * exp(s(x0)) + t(x0)
// log|det J| = sum(s)
interface CouplingLayer {
  mlp_s: MLP; // outputs scale (2 -> H -> 1)
  mlp_t: MLP; // outputs translate (2 -> H -> 1)
  mask: number; // 0 = condition on x[0], transform x[1]; 1 = opposite
}

function makeCoupling(hidden: number, mask: number): CouplingLayer {
  return {
    mlp_s: makeMLP(1, hidden, 1),
    mlp_t: makeMLP(1, hidden, 1),
    mask,
  };
}

interface CouplingFwd {
  y: [number, number];
  logdet: number;
  s: number;
  fwdS: MLPFwd; fwdT: MLPFwd;
  cond: number[]; pass: number;
}

function couplingForward(l: CouplingLayer, x: [number, number]): CouplingFwd {
  const c = l.mask === 0 ? x[0] : x[1];   // condition
  const p = l.mask === 0 ? x[1] : x[0];   // passthrough / transform
  const cond = [c];
  const fwdS = mlpForward(l.mlp_s, cond);
  const fwdT = mlpForward(l.mlp_t, cond);
  const s = fwdS.out[0];
  const t = fwdT.out[0];
  const yp = p * Math.exp(s) + t;
  const y: [number, number] = l.mask === 0 ? [c, yp] : [yp, c];
  return { y, logdet: s, s, fwdS, fwdT, cond, pass: p };
}

// backward: given dL/dy return dL/dx and accumulate grads
function couplingBackward(
  l: CouplingLayer,
  fwd: CouplingFwd,
  dy: [number, number],
  dlogdet: number   // dL/d(logdet) from outside
): [number, number] {
  // dy decomposed
  const dyp = l.mask === 0 ? dy[1] : dy[0];
  const dyc = l.mask === 0 ? dy[0] : dy[1];

  // yp = pass * exp(s) + t
  const expS = Math.exp(fwd.s);
  const dp = dyp * expS;

  // grad through s and t
  const ds_from_yp = dyp * fwd.pass * expS; // d(yp)/d(s)
  const dt = dyp;                             // d(yp)/d(t)
  const ds_from_logdet = dlogdet;             // d(logdet)/d(s) = 1

  const ds_total = ds_from_yp + ds_from_logdet;

  // backprop through mlp_s and mlp_t
  const dcond_s = mlpBackward(l.mlp_s, fwd.cond, fwd.fwdS, [ds_total]);
  const dcond_t = mlpBackward(l.mlp_t, fwd.cond, fwd.fwdT, [dt]);
  const dc = dyc + dcond_s[0] + dcond_t[0];

  const dx: [number, number] = l.mask === 0 ? [dc, dp] : [dp, dc];
  return dx;
}

function couplingInverse(l: CouplingLayer, y: [number, number]): [number, number] {
  const c = l.mask === 0 ? y[0] : y[1];
  const yp = l.mask === 0 ? y[1] : y[0];
  const cond = [c];
  const s = mlpForward(l.mlp_s, cond).out[0];
  const t = mlpForward(l.mlp_t, cond).out[0];
  const p = (yp - t) * Math.exp(-s);
  return l.mask === 0 ? [c, p] : [p, c];
}

// ── Flow model ────────────────────────────────────────────────
interface FlowModel {
  layers: CouplingLayer[];
  adamT: number;
}

function makeFlow(nLayers: number, hidden: number): FlowModel {
  const layers: CouplingLayer[] = [];
  for (let i = 0; i < nLayers; i++) layers.push(makeCoupling(hidden, i % 2));
  return { layers, adamT: 0 };
}

// forward: x (data space) -> z (latent), return logdet
interface FlowFwd { z: [number, number]; logdet: number; fwds: CouplingFwd[]; }

function flowForward(model: FlowModel, x: [number, number]): FlowFwd {
  let cur: [number, number] = [x[0], x[1]];
  let logdet = 0;
  const fwds: CouplingFwd[] = [];
  for (const l of model.layers) {
    const f = couplingForward(l, cur);
    cur = f.y;
    logdet += f.logdet;
    fwds.push(f);
  }
  return { z: cur, logdet, fwds };
}

// inverse: z -> x (sample)
function flowInverse(model: FlowModel, z: [number, number]): [number, number] {
  let cur: [number, number] = [z[0], z[1]];
  for (let i = model.layers.length - 1; i >= 0; i--) {
    cur = couplingInverse(model.layers[i], cur);
  }
  return cur;
}

// NLL loss for one sample, + backward pass
function flowNLLAndGrad(model: FlowModel, x: [number, number]): number {
  const fwd = flowForward(model, x);
  const { z, logdet, fwds } = fwd;
  const logpz = logGaussian(z[0], z[1]);
  const logpx = logpz + logdet;
  // loss = -logpx; dL/d(logpx) = -1
  // dL/d(z0) = -dlogpz/dz0 = z0,  same for z1
  let dz: [number, number] = [z[0], z[1]];  // grad of -logpz wrt z
  let dlogdet = -1;

  // backprop through layers in reverse
  for (let i = model.layers.length - 1; i >= 0; i--) {
    const dx = couplingBackward(model.layers[i], fwds[i], dz, dlogdet);
    dlogdet = 0; // logdet grad only needed once for outermost
    dz = dx;
  }
  return -logpx;
}

function flowStep(model: FlowModel, batch: [number, number][], lr: number): number {
  model.adamT++;
  let totalLoss = 0;
  for (const x of batch) totalLoss += flowNLLAndGrad(model, x);
  totalLoss /= batch.length;
  // scale grads by 1/batch
  const scale = 1 / batch.length;
  for (const l of model.layers) {
    scaleGrads(l.mlp_s, scale);
    scaleGrads(l.mlp_t, scale);
    adamUpdate(l.mlp_s, lr, model.adamT);
    adamUpdate(l.mlp_t, lr, model.adamT);
  }
  return totalLoss;
}

function scaleGrads(m: MLP, s: number): void {
  for (let i = 0; i < m.gw1.length; i++)
    for (let j = 0; j < m.gw1[i].length; j++) m.gw1[i][j] *= s;
  for (let i = 0; i < m.gb1.length; i++) m.gb1[i] *= s;
  for (let i = 0; i < m.gw2.length; i++)
    for (let j = 0; j < m.gw2[i].length; j++) m.gw2[i][j] *= s;
  for (let i = 0; i < m.gb2.length; i++) m.gb2[i] *= s;
}

// ── Target distributions ──────────────────────────────────────
type DistName = 'two-moons' | 'circles' | 'spiral' | '8gaussians' | 'checkerboard';

function sampleTarget(name: DistName): [number, number] {
  switch (name) {
    case 'two-moons': {
      const upper = Math.random() < 0.5;
      const angle = Math.random() * Math.PI;
      const r = 1.0 + randn() * 0.1;
      if (upper) return [r * Math.cos(angle) - 0.5, r * Math.sin(angle) - 0.25];
      return [r * Math.cos(angle + Math.PI) + 0.5, r * Math.sin(angle + Math.PI) + 0.25];
    }
    case 'circles': {
      const idx = Math.random() < 0.5 ? 0 : 1;
      const radii = [0.5, 1.2];
      const a = Math.random() * 2 * Math.PI;
      const r = radii[idx] + randn() * 0.08;
      return [r * Math.cos(a), r * Math.sin(a)];
    }
    case 'spiral': {
      const t = Math.random() * 3 * Math.PI;
      const arm = Math.random() < 0.5 ? 1 : -1;
      const r = t / (3 * Math.PI);
      return [
        arm * (r * Math.cos(t) + randn() * 0.07),
        arm * (r * Math.sin(t) + randn() * 0.07),
      ];
    }
    case '8gaussians': {
      const centers: [number, number][] = [
        [1.5, 0], [-1.5, 0], [0, 1.5], [0, -1.5],
        [1.06, 1.06], [-1.06, 1.06], [1.06, -1.06], [-1.06, -1.06],
      ];
      const c = centers[Math.floor(Math.random() * 8)];
      return [c[0] + randn() * 0.2, c[1] + randn() * 0.2];
    }
    case 'checkerboard': {
      // rejection sample from 3x3 checkerboard
      for (;;) {
        const x = (Math.random() * 2 - 1) * 2;
        const y = (Math.random() * 2 - 1) * 2;
        const ix = Math.floor((x + 2) * 1.5);
        const iy = Math.floor((y + 2) * 1.5);
        if ((ix + iy) % 2 === 0) return [x, y];
      }
    }
  }
}

function sampleBatch(name: DistName, n: number): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i < n; i++) out.push(sampleTarget(name));
  return out;
}

// ── Global state ──────────────────────────────────────────────
let model: FlowModel = makeFlow(4, 32);
let dist: DistName = 'two-moons';
let lr = 0.003;
let playing = false;
let stepCount = 0;
let lossHistory: number[] = [];
let showGrid = true;
let rafId = 0;

// cached target samples for display
let targetSamples: [number, number][] = sampleBatch(dist, 500);

// ── DOM ───────────────────────────────────────────────────────
const canvDensity = document.getElementById('canv-density') as HTMLCanvasElement;
const canvWarp   = document.getElementById('canv-warp')    as HTMLCanvasElement;
const svgLoss    = document.getElementById('svg-loss')     as unknown as SVGSVGElement;

const btnPlay    = document.getElementById('btn-play')!;
const btnPause   = document.getElementById('btn-pause')!;
const btnStep    = document.getElementById('btn-step')!;
const btnReset   = document.getElementById('btn-reset')!;
const btnSample  = document.getElementById('btn-sample')!;

const selDist    = document.getElementById('sel-dist')    as HTMLSelectElement;
const selLayers  = document.getElementById('sel-layers')  as HTMLSelectElement;
const selHidden  = document.getElementById('sel-hidden')  as HTMLSelectElement;
const ctrlLr     = document.getElementById('ctrl-lr')     as HTMLInputElement;
const lblLr      = document.getElementById('lbl-lr')!;
const togGrid    = document.getElementById('tog-grid')    as HTMLInputElement;

const statStep   = document.getElementById('stat-step')!;
const statLoss   = document.getElementById('stat-loss')!;

// ── d3 loss curve ─────────────────────────────────────────────
import * as d3 from 'd3';

const lossW = 700, lossH = 130;
const lossMargin = { top: 8, right: 16, bottom: 28, left: 48 };

const svgL = d3.select('#svg-loss')
  .attr('viewBox', `0 0 ${lossW} ${lossH}`)
  .attr('preserveAspectRatio', 'xMidYMid meet');

const lxScale = d3.scaleLinear().range([lossMargin.left, lossW - lossMargin.right]);
const lyScale = d3.scaleLinear().range([lossH - lossMargin.bottom, lossMargin.top]);

const gLossX = svgL.append('g')
  .attr('transform', `translate(0,${lossH - lossMargin.bottom})`);
const gLossY = svgL.append('g')
  .attr('transform', `translate(${lossMargin.left},0)`);

const lossPath = svgL.append('path')
  .attr('fill', 'none')
  .attr('stroke', '#a78bfa')
  .attr('stroke-width', 1.5);

function updateLossCurve(): void {
  const data = lossHistory.slice(-300);
  if (data.length < 2) return;
  lxScale.domain([0, data.length - 1]);
  const mn = d3.min(data) as number;
  const mx = d3.max(data) as number;
  lyScale.domain([Math.min(mn * 0.98, mn - 0.1), mx * 1.02]);

  gLossX.call(d3.axisBottom(lxScale).ticks(5) as any);
  gLossY.call(d3.axisLeft(lyScale).ticks(4) as any);

  const line = d3.line<number>()
    .x((_, i) => lxScale(i))
    .y(d => lyScale(d));
  lossPath.attr('d', line(data) as string);
}

// ── density heatmap ───────────────────────────────────────────
const GRID_SIZE = 60;

function evalDensityGrid(): Float32Array {
  const W = GRID_SIZE, H = GRID_SIZE;
  const buf = new Float32Array(W * H);
  const ext = 3;
  for (let iy = 0; iy < H; iy++) {
    for (let ix = 0; ix < W; ix++) {
      const x = (ix / (W - 1)) * 2 * ext - ext;
      const y = ((H - 1 - iy) / (H - 1)) * 2 * ext - ext;
      const fwd = flowForward(model, [x, y]);
      const lp = logGaussian(fwd.z[0], fwd.z[1]) + fwd.logdet;
      buf[iy * W + ix] = lp;
    }
  }
  return buf;
}

function drawDensity(): void {
  const ctx = canvDensity.getContext('2d')!;
  const W = canvDensity.width, H = canvDensity.height;
  const buf = evalDensityGrid();

  // map log-density to color
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < buf.length; i++) {
    if (isFinite(buf[i])) { mn = Math.min(mn, buf[i]); mx = Math.max(mx, buf[i]); }
  }

  const img = ctx.createImageData(GRID_SIZE, GRID_SIZE);
  for (let i = 0; i < buf.length; i++) {
    const t = isFinite(buf[i]) ? Math.max(0, Math.min(1, (buf[i] - mn) / (mx - mn + 1e-9))) : 0;
    // purple -> yellow color map
    const r = Math.round(t < 0.5 ? t * 2 * 100 : 100 + (t - 0.5) * 2 * 155);
    const g = Math.round(t * 200);
    const b = Math.round(t < 0.5 ? 80 + t * 2 * 120 : 200 - (t - 0.5) * 2 * 180);
    img.data[i * 4 + 0] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 200;
  }
  // draw scaled to canvas
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = GRID_SIZE; tmpCanvas.height = GRID_SIZE;
  tmpCanvas.getContext('2d')!.putImageData(img, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.drawImage(tmpCanvas, 0, 0, W, H);

  // overlay flow samples
  const nSamples = 300;
  ctx.fillStyle = 'rgba(251,191,36,0.7)';
  const ext = 3;
  for (let i = 0; i < nSamples; i++) {
    const z: [number, number] = [randn(), randn()];
    const x = flowInverse(model, z);
    const px = ((x[0] + ext) / (2 * ext)) * W;
    const py = (1 - (x[1] + ext) / (2 * ext)) * H;
    if (px >= 0 && px < W && py >= 0 && py < H) {
      ctx.beginPath();
      ctx.arc(px, py, 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  // overlay target samples (blue)
  ctx.fillStyle = 'rgba(99,179,237,0.6)';
  for (const [tx, ty] of targetSamples) {
    const px = ((tx + ext) / (2 * ext)) * W;
    const py = (1 - (ty + ext) / (2 * ext)) * H;
    if (px >= 0 && px < W && py >= 0 && py < H) {
      ctx.beginPath();
      ctx.arc(px, py, 1.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

// ── grid warp ─────────────────────────────────────────────────
function drawWarp(): void {
  const ctx = canvWarp.getContext('2d')!;
  const W = canvWarp.width, H = canvWarp.height;
  ctx.clearRect(0, 0, W, H);

  const N = 20;
  const ext = 2.5;
  const toCanv = (v: number, dim: 'x' | 'y') => {
    return dim === 'x'
      ? ((v + ext) / (2 * ext)) * W
      : (1 - (v + ext) / (2 * ext)) * H;
  };

  // build grid of transformed points
  const pts: [number, number][][] = [];
  for (let iy = 0; iy <= N; iy++) {
    const row: [number, number][] = [];
    for (let ix = 0; ix <= N; ix++) {
      const zx = (ix / N) * 2 * ext - ext;
      const zy = (iy / N) * 2 * ext - ext;
      const mapped = flowInverse(model, [zx, zy]);
      row.push([toCanv(mapped[0], 'x'), toCanv(mapped[1], 'y')]);
    }
    pts.push(row);
  }

  ctx.strokeStyle = 'rgba(139,92,246,0.5)';
  ctx.lineWidth = 0.8;

  // horizontal lines
  for (let iy = 0; iy <= N; iy++) {
    ctx.beginPath();
    for (let ix = 0; ix <= N; ix++) {
      const [px, py] = pts[iy][ix];
      if (ix === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  // vertical lines
  for (let ix = 0; ix <= N; ix++) {
    ctx.beginPath();
    for (let iy = 0; iy <= N; iy++) {
      const [px, py] = pts[iy][ix];
      if (iy === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
}

// ── render all ────────────────────────────────────────────────
function render(): void {
  drawDensity();
  if (showGrid) drawWarp();
  else {
    const ctx = canvWarp.getContext('2d')!;
    ctx.clearRect(0, 0, canvWarp.width, canvWarp.height);
  }
  updateLossCurve();
  statStep.textContent = String(stepCount);
  if (lossHistory.length > 0) {
    statLoss.textContent = lossHistory[lossHistory.length - 1].toFixed(3);
  }
}

// ── training loop ─────────────────────────────────────────────
const BATCH = 128;
const STEPS_PER_FRAME = 5;

function trainStep(): void {
  for (let k = 0; k < STEPS_PER_FRAME; k++) {
    const batch = sampleBatch(dist, BATCH);
    const loss = flowStep(model, batch, lr);
    lossHistory.push(loss);
    stepCount++;
  }
}

function loop(): void {
  if (!playing) return;
  trainStep();
  render();
  rafId = requestAnimationFrame(loop);
}

// ── controls wiring ───────────────────────────────────────────
function reset(): void {
  cancelAnimationFrame(rafId);
  playing = false;
  stepCount = 0;
  lossHistory = [];
  const nLayers = parseInt(selLayers.value, 10);
  const hidden  = parseInt(selHidden.value, 10);
  model = makeFlow(nLayers, hidden);
  targetSamples = sampleBatch(dist, 500);
  render();
}

btnPlay.addEventListener('click', () => {
  playing = true;
  loop();
});
btnPause.addEventListener('click', () => {
  playing = false;
  cancelAnimationFrame(rafId);
});
btnStep.addEventListener('click', () => {
  playing = false;
  cancelAnimationFrame(rafId);
  trainStep();
  render();
});
btnReset.addEventListener('click', () => reset());
btnSample.addEventListener('click', () => {
  targetSamples = sampleBatch(dist, 500);
  render();
});

selDist.addEventListener('change', () => {
  dist = selDist.value as DistName;
  targetSamples = sampleBatch(dist, 500);
  render();
});

ctrlLr.addEventListener('input', () => {
  lr = parseFloat(ctrlLr.value);
  lblLr.textContent = lr.toFixed(4);
});

togGrid.addEventListener('change', () => {
  showGrid = togGrid.checked;
  render();
});

selLayers.addEventListener('change', () => reset());
selHidden.addEventListener('change', () => reset());

// ── style d3 axes ─────────────────────────────────────────────
d3.selectAll('.tick text').style('fill', '#6b7280');
d3.selectAll('.tick line, .domain').style('stroke', '#374151');

// ── init ──────────────────────────────────────────────────────
render();
