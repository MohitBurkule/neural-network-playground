// mdn.ts — Mixture Density Network visualization lab
// MLP → K gaussian mixture params (pi, mu, sigma) → train via NLL
// Classic inverse-sine dataset shows multi-modal p(y|x)

import * as d3 from 'd3';

export {};

// ─── Seeded RNG ──────────────────────────────────────────────────────────────

let _seed = 42;
function lcg(): number {
  _seed = (_seed * 1664525 + 1013904223) >>> 0;
  return _seed / 0xffffffff;
}
function randNorm(): number {
  const u = lcg() + 1e-10, v = lcg() + 1e-10;
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function resetSeed(): void { _seed = 42; }

// ─── Tensor helpers (flat arrays) ────────────────────────────────────────────

type Vec = Float64Array;
function zeros(n: number): Vec { return new Float64Array(n); }
function randn(n: number, scale: number): Vec {
  const v = new Float64Array(n);
  for (let i = 0; i < n; i++) v[i] = randNorm() * scale;
  return v;
}

// ─── Datasets ────────────────────────────────────────────────────────────────

interface Point { x: number; y: number; }

function makeInverseSine(n: number): Point[] {
  resetSeed();
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const y = lcg() * 2 - 1; // uniform [-1,1]
    const x = y + 0.3 * Math.sin(2 * Math.PI * y) + randNorm() * 0.05;
    pts.push({ x, y });
  }
  return pts;
}

function makeMultiBranch(n: number): Point[] {
  resetSeed();
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const x = lcg() * 2 - 1;
    const branch = Math.floor(lcg() * 3);
    const y = branch === 0 ? 0.7 * x + randNorm() * 0.05
            : branch === 1 ? -0.5 * x + 0.3 + randNorm() * 0.05
            : 0.1 * Math.sin(4 * x) - 0.5 + randNorm() * 0.05;
    pts.push({ x, y });
  }
  return pts;
}

function makeHeteroscedastic(n: number): Point[] {
  resetSeed();
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const x = lcg() * 2 - 1;
    const sigma = 0.05 + 0.4 * Math.abs(x);
    const y = Math.sin(2 * x) + randNorm() * sigma;
    pts.push({ x, y });
  }
  return pts;
}

// ─── MDN model ───────────────────────────────────────────────────────────────

interface Layer { W: Vec; b: Vec; inN: number; outN: number; }

function makeLayer(inN: number, outN: number): Layer {
  return { W: randn(inN * outN, Math.sqrt(2 / inN)), b: zeros(outN), inN, outN };
}

function forwardLinear(layer: Layer, x: Vec): Vec {
  const { W, b, inN, outN } = layer;
  const out = new Float64Array(outN);
  for (let j = 0; j < outN; j++) {
    let s = b[j];
    for (let i = 0; i < inN; i++) s += W[j * inN + i] * x[i];
    out[j] = s;
  }
  return out;
}

function relu(v: Vec): Vec {
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] > 0 ? v[i] : 0;
  return out;
}

function softmax(v: Vec, start: number, k: number): number[] {
  let max = -Infinity;
  for (let i = 0; i < k; i++) if (v[start + i] > max) max = v[start + i];
  let sum = 0;
  const out: number[] = [];
  for (let i = 0; i < k; i++) { const e = Math.exp(v[start + i] - max); sum += e; out.push(e); }
  return out.map(e => e / sum);
}

interface MDNParams { pi: number[]; mu: number[]; sigma: number[]; }

interface MDNModel {
  h1: Layer;
  h2: Layer;
  out: Layer;
  K: number;
}

function makeMDN(hiddenSize: number, K: number): MDNModel {
  resetSeed();
  return {
    h1: makeLayer(1, hiddenSize),
    h2: makeLayer(hiddenSize, hiddenSize),
    out: makeLayer(hiddenSize, 3 * K),
    K,
  };
}

function forwardMDN(model: MDNModel, x: number): MDNParams {
  const { h1, h2, out, K } = model;
  const inp = new Float64Array([x]);
  const a1 = relu(forwardLinear(h1, inp));
  const a2 = relu(forwardLinear(h2, a1));
  const raw = forwardLinear(out, a2);

  const pi = softmax(raw, 0, K);
  const mu: number[] = [];
  const sigma: number[] = [];
  for (let k = 0; k < K; k++) {
    mu.push(raw[K + k]);
    sigma.push(Math.exp(raw[2 * K + k]) + 1e-4);
  }
  return { pi, mu, sigma };
}

function gaussianPDF(y: number, mu: number, sigma: number): number {
  const z = (y - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

function nllLoss(model: MDNModel, pts: Point[]): number {
  let loss = 0;
  for (const p of pts) {
    const { pi, mu, sigma } = forwardMDN(model, p.x);
    let density = 0;
    for (let k = 0; k < model.K; k++) density += pi[k] * gaussianPDF(p.y, mu[k], sigma[k]);
    loss -= Math.log(density + 1e-20);
  }
  return loss / pts.length;
}

// ─── Backprop ─────────────────────────────────────────────────────────────────
// Manual backprop through the MDN

interface Grads { dW: Vec; db: Vec; }

function backwardMDN(
  model: MDNModel,
  pt: Point,
  lr: number,
  m1: MDNModel, v1: MDNModel,
  t: number
): void {
  const { h1, h2, out, K } = model;
  const x = pt.x, y = pt.y;

  // Forward with cache
  const inp = new Float64Array([x]);
  const z1 = forwardLinear(h1, inp);
  const a1 = relu(z1);
  const z2 = forwardLinear(h2, a1);
  const a2 = relu(z2);
  const raw = forwardLinear(out, a2);

  // Softmax for pi
  const pi = softmax(raw, 0, K);
  const mu: number[] = [];
  const sigma: number[] = [];
  const logSigRaw: number[] = [];
  for (let k = 0; k < K; k++) {
    logSigRaw.push(raw[2 * K + k]);
    mu.push(raw[K + k]);
    sigma.push(Math.exp(raw[2 * K + k]) + 1e-4);
  }

  // Density per component
  const phi: number[] = [];
  for (let k = 0; k < K; k++) phi.push(gaussianPDF(y, mu[k], sigma[k]));

  let density = 0;
  for (let k = 0; k < K; k++) density += pi[k] * phi[k];
  density = Math.max(density, 1e-20);

  // Responsibility r_k = pi_k * phi_k / density
  const r: number[] = phi.map((pk, k) => pi[k] * pk / density);

  // dL/d_raw[k] for pi (softmax cross-entropy-like)
  const dRaw = new Float64Array(3 * K);
  for (let k = 0; k < K; k++) {
    // dL/d_logit_k = pi_k - r_k
    dRaw[k] = pi[k] - r[k];
  }
  // dL/d_mu_k = r_k * (mu_k - y) / sigma_k^2
  for (let k = 0; k < K; k++) {
    dRaw[K + k] = r[k] * (mu[k] - y) / (sigma[k] * sigma[k]);
  }
  // dL/d_logSigmaRaw_k = r_k * (1 - (y-mu_k)^2/sigma_k^2) * sigma_k / sigma_k
  // Actually: sigma = exp(s) + eps, d_sigma/d_s = exp(s)
  // dL/d_s = dL/d_sigma * exp(s)
  // dL/d_sigma = r_k * phi_k * [((y-mu)^2/sigma^3) - 1/sigma] / (density * ... actually already in r_k)
  // Cleaner: dL/d_sigma_k = -r_k * [(y-mu_k)^2/sigma_k^3 - 1/sigma_k]
  for (let k = 0; k < K; k++) {
    const dy = y - mu[k];
    const dLdSigma = -r[k] * (dy * dy / (sigma[k] * sigma[k] * sigma[k]) - 1 / sigma[k]);
    const expS = Math.exp(logSigRaw[k]);
    dRaw[2 * K + k] = dLdSigma * expS;
  }

  // Backprop through out layer
  const da2 = new Float64Array(h2.outN);
  for (let j = 0; j < out.outN; j++) {
    for (let i = 0; i < out.inN; i++) {
      da2[i] += out.W[j * out.inN + i] * dRaw[j];
    }
  }
  // Apply ReLU mask for a2
  const dz2 = new Float64Array(h2.outN);
  for (let i = 0; i < h2.outN; i++) dz2[i] = z2[i] > 0 ? da2[i] : 0;

  const da1 = new Float64Array(h1.outN);
  for (let j = 0; j < h2.outN; j++) {
    for (let i = 0; i < h2.inN; i++) {
      da1[i] += h2.W[j * h2.inN + i] * dz2[j];
    }
  }
  const dz1 = new Float64Array(h1.outN);
  for (let i = 0; i < h1.outN; i++) dz1[i] = z1[i] > 0 ? da1[i] : 0;

  // Adam update helper
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  function adamUpdate(layer: Layer, mLayer: Layer, vLayer: Layer, dW_in: Vec, db_in: Vec) {
    for (let i = 0; i < layer.W.length; i++) {
      mLayer.W[i] = beta1 * mLayer.W[i] + (1 - beta1) * dW_in[i];
      vLayer.W[i] = beta2 * vLayer.W[i] + (1 - beta2) * dW_in[i] * dW_in[i];
      const mHat = mLayer.W[i] / (1 - Math.pow(beta1, t));
      const vHat = vLayer.W[i] / (1 - Math.pow(beta2, t));
      layer.W[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
    for (let i = 0; i < layer.b.length; i++) {
      mLayer.b[i] = beta1 * mLayer.b[i] + (1 - beta1) * db_in[i];
      vLayer.b[i] = beta2 * vLayer.b[i] + (1 - beta2) * db_in[i] * db_in[i];
      const mHat = mLayer.b[i] / (1 - Math.pow(beta1, t));
      const vHat = vLayer.b[i] / (1 - Math.pow(beta2, t));
      layer.b[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }

  // Compute dW, db for each layer
  function computeGrads(inAct: Vec, dOut: Vec, layerOut: number, layerIn: number): [Vec, Vec] {
    const dW = new Float64Array(layerOut * layerIn);
    const db = new Float64Array(layerOut);
    for (let j = 0; j < layerOut; j++) {
      db[j] = dOut[j];
      for (let i = 0; i < layerIn; i++) {
        dW[j * layerIn + i] = dOut[j] * inAct[i];
      }
    }
    return [dW, db];
  }

  const [dWout, dbout] = computeGrads(a2, dRaw, out.outN, out.inN);
  adamUpdate(out, m1.out, v1.out, dWout, dbout);

  const [dWh2, dbh2] = computeGrads(a1, dz2, h2.outN, h2.inN);
  adamUpdate(h2, m1.h2, v1.h2, dWh2, dbh2);

  const [dWh1, dbh1] = computeGrads(inp, dz1, h1.outN, h1.inN);
  adamUpdate(h1, m1.h1, v1.h1, dWh1, dbh1);
}

// ─── App state ────────────────────────────────────────────────────────────────

type DatasetName = 'inverse-sine' | 'multi-branch' | 'heteroscedastic';

const N_TRAIN = 300;
let dataset: DatasetName = 'inverse-sine';
let K = 5;
let hiddenSize = 32;
let lr = 0.005;
let running = false;
let showSamples = true;
let epoch = 0;
let stepCount = 0;

let pts: Point[] = [];
let model: MDNModel;
let m1Adam: MDNModel, v1Adam: MDNModel;
let modelK1: MDNModel;
let m1K1: MDNModel, v1K1: MDNModel;
const lossHistory: number[] = [];
const lossK1History: number[] = [];
let animId = 0;

function getDataset(): Point[] {
  if (dataset === 'inverse-sine') return makeInverseSine(N_TRAIN);
  if (dataset === 'multi-branch') return makeMultiBranch(N_TRAIN);
  return makeHeteroscedastic(N_TRAIN);
}

function zeroModel(m: MDNModel): MDNModel {
  function zeroLayer(l: Layer): Layer {
    return { W: zeros(l.W.length), b: zeros(l.b.length), inN: l.inN, outN: l.outN };
  }
  return { h1: zeroLayer(m.h1), h2: zeroLayer(m.h2), out: zeroLayer(m.out), K: m.K };
}

function init() {
  pts = getDataset();
  model = makeMDN(hiddenSize, K);
  m1Adam = zeroModel(model);
  v1Adam = zeroModel(model);
  modelK1 = makeMDN(hiddenSize, 1);
  m1K1 = zeroModel(modelK1);
  v1K1 = zeroModel(modelK1);
  epoch = 0;
  stepCount = 0;
  lossHistory.length = 0;
  lossK1History.length = 0;
}

function trainStep() {
  // Mini-batch SGD — one pass per step
  const shuffled = pts.slice().sort(() => lcg() - 0.5);
  const batchSize = 32;
  for (let b = 0; b < shuffled.length; b += batchSize) {
    const batch = shuffled.slice(b, b + batchSize);
    for (const pt of batch) {
      stepCount++;
      backwardMDN(model, pt, lr, m1Adam, v1Adam, stepCount);
      backwardMDN(modelK1, pt, lr, m1K1, v1K1, stepCount);
    }
  }
  epoch++;
  if (epoch % 5 === 0 || epoch <= 10) {
    lossHistory.push(nllLoss(model, pts));
    lossK1History.push(nllLoss(modelK1, pts));
  }
}

// ─── Canvas rendering ─────────────────────────────────────────────────────────

const W_CANVAS = 500, H_CANVAS = 500;
const GRID = 80; // resolution for density heatmap

// x range [-1.5, 1.5], y range [-1.5, 1.5]
const XMIN = -1.5, XMAX = 1.5, YMIN = -1.5, YMAX = 1.5;

function toCanvasX(x: number) { return (x - XMIN) / (XMAX - XMIN) * W_CANVAS; }
function toCanvasY(y: number) { return H_CANVAS - (y - YMIN) / (YMAX - YMIN) * H_CANVAS; }

let canvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;
let lossCanvas: HTMLCanvasElement;
let lossCtx: CanvasRenderingContext2D;

function renderDensity() {
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, W_CANVAS, H_CANVAS);

  // Compute density grid
  const xs = d3.range(GRID).map(i => XMIN + (i + 0.5) / GRID * (XMAX - XMIN));
  const ys = d3.range(GRID).map(j => YMIN + (j + 0.5) / GRID * (YMAX - YMIN));
  const cellW = W_CANVAS / GRID, cellH = H_CANVAS / GRID;

  let maxD = 0;
  const densities: number[][] = xs.map(x => {
    const { pi, mu, sigma } = forwardMDN(model, x);
    return ys.map(y => {
      let d = 0;
      for (let k = 0; k < K; k++) d += pi[k] * gaussianPDF(y, mu[k], sigma[k]);
      if (d > maxD) maxD = d;
      return d;
    });
  });

  if (maxD < 1e-10) maxD = 1;

  // Draw density heatmap
  for (let i = 0; i < GRID; i++) {
    for (let j = 0; j < GRID; j++) {
      const t = Math.pow(densities[i][j] / maxD, 0.45); // gamma for visibility
      // Purple-teal colormap
      const r = Math.round(20 + t * 120);
      const g = Math.round(20 + t * 160);
      const b = Math.round(40 + t * 200);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      const px = i / GRID * W_CANVAS;
      const py = H_CANVAS - (j + 1) / GRID * H_CANVAS;
      ctx.fillRect(px, py, cellW + 1, cellH + 1);
    }
  }

  // Draw training points
  ctx.globalAlpha = 0.6;
  ctx.fillStyle = '#fcd34d';
  for (const p of pts) {
    const px = toCanvasX(p.x), py = toCanvasY(p.y);
    ctx.beginPath();
    ctx.arc(px, py, 2, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  // Draw component means as lines
  const lineXs = d3.range(120).map(i => XMIN + i / 119 * (XMAX - XMIN));
  const colors = ['#f87171', '#fb923c', '#a3e635', '#34d399', '#60a5fa', '#c084fc', '#f472b6', '#e879f9'];
  for (let k = 0; k < K; k++) {
    ctx.beginPath();
    ctx.strokeStyle = colors[k % colors.length];
    ctx.lineWidth = 1.5;
    let first = true;
    for (const x of lineXs) {
      const { pi, mu } = forwardMDN(model, x);
      const alpha = Math.max(0.1, pi[k]);
      ctx.globalAlpha = alpha;
      const px = toCanvasX(x), py = toCanvasY(mu[k]);
      if (py < 0 || py > H_CANVAS) { first = true; continue; }
      if (first) { ctx.moveTo(px, py); first = false; }
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Draw samples from predictive distribution
  if (showSamples) {
    ctx.fillStyle = '#ffffff';
    ctx.globalAlpha = 0.4;
    for (const p of pts) {
      const { pi, mu, sigma } = forwardMDN(model, p.x);
      // Pick component by pi
      let r = lcg(), cumpi = 0, comp = K - 1;
      for (let k = 0; k < K; k++) { cumpi += pi[k]; if (r < cumpi) { comp = k; break; } }
      const sy = mu[comp] + randNorm() * sigma[comp];
      ctx.beginPath();
      ctx.arc(toCanvasX(p.x), toCanvasY(sy), 1.5, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // Epoch counter
  ctx.fillStyle = '#9ca3af';
  ctx.font = '12px monospace';
  ctx.fillText(`epoch ${epoch}`, 8, 16);
}

function renderLoss() {
  const W = lossCanvas.width, H = lossCanvas.height;
  lossCtx.fillStyle = '#111827';
  lossCtx.fillRect(0, 0, W, H);

  if (lossHistory.length < 2) return;

  const allVals = [...lossHistory, ...lossK1History].filter(isFinite);
  if (allVals.length === 0) return;

  const minV = Math.min(...allVals), maxV = Math.max(...allVals);
  const pad = { t: 16, r: 16, b: 32, l: 48 };
  const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;

  function px(i: number, arr: number[]) { return pad.l + i / (arr.length - 1) * iW; }
  function py(v: number) { return pad.t + (1 - (v - minV) / (maxV - minV + 1e-10)) * iH; }

  function drawLine(arr: number[], color: string) {
    lossCtx.beginPath();
    lossCtx.strokeStyle = color;
    lossCtx.lineWidth = 2;
    arr.forEach((v, i) => {
      if (!isFinite(v)) return;
      if (i === 0) lossCtx.moveTo(px(i, arr), py(v));
      else lossCtx.lineTo(px(i, arr), py(v));
    });
    lossCtx.stroke();
  }

  drawLine(lossK1History, '#f87171'); // K=1 baseline
  drawLine(lossHistory, '#34d399');   // MDN

  // Legend
  lossCtx.font = '10px monospace';
  lossCtx.fillStyle = '#34d399';
  lossCtx.fillText(`MDN K=${K}: ${lossHistory[lossHistory.length - 1].toFixed(3)}`, pad.l, H - 4);
  lossCtx.fillStyle = '#f87171';
  lossCtx.fillText(`K=1: ${lossK1History[lossK1History.length - 1].toFixed(3)}`, W / 2, H - 4);

  // Axes
  lossCtx.strokeStyle = '#374151';
  lossCtx.lineWidth = 1;
  lossCtx.beginPath();
  lossCtx.moveTo(pad.l, pad.t); lossCtx.lineTo(pad.l, pad.t + iH);
  lossCtx.lineTo(pad.l + iW, pad.t + iH);
  lossCtx.stroke();
  lossCtx.fillStyle = '#6b7280';
  lossCtx.fillText('NLL', 4, pad.t + 10);
}

function render() {
  renderDensity();
  renderLoss();
}

// ─── Animation loop ───────────────────────────────────────────────────────────

function loop() {
  if (!running) return;
  trainStep();
  render();
  animId = requestAnimationFrame(loop);
}

// ─── UI wiring ────────────────────────────────────────────────────────────────

function wireControls() {
  const playBtn = document.getElementById('btn-play') as HTMLButtonElement;
  const stepBtn = document.getElementById('btn-step') as HTMLButtonElement;
  const resetBtn = document.getElementById('btn-reset') as HTMLButtonElement;

  playBtn.addEventListener('click', () => {
    running = !running;
    playBtn.textContent = running ? 'Pause' : 'Play';
    if (running) loop();
  });

  stepBtn.addEventListener('click', () => {
    running = false;
    playBtn.textContent = 'Play';
    trainStep();
    render();
  });

  resetBtn.addEventListener('click', () => {
    running = false;
    playBtn.textContent = 'Play';
    cancelAnimationFrame(animId);
    init();
    render();
  });

  const datasetSel = document.getElementById('sel-dataset') as HTMLSelectElement;
  datasetSel.addEventListener('change', () => {
    dataset = datasetSel.value as DatasetName;
    running = false;
    playBtn.textContent = 'Play';
    cancelAnimationFrame(animId);
    init();
    render();
  });

  const kSel = document.getElementById('sel-k') as HTMLSelectElement;
  kSel.addEventListener('change', () => {
    K = parseInt(kSel.value, 10);
    running = false;
    playBtn.textContent = 'Play';
    cancelAnimationFrame(animId);
    init();
    render();
  });

  const lrSel = document.getElementById('sel-lr') as HTMLSelectElement;
  lrSel.addEventListener('change', () => {
    lr = parseFloat(lrSel.value);
  });

  const hidSel = document.getElementById('sel-hidden') as HTMLSelectElement;
  hidSel.addEventListener('change', () => {
    hiddenSize = parseInt(hidSel.value, 10);
    running = false;
    playBtn.textContent = 'Play';
    cancelAnimationFrame(animId);
    init();
    render();
  });

  const sampleChk = document.getElementById('chk-samples') as HTMLInputElement;
  sampleChk.addEventListener('change', () => {
    showSamples = sampleChk.checked;
    render();
  });
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  canvas = document.getElementById('density-canvas') as HTMLCanvasElement;
  ctx = canvas.getContext('2d')!;
  lossCanvas = document.getElementById('loss-canvas') as HTMLCanvasElement;
  lossCtx = lossCanvas.getContext('2d')!;

  init();
  wireControls();
  render();
});
