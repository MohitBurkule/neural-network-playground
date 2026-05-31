import * as d3 from 'd3';

// ---------------------------------------------------------------------------
// Tiny neural-net primitives (no ML library)
// ---------------------------------------------------------------------------

type Matrix = number[][];

function zeros(rows: number, cols: number): Matrix {
  return Array.from({length: rows}, () => new Array(cols).fill(0));
}

function randn(): number {
  // Box-Muller
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

interface Layer {
  W: Matrix;   // [out x in]
  b: number[]; // [out]
  dW: Matrix;
  db: number[];
}

function makeLayer(inDim: number, outDim: number): Layer {
  const scale = Math.sqrt(2 / inDim);
  const W = Array.from({length: outDim}, () =>
    Array.from({length: inDim}, () => randn() * scale)
  );
  const b = new Array(outDim).fill(0);
  return {W, b, dW: zeros(outDim, inDim), db: new Array(outDim).fill(0)};
}

function linearForward(layer: Layer, x: number[]): number[] {
  return layer.W.map((row, i) => row.reduce((s, w, j) => s + w * x[j], 0) + layer.b[i]);
}

function relu(x: number[]): number[] {
  return x.map(v => Math.max(0, v));
}

function reluGrad(x: number[]): number[] {
  return x.map(v => (v > 0 ? 1 : 0));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, x))));
}

// ---------------------------------------------------------------------------
// MLP: forward + backward (single sample, accumulates grads)
// ---------------------------------------------------------------------------

interface MLP {
  layers: Layer[];
  hiddenSize: number;
  inDim: number;
  outDim: number;
}

function makeMLP(inDim: number, hiddenSize: number, outDim: number, numHidden: number): MLP {
  const layers: Layer[] = [];
  let d = inDim;
  for (let i = 0; i < numHidden; i++) {
    layers.push(makeLayer(d, hiddenSize));
    d = hiddenSize;
  }
  layers.push(makeLayer(d, outDim));
  return {layers, hiddenSize, inDim, outDim};
}

function resetGrads(mlp: MLP): void {
  for (const l of mlp.layers) {
    l.dW = zeros(l.W.length, l.W[0].length);
    l.db = new Array(l.b.length).fill(0);
  }
}

interface ForwardResult {
  preacts: number[][];  // pre-activation at each layer
  acts: number[][];     // post-activation (after relu, except last)
}

function mlpForward(mlp: MLP, input: number[]): ForwardResult {
  const preacts: number[][] = [];
  const acts: number[][] = [input];
  let x = input;
  for (let i = 0; i < mlp.layers.length; i++) {
    const pre = linearForward(mlp.layers[i], x);
    preacts.push(pre);
    if (i < mlp.layers.length - 1) {
      x = relu(pre);
    } else {
      x = pre; // raw logit for last layer
    }
    acts.push(x);
  }
  return {preacts, acts};
}

// Returns gradient w.r.t. input
function mlpBackward(mlp: MLP, fwd: ForwardResult, dLoss_dOut: number[]): number[] {
  let delta = dLoss_dOut;
  for (let i = mlp.layers.length - 1; i >= 0; i--) {
    const layer = mlp.layers[i];
    const xIn = fwd.acts[i]; // input to this layer
    // accumulate grads
    for (let o = 0; o < layer.W.length; o++) {
      for (let j = 0; j < layer.W[0].length; j++) {
        layer.dW[o][j] += delta[o] * xIn[j];
      }
      layer.db[o] += delta[o];
    }
    // propagate through linear
    const dX = new Array(xIn.length).fill(0);
    for (let j = 0; j < xIn.length; j++) {
      for (let o = 0; o < layer.W.length; o++) {
        dX[j] += layer.W[o][j] * delta[o];
      }
    }
    // propagate through relu (skip for input layer activation)
    if (i > 0) {
      const preact = fwd.preacts[i - 1];
      const rg = reluGrad(preact);
      delta = dX.map((v, j) => v * rg[j]);
    } else {
      delta = dX;
    }
  }
  return delta;
}

function sgdStep(mlp: MLP, lr: number): void {
  for (const layer of mlp.layers) {
    for (let o = 0; o < layer.W.length; o++) {
      for (let j = 0; j < layer.W[0].length; j++) {
        layer.W[o][j] -= lr * layer.dW[o][j];
      }
      layer.b[o] -= lr * layer.db[o];
    }
  }
}

// ---------------------------------------------------------------------------
// Distributions
// ---------------------------------------------------------------------------

type DistName = 'ring' | 'two-moons' | '8-gaussians' | 'spiral';

function sampleReal(dist: DistName): [number, number] {
  switch (dist) {
    case 'ring': {
      const angle = Math.random() * 2 * Math.PI;
      const r = 0.7 + randn() * 0.05;
      return [r * Math.cos(angle), r * Math.sin(angle)];
    }
    case 'two-moons': {
      const top = Math.random() > 0.5;
      const t = Math.random() * Math.PI;
      const x = (top ? 1 : 0) + Math.cos(t) * 0.5 + randn() * 0.05;
      const y = (top ? 0.3 : -0.3) + Math.sin(t) * 0.5 * (top ? 1 : -1) + randn() * 0.05;
      return [x - 0.5, y];
    }
    case '8-gaussians': {
      const centers: [number, number][] = [
        [0.7, 0], [-0.7, 0], [0, 0.7], [0, -0.7],
        [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]
      ];
      const c = centers[Math.floor(Math.random() * 8)];
      return [c[0] + randn() * 0.05, c[1] + randn() * 0.05];
    }
    case 'spiral': {
      const branch = Math.random() > 0.5 ? 0 : 1;
      const t = Math.random() * 3 * Math.PI;
      const r = t / (3 * Math.PI) * 0.85;
      const angle = t + branch * Math.PI;
      return [r * Math.cos(angle) + randn() * 0.03, r * Math.sin(angle) + randn() * 0.03];
    }
  }
}

// ---------------------------------------------------------------------------
// GAN state
// ---------------------------------------------------------------------------

interface GANConfig {
  latentDim: number;
  hiddenSize: number;
  dStepsPerG: number;
  lr: number;
  batchSize: number;
  dist: DistName;
}

interface GANState {
  G: MLP;
  D: MLP;
  config: GANConfig;
  step: number;
  lossD: number;
  lossG: number;
  lossDHistory: number[];
  lossGHistory: number[];
}

function makeGAN(cfg: GANConfig): GANState {
  const G = makeMLP(cfg.latentDim, cfg.hiddenSize, 2, 2);
  const D = makeMLP(2, cfg.hiddenSize, 1, 2);
  return {G, D, config: cfg, step: 0, lossD: Math.log(2), lossG: Math.log(2), lossDHistory: [], lossGHistory: []};
}

function sampleLatent(latentDim: number): number[] {
  return Array.from({length: latentDim}, () => randn());
}

function generate(G: MLP, z: number[]): [number, number] {
  const fwd = mlpForward(G, z);
  const out = fwd.acts[fwd.acts.length - 1];
  return [out[0], out[1]];
}

function discriminate(D: MLP, pt: [number, number]): number {
  const fwd = mlpForward(D, [pt[0], pt[1]]);
  const logit = fwd.acts[fwd.acts.length - 1][0];
  return sigmoid(logit);
}

// bce loss + grad w.r.t. logit
function bceLossAndGrad(logit: number, label: number): [number, number] {
  const p = sigmoid(logit);
  const loss = -(label * Math.log(p + 1e-8) + (1 - label) * Math.log(1 - p + 1e-8));
  const grad = p - label; // d(BCE)/d(logit)
  return [loss, grad];
}

function ganTrainStep(state: GANState): void {
  const {G, D, config} = state;
  const {latentDim, lr, batchSize, dist, dStepsPerG} = config;

  // --- Train D ---
  let totalLossD = 0;
  for (let ds = 0; ds < dStepsPerG; ds++) {
    resetGrads(D);
    let ld = 0;
    for (let i = 0; i < batchSize; i++) {
      // real sample
      const real = sampleReal(dist);
      const fwdReal = mlpForward(D, [real[0], real[1]]);
      const logitReal = fwdReal.acts[fwdReal.acts.length - 1][0];
      const [lossReal, gradReal] = bceLossAndGrad(logitReal, 1);
      mlpBackward(D, fwdReal, [gradReal / batchSize]);
      ld += lossReal;

      // fake sample
      const z = sampleLatent(latentDim);
      const fwdG = mlpForward(G, z);
      const fake: [number, number] = [fwdG.acts[fwdG.acts.length - 1][0], fwdG.acts[fwdG.acts.length - 1][1]];
      const fwdFake = mlpForward(D, [fake[0], fake[1]]);
      const logitFake = fwdFake.acts[fwdFake.acts.length - 1][0];
      const [lossFake, gradFake] = bceLossAndGrad(logitFake, 0);
      mlpBackward(D, fwdFake, [gradFake / batchSize]);
      ld += lossFake;
    }
    sgdStep(D, lr);
    totalLossD = ld / (2 * batchSize);
  }

  // --- Train G ---
  resetGrads(G);
  let lg = 0;
  for (let i = 0; i < batchSize; i++) {
    const z = sampleLatent(latentDim);
    const fwdG = mlpForward(G, z);
    const fake: [number, number] = [fwdG.acts[fwdG.acts.length - 1][0], fwdG.acts[fwdG.acts.length - 1][1]];

    const fwdD = mlpForward(D, [fake[0], fake[1]]);
    const logitD = fwdD.acts[fwdD.acts.length - 1][0];
    // G wants D to output 1 (real)
    const [lossG, gradD_logit] = bceLossAndGrad(logitD, 1);
    lg += lossG;

    // backprop through D to get grad w.r.t. fake point
    const dFake = mlpBackward(D, fwdD, [gradD_logit / batchSize]);
    // backprop through G
    mlpBackward(G, fwdG, [dFake[0] / batchSize, dFake[1] / batchSize]);
  }
  sgdStep(G, lr);

  state.step++;
  state.lossD = totalLossD;
  state.lossG = lg / batchSize;
  state.lossDHistory.push(state.lossD);
  state.lossGHistory.push(state.lossG);
  if (state.lossDHistory.length > 300) {
    state.lossDHistory.shift();
    state.lossGHistory.shift();
  }
}

// ---------------------------------------------------------------------------
// Visualization
// ---------------------------------------------------------------------------

const PLOT_SIZE = 380;
const HEATMAP_RES = 40;
const VIEW_RANGE = 1.2;

// Map from data coords to canvas coords
function toCanvas(v: number): number {
  return ((v + VIEW_RANGE) / (2 * VIEW_RANGE)) * PLOT_SIZE;
}

function fromCanvas(px: number): number {
  return (px / PLOT_SIZE) * 2 * VIEW_RANGE - VIEW_RANGE;
}

function drawScene(
  ctx: CanvasRenderingContext2D,
  state: GANState,
  nSamples: number
): void {
  const {G, D, config} = state;

  // --- Background heatmap: D(x) ---
  const step = PLOT_SIZE / HEATMAP_RES;
  for (let row = 0; row < HEATMAP_RES; row++) {
    for (let col = 0; col < HEATMAP_RES; col++) {
      const px = col * step + step / 2;
      const py = row * step + step / 2;
      const x = fromCanvas(px);
      const y = fromCanvas(py);
      const score = discriminate(D, [x, y]);
      // blue = fake (low), red = real (high)
      const r = Math.round(score * 160);
      const b = Math.round((1 - score) * 160);
      ctx.fillStyle = `rgb(${r},20,${b})`;
      ctx.fillRect(col * step, row * step, step, step);
    }
  }

  // --- Real samples ---
  ctx.fillStyle = 'rgba(100,220,100,0.75)';
  for (let i = 0; i < nSamples; i++) {
    const pt = sampleReal(config.dist);
    const cx = toCanvas(pt[0]);
    const cy = toCanvas(pt[1]);
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
    ctx.fill();
  }

  // --- Fake samples ---
  ctx.fillStyle = 'rgba(255,180,30,0.75)';
  for (let i = 0; i < nSamples; i++) {
    const z = sampleLatent(config.latentDim);
    const fwd = mlpForward(G, z);
    const out = fwd.acts[fwd.acts.length - 1];
    const cx = toCanvas(out[0]);
    const cy = toCanvas(out[1]);
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
    ctx.fill();
  }

  // --- Axes ---
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1;
  const mid = toCanvas(0);
  ctx.beginPath(); ctx.moveTo(mid, 0); ctx.lineTo(mid, PLOT_SIZE); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(PLOT_SIZE, mid); ctx.stroke();
}

function drawLossCurves(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  lossDHistory: number[],
  lossGHistory: number[]
): void {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, width, height);

  if (lossDHistory.length < 2) return;

  const allVals = [...lossDHistory, ...lossGHistory];
  const maxV = Math.min(Math.max(...allVals) * 1.1, 4);
  const minV = 0;
  const pad = {t: 10, r: 10, b: 25, l: 40};
  const pw = width - pad.l - pad.r;
  const ph = height - pad.t - pad.b;

  function mapX(i: number): number {
    return pad.l + (i / (lossDHistory.length - 1)) * pw;
  }
  function mapY(v: number): number {
    return pad.t + (1 - (v - minV) / (maxV - minV)) * ph;
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth = 1;
  for (let v = 0; v <= 2; v += 0.5) {
    const y = mapY(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '9px monospace';
    ctx.fillText(v.toFixed(1), 2, y + 3);
  }

  // D loss
  ctx.strokeStyle = '#4fc3f7';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  lossDHistory.forEach((v, i) => {
    const x = mapX(i); const y = mapY(v);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // G loss
  ctx.strokeStyle = '#ffb74d';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  lossGHistory.forEach((v, i) => {
    const x = mapX(i); const y = mapY(v);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Labels
  ctx.font = '10px monospace';
  ctx.fillStyle = '#4fc3f7';
  ctx.fillText('D loss', pad.l + 4, pad.t + 12);
  ctx.fillStyle = '#ffb74d';
  ctx.fillText('G loss', pad.l + 55, pad.t + 12);

  // X axis
  ctx.strokeStyle = 'rgba(255,255,255,0.3)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + ph);
  ctx.lineTo(pad.l + pw, pad.t + ph);
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.font = '9px monospace';
  ctx.fillText('steps (last 300)', pad.l, height - 4);
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

interface Controls {
  playPauseBtn: HTMLButtonElement;
  stepBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  distSelect: HTMLSelectElement;
  lrInput: HTMLInputElement;
  lrVal: HTMLSpanElement;
  latentInput: HTMLInputElement;
  latentVal: HTMLSpanElement;
  hiddenInput: HTMLInputElement;
  hiddenVal: HTMLSpanElement;
  dStepsInput: HTMLInputElement;
  dStepsVal: HTMLSpanElement;
  stepCount: HTMLElement;
  lossDEl: HTMLElement;
  lossGEl: HTMLElement;
}

function getConfig(controls: Controls): GANConfig {
  return {
    latentDim: parseInt(controls.latentInput.value, 10),
    hiddenSize: parseInt(controls.hiddenInput.value, 10),
    dStepsPerG: parseInt(controls.dStepsInput.value, 10),
    lr: parseFloat(controls.lrInput.value),
    batchSize: 32,
    dist: controls.distSelect.value as DistName
  };
}

function init(): void {
  const mainCanvas = document.getElementById('mainCanvas') as HTMLCanvasElement;
  const lossCanvas = document.getElementById('lossCanvas') as HTMLCanvasElement;
  const ctx = mainCanvas.getContext('2d')!;
  const lossCtx = lossCanvas.getContext('2d')!;

  mainCanvas.width = PLOT_SIZE;
  mainCanvas.height = PLOT_SIZE;
  lossCanvas.width = lossCanvas.offsetWidth || 380;
  lossCanvas.height = lossCanvas.offsetHeight || 120;

  const controls: Controls = {
    playPauseBtn: document.getElementById('playPause') as HTMLButtonElement,
    stepBtn: document.getElementById('stepBtn') as HTMLButtonElement,
    resetBtn: document.getElementById('resetBtn') as HTMLButtonElement,
    distSelect: document.getElementById('distSelect') as HTMLSelectElement,
    lrInput: document.getElementById('lrInput') as HTMLInputElement,
    lrVal: document.getElementById('lrVal') as HTMLSpanElement,
    latentInput: document.getElementById('latentInput') as HTMLInputElement,
    latentVal: document.getElementById('latentVal') as HTMLSpanElement,
    hiddenInput: document.getElementById('hiddenInput') as HTMLInputElement,
    hiddenVal: document.getElementById('hiddenVal') as HTMLSpanElement,
    dStepsInput: document.getElementById('dStepsInput') as HTMLInputElement,
    dStepsVal: document.getElementById('dStepsVal') as HTMLSpanElement,
    stepCount: document.getElementById('stepCount')!,
    lossDEl: document.getElementById('lossDVal')!,
    lossGEl: document.getElementById('lossGVal')!
  };

  let state = makeGAN(getConfig(controls));
  let running = false;
  let animId = 0;
  let stepsPerFrame = 5;

  function render(): void {
    drawScene(ctx, state, 150);
    drawLossCurves(lossCtx, lossCanvas.width, lossCanvas.height,
      state.lossDHistory, state.lossGHistory);
    controls.stepCount.textContent = String(state.step);
    controls.lossDEl.textContent = state.lossD.toFixed(4);
    controls.lossGEl.textContent = state.lossG.toFixed(4);
  }

  function loop(): void {
    for (let i = 0; i < stepsPerFrame; i++) {
      ganTrainStep(state);
    }
    render();
    if (running) animId = requestAnimationFrame(loop);
  }

  controls.playPauseBtn.addEventListener('click', () => {
    running = !running;
    controls.playPauseBtn.textContent = running ? 'Pause' : 'Play';
    if (running) loop();
    else cancelAnimationFrame(animId);
  });

  controls.stepBtn.addEventListener('click', () => {
    if (!running) { ganTrainStep(state); render(); }
  });

  controls.resetBtn.addEventListener('click', () => {
    running = false;
    cancelAnimationFrame(animId);
    controls.playPauseBtn.textContent = 'Play';
    state = makeGAN(getConfig(controls));
    render();
  });

  function syncLabel(input: HTMLInputElement, span: HTMLSpanElement): void {
    span.textContent = input.value;
    input.addEventListener('input', () => { span.textContent = input.value; });
  }
  syncLabel(controls.lrInput, controls.lrVal);
  syncLabel(controls.latentInput, controls.latentVal);
  syncLabel(controls.hiddenInput, controls.hiddenVal);
  syncLabel(controls.dStepsInput, controls.dStepsVal);

  render();
}

window.addEventListener('DOMContentLoaded', init);
