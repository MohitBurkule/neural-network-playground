// Elman RNN Sequence Lab
// Task: predict input delayed by k steps (echo/delay task)

// ── Matrix helpers ────────────────────────────────────────────────────────────
type Mat = { rows: number; cols: number; data: Float64Array };

function mat(rows: number, cols: number): Mat {
  return { rows, cols, data: new Float64Array(rows * cols) };
}

function matGet(m: Mat, r: number, c: number): number {
  return m.data[r * m.cols + c];
}

function matSet(m: Mat, r: number, c: number, v: number): void {
  m.data[r * m.cols + c] = v;
}

function matFill(m: Mat, v: number): void {
  m.data.fill(v);
}

function matRand(m: Mat, scale: number): void {
  for (let i = 0; i < m.data.length; i++) {
    m.data[i] = (Math.random() * 2 - 1) * scale;
  }
}

// result = A (rows_a x cols_a) @ B (cols_a x cols_b) + bias (cols_b)
function matMulVec(A: Mat, x: Float64Array, bias: Float64Array): Float64Array {
  const out = new Float64Array(A.rows);
  for (let r = 0; r < A.rows; r++) {
    let s = bias[r];
    for (let c = 0; c < A.cols; c++) {
      s += matGet(A, r, c) * x[c];
    }
    out[r] = s;
  }
  return out;
}

function tanh(x: number): number { return Math.tanh(x); }
function dtanh(y: number): number { return 1 - y * y; } // derivative given tanh output

// ── RNN Parameters ────────────────────────────────────────────────────────────
interface RNNParams {
  Wxh: Mat; // input->hidden  [H x inputSize]
  Whh: Mat; // hidden->hidden [H x H]
  bh: Float64Array;
  Why: Mat; // hidden->output [outputSize x H]
  by: Float64Array;
}

interface RNNGrads {
  Wxh: Mat;
  Whh: Mat;
  bh: Float64Array;
  Why: Mat;
  by: Float64Array;
}

function makeParams(inputSize: number, hiddenSize: number, outputSize: number): RNNParams {
  const scale = 0.1;
  const Wxh = mat(hiddenSize, inputSize); matRand(Wxh, scale);
  const Whh = mat(hiddenSize, hiddenSize); matRand(Whh, scale * 0.5);
  const bh = new Float64Array(hiddenSize);
  const Why = mat(outputSize, hiddenSize); matRand(Why, scale);
  const by = new Float64Array(outputSize);
  return { Wxh, Whh, bh, Why, by };
}

function zeroGrads(p: RNNParams): RNNGrads {
  return {
    Wxh: mat(p.Wxh.rows, p.Wxh.cols),
    Whh: mat(p.Whh.rows, p.Whh.cols),
    bh: new Float64Array(p.bh.length),
    Why: mat(p.Why.rows, p.Why.cols),
    by: new Float64Array(p.by.length),
  };
}

// ── Forward pass (returns all hidden states & pre-tanh activations) ───────────
interface StepCache {
  x: Float64Array;
  hPrev: Float64Array;
  z: Float64Array;   // pre-tanh
  h: Float64Array;   // post-tanh hidden state
  y: Float64Array;   // raw output (linear)
}

function forward(
  p: RNNParams,
  xs: Float64Array[], // sequence of input vectors
  h0: Float64Array
): { caches: StepCache[]; hLast: Float64Array } {
  const caches: StepCache[] = [];
  let hPrev = h0;
  for (const x of xs) {
    // z = Wxh*x + Whh*h_prev + bh
    const zx = matMulVec(p.Wxh, x, p.bh);
    const zh = matMulVec(p.Whh, hPrev, new Float64Array(p.bh.length));
    const z = new Float64Array(zx.length);
    for (let i = 0; i < z.length; i++) z[i] = zx[i] + zh[i];
    const h = z.map(tanh);
    const y = matMulVec(p.Why, h, p.by);
    caches.push({ x, hPrev, z, h, y });
    hPrev = h;
  }
  return { caches, hLast: hPrev };
}

// ── BPTT ──────────────────────────────────────────────────────────────────────
// MSE loss, returns grads and loss
function bptt(
  p: RNNParams,
  caches: StepCache[],
  targets: Float64Array[]  // same length as caches
): { grads: RNNGrads; loss: number } {
  const grads = zeroGrads(p);
  let loss = 0;
  let dh_next = new Float64Array(p.Whh.rows);

  for (let t = caches.length - 1; t >= 0; t--) {
    const { x, hPrev, z: _z, h, y } = caches[t];
    const target = targets[t];

    // output loss (MSE) and grad
    const dy = new Float64Array(y.length);
    for (let i = 0; i < y.length; i++) {
      const diff = y[i] - target[i];
      loss += diff * diff;
      dy[i] = 2 * diff; // dL/dy
    }

    // grad Why, by
    for (let r = 0; r < p.Why.rows; r++) {
      grads.by[r] += dy[r];
      for (let c = 0; c < p.Why.cols; c++) {
        grads.Why.data[r * p.Why.cols + c] += dy[r] * h[c];
      }
    }

    // dh = Why^T * dy + dh_next
    const dh = new Float64Array(h.length);
    for (let c = 0; c < p.Why.cols; c++) {
      for (let r = 0; r < p.Why.rows; r++) {
        dh[c] += matGet(p.Why, r, c) * dy[r];
      }
      dh[c] += dh_next[c];
    }

    // backprop through tanh: dz = dh * (1 - h^2)
    const dz = new Float64Array(dh.length);
    for (let i = 0; i < dz.length; i++) dz[i] = dh[i] * dtanh(h[i]);

    // grad bh
    for (let i = 0; i < grads.bh.length; i++) grads.bh[i] += dz[i];

    // grad Wxh
    for (let r = 0; r < p.Wxh.rows; r++) {
      for (let c = 0; c < p.Wxh.cols; c++) {
        grads.Wxh.data[r * p.Wxh.cols + c] += dz[r] * x[c];
      }
    }

    // grad Whh
    for (let r = 0; r < p.Whh.rows; r++) {
      for (let c = 0; c < p.Whh.cols; c++) {
        grads.Whh.data[r * p.Whh.cols + c] += dz[r] * hPrev[c];
      }
    }

    // dh_next = Whh^T * dz
    dh_next = new Float64Array(p.Whh.cols);
    for (let c = 0; c < p.Whh.cols; c++) {
      for (let r = 0; r < p.Whh.rows; r++) {
        dh_next[c] += matGet(p.Whh, r, c) * dz[r];
      }
    }
  }

  loss /= caches.length;
  return { grads, loss };
}

function clipGrads(grads: RNNGrads, clip: number): void {
  const arrays = [grads.Wxh.data, grads.Whh.data, grads.bh, grads.Why.data, grads.by];
  for (const arr of arrays) {
    for (let i = 0; i < arr.length; i++) {
      arr[i] = Math.max(-clip, Math.min(clip, arr[i]));
    }
  }
}

function sgdUpdate(p: RNNParams, grads: RNNGrads, lr: number): void {
  const pairs: [Float64Array, Float64Array][] = [
    [p.Wxh.data, grads.Wxh.data],
    [p.Whh.data, grads.Whh.data],
    [p.bh, grads.bh],
    [p.Why.data, grads.Why.data],
    [p.by, grads.by],
  ];
  for (const [param, grad] of pairs) {
    for (let i = 0; i < param.length; i++) {
      param[i] -= lr * grad[i];
    }
  }
}

// ── Task generation ───────────────────────────────────────────────────────────
// Echo/delay task: output = input shifted by `delay` steps (pad with 0)
function generateSequence(seqLen: number, delay: number = 2): { inputs: number[]; targets: number[] } {
  const inputs: number[] = [];
  for (let i = 0; i < seqLen; i++) inputs.push(Math.random() > 0.5 ? 1 : 0);
  const targets: number[] = inputs.map((_v, i) => (i >= delay ? inputs[i - delay] : 0));
  return { inputs, targets };
}

// ── State ─────────────────────────────────────────────────────────────────────
interface AppState {
  params: RNNParams;
  hiddenSize: number;
  seqLen: number;
  bpttLen: number;
  lr: number;
  delay: number;
  running: boolean;
  step: number;
  lossHistory: number[];
  // last forward pass data for visualization
  lastInputs: number[];
  lastTargets: number[];
  lastPredictions: number[];
  lastHiddenStates: Float64Array[]; // per timestep
  h0: Float64Array;
}

let state: AppState;

function initState(hiddenSize: number, seqLen: number, bpttLen: number, lr: number, delay: number): AppState {
  const inputSize = 1;
  const outputSize = 1;
  const params = makeParams(inputSize, hiddenSize, outputSize);
  return {
    params, hiddenSize, seqLen, bpttLen, lr, delay,
    running: false, step: 0,
    lossHistory: [],
    lastInputs: [], lastTargets: [], lastPredictions: [], lastHiddenStates: [],
    h0: new Float64Array(hiddenSize),
  };
}

function trainStep(s: AppState): void {
  const { inputs, targets } = generateSequence(s.seqLen, s.delay);
  s.lastInputs = inputs;
  s.lastTargets = targets;

  // truncated BPTT: chunk by bpttLen
  let h = new Float64Array(s.h0);
  let totalLoss = 0;
  let chunks = 0;
  const allCaches: StepCache[] = [];

  for (let start = 0; start < s.seqLen; start += s.bpttLen) {
    const end = Math.min(start + s.bpttLen, s.seqLen);
    const xs = inputs.slice(start, end).map(v => new Float64Array([v]));
    const ts = targets.slice(start, end).map(v => new Float64Array([v]));

    const { caches, hLast } = forward(s.params, xs, h);
    const { grads, loss } = bptt(s.params, caches, ts);
    clipGrads(grads, 5);
    sgdUpdate(s.params, grads, s.lr);

    h = hLast;
    totalLoss += loss;
    chunks++;
    allCaches.push(...caches);
  }

  s.h0 = h; // carry hidden state
  s.lossHistory.push(totalLoss / chunks);
  if (s.lossHistory.length > 300) s.lossHistory.shift();

  // store vis data
  s.lastPredictions = allCaches.map(c => c.y[0]);
  s.lastHiddenStates = allCaches.map(c => c.h);
  s.step++;
}

// ── Canvas drawing helpers ────────────────────────────────────────────────────
function clearCanvas(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, w, h);
}

function heatColor(v: number, vmin: number, vmax: number): string {
  const t = vmax === vmin ? 0.5 : (v - vmin) / (vmax - vmin);
  const c = Math.round(t * 255);
  // blue->white->red
  if (t < 0.5) {
    const r = Math.round(2 * t * 200);
    return `rgb(${r},${r},255)`;
  } else {
    const g = Math.round((1 - (t - 0.5) * 2) * 200);
    return `rgb(255,${g},${g})`;
  }
}

// ── Draw sequence prediction ──────────────────────────────────────────────────
function drawSequence(canvas: HTMLCanvasElement, s: AppState): void {
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  clearCanvas(ctx, W, H);

  const n = s.lastInputs.length;
  if (n === 0) return;

  const pad = 30;
  const innerW = W - 2 * pad;
  const innerH = H - 2 * pad;
  const midY = pad + innerH / 2;

  ctx.strokeStyle = '#333';
  ctx.beginPath(); ctx.moveTo(pad, midY); ctx.lineTo(W - pad, midY); ctx.stroke();

  // axes labels
  ctx.fillStyle = '#aaa';
  ctx.font = '11px monospace';
  ctx.fillText('1', pad - 18, pad + 4);
  ctx.fillText('0', pad - 18, H - pad + 4);

  const dx = innerW / (n - 1 || 1);

  // draw input
  ctx.strokeStyle = '#4fc3f7';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  s.lastInputs.forEach((v, i) => {
    const x = pad + i * dx;
    const y = H - pad - v * innerH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // draw target
  ctx.strokeStyle = '#81c784';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  s.lastTargets.forEach((v, i) => {
    const x = pad + i * dx;
    const y = H - pad - v * innerH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  // draw prediction
  if (s.lastPredictions.length === n) {
    ctx.strokeStyle = '#ff8a65';
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.lastPredictions.forEach((v, i) => {
      const x = pad + i * dx;
      const y = H - pad - Math.max(0, Math.min(1, v)) * innerH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // legend
  ctx.font = '10px monospace';
  const items = [['#4fc3f7', 'input'], ['#81c784', 'target'], ['#ff8a65', 'predicted']];
  items.forEach(([color, label], i) => {
    ctx.fillStyle = color as string;
    ctx.fillRect(pad + i * 90, 6, 12, 8);
    ctx.fillStyle = '#ccc';
    ctx.fillText(label, pad + i * 90 + 15, 14);
  });

  ctx.fillStyle = '#888';
  ctx.fillText(`step ${s.step}`, W - 70, 14);
}

// ── Draw hidden state heatmap ─────────────────────────────────────────────────
function drawHiddenStates(canvas: HTMLCanvasElement, s: AppState): void {
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  clearCanvas(ctx, W, H);

  const states = s.lastHiddenStates;
  if (!states.length) return;

  const T = states.length;
  const HS = states[0].length;
  const padL = 50, padT = 20, padR = 10, padB = 20;
  const cellW = (W - padL - padR) / T;
  const cellH = (H - padT - padB) / HS;

  for (let t = 0; t < T; t++) {
    for (let h = 0; h < HS; h++) {
      const v = states[t][h];
      ctx.fillStyle = heatColor(v, -1, 1);
      ctx.fillRect(padL + t * cellW, padT + h * cellH, Math.max(1, cellW - 0.5), Math.max(1, cellH - 0.5));
    }
  }

  // axis labels
  ctx.fillStyle = '#aaa';
  ctx.font = '10px monospace';
  ctx.fillText('h units', 2, H / 2);
  ctx.fillText('time →', padL, H - 4);

  // tick a few time labels
  [0, Math.floor(T / 2), T - 1].forEach(t => {
    ctx.fillStyle = '#888';
    ctx.fillText(String(t), padL + t * cellW, padT - 4);
  });
}

// ── Draw Whh heatmap ──────────────────────────────────────────────────────────
function drawWhh(canvas: HTMLCanvasElement, s: AppState): void {
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  clearCanvas(ctx, W, H);

  const Whh = s.params.Whh;
  const HS = Whh.rows;
  const padL = 40, padT = 20, padR = 10, padB = 20;
  const cellW = (W - padL - padR) / HS;
  const cellH = (H - padT - padB) / HS;

  let vmin = Infinity, vmax = -Infinity;
  for (let i = 0; i < Whh.data.length; i++) {
    vmin = Math.min(vmin, Whh.data[i]);
    vmax = Math.max(vmax, Whh.data[i]);
  }

  for (let r = 0; r < HS; r++) {
    for (let c = 0; c < HS; c++) {
      ctx.fillStyle = heatColor(matGet(Whh, r, c), vmin, vmax);
      ctx.fillRect(padL + c * cellW, padT + r * cellH, Math.max(1, cellW - 0.5), Math.max(1, cellH - 0.5));
    }
  }

  ctx.fillStyle = '#aaa';
  ctx.font = '10px monospace';
  ctx.fillText('Whh', 4, H / 2);
  ctx.fillText(`[${vmin.toFixed(2)}, ${vmax.toFixed(2)}]`, padL, H - 4);
}

// ── Draw loss curve ───────────────────────────────────────────────────────────
function drawLoss(canvas: HTMLCanvasElement, s: AppState): void {
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  clearCanvas(ctx, W, H);

  const history = s.lossHistory;
  if (history.length < 2) return;

  const pad = 30;
  const innerW = W - 2 * pad;
  const innerH = H - 2 * pad;
  const maxLoss = Math.max(...history, 0.01);
  const minLoss = Math.min(...history);

  ctx.strokeStyle = '#555';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.lineTo(W - pad, H - pad); ctx.stroke();

  ctx.strokeStyle = '#ffd54f';
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((v, i) => {
    const x = pad + (i / (history.length - 1)) * innerW;
    const y = H - pad - ((v - minLoss) / (maxLoss - minLoss || 1)) * innerH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = '#aaa';
  ctx.font = '10px monospace';
  ctx.fillText('loss', 2, pad + 4);
  ctx.fillText(maxLoss.toFixed(3), pad + 2, pad + 12);
  ctx.fillText(minLoss.toFixed(3), pad + 2, H - pad - 2);
  const last = history[history.length - 1];
  ctx.fillStyle = '#ffd54f';
  ctx.fillText(`cur: ${last.toFixed(4)}`, W - 100, pad + 12);
}

// ── Main loop ─────────────────────────────────────────────────────────────────
let animId: number | null = null;

function render(): void {
  const canvSeq = document.getElementById('canvas-seq') as HTMLCanvasElement;
  const canvHidden = document.getElementById('canvas-hidden') as HTMLCanvasElement;
  const canvWhh = document.getElementById('canvas-whh') as HTMLCanvasElement;
  const canvLoss = document.getElementById('canvas-loss') as HTMLCanvasElement;

  drawSequence(canvSeq, state);
  drawHiddenStates(canvHidden, state);
  drawWhh(canvWhh, state);
  drawLoss(canvLoss, state);
}

function loop(): void {
  if (state.running) {
    trainStep(state);
    render();
    animId = requestAnimationFrame(loop);
  }
}

function doStep(): void {
  trainStep(state);
  render();
}

function resetState(): void {
  const hiddenSize = parseInt((document.getElementById('ctrl-hidden') as HTMLInputElement).value);
  const seqLen = parseInt((document.getElementById('ctrl-seqlen') as HTMLInputElement).value);
  const bpttLen = parseInt((document.getElementById('ctrl-bptt') as HTMLInputElement).value);
  const lr = parseFloat((document.getElementById('ctrl-lr') as HTMLInputElement).value);
  const delay = parseInt((document.getElementById('ctrl-delay') as HTMLInputElement).value);
  state = initState(hiddenSize, seqLen, bpttLen, lr, delay);
  state.running = false;
  render();
}

// ── Boot ──────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  // read initial slider values
  const hiddenSize = parseInt((document.getElementById('ctrl-hidden') as HTMLInputElement).value);
  const seqLen = parseInt((document.getElementById('ctrl-seqlen') as HTMLInputElement).value);
  const bpttLen = parseInt((document.getElementById('ctrl-bptt') as HTMLInputElement).value);
  const lr = parseFloat((document.getElementById('ctrl-lr') as HTMLInputElement).value);
  const delay = parseInt((document.getElementById('ctrl-delay') as HTMLInputElement).value);
  state = initState(hiddenSize, seqLen, bpttLen, lr, delay);

  // sync display labels
  function syncLabel(id: string, labelId: string): void {
    const el = document.getElementById(id) as HTMLInputElement;
    const lbl = document.getElementById(labelId) as HTMLSpanElement;
    lbl.textContent = el.value;
    el.addEventListener('input', () => {
      lbl.textContent = el.value;
      // live-update lr
      if (id === 'ctrl-lr') state.lr = parseFloat(el.value);
    });
  }
  syncLabel('ctrl-lr', 'lbl-lr');
  syncLabel('ctrl-hidden', 'lbl-hidden');
  syncLabel('ctrl-seqlen', 'lbl-seqlen');
  syncLabel('ctrl-bptt', 'lbl-bptt');
  syncLabel('ctrl-delay', 'lbl-delay');

  document.getElementById('btn-play')!.addEventListener('click', () => {
    state.running = true;
    if (animId === null) loop();
  });

  document.getElementById('btn-pause')!.addEventListener('click', () => {
    state.running = false;
    if (animId !== null) { cancelAnimationFrame(animId); animId = null; }
  });

  document.getElementById('btn-step')!.addEventListener('click', () => {
    state.running = false;
    if (animId !== null) { cancelAnimationFrame(animId); animId = null; }
    doStep();
  });

  document.getElementById('btn-reset')!.addEventListener('click', () => {
    state.running = false;
    if (animId !== null) { cancelAnimationFrame(animId); animId = null; }
    resetState();
  });

  render();
});
