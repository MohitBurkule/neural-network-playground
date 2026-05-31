// Echo State Network (Reservoir Computing) Lab
export {};

// ── Tiny matrix helpers ───────────────────────────────────────────────────────
type Mat = { rows: number; cols: number; data: Float64Array };

function mat(rows: number, cols: number): Mat {
  return { rows, cols, data: new Float64Array(rows * cols) };
}
function mget(m: Mat, r: number, c: number): number { return m.data[r * m.cols + c]; }
function mset(m: Mat, r: number, c: number, v: number): void { m.data[r * m.cols + c] = v; }

// A (rows x k) @ x (k) -> out (rows)
function mvmul(A: Mat, x: Float64Array, out: Float64Array): void {
  const R = A.rows, C = A.cols;
  for (let r = 0; r < R; r++) {
    let s = 0;
    for (let c = 0; c < C; c++) s += mget(A, r, c) * x[c];
    out[r] = s;
  }
}

// A^T (cols x rows) @ x (rows) -> out (cols)
function mvmulT(A: Mat, x: Float64Array, out: Float64Array): void {
  const R = A.rows, C = A.cols;
  out.fill(0);
  for (let r = 0; r < R; r++) {
    const xr = x[r];
    for (let c = 0; c < C; c++) out[c] += mget(A, r, c) * xr;
  }
}

// ── Random helpers ────────────────────────────────────────────────────────────
function randn(): number {
  // Box-Muller
  const u = 1 - Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ── Signals ───────────────────────────────────────────────────────────────────
type SignalName = 'sine' | 'sum-of-sines' | 'mackey-glass' | 'square' | 'lorenz';

function generateSignal(name: SignalName, T: number): Float64Array {
  const s = new Float64Array(T);
  if (name === 'sine') {
    for (let t = 0; t < T; t++) s[t] = Math.sin(2 * Math.PI * t / 40);
  } else if (name === 'sum-of-sines') {
    for (let t = 0; t < T; t++)
      s[t] = 0.5 * Math.sin(2 * Math.PI * t / 30) + 0.5 * Math.sin(2 * Math.PI * t / 73);
  } else if (name === 'mackey-glass') {
    // Mackey-Glass delay differential equation (Euler, tau=17)
    const tau = 17, a = 0.2, b = 0.1, n = 10, dt = 1;
    const buf = new Float64Array(T + tau + 1);
    for (let i = 0; i <= tau; i++) buf[i] = 1.2;
    for (let t = tau; t < T + tau; t++) {
      const xt = buf[t], xtau = buf[t - tau];
      buf[t + 1] = xt + dt * (a * xtau / (1 + Math.pow(xtau, n)) - b * xt);
    }
    // normalise to [-1,1]
    let mn = Infinity, mx = -Infinity;
    for (let t = 0; t < T; t++) { mn = Math.min(mn, buf[t + tau]); mx = Math.max(mx, buf[t + tau]); }
    const rng = mx - mn || 1;
    for (let t = 0; t < T; t++) s[t] = 2 * (buf[t + tau] - mn) / rng - 1;
  } else if (name === 'square') {
    for (let t = 0; t < T; t++) s[t] = Math.sin(2 * Math.PI * t / 50) >= 0 ? 1 : -1;
  } else if (name === 'lorenz') {
    // Lorenz x-component, normalised
    const dt = 0.02, sigma = 10, rho = 28, beta = 8 / 3;
    let x = 1, y = 1, z = 1;
    const buf: number[] = [];
    for (let t = 0; t < T + 500; t++) {
      const dx = sigma * (y - x), dy = x * (rho - z) - y, dz = x * y - beta * z;
      x += dt * dx; y += dt * dy; z += dt * dz;
      if (t >= 500) buf.push(x);
    }
    let mn = Infinity, mx = -Infinity;
    buf.forEach(v => { mn = Math.min(mn, v); mx = Math.max(mx, v); });
    const rng = mx - mn || 1;
    for (let t = 0; t < T; t++) s[t] = 2 * (buf[t] - mn) / rng - 1;
  }
  return s;
}

// ── Reservoir construction ────────────────────────────────────────────────────
function buildReservoir(N: number, sparsity: number, targetSR: number): Mat {
  const W = mat(N, N);
  // sparse random weights
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      if (Math.random() < sparsity) mset(W, r, c, randn());
    }
  }
  // power iteration to estimate spectral radius
  const sr = spectralRadiusApprox(W, N);
  if (sr > 0) {
    const scale = targetSR / sr;
    for (let i = 0; i < W.data.length; i++) W.data[i] *= scale;
  }
  return W;
}

function spectralRadiusApprox(W: Mat, N: number): number {
  // ~30 iterations of power method
  let v = new Float64Array(N);
  for (let i = 0; i < N; i++) v[i] = randn();
  let norm = 0;
  for (let i = 0; i < N; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < N; i++) v[i] /= norm;
  let ev = 1;
  const tmp = new Float64Array(N);
  for (let iter = 0; iter < 40; iter++) {
    mvmul(W, v, tmp);
    norm = 0;
    for (let i = 0; i < N; i++) norm += tmp[i] * tmp[i];
    ev = Math.sqrt(norm) || 1e-12;
    for (let i = 0; i < N; i++) v[i] = tmp[i] / ev;
  }
  return ev;
}

// ── Ridge regression (closed form): X (T x N) W = y (T)  =>  w = (X^T X + λI)^-1 X^T y
// Solved via Cholesky / direct inversion for small N
function ridgeRegression(X: Float64Array[], y: Float64Array, lambda: number): Float64Array {
  const T = X.length, N = X[0].length;
  // A = X^T X + lambda I,  b = X^T y
  const A = new Float64Array(N * N);
  const b = new Float64Array(N);
  for (let t = 0; t < T; t++) {
    const xt = X[t];
    for (let i = 0; i < N; i++) {
      b[i] += xt[i] * y[t];
      for (let j = 0; j < N; j++) A[i * N + j] += xt[i] * xt[j];
    }
  }
  for (let i = 0; i < N; i++) A[i * N + i] += lambda;
  // Solve A w = b via Gaussian elimination
  return solveLinear(A, b, N);
}

function solveLinear(A: Float64Array, b: Float64Array, N: number): Float64Array {
  // Augmented matrix [A | b]
  const M = new Float64Array(N * (N + 1));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) M[i * (N + 1) + j] = A[i * N + j];
    M[i * (N + 1) + N] = b[i];
  }
  // Forward elimination with partial pivot
  for (let col = 0; col < N; col++) {
    let pivot = col;
    let pivVal = Math.abs(M[col * (N + 1) + col]);
    for (let row = col + 1; row < N; row++) {
      const v = Math.abs(M[row * (N + 1) + col]);
      if (v > pivVal) { pivVal = v; pivot = row; }
    }
    if (pivot !== col) {
      for (let j = 0; j <= N; j++) {
        const tmp = M[col * (N + 1) + j];
        M[col * (N + 1) + j] = M[pivot * (N + 1) + j];
        M[pivot * (N + 1) + j] = tmp;
      }
    }
    const diag = M[col * (N + 1) + col];
    if (Math.abs(diag) < 1e-15) continue;
    for (let row = col + 1; row < N; row++) {
      const factor = M[row * (N + 1) + col] / diag;
      for (let j = col; j <= N; j++) M[row * (N + 1) + j] -= factor * M[col * (N + 1) + j];
    }
  }
  // Back substitution
  const x = new Float64Array(N);
  for (let row = N - 1; row >= 0; row--) {
    let sum = M[row * (N + 1) + N];
    for (let j = row + 1; j < N; j++) sum -= M[row * (N + 1) + j] * x[j];
    const d = M[row * (N + 1) + row];
    x[row] = Math.abs(d) < 1e-15 ? 0 : sum / d;
  }
  return x;
}

// ── ESN state update ──────────────────────────────────────────────────────────
function esnStep(
  state: Float64Array, Wr: Mat, Win: Float64Array,
  input: number, leak: number
): void {
  const N = state.length;
  const pre = new Float64Array(N);
  mvmul(Wr, state, pre);
  for (let i = 0; i < N; i++) {
    pre[i] += Win[i] * input;
    pre[i] = Math.tanh(pre[i]);
    state[i] = (1 - leak) * state[i] + leak * pre[i];
  }
}

// ── ESN class ─────────────────────────────────────────────────────────────────
interface ESNConfig {
  N: number;
  spectralRadius: number;
  inputScale: number;
  leak: number;
  ridge: number;
  sparsity: number;
}

class ESN {
  Wr: Mat;
  Win: Float64Array;
  Wout: Float64Array | null = null;
  state: Float64Array;
  cfg: ESNConfig;

  constructor(cfg: ESNConfig) {
    this.cfg = cfg;
    const { N, spectralRadius, inputScale, sparsity } = cfg;
    this.Wr = buildReservoir(N, sparsity, spectralRadius);
    this.Win = new Float64Array(N);
    for (let i = 0; i < N; i++) this.Win[i] = (Math.random() * 2 - 1) * inputScale;
    this.state = new Float64Array(N);
  }

  reset(): void { this.state.fill(0); }

  // Collect states over a signal (teacher forcing), skip washout steps
  collectStates(signal: Float64Array, washout: number): { states: Float64Array[]; targets: Float64Array } {
    const { N, leak } = this.cfg;
    this.reset();
    const states: Float64Array[] = [];
    const targets: number[] = [];
    for (let t = 0; t < signal.length - 1; t++) {
      esnStep(this.state, this.Wr, this.Win, signal[t], leak);
      if (t >= washout) {
        states.push(new Float64Array(this.state));
        targets.push(signal[t + 1]);
      }
    }
    return { states, targets: new Float64Array(targets) };
  }

  train(signal: Float64Array, washout: number): void {
    const { states, targets } = this.collectStates(signal, washout);
    this.Wout = ridgeRegression(states, targets, this.cfg.ridge);
  }

  // Read out from current state
  readout(): number {
    if (!this.Wout) return 0;
    let s = 0;
    for (let i = 0; i < this.state.length; i++) s += this.Wout[i] * this.state[i];
    return s;
  }

  // Teacher-forced prediction (for training region visualisation)
  predictTeacherForced(signal: Float64Array, washout: number): Float64Array {
    const { leak } = this.cfg;
    this.reset();
    const preds: number[] = [];
    for (let t = 0; t < signal.length - 1; t++) {
      esnStep(this.state, this.Wr, this.Win, signal[t], leak);
      if (t >= washout) preds.push(this.readout());
    }
    return new Float64Array(preds);
  }

  // Free-run: feed own output back as next input
  freeRun(seed: number, steps: number): Float64Array {
    const { leak } = this.cfg;
    const out = new Float64Array(steps);
    let inp = seed;
    for (let t = 0; t < steps; t++) {
      esnStep(this.state, this.Wr, this.Win, inp, leak);
      inp = this.readout();
      out[t] = inp;
    }
    return out;
  }

  // Reservoir activations for a sample of neurons
  getActivations(signal: Float64Array, washout: number, nSample: number): Float64Array[] {
    const { leak } = this.cfg;
    this.reset();
    const rows: Float64Array[] = [];
    const N = this.state.length;
    const step = Math.max(1, Math.floor(N / nSample));
    const indices: number[] = [];
    for (let i = 0; i < N && indices.length < nSample; i += step) indices.push(i);
    for (let _ = 0; _ < indices.length; _++) rows.push(new Float64Array(signal.length - 1 - washout));
    for (let t = 0; t < signal.length - 1; t++) {
      esnStep(this.state, this.Wr, this.Win, signal[t], leak);
      if (t >= washout) {
        for (let k = 0; k < indices.length; k++) {
          rows[k][t - washout] = this.state[indices[k]];
        }
      }
    }
    return rows;
  }
}

// ── UI ────────────────────────────────────────────────────────────────────────
interface AppState {
  esn: ESN | null;
  signal: Float64Array | null;
  trainPreds: Float64Array | null;
  freeRunPreds: Float64Array | null;
  activations: Float64Array[] | null;
  trained: boolean;
  trainLen: number;
  washout: number;
}

const app: AppState = {
  esn: null,
  signal: null,
  trainPreds: null,
  freeRunPreds: null,
  activations: null,
  trained: false,
  trainLen: 500,
  washout: 100,
};

const TOTAL_LEN = 1200;
const FREERUN_STEPS = 400;
const ACT_SAMPLE = 20;

function getConfig(): ESNConfig {
  return {
    N: parseInt((document.getElementById('res-size') as HTMLInputElement).value),
    spectralRadius: parseFloat((document.getElementById('spectral-radius') as HTMLInputElement).value),
    inputScale: parseFloat((document.getElementById('input-scale') as HTMLInputElement).value),
    leak: parseFloat((document.getElementById('leak-rate') as HTMLInputElement).value),
    ridge: parseFloat((document.getElementById('ridge') as HTMLInputElement).value),
    sparsity: 0.1,
  };
}

function getSignalName(): SignalName {
  return (document.getElementById('signal-sel') as HTMLSelectElement).value as SignalName;
}

function rmse(a: Float64Array, b: Float64Array, start = 0, end?: number): number {
  const e = end ?? Math.min(a.length, b.length);
  let s = 0, n = 0;
  for (let i = start; i < e; i++) { const d = a[i] - b[i]; s += d * d; n++; }
  return n > 0 ? Math.sqrt(s / n) : 0;
}

// ── Canvas drawing ────────────────────────────────────────────────────────────
function drawSignalPlot(): void {
  const canvas = document.getElementById('signal-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0a14';
  ctx.fillRect(0, 0, W, H);

  if (!app.signal) return;

  const sig = app.signal;
  const trainLen = app.trainLen;
  const washout = app.washout;

  // x scale: total points to show
  const showTrain = trainLen - washout;
  const showFree = FREERUN_STEPS;
  const totalShow = showTrain + showFree;

  const xScale = (W - 60) / totalShow;
  const yMid = H / 2;
  const yScale = (H - 40) / 2.2;

  const toX = (i: number) => 40 + i * xScale;
  const toY = (v: number) => yMid - v * yScale;

  // Background regions
  ctx.fillStyle = 'rgba(56,189,248,0.04)';
  ctx.fillRect(40, 10, showTrain * xScale, H - 20);
  ctx.fillStyle = 'rgba(251,113,133,0.06)';
  ctx.fillRect(40 + showTrain * xScale, 10, showFree * xScale, H - 20);

  // Divider
  ctx.strokeStyle = '#666';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(40 + showTrain * xScale, 10);
  ctx.lineTo(40 + showTrain * xScale, H - 10);
  ctx.stroke();
  ctx.setLineDash([]);

  // Labels
  ctx.fillStyle = '#38bdf8';
  ctx.font = '11px monospace';
  ctx.fillText('Training', 50, 22);
  ctx.fillStyle = '#fb7185';
  ctx.fillText('Free-run', 40 + showTrain * xScale + 8, 22);

  // Target signal (training region)
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < showTrain; i++) {
    const v = sig[i + washout + 1];
    if (i === 0) ctx.moveTo(toX(i), toY(v));
    else ctx.lineTo(toX(i), toY(v));
  }
  ctx.stroke();

  // Target signal (free-run region if available)
  if (app.freeRunPreds) {
    // ground truth continuation
    ctx.strokeStyle = 'rgba(56,189,248,0.35)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < showFree; i++) {
      const v = sig[trainLen + i + 1] ?? 0;
      if (i === 0) ctx.moveTo(toX(showTrain + i), toY(v));
      else ctx.lineTo(toX(showTrain + i), toY(v));
    }
    ctx.stroke();
  }

  // Train predictions
  if (app.trainPreds) {
    ctx.strokeStyle = '#facc15';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < Math.min(showTrain, app.trainPreds.length); i++) {
      const v = app.trainPreds[i];
      if (i === 0) ctx.moveTo(toX(i), toY(v));
      else ctx.lineTo(toX(i), toY(v));
    }
    ctx.stroke();
  }

  // Free-run predictions
  if (app.freeRunPreds) {
    ctx.strokeStyle = '#fb7185';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < app.freeRunPreds.length; i++) {
      const v = app.freeRunPreds[i];
      if (i === 0) ctx.moveTo(toX(showTrain + i), toY(v));
      else ctx.lineTo(toX(showTrain + i), toY(v));
    }
    ctx.stroke();
  }

  // Y axis
  ctx.strokeStyle = '#334';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(40, 10); ctx.lineTo(40, H - 10); ctx.stroke();
  ctx.fillStyle = '#888';
  ctx.font = '10px monospace';
  ctx.fillText('1', 24, toY(1) + 4);
  ctx.fillText('0', 28, toY(0) + 4);
  ctx.fillText('-1', 20, toY(-1) + 4);
  ctx.beginPath(); ctx.moveTo(36, toY(1)); ctx.lineTo(40, toY(1)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(36, toY(0)); ctx.lineTo(40, toY(0)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(36, toY(-1)); ctx.lineTo(40, toY(-1)); ctx.stroke();

  // Legend
  const lx = W - 200;
  ctx.fillStyle = '#38bdf8'; ctx.fillRect(lx, H - 42, 18, 3);
  ctx.fillStyle = '#ccc'; ctx.font = '11px sans-serif'; ctx.fillText('Target', lx + 22, H - 35);
  ctx.fillStyle = '#facc15'; ctx.fillRect(lx, H - 28, 18, 3);
  ctx.fillStyle = '#ccc'; ctx.fillText('Train pred', lx + 22, H - 21);
  ctx.fillStyle = '#fb7185'; ctx.fillRect(lx + 100, H - 42, 18, 3);
  ctx.fillStyle = '#ccc'; ctx.fillText('Free-run', lx + 122, H - 35);
}

function drawActivationPlot(): void {
  const canvas = document.getElementById('act-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0a14';
  ctx.fillRect(0, 0, W, H);

  if (!app.activations || app.activations.length === 0) {
    ctx.fillStyle = '#555';
    ctx.font = '13px sans-serif';
    ctx.fillText('Train to see reservoir activations', W / 2 - 130, H / 2);
    return;
  }

  const acts = app.activations;
  const nNeurons = acts.length;
  const T = acts[0].length;
  const rowH = (H - 20) / nNeurons;
  const xScale = (W - 40) / T;

  // Heatmap: each neuron is a row, time is x, activation is colour
  for (let k = 0; k < nNeurons; k++) {
    const y0 = 10 + k * rowH;
    for (let t = 0; t < T; t++) {
      const v = Math.max(-1, Math.min(1, acts[k][t]));
      const x = 40 + t * xScale;
      // colour: negative = blue, positive = orange
      let r, g, b;
      if (v >= 0) { r = Math.round(251 * v + 20 * (1 - v)); g = Math.round(80 * v + 20 * (1 - v)); b = Math.round(10 * v + 20 * (1 - v)); }
      else { const u = -v; r = Math.round(20 * (1 - u)); g = Math.round(80 * (1 - u)); b = Math.round(200 * u + 20 * (1 - u)); }
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x, y0, Math.ceil(xScale) + 1, Math.ceil(rowH) + 1);
    }
  }

  ctx.fillStyle = '#888';
  ctx.font = '10px monospace';
  ctx.fillText('Neuron', 2, H / 2);
  ctx.fillStyle = '#555';
  ctx.font = '10px monospace';
  ctx.fillText('← time →', 40, H - 2);

  ctx.fillStyle = '#555';
  ctx.font = '11px sans-serif';
  ctx.fillText(`${nNeurons} reservoir neurons  ·  training window`, 50, 14);
}

function drawWeightsPlot(): void {
  const canvas = document.getElementById('wout-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0a14';
  ctx.fillRect(0, 0, W, H);

  if (!app.esn || !app.esn.Wout) {
    ctx.fillStyle = '#555';
    ctx.font = '13px sans-serif';
    ctx.fillText('Train to see readout weights', W / 2 - 110, H / 2);
    return;
  }

  const w = app.esn.Wout;
  const N = w.length;
  let mx = 0;
  for (let i = 0; i < N; i++) mx = Math.max(mx, Math.abs(w[i]));
  if (mx === 0) mx = 1;

  const barW = Math.max(1, (W - 40) / N);
  const yMid = H / 2;
  const yScale = (H - 30) / 2;

  for (let i = 0; i < N; i++) {
    const v = w[i] / mx;
    const x = 40 + i * barW;
    const barH = Math.abs(v) * yScale;
    ctx.fillStyle = v >= 0 ? 'rgba(56,189,248,0.7)' : 'rgba(251,113,133,0.7)';
    ctx.fillRect(x, v >= 0 ? yMid - barH : yMid, barW - 0.5, barH);
  }

  ctx.strokeStyle = '#334';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(40, yMid); ctx.lineTo(W, yMid); ctx.stroke();
  ctx.fillStyle = '#777';
  ctx.font = '11px monospace';
  ctx.fillText(`Readout weights (N=${N})`, 44, 14);
}

function updateStats(): void {
  const elTrain = document.getElementById('stat-train')!;
  const elTest = document.getElementById('stat-test')!;
  if (!app.signal || !app.trainPreds) { elTrain.textContent = '—'; elTest.textContent = '—'; return; }

  const washout = app.washout;
  const trainLen = app.trainLen;
  // Training RMSE
  const target = app.signal.slice(washout + 1, trainLen + 1);
  const tErr = rmse(target, app.trainPreds, 0, Math.min(target.length, app.trainPreds.length));
  elTrain.textContent = tErr.toFixed(5);

  if (app.freeRunPreds) {
    const freeTarget = app.signal.slice(trainLen + 1, trainLen + 1 + FREERUN_STEPS);
    const fErr = rmse(freeTarget, app.freeRunPreds, 0, Math.min(freeTarget.length, app.freeRunPreds.length));
    elTest.textContent = fErr.toFixed(5);
  } else {
    elTest.textContent = '—';
  }
}

function redrawAll(): void {
  drawSignalPlot();
  drawActivationPlot();
  drawWeightsPlot();
  updateStats();
}

// ── Button handlers ───────────────────────────────────────────────────────────
function handleReset(): void {
  const cfg = getConfig();
  app.esn = new ESN(cfg);
  app.signal = generateSignal(getSignalName(), TOTAL_LEN);
  app.trainPreds = null;
  app.freeRunPreds = null;
  app.activations = null;
  app.trained = false;
  (document.getElementById('btn-freerun') as HTMLButtonElement).disabled = true;
  redrawAll();
  setStatus('Reservoir reset. Click "Train" to train readout weights.');
}

function handleTrain(): void {
  if (!app.esn || !app.signal) { handleReset(); return; }
  const washout = app.washout;
  const trainLen = app.trainLen;
  const trainSig = app.signal.slice(0, trainLen);

  setStatus('Training…');
  setTimeout(() => {
    // train
    app.esn!.train(trainSig, washout);
    // teacher-forced preds
    app.trainPreds = app.esn!.predictTeacherForced(trainSig, washout);
    // activations
    app.activations = app.esn!.getActivations(trainSig, washout, ACT_SAMPLE);
    app.trained = true;
    (document.getElementById('btn-freerun') as HTMLButtonElement).disabled = false;
    app.freeRunPreds = null;
    redrawAll();
    setStatus('Trained! Train RMSE updated. Click "Free-run" to generate.');
  }, 10);
}

function handleFreeRun(): void {
  if (!app.esn || !app.signal || !app.trained) return;
  setStatus('Free-running…');
  setTimeout(() => {
    // seed from last training step state (esn state is at end of training after getActivations)
    // re-run teacher forcing to get state at trainLen
    const trainLen = app.trainLen;
    const washout = app.washout;
    const trainSig = app.signal!.slice(0, trainLen);
    app.esn!.reset();
    const { leak } = app.esn!.cfg;
    for (let t = 0; t < trainSig.length - 1; t++) {
      esnStep(app.esn!.state, app.esn!.Wr, app.esn!.Win, trainSig[t], leak);
    }
    const seed = trainSig[trainSig.length - 1];
    app.freeRunPreds = app.esn!.freeRun(seed, FREERUN_STEPS);
    redrawAll();
    setStatus('Free-run complete.');
  }, 10);
}

function setStatus(msg: string): void {
  const el = document.getElementById('status');
  if (el) el.textContent = msg;
}

// ── Slider label sync ─────────────────────────────────────────────────────────
function bindSlider(id: string, labelId: string, fmt?: (v: number) => string): void {
  const el = document.getElementById(id) as HTMLInputElement;
  const lbl = document.getElementById(labelId)!;
  const update = () => {
    const v = parseFloat(el.value);
    lbl.textContent = fmt ? fmt(v) : String(v);
  };
  el.addEventListener('input', update);
  update();
}

// ── Init ──────────────────────────────────────────────────────────────────────
function init(): void {
  bindSlider('res-size', 'lbl-res-size', v => String(Math.round(v)));
  bindSlider('spectral-radius', 'lbl-sr');
  bindSlider('input-scale', 'lbl-is');
  bindSlider('leak-rate', 'lbl-lr');
  bindSlider('ridge', 'lbl-ridge', v => v.toExponential(0));

  document.getElementById('btn-reset')!.addEventListener('click', handleReset);
  document.getElementById('btn-train')!.addEventListener('click', handleTrain);
  document.getElementById('btn-freerun')!.addEventListener('click', handleFreeRun);
  document.getElementById('signal-sel')!.addEventListener('change', handleReset);

  handleReset();
}

window.addEventListener('DOMContentLoaded', init);
