import * as d3 from 'd3';

// ---------------------------------------------------------------------------
// Tiny neural-net primitives (no ML library)
// ---------------------------------------------------------------------------

type Matrix = number[][];

function zeros(rows: number, cols: number): Matrix {
  return Array.from({length: rows}, () => new Array(cols).fill(0));
}

function randn(): number {
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

interface Layer {
  W: Matrix;
  b: number[];
  // Adam state
  mW: Matrix;
  vW: Matrix;
  mb: number[];
  vb: number[];
  dW: Matrix;
  db: number[];
}

function makeLayer(inDim: number, outDim: number): Layer {
  const scale = Math.sqrt(2 / inDim);
  const W = Array.from({length: outDim}, () =>
    Array.from({length: inDim}, () => randn() * scale)
  );
  const b = new Array(outDim).fill(0);
  return {
    W, b,
    mW: zeros(outDim, inDim), vW: zeros(outDim, inDim),
    mb: new Array(outDim).fill(0), vb: new Array(outDim).fill(0),
    dW: zeros(outDim, inDim), db: new Array(outDim).fill(0),
  };
}

function linearForward(layer: Layer, x: number[]): number[] {
  return layer.W.map((row, i) =>
    row.reduce((s, w, j) => s + w * x[j], 0) + layer.b[i]
  );
}

function relu(x: number[]): number[] {
  return x.map(v => Math.max(0, v));
}

function reluGrad(x: number[]): number[] {
  return x.map(v => (v > 0 ? 1 : 0));
}

// ---------------------------------------------------------------------------
// MLP: forward + backward (single sample)
// ---------------------------------------------------------------------------

interface MLP {
  layers: Layer[];
  t: number; // Adam step counter
}

function makeMLP(inDim: number, hiddenSize: number, outDim: number, numHidden: number): MLP {
  const layers: Layer[] = [];
  let d = inDim;
  for (let i = 0; i < numHidden; i++) {
    layers.push(makeLayer(d, hiddenSize));
    d = hiddenSize;
  }
  layers.push(makeLayer(d, outDim));
  return {layers, t: 0};
}

function resetGrads(mlp: MLP): void {
  for (const l of mlp.layers) {
    l.dW = zeros(l.W.length, l.W[0].length);
    l.db = new Array(l.b.length).fill(0);
  }
}

interface FwdResult {
  preActs: number[][];  // pre-activation (linear output), length = layers
  acts: number[][];     // post-activation, length = layers (last is output)
}

// Forward pass: hidden layers use ReLU, output is linear
function forward(mlp: MLP, x: number[]): FwdResult {
  const preActs: number[][] = [];
  const acts: number[][] = [];
  let cur = x;
  for (let i = 0; i < mlp.layers.length; i++) {
    const pre = linearForward(mlp.layers[i], cur);
    preActs.push(pre);
    if (i < mlp.layers.length - 1) {
      cur = relu(pre);
    } else {
      cur = pre; // linear output
    }
    acts.push(cur);
  }
  return {preActs, acts};
}

// Backward pass accumulates grads into layer.dW, layer.db
// dLoss_dOut is gradient wrt output (shape = outDim)
function backward(mlp: MLP, x: number[], fwd: FwdResult, dOut: number[]): void {
  let delta = dOut;
  for (let i = mlp.layers.length - 1; i >= 0; i--) {
    const layer = mlp.layers[i];
    // gradient through activation (ReLU) for hidden, pass-through for output
    if (i < mlp.layers.length - 1) {
      delta = delta.map((d, k) => d * reluGrad(fwd.preActs[i])[k]);
    }
    // input to this layer
    const inp = i === 0 ? x : fwd.acts[i - 1];
    // accumulate weight grads
    for (let r = 0; r < layer.W.length; r++) {
      for (let c = 0; c < layer.W[r].length; c++) {
        layer.dW[r][c] += delta[r] * inp[c];
      }
      layer.db[r] += delta[r];
    }
    // propagate gradient backward
    if (i > 0) {
      const newDelta = new Array(layer.W[0].length).fill(0);
      for (let c = 0; c < layer.W[0].length; c++) {
        for (let r = 0; r < layer.W.length; r++) {
          newDelta[c] += layer.W[r][c] * delta[r];
        }
      }
      delta = newDelta;
    }
  }
}

// Adam update
function adamStep(mlp: MLP, lr: number, beta1 = 0.9, beta2 = 0.999, eps = 1e-8): void {
  mlp.t += 1;
  const t = mlp.t;
  const bc1 = 1 - Math.pow(beta1, t);
  const bc2 = 1 - Math.pow(beta2, t);
  for (const layer of mlp.layers) {
    for (let r = 0; r < layer.W.length; r++) {
      for (let c = 0; c < layer.W[r].length; c++) {
        layer.mW[r][c] = beta1 * layer.mW[r][c] + (1 - beta1) * layer.dW[r][c];
        layer.vW[r][c] = beta2 * layer.vW[r][c] + (1 - beta2) * layer.dW[r][c] * layer.dW[r][c];
        const mHat = layer.mW[r][c] / bc1;
        const vHat = layer.vW[r][c] / bc2;
        layer.W[r][c] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
      layer.mb[r] = beta1 * layer.mb[r] + (1 - beta1) * layer.db[r];
      layer.vb[r] = beta2 * layer.vb[r] + (1 - beta2) * layer.db[r] * layer.db[r];
      const mHat = layer.mb[r] / bc1;
      const vHat = layer.vb[r] / bc2;
      layer.b[r] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }
}

// ---------------------------------------------------------------------------
// Timestep sinusoidal embedding
// ---------------------------------------------------------------------------

function sinEmbedding(t: number, dim: number): number[] {
  // fourier features for timestep t in [0,1]
  const half = dim >> 1;
  const emb: number[] = [];
  for (let i = 0; i < half; i++) {
    const freq = Math.pow(10000, -i / half);
    emb.push(Math.sin(t * freq * Math.PI * 2));
    emb.push(Math.cos(t * freq * Math.PI * 2));
  }
  return emb.slice(0, dim);
}

// ---------------------------------------------------------------------------
// DDPM forward process: variance schedule
// ---------------------------------------------------------------------------

interface DiffusionSchedule {
  T: number;
  betas: number[];
  alphas: number[];
  alphaBar: number[]; // cumulative product
}

function makeSchedule(T: number, betaStart = 0.0001, betaEnd = 0.02): DiffusionSchedule {
  const betas: number[] = [];
  const alphas: number[] = [];
  const alphaBar: number[] = [];
  let cumAlpha = 1;
  for (let i = 0; i < T; i++) {
    const beta = betaStart + (betaEnd - betaStart) * i / (T - 1);
    betas.push(beta);
    const alpha = 1 - beta;
    alphas.push(alpha);
    cumAlpha *= alpha;
    alphaBar.push(cumAlpha);
  }
  return {T, betas, alphas, alphaBar};
}

// q(x_t | x_0) = sqrt(alphabar_t)*x0 + sqrt(1-alphabar_t)*eps
function qSample(x0: number[], t: number, sched: DiffusionSchedule): {xt: number[], eps: number[]} {
  const ab = sched.alphaBar[t];
  const sqrtAb = Math.sqrt(ab);
  const sqrtOneMinusAb = Math.sqrt(1 - ab);
  const eps = [randn(), randn()];
  const xt = x0.map((v, i) => sqrtAb * v + sqrtOneMinusAb * eps[i]);
  return {xt, eps};
}

// ---------------------------------------------------------------------------
// 2D Distributions
// ---------------------------------------------------------------------------

type DistName = 'ring' | 'two-moons' | 'spiral' | '8-gaussians' | 's-curve';

function sampleDist(name: DistName, n: number): number[][] {
  const pts: number[][] = [];
  if (name === 'ring') {
    for (let i = 0; i < n; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const r = 0.9 + randn() * 0.08;
      pts.push([r * Math.cos(angle), r * Math.sin(angle)]);
    }
  } else if (name === 'two-moons') {
    for (let i = 0; i < n; i++) {
      const upper = Math.random() < 0.5;
      const angle = Math.PI * Math.random();
      const r = 0.85 + randn() * 0.07;
      if (upper) {
        pts.push([r * Math.cos(angle), r * Math.sin(angle)]);
      } else {
        pts.push([r * Math.cos(angle + Math.PI) + 0.05, r * Math.sin(angle + Math.PI) + 0.45]);
      }
    }
  } else if (name === 'spiral') {
    for (let i = 0; i < n; i++) {
      const arm = Math.floor(Math.random() * 2);
      const frac = Math.random();
      const angle = frac * 3 * Math.PI + arm * Math.PI;
      const r = 0.1 + 0.8 * frac;
      pts.push([r * Math.cos(angle) + randn() * 0.04, r * Math.sin(angle) + randn() * 0.04]);
    }
  } else if (name === '8-gaussians') {
    const centers = [
      [0.85, 0], [-0.85, 0], [0, 0.85], [0, -0.85],
      [0.6, 0.6], [-0.6, 0.6], [0.6, -0.6], [-0.6, -0.6],
    ];
    for (let i = 0; i < n; i++) {
      const c = centers[Math.floor(Math.random() * centers.length)];
      pts.push([c[0] + randn() * 0.1, c[1] + randn() * 0.1]);
    }
  } else if (name === 's-curve') {
    for (let i = 0; i < n; i++) {
      const t = Math.random() * 2 * Math.PI;
      const x = Math.sin(t) * 0.9 + randn() * 0.06;
      const y = (Math.cos(t) * 0.45 + (t > Math.PI ? 0.45 : -0.45)) * 0.9 + randn() * 0.06;
      pts.push([x, y]);
    }
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Denoiser network input: [x_t (2), t_emb (embDim)]
// Denoiser output: predicted noise eps_theta (2)
// ---------------------------------------------------------------------------

const T_EMB_DIM = 16;

function makeDenoiser(hiddenSize: number): MLP {
  return makeMLP(2 + T_EMB_DIM, hiddenSize, 2, 2);
}

function predictNoise(mlp: MLP, xt: number[], tNorm: number): number[] {
  const emb = sinEmbedding(tNorm, T_EMB_DIM);
  const inp = xt.concat(emb);
  const fwd = forward(mlp, inp);
  return fwd.acts[fwd.acts.length - 1];
}

function predictNoiseFwd(mlp: MLP, xt: number[], tNorm: number): {pred: number[], fwd: FwdResult, inp: number[]} {
  const emb = sinEmbedding(tNorm, T_EMB_DIM);
  const inp = xt.concat(emb);
  const fwd = forward(mlp, inp);
  const pred = fwd.acts[fwd.acts.length - 1];
  return {pred, fwd, inp};
}

// ---------------------------------------------------------------------------
// Training step: MSE on predicted noise
// ---------------------------------------------------------------------------

function trainStep(
  mlp: MLP,
  data: number[][],
  sched: DiffusionSchedule,
  lr: number,
  batchSize: number
): number {
  resetGrads(mlp);
  let totalLoss = 0;
  for (let b = 0; b < batchSize; b++) {
    const x0 = data[Math.floor(Math.random() * data.length)];
    const t = Math.floor(Math.random() * sched.T);
    const {xt, eps} = qSample(x0, t, sched);
    const tNorm = t / (sched.T - 1);
    const {pred, fwd, inp} = predictNoiseFwd(mlp, xt, tNorm);
    // MSE loss: mean((pred - eps)^2) / 2
    const dOut = pred.map((p, i) => (p - eps[i]) / batchSize);
    totalLoss += pred.reduce((s, p, i) => s + (p - eps[i]) * (p - eps[i]), 0);
    backward(mlp, inp, fwd, dOut);
  }
  adamStep(mlp, lr);
  return totalLoss / (batchSize * 2);
}

// ---------------------------------------------------------------------------
// Reverse process sampling (DDPM ancestral sampling)
// ---------------------------------------------------------------------------

function reverseSample(
  mlp: MLP,
  sched: DiffusionSchedule,
  n: number
): number[][][] {
  // returns trajectory: T+1 snapshots of n points
  // trajectory[0] = pure noise, trajectory[T] = denoised
  const traj: number[][][] = [];
  // init: pure Gaussian noise
  let pts = Array.from({length: n}, () => [randn(), randn()]);
  traj.push(pts.map(p => [p[0], p[1]]));

  for (let t = sched.T - 1; t >= 0; t--) {
    const tNorm = t / (sched.T - 1);
    const beta = sched.betas[t];
    const alpha = sched.alphas[t];
    const alphaBar = sched.alphaBar[t];
    const alphaBarPrev = t > 0 ? sched.alphaBar[t - 1] : 1;

    pts = pts.map(xt => {
      const epsTheta = predictNoise(mlp, xt, tNorm);
      // DDPM mean: 1/sqrt(alpha) * (x_t - beta/sqrt(1-alphabar) * eps_theta)
      const coeff = beta / Math.sqrt(1 - alphaBar);
      const mean = xt.map((v, i) => (v - coeff * epsTheta[i]) / Math.sqrt(alpha));
      if (t === 0) {
        return mean;
      }
      // posterior variance: beta * (1-alphabar_prev)/(1-alphabar)
      const variance = beta * (1 - alphaBarPrev) / (1 - alphaBar);
      const sigma = Math.sqrt(variance);
      return mean.map(v => v + sigma * randn());
    });
    traj.push(pts.map(p => [p[0], p[1]]));
  }
  return traj;
}

// ---------------------------------------------------------------------------
// App state & UI wiring
// ---------------------------------------------------------------------------

interface AppState {
  dist: DistName;
  T: number;
  hiddenSize: number;
  lr: number;
  batchSize: number;
  data: number[][];
  sched: DiffusionSchedule;
  denoiser: MLP;
  running: boolean;
  step: number;
  losses: number[];
  // forward process scrub
  fwdT: number;
  // sampling animation
  sampling: boolean;
  sampleTraj: number[][][];
  sampleFrame: number;
  sampleAnimId: number | null;
}

const NUM_DATA = 300;
const SAMPLE_N = 150;

let state: AppState;

function initState(): AppState {
  const dist: DistName = 'ring';
  const T = 40;
  const hiddenSize = 32;
  const data = sampleDist(dist, NUM_DATA);
  const sched = makeSchedule(T);
  return {
    dist, T, hiddenSize, lr: 0.005, batchSize: 32,
    data, sched,
    denoiser: makeDenoiser(hiddenSize),
    running: false, step: 0, losses: [],
    fwdT: 0,
    sampling: false, sampleTraj: [], sampleFrame: 0, sampleAnimId: null,
  };
}

// ---------------------------------------------------------------------------
// Canvas rendering
// ---------------------------------------------------------------------------

const SCALE = 160;
const OFFSET_X = 200;
const OFFSET_Y = 200;
const CANVAS_W = 400;
const CANVAS_H = 400;
const LOSS_W = 400;
const LOSS_H = 120;

function ptToCanvas(p: number[]): [number, number] {
  return [p[0] * SCALE + OFFSET_X, -p[1] * SCALE + OFFSET_Y];
}

function drawMain(canvas: HTMLCanvasElement, st: AppState): void {
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

  // background
  ctx.fillStyle = '#0d0d1f';
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  // grid lines
  ctx.strokeStyle = '#1a1a3a';
  ctx.lineWidth = 1;
  for (let g = -1; g <= 1; g += 0.5) {
    ctx.beginPath();
    const [gx, gy] = ptToCanvas([g, 0]);
    ctx.moveTo(gx, 0); ctx.lineTo(gx, CANVAS_H);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, gy); ctx.lineTo(CANVAS_W, gy);
    ctx.stroke();
  }

  // mode: show sample trajectory or forward noising
  if (st.sampling && st.sampleTraj.length > 0) {
    const frame = Math.min(st.sampleFrame, st.sampleTraj.length - 1);
    const pts = st.sampleTraj[frame];
    const progress = frame / (st.sampleTraj.length - 1);
    const r = Math.round(100 + 155 * (1 - progress));
    const g = Math.round(200 * progress);
    const b = Math.round(255 * progress);
    const color = `rgba(${r},${g},${b},0.75)`;
    pts.forEach(p => {
      const [cx, cy] = ptToCanvas(p);
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    });
    // real data faint
    st.data.forEach(p => {
      const [cx, cy] = ptToCanvas(p);
      ctx.beginPath();
      ctx.arc(cx, cy, 2, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(100,220,100,0.18)';
      ctx.fill();
    });
  } else {
    // forward noising view: show data noised to fwdT
    const ab = st.sched.alphaBar[st.fwdT];
    const sqrtAb = Math.sqrt(ab);
    const sqrtOneMinusAb = Math.sqrt(1 - ab);
    st.data.forEach(x0 => {
      const xt = [
        sqrtAb * x0[0] + sqrtOneMinusAb * randn(),
        sqrtAb * x0[1] + sqrtOneMinusAb * randn(),
      ];
      const [cx, cy] = ptToCanvas(xt);
      const noiseLevel = 1 - ab;
      const rr = Math.round(100 + 155 * noiseLevel);
      const gb = Math.round(220 * (1 - noiseLevel));
      ctx.beginPath();
      ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(${rr},${gb},${gb},0.7)`;
      ctx.fill();
    });
  }
}

function drawLoss(canvas: HTMLCanvasElement, losses: number[]): void {
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, LOSS_W, LOSS_H);
  ctx.fillStyle = '#0d0d1f';
  ctx.fillRect(0, 0, LOSS_W, LOSS_H);

  if (losses.length < 2) return;

  const maxLoss = Math.max(...losses.slice(0, 50)); // avoid log scale issues
  const minLoss = Math.min(...losses);
  const range = Math.max(maxLoss - minLoss, 1e-6);
  const pad = 20;
  const w = LOSS_W - 2 * pad;
  const h = LOSS_H - 2 * pad;

  // axes
  ctx.strokeStyle = '#2a2a4a';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad); ctx.lineTo(pad, pad + h);
  ctx.lineTo(pad + w, pad + h);
  ctx.stroke();

  // loss curve
  ctx.strokeStyle = '#a78bfa';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const step = Math.max(1, Math.floor(losses.length / w));
  losses.forEach((l, i) => {
    const x = pad + (i / (losses.length - 1)) * w;
    const y = pad + h - ((l - minLoss) / range) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // label
  ctx.fillStyle = '#6b7280';
  ctx.font = '10px monospace';
  ctx.fillText(`loss: ${losses[losses.length - 1].toFixed(4)}`, pad + 4, pad + 12);
}

// ---------------------------------------------------------------------------
// Main app
// ---------------------------------------------------------------------------

function main(): void {
  state = initState();

  // Get DOM elements
  const mainCanvas = document.getElementById('mainCanvas') as HTMLCanvasElement;
  const lossCanvas = document.getElementById('lossCanvas') as HTMLCanvasElement;
  mainCanvas.width = CANVAS_W;
  mainCanvas.height = CANVAS_H;
  lossCanvas.width = LOSS_W;
  lossCanvas.height = LOSS_H;

  const distSelect = document.getElementById('distSelect') as HTMLSelectElement;
  const lrInput = document.getElementById('lrInput') as HTMLInputElement;
  const lrVal = document.getElementById('lrVal') as HTMLSpanElement;
  const hiddenInput = document.getElementById('hiddenInput') as HTMLInputElement;
  const hiddenVal = document.getElementById('hiddenVal') as HTMLSpanElement;
  const tInput = document.getElementById('tInput') as HTMLInputElement;
  const tVal = document.getElementById('tVal') as HTMLSpanElement;
  const fwdSlider = document.getElementById('fwdSlider') as HTMLInputElement;
  const fwdTVal = document.getElementById('fwdTVal') as HTMLSpanElement;
  const playPause = document.getElementById('playPause') as HTMLButtonElement;
  const stepBtn = document.getElementById('stepBtn') as HTMLButtonElement;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
  const sampleBtn = document.getElementById('sampleBtn') as HTMLButtonElement;
  const stepCount = document.getElementById('stepCount') as HTMLSpanElement;
  const lossVal = document.getElementById('lossVal') as HTMLSpanElement;
  const sampleStatus = document.getElementById('sampleStatus') as HTMLSpanElement;

  function syncFwdSlider(): void {
    fwdSlider.max = String(state.T - 1);
    fwdSlider.value = String(state.fwdT);
    fwdTVal.textContent = String(state.fwdT);
  }

  function reset(): void {
    state.running = false;
    playPause.textContent = 'Play';
    state.step = 0;
    state.losses = [];
    state.data = sampleDist(state.dist, NUM_DATA);
    state.sched = makeSchedule(state.T);
    state.denoiser = makeDenoiser(state.hiddenSize);
    state.fwdT = 0;
    state.sampling = false;
    state.sampleTraj = [];
    state.sampleFrame = 0;
    if (state.sampleAnimId !== null) {
      cancelAnimationFrame(state.sampleAnimId);
      state.sampleAnimId = null;
    }
    syncFwdSlider();
    stepCount.textContent = '0';
    lossVal.textContent = '—';
    sampleStatus.textContent = '';
    draw();
  }

  function draw(): void {
    drawMain(mainCanvas, state);
    drawLoss(lossCanvas, state.losses);
  }

  function doTrainStep(): void {
    const loss = trainStep(state.denoiser, state.data, state.sched, state.lr, state.batchSize);
    state.step++;
    state.losses.push(loss);
    // Keep losses array manageable
    if (state.losses.length > 500) state.losses.splice(0, 1);
    stepCount.textContent = String(state.step);
    lossVal.textContent = loss.toFixed(5);
    if (!state.sampling) draw();
  }

  // Training loop
  let trainRafId: number | null = null;
  let lastTrainTime = 0;
  const TRAIN_INTERVAL = 50; // ms between training renders

  function trainLoop(now: number): void {
    if (!state.running) return;
    // Do multiple steps per frame for speed
    const elapsed = now - lastTrainTime;
    const stepsPerFrame = 5;
    for (let i = 0; i < stepsPerFrame; i++) {
      doTrainStep();
    }
    if (elapsed > TRAIN_INTERVAL) {
      draw();
      lastTrainTime = now;
    }
    trainRafId = requestAnimationFrame(trainLoop);
  }

  // Sampling animation
  function startSampleAnim(): void {
    if (state.sampleAnimId !== null) {
      cancelAnimationFrame(state.sampleAnimId);
      state.sampleAnimId = null;
    }
    state.sampling = true;
    state.sampleFrame = 0;
    sampleStatus.textContent = 'Denoising…';

    function animFrame(): void {
      if (!state.sampling) return;
      state.sampleFrame++;
      if (state.sampleFrame >= state.sampleTraj.length) {
        state.sampleFrame = state.sampleTraj.length - 1;
        sampleStatus.textContent = `Done (${state.sampleTraj.length - 1} reverse steps)`;
        draw();
        state.sampleAnimId = null;
        return;
      }
      draw();
      state.sampleAnimId = requestAnimationFrame(animFrame);
    }
    state.sampleAnimId = requestAnimationFrame(animFrame);
  }

  // Controls
  playPause.addEventListener('click', () => {
    state.running = !state.running;
    playPause.textContent = state.running ? 'Pause' : 'Play';
    if (state.running) {
      lastTrainTime = 0;
      trainRafId = requestAnimationFrame(trainLoop);
    } else {
      if (trainRafId !== null) {
        cancelAnimationFrame(trainRafId);
        trainRafId = null;
      }
      draw();
    }
  });

  stepBtn.addEventListener('click', () => {
    doTrainStep();
    draw();
  });

  resetBtn.addEventListener('click', reset);

  sampleBtn.addEventListener('click', () => {
    if (state.step === 0) {
      sampleStatus.textContent = 'Train first!';
      return;
    }
    const traj = reverseSample(state.denoiser, state.sched, SAMPLE_N);
    state.sampleTraj = traj;
    state.sampleFrame = 0;
    startSampleAnim();
  });

  distSelect.addEventListener('change', () => {
    state.dist = distSelect.value as DistName;
    reset();
  });

  lrInput.addEventListener('input', () => {
    state.lr = parseFloat(lrInput.value);
    lrVal.textContent = state.lr.toFixed(4);
  });

  hiddenInput.addEventListener('change', () => {
    state.hiddenSize = parseInt(hiddenInput.value, 10);
    hiddenVal.textContent = String(state.hiddenSize);
    reset();
  });

  tInput.addEventListener('change', () => {
    state.T = parseInt(tInput.value, 10);
    tVal.textContent = String(state.T);
    reset();
  });

  fwdSlider.addEventListener('input', () => {
    state.fwdT = parseInt(fwdSlider.value, 10);
    fwdTVal.textContent = String(state.fwdT);
    if (!state.sampling) draw();
  });

  syncFwdSlider();
  draw();
}

document.addEventListener('DOMContentLoaded', main);
