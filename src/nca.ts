export {};

// Neural Cellular Automata
// Grow a target pattern from a single seed via a shared tiny MLP update rule.

const CELL_SIZE = 8;
const GRID_W = 28;
const GRID_H = 28;
const N_CHANNELS = 12;  // first 4 are RGBA visible
const PERCEPTION_CHANNELS = N_CHANNELS * 3; // identity + sobel_x + sobel_y
const HIDDEN = 64;

// Adam optimizer state
interface AdamState {
  m: Float32Array;
  v: Float32Array;
  t: number;
}

// ---------- helpers ----------
function zeros(n: number): Float32Array { return new Float32Array(n); }
function randn(): number { let u=0,v=0; while(!u) u=Math.random(); while(!v) v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }

// ---------- MLP weights ----------
// W1: [HIDDEN x PERCEPTION_CHANNELS], b1: [HIDDEN]
// W2: [N_CHANNELS x HIDDEN], b2: [N_CHANNELS] (zero-init)
function initWeights(nChannels: number, hidden: number, perceptionC: number) {
  const scale = Math.sqrt(2 / perceptionC);
  const W1 = new Float32Array(hidden * perceptionC).map(() => randn() * scale);
  const b1 = zeros(hidden);
  const W2 = zeros(nChannels * hidden);
  const b2 = zeros(nChannels);
  return { W1, b1, W2, b2 };
}

function initAdam(weights: ReturnType<typeof initWeights>): AdamState {
  const total = weights.W1.length + weights.b1.length + weights.W2.length + weights.b2.length;
  return { m: zeros(total), v: zeros(total), t: 0 };
}

function flatWeights(w: ReturnType<typeof initWeights>): Float32Array {
  const out = new Float32Array(w.W1.length + w.b1.length + w.W2.length + w.b2.length);
  out.set(w.W1, 0);
  out.set(w.b1, w.W1.length);
  out.set(w.W2, w.W1.length + w.b1.length);
  out.set(w.b2, w.W1.length + w.b1.length + w.W2.length);
  return out;
}

function unflatWeights(flat: Float32Array, nChannels: number, hidden: number, perceptionC: number): ReturnType<typeof initWeights> {
  let off = 0;
  const W1 = flat.slice(off, off + hidden * perceptionC); off += W1.length;
  const b1 = flat.slice(off, off + hidden); off += hidden;
  const W2 = flat.slice(off, off + nChannels * hidden); off += W2.length;
  const b2 = flat.slice(off, off + nChannels);
  return { W1, b1, W2, b2 };
}

// ---------- perception (Sobel) ----------
// grid: Float32Array[GRID_H * GRID_W * N_CHANNELS]
function perceive(grid: Float32Array, gw: number, gh: number, nCh: number): Float32Array {
  // output: [gh * gw * nCh * 3]
  const out = new Float32Array(gh * gw * nCh * 3);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const ci = (y * gw + x) * nCh * 3;
      for (let c = 0; c < nCh; c++) {
        const gat = (yy: number, xx: number) => {
          const ny = Math.min(gh-1, Math.max(0, y + yy));
          const nx = Math.min(gw-1, Math.max(0, x + xx));
          return grid[(ny * gw + nx) * nCh + c];
        };
        // identity
        out[ci + c] = gat(0, 0);
        // sobel x (Gx)
        out[ci + nCh + c] = (
          -gat(-1,-1) + gat(-1,1)
          -2*gat(0,-1) + 2*gat(0,1)
          -gat(1,-1) + gat(1,1)
        ) / 8;
        // sobel y (Gy)
        out[ci + nCh*2 + c] = (
          -gat(-1,-1) - 2*gat(-1,0) - gat(-1,1)
          +gat(1,-1) + 2*gat(1,0) + gat(1,1)
        ) / 8;
      }
    }
  }
  return out;
}

// ---------- MLP forward ----------
function mlpForward(perc: Float32Array, gw: number, gh: number, nCh: number, hidden: number, W1: Float32Array, b1: Float32Array, W2: Float32Array, b2: Float32Array) {
  const N = gw * gh;
  const pCh = nCh * 3;
  const delta = new Float32Array(N * nCh);
  // store hidden pre-act and post-act for backprop
  const h_pre = new Float32Array(N * hidden);
  const h_act = new Float32Array(N * hidden);

  for (let i = 0; i < N; i++) {
    // layer 1: hidden = relu(W1 * perc[i] + b1)
    for (let h = 0; h < hidden; h++) {
      let s = b1[h];
      for (let p = 0; p < pCh; p++) s += W1[h * pCh + p] * perc[i * pCh + p];
      h_pre[i * hidden + h] = s;
      h_act[i * hidden + h] = s > 0 ? s : 0;
    }
    // layer 2: delta = W2 * h + b2
    for (let c = 0; c < nCh; c++) {
      let s = b2[c];
      for (let h = 0; h < hidden; h++) s += W2[c * hidden + h] * h_act[i * hidden + h];
      delta[i * nCh + c] = s;
    }
  }
  return { delta, h_pre, h_act };
}

// ---------- NCA step ----------
function ncaStep(grid: Float32Array, gw: number, gh: number, nCh: number, hidden: number, W1: Float32Array, b1: Float32Array, W2: Float32Array, b2: Float32Array, updateRate: number) {
  const perc = perceive(grid, gw, gh, nCh);
  const { delta } = mlpForward(perc, gw, gh, nCh, hidden, W1, b1, W2, b2);
  const N = gw * gh;
  const newGrid = new Float32Array(grid);
  for (let i = 0; i < N; i++) {
    if (Math.random() > updateRate) continue;
    // alive check: alpha of self or neighbor > 0.1
    const y = Math.floor(i / gw), x = i % gw;
    const selfAlpha = grid[i * nCh + 3];
    let neighborAlive = false;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const ny = Math.min(gh-1, Math.max(0, y+dy));
        const nx = Math.min(gw-1, Math.max(0, x+dx));
        if (grid[(ny*gw+nx)*nCh+3] > 0.1) { neighborAlive = true; break; }
      }
      if (neighborAlive) break;
    }
    if (!neighborAlive && selfAlpha < 0.1) continue;
    for (let c = 0; c < nCh; c++) {
      newGrid[i * nCh + c] = grid[i * nCh + c] + delta[i * nCh + c];
    }
    // clamp alpha
    newGrid[i * nCh + 3] = clamp01(newGrid[i * nCh + 3]);
  }
  return newGrid;
}

// ---------- Loss & Backprop (single rollout step) ----------
// Returns loss and gradients wrt W1,b1,W2,b2
function computeGradients(grid: Float32Array, target: Float32Array, gw: number, gh: number, nCh: number, hidden: number, W1: Float32Array, b1: Float32Array, W2: Float32Array, b2: Float32Array) {
  const N = gw * gh;
  const pCh = nCh * 3;
  const perc = perceive(grid, gw, gh, nCh);
  const { delta, h_pre, h_act } = mlpForward(perc, gw, gh, nCh, hidden, W1, b1, W2, b2);

  // MSE loss over RGBA (channels 0-3) comparing grid+delta to target
  let loss = 0;
  const dLoss_dDelta = new Float32Array(N * nCh);
  for (let i = 0; i < N; i++) {
    for (let c = 0; c < 4; c++) {
      const pred = grid[i * nCh + c] + delta[i * nCh + c];
      const diff = pred - target[i * nCh + c];
      loss += diff * diff;
      dLoss_dDelta[i * nCh + c] = 2 * diff / (N * 4);
    }
  }
  loss /= (N * 4);

  // backprop through layer 2
  const gW2 = zeros(nCh * hidden);
  const gb2 = zeros(nCh);
  const dLoss_dH = zeros(N * hidden);

  for (let i = 0; i < N; i++) {
    for (let c = 0; c < nCh; c++) {
      const g = dLoss_dDelta[i * nCh + c];
      gb2[c] += g;
      for (let h = 0; h < hidden; h++) {
        gW2[c * hidden + h] += g * h_act[i * hidden + h];
        dLoss_dH[i * hidden + h] += g * W2[c * hidden + h];
      }
    }
  }

  // backprop through layer 1 (ReLU)
  const gW1 = zeros(hidden * pCh);
  const gb1 = zeros(hidden);

  for (let i = 0; i < N; i++) {
    for (let h = 0; h < hidden; h++) {
      const d = h_pre[i * hidden + h] > 0 ? dLoss_dH[i * hidden + h] : 0;
      gb1[h] += d;
      for (let p = 0; p < pCh; p++) {
        gW1[h * pCh + p] += d * perc[i * pCh + p];
      }
    }
  }

  return { loss, gW1, gb1, gW2, gb2 };
}

// ---------- Adam update ----------
function adamUpdate(weights: ReturnType<typeof initWeights>, grads: { gW1: Float32Array, gb1: Float32Array, gW2: Float32Array, gb2: Float32Array }, state: AdamState, lr: number) {
  const flat = flatWeights(weights);
  const gflat = new Float32Array(flat.length);
  let off = 0;
  gflat.set(grads.gW1, off); off += grads.gW1.length;
  gflat.set(grads.gb1, off); off += grads.gb1.length;
  gflat.set(grads.gW2, off); off += grads.gW2.length;
  gflat.set(grads.gb2, off);

  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  state.t += 1;
  const bc1 = 1 - Math.pow(beta1, state.t);
  const bc2 = 1 - Math.pow(beta2, state.t);

  for (let i = 0; i < flat.length; i++) {
    state.m[i] = beta1 * state.m[i] + (1 - beta1) * gflat[i];
    state.v[i] = beta2 * state.v[i] + (1 - beta2) * gflat[i] * gflat[i];
    flat[i] -= lr * (state.m[i] / bc1) / (Math.sqrt(state.v[i] / bc2) + eps);
  }

  return unflatWeights(flat, N_CHANNELS, HIDDEN, PERCEPTION_CHANNELS);
}

// ---------- Target patterns ----------
function makeTarget(name: string, gw: number, gh: number): Float32Array {
  const t = new Float32Array(gw * gh * 4);
  const cx = gw / 2, cy = gh / 2;
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const i = (y * gw + x) * 4;
      const dx = x - cx, dy = y - cy;
      const r = Math.sqrt(dx*dx + dy*dy);
      let R=0, G=0, B=0, A=0;

      if (name === 'circle') {
        A = r < gw*0.4 ? 1 : 0;
        R=0.2; G=0.6; B=1.0;
      } else if (name === 'ring') {
        A = (r > gw*0.25 && r < gw*0.42) ? 1 : 0;
        R=1.0; G=0.4; B=0.1;
      } else if (name === 'smile') {
        // face circle
        if (r < gw*0.42) { A=1; R=1; G=0.85; B=0.2; }
        // eyes
        const le = Math.sqrt((x-cx+gw*0.15)*(x-cx+gw*0.15)+(y-cy+gh*0.12)*(y-cy+gh*0.12));
        const re = Math.sqrt((x-cx-gw*0.15)*(x-cx-gw*0.15)+(y-cy+gh*0.12)*(y-cy+gh*0.12));
        if (le < gw*0.08 || re < gw*0.08) { A=1; R=0.1; G=0.1; B=0.1; }
        // smile arc (bottom half, parabolic)
        if (r < gw*0.42 && dy > gh*0.05) {
          const smileR = Math.sqrt(dx*dx + (dy - gh*0.1)*(dy - gh*0.1));
          if (smileR > gw*0.2 && smileR < gw*0.28 && dy > 0) { A=1; R=0.1; G=0.1; B=0.1; }
        }
      } else if (name === 'cross') {
        A = (Math.abs(dx) < gw*0.1 || Math.abs(dy) < gh*0.1) && r < gw*0.45 ? 1 : 0;
        R=0.9; G=0.2; B=0.2;
      } else if (name === 'star') {
        const angle = Math.atan2(dy, dx);
        const points = 5;
        const outerR = gw * 0.42, innerR = gw * 0.18;
        let minR = 1e9;
        for (let p = 0; p < points; p++) {
          const a1 = (p / points) * Math.PI * 2 - Math.PI / 2;
          const a2 = ((p + 0.5) / points) * Math.PI * 2 - Math.PI / 2;
          // rough test: interpolate between inner/outer by angle
          const normAngle = ((angle - a1 + Math.PI * 4) % (Math.PI * 2 / points));
          const half = Math.PI / points;
          const frac = normAngle < half ? normAngle / half : (2 * half - normAngle) / half;
          const threshold = innerR + (outerR - innerR) * frac;
          minR = Math.min(minR, threshold);
        }
        // simpler star: use polar
        const starA = Math.atan2(dy, dx) + Math.PI / 2;
        const starR2 = 0.5 * (outerR + innerR) + 0.5 * (outerR - innerR) * Math.cos(points * starA);
        A = r < starR2 ? 1 : 0;
        R=1.0; G=0.85; B=0.0;
      } else {
        // checkerboard
        A = ((Math.floor(x/4) + Math.floor(y/4)) % 2 === 0 && r < gw*0.42) ? 1 : 0;
        R=0.4; G=0.9; B=0.5;
      }
      t[i]=R; t[i+1]=G; t[i+2]=B; t[i+3]=A;
    }
  }
  return t;
}

// ---------- Grid helpers ----------
function makeSeedGrid(gw: number, gh: number, nCh: number): Float32Array {
  const g = new Float32Array(gw * gh * nCh);
  const cy = Math.floor(gh/2), cx = Math.floor(gw/2);
  const i = (cy * gw + cx) * nCh;
  g[i+3] = 1; // alpha=1 seed
  return g;
}

function gridToImageData(grid: Float32Array, gw: number, gh: number, nCh: number, imgData: ImageData) {
  for (let i = 0; i < gw * gh; i++) {
    const alpha = clamp01(grid[i * nCh + 3]);
    imgData.data[i*4]   = clamp01(grid[i*nCh+0]) * alpha * 255;
    imgData.data[i*4+1] = clamp01(grid[i*nCh+1]) * alpha * 255;
    imgData.data[i*4+2] = clamp01(grid[i*nCh+2]) * alpha * 255;
    imgData.data[i*4+3] = alpha * 255;
  }
}

function targetToImageData(target: Float32Array, gw: number, gh: number, imgData: ImageData) {
  for (let i = 0; i < gw * gh; i++) {
    const alpha = target[i*4+3];
    imgData.data[i*4]   = target[i*4+0] * alpha * 255;
    imgData.data[i*4+1] = target[i*4+1] * alpha * 255;
    imgData.data[i*4+2] = target[i*4+2] * alpha * 255;
    imgData.data[i*4+3] = alpha * 255;
  }
}

// ---------- Main ----------
document.addEventListener('DOMContentLoaded', () => {
  // State
  let nCh = N_CHANNELS;
  let hidden = HIDDEN;
  let percC = nCh * 3;

  let weights = initWeights(nCh, hidden, percC);
  let adamState = initAdam(weights);
  let grid = makeSeedGrid(GRID_W, GRID_H, nCh);
  let targetName = 'smile';
  let target = makeTarget(targetName, GRID_W, GRID_H);

  let lr = 2e-3;
  let rolloutLen = 8;
  let updateRate = 0.5;
  let running = false;
  let trainEvery = 1;  // train every N render frames
  let frameCount = 0;
  let speed = 1; // steps per render frame
  let losses: number[] = [];
  const MAX_LOSS_HISTORY = 200;

  // DOM
  const gridCanvas = document.getElementById('grid-canvas') as HTMLCanvasElement;
  const targetCanvas = document.getElementById('target-canvas') as HTMLCanvasElement;
  const lossCanvas = document.getElementById('loss-canvas') as HTMLCanvasElement;
  const lossLabel = document.getElementById('loss-label') as HTMLSpanElement;
  const stepCount = document.getElementById('step-count') as HTMLSpanElement;

  gridCanvas.width = GRID_W * CELL_SIZE;
  gridCanvas.height = GRID_H * CELL_SIZE;
  targetCanvas.width = GRID_W * CELL_SIZE;
  targetCanvas.height = GRID_H * CELL_SIZE;

  const gridCtx = gridCanvas.getContext('2d')!;
  const targetCtx = targetCanvas.getContext('2d')!;
  const lossCtx = lossCanvas.getContext('2d')!;

  // Controls
  const lrInput = document.getElementById('lr') as HTMLInputElement;
  const lrVal = document.getElementById('lr-val') as HTMLSpanElement;
  const rolloutInput = document.getElementById('rollout') as HTMLInputElement;
  const rolloutVal = document.getElementById('rollout-val') as HTMLSpanElement;
  const speedInput = document.getElementById('speed') as HTMLInputElement;
  const speedVal = document.getElementById('speed-val') as HTMLSpanElement;
  const patternSel = document.getElementById('pattern') as HTMLSelectElement;
  const playBtn = document.getElementById('play-btn') as HTMLButtonElement;
  const stepBtn = document.getElementById('step-btn') as HTMLButtonElement;
  const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;
  const seedBtn = document.getElementById('seed-btn') as HTMLButtonElement;

  lrInput.addEventListener('input', () => { lr = Math.pow(10, parseFloat(lrInput.value)); lrVal.textContent = lr.toExponential(1); });
  rolloutInput.addEventListener('input', () => { rolloutLen = parseInt(rolloutInput.value); rolloutVal.textContent = String(rolloutLen); });
  speedInput.addEventListener('input', () => { speed = parseInt(speedInput.value); speedVal.textContent = String(speed); });
  patternSel.addEventListener('change', () => { targetName = patternSel.value; target = makeTarget(targetName, GRID_W, GRID_H); drawTarget(); });

  playBtn.addEventListener('click', () => { running = !running; playBtn.textContent = running ? 'Pause' : 'Play'; });
  stepBtn.addEventListener('click', () => { doStep(); render(); });
  resetBtn.addEventListener('click', () => {
    weights = initWeights(nCh, hidden, percC);
    adamState = initAdam(weights);
    grid = makeSeedGrid(GRID_W, GRID_H, nCh);
    losses = [];
    frameCount = 0;
    render();
  });
  seedBtn.addEventListener('click', () => {
    grid = makeSeedGrid(GRID_W, GRID_H, nCh);
    render();
  });

  // Click to damage
  gridCanvas.addEventListener('click', (e) => {
    const rect = gridCanvas.getBoundingClientRect();
    const mx = Math.floor((e.clientX - rect.left) / CELL_SIZE);
    const my = Math.floor((e.clientY - rect.top) / CELL_SIZE);
    const radius = 3;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx*dx+dy*dy > radius*radius) continue;
        const nx = mx+dx, ny = my+dy;
        if (nx < 0 || ny < 0 || nx >= GRID_W || ny >= GRID_H) continue;
        const base = (ny*GRID_W+nx)*nCh;
        for (let c = 0; c < nCh; c++) grid[base+c] = 0;
      }
    }
    render();
  });

  let totalSteps = 0;

  function doStep() {
    grid = ncaStep(grid, GRID_W, GRID_H, nCh, hidden, weights.W1, weights.b1, weights.W2, weights.b2, updateRate);
    totalSteps++;
    stepCount.textContent = String(totalSteps);
  }

  function doTrain() {
    // Short rollout: run rolloutLen steps, accumulate gradients via single-step approx
    // We do N forward steps then compute gradients on final state vs target
    let g = new Float32Array(grid);
    for (let s = 0; s < rolloutLen; s++) {
      g = ncaStep(g, GRID_W, GRID_H, nCh, hidden, weights.W1, weights.b1, weights.W2, weights.b2, updateRate);
    }
    const { loss, gW1, gb1, gW2, gb2 } = computeGradients(g, target, GRID_W, GRID_H, nCh, hidden, weights.W1, weights.b1, weights.W2, weights.b2);
    weights = adamUpdate(weights, { gW1, gb1, gW2, gb2 }, adamState, lr);

    losses.push(loss);
    if (losses.length > MAX_LOSS_HISTORY) losses.shift();
    lossLabel.textContent = loss.toFixed(5);
  }

  function drawTarget() {
    const imgData = targetCtx.createImageData(GRID_W, GRID_H);
    targetToImageData(target, GRID_W, GRID_H, imgData);
    const offscreen = document.createElement('canvas');
    offscreen.width = GRID_W; offscreen.height = GRID_H;
    offscreen.getContext('2d')!.putImageData(imgData, 0, 0);
    targetCtx.imageSmoothingEnabled = false;
    targetCtx.drawImage(offscreen, 0, 0, GRID_W * CELL_SIZE, GRID_H * CELL_SIZE);
  }

  function drawGrid() {
    const imgData = gridCtx.createImageData(GRID_W, GRID_H);
    gridToImageData(grid, GRID_W, GRID_H, nCh, imgData);
    const offscreen = document.createElement('canvas');
    offscreen.width = GRID_W; offscreen.height = GRID_H;
    offscreen.getContext('2d')!.putImageData(imgData, 0, 0);
    gridCtx.imageSmoothingEnabled = false;
    gridCtx.drawImage(offscreen, 0, 0, GRID_W * CELL_SIZE, GRID_H * CELL_SIZE);
  }

  function drawLoss() {
    const W = lossCanvas.width, H = lossCanvas.height;
    lossCtx.fillStyle = '#1a1a2e';
    lossCtx.fillRect(0, 0, W, H);
    if (losses.length < 2) return;
    const maxL = Math.max(...losses);
    const minL = Math.min(...losses);
    const range = maxL - minL || 1e-8;
    lossCtx.strokeStyle = '#4fc3f7';
    lossCtx.lineWidth = 1.5;
    lossCtx.beginPath();
    for (let i = 0; i < losses.length; i++) {
      const px = (i / (MAX_LOSS_HISTORY - 1)) * W;
      const py = H - ((losses[i] - minL) / range) * (H - 4) - 2;
      if (i === 0) lossCtx.moveTo(px, py); else lossCtx.lineTo(px, py);
    }
    lossCtx.stroke();
    lossCtx.fillStyle = '#aaa';
    lossCtx.font = '10px monospace';
    lossCtx.fillText(maxL.toFixed(4), 2, 12);
    lossCtx.fillText(minL.toFixed(4), 2, H - 4);
  }

  function render() {
    drawGrid();
    drawLoss();
  }

  function loop() {
    requestAnimationFrame(loop);
    if (!running) return;
    for (let s = 0; s < speed; s++) {
      doStep();
    }
    frameCount++;
    doTrain();
    render();
  }

  // Initial draw
  drawTarget();
  drawGrid();
  drawLoss();
  requestAnimationFrame(loop);
});
