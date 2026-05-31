import * as d3 from 'd3';

// ─── Types ────────────────────────────────────────────────────────────────────

type DatasetName = 'ring' | 'moons' | 'grid' | 'blobs' | 'rgb';

interface Neuron {
  weights: number[];  // length 2 for 2D datasets, 3 for RGB
  gi: number;         // grid row
  gj: number;         // grid col
}

interface SOMState {
  neurons: Neuron[];
  gridRows: number;
  gridCols: number;
  iteration: number;
  maxIter: number;
  lr0: number;
  sigma0: number;
  inputDim: number;
  dataset: DatasetName;
}

interface Point2D {
  x: number;
  y: number;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function randn(): number {
  const u = Math.random() + 1e-12;
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

// ─── Dataset generators ──────────────────────────────────────────────────────

function genRing(n: number): number[][] {
  const pts: number[][] = [];
  for (let i = 0; i < n; i++) {
    const t = Math.random() * 2 * Math.PI;
    const r = 0.75 + randn() * 0.04;
    pts.push([r * Math.cos(t), r * Math.sin(t)]);
  }
  return pts;
}

function genMoons(n: number): number[][] {
  const pts: number[][] = [];
  for (let i = 0; i < n; i++) {
    const upper = i < n / 2;
    const t = Math.PI * Math.random();
    if (upper) {
      pts.push([Math.cos(t) * 0.75, Math.sin(t) * 0.75 + randn() * 0.05]);
    } else {
      pts.push([0.75 - Math.cos(t) * 0.75, -Math.sin(t) * 0.75 + 0.25 + randn() * 0.05]);
    }
  }
  return pts;
}

function genGrid(n: number): number[][] {
  const pts: number[][] = [];
  const side = Math.ceil(Math.sqrt(n));
  for (let i = 0; i < side; i++) {
    for (let j = 0; j < side; j++) {
      if (pts.length >= n) break;
      pts.push([
        (i / (side - 1)) * 1.6 - 0.8 + randn() * 0.02,
        (j / (side - 1)) * 1.6 - 0.8 + randn() * 0.02,
      ]);
    }
  }
  return pts;
}

function genBlobs(n: number): number[][] {
  const centers: [number, number][] = [
    [-0.6, -0.6], [0.6, 0.6], [-0.6, 0.6], [0.6, -0.6],
  ];
  const pts: number[][] = [];
  for (let i = 0; i < n; i++) {
    const c = centers[i % centers.length];
    pts.push([c[0] + randn() * 0.18, c[1] + randn() * 0.18]);
  }
  return pts;
}

function genRGB(n: number): number[][] {
  const pts: number[][] = [];
  for (let i = 0; i < n; i++) {
    pts.push([Math.random(), Math.random(), Math.random()]);
  }
  return pts;
}

function generateDataset(name: DatasetName, n: number): number[][] {
  switch (name) {
    case 'ring':  return genRing(n);
    case 'moons': return genMoons(n);
    case 'grid':  return genGrid(n);
    case 'blobs': return genBlobs(n);
    case 'rgb':   return genRGB(n);
  }
}

// ─── SOM core ────────────────────────────────────────────────────────────────

function initNeurons(
  rows: number, cols: number, inputDim: number, dataset: DatasetName
): Neuron[] {
  const neurons: Neuron[] = [];
  for (let gi = 0; gi < rows; gi++) {
    for (let gj = 0; gj < cols; gj++) {
      let weights: number[];
      if (inputDim === 3) {
        // initialise with a smooth colour ramp across the grid
        weights = [gi / (rows - 1), gj / (cols - 1), 0.5];
      } else {
        // small random patch in [-0.2, 0.2]² around centre
        weights = [
          (gi / (rows - 1) - 0.5) * 0.4 + randn() * 0.05,
          (gj / (cols - 1) - 0.5) * 0.4 + randn() * 0.05,
        ];
      }
      neurons.push({ weights, gi, gj });
    }
  }
  return neurons;
}

function euclidSq(a: number[], b: number[]): number {
  let s = 0;
  for (let k = 0; k < a.length; k++) {
    const d = a[k] - b[k];
    s += d * d;
  }
  return s;
}

function findBMU(neurons: Neuron[], input: number[]): number {
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let i = 0; i < neurons.length; i++) {
    const d = euclidSq(neurons[i].weights, input);
    if (d < bestDist) { bestDist = d; bestIdx = i; }
  }
  return bestIdx;
}

function gridDist2(a: Neuron, b: Neuron): number {
  const dr = a.gi - b.gi;
  const dc = a.gj - b.gj;
  return dr * dr + dc * dc;
}

function somStep(state: SOMState, input: number[]): void {
  const t = state.iteration;
  const T = state.maxIter;
  // decaying learning rate and neighbourhood radius
  const lr = state.lr0 * Math.exp(-t / T);
  const sigma = state.sigma0 * Math.exp(-t / (T / Math.log(state.sigma0 + 1)));
  const sigma2 = 2 * sigma * sigma;

  const bmuIdx = findBMU(state.neurons, input);
  const bmu = state.neurons[bmuIdx];

  for (let i = 0; i < state.neurons.length; i++) {
    const n = state.neurons[i];
    const gd2 = gridDist2(n, bmu);
    const h = Math.exp(-gd2 / (sigma2 + 1e-10));
    const delta = lr * h;
    for (let k = 0; k < n.weights.length; k++) {
      n.weights[k] += delta * (input[k] - n.weights[k]);
    }
  }
  state.iteration++;
}

// ─── U-Matrix ────────────────────────────────────────────────────────────────

function computeUMatrix(state: SOMState): number[] {
  const { neurons, gridRows, gridCols } = state;
  const umat: number[] = new Array(neurons.length).fill(0);
  const idxOf = (gi: number, gj: number) => gi * gridCols + gj;

  for (let gi = 0; gi < gridRows; gi++) {
    for (let gj = 0; gj < gridCols; gj++) {
      const n = neurons[idxOf(gi, gj)];
      let sum = 0;
      let cnt = 0;
      const neighbours: [number, number][] = [
        [gi - 1, gj], [gi + 1, gj], [gi, gj - 1], [gi, gj + 1],
      ];
      neighbours.forEach(function(nb) {
        const ni = nb[0], nj = nb[1];
        if (ni >= 0 && ni < gridRows && nj >= 0 && nj < gridCols) {
          sum += Math.sqrt(euclidSq(n.weights, neurons[idxOf(ni, nj)].weights));
          cnt++;
        }
      });
      umat[idxOf(gi, gj)] = cnt > 0 ? sum / cnt : 0;
    }
  }
  return umat;
}

// ─── Canvas rendering helpers ─────────────────────────────────────────────────

function clearCanvas(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  ctx.fillStyle = '#0d0d1e';
  ctx.fillRect(0, 0, w, h);
}

// ─── 2D lattice view ──────────────────────────────────────────────────────────

function draw2D(
  ctx: CanvasRenderingContext2D,
  state: SOMState,
  data: number[][],
  w: number,
  h: number
): void {
  clearCanvas(ctx, w, h);

  // compute data bounds with some padding
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  data.forEach(function(p) {
    if (p[0] < xMin) xMin = p[0];
    if (p[0] > xMax) xMax = p[0];
    if (p[1] < yMin) yMin = p[1];
    if (p[1] > yMax) yMax = p[1];
  });
  // also include neuron positions in bounds
  state.neurons.forEach(function(n) {
    if (n.weights[0] < xMin) xMin = n.weights[0];
    if (n.weights[0] > xMax) xMax = n.weights[0];
    if (n.weights[1] < yMin) yMin = n.weights[1];
    if (n.weights[1] > yMax) yMax = n.weights[1];
  });
  const pad = 0.12;
  const rx = xMax - xMin || 1;
  const ry = yMax - yMin || 1;
  xMin -= rx * pad; xMax += rx * pad;
  yMin -= ry * pad; yMax += ry * pad;

  const sx = (v: number) => ((v - xMin) / (xMax - xMin)) * w;
  const sy = (v: number) => h - ((v - yMin) / (yMax - yMin)) * h;

  // data points
  ctx.globalAlpha = 0.45;
  ctx.fillStyle = '#4fc3f7';
  data.forEach(function(p) {
    ctx.beginPath();
    ctx.arc(sx(p[0]), sy(p[1]), 2.5, 0, 2 * Math.PI);
    ctx.fill();
  });
  ctx.globalAlpha = 1;

  const { gridRows, gridCols, neurons } = state;
  const idxOf = (gi: number, gj: number) => gi * gridCols + gj;

  // grid edges
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(167,139,250,0.55)';
  ctx.beginPath();
  for (let gi = 0; gi < gridRows; gi++) {
    for (let gj = 0; gj < gridCols; gj++) {
      const n = neurons[idxOf(gi, gj)];
      const nx = sx(n.weights[0]);
      const ny = sy(n.weights[1]);
      if (gj + 1 < gridCols) {
        const nb = neurons[idxOf(gi, gj + 1)];
        ctx.moveTo(nx, ny);
        ctx.lineTo(sx(nb.weights[0]), sy(nb.weights[1]));
      }
      if (gi + 1 < gridRows) {
        const nb = neurons[idxOf(gi + 1, gj)];
        ctx.moveTo(nx, ny);
        ctx.lineTo(sx(nb.weights[0]), sy(nb.weights[1]));
      }
    }
  }
  ctx.stroke();

  // neuron nodes
  for (let i = 0; i < neurons.length; i++) {
    const n = neurons[i];
    ctx.beginPath();
    ctx.arc(sx(n.weights[0]), sy(n.weights[1]), 3, 0, 2 * Math.PI);
    ctx.fillStyle = '#a78bfa';
    ctx.fill();
  }
}

// ─── RGB color-map view ───────────────────────────────────────────────────────

function drawRGB(
  ctx: CanvasRenderingContext2D,
  state: SOMState,
  w: number,
  h: number
): void {
  clearCanvas(ctx, w, h);
  const { gridRows, gridCols, neurons } = state;
  const cellW = w / gridCols;
  const cellH = h / gridRows;

  for (let i = 0; i < neurons.length; i++) {
    const n = neurons[i];
    const r = clamp(Math.round(n.weights[0] * 255), 0, 255);
    const g = clamp(Math.round(n.weights[1] * 255), 0, 255);
    const b = clamp(Math.round(n.weights[2] * 255), 0, 255);
    ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
    ctx.fillRect(
      n.gj * cellW,
      n.gi * cellH,
      Math.ceil(cellW),
      Math.ceil(cellH)
    );
  }

  // light grid lines
  ctx.strokeStyle = 'rgba(0,0,0,0.25)';
  ctx.lineWidth = 0.5;
  for (let gi = 0; gi <= gridRows; gi++) {
    ctx.beginPath();
    ctx.moveTo(0, gi * cellH);
    ctx.lineTo(w, gi * cellH);
    ctx.stroke();
  }
  for (let gj = 0; gj <= gridCols; gj++) {
    ctx.beginPath();
    ctx.moveTo(gj * cellW, 0);
    ctx.lineTo(gj * cellW, h);
    ctx.stroke();
  }
}

// ─── U-Matrix view ────────────────────────────────────────────────────────────

function drawUMatrix(
  ctx: CanvasRenderingContext2D,
  state: SOMState,
  w: number,
  h: number
): void {
  const umat = computeUMatrix(state);
  const maxU = umat.reduce(function(a, b) { return b > a ? b : a; }, 0) || 1;

  const { gridRows, gridCols } = state;
  const cellW = w / gridCols;
  const cellH = h / gridRows;

  for (let i = 0; i < state.neurons.length; i++) {
    const n = state.neurons[i];
    const v = umat[i] / maxU; // 0=close neighbors (cluster interior), 1=far (boundary)
    // cool-to-warm: low=dark-blue, high=orange-red
    const col = d3.interpolateYlOrRd(v);
    ctx.fillStyle = col;
    ctx.fillRect(
      n.gj * cellW,
      n.gi * cellH,
      Math.ceil(cellW),
      Math.ceil(cellH)
    );
  }
}

// ─── App state ────────────────────────────────────────────────────────────────

let somState: SOMState;
let trainingData: number[][];
let shuffleIdx: number[] = [];
let shufflePos = 0;
let animHandle: number | null = null;
let isPlaying = false;
let stepsPerFrame = 30;

// DOM refs (assigned after DOMContentLoaded)
let mainCanvas: HTMLCanvasElement;
let umatCanvas: HTMLCanvasElement;
let mainCtx: CanvasRenderingContext2D;
let umatCtx: CanvasRenderingContext2D;
let iterLabel: HTMLElement;
let lrLabel: HTMLElement;
let sigLabel: HTMLElement;

function getCurrentLR(): number {
  const t = somState.iteration;
  const T = somState.maxIter;
  return somState.lr0 * Math.exp(-t / T);
}

function getCurrentSigma(): number {
  const t = somState.iteration;
  const T = somState.maxIter;
  return somState.sigma0 * Math.exp(-t / (T / Math.log(somState.sigma0 + 1)));
}

function nextSample(): number[] {
  if (shufflePos >= shuffleIdx.length) {
    // reshuffle
    for (let i = shuffleIdx.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = shuffleIdx[i];
      shuffleIdx[i] = shuffleIdx[j];
      shuffleIdx[j] = tmp;
    }
    shufflePos = 0;
  }
  return trainingData[shuffleIdx[shufflePos++]];
}

function redraw(): void {
  const W = mainCanvas.width;
  const H = mainCanvas.height;
  const UW = umatCanvas.width;
  const UH = umatCanvas.height;

  if (somState.dataset === 'rgb') {
    drawRGB(mainCtx, somState, W, H);
  } else {
    draw2D(mainCtx, somState, trainingData as number[][], W, H);
  }
  drawUMatrix(umatCtx, somState, UW, UH);

  iterLabel.textContent = String(somState.iteration);
  lrLabel.textContent = getCurrentLR().toFixed(4);
  sigLabel.textContent = getCurrentSigma().toFixed(2);
}

function doSteps(n: number): void {
  const maxReached = somState.iteration >= somState.maxIter;
  if (maxReached) {
    if (isPlaying) pauseTraining();
    return;
  }
  const stepsToRun = Math.min(n, somState.maxIter - somState.iteration);
  for (let s = 0; s < stepsToRun; s++) {
    somStep(somState, nextSample());
  }
  redraw();
}

function loop(): void {
  doSteps(stepsPerFrame);
  if (isPlaying && somState.iteration < somState.maxIter) {
    animHandle = requestAnimationFrame(loop);
  } else {
    isPlaying = false;
    animHandle = null;
  }
}

function playTraining(): void {
  if (isPlaying) return;
  isPlaying = true;
  animHandle = requestAnimationFrame(loop);
}

function pauseTraining(): void {
  isPlaying = false;
  if (animHandle !== null) {
    cancelAnimationFrame(animHandle);
    animHandle = null;
  }
}

function buildState(
  dataset: DatasetName,
  gridRows: number,
  gridCols: number,
  lr0: number,
  sigma0: number,
  maxIter: number
): void {
  const inputDim = dataset === 'rgb' ? 3 : 2;
  const dataN = dataset === 'rgb' ? 2000 : 600;
  trainingData = generateDataset(dataset, dataN);
  shuffleIdx = [];
  for (let i = 0; i < trainingData.length; i++) shuffleIdx.push(i);
  shufflePos = trainingData.length; // force reshuffle on first access
  somState = {
    neurons: initNeurons(gridRows, gridCols, inputDim, dataset),
    gridRows,
    gridCols,
    iteration: 0,
    maxIter,
    lr0,
    sigma0,
    inputDim,
    dataset,
  };
}

// ─── Controls wiring ─────────────────────────────────────────────────────────

function getInt(id: string): number {
  return parseInt((document.getElementById(id) as HTMLInputElement).value, 10);
}

function getFloat(id: string): number {
  return parseFloat((document.getElementById(id) as HTMLInputElement).value);
}

function getSelect(id: string): string {
  return (document.getElementById(id) as HTMLSelectElement).value;
}

function resetAll(): void {
  pauseTraining();
  const dataset = getSelect('sel-dataset') as DatasetName;
  const gridSize = getInt('sl-grid');
  const lr0 = getFloat('sl-lr');
  const sigma0 = getFloat('sl-sigma');
  const maxIter = getInt('sl-maxiter');
  buildState(dataset, gridSize, gridSize, lr0, sigma0, maxIter);
  redraw();
}

// ─── Entry point ─────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function() {
  mainCanvas = document.getElementById('main-canvas') as HTMLCanvasElement;
  umatCanvas = document.getElementById('umat-canvas') as HTMLCanvasElement;
  mainCtx = mainCanvas.getContext('2d')!;
  umatCtx = umatCanvas.getContext('2d')!;
  iterLabel = document.getElementById('stat-iter')!;
  lrLabel = document.getElementById('stat-lr')!;
  sigLabel = document.getElementById('stat-sigma')!;

  // Resize canvases to display size
  function resizeCanvases(): void {
    const dpr = window.devicePixelRatio || 1;
    const sizeMain = mainCanvas.getBoundingClientRect();
    mainCanvas.width = Math.round(sizeMain.width * dpr);
    mainCanvas.height = Math.round(sizeMain.height * dpr);
    mainCtx.scale(dpr, dpr);
    // umat canvas sized from its container too
    const sizeUmat = umatCanvas.getBoundingClientRect();
    umatCanvas.width = Math.round(sizeUmat.width * dpr);
    umatCanvas.height = Math.round(sizeUmat.height * dpr);
    umatCtx.scale(dpr, dpr);
  }

  // Helper to wire slider + display label
  function wireSlider(sliderId: string, displayId: string, fmt: (v: number) => string): void {
    const sl = document.getElementById(sliderId) as HTMLInputElement;
    const dsp = document.getElementById(displayId)!;
    dsp.textContent = fmt(parseFloat(sl.value));
    sl.addEventListener('input', function() {
      dsp.textContent = fmt(parseFloat(sl.value));
    });
  }

  wireSlider('sl-grid', 'val-grid', function(v) { return String(Math.round(v)) + 'x' + String(Math.round(v)); });
  wireSlider('sl-lr', 'val-lr', function(v) { return v.toFixed(2); });
  wireSlider('sl-sigma', 'val-sigma', function(v) { return v.toFixed(1); });
  wireSlider('sl-maxiter', 'val-maxiter', function(v) { return String(Math.round(v)); });
  wireSlider('sl-speed', 'val-speed', function(v) {
    stepsPerFrame = Math.round(v);
    return String(Math.round(v));
  });

  document.getElementById('btn-play')!.addEventListener('click', playTraining);
  document.getElementById('btn-pause')!.addEventListener('click', pauseTraining);
  document.getElementById('btn-step')!.addEventListener('click', function() {
    pauseTraining();
    doSteps(1);
  });
  document.getElementById('btn-reset')!.addEventListener('click', resetAll);

  // dataset change triggers reset
  document.getElementById('sel-dataset')!.addEventListener('change', resetAll);

  // initial state
  resizeCanvases();
  resetAll();
});
