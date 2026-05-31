import * as d3 from 'd3';

// ── Types ──────────────────────────────────────────────────────────────────
interface Point { x: number; y: number; label: number; }
type KernelType = 'linear' | 'poly' | 'rbf';
type DatasetType = 'blobs' | 'moons' | 'circles' | 'xor' | 'overlapping';

// ── Kernel functions ───────────────────────────────────────────────────────
function kernelLinear(a: number[], b: number[]): number {
  return a[0] * b[0] + a[1] * b[1];
}
function kernelPoly(a: number[], b: number[], degree: number): number {
  return Math.pow(1 + a[0] * b[0] + a[1] * b[1], degree);
}
function kernelRBF(a: number[], b: number[], gamma: number): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.exp(-gamma * (dx * dx + dy * dy));
}

// ── SMO-lite SVM ───────────────────────────────────────────────────────────
class SVM {
  alphas: number[] = [];
  bias: number = 0;
  private xs: number[][] = [];
  private ys: number[] = [];
  private C: number = 1;
  private kernelType: KernelType = 'rbf';
  private gamma: number = 0.5;
  private degree: number = 3;
  private cache: Float64Array | null = null;
  private n: number = 0;

  constructor(C: number, kernelType: KernelType, gamma: number, degree: number) {
    this.C = C;
    this.kernelType = kernelType;
    this.gamma = gamma;
    this.degree = degree;
  }

  private K(i: number, j: number): number {
    if (this.cache) {
      const idx = i * this.n + j;
      if (this.cache[idx] !== -999) return this.cache[idx];
      const v = this.rawK(this.xs[i], this.xs[j]);
      this.cache[idx] = v;
      return v;
    }
    return this.rawK(this.xs[i], this.xs[j]);
  }

  private rawK(a: number[], b: number[]): number {
    if (this.kernelType === 'linear') return kernelLinear(a, b);
    if (this.kernelType === 'poly') return kernelPoly(a, b, this.degree);
    return kernelRBF(a, b, this.gamma);
  }

  private decisionRaw(i: number): number {
    let s = 0;
    for (let j = 0; j < this.n; j++) {
      if (this.alphas[j] > 1e-8) {
        s += this.alphas[j] * this.ys[j] * this.K(j, i);
      }
    }
    return s + this.bias;
  }

  train(points: Point[], maxIter: number = 200): void {
    this.n = points.length;
    this.xs = points.map(function(p) { return [p.x, p.y]; });
    this.ys = points.map(function(p) { return p.label; });
    this.alphas = new Array(this.n).fill(0);
    this.bias = 0;

    // Build kernel cache — but limit size to avoid memory issues
    if (this.n <= 500) {
      this.cache = new Float64Array(this.n * this.n).fill(-999);
    } else {
      this.cache = null;
    }

    const C = this.C;
    const tol = 1e-3;

    for (let iter = 0; iter < maxIter; iter++) {
      let numChanged = 0;
      for (let i = 0; i < this.n; i++) {
        const Ei = this.decisionRaw(i) - this.ys[i];
        const yi = this.ys[i];
        const ai = this.alphas[i];
        // KKT violation check
        if ((yi * Ei < -tol && ai < C) || (yi * Ei > tol && ai > 0)) {
          // Pick j != i randomly (simple heuristic)
          let j = Math.floor(Math.random() * (this.n - 1));
          if (j >= i) j++;
          const Ej = this.decisionRaw(j) - this.ys[j];
          const yj = this.ys[j];
          const aj = this.alphas[j];

          // Compute bounds
          let L: number, H: number;
          if (yi !== yj) {
            L = Math.max(0, aj - ai);
            H = Math.min(C, C + aj - ai);
          } else {
            L = Math.max(0, ai + aj - C);
            H = Math.min(C, ai + aj);
          }
          if (L >= H) continue;

          const eta = 2 * this.K(i, j) - this.K(i, i) - this.K(j, j);
          if (eta >= 0) continue;

          let ajNew = aj - yj * (Ei - Ej) / eta;
          if (ajNew > H) ajNew = H;
          else if (ajNew < L) ajNew = L;
          if (Math.abs(ajNew - aj) < 1e-5) continue;

          const aiNew = ai + yi * yj * (aj - ajNew);
          this.alphas[i] = aiNew;
          this.alphas[j] = ajNew;

          // Update bias
          const b1 = this.bias - Ei - yi * (aiNew - ai) * this.K(i, i) - yj * (ajNew - aj) * this.K(i, j);
          const b2 = this.bias - Ej - yi * (aiNew - ai) * this.K(i, j) - yj * (ajNew - aj) * this.K(j, j);
          if (aiNew > 0 && aiNew < C) this.bias = b1;
          else if (ajNew > 0 && ajNew < C) this.bias = b2;
          else this.bias = (b1 + b2) / 2;
          numChanged++;
        }
      }
      if (numChanged === 0) break;
    }
  }

  predict(x: number, y: number): number {
    let s = 0;
    const pt = [x, y];
    for (let i = 0; i < this.n; i++) {
      if (this.alphas[i] > 1e-8) {
        s += this.alphas[i] * this.ys[i] * this.rawK(this.xs[i], pt);
      }
    }
    return s + this.bias;
  }

  getSupportVectorIndices(): number[] {
    const svs: number[] = [];
    for (let i = 0; i < this.n; i++) {
      if (this.alphas[i] > 1e-8) svs.push(i);
    }
    return svs;
  }

  accuracy(points: Point[]): number {
    let correct = 0;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const pred = this.predict(p.x, p.y) >= 0 ? 1 : -1;
      if (pred === p.label) correct++;
    }
    return correct / points.length;
  }
}

// ── Dataset generators ─────────────────────────────────────────────────────
function randn(): number {
  // Box-Muller
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u + 1e-10)) * Math.cos(2 * Math.PI * v);
}

function generateBlobs(n: number, noise: number): Point[] {
  const pts: Point[] = [];
  const centers = [[-1.5, -1.5], [1.5, 1.5]];
  for (let i = 0; i < n; i++) {
    const label = i < n / 2 ? -1 : 1;
    const c = label === -1 ? centers[0] : centers[1];
    pts.push({ x: c[0] + randn() * (0.5 + noise), y: c[1] + randn() * (0.5 + noise), label });
  }
  return pts;
}

function generateMoons(n: number, noise: number): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const t = Math.PI * i / (n / 2);
    const label = i < n / 2 ? -1 : 1;
    if (i < n / 2) {
      pts.push({ x: Math.cos(t) * 2 + randn() * noise, y: Math.sin(t) * 2 + randn() * noise, label });
    } else {
      const t2 = Math.PI * (i - n / 2) / (n / 2);
      pts.push({ x: Math.cos(t2) * 2 + 1 + randn() * noise, y: -Math.sin(t2) * 2 + 0.5 + randn() * noise, label });
    }
  }
  return pts;
}

function generateCircles(n: number, noise: number): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const label = i < n / 2 ? -1 : 1;
    const r = label === -1 ? 0.8 : 2.0;
    const t = 2 * Math.PI * Math.random();
    pts.push({ x: Math.cos(t) * r + randn() * noise, y: Math.sin(t) * r + randn() * noise, label });
  }
  return pts;
}

function generateXOR(n: number, noise: number): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() * 2 - 1) * 2;
    const y = (Math.random() * 2 - 1) * 2;
    const label = x * y > 0 ? 1 : -1;
    pts.push({ x: x + randn() * noise, y: y + randn() * noise, label });
  }
  return pts;
}

function generateOverlapping(n: number, noise: number): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const label = i < n / 2 ? -1 : 1;
    const cx = label * 0.6;
    pts.push({ x: cx + randn() * (0.8 + noise), y: randn() * (0.8 + noise), label });
  }
  return pts;
}

function generateDataset(type: DatasetType, n: number, noise: number): Point[] {
  if (type === 'blobs') return generateBlobs(n, noise);
  if (type === 'moons') return generateMoons(n, noise);
  if (type === 'circles') return generateCircles(n, noise);
  if (type === 'xor') return generateXOR(n, noise);
  return generateOverlapping(n, noise);
}

// ── App State ──────────────────────────────────────────────────────────────
let points: Point[] = [];
let svm: SVM | null = null;
let trainedPoints: Point[] = [];

// ── DOM refs ───────────────────────────────────────────────────────────────
const canvas = document.getElementById('svmCanvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const btnTrain = document.getElementById('btnTrain') as HTMLButtonElement;
const btnReset = document.getElementById('btnReset') as HTMLButtonElement;
const selDataset = document.getElementById('selDataset') as HTMLSelectElement;
const selKernel = document.getElementById('selKernel') as HTMLSelectElement;
const inpC = document.getElementById('inpC') as HTMLInputElement;
const inpGamma = document.getElementById('inpGamma') as HTMLInputElement;
const inpDegree = document.getElementById('inpDegree') as HTMLInputElement;
const inpN = document.getElementById('inpN') as HTMLInputElement;
const inpNoise = document.getElementById('inpNoise') as HTMLInputElement;
const lblC = document.getElementById('lblC') as HTMLSpanElement;
const lblGamma = document.getElementById('lblGamma') as HTMLSpanElement;
const lblDegree = document.getElementById('lblDegree') as HTMLSpanElement;
const lblN = document.getElementById('lblN') as HTMLSpanElement;
const lblNoise = document.getElementById('lblNoise') as HTMLSpanElement;
const statsEl = document.getElementById('stats') as HTMLDivElement;
const rowGamma = document.getElementById('rowGamma') as HTMLDivElement;
const rowDegree = document.getElementById('rowDegree') as HTMLDivElement;
const statusEl = document.getElementById('status') as HTMLDivElement;

// ── Scale helpers ──────────────────────────────────────────────────────────
const RANGE = 3.5;
let W = canvas.width;
let H = canvas.height;

function dataToCanvas(x: number, y: number): [number, number] {
  const cx = (x + RANGE) / (2 * RANGE) * W;
  const cy = (1 - (y + RANGE) / (2 * RANGE)) * H;
  return [cx, cy];
}

function canvasToData(cx: number, cy: number): [number, number] {
  const x = cx / W * 2 * RANGE - RANGE;
  const y = -(cy / H * 2 * RANGE - RANGE);
  return [x, y];
}

// ── Draw ───────────────────────────────────────────────────────────────────
const GRID = 80;

function drawDecisionBoundary() {
  if (!svm) return;
  const imageData = ctx.createImageData(W, H);
  const data = imageData.data;
  const step = 1;
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const [x, y] = canvasToData(px + 0.5, py + 0.5);
      const val = svm.predict(x, y);
      let r = 26, g = 26, b = 46;
      if (val > 0) {
        const t = Math.min(1, Math.abs(val) / 2);
        r = Math.round(26 + t * (100 - 26));
        g = Math.round(26 + t * (60 - 26));
        b = Math.round(46 + t * (160 - 46));
      } else {
        const t = Math.min(1, Math.abs(val) / 2);
        r = Math.round(26 + t * (160 - 26));
        g = Math.round(26 + t * (60 - 26));
        b = Math.round(46 + t * (80 - 46));
      }
      const idx = (py * W + px) * 4;
      data[idx] = r; data[idx + 1] = g; data[idx + 2] = b; data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);

  // Draw margin contours (f=+1 and f=-1) and boundary (f=0) using marching-squares-lite
  drawContour(0, '#ffffff', 2);
  drawContour(1, '#64d8cb', 1);
  drawContour(-1, '#f4a261', 1);
}

function contourInterp(va: number, vb: number, pa: number, pb: number): number {
  if (Math.abs(vb - va) < 1e-10) return (pa + pb) / 2;
  return pa + (pb - pa) * (-va) / (vb - va);
}

function drawContour(level: number, color: string, lineWidth: number) {
  if (!svm) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(level === 0 ? [] : [4, 4]);
  ctx.beginPath();
  const cols = GRID, rows = GRID;
  const vals: number[][] = [];
  for (let row = 0; row <= rows; row++) {
    vals[row] = [];
    for (let col = 0; col <= cols; col++) {
      const [x, y] = canvasToData(col / cols * W, row / rows * H);
      vals[row][col] = svm.predict(x, y) - level;
    }
  }
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const a = vals[row][col];
      const b = vals[row][col + 1];
      const c = vals[row + 1][col + 1];
      const d = vals[row + 1][col];
      const x0 = col / cols * W;
      const x1 = (col + 1) / cols * W;
      const y0 = row / rows * H;
      const y1 = (row + 1) / rows * H;
      const top    = a * b <= 0 ? [contourInterp(a, b, x0, x1), y0] : null;
      const right  = b * c <= 0 ? [x1, contourInterp(b, c, y0, y1)] : null;
      const bottom = d * c <= 0 ? [contourInterp(d, c, x0, x1), y1] : null;
      const left   = a * d <= 0 ? [x0, contourInterp(a, d, y0, y1)] : null;
      const segs: Array<[number, number]> = [];
      if (top)    segs.push(top    as [number, number]);
      if (right)  segs.push(right  as [number, number]);
      if (bottom) segs.push(bottom as [number, number]);
      if (left)   segs.push(left   as [number, number]);
      if (segs.length >= 2) {
        ctx.moveTo(segs[0][0], segs[0][1]);
        ctx.lineTo(segs[1][0], segs[1][1]);
      }
    }
  }
  ctx.stroke();
  ctx.restore();
}

function drawPoints(svIndices: Set<number>) {
  const radius = 5;
  trainedPoints.forEach(function(p, i) {
    const [cx, cy] = dataToCanvas(p.x, p.y);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.fillStyle = p.label === 1 ? '#60a5fa' : '#f87171';
    ctx.fill();
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth = 1;
    ctx.stroke();

    if (svIndices.has(i)) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius + 4, 0, 2 * Math.PI);
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  });
}

function drawUntrained() {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, W, H);
  points.forEach(function(p) {
    const [cx, cy] = dataToCanvas(p.x, p.y);
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
    ctx.fillStyle = p.label === 1 ? '#60a5fa' : '#f87171';
    ctx.fill();
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

function redraw() {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, W, H);
  if (svm) {
    drawDecisionBoundary();
    const svSet = new Set<number>(svm.getSupportVectorIndices());
    drawPoints(svSet);
    const acc = svm.accuracy(trainedPoints);
    statsEl.textContent = 'Accuracy: ' + (acc * 100).toFixed(1) + '%  |  Support Vectors: ' + svSet.size;
  } else {
    drawUntrained();
    statsEl.textContent = '';
  }
}

// ── Controls wiring ────────────────────────────────────────────────────────
function updateKernelUI() {
  const k = selKernel.value as KernelType;
  rowGamma.style.display = k === 'rbf' ? '' : 'none';
  rowDegree.style.display = k === 'poly' ? '' : 'none';
}

inpC.addEventListener('input', function() { lblC.textContent = parseFloat(inpC.value).toFixed(2); });
inpGamma.addEventListener('input', function() { lblGamma.textContent = parseFloat(inpGamma.value).toFixed(2); });
inpDegree.addEventListener('input', function() { lblDegree.textContent = inpDegree.value; });
inpN.addEventListener('input', function() { lblN.textContent = inpN.value; });
inpNoise.addEventListener('input', function() { lblNoise.textContent = parseFloat(inpNoise.value).toFixed(2); });
selKernel.addEventListener('change', updateKernelUI);

btnReset.addEventListener('click', function() {
  svm = null;
  trainedPoints = [];
  points = generateDataset(
    selDataset.value as DatasetType,
    parseInt(inpN.value),
    parseFloat(inpNoise.value)
  );
  statusEl.textContent = '';
  redraw();
});

btnTrain.addEventListener('click', function() {
  points = generateDataset(
    selDataset.value as DatasetType,
    parseInt(inpN.value),
    parseFloat(inpNoise.value)
  );
  trainedPoints = points.slice();
  const kernel = selKernel.value as KernelType;
  const C = parseFloat(inpC.value);
  const gamma = parseFloat(inpGamma.value);
  const degree = parseInt(inpDegree.value);

  statusEl.textContent = 'Training…';
  btnTrain.disabled = true;
  setTimeout(function() {
    svm = new SVM(C, kernel, gamma, degree);
    svm.train(trainedPoints, 300);
    redraw();
    statusEl.textContent = 'Done.';
    btnTrain.disabled = false;
  }, 20);
});

// ── Init ───────────────────────────────────────────────────────────────────
updateKernelUI();
lblC.textContent = parseFloat(inpC.value).toFixed(2);
lblGamma.textContent = parseFloat(inpGamma.value).toFixed(2);
lblDegree.textContent = inpDegree.value;
lblN.textContent = inpN.value;
lblNoise.textContent = parseFloat(inpNoise.value).toFixed(2);

points = generateDataset('blobs', parseInt(inpN.value), parseFloat(inpNoise.value));
redraw();
