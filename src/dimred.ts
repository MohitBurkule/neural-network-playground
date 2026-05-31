import * as d3 from 'd3';

// ─── Types ────────────────────────────────────────────────────────────────────

interface Point {
  x: number[];   // high-dim
  label: number;
}

interface Embedding {
  x: number;
  y: number;
  label: number;
}

// ─── Math helpers ────────────────────────────────────────────────────────────

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function vecScale(a: number[], s: number): number[] {
  return a.map(function(v) { return v * s; });
}

function vecAdd(a: number[], b: number[]): number[] {
  return a.map(function(v, i) { return v + b[i]; });
}

function vecSub(a: number[], b: number[]): number[] {
  return a.map(function(v, i) { return v - b[i]; });
}

function vecNorm(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

function vecNormalize(a: number[]): number[] {
  const n = vecNorm(a);
  return n === 0 ? a.slice() : vecScale(a, 1 / n);
}

function randNormal(rng: () => number): number {
  // Box-Muller
  const u = rng(), v = rng();
  return Math.sqrt(-2 * Math.log(Math.max(u, 1e-10))) * Math.cos(2 * Math.PI * v);
}

// Seeded LCG RNG
function makeLCG(seed: number): () => number {
  let s = seed >>> 0;
  return function() {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// ─── Dataset generators ──────────────────────────────────────────────────────

function generateClusters(n: number, dim: number): Point[] {
  const rng = makeLCG(42);
  const k = 6;
  // cluster centers
  const centers: number[][] = [];
  for (let c = 0; c < k; c++) {
    const center: number[] = [];
    for (let d = 0; d < dim; d++) center.push(randNormal(rng) * 4);
    centers.push(center);
  }
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % k;
    const c = centers[label];
    const x: number[] = [];
    for (let d = 0; d < dim; d++) x.push(c[d] + randNormal(rng) * 0.7);
    pts.push({ x, label });
  }
  return pts;
}

function generateSwissRoll(n: number, dim: number): Point[] {
  const rng = makeLCG(7);
  const pts: Point[] = [];
  const nLabels = 5;
  for (let i = 0; i < n; i++) {
    const t = 1.5 * Math.PI * (1 + 2 * rng());
    const height = 10 * rng();
    const x3: number[] = [
      t * Math.cos(t),
      height,
      t * Math.sin(t)
    ];
    // pad with small noise to fill `dim`
    const x: number[] = x3.slice();
    for (let d = 3; d < dim; d++) x.push(randNormal(rng) * 0.3);
    const label = Math.floor(((t - 1.5 * Math.PI) / (3 * Math.PI)) * nLabels);
    pts.push({ x, label: Math.min(label, nLabels - 1) });
  }
  return pts;
}

function generateConcentricSpheres(n: number, dim: number): Point[] {
  const rng = makeLCG(13);
  const radii = [1, 2.5, 4];
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % radii.length;
    const r = radii[label];
    // random unit vector in 3D then pad
    const raw: number[] = [randNormal(rng), randNormal(rng), randNormal(rng)];
    const norm = Math.sqrt(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]);
    const x: number[] = raw.map(function(v) { return v / norm * r; });
    for (let d = 3; d < dim; d++) x.push(randNormal(rng) * 0.15);
    pts.push({ x, label });
  }
  return pts;
}

function generateDigits(n: number): Point[] {
  const rng = makeLCG(99);
  // 8 "digit" templates: 8x8 binary patterns (simplified)
  const templates: number[][] = [];
  for (let digit = 0; digit < 8; digit++) {
    const tpl: number[] = [];
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        // create a rough digit-like pattern per class
        let val = 0;
        if (digit === 0) val = (r === 0 || r === 7 || c === 0 || c === 7) ? 1 : 0;
        else if (digit === 1) val = (c === 4) ? 1 : 0;
        else if (digit === 2) val = (r === 0 || r === 3 || r === 7 || (r < 3 && c === 7) || (r > 3 && c === 0)) ? 1 : 0;
        else if (digit === 3) val = (r === 0 || r === 3 || r === 7 || c === 7) ? 1 : 0;
        else if (digit === 4) val = (r < 4 && (c === 0 || c === 7) || r === 3) ? 1 : 0;
        else if (digit === 5) val = (r === 0 || r === 3 || r === 7 || (r < 3 && c === 0) || (r > 3 && c === 7)) ? 1 : 0;
        else if (digit === 6) val = (r === 0 || r === 3 || r === 7 || c === 0 || (r > 3 && c === 7)) ? 1 : 0;
        else if (digit === 7) val = (r === 0 || c === 7) ? 1 : 0;
        tpl.push(val);
      }
    }
    templates.push(tpl);
  }
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % 8;
    const tpl = templates[label];
    const x: number[] = tpl.map(function(v) { return v + randNormal(rng) * 0.2; });
    pts.push({ x, label });
  }
  return pts;
}

function buildDataset(name: string, n: number): Point[] {
  if (name === 'clusters') return generateClusters(n, 8);
  if (name === 'swissroll') return generateSwissRoll(n, 8);
  if (name === 'spheres') return generateConcentricSpheres(n, 8);
  if (name === 'digits') return generateDigits(n);
  return generateClusters(n, 8);
}

// ─── PCA ─────────────────────────────────────────────────────────────────────

function meanCenter(data: number[][]): { centered: number[][], mean: number[] } {
  const n = data.length;
  const dim = data[0].length;
  const mean: number[] = [];
  for (let d = 0; d < dim; d++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += data[i][d];
    mean.push(s / n);
  }
  const centered = data.map(function(row) {
    return row.map(function(v, d) { return v - mean[d]; });
  });
  return { centered, mean };
}

// Power iteration to find top eigenvector of X^T X (covariance)
function powerIter(data: number[][], iters: number): number[] {
  const dim = data[0].length;
  const rng = makeLCG(1234);
  let v: number[] = [];
  for (let d = 0; d < dim; d++) v.push(randNormal(rng));
  v = vecNormalize(v);
  for (let it = 0; it < iters; it++) {
    // Av = X^T (X v)
    const Xv: number[] = data.map(function(row) { return dot(row, v); });
    const newV: number[] = new Array(dim).fill(0);
    for (let i = 0; i < data.length; i++) {
      for (let d = 0; d < dim; d++) newV[d] += data[i][d] * Xv[i];
    }
    v = vecNormalize(newV);
  }
  return v;
}

function projectPCA(points: Point[]): { embeddings: Embedding[], varPC1: number, varPC2: number } {
  const data = points.map(function(p) { return p.x; });
  const { centered } = meanCenter(data);

  const pc1 = powerIter(centered, 100);
  // deflate
  const deflated = centered.map(function(row) {
    const proj = dot(row, pc1);
    return vecSub(row, vecScale(pc1, proj));
  });
  const pc2 = powerIter(deflated, 100);

  // variance explained
  const totalVar = centered.reduce(function(acc, row) {
    return acc + dot(row, row);
  }, 0);
  const var1 = centered.reduce(function(acc, row) {
    const p = dot(row, pc1);
    return acc + p * p;
  }, 0);
  const var2 = deflated.reduce(function(acc, row) {
    const p = dot(row, pc2);
    return acc + p * p;
  }, 0);

  const embeddings: Embedding[] = centered.map(function(row, i) {
    return { x: dot(row, pc1), y: dot(row, pc2), label: points[i].label };
  });
  return { embeddings, varPC1: var1 / totalVar, varPC2: var2 / totalVar };
}

// ─── t-SNE ───────────────────────────────────────────────────────────────────

function squaredDist(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return s;
}

function computePerplexityAffinities(data: number[][], perplexity: number): number[][] {
  const n = data.length;
  const P: number[][] = [];
  const targetH = Math.log(perplexity);

  for (let i = 0; i < n; i++) {
    const dists: number[] = [];
    for (let j = 0; j < n; j++) {
      dists.push(j === i ? 0 : squaredDist(data[i], data[j]));
    }
    // binary search for beta (1/2sigma^2)
    let betaMin = -Infinity, betaMax = Infinity;
    let beta = 1.0;
    const pi: number[] = new Array(n).fill(0);
    for (let iter = 0; iter < 50; iter++) {
      let sumP = 0;
      for (let j = 0; j < n; j++) {
        pi[j] = j === i ? 0 : Math.exp(-dists[j] * beta);
        sumP += pi[j];
      }
      if (sumP === 0) sumP = 1e-10;
      let H = 0;
      for (let j = 0; j < n; j++) {
        if (pi[j] > 1e-15) H += -(pi[j] / sumP) * Math.log(pi[j] / sumP);
      }
      const Hdiff = H - targetH;
      if (Math.abs(Hdiff) < 1e-5) break;
      if (Hdiff > 0) {
        betaMin = beta;
        beta = betaMax === Infinity ? beta * 2 : (beta + betaMax) / 2;
      } else {
        betaMax = beta;
        beta = betaMin === -Infinity ? beta / 2 : (beta + betaMin) / 2;
      }
    }
    let sumP = 0;
    for (let j = 0; j < n; j++) sumP += pi[j];
    P.push(pi.map(function(v) { return v / (sumP || 1e-10); }));
  }

  // Symmetrize
  const Psym: number[][] = [];
  for (let i = 0; i < n; i++) {
    Psym.push(new Array(n).fill(0));
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const val = (P[i][j] + P[j][i]) / (2 * n);
      Psym[i][j] = Math.max(val, 1e-12);
    }
  }
  return Psym;
}

class TSNE {
  n: number;
  P: number[][];
  Y: number[][];       // 2D embedding
  gains: number[][];
  vel: number[][];
  iter: number;
  klCost: number;

  constructor(data: number[][], perplexity: number) {
    this.n = data.length;
    this.iter = 0;
    this.klCost = 0;
    this.P = computePerplexityAffinities(data, perplexity);

    const rng = makeLCG(555);
    this.Y = [];
    this.gains = [];
    this.vel = [];
    for (let i = 0; i < this.n; i++) {
      this.Y.push([randNormal(rng) * 1e-4, randNormal(rng) * 1e-4]);
      this.gains.push([1, 1]);
      this.vel.push([0, 0]);
    }
    // early exaggeration
    for (let i = 0; i < this.n; i++) {
      for (let j = 0; j < this.n; j++) {
        this.P[i][j] *= 4;
      }
    }
  }

  step(lr: number): void {
    const n = this.n;
    this.iter++;
    if (this.iter === 100) {
      // remove early exaggeration
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          this.P[i][j] /= 4;
        }
      }
    }

    // Compute Q (t-distributed)
    const num: number[][] = [];
    let sumQ = 0;
    for (let i = 0; i < n; i++) {
      num.push(new Array(n).fill(0));
      for (let j = i + 1; j < n; j++) {
        const dy0 = this.Y[i][0] - this.Y[j][0];
        const dy1 = this.Y[i][1] - this.Y[j][1];
        const v = 1 / (1 + dy0 * dy0 + dy1 * dy1);
        num[i][j] = v;
        sumQ += v;
      }
    }
    // mirror & finalize
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) {
        num[i][j] = num[j][i];
      }
    }
    sumQ = Math.max(sumQ * 2, 1e-10);

    // Gradient and KL
    let kl = 0;
    const grad: number[][] = [];
    for (let i = 0; i < n; i++) grad.push([0, 0]);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const q = Math.max(num[i][j] / sumQ, 1e-12);
        const p = this.P[i][j];
        if (i < j && p > 1e-12) kl += p * Math.log(p / q);
        const mult = 4 * (p - q) * num[i][j];
        grad[i][0] += mult * (this.Y[i][0] - this.Y[j][0]);
        grad[i][1] += mult * (this.Y[i][1] - this.Y[j][1]);
      }
    }
    this.klCost = kl;

    // momentum + adaptive learning rate
    const momentum = this.iter < 250 ? 0.5 : 0.8;
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 2; d++) {
        // adaptive gain
        if ((grad[i][d] > 0) !== (this.vel[i][d] > 0)) {
          this.gains[i][d] = Math.min(this.gains[i][d] + 0.2, 10);
        } else {
          this.gains[i][d] = Math.max(this.gains[i][d] * 0.8, 0.01);
        }
        this.vel[i][d] = momentum * this.vel[i][d] - lr * this.gains[i][d] * grad[i][d];
        this.Y[i][d] += this.vel[i][d];
      }
    }

    // center
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += this.Y[i][0]; my += this.Y[i][1]; }
    mx /= n; my /= n;
    for (let i = 0; i < n; i++) { this.Y[i][0] -= mx; this.Y[i][1] -= my; }
  }

  getEmbeddings(labels: number[]): Embedding[] {
    return this.Y.map(function(y, i) {
      return { x: y[0], y: y[1], label: labels[i] };
    });
  }
}

// ─── App State ───────────────────────────────────────────────────────────────

interface AppState {
  method: 'pca' | 'tsne';
  dataset: string;
  nPoints: number;
  perplexity: number;
  lr: number;
  points: Point[];
  tsne: TSNE | null;
  embeddings: Embedding[];
  pcaVarPC1: number;
  pcaVarPC2: number;
  running: boolean;
  animFrame: number | null;
}

const state: AppState = {
  method: 'pca',
  dataset: 'clusters',
  nPoints: 150,
  perplexity: 20,
  lr: 100,
  points: [],
  tsne: null,
  embeddings: [],
  pcaVarPC1: 0,
  pcaVarPC2: 0,
  running: false,
  animFrame: null
};

// ─── Color Scale ─────────────────────────────────────────────────────────────

const PALETTE = [
  '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
  '#1abc9c', '#e67e22', '#e91e63'
];

function getColor(label: number): string {
  return PALETTE[label % PALETTE.length];
}

// ─── Canvas render ───────────────────────────────────────────────────────────

function render(canvas: HTMLCanvasElement, infoEl: HTMLElement): void {
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const emb = state.embeddings;
  if (emb.length === 0) return;

  const xs = emb.map(function(e) { return e.x; });
  const ys = emb.map(function(e) { return e.y; });
  const xExt = d3.extent(xs) as [number, number];
  const yExt = d3.extent(ys) as [number, number];

  const pad = 40;
  const scaleX = d3.scaleLinear().domain([xExt[0], xExt[1]]).range([pad, W - pad]);
  const scaleY = d3.scaleLinear().domain([yExt[0], yExt[1]]).range([H - pad, pad]);

  // axes (light grid)
  ctx.strokeStyle = '#2a2a4a';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const gx = pad + (i / 4) * (W - 2 * pad);
    const gy = pad + (i / 4) * (H - 2 * pad);
    ctx.beginPath(); ctx.moveTo(gx, pad); ctx.lineTo(gx, H - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(W - pad, gy); ctx.stroke();
  }

  emb.forEach(function(e) {
    ctx.beginPath();
    ctx.arc(scaleX(e.x), scaleY(e.y), 4.5, 0, Math.PI * 2);
    ctx.fillStyle = getColor(e.label);
    ctx.globalAlpha = 0.82;
    ctx.fill();
    ctx.globalAlpha = 1;
  });

  // info text
  let info = '';
  if (state.method === 'pca') {
    info = 'PC1: ' + (state.pcaVarPC1 * 100).toFixed(1) + '% var   PC2: ' + (state.pcaVarPC2 * 100).toFixed(1) + '% var';
  } else if (state.tsne) {
    info = 'iter: ' + state.tsne.iter + '   KL: ' + state.tsne.klCost.toFixed(4);
  }
  infoEl.textContent = info;
}

// ─── Logic ───────────────────────────────────────────────────────────────────

function stopLoop(): void {
  state.running = false;
  if (state.animFrame !== null) {
    cancelAnimationFrame(state.animFrame);
    state.animFrame = null;
  }
}

function resetEverything(canvas: HTMLCanvasElement, infoEl: HTMLElement): void {
  stopLoop();
  state.points = buildDataset(state.dataset, state.nPoints);
  if (state.method === 'pca') {
    const result = projectPCA(state.points);
    state.embeddings = result.embeddings;
    state.pcaVarPC1 = result.varPC1;
    state.pcaVarPC2 = result.varPC2;
    state.tsne = null;
  } else {
    state.tsne = new TSNE(state.points.map(function(p) { return p.x; }), state.perplexity);
    state.embeddings = state.tsne.getEmbeddings(state.points.map(function(p) { return p.label; }));
  }
  render(canvas, infoEl);
}

function doStep(canvas: HTMLCanvasElement, infoEl: HTMLElement): void {
  if (state.method !== 'tsne' || !state.tsne) return;
  state.tsne.step(state.lr);
  state.embeddings = state.tsne.getEmbeddings(state.points.map(function(p) { return p.label; }));
  render(canvas, infoEl);
}

function startLoop(canvas: HTMLCanvasElement, infoEl: HTMLElement): void {
  if (state.method !== 'tsne') return;
  state.running = true;
  function loop() {
    if (!state.running) return;
    doStep(canvas, infoEl);
    state.animFrame = requestAnimationFrame(loop);
  }
  state.animFrame = requestAnimationFrame(loop);
}

// ─── Legend ──────────────────────────────────────────────────────────────────

function buildLegend(legendEl: HTMLElement): void {
  legendEl.innerHTML = '';
  const labels: number[] = [];
  state.points.forEach(function(p) {
    if (labels.indexOf(p.label) === -1) labels.push(p.label);
  });
  labels.sort(function(a, b) { return a - b; });
  labels.forEach(function(l) {
    const item = document.createElement('div');
    item.style.cssText = 'display:flex;align-items:center;gap:5px;font-size:0.78rem;color:#ccc;';
    const dot = document.createElement('div');
    dot.style.cssText = 'width:10px;height:10px;border-radius:50%;background:' + getColor(l) + ';flex-shrink:0;';
    item.appendChild(dot);
    item.appendChild(document.createTextNode('class ' + l));
    legendEl.appendChild(item);
  });
}

// ─── Main ────────────────────────────────────────────────────────────────────

(function main() {
  const canvas = document.getElementById('main-canvas') as HTMLCanvasElement;
  const infoEl = document.getElementById('info-bar') as HTMLElement;
  const legendEl = document.getElementById('legend') as HTMLElement;

  const selMethod = document.getElementById('sel-method') as HTMLSelectElement;
  const selDataset = document.getElementById('sel-dataset') as HTMLSelectElement;
  const sliderN = document.getElementById('slider-n') as HTMLInputElement;
  const valN = document.getElementById('val-n') as HTMLElement;
  const sliderPerp = document.getElementById('slider-perp') as HTMLInputElement;
  const valPerp = document.getElementById('val-perp') as HTMLElement;
  const sliderLR = document.getElementById('slider-lr') as HTMLInputElement;
  const valLR = document.getElementById('val-lr') as HTMLElement;
  const btnPlay = document.getElementById('btn-play') as HTMLButtonElement;
  const btnPause = document.getElementById('btn-pause') as HTMLButtonElement;
  const btnStep = document.getElementById('btn-step') as HTMLButtonElement;
  const btnReset = document.getElementById('btn-reset') as HTMLButtonElement;
  const tsneControls = document.getElementById('tsne-controls') as HTMLElement;

  function refreshUI(): void {
    valN.textContent = String(state.nPoints);
    valPerp.textContent = String(state.perplexity);
    valLR.textContent = String(state.lr);
    tsneControls.style.display = state.method === 'tsne' ? 'flex' : 'none';
  }

  function rebuild(): void {
    stopLoop();
    resetEverything(canvas, infoEl);
    buildLegend(legendEl);
    refreshUI();
  }

  selMethod.addEventListener('change', function() {
    state.method = selMethod.value as 'pca' | 'tsne';
    rebuild();
  });

  selDataset.addEventListener('change', function() {
    state.dataset = selDataset.value;
    rebuild();
  });

  sliderN.addEventListener('input', function() {
    state.nPoints = Math.min(parseInt(sliderN.value), state.method === 'tsne' ? 200 : 500);
    valN.textContent = String(state.nPoints);
    rebuild();
  });

  sliderPerp.addEventListener('input', function() {
    state.perplexity = parseInt(sliderPerp.value);
    valPerp.textContent = String(state.perplexity);
    if (state.method === 'tsne') rebuild();
  });

  sliderLR.addEventListener('input', function() {
    state.lr = parseInt(sliderLR.value);
    valLR.textContent = String(state.lr);
  });

  btnPlay.addEventListener('click', function() { startLoop(canvas, infoEl); });
  btnPause.addEventListener('click', function() { stopLoop(); });
  btnStep.addEventListener('click', function() { doStep(canvas, infoEl); });
  btnReset.addEventListener('click', function() { rebuild(); });

  // resize canvas to fit container
  function resizeCanvas(): void {
    const panel = canvas.parentElement!;
    canvas.width = panel.clientWidth - 24;
    canvas.height = Math.max(400, Math.min(560, window.innerHeight - 160));
    render(canvas, infoEl);
  }
  window.addEventListener('resize', resizeCanvas);

  // init
  state.method = 'pca';
  state.dataset = 'clusters';
  state.nPoints = 150;
  selMethod.value = 'pca';
  selDataset.value = 'clusters';
  sliderN.value = '150';
  sliderPerp.value = '20';
  sliderLR.value = '100';

  resizeCanvas();
  rebuild();
})();
