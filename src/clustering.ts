import * as d3 from 'd3';

// ─── Types ──────────────────────────────────────────────────────────────────

interface Point {
  x: number;
  y: number;
  cluster: number;   // -1 = noise (DBSCAN), 0-based otherwise
  role?: 'core' | 'border' | 'noise';  // DBSCAN
}

interface KMeansCentroid {
  x: number;
  y: number;
  trail: { x: number; y: number }[];
}

interface GMMComponent {
  mean: [number, number];
  cov: [[number, number], [number, number]];
  weight: number;
  // responsibility for each point cached during E step
}

type Algorithm = 'kmeans' | 'dbscan' | 'gmm';
type DatasetName = 'blobs' | 'moons' | 'circles' | 'aniso' | 'uniform';

// ─── Random helpers ──────────────────────────────────────────────────────────

function randn(): number {
  // Box-Muller
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u + 1e-10)) * Math.cos(2 * Math.PI * v);
}

function randnPair(mx: number, my: number, sx: number, sy: number, rho: number): [number, number] {
  const z1 = randn(), z2 = randn();
  const x = mx + sx * z1;
  const y = my + sy * (rho * z1 + Math.sqrt(1 - rho * rho) * z2);
  return [x, y];
}

// ─── Dataset generators (output in roughly [-3, 3]²) ────────────────────────

function genBlobs(n: number): Point[] {
  const centers: [number, number][] = [[-1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [1.5, -1.5]];
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const c = centers[i % centers.length];
    pts.push({ x: c[0] + randn() * 0.5, y: c[1] + randn() * 0.5, cluster: -1 });
  }
  return pts;
}

function genMoons(n: number): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const half = i < n / 2;
    const t = Math.PI * Math.random();
    if (half) {
      pts.push({ x: Math.cos(t) * 1.5, y: Math.sin(t) * 1.5 + randn() * 0.15, cluster: -1 });
    } else {
      pts.push({ x: 1.5 - Math.cos(t) * 1.5, y: -Math.sin(t) * 1.5 + 0.5 + randn() * 0.15, cluster: -1 });
    }
  }
  return pts;
}

function genCircles(n: number): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const inner = i < n / 2;
    const r = inner ? 0.6 + randn() * 0.08 : 1.5 + randn() * 0.08;
    const t = Math.random() * 2 * Math.PI;
    pts.push({ x: r * Math.cos(t), y: r * Math.sin(t), cluster: -1 });
  }
  return pts;
}

function genAniso(n: number): Point[] {
  const pts: Point[] = [];
  const groups: [number, number, number, number, number][] = [
    [-1.5, 0, 1.2, 0.15, 0.8],
    [1.5, 0, 1.2, 0.15, -0.8],
    [0, 1.5, 0.15, 1.2, 0.0],
  ];
  for (let i = 0; i < n; i++) {
    const g = groups[i % groups.length];
    const [x, y] = randnPair(g[0], g[1], g[2], g[3], g[4]);
    pts.push({ x, y, cluster: -1 });
  }
  return pts;
}

function genUniformOutliers(n: number): Point[] {
  const pts: Point[] = [];
  const nOutliers = Math.max(1, Math.floor(n * 0.1));
  const nBlob = n - nOutliers;
  const centers: [number, number][] = [[-1, -1], [1, 1], [0, 1.5]];
  for (let i = 0; i < nBlob; i++) {
    const c = centers[i % centers.length];
    pts.push({ x: c[0] + randn() * 0.4, y: c[1] + randn() * 0.4, cluster: -1 });
  }
  for (let i = 0; i < nOutliers; i++) {
    pts.push({ x: (Math.random() - 0.5) * 5, y: (Math.random() - 0.5) * 5, cluster: -1 });
  }
  return pts;
}

function generateDataset(name: DatasetName, n: number): Point[] {
  switch (name) {
    case 'blobs':   return genBlobs(n);
    case 'moons':   return genMoons(n);
    case 'circles': return genCircles(n);
    case 'aniso':   return genAniso(n);
    case 'uniform': return genUniformOutliers(n);
  }
}

// ─── K-Means ─────────────────────────────────────────────────────────────────

function dist2(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}

function kmeansInit(pts: Point[], k: number): KMeansCentroid[] {
  // k-means++ init
  const centroids: KMeansCentroid[] = [];
  const first = pts[Math.floor(Math.random() * pts.length)];
  centroids.push({ x: first.x, y: first.y, trail: [] });
  while (centroids.length < k) {
    const dists = pts.map(p => Math.min(...centroids.map(c => dist2(p, c))));
    const sum = dists.reduce((a, b) => a + b, 0);
    let r = Math.random() * sum;
    let idx = 0;
    for (let i = 0; i < dists.length; i++) {
      r -= dists[i];
      if (r <= 0) { idx = i; break; }
    }
    const chosen = pts[idx];
    centroids.push({ x: chosen.x, y: chosen.y, trail: [] });
  }
  return centroids;
}

// returns true if assignments changed
function kmeansAssign(pts: Point[], centroids: KMeansCentroid[]): boolean {
  let changed = false;
  for (const p of pts) {
    let best = 0, bestD = Infinity;
    centroids.forEach((c, i) => {
      const d = dist2(p, c);
      if (d < bestD) { bestD = d; best = i; }
    });
    if (p.cluster !== best) { changed = true; p.cluster = best; }
  }
  return changed;
}

function kmeansUpdate(pts: Point[], centroids: KMeansCentroid[]): void {
  centroids.forEach((c, i) => {
    const group = pts.filter(p => p.cluster === i);
    if (group.length === 0) return;
    const nx = group.reduce((s, p) => s + p.x, 0) / group.length;
    const ny = group.reduce((s, p) => s + p.y, 0) / group.length;
    c.trail.push({ x: c.x, y: c.y });
    if (c.trail.length > 20) c.trail.shift();
    c.x = nx; c.y = ny;
  });
}

function kmeansInertia(pts: Point[], centroids: KMeansCentroid[]): number {
  return pts.reduce((s, p) => s + dist2(p, centroids[p.cluster] ?? centroids[0]), 0);
}

// ─── DBSCAN ──────────────────────────────────────────────────────────────────

function dbscan(pts: Point[], eps: number, minPts: number): void {
  const eps2 = eps * eps;
  pts.forEach(p => { p.cluster = -1; p.role = 'noise'; });

  function neighbors(idx: number): number[] {
    return pts.reduce((acc: number[], p, i) => {
      if (dist2(pts[idx], p) <= eps2) acc.push(i);
      return acc;
    }, []);
  }

  let clusterId = 0;
  const visited = new Set<number>();

  for (let i = 0; i < pts.length; i++) {
    if (visited.has(i)) continue;
    visited.add(i);
    const nbrs = neighbors(i);
    if (nbrs.length < minPts) {
      pts[i].role = 'noise';
      continue;
    }
    pts[i].role = 'core';
    pts[i].cluster = clusterId;
    const queue = nbrs.filter(j => j !== i);
    while (queue.length > 0) {
      const j = queue.shift()!;
      if (!visited.has(j)) {
        visited.add(j);
        const nbrs2 = neighbors(j);
        if (nbrs2.length >= minPts) {
          pts[j].role = 'core';
          queue.push(...nbrs2.filter(x => !visited.has(x)));
        } else {
          pts[j].role = 'border';
        }
      }
      if (pts[j].cluster === -1) pts[j].cluster = clusterId;
    }
    clusterId++;
  }
}

// ─── GMM / EM ────────────────────────────────────────────────────────────────

function mat2Inv(m: [[number, number], [number, number]]): [[number, number], [number, number]] {
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  const d = det === 0 ? 1e-10 : det;
  return [
    [m[1][1] / d, -m[0][1] / d],
    [-m[1][0] / d, m[0][0] / d],
  ];
}

function mat2Det(m: [[number, number], [number, number]]): number {
  return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

function gaussian2D(
  x: number, y: number,
  mean: [number, number],
  cov: [[number, number], [number, number]]
): number {
  const inv = mat2Inv(cov);
  const det = mat2Det(cov);
  const dx = x - mean[0], dy = y - mean[1];
  const mah = inv[0][0] * dx * dx + (inv[0][1] + inv[1][0]) * dx * dy + inv[1][1] * dy * dy;
  const denom = 2 * Math.PI * Math.sqrt(Math.abs(det) + 1e-10);
  return Math.exp(-0.5 * mah) / denom;
}

function gmmInit(pts: Point[], k: number): GMMComponent[] {
  // Use k-means-style random init for means
  const shuffled = [...pts].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, k).map(p => ({
    mean: [p.x, p.y] as [number, number],
    cov: [[0.5, 0], [0, 0.5]] as [[number, number], [number, number]],
    weight: 1 / k,
  }));
}

function gmmEStep(pts: Point[], components: GMMComponent[]): number[][] {
  // returns responsibilities[i][k]
  return pts.map(p => {
    const raw = components.map(c => c.weight * gaussian2D(p.x, p.y, c.mean, c.cov));
    const sum = raw.reduce((a, b) => a + b, 1e-300);
    return raw.map(r => r / sum);
  });
}

function gmmMStep(pts: Point[], components: GMMComponent[], resp: number[][]): void {
  const N = pts.length;
  components.forEach((c, k) => {
    const Nk = resp.reduce((s, r) => s + r[k], 0) + 1e-10;
    const mx = resp.reduce((s, r, i) => s + r[k] * pts[i].x, 0) / Nk;
    const my = resp.reduce((s, r, i) => s + r[k] * pts[i].y, 0) / Nk;
    c.mean = [mx, my];
    let c00 = 0, c01 = 0, c11 = 0;
    pts.forEach((p, i) => {
      const dx = p.x - mx, dy = p.y - my;
      c00 += resp[i][k] * dx * dx;
      c01 += resp[i][k] * dx * dy;
      c11 += resp[i][k] * dy * dy;
    });
    c.cov = [
      [c00 / Nk + 1e-3, c01 / Nk],
      [c01 / Nk, c11 / Nk + 1e-3],
    ];
    c.weight = Nk / N;
  });
}

function gmmAssign(pts: Point[], resp: number[][]): void {
  pts.forEach((p, i) => {
    p.cluster = resp[i].indexOf(Math.max(...resp[i]));
  });
}

// ─── Silhouette estimate (sampled for performance) ───────────────────────────

function silhouetteSample(pts: Point[], maxSample = 200): number {
  const seenC: Record<number, boolean> = {};
  pts.forEach(p => { seenC[p.cluster] = true; });
  const clusters = Object.keys(seenC).map(Number).filter(c => c >= 0);
  if (clusters.length < 2) return 0;
  const sample = pts.filter(p => p.cluster >= 0)
    .sort(() => Math.random() - 0.5)
    .slice(0, maxSample);
  let total = 0;
  for (const p of sample) {
    const intraD = sample.filter(q => q !== p && q.cluster === p.cluster)
      .map(q => Math.sqrt(dist2(p, q)));
    const a = intraD.length > 0 ? intraD.reduce((s, d) => s + d, 0) / intraD.length : 0;
    let b = Infinity;
    for (const c of clusters) {
      if (c === p.cluster) continue;
      const interD = sample.filter(q => q.cluster === c)
        .map(q => Math.sqrt(dist2(p, q)));
      if (interD.length > 0) {
        const mean = interD.reduce((s, d) => s + d, 0) / interD.length;
        if (mean < b) b = mean;
      }
    }
    const s = b === Infinity ? 0 : (b - a) / Math.max(a, b);
    total += s;
  }
  return total / sample.length;
}

// ─── Colour palette ──────────────────────────────────────────────────────────

const PALETTE = [
  '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
  '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4',
];

function clusterColor(id: number): string {
  if (id < 0) return '#888888';
  return PALETTE[id % PALETTE.length];
}

// ─── App state ───────────────────────────────────────────────────────────────

interface AppState {
  pts: Point[];
  algorithm: Algorithm;
  dataset: DatasetName;
  k: number;
  eps: number;
  minPts: number;
  nPoints: number;
  running: boolean;
  iter: number;
  // kmeans
  centroids: KMeansCentroid[];
  kmDone: boolean;
  // dbscan (one-shot)
  dbDone: boolean;
  // gmm
  components: GMMComponent[];
  gmmResp: number[][];
  gmmPhase: 'E' | 'M';
  gmmDone: boolean;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

function main() {
  // ── DOM references ──────────────────────────────────────────────────────────
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;

  const selAlgo = document.getElementById('sel-algo') as HTMLSelectElement;
  const selData = document.getElementById('sel-data') as HTMLSelectElement;
  const sliderK = document.getElementById('slider-k') as HTMLInputElement;
  const labelK = document.getElementById('label-k') as HTMLSpanElement;
  const sliderEps = document.getElementById('slider-eps') as HTMLInputElement;
  const labelEps = document.getElementById('label-eps') as HTMLSpanElement;
  const sliderMinPts = document.getElementById('slider-minpts') as HTMLInputElement;
  const labelMinPts = document.getElementById('label-minpts') as HTMLSpanElement;
  const sliderN = document.getElementById('slider-n') as HTMLInputElement;
  const labelN = document.getElementById('label-n') as HTMLSpanElement;
  const btnPlay = document.getElementById('btn-play') as HTMLButtonElement;
  const btnPause = document.getElementById('btn-pause') as HTMLButtonElement;
  const btnStep = document.getElementById('btn-step') as HTMLButtonElement;
  const btnReset = document.getElementById('btn-reset') as HTMLButtonElement;
  const spanIter = document.getElementById('span-iter') as HTMLSpanElement;
  const spanMetric = document.getElementById('span-metric') as HTMLSpanElement;
  const rowK = document.getElementById('row-k') as HTMLDivElement;
  const rowEps = document.getElementById('row-eps') as HTMLDivElement;
  const rowMinPts = document.getElementById('row-minpts') as HTMLDivElement;
  const legendDiv = document.getElementById('legend') as HTMLDivElement;

  // ── State ───────────────────────────────────────────────────────────────────
  const state: AppState = {
    pts: [],
    algorithm: 'kmeans',
    dataset: 'blobs',
    k: 3,
    eps: 0.5,
    minPts: 5,
    nPoints: 200,
    running: false,
    iter: 0,
    centroids: [],
    kmDone: false,
    dbDone: false,
    components: [],
    gmmResp: [],
    gmmPhase: 'E',
    gmmDone: false,
  };

  let animId = -1;
  let lastStepTime = 0;
  const STEP_INTERVAL_MS = 600;

  // ── Canvas sizing ────────────────────────────────────────────────────────────
  function resizeCanvas() {
    const container = canvas.parentElement!;
    const size = Math.min(container.clientWidth, container.clientHeight, 600);
    canvas.width = size;
    canvas.height = size;
  }
  resizeCanvas();
  window.addEventListener('resize', () => { resizeCanvas(); draw(); });

  // ── Coordinate transforms ────────────────────────────────────────────────────
  const DATA_RANGE = 3.5;
  function toCanvasX(x: number): number {
    return ((x + DATA_RANGE) / (2 * DATA_RANGE)) * canvas.width;
  }
  function toCanvasY(y: number): number {
    return ((DATA_RANGE - y) / (2 * DATA_RANGE)) * canvas.height;
  }

  // ── UI helpers ───────────────────────────────────────────────────────────────
  function updateParamVisibility() {
    const algo = state.algorithm;
    rowK.style.display = (algo === 'kmeans' || algo === 'gmm') ? 'block' : 'none';
    rowEps.style.display = algo === 'dbscan' ? 'block' : 'none';
    rowMinPts.style.display = algo === 'dbscan' ? 'block' : 'none';
  }

  function updateLegend() {
    legendDiv.innerHTML = '';
    if (state.algorithm === 'dbscan') {
      const roles: { label: string; color: string; shape: string }[] = [
        { label: 'Core', color: '#3498db', shape: '●' },
        { label: 'Border', color: '#f39c12', shape: '◉' },
        { label: 'Noise', color: '#888', shape: '×' },
      ];
      roles.forEach(r => {
        const el = document.createElement('span');
        el.innerHTML = `<span style="color:${r.color};font-size:1.2em">${r.shape}</span> ${r.label}`;
        el.className = 'legend-item';
        legendDiv.appendChild(el);
      });
      return;
    }
    const seenL: Record<number, boolean> = {};
    state.pts.forEach(p => { seenL[p.cluster] = true; });
    const clusters = Object.keys(seenL).map(Number).filter(c => c >= 0).sort((a, b) => a - b);
    clusters.forEach(c => {
      const el = document.createElement('span');
      el.innerHTML = `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${clusterColor(c)};vertical-align:middle;margin-right:4px"></span>Cluster ${c + 1}`;
      el.className = 'legend-item';
      legendDiv.appendChild(el);
    });
  }

  // ── Reset / init ─────────────────────────────────────────────────────────────
  function resetState() {
    state.running = false;
    clearInterval(animId);
    state.iter = 0;
    state.kmDone = false;
    state.dbDone = false;
    state.gmmDone = false;
    state.gmmPhase = 'E';
    state.pts = generateDataset(state.dataset, state.nPoints);
    state.centroids = [];
    state.components = [];
    state.gmmResp = [];

    if (state.algorithm === 'kmeans') {
      state.centroids = kmeansInit(state.pts, state.k);
      kmeansAssign(state.pts, state.centroids);
    } else if (state.algorithm === 'gmm') {
      state.components = gmmInit(state.pts, state.k);
      state.gmmResp = gmmEStep(state.pts, state.components);
      gmmAssign(state.pts, state.gmmResp);
    } else {
      // dbscan: run once on demand (when play/step pressed)
    }
    updateStatus();
    updateLegend();
    draw();
  }

  // ── One step ─────────────────────────────────────────────────────────────────
  function step() {
    if (state.algorithm === 'kmeans') {
      if (state.kmDone) return;
      kmeansUpdate(state.pts, state.centroids);
      const changed = kmeansAssign(state.pts, state.centroids);
      state.iter++;
      if (!changed) { state.kmDone = true; state.running = false; }
    } else if (state.algorithm === 'dbscan') {
      if (state.dbDone) return;
      dbscan(state.pts, state.eps, state.minPts);
      state.iter++;
      state.dbDone = true;
      state.running = false;
    } else if (state.algorithm === 'gmm') {
      if (state.gmmDone) return;
      if (state.gmmPhase === 'E') {
        state.gmmResp = gmmEStep(state.pts, state.components);
        gmmAssign(state.pts, state.gmmResp);
        state.gmmPhase = 'M';
      } else {
        gmmMStep(state.pts, state.components, state.gmmResp);
        state.gmmPhase = 'E';
        state.iter++;
        if (state.iter >= 100) { state.gmmDone = true; state.running = false; }
      }
    }
    updateStatus();
    updateLegend();
    draw();
  }

  function updateStatus() {
    spanIter.textContent = String(state.iter);
    if (state.algorithm === 'kmeans' && state.centroids.length > 0) {
      const inertia = kmeansInertia(state.pts, state.centroids).toFixed(1);
      spanMetric.textContent = `Inertia: ${inertia}`;
    } else if (state.algorithm === 'dbscan' && state.dbDone) {
      const sil = silhouetteSample(state.pts).toFixed(3);
      spanMetric.textContent = `Silhouette: ${sil}`;
    } else if (state.algorithm === 'gmm' && state.gmmResp.length > 0) {
      const sil = silhouetteSample(state.pts).toFixed(3);
      spanMetric.textContent = `Silhouette: ${sil}`;
    } else {
      spanMetric.textContent = '—';
    }
  }

  // ── Draw ──────────────────────────────────────────────────────────────────────
  function draw() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let gx = -3; gx <= 3; gx++) {
      const cx = toCanvasX(gx);
      ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
    }
    for (let gy = -3; gy <= 3; gy++) {
      const cy = toCanvasY(gy);
      ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
    }

    // GMM ellipses (behind points)
    if (state.algorithm === 'gmm' && state.components.length > 0) {
      drawGMMEllipses();
    }

    // K-Means centroid trails
    if (state.algorithm === 'kmeans') {
      state.centroids.forEach((c, i) => {
        if (c.trail.length < 2) return;
        ctx.strokeStyle = clusterColor(i);
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.globalAlpha = 0.4;
        ctx.beginPath();
        c.trail.forEach((t, ti) => {
          const tx = toCanvasX(t.x), ty = toCanvasY(t.y);
          ti === 0 ? ctx.moveTo(tx, ty) : ctx.lineTo(tx, ty);
        });
        ctx.lineTo(toCanvasX(c.x), toCanvasY(c.y));
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      });
    }

    // Points
    const r = Math.max(3, Math.min(6, 500 / state.pts.length));
    for (const p of state.pts) {
      const cx = toCanvasX(p.x), cy = toCanvasY(p.y);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, 2 * Math.PI);

      if (state.algorithm === 'dbscan') {
        switch (p.role) {
          case 'core':
            ctx.fillStyle = clusterColor(p.cluster);
            ctx.fill();
            ctx.strokeStyle = 'rgba(255,255,255,0.6)';
            ctx.lineWidth = 1;
            ctx.stroke();
            break;
          case 'border':
            ctx.fillStyle = clusterColor(p.cluster);
            ctx.globalAlpha = 0.5;
            ctx.fill();
            ctx.globalAlpha = 1;
            ctx.strokeStyle = clusterColor(p.cluster);
            ctx.lineWidth = 1.5;
            ctx.stroke();
            break;
          case 'noise':
            ctx.fillStyle = '#555';
            ctx.fill();
            // Draw × mark
            ctx.strokeStyle = '#999';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(cx - r, cy - r); ctx.lineTo(cx + r, cy + r);
            ctx.moveTo(cx + r, cy - r); ctx.lineTo(cx - r, cy + r);
            ctx.stroke();
            break;
        }
      } else {
        ctx.fillStyle = clusterColor(p.cluster);
        ctx.fill();
        ctx.strokeStyle = 'rgba(0,0,0,0.3)';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }

    // K-Means centroids
    if (state.algorithm === 'kmeans') {
      state.centroids.forEach((c, i) => {
        const cx = toCanvasX(c.x), cy = toCanvasY(c.y);
        // Outer ring
        ctx.beginPath();
        ctx.arc(cx, cy, 10, 0, 2 * Math.PI);
        ctx.fillStyle = clusterColor(i);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        // Inner star (cross)
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx - 5, cy); ctx.lineTo(cx + 5, cy);
        ctx.moveTo(cx, cy - 5); ctx.lineTo(cx, cy + 5);
        ctx.stroke();
      });
    }

    // GMM means
    if (state.algorithm === 'gmm' && state.components.length > 0) {
      state.components.forEach((c, i) => {
        const cx = toCanvasX(c.mean[0]), cy = toCanvasY(c.mean[1]);
        ctx.beginPath();
        ctx.arc(cx, cy, 8, 0, 2 * Math.PI);
        ctx.fillStyle = clusterColor(i);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    }
  }

  function drawGMMEllipses() {
    const nSigma = [1, 2]; // draw 1σ and 2σ contours
    state.components.forEach((c, i) => {
      // Eigen decomposition of 2x2 covariance
      const { angle, s1, s2 } = eigenDecomp2x2(c.cov);
      const cx = toCanvasX(c.mean[0]);
      const cy = toCanvasY(c.mean[1]);
      const scaleX = (2 * DATA_RANGE / canvas.width);
      const scaleY = (2 * DATA_RANGE / canvas.height);

      for (const ns of nSigma) {
        const rx = Math.sqrt(s1) * ns / scaleX;
        const ry = Math.sqrt(s2) * ns / scaleY;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(-angle); // canvas Y is flipped
        ctx.beginPath();
        ctx.ellipse(0, 0, rx, ry, 0, 0, 2 * Math.PI);
        ctx.restore();
        ctx.strokeStyle = clusterColor(i);
        ctx.lineWidth = ns === 1 ? 2 : 1;
        ctx.globalAlpha = ns === 1 ? 0.8 : 0.4;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });
  }

  function eigenDecomp2x2(m: [[number, number], [number, number]]): { angle: number; s1: number; s2: number } {
    const a = m[0][0], b = m[0][1], d = m[1][1];
    const trace = a + d;
    const det = a * d - b * b;
    const disc = Math.sqrt(Math.max(0, (trace / 2) ** 2 - det));
    const l1 = trace / 2 + disc;
    const l2 = trace / 2 - disc;
    let angle = 0;
    if (Math.abs(b) > 1e-10) {
      angle = Math.atan2(l1 - a, b);
    }
    return { angle, s1: Math.max(l1, 1e-6), s2: Math.max(l2, 1e-6) };
  }

  // ── Animation loop ────────────────────────────────────────────────────────────
  function animLoop(now: number) {
    if (!state.running) return;
    if (now - lastStepTime >= STEP_INTERVAL_MS) {
      lastStepTime = now;
      step();
    }
    animId = requestAnimationFrame(animLoop);
  }

  // ── Event listeners ───────────────────────────────────────────────────────────
  selAlgo.addEventListener('change', () => {
    state.algorithm = selAlgo.value as Algorithm;
    updateParamVisibility();
    resetState();
  });

  selData.addEventListener('change', () => {
    state.dataset = selData.value as DatasetName;
    resetState();
  });

  sliderK.addEventListener('input', () => {
    state.k = parseInt(sliderK.value, 10);
    labelK.textContent = sliderK.value;
    resetState();
  });

  sliderEps.addEventListener('input', () => {
    state.eps = parseFloat(sliderEps.value);
    labelEps.textContent = sliderEps.value;
    resetState();
  });

  sliderMinPts.addEventListener('input', () => {
    state.minPts = parseInt(sliderMinPts.value, 10);
    labelMinPts.textContent = sliderMinPts.value;
    resetState();
  });

  sliderN.addEventListener('input', () => {
    state.nPoints = parseInt(sliderN.value, 10);
    labelN.textContent = sliderN.value;
    resetState();
  });

  btnPlay.addEventListener('click', () => {
    state.running = true;
    lastStepTime = 0;
    animId = requestAnimationFrame(animLoop);
  });

  btnPause.addEventListener('click', () => {
    state.running = false;
    cancelAnimationFrame(animId);
  });

  btnStep.addEventListener('click', () => {
    state.running = false;
    cancelAnimationFrame(animId);
    step();
  });

  btnReset.addEventListener('click', () => {
    resetState();
  });

  // ── Init ──────────────────────────────────────────────────────────────────────
  updateParamVisibility();
  resetState();
}

// ── Entry point ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', main);
