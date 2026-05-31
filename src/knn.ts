export {};

// ── types ──────────────────────────────────────────────────────────────────
interface Point { x: number; y: number; label: number; }

type Metric = "euclidean" | "manhattan" | "chebyshev";
type Dataset = "blobs" | "moons" | "circles" | "xor" | "spiral" | "three-class";

// ── state ──────────────────────────────────────────────────────────────────
let trainData: Point[] = [];
let testData:  Point[] = [];
let k          = 5;
let metric: Metric  = "euclidean";
let weighted   = false;
let nPoints    = 200;
let noise      = 0.1;
let dataset: Dataset = "blobs";
let queryPoint: {x: number; y: number} | null = null;

// canvas / context refs
let mainCanvas: HTMLCanvasElement;
let mainCtx:    CanvasRenderingContext2D;
let curveCanvas: HTMLCanvasElement;
let curveCtx:   CanvasRenderingContext2D;

const GRID = 80;  // decision boundary grid resolution

// ── colour palette (per class) ──────────────────────────────────────────────
const PALETTE = [
  { fill: "#6366f1", region: "rgba(99,102,241,0.18)",  border: "#818cf8" },
  { fill: "#f43f5e", region: "rgba(244,63,94,0.18)",   border: "#fb7185" },
  { fill: "#10b981", region: "rgba(16,185,129,0.18)",  border: "#34d399" },
  { fill: "#f59e0b", region: "rgba(245,158,11,0.18)",  border: "#fbbf24" },
];

// ── distance functions ─────────────────────────────────────────────────────
function dist(a: {x:number;y:number}, b: {x:number;y:number}, m: Metric): number {
  const dx = a.x - b.x, dy = a.y - b.y;
  if (m === "euclidean")  return Math.sqrt(dx*dx + dy*dy);
  if (m === "manhattan")  return Math.abs(dx) + Math.abs(dy);
  // chebyshev
  return Math.max(Math.abs(dx), Math.abs(dy));
}

// ── k-NN classify ─────────────────────────────────────────────────────────
function classify(
  q: {x:number;y:number},
  data: Point[],
  kk: number,
  m: Metric,
  w: boolean
): { label: number; votes: number[]; neighbors: Point[] } {
  const sorted = data.slice().sort((a, b) => dist(q, a, m) - dist(q, b, m));
  const neighbors = sorted.slice(0, kk);
  const nClasses = Math.max(...data.map(p => p.label)) + 1;
  const votes = new Array(nClasses).fill(0);
  for (const nb of neighbors) {
    const d = dist(q, nb, m);
    const weight = w ? (d < 1e-9 ? 1e9 : 1 / (d * d)) : 1;
    votes[nb.label] += weight;
  }
  let label = 0;
  for (let i = 1; i < votes.length; i++) if (votes[i] > votes[label]) label = i;
  return { label, votes, neighbors };
}

// ── dataset generators ────────────────────────────────────────────────────
function seededRand(seed: number) {
  // simple LCG
  let s = seed >>> 0;
  return () => { s = (1664525 * s + 1013904223) >>> 0; return s / 4294967296; };
}

function addNoise(v: number, n: number, rng: () => number): number {
  return v + (rng() * 2 - 1) * n;
}

function generateData(type: Dataset, n: number, ns: number, seed = 42): Point[] {
  const rng = seededRand(seed);
  const pts: Point[] = [];

  if (type === "blobs") {
    const centers = [{x:0.3,y:0.3},{x:0.7,y:0.7},{x:0.3,y:0.7},{x:0.7,y:0.3}];
    const nc = 2;
    for (let i = 0; i < n; i++) {
      const c = i % nc;
      pts.push({ x: addNoise(centers[c].x, 0.12 + ns * 0.2, rng), y: addNoise(centers[c].y, 0.12 + ns * 0.2, rng), label: c });
    }
  } else if (type === "moons") {
    for (let i = 0; i < n; i++) {
      const t = (i / n) * Math.PI;
      const label = i < n / 2 ? 0 : 1;
      const cx = label === 0 ? 0.5 + 0.35 * Math.cos(t) : 0.5 - 0.35 * Math.cos(t);
      const cy = label === 0 ? 0.5 - 0.18 * Math.sin(t) : 0.5 + 0.18 * Math.sin(t);
      pts.push({ x: addNoise(cx, ns * 0.15, rng), y: addNoise(cy, ns * 0.15, rng), label });
    }
  } else if (type === "circles") {
    for (let i = 0; i < n; i++) {
      const label = i < n / 2 ? 0 : 1;
      const r = label === 0 ? 0.15 : 0.32;
      const t = rng() * 2 * Math.PI;
      pts.push({ x: addNoise(0.5 + r * Math.cos(t), ns * 0.07, rng), y: addNoise(0.5 + r * Math.sin(t), ns * 0.07, rng), label });
    }
  } else if (type === "xor") {
    for (let i = 0; i < n; i++) {
      const x = rng();
      const y = rng();
      const label = ((x > 0.5) !== (y > 0.5)) ? 1 : 0;
      pts.push({ x: addNoise(x, ns * 0.08, rng), y: addNoise(y, ns * 0.08, rng), label });
    }
  } else if (type === "spiral") {
    const nPerClass = Math.floor(n / 2);
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < nPerClass; i++) {
        const t = (i / nPerClass) * 3 * Math.PI + c * Math.PI;
        const r = 0.05 + 0.38 * (i / nPerClass);
        pts.push({ x: addNoise(0.5 + r * Math.cos(t), ns * 0.06, rng), y: addNoise(0.5 + r * Math.sin(t), ns * 0.06, rng), label: c });
      }
    }
  } else if (type === "three-class") {
    const centers = [{x:0.25,y:0.75},{x:0.75,y:0.75},{x:0.5,y:0.25}];
    for (let i = 0; i < n; i++) {
      const c = i % 3;
      pts.push({ x: addNoise(centers[c].x, 0.1 + ns * 0.18, rng), y: addNoise(centers[c].y, 0.1 + ns * 0.18, rng), label: c });
    }
  }

  // clamp
  for (const p of pts) { p.x = Math.max(0.01, Math.min(0.99, p.x)); p.y = Math.max(0.01, Math.min(0.99, p.y)); }
  return pts;
}

function splitTrainTest(pts: Point[], trainFrac = 0.8, seed = 7): [Point[], Point[]] {
  const rng = seededRand(seed);
  const shuffled = pts.slice().sort(() => rng() - 0.5);
  const n = Math.floor(pts.length * trainFrac);
  return [shuffled.slice(0, n), shuffled.slice(n)];
}

// ── accuracy ──────────────────────────────────────────────────────────────
function accuracy(data: Point[], kk: number, m: Metric, w: boolean): number {
  if (data.length === 0) return 0;
  let correct = 0;
  for (const p of data) {
    const { label } = classify(p, trainData, kk, m, w);
    if (label === p.label) correct++;
  }
  return correct / data.length;
}

// ── k-accuracy curve data ─────────────────────────────────────────────────
function kAccuracyCurve(maxK: number): { k: number; train: number; test: number }[] {
  const results: { k: number; train: number; test: number }[] = [];
  for (let kk = 1; kk <= maxK; kk += (maxK > 20 ? 2 : 1)) {
    results.push({ k: kk, train: accuracy(trainData, kk, metric, weighted), test: accuracy(testData, kk, metric, weighted) });
  }
  return results;
}

// ── draw decision boundary ────────────────────────────────────────────────
function drawDecisionBoundary() {
  const W = mainCanvas.width, H = mainCanvas.height;
  mainCtx.clearRect(0, 0, W, H);

  if (trainData.length === 0) return;

  // draw grid cells
  const cw = W / GRID, ch = H / GRID;
  for (let gi = 0; gi < GRID; gi++) {
    for (let gj = 0; gj < GRID; gj++) {
      const qx = (gi + 0.5) / GRID, qy = (gj + 0.5) / GRID;
      const { label } = classify({ x: qx, y: qy }, trainData, k, metric, weighted);
      mainCtx.fillStyle = PALETTE[label % PALETTE.length].region;
      mainCtx.fillRect(gi * cw, gj * ch, cw + 1, ch + 1);
    }
  }

  // draw training points
  for (const p of testData) {
    const px = p.x * W, py = p.y * H;
    const col = PALETTE[p.label % PALETTE.length];
    mainCtx.beginPath();
    mainCtx.arc(px, py, 5, 0, 2 * Math.PI);
    mainCtx.fillStyle = "#1a1a2e";
    mainCtx.fill();
    mainCtx.strokeStyle = col.border;
    mainCtx.lineWidth = 1.5;
    mainCtx.setLineDash([3, 2]);
    mainCtx.stroke();
    mainCtx.setLineDash([]);
  }

  for (const p of trainData) {
    const px = p.x * W, py = p.y * H;
    const col = PALETTE[p.label % PALETTE.length];
    mainCtx.beginPath();
    mainCtx.arc(px, py, 5, 0, 2 * Math.PI);
    mainCtx.fillStyle = col.fill;
    mainCtx.fill();
    mainCtx.strokeStyle = "#ffffff33";
    mainCtx.lineWidth = 0.8;
    mainCtx.stroke();
  }

  // draw query point and neighbor lines
  if (queryPoint) {
    const { neighbors, label, votes } = classify(queryPoint, trainData, k, metric, weighted);
    const qpx = queryPoint.x * W, qpy = queryPoint.y * H;

    // lines to neighbors
    for (const nb of neighbors) {
      mainCtx.beginPath();
      mainCtx.moveTo(qpx, qpy);
      mainCtx.lineTo(nb.x * W, nb.y * H);
      mainCtx.strokeStyle = "rgba(255,255,255,0.35)";
      mainCtx.lineWidth = 1;
      mainCtx.stroke();
    }

    // highlight neighbors
    for (const nb of neighbors) {
      mainCtx.beginPath();
      mainCtx.arc(nb.x * W, nb.y * H, 8, 0, 2 * Math.PI);
      mainCtx.strokeStyle = "#ffffff88";
      mainCtx.lineWidth = 2;
      mainCtx.stroke();
    }

    // query dot
    mainCtx.beginPath();
    mainCtx.arc(qpx, qpy, 7, 0, 2 * Math.PI);
    mainCtx.fillStyle = PALETTE[label % PALETTE.length].fill;
    mainCtx.fill();
    mainCtx.strokeStyle = "#fff";
    mainCtx.lineWidth = 2;
    mainCtx.stroke();

    // vote tooltip
    const totalVotes = votes.reduce((a, b) => a + b, 0);
    const lines: string[] = [`Predicted: Class ${label}`];
    for (let i = 0; i < votes.length; i++) {
      if (votes[i] > 0) lines.push(`  C${i}: ${(100 * votes[i] / totalVotes).toFixed(0)}%`);
    }
    const tw = 130, th = lines.length * 16 + 12;
    let tx = qpx + 14, ty = qpy - th / 2;
    if (tx + tw > W) tx = qpx - tw - 14;
    if (ty < 4) ty = 4;
    mainCtx.fillStyle = "rgba(20,20,40,0.88)";
    mainCtx.strokeStyle = "#ffffff22";
    mainCtx.lineWidth = 1;
    mainCtx.beginPath();
    mainCtx.roundRect(tx, ty, tw, th, 5);
    mainCtx.fill();
    mainCtx.stroke();
    mainCtx.fillStyle = "#e0e0f0";
    mainCtx.font = "12px 'Segoe UI', monospace";
    for (let i = 0; i < lines.length; i++) {
      mainCtx.fillText(lines[i], tx + 8, ty + 14 + i * 16);
    }
  }
}

// ── draw k-accuracy curve ─────────────────────────────────────────────────
function drawKCurve() {
  const W = curveCanvas.width, H = curveCanvas.height;
  curveCtx.clearRect(0, 0, W, H);

  const maxK = Math.min(trainData.length, 25);
  if (maxK < 1) return;

  const data = kAccuracyCurve(maxK);
  const pad = { l: 40, r: 16, t: 14, b: 32 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  // axes
  curveCtx.strokeStyle = "#ffffff22";
  curveCtx.lineWidth = 1;
  curveCtx.beginPath();
  curveCtx.moveTo(pad.l, pad.t);
  curveCtx.lineTo(pad.l, pad.t + ph);
  curveCtx.lineTo(pad.l + pw, pad.t + ph);
  curveCtx.stroke();

  // y labels
  curveCtx.fillStyle = "#8888aa";
  curveCtx.font = "10px 'Segoe UI'";
  curveCtx.textAlign = "right";
  for (let v = 0; v <= 100; v += 25) {
    const y = pad.t + ph - (v / 100) * ph;
    curveCtx.fillText(v + "%", pad.l - 4, y + 3);
    curveCtx.strokeStyle = "#ffffff0a";
    curveCtx.beginPath();
    curveCtx.moveTo(pad.l, y);
    curveCtx.lineTo(pad.l + pw, y);
    curveCtx.stroke();
  }

  // x labels
  curveCtx.textAlign = "center";
  const kVals = data.map(d => d.k);
  const kMin = kVals[0], kMax = kVals[kVals.length - 1];
  const xOf = (kk: number) => pad.l + ((kk - kMin) / (kMax - kMin || 1)) * pw;
  const yOf = (v: number) => pad.t + ph - v * ph;
  for (const d of data) {
    if (d.k === 1 || d.k % 5 === 0 || d.k === kMax) {
      curveCtx.fillStyle = "#8888aa";
      curveCtx.fillText(String(d.k), xOf(d.k), pad.t + ph + 16);
    }
  }
  curveCtx.fillStyle = "#8888aa";
  curveCtx.fillText("k", pad.l + pw / 2, H - 2);

  // train line
  curveCtx.beginPath();
  curveCtx.strokeStyle = "#6366f1";
  curveCtx.lineWidth = 2;
  for (let i = 0; i < data.length; i++) {
    const x = xOf(data[i].k), y = yOf(data[i].train);
    i === 0 ? curveCtx.moveTo(x, y) : curveCtx.lineTo(x, y);
  }
  curveCtx.stroke();

  // test line
  curveCtx.beginPath();
  curveCtx.strokeStyle = "#f43f5e";
  curveCtx.lineWidth = 2;
  for (let i = 0; i < data.length; i++) {
    const x = xOf(data[i].k), y = yOf(data[i].test);
    i === 0 ? curveCtx.moveTo(x, y) : curveCtx.lineTo(x, y);
  }
  curveCtx.stroke();

  // current k marker
  const curX = xOf(k);
  curveCtx.strokeStyle = "#ffffff55";
  curveCtx.lineWidth = 1;
  curveCtx.setLineDash([3, 3]);
  curveCtx.beginPath();
  curveCtx.moveTo(curX, pad.t);
  curveCtx.lineTo(curX, pad.t + ph);
  curveCtx.stroke();
  curveCtx.setLineDash([]);

  // legend
  curveCtx.font = "10px 'Segoe UI'";
  curveCtx.textAlign = "left";
  curveCtx.fillStyle = "#6366f1";
  curveCtx.fillText("▬ Train", pad.l + 4, pad.t + 12);
  curveCtx.fillStyle = "#f43f5e";
  curveCtx.fillText("▬ Test", pad.l + 60, pad.t + 12);
}

// ── update accuracy display ────────────────────────────────────────────────
function updateAccuracy() {
  const trAcc = accuracy(trainData, k, metric, weighted);
  const teAcc = accuracy(testData,  k, metric, weighted);
  const el = document.getElementById("accuracy-display");
  if (el) el.textContent = `Train: ${(trAcc * 100).toFixed(1)}%   Test: ${(teAcc * 100).toFixed(1)}%`;
}

// ── full redraw ────────────────────────────────────────────────────────────
function redraw() {
  drawDecisionBoundary();
  drawKCurve();
  updateAccuracy();
}

// ── regenerate dataset ─────────────────────────────────────────────────────
function regenerate(newSeed?: number) {
  const seed = newSeed ?? Math.floor(Math.random() * 100000);
  const all = generateData(dataset, nPoints, noise, seed);
  [trainData, testData] = splitTrainTest(all, 0.8, seed + 1);
  queryPoint = null;
  redraw();
}

// ── canvas mouse handling ──────────────────────────────────────────────────
function canvasCoord(e: MouseEvent, canvas: HTMLCanvasElement): {x: number; y: number} {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  return { x: ((e.clientX - rect.left) * scaleX) / canvas.width, y: ((e.clientY - rect.top) * scaleY) / canvas.height };
}

// ── init ───────────────────────────────────────────────────────────────────
function init() {
  mainCanvas  = document.getElementById("main-canvas")  as HTMLCanvasElement;
  curveCanvas = document.getElementById("curve-canvas") as HTMLCanvasElement;
  mainCtx  = mainCanvas.getContext("2d")!;
  curveCtx = curveCanvas.getContext("2d")!;

  // controls
  const datasetSel   = document.getElementById("dataset-sel")   as HTMLSelectElement;
  const kSlider      = document.getElementById("k-slider")      as HTMLInputElement;
  const kVal         = document.getElementById("k-val")         as HTMLSpanElement;
  const metricSel    = document.getElementById("metric-sel")    as HTMLSelectElement;
  const weightToggle = document.getElementById("weight-toggle") as HTMLInputElement;
  const nSlider      = document.getElementById("n-slider")      as HTMLInputElement;
  const nVal         = document.getElementById("n-val")         as HTMLSpanElement;
  const noiseSlider  = document.getElementById("noise-slider")  as HTMLInputElement;
  const noiseVal     = document.getElementById("noise-val")     as HTMLSpanElement;
  const regenBtn     = document.getElementById("regen-btn")     as HTMLButtonElement;

  datasetSel.addEventListener("change", () => { dataset = datasetSel.value as Dataset; regenerate(); });
  kSlider.addEventListener("input", () => { k = parseInt(kSlider.value); kVal.textContent = String(k); redraw(); });
  metricSel.addEventListener("change", () => { metric = metricSel.value as Metric; redraw(); });
  weightToggle.addEventListener("change", () => { weighted = weightToggle.checked; redraw(); });
  nSlider.addEventListener("input", () => { nPoints = parseInt(nSlider.value); nVal.textContent = String(nPoints); regenerate(); });
  noiseSlider.addEventListener("input", () => { noise = parseFloat(noiseSlider.value); noiseVal.textContent = noise.toFixed(2); regenerate(); });
  regenBtn.addEventListener("click", () => regenerate());

  mainCanvas.addEventListener("mousemove", (e) => {
    queryPoint = canvasCoord(e, mainCanvas);
    drawDecisionBoundary();
  });
  mainCanvas.addEventListener("click", (e) => {
    queryPoint = canvasCoord(e, mainCanvas);
    drawDecisionBoundary();
  });
  mainCanvas.addEventListener("mouseleave", () => {
    queryPoint = null;
    drawDecisionBoundary();
  });

  regenerate(42);
}

document.addEventListener("DOMContentLoaded", init);
