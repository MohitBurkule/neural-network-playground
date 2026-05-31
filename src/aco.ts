export {};

// ── Types ──────────────────────────────────────────────────────────────────

interface City { x: number; y: number; }
interface Ant  { tour: number[]; dist: number; }

// ── State ──────────────────────────────────────────────────────────────────

let cities: City[]      = [];
let phero:  number[][]  = [];   // phero[i][j]
let bestTour: number[]  = [];
let bestDist            = Infinity;
let iteration           = 0;
let running             = false;
let timerId: number | null = null;

// ── Parameters ─────────────────────────────────────────────────────────────

let NUM_CITIES  = 20;
let NUM_ANTS    = 30;
let ALPHA       = 1.0;   // pheromone exponent
let BETA        = 3.0;   // heuristic exponent
let RHO         = 0.1;   // evaporation rate
let Q           = 100;   // deposit constant
let SPEED       = 80;    // ms per iteration

// ── Canvas setup ───────────────────────────────────────────────────────────

const mainCanvas  = document.getElementById("main-canvas")  as HTMLCanvasElement;
const chartCanvas = document.getElementById("chart-canvas") as HTMLCanvasElement;
const ctx         = mainCanvas.getContext("2d")!;
const cctx        = chartCanvas.getContext("2d")!;

const W = mainCanvas.width;
const H = mainCanvas.height;
const MARGIN = 40; // canvas inner margin for cities

// ── Helpers ────────────────────────────────────────────────────────────────

function dist(a: City, b: City): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function tourLength(tour: number[]): number {
  let d = 0;
  for (let i = 0; i < tour.length; i++) {
    d += dist(cities[tour[i]], cities[tour[(i + 1) % tour.length]]);
  }
  return d;
}

function initPhero(): void {
  const n = cities.length;
  phero = [];
  for (let i = 0; i < n; i++) {
    phero[i] = new Array(n).fill(1.0);
  }
}

function randomCities(n: number): City[] {
  const result: City[] = [];
  for (let i = 0; i < n; i++) {
    result.push({
      x: MARGIN + Math.random() * (W - 2 * MARGIN),
      y: MARGIN + Math.random() * (H - 2 * MARGIN),
    });
  }
  return result;
}

// ── ACO core ───────────────────────────────────────────────────────────────

function buildTour(): Ant {
  const n = cities.length;
  const visited = new Uint8Array(n);
  const tour: number[] = [];

  // random start city
  let current = Math.floor(Math.random() * n);
  visited[current] = 1;
  tour.push(current);

  for (let step = 1; step < n; step++) {
    // compute weights
    let sum = 0;
    const weights = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      if (visited[j]) continue;
      const d = dist(cities[current], cities[j]);
      if (d === 0) { weights[j] = 0; continue; }
      const w = Math.pow(phero[current][j], ALPHA) * Math.pow(1 / d, BETA);
      weights[j] = w;
      sum += w;
    }

    // roulette selection
    let r = Math.random() * sum;
    let next = -1;
    for (let j = 0; j < n; j++) {
      if (visited[j]) continue;
      r -= weights[j];
      if (r <= 0) { next = j; break; }
    }
    if (next === -1) {
      // fallback: pick first unvisited
      for (let j = 0; j < n; j++) {
        if (!visited[j]) { next = j; break; }
      }
    }

    visited[next] = 1;
    tour.push(next);
    current = next;
  }

  return { tour, dist: tourLength(tour) };
}

function stepACO(): void {
  if (cities.length < 2) return;
  const n = cities.length;

  // build ant tours
  const ants: Ant[] = [];
  for (let k = 0; k < NUM_ANTS; k++) {
    ants.push(buildTour());
  }

  // evaporate
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      phero[i][j] *= (1 - RHO);
      if (phero[i][j] < 1e-6) phero[i][j] = 1e-6;
    }
  }

  // deposit
  for (const ant of ants) {
    const deposit = Q / ant.dist;
    for (let s = 0; s < ant.tour.length; s++) {
      const a = ant.tour[s];
      const b = ant.tour[(s + 1) % ant.tour.length];
      phero[a][b] += deposit;
      phero[b][a] += deposit;
    }
    if (ant.dist < bestDist) {
      bestDist = ant.dist;
      bestTour = ant.tour.slice();
    }
  }

  iteration++;
  distHistory.push(bestDist);
  updateLabels();
  draw();
  drawChart();
}

// ── Distance history for chart ──────────────────────────────────────────────

const distHistory: number[] = [];

// ── Drawing ────────────────────────────────────────────────────────────────

function maxPhero(): number {
  let m = 0;
  for (const row of phero) for (const v of row) if (v > m) m = v;
  return m || 1;
}

function draw(): void {
  ctx.clearRect(0, 0, W, H);

  // background
  ctx.fillStyle = "#0a0a14";
  ctx.fillRect(0, 0, W, H);

  if (cities.length < 2) {
    drawCities();
    return;
  }

  const n = cities.length;
  const maxP = maxPhero();

  // draw pheromone edges
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const p = phero[i][j] / maxP;
      if (p < 0.02) continue;
      const alpha = Math.min(0.9, p * 0.9);
      const width = Math.min(6, 0.5 + p * 5);
      ctx.beginPath();
      ctx.moveTo(cities[i].x, cities[i].y);
      ctx.lineTo(cities[j].x, cities[j].y);
      ctx.strokeStyle = `rgba(56,189,248,${alpha.toFixed(3)})`;
      ctx.lineWidth = width;
      ctx.stroke();
    }
  }

  // draw best tour
  if (bestTour.length === n) {
    ctx.beginPath();
    ctx.moveTo(cities[bestTour[0]].x, cities[bestTour[0]].y);
    for (let i = 1; i < n; i++) {
      ctx.lineTo(cities[bestTour[i]].x, cities[bestTour[i]].y);
    }
    ctx.closePath();
    ctx.strokeStyle = "rgba(251,191,36,0.85)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  drawCities();
}

function drawCities(): void {
  for (let i = 0; i < cities.length; i++) {
    const c = cities[i];
    ctx.beginPath();
    ctx.arc(c.x, c.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#f0f0ff";
    ctx.fill();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.fillStyle = "#0a0a14";
    ctx.font = "bold 9px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    if (cities.length <= 30) ctx.fillText(String(i + 1), c.x, c.y);
  }
}

function drawChart(): void {
  const cw = chartCanvas.width;
  const ch = chartCanvas.height;
  cctx.clearRect(0, 0, cw, ch);

  cctx.fillStyle = "#0a0a14";
  cctx.fillRect(0, 0, cw, ch);

  const data = distHistory;
  if (data.length < 2) return;

  const pad = { top: 12, right: 12, bottom: 28, left: 48 };
  const pw = cw - pad.left - pad.right;
  const ph = ch - pad.top - pad.bottom;

  const minD = Math.min(...data) * 0.95;
  const maxD = Math.max(...data) * 1.05;
  const xScale = (i: number) => pad.left + (i / (data.length - 1)) * pw;
  const yScale = (v: number) => pad.top + ph - ((v - minD) / (maxD - minD)) * ph;

  // axes
  cctx.strokeStyle = "#334";
  cctx.lineWidth = 1;
  cctx.beginPath();
  cctx.moveTo(pad.left, pad.top);
  cctx.lineTo(pad.left, pad.top + ph);
  cctx.lineTo(pad.left + pw, pad.top + ph);
  cctx.stroke();

  // y ticks
  cctx.fillStyle = "#7777aa";
  cctx.font = "9px monospace";
  cctx.textAlign = "right";
  for (let t = 0; t <= 4; t++) {
    const v = minD + (t / 4) * (maxD - minD);
    const y = yScale(v);
    cctx.fillText(v.toFixed(0), pad.left - 4, y + 3);
    cctx.strokeStyle = "#1a1a2e";
    cctx.lineWidth = 1;
    cctx.beginPath();
    cctx.moveTo(pad.left, y);
    cctx.lineTo(pad.left + pw, y);
    cctx.stroke();
  }

  // x label
  cctx.fillStyle = "#7777aa";
  cctx.textAlign = "center";
  cctx.fillText("Iteration", pad.left + pw / 2, ch - 4);

  // line
  cctx.beginPath();
  cctx.moveTo(xScale(0), yScale(data[0]));
  for (let i = 1; i < data.length; i++) {
    cctx.lineTo(xScale(i), yScale(data[i]));
  }
  cctx.strokeStyle = "#f59e0b";
  cctx.lineWidth = 1.5;
  cctx.stroke();

  // current point
  const lastX = xScale(data.length - 1);
  const lastY = yScale(data[data.length - 1]);
  cctx.beginPath();
  cctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
  cctx.fillStyle = "#fbbf24";
  cctx.fill();
}

// ── Controls ───────────────────────────────────────────────────────────────

function updateLabels(): void {
  const iterLabel = document.getElementById("iter-label")!;
  iterLabel.textContent = `Iter ${iteration}  |  Best: ${bestDist === Infinity ? "–" : bestDist.toFixed(1)}`;
}

function reset(): void {
  pause();
  cities = randomCities(NUM_CITIES);
  initPhero();
  bestTour = [];
  bestDist = Infinity;
  iteration = 0;
  distHistory.length = 0;
  updateLabels();
  draw();
  drawChart();
}

function pause(): void {
  running = false;
  if (timerId !== null) { clearTimeout(timerId); timerId = null; }
  const btn = document.getElementById("btn-play") as HTMLButtonElement;
  btn.textContent = "▶ Play";
}

function play(): void {
  if (running) return;
  running = true;
  const btn = document.getElementById("btn-play") as HTMLButtonElement;
  btn.textContent = "⏸ Pause";
  function loop() {
    if (!running) return;
    stepACO();
    timerId = window.setTimeout(loop, SPEED);
  }
  loop();
}

function togglePlay(): void {
  if (running) pause(); else play();
}

// ── Slider wiring ──────────────────────────────────────────────────────────

function wireSlider(id: string, valId: string, decimals: number, setter: (v: number) => void): void {
  const inp = document.getElementById(id) as HTMLInputElement;
  const lbl = document.getElementById(valId) as HTMLElement;
  lbl.textContent = parseFloat(inp.value).toFixed(decimals);
  inp.addEventListener("input", () => {
    const v = parseFloat(inp.value);
    lbl.textContent = v.toFixed(decimals);
    setter(v);
  });
}

// ── Init ───────────────────────────────────────────────────────────────────

function init(): void {
  // wire buttons
  document.getElementById("btn-play")!.addEventListener("click", togglePlay);
  document.getElementById("btn-step")!.addEventListener("click", () => { pause(); stepACO(); });
  document.getElementById("btn-reset")!.addEventListener("click", reset);

  // wire sliders
  wireSlider("inp-cities",  "lbl-cities",  0, v => { NUM_CITIES = v; reset(); });
  wireSlider("inp-ants",    "lbl-ants",    0, v => { NUM_ANTS   = v; });
  wireSlider("inp-alpha",   "lbl-alpha",   1, v => { ALPHA      = v; });
  wireSlider("inp-beta",    "lbl-beta",    1, v => { BETA       = v; });
  wireSlider("inp-rho",     "lbl-rho",     2, v => { RHO        = v; });
  wireSlider("inp-q",       "lbl-q",       0, v => { Q          = v; });
  wireSlider("inp-speed",   "lbl-speed",   0, v => { SPEED      = v; });

  // click to add city
  mainCanvas.addEventListener("click", (e: MouseEvent) => {
    const rect = mainCanvas.getBoundingClientRect();
    const sx = mainCanvas.width  / rect.width;
    const sy = mainCanvas.height / rect.height;
    cities.push({ x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy });
    if (cities.length >= 2) initPhero();
    bestTour = [];
    bestDist = Infinity;
    iteration = 0;
    distHistory.length = 0;
    updateLabels();
    draw();
    drawChart();
  });

  reset();
}

document.addEventListener("DOMContentLoaded", init);
