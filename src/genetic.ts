import * as d3 from 'd3';

// ─── Types ────────────────────────────────────────────────────────────────────

type FnName = 'rastrigin' | 'ackley' | 'rosenbrock' | 'sphere' | 'himmelblau';
type SelectionMethod = 'tournament' | 'roulette';
type OptDir = 'minimize' | 'maximize';

interface Individual {
  x: number;
  y: number;
  fitness: number;
}

interface GAParams {
  fn: FnName;
  popSize: number;
  mutRate: number;
  mutSigma: number;
  crossRate: number;
  selection: SelectionMethod;
  elitism: number;
  speed: number; // ms per frame (lower = faster)
}

interface GenRecord {
  gen: number;
  best: number;
  mean: number;
}

// ─── Fitness landscapes ───────────────────────────────────────────────────────

// All functions defined over domain [-5.12, 5.12] for consistency.
// We negate for minimisation problems so the GA always maximises internally.

const DOMAIN_MIN = -5.12;
const DOMAIN_MAX = 5.12;

function rastriginRaw(x: number, y: number): number {
  // Minimum at (0,0) = 0, many local minima
  return 20 + x * x + y * y
    - 10 * (Math.cos(2 * Math.PI * x) + Math.cos(2 * Math.PI * y));
}

function ackleyRaw(x: number, y: number): number {
  // Minimum at (0,0) = 0
  const a = 20, b = 0.2, c = 2 * Math.PI;
  const sq = Math.sqrt(0.5 * (x * x + y * y));
  const cos = 0.5 * (Math.cos(c * x) + Math.cos(c * y));
  return -a * Math.exp(-b * sq) - Math.exp(cos) + a + Math.E;
}

function rosenbrockRaw(x: number, y: number): number {
  // Minimum at (1,1) = 0, banana-shaped valley
  return (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
}

function sphereRaw(x: number, y: number): number {
  return x * x + y * y;
}

function himmelblauRaw(x: number, y: number): number {
  // Four equal minima at ~(3,2),(-2.805,3.131),(-3.779,-3.283),(3.584,-1.848) = 0
  return (x * x + y - 11) * (x * x + y - 11)
       + (x + y * y - 7) * (x + y * y - 7);
}

const FN_META: Record<FnName, { raw: (x: number, y: number) => number; dir: OptDir; label: string }> = {
  rastrigin: { raw: rastriginRaw, dir: 'minimize', label: 'Rastrigin (minimize)' },
  ackley:    { raw: ackleyRaw,    dir: 'minimize', label: 'Ackley (minimize)' },
  rosenbrock:{ raw: rosenbrockRaw,dir: 'minimize', label: 'Rosenbrock (minimize)' },
  sphere:    { raw: sphereRaw,    dir: 'minimize', label: 'Sphere (minimize)' },
  himmelblau:{ raw: himmelblauRaw,dir: 'minimize', label: 'Himmelblau (minimize)' },
};

// Returns a fitness value: higher is always better for the GA internally.
function evalFitness(fn: FnName, x: number, y: number): number {
  const meta = FN_META[fn];
  const raw = meta.raw(x, y);
  // Negate minimisation problems so GA maximises
  return meta.dir === 'minimize' ? -raw : raw;
}

// ─── Heatmap grid ─────────────────────────────────────────────────────────────

const GRID = 120; // cells per side

function buildHeatmap(fn: FnName): { data: Float32Array; min: number; max: number } {
  const data = new Float32Array(GRID * GRID);
  let min = Infinity, max = -Infinity;
  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      const x = DOMAIN_MIN + (col / (GRID - 1)) * (DOMAIN_MAX - DOMAIN_MIN);
      const y = DOMAIN_MIN + (row / (GRID - 1)) * (DOMAIN_MAX - DOMAIN_MIN);
      const v = FN_META[fn].raw(x, y);
      data[row * GRID + col] = v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return { data, min, max };
}

// ─── GA helpers ──────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function randn(): number {
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u + 1e-12)) * Math.cos(2 * Math.PI * v);
}

function createIndividual(fn: FnName): Individual {
  const x = DOMAIN_MIN + Math.random() * (DOMAIN_MAX - DOMAIN_MIN);
  const y = DOMAIN_MIN + Math.random() * (DOMAIN_MAX - DOMAIN_MIN);
  return { x, y, fitness: evalFitness(fn, x, y) };
}

function tournamentSelect(pop: Individual[]): Individual {
  const k = 3;
  let best = pop[Math.floor(Math.random() * pop.length)];
  for (let i = 1; i < k; i++) {
    const cand = pop[Math.floor(Math.random() * pop.length)];
    if (cand.fitness > best.fitness) best = cand;
  }
  return best;
}

function rouletteSelect(pop: Individual[], totalFit: number, minFit: number): Individual {
  // Shift so all weights positive
  const shifted = pop.map(ind => ind.fitness - minFit + 1e-6);
  const shiftedTotal = shifted.reduce((a, b) => a + b, 0);
  let r = Math.random() * shiftedTotal;
  for (let i = 0; i < pop.length; i++) {
    r -= shifted[i];
    if (r <= 0) return pop[i];
  }
  return pop[pop.length - 1];
}

function crossover(a: Individual, b: Individual, crossRate: number, fn: FnName): Individual {
  if (Math.random() > crossRate) {
    // Clone one parent
    const src = Math.random() < 0.5 ? a : b;
    return { x: src.x, y: src.y, fitness: src.fitness };
  }
  // BLX-alpha blend crossover
  const alpha = 0.3;
  const xLo = Math.min(a.x, b.x), xHi = Math.max(a.x, b.x);
  const yLo = Math.min(a.y, b.y), yHi = Math.max(a.y, b.y);
  const dx = xHi - xLo, dy = yHi - yLo;
  const cx = clamp(xLo - alpha * dx + Math.random() * (dx * (1 + 2 * alpha)), DOMAIN_MIN, DOMAIN_MAX);
  const cy = clamp(yLo - alpha * dy + Math.random() * (dy * (1 + 2 * alpha)), DOMAIN_MIN, DOMAIN_MAX);
  return { x: cx, y: cy, fitness: evalFitness(fn, cx, cy) };
}

function mutate(ind: Individual, mutRate: number, mutSigma: number, fn: FnName): Individual {
  let { x, y } = ind;
  if (Math.random() < mutRate) x = clamp(x + randn() * mutSigma, DOMAIN_MIN, DOMAIN_MAX);
  if (Math.random() < mutRate) y = clamp(y + randn() * mutSigma, DOMAIN_MIN, DOMAIN_MAX);
  return { x, y, fitness: evalFitness(fn, x, y) };
}

function stepGeneration(pop: Individual[], params: GAParams): Individual[] {
  const { fn, popSize, mutRate, mutSigma, crossRate, selection, elitism } = params;

  // Sort descending by fitness
  pop.sort((a, b) => b.fitness - a.fitness);

  const next: Individual[] = [];

  // Elitism
  const eliteCount = Math.min(elitism, popSize);
  for (let i = 0; i < eliteCount; i++) {
    next.push({ x: pop[i].x, y: pop[i].y, fitness: pop[i].fitness });
  }

  const minFit = pop[pop.length - 1].fitness;
  const totalFit = pop.reduce((s, ind) => s + ind.fitness, 0);

  while (next.length < popSize) {
    let parent1: Individual, parent2: Individual;
    if (selection === 'tournament') {
      parent1 = tournamentSelect(pop);
      parent2 = tournamentSelect(pop);
    } else {
      parent1 = rouletteSelect(pop, totalFit, minFit);
      parent2 = rouletteSelect(pop, totalFit, minFit);
    }
    const child = crossover(parent1, parent2, crossRate, fn);
    const mutated = mutate(child, mutRate, mutSigma, fn);
    next.push(mutated);
  }

  return next;
}

// ─── UI & D3 rendering ───────────────────────────────────────────────────────

const canvasLandscape = document.getElementById('canvas-landscape') as HTMLCanvasElement;
const canvasCurve = document.getElementById('canvas-curve') as HTMLCanvasElement;
const ctxL = canvasLandscape.getContext('2d')!;
const ctxC = canvasCurve.getContext('2d')!;

// Sliders / controls
function getEl<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

const selFn = getEl<HTMLSelectElement>('sel-fn');
const selSel = getEl<HTMLSelectElement>('sel-sel');
const sliderPop = getEl<HTMLInputElement>('slider-pop');
const sliderMutRate = getEl<HTMLInputElement>('slider-mutrate');
const sliderMutSigma = getEl<HTMLInputElement>('slider-mutsigma');
const sliderCross = getEl<HTMLInputElement>('slider-cross');
const sliderElite = getEl<HTMLInputElement>('slider-elite');
const sliderSpeed = getEl<HTMLInputElement>('slider-speed');
const btnPlay = getEl<HTMLButtonElement>('btn-play');
const btnPause = getEl<HTMLButtonElement>('btn-pause');
const btnStep = getEl<HTMLButtonElement>('btn-step');
const btnReset = getEl<HTMLButtonElement>('btn-reset');
const spanGen = getEl<HTMLSpanElement>('span-gen');
const spanBest = getEl<HTMLSpanElement>('span-best');
const spanMean = getEl<HTMLSpanElement>('span-mean');
const labelPop = getEl<HTMLSpanElement>('label-pop');
const labelMutRate = getEl<HTMLSpanElement>('label-mutrate');
const labelMutSigma = getEl<HTMLSpanElement>('label-mutsigma');
const labelCross = getEl<HTMLSpanElement>('label-cross');
const labelElite = getEl<HTMLSpanElement>('label-elite');
const labelSpeed = getEl<HTMLSpanElement>('label-speed');

// ─── State ────────────────────────────────────────────────────────────────────

let population: Individual[] = [];
let history: GenRecord[] = [];
let generation = 0;
let running = false;
let animHandle: number | null = null;
let lastFrameTime = 0;

let heatmap: { data: Float32Array; min: number; max: number } | null = null;
let heatImageData: ImageData | null = null;

// Colour scale for heatmap: viridis-ish (low=dark purple, high=yellow)
const COLORS: Array<[number, number, number]> = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];

function colorAt(t: number): [number, number, number] {
  t = clamp(t, 0, 1);
  const scaled = t * (COLORS.length - 1);
  const lo = Math.floor(scaled);
  const hi = Math.min(lo + 1, COLORS.length - 1);
  const f = scaled - lo;
  const a = COLORS[lo], b = COLORS[hi];
  return [
    Math.round(a[0] + f * (b[0] - a[0])),
    Math.round(a[1] + f * (b[1] - a[1])),
    Math.round(a[2] + f * (b[2] - a[2])),
  ];
}

function getParams(): GAParams {
  return {
    fn: selFn.value as FnName,
    popSize: parseInt(sliderPop.value, 10),
    mutRate: parseFloat(sliderMutRate.value),
    mutSigma: parseFloat(sliderMutSigma.value),
    crossRate: parseFloat(sliderCross.value),
    selection: selSel.value as SelectionMethod,
    elitism: parseInt(sliderElite.value, 10),
    speed: parseInt(sliderSpeed.value, 10),
  };
}

// ─── Heatmap rendering ────────────────────────────────────────────────────────

function buildHeatImageData(fn: FnName, W: number, H: number): ImageData {
  heatmap = buildHeatmap(fn);
  const imgData = ctxL.createImageData(W, H);
  const { data, min, max } = heatmap;
  const range = max - min || 1;

  for (let row = 0; row < H; row++) {
    for (let col = 0; col < W; col++) {
      // Sample from grid using nearest-neighbour
      const gridCol = Math.round((col / (W - 1)) * (GRID - 1));
      const gridRow = Math.round((row / (H - 1)) * (GRID - 1));
      const v = data[gridRow * GRID + gridCol];
      const t = (v - min) / range;
      // Invert so low values (minima) are bright (yellow) for minimisation
      const tDisplay = FN_META[fn].dir === 'minimize' ? 1 - t : t;
      const [r, g, b] = colorAt(tDisplay);
      const idx = (row * W + col) * 4;
      imgData.data[idx]     = r;
      imgData.data[idx + 1] = g;
      imgData.data[idx + 2] = b;
      imgData.data[idx + 3] = 255;
    }
  }
  return imgData;
}

function domainToCanvas(v: number, W: number): number {
  return ((v - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * W;
}

function drawLandscape(params: GAParams): void {
  const W = canvasLandscape.width;
  const H = canvasLandscape.height;

  // Draw heatmap
  if (heatImageData) {
    ctxL.putImageData(heatImageData, 0, 0);
  } else {
    ctxL.fillStyle = '#0d0d1e';
    ctxL.fillRect(0, 0, W, H);
  }

  // Draw population dots
  const fn = params.fn;
  let bestIdx = 0;
  population.forEach((ind, i) => {
    if (ind.fitness > population[bestIdx].fitness) bestIdx = i;
  });

  population.forEach((ind, i) => {
    const px = domainToCanvas(ind.x, W);
    const py = domainToCanvas(ind.y, H);
    const isBest = i === bestIdx;
    ctxL.beginPath();
    ctxL.arc(px, py, isBest ? 6 : 3.5, 0, 2 * Math.PI);
    if (isBest) {
      ctxL.fillStyle = '#ffffff';
      ctxL.strokeStyle = '#ff4444';
      ctxL.lineWidth = 2;
      ctxL.fill();
      ctxL.stroke();
    } else {
      ctxL.fillStyle = 'rgba(255,220,80,0.75)';
      ctxL.fill();
    }
  });

  // Label best individual
  if (population.length > 0) {
    const best = population[bestIdx];
    const px = domainToCanvas(best.x, W);
    const py = domainToCanvas(best.y, H);
    const rawBest = FN_META[fn].raw(best.x, best.y);
    ctxL.fillStyle = '#fff';
    ctxL.font = '11px monospace';
    ctxL.fillText(`(${best.x.toFixed(2)}, ${best.y.toFixed(2)}) → ${rawBest.toFixed(4)}`,
      px + 8, py - 6);
  }
}

// ─── Fitness curve ────────────────────────────────────────────────────────────

function drawCurve(): void {
  const W = canvasCurve.width;
  const H = canvasCurve.height;
  const pad = { top: 18, right: 14, bottom: 30, left: 48 };

  ctxC.fillStyle = '#0d0d1e';
  ctxC.fillRect(0, 0, W, H);

  if (history.length < 2) return;

  const innerW = W - pad.left - pad.right;
  const innerH = H - pad.top - pad.bottom;

  // Compute y domain from history (raw fitness = negated internal fitness for minimise)
  const fn = selFn.value as FnName;
  const isMin = FN_META[fn].dir === 'minimize';

  // Convert internal fitness back to raw for display
  const rawHistory = history.map(r => ({
    gen: r.gen,
    best: isMin ? -r.best : r.best,
    mean: isMin ? -r.mean : r.mean,
  }));

  const allVals = rawHistory.reduce((acc: number[], r) => {
    acc.push(r.best, r.mean);
    return acc;
  }, []);
  const yMin = Math.min(...allVals);
  const yMax = Math.max(...allVals);
  const yRange = yMax - yMin || 1;

  const xScale = (gen: number) => pad.left + (gen / Math.max(history.length - 1, 1)) * innerW;
  const yScale = (v: number) => pad.top + innerH - ((v - yMin) / yRange) * innerH;

  // Grid lines
  ctxC.strokeStyle = 'rgba(255,255,255,0.07)';
  ctxC.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const yy = pad.top + (i / 4) * innerH;
    ctxC.beginPath();
    ctxC.moveTo(pad.left, yy);
    ctxC.lineTo(W - pad.right, yy);
    ctxC.stroke();
    const label = (yMax - (i / 4) * yRange).toFixed(1);
    ctxC.fillStyle = '#778899';
    ctxC.font = '10px monospace';
    ctxC.textAlign = 'right';
    ctxC.fillText(label, pad.left - 4, yy + 3);
  }

  // X axis labels
  ctxC.fillStyle = '#778899';
  ctxC.font = '10px monospace';
  ctxC.textAlign = 'center';
  const tickCount = Math.min(6, history.length);
  for (let i = 0; i <= tickCount; i++) {
    const gIdx = Math.round((i / tickCount) * (history.length - 1));
    ctxC.fillText(String(rawHistory[gIdx].gen), xScale(gIdx), H - pad.bottom + 14);
  }

  // Mean line (dimmer)
  ctxC.beginPath();
  ctxC.strokeStyle = 'rgba(100,180,255,0.6)';
  ctxC.lineWidth = 1.5;
  rawHistory.forEach((r, i) => {
    const x = xScale(i), y = yScale(r.mean);
    if (i === 0) ctxC.moveTo(x, y); else ctxC.lineTo(x, y);
  });
  ctxC.stroke();

  // Best line (bright)
  ctxC.beginPath();
  ctxC.strokeStyle = '#ff6666';
  ctxC.lineWidth = 2;
  rawHistory.forEach((r, i) => {
    const x = xScale(i), y = yScale(r.best);
    if (i === 0) ctxC.moveTo(x, y); else ctxC.lineTo(x, y);
  });
  ctxC.stroke();

  // Legend
  ctxC.font = '10px sans-serif';
  ctxC.textAlign = 'left';
  ctxC.fillStyle = '#ff6666';
  ctxC.fillText('Best', pad.left + 4, pad.top + 12);
  ctxC.fillStyle = 'rgba(100,180,255,0.9)';
  ctxC.fillText('Mean', pad.left + 36, pad.top + 12);

  // Y axis label
  ctxC.save();
  ctxC.translate(12, pad.top + innerH / 2);
  ctxC.rotate(-Math.PI / 2);
  ctxC.fillStyle = '#8899aa';
  ctxC.font = '10px sans-serif';
  ctxC.textAlign = 'center';
  ctxC.fillText(isMin ? 'f(x,y) — lower is better' : 'Fitness — higher is better', 0, 0);
  ctxC.restore();
}

// ─── Main loop ────────────────────────────────────────────────────────────────

function recordGeneration(): void {
  if (population.length === 0) return;
  const fitnesses = population.map(ind => ind.fitness);
  const best = fitnesses.reduce((a, b) => Math.max(a, b), -Infinity);
  const mean = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
  history.push({ gen: generation, best, mean });

  // Update status
  const fn = selFn.value as FnName;
  const isMin = FN_META[fn].dir === 'minimize';
  spanGen.textContent = String(generation);
  spanBest.textContent = (isMin ? -best : best).toFixed(5);
  spanMean.textContent = (isMin ? -mean : mean).toFixed(5);
}

function doStep(): void {
  const params = getParams();
  population = stepGeneration(population, params);
  generation++;
  recordGeneration();
  drawLandscape(params);
  drawCurve();
}

function animate(timestamp: number): void {
  if (!running) return;
  const params = getParams();
  const delay = params.speed;
  if (timestamp - lastFrameTime >= delay) {
    doStep();
    lastFrameTime = timestamp;
  }
  animHandle = requestAnimationFrame(animate);
}

function reset(): void {
  running = false;
  if (animHandle !== null) { cancelAnimationFrame(animHandle); animHandle = null; }
  generation = 0;
  history = [];
  const params = getParams();
  population = Array.from({ length: params.popSize }, () => createIndividual(params.fn));
  recordGeneration();
  // Rebuild heatmap
  heatImageData = buildHeatImageData(params.fn, canvasLandscape.width, canvasLandscape.height);
  drawLandscape(params);
  drawCurve();
  updateLabels();
}

function play(): void {
  if (running) return;
  running = true;
  lastFrameTime = 0;
  animHandle = requestAnimationFrame(animate);
}

function pause(): void {
  running = false;
  if (animHandle !== null) { cancelAnimationFrame(animHandle); animHandle = null; }
}

// ─── Canvas sizing ────────────────────────────────────────────────────────────

function resizeCanvases(): void {
  const panel = document.getElementById('canvas-panel')!;
  const W = Math.min(panel.clientWidth - 24, 560);
  canvasLandscape.width = W;
  canvasLandscape.height = W;
  canvasCurve.width = W;
  canvasCurve.height = Math.round(W * 0.36);
  // Rebuild heatmap at new size
  const params = getParams();
  heatImageData = buildHeatImageData(params.fn, W, W);
  drawLandscape(params);
  drawCurve();
}

// ─── Slider label updates ─────────────────────────────────────────────────────

function updateLabels(): void {
  labelPop.textContent = sliderPop.value;
  labelMutRate.textContent = sliderMutRate.value;
  labelMutSigma.textContent = sliderMutSigma.value;
  labelCross.textContent = sliderCross.value;
  labelElite.textContent = sliderElite.value;
  labelSpeed.textContent = sliderSpeed.value + ' ms';
}

// ─── Event wiring ─────────────────────────────────────────────────────────────

btnPlay.addEventListener('click', play);
btnPause.addEventListener('click', pause);
btnStep.addEventListener('click', () => { pause(); doStep(); });
btnReset.addEventListener('click', reset);

selFn.addEventListener('change', reset);
selSel.addEventListener('change', () => { /* live, no reset needed */ });

[sliderPop, sliderMutRate, sliderMutSigma, sliderCross, sliderElite, sliderSpeed].forEach(s => {
  s.addEventListener('input', updateLabels);
});

// Rebuilding pop on pop-size change requires reset
sliderPop.addEventListener('change', reset);

window.addEventListener('resize', () => {
  pause();
  resizeCanvases();
});

// ─── Boot ─────────────────────────────────────────────────────────────────────

resizeCanvases();
reset();
