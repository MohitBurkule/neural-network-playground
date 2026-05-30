import * as d3 from 'd3';

// ── Types ────────────────────────────────────────────────────────────────────

type ActivationFn = 'relu' | 'tanh';
type Method = 'mc-dropout' | 'ensemble';
type Preset = 'sine' | 'step' | 'cubic';

interface Config {
  lr: number;
  hiddenSize: number;
  dropoutRate: number;
  numSamples: number;    // MC samples or ensemble size
  method: Method;
  preset: Preset;
}

interface DataPoint {
  x: number;
  y: number;
}

// ── MLP ──────────────────────────────────────────────────────────────────────

class Matrix {
  data: Float64Array;
  rows: number;
  cols: number;

  constructor(rows: number, cols: number, init: 'zeros' | 'randn' | 'randn-small' = 'zeros') {
    this.rows = rows;
    this.cols = cols;
    this.data = new Float64Array(rows * cols);
    if (init === 'randn') {
      for (let i = 0; i < this.data.length; i++) this.data[i] = randn() * Math.sqrt(2 / rows);
    } else if (init === 'randn-small') {
      for (let i = 0; i < this.data.length; i++) this.data[i] = randn() * 0.1;
    }
  }

  get(r: number, c: number): number { return this.data[r * this.cols + c]; }
  set(r: number, c: number, v: number): void { this.data[r * this.cols + c] = v; }
  clone(): Matrix {
    const m = new Matrix(this.rows, this.cols);
    m.data.set(this.data);
    return m;
  }
}

function randn(): number {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Simple 1-hidden-layer MLP: input(1) -> hidden(H) -> output(1)
class MLP {
  W1: Matrix; b1: Matrix; W2: Matrix; b2: Matrix;
  // Gradients
  dW1: Matrix; db1: Matrix; dW2: Matrix; db2: Matrix;
  // Adam state
  mW1: Matrix; vW1: Matrix; mb1: Matrix; vb1: Matrix;
  mW2: Matrix; vW2: Matrix; mb2: Matrix; vb2: Matrix;
  step: number = 0;

  hidden: number;
  dropoutRate: number;

  // cache for backward
  private _x: number = 0;
  private _h1pre: Float64Array;
  private _h1: Float64Array;
  private _dropMask: Float64Array;

  constructor(hidden: number, dropoutRate: number = 0.1) {
    this.hidden = hidden;
    this.dropoutRate = dropoutRate;
    this.W1 = new Matrix(hidden, 1, 'randn');
    this.b1 = new Matrix(hidden, 1, 'zeros');
    this.W2 = new Matrix(1, hidden, 'randn');
    this.b2 = new Matrix(1, 1, 'zeros');
    this.dW1 = new Matrix(hidden, 1);
    this.db1 = new Matrix(hidden, 1);
    this.dW2 = new Matrix(1, hidden);
    this.db2 = new Matrix(1, 1);
    this.mW1 = new Matrix(hidden, 1); this.vW1 = new Matrix(hidden, 1);
    this.mb1 = new Matrix(hidden, 1); this.vb1 = new Matrix(hidden, 1);
    this.mW2 = new Matrix(1, hidden); this.vW2 = new Matrix(1, hidden);
    this.mb2 = new Matrix(1, 1);     this.vb2 = new Matrix(1, 1);
    this._h1pre = new Float64Array(hidden);
    this._h1 = new Float64Array(hidden);
    this._dropMask = new Float64Array(hidden);
  }

  forward(x: number, training: boolean): number {
    this._x = x;
    const H = this.hidden;
    const keep = 1 - this.dropoutRate;
    // Layer 1
    for (let j = 0; j < H; j++) {
      this._h1pre[j] = this.W1.get(j, 0) * x + this.b1.get(j, 0);
      this._h1[j] = Math.tanh(this._h1pre[j]);
      if (training) {
        this._dropMask[j] = Math.random() < keep ? 1 / keep : 0;
        this._h1[j] *= this._dropMask[j];
      } else {
        this._dropMask[j] = 1.0;
      }
    }
    // Layer 2
    let out = this.b2.get(0, 0);
    for (let j = 0; j < H; j++) out += this.W2.get(0, j) * this._h1[j];
    return out;
  }

  backward(x: number, pred: number, target: number): number {
    const H = this.hidden;
    const loss = 0.5 * (pred - target) * (pred - target);
    const dOut = pred - target;

    this.db2.set(0, 0, dOut);
    for (let j = 0; j < H; j++) {
      this.dW2.set(0, j, dOut * this._h1[j]);
      const dh1 = dOut * this.W2.get(0, j) * this._dropMask[j];
      const dtanh = dh1 * (1 - Math.tanh(this._h1pre[j]) * Math.tanh(this._h1pre[j]));
      this.db1.set(j, 0, dtanh);
      this.dW1.set(j, 0, dtanh * x);
    }
    return loss;
  }

  adamUpdate(lr: number): void {
    this.step++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const t = this.step;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);

    const update = (w: Matrix, dw: Matrix, m: Matrix, v: Matrix) => {
      for (let i = 0; i < w.data.length; i++) {
        m.data[i] = b1 * m.data[i] + (1 - b1) * dw.data[i];
        v.data[i] = b2 * v.data[i] + (1 - b2) * dw.data[i] * dw.data[i];
        const mHat = m.data[i] / bc1;
        const vHat = v.data[i] / bc2;
        w.data[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
    };

    update(this.W1, this.dW1, this.mW1, this.vW1);
    update(this.b1, this.db1, this.mb1, this.vb1);
    update(this.W2, this.dW2, this.mW2, this.vW2);
    update(this.b2, this.db2, this.mb2, this.vb2);
  }

  clone(): MLP {
    const c = new MLP(this.hidden, this.dropoutRate);
    c.W1 = this.W1.clone(); c.b1 = this.b1.clone();
    c.W2 = this.W2.clone(); c.b2 = this.b2.clone();
    c.step = this.step;
    return c;
  }
}

// ── Dataset / Target functions ───────────────────────────────────────────────

function targetFn(preset: Preset, x: number): number {
  if (preset === 'sine')  return Math.sin(2.5 * x);
  if (preset === 'step')  return x < 0 ? -0.7 : 0.7;
  // cubic
  return 0.5 * x * x * x - 0.3 * x;
}

function makeDataset(preset: Preset): DataPoint[] {
  const pts: DataPoint[] = [];
  // Left cluster: -1.0 to -0.25
  for (let i = 0; i < 15; i++) {
    const x = -1.0 + Math.random() * 0.75;
    pts.push({ x, y: targetFn(preset, x) + randn() * 0.05 });
  }
  // Right cluster: 0.25 to 1.0   (deliberate gap in centre)
  for (let i = 0; i < 15; i++) {
    const x = 0.25 + Math.random() * 0.75;
    pts.push({ x, y: targetFn(preset, x) + randn() * 0.05 });
  }
  return pts;
}

// ── Prediction / Uncertainty ─────────────────────────────────────────────────

interface PredResult {
  xs: number[];
  mean: number[];
  std: number[];
  samples: number[][];   // individual sample curves (faint lines)
}

function predictMCDropout(net: MLP, xs: number[], nSamples: number): PredResult {
  const S = Math.min(nSamples, 50);
  const N = xs.length;
  const allPreds: number[][] = [];
  for (let s = 0; s < S; s++) {
    const row: number[] = [];
    for (let i = 0; i < N; i++) row.push(net.forward(xs[i], true)); // training=true keeps dropout on
    allPreds.push(row);
  }
  const mean: number[] = new Array(N).fill(0);
  const std: number[] = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    let sum = 0;
    for (let s = 0; s < S; s++) sum += allPreds[s][i];
    mean[i] = sum / S;
    let varSum = 0;
    for (let s = 0; s < S; s++) varSum += (allPreds[s][i] - mean[i]) ** 2;
    std[i] = Math.sqrt(varSum / S);
  }
  // Pick up to 5 sample curves
  const samples: number[][] = [];
  const take = Math.min(5, S);
  for (let s = 0; s < take; s++) samples.push(allPreds[s]);
  return { xs, mean, std, samples };
}

function predictEnsemble(nets: MLP[], xs: number[]): PredResult {
  const N = xs.length;
  const M = nets.length;
  const allPreds: number[][] = [];
  for (let m = 0; m < M; m++) {
    const row: number[] = [];
    for (let i = 0; i < N; i++) row.push(nets[m].forward(xs[i], false));
    allPreds.push(row);
  }
  const mean: number[] = new Array(N).fill(0);
  const std: number[] = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    let sum = 0;
    for (let m = 0; m < M; m++) sum += allPreds[m][i];
    mean[i] = sum / M;
    let varSum = 0;
    for (let m = 0; m < M; m++) varSum += (allPreds[m][i] - mean[i]) ** 2;
    std[i] = Math.sqrt(varSum / M);
  }
  const samples: number[][] = [];
  const take = Math.min(5, M);
  for (let m = 0; m < take; m++) samples.push(allPreds[m]);
  return { xs, mean, std, samples };
}

// ── Main Application ─────────────────────────────────────────────────────────

const PLOT_W = 560, PLOT_H = 320;
const MARGIN = { top: 16, right: 20, bottom: 36, left: 44 };
const LOSS_W = 220, LOSS_H = 120;
const X_RANGE: [number, number] = [-1.4, 1.4];
const PRED_POINTS = 120;

let cfg: Config = {
  lr: 0.005,
  hiddenSize: 32,
  dropoutRate: 0.2,
  numSamples: 20,
  method: 'mc-dropout',
  preset: 'sine',
};

let data: DataPoint[] = [];
let nets: MLP[] = [];         // ensemble members or [single net]
let lossHistory: number[] = [];
let running = false;
let animFrame: number | null = null;
let trainStep = 0;

// Build xs for prediction grid
function predXs(): number[] {
  const xs: number[] = [];
  for (let i = 0; i <= PRED_POINTS; i++) {
    xs.push(X_RANGE[0] + (X_RANGE[1] - X_RANGE[0]) * i / PRED_POINTS);
  }
  return xs;
}

// ── D3 setup ─────────────────────────────────────────────────────────────────

const xScale = d3.scaleLinear().domain(X_RANGE).range([MARGIN.left, PLOT_W - MARGIN.right]);
const yScale = d3.scaleLinear().domain([-2.2, 2.2]).range([PLOT_H - MARGIN.bottom, MARGIN.top]);
const lossXScale = d3.scaleLinear().range([30, LOSS_W - 10]);
const lossYScale = d3.scaleLinear().range([LOSS_H - 20, 10]);

// Main plot SVG
const mainSvg = d3.select('#main-plot')
  .attr('width', PLOT_W)
  .attr('height', PLOT_H);

// Clip path
mainSvg.append('defs').append('clipPath').attr('id', 'clip-main')
  .append('rect')
  .attr('x', MARGIN.left).attr('y', MARGIN.top)
  .attr('width', PLOT_W - MARGIN.left - MARGIN.right)
  .attr('height', PLOT_H - MARGIN.top - MARGIN.bottom);

const plotArea = mainSvg.append('g').attr('clip-path', 'url(#clip-main)');

// Background
plotArea.append('rect')
  .attr('x', MARGIN.left).attr('y', MARGIN.top)
  .attr('width', PLOT_W - MARGIN.left - MARGIN.right)
  .attr('height', PLOT_H - MARGIN.top - MARGIN.bottom)
  .attr('fill', '#12151f');

// Gap indicator
const gapRect = plotArea.append('rect')
  .attr('x', xScale(-0.25)).attr('y', MARGIN.top)
  .attr('width', xScale(0.25) - xScale(-0.25))
  .attr('height', PLOT_H - MARGIN.top - MARGIN.bottom)
  .attr('fill', 'rgba(255,200,50,0.04)');

// Grid lines
const xAxis = d3.axisBottom(xScale).ticks(7).tickSize(-(PLOT_H - MARGIN.top - MARGIN.bottom));
const yAxis = d3.axisLeft(yScale).ticks(6).tickSize(-(PLOT_W - MARGIN.left - MARGIN.right));

mainSvg.append('g').attr('class', 'grid x-grid')
  .attr('transform', `translate(0,${PLOT_H - MARGIN.bottom})`)
  .call(xAxis)
  .selectAll('line').attr('stroke', '#2a2e44').attr('stroke-dasharray', '3,3');
mainSvg.select('.x-grid .domain').attr('stroke', '#444');
mainSvg.select('.x-grid').selectAll('text').attr('fill', '#888').attr('font-size', '11px');

mainSvg.append('g').attr('class', 'grid y-grid')
  .attr('transform', `translate(${MARGIN.left},0)`)
  .call(yAxis)
  .selectAll('line').attr('stroke', '#2a2e44').attr('stroke-dasharray', '3,3');
mainSvg.select('.y-grid .domain').attr('stroke', '#444');
mainSvg.select('.y-grid').selectAll('text').attr('fill', '#888').attr('font-size', '11px');

// Zero line
plotArea.append('line')
  .attr('x1', xScale(X_RANGE[0])).attr('x2', xScale(X_RANGE[1]))
  .attr('y1', yScale(0)).attr('y2', yScale(0))
  .attr('stroke', '#3a3e58').attr('stroke-width', 1);

// ±2σ band
const band2 = plotArea.append('path').attr('class', 'band2').attr('fill', 'rgba(100,160,255,0.12)').attr('stroke', 'none');
// ±1σ band
const band1 = plotArea.append('path').attr('class', 'band1').attr('fill', 'rgba(100,160,255,0.22)').attr('stroke', 'none');
// Sample curves group
const samplesG = plotArea.append('g').attr('class', 'samples');
// Mean curve
const meanPath = plotArea.append('path').attr('class', 'mean').attr('fill', 'none')
  .attr('stroke', '#64a0ff').attr('stroke-width', 2.5);
// True function (faint reference)
const truePath = plotArea.append('path').attr('class', 'true-fn').attr('fill', 'none')
  .attr('stroke', 'rgba(200,200,120,0.4)').attr('stroke-width', 1.5).attr('stroke-dasharray', '5,4');
// Data points
const pointsG = plotArea.append('g').attr('class', 'data-points');

// Axis labels
mainSvg.append('text').attr('x', PLOT_W / 2).attr('y', PLOT_H - 2)
  .attr('text-anchor', 'middle').attr('fill', '#888').attr('font-size', '12px').text('x');
mainSvg.append('text').attr('transform', `translate(12,${PLOT_H / 2}) rotate(-90)`)
  .attr('text-anchor', 'middle').attr('fill', '#888').attr('font-size', '12px').text('y');

// Click-to-add points
mainSvg.append('rect')
  .attr('x', MARGIN.left).attr('y', MARGIN.top)
  .attr('width', PLOT_W - MARGIN.left - MARGIN.right)
  .attr('height', PLOT_H - MARGIN.top - MARGIN.bottom)
  .attr('fill', 'transparent').attr('cursor', 'crosshair')
  .on('click', function(event: MouseEvent) {
    const [mx, my] = d3.pointer(event);
    const px = xScale.invert(mx);
    const py = yScale.invert(my);
    if (px < X_RANGE[0] || px > X_RANGE[1]) return;
    // If near an existing point, remove it
    let removed = false;
    const thresh = 0.08;
    const before = data.length;
    data = data.filter(d => Math.abs(d.x - px) > thresh || Math.abs(d.y - py) > thresh * 4);
    if (data.length === before) {
      data.push({ x: px, y: py });
    } else {
      removed = true;
    }
    renderPoints();
    // Partial reset of nets to relearn
    if (!removed) partialReset();
  });

// ── Loss chart ───────────────────────────────────────────────────────────────
const lossSvg = d3.select('#loss-plot').attr('width', LOSS_W).attr('height', LOSS_H);
lossSvg.append('rect').attr('width', LOSS_W).attr('height', LOSS_H).attr('fill', '#12151f').attr('rx', 4);
const lossPath2 = lossSvg.append('path').attr('fill', 'none').attr('stroke', '#f08050').attr('stroke-width', 1.5);
lossSvg.append('text').attr('x', LOSS_W / 2).attr('y', LOSS_H - 2).attr('text-anchor', 'middle')
  .attr('fill', '#666').attr('font-size', '10px').text('Steps');
lossSvg.append('text').attr('x', 30).attr('y', 14).attr('fill', '#888').attr('font-size', '10px').text('Loss');

// ── Net creation ─────────────────────────────────────────────────────────────

function createNets(): void {
  const count = cfg.method === 'ensemble' ? Math.max(2, Math.min(cfg.numSamples, 10)) : 1;
  nets = [];
  for (let i = 0; i < count; i++) {
    nets.push(new MLP(cfg.hiddenSize, cfg.dropoutRate));
  }
  lossHistory = [];
  trainStep = 0;
}

function partialReset(): void {
  // Don't fully reset; just let training continue
}

// ── Training loop ─────────────────────────────────────────────────────────────

function trainBatch(stepsPerFrame: number): void {
  if (data.length < 2) return;
  let totalLoss = 0;
  for (let s = 0; s < stepsPerFrame; s++) {
    let batchLoss = 0;
    for (let ni = 0; ni < nets.length; ni++) {
      // Shuffle data indices
      const indices = Array.from({length: data.length}, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
      }
      for (let idx = 0; idx < indices.length; idx++) {
        const pt = data[indices[idx]];
        const pred = nets[ni].forward(pt.x, true);
        const loss = nets[ni].backward(pt.x, pred, pt.y);
        nets[ni].adamUpdate(cfg.lr);
        batchLoss += loss;
      }
    }
    totalLoss = batchLoss / (data.length * nets.length);
    trainStep++;
  }
  lossHistory.push(totalLoss);
  if (lossHistory.length > 300) lossHistory.shift();
}

// ── Render ────────────────────────────────────────────────────────────────────

function renderPrediction(): void {
  const xs = predXs();
  let result: PredResult;
  if (cfg.method === 'mc-dropout') {
    result = predictMCDropout(nets[0], xs, cfg.numSamples);
  } else {
    result = predictEnsemble(nets, xs);
  }

  const lineGen = d3.line<{x: number; y: number}>()
    .x(d => xScale(d.x)).y(d => yScale(d.y))
    .curve(d3.curveCatmullRom);

  // ±2σ area
  const area2 = d3.area<number>()
    .x((_, i) => xScale(xs[i]))
    .y0(i => yScale(Math.max(-2.5, result.mean[i] - 2 * result.std[i])))
    .y1(i => yScale(Math.min(2.5, result.mean[i] + 2 * result.std[i])))
    .curve(d3.curveCatmullRom);
  band2.attr('d', area2(Array.from({length: xs.length}, (_, i) => i)));

  // ±1σ area
  const area1 = d3.area<number>()
    .x((_, i) => xScale(xs[i]))
    .y0(i => yScale(Math.max(-2.5, result.mean[i] - result.std[i])))
    .y1(i => yScale(Math.min(2.5, result.mean[i] + result.std[i])))
    .curve(d3.curveCatmullRom);
  band1.attr('d', area1(Array.from({length: xs.length}, (_, i) => i)));

  // Sample curves
  const samplePaths = samplesG.selectAll<SVGPathElement, number[]>('path')
    .data(result.samples);
  samplePaths.enter().append('path')
    .merge(samplePaths)
    .attr('fill', 'none')
    .attr('stroke', 'rgba(150,200,255,0.20)')
    .attr('stroke-width', 1)
    .attr('d', (samp) => {
      const pts = xs.map((x, i) => ({ x, y: samp[i] }));
      return lineGen(pts) || '';
    });
  samplePaths.exit().remove();

  // Mean curve
  meanPath.attr('d', lineGen(xs.map((x, i) => ({ x, y: result.mean[i] }))) || '');
}

function renderTrueFn(): void {
  const xs = predXs();
  const pts = xs.map(x => ({ x, y: targetFn(cfg.preset, x) }));
  const lineGen = d3.line<{x: number; y: number}>()
    .x(d => xScale(d.x)).y(d => yScale(d.y))
    .curve(d3.curveCatmullRom);
  truePath.attr('d', lineGen(pts) || '');
}

function renderPoints(): void {
  const circles = pointsG.selectAll<SVGCircleElement, DataPoint>('circle').data(data);
  circles.enter().append('circle')
    .attr('r', 5)
    .attr('fill', '#ffcc55')
    .attr('stroke', '#333').attr('stroke-width', 1)
    .merge(circles)
    .attr('cx', d => xScale(d.x))
    .attr('cy', d => yScale(d.y));
  circles.exit().remove();
}

function renderLoss(): void {
  if (lossHistory.length < 2) return;
  const N = lossHistory.length;
  lossXScale.domain([0, N - 1]);
  lossYScale.domain([0, Math.max(0.01, d3.max(lossHistory) || 0.1)]);

  const lineGen = d3.line<number>()
    .x((_, i) => lossXScale(i))
    .y(d => lossYScale(d))
    .curve(d3.curveBasis);
  lossPath2.attr('d', lineGen(lossHistory) || '');
}

function renderStepCount(): void {
  const el = document.getElementById('step-count');
  if (el) el.textContent = `Step: ${trainStep}`;
}

function renderAll(): void {
  if (data.length >= 2) renderPrediction();
  renderPoints();
  renderLoss();
  renderStepCount();
}

// ── Animation loop ────────────────────────────────────────────────────────────

function tick(): void {
  if (!running) return;
  trainBatch(5);
  renderAll();
  animFrame = requestAnimationFrame(tick);
}

function startTraining(): void {
  if (running) return;
  running = true;
  animFrame = requestAnimationFrame(tick);
  (document.getElementById('btn-play') as HTMLButtonElement).disabled = true;
  (document.getElementById('btn-pause') as HTMLButtonElement).disabled = false;
}

function pauseTraining(): void {
  running = false;
  if (animFrame !== null) cancelAnimationFrame(animFrame);
  animFrame = null;
  (document.getElementById('btn-play') as HTMLButtonElement).disabled = false;
  (document.getElementById('btn-pause') as HTMLButtonElement).disabled = true;
}

function stepOnce(): void {
  if (running) pauseTraining();
  trainBatch(1);
  renderAll();
}

function resetAll(): void {
  pauseTraining();
  data = makeDataset(cfg.preset);
  createNets();
  renderTrueFn();
  gapRect.attr('x', xScale(-0.25)).attr('width', xScale(0.25) - xScale(-0.25));
  renderAll();
}

function clearData(): void {
  pauseTraining();
  data = [];
  createNets();
  renderPoints();
  lossHistory = [];
  renderLoss();
}

// ── Controls wiring ───────────────────────────────────────────────────────────

function readControls(): void {
  const lr = document.getElementById('ctrl-lr') as HTMLInputElement;
  const hs = document.getElementById('ctrl-hidden') as HTMLInputElement;
  const dr = document.getElementById('ctrl-dropout') as HTMLInputElement;
  const ns = document.getElementById('ctrl-samples') as HTMLInputElement;
  const method = document.getElementById('ctrl-method') as HTMLSelectElement;
  const preset = document.getElementById('ctrl-preset') as HTMLSelectElement;

  cfg.lr = parseFloat(lr.value);
  cfg.hiddenSize = parseInt(hs.value);
  cfg.dropoutRate = parseFloat(dr.value);
  cfg.numSamples = parseInt(ns.value);
  cfg.method = method.value as Method;
  cfg.preset = preset.value as Preset;
}

function syncLabels(): void {
  const fields: Array<[string, string]> = [
    ['ctrl-lr', 'lbl-lr'],
    ['ctrl-hidden', 'lbl-hidden'],
    ['ctrl-dropout', 'lbl-dropout'],
    ['ctrl-samples', 'lbl-samples'],
  ];
  fields.forEach(([inputId, lblId]) => {
    const inp = document.getElementById(inputId) as HTMLInputElement;
    const lbl = document.getElementById(lblId);
    if (inp && lbl) lbl.textContent = inp.value;
  });
}

function wireControls(): void {
  const sliders = ['ctrl-lr', 'ctrl-hidden', 'ctrl-dropout', 'ctrl-samples'];
  sliders.forEach(id => {
    const el = document.getElementById(id) as HTMLInputElement;
    el.addEventListener('input', () => { syncLabels(); readControls(); });
  });

  const dropoutRow = document.getElementById('dropout-row') as HTMLElement;
  const samplesLabel = document.getElementById('samples-label') as HTMLElement;

  const methodSel = document.getElementById('ctrl-method') as HTMLSelectElement;
  methodSel.addEventListener('change', () => {
    readControls();
    // Toggle dropout visibility
    const isDropout = cfg.method === 'mc-dropout';
    dropoutRow.style.display = isDropout ? 'flex' : 'none';
    samplesLabel.textContent = isDropout ? 'MC Samples' : 'Ensemble Members';
    resetNets();
  });

  const presetSel = document.getElementById('ctrl-preset') as HTMLSelectElement;
  presetSel.addEventListener('change', () => {
    readControls();
    resetAll();
  });

  document.getElementById('btn-play')!.addEventListener('click', startTraining);
  document.getElementById('btn-pause')!.addEventListener('click', pauseTraining);
  document.getElementById('btn-step')!.addEventListener('click', stepOnce);
  document.getElementById('btn-reset')!.addEventListener('click', resetAll);
  document.getElementById('btn-clear')!.addEventListener('click', clearData);
}

function resetNets(): void {
  pauseTraining();
  createNets();
  lossHistory = [];
  renderAll();
}

// ── Startup ───────────────────────────────────────────────────────────────────

function init(): void {
  readControls();
  syncLabels();
  wireControls();
  data = makeDataset(cfg.preset);
  createNets();
  renderTrueFn();
  renderAll();
  // Auto-start
  startTraining();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
