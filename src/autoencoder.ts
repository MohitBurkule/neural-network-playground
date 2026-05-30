// autoencoder.ts — Fully-connected Autoencoder visualization lab
// Encoder -> 2D bottleneck -> Decoder, trained with SGD + MSE
// Visualizes: originals vs reconstructions, 2D latent space, latent grid decode, loss curve

import * as d3 from 'd3';

// ─── Seeded RNG ───────────────────────────────────────────────────────────────

let _seed = 42;
function lcg(): number {
  _seed = (_seed * 1664525 + 1013904223) >>> 0;
  return _seed / 0xffffffff;
}
function randNorm(): number {
  const u1 = lcg() + 1e-10;
  const u2 = lcg() + 1e-10;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
function resetSeed(): void { _seed = 42; }

// ─── Dataset: 8x8 grayscale shapes (5 classes) ───────────────────────────────

const IMG_SIZE = 8;
const NUM_INPUT = IMG_SIZE * IMG_SIZE; // 64
const NUM_CLASSES = 5;
const CLASS_NAMES = ['Square', 'Circle', 'Cross', 'Diamond', 'Triangle'];
const SAMPLES_PER_CLASS = 20;

interface Example {
  pixels: number[]; // 64 floats [0,1]
  label: number;
}

function makeSquare(cx: number, cy: number, r: number): number[] {
  const px = new Array(NUM_INPUT).fill(0);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      if (Math.abs(x - cx) <= r && Math.abs(y - cy) <= r) {
        px[y * IMG_SIZE + x] = 1;
      }
    }
  }
  return px;
}

function makeCircle(cx: number, cy: number, r: number): number[] {
  const px = new Array(NUM_INPUT).fill(0);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
        px[y * IMG_SIZE + x] = 1;
      }
    }
  }
  return px;
}

function makeCross(cx: number, cy: number, r: number): number[] {
  const px = new Array(NUM_INPUT).fill(0);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      if (Math.abs(x - cx) <= 1 && Math.abs(y - cy) <= r) {
        px[y * IMG_SIZE + x] = 1;
      }
      if (Math.abs(y - cy) <= 1 && Math.abs(x - cx) <= r) {
        px[y * IMG_SIZE + x] = 1;
      }
    }
  }
  return px;
}

function makeDiamond(cx: number, cy: number, r: number): number[] {
  const px = new Array(NUM_INPUT).fill(0);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      if (Math.abs(x - cx) + Math.abs(y - cy) <= r) {
        px[y * IMG_SIZE + x] = 1;
      }
    }
  }
  return px;
}

function makeTriangle(cx: number, cy: number, r: number): number[] {
  const px = new Array(NUM_INPUT).fill(0);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const dy = y - (cy - r);
      const maxDx = (dy / (2 * r)) * r;
      if (dy >= 0 && dy <= 2 * r && Math.abs(x - cx) <= maxDx) {
        px[y * IMG_SIZE + x] = 1;
      }
    }
  }
  return px;
}

function addNoise(px: number[], amount: number): number[] {
  return px.map(v => Math.max(0, Math.min(1, v + (lcg() - 0.5) * amount)));
}

function generateDataset(): Example[] {
  const examples: Example[] = [];
  const offsets = [0, -1, 1, -0.5, 0.5];
  for (let s = 0; s < SAMPLES_PER_CLASS; s++) {
    const jitterX = (lcg() - 0.5) * 2;
    const jitterY = (lcg() - 0.5) * 2;
    const cx = 3.5 + jitterX;
    const cy = 3.5 + jitterY;
    const r = 2 + (lcg() - 0.5) * 0.5;

    examples.push({ pixels: makeSquare(cx, cy, r), label: 0 });
    examples.push({ pixels: makeCircle(cx, cy, r), label: 1 });
    examples.push({ pixels: makeCross(cx, cy, r + 0.5), label: 2 });
    examples.push({ pixels: makeDiamond(cx, cy, r + 1), label: 3 });
    examples.push({ pixels: makeTriangle(cx, cy, r), label: 4 });
  }
  return examples;
}

// ─── Neural network: fully-connected layer ────────────────────────────────────

interface FCLayer {
  w: number[][];  // [outN][inN]
  b: number[];    // [outN]
  dw: number[][];
  db: number[];
  inN: number;
  outN: number;
}

function makeFC(inN: number, outN: number, scale: number = 0.1): FCLayer {
  const w: number[][] = [];
  const dw: number[][] = [];
  for (let o = 0; o < outN; o++) {
    const row: number[] = [];
    const drow: number[] = [];
    for (let i = 0; i < inN; i++) {
      row.push(randNorm() * scale);
      drow.push(0);
    }
    w.push(row);
    dw.push(drow);
  }
  return { w, b: new Array(outN).fill(0), dw, db: new Array(outN).fill(0), inN, outN };
}

function fcForward(layer: FCLayer, x: number[]): number[] {
  const out: number[] = [];
  for (let o = 0; o < layer.outN; o++) {
    let s = layer.b[o];
    for (let i = 0; i < layer.inN; i++) s += layer.w[o][i] * x[i];
    out.push(s);
  }
  return out;
}

function fcBackward(layer: FCLayer, x: number[], dOut: number[]): number[] {
  const dIn = new Array(layer.inN).fill(0);
  for (let o = 0; o < layer.outN; o++) {
    layer.db[o] += dOut[o];
    for (let i = 0; i < layer.inN; i++) {
      layer.dw[o][i] += dOut[o] * x[i];
      dIn[i] += dOut[o] * layer.w[o][i];
    }
  }
  return dIn;
}

function fcUpdate(layer: FCLayer, lr: number): void {
  for (let o = 0; o < layer.outN; o++) {
    layer.b[o] -= lr * layer.db[o];
    layer.db[o] = 0;
    for (let i = 0; i < layer.inN; i++) {
      layer.w[o][i] -= lr * layer.dw[o][i];
      layer.dw[o][i] = 0;
    }
  }
}

function relu(x: number[]): number[] {
  return x.map(v => Math.max(0, v));
}

function reluGrad(x: number[], dOut: number[]): number[] {
  return x.map((v, i) => v > 0 ? dOut[i] : 0);
}

function sigmoid(x: number[]): number[] {
  return x.map(v => 1 / (1 + Math.exp(-v)));
}

function sigmoidGrad(sig: number[], dOut: number[]): number[] {
  return sig.map((s, i) => dOut[i] * s * (1 - s));
}

// ─── Autoencoder ──────────────────────────────────────────────────────────────

interface AutoencoderConfig {
  hiddenSize: number;
  tiedWeights: boolean;
  denoise: boolean;
  denoiseAmount: number;
}

interface Autoencoder {
  // Encoder: input -> hidden -> latent(2)
  enc1: FCLayer;
  enc2: FCLayer;
  // Decoder: latent(2) -> hidden -> output
  dec1: FCLayer;
  dec2: FCLayer;
  config: AutoencoderConfig;
}

interface ForwardCache {
  input: number[];
  noisyInput: number[];
  enc1Pre: number[];
  enc1Act: number[];
  enc2: number[]; // latent (2D)
  dec1Pre: number[];
  dec1Act: number[];
  dec2Pre: number[];
  output: number[]; // sigmoid output
}

function makeAutoencoder(cfg: AutoencoderConfig): Autoencoder {
  const h = cfg.hiddenSize;
  return {
    enc1: makeFC(NUM_INPUT, h, 0.05),
    enc2: makeFC(h, 2, 0.05),
    dec1: makeFC(2, h, 0.05),
    dec2: makeFC(h, NUM_INPUT, 0.05),
    config: cfg,
  };
}

function aeForward(ae: Autoencoder, input: number[], training: boolean): ForwardCache {
  const noisyInput = training && ae.config.denoise
    ? addNoise(input, ae.config.denoiseAmount)
    : input.slice();

  const enc1Pre = fcForward(ae.enc1, noisyInput);
  const enc1Act = relu(enc1Pre);
  const enc2 = fcForward(ae.enc2, enc1Act); // raw latent, no activation
  const dec1Pre = fcForward(ae.dec1, enc2);
  const dec1Act = relu(dec1Pre);
  const dec2Pre = fcForward(ae.dec2, dec1Act);
  const output = sigmoid(dec2Pre); // output in [0,1]

  return { input, noisyInput, enc1Pre, enc1Act, enc2, dec1Pre, dec1Act, dec2Pre, output };
}

function aeBackward(ae: Autoencoder, cache: ForwardCache): number {
  const target = cache.input; // reconstruct clean input
  const n = NUM_INPUT;

  // MSE loss
  let loss = 0;
  const dOut: number[] = [];
  for (let i = 0; i < n; i++) {
    const diff = cache.output[i] - target[i];
    loss += diff * diff;
    dOut.push((2 / n) * diff);
  }
  loss /= n;

  // Backprop through sigmoid
  const dDec2Pre = sigmoidGrad(cache.output, dOut);
  // Backprop through dec2
  const dDec1Act = fcBackward(ae.dec2, cache.dec1Act, dDec2Pre);
  // Backprop through relu
  const dDec1Pre = reluGrad(cache.dec1Pre, dDec1Act);
  // Backprop through dec1
  const dEnc2 = fcBackward(ae.dec1, cache.enc2, dDec1Pre);

  // If tied weights: dec1.w = enc2.w^T, so we add gradients symmetrically
  // (optional: for simplicity we just train both independently unless tied)
  if (ae.config.tiedWeights) {
    // dec1 gradients flow back into enc2 transpose
    for (let o = 0; o < ae.dec1.outN; o++) {
      for (let i = 0; i < ae.dec1.inN; i++) {
        ae.enc2.dw[i][o] += ae.dec1.dw[o][i];
      }
    }
    // zero dec1 dw since it's tied
    for (let o = 0; o < ae.dec1.outN; o++) {
      for (let i = 0; i < ae.dec1.inN; i++) {
        ae.dec1.dw[o][i] = 0;
      }
    }
  }

  // Backprop through enc2
  const dEnc1Act = fcBackward(ae.enc2, cache.enc1Act, dEnc2);
  // Backprop through relu
  const dEnc1Pre = reluGrad(cache.enc1Pre, dEnc1Act);
  // Backprop through enc1
  fcBackward(ae.enc1, cache.noisyInput, dEnc1Pre);

  return loss;
}

function aeUpdate(ae: Autoencoder, lr: number): void {
  fcUpdate(ae.enc1, lr);
  fcUpdate(ae.enc2, lr);
  if (!ae.config.tiedWeights) {
    fcUpdate(ae.dec1, lr);
  } else {
    // sync dec1 weights from enc2 transpose
    for (let o = 0; o < ae.dec1.outN; o++) {
      for (let i = 0; i < ae.dec1.inN; i++) {
        ae.dec1.w[o][i] = ae.enc2.w[i][o];
      }
      ae.dec1.db[o] = 0;
    }
  }
  fcUpdate(ae.dec2, lr);
  // If tied, also sync dec2 from enc1 transpose
  if (ae.config.tiedWeights) {
    for (let o = 0; o < ae.dec2.outN; o++) {
      for (let i = 0; i < ae.dec2.inN; i++) {
        ae.dec2.w[o][i] = ae.enc1.w[i][o];
      }
      ae.dec2.db[o] = 0;
    }
  }
}

function aeEncode(ae: Autoencoder, input: number[]): number[] {
  const enc1Pre = fcForward(ae.enc1, input);
  const enc1Act = relu(enc1Pre);
  return fcForward(ae.enc2, enc1Act);
}

function aeDecode(ae: Autoencoder, latent: number[]): number[] {
  const dec1Pre = fcForward(ae.dec1, latent);
  const dec1Act = relu(dec1Pre);
  const dec2Pre = fcForward(ae.dec2, dec1Act);
  return sigmoid(dec2Pre);
}

// ─── State ────────────────────────────────────────────────────────────────────

const CLASS_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'];

let dataset: Example[] = [];
let ae: Autoencoder;
let step = 0;
let running = false;
let animHandle: number | null = null;
let lossHistory: number[] = [];
let latentCache: Array<{ z: number[]; label: number }> = [];
const BATCH_SIZE = 10;
const STEPS_PER_FRAME = 5;

// ─── DOM references ───────────────────────────────────────────────────────────

const btnPlay = document.getElementById('btn-play') as HTMLButtonElement;
const btnPause = document.getElementById('btn-pause') as HTMLButtonElement;
const btnStep = document.getElementById('btn-step') as HTMLButtonElement;
const btnReset = document.getElementById('btn-reset') as HTMLButtonElement;
const lrSlider = document.getElementById('lr-slider') as HTMLInputElement;
const lrDisplay = document.getElementById('lr-display') as HTMLSpanElement;
const hiddenSlider = document.getElementById('hidden-slider') as HTMLInputElement;
const hiddenDisplay = document.getElementById('hidden-display') as HTMLSpanElement;
const denoiseToggle = document.getElementById('denoise-toggle') as HTMLInputElement;
const tiedToggle = document.getElementById('tied-toggle') as HTMLInputElement;
const stepCount = document.getElementById('step-count') as HTMLElement;
const lossDisplay = document.getElementById('loss-display') as HTMLElement;

// Canvas elements
const origCanvas = document.getElementById('orig-canvas') as HTMLCanvasElement;
const reconCanvas = document.getElementById('recon-canvas') as HTMLCanvasElement;
const latentSvgEl = document.getElementById('latent-svg') as unknown as SVGSVGElement;
const gridContainer = document.getElementById('grid-container') as HTMLDivElement;
const lossChartSvg = document.getElementById('loss-chart') as unknown as SVGSVGElement;

// Latent probe (click/drag)
let probeLatent: number[] | null = null;
const probeCanvas = document.getElementById('probe-canvas') as HTMLCanvasElement;

// ─── Canvas drawing ───────────────────────────────────────────────────────────

const PIXEL_SIZE = 20; // px per pixel in the 8x8 images

function drawPixels(canvas: HTMLCanvasElement, pixels: number[], color?: string): void {
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const v = pixels[y * IMG_SIZE + x];
      ctx.fillStyle = color
        ? d3.interpolateRgb('#000', color)(v)
        : `rgb(${Math.round(v * 255)},${Math.round(v * 255)},${Math.round(v * 255)})`;
      ctx.fillRect(x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
    }
  }
}

function drawSmallPixels(canvas: HTMLCanvasElement, pixels: number[], ps: number, color?: string): void {
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const v = pixels[y * IMG_SIZE + x];
      ctx.fillStyle = color
        ? d3.interpolateRgb('#111', color)(v)
        : `rgb(${Math.round(v * 255)},${Math.round(v * 255)},${Math.round(v * 255)})`;
      ctx.fillRect(x * ps, y * ps, ps, ps);
    }
  }
}

// ─── Latent space scatter ─────────────────────────────────────────────────────

const LATENT_W = 340;
const LATENT_H = 280;
let latentXScale: d3.ScaleLinear<number, number>;
let latentYScale: d3.ScaleLinear<number, number>;
const latentSvg = d3.select(latentSvgEl);

function setupLatentSvg(): void {
  latentSvg.selectAll('*').remove();
  latentSvg.attr('width', LATENT_W).attr('height', LATENT_H);

  latentXScale = d3.scaleLinear().domain([-4, 4]).range([30, LATENT_W - 10]);
  latentYScale = d3.scaleLinear().domain([-4, 4]).range([LATENT_H - 25, 10]);

  latentSvg.append('g')
    .attr('class', 'x-axis')
    .attr('transform', `translate(0,${LATENT_H - 25})`)
    .call(d3.axisBottom(latentXScale).ticks(5).tickSize(3));

  latentSvg.append('g')
    .attr('class', 'y-axis')
    .attr('transform', 'translate(30,0)')
    .call(d3.axisLeft(latentYScale).ticks(5).tickSize(3));

  latentSvg.append('g').attr('class', 'dots');

  // Legend
  const leg = latentSvg.append('g').attr('transform', `translate(${LATENT_W - 95}, 10)`);
  CLASS_NAMES.forEach((name, i) => {
    leg.append('circle').attr('cx', 6).attr('cy', i * 16 + 6).attr('r', 5).attr('fill', CLASS_COLORS[i]);
    leg.append('text').attr('x', 14).attr('y', i * 16 + 10).attr('font-size', '9px').attr('fill', '#aaa').text(name);
  });

  // Probe crosshair group
  latentSvg.append('g').attr('class', 'probe');
}

function updateLatentScatter(): void {
  if (latentCache.length === 0) return;

  // Recompute domain from current latents
  const zx = latentCache.map(d => d.z[0]);
  const zy = latentCache.map(d => d.z[1]);
  const xMin = Math.min(...zx), xMax = Math.max(...zx);
  const yMin = Math.min(...zy), yMax = Math.max(...zy);
  const xPad = (xMax - xMin) * 0.2 + 0.5;
  const yPad = (yMax - yMin) * 0.2 + 0.5;

  latentXScale.domain([xMin - xPad, xMax + xPad]);
  latentYScale.domain([yMin - yPad, yMax + yPad]);

  latentSvg.select<SVGGElement>('g.x-axis')
    .call(d3.axisBottom(latentXScale).ticks(5).tickSize(3) as any);
  latentSvg.select<SVGGElement>('g.y-axis')
    .call(d3.axisLeft(latentYScale).ticks(5).tickSize(3) as any);

  const dots = latentSvg.select('g.dots')
    .selectAll<SVGCircleElement, typeof latentCache[0]>('circle')
    .data(latentCache);

  dots.enter().append('circle')
    .attr('r', 4)
    .attr('opacity', 0.8)
    .merge(dots)
    .attr('cx', d => latentXScale(d.z[0]))
    .attr('cy', d => latentYScale(d.z[1]))
    .attr('fill', d => CLASS_COLORS[d.label]);

  dots.exit().remove();

  // Update probe crosshair
  if (probeLatent) {
    const pg = latentSvg.select('g.probe');
    pg.selectAll('*').remove();
    const px = latentXScale(probeLatent[0]);
    const py = latentYScale(probeLatent[1]);
    pg.append('line').attr('x1', px).attr('y1', 10).attr('x2', px).attr('y2', LATENT_H - 25)
      .attr('stroke', '#fff').attr('stroke-width', 1).attr('stroke-dasharray', '3,3').attr('opacity', 0.5);
    pg.append('line').attr('x1', 30).attr('y1', py).attr('x2', LATENT_W - 10).attr('y2', py)
      .attr('stroke', '#fff').attr('stroke-width', 1).attr('stroke-dasharray', '3,3').attr('opacity', 0.5);
    pg.append('circle').attr('cx', px).attr('cy', py).attr('r', 6)
      .attr('fill', 'none').attr('stroke', '#fff').attr('stroke-width', 2);
  }
}

// ─── Latent grid ─────────────────────────────────────────────────────────────

const GRID_N = 7; // 7x7 grid
const GRID_PS = 4; // pixel size for each mini-image

function updateLatentGrid(): void {
  if (latentCache.length === 0) return;

  const zx = latentCache.map(d => d.z[0]);
  const zy = latentCache.map(d => d.z[1]);
  const xMin = Math.min(...zx), xMax = Math.max(...zx);
  const yMin = Math.min(...zy), yMax = Math.max(...zy);
  const xPad = (xMax - xMin) * 0.15 + 0.3;
  const yPad = (yMax - yMin) * 0.15 + 0.3;

  const xs = d3.range(GRID_N).map(i => xMin - xPad + (i / (GRID_N - 1)) * (xMax - xMin + 2 * xPad));
  const ys = d3.range(GRID_N).map(i => yMax + yPad - (i / (GRID_N - 1)) * (yMax - yMin + 2 * yPad));

  // Ensure we have GRID_N*GRID_N canvases
  const needed = GRID_N * GRID_N;
  while (gridContainer.children.length < needed) {
    const c = document.createElement('canvas');
    c.width = IMG_SIZE * GRID_PS;
    c.height = IMG_SIZE * GRID_PS;
    c.style.imageRendering = 'pixelated';
    c.style.border = '1px solid #2a2a4a';
    c.style.borderRadius = '2px';
    gridContainer.appendChild(c);
  }

  let idx = 0;
  for (let gy = 0; gy < GRID_N; gy++) {
    for (let gx = 0; gx < GRID_N; gx++) {
      const latent = [xs[gx], ys[gy]];
      const decoded = aeDecode(ae, latent);
      const canvas = gridContainer.children[idx] as HTMLCanvasElement;
      drawSmallPixels(canvas, decoded, GRID_PS);
      idx++;
    }
  }
}

// ─── Loss chart ───────────────────────────────────────────────────────────────

const CHART_W = 220;
const CHART_H = 100;
const chartSvg = d3.select(lossChartSvg).attr('width', CHART_W).attr('height', CHART_H);
const chartMargin = { top: 8, right: 10, bottom: 20, left: 38 };
const chartInnerW = CHART_W - chartMargin.left - chartMargin.right;
const chartInnerH = CHART_H - chartMargin.top - chartMargin.bottom;

const chartG = chartSvg.append('g').attr('transform', `translate(${chartMargin.left},${chartMargin.top})`);
chartG.append('g').attr('class', 'x-axis').attr('transform', `translate(0,${chartInnerH})`);
chartG.append('g').attr('class', 'y-axis');
chartG.append('path').attr('class', 'loss-line').attr('fill', 'none').attr('stroke', '#e67e22').attr('stroke-width', 1.5);

function updateLossChart(): void {
  if (lossHistory.length < 2) return;
  const maxShow = 200;
  const data = lossHistory.length > maxShow ? lossHistory.slice(-maxShow) : lossHistory;

  const xScale = d3.scaleLinear().domain([0, data.length - 1]).range([0, chartInnerW]);
  const yScale = d3.scaleLinear().domain([0, Math.max(0.01, d3.max(data)!)]).range([chartInnerH, 0]);

  chartG.select<SVGGElement>('g.x-axis').call(d3.axisBottom(xScale).ticks(4).tickSize(3) as any);
  chartG.select<SVGGElement>('g.y-axis').call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format('.3f')).tickSize(3) as any);

  const line = d3.line<number>().x((_, i) => xScale(i)).y(d => yScale(d));
  chartG.select('path.loss-line').attr('d', line(data)!);
}

// ─── Example display ──────────────────────────────────────────────────────────

let displayIdx = 0;

function updateExampleDisplay(): void {
  if (!dataset.length) return;
  const ex = dataset[displayIdx % dataset.length];
  const cache = aeForward(ae, ex.pixels, false);
  drawPixels(origCanvas, ex.pixels);
  drawPixels(reconCanvas, cache.output);
  const label = document.getElementById('example-label')!;
  label.textContent = `Class: ${CLASS_NAMES[ex.label]}`;
}

// ─── Training step ────────────────────────────────────────────────────────────

function trainStep(): void {
  const lr = parseFloat(lrSlider.value);
  let totalLoss = 0;

  // Sample a random batch
  for (let b = 0; b < BATCH_SIZE; b++) {
    const idx = Math.floor(lcg() * dataset.length);
    const ex = dataset[idx];
    const cache = aeForward(ae, ex.pixels, true);
    totalLoss += aeBackward(ae, cache);
    aeUpdate(ae, lr / BATCH_SIZE);
  }

  step++;
  lossHistory.push(totalLoss / BATCH_SIZE);
  stepCount.textContent = `Step: ${step}`;
  lossDisplay.textContent = `Loss: ${(totalLoss / BATCH_SIZE).toFixed(4)}`;
}

function updateAllLatents(): void {
  latentCache = dataset.map(ex => ({
    z: aeEncode(ae, ex.pixels),
    label: ex.label,
  }));
}

function fullRender(): void {
  updateAllLatents();
  updateExampleDisplay();
  updateLatentScatter();
  updateLatentGrid();
  updateLossChart();
}

function animLoop(): void {
  if (!running) return;
  for (let i = 0; i < STEPS_PER_FRAME; i++) trainStep();
  fullRender();
  animHandle = requestAnimationFrame(animLoop);
}

// ─── Probe interaction ────────────────────────────────────────────────────────

function setupProbeInteraction(): void {
  const svg = latentSvgEl as unknown as SVGSVGElement;

  function handleLatentClick(event: MouseEvent): void {
    const rect = (svg as unknown as Element).getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const lx = latentXScale.invert(mx);
    const ly = latentYScale.invert(my);
    probeLatent = [lx, ly];

    const decoded = aeDecode(ae, probeLatent);
    drawSmallPixels(probeCanvas, decoded, 10);
    updateLatentScatter();
  }

  (svg as unknown as Element).addEventListener('mousedown', handleLatentClick as EventListener);
  (svg as unknown as Element).addEventListener('mousemove', (e: Event) => {
    if ((e as MouseEvent).buttons > 0) handleLatentClick(e as MouseEvent);
  });
}

// ─── Init & controls ──────────────────────────────────────────────────────────

function getConfig(): AutoencoderConfig {
  return {
    hiddenSize: parseInt(hiddenSlider.value),
    tiedWeights: tiedToggle.checked,
    denoise: denoiseToggle.checked,
    denoiseAmount: 0.3,
  };
}

function initAll(): void {
  resetSeed();
  dataset = generateDataset();
  ae = makeAutoencoder(getConfig());
  step = 0;
  lossHistory = [];
  probeLatent = null;

  stepCount.textContent = 'Step: 0';
  lossDisplay.textContent = 'Loss: —';

  // Clear probe canvas
  const pctx = probeCanvas.getContext('2d')!;
  pctx.fillStyle = '#0f0f1a';
  pctx.fillRect(0, 0, probeCanvas.width, probeCanvas.height);
  pctx.fillStyle = '#555';
  pctx.font = '9px sans-serif';
  pctx.fillText('click latent', 4, 42);
  pctx.fillText('space to probe', 2, 52);

  fullRender();
}

function setupExampleNav(): void {
  const prevBtn = document.getElementById('ex-prev') as HTMLButtonElement;
  const nextBtn = document.getElementById('ex-next') as HTMLButtonElement;
  prevBtn.addEventListener('click', () => {
    displayIdx = (displayIdx - 1 + dataset.length) % dataset.length;
    updateExampleDisplay();
  });
  nextBtn.addEventListener('click', () => {
    displayIdx = (displayIdx + 1) % dataset.length;
    updateExampleDisplay();
  });
}

// ─── Entry point ─────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  origCanvas.width = IMG_SIZE * PIXEL_SIZE;
  origCanvas.height = IMG_SIZE * PIXEL_SIZE;
  reconCanvas.width = IMG_SIZE * PIXEL_SIZE;
  reconCanvas.height = IMG_SIZE * PIXEL_SIZE;
  probeCanvas.width = IMG_SIZE * 10;
  probeCanvas.height = IMG_SIZE * 10;

  setupLatentSvg();
  setupProbeInteraction();
  setupExampleNav();

  lrSlider.addEventListener('input', () => {
    lrDisplay.textContent = parseFloat(lrSlider.value).toFixed(4);
  });
  hiddenSlider.addEventListener('input', () => {
    hiddenDisplay.textContent = hiddenSlider.value;
  });

  btnPlay.addEventListener('click', () => {
    if (running) return;
    running = true;
    animLoop();
  });
  btnPause.addEventListener('click', () => {
    running = false;
    if (animHandle !== null) { cancelAnimationFrame(animHandle); animHandle = null; }
  });
  btnStep.addEventListener('click', () => {
    running = false;
    if (animHandle !== null) { cancelAnimationFrame(animHandle); animHandle = null; }
    for (let i = 0; i < STEPS_PER_FRAME; i++) trainStep();
    fullRender();
  });
  btnReset.addEventListener('click', () => {
    running = false;
    if (animHandle !== null) { cancelAnimationFrame(animHandle); animHandle = null; }
    initAll();
  });

  initAll();
});
