// vae.ts — Variational Autoencoder visualization lab
// Encoder → μ+logσ² over 2D latent → reparameterization → Decoder
// ELBO = recon loss + β·KL divergence (beta-VAE)

import * as d3 from 'd3';

// ─── Seeded RNG ──────────────────────────────────────────────────────────────

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

// ─── Dataset: 8×8 binary glyphs (5 classes) ──────────────────────────────────

const IMG_SIZE = 8;
const NUM_INPUT = IMG_SIZE * IMG_SIZE;
const NUM_CLASSES = 5;
const CLASS_NAMES = ['Square', 'Circle', 'Cross', 'Diamond', 'Triangle'];
const SAMPLES_PER_CLASS = 20;

interface Example {
  pixels: number[];
  label: number;
}

function makeSquare(cx: number, cy: number, r: number): number[] {
  const px = new Array(NUM_INPUT).fill(0);
  for (let y = 0; y < IMG_SIZE; y++)
    for (let x = 0; x < IMG_SIZE; x++)
      if (Math.abs(x - cx) <= r && Math.abs(y - cy) <= r)
        px[y * IMG_SIZE + x] = 1;
  return px;
}
function makeCircle(cx: number, cy: number, r: number): number[] {
  const px = new Array(NUM_INPUT).fill(0);
  for (let y = 0; y < IMG_SIZE; y++)
    for (let x = 0; x < IMG_SIZE; x++)
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r)
        px[y * IMG_SIZE + x] = 1;
  return px;
}
function makeCross(cx: number, cy: number, r: number): number[] {
  const px = new Array(NUM_INPUT).fill(0);
  for (let y = 0; y < IMG_SIZE; y++)
    for (let x = 0; x < IMG_SIZE; x++)
      if (Math.abs(x - cx) <= 1 || Math.abs(y - cy) <= 1)
        if (Math.abs(x - cx) <= r && Math.abs(y - cy) <= r)
          px[y * IMG_SIZE + x] = 1;
  return px;
}
function makeDiamond(cx: number, cy: number, r: number): number[] {
  const px = new Array(NUM_INPUT).fill(0);
  for (let y = 0; y < IMG_SIZE; y++)
    for (let x = 0; x < IMG_SIZE; x++)
      if (Math.abs(x - cx) + Math.abs(y - cy) <= r)
        px[y * IMG_SIZE + x] = 1;
  return px;
}
function makeTriangle(cx: number, cy: number, r: number): number[] {
  const px = new Array(NUM_INPUT).fill(0);
  for (let y = 0; y < IMG_SIZE; y++)
    for (let x = 0; x < IMG_SIZE; x++) {
      const dy = y - (cy - r);
      if (dy >= 0 && dy <= 2 * r && Math.abs(x - cx) <= dy * r / (2 * r))
        px[y * IMG_SIZE + x] = 1;
    }
  return px;
}

function buildDataset(): Example[] {
  const ds: Example[] = [];
  for (let s = 0; s < SAMPLES_PER_CLASS; s++) {
    const n = (s % 3) + 2;
    const cx = 3 + (s % 2);
    const cy = 3 + Math.floor((s % 4) / 2);
    ds.push({ pixels: makeSquare(cx, cy, n),   label: 0 });
    ds.push({ pixels: makeCircle(cx, cy, n),   label: 1 });
    ds.push({ pixels: makeCross(cx, cy, n + 1), label: 2 });
    ds.push({ pixels: makeDiamond(cx, cy, n + 1), label: 3 });
    ds.push({ pixels: makeTriangle(cx, cy, n), label: 4 });
  }
  return ds;
}

// ─── Neural Network Primitives ───────────────────────────────────────────────

type Matrix = number[][];

function zeros(r: number, c: number): Matrix {
  const m: Matrix = [];
  for (let i = 0; i < r; i++) m.push(new Array(c).fill(0));
  return m;
}
function randMatrix(r: number, c: number, scale: number): Matrix {
  const m: Matrix = [];
  for (let i = 0; i < r; i++) {
    const row: number[] = [];
    for (let j = 0; j < c; j++) row.push(randNorm() * scale);
    m.push(row);
  }
  return m;
}
function relu(x: number): number { return x > 0 ? x : 0; }
function reluD(x: number): number { return x > 0 ? 1 : 0; }
function sigmoid(x: number): number { return 1 / (1 + Math.exp(-x)); }
function sigmoidD(x: number): number { const s = sigmoid(x); return s * (1 - s); }

interface Layer {
  W: Matrix; b: number[];
  dW: Matrix; db: number[];
  mW: Matrix; mb: number[];   // adam m
  vW: Matrix; vb: number[];   // adam v
  inSize: number; outSize: number;
}

function makeLayer(inSize: number, outSize: number): Layer {
  const sc = Math.sqrt(2 / inSize);
  return {
    W: randMatrix(outSize, inSize, sc),
    b: new Array(outSize).fill(0),
    dW: zeros(outSize, inSize),
    db: new Array(outSize).fill(0),
    mW: zeros(outSize, inSize), mb: new Array(outSize).fill(0),
    vW: zeros(outSize, inSize), vb: new Array(outSize).fill(0),
    inSize, outSize,
  };
}

function forwardLayer(layer: Layer, x: number[]): { z: number[]; a: number[]; x: number[] } {
  const z: number[] = new Array(layer.outSize).fill(0);
  for (let i = 0; i < layer.outSize; i++) {
    z[i] = layer.b[i];
    for (let j = 0; j < layer.inSize; j++) z[i] += layer.W[i][j] * x[j];
  }
  return { z, a: z.map(relu), x };
}
function forwardLayerSigmoid(layer: Layer, x: number[]): { z: number[]; a: number[]; x: number[] } {
  const z: number[] = new Array(layer.outSize).fill(0);
  for (let i = 0; i < layer.outSize; i++) {
    z[i] = layer.b[i];
    for (let j = 0; j < layer.inSize; j++) z[i] += layer.W[i][j] * x[j];
  }
  return { z, a: z.map(sigmoid), x };
}
function forwardLayerLinear(layer: Layer, x: number[]): { z: number[]; a: number[]; x: number[] } {
  const z: number[] = new Array(layer.outSize).fill(0);
  for (let i = 0; i < layer.outSize; i++) {
    z[i] = layer.b[i];
    for (let j = 0; j < layer.inSize; j++) z[i] += layer.W[i][j] * x[j];
  }
  return { z, a: z.slice(), x };
}

function adamUpdate(layer: Layer, lr: number, t: number): void {
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  const bc1 = 1 - Math.pow(beta1, t);
  const bc2 = 1 - Math.pow(beta2, t);
  for (let i = 0; i < layer.outSize; i++) {
    for (let j = 0; j < layer.inSize; j++) {
      layer.mW[i][j] = beta1 * layer.mW[i][j] + (1 - beta1) * layer.dW[i][j];
      layer.vW[i][j] = beta2 * layer.vW[i][j] + (1 - beta2) * layer.dW[i][j] ** 2;
      layer.W[i][j] -= lr * (layer.mW[i][j] / bc1) / (Math.sqrt(layer.vW[i][j] / bc2) + eps);
      layer.dW[i][j] = 0;
    }
    layer.mb[i] = beta1 * layer.mb[i] + (1 - beta1) * layer.db[i];
    layer.vb[i] = beta2 * layer.vb[i] + (1 - beta2) * layer.db[i] ** 2;
    layer.b[i] -= lr * (layer.mb[i] / bc1) / (Math.sqrt(layer.vb[i] / bc2) + eps);
    layer.db[i] = 0;
  }
}

// ─── VAE Model ───────────────────────────────────────────────────────────────

const LATENT_DIM = 2;

interface VAEState {
  encH: Layer;
  encMu: Layer;
  encLogVar: Layer;
  decH: Layer;
  decOut: Layer;
  step: number;
}

let vae: VAEState;
let hiddenSize = 32;

function buildVAE(h: number): VAEState {
  resetSeed();
  return {
    encH:      makeLayer(NUM_INPUT,  h),
    encMu:     makeLayer(h,          LATENT_DIM),
    encLogVar: makeLayer(h,          LATENT_DIM),
    decH:      makeLayer(LATENT_DIM, h),
    decOut:    makeLayer(h,          NUM_INPUT),
    step: 0,
  };
}

interface EncResult {
  hA: number[]; hZ: number[];
  mu: number[]; logVar: number[];
  z: number[];
  eps: number[];
}

function encode(x: number[]): EncResult {
  const { z: hZ, a: hA } = forwardLayer(vae.encH, x);
  const { a: mu }     = forwardLayerLinear(vae.encMu, hA);
  const { a: logVar } = forwardLayerLinear(vae.encLogVar, hA);
  const eps   = [randNorm(), randNorm()];
  const z     = mu.map((m, i) => m + Math.exp(0.5 * logVar[i]) * eps[i]);
  return { hA, hZ, mu, logVar, z, eps };
}

function decode(z: number[]): { dhA: number[]; dhZ: number[]; recon: number[] } {
  const { z: dhZ, a: dhA } = forwardLayer(vae.decH, z);
  const { a: recon }       = forwardLayerSigmoid(vae.decOut, dhA);
  return { dhA, dhZ, recon };
}

function trainStep(batch: Example[], lr: number, beta: number): { recon: number; kl: number } {
  vae.step++;
  let totalRecon = 0, totalKL = 0;
  const N = batch.length;

  batch.forEach(ex => {
    const enc = encode(ex.pixels);
    const dec = decode(enc.z);
    const { recon } = dec;

    // BCE reconstruction loss
    let reconLoss = 0;
    const dRecon: number[] = new Array(NUM_INPUT).fill(0);
    for (let i = 0; i < NUM_INPUT; i++) {
      const y = ex.pixels[i], r = recon[i];
      reconLoss += -(y * Math.log(r + 1e-8) + (1 - y) * Math.log(1 - r + 1e-8));
      dRecon[i] = (r - y) / N;
    }
    totalRecon += reconLoss / N;

    // KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    let kl = 0;
    const dMu: number[]     = new Array(LATENT_DIM).fill(0);
    const dLogVar: number[] = new Array(LATENT_DIM).fill(0);
    for (let i = 0; i < LATENT_DIM; i++) {
      kl += -0.5 * (1 + enc.logVar[i] - enc.mu[i] ** 2 - Math.exp(enc.logVar[i]));
      // Gradient of KL w.r.t. mu and logvar
      dMu[i]     = enc.mu[i] / N;
      dLogVar[i] = 0.5 * (Math.exp(enc.logVar[i]) - 1) / N;
    }
    totalKL += kl / N;

    // ── Backprop through decoder ──
    // dL/d(decOut pre-sigmoid) via BCE+sigmoid combo = recon - target
    const dDecOutZ: number[] = dRecon.slice(); // already = (r-y)/N
    // accumulate dDecOut weights
    for (let i = 0; i < vae.decOut.outSize; i++) {
      vae.decOut.db[i] += dDecOutZ[i];
      for (let j = 0; j < vae.decOut.inSize; j++)
        vae.decOut.dW[i][j] += dDecOutZ[i] * dec.dhA[j];
    }
    // backprop into decH activations
    const dDecHAct: number[] = new Array(vae.decH.outSize).fill(0);
    for (let j = 0; j < vae.decOut.inSize; j++)
      for (let i = 0; i < vae.decOut.outSize; i++)
        dDecHAct[j] += dDecOutZ[i] * vae.decOut.W[i][j];
    // through relu
    const dDecHZ: number[] = dDecHAct.map((d, j) => d * reluD(dec.dhZ[j]));
    for (let i = 0; i < vae.decH.outSize; i++) {
      vae.decH.db[i] += dDecHZ[i];
      for (let j = 0; j < vae.decH.inSize; j++)
        vae.decH.dW[i][j] += dDecHZ[i] * enc.z[j];
    }
    // dL/dz from decoder
    const dZ: number[] = new Array(LATENT_DIM).fill(0);
    for (let j = 0; j < vae.decH.inSize; j++)
      for (let i = 0; i < vae.decH.outSize; i++)
        dZ[j] += dDecHZ[i] * vae.decH.W[i][j];

    // ── Reparameterization: dL/dmu, dL/dlogvar (from decoder) ──
    const dMuTotal:     number[] = new Array(LATENT_DIM).fill(0);
    const dLogVarTotal: number[] = new Array(LATENT_DIM).fill(0);
    for (let i = 0; i < LATENT_DIM; i++) {
      const std = Math.exp(0.5 * enc.logVar[i]);
      dMuTotal[i]     = dZ[i] + beta * dMu[i];
      dLogVarTotal[i] = dZ[i] * enc.eps[i] * std * 0.5 + beta * dLogVar[i];
    }

    // ── Backprop through encoder ──
    for (let i = 0; i < vae.encMu.outSize; i++) {
      vae.encMu.db[i] += dMuTotal[i];
      for (let j = 0; j < vae.encMu.inSize; j++)
        vae.encMu.dW[i][j] += dMuTotal[i] * enc.hA[j];
    }
    for (let i = 0; i < vae.encLogVar.outSize; i++) {
      vae.encLogVar.db[i] += dLogVarTotal[i];
      for (let j = 0; j < vae.encLogVar.inSize; j++)
        vae.encLogVar.dW[i][j] += dLogVarTotal[i] * enc.hA[j];
    }
    // combined gradient into encoder hidden
    const dEncHA: number[] = new Array(vae.encH.outSize).fill(0);
    for (let j = 0; j < vae.encMu.inSize; j++) {
      for (let i = 0; i < vae.encMu.outSize; i++)
        dEncHA[j] += dMuTotal[i] * vae.encMu.W[i][j];
      for (let i = 0; i < vae.encLogVar.outSize; i++)
        dEncHA[j] += dLogVarTotal[i] * vae.encLogVar.W[i][j];
    }
    const dEncHZ: number[] = dEncHA.map((d, j) => d * reluD(enc.hZ[j]));
    for (let i = 0; i < vae.encH.outSize; i++) {
      vae.encH.db[i] += dEncHZ[i];
      for (let j = 0; j < vae.encH.inSize; j++)
        vae.encH.dW[i][j] += dEncHZ[i] * ex.pixels[j];
    }
  });

  adamUpdate(vae.encH,      lr, vae.step);
  adamUpdate(vae.encMu,     lr, vae.step);
  adamUpdate(vae.encLogVar, lr, vae.step);
  adamUpdate(vae.decH,      lr, vae.step);
  adamUpdate(vae.decOut,    lr, vae.step);

  return { recon: totalRecon, kl: totalKL };
}

// ─── Global state ────────────────────────────────────────────────────────────

let dataset: Example[] = [];
let trainLoss: { recon: number; kl: number; elbo: number; step: number }[] = [];
let running = false;
let rafId   = 0;
let lr      = 0.003;
let beta    = 1.0;
let batchSize = 20;

// ─── Layout constants ────────────────────────────────────────────────────────

const CELL = 24;   // px per glyph pixel
const GRID_N = 12; // latent grid size
const GRID_RANGE = 3.0;

// ─── DOM refs ────────────────────────────────────────────────────────────────

let svgLatent:  d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
let svgGrid:    d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
let svgLoss:    d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
let canvOrig:   HTMLCanvasElement;
let canvRecon:  HTMLCanvasElement;
let canvSample: HTMLCanvasElement;

let latentW = 300, latentH = 300;
let lossW   = 500, lossH   = 160;

// ─── Rendering helpers ───────────────────────────────────────────────────────

const CLASS_COLORS = ['#f87171','#fb923c','#a3e635','#34d399','#60a5fa'];

function drawGlyph(canvas: HTMLCanvasElement, pixels: number[]): void {
  const ctx = canvas.getContext('2d')!;
  const sz  = canvas.width;
  const cs  = sz / IMG_SIZE;
  ctx.clearRect(0, 0, sz, sz);
  for (let i = 0; i < NUM_INPUT; i++) {
    const v = Math.round(pixels[i] * 255);
    ctx.fillStyle = `rgb(${v},${v},${v})`;
    ctx.fillRect((i % IMG_SIZE) * cs, Math.floor(i / IMG_SIZE) * cs, cs, cs);
  }
}

function drawLatentSpace(): void {
  if (!svgLatent) return;
  svgLatent.selectAll('*').remove();

  const pad = 30;
  const iw = latentW - pad * 2;
  const ih = latentH - pad * 2;

  const xScale = d3.scaleLinear().domain([-GRID_RANGE, GRID_RANGE]).range([0, iw]);
  const yScale = d3.scaleLinear().domain([-GRID_RANGE, GRID_RANGE]).range([ih, 0]);

  const g = svgLatent.append('g').attr('transform', `translate(${pad},${pad})`);

  // axes
  g.append('line').attr('x1', 0).attr('x2', iw).attr('y1', yScale(0)).attr('y2', yScale(0))
    .attr('stroke','#444').attr('stroke-width',1);
  g.append('line').attr('x1', xScale(0)).attr('x2', xScale(0)).attr('y1', 0).attr('y2', ih)
    .attr('stroke','#444').attr('stroke-width',1);

  // encode all points (use mu only)
  const points: { mu: number[]; std: number[]; label: number }[] = [];
  dataset.forEach(ex => {
    const { z: hZ, a: hA } = forwardLayer(vae.encH, ex.pixels);
    const { a: mu }     = forwardLayerLinear(vae.encMu, hA);
    const { a: logVar } = forwardLayerLinear(vae.encLogVar, hA);
    const std = logVar.map(v => Math.exp(0.5 * v));
    points.push({ mu, std, label: ex.label });
  });

  // gaussian spread ellipses (class mean std)
  const classMus: number[][]  = [];
  const classStds: number[][] = [];
  for (let c = 0; c < NUM_CLASSES; c++) {
    const pts = points.filter(p => p.label === c);
    const avgMu  = [0,1].map(d => pts.reduce((s,p) => s + p.mu[d],  0) / pts.length);
    const avgStd = [0,1].map(d => pts.reduce((s,p) => s + p.std[d], 0) / pts.length);
    classMus.push(avgMu);
    classStds.push(avgStd);
    g.append('ellipse')
      .attr('cx', xScale(avgMu[0])).attr('cy', yScale(avgMu[1]))
      .attr('rx', xScale(avgMu[0] + avgStd[0]) - xScale(avgMu[0]))
      .attr('ry', yScale(avgMu[1]) - yScale(avgMu[1] + avgStd[1]))
      .attr('fill', CLASS_COLORS[c]).attr('opacity', 0.08)
      .attr('stroke', CLASS_COLORS[c]).attr('stroke-width', 1).attr('stroke-dasharray','3,3');
  }

  // scatter points
  points.forEach(p => {
    g.append('circle')
      .attr('cx', xScale(p.mu[0])).attr('cy', yScale(p.mu[1]))
      .attr('r', 4)
      .attr('fill', CLASS_COLORS[p.label])
      .attr('opacity', 0.75);
  });

  // legend
  CLASS_NAMES.forEach((name, i) => {
    g.append('circle').attr('cx', iw - 70).attr('cy', 12 + i * 16).attr('r', 5).attr('fill', CLASS_COLORS[i]);
    g.append('text').attr('x', iw - 62).attr('y', 16 + i * 16)
      .attr('fill','#ccc').attr('font-size', 10).text(name);
  });

  // click/drag to decode
  const overlay = g.append('rect')
    .attr('x', 0).attr('y', 0).attr('width', iw).attr('height', ih)
    .attr('fill', 'transparent').attr('cursor', 'crosshair');

  function decodeAt(event: MouseEvent): void {
    const [mx, my] = d3.pointer(event, g.node()!);
    const zx = xScale.invert(mx), zy = yScale.invert(my);
    const { recon } = decode([zx, zy]);
    drawGlyph(canvSample, recon);
    (document.getElementById('probe-coords') as HTMLElement).textContent =
      `z = (${zx.toFixed(2)}, ${zy.toFixed(2)})`;
  }

  overlay.on('click', (e: MouseEvent) => decodeAt(e));
  overlay.call(d3.drag<SVGRectElement, unknown>()
    .on('drag', (event) => decodeAt(event.sourceEvent as MouseEvent)) as any);
}

function drawLatentGrid(): void {
  if (!svgGrid) return;
  svgGrid.selectAll('*').remove();
  const cs = Math.floor(latentW / GRID_N);
  svgGrid.attr('width', cs * GRID_N).attr('height', cs * GRID_N);

  for (let row = 0; row < GRID_N; row++) {
    for (let col = 0; col < GRID_N; col++) {
      // map grid position to latent coords (bottom=negative y)
      const zx = -GRID_RANGE + (col + 0.5) * (2 * GRID_RANGE / GRID_N);
      const zy =  GRID_RANGE - (row + 0.5) * (2 * GRID_RANGE / GRID_N);
      const { recon } = decode([zx, zy]);
      // draw each pixel as a rect
      const gx = col * cs, gy = row * cs;
      for (let pi = 0; pi < NUM_INPUT; pi++) {
        const v = Math.round(recon[pi] * 255);
        const px = pi % IMG_SIZE, py = Math.floor(pi / IMG_SIZE);
        const pcs = cs / IMG_SIZE;
        svgGrid.append('rect')
          .attr('x', gx + px * pcs).attr('y', gy + py * pcs)
          .attr('width', pcs).attr('height', pcs)
          .attr('fill', `rgb(${v},${v},${v})`);
      }
    }
  }

  // border lines
  for (let i = 0; i <= GRID_N; i++) {
    svgGrid.append('line').attr('x1', i * cs).attr('x2', i * cs).attr('y1', 0).attr('y2', cs * GRID_N)
      .attr('stroke','#333').attr('stroke-width', 0.5);
    svgGrid.append('line').attr('x1', 0).attr('x2', cs * GRID_N).attr('y1', i * cs).attr('y2', i * cs)
      .attr('stroke','#333').attr('stroke-width', 0.5);
  }
}

function drawLossCurve(): void {
  if (!svgLoss || trainLoss.length < 2) return;
  svgLoss.selectAll('*').remove();

  const pad = { top: 10, right: 20, bottom: 30, left: 55 };
  const iw = lossW - pad.left - pad.right;
  const ih = lossH - pad.top  - pad.bottom;

  const xExt = [0, trainLoss[trainLoss.length - 1].step] as [number,number];
  const allVals = trainLoss.map(d => d.recon).concat(trainLoss.map(d => d.kl));
  const yMax = Math.max(...allVals) * 1.1;
  const yExt = [0, yMax] as [number,number];

  const xS = d3.scaleLinear().domain(xExt).range([0, iw]);
  const yS = d3.scaleLinear().domain(yExt).range([ih, 0]);

  const g = svgLoss.append('g').attr('transform', `translate(${pad.left},${pad.top})`);

  g.append('g').attr('transform', `translate(0,${ih})`)
    .call(d3.axisBottom(xS).ticks(5).tickFormat(d3.format('d')))
    .selectAll('text,line,path').attr('stroke','#666').attr('fill','#888');
  g.append('g').call(d3.axisLeft(yS).ticks(4).tickFormat(d3.format('.1f')))
    .selectAll('text,line,path').attr('stroke','#666').attr('fill','#888');

  const lineRecon = d3.line<{step:number;recon:number}>().x(d => xS(d.step)).y(d => yS(d.recon));
  const lineKL    = d3.line<{step:number;kl:number}>().x(d => xS(d.step)).y(d => yS(d.kl));

  g.append('path').datum(trainLoss).attr('d', lineRecon as any)
    .attr('fill','none').attr('stroke','#f87171').attr('stroke-width',1.5);
  g.append('path').datum(trainLoss).attr('d', lineKL as any)
    .attr('fill','none').attr('stroke','#60a5fa').attr('stroke-width',1.5);

  // legend
  g.append('circle').attr('cx', iw - 80).attr('cy', 8).attr('r', 4).attr('fill','#f87171');
  g.append('text').attr('x', iw - 72).attr('y', 12).attr('fill','#ccc').attr('font-size', 10).text('Recon');
  g.append('circle').attr('cx', iw - 80).attr('cy', 24).attr('r', 4).attr('fill','#60a5fa');
  g.append('text').attr('x', iw - 72).attr('y', 28).attr('fill','#ccc').attr('font-size', 10).text('KL');
}

function showRecon(): void {
  // pick one example per class and show orig + recon
  const ctxO = canvOrig.getContext('2d')!;
  const ctxR = canvRecon.getContext('2d')!;
  const n = NUM_CLASSES;
  const sz = CELL * IMG_SIZE;
  canvOrig.width  = n * sz; canvOrig.height = sz;
  canvRecon.width = n * sz; canvRecon.height = sz;
  ctxO.clearRect(0, 0, canvOrig.width, canvOrig.height);
  ctxR.clearRect(0, 0, canvRecon.width, canvRecon.height);

  for (let c = 0; c < n; c++) {
    const ex = dataset.find(e => e.label === c)!;
    const { recon } = decode(encode(ex.pixels).z);
    const cs2 = sz / IMG_SIZE;
    for (let i = 0; i < NUM_INPUT; i++) {
      const px = i % IMG_SIZE, py = Math.floor(i / IMG_SIZE);
      const vo = Math.round(ex.pixels[i] * 255);
      const vr = Math.round(recon[i] * 255);
      ctxO.fillStyle = `rgb(${vo},${vo},${vo})`;
      ctxO.fillRect(c * sz + px * cs2, py * cs2, cs2, cs2);
      ctxR.fillStyle = `rgb(${vr},${vr},${vr})`;
      ctxR.fillRect(c * sz + px * cs2, py * cs2, cs2, cs2);
    }
  }
}

function samplePrior(): void {
  const z = [randNorm(), randNorm()];
  const { recon } = decode(z);
  drawGlyph(canvSample, recon);
  (document.getElementById('probe-coords') as HTMLElement).textContent =
    `z = (${z[0].toFixed(2)}, ${z[1].toFixed(2)})`;
}

function updateStats(recon: number, kl: number): void {
  const elbo = recon + beta * kl;
  (document.getElementById('stat-elbo')  as HTMLElement).textContent = elbo.toFixed(3);
  (document.getElementById('stat-recon') as HTMLElement).textContent = recon.toFixed(3);
  (document.getElementById('stat-kl')    as HTMLElement).textContent = kl.toFixed(3);
  (document.getElementById('stat-step')  as HTMLElement).textContent = String(vae.step);
}

function doStep(): void {
  // sample a mini-batch
  const batch: Example[] = [];
  for (let i = 0; i < batchSize; i++)
    batch.push(dataset[Math.floor(lcg() * dataset.length)]);
  const { recon, kl } = trainStep(batch, lr, beta);
  const elbo = recon + beta * kl;
  trainLoss.push({ recon, kl, elbo, step: vae.step });
  if (trainLoss.length > 400) trainLoss.shift();
  updateStats(recon, kl);
}

function renderAll(): void {
  showRecon();
  drawLatentSpace();
  drawLatentGrid();
  drawLossCurve();
}

let frameCount = 0;
function loop(): void {
  if (!running) return;
  doStep();
  frameCount++;
  if (frameCount % 5 === 0) renderAll();
  rafId = requestAnimationFrame(loop);
}

// ─── UI wiring ───────────────────────────────────────────────────────────────

function init(): void {
  dataset = buildDataset();

  canvOrig   = document.getElementById('canv-orig')   as HTMLCanvasElement;
  canvRecon  = document.getElementById('canv-recon')  as HTMLCanvasElement;
  canvSample = document.getElementById('canv-sample') as HTMLCanvasElement;

  const latentEl = document.getElementById('svg-latent') as unknown as SVGSVGElement;
  const gridEl   = document.getElementById('svg-grid')   as unknown as SVGSVGElement;
  const lossEl   = document.getElementById('svg-loss')   as unknown as SVGSVGElement;

  svgLatent = d3.select<SVGSVGElement, unknown>(latentEl);
  svgGrid   = d3.select<SVGSVGElement, unknown>(gridEl);
  svgLoss   = d3.select<SVGSVGElement, unknown>(lossEl);

  latentW = latentEl.clientWidth  || 300;
  latentH = latentEl.clientHeight || 300;
  lossW   = lossEl.clientWidth    || 500;
  lossH   = lossEl.clientHeight   || 160;

  svgLatent.attr('width', latentW).attr('height', latentH);
  svgLoss.attr('width', lossW).attr('height', lossH);

  hiddenSize = parseInt((document.getElementById('ctrl-hidden') as HTMLInputElement).value, 10);
  vae = buildVAE(hiddenSize);
  trainLoss = [];

  renderAll();

  // controls
  (document.getElementById('btn-play')  as HTMLButtonElement).addEventListener('click', () => {
    running = true;
    frameCount = 0;
    loop();
  });
  (document.getElementById('btn-pause') as HTMLButtonElement).addEventListener('click', () => {
    running = false;
    cancelAnimationFrame(rafId);
  });
  (document.getElementById('btn-step')  as HTMLButtonElement).addEventListener('click', () => {
    running = false;
    cancelAnimationFrame(rafId);
    doStep();
    renderAll();
  });
  (document.getElementById('btn-reset') as HTMLButtonElement).addEventListener('click', () => {
    running = false;
    cancelAnimationFrame(rafId);
    hiddenSize = parseInt((document.getElementById('ctrl-hidden') as HTMLInputElement).value, 10);
    vae = buildVAE(hiddenSize);
    trainLoss = [];
    renderAll();
  });
  (document.getElementById('btn-sample') as HTMLButtonElement).addEventListener('click', samplePrior);

  function readSliders(): void {
    lr        = parseFloat((document.getElementById('ctrl-lr')    as HTMLInputElement).value);
    beta      = parseFloat((document.getElementById('ctrl-beta')  as HTMLInputElement).value);
    batchSize = parseInt  ((document.getElementById('ctrl-batch') as HTMLInputElement).value, 10);
    (document.getElementById('lbl-lr')    as HTMLElement).textContent = lr.toFixed(4);
    (document.getElementById('lbl-beta')  as HTMLElement).textContent = beta.toFixed(2);
    (document.getElementById('lbl-batch') as HTMLElement).textContent = String(batchSize);
    (document.getElementById('lbl-hidden') as HTMLElement).textContent =
      (document.getElementById('ctrl-hidden') as HTMLInputElement).value;
  }
  ['ctrl-lr','ctrl-beta','ctrl-batch','ctrl-hidden'].forEach(id =>
    (document.getElementById(id) as HTMLInputElement).addEventListener('input', readSliders)
  );
  readSliders();
}

window.addEventListener('DOMContentLoaded', init);
