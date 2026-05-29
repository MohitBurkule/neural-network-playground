// cnn.ts — CNN visualization & learning lab
// Implements a small CNN from scratch: Conv->ReLU->Pool->Conv->ReLU->Pool->FC->Softmax
// Visualizes filters, feature maps, and training progress with canvas/d3.

import * as d3 from 'd3';
import {
  IMAGE_SIZE, NUM_CLASSES, CLASS_NAMES,
  Example, getTrainData, getTestData, getAllData
} from './cnn_data';

// ─── Tensor helpers ──────────────────────────────────────────────────────────

// A 3D tensor: [channels][rows][cols]
type Tensor3 = number[][][];

function zeros3(c: number, h: number, w: number): Tensor3 {
  const t: Tensor3 = [];
  for (let i = 0; i < c; i++) {
    const plane: number[][] = [];
    for (let r = 0; r < h; r++) {
      plane.push(new Array(w).fill(0));
    }
    t.push(plane);
  }
  return t;
}

function clone3(t: Tensor3): Tensor3 {
  return t.map(c => c.map(r => r.slice()));
}

// Random normal (Box-Muller)
let _seed = 12345;
function randNorm(): number {
  _seed = (_seed * 1664525 + 1013904223) >>> 0;
  const u1 = (_seed / 0xffffffff) + 1e-10;
  _seed = (_seed * 1664525 + 1013904223) >>> 0;
  const u2 = (_seed / 0xffffffff) + 1e-10;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function randSmall(): number {
  return randNorm() * 0.1;
}

// ─── Conv layer ───────────────────────────────────────────────────────────────

interface ConvLayer {
  // kernels: [outC][inC][kH][kW]
  kernels: number[][][][];
  biases: number[];  // [outC]
  outC: number;
  inC: number;
  kSize: number;
  // gradients (accumulated)
  dKernels: number[][][][];
  dBiases: number[];
}

function makeConvLayer(inC: number, outC: number, kSize: number): ConvLayer {
  const kernels: number[][][][] = [];
  const dKernels: number[][][][] = [];
  for (let oc = 0; oc < outC; oc++) {
    const kSet: number[][][] = [];
    const dkSet: number[][][] = [];
    for (let ic = 0; ic < inC; ic++) {
      const k: number[][] = [];
      const dk: number[][] = [];
      for (let r = 0; r < kSize; r++) {
        const row: number[] = [];
        const drow: number[] = [];
        for (let c = 0; c < kSize; c++) {
          row.push(randSmall());
          drow.push(0);
        }
        k.push(row);
        dk.push(drow);
      }
      kSet.push(k);
      dkSet.push(dk);
    }
    kernels.push(kSet);
    dKernels.push(dkSet);
  }
  return {
    kernels,
    biases: new Array(outC).fill(0),
    outC,
    inC,
    kSize,
    dKernels,
    dBiases: new Array(outC).fill(0)
  };
}

// Forward conv (same padding)
function convForward(input: Tensor3, layer: ConvLayer): Tensor3 {
  const inC = input.length;
  const H = input[0].length;
  const W = input[0][0].length;
  const k = layer.kSize;
  const pad = Math.floor(k / 2);
  const output = zeros3(layer.outC, H, W);

  for (let oc = 0; oc < layer.outC; oc++) {
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        let sum = layer.biases[oc];
        for (let ic = 0; ic < inC; ic++) {
          for (let kr = 0; kr < k; kr++) {
            for (let kc = 0; kc < k; kc++) {
              const ir = r + kr - pad;
              const ic2 = c + kc - pad;
              if (ir >= 0 && ir < H && ic2 >= 0 && ic2 < W) {
                sum += input[ic][ir][ic2] * layer.kernels[oc][ic][kr][kc];
              }
            }
          }
        }
        output[oc][r][c] = sum;
      }
    }
  }
  return output;
}

// ReLU
function relu3(t: Tensor3): Tensor3 {
  return t.map(c => c.map(r => r.map(v => Math.max(0, v))));
}

// Max pool 2x2, stride 2
function maxPool2x2(t: Tensor3): { out: Tensor3; mask: boolean[][][][] } {
  const C = t.length;
  const H = t[0].length;
  const W = t[0][0].length;
  const oH = Math.floor(H / 2);
  const oW = Math.floor(W / 2);
  const out = zeros3(C, oH, oW);
  // mask[c][r][c2][0..3] = which of the 4 positions was max
  const mask: boolean[][][][] = [];
  for (let c = 0; c < C; c++) {
    const cm: boolean[][][] = [];
    for (let r = 0; r < oH; r++) {
      const rm: boolean[][] = [];
      for (let cc = 0; cc < oW; cc++) {
        const r0 = r * 2, c0 = cc * 2;
        const vals = [
          t[c][r0][c0],
          (c0 + 1 < W) ? t[c][r0][c0 + 1] : -Infinity,
          (r0 + 1 < H) ? t[c][r0 + 1][c0] : -Infinity,
          (r0 + 1 < H && c0 + 1 < W) ? t[c][r0 + 1][c0 + 1] : -Infinity
        ];
        const mx = Math.max(...vals);
        out[c][r][cc] = mx;
        rm.push(vals.map(v => v === mx));
      }
      cm.push(rm);
    }
    mask.push(cm);
  }
  return { out, mask };
}

// ─── FC layer ────────────────────────────────────────────────────────────────

interface FCLayer {
  W: number[][];   // [outN][inN]
  b: number[];     // [outN]
  dW: number[][];
  db: number[];
  outN: number;
  inN: number;
}

function makeFCLayer(inN: number, outN: number): FCLayer {
  const W: number[][] = [];
  const dW: number[][] = [];
  for (let o = 0; o < outN; o++) {
    const row: number[] = [];
    const drow: number[] = [];
    for (let i = 0; i < inN; i++) {
      row.push(randNorm() * Math.sqrt(2 / inN));
      drow.push(0);
    }
    W.push(row);
    dW.push(drow);
  }
  return { W, b: new Array(outN).fill(0), dW, db: new Array(outN).fill(0), outN, inN };
}

function fcForward(x: number[], layer: FCLayer): number[] {
  return layer.W.map((row, o) =>
    row.reduce((s, w, i) => s + w * x[i], 0) + layer.b[o]
  );
}

function softmax(logits: number[]): number[] {
  const mx = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - mx));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

function flatten3(t: Tensor3): number[] {
  const out: number[] = [];
  for (const plane of t) {
    for (const row of plane) {
      for (const v of row) {
        out.push(v);
      }
    }
  }
  return out;
}

// ─── CNN model ───────────────────────────────────────────────────────────────

interface ForwardCache {
  input: Tensor3;
  conv1Out: Tensor3;
  relu1Out: Tensor3;
  pool1Out: Tensor3;
  pool1Mask: boolean[][][][];
  conv2Out: Tensor3;
  relu2Out: Tensor3;
  pool2Out: Tensor3;
  pool2Mask: boolean[][][][];
  flat: number[];
  logits: number[];
  probs: number[];
}

interface CNN {
  conv1: ConvLayer;
  conv2: ConvLayer;
  fc: FCLayer;
}

// After two 2x2 pool on 8x8 input: 8->4->2
const POOL1_SIZE = 4; // 8/2
const POOL2_SIZE = 2; // 4/2
const CONV1_OUT_C = 4;
const CONV2_OUT_C = 8;
const FC_IN = CONV2_OUT_C * POOL2_SIZE * POOL2_SIZE; // 8*2*2 = 32

function makeCNN(): CNN {
  return {
    conv1: makeConvLayer(1, CONV1_OUT_C, 3),
    conv2: makeConvLayer(CONV1_OUT_C, CONV2_OUT_C, 3),
    fc: makeFCLayer(FC_IN, NUM_CLASSES)
  };
}

function forward(cnn: CNN, pixels: number[]): ForwardCache {
  // Input: [1][8][8]
  const input: Tensor3 = [[]];
  for (let r = 0; r < IMAGE_SIZE; r++) {
    const row: number[] = [];
    for (let c = 0; c < IMAGE_SIZE; c++) {
      row.push(pixels[r * IMAGE_SIZE + c]);
    }
    input[0].push(row);
  }

  const conv1Out = convForward(input, cnn.conv1);
  const relu1Out = relu3(conv1Out);
  const { out: pool1Out, mask: pool1Mask } = maxPool2x2(relu1Out);

  const conv2Out = convForward(pool1Out, cnn.conv2);
  const relu2Out = relu3(conv2Out);
  const { out: pool2Out, mask: pool2Mask } = maxPool2x2(relu2Out);

  const flat = flatten3(pool2Out);
  const logits = fcForward(flat, cnn.fc);
  const probs = softmax(logits);

  return {
    input, conv1Out, relu1Out, pool1Out, pool1Mask,
    conv2Out, relu2Out, pool2Out, pool2Mask,
    flat, logits, probs
  };
}

// ─── Backprop ─────────────────────────────────────────────────────────────────

function zeroGrads(cnn: CNN) {
  for (const layer of [cnn.conv1, cnn.conv2]) {
    for (let oc = 0; oc < layer.outC; oc++) {
      layer.dBiases[oc] = 0;
      for (let ic = 0; ic < layer.inC; ic++) {
        for (let kr = 0; kr < layer.kSize; kr++) {
          for (let kc = 0; kc < layer.kSize; kc++) {
            layer.dKernels[oc][ic][kr][kc] = 0;
          }
        }
      }
    }
  }
  for (let o = 0; o < cnn.fc.outN; o++) {
    cnn.fc.db[o] = 0;
    for (let i = 0; i < cnn.fc.inN; i++) {
      cnn.fc.dW[o][i] = 0;
    }
  }
}

// dLoss/dLogits for cross-entropy + softmax = probs - one_hot(label)
function backward(cnn: CNN, cache: ForwardCache, label: number) {
  // dLogits
  const dLogits = cache.probs.slice();
  dLogits[label] -= 1;

  // FC backward
  const dFlat = new Array(cnn.fc.inN).fill(0);
  for (let o = 0; o < cnn.fc.outN; o++) {
    cnn.fc.db[o] += dLogits[o];
    for (let i = 0; i < cnn.fc.inN; i++) {
      cnn.fc.dW[o][i] += dLogits[o] * cache.flat[i];
      dFlat[i] += dLogits[o] * cnn.fc.W[o][i];
    }
  }

  // Unflatten dFlat -> dPool2Out [CONV2_OUT_C][POOL2_SIZE][POOL2_SIZE]
  let idx = 0;
  const dPool2Out = zeros3(CONV2_OUT_C, POOL2_SIZE, POOL2_SIZE);
  for (let c = 0; c < CONV2_OUT_C; c++) {
    for (let r = 0; r < POOL2_SIZE; r++) {
      for (let cc = 0; cc < POOL2_SIZE; cc++) {
        dPool2Out[c][r][cc] = dFlat[idx++];
      }
    }
  }

  // Pool2 backward -> dRelu2Out [CONV2_OUT_C][POOL1_SIZE][POOL1_SIZE]
  const dRelu2Out = zeros3(CONV2_OUT_C, POOL1_SIZE, POOL1_SIZE);
  for (let c = 0; c < CONV2_OUT_C; c++) {
    for (let r = 0; r < POOL2_SIZE; r++) {
      for (let cc = 0; cc < POOL2_SIZE; cc++) {
        const r0 = r * 2, c0 = cc * 2;
        const mask = cache.pool2Mask[c][r][cc];
        if (mask[0]) dRelu2Out[c][r0][c0] += dPool2Out[c][r][cc];
        if (mask[1] && c0 + 1 < POOL1_SIZE) dRelu2Out[c][r0][c0 + 1] += dPool2Out[c][r][cc];
        if (mask[2] && r0 + 1 < POOL1_SIZE) dRelu2Out[c][r0 + 1][c0] += dPool2Out[c][r][cc];
        if (mask[3] && r0 + 1 < POOL1_SIZE && c0 + 1 < POOL1_SIZE) dRelu2Out[c][r0 + 1][c0 + 1] += dPool2Out[c][r][cc];
      }
    }
  }

  // ReLU2 backward
  const dConv2Out = zeros3(CONV2_OUT_C, POOL1_SIZE, POOL1_SIZE);
  for (let c = 0; c < CONV2_OUT_C; c++) {
    for (let r = 0; r < POOL1_SIZE; r++) {
      for (let cc = 0; cc < POOL1_SIZE; cc++) {
        dConv2Out[c][r][cc] = cache.conv2Out[c][r][cc] > 0 ? dRelu2Out[c][r][cc] : 0;
      }
    }
  }

  // Conv2 backward
  const dPool1Out = zeros3(CONV1_OUT_C, POOL1_SIZE, POOL1_SIZE);
  convBackward(cache.pool1Out, cnn.conv2, dConv2Out, dPool1Out);

  // Pool1 backward
  const dRelu1Out = zeros3(CONV1_OUT_C, IMAGE_SIZE, IMAGE_SIZE);
  for (let c = 0; c < CONV1_OUT_C; c++) {
    for (let r = 0; r < POOL1_SIZE; r++) {
      for (let cc = 0; cc < POOL1_SIZE; cc++) {
        const r0 = r * 2, c0 = cc * 2;
        const mask = cache.pool1Mask[c][r][cc];
        if (mask[0]) dRelu1Out[c][r0][c0] += dPool1Out[c][r][cc];
        if (mask[1] && c0 + 1 < IMAGE_SIZE) dRelu1Out[c][r0][c0 + 1] += dPool1Out[c][r][cc];
        if (mask[2] && r0 + 1 < IMAGE_SIZE) dRelu1Out[c][r0 + 1][c0] += dPool1Out[c][r][cc];
        if (mask[3] && r0 + 1 < IMAGE_SIZE && c0 + 1 < IMAGE_SIZE) dRelu1Out[c][r0 + 1][c0 + 1] += dPool1Out[c][r][cc];
      }
    }
  }

  // ReLU1 backward
  const dConv1Out = zeros3(CONV1_OUT_C, IMAGE_SIZE, IMAGE_SIZE);
  for (let c = 0; c < CONV1_OUT_C; c++) {
    for (let r = 0; r < IMAGE_SIZE; r++) {
      for (let cc = 0; cc < IMAGE_SIZE; cc++) {
        dConv1Out[c][r][cc] = cache.conv1Out[c][r][cc] > 0 ? dRelu1Out[c][r][cc] : 0;
      }
    }
  }

  // Conv1 backward (dInput ignored — it's the data)
  const dInputIgnored = zeros3(1, IMAGE_SIZE, IMAGE_SIZE);
  convBackward(cache.input, cnn.conv1, dConv1Out, dInputIgnored);
}

function convBackward(
  input: Tensor3,
  layer: ConvLayer,
  dOut: Tensor3,
  dInput: Tensor3
) {
  const H = input[0].length;
  const W = input[0][0].length;
  const k = layer.kSize;
  const pad = Math.floor(k / 2);

  for (let oc = 0; oc < layer.outC; oc++) {
    for (let r = 0; r < dOut[0].length; r++) {
      for (let c = 0; c < dOut[0][0].length; c++) {
        const d = dOut[oc][r][c];
        layer.dBiases[oc] += d;
        for (let ic = 0; ic < layer.inC; ic++) {
          for (let kr = 0; kr < k; kr++) {
            for (let kc = 0; kc < k; kc++) {
              const ir = r + kr - pad;
              const ic2 = c + kc - pad;
              if (ir >= 0 && ir < H && ic2 >= 0 && ic2 < W) {
                layer.dKernels[oc][ic][kr][kc] += d * input[ic][ir][ic2];
                dInput[ic][ir][ic2] += d * layer.kernels[oc][ic][kr][kc];
              }
            }
          }
        }
      }
    }
  }
}

function sgdStep(cnn: CNN, lr: number) {
  for (const layer of [cnn.conv1, cnn.conv2]) {
    for (let oc = 0; oc < layer.outC; oc++) {
      layer.biases[oc] -= lr * layer.dBiases[oc];
      for (let ic = 0; ic < layer.inC; ic++) {
        for (let kr = 0; kr < layer.kSize; kr++) {
          for (let kc = 0; kc < layer.kSize; kc++) {
            layer.kernels[oc][ic][kr][kc] -= lr * layer.dKernels[oc][ic][kr][kc];
          }
        }
      }
    }
  }
  for (let o = 0; o < cnn.fc.outN; o++) {
    cnn.fc.b[o] -= lr * cnn.fc.db[o];
    for (let i = 0; i < cnn.fc.inN; i++) {
      cnn.fc.W[o][i] -= lr * cnn.fc.dW[o][i];
    }
  }
}

// ─── Training state ──────────────────────────────────────────────────────────

let cnn = makeCNN();
let trainData = getTrainData();
let testData = getTestData();
let trainIdx = 0;

let lr = 0.01;
let running = false;
let animFrame: number | null = null;
let stepCount = 0;

let selectedExampleIdx = 0;
let lossHistory: number[] = [];
let accHistory: number[] = [];

let currentCache: ForwardCache | null = null;

function computeAccuracy(data: Example[]): number {
  let correct = 0;
  for (const ex of data) {
    const cache = forward(cnn, ex.pixels);
    const pred = cache.probs.indexOf(Math.max(...cache.probs));
    if (pred === ex.label) correct++;
  }
  return correct / data.length;
}

function trainStep() {
  zeroGrads(cnn);
  const ex = trainData[trainIdx % trainData.length];
  trainIdx++;
  const cache = forward(cnn, ex.pixels);
  const loss = -Math.log(cache.probs[ex.label] + 1e-10);
  backward(cnn, cache, ex.label);
  sgdStep(cnn, lr);

  stepCount++;
  lossHistory.push(loss);
  if (lossHistory.length > 200) lossHistory.shift();

  if (stepCount % 10 === 0) {
    const acc = computeAccuracy(testData);
    accHistory.push(acc);
    if (accHistory.length > 200) accHistory.shift();
  }

  // Update current display cache
  const displayEx = getAllData()[selectedExampleIdx];
  currentCache = forward(cnn, displayEx.pixels);

  render();
}

// ─── Rendering ───────────────────────────────────────────────────────────────

function drawGrayscale(canvas: HTMLCanvasElement, data: number[][], scale: number = 20) {
  const H = data.length;
  const W = data[0].length;
  canvas.width = W * scale;
  canvas.height = H * scale;
  const ctx = canvas.getContext('2d')!;
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const v = Math.min(1, Math.max(0, data[r][c]));
      const g = Math.round(v * 255);
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect(c * scale, r * scale, scale, scale);
    }
  }
}

function drawKernel(canvas: HTMLCanvasElement, kernel: number[][], scale: number = 30) {
  const H = kernel.length;
  const W = kernel[0].length;
  canvas.width = W * scale;
  canvas.height = H * scale;
  const ctx = canvas.getContext('2d')!;

  // Find min/max for normalization
  let mn = Infinity, mx = -Infinity;
  for (const row of kernel) {
    for (const v of row) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  const range = mx - mn || 1;

  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const t = (kernel[r][c] - mn) / range; // 0..1
      // diverging: negative=blue, positive=red, zero=white
      let R, G, B;
      if (t < 0.5) {
        const s = t * 2;
        R = Math.round(s * 255);
        G = Math.round(s * 255);
        B = 255;
      } else {
        const s = (t - 0.5) * 2;
        R = 255;
        G = Math.round((1 - s) * 255);
        B = Math.round((1 - s) * 255);
      }
      ctx.fillStyle = `rgb(${R},${G},${B})`;
      ctx.fillRect(c * scale, r * scale, scale, scale);
    }
  }

  // Draw grid
  ctx.strokeStyle = 'rgba(0,0,0,0.2)';
  ctx.lineWidth = 0.5;
  for (let r = 0; r <= H; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * scale);
    ctx.lineTo(W * scale, r * scale);
    ctx.stroke();
  }
  for (let c = 0; c <= W; c++) {
    ctx.beginPath();
    ctx.moveTo(c * scale, 0);
    ctx.lineTo(c * scale, H * scale);
    ctx.stroke();
  }
}

function drawFeatureMap(canvas: HTMLCanvasElement, data: number[][], scale: number = 16) {
  const H = data.length;
  const W = data[0].length;
  canvas.width = W * scale;
  canvas.height = H * scale;
  const ctx = canvas.getContext('2d')!;

  let mn = Infinity, mx = -Infinity;
  for (const row of data) {
    for (const v of row) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  const range = mx - mn || 1;

  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const t = (data[r][c] - mn) / range;
      const g = Math.round(t * 255);
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect(c * scale, r * scale, scale, scale);
    }
  }
}

function renderProbBars(probs: number[], label: number) {
  const container = document.getElementById('prob-bars')!;
  container.innerHTML = '';

  for (let i = 0; i < NUM_CLASSES; i++) {
    const p = probs[i];
    const pct = (p * 100).toFixed(1);
    const isCorrect = i === label;

    const row = document.createElement('div');
    row.className = 'prob-row';

    const nameEl = document.createElement('span');
    nameEl.className = 'prob-label';
    nameEl.textContent = CLASS_NAMES[i];

    const barWrap = document.createElement('div');
    barWrap.className = 'prob-bar-wrap';

    const bar = document.createElement('div');
    bar.className = 'prob-bar' + (isCorrect ? ' correct' : '');
    bar.style.width = `${p * 100}%`;

    const pctEl = document.createElement('span');
    pctEl.className = 'prob-pct';
    pctEl.textContent = `${pct}%`;

    barWrap.appendChild(bar);
    row.appendChild(nameEl);
    row.appendChild(barWrap);
    row.appendChild(pctEl);
    container.appendChild(row);
  }
}

// Line chart using d3
function renderLineChart(
  svgId: string,
  data: number[],
  color: string,
  yLabel: string,
  yDomain?: [number, number]
) {
  const svg = d3.select(`#${svgId}`);
  svg.selectAll('*').remove();

  const W = 300, H = 100;
  const margin = { top: 10, right: 10, bottom: 20, left: 40 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;

  const g = svg
    .attr('width', W)
    .attr('height', H)
    .append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  const xScale = d3.scaleLinear().domain([0, Math.max(data.length - 1, 1)]).range([0, innerW]);
  const mn = yDomain ? yDomain[0] : Math.min(...data, 0);
  const mx = yDomain ? yDomain[1] : Math.max(...data, 0.01);
  const yScale = d3.scaleLinear().domain([mn, mx]).range([innerH, 0]);

  g.append('g')
    .attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(xScale).ticks(4))
    .selectAll('text').style('font-size', '10px');

  g.append('g')
    .call(d3.axisLeft(yScale).ticks(4))
    .selectAll('text').style('font-size', '10px');

  g.append('text')
    .attr('x', -margin.left)
    .attr('y', -4)
    .style('font-size', '10px')
    .text(yLabel);

  const line = d3.line<number>()
    .x((_, i) => xScale(i))
    .y(d => yScale(d));

  g.append('path')
    .datum(data)
    .attr('fill', 'none')
    .attr('stroke', color)
    .attr('stroke-width', 1.5)
    .attr('d', line);
}

function renderKernelRow(containerId: string, layer: ConvLayer) {
  const container = document.getElementById(containerId)!;
  container.innerHTML = '';
  for (let oc = 0; oc < layer.outC; oc++) {
    // sum over input channels
    const sumKernel: number[][] = [];
    for (let kr = 0; kr < layer.kSize; kr++) {
      const row: number[] = [];
      for (let kc = 0; kc < layer.kSize; kc++) {
        let s = 0;
        for (let ic = 0; ic < layer.inC; ic++) {
          s += layer.kernels[oc][ic][kr][kc];
        }
        row.push(s);
      }
      sumKernel.push(row);
    }
    const wrap = document.createElement('div');
    wrap.className = 'kernel-wrap';
    const lbl = document.createElement('div');
    lbl.className = 'kernel-label';
    lbl.textContent = `F${oc + 1}`;
    const canvas = document.createElement('canvas');
    drawKernel(canvas, sumKernel, 24);
    wrap.appendChild(lbl);
    wrap.appendChild(canvas);
    container.appendChild(wrap);
  }
}

function renderFeatureMaps(containerId: string, maps: Tensor3, scale: number) {
  const container = document.getElementById(containerId)!;
  container.innerHTML = '';
  for (let c = 0; c < maps.length; c++) {
    const wrap = document.createElement('div');
    wrap.className = 'fmap-wrap';
    const lbl = document.createElement('div');
    lbl.className = 'kernel-label';
    lbl.textContent = `Ch${c + 1}`;
    const canvas = document.createElement('canvas');
    drawFeatureMap(canvas, maps[c], scale);
    wrap.appendChild(lbl);
    wrap.appendChild(canvas);
    container.appendChild(wrap);
  }
}

function render() {
  if (!currentCache) return;
  const allData = getAllData();
  const ex = allData[selectedExampleIdx];

  // Input image
  const inputCanvas = document.getElementById('input-canvas') as HTMLCanvasElement;
  drawGrayscale(inputCanvas, currentCache.input[0], 32);

  // True label
  const lblEl = document.getElementById('true-label')!;
  lblEl.textContent = `True: ${CLASS_NAMES[ex.label]}`;

  // Conv1 kernels
  renderKernelRow('conv1-kernels', cnn.conv1);
  // Conv1 feature maps (after relu)
  renderFeatureMaps('conv1-fmaps', currentCache.relu1Out, 14);
  // Pool1 feature maps
  renderFeatureMaps('pool1-fmaps', currentCache.pool1Out, 18);

  // Conv2 kernels
  renderKernelRow('conv2-kernels', cnn.conv2);
  // Conv2 feature maps (after relu)
  renderFeatureMaps('conv2-fmaps', currentCache.relu2Out, 18);
  // Pool2 feature maps
  renderFeatureMaps('pool2-fmaps', currentCache.pool2Out, 24);

  // Probabilities
  renderProbBars(currentCache.probs, ex.label);

  // Charts
  if (lossHistory.length > 1) renderLineChart('loss-chart', lossHistory, '#e74c3c', 'Loss');
  if (accHistory.length > 1) renderLineChart('acc-chart', accHistory, '#27ae60', 'Acc', [0, 1]);

  // Step count
  const stepEl = document.getElementById('step-count')!;
  stepEl.textContent = `Step: ${stepCount}`;
}

// ─── UI wiring ───────────────────────────────────────────────────────────────

function runLoop() {
  if (!running) return;
  trainStep();
  animFrame = requestAnimationFrame(runLoop);
}

function resetCNN() {
  running = false;
  if (animFrame !== null) cancelAnimationFrame(animFrame);
  cnn = makeCNN();
  trainIdx = 0;
  stepCount = 0;
  lossHistory = [];
  accHistory = [];
  const displayEx = getAllData()[selectedExampleIdx];
  currentCache = forward(cnn, displayEx.pixels);
  render();
}

// Build the example selector
function buildExampleSelector() {
  const allData = getAllData();
  const sel = document.getElementById('example-select') as HTMLSelectElement;
  sel.innerHTML = '';
  allData.forEach((ex, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = `#${i + 1} — ${CLASS_NAMES[ex.label]}`;
    sel.appendChild(opt);
  });
  sel.value = String(selectedExampleIdx);
  sel.addEventListener('change', () => {
    selectedExampleIdx = parseInt(sel.value);
    const ex = allData[selectedExampleIdx];
    currentCache = forward(cnn, ex.pixels);
    render();
  });
}

// ─── Entry point ─────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  buildExampleSelector();
  resetCNN();

  document.getElementById('btn-play')!.addEventListener('click', () => {
    running = true;
    runLoop();
  });

  document.getElementById('btn-pause')!.addEventListener('click', () => {
    running = false;
    if (animFrame !== null) cancelAnimationFrame(animFrame);
  });

  document.getElementById('btn-reset')!.addEventListener('click', resetCNN);

  document.getElementById('btn-step')!.addEventListener('click', () => {
    running = false;
    if (animFrame !== null) cancelAnimationFrame(animFrame);
    trainStep();
  });

  const lrSlider = document.getElementById('lr-slider') as HTMLInputElement;
  const lrDisplay = document.getElementById('lr-display')!;
  lrSlider.addEventListener('input', () => {
    lr = parseFloat(lrSlider.value);
    lrDisplay.textContent = lr.toFixed(4);
  });
  lrDisplay.textContent = lr.toFixed(4);
});
