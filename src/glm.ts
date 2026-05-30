import * as d3 from 'd3';

// ─── Types ───────────────────────────────────────────────────────────────────

type ModelName = 'linear' | 'logistic' | 'gnb';
type DatasetName = 'noisy-line' | 'noisy-quad' | 'blobs' | 'moons' | 'circles' | 'xor';

interface Point {
  x: number;
  y: number;
  label: number; // 0 or 1 for classification; raw y for regression
}

// ─── Random helpers ──────────────────────────────────────────────────────────

function randn(): number {
  const u = Math.random() + 1e-12, v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ─── Dataset generators ──────────────────────────────────────────────────────

function genNoisyLine(n: number, noise: number): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() * 2 - 1) * 3;
    const y = 1.5 * x + 0.5 + randn() * noise * 3;
    pts.push({ x, y, label: y });
  }
  return pts;
}

function genNoisyQuad(n: number, noise: number): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() * 2 - 1) * 3;
    const y = 0.8 * x * x - 1.0 * x + 0.3 + randn() * noise * 3;
    pts.push({ x, y, label: y });
  }
  return pts;
}

function genBlobs(n: number, noise: number): Point[] {
  const centers: [number, number, number][] = [[-1.5, -1.5, 0], [1.5, 1.5, 1]];
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const c = centers[i % 2];
    pts.push({ x: c[0] + randn() * (0.6 + noise), y: c[1] + randn() * (0.6 + noise), label: c[2] });
  }
  return pts;
}

function genMoons(n: number, noise: number): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2;
    const angle = Math.PI * Math.random();
    const r = 2.0;
    const cx = label === 0 ? 0 : r * 0.5;
    const cy = label === 0 ? 0 : 0.3;
    const x = (label === 0 ? Math.cos(angle) : -Math.cos(angle)) * r * 0.5 + cx + randn() * noise * 0.5;
    const y = (label === 0 ? Math.sin(angle) : -Math.sin(angle)) * r * 0.3 + cy + randn() * noise * 0.5;
    pts.push({ x, y, label });
  }
  return pts;
}

function genCircles(n: number, noise: number): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2;
    const r = label === 0 ? 1.0 : 2.2;
    const angle = Math.random() * 2 * Math.PI;
    pts.push({
      x: Math.cos(angle) * r + randn() * noise * 0.4,
      y: Math.sin(angle) * r + randn() * noise * 0.4,
      label
    });
  }
  return pts;
}

function genXor(n: number, noise: number): Point[] {
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const qx = Math.random() > 0.5 ? 1 : -1;
    const qy = Math.random() > 0.5 ? 1 : -1;
    const label = (qx * qy > 0) ? 1 : 0;
    pts.push({ x: qx * (1.2 + Math.random() * 0.8) + randn() * noise * 0.4, y: qy * (1.2 + Math.random() * 0.8) + randn() * noise * 0.4, label });
  }
  return pts;
}

function generateDataset(name: DatasetName, n: number, noise: number): Point[] {
  switch (name) {
    case 'noisy-line': return genNoisyLine(n, noise);
    case 'noisy-quad': return genNoisyQuad(n, noise);
    case 'blobs': return genBlobs(n, noise);
    case 'moons': return genMoons(n, noise);
    case 'circles': return genCircles(n, noise);
    case 'xor': return genXor(n, noise);
  }
}

// ─── Feature expansion ───────────────────────────────────────────────────────

function polyFeatures(x: number, y: number, degree: number): number[] {
  // returns [1, x, y, x^2, xy, y^2, ...] up to given degree
  const feats: number[] = [1];
  for (let d = 1; d <= degree; d++) {
    for (let i = 0; i <= d; i++) {
      feats.push(Math.pow(x, d - i) * Math.pow(y, i));
    }
  }
  return feats;
}

// ─── Math helpers ────────────────────────────────────────────────────────────

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// ─── Models ──────────────────────────────────────────────────────────────────

// Linear regression state
interface LinRegState {
  weights: number[]; // [bias, w1] or more for polynomial (but we keep 1D x → y)
  loss: number;
  iter: number;
}

function initLinReg(): LinRegState {
  return { weights: [0, 0], loss: Infinity, iter: 0 };
}

function stepLinReg(state: LinRegState, data: Point[], lr: number): LinRegState {
  const w = state.weights.slice();
  const n = data.length;
  let loss = 0;
  const grad = [0, 0];
  for (let i = 0; i < n; i++) {
    const xi = data[i].x;
    const yi = data[i].y;
    const pred = w[0] + w[1] * xi;
    const err = pred - yi;
    loss += err * err;
    grad[0] += err;
    grad[1] += err * xi;
  }
  loss /= n;
  w[0] -= lr * grad[0] / n;
  w[1] -= lr * grad[1] / n;
  return { weights: w, loss, iter: state.iter + 1 };
}

// Logistic regression state
interface LogRegState {
  weights: number[];
  loss: number;
  acc: number;
  iter: number;
}

function initLogReg(nFeats: number): LogRegState {
  return { weights: new Array(nFeats).fill(0), loss: Infinity, acc: 0, iter: 0 };
}

function stepLogReg(state: LogRegState, data: Point[], lr: number, l2: number, degree: number): LogRegState {
  const w = state.weights.slice();
  const n = data.length;
  const grad = new Array(w.length).fill(0);
  let loss = 0;
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const feats = polyFeatures(data[i].x, data[i].y, degree);
    const z = dot(w, feats);
    const p = sigmoid(z);
    const yi = data[i].label;
    loss += -(yi * Math.log(p + 1e-12) + (1 - yi) * Math.log(1 - p + 1e-12));
    const err = p - yi;
    for (let j = 0; j < w.length; j++) {
      grad[j] += err * feats[j];
    }
    if ((p >= 0.5 ? 1 : 0) === yi) correct++;
  }
  loss /= n;
  // L2 regularization
  for (let j = 1; j < w.length; j++) {
    loss += 0.5 * l2 * w[j] * w[j];
    grad[j] += l2 * w[j] * n; // scale to match
  }
  for (let j = 0; j < w.length; j++) {
    w[j] -= lr * grad[j] / n;
  }
  return { weights: w, loss: loss / (1 + l2), acc: correct / n, iter: state.iter + 1 };
}

// Gaussian Naive Bayes
interface GNBClass {
  mean: [number, number];
  variance: [number, number];
  prior: number;
}

interface GNBState {
  classes: GNBClass[];
  acc: number;
}

function trainGNB(data: Point[]): GNBState {
  const n = data.length;
  const classes: GNBClass[] = [];
  for (let c = 0; c <= 1; c++) {
    const pts = data.filter(p => p.label === c);
    const k = pts.length;
    if (k === 0) {
      classes.push({ mean: [0, 0], variance: [1, 1], prior: 0 });
      continue;
    }
    let mx = 0, my = 0;
    pts.forEach(p => { mx += p.x; my += p.y; });
    mx /= k; my /= k;
    let vx = 0, vy = 0;
    pts.forEach(p => { vx += (p.x - mx) ** 2; vy += (p.y - my) ** 2; });
    vx = vx / k + 1e-6; vy = vy / k + 1e-6;
    classes.push({ mean: [mx, my], variance: [vx, vy], prior: k / n });
  }
  // compute accuracy
  let correct = 0;
  data.forEach(p => {
    let bestC = 0, bestScore = -Infinity;
    classes.forEach((cls, c) => {
      if (cls.prior === 0) return;
      const lp = Math.log(cls.prior)
        - 0.5 * Math.log(cls.variance[0]) - 0.5 * (p.x - cls.mean[0]) ** 2 / cls.variance[0]
        - 0.5 * Math.log(cls.variance[1]) - 0.5 * (p.y - cls.mean[1]) ** 2 / cls.variance[1];
      if (lp > bestScore) { bestScore = lp; bestC = c; }
    });
    if (bestC === p.label) correct++;
  });
  return { classes, acc: correct / n };
}

function gnbPredict(state: GNBState, x: number, y: number): number {
  // returns P(class=1)
  const scores: number[] = [];
  state.classes.forEach(cls => {
    if (cls.prior === 0) { scores.push(-Infinity); return; }
    scores.push(
      Math.log(cls.prior)
      - 0.5 * Math.log(cls.variance[0]) - 0.5 * (x - cls.mean[0]) ** 2 / cls.variance[0]
      - 0.5 * Math.log(cls.variance[1]) - 0.5 * (y - cls.mean[1]) ** 2 / cls.variance[1]
    );
  });
  const maxS = Math.max(scores[0], scores[1]);
  const e0 = Math.exp(scores[0] - maxS);
  const e1 = Math.exp(scores[1] - maxS);
  return e1 / (e0 + e1);
}

// ─── App State ───────────────────────────────────────────────────────────────

let model: ModelName = 'linear';
let dataset: DatasetName = 'noisy-line';
let nPoints = 120;
let noiseLevel = 0.4;
let learningRate = 0.05;
let l2Reg = 0.0;
let polyDegree = 1;
let running = false;
let animFrame: number | null = null;
let data: Point[] = [];
let linState: LinRegState = initLinReg();
let logState: LogRegState = initLogReg(polyFeatures(0, 0, 1).length);
let gnbState: GNBState = { classes: [], acc: 0 };
let lossHistory: number[] = [];
let accHistory: number[] = [];

// ─── Canvas / SVG setup ──────────────────────────────────────────────────────

const CANVAS_W = 480;
const CANVAS_H = 480;
const MARGIN = 30;
const CHART_W = CANVAS_W - MARGIN * 2;
const CHART_H = CANVAS_H - MARGIN * 2;

let mainCanvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;
let lossChartSvg: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;

// Scale helpers (data → canvas)
let xScale: d3.ScaleLinear<number, number>;
let yScale: d3.ScaleLinear<number, number>;

function computeScales(pts: Point[]): void {
  if (pts.length === 0) {
    xScale = d3.scaleLinear().domain([-3, 3]).range([MARGIN, CANVAS_W - MARGIN]);
    yScale = d3.scaleLinear().domain([-3, 3]).range([CANVAS_H - MARGIN, MARGIN]);
    return;
  }
  const xExt = d3.extent(pts, d => d.x) as [number, number];
  const yExt = d3.extent(pts, d => d.y) as [number, number];
  const xPad = Math.max((xExt[1] - xExt[0]) * 0.15, 0.5);
  const yPad = Math.max((yExt[1] - yExt[0]) * 0.15, 0.5);
  xScale = d3.scaleLinear().domain([xExt[0] - xPad, xExt[1] + xPad]).range([MARGIN, CANVAS_W - MARGIN]);
  yScale = d3.scaleLinear().domain([yExt[0] - yPad, yExt[1] + yPad]).range([CANVAS_H - MARGIN, MARGIN]);
}

// ─── Rendering ───────────────────────────────────────────────────────────────

function drawGrid(): void {
  ctx.strokeStyle = '#2a2a5a';
  ctx.lineWidth = 1;
  const xDom = xScale.domain();
  const yDom = yScale.domain();
  const xTicks = xScale.ticks(6);
  const yTicks = yScale.ticks(6);
  ctx.beginPath();
  xTicks.forEach(t => {
    const cx = xScale(t);
    ctx.moveTo(cx, MARGIN);
    ctx.lineTo(cx, CANVAS_H - MARGIN);
  });
  yTicks.forEach(t => {
    const cy = yScale(t);
    ctx.moveTo(MARGIN, cy);
    ctx.lineTo(CANVAS_W - MARGIN, cy);
  });
  ctx.stroke();

  // Axes
  ctx.strokeStyle = '#555580';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const x0 = Math.max(MARGIN, Math.min(CANVAS_W - MARGIN, xScale(0)));
  const y0 = Math.max(MARGIN, Math.min(CANVAS_H - MARGIN, yScale(0)));
  ctx.moveTo(x0, MARGIN); ctx.lineTo(x0, CANVAS_H - MARGIN);
  ctx.moveTo(MARGIN, y0); ctx.lineTo(CANVAS_W - MARGIN, y0);
  ctx.stroke();
}

function drawDecisionBackground(predictFn: (x: number, y: number) => number, isClassification: boolean): void {
  const step = 4;
  const xDom = xScale.domain();
  const yDom = yScale.domain();
  const xMin = xDom[0], xMax = xDom[1];
  const yMin = yDom[0], yMax = yDom[1];
  const nx = Math.ceil(CHART_W / step);
  const ny = Math.ceil(CHART_H / step);
  const imageData = ctx.createImageData(nx * step, ny * step);
  const buf = imageData.data;

  for (let py = 0; py < ny; py++) {
    for (let px = 0; px < nx; px++) {
      const wx = xMin + (px / nx) * (xMax - xMin);
      const wy = yMax - (py / ny) * (yMax - yMin); // canvas y flipped
      const val = predictFn(wx, wy); // 0..1
      let r = 0, g = 0, b = 0, a = 0;
      if (isClassification) {
        // blue = class 0, orange = class 1
        if (val < 0.5) {
          const t = val / 0.5;
          r = Math.round(30 + t * 20);
          g = Math.round(60 + t * 20);
          b = Math.round(180 - t * 30);
          a = Math.round(80 + (0.5 - val) * 160);
        } else {
          const t = (val - 0.5) / 0.5;
          r = Math.round(200 + t * 55);
          g = Math.round(100 - t * 50);
          b = Math.round(30);
          a = Math.round(80 + (val - 0.5) * 160);
        }
      } else {
        // heat map for regression residual visual
        const v = Math.max(0, Math.min(1, val));
        r = Math.round(v * 200);
        g = Math.round(60);
        b = Math.round((1 - v) * 200);
        a = 60;
      }
      // fill step x step block
      for (let dy = 0; dy < step; dy++) {
        for (let dx = 0; dx < step; dx++) {
          const idx = ((py * step + dy) * nx * step + (px * step + dx)) * 4;
          buf[idx] = r; buf[idx + 1] = g; buf[idx + 2] = b; buf[idx + 3] = a;
        }
      }
    }
  }
  ctx.putImageData(imageData, MARGIN, MARGIN);
}

function colorForLabel(label: number, isClassification: boolean): string {
  if (!isClassification) {
    // color by y value via viridis-like
    return '#a78bfa';
  }
  return label === 0 ? '#60a5fa' : '#fb923c';
}

const colorScale0 = d3.scaleSequential(d3.interpolateViridis).domain([-3, 3]);

function drawPoints(pts: Point[], isClassification: boolean, yMin: number, yMax: number): void {
  pts.forEach(p => {
    const cx = xScale(p.x);
    const cy = yScale(isClassification ? p.y : p.label);
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    if (isClassification) {
      ctx.fillStyle = colorForLabel(p.label, true);
    } else {
      ctx.fillStyle = colorScale0((p.label - yMin) / (yMax - yMin) * 6 - 3);
    }
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
  });
}

function drawLinRegLine(w: number[]): void {
  // w = [bias, slope]
  const xDom = xScale.domain();
  const x0 = xDom[0], x1 = xDom[1];
  const y0 = w[0] + w[1] * x0;
  const y1 = w[0] + w[1] * x1;
  ctx.beginPath();
  ctx.moveTo(xScale(x0), yScale(y0));
  ctx.lineTo(xScale(x1), yScale(y1));
  ctx.strokeStyle = '#facc15';
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Residuals
  data.forEach(p => {
    const pred = w[0] + w[1] * p.x;
    ctx.beginPath();
    ctx.moveTo(xScale(p.x), yScale(p.label));
    ctx.lineTo(xScale(p.x), yScale(pred));
    ctx.strokeStyle = 'rgba(250,204,21,0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

function drawLogisticBoundary(weights: number[], degree: number): void {
  // Draw contour line at p=0.5 (z=0) by sampling
  const xDom = xScale.domain();
  const yDom = yScale.domain();
  const steps = 200;
  ctx.beginPath();
  let started = false;
  for (let px = 0; px <= steps; px++) {
    const wx = xDom[0] + (px / steps) * (xDom[1] - xDom[0]);
    for (let py = 0; py <= steps; py++) {
      const wy = yDom[0] + (py / steps) * (yDom[1] - yDom[0]);
      const z = dot(weights, polyFeatures(wx, wy, degree));
      // look for sign change in neighbours
      if (Math.abs(z) < (xDom[1] - xDom[0]) / steps * 2) {
        const cx2 = xScale(wx), cy2 = yScale(wy);
        if (!started) { ctx.moveTo(cx2, cy2); started = true; }
        else ctx.lineTo(cx2, cy2);
      }
    }
  }
  ctx.strokeStyle = '#facc15';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawGNBEllipses(state: GNBState): void {
  const colors = ['#60a5fa', '#fb923c'];
  state.classes.forEach((cls, c) => {
    if (cls.prior === 0) return;
    const sx = Math.sqrt(cls.variance[0]);
    const sy = Math.sqrt(cls.variance[1]);
    // draw 1σ, 2σ ellipses in canvas space
    [1, 2].forEach(sigma => {
      ctx.beginPath();
      for (let t = 0; t <= 361; t++) {
        const angle = (t / 360) * 2 * Math.PI;
        const wx = cls.mean[0] + Math.cos(angle) * sx * sigma;
        const wy = cls.mean[1] + Math.sin(angle) * sy * sigma;
        const cx2 = xScale(wx), cy2 = yScale(wy);
        if (t === 0) ctx.moveTo(cx2, cy2);
        else ctx.lineTo(cx2, cy2);
      }
      ctx.strokeStyle = colors[c];
      ctx.lineWidth = sigma === 1 ? 2.5 : 1.2;
      ctx.globalAlpha = sigma === 1 ? 0.9 : 0.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
    // mean cross
    const mx = xScale(cls.mean[0]);
    const my = yScale(cls.mean[1]);
    ctx.strokeStyle = colors[c];
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(mx - 8, my); ctx.lineTo(mx + 8, my);
    ctx.moveTo(mx, my - 8); ctx.lineTo(mx, my + 8);
    ctx.stroke();
  });
}

function render(): void {
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

  // Background
  ctx.fillStyle = '#0f0f23';
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  const isClassification = model === 'logistic' || model === 'gnb';

  // Decision background
  if (isClassification && data.length > 0) {
    if (model === 'logistic') {
      const w = logState.weights;
      const deg = polyDegree;
      drawDecisionBackground((x, y) => sigmoid(dot(w, polyFeatures(x, y, deg))), true);
    } else if (model === 'gnb' && gnbState.classes.length > 0) {
      drawDecisionBackground((x, y) => gnbPredict(gnbState, x, y), true);
    }
  }

  drawGrid();

  if (data.length === 0) return;

  const yExt = d3.extent(data, d => d.label) as [number, number];

  if (model === 'linear') {
    drawPoints(data, false, yExt[0], yExt[1]);
    drawLinRegLine(linState.weights);
  } else if (model === 'logistic') {
    if (logState.weights.some(w => isFinite(w))) {
      drawLogisticBoundary(logState.weights, polyDegree);
    }
    drawPoints(data, true, yExt[0], yExt[1]);
  } else if (model === 'gnb') {
    if (gnbState.classes.length > 0) drawGNBEllipses(gnbState);
    drawPoints(data, true, yExt[0], yExt[1]);
  }

  // Border
  ctx.strokeStyle = '#2a2a5a';
  ctx.lineWidth = 1;
  ctx.strokeRect(MARGIN, MARGIN, CHART_W, CHART_H);
}

// ─── Loss Chart ───────────────────────────────────────────────────────────────

const LOSS_W = 240, LOSS_H = 100;
const LOSS_MARGIN = { t: 10, r: 10, b: 25, l: 40 };

function renderLossChart(): void {
  const hist = model === 'linear' ? lossHistory : accHistory;
  const label = model === 'linear' ? 'Loss' : 'Accuracy';
  const svg = lossChartSvg;
  svg.selectAll('*').remove();
  if (hist.length < 2) return;

  const iw = LOSS_W - LOSS_MARGIN.l - LOSS_MARGIN.r;
  const ih = LOSS_H - LOSS_MARGIN.t - LOSS_MARGIN.b;

  const xS = d3.scaleLinear().domain([0, hist.length - 1]).range([0, iw]);
  const yS = d3.scaleLinear().domain([
    model === 'linear' ? 0 : 0,
    model === 'linear' ? (d3.max(hist) || 1) : 1
  ]).range([ih, 0]);

  const g = svg.append('g').attr('transform', `translate(${LOSS_MARGIN.l},${LOSS_MARGIN.t})`);

  g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(xS).ticks(4))
    .selectAll('text').style('fill', '#8888aa').style('font-size', '10px');
  g.append('g').call(d3.axisLeft(yS).ticks(4))
    .selectAll('text').style('fill', '#8888aa').style('font-size', '10px');

  g.selectAll('.domain, .tick line').style('stroke', '#2a2a5a');

  const line = d3.line<number>().x((_, i) => xS(i)).y(d => yS(d));
  g.append('path')
    .datum(hist)
    .attr('fill', 'none')
    .attr('stroke', '#a78bfa')
    .attr('stroke-width', 1.5)
    .attr('d', line);

  g.append('text').attr('x', iw / 2).attr('y', ih + 22)
    .attr('text-anchor', 'middle').attr('fill', '#8888aa')
    .style('font-size', '11px').text(`Iter / ${label}`);
}

// ─── Weights bar display ──────────────────────────────────────────────────────

function renderWeights(): void {
  const container = document.getElementById('weights-display')!;
  container.innerHTML = '';

  let entries: { name: string; val: number }[] = [];

  if (model === 'linear') {
    entries = [
      { name: 'bias', val: linState.weights[0] },
      { name: 'w₁', val: linState.weights[1] }
    ];
  } else if (model === 'logistic') {
    const w = logState.weights;
    const names = ['bias', 'x', 'y'];
    if (polyDegree >= 2) names.push('x²', 'xy', 'y²');
    if (polyDegree >= 3) names.push('x³', 'x²y', 'xy²', 'y³');
    w.forEach((wv, i) => entries.push({ name: names[i] || `w${i}`, val: wv }));
  } else if (model === 'gnb') {
    gnbState.classes.forEach((cls, c) => {
      entries.push({ name: `μ${c}ₓ`, val: cls.mean[0] });
      entries.push({ name: `μ${c}ᵧ`, val: cls.mean[1] });
      entries.push({ name: `σ${c}ₓ`, val: Math.sqrt(cls.variance[0]) });
      entries.push({ name: `σ${c}ᵧ`, val: Math.sqrt(cls.variance[1]) });
    });
  }

  if (entries.length === 0) return;
  const maxAbs = Math.max(...entries.map(e => Math.abs(e.val)), 0.01);

  entries.forEach(e => {
    const row = document.createElement('div');
    row.className = 'weight-row';
    const nameEl = document.createElement('span');
    nameEl.className = 'weight-name';
    nameEl.textContent = e.name;
    const barWrap = document.createElement('div');
    barWrap.className = 'weight-bar-wrap';
    const barPos = document.createElement('div');
    barPos.className = 'weight-bar-pos';
    const barNeg = document.createElement('div');
    barNeg.className = 'weight-bar-neg';
    const pct = Math.min(Math.abs(e.val) / maxAbs * 100, 100);
    if (e.val >= 0) {
      barPos.style.width = pct + '%';
    } else {
      barNeg.style.width = pct + '%';
    }
    barWrap.appendChild(barNeg);
    barWrap.appendChild(barPos);
    const valEl = document.createElement('span');
    valEl.className = 'weight-val';
    valEl.textContent = e.val.toFixed(3);
    row.appendChild(nameEl);
    row.appendChild(barWrap);
    row.appendChild(valEl);
    container.appendChild(row);
  });
}

// ─── Stats display ────────────────────────────────────────────────────────────

function renderStats(): void {
  const el = document.getElementById('stats-display')!;
  let html = '';
  if (model === 'linear') {
    html = `<span>Loss: <b>${linState.loss.toFixed(4)}</b></span> <span>Iter: <b>${linState.iter}</b></span>`;
  } else if (model === 'logistic') {
    html = `<span>Loss: <b>${logState.loss.toFixed(4)}</b></span> <span>Acc: <b>${(logState.acc * 100).toFixed(1)}%</b></span> <span>Iter: <b>${logState.iter}</b></span>`;
  } else if (model === 'gnb') {
    html = `<span>Acc: <b>${(gnbState.acc * 100).toFixed(1)}%</b></span>`;
  }
  el.innerHTML = html;
}

// ─── Update loop ─────────────────────────────────────────────────────────────

function update(): void {
  render();
  renderLossChart();
  renderWeights();
  renderStats();
}

function step(): void {
  if (model === 'linear') {
    linState = stepLinReg(linState, data, learningRate);
    lossHistory.push(linState.loss);
  } else if (model === 'logistic') {
    logState = stepLogReg(logState, data, learningRate, l2Reg, polyDegree);
    accHistory.push(logState.acc);
  }
}

function loop(): void {
  if (!running) return;
  for (let i = 0; i < 5; i++) step();
  update();
  animFrame = requestAnimationFrame(loop);
}

function startLoop(): void {
  if (running) return;
  running = true;
  loop();
}

function pauseLoop(): void {
  running = false;
  if (animFrame !== null) { cancelAnimationFrame(animFrame); animFrame = null; }
}

function resetModel(): void {
  pauseLoop();
  lossHistory = [];
  accHistory = [];
  if (model === 'linear') {
    linState = initLinReg();
  } else if (model === 'logistic') {
    const nf = polyFeatures(0, 0, polyDegree).length;
    logState = initLogReg(nf);
  } else if (model === 'gnb') {
    gnbState = { classes: [], acc: 0 };
  }
  update();
}

function regenData(): void {
  pauseLoop();
  data = generateDataset(dataset, nPoints, noiseLevel);
  computeScales(data);
  resetModel();
  if (model === 'gnb') {
    gnbState = trainGNB(data);
    update();
  }
}

// ─── UI wiring ────────────────────────────────────────────────────────────────

function setModel(m: ModelName): void {
  model = m;
  // update visible controls
  const gdControls = document.getElementById('gd-controls')!;
  const l2Row = document.getElementById('l2-row')!;
  const polyRow = document.getElementById('poly-row')!;
  const trainBtn = document.getElementById('btn-train')!;
  const playBtn = document.getElementById('btn-play')!;
  const pauseBtn = document.getElementById('btn-pause')!;
  const stepBtn = document.getElementById('btn-step')!;

  if (m === 'gnb') {
    gdControls.style.display = 'none';
    trainBtn.style.display = 'block';
    playBtn.style.display = 'none';
    pauseBtn.style.display = 'none';
    stepBtn.style.display = 'none';
  } else {
    gdControls.style.display = 'flex';
    trainBtn.style.display = 'none';
    playBtn.style.display = 'inline-block';
    pauseBtn.style.display = 'inline-block';
    stepBtn.style.display = 'inline-block';
  }
  l2Row.style.display = (m === 'logistic') ? 'flex' : 'none';
  polyRow.style.display = (m === 'logistic') ? 'flex' : 'none';

  // Update dataset options
  const dsSelect = document.getElementById('dataset-select') as HTMLSelectElement;
  const regressionDs = ['noisy-line', 'noisy-quad'];
  const classDs = ['blobs', 'moons', 'circles', 'xor'];
  Array.from(dsSelect.options).forEach(opt => {
    const isReg = regressionDs.indexOf(opt.value) !== -1;
    opt.hidden = (m === 'linear') ? !isReg : isReg;
  });
  if (m === 'linear' && classDs.indexOf(dsSelect.value) !== -1) {
    dsSelect.value = 'noisy-line';
    dataset = 'noisy-line';
  } else if (m !== 'linear' && regressionDs.indexOf(dsSelect.value) !== -1) {
    dsSelect.value = 'blobs';
    dataset = 'blobs';
  }
  regenData();
}

function initUI(): void {
  mainCanvas = document.getElementById('main-canvas') as HTMLCanvasElement;
  ctx = mainCanvas.getContext('2d')!;
  mainCanvas.width = CANVAS_W;
  mainCanvas.height = CANVAS_H;

  lossChartSvg = d3.select<SVGSVGElement, unknown>('#loss-chart')
    .attr('width', LOSS_W)
    .attr('height', LOSS_H);

  // Model tabs
  document.querySelectorAll('.model-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.model-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      setModel((btn as HTMLElement).dataset['model'] as ModelName);
    });
  });

  // Dataset
  const dsSelect = document.getElementById('dataset-select') as HTMLSelectElement;
  dsSelect.addEventListener('change', () => {
    dataset = dsSelect.value as DatasetName;
    regenData();
  });

  // N points
  const nPointsSlider = document.getElementById('n-points') as HTMLInputElement;
  const nPointsVal = document.getElementById('n-points-val')!;
  nPointsSlider.addEventListener('input', () => {
    nPoints = parseInt(nPointsSlider.value);
    nPointsVal.textContent = String(nPoints);
    regenData();
  });

  // Noise
  const noiseSlider = document.getElementById('noise') as HTMLInputElement;
  const noiseVal = document.getElementById('noise-val')!;
  noiseSlider.addEventListener('input', () => {
    noiseLevel = parseFloat(noiseSlider.value);
    noiseVal.textContent = noiseLevel.toFixed(1);
    regenData();
  });

  // Learning rate
  const lrSlider = document.getElementById('learning-rate') as HTMLInputElement;
  const lrVal = document.getElementById('lr-val')!;
  lrSlider.addEventListener('input', () => {
    learningRate = parseFloat(lrSlider.value);
    lrVal.textContent = learningRate.toFixed(3);
  });

  // L2
  const l2Slider = document.getElementById('l2') as HTMLInputElement;
  const l2Val = document.getElementById('l2-val')!;
  l2Slider.addEventListener('input', () => {
    l2Reg = parseFloat(l2Slider.value);
    l2Val.textContent = l2Reg.toFixed(3);
  });

  // Poly degree
  const polySlider = document.getElementById('poly-degree') as HTMLInputElement;
  const polyVal = document.getElementById('poly-val')!;
  polySlider.addEventListener('input', () => {
    polyDegree = parseInt(polySlider.value);
    polyVal.textContent = String(polyDegree);
    const nf = polyFeatures(0, 0, polyDegree).length;
    logState = initLogReg(nf);
    lossHistory = [];
    accHistory = [];
    update();
  });

  // Buttons
  document.getElementById('btn-play')!.addEventListener('click', startLoop);
  document.getElementById('btn-pause')!.addEventListener('click', pauseLoop);
  document.getElementById('btn-step')!.addEventListener('click', () => { step(); update(); });
  document.getElementById('btn-reset')!.addEventListener('click', resetModel);
  document.getElementById('btn-regen')!.addEventListener('click', regenData);
  document.getElementById('btn-train')!.addEventListener('click', () => {
    gnbState = trainGNB(data);
    update();
  });

  // Initial
  setModel('linear');
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', initUI);
