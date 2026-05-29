/**
 * ML Security Lab — entry point.
 *
 * Bundles together:
 *   - adversarial.ts  (FGSM / PGD attacks)
 *   - unlearning.ts   (gradient-ascent forgetting + retrain baseline)
 *   - customdataset.ts (CSV parser)
 *
 * The UI is defined in labs/security.html.  This file provides all
 * interactive logic: training, attack, unlearning, custom dataset loading,
 * and canvas-based visualisation of the decision boundary.
 *
 * Architecture assumptions:
 *   Network shape [2, 6, 6, 1] with TANH hidden activations and LINEAR output.
 *   Raw (x, y) inputs only — domain [-6, 6].
 */

import {
  buildNetwork, forwardProp, backProp, updateWeights,
  Activations, Errors, OptimizerType, Node
} from './nn';
import {
  Example2D,
  classifyCircleData,
  classifySpiralData,
  classifyXORData,
  classifyTwoGaussData
} from './dataset';
import {fgsm, pgd, predict} from './adversarial';
import {
  forgetPoints, retrainWithout, measureForgetMetrics,
  ForgetMetrics, evalMetrics
} from './unlearning';
import {parseCSV, serializeCSV} from './customdataset';

// ---------------------------------------------------------------------------
// Types / globals
// ---------------------------------------------------------------------------

/** Canvas size in pixels. */
const CANVAS_SIZE = 400;
/** Data domain: [-DOMAIN, DOMAIN]. */
const DOMAIN = 6;
/** Default network shape. */
const NET_SHAPE = [2, 6, 6, 1];

/** Application state. */
interface AppState {
  network: Node[][] | null;
  trainData: Example2D[];
  /** Points the user has brushed for forgetting. */
  forgetSet: Example2D[];
  /** Adversarial counterparts (paired with trainData subset). */
  advPoints: Array<{orig: Example2D; adv: Example2D}>;
  trainStep: number;
  animHandle: number | null;
}

const state: AppState = {
  network: null,
  trainData: [],
  forgetSet: [],
  advPoints: [],
  trainStep: 0,
  animHandle: null
};

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

function el<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

function setStatus(msg: string): void {
  el('status').textContent = msg;
}

// ---------------------------------------------------------------------------
// Canvas rendering
// ---------------------------------------------------------------------------

/**
 * Draw the decision boundary heat-map plus data points on a canvas.
 *
 * @param canvas       Target canvas element.
 * @param network      Trained network (or null → blank).
 * @param points       Data points to overlay.
 * @param advPoints    Adversarial points to overlay (red crosses).
 * @param forgetSet    Points marked for forgetting (hollow circles).
 */
function renderCanvas(
    canvas: HTMLCanvasElement,
    network: Node[][] | null,
    points: Example2D[],
    advPoints: Array<{orig: Example2D; adv: Example2D}> = [],
    forgetSet: Example2D[] = []): void {

  const ctx = canvas.getContext('2d')!;
  const w = canvas.width;
  const h = canvas.height;

  // 1. Draw background heat-map (decision boundary)
  const imageData = ctx.createImageData(w, h);
  if (network) {
    for (let px = 0; px < w; px++) {
      for (let py = 0; py < h; py++) {
        const x = (px / w) * 2 * DOMAIN - DOMAIN;
        const y = DOMAIN - (py / h) * 2 * DOMAIN;
        const val = predict(network, x, y);
        // val in roughly [-1, 1] → colour
        const t = Math.max(-1, Math.min(1, val));
        const r = t > 0 ? Math.round(255 * t) : 0;
        const b = t < 0 ? Math.round(255 * -t) : 0;
        const g = 0;
        const a = 80 + Math.round(80 * Math.abs(t));
        const idx = (py * w + px) * 4;
        imageData.data[idx]     = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = a;
      }
    }
  } else {
    // Grey blank
    for (let i = 0; i < w * h * 4; i += 4) {
      imageData.data[i] = imageData.data[i + 1] = imageData.data[i + 2] = 220;
      imageData.data[i + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);

  // Helper: map domain coords → canvas pixels
  function toPixel(x: number, y: number): [number, number] {
    const px = ((x + DOMAIN) / (2 * DOMAIN)) * w;
    const py = ((DOMAIN - y) / (2 * DOMAIN)) * h;
    return [px, py];
  }

  // 2. Draw normal data points
  const forgetIds = new Set(forgetSet.map(p => `${p.x.toFixed(4)}_${p.y.toFixed(4)}`));
  for (const pt of points) {
    const [px, py] = toPixel(pt.x, pt.y);
    const isForgotten = forgetIds.has(`${pt.x.toFixed(4)}_${pt.y.toFixed(4)}`);
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, 2 * Math.PI);
    if (isForgotten) {
      // Hollow circle with thick border for forget-set points
      ctx.strokeStyle = pt.label > 0 ? '#cc0000' : '#0000cc';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      ctx.fillStyle = pt.label > 0 ? '#ff6666' : '#6666ff';
      ctx.fill();
      ctx.strokeStyle = pt.label > 0 ? '#990000' : '#000099';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  }

  // 3. Draw adversarial arrows: original → perturbed
  for (const {orig, adv} of advPoints) {
    const [ox, oy] = toPixel(orig.x, orig.y);
    const [ax, ay] = toPixel(adv.x, adv.y);

    // Arrow line
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    ctx.lineTo(ax, ay);
    ctx.strokeStyle = '#ff9900';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Perturbed point (orange cross)
    ctx.strokeStyle = '#ff6600';
    ctx.lineWidth = 2;
    const cs = 5;
    ctx.beginPath();
    ctx.moveTo(ax - cs, ay - cs); ctx.lineTo(ax + cs, ay + cs);
    ctx.moveTo(ax + cs, ay - cs); ctx.lineTo(ax - cs, ay + cs);
    ctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// Dataset selection
// ---------------------------------------------------------------------------

function getDataset(): Example2D[] {
  const sel = (el<HTMLSelectElement>('datasetSelect')).value;
  const n = parseInt((el<HTMLInputElement>('numSamples')).value, 10) || 200;
  const noise = parseFloat((el<HTMLInputElement>('noiseLevel')).value) || 0;
  switch (sel) {
    case 'circle':  return classifyCircleData(n, noise);
    case 'xor':     return classifyXORData(n, noise);
    case 'spiral':  return classifySpiralData(n, noise);
    case 'gauss':   return classifyTwoGaussData(n, noise);
    default:        return classifyCircleData(n, noise);
  }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

function buildFreshNetwork(): Node[][] {
  return buildNetwork(NET_SHAPE, Activations.TANH, Activations.LINEAR, ['x', 'y']);
}

function trainStep(network: Node[][], data: Example2D[], lr: number): void {
  // Shuffle order each step
  const perm = data.slice().sort(() => Math.random() - 0.5);
  for (const ex of perm) {
    forwardProp(network, [ex.x, ex.y], null);
    backProp(network, ex.label, Errors.SQUARE);
    updateWeights(network, lr, null, 0, OptimizerType.SGD);
  }
}

let _trainInterval: number | null = null;

function startTraining(): void {
  if (_trainInterval !== null) return;

  const lr = parseFloat((el<HTMLInputElement>('learningRate')).value) || 0.03;
  _trainInterval = window.setInterval(() => {
    if (!state.network || !state.trainData.length) return;
    trainStep(state.network, state.trainData, lr);
    state.trainStep++;
    const metrics = evalMetrics(state.network, state.trainData);
    setStatus(`Step ${state.trainStep} | Loss: ${metrics.loss.toFixed(4)} | Acc: ${(metrics.accuracy * 100).toFixed(1)}%`);
    renderCanvas(
      el<HTMLCanvasElement>('mainCanvas'),
      state.network, state.trainData, state.advPoints, state.forgetSet);
  }, 80);
}

function stopTraining(): void {
  if (_trainInterval !== null) {
    clearInterval(_trainInterval);
    _trainInterval = null;
  }
}

// ---------------------------------------------------------------------------
// Section A: Adversarial examples
// ---------------------------------------------------------------------------

function runAdversarial(): void {
  if (!state.network) { alert('Train a network first.'); return; }

  const method = (el<HTMLSelectElement>('attackMethod')).value;
  const epsilon = parseFloat((el<HTMLInputElement>('epsilon')).value) || 0.5;
  const steps = parseInt((el<HTMLInputElement>('pgdSteps')).value, 10) || 10;

  // Sample up to 30 points
  const sample = state.trainData.slice().sort(() => Math.random() - 0.5).slice(0, 30);
  state.advPoints = [];

  let flipped = 0;
  for (const pt of sample) {
    const adv = method === 'pgd'
      ? pgd(state.network, pt, epsilon, steps, epsilon / (steps * 2))
      : fgsm(state.network, pt, epsilon);

    const origPred = predict(state.network, pt.x, pt.y);
    const advPred  = predict(state.network, adv.x, adv.y);
    if (Math.sign(origPred) !== Math.sign(advPred)) flipped++;

    state.advPoints.push({orig: pt, adv});
  }

  const flipPct = ((flipped / sample.length) * 100).toFixed(1);
  el('advResults').textContent =
    `Attack: ${method.toUpperCase()}, ε=${epsilon} | ` +
    `${flipped}/${sample.length} predictions flipped (${flipPct}%). ` +
    `Orange arrows show perturbations.`;

  renderCanvas(
    el<HTMLCanvasElement>('mainCanvas'),
    state.network, state.trainData, state.advPoints, state.forgetSet);
}

function clearAdversarial(): void {
  state.advPoints = [];
  renderCanvas(
    el<HTMLCanvasElement>('mainCanvas'),
    state.network, state.trainData, [], state.forgetSet);
  el('advResults').textContent = '';
}

// ---------------------------------------------------------------------------
// Section B: Unlearning — brush selection
// ---------------------------------------------------------------------------

let _brushing = false;
let _brushStart: {x: number; y: number} | null = null;
let _brushRect: {x1: number; y1: number; x2: number; y2: number} | null = null;

function canvasToWorld(canvas: HTMLCanvasElement, cx: number, cy: number): {x: number; y: number} {
  const rect = canvas.getBoundingClientRect();
  const px = (cx - rect.left) * (canvas.width / rect.width);
  const py = (cy - rect.top)  * (canvas.height / rect.height);
  return {
    x: (px / canvas.width)  * 2 * DOMAIN - DOMAIN,
    y: DOMAIN - (py / canvas.height) * 2 * DOMAIN
  };
}

function initBrush(): void {
  const canvas = el<HTMLCanvasElement>('mainCanvas');

  canvas.addEventListener('mousedown', (e) => {
    if (!el<HTMLInputElement>('brushMode').checked) return;
    _brushing = true;
    _brushStart = canvasToWorld(canvas, e.clientX, e.clientY);
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!_brushing || !_brushStart) return;
    const cur = canvasToWorld(canvas, e.clientX, e.clientY);
    _brushRect = {
      x1: Math.min(_brushStart.x, cur.x),
      y1: Math.min(_brushStart.y, cur.y),
      x2: Math.max(_brushStart.x, cur.x),
      y2: Math.max(_brushStart.y, cur.y)
    };
    // Overlay brush rectangle
    renderCanvas(canvas, state.network, state.trainData, state.advPoints, state.forgetSet);
    drawBrushOverlay(canvas, _brushRect);
  });

  canvas.addEventListener('mouseup', () => {
    if (!_brushing || !_brushRect) { _brushing = false; return; }
    _brushing = false;
    // Select points inside rectangle
    const r = _brushRect;
    state.forgetSet = state.trainData.filter(
      p => p.x >= r.x1 && p.x <= r.x2 && p.y >= r.y1 && p.y <= r.y2);
    el('forgetCount').textContent = `${state.forgetSet.length} points selected for forgetting.`;
    renderCanvas(canvas, state.network, state.trainData, state.advPoints, state.forgetSet);
    _brushRect = null;
  });
}

function drawBrushOverlay(
    canvas: HTMLCanvasElement,
    r: {x1: number; y1: number; x2: number; y2: number}): void {
  const ctx = canvas.getContext('2d')!;
  const w = canvas.width, h = canvas.height;
  function toPixel(x: number, y: number): [number, number] {
    return [((x + DOMAIN) / (2 * DOMAIN)) * w, ((DOMAIN - y) / (2 * DOMAIN)) * h];
  }
  const [px1, py2] = toPixel(r.x1, r.y1);
  const [px2, py1] = toPixel(r.x2, r.y2);
  ctx.save();
  ctx.strokeStyle = '#ff9900';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);
  ctx.fillStyle = 'rgba(255,153,0,0.08)';
  ctx.fillRect(px1, py1, px2 - px1, py2 - py1);
  ctx.restore();
}

function runForget(): void {
  if (!state.network) { alert('Train a network first.'); return; }
  if (state.forgetSet.length === 0) { alert('Select points to forget by drawing a brush rectangle (enable Brush Mode).'); return; }

  stopTraining();
  const retainSet = state.trainData.filter(
    p => !state.forgetSet.some(f => f.x === p.x && f.y === p.y));

  const steps = parseInt((el<HTMLInputElement>('unlearnSteps')).value, 10) || 200;
  const lr = parseFloat((el<HTMLInputElement>('unlearnLR')).value) || 0.03;

  const result = forgetPoints(state.network, state.forgetSet, retainSet, {steps, learningRate: lr});

  const b = result.before;
  const a = result.after;
  el('unlearnResults').innerHTML =
    `<b>Gradient-Ascent Unlearning</b> (${steps} steps)<br>` +
    `<table><tr><th></th><th>Forget set</th><th>Retain set</th></tr>` +
    `<tr><td>Loss before</td><td>${b.forgetLoss.toFixed(4)}</td><td>${b.retainLoss.toFixed(4)}</td></tr>` +
    `<tr><td>Loss after</td><td>${a.forgetLoss.toFixed(4)}</td><td>${a.retainLoss.toFixed(4)}</td></tr>` +
    `<tr><td>Acc before</td><td>${(b.forgetAccuracy*100).toFixed(1)}%</td><td>${(b.retainAccuracy*100).toFixed(1)}%</td></tr>` +
    `<tr><td>Acc after</td><td>${(a.forgetAccuracy*100).toFixed(1)}%</td><td>${(a.retainAccuracy*100).toFixed(1)}%</td></tr>` +
    `</table>`;

  renderCanvas(el<HTMLCanvasElement>('mainCanvas'), state.network, state.trainData, [], state.forgetSet);
}

function runRetrain(): void {
  if (state.forgetSet.length === 0) { alert('Select points to forget first.'); return; }
  stopTraining();

  const retainSet = state.trainData.filter(
    p => !state.forgetSet.some(f => f.x === p.x && f.y === p.y));
  const steps = parseInt((el<HTMLInputElement>('unlearnSteps')).value, 10) || 1000;
  const lr = parseFloat((el<HTMLInputElement>('unlearnLR')).value) || 0.03;

  const {network, metrics} = retrainWithout(NET_SHAPE, retainSet, state.forgetSet, {steps, learningRate: lr});
  state.network = network;
  state.trainStep = steps;

  el('unlearnResults').innerHTML =
    `<b>Retrain-from-Scratch Baseline</b> (${steps} steps on retain set only)<br>` +
    `<table><tr><th></th><th>Forget set</th><th>Retain set</th></tr>` +
    `<tr><td>Loss</td><td>${metrics.forgetLoss.toFixed(4)}</td><td>${metrics.retainLoss.toFixed(4)}</td></tr>` +
    `<tr><td>Acc</td><td>${(metrics.forgetAccuracy*100).toFixed(1)}%</td><td>${(metrics.retainAccuracy*100).toFixed(1)}%</td></tr>` +
    `</table>`;

  renderCanvas(el<HTMLCanvasElement>('mainCanvas'), state.network, state.trainData, [], state.forgetSet);
}

// ---------------------------------------------------------------------------
// Section C: Custom dataset
// ---------------------------------------------------------------------------

function loadCustomDataset(): void {
  const text = (el<HTMLTextAreaElement>('csvInput')).value;
  const {examples, errors} = parseCSV(text);

  if (errors.length > 0) {
    el('csvErrors').textContent = 'Warnings:\n' + errors.join('\n');
  } else {
    el('csvErrors').textContent = '';
  }

  if (examples.length === 0) {
    alert('No valid examples parsed. Check your CSV format.');
    return;
  }

  stopTraining();
  state.trainData = examples;
  state.forgetSet = [];
  state.advPoints = [];
  state.network = buildFreshNetwork();
  state.trainStep = 0;
  setStatus(`Loaded ${examples.length} custom examples. Press Train to begin.`);
  renderCanvas(el<HTMLCanvasElement>('mainCanvas'), null, state.trainData);
}

function exportDataset(): void {
  if (!state.trainData.length) { alert('No data to export.'); return; }
  const text = serializeCSV(state.trainData);
  const blob = new Blob([text], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'dataset.csv'; a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Wire up controls (called on DOMContentLoaded)
// ---------------------------------------------------------------------------

function init(): void {
  // Load initial dataset
  state.trainData = getDataset();
  state.network = buildFreshNetwork();
  renderCanvas(el<HTMLCanvasElement>('mainCanvas'), null, state.trainData);
  setStatus('Dataset loaded. Press "Train" to start.');

  el('btnGenData').addEventListener('click', () => {
    stopTraining();
    state.trainData = getDataset();
    state.forgetSet = [];
    state.advPoints = [];
    state.network = buildFreshNetwork();
    state.trainStep = 0;
    setStatus('New dataset generated. Press "Train" to start.');
    renderCanvas(el<HTMLCanvasElement>('mainCanvas'), null, state.trainData);
  });

  el('btnTrain').addEventListener('click', () => {
    if (_trainInterval !== null) {
      stopTraining();
      el('btnTrain').textContent = 'Train';
    } else {
      el('btnTrain').textContent = 'Pause';
      startTraining();
    }
  });

  el('btnReset').addEventListener('click', () => {
    stopTraining();
    state.network = buildFreshNetwork();
    state.trainStep = 0;
    state.advPoints = [];
    el('btnTrain').textContent = 'Train';
    setStatus('Network reset. Press "Train" to start.');
    renderCanvas(el<HTMLCanvasElement>('mainCanvas'), null, state.trainData);
  });

  // Section A
  el('btnAttack').addEventListener('click', runAdversarial);
  el('btnClearAdv').addEventListener('click', clearAdversarial);

  // Section B
  el('btnForget').addEventListener('click', runForget);
  el('btnRetrain').addEventListener('click', runRetrain);
  el('btnClearForget').addEventListener('click', () => {
    state.forgetSet = [];
    el('forgetCount').textContent = 'No points selected.';
    renderCanvas(el<HTMLCanvasElement>('mainCanvas'), state.network, state.trainData, state.advPoints, []);
  });

  // Section C
  el('btnLoadCSV').addEventListener('click', loadCustomDataset);
  el('btnExportCSV').addEventListener('click', exportDataset);

  initBrush();
}

// Boot
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
