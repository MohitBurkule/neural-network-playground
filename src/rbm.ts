// Restricted Boltzmann Machine — Contrastive Divergence training
// Binary RBM: visible ↔ hidden, trained with CD-k

// ─── Types ────────────────────────────────────────────────────────────────────

interface RBM {
  nV: number;        // #visible
  nH: number;        // #hidden
  W: Float64Array;   // nV × nH, row-major
  bV: Float64Array;  // visible biases
  bH: Float64Array;  // hidden biases
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function sampleBernoulli(p: number): number {
  return Math.random() < p ? 1 : 0;
}

// ─── RBM construction ─────────────────────────────────────────────────────────

function createRBM(nV: number, nH: number): RBM {
  const W = new Float64Array(nV * nH);
  const bV = new Float64Array(nV);
  const bH = new Float64Array(nH);
  // small random weights
  for (let i = 0; i < W.length; i++) {
    W[i] = (Math.random() * 2 - 1) * 0.05;
  }
  return { nV, nH, W, bV, bH };
}

// p(h=1 | v) for each hidden unit; fills result array
function hiddenProbs(rbm: RBM, v: Float64Array, pH: Float64Array): void {
  for (let j = 0; j < rbm.nH; j++) {
    let act = rbm.bH[j];
    for (let i = 0; i < rbm.nV; i++) {
      act += v[i] * rbm.W[i * rbm.nH + j];
    }
    pH[j] = sigmoid(act);
  }
}

// p(v=1 | h) for each visible unit; fills result array
function visibleProbs(rbm: RBM, h: Float64Array, pV: Float64Array): void {
  for (let i = 0; i < rbm.nV; i++) {
    let act = rbm.bV[i];
    for (let j = 0; j < rbm.nH; j++) {
      act += h[j] * rbm.W[i * rbm.nH + j];
    }
    pV[i] = sigmoid(act);
  }
}

function sampleH(rbm: RBM, v: Float64Array, pH: Float64Array, hSample: Float64Array): void {
  hiddenProbs(rbm, v, pH);
  for (let j = 0; j < rbm.nH; j++) {
    hSample[j] = sampleBernoulli(pH[j]);
  }
}

function sampleV(rbm: RBM, h: Float64Array, pV: Float64Array, vSample: Float64Array): void {
  visibleProbs(rbm, h, pV);
  for (let i = 0; i < rbm.nV; i++) {
    vSample[i] = sampleBernoulli(pV[i]);
  }
}

// CD-k training step on one data sample; returns reconstruction error
function cdStep(rbm: RBM, v0: Float64Array, k: number, lr: number): number {
  const { nV, nH } = rbm;
  const pH0 = new Float64Array(nH);
  const h0  = new Float64Array(nH);
  const pV  = new Float64Array(nV);
  const vK  = new Float64Array(nV);
  const pHK = new Float64Array(nH);

  // positive phase
  sampleH(rbm, v0, pH0, h0);

  // negative phase: k Gibbs steps
  let hCur = h0;
  vK.set(v0);
  for (let step = 0; step < k; step++) {
    sampleV(rbm, hCur, pV, vK);
    if (step < k - 1) {
      const hNext = new Float64Array(nH);
      sampleH(rbm, vK, pHK, hNext);
      hCur = hNext;
    } else {
      hiddenProbs(rbm, vK, pHK);
    }
  }

  // update weights and biases
  for (let i = 0; i < nV; i++) {
    for (let j = 0; j < nH; j++) {
      rbm.W[i * nH + j] += lr * (v0[i] * pH0[j] - vK[i] * pHK[j]);
    }
    rbm.bV[i] += lr * (v0[i] - vK[i]);
  }
  for (let j = 0; j < nH; j++) {
    rbm.bH[j] += lr * (pH0[j] - pHK[j]);
  }

  // reconstruction error (MSE visible)
  visibleProbs(rbm, h0, pV);
  let err = 0;
  for (let i = 0; i < nV; i++) {
    const d = v0[i] - pV[i];
    err += d * d;
  }
  return err / nV;
}

// Gibbs sampling: start from v, run steps, return final visible probs
function gibbsSample(rbm: RBM, vStart: Float64Array, steps: number): Float64Array {
  const { nV, nH } = rbm;
  const pV = new Float64Array(nV);
  const pH = new Float64Array(nH);
  const h  = new Float64Array(nH);
  const v  = new Float64Array(nV);
  v.set(vStart);
  for (let s = 0; s < steps; s++) {
    sampleH(rbm, v, pH, h);
    sampleV(rbm, h, pV, v);
  }
  // return visible probs from last hidden sample
  visibleProbs(rbm, h, pV);
  return pV;
}

// Reconstruct: one forward-backward pass (probabilities)
function reconstruct(rbm: RBM, v: Float64Array): Float64Array {
  const pH = new Float64Array(rbm.nH);
  const pV = new Float64Array(rbm.nV);
  hiddenProbs(rbm, v, pH);
  visibleProbs(rbm, pH, pV);
  return pV;
}

// ─── Dataset: 6×6 binary glyphs ───────────────────────────────────────────────
// 36 bits each

const GLYPH_SIZE = 6;
// prettier-ignore
const GLYPHS: number[][] = [
  // 0: zero
  [0,1,1,1,1,0,
   1,0,0,0,1,1,
   1,0,0,1,0,1,
   1,0,1,0,0,1,
   1,1,0,0,0,1,
   0,1,1,1,1,0],
  // 1: one
  [0,0,1,1,0,0,
   0,1,1,1,0,0,
   0,0,1,1,0,0,
   0,0,1,1,0,0,
   0,0,1,1,0,0,
   1,1,1,1,1,1],
  // 2: two
  [0,1,1,1,1,0,
   1,0,0,0,0,1,
   0,0,0,0,1,1,
   0,0,1,1,0,0,
   0,1,1,0,0,0,
   1,1,1,1,1,1],
  // 3: three
  [0,1,1,1,1,0,
   1,0,0,0,0,1,
   0,0,1,1,1,0,
   0,0,0,0,0,1,
   1,0,0,0,0,1,
   0,1,1,1,1,0],
  // 4: four
  [0,0,0,1,1,0,
   0,0,1,0,1,0,
   0,1,0,0,1,0,
   1,1,1,1,1,1,
   0,0,0,0,1,0,
   0,0,0,0,1,0],
  // 5: five
  [1,1,1,1,1,1,
   1,0,0,0,0,0,
   1,1,1,1,1,0,
   0,0,0,0,0,1,
   1,0,0,0,0,1,
   0,1,1,1,1,0],
  // 6: six
  [0,1,1,1,1,0,
   1,0,0,0,0,0,
   1,1,1,1,1,0,
   1,0,0,0,0,1,
   1,0,0,0,0,1,
   0,1,1,1,1,0],
  // 7: seven
  [1,1,1,1,1,1,
   0,0,0,0,1,0,
   0,0,0,1,0,0,
   0,0,1,0,0,0,
   0,1,0,0,0,0,
   0,1,0,0,0,0],
  // 8: eight
  [0,1,1,1,1,0,
   1,0,0,0,0,1,
   0,1,1,1,1,0,
   1,0,0,0,0,1,
   1,0,0,0,0,1,
   0,1,1,1,1,0],
  // 9: nine
  [0,1,1,1,1,0,
   1,0,0,0,0,1,
   0,1,1,1,1,1,
   0,0,0,0,0,1,
   0,0,0,0,0,1,
   0,1,1,1,1,0],
  // cross
  [0,0,1,1,0,0,
   0,0,1,1,0,0,
   1,1,1,1,1,1,
   1,1,1,1,1,1,
   0,0,1,1,0,0,
   0,0,1,1,0,0],
  // diagonal \
  [1,1,0,0,0,0,
   0,1,1,0,0,0,
   0,0,1,1,0,0,
   0,0,0,1,1,0,
   0,0,0,0,1,1,
   0,0,0,0,0,1],
  // checkerboard top-half
  [1,0,1,0,1,0,
   0,1,0,1,0,1,
   1,0,1,0,1,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0,
   0,0,0,0,0,0],
  // diamond
  [0,0,1,1,0,0,
   0,1,0,0,1,0,
   1,0,0,0,0,1,
   1,0,0,0,0,1,
   0,1,0,0,1,0,
   0,0,1,1,0,0],
  // H shape
  [1,0,0,0,0,1,
   1,0,0,0,0,1,
   1,1,1,1,1,1,
   1,0,0,0,0,1,
   1,0,0,0,0,1,
   1,0,0,0,0,1],
  // U shape
  [1,0,0,0,0,1,
   1,0,0,0,0,1,
   1,0,0,0,0,1,
   1,0,0,0,0,1,
   1,0,0,0,0,1,
   0,1,1,1,1,0],
  // Z shape
  [1,1,1,1,1,1,
   0,0,0,0,1,0,
   0,0,0,1,0,0,
   0,0,1,0,0,0,
   0,1,0,0,0,0,
   1,1,1,1,1,1],
  // T shape
  [1,1,1,1,1,1,
   0,0,1,1,0,0,
   0,0,1,1,0,0,
   0,0,1,1,0,0,
   0,0,1,1,0,0,
   0,0,1,1,0,0],
];

const GLYPH_NAMES = [
  '0','1','2','3','4','5','6','7','8','9',
  'Cross','Diag','Checker','Diamond','H','U','Z','T'
];

// ─── App state ────────────────────────────────────────────────────────────────

let rbm: RBM;
let nHidden = 16;
let cdK = 1;
let learningRate = 0.02;
let epochInterval: number | null = null;
let epoch = 0;
let errorHistory: number[] = [];
let selectedExample = 0;
let dreamSteps = 100;

const dataset: Float64Array[] = GLYPHS.map(g => {
  const v = new Float64Array(GLYPH_SIZE * GLYPH_SIZE);
  for (let i = 0; i < g.length; i++) v[i] = g[i];
  return v;
});

function initRBM(): void {
  rbm = createRBM(GLYPH_SIZE * GLYPH_SIZE, nHidden);
  epoch = 0;
  errorHistory = [];
}

// One epoch: one CD pass over shuffled dataset
function runEpoch(): void {
  // shuffle indices
  const idx: number[] = [];
  for (let i = 0; i < dataset.length; i++) idx.push(i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
  }
  let totalErr = 0;
  idx.forEach(i => {
    totalErr += cdStep(rbm, dataset[i], cdK, learningRate);
  });
  epoch++;
  errorHistory.push(totalErr / dataset.length);
}

// ─── Canvas rendering helpers ─────────────────────────────────────────────────

function renderBinaryImage(
  canvas: HTMLCanvasElement,
  data: Float64Array | number[],
  cols: number,
  rows: number,
  cellSize: number,
  colorOn = '#c084fc',
  colorOff = '#1a1a38'
): void {
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = data[r * cols + c];
      // soft color by probability
      ctx.fillStyle = lerpColor(colorOff, colorOn, val);
      ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
    }
  }
}

function lerpColor(hex1: string, hex2: string, t: number): string {
  const r1 = parseInt(hex1.slice(1,3),16), g1 = parseInt(hex1.slice(3,5),16), b1 = parseInt(hex1.slice(5,7),16);
  const r2 = parseInt(hex2.slice(1,3),16), g2 = parseInt(hex2.slice(3,5),16), b2 = parseInt(hex2.slice(5,7),16);
  const r = Math.round(r1 + (r2-r1)*t);
  const g = Math.round(g1 + (g2-g1)*t);
  const b = Math.round(b1 + (b2-b1)*t);
  return `rgb(${r},${g},${b})`;
}

// Render receptive fields (one per hidden unit)
function renderReceptiveFields(canvas: HTMLCanvasElement): void {
  const nH = rbm.nH;
  const nV = rbm.nV;
  const gs = GLYPH_SIZE;
  const cell = 3; // pixels per visible unit
  const gap = 2;
  const cols = Math.ceil(Math.sqrt(nH));
  const rows = Math.ceil(nH / cols);
  const fw = gs * cell;
  const fh = gs * cell;
  canvas.width  = cols * (fw + gap) + gap;
  canvas.height = rows * (fh + gap) + gap;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // find global weight range for normalisation
  let wMin = Infinity, wMax = -Infinity;
  for (let i = 0; i < rbm.W.length; i++) {
    if (rbm.W[i] < wMin) wMin = rbm.W[i];
    if (rbm.W[i] > wMax) wMax = rbm.W[i];
  }
  const wRange = wMax - wMin || 1;

  for (let j = 0; j < nH; j++) {
    const col = j % cols;
    const row = Math.floor(j / cols);
    const ox = gap + col * (fw + gap);
    const oy = gap + row * (fh + gap);
    for (let i = 0; i < nV; i++) {
      const t = (rbm.W[i * nH + j] - wMin) / wRange;
      ctx.fillStyle = lerpColor('#1a0a3a', '#f0abfc', t);
      const vi = i % gs;
      const vr = Math.floor(i / gs);
      ctx.fillRect(ox + vi * cell, oy + vr * cell, cell, cell);
    }
  }
}

// Render error curve
function renderErrorCurve(canvas: HTMLCanvasElement): void {
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, W, H);

  if (errorHistory.length < 2) return;
  const maxErr = Math.max(...errorHistory) || 1;
  const minErr = Math.min(...errorHistory);

  ctx.strokeStyle = '#c084fc';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  errorHistory.forEach((e, i) => {
    const x = (i / (errorHistory.length - 1)) * (W - 20) + 10;
    const y = H - 10 - ((e - minErr) / (maxErr - minErr || 1)) * (H - 20);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // axes labels
  ctx.fillStyle = '#5a5a8a';
  ctx.font = '10px monospace';
  ctx.fillText(`err: ${errorHistory[errorHistory.length-1].toFixed(4)}`, 12, 14);
  ctx.fillText(`epoch ${epoch}`, W - 70, 14);
}

// ─── Dream: generate samples by Gibbs from random initial states ──────────────

function renderDreams(canvas: HTMLCanvasElement, count: number): void {
  const gs = GLYPH_SIZE;
  const cell = 6;
  const gap = 4;
  const perRow = count;
  canvas.width  = perRow * (gs * cell + gap) + gap;
  canvas.height = gs * cell + gap * 2;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let s = 0; s < count; s++) {
    // start from random noise
    const vStart = new Float64Array(gs * gs);
    for (let i = 0; i < vStart.length; i++) vStart[i] = Math.random() < 0.5 ? 1 : 0;
    const pV = gibbsSample(rbm, vStart, dreamSteps);
    const ox = gap + s * (gs * cell + gap);
    const oy = gap;
    for (let r = 0; r < gs; r++) {
      for (let c = 0; c < gs; c++) {
        const t = pV[r * gs + c];
        ctx.fillStyle = lerpColor('#1a1a38', '#c084fc', t);
        ctx.fillRect(ox + c * cell, oy + r * cell, cell, cell);
      }
    }
  }
}

// ─── Build UI ─────────────────────────────────────────────────────────────────

function buildUI(): void {
  document.body.innerHTML = `
<h1>Restricted Boltzmann Machine</h1>
<p class="subtitle">
  Binary RBM trained with Contrastive Divergence (CD-k) on 6×6 binary glyphs.
  Watch the model learn features, reconstruct inputs, and generate new samples.
</p>

<div class="main-row">

  <!-- ===== LEFT: visualisations ===== -->
  <div class="left-col">

    <!-- Receptive fields -->
    <div class="panel">
      <div class="panel-title">Receptive Fields — learned weights per hidden unit</div>
      <canvas id="rfCanvas"></canvas>
    </div>

    <!-- Input / Reconstruction -->
    <div class="panel">
      <div class="panel-title">Input &amp; Reconstruction (one forward–backward pass)</div>
      <div class="recon-row">
        <div class="recon-block">
          <div class="recon-label">Input</div>
          <canvas id="inputCanvas"></canvas>
        </div>
        <div class="recon-block">
          <div class="recon-label">Reconstruction</div>
          <canvas id="reconCanvas"></canvas>
        </div>
      </div>
    </div>

    <!-- Dreams -->
    <div class="panel">
      <div class="panel-title">Dreams — Gibbs samples from random noise</div>
      <canvas id="dreamCanvas"></canvas>
    </div>

    <!-- Error curve -->
    <div class="panel">
      <div class="panel-title">Reconstruction Error vs Epoch</div>
      <canvas id="errorCanvas" width="480" height="110"></canvas>
    </div>

  </div>

  <!-- ===== RIGHT: controls ===== -->
  <div class="controls-col">

    <div class="section-title">Model</div>

    <div class="ctrl-group">
      <label>Hidden Units</label>
      <div class="range-row">
        <input type="range" id="nHiddenSlider" min="4" max="64" step="4" value="16" />
        <span id="nHiddenVal">16</span>
      </div>
    </div>

    <div class="ctrl-group">
      <label>Learning Rate</label>
      <div class="range-row">
        <input type="range" id="lrSlider" min="-4" max="-1" step="0.25" value="-2" />
        <span id="lrVal">0.0100</span>
      </div>
    </div>

    <div class="ctrl-group">
      <label>CD-k Steps</label>
      <div class="range-row">
        <input type="range" id="cdkSlider" min="1" max="10" step="1" value="1" />
        <span id="cdkVal">1</span>
      </div>
    </div>

    <div class="section-title">Training</div>

    <div class="btn-row">
      <button id="playBtn">▶ Play</button>
      <button id="stepBtn">Step</button>
    </div>
    <div class="btn-row">
      <button id="resetBtn">Reset</button>
    </div>

    <div class="section-title">Example</div>

    <div class="ctrl-group">
      <label>Input Glyph</label>
      <select id="exampleSelect"></select>
    </div>

    <div class="section-title">Dream</div>

    <div class="ctrl-group">
      <label>Gibbs Steps</label>
      <div class="range-row">
        <input type="range" id="dreamSlider" min="10" max="1000" step="10" value="100" />
        <span id="dreamVal">100</span>
      </div>
    </div>

    <div class="btn-row">
      <button id="dreamBtn">Dream ✦</button>
    </div>

    <div class="section-title">Status</div>
    <div id="status">Ready. Press ▶ Play to train.</div>

    <div class="section-title">Theory</div>
    <div class="theory-box">
      <strong>CD-k:</strong> positive phase <code>⟨v h⟩_data</code> minus negative phase <code>⟨v h⟩_k</code><br/>
      <strong>Update:</strong> <code>ΔW = lr × (p_h0 v0ᵀ − p_hk vkᵀ)</code><br/>
      <strong>Dream:</strong> run Gibbs chain from noise → model's prior.
    </div>

  </div>
</div>
`;

  // Populate example select
  const sel = document.getElementById('exampleSelect') as HTMLSelectElement;
  GLYPH_NAMES.forEach((name, i) => {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = name;
    sel.appendChild(opt);
  });

  // Slider wiring
  wireSlider('nHiddenSlider', 'nHiddenVal', v => {
    nHidden = v;
  }, v => String(v));

  wireSlider('lrSlider', 'lrVal', v => {
    learningRate = Math.pow(10, v);
  }, v => Math.pow(10, v).toFixed(4));

  wireSlider('cdkSlider', 'cdkVal', v => {
    cdK = v;
  }, v => String(v));

  wireSlider('dreamSlider', 'dreamVal', v => {
    dreamSteps = v;
  }, v => String(v));

  sel.addEventListener('change', () => {
    selectedExample = parseInt(sel.value);
    renderAll();
  });

  document.getElementById('playBtn')!.addEventListener('click', togglePlay);
  document.getElementById('stepBtn')!.addEventListener('click', () => { rbmDoStep(); renderAll(); });
  document.getElementById('resetBtn')!.addEventListener('click', () => {
    stopPlay();
    nHidden = parseInt((document.getElementById('nHiddenSlider') as HTMLInputElement).value);
    initRBM();
    renderAll();
    setStatus('Model reset.');
  });
  document.getElementById('dreamBtn')!.addEventListener('click', () => {
    const dc = document.getElementById('dreamCanvas') as HTMLCanvasElement;
    renderDreams(dc, 6);
  });
}

function wireSlider(
  sliderId: string,
  valId: string,
  onValue: (v: number) => void,
  fmt: (v: number) => string
): void {
  const slider = document.getElementById(sliderId) as HTMLInputElement;
  const valEl  = document.getElementById(valId)!;
  const update = () => {
    const v = parseFloat(slider.value);
    onValue(v);
    valEl.textContent = fmt(v);
  };
  slider.addEventListener('input', update);
  update();
}

function setStatus(msg: string): void {
  const el = document.getElementById('status');
  if (el) el.textContent = msg;
}

let playing = false;
function togglePlay(): void {
  playing = !playing;
  const btn = document.getElementById('playBtn')!;
  if (playing) {
    btn.textContent = '⏸ Pause';
    scheduleEpoch();
  } else {
    btn.textContent = '▶ Play';
    if (epochInterval !== null) {
      cancelAnimationFrame(epochInterval);
      epochInterval = null;
    }
  }
}

function stopPlay(): void {
  playing = false;
  const btn = document.getElementById('playBtn');
  if (btn) btn.textContent = '▶ Play';
  if (epochInterval !== null) {
    cancelAnimationFrame(epochInterval);
    epochInterval = null;
  }
}

function scheduleEpoch(): void {
  epochInterval = requestAnimationFrame(() => {
    if (!playing) return;
    rbmDoStep();
    renderAll();
    scheduleEpoch();
  });
}

function rbmDoStep(): void {
  runEpoch();
  const lastErr = errorHistory[errorHistory.length - 1];
  setStatus(`Epoch ${epoch} — avg reconstruction error: ${lastErr.toFixed(5)}`);
}

function renderAll(): void {
  const rfCanvas    = document.getElementById('rfCanvas')    as HTMLCanvasElement;
  const inputCanvas = document.getElementById('inputCanvas') as HTMLCanvasElement;
  const reconCanvas = document.getElementById('reconCanvas') as HTMLCanvasElement;
  const errorCanvas = document.getElementById('errorCanvas') as HTMLCanvasElement;

  renderReceptiveFields(rfCanvas);

  const v0 = dataset[selectedExample];
  renderBinaryImage(inputCanvas, v0, GLYPH_SIZE, GLYPH_SIZE, 10);
  const recon = reconstruct(rbm, v0);
  renderBinaryImage(reconCanvas, recon, GLYPH_SIZE, GLYPH_SIZE, 10);

  renderErrorCurve(errorCanvas);

  const dc = document.getElementById('dreamCanvas') as HTMLCanvasElement;
  renderDreams(dc, 6);
}

// ─── CSS injection ────────────────────────────────────────────────────────────

function injectStyles(): void {
  const style = document.createElement('style');
  style.textContent = `
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #0a0a1a;
  color: #d0d0e8;
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 12px 48px;
}

h1 {
  font-size: 1.5rem;
  font-weight: 700;
  color: #c084fc;
  margin-bottom: 4px;
  letter-spacing: 0.04em;
}

.subtitle {
  color: #6b7280;
  font-size: 0.82rem;
  margin-bottom: 20px;
  text-align: center;
  max-width: 640px;
}

.main-row {
  display: flex;
  gap: 16px;
  align-items: flex-start;
  flex-wrap: wrap;
  justify-content: center;
  width: 100%;
  max-width: 1100px;
}

.left-col {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.panel {
  background: #0d0d1f;
  border: 1px solid #2a2a4a;
  border-radius: 10px;
  padding: 12px 14px;
}

.panel-title {
  font-size: 0.70rem;
  color: #5a5a8a;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}

.recon-row {
  display: flex;
  gap: 20px;
  align-items: flex-start;
}

.recon-block {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.recon-label {
  font-size: 0.70rem;
  color: #7070a0;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

canvas { display: block; image-rendering: pixelated; border-radius: 4px; }

.controls-col {
  background: #10102a;
  border: 1px solid #2a2a4a;
  border-radius: 10px;
  padding: 18px;
  width: 240px;
  min-width: 200px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.section-title {
  font-size: 0.70rem;
  color: #5a5a8a;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  border-bottom: 1px solid #22224a;
  padding-bottom: 4px;
}

.ctrl-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.ctrl-group label {
  font-size: 0.76rem;
  color: #8888bb;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

select, input[type=range] {
  width: 100%;
  background: #1a1a38;
  border: 1px solid #3a3a5c;
  border-radius: 5px;
  color: #d0d0e8;
  padding: 4px 6px;
  font-size: 0.85rem;
  cursor: pointer;
}

input[type=range] {
  padding: 0;
  accent-color: #c084fc;
  height: 18px;
}

.range-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 6px;
}

.range-row span {
  min-width: 40px;
  text-align: right;
  font-size: 0.82rem;
  color: #c0c0e0;
  font-variant-numeric: tabular-nums;
}

.btn-row {
  display: flex;
  gap: 5px;
  flex-wrap: wrap;
}

button {
  flex: 1;
  padding: 6px 4px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.78rem;
  font-weight: 600;
  transition: opacity 0.12s;
  white-space: nowrap;
}

button:hover { opacity: 0.82; }

#playBtn  { background: #7c3aed; color: #fff; }
#stepBtn  { background: #1e3a5f; color: #7dd3fc; }
#resetBtn { background: #3b1f1f; color: #fca5a5; }
#dreamBtn { background: #166534; color: #86efac; flex: 2; }

#status {
  font-size: 0.78rem;
  color: #8888cc;
  background: #0d0d22;
  border: 1px solid #2a2a44;
  border-radius: 6px;
  padding: 5px 10px;
  min-height: 28px;
  word-break: break-word;
}

.theory-box {
  background: #0d0d22;
  border-left: 3px solid #7c3aed;
  border-radius: 0 6px 6px 0;
  padding: 8px 12px;
  font-size: 0.75rem;
  color: #7070a0;
  line-height: 1.55;
}

.theory-box code {
  color: #c4b5fd;
  font-family: monospace;
}
  `;
  document.head.appendChild(style);
}

// ─── Entry point ──────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  injectStyles();
  initRBM();
  buildUI();
  renderAll();
});

export {};
