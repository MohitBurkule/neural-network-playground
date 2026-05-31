// Fourier Lab — two modes: Epicycles (2D DFT) and 1D Signal
// Pure DFT implementation, no external math libraries.

import * as d3 from 'd3';

// ─── Complex number helpers ───────────────────────────────────────────────────

interface Complex { re: number; im: number; }

function cadd(a: Complex, b: Complex): Complex { return { re: a.re + b.re, im: a.im + b.im }; }
function cmul(a: Complex, b: Complex): Complex {
  return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}
function cexp(theta: number): Complex { return { re: Math.cos(theta), im: Math.sin(theta) }; }
function cabs(a: Complex): number { return Math.sqrt(a.re * a.re + a.im * a.im); }
function carg(a: Complex): number { return Math.atan2(a.im, a.re); }

// ─── DFT ─────────────────────────────────────────────────────────────────────

interface Phasor {
  freq: number;   // integer frequency index
  amp: number;    // amplitude |X[k]|
  phase: number;  // arg(X[k])
  re: number;
  im: number;
}

function dft(signal: Complex[]): Phasor[] {
  const N = signal.length;
  const result: Phasor[] = [];
  for (let k = 0; k < N; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const phi = (2 * Math.PI * k * n) / N;
      const c = cexp(-phi);
      const p = cmul(signal[n], c);
      re += p.re;
      im += p.im;
    }
    re /= N;
    im /= N;
    result.push({ freq: k, amp: Math.sqrt(re * re + im * im), phase: Math.atan2(im, re), re, im });
  }
  return result;
}

// ─── Preset shape generators ──────────────────────────────────────────────────

function sampleSquare(N: number): Complex[] {
  const pts: Complex[] = [];
  const sides = 4;
  for (let i = 0; i < N; i++) {
    const t = (i / N) * sides;
    const side = Math.floor(t);
    const frac = t - side;
    let x = 0, y = 0;
    const s = 150;
    if (side === 0) { x = -s + 2 * s * frac; y = -s; }
    else if (side === 1) { x = s; y = -s + 2 * s * frac; }
    else if (side === 2) { x = s - 2 * s * frac; y = s; }
    else { x = -s; y = s - 2 * s * frac; }
    pts.push({ re: x, im: y });
  }
  return pts;
}

function sampleStar(N: number, points: number = 5): Complex[] {
  const pts: Complex[] = [];
  const outer = 140, inner = 55;
  for (let i = 0; i < N; i++) {
    const t = (i / N) * 2 * Math.PI;
    const totalPeaks = points * 2;
    const sector = (t / (2 * Math.PI)) * totalPeaks;
    const frac = sector - Math.floor(sector);
    const r1 = Math.floor(sector) % 2 === 0 ? outer : inner;
    const r2 = Math.floor(sector) % 2 === 0 ? inner : outer;
    const r = r1 + (r2 - r1) * frac;
    pts.push({ re: r * Math.cos(t - Math.PI / 2), im: r * Math.sin(t - Math.PI / 2) });
  }
  return pts;
}

function sampleHeart(N: number): Complex[] {
  const pts: Complex[] = [];
  for (let i = 0; i < N; i++) {
    const t = (i / N) * 2 * Math.PI;
    const x = 130 * (16 * Math.pow(Math.sin(t), 3)) / 16;
    const y = -130 * (13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) / 16;
    pts.push({ re: x, im: y });
  }
  return pts;
}

function sampleFigure8(N: number): Complex[] {
  const pts: Complex[] = [];
  for (let i = 0; i < N; i++) {
    const t = (i / N) * 2 * Math.PI;
    pts.push({ re: 150 * Math.sin(t), im: 100 * Math.sin(2 * t) });
  }
  return pts;
}

// ─── App state ────────────────────────────────────────────────────────────────

type Mode = 'epicycles' | '1d';
type Preset = 'draw' | 'square' | 'star' | 'heart' | 'figure8';
type Waveform = 'square' | 'sawtooth' | 'triangle' | 'custom';

interface AppState {
  mode: Mode;
  // epicycles
  preset: Preset;
  drawing: boolean;
  drawnPath: Complex[];
  phasors: Phasor[];
  numTerms: number;
  epicycleTime: number;
  epicycleSpeed: number;
  tracePoints: Array<{ x: number; y: number }>;
  animating: boolean;
  // 1d
  waveform: Waveform;
  numHarmonics: number;
  showSpectrum: boolean;
  signal1dTime: number;
  signal1dSpeed: number;
}

const state: AppState = {
  mode: 'epicycles',
  preset: 'square',
  drawing: false,
  drawnPath: [],
  phasors: [],
  numTerms: 10,
  epicycleTime: 0,
  epicycleSpeed: 1,
  tracePoints: [],
  animating: false,
  waveform: 'square',
  numHarmonics: 5,
  showSpectrum: true,
  signal1dTime: 0,
  signal1dSpeed: 1,
};

// ─── DOM references ───────────────────────────────────────────────────────────

const epicyclesCanvas = document.getElementById('epicycles-canvas') as HTMLCanvasElement;
const signal1dCanvas = document.getElementById('signal1d-canvas') as HTMLCanvasElement;
const spectrumCanvas = document.getElementById('spectrum-canvas') as HTMLCanvasElement;

const ctxE = epicyclesCanvas.getContext('2d')!;
const ctx1d = signal1dCanvas.getContext('2d')!;
const ctxS = spectrumCanvas.getContext('2d')!;

// ─── Colors ───────────────────────────────────────────────────────────────────

const C = {
  bg: '#0f0f1a',
  grid: '#1e1e3a',
  circle: 'rgba(99,102,241,0.35)',
  circleStroke: 'rgba(99,102,241,0.8)',
  arm: '#818cf8',
  trace: '#f472b6',
  target: 'rgba(251,191,36,0.5)',
  approx: '#34d399',
  spectrum: '#60a5fa',
  text: '#e2e8f0',
  accent: '#f472b6',
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function resizeCanvas(c: HTMLCanvasElement): void {
  c.width = c.offsetWidth;
  c.height = c.offsetHeight;
}

function clearCanvas(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawGrid(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement): void {
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  const step = 40;
  for (let x = 0; x < canvas.width; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }
}

// ─── Epicycles rendering ─────────────────────────────────────────────────────

function preparePhasors(path: Complex[]): void {
  if (path.length < 4) return;
  // Resample to power-of-2 for speed but any N works
  const N = Math.min(512, path.length);
  const sampled: Complex[] = [];
  for (let i = 0; i < N; i++) {
    const idx = Math.floor((i / N) * path.length);
    sampled.push(path[idx]);
  }
  state.phasors = dft(sampled);
  // Sort descending by amplitude
  state.phasors.sort((a, b) => b.amp - a.amp);
  state.numTerms = Math.min(state.numTerms, state.phasors.length);
  state.tracePoints = [];
  state.epicycleTime = 0;

  // Update slider max
  const slider = document.getElementById('terms-slider') as HTMLInputElement;
  if (slider) {
    slider.max = String(state.phasors.length);
    slider.value = String(state.numTerms);
    const label = document.getElementById('terms-label');
    if (label) label.textContent = String(state.numTerms);
  }
}

function loadPreset(preset: Preset): void {
  if (preset === 'draw') return;
  const N = 256;
  if (preset === 'square') state.drawnPath = sampleSquare(N);
  else if (preset === 'star') state.drawnPath = sampleStar(N);
  else if (preset === 'heart') state.drawnPath = sampleHeart(N);
  else if (preset === 'figure8') state.drawnPath = sampleFigure8(N);
  preparePhasors(state.drawnPath);
}

function drawEpicycles(t: number): void {
  const canvas = epicyclesCanvas;
  const ctx = ctxE;
  clearCanvas(ctx, canvas);
  drawGrid(ctx, canvas);

  const cx = canvas.width / 2;
  const cy = canvas.height / 2;

  const terms = Math.max(1, state.numTerms);
  const usedPhasors = state.phasors.slice(0, terms);

  let x = cx, y = cy;

  usedPhasors.forEach((p) => {
    const prevX = x, prevY = y;
    const angle = 2 * Math.PI * p.freq * t / state.phasors.length + p.phase;
    const dx = p.amp * Math.cos(angle);
    const dy = p.amp * Math.sin(angle);
    const nx = prevX + dx;
    const ny = prevY + dy;

    // Draw circle
    ctx.beginPath();
    ctx.arc(prevX, prevY, p.amp, 0, 2 * Math.PI);
    ctx.strokeStyle = C.circle;
    ctx.lineWidth = 1;
    ctx.stroke();

    // Draw arm
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(nx, ny);
    ctx.strokeStyle = C.arm;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    x = nx;
    y = ny;
  });

  // Record trace
  state.tracePoints.push({ x, y });
  const maxTrace = state.phasors.length;
  if (state.tracePoints.length > maxTrace) state.tracePoints.shift();

  // Draw trace
  if (state.tracePoints.length > 1) {
    ctx.beginPath();
    ctx.moveTo(state.tracePoints[0].x, state.tracePoints[0].y);
    for (let i = 1; i < state.tracePoints.length; i++) {
      ctx.lineTo(state.tracePoints[i].x, state.tracePoints[i].y);
    }
    ctx.strokeStyle = C.trace;
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Draw tip dot
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, 2 * Math.PI);
  ctx.fillStyle = C.trace;
  ctx.fill();
}

// ─── 1D Signal rendering ─────────────────────────────────────────────────────

function targetWaveform(t: number, wave: Waveform): number {
  const x = t / (2 * Math.PI);
  if (wave === 'square') return x % 1 < 0.5 ? 1 : -1;
  if (wave === 'sawtooth') return 2 * (x % 1) - 1;
  if (wave === 'triangle') {
    const u = x % 1;
    return u < 0.5 ? 4 * u - 1 : 3 - 4 * u;
  }
  // custom: sum of a few harmonics
  return Math.sin(t) + 0.5 * Math.sin(3 * t) + 0.25 * Math.cos(5 * t);
}

function fourierApprox(t: number, n: number, wave: Waveform): number {
  let sum = 0;
  if (wave === 'square') {
    for (let k = 1; k <= n; k++) {
      const h = 2 * k - 1; // odd harmonics only
      sum += (4 / Math.PI) * (1 / h) * Math.sin(h * t);
    }
  } else if (wave === 'sawtooth') {
    for (let k = 1; k <= n; k++) {
      sum += (2 / Math.PI) * (Math.pow(-1, k + 1) / k) * Math.sin(k * t);
    }
  } else if (wave === 'triangle') {
    for (let k = 1; k <= n; k++) {
      const h = 2 * k - 1;
      sum += (8 / (Math.PI * Math.PI)) * (Math.pow(-1, k + 1) / (h * h)) * Math.sin(h * t);
    }
  } else {
    // custom: finite partial sums
    for (let k = 1; k <= n; k++) {
      sum += (1 / k) * Math.sin(k * t) * Math.cos(k * 0.1);
    }
  }
  return sum;
}

function getSpectrumCoeffs(n: number, wave: Waveform): Array<{ k: number; amp: number }> {
  const coeffs: Array<{ k: number; amp: number }> = [];
  if (wave === 'square') {
    for (let k = 1; k <= n; k++) {
      const h = 2 * k - 1;
      coeffs.push({ k: h, amp: (4 / Math.PI) / h });
    }
  } else if (wave === 'sawtooth') {
    for (let k = 1; k <= n; k++) {
      coeffs.push({ k, amp: (2 / Math.PI) / k });
    }
  } else if (wave === 'triangle') {
    for (let k = 1; k <= n; k++) {
      const h = 2 * k - 1;
      coeffs.push({ k: h, amp: (8 / (Math.PI * Math.PI)) / (h * h) });
    }
  } else {
    for (let k = 1; k <= n; k++) {
      coeffs.push({ k, amp: 1 / k });
    }
  }
  return coeffs;
}

function draw1dSignal(): void {
  const canvas = signal1dCanvas;
  const ctx = ctx1d;
  clearCanvas(ctx, canvas);
  drawGrid(ctx, canvas);

  const W = canvas.width, H = canvas.height;
  const cy = H / 2;
  const scaleY = H * 0.35;
  const points = 600;

  // Draw target waveform
  ctx.beginPath();
  for (let i = 0; i <= points; i++) {
    const t = (i / points) * 4 * Math.PI;
    const y = targetWaveform(t, state.waveform);
    const px = (i / points) * W;
    const py = cy - y * scaleY;
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.strokeStyle = C.target;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Draw Fourier approximation
  ctx.beginPath();
  for (let i = 0; i <= points; i++) {
    const t = (i / points) * 4 * Math.PI;
    const y = fourierApprox(t, state.numHarmonics, state.waveform);
    const px = (i / points) * W;
    const py = cy - y * scaleY;
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.strokeStyle = C.approx;
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // Animated vertical time marker
  const markerX = ((state.signal1dTime % (4 * Math.PI)) / (4 * Math.PI)) * W;
  ctx.beginPath();
  ctx.moveTo(markerX, 0); ctx.lineTo(markerX, H);
  ctx.strokeStyle = 'rgba(244,114,182,0.5)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Legend
  ctx.font = '13px monospace';
  ctx.fillStyle = C.target;
  ctx.fillText('● target', 16, 22);
  ctx.fillStyle = C.approx;
  ctx.fillText('● approx (' + state.numHarmonics + ' harmonics)', 16, 42);
  ctx.fillStyle = C.text;
  ctx.fillText('waveform: ' + state.waveform, W - 180, 22);
}

function drawSpectrum(): void {
  const canvas = spectrumCanvas;
  const ctx = ctxS;
  clearCanvas(ctx, canvas);

  if (!state.showSpectrum) {
    ctx.fillStyle = C.text;
    ctx.font = '14px monospace';
    ctx.fillText('spectrum hidden', canvas.width / 2 - 60, canvas.height / 2);
    return;
  }

  const coeffs = getSpectrumCoeffs(state.numHarmonics, state.waveform);
  if (coeffs.length === 0) return;

  const W = canvas.width, H = canvas.height;
  const maxAmp = Math.max.apply(null, coeffs.map((c) => c.amp));
  const padL = 40, padR = 20, padT = 20, padB = 40;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // Axes
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  // Y axis label
  ctx.fillStyle = C.text;
  ctx.font = '11px monospace';
  ctx.save();
  ctx.translate(12, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('amplitude', -30, 0);
  ctx.restore();
  ctx.fillText('harmonic k', padL + plotW / 2 - 30, H - 4);

  const barW = Math.max(4, plotW / (coeffs.length * 2));

  coeffs.forEach((c, i) => {
    const bx = padL + (i / coeffs.length) * plotW + (plotW / coeffs.length - barW) / 2;
    const bh = (c.amp / maxAmp) * plotH;
    const by = padT + plotH - bh;

    // Gradient fill
    const grad = ctx.createLinearGradient(bx, by, bx, by + bh);
    grad.addColorStop(0, '#818cf8');
    grad.addColorStop(1, C.spectrum);
    ctx.fillStyle = grad;
    ctx.fillRect(bx, by, barW, bh);

    // k label
    ctx.fillStyle = C.text;
    ctx.font = '10px monospace';
    ctx.fillText(String(c.k), bx + barW / 2 - 4, padT + plotH + 16);
  });
}

// ─── Animation loop ───────────────────────────────────────────────────────────

let lastRaf = 0;

function animate(ts: number): void {
  const dt = Math.min((ts - lastRaf) / 1000, 0.05);
  lastRaf = ts;

  if (state.mode === 'epicycles') {
    if (state.phasors.length > 0) {
      const step = state.epicycleSpeed * 2;
      state.epicycleTime += step;
      if (state.epicycleTime >= state.phasors.length) {
        state.epicycleTime = 0;
        state.tracePoints = [];
      }
      drawEpicycles(state.epicycleTime);
    } else {
      clearCanvas(ctxE, epicyclesCanvas);
      drawGrid(ctxE, epicyclesCanvas);
      ctxE.fillStyle = 'rgba(148,163,184,0.6)';
      ctxE.font = '15px monospace';
      ctxE.fillText('Draw a path or pick a preset', epicyclesCanvas.width / 2 - 120, epicyclesCanvas.height / 2);
    }
  } else {
    state.signal1dTime += dt * state.signal1dSpeed * 2;
    draw1dSignal();
    drawSpectrum();
  }

  requestAnimationFrame(animate);
}

// ─── Mouse drawing for epicycles ─────────────────────────────────────────────

function setupDrawing(): void {
  function toComplex(e: MouseEvent): Complex {
    const r = epicyclesCanvas.getBoundingClientRect();
    return {
      re: e.clientX - r.left - epicyclesCanvas.width / 2,
      im: e.clientY - r.top - epicyclesCanvas.height / 2,
    };
  }

  epicyclesCanvas.addEventListener('mousedown', (e) => {
    if (state.preset !== 'draw') return;
    state.drawing = true;
    state.drawnPath = [];
    state.phasors = [];
    state.tracePoints = [];
    state.drawnPath.push(toComplex(e));
  });

  epicyclesCanvas.addEventListener('mousemove', (e) => {
    if (!state.drawing) return;
    state.drawnPath.push(toComplex(e));
    // Preview drawn path
    clearCanvas(ctxE, epicyclesCanvas);
    drawGrid(ctxE, epicyclesCanvas);
    ctxE.beginPath();
    const cx = epicyclesCanvas.width / 2, cy = epicyclesCanvas.height / 2;
    state.drawnPath.forEach((p, i) => {
      i === 0 ? ctxE.moveTo(cx + p.re, cy + p.im) : ctxE.lineTo(cx + p.re, cy + p.im);
    });
    ctxE.strokeStyle = C.trace;
    ctxE.lineWidth = 2;
    ctxE.stroke();
  });

  epicyclesCanvas.addEventListener('mouseup', () => {
    if (!state.drawing) return;
    state.drawing = false;
    if (state.drawnPath.length > 10) preparePhasors(state.drawnPath);
  });
}

// ─── Controls wiring ──────────────────────────────────────────────────────────

function setupControls(): void {
  // Mode selector
  const modeBtns = document.querySelectorAll<HTMLButtonElement>('.mode-btn');
  modeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      state.mode = btn.dataset.mode as Mode;
      modeBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const ePanel = document.getElementById('epicycles-panel')!;
      const sPanel = document.getElementById('signal1d-panel')!;
      if (state.mode === 'epicycles') {
        ePanel.style.display = '';
        sPanel.style.display = 'none';
      } else {
        ePanel.style.display = 'none';
        sPanel.style.display = '';
      }
    });
  });

  // Preset selector
  const presetSel = document.getElementById('preset-select') as HTMLSelectElement;
  presetSel.addEventListener('change', () => {
    state.preset = presetSel.value as Preset;
    if (state.preset !== 'draw') {
      loadPreset(state.preset);
    } else {
      state.drawnPath = [];
      state.phasors = [];
      state.tracePoints = [];
    }
  });

  // Terms slider (epicycles)
  const termsSlider = document.getElementById('terms-slider') as HTMLInputElement;
  const termsLabel = document.getElementById('terms-label')!;
  termsSlider.addEventListener('input', () => {
    state.numTerms = parseInt(termsSlider.value, 10);
    termsLabel.textContent = String(state.numTerms);
    state.tracePoints = [];
    state.epicycleTime = 0;
  });

  // Speed slider (epicycles)
  const speedSlider = document.getElementById('speed-slider') as HTMLInputElement;
  const speedLabel = document.getElementById('speed-label')!;
  speedSlider.addEventListener('input', () => {
    state.epicycleSpeed = parseFloat(speedSlider.value);
    speedLabel.textContent = state.epicycleSpeed.toFixed(1) + 'x';
  });

  // Clear button
  const clearBtn = document.getElementById('clear-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      state.drawnPath = [];
      state.phasors = [];
      state.tracePoints = [];
      state.preset = 'draw';
      presetSel.value = 'draw';
    });
  }

  // Waveform selector
  const waveformSel = document.getElementById('waveform-select') as HTMLSelectElement;
  waveformSel.addEventListener('change', () => {
    state.waveform = waveformSel.value as Waveform;
  });

  // Harmonics slider
  const harmonicsSlider = document.getElementById('harmonics-slider') as HTMLInputElement;
  const harmonicsLabel = document.getElementById('harmonics-label')!;
  harmonicsSlider.addEventListener('input', () => {
    state.numHarmonics = parseInt(harmonicsSlider.value, 10);
    harmonicsLabel.textContent = String(state.numHarmonics);
  });

  // Signal speed slider
  const sig1dSpeedSlider = document.getElementById('signal1d-speed-slider') as HTMLInputElement;
  const sig1dSpeedLabel = document.getElementById('signal1d-speed-label')!;
  sig1dSpeedSlider.addEventListener('input', () => {
    state.signal1dSpeed = parseFloat(sig1dSpeedSlider.value);
    sig1dSpeedLabel.textContent = state.signal1dSpeed.toFixed(1) + 'x';
  });

  // Show spectrum toggle
  const spectrumToggle = document.getElementById('spectrum-toggle') as HTMLInputElement;
  spectrumToggle.addEventListener('change', () => {
    state.showSpectrum = spectrumToggle.checked;
    const sc = document.getElementById('spectrum-container')!;
    sc.style.display = state.showSpectrum ? '' : 'none';
  });
}

// ─── Init ─────────────────────────────────────────────────────────────────────

function init(): void {
  Array.from(document.querySelectorAll('canvas')).forEach((c) => resizeCanvas(c as HTMLCanvasElement));

  window.addEventListener('resize', () => {
    Array.from(document.querySelectorAll('canvas')).forEach((c) => resizeCanvas(c as HTMLCanvasElement));
  });

  setupDrawing();
  setupControls();

  // Load default preset
  loadPreset('square');

  requestAnimationFrame((ts) => { lastRaf = ts; requestAnimationFrame(animate); });
}

document.addEventListener('DOMContentLoaded', init);
