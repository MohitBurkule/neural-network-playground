import * as d3 from 'd3';

// ---------------------------------------------------------------------------
// Hopfield Network — binary ±1, Hebbian learning, async/sync recall
// ---------------------------------------------------------------------------

type Pattern = number[]; // flat array of ±1, length N*N

// ---------------------------------------------------------------------------
// Built-in glyphs (10×10)
// ---------------------------------------------------------------------------

function glyph(rows: string[]): Pattern {
  const pat: Pattern = [];
  rows.forEach(row => {
    for (let c = 0; c < row.length; c++) {
      pat.push(row[c] === '#' ? 1 : -1);
    }
  });
  return pat;
}

const GLYPHS: {name: string; pat: Pattern}[] = [
  {
    name: 'Letter A',
    pat: glyph([
      '    ##    ',
      '   ####   ',
      '  ##  ##  ',
      ' ##    ## ',
      ' ######## ',
      ' ######## ',
      ' ##    ## ',
      ' ##    ## ',
      ' ##    ## ',
      ' ##    ## ',
    ]),
  },
  {
    name: 'Letter T',
    pat: glyph([
      ' ######## ',
      ' ######## ',
      '    ##    ',
      '    ##    ',
      '    ##    ',
      '    ##    ',
      '    ##    ',
      '    ##    ',
      '    ##    ',
      '    ##    ',
    ]),
  },
  {
    name: 'Smiley',
    pat: glyph([
      '  ######  ',
      ' ##    ## ',
      '##  ##  ##',
      '##  ##  ##',
      '##      ##',
      '## #### ##',
      '##  ##  ##',
      ' ##    ## ',
      '  ######  ',
      '          ',
    ]),
  },
  {
    name: 'Cross',
    pat: glyph([
      '    ##    ',
      '    ##    ',
      '    ##    ',
      '    ##    ',
      '##########',
      '##########',
      '    ##    ',
      '    ##    ',
      '    ##    ',
      '    ##    ',
    ]),
  },
  {
    name: 'Diamond',
    pat: glyph([
      '    ##    ',
      '   ####   ',
      '  ######  ',
      ' ######## ',
      '##########',
      '##########',
      ' ######## ',
      '  ######  ',
      '   ####   ',
      '    ##    ',
    ]),
  },
];

// ---------------------------------------------------------------------------
// Hopfield Network core
// ---------------------------------------------------------------------------

class HopfieldNet {
  N: number;          // total neurons = gridN * gridN
  W: number[][];      // weight matrix N×N
  patterns: Pattern[];

  constructor(N: number) {
    this.N = N;
    this.W = Array.from({length: N}, () => new Array(N).fill(0));
    this.patterns = [];
  }

  /** Hebbian outer-product learning (replace current weights) */
  train(patterns: Pattern[]): void {
    this.patterns = patterns.slice();
    const N = this.N;
    // reset weights
    this.W = Array.from({length: N}, () => new Array(N).fill(0));
    patterns.forEach(p => {
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (i !== j) {
            this.W[i][j] += p[i] * p[j];
          }
        }
      }
    });
    // normalise by number of neurons
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        this.W[i][j] /= N;
      }
    }
  }

  /** Compute energy E = -½ Σᵢⱼ wᵢⱼ sᵢ sⱼ */
  energy(state: Pattern): number {
    const N = this.N;
    let e = 0;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        e -= this.W[i][j] * state[i] * state[j];
      }
    }
    return 0.5 * e;
  }

  /** Single asynchronous update: pick neuron index and update */
  asyncStep(state: Pattern, neuronIdx: number): boolean {
    const N = this.N;
    let net = 0;
    for (let j = 0; j < N; j++) {
      net += this.W[neuronIdx][j] * state[j];
    }
    const newVal = net >= 0 ? 1 : -1;
    const changed = newVal !== state[neuronIdx];
    state[neuronIdx] = newVal;
    return changed;
  }

  /** Full synchronous sweep: update all neurons simultaneously */
  syncStep(state: Pattern): boolean {
    const N = this.N;
    const next = new Array(N) as Pattern;
    for (let i = 0; i < N; i++) {
      let net = 0;
      for (let j = 0; j < N; j++) {
        net += this.W[i][j] * state[j];
      }
      next[i] = net >= 0 ? 1 : -1;
    }
    let changed = false;
    for (let i = 0; i < N; i++) {
      if (next[i] !== state[i]) changed = true;
      state[i] = next[i];
    }
    return changed;
  }

  /** Compute all async updates in a random order (one epoch) */
  asyncEpoch(state: Pattern): boolean {
    const N = this.N;
    // shuffle indices
    const order = Array.from({length: N}, (_, i) => i);
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }
    let anyChanged = false;
    order.forEach(idx => {
      if (this.asyncStep(state, idx)) anyChanged = true;
    });
    return anyChanged;
  }
}

// ---------------------------------------------------------------------------
// UI state
// ---------------------------------------------------------------------------

let GRID_N = 10;
let net = new HopfieldNet(GRID_N * GRID_N);
let storedPatterns: Pattern[] = [];
let currentState: Pattern = new Array(GRID_N * GRID_N).fill(-1);
let energyHistory: number[] = [];
let animFrameId: number | null = null;
let isRunning = false;
let useSync = false;

// ---------------------------------------------------------------------------
// Canvas / layout references
// ---------------------------------------------------------------------------

let gridCanvas: HTMLCanvasElement;
let energyCanvas: HTMLCanvasElement;
let weightCanvas: HTMLCanvasElement;
let thumbnailsDiv: HTMLElement;

const CELL_PX = 36;
const THUMB_PX = 18;

// ---------------------------------------------------------------------------
// Draw helpers
// ---------------------------------------------------------------------------

function drawGrid(canvas: HTMLCanvasElement, state: Pattern, gridN: number, cellPx: number): void {
  const ctx = canvas.getContext('2d')!;
  const size = gridN * cellPx;
  canvas.width = size;
  canvas.height = size;
  for (let r = 0; r < gridN; r++) {
    for (let c = 0; c < gridN; c++) {
      const val = state[r * gridN + c];
      ctx.fillStyle = val === 1 ? '#c4b5fd' : '#0f0f1e';
      ctx.fillRect(c * cellPx, r * cellPx, cellPx, cellPx);
      ctx.strokeStyle = '#2a2a4a';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(c * cellPx, r * cellPx, cellPx, cellPx);
    }
  }
}

function drawThumb(canvas: HTMLCanvasElement, state: Pattern, gridN: number, cellPx: number): void {
  const ctx = canvas.getContext('2d')!;
  const size = gridN * cellPx;
  canvas.width = size;
  canvas.height = size;
  for (let r = 0; r < gridN; r++) {
    for (let c = 0; c < gridN; c++) {
      const val = state[r * gridN + c];
      ctx.fillStyle = val === 1 ? '#c4b5fd' : '#0f0f1e';
      ctx.fillRect(c * cellPx, r * cellPx, cellPx, cellPx);
    }
  }
}

function drawEnergyChart(): void {
  const canvas = energyCanvas;
  const W = canvas.clientWidth || 340;
  const H = canvas.clientHeight || 140;
  canvas.width = W;
  canvas.height = H;

  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#0d0d1f';
  ctx.fillRect(0, 0, W, H);

  if (energyHistory.length < 2) {
    ctx.fillStyle = '#444466';
    ctx.font = '12px monospace';
    ctx.fillText('Run recall to see energy curve', 16, H / 2);
    return;
  }

  const pad = {top: 12, right: 12, bottom: 28, left: 52};
  const pw = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  const xScale = d3.scaleLinear().domain([0, energyHistory.length - 1]).range([0, pw]);
  const yMin = d3.min(energyHistory) as number;
  const yMax = d3.max(energyHistory) as number;
  const yPad = Math.abs(yMax - yMin) * 0.1 || 1;
  const yScale = d3.scaleLinear().domain([yMin - yPad, yMax + yPad]).range([ph, 0]);

  // grid lines
  ctx.strokeStyle = '#1e1e3a';
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach(t => {
    const y = pad.top + ph * t;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + pw, y);
    ctx.stroke();
  });

  // energy line
  ctx.strokeStyle = '#7c3aed';
  ctx.lineWidth = 2;
  ctx.beginPath();
  energyHistory.forEach((e, i) => {
    const x = pad.left + xScale(i);
    const y = pad.top + yScale(e);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // axes labels
  ctx.fillStyle = '#7070a0';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('E', 4, pad.top + 4);
  ctx.textAlign = 'center';
  ctx.fillText('Iterations', pad.left + pw / 2, H - 4);

  // y tick values
  ctx.textAlign = 'right';
  [0, 0.5, 1].forEach(t => {
    const val = yMin - yPad + (yMax + yPad - (yMin - yPad)) * (1 - t);
    const y = pad.top + ph * t;
    ctx.fillText(val.toFixed(0), pad.left - 4, y + 4);
  });
}

function drawWeightMatrix(): void {
  const canvas = weightCanvas;
  if (!canvas) return;
  const N = net.N;
  const maxDim = 120;
  const cell = Math.max(1, Math.floor(maxDim / N));
  const size = N * cell;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  // find max abs weight
  let maxW = 0.001;
  net.W.forEach(row => row.forEach(w => { if (Math.abs(w) > maxW) maxW = Math.abs(w); }));

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const v = net.W[i][j] / maxW; // -1 to 1
      let r: number, g: number, b: number;
      if (v > 0) {
        r = Math.round(124 + v * 131); g = Math.round(58 + v * 0); b = Math.round(237 - v * 0);
      } else {
        const u = -v;
        r = Math.round(252 - u * 252); g = Math.round(165 - u * 165); b = Math.round(0);
      }
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(j * cell, i * cell, cell, cell);
    }
  }
}

function refreshThumbnails(): void {
  thumbnailsDiv.innerHTML = '';
  storedPatterns.forEach((p, idx) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'thumb-wrapper';
    wrapper.title = `Pattern ${idx + 1}`;

    const canvas = document.createElement('canvas');
    drawThumb(canvas, p, GRID_N, THUMB_PX);

    const label = document.createElement('div');
    label.className = 'thumb-label';
    label.textContent = `P${idx + 1}`;

    const loadBtn = document.createElement('button');
    loadBtn.className = 'thumb-load-btn';
    loadBtn.textContent = 'Load';
    loadBtn.addEventListener('click', () => {
      stopAnimation();
      currentState = p.slice();
      energyHistory = [net.energy(currentState)];
      drawGrid(gridCanvas, currentState, GRID_N, CELL_PX);
      drawEnergyChart();
    });

    const delBtn = document.createElement('button');
    delBtn.className = 'thumb-del-btn';
    delBtn.textContent = '✕';
    delBtn.addEventListener('click', () => {
      stopAnimation();
      storedPatterns.splice(idx, 1);
      net.train(storedPatterns);
      refreshThumbnails();
      drawWeightMatrix();
      updateStatus(`Pattern ${idx + 1} removed. ${storedPatterns.length} stored.`);
    });

    wrapper.appendChild(canvas);
    wrapper.appendChild(label);
    wrapper.appendChild(loadBtn);
    wrapper.appendChild(delBtn);
    thumbnailsDiv.appendChild(wrapper);
  });

  if (storedPatterns.length === 0) {
    const msg = document.createElement('div');
    msg.style.color = '#444466';
    msg.style.fontSize = '0.78rem';
    msg.style.padding = '8px';
    msg.textContent = 'No patterns stored.';
    thumbnailsDiv.appendChild(msg);
  }
}

function updateStatus(msg: string): void {
  const el = document.getElementById('status');
  if (el) el.textContent = msg;
}

// ---------------------------------------------------------------------------
// Recall animation
// ---------------------------------------------------------------------------

function stopAnimation(): void {
  if (animFrameId !== null) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  isRunning = false;
  const btn = document.getElementById('runBtn') as HTMLButtonElement;
  if (btn) btn.textContent = 'Run Recall';
}

function doStep(): boolean {
  if (storedPatterns.length === 0) {
    updateStatus('Store at least one pattern first.');
    return false;
  }
  let changed: boolean;
  if (useSync) {
    changed = net.syncStep(currentState);
  } else {
    changed = net.asyncEpoch(currentState);
  }
  const e = net.energy(currentState);
  energyHistory.push(e);
  drawGrid(gridCanvas, currentState, GRID_N, CELL_PX);
  drawEnergyChart();
  updateStatus(`Iter ${energyHistory.length - 1} — Energy: ${e.toFixed(2)} — ${changed ? 'changed' : 'converged'}`);
  return changed;
}

function animate(): void {
  if (!isRunning) return;
  const changed = doStep();
  if (!changed) {
    stopAnimation();
    updateStatus(`Converged after ${energyHistory.length - 1} iterations. Energy: ${energyHistory[energyHistory.length - 1].toFixed(2)}`);
    return;
  }
  animFrameId = requestAnimationFrame(animate);
}

// ---------------------------------------------------------------------------
// Grid mouse interaction
// ---------------------------------------------------------------------------

function attachGridInteraction(): void {
  let drawing = false;
  let drawValue = 1;

  function cellFromEvent(e: MouseEvent): number {
    const rect = gridCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor(x / CELL_PX);
    const row = Math.floor(y / CELL_PX);
    if (col < 0 || col >= GRID_N || row < 0 || row >= GRID_N) return -1;
    return row * GRID_N + col;
  }

  gridCanvas.addEventListener('mousedown', e => {
    stopAnimation();
    const idx = cellFromEvent(e);
    if (idx < 0) return;
    drawValue = currentState[idx] === 1 ? -1 : 1;
    currentState[idx] = drawValue;
    drawing = true;
    drawGrid(gridCanvas, currentState, GRID_N, CELL_PX);
  });

  gridCanvas.addEventListener('mousemove', e => {
    if (!drawing) return;
    const idx = cellFromEvent(e);
    if (idx < 0) return;
    currentState[idx] = drawValue;
    drawGrid(gridCanvas, currentState, GRID_N, CELL_PX);
  });

  window.addEventListener('mouseup', () => { drawing = false; });
}

// ---------------------------------------------------------------------------
// Corruption
// ---------------------------------------------------------------------------

function corruptState(noiseLevel: number): void {
  stopAnimation();
  for (let i = 0; i < currentState.length; i++) {
    if (Math.random() < noiseLevel) {
      currentState[i] = -currentState[i] as (1 | -1);
    }
  }
  energyHistory = [net.energy(currentState)];
  drawGrid(gridCanvas, currentState, GRID_N, CELL_PX);
  drawEnergyChart();
  updateStatus(`Corrupted with ${Math.round(noiseLevel * 100)}% noise. Energy: ${energyHistory[0].toFixed(2)}`);
}

// ---------------------------------------------------------------------------
// Grid size resize
// ---------------------------------------------------------------------------

function resizeGrid(newN: number): void {
  stopAnimation();
  GRID_N = newN;
  net = new HopfieldNet(GRID_N * GRID_N);
  storedPatterns = [];
  currentState = new Array(GRID_N * GRID_N).fill(-1);
  energyHistory = [];

  gridCanvas.width = GRID_N * CELL_PX;
  gridCanvas.height = GRID_N * CELL_PX;

  drawGrid(gridCanvas, currentState, GRID_N, CELL_PX);
  drawWeightMatrix();
  drawEnergyChart();
  refreshThumbnails();
  updateStatus(`Grid resized to ${newN}×${newN}. Patterns cleared.`);
}

// ---------------------------------------------------------------------------
// Scale built-in glyph to current grid size
// ---------------------------------------------------------------------------

function scaleGlyph(srcPat: Pattern, srcN: number, dstN: number): Pattern {
  const out: Pattern = new Array(dstN * dstN).fill(-1);
  for (let r = 0; r < dstN; r++) {
    for (let c = 0; c < dstN; c++) {
      const sr = Math.floor(r * srcN / dstN);
      const sc = Math.floor(c * srcN / dstN);
      out[r * dstN + c] = srcPat[sr * srcN + sc];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

function init(): void {
  gridCanvas = document.getElementById('gridCanvas') as HTMLCanvasElement;
  energyCanvas = document.getElementById('energyCanvas') as HTMLCanvasElement;
  weightCanvas = document.getElementById('weightCanvas') as HTMLCanvasElement;
  thumbnailsDiv = document.getElementById('thumbnails') as HTMLElement;

  gridCanvas.width = GRID_N * CELL_PX;
  gridCanvas.height = GRID_N * CELL_PX;

  drawGrid(gridCanvas, currentState, GRID_N, CELL_PX);
  drawEnergyChart();
  drawWeightMatrix();
  refreshThumbnails();
  attachGridInteraction();

  // ---- controls ----

  // Noise slider
  const noiseSlider = document.getElementById('noiseSlider') as HTMLInputElement;
  const noiseVal = document.getElementById('noiseVal') as HTMLSpanElement;
  noiseSlider.addEventListener('input', () => {
    noiseVal.textContent = `${Math.round(parseFloat(noiseSlider.value) * 100)}%`;
  });

  // Grid size select
  const gridSizeSelect = document.getElementById('gridSizeSelect') as HTMLSelectElement;
  gridSizeSelect.addEventListener('change', () => {
    resizeGrid(parseInt(gridSizeSelect.value, 10));
  });

  // Sync mode toggle
  const syncToggle = document.getElementById('syncToggle') as HTMLInputElement;
  syncToggle.addEventListener('change', () => {
    useSync = syncToggle.checked;
  });

  // Store pattern button
  const storeBtn = document.getElementById('storeBtn') as HTMLButtonElement;
  storeBtn.addEventListener('click', () => {
    if (storedPatterns.length >= 20) {
      updateStatus('Max 20 patterns.');
      return;
    }
    storedPatterns.push(currentState.slice());
    net.train(storedPatterns);
    refreshThumbnails();
    drawWeightMatrix();
    updateStatus(`Stored pattern ${storedPatterns.length}. Capacity ~${Math.floor(0.138 * GRID_N * GRID_N)} patterns.`);
  });

  // Load glyph
  const glyphSelect = document.getElementById('glyphSelect') as HTMLSelectElement;
  const loadGlyphBtn = document.getElementById('loadGlyphBtn') as HTMLButtonElement;
  loadGlyphBtn.addEventListener('click', () => {
    const idx = parseInt(glyphSelect.value, 10);
    const g = GLYPHS[idx];
    currentState = scaleGlyph(g.pat, 10, GRID_N);
    energyHistory = storedPatterns.length > 0 ? [net.energy(currentState)] : [];
    drawGrid(gridCanvas, currentState, GRID_N, CELL_PX);
    drawEnergyChart();
    updateStatus(`Loaded "${g.name}".`);
  });

  // Clear grid
  const clearBtn = document.getElementById('clearBtn') as HTMLButtonElement;
  clearBtn.addEventListener('click', () => {
    stopAnimation();
    currentState = new Array(GRID_N * GRID_N).fill(-1);
    energyHistory = [];
    drawGrid(gridCanvas, currentState, GRID_N, CELL_PX);
    drawEnergyChart();
    updateStatus('Grid cleared.');
  });

  // Invert
  const invertBtn = document.getElementById('invertBtn') as HTMLButtonElement;
  invertBtn.addEventListener('click', () => {
    stopAnimation();
    currentState = currentState.map(v => -v) as Pattern;
    drawGrid(gridCanvas, currentState, GRID_N, CELL_PX);
    updateStatus('Grid inverted.');
  });

  // Corrupt
  const corruptBtn = document.getElementById('corruptBtn') as HTMLButtonElement;
  corruptBtn.addEventListener('click', () => {
    corruptState(parseFloat(noiseSlider.value));
  });

  // Run / pause
  const runBtn = document.getElementById('runBtn') as HTMLButtonElement;
  runBtn.addEventListener('click', () => {
    if (isRunning) {
      stopAnimation();
    } else {
      if (storedPatterns.length === 0) {
        updateStatus('Store at least one pattern first.');
        return;
      }
      isRunning = true;
      runBtn.textContent = 'Pause';
      animate();
    }
  });

  // Step
  const stepBtn = document.getElementById('stepBtn') as HTMLButtonElement;
  stepBtn.addEventListener('click', () => {
    if (isRunning) stopAnimation();
    doStep();
  });

  // Reset recall (keep stored patterns, reset current to blank)
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
  resetBtn.addEventListener('click', () => {
    stopAnimation();
    currentState = new Array(GRID_N * GRID_N).fill(-1);
    energyHistory = [];
    drawGrid(gridCanvas, currentState, GRID_N, CELL_PX);
    drawEnergyChart();
    updateStatus('Current state reset.');
  });

  // Fill random
  const randomBtn = document.getElementById('randomBtn') as HTMLButtonElement;
  randomBtn.addEventListener('click', () => {
    stopAnimation();
    currentState = currentState.map(() => (Math.random() < 0.5 ? 1 : -1)) as Pattern;
    energyHistory = storedPatterns.length > 0 ? [net.energy(currentState)] : [];
    drawGrid(gridCanvas, currentState, GRID_N, CELL_PX);
    drawEnergyChart();
    updateStatus('Random state generated.');
  });

  // Store all glyphs convenience button
  const storeAllBtn = document.getElementById('storeAllBtn') as HTMLButtonElement;
  storeAllBtn.addEventListener('click', () => {
    storedPatterns = [];
    const subset = GLYPHS.slice(0, 4);
    subset.forEach(g => {
      storedPatterns.push(scaleGlyph(g.pat, 10, GRID_N));
    });
    net.train(storedPatterns);
    refreshThumbnails();
    drawWeightMatrix();
    updateStatus(`Stored ${storedPatterns.length} built-in patterns. Capacity ~${Math.floor(0.138 * GRID_N * GRID_N)}.`);
  });
}

window.addEventListener('DOMContentLoaded', init);
