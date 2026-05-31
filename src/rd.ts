import * as d3 from 'd3';

// Gray-Scott reaction-diffusion system
// dU/dt = Du * laplacian(U) - U*V^2 + F*(1-U)
// dV/dt = Dv * laplacian(V) + U*V^2 - (F+k)*V

interface Preset {
  name: string;
  F: number;
  k: number;
  Du: number;
  Dv: number;
}

const PRESETS: Preset[] = [
  { name: 'coral',   F: 0.0545, k: 0.062,  Du: 0.2097, Dv: 0.1050 },
  { name: 'mitosis', F: 0.0367, k: 0.0649, Du: 0.2097, Dv: 0.1050 },
  { name: 'maze',    F: 0.029,  k: 0.057,  Du: 0.2097, Dv: 0.1050 },
  { name: 'spots',   F: 0.025,  k: 0.060,  Du: 0.2097, Dv: 0.1050 },
  { name: 'waves',   F: 0.014,  k: 0.054,  Du: 0.2097, Dv: 0.1050 },
  { name: 'stripes', F: 0.022,  k: 0.051,  Du: 0.2097, Dv: 0.1050 },
];

interface Colormap {
  name: string;
  fn: (t: number) => string;
}

const COLORMAPS: Colormap[] = [
  {
    name: 'plasma',
    fn: (t: number) => {
      const r = Math.round(255 * Math.min(1, Math.max(0, 0.05 + 1.0 * t * t + 0.3 * t)));
      const g = Math.round(255 * Math.min(1, Math.max(0, 0.02 + 0.9 * t * (1 - t) * 2)));
      const b = Math.round(255 * Math.min(1, Math.max(0, 0.53 - 0.5 * t + 0.3 * t * t)));
      return `rgb(${r},${g},${b})`;
    }
  },
  {
    name: 'viridis',
    fn: (t: number) => {
      const r = Math.round(255 * Math.min(1, Math.max(0, 0.267 + 0.004 * t - 1.263 * t * t + 2.17 * t * t * t)));
      const g = Math.round(255 * Math.min(1, Math.max(0, 0.0 + 1.418 * t - 0.5 * t * t)));
      const b = Math.round(255 * Math.min(1, Math.max(0, 0.329 + 1.498 * t - 3.0 * t * t + 2.0 * t * t * t)));
      return `rgb(${r},${g},${b})`;
    }
  },
  {
    name: 'fire',
    fn: (t: number) => {
      const r = Math.round(255 * Math.min(1, Math.max(0, t * 2.5)));
      const g = Math.round(255 * Math.min(1, Math.max(0, t * t * 2.0)));
      const b = Math.round(255 * Math.min(1, Math.max(0, t * t * t * 3.0)));
      return `rgb(${r},${g},${b})`;
    }
  },
  {
    name: 'ice',
    fn: (t: number) => {
      const r = Math.round(255 * Math.min(1, Math.max(0, t * t)));
      const g = Math.round(255 * Math.min(1, Math.max(0, t * 1.2)));
      const b = Math.round(255 * Math.min(1, Math.max(0, 0.2 + t * 0.8)));
      return `rgb(${r},${g},${b})`;
    }
  },
];

// Build an LUT (256 entries) for fast colormap lookup
function buildLUT(cm: Colormap): Uint8Array {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const c = cm.fn(i / 255);
    const m = c.match(/rgb\((\d+),(\d+),(\d+)\)/);
    if (m) {
      lut[i * 3 + 0] = parseInt(m[1]);
      lut[i * 3 + 1] = parseInt(m[2]);
      lut[i * 3 + 2] = parseInt(m[3]);
    }
  }
  return lut;
}

class RDSim {
  W: number;
  H: number;
  U: Float32Array;
  V: Float32Array;
  U2: Float32Array;
  V2: Float32Array;
  F: number;
  k: number;
  Du: number;
  Dv: number;
  dt: number = 1.0;

  constructor(W: number, H: number) {
    this.W = W;
    this.H = H;
    const n = W * H;
    this.U = new Float32Array(n);
    this.V = new Float32Array(n);
    this.U2 = new Float32Array(n);
    this.V2 = new Float32Array(n);
    this.F = PRESETS[0].F;
    this.k = PRESETS[0].k;
    this.Du = PRESETS[0].Du;
    this.Dv = PRESETS[0].Dv;
    this.reset();
  }

  reset() {
    const n = this.W * this.H;
    for (let i = 0; i < n; i++) {
      this.U[i] = 1.0;
      this.V[i] = 0.0;
    }
    // Seed a few random square blobs of V
    const blobCount = 8;
    const blobSize = Math.max(4, Math.floor(Math.min(this.W, this.H) / 12));
    for (let b = 0; b < blobCount; b++) {
      const cx = Math.floor(Math.random() * this.W);
      const cy = Math.floor(Math.random() * this.H);
      for (let dy = -blobSize; dy <= blobSize; dy++) {
        for (let dx = -blobSize; dx <= blobSize; dx++) {
          const nx = (cx + dx + this.W) % this.W;
          const ny = (cy + dy + this.H) % this.H;
          this.U[ny * this.W + nx] = 0.5 + Math.random() * 0.1;
          this.V[ny * this.W + nx] = 0.25 + Math.random() * 0.1;
        }
      }
    }
  }

  // Paint V (and disturb U) at canvas pixel (px, py) -> grid coords
  paint(gx: number, gy: number, radius: number) {
    const r = Math.max(1, Math.round(radius));
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (dx * dx + dy * dy <= r * r) {
          const nx = (gx + dx + this.W) % this.W;
          const ny = (gy + dy + this.H) % this.H;
          const idx = ny * this.W + nx;
          this.U[idx] = 0.5 + Math.random() * 0.1;
          this.V[idx] = 0.25 + Math.random() * 0.1;
        }
      }
    }
  }

  step(substeps: number) {
    const W = this.W, H = this.H;
    const U = this.U, V = this.V;
    const U2 = this.U2, V2 = this.V2;
    const F = this.F, k = this.k;
    const Du = this.Du, Dv = this.Dv;
    const dt = this.dt;

    for (let s = 0; s < substeps; s++) {
      for (let y = 0; y < H; y++) {
        const yn = (y - 1 + H) % H;
        const yp = (y + 1) % H;
        const rowY  = y  * W;
        const rowYn = yn * W;
        const rowYp = yp * W;

        for (let x = 0; x < W; x++) {
          const xn = (x - 1 + W) % W;
          const xp = (x + 1) % W;
          const idx = rowY + x;

          const u = U[idx];
          const v = V[idx];

          // 5-point Laplacian (periodic boundaries)
          const lapU = U[rowY + xn] + U[rowY + xp] + U[rowYn + x] + U[rowYp + x] - 4.0 * u;
          const lapV = V[rowY + xn] + V[rowY + xp] + V[rowYn + x] + V[rowYp + x] - 4.0 * v;

          const uvv = u * v * v;
          U2[idx] = u + dt * (Du * lapU - uvv + F * (1.0 - u));
          V2[idx] = v + dt * (Dv * lapV + uvv - (F + k) * v);
        }
      }
      // Swap buffers
      const tmpU = this.U;
      this.U = this.U2;
      this.U2 = tmpU;
      const tmpV = this.V;
      this.V = this.V2;
      this.V2 = tmpV;
    }
  }
}

// ---- Main app ----

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

// Controls
const presetSel   = document.getElementById('preset')    as HTMLSelectElement;
const colormapSel = document.getElementById('colormap')  as HTMLSelectElement;
const inF         = document.getElementById('inF')       as HTMLInputElement;
const inK         = document.getElementById('inK')       as HTMLInputElement;
const inDu        = document.getElementById('inDu')      as HTMLInputElement;
const inDv        = document.getElementById('inDv')      as HTMLInputElement;
const inSpeed     = document.getElementById('inSpeed')   as HTMLInputElement;
const inBrush     = document.getElementById('inBrush')   as HTMLInputElement;
const inGridSize  = document.getElementById('inGridSize')as HTMLInputElement;
const btnPlay     = document.getElementById('btnPlay')   as HTMLButtonElement;
const btnStep     = document.getElementById('btnStep')   as HTMLButtonElement;
const btnReset    = document.getElementById('btnReset')  as HTMLButtonElement;
const lblF        = document.getElementById('lblF')      as HTMLSpanElement;
const lblK        = document.getElementById('lblK')      as HTMLSpanElement;
const lblDu       = document.getElementById('lblDu')     as HTMLSpanElement;
const lblDv       = document.getElementById('lblDv')     as HTMLSpanElement;
const lblSpeed    = document.getElementById('lblSpeed')  as HTMLSpanElement;
const lblBrush    = document.getElementById('lblBrush')  as HTMLSpanElement;
const lblGridSize = document.getElementById('lblGridSize')as HTMLSpanElement;
const fpsEl       = document.getElementById('fps')       as HTMLSpanElement;

// Populate preset select
PRESETS.forEach((p, i) => {
  const opt = document.createElement('option');
  opt.value = String(i);
  opt.textContent = p.name;
  presetSel.appendChild(opt);
});

// Populate colormap select
COLORMAPS.forEach((c, i) => {
  const opt = document.createElement('option');
  opt.value = String(i);
  opt.textContent = c.name;
  colormapSel.appendChild(opt);
});

let gridSize = parseInt(inGridSize.value);
let sim = new RDSim(gridSize, gridSize);
let playing = true;
let substeps = parseInt(inSpeed.value);
let brushRadius = parseInt(inBrush.value);
let cmIndex = 0;
let lut = buildLUT(COLORMAPS[0]);

let imageData = ctx.createImageData(gridSize, gridSize);

function syncParamsFromSim() {
  inF.value  = String(sim.F);
  inK.value  = String(sim.k);
  inDu.value = String(sim.Du);
  inDv.value = String(sim.Dv);
  lblF.textContent  = sim.F.toFixed(4);
  lblK.textContent  = sim.k.toFixed(4);
  lblDu.textContent = sim.Du.toFixed(4);
  lblDv.textContent = sim.Dv.toFixed(4);
}
syncParamsFromSim();

function applyPreset(idx: number) {
  const p = PRESETS[idx];
  sim.F = p.F; sim.k = p.k; sim.Du = p.Du; sim.Dv = p.Dv;
  syncParamsFromSim();
}

presetSel.addEventListener('change', () => {
  applyPreset(parseInt(presetSel.value));
});

colormapSel.addEventListener('change', () => {
  cmIndex = parseInt(colormapSel.value);
  lut = buildLUT(COLORMAPS[cmIndex]);
});

inF.addEventListener('input', () => { sim.F = parseFloat(inF.value); lblF.textContent = sim.F.toFixed(4); });
inK.addEventListener('input', () => { sim.k = parseFloat(inK.value); lblK.textContent = sim.k.toFixed(4); });
inDu.addEventListener('input', () => { sim.Du = parseFloat(inDu.value); lblDu.textContent = sim.Du.toFixed(4); });
inDv.addEventListener('input', () => { sim.Dv = parseFloat(inDv.value); lblDv.textContent = sim.Dv.toFixed(4); });

inSpeed.addEventListener('input', () => {
  substeps = parseInt(inSpeed.value);
  lblSpeed.textContent = String(substeps);
});
inBrush.addEventListener('input', () => {
  brushRadius = parseInt(inBrush.value);
  lblBrush.textContent = String(brushRadius);
});

inGridSize.addEventListener('change', () => {
  gridSize = parseInt(inGridSize.value);
  lblGridSize.textContent = String(gridSize);
  const newSim = new RDSim(gridSize, gridSize);
  newSim.F = sim.F; newSim.k = sim.k; newSim.Du = sim.Du; newSim.Dv = sim.Dv;
  sim = newSim;
  imageData = ctx.createImageData(gridSize, gridSize);
});

btnPlay.addEventListener('click', () => {
  playing = !playing;
  btnPlay.textContent = playing ? 'Pause' : 'Play';
});

btnStep.addEventListener('click', () => {
  playing = false;
  btnPlay.textContent = 'Play';
  sim.step(1);
  render();
});

btnReset.addEventListener('click', () => {
  sim.reset();
  render();
});

// Mouse drawing
let mouseDown = false;

function canvasToGrid(cx: number, cy: number): [number, number] {
  const rect = canvas.getBoundingClientRect();
  const px = cx - rect.left;
  const py = cy - rect.top;
  const gx = Math.floor((px / rect.width)  * gridSize);
  const gy = Math.floor((py / rect.height) * gridSize);
  return [gx, gy];
}

canvas.addEventListener('mousedown', (e) => {
  mouseDown = true;
  const [gx, gy] = canvasToGrid(e.clientX, e.clientY);
  sim.paint(gx, gy, brushRadius);
});
canvas.addEventListener('mousemove', (e) => {
  if (!mouseDown) return;
  const [gx, gy] = canvasToGrid(e.clientX, e.clientY);
  sim.paint(gx, gy, brushRadius);
});
canvas.addEventListener('mouseup', () => { mouseDown = false; });
canvas.addEventListener('mouseleave', () => { mouseDown = false; });

canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  mouseDown = true;
  const t = e.touches[0];
  const [gx, gy] = canvasToGrid(t.clientX, t.clientY);
  sim.paint(gx, gy, brushRadius);
}, { passive: false });
canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  const t = e.touches[0];
  const [gx, gy] = canvasToGrid(t.clientX, t.clientY);
  sim.paint(gx, gy, brushRadius);
}, { passive: false });
canvas.addEventListener('touchend', () => { mouseDown = false; });

function render() {
  const V = sim.V;
  const W = sim.W, H = sim.H;
  const data = imageData.data;
  for (let i = 0; i < W * H; i++) {
    const v = V[i];
    const idx8 = Math.min(255, Math.max(0, Math.round(v * 255 * 3.5)));
    const li = idx8 * 3;
    const d = i * 4;
    data[d + 0] = lut[li + 0];
    data[d + 1] = lut[li + 1];
    data[d + 2] = lut[li + 2];
    data[d + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

// Resize canvas to match CSS size
function resizeCanvas() {
  canvas.width  = gridSize;
  canvas.height = gridSize;
}
resizeCanvas();

// Animation loop
let lastTime = 0;
let frameCount = 0;
let fpsTime = 0;

function loop(ts: number) {
  const dt = ts - lastTime;
  lastTime = ts;
  fpsTime += dt;
  frameCount++;
  if (fpsTime >= 500) {
    fpsEl.textContent = (frameCount / (fpsTime / 1000)).toFixed(1);
    fpsTime = 0;
    frameCount = 0;
  }

  if (playing) {
    sim.step(substeps);
  }
  render();
  requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
