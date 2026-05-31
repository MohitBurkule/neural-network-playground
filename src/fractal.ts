// Fractal Explorer — Mandelbrot & Julia sets with smooth escape-time coloring
// Canvas/ImageData rendering, no external libraries.
export {};

// ─── Types ────────────────────────────────────────────────────────────────────

interface ViewState {
  cx: number;   // center x (real)
  cy: number;   // center y (imag)
  zoom: number; // pixels per unit
  maxIter: number;
}

type PaletteName = 'ultra' | 'fire' | 'ice' | 'gold' | 'electric';
type Mode = 'mandelbrot' | 'julia' | 'linked';

// ─── Palettes ─────────────────────────────────────────────────────────────────

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpColor(c1: [number,number,number], c2: [number,number,number], t: number): [number,number,number] {
  return [
    Math.round(lerp(c1[0], c2[0], t)),
    Math.round(lerp(c1[1], c2[1], t)),
    Math.round(lerp(c1[2], c2[2], t))
  ];
}

const PALETTES: Record<PaletteName, Array<[number,number,number]>> = {
  ultra: [
    [0,7,100],[32,107,203],[237,255,255],[255,170,0],[0,2,0]
  ],
  fire: [
    [0,0,0],[128,0,0],[255,64,0],[255,200,0],[255,255,200],[255,255,255]
  ],
  ice: [
    [0,0,30],[0,50,120],[0,160,220],[180,230,255],[255,255,255]
  ],
  gold: [
    [10,5,0],[80,40,0],[200,130,0],[255,220,80],[255,255,200],[255,255,255]
  ],
  electric: [
    [0,0,0],[20,0,60],[100,0,200],[0,150,255],[0,255,200],[255,255,255]
  ]
};

function paletteColor(t: number, name: PaletteName): [number,number,number] {
  const stops = PALETTES[name];
  const n = stops.length - 1;
  const scaled = t * n;
  const idx = Math.min(Math.floor(scaled), n - 1);
  const frac = scaled - idx;
  return lerpColor(stops[idx], stops[idx + 1], frac);
}

// ─── Escape time with smooth coloring ────────────────────────────────────────

function mandelbrotIter(cr: number, ci: number, maxIter: number): number {
  let zr = 0, zi = 0, zr2 = 0, zi2 = 0;
  for (let i = 0; i < maxIter; i++) {
    zi = 2 * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    zr2 = zr * zr;
    zi2 = zi * zi;
    if (zr2 + zi2 > 256) {
      // Smooth iteration count
      const log2abs = Math.log(zr2 + zi2) * 0.5 / Math.LN2;
      return i + 1 - Math.log(log2abs) / Math.LN2;
    }
  }
  return -1; // inside
}

function juliaIter(zr: number, zi: number, cr: number, ci: number, maxIter: number): number {
  let zr2 = zr * zr, zi2 = zi * zi;
  for (let i = 0; i < maxIter; i++) {
    zi = 2 * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    zr2 = zr * zr;
    zi2 = zi * zi;
    if (zr2 + zi2 > 256) {
      const log2abs = Math.log(zr2 + zi2) * 0.5 / Math.LN2;
      return i + 1 - Math.log(log2abs) / Math.LN2;
    }
  }
  return -1;
}

// ─── Rendering ────────────────────────────────────────────────────────────────

function renderFractal(
  canvas: HTMLCanvasElement,
  view: ViewState,
  palette: PaletteName,
  isJulia: boolean,
  juliaC: [number, number],
  scale: number = 1
): void {
  const w = Math.floor(canvas.width * scale);
  const h = Math.floor(canvas.height * scale);
  const ctx = canvas.getContext('2d')!;

  const buf = new Uint8ClampedArray(w * h * 4);
  const { cx, cy, zoom, maxIter } = view;
  const zs = zoom * scale;

  for (let py = 0; py < h; py++) {
    const ci = cy + (py - h / 2) / zs;
    for (let px = 0; px < w; px++) {
      const cr = cx + (px - w / 2) / zs;
      let t: number;
      if (isJulia) {
        t = juliaIter(cr, ci, juliaC[0], juliaC[1], maxIter);
      } else {
        t = mandelbrotIter(cr, ci, maxIter);
      }
      const idx = (py * w + px) * 4;
      if (t < 0) {
        buf[idx] = 0; buf[idx+1] = 0; buf[idx+2] = 0; buf[idx+3] = 255;
      } else {
        const nt = (t % maxIter) / maxIter;
        const smoothT = Math.pow(nt, 0.5);
        const [r, g, b] = paletteColor(smoothT, palette);
        buf[idx] = r; buf[idx+1] = g; buf[idx+2] = b; buf[idx+3] = 255;
      }
    }
  }

  const imgData = new ImageData(buf, w, h);
  if (scale === 1) {
    ctx.putImageData(imgData, 0, 0);
  } else {
    // Draw scaled up
    const offscreen = document.createElement('canvas');
    offscreen.width = w;
    offscreen.height = h;
    offscreen.getContext('2d')!.putImageData(imgData, 0, 0);
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
    ctx.restore();
  }
}

// ─── State ────────────────────────────────────────────────────────────────────

const DEFAULT_MANDELBROT: ViewState = { cx: -0.5, cy: 0, zoom: 200, maxIter: 100 };
const DEFAULT_JULIA: ViewState = { cx: 0, cy: 0, zoom: 200, maxIter: 100 };

// ─── App ──────────────────────────────────────────────────────────────────────

function autoIter(zoom: number): number {
  return Math.min(1000, Math.max(100, Math.floor(50 + Math.log2(zoom) * 15)));
}

function initApp(): void {
  const modeSelect = document.getElementById('mode-select') as HTMLSelectElement;
  const paletteSelect = document.getElementById('palette-select') as HTMLSelectElement;
  const maxIterInput = document.getElementById('max-iter') as HTMLInputElement;
  const maxIterDisplay = document.getElementById('max-iter-display') as HTMLSpanElement;
  const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;
  const infoDiv = document.getElementById('info') as HTMLDivElement;
  const juliaControls = document.getElementById('julia-controls') as HTMLDivElement;
  const cRealInput = document.getElementById('c-real') as HTMLInputElement;
  const cImagInput = document.getElementById('c-imag') as HTMLInputElement;
  const cRealDisplay = document.getElementById('c-real-display') as HTMLSpanElement;
  const cImagDisplay = document.getElementById('c-imag-display') as HTMLSpanElement;
  const mainCanvas = document.getElementById('main-canvas') as HTMLCanvasElement;
  const pickerCanvas = document.getElementById('picker-canvas') as HTMLCanvasElement;
  const pickerContainer = document.getElementById('picker-container') as HTMLDivElement;

  let mode: Mode = 'mandelbrot';
  let palette: PaletteName = 'ultra';
  let mainView: ViewState = { ...DEFAULT_MANDELBROT };
  let juliaView: ViewState = { ...DEFAULT_JULIA };
  let juliaC: [number, number] = [-0.7, 0.27];
  let pickerView: ViewState = { ...DEFAULT_MANDELBROT };

  let renderTimeout: number | null = null;
  let isDragging = false;
  let dragStart: { x: number; y: number; cx: number; cy: number } | null = null;
  let isBoxSelecting = false;
  let boxStart: { x: number; y: number } | null = null;
  let overlayCanvas: HTMLCanvasElement | null = null;

  function getMode(): Mode { return mode; }

  function scheduleRender(fast = false): void {
    if (renderTimeout !== null) clearTimeout(renderTimeout);
    if (fast) {
      doRender(0.25);
      renderTimeout = window.setTimeout(() => {
        doRender(0.5);
        renderTimeout = window.setTimeout(() => doRender(1), 150);
      }, 80);
    } else {
      doRender(1);
    }
  }

  function doRender(scale: number): void {
    const m = getMode();
    if (m === 'mandelbrot') {
      renderFractal(mainCanvas, mainView, palette, false, juliaC, scale);
    } else if (m === 'julia') {
      renderFractal(mainCanvas, mainView, palette, true, juliaC, scale);
    } else {
      // linked: main = mandelbrot, picker shows julia
      renderFractal(mainCanvas, mainView, palette, false, juliaC, scale);
      renderFractal(pickerCanvas, juliaView, palette, true, juliaC, scale);
    }
    updateInfo();
  }

  function updateInfo(): void {
    const v = (mode === 'julia') ? mainView : mainView;
    infoDiv.textContent =
      `center: (${v.cx.toFixed(6)}, ${v.cy.toFixed(6)})  zoom: ${Math.round(v.zoom)}  iters: ${v.maxIter}`;
  }

  function pixelToComplex(canvas: HTMLCanvasElement, px: number, py: number, view: ViewState): [number, number] {
    const rect = canvas.getBoundingClientRect();
    const x = (px - rect.left) * (canvas.width / rect.width);
    const y = (py - rect.top) * (canvas.height / rect.height);
    return [
      view.cx + (x - canvas.width / 2) / view.zoom,
      view.cy + (y - canvas.height / 2) / view.zoom
    ];
  }

  function onModeChange(): void {
    mode = modeSelect.value as Mode;
    const isJuliaRelated = mode === 'julia' || mode === 'linked';
    juliaControls.style.display = isJuliaRelated ? 'flex' : 'none';
    pickerContainer.style.display = mode === 'linked' ? 'flex' : 'none';

    if (mode === 'mandelbrot') {
      mainView = { ...DEFAULT_MANDELBROT };
    } else if (mode === 'julia') {
      mainView = { ...DEFAULT_JULIA };
    } else {
      mainView = { ...DEFAULT_MANDELBROT };
      juliaView = { ...DEFAULT_JULIA };
    }
    scheduleRender();
  }

  // ── Zoom helpers ─────────────────────────────────────────────────────────────

  function zoomView(view: ViewState, factor: number, pivotX: number, pivotY: number, canvas: HTMLCanvasElement): void {
    const [pr, pi] = pixelToComplex(canvas, pivotX, pivotY, view);
    view.zoom *= factor;
    // Keep pivot point stationary
    view.cx = pr - (pivotX - canvas.getBoundingClientRect().left) * (canvas.width / canvas.getBoundingClientRect().width) / view.zoom +
      canvas.width / 2 / view.zoom;
    view.cy = pi - (pivotY - canvas.getBoundingClientRect().top) * (canvas.height / canvas.getBoundingClientRect().height) / view.zoom +
      canvas.height / 2 / view.zoom;
    if (autoIterCheck()) {
      view.maxIter = autoIter(view.zoom);
      maxIterInput.value = String(view.maxIter);
      maxIterDisplay.textContent = String(view.maxIter);
    }
  }

  function zoomViewCentered(view: ViewState, factor: number, pivotCr: number, pivotCi: number): void {
    view.zoom *= factor;
    if (autoIterCheck()) {
      view.maxIter = autoIter(view.zoom);
      maxIterInput.value = String(view.maxIter);
      maxIterDisplay.textContent = String(view.maxIter);
    }
  }

  function autoIterCheck(): boolean {
    return (document.getElementById('auto-iter') as HTMLInputElement)?.checked ?? true;
  }

  // ── Main canvas mouse events ──────────────────────────────────────────────

  mainCanvas.addEventListener('wheel', (e: WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.3 : 1 / 1.3;
    const v = mainView;
    const [pr, pi] = pixelToComplex(mainCanvas, e.clientX, e.clientY, v);
    v.zoom *= factor;
    v.cx = pr - (e.clientX - mainCanvas.getBoundingClientRect().left) * (mainCanvas.width / mainCanvas.getBoundingClientRect().width - 0) / v.zoom + mainCanvas.width / (2 * v.zoom);
    v.cy = pi - (e.clientY - mainCanvas.getBoundingClientRect().top) * (mainCanvas.height / mainCanvas.getBoundingClientRect().height) / v.zoom + mainCanvas.height / (2 * v.zoom);
    if (autoIterCheck()) {
      v.maxIter = autoIter(v.zoom);
      maxIterInput.value = String(v.maxIter);
      maxIterDisplay.textContent = String(v.maxIter);
    }
    scheduleRender(true);
  }, { passive: false });

  mainCanvas.addEventListener('mousedown', (e: MouseEvent) => {
    if (e.button === 0) {
      if (e.shiftKey) {
        // Box zoom
        isBoxSelecting = true;
        boxStart = { x: e.clientX, y: e.clientY };
        if (!overlayCanvas) {
          overlayCanvas = document.createElement('canvas');
          overlayCanvas.style.position = 'absolute';
          overlayCanvas.style.pointerEvents = 'none';
          overlayCanvas.style.top = mainCanvas.offsetTop + 'px';
          overlayCanvas.style.left = mainCanvas.offsetLeft + 'px';
          mainCanvas.parentElement!.appendChild(overlayCanvas);
        }
        overlayCanvas.width = mainCanvas.width;
        overlayCanvas.height = mainCanvas.height;
        overlayCanvas.style.display = 'block';
      } else {
        isDragging = true;
        const rect = mainCanvas.getBoundingClientRect();
        dragStart = { x: e.clientX, y: e.clientY, cx: mainView.cx, cy: mainView.cy };
      }
    }
  });

  window.addEventListener('mousemove', (e: MouseEvent) => {
    if (isDragging && dragStart) {
      const rect = mainCanvas.getBoundingClientRect();
      const dx = (e.clientX - dragStart.x) * (mainCanvas.width / rect.width);
      const dy = (e.clientY - dragStart.y) * (mainCanvas.height / rect.height);
      mainView.cx = dragStart.cx - dx / mainView.zoom;
      mainView.cy = dragStart.cy - dy / mainView.zoom;
      scheduleRender(true);
    }
    if (isBoxSelecting && boxStart && overlayCanvas) {
      const rect = mainCanvas.getBoundingClientRect();
      const ctx = overlayCanvas.getContext('2d')!;
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      const x1 = (boxStart.x - rect.left) * (mainCanvas.width / rect.width);
      const y1 = (boxStart.y - rect.top) * (mainCanvas.height / rect.height);
      const x2 = (e.clientX - rect.left) * (mainCanvas.width / rect.width);
      const y2 = (e.clientY - rect.top) * (mainCanvas.height / rect.height);
      ctx.strokeStyle = 'rgba(255,255,255,0.8)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }
    if (mode === 'linked') {
      const rect = mainCanvas.getBoundingClientRect();
      if (e.clientX >= rect.left && e.clientX <= rect.right &&
          e.clientY >= rect.top && e.clientY <= rect.bottom) {
        const [cr, ci] = pixelToComplex(mainCanvas, e.clientX, e.clientY, mainView);
        juliaC = [cr, ci];
        cRealInput.value = String(cr.toFixed(4));
        cImagInput.value = String(ci.toFixed(4));
        cRealDisplay.textContent = cr.toFixed(4);
        cImagDisplay.textContent = ci.toFixed(4);
        // Draw crosshair on main canvas overlay
        drawPickerCrosshair(e.clientX, e.clientY);
        renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 0.4);
        if (renderTimeout !== null) clearTimeout(renderTimeout);
        renderTimeout = window.setTimeout(() => {
          renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1);
        }, 120);
      }
    }
  });

  function drawPickerCrosshair(clientX: number, clientY: number): void {
    const rect = mainCanvas.getBoundingClientRect();
    if (!overlayCanvas) return;
    overlayCanvas.width = mainCanvas.width;
    overlayCanvas.height = mainCanvas.height;
    overlayCanvas.style.display = 'block';
    const ctx = overlayCanvas.getContext('2d')!;
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    const x = (clientX - rect.left) * (mainCanvas.width / rect.width);
    const y = (clientY - rect.top) * (mainCanvas.height / rect.height);
    ctx.strokeStyle = 'rgba(255,255,100,0.8)';
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x - 10, y); ctx.lineTo(x + 10, y);
    ctx.moveTo(x, y - 10); ctx.lineTo(x, y + 10);
    ctx.stroke();
  }

  window.addEventListener('mouseup', (e: MouseEvent) => {
    if (isDragging) {
      isDragging = false;
      dragStart = null;
      scheduleRender(false);
    }
    if (isBoxSelecting && boxStart) {
      isBoxSelecting = false;
      if (overlayCanvas) { overlayCanvas.style.display = 'none'; }
      const rect = mainCanvas.getBoundingClientRect();
      const x1 = (boxStart.x - rect.left) * (mainCanvas.width / rect.width);
      const y1 = (boxStart.y - rect.top) * (mainCanvas.height / rect.height);
      const x2 = (e.clientX - rect.left) * (mainCanvas.width / rect.width);
      const y2 = (e.clientY - rect.top) * (mainCanvas.height / rect.height);
      const w = Math.abs(x2 - x1);
      const h = Math.abs(y2 - y1);
      if (w > 5 && h > 5) {
        const cx = mainView.cx + (Math.min(x1, x2) + w / 2 - mainCanvas.width / 2) / mainView.zoom;
        const cy = mainView.cy + (Math.min(y1, y2) + h / 2 - mainCanvas.height / 2) / mainView.zoom;
        const factor = Math.min(mainCanvas.width / w, mainCanvas.height / h);
        mainView.cx = cx;
        mainView.cy = cy;
        mainView.zoom *= factor;
        if (autoIterCheck()) {
          mainView.maxIter = autoIter(mainView.zoom);
          maxIterInput.value = String(mainView.maxIter);
          maxIterDisplay.textContent = String(mainView.maxIter);
        }
        scheduleRender(true);
      }
      boxStart = null;
    }
  });

  // Double-click to zoom in centered
  mainCanvas.addEventListener('dblclick', (e: MouseEvent) => {
    const [pr, pi] = pixelToComplex(mainCanvas, e.clientX, e.clientY, mainView);
    mainView.cx = pr;
    mainView.cy = pi;
    mainView.zoom *= 2.5;
    if (autoIterCheck()) {
      mainView.maxIter = autoIter(mainView.zoom);
      maxIterInput.value = String(mainView.maxIter);
      maxIterDisplay.textContent = String(mainView.maxIter);
    }
    scheduleRender(true);
  });

  // ── Picker canvas (Julia linked view) ───────────────────────────────────────

  pickerCanvas.addEventListener('wheel', (e: WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.3 : 1 / 1.3;
    const [pr, pi] = pixelToComplex(pickerCanvas, e.clientX, e.clientY, juliaView);
    juliaView.zoom *= factor;
    juliaView.cx = pr - (e.clientX - pickerCanvas.getBoundingClientRect().left) * (pickerCanvas.width / pickerCanvas.getBoundingClientRect().width) / juliaView.zoom + pickerCanvas.width / (2 * juliaView.zoom);
    juliaView.cy = pi - (e.clientY - pickerCanvas.getBoundingClientRect().top) * (pickerCanvas.height / pickerCanvas.getBoundingClientRect().height) / juliaView.zoom + pickerCanvas.height / (2 * juliaView.zoom);
    renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 0.25);
    if (renderTimeout !== null) clearTimeout(renderTimeout);
    renderTimeout = window.setTimeout(() => renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1), 200);
  }, { passive: false });

  let pickerDrag: { x: number; y: number; cx: number; cy: number } | null = null;
  pickerCanvas.addEventListener('mousedown', (e: MouseEvent) => {
    pickerDrag = { x: e.clientX, y: e.clientY, cx: juliaView.cx, cy: juliaView.cy };
  });
  window.addEventListener('mousemove', (e: MouseEvent) => {
    if (!pickerDrag) return;
    const rect = pickerCanvas.getBoundingClientRect();
    const dx = (e.clientX - pickerDrag.x) * (pickerCanvas.width / rect.width);
    const dy = (e.clientY - pickerDrag.y) * (pickerCanvas.height / rect.height);
    juliaView.cx = pickerDrag.cx - dx / juliaView.zoom;
    juliaView.cy = pickerDrag.cy - dy / juliaView.zoom;
    renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 0.25);
  });
  window.addEventListener('mouseup', () => {
    if (pickerDrag) {
      pickerDrag = null;
      renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1);
    }
  });

  // ── Controls ─────────────────────────────────────────────────────────────────

  modeSelect.addEventListener('change', onModeChange);

  paletteSelect.addEventListener('change', () => {
    palette = paletteSelect.value as PaletteName;
    scheduleRender(false);
    if (mode === 'linked') renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1);
  });

  maxIterInput.addEventListener('input', () => {
    const v = parseInt(maxIterInput.value, 10);
    mainView.maxIter = v;
    juliaView.maxIter = v;
    maxIterDisplay.textContent = String(v);
    scheduleRender(false);
  });

  cRealInput.addEventListener('input', () => {
    juliaC[0] = parseFloat(cRealInput.value);
    cRealDisplay.textContent = parseFloat(cRealInput.value).toFixed(4);
    scheduleRender(false);
    if (mode === 'linked') renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1);
  });

  cImagInput.addEventListener('input', () => {
    juliaC[1] = parseFloat(cImagInput.value);
    cImagDisplay.textContent = parseFloat(cImagInput.value).toFixed(4);
    scheduleRender(false);
    if (mode === 'linked') renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1);
  });

  resetBtn.addEventListener('click', () => {
    if (mode === 'mandelbrot') {
      mainView = { ...DEFAULT_MANDELBROT };
    } else if (mode === 'julia') {
      mainView = { ...DEFAULT_JULIA };
    } else {
      mainView = { ...DEFAULT_MANDELBROT };
      juliaView = { ...DEFAULT_JULIA };
    }
    scheduleRender(false);
    if (mode === 'linked') renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1);
  });

  // ── Overlay canvas positioning ───────────────────────────────────────────────

  function setupOverlay(): void {
    if (!overlayCanvas) {
      overlayCanvas = document.createElement('canvas');
      overlayCanvas.style.position = 'absolute';
      overlayCanvas.style.pointerEvents = 'none';
      overlayCanvas.style.display = 'none';
      mainCanvas.parentElement!.style.position = 'relative';
      mainCanvas.parentElement!.appendChild(overlayCanvas);
    }
    overlayCanvas.width = mainCanvas.width;
    overlayCanvas.height = mainCanvas.height;
    overlayCanvas.style.top = '0px';
    overlayCanvas.style.left = '0px';
  }

  // ── Init ──────────────────────────────────────────────────────────────────────

  setupOverlay();
  onModeChange();
}

document.addEventListener('DOMContentLoaded', initApp);
