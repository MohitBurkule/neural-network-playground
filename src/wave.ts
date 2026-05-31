/**
 * 2D Wave Equation simulator.
 *
 * Integrates the discrete wave equation u_tt = c^2 (u_xx + u_yy) on a grid with
 * a finite-difference stencil, rendered to a canvas via ImageData. Click/drag to
 * drop ripples; optional walls/obstacles demonstrate reflection, diffraction and
 * interference. Pure TypeScript, no external libraries.
 */

let N = 200;            // grid size (NxN)
let u: Float32Array;    // current displacement
let uPrev: Float32Array;
let uNext: Float32Array;
let walls: Uint8Array;  // 1 = fixed/wall cell
let c = 0.5;            // wave speed (CFL: c^2 <= 0.5 for stability here)
let damping = 0.999;
let running = true;
let speed = 2;
let mode: 'ripple' | 'doubleslit' = 'ripple';
const palette = buildPalette();

function idx(x: number, y: number): number { return y * N + x; }

function buildPalette(): Uint8Array {
  // Blue (negative) -> black (zero) -> red/orange (positive).
  const p = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * 2 - 1; // -1..1
    let r: number, g: number, b: number;
    if (t < 0) { r = 0; g = Math.round(-t * 90); b = Math.round(-t * 255); }
    else { r = Math.round(t * 255); g = Math.round(t * 150); b = 0; }
    p[i * 3] = r; p[i * 3 + 1] = g; p[i * 3 + 2] = b;
  }
  return p;
}

function allocate(): void {
  u = new Float32Array(N * N);
  uPrev = new Float32Array(N * N);
  uNext = new Float32Array(N * N);
  walls = new Uint8Array(N * N);
}

function setupWalls(): void {
  walls.fill(0);
  if (mode === 'doubleslit') {
    const bx = Math.floor(N * 0.42);
    const slit = Math.floor(N * 0.04);
    const gap = Math.floor(N * 0.07);
    const c0 = Math.floor(N / 2);
    for (let y = 0; y < N; y++) {
      const inSlit =
        (Math.abs(y - (c0 - gap)) < slit) || (Math.abs(y - (c0 + gap)) < slit);
      if (!inSlit) { walls[idx(bx, y)] = 1; walls[idx(bx + 1, y)] = 1; }
    }
  }
}

function reset(): void {
  u.fill(0); uPrev.fill(0); uNext.fill(0);
  setupWalls();
}

function drip(cx: number, cy: number, amp: number, rad: number): void {
  for (let y = -rad; y <= rad; y++) {
    for (let x = -rad; x <= rad; x++) {
      const px = cx + x, py = cy + y;
      if (px < 1 || py < 1 || px >= N - 1 || py >= N - 1) continue;
      const r = Math.sqrt(x * x + y * y);
      if (r > rad) continue;
      const v = amp * Math.cos((r / rad) * (Math.PI / 2));
      u[idx(px, py)] = v;
      uPrev[idx(px, py)] = v;
    }
  }
}

let oscPhase = 0;
function step(): void {
  const c2 = c * c;
  for (let y = 1; y < N - 1; y++) {
    for (let x = 1; x < N - 1; x++) {
      const i = idx(x, y);
      if (walls[i]) { uNext[i] = 0; continue; }
      const lap = u[i - 1] + u[i + 1] + u[i - N] + u[i + N] - 4 * u[i];
      uNext[i] = (2 * u[i] - uPrev[i] + c2 * lap) * damping;
    }
  }
  // Continuous line source for the double-slit demo.
  if (mode === 'doubleslit') {
    oscPhase += 0.35;
    const sx = Math.floor(N * 0.18);
    const v = Math.sin(oscPhase) * 0.8;
    for (let y = 1; y < N - 1; y++) uNext[idx(sx, y)] = v;
  }
  const tmp = uPrev; uPrev = u; u = uNext; uNext = tmp;
}

function render(ctx: CanvasRenderingContext2D, img: ImageData, scale: number): void {
  const data = img.data;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = idx(x, y);
      let v = u[i] * 4;
      if (v > 1) v = 1; else if (v < -1) v = -1;
      const ci = Math.round(((v + 1) / 2) * 255);
      let r = palette[ci * 3], g = palette[ci * 3 + 1], b = palette[ci * 3 + 2];
      if (walls[i]) { r = 90; g = 95; b = 110; }
      const o = i * 4;
      data[o] = r; data[o + 1] = g; data[o + 2] = b; data[o + 3] = 255;
    }
  }
  // Put the NxN image then scale it up onto the visible canvas.
  ctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(ctx.canvas, 0, 0, N, N, 0, 0, N * scale, N * scale);
}

function init(): void {
  const canvas = document.getElementById('wave-canvas') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
  allocate();
  reset();
  const scale = Math.max(1, Math.floor(canvas.width / N));
  let img = ctx.createImageData(N, N);

  let down = false;
  const toGrid = (e: MouseEvent): [number, number] => {
    const rect = canvas.getBoundingClientRect();
    const gx = Math.floor(((e.clientX - rect.left) / rect.width) * N);
    const gy = Math.floor(((e.clientY - rect.top) / rect.height) * N);
    return [gx, gy];
  };
  canvas.addEventListener('mousedown', e => { down = true; const [x, y] = toGrid(e); drip(x, y, 1.2, 4); });
  canvas.addEventListener('mousemove', e => { if (down) { const [x, y] = toGrid(e); drip(x, y, 0.8, 3); } });
  window.addEventListener('mouseup', () => (down = false));

  const bind = (id: string, fn: (v: number) => void) => {
    const node = document.getElementById(id) as HTMLInputElement;
    if (node) node.addEventListener('input', () => fn(parseFloat(node.value)));
  };
  bind('wave-speed', v => (speed = v));
  bind('wave-c', v => (c = v));
  bind('wave-damp', v => (damping = 1 - v / 10000));

  const modeSel = document.getElementById('wave-mode') as HTMLSelectElement;
  if (modeSel) modeSel.addEventListener('change', () => {
    mode = modeSel.value as 'ripple' | 'doubleslit';
    reset();
  });
  const play = document.getElementById('wave-play');
  if (play) play.addEventListener('click', () => {
    running = !running;
    play.textContent = running ? 'Pause' : 'Play';
  });
  const rst = document.getElementById('wave-reset');
  if (rst) rst.addEventListener('click', () => reset());

  function loop(): void {
    if (running) { for (let s = 0; s < speed; s++) step(); }
    render(ctx, img, scale);
    ctx.fillStyle = '#888';
    ctx.font = '12px monospace';
    ctx.fillText('2D wave equation — click & drag to make ripples', 10, 18);
    requestAnimationFrame(loop);
  }
  loop();
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}

export {};
