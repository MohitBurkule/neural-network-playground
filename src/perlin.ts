/**
 * Perlin / value noise explorer.
 *
 * Implements 2D gradient (Perlin) noise and fractal Brownian motion (fBm) and
 * renders it as a colormapped field on a canvas. Useful for procedural terrain,
 * clouds and textures. Pure TypeScript, no external libraries.
 */

// Permutation table (Ken Perlin's reference values), duplicated to 512.
const PERM_BASE = [151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,
103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,
219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,
165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,
41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,
169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,
124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,
42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,
22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,
193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,181,
199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,
29,24,72,243,141,128,195,78,66,215,61,156,180];

let perm = new Uint8Array(512);

function seedPerm(seed: number): void {
  const p = PERM_BASE.slice();
  // Simple deterministic shuffle from seed.
  let s = seed >>> 0 || 1;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  for (let i = p.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const t = p[i]; p[i] = p[j]; p[j] = t;
  }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
}

function fade(t: number): number { return t * t * t * (t * (t * 6 - 15) + 10); }
function lerp(a: number, b: number, t: number): number { return a + t * (b - a); }
function grad(hash: number, x: number, y: number): number {
  const h = hash & 7;
  const u = h < 4 ? x : y;
  const v = h < 4 ? y : x;
  return ((h & 1) ? -u : u) + ((h & 2) ? -2 * v : 2 * v);
}

function perlin2(x: number, y: number): number {
  const X = Math.floor(x) & 255;
  const Y = Math.floor(y) & 255;
  const xf = x - Math.floor(x);
  const yf = y - Math.floor(y);
  const u = fade(xf), v = fade(yf);
  const aa = perm[perm[X] + Y];
  const ab = perm[perm[X] + Y + 1];
  const ba = perm[perm[X + 1] + Y];
  const bb = perm[perm[X + 1] + Y + 1];
  const x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u);
  const x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u);
  return lerp(x1, x2, v); // ~[-1,1]
}

function fbm(x: number, y: number, octaves: number, persistence: number, lacunarity: number): number {
  let total = 0, amp = 1, freq = 1, max = 0;
  for (let o = 0; o < octaves; o++) {
    total += perlin2(x * freq, y * freq) * amp;
    max += amp;
    amp *= persistence;
    freq *= lacunarity;
  }
  return total / max;
}

interface Cfg {
  scale: number; octaves: number; persistence: number; lacunarity: number;
  seed: number; palette: string; animate: boolean;
}
const cfg: Cfg = { scale: 0.012, octaves: 5, persistence: 0.5, lacunarity: 2, seed: 1, palette: 'terrain', animate: false };
let zoff = 0;

function color(v: number, pal: string, out: number[]): void {
  // v in [-1,1] -> [0,1]
  const t = (v + 1) / 2;
  if (pal === 'gray') {
    const g = Math.round(t * 255); out[0] = g; out[1] = g; out[2] = g; return;
  }
  if (pal === 'terrain') {
    if (t < 0.38) { out[0] = 20; out[1] = 40; out[2] = Math.round(80 + t * 200); }       // deep water
    else if (t < 0.45) { out[0] = 30; out[1] = 90; out[2] = 180; }                         // shallow
    else if (t < 0.5) { out[0] = 210; out[1] = 200; out[2] = 140; }                        // beach
    else if (t < 0.7) { out[0] = Math.round(60 + t * 60); out[1] = Math.round(140 - t * 40); out[2] = 60; } // grass
    else if (t < 0.85) { out[0] = 110; out[1] = 90; out[2] = 70; }                         // rock
    else { const g = Math.round(200 + t * 55); out[0] = g; out[1] = g; out[2] = g; }       // snow
    return;
  }
  // fire
  out[0] = Math.round(Math.min(255, t * 400));
  out[1] = Math.round(Math.min(255, Math.max(0, t * 400 - 150)));
  out[2] = Math.round(Math.min(255, Math.max(0, t * 400 - 300)));
}

function render(ctx: CanvasRenderingContext2D, img: ImageData, w: number, h: number): void {
  const data = img.data;
  const out = [0, 0, 0];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = fbm(x * cfg.scale + zoff, y * cfg.scale + zoff, cfg.octaves, cfg.persistence, cfg.lacunarity);
      color(v, cfg.palette, out);
      const o = (y * w + x) * 4;
      data[o] = out[0]; data[o + 1] = out[1]; data[o + 2] = out[2]; data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function init(): void {
  const canvas = document.getElementById('perlin-canvas') as HTMLCanvasElement;
  const w = canvas.width, h = canvas.height;
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
  let img = ctx.createImageData(w, h);
  seedPerm(cfg.seed);

  const redraw = () => render(ctx, img, w, h);
  const bind = (id: string, fn: (v: number) => void) => {
    const node = document.getElementById(id) as HTMLInputElement;
    if (node) node.addEventListener('input', () => { fn(parseFloat(node.value)); if (!cfg.animate) redraw(); });
  };
  bind('pl-scale', v => (cfg.scale = v / 1000));
  bind('pl-octaves', v => (cfg.octaves = Math.round(v)));
  bind('pl-persist', v => (cfg.persistence = v));
  bind('pl-lacun', v => (cfg.lacunarity = v));
  const pal = document.getElementById('pl-palette') as HTMLSelectElement;
  if (pal) pal.addEventListener('change', () => { cfg.palette = pal.value; if (!cfg.animate) redraw(); });
  const reseed = document.getElementById('pl-reseed');
  if (reseed) reseed.addEventListener('click', () => { cfg.seed = Math.floor(Math.random() * 1e9); seedPerm(cfg.seed); redraw(); });
  const anim = document.getElementById('pl-animate') as HTMLInputElement;
  if (anim) anim.addEventListener('change', () => (cfg.animate = anim.checked));

  redraw();
  function loop(): void {
    if (cfg.animate) { zoff += 0.003; redraw(); }
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
