/**
 * L-System (Lindenmayer system) lab.
 *
 * Rewrites an axiom with production rules for N iterations, then interprets the
 * resulting string with a turtle to draw fractal plants and curves on a canvas.
 * Several classic presets (Koch, dragon, Sierpinski, plants, Hilbert) included.
 * Pure TypeScript, no external libraries.
 */

interface Preset {
  name: string;
  axiom: string;
  rules: { [k: string]: string };
  angle: number;     // degrees
  iterations: number;
  start: number;     // initial heading (degrees, 0 = up)
}

const PRESETS: { [k: string]: Preset } = {
  plant: {
    name: 'Fractal plant',
    axiom: 'X',
    rules: { X: 'F+[[X]-X]-F[-FX]+X', F: 'FF' },
    angle: 25, iterations: 5, start: 0
  },
  koch: {
    name: 'Koch snowflake',
    axiom: 'F++F++F',
    rules: { F: 'F-F++F-F' },
    angle: 60, iterations: 4, start: 0
  },
  dragon: {
    name: 'Dragon curve',
    axiom: 'FX',
    rules: { X: 'X+YF+', Y: '-FX-Y' },
    angle: 90, iterations: 11, start: 0
  },
  sierpinski: {
    name: 'Sierpinski triangle',
    axiom: 'F-G-G',
    rules: { F: 'F-G+F+G-F', G: 'GG' },
    angle: 120, iterations: 5, start: 0
  },
  hilbert: {
    name: 'Hilbert curve',
    axiom: 'A',
    rules: { A: '+BF-AFA-FB+', B: '-AF+BFB+FA-' },
    angle: 90, iterations: 6, start: 0
  },
  bush: {
    name: 'Bushy plant',
    axiom: 'F',
    rules: { F: 'FF+[+F-F-F]-[-F+F+F]' },
    angle: 22, iterations: 4, start: 0
  }
};

let current: Preset = JSON.parse(JSON.stringify(PRESETS.plant));
let angleJitter = 0;

function expand(p: Preset): string {
  let s = p.axiom;
  for (let i = 0; i < p.iterations; i++) {
    let next = '';
    for (let k = 0; k < s.length; k++) {
      const ch = s[k];
      next += (p.rules[ch] !== undefined) ? p.rules[ch] : ch;
      if (next.length > 2_000_000) break; // safety cap
    }
    s = next;
    if (s.length > 2_000_000) break;
  }
  return s;
}

interface Turtle { x: number; y: number; a: number; }

function draw(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  const str = expand(current);
  const angle = (current.angle * Math.PI) / 180;
  const jitter = (angleJitter * Math.PI) / 180;

  // First pass: compute bounding box at unit step to auto-fit.
  let t: Turtle = { x: 0, y: 0, a: (current.start - 90) * Math.PI / 180 };
  const stack: Turtle[] = [];
  let minX = 0, maxX = 0, minY = 0, maxY = 0;
  const drawable = (ch: string) => ch === 'F' || ch === 'G' || ch === 'A' || ch === 'B';
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (drawable(ch)) {
      t.x += Math.cos(t.a); t.y += Math.sin(t.a);
      minX = Math.min(minX, t.x); maxX = Math.max(maxX, t.x);
      minY = Math.min(minY, t.y); maxY = Math.max(maxY, t.y);
    } else if (ch === '+') t.a += angle;
    else if (ch === '-') t.a -= angle;
    else if (ch === '[') stack.push({ ...t });
    else if (ch === ']') { const s = stack.pop(); if (s) t = s; }
  }

  const bw = Math.max(1e-6, maxX - minX);
  const bh = Math.max(1e-6, maxY - minY);
  const pad = 30;
  const step = Math.min((w - 2 * pad) / bw, (h - 2 * pad) / bh);
  const ox = pad - minX * step + ((w - 2 * pad) - bw * step) / 2;
  const oy = pad - minY * step + ((h - 2 * pad) - bh * step) / 2;

  ctx.fillStyle = '#0f1117';
  ctx.fillRect(0, 0, w, h);
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#4dffa0';

  // Second pass: draw.
  t = { x: 0, y: 0, a: (current.start - 90) * Math.PI / 180 };
  const stack2: Turtle[] = [];
  let depth = 0;
  ctx.beginPath();
  for (let i = 0; i < str.length; i++) {
    const ch = str[i];
    if (drawable(ch)) {
      const nx = t.x + Math.cos(t.a);
      const ny = t.y + Math.sin(t.a);
      // Color by stack depth for a plant-like gradient.
      ctx.stroke();
      const g = Math.min(255, 120 + depth * 30);
      ctx.strokeStyle = `rgb(${Math.max(40, 90 - depth * 10)},${g},${Math.max(60, 120 - depth * 8)})`;
      ctx.beginPath();
      ctx.moveTo(ox + t.x * step, oy + t.y * step);
      ctx.lineTo(ox + nx * step, oy + ny * step);
      t.x = nx; t.y = ny;
    } else if (ch === '+') t.a += angle + (Math.random() * 2 - 1) * jitter;
    else if (ch === '-') t.a -= angle + (Math.random() * 2 - 1) * jitter;
    else if (ch === '[') { stack2.push({ ...t }); depth++; }
    else if (ch === ']') { const s = stack2.pop(); if (s) t = s; depth = Math.max(0, depth - 1); }
  }
  ctx.stroke();

  const lenEl = document.getElementById('ls-len');
  if (lenEl) lenEl.textContent = str.length.toLocaleString();
}

function init(): void {
  const canvas = document.getElementById('ls-canvas') as HTMLCanvasElement;
  const w = canvas.width, h = canvas.height;
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

  const redraw = () => draw(ctx, w, h);

  const sel = document.getElementById('ls-preset') as HTMLSelectElement;
  if (sel) {
    // populate
    for (const key of Object.keys(PRESETS)) {
      const opt = document.createElement('option');
      opt.value = key; opt.textContent = PRESETS[key].name;
      sel.appendChild(opt);
    }
    sel.addEventListener('change', () => {
      current = JSON.parse(JSON.stringify(PRESETS[sel.value]));
      const itEl = document.getElementById('ls-iter') as HTMLInputElement;
      const anEl = document.getElementById('ls-angle') as HTMLInputElement;
      if (itEl) itEl.value = String(current.iterations);
      if (anEl) anEl.value = String(current.angle);
      redraw();
    });
  }
  const it = document.getElementById('ls-iter') as HTMLInputElement;
  if (it) it.addEventListener('input', () => { current.iterations = parseInt(it.value, 10); redraw(); });
  const an = document.getElementById('ls-angle') as HTMLInputElement;
  if (an) an.addEventListener('input', () => { current.angle = parseFloat(an.value); redraw(); });
  const jit = document.getElementById('ls-jitter') as HTMLInputElement;
  if (jit) jit.addEventListener('input', () => { angleJitter = parseFloat(jit.value); redraw(); });

  redraw();
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}

export {};
