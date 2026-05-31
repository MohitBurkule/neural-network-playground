/**
 * Double Pendulum — a classic chaotic system.
 *
 * Integrates the equations of motion of a double pendulum with RK4 and renders
 * it on a canvas, tracing the path of the lower bob. A second, near-identical
 * pendulum can be overlaid to demonstrate sensitive dependence on initial
 * conditions (the hallmark of chaos).
 *
 * Pure TypeScript, no external libraries. Module-scoped (export {} at the end).
 */

interface State {
  a1: number;  // angle of first arm (from vertical)
  a2: number;  // angle of second arm
  v1: number;  // angular velocity 1
  v2: number;  // angular velocity 2
}

interface Params {
  m1: number;
  m2: number;
  l1: number;
  l2: number;
  g: number;
  damping: number;
}

const params: Params = { m1: 10, m2: 10, l1: 120, l2: 120, g: 1, damping: 0 };

let showSecond = true;
let showTrail = true;
let running = true;
let speed = 1;

let p1: State;
let p2: State;
const trail1: Array<[number, number]> = [];
const trail2: Array<[number, number]> = [];
const MAX_TRAIL = 1500;

function deriv(s: State): State {
  const { m1, m2, l1, l2, g } = params;
  const { a1, a2, v1, v2 } = s;
  const d = a1 - a2;
  const den1 = (m1 + m2) * l1 - m2 * l1 * Math.cos(d) * Math.cos(d);
  const den2 = (l2 / l1) * den1;

  const dv1 =
    (-g * (2 * m1 + m2) * Math.sin(a1) -
      m2 * g * Math.sin(a1 - 2 * a2) -
      2 * Math.sin(d) * m2 * (v2 * v2 * l2 + v1 * v1 * l1 * Math.cos(d))) /
    (l1 * (2 * m1 + m2 - m2 * Math.cos(2 * a1 - 2 * a2)));

  const dv2 =
    (2 * Math.sin(d) *
      (v1 * v1 * l1 * (m1 + m2) +
        g * (m1 + m2) * Math.cos(a1) +
        v2 * v2 * l2 * m2 * Math.cos(d))) /
    (l2 * (2 * m1 + m2 - m2 * Math.cos(2 * a1 - 2 * a2)));

  return { a1: v1, a2: v2, v1: dv1, v2: dv2 };
}

function add(s: State, k: State, h: number): State {
  return {
    a1: s.a1 + k.a1 * h,
    a2: s.a2 + k.a2 * h,
    v1: s.v1 + k.v1 * h,
    v2: s.v2 + k.v2 * h
  };
}

function rk4(s: State, h: number): State {
  const k1 = deriv(s);
  const k2 = deriv(add(s, k1, h / 2));
  const k3 = deriv(add(s, k2, h / 2));
  const k4 = deriv(add(s, k3, h));
  const damp = 1 - params.damping;
  return {
    a1: s.a1 + (h / 6) * (k1.a1 + 2 * k2.a1 + 2 * k3.a1 + k4.a1),
    a2: s.a2 + (h / 6) * (k1.a2 + 2 * k2.a2 + 2 * k3.a2 + k4.a2),
    v1: (s.v1 + (h / 6) * (k1.v1 + 2 * k2.v1 + 2 * k3.v1 + k4.v1)) * damp,
    v2: (s.v2 + (h / 6) * (k1.v2 + 2 * k2.v2 + 2 * k3.v2 + k4.v2)) * damp
  };
}

function reset(): void {
  const a = (2 / 3) * Math.PI;
  p1 = { a1: a, a2: a, v1: 0, v2: 0 };
  p2 = { a1: a + 0.001, a2: a, v1: 0, v2: 0 };  // tiny perturbation
  trail1.length = 0;
  trail2.length = 0;
}

function bobPos(s: State, ox: number, oy: number): [number, number, number, number] {
  const x1 = ox + params.l1 * Math.sin(s.a1);
  const y1 = oy + params.l1 * Math.cos(s.a1);
  const x2 = x1 + params.l2 * Math.sin(s.a2);
  const y2 = y1 + params.l2 * Math.cos(s.a2);
  return [x1, y1, x2, y2];
}

function drawPendulum(
  ctx: CanvasRenderingContext2D, s: State, ox: number, oy: number,
  color: string, trail: Array<[number, number]>
): void {
  const [x1, y1, x2, y2] = bobPos(s, ox, oy);

  if (showTrail && trail.length > 1) {
    ctx.beginPath();
    for (let i = 0; i < trail.length; i++) {
      const [tx, ty] = trail[i];
      if (i === 0) ctx.moveTo(tx, ty); else ctx.lineTo(tx, ty);
    }
    ctx.strokeStyle = color + '55';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(ox, oy);
  ctx.lineTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  ctx.fillStyle = color;
  for (const [bx, by, r] of [[x1, y1, 6], [x2, y2, 8]] as Array<[number, number, number]>) {
    ctx.beginPath();
    ctx.arc(bx, by, r, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function step(): void {
  const h = 0.2 * speed;
  const sub = 4;
  for (let i = 0; i < sub; i++) {
    p1 = rk4(p1, h / sub);
    if (showSecond) p2 = rk4(p2, h / sub);
  }
}

function loop(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  const ox = w / 2;
  const oy = h / 2 - 60;

  if (running) {
    step();
    const [, , x2, y2] = bobPos(p1, ox, oy);
    trail1.push([x2, y2]);
    if (trail1.length > MAX_TRAIL) trail1.shift();
    if (showSecond) {
      const [, , x2b, y2b] = bobPos(p2, ox, oy);
      trail2.push([x2b, y2b]);
      if (trail2.length > MAX_TRAIL) trail2.shift();
    }
  }

  ctx.fillStyle = '#0f1117';
  ctx.fillRect(0, 0, w, h);

  if (showSecond) drawPendulum(ctx, p2, ox, oy, '#4d9fff', trail2);
  drawPendulum(ctx, p1, ox, oy, '#ff4d6d', trail1);

  ctx.fillStyle = '#888';
  ctx.font = '12px monospace';
  ctx.fillText('Double pendulum — chaotic for most initial conditions', 12, 20);
  if (showSecond) {
    ctx.fillText('Two pendulums start 0.001 rad apart and diverge — sensitive dependence on initial conditions.', 12, h - 14);
  }

  requestAnimationFrame(() => loop(ctx, w, h));
}

function el<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

function init(): void {
  const canvas = el<HTMLCanvasElement>('pendulum-canvas');
  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

  reset();

  const bind = (id: string, fn: (v: number) => void) => {
    const node = document.getElementById(id) as HTMLInputElement;
    if (!node) return;
    node.addEventListener('input', () => fn(parseFloat(node.value)));
  };
  bind('pen-speed', v => (speed = v));
  bind('pen-m1', v => (params.m1 = v));
  bind('pen-m2', v => (params.m2 = v));
  bind('pen-l1', v => (params.l1 = v));
  bind('pen-l2', v => (params.l2 = v));
  bind('pen-damping', v => (params.damping = v / 10000));

  const playBtn = document.getElementById('pen-play');
  if (playBtn) playBtn.addEventListener('click', () => {
    running = !running;
    playBtn.textContent = running ? 'Pause' : 'Play';
  });
  const resetBtn = document.getElementById('pen-reset');
  if (resetBtn) resetBtn.addEventListener('click', () => reset());
  const trailChk = document.getElementById('pen-trail') as HTMLInputElement;
  if (trailChk) trailChk.addEventListener('change', () => (showTrail = trailChk.checked));
  const secondChk = document.getElementById('pen-second') as HTMLInputElement;
  if (secondChk) secondChk.addEventListener('change', () => {
    showSecond = secondChk.checked;
    trail2.length = 0;
  });

  loop(ctx, w, h);
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}

export {};
