import * as d3 from 'd3';

// ─── Types ────────────────────────────────────────────────────────────────────

interface Vec2 { x: number; y: number; }

interface Surface {
  name: string;
  domain: [number, number]; // symmetric square domain
  f: (x: number, y: number) => number;
  grad: (x: number, y: number) => Vec2;
  minima?: Vec2[]; // known minima for reference
}

interface OptimizerState {
  name: string;
  color: string;
  path: Vec2[];
  losses: number[];
  enabled: boolean;
  // internal state (varies by optimizer)
  vx: number; vy: number;       // momentum / first moment
  sx: number; sy: number;       // second moment / squared gradient
  t: number;                    // step count
}

interface Params {
  lr: number;
  beta1: number;   // momentum / Adam beta1
  beta2: number;   // Adam beta2
  eps: number;     // Adam epsilon
  decay: number;   // RMSProp decay
}

// ─── Loss Surfaces ────────────────────────────────────────────────────────────

const SURFACES: Surface[] = [
  {
    name: 'Rosenbrock',
    domain: [-2, 2],
    f: (x, y) => (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x),
    grad: (x, y) => ({
      x: -2 * (1 - x) - 400 * x * (y - x * x),
      y: 200 * (y - x * x)
    }),
    minima: [{ x: 1, y: 1 }]
  },
  {
    name: 'Saddle Point',
    domain: [-3, 3],
    f: (x, y) => x * x - y * y,
    grad: (x, y) => ({ x: 2 * x, y: -2 * y }),
    minima: []
  },
  {
    name: 'Beale',
    domain: [-4.5, 4.5],
    f: (x, y) => {
      const a = 1.5 - x + x * y;
      const b = 2.25 - x + x * y * y;
      const c = 2.625 - x + x * y * y * y;
      return a * a + b * b + c * c;
    },
    grad: (x, y) => {
      const a = 1.5 - x + x * y;
      const b = 2.25 - x + x * y * y;
      const c = 2.625 - x + x * y * y * y;
      return {
        x: 2 * a * (-1 + y) + 2 * b * (-1 + y * y) + 2 * c * (-1 + y * y * y),
        y: 2 * a * x + 2 * b * (2 * x * y) + 2 * c * (3 * x * y * y)
      };
    },
    minima: [{ x: 3, y: 0.5 }]
  },
  {
    name: 'Ravine (Ill-cond.)',
    domain: [-3, 3],
    f: (x, y) => x * x + 50 * y * y,
    grad: (x, y) => ({ x: 2 * x, y: 100 * y }),
    minima: [{ x: 0, y: 0 }]
  },
  {
    name: 'Himmelblau',
    domain: [-5, 5],
    f: (x, y) => {
      const a = x * x + y - 11;
      const b = x + y * y - 7;
      return a * a + b * b;
    },
    grad: (x, y) => {
      const a = x * x + y - 11;
      const b = x + y * y - 7;
      return {
        x: 4 * x * a + 2 * b,
        y: 2 * a + 4 * y * b
      };
    },
    minima: [
      { x: 3, y: 2 }, { x: -2.805, y: 3.131 },
      { x: -3.779, y: -3.283 }, { x: 3.584, y: -1.848 }
    ]
  },
  {
    name: 'Rastrigin',
    domain: [-3, 3],
    f: (x, y) => {
      const n = 2;
      return 10 * n + (x * x - 10 * Math.cos(2 * Math.PI * x))
                    + (y * y - 10 * Math.cos(2 * Math.PI * y));
    },
    grad: (x, y) => ({
      x: 2 * x + 20 * Math.PI * Math.sin(2 * Math.PI * x),
      y: 2 * y + 20 * Math.PI * Math.sin(2 * Math.PI * y)
    }),
    minima: [{ x: 0, y: 0 }]
  }
];

// ─── Optimizer definitions ────────────────────────────────────────────────────

const OPTIMIZER_DEFS: { name: string; color: string }[] = [
  { name: 'SGD',       color: '#e74c3c' },
  { name: 'Momentum',  color: '#e67e22' },
  { name: 'Nesterov',  color: '#f1c40f' },
  { name: 'Adagrad',   color: '#2ecc71' },
  { name: 'RMSProp',   color: '#1abc9c' },
  { name: 'Adam',      color: '#3498db' },
];

function makeOptimizer(def: { name: string; color: string }): OptimizerState {
  return { ...def, path: [], losses: [], enabled: true, vx: 0, vy: 0, sx: 0, sy: 0, t: 0 };
}

function stepOptimizer(
  opt: OptimizerState,
  surface: Surface,
  params: Params
): void {
  const pos = opt.path[opt.path.length - 1];
  const { lr, beta1, beta2, eps, decay } = params;

  let gx: number, gy: number;
  let nx = pos.x, ny = pos.y;

  switch (opt.name) {
    case 'SGD': {
      const g = surface.grad(pos.x, pos.y);
      nx = pos.x - lr * g.x;
      ny = pos.y - lr * g.y;
      break;
    }
    case 'Momentum': {
      const g = surface.grad(pos.x, pos.y);
      opt.vx = beta1 * opt.vx - lr * g.x;
      opt.vy = beta1 * opt.vy - lr * g.y;
      nx = pos.x + opt.vx;
      ny = pos.y + opt.vy;
      break;
    }
    case 'Nesterov': {
      // Evaluate gradient at lookahead point
      const lx = pos.x + beta1 * opt.vx;
      const ly = pos.y + beta1 * opt.vy;
      const g = surface.grad(lx, ly);
      opt.vx = beta1 * opt.vx - lr * g.x;
      opt.vy = beta1 * opt.vy - lr * g.y;
      nx = pos.x + opt.vx;
      ny = pos.y + opt.vy;
      break;
    }
    case 'Adagrad': {
      const g = surface.grad(pos.x, pos.y);
      opt.sx += g.x * g.x;
      opt.sy += g.y * g.y;
      nx = pos.x - lr * g.x / (Math.sqrt(opt.sx) + eps);
      ny = pos.y - lr * g.y / (Math.sqrt(opt.sy) + eps);
      break;
    }
    case 'RMSProp': {
      const g = surface.grad(pos.x, pos.y);
      opt.sx = decay * opt.sx + (1 - decay) * g.x * g.x;
      opt.sy = decay * opt.sy + (1 - decay) * g.y * g.y;
      nx = pos.x - lr * g.x / (Math.sqrt(opt.sx) + eps);
      ny = pos.y - lr * g.y / (Math.sqrt(opt.sy) + eps);
      break;
    }
    case 'Adam': {
      opt.t += 1;
      const g = surface.grad(pos.x, pos.y);
      opt.vx = beta1 * opt.vx + (1 - beta1) * g.x;
      opt.vy = beta1 * opt.vy + (1 - beta1) * g.y;
      opt.sx = beta2 * opt.sx + (1 - beta2) * g.x * g.x;
      opt.sy = beta2 * opt.sy + (1 - beta2) * g.y * g.y;
      const mhx = opt.vx / (1 - Math.pow(beta1, opt.t));
      const mhy = opt.vy / (1 - Math.pow(beta1, opt.t));
      const vhx = opt.sx / (1 - Math.pow(beta2, opt.t));
      const vhy = opt.sy / (1 - Math.pow(beta2, opt.t));
      nx = pos.x - lr * mhx / (Math.sqrt(vhx) + eps);
      ny = pos.y - lr * mhy / (Math.sqrt(vhy) + eps);
      break;
    }
    default:
      return;
  }

  // Clamp to domain
  const [dMin, dMax] = surface.domain;
  nx = Math.max(dMin, Math.min(dMax, nx));
  ny = Math.max(dMin, Math.min(dMax, ny));
  opt.path.push({ x: nx, y: ny });
  opt.losses.push(surface.f(nx, ny));
}

// ─── App State ────────────────────────────────────────────────────────────────

let surfaceIdx = 0;
let startPoint: Vec2 = { x: -1.5, y: 1.5 };
let optimizers: OptimizerState[] = OPTIMIZER_DEFS.map(makeOptimizer);
let params: Params = { lr: 0.01, beta1: 0.9, beta2: 0.999, eps: 1e-8, decay: 0.9 };
let playing = false;
let speed = 5; // steps per frame
let animId: number | null = null;
let maxSteps = 300;

// Canvas / chart sizes
const CANVAS_SIZE = 480;
const CHART_H = 160;
const CHART_W = 480;

// ─── DOM setup ────────────────────────────────────────────────────────────────

const root = d3.select('#app');

// Title
root.append('h1').text('Gradient Descent Optimizer Visualizer');

const mainRow = root.append('div').attr('class', 'main-row');

// Left: contour canvas
const leftCol = mainRow.append('div').attr('class', 'left-col');
const canvasEl = leftCol.append('canvas')
  .attr('width', CANVAS_SIZE)
  .attr('height', CANVAS_SIZE)
  .attr('id', 'contour-canvas')
  .node() as HTMLCanvasElement;
const ctx = canvasEl.getContext('2d')!;

// Right: controls + legend
const rightCol = mainRow.append('div').attr('class', 'right-col');

// Surface selector
rightCol.append('label').text('Surface:');
const surfSel = rightCol.append('select').attr('id', 'surf-select');
SURFACES.forEach((s, i) => surfSel.append('option').attr('value', i).text(s.name));

// LR
rightCol.append('br');
rightCol.append('label').text('Learning Rate:');
const lrInput = rightCol.append('input').attr('type', 'number').attr('step', '0.001')
  .attr('min', '0.0001').attr('max', '1').attr('value', params.lr);

// beta1
rightCol.append('br');
rightCol.append('label').text('β₁ (momentum):');
const b1Input = rightCol.append('input').attr('type', 'number').attr('step', '0.01')
  .attr('min', '0').attr('max', '0.999').attr('value', params.beta1);

// beta2
rightCol.append('br');
rightCol.append('label').text('β₂ (Adam):');
const b2Input = rightCol.append('input').attr('type', 'number').attr('step', '0.001')
  .attr('min', '0').attr('max', '0.9999').attr('value', params.beta2);

// speed
rightCol.append('br');
rightCol.append('label').text('Speed (steps/frame):');
const speedInput = rightCol.append('input').attr('type', 'range')
  .attr('min', '1').attr('max', '20').attr('value', speed);
const speedLabel = rightCol.append('span').text(` ${speed}`);

// max steps
rightCol.append('br');
rightCol.append('label').text('Max steps:');
const maxStepsInput = rightCol.append('input').attr('type', 'number')
  .attr('min', '50').attr('max', '2000').attr('step', '50').attr('value', maxSteps);

// Optimizer checkboxes
rightCol.append('br');
rightCol.append('div').attr('class', 'section-title').text('Optimizers:');
const checkboxDiv = rightCol.append('div').attr('id', 'opt-checkboxes');

OPTIMIZER_DEFS.forEach((def, i) => {
  const lbl = checkboxDiv.append('label').style('color', def.color);
  lbl.append('input').attr('type', 'checkbox').attr('checked', true)
    .attr('data-idx', i);
  lbl.append('span').text(' ' + def.name);
  checkboxDiv.append('br');
});

// Buttons
rightCol.append('div').attr('class', 'btn-row').html(
  '<button id="btn-play">▶ Play</button>' +
  '<button id="btn-pause">⏸ Pause</button>' +
  '<button id="btn-step">⏭ Step</button>' +
  '<button id="btn-reset">↺ Reset</button>'
);

// Hint
rightCol.append('p').attr('class', 'hint').text('Click on the surface to set start point.');

// Legend
const legendDiv = rightCol.append('div').attr('class', 'legend');
legendDiv.append('div').attr('class', 'section-title').text('Legend:');
OPTIMIZER_DEFS.forEach(def => {
  const row = legendDiv.append('div').attr('class', 'legend-row');
  row.append('span').style('background', def.color).attr('class', 'legend-swatch');
  row.append('span').text(def.name);
});

// Bottom: loss chart
const chartDiv = root.append('div').attr('class', 'chart-container');
chartDiv.append('div').attr('class', 'section-title').text('Loss vs. Step');
const svgChart = chartDiv.append('svg')
  .attr('width', CHART_W)
  .attr('height', CHART_H + 30);

// ─── Scales ──────────────────────────────────────────────────────────────────

function getSurface(): Surface { return SURFACES[surfaceIdx]; }

// Map domain coords to canvas pixels
function domToCanvas(v: number, domain: [number, number]): number {
  const [dMin, dMax] = domain;
  return ((v - dMin) / (dMax - dMin)) * CANVAS_SIZE;
}
function canvasToDom(p: number, domain: [number, number]): number {
  const [dMin, dMax] = domain;
  return dMin + (p / CANVAS_SIZE) * (dMax - dMin);
}

// ─── Heatmap / Contour rendering ─────────────────────────────────────────────

let heatmapCache: ImageData | null = null;
let heatmapSurfaceIdx = -1;

function buildHeatmap(): void {
  const surf = getSurface();
  const [dMin, dMax] = surf.domain;
  const N = CANVAS_SIZE;
  const imgData = ctx.createImageData(N, N);
  let fMin = Infinity, fMax = -Infinity;
  const vals = new Float32Array(N * N);

  for (let py = 0; py < N; py++) {
    for (let px = 0; px < N; px++) {
      const x = dMin + (px / N) * (dMax - dMin);
      const y = dMin + ((N - 1 - py) / N) * (dMax - dMin);
      const v = surf.f(x, y);
      vals[py * N + px] = v;
      if (v < fMin) fMin = v;
      if (v > fMax) fMax = v;
    }
  }

  // Use log scale for better visual range
  const logMin = Math.log1p(Math.max(0, fMin));
  const logMax = Math.log1p(Math.max(0, fMax));

  for (let py = 0; py < N; py++) {
    for (let px = 0; px < N; px++) {
      const v = vals[py * N + px];
      const vNorm = (logMax > logMin)
        ? (Math.log1p(Math.max(0, v - fMin)) - logMin) / (logMax - logMin)
        : 0;

      // Viridis-ish palette: dark blue → teal → yellow
      const t = Math.max(0, Math.min(1, vNorm));
      let r: number, g: number, b: number;
      if (t < 0.25) {
        const s = t / 0.25;
        r = Math.round(68 + s * (59 - 68));
        g = Math.round(1 + s * (82 - 1));
        b = Math.round(84 + s * (139 - 84));
      } else if (t < 0.5) {
        const s = (t - 0.25) / 0.25;
        r = Math.round(59 + s * (33 - 59));
        g = Math.round(82 + s * (145 - 82));
        b = Math.round(139 + s * (140 - 139));
      } else if (t < 0.75) {
        const s = (t - 0.5) / 0.25;
        r = Math.round(33 + s * (94 - 33));
        g = Math.round(145 + s * (201 - 145));
        b = Math.round(140 + s * (98 - 140));
      } else {
        const s = (t - 0.75) / 0.25;
        r = Math.round(94 + s * (253 - 94));
        g = Math.round(201 + s * (231 - 201));
        b = Math.round(98 + s * (37 - 98));
      }

      const idx = (py * N + px) * 4;
      imgData.data[idx] = r;
      imgData.data[idx + 1] = g;
      imgData.data[idx + 2] = b;
      imgData.data[idx + 3] = 255;
    }
  }

  heatmapCache = imgData;
  heatmapSurfaceIdx = surfaceIdx;
}

// ─── Draw contour canvas ──────────────────────────────────────────────────────

function drawCanvas(): void {
  if (heatmapSurfaceIdx !== surfaceIdx || heatmapCache === null) buildHeatmap();
  ctx.putImageData(heatmapCache!, 0, 0);

  const surf = getSurface();

  // Draw known minima
  if (surf.minima) {
    surf.minima.forEach(m => {
      const px = domToCanvas(m.x, surf.domain);
      const py = CANVAS_SIZE - domToCanvas(m.y, surf.domain);
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, 2 * Math.PI);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.fill();
    });
  }

  // Draw paths for each optimizer
  optimizers.forEach(opt => {
    if (!opt.enabled || opt.path.length < 2) return;
    ctx.strokeStyle = opt.color;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    const p0 = opt.path[0];
    ctx.moveTo(domToCanvas(p0.x, surf.domain), CANVAS_SIZE - domToCanvas(p0.y, surf.domain));
    for (let i = 1; i < opt.path.length; i++) {
      const p = opt.path[i];
      ctx.lineTo(domToCanvas(p.x, surf.domain), CANVAS_SIZE - domToCanvas(p.y, surf.domain));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Current point
    const last = opt.path[opt.path.length - 1];
    const cx = domToCanvas(last.x, surf.domain);
    const cy = CANVAS_SIZE - domToCanvas(last.y, surf.domain);
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
    ctx.fillStyle = opt.color;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.stroke();
  });

  // Draw start point
  if (startPoint) {
    const sx = domToCanvas(startPoint.x, surf.domain);
    const sy = CANVAS_SIZE - domToCanvas(startPoint.y, surf.domain);
    ctx.beginPath();
    ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.fill();
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

// ─── Draw loss chart ──────────────────────────────────────────────────────────

function drawChart(): void {
  svgChart.selectAll('*').remove();

  const maxLen = optimizers.reduce((m, o) => Math.max(m, o.losses.length), 0);
  if (maxLen === 0) return;

  let allLoss: number[] = [];
  optimizers.forEach(o => { if (o.enabled) allLoss = allLoss.concat(o.losses); });
  if (allLoss.length === 0) return;

  const lossMax = d3.max(allLoss) ?? 1;
  const lossMin = Math.max(0, d3.min(allLoss) ?? 0);

  const xScale = d3.scaleLinear().domain([0, maxLen - 1]).range([40, CHART_W - 10]);
  const yScale = d3.scaleLinear()
    .domain([lossMin, lossMax])
    .range([CHART_H, 0]);

  // Axes
  const xAxis = d3.axisBottom(xScale).ticks(6);
  const yAxis = d3.axisLeft(yScale).ticks(4).tickFormat(d3.format('.2e'));
  svgChart.append('g').attr('transform', `translate(0,${CHART_H})`).call(xAxis);
  svgChart.append('g').attr('transform', 'translate(40,0)').call(yAxis);

  optimizers.forEach(opt => {
    if (!opt.enabled || opt.losses.length < 2) return;
    const line = d3.line<number>()
      .x((_, i) => xScale(i))
      .y(v => yScale(v));
    svgChart.append('path')
      .datum(opt.losses)
      .attr('fill', 'none')
      .attr('stroke', opt.color)
      .attr('stroke-width', 1.5)
      .attr('d', line);
  });
}

// ─── Optimizer reset / init ───────────────────────────────────────────────────

function resetOptimizers(): void {
  optimizers = optimizers.map((opt, i) => {
    const enabled = opt.enabled;
    const fresh = makeOptimizer(OPTIMIZER_DEFS[i]);
    fresh.enabled = enabled;
    fresh.path.push({ ...startPoint });
    fresh.losses.push(getSurface().f(startPoint.x, startPoint.y));
    return fresh;
  });
}

// ─── Step all optimizers ──────────────────────────────────────────────────────

function stepAll(): void {
  const surf = getSurface();
  optimizers.forEach(opt => {
    if (!opt.enabled) return;
    if (opt.path.length >= maxSteps) return;
    stepOptimizer(opt, surf, params);
  });
}

// ─── Animation loop ───────────────────────────────────────────────────────────

function frame(): void {
  const maxLen = optimizers.reduce((m, o) => o.enabled ? Math.max(m, o.path.length) : m, 0);
  if (maxLen >= maxSteps) {
    playing = false;
    updateButtons();
    return;
  }
  for (let i = 0; i < speed; i++) stepAll();
  drawCanvas();
  drawChart();
  if (playing) animId = requestAnimationFrame(frame);
}

// ─── UI helpers ──────────────────────────────────────────────────────────────

function updateButtons(): void {
  (document.getElementById('btn-play') as HTMLButtonElement).disabled = playing;
  (document.getElementById('btn-pause') as HTMLButtonElement).disabled = !playing;
}

function initStart(): void {
  playing = false;
  if (animId !== null) { cancelAnimationFrame(animId); animId = null; }
  resetOptimizers();
  drawCanvas();
  drawChart();
  updateButtons();
}

// ─── Event listeners ─────────────────────────────────────────────────────────

surfSel.on('change', function () {
  surfaceIdx = +(this as HTMLSelectElement).value;
  heatmapCache = null;
  initStart();
});

lrInput.on('change', function () {
  params.lr = +(this as HTMLInputElement).value;
  initStart();
});
b1Input.on('change', function () {
  params.beta1 = +(this as HTMLInputElement).value;
  initStart();
});
b2Input.on('change', function () {
  params.beta2 = +(this as HTMLInputElement).value;
  initStart();
});
speedInput.on('input', function () {
  speed = +(this as HTMLInputElement).value;
  speedLabel.text(` ${speed}`);
});
maxStepsInput.on('change', function () {
  maxSteps = +(this as HTMLInputElement).value;
  initStart();
});

// Checkbox changes
checkboxDiv.selectAll('input[type=checkbox]').on('change', function () {
  const el = this as HTMLInputElement;
  const idx = +el.getAttribute('data-idx')!;
  optimizers[idx].enabled = el.checked;
  drawCanvas();
  drawChart();
});

document.getElementById('btn-play')!.addEventListener('click', () => {
  if (!playing) {
    playing = true;
    updateButtons();
    animId = requestAnimationFrame(frame);
  }
});
document.getElementById('btn-pause')!.addEventListener('click', () => {
  playing = false;
  if (animId !== null) { cancelAnimationFrame(animId); animId = null; }
  updateButtons();
});
document.getElementById('btn-step')!.addEventListener('click', () => {
  if (!playing) {
    stepAll();
    drawCanvas();
    drawChart();
  }
});
document.getElementById('btn-reset')!.addEventListener('click', () => {
  initStart();
});

// Click on canvas to set start point
canvasEl.addEventListener('click', (e: MouseEvent) => {
  const rect = canvasEl.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  const surf = getSurface();
  startPoint = {
    x: canvasToDom(px, surf.domain),
    y: canvasToDom(CANVAS_SIZE - py, surf.domain)
  };
  initStart();
});

// ─── Boot ─────────────────────────────────────────────────────────────────────

initStart();
