import * as d3 from 'd3';

// ── Kernel types ────────────────────────────────────────────────────────────

type KernelType = 'rbf' | 'matern32' | 'periodic';

interface HyperParams {
  lengthscale: number;
  signalVar: number;
  noiseVar: number;
  period: number; // only for periodic
}

function kernelRBF(x1: number, x2: number, hp: HyperParams): number {
  const r = (x1 - x2) / hp.lengthscale;
  return hp.signalVar * Math.exp(-0.5 * r * r);
}

function kernelMatern32(x1: number, x2: number, hp: HyperParams): number {
  const r = Math.abs(x1 - x2) / hp.lengthscale;
  return hp.signalVar * (1 + Math.sqrt(3) * r) * Math.exp(-Math.sqrt(3) * r);
}

function kernelPeriodic(x1: number, x2: number, hp: HyperParams): number {
  const diff = Math.abs(x1 - x2);
  const s = Math.sin(Math.PI * diff / hp.period);
  return hp.signalVar * Math.exp(-2 * s * s / (hp.lengthscale * hp.lengthscale));
}

function evalKernel(k: KernelType, x1: number, x2: number, hp: HyperParams): number {
  if (k === 'rbf') return kernelRBF(x1, x2, hp);
  if (k === 'matern32') return kernelMatern32(x1, x2, hp);
  return kernelPeriodic(x1, x2, hp);
}

// ── Cholesky decomposition (lower triangular L s.t. A = L L^T) ─────────────

function cholesky(A: number[][]): number[][] | null {
  const n = A.length;
  const L: number[][] = Array.from({length: n}, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
      if (i === j) {
        const val = A[i][i] - sum;
        if (val <= 0) return null;
        L[i][j] = Math.sqrt(val);
      } else {
        L[i][j] = (A[i][j] - sum) / L[j][j];
      }
    }
  }
  return L;
}

// Solve L x = b (forward substitution)
function solveL(L: number[][], b: number[]): number[] {
  const n = b.length;
  const x = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) sum += L[i][j] * x[j];
    x[i] = (b[i] - sum) / L[i][i];
  }
  return x;
}

// Solve L^T x = b (back substitution)
function solveLT(L: number[][], b: number[]): number[] {
  const n = b.length;
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) sum += L[j][i] * x[j];
    x[i] = (b[i] - sum) / L[i][i];
  }
  return x;
}

// ── GP Posterior ─────────────────────────────────────────────────────────────

interface GPPosterior {
  mean: number[];
  variance: number[];
  // L and alpha stored for sampling
  L: number[][];
  alpha: number[];
  xTrain: number[];
  kernel: KernelType;
  hp: HyperParams;
  xTest: number[];
}

function computeGPPosterior(
  xTrain: number[], yTrain: number[],
  xTest: number[],
  kernel: KernelType, hp: HyperParams
): GPPosterior | null {
  const n = xTrain.length;
  const m = xTest.length;

  // Build K(X,X) + sigma^2 I
  const K: number[][] = Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (_, j) =>
      evalKernel(kernel, xTrain[i], xTrain[j], hp) + (i === j ? hp.noiseVar : 0)
    )
  );

  const L = cholesky(K);
  if (!L) return null;

  // alpha = K^{-1} y  via L L^T alpha = y
  const alpha = solveLT(L, solveL(L, yTrain));

  // K_* (n x m): kernel between training and test
  const Kstar: number[][] = Array.from({length: n}, (_, i) =>
    Array.from({length: m}, (_, j) => evalKernel(kernel, xTrain[i], xTest[j], hp))
  );

  // Posterior mean: K_*^T alpha  (m-vector)
  const mean = Array.from({length: m}, (_, j) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += Kstar[i][j] * alpha[i];
    return s;
  });

  // V = L^{-1} K_*  (n x m)  via forward substitution column by column
  const V: number[][] = Array.from({length: m}, (_, j) => {
    const col = Array.from({length: n}, (_, i) => Kstar[i][j]);
    return solveL(L, col);
  });

  // Posterior variance: k(x*,x*) - v^T v
  const variance = Array.from({length: m}, (_, j) => {
    const kss = evalKernel(kernel, xTest[j], xTest[j], hp);
    let vTv = 0;
    for (let i = 0; i < n; i++) vTv += V[j][i] * V[j][i];
    return Math.max(0, kss - vTv);
  });

  return {mean, variance, L, alpha, xTrain, kernel, hp, xTest};
}

// ── Posterior sampling ────────────────────────────────────────────────────────

function samplePrior(xTest: number[], kernel: KernelType, hp: HyperParams): number[] {
  const m = xTest.length;
  const K: number[][] = Array.from({length: m}, (_, i) =>
    Array.from({length: m}, (_, j) =>
      evalKernel(kernel, xTest[i], xTest[j], hp) + (i === j ? 1e-9 : 0)
    )
  );
  const L = cholesky(K);
  if (!L) return new Array(m).fill(0);
  // sample z ~ N(0,I), return L z
  const z = Array.from({length: m}, () => randn());
  return Array.from({length: m}, (_, i) => {
    let s = 0;
    for (let j = 0; j <= i; j++) s += L[i][j] * z[j];
    return s;
  });
}

function samplePosterior(gp: GPPosterior): number[] {
  const m = gp.xTest.length;
  // Compute K_** (test covariance)
  const Kss: number[][] = Array.from({length: m}, (_, i) =>
    Array.from({length: m}, (_, j) =>
      evalKernel(gp.kernel, gp.xTest[i], gp.xTest[j], gp.hp) + (i === j ? 1e-9 : 0)
    )
  );
  // K_* (training x test)
  const n = gp.xTrain.length;
  const Kstar: number[][] = Array.from({length: n}, (_, i) =>
    Array.from({length: m}, (_, j) => evalKernel(gp.kernel, gp.xTrain[i], gp.xTest[j], gp.hp))
  );
  // V = L^{-1} K_*
  const V: number[][] = Array.from({length: m}, (_, j) => {
    const col = Array.from({length: n}, (_, i) => Kstar[i][j]);
    return solveL(gp.L, col);
  });
  // Posterior cov = K_** - V^T V
  const covPost: number[][] = Array.from({length: m}, (_, i) =>
    Array.from({length: m}, (_, j) => {
      let vtv = 0;
      for (let k = 0; k < n; k++) vtv += V[i][k] * V[j][k];
      return Kss[i][j] - vtv;
    })
  );
  // Add small jitter for numerical stability
  for (let i = 0; i < m; i++) covPost[i][i] += 1e-8;
  const Lp = cholesky(covPost);
  if (!Lp) return gp.mean.slice();
  const z = Array.from({length: m}, () => randn());
  return Array.from({length: m}, (_, i) => {
    let s = gp.mean[i];
    for (let j = 0; j <= i; j++) s += Lp[i][j] * z[j];
    return s;
  });
}

function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ── Target functions ──────────────────────────────────────────────────────────

type TargetFn = (x: number) => number;
const TARGETS: Record<string, TargetFn> = {
  sine: (x) => Math.sin(2 * Math.PI * x),
  step: (x) => (x > 0 ? 1 : -1),
  sinc: (x) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x)),
};

// ── App State ─────────────────────────────────────────────────────────────────

interface State {
  xTrain: number[];
  yTrain: number[];
  kernel: KernelType;
  hp: HyperParams;
  showSamples: boolean;
  numSamples: number;
  target: string;
  posterior: GPPosterior | null;
  samples: number[][];
}

const state: State = {
  xTrain: [],
  yTrain: [],
  kernel: 'rbf',
  hp: {lengthscale: 0.5, signalVar: 1.0, noiseVar: 0.05, period: 1.0},
  showSamples: false,
  numSamples: 5,
  target: 'sine',
  posterior: null,
  samples: [],
};

// ── Layout constants ──────────────────────────────────────────────────────────

const MARGIN = {top: 30, right: 30, bottom: 50, left: 55};
const N_TEST = 200;
const X_MIN = -3;
const X_MAX = 3;
const Y_MIN = -3.5;
const Y_MAX = 3.5;

// ── D3 setup ──────────────────────────────────────────────────────────────────

let svgWidth = 700;
let svgHeight = 420;

const svg = d3.select('#gp-plot')
  .attr('width', svgWidth)
  .attr('height', svgHeight);

const plotW = svgWidth - MARGIN.left - MARGIN.right;
const plotH = svgHeight - MARGIN.top - MARGIN.bottom;

const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

const xScale = d3.scaleLinear().domain([X_MIN, X_MAX]).range([0, plotW]);
const yScale = d3.scaleLinear().domain([Y_MIN, Y_MAX]).range([plotH, 0]);

// Grid
g.append('g').attr('class', 'axis x-axis').attr('transform', `translate(0,${plotH})`)
  .call(d3.axisBottom(xScale).ticks(7));
g.append('g').attr('class', 'axis y-axis')
  .call(d3.axisLeft(yScale).ticks(7));

// Axis labels
g.append('text').attr('class', 'axis-label')
  .attr('x', plotW / 2).attr('y', plotH + 42)
  .attr('text-anchor', 'middle').text('x');
g.append('text').attr('class', 'axis-label')
  .attr('transform', 'rotate(-90)')
  .attr('x', -plotH / 2).attr('y', -44)
  .attr('text-anchor', 'middle').text('y');

// Zero line
g.append('line').attr('class', 'zero-line')
  .attr('x1', 0).attr('x2', plotW)
  .attr('y1', yScale(0)).attr('y2', yScale(0));

// Clip path
svg.append('defs').append('clipPath').attr('id', 'clip')
  .append('rect').attr('width', plotW).attr('height', plotH);

const plotArea = g.append('g').attr('clip-path', 'url(#clip)');

// Confidence band
const confidenceBand = plotArea.append('path').attr('class', 'confidence-band');

// Sample paths group
const samplesGroup = plotArea.append('g').attr('class', 'samples-group');

// Mean line
const meanLine = plotArea.append('path').attr('class', 'mean-line');

// Training points
const pointsGroup = plotArea.append('g').attr('class', 'points-group');

// Invisible overlay for clicks
const overlay = g.append('rect')
  .attr('class', 'overlay')
  .attr('width', plotW)
  .attr('height', plotH)
  .style('fill', 'transparent')
  .style('cursor', 'crosshair');

// ── Render ────────────────────────────────────────────────────────────────────

function getXTest(): number[] {
  return Array.from({length: N_TEST}, (_, i) => X_MIN + i * (X_MAX - X_MIN) / (N_TEST - 1));
}

function update(): void {
  const xTest = getXTest();

  if (state.xTrain.length === 0) {
    // Prior: zero mean, show prior variance
    const priorVar = state.hp.signalVar;
    const upper = xTest.map(x => yScale(Math.sqrt(priorVar) * 2));
    const lower = xTest.map(x => yScale(-Math.sqrt(priorVar) * 2));

    const areaData = xTest.map((x, i) => ({
      x: xScale(x),
      y0: lower[i],
      y1: upper[i],
    }));

    const bandPath = d3.area<{x: number; y0: number; y1: number}>()
      .x(d => d.x).y0(d => d.y0).y1(d => d.y1)(areaData) || '';
    confidenceBand.attr('d', bandPath);
    meanLine.attr('d', d3.line<number>().x(x => xScale(x)).y(() => yScale(0))(xTest) || '');
    state.posterior = null;
    state.samples = [];

    if (state.showSamples) {
      const newSamples: number[][] = [];
      for (let s = 0; s < state.numSamples; s++) {
        newSamples.push(samplePrior(xTest, state.kernel, state.hp));
      }
      state.samples = newSamples;
    }
  } else {
    const post = computeGPPosterior(
      state.xTrain, state.yTrain, xTest, state.kernel, state.hp
    );
    state.posterior = post;

    if (post) {
      // Confidence band
      const areaData = xTest.map((x, i) => ({
        x: xScale(x),
        y0: yScale(post.mean[i] - 2 * Math.sqrt(post.variance[i])),
        y1: yScale(post.mean[i] + 2 * Math.sqrt(post.variance[i])),
      }));
      const bandPath = d3.area<{x: number; y0: number; y1: number}>()
        .x(d => d.x).y0(d => d.y0).y1(d => d.y1)(areaData) || '';
      confidenceBand.attr('d', bandPath);

      // Mean line
      meanLine.attr('d', d3.line<number>()
        .x((_, i) => xScale(xTest[i]))
        .y((_, i) => yScale(post.mean[i]))(xTest) || '');

      // Posterior samples
      if (state.showSamples) {
        const newSamples: number[][] = [];
        for (let s = 0; s < state.numSamples; s++) {
          newSamples.push(samplePosterior(post));
        }
        state.samples = newSamples;
      } else {
        state.samples = [];
      }
    }
  }

  renderSamples(xTest);
  renderPoints();
  updateKernelInfo();
}

const SAMPLE_COLORS = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];

function renderSamples(xTest: number[]): void {
  const paths = samplesGroup.selectAll<SVGPathElement, number[]>('.sample-path')
    .data(state.showSamples ? state.samples : []);
  paths.enter().append('path').attr('class', 'sample-path')
    .merge(paths)
    .attr('d', (sampleY, i) =>
      d3.line<number>()
        .x((_, j) => xScale(xTest[j]))
        .y((v) => yScale(v))(sampleY) || ''
    )
    .style('stroke', (_, i) => SAMPLE_COLORS[i % SAMPLE_COLORS.length]);
  paths.exit().remove();
}

function renderPoints(): void {
  const pts = pointsGroup.selectAll<SVGCircleElement, [number, number]>('.train-point')
    .data(state.xTrain.map((x, i) => [x, state.yTrain[i]] as [number, number]));
  pts.enter().append('circle').attr('class', 'train-point')
    .merge(pts)
    .attr('cx', d => xScale(d[0]))
    .attr('cy', d => yScale(d[1]))
    .attr('r', 5);
  pts.exit().remove();
}

function updateKernelInfo(): void {
  const el = document.getElementById('kernel-info');
  if (!el) return;
  el.textContent =
    `Kernel: ${state.kernel} | ℓ=${state.hp.lengthscale.toFixed(2)} | σ²=${state.hp.signalVar.toFixed(2)} | noise=${state.hp.noiseVar.toFixed(3)}` +
    (state.kernel === 'periodic' ? ` | T=${state.hp.period.toFixed(2)}` : '');
}

// ── Click handler ─────────────────────────────────────────────────────────────

overlay.on('click', function(event: MouseEvent) {
  const [mx, my] = d3.pointer(event, this);
  const x = xScale.invert(mx);
  const y = yScale.invert(my);
  if (x < X_MIN || x > X_MAX || y < Y_MIN || y > Y_MAX) return;
  if (state.xTrain.length >= 60) return;
  state.xTrain.push(x);
  state.yTrain.push(y);
  update();
});

// ── Controls wiring ───────────────────────────────────────────────────────────

function wireSlider(id: string, key: keyof HyperParams, transform?: (v: number) => number): void {
  const slider = document.getElementById(id) as HTMLInputElement | null;
  const display = document.getElementById(id + '-val');
  if (!slider) return;
  slider.addEventListener('input', () => {
    const raw = parseFloat(slider.value);
    const val = transform ? transform(raw) : raw;
    (state.hp as any)[key] = val;
    if (display) display.textContent = val.toFixed(3);
    update();
  });
}

function initControls(): void {
  // Kernel selector
  const kernelSel = document.getElementById('kernel-select') as HTMLSelectElement | null;
  if (kernelSel) {
    kernelSel.addEventListener('change', () => {
      state.kernel = kernelSel.value as KernelType;
      const periodicRow = document.getElementById('period-row');
      if (periodicRow) periodicRow.style.display = state.kernel === 'periodic' ? 'flex' : 'none';
      update();
    });
  }

  wireSlider('ls-slider', 'lengthscale', v => Math.exp(v));
  wireSlider('sv-slider', 'signalVar', v => Math.exp(v));
  wireSlider('nv-slider', 'noiseVar', v => Math.exp(v));
  wireSlider('period-slider', 'period', v => Math.exp(v));

  // Show samples toggle
  const samplesToggle = document.getElementById('show-samples') as HTMLInputElement | null;
  if (samplesToggle) {
    samplesToggle.addEventListener('change', () => {
      state.showSamples = samplesToggle.checked;
      update();
    });
  }

  // Clear button
  const clearBtn = document.getElementById('clear-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      state.xTrain = [];
      state.yTrain = [];
      update();
    });
  }

  // Target selector
  const targetSel = document.getElementById('target-select') as HTMLSelectElement | null;
  if (targetSel) {
    targetSel.addEventListener('change', () => {
      state.target = targetSel.value;
    });
  }

  // Sample from target
  const sampleBtn = document.getElementById('sample-target-btn');
  if (sampleBtn) {
    sampleBtn.addEventListener('click', () => {
      const fn = TARGETS[state.target];
      if (!fn) return;
      // Sample 15 points uniformly with noise
      const n = 15;
      state.xTrain = [];
      state.yTrain = [];
      for (let i = 0; i < n; i++) {
        const x = X_MIN + (i + 0.5) * (X_MAX - X_MIN) / n + (Math.random() - 0.5) * 0.3;
        const noise = randn() * Math.sqrt(state.hp.noiseVar);
        state.xTrain.push(x);
        state.yTrain.push(fn(x) + noise);
      }
      update();
    });
  }

  // Resample samples button
  const resampleBtn = document.getElementById('resample-btn');
  if (resampleBtn) {
    resampleBtn.addEventListener('click', () => {
      update();
    });
  }

  // Init slider displays
  function initSliderDisplay(id: string, val: number): void {
    const display = document.getElementById(id + '-val');
    if (display) display.textContent = val.toFixed(3);
    const slider = document.getElementById(id) as HTMLInputElement | null;
    if (slider) slider.value = String(Math.log(val));
  }
  initSliderDisplay('ls-slider', state.hp.lengthscale);
  initSliderDisplay('sv-slider', state.hp.signalVar);
  initSliderDisplay('nv-slider', state.hp.noiseVar);
  initSliderDisplay('period-slider', state.hp.period);
}

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initControls();
  update();
});
