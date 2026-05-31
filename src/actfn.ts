import * as d3 from "d3";

export {};

// ── Activation function definitions ──────────────────────────────────────────

interface ActFn {
  name: string;
  fn: (x: number, p: Params) => number;
  dfn: (x: number, p: Params) => number;
  color: string;
  group: string;
}

interface Params {
  preluAlpha: number;
  swishBeta: number;
  eluAlpha: number;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function erf(x: number): number {
  // Abramowitz & Stegun approximation
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const val = 1 - poly * Math.exp(-x * x);
  return x >= 0 ? val : -val;
}

// GELU using erf approximation
function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

function geluD(x: number): number {
  const phi = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  const Phi = 0.5 * (1 + erf(x / Math.SQRT2));
  return Phi + x * phi;
}

// Mish: x * tanh(softplus(x))
function mish(x: number): number {
  const sp = Math.log(1 + Math.exp(x));
  return x * Math.tanh(sp);
}

function mishD(x: number): number {
  const ex = Math.exp(x);
  const sp = Math.log(1 + ex);
  const th = Math.tanh(sp);
  const sech2 = 1 - th * th;
  return th + x * sech2 * (ex / (1 + ex));
}

// Sinc
function sinc(x: number): number {
  if (Math.abs(x) < 1e-9) return 1;
  return Math.sin(x) / x;
}

function sincD(x: number): number {
  if (Math.abs(x) < 1e-9) return 0;
  return (Math.cos(x) * x - Math.sin(x)) / (x * x);
}

// Gaussian
function gaussian(x: number): number {
  return Math.exp(-x * x);
}

function gaussianD(x: number): number {
  return -2 * x * Math.exp(-x * x);
}

// Snake: x + sin^2(x)
function snake(x: number): number {
  return x + Math.sin(x) * Math.sin(x);
}

function snakeD(x: number): number {
  return 1 + Math.sin(2 * x);
}

// Bent Identity
function bentIdentity(x: number): number {
  return (Math.sqrt(x * x + 1) - 1) / 2 + x;
}

function bentIdentityD(x: number): number {
  return x / (2 * Math.sqrt(x * x + 1)) + 1;
}

// Softplus
function softplus(x: number): number {
  if (x > 20) return x;
  return Math.log(1 + Math.exp(x));
}

function softplusD(x: number): number {
  return sigmoid(x);
}

// Softsign
function softsign(x: number): number {
  return x / (1 + Math.abs(x));
}

function softsignD(x: number): number {
  const d = 1 + Math.abs(x);
  return 1 / (d * d);
}

// LogSigmoid
function logSigmoid(x: number): number {
  if (x >= 0) return -Math.log(1 + Math.exp(-x));
  return x - Math.log(1 + Math.exp(x));
}

function logSigmoidD(x: number): number {
  return 1 - sigmoid(x);
}

// Hard Sigmoid
function hardSigmoid(x: number): number {
  if (x <= -3) return 0;
  if (x >= 3) return 1;
  return x / 6 + 0.5;
}

function hardSigmoidD(x: number): number {
  if (x <= -3 || x >= 3) return 0;
  return 1 / 6;
}

// Hard Tanh
function hardTanh(x: number): number {
  if (x < -1) return -1;
  if (x > 1) return 1;
  return x;
}

function hardTanhD(x: number): number {
  if (x < -1 || x > 1) return 0;
  return 1;
}

// Hard Swish
function hardSwish(x: number): number {
  if (x <= -3) return 0;
  if (x >= 3) return x;
  return x * (x + 3) / 6;
}

function hardSwishD(x: number): number {
  if (x <= -3) return 0;
  if (x >= 3) return 1;
  return (2 * x + 3) / 6;
}

// ReLU6
function relu6(x: number): number {
  return Math.min(Math.max(0, x), 6);
}

function relu6D(x: number): number {
  if (x <= 0 || x >= 6) return 0;
  return 1;
}

// Tanhshrink
function tanhshrink(x: number): number {
  return x - Math.tanh(x);
}

function tanhshrinkD(x: number): number {
  const t = Math.tanh(x);
  return 1 - (1 - t * t);
}

// ArcTan
function arctan(x: number): number {
  return Math.atan(x);
}

function arctanD(x: number): number {
  return 1 / (1 + x * x);
}

// Bipolar Sigmoid: (1-e^-x)/(1+e^-x)
function bipolarSigmoid(x: number): number {
  const ex = Math.exp(-x);
  return (1 - ex) / (1 + ex);
}

function bipolarSigmoidD(x: number): number {
  const s = bipolarSigmoid(x);
  return 0.5 * (1 - s * s);
}

// CELU
function celu(x: number, alpha: number): number {
  if (x >= 0) return x;
  return alpha * (Math.exp(x / alpha) - 1);
}

function celuD(x: number, alpha: number): number {
  if (x >= 0) return 1;
  return Math.exp(x / alpha);
}

// SELU constants
const SELU_LAMBDA = 1.0507009873554804934193349852946;
const SELU_ALPHA = 1.6732632423543772848170429916717;

function selu(x: number): number {
  if (x >= 0) return SELU_LAMBDA * x;
  return SELU_LAMBDA * SELU_ALPHA * (Math.exp(x) - 1);
}

function seluD(x: number): number {
  if (x >= 0) return SELU_LAMBDA;
  return SELU_LAMBDA * SELU_ALPHA * Math.exp(x);
}

const ACTIVATION_FUNCTIONS: ActFn[] = [
  {
    name: "ReLU", group: "ReLU family", color: "#f97316",
    fn: (x) => Math.max(0, x),
    dfn: (x) => x > 0 ? 1 : 0,
  },
  {
    name: "Leaky ReLU", group: "ReLU family", color: "#fb923c",
    fn: (x, p) => x >= 0 ? x : 0.01 * x,
    dfn: (x) => x >= 0 ? 1 : 0.01,
  },
  {
    name: "PReLU", group: "ReLU family", color: "#fdba74",
    fn: (x, p) => x >= 0 ? x : p.preluAlpha * x,
    dfn: (x, p) => x >= 0 ? 1 : p.preluAlpha,
  },
  {
    name: "ELU", group: "ReLU family", color: "#a78bfa",
    fn: (x, p) => x >= 0 ? x : p.eluAlpha * (Math.exp(x) - 1),
    dfn: (x, p) => x >= 0 ? 1 : p.eluAlpha * Math.exp(x),
  },
  {
    name: "SELU", group: "ReLU family", color: "#c4b5fd",
    fn: (x) => selu(x),
    dfn: (x) => seluD(x),
  },
  {
    name: "CELU", group: "ReLU family", color: "#ddd6fe",
    fn: (x, p) => celu(x, p.eluAlpha),
    dfn: (x, p) => celuD(x, p.eluAlpha),
  },
  {
    name: "ReLU6", group: "ReLU family", color: "#ea580c",
    fn: (x) => relu6(x),
    dfn: (x) => relu6D(x),
  },
  {
    name: "GELU", group: "Smooth", color: "#22d3ee",
    fn: (x) => gelu(x),
    dfn: (x) => geluD(x),
  },
  {
    name: "Swish/SiLU", group: "Smooth", color: "#06b6d4",
    fn: (x, p) => x * sigmoid(p.swishBeta * x),
    dfn: (x, p) => {
      const s = sigmoid(p.swishBeta * x);
      return s + p.swishBeta * x * s * (1 - s);
    },
  },
  {
    name: "Mish", group: "Smooth", color: "#0ea5e9",
    fn: (x) => mish(x),
    dfn: (x) => mishD(x),
  },
  {
    name: "Softplus", group: "Smooth", color: "#38bdf8",
    fn: (x) => softplus(x),
    dfn: (x) => softplusD(x),
  },
  {
    name: "Softsign", group: "Smooth", color: "#7dd3fc",
    fn: (x) => softsign(x),
    dfn: (x) => softsignD(x),
  },
  {
    name: "Sigmoid", group: "Saturating", color: "#4ade80",
    fn: (x) => sigmoid(x),
    dfn: (x) => { const s = sigmoid(x); return s * (1 - s); },
  },
  {
    name: "Tanh", group: "Saturating", color: "#22c55e",
    fn: (x) => Math.tanh(x),
    dfn: (x) => { const t = Math.tanh(x); return 1 - t * t; },
  },
  {
    name: "Hard Sigmoid", group: "Saturating", color: "#86efac",
    fn: (x) => hardSigmoid(x),
    dfn: (x) => hardSigmoidD(x),
  },
  {
    name: "Hard Tanh", group: "Saturating", color: "#bbf7d0",
    fn: (x) => hardTanh(x),
    dfn: (x) => hardTanhD(x),
  },
  {
    name: "Hard Swish", group: "Saturating", color: "#6ee7b7",
    fn: (x) => hardSwish(x),
    dfn: (x) => hardSwishD(x),
  },
  {
    name: "Bipolar Sigmoid", group: "Saturating", color: "#34d399",
    fn: (x) => bipolarSigmoid(x),
    dfn: (x) => bipolarSigmoidD(x),
  },
  {
    name: "LogSigmoid", group: "Saturating", color: "#a7f3d0",
    fn: (x) => logSigmoid(x),
    dfn: (x) => logSigmoidD(x),
  },
  {
    name: "ArcTan", group: "Saturating", color: "#059669",
    fn: (x) => arctan(x),
    dfn: (x) => arctanD(x),
  },
  {
    name: "Sine", group: "Oscillatory", color: "#f43f5e",
    fn: (x) => Math.sin(x),
    dfn: (x) => Math.cos(x),
  },
  {
    name: "Sinc", group: "Oscillatory", color: "#fb7185",
    fn: (x) => sinc(x),
    dfn: (x) => sincD(x),
  },
  {
    name: "Gaussian", group: "Oscillatory", color: "#fda4af",
    fn: (x) => gaussian(x),
    dfn: (x) => gaussianD(x),
  },
  {
    name: "Snake", group: "Oscillatory", color: "#e11d48",
    fn: (x) => snake(x),
    dfn: (x) => snakeD(x),
  },
  {
    name: "Bent Identity", group: "Smooth", color: "#fbbf24",
    fn: (x) => bentIdentity(x),
    dfn: (x) => bentIdentityD(x),
  },
  {
    name: "Tanhshrink", group: "Smooth", color: "#f59e0b",
    fn: (x) => tanhshrink(x),
    dfn: (x) => tanhshrinkD(x),
  },
];

// ── Properties table data ─────────────────────────────────────────────────────

interface Props {
  range: string;
  monotonic: boolean;
  saturating: boolean;
  zeroCentered: boolean;
  smooth: boolean;
  deadReLU: boolean;
}

const PROPERTIES: Record<string, Props> = {
  "ReLU":           { range: "[0, ∞)", monotonic: true,  saturating: false, zeroCentered: false, smooth: false, deadReLU: true  },
  "Leaky ReLU":     { range: "(-∞, ∞)", monotonic: true,  saturating: false, zeroCentered: false, smooth: false, deadReLU: false },
  "PReLU":          { range: "(-∞, ∞)", monotonic: true,  saturating: false, zeroCentered: false, smooth: false, deadReLU: false },
  "ELU":            { range: "(-α, ∞)", monotonic: true,  saturating: false, zeroCentered: false, smooth: false, deadReLU: false },
  "SELU":           { range: "(-λα, ∞)", monotonic: true,  saturating: false, zeroCentered: false, smooth: false, deadReLU: false },
  "CELU":           { range: "(-α, ∞)", monotonic: true,  saturating: false, zeroCentered: false, smooth: false, deadReLU: false },
  "ReLU6":          { range: "[0, 6]",  monotonic: true,  saturating: true,  zeroCentered: false, smooth: false, deadReLU: true  },
  "GELU":           { range: "≈(-0.17,∞)", monotonic: false, saturating: false, zeroCentered: false, smooth: true,  deadReLU: false },
  "Swish/SiLU":     { range: "≈(-0.28,∞)", monotonic: false, saturating: false, zeroCentered: false, smooth: true,  deadReLU: false },
  "Mish":           { range: "≈(-0.31,∞)", monotonic: false, saturating: false, zeroCentered: false, smooth: true,  deadReLU: false },
  "Softplus":       { range: "(0, ∞)", monotonic: true,  saturating: false, zeroCentered: false, smooth: true,  deadReLU: false },
  "Softsign":       { range: "(-1, 1)", monotonic: true,  saturating: true,  zeroCentered: true,  smooth: true,  deadReLU: false },
  "Sigmoid":        { range: "(0, 1)", monotonic: true,  saturating: true,  zeroCentered: false, smooth: true,  deadReLU: false },
  "Tanh":           { range: "(-1, 1)", monotonic: true,  saturating: true,  zeroCentered: true,  smooth: true,  deadReLU: false },
  "Hard Sigmoid":   { range: "[0, 1]", monotonic: true,  saturating: true,  zeroCentered: false, smooth: false, deadReLU: false },
  "Hard Tanh":      { range: "[-1, 1]", monotonic: true,  saturating: true,  zeroCentered: true,  smooth: false, deadReLU: false },
  "Hard Swish":     { range: "[0, ∞)", monotonic: false, saturating: false, zeroCentered: false, smooth: false, deadReLU: false },
  "Bipolar Sigmoid":{ range: "(-1, 1)", monotonic: true,  saturating: true,  zeroCentered: true,  smooth: true,  deadReLU: false },
  "LogSigmoid":     { range: "(-∞, 0)", monotonic: true,  saturating: true,  zeroCentered: false, smooth: true,  deadReLU: false },
  "ArcTan":         { range: "(-π/2,π/2)", monotonic: true, saturating: true, zeroCentered: true, smooth: true,  deadReLU: false },
  "Sine":           { range: "[-1, 1]", monotonic: false, saturating: true,  zeroCentered: true,  smooth: true,  deadReLU: false },
  "Sinc":           { range: "[-0.22,1]", monotonic: false, saturating: true, zeroCentered: false, smooth: true, deadReLU: false },
  "Gaussian":       { range: "(0, 1]", monotonic: false, saturating: true,  zeroCentered: false, smooth: true,  deadReLU: false },
  "Snake":          { range: "(-∞, ∞)", monotonic: false, saturating: false, zeroCentered: true,  smooth: true,  deadReLU: false },
  "Bent Identity":  { range: "(-∞, ∞)", monotonic: true,  saturating: false, zeroCentered: true,  smooth: true,  deadReLU: false },
  "Tanhshrink":     { range: "(-∞, ∞)", monotonic: true,  saturating: false, zeroCentered: true,  smooth: true,  deadReLU: false },
};

// ── State ─────────────────────────────────────────────────────────────────────

const state = {
  selected: new Set<string>(["ReLU", "Sigmoid", "Tanh", "GELU"]),
  showDerivatives: true,
  xMin: -5,
  xMax: 5,
  params: { preluAlpha: 0.25, swishBeta: 1.0, eluAlpha: 1.0 } as Params,
  hoverX: null as number | null,
  gradientView: false,
};

// ── Layout constants ──────────────────────────────────────────────────────────

const MARGIN = { top: 30, right: 30, bottom: 50, left: 60 };
const PLOT_W = 620;
const PLOT_H = 420;
const W = PLOT_W + MARGIN.left + MARGIN.right;
const H = PLOT_H + MARGIN.top + MARGIN.bottom;

const N_POINTS = 400;

// ── Build UI ──────────────────────────────────────────────────────────────────

function buildUI(): void {
  const app = d3.select("#app");

  // ── Header ────────────────────────────────────────────────────────────────
  const header = app.append("div").attr("class", "header");
  header.append("h1").text("Activation Function Explorer");
  header.append("p").attr("class", "subtitle")
    .text("Compare 26 activation functions, their derivatives, and learn their properties.");

  // ── Main layout ───────────────────────────────────────────────────────────
  const main = app.append("div").attr("class", "main-layout");

  // Left: controls
  const sidebar = main.append("div").attr("class", "sidebar");

  // Right: chart + panels
  const content = main.append("div").attr("class", "content");

  // ── Controls ──────────────────────────────────────────────────────────────
  buildControls(sidebar);

  // ── Chart ─────────────────────────────────────────────────────────────────
  const chartArea = content.append("div").attr("class", "chart-area");
  buildChart(chartArea);

  // ── Hover info bar ────────────────────────────────────────────────────────
  content.append("div").attr("id", "hover-info").attr("class", "hover-info")
    .text("Hover over the chart to read values");

  // ── Gradient magnitude view ───────────────────────────────────────────────
  const gradArea = content.append("div").attr("class", "grad-area");
  gradArea.append("div").attr("class", "panel-title").text("Gradient Magnitude View");
  gradArea.append("div").attr("class", "grad-subtitle")
    .text("Shows |f′(x)| — near zero means vanishing gradients.");
  buildGradChart(gradArea);

  // ── Properties table ──────────────────────────────────────────────────────
  const tableArea = content.append("div").attr("class", "table-area");
  tableArea.append("div").attr("class", "panel-title").text("Function Properties");
  buildTable(tableArea);
}

function buildControls(parent: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>): void {
  // x-range slider
  const rangeSection = parent.append("div").attr("class", "ctrl-section");
  rangeSection.append("div").attr("class", "ctrl-label").text("X Range");

  const rangeRow = rangeSection.append("div").attr("class", "ctrl-row");
  rangeRow.append("label").text("Min");
  const xMinInput = rangeRow.append("input")
    .attr("type", "range").attr("min", -10).attr("max", -1).attr("step", 0.5)
    .attr("value", state.xMin).attr("id", "xmin-slider");
  rangeRow.append("span").attr("id", "xmin-val").text(state.xMin.toString());

  const rangeRow2 = rangeSection.append("div").attr("class", "ctrl-row");
  rangeRow2.append("label").text("Max");
  const xMaxInput = rangeRow2.append("input")
    .attr("type", "range").attr("min", 1).attr("max", 10).attr("step", 0.5)
    .attr("value", state.xMax).attr("id", "xmax-slider");
  rangeRow2.append("span").attr("id", "xmax-val").text(state.xMax.toString());

  xMinInput.on("input", function() {
    state.xMin = +((this as HTMLInputElement).value);
    d3.select("#xmin-val").text(state.xMin.toString());
    render();
  });
  xMaxInput.on("input", function() {
    state.xMax = +((this as HTMLInputElement).value);
    d3.select("#xmax-val").text(state.xMax.toString());
    render();
  });

  // Parameters
  const paramSection = parent.append("div").attr("class", "ctrl-section");
  paramSection.append("div").attr("class", "ctrl-label").text("Parameters");

  function makeParamSlider(
    label: string, id: string, min: number, max: number, step: number, init: number,
    onChange: (v: number) => void
  ): void {
    const row = paramSection.append("div").attr("class", "ctrl-row");
    row.append("label").text(label);
    row.append("input")
      .attr("type", "range").attr("min", min).attr("max", max).attr("step", step)
      .attr("value", init).attr("id", id)
      .on("input", function() {
        const v = +((this as HTMLInputElement).value);
        d3.select("#" + id + "-val").text(v.toFixed(2));
        onChange(v);
        render();
      });
    row.append("span").attr("id", id + "-val").text(init.toFixed(2));
  }

  makeParamSlider("PReLU α", "prelu-alpha", 0.01, 1, 0.01, state.params.preluAlpha,
    v => { state.params.preluAlpha = v; });
  makeParamSlider("Swish β", "swish-beta", 0.1, 5, 0.1, state.params.swishBeta,
    v => { state.params.swishBeta = v; });
  makeParamSlider("ELU/CELU α", "elu-alpha", 0.1, 3, 0.1, state.params.eluAlpha,
    v => { state.params.eluAlpha = v; });

  // Toggle derivative
  const toggleSection = parent.append("div").attr("class", "ctrl-section");
  const derivRow = toggleSection.append("div").attr("class", "ctrl-row toggle-row");
  const derivCb = derivRow.append("input").attr("type", "checkbox").attr("id", "deriv-toggle");
  (derivCb.node() as HTMLInputElement).checked = state.showDerivatives;
  derivRow.append("label").attr("for", "deriv-toggle").text("Show Derivatives (dashed)");
  derivCb.on("change", function() {
    state.showDerivatives = (this as HTMLInputElement).checked;
    render();
  });

  // Function checkboxes grouped
  parent.append("div").attr("class", "ctrl-label").style("margin-top", "14px").text("Functions");

  const groups = Array.from(new Set(ACTIVATION_FUNCTIONS.map(a => a.group)));
  for (const grp of groups) {
    const grpDiv = parent.append("div").attr("class", "fn-group");
    grpDiv.append("div").attr("class", "fn-group-label").text(grp);
    const fns = ACTIVATION_FUNCTIONS.filter(a => a.group === grp);
    for (const fn of fns) {
      const row = grpDiv.append("div").attr("class", "fn-checkbox-row");
      const cb = row.append("input")
        .attr("type", "checkbox")
        .attr("id", "cb-" + fn.name.replace(/[\s/]/g, "-"))
        .attr("class", "fn-cb");
      (cb.node() as HTMLInputElement).checked = state.selected.has(fn.name);

      // Color swatch
      row.append("span").attr("class", "color-swatch")
        .style("background", fn.color);

      row.append("label")
        .attr("for", "cb-" + fn.name.replace(/[\s/]/g, "-"))
        .text(fn.name);

      cb.on("change", function() {
        if ((this as HTMLInputElement).checked) {
          state.selected.add(fn.name);
        } else {
          state.selected.delete(fn.name);
        }
        render();
        updateTable();
      });
    }
  }
}

// ── Chart ─────────────────────────────────────────────────────────────────────

let svgSel: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
let plotG: d3.Selection<SVGGElement, unknown, HTMLElement, any>;
let xScale: d3.ScaleLinear<number, number>;
let yScale: d3.ScaleLinear<number, number>;
let xAxisG: d3.Selection<SVGGElement, unknown, HTMLElement, any>;
let yAxisG: d3.Selection<SVGGElement, unknown, HTMLElement, any>;
let hoverLine: d3.Selection<SVGLineElement, unknown, HTMLElement, any>;
let hoverDots: d3.Selection<SVGGElement, unknown, HTMLElement, any>;

function buildChart(parent: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>): void {
  svgSel = parent.append("svg")
    .attr("width", W).attr("height", H)
    .attr("class", "chart-svg");

  plotG = svgSel.append("g")
    .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

  // Grid
  plotG.append("g").attr("class", "grid-lines");

  // Axes
  xAxisG = plotG.append("g").attr("class", "axis x-axis")
    .attr("transform", `translate(0,${PLOT_H})`);
  yAxisG = plotG.append("g").attr("class", "axis y-axis");

  // Zero lines
  plotG.append("line").attr("class", "zero-line h-zero")
    .attr("x1", 0).attr("x2", PLOT_W);
  plotG.append("line").attr("class", "zero-line v-zero")
    .attr("y1", 0).attr("y2", PLOT_H);

  // Paths group
  plotG.append("g").attr("class", "paths-group");

  // Hover overlay
  hoverLine = plotG.append("line")
    .attr("class", "hover-line")
    .attr("y1", 0).attr("y2", PLOT_H)
    .style("display", "none");

  hoverDots = plotG.append("g").attr("class", "hover-dots");

  // Invisible rect for mouse events
  plotG.append("rect")
    .attr("width", PLOT_W).attr("height", PLOT_H)
    .attr("fill", "transparent")
    .on("mousemove", onMouseMove)
    .on("mouseleave", onMouseLeave);

  // Axis labels
  svgSel.append("text").attr("class", "axis-label")
    .attr("x", MARGIN.left + PLOT_W / 2).attr("y", H - 4)
    .attr("text-anchor", "middle").text("x");

  svgSel.append("text").attr("class", "axis-label")
    .attr("transform", `rotate(-90)`)
    .attr("x", -(MARGIN.top + PLOT_H / 2)).attr("y", 14)
    .attr("text-anchor", "middle").text("f(x)");

  xScale = d3.scaleLinear().range([0, PLOT_W]);
  yScale = d3.scaleLinear().range([PLOT_H, 0]);

  render();
}

function sampleFn(fn: ActFn): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i <= N_POINTS; i++) {
    const x = state.xMin + (i / N_POINTS) * (state.xMax - state.xMin);
    const y = fn.fn(x, state.params);
    pts.push([x, isFinite(y) ? y : NaN]);
  }
  return pts;
}

function sampleDfn(fn: ActFn): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i <= N_POINTS; i++) {
    const x = state.xMin + (i / N_POINTS) * (state.xMax - state.xMin);
    const y = fn.dfn(x, state.params);
    pts.push([x, isFinite(y) ? y : NaN]);
  }
  return pts;
}

function render(): void {
  const selectedFns = ACTIVATION_FUNCTIONS.filter(f => state.selected.has(f.name));

  // Compute y domain from all selected functions
  let yMin = -2, yMax = 2;
  for (const fn of selectedFns) {
    const pts = sampleFn(fn);
    for (const [, y] of pts) {
      if (isFinite(y)) { yMin = Math.min(yMin, y); yMax = Math.max(yMax, y); }
    }
    if (state.showDerivatives) {
      const dpts = sampleDfn(fn);
      for (const [, y] of dpts) {
        if (isFinite(y)) { yMin = Math.min(yMin, y); yMax = Math.max(yMax, y); }
      }
    }
  }
  // Add padding
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad; yMax += yPad;
  // Clamp extreme ranges
  yMin = Math.max(yMin, -12); yMax = Math.min(yMax, 12);

  xScale.domain([state.xMin, state.xMax]);
  yScale.domain([yMin, yMax]);

  // Update axes
  xAxisG.call(d3.axisBottom(xScale).ticks(10) as any);
  yAxisG.call(d3.axisLeft(yScale).ticks(8) as any);

  // Zero lines
  plotG.select(".h-zero")
    .attr("y1", yScale(0)).attr("y2", yScale(0));
  plotG.select(".v-zero")
    .attr("x1", xScale(0)).attr("x2", xScale(0));

  // Grid
  const gridG = plotG.select(".grid-lines");
  gridG.selectAll("*").remove();
  const xTicks = xScale.ticks(10);
  const yTicks = yScale.ticks(8);
  xTicks.forEach(t => {
    gridG.append("line").attr("class", "grid-line")
      .attr("x1", xScale(t)).attr("x2", xScale(t))
      .attr("y1", 0).attr("y2", PLOT_H);
  });
  yTicks.forEach(t => {
    gridG.append("line").attr("class", "grid-line")
      .attr("x1", 0).attr("x2", PLOT_W)
      .attr("y1", yScale(t)).attr("y2", yScale(t));
  });

  // Build line generator
  const line = d3.line<[number, number]>()
    .defined(d => isFinite(d[1]))
    .x(d => xScale(d[0]))
    .y(d => yScale(d[1]));

  const pathsG = plotG.select(".paths-group");
  pathsG.selectAll("*").remove();

  for (const fn of selectedFns) {
    const pts = sampleFn(fn);
    pathsG.append("path")
      .datum(pts)
      .attr("class", "fn-path")
      .attr("d", line as any)
      .attr("stroke", fn.color)
      .attr("fill", "none")
      .attr("stroke-width", 2.2)
      .attr("stroke-linejoin", "round");

    if (state.showDerivatives) {
      const dpts = sampleDfn(fn);
      pathsG.append("path")
        .datum(dpts)
        .attr("class", "fn-path deriv-path")
        .attr("d", line as any)
        .attr("stroke", fn.color)
        .attr("fill", "none")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "5,4")
        .attr("stroke-linejoin", "round")
        .attr("opacity", 0.7);
    }
  }

  // Render gradient chart
  renderGradChart();
  // Update hover if active
  if (state.hoverX !== null) renderHover(state.hoverX);
}

function onMouseMove(event: MouseEvent): void {
  const [mx] = d3.pointer(event);
  const x = xScale.invert(mx);
  state.hoverX = x;
  renderHover(x);
}

function renderHover(x: number): void {
  const px = xScale(x);
  hoverLine
    .attr("x1", px).attr("x2", px)
    .style("display", null);

  hoverDots.selectAll("*").remove();

  const selectedFns = ACTIVATION_FUNCTIONS.filter(f => state.selected.has(f.name));
  const infoLines: string[] = [`x = ${x.toFixed(3)}`];

  for (const fn of selectedFns) {
    const y = fn.fn(x, state.params);
    const dy = fn.dfn(x, state.params);
    if (isFinite(y) && yScale(y) >= 0 && yScale(y) <= PLOT_H) {
      hoverDots.append("circle")
        .attr("cx", px).attr("cy", yScale(y))
        .attr("r", 4).attr("fill", fn.color)
        .attr("stroke", "#1a1a2e").attr("stroke-width", 1.5);
    }
    infoLines.push(`${fn.name}: f=${y.toFixed(4)}, f′=${dy.toFixed(4)}`);
  }

  d3.select("#hover-info").html(infoLines.join("&ensp;|&ensp;"));
}

function onMouseLeave(): void {
  state.hoverX = null;
  hoverLine.style("display", "none");
  hoverDots.selectAll("*").remove();
  d3.select("#hover-info").text("Hover over the chart to read values");
}

// ── Gradient magnitude chart ──────────────────────────────────────────────────

const GRAD_H = 120;
const GRAD_W = PLOT_W;
let gradSvg: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
let gradG: d3.Selection<SVGGElement, unknown, HTMLElement, any>;
let gxScale: d3.ScaleLinear<number, number>;
let gyScale: d3.ScaleLinear<number, number>;
let gxAxisG: d3.Selection<SVGGElement, unknown, HTMLElement, any>;

function buildGradChart(parent: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>): void {
  const GM = { top: 10, right: 30, bottom: 36, left: 60 };
  gradSvg = parent.append("svg")
    .attr("width", GRAD_W + GM.left + GM.right)
    .attr("height", GRAD_H + GM.top + GM.bottom)
    .attr("class", "chart-svg");

  gradG = gradSvg.append("g").attr("transform", `translate(${GM.left},${GM.top})`);
  gradG.append("g").attr("class", "grad-paths");
  gxAxisG = gradG.append("g").attr("class", "axis x-axis")
    .attr("transform", `translate(0,${GRAD_H})`);
  gradG.append("g").attr("class", "axis y-axis grad-y-axis");

  gxScale = d3.scaleLinear().range([0, GRAD_W]);
  gyScale = d3.scaleLinear().domain([0, 2]).range([GRAD_H, 0]);

  gradSvg.append("text").attr("class", "axis-label")
    .attr("x", GM.left + GRAD_W / 2).attr("y", GRAD_H + GM.top + GM.bottom - 2)
    .attr("text-anchor", "middle").text("x");
  gradSvg.append("text").attr("class", "axis-label")
    .attr("transform", "rotate(-90)")
    .attr("x", -(GM.top + GRAD_H / 2)).attr("y", 14)
    .attr("text-anchor", "middle").text("|f′(x)|");

  // vanish band
  gradG.append("rect").attr("class", "vanish-band")
    .attr("x", 0).attr("width", GRAD_W)
    .attr("y", gyScale(0.1)).attr("height", gyScale(0) - gyScale(0.1));
}

function renderGradChart(): void {
  gxScale.domain([state.xMin, state.xMax]);
  gxAxisG.call(d3.axisBottom(gxScale).ticks(10) as any);
  gradG.select(".grad-y-axis").call(d3.axisLeft(gyScale).ticks(4) as any);

  const line = d3.line<[number, number]>()
    .defined(d => isFinite(d[1]))
    .x(d => gxScale(d[0]))
    .y(d => gyScale(Math.min(d[1], gyScale.domain()[1])));

  const pathsG = gradG.select(".grad-paths");
  pathsG.selectAll("*").remove();

  const selectedFns = ACTIVATION_FUNCTIONS.filter(f => state.selected.has(f.name));

  // auto-scale y
  let maxG = 0.1;
  for (const fn of selectedFns) {
    for (let i = 0; i <= N_POINTS; i++) {
      const x = state.xMin + (i / N_POINTS) * (state.xMax - state.xMin);
      const dy = Math.abs(fn.dfn(x, state.params));
      if (isFinite(dy)) maxG = Math.max(maxG, dy);
    }
  }
  gyScale.domain([0, Math.min(maxG * 1.1, 5)]);
  gradG.select(".grad-y-axis").call(d3.axisLeft(gyScale).ticks(4) as any);

  // Update vanish band
  gradG.select(".vanish-band")
    .attr("y", gyScale(0.1))
    .attr("height", Math.max(0, gyScale(0) - gyScale(0.1)));

  for (const fn of selectedFns) {
    const pts: [number, number][] = [];
    for (let i = 0; i <= N_POINTS; i++) {
      const x = state.xMin + (i / N_POINTS) * (state.xMax - state.xMin);
      const dy = Math.abs(fn.dfn(x, state.params));
      pts.push([x, isFinite(dy) ? dy : NaN]);
    }
    pathsG.append("path")
      .datum(pts)
      .attr("d", line as any)
      .attr("stroke", fn.color)
      .attr("fill", "none")
      .attr("stroke-width", 2);
  }

  // Legend for vanish band
  const existing = gradG.select(".vanish-label");
  if (existing.empty()) {
    gradG.append("text").attr("class", "vanish-label")
      .attr("x", 4).attr("fill", "#f87171").attr("font-size", 10);
  }
  gradG.select(".vanish-label")
    .attr("y", gyScale(0.05) - 2)
    .text("vanishing zone (|f′|<0.1)");
}

// ── Properties table ──────────────────────────────────────────────────────────

function buildTable(parent: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>): void {
  const wrap = parent.append("div").attr("class", "table-wrap");
  const tbl = wrap.append("table").attr("id", "props-table").attr("class", "props-table");
  const thead = tbl.append("thead");
  const hr = thead.append("tr");
  ["Function", "Range", "Monotonic", "Saturating", "Zero-Centered", "Smooth", "Dead-ReLU Risk"]
    .forEach(h => hr.append("th").text(h));
  tbl.append("tbody").attr("id", "props-tbody");
  updateTable();
}

function yesNo(v: boolean, danger?: boolean): string {
  if (danger) return v ? `<span class="badge badge-danger">Yes</span>` : `<span class="badge badge-ok">No</span>`;
  return v ? `<span class="badge badge-yes">Yes</span>` : `<span class="badge badge-no">No</span>`;
}

function updateTable(): void {
  const tbody = d3.select("#props-tbody");
  tbody.selectAll("*").remove();

  const selectedFns = ACTIVATION_FUNCTIONS.filter(f => state.selected.has(f.name));

  if (selectedFns.length === 0) {
    tbody.append("tr").append("td").attr("colspan", 7)
      .attr("class", "empty-row").text("No functions selected");
    return;
  }

  for (const fn of selectedFns) {
    const p = PROPERTIES[fn.name];
    if (!p) continue;
    const tr = tbody.append("tr");
    // Name cell with color indicator
    const nameTd = tr.append("td");
    nameTd.append("span").attr("class", "color-swatch").style("background", fn.color);
    nameTd.append("span").text(fn.name);
    tr.append("td").text(p.range);
    tr.append("td").html(yesNo(p.monotonic));
    tr.append("td").html(yesNo(p.saturating));
    tr.append("td").html(yesNo(p.zeroCentered));
    tr.append("td").html(yesNo(p.smooth));
    tr.append("td").html(yesNo(p.deadReLU, true));
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", () => {
  buildUI();
});
