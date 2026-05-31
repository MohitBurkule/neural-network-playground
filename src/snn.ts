import * as d3 from 'd3';

export {};

// ─── Types ────────────────────────────────────────────────────────────────────

interface LIFNeuron {
  v: number;          // membrane potential
  refrac: number;     // remaining refractory ticks
  spikeTimes: number[];
  vTrace: number[];   // ring buffer for voltage trace
}

interface Synapse {
  pre: number;        // index into all neurons (0..nInput-1 = input, rest = LIF)
  post: number;       // index into LIF neurons
  weight: number;
}

// ─── Simulation state ─────────────────────────────────────────────────────────

const TRACE_LEN = 300;   // samples kept in voltage trace display
const RASTER_WIN = 600;  // ms of raster shown

let params = {
  inputRate: 40,       // Hz
  threshold: 1.0,
  tauMs: 20,           // membrane time constant ms
  refracMs: 5,         // refractory period ms
  weightScale: 0.8,
  nInput: 8,
  nLIF: 6,
  stdpOn: false,
  speedMult: 1,
  dtMs: 1.0,
};

let simTime = 0;          // ms
let running = false;
let rafId = 0;
let lastRealMs = 0;

let inputNeurons: { spikeTimes: number[] }[] = [];
let lifNeurons: LIFNeuron[] = [];
let synapses: Synapse[] = [];
let firingRates: number[] = [];   // spikes / last 200 ms

// STDP params
const A_PLUS = 0.01;
const A_MINUS = 0.012;
const TAU_STDP = 20; // ms

// ─── Init / Reset ─────────────────────────────────────────────────────────────

function initNetwork() {
  simTime = 0;

  inputNeurons = Array.from({ length: params.nInput }, () => ({ spikeTimes: [] }));

  lifNeurons = Array.from({ length: params.nLIF }, () => ({
    v: 0,
    refrac: 0,
    spikeTimes: [],
    vTrace: Array.from({ length: TRACE_LEN }, () => 0),
  }));

  // Fully-connected input → LIF, random weights
  synapses = [];
  for (let i = 0; i < params.nInput; i++) {
    for (let j = 0; j < params.nLIF; j++) {
      synapses.push({
        pre: i,
        post: j,
        weight: (0.5 + Math.random() * 0.5) * params.weightScale,
      });
    }
  }

  firingRates = Array.from({ length: params.nLIF }, () => 0);
}

// ─── Single simulation step (dtMs) ───────────────────────────────────────────

function stepSim() {
  const dt = params.dtMs;
  const t = simTime;

  // --- Generate Poisson spikes for input neurons ---
  const pSpike = params.inputRate * dt / 1000;
  Array.from({ length: params.nInput }).forEach((_, i) => {
    if (Math.random() < pSpike) {
      inputNeurons[i].spikeTimes.push(t);
    }
  });

  // --- Integrate LIF neurons ---
  const tau = params.tauMs;
  const thresh = params.threshold;
  const refracTicks = params.refracMs / dt;

  lifNeurons.forEach((n, j) => {
    // Collect synaptic input (spikes that arrived this tick: spike in [t-dt, t))
    let I = 0;
    synapses.forEach(s => {
      if (s.post !== j) return;
      const src = inputNeurons[s.pre];
      const recent = src.spikeTimes.filter(ts => ts >= t - dt && ts < t + dt * 0.5);
      I += recent.length * s.weight;
    });

    if (n.refrac > 0) {
      n.refrac -= 1;
      n.v = 0;
    } else {
      // dV/dt = (-V + I) / tau  → Euler
      n.v += dt * ((-n.v + I * 10) / tau);

      if (n.v >= thresh) {
        n.v = 0;
        n.refrac = refracTicks;
        n.spikeTimes.push(t);

        // STDP: for each synapse onto this post neuron
        if (params.stdpOn) {
          synapses.forEach(s => {
            if (s.post !== j) return;
            const preSrc = inputNeurons[s.pre];
            // LTP: pre fired before post
            preSrc.spikeTimes.forEach(tpre => {
              const dt2 = t - tpre;
              if (dt2 >= 0 && dt2 < 5 * TAU_STDP) {
                s.weight += A_PLUS * Math.exp(-dt2 / TAU_STDP);
              }
            });
            // LTD: post fired before pre (use last post spike — current)
            n.spikeTimes.slice(-5).forEach(tpost => {
              const dt2 = tpost - t;
              if (dt2 > 0 && dt2 < 5 * TAU_STDP) {
                s.weight -= A_MINUS * Math.exp(-dt2 / TAU_STDP);
              }
            });
            s.weight = Math.max(0, Math.min(3, s.weight));
          });
        }
      }
    }

    // Update vTrace ring buffer (shift left, push new value)
    n.vTrace.shift();
    n.vTrace.push(n.v);
  });

  // --- Prune old spike times to keep memory bounded ---
  const cutoff = t - RASTER_WIN - 50;
  inputNeurons.forEach(n => {
    if (n.spikeTimes.length > 500) {
      n.spikeTimes = n.spikeTimes.filter(ts => ts >= cutoff);
    }
  });
  lifNeurons.forEach(n => {
    if (n.spikeTimes.length > 500) {
      n.spikeTimes = n.spikeTimes.filter(ts => ts >= cutoff);
    }
  });

  // --- Update firing rates (spikes in last 200 ms) ---
  const win = 200;
  lifNeurons.forEach((n, j) => {
    firingRates[j] = n.spikeTimes.filter(ts => ts >= t - win).length / (win / 1000);
  });

  simTime += dt;
}

// ─── D3 Visualisation ─────────────────────────────────────────────────────────

const PANEL_BG = '#0d0d1f';
const GRID_COLOR = '#1e1e3a';
const SPIKE_COLOR_INPUT = '#38bdf8';
const SPIKE_COLOR_LIF = '#f472b6';
const VOLT_COLOR = '#a78bfa';
const FIRE_COLOR = '#f97316';

// SVG containers
let svgRaster: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
let svgVolt: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
let svgRate: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;

const RASTER_W = 560, RASTER_H = 280;
const VOLT_W = 560, VOLT_H = 200;
const RATE_W = 200, RATE_H = 280;

const rasterMargin = { top: 20, right: 10, bottom: 30, left: 50 };
const voltMargin   = { top: 20, right: 10, bottom: 30, left: 50 };
const rateMargin   = { top: 20, right: 10, bottom: 30, left: 40 };

function initViz() {
  // ── Raster ──
  svgRaster = d3.select<SVGSVGElement, unknown>('#rasterSvg')
    .attr('width', RASTER_W).attr('height', RASTER_H);
  svgRaster.append('rect').attr('width', RASTER_W).attr('height', RASTER_H).attr('fill', PANEL_BG);
  svgRaster.append('g').attr('class', 'raster-content')
    .attr('transform', `translate(${rasterMargin.left},${rasterMargin.top})`);
  svgRaster.append('g').attr('class', 'x-axis')
    .attr('transform', `translate(${rasterMargin.left},${RASTER_H - rasterMargin.bottom})`);
  svgRaster.append('g').attr('class', 'y-axis')
    .attr('transform', `translate(${rasterMargin.left},${rasterMargin.top})`);

  // ── Voltage ──
  svgVolt = d3.select<SVGSVGElement, unknown>('#voltSvg')
    .attr('width', VOLT_W).attr('height', VOLT_H);
  svgVolt.append('rect').attr('width', VOLT_W).attr('height', VOLT_H).attr('fill', PANEL_BG);
  svgVolt.append('g').attr('class', 'volt-content')
    .attr('transform', `translate(${voltMargin.left},${voltMargin.top})`);
  svgVolt.append('g').attr('class', 'x-axis')
    .attr('transform', `translate(${voltMargin.left},${VOLT_H - voltMargin.bottom})`);
  svgVolt.append('g').attr('class', 'y-axis')
    .attr('transform', `translate(${voltMargin.left},${voltMargin.top})`);

  // ── Firing rate ──
  svgRate = d3.select<SVGSVGElement, unknown>('#rateSvg')
    .attr('width', RATE_W).attr('height', RATE_H);
  svgRate.append('rect').attr('width', RATE_W).attr('height', RATE_H).attr('fill', PANEL_BG);
  svgRate.append('g').attr('class', 'rate-content')
    .attr('transform', `translate(${rateMargin.left},${rateMargin.top})`);
  svgRate.append('g').attr('class', 'x-axis')
    .attr('transform', `translate(${rateMargin.left},${RATE_H - rateMargin.bottom})`);
  svgRate.append('g').attr('class', 'y-axis')
    .attr('transform', `translate(${rateMargin.left},${rateMargin.top})`);
}

// Colors for voltage traces
const voltColors = ['#a78bfa', '#34d399', '#f472b6', '#60a5fa', '#fbbf24', '#f87171'];

function drawRaster() {
  const tEnd = simTime;
  const tStart = tEnd - RASTER_WIN;
  const innerW = RASTER_W - rasterMargin.left - rasterMargin.right;
  const innerH = RASTER_H - rasterMargin.top - rasterMargin.bottom;

  const totalRows = params.nInput + params.nLIF;

  const xScale = d3.scaleLinear().domain([tStart, tEnd]).range([0, innerW]);
  const yScale = d3.scaleBand()
    .domain(Array.from({ length: totalRows }, (_, i) => String(i)))
    .range([0, innerH]).padding(0.1);

  // X axis
  const xAxis = d3.axisBottom(xScale).ticks(6).tickFormat(d => `${d}ms`);
  svgRaster.select<SVGGElement>('.x-axis')
    .call(xAxis)
    .call(g => g.selectAll('text').attr('fill', '#5a5a8a').attr('font-size', '10px'))
    .call(g => g.selectAll('line,path').attr('stroke', '#3a3a5a'));

  // Y axis
  const yAxis = d3.axisLeft(yScale).tickFormat((d, i) => {
    const idx = Number(d);
    return idx < params.nInput ? `I${idx}` : `N${idx - params.nInput}`;
  });
  svgRaster.select<SVGGElement>('.y-axis')
    .call(yAxis)
    .call(g => g.selectAll('text').attr('fill', '#5a5a8a').attr('font-size', '9px'))
    .call(g => g.selectAll('line,path').attr('stroke', '#3a3a5a'));

  const content = svgRaster.select('.raster-content');

  // Combine all spikes
  type SpikeDatum = { row: number; t: number; isInput: boolean };
  const spikes: SpikeDatum[] = [];

  inputNeurons.forEach((n, i) => {
    n.spikeTimes.forEach(ts => {
      if (ts >= tStart && ts <= tEnd) spikes.push({ row: i, t: ts, isInput: true });
    });
  });

  lifNeurons.forEach((n, j) => {
    n.spikeTimes.forEach(ts => {
      if (ts >= tStart && ts <= tEnd) spikes.push({ row: params.nInput + j, t: ts, isInput: false });
    });
  });

  const dots = content.selectAll<SVGCircleElement, SpikeDatum>('circle.spike')
    .data(spikes, (d: SpikeDatum) => `${d.row}-${d.t}`);

  dots.enter().append('circle')
    .attr('class', 'spike')
    .attr('r', 2)
    .attr('cx', (d: SpikeDatum) => xScale(d.t))
    .attr('cy', (d: SpikeDatum) => (yScale(String(d.row)) ?? 0) + yScale.bandwidth() / 2)
    .attr('fill', (d: SpikeDatum) => d.isInput ? SPIKE_COLOR_INPUT : SPIKE_COLOR_LIF)
    .attr('opacity', 0.85);

  // Update positions of existing dots (time window shifts)
  dots.attr('cx', (d: SpikeDatum) => xScale(d.t));

  dots.exit().remove();

  // Divider between input and LIF layers
  const divY = (yScale(String(params.nInput)) ?? 0);
  content.selectAll('line.divider').remove();
  content.append('line').attr('class', 'divider')
    .attr('x1', 0).attr('x2', innerW)
    .attr('y1', divY).attr('y2', divY)
    .attr('stroke', '#3a3a5a').attr('stroke-dasharray', '4,3');
}

function drawVoltage() {
  const innerW = VOLT_W - voltMargin.left - voltMargin.right;
  const innerH = VOLT_H - voltMargin.top - voltMargin.bottom;

  const xScale = d3.scaleLinear().domain([0, TRACE_LEN - 1]).range([0, innerW]);
  const yScale = d3.scaleLinear().domain([-0.2, params.threshold * 1.2]).range([innerH, 0]);

  const xAxis = d3.axisBottom(xScale).ticks(5).tickFormat(d => `${Number(d) * params.dtMs}ms`);
  svgVolt.select<SVGGElement>('.x-axis')
    .call(xAxis)
    .call(g => g.selectAll('text').attr('fill', '#5a5a8a').attr('font-size', '10px'))
    .call(g => g.selectAll('line,path').attr('stroke', '#3a3a5a'));

  const yAxis = d3.axisLeft(yScale).ticks(4);
  svgVolt.select<SVGGElement>('.y-axis')
    .call(yAxis)
    .call(g => g.selectAll('text').attr('fill', '#5a5a8a').attr('font-size', '10px'))
    .call(g => g.selectAll('line,path').attr('stroke', '#3a3a5a'));

  const content = svgVolt.select('.volt-content');

  // Threshold line
  content.selectAll('line.thresh').remove();
  content.append('line').attr('class', 'thresh')
    .attr('x1', 0).attr('x2', innerW)
    .attr('y1', yScale(params.threshold)).attr('y2', yScale(params.threshold))
    .attr('stroke', '#fbbf24').attr('stroke-dasharray', '6,3').attr('opacity', 0.5);

  // Draw up to 4 LIF neuron traces
  const showCount = Math.min(4, params.nLIF);
  const lineGen = d3.line<number>()
    .x((_d, i) => xScale(i))
    .y(d => yScale(d));

  Array.from({ length: showCount }).forEach((_, j) => {
    const color = voltColors[j % voltColors.length];
    const pathId = `volt-path-${j}`;

    if (content.selectAll(`path#${pathId}`).empty()) {
      content.append('path').attr('id', pathId)
        .attr('fill', 'none')
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.9);
    }
    (content.selectAll<SVGPathElement, number[]>(`path#${pathId}`)
      .datum(lifNeurons[j].vTrace) as d3.Selection<SVGPathElement, number[], d3.BaseType, unknown>)
      .attr('stroke', color)
      .attr('d', lineGen);
  });

  // Remove extra paths if nLIF decreased
  for (let j = showCount; j < 6; j++) {
    content.selectAll(`path#volt-path-${j}`).remove();
  }

  // Legend
  content.selectAll('g.volt-legend').remove();
  const leg = content.append('g').attr('class', 'volt-legend')
    .attr('transform', `translate(${innerW - 80}, 2)`);
  Array.from({ length: showCount }).forEach((_, j) => {
    leg.append('rect').attr('x', 0).attr('y', j * 14).attr('width', 10).attr('height', 3)
      .attr('fill', voltColors[j % voltColors.length]);
    leg.append('text').attr('x', 13).attr('y', j * 14 + 6)
      .attr('fill', '#8888aa').attr('font-size', '9px').text(`N${j}`);
  });
}

function drawRates() {
  const innerW = RATE_W - rateMargin.left - rateMargin.right;
  const innerH = RATE_H - rateMargin.top - rateMargin.bottom;

  const maxRate = Math.max(1, d3.max(firingRates) ?? 1);
  const xScale = d3.scaleLinear().domain([0, maxRate * 1.1]).range([0, innerW]);
  const yScale = d3.scaleBand()
    .domain(Array.from({ length: params.nLIF }, (_, i) => String(i)))
    .range([0, innerH]).padding(0.15);

  const xAxis = d3.axisBottom(xScale).ticks(4).tickFormat(d => `${d}Hz`);
  svgRate.select<SVGGElement>('.x-axis')
    .call(xAxis)
    .call(g => g.selectAll('text').attr('fill', '#5a5a8a').attr('font-size', '10px'))
    .call(g => g.selectAll('line,path').attr('stroke', '#3a3a5a'));

  const yAxis = d3.axisLeft(yScale).tickFormat(d => `N${d}`);
  svgRate.select<SVGGElement>('.y-axis')
    .call(yAxis)
    .call(g => g.selectAll('text').attr('fill', '#5a5a8a').attr('font-size', '10px'))
    .call(g => g.selectAll('line,path').attr('stroke', '#3a3a5a'));

  const content = svgRate.select('.rate-content');

  const bars = content.selectAll<SVGRectElement, number>('rect.rate-bar')
    .data(firingRates);

  bars.enter().append('rect').attr('class', 'rate-bar')
    .attr('fill', FIRE_COLOR).attr('opacity', 0.8).attr('rx', 2)
    .merge(bars)
    .attr('x', 0)
    .attr('y', (_d, i) => yScale(String(i)) ?? 0)
    .attr('height', yScale.bandwidth())
    .attr('width', d => xScale(d));

  bars.exit().remove();

  // Rate labels
  const labels = content.selectAll<SVGTextElement, number>('text.rate-label')
    .data(firingRates);

  labels.enter().append('text').attr('class', 'rate-label')
    .attr('fill', '#d0d0e8').attr('font-size', '9px').attr('dominant-baseline', 'middle')
    .merge(labels)
    .attr('x', d => xScale(d) + 3)
    .attr('y', (_d, i) => (yScale(String(i)) ?? 0) + yScale.bandwidth() / 2)
    .text(d => `${d.toFixed(1)}`);

  labels.exit().remove();
}

function updateClock() {
  const el = document.getElementById('simClock');
  if (el) el.textContent = `t = ${simTime.toFixed(0)} ms`;
}

// ─── Animation loop ───────────────────────────────────────────────────────────

const STEPS_PER_FRAME = 10;

function animate(nowMs: number) {
  if (!running) return;
  const elapsed = nowMs - lastRealMs;
  lastRealMs = nowMs;

  // sim speed: speedMult steps per ms of real time
  const stepsTarget = Math.round(elapsed * params.speedMult / params.dtMs);
  const steps = Math.min(stepsTarget, 200); // cap to avoid freezing

  for (let i = 0; i < steps; i++) {
    stepSim();
  }

  drawRaster();
  drawVoltage();
  drawRates();
  updateClock();

  rafId = requestAnimationFrame(animate);
}

function play() {
  if (running) return;
  running = true;
  lastRealMs = performance.now();
  rafId = requestAnimationFrame(animate);
  (document.getElementById('playBtn') as HTMLButtonElement).textContent = '⏸ Pause';
}

function pause() {
  running = false;
  cancelAnimationFrame(rafId);
  (document.getElementById('playBtn') as HTMLButtonElement).textContent = '▶ Play';
}

function step() {
  pause();
  for (let i = 0; i < STEPS_PER_FRAME; i++) stepSim();
  drawRaster();
  drawVoltage();
  drawRates();
  updateClock();
}

function reset() {
  pause();
  initNetwork();
  drawRaster();
  drawVoltage();
  drawRates();
  updateClock();
}

// ─── Controls wiring ──────────────────────────────────────────────────────────

function setNumParam(param: keyof typeof params, val: number) {
  switch (param) {
    case 'inputRate':   params.inputRate = val; break;
    case 'threshold':   params.threshold = val; break;
    case 'tauMs':       params.tauMs = val; break;
    case 'refracMs':    params.refracMs = val; break;
    case 'weightScale': params.weightScale = val; break;
    case 'speedMult':   params.speedMult = val; break;
    case 'nInput':      params.nInput = val; break;
    case 'nLIF':        params.nLIF = val; break;
    default: break;
  }
}

function bindSlider(id: string, param: keyof typeof params, transform?: (v: number) => number) {
  const el = document.getElementById(id) as HTMLInputElement;
  const lbl = document.getElementById(id + 'Val');
  if (!el) return;
  el.addEventListener('input', () => {
    const raw = parseFloat(el.value);
    const val = transform ? transform(raw) : raw;
    const oldWeight = params.weightScale;
    setNumParam(param, val);
    if (lbl) lbl.textContent = String(raw);
    if (param === 'weightScale' && oldWeight > 0) {
      synapses.forEach(s => { s.weight = Math.min(3, s.weight * val / oldWeight); });
    }
  });
}

function bindSelect(id: string, param: keyof typeof params) {
  const el = document.getElementById(id) as HTMLSelectElement;
  if (!el) return;
  el.addEventListener('change', () => {
    (params as unknown as Record<string, number>)[param as string] = parseInt(el.value, 10);
    reset();
  });
}

function wireControls() {
  bindSlider('inputRateSlider', 'inputRate');
  bindSlider('threshSlider', 'threshold');
  bindSlider('tauSlider', 'tauMs');
  bindSlider('refracSlider', 'refracMs');
  bindSlider('weightSlider', 'weightScale');
  bindSlider('speedSlider', 'speedMult');

  bindSelect('nInputSelect', 'nInput');
  bindSelect('nLIFSelect', 'nLIF');

  const stdpCb = document.getElementById('stdpToggle') as HTMLInputElement;
  if (stdpCb) {
    stdpCb.addEventListener('change', () => { params.stdpOn = stdpCb.checked; });
  }

  const playBtn = document.getElementById('playBtn');
  if (playBtn) playBtn.addEventListener('click', () => running ? pause() : play());

  const stepBtn = document.getElementById('stepBtn');
  if (stepBtn) stepBtn.addEventListener('click', step);

  const resetBtn = document.getElementById('resetBtn');
  if (resetBtn) resetBtn.addEventListener('click', reset);
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initNetwork();
  initViz();
  wireControls();
  drawRaster();
  drawVoltage();
  drawRates();
  updateClock();
  play();
});
