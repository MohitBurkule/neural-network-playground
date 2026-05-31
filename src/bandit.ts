import * as d3 from 'd3';

// ---- Types ----
type RewardType = 'bernoulli' | 'gaussian';
type StrategyName = 'epsilon-greedy' | 'optimistic' | 'ucb1' | 'thompson';

interface Arm {
  trueMean: number;  // Bernoulli: p in [0,1]; Gaussian: mu
  trueStd: number;   // Gaussian only
}

interface StrategyState {
  name: StrategyName;
  label: string;
  color: string;
  counts: number[];
  values: number[];     // estimated Q
  alphas: number[];     // Thompson: beta alpha
  betas: number[];      // Thompson: beta beta
  cumReward: number;
  cumRegret: number[];  // per step
  optimalCount: number;
  totalPulls: number;
}

// ---- Config ----
let K = 5;
let rewardType: RewardType = 'bernoulli';
let epsilon = 0.1;
let ucbC = 2.0;
let optInitVal = 1.0;
let compareAll = true;
let activeStrategy: StrategyName = 'epsilon-greedy';
let speed = 50; // ms per frame
let running = false;

// ---- Bandit State ----
let arms: Arm[] = [];
let strategies: StrategyState[] = [];
let stepCount = 0;
let animHandle: number | null = null;

const STRATEGY_DEFS: { name: StrategyName; label: string; color: string }[] = [
  { name: 'epsilon-greedy', label: 'ε-Greedy',        color: '#a78bfa' },
  { name: 'optimistic',     label: 'Optimistic Init',  color: '#34d399' },
  { name: 'ucb1',           label: 'UCB1',             color: '#fbbf24' },
  { name: 'thompson',       label: 'Thompson Sampling', color: '#f87171' },
];

// ---- Math helpers ----
function randGaussian(): number {
  // Box-Muller
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function sampleReward(arm: Arm): number {
  if (rewardType === 'bernoulli') {
    return Math.random() < arm.trueMean ? 1 : 0;
  } else {
    return arm.trueMean + arm.trueStd * randGaussian();
  }
}

function sampleBeta(alpha: number, beta: number): number {
  // Johnk's method approximation via gamma samples
  const x = sampleGamma(alpha);
  const y = sampleGamma(beta);
  if (x + y === 0) return 0.5;
  return x / (x + y);
}

function sampleGamma(shape: number): number {
  // Marsaglia and Tsang's method
  if (shape < 1) {
    return sampleGamma(1 + shape) * Math.pow(Math.random(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x = randGaussian();
    let v = 1 + c * x;
    if (v <= 0) continue;
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function optimalArm(): number {
  let best = 0;
  for (let i = 1; i < K; i++) {
    if (arms[i].trueMean > arms[best].trueMean) best = i;
  }
  return best;
}

// ---- Init ----
function randomizeArms(): void {
  arms = [];
  for (let i = 0; i < K; i++) {
    if (rewardType === 'bernoulli') {
      arms.push({ trueMean: Math.random() * 0.7 + 0.1, trueStd: 0 });
    } else {
      arms.push({ trueMean: randGaussian() * 1.5, trueStd: 0.5 + Math.random() * 1.0 });
    }
  }
}

function makeStrategy(def: { name: StrategyName; label: string; color: string }): StrategyState {
  const initVal = def.name === 'optimistic' ? optInitVal : 0;
  const alphas = [];
  const betas_ = [];
  const values = [];
  for (let i = 0; i < K; i++) {
    values.push(initVal);
    alphas.push(1);
    betas_.push(1);
  }
  return {
    name: def.name,
    label: def.label,
    color: def.color,
    counts: new Array(K).fill(0),
    values,
    alphas,
    betas: betas_,
    cumReward: 0,
    cumRegret: [0],
    optimalCount: 0,
    totalPulls: 0,
  };
}

function resetAll(): void {
  stepCount = 0;
  randomizeArms();
  const defs = compareAll ? STRATEGY_DEFS : STRATEGY_DEFS.filter(d => d.name === activeStrategy);
  strategies = defs.map(makeStrategy);
  if (animHandle !== null) { cancelAnimationFrame(animHandle); animHandle = null; }
  running = false;
  updatePlayPauseBtn();
  render();
}

// ---- Strategy: pick arm ----
function pickArm(s: StrategyState): number {
  const t = s.totalPulls;
  if (s.name === 'epsilon-greedy') {
    if (Math.random() < epsilon) return Math.floor(Math.random() * K);
    let best = 0;
    for (let i = 1; i < K; i++) if (s.values[i] > s.values[best]) best = i;
    return best;
  } else if (s.name === 'optimistic') {
    // Greedy on optimistic init values
    let best = 0;
    for (let i = 1; i < K; i++) if (s.values[i] > s.values[best]) best = i;
    return best;
  } else if (s.name === 'ucb1') {
    if (t < K) return t % K; // pull each once first
    let best = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < K; i++) {
      const ucb = s.values[i] + ucbC * Math.sqrt(Math.log(t) / s.counts[i]);
      if (ucb > bestVal) { bestVal = ucb; best = i; }
    }
    return best;
  } else {
    // Thompson
    let best = 0;
    let bestSample = -Infinity;
    for (let i = 0; i < K; i++) {
      let sample: number;
      if (rewardType === 'bernoulli') {
        sample = sampleBeta(s.alphas[i], s.betas[i]);
      } else {
        // Gaussian Thompson: normal approximation using variance from counts
        const mu = s.values[i];
        const sig = s.counts[i] > 0 ? 1.0 / Math.sqrt(s.counts[i]) : 1.0;
        sample = mu + sig * randGaussian();
      }
      if (sample > bestSample) { bestSample = sample; best = i; }
    }
    return best;
  }
}

function updateStrategy(s: StrategyState, arm: number, reward: number): void {
  s.counts[arm]++;
  s.totalPulls++;
  // Incremental mean update
  s.values[arm] += (reward - s.values[arm]) / s.counts[arm];
  // Thompson Beta update (Bernoulli)
  if (s.name === 'thompson' && rewardType === 'bernoulli') {
    if (reward >= 0.5) s.alphas[arm]++;
    else s.betas[arm]++;
  }
  s.cumReward += reward;
  const opt = optimalArm();
  const maxMean = arms[opt].trueMean;
  const regretStep = maxMean - arms[arm].trueMean;
  const prevRegret = s.cumRegret.length > 0 ? s.cumRegret[s.cumRegret.length - 1] : 0;
  s.cumRegret.push(prevRegret + regretStep);
  if (arm === opt) s.optimalCount++;
}

function stepOnce(): void {
  const opt = optimalArm();
  strategies.forEach(s => {
    const arm = pickArm(s);
    const reward = sampleReward(arms[arm]);
    updateStrategy(s, arm, reward);
  });
  stepCount++;
}

// ---- Animation ----
let lastFrameTime = 0;
let stepsPerFrame = 1;

function animate(ts: number): void {
  if (!running) return;
  const elapsed = ts - lastFrameTime;
  const interval = Math.max(16, 1000 - speed * 9.9);
  if (elapsed >= interval) {
    lastFrameTime = ts;
    const burst = stepsPerFrame;
    for (let i = 0; i < burst; i++) stepOnce();
    render();
  }
  animHandle = requestAnimationFrame(animate);
}

function startAnimation(): void {
  running = true;
  updatePlayPauseBtn();
  lastFrameTime = 0;
  animHandle = requestAnimationFrame(animate);
}

function pauseAnimation(): void {
  running = false;
  if (animHandle !== null) { cancelAnimationFrame(animHandle); animHandle = null; }
  updatePlayPauseBtn();
}

// ---- DOM refs ----
let svgArms: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
let svgCounts: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
let svgRegret: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
let svgOptimal: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
let playPauseBtn: HTMLButtonElement;
let stepCountEl: HTMLElement;

function updatePlayPauseBtn(): void {
  if (playPauseBtn) playPauseBtn.textContent = running ? '⏸ Pause' : '▶ Play';
}

// ---- Render ----
function render(): void {
  if (stepCountEl) stepCountEl.textContent = String(stepCount);
  renderArmBars();
  renderCounts();
  renderRegret();
  renderOptimal();
}

function renderArmBars(): void {
  const svg = svgArms;
  if (!svg) return;
  svg.selectAll('*').remove();

  const node = (svg.node() as SVGSVGElement);
  const W = node.clientWidth || 500;
  const H = node.clientHeight || 240;
  const margin = { top: 28, right: 16, bottom: 36, left: 44 };
  const w = W - margin.left - margin.right;
  const h = H - margin.top - margin.bottom;
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  // title
  svg.append('text').attr('x', W / 2).attr('y', 14)
    .attr('text-anchor', 'middle').attr('fill', '#aaa').attr('font-size', 12)
    .text('Arm Values: Estimated vs True');

  const allMeans = arms.map(a => a.trueMean);
  const allEst: number[] = [];
  strategies.forEach(s => s.values.forEach(v => allEst.push(v)));
  const yMin = Math.min(0, ...allMeans, ...allEst);
  const yMax = Math.max(0.01, ...allMeans, ...allEst) * 1.1;

  const armIndices = d3.range(K);
  const nStrats = strategies.length;
  // group by arm, sub-group by strategy + true
  const groupW = w / K;
  const barW = Math.max(4, (groupW - 4) / (nStrats + 1) - 2);

  const yScale = d3.scaleLinear().domain([yMin, yMax]).range([h, 0]);

  // axes
  g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(d3.scaleBand().domain(armIndices.map(String)).range([0, w])).tickSize(0))
    .call(ax => ax.select('.domain').remove())
    .selectAll('text').attr('fill', '#999').attr('font-size', 11);
  g.append('g').call(d3.axisLeft(yScale).ticks(4).tickFormat(d => String(d)))
    .call(ax => { ax.select('.domain').remove(); ax.selectAll('.tick line').attr('stroke', '#333'); ax.selectAll('text').attr('fill', '#999').attr('font-size', 11); });
  // gridlines
  g.append('g').selectAll('line').data(yScale.ticks(4)).enter().append('line')
    .attr('x1', 0).attr('x2', w).attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
    .attr('stroke', '#2a2a3a').attr('stroke-dasharray', '3,3');

  armIndices.forEach(i => {
    const gx = (i / K) * w;
    // true mean bar
    const trueY = yScale(arms[i].trueMean);
    const trueH = Math.abs(yScale(0) - trueY);
    g.append('rect')
      .attr('x', gx + 2)
      .attr('y', arms[i].trueMean >= 0 ? trueY : yScale(0))
      .attr('width', barW)
      .attr('height', Math.max(1, trueH))
      .attr('fill', '#555')
      .attr('opacity', 0.8);
    g.append('text').attr('x', gx + 2 + barW / 2).attr('y', (arms[i].trueMean >= 0 ? trueY : yScale(0)) - 3)
      .attr('text-anchor', 'middle').attr('fill', '#888').attr('font-size', 9).text('true');

    strategies.forEach((s, si) => {
      const est = s.values[i];
      const estY = yScale(est);
      const estH = Math.abs(yScale(0) - estY);
      const bx = gx + 2 + (si + 1) * (barW + 2);
      g.append('rect')
        .attr('x', bx)
        .attr('y', est >= 0 ? estY : yScale(0))
        .attr('width', barW)
        .attr('height', Math.max(1, estH))
        .attr('fill', s.color)
        .attr('opacity', 0.75);
      // count label
      if (barW > 10) {
        g.append('text').attr('x', bx + barW / 2).attr('y', h + 14)
          .attr('text-anchor', 'middle').attr('fill', s.color).attr('font-size', 8)
          .text(String(s.counts[i]));
      }
    });
    // arm label
    g.append('text').attr('x', gx + groupW / 2).attr('y', h + 26)
      .attr('text-anchor', 'middle').attr('fill', '#ccc').attr('font-size', 11)
      .text(`A${i + 1}`);
  });

  // legend
  const legendG = svg.append('g').attr('transform', `translate(${margin.left},${H - 6})`);
  legendG.append('rect').attr('width', 10).attr('height', 10).attr('y', -10).attr('fill', '#555');
  legendG.append('text').attr('x', 13).attr('y', 0).attr('fill', '#888').attr('font-size', 9).text('true');
  strategies.forEach((s, si) => {
    const lx = 50 + si * 110;
    legendG.append('rect').attr('x', lx).attr('width', 10).attr('height', 10).attr('y', -10).attr('fill', s.color);
    legendG.append('text').attr('x', lx + 13).attr('y', 0).attr('fill', s.color).attr('font-size', 9).text(s.label);
  });
}

function renderCounts(): void {
  const svg = svgCounts;
  if (!svg) return;
  svg.selectAll('*').remove();

  const node = (svg.node() as SVGSVGElement);
  const W = node.clientWidth || 500;
  const H = node.clientHeight || 180;
  const margin = { top: 28, right: 16, bottom: 36, left: 44 };
  const w = W - margin.left - margin.right;
  const h = H - margin.top - margin.bottom;
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  svg.append('text').attr('x', W / 2).attr('y', 14)
    .attr('text-anchor', 'middle').attr('fill', '#aaa').attr('font-size', 12)
    .text('Pull Counts per Arm');

  const maxCount = Math.max(1, ...strategies.flatMap(s => s.counts));
  const armIndices = d3.range(K);
  const nStrats = strategies.length;
  const groupW = w / K;
  const barW = Math.max(4, (groupW - 4) / nStrats - 2);
  const yScale = d3.scaleLinear().domain([0, maxCount]).range([h, 0]);

  g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(d3.scaleBand().domain(armIndices.map(String)).range([0, w])).tickSize(0))
    .call(ax => ax.select('.domain').remove())
    .selectAll('text').attr('fill', '#999').attr('font-size', 11);
  g.append('g').call(d3.axisLeft(yScale).ticks(4))
    .call(ax => { ax.select('.domain').remove(); ax.selectAll('.tick line').attr('stroke', '#333'); ax.selectAll('text').attr('fill', '#999').attr('font-size', 11); });
  g.append('g').selectAll('line').data(yScale.ticks(4)).enter().append('line')
    .attr('x1', 0).attr('x2', w).attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
    .attr('stroke', '#2a2a3a').attr('stroke-dasharray', '3,3');

  armIndices.forEach(i => {
    const gx = (i / K) * w;
    strategies.forEach((s, si) => {
      const bx = gx + 2 + si * (barW + 2);
      const bh = Math.max(1, h - yScale(s.counts[i]));
      g.append('rect').attr('x', bx).attr('y', h - bh).attr('width', barW).attr('height', bh)
        .attr('fill', s.color).attr('opacity', 0.8);
    });
    g.append('text').attr('x', gx + groupW / 2).attr('y', h + 14)
      .attr('text-anchor', 'middle').attr('fill', '#ccc').attr('font-size', 11)
      .text(`A${i + 1}` + (optimalArm() === i ? ' ★' : ''));
  });
}

function renderRegret(): void {
  const svg = svgRegret;
  if (!svg) return;
  svg.selectAll('*').remove();

  const node = (svg.node() as SVGSVGElement);
  const W = node.clientWidth || 500;
  const H = node.clientHeight || 200;
  const margin = { top: 28, right: 16, bottom: 36, left: 52 };
  const w = W - margin.left - margin.right;
  const h = H - margin.top - margin.bottom;
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  svg.append('text').attr('x', W / 2).attr('y', 14)
    .attr('text-anchor', 'middle').attr('fill', '#aaa').attr('font-size', 12)
    .text('Cumulative Regret Over Time');

  if (stepCount === 0) return;

  const maxRegret = Math.max(1, ...strategies.map(s => s.cumRegret[s.cumRegret.length - 1]));
  const xScale = d3.scaleLinear().domain([0, stepCount]).range([0, w]);
  const yScale = d3.scaleLinear().domain([0, maxRegret]).range([h, 0]);

  g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(5))
    .call(ax => { ax.select('.domain').attr('stroke', '#444'); ax.selectAll('text').attr('fill', '#999').attr('font-size', 11); });
  g.append('g').call(d3.axisLeft(yScale).ticks(5))
    .call(ax => { ax.select('.domain').attr('stroke', '#444'); ax.selectAll('.tick line').attr('stroke', '#333'); ax.selectAll('text').attr('fill', '#999').attr('font-size', 11); });
  g.append('g').selectAll('line').data(yScale.ticks(5)).enter().append('line')
    .attr('x1', 0).attr('x2', w).attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
    .attr('stroke', '#2a2a3a').attr('stroke-dasharray', '3,3');

  const lineGen = d3.line<number>().x((_d, i) => xScale(i)).y(d => yScale(d));
  const downsample = (arr: number[], maxPts: number): number[] => {
    if (arr.length <= maxPts) return arr;
    const result: number[] = [];
    const step2 = arr.length / maxPts;
    for (let i = 0; i < maxPts; i++) {
      result.push(arr[Math.floor(i * step2)]);
    }
    return result;
  };

  strategies.forEach(s => {
    const pts = downsample(s.cumRegret, 500);
    g.append('path').datum(pts)
      .attr('fill', 'none').attr('stroke', s.color).attr('stroke-width', 2)
      .attr('d', lineGen);
  });

  // legend
  const lx = w - strategies.length * 110 - 8;
  strategies.forEach((s, si) => {
    const x = lx + si * 110;
    g.append('line').attr('x1', x).attr('x2', x + 20).attr('y1', 8).attr('y2', 8)
      .attr('stroke', s.color).attr('stroke-width', 2);
    g.append('text').attr('x', x + 24).attr('y', 12).attr('fill', s.color).attr('font-size', 10).text(s.label);
  });
}

function renderOptimal(): void {
  const svg = svgOptimal;
  if (!svg) return;
  svg.selectAll('*').remove();

  const node = (svg.node() as SVGSVGElement);
  const W = node.clientWidth || 500;
  const H = node.clientHeight || 200;
  const margin = { top: 28, right: 16, bottom: 36, left: 52 };
  const w = W - margin.left - margin.right;
  const h = H - margin.top - margin.bottom;
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  svg.append('text').attr('x', W / 2).attr('y', 14)
    .attr('text-anchor', 'middle').attr('fill', '#aaa').attr('font-size', 12)
    .text('% Optimal Action Over Time');

  if (stepCount === 0) return;

  const xScale = d3.scaleLinear().domain([0, stepCount]).range([0, w]);
  const yScale = d3.scaleLinear().domain([0, 100]).range([h, 0]);

  g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(xScale).ticks(5))
    .call(ax => { ax.select('.domain').attr('stroke', '#444'); ax.selectAll('text').attr('fill', '#999').attr('font-size', 11); });
  g.append('g').call(d3.axisLeft(yScale).ticks(5).tickFormat(d => `${d}%`))
    .call(ax => { ax.select('.domain').attr('stroke', '#444'); ax.selectAll('.tick line').attr('stroke', '#333'); ax.selectAll('text').attr('fill', '#999').attr('font-size', 11); });
  g.append('g').selectAll('line').data([0, 25, 50, 75, 100]).enter().append('line')
    .attr('x1', 0).attr('x2', w).attr('y1', d => yScale(d)).attr('y2', d => yScale(d))
    .attr('stroke', '#2a2a3a').attr('stroke-dasharray', '3,3');
  // chance line
  g.append('line').attr('x1', 0).attr('x2', w)
    .attr('y1', yScale(100 / K)).attr('y2', yScale(100 / K))
    .attr('stroke', '#444').attr('stroke-dasharray', '6,4').attr('stroke-width', 1);

  strategies.forEach(s => {
    const pct = s.totalPulls > 0 ? (s.optimalCount / s.totalPulls) * 100 : 0;
    // draw a horizontal dot (current %) and a smooth curve if we track history
    // For simplicity: draw a horizontal line at current pct
    g.append('line').attr('x1', 0).attr('x2', w)
      .attr('y1', yScale(pct)).attr('y2', yScale(pct))
      .attr('stroke', s.color).attr('stroke-width', 2).attr('stroke-dasharray', '5,3');
    g.append('text').attr('x', w + 4).attr('y', yScale(pct) + 4)
      .attr('fill', s.color).attr('font-size', 10).text(`${pct.toFixed(0)}%`);
  });

  // legend
  const lx = w - strategies.length * 110 - 8;
  strategies.forEach((s, si) => {
    const x = lx + si * 110;
    g.append('line').attr('x1', x).attr('x2', x + 20).attr('y1', 8).attr('y2', 8)
      .attr('stroke', s.color).attr('stroke-width', 2);
    g.append('text').attr('x', x + 24).attr('y', 12).attr('fill', s.color).attr('font-size', 10).text(s.label);
  });
}

// ---- Build UI ----
function buildUI(): void {
  const app = d3.select('#app');

  // Header
  app.append('h1').text('Multi-Armed Bandit Lab');
  app.append('p').attr('class', 'subtitle')
    .text('Compare exploration strategies on k-armed bandits with live regret and optimal-action tracking.');

  // Controls row
  const ctrl = app.append('div').attr('class', 'controls');

  // Arms
  const armGroup = ctrl.append('div').attr('class', 'ctrl-group');
  armGroup.append('label').text('Arms (k)');
  const armSel = armGroup.append('select').attr('id', 'sel-arms');
  [2, 3, 4, 5, 6, 8, 10].forEach(v => armSel.append('option').attr('value', v).text(String(v)).property('selected', v === K));
  armSel.on('change', function(this: HTMLSelectElement) {
    K = parseInt(this.value);
    resetAll();
  });

  // Reward type
  const rtGroup = ctrl.append('div').attr('class', 'ctrl-group');
  rtGroup.append('label').text('Reward Type');
  const rtSel = rtGroup.append('select').attr('id', 'sel-reward');
  rtSel.append('option').attr('value', 'bernoulli').text('Bernoulli').property('selected', true);
  rtSel.append('option').attr('value', 'gaussian').text('Gaussian');
  rtSel.on('change', function(this: HTMLSelectElement) {
    rewardType = this.value as RewardType;
    resetAll();
  });

  // Epsilon
  const epsGroup = ctrl.append('div').attr('class', 'ctrl-group');
  epsGroup.append('label').attr('id', 'lbl-eps').text(`ε = ${epsilon.toFixed(2)}`);
  epsGroup.append('input').attr('type', 'range').attr('min', 0).attr('max', 1).attr('step', 0.01).attr('value', epsilon)
    .on('input', function(this: HTMLInputElement) {
      epsilon = parseFloat(this.value);
      d3.select('#lbl-eps').text(`ε = ${epsilon.toFixed(2)}`);
    });

  // UCB c
  const ucbGroup = ctrl.append('div').attr('class', 'ctrl-group');
  ucbGroup.append('label').attr('id', 'lbl-ucb').text(`UCB c = ${ucbC.toFixed(1)}`);
  ucbGroup.append('input').attr('type', 'range').attr('min', 0.1).attr('max', 5).attr('step', 0.1).attr('value', ucbC)
    .on('input', function(this: HTMLInputElement) {
      ucbC = parseFloat(this.value);
      d3.select('#lbl-ucb').text(`UCB c = ${ucbC.toFixed(1)}`);
    });

  // Optimistic init
  const optGroup = ctrl.append('div').attr('class', 'ctrl-group');
  optGroup.append('label').attr('id', 'lbl-opt').text(`Opt Init = ${optInitVal.toFixed(1)}`);
  optGroup.append('input').attr('type', 'range').attr('min', 0.1).attr('max', 5).attr('step', 0.1).attr('value', optInitVal)
    .on('input', function(this: HTMLInputElement) {
      optInitVal = parseFloat(this.value);
      d3.select('#lbl-opt').text(`Opt Init = ${optInitVal.toFixed(1)}`);
    });

  // Strategy selector
  const stratGroup = ctrl.append('div').attr('class', 'ctrl-group');
  stratGroup.append('label').text('Strategy');
  const stratSel = stratGroup.append('select').attr('id', 'sel-strat');
  stratSel.append('option').attr('value', 'all').text('Compare All').property('selected', compareAll);
  STRATEGY_DEFS.forEach(d => {
    stratSel.append('option').attr('value', d.name).text(d.label).property('selected', !compareAll && activeStrategy === d.name);
  });
  stratSel.on('change', function(this: HTMLSelectElement) {
    if (this.value === 'all') { compareAll = true; }
    else { compareAll = false; activeStrategy = this.value as StrategyName; }
    resetAll();
  });

  // Speed
  const speedGroup = ctrl.append('div').attr('class', 'ctrl-group');
  speedGroup.append('label').attr('id', 'lbl-speed').text(`Speed: ${speed}`);
  speedGroup.append('input').attr('type', 'range').attr('min', 1).attr('max', 100).attr('step', 1).attr('value', speed)
    .on('input', function(this: HTMLInputElement) {
      speed = parseInt(this.value);
      d3.select('#lbl-speed').text(`Speed: ${speed}`);
      stepsPerFrame = speed > 80 ? 50 : speed > 60 ? 20 : speed > 40 ? 5 : 1;
    });

  // Buttons row
  const btns = app.append('div').attr('class', 'btn-row');
  const pp = btns.append('button').attr('id', 'btn-play').attr('class', 'btn-primary').text('▶ Play');
  playPauseBtn = pp.node() as HTMLButtonElement;
  pp.on('click', () => { if (running) pauseAnimation(); else startAnimation(); });

  btns.append('button').attr('class', 'btn-sec').text('⏭ Step').on('click', () => { stepOnce(); render(); });
  btns.append('button').attr('class', 'btn-sec').text('⏹ Reset').on('click', () => { pauseAnimation(); resetAll(); });
  btns.append('button').attr('class', 'btn-sec').text('🎲 Randomize Arms').on('click', () => {
    const wasRunning = running;
    pauseAnimation();
    resetAll();
    if (wasRunning) startAnimation();
  });

  // Run 1000 / 10000
  btns.append('button').attr('class', 'btn-sec').text('+1000 steps').on('click', () => {
    for (let i = 0; i < 1000; i++) stepOnce();
    render();
  });
  btns.append('button').attr('class', 'btn-sec').text('+10000 steps').on('click', () => {
    for (let i = 0; i < 10000; i++) stepOnce();
    render();
  });

  // Step counter
  const info = app.append('div').attr('class', 'info-row');
  info.append('span').text('Total steps: ');
  stepCountEl = info.append('span').attr('id', 'step-count').text('0').node() as HTMLElement;

  // Charts
  const charts = app.append('div').attr('class', 'charts');
  svgArms = charts.append('svg').attr('id', 'svg-arms').attr('class', 'chart-svg chart-tall') as d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
  svgCounts = charts.append('svg').attr('id', 'svg-counts').attr('class', 'chart-svg chart-med') as d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
  svgRegret = charts.append('svg').attr('id', 'svg-regret').attr('class', 'chart-svg chart-med') as d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
  svgOptimal = charts.append('svg').attr('id', 'svg-optimal').attr('class', 'chart-svg chart-med') as d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
}

// ---- Boot ----
document.addEventListener('DOMContentLoaded', () => {
  buildUI();
  resetAll();
});
