import * as d3 from 'd3';

export {};

// ── Types ────────────────────────────────────────────────────────────────────
interface HMMParams {
  states: string[];
  observations: string[];
  A: number[][];   // transition matrix [from][to]
  B: number[][];   // emission matrix [state][obs]
  pi: number[];    // initial distribution
}

interface Scenario {
  name: string;
  hmm: HMMParams;
  description: string;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
const SCENARIOS: Scenario[] = [
  {
    name: 'Weather → Activities',
    description: 'Hidden weather states (Sunny/Rainy) emit observable activities.',
    hmm: {
      states: ['Sunny', 'Rainy'],
      observations: ['Walk', 'Shop', 'Clean'],
      A: [
        [0.7, 0.3],
        [0.4, 0.6]
      ],
      B: [
        [0.6, 0.3, 0.1],
        [0.1, 0.4, 0.5]
      ],
      pi: [0.6, 0.4]
    }
  },
  {
    name: 'Casino: Fair vs Loaded Die',
    description: 'Casino switches between Fair and Loaded dice.',
    hmm: {
      states: ['Fair', 'Loaded'],
      observations: ['1', '2', '3', '4', '5', '6'],
      A: [
        [0.95, 0.05],
        [0.10, 0.90]
      ],
      B: [
        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        [0.10, 0.10, 0.10, 0.10, 0.10, 0.50]
      ],
      pi: [0.5, 0.5]
    }
  },
  {
    name: 'Health: Healthy vs Fever',
    description: 'Patient health (Healthy/Fever) inferred from symptoms.',
    hmm: {
      states: ['Healthy', 'Fever'],
      observations: ['Normal', 'Cold', 'Dizzy'],
      A: [
        [0.7, 0.3],
        [0.4, 0.6]
      ],
      B: [
        [0.5, 0.4, 0.1],
        [0.1, 0.3, 0.6]
      ],
      pi: [0.6, 0.4]
    }
  }
];

// ── Math helpers ─────────────────────────────────────────────────────────────
function logSum(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const mx = Math.max(a, b);
  return mx + Math.log(Math.exp(a - mx) + Math.exp(b - mx));
}

// ── HMM Algorithms ───────────────────────────────────────────────────────────
function forward(hmm: HMMParams, obs: number[]): { alpha: number[][], logProb: number } {
  const N = hmm.states.length;
  const T = obs.length;
  const alpha: number[][] = [];

  // Init
  const a0: number[] = [];
  for (let i = 0; i < N; i++) {
    a0.push(hmm.pi[i] * hmm.B[i][obs[0]]);
  }
  alpha.push(a0);

  // Recurse
  for (let t = 1; t < T; t++) {
    const at: number[] = [];
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let i = 0; i < N; i++) sum += alpha[t-1][i] * hmm.A[i][j];
      at.push(sum * hmm.B[j][obs[t]]);
    }
    alpha.push(at);
  }

  let logProb = -Infinity;
  for (let i = 0; i < N; i++) {
    logProb = logSum(logProb, Math.log(alpha[T-1][i] + 1e-300));
  }
  return { alpha, logProb };
}

function backward(hmm: HMMParams, obs: number[]): number[][] {
  const N = hmm.states.length;
  const T = obs.length;
  const beta: number[][] = new Array(T);

  beta[T-1] = new Array(N).fill(1);
  for (let t = T-2; t >= 0; t--) {
    beta[t] = [];
    for (let i = 0; i < N; i++) {
      let sum = 0;
      for (let j = 0; j < N; j++) {
        sum += hmm.A[i][j] * hmm.B[j][obs[t+1]] * beta[t+1][j];
      }
      beta[t].push(sum);
    }
  }
  return beta;
}

function forwardBackward(hmm: HMMParams, obs: number[]): {
  alpha: number[][], beta: number[][], gamma: number[][], xi: number[][][], logProb: number
} {
  const N = hmm.states.length;
  const T = obs.length;
  const { alpha, logProb } = forward(hmm, obs);
  const beta = backward(hmm, obs);

  // gamma[t][i]
  const gamma: number[][] = [];
  for (let t = 0; t < T; t++) {
    const gt: number[] = [];
    let sum = 0;
    for (let i = 0; i < N; i++) sum += alpha[t][i] * beta[t][i];
    for (let i = 0; i < N; i++) gt.push((alpha[t][i] * beta[t][i]) / (sum + 1e-300));
    gamma.push(gt);
  }

  // xi[t][i][j]
  const xi: number[][][] = [];
  for (let t = 0; t < T-1; t++) {
    const xt: number[][] = [];
    let denom = 0;
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++)
        denom += alpha[t][i] * hmm.A[i][j] * hmm.B[j][obs[t+1]] * beta[t+1][j];
    for (let i = 0; i < N; i++) {
      const row: number[] = [];
      for (let j = 0; j < N; j++) {
        row.push((alpha[t][i] * hmm.A[i][j] * hmm.B[j][obs[t+1]] * beta[t+1][j]) / (denom + 1e-300));
      }
      xt.push(row);
    }
    xi.push(xt);
  }

  return { alpha, beta, gamma, xi, logProb };
}

function viterbi(hmm: HMMParams, obs: number[]): { path: number[], delta: number[][], psi: number[][] } {
  const N = hmm.states.length;
  const T = obs.length;
  const delta: number[][] = [];
  const psi: number[][] = [];

  // Init (log space)
  const d0: number[] = [];
  const p0: number[] = [];
  for (let i = 0; i < N; i++) {
    d0.push(Math.log(hmm.pi[i] + 1e-300) + Math.log(hmm.B[i][obs[0]] + 1e-300));
    p0.push(0);
  }
  delta.push(d0);
  psi.push(p0);

  for (let t = 1; t < T; t++) {
    const dt: number[] = [];
    const pt: number[] = [];
    for (let j = 0; j < N; j++) {
      let best = -Infinity;
      let bestPsi = 0;
      for (let i = 0; i < N; i++) {
        const val = delta[t-1][i] + Math.log(hmm.A[i][j] + 1e-300);
        if (val > best) { best = val; bestPsi = i; }
      }
      dt.push(best + Math.log(hmm.B[j][obs[t]] + 1e-300));
      pt.push(bestPsi);
    }
    delta.push(dt);
    psi.push(pt);
  }

  // Backtrack
  const path: number[] = new Array(T);
  let best = -Infinity;
  for (let i = 0; i < N; i++) {
    if (delta[T-1][i] > best) { best = delta[T-1][i]; path[T-1] = i; }
  }
  for (let t = T-2; t >= 0; t--) {
    path[t] = psi[t+1][path[t+1]];
  }

  return { path, delta, psi };
}

function baumWelch(
  initHmm: HMMParams,
  obsSeqs: number[][],
  iterations: number,
  onStep: (hmm: HMMParams, iter: number, ll: number) => void
): HMMParams {
  const N = initHmm.states.length;
  const M = initHmm.observations.length;
  let hmm: HMMParams = JSON.parse(JSON.stringify(initHmm));

  for (let iter = 0; iter < iterations; iter++) {
    // Accumulators
    const piAcc = new Array(N).fill(0);
    const AAcc: number[][] = Array.from({length: N}, () => new Array(N).fill(0));
    const BAcc: number[][] = Array.from({length: N}, () => new Array(M).fill(0));
    let totalLL = 0;

    obsSeqs.forEach(obs => {
      const { alpha, gamma, xi, logProb } = forwardBackward(hmm, obs);
      totalLL += logProb;
      const T = obs.length;

      for (let i = 0; i < N; i++) piAcc[i] += gamma[0][i];
      for (let t = 0; t < T-1; t++) {
        for (let i = 0; i < N; i++) {
          for (let j = 0; j < N; j++) AAcc[i][j] += xi[t][i][j];
        }
      }
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < N; i++) BAcc[i][obs[t]] += gamma[t][i];
      }
    });

    // Normalize pi
    let piSum = piAcc.reduce((a, b) => a + b, 0);
    for (let i = 0; i < N; i++) hmm.pi[i] = piAcc[i] / (piSum + 1e-300);

    // Normalize A
    for (let i = 0; i < N; i++) {
      let rowSum = AAcc[i].reduce((a, b) => a + b, 0);
      for (let j = 0; j < N; j++) hmm.A[i][j] = AAcc[i][j] / (rowSum + 1e-300);
    }

    // Normalize B
    for (let i = 0; i < N; i++) {
      let rowSum = BAcc[i].reduce((a, b) => a + b, 0);
      for (let k = 0; k < M; k++) hmm.B[i][k] = BAcc[i][k] / (rowSum + 1e-300);
    }

    onStep(JSON.parse(JSON.stringify(hmm)), iter + 1, totalLL);
  }

  return hmm;
}

// ── State ─────────────────────────────────────────────────────────────────────
let currentScenario = 0;
let currentHmm: HMMParams = JSON.parse(JSON.stringify(SCENARIOS[0].hmm));
let trainedHmm: HMMParams | null = null;
let currentObsSeq: number[] = [];
let viterbiResult: { path: number[], delta: number[][], psi: number[][] } | null = null;
let fwdResult: { alpha: number[][], logProb: number } | null = null;
let llHistory: number[] = [];
let bwIterSnapshots: HMMParams[] = [];

// ── Dimensions ────────────────────────────────────────────────────────────────
const W = 900, H = 520;
const margin = { top: 20, right: 20, bottom: 20, left: 20 };

// ── SVG setup ─────────────────────────────────────────────────────────────────
const svg = d3.select('#main-svg')
  .attr('width', W)
  .attr('height', H);

const transitionG = svg.append('g').attr('id', 'transition-g').attr('transform', 'translate(20,30)');
const emissionG = svg.append('g').attr('id', 'emission-g').attr('transform', 'translate(290,30)');
const trellisG = svg.append('g').attr('id', 'trellis-g').attr('transform', 'translate(20,290)');
const llG = svg.append('g').attr('id', 'll-g').attr('transform', 'translate(650,290)');

// ── Arrowhead marker ──────────────────────────────────────────────────────────
svg.append('defs').append('marker')
  .attr('id', 'arrow')
  .attr('viewBox', '0 -5 10 10')
  .attr('refX', 10)
  .attr('refY', 0)
  .attr('markerWidth', 6)
  .attr('markerHeight', 6)
  .attr('orient', 'auto')
  .append('path')
  .attr('d', 'M0,-5L10,0L0,5')
  .attr('fill', '#aaa');

svg.append('defs').append('marker')
  .attr('id', 'arrow-viterbi')
  .attr('viewBox', '0 -5 10 10')
  .attr('refX', 10)
  .attr('refY', 0)
  .attr('markerWidth', 6)
  .attr('markerHeight', 6)
  .attr('orient', 'auto')
  .append('path')
  .attr('d', 'M0,-5L10,0L0,5')
  .attr('fill', '#f9a825');

// ── Draw Transition Diagram ───────────────────────────────────────────────────
function drawTransitionDiagram(hmm: HMMParams) {
  const g = transitionG;
  g.selectAll('*').remove();

  g.append('text').attr('x', 110).attr('y', -5)
    .attr('text-anchor', 'middle')
    .attr('fill', '#90caf9').attr('font-size', 13).attr('font-weight', 'bold')
    .text('State Transition Diagram');

  const N = hmm.states.length;
  const cx = 110, cy = 110, r = 70;
  const nodeR = 30;

  // Compute node positions in a circle
  const positions: {x: number, y: number}[] = [];
  for (let i = 0; i < N; i++) {
    const angle = (2 * Math.PI * i / N) - Math.PI / 2;
    positions.push({ x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) });
  }

  const colorScale = d3.scaleOrdinal<string>()
    .domain(hmm.states)
    .range(['#1976d2', '#c62828', '#2e7d32', '#f57c00']);

  // Draw self-loops and transition edges
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const prob = hmm.A[i][j];
      if (prob < 0.01) continue;
      const strokeW = 0.5 + prob * 4;

      if (i === j) {
        // Self-loop
        const { x, y } = positions[i];
        const loopR = 18;
        const offsetAngle = (2 * Math.PI * i / N) - Math.PI / 2;
        const lx = x + (nodeR + loopR) * Math.cos(offsetAngle);
        const ly = y + (nodeR + loopR) * Math.sin(offsetAngle);
        g.append('circle')
          .attr('cx', lx).attr('cy', ly).attr('r', loopR)
          .attr('fill', 'none')
          .attr('stroke', colorScale(hmm.states[i]))
          .attr('stroke-width', strokeW)
          .attr('opacity', 0.7)
          .attr('marker-end', 'url(#arrow)');
        g.append('text')
          .attr('x', lx).attr('y', ly)
          .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
          .attr('fill', '#ccc').attr('font-size', 10)
          .text(prob.toFixed(2));
      } else {
        // Curved edge
        const s = positions[i], t = positions[j];
        const dx = t.x - s.x, dy = t.y - s.y;
        const len = Math.sqrt(dx*dx + dy*dy);
        const curve = 0.25;
        const mx = (s.x + t.x) / 2 - curve * dy;
        const my = (s.y + t.y) / 2 + curve * dx;

        // End point on circle boundary
        const ex = t.x - (nodeR + 2) * dx / len;
        const ey = t.y - (nodeR + 2) * dy / len;

        g.append('path')
          .attr('d', `M${s.x},${s.y} Q${mx},${my} ${ex},${ey}`)
          .attr('fill', 'none')
          .attr('stroke', colorScale(hmm.states[i]))
          .attr('stroke-width', strokeW)
          .attr('opacity', 0.8)
          .attr('marker-end', 'url(#arrow)');

        g.append('text')
          .attr('x', mx).attr('y', my)
          .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
          .attr('fill', '#eee').attr('font-size', 10)
          .text(prob.toFixed(2));
      }
    }
  }

  // Nodes
  positions.forEach((pos, i) => {
    g.append('circle')
      .attr('cx', pos.x).attr('cy', pos.y).attr('r', nodeR)
      .attr('fill', colorScale(hmm.states[i]))
      .attr('stroke', '#fff').attr('stroke-width', 2)
      .attr('opacity', 0.9);
    g.append('text')
      .attr('x', pos.x).attr('y', pos.y)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
      .attr('fill', '#fff').attr('font-size', 11).attr('font-weight', 'bold')
      .text(hmm.states[i]);
    // π label
    g.append('text')
      .attr('x', pos.x).attr('y', pos.y + nodeR + 13)
      .attr('text-anchor', 'middle')
      .attr('fill', '#aaa').attr('font-size', 10)
      .text(`π=${hmm.pi[i].toFixed(2)}`);
  });
}

// ── Draw Emission Heatmap ─────────────────────────────────────────────────────
function drawEmissionHeatmap(hmm: HMMParams) {
  const g = emissionG;
  g.selectAll('*').remove();

  const N = hmm.states.length;
  const M = hmm.observations.length;
  const cellW = Math.min(55, 300 / M);
  const cellH = 36;
  const padLeft = 55, padTop = 25;

  g.append('text').attr('x', padLeft + (M * cellW) / 2).attr('y', -5)
    .attr('text-anchor', 'middle')
    .attr('fill', '#90caf9').attr('font-size', 13).attr('font-weight', 'bold')
    .text('Emission Matrix B');

  const colorFn = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);

  // Col headers
  hmm.observations.forEach((obs, k) => {
    g.append('text')
      .attr('x', padLeft + k * cellW + cellW / 2)
      .attr('y', padTop - 6)
      .attr('text-anchor', 'middle')
      .attr('fill', '#bbb').attr('font-size', 11)
      .text(obs);
  });

  // Rows
  hmm.states.forEach((state, i) => {
    g.append('text')
      .attr('x', padLeft - 5)
      .attr('y', padTop + i * cellH + cellH / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .attr('fill', '#bbb').attr('font-size', 11)
      .text(state);

    hmm.observations.forEach((_obs, k) => {
      const val = hmm.B[i][k];
      g.append('rect')
        .attr('x', padLeft + k * cellW)
        .attr('y', padTop + i * cellH)
        .attr('width', cellW - 2)
        .attr('height', cellH - 2)
        .attr('fill', colorFn(val))
        .attr('rx', 3);
      g.append('text')
        .attr('x', padLeft + k * cellW + cellW / 2)
        .attr('y', padTop + i * cellH + cellH / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', val > 0.5 ? '#111' : '#eee')
        .attr('font-size', 11)
        .text(val.toFixed(2));
    });
  });
}

// ── Draw Trellis ──────────────────────────────────────────────────────────────
function drawTrellis(
  hmm: HMMParams,
  obs: number[],
  vr: { path: number[], delta: number[][] } | null,
  fwd: { alpha: number[][] } | null
) {
  const g = trellisG;
  g.selectAll('*').remove();

  if (obs.length === 0) return;

  const N = hmm.states.length;
  const T = obs.length;
  const padLeft = 70, padTop = 20;
  const trellisW = 580, trellisH = 180;
  const stepW = Math.min(70, (trellisW - padLeft) / T);
  const stepH = Math.min(60, (trellisH - padTop) / N);

  g.append('text').attr('x', padLeft + T * stepW / 2).attr('y', -5)
    .attr('text-anchor', 'middle')
    .attr('fill', '#90caf9').attr('font-size', 13).attr('font-weight', 'bold')
    .text('Trellis / Viterbi Path');

  // Color by forward probability
  const allAlpha: number[] = [];
  if (fwd) fwd.alpha.forEach(row => row.forEach(v => allAlpha.push(v)));
  const maxAlpha = allAlpha.length > 0 ? Math.max(...allAlpha) : 1;
  const alphaColor = d3.scaleSequential(d3.interpolateGreens).domain([0, maxAlpha]);

  const colorScale = d3.scaleOrdinal<string>()
    .domain(hmm.states)
    .range(['#1976d2', '#c62828', '#2e7d32', '#f57c00']);

  // Node positions
  const pos = (t: number, i: number) => ({
    x: padLeft + t * stepW + stepW / 2,
    y: padTop + i * stepH + stepH / 2
  });

  // Draw Viterbi path edges first
  if (vr) {
    for (let t = 0; t < T - 1; t++) {
      const from = vr.path[t], to = vr.path[t+1];
      const s = pos(t, from), e = pos(t+1, to);
      g.append('line')
        .attr('x1', s.x).attr('y1', s.y)
        .attr('x2', e.x).attr('y2', e.y)
        .attr('stroke', '#f9a825').attr('stroke-width', 3).attr('opacity', 0.9);
    }
  }

  // All edges (thin)
  for (let t = 0; t < T-1; t++) {
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (hmm.A[i][j] < 0.01) continue;
        const s = pos(t, i), e = pos(t+1, j);
        const isViterbi = vr && vr.path[t] === i && vr.path[t+1] === j;
        if (!isViterbi) {
          g.append('line')
            .attr('x1', s.x).attr('y1', s.y)
            .attr('x2', e.x).attr('y2', e.y)
            .attr('stroke', '#444').attr('stroke-width', 0.8).attr('opacity', 0.5);
        }
      }
    }
  }

  // State labels on left
  hmm.states.forEach((state, i) => {
    g.append('text')
      .attr('x', padLeft - 5)
      .attr('y', padTop + i * stepH + stepH / 2)
      .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
      .attr('fill', colorScale(state)).attr('font-size', 11).attr('font-weight', 'bold')
      .text(state);
  });

  // Observation labels at top
  for (let t = 0; t < T; t++) {
    g.append('text')
      .attr('x', pos(t, 0).x).attr('y', 10)
      .attr('text-anchor', 'middle')
      .attr('fill', '#aaa').attr('font-size', 10)
      .text(hmm.observations[obs[t]]);
  }

  // Nodes
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < N; i++) {
      const p = pos(t, i);
      const isPath = vr && vr.path[t] === i;
      const alphaVal = fwd ? fwd.alpha[t][i] : 0;
      g.append('circle')
        .attr('cx', p.x).attr('cy', p.y).attr('r', 14)
        .attr('fill', isPath ? '#f9a825' : (fwd ? alphaColor(alphaVal) : '#333'))
        .attr('stroke', isPath ? '#fff' : colorScale(hmm.states[i]))
        .attr('stroke-width', isPath ? 2.5 : 1.5);

      if (fwd) {
        g.append('text')
          .attr('x', p.x).attr('y', p.y)
          .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
          .attr('fill', isPath ? '#111' : '#eee').attr('font-size', 9)
          .text(alphaVal.toExponential(1));
      }
    }
  }

  // Legend
  if (vr) {
    g.append('circle').attr('cx', padLeft).attr('cy', trellisH + 10).attr('r', 6).attr('fill', '#f9a825');
    g.append('text').attr('x', padLeft + 12).attr('y', trellisH + 10)
      .attr('dominant-baseline', 'middle').attr('fill', '#ccc').attr('font-size', 11)
      .text('Viterbi path');
    if (fwd) {
      g.append('text').attr('x', padLeft + 120).attr('y', trellisH + 10)
        .attr('dominant-baseline', 'middle').attr('fill', '#aaa').attr('font-size', 11)
        .text('Node color = forward prob α');
    }
  }
}

// ── Draw LL Chart ─────────────────────────────────────────────────────────────
function drawLLChart(ll: number[]) {
  const g = llG;
  g.selectAll('*').remove();

  const cw = 220, ch = 180;
  const pad = { top: 20, right: 10, bottom: 30, left: 50 };

  g.append('text').attr('x', cw / 2).attr('y', -5)
    .attr('text-anchor', 'middle')
    .attr('fill', '#90caf9').attr('font-size', 13).attr('font-weight', 'bold')
    .text('Baum-Welch Log-Likelihood');

  if (ll.length === 0) {
    g.append('text').attr('x', cw / 2).attr('y', ch / 2)
      .attr('text-anchor', 'middle').attr('fill', '#666').attr('font-size', 12)
      .text('Run Baum-Welch to see');
    return;
  }

  const innerW = cw - pad.left - pad.right;
  const innerH = ch - pad.top - pad.bottom;

  const xScale = d3.scaleLinear().domain([0, ll.length - 1]).range([0, innerW]);
  const yMin = Math.min(...ll), yMax = Math.max(...ll);
  const yScale = d3.scaleLinear().domain([yMin - 1, yMax + 1]).range([innerH, 0]);

  const ig = g.append('g').attr('transform', `translate(${pad.left},${pad.top})`);

  // Axes
  ig.append('g').attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => `${d}`))
    .selectAll('text,line,path').attr('stroke', '#555').attr('fill', '#888');
  ig.append('g')
    .call(d3.axisLeft(yScale).ticks(5))
    .selectAll('text,line,path').attr('stroke', '#555').attr('fill', '#888');

  // Line
  const line = d3.line<number>()
    .x((_d, i) => xScale(i))
    .y(d => yScale(d));
  ig.append('path')
    .datum(ll)
    .attr('d', line)
    .attr('fill', 'none')
    .attr('stroke', '#66bb6a')
    .attr('stroke-width', 2);

  ig.append('text').attr('x', -pad.left + 5).attr('y', innerH / 2)
    .attr('transform', `rotate(-90,${-pad.left + 10},${innerH / 2})`)
    .attr('fill', '#888').attr('font-size', 10)
    .text('Log-Likelihood');
  ig.append('text').attr('x', innerW / 2).attr('y', innerH + 25)
    .attr('text-anchor', 'middle').attr('fill', '#888').attr('font-size', 10)
    .text('Iteration');
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function getActiveHmm(): HMMParams {
  return trainedHmm ? trainedHmm : currentHmm;
}

function obsToIndices(hmm: HMMParams, text: string): number[] {
  const tokens = text.trim().split(/[\s,]+/).filter(s => s.length > 0);
  const result: number[] = [];
  tokens.forEach(t => {
    const idx = hmm.observations.indexOf(t);
    if (idx >= 0) result.push(idx);
  });
  return result;
}

function indicesToStr(hmm: HMMParams, indices: number[]): string {
  return indices.map(i => hmm.observations[i]).join(' ');
}

function randomObsSeq(hmm: HMMParams, len: number): number[] {
  const N = hmm.states.length;
  const M = hmm.observations.length;
  const seq: number[] = [];
  let state = 0;
  // Sample initial state
  let r = Math.random();
  let cumPi = 0;
  for (let i = 0; i < N; i++) {
    cumPi += hmm.pi[i];
    if (r < cumPi) { state = i; break; }
  }
  for (let t = 0; t < len; t++) {
    // Sample observation
    let ro = Math.random();
    let cumB = 0;
    let obs = 0;
    for (let k = 0; k < M; k++) {
      cumB += hmm.B[state][k];
      if (ro < cumB) { obs = k; break; }
    }
    seq.push(obs);
    // Transition
    let ra = Math.random();
    let cumA = 0;
    let nextState = 0;
    for (let j = 0; j < N; j++) {
      cumA += hmm.A[state][j];
      if (ra < cumA) { nextState = j; break; }
    }
    state = nextState;
  }
  return seq;
}

function updateStatusBar(msg: string) {
  const el = document.getElementById('status-bar');
  if (el) el.textContent = msg;
}

function renderAll() {
  const hmm = getActiveHmm();
  drawTransitionDiagram(hmm);
  drawEmissionHeatmap(hmm);
  drawTrellis(hmm, currentObsSeq, viterbiResult, fwdResult);
  drawLLChart(llHistory);
}

// ── Event wiring ──────────────────────────────────────────────────────────────
function init() {
  const scenarioSel = document.getElementById('scenario-sel') as HTMLSelectElement;
  const obsInput = document.getElementById('obs-input') as HTMLInputElement;
  const genBtn = document.getElementById('gen-btn') as HTMLButtonElement;
  const viterbiBtn = document.getElementById('viterbi-btn') as HTMLButtonElement;
  const fwdBtn = document.getElementById('fwd-btn') as HTMLButtonElement;
  const bwBtn = document.getElementById('bw-btn') as HTMLButtonElement;
  const resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;
  const iterInput = document.getElementById('iter-input') as HTMLInputElement;
  const seqLenInput = document.getElementById('seqlen-input') as HTMLInputElement;
  const descEl = document.getElementById('scenario-desc') as HTMLElement;

  // Populate scenario select
  SCENARIOS.forEach((sc, i) => {
    const opt = document.createElement('option');
    opt.value = `${i}`;
    opt.textContent = sc.name;
    scenarioSel.appendChild(opt);
  });

  function loadScenario(idx: number) {
    currentScenario = idx;
    currentHmm = JSON.parse(JSON.stringify(SCENARIOS[idx].hmm));
    trainedHmm = null;
    viterbiResult = null;
    fwdResult = null;
    llHistory = [];
    currentObsSeq = [];
    obsInput.value = '';
    descEl.textContent = SCENARIOS[idx].description;
    updateStatusBar(`Loaded scenario: ${SCENARIOS[idx].name}`);
    renderAll();
  }

  scenarioSel.addEventListener('change', () => loadScenario(parseInt(scenarioSel.value)));

  genBtn.addEventListener('click', () => {
    const hmm = getActiveHmm();
    const len = parseInt(seqLenInput.value) || 10;
    currentObsSeq = randomObsSeq(hmm, len);
    obsInput.value = indicesToStr(hmm, currentObsSeq);
    viterbiResult = null;
    fwdResult = null;
    renderAll();
    updateStatusBar(`Generated sequence of length ${len}`);
  });

  function parseObs() {
    const hmm = getActiveHmm();
    const text = obsInput.value;
    const indices = obsToIndices(hmm, text);
    if (indices.length === 0) {
      updateStatusBar('No valid observations parsed. Use space-separated observation names.');
      return false;
    }
    currentObsSeq = indices;
    return true;
  }

  viterbiBtn.addEventListener('click', () => {
    if (!parseObs()) return;
    const hmm = getActiveHmm();
    viterbiResult = viterbi(hmm, currentObsSeq);
    const pathNames = viterbiResult.path.map(i => hmm.states[i]).join(' → ');
    updateStatusBar(`Viterbi path: ${pathNames}`);
    renderAll();
  });

  fwdBtn.addEventListener('click', () => {
    if (!parseObs()) return;
    const hmm = getActiveHmm();
    const { alpha, logProb } = forward(hmm, currentObsSeq);
    fwdResult = { alpha, logProb };
    updateStatusBar(`Forward prob: P(obs) = exp(${logProb.toFixed(4)}) ≈ ${Math.exp(logProb).toExponential(4)}`);
    renderAll();
  });

  bwBtn.addEventListener('click', () => {
    if (!parseObs()) return;
    const iterations = parseInt(iterInput.value) || 20;
    llHistory = [];
    bwIterSnapshots = [];
    trainedHmm = null;
    viterbiResult = null;
    fwdResult = null;

    // Animate step by step with setTimeout
    let iter = 0;
    const baseHmm: HMMParams = JSON.parse(JSON.stringify(currentHmm));
    let hmm: HMMParams = JSON.parse(JSON.stringify(baseHmm));

    function step() {
      if (iter >= iterations) {
        trainedHmm = hmm;
        updateStatusBar(`Baum-Welch done. Final LL: ${llHistory[llHistory.length-1].toFixed(4)}`);
        renderAll();
        return;
      }
      // One iteration
      const N = hmm.states.length;
      const M = hmm.observations.length;
      const obsSeqs = [currentObsSeq];
      const piAcc = new Array(N).fill(0);
      const AAcc: number[][] = Array.from({length: N}, () => new Array(N).fill(0));
      const BAcc: number[][] = Array.from({length: N}, () => new Array(M).fill(0));
      let totalLL = 0;

      obsSeqs.forEach(obs => {
        const { gamma, xi, logProb } = forwardBackward(hmm, obs);
        totalLL += logProb;
        const T = obs.length;
        for (let i = 0; i < N; i++) piAcc[i] += gamma[0][i];
        for (let t = 0; t < T-1; t++) {
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) AAcc[i][j] += xi[t][i][j];
          }
        }
        for (let t = 0; t < T; t++) {
          for (let i = 0; i < N; i++) BAcc[i][obs[t]] += gamma[t][i];
        }
      });

      let piSum = piAcc.reduce((a: number, b: number) => a + b, 0);
      for (let i = 0; i < N; i++) hmm.pi[i] = piAcc[i] / (piSum + 1e-300);
      for (let i = 0; i < N; i++) {
        let rowSum = AAcc[i].reduce((a: number, b: number) => a + b, 0);
        for (let j = 0; j < N; j++) hmm.A[i][j] = AAcc[i][j] / (rowSum + 1e-300);
      }
      for (let i = 0; i < N; i++) {
        let rowSum = BAcc[i].reduce((a: number, b: number) => a + b, 0);
        for (let k = 0; k < M; k++) hmm.B[i][k] = BAcc[i][k] / (rowSum + 1e-300);
      }

      llHistory.push(totalLL);
      iter++;

      // Update displays mid-animation
      trainedHmm = JSON.parse(JSON.stringify(hmm));
      drawTransitionDiagram(trainedHmm!);
      drawEmissionHeatmap(trainedHmm!);
      drawLLChart(llHistory);
      updateStatusBar(`Baum-Welch iter ${iter}/${iterations}: LL = ${totalLL.toFixed(4)}`);

      setTimeout(step, 80);
    }
    step();
  });

  resetBtn.addEventListener('click', () => {
    trainedHmm = null;
    viterbiResult = null;
    fwdResult = null;
    llHistory = [];
    updateStatusBar('Reset. Parameters restored to original scenario.');
    renderAll();
  });

  // Initial load
  loadScenario(0);
}

document.addEventListener('DOMContentLoaded', init);
