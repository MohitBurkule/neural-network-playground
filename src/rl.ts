import * as d3 from 'd3';

// ---- Types ----
type CellType = 'empty' | 'wall' | 'goal' | 'pit' | 'start';
type Action = 0 | 1 | 2 | 3; // up, down, left, right
const ACTIONS: Action[] = [0, 1, 2, 3];
const ACTION_LABELS = ['↑', '↓', '←', '→'];
const ACTION_DELTAS: [number, number][] = [[-1, 0], [1, 0], [0, -1], [0, 1]];

interface Cell {
  type: CellType;
}

interface AgentPos {
  row: number;
  col: number;
}

// ---- State ----
const ROWS = 7;
const COLS = 10;
let grid: Cell[][] = [];
let Q: number[][][] = []; // Q[row][col][action]
let episode = 0;
let step = 0;
let totalReward = 0;
let rewardHistory: number[] = [];
let agentPos: AgentPos = { row: 0, col: 0 };
let episodeReward = 0;
let running = false;
let animFrame: number | null = null;
let stepInterval: ReturnType<typeof setInterval> | null = null;
let paletteSelected: CellType = 'wall';
let hoveredCell: AgentPos | null = null;
let selectedCell: AgentPos | null = null;
let speedMs = 80;
let alpha = 0.1;
let gamma = 0.95;
let epsilon = 0.2;
let epsilonDecay = false;
let algorithm: 'qlearning' | 'sarsa' = 'qlearning';
let episodeInProgress = false;
let sarsaNextAction: Action | null = null;
let startPos: AgentPos = { row: 0, col: 0 };

// ---- Grid init ----
function initGrid() {
  grid = [];
  for (let r = 0; r < ROWS; r++) {
    grid.push([]);
    for (let c = 0; c < COLS; c++) {
      grid[r].push({ type: 'empty' });
    }
  }
  // Default layout
  grid[0][0].type = 'start';
  grid[ROWS - 1][COLS - 1].type = 'goal';
  // Some walls
  const walls: [number, number][] = [[1,2],[2,2],[3,2],[4,2],[2,5],[2,6],[2,7],[5,7],[4,7],[3,7]];
  walls.forEach(([r, c]) => { grid[r][c].type = 'wall'; });
  // Some pits
  const pits: [number, number][] = [[1,4],[3,5],[5,4]];
  pits.forEach(([r, c]) => { grid[r][c].type = 'pit'; });
  startPos = findCell('start') || { row: 0, col: 0 };
}

function findCell(type: CellType): AgentPos | null {
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (grid[r][c].type === type) return { row: r, col: c };
    }
  }
  return null;
}

function initQ() {
  Q = [];
  for (let r = 0; r < ROWS; r++) {
    Q.push([]);
    for (let c = 0; c < COLS; c++) {
      Q[r].push([0, 0, 0, 0]);
    }
  }
  episode = 0;
  step = 0;
  rewardHistory = [];
  episodeReward = 0;
  episodeInProgress = false;
  sarsaNextAction = null;
  agentPos = { ...startPos };
  renderAll();
}

// ---- RL logic ----
function reward(r: number, c: number): number {
  const t = grid[r][c].type;
  if (t === 'goal') return 1;
  if (t === 'pit') return -1;
  return -0.01;
}

function isTerminal(r: number, c: number): boolean {
  const t = grid[r][c].type;
  return t === 'goal' || t === 'pit';
}

function chooseAction(r: number, c: number, eps: number): Action {
  if (Math.random() < eps) {
    return ACTIONS[Math.floor(Math.random() * 4)];
  }
  return greedyAction(r, c);
}

function greedyAction(r: number, c: number): Action {
  let best: Action = 0;
  let bestQ = -Infinity;
  for (const a of ACTIONS) {
    if (Q[r][c][a] > bestQ) {
      bestQ = Q[r][c][a];
      best = a;
    }
  }
  return best;
}

function nextState(r: number, c: number, a: Action): [number, number] {
  const [dr, dc] = ACTION_DELTAS[a];
  const nr = r + dr;
  const nc = c + dc;
  if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS || grid[nr][nc].type === 'wall') {
    return [r, c];
  }
  return [nr, nc];
}

function currentEpsilon(): number {
  if (!epsilonDecay) return epsilon;
  return Math.max(0.01, epsilon * Math.pow(0.995, episode));
}

function doStep(): boolean {
  // Returns true if episode ended
  const { row: r, col: c } = agentPos;
  if (isTerminal(r, c)) {
    // start new episode
    rewardHistory.push(episodeReward);
    episode++;
    episodeReward = 0;
    agentPos = { ...startPos };
    sarsaNextAction = null;
    episodeInProgress = false;
    return true;
  }

  const eps = currentEpsilon();

  if (algorithm === 'qlearning') {
    const a = chooseAction(r, c, eps);
    const [nr, nc] = nextState(r, c, a);
    const rew = reward(nr, nc);
    const maxNextQ = Math.max(...Q[nr][nc]);
    Q[r][c][a] += alpha * (rew + gamma * maxNextQ - Q[r][c][a]);
    agentPos = { row: nr, col: nc };
    episodeReward += rew;
  } else {
    // SARSA
    if (sarsaNextAction === null) {
      sarsaNextAction = chooseAction(r, c, eps);
    }
    const a = sarsaNextAction;
    const [nr, nc] = nextState(r, c, a);
    const rew = reward(nr, nc);
    const nextA = isTerminal(nr, nc) ? (0 as Action) : chooseAction(nr, nc, eps);
    const nextQVal = isTerminal(nr, nc) ? 0 : Q[nr][nc][nextA];
    Q[r][c][a] += alpha * (rew + gamma * nextQVal - Q[r][c][a]);
    sarsaNextAction = nextA;
    agentPos = { row: nr, col: nc };
    episodeReward += rew;
  }

  step++;
  episodeInProgress = true;

  if (isTerminal(agentPos.row, agentPos.col)) {
    rewardHistory.push(episodeReward);
    episode++;
    episodeReward = 0;
    agentPos = { ...startPos };
    sarsaNextAction = null;
    episodeInProgress = false;
    return true;
  }
  return false;
}

function doEpisode() {
  let maxSteps = ROWS * COLS * 10;
  agentPos = { ...startPos };
  episodeReward = 0;
  sarsaNextAction = null;
  while (maxSteps-- > 0) {
    const done = doStep();
    if (done) break;
  }
  renderAll();
}

// ---- Canvas / SVG sizes ----
const CELL = 64;
const PAD = 2;
const CHART_H = 120;

// ---- Color scales ----
const valueScale = d3.scaleSequential(d3.interpolateRdYlGn).domain([-1, 1]);

// ---- Draw ----
let canvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;
let chartSvg: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
let qPanel: HTMLElement;
let statusEl: HTMLElement;

function stateValue(r: number, c: number): number {
  return Math.max(...Q[r][c]);
}

function drawGrid() {
  const W = COLS * (CELL + PAD) + PAD;
  const H = ROWS * (CELL + PAD) + PAD;
  ctx.clearRect(0, 0, W, H);

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const x = PAD + c * (CELL + PAD);
      const y = PAD + r * (CELL + PAD);
      const cell = grid[r][c];

      // Background
      if (cell.type === 'wall') {
        ctx.fillStyle = '#2a2a3e';
      } else if (cell.type === 'goal') {
        ctx.fillStyle = '#16a34a';
      } else if (cell.type === 'pit') {
        ctx.fillStyle = '#dc2626';
      } else if (cell.type === 'start') {
        ctx.fillStyle = '#2563eb';
      } else {
        // Heatmap
        const v = stateValue(r, c);
        ctx.fillStyle = valueScale(Math.max(-1, Math.min(1, v)));
      }
      ctx.fillRect(x, y, CELL, CELL);

      // Border
      if (hoveredCell && hoveredCell.row === r && hoveredCell.col === c) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 1, y + 1, CELL - 2, CELL - 2);
      } else if (selectedCell && selectedCell.row === r && selectedCell.col === c) {
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 1, y + 1, CELL - 2, CELL - 2);
      }

      // Policy arrow
      if (cell.type === 'empty' || cell.type === 'start') {
        const a = greedyAction(r, c);
        const [dr, dc] = ACTION_DELTAS[a];
        const cx2 = x + CELL / 2;
        const cy2 = y + CELL / 2;
        const len = CELL * 0.28;
        ctx.strokeStyle = 'rgba(255,255,255,0.75)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx2 - dc * len, cy2 - dr * len);
        ctx.lineTo(cx2 + dc * len, cy2 + dr * len);
        // arrowhead
        const ax = cx2 + dc * len;
        const ay = cy2 + dr * len;
        const hs = 6;
        ctx.lineTo(ax - dc * hs + dr * hs * 0.5, ay - dr * hs + dc * hs * 0.5);
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax - dc * hs - dr * hs * 0.5, ay - dr * hs - dc * hs * 0.5);
        ctx.stroke();
      }

      // Cell type label
      ctx.font = `bold ${CELL * 0.22}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      if (cell.type === 'goal') ctx.fillText('GOAL', x + CELL / 2, y + CELL - 2);
      else if (cell.type === 'pit') ctx.fillText('PIT', x + CELL / 2, y + CELL - 2);
      else if (cell.type === 'start') ctx.fillText('START', x + CELL / 2, y + CELL - 2);

      // Value text
      if (cell.type === 'empty' || cell.type === 'start') {
        const v = stateValue(r, c);
        ctx.font = `${CELL * 0.18}px monospace`;
        ctx.textBaseline = 'top';
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.fillText(v.toFixed(2), x + CELL / 2, y + 2);
      }
    }
  }

  // Agent
  const ax = PAD + agentPos.col * (CELL + PAD) + CELL / 2;
  const ay = PAD + agentPos.row * (CELL + PAD) + CELL / 2;
  const radius = CELL * 0.22;
  ctx.beginPath();
  ctx.arc(ax, ay, radius, 0, Math.PI * 2);
  ctx.fillStyle = '#fde047';
  ctx.fill();
  ctx.strokeStyle = '#78350f';
  ctx.lineWidth = 2;
  ctx.stroke();
  // Agent face
  ctx.fillStyle = '#78350f';
  ctx.beginPath();
  ctx.arc(ax - radius * 0.3, ay - radius * 0.2, radius * 0.12, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(ax + radius * 0.3, ay - radius * 0.2, radius * 0.12, 0, Math.PI * 2);
  ctx.fill();
}

function drawChart() {
  const node = chartSvg.node()!;
  const W = node.clientWidth || 600;
  chartSvg.attr('width', W).attr('height', CHART_H);
  chartSvg.selectAll('*').remove();

  const margin = { top: 10, right: 16, bottom: 28, left: 44 };
  const w = W - margin.left - margin.right;
  const h = CHART_H - margin.top - margin.bottom;

  const g = chartSvg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const data = rewardHistory.slice(-200);
  if (data.length === 0) return;

  const xScale = d3.scaleLinear().domain([0, data.length - 1]).range([0, w]);
  const yExtent = d3.extent(data) as [number, number];
  const yScale = d3.scaleLinear().domain([yExtent[0] - 0.01, yExtent[1] + 0.01]).range([h, 0]);

  // Grid lines
  g.append('g').attr('class', 'grid')
    .call(d3.axisLeft(yScale).ticks(4).tickSize(-w).tickFormat(() => ''))
    .selectAll('line').attr('stroke', '#334155').attr('stroke-dasharray', '2,2');
  g.select('.grid .domain').remove();

  // Line
  const line = d3.line<number>()
    .x((_, i) => xScale(i))
    .y(d => yScale(d))
    .curve(d3.curveBasis);

  g.append('path')
    .datum(data)
    .attr('fill', 'none')
    .attr('stroke', '#a78bfa')
    .attr('stroke-width', 2)
    .attr('d', line);

  // Axes
  g.append('g').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(xScale).ticks(5))
    .selectAll('text').attr('fill', '#94a3b8').attr('font-size', '10px');
  g.append('g')
    .call(d3.axisLeft(yScale).ticks(4).tickFormat(d3.format('.2f')))
    .selectAll('text').attr('fill', '#94a3b8').attr('font-size', '10px');

  g.selectAll('.domain').attr('stroke', '#475569');
  g.selectAll('line').attr('stroke', '#475569');

  // Label
  chartSvg.append('text')
    .attr('x', margin.left - 38).attr('y', margin.top + h / 2)
    .attr('fill', '#94a3b8').attr('font-size', '10px')
    .attr('writing-mode', 'vertical-rl')
    .text('Reward/Ep');
}

function drawQPanel() {
  const cell = selectedCell || hoveredCell;
  if (!cell) {
    qPanel.innerHTML = '<div style="color:#64748b;padding:8px">Hover or click a cell to inspect Q-values</div>';
    return;
  }
  const { row: r, col: c } = cell;
  const qs = Q[r][c];
  const maxQ = Math.max(...qs);
  let html = `<div style="font-size:0.8rem;color:#94a3b8;margin-bottom:6px">Q-values at (${r},${c}) — ${grid[r][c].type}</div>`;
  ACTIONS.forEach((a) => {
    const pct = maxQ !== 0 ? Math.abs(qs[a] / (Math.abs(maxQ) || 1)) * 100 : 0;
    const col = qs[a] >= 0 ? '#4ade80' : '#f87171';
    html += `<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
      <span style="width:18px;text-align:center;font-size:1rem">${ACTION_LABELS[a]}</span>
      <div style="flex:1;background:#1e293b;border-radius:3px;height:12px;overflow:hidden">
        <div style="height:100%;width:${pct.toFixed(1)}%;background:${col};transition:width 0.2s"></div>
      </div>
      <span style="width:52px;text-align:right;font-size:0.75rem;color:${col}">${qs[a].toFixed(4)}</span>
    </div>`;
  });
  qPanel.innerHTML = html;
}

function renderAll() {
  drawGrid();
  drawChart();
  drawQPanel();
  if (statusEl) {
    const eps = currentEpsilon();
    statusEl.textContent = `Episode: ${episode}  |  Step: ${step}  |  ε: ${eps.toFixed(3)}  |  Alg: ${algorithm === 'qlearning' ? 'Q-Learning' : 'SARSA'}`;
  }
}

// ---- Training loop ----
function startTraining() {
  if (running) return;
  running = true;
  function tick() {
    if (!running) return;
    doStep();
    renderAll();
    animFrame = requestAnimationFrame(() => {
      setTimeout(tick, speedMs);
    });
  }
  tick();
}

function stopTraining() {
  running = false;
  if (animFrame !== null) { cancelAnimationFrame(animFrame); animFrame = null; }
}

// ---- HTML setup ----
function setup() {
  initGrid();
  initQ();

  const app = document.getElementById('app')!;
  app.innerHTML = '';

  // Layout: left = grid+chart, right = controls+q-panel
  const layout = document.createElement('div');
  layout.style.cssText = 'display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap';
  app.appendChild(layout);

  // Left column
  const left = document.createElement('div');
  left.style.cssText = 'display:flex;flex-direction:column;gap:12px';
  layout.appendChild(left);

  // Canvas
  const W = COLS * (CELL + PAD) + PAD;
  const H = ROWS * (CELL + PAD) + PAD;
  canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  canvas.style.cssText = 'border-radius:8px;cursor:crosshair;display:block';
  ctx = canvas.getContext('2d')!;
  left.appendChild(canvas);

  // Status
  statusEl = document.createElement('div');
  statusEl.style.cssText = 'font-size:0.8rem;color:#94a3b8;font-family:monospace';
  left.appendChild(statusEl);

  // Chart
  const chartWrap = document.createElement('div');
  chartWrap.style.cssText = 'background:#0f172a;border-radius:8px;padding:8px';
  const chartTitle = document.createElement('div');
  chartTitle.style.cssText = 'font-size:0.8rem;color:#94a3b8;margin-bottom:4px';
  chartTitle.textContent = 'Reward per Episode (last 200)';
  chartWrap.appendChild(chartTitle);
  const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svgEl.setAttribute('width', '640');
  svgEl.setAttribute('height', String(CHART_H));
  svgEl.style.cssText = 'display:block;width:100%';
  chartWrap.appendChild(svgEl);
  left.appendChild(chartWrap);
  chartSvg = d3.select(svgEl as SVGSVGElement);

  // Right column
  const right = document.createElement('div');
  right.style.cssText = 'display:flex;flex-direction:column;gap:16px;min-width:240px;max-width:280px';
  layout.appendChild(right);

  // -- Edit palette --
  const paletteBox = document.createElement('div');
  paletteBox.style.cssText = 'background:#0f172a;border-radius:8px;padding:12px';
  const paletteTitle = document.createElement('div');
  paletteTitle.style.cssText = 'font-size:0.85rem;font-weight:600;color:#a78bfa;margin-bottom:8px';
  paletteTitle.textContent = 'Grid Editor';
  paletteBox.appendChild(paletteTitle);

  const palettes: { type: CellType; label: string; color: string }[] = [
    { type: 'wall', label: '🧱 Wall', color: '#2a2a3e' },
    { type: 'goal', label: '🎯 Goal (+1)', color: '#16a34a' },
    { type: 'pit', label: '💀 Pit (−1)', color: '#dc2626' },
    { type: 'start', label: '🚀 Start', color: '#2563eb' },
    { type: 'empty', label: '⬜ Erase', color: '#334155' },
  ];
  const paletteButtons: HTMLButtonElement[] = [];
  const palGrid = document.createElement('div');
  palGrid.style.cssText = 'display:flex;flex-direction:column;gap:4px';
  palettes.forEach(p => {
    const btn = document.createElement('button');
    btn.textContent = p.label;
    btn.style.cssText = `background:${p.color};color:#fff;border:2px solid transparent;border-radius:6px;padding:5px 8px;cursor:pointer;text-align:left;font-size:0.82rem`;
    btn.addEventListener('click', () => {
      paletteSelected = p.type;
      paletteButtons.forEach((b, i) => {
        b.style.borderColor = palettes[i].type === paletteSelected ? '#facc15' : 'transparent';
      });
    });
    paletteButtons.push(btn);
    palGrid.appendChild(btn);
  });
  paletteButtons[0].style.borderColor = '#facc15';
  paletteBox.appendChild(palGrid);
  right.appendChild(paletteBox);

  // -- Controls --
  const ctrlBox = document.createElement('div');
  ctrlBox.style.cssText = 'background:#0f172a;border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:10px';
  const ctrlTitle = document.createElement('div');
  ctrlTitle.style.cssText = 'font-size:0.85rem;font-weight:600;color:#a78bfa';
  ctrlTitle.textContent = 'Training Controls';
  ctrlBox.appendChild(ctrlTitle);

  // Playback buttons
  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap';

  function mkBtn(label: string, onClick: () => void) {
    const b = document.createElement('button');
    b.textContent = label;
    b.style.cssText = 'background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:5px 10px;cursor:pointer;font-size:0.82rem;flex:1';
    b.addEventListener('click', onClick);
    return b;
  }

  const playBtn = mkBtn('▶ Play', () => {
    if (running) {
      stopTraining();
      playBtn.textContent = '▶ Play';
    } else {
      startTraining();
      playBtn.textContent = '⏸ Pause';
    }
  });
  btnRow.appendChild(playBtn);

  btnRow.appendChild(mkBtn('Step', () => {
    stopTraining();
    playBtn.textContent = '▶ Play';
    doStep();
    renderAll();
  }));

  btnRow.appendChild(mkBtn('Episode', () => {
    stopTraining();
    playBtn.textContent = '▶ Play';
    doEpisode();
  }));

  btnRow.appendChild(mkBtn('Reset Q', () => {
    stopTraining();
    playBtn.textContent = '▶ Play';
    initQ();
  }));

  ctrlBox.appendChild(btnRow);

  // Algorithm selector
  function mkRow(label: string, el: HTMLElement) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:8px';
    const lbl = document.createElement('label');
    lbl.textContent = label;
    lbl.style.cssText = 'font-size:0.8rem;color:#94a3b8;flex-shrink:0';
    row.appendChild(lbl);
    row.appendChild(el);
    return row;
  }

  const algSel = document.createElement('select');
  algSel.style.cssText = 'background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:3px 6px;font-size:0.8rem;width:120px';
  [['qlearning','Q-Learning'],['sarsa','SARSA']].forEach(([val, lbl]) => {
    const opt = document.createElement('option');
    opt.value = val; opt.textContent = lbl;
    algSel.appendChild(opt);
  });
  algSel.addEventListener('change', () => { algorithm = algSel.value as 'qlearning' | 'sarsa'; });
  ctrlBox.appendChild(mkRow('Algorithm', algSel));

  function mkSlider(min: number, max: number, step2: number, val: number, fmt: (v: number) => string, onChange: (v: number) => void) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;align-items:center;gap:6px;width:160px';
    const input = document.createElement('input');
    input.type = 'range'; input.min = String(min); input.max = String(max);
    input.step = String(step2); input.value = String(val);
    input.style.cssText = 'flex:1;accent-color:#a78bfa';
    const valSpan = document.createElement('span');
    valSpan.style.cssText = 'font-size:0.75rem;color:#e2e8f0;width:42px;text-align:right;font-family:monospace';
    valSpan.textContent = fmt(val);
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      onChange(v);
      valSpan.textContent = fmt(v);
    });
    wrap.appendChild(input);
    wrap.appendChild(valSpan);
    return wrap;
  }

  ctrlBox.appendChild(mkRow('α (lr)', mkSlider(0.01, 1, 0.01, alpha, v => v.toFixed(2), v => { alpha = v; })));
  ctrlBox.appendChild(mkRow('γ (discount)', mkSlider(0.5, 0.999, 0.001, gamma, v => v.toFixed(3), v => { gamma = v; })));
  ctrlBox.appendChild(mkRow('ε (explore)', mkSlider(0, 1, 0.01, epsilon, v => v.toFixed(2), v => { epsilon = v; })));
  ctrlBox.appendChild(mkRow('Speed (ms)', mkSlider(0, 300, 10, speedMs, v => String(v) + 'ms', v => { speedMs = v; })));

  // Epsilon decay toggle
  const decayRow = document.createElement('div');
  decayRow.style.cssText = 'display:flex;align-items:center;justify-content:space-between';
  const decayLbl = document.createElement('label');
  decayLbl.textContent = 'ε Decay';
  decayLbl.style.cssText = 'font-size:0.8rem;color:#94a3b8';
  const decayChk = document.createElement('input');
  decayChk.type = 'checkbox';
  decayChk.style.cssText = 'accent-color:#a78bfa';
  decayChk.addEventListener('change', () => { epsilonDecay = decayChk.checked; });
  decayRow.appendChild(decayLbl);
  decayRow.appendChild(decayChk);
  ctrlBox.appendChild(decayRow);

  right.appendChild(ctrlBox);

  // -- Q-values panel --
  const qBox = document.createElement('div');
  qBox.style.cssText = 'background:#0f172a;border-radius:8px;padding:12px';
  const qTitle = document.createElement('div');
  qTitle.style.cssText = 'font-size:0.85rem;font-weight:600;color:#a78bfa;margin-bottom:8px';
  qTitle.textContent = 'Q-Value Inspector';
  qBox.appendChild(qTitle);
  qPanel = document.createElement('div');
  qBox.appendChild(qPanel);
  right.appendChild(qBox);

  // Legend
  const legendBox = document.createElement('div');
  legendBox.style.cssText = 'background:#0f172a;border-radius:8px;padding:12px';
  const legendTitle = document.createElement('div');
  legendTitle.style.cssText = 'font-size:0.82rem;font-weight:600;color:#a78bfa;margin-bottom:6px';
  legendTitle.textContent = 'Value Heatmap';
  legendBox.appendChild(legendTitle);
  // Gradient bar
  const gradWrap = document.createElement('div');
  gradWrap.style.cssText = 'display:flex;align-items:center;gap:6px';
  const grad = document.createElement('div');
  grad.style.cssText = 'flex:1;height:12px;border-radius:4px;background:linear-gradient(to right,#d73027,#ffffbf,#1a9850)';
  const lLbl = document.createElement('span');
  lLbl.style.cssText = 'font-size:0.72rem;color:#94a3b8'; lLbl.textContent = '-1';
  const rLbl = document.createElement('span');
  rLbl.style.cssText = 'font-size:0.72rem;color:#94a3b8'; rLbl.textContent = '+1';
  gradWrap.appendChild(lLbl); gradWrap.appendChild(grad); gradWrap.appendChild(rLbl);
  legendBox.appendChild(gradWrap);
  right.appendChild(legendBox);

  // ---- Canvas interactions ----
  function cellFromEvent(e: MouseEvent): AgentPos | null {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;
    const c = Math.floor((mx - PAD) / (CELL + PAD));
    const r = Math.floor((my - PAD) / (CELL + PAD));
    if (r >= 0 && r < ROWS && c >= 0 && c < COLS) return { row: r, col: c };
    return null;
  }

  canvas.addEventListener('mousemove', (e) => {
    hoveredCell = cellFromEvent(e);
    drawQPanel();
    drawGrid();
  });

  canvas.addEventListener('mouseleave', () => {
    hoveredCell = null;
    drawGrid();
    drawQPanel();
  });

  canvas.addEventListener('click', (e) => {
    const cell = cellFromEvent(e);
    if (!cell) return;
    const { row: r, col: c } = cell;

    // Update selection
    if (selectedCell && selectedCell.row === r && selectedCell.col === c) {
      selectedCell = null;
    } else {
      selectedCell = { row: r, col: c };
    }

    // Edit cell
    if (paletteSelected === 'start') {
      // Remove existing start
      for (let rr = 0; rr < ROWS; rr++) {
        for (let cc = 0; cc < COLS; cc++) {
          if (grid[rr][cc].type === 'start') grid[rr][cc].type = 'empty';
        }
      }
      grid[r][c].type = 'start';
      startPos = { row: r, col: c };
      agentPos = { ...startPos };
    } else if (paletteSelected === 'goal') {
      // Remove existing goal
      for (let rr = 0; rr < ROWS; rr++) {
        for (let cc = 0; cc < COLS; cc++) {
          if (grid[rr][cc].type === 'goal') grid[rr][cc].type = 'empty';
        }
      }
      grid[r][c].type = 'goal';
    } else {
      grid[r][c].type = paletteSelected;
    }

    renderAll();
  });

  renderAll();
}

// Boot
document.addEventListener('DOMContentLoaded', setup);
