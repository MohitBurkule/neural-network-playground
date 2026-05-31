export {};

// ── Types ────────────────────────────────────────────────────────────────────

type CellType = "empty" | "wall" | "weight";
type DrawMode = "wall" | "weight" | "erase" | "start" | "goal";
type Algorithm = "astar" | "dijkstra" | "greedy" | "bfs";
type Heuristic = "manhattan" | "euclidean" | "chebyshev";

interface Cell {
  row: number;
  col: number;
  type: CellType;
  weight: number; // 1 = normal, 5 = slow terrain
}

interface NodeState {
  row: number;
  col: number;
  g: number;
  h: number;
  f: number;
  parent: NodeState | null;
}

// ── State ────────────────────────────────────────────────────────────────────

let ROWS = 25;
let COLS = 40;
let CELL_SIZE = 0;

let grid: Cell[][] = [];
let startRow = 2, startCol = 2;
let goalRow = ROWS - 3, goalCol = COLS - 3;

let algorithm: Algorithm = "astar";
let heuristic: Heuristic = "manhattan";
let diagonals = false;
let speedMs = 20;

let visitedSet: Set<string> = new Set();
let frontierSet: Set<string> = new Set();
let pathCells: Array<[number, number]> = [];

let nodesExpanded = 0;
let pathLength = 0;
let elapsedMs = 0;

type PlayState = "idle" | "running" | "paused" | "done";
let playState: PlayState = "idle";

// Generator-based step iterator
let stepGen: Generator<StepResult, StepResult, unknown> | null = null;
let animFrame = 0;
let lastStepTime = 0;

interface StepResult {
  done: boolean;
  visited: Array<[number, number]>;
  frontier: Array<[number, number]>;
  path: Array<[number, number]>;
  nodesExpanded: number;
}

// Mouse state
let mouseDown = false;
let drawMode: DrawMode = "wall";

// ── Canvas ───────────────────────────────────────────────────────────────────

const canvas = document.getElementById("pathfind-canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;

function resizeCanvas(): void {
  const wrap = document.getElementById("canvas-wrap")!;
  const W = wrap.clientWidth;
  const H = wrap.clientHeight;
  // compute cell size to fit grid
  const cs = Math.floor(Math.min(W / COLS, H / ROWS));
  CELL_SIZE = Math.max(cs, 4);
  canvas.width = COLS * CELL_SIZE;
  canvas.height = ROWS * CELL_SIZE;
  draw();
}

// ── Grid init ────────────────────────────────────────────────────────────────

function initGrid(): void {
  grid = [];
  for (let r = 0; r < ROWS; r++) {
    grid[r] = [];
    for (let c = 0; c < COLS; c++) {
      grid[r][c] = { row: r, col: c, type: "empty", weight: 1 };
    }
  }
}

function resetSearch(): void {
  visitedSet = new Set();
  frontierSet = new Set();
  pathCells = [];
  nodesExpanded = 0;
  pathLength = 0;
  elapsedMs = 0;
  playState = "idle";
  stepGen = null;
  cancelAnimationFrame(animFrame);
  updateStats();
  draw();
}

// ── Colors ───────────────────────────────────────────────────────────────────

const COLOR = {
  bg: "#0d1117",
  empty: "#161b22",
  wall: "#58a6ff",
  weight: "#e3b341",
  start: "#3fb950",
  goal: "#f85149",
  visited: "#1c3a5c",
  frontier: "#388bfd",
  path: "#ffa657",
  grid: "#21262d",
};

// ── Draw ─────────────────────────────────────────────────────────────────────

function draw(): void {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cs = CELL_SIZE;

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = grid[r][c];
      const key = `${r},${c}`;
      let fill = COLOR.empty;

      if (cell.type === "wall") fill = COLOR.wall;
      else if (cell.type === "weight") fill = COLOR.weight;
      else if (visitedSet.has(key)) fill = COLOR.visited;
      else if (frontierSet.has(key)) fill = COLOR.frontier;

      ctx.fillStyle = fill;
      ctx.fillRect(c * cs, r * cs, cs, cs);

      // grid lines
      ctx.strokeStyle = COLOR.grid;
      ctx.lineWidth = 0.5;
      ctx.strokeRect(c * cs, r * cs, cs, cs);
    }
  }

  // draw path
  if (pathCells.length > 0) {
    ctx.fillStyle = COLOR.path;
    pathCells.forEach(([r, c]) => {
      ctx.fillRect(c * cs + 1, r * cs + 1, cs - 2, cs - 2);
    });
  }

  // draw start
  drawSpecial(startRow, startCol, COLOR.start, "S");
  // draw goal
  drawSpecial(goalRow, goalCol, COLOR.goal, "G");
}

function drawSpecial(r: number, c: number, color: string, label: string): void {
  const cs = CELL_SIZE;
  ctx.fillStyle = color;
  ctx.fillRect(c * cs + 1, r * cs + 1, cs - 2, cs - 2);
  if (cs >= 12) {
    ctx.fillStyle = "#0d1117";
    ctx.font = `bold ${Math.floor(cs * 0.55)}px monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, c * cs + cs / 2, r * cs + cs / 2);
  }
}

// ── Heuristics ───────────────────────────────────────────────────────────────

function heuristicFn(r1: number, c1: number, r2: number, c2: number): number {
  const dr = Math.abs(r1 - r2);
  const dc = Math.abs(c1 - c2);
  if (heuristic === "manhattan") return dr + dc;
  if (heuristic === "euclidean") return Math.sqrt(dr * dr + dc * dc);
  // chebyshev
  return Math.max(dr, dc);
}

// ── Neighbors ────────────────────────────────────────────────────────────────

function neighbors(r: number, c: number): Array<[number, number, number]> {
  const dirs4: Array<[number, number]> = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  const dirs8: Array<[number, number]> = [
    [-1, 0], [1, 0], [0, -1], [0, 1],
    [-1, -1], [-1, 1], [1, -1], [1, 1],
  ];
  const dirs = diagonals ? dirs8 : dirs4;
  const result: Array<[number, number, number]> = [];
  dirs.forEach(([dr, dc]) => {
    const nr = r + dr;
    const nc = c + dc;
    if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
      const cell = grid[nr][nc];
      if (cell.type !== "wall") {
        // diagonal cost = sqrt(2) * weight
        const isDiag = dr !== 0 && dc !== 0;
        const moveCost = (isDiag ? 1.414 : 1) * cell.weight;
        result.push([nr, nc, moveCost]);
      }
    }
  });
  return result;
}

// ── Priority Queue (min-heap) ────────────────────────────────────────────────

class MinHeap<T> {
  private data: Array<[number, T]> = [];

  push(priority: number, value: T): void {
    this.data.push([priority, value]);
    this._bubbleUp(this.data.length - 1);
  }

  pop(): T | undefined {
    if (this.data.length === 0) return undefined;
    const top = this.data[0][1];
    const last = this.data.pop()!;
    if (this.data.length > 0) {
      this.data[0] = last;
      this._sinkDown(0);
    }
    return top;
  }

  get size(): number { return this.data.length; }

  private _bubbleUp(i: number): void {
    while (i > 0) {
      const parent = Math.floor((i - 1) / 2);
      if (this.data[parent][0] <= this.data[i][0]) break;
      [this.data[parent], this.data[i]] = [this.data[i], this.data[parent]];
      i = parent;
    }
  }

  private _sinkDown(i: number): void {
    const n = this.data.length;
    while (true) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.data[l][0] < this.data[smallest][0]) smallest = l;
      if (r < n && this.data[r][0] < this.data[smallest][0]) smallest = r;
      if (smallest === i) break;
      [this.data[smallest], this.data[i]] = [this.data[i], this.data[smallest]];
      i = smallest;
    }
  }
}

// ── Algorithm generators ─────────────────────────────────────────────────────

function* astarGen(isDijkstra: boolean, isGreedy: boolean): Generator<StepResult, StepResult, unknown> {
  const visited: Array<[number, number]> = [];
  const frontier: Array<[number, number]> = [];

  const open = new MinHeap<NodeState>();
  const gScore = new Map<string, number>();
  const closed = new Set<string>();
  const frontierKeys = new Set<string>();

  const startNode: NodeState = {
    row: startRow, col: startCol,
    g: 0, h: heuristicFn(startRow, startCol, goalRow, goalCol),
    f: 0, parent: null,
  };
  startNode.f = isGreedy ? startNode.h : (isDijkstra ? startNode.g : startNode.g + startNode.h);

  open.push(startNode.f, startNode);
  gScore.set(`${startRow},${startCol}`, 0);
  frontierKeys.add(`${startRow},${startCol}`);

  let expanded = 0;

  while (open.size > 0) {
    const current = open.pop()!;
    const key = `${current.row},${current.col}`;

    if (closed.has(key)) continue;
    closed.add(key);
    frontierKeys.delete(key);
    expanded++;

    visited.push([current.row, current.col]);

    if (current.row === goalRow && current.col === goalCol) {
      // reconstruct path
      const path: Array<[number, number]> = [];
      let node: NodeState | null = current;
      while (node !== null) {
        path.push([node.row, node.col]);
        node = node.parent;
      }
      path.reverse();
      return {
        done: true,
        visited: Array.from(visited),
        frontier: Array.from(frontierKeys).map(k => {
          const [r, c] = k.split(",").map(Number);
          return [r, c] as [number, number];
        }),
        path,
        nodesExpanded: expanded,
      };
    }

    const frontierSnap = Array.from(frontierKeys).map(k => {
      const [r, c] = k.split(",").map(Number);
      return [r, c] as [number, number];
    });

    yield {
      done: false,
      visited: Array.from(visited),
      frontier: frontierSnap,
      path: [],
      nodesExpanded: expanded,
    };

    const nbrs = neighbors(current.row, current.col);
    for (let i = 0; i < nbrs.length; i++) {
      const [nr, nc, cost] = nbrs[i];
      const nkey = `${nr},${nc}`;
      if (closed.has(nkey)) continue;

      const tentativeG = isDijkstra || !isGreedy ? current.g + cost : 0;
      const prevG = gScore.get(nkey) ?? Infinity;
      if (tentativeG < prevG) {
        gScore.set(nkey, tentativeG);
        const h = heuristicFn(nr, nc, goalRow, goalCol);
        const f = isGreedy ? h : (isDijkstra ? tentativeG : tentativeG + h);
        const node: NodeState = { row: nr, col: nc, g: tentativeG, h, f, parent: current };
        open.push(f, node);
        frontierKeys.add(nkey);
      }
    }
  }

  return { done: true, visited: Array.from(visited), frontier: [], path: [], nodesExpanded: expanded };
}

function* bfsGen(): Generator<StepResult, StepResult, unknown> {
  const visitedArr: Array<[number, number]> = [];
  const queue: Array<NodeState> = [];
  const visited = new Set<string>();

  const startNode: NodeState = { row: startRow, col: startCol, g: 0, h: 0, f: 0, parent: null };
  queue.push(startNode);
  visited.add(`${startRow},${startCol}`);

  let expanded = 0;

  while (queue.length > 0) {
    const current = queue.shift()!;
    expanded++;
    visitedArr.push([current.row, current.col]);

    if (current.row === goalRow && current.col === goalCol) {
      const path: Array<[number, number]> = [];
      let node: NodeState | null = current;
      while (node !== null) {
        path.push([node.row, node.col]);
        node = node.parent;
      }
      path.reverse();
      return { done: true, visited: Array.from(visitedArr), frontier: [], path, nodesExpanded: expanded };
    }

    const frontierSnap = queue.map(n => [n.row, n.col] as [number, number]);

    yield {
      done: false,
      visited: Array.from(visitedArr),
      frontier: frontierSnap,
      path: [],
      nodesExpanded: expanded,
    };

    const nbrs = neighbors(current.row, current.col);
    for (let i = 0; i < nbrs.length; i++) {
      const [nr, nc] = nbrs[i];
      const nkey = `${nr},${nc}`;
      if (!visited.has(nkey)) {
        visited.add(nkey);
        queue.push({ row: nr, col: nc, g: current.g + 1, h: 0, f: 0, parent: current });
      }
    }
  }

  return { done: true, visited: Array.from(visitedArr), frontier: [], path: [], nodesExpanded: expanded };
}

function makeGenerator(): Generator<StepResult, StepResult, unknown> {
  if (algorithm === "bfs") return bfsGen();
  if (algorithm === "dijkstra") return astarGen(true, false);
  if (algorithm === "greedy") return astarGen(false, true);
  return astarGen(false, false); // astar
}

// ── Animation loop ───────────────────────────────────────────────────────────

function applyStepResult(result: StepResult): void {
  visitedSet = new Set(result.visited.map(([r, c]) => `${r},${c}`));
  frontierSet = new Set(result.frontier.map(([r, c]) => `${r},${c}`));
  pathCells = result.path;
  nodesExpanded = result.nodesExpanded;

  // compute path length (sum of weights)
  if (pathCells.length > 1) {
    let len = 0;
    for (let i = 1; i < pathCells.length; i++) {
      const [r, c] = pathCells[i];
      const [pr, pc] = pathCells[i - 1];
      const dr = Math.abs(r - pr), dc = Math.abs(c - pc);
      const isDiag = dr + dc > 1;
      len += (isDiag ? 1.414 : 1) * grid[r][c].weight;
    }
    pathLength = Math.round(len * 10) / 10;
  } else {
    pathLength = 0;
  }

  updateStats();
  draw();
}

function animLoop(ts: number): void {
  if (playState !== "running") return;
  if (!stepGen) return;

  if (ts - lastStepTime < speedMs) {
    animFrame = requestAnimationFrame(animLoop);
    return;
  }
  lastStepTime = ts;
  elapsedMs += speedMs;

  const res = stepGen.next();
  const value = res.value as StepResult;
  applyStepResult(value);

  if (value.done || res.done) {
    playState = "done";
    updateButtons();
    return;
  }

  animFrame = requestAnimationFrame(animLoop);
}

// ── Controls ─────────────────────────────────────────────────────────────────

function updateStats(): void {
  const el = document.getElementById("stats")!;
  el.innerHTML = `
    <div class="stat"><span>Nodes expanded</span><strong>${nodesExpanded}</strong></div>
    <div class="stat"><span>Path length</span><strong>${pathLength || "—"}</strong></div>
    <div class="stat"><span>Status</span><strong>${playState}</strong></div>
  `;
}

function updateButtons(): void {
  const play = document.getElementById("btn-play") as HTMLButtonElement;
  const pause = document.getElementById("btn-pause") as HTMLButtonElement;
  const step = document.getElementById("btn-step") as HTMLButtonElement;

  play.disabled = playState === "running" || playState === "done";
  pause.disabled = playState !== "running";
  step.disabled = playState === "running" || playState === "done";
}

function onPlay(): void {
  if (playState === "idle") {
    stepGen = makeGenerator();
    playState = "running";
    lastStepTime = 0;
  } else if (playState === "paused") {
    playState = "running";
  }
  updateButtons();
  animFrame = requestAnimationFrame(animLoop);
}

function onPause(): void {
  playState = "paused";
  cancelAnimationFrame(animFrame);
  updateButtons();
}

function onStep(): void {
  if (playState === "idle") {
    stepGen = makeGenerator();
    playState = "paused";
  }
  if (!stepGen || playState === "done") return;

  const res = stepGen.next();
  const value = res.value as StepResult;
  applyStepResult(value);

  if (value.done || res.done) {
    playState = "done";
  }
  updateButtons();
}

function onReset(): void {
  resetSearch();
  updateButtons();
}

// ── Maze generation (recursive backtracker) ──────────────────────────────────

function generateMaze(): void {
  resetSearch();
  // Fill all with walls
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      grid[r][c].type = "wall";
      grid[r][c].weight = 1;
    }
  }

  // Carve passages from odd cells
  const visited = new Set<string>();

  function carve(r: number, c: number): void {
    visited.add(`${r},${c}`);
    grid[r][c].type = "empty";

    const dirs: Array<[number, number]> = [[-2, 0], [2, 0], [0, -2], [0, 2]];
    // shuffle
    for (let i = dirs.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [dirs[i], dirs[j]] = [dirs[j], dirs[i]];
    }

    dirs.forEach(([dr, dc]) => {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && !visited.has(`${nr},${nc}`)) {
        // carve wall between
        grid[r + dr / 2][c + dc / 2].type = "empty";
        carve(nr, nc);
      }
    });
  }

  // start from odd cell
  const sr = 1, sc = 1;
  carve(sr, sc);

  // ensure start and goal are clear
  grid[startRow][startCol].type = "empty";
  grid[goalRow][goalCol].type = "empty";

  draw();
}

// ── Mouse interactions ───────────────────────────────────────────────────────

function cellFromEvent(e: MouseEvent): [number, number] | null {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const c = Math.floor(x / CELL_SIZE);
  const r = Math.floor(y / CELL_SIZE);
  if (r >= 0 && r < ROWS && c >= 0 && c < COLS) return [r, c];
  return null;
}

function paintCell(r: number, c: number): void {
  if (r === startRow && c === startCol) return;
  if (r === goalRow && c === goalCol) return;

  if (drawMode === "wall") {
    grid[r][c].type = "wall";
    grid[r][c].weight = 1;
  } else if (drawMode === "weight") {
    grid[r][c].type = "weight";
    grid[r][c].weight = 5;
  } else if (drawMode === "erase") {
    grid[r][c].type = "empty";
    grid[r][c].weight = 1;
  }
  draw();
}

canvas.addEventListener("mousedown", (e) => {
  const pos = cellFromEvent(e);
  if (!pos) return;
  const [r, c] = pos;

  if (drawMode === "start") {
    startRow = r; startCol = c;
    resetSearch();
    draw();
    return;
  }
  if (drawMode === "goal") {
    goalRow = r; goalCol = c;
    resetSearch();
    draw();
    return;
  }
  mouseDown = true;
  paintCell(r, c);
});

canvas.addEventListener("mousemove", (e) => {
  if (!mouseDown) return;
  const pos = cellFromEvent(e);
  if (!pos) return;
  paintCell(pos[0], pos[1]);
});

canvas.addEventListener("mouseup", () => { mouseDown = false; });
canvas.addEventListener("mouseleave", () => { mouseDown = false; });
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

// ── Wire up controls ──────────────────────────────────────────────────────────

function wireControls(): void {
  const algoSel = document.getElementById("algo-select") as HTMLSelectElement;
  algoSel.addEventListener("change", () => {
    algorithm = algoSel.value as Algorithm;
    resetSearch();
  });

  const heuristicSel = document.getElementById("heuristic-select") as HTMLSelectElement;
  heuristicSel.addEventListener("change", () => {
    heuristic = heuristicSel.value as Heuristic;
    resetSearch();
  });

  const diagCheck = document.getElementById("diag-check") as HTMLInputElement;
  diagCheck.addEventListener("change", () => {
    diagonals = diagCheck.checked;
    resetSearch();
  });

  const speedSlider = document.getElementById("speed-slider") as HTMLInputElement;
  const speedLabel = document.getElementById("speed-label")!;
  speedSlider.addEventListener("input", () => {
    const v = parseInt(speedSlider.value);
    speedMs = v;
    speedLabel.textContent = v + "ms";
  });

  const sizeSlider = document.getElementById("size-slider") as HTMLInputElement;
  const sizeLabel = document.getElementById("size-label")!;
  sizeSlider.addEventListener("input", () => {
    const s = parseInt(sizeSlider.value);
    ROWS = s; COLS = Math.round(s * 1.6);
    sizeLabel.textContent = `${ROWS}×${COLS}`;
    startRow = 2; startCol = 2;
    goalRow = ROWS - 3; goalCol = COLS - 3;
    initGrid();
    resetSearch();
    resizeCanvas();
  });

  document.getElementById("btn-play")!.addEventListener("click", onPlay);
  document.getElementById("btn-pause")!.addEventListener("click", onPause);
  document.getElementById("btn-step")!.addEventListener("click", onStep);
  document.getElementById("btn-reset")!.addEventListener("click", onReset);
  document.getElementById("btn-maze")!.addEventListener("click", generateMaze);
  document.getElementById("btn-clear")!.addEventListener("click", () => {
    initGrid();
    resetSearch();
    resizeCanvas();
  });

  const drawBtns = document.querySelectorAll<HTMLButtonElement>(".draw-btn");
  drawBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      drawMode = btn.dataset["mode"] as DrawMode;
      drawBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });
}

// ── Init ─────────────────────────────────────────────────────────────────────

function init(): void {
  initGrid();
  wireControls();
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
  updateStats();
  updateButtons();
}

window.addEventListener("DOMContentLoaded", init);
