// Cellular Automata Lab
// Supports: Conway's Game of Life (2D toroidal) and Elementary CA (Wolfram 1D)

// ─── Utility ────────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function randomInt(n: number): number {
  return Math.floor(Math.random() * n);
}

// ─── Life patterns ──────────────────────────────────────────────────────────

interface Pattern {
  name: string;
  cells: [number, number][];
}

const PATTERNS: Pattern[] = [
  {
    name: "Glider",
    cells: [[0,1],[1,2],[2,0],[2,1],[2,2]]
  },
  {
    name: "Pulsar",
    cells: [
      [0,2],[0,3],[0,4],[0,8],[0,9],[0,10],
      [2,0],[2,5],[2,7],[2,12],
      [3,0],[3,5],[3,7],[3,12],
      [4,0],[4,5],[4,7],[4,12],
      [5,2],[5,3],[5,4],[5,8],[5,9],[5,10],
      [7,2],[7,3],[7,4],[7,8],[7,9],[7,10],
      [8,0],[8,5],[8,7],[8,12],
      [9,0],[9,5],[9,7],[9,12],
      [10,0],[10,5],[10,7],[10,12],
      [12,2],[12,3],[12,4],[12,8],[12,9],[12,10]
    ]
  },
  {
    name: "Gosper Gun",
    cells: [
      [0,24],[1,22],[1,24],[2,12],[2,13],[2,20],[2,21],[2,34],[2,35],
      [3,11],[3,15],[3,20],[3,21],[3,34],[3,35],[4,0],[4,1],[4,10],
      [4,16],[4,20],[4,21],[5,0],[5,1],[5,10],[5,14],[5,16],[5,17],
      [5,22],[5,24],[6,10],[6,16],[6,24],[7,11],[7,15],[8,12],[8,13]
    ]
  },
  {
    name: "LWSS",
    cells: [
      [0,1],[0,2],[0,3],[0,4],[1,0],[1,4],[2,4],[3,0],[3,3]
    ]
  }
];

// ─── Game of Life ────────────────────────────────────────────────────────────

class GameOfLife {
  rows: number;
  cols: number;
  grid: Uint8Array;
  next: Uint8Array;

  constructor(rows: number, cols: number) {
    this.rows = rows;
    this.cols = cols;
    this.grid = new Uint8Array(rows * cols);
    this.next = new Uint8Array(rows * cols);
  }

  idx(r: number, c: number): number {
    return r * this.cols + c;
  }

  get(r: number, c: number): number {
    r = ((r % this.rows) + this.rows) % this.rows;
    c = ((c % this.cols) + this.cols) % this.cols;
    return this.grid[this.idx(r, c)];
  }

  set(r: number, c: number, v: number): void {
    r = ((r % this.rows) + this.rows) % this.rows;
    c = ((c % this.cols) + this.cols) % this.cols;
    this.grid[this.idx(r, c)] = v;
  }

  clear(): void {
    this.grid.fill(0);
  }

  randomize(density: number = 0.3): void {
    for (let i = 0; i < this.grid.length; i++) {
      this.grid[i] = Math.random() < density ? 1 : 0;
    }
  }

  placePattern(pattern: Pattern, row: number, col: number): void {
    pattern.cells.forEach(function(cell) {
      const r = cell[0];
      const c = cell[1];
      const tr = ((row + r) % this.rows + this.rows) % this.rows;
      const tc = ((col + c) % this.cols + this.cols) % this.cols;
      this.grid[this.idx(tr, tc)] = 1;
    }, this);
  }

  population(): number {
    let count = 0;
    for (let i = 0; i < this.grid.length; i++) {
      count += this.grid[i];
    }
    return count;
  }

  step(): void {
    const rows = this.rows;
    const cols = this.cols;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const n = this.get(r-1,c-1)+this.get(r-1,c)+this.get(r-1,c+1)
                + this.get(r,  c-1)                +this.get(r,  c+1)
                + this.get(r+1,c-1)+this.get(r+1,c)+this.get(r+1,c+1);
        const alive = this.grid[this.idx(r, c)];
        this.next[this.idx(r, c)] = (alive && (n === 2 || n === 3)) || (!alive && n === 3) ? 1 : 0;
      }
    }
    const tmp = this.grid;
    this.grid = this.next;
    this.next = tmp;
  }
}

// ─── Elementary CA ───────────────────────────────────────────────────────────

class ElementaryCA {
  rule: number;
  width: number;
  rows: Uint8Array[];
  maxRows: number;

  constructor(width: number, maxRows: number) {
    this.rule = 30;
    this.width = width;
    this.maxRows = maxRows;
    this.rows = [];
  }

  reset(singleCell: boolean): void {
    this.rows = [];
    const first = new Uint8Array(this.width);
    if (singleCell) {
      first[Math.floor(this.width / 2)] = 1;
    } else {
      for (let i = 0; i < this.width; i++) {
        first[i] = Math.random() < 0.5 ? 1 : 0;
      }
    }
    this.rows.push(first);
  }

  computeNext(): void {
    if (this.rows.length >= this.maxRows) return;
    const prev = this.rows[this.rows.length - 1];
    const next = new Uint8Array(this.width);
    const w = this.width;
    const rule = this.rule;
    for (let i = 0; i < w; i++) {
      const l = prev[(i - 1 + w) % w];
      const m = prev[i];
      const r = prev[(i + 1) % w];
      const idx = (l << 2) | (m << 1) | r;
      next[i] = (rule >> idx) & 1;
    }
    this.rows.push(next);
  }

  fillAll(): void {
    while (this.rows.length < this.maxRows) {
      this.computeNext();
    }
  }
}

// ─── Renderer ────────────────────────────────────────────────────────────────

type Mode = "life" | "elementary";

interface CAAppState {
  mode: Mode;
  running: boolean;
  speed: number; // ms per frame
  generation: number;
  // Life
  life: GameOfLife;
  cellSize: number;
  selectedPattern: number;
  drawMode: boolean;
  drawValue: number;
  // Elementary
  eca: ElementaryCA;
  ecaSingleCell: boolean;
  ecaRule: number;
  ecaRunning: boolean;
}

const CANVAS_W = 800;
const CANVAS_H = 600;
const LIFE_COLS = 80;
const LIFE_ROWS = 60;
const ECA_WIDTH = 200;
const ECA_MAX_ROWS = 150;

// Colors
const COLOR_BG = "#0d1117";
const COLOR_CELL_LIFE = "#58a6ff";
const COLOR_CELL_DEAD = "#161b22";
const COLOR_GRID = "#21262d";
const COLOR_ECA_ON = "#e6c84a";
const COLOR_ECA_OFF = "#161b22";

let caState: CAAppState;
let caCanvas: HTMLCanvasElement;
let caCtx: CanvasRenderingContext2D;
let caAnimId: number | null = null;
let caLastTime = 0;

function initCAState(): void {
  const life = new GameOfLife(LIFE_ROWS, LIFE_COLS);
  life.randomize();
  const eca = new ElementaryCA(ECA_WIDTH, ECA_MAX_ROWS);
  eca.rule = 30;
  eca.reset(true);
  eca.fillAll();
  caState = {
    mode: "life",
    running: false,
    speed: 100,
    generation: 0,
    life,
    cellSize: Math.floor(Math.min(CANVAS_W / LIFE_COLS, CANVAS_H / LIFE_ROWS)),
    selectedPattern: 0,
    drawMode: false,
    drawValue: 1,
    eca,
    ecaSingleCell: true,
    ecaRule: 30,
    ecaRunning: false
  };
}

// ─── Drawing ─────────────────────────────────────────────────────────────────

function caDrawLife(): void {
  const s = caState.cellSize;
  const cols = caState.life.cols;
  const rows = caState.life.rows;
  const offX = Math.floor((CANVAS_W - cols * s) / 2);
  const offY = Math.floor((CANVAS_H - rows * s) / 2);

  // Background
  caCtx.fillStyle = COLOR_BG;
  caCtx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  // Cells
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const alive = caState.life.grid[caState.life.idx(r, c)];
      caCtx.fillStyle = alive ? COLOR_CELL_LIFE : COLOR_CELL_DEAD;
      caCtx.fillRect(offX + c * s, offY + r * s, s - 1, s - 1);
    }
  }

  // Population
  caCtx.fillStyle = "#8b949e";
  caCtx.font = "12px monospace";
  caCtx.fillText("Gen " + caState.generation + "  Pop " + caState.life.population(), 8, 18);
}

function caDrawECA(): void {
  caCtx.fillStyle = COLOR_BG;
  caCtx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  const eca = caState.eca;
  const numRows = eca.rows.length;
  const numCols = eca.width;
  const cellW = Math.floor(CANVAS_W / numCols);
  const cellH = Math.floor(CANVAS_H / ECA_MAX_ROWS);
  const offX = Math.floor((CANVAS_W - numCols * cellW) / 2);

  for (let r = 0; r < numRows; r++) {
    const row = eca.rows[r];
    for (let c = 0; c < numCols; c++) {
      caCtx.fillStyle = row[c] ? COLOR_ECA_ON : COLOR_ECA_OFF;
      caCtx.fillRect(offX + c * cellW, r * cellH, cellW, cellH);
    }
  }

  caCtx.fillStyle = "#8b949e";
  caCtx.font = "12px monospace";
  caCtx.fillText("Rule " + eca.rule + "  Gen " + numRows, 8, 18);
}

function caDraw(): void {
  if (caState.mode === "life") {
    caDrawLife();
  } else {
    caDrawECA();
  }
}

// ─── Animation loop ───────────────────────────────────────────────────────────

function caLoop(ts: number): void {
  caAnimId = requestAnimationFrame(caLoop);
  if (!caState.running) {
    caDraw();
    return;
  }
  if (ts - caLastTime < caState.speed) {
    caDraw();
    return;
  }
  caLastTime = ts;

  if (caState.mode === "life") {
    caState.life.step();
    caState.generation++;
  } else {
    if (caState.eca.rows.length < caState.eca.maxRows) {
      caState.eca.computeNext();
    } else {
      caState.running = false;
      caUpdatePlayPauseButton();
    }
  }
  caDraw();
}

function caStartLoop(): void {
  if (caAnimId === null) {
    caAnimId = requestAnimationFrame(caLoop);
  }
}

// ─── UI helpers ──────────────────────────────────────────────────────────────

function caUpdatePlayPauseButton(): void {
  const btn = document.getElementById("btn-playpause") as HTMLButtonElement;
  if (btn) btn.textContent = caState.running ? "⏸ Pause" : "▶ Play";
}

function caUpdateModeUI(): void {
  const lifeControls = document.getElementById("life-controls") as HTMLElement;
  const ecaControls = document.getElementById("eca-controls") as HTMLElement;
  if (caState.mode === "life") {
    lifeControls.style.display = "flex";
    ecaControls.style.display = "none";
  } else {
    lifeControls.style.display = "none";
    ecaControls.style.display = "flex";
  }
}

// ─── Mouse editing for Life ───────────────────────────────────────────────────

function caCanvasToCell(mx: number, my: number): [number, number] | null {
  const s = caState.cellSize;
  const offX = Math.floor((CANVAS_W - caState.life.cols * s) / 2);
  const offY = Math.floor((CANVAS_H - caState.life.rows * s) / 2);
  const c = Math.floor((mx - offX) / s);
  const r = Math.floor((my - offY) / s);
  if (r < 0 || r >= caState.life.rows || c < 0 || c >= caState.life.cols) return null;
  return [r, c];
}

// ─── Bind UI ──────────────────────────────────────────────────────────────────

function caBindUI(): void {
  // Mode selector
  const modeSelect = document.getElementById("mode-select") as HTMLSelectElement;
  modeSelect.addEventListener("change", function() {
    caState.mode = modeSelect.value as Mode;
    caState.running = false;
    caUpdatePlayPauseButton();
    caUpdateModeUI();
    caDraw();
  });

  // Play/Pause
  const btnPlay = document.getElementById("btn-playpause") as HTMLButtonElement;
  btnPlay.addEventListener("click", function() {
    caState.running = !caState.running;
    caUpdatePlayPauseButton();
  });

  // Step
  const btnStep = document.getElementById("btn-step") as HTMLButtonElement;
  btnStep.addEventListener("click", function() {
    if (caState.mode === "life") {
      caState.life.step();
      caState.generation++;
    } else {
      caState.eca.computeNext();
    }
    caDraw();
  });

  // Speed
  const speedRange = document.getElementById("speed-range") as HTMLInputElement;
  speedRange.addEventListener("input", function() {
    const v = parseInt(speedRange.value, 10);
    // map 1-10 to 500ms-20ms
    caState.speed = Math.round(520 - v * 50);
    const label = document.getElementById("speed-label") as HTMLElement;
    if (label) label.textContent = v.toString();
  });

  // ── Life controls ──

  // Clear
  const btnClear = document.getElementById("btn-clear") as HTMLButtonElement;
  btnClear.addEventListener("click", function() {
    caState.life.clear();
    caState.generation = 0;
    caDraw();
  });

  // Randomize
  const btnRandom = document.getElementById("btn-random") as HTMLButtonElement;
  btnRandom.addEventListener("click", function() {
    caState.life.randomize();
    caState.generation = 0;
    caDraw();
  });

  // Pattern palette
  const patternSel = document.getElementById("pattern-select") as HTMLSelectElement;
  PATTERNS.forEach(function(p, i) {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = p.name;
    patternSel.appendChild(opt);
  });
  patternSel.addEventListener("change", function() {
    caState.selectedPattern = parseInt(patternSel.value, 10);
  });

  // Place pattern
  const btnPlace = document.getElementById("btn-place") as HTMLButtonElement;
  btnPlace.addEventListener("click", function() {
    const pat = PATTERNS[caState.selectedPattern];
    caState.life.placePattern(pat, Math.floor(caState.life.rows / 2) - 5, Math.floor(caState.life.cols / 2) - 5);
    caDraw();
  });

  // Grid size
  const gridSel = document.getElementById("grid-size") as HTMLSelectElement;
  gridSel.addEventListener("change", function() {
    const val = gridSel.value.split("x");
    const cols = parseInt(val[0], 10);
    const rows = parseInt(val[1], 10);
    caState.life = new GameOfLife(rows, cols);
    caState.life.randomize();
    caState.generation = 0;
    caState.cellSize = Math.floor(Math.min(CANVAS_W / cols, CANVAS_H / rows));
    caDraw();
  });

  // Mouse draw
  let mouseDown = false;
  caCanvas.addEventListener("mousedown", function(e: MouseEvent) {
    if (caState.mode !== "life") return;
    mouseDown = true;
    const cell = caCanvasToCell(e.offsetX, e.offsetY);
    if (!cell) return;
    caState.drawValue = caState.life.get(cell[0], cell[1]) ? 0 : 1;
    caState.life.set(cell[0], cell[1], caState.drawValue);
    caDraw();
  });
  caCanvas.addEventListener("mousemove", function(e: MouseEvent) {
    if (!mouseDown || caState.mode !== "life") return;
    const cell = caCanvasToCell(e.offsetX, e.offsetY);
    if (!cell) return;
    caState.life.set(cell[0], cell[1], caState.drawValue);
    caDraw();
  });
  window.addEventListener("mouseup", function() { mouseDown = false; });

  // ── ECA controls ──

  // Rule input
  const ruleInput = document.getElementById("eca-rule") as HTMLInputElement;
  ruleInput.addEventListener("input", function() {
    let v = parseInt(ruleInput.value, 10);
    if (isNaN(v)) v = 0;
    v = clamp(v, 0, 255);
    caState.eca.rule = v;
    caState.ecaRule = v;
    const label = document.getElementById("eca-rule-label") as HTMLElement;
    if (label) label.textContent = String(v);
    caState.eca.reset(caState.ecaSingleCell);
    caState.eca.fillAll();
    caDraw();
  });

  // Famous rule buttons
  const famousRules = [30, 90, 110, 184];
  famousRules.forEach(function(r) {
    const btn = document.getElementById("eca-rule-" + r) as HTMLButtonElement;
    if (!btn) return;
    btn.addEventListener("click", function() {
      ruleInput.value = String(r);
      const label = document.getElementById("eca-rule-label") as HTMLElement;
      if (label) label.textContent = String(r);
      caState.eca.rule = r;
      caState.ecaRule = r;
      caState.eca.reset(caState.ecaSingleCell);
      caState.eca.fillAll();
      caDraw();
    });
  });

  // Init mode
  const ecaInitSel = document.getElementById("eca-init") as HTMLSelectElement;
  ecaInitSel.addEventListener("change", function() {
    caState.ecaSingleCell = ecaInitSel.value === "single";
    caState.eca.reset(caState.ecaSingleCell);
    caState.eca.fillAll();
    caDraw();
  });

  // ECA reset
  const btnEcaReset = document.getElementById("btn-eca-reset") as HTMLButtonElement;
  btnEcaReset.addEventListener("click", function() {
    caState.eca.reset(caState.ecaSingleCell);
    if (!caState.running) {
      caState.eca.fillAll();
    }
    caDraw();
  });
}

// ─── Main ─────────────────────────────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", function() {
  caCanvas = document.getElementById("ca-caCanvas") as HTMLCanvasElement;
  caCtx = caCanvas.getContext("2d")!;
  caCanvas.width = CANVAS_W;
  caCanvas.height = CANVAS_H;

  initCAState();
  caBindUI();
  caUpdateModeUI();
  caUpdatePlayPauseButton();
  caStartLoop();
});

export {};
