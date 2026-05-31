export {};

// ─── Types ─────────────────────────────────────────────────────────────────

type Player = "X" | "O";
type Cell = Player | null;
type GameMode = "ttt" | "c4";

interface TreeNode {
  id: number;
  move: number | null;
  value: number;
  alpha: number;
  beta: number;
  depth: number;
  player: Player;
  children: TreeNode[];
  pruned: boolean;
  terminal: boolean;
}

// ─── State ──────────────────────────────────────────────────────────────────

let gameMode: GameMode = "ttt";
let humanPlayer: Player = "X";
let aiPlayer: Player = "O";
let board: Cell[] = [];
let history: Cell[][] = [];
let aiDepth = 5;
let gameOver = false;
let winner: Player | null = null;
let isDraw = false;
let showTree = true;
let currentTree: TreeNode | null = null;
let nodeIdCounter = 0;
let nodesWithPruning = 0;
let nodesWithout = 0;
let c4Scores: number[] = [];
let aiStrength: "easy" | "medium" | "hard" = "hard";

// ─── Board dims ─────────────────────────────────────────────────────────────

const TTT_SIZE = 3;
const C4_ROWS = 6;
const C4_COLS = 7;

function boardSize() {
  return gameMode === "ttt" ? TTT_SIZE * TTT_SIZE : C4_ROWS * C4_COLS;
}

function freshBoard(): Cell[] {
  return Array(boardSize()).fill(null);
}

// ─── Tic-Tac-Toe logic ──────────────────────────────────────────────────────

const TTT_WINS = [
  [0,1,2],[3,4,5],[6,7,8],
  [0,3,6],[1,4,7],[2,5,8],
  [0,4,8],[2,4,6]
];

function tttWinner(b: Cell[]): Player | null {
  for (const [a,c,d] of TTT_WINS) {
    if (b[a] && b[a] === b[c] && b[a] === b[d]) return b[a] as Player;
  }
  return null;
}

function tttMoves(b: Cell[]): number[] {
  return b.map((c,i) => c === null ? i : -1).filter(i => i >= 0);
}

function tttApply(b: Cell[], move: number, p: Player): Cell[] {
  const nb = [...b];
  nb[move] = p;
  return nb;
}

// ─── Connect-Four logic ─────────────────────────────────────────────────────

function c4Col(idx: number) { return idx % C4_COLS; }
function c4Row(idx: number) { return Math.floor(idx / C4_COLS); }
function c4Idx(row: number, col: number) { return row * C4_COLS + col; }

function c4DropRow(b: Cell[], col: number): number {
  for (let r = C4_ROWS - 1; r >= 0; r--) {
    if (!b[c4Idx(r, col)]) return r;
  }
  return -1;
}

function c4Moves(b: Cell[]): number[] {
  const cols: number[] = [];
  for (let c = 0; c < C4_COLS; c++) {
    if (c4DropRow(b, c) >= 0) cols.push(c);
  }
  return cols;
}

function c4Apply(b: Cell[], col: number, p: Player): Cell[] {
  const nb = [...b];
  const r = c4DropRow(nb, col);
  if (r >= 0) nb[c4Idx(r, col)] = p;
  return nb;
}

function c4Check4(b: Cell[], p: Player): boolean {
  // horizontal
  for (let r = 0; r < C4_ROWS; r++) {
    for (let c = 0; c <= C4_COLS - 4; c++) {
      if ([0,1,2,3].every(d => b[c4Idx(r,c+d)] === p)) return true;
    }
  }
  // vertical
  for (let r = 0; r <= C4_ROWS - 4; r++) {
    for (let c = 0; c < C4_COLS; c++) {
      if ([0,1,2,3].every(d => b[c4Idx(r+d,c)] === p)) return true;
    }
  }
  // diag down-right
  for (let r = 0; r <= C4_ROWS - 4; r++) {
    for (let c = 0; c <= C4_COLS - 4; c++) {
      if ([0,1,2,3].every(d => b[c4Idx(r+d,c+d)] === p)) return true;
    }
  }
  // diag up-right
  for (let r = 3; r < C4_ROWS; r++) {
    for (let c = 0; c <= C4_COLS - 4; c++) {
      if ([0,1,2,3].every(d => b[c4Idx(r-d,c+d)] === p)) return true;
    }
  }
  return false;
}

function c4Score(b: Cell[], p: Player): number {
  const opp: Player = p === "X" ? "O" : "X";
  let score = 0;
  function scoreWindow(window: Cell[]) {
    const mine = window.filter(c => c === p).length;
    const empty = window.filter(c => c === null).length;
    const theirs = window.filter(c => c === opp).length;
    if (mine === 4) return 100;
    if (mine === 3 && empty === 1) return 5;
    if (mine === 2 && empty === 2) return 2;
    if (theirs === 3 && empty === 1) return -4;
    return 0;
  }
  // center column preference
  const center = Math.floor(C4_COLS / 2);
  for (let r = 0; r < C4_ROWS; r++) {
    if (b[c4Idx(r, center)] === p) score += 3;
  }
  // horizontal
  for (let r = 0; r < C4_ROWS; r++) {
    for (let c = 0; c <= C4_COLS - 4; c++) {
      score += scoreWindow([0,1,2,3].map(d => b[c4Idx(r,c+d)]));
    }
  }
  // vertical
  for (let r = 0; r <= C4_ROWS - 4; r++) {
    for (let c = 0; c < C4_COLS; c++) {
      score += scoreWindow([0,1,2,3].map(d => b[c4Idx(r+d,c)]));
    }
  }
  // diag
  for (let r = 0; r <= C4_ROWS - 4; r++) {
    for (let c = 0; c <= C4_COLS - 4; c++) {
      score += scoreWindow([0,1,2,3].map(d => b[c4Idx(r+d,c+d)]));
    }
  }
  for (let r = 3; r < C4_ROWS; r++) {
    for (let c = 0; c <= C4_COLS - 4; c++) {
      score += scoreWindow([0,1,2,3].map(d => b[c4Idx(r-d,c+d)]));
    }
  }
  return score;
}

// ─── Generic helpers ─────────────────────────────────────────────────────────

function currentWinner(b: Cell[]): Player | null {
  if (gameMode === "ttt") return tttWinner(b);
  if (c4Check4(b, "X")) return "X";
  if (c4Check4(b, "O")) return "O";
  return null;
}

function getMoves(b: Cell[]): number[] {
  return gameMode === "ttt" ? tttMoves(b) : c4Moves(b);
}

function applyMove(b: Cell[], move: number, p: Player): Cell[] {
  return gameMode === "ttt" ? tttApply(b, move, p) : c4Apply(b, move, p);
}

function opponent(p: Player): Player { return p === "X" ? "O" : "X"; }

// ─── Minimax with Alpha-Beta ─────────────────────────────────────────────────

function minimax(
  b: Cell[],
  depth: number,
  isMax: boolean,
  alpha: number,
  beta: number,
  player: Player,
  buildTree: boolean,
  parentNode: TreeNode | null,
  move: number | null
): number {
  nodesWithout++;
  const nodeId = nodeIdCounter++;
  let node: TreeNode | null = null;

  const w = currentWinner(b);
  const moves = getMoves(b);
  const terminal = w !== null || moves.length === 0;

  let value: number;
  if (w !== null) {
    value = w === aiPlayer ? 1000 : -1000;
    if (gameMode === "c4") value = w === aiPlayer ? 10000 : -10000;
  } else if (moves.length === 0) {
    value = 0;
  } else if (depth === 0) {
    value = gameMode === "c4" ? c4Score(b, aiPlayer) - c4Score(b, humanPlayer) : 0;
  } else {
    value = isMax ? -Infinity : Infinity;
    nodesWithPruning++;

    if (buildTree) {
      node = { id: nodeId, move, value: 0, alpha, beta, depth: aiDepth - depth, player, children: [], pruned: false, terminal: false };
      if (parentNode) parentNode.children.push(node);
      if (!currentTree) currentTree = node;
    }

    for (const m of moves) {
      const nb = applyMove(b, m, player);
      const childVal = minimax(nb, depth - 1, !isMax, alpha, beta, opponent(player), buildTree, node, m);

      if (isMax) {
        value = Math.max(value, childVal);
        alpha = Math.max(alpha, value);
      } else {
        value = Math.min(value, childVal);
        beta = Math.min(beta, value);
      }

      if (beta <= alpha) {
        // mark remaining children as pruned
        if (buildTree && node) {
          // add a pruned placeholder
          const pruneNode: TreeNode = {
            id: nodeIdCounter++, move: -1, value: 0, alpha, beta,
            depth: aiDepth - depth + 1, player: opponent(player),
            children: [], pruned: true, terminal: false
          };
          node.children.push(pruneNode);
        }
        break;
      }
    }

    if (node) node.value = value;
    return value;
  }

  nodesWithPruning++;
  if (buildTree && terminal) {
    node = { id: nodeId, move, value, alpha, beta, depth: aiDepth - depth, player, children: [], pruned: false, terminal: true };
    if (parentNode) parentNode.children.push(node);
    if (!currentTree) currentTree = node;
  }

  return value;
}

function getBestMove(b: Cell[]): { move: number; colScores: number[] } {
  nodeIdCounter = 0;
  nodesWithPruning = 0;
  nodesWithout = 0;
  currentTree = null;

  const moves = getMoves(b);
  let bestMove = moves[0];
  let bestVal = -Infinity;
  const colScores: number[] = Array(gameMode === "c4" ? C4_COLS : 9).fill(-Infinity);

  // Count nodes without pruning (simulate)
  const savedNodesW = nodesWithPruning;
  nodesWithout = 0;

  const depth = aiStrength === "easy" ? 1 : aiStrength === "medium" ? 3 : aiDepth;

  // First pass: get tree for first move only (TTT) or skip (C4)
  const doBuildTree = gameMode === "ttt" && showTree;

  let firstMove = true;
  for (const m of moves) {
    const nb = applyMove(b, m, aiPlayer);
    nodeIdCounter = 0;
    const doTree = doBuildTree && firstMove;
    if (doTree) currentTree = null;

    const val = minimax(nb, depth - 1, false, -Infinity, Infinity, humanPlayer, doTree, null, m);
    colScores[m] = val;

    if (val > bestVal) {
      bestVal = val;
      bestMove = m;
      if (doTree && currentTree) {
        // wrap with root
      }
    }
    firstMove = false;
  }

  // For TTT: rebuild full tree from root
  if (gameMode === "ttt" && showTree) {
    nodeIdCounter = 0;
    nodesWithPruning = 0;
    currentTree = null;
    const rootNode: TreeNode = {
      id: nodeIdCounter++, move: null, value: 0, alpha: -Infinity, beta: Infinity,
      depth: 0, player: aiPlayer, children: [], pruned: false, terminal: false
    };
    currentTree = rootNode;

    let rv = -Infinity;
    let ra = -Infinity;
    for (const m of moves) {
      const nb = applyMove(b, m, aiPlayer);
      const childNode: TreeNode = {
        id: nodeIdCounter++, move: m, value: 0, alpha: -Infinity, beta: Infinity,
        depth: 1, player: humanPlayer, children: [], pruned: false, terminal: false
      };
      rootNode.children.push(childNode);
      const val = minimax(nb, depth - 1, false, ra, Infinity, humanPlayer, true, childNode, m);
      childNode.value = val;
      rv = Math.max(rv, val);
      ra = Math.max(ra, rv);
    }
    rootNode.value = rv;

    // count without pruning
    nodesWithout = 0;
    for (const m of moves) {
      const nb = applyMove(b, m, aiPlayer);
      countNodes(nb, depth - 1, false, humanPlayer);
    }
    nodesWithout += moves.length + 1;
  }

  // C4: build column scores
  if (gameMode === "c4") {
    c4Scores = Array(C4_COLS).fill(-Infinity);
    for (const m of moves) {
      c4Scores[m] = colScores[m];
    }
  }

  return { move: bestMove, colScores };
}

function countNodes(b: Cell[], depth: number, isMax: boolean, player: Player) {
  nodesWithout++;
  const w = currentWinner(b);
  const moves = getMoves(b);
  if (w !== null || moves.length === 0 || depth === 0) return;
  for (const m of moves) {
    const nb = applyMove(b, m, player);
    countNodes(nb, depth - 1, !isMax, opponent(player));
  }
}

// ─── DOM helpers ─────────────────────────────────────────────────────────────

function el<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

function svg(tag: string, attrs: Record<string,string|number> = {}): SVGElement {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  return e;
}

// ─── Render board ────────────────────────────────────────────────────────────

function renderBoard() {
  const container = el("board-container");
  container.innerHTML = "";

  if (gameMode === "ttt") renderTTTBoard(container);
  else renderC4Board(container);
}

function renderTTTBoard(container: HTMLElement) {
  const size = 300;
  const cell = size / 3;
  const s = svg("svg", { width: size, height: size });
  (s as SVGElement & { style: CSSStyleDeclaration }).style.display = "block";

  for (let i = 0; i < 9; i++) {
    const r = Math.floor(i / 3);
    const c = i % 3;
    const rect = svg("rect", {
      x: c * cell, y: r * cell, width: cell - 4, height: cell - 4,
      rx: 8, fill: "#1e2040", stroke: "#3a3a6a", "stroke-width": 2,
      transform: `translate(2,2)`
    });
    if (!gameOver && board[i] === null) {
      rect.style.cursor = "pointer";
      rect.addEventListener("click", () => humanMove(i));
      rect.addEventListener("mouseenter", () => { (rect as SVGElement).setAttribute("fill", "#252850"); });
      rect.addEventListener("mouseleave", () => { (rect as SVGElement).setAttribute("fill", "#1e2040"); });
    }
    s.appendChild(rect);

    if (board[i]) {
      const text = svg("text", {
        x: c * cell + cell / 2, y: r * cell + cell / 2 + 14,
        "text-anchor": "middle", "font-size": 52, "font-weight": "bold",
        fill: board[i] === "X" ? "#f87171" : "#60a5fa"
      });
      text.textContent = board[i];
      s.appendChild(text);
    }
  }

  // grid lines
  for (let i = 1; i < 3; i++) {
    s.appendChild(svg("line", { x1: i*cell, y1: 5, x2: i*cell, y2: size-5, stroke: "#4a4a8a", "stroke-width": 2 }));
    s.appendChild(svg("line", { x1: 5, y1: i*cell, x2: size-5, y2: i*cell, stroke: "#4a4a8a", "stroke-width": 2 }));
  }

  // highlight win
  if (winner) {
    for (const [a,b2,c] of TTT_WINS) {
      if (board[a] === winner && board[b2] === winner && board[c] === winner) {
        const positions = [a,b2,c].map(idx => ({
          x: (idx%3)*cell + cell/2, y: Math.floor(idx/3)*cell + cell/2
        }));
        const line = svg("line", {
          x1: positions[0].x, y1: positions[0].y,
          x2: positions[2].x, y2: positions[2].y,
          stroke: "#fbbf24", "stroke-width": 4, "stroke-linecap": "round", opacity: 0.85
        });
        s.appendChild(line);
        break;
      }
    }
  }

  container.appendChild(s);
}

function renderC4Board(container: HTMLElement) {
  const cellW = 60, cellH = 60;
  const width = C4_COLS * cellW;
  const height = C4_ROWS * cellH + 50; // extra for score bars
  const s = svg("svg", { width, height });

  // column score bars above
  if (c4Scores.length) {
    const maxAbs = Math.max(...c4Scores.filter(v => v !== -Infinity).map(Math.abs), 1);
    for (let c = 0; c < C4_COLS; c++) {
      const val = c4Scores[c];
      if (val === -Infinity) continue;
      const norm = Math.max(-1, Math.min(1, val / maxAbs));
      const barH = Math.abs(norm) * 20;
      const fill = norm > 0 ? "#60a5fa" : "#f87171";
      s.appendChild(svg("rect", {
        x: c * cellW + 8, y: norm > 0 ? 30 - barH : 30,
        width: cellW - 16, height: barH, fill, rx: 3, opacity: 0.85
      }));
      const label = svg("text", {
        x: c * cellW + cellW / 2, y: 48, "text-anchor": "middle",
        "font-size": 9, fill: "#8888aa"
      });
      label.textContent = val.toFixed(0);
      s.appendChild(label);
    }
    // baseline
    s.appendChild(svg("line", { x1: 0, y1: 30, x2: width, y2: 30, stroke: "#3a3a6a", "stroke-width": 1 }));
  }

  const offsetY = 50;

  // board background
  s.appendChild(svg("rect", { x: 0, y: offsetY, width, height: C4_ROWS*cellH, fill: "#1a2a5a", rx: 8 }));

  // column hover areas
  for (let c = 0; c < C4_COLS; c++) {
    const dropRow = c4DropRow(board, c);
    const colRect = svg("rect", {
      x: c * cellW, y: offsetY, width: cellW, height: C4_ROWS * cellH,
      fill: "transparent"
    });
    if (!gameOver && dropRow >= 0) {
      colRect.style.cursor = "pointer";
      colRect.addEventListener("click", () => humanMove(c));
      colRect.addEventListener("mouseenter", () => {
        if (dropRow >= 0) {
          const preview = document.getElementById(`c4-preview-${c}`);
          if (preview) preview.setAttribute("opacity", "0.35");
        }
      });
      colRect.addEventListener("mouseleave", () => {
        const preview = document.getElementById(`c4-preview-${c}`);
        if (preview) preview.setAttribute("opacity", "0");
      });
    }
    s.appendChild(colRect);

    // preview circle
    if (dropRow >= 0 && !gameOver) {
      const circ = svg("circle", {
        id: `c4-preview-${c}`,
        cx: c * cellW + cellW / 2, cy: offsetY + dropRow * cellH + cellH / 2,
        r: cellW / 2 - 6, fill: humanPlayer === "X" ? "#f87171" : "#60a5fa", opacity: 0
      });
      s.appendChild(circ);
    }
  }

  // cells
  for (let r = 0; r < C4_ROWS; r++) {
    for (let c = 0; c < C4_COLS; c++) {
      const cell = board[c4Idx(r, c)];
      const cx = c * cellW + cellW / 2;
      const cy = offsetY + r * cellH + cellH / 2;
      const fill = cell === "X" ? "#f87171" : cell === "O" ? "#60a5fa" : "#0f172a";
      s.appendChild(svg("circle", {
        cx, cy, r: cellW / 2 - 5,
        fill, stroke: "#2a3a7a", "stroke-width": 2
      }));
    }
  }

  container.appendChild(s);
}

// ─── Game tree visualization ─────────────────────────────────────────────────

function renderTree() {
  const panel = el("tree-panel");
  panel.innerHTML = "";

  if (gameMode === "c4") {
    renderC4Info(panel);
    return;
  }

  if (!showTree || !currentTree) {
    panel.innerHTML = `<p style="color:#8888aa;font-size:0.85rem;margin-top:8px;">
      ${showTree ? "Make a move to see the game tree." : "Toggle 'Show Tree' to visualize minimax."}
    </p>`;
    return;
  }

  // Stats
  const stats = document.createElement("div");
  stats.className = "stats";
  stats.innerHTML = `
    <span>Nodes with α-β pruning: <b style="color:#60a5fa">${nodesWithPruning}</b></span>
    <span>Nodes without: <b style="color:#f87171">${nodesWithout}</b></span>
    <span>Saved: <b style="color:#4ade80">${Math.max(0, nodesWithout - nodesWithPruning)}</b></span>
  `;
  panel.appendChild(stats);

  // SVG tree
  const maxDisplayDepth = 4;
  const nodeR = 16;
  const levelH = 70;

  // BFS layout
  interface LayoutNode {
    node: TreeNode;
    x: number;
    y: number;
    parentX?: number;
    parentY?: number;
  }

  const laid: LayoutNode[] = [];
  let maxX = 0;

  function layout(n: TreeNode, depth: number, left: number, right: number, px?: number, py?: number) {
    if (depth > maxDisplayDepth) return;
    const x = (left + right) / 2;
    const y = depth * levelH + nodeR + 10;
    laid.push({ node: n, x, y, parentX: px, parentY: py });
    if (x > maxX) maxX = x;
    const count = n.children.length;
    if (count > 0 && depth < maxDisplayDepth) {
      const w = (right - left) / count;
      n.children.forEach((child, i) => {
        layout(child, depth + 1, left + i * w, left + (i + 1) * w, x, y);
      });
    }
  }

  const treeWidth = Math.min(800, Math.max(400, 80 * Math.pow(2, Math.min(maxDisplayDepth, 3))));
  layout(currentTree, 0, 0, treeWidth, undefined, undefined);
  const treeHeight = (Math.min(maxDisplayDepth, 4) + 1) * levelH + nodeR * 2 + 20;

  const treeSvg = svg("svg", { width: treeWidth, height: treeHeight });
  (treeSvg as any).style.cssText = "display:block;overflow:visible;";

  // Edges
  for (const ln of laid) {
    if (ln.parentX !== undefined && ln.parentY !== undefined) {
      const edge = svg("line", {
        x1: ln.parentX, y1: ln.parentY,
        x2: ln.x, y2: ln.y,
        stroke: ln.node.pruned ? "#3a3a5a" : "#4a4a7a",
        "stroke-width": ln.node.pruned ? 1 : 1.5,
        "stroke-dasharray": ln.node.pruned ? "4,3" : "none"
      });
      treeSvg.appendChild(edge);
    }
  }

  // Nodes
  for (const ln of laid) {
    const n = ln.node;
    const isAI = n.player !== aiPlayer; // This node's value is evaluated for parent
    const pruned = n.pruned;

    const g = svg("g", { transform: `translate(${ln.x},${ln.y})` });

    const fill = pruned ? "#1a1a3a" :
                 n.terminal ? (n.value > 0 ? "#1a3a2a" : n.value < 0 ? "#3a1a1a" : "#2a2a2a") :
                 isAI ? "#1e2a4a" : "#2a1e3a";
    const stroke = pruned ? "#3a3a5a" :
                   n.player === aiPlayer ? "#60a5fa" : "#f87171";

    g.appendChild(svg("circle", {
      r: nodeR, fill, stroke, "stroke-width": pruned ? 1 : 2
    }));

    if (!pruned) {
      const valText = svg("text", {
        "text-anchor": "middle", dy: "0.35em",
        "font-size": n.value === Infinity || n.value === -Infinity ? 9 : 10,
        "font-weight": "bold",
        fill: n.value > 0 ? "#4ade80" : n.value < 0 ? "#f87171" : "#e0e0f0"
      });
      valText.textContent = n.value === Infinity ? "∞" : n.value === -Infinity ? "-∞" :
        Math.abs(n.value) >= 1000 ? (n.value > 0 ? "WIN" : "LOS") : String(n.value);
      g.appendChild(valText);

      if (n.move !== null) {
        const moveLabel = svg("text", {
          "text-anchor": "middle", dy: "-1.6em", "font-size": 9, fill: "#8888aa"
        });
        moveLabel.textContent = gameMode === "ttt" ? `[${n.move}]` : `c${n.move}`;
        g.appendChild(moveLabel);
      }
    } else {
      const x = svg("text", { "text-anchor": "middle", dy: "0.35em", "font-size": 11, fill: "#444466" });
      x.textContent = "✂";
      g.appendChild(x);
    }

    // α/β labels for non-terminal non-pruned
    if (!pruned && !n.terminal && n.depth < maxDisplayDepth) {
      const abText = svg("text", {
        "text-anchor": "middle", dy: `${nodeR + 12}px`, "font-size": 8, fill: "#6666aa"
      });
      abText.textContent = `α${n.alpha === -Infinity ? "-∞" : n.alpha} β${n.beta === Infinity ? "∞" : n.beta}`;
      g.appendChild(abText);
    }

    treeSvg.appendChild(g);
  }

  // Legend
  const legend = svg("g", { transform: `translate(4, ${treeHeight - 18})` });
  const items: [string, string, string][] = [
    ["#60a5fa", "●", "AI (max)"],
    ["#f87171", "●", "Human (min)"],
    ["#3a3a5a", "- -", "Pruned branch"]
  ];
  items.forEach(([color, sym, label], i) => {
    const t = svg("text", { x: i * 130, "font-size": 9, fill: color });
    t.textContent = `${sym} ${label}`;
    legend.appendChild(t);
  });
  treeSvg.appendChild(legend);

  const wrapper = document.createElement("div");
  wrapper.style.cssText = "overflow-x:auto;margin-top:8px;";
  wrapper.appendChild(treeSvg);
  panel.appendChild(wrapper);
}

function renderC4Info(panel: HTMLElement) {
  const div = document.createElement("div");
  div.innerHTML = `
    <div class="stats">
      <span>Nodes explored: <b style="color:#60a5fa">${nodesWithPruning}</b></span>
      <span>Search depth: <b style="color:#4ade80">${aiStrength === "easy" ? 1 : aiStrength === "medium" ? 3 : aiDepth}</b></span>
    </div>
    <div style="margin-top:12px;font-size:0.82rem;color:#8888aa;">
      <p>Column scores shown above the board.</p>
      <p style="margin-top:4px;"><span style="color:#60a5fa">Blue bars</span> = AI favors this column.</p>
      <p style="margin-top:4px;"><span style="color:#f87171">Red bars</span> = AI disfavors this column.</p>
    </div>
  `;
  panel.appendChild(div);
}

// ─── Status display ──────────────────────────────────────────────────────────

function updateStatus() {
  const statusEl = el("status");
  if (winner) {
    statusEl.textContent = winner === humanPlayer ? "You win! 🎉" : "AI wins!";
    statusEl.style.color = winner === humanPlayer ? "#4ade80" : "#f87171";
  } else if (isDraw) {
    statusEl.textContent = "It's a draw!";
    statusEl.style.color = "#fbbf24";
  } else {
    const moves = getMoves(board);
    if (moves.length === 0) {
      statusEl.textContent = "Draw!";
      isDraw = true;
    } else {
      const whose = currentTurn() === humanPlayer ? "Your turn" : "AI thinking…";
      statusEl.textContent = whose;
      statusEl.style.color = currentTurn() === humanPlayer ? "#a78bfa" : "#8888aa";
    }
  }
}

function currentTurn(): Player {
  const placed = board.filter(c => c !== null).length;
  // X always goes first; determine who X is based on humanPlayer
  const xFirst = "X";
  if (placed % 2 === 0) return xFirst;
  return opponent(xFirst);
}

// ─── Game flow ───────────────────────────────────────────────────────────────

function humanMove(move: number) {
  if (gameOver) return;
  if (currentTurn() !== humanPlayer) return;

  const newBoard = applyMove(board, move, humanPlayer);
  if (JSON.stringify(newBoard) === JSON.stringify(board)) return; // invalid

  history.push([...board]);
  board = newBoard;

  checkEnd();
  renderBoard();
  updateStatus();

  if (!gameOver) {
    setTimeout(aiMove, 100);
  }
}

function aiMove() {
  if (gameOver || currentTurn() !== aiPlayer) return;

  const { move } = getBestMove(board);
  history.push([...board]);
  board = applyMove(board, move, aiPlayer);

  checkEnd();
  renderBoard();
  renderTree();
  updateStatus();
}

function checkEnd() {
  const w = currentWinner(board);
  if (w) {
    winner = w;
    gameOver = true;
  } else if (getMoves(board).length === 0) {
    isDraw = true;
    gameOver = true;
  }
}

function newGame() {
  board = freshBoard();
  history = [];
  gameOver = false;
  winner = null;
  isDraw = false;
  currentTree = null;
  c4Scores = [];
  nodesWithPruning = 0;
  nodesWithout = 0;

  renderBoard();
  renderTree();
  updateStatus();

  // If AI goes first
  if (currentTurn() === aiPlayer) {
    setTimeout(aiMove, 200);
  }
}

function undoMove() {
  if (history.length < 1) return;
  board = history.pop()!;
  // undo AI move too if last move was AI
  if (history.length >= 1 && currentTurn() === humanPlayer) {
    // already human's turn — that's fine
  }
  gameOver = false;
  winner = null;
  isDraw = false;
  currentTree = null;
  c4Scores = [];
  renderBoard();
  renderTree();
  updateStatus();
}

// ─── Init ────────────────────────────────────────────────────────────────────

function init() {
  // Game selector
  const gameSelect = el<HTMLSelectElement>("game-select");
  gameSelect.addEventListener("change", () => {
    gameMode = gameSelect.value as GameMode;
    const treeToggle = el("tree-toggle-row");
    treeToggle.style.display = gameMode === "ttt" ? "flex" : "none";
    const depthRow = el("depth-row");
    depthRow.style.display = gameMode === "c4" ? "flex" : "none";
    newGame();
  });

  // Who goes first
  const whoFirst = el<HTMLSelectElement>("who-first");
  whoFirst.addEventListener("change", () => {
    humanPlayer = whoFirst.value as Player;
    aiPlayer = opponent(humanPlayer);
    newGame();
  });

  // Show tree toggle
  const showTreeChk = el<HTMLInputElement>("show-tree");
  showTreeChk.addEventListener("change", () => {
    showTree = showTreeChk.checked;
    renderTree();
  });

  // Depth slider
  const depthSlider = el<HTMLInputElement>("depth-slider");
  const depthLabel = el("depth-label");
  depthSlider.addEventListener("input", () => {
    aiDepth = parseInt(depthSlider.value);
    depthLabel.textContent = String(aiDepth);
  });

  // Strength
  const strengthSel = el<HTMLSelectElement>("strength-select");
  strengthSel.addEventListener("change", () => {
    aiStrength = strengthSel.value as "easy" | "medium" | "hard";
  });

  // New game btn
  el("new-game-btn").addEventListener("click", newGame);

  // Undo btn
  el("undo-btn").addEventListener("click", undoMove);

  newGame();
}

document.addEventListener("DOMContentLoaded", init);
