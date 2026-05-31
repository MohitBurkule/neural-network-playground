export {};

// ─── Types ───────────────────────────────────────────────────────────────────

interface Point {
  x: number;
  y: number;
  label: number; // +1 or -1
}

interface PerceptronState {
  w: [number, number]; // weights for x, y
  b: number;           // bias
  pocketW: [number, number];
  pocketB: number;
  pocketErrors: number;
  iteration: number;
  epoch: number;
  currentIdx: number;
  misclassified: Set<number>;
  errorHistory: number[];
  converged: boolean;
  running: boolean;
  usePocket: boolean;
  showSVM: boolean;
  lr: number;
  speed: number; // ms per step
  animFrame: number | null;
  lastStepTime: number;
  dataset: string;
  points: Point[];
  addingClass: number; // +1 or -1 for click-to-add
  svmW: [number, number];
  svmB: number;
}

// ─── Datasets ─────────────────────────────────────────────────────────────────

function makeBlobs(n: number, seed: number): Point[] {
  const pts: Point[] = [];
  const rng = mulberry32(seed);
  for (let i = 0; i < n; i++) {
    const angle = rng() * Math.PI * 2;
    const r = rng() * 0.25;
    pts.push({ x: -0.45 + Math.cos(angle) * r, y: 0.1 * (rng() - 0.5) + Math.sin(angle) * r, label: -1 });
  }
  for (let i = 0; i < n; i++) {
    const angle = rng() * Math.PI * 2;
    const r = rng() * 0.25;
    pts.push({ x: 0.45 + Math.cos(angle) * r, y: 0.1 * (rng() - 0.5) + Math.sin(angle) * r, label: 1 });
  }
  return pts;
}

function makeNearSeparable(n: number, seed: number): Point[] {
  const pts: Point[] = [];
  const rng = mulberry32(seed);
  for (let i = 0; i < n; i++) {
    const angle = rng() * Math.PI * 2;
    const r = rng() * 0.35;
    pts.push({ x: -0.3 + Math.cos(angle) * r, y: Math.sin(angle) * r * 0.8, label: -1 });
  }
  for (let i = 0; i < n; i++) {
    const angle = rng() * Math.PI * 2;
    const r = rng() * 0.35;
    pts.push({ x: 0.3 + Math.cos(angle) * r, y: Math.sin(angle) * r * 0.8, label: 1 });
  }
  // add a few outliers
  pts.push({ x: 0.28, y: 0.05, label: -1 });
  pts.push({ x: -0.28, y: -0.05, label: 1 });
  return pts;
}

function makeXOR(): Point[] {
  return [
    { x: -0.5, y: -0.5, label: 1 },
    { x:  0.5, y:  0.5, label: 1 },
    { x: -0.5, y:  0.5, label: -1 },
    { x:  0.5, y: -0.5, label: -1 },
    { x: -0.6, y: -0.4, label: 1 },
    { x:  0.6, y:  0.4, label: 1 },
    { x: -0.4, y:  0.6, label: -1 },
    { x:  0.4, y: -0.6, label: -1 },
  ];
}

function makeAND(): Point[] {
  return [
    { x: -0.7, y: -0.7, label: -1 },
    { x:  0.7, y: -0.7, label: -1 },
    { x: -0.7, y:  0.7, label: -1 },
    { x:  0.7, y:  0.7, label: 1 },
    { x: -0.5, y: -0.5, label: -1 },
    { x:  0.5, y: -0.5, label: -1 },
    { x: -0.5, y:  0.5, label: -1 },
    { x:  0.5, y:  0.5, label: 1 },
  ];
}

function makeOR(): Point[] {
  return [
    { x: -0.7, y: -0.7, label: -1 },
    { x:  0.7, y: -0.7, label: 1 },
    { x: -0.7, y:  0.7, label: 1 },
    { x:  0.7, y:  0.7, label: 1 },
    { x: -0.5, y: -0.5, label: -1 },
    { x:  0.5, y: -0.5, label: 1 },
    { x: -0.5, y:  0.5, label: 1 },
    { x:  0.5, y:  0.5, label: 1 },
  ];
}

function getDataset(name: string): Point[] {
  switch (name) {
    case "blobs": return makeBlobs(15, 42);
    case "near": return makeNearSeparable(12, 77);
    case "xor": return makeXOR();
    case "and": return makeAND();
    case "or": return makeOR();
    default: return makeBlobs(15, 42);
  }
}

// ─── RNG ──────────────────────────────────────────────────────────────────────

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s += 0x6D2B79F5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ─── SVM (simple gradient descent for max-margin separator) ──────────────────

function fitSVM(points: Point[], maxIter: number = 2000): { w: [number, number]; b: number } {
  let w: [number, number] = [0.01, 0.01];
  let b = 0;
  const lr = 0.01;
  const C = 10;
  for (let iter = 0; iter < maxIter; iter++) {
    let dw: [number, number] = [w[0], w[1]]; // regularization gradient
    let db = 0;
    for (const p of points) {
      const margin = p.label * (w[0] * p.x + w[1] * p.y + b);
      if (margin < 1) {
        dw[0] -= C * p.label * p.x;
        dw[1] -= C * p.label * p.y;
        db -= C * p.label;
      }
    }
    w[0] -= lr * dw[0];
    w[1] -= lr * dw[1];
    b -= lr * db;
  }
  return { w, b };
}

// ─── Perceptron step ──────────────────────────────────────────────────────────

function predict(w: [number, number], b: number, x: number, y: number): number {
  return w[0] * x + w[1] * y + b >= 0 ? 1 : -1;
}

function countErrors(pts: Point[], w: [number, number], b: number): number {
  let c = 0;
  for (const p of pts) if (predict(w, b, p.x, p.y) !== p.label) c++;
  return c;
}

function perceptronStep(state: PerceptronState): boolean {
  const pts = state.points;
  if (pts.length === 0) return false;

  // cycle through points sequentially
  const idx = state.currentIdx % pts.length;
  state.currentIdx = idx;
  const p = pts[idx];

  const pred = predict(state.w, state.b, p.x, p.y);
  const misclass = pred !== p.label;

  if (misclass) {
    state.w[0] += state.lr * p.label * p.x;
    state.w[1] += state.lr * p.label * p.y;
    state.b += state.lr * p.label;
  }

  state.currentIdx = (idx + 1) % pts.length;

  // track epoch boundary
  if (state.currentIdx === 0) {
    state.epoch++;
    const errs = countErrors(pts, state.w, state.b);
    state.errorHistory.push(errs);

    // pocket: keep best weights
    if (state.usePocket) {
      if (errs < state.pocketErrors) {
        state.pocketErrors = errs;
        state.pocketW = [state.w[0], state.w[1]];
        state.pocketB = state.b;
      }
    }

    if (errs === 0) {
      state.converged = true;
      return false; // done
    }
  }

  state.iteration++;
  // recompute misclassified set
  state.misclassified = new Set();
  for (let i = 0; i < pts.length; i++) {
    if (predict(state.w, state.b, pts[i].x, pts[i].y) !== pts[i].label) {
      state.misclassified.add(i);
    }
  }
  return true; // not done
}

// ─── Canvas rendering ─────────────────────────────────────────────────────────

const CANVAS_W = 500;
const CANVAS_H = 500;
const PAD = 40;

function toCanvasX(x: number): number {
  return PAD + (x + 1) / 2 * (CANVAS_W - 2 * PAD);
}
function toCanvasY(y: number): number {
  return PAD + (1 - (y + 1) / 2) * (CANVAS_H - 2 * PAD);
}
function fromCanvasX(cx: number): number {
  return (cx - PAD) / (CANVAS_W - 2 * PAD) * 2 - 1;
}
function fromCanvasY(cy: number): number {
  return 1 - (cy - PAD) / (CANVAS_H - 2 * PAD) * 2;
}

function drawCanvas(ctx: CanvasRenderingContext2D, state: PerceptronState): void {
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

  // background
  ctx.fillStyle = "#0f0f1a";
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  // grid
  ctx.strokeStyle = "#1e1e38";
  ctx.lineWidth = 1;
  for (let gx = -1; gx <= 1.01; gx += 0.25) {
    ctx.beginPath();
    ctx.moveTo(toCanvasX(gx), PAD);
    ctx.lineTo(toCanvasX(gx), CANVAS_H - PAD);
    ctx.stroke();
  }
  for (let gy = -1; gy <= 1.01; gy += 0.25) {
    ctx.beginPath();
    ctx.moveTo(PAD, toCanvasY(gy));
    ctx.lineTo(CANVAS_W - PAD, toCanvasY(gy));
    ctx.stroke();
  }

  // axes
  ctx.strokeStyle = "#3a3a6a";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(PAD, toCanvasY(0)); ctx.lineTo(CANVAS_W - PAD, toCanvasY(0));
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(toCanvasX(0), PAD); ctx.lineTo(toCanvasX(0), CANVAS_H - PAD);
  ctx.stroke();

  // decision boundary: w[0]*x + w[1]*y + b = 0
  const [w0, w1] = state.w;
  const wb = state.b;
  const dw = state.usePocket && state.pocketErrors <= countErrors(state.points, state.w, state.b)
    ? { w: state.pocketW, b: state.pocketB }
    : { w: state.w, b: state.b };

  // draw SVM separator if enabled
  if (state.showSVM && state.svmW[0] !== 0) {
    ctx.save();
    ctx.setLineDash([8, 5]);
    ctx.strokeStyle = "#fbbf24aa";
    ctx.lineWidth = 1.5;
    drawBoundaryLine(ctx, state.svmW[0], state.svmW[1], state.svmB);
    ctx.restore();
    // label
    ctx.fillStyle = "#fbbf24";
    ctx.font = "11px monospace";
    ctx.fillText("SVM max-margin", PAD + 4, PAD + 14);
  }

  // pocket boundary (dashed purple if different from current)
  if (state.usePocket) {
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = "#a78bfa88";
    ctx.lineWidth = 1.5;
    drawBoundaryLine(ctx, state.pocketW[0], state.pocketW[1], state.pocketB);
    ctx.restore();
  }

  // current perceptron boundary
  if (w0 !== 0 || w1 !== 0) {
    ctx.save();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "#38bdf8";
    ctx.shadowBlur = 8;
    drawBoundaryLine(ctx, w0, w1, wb);
    ctx.restore();

    // normal (weight) vector from origin
    const wmag = Math.sqrt(w0 * w0 + w1 * w1);
    if (wmag > 0) {
      const nx = w0 / wmag * 0.18;
      const ny = w1 / wmag * 0.18;
      const ox = toCanvasX(0), oy = toCanvasY(0);
      const tx = toCanvasX(nx), ty = toCanvasY(ny);
      ctx.save();
      ctx.strokeStyle = "#34d399";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(tx, ty);
      ctx.stroke();
      // arrowhead
      const angle = Math.atan2(oy - ty, ox - tx);
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx + Math.cos(angle - 0.4) * 9, ty + Math.sin(angle - 0.4) * 9);
      ctx.lineTo(tx + Math.cos(angle + 0.4) * 9, ty + Math.sin(angle + 0.4) * 9);
      ctx.closePath();
      ctx.fillStyle = "#34d399";
      ctx.fill();
      ctx.restore();
      ctx.fillStyle = "#34d399";
      ctx.font = "11px monospace";
      ctx.fillText("w", toCanvasX(nx * 1.3), toCanvasY(ny * 1.3));
    }
  }

  // half-plane shading (subtle)
  if (w0 !== 0 || w1 !== 0) {
    const imgData = ctx.getImageData(PAD, PAD, CANVAS_W - 2 * PAD, CANVAS_H - 2 * PAD);
    const dat = imgData.data;
    const pw = CANVAS_W - 2 * PAD;
    const ph = CANVAS_H - 2 * PAD;
    for (let py = 0; py < ph; py++) {
      for (let px = 0; px < pw; px++) {
        const wx = fromCanvasX(px + PAD);
        const wy = fromCanvasY(py + PAD);
        const val = w0 * wx + w1 * wy + wb;
        const idx4 = (py * pw + px) * 4;
        if (val > 0) {
          dat[idx4] = Math.min(255, dat[idx4] + 10);
          dat[idx4 + 2] = Math.min(255, dat[idx4 + 2] + 28);
        } else {
          dat[idx4] = Math.min(255, dat[idx4] + 28);
          dat[idx4 + 2] = Math.min(255, dat[idx4 + 2] + 8);
        }
      }
    }
    ctx.putImageData(imgData, PAD, PAD);
  }

  // points
  for (let i = 0; i < state.points.length; i++) {
    const p = state.points[i];
    const cx = toCanvasX(p.x), cy = toCanvasY(p.y);
    const isMis = state.misclassified.has(i);
    const isCurrent = (i === ((state.currentIdx - 1 + state.points.length) % state.points.length));

    ctx.beginPath();
    ctx.arc(cx, cy, isCurrent ? 10 : 7, 0, Math.PI * 2);

    if (p.label === 1) {
      ctx.fillStyle = isMis ? "#f87171" : "#38bdf8";
      ctx.strokeStyle = isCurrent ? "#ffffff" : "#1e1e38";
    } else {
      ctx.fillStyle = isMis ? "#f87171" : "#fb923c";
      ctx.strokeStyle = isCurrent ? "#ffffff" : "#1e1e38";
    }
    ctx.lineWidth = isCurrent ? 2.5 : 1.5;
    ctx.fill();
    ctx.stroke();

    // shape distinction
    if (p.label === -1) {
      // draw X inside
      ctx.strokeStyle = "#0f0f1a";
      ctx.lineWidth = 1.5;
      const r = 3.5;
      ctx.beginPath();
      ctx.moveTo(cx - r, cy - r); ctx.lineTo(cx + r, cy + r);
      ctx.moveTo(cx + r, cy - r); ctx.lineTo(cx - r, cy + r);
      ctx.stroke();
    } else {
      // dot inside
      ctx.beginPath();
      ctx.arc(cx, cy, 2, 0, Math.PI * 2);
      ctx.fillStyle = "#0f0f1a";
      ctx.fill();
    }

    // misclassified ring
    if (isMis) {
      ctx.beginPath();
      ctx.arc(cx, cy, 11, 0, Math.PI * 2);
      ctx.strokeStyle = "#f87171aa";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 2]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // border
  ctx.strokeStyle = "#2a2a4a";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(PAD, PAD, CANVAS_W - 2 * PAD, CANVAS_H - 2 * PAD);
}

function drawBoundaryLine(
  ctx: CanvasRenderingContext2D,
  w0: number, w1: number, b: number
): void {
  // w0*x + w1*y + b = 0 → y = -(w0*x + b)/w1
  if (Math.abs(w1) < 1e-10 && Math.abs(w0) < 1e-10) return;
  let x1: number, y1: number, x2: number, y2: number;
  if (Math.abs(w1) > Math.abs(w0)) {
    x1 = -1.05; y1 = -(w0 * x1 + b) / w1;
    x2 = 1.05; y2 = -(w0 * x2 + b) / w1;
  } else {
    y1 = -1.05; x1 = -(w1 * y1 + b) / w0;
    y2 = 1.05; x2 = -(w1 * y2 + b) / w0;
  }
  ctx.beginPath();
  ctx.moveTo(toCanvasX(x1), toCanvasY(y1));
  ctx.lineTo(toCanvasX(x2), toCanvasY(y2));
  ctx.stroke();
}

// ─── Error curve (mini canvas) ────────────────────────────────────────────────

function drawErrorCurve(ctx2: CanvasRenderingContext2D, history: number[], total: number): void {
  const W = ctx2.canvas.width, H = ctx2.canvas.height;
  ctx2.clearRect(0, 0, W, H);
  ctx2.fillStyle = "#0f0f1a";
  ctx2.fillRect(0, 0, W, H);

  if (history.length < 2) {
    ctx2.fillStyle = "#555580";
    ctx2.font = "12px monospace";
    ctx2.fillText("Run to see error curve", 10, H / 2);
    return;
  }

  const px = 28, py = 12, pr = 18, pb = 22;
  const iw = W - px - pr, ih = H - py - pb;

  const maxE = Math.max(total, ...history);
  const scaleX = (i: number) => px + (i / (history.length - 1)) * iw;
  const scaleY = (e: number) => py + ih - (e / maxE) * ih;

  // grid
  ctx2.strokeStyle = "#1e1e38";
  ctx2.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = py + (i / 4) * ih;
    ctx2.beginPath(); ctx2.moveTo(px, y); ctx2.lineTo(px + iw, y); ctx2.stroke();
    const val = Math.round(maxE * (1 - i / 4));
    ctx2.fillStyle = "#555580";
    ctx2.font = "9px monospace";
    ctx2.fillText(String(val), 2, y + 3);
  }

  // axes
  ctx2.strokeStyle = "#3a3a6a";
  ctx2.lineWidth = 1;
  ctx2.beginPath();
  ctx2.moveTo(px, py); ctx2.lineTo(px, py + ih); ctx2.lineTo(px + iw, py + ih);
  ctx2.stroke();

  // curve
  ctx2.beginPath();
  ctx2.strokeStyle = "#f87171";
  ctx2.lineWidth = 2;
  for (let i = 0; i < history.length; i++) {
    const x = scaleX(i), y = scaleY(history[i]);
    if (i === 0) ctx2.moveTo(x, y); else ctx2.lineTo(x, y);
  }
  ctx2.stroke();

  // labels
  ctx2.fillStyle = "#8888bb";
  ctx2.font = "10px monospace";
  ctx2.fillText("epoch", px + iw / 2 - 15, H - 4);
  ctx2.fillStyle = "#8888bb";
  ctx2.save();
  ctx2.translate(8, py + ih / 2);
  ctx2.rotate(-Math.PI / 2);
  ctx2.fillText("errors", -18, 0);
  ctx2.restore();
}

// ─── Main app ─────────────────────────────────────────────────────────────────

function initState(dataset: string): PerceptronState {
  const points = getDataset(dataset);
  const total = points.length;
  return {
    w: [0.01, 0.01],
    b: 0,
    pocketW: [0.01, 0.01],
    pocketB: 0,
    pocketErrors: total,
    iteration: 0,
    epoch: 0,
    currentIdx: 0,
    misclassified: new Set(),
    errorHistory: [],
    converged: false,
    running: false,
    usePocket: false,
    showSVM: false,
    lr: 0.1,
    speed: 80,
    animFrame: null,
    lastStepTime: 0,
    dataset,
    points,
    addingClass: 1,
    svmW: [0, 0],
    svmB: 0,
  };
}

function recomputeSVM(state: PerceptronState): void {
  if (state.points.length >= 2) {
    const res = fitSVM(state.points);
    state.svmW = res.w;
    state.svmB = res.b;
  }
}

function main(): void {
  const app = document.getElementById("app")!;
  app.innerHTML = buildHTML();

  let state = initState("blobs");
  recomputeSVM(state);

  const canvas = document.getElementById("main-canvas") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  const errCanvas = document.getElementById("err-canvas") as HTMLCanvasElement;
  const ctx2 = errCanvas.getContext("2d")!;

  function render(): void {
    drawCanvas(ctx, state);
    drawErrorCurve(ctx2, state.errorHistory, state.points.length);
    updateStats();
  }

  function updateStats(): void {
    const errs = countErrors(state.points, state.w, state.b);
    const pocketErrs = state.usePocket ? state.pocketErrors : errs;
    document.getElementById("stat-iter")!.textContent = String(state.iteration);
    document.getElementById("stat-epoch")!.textContent = String(state.epoch);
    document.getElementById("stat-errors")!.textContent = String(errs);
    document.getElementById("stat-pocket")!.textContent = state.usePocket ? String(pocketErrs) : "—";
    const convEl = document.getElementById("stat-conv")!;
    if (state.converged) {
      convEl.textContent = "CONVERGED ✓";
      convEl.style.color = "#4ade80";
    } else if (state.epoch > 200) {
      convEl.textContent = "NOT CONVERGING";
      convEl.style.color = "#f87171";
    } else {
      convEl.textContent = state.running ? "running…" : "—";
      convEl.style.color = "#8888bb";
    }
    document.getElementById("lbl-lr")!.textContent = state.lr.toFixed(3);
    document.getElementById("lbl-speed")!.textContent = state.speed + "ms";
    const wEl = document.getElementById("stat-weights")!;
    wEl.textContent = `w=[${state.w[0].toFixed(3)}, ${state.w[1].toFixed(3)}] b=${state.b.toFixed(3)}`;
  }

  // animation loop
  function loop(ts: number): void {
    if (!state.running) return;
    if (ts - state.lastStepTime >= state.speed) {
      state.lastStepTime = ts;
      const cont = perceptronStep(state);
      if (!cont) {
        state.running = false;
        updatePlayPauseBtn();
      }
    }
    render();
    if (state.running) {
      state.animFrame = requestAnimationFrame(loop);
    }
  }

  function startLoop(): void {
    if (state.animFrame !== null) cancelAnimationFrame(state.animFrame);
    state.animFrame = requestAnimationFrame(loop);
  }

  function updatePlayPauseBtn(): void {
    const btn = document.getElementById("btn-playpause")!;
    btn.textContent = state.running ? "⏸ Pause" : "▶ Play";
  }

  // Controls
  document.getElementById("btn-playpause")!.addEventListener("click", () => {
    if (state.converged) return;
    state.running = !state.running;
    updatePlayPauseBtn();
    if (state.running) startLoop();
  });

  document.getElementById("btn-step")!.addEventListener("click", () => {
    if (state.converged) return;
    state.running = false;
    updatePlayPauseBtn();
    perceptronStep(state);
    render();
  });

  document.getElementById("btn-reset")!.addEventListener("click", () => {
    const ds = state.dataset;
    const lr = state.lr;
    const speed = state.speed;
    const pocket = state.usePocket;
    const showSVM = state.showSVM;
    const pts = state.points.slice(); // keep custom points
    if (state.animFrame !== null) cancelAnimationFrame(state.animFrame);
    state = initState(ds);
    state.lr = lr;
    state.speed = speed;
    state.usePocket = pocket;
    state.showSVM = showSVM;
    state.points = pts;
    recomputeSVM(state);
    updatePlayPauseBtn();
    render();
  });

  document.getElementById("sel-dataset")!.addEventListener("change", (e) => {
    const ds = (e.target as HTMLSelectElement).value;
    if (state.animFrame !== null) cancelAnimationFrame(state.animFrame);
    const lr = state.lr;
    const speed = state.speed;
    const pocket = state.usePocket;
    const showSVM = state.showSVM;
    state = initState(ds);
    state.lr = lr;
    state.speed = speed;
    state.usePocket = pocket;
    state.showSVM = showSVM;
    recomputeSVM(state);

    // show note for non-separable
    const noteEl = document.getElementById("sep-note")!;
    if (ds === "xor") {
      noteEl.textContent = "⚠ XOR is NOT linearly separable — the perceptron will oscillate forever. Toggle Pocket to keep best-so-far weights.";
      noteEl.style.display = "block";
    } else if (ds === "near") {
      noteEl.textContent = "ℹ Near-separable: a few overlapping points — perceptron may not converge. Pocket helps.";
      noteEl.style.display = "block";
    } else {
      noteEl.style.display = "none";
    }
    updatePlayPauseBtn();
    render();
  });

  document.getElementById("sl-lr")!.addEventListener("input", (e) => {
    state.lr = parseFloat((e.target as HTMLInputElement).value);
    document.getElementById("lbl-lr")!.textContent = state.lr.toFixed(3);
  });

  document.getElementById("sl-speed")!.addEventListener("input", (e) => {
    state.speed = parseInt((e.target as HTMLInputElement).value);
    document.getElementById("lbl-speed")!.textContent = state.speed + "ms";
  });

  document.getElementById("chk-pocket")!.addEventListener("change", (e) => {
    state.usePocket = (e.target as HTMLInputElement).checked;
    render();
  });

  document.getElementById("chk-svm")!.addEventListener("change", (e) => {
    state.showSVM = (e.target as HTMLInputElement).checked;
    if (state.showSVM) recomputeSVM(state);
    render();
  });

  document.getElementById("btn-class1")!.addEventListener("click", () => {
    state.addingClass = 1;
    document.getElementById("btn-class1")!.classList.add("active-class");
    document.getElementById("btn-class-1")!.classList.remove("active-class");
  });
  document.getElementById("btn-class-1")!.addEventListener("click", () => {
    state.addingClass = -1;
    document.getElementById("btn-class-1")!.classList.add("active-class");
    document.getElementById("btn-class1")!.classList.remove("active-class");
  });

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = CANVAS_W / rect.width;
    const scaleY = CANVAS_H / rect.height;
    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top) * scaleY;
    const wx = fromCanvasX(cx);
    const wy = fromCanvasY(cy);
    if (wx < -1 || wx > 1 || wy < -1 || wy > 1) return;
    state.points.push({ x: wx, y: wy, label: state.addingClass });
    if (state.showSVM) recomputeSVM(state);
    render();
  });

  render();
}

// ─── HTML builder ─────────────────────────────────────────────────────────────

function buildHTML(): string {
  return `
<div class="header">
  <h1>Perceptron Lab</h1>
  <div class="subtitle">Rosenblatt (1957) — visualize the learning algorithm, convergence theorem, and pocket algorithm</div>
</div>
<div class="main-layout">
  <div class="sidebar">
    <div class="ctrl-section">
      <div class="ctrl-label">Dataset</div>
      <select id="sel-dataset" class="ctrl-select">
        <option value="blobs">Linearly Separable Blobs</option>
        <option value="near">Near-Separable</option>
        <option value="xor">XOR (non-separable)</option>
        <option value="and">AND gate</option>
        <option value="or">OR gate</option>
      </select>
    </div>

    <div id="sep-note" class="sep-note" style="display:none"></div>

    <div class="ctrl-section">
      <div class="ctrl-label">Learning Rate</div>
      <div class="ctrl-row">
        <label>lr</label>
        <input id="sl-lr" type="range" min="0.001" max="1" step="0.001" value="0.1">
        <span id="lbl-lr">0.100</span>
      </div>
    </div>

    <div class="ctrl-section">
      <div class="ctrl-label">Animation Speed</div>
      <div class="ctrl-row">
        <label>ms/step</label>
        <input id="sl-speed" type="range" min="10" max="800" step="10" value="80">
        <span id="lbl-speed">80ms</span>
      </div>
    </div>

    <div class="ctrl-section">
      <div class="ctrl-label">Options</div>
      <div class="ctrl-row toggle-row">
        <input id="chk-pocket" type="checkbox">
        <label for="chk-pocket">Pocket algorithm (best-so-far)</label>
      </div>
      <div class="ctrl-row toggle-row" style="margin-top:4px">
        <input id="chk-svm" type="checkbox">
        <label for="chk-svm">Overlay SVM max-margin separator</label>
      </div>
    </div>

    <div class="ctrl-section">
      <div class="ctrl-label">Playback</div>
      <div class="btn-row">
        <button id="btn-playpause" class="ctrl-btn primary">▶ Play</button>
        <button id="btn-step" class="ctrl-btn">⏭ Step</button>
        <button id="btn-reset" class="ctrl-btn danger">↺ Reset</button>
      </div>
    </div>

    <div class="ctrl-section">
      <div class="ctrl-label">Add Points (click canvas)</div>
      <div class="btn-row">
        <button id="btn-class1" class="ctrl-btn active-class class-btn class1">● Class +1</button>
        <button id="btn-class-1" class="ctrl-btn class-btn class-neg1">✕ Class −1</button>
      </div>
    </div>

    <div class="ctrl-section">
      <div class="ctrl-label">Stats</div>
      <div class="stat-grid">
        <div class="stat-item"><span class="stat-key">Iteration</span><span id="stat-iter" class="stat-val">0</span></div>
        <div class="stat-item"><span class="stat-key">Epoch</span><span id="stat-epoch" class="stat-val">0</span></div>
        <div class="stat-item"><span class="stat-key">Errors</span><span id="stat-errors" class="stat-val">—</span></div>
        <div class="stat-item"><span class="stat-key">Pocket best</span><span id="stat-pocket" class="stat-val">—</span></div>
        <div class="stat-item" style="grid-column:1/-1"><span class="stat-key">Status</span><span id="stat-conv" class="stat-val">—</span></div>
        <div class="stat-item" style="grid-column:1/-1"><span class="stat-key">Weights</span><span id="stat-weights" class="stat-val mono">—</span></div>
      </div>
    </div>

    <div class="ctrl-section">
      <div class="ctrl-label">Legend</div>
      <div class="legend-row"><span class="leg-dot dot-pos"></span> Class +1 (blue circle)</div>
      <div class="legend-row"><span class="leg-dot dot-neg"></span> Class −1 (orange ✕)</div>
      <div class="legend-row"><span class="leg-dot dot-mis"></span> Misclassified</div>
      <div class="legend-row"><span class="leg-line line-current"></span> Perceptron boundary</div>
      <div class="legend-row"><span class="leg-line line-pocket"></span> Pocket best boundary</div>
      <div class="legend-row"><span class="leg-line line-svm"></span> SVM max-margin</div>
      <div class="legend-row"><span class="leg-line line-w"></span> Weight vector</div>
    </div>

    <div class="algo-box">
      <div class="ctrl-label">Algorithm</div>
      <div class="algo-text">For each point (xᵢ, yᵢ) with label yᵢ ∈ {+1,−1}:<br>
      If sign(w·xᵢ + b) ≠ yᵢ:<br>
      &nbsp;&nbsp;w ← w + lr·yᵢ·xᵢ<br>
      &nbsp;&nbsp;b ← b + lr·yᵢ<br><br>
      <b>Convergence theorem:</b> if data is linearly separable, the algorithm terminates in finite steps.<br><br>
      <b>Pocket:</b> when data is NOT separable, standard perceptron cycles. Pocket keeps the weight vector with fewest errors seen so far.</div>
    </div>
  </div>

  <div class="content">
    <div class="canvas-wrap">
      <canvas id="main-canvas" width="${CANVAS_W}" height="${CANVAS_H}"></canvas>
    </div>
    <div class="err-panel">
      <div class="panel-title">Misclassification Error per Epoch</div>
      <canvas id="err-canvas" width="460" height="130"></canvas>
    </div>
  </div>
</div>
  `;
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", main);
