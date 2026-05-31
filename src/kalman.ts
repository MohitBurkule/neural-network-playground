// Kalman Filter 2D Tracking Lab
// State: [x, y, vx, vy], constant-velocity motion model

// ─── Matrix math ─────────────────────────────────────────────────────────────
// Prefixed KMat/KPt to avoid conflicts with other files in the project.

type KMat = { rows: number; cols: number; data: number[] };

function kmat(rows: number, cols: number, data?: number[]): KMat {
  return { rows, cols, data: data ? data.slice() : new Array(rows * cols).fill(0) };
}

function mget(m: KMat, r: number, c: number): number {
  return m.data[r * m.cols + c];
}

function mset(m: KMat, r: number, c: number, v: number): void {
  m.data[r * m.cols + c] = v;
}

function mmul(a: KMat, b: KMat): KMat {
  const out = kmat(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < b.cols; j++) {
      let s = 0;
      for (let k = 0; k < a.cols; k++) s += mget(a, i, k) * mget(b, k, j);
      mset(out, i, j, s);
    }
  }
  return out;
}

function madd(a: KMat, b: KMat): KMat {
  const out = kmat(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] + b.data[i];
  return out;
}

function msub(a: KMat, b: KMat): KMat {
  const out = kmat(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] - b.data[i];
  return out;
}

function mscale(a: KMat, s: number): KMat {
  const out = kmat(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] * s;
  return out;
}

function mtranspose(a: KMat): KMat {
  const out = kmat(a.cols, a.rows);
  for (let r = 0; r < a.rows; r++) {
    for (let c = 0; c < a.cols; c++) {
      mset(out, c, r, mget(a, r, c));
    }
  }
  return out;
}

function keye(n: number): KMat {
  const m = kmat(n, n);
  for (let i = 0; i < n; i++) mset(m, i, i, 1);
  return m;
}

function inv2(m: KMat): KMat {
  const a = mget(m, 0, 0), b = mget(m, 0, 1);
  const c = mget(m, 1, 0), d = mget(m, 1, 1);
  const det = a * d - b * c;
  return kmat(2, 2, [d / det, -b / det, -c / det, a / det]);
}

// ─── Kalman filter state ──────────────────────────────────────────────────────

interface KalmanState {
  x: KMat;  // 4×1 state vector [x, y, vx, vy]
  P: KMat;  // 4×4 covariance
}

function buildF(dt: number): KMat {
  return kmat(4, 4, [
    1, 0, dt, 0,
    0, 1, 0, dt,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

function buildQ(dt: number, q: number): KMat {
  const dt2 = dt * dt;
  const dt3 = dt2 * dt;
  const dt4 = dt3 * dt;
  return kmat(4, 4, [
    q * dt4 / 4, 0, q * dt3 / 2, 0,
    0, q * dt4 / 4, 0, q * dt3 / 2,
    q * dt3 / 2, 0, q * dt2, 0,
    0, q * dt3 / 2, 0, q * dt2,
  ]);
}

// H: 2×4 measurement matrix (observe x,y only)
const KH = kmat(2, 4, [
  1, 0, 0, 0,
  0, 1, 0, 0,
]);

function buildR(r: number): KMat {
  return kmat(2, 2, [r, 0, 0, r]);
}

function kalmanPredict(state: KalmanState, F: KMat, Q: KMat): KalmanState {
  const xp = mmul(F, state.x);
  const Pp = madd(mmul(mmul(F, state.P), mtranspose(F)), Q);
  return { x: xp, P: Pp };
}

function kalmanUpdate(state: KalmanState, z: KMat, R: KMat): KalmanState {
  const y = msub(z, mmul(KH, state.x));
  const S = madd(mmul(mmul(KH, state.P), mtranspose(KH)), R);
  const K = mmul(mmul(state.P, mtranspose(KH)), inv2(S));
  const xu = madd(state.x, mmul(K, y));
  const I_KH = msub(keye(4), mmul(K, KH));
  const Pu = mmul(I_KH, state.P);
  return { x: xu, P: Pu };
}

// ─── Trajectory generators ───────────────────────────────────────────────────

interface KPt { x: number; y: number }

function circleTraj(t: number, cx: number, cy: number, r: number, speed: number): KPt {
  const angle = t * speed;
  return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
}

function fig8Traj(t: number, cx: number, cy: number, r: number, speed: number): KPt {
  const angle = t * speed;
  return {
    x: cx + r * Math.cos(angle),
    y: cy + r * 0.5 * Math.sin(2 * angle),
  };
}

// ─── App ──────────────────────────────────────────────────────────────────────

interface TrailPoint {
  true_x: number;
  true_y: number;
  meas_x: number;
  meas_y: number;
  est_x: number;
  est_y: number;
}

type KTrajMode = "circle" | "figure8" | "draw";

class KalmanApp {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private plotCanvas: HTMLCanvasElement;
  private plotCtx: CanvasRenderingContext2D;

  private playing = false;
  private t = 0;
  private dt = 0.05;
  private speed = 1.0;
  private qNoise = 0.1;
  private rNoise = 10.0;
  private mode: KTrajMode = "circle";
  private animId = 0;

  private kalmanState: KalmanState;
  private trail: TrailPoint[] = [];
  private maxTrail = 300;

  private drawPath: KPt[] = [];
  private drawPathIndex = 0;
  private isDrawing = false;

  private rSlider!: HTMLInputElement;
  private qSlider!: HTMLInputElement;
  private speedSlider!: HTMLInputElement;
  private rLabel!: HTMLSpanElement;
  private qLabel!: HTMLSpanElement;
  private speedLabel!: HTMLSpanElement;
  private modeSelect!: HTMLSelectElement;
  private playBtn!: HTMLButtonElement;
  private stepBtn!: HTMLButtonElement;
  private resetBtn!: HTMLButtonElement;

  constructor() {
    this.canvas = document.getElementById("mainCanvas") as HTMLCanvasElement;
    this.ctx = this.canvas.getContext("2d")!;
    this.plotCanvas = document.getElementById("plotCanvas") as HTMLCanvasElement;
    this.plotCtx = this.plotCanvas.getContext("2d")!;
    this.kalmanState = this.initKalman();
    this.bindControls();
    this.bindDrawEvents();
    this.render();
  }

  private initKalman(): KalmanState {
    const start = this.groundTruth(0);
    return {
      x: kmat(4, 1, [start.x, start.y, 0, 0]),
      P: mscale(keye(4), 100),
    };
  }

  private groundTruth(t: number): KPt {
    const cx = this.canvas.width / 2;
    const cy = this.canvas.height / 2;
    const r = Math.min(cx, cy) * 0.55;
    if (this.mode === "circle") return circleTraj(t, cx, cy, r, this.speed);
    if (this.mode === "figure8") return fig8Traj(t, cx, cy, r, this.speed);
    if (this.drawPath.length < 2) return { x: cx, y: cy };
    const idx = this.drawPathIndex % this.drawPath.length;
    return this.drawPath[idx];
  }

  private addNoise(p: KPt, sigma: number): KPt {
    const u1 = Math.random(), u2 = Math.random();
    const n1 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    const n2 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.sin(2 * Math.PI * u2);
    return { x: p.x + n1 * sigma, y: p.y + n2 * sigma };
  }

  private step(): void {
    this.t += this.dt;
    if (this.mode === "draw") this.drawPathIndex++;

    const truth = this.groundTruth(this.t);
    const meas = this.addNoise(truth, Math.sqrt(this.rNoise));

    const F = buildF(this.dt);
    const Q = buildQ(this.dt, this.qNoise);
    const R = buildR(this.rNoise);

    let state = kalmanPredict(this.kalmanState, F, Q);
    const z = kmat(2, 1, [meas.x, meas.y]);
    state = kalmanUpdate(state, z, R);
    this.kalmanState = state;

    const ex = mget(state.x, 0, 0);
    const ey = mget(state.x, 1, 0);

    this.trail.push({
      true_x: truth.x, true_y: truth.y,
      meas_x: meas.x, meas_y: meas.y,
      est_x: ex, est_y: ey,
    });
    if (this.trail.length > this.maxTrail) this.trail.shift();
  }

  private reset(): void {
    this.t = 0;
    this.drawPathIndex = 0;
    this.trail = [];
    this.kalmanState = this.initKalman();
    this.render();
  }

  private drawCovarEllipse(cx: number, cy: number): void {
    const P = this.kalmanState.P;
    const pxx = mget(P, 0, 0);
    const pxy = mget(P, 0, 1);
    const pyy = mget(P, 1, 1);

    const trace = pxx + pyy;
    const det = pxx * pyy - pxy * pxy;
    const disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
    const l1 = trace / 2 + disc;
    const l2 = trace / 2 - disc;

    const a = Math.sqrt(Math.max(0, l1)) * 2.45;
    const b = Math.sqrt(Math.max(0, l2)) * 2.45;
    const angle = Math.atan2(pxy, pxx - l2);

    this.ctx.save();
    this.ctx.translate(cx, cy);
    this.ctx.rotate(angle);
    this.ctx.beginPath();
    this.ctx.ellipse(0, 0, Math.max(a, 1), Math.max(b, 1), 0, 0, Math.PI * 2);
    this.ctx.strokeStyle = "rgba(100,220,255,0.6)";
    this.ctx.lineWidth = 1.5;
    this.ctx.fillStyle = "rgba(100,220,255,0.07)";
    this.ctx.fill();
    this.ctx.stroke();
    this.ctx.restore();
  }

  private render(): void {
    const W = this.canvas.width;
    const H2 = this.canvas.height;
    const ctx = this.ctx;

    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, W, H2);

    if (this.mode === "draw") {
      if (this.drawPath.length < 2 && !this.isDrawing) {
        ctx.fillStyle = "#888";
        ctx.font = "16px monospace";
        ctx.textAlign = "center";
        ctx.fillText("Draw your path on this canvas", W / 2, H2 / 2);
        ctx.fillText("(click and drag, then press Play)", W / 2, H2 / 2 + 24);
      }
      if (this.drawPath.length > 1) {
        ctx.strokeStyle = "rgba(255,255,100,0.25)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(this.drawPath[0].x, this.drawPath[0].y);
        for (let i = 1; i < this.drawPath.length; i++) {
          ctx.lineTo(this.drawPath[i].x, this.drawPath[i].y);
        }
        ctx.stroke();
      }
    } else {
      // Faint future path
      const steps = 120;
      ctx.beginPath();
      ctx.setLineDash([4, 6]);
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= steps; i++) {
        const pt = this.groundTruth(i / steps * (Math.PI * 2 / this.speed));
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Trail
    for (let i = 0; i < this.trail.length; i++) {
      const alpha = (i + 1) / this.trail.length;
      const tp = this.trail[i];

      if (i > 0) {
        const prev = this.trail[i - 1];
        ctx.beginPath();
        ctx.moveTo(prev.true_x, prev.true_y);
        ctx.lineTo(tp.true_x, tp.true_y);
        ctx.strokeStyle = "rgba(80,220,80," + (alpha * 0.7) + ")";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(tp.meas_x, tp.meas_y, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,120,80," + (alpha * 0.6) + ")";
      ctx.fill();

      if (i > 0) {
        const prev = this.trail[i - 1];
        ctx.beginPath();
        ctx.moveTo(prev.est_x, prev.est_y);
        ctx.lineTo(tp.est_x, tp.est_y);
        ctx.strokeStyle = "rgba(100,180,255," + (alpha * 0.9) + ")";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    if (this.trail.length > 0) {
      const last = this.trail[this.trail.length - 1];
      this.drawCovarEllipse(last.est_x, last.est_y);

      ctx.beginPath();
      ctx.arc(last.est_x, last.est_y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#64b4ff";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(last.true_x, last.true_y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#50dc50";
      ctx.fill();
    }

    // Legend
    ctx.font = "13px monospace";
    ctx.textAlign = "left";
    const legendItems: Array<[string, string]> = [
      ["#50dc50", "True trajectory"],
      ["rgba(255,120,80,0.9)", "Noisy measurements"],
      ["#64b4ff", "Kalman estimate"],
      ["rgba(100,220,255,0.6)", "95% covariance ellipse"],
    ];
    legendItems.forEach(function(item, i) {
      ctx.fillStyle = item[0];
      ctx.fillRect(14, 14 + i * 20, 14, 3);
      ctx.fillStyle = "#aaa";
      ctx.fillText(item[1], 34, 22 + i * 20);
    });

    this.renderPlot();
  }

  private renderPlot(): void {
    const W = this.plotCanvas.width;
    const H = this.plotCanvas.height;
    const ctx = this.plotCtx;

    ctx.fillStyle = "#0d0d0d";
    ctx.fillRect(0, 0, W, H);

    if (this.trail.length < 2) {
      ctx.fillStyle = "#555";
      ctx.font = "12px monospace";
      ctx.textAlign = "center";
      ctx.fillText("x(t) will appear here", W / 2, H / 2);
      return;
    }

    let minX = Infinity, maxX = -Infinity;
    this.trail.forEach(function(tp) {
      if (tp.true_x < minX) minX = tp.true_x;
      if (tp.meas_x < minX) minX = tp.meas_x;
      if (tp.est_x < minX) minX = tp.est_x;
      if (tp.true_x > maxX) maxX = tp.true_x;
      if (tp.meas_x > maxX) maxX = tp.meas_x;
      if (tp.est_x > maxX) maxX = tp.est_x;
    });
    const pad = 20;
    const xRange = maxX - minX || 1;
    const n = this.trail.length;

    const toPlot = function(val: number, row: number): [number, number] {
      const px = pad + (row / (n - 1)) * (W - 2 * pad);
      const py = pad + (1 - (val - minX) / xRange) * (H - 2 * pad);
      return [px, py];
    };

    // Uncertainty band — use current sigma (approximation)
    const sigma = Math.sqrt(Math.max(0, mget(this.kalmanState.P, 0, 0)));
    ctx.beginPath();
    this.trail.forEach(function(tp, i) {
      const p = toPlot(tp.est_x + sigma * 2, i);
      if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
    });
    for (let i = n - 1; i >= 0; i--) {
      const p = toPlot(this.trail[i].est_x - sigma * 2, i);
      ctx.lineTo(p[0], p[1]);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(100,180,255,0.15)";
    ctx.fill();

    // True x
    ctx.beginPath();
    this.trail.forEach(function(tp, i) {
      const p = toPlot(tp.true_x, i);
      if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
    });
    ctx.strokeStyle = "rgba(80,220,80,0.7)";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Measurement x dots
    const plotCtx = ctx;
    this.trail.forEach(function(tp, i) {
      const p = toPlot(tp.meas_x, i);
      plotCtx.beginPath();
      plotCtx.arc(p[0], p[1], 1.5, 0, Math.PI * 2);
      plotCtx.fillStyle = "rgba(255,120,80,0.5)";
      plotCtx.fill();
    });

    // Estimate x
    ctx.beginPath();
    this.trail.forEach(function(tp, i) {
      const p = toPlot(tp.est_x, i);
      if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
    });
    ctx.strokeStyle = "rgba(100,180,255,0.9)";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#555";
    ctx.font = "11px monospace";
    ctx.textAlign = "left";
    ctx.fillText("x-position over time  (blue=estimate, green=true, red=meas, band=±2σ)", 6, 11);
  }

  private loop(): void {
    this.step();
    this.render();
    if (this.playing) {
      this.animId = requestAnimationFrame(() => this.loop());
    }
  }

  private bindControls(): void {
    this.rSlider = document.getElementById("rSlider") as HTMLInputElement;
    this.qSlider = document.getElementById("qSlider") as HTMLInputElement;
    this.speedSlider = document.getElementById("speedSlider") as HTMLInputElement;
    this.rLabel = document.getElementById("rLabel") as HTMLSpanElement;
    this.qLabel = document.getElementById("qLabel") as HTMLSpanElement;
    this.speedLabel = document.getElementById("speedLabel") as HTMLSpanElement;
    this.modeSelect = document.getElementById("modeSelect") as HTMLSelectElement;
    this.playBtn = document.getElementById("playBtn") as HTMLButtonElement;
    this.stepBtn = document.getElementById("stepBtn") as HTMLButtonElement;
    this.resetBtn = document.getElementById("resetBtn") as HTMLButtonElement;

    this.rSlider.addEventListener("input", () => {
      this.rNoise = parseFloat(this.rSlider.value);
      this.rLabel.textContent = this.rNoise.toFixed(1);
    });
    this.qSlider.addEventListener("input", () => {
      this.qNoise = parseFloat(this.qSlider.value);
      this.qLabel.textContent = this.qNoise.toFixed(3);
    });
    this.speedSlider.addEventListener("input", () => {
      this.speed = parseFloat(this.speedSlider.value);
      this.speedLabel.textContent = this.speed.toFixed(2);
    });
    this.modeSelect.addEventListener("change", () => {
      this.mode = this.modeSelect.value as KTrajMode;
      this.playing = false;
      cancelAnimationFrame(this.animId);
      this.playBtn.textContent = "▶ Play";
      this.drawPath = [];
      this.drawPathIndex = 0;
      this.reset();
    });
    this.playBtn.addEventListener("click", () => {
      if (this.mode === "draw" && this.drawPath.length < 2) {
        alert("Draw a path on the canvas first!");
        return;
      }
      this.playing = !this.playing;
      this.playBtn.textContent = this.playing ? "⏸ Pause" : "▶ Play";
      if (this.playing) this.loop();
    });
    this.stepBtn.addEventListener("click", () => {
      if (this.mode === "draw" && this.drawPath.length < 2) return;
      this.playing = false;
      this.playBtn.textContent = "▶ Play";
      this.step();
      this.render();
    });
    this.resetBtn.addEventListener("click", () => {
      this.playing = false;
      cancelAnimationFrame(this.animId);
      this.playBtn.textContent = "▶ Play";
      this.reset();
    });
  }

  private bindDrawEvents(): void {
    this.canvas.addEventListener("mousedown", (e) => {
      if (this.mode !== "draw") return;
      this.isDrawing = true;
      this.drawPath = [];
      const rect = this.canvas.getBoundingClientRect();
      this.drawPath.push({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    });
    this.canvas.addEventListener("mousemove", (e) => {
      if (!this.isDrawing || this.mode !== "draw") return;
      const rect = this.canvas.getBoundingClientRect();
      this.drawPath.push({ x: e.clientX - rect.left, y: e.clientY - rect.top });
      this.render();
    });
    this.canvas.addEventListener("mouseup", () => {
      if (this.mode !== "draw") return;
      this.isDrawing = false;
      this.drawPathIndex = 0;
      this.reset();
      this.render();
    });
    this.canvas.addEventListener("touchstart", (e) => {
      if (this.mode !== "draw") return;
      e.preventDefault();
      this.isDrawing = true;
      this.drawPath = [];
      const rect = this.canvas.getBoundingClientRect();
      const touch = e.touches[0];
      this.drawPath.push({ x: touch.clientX - rect.left, y: touch.clientY - rect.top });
    });
    this.canvas.addEventListener("touchmove", (e) => {
      if (!this.isDrawing || this.mode !== "draw") return;
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      const touch = e.touches[0];
      this.drawPath.push({ x: touch.clientX - rect.left, y: touch.clientY - rect.top });
      this.render();
    });
    this.canvas.addEventListener("touchend", () => {
      if (this.mode !== "draw") return;
      this.isDrawing = false;
      this.drawPathIndex = 0;
      this.reset();
      this.render();
    });
  }
}

window.addEventListener("load", () => {
  new KalmanApp();
});

export {};
