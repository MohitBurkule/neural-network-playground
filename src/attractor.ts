export {};

// ─── Types ────────────────────────────────────────────────────────────────────

interface Vec3 { x: number; y: number; z: number; }

type DerivFn = (p: Vec3, params: number[]) => Vec3;

interface SystemDef {
  name: string;
  params: { label: string; min: number; max: number; value: number }[];
  deriv: DerivFn;
  initPos: Vec3;
  scale: number;
  offset: Vec3;
}

interface TrailPoint {
  x: number; y: number; z: number;
  speed: number;
  age: number;
}

// ─── Attractor Definitions ────────────────────────────────────────────────────

const SYSTEMS: Record<string, SystemDef> = {
  Lorenz: {
    name: "Lorenz",
    params: [
      { label: "σ", min: 1, max: 20, value: 10 },
      { label: "ρ", min: 1, max: 60, value: 28 },
      { label: "β", min: 0.1, max: 8, value: 2.667 },
    ],
    deriv: (p, [s, r, b]) => ({
      x: s * (p.y - p.x),
      y: p.x * (r - p.z) - p.y,
      z: p.x * p.y - b * p.z,
    }),
    initPos: { x: 1, y: 1, z: 1 },
    scale: 14,
    offset: { x: 0, y: 0, z: -25 },
  },
  Rössler: {
    name: "Rössler",
    params: [
      { label: "a", min: 0.01, max: 0.5, value: 0.2 },
      { label: "b", min: 0.01, max: 0.5, value: 0.2 },
      { label: "c", min: 1, max: 15, value: 5.7 },
    ],
    deriv: (p, [a, b, c]) => ({
      x: -(p.y + p.z),
      y: p.x + a * p.y,
      z: b + p.z * (p.x - c),
    }),
    initPos: { x: 1, y: 1, z: 1 },
    scale: 8,
    offset: { x: 0, y: 0, z: -5 },
  },
  Aizawa: {
    name: "Aizawa",
    params: [
      { label: "a", min: 0.5, max: 1.5, value: 0.95 },
      { label: "b", min: 0.1, max: 1.0, value: 0.7 },
      { label: "c", min: 0.1, max: 1.0, value: 0.6 },
      { label: "d", min: 1, max: 5, value: 3.5 },
      { label: "e", min: 0.1, max: 1.0, value: 0.25 },
      { label: "f", min: 0.01, max: 0.5, value: 0.1 },
    ],
    deriv: (p, [a, b, c, d, e, f]) => ({
      x: (p.z - b) * p.x - d * p.y,
      y: d * p.x + (p.z - b) * p.y,
      z: c + a * p.z - (p.z * p.z * p.z) / 3 - (p.x * p.x + p.y * p.y) * (1 + e * p.z) + f * p.z * p.x * p.x * p.x,
    }),
    initPos: { x: 0.1, y: 0, z: 0 },
    scale: 70,
    offset: { x: 0, y: 0, z: 0 },
  },
  Thomas: {
    name: "Thomas",
    params: [
      { label: "b", min: 0.1, max: 0.5, value: 0.19 },
    ],
    deriv: (p, [b]) => ({
      x: Math.sin(p.y) - b * p.x,
      y: Math.sin(p.z) - b * p.y,
      z: Math.sin(p.x) - b * p.z,
    }),
    initPos: { x: 0.1, y: 0, z: 0 },
    scale: 18,
    offset: { x: 0, y: 0, z: 0 },
  },
  Halvorsen: {
    name: "Halvorsen",
    params: [
      { label: "a", min: 0.5, max: 3, value: 1.4 },
    ],
    deriv: (p, [a]) => ({
      x: -a * p.x - 4 * p.y - 4 * p.z - p.y * p.y,
      y: -a * p.y - 4 * p.z - 4 * p.x - p.z * p.z,
      z: -a * p.z - 4 * p.x - 4 * p.y - p.x * p.x,
    }),
    initPos: { x: -5, y: 0, z: 0 },
    scale: 8,
    offset: { x: 0, y: 0, z: 0 },
  },
  Chen: {
    name: "Chen",
    params: [
      { label: "a", min: 20, max: 50, value: 35 },
      { label: "b", min: 1, max: 10, value: 3 },
      { label: "c", min: 10, max: 30, value: 28 },
    ],
    deriv: (p, [a, b, c]) => ({
      x: a * (p.y - p.x),
      y: (c - a) * p.x - p.x * p.z + c * p.y,
      z: p.x * p.y - b * p.z,
    }),
    initPos: { x: -0.1, y: 0.5, z: -0.6 },
    scale: 5,
    offset: { x: 0, y: 0, z: -5 },
  },
};

// ─── RK4 integrator ──────────────────────────────────────────────────────────

function rk4(p: Vec3, dt: number, deriv: DerivFn, params: number[]): Vec3 {
  const k1 = deriv(p, params);
  const p2: Vec3 = { x: p.x + k1.x * dt / 2, y: p.y + k1.y * dt / 2, z: p.z + k1.z * dt / 2 };
  const k2 = deriv(p2, params);
  const p3: Vec3 = { x: p.x + k2.x * dt / 2, y: p.y + k2.y * dt / 2, z: p.z + k2.z * dt / 2 };
  const k3 = deriv(p3, params);
  const p4: Vec3 = { x: p.x + k3.x * dt, y: p.y + k3.y * dt, z: p.z + k3.z * dt };
  const k4 = deriv(p4, params);
  return {
    x: p.x + (k1.x + 2 * k2.x + 2 * k3.x + k4.x) * dt / 6,
    y: p.y + (k1.y + 2 * k2.y + 2 * k3.y + k4.y) * dt / 6,
    z: p.z + (k1.z + 2 * k2.z + 2 * k3.z + k4.z) * dt / 6,
  };
}

// ─── 3D→2D Projection ────────────────────────────────────────────────────────

function project(p: Vec3, rotX: number, rotY: number, scale: number, cx: number, cy: number): [number, number] {
  // rotate around Y
  const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
  const x1 = p.x * cosY + p.z * sinY;
  const z1 = -p.x * sinY + p.z * cosY;
  // rotate around X
  const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
  const y2 = p.y * cosX - z1 * sinX;
  const z2 = p.y * sinX + z1 * cosX;
  const depth = 1 / (1 + z2 * 0.01);
  return [cx + x1 * scale * depth, cy - y2 * scale * depth];
}

// ─── Color helpers ────────────────────────────────────────────────────────────

function speedColor(speed: number, alpha: number): string {
  // map speed to hue: slow=blue, fast=red
  const h = Math.max(0, 240 - speed * 2);
  return `hsla(${h},90%,60%,${alpha.toFixed(3)})`;
}

function altColor(speed: number, alpha: number): string {
  const h = Math.max(0, 60 - speed * 1.5);
  return `hsla(${h},90%,65%,${alpha.toFixed(3)})`;
}

// ─── Logistic Map Bifurcation ─────────────────────────────────────────────────

function drawBifurcation(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext("2d")!;
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#0a0a0f";
  ctx.fillRect(0, 0, W, H);

  const rMin = 2.5, rMax = 4.0;
  const steps = W * 2;
  const warmup = 500, plot = 300;

  ctx.fillStyle = "rgba(100,200,255,0.35)";
  for (let i = 0; i <= steps; i++) {
    const r = rMin + (rMax - rMin) * i / steps;
    let x = 0.5;
    for (let j = 0; j < warmup; j++) x = r * x * (1 - x);
    for (let j = 0; j < plot; j++) {
      x = r * x * (1 - x);
      const px = (r - rMin) / (rMax - rMin) * W;
      const py = (1 - x) * H;
      ctx.fillRect(px, py, 1, 1);
    }
  }

  // Labels
  ctx.fillStyle = "#aaa";
  ctx.font = "13px monospace";
  ctx.fillText("Logistic Map Bifurcation  x→ r·x·(1−x)", 14, 22);
  ctx.fillText("r →", W - 40, H - 8);
  ctx.fillText("x", 6, H / 2);
}

// ─── Main App ─────────────────────────────────────────────────────────────────

(function main() {
  // ── DOM ──
  const root = document.getElementById("app")!;
  root.innerHTML = `
    <div id="sidebar">
      <h2>Strange Attractors</h2>
      <label>System</label>
      <select id="sysSelect">${Object.keys(SYSTEMS).map(k => `<option value="${k}">${k}</option>`).join("")}<option value="Bifurcation">Bifurcation (2D)</option></select>

      <div id="paramSliders"></div>

      <hr/>
      <label>Speed / dt <span id="dtVal"></span></label>
      <input type="range" id="dtSlider" min="0.1" max="5" step="0.05" value="1"/>

      <label>Point budget <span id="budgetVal"></span></label>
      <input type="range" id="budgetSlider" min="500" max="20000" step="500" value="6000"/>

      <hr/>
      <label class="row"><input type="checkbox" id="autoRotate" checked/> Auto-rotate</label>
      <label class="row"><input type="checkbox" id="diverge"/> Two-trajectory divergence</label>

      <button id="resetBtn">Reset</button>
      <button id="bifBtn">Bifurcation diagram</button>

      <div id="info">
        <small>Drag to rotate • Scroll reserved</small><br/>
        <small id="statsLine"></small>
      </div>
    </div>
    <canvas id="canvas"></canvas>
  `;

  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  const sysSelect = document.getElementById("sysSelect") as HTMLSelectElement;
  const paramDiv = document.getElementById("paramSliders")!;
  const dtSlider = document.getElementById("dtSlider") as HTMLInputElement;
  const dtVal = document.getElementById("dtVal")!;
  const budgetSlider = document.getElementById("budgetSlider") as HTMLInputElement;
  const budgetVal = document.getElementById("budgetVal")!;
  const autoRotateCb = document.getElementById("autoRotate") as HTMLInputElement;
  const divergeCb = document.getElementById("diverge") as HTMLInputElement;
  const resetBtn = document.getElementById("resetBtn")!;
  const bifBtn = document.getElementById("bifBtn")!;
  const statsLine = document.getElementById("statsLine")!;

  // ── State ──
  let currentKey = "Lorenz";
  let trail1: TrailPoint[] = [];
  let trail2: TrailPoint[] = [];
  let pos1: Vec3 = { x: 0, y: 0, z: 0 };
  let pos2: Vec3 = { x: 0, y: 0, z: 0 };
  let rotX = 0.3, rotY = 0;
  let dragging = false, lastMX = 0, lastMY = 0;
  let bifMode = false;
  let animId = 0;
  let frameCount = 0;

  function resize() {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    if (bifMode) drawBifurcation(canvas);
  }

  window.addEventListener("resize", resize);

  // ── Build param sliders ──
  function buildParams(key: string) {
    paramDiv.innerHTML = "";
    if (!(key in SYSTEMS)) return;
    const sys = SYSTEMS[key];
    sys.params.forEach((p, i) => {
      const lbl = document.createElement("label");
      const span = document.createElement("span");
      span.textContent = `${p.label}  `;
      const valSpan = document.createElement("span");
      valSpan.id = `pv_${i}`;
      valSpan.textContent = p.value.toFixed(3);
      lbl.appendChild(span);
      lbl.appendChild(valSpan);
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = String(p.min);
      slider.max = String(p.max);
      slider.step = String((p.max - p.min) / 200);
      slider.value = String(p.value);
      slider.id = `ps_${i}`;
      slider.addEventListener("input", () => {
        p.value = parseFloat(slider.value);
        valSpan.textContent = p.value.toFixed(3);
        resetTrails();
      });
      paramDiv.appendChild(lbl);
      paramDiv.appendChild(slider);
    });
  }

  function getParams(): number[] {
    if (!(currentKey in SYSTEMS)) return [];
    return SYSTEMS[currentKey].params.map(p => p.value);
  }

  function resetTrails() {
    if (!(currentKey in SYSTEMS)) return;
    const sys = SYSTEMS[currentKey];
    pos1 = { ...sys.initPos };
    pos2 = { ...sys.initPos, x: sys.initPos.x + 1e-5 };
    trail1 = [];
    trail2 = [];
    frameCount = 0;
  }

  // ── Drag to rotate ──
  canvas.addEventListener("mousedown", e => { dragging = true; lastMX = e.clientX; lastMY = e.clientY; });
  window.addEventListener("mouseup", () => { dragging = false; });
  window.addEventListener("mousemove", e => {
    if (!dragging) return;
    rotY += (e.clientX - lastMX) * 0.005;
    rotX += (e.clientY - lastMY) * 0.005;
    lastMX = e.clientX; lastMY = e.clientY;
  });

  canvas.addEventListener("touchstart", e => {
    dragging = true;
    lastMX = e.touches[0].clientX; lastMY = e.touches[0].clientY;
  }, { passive: true });
  canvas.addEventListener("touchend", () => { dragging = false; });
  canvas.addEventListener("touchmove", e => {
    if (!dragging) return;
    rotY += (e.touches[0].clientX - lastMX) * 0.005;
    rotX += (e.touches[0].clientY - lastMY) * 0.005;
    lastMX = e.touches[0].clientX; lastMY = e.touches[0].clientY;
  }, { passive: true });

  // ── System select ──
  sysSelect.addEventListener("change", () => {
    currentKey = sysSelect.value;
    bifMode = (currentKey === "Bifurcation");
    buildParams(currentKey);
    resetTrails();
    if (bifMode) {
      setTimeout(() => { resize(); }, 50);
    }
  });

  resetBtn.addEventListener("click", resetTrails);
  bifBtn.addEventListener("click", () => {
    sysSelect.value = "Bifurcation";
    currentKey = "Bifurcation";
    bifMode = true;
    buildParams("Bifurcation");
    setTimeout(() => { resize(); }, 50);
  });

  // ── Animation loop ──
  function step() {
    if (bifMode) {
      animId = requestAnimationFrame(step);
      return;
    }

    const sys = SYSTEMS[currentKey];
    if (!sys) { animId = requestAnimationFrame(step); return; }

    const params = getParams();
    const dtMult = parseFloat(dtSlider.value);
    const budget = parseInt(budgetSlider.value, 10);
    dtVal.textContent = (0.005 * dtMult).toFixed(4);
    budgetVal.textContent = String(budget);

    const dt = 0.005 * dtMult;
    const stepsPerFrame = Math.max(1, Math.floor(dtMult * 3));

    for (let s = 0; s < stepsPerFrame; s++) {
      const prev1 = { ...pos1 };
      pos1 = rk4(pos1, dt, sys.deriv, params);
      const dx1 = pos1.x - prev1.x, dy1 = pos1.y - prev1.y, dz1 = pos1.z - prev1.z;
      const spd1 = Math.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1) / dt;
      trail1.push({ ...pos1, speed: spd1, age: 0 });

      if (divergeCb.checked) {
        const prev2 = { ...pos2 };
        pos2 = rk4(pos2, dt, sys.deriv, params);
        const dx2 = pos2.x - prev2.x, dy2 = pos2.y - prev2.y, dz2 = pos2.z - prev2.z;
        const spd2 = Math.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2) / dt;
        trail2.push({ ...pos2, speed: spd2, age: 0 });
      }
    }

    // Age & trim
    for (let i = 0; i < trail1.length; i++) trail1[i].age++;
    for (let i = 0; i < trail2.length; i++) trail2[i].age++;
    if (trail1.length > budget) trail1 = trail1.slice(trail1.length - budget);
    if (trail2.length > budget) trail2 = trail2.slice(trail2.length - budget);

    if (autoRotateCb.checked && !dragging) {
      rotY += 0.003;
    }

    frameCount++;
    draw();
    animId = requestAnimationFrame(step);
  }

  function draw() {
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = "rgba(10,10,15,0.25)";
    ctx.fillRect(0, 0, W, H);

    const sys = SYSTEMS[currentKey];
    if (!sys) return;

    const scale = sys.scale * Math.min(W, H) / 600;
    const cx = W / 2 + sys.offset.x * scale;
    const cy = H / 2 - sys.offset.y * scale;

    // Draw trail1
    const len1 = trail1.length;
    for (let i = 1; i < len1; i++) {
      const t = i / len1;
      const alpha = Math.pow(t, 1.5) * 0.9;
      const pt = trail1[i];
      const [sx, sy] = project(pt, rotX, rotY, scale, cx, cy);
      const [px, py] = project(trail1[i - 1], rotX, rotY, scale, cx, cy);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(sx, sy);
      ctx.strokeStyle = speedColor(pt.speed * 0.05, alpha);
      ctx.lineWidth = t * 1.5 + 0.3;
      ctx.stroke();
    }

    // Draw trail2
    if (divergeCb.checked) {
      const len2 = trail2.length;
      for (let i = 1; i < len2; i++) {
        const t = i / len2;
        const alpha = Math.pow(t, 1.5) * 0.85;
        const pt = trail2[i];
        const [sx, sy] = project(pt, rotX, rotY, scale, cx, cy);
        const [px, py] = project(trail2[i - 1], rotX, rotY, scale, cx, cy);
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(sx, sy);
        ctx.strokeStyle = altColor(pt.speed * 0.05, alpha);
        ctx.lineWidth = t * 1.5 + 0.3;
        ctx.stroke();
      }

      // Divergence distance
      if (trail1.length > 0 && trail2.length > 0) {
        const p1 = trail1[trail1.length - 1];
        const p2 = trail2[trail2.length - 1];
        const dist = Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2);
        statsLine.textContent = `Δ separation = ${dist.toFixed(4)}  (started at 1e-5)`;
      }
    } else {
      statsLine.textContent = `pts: ${trail1.length}  frame: ${frameCount}`;
    }
  }

  // ── Axis indicator ──
  function drawAxes() {
    // small axes in bottom-left
  }
  void drawAxes;

  // ── Init ──
  buildParams(currentKey);
  resetTrails();
  resize();

  cancelAnimationFrame(animId);
  animId = requestAnimationFrame(step);
})();
