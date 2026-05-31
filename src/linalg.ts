export {};

// ── Types ────────────────────────────────────────────────────────────────────
type Mat2 = [number, number, number, number]; // [a, b, c, d] => [[a,b],[c,d]]

// ── State ────────────────────────────────────────────────────────────────────
let matA: Mat2 = [1, 0, 0, 1];
let matB: Mat2 = [1, 0, 0, 1];
let composeMode = false;
let animating = false;
let showEigen = true;
let showDetArea = true;
let showSVD = false;
let animT = 0; // 0=identity, 1=fully transformed
let animDir = 1;
let animHandle = -1;
let lastTs = 0;
const ANIM_DUR = 1200; // ms

// ── DOM references ────────────────────────────────────────────────────────────
const svg = document.getElementById('plane') as unknown as SVGSVGElement;
const inputA = Array.from({length: 4}, (_, i) =>
  document.getElementById('ma' + i) as HTMLInputElement);
const inputB = Array.from({length: 4}, (_, i) =>
  document.getElementById('mb' + i) as HTMLInputElement);
const detReadout = document.getElementById('det-readout') as HTMLElement;
const eigenReadout = document.getElementById('eigen-readout') as HTMLElement;
const svdReadout = document.getElementById('svd-readout') as HTMLElement;
const composePanel = document.getElementById('compose-panel') as HTMLElement;

// ── SVG helpers ───────────────────────────────────────────────────────────────
const NS = 'http://www.w3.org/2000/svg';
function el(tag: string, attrs: Record<string, string|number> = {}, parent?: SVGElement): SVGElement {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (parent) parent.appendChild(e);
  return e;
}

// ── Canvas sizing ────────────────────────────────────────────────────────────
let W = 0, H = 0, CX = 0, CY = 0;
const GRID_CELLS = 8;
let SCALE = 50;

function resize() {
  const box = (svg as any).parentElement.getBoundingClientRect();
  W = box.width;
  H = box.height;
  (svg as any).setAttribute('width', W);
  (svg as any).setAttribute('height', H);
  CX = W / 2;
  CY = H / 2;
  SCALE = Math.min(W, H) / (GRID_CELLS * 2 + 2) * 1.4;
  render();
}

// Convert logical→screen
function sx(x: number, y: number, m: Mat2, t: number): number {
  const [a, b, , d] = lerp2(m, t);
  const c2 = lerp2(m, t)[2];
  return CX + (interp(1, a, t) * x + interp(0, b, t) * y) * SCALE;
}
function sy(x: number, y: number, m: Mat2, t: number): number {
  const lm = lerp2(m, t);
  return CY - (lm[2] * x + lm[3] * y) * SCALE;
}

// Interpolate identity→matrix
function lerp2(m: Mat2, t: number): Mat2 {
  return [
    interp(1, m[0], t),
    interp(0, m[1], t),
    interp(0, m[2], t),
    interp(1, m[3], t),
  ];
}

function interp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// ── Matrix math ───────────────────────────────────────────────────────────────
function det(m: Mat2): number { return m[0]*m[3] - m[1]*m[2]; }
function mul(a: Mat2, b: Mat2): Mat2 {
  return [
    a[0]*b[0] + a[1]*b[2],
    a[0]*b[1] + a[1]*b[3],
    a[2]*b[0] + a[3]*b[2],
    a[2]*b[1] + a[3]*b[3],
  ];
}

function eigenvalues(m: Mat2): {real: true; l1: number; l2: number} | {real: false} {
  const tr = m[0] + m[3];
  const d = det(m);
  const disc = tr*tr - 4*d;
  if (disc < 0) return {real: false};
  const sq = Math.sqrt(disc);
  return {real: true, l1: (tr + sq) / 2, l2: (tr - sq) / 2};
}

function eigenvector(m: Mat2, lam: number): [number, number] {
  // (A - λI)v = 0
  const a = m[0] - lam, b = m[1];
  const c = m[2], d = m[3] - lam;
  if (Math.abs(b) > 1e-9) return [b, lam - m[0]];
  if (Math.abs(c) > 1e-9) return [lam - m[3], c];
  if (Math.abs(a) > 1e-9) return [0, 1];
  return [1, 0];
}

function normalise(v: [number, number]): [number, number] {
  const len = Math.sqrt(v[0]*v[0] + v[1]*v[1]);
  if (len < 1e-12) return [1, 0];
  return [v[0]/len, v[1]/len];
}

// SVD of 2×2: returns {U, S, V} where S=[s1,s2]
function svd2(m: Mat2): {s1: number; s2: number; u1: [number,number]; u2: [number,number]; v1: [number,number]; v2: [number,number]} {
  // Bidiagonalise: compute A^T A then eigendecompose
  const ata: Mat2 = [
    m[0]*m[0]+m[2]*m[2], m[0]*m[1]+m[2]*m[3],
    m[0]*m[1]+m[2]*m[3], m[1]*m[1]+m[3]*m[3],
  ];
  const ev = eigenvalues(ata);
  let l1 = 1, l2 = 0;
  if (ev.real) { l1 = ev.l1; l2 = ev.l2; }
  const s1 = Math.sqrt(Math.max(0, l1));
  const s2 = Math.sqrt(Math.max(0, l2));
  const v1 = normalise(eigenvector(ata, l1));
  const v2: [number,number] = [-v1[1], v1[0]];
  // u = Av/s
  const u1raw: [number,number] = s1 > 1e-9 ?
    [(m[0]*v1[0]+m[1]*v1[1])/s1, (m[2]*v1[0]+m[3]*v1[1])/s1] : [1,0];
  const u2raw: [number,number] = s2 > 1e-9 ?
    [(m[0]*v2[0]+m[1]*v2[1])/s2, (m[2]*v2[0]+m[3]*v2[1])/s2] : [-u1raw[1], u1raw[0]];
  return {s1, s2, u1: normalise(u1raw), u2: normalise(u2raw), v1, v2};
}

// ── Render layers ─────────────────────────────────────────────────────────────
function render() {
  while ((svg as any).firstChild) (svg as any).removeChild((svg as any).firstChild);

  const effMat: Mat2 = composeMode ? mul(matB, matA) : matA;
  const t = animT;

  // Background
  el('rect', {x: 0, y: 0, width: W, height: H, fill: '#0d1117'}, svg as any);

  drawDeterminantArea(effMat, t);
  drawGrid(effMat, t);
  drawSampleShape(effMat, t);
  if (showSVD) drawSVDEllipse(effMat, t);
  drawAxes(effMat, t);
  drawBasisVectors(effMat, t);
  if (showEigen) drawEigenvectors(effMat, t);
  drawLabels();
  updateReadouts(effMat);
}

function drawGrid(m: Mat2, t: number) {
  const N = GRID_CELLS;
  const g = el('g', {opacity: '0.3'}, svg as any) as SVGGElement;

  for (let i = -N; i <= N; i++) {
    // vertical lines in logical space
    const x1 = sx(i, -N, m, t), y1 = sy(i, -N, m, t);
    const x2 = sx(i,  N, m, t), y2 = sy(i,  N, m, t);
    el('line', {x1, y1, x2, y2, stroke: i === 0 ? '#8b949e' : '#30363d', 'stroke-width': i===0?1.5:0.8}, g);

    // horizontal lines
    const x3 = sx(-N, i, m, t), y3 = sy(-N, i, m, t);
    const x4 = sx( N, i, m, t), y4 = sy( N, i, m, t);
    el('line', {x1: x3, y1: y3, x2: x4, y2: y4, stroke: i===0?'#8b949e':'#30363d', 'stroke-width': i===0?1.5:0.8}, g);
  }
}

function drawAxes(m: Mat2, t: number) {
  // standard axes (identity space)
  const g = el('g', {}, svg as any) as SVGGElement;
  const N = GRID_CELLS + 1;
  el('line', {
    x1: sx(-N, 0, m, t), y1: sy(-N, 0, m, t),
    x2: sx( N, 0, m, t), y2: sy( N, 0, m, t),
    stroke: '#8b949e', 'stroke-width': 1.5, opacity: 0.5,
  }, g);
  el('line', {
    x1: sx(0, -N, m, t), y1: sy(0, -N, m, t),
    x2: sx(0,  N, m, t), y2: sy(0,  N, m, t),
    stroke: '#8b949e', 'stroke-width': 1.5, opacity: 0.5,
  }, g);
}

function drawBasisVectors(m: Mat2, t: number) {
  const lm = lerp2(m, t);
  // î: red
  drawArrow(CX, CY, CX + lm[0]*SCALE, CY - lm[2]*SCALE, '#f85149', 'î', svg as any);
  // ĵ: green
  drawArrow(CX, CY, CX + lm[1]*SCALE, CY - lm[3]*SCALE, '#3fb950', 'ĵ', svg as any);
}

function drawArrow(x1: number, y1: number, x2: number, y2: number, color: string, label: string, parent: SVGElement) {
  const g = el('g', {}, parent) as SVGGElement;
  const dx = x2-x1, dy = y2-y1;
  const len = Math.sqrt(dx*dx+dy*dy);
  if (len < 1) return;
  const ux = dx/len, uy = dy/len;
  const hs = 10;
  const ax = x2 - ux*hs, ay = y2 - uy*hs;
  const px = -uy, py = ux;
  el('line', {x1, y1, x2: ax, y2: ay, stroke: color, 'stroke-width': 2.5, 'stroke-linecap': 'round'}, g);
  const pts = `${x2},${y2} ${ax+px*4},${ay+py*4} ${ax-px*4},${ay-py*4}`;
  el('polygon', {points: pts, fill: color}, g);
  el('text', {x: x2 + ux*14 + px*6, y: y2 + uy*14 + py*6,
    fill: color, 'font-size': 14, 'font-weight': 'bold',
    'text-anchor': 'middle', 'dominant-baseline': 'middle'}, g).textContent = label;
}

function drawDeterminantArea(m: Mat2, t: number) {
  if (!showDetArea) return;
  const lm = lerp2(m, t);
  const d = det(lm);
  const color = d >= 0 ? '#58a6ff' : '#f85149';
  const pts = [
    [CX, CY],
    [CX + lm[0]*SCALE, CY - lm[2]*SCALE],
    [CX + (lm[0]+lm[1])*SCALE, CY - (lm[2]+lm[3])*SCALE],
    [CX + lm[1]*SCALE, CY - lm[3]*SCALE],
  ].map(p => p.join(',')).join(' ');
  el('polygon', {points: pts, fill: color, opacity: 0.15, stroke: color, 'stroke-width': 1, 'stroke-opacity': 0.5}, svg as any);
}

const SHAPE_POINTS: Array<[number,number]> = [
  [1.5,0],[2,0.5],[2,1],[1.5,1.5],[1,2],[0,2],
  [-1,1.5],[-1.5,1],[-1.5,0],[-1,-0.5],[0,-1],[1,-0.5],
];

function drawSampleShape(m: Mat2, t: number) {
  const lm = lerp2(m, t);
  const tx = ([x,y]: [number,number]): [number,number] => [
    CX + (lm[0]*x + lm[1]*y)*SCALE,
    CY - (lm[2]*x + lm[3]*y)*SCALE,
  ];
  const pts = SHAPE_POINTS.map(p => tx(p).join(',')).join(' ');
  el('polygon', {points: pts, fill: 'none', stroke: '#e3b341', 'stroke-width': 1.5, opacity: 0.7, 'stroke-linejoin': 'round'}, svg as any);
  for (const p of SHAPE_POINTS) {
    const [px, py] = tx(p);
    el('circle', {cx: px, cy: py, r: 2.5, fill: '#e3b341', opacity: 0.7}, svg as any);
  }
}

function drawEigenvectors(m: Mat2, t: number) {
  const lm = lerp2(m, t);
  const ev = eigenvalues(lm);
  if (!ev.real) return;
  for (const [lam, color] of [[ev.l1, '#d2a8ff'], [ev.l2, '#79c0ff']] as Array<[number,string]>) {
    const v = normalise(eigenvector(lm, lam));
    const ext = GRID_CELLS * SCALE * 1.2;
    el('line', {
      x1: CX - v[0]*ext, y1: CY + v[1]*ext,
      x2: CX + v[0]*ext, y2: CY - v[1]*ext,
      stroke: color, 'stroke-width': 1.5, 'stroke-dasharray': '6 4', opacity: 0.8,
    }, svg as any);
    // arrow showing λ*v
    const scale = Math.min(Math.abs(lam), 4) * Math.sign(lam);
    drawArrow(CX, CY,
      CX + v[0]*scale*SCALE, CY - v[1]*scale*SCALE,
      color, `λ=${lam.toFixed(2)}`, svg as any);
  }
}

function drawSVDEllipse(m: Mat2, t: number) {
  const lm = lerp2(m, t);
  const {s1, s2, u1, u2, v1, v2} = svd2(lm);

  // Draw unit circle (dashed)
  el('circle', {cx: CX, cy: CY, r: SCALE, stroke: '#6e7681', 'stroke-width': 1,
    'stroke-dasharray': '4 4', fill: 'none', opacity: 0.6}, svg as any);

  // Draw ellipse via 60 sample points
  const pts: string[] = [];
  const N = 60;
  for (let i = 0; i <= N; i++) {
    const th = (i / N) * Math.PI * 2;
    // unit circle point: v1*cos(th) + v2*sin(th)
    const ux2 = v1[0]*Math.cos(th) + v2[0]*Math.sin(th);
    const uy2 = v1[1]*Math.cos(th) + v2[1]*Math.sin(th);
    // map through M: M*(ux2,uy2)
    const ex = lm[0]*ux2 + lm[1]*uy2;
    const ey = lm[2]*ux2 + lm[3]*uy2;
    pts.push(`${CX + ex*SCALE},${CY - ey*SCALE}`);
  }
  el('polyline', {points: pts.join(' '), stroke: '#f0883e', 'stroke-width': 2, fill: 'none', opacity: 0.85}, svg as any);

  // Semi-axes
  drawArrow(CX, CY, CX + u1[0]*s1*SCALE, CY - u1[1]*s1*SCALE, '#f0883e', `σ₁=${s1.toFixed(2)}`, svg as any);
  if (s2 > 0.01) {
    drawArrow(CX, CY, CX + u2[0]*s2*SCALE, CY - u2[1]*s2*SCALE, '#ffa657', `σ₂=${s2.toFixed(2)}`, svg as any);
  }
}

function drawLabels() {
  const g = el('g', {'font-size': '11', fill: '#8b949e'}, svg as any) as SVGGElement;
  // axis labels at edges
  el('text', {x: CX + GRID_CELLS*SCALE + 4, y: CY, 'dominant-baseline': 'middle'}, g).textContent = 'x';
  el('text', {x: CX + 4, y: CY - GRID_CELLS*SCALE - 4}, g).textContent = 'y';
}

function updateReadouts(m: Mat2) {
  const d = det(m);
  detReadout.textContent = `det = ${d.toFixed(3)}`;
  detReadout.style.color = d >= 0 ? '#58a6ff' : '#f85149';

  const ev = eigenvalues(m);
  if (ev.real) {
    eigenReadout.textContent = `λ₁ = ${ev.l1.toFixed(3)},  λ₂ = ${ev.l2.toFixed(3)}`;
    eigenReadout.style.color = '#d2a8ff';
  } else {
    const tr = m[0]+m[3];
    const disc = Math.abs((m[0]+m[3])*(m[0]+m[3]) - 4*det(m));
    const im = Math.sqrt(disc)/2;
    eigenReadout.textContent = `λ = ${(tr/2).toFixed(2)} ± ${im.toFixed(2)}i  (complex)`;
    eigenReadout.style.color = '#8b949e';
  }

  const {s1, s2} = svd2(m);
  svdReadout.textContent = `σ₁=${s1.toFixed(3)}, σ₂=${s2.toFixed(3)}  cond=${s2>0.001?(s1/s2).toFixed(2):'∞'}`;
}

// ── Animation loop ────────────────────────────────────────────────────────────
function startAnim() {
  if (animHandle !== -1) cancelAnimationFrame(animHandle);
  lastTs = 0;
  animating = true;
  animHandle = requestAnimationFrame(frame);
}

function frame(ts: number) {
  if (!lastTs) lastTs = ts;
  const dt = ts - lastTs; lastTs = ts;
  animT = Math.max(0, Math.min(1, animT + animDir * dt / ANIM_DUR));
  render();
  if (animT > 0 && animT < 1) {
    animHandle = requestAnimationFrame(frame);
  } else {
    animating = false;
    animHandle = -1;
  }
}

// ── Controls ──────────────────────────────────────────────────────────────────
function readMatrix(inputs: HTMLInputElement[]): Mat2 {
  return inputs.map(i => parseFloat(i.value) || 0) as unknown as Mat2;
}

function writeMatrix(inputs: HTMLInputElement[], m: Mat2) {
  inputs.forEach((inp, i) => { inp.value = String(m[i]); });
}

function applyMatrix() {
  matA = readMatrix(inputA);
  animT = 0; animDir = 1;
  startAnim();
}

function resetMatrix() {
  matA = [1,0,0,1];
  writeMatrix(inputA, matA);
  animT = 0; animDir = 1;
  startAnim();
}

function setPreset(name: string) {
  const presets: Record<string, Mat2> = {
    identity:    [1, 0, 0, 1],
    rotate90:    [0, -1, 1, 0],
    rotate45:    [Math.SQRT1_2, -Math.SQRT1_2, Math.SQRT1_2, Math.SQRT1_2],
    scale2:      [2, 0, 0, 2],
    scalexy:     [2, 0, 0, 0.5],
    shearx:      [1, 1, 0, 1],
    sheary:      [1, 0, 1, 1],
    reflectx:    [1, 0, 0, -1],
    reflecty:    [-1, 0, 0, 1],
    reflectdiag: [0, 1, 1, 0],
    projectx:    [1, 0, 0, 0],
    rotscale:    [1.5*Math.SQRT1_2, -1.5*Math.SQRT1_2, 1.5*Math.SQRT1_2, 1.5*Math.SQRT1_2],
    squeeze:     [2, 0, 0, 0.5],
    singular:    [1, 2, 2, 4],
  };
  const p = presets[name];
  if (!p) return;
  matA = p;
  writeMatrix(inputA, matA);
  animT = 0; animDir = 1;
  startAnim();
}

// ── Init ──────────────────────────────────────────────────────────────────────
function init() {
  writeMatrix(inputA, matA);
  writeMatrix(inputB, matB);

  // Matrix A inputs
  inputA.forEach(inp => inp.addEventListener('input', () => {
    matA = readMatrix(inputA); render();
  }));
  inputB.forEach(inp => inp.addEventListener('input', () => {
    matB = readMatrix(inputB); render();
  }));

  (document.getElementById('btn-apply') as HTMLElement).addEventListener('click', applyMatrix);
  (document.getElementById('btn-reset') as HTMLElement).addEventListener('click', resetMatrix);

  (document.getElementById('btn-animate') as HTMLElement).addEventListener('click', () => {
    matA = readMatrix(inputA);
    if (animT >= 1) { animT = 1; animDir = -1; }
    else if (animT <= 0) { animT = 0; animDir = 1; }
    else { animDir = -animDir; }
    startAnim();
  });

  (document.getElementById('btn-compose') as HTMLElement).addEventListener('click', () => {
    composeMode = !composeMode;
    composePanel.style.display = composeMode ? 'block' : 'none';
    (document.getElementById('btn-compose') as HTMLElement).classList.toggle('active', composeMode);
    render();
  });

  (document.getElementById('btn-eigen') as HTMLElement).addEventListener('click', () => {
    showEigen = !showEigen;
    (document.getElementById('btn-eigen') as HTMLElement).classList.toggle('active', showEigen);
    render();
  });

  (document.getElementById('btn-det') as HTMLElement).addEventListener('click', () => {
    showDetArea = !showDetArea;
    (document.getElementById('btn-det') as HTMLElement).classList.toggle('active', showDetArea);
    render();
  });

  (document.getElementById('btn-svd') as HTMLElement).addEventListener('click', () => {
    showSVD = !showSVD;
    (document.getElementById('btn-svd') as HTMLElement).classList.toggle('active', showSVD);
    render();
  });

  // Preset selector
  (document.getElementById('preset-select') as HTMLSelectElement).addEventListener('change', (e) => {
    const v = (e.target as HTMLSelectElement).value;
    if (v) setPreset(v);
  });

  // Draggable basis vector tips
  setupDrag();

  window.addEventListener('resize', resize);
  resize();
}

// ── Draggable vector tips ─────────────────────────────────────────────────────
function setupDrag() {
  let dragging: 'i'|'j'|null = null;

  (svg as any).addEventListener('mousedown', (e: MouseEvent) => {
    const lm = lerp2(matA, animT);
    const tipI = [CX + lm[0]*SCALE, CY - lm[2]*SCALE];
    const tipJ = [CX + lm[1]*SCALE, CY - lm[3]*SCALE];
    const mx = e.offsetX, my = e.offsetY;
    if (dist(mx, my, tipI[0], tipI[1]) < 16) dragging = 'i';
    else if (dist(mx, my, tipJ[0], tipJ[1]) < 16) dragging = 'j';
  });

  (svg as any).addEventListener('mousemove', (e: MouseEvent) => {
    if (!dragging) return;
    const lx = (e.offsetX - CX) / SCALE;
    const ly = -(e.offsetY - CY) / SCALE;
    if (dragging === 'i') { matA[0] = +lx.toFixed(2); matA[2] = +ly.toFixed(2); }
    else                  { matA[1] = +lx.toFixed(2); matA[3] = +ly.toFixed(2); }
    writeMatrix(inputA, matA);
    animT = 1;
    render();
  });

  (svg as any).addEventListener('mouseup', () => { dragging = null; });
  (svg as any).addEventListener('mouseleave', () => { dragging = null; });
}

function dist(x1: number, y1: number, x2: number, y2: number): number {
  return Math.sqrt((x1-x2)**2 + (y1-y2)**2);
}

document.addEventListener('DOMContentLoaded', init);
