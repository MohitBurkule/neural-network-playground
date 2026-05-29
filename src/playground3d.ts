/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// THREE is loaded as a global via <script> tag in the HTML page.
declare const THREE: any;

import {
  buildNetwork, forwardProp, backProp, updateWeights,
  Activations, Errors, Node
} from './nn';
import {
  Example3D, DataGenerator3D,
  classifyTwoGaussBlobs, classifyConcentricSpheres,
  classifyHelix, classifySwissRoll
} from './dataset3d';

// ─── State ───────────────────────────────────────────────────────────────────

type DatasetKey = 'blobs' | 'spheres' | 'helix' | 'swissroll';

const DATASETS: Record<DatasetKey, DataGenerator3D> = {
  blobs: classifyTwoGaussBlobs,
  spheres: classifyConcentricSpheres,
  helix: classifyHelix,
  swissroll: classifySwissRoll
};

let currentDataset: DatasetKey = 'blobs';
let numSamples = 200;
let noise = 0.1;
let learningRate = 0.03;
let numHidden = 6;
let useSquaredFeatures = true;
let isTraining = false;
let epoch = 0;
let totalLoss = 0;

let data: Example3D[] = [];
let network: Node[][] = [];

// ─── Three.js setup ──────────────────────────────────────────────────────────

const container = document.getElementById('canvas-container') as HTMLElement;
const WIDTH = container.clientWidth || 600;
const HEIGHT = container.clientHeight || 500;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(WIDTH, HEIGHT);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(50, WIDTH / HEIGHT, 0.1, 100);
camera.position.set(0, 0, 14);

// Ambient + directional light
const ambient = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(5, 10, 7);
scene.add(dirLight);

// Grid helper
const gridHelper = new THREE.GridHelper(12, 12, 0x333355, 0x222244);
scene.add(gridHelper);

// ─── Simple mouse-drag rotation ──────────────────────────────────────────────

const rotGroup = new THREE.Group();
scene.add(rotGroup);

let isDragging = false;
let prevMouse = { x: 0, y: 0 };
let rotX = 0.3;
let rotY = 0.5;

function applyRotation() {
  rotGroup.rotation.x = rotX;
  rotGroup.rotation.y = rotY;
}
applyRotation();

const canvas = renderer.domElement;
canvas.addEventListener('mousedown', (e: MouseEvent) => {
  isDragging = true;
  prevMouse = { x: e.clientX, y: e.clientY };
});
window.addEventListener('mouseup', () => { isDragging = false; });
window.addEventListener('mousemove', (e: MouseEvent) => {
  if (!isDragging) return;
  const dx = e.clientX - prevMouse.x;
  const dy = e.clientY - prevMouse.y;
  rotY += dx * 0.01;
  rotX += dy * 0.01;
  applyRotation();
  prevMouse = { x: e.clientX, y: e.clientY };
});

// Touch support
canvas.addEventListener('touchstart', (e: TouchEvent) => {
  isDragging = true;
  prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
});
window.addEventListener('touchend', () => { isDragging = false; });
window.addEventListener('touchmove', (e: TouchEvent) => {
  if (!isDragging) return;
  const dx = e.touches[0].clientX - prevMouse.x;
  const dy = e.touches[0].clientY - prevMouse.y;
  rotY += dx * 0.01;
  rotX += dy * 0.01;
  applyRotation();
  prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
});

// Scroll to zoom
canvas.addEventListener('wheel', (e: WheelEvent) => {
  camera.position.z = Math.max(4, Math.min(30, camera.position.z + e.deltaY * 0.02));
  e.preventDefault();
}, { passive: false });

// ─── Scene objects ───────────────────────────────────────────────────────────

let dataPointsMesh: any = null;
let boundaryMesh: any = null;

const POS_COLOR = new THREE.Color(0xff4d6d);  // pink/red
const NEG_COLOR = new THREE.Color(0x4d9fff);  // blue

function buildDataPoints() {
  if (dataPointsMesh) {
    rotGroup.remove(dataPointsMesh);
    dataPointsMesh.geometry.dispose();
    dataPointsMesh.material.dispose();
    dataPointsMesh = null;
  }

  const positions: number[] = [];
  const colors: number[] = [];

  for (const pt of data) {
    positions.push(pt.x, pt.y, pt.z);
    const c = pt.label > 0 ? POS_COLOR : NEG_COLOR;
    colors.push(c.r, c.g, c.b);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.18,
    vertexColors: true,
    sizeAttenuation: true
  });

  dataPointsMesh = new THREE.Points(geo, mat);
  rotGroup.add(dataPointsMesh);
}

// Coarse 3D grid for decision boundary visualization
const GRID_RES = 14;  // 14^3 = 2744 points — fast enough

function buildBoundaryPoints() {
  if (boundaryMesh) {
    rotGroup.remove(boundaryMesh);
    boundaryMesh.geometry.dispose();
    boundaryMesh.material.dispose();
    boundaryMesh = null;
  }

  const positions: number[] = [];
  const colors: number[] = [];

  const range = 4.5;
  const step = (2 * range) / (GRID_RES - 1);

  for (let ix = 0; ix < GRID_RES; ix++) {
    for (let iy = 0; iy < GRID_RES; iy++) {
      for (let iz = 0; iz < GRID_RES; iz++) {
        const x = -range + ix * step;
        const y = -range + iy * step;
        const z = -range + iz * step;

        const inputs = buildInputs(x, y, z);
        const out = forwardProp(network, inputs, null as any);
        // confidence: 0 = boundary, 1 = sure
        const conf = Math.abs(out);

        // Only render near decision boundary (conf < 0.6) to avoid clutter
        if (conf < 0.6) {
          positions.push(x, y, z);
          const t = (out + 1) / 2;  // 0..1
          const r = POS_COLOR.r * t + NEG_COLOR.r * (1 - t);
          const g = POS_COLOR.g * t + NEG_COLOR.g * (1 - t);
          const b = POS_COLOR.b * t + NEG_COLOR.b * (1 - t);
          colors.push(r, g, b);
        }
      }
    }
  }

  if (positions.length === 0) return;

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.30,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.45,
    depthWrite: false
  });

  boundaryMesh = new THREE.Points(geo, mat);
  rotGroup.add(boundaryMesh);
}

// ─── Neural network helpers ──────────────────────────────────────────────────

function buildInputs(x: number, y: number, z: number): number[] {
  const inputs: number[] = [x, y, z];
  if (useSquaredFeatures) {
    inputs.push(x * x, y * y, z * z);
  }
  return inputs;
}

function inputIds(): string[] {
  const ids = ['x', 'y', 'z'];
  if (useSquaredFeatures) {
    ids.push('x2', 'y2', 'z2');
  }
  return ids;
}

function rebuildNetwork() {
  const numInputs = useSquaredFeatures ? 6 : 3;
  const shape = [numInputs, numHidden, numHidden, 1];
  network = buildNetwork(shape, Activations.TANH, Activations.TANH, inputIds());
}

function computeLoss(): number {
  let loss = 0;
  for (const pt of data) {
    const inputs = buildInputs(pt.x, pt.y, pt.z);
    const out = forwardProp(network, inputs, null as any);
    loss += Errors.SQUARE.error(out, pt.label);
  }
  return loss / data.length;
}

// ─── Training step ───────────────────────────────────────────────────────────

let boundaryUpdateCounter = 0;
const BOUNDARY_UPDATE_EVERY = 20;

function trainStep() {
  // One full epoch of SGD
  for (const pt of data) {
    const inputs = buildInputs(pt.x, pt.y, pt.z);
    forwardProp(network, inputs, null as any);
    backProp(network, pt.label, Errors.SQUARE);
    updateWeights(network, learningRate, null as any, 0);
  }
  epoch++;
  totalLoss = computeLoss();

  // Update UI
  const lossEl = document.getElementById('loss-value');
  if (lossEl) lossEl.textContent = totalLoss.toFixed(4);
  const epochEl = document.getElementById('epoch-value');
  if (epochEl) epochEl.textContent = String(epoch);

  // Rebuild boundary every N epochs
  boundaryUpdateCounter++;
  if (boundaryUpdateCounter >= BOUNDARY_UPDATE_EVERY) {
    buildBoundaryPoints();
    boundaryUpdateCounter = 0;
  }
}

// ─── Animation loop ──────────────────────────────────────────────────────────

function animate() {
  requestAnimationFrame(animate);
  if (isTraining) {
    trainStep();
  }
  renderer.render(scene, camera);
}

// ─── Reset / init ─────────────────────────────────────────────────────────────

function reset() {
  epoch = 0;
  totalLoss = 0;
  boundaryUpdateCounter = 0;

  data = DATASETS[currentDataset](numSamples, noise);
  rebuildNetwork();
  buildDataPoints();
  buildBoundaryPoints();

  const lossEl = document.getElementById('loss-value');
  if (lossEl) lossEl.textContent = '—';
  const epochEl = document.getElementById('epoch-value');
  if (epochEl) epochEl.textContent = '0';
}

// ─── Control wiring ──────────────────────────────────────────────────────────

function wireControls() {
  const btnPlay = document.getElementById('btn-play') as HTMLButtonElement;
  const btnPause = document.getElementById('btn-pause') as HTMLButtonElement;
  const btnReset = document.getElementById('btn-reset') as HTMLButtonElement;
  const selDataset = document.getElementById('sel-dataset') as HTMLSelectElement;
  const sliderLR = document.getElementById('slider-lr') as HTMLInputElement;
  const sliderNoise = document.getElementById('slider-noise') as HTMLInputElement;
  const sliderHidden = document.getElementById('slider-hidden') as HTMLInputElement;
  const chkSquared = document.getElementById('chk-squared') as HTMLInputElement;
  const lblLR = document.getElementById('lbl-lr') as HTMLElement;
  const lblNoise = document.getElementById('lbl-noise') as HTMLElement;
  const lblHidden = document.getElementById('lbl-hidden') as HTMLElement;

  btnPlay.addEventListener('click', () => {
    isTraining = true;
    btnPlay.disabled = true;
    btnPause.disabled = false;
  });

  btnPause.addEventListener('click', () => {
    isTraining = false;
    btnPlay.disabled = false;
    btnPause.disabled = true;
  });

  btnReset.addEventListener('click', () => {
    isTraining = false;
    btnPlay.disabled = false;
    btnPause.disabled = true;
    reset();
  });

  selDataset.addEventListener('change', () => {
    currentDataset = selDataset.value as DatasetKey;
    reset();
  });

  sliderLR.addEventListener('input', () => {
    learningRate = parseFloat(sliderLR.value);
    lblLR.textContent = learningRate.toFixed(3);
  });

  sliderNoise.addEventListener('input', () => {
    noise = parseFloat(sliderNoise.value);
    lblNoise.textContent = noise.toFixed(2);
    reset();
  });

  sliderHidden.addEventListener('input', () => {
    numHidden = parseInt(sliderHidden.value, 10);
    lblHidden.textContent = String(numHidden);
    rebuildNetwork();
    buildBoundaryPoints();
    epoch = 0;
  });

  chkSquared.addEventListener('change', () => {
    useSquaredFeatures = chkSquared.checked;
    rebuildNetwork();
    buildBoundaryPoints();
    epoch = 0;
  });
}

// ─── Resize ───────────────────────────────────────────────────────────────────

window.addEventListener('resize', () => {
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});

// ─── Boot ─────────────────────────────────────────────────────────────────────

wireControls();
reset();
animate();
