export {};

// ── Image Convolution Kernel Playground ──────────────────────────────────────

// ── Types ────────────────────────────────────────────────────────────────────

interface KernelPreset {
  name: string;
  size: 3 | 5;
  values: number[];
  divisor: number;
  offset: number;
}

interface AppState {
  kernelSize: 3 | 5;
  kernelValues: number[];
  divisor: number;
  offset: number;
  imageIndex: number;
  grayscale: boolean;
  showWindow: boolean;
  animating: boolean;
  animFrame: number | null;
  animX: number;
  animY: number;
  animDirX: number;
  animDirY: number;
  uploadedImageData: ImageData | null;
  usingUpload: boolean;
}

// ── Kernel presets ───────────────────────────────────────────────────────────

const PRESETS_3: KernelPreset[] = [
  {
    name: "Identity",
    size: 3,
    values: [0, 0, 0, 0, 1, 0, 0, 0, 0],
    divisor: 1,
    offset: 0,
  },
  {
    name: "Box Blur",
    size: 3,
    values: [1, 1, 1, 1, 1, 1, 1, 1, 1],
    divisor: 9,
    offset: 0,
  },
  {
    name: "Gaussian Blur",
    size: 3,
    values: [1, 2, 1, 2, 4, 2, 1, 2, 1],
    divisor: 16,
    offset: 0,
  },
  {
    name: "Sharpen",
    size: 3,
    values: [0, -1, 0, -1, 5, -1, 0, -1, 0],
    divisor: 1,
    offset: 0,
  },
  {
    name: "Edge Detect (Laplacian)",
    size: 3,
    values: [0, 1, 0, 1, -4, 1, 0, 1, 0],
    divisor: 1,
    offset: 128,
  },
  {
    name: "Sobel X",
    size: 3,
    values: [-1, 0, 1, -2, 0, 2, -1, 0, 1],
    divisor: 1,
    offset: 128,
  },
  {
    name: "Sobel Y",
    size: 3,
    values: [-1, -2, -1, 0, 0, 0, 1, 2, 1],
    divisor: 1,
    offset: 128,
  },
  {
    name: "Emboss",
    size: 3,
    values: [-2, -1, 0, -1, 1, 1, 0, 1, 2],
    divisor: 1,
    offset: 128,
  },
  {
    name: "Outline",
    size: 3,
    values: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
    divisor: 1,
    offset: 0,
  },
  {
    name: "Motion Blur",
    size: 3,
    values: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    divisor: 3,
    offset: 0,
  },
];

const PRESETS_5: KernelPreset[] = [
  {
    name: "Identity (5x5)",
    size: 5,
    values: [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0,
    ],
    divisor: 1,
    offset: 0,
  },
  {
    name: "Box Blur (5x5)",
    size: 5,
    values: new Array(25).fill(1),
    divisor: 25,
    offset: 0,
  },
  {
    name: "Gaussian Blur (5x5)",
    size: 5,
    values: [
      1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1,
      4, 6, 4, 1,
    ],
    divisor: 256,
    offset: 0,
  },
  {
    name: "Edge Detect (5x5)",
    size: 5,
    values: [
      0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0,
      0, 0, -1, 0, 0,
    ],
    divisor: 1,
    offset: 128,
  },
  {
    name: "Unsharp Mask (5x5)",
    size: 5,
    values: [
      -1, -4, -6, -4, -1, -4, -16, -24, -16, -4, -6, -24, 476, -24, -6, -4,
      -16, -24, -16, -4, -1, -4, -6, -4, -1,
    ],
    divisor: 256,
    offset: 0,
  },
];

// ── Built-in image generators ────────────────────────────────────────────────

const IMG_SIZE = 128;

function makeGradient(): ImageData {
  const d = new ImageData(IMG_SIZE, IMG_SIZE);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const i = (y * IMG_SIZE + x) * 4;
      d.data[i] = Math.round((x / (IMG_SIZE - 1)) * 255);
      d.data[i + 1] = Math.round((y / (IMG_SIZE - 1)) * 255);
      d.data[i + 2] = 128;
      d.data[i + 3] = 255;
    }
  }
  return d;
}

function makeCheckerboard(): ImageData {
  const d = new ImageData(IMG_SIZE, IMG_SIZE);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const i = (y * IMG_SIZE + x) * 4;
      const v = (Math.floor(x / 8) + Math.floor(y / 8)) % 2 === 0 ? 240 : 20;
      d.data[i] = v;
      d.data[i + 1] = v;
      d.data[i + 2] = v;
      d.data[i + 3] = 255;
    }
  }
  return d;
}

function makeShapes(): ImageData {
  const d = new ImageData(IMG_SIZE, IMG_SIZE);
  // fill dark
  for (let i = 0; i < IMG_SIZE * IMG_SIZE * 4; i += 4) {
    d.data[i] = 20;
    d.data[i + 1] = 20;
    d.data[i + 2] = 30;
    d.data[i + 3] = 255;
  }
  const setP = (x: number, y: number, r: number, g: number, b: number) => {
    if (x < 0 || x >= IMG_SIZE || y < 0 || y >= IMG_SIZE) return;
    const i = (y * IMG_SIZE + x) * 4;
    d.data[i] = r;
    d.data[i + 1] = g;
    d.data[i + 2] = b;
    d.data[i + 3] = 255;
  };
  // White filled rectangle
  for (let y = 10; y < 40; y++)
    for (let x = 10; x < 55; x++) setP(x, y, 220, 220, 220);
  // Colored circle
  const cx = 88,
    cy = 55,
    cr = 28;
  for (let y = cy - cr; y <= cy + cr; y++) {
    for (let x = cx - cr; x <= cx + cr; x++) {
      if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= cr * cr)
        setP(x, y, 80, 160, 240);
    }
  }
  // Triangle outline
  for (let t = 0; t <= 100; t++) {
    const f = t / 100;
    setP(Math.round(30 + f * 40), Math.round(110 - f * 50), 240, 180, 60);
    setP(Math.round(70 - f * 40), Math.round(60 + f * 50), 240, 180, 60);
    setP(Math.round(30 + f * 0), Math.round(110 + f * 0), 240, 180, 60);
  }
  // Diagonal lines
  for (let i = 0; i < IMG_SIZE; i++) {
    setP(i, Math.floor((i * 70) / IMG_SIZE) + 80, 200, 80, 200);
  }
  return d;
}

function makeTextLike(): ImageData {
  const d = new ImageData(IMG_SIZE, IMG_SIZE);
  for (let i = 0; i < IMG_SIZE * IMG_SIZE * 4; i += 4) {
    d.data[i] = 245;
    d.data[i + 1] = 245;
    d.data[i + 2] = 245;
    d.data[i + 3] = 255;
  }
  // Draw horizontal text-like lines
  const setP = (x: number, y: number, v: number) => {
    if (x < 0 || x >= IMG_SIZE || y < 0 || y >= IMG_SIZE) return;
    const i = (y * IMG_SIZE + x) * 4;
    d.data[i] = v;
    d.data[i + 1] = v;
    d.data[i + 2] = v;
    d.data[i + 3] = 255;
  };
  const rows = [10, 20, 30, 42, 54, 66, 78, 90, 102, 114];
  for (const row of rows) {
    let x = 6;
    while (x < IMG_SIZE - 6) {
      const len = 4 + Math.floor(((x * 31 + row * 17) % 7) * 1.5);
      for (let dx = 0; dx < len && x + dx < IMG_SIZE - 6; dx++) {
        setP(x + dx, row, 30);
        setP(x + dx, row + 1, 30);
      }
      x += len + 2 + ((x + row) % 4);
    }
  }
  return d;
}

function makeSyntheticPhoto(): ImageData {
  const d = new ImageData(IMG_SIZE, IMG_SIZE);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const i = (y * IMG_SIZE + x) * 4;
      // Sky gradient top half
      if (y < 60) {
        d.data[i] = Math.round(50 + (y / 60) * 80);
        d.data[i + 1] = Math.round(100 + (y / 60) * 60);
        d.data[i + 2] = Math.round(200 - (y / 60) * 40);
      } else {
        // Ground
        d.data[i] = Math.round(60 + ((y - 60) / 68) * 30);
        d.data[i + 1] = Math.round(90 + ((y - 60) / 68) * 20);
        d.data[i + 2] = 40;
      }
      d.data[i + 3] = 255;
    }
  }
  // Sun
  const sx = 90,
    sy = 20,
    sr = 14;
  for (let y = sy - sr; y <= sy + sr; y++) {
    for (let x = sx - sr; x <= sx + sr; x++) {
      if (
        x >= 0 &&
        x < IMG_SIZE &&
        y >= 0 &&
        y < IMG_SIZE &&
        (x - sx) * (x - sx) + (y - sy) * (y - sy) <= sr * sr
      ) {
        const i = (y * IMG_SIZE + x) * 4;
        d.data[i] = 255;
        d.data[i + 1] = 230;
        d.data[i + 2] = 80;
      }
    }
  }
  // House outline
  for (let y = 75; y < 115; y++) {
    for (let x = 30; x < 70; x++) {
      const i = (y * IMG_SIZE + x) * 4;
      if (y === 75 || y === 114 || x === 30 || x === 69) {
        d.data[i] = 180;
        d.data[i + 1] = 140;
        d.data[i + 2] = 90;
      } else {
        d.data[i] = 220;
        d.data[i + 1] = 190;
        d.data[i + 2] = 140;
      }
    }
  }
  // Roof
  for (let t = 0; t <= 40; t++) {
    const x1 = 30 - t,
      x2 = 69 + t,
      y = 74 - t;
    const mx = 49;
    if (y >= 45) {
      for (let x = Math.min(x1, mx); x <= Math.max(x2, mx); x++) {
        if (
          x >= x1 &&
          x <= x2 &&
          x >= 0 &&
          x < IMG_SIZE &&
          y >= 0 &&
          y < IMG_SIZE
        ) {
          const xi =
            x >= x1 && x <= mx ? x1 + ((x - x1) * 2) / 1 : x2;
          const i = (y * IMG_SIZE + x) * 4;
          d.data[i] = 180;
          d.data[i + 1] = 60;
          d.data[i + 2] = 60;
        }
      }
    }
  }
  // Simple triangular roof fill
  for (let t = 0; t < 30; t++) {
    const y = 74 - t;
    const xl = 30 - t,
      xr = 69 + t;
    for (let x = xl; x <= xr; x++) {
      if (x >= 0 && x < IMG_SIZE && y >= 0 && y < IMG_SIZE) {
        const i = (y * IMG_SIZE + x) * 4;
        if (d.data[i + 3] === 255 && d.data[i] > 100) {
          d.data[i] = 190;
          d.data[i + 1] = 70;
          d.data[i + 2] = 70;
        }
      }
    }
  }
  return d;
}

const IMAGE_GENERATORS: { name: string; fn: () => ImageData }[] = [
  { name: "Gradient", fn: makeGradient },
  { name: "Checkerboard", fn: makeCheckerboard },
  { name: "Shapes", fn: makeShapes },
  { name: "Text-like", fn: makeTextLike },
  { name: "Synthetic Photo", fn: makeSyntheticPhoto },
];

// ── Convolution ──────────────────────────────────────────────────────────────

function applyKernel(
  src: ImageData,
  kernel: number[],
  kSize: number,
  divisor: number,
  offset: number,
  grayscale: boolean
): ImageData {
  const w = src.width,
    h = src.height;
  const out = new ImageData(w, h);
  const half = Math.floor(kSize / 2);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0,
        g = 0,
        b = 0;
      for (let ky = 0; ky < kSize; ky++) {
        for (let kx = 0; kx < kSize; kx++) {
          const sy = Math.min(h - 1, Math.max(0, y + ky - half));
          const sx = Math.min(w - 1, Math.max(0, x + kx - half));
          const si = (sy * w + sx) * 4;
          const kv = kernel[ky * kSize + kx];
          if (grayscale) {
            const gray =
              0.299 * src.data[si] +
              0.587 * src.data[si + 1] +
              0.114 * src.data[si + 2];
            r += gray * kv;
            g += gray * kv;
            b += gray * kv;
          } else {
            r += src.data[si] * kv;
            g += src.data[si + 1] * kv;
            b += src.data[si + 2] * kv;
          }
        }
      }
      const div = divisor === 0 ? 1 : divisor;
      const oi = (y * w + x) * 4;
      out.data[oi] = Math.min(255, Math.max(0, Math.round(r / div + offset)));
      out.data[oi + 1] = Math.min(
        255,
        Math.max(0, Math.round(g / div + offset))
      );
      out.data[oi + 2] = Math.min(
        255,
        Math.max(0, Math.round(b / div + offset))
      );
      out.data[oi + 3] = 255;
    }
  }
  return out;
}

// Compute a single output pixel value (for animation display)
function computePixel(
  src: ImageData,
  px: number,
  py: number,
  kernel: number[],
  kSize: number,
  divisor: number,
  offset: number,
  grayscale: boolean
): { r: number; g: number; b: number; products: number[] } {
  const w = src.width,
    h = src.height;
  const half = Math.floor(kSize / 2);
  let r = 0,
    g = 0,
    b = 0;
  const products: number[] = [];
  for (let ky = 0; ky < kSize; ky++) {
    for (let kx = 0; kx < kSize; kx++) {
      const sy = Math.min(h - 1, Math.max(0, py + ky - half));
      const sx = Math.min(w - 1, Math.max(0, px + kx - half));
      const si = (sy * w + sx) * 4;
      const kv = kernel[ky * kSize + kx];
      if (grayscale) {
        const gray =
          0.299 * src.data[si] +
          0.587 * src.data[si + 1] +
          0.114 * src.data[si + 2];
        r += gray * kv;
        g += gray * kv;
        b += gray * kv;
        products.push(Math.round(gray * kv));
      } else {
        const rv = src.data[si];
        r += rv * kv;
        products.push(Math.round(rv * kv));
        g += src.data[si + 1] * kv;
        b += src.data[si + 2] * kv;
      }
    }
  }
  const div = divisor === 0 ? 1 : divisor;
  return {
    r: Math.min(255, Math.max(0, Math.round(r / div + offset))),
    g: Math.min(255, Math.max(0, Math.round(g / div + offset))),
    b: Math.min(255, Math.max(0, Math.round(b / div + offset))),
    products,
  };
}

// ── Main app ─────────────────────────────────────────────────────────────────

function main() {
  // ── State ────────────────────────────────────────────────────────────────
  const state: AppState = {
    kernelSize: 3,
    kernelValues: [...PRESETS_3[0].values],
    divisor: PRESETS_3[0].divisor,
    offset: PRESETS_3[0].offset,
    imageIndex: 0,
    grayscale: false,
    showWindow: true,
    animating: true,
    animFrame: null,
    animX: 30,
    animY: 30,
    animDirX: 1,
    animDirY: 1,
    uploadedImageData: null,
    usingUpload: false,
  };

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const inputCanvas = document.getElementById("inputCanvas") as HTMLCanvasElement;
  const outputCanvas = document.getElementById("outputCanvas") as HTMLCanvasElement;
  const inCtx = inputCanvas.getContext("2d")!;
  const outCtx = outputCanvas.getContext("2d")!;

  const kernelGridEl = document.getElementById("kernelGrid") as HTMLDivElement;
  const divisorInput = document.getElementById("divisorInput") as HTMLInputElement;
  const offsetInput = document.getElementById("offsetInput") as HTMLInputElement;
  const presetSelect = document.getElementById("presetSelect") as HTMLSelectElement;
  const imageSel = document.getElementById("imageSel") as HTMLSelectElement;
  const grayscaleCb = document.getElementById("grayscaleCb") as HTMLInputElement;
  const showWindowCb = document.getElementById("showWindowCb") as HTMLInputElement;
  const animateCb = document.getElementById("animateCb") as HTMLInputElement;
  const kernelSizeRadios = document.querySelectorAll<HTMLInputElement>(
    'input[name="ksize"]'
  );
  const fileInput = document.getElementById("fileInput") as HTMLInputElement;
  const pixelInfoEl = document.getElementById("pixelInfo") as HTMLDivElement;
  const productsEl = document.getElementById("productsDisplay") as HTMLDivElement;

  // ── Canvas sizing ─────────────────────────────────────────────────────────
  inputCanvas.width = IMG_SIZE;
  inputCanvas.height = IMG_SIZE;
  outputCanvas.width = IMG_SIZE;
  outputCanvas.height = IMG_SIZE;

  // Cache generated images
  const imageCache: (ImageData | null)[] = new Array(IMAGE_GENERATORS.length).fill(null);
  let outputImageData: ImageData | null = null;

  function getSourceImage(): ImageData {
    if (state.usingUpload && state.uploadedImageData) {
      return state.uploadedImageData;
    }
    const idx = state.imageIndex;
    if (!imageCache[idx]) {
      imageCache[idx] = IMAGE_GENERATORS[idx].fn();
    }
    return imageCache[idx]!;
  }

  // ── Kernel grid rendering ─────────────────────────────────────────────────
  let kernelInputEls: HTMLInputElement[] = [];

  function buildKernelGrid() {
    kernelGridEl.innerHTML = "";
    kernelGridEl.style.gridTemplateColumns = `repeat(${state.kernelSize}, 1fr)`;
    kernelInputEls = [];
    for (let i = 0; i < state.kernelSize * state.kernelSize; i++) {
      const inp = document.createElement("input");
      inp.type = "number";
      inp.className = "kernel-cell";
      inp.value = String(state.kernelValues[i] ?? 0);
      inp.step = "0.01";
      inp.addEventListener("input", () => {
        state.kernelValues[i] = parseFloat(inp.value) || 0;
        renderOutput();
      });
      kernelGridEl.appendChild(inp);
      kernelInputEls.push(inp);
    }
  }

  function syncKernelInputs() {
    for (let i = 0; i < kernelInputEls.length; i++) {
      kernelInputEls[i].value = String(state.kernelValues[i] ?? 0);
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────
  function renderInput() {
    const src = getSourceImage();
    inCtx.putImageData(src, 0, 0);
  }

  function renderOutput() {
    const src = getSourceImage();
    outputImageData = applyKernel(
      src,
      state.kernelValues,
      state.kernelSize,
      state.divisor,
      state.offset,
      state.grayscale
    );
    outCtx.putImageData(outputImageData, 0, 0);
    if (state.showWindow) {
      drawSlideWindow();
    }
  }

  // ── Sliding window overlay ────────────────────────────────────────────────
  function drawSlideWindow() {
    const src = getSourceImage();
    const px = state.animX;
    const py = state.animY;
    const half = Math.floor(state.kernelSize / 2);
    const scale = inputCanvas.clientWidth / IMG_SIZE || 1;

    // Draw on input: highlight neighborhood
    inCtx.putImageData(src, 0, 0);

    // Neighborhood highlight on input
    const nx = (px - half) * scale;
    const ny = (py - half) * scale;
    const nw = state.kernelSize * scale;

    // Use physical pixels
    inCtx.save();
    inCtx.strokeStyle = "rgba(255, 220, 50, 0.9)";
    inCtx.lineWidth = 2;
    inCtx.strokeRect(px - half - 0.5, py - half - 0.5, state.kernelSize, state.kernelSize);
    // Highlight center pixel
    inCtx.strokeStyle = "rgba(255,80,80,0.9)";
    inCtx.lineWidth = 1.5;
    inCtx.strokeRect(px - 0.5, py - 0.5, 1, 1);
    inCtx.restore();

    // Draw on output: highlight computed pixel
    if (outputImageData) {
      outCtx.putImageData(outputImageData, 0, 0);
    }
    outCtx.save();
    outCtx.strokeStyle = "rgba(255, 80, 80, 0.9)";
    outCtx.lineWidth = 2;
    outCtx.strokeRect(px - 0.5, py - 0.5, 1, 1);
    outCtx.restore();

    // Pixel info
    const { r, g, b, products } = computePixel(
      src,
      px,
      py,
      state.kernelValues,
      state.kernelSize,
      state.divisor,
      state.offset,
      state.grayscale
    );

    pixelInfoEl.innerHTML =
      `<span class="info-label">Pixel (${px}, ${py})</span> ` +
      `&rarr; <span class="out-val">R:${r} G:${g} B:${b}</span>`;

    // Products display
    const kSize = state.kernelSize;
    let html = `<div class="prod-grid" style="grid-template-columns:repeat(${kSize},1fr)">`;
    for (let i = 0; i < products.length; i++) {
      const v = products[i];
      const intensity = Math.min(1, Math.abs(v) / 255);
      const col = v >= 0 ? `rgba(80,200,120,${0.3 + intensity * 0.7})` : `rgba(220,80,80,${0.3 + intensity * 0.7})`;
      html += `<div class="prod-cell" style="background:${col}">${v > 999 ? "…" : v}</div>`;
    }
    html += `</div>`;
    html += `<div class="prod-sum">Sum = ${products.reduce((a, b) => a + b, 0)} &divide; ${state.divisor} + ${state.offset} &rarr; <b>${r}</b></div>`;
    productsEl.innerHTML = html;
  }

  // ── Animation loop ────────────────────────────────────────────────────────
  let lastAnimTime = 0;

  function animStep(ts: number) {
    if (!state.animating || !state.showWindow) {
      state.animFrame = null;
      return;
    }
    const elapsed = ts - lastAnimTime;
    if (elapsed > 60) {
      lastAnimTime = ts;
      state.animX += state.animDirX;
      state.animY += state.animDirY;
      const half = Math.floor(state.kernelSize / 2);
      if (state.animX <= half || state.animX >= IMG_SIZE - half - 1) {
        state.animDirX *= -1;
        state.animX += state.animDirX * 2;
      }
      if (state.animY <= half || state.animY >= IMG_SIZE - half - 1) {
        state.animDirY *= -1;
        state.animY += state.animDirY * 2;
      }
      drawSlideWindow();
    }
    state.animFrame = requestAnimationFrame(animStep);
  }

  function startAnim() {
    if (state.animFrame !== null) return;
    state.animFrame = requestAnimationFrame(animStep);
  }

  function stopAnim() {
    if (state.animFrame !== null) {
      cancelAnimationFrame(state.animFrame);
      state.animFrame = null;
    }
  }

  // ── Preset population ─────────────────────────────────────────────────────
  function populatePresets() {
    presetSelect.innerHTML = "";
    const presets = state.kernelSize === 3 ? PRESETS_3 : PRESETS_5;
    presets.forEach((p, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = p.name;
      presetSelect.appendChild(opt);
    });
  }

  function applyPreset(idx: number) {
    const presets = state.kernelSize === 3 ? PRESETS_3 : PRESETS_5;
    const p = presets[idx];
    if (!p) return;
    state.kernelValues = [...p.values];
    state.divisor = p.divisor;
    state.offset = p.offset;
    divisorInput.value = String(p.divisor);
    offsetInput.value = String(p.offset);
    syncKernelInputs();
  }

  // ── Event listeners ───────────────────────────────────────────────────────
  presetSelect.addEventListener("change", () => {
    applyPreset(parseInt(presetSelect.value));
    renderOutput();
  });

  imageSel.addEventListener("change", () => {
    state.imageIndex = parseInt(imageSel.value);
    state.usingUpload = false;
    renderInput();
    renderOutput();
  });

  grayscaleCb.addEventListener("change", () => {
    state.grayscale = grayscaleCb.checked;
    renderOutput();
  });

  showWindowCb.addEventListener("change", () => {
    state.showWindow = showWindowCb.checked;
    if (state.showWindow) {
      renderOutput();
      if (state.animating) startAnim();
    } else {
      stopAnim();
      renderInput();
      renderOutput();
      pixelInfoEl.innerHTML = "";
      productsEl.innerHTML = "";
    }
  });

  animateCb.addEventListener("change", () => {
    state.animating = animateCb.checked;
    if (state.animating && state.showWindow) {
      startAnim();
    } else {
      stopAnim();
    }
  });

  divisorInput.addEventListener("input", () => {
    state.divisor = parseFloat(divisorInput.value) || 1;
    renderOutput();
  });

  offsetInput.addEventListener("input", () => {
    state.offset = parseFloat(offsetInput.value) || 0;
    renderOutput();
  });

  kernelSizeRadios.forEach((r) => {
    r.addEventListener("change", () => {
      if (r.checked) {
        state.kernelSize = parseInt(r.value) as 3 | 5;
        populatePresets();
        applyPreset(0);
        buildKernelGrid();
        renderOutput();
      }
    });
  });

  // Canvas click to position window
  inputCanvas.addEventListener("click", (e) => {
    const rect = inputCanvas.getBoundingClientRect();
    const scaleX = IMG_SIZE / rect.width;
    const scaleY = IMG_SIZE / rect.height;
    const px = Math.round((e.clientX - rect.left) * scaleX);
    const py = Math.round((e.clientY - rect.top) * scaleY);
    const half = Math.floor(state.kernelSize / 2);
    state.animX = Math.min(IMG_SIZE - half - 1, Math.max(half, px));
    state.animY = Math.min(IMG_SIZE - half - 1, Math.max(half, py));
    if (!state.showWindow) {
      showWindowCb.checked = true;
      state.showWindow = true;
    }
    drawSlideWindow();
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const tmp = document.createElement("canvas");
      tmp.width = IMG_SIZE;
      tmp.height = IMG_SIZE;
      const tc = tmp.getContext("2d")!;
      tc.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);
      state.uploadedImageData = tc.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
      state.usingUpload = true;
      URL.revokeObjectURL(url);
      renderInput();
      renderOutput();
    };
    img.src = url;
  });

  // ── Init ──────────────────────────────────────────────────────────────────
  populatePresets();
  buildKernelGrid();
  divisorInput.value = String(state.divisor);
  offsetInput.value = String(state.offset);
  renderInput();
  renderOutput();
  if (state.animating && state.showWindow) startAnim();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", main);
} else {
  main();
}
