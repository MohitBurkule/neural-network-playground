// cnn_data.ts — tiny built-in image dataset
// 8x8 grayscale images of three shapes: circle, cross, square
// Values in [0, 1]. Labels: 0=circle, 1=cross, 2=square

export const IMAGE_SIZE = 8;
export const NUM_CLASSES = 3;
export const CLASS_NAMES = ['Circle', 'Cross', 'Square'];

export interface Example {
  pixels: number[];  // length IMAGE_SIZE*IMAGE_SIZE, row-major, [0,1]
  label: number;     // 0, 1, or 2
}

// --- shape generators ---

function makeBlank(): number[] {
  return new Array(IMAGE_SIZE * IMAGE_SIZE).fill(0);
}

function setPixel(img: number[], r: number, c: number, v: number) {
  if (r >= 0 && r < IMAGE_SIZE && c >= 0 && c < IMAGE_SIZE) {
    img[r * IMAGE_SIZE + c] = v;
  }
}

function addNoise(img: number[], seed: number, amount: number): number[] {
  // simple deterministic LCG noise
  let s = seed;
  return img.map(v => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    const n = ((s >>> 0) / 0xffffffff - 0.5) * 2 * amount;
    return Math.min(1, Math.max(0, v + n));
  });
}

// Circle: draw a ring of pixels around center
function makeCircle(cx: number, cy: number, r: number, seed: number): number[] {
  const img = makeBlank();
  for (let row = 0; row < IMAGE_SIZE; row++) {
    for (let col = 0; col < IMAGE_SIZE; col++) {
      const dist = Math.sqrt((row - cy) ** 2 + (col - cx) ** 2);
      if (Math.abs(dist - r) < 1.0) {
        setPixel(img, row, col, 1.0);
      }
    }
  }
  return addNoise(img, seed, 0.15);
}

// Cross: horizontal and vertical bars through center
function makeCross(cx: number, cy: number, seed: number): number[] {
  const img = makeBlank();
  const icx = Math.round(cx);
  const icy = Math.round(cy);
  for (let i = 1; i < IMAGE_SIZE - 1; i++) {
    setPixel(img, icy, i, 1.0);  // horizontal bar
    setPixel(img, i, icx, 1.0);  // vertical bar
  }
  return addNoise(img, seed, 0.15);
}

// Square: draw a hollow square outline
function makeSquare(x0: number, y0: number, sz: number, seed: number): number[] {
  const img = makeBlank();
  const x1 = x0 + sz - 1;
  const y1 = y0 + sz - 1;
  for (let i = x0; i <= x1; i++) {
    setPixel(img, y0, i, 1.0);
    setPixel(img, y1, i, 1.0);
  }
  for (let j = y0; j <= y1; j++) {
    setPixel(img, j, x0, 1.0);
    setPixel(img, j, x1, 1.0);
  }
  return addNoise(img, seed, 0.15);
}

// Generate a set of examples
function generateExamples(n: number): Example[] {
  const examples: Example[] = [];
  let seed = 42;

  for (let i = 0; i < n; i++) {
    seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xffffffff;
    const s = (seed >>> 0);

    // Circles: vary center slightly and radius
    const cx = 3 + (s % 3) - 1;
    const cy = 3 + ((s >> 4) % 3) - 1;
    const r = 2.0 + ((s >> 8) % 3) * 0.3;
    examples.push({ pixels: makeCircle(cx, cy, r, seed), label: 0 });

    seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xffffffff;
    const s2 = (seed >>> 0);
    const cx2 = 3 + (s2 % 3) - 1;
    const cy2 = 3 + ((s2 >> 4) % 3) - 1;
    examples.push({ pixels: makeCross(cx2, cy2, seed), label: 1 });

    seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xffffffff;
    const s3 = (seed >>> 0);
    const x0 = 1 + (s3 % 2);
    const y0 = 1 + ((s3 >> 3) % 2);
    const sz = 4 + ((s3 >> 6) % 2);
    examples.push({ pixels: makeSquare(x0, y0, sz, seed), label: 2 });
  }

  return examples;
}

const ALL_EXAMPLES = generateExamples(20);  // 60 total (20 each class)

// shuffle deterministically
function shuffle(arr: Example[], seed: number): Example[] {
  const a = arr.slice();
  let s = seed;
  for (let i = a.length - 1; i > 0; i--) {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    const j = (s >>> 0) % (i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const SHUFFLED = shuffle(ALL_EXAMPLES, 99);
const SPLIT = Math.floor(SHUFFLED.length * 0.75);

export function getTrainData(): Example[] {
  return SHUFFLED.slice(0, SPLIT);
}

export function getTestData(): Example[] {
  return SHUFFLED.slice(SPLIT);
}

export function getAllData(): Example[] {
  return SHUFFLED;
}
