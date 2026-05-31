/**
 * Tests for src/dataset3d.ts 3D dataset generators.
 */

import {
  classifyTwoGaussBlobs,
  classifyConcentricSpheres,
  classifyHelix,
  classifySwissRoll,
  Example3D,
} from '../src/dataset3d';

const GENERATORS = [
  {name: 'classifyTwoGaussBlobs', fn: classifyTwoGaussBlobs},
  {name: 'classifyConcentricSpheres', fn: classifyConcentricSpheres},
  {name: 'classifyHelix', fn: classifyHelix},
  {name: 'classifySwissRoll', fn: classifySwissRoll},
];

// ─── Common property checks for every generator ───────────────────────────────

describe.each(GENERATORS)('$name — common properties', ({name, fn}) => {
  const N = 100;
  const noise = 0.0;
  let samples: Example3D[];

  beforeAll(() => {
    samples = fn(N, noise);
  });

  test('returns correct sample count', () => {
    expect(samples.length).toBe(N);
  });

  test('labels are in {-1, 1}', () => {
    for (const s of samples) {
      expect(s.label === 1 || s.label === -1).toBe(true);
    }
  });

  test('coordinates are finite numbers', () => {
    for (const s of samples) {
      expect(isFinite(s.x)).toBe(true);
      expect(isFinite(s.y)).toBe(true);
      expect(isFinite(s.z)).toBe(true);
    }
  });

  test('both label classes are represented', () => {
    const labels = new Set(samples.map(s => s.label));
    expect(labels.size).toBe(2);
    expect(labels.has(1)).toBe(true);
    expect(labels.has(-1)).toBe(true);
  });
});

// ─── classifyTwoGaussBlobs — specific checks ──────────────────────────────────

describe('classifyTwoGaussBlobs', () => {
  test('roughly half positive, half negative (±20%) at 200 samples', () => {
    const samples = classifyTwoGaussBlobs(200, 0);
    const pos = samples.filter(s => s.label === 1).length;
    expect(pos).toBeGreaterThan(60);  // at least 30%
    expect(pos).toBeLessThan(140);    // at most 70%
  });

  test('coordinates are plausibly near ±2 centroid', () => {
    const samples = classifyTwoGaussBlobs(500, 0);
    const posMean = samples.filter(s => s.label === 1)
      .reduce((acc, s) => ({x: acc.x + s.x, y: acc.y + s.y, z: acc.z + s.z}),
              {x: 0, y: 0, z: 0});
    const posCount = samples.filter(s => s.label === 1).length;
    // mean of positive class should be near (-2, -2, -2)
    expect(posMean.x / posCount).toBeLessThan(0);
    expect(posMean.y / posCount).toBeLessThan(0);
    expect(posMean.z / posCount).toBeLessThan(0);
  });
});

// ─── classifyConcentricSpheres — specific checks ──────────────────────────────

describe('classifyConcentricSpheres', () => {
  test('inner-sphere points (label=1) have smaller radius than outer (label=-1)', () => {
    const samples = classifyConcentricSpheres(200, 0);
    const radii1 = samples.filter(s => s.label === 1)
      .map(s => Math.sqrt(s.x*s.x + s.y*s.y + s.z*s.z));
    const radiim1 = samples.filter(s => s.label === -1)
      .map(s => Math.sqrt(s.x*s.x + s.y*s.y + s.z*s.z));
    const mean1 = radii1.reduce((a, b) => a + b, 0) / radii1.length;
    const meanm1 = radiim1.reduce((a, b) => a + b, 0) / radiim1.length;
    expect(mean1).toBeLessThan(meanm1);
  });
});

// ─── classifyHelix — specific checks ─────────────────────────────────────────

describe('classifyHelix', () => {
  test('produces expected number of samples', () => {
    expect(classifyHelix(50, 0).length).toBe(50);
    expect(classifyHelix(51, 0).length).toBe(51);
  });

  test('coordinates are bounded (helix radius ~2, no noise)', () => {
    const samples = classifyHelix(100, 0);
    for (const s of samples) {
      // With r=2 and no noise, x and z should be within ~2
      expect(Math.abs(s.x)).toBeLessThanOrEqual(3);
      expect(Math.abs(s.z)).toBeLessThanOrEqual(3);
    }
  });
});

// ─── classifySwissRoll — specific checks ─────────────────────────────────────

describe('classifySwissRoll', () => {
  test('no coordinate exceeds extreme bounds', () => {
    const samples = classifySwissRoll(200, 0);
    for (const s of samples) {
      // Swiss roll is scaled by /5, t in [1.5pi, 4.5pi], so max ~4pi/5 ≈ 2.5
      expect(Math.abs(s.x)).toBeLessThan(20);
      expect(Math.abs(s.y)).toBeLessThan(10);
      expect(Math.abs(s.z)).toBeLessThan(20);
    }
  });
});

// ─── Reproducibility with seedrandom (if available) ──────────────────────────

describe('Generator output variety', () => {
  test('calling twice with same N gives same structure (array shape)', () => {
    const a = classifyTwoGaussBlobs(50, 0.1);
    const b = classifyTwoGaussBlobs(50, 0.1);
    expect(a.length).toBe(b.length);
    // Both should have same label sequence (labels are deterministic in structure)
    // For TwoGaussBlobs: first half label=1, second half label=-1
    for (let i = 0; i < a.length; i++) {
      expect(a[i].label).toBe(b[i].label);
    }
  });

  test('noise=0 and noise=0.5 give different coordinate values', () => {
    // With Math.random already seeded, we just check they're not identical in value
    // (statistically almost certain to differ)
    const noNoise = classifyTwoGaussBlobs(10, 0);
    const withNoise = classifyTwoGaussBlobs(10, 0.5);
    // At least one coordinate differs (variance is different)
    const allSame = noNoise.every((s, i) =>
      s.x === withNoise[i].x && s.y === withNoise[i].y && s.z === withNoise[i].z);
    expect(allSame).toBe(false);
  });
});
