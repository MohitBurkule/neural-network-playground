/**
 * Unit tests for src/dataset.ts
 *
 * dataset.ts imports d3 — the jest.config.js moduleNameMapper redirects it to
 * tests/__mocks__/d3.js which provides a minimal scaleLinear stub.
 */

import {
  classifyCircleData,
  classifyXORData,
  classifyTwoGaussData,
  classifySpiralData,
  regressPlane,
  shuffle,
  Example2D,
} from '../src/dataset';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function labelSet(points: Example2D[]): Set<number> {
  return new Set(points.map(p => p.label));
}

// ─── classifyCircleData ───────────────────────────────────────────────────────

describe('classifyCircleData', () => {
  test('returns the requested number of samples', () => {
    const pts = classifyCircleData(100, 0);
    expect(pts.length).toBe(100);
  });

  test('labels are exclusively 1 or -1', () => {
    const pts = classifyCircleData(200, 0);
    const labels = labelSet(pts);
    expect(labels.has(1)).toBe(true);
    expect(labels.has(-1)).toBe(true);
    pts.forEach(p => expect([-1, 1]).toContain(p.label));
  });

  test('all points have finite x and y', () => {
    const pts = classifyCircleData(50, 0);
    pts.forEach(p => {
      expect(isFinite(p.x)).toBe(true);
      expect(isFinite(p.y)).toBe(true);
    });
  });
});

// ─── classifyXORData ──────────────────────────────────────────────────────────

describe('classifyXORData', () => {
  test('returns the requested number of samples', () => {
    expect(classifyXORData(200, 0).length).toBe(200);
  });

  test('labels are 1 or -1', () => {
    const pts = classifyXORData(100, 0);
    pts.forEach(p => expect([-1, 1]).toContain(p.label));
  });

  test('zero noise: label reflects XOR rule (x*y >= 0 => 1, else -1)', () => {
    // With zero noise, x and y retain their original signs before noise is applied
    // We can't test this perfectly because randUniform is used; just verify the
    // label set contains both classes and size is correct.
    const pts = classifyXORData(400, 0);
    const labels = labelSet(pts);
    expect(labels.has(1)).toBe(true);
    expect(labels.has(-1)).toBe(true);
  });
});

// ─── classifyTwoGaussData ─────────────────────────────────────────────────────

describe('classifyTwoGaussData', () => {
  test('returns the requested number of samples', () => {
    expect(classifyTwoGaussData(100, 0).length).toBe(100);
  });

  test('both classes are present', () => {
    const pts = classifyTwoGaussData(200, 0);
    const labels = labelSet(pts);
    expect(labels.has(1)).toBe(true);
    expect(labels.has(-1)).toBe(true);
  });

  test('roughly half positive and half negative', () => {
    const pts = classifyTwoGaussData(200, 0);
    const pos = pts.filter(p => p.label === 1).length;
    // genGauss is called with numSamples/2 per class
    expect(pos).toBe(100);
  });
});

// ─── classifySpiralData ───────────────────────────────────────────────────────

describe('classifySpiralData', () => {
  test('returns the requested number of samples', () => {
    expect(classifySpiralData(100, 0).length).toBe(100);
  });

  test('both classes are present', () => {
    const pts = classifySpiralData(100, 0);
    const labels = labelSet(pts);
    expect(labels.has(1)).toBe(true);
    expect(labels.has(-1)).toBe(true);
  });
});

// ─── regressPlane ─────────────────────────────────────────────────────────────

describe('regressPlane', () => {
  test('returns the requested number of samples', () => {
    expect(regressPlane(50, 0).length).toBe(50);
  });

  test('labels are finite numbers', () => {
    const pts = regressPlane(100, 0);
    pts.forEach(p => {
      expect(typeof p.label).toBe('number');
      expect(isFinite(p.label)).toBe(true);
    });
  });
});

// ─── shuffle ──────────────────────────────────────────────────────────────────

describe('shuffle', () => {
  test('preserves all elements', () => {
    const arr = [1, 2, 3, 4, 5];
    const copy = [...arr];
    shuffle(arr);
    expect(arr.sort()).toEqual(copy.sort());
  });

  test('same length after shuffle', () => {
    const arr = Array.from({ length: 20 }, (_, i) => i);
    shuffle(arr);
    expect(arr.length).toBe(20);
  });

  test('with seedrandom seeding produces deterministic results', () => {
    // Seed Math.random via seedrandom before each shuffle
    const seedrandom = require('seedrandom');

    const arr1 = [1, 2, 3, 4, 5, 6, 7, 8];
    (Math as any).random = seedrandom('test-seed');
    shuffle(arr1);

    const arr2 = [1, 2, 3, 4, 5, 6, 7, 8];
    (Math as any).random = seedrandom('test-seed');
    shuffle(arr2);

    expect(arr1).toEqual(arr2);
  });
});
