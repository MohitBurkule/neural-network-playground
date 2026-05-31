/**
 * Tests for src/adversarial.ts: fgsm, pgd, predict.
 */

import {fgsm, pgd, predict} from '../src/adversarial';
import {buildNetwork, Activations, forwardProp} from '../src/nn';

/** Build a small trained-ish 2-input network with deterministic weights. */
function makeTinyNet() {
  const net = buildNetwork([2, 3, 1], Activations.TANH, Activations.LINEAR, ['x', 'y'], true);
  // Layer 1 weights
  net[1][0].inputLinks[0].weight = 0.5;
  net[1][0].inputLinks[1].weight = -0.3;
  net[1][0].bias = 0.1;
  net[1][1].inputLinks[0].weight = -0.4;
  net[1][1].inputLinks[1].weight = 0.6;
  net[1][1].bias = -0.1;
  net[1][2].inputLinks[0].weight = 0.3;
  net[1][2].inputLinks[1].weight = 0.3;
  net[1][2].bias = 0.0;
  // Layer 2 weights
  net[2][0].inputLinks[0].weight = 0.7;
  net[2][0].inputLinks[1].weight = -0.5;
  net[2][0].inputLinks[2].weight = 0.4;
  net[2][0].bias = 0.05;
  return net;
}

// ─── predict ─────────────────────────────────────────────────────────────────

describe('predict', () => {
  test('returns a finite number', () => {
    const net = makeTinyNet();
    const out = predict(net, 1.0, 2.0);
    expect(typeof out).toBe('number');
    expect(isFinite(out)).toBe(true);
  });

  test('agrees with direct forwardProp', () => {
    const net = makeTinyNet();
    const expected = forwardProp(net, [1.0, -0.5], null);
    const actual = predict(net, 1.0, -0.5);
    expect(actual).toBeCloseTo(expected);
  });
});

// ─── fgsm ─────────────────────────────────────────────────────────────────────

describe('fgsm', () => {
  test('returns object with x, y, label fields', () => {
    const net = makeTinyNet();
    const point = {x: 1.0, y: 0.5, label: 1};
    const result = fgsm(net, point, 0.3);
    expect(result).toHaveProperty('x');
    expect(result).toHaveProperty('y');
    expect(result).toHaveProperty('label');
  });

  test('preserves original label', () => {
    const net = makeTinyNet();
    const point = {x: 1.0, y: -1.0, label: -1};
    const result = fgsm(net, point, 0.5);
    expect(result.label).toBe(-1);
  });

  test('adversarial point stays within L∞ ball of radius epsilon', () => {
    const net = makeTinyNet();
    const epsilon = 0.4;
    const point = {x: 1.0, y: 0.5, label: 1};
    const result = fgsm(net, point, epsilon);
    expect(Math.abs(result.x - point.x)).toBeLessThanOrEqual(epsilon + 1e-9);
    expect(Math.abs(result.y - point.y)).toBeLessThanOrEqual(epsilon + 1e-9);
  });

  test('adversarial point differs from original (gradient is non-zero)', () => {
    const net = makeTinyNet();
    const point = {x: 1.0, y: -0.5, label: 1};
    const result = fgsm(net, point, 0.3);
    // At least one coordinate should change
    const moved = result.x !== point.x || result.y !== point.y;
    expect(moved).toBe(true);
  });

  test('epsilon=0 returns same coordinates', () => {
    const net = makeTinyNet();
    const point = {x: 1.0, y: 0.5, label: 1};
    const result = fgsm(net, point, 0);
    expect(result.x).toBeCloseTo(point.x);
    expect(result.y).toBeCloseTo(point.y);
  });

  test('larger epsilon produces larger displacement', () => {
    const net = makeTinyNet();
    const point = {x: 0.5, y: -0.5, label: 1};
    const r1 = fgsm(net, point, 0.1);
    const r2 = fgsm(net, point, 0.5);
    const d1 = Math.abs(r1.x - point.x) + Math.abs(r1.y - point.y);
    const d2 = Math.abs(r2.x - point.x) + Math.abs(r2.y - point.y);
    expect(d2).toBeGreaterThanOrEqual(d1);
  });
});

// ─── pgd ──────────────────────────────────────────────────────────────────────

describe('pgd', () => {
  test('returns object with x, y, label fields', () => {
    const net = makeTinyNet();
    const point = {x: 1.0, y: 0.5, label: 1};
    const result = pgd(net, point, 0.3, 5, 0.1);
    expect(result).toHaveProperty('x');
    expect(result).toHaveProperty('y');
    expect(result).toHaveProperty('label');
  });

  test('preserves original label', () => {
    const net = makeTinyNet();
    const point = {x: -1.0, y: 1.0, label: -1};
    const result = pgd(net, point, 0.5, 10, 0.1);
    expect(result.label).toBe(-1);
  });

  test('adversarial point stays within L∞ ball of radius epsilon', () => {
    const net = makeTinyNet();
    const epsilon = 0.5;
    const point = {x: 0.8, y: -0.3, label: 1};
    const result = pgd(net, point, epsilon, 20, 0.1);
    expect(Math.abs(result.x - point.x)).toBeLessThanOrEqual(epsilon + 1e-9);
    expect(Math.abs(result.y - point.y)).toBeLessThanOrEqual(epsilon + 1e-9);
  });

  test('L∞ constraint holds for many points and epsilons', () => {
    const net = makeTinyNet();
    const testCases = [
      {x: 1.0, y: 0.5, label: 1, eps: 0.1},
      {x: -2.0, y: 1.5, label: -1, eps: 0.3},
      {x: 0.0, y: 0.0, label: 1, eps: 0.5},
      {x: 3.0, y: -3.0, label: -1, eps: 0.2},
    ];
    for (const tc of testCases) {
      const result = pgd(net, tc, tc.eps, 10, tc.eps / 4);
      expect(Math.abs(result.x - tc.x)).toBeLessThanOrEqual(tc.eps + 1e-9);
      expect(Math.abs(result.y - tc.y)).toBeLessThanOrEqual(tc.eps + 1e-9);
    }
  });

  test('0 steps returns original coordinates', () => {
    const net = makeTinyNet();
    const point = {x: 1.0, y: 0.5, label: 1};
    const result = pgd(net, point, 0.5, 0, 0.1);
    expect(result.x).toBeCloseTo(point.x);
    expect(result.y).toBeCloseTo(point.y);
  });
});
