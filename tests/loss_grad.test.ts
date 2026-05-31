/**
 * Loss function gradient correctness tests.
 *
 * For every ErrorFunction in Errors, verify that the analytic derivative
 * `der(output, target)` matches a central finite-difference of
 * `error(output, target)` with respect to `output`:
 *
 *   numDer ≈ (error(output+h, target) - error(output-h, target)) / (2h)
 *
 * Test points avoid known kinks:
 *   HINGE    – kink where output*target = 1; tested with output far from 1/target
 *   ABSOLUTE – kink at output = target; tested with output ≠ target
 *   HUBER    – kink at |output-target| = delta(=1); tested with |d|<1 and |d|>1
 *              but not at exactly 1.
 *   LOGLOSS  – smooth; output clamped to (eps, 1-eps) so avoid near ±1.
 */

import { Errors } from '../src/nn';

const H = 1e-5;
const TOL = 1e-4;

function numDerLoss(
  errorFn: (output: number, target: number) => number,
  output: number,
  target: number,
  h: number = H
): number {
  return (errorFn(output + h, target) - errorFn(output - h, target)) / (2 * h);
}

function checkLossGradient(
  name: string,
  errorFn: (output: number, target: number) => number,
  derFn: (output: number, target: number) => number,
  testCases: Array<{ output: number; target: number }>,
  tol: number = TOL
): void {
  describe(name, () => {
    for (const { output, target } of testCases) {
      test(`der(output=${output}, target=${target}) matches finite-difference`, () => {
        const analytic = derFn(output, target);
        const numeric = numDerLoss(errorFn, output, target);
        expect(Math.abs(analytic - numeric)).toBeLessThan(tol);
      });
    }
  });
}

describe('Loss function gradient checks (analytic vs finite-difference)', () => {

  // ── SQUARE: error = 0.5*(output-target)^2; der = output-target
  // Smooth everywhere; no kinks.
  checkLossGradient(
    'SQUARE',
    Errors.SQUARE.error,
    Errors.SQUARE.der,
    [
      { output: 0.5,  target: 0.0 },
      { output: -0.5, target: 0.0 },
      { output: 1.0,  target: -1.0 },
      { output: 0.0,  target: 1.0 },
      { output: 2.5,  target: 2.5 },  // zero gradient point
      { output: -3.0, target: 0.5 },
      { output: 0.0,  target: 0.0 },  // output == target, but square is smooth
    ]
  );

  // ── HINGE: error = max(0, 1 - output*target); kink at output*target = 1
  // Targets are ±1. Avoid output such that output*target = 1 exactly.
  // In the active region (output*target < 1): der = -target
  // In the inactive region (output*target > 1): der = 0
  // Note: at the kink the FD and analytic both yield 0 or -target depending on
  //       which side h lands, so we stay well away from the kink.
  checkLossGradient(
    'HINGE',
    Errors.HINGE.error,
    Errors.HINGE.der,
    [
      // target=+1, active region (output < 1)
      { output: -0.5, target: 1 },
      { output: 0.0,  target: 1 },
      { output: 0.5,  target: 1 },
      // target=+1, inactive region (output > 1), well past kink
      { output: 1.5,  target: 1 },
      { output: 2.0,  target: 1 },
      // target=-1, active region (output > -1)
      { output: 0.5,  target: -1 },
      { output: 0.0,  target: -1 },
      { output: -0.5, target: -1 },
      // target=-1, inactive region (output < -1), well past kink
      { output: -1.5, target: -1 },
      { output: -2.0, target: -1 },
    ]
  );

  // ── LOGLOSS: smooth but clipped to (eps, 1-eps) in probability space.
  // output maps to p=(output+1)/2; avoid output near ±1 (where p ≈ 0 or 1).
  // Use target ∈ {-1, 1} (playground convention).
  // Tolerance slightly relaxed at edges because of the eps clamp.
  checkLossGradient(
    'LOGLOSS',
    Errors.LOGLOSS.error,
    Errors.LOGLOSS.der,
    [
      // target = +1
      { output: -0.5, target: 1 },
      { output:  0.0, target: 1 },
      { output:  0.5, target: 1 },
      { output:  0.8, target: 1 },
      // target = -1
      { output: -0.8, target: -1 },
      { output: -0.5, target: -1 },
      { output:  0.0, target: -1 },
      { output:  0.5, target: -1 },
    ]
  );

  // ── HUBER (delta=1): smooth for |d|<1 and |d|>1, kink at |d|=1
  // Avoid output-target = ±1 exactly.
  checkLossGradient(
    'HUBER',
    Errors.HUBER.error,
    Errors.HUBER.der,
    [
      // |d| < 1 (quadratic region)
      { output:  0.0, target:  0.5 },
      { output:  0.5, target:  0.0 },
      { output: -0.3, target:  0.3 },
      { output:  0.8, target:  0.0 },
      { output:  0.0, target: -0.8 },
      // |d| > 1 (linear region), well away from kink
      { output:  2.0, target:  0.0 },
      { output: -2.0, target:  0.0 },
      { output:  0.0, target:  2.5 },
      { output:  3.0, target:  0.5 },
      { output: -3.0, target: -0.5 },
    ]
  );

  // ── ABSOLUTE: kink at output = target; avoid output == target
  checkLossGradient(
    'ABSOLUTE',
    Errors.ABSOLUTE.error,
    Errors.ABSOLUTE.der,
    [
      { output:  0.5,  target:  0.0 },
      { output: -0.5,  target:  0.0 },
      { output:  1.5,  target:  0.0 },
      { output: -1.5,  target:  0.0 },
      { output:  0.5,  target:  1.0 },
      { output:  1.5,  target:  1.0 },
      { output: -0.5,  target: -1.0 },
      { output: -1.5,  target: -1.0 },
      { output:  2.5,  target:  0.5 },
      { output: -2.5,  target: -0.5 },
    ]
  );

  // ── Sanity: confirm analytic values for well-known special cases ──────────────

  describe('SQUARE analytic value checks', () => {
    test('error(1, 0) == 0.5', () => {
      expect(Errors.SQUARE.error(1, 0)).toBeCloseTo(0.5);
    });
    test('der(1, 0) == 1', () => {
      expect(Errors.SQUARE.der(1, 0)).toBeCloseTo(1);
    });
    test('error(output, target) == 0 when output==target', () => {
      expect(Errors.SQUARE.error(3, 3)).toBeCloseTo(0);
    });
  });

  describe('HINGE analytic value checks', () => {
    test('error(0, 1) == 1 (fully in loss region)', () => {
      expect(Errors.HINGE.error(0, 1)).toBeCloseTo(1);
    });
    test('error(2, 1) == 0 (outside loss region)', () => {
      expect(Errors.HINGE.error(2, 1)).toBeCloseTo(0);
    });
    test('der(0.5, 1) == -1', () => {
      expect(Errors.HINGE.der(0.5, 1)).toBeCloseTo(-1);
    });
    test('der(1.5, 1) == 0', () => {
      expect(Errors.HINGE.der(1.5, 1)).toBeCloseTo(0);
    });
  });

  describe('HUBER analytic value checks', () => {
    test('error(0.5, 0) == 0.125 (quadratic region)', () => {
      expect(Errors.HUBER.error(0.5, 0)).toBeCloseTo(0.125);
    });
    test('error(2, 0) == 1.5 (linear region: 1*(2-0.5) = 1.5)', () => {
      expect(Errors.HUBER.error(2, 0)).toBeCloseTo(1.5);
    });
    test('der(0.5, 0) == 0.5 (quadratic region)', () => {
      expect(Errors.HUBER.der(0.5, 0)).toBeCloseTo(0.5);
    });
    test('der(2, 0) == 1 (linear region)', () => {
      expect(Errors.HUBER.der(2, 0)).toBeCloseTo(1);
    });
    test('der(-2, 0) == -1 (linear region, negative side)', () => {
      expect(Errors.HUBER.der(-2, 0)).toBeCloseTo(-1);
    });
  });

  describe('ABSOLUTE analytic value checks', () => {
    test('error(3, 1) == 2', () => {
      expect(Errors.ABSOLUTE.error(3, 1)).toBeCloseTo(2);
    });
    test('der(3, 1) == 1', () => {
      expect(Errors.ABSOLUTE.der(3, 1)).toBeCloseTo(1);
    });
    test('der(-1, 1) == -1', () => {
      expect(Errors.ABSOLUTE.der(-1, 1)).toBeCloseTo(-1);
    });
  });

  describe('LOGLOSS analytic value checks', () => {
    test('LOGLOSS error is non-negative', () => {
      const cases = [
        { output: 0.5, target: 1 },
        { output: -0.5, target: 1 },
        { output: 0.0, target: -1 },
      ];
      for (const { output, target } of cases) {
        expect(Errors.LOGLOSS.error(output, target)).toBeGreaterThanOrEqual(0);
      }
    });
    test('LOGLOSS is minimized when prediction matches target', () => {
      // output=1 (near perfect positive), target=1 → low loss
      const lossGood = Errors.LOGLOSS.error(0.9, 1);
      const lossBad = Errors.LOGLOSS.error(-0.9, 1);
      expect(lossGood).toBeLessThan(lossBad);
    });
  });

});
