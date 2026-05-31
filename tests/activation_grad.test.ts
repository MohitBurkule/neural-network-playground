/**
 * Activation gradient correctness tests.
 *
 * For every activation function in Activations, verify that the analytic
 * derivative `der(x)` matches a central finite-difference of `output(x)`:
 *
 *   numDer ≈ (output(x+h) - output(x-h)) / (2h)
 *
 * Test points are chosen away from known kinks so that the derivative
 * is smooth and self-consistent at every tested point.
 *
 * (Earlier formula bugs in MISH.der and SINC.output were found by these checks
 * and have since been fixed in nn.ts; MISH and SINC now pass the FD check for
 * both signs.)
 *
 * Non-differentiable / kink notes (skipped points documented):
 *   RELU, LEAKY_RELU, ELU, SELU, EXPONENTIAL_LINEAR – kink at x=0
 *   HARD_SIGMOID – kinks at x=±2.5
 *   HARD_TANH    – kinks at x=±1
 *   HARD_SWISH   – kinks at x=±3
 *   RELU6        – kinks at x=0 and x=6
 */

import { Activations } from '../src/nn';

// Step size for central finite difference
const H = 1e-5;
// Absolute tolerance for derivative agreement
const TOL = 1e-4;

/**
 * Central finite-difference approximation of d/dx f(x).
 */
function numDer(f: (x: number) => number, x: number, h: number = H): number {
  return (f(x + h) - f(x - h)) / (2 * h);
}

/**
 * Assert that analytic and numeric derivatives agree at several x values.
 * `testPoints` should avoid kinks.
 */
function checkGradient(
  name: string,
  output: (x: number) => number,
  der: (x: number) => number,
  testPoints: number[],
  tol: number = TOL
): void {
  describe(name, () => {
    for (const x of testPoints) {
      test(`der(${x}) matches finite-difference`, () => {
        const analytic = der(x);
        const numeric = numDer(output, x);
        expect(Math.abs(analytic - numeric)).toBeLessThan(tol);
      });
    }
  });
}

// ─── Tests for every activation ─────────────────────────────────────────────

describe('Activation gradient checks (analytic vs finite-difference)', () => {

  // TANH – smooth everywhere
  checkGradient(
    'TANH',
    Activations.TANH.output,
    Activations.TANH.der,
    [-2, -1, -0.5, 0, 0.5, 1, 2]
  );

  // RELU – kink at x=0; skip x=0
  checkGradient(
    'RELU',
    Activations.RELU.output,
    Activations.RELU.der,
    [-2, -1, -0.5, 0.5, 1, 2]
  );

  // SIGMOID – smooth everywhere
  checkGradient(
    'SIGMOID',
    Activations.SIGMOID.output,
    Activations.SIGMOID.der,
    [-3, -1, -0.5, 0, 0.5, 1, 3]
  );

  // LINEAR – trivially smooth; der = 1 everywhere
  checkGradient(
    'LINEAR',
    Activations.LINEAR.output,
    Activations.LINEAR.der,
    [-5, -1, 0, 1, 5]
  );

  // SINE – smooth everywhere
  checkGradient(
    'SINE',
    Activations.SINE.output,
    Activations.SINE.der,
    [-Math.PI, -1, -0.5, 0, 0.5, 1, Math.PI]
  );

  // SINC – sinc(x) = sin(x)/x, even function, sinc(0)=1. The small-value guard
  // uses x*x < threshold so it is symmetric in x (fixed from an earlier unsigned
  // `x < threshold` bug). Derivative matches FD for both positive and negative x.
  describe('SINC (even function; checks both signs)', () => {
    const points = [-3, -2.5, -2, -1.5, -1, -0.5, -0.1, 0.1, 0.5, 1, 1.5, 2, 2.5, 3];
    for (const x of points) {
      test(`SINC der(${x}) matches finite-difference`, () => {
        const analytic = Activations.SINC.der(x);
        const numeric = numDer(Activations.SINC.output, x);
        expect(Math.abs(analytic - numeric)).toBeLessThan(TOL);
      });
    }
    test('SINC is even: output(-1) === output(1) === sin(1)', () => {
      expect(Activations.SINC.output(-1)).toBeCloseTo(Math.sin(1), 6);
      expect(Activations.SINC.output(1)).toBeCloseTo(Math.sin(1), 6);
    });
  });

  // MISH – mish(x) = x * tanh(softplus(x)); der = tanh(sp) + x*sig(x)*(1-tanh²(sp)).
  // The analytic derivative matches the finite-difference of the output.
  describe('MISH (output + analytic derivative)', () => {
    const points = [-2, -1, -0.5, 0, 0.5, 1, 2];
    for (const x of points) {
      test(`MISH der(${x}) matches finite-difference`, () => {
        const analytic = Activations.MISH.der(x);
        const numeric = numDer(Activations.MISH.output, x);
        expect(Math.abs(analytic - numeric)).toBeLessThan(TOL);
      });
    }

    test('MISH output(0) is 0', () => {
      expect(Activations.MISH.output(0)).toBeCloseTo(0);
    });
    test('MISH output(1) matches x*tanh(softplus(x))', () => {
      const x = 1;
      const sp = Math.log(1 + Math.exp(x));
      const expected = x * Math.tanh(sp);
      expect(Activations.MISH.output(x)).toBeCloseTo(expected, 6);
    });
    test('MISH.der(0) is ≈0.6 (correct derivative)', () => {
      expect(Activations.MISH.der(0)).toBeCloseTo(0.6, 2);
    });
  });

  // GELU – smooth everywhere
  checkGradient(
    'GELU',
    Activations.GELU.output,
    Activations.GELU.der,
    [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
  );

  // LEAKY_RELU – kink at x=0; skip x=0
  checkGradient(
    'LEAKY_RELU',
    Activations.LEAKY_RELU.output,
    Activations.LEAKY_RELU.der,
    [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
  );

  // ELU – kink at x=0; skip x=0
  checkGradient(
    'ELU',
    Activations.ELU.output,
    Activations.ELU.der,
    [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
  );

  // SELU – kink at x=0; skip x=0
  checkGradient(
    'SELU',
    Activations.SELU.output,
    Activations.SELU.der,
    [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
  );

  // SWISH – smooth everywhere
  checkGradient(
    'SWISH',
    Activations.SWISH.output,
    Activations.SWISH.der,
    [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
  );

  // SOFTPLUS – smooth everywhere
  checkGradient(
    'SOFTPLUS',
    Activations.SOFTPLUS.output,
    Activations.SOFTPLUS.der,
    [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
  );

  // SOFTSIGN – smooth everywhere
  checkGradient(
    'SOFTSIGN',
    Activations.SOFTSIGN.output,
    Activations.SOFTSIGN.der,
    [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
  );

  // HARD_SIGMOID – kinks at x=−2.5 and x=2.5; test inside (−2.5,2.5) and well outside
  // Inside: der=0.2; outside: der=0. Both are flat regions (FD ≈ analytic).
  checkGradient(
    'HARD_SIGMOID',
    Activations.HARD_SIGMOID.output,
    Activations.HARD_SIGMOID.der,
    [-2, -1, 0, 1, 2, -4, 4]  // avoid ±2.5 exactly
  );

  // HARD_TANH – kinks at x=−1 and x=1; test in (−1,1) and far outside
  checkGradient(
    'HARD_TANH',
    Activations.HARD_TANH.output,
    Activations.HARD_TANH.der,
    [-0.5, 0, 0.5, -3, 3]  // avoid ±1 exactly
  );

  // HARD_SWISH – kinks at x=−3 and x=3; test inside (−3,3) and far outside
  checkGradient(
    'HARD_SWISH',
    Activations.HARD_SWISH.output,
    Activations.HARD_SWISH.der,
    [-2, -1, 0, 1, 2, -5, 5]  // avoid ±3 exactly
  );

  // RELU6 – kinks at x=0 and x=6; test inside (0,6) and flat regions
  checkGradient(
    'RELU6',
    Activations.RELU6.output,
    Activations.RELU6.der,
    [0.5, 1, 2, 3, 4, 5, 5.5, -1, 7]  // avoid 0 and 6 exactly
  );

  // BENT_IDENTITY – smooth everywhere
  checkGradient(
    'BENT_IDENTITY',
    Activations.BENT_IDENTITY.output,
    Activations.BENT_IDENTITY.der,
    [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
  );

  // GAUSSIAN – smooth everywhere
  checkGradient(
    'GAUSSIAN',
    Activations.GAUSSIAN.output,
    Activations.GAUSSIAN.der,
    [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
  );

  // SNAKE – smooth everywhere
  checkGradient(
    'SNAKE',
    Activations.SNAKE.output,
    Activations.SNAKE.der,
    [-Math.PI, -2, -1, -0.5, 0, 0.5, 1, 2, Math.PI]
  );

  // ARCTAN – smooth everywhere
  checkGradient(
    'ARCTAN',
    Activations.ARCTAN.output,
    Activations.ARCTAN.der,
    [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
  );

  // ISRU – smooth everywhere
  checkGradient(
    'ISRU',
    Activations.ISRU.output,
    Activations.ISRU.der,
    [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
  );

  // EXPONENTIAL_LINEAR – kink at x=0; skip x=0
  checkGradient(
    'EXPONENTIAL_LINEAR',
    Activations.EXPONENTIAL_LINEAR.output,
    Activations.EXPONENTIAL_LINEAR.der,
    [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
  );

  // PReLU – factory function; test with alpha=0.1 and alpha=0.5
  // Kink at x=0; skip x=0
  describe('PReLU(alpha=0.1)', () => {
    const fn = Activations.PReLU(0.1);
    const pts = [-3, -2, -1, -0.5, 0.5, 1, 2, 3];
    for (const x of pts) {
      test(`PReLU(0.1) der(${x}) matches finite-difference`, () => {
        const analytic = fn.der(x);
        const numeric = numDer(fn.output, x);
        expect(Math.abs(analytic - numeric)).toBeLessThan(TOL);
      });
    }
  });

  describe('PReLU(alpha=0.5)', () => {
    const fn = Activations.PReLU(0.5);
    const pts = [-3, -2, -1, -0.5, 0.5, 1, 2, 3];
    for (const x of pts) {
      test(`PReLU(0.5) der(${x}) matches finite-difference`, () => {
        const analytic = fn.der(x);
        const numeric = numDer(fn.output, x);
        expect(Math.abs(analytic - numeric)).toBeLessThan(TOL);
      });
    }
  });

  // ── Sanity checks for specific known values ──────────────────────────────

  describe('Specific known values', () => {
    test('TANH.output(0) == 0', () => {
      expect(Activations.TANH.output(0)).toBeCloseTo(0);
    });
    test('TANH.der(0) == 1', () => {
      expect(Activations.TANH.der(0)).toBeCloseTo(1);
    });
    test('SIGMOID.output(0) == 0.5', () => {
      expect(Activations.SIGMOID.output(0)).toBeCloseTo(0.5);
    });
    test('SIGMOID.der(0) == 0.25', () => {
      expect(Activations.SIGMOID.der(0)).toBeCloseTo(0.25);
    });
    test('RELU.output(-1) == 0', () => {
      expect(Activations.RELU.output(-1)).toBe(0);
    });
    test('RELU.output(2) == 2', () => {
      expect(Activations.RELU.output(2)).toBe(2);
    });
    test('RELU.der(1) == 1', () => {
      expect(Activations.RELU.der(1)).toBe(1);
    });
    test('RELU.der(-1) == 0', () => {
      expect(Activations.RELU.der(-1)).toBe(0);
    });
    test('ELU.output(0) == 0', () => {
      expect(Activations.ELU.output(0)).toBeCloseTo(0);
    });
    test('GELU.output(0) == 0', () => {
      expect(Activations.GELU.output(0)).toBeCloseTo(0);
    });
    test('SWISH.output(0) == 0', () => {
      expect(Activations.SWISH.output(0)).toBeCloseTo(0);
    });
    test('GAUSSIAN.output(0) == 1', () => {
      expect(Activations.GAUSSIAN.output(0)).toBeCloseTo(1);
    });
    test('ARCTAN.output(0) == 0', () => {
      expect(Activations.ARCTAN.output(0)).toBeCloseTo(0);
    });
    test('ISRU.output(0) == 0', () => {
      expect(Activations.ISRU.output(0)).toBeCloseTo(0);
    });
    test('SOFTSIGN.output(0) == 0', () => {
      expect(Activations.SOFTSIGN.output(0)).toBeCloseTo(0);
    });
    test('BENT_IDENTITY.der(0) == 1', () => {
      expect(Activations.BENT_IDENTITY.der(0)).toBeCloseTo(1);
    });
    test('LEAKY_RELU: negative slope is 0.01', () => {
      expect(Activations.LEAKY_RELU.output(-2)).toBeCloseTo(-0.02);
      expect(Activations.LEAKY_RELU.der(-2)).toBeCloseTo(0.01);
    });
    test('HARD_SIGMOID.output(0) == 0.5', () => {
      expect(Activations.HARD_SIGMOID.output(0)).toBeCloseTo(0.5);
    });
    test('HARD_TANH clamps to ±1', () => {
      expect(Activations.HARD_TANH.output(5)).toBe(1);
      expect(Activations.HARD_TANH.output(-5)).toBe(-1);
    });
    test('RELU6 clamps to 6', () => {
      expect(Activations.RELU6.output(10)).toBe(6);
      expect(Activations.RELU6.output(-1)).toBe(0);
    });
    test('SNAKE.output(0) == 0', () => {
      expect(Activations.SNAKE.output(0)).toBeCloseTo(0);
    });
  });

});
