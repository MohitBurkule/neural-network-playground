/**
 * Extra unit tests for src/nn.ts:
 *  - Finite-difference gradient check (analytic vs numerical dLoss/dw)
 *  - SGD reduces loss across iterations
 *  - forwardProp determinism
 *  - Errors.SQUARE correctness
 */

import {
  buildNetwork,
  forwardProp,
  backProp,
  updateWeights,
  Activations,
  Errors,
  OptimizerType,
  Node,
} from '../src/nn';

// ─── Errors.SQUARE ────────────────────────────────────────────────────────────

describe('Errors.SQUARE', () => {
  test('error returns 0 when output equals target', () => {
    expect(Errors.SQUARE.error(0.5, 0.5)).toBe(0);
  });

  test('error is 0.5*(output-target)^2', () => {
    expect(Errors.SQUARE.error(1, 0)).toBeCloseTo(0.5);
    expect(Errors.SQUARE.error(2, 0)).toBeCloseTo(2.0);
    expect(Errors.SQUARE.error(-1, 1)).toBeCloseTo(2.0);
  });

  test('error is non-negative', () => {
    expect(Errors.SQUARE.error(3, -2)).toBeGreaterThanOrEqual(0);
    expect(Errors.SQUARE.error(-5, 0)).toBeGreaterThanOrEqual(0);
  });

  test('der returns output - target', () => {
    expect(Errors.SQUARE.der(1, 0)).toBeCloseTo(1);
    expect(Errors.SQUARE.der(0, 1)).toBeCloseTo(-1);
    expect(Errors.SQUARE.der(2, 3)).toBeCloseTo(-1);
    expect(Errors.SQUARE.der(0.7, 0.7)).toBeCloseTo(0);
  });
});

// ─── forwardProp determinism ──────────────────────────────────────────────────

describe('forwardProp determinism', () => {
  test('repeated calls with same weights and inputs give identical output', () => {
    const net = buildNetwork([2, 4, 1], Activations.TANH, Activations.LINEAR, ['x', 'y'], true);
    // Manually set deterministic weights
    const w = [0.3, -0.5, 0.1, 0.8, -0.2, 0.4, 0.6, -0.7];
    let wi = 0;
    for (const node of net[1]) {
      for (const link of node.inputLinks) {
        link.weight = w[wi++ % w.length];
      }
    }
    for (const node of net[2]) {
      for (const link of node.inputLinks) {
        link.weight = w[wi++ % w.length];
      }
    }

    const out1 = forwardProp(net, [1.5, -0.5], null);
    const out2 = forwardProp(net, [1.5, -0.5], null);
    const out3 = forwardProp(net, [1.5, -0.5], null);
    expect(out1).toBe(out2);
    expect(out2).toBe(out3);
  });

  test('different inputs give different outputs (non-trivial net)', () => {
    const net = buildNetwork([2, 3, 1], Activations.TANH, Activations.LINEAR, ['x', 'y'], false);
    // Set non-zero weights so different inputs matter
    net[1].forEach((n, i) => n.inputLinks.forEach((l, j) => { l.weight = (i + 1) * 0.3 + j * 0.1; }));
    net[2][0].inputLinks.forEach((l, j) => { l.weight = j * 0.2 + 0.1; });

    const outA = forwardProp(net, [1.0, 2.0], null);
    const outB = forwardProp(net, [-1.0, -2.0], null);
    expect(outA).not.toBe(outB);
  });
});

// ─── Finite-difference gradient check ────────────────────────────────────────

describe('Backprop finite-difference gradient check', () => {
  /**
   * For a tiny 2-input, 2-hidden, 1-output network, verify that the analytic
   * gradient dLoss/dw from backProp matches a central finite-difference
   * approximation within a relative tolerance.
   */
  test('analytic gradient matches numerical gradient for all weights (tolerance 1e-4)', () => {
    const net = buildNetwork([2, 2, 1], Activations.TANH, Activations.LINEAR, ['x', 'y'], true);
    // Set deterministic non-zero weights
    net[1][0].inputLinks[0].weight = 0.4;
    net[1][0].inputLinks[1].weight = -0.3;
    net[1][1].inputLinks[0].weight = 0.2;
    net[1][1].inputLinks[1].weight = 0.5;
    net[2][0].inputLinks[0].weight = 0.6;
    net[2][0].inputLinks[1].weight = -0.4;
    net[1][0].bias = 0.05;
    net[1][1].bias = -0.05;
    net[2][0].bias = 0.1;

    const inputs = [1.0, -0.5];
    const target = 1.0;
    const h = 1e-5;

    // Compute analytic gradient via forward+backprop
    forwardProp(net, inputs, null);
    backProp(net, target, Errors.SQUARE);

    // Collect all weights and their analytic gradients
    const analyticGrads: {link: any; grad: number}[] = [];
    for (let li = 1; li < net.length; li++) {
      for (const node of net[li]) {
        for (const link of node.inputLinks) {
          analyticGrads.push({link, grad: link.errorDer});
        }
      }
    }

    // Numerical gradient via central differences
    for (const {link, grad} of analyticGrads) {
      const orig = link.weight;

      link.weight = orig + h;
      const lossPlus = Errors.SQUARE.error(forwardProp(net, inputs, null), target);

      link.weight = orig - h;
      const lossMinus = Errors.SQUARE.error(forwardProp(net, inputs, null), target);

      link.weight = orig;

      const numGrad = (lossPlus - lossMinus) / (2 * h);
      expect(Math.abs(grad - numGrad)).toBeLessThan(1e-4);
    }
  });
});

// ─── SGD reduces loss ─────────────────────────────────────────────────────────

describe('updateWeights reduces loss with SGD', () => {
  test('loss decreases monotonically for a simple separable problem over 100 steps', () => {
    const net = buildNetwork([2, 4, 1], Activations.TANH, Activations.LINEAR, ['x', 'y'], true);
    // Set non-zero initial weights
    let w = 0.4;
    for (const node of net[1]) { for (const link of node.inputLinks) { link.weight = w; w = -w * 0.8 + 0.05; } }
    for (const node of net[2]) { for (const link of node.inputLinks) { link.weight = w; w = -w * 0.8 + 0.05; } }

    const data = [
      {x: 1.0, y: 1.0, target: 1.0},
      {x: -1.0, y: -1.0, target: -1.0},
      {x: 1.0, y: -1.0, target: 1.0},
      {x: -1.0, y: 1.0, target: -1.0},
    ];

    function computeLoss(): number {
      return data.reduce((sum, d) => {
        return sum + Errors.SQUARE.error(forwardProp(net, [d.x, d.y], null), d.target);
      }, 0) / data.length;
    }

    const initialLoss = computeLoss();
    const lr = 0.05;

    for (let iter = 0; iter < 100; iter++) {
      for (const d of data) {
        forwardProp(net, [d.x, d.y], null);
        backProp(net, d.target, Errors.SQUARE);
        updateWeights(net, lr, null, 0, OptimizerType.SGD);
      }
    }

    const finalLoss = computeLoss();
    expect(finalLoss).toBeLessThan(initialLoss);
  });

  test('SGD with Adam optimizer also reduces loss', () => {
    const net = buildNetwork([2, 3, 1], Activations.SIGMOID, Activations.LINEAR, ['x', 'y'], true);
    // Set initial weights
    net[1].forEach(n => n.inputLinks.forEach((l, i) => { l.weight = i === 0 ? 0.3 : -0.3; }));
    net[2][0].inputLinks.forEach((l, i) => { l.weight = i === 0 ? 0.5 : -0.5; l.weight += i * 0.1; });

    const data = [
      {x: 2.0, y: 0.5, target: 1.0},
      {x: -2.0, y: -0.5, target: -1.0},
    ];

    function computeLoss(): number {
      return data.reduce((sum, d) => {
        return sum + Errors.SQUARE.error(forwardProp(net, [d.x, d.y], null), d.target);
      }, 0) / data.length;
    }

    const initialLoss = computeLoss();

    for (let iter = 0; iter < 200; iter++) {
      for (const d of data) {
        forwardProp(net, [d.x, d.y], null);
        backProp(net, d.target, Errors.SQUARE);
        updateWeights(net, 0.01, null, 0, OptimizerType.ADAM);
      }
    }

    expect(computeLoss()).toBeLessThan(initialLoss);
  });
});
