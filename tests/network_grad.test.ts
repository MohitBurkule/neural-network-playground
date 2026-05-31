/**
 * End-to-end gradient check for backProp.
 *
 * Build a small network with fixed weights, compute the analytic gradient
 * dLoss/dw for a sample via backProp, and compare against a central
 * finite-difference approximation of the loss w.r.t. that weight.
 *
 *   numGrad ≈ (loss(w+h) - loss(w-h)) / (2h)
 *
 * Checks are performed for:
 *   1. All link weights in the network.
 *   2. All node biases in the network.
 *   3. Multiple loss functions (SQUARE, HUBER, LOGLOSS).
 *   4. Multiple activation / network shapes.
 */

import {
  buildNetwork,
  forwardProp,
  backProp,
  Activations,
  Errors,
  ErrorFunction,
  Node,
  Link,
} from '../src/nn';

// ─── Configuration ────────────────────────────────────────────────────────────

const H = 1e-5;
const TOL = 1e-4;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Compute the scalar loss for a single (inputs, target) pair with
 * the provided error function.
 */
function computeLoss(
  net: Node[][],
  inputs: number[],
  target: number,
  errorFn: ErrorFunction
): number {
  const out = forwardProp(net, inputs, null);
  return errorFn.error(out, target);
}

/**
 * Set specific deterministic weights on a [2,3,1] network.
 */
function setWeights231(net: Node[][]): void {
  // Hidden layer
  net[1][0].inputLinks[0].weight =  0.4;
  net[1][0].inputLinks[1].weight = -0.3;
  net[1][1].inputLinks[0].weight =  0.2;
  net[1][1].inputLinks[1].weight =  0.5;
  net[1][2].inputLinks[0].weight = -0.15;
  net[1][2].inputLinks[1].weight =  0.35;
  net[1][0].bias =  0.05;
  net[1][1].bias = -0.1;
  net[1][2].bias =  0.07;
  // Output layer
  net[2][0].inputLinks[0].weight =  0.6;
  net[2][0].inputLinks[1].weight = -0.4;
  net[2][0].inputLinks[2].weight =  0.25;
  net[2][0].bias = 0.1;
}

/**
 * Run a full gradient check for all link weights (and optionally biases)
 * in the network.
 */
function runGradientCheck(
  net: Node[][],
  inputs: number[],
  target: number,
  errorFn: ErrorFunction,
  checkBiases: boolean = true,
  tol: number = TOL
): void {
  // Analytic gradients via backProp
  forwardProp(net, inputs, null);
  backProp(net, target, errorFn);

  for (let layerIdx = 1; layerIdx < net.length; layerIdx++) {
    for (const node of net[layerIdx]) {
      // Check each incoming link weight
      for (const link of node.inputLinks) {
        const origW = link.weight;
        const analyticGrad = link.errorDer;  // dLoss/dw

        link.weight = origW + H;
        const lossPlus = computeLoss(net, inputs, target, errorFn);

        link.weight = origW - H;
        const lossMinus = computeLoss(net, inputs, target, errorFn);

        link.weight = origW;
        const numGrad = (lossPlus - lossMinus) / (2 * H);

        expect(Math.abs(analyticGrad - numGrad)).toBeLessThan(tol);
      }

      // Check node bias
      if (checkBiases) {
        const origBias = node.bias;
        // Re-run forward+back at original bias (already done above, but biases
        // are part of node state so we re-derive via perturbation).
        // The analytic gradient for the bias is node.accInputDer / node.numAccumulatedDers
        // which was reduced to node.inputDer after the most recent single backprop.
        const analyticBiasGrad = node.inputDer;  // = node.accInputDer (1 sample)

        node.bias = origBias + H;
        const lossPlus = computeLoss(net, inputs, target, errorFn);

        node.bias = origBias - H;
        const lossMinus = computeLoss(net, inputs, target, errorFn);

        node.bias = origBias;
        const numGrad = (lossPlus - lossMinus) / (2 * H);

        expect(Math.abs(analyticBiasGrad - numGrad)).toBeLessThan(tol);
      }
    }
  }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('End-to-end backprop gradient check', () => {

  // ── 1. [2,3,1] TANH network, SQUARE loss ──────────────────────────────────

  describe('[2,3,1] TANH + SQUARE loss', () => {
    const INPUTS_A = [1.0, -0.5];
    const INPUTS_B = [-0.3, 0.8];
    const TARGET_A = 1.0;
    const TARGET_B = -1.0;

    let net: Node[][];

    beforeEach(() => {
      net = buildNetwork([2, 3, 1], Activations.TANH, Activations.LINEAR, ['x', 'y'], true);
      setWeights231(net);
    });

    test('all link gradients match FD for input A, target 1 (SQUARE)', () => {
      runGradientCheck(net, INPUTS_A, TARGET_A, Errors.SQUARE);
    });

    test('all link gradients match FD for input B, target -1 (SQUARE)', () => {
      runGradientCheck(net, INPUTS_B, TARGET_B, Errors.SQUARE);
    });
  });

  // ── 2. [2,3,1] TANH network, HUBER loss ───────────────────────────────────

  describe('[2,3,1] TANH + HUBER loss', () => {
    let net: Node[][];

    beforeEach(() => {
      net = buildNetwork([2, 3, 1], Activations.TANH, Activations.LINEAR, ['x', 'y'], true);
      setWeights231(net);
    });

    test('all link gradients match FD (HUBER)', () => {
      runGradientCheck(net, [0.7, -0.4], 0.5, Errors.HUBER);
    });
  });

  // ── 3. [2,3,1] SIGMOID output, LOGLOSS ────────────────────────────────────
  //    LOGLOSS uses output remapped through p=(output+1)/2 then cross-entropy.
  //    Use TANH hidden + TANH output so output ∈ (-1,1), avoiding saturation.

  describe('[2,3,1] TANH + LOGLOSS', () => {
    let net: Node[][];

    beforeEach(() => {
      net = buildNetwork([2, 3, 1], Activations.TANH, Activations.TANH, ['x', 'y'], true);
      setWeights231(net);
    });

    test('all link gradients match FD for target 1 (LOGLOSS)', () => {
      runGradientCheck(net, [0.5, -0.2], 1, Errors.LOGLOSS);
    });

    test('all link gradients match FD for target -1 (LOGLOSS)', () => {
      runGradientCheck(net, [-0.5, 0.2], -1, Errors.LOGLOSS);
    });
  });

  // ── 4. Deeper network [2,4,3,1] TANH + SQUARE ─────────────────────────────

  describe('[2,4,3,1] TANH + SQUARE loss (deeper)', () => {
    let net: Node[][];

    beforeEach(() => {
      net = buildNetwork([2, 4, 3, 1], Activations.TANH, Activations.LINEAR, ['x', 'y'], true);
      // Set deterministic weights across all layers
      const ws = [0.3, -0.2, 0.4, -0.3, 0.1, 0.2, -0.1, 0.3,
                  0.5, -0.4, 0.2, -0.1, 0.3, -0.2, 0.1, 0.4,
                  0.2, -0.3, 0.1, -0.15, 0.25, -0.2, 0.15, -0.1];
      let wi = 0;
      for (let li = 1; li < net.length; li++) {
        for (const node of net[li]) {
          for (const link of node.inputLinks) {
            link.weight = ws[wi++ % ws.length];
          }
          node.bias = 0.05 * (wi % 5 + 1);
        }
      }
    });

    test('all link gradients match FD (deeper net, SQUARE)', () => {
      runGradientCheck(net, [1.0, -0.5], 1.0, Errors.SQUARE);
    });

    test('all link gradients match FD (different input, deeper net)', () => {
      runGradientCheck(net, [-0.7, 0.9], -1.0, Errors.SQUARE);
    });
  });

  // ── 5. RELU hidden + LINEAR output ────────────────────────────────────────
  //    Use inputs/weights well away from x=0 so ReLU is differentiable.

  describe('[2,3,1] RELU hidden + LINEAR output + SQUARE', () => {
    let net: Node[][];

    beforeEach(() => {
      net = buildNetwork([2, 3, 1], Activations.RELU, Activations.LINEAR, ['x', 'y'], true);
      // Weights chosen so all pre-activations are positive (ReLU is smooth there).
      net[1][0].inputLinks[0].weight =  0.4;
      net[1][0].inputLinks[1].weight =  0.3;
      net[1][1].inputLinks[0].weight =  0.2;
      net[1][1].inputLinks[1].weight =  0.5;
      net[1][2].inputLinks[0].weight =  0.15;
      net[1][2].inputLinks[1].weight =  0.35;
      net[1][0].bias =  0.5;
      net[1][1].bias =  0.5;
      net[1][2].bias =  0.5;
      net[2][0].inputLinks[0].weight =  0.6;
      net[2][0].inputLinks[1].weight =  0.4;
      net[2][0].inputLinks[2].weight =  0.25;
      net[2][0].bias = 0.0;
    });

    test('all link gradients match FD (RELU, positive pre-activations)', () => {
      // Inputs > 0 so all hidden pre-activations are positive → ReLU is smooth.
      runGradientCheck(net, [1.0, 0.5], 1.0, Errors.SQUARE, false);
    });
  });

  // ── 6. SIGMOID hidden + SQUARE loss ───────────────────────────────────────

  describe('[2,3,1] SIGMOID hidden + SQUARE loss', () => {
    let net: Node[][];

    beforeEach(() => {
      net = buildNetwork([2, 3, 1], Activations.SIGMOID, Activations.LINEAR, ['x', 'y'], true);
      setWeights231(net);
    });

    test('all link gradients match FD (SIGMOID)', () => {
      runGradientCheck(net, [0.8, -0.6], 0.5, Errors.SQUARE);
    });
  });

  // ── 7. ABSOLUTE loss (away from kink) ─────────────────────────────────────
  //    Use a TANH network so output is bounded in (-1,1).
  //    Target near ±1 ensures output ≠ target (ABSOLUTE kink only at d=0).

  describe('[2,3,1] TANH + ABSOLUTE loss', () => {
    let net: Node[][];

    beforeEach(() => {
      net = buildNetwork([2, 3, 1], Activations.TANH, Activations.TANH, ['x', 'y'], true);
      // Use weights that push output away from target so |d| >> 0
      net[1][0].inputLinks[0].weight =  0.3;
      net[1][0].inputLinks[1].weight = -0.2;
      net[1][1].inputLinks[0].weight =  0.1;
      net[1][1].inputLinks[1].weight =  0.4;
      net[1][2].inputLinks[0].weight = -0.1;
      net[1][2].inputLinks[1].weight =  0.3;
      net[1][0].bias = 0.1; net[1][1].bias = 0.1; net[1][2].bias = 0.1;
      net[2][0].inputLinks[0].weight = 0.5;
      net[2][0].inputLinks[1].weight = 0.3;
      net[2][0].inputLinks[2].weight = 0.2;
      net[2][0].bias = 0.0;
    });

    test('all link gradients match FD (ABSOLUTE, target=−1, output well above −1)', () => {
      // With these weights and input [0.5, 0.5], network output is positive,
      // so |output - (-1)| >> 0, safely away from the kink.
      runGradientCheck(net, [0.5, 0.5], -1.0, Errors.ABSOLUTE, false);
    });
  });

  // ── 8. Specific link gradient numerical value check ────────────────────────

  describe('Specific link gradient value check', () => {
    test('analytic gradient matches FD for a single specific weight', () => {
      const net = buildNetwork([2, 2, 1], Activations.TANH, Activations.LINEAR, ['x', 'y'], true);
      net[1][0].inputLinks[0].weight =  0.5;
      net[1][0].inputLinks[1].weight = -0.4;
      net[1][1].inputLinks[0].weight =  0.3;
      net[1][1].inputLinks[1].weight =  0.2;
      net[2][0].inputLinks[0].weight =  0.7;
      net[2][0].inputLinks[1].weight = -0.6;
      net[1][0].bias = 0.1;
      net[1][1].bias = 0.1;
      net[2][0].bias = 0.1;

      const inputs = [0.8, -0.3];
      const target = 1.0;
      const errorFn = Errors.SQUARE;

      // Analytic
      forwardProp(net, inputs, null);
      backProp(net, target, errorFn);
      const analyticGrad = net[2][0].inputLinks[0].errorDer;

      // Numerical
      const link = net[2][0].inputLinks[0];
      link.weight += H;
      const lp = computeLoss(net, inputs, target, errorFn);
      link.weight -= 2 * H;
      const lm = computeLoss(net, inputs, target, errorFn);
      link.weight += H;

      const numGrad = (lp - lm) / (2 * H);
      expect(Math.abs(analyticGrad - numGrad)).toBeLessThan(TOL);
    });
  });

});
