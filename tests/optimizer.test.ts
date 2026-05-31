/**
 * Optimizer correctness tests.
 *
 * For each OptimizerType (SGD, MOMENTUM, RMSPROP, ADAM, NESTEROV, ADAGRAD,
 * ADADELTA, AMSGRAD, NADAM, ADAMW), build a tiny network and train it on a
 * trivially separable 2-point dataset.  Assert that the final loss is strictly
 * below the initial loss, confirming the optimizer path runs and learns.
 *
 * Dataset: { [1, 0] → 1 }, { [-1, 0] → -1 }  (linearly separable)
 * Network: [2, 3, 1] with TANH activations
 * Loss: SQUARE
 *
 * Deterministic initial weights are set for every test so results are
 * independent of Math.random().
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

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Build a [2,3,1] TANH network with fixed deterministic weights. */
function buildTestNet() {
  const net = buildNetwork(
    [2, 3, 1],
    Activations.TANH,
    Activations.TANH,
    ['x', 'y'],
    true  // initZero to start clean, then set weights below
  );
  // Deterministic non-zero initial weights so gradients are non-trivial.
  const hiddenWeights = [
    [0.4, -0.3],
    [0.2,  0.5],
    [-0.1, 0.3],
  ];
  const outputWeights = [0.6, -0.4, 0.3];
  for (let i = 0; i < 3; i++) {
    net[1][i].inputLinks[0].weight = hiddenWeights[i][0];
    net[1][i].inputLinks[1].weight = hiddenWeights[i][1];
    net[1][i].bias = 0.05 * (i + 1);
  }
  for (let i = 0; i < 3; i++) {
    net[2][0].inputLinks[i].weight = outputWeights[i];
  }
  net[2][0].bias = 0.05;
  return net;
}

const TRAIN_DATA = [
  { input: [1.0,  0.0], target:  1.0 },
  { input: [-1.0, 0.0], target: -1.0 },
];

function computeLoss(net: ReturnType<typeof buildTestNet>): number {
  let total = 0;
  for (const d of TRAIN_DATA) {
    const out = forwardProp(net, d.input, null);
    total += Errors.SQUARE.error(out, d.target);
  }
  return total / TRAIN_DATA.length;
}

/**
 * Run `steps` training iterations with the given optimizer and assert
 * that the final loss is strictly below the initial loss.
 */
function trainAndAssertConvergence(
  optimizerType: OptimizerType,
  steps: number,
  learningRate: number
): void {
  const net = buildTestNet();
  const initialLoss = computeLoss(net);

  for (let iter = 0; iter < steps; iter++) {
    for (const d of TRAIN_DATA) {
      forwardProp(net, d.input, null);
      backProp(net, d.target, Errors.SQUARE);
      updateWeights(net, learningRate, null, 0, optimizerType);
    }
  }

  const finalLoss = computeLoss(net);
  expect(finalLoss).toBeLessThan(initialLoss);
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('Optimizer convergence tests', () => {

  test('SGD reduces loss', () => {
    trainAndAssertConvergence(OptimizerType.SGD, 300, 0.1);
  });

  test('MOMENTUM reduces loss', () => {
    trainAndAssertConvergence(OptimizerType.MOMENTUM, 300, 0.05);
  });

  test('RMSPROP reduces loss', () => {
    trainAndAssertConvergence(OptimizerType.RMSPROP, 300, 0.01);
  });

  test('ADAM reduces loss', () => {
    trainAndAssertConvergence(OptimizerType.ADAM, 300, 0.01);
  });

  test('NESTEROV reduces loss', () => {
    trainAndAssertConvergence(OptimizerType.NESTEROV, 300, 0.05);
  });

  test('ADAGRAD reduces loss', () => {
    trainAndAssertConvergence(OptimizerType.ADAGRAD, 300, 0.1);
  });

  test('ADADELTA reduces loss', () => {
    // Adadelta is self-scaling; it needs more steps to warm up.
    trainAndAssertConvergence(OptimizerType.ADADELTA, 500, 1.0);
  });

  test('AMSGRAD reduces loss', () => {
    trainAndAssertConvergence(OptimizerType.AMSGRAD, 300, 0.01);
  });

  test('NADAM reduces loss', () => {
    trainAndAssertConvergence(OptimizerType.NADAM, 300, 0.01);
  });

  test('ADAMW reduces loss', () => {
    trainAndAssertConvergence(OptimizerType.ADAMW, 300, 0.01);
  });

  // ── Additional: optimizer state is maintained across steps ─────────────────

  describe('Optimizer state persistence (Adam)', () => {
    test('Adam m/v state grows from zero after the first step', () => {
      const net = buildTestNet();
      const firstLink = net[1][0].inputLinks[0];

      // Before training: state is empty
      expect(firstLink.optimizerState.m).toBeUndefined();
      expect(firstLink.optimizerState.v).toBeUndefined();

      forwardProp(net, TRAIN_DATA[0].input, null);
      backProp(net, TRAIN_DATA[0].target, Errors.SQUARE);
      updateWeights(net, 0.01, null, 0, OptimizerType.ADAM);

      // After one step: state should be initialized
      expect(firstLink.optimizerState.m).toBeDefined();
      expect(firstLink.optimizerState.v).toBeDefined();
      expect(firstLink.optimizerState.t).toBe(1);
    });

    test('Adam step counter increments correctly', () => {
      const net = buildTestNet();
      const firstLink = net[1][0].inputLinks[0];

      for (let i = 1; i <= 5; i++) {
        forwardProp(net, TRAIN_DATA[0].input, null);
        backProp(net, TRAIN_DATA[0].target, Errors.SQUARE);
        updateWeights(net, 0.01, null, 0, OptimizerType.ADAM);
        expect(firstLink.optimizerState.t).toBe(i);
      }
    });
  });

  // ── XOR-like task (more demanding) ────────────────────────────────────────

  describe('SGD on XOR-like 4-point task', () => {
    test('reduces loss over 500 iterations', () => {
      const net = buildNetwork(
        [2, 4, 1],
        Activations.TANH,
        Activations.TANH,
        ['x', 'y'],
        true
      );
      // Fixed initial weights for determinism
      const hw = [0.3, -0.4, 0.5, -0.2, -0.3, 0.4, 0.2, 0.1];
      for (let i = 0; i < 4; i++) {
        net[1][i].inputLinks[0].weight = hw[i * 2];
        net[1][i].inputLinks[1].weight = hw[i * 2 + 1];
        net[1][i].bias = 0.1;
      }
      net[2][0].inputLinks.forEach((l, i) => { l.weight = (i % 2 === 0) ? 0.5 : -0.5; });
      net[2][0].bias = 0;

      const xorData = [
        { input: [1,  1],  target: -1 },
        { input: [1,  -1], target:  1 },
        { input: [-1, 1],  target:  1 },
        { input: [-1, -1], target: -1 },
      ];

      function loss(): number {
        return xorData.reduce((s, d) =>
          s + Errors.SQUARE.error(forwardProp(net, d.input, null), d.target), 0
        ) / xorData.length;
      }

      const initialLoss = loss();

      for (let iter = 0; iter < 500; iter++) {
        for (const d of xorData) {
          forwardProp(net, d.input, null);
          backProp(net, d.target, Errors.SQUARE);
          updateWeights(net, 0.1, null, 0, OptimizerType.SGD);
        }
      }

      expect(loss()).toBeLessThan(initialLoss);
    });
  });

  // ── Optimizer enum completeness ────────────────────────────────────────────

  describe('OptimizerType enum values', () => {
    test('all expected optimizer types are present', () => {
      const expected = [
        'sgd', 'momentum', 'rmsprop', 'adam',
        'nesterov', 'adagrad', 'adadelta', 'amsgrad', 'nadam', 'adamw'
      ];
      for (const v of expected) {
        expect(Object.values(OptimizerType)).toContain(v);
      }
    });
  });

});
