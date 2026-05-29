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

// ─── Activation function tests ────────────────────────────────────────────────

describe('Activations', () => {
  test('RELU at 0 returns 0', () => {
    expect(Activations.RELU.output(0)).toBe(0);
  });
  test('RELU positive pass-through', () => {
    expect(Activations.RELU.output(3)).toBe(3);
  });
  test('RELU clips negatives to 0', () => {
    expect(Activations.RELU.output(-5)).toBe(0);
  });

  test('SIGMOID at 0 returns 0.5', () => {
    expect(Activations.SIGMOID.output(0)).toBeCloseTo(0.5);
  });
  test('SIGMOID output is in (0,1)', () => {
    expect(Activations.SIGMOID.output(-10)).toBeGreaterThan(0);
    expect(Activations.SIGMOID.output(10)).toBeLessThan(1);
  });

  test('TANH at 0 returns 0', () => {
    expect(Activations.TANH.output(0)).toBeCloseTo(0);
  });
  test('TANH is bounded in (-1,1)', () => {
    expect(Activations.TANH.output(100)).toBeCloseTo(1);
    expect(Activations.TANH.output(-100)).toBeCloseTo(-1);
  });

  test('LINEAR is identity', () => {
    expect(Activations.LINEAR.output(7)).toBe(7);
    expect(Activations.LINEAR.output(-3.5)).toBe(-3.5);
  });
});

// ─── buildNetwork tests ───────────────────────────────────────────────────────

describe('buildNetwork', () => {
  test('produces the correct number of layers', () => {
    // shape: [2 inputs, 3 hidden, 1 output]
    const net = buildNetwork([2, 3, 1], Activations.RELU, Activations.LINEAR, ['x', 'y']);
    expect(net.length).toBe(3);
  });

  test('each layer has the correct number of nodes', () => {
    const net = buildNetwork([2, 4, 3, 1], Activations.TANH, Activations.LINEAR, ['x', 'y']);
    expect(net[0].length).toBe(2);
    expect(net[1].length).toBe(4);
    expect(net[2].length).toBe(3);
    expect(net[3].length).toBe(1);
  });

  test('input layer nodes use the provided IDs', () => {
    const net = buildNetwork([2, 1], Activations.RELU, Activations.LINEAR, ['feat1', 'feat2']);
    expect(net[0][0].id).toBe('feat1');
    expect(net[0][1].id).toBe('feat2');
  });

  test('links are wired: every non-input node has inputLinks from previous layer', () => {
    const net = buildNetwork([2, 3, 1], Activations.RELU, Activations.LINEAR, ['x', 'y']);
    // Hidden layer nodes each have 2 input links (from the 2 input nodes)
    net[1].forEach(node => {
      expect(node.inputLinks.length).toBe(2);
    });
    // Output node has 3 input links (from the 3 hidden nodes)
    expect(net[2][0].inputLinks.length).toBe(3);
  });

  test('links are wired: previous layer nodes have output links', () => {
    const net = buildNetwork([2, 3, 1], Activations.RELU, Activations.LINEAR, ['x', 'y']);
    // Each input node should have 3 output links (one to each hidden node)
    net[0].forEach(node => {
      expect(node.outputs.length).toBe(3);
    });
  });

  test('initZero sets all biases and weights to zero', () => {
    const net = buildNetwork([2, 3, 1], Activations.RELU, Activations.LINEAR, ['x', 'y'], true);
    net.forEach(layer => {
      layer.forEach(node => {
        expect(node.bias).toBe(0);
        node.inputLinks.forEach(link => {
          expect(link.weight).toBe(0);
        });
      });
    });
  });
});

// ─── forwardProp tests ────────────────────────────────────────────────────────

describe('forwardProp', () => {
  test('runs without error and returns a number', () => {
    const net = buildNetwork([2, 2, 1], Activations.RELU, Activations.LINEAR, ['x', 'y'], true);
    const out = forwardProp(net, [1, 2], null, false);
    expect(typeof out).toBe('number');
  });

  test('is deterministic for fixed weights', () => {
    const net = buildNetwork([2, 3, 1], Activations.TANH, Activations.LINEAR, ['x', 'y'], true);
    // Set some known weights
    net[1][0].inputLinks[0].weight = 0.5;
    net[1][0].inputLinks[1].weight = -0.3;
    net[1][1].inputLinks[0].weight = 0.1;
    net[1][1].inputLinks[1].weight = 0.8;
    net[2][0].inputLinks[0].weight = 0.4;
    net[2][0].inputLinks[1].weight = -0.6;

    const out1 = forwardProp(net, [1.0, 2.0], null, false);
    const out2 = forwardProp(net, [1.0, 2.0], null, false);
    expect(out1).toBe(out2);
  });

  test('throws when input length does not match input layer size', () => {
    const net = buildNetwork([2, 1], Activations.RELU, Activations.LINEAR, ['x', 'y']);
    expect(() => forwardProp(net, [1], null, false)).toThrow();
  });

  test('zero-weight network with zero bias outputs 0 for LINEAR activation', () => {
    const net = buildNetwork([2, 1], Activations.LINEAR, Activations.LINEAR, ['x', 'y'], true);
    const out = forwardProp(net, [5, 3], null, false);
    expect(out).toBe(0);
  });
});

// ─── Learning test ────────────────────────────────────────────────────────────

describe('Network learning (forward/back/update)', () => {
  /**
   * Trivially separable task: sign(x) — output +1 when x>0, -1 when x<0.
   * Uses a tiny 1-input, 2-hidden, 1-output network with TANH.
   */
  test('loss decreases over 200 SGD iterations on a linearly separable task', () => {
    const net = buildNetwork([1, 2, 1], Activations.TANH, Activations.TANH, ['x']);
    // Use fixed seed-like weights to make the test deterministic
    net[1][0].inputLinks[0].weight = 0.3;
    net[1][1].inputLinks[0].weight = -0.3;
    net[2][0].inputLinks[0].weight = 0.5;
    net[2][0].inputLinks[1].weight = 0.5;

    // Simple dataset: {x:1, y:1} and {x:-1, y:-1}
    const data = [
      { input: [1],  target: 1 },
      { input: [-1], target: -1 },
    ];

    function computeLoss(): number {
      return data.reduce((sum, d) => {
        const out = forwardProp(net, d.input, null, false);
        return sum + Errors.SQUARE.error(out, d.target);
      }, 0) / data.length;
    }

    const initialLoss = computeLoss();

    const lr = 0.1;
    for (let iter = 0; iter < 200; iter++) {
      for (const d of data) {
        forwardProp(net, d.input, null, false);
        backProp(net, d.target, Errors.SQUARE);
        updateWeights(net, lr, null, 0, OptimizerType.SGD);
      }
    }

    const finalLoss = computeLoss();
    expect(finalLoss).toBeLessThan(initialLoss);
  });
});
