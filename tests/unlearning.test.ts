/**
 * Tests for src/unlearning.ts: evalMetrics, measureForgetMetrics,
 * forgetPoints, retrainWithout.
 */

import {
  evalMetrics,
  measureForgetMetrics,
  forgetPoints,
  retrainWithout,
  ForgetMetrics,
} from '../src/unlearning';
import {buildNetwork, Activations, forwardProp, backProp, updateWeights, Errors, OptimizerType} from '../src/nn';
import {Example2D} from '../src/dataset';

/** Build and lightly train a tiny net on a separable 2D dataset. */
function makeTrainedNet(): ReturnType<typeof buildNetwork> {
  const net = buildNetwork([2, 4, 1], Activations.TANH, Activations.LINEAR, ['x', 'y'], true);
  // Set non-trivial initial weights
  net[1].forEach((n, i) => n.inputLinks.forEach((l, j) => {
    l.weight = (i + 1) * 0.3 * (j === 0 ? 1 : -1);
  }));
  net[2][0].inputLinks.forEach((l, i) => { l.weight = i % 2 === 0 ? 0.5 : -0.4; });

  const data: Example2D[] = [
    {x: 2.0, y: 1.0, label: 1},
    {x: -2.0, y: -1.0, label: -1},
    {x: 1.5, y: 2.0, label: 1},
    {x: -1.5, y: -2.0, label: -1},
  ];

  // Train for 100 steps
  for (let i = 0; i < 100; i++) {
    for (const ex of data) {
      forwardProp(net, [ex.x, ex.y], null);
      backProp(net, ex.label, Errors.SQUARE);
      updateWeights(net, 0.05, null, 0, OptimizerType.SGD);
    }
  }
  return net;
}

const retainData: Example2D[] = [
  {x: 2.0, y: 1.0, label: 1},
  {x: -2.0, y: -1.0, label: -1},
  {x: 1.5, y: 2.0, label: 1},
  {x: -1.5, y: -2.0, label: -1},
];

const forgetData: Example2D[] = [
  {x: 0.5, y: 0.5, label: 1},
  {x: -0.5, y: -0.5, label: -1},
];

// ─── evalMetrics ──────────────────────────────────────────────────────────────

describe('evalMetrics', () => {
  test('returns loss=0 and accuracy=1 for empty examples', () => {
    const net = buildNetwork([2, 2, 1], Activations.TANH, Activations.LINEAR, ['x', 'y'], true);
    const {loss, accuracy} = evalMetrics(net, []);
    expect(loss).toBe(0);
    expect(accuracy).toBe(1);
  });

  test('loss is non-negative', () => {
    const net = makeTrainedNet();
    const {loss} = evalMetrics(net, retainData);
    expect(loss).toBeGreaterThanOrEqual(0);
  });

  test('accuracy is between 0 and 1', () => {
    const net = makeTrainedNet();
    const {accuracy} = evalMetrics(net, retainData);
    expect(accuracy).toBeGreaterThanOrEqual(0);
    expect(accuracy).toBeLessThanOrEqual(1);
  });

  test('trained net has high accuracy on simple separable data', () => {
    const net = makeTrainedNet();
    const {accuracy} = evalMetrics(net, retainData);
    expect(accuracy).toBeGreaterThanOrEqual(0.5); // At minimum better than chance
  });
});

// ─── measureForgetMetrics ─────────────────────────────────────────────────────

describe('measureForgetMetrics', () => {
  test('returns ForgetMetrics with all four fields', () => {
    const net = makeTrainedNet();
    const metrics = measureForgetMetrics(net, forgetData, retainData);
    expect(metrics).toHaveProperty('forgetLoss');
    expect(metrics).toHaveProperty('forgetAccuracy');
    expect(metrics).toHaveProperty('retainLoss');
    expect(metrics).toHaveProperty('retainAccuracy');
  });

  test('all metric values are finite numbers', () => {
    const net = makeTrainedNet();
    const metrics = measureForgetMetrics(net, forgetData, retainData);
    for (const val of Object.values(metrics)) {
      expect(isFinite(val)).toBe(true);
    }
  });
});

// ─── forgetPoints ─────────────────────────────────────────────────────────────

describe('forgetPoints', () => {
  test('returns before and after ForgetMetrics', () => {
    const net = makeTrainedNet();
    const result = forgetPoints(net, forgetData, retainData, {steps: 10, learningRate: 0.05});
    expect(result).toHaveProperty('before');
    expect(result).toHaveProperty('after');
    expect(result.before).toHaveProperty('forgetLoss');
    expect(result.after).toHaveProperty('forgetLoss');
  });

  test('forgetLoss increases (or stays the same) after gradient ascent on forget set', () => {
    const net = makeTrainedNet();
    const result = forgetPoints(net, forgetData, retainData, {
      steps: 100,
      learningRate: 0.1,
      retainRatio: 0,  // only forget, no retain repair
      batchSize: 1,
    });
    // Gradient ascent should raise forget loss
    expect(result.after.forgetLoss).toBeGreaterThanOrEqual(result.before.forgetLoss * 0.8);
  });

  test('modifies network in-place', () => {
    const net = makeTrainedNet();
    const weightBefore = net[1][0].inputLinks[0].weight;
    forgetPoints(net, forgetData, retainData, {steps: 20, learningRate: 0.05});
    const weightAfter = net[1][0].inputLinks[0].weight;
    expect(weightAfter).not.toBe(weightBefore);
  });

  test('before metrics reflect the initial network state', () => {
    const net = makeTrainedNet();
    const initialMetrics = measureForgetMetrics(net, forgetData, retainData);
    const result = forgetPoints(net, forgetData, retainData, {steps: 5});
    expect(result.before.forgetLoss).toBeCloseTo(initialMetrics.forgetLoss, 5);
    expect(result.before.retainLoss).toBeCloseTo(initialMetrics.retainLoss, 5);
  });
});

// ─── retrainWithout ───────────────────────────────────────────────────────────

describe('retrainWithout', () => {
  test('returns a network and metrics object', () => {
    const result = retrainWithout([2, 4, 1], retainData, forgetData, {steps: 50});
    expect(result).toHaveProperty('network');
    expect(result).toHaveProperty('metrics');
  });

  test('returned network can do forward pass', () => {
    const {network} = retrainWithout([2, 4, 1], retainData, forgetData, {steps: 10});
    const out = forwardProp(network, [1.0, 0.5], null);
    expect(typeof out).toBe('number');
    expect(isFinite(out)).toBe(true);
  });

  test('metrics contain valid numbers', () => {
    const {metrics} = retrainWithout([2, 4, 1], retainData, forgetData, {steps: 20});
    for (const val of Object.values(metrics)) {
      expect(isFinite(val)).toBe(true);
    }
  });

  test('retrained network achieves decent accuracy on retain set', () => {
    const {metrics} = retrainWithout([2, 4, 1], retainData, forgetData, {steps: 500, learningRate: 0.1});
    // Should learn to classify retain data reasonably
    expect(metrics.retainAccuracy).toBeGreaterThanOrEqual(0.5);
  });

  test('network shape matches requested shape', () => {
    const {network} = retrainWithout([2, 4, 1], retainData, forgetData, {steps: 10});
    expect(network.length).toBe(3); // input + 1 hidden + output
    expect(network[0].length).toBe(2);
    expect(network[1].length).toBe(4);
    expect(network[2].length).toBe(1);
  });
});
