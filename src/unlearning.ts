/**
 * Machine Unlearning for the 2D playground classifier.
 *
 * Two strategies are provided:
 *
 * 1. forgetPoints — "gradient-ascent forgetting":
 *    Takes N gradient-ASCENT steps on the forget set (maximising loss on those
 *    points) while taking normal gradient-DESCENT steps on the retain set so
 *    the model's accuracy on retained data is preserved as much as possible.
 *
 * 2. retrainWithout — "retrain from scratch" baseline:
 *    Builds a fresh network with the same shape and trains it only on the
 *    retain set. Acts as the gold-standard unlearning baseline.
 *
 * Both functions also return a ForgetMetrics snapshot describing loss/accuracy
 * on forget vs retain sets before and after the operation.
 */

import {
  Node, Link, buildNetwork, forwardProp, backProp,
  Activations, Errors, OptimizerType
} from './nn';
import {Example2D} from './dataset';

export interface UnlearnOpts {
  /** Learning rate for ascent/descent steps. Default 0.03. */
  learningRate?: number;
  /** Number of unlearning steps. Default 200. */
  steps?: number;
  /** Ratio of retain-set gradient-descent steps per forget-ascent step. Default 1. */
  retainRatio?: number;
  /** Batch size drawn from retain set each step. Default 32. */
  batchSize?: number;
}

export interface ForgetMetrics {
  /** Mean squared error on the forget set. */
  forgetLoss: number;
  /** Classification accuracy on the forget set (prediction sign matches label). */
  forgetAccuracy: number;
  /** Mean squared error on the retain set. */
  retainLoss: number;
  /** Classification accuracy on the retain set. */
  retainAccuracy: number;
}

/** Compute MSE and accuracy for a set of examples on the current network. */
export function evalMetrics(
    network: Node[][],
    examples: Example2D[]): {loss: number; accuracy: number} {
  if (examples.length === 0) return {loss: 0, accuracy: 1};
  let totalLoss = 0;
  let correct = 0;
  for (const ex of examples) {
    const pred = forwardProp(network, [ex.x, ex.y], null);
    totalLoss += Errors.SQUARE.error(pred, ex.label);
    if (Math.sign(pred) === Math.sign(ex.label)) correct++;
  }
  return {
    loss: totalLoss / examples.length,
    accuracy: correct / examples.length
  };
}

/** Compute full ForgetMetrics (forget + retain) on a given network. */
export function measureForgetMetrics(
    network: Node[][],
    forgetSet: Example2D[],
    retainSet: Example2D[]): ForgetMetrics {
  const fm = evalMetrics(network, forgetSet);
  const rm = evalMetrics(network, retainSet);
  return {
    forgetLoss: fm.loss,
    forgetAccuracy: fm.accuracy,
    retainLoss: rm.loss,
    retainAccuracy: rm.accuracy
  };
}

/** Apply a single SGD update (descent or ascent) for one example. */
function sgdStep(
    network: Node[][],
    example: Example2D,
    learningRate: number,
    ascent: boolean): void {
  forwardProp(network, [example.x, example.y], null);
  backProp(network, example.label, Errors.SQUARE);

  const sign = ascent ? -1 : 1;  // ascent: subtract a negative gradient = add

  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    for (const node of network[layerIdx]) {
      // Update bias
      node.bias -= sign * learningRate * node.accInputDer;
      node.accInputDer = 0;
      node.numAccumulatedDers = 0;
      // Update weights
      for (const link of node.inputLinks) {
        if (link.isDead) continue;
        link.weight -= sign * learningRate * link.accErrorDer;
        link.accErrorDer = 0;
        link.numAccumulatedDers = 0;
      }
    }
  }
}

/** Sample k items uniformly at random from array (with replacement if k > arr.length). */
function sampleBatch<T>(arr: T[], k: number): T[] {
  if (arr.length === 0) return [];
  const batch: T[] = [];
  for (let i = 0; i < k; i++) {
    batch.push(arr[Math.floor(Math.random() * arr.length)]);
  }
  return batch;
}

/**
 * Forget a subset of training points via gradient ascent.
 *
 * Modifies the network IN-PLACE. Returns metrics before and after unlearning.
 *
 * @param network    Trained network to unlearn from (mutated in-place).
 * @param forgetSet  Points to forget.
 * @param retainSet  Points to retain.
 * @param opts       Hyper-parameters.
 */
export function forgetPoints(
    network: Node[][],
    forgetSet: Example2D[],
    retainSet: Example2D[],
    opts: UnlearnOpts = {}): {before: ForgetMetrics; after: ForgetMetrics} {

  const lr = opts.learningRate ?? 0.03;
  const steps = opts.steps ?? 200;
  const retainRatio = opts.retainRatio ?? 1;
  const batchSize = opts.batchSize ?? 32;

  const before = measureForgetMetrics(network, forgetSet, retainSet);

  for (let step = 0; step < steps; step++) {
    // Gradient ascent on forget set
    for (const ex of sampleBatch(forgetSet, 1)) {
      sgdStep(network, ex, lr, true);
    }
    // Gradient descent on retain set (to maintain accuracy)
    if (retainSet.length > 0) {
      for (let r = 0; r < retainRatio; r++) {
        for (const ex of sampleBatch(retainSet, batchSize)) {
          sgdStep(network, ex, lr, false);
        }
      }
    }
  }

  const after = measureForgetMetrics(network, forgetSet, retainSet);
  return {before, after};
}

export interface RetrainOpts {
  learningRate?: number;
  steps?: number;
  batchSize?: number;
}

/**
 * Retrain a fresh network from scratch on the retain set only.
 *
 * @param shape      Network shape, e.g. [2, 4, 4, 1].
 * @param retainSet  Examples to train on.
 * @param forgetSet  Examples to measure forgetting against (not trained on).
 * @param opts       Hyper-parameters.
 * @returns {network, metrics}  Freshly trained network and ForgetMetrics.
 */
export function retrainWithout(
    shape: number[],
    retainSet: Example2D[],
    forgetSet: Example2D[],
    opts: RetrainOpts = {}): {network: Node[][]; metrics: ForgetMetrics} {

  const lr = opts.learningRate ?? 0.03;
  const steps = opts.steps ?? 1000;
  const batchSize = opts.batchSize ?? 32;

  const inputIds = shape[0] === 2 ? ['x', 'y'] :
    Array.from({length: shape[0]}, (_, i) => `i${i}`);

  const network = buildNetwork(
    shape, Activations.TANH, Activations.LINEAR, inputIds);

  for (let step = 0; step < steps; step++) {
    for (const ex of sampleBatch(retainSet, batchSize)) {
      sgdStep(network, ex, lr, false);
    }
  }

  const metrics = measureForgetMetrics(network, forgetSet, retainSet);
  return {network, metrics};
}

// ============================================================================
// Additional unlearning analysis & variants (additive; existing exports kept).
// ============================================================================

/**
 * Fine-tune-on-retain unlearning variant: continue plain gradient DESCENT on
 * the retain set only for K steps. The forget set is never trained on, so the
 * model gradually drifts away from fitting it. Mutates the network in-place.
 */
export function fineTuneOnRetain(
    network: Node[][],
    forgetSet: Example2D[],
    retainSet: Example2D[],
    opts: UnlearnOpts = {}): {before: ForgetMetrics; after: ForgetMetrics} {
  const lr = opts.learningRate ?? 0.03;
  const steps = opts.steps ?? 200;
  const batchSize = opts.batchSize ?? 32;
  const before = measureForgetMetrics(network, forgetSet, retainSet);
  for (let step = 0; step < steps; step++) {
    for (const ex of sampleBatch(retainSet, batchSize)) {
      sgdStep(network, ex, lr, false);
    }
  }
  const after = measureForgetMetrics(network, forgetSet, retainSet);
  return {before, after};
}

/**
 * Membership-inference proxy. A simple MIA signal is the loss gap between the
 * forget set and a random subset of the retain set: a model that memorised the
 * forget set has much lower loss there than on unseen-like retain points.
 * After unlearning the gap should shrink toward zero (or invert).
 */
export function membershipInferenceGap(
    network: Node[][],
    forgetSet: Example2D[],
    retainSet: Example2D[],
    subsetSize: number = 0): {forgetLoss: number; retainLoss: number;
                              gap: number} {
  const k = subsetSize > 0 ?
    Math.min(subsetSize, retainSet.length) :
    Math.min(forgetSet.length || retainSet.length, retainSet.length);
  const subset = retainSet.length <= k ? retainSet : sampleBatch(retainSet, k);
  const fl = evalMetrics(network, forgetSet).loss;
  const rl = evalMetrics(network, subset).loss;
  // Positive gap => forget set fits better than retain (memorisation signal).
  return {forgetLoss: fl, retainLoss: rl, gap: rl - fl};
}

/**
 * Forget-quality score in [0, 100]. Combines how much accuracy DROPPED on the
 * forget set (we want low forget accuracy) with how well retain accuracy is
 * RETAINED. Both before and after metrics are supplied.
 */
export function forgetQualityScore(
    before: ForgetMetrics, after: ForgetMetrics): number {
  // Forget component: drop relative to before (clamped to [0,1]).
  const drop = Math.max(0, Math.min(1,
      before.forgetAccuracy - after.forgetAccuracy +
      (1 - after.forgetAccuracy)));
  const forgetComponent = Math.max(0, Math.min(1, 1 - after.forgetAccuracy));
  // Retention component: retain accuracy preserved relative to before.
  const retainComponent = before.retainAccuracy <= 0 ? 1 :
      Math.max(0, Math.min(1, after.retainAccuracy / before.retainAccuracy));
  void drop;
  return Math.round(100 * (0.5 * forgetComponent + 0.5 * retainComponent));
}

/**
 * Relearn-time probe: after forgetting, measure how many gradient-descent steps
 * (on the forget set) are needed to recover a target accuracy. Restores all
 * weights/biases afterwards so the probe is non-destructive.
 *
 * @returns steps needed (or maxSteps if not reached) and the achieved accuracy.
 */
export function relearnTimeProbe(
    network: Node[][],
    forgetSet: Example2D[],
    opts: {learningRate?: number; maxSteps?: number;
           targetAccuracy?: number} = {}): {steps: number; accuracy: number;
                                            reached: boolean} {
  const lr = opts.learningRate ?? 0.03;
  const maxSteps = opts.maxSteps ?? 300;
  const target = opts.targetAccuracy ?? 1;
  if (forgetSet.length === 0) return {steps: 0, accuracy: 1, reached: true};

  // Snapshot weights & biases.
  const snapshot: {weights: number[][]; biases: number[]} = {
    weights: [], biases: []
  };
  for (let l = 1; l < network.length; l++) {
    for (const node of network[l]) {
      snapshot.biases.push(node.bias);
      snapshot.weights.push(node.inputLinks.map(lk => lk.weight));
    }
  }

  let steps = maxSteps;
  let acc = 0;
  for (let s = 1; s <= maxSteps; s++) {
    for (const ex of forgetSet) sgdStep(network, ex, lr, false);
    acc = evalMetrics(network, forgetSet).accuracy;
    if (acc >= target) { steps = s; break; }
  }
  if (acc < target) acc = evalMetrics(network, forgetSet).accuracy;

  // Restore.
  let bi = 0;
  let wi = 0;
  for (let l = 1; l < network.length; l++) {
    for (const node of network[l]) {
      node.bias = snapshot.biases[bi++];
      const ws = snapshot.weights[wi++];
      node.inputLinks.forEach((lk, i) => { lk.weight = ws[i]; });
    }
  }
  return {steps, accuracy: acc, reached: acc >= target};
}
