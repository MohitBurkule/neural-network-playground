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
