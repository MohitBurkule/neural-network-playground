/**
 * Adversarial example generation for the 2D playground classifier.
 *
 * NOTE ON INPUT FEATURES:
 *   The playground normally constructs up to 7 input features from (x, y):
 *   x, y, x^2, y^2, x*y, sin(x), sin(y).
 *   These functions operate on RAW (x, y) only — the network is expected to
 *   have exactly 2 input nodes whose outputs equal x and y respectively.
 *   If your network uses derived features, map them yourself before calling
 *   forwardProp, and use the gradient w.r.t. the raw inputs accordingly.
 *
 * Gradient computation strategy:
 *   We propagate inputDer back to the input layer analytically.
 *   After backProp() runs, the first hidden layer has inputDer values set.
 *   We then walk from layer 1 back to layer 0 to compute dLoss/d(x) and
 *   dLoss/d(y) exactly the same way backProp does for earlier layers.
 */

import {Node, forwardProp, backProp, Errors} from './nn';

/**
 * Compute dLoss/d(input_i) for each raw input node analytically.
 * Requires that forwardProp and backProp have already been run on the network
 * with the target label so that inputDer is populated on every node.
 */
function inputGradients(network: Node[][]): number[] {
  const inputLayer = network[0];
  const firstHiddenLayer = network[1];

  const grads: number[] = inputLayer.map(() => 0);

  for (let h = 0; h < firstHiddenLayer.length; h++) {
    const hiddenNode = firstHiddenLayer[h];
    for (let i = 0; i < hiddenNode.inputLinks.length; i++) {
      const link = hiddenNode.inputLinks[i];
      if (link.isDead) continue;
      // dLoss/d(input_i) += w_ih * inputDer_h
      const srcIdx = inputLayer.indexOf(link.source);
      if (srcIdx >= 0) {
        grads[srcIdx] += link.weight * hiddenNode.inputDer;
      }
    }
  }
  return grads;
}

/**
 * Fast Gradient Sign Method (FGSM).
 *
 * Perturbs (x, y) by epsilon in the direction of the loss gradient sign.
 * Returns a new point {x, y, label} with the original label preserved.
 *
 * @param network  Trained network (Node[][]).
 * @param point    Input point {x, y, label}.
 * @param epsilon  Perturbation magnitude (try 0.1 – 1.0 in the [-6,6] domain).
 */
export function fgsm(
    network: Node[][],
    point: {x: number; y: number; label: number},
    epsilon: number): {x: number; y: number; label: number} {

  const inputs = [point.x, point.y];
  forwardProp(network, inputs, null);
  backProp(network, point.label, Errors.SQUARE);

  const grads = inputGradients(network);
  return {
    x: point.x + epsilon * Math.sign(grads[0]),
    y: point.y + epsilon * Math.sign(grads[1]),
    label: point.label
  };
}

/**
 * Projected Gradient Descent (PGD) adversarial attack.
 *
 * Iteratively applies FGSM-like steps, projecting back to an L∞ ball of
 * radius epsilon around the original point after each step.
 *
 * @param network   Trained network (Node[][]).
 * @param point     Input point {x, y, label}.
 * @param epsilon   Maximum L∞ perturbation radius.
 * @param steps     Number of PGD iterations (default 10).
 * @param stepSize  Step size per iteration (default epsilon/4).
 */
export function pgd(
    network: Node[][],
    point: {x: number; y: number; label: number},
    epsilon: number,
    steps: number = 10,
    stepSize: number = epsilon / 4): {x: number; y: number; label: number} {

  let ax = point.x;
  let ay = point.y;

  for (let s = 0; s < steps; s++) {
    const inputs = [ax, ay];
    forwardProp(network, inputs, null);
    backProp(network, point.label, Errors.SQUARE);

    const grads = inputGradients(network);
    ax = ax + stepSize * Math.sign(grads[0]);
    ay = ay + stepSize * Math.sign(grads[1]);

    // Project back to L∞ ball around original point
    ax = Math.max(point.x - epsilon, Math.min(point.x + epsilon, ax));
    ay = Math.max(point.y - epsilon, Math.min(point.y + epsilon, ay));
  }

  return {x: ax, y: ay, label: point.label};
}

/**
 * Convenience: run forwardProp and return the network's scalar prediction.
 */
export function predict(network: Node[][], x: number, y: number): number {
  return forwardProp(network, [x, y], null);
}
