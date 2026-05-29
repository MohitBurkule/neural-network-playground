/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modified by David Cato
==============================================================================*/

/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
  error: (output: number, target: number) => number;
  der: (output: number, target: number) => number;
}

/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output: (input: number) => number;
  der: (input: number) => number;
  compileToJs: (arg: string) => string;
}

/** Function that computes a penalty cost for a given weight in the network. */
export interface RegularizationFunction {
  output: (weight: number) => number;
  der: (weight: number) => number;
}

/** Function that quantizes a given weight in the network. */
export interface WeightQuantizationFunction {
  output: (weight: number) => number;
}

/**
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
  id: string;
  /** List of input links. */
  inputLinks: Link[] = [];
  bias = 0.1;
  /** List of output links. */
  outputs: Link[] = [];
  totalInput: number;
  output: number;
  /** Error derivative with respect to this node's output. */
  outputDer = 0;
  /** Error derivative with respect to this node's total input. */
  inputDer = 0;
  /**
   * Accumulated error derivative with respect to this node's total input since
   * the last update. This derivative equals dE/db where b is the node's
   * bias term.
   */
  accInputDer = 0;
  /**
   * Number of accumulated err. derivatives with respect to the total input
   * since the last update.
   */
  numAccumulatedDers = 0;
  /** Activation function that takes total input and returns node's output */
  activation: ActivationFunction;
  /** Per-parameter optimizer state for the bias. */
  biasOptimizerState: OptimizerState = {};

  /**
   * Creates a new node with the provided id and activation function.
   */
  constructor(id: string, activation: ActivationFunction, initZero?: boolean) {
    this.id = id;
    this.activation = activation;
    if (initZero) {
      this.bias = 0;
    }
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(weightQuantizationFunction: WeightQuantizationFunction): number {
    // Stores total input into the node.
    this.totalInput = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      let weight = weightQuantizationFunction ? weightQuantizationFunction.output(link.weight) : link.weight
      this.totalInput += weight * link.source.output;
    }
    this.output = this.activation.output(this.totalInput);
    return this.output;
  }

  compileToJs(): string {
    // Stores total input into the node.
    let js = this.bias.toPrecision(2) + "";
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      js += ` + (${link.weight.toPrecision(2)} * ${link.source.compileToJsName()})`;
    }
    return this.activation.compileToJs(js);
  }

  compileToJsName(): string {
    return "v" + this.id;
  }

}
/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
  error: (output: number, target: number) => number;
  der: (output: number, target: number) => number;
}

/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output: (input: number) => number;
  der: (input: number) => number;
}

/** Function that computes a penalty cost for a given weight in the network. */
export interface RegularizationFunction {
  output: (weight: number) => number;
  der: (weight: number) => number;
}

/** Built-in error functions */
export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: number, target: number) =>
               0.5 * Math.pow(output - target, 2),
    der: (output: number, target: number) => output - target
  };
}

/** Polyfill for TANH */
(Math as any).tanh = (Math as any).tanh || function(x) {
  if (x === Infinity) {
    return 1;
  } else if (x === -Infinity) {
    return -1;
  } else {
    let e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }
};

/** Polyfill for SOFTPLUS */
(Math as any).softplus = (Math as any).softplus || function(x) {
  let threshold = 20;
  let beta = 1;
  if (x * beta > threshold) {
    return x;
  } else if (x === -Infinity) {
    return 0;
  } else {
    return 1 / beta * (Math as any).log(1 + (Math as any).exp(beta * x));
  }
};

/** Built-in activation functions */
export class Activations {
  public static TANH: ActivationFunction = {
    output: x => (Math as any).tanh(x),
    der: x => {
      let output = Activations.TANH.output(x);
      return 1 - output * output;
    },
    compileToJs: arg => `Math.tanh(${arg})`
  };
  public static RELU: ActivationFunction = {
    output: x => Math.max(0, x),
    der: x => x <= 0 ? 0 : 1,
    compileToJs: arg => `Math.max(0, ${arg})`
  };
  public static SIGMOID: ActivationFunction = {
    output: x => 1 / (1 + Math.exp(-x)),
    der: x => {
      let output = Activations.SIGMOID.output(x);
      return output * (1 - output);
    },
    compileToJs: arg => `(1 / (1 + Math.exp(-(${arg}))))`
  };
  public static LINEAR: ActivationFunction = {
    output: x => x,
    der: x => 1,
    compileToJs: arg => arg
  };
  public static SINE: ActivationFunction = {
    output: x => (Math as any).sin(x),
    der: x => (Math as any).cos(x),
    compileToJs: arg => `Math.sin(${arg})`
  };
  public static SINC: ActivationFunction = {
    output: x => x < 0.000001 ? 1 : (Math as any).sin(x) / x,
    der: x => (x*x) < 0.000001 ? 0 : (x * (Math as any).cos(x) - (Math as any).sin(x)) / (x*x),
    compileToJs: arg => `Math.sinc(${arg})`
  };
  public static MISH: ActivationFunction = {
    output: x => x * Activations.TANH.output((Math as any).softplus(x)),
    der: x => {
      let sig_x = Activations.SIGMOID.output(x);
      let tanh_sp_x = Activations.TANH.output((Math as any).softplus(x));
      return tanh_sp_x * x * sig_x * (1 - tanh_sp_x * tanh_sp_x);
    },
    compileToJs: arg => `mish(${arg})`
  };
  public static GELU: ActivationFunction = {
    output: x => 0.5 * x * (1 + (Math as any).tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))),
    der: x => {
      const tanhPart = (Math as any).tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)));
      return 0.5 * (1 + tanhPart) + 0.5 * x * (1 - Math.pow(tanhPart, 2)) * Math.sqrt(2 / Math.PI) * (1 + 3 * 0.044715 * Math.pow(x, 2));
    },
    compileToJs: arg => `gelu(${arg})`
  };
  public static LEAKY_RELU: ActivationFunction = {
    output: x => x >= 0 ? x : 0.01 * x,//TODO: make this a parameter
    der: x => x >= 0 ? 1 : 0.01,
    compileToJs: arg => `leakyrelu(${arg})`
  };
  public static PReLU: (alpha: number) => ActivationFunction = (alpha) => ({
    output: x => x >= 0 ? x : alpha * x,
    der: x => x >= 0 ? 1 : alpha,
    compileToJs: arg => `prelu(${arg})`
  });
}

/** Build-in regularization functions */
export class RegularizationFunction {
  public static L1: RegularizationFunction = {
    output: w => Math.abs(w),
    der: w => w < 0 ? -1 : (w > 0 ? 1 : 0)
  };
  public static L2: RegularizationFunction = {
    output: w => 0.5 * w * w,
    der: w => w
  };
}

/** Built-in weight quantization functions */
export class WeightQuantizationFunction {
  public static q16bit: WeightQuantizationFunction = {
    output: w => (Math as any).round(w * 65536) / 65536
  };
  public static q8bit: WeightQuantizationFunction = {
    output: w => (Math as any).round(w * 256) / 256
  };
  public static q4bit: WeightQuantizationFunction = {
    output: w => (Math as any).round(w * 16) / 16
  };
  public static q2bit: WeightQuantizationFunction = {
    output: w => (Math as any).round(w * 4) / 4
  };
}

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
export class Link {
  id: string;
  source: Node;
  dest: Node;
  weight = Math.random() - 0.5;
  isDead = false;
  /** Error derivative with respect to this weight. */
  errorDer = 0;
  /** Accumulated error derivative since the last update. */
  accErrorDer = 0;
  /** Number of accumulated derivatives since the last update. */
  numAccumulatedDers = 0;
  /** Per-parameter optimizer state. */
  optimizerState: OptimizerState = {};
//   regularization: RegularizationFunction;

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
//    * @param regularization The regularization function that computes the
//    *     penalty for this weight. If null, there will be no regularization.
   */
  constructor(source: Node, dest: Node, initZero?: boolean) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    if (initZero) {
      this.weight = 0;
    }
  }
}

/**
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param activation The activation function of every hidden node.
 * @param outputActivation The activation function for the output nodes.
 * @param regularization The regularization function that computes a penalty
 *     for a given weight (parameter) in the network. If null, there will be
 *     no regularization.
 * @param inputIds List of ids for the input nodes.
 */
export function buildNetwork(
    networkShape: number[], activation: ActivationFunction,
    outputActivation: ActivationFunction,
    inputIds: string[], initZero?: boolean): Node[][] {
  let numLayers = networkShape.length;
  let id = 1;
  /** List of layers, with each layer being a list of nodes. */
  let network: Node[][] = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1;
    let isInputLayer = layerIdx === 0;
    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = networkShape[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      let nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      let node = new Node(nodeId,
          isOutputLayer ? outputActivation : activation, initZero);
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node, initZero);
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }
      }
    }
  }
  return network;
}

/**
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * total input and output of each node in the network.
 *
 * @param network The neural network.
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @return The final output of the network.
 */
export function forwardProp(network: Node[][], inputs: number[],
    weightQuantizationFunction: WeightQuantizationFunction,
    layerNorm: boolean = false): number {
  let inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error("The number of inputs must match the number of nodes in" +
        " the input layer");
  }
  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    node.output = inputs[i];
  }
  let isOutputLayer: boolean;
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    isOutputLayer = layerIdx === network.length - 1;
    // Compute totalInput for all nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.totalInput = node.bias;
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let weight = weightQuantizationFunction ?
            weightQuantizationFunction.output(link.weight) : link.weight;
        node.totalInput += weight * link.source.output;
      }
    }
    // Apply layer normalization to hidden layers if enabled.
    if (layerNorm && !isOutputLayer && currentLayer.length > 1) {
      let mean = 0;
      for (let i = 0; i < currentLayer.length; i++) {
        mean += currentLayer[i].totalInput;
      }
      mean /= currentLayer.length;
      let variance = 0;
      for (let i = 0; i < currentLayer.length; i++) {
        let diff = currentLayer[i].totalInput - mean;
        variance += diff * diff;
      }
      variance /= currentLayer.length;
      let std = Math.sqrt(variance + 1e-8);
      for (let i = 0; i < currentLayer.length; i++) {
        currentLayer[i].totalInput = (currentLayer[i].totalInput - mean) / std;
      }
    }
    // Apply activation.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.output = node.activation.output(node.totalInput);
    }
  }
  return network[network.length - 1][0].output;
}

/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 */
export function backProp(network: Node[][], target: number,
    errorFunc: ErrorFunction): void {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  let outputNode = network[network.length - 1][0];
  outputNode.outputDer = errorFunc.der(outputNode.output, target);

  // Go through the layers backwards.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    let currentLayer = network[layerIdx];
    // Compute the error derivative of each node with respect to:
    // 1) its total input
    // 2) each of its input weights.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.inputDer = node.outputDer * node.activation.der(node.totalInput);
      node.accInputDer += node.inputDer;
      node.numAccumulatedDers++;
    }

    // Error derivative with respect to each weight coming into the node.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        link.errorDer = node.inputDer * link.source.output;
        link.accErrorDer += link.errorDer;
        link.numAccumulatedDers++;
      }
    }
    if (layerIdx === 1) {
      continue;
    }
    let prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      let node = prevLayer[i];
      // Compute the error derivative with respect to each node's output.
      node.outputDer = 0;
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        node.outputDer += output.weight * output.dest.inputDer;
      }
    }
  }
}

/** Optimizer state stored per-parameter (Link or Node bias). */
export interface OptimizerState {
  m?: number;  // first moment (momentum / Adam m)
  v?: number;  // second moment (RMSProp / Adam v)
  t?: number;  // step count (Adam)
}

/** Optimizer types. */
export enum OptimizerType {
  SGD = "sgd",
  MOMENTUM = "momentum",
  RMSPROP = "rmsprop",
  ADAM = "adam"
}

/** Hyperparameters for optimizers. */
export const OPTIMIZER_BETA1 = 0.9;
export const OPTIMIZER_BETA2 = 0.999;
export const OPTIMIZER_EPSILON = 1e-8;

/**
 * Applies an optimizer update step to a single parameter.
 * Returns the delta to subtract from the parameter.
 */
function optimizerDelta(
    grad: number, learningRate: number,
    optimizerType: OptimizerType, state: OptimizerState): number {
  switch (optimizerType) {
    case OptimizerType.SGD:
      return learningRate * grad;
    case OptimizerType.MOMENTUM: {
      state.m = state.m == null ? 0 : state.m;
      state.m = OPTIMIZER_BETA1 * state.m + (1 - OPTIMIZER_BETA1) * grad;
      return learningRate * state.m;
    }
    case OptimizerType.RMSPROP: {
      state.v = state.v == null ? 0 : state.v;
      state.v = OPTIMIZER_BETA2 * state.v + (1 - OPTIMIZER_BETA2) * grad * grad;
      return learningRate * grad / (Math.sqrt(state.v) + OPTIMIZER_EPSILON);
    }
    case OptimizerType.ADAM: {
      state.m = state.m == null ? 0 : state.m;
      state.v = state.v == null ? 0 : state.v;
      state.t = state.t == null ? 0 : state.t;
      state.t += 1;
      state.m = OPTIMIZER_BETA1 * state.m + (1 - OPTIMIZER_BETA1) * grad;
      state.v = OPTIMIZER_BETA2 * state.v + (1 - OPTIMIZER_BETA2) * grad * grad;
      let mHat = state.m / (1 - Math.pow(OPTIMIZER_BETA1, state.t));
      let vHat = state.v / (1 - Math.pow(OPTIMIZER_BETA2, state.t));
      return learningRate * mHat / (Math.sqrt(vHat) + OPTIMIZER_EPSILON);
    }
    default:
      return learningRate * grad;
  }
}

/**
 * Updates the weights of the network using the previously accumulated error
 * derivatives.
 */
export function updateWeights(network: Node[][], learningRate: number,
    regularization: RegularizationFunction, regularizationRate: number,
    optimizerType: OptimizerType = OptimizerType.SGD) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // Update the node's bias.
      if (node.numAccumulatedDers > 0) {
        let biasGrad = node.accInputDer / node.numAccumulatedDers;
        node.bias -= optimizerDelta(biasGrad, learningRate, optimizerType, node.biasOptimizerState);
        node.accInputDer = 0;
        node.numAccumulatedDers = 0;
      }
      // Update the weights coming into this node.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        let regulDer = regularization ?
            regularization.der(link.weight) : 0;
        if (link.numAccumulatedDers > 0) {
          let grad = link.accErrorDer / link.numAccumulatedDers;
          // Update the weight based on dE/dw.
          link.weight -= optimizerDelta(grad, learningRate, optimizerType, link.optimizerState);
          // Further update the weight based on regularization.
          let newLinkWeight = link.weight -
              (learningRate * regularizationRate) * regulDer;
          if (regularization === RegularizationFunction.L1 &&
              link.weight * newLinkWeight < 0) {
            // The weight crossed 0 due to the regularization term. Set it to 0.
            link.weight = 0;
            link.isDead = true;
          } else {
            link.weight = newLinkWeight;
          }
          link.accErrorDer = 0;
          link.numAccumulatedDers = 0;
        }
      }
    }
  }
}

/** Iterates over every node in the network/ */
export function forEachNode(network: Node[][], ignoreInputs: boolean,
    accessor: (node: Node) => any) {
  for (let layerIdx = ignoreInputs ? 1 : 0;
      layerIdx < network.length;
      layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      accessor(node);
    }
  }
}

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}

export function compileNetworkToJs(network: Node[][]): string {
  const inputLayer = network[0];
  let js = `function(${inputLayer.map(node => node.compileToJsName()).join(", ")}) {\n`;  
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      js += `  const ${node.compileToJsName()} = ${node.compileToJs()};\n`;
    }
  }
  js += `  return ${network[network.length - 1][0].compileToJsName()};\n`;
  js += `}`;
  return js;
}