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
   * Whether this node is frozen. Frozen nodes do not have their bias or
   * incoming link weights updated during training (used for fine-tuning).
   */
  frozen = false;

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
  /** Hinge loss for targets in {-1, +1}. */
  public static HINGE: ErrorFunction = {
    error: (output: number, target: number) =>
               Math.max(0, 1 - output * target),
    der: (output: number, target: number) =>
               (1 - output * target > 0) ? -target : 0
  };
  /**
   * Logistic / cross-entropy loss. Maps output and target from (-1,1) to (0,1)
   * probabilities (p = (v+1)/2), so it works with the playground's tanh-style
   * outputs without changing any existing behavior elsewhere.
   */
  public static LOGLOSS: ErrorFunction = {
    error: (output: number, target: number) => {
      let eps = 1e-7;
      let p = Math.min(1 - eps, Math.max(eps, (output + 1) / 2));
      let t = (target + 1) / 2;
      return -(t * Math.log(p) + (1 - t) * Math.log(1 - p));
    },
    der: (output: number, target: number) => {
      let eps = 1e-7;
      let p = Math.min(1 - eps, Math.max(eps, (output + 1) / 2));
      let t = (target + 1) / 2;
      // d/d(output): chain through p = (output+1)/2 (dp/doutput = 1/2).
      return 0.5 * (p - t) / (p * (1 - p));
    }
  };
  /** Huber loss (smooth L1) with delta = 1. */
  public static HUBER: ErrorFunction = {
    error: (output: number, target: number) => {
      let d = output - target;
      let delta = 1;
      return Math.abs(d) <= delta ?
          0.5 * d * d : delta * (Math.abs(d) - 0.5 * delta);
    },
    der: (output: number, target: number) => {
      let d = output - target;
      let delta = 1;
      return Math.abs(d) <= delta ? d : delta * Math.sign(d);
    }
  };
  /** Absolute (L1) error. */
  public static ABSOLUTE: ErrorFunction = {
    error: (output: number, target: number) => Math.abs(output - target),
    der: (output: number, target: number) =>
               output > target ? 1 : (output < target ? -1 : 0)
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
    output: x => (x * x) < 0.000001 ? 1 : (Math as any).sin(x) / x,
    der: x => (x*x) < 0.000001 ? 0 : (x * (Math as any).cos(x) - (Math as any).sin(x)) / (x*x),
    // Emit a `sinc` helper (defined in compileNetworkToJs' prelude) rather than
    // `Math.sinc`, which is not a standard Math method.
    compileToJs: arg => `sinc(${arg})`
  };
  public static MISH: ActivationFunction = {
    output: x => x * Activations.TANH.output((Math as any).softplus(x)),
    der: x => {
      let sig_x = Activations.SIGMOID.output(x);
      let tanh_sp_x = Activations.TANH.output((Math as any).softplus(x));
      // d/dx[x * tanh(softplus(x))] = tanh(sp) + x * sigmoid(x) * (1 - tanh(sp)^2)
      return tanh_sp_x + x * sig_x * (1 - tanh_sp_x * tanh_sp_x);
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
    // Inline the per-instance alpha so the emitted JS is self-contained
    // (a shared `prelu` helper could not capture differing alpha values).
    compileToJs: arg => `((${arg}) >= 0 ? (${arg}) : ${alpha} * (${arg}))`
  });
  public static ELU: ActivationFunction = {
    output: x => x >= 0 ? x : Math.exp(x) - 1,
    der: x => x >= 0 ? 1 : Math.exp(x),
    compileToJs: arg => `((${arg}) >= 0 ? (${arg}) : Math.exp(${arg}) - 1)`
  };
  public static SELU: ActivationFunction = {
    output: x => {
      let a = 1.6732632423543772, s = 1.0507009873554805;
      return s * (x >= 0 ? x : a * (Math.exp(x) - 1));
    },
    der: x => {
      let a = 1.6732632423543772, s = 1.0507009873554805;
      return s * (x >= 0 ? 1 : a * Math.exp(x));
    },
    compileToJs: arg => `(1.0507009873554805 * ((${arg}) >= 0 ? (${arg}) : 1.6732632423543772 * (Math.exp(${arg}) - 1)))`
  };
  public static SWISH: ActivationFunction = {
    output: x => x / (1 + Math.exp(-x)),
    der: x => {
      let sig = 1 / (1 + Math.exp(-x));
      return sig + x * sig * (1 - sig);
    },
    compileToJs: arg => `((${arg}) / (1 + Math.exp(-(${arg}))))`
  };
  public static SOFTPLUS: ActivationFunction = {
    output: x => (Math as any).softplus(x),
    der: x => 1 / (1 + Math.exp(-x)),
    compileToJs: arg => `Math.log(1 + Math.exp(${arg}))`
  };
  public static SOFTSIGN: ActivationFunction = {
    output: x => x / (1 + Math.abs(x)),
    der: x => 1 / Math.pow(1 + Math.abs(x), 2),
    compileToJs: arg => `((${arg}) / (1 + Math.abs(${arg})))`
  };
  public static HARD_SIGMOID: ActivationFunction = {
    output: x => Math.max(0, Math.min(1, 0.2 * x + 0.5)),
    der: x => (x > -2.5 && x < 2.5) ? 0.2 : 0,
    compileToJs: arg => `Math.max(0, Math.min(1, 0.2 * (${arg}) + 0.5))`
  };
  public static HARD_TANH: ActivationFunction = {
    output: x => Math.max(-1, Math.min(1, x)),
    der: x => (x > -1 && x < 1) ? 1 : 0,
    compileToJs: arg => `Math.max(-1, Math.min(1, ${arg}))`
  };
  public static HARD_SWISH: ActivationFunction = {
    output: x => x * Math.max(0, Math.min(1, (x + 3) / 6)),
    der: x => {
      if (x <= -3) return 0;
      if (x >= 3) return 1;
      return (2 * x + 3) / 6;
    },
    compileToJs: arg => `((${arg}) * Math.max(0, Math.min(1, ((${arg}) + 3) / 6)))`
  };
  public static RELU6: ActivationFunction = {
    output: x => Math.min(6, Math.max(0, x)),
    der: x => (x > 0 && x < 6) ? 1 : 0,
    compileToJs: arg => `Math.min(6, Math.max(0, ${arg}))`
  };
  public static BENT_IDENTITY: ActivationFunction = {
    output: x => (Math.sqrt(x * x + 1) - 1) / 2 + x,
    der: x => x / (2 * Math.sqrt(x * x + 1)) + 1,
    compileToJs: arg => `((Math.sqrt((${arg}) * (${arg}) + 1) - 1) / 2 + (${arg}))`
  };
  public static GAUSSIAN: ActivationFunction = {
    output: x => Math.exp(-x * x),
    der: x => -2 * x * Math.exp(-x * x),
    compileToJs: arg => `Math.exp(-(${arg}) * (${arg}))`
  };
  public static SNAKE: ActivationFunction = {
    output: x => x + Math.pow(Math.sin(x), 2),
    der: x => 1 + 2 * Math.sin(x) * Math.cos(x),
    compileToJs: arg => `((${arg}) + Math.pow(Math.sin(${arg}), 2))`
  };
  public static ARCTAN: ActivationFunction = {
    output: x => Math.atan(x),
    der: x => 1 / (1 + x * x),
    compileToJs: arg => `Math.atan(${arg})`
  };
  public static ISRU: ActivationFunction = {
    // Inverse square root unit (alpha = 1).
    output: x => x / Math.sqrt(1 + x * x),
    der: x => Math.pow(1 / Math.sqrt(1 + x * x), 3),
    compileToJs: arg => `((${arg}) / Math.sqrt(1 + (${arg}) * (${arg})))`
  };
  public static EXPONENTIAL_LINEAR: ActivationFunction = {
    // Smooth exponential-linear blend: x for x>=0, scaled exp ramp below.
    output: x => x >= 0 ? x : 0.5 * (Math.exp(x) - 1),
    der: x => x >= 0 ? 1 : 0.5 * Math.exp(x),
    compileToJs: arg => `((${arg}) >= 0 ? (${arg}) : 0.5 * (Math.exp(${arg}) - 1))`
  };
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

/** Weight initialization schemes. */
export enum WeightInit {
  RANDOM_UNIFORM = "random-uniform",
  XAVIER = "xavier",
  HE = "he",
  LECUN = "lecun",
  ZEROS = "zeros",
  ORTHOGONAL = "orthogonal"
}

/** Standard normal sample via Box-Muller. */
function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Re-initializes all link weights (and biases) using the chosen scheme.
 * fanIn is the number of incoming links to a node. RANDOM_UNIFORM reproduces
 * the original default (uniform in [-0.5, 0.5]).
 */
export function applyWeightInit(network: Node[][], scheme: WeightInit): void {
  if (scheme == null || scheme === WeightInit.RANDOM_UNIFORM) {
    return;  // Keep the default random weights assigned at construction.
  }
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let layer = network[layerIdx];
    for (let i = 0; i < layer.length; i++) {
      let node = layer[i];
      let fanIn = node.inputLinks.length || 1;
      let fanOut = node.outputs.length || 1;
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        switch (scheme) {
          case WeightInit.XAVIER:
            link.weight = randn() * Math.sqrt(2 / (fanIn + fanOut));
            break;
          case WeightInit.HE:
            link.weight = randn() * Math.sqrt(2 / fanIn);
            break;
          case WeightInit.LECUN:
            link.weight = randn() * Math.sqrt(1 / fanIn);
            break;
          case WeightInit.ZEROS:
            link.weight = 0;
            break;
          case WeightInit.ORTHOGONAL:
            // Orthogonal-lite: scaled normal that approximately preserves
            // variance across the layer (cheap stand-in for true QR ortho).
            link.weight = randn() / Math.sqrt(fanIn);
            break;
        }
      }
    }
  }
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
    layerNorm: boolean = false, dropout: number = 0,
    training: boolean = false, batchNorm: boolean = false): number {
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
    // Apply batch normalization (over this batch's single-example stats, i.e.
    // normalize across the units of the layer) to hidden layers if enabled.
    // Kept intentionally simple: uses the current activations' mean/variance.
    if (batchNorm && !isOutputLayer && currentLayer.length > 1) {
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
    // Inverted dropout on hidden layers during training only.
    if (training && dropout > 0 && dropout < 1 && !isOutputLayer) {
      let scale = 1 / (1 - dropout);
      for (let i = 0; i < currentLayer.length; i++) {
        if (Math.random() < dropout) {
          currentLayer[i].output = 0;
        } else {
          currentLayer[i].output *= scale;
        }
      }
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
  vMax?: number;  // max second moment (AMSGrad)
  acc?: number;  // accumulated sq grad (Adagrad)
  accDelta?: number;  // accumulated sq update (Adadelta)
}

/** Optimizer types. */
export enum OptimizerType {
  SGD = "sgd",
  MOMENTUM = "momentum",
  RMSPROP = "rmsprop",
  ADAM = "adam",
  NESTEROV = "nesterov",
  ADAGRAD = "adagrad",
  ADADELTA = "adadelta",
  AMSGRAD = "amsgrad",
  NADAM = "nadam",
  ADAMW = "adamw"
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
    case OptimizerType.NESTEROV: {
      // Nesterov accelerated momentum (look-ahead form).
      state.m = state.m == null ? 0 : state.m;
      let prev = state.m;
      state.m = OPTIMIZER_BETA1 * state.m + grad;
      return learningRate * (OPTIMIZER_BETA1 * prev + (1 + OPTIMIZER_BETA1) * grad);
    }
    case OptimizerType.ADAGRAD: {
      state.acc = (state.acc == null ? 0 : state.acc) + grad * grad;
      return learningRate * grad / (Math.sqrt(state.acc) + OPTIMIZER_EPSILON);
    }
    case OptimizerType.ADADELTA: {
      let rho = 0.95;
      state.acc = state.acc == null ? 0 : state.acc;
      state.accDelta = state.accDelta == null ? 0 : state.accDelta;
      state.acc = rho * state.acc + (1 - rho) * grad * grad;
      let update = Math.sqrt(state.accDelta + OPTIMIZER_EPSILON) /
          Math.sqrt(state.acc + OPTIMIZER_EPSILON) * grad;
      state.accDelta = rho * state.accDelta + (1 - rho) * update * update;
      // Adadelta is self-scaling; learningRate acts as a global multiplier.
      return learningRate * update;
    }
    case OptimizerType.AMSGRAD: {
      state.m = state.m == null ? 0 : state.m;
      state.v = state.v == null ? 0 : state.v;
      state.vMax = state.vMax == null ? 0 : state.vMax;
      state.t = (state.t == null ? 0 : state.t) + 1;
      state.m = OPTIMIZER_BETA1 * state.m + (1 - OPTIMIZER_BETA1) * grad;
      state.v = OPTIMIZER_BETA2 * state.v + (1 - OPTIMIZER_BETA2) * grad * grad;
      state.vMax = Math.max(state.vMax, state.v);
      let mHat = state.m / (1 - Math.pow(OPTIMIZER_BETA1, state.t));
      return learningRate * mHat / (Math.sqrt(state.vMax) + OPTIMIZER_EPSILON);
    }
    case OptimizerType.NADAM: {
      state.m = state.m == null ? 0 : state.m;
      state.v = state.v == null ? 0 : state.v;
      state.t = (state.t == null ? 0 : state.t) + 1;
      state.m = OPTIMIZER_BETA1 * state.m + (1 - OPTIMIZER_BETA1) * grad;
      state.v = OPTIMIZER_BETA2 * state.v + (1 - OPTIMIZER_BETA2) * grad * grad;
      let mHat = state.m / (1 - Math.pow(OPTIMIZER_BETA1, state.t));
      let vHat = state.v / (1 - Math.pow(OPTIMIZER_BETA2, state.t));
      let mNes = OPTIMIZER_BETA1 * mHat +
          (1 - OPTIMIZER_BETA1) * grad / (1 - Math.pow(OPTIMIZER_BETA1, state.t));
      return learningRate * mNes / (Math.sqrt(vHat) + OPTIMIZER_EPSILON);
    }
    case OptimizerType.ADAMW: {
      // Same step as Adam; decoupled weight decay applied separately in updateWeights.
      state.m = state.m == null ? 0 : state.m;
      state.v = state.v == null ? 0 : state.v;
      state.t = (state.t == null ? 0 : state.t) + 1;
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
    optimizerType: OptimizerType = OptimizerType.SGD,
    gradClip: number = 0, weightDecay: number = 0) {
  // Clip a gradient to [-gradClip, gradClip] when clipping is enabled.
  let clip = (g: number) => {
    if (gradClip > 0) {
      if (g > gradClip) return gradClip;
      if (g < -gradClip) return -gradClip;
    }
    return g;
  };
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      // Skip frozen nodes entirely: their bias and incoming link weights stay
      // fixed during training. We still clear accumulated derivatives so they
      // don't leak into a later update if the node is unfrozen.
      if (node.frozen) {
        node.accInputDer = 0;
        node.numAccumulatedDers = 0;
        for (let j = 0; j < node.inputLinks.length; j++) {
          node.inputLinks[j].accErrorDer = 0;
          node.inputLinks[j].numAccumulatedDers = 0;
        }
        continue;
      }
      // Update the node's bias.
      if (node.numAccumulatedDers > 0) {
        let biasGrad = clip(node.accInputDer / node.numAccumulatedDers);
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
          let grad = clip(link.accErrorDer / link.numAccumulatedDers);
          // Update the weight based on dE/dw.
          link.weight -= optimizerDelta(grad, learningRate, optimizerType, link.optimizerState);
          // Decoupled weight decay (AdamW-style): pull weight toward zero
          // proportionally to the learning rate, independent of the gradient.
          if (weightDecay > 0) {
            link.weight -= learningRate * weightDecay * link.weight;
          }
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

/**
 * Definitions for the non-standard helper functions that some activations'
 * `compileToJs` emit by name (rather than fully inlining). Each entry maps a
 * helper "token" that may appear in the compiled body to the JS source line
 * that defines it. `compileNetworkToJs` prepends the definitions for whichever
 * helpers a given network actually references, so the emitted snippet is fully
 * self-contained (copy-paste runnable, no ReferenceError).
 *
 * The math of every helper mirrors the corresponding `output` function above:
 *   - sinc(x)      = (x*x < 1e-6) ? 1 : sin(x)/x      (SINC, with x==0 -> 1)
 *   - mish(x)      = x * tanh(softplus(x))            (Activations.MISH)
 *   - gelu(x)      = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))  (GELU)
 *   - leakyrelu(x) = x >= 0 ? x : 0.01 * x            (LEAKY_RELU, slope 0.01)
 *   - softplus(x)  = x > 20 ? x : log(1 + exp(x))     (matches runtime polyfill;
 *                    the x>20 branch avoids Math.exp overflow). Pulled in only
 *                    when mish is used, since mish's definition references it.
 *
 * Definitions are ordered so that any helper appears after the helpers it
 * depends on (softplus before mish).
 */
const JS_HELPERS: {token: string, def: string}[] = [
  {token: "softplus",
   def: "const softplus = x => x > 20 ? x : Math.log(1 + Math.exp(x));"},
  {token: "mish",
   def: "const mish = x => x * Math.tanh(softplus(x));"},
  {token: "gelu",
   def: "const gelu = x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * " +
        "(x + 0.044715 * Math.pow(x, 3))));"},
  {token: "leakyrelu",
   def: "const leakyrelu = x => x >= 0 ? x : 0.01 * x;"},
  {token: "sinc",
   def: "const sinc = x => (x * x) < 0.000001 ? 1 : Math.sin(x) / x;"},
];

/**
 * Builds the prelude of helper definitions needed by the compiled network
 * body. Only helpers whose token actually appears in `body` are emitted (so
 * the snippet stays minimal). `mish` additionally pulls in `softplus` because
 * its definition references it.
 */
function compileJsHelperPrelude(body: string): string {
  let needsSoftplus = body.indexOf("mish(") !== -1;
  let lines: string[] = [];
  for (let i = 0; i < JS_HELPERS.length; i++) {
    let helper = JS_HELPERS[i];
    let used = body.indexOf(helper.token + "(") !== -1 ||
        (helper.token === "softplus" && needsSoftplus);
    if (used) {
      lines.push(helper.def);
    }
  }
  return lines.join("\n");
}

export function compileNetworkToJs(network: Node[][]): string {
  const inputLayer = network[0];
  let body = `function(${inputLayer.map(node => node.compileToJsName()).join(", ")}) {\n`;
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      body += `  const ${node.compileToJsName()} = ${node.compileToJs()};\n`;
    }
  }
  body += `  return ${network[network.length - 1][0].compileToJsName()};\n`;
  body += `}`;
  // Prepend definitions for any non-standard helpers the body references so the
  // emitted JS is fully self-contained. Wrapped in an IIFE that returns the
  // network function, so the whole string is a single expression: it can be
  // eval'd directly and pasted as `const net = <snippet>;`.
  let prelude = compileJsHelperPrelude(body);
  if (!prelude) {
    return body;
  }
  let indented = prelude.split("\n").map(line => "  " + line).join("\n");
  return `(function() {\n${indented}\n  return ${body};\n})()`;
}