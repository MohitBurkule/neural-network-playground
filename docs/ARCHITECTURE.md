# Architecture Overview

This document describes the module structure of the Neural Network Playground Extended Edition and traces the data flow through a single training step.

---

## Repository Layout

```
/
├── src/                  TypeScript source files
│   ├── nn.ts             Core ML engine
│   ├── playground.ts     Main UI / entry point
│   ├── state.ts          URL-serialisable application state + registration maps
│   ├── heatmap.ts        Canvas-based heatmap renderer
│   ├── linechart.ts      SVG line chart for loss over time
│   ├── dataset.ts        2D classification and regression dataset generators
│   ├── dataset3d.ts      3D dataset generators
│   ├── threeview.ts      three.js WebGL scene wrapper
│   ├── adversarial.ts    FGSM and PGD attack implementations
│   ├── unlearning.ts     Machine unlearning (gradient ascent + retrain baseline)
│   ├── customdataset.ts  CSV parser / serialiser for user data
│   ├── cnn.ts            CNN lab entry point
│   └── transformer.ts    Transformer lab entry point
├── labs/                 HTML templates for the standalone labs
├── dist/                 Compiled output (not checked in to source)
├── tests/                Jest unit tests
├── e2e/                  Playwright end-to-end tests
├── index.html            Main page template
├── styles.css            Light theme styles
├── styles_dark.css       Dark theme styles
└── package.json          Build scripts and dependencies
```

---

## Module Descriptions

### `nn.ts` — Core ML Engine

The neural network engine. No DOM dependencies; fully pure TypeScript.

**Key types and classes**

| Export | Description |
|---|---|
| `Node` | A single neuron. Stores `totalInput`, `output`, `inputDer`, `outputDer`, accumulated gradients, per-parameter optimizer state, and a `frozen` flag. |
| `Link` | A weighted connection between two `Node`s. Stores `weight`, `errorDer`, accumulated gradient, optimizer state, and `isDead` (set when L1 drives a weight to exactly 0). |
| `ActivationFunction` | Interface with `output(x)`, `der(x)`, and `compileToJs(arg)` methods. |
| `Activations` | Static class providing all built-in activations: `TANH`, `RELU`, `SIGMOID`, `LINEAR`, `SINE`, `SINC`, `MISH`, `GELU`, `LEAKY_RELU`, `PReLU(alpha)`. |
| `ErrorFunction` | Interface with `error(output, target)` and `der(output, target)`. `Errors.SQUARE` is the only built-in. |
| `RegularizationFunction` | Interface with `output(w)` and `der(w)`. `RegularizationFunction.L1` and `.L2` are provided. |
| `WeightQuantizationFunction` | Interface with `output(w)`. Implementations: `q16bit`, `q8bit`, `q4bit`, `q2bit`. |
| `OptimizerType` | Enum: `SGD`, `MOMENTUM`, `RMSPROP`, `ADAM`. |
| `OptimizerState` | Per-parameter state object `{ m?, v?, t? }` for first/second moment and step count. |

**Key functions**

| Function | Signature | Description |
|---|---|---|
| `buildNetwork` | `(shape, activation, outputActivation, inputIds, initZero?) → Node[][]` | Constructs a fully-connected network of the given shape. Returns a 2-D array of layers. |
| `forwardProp` | `(network, inputs, weightQuantizationFunction, layerNorm?) → number` | Runs one forward pass. Optionally applies weight quantization and/or layer normalization. Returns the scalar output. |
| `backProp` | `(network, target, errorFunc) → void` | Runs one backward pass. Accumulates `accInputDer` on each node and `accErrorDer` on each link. |
| `updateWeights` | `(network, learningRate, regularization, regularizationRate, optimizerType) → void` | Applies one optimizer step to all non-frozen parameters, then clears accumulated gradients. |
| `forEachNode` | `(network, ignoreInputs, accessor) → void` | Iterates over all nodes; used for freezing/unfreezing. |
| `compileNetworkToJs` | `(network) → string` | Emits a JavaScript function string representing the network's current weights (used for model export). |

---

### `state.ts` — Application State and Registration Maps

`State` is a plain TypeScript class whose properties map directly to URL hash parameters via `serialize()` / `deserializeState()`. This makes every configuration shareable as a URL.

**Registration maps** — these are the single source of truth for which options appear in the UI dropdowns:

```typescript
activations: { [name: string]: nn.ActivationFunction }
optimizers:  { [name: string]: nn.OptimizerType }
datasets:    { [name: string]: dataset.DataGenerator }
regDatasets: { [name: string]: dataset.DataGenerator }
regularizations:      { [name: string]: nn.RegularizationFunction }
weightQuantizations:  { [name: string]: nn.WeightQuantizationFunction }
```

`State.PROPS` is an array of `Property` descriptors that drives serialization; any new state variable added here will be automatically persisted in the URL.

---

### `playground.ts` — Main UI

The browser entry point bundled as `dist/bundle.js`. It:

1. Calls `State.deserializeState()` at startup to restore configuration from the URL hash.
2. Calls `buildNetwork()` to construct the network.
3. Wires D3 event handlers to all controls; every change updates `state` and triggers `reset()` or `generateData()`.
4. Runs the training loop via a `d3.timer` inside the `Player` class. Each tick calls `oneStep()`.
5. After every step, calls `updateUI()` which re-renders the heatmap, per-node thumbnails, loss chart, accuracy readout, and confusion matrix.

**Notable functions in `playground.ts`**

| Function | Purpose |
|---|---|
| `makeGUI()` | Attaches all D3 event listeners to DOM controls |
| `reset()` | Rebuilds the network, applies frozen-layer state, redraws the diagram |
| `generateData()` | Calls the active dataset generator, or generates 3D data, or parses custom CSV |
| `oneStep()` | Runs `forwardProp` + `backProp` + `updateWeights` for one mini-batch; handles adversarial augmentation |
| `updateUI()` | Redraws heatmap, thumbnails, line chart, accuracy, confusion matrix |
| `exportModel()` | Serialises network weights to JSON and triggers a download |
| `importModel()` | Parses uploaded JSON and calls `buildNetwork` with loaded weights |
| `unlearn()` | Delegates to `unlearning.forgetPoints`; updates readout with before/after metrics |
| `applyFrozenLayers()` | Iterates `state.frozenLayers` and sets `node.frozen` on matching layers |

---

### `heatmap.ts` — Decision Boundary Renderer

`HeatMap` class renders onto an HTML `<canvas>` element using a 100×100 density grid (`DENSITY = 100` in `playground.ts`). It also renders data point overlays. `reduceMatrix` is a helper that downsamples a fine-grained grid for the smaller per-node thumbnail heatmaps.

---

### `linechart.ts` — Loss Chart

`AppendingLineChart` class renders a multi-series SVG chart. It appends a new data point each epoch and keeps a rolling window, so the x-axis always shows the most recent history. Two series are drawn: train loss and test loss.

---

### `dataset.ts` — 2D Dataset Generators

Exports the `Example2D` type `{ x, y, label }` and `DataGenerator = (numSamples, noise) => Example2D[]`. All classification and regression generators return `Example2D[]` with labels in {−1, +1} (classification) or continuous values (regression). Uses D3 scale utilities for noise scaling.

---

### `dataset3d.ts` — 3D Dataset Generators

Exports `Example3D = { x, y, z, label }` and `DataGenerator3D`. Generators: `classifyTwoGaussBlobs`, `classifyConcentricSpheres`, `classifyHelix`, `classifySwissRoll`. In 3D mode, `playground.ts` projects each point's (x, y) to the network; the z coordinate is visual only.

---

### `threeview.ts` — 3D Renderer

`ThreeView` wraps a `THREE.WebGLRenderer` scene. It manages a rotation group for drag-to-rotate interaction, renders data points as coloured spheres (pink for +1, blue for −1), and can optionally render a semi-transparent decision boundary mesh. THREE.js is loaded as a global from a `<script>` tag in `index.html`.

---

### `adversarial.ts` — Attack Implementations

Provides `fgsm` and `pgd` as pure functions operating on `Node[][]` and a raw `{x, y, label}` point. The key internal helper `inputGradients` reconstructs `dLoss/d(x)` and `dLoss/d(y)` by walking the `inputDer` values of the first hidden layer back through their incoming link weights — exactly mirroring what `backProp` would do for an additional input layer.

Note: these functions call `forwardProp` and `backProp` internally, bypassing weight quantization and layer norm (pass `null` for the quantization function) since attack quality degrades under quantization.

---

### `unlearning.ts` — Machine Unlearning

Two strategies:

1. **`forgetPoints(network, forgetSet, retainSet, opts)`** — modifies the network in-place. Each step: one gradient-ascent step on a sampled forget-set point, followed by `retainRatio` gradient-descent steps on a sampled retain-set batch. Returns `{ before, after }` `ForgetMetrics`.

2. **`retrainWithout(shape, retainSet, forgetSet, opts)`** — builds a fresh network with `buildNetwork`, trains it on `retainSet` only, returns the new network and `ForgetMetrics`.

`ForgetMetrics` captures MSE loss and classification accuracy on both sets so the caller can display the before/after comparison.

---

### `customdataset.ts` — CSV I/O

Pure utility functions with no DOM or network dependencies:

- `parseCSV(text)` — splits on newlines, parses three-column rows, auto-skips header, normalises label 0→−1, collects per-row errors.
- `serializeCSV(examples)` — writes a header row then one `x,y,label` row per example.
- `trainTestSplit(examples, trainRatio)` — Fisher-Yates shuffle then split.

---

## Training Data Flow

A single training step in `oneStep()` follows this sequence:

```
1. Sample a mini-batch of size state.batchSize from trainData.

2. [Optional] If adversarialTraining is enabled:
   For each point in the batch, call fgsm() or pgd() to generate
   an adversarial twin. Append twins to the batch.

3. For each point in the (possibly augmented) batch:
   a. Construct the input feature vector from (x, y) using the
      active INPUTS functions (x, y, x², y², x·y, sin(x), sin(y), …).
   b. forwardProp(network, inputs, weightQuantizationFunction, layerNorm)
      — sets node.totalInput and node.output for every node
   c. backProp(network, label, Errors.SQUARE)
      — accumulates node.accInputDer and link.accErrorDer

4. updateWeights(network, learningRate, regularization,
                 regularizationRate, optimizerType)
   — for each non-frozen node/link:
     a. average the accumulated gradient
     b. compute optimizerDelta (SGD/Momentum/RMSProp/Adam)
     c. subtract delta from bias or weight
     d. apply regularization penalty
     e. if L1 drives weight through zero, mark link.isDead = true
     f. clear accumulators

5. Increment iter; call updateUI().
```

---

## 2D Mode vs 3D Mode

| Aspect | 2D Mode | 3D Mode |
|---|---|---|
| Data | `Example2D[]` from `dataset.ts` | `Example3D[]` from `dataset3d.ts` |
| Visualisation | D3 canvas heatmap (`HeatMap`) | three.js WebGL scene (`ThreeView`) |
| Input to network | Feature vector computed from (x, y) | Only (x, y) coordinates; z is not fed to the network |
| Loss / accuracy | Computed over all 2D train/test points | Computed by iterating `threeData`, projecting to (x, y) |
| Decision boundary | 100×100 heatmap grid | Optional: a sampled mesh surface added to the three.js scene |
| Confusion matrix | Updated normally | Updated from 3D point labels vs predictions |

The key coupling point is `generateData()` in `playground.ts`: if `state.threeD` is true it calls a `THREE_GENERATORS[state.threeDDataset]` function and stores the result in the module-level `threeData` variable; otherwise it calls the active 2D generator.
