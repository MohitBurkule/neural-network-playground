# Architecture Overview

This document describes the module structure of the Neural Network Playground Extended Edition and traces the data flow through a single training step.

---

## Repository Layout

```
/
├── src/                      TypeScript source files
│   ├── nn.ts                 Core ML engine
│   ├── playground.ts         Main UI / entry point (bundle.js)
│   ├── state.ts              URL-serializable application state + registration maps
│   ├── heatmap.ts            Canvas-based heatmap renderer
│   ├── linechart.ts          SVG line chart for loss over time
│   ├── dataset.ts            2D classification and regression dataset generators
│   ├── dataset3d.ts          3D dataset generators (16 scenes)
│   ├── threeview.ts          three.js WebGL scene wrapper
│   ├── adversarial.ts        Attack implementations (FGSM, PGD, targeted, DeepFool-lite)
│   ├── unlearning.ts         Machine unlearning (gradient ascent + retrain baseline)
│   ├── customdataset.ts      CSV parser / serializer for user data
│   ├── cnn.ts                CNN lab entry point
│   ├── cnn_data.ts           Synthetic 8×8 shape dataset for the CNN and autoencoder labs
│   ├── transformer.ts        Transformer lab entry point
│   ├── autoencoder.ts        Autoencoder lab entry point
│   ├── rnn.ts                RNN lab entry point
│   ├── gan.ts                GAN lab entry point
│   ├── clustering.ts         Clustering lab entry point
│   ├── rl.ts                 RL Gridworld lab entry point
│   ├── dtree.ts              Decision Tree / Random Forest lab entry point
│   ├── dimred.ts             PCA / t-SNE lab entry point
│   ├── svm.ts                SVM lab entry point
│   ├── glm.ts                Linear / Logistic / Naive Bayes lab entry point
│   └── gp.ts                 Gaussian Process lab entry point
├── labs/                     HTML templates for the standalone lab pages
│   ├── cnn.html
│   ├── transformer.html
│   ├── autoencoder.html
│   ├── rnn.html
│   ├── gan.html
│   ├── clustering.html
│   ├── rl.html
│   ├── dtree.html
│   ├── dimred.html
│   ├── svm.html
│   ├── glm.html
│   └── gp.html
├── dist/                     Compiled output (not checked in to source)
│   ├── bundle.js             Main playground JS bundle
│   ├── bundle.css            Material Design Lite + styles
│   ├── styles.css            Light theme
│   ├── styles_dark.css       Dark theme
│   ├── lib.js                Vendored MDL + seedrandom
│   ├── bundlecnn.js          CNN lab bundle
│   ├── bundletransformer.js  Transformer lab bundle
│   ├── bundleautoencoder.js  Autoencoder lab bundle
│   ├── bundlernn.js          RNN lab bundle
│   ├── bundlegan.js          GAN lab bundle
│   ├── bundleclustering.js   Clustering lab bundle
│   ├── bundlerl.js           RL Gridworld lab bundle
│   ├── bundledtree.js        Decision Tree lab bundle
│   ├── bundledimred.js       PCA / t-SNE lab bundle
│   ├── bundlesvm.js          SVM lab bundle
│   ├── bundleglm.js          GLM lab bundle
│   ├── bundlegp.js           Gaussian Process lab bundle
│   ├── index.html            Main page
│   └── *.html                One HTML file per lab (copied from labs/)
├── tests/                    Jest unit tests
├── e2e/                      Playwright end-to-end tests
├── index.html                Main page template
├── styles.css                Light theme source
├── styles_dark.css           Dark theme source
├── package.json              Build scripts and npm dependencies
├── tsconfig.json             TypeScript compiler configuration
└── playwright.config.ts      Playwright E2E configuration
```

---

## Build System

### Bundler: Browserify + tsify

Every TypeScript entry point is compiled and bundled using **Browserify** with the **tsify** plugin (which invokes the TypeScript compiler directly). There is no Webpack, Rollup, or esbuild involved.

```bash
browserify src/playground.ts -p [tsify] > dist/bundle.js
```

Each lab has its own independent bundle command:

```bash
browserify src/cnn.ts -p [tsify] > dist/bundlecnn.js
# …and so on for each lab
```

### D3 — UMD alias

D3 v7 is listed as an npm dependency, but the `"browser"` field in `package.json` aliases the `d3` import to the pre-built UMD bundle:

```json
"browser": {
  "d3": "./node_modules/d3/dist/d3.min.js"
}
```

This ensures Browserify picks up the UMD build (which registers `window.d3`) rather than the ES-module tree, avoiding any ESM-to-CJS bridging issues.

### three.js — CDN script tag

three.js is **not** bundled. Instead, `index.html` loads it from a CDN script tag:

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
```

`src/threeview.ts` accesses `THREE` as a global (declared via `src/seedrandom.d.ts`-style ambient typing). The lab HTML files that do not use three.js do not include the CDN tag.

### mathjs

`mathjs` is included as an npm dependency (used for expression evaluation in the custom-feature inline formula input in `playground.ts`).

### CSS pipeline

CSS is concatenated (not minified or processed by PostCSS):

```bash
concat node_modules/material-design-lite/material.min.css styles.css > dist/bundle.css
concat styles_dark.css > dist/styles_dark.css
concat styles.css > dist/styles.css
```

### Build scripts

| Script | What it does |
|---|---|
| `npm run build` | `prep` + `build-js` + `build-css` + `build-darkcss` + `build-lightcss` + `build-html` |
| `npm run build-labs` | Builds all 12 lab bundles and copies `labs/*.html` into `dist/` |
| `npm run build-all` | `build` + `build-labs` (builds everything; use this for deployment) |
| `npm run watch` | `prep` + concurrent `watch-js` + `watch-css` + `watch-html` (no lab watch) |
| `npm run serve` | `npx serve dist/` on port 3000 |
| `npm run serve-watch` | Concurrent `serve` + `watch` |
| `npm run clean` | `rimraf dist` |

For Cloudflare Pages: build command `npm run build-all`, output directory `dist`, `NODE_VERSION=22`.

---

## Module Descriptions

### `nn.ts` — Core ML Engine

No DOM dependencies; fully pure TypeScript. Handles the entire forward and backward pass.

**Key types and classes**

| Export | Description |
|---|---|
| `Node` | A single neuron. Stores `totalInput`, `output`, `inputDer`, `outputDer`, accumulated gradients, per-parameter `biasOptimizerState`, and a `frozen` flag. |
| `Link` | A weighted edge between two `Node`s. Stores `weight`, `errorDer`, accumulated gradient, `optimizerState`, and `isDead` (set when L1 drives weight to 0). |
| `ActivationFunction` | Interface: `output(x)`, `der(x)`, `compileToJs(arg)`. 24 implementations in `Activations`. |
| `ErrorFunction` | Interface: `error(output, target)`, `der(output, target)`. 5 implementations in `Errors`. |
| `RegularizationFunction` | Interface: `output(w)`, `der(w)`. `L1` and `L2` implementations. |
| `WeightQuantizationFunction` | Interface: `output(w)`. `q16bit`, `q8bit`, `q4bit`, `q2bit` implementations. |
| `WeightInit` | Enum: `RANDOM_UNIFORM`, `XAVIER`, `HE`, `LECUN`, `ZEROS`, `ORTHOGONAL`. |
| `OptimizerType` | Enum: `SGD`, `MOMENTUM`, `RMSPROP`, `ADAM`, `NESTEROV`, `ADAGRAD`, `ADADELTA`, `AMSGRAD`, `NADAM`, `ADAMW`. |
| `OptimizerState` | Per-parameter state: `{ m?, v?, t?, vMax?, acc?, accDelta? }`. |

**Key functions**

| Function | Description |
|---|---|
| `buildNetwork(shape, activation, outputActivation, inputIds, initZero?)` | Constructs a fully-connected network; returns `Node[][]`. |
| `applyWeightInit(network, scheme)` | Re-initializes all link weights and biases using the chosen `WeightInit` scheme. |
| `forwardProp(network, inputs, quantFn, layerNorm?, dropout?, training?, batchNorm?)` | One forward pass. Applies weight quantization, batch norm, layer norm, and dropout as configured. Returns scalar output. |
| `backProp(network, target, errorFunc)` | One backward pass. Accumulates `accInputDer` on nodes and `accErrorDer` on links. |
| `updateWeights(network, lr, regularization, regularizationRate, optimizerType, gradClip?, weightDecay?)` | Applies one optimizer step to all non-frozen parameters; handles gradient clipping and decoupled weight decay; clears accumulators. |
| `compileNetworkToJs(network)` | Emits a standalone JavaScript function string for the current network. |

---

### `state.ts` — Application State and Registration Maps

`State` is a plain TypeScript class whose properties map directly to URL hash parameters via `serialize()` / `deserializeState()`.

**Registration maps** (single source of truth for UI dropdowns):

```typescript
activations:         { [name: string]: nn.ActivationFunction }     // 24 entries
optimizers:          { [name: string]: nn.OptimizerType }           // 10 entries
lossFunctions:       { [name: string]: nn.ErrorFunction }           // 5 entries
weightInits:         { [name: string]: nn.WeightInit }              // 6 entries
lrSchedules:         { [name: string]: string }                     // 6 entries
datasets:            { [name: string]: dataset.DataGenerator }      // 48+ entries
regDatasets:         { [name: string]: dataset.DataGenerator }      // 17 entries
regularizations:     { [name: string]: nn.RegularizationFunction }  // 3 entries
weightQuantizations: { [name: string]: nn.WeightQuantizationFunction } // 5 entries
```

`State.PROPS` is an array of `Property` descriptors that drives serialization. Any new state variable added to `PROPS` is automatically persisted in the URL hash. Properties include all hyperparameters, dataset selection, adversarial config, augmentation flags, and display toggles.

---

### `playground.ts` — Main UI

The browser entry point bundled as `dist/bundle.js`. It is the largest file in the codebase, having grown to incorporate all feature modules inline. Key responsibilities:

1. Calls `State.deserializeState()` at startup to restore configuration from the URL hash.
2. Calls `buildNetwork()` → `applyWeightInit()` to construct the network.
3. Wires D3 event handlers to all controls; every change updates `state` and triggers `reset()` or `generateData()`.
4. Runs the training loop via `d3.timer` inside the `Player` class. Each tick calls `oneStep()`.
5. After every step calls `updateUI()` which re-renders the heatmap, thumbnails, loss chart, status bar, and all analysis panels.

**Notable functions and modules within `playground.ts`**

| Function / Module | Purpose |
|---|---|
| `makeGUI()` | Attaches all D3 event listeners to DOM controls |
| `reset()` | Rebuilds the network, applies frozen-layer state, redraws diagram |
| `generateData()` | Calls the active 2D or 3D dataset generator, or parses custom CSV |
| `oneStep()` | Runs forward/back/update for one mini-batch; handles adversarial augmentation, mixup, label noise, input jitter, gradient noise, class weighting, and LR scheduling |
| `updateUI()` | Redraws heatmap, thumbnails, line chart, status bar, confusion matrix, and all open analysis panels |
| `exportModel()` | Serializes network weights to JSON and triggers a download |
| `importModel()` | Parses pasted/loaded JSON and calls `buildNetwork` with loaded weights |
| `applyFrozenLayers()` | Iterates `state.frozenLayers` and sets `node.frozen` on matching hidden layers |
| Analysis section | Computes and renders ROC, PR, calibration, histograms, gradient flow, weight-magnitude, loss landscape slice |
| Interpretability section | Saliency, occlusion, drop-feature, PDP, ablation, actmax, counterfactual, hidden-PCA, surrogate tree, entropy, what-if, k-NN, contribution |
| Adversarial panel | Calls `adversarial.ts` functions; robustness sweep; saliency readout |
| Unlearning panel | Calls `unlearning.ts` functions; brush-select; relearn-time probe |
| Experiments section | Leaderboard, overlay chart, A/B compare, grid search, run I/O |
| UX toolbar | Presets, share link, snapshot/restore, speed, auto-stop, run-N |
| a11y toolbar | Language (i18n strings), high-contrast, reduced-motion, compact, UI scale, fast-training, redraw-K, perf meter |

---

### `heatmap.ts` — Decision Boundary Renderer

`HeatMap` class renders onto an HTML `<canvas>` element using a configurable density grid (100×100 for the main heatmap). It also renders data-point overlays (filled circles, orange/blue). `reduceMatrix` downsamples a dense grid for the smaller per-node thumbnail heatmaps.

---

### `linechart.ts` — Loss Chart

`AppendingLineChart` class renders a multi-series SVG chart. It appends a new data point each epoch and keeps a rolling window so the x-axis always shows recent history. Two series are drawn: train loss (grey) and test loss (black).

---

### `dataset.ts` — 2D Dataset Generators

Exports `Example2D = { x, y, label }` and `DataGenerator = (numSamples, noise) => Example2D[]`. All generators return labels in {−1, +1} (classification) or continuous values (regression). Uses `Math.seedrandom` for reproducibility.

The file contains 48+ classification generators and 17 regression generators. All are registered in `state.ts`.

---

### `dataset3d.ts` — 3D Dataset Generators

Exports `Example3D = { x, y, z, label }` and `DataGenerator3D`. Contains 16 generators registered in the `THREE_GENERATORS` map in `playground.ts`:

Two Gaussian Blobs, Concentric Spheres, Helix, Swiss Roll, Linked Rings, 3D Checkerboard Cube, Double Helix, 3D XOR (Octants), Shell vs Core, 3D S-Curve, Trefoil Knot, Möbius Band, Stacked Planes, 3D Spiral Tower, Octant Checker, Sphere Grid.

In 3D mode `playground.ts` projects each point's (x, y) to the network; z is visual only.

---

### `threeview.ts` — 3D Renderer

`ThreeView` wraps a `THREE.WebGLRenderer` scene. Manages a rotation group for drag-to-rotate interaction, renders data points as coloured spheres (pink for +1, blue for −1), and can optionally render a semi-transparent decision-boundary mesh. THREE.js is loaded as a global from the CDN `<script>` tag in `index.html`.

---

### `adversarial.ts` — Attack Implementations

Provides five adversarial attacks and supporting analysis utilities:

- **`fgsm`** — Fast Gradient Sign Method: analytic gradient through `inputGradients` (walks `inputDer` from the first hidden layer back to the raw x/y inputs via link weights).
- **`pgd`** — Projected Gradient Descent: iterative FGSM with L∞ projection.
- **`randomNoiseAttack`** — Uniform L∞ noise baseline; no gradient used.
- **`targetedFgsm`** — Iterative FGSM toward the opposite label; accepts a pluggable `GradFn` so it works with finite-difference gradients for networks using derived input features.
- **`deepFoolLite`** — Gradient-direction steps until the prediction sign flips.

Supporting: `rawLossGrad` (builds a finite-difference `{loss, grad}` pair for a 2-input network), `perturbationBudget` (mean/max L2 and L∞ distances between clean and adversarial sets).

---

### `unlearning.ts` — Machine Unlearning

Two strategies exposed as pure functions:

1. **`forgetPoints(network, forgetSet, retainSet, opts)`** — modifies the network in-place. Each step: one gradient-ascent step on a sampled forget-set point followed by `retainRatio` gradient-descent steps on a sampled retain-set batch. Returns `{ before, after }` `ForgetMetrics`.

2. **`retrainWithout(shape, retainSet, forgetSet, opts)`** — builds a fresh network via `buildNetwork`, trains it on `retainSet` only, returns the new network and `ForgetMetrics`.

`ForgetMetrics` captures MSE loss and classification accuracy on both sets so the UI can display before/after comparisons.

---

### `customdataset.ts` — CSV I/O

Pure utility functions with no DOM or network dependencies:

- `parseCSV(text)` — splits on newlines, parses three-column rows, auto-skips header, normalizes label 0→−1, collects per-row errors.
- `serializeCSV(examples)` — writes a header row then one `x,y,label` row per example.
- `trainTestSplit(examples, trainRatio)` — Fisher-Yates shuffle then split.

---

### Lab entry points (`cnn.ts`, `transformer.ts`, `autoencoder.ts`, `rnn.ts`, `gan.ts`, `clustering.ts`, `rl.ts`, `dtree.ts`, `dimred.ts`, `svm.ts`, `glm.ts`, `gp.ts`)

Each lab is a self-contained TypeScript module with no shared state with `playground.ts`. They import D3 (via the same `d3` UMD alias from `package.json`) and implement their own algorithm, renderer, and event handling. They are compiled to separate bundles (`dist/bundle<name>.js`) and served from their own HTML pages (`dist/<name>.html`, copied from `labs/<name>.html`).

See [docs/LABS.md](LABS.md) for per-lab descriptions.

---

## Test Setup

### Unit tests — Jest

Test files live in `tests/`. Run with:

```bash
npm test
```

Uses `ts-jest` to compile TypeScript on the fly. No DOM emulation is needed because all tested modules (`nn.ts`, `dataset.ts`, `dataset3d.ts`, `adversarial.ts`, `unlearning.ts`, `customdataset.ts`) are pure TypeScript with no browser APIs.

| Test file | Coverage |
|---|---|
| `tests/nn.test.ts` | Core network: build, forward, backward, weight update, optimizer steps |
| `tests/nn_extra.test.ts` | Additional activation and optimizer edge cases |
| `tests/dataset.test.ts` | 2D dataset generators: point counts, label range, noise behavior |
| `tests/dataset3d.test.ts` | 3D dataset generators |
| `tests/adversarial.test.ts` | FGSM, PGD, targeted, DeepFool, perturbationBudget |
| `tests/unlearning.test.ts` | forgetPoints and retrainWithout metrics |
| `tests/customdataset.test.ts` | CSV parse/serialize round-trips, label normalization, error reporting |

### End-to-end tests — Playwright

Test files live in `e2e/`. Run with:

```bash
npm run test:e2e
```

Configuration is in `playwright.config.ts`. Tests exercise the main playground UI (play/pause, dataset switching, activation changes, adversarial panel) and key feature interactions.

---

## Training Data Flow

A single training step in `oneStep()` follows this sequence:

```
1. Compute effective learning rate from base LR × LR schedule factor.

2. Sample a mini-batch of size state.batchSize from trainData.
   If epochShuffle is enabled, re-shuffle at epoch boundaries.

3. [Optional] Data augmentation on the batch:
   a. Label noise: randomly flip labels with probability labelNoise.
   b. Input jitter: add Gaussian noise N(0, inputJitter) to (x, y).
   c. Mixup: replace pairs (x_i, x_j) with λx_i + (1−λ)x_j and mixed labels.

4. [Optional] Adversarial augmentation (adversarialTraining enabled):
   For each point in the batch, call the selected attack to generate an
   adversarial twin. Append twins to the batch.

5. For each point in the (possibly augmented) batch:
   a. Construct the input feature vector from (x, y) using the active
      INPUTS functions (x, y, x², y², x·y, sin(x), sin(y), custom…).
   b. forwardProp(network, inputs, weightQuantizationFn,
                  layerNorm, dropout, training=true, batchNorm)
      — sets node.totalInput and node.output for every node.
   c. backProp(network, label, errorFunc)
      — accumulates node.accInputDer and link.accErrorDer.

6. updateWeights(network, effectiveLR, regularization,
                 regularizationRate, optimizerType, gradClip, weightDecay)
   — for each non-frozen node/link:
     a. average the accumulated gradient
     b. [optional] clip to [−gradClip, gradClip]
     c. [optional] add gradient noise N(0, gradientNoise) (annealed)
     d. compute optimizerDelta
     e. subtract delta from bias or weight
     f. [optional] decoupled weight decay: w -= lr · weightDecay · w
     g. apply regularization penalty
     h. if L1 drives weight through zero, mark link.isDead = true
     i. clear accumulators

7. Increment iter; call updateUI() (with redraw throttled by perf-redraw-k).
```

---

## 2D Mode vs 3D Mode

| Aspect | 2D Mode | 3D Mode |
|---|---|---|
| Data | `Example2D[]` from `dataset.ts` | `Example3D[]` from `dataset3d.ts` |
| Visualization | D3 canvas heatmap (`HeatMap`) | three.js WebGL scene (`ThreeView`) |
| Input to network | Feature vector from (x, y) | Only (x, y); z is not fed to the network |
| Loss / accuracy | Computed over all 2D train/test points | Iterated over `threeData`, projecting to (x, y) |
| Decision boundary | 100×100 heatmap grid | Optional sampled mesh added to three.js scene |
| Confusion matrix | Updated normally | Updated from 3D point labels vs predictions |

The key coupling point is `generateData()` in `playground.ts`: when `state.threeD` is true it calls a `THREE_GENERATORS[state.threeDDataset]` function and stores the result in the module-level `threeData` variable; otherwise it calls the active 2D generator.
