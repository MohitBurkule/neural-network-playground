# Feature Catalog

This document enumerates every feature available in the Neural Network Playground Extended Edition, with a short explanation and the location in the UI where it can be found.

---

## Training Controls

### Play / Pause
Start or stop the training loop. The button is in the top toolbar. Each "tick" of the loop runs one mini-batch forward pass, backward pass, and weight update.

### Step
Advance training by exactly one mini-batch without entering continuous play mode. Useful for watching the decision boundary evolve one step at a time.

### Reset
Re-initialises all network weights (randomly or to zero, depending on the **Init Zero** toggle) and clears the loss chart. Found in the top toolbar alongside Play/Pause.

### Learning Rate
Dropdown in the left panel. Controls the global step size multiplier applied to every weight update regardless of optimizer choice.

### Batch Size
Dropdown in the left panel. Number of training examples used to compute each gradient update. Smaller batches are noisier but update more frequently.

### Noise
Slider in the data panel. Adds Gaussian noise to generated dataset samples, making the classification boundary harder to learn.

### Ratio of Training to Test Data
Slider (percTrainData) in the data panel. Splits the generated dataset into training and held-out test sets.

### Seed
Text input in the data panel. Seeds both the random data generator and weight initialiser so runs are reproducible.

---

## Network Architecture

### Number of Hidden Layers
+/- buttons in the network diagram area. Add or remove hidden layers up to the supported maximum.

### Nodes per Layer
+/- buttons on each hidden layer column. Adjust width of each hidden layer independently.

### Init Zero
Toggle in the settings panel. When enabled, all weights and biases start at zero rather than random uniform values.

---

## Activation Functions

Selected from the **Activation** dropdown in the left panel. Applies to all hidden-layer nodes; the output node always uses a linear activation for regression or a fixed activation for classification.

| Name | Behaviour |
|---|---|
| **ReLU** | `max(0, x)` — piecewise linear, sparse activations |
| **Tanh** | Smooth S-curve in (−1, 1) — the default |
| **Sigmoid** | Smooth S-curve in (0, 1) |
| **Linear** | Identity — passes the pre-activation through unchanged |
| **Sine** | `sin(x)` — periodic activation, useful for wave-like targets |
| **Sinc** | `sin(x)/x` (1 at x=0) — a peaked periodic function |
| **Mish** | `x · tanh(softplus(x))` — smooth, non-monotonic |
| **GELU** | Gaussian Error Linear Unit — approximated via tanh |
| **Leaky ReLU** | `max(0.01x, x)` — avoids dead neurons |
| **PReLU** | Parametric ReLU with α=0.2 — learned negative slope |

---

## Optimizers

Selected from the **Optimizer** dropdown. Each optimizer maintains per-parameter state on every `Link` and `Node` bias.

| Name | Notes |
|---|---|
| **SGD** | Vanilla stochastic gradient descent: `w -= lr * grad` |
| **Momentum** | Exponential moving average of gradients (β1 = 0.9) |
| **RMSProp** | Divides by root-mean-squared gradient history (β2 = 0.999) |
| **Adam** | Combines Momentum and RMSProp with bias correction (β1=0.9, β2=0.999, ε=1e-8) |

---

## Regularization

Dropdown in the left panel, paired with a **Regularization Rate** slider.

- **None** — no penalty applied
- **L1** — adds `|w|` penalty; drives small weights exactly to zero (weights can become "dead")
- **L2** — adds `0.5w²` penalty; shrinks weights smoothly toward zero

---

## Weight Quantization

Dropdown in the left panel. Simulates reduced-precision inference by snapping each weight to a fixed grid during the forward pass (gradients remain full precision).

| Setting | Effective bits | Grid step |
|---|---|---|
| None | 32-bit float | — |
| 16-bit | 16 | 1/65536 |
| 8-bit | 8 | 1/256 |
| 4-bit | 4 | 1/16 |
| 2-bit | 2 | 1/4 |

---

## Layer Normalization

Toggle in the left panel (**Layer Norm**). When enabled, each hidden layer's pre-activation values are normalised to zero mean and unit variance before the activation function is applied, using a small epsilon (1e-8) for numerical stability.

---

## Input Features

Checkboxes in the **Features** panel on the left side of the network diagram. Each selected feature becomes an additional input node.

| Feature | Formula |
|---|---|
| x | raw x coordinate |
| y | raw y coordinate |
| x² | `x * x` |
| y² | `y * y` |
| x·y | `x * y` |
| sin(x) | `Math.sin(x)` |
| sin(y) | `Math.sin(y)` |
| sptheta | `Math.atan2(y, x)` — polar angle |
| spradius | `Math.sqrt(x²+y²)` — polar radius |
| \|x−y\|−3 | `Math.abs(x-y) - 3` |
| \|x+y\|−3 | `Math.abs(x+y) - 3` |

---

## Datasets (Classification)

Thumbnail picker in the **Data** panel. Each dataset generates `Example2D` points with label ∈ {−1, +1}.

| Key | Description |
|---|---|
| circle | Points inside a circle labelled +1, outside −1 |
| xor | Four quadrants alternating sign — the classic XOR problem |
| gauss | Two Gaussian blobs at (2,2) and (−2,−2) |
| spiral | Two interleaved spirals |
| concentric-circles | Multiple concentric rings alternating in label |
| biclusters | Two axis-aligned rectangular clusters |
| moons | Two crescent-shaped clusters |
| hash | Checkerboard / hash-grid pattern |
| three | Digit "3" pattern derived from MNIST |

---

## Datasets (Regression)

Selected when **Problem Type** is set to Regression.

| Key | Description |
|---|---|
| reg-plane | A tilted plane: label = x + y |
| reg-gauss | Gaussian bump centred at origin |
| reg-sine-wave | Sinusoidal surface |
| reg-maximum | `max(x, y)` |
| reg-argmax | Returns 0 or 1 depending on which of x/y is larger |
| reg-friedman1 | Friedman #1 benchmark (5 inputs, uses x and y among others) |
| reg-friedman2 | Friedman #2 benchmark |
| reg-friedman3 | Friedman #3 benchmark |

---

## Custom CSV Dataset

Button in the **Data** panel labeled **Load CSV**. Accepts text in `x,y,label` format (one row per example). Lines starting with `#` are treated as comments; a header row is auto-skipped. Labels 1 and −1 are accepted; 0 is mapped to −1. Parse errors are reported row-by-row without aborting.

---

## 3D Dataset Mode

Toggle **3D Mode** in the data panel (state property `threeD`). Replaces the 2D heatmap canvas with a three.js WebGL scene. The network still receives 2D inputs; the 3D coordinates are projected to (x, y) for the forward pass, and the z-axis is visual only. Drag to rotate, scroll to zoom.

### 3D Datasets

| Key | Description |
|---|---|
| blobs | Two Gaussian blobs centred at (−2,−2,−2) and (2,2,2) |
| spheres | Concentric spheres — inner label +1, outer shell −1 |
| helix | Two interleaved 3D helices |
| swissroll | Swiss roll manifold, label by which half of the roll |

---

## Metrics and Visualization

### Loss Line Chart
A real-time SVG chart below the main diagram showing training loss (grey) and test loss (black) over epochs. Rendered by `AppendingLineChart`.

### Accuracy Readout
Percentage accuracy for both train and test sets, updated every epoch. Located in the output panel on the right.

### Confusion Matrix
A 2×2 confusion matrix (TP, FP, FN, TN) for classification problems, refreshed every epoch alongside the accuracy readout. Located below the accuracy display.

### Decision Boundary Heatmap
A 100×100 canvas grid that colours each pixel by the network's output — orange for negative, blue for positive, white at the decision boundary. Rendered by `HeatMap`.

### Per-node Activation Thumbnails
Small heatmap thumbnails on each hidden-layer node show the node's activation surface across the input domain, giving insight into what each neuron has learned.

---

## Adversarial Sampling

Controls in the **Adversarial** panel (bottom-right area).

### FGSM (Fast Gradient Sign Method)
Single-step attack. Perturbs each training point by `epsilon * sign(∇_x Loss)`. The gradient with respect to the raw (x, y) inputs is computed analytically by walking from the first hidden layer back to the input layer after a standard `backProp` call.

### PGD (Projected Gradient Descent)
Multi-step attack. Iteratively applies FGSM-like steps (default 10 steps, step size epsilon/4) and projects back to the L∞ ball of radius epsilon around the original point after each step.

### Adversarial Training (Defensive)
Toggle **Adversarial Training** in the adversarial panel. When enabled, each training mini-batch is augmented with adversarially perturbed copies of the same points using the selected method and epsilon. The network learns to be robust to those perturbations.

### Epsilon
Slider that controls the perturbation magnitude for both FGSM and PGD.

---

## Machine Unlearning

Panel labeled **Unlearn** (bottom area of the page).

### Forget Set Selection
Click data points in the training set to mark them as the "forget set". The rest become the "retain set".

### Gradient-Ascent Forgetting
Button **Forget (gradient ascent)**. Runs `forgetPoints` from `unlearning.ts`: takes gradient-ascent steps on the forget set (maximising loss on those points) while simultaneously running gradient-descent steps on the retain set to preserve accuracy. Configurable via the **Steps** input.

### Retrain from Scratch
Button **Retrain without forget set**. Builds a fresh network with the same architecture and trains it only on the retain set. This is the gold-standard unlearning baseline.

### Unlearning Metrics
After either unlearning operation the UI displays before/after accuracy on both the forget set and the retain set.

---

## Fine-Tuning and Model I/O

Controls in the **Fine-Tune** panel.

### Freeze / Unfreeze Layers
A list of checkboxes — one per hidden layer. Frozen layers have their weights and biases held constant during `updateWeights`; only unfrozen layers are trained.

### Freeze All But Last
Button that freezes every hidden layer except the final one — a common transfer-learning pattern.

### Unfreeze All
Button that clears all frozen layers, resuming full training.

### Export Model
Button **Export JSON**. Downloads a JSON file containing the network weights, biases, architecture (shape), active input features, and frozen-layer configuration.

### Import Model
Button **Import JSON**. Loads a previously exported JSON file, rebuilds the network with the saved weights, and restores frozen-layer state.

---

## Advanced Labs

Accessed via links on the main page or directly in the browser.

### CNN Visualizer (`dist/cnn.html`)
A standalone lab (entry point `src/cnn.ts`) that visualises a from-scratch convolutional neural network: filters, feature maps, pooling layers, and the classification head.

### Transformer Visualizer (`dist/transformer.html`)
A standalone lab (entry point `src/transformer.ts`) that visualises the attention mechanism: token embeddings, multi-head self-attention weight matrices, and the resulting attended representations.
