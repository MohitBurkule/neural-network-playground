# Neural Network Playground — Extended Edition

An interactive, browser-based visualization of feedforward neural networks, built on top of the original [Google / TensorFlow Neural Network Playground](https://playground.tensorflow.org). This fork extends it with a large set of additional features covering modern training techniques, adversarial robustness, machine unlearning, fine-tuning, 3D datasets, experiment tracking, interpretability tools, and twelve standalone advanced labs.

![preview](dist/preview.png)

---

## Features at a Glance

### Core Training
- **Activation functions** — 24 choices: ReLU, Tanh, Sigmoid, Linear, GELU, Leaky ReLU, PReLU, Sine, Sinc, Mish, ELU, SELU, Swish/SiLU, Softplus, Softsign, Hard Sigmoid, Hard Tanh, Hard Swish, ReLU6, Bent Identity, Gaussian, Snake, ArcTan, ISRU, Exponential Linear
- **Optimizers** — 10 choices: SGD, Momentum, RMSProp, Adam, Nesterov Momentum, Adagrad, Adadelta, AMSGrad, Nadam, AdamW (all with per-parameter state)
- **Loss functions** — Square (MSE), Hinge, Log loss / Cross-entropy, Huber, Absolute (L1)
- **Weight initialization** — Random uniform, Xavier/Glorot, He/Kaiming, LeCun, Zeros, Orthogonal-lite
- **Learning rate schedules** — Constant, Step decay, Exponential decay, Cosine annealing, Warmup + decay, 1cycle-lite
- **Regularization** — L1, L2 with adjustable rate
- **Weight quantization** — 16-bit, 8-bit, 4-bit, 2-bit simulation (forward pass only)
- **Normalization** — Batch Norm, Layer Norm (per hidden layer)
- **Regularization extras** — Dropout (inverted), gradient clipping, weight decay (decoupled, AdamW-style)

### Training Methodology
- **Data augmentation** — Mixup, input jitter (Gaussian noise on inputs), gradient noise (annealed)
- **Label noise** — randomly flip a configurable fraction of training labels
- **Class weighting** — inverse-frequency class weights applied to gradients
- **Epoch shuffle** — re-shuffle training data at the start of each epoch
- **Early stopping** — configurable patience; stops when test loss plateaus
- **Run N epochs** — execute a fixed number of steps then pause
- **Auto-stop** — halt when training loss delta drops below a threshold
- **Speed control** — steps per animation frame (1–50)
- **Re-shuffle split** — new train/test split without regenerating data
- **k-fold cross validation** — configurable k; reports per-fold accuracy
- **LR finder** — sweeps learning rates and plots loss vs LR to identify a good range
- **Ensemble training** — train N models (with optional bagging); reports mean accuracy
- **Weight perturbation probe** — add Gaussian noise to weights and measure accuracy degradation

### Datasets
- **Classification (2D)** — 48 datasets including Circle, XOR, Two Gaussians, Spiral, Concentric Circles, Biclusters, Moons, Hash, MNIST-Three, Checkerboard, Quadrant Blobs, Concentric Rings, Three/Four-Arm Spiral, Tight Spiral, Clean Moons, Nested U, Gaussian Mixture, Diagonal Stripes, Sine Boundary, Circle in Square, Cross, S-Curve, Pinwheel, Islands, Ring vs Center, Gaussian Quantiles, Anisotropic Blobs, Random-Label Blobs, Target Rings, Spiral Galaxy, Yin-Yang, Smiley, Grid Blobs, Interleaving Waves, Blob in Ring, Triangle vs Circle, Gaussian Cross, Noisy XOR-4, Crescent Pair, Dartboard, Comb, Diagonal Checker, Cluster Chain, and more
- **Regression (2D)** — 17 targets: Plane, Gaussian, Sine Wave, Maximum, ArgMax, Friedman 1/2/3, Ripple, Saddle, Gaussian Bump, Staircase, sin(x)cos(y), Radial, |x|+|y|, Step Circle, Waves
- **3D datasets** — 16 scenes rendered with three.js WebGL: Two Gaussian Blobs, Concentric Spheres, Helix, Swiss Roll, Linked Rings, 3D Checkerboard Cube, Double Helix, 3D XOR, Shell vs Core, 3D S-Curve, Trefoil Knot, Mobius Band, Stacked Planes, 3D Spiral Tower, Octant Checker, Sphere Grid
- **Custom CSV** — paste `x,y,label` data directly in the page; labels 1 or −1 accepted

### Input Feature Engineering
Toggle derived features: x, y, x², y², x·y, sin(x), sin(y), and two additional custom inputs (atan2(y,x), |x−y|, etc.). A **"Custom feature"** button lets you define an arbitrary formula inline.

### Metrics and Visualization
- Real-time **train/test loss** line chart
- **Accuracy** readout (train and test) updated every epoch
- **Confusion matrix** (classification)
- Decision boundary **heatmap** (100×100 canvas grid)
- Per-node **activation thumbnails** (hover to enlarge)
- Compact **status bar** (epoch, loss, accuracy, steps/sec)
- **ROC curve** and **Precision-Recall curve** with AUC / average-precision
- **Calibration / reliability diagram**
- **Weight, bias, activation, and confidence histograms**
- **Gradient flow** bar chart (mean |grad| per layer)
- **Per-layer mean |weight| over time** chart
- **Loss landscape 1D slice** (compute on demand)
- Export: decision boundary PNG, training history CSV, metrics snapshot JSON

### Interpretability Tools
- **Input saliency field** — gradient magnitude at every point in the 2D grid
- **Occlusion sensitivity** — per-feature accuracy drop when that input is zeroed
- **Drop-feature importance** — retrain-free feature removal test
- **Partial dependence plots** — marginal effect of x and y on the output
- **Neuron ablation** — zero out a selected neuron and observe boundary change
- **Activation maximization** — find the 2D input that maximally activates a neuron
- **Counterfactual search** — find the nearest opposite-class input to a clicked point
- **Hidden-activation PCA** — 2D PCA of the last hidden layer's activations
- **Decision-tree surrogate** — fit a shallow tree to the network's predictions
- **Confidence contours / entropy** — iso-confidence lines overlaid on the heatmap
- **What-if point inspector** — click a point to see network activations and prediction
- **k-NN baseline** — accuracy of a simple k-nearest-neighbor classifier on the same data
- **Per-feature contribution** — decompose the network output for a clicked point

### Advanced Features
- **Adversarial sampling** — FGSM, PGD, Random-noise baseline, Targeted (least-likely), DeepFool-lite; configurable epsilon; robustness curve sweep; saliency at last clicked point
- **Adversarial training** — mix adversarial examples into each training batch (defensive)
- **Machine unlearning** — gradient-ascent forgetting (forget orange, forget blue, forget misclassified, brush-select, forget nearest N, relearn-time probe) plus fine-tune-on-retain variant and retrain-without-forgotten baseline
- **Fine-tuning: freeze layers** — per-layer freeze checkboxes; "Freeze all but last" button; unfrozen layers train normally
- **Model export/import** — save network weights, architecture, and config to JSON; re-load to continue training

### UX / Sharing
- **Preset model zoo** — curated configurations selectable from a dropdown
- **Copy share link** — one-click URL with the full state encoded in the hash
- **Snapshot / Restore** — save and restore weights in-memory without a file
- **Randomize weights** — re-initialize without changing architecture or data
- **Dark mode** toggle (top-right switch)
- **Keyboard shortcuts** — Space (play/pause), S (step), R (reset), D (regenerate data), ? (shortcuts help)
- **Fullscreen output** — expand the heatmap panel

### Accessibility / i18n / Performance
- **Language selector** — English, Español, Français, हिन्दी, 中文
- **High contrast** mode
- **Reduced motion** mode
- **Compact mode** — hide advanced panels
- **UI scale** slider (80%–150%)
- **Fast training** mode — run training off the main render path
- **Redraw every K steps** — control repaint frequency at high training speed
- **Steps/sec · FPS · budget** meter

### Experiment Tracking
- **Save run / leaderboard** — store named runs with final metrics; sort by test accuracy, test loss, train accuracy, or epoch
- **Overlay comparison** — plot loss or accuracy curves for multiple runs on one chart
- **A/B compare** — side-by-side metric table for any two saved runs
- **Mini grid search** — sweep a small hyperparameter grid and view a result table
- **Export/import runs** — persist the leaderboard as JSON

---

## Labs

Twelve standalone algorithm labs are bundled alongside the main playground. Each is a self-contained HTML page with its own TypeScript bundle.

| Lab | File | Description |
|---|---|---|
| **CNN** | `dist/cnn.html` | From-scratch convolutional neural network: Conv→ReLU→Pool→Conv→ReLU→Pool→FC→Softmax. Visualizes filters, feature maps, and pooling layers; trains on synthetic 8×8 shape images. |
| **Transformer** | `dist/transformer.html` | Single/multi-head transformer block with full backprop. Visualizes token embeddings, Q/K/V projections, attention weight matrices, and the attended representations; trains on toy sequence tasks. |
| **Autoencoder** | `dist/autoencoder.html` | Fully-connected autoencoder (encoder → 2D bottleneck → decoder). Visualizes original vs reconstructed 8×8 shape images, the 2D latent space scatter, a latent-space grid decode, and the reconstruction loss curve. |
| **RNN** | `dist/rnn.html` | Elman RNN trained on a sequence echo/delay task (predict input delayed by k steps). Visualizes hidden-state heatmap over time, weight matrices, and per-step prediction vs target. |
| **GAN** | `dist/gan.html` | Minimax GAN with MLP generator and discriminator. Visualizes generated 2D point distributions, discriminator decision boundary, and separate generator/discriminator loss curves. |
| **Clustering** | `dist/clustering.html` | Interactive unsupervised clustering lab: k-means (with centroid trails), DBSCAN (core/border/noise roles), and GMM with EM. Supports blobs, moons, circles, anisotropic, and uniform datasets. |
| **RL Gridworld** | `dist/rl.html` | Tabular Q-learning and SARSA on a 7×10 editable grid with walls, pits, a goal, and a start cell. Visualizes the Q-table as colored arrows, the agent trajectory, and cumulative reward over episodes. |
| **Decision Tree / Random Forest** | `dist/dtree.html` | Axis-aligned decision tree (Gini or entropy criterion) and random forest. Visualizes the split tree structure, the 2D decision boundary, and per-class accuracy; configurable depth, min-samples, and number of trees. |
| **PCA / t-SNE** | `dist/dimred.html` | Dimensionality reduction lab. Runs PCA (exact, via power iteration) or t-SNE (Barnes-Hut-lite) on synthetic high-dimensional datasets and renders the 2D embedding with class-colored points. |
| **SVM** | `dist/svm.html` | Support Vector Machine with SMO-lite solver. Supports linear, polynomial, and RBF kernels with configurable C and gamma. Visualizes the margin, support vectors, and decision boundary. |
| **Linear / Logistic / Naive Bayes** | `dist/glm.html` | Generalized linear models lab: ordinary least-squares linear regression, logistic regression (gradient descent), and Gaussian Naive Bayes. Shows the fit line/boundary and class probabilities. |
| **Gaussian Process** | `dist/gp.html` | GP regression with exact inference (Cholesky). Supports RBF, Matérn-3/2, and periodic kernels with configurable length-scale, signal variance, and noise variance. Visualizes the posterior mean, 95% credible band, and sampled functions. |

---

## Getting Started Locally

```bash
# 1. Install dependencies
npm install

# 2. Build the main playground
npm run build

# 3. Serve from the dist/ directory (http://localhost:3000 by default)
npm run serve
```

For a fast edit-refresh cycle during development:

```bash
npm run serve-watch
```

This starts a live server and recompiles TypeScript, HTML, and CSS whenever a source file changes.

To build the main playground **and** all twelve labs:

```bash
npm run build-all
```

---

## Running Tests

### Unit tests (Jest)

```bash
npm test
```

Tests live in `tests/` and cover the core ML engine (`nn.ts`), dataset generators, adversarial attacks, custom-dataset CSV parsing, and machine unlearning.

### End-to-end tests (Playwright)

```bash
npm run test:e2e
```

End-to-end tests live in `e2e/` and test the main playground UI and key feature interactions.

---

## Deployment

### Cloudflare Pages (recommended)

| Setting | Value |
|---|---|
| Build command | `npm run build-all` |
| Output directory | `dist` |
| Environment variable | `NODE_VERSION=22` |

### Wrangler direct upload

```bash
npx wrangler pages deploy dist --project-name <your-project-name>
```

---

## Architecture Overview

All source files live in `src/`. Compiled output goes to `dist/`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a full module-by-module breakdown.

| Module | Role |
|---|---|
| `nn.ts` | Core ML engine: `Node`, `Link`, `buildNetwork`, `forwardProp`, `backProp`, `updateWeights`. Defines all activations, optimizers, loss functions, regularizers, weight initializers, and weight quantizers. |
| `playground.ts` | Main UI entry point bundled as `dist/bundle.js`. Wires D3 controls, the heatmap, line chart, analysis panels, interpretability tools, adversarial panel, unlearning panel, experiment tracker, and the training loop. |
| `state.ts` | URL-hash-serializable `State` class. Contains all registration maps (`activations`, `optimizers`, `datasets`, `regDatasets`, `lossFunctions`, `weightInits`, `lrSchedules`, `weightQuantizations`). |
| `heatmap.ts` | Canvas-based `HeatMap`; renders the decision boundary and per-node thumbnails. |
| `linechart.ts` | SVG `AppendingLineChart`; streams train/test loss over epochs. |
| `dataset.ts` | All 2D classification and regression dataset generators. |
| `dataset3d.ts` | 3D dataset generators (16 scenes). |
| `threeview.ts` | `ThreeView` wrapping a three.js WebGL scene. Drag-to-rotate, scroll-to-zoom. |
| `adversarial.ts` | FGSM, PGD, random-noise, targeted FGSM, DeepFool-lite attack implementations; perturbation-budget analysis. |
| `unlearning.ts` | `forgetPoints` (gradient-ascent forgetting + retain-set correction) and `retrainWithout` (retrain-from-scratch baseline). |
| `customdataset.ts` | CSV parser/serializer (`parseCSV`, `serializeCSV`, `trainTestSplit`) for user-supplied datasets. |
| `cnn.ts` | CNN lab entry point — bundled to `dist/bundlecnn.js`. |
| `transformer.ts` | Transformer lab entry point — bundled to `dist/bundletransformer.js`. |
| `autoencoder.ts` | Autoencoder lab entry point — bundled to `dist/bundleautoencoder.js`. |
| `rnn.ts` | RNN lab entry point — bundled to `dist/bundlernn.js`. |
| `gan.ts` | GAN lab entry point — bundled to `dist/bundlegan.js`. |
| `clustering.ts` | Clustering lab entry point — bundled to `dist/bundleclustering.js`. |
| `rl.ts` | RL Gridworld lab entry point — bundled to `dist/bundlerl.js`. |
| `dtree.ts` | Decision Tree / Random Forest lab entry point — bundled to `dist/bundledtree.js`. |
| `dimred.ts` | PCA / t-SNE lab entry point — bundled to `dist/bundledimred.js`. |
| `svm.ts` | SVM lab entry point — bundled to `dist/bundlesvm.js`. |
| `glm.ts` | Linear/Logistic/Naive Bayes lab entry point — bundled to `dist/bundleglm.js`. |
| `gp.ts` | Gaussian Process lab entry point — bundled to `dist/bundlegp.js`. |

---

## Credits

Original [TensorFlow Neural Network Playground](https://github.com/tensorflow/playground) by **Google / TensorFlow team**. Extended by **David Cato**. Additional features, labs, and this documentation by **Mohit Burkule** (mohit@fruitcast.co.uk).

This project is licensed under the Apache 2.0 License — see [LICENSE](LICENSE).
