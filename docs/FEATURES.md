# Feature Catalog

This document enumerates every feature available in the Neural Network Playground Extended Edition, grouped by category. Sources of truth used: `index.html`, `src/state.ts`, `src/nn.ts`, `src/dataset.ts`, `src/dataset3d.ts`, `src/adversarial.ts`, `src/unlearning.ts`, and `src/playground.ts`.

**Approximate feature total: ~130 distinct controls and capabilities on the main page, plus 42 standalone labs.**

---

## 1. Training Controls

### Play / Pause
Start or stop the continuous training loop. Each tick runs one mini-batch forward pass, backward pass, and weight update. Keyboard shortcut: **Space**.

### Step
Advance training by exactly one mini-batch without entering continuous play mode. Keyboard shortcut: **S**.

### Reset
Re-initializes all network weights using the active weight-init scheme and clears the loss chart. Keyboard shortcut: **R**.

### Regenerate Data
Re-samples the active dataset with the current noise and split settings. Keyboard shortcut: **D**.

### Speed slider
Controls how many training steps are run per animation frame (1–50). Found in the UX toolbar.

### Run N epochs
Enter a step count and click the button to run exactly that many steps then pause automatically.

### Auto-stop
Enable with the "Auto-stop" checkbox. Training pauses when the change in training loss over the last 20 steps drops below a configurable threshold.

### Learning Rate
Dropdown (0.00001–10) plus a free-text numeric input for arbitrary values. Controls the global step-size multiplier.

### Batch Size
Slider (1–30 examples). Number of training examples accumulated before each weight update.

### Noise
Slider (0–50). Adds Gaussian noise to generated dataset samples, making the boundary harder to learn.

### Train/Test Split
Slider (percTrainData, 10%–90%). Splits the generated dataset into training and held-out test sets.

### Seed
Text input. Seeds the data generator and weight initializer for reproducible runs.

### Randomize Weights
Re-initializes weights without changing architecture, data, or any other setting. Preserves optimizer state reset.

### Snapshot / Restore
Save the current set of weights and biases in memory (Snapshot), then restore them at any point (Restore). No file I/O required.

---

## 2. Network Architecture

### Number of Hidden Layers
+/− buttons in the network diagram header. Add or remove layers.

### Nodes per Layer
+/− buttons on each hidden layer column. Adjust width independently.

### Custom Feature button
Opens an inline formula input to add an arbitrary input feature computed from (x, y).

### Add custom activation button
Allows defining a custom activation function inline for the network.

---

## 3. Activation Functions (24)

Selected from the **Activation** dropdown. Applies to all hidden-layer nodes.

| Key | Name | Formula / Notes |
|---|---|---|
| `relu` | ReLU | `max(0, x)` |
| `tanh` | Tanh | `tanh(x)` — default |
| `sigmoid` | Sigmoid | `1 / (1 + exp(-x))` |
| `linear` | Linear | `x` |
| `gelu` | GELU | `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` |
| `leaky-relu` | Leaky ReLU | `x ≥ 0 ? x : 0.01x` |
| `prelu` | PReLU | `x ≥ 0 ? x : 0.2x` (α=0.2) |
| `sine` | Sine | `sin(x)` |
| `sinc` | Sinc | `sin(x)/x` (1 at x=0) |
| `mish` | Mish | `x · tanh(softplus(x))` |
| `elu` | ELU | `x ≥ 0 ? x : exp(x) − 1` |
| `selu` | SELU | `s · (x ≥ 0 ? x : a(exp(x)−1))`, self-normalizing constants |
| `swish` | Swish / SiLU | `x / (1 + exp(-x))` |
| `softplus` | Softplus | `log(1 + exp(x))` |
| `softsign` | Softsign | `x / (1 + |x|)` |
| `hard-sigmoid` | Hard Sigmoid | `clip(0.2x + 0.5, 0, 1)` |
| `hard-tanh` | Hard Tanh | `clip(x, -1, 1)` |
| `hard-swish` | Hard Swish | `x · clip((x+3)/6, 0, 1)` |
| `relu6` | ReLU6 | `min(6, max(0, x))` |
| `bent-identity` | Bent Identity | `(√(x²+1)−1)/2 + x` |
| `gaussian` | Gaussian | `exp(-x²)` |
| `snake` | Snake | `x + sin²(x)` |
| `arctan` | ArcTan | `atan(x)` |
| `isru` | ISRU | `x / √(1+x²)` |
| `exp-linear` | Exponential Linear | `x ≥ 0 ? x : 0.5(exp(x)−1)` |

---

## 4. Optimizers (10)

Selected from the **Optimizer** dropdown. Each maintains per-parameter state on every `Link` and `Node` bias.

| Key | Name | Notes |
|---|---|---|
| `sgd` | SGD | `w -= lr · grad` |
| `momentum` | Momentum | EMA of gradients, β1=0.9 |
| `rmsprop` | RMSProp | Root-mean-square gradient scaling, β2=0.999 |
| `adam` | Adam | Momentum + RMSProp with bias correction, β1=0.9, β2=0.999, ε=1e-8 |
| `nesterov` | Nesterov Momentum | Look-ahead momentum |
| `adagrad` | Adagrad | Accumulates squared gradients |
| `adadelta` | Adadelta | Self-scaling; no global LR required (LR acts as multiplier) |
| `amsgrad` | AMSGrad | Adam variant using max of past v_t |
| `nadam` | Nadam | Nesterov + Adam |
| `adamw` | AdamW | Adam + decoupled weight decay |

---

## 5. Loss Functions (5)

Selected from the **Loss** dropdown in the Training Methodology panel.

| Key | Name | Formula |
|---|---|---|
| `square` | Square (MSE) | `0.5(output − target)²` — default |
| `hinge` | Hinge | `max(0, 1 − output·target)` |
| `logloss` | Log loss / Cross-entropy | Cross-entropy via sigmoid mapping of output to (0,1) |
| `huber` | Huber | Smooth L1; quadratic inside δ=1, linear outside |
| `absolute` | Absolute (L1) | `|output − target|` |

---

## 6. Weight Initialization (6)

Selected from the **Weight init** dropdown.

| Key | Name | Notes |
|---|---|---|
| `random-uniform` | Random uniform | Default; weights in [−0.5, 0.5] |
| `xavier` | Xavier / Glorot | `N(0, 2/(fanIn+fanOut))` |
| `he` | He / Kaiming | `N(0, 2/fanIn)` |
| `lecun` | LeCun | `N(0, 1/fanIn)` |
| `zeros` | Zeros | All weights start at 0 |
| `orthogonal` | Orthogonal-lite | Scaled normal `N(0, 1/√fanIn)` |

---

## 7. Learning Rate Schedules (6)

Selected from the **LR schedule** dropdown. The **Effective LR** display shows the current value.

| Key | Name |
|---|---|
| `constant` | Constant (no decay) |
| `step` | Step decay |
| `exponential` | Exponential decay |
| `cosine` | Cosine annealing |
| `warmup-decay` | Warmup + decay |
| `onecycle` | 1cycle-lite |

---

## 8. Regularization and Normalization

### Regularization (L1 / L2)
Dropdown (`none`, `L1`, `L2`) paired with a **Regularization Rate** slider. L1 can drive weights to exactly zero (marking them `isDead`); L2 applies smooth weight shrinkage.

### Dropout
Slider (0–0.9). Inverted dropout applied to hidden layers during the forward pass in training mode only.

### Gradient Clipping
Slider (0–5). Clips each gradient component to [−gradClip, +gradClip] before the optimizer step.

### Weight Decay
Slider (0–0.1). Decoupled weight decay (AdamW-style): pulls weights toward zero by `lr · weightDecay · w` each step, independent of the gradient.

### Batch Normalization
Checkbox. Normalizes each hidden layer's pre-activation values to zero mean and unit variance (over the layer's units, single-example approximation).

### Layer Normalization
Checkbox. Same normalization formula as Batch Norm but labeled and implemented as Layer Norm; applied per-layer during the forward pass.

### Weight Quantization (5 levels)
Dropdown. Simulates reduced-precision inference by snapping weights to a fixed grid during the forward pass; gradients remain full precision.

| Setting | Grid step |
|---|---|
| None | full float |
| 16-bit | 1/65536 |
| 8-bit | 1/256 |
| 4-bit | 1/16 |
| 2-bit | 1/4 |

---

## 9. Training Methodology Panel

Collapsible panel ("Training methodology & experiments").

- **Class weighting** — weight loss/gradient by inverse class frequency
- **Epoch shuffle** — re-shuffle training data at the start of each epoch
- **Label noise** — randomly flip this fraction of training labels (0–50%)
- **Input jitter** — Gaussian noise added to inputs during training
- **Gradient noise** — annealed Gaussian noise added to gradients
- **Mixup** — convex-combination data augmentation (random λ between pairs)
- **Early stopping** — pause when test loss stops improving; configurable patience
- **Re-shuffle split** — new train/test split without regenerating data
- **k-fold CV** — run k-fold cross-validation; configurable k; shows per-fold and mean accuracy
- **LR finder** — sweep learning rates; plots loss vs LR curve; highlights suggested range
- **Ensemble training** — train N models with optional bagging; shows mean/std accuracy
- **Weight perturbation probe** — add Gaussian noise (configurable σ) to weights; reports accuracy degradation

---

## 10. Input Features

Checkboxes in the **Features** panel. Each selected feature adds an input node.

| Feature | Formula |
|---|---|
| x | raw x coordinate |
| y | raw y coordinate |
| x² | `x * x` |
| y² | `y * y` |
| x·y | `x * y` |
| sin(x) | `Math.sin(x)` |
| sin(y) | `Math.sin(y)` |
| \|x−y\| | `Math.abs(x - y)` |
| \|x+y\| | `Math.abs(x + y)` |

Additional custom features can be added inline via the **Custom feature** button.

---

## 11. Datasets — Classification (88)

Thumbnail picker in the **Data** panel. Labels ∈ {−1, +1}.

**Original classics:** Circle, XOR, Two Gaussians (gauss), Spiral, Hash, MNIST-Three, Concentric Circles, Biclusters, Moons

**Extended set (79 additional):** Checkerboard, Quadrant XOR Blobs, Concentric Rings, Three-Arm Spiral, Four-Arm Spiral, Tight Spirals, Clean Moons, Nested U Shapes, Gaussian Mixture, Diagonal Stripes, Sine Boundary, Circle in Square, Cross/Plus, S-Curve Boundary, Pinwheel, Island Clusters, Ring vs Center, Gaussian Quantiles, Anisotropic Blobs, Random-Label Blobs, Target Rings, Spiral Galaxy, Yin-Yang, Smiley, Grid of Blobs, Interleaving Waves, Blob in Ring, Triangle vs Circle, Gaussian Cross, Noisy XOR (4-quadrant), Crescent Pair, Dartboard, Comb (vertical stripes), Diagonal Checker, Cluster Chain, Nested Squares, Polygon Boundary, Voronoi Regions, Gaussian Grid 9, Two Rings XOR, Checker-4, Radial Petals, Heart Shape, Wave Interference, Gradient Blobs, Three-Class Binary, Noisy Concentric 3, Diagonal Bands 5, Blob Constellation, Sparse vs Dense, Half-Plane Noisy, Lens, Hourglass, Zigzag Boundary, Comb Teeth, Target 3-Rings, Double Spiral Tight, Quadrant Stripes, Gaussian Ring, Trefoil 2D, Gear, Star-6, Crescent Moons 3, Blob Lattice 16, Noisy Checker 6, Spiral 3-Arm Tight, Ring Segments, Pie Slices, Droplet, Infinity Symbol, Bowtie, Parallel Sines, Radial Gradient Class, Concentric Arcs, Clustered Outliers, Two Blobs Overlap, Diamond Grid, Wavy Stripes, Plus-Minus Grid

---

## 12. Datasets — Regression (32)

Selected when **Problem type** is Regression.

| Key | Description |
|---|---|
| `reg-plane` | Tilted plane: label = x + y |
| `reg-gauss` | Multi-Gaussian bump |
| `reg-sine-wave` | Sinusoidal surface |
| `reg-maximum` | max(x, y) |
| `reg-argmax` | 0 or 1 depending on which of x/y is larger |
| `reg-friedman1` | Friedman #1 benchmark |
| `reg-friedman2` | Friedman #2 benchmark |
| `reg-friedman3` | Friedman #3 benchmark |
| `reg-ripple` | Ripple surface |
| `reg-saddle` | Saddle (x² − y²) |
| `reg-gauss-bump` | Single Gaussian bump centred at origin |
| `reg-staircase` | Piecewise-constant staircase |
| `reg-sincos` | sin(x)cos(y) |
| `reg-radial` | Radial: function of distance from origin |
| `reg-abs` | |x| + |y| |
| `reg-step-circle` | Step function based on distance from origin |
| `reg-waves` | Superimposed wave surface |
| `reg-sin2d` | sin(x) + sin(y) |
| `reg-product` | x · y |
| `reg-distance` | Distance from origin |
| `reg-rosenbrock-slice` | 2D Rosenbrock slice |
| `reg-checkerboard-smooth` | Smooth checkerboard |
| `reg-peaks` | MATLAB-style peaks function |
| `reg-mexican-hat` | Mexican hat wavelet |
| `reg-ripple2` | Two-frequency ripple |
| `reg-saddle2` | Rotated saddle |
| `reg-gaussian-hill` | Offset Gaussian hill |
| `reg-sin-product` | sin(x) · sin(y) |
| `reg-abs-diff` | \|x − y\| |
| `reg-log` | log(r + 1) |
| `reg-tanh-wave` | tanh-modulated wave |
| `reg-cross-ridge` | Cross-shaped ridge |

---

## 13. 3D Dataset Mode (16 scenes)

Toggle **3D dataset mode** in the data panel. Replaces the 2D heatmap with a three.js WebGL scene. The network still receives 2D (x, y) inputs; z is visual only. Drag to rotate, scroll to zoom.

| Key | Description |
|---|---|
| `blobs` | Two Gaussian blobs |
| `spheres` | Concentric spheres |
| `helix` | Interleaved helix |
| `swiss-roll` | Swiss roll manifold |
| `linked-rings` | Two interlocked rings |
| `checkerboard-cube` | 3D checkerboard within a cube |
| `double-helix` | Double DNA-style helix |
| `xor3d` | 3D XOR — octant labeling |
| `shell-vs-core` | Shell vs interior sphere |
| `s-curve-3d` | 3D S-curve |
| `trefoil-knot` | Trefoil knot |
| `mobius-band` | Möbius band |
| `stacked-planes` | Stacked horizontal planes |
| `spiral-tower` | Spiral tower |
| `octant-checker` | Alternating octant labels |
| `sphere-grid` | Grid of small spheres |

---

## 14. Custom CSV Dataset

**Custom data** panel (bottom of main page). Paste `x,y,label` rows directly into a textarea. Label 1 or −1; 0 is mapped to −1. Parse errors are reported row-by-row. Click **Load custom dataset** to use the data.

---

## 15. Metrics and Visualization

### Main output panel
- Training loss / test loss numeric readout
- Train accuracy / test accuracy numeric readout
- **Compact status bar** — epoch, train/test loss, train/test accuracy, steps/sec
- **Loss line chart** — dual-series SVG chart (train and test loss over epochs)
- **Confusion matrix** — 2×2 live matrix for classification

### Decision boundary display
- **Heatmap** — 100×100 canvas grid colored by network output (orange = negative, blue = positive)
- **Show test data** checkbox — overlay test-set points on the heatmap
- **Discretize output** checkbox — snap heatmap to hard −1/+1 colors

### Per-node thumbnails
Hover over any hidden-layer node to see a zoomed activation surface thumbnail.

### Analysis section (collapsible)
- **Parameters** count
- **Steps/sec** throughput
- **Precision / Recall**, **F1 / Specificity**, **AUC (ROC)**, **Average Precision (PR)**, **Decision margin**, **Class balance**
- **ROC curve** (SVG)
- **Precision-Recall curve** (SVG)
- **Calibration / reliability diagram** (SVG)
- **Weight histogram** (SVG)
- **Bias histogram** (SVG)
- **Activation histogram** on test set (SVG)
- **Confidence histogram** |out| on test set (SVG)
- **Gradient flow** bar chart — mean |grad| per layer (SVG)
- **Per-layer mean |weight| over time** chart (SVG)
- **Loss landscape 1D slice** — on-demand compute (SVG)
- **Export** buttons: decision boundary PNG, training history CSV, metrics snapshot JSON

---

## 16. Interpretability Tools

**Interpretability section** (collapsible, below Analysis). All tools require 2D classification mode.

| Tool | Description |
|---|---|
| Input saliency field | Gradient magnitude ∥∇_x Loss∥ at every grid cell, overlaid as a color field |
| Occlusion sensitivity | Per-feature accuracy drop when that input is clamped to zero |
| Drop-feature importance | Retrain-free feature removal test; ranks features by accuracy impact |
| Partial dependence (x, y) | Marginal effect of each input on the network output, averaged over data |
| Neuron ablation | Select a hidden neuron; zero its output and observe boundary change; accuracy delta shown |
| Activation maximization | Gradient-ascent search for the 2D input that maximally activates a selected neuron |
| Counterfactual | Find the nearest opposite-class input to the last-clicked point |
| Hidden-activation PCA | 2D PCA projection of the last hidden layer's activations, colored by class |
| Decision-tree surrogate | Fit a shallow CART tree to the network's predictions; display rules |
| Confidence contours / entropy | Iso-confidence contour lines + per-cell entropy heatmap toggle |
| What-if inspector | Click a point to see all layer activations, output, and confidence |
| k-NN baseline | Compute k-NN accuracy on the same train/test split as a reference |
| Per-feature contribution | Decompose network output for the last-clicked point by zeroing each feature |

---

## 17. Adversarial Panel

**Adversarial** panel (bottom-left of main page).

### Attack methods (5)
| Method | Description |
|---|---|
| FGSM | Fast Gradient Sign Method: single step, ε·sign(∇_x Loss) |
| PGD | Projected Gradient Descent: multi-step (10 steps, step size ε/4), projected to L∞ ball |
| Random-noise baseline | Uniform L∞ perturbation; no gradient; control baseline |
| Targeted (least-likely) | Iterative FGSM toward the opposite label |
| DeepFool-lite | Step along gradient until prediction flips sign |

### Controls
- **Epsilon slider** (0–3) — perturbation magnitude
- **Generate adversarial examples** button — perturb test set, display result in readout
- **Robustness curve** — sweep ε and plot test accuracy vs ε
- **Saliency at last point** — gradient saliency readout for the last clicked point
- **Adversarial training** checkbox — augment each training batch with adversarial twins (defensive training)

---

## 18. Machine Unlearning Panel

**Machine Unlearning** panel (bottom-center of main page).

### Unlearning methods
- **Gradient ascent** — take gradient-ascent steps on the forget set, corrected by descent steps on the retain set
- **Fine-tune on retain** — continue training on the retain set only; pushes the model away from the forgotten distribution

### Forget set selection
- **Forget Orange** — mark all orange (+1) points as the forget set
- **Forget Blue** — mark all blue (−1) points as the forget set
- **Forget misclassified** — mark currently misclassified training points as the forget set
- **Brush-select** — drag a rectangle on the heatmap to select points by area
- **Forget nearest N** — N nearest points (Euclidean) to the last clicked point
- **Forget selected** — apply unlearning to the brush/nearest selection

### Additional controls
- **Steps slider** (10–500) — gradient-ascent steps
- **Retrain without forgotten** — gold-standard baseline: rebuild and retrain on the retain set only
- **Relearn-time probe** — measure how many steps it takes to re-learn the forgotten data
- **Before/after readout** — accuracy on forget set and retain set before and after unlearning

---

## 19. Fine-Tuning and Model I/O

**Fine-tuning: freeze layers** panel and **Save / Load model** panel.

### Layer freezing
- Per-layer checkboxes — freeze individual hidden layers; their weights and biases are not updated
- **Freeze all but last** — freeze all layers except the final hidden layer (classic transfer-learning head-only fine-tuning)
- **Unfreeze all** — clear all frozen layers

### Model export / import
- **Export model** — serializes network shape, weights, biases, activation, active inputs, and problem type to JSON; triggers download
- **Import model** — paste JSON or use the textarea; rebuilds the network and restores weights and frozen-layer state
- **Download JSON** link — appears after export

---

## 20. Experiment Tracking

**Experiments** section (collapsible).

- **Save run** — store the current epoch's metrics (loss, accuracy, architecture summary) as a named run
- **Leaderboard** — sortable table of all saved runs by test accuracy, test loss, train accuracy, or epoch
- **Overlay comparison** — plot multiple runs' loss or accuracy curves on one SVG chart with a color legend
- **A/B compare** — side-by-side metric table for any two selected runs
- **Mini grid search** — sweep a small hyperparameter grid; results displayed in a table
- **Clear all runs** / **Export runs (JSON)** / **Import runs (JSON)** — persist and share the leaderboard
- **Export this run history (CSV)** — export the current training history

---

## 21. UX, Sharing, and Accessibility

### UX toolbar
- **Copy share link** — one-click URL with the full state encoded in the URL hash
- **Preset selector** ("Model zoo") — apply curated configurations and reset
- **Speed slider** — steps per animation frame
- **Run N epochs** — run a fixed step count then pause
- **Auto-stop** checkbox + threshold — halt when loss delta is below threshold
- **Randomize weights** — re-initialize without touching architecture or data
- **Snapshot / Restore** — in-memory weight checkpoint
- **Reset view** — clear saved localStorage preferences and reload
- **? button / keyboard shortcut overlay** — lists all shortcuts

### Dark mode
Toggle switch (top-right). Loads `styles_dark.css` and disables `styles.css`.

### Keyboard shortcuts
| Key | Action |
|---|---|
| Space | Play / pause |
| S | Single step |
| R | Reset network |
| D | Regenerate data |
| ? | Toggle shortcuts help |
| Esc | Close help overlay |

### Fullscreen output
Button (⛶) in the Output panel header expands the heatmap area.

### Accessibility / i18n toolbar
- **Language selector** — English, Español, Français, हिन्दी, 中文
- **High contrast** mode toggle
- **Reduced motion** mode toggle
- **Compact mode** toggle — hides advanced panels
- **UI scale** slider (80%–150%)
- **Fast training** mode — run training steps off the main render path
- **Redraw every K steps** — control repaint frequency
- **Steps/sec · FPS · budget** meter (live, aria-live)

### Onboarding banner
Dismissable tip bar showing the most useful keyboard shortcuts on first visit.

---

## 22. Network as JavaScript

A `<code>/<pre>` panel at the bottom of the page shows the current network compiled to a standalone JavaScript function via `compileNetworkToJs`. Updates after every epoch.

The export is fully self-contained: PReLU's learned alpha value is inlined directly into the emitted expression (so a shared helper cannot accidentally capture the wrong alpha), and a helper prelude is prepended that defines `sinc`, `mish`, `gelu`, `leakyrelu`, and `softplus` as plain JS functions. The result can be copy-pasted and executed in any JavaScript environment without any additional imports or dependencies.

---

## 23. Advanced Labs (42) + Gallery

Accessible via the **lab gallery** (`labs.html`, linked from the bottom of the output panel), or directly by URL.

| Lab | URL | Algorithm(s) |
|---|---|---|
| Perceptron | `perceptron.html` | Single-layer perceptron with Rosenblatt learning rule |
| CNN | `cnn.html` | Convolutional neural network (Conv→ReLU→Pool×2 → FC → Softmax) |
| Transformer | `transformer.html` | Single/multi-head self-attention transformer block |
| RNN | `rnn.html` | Elman RNN trained on sequence echo/delay task |
| Autoencoder | `autoencoder.html` | Fully-connected autoencoder with 2D bottleneck |
| VAE | `vae.html` | Variational autoencoder with reparameterization trick |
| GAN | `gan.html` | Minimax GAN with MLP generator and discriminator |
| Diffusion (DDPM) | `diffusion.html` | DDPM forward noising + learned reverse denoising on 2D data |
| Word2Vec | `word2vec.html` | Skip-gram embeddings with negative sampling; analogy arithmetic |
| Bayesian NN | `bayesnn.html` | MC-dropout and deep ensembles for predictive uncertainty |
| RBM | `rbm.html` | Restricted Boltzmann Machine with CD-1 training |
| SNN | `snn.html` | Spiking neural network with LIF neurons and rate coding |
| MDN | `mdn.html` | Mixture Density Network for multi-modal conditional distributions |
| ESN | `esn.html` | Echo State Network (reservoir computing) |
| Activation Explorer | `actfn.html` | Interactive explorer for all 24 activation functions |
| Decision Tree / Forest | `dtree.html` | CART decision tree and random forest |
| SVM | `svm.html` | SMO-lite SVM with linear, poly, and RBF kernels |
| Linear/Logistic/Naive Bayes | `glm.html` | OLS linear regression, logistic regression, Gaussian Naive Bayes |
| Gaussian Process | `gp.html` | GP regression with RBF, Matérn-3/2, and periodic kernels |
| Clustering | `clustering.html` | k-means, DBSCAN, GMM with EM |
| PCA / t-SNE | `dimred.html` | PCA (power iteration) and t-SNE (Barnes-Hut-lite) |
| Self-Organizing Map | `som.html` | Kohonen SOM lattice; topology-preserving 2D map |
| k-NN | `knn.html` | k-Nearest Neighbor classifier with interactive Voronoi boundary |
| Conv Kernel Explorer | `conv.html` | 2D convolution kernel explorer (preset + custom kernels) |
| RL Gridworld | `rl.html` | Q-learning and SARSA on a tabular 7×10 grid |
| Genetic Algorithm | `genetic.html` | Evolutionary optimization: selection, crossover, mutation |
| PSO | `pso.html` | Particle Swarm Optimization on 2D fitness landscapes |
| Multi-Armed Bandit | `bandit.html` | ε-greedy, UCB, and Thompson sampling exploration strategies |
| Optimizer Visualizer | `optviz.html` | Gradient descent trajectories on 2D loss surfaces |
| Ant Colony Optimization | `aco.html` | ACO metaheuristic on configurable TSP graphs |
| Pathfinding | `pathfind.html` | A*, Dijkstra, BFS, DFS on interactive grid |
| Hopfield Network | `hopfield.html` | Associative memory with Hebbian weights; energy-descent recall |
| Markov Chain | `markov.html` | Discrete Markov chain with stationary distribution |
| MCMC | `mcmc.html` | Metropolis-Hastings and Gibbs sampling on 2D targets |
| Cellular Automata | `ca.html` | 1D (all 256 rules) and 2D (Game of Life, custom rules) automata |
| Neural Cellular Automata | `nca.html` | Self-organizing patterns via neural update rules |
| Reaction-Diffusion | `rd.html` | Gray-Scott system producing spots, stripes, and mazes |
| Fourier | `fourier.html` | Fourier series/transform with phasor and spectrum visualization |
| Kalman Filter | `kalman.html` | 1D/2D Kalman filter with predict/update animation |
| Boids | `boids.html` | Craig Reynolds' flocking (separation, alignment, cohesion) |
| Strange Attractors | `attractor.html` | Lorenz, Rössler, and other chaotic attractors via particle integration |
| Fractal Explorer | `fractal.html` | WebGL Mandelbrot and Julia set renderer with pan/zoom |
| **Gallery** | `labs.html` | Static entry-point card grid linking all 42 labs (no bundle) |

See [docs/LABS.md](LABS.md) for a detailed description of each lab.

---

## 24. Test Suite (445 unit tests + lab smoke tests)

The Jest unit test suite (`npm test`) has 445 tests across 11 test files. A separate runtime smoke test (`npm run test:labs`, via `scripts/smoke-labs.js`) boots all 42 lab HTML pages headlessly and verifies they load without JavaScript errors; this runs as part of CI alongside the unit tests. In addition to the existing coverage of the core ML engine, dataset generators, adversarial attacks, unlearning, and CSV parsing, the suite now includes:

- **Finite-difference gradient checks for all 24 activation functions** (`tests/activation_grad.test.ts`) — verifies that each `Activations.*` analytic derivative agrees with a central finite-difference approximation at multiple test points, skipping known kinks.
- **Finite-difference gradient checks for all 5 loss functions** (`tests/loss_grad.test.ts`) — same approach applied to every `Errors.*` derivative.
- **End-to-end backpropagation gradient check** (`tests/network_grad.test.ts`) — builds small networks with fixed weights, runs `backProp`, and verifies every link weight gradient and node bias gradient against finite differences, across multiple loss functions and activation shapes.
- **Optimizer step tests** (`tests/optimizer.test.ts`) — verifies weight updates for all 10 optimizers.

These tests were also responsible for catching two engine bugs that have since been fixed in `nn.ts`:

- **MISH derivative bug** — `MISH.der(x)` previously used an incorrect formula; the correct derivative `tanh(softplus(x)) + x * sigmoid(x) * (1 − tanh(softplus(x))²)` is now implemented and verified by the finite-difference check.
- **SINC small-value guard** — `SINC.output(x)` previously returned `sin(x)/x` for all x, producing a division-by-zero at x=0; a guard `(x*x < 1e-6) ? 1 : sin(x)/x` is now in place, matching the mathematical limit sinc(0) = 1.
