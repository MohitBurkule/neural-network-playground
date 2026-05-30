# Advanced Labs

This page describes each of the twelve standalone algorithm labs bundled with the Neural Network Playground Extended Edition. Each lab is a self-contained TypeScript module compiled to its own browser bundle; no server-side code is required.

Labs are accessible from the "Advanced labs" link row at the bottom of the main page output panel, or by navigating directly to the HTML file.

---

## CNN — Convolutional Neural Network

**File:** `dist/cnn.html` | **Source:** `src/cnn.ts` + `src/cnn_data.ts`

Implements a two-stage convolutional neural network from scratch (no ML library): Conv→ReLU→MaxPool→Conv→ReLU→MaxPool→FC→Softmax. The network is trained on a synthetic dataset of 8×8 grayscale shape images (five classes: Square, Circle, Cross, Diamond, Triangle) generated directly in the browser.

The visualization renders each convolutional filter as a small grid, the resulting feature maps after each conv and pool stage, and a training/test accuracy and loss chart that updates during training. Controls expose the number of filters per layer, kernel size, and learning rate. The entire pipeline — convolution, pooling, backpropagation through conv layers, and gradient descent — runs in-browser using typed arrays.

---

## Transformer — Attention Mechanism

**File:** `dist/transformer.html` | **Source:** `src/transformer.ts`

Implements a single-layer transformer block with configurable single or multi-head self-attention (full forward pass and SGD backprop, no ML library). The network is trained on toy sequence-to-sequence tasks (e.g., copy, reverse, shift) defined by small integer vocabularies.

The visualization renders token embeddings as heat-colored rows, the Q/K/V projection matrices, the scaled dot-product attention weight matrix (one heatmap per head), the attended output vectors, and the final per-position logits. Training loss and per-step accuracy are charted in real time. Controls expose sequence length, embedding dimension, number of heads, and the choice of sequence task.

---

## Autoencoder

**File:** `dist/autoencoder.html` | **Source:** `src/autoencoder.ts`

Implements a fully-connected autoencoder (encoder → 2D bottleneck → decoder) trained with SGD and mean-squared-error loss on the same 8×8 grayscale shape dataset used by the CNN lab (five shape classes, 20 samples per class).

Three panels are rendered simultaneously: (1) a grid of original images alongside their reconstructions, color-coded by class, so compression quality is immediately visible; (2) a 2D scatter plot of the bottleneck (latent) activations for every training example, revealing class clustering; (3) a regular grid decoded back through the decoder to show how the latent space is organized. A reconstruction loss curve streams during training. Encoder/decoder depth and bottleneck dimension can be adjusted.

---

## RNN — Recurrent Neural Network

**File:** `dist/rnn.html` | **Source:** `src/rnn.ts`

Implements an Elman (simple) RNN from scratch using typed-array matrix operations and BPTT (backpropagation through time). The task is a configurable echo/delay problem: the network must reproduce its scalar input sequence delayed by k steps, making the hidden state's memory capacity directly testable.

The visualization renders the hidden-state activation matrix as a heatmap over time (rows = hidden units, columns = time steps), the input and target/prediction sequences as aligned line plots, and the weight matrices (W_xh, W_hh, W_hy) as color-coded grids. Per-step MSE loss is charted during training. Controls expose hidden size, sequence length, delay k, and learning rate.

---

## GAN — Generative Adversarial Network

**File:** `dist/gan.html` | **Source:** `src/gan.ts`

Implements a minimax GAN with MLP generator and MLP discriminator, both trained with SGD and separate Adam-style accumulators. The generator maps a low-dimensional Gaussian latent vector to 2D points; the discriminator classifies real vs generated points.

The visualization renders the real data distribution and the current generated point cloud on the same 2D scatter plot, with the discriminator's decision boundary as a heatmap background. Separate loss curves for the generator and discriminator stream in real time, making it easy to observe mode collapse, training instability, or successful convergence. Controls expose hidden size, latent dimension, learning rates for G and D, and the number of discriminator steps per generator step.

---

## Clustering

**File:** `dist/clustering.html` | **Source:** `src/clustering.ts`

Implements three unsupervised clustering algorithms in the browser: **k-means** (Lloyd's algorithm with centroid trails), **DBSCAN** (density-based; labels each point as core, border, or noise), and **GMM with EM** (Gaussian Mixture Model with E and M steps, full covariance per component).

The visualization renders data points colored by cluster assignment, k-means centroid positions connected by trails showing their movement across iterations, DBSCAN core/border/noise role indicators, and GMM confidence ellipses for each component. Five synthetic datasets are available: blobs, moons, circles, anisotropic blobs, and uniform random. Controls expose the number of clusters (k-means/GMM), DBSCAN ε and minPts, number of EM iterations, and the dataset.

---

## RL Gridworld — Reinforcement Learning

**File:** `dist/rl.html` | **Source:** `src/rl.ts`

Implements tabular Q-learning and SARSA on a fully editable 7×10 grid world. The grid supports five cell types: empty, wall, goal (positive reward), pit (negative reward), and start. The agent observes its (row, col) state and selects one of four actions (up, down, left, right).

The visualization renders the grid with the agent's current position, overlays each cell with Q-value arrows (direction = greedy action, size/color = value), and traces the agent's current episode trajectory. A cumulative reward chart updates after each episode. Controls expose the learning rate α, discount γ, exploration ε (with optional ε-decay), the choice of algorithm (Q-learning or SARSA), animation speed, and a palette for painting the grid interactively.

---

## Decision Tree / Random Forest

**File:** `dist/dtree.html` | **Source:** `src/dtree.ts`

Implements CART-style axis-aligned decision trees (Gini impurity or entropy criterion) and random forests (ensemble of independently-grown trees with bootstrap sampling) from scratch. Six 2D datasets are available: circle, XOR, spiral, moons, blobs, and four-quadrants.

The visualization renders the learned tree structure as an indented node-link diagram (feature, threshold, impurity, and sample count at each node), the 2D decision boundary heatmap colored by predicted class, and a per-class accuracy table. Switching to Random Forest replaces the single tree with the ensemble and aggregates the boundary by majority vote. Controls expose maximum depth, minimum samples per leaf, number of trees, dataset, and noise level.

---

## PCA / t-SNE — Dimensionality Reduction

**File:** `dist/dimred.html` | **Source:** `src/dimred.ts`

Implements two dimensionality reduction algorithms for high-dimensional data: **PCA** via iterative power iteration (exact for the top-2 principal components) and **t-SNE** (a Barnes-Hut-lite approximation) for non-linear embedding. Datasets are generated as synthetic high-dimensional point clouds with class structure (e.g., Gaussian clusters, ring manifolds) and are configurable in dimension and number of classes.

The visualization renders the 2D embedding as a scatter plot with points colored by class label, updates interactively as t-SNE iterations run, and displays the explained-variance ratio for PCA. Controls expose the algorithm choice, perplexity (t-SNE), learning rate, number of iterations, dataset type, and the ambient dimension of the generated data.

---

## SVM — Support Vector Machine

**File:** `dist/svm.html` | **Source:** `src/svm.ts`

Implements a Support Vector Machine with an SMO-lite (Sequential Minimal Optimization) solver that supports three kernel types: **linear**, **polynomial** (configurable degree), and **RBF** (configurable γ). Five 2D datasets are available: blobs, moons, circles, XOR, and overlapping Gaussians.

The visualization renders the kernel-induced decision boundary as a heatmap, overlays the data points with support vectors highlighted (larger markers), and shows the margin lines for linear and RBF kernels. Controls expose the kernel type, regularization parameter C, RBF γ, polynomial degree, dataset, and noise. Accuracy on a held-out test split is reported after each fit.

---

## Linear / Logistic / Naive Bayes (GLM)

**File:** `dist/glm.html` | **Source:** `src/glm.ts`

Implements three classical statistical/generative models: **ordinary least-squares linear regression** (closed-form normal-equation solution), **logistic regression** (iterative gradient descent with cross-entropy loss), and **Gaussian Naive Bayes** (class-conditional Gaussian fit with maximum-likelihood parameters). Six 2D datasets span both regression and classification tasks: noisy line, noisy quadratic, blobs, moons, circles, and XOR.

The visualization renders the fitted line (linear regression) or decision boundary (logistic/GNB) overlaid on the data scatter, reports training accuracy and loss, and for Gaussian Naive Bayes draws class-conditional confidence ellipses. Controls expose model selection, dataset, noise level, sample count, and logistic-regression learning rate and step count.

---

## Gaussian Process

**File:** `dist/gp.html` | **Source:** `src/gp.ts`

Implements exact Gaussian Process regression using Cholesky decomposition for posterior inference. Three kernel functions are supported: **RBF** (squared exponential), **Matérn-3/2**, and **periodic**. Hyperparameters (length-scale, signal variance, noise variance, and period for the periodic kernel) are configurable via sliders.

The visualization renders a 1D function view: observed training points as scatter, the posterior mean as a solid line, the 95% credible band (±2 posterior std dev) as a shaded region, and several randomly-drawn posterior samples as thin curves. The Cholesky solve is rerun on every hyperparameter change, giving immediate visual feedback on how each hyperparameter controls smoothness, amplitude, and periodicity. Controls also expose the number of training points and the ability to place points interactively by clicking the plot.
