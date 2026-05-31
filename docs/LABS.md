# Advanced Labs

This page describes each of the **42 standalone algorithm labs** bundled with the Neural Network Playground Extended Edition. Each lab is a self-contained TypeScript module compiled to its own browser bundle; no server-side code is required.

Labs are accessible from the **lab gallery** (`labs.html` / `dist/labs.html`), which is linked from the main playground as the central entry point. They can also be reached directly by navigating to the individual HTML file.

---

## Gallery — `labs.html`

**File:** `dist/labs.html` | **Source:** `labs/labs.html` (no bundle; static HTML only)

The lab gallery is the recommended entry point for all standalone labs. It is a fully static HTML page (no JavaScript bundle required) that presents all 42 labs in a card grid grouped by category. Each card links directly to the corresponding lab page. A "Main Playground" card at the top links back to `index.html`.

The gallery is linked from the bottom of the main playground output panel under "Advanced labs".

---

## Deep Learning Labs

---

## Perceptron

**File:** `dist/perceptron.html` | **Source:** `src/perceptron.ts`

Implements a single-layer perceptron with the classic Rosenblatt perceptron learning rule. Trains on interactive 2D linearly-separable datasets and visualizes the decision hyperplane, weight vector, and per-step update history, making the convergence theorem tangible.

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

## Diffusion — Denoising Diffusion Probabilistic Model (DDPM)

**File:** `dist/diffusion.html` | **Source:** `src/diffusion.ts`

Implements a DDPM (denoising diffusion probabilistic model) operating on 2D point clouds. A configurable forward process progressively adds Gaussian noise to the training samples over T timesteps, destroying all structure. A small MLP (the reverse model) is then trained to predict and remove the noise at each step, learning to reverse the diffusion process.

The visualization shows the forward noising trajectory as an animated scatter plot, the denoised output produced by the reverse diffusion chain starting from pure Gaussian noise, and the training loss curve (MSE on the predicted noise). Controls expose the number of diffusion timesteps, the noise schedule (linear or cosine), network hidden size, and learning rate.

---

## Word2Vec — Skip-Gram Embeddings

**File:** `dist/word2vec.html` | **Source:** `src/word2vec.ts`

Implements the skip-gram variant of Word2Vec trained on a small configurable corpus using negative-sampling loss. A two-layer embedding network maps token indices to dense vector representations; the training objective pushes context words close together while repelling noise samples.

The visualization renders the learned 2D embedding space (after a PCA projection from the full embedding dimension) as an interactive scatter plot with word labels. A nearest-neighbor readout shows the closest words in embedding space to any selected token, and an analogy panel lets users probe vector arithmetic (king − man + woman ≈ queen) on the trained embeddings. Controls expose vocabulary size, embedding dimension, window size, negative samples, and learning rate.

---

## Bayesian Neural Network (MC-Dropout / Deep Ensembles)

**File:** `dist/bayesnn.html` | **Source:** `src/bayesnn.ts`

Implements two complementary approaches to predictive uncertainty in neural networks: **MC-dropout** (apply dropout at test time and average multiple stochastic forward passes) and **deep ensembles** (train several independent networks with different random seeds and aggregate their predictions).

The visualization renders the mean predicted boundary as a heatmap alongside a separate uncertainty map (predictive standard deviation across passes or ensemble members). The uncertainty envelope widens visibly in regions far from training data, near the decision boundary, or in areas of label ambiguity. Controls expose the number of MC-dropout samples (or ensemble members), dropout rate, network depth and width, and the choice of method.

---

## VAE — Variational Autoencoder

**File:** `dist/vae.html` | **Source:** `src/vae.ts`

Implements a Variational Autoencoder using the reparameterization trick to enable end-to-end gradient-based training of the encoder (which outputs μ and σ for the latent Gaussian) and decoder. Trained on the same 8×8 shape dataset used by the CNN and Autoencoder labs.

The visualization renders original vs reconstructed images, a 2D latent-space scatter colored by class, and a grid of images decoded from a regular lattice in latent space. The KL-divergence and reconstruction terms in the ELBO loss are charted separately, illustrating the trade-off between compression and reconstruction quality.

---

## RBM — Restricted Boltzmann Machine

**File:** `dist/rbm.html` | **Source:** `src/rbm.ts`

Implements a Restricted Boltzmann Machine trained with 1-step Contrastive Divergence (CD-1). The model learns a generative distribution over binary visible vectors and demonstrates fantasy particle generation via Gibbs sampling from the trained weights.

The visualization renders the learned weight matrix as a tiled grid, plots the reconstruction error over training steps, and shows samples drawn from the model by running multiple Gibbs sampling steps from random initial states.

---

## SNN — Spiking Neural Network

**File:** `dist/snn.html` | **Source:** `src/snn.ts`

Implements a network of leaky integrate-and-fire (LIF) neurons with configurable membrane time constants, thresholds, and refractory periods. Encodes inputs using rate coding and demonstrates how information propagates through spiking activity.

The visualization renders membrane potential traces for each neuron, a spike raster plot, and a firing-rate heatmap across the network, illustrating how spiking dynamics differ fundamentally from continuous activations.

---

## MDN — Mixture Density Network

**File:** `dist/mdn.html` | **Source:** `src/mdn.ts`

Implements a Mixture Density Network that predicts the parameters (means, variances, mixing coefficients) of a Gaussian mixture model conditioned on the input, enabling the network to represent multi-valued inverse mappings.

The visualization plots the input-output data alongside the predicted mixture components and sampled outputs, clearly showing how the MDN handles ambiguous one-to-many relationships that a standard regression network cannot capture.

---

## ESN — Echo State Network

**File:** `dist/esn.html` | **Source:** `src/esn.ts`

Implements an Echo State Network (reservoir computing), where a large fixed random recurrent reservoir projects inputs into a high-dimensional dynamic state space, and only a linear readout layer is trained via ridge regression.

The visualization renders the reservoir state matrix over time, the readout weights, and prediction vs target for chaotic time-series tasks (e.g., Mackey-Glass), showing how a fixed nonlinear reservoir can learn complex temporal patterns with minimal training.

---

## Activation Explorer

**File:** `dist/actfn.html` | **Source:** `src/actfn.ts`

Interactive explorer for activation functions. Plots f(x) and its derivative f′(x) side-by-side for all 24 activation functions supported by the main playground. Also lets you stack multiple activations in a tiny network and visualize how they compose and interact.

Useful as a reference tool for understanding saturation zones, gradient magnitudes, and the qualitative differences between activation choices.

---

## Classic ML Labs

---

## k-NN — k-Nearest Neighbors

**File:** `dist/knn.html` | **Source:** `src/knn.ts`

Implements the k-Nearest Neighbor classifier on interactive 2D datasets. Users can adjust k via a slider and see the Voronoi-style decision regions update instantly; hovering a point highlights its k nearest neighbors.

Accuracy vs k is plotted on a held-out test set, making the bias-variance trade-off concrete.

---

## Conv Kernel Explorer

**File:** `dist/conv.html` | **Source:** `src/conv.ts`

Visual explorer for 2D image convolution kernels. Preset kernels (Gaussian blur, Sobel edge-detect, sharpen, emboss) and a custom matrix editor apply the selected kernel to a sample image in real time, showing filter matrix, input, and convolved output side-by-side.

Useful for building intuition about how convolutional layers process spatial information through learned filters.

---

## Clustering

**File:** `dist/clustering.html` | **Source:** `src/clustering.ts`

Implements three unsupervised clustering algorithms in the browser: **k-means** (Lloyd's algorithm with centroid trails), **DBSCAN** (density-based; labels each point as core, border, or noise), and **GMM with EM** (Gaussian Mixture Model with E and M steps, full covariance per component).

The visualization renders data points colored by cluster assignment, k-means centroid positions connected by trails showing their movement across iterations, DBSCAN core/border/noise role indicators, and GMM confidence ellipses for each component. Five synthetic datasets are available: blobs, moons, circles, anisotropic blobs, and uniform random. Controls expose the number of clusters (k-means/GMM), DBSCAN ε and minPts, number of EM iterations, and the dataset.

---

## Other Labs

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

## Self-Organizing Map (SOM)

**File:** `dist/som.html` | **Source:** `src/som.ts`

Implements a Kohonen Self-Organizing Map: a 2D lattice of weight vectors trained via competitive learning so that topologically nearby neurons respond to similar inputs. The best-matching unit (BMU) and its neighborhood are updated each step, causing the lattice to progressively fold over the input distribution.

The visualization renders the SOM grid as a mesh overlaid on the input data scatter plot. As training progresses, the grid unfolds and conforms to the data topology, providing a topology-preserving 2D map of the input space. An additional heatmap shows the U-matrix (distance between adjacent neurons) which highlights cluster boundaries. Controls expose grid dimensions, initial learning rate, initial neighborhood radius, number of training epochs, and the input dataset.

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

---

## Hopfield Network

**File:** `dist/hopfield.html` | **Source:** `src/hopfield.ts`

Implements a classical Hopfield network (synchronous and asynchronous update modes) as an associative memory model. Binary patterns are stored by computing Hebbian weight matrices; retrieval is demonstrated by presenting a corrupted or partial version of a stored pattern and running the network's energy-minimization dynamics until it converges.

The visualization renders the stored patterns as binary pixel grids alongside the current network state as it evolves during recall. An energy plot tracks the Lyapunov energy as it descends toward a fixed point. The number of storable patterns vs the network capacity limit is displayed. Controls expose the pattern size, the number of stored patterns (editable by clicking), the amount of corruption applied to the probe, and the update mode (synchronous vs asynchronous).

---

## Genetic Algorithm

**File:** `dist/genetic.html` | **Source:** `src/genetic.ts`

Implements a genetic algorithm for single-objective optimization over a configurable 1D or 2D fitness landscape. A population of candidate solutions is evolved over generations via fitness-proportionate or tournament selection, uniform or single-point crossover, and Gaussian or bit-flip mutation.

The visualization renders the population distribution on the fitness landscape, highlights the current best individual, and plots the best and mean fitness over generations. A population diversity metric tracks premature convergence. Controls expose population size, selection pressure, mutation rate, crossover rate, the choice of fitness function (sphere, Rastrigin, Ackley, Rosenbrock, custom), and the termination criterion (max generations or fitness threshold).

---

## Optimization & Search Labs

---

## PSO — Particle Swarm Optimization

**File:** `dist/pso.html` | **Source:** `src/pso.ts`

Implements Particle Swarm Optimization on a 2D fitness landscape. A swarm of particles updates velocities based on personal best and global best positions, navigating toward the optimum via configurable inertia, cognitive, and social weight parameters.

The visualization animates particle trajectories on a contour plot of the fitness landscape and tracks the convergence of the global best value over iterations.

---

## Multi-Armed Bandit

**File:** `dist/bandit.html` | **Source:** `src/bandit.ts`

Implements the multi-armed bandit problem with three exploration strategies: **ε-greedy**, **Upper Confidence Bound (UCB)**, and **Thompson sampling** (Beta-Bernoulli). Each arm has a configurable true reward probability; the goal is to maximize cumulative reward.

The visualization plots per-arm selection counts, the estimated vs true arm values, and cumulative regret over rounds, making the exploration-exploitation trade-off immediately visible.

---

## Optimizer Visualizer

**File:** `dist/optviz.html` | **Source:** `src/optviz.ts`

Animates gradient descent trajectories for multiple optimizers (SGD, Momentum, RMSProp, Adam, AdamW, and others) simultaneously on a 2D loss surface. Surfaces include Rosenbrock, Beale, Himmelblau, and a custom formula input.

The visualization overlays each optimizer's path on the contour plot with a distinct color, making convergence speed and pathological behaviors (oscillation, overshooting) directly comparable.

---

## Ant Colony Optimization (ACO)

**File:** `dist/aco.html` | **Source:** `src/aco.ts`

Implements the Ant Colony Optimization metaheuristic on a configurable Travelling Salesman Problem (TSP) graph. Artificial ants deposit pheromones on traversed edges; evaporation and the positive feedback of short-tour ants guide the colony toward good solutions.

The visualization animates pheromone trail intensities on the graph edges and tracks the best-tour length over iterations, with configurable evaporation rate, pheromone influence, and heuristic influence parameters.

---

## Pathfinding

**File:** `dist/pathfind.html` | **Source:** `src/pathfind.ts`

Interactive grid pathfinding demonstrating A*, Dijkstra's algorithm, BFS, and DFS. Users can paint walls, set the start and goal cells, and choose the heuristic (Manhattan, Euclidean, or zero for Dijkstra). The search frontier expands step by step with configurable animation speed.

The visualization color-codes explored cells, the frontier, and the final shortest path, making the differences in search order and efficiency across algorithms concrete.

---

## Probabilistic & Statistical Labs

---

## Markov Chain

**File:** `dist/markov.html` | **Source:** `src/markov.ts`

Simulates discrete-state Markov chains with user-editable transition matrices. Computes and displays the stationary distribution analytically (dominant eigenvector) and by simulation, and animates state trajectories as the chain evolves.

The visualization renders the transition matrix as a heatmap, the state probability distribution as a bar chart updating in real time, and a state-sequence timeline, illustrating convergence to stationarity and the effect of absorbing or periodic states.

---

## MCMC — Markov Chain Monte Carlo

**File:** `dist/mcmc.html` | **Source:** `src/mcmc.ts`

Implements Metropolis-Hastings and Gibbs sampling on configurable 2D target distributions (Gaussian mixture, banana, ring, and custom). Visualizes the proposal, acceptance/rejection steps, and the accumulating sample density as a 2D histogram.

Users can adjust the proposal variance for Metropolis-Hastings to observe the effect on acceptance rate and mixing, making the relationship between step size, exploration, and convergence rate tangible.

---

## Dynamics & Signals Labs

---

## Cellular Automata (CA)

**File:** `dist/ca.html` | **Source:** `src/ca.ts`

Implements 1D elementary cellular automata (all 256 Wolfram rules) and 2D automata including Conway's Game of Life and user-configurable birth/survival rule strings. Supports configurable grid size, initial conditions (random, single cell, custom), and wrap-around boundaries.

The visualization renders the 1D spacetime diagram (rows = generations) and the 2D grid animating in real time, illustrating how simple local rules produce complex global patterns.

---

## Neural Cellular Automata (NCA)

**File:** `dist/nca.html` | **Source:** `src/nca.ts`

Implements Neural Cellular Automata where each cell's update rule is a shared convolutional neural network. The network is trained (or loaded with preset weights) to reproduce a target texture or pattern from a seed cell, demonstrating self-organization and robustness to perturbation.

The visualization renders the NCA grid evolving in real time, showing how a locally-applied neural update rule gives rise to globally coherent, regenerating patterns.

---

## Reaction-Diffusion (RD)

**File:** `dist/rd.html` | **Source:** `src/rd.ts`

Simulates the Gray-Scott reaction-diffusion system on a 2D grid with configurable feed rate (f) and kill rate (k). The two chemical species diffuse and react according to the Gray-Scott equations, producing a rich variety of self-organizing patterns.

The visualization renders the chemical concentration field in real time with a configurable color map, and a phase-diagram overlay shows where the chosen (f, k) parameters fall relative to known pattern types (spots, stripes, solitons, mazes).

---

## Fourier

**File:** `dist/fourier.html` | **Source:** `src/fourier.ts`

Interactive Fourier series and transform visualizer. Users add sinusoidal components (amplitude, frequency, phase) and see their sum update immediately; a rotating phasor diagram shows the complex addition in the frequency domain. A DFT panel computes the magnitude and phase spectrum of an arbitrary drawn waveform.

Demonstrates the relationship between time-domain signals and frequency-domain representations, Gibbs phenomenon, and the effect of windowing.

---

## Kalman Filter

**File:** `dist/kalman.html` | **Source:** `src/kalman.ts`

Implements a 1D and 2D Kalman filter tracking a noisy simulated target. The predict step propagates the state estimate through the motion model; the update step fuses in noisy sensor measurements. Both the true trajectory and noisy observations are generated in the browser.

The visualization animates the predicted state, measurement, and filtered estimate as a time series (1D) or trajectory with uncertainty ellipse (2D), with configurable process noise Q and measurement noise R.

---

## Boids

**File:** `dist/boids.html` | **Source:** `src/boids.ts`

Implements Craig Reynolds' boids flocking algorithm with three steering rules: separation (avoid crowding neighbors), alignment (steer toward average heading), and cohesion (steer toward average position). A configurable number of boids fly in a wrapped 2D canvas.

The visualization renders each boid as an oriented triangle and optionally overlays neighborhood circles and velocity vectors. Weight sliders for each rule let users explore how the interplay of local rules produces global flocking behavior.

---

## Strange Attractors

**File:** `dist/attractor.html` | **Source:** `src/attractor.ts`

Renders strange attractors (Lorenz, Rössler, Thomas, Halvorsen, and others) by numerically integrating the underlying ODEs and plotting a large number of particle trajectories in real time using a canvas-based renderer with alpha blending.

The visualization accumulates the attractor's fractal geometry over time with configurable integration step size, particle count, and color gradient, allowing the intricate structure of chaotic systems to be explored interactively.

---

## Fractal Explorer

**File:** `dist/fractal.html` | **Source:** `src/fractal.ts`

WebGL-accelerated Mandelbrot and Julia set renderer. Supports smooth coloring, configurable maximum iteration count, and multiple color maps. Pan and zoom are implemented via GPU uniform updates for real-time interaction at full canvas resolution.

The visualization also provides a Julia-set preview that updates as the cursor moves over the Mandelbrot set, showing the connection between the parameter c and the topology of the corresponding Julia set.

