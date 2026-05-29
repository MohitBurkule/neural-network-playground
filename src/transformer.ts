/* Transformer Attention Lab — visual/interactive single-page demo.
   Implements a tiny single-head (or multi-head) transformer block with
   full forward + backprop + SGD, trained on toy sequence tasks in-browser.
   Visualised with d3 v7.
*/

import * as d3 from 'd3';

// ---------------------------------------------------------------------------
// Tiny math helpers (no external ML library)
// ---------------------------------------------------------------------------

function randn(): number {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function randMatrix(rows: number, cols: number, scale = 0.1): number[][] {
  return Array.from({length: rows}, () =>
    Array.from({length: cols}, () => randn() * scale)
  );
}

function zeroMatrix(rows: number, cols: number): number[][] {
  return Array.from({length: rows}, () => new Array(cols).fill(0));
}

function zeroVec(n: number): number[] {
  return new Array(n).fill(0);
}

// Matrix multiply: (m×k) x (k×n) → (m×n)
function matmul(A: number[][], B: number[][]): number[][] {
  const m = A.length, k = A[0].length, n = B[0].length;
  const C = zeroMatrix(m, n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let p = 0; p < k; p++)
        C[i][j] += A[i][p] * B[p][j];
  return C;
}

// Transpose a matrix
function transpose(A: number[][]): number[][] {
  const m = A.length, n = A[0].length;
  const T = zeroMatrix(n, m);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      T[j][i] = A[i][j];
  return T;
}

// Row-wise softmax
function softmax(logits: number[]): number[] {
  const mx = Math.max(...logits);
  const exp = logits.map(x => Math.exp(x - mx));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

// ReLU
function relu(x: number): number { return x > 0 ? x : 0; }
function reluDer(x: number): number { return x > 0 ? 1 : 0; }

// Cross-entropy loss over a sequence of one-hot targets
function crossEntropyLoss(logits: number[][], targets: number[]): number {
  // logits: [seqLen × vocabSize], targets: [seqLen] integer indices
  let loss = 0;
  for (let t = 0; t < targets.length; t++) {
    const probs = softmax(logits[t]);
    loss -= Math.log(Math.max(probs[targets[t]], 1e-9));
  }
  return loss / targets.length;
}

// ---------------------------------------------------------------------------
// Hyperparameters & config
// ---------------------------------------------------------------------------

const VOCAB_SIZE = 8;    // tokens 0..7
const SEQ_LEN    = 6;    // sequence length
const D_MODEL    = 16;   // embedding dimension
const D_HEAD     = 8;    // key/query/value dimension per head
const D_FF       = 32;   // feed-forward hidden size
const N_HEADS    = 2;    // number of attention heads

// ---------------------------------------------------------------------------
// Parameter pack — all weights & their gradients live here
// ---------------------------------------------------------------------------

interface Params {
  // Token embedding: VOCAB_SIZE × D_MODEL
  E:    number[][];
  dE:   number[][];

  // Positional embedding: SEQ_LEN × D_MODEL
  PE:   number[][];
  dPE:  number[][];

  // Per-head Q/K/V projection: D_MODEL × D_HEAD
  Wq:   number[][][];   // [nHeads][D_MODEL][D_HEAD]
  dWq:  number[][][];
  Wk:   number[][][];
  dWk:  number[][][];
  Wv:   number[][][];
  dWv:  number[][][];

  // Output projection: (nHeads*D_HEAD) × D_MODEL
  Wo:   number[][];
  dWo:  number[][];

  // FFN
  W1:   number[][];   // D_MODEL × D_FF
  dW1:  number[][];
  b1:   number[];
  db1:  number[];
  W2:   number[][];   // D_FF × D_MODEL
  dW2:  number[][];
  b2:   number[];
  db2:  number[];

  // Output unembedding (logit projection): D_MODEL × VOCAB_SIZE
  Wout: number[][];
  dWout: number[][];
  bout:  number[];
  dbout: number[];
}

function initParams(): Params {
  const scale = 0.1;
  const heads = N_HEADS;
  return {
    E:     randMatrix(VOCAB_SIZE, D_MODEL, scale),
    dE:    zeroMatrix(VOCAB_SIZE, D_MODEL),
    PE:    randMatrix(SEQ_LEN, D_MODEL, scale),
    dPE:   zeroMatrix(SEQ_LEN, D_MODEL),

    Wq:    Array.from({length: heads}, () => randMatrix(D_MODEL, D_HEAD, scale)),
    dWq:   Array.from({length: heads}, () => zeroMatrix(D_MODEL, D_HEAD)),
    Wk:    Array.from({length: heads}, () => randMatrix(D_MODEL, D_HEAD, scale)),
    dWk:   Array.from({length: heads}, () => zeroMatrix(D_MODEL, D_HEAD)),
    Wv:    Array.from({length: heads}, () => randMatrix(D_MODEL, D_HEAD, scale)),
    dWv:   Array.from({length: heads}, () => zeroMatrix(D_MODEL, D_HEAD)),

    Wo:    randMatrix(heads * D_HEAD, D_MODEL, scale),
    dWo:   zeroMatrix(heads * D_HEAD, D_MODEL),

    W1:    randMatrix(D_MODEL, D_FF, scale),
    dW1:   zeroMatrix(D_MODEL, D_FF),
    b1:    zeroVec(D_FF),
    db1:   zeroVec(D_FF),
    W2:    randMatrix(D_FF, D_MODEL, scale),
    dW2:   zeroMatrix(D_FF, D_MODEL),
    b2:    zeroVec(D_MODEL),
    db2:   zeroVec(D_MODEL),

    Wout:  randMatrix(D_MODEL, VOCAB_SIZE, scale),
    dWout: zeroMatrix(D_MODEL, VOCAB_SIZE),
    bout:  zeroVec(VOCAB_SIZE),
    dbout: zeroVec(VOCAB_SIZE),
  };
}

function zeroGrads(p: Params): void {
  const zero2d = (m: number[][]) => m.forEach(r => r.fill(0));
  const zero3d = (t: number[][][]) => t.forEach(m => m.forEach(r => r.fill(0)));
  zero2d(p.dE); zero2d(p.dPE);
  zero3d(p.dWq); zero3d(p.dWk); zero3d(p.dWv);
  zero2d(p.dWo);
  zero2d(p.dW1); p.db1.fill(0);
  zero2d(p.dW2); p.db2.fill(0);
  zero2d(p.dWout); p.dbout.fill(0);
}

function sgdStep(p: Params, lr: number): void {
  const upd2d = (W: number[][], dW: number[][]) => {
    for (let i = 0; i < W.length; i++)
      for (let j = 0; j < W[0].length; j++)
        W[i][j] -= lr * dW[i][j];
  };
  const upd3d = (T: number[][][], dT: number[][][]) => T.forEach((W, h) => upd2d(W, dT[h]));
  const upd1d = (b: number[], db: number[]) => b.forEach((_, i) => b[i] -= lr * db[i]);
  upd2d(p.E, p.dE); upd2d(p.PE, p.dPE);
  upd3d(p.Wq, p.dWq); upd3d(p.Wk, p.dWk); upd3d(p.Wv, p.dWv);
  upd2d(p.Wo, p.dWo);
  upd2d(p.W1, p.dW1); upd1d(p.b1, p.db1);
  upd2d(p.W2, p.dW2); upd1d(p.b2, p.db2);
  upd2d(p.Wout, p.dWout); upd1d(p.bout, p.dbout);
}

// ---------------------------------------------------------------------------
// Forward pass — returns cache for backprop
// ---------------------------------------------------------------------------

interface Cache {
  tokens:   number[];           // input token indices
  X:        number[][];         // [SEQ_LEN × D_MODEL] embedded input
  // Multi-head attention cache per head
  Q:        number[][][];       // [nHeads][SEQ_LEN × D_HEAD]
  K:        number[][][];
  V:        number[][][];
  scores:   number[][][];       // [nHeads][SEQ_LEN × SEQ_LEN] pre-softmax
  attnW:    number[][][];       // [nHeads][SEQ_LEN × SEQ_LEN] post-softmax
  attnOut:  number[][][];       // [nHeads][SEQ_LEN × D_HEAD] attended values
  concat:   number[][];         // [SEQ_LEN × (nHeads*D_HEAD)]
  attnProj: number[][];         // [SEQ_LEN × D_MODEL] after Wo
  res1:     number[][];         // [SEQ_LEN × D_MODEL] residual after attn
  // FFN
  ffn1pre:  number[][];         // [SEQ_LEN × D_FF] pre-activation
  ffn1:     number[][];         // [SEQ_LEN × D_FF] post-ReLU
  ffn2:     number[][];         // [SEQ_LEN × D_MODEL]
  res2:     number[][];         // [SEQ_LEN × D_MODEL] final repr
  // Logits
  logits:   number[][];         // [SEQ_LEN × VOCAB_SIZE]
}

function forward(tokens: number[], p: Params): Cache {
  const T = tokens.length;

  // 1. Embed
  const X: number[][] = tokens.map((tok, pos) =>
    p.E[tok].map((e, d) => e + p.PE[pos][d])
  );

  // 2. Multi-head self-attention
  const Q: number[][][] = [], K: number[][][] = [], V: number[][][] = [];
  const scores: number[][][] = [], attnW: number[][][] = [], attnOut: number[][][] = [];

  for (let h = 0; h < N_HEADS; h++) {
    Q.push(matmul(X, p.Wq[h]));  // T × D_HEAD
    K.push(matmul(X, p.Wk[h]));
    V.push(matmul(X, p.Wv[h]));

    // Scaled dot-product attention
    const scale = 1.0 / Math.sqrt(D_HEAD);
    const sc: number[][] = matmul(Q[h], transpose(K[h]));  // T × T
    for (let i = 0; i < T; i++)
      for (let j = 0; j < T; j++)
        sc[i][j] *= scale;
    scores.push(sc);

    const aw: number[][] = sc.map(row => softmax(row));
    attnW.push(aw);

    attnOut.push(matmul(aw, V[h]));  // T × D_HEAD
  }

  // 3. Concatenate heads → T × (nHeads*D_HEAD)
  const concat: number[][] = Array.from({length: T}, (_, i) => {
    const row: number[] = [];
    for (let h = 0; h < N_HEADS; h++)
      row.push(...attnOut[h][i]);
    return row;
  });

  // 4. Output projection
  const attnProj = matmul(concat, p.Wo);  // T × D_MODEL

  // 5. Residual 1
  const res1: number[][] = X.map((row, i) => row.map((x, d) => x + attnProj[i][d]));

  // 6. FFN
  // ffn1pre[i][j] = sum_d res1[i][d] * W1[d][j] + b1[j]
  const ffn1pre = matmul(res1, p.W1);
  for (let i = 0; i < T; i++)
    for (let j = 0; j < D_FF; j++)
      ffn1pre[i][j] += p.b1[j];

  const ffn1 = ffn1pre.map(row => row.map(relu));

  const ffn2 = matmul(ffn1, p.W2);
  for (let i = 0; i < T; i++)
    for (let j = 0; j < D_MODEL; j++)
      ffn2[i][j] += p.b2[j];

  // 7. Residual 2
  const res2: number[][] = res1.map((row, i) => row.map((x, d) => x + ffn2[i][d]));

  // 8. Logits
  const logits = matmul(res2, p.Wout);
  for (let i = 0; i < T; i++)
    for (let j = 0; j < VOCAB_SIZE; j++)
      logits[i][j] += p.bout[j];

  return {tokens, X, Q, K, V, scores, attnW, attnOut, concat, attnProj, res1,
          ffn1pre, ffn1, ffn2, res2, logits};
}

// ---------------------------------------------------------------------------
// Backward pass
// ---------------------------------------------------------------------------

function backward(cache: Cache, targets: number[], p: Params): void {
  const T = SEQ_LEN;

  // ---- Loss gradient: dlogits ----
  // dL/dlogits[i][j] = (probs[i][j] - 1{j==targets[i]}) / T
  const dLogits: number[][] = cache.logits.map((row, i) => {
    const probs = softmax(row);
    return probs.map((pr, j) => (pr - (j === targets[i] ? 1 : 0)) / T);
  });

  // ---- Wout, bout ----
  // logits = res2 @ Wout + bout
  for (let i = 0; i < T; i++)
    for (let j = 0; j < VOCAB_SIZE; j++)
      p.dbout[j] += dLogits[i][j];
  // dWout += res2^T @ dLogits
  const dWout_contrib = matmul(transpose(cache.res2), dLogits);
  for (let d = 0; d < D_MODEL; d++)
    for (let j = 0; j < VOCAB_SIZE; j++)
      p.dWout[d][j] += dWout_contrib[d][j];

  // dres2 = dLogits @ Wout^T
  const dRes2 = matmul(dLogits, transpose(p.Wout));  // T × D_MODEL

  // ---- FFN backward (res2 = res1 + ffn2) ----
  const dFfn2 = dRes2.map(r => [...r]);  // T × D_MODEL
  const dRes1_from_ffn = dRes2.map(r => [...r]);  // residual passes through

  // ffn2 = ffn1 @ W2 + b2
  for (let i = 0; i < T; i++)
    for (let j = 0; j < D_MODEL; j++)
      p.db2[j] += dFfn2[i][j];
  const dW2_c = matmul(transpose(cache.ffn1), dFfn2);
  for (let d = 0; d < D_FF; d++)
    for (let j = 0; j < D_MODEL; j++)
      p.dW2[d][j] += dW2_c[d][j];
  const dFfn1 = matmul(dFfn2, transpose(p.W2));  // T × D_FF

  // ffn1 = relu(ffn1pre)
  const dFfn1pre: number[][] = dFfn1.map((row, i) =>
    row.map((g, j) => g * reluDer(cache.ffn1pre[i][j]))
  );

  // ffn1pre = res1 @ W1 + b1
  for (let i = 0; i < T; i++)
    for (let j = 0; j < D_FF; j++)
      p.db1[j] += dFfn1pre[i][j];
  const dW1_c = matmul(transpose(cache.res1), dFfn1pre);
  for (let d = 0; d < D_MODEL; d++)
    for (let j = 0; j < D_FF; j++)
      p.dW1[d][j] += dW1_c[d][j];
  const dRes1_from_w1 = matmul(dFfn1pre, transpose(p.W1));  // T × D_MODEL

  // Combine residual
  const dRes1: number[][] = dRes1_from_ffn.map((row, i) =>
    row.map((g, d) => g + dRes1_from_w1[i][d])
  );

  // ---- Attention backward (res1 = X + attnProj) ----
  const dAttnProj = dRes1.map(r => [...r]);  // T × D_MODEL
  const dX = dRes1.map(r => [...r]);         // residual branch

  // attnProj = concat @ Wo
  const dWo_c = matmul(transpose(cache.concat), dAttnProj);
  for (let d = 0; d < N_HEADS * D_HEAD; d++)
    for (let j = 0; j < D_MODEL; j++)
      p.dWo[d][j] += dWo_c[d][j];
  const dConcat = matmul(dAttnProj, transpose(p.Wo));  // T × (nH*D_HEAD)

  // Split dConcat into per-head dAttnOut
  for (let h = 0; h < N_HEADS; h++) {
    const start = h * D_HEAD;
    const dAttnOut_h: number[][] = dConcat.map(row => row.slice(start, start + D_HEAD));

    // attnOut[h] = attnW[h] @ V[h]
    const dV_h = matmul(transpose(cache.attnW[h]), dAttnOut_h);  // T × D_HEAD
    const dAttnW_h = matmul(dAttnOut_h, transpose(cache.V[h]));  // T × T

    // Accumulate dV gradients back to Wv
    const dWv_c = matmul(transpose(cache.X), dV_h);
    for (let d = 0; d < D_MODEL; d++)
      for (let j = 0; j < D_HEAD; j++)
        p.dWv[h][d][j] += dWv_c[d][j];

    // Softmax backward: dScores from dAttnW
    const scale = 1.0 / Math.sqrt(D_HEAD);
    const dScores_h: number[][] = dAttnW_h.map((row, i) => {
      const aw = cache.attnW[h][i];
      // d(softmax)/d(logit): Jacobian-vector product
      const dotProduct = row.reduce((s, g, j) => s + g * aw[j], 0);
      return aw.map((a, j) => scale * a * (row[j] - dotProduct));
    });

    // dScores = Q @ K^T  →  dQ from dScores, dK from dScores^T
    const dQ_h = matmul(dScores_h, cache.K[h]);          // T × D_HEAD
    const dK_h = matmul(transpose(dScores_h), cache.Q[h]); // T × D_HEAD

    // Accumulate into Wq, Wk
    const dWq_c = matmul(transpose(cache.X), dQ_h);
    const dWk_c = matmul(transpose(cache.X), dK_h);
    for (let d = 0; d < D_MODEL; d++)
      for (let j = 0; j < D_HEAD; j++) {
        p.dWq[h][d][j] += dWq_c[d][j];
        p.dWk[h][d][j] += dWk_c[d][j];
      }

    // Accumulate dX from Q, K, V projections
    const dX_h_q = matmul(dQ_h, transpose(p.Wq[h]));  // T × D_MODEL
    const dX_h_k = matmul(dK_h, transpose(p.Wk[h]));
    const dX_h_v = matmul(dV_h, transpose(p.Wv[h]));
    for (let i = 0; i < T; i++)
      for (let d = 0; d < D_MODEL; d++)
        dX[i][d] += dX_h_q[i][d] + dX_h_k[i][d] + dX_h_v[i][d];
  }

  // ---- Embedding backward ----
  for (let i = 0; i < T; i++) {
    const tok = cache.tokens[i];
    for (let d = 0; d < D_MODEL; d++) {
      p.dE[tok][d]  += dX[i][d];
      p.dPE[i][d]   += dX[i][d];
    }
  }
}

// ---------------------------------------------------------------------------
// Toy sequence tasks
// ---------------------------------------------------------------------------

type Task = 'copy' | 'reverse' | 'shift';

function makeExample(task: Task): {input: number[], target: number[]} {
  // Random input tokens [1..VOCAB_SIZE-1] (reserve 0 as PAD)
  const input = Array.from({length: SEQ_LEN}, () =>
    1 + Math.floor(Math.random() * (VOCAB_SIZE - 1))
  );
  let target: number[];
  if (task === 'copy') {
    target = [...input];
  } else if (task === 'reverse') {
    target = [...input].reverse();
  } else {
    // shift-by-one: predict next position (circular)
    target = input.slice(1).concat([input[0]]);
  }
  return {input, target};
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

let params: Params = initParams();
let task: Task = 'copy';
let learningRate = 0.003;
let isPlaying = false;
let stepCount = 0;
let lossHistory: number[] = [];
let currentCache: Cache | null = null;
let currentTarget: number[] = [];
let animFrameId: number | null = null;

// ---------------------------------------------------------------------------
// D3 Visualisation
// ---------------------------------------------------------------------------

const COLORS = {
  bg:     '#1a1a2e',
  panel:  '#16213e',
  accent: '#0f3460',
  hot:    '#e94560',
  cold:   '#4fc3f7',
  text:   '#e0e0e0',
  muted:  '#888',
  green:  '#69f0ae',
  yellow: '#ffeb3b',
};

// Token display names
const TOKEN_LABELS = ['·', 'A', 'B', 'C', 'D', 'E', 'F', 'G'];

// We use a simple colour scale from near-white (0) to hot-red (1)
const attnColor = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 1]);
const embColor  = d3.scaleDiverging(d3.interpolateRdBu).domain([-1, 0, 1]);

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const MARGIN = {top: 20, right: 20, bottom: 20, left: 20};

// Attention heatmap
const ATTN_CELL = 46;
const ATTN_PAD  = 4;

// Token strip
const TOK_W = 50, TOK_H = 34;

// Embedding strip height
const EMB_CELL_H = 12;

// Loss chart
const LOSS_W = 300, LOSS_H = 130;

// ---------------------------------------------------------------------------
// Build the page layout
// ---------------------------------------------------------------------------

function buildLayout(): void {
  const body = d3.select('body');

  // ---- header ----
  body.append('header')
    .style('padding', '18px 32px 10px')
    .style('border-bottom', `1px solid ${COLORS.accent}`)
    .html(`
      <h1 style="margin:0;font-size:1.5rem;letter-spacing:.04em;color:${COLORS.cold}">
        Transformer Attention Lab
      </h1>
      <p style="margin:4px 0 0;color:${COLORS.muted};font-size:.88rem">
        A tiny single-block transformer trained in your browser — watch attention patterns emerge.
      </p>
    `);

  // ---- controls bar ----
  const ctrl = body.append('div').attr('id', 'controls')
    .style('display', 'flex')
    .style('align-items', 'center')
    .style('gap', '18px')
    .style('padding', '12px 32px')
    .style('background', COLORS.accent)
    .style('flex-wrap', 'wrap');

  // Play/Pause
  ctrl.append('button').attr('id', 'btn-play')
    .attr('title', 'Play / Pause training')
    .style('font-size', '1.4rem')
    .style('cursor', 'pointer')
    .style('background', COLORS.hot)
    .style('color', '#fff')
    .style('border', 'none')
    .style('border-radius', '50%')
    .style('width', '42px').style('height', '42px')
    .text('▶')
    .on('click', togglePlay);

  // Step
  ctrl.append('button').attr('id', 'btn-step')
    .attr('title', 'Single step')
    .style('cursor', 'pointer')
    .style('background', COLORS.panel)
    .style('color', COLORS.text)
    .style('border', `1px solid ${COLORS.muted}`)
    .style('border-radius', '6px')
    .style('padding', '6px 14px')
    .text('Step')
    .on('click', () => { if (!isPlaying) trainStep(); });

  // Reset
  ctrl.append('button').attr('id', 'btn-reset')
    .attr('title', 'Reset weights')
    .style('cursor', 'pointer')
    .style('background', COLORS.panel)
    .style('color', COLORS.text)
    .style('border', `1px solid ${COLORS.muted}`)
    .style('border-radius', '6px')
    .style('padding', '6px 14px')
    .text('Reset')
    .on('click', resetAll);

  // LR slider
  const lrGroup = ctrl.append('label').style('display', 'flex').style('align-items', 'center').style('gap', '8px').style('color', COLORS.text);
  lrGroup.append('span').text('Learning rate');
  const lrSlider = lrGroup.append('input')
    .attr('type', 'range')
    .attr('min', '-4').attr('max', '-1').attr('step', '0.1')
    .attr('value', String(Math.log10(learningRate)))
    .style('width', '120px');
  const lrDisplay = lrGroup.append('span').attr('id', 'lr-display').text(learningRate.toFixed(4));
  lrSlider.on('input', function() {
    learningRate = Math.pow(10, +(this as HTMLInputElement).value);
    lrDisplay.text(learningRate.toFixed(4));
  });

  // Task selector
  const taskGroup = ctrl.append('label').style('display', 'flex').style('align-items', 'center').style('gap', '8px').style('color', COLORS.text);
  taskGroup.append('span').text('Task');
  const taskSel = taskGroup.append('select')
    .style('background', COLORS.panel)
    .style('color', COLORS.text)
    .style('border', `1px solid ${COLORS.muted}`)
    .style('border-radius', '4px')
    .style('padding', '4px 8px');
  (['copy', 'reverse', 'shift'] as Task[]).forEach(t => {
    taskSel.append('option').attr('value', t).text(
      t === 'copy' ? 'Copy sequence' : t === 'reverse' ? 'Reverse sequence' : 'Shift-by-one'
    );
  });
  taskSel.on('change', function() {
    task = (this as HTMLSelectElement).value as Task;
    resetAll();
  });

  // Step counter
  ctrl.append('span').attr('id', 'step-counter')
    .style('color', COLORS.muted)
    .style('font-size', '.85rem')
    .text('Step: 0');

  // Loss display
  ctrl.append('span').attr('id', 'loss-display')
    .style('color', COLORS.yellow)
    .style('font-size', '.85rem')
    .text('Loss: —');

  // ---- main content grid ----
  const main = body.append('div').attr('id', 'main')
    .style('display', 'grid')
    .style('grid-template-columns', '1fr 1fr')
    .style('grid-template-rows', 'auto auto')
    .style('gap', '24px')
    .style('padding', '24px 32px');

  // Panel: Attention heatmaps
  const attnPanel = main.append('div').attr('id', 'panel-attn')
    .style('background', COLORS.panel)
    .style('border-radius', '10px')
    .style('padding', '18px')
    .style('grid-column', '1');
  attnPanel.append('h2').text('Attention Weights')
    .style('margin', '0 0 6px')
    .style('font-size', '1rem')
    .style('color', COLORS.cold);
  attnPanel.append('p').text('Each cell (row i, col j) = how much token i attends to token j.')
    .style('margin', '0 0 14px')
    .style('font-size', '.78rem')
    .style('color', COLORS.muted);
  attnPanel.append('div').attr('id', 'attn-heatmaps');

  // Panel: Token predictions
  const predPanel = main.append('div').attr('id', 'panel-pred')
    .style('background', COLORS.panel)
    .style('border-radius', '10px')
    .style('padding', '18px')
    .style('grid-column', '2');
  predPanel.append('h2').text('Predictions vs. Targets')
    .style('margin', '0 0 6px')
    .style('font-size', '1rem')
    .style('color', COLORS.cold);
  predPanel.append('p').text('Input → predicted token (argmax) vs. expected. Green = correct.')
    .style('margin', '0 0 14px')
    .style('font-size', '.78rem')
    .style('color', COLORS.muted);
  predPanel.append('div').attr('id', 'token-display');

  // Panel: Q/K/V embeddings
  const qkvPanel = main.append('div').attr('id', 'panel-qkv')
    .style('background', COLORS.panel)
    .style('border-radius', '10px')
    .style('padding', '18px')
    .style('grid-column', '1');
  qkvPanel.append('h2').text('Q / K / V Vectors (head 0)')
    .style('margin', '0 0 6px')
    .style('font-size', '1rem')
    .style('color', COLORS.cold);
  qkvPanel.append('p').text('Each row = one sequence position. Colour = vector value (red=neg, blue=pos).')
    .style('margin', '0 0 14px')
    .style('font-size', '.78rem')
    .style('color', COLORS.muted);
  qkvPanel.append('div').attr('id', 'qkv-display');

  // Panel: Loss chart
  const lossPanel = main.append('div').attr('id', 'panel-loss')
    .style('background', COLORS.panel)
    .style('border-radius', '10px')
    .style('padding', '18px')
    .style('grid-column', '2');
  lossPanel.append('h2').text('Training Loss')
    .style('margin', '0 0 6px')
    .style('font-size', '1rem')
    .style('color', COLORS.cold);
  lossPanel.append('div').attr('id', 'loss-chart');
}

// ---------------------------------------------------------------------------
// Render the attention heatmaps
// ---------------------------------------------------------------------------

function renderAttn(): void {
  if (!currentCache) return;
  const container = d3.select('#attn-heatmaps');
  container.selectAll('*').remove();

  const cellSize = ATTN_CELL;
  const pad = ATTN_PAD;
  const n = SEQ_LEN;
  const gridSize = n * cellSize + (n - 1) * pad;
  const labelOff = 30;
  const svgW = gridSize + labelOff + 10;
  const svgH = gridSize + labelOff + 10;

  for (let h = 0; h < N_HEADS; h++) {
    const wrap = container.append('div')
      .style('display', 'inline-block')
      .style('margin-right', '24px')
      .style('vertical-align', 'top');

    wrap.append('div')
      .style('color', COLORS.muted)
      .style('font-size', '.76rem')
      .style('margin-bottom', '6px')
      .text(`Head ${h + 1}`);

    const svg = wrap.append('svg')
      .attr('width', svgW)
      .attr('height', svgH)
      .style('font-family', 'monospace');

    const g = svg.append('g').attr('transform', `translate(${labelOff},${labelOff})`);

    const attnW = currentCache!.attnW[h];  // SEQ_LEN × SEQ_LEN

    // Cells
    for (let qi = 0; qi < n; qi++) {
      for (let ki = 0; ki < n; ki++) {
        const x = ki * (cellSize + pad);
        const y = qi * (cellSize + pad);
        const w = attnW[qi][ki];

        g.append('rect')
          .attr('x', x).attr('y', y)
          .attr('width', cellSize).attr('height', cellSize)
          .attr('rx', 4)
          .attr('fill', attnColor(w))
          .append('title').text(`q=${qi} k=${ki}: ${w.toFixed(3)}`);

        // Show value if >= 0.07
        if (w >= 0.07) {
          g.append('text')
            .attr('x', x + cellSize / 2)
            .attr('y', y + cellSize / 2 + 5)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('fill', w > 0.5 ? '#222' : '#eee')
            .text(w.toFixed(2));
        }
      }
    }

    // Row labels (query token)
    for (let i = 0; i < n; i++) {
      const y = i * (cellSize + pad) + cellSize / 2;
      const tok = currentCache!.tokens[i];
      svg.append('text')
        .attr('x', labelOff - 6)
        .attr('y', labelOff + y + 4)
        .attr('text-anchor', 'end')
        .attr('font-size', '12px')
        .attr('fill', COLORS.text)
        .text(TOKEN_LABELS[tok]);
    }

    // Column labels (key token)
    for (let j = 0; j < n; j++) {
      const x = j * (cellSize + pad) + cellSize / 2;
      const tok = currentCache!.tokens[j];
      svg.append('text')
        .attr('x', labelOff + x)
        .attr('y', labelOff - 6)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', COLORS.text)
        .text(TOKEN_LABELS[tok]);
    }

    // Axis labels
    svg.append('text').attr('x', 4).attr('y', labelOff + gridSize / 2)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', COLORS.muted)
      .attr('transform', `rotate(-90, 10, ${labelOff + gridSize / 2})`)
      .text('Query →');
    svg.append('text').attr('x', labelOff + gridSize / 2).attr('y', 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', COLORS.muted)
      .text('Key →');
  }
}

// ---------------------------------------------------------------------------
// Render token predictions
// ---------------------------------------------------------------------------

function renderPredictions(): void {
  if (!currentCache) return;
  const container = d3.select('#token-display');
  container.selectAll('*').remove();

  const svg = container.append('svg')
    .attr('width', SEQ_LEN * (TOK_W + 10) + 160)
    .attr('height', 4 * (TOK_H + 12) + 20)
    .style('font-family', 'monospace, sans-serif');

  const rowLabels = ['Input', 'Predicted', 'Target', 'Correct?'];
  const rowColors = [COLORS.cold, COLORS.yellow, COLORS.text, COLORS.green];

  rowLabels.forEach((label, row) => {
    const y = row * (TOK_H + 12) + 10;

    svg.append('text')
      .attr('x', 75)
      .attr('y', y + TOK_H / 2 + 5)
      .attr('text-anchor', 'end')
      .attr('font-size', '12px')
      .attr('fill', rowColors[row])
      .text(label);

    for (let i = 0; i < SEQ_LEN; i++) {
      const x = 90 + i * (TOK_W + 10);
      let text = '';
      let fillColor = COLORS.accent;

      if (row === 0) {
        text = TOKEN_LABELS[currentCache!.tokens[i]];
        fillColor = COLORS.accent;
      } else if (row === 1) {
        const pred = argmax(softmax(currentCache!.logits[i]));
        text = TOKEN_LABELS[pred];
        fillColor = pred === currentTarget[i] ? '#1a5c38' : '#5c1a1a';
      } else if (row === 2) {
        text = TOKEN_LABELS[currentTarget[i]];
        fillColor = COLORS.accent;
      } else {
        const pred = argmax(softmax(currentCache!.logits[i]));
        text = pred === currentTarget[i] ? '✓' : '✗';
        fillColor = pred === currentTarget[i] ? '#1a5c38' : '#5c1a1a';
      }

      svg.append('rect')
        .attr('x', x).attr('y', y)
        .attr('width', TOK_W).attr('height', TOK_H)
        .attr('rx', 6)
        .attr('fill', fillColor)
        .attr('stroke', COLORS.muted)
        .attr('stroke-width', 1);

      svg.append('text')
        .attr('x', x + TOK_W / 2)
        .attr('y', y + TOK_H / 2 + 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', row === 3 ? '16px' : '14px')
        .attr('fill', rowColors[row])
        .text(text);
    }
  });

  // Show softmax distribution for one position (position 0)
  const probG = svg.append('g').attr('transform', `translate(90, ${4 * (TOK_H + 12) + 30})`);
  // small prob bars
}

function argmax(arr: number[]): number {
  let best = 0;
  for (let i = 1; i < arr.length; i++)
    if (arr[i] > arr[best]) best = i;
  return best;
}

// ---------------------------------------------------------------------------
// Render Q/K/V strips
// ---------------------------------------------------------------------------

function renderQKV(): void {
  if (!currentCache) return;
  const container = d3.select('#qkv-display');
  container.selectAll('*').remove();

  const cellW = 8;
  const cellH = EMB_CELL_H;
  const gap = 6;
  const n = SEQ_LEN;
  const d = D_HEAD;

  const matrices = [
    {label: 'Q', data: currentCache.Q[0]},
    {label: 'K', data: currentCache.K[0]},
    {label: 'V', data: currentCache.V[0]},
  ];

  const totalW = matrices.length * (d * cellW + 30) + 10;
  const totalH = n * cellH + n * 2 + 40;

  const svg = container.append('svg')
    .attr('width', totalW)
    .attr('height', totalH + 20)
    .style('font-family', 'monospace');

  let xOff = 10;
  matrices.forEach(({label, data}) => {
    svg.append('text')
      .attr('x', xOff + (d * cellW) / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .attr('font-size', '13px')
      .attr('fill', COLORS.cold)
      .text(label);

    for (let i = 0; i < n; i++) {
      const tok = currentCache!.tokens[i];
      svg.append('text')
        .attr('x', xOff - 4)
        .attr('y', 24 + i * (cellH + 2) + cellH / 2 + 3)
        .attr('text-anchor', 'end')
        .attr('font-size', '10px')
        .attr('fill', COLORS.muted)
        .text(TOKEN_LABELS[tok]);

      for (let j = 0; j < d; j++) {
        const v = data[i][j];
        svg.append('rect')
          .attr('x', xOff + j * cellW)
          .attr('y', 20 + i * (cellH + 2))
          .attr('width', cellW - 1)
          .attr('height', cellH)
          .attr('fill', embColor(Math.max(-1, Math.min(1, v))))
          .append('title').text(`${label}[${i}][${j}] = ${v.toFixed(3)}`);
      }
    }
    xOff += d * cellW + 30;
  });
}

// ---------------------------------------------------------------------------
// Render loss chart
// ---------------------------------------------------------------------------

function renderLoss(): void {
  const container = d3.select('#loss-chart');
  container.selectAll('*').remove();

  const margin = {top: 10, right: 10, bottom: 24, left: 36};
  const w = LOSS_W - margin.left - margin.right;
  const h = LOSS_H - margin.top - margin.bottom;

  const svg = container.append('svg')
    .attr('width', LOSS_W)
    .attr('height', LOSS_H);

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const data = lossHistory.slice(-300);
  if (data.length < 2) return;

  const xScale = d3.scaleLinear().domain([0, data.length - 1]).range([0, w]);
  const yScale = d3.scaleLinear()
    .domain([0, Math.max(d3.max(data) ?? 1, 0.1)])
    .range([h, 0]);

  // Grid lines
  g.append('g').attr('class', 'grid')
    .attr('opacity', 0.2)
    .call(d3.axisLeft(yScale).ticks(4).tickSize(-w).tickFormat(() => ''));

  // Line
  const line = d3.line<number>()
    .x((_, i) => xScale(i))
    .y(d => yScale(d))
    .curve(d3.curveBasis);

  g.append('path')
    .datum(data)
    .attr('fill', 'none')
    .attr('stroke', COLORS.hot)
    .attr('stroke-width', 2)
    .attr('d', line);

  // Axes
  g.append('g').attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(xScale).ticks(4))
    .selectAll('text').attr('fill', COLORS.muted).attr('font-size', '10px');

  g.append('g')
    .call(d3.axisLeft(yScale).ticks(4))
    .selectAll('text').attr('fill', COLORS.muted).attr('font-size', '10px');

  svg.selectAll('.domain, .tick line').attr('stroke', COLORS.muted);

  // Latest loss label
  const latest = data[data.length - 1];
  g.append('text')
    .attr('x', w - 4)
    .attr('y', yScale(latest) - 6)
    .attr('text-anchor', 'end')
    .attr('font-size', '11px')
    .attr('fill', COLORS.yellow)
    .text(latest.toFixed(3));
}

// ---------------------------------------------------------------------------
// Training step
// ---------------------------------------------------------------------------

function trainStep(): void {
  const ex = makeExample(task);
  const cache = forward(ex.input, params);
  const loss = crossEntropyLoss(cache.logits, ex.target);

  zeroGrads(params);
  backward(cache, ex.target, params);
  sgdStep(params, learningRate);

  currentCache = cache;
  currentTarget = ex.target;
  stepCount++;
  lossHistory.push(loss);
  if (lossHistory.length > 500) lossHistory.shift();

  // Update UI
  d3.select('#step-counter').text(`Step: ${stepCount}`);
  d3.select('#loss-display').text(`Loss: ${loss.toFixed(4)}`);

  renderAttn();
  renderPredictions();
  renderQKV();
  renderLoss();
}

// ---------------------------------------------------------------------------
// Play / pause
// ---------------------------------------------------------------------------

const STEPS_PER_FRAME = 5;  // batched for speed

function trainingLoop(): void {
  if (!isPlaying) return;
  for (let i = 0; i < STEPS_PER_FRAME; i++) trainStep();
  animFrameId = requestAnimationFrame(trainingLoop);
}

function togglePlay(): void {
  isPlaying = !isPlaying;
  d3.select('#btn-play').text(isPlaying ? '⏸' : '▶');
  if (isPlaying) trainingLoop();
}

function resetAll(): void {
  isPlaying = false;
  if (animFrameId !== null) cancelAnimationFrame(animFrameId);
  animFrameId = null;
  d3.select('#btn-play').text('▶');
  params = initParams();
  stepCount = 0;
  lossHistory = [];
  currentCache = null;
  currentTarget = [];
  d3.select('#step-counter').text('Step: 0');
  d3.select('#loss-display').text('Loss: —');
  // Do one initial forward to populate visuals
  const ex = makeExample(task);
  currentCache = forward(ex.input, params);
  currentTarget = ex.target;
  renderAttn();
  renderPredictions();
  renderQKV();
  renderLoss();
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  buildLayout();
  resetAll();
});
