import * as d3 from 'd3';

// ─── Corpus & vocabulary ─────────────────────────────────────────────────────

interface WordMeta {
  word: string;
  category: string;
}

const WORDS: WordMeta[] = [
  // royalty
  { word: 'king',    category: 'royalty' },
  { word: 'queen',   category: 'royalty' },
  { word: 'prince',  category: 'royalty' },
  { word: 'princess',category: 'royalty' },
  { word: 'knight',  category: 'royalty' },
  { word: 'throne',  category: 'royalty' },
  // animals
  { word: 'dog',     category: 'animal' },
  { word: 'cat',     category: 'animal' },
  { word: 'bird',    category: 'animal' },
  { word: 'fish',    category: 'animal' },
  { word: 'horse',   category: 'animal' },
  { word: 'wolf',    category: 'animal' },
  // colors
  { word: 'red',     category: 'color' },
  { word: 'blue',    category: 'color' },
  { word: 'green',   category: 'color' },
  { word: 'yellow',  category: 'color' },
  { word: 'black',   category: 'color' },
  { word: 'white',   category: 'color' },
  // foods
  { word: 'apple',   category: 'food' },
  { word: 'bread',   category: 'food' },
  { word: 'cheese',  category: 'food' },
  { word: 'milk',    category: 'food' },
  { word: 'rice',    category: 'food' },
  { word: 'soup',    category: 'food' },
  // actions
  { word: 'run',     category: 'action' },
  { word: 'eat',     category: 'action' },
  { word: 'sleep',   category: 'action' },
  { word: 'swim',    category: 'action' },
  { word: 'fly',     category: 'action' },
  { word: 'hunt',    category: 'action' },
  // places
  { word: 'castle',  category: 'place' },
  { word: 'forest',  category: 'place' },
  { word: 'ocean',   category: 'place' },
  { word: 'farm',    category: 'place' },
  { word: 'city',    category: 'place' },
  { word: 'river',   category: 'place' },
  // gender
  { word: 'man',     category: 'gender' },
  { word: 'woman',   category: 'gender' },
  { word: 'boy',     category: 'gender' },
  { word: 'girl',    category: 'gender' },
];

// Toy corpus: hundreds of sentences that form semantic patterns
const RAW_SENTENCES: string[] = [
  // royalty + places + gender
  'the king lives in the castle',
  'the queen lives in the castle',
  'the king and queen rule the city',
  'the prince became a king',
  'the princess became a queen',
  'the knight serves the king',
  'the knight serves the queen',
  'the king sat on the throne',
  'the queen sat on the throne',
  'the man became king',
  'the woman became queen',
  'the boy became a prince',
  'the girl became a princess',
  'the king is a man',
  'the queen is a woman',
  'the prince is a boy',
  'the princess is a girl',
  'the knight fought for the king',
  'the knight fought for the queen',
  'the prince trained to be a knight',
  'the castle stands in the city',
  'the king rode a horse',
  'the queen rode a horse',
  'the knight rode a horse',
  // animals + actions + places
  'the dog runs in the forest',
  'the cat sleeps on the farm',
  'the bird flies over the river',
  'the fish swims in the ocean',
  'the horse runs across the farm',
  'the wolf hunts in the forest',
  'the dog eats bread on the farm',
  'the cat drinks milk on the farm',
  'the bird hunts fish in the river',
  'the wolf runs in the forest',
  'the fish swims in the river',
  'the horse lives on the farm',
  'the dog hunts in the forest',
  'the cat sleeps in the city',
  'the bird flies over the ocean',
  'the wolf sleeps in the forest',
  'the dog swims in the river',
  'the cat eats fish every day',
  'the bird eats bread and rice',
  'the horse runs along the river',
  'the wolf drinks from the river',
  'the fish lives in the ocean',
  'the farm has a dog and a cat',
  'the forest has wolves and birds',
  // colors + objects
  'the king wore red and blue',
  'the queen wore white and yellow',
  'the red apple fell from the tree',
  'the green apple is sweet',
  'the black cat sat in the city',
  'the white horse ran on the farm',
  'the blue river flows to the ocean',
  'the green forest is deep',
  'the yellow bird flew over the farm',
  'the red fish lives in the river',
  'the white wolf hunts at night',
  'the black knight rode a black horse',
  // foods + actions + animals
  'the dog eats bread every morning',
  'the cat drinks milk and sleeps',
  'the man eats soup and bread',
  'the woman eats rice and soup',
  'the boy eats apple and bread',
  'the girl drinks milk and eats cheese',
  'the king eats bread and cheese',
  'the queen eats soup and rice',
  'bread and cheese are food',
  'milk and rice are food',
  'the farm grows rice and bread',
  'the fish eats other fish in the ocean',
  'the bird eats fish and flies away',
  // analogy seeds: king-man+woman=queen, dog-wolf (domestic-wild)
  'the man hunts in the forest',
  'the woman hunts in the forest',
  'the man lives in the city',
  'the woman lives in the city',
  'the man and the woman live in the city',
  'the boy and the girl play in the city',
  'the man rides a horse in the forest',
  'the woman rides a horse in the city',
  'the king is a powerful man',
  'the queen is a powerful woman',
  'the prince is a young man',
  'the princess is a young woman',
  'the dog is a friendly animal',
  'the wolf is a wild animal',
  'the cat is a small animal',
  'the horse is a large animal',
  'the bird is a flying animal',
  'the fish is a swimming animal',
  // more repetition to reinforce clusters
  'the dog runs and eats',
  'the cat sleeps and eats',
  'the wolf runs and hunts',
  'the bird flies and eats',
  'the fish swims and eats',
  'the horse runs and sleeps',
  'eat bread and soup',
  'drink milk at the farm',
  'the city has a castle',
  'the forest has a river',
  'the ocean is blue',
  'the forest is green',
  'the farm grows food',
  'the city has food',
  'red and blue are colors',
  'green and yellow are colors',
  'black and white are colors',
  'the king wore a red robe',
  'the queen wore a white dress',
  'the knight wore black armor',
  'the prince wore blue robes',
  'the princess wore yellow dress',
  'the dog has black and white fur',
  'the cat has white and yellow fur',
  'the wolf has grey fur',
  'the horse has brown fur',
  'bread is white food',
  'apple is red or green food',
  'cheese is yellow food',
  'milk is white food',
  'soup is warm food',
  'rice is white food',
];

// build corpus token sequences
function buildCorpus(): { vocab: string[]; wordToIdx: Map<string, number>; sentences: number[][] } {
  const vocab = WORDS.map(w => w.word);
  const wordToIdx = new Map<string, number>();
  vocab.forEach((w, i) => wordToIdx.set(w, i));

  const sentences: number[][] = [];
  RAW_SENTENCES.forEach(sent => {
    const tokens: number[] = [];
    sent.toLowerCase().split(/\s+/).forEach(tok => {
      const idx = wordToIdx.get(tok);
      if (idx !== undefined) tokens.push(idx);
    });
    if (tokens.length >= 2) sentences.push(tokens);
  });
  return { vocab, wordToIdx, sentences };
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

function vecNorm(a: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s) + 1e-10;
}

function cosineSim(a: Float64Array, b: Float64Array): number {
  return dot(a, b) / (vecNorm(a) * vecNorm(b));
}

// PCA: project V (n x d) to 2D using top-2 eigenvectors via power iteration
function pca2d(vecs: Float64Array[]): [number, number][] {
  const n = vecs.length;
  const d = vecs[0].length;
  if (d <= 2) return vecs.map(v => [v[0], v[1]] as [number, number]);

  // center
  const mean = new Float64Array(d);
  vecs.forEach(v => { for (let j = 0; j < d; j++) mean[j] += v[j]; });
  for (let j = 0; j < d; j++) mean[j] /= n;
  const C: Float64Array[] = vecs.map(v => {
    const c = new Float64Array(d);
    for (let j = 0; j < d; j++) c[j] = v[j] - mean[j];
    return c;
  });

  // covariance-vector product: Cv = X^T(Xv)/n
  function covMul(u: Float64Array): Float64Array {
    const Xu = new Float64Array(n);
    for (let i = 0; i < n; i++) Xu[i] = dot(C[i], u);
    const res = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) res[j] += Xu[i] * C[i][j];
    }
    for (let j = 0; j < d; j++) res[j] /= n;
    return res;
  }

  function powerIter(deflate?: Float64Array): Float64Array {
    const u = new Float64Array(d);
    for (let j = 0; j < d; j++) u[j] = Math.random() - 0.5;
    for (let iter = 0; iter < 80; iter++) {
      const Cv = covMul(u);
      if (deflate) {
        const dp = dot(Cv, deflate);
        for (let j = 0; j < d; j++) Cv[j] -= dp * deflate[j];
      }
      const nm = vecNorm(Cv);
      for (let j = 0; j < d; j++) u[j] = Cv[j] / nm;
    }
    return u;
  }

  const pc1 = powerIter();
  const pc2 = powerIter(pc1);

  return C.map(v => [dot(v, pc1), dot(v, pc2)] as [number, number]);
}

// ─── Word2Vec (skip-gram + negative sampling) ────────────────────────────────

class Word2Vec {
  vocabSize: number;
  dim: number;
  W: Float64Array[];  // input embeddings  [vocabSize x dim]
  C: Float64Array[];  // output embeddings [vocabSize x dim]

  constructor(vocabSize: number, dim: number) {
    this.vocabSize = vocabSize;
    this.dim = dim;
    this.W = [];
    this.C = [];
    for (let i = 0; i < vocabSize; i++) {
      this.W.push(this._randVec());
      this.C.push(this._randVec());
    }
  }

  private _randVec(): Float64Array {
    const v = new Float64Array(this.dim);
    for (let j = 0; j < this.dim; j++) v[j] = (Math.random() - 0.5) * 0.5;
    return v;
  }

  reset(dim: number): void {
    this.dim = dim;
    for (let i = 0; i < this.vocabSize; i++) {
      this.W[i] = this._randVec();
      this.C[i] = this._randVec();
    }
  }

  // one skip-gram step: center word, context word, negative samples
  // returns loss contribution
  trainPair(
    centerIdx: number,
    contextIdx: number,
    negIndices: number[],
    lr: number,
  ): number {
    const w = this.W[centerIdx];
    const gradW = new Float64Array(this.dim);
    let loss = 0;

    // positive pair
    const posScore = dot(w, this.C[contextIdx]);
    const posSig = sigmoid(posScore);
    const posGrad = (posSig - 1);  // d(loss)/d(score) for positive = sig(x) - 1
    loss -= Math.log(posSig + 1e-10);

    for (let j = 0; j < this.dim; j++) {
      gradW[j] += posGrad * this.C[contextIdx][j];
      this.C[contextIdx][j] -= lr * posGrad * w[j];
    }

    // negative pairs
    negIndices.forEach(negIdx => {
      const negScore = dot(w, this.C[negIdx]);
      const negSig = sigmoid(negScore);
      const negGrad = negSig;  // d(loss)/d(score) for negative = sig(x)
      loss -= Math.log(1 - negSig + 1e-10);

      for (let j = 0; j < this.dim; j++) {
        gradW[j] += negGrad * this.C[negIdx][j];
        this.C[negIdx][j] -= lr * negGrad * w[j];
      }
    });

    // update center embedding
    for (let j = 0; j < this.dim; j++) w[j] -= lr * gradW[j];

    return loss;
  }
}

// ─── Training state ──────────────────────────────────────────────────────────

interface Params {
  lr: number;
  dim: number;
  windowSize: number;
  negSamples: number;
  usePca: boolean;
}

interface TrainingState {
  model: Word2Vec | null;
  sentences: number[][];
  vocab: string[];
  wordToIdx: Map<string, number>;
  epoch: number;
  step: number;       // total training pairs processed
  lossHistory: number[];
  currentEpochLoss: number;
  currentEpochPairs: number;
  sentIdx: number;    // cursor through sentences
  playing: boolean;
  params: Params;
}

let state: TrainingState;

function initState(params: Params): void {
  const { vocab, wordToIdx, sentences } = buildCorpus();
  const model = new Word2Vec(vocab.length, params.dim);
  state = {
    model,
    sentences,
    vocab,
    wordToIdx,
    epoch: 0,
    step: 0,
    lossHistory: [],
    currentEpochLoss: 0,
    currentEpochPairs: 0,
    sentIdx: 0,
    playing: false,
    params,
  };
}

// Sample negative indices (uniform for simplicity; exclude center+context)
function sampleNegatives(
  exclude1: number,
  exclude2: number,
  k: number,
  vocabSize: number,
): number[] {
  const negs: number[] = [];
  let attempts = 0;
  while (negs.length < k && attempts < k * 10) {
    const idx = Math.floor(Math.random() * vocabSize);
    if (idx !== exclude1 && idx !== exclude2) negs.push(idx);
    attempts++;
  }
  return negs;
}

// Process one batch of skip-gram pairs (from one sentence)
// returns { loss, pairs } processed this call
function trainStep(): { loss: number; pairs: number } {
  if (!state.model) return { loss: 0, pairs: 0 };
  const { model, sentences, params } = state;
  if (sentences.length === 0) return { loss: 0, pairs: 0 };

  const sent = sentences[state.sentIdx];
  state.sentIdx = (state.sentIdx + 1) % sentences.length;

  // new epoch when we wrap around
  if (state.sentIdx === 0) {
    if (state.currentEpochPairs > 0) {
      state.lossHistory.push(state.currentEpochLoss / state.currentEpochPairs);
    }
    state.currentEpochLoss = 0;
    state.currentEpochPairs = 0;
    state.epoch++;
  }

  let batchLoss = 0;
  let batchPairs = 0;

  sent.forEach((centerIdx, pos) => {
    const win = params.windowSize;
    for (let offset = -win; offset <= win; offset++) {
      if (offset === 0) continue;
      const ctxPos = pos + offset;
      if (ctxPos < 0 || ctxPos >= sent.length) continue;
      const ctxIdx = sent[ctxPos];
      const negs = sampleNegatives(centerIdx, ctxIdx, params.negSamples, model.vocabSize);
      batchLoss += model.trainPair(centerIdx, ctxIdx, negs, params.lr);
      batchPairs++;
    }
  });

  state.currentEpochLoss += batchLoss;
  state.currentEpochPairs += batchPairs;
  state.step += batchPairs;

  return { loss: batchLoss, pairs: batchPairs };
}

// ─── Category color scale ─────────────────────────────────────────────────────

const CATEGORY_COLORS: Record<string, string> = {
  royalty: '#f59e0b',
  animal:  '#34d399',
  color:   '#f87171',
  food:    '#a78bfa',
  action:  '#60a5fa',
  place:   '#fb923c',
  gender:  '#e879f9',
};

function wordColor(word: string): string {
  const meta = WORDS.find(w => w.word === word);
  return meta ? (CATEGORY_COLORS[meta.category] || '#aaa') : '#aaa';
}

function wordCategory(word: string): string {
  const meta = WORDS.find(w => w.word === word);
  return meta ? meta.category : 'other';
}

// ─── 2D projection ────────────────────────────────────────────────────────────

function getProjected(usePca: boolean): [number, number][] {
  if (!state.model) return [];
  const vecs = state.model.W;
  if (usePca || state.params.dim > 2) {
    return pca2d(vecs);
  }
  // raw 2D
  return vecs.map(v => [v[0], v[1]] as [number, number]);
}

// ─── Scatter plot (SVG) ───────────────────────────────────────────────────────

interface ScatterPoint {
  word: string;
  px: number;
  py: number;
  category: string;
}

let svgScatter: d3.Selection<SVGSVGElement, unknown, HTMLElement, unknown>;
let scatterG: d3.Selection<SVGGElement, unknown, HTMLElement, unknown>;
let xScale: d3.ScaleLinear<number, number>;
let yScale: d3.ScaleLinear<number, number>;
let highlightedWord: string | null = null;
const SCATTER_W = 580;
const SCATTER_H = 480;
const MARGIN = { top: 16, right: 16, bottom: 16, left: 16 };

function initScatter(): void {
  svgScatter = d3.select<SVGSVGElement, unknown>('#scatter-svg');
  svgScatter.attr('width', SCATTER_W).attr('height', SCATTER_H);
  scatterG = svgScatter.append('g');

  xScale = d3.scaleLinear().range([MARGIN.left, SCATTER_W - MARGIN.right]);
  yScale = d3.scaleLinear().range([SCATTER_H - MARGIN.bottom, MARGIN.top]);
}

function updateScatter(): void {
  if (!state.model) return;
  const pts2d = getProjected(state.params.usePca);

  const points: ScatterPoint[] = state.vocab.map((word, i) => ({
    word,
    px: pts2d[i][0],
    py: pts2d[i][1],
    category: wordCategory(word),
  }));

  const xs = points.map(p => p.px);
  const ys = points.map(p => p.py);
  const xMin = d3.min(xs) ?? -1;
  const xMax = d3.max(xs) ?? 1;
  const yMin = d3.min(ys) ?? -1;
  const yMax = d3.max(ys) ?? 1;
  const xPad = (xMax - xMin) * 0.15 + 0.01;
  const yPad = (yMax - yMin) * 0.15 + 0.01;
  xScale.domain([xMin - xPad, xMax + xPad]);
  yScale.domain([yMin - yPad, yMax + yPad]);

  // circles
  const circles = scatterG.selectAll<SVGCircleElement, ScatterPoint>('circle.word-dot')
    .data(points, d => d.word);

  const circlesEnter = circles.enter().append('circle')
    .attr('class', 'word-dot')
    .attr('r', 7)
    .attr('stroke-width', 2)
    .style('cursor', 'pointer')
    .on('mouseenter', (_evt, d) => {
      highlightedWord = d.word;
      renderNeighbors(d.word);
      updateScatter();
    })
    .on('mouseleave', () => {
      highlightedWord = null;
      updateScatter();
    })
    .on('click', (_evt, d) => {
      (document.getElementById('nn-word-input') as HTMLInputElement).value = d.word;
      renderNeighbors(d.word);
    });

  circlesEnter.merge(circles)
    .transition().duration(80)
    .attr('cx', d => xScale(d.px))
    .attr('cy', d => yScale(d.py))
    .attr('fill', d => wordColor(d.word))
    .attr('stroke', d => d.word === highlightedWord ? '#fff' : 'rgba(0,0,0,0.4)')
    .attr('r', d => d.word === highlightedWord ? 10 : 7)
    .attr('opacity', d => highlightedWord && d.word !== highlightedWord ? 0.5 : 1.0);

  circles.exit().remove();

  // labels
  const labels = scatterG.selectAll<SVGTextElement, ScatterPoint>('text.word-label')
    .data(points, d => d.word);

  const labelsEnter = labels.enter().append('text')
    .attr('class', 'word-label')
    .attr('font-size', '11px')
    .attr('font-family', 'Segoe UI, Arial, sans-serif')
    .attr('pointer-events', 'none');

  labelsEnter.merge(labels)
    .transition().duration(80)
    .attr('x', d => xScale(d.px) + 9)
    .attr('y', d => yScale(d.py) + 4)
    .attr('fill', d => d.word === highlightedWord ? '#fff' : wordColor(d.word))
    .attr('font-weight', d => d.word === highlightedWord ? '700' : '400')
    .attr('font-size', d => d.word === highlightedWord ? '13px' : '11px')
    .text(d => d.word);

  labels.exit().remove();
}

// ─── Loss curve (canvas) ──────────────────────────────────────────────────────

let lossCanvas: HTMLCanvasElement;
let lossCtx: CanvasRenderingContext2D;

function initLossCanvas(): void {
  lossCanvas = document.getElementById('loss-canvas') as HTMLCanvasElement;
  lossCtx = lossCanvas.getContext('2d')!;
}

function drawLoss(): void {
  const w = lossCanvas.width;
  const h = lossCanvas.height;
  lossCtx.clearRect(0, 0, w, h);

  const hist = state.lossHistory;
  if (hist.length < 2) {
    lossCtx.fillStyle = '#555';
    lossCtx.font = '12px Segoe UI, Arial';
    lossCtx.fillText('Loss curve (training...)', 10, h / 2);
    return;
  }

  const pad = { top: 8, right: 8, bottom: 28, left: 48 };
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;

  const minL = d3.min(hist) ?? 0;
  const maxL = d3.max(hist) ?? 1;
  const rng = maxL - minL || 1;

  const xS = (i: number) => pad.left + (i / (hist.length - 1)) * cw;
  const yS = (v: number) => pad.top + ch - ((v - minL) / rng) * ch;

  // grid lines
  lossCtx.strokeStyle = '#2a2a4a';
  lossCtx.lineWidth = 1;
  [0.25, 0.5, 0.75, 1].forEach(t => {
    const y = pad.top + ch * (1 - t);
    lossCtx.beginPath();
    lossCtx.moveTo(pad.left, y);
    lossCtx.lineTo(pad.left + cw, y);
    lossCtx.stroke();
    lossCtx.fillStyle = '#666';
    lossCtx.font = '10px Segoe UI, Arial';
    const val = minL + t * rng;
    lossCtx.fillText(val.toFixed(2), 2, y + 4);
  });

  // curve
  lossCtx.beginPath();
  lossCtx.strokeStyle = '#a78bfa';
  lossCtx.lineWidth = 2;
  hist.forEach((v, i) => {
    const x = xS(i);
    const y = yS(v);
    if (i === 0) lossCtx.moveTo(x, y); else lossCtx.lineTo(x, y);
  });
  lossCtx.stroke();

  // x-axis label
  lossCtx.fillStyle = '#888';
  lossCtx.font = '10px Segoe UI, Arial';
  lossCtx.fillText('Epoch', pad.left + cw / 2 - 12, h - 4);
  lossCtx.fillStyle = '#a78bfa';
  lossCtx.fillText(`Loss: ${hist[hist.length - 1].toFixed(3)}`, pad.left + cw - 60, pad.top + 14);
}

// ─── Nearest neighbors ────────────────────────────────────────────────────────

function getNearestNeighbors(word: string, k: number): { word: string; sim: number }[] {
  if (!state.model) return [];
  const idx = state.wordToIdx.get(word);
  if (idx === undefined) return [];
  const vec = state.model.W[idx];
  const sims: { word: string; sim: number }[] = [];
  state.vocab.forEach((w, i) => {
    if (i === idx) return;
    sims.push({ word: w, sim: cosineSim(vec, state.model!.W[i]) });
  });
  sims.sort((a, b) => b.sim - a.sim);
  return sims.slice(0, k);
}

function renderNeighbors(word: string): void {
  const container = document.getElementById('nn-results')!;
  if (!word) { container.innerHTML = '<p class="placeholder">Type a word above</p>'; return; }
  const idx = state.wordToIdx.get(word);
  if (idx === undefined) {
    container.innerHTML = `<p class="placeholder">Unknown word: "${word}"</p>`;
    return;
  }
  const neighbors = getNearestNeighbors(word, 8);
  const color = wordColor(word);
  let html = `<div class="nn-query">Neighbors of <span style="color:${color};font-weight:700">${word}</span></div>`;
  html += '<div class="nn-list">';
  neighbors.forEach(nb => {
    const c = wordColor(nb.word);
    const barW = Math.max(0, Math.min(100, ((nb.sim + 1) / 2) * 100));
    html += `<div class="nn-row">
      <span class="nn-word" style="color:${c}">${nb.word}</span>
      <div class="nn-bar-wrap"><div class="nn-bar" style="width:${barW.toFixed(1)}%;background:${c}"></div></div>
      <span class="nn-sim">${nb.sim.toFixed(3)}</span>
    </div>`;
  });
  html += '</div>';
  container.innerHTML = html;
}

// ─── Analogy ──────────────────────────────────────────────────────────────────

function runAnalogy(a: string, b: string, c: string): { word: string; score: number }[] {
  if (!state.model) return [];
  const idxA = state.wordToIdx.get(a);
  const idxB = state.wordToIdx.get(b);
  const idxC = state.wordToIdx.get(c);
  if (idxA === undefined || idxB === undefined || idxC === undefined) return [];

  // target = W[b] - W[a] + W[c]
  const dim = state.params.dim;
  const target = new Float64Array(dim);
  for (let j = 0; j < dim; j++) {
    target[j] = state.model.W[idxB][j] - state.model.W[idxA][j] + state.model.W[idxC][j];
  }

  const exclude = new Set<number>([idxA, idxB, idxC]);
  const results: { word: string; score: number }[] = [];
  state.vocab.forEach((w, i) => {
    if (exclude.has(i)) return;
    results.push({ word: w, score: cosineSim(target, state.model!.W[i]) });
  });
  results.sort((r1, r2) => r2.score - r1.score);
  return results.slice(0, 5);
}

function renderAnalogy(): void {
  const a = (document.getElementById('analogy-a') as HTMLInputElement).value.trim().toLowerCase();
  const b = (document.getElementById('analogy-b') as HTMLInputElement).value.trim().toLowerCase();
  const c = (document.getElementById('analogy-c') as HTMLInputElement).value.trim().toLowerCase();
  const container = document.getElementById('analogy-results')!;
  if (!a || !b || !c) {
    container.innerHTML = '<p class="placeholder">Fill in all three words</p>';
    return;
  }
  const results = runAnalogy(a, b, c);
  if (results.length === 0) {
    container.innerHTML = '<p class="placeholder">Unknown word(s)</p>';
    return;
  }
  let html = `<div class="nn-query">"${a}" is to "${b}" as "${c}" is to...</div><div class="nn-list">`;
  results.forEach((r, rank) => {
    const c2 = wordColor(r.word);
    html += `<div class="nn-row">
      <span style="color:#888;font-size:0.8rem;margin-right:4px">${rank + 1}.</span>
      <span class="nn-word" style="color:${c2}">${r.word}</span>
      <span class="nn-sim">${r.score.toFixed(3)}</span>
    </div>`;
  });
  html += '</div>';
  container.innerHTML = html;
}

// ─── UI wiring ────────────────────────────────────────────────────────────────

let animId: number | null = null;
const STEPS_PER_FRAME = 4;   // sentences processed per animation frame

function readParams(): Params {
  const lr = parseFloat((document.getElementById('slider-lr') as HTMLInputElement).value);
  const dim = parseInt((document.getElementById('slider-dim') as HTMLInputElement).value, 10);
  const win = parseInt((document.getElementById('slider-win') as HTMLInputElement).value, 10);
  const neg = parseInt((document.getElementById('slider-neg') as HTMLInputElement).value, 10);
  const usePca = (document.getElementById('toggle-pca') as HTMLInputElement).checked;
  return { lr, dim, windowSize: win, negSamples: neg, usePca };
}

function updateStatusBar(): void {
  const el = document.getElementById('status-bar')!;
  el.innerHTML = `Epoch <b>${state.epoch}</b> &nbsp;|&nbsp; Pairs <b>${state.step.toLocaleString()}</b>`;
}

function frame(): void {
  for (let i = 0; i < STEPS_PER_FRAME; i++) trainStep();
  updateScatter();
  drawLoss();
  updateStatusBar();
  if (state.playing) animId = requestAnimationFrame(frame);
}

function play(): void {
  if (state.playing) return;
  state.playing = true;
  animId = requestAnimationFrame(frame);
}

function pause(): void {
  state.playing = false;
  if (animId !== null) { cancelAnimationFrame(animId); animId = null; }
}

function step(): void {
  pause();
  for (let i = 0; i < STEPS_PER_FRAME; i++) trainStep();
  updateScatter();
  drawLoss();
  updateStatusBar();
}

function reset(): void {
  pause();
  const params = readParams();
  initState(params);
  updateScatter();
  drawLoss();
  updateStatusBar();
}

function bindSlider(id: string, labelId: string, fmt?: (v: number) => string): void {
  const slider = document.getElementById(id) as HTMLInputElement;
  const label = document.getElementById(labelId) as HTMLElement;
  label.textContent = fmt ? fmt(parseFloat(slider.value)) : slider.value;
  slider.addEventListener('input', () => {
    label.textContent = fmt ? fmt(parseFloat(slider.value)) : slider.value;
  });
}

function renderLegend(): void {
  const container = document.getElementById('legend-items')!;
  const entries = Object.entries(CATEGORY_COLORS);
  container.innerHTML = entries.map(([cat, col]) =>
    `<span class="legend-chip" style="background:${col}22;border:1px solid ${col};color:${col}">${cat}</span>`
  ).join('');
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  initScatter();
  initLossCanvas();
  renderLegend();

  bindSlider('slider-lr', 'label-lr', v => v.toFixed(3));
  bindSlider('slider-dim', 'label-dim');
  bindSlider('slider-win', 'label-win');
  bindSlider('slider-neg', 'label-neg');

  document.getElementById('btn-play')!.addEventListener('click', play);
  document.getElementById('btn-pause')!.addEventListener('click', pause);
  document.getElementById('btn-step')!.addEventListener('click', step);
  document.getElementById('btn-reset')!.addEventListener('click', reset);

  document.getElementById('toggle-pca')!.addEventListener('change', () => {
    state.params.usePca = (document.getElementById('toggle-pca') as HTMLInputElement).checked;
    updateScatter();
  });

  document.getElementById('nn-word-btn')!.addEventListener('click', () => {
    const w = (document.getElementById('nn-word-input') as HTMLInputElement).value.trim().toLowerCase();
    renderNeighbors(w);
  });

  document.getElementById('nn-word-input')!.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key === 'Enter') {
      const w = (document.getElementById('nn-word-input') as HTMLInputElement).value.trim().toLowerCase();
      renderNeighbors(w);
    }
  });

  document.getElementById('analogy-btn')!.addEventListener('click', renderAnalogy);

  // populate word datalist
  const dl = document.getElementById('word-list') as HTMLDataListElement;
  WORDS.forEach(wm => {
    const opt = document.createElement('option');
    opt.value = wm.word;
    dl.appendChild(opt);
  });

  const params = readParams();
  initState(params);
  updateScatter();
  drawLoss();
  updateStatusBar();

  // auto-start
  play();
});
