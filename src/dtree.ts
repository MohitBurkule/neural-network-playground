import * as d3 from 'd3';

// ─── Types ───────────────────────────────────────────────────────────────────

interface Sample {
  x: number;
  y: number;
  label: number;
}

type Criterion = 'gini' | 'entropy';
type DatasetName = 'circle' | 'xor' | 'spiral' | 'moons' | 'blobs' | 'fourquadrants';
type ModelType = 'tree' | 'forest';

interface TreeNode {
  isLeaf: boolean;
  label?: number;       // only if leaf
  feature?: number;     // 0=x, 1=y
  threshold?: number;
  impurity?: number;
  samples?: number;
  left?: TreeNode;
  right?: TreeNode;
  depth?: number;
}

// ─── State ───────────────────────────────────────────────────────────────────

let state = {
  dataset: 'circle' as DatasetName,
  model: 'tree' as ModelType,
  criterion: 'gini' as Criterion,
  maxDepth: 4,
  minSamples: 5,
  nTrees: 10,
  nPoints: 200,
  noise: 0.1,
};

let trainData: Sample[] = [];
let treeRoot: TreeNode | null = null;
let forest: TreeNode[] = [];

// ─── Random helpers ──────────────────────────────────────────────────────────

function randn(): number {
  const u = Math.random() + 1e-10;
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ─── Dataset generators ───────────────────────────────────────────────────────

function genCircle(n: number, noise: number): Sample[] {
  const pts: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const r = Math.random() < 0.5 ? Math.random() * 0.4 + 0.05 : Math.random() * 0.35 + 0.55;
    const label = r < 0.5 ? 0 : 1;
    const theta = Math.random() * 2 * Math.PI;
    pts.push({
      x: r * Math.cos(theta) + randn() * noise,
      y: r * Math.sin(theta) + randn() * noise,
      label
    });
  }
  return pts;
}

function genXOR(n: number, noise: number): Sample[] {
  const pts: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() - 0.5) * 2;
    const y = (Math.random() - 0.5) * 2;
    const label = (x * y > 0) ? 0 : 1;
    pts.push({ x: x + randn() * noise, y: y + randn() * noise, label });
  }
  return pts;
}

function genSpiral(n: number, noise: number): Sample[] {
  const pts: Sample[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = (i / half) * 3 * Math.PI;
    const r = t / (3 * Math.PI);
    pts.push({ x: r * Math.cos(t) + randn() * noise * 0.5, y: r * Math.sin(t) + randn() * noise * 0.5, label: 0 });
  }
  for (let i = 0; i < n - half; i++) {
    const t = (i / half) * 3 * Math.PI + Math.PI;
    const r = t / (3 * Math.PI);
    pts.push({ x: r * Math.cos(t) + randn() * noise * 0.5, y: r * Math.sin(t) + randn() * noise * 0.5, label: 1 });
  }
  return pts;
}

function genMoons(n: number, noise: number): Sample[] {
  const pts: Sample[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = (i / half) * Math.PI;
    pts.push({ x: Math.cos(t) + randn() * noise, y: Math.sin(t) + randn() * noise, label: 0 });
  }
  for (let i = 0; i < n - half; i++) {
    const t = (i / (n - half)) * Math.PI;
    pts.push({ x: 1 - Math.cos(t) + randn() * noise, y: 0.5 - Math.sin(t) + randn() * noise, label: 1 });
  }
  return pts;
}

function genBlobs(n: number, noise: number): Sample[] {
  const centers: [number, number, number][] = [
    [-0.6, -0.6, 0], [0.6, 0.6, 1], [-0.6, 0.6, 2], [0.6, -0.6, 3]
  ];
  const pts: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const c = centers[i % centers.length];
    pts.push({ x: c[0] + randn() * (0.15 + noise), y: c[1] + randn() * (0.15 + noise), label: c[2] });
  }
  return pts;
}

function genFourQuadrants(n: number, noise: number): Sample[] {
  const pts: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const x = (Math.random() - 0.5) * 2;
    const y = (Math.random() - 0.5) * 2;
    const label = (x > 0 && y > 0) ? 0 : (x < 0 && y > 0) ? 1 : (x < 0 && y < 0) ? 2 : 3;
    pts.push({ x: x + randn() * noise * 0.5, y: y + randn() * noise * 0.5, label });
  }
  return pts;
}

function generateDataset(): Sample[] {
  const { dataset, nPoints, noise } = state;
  switch (dataset) {
    case 'circle': return genCircle(nPoints, noise);
    case 'xor': return genXOR(nPoints, noise);
    case 'spiral': return genSpiral(nPoints, noise);
    case 'moons': return genMoons(nPoints, noise);
    case 'blobs': return genBlobs(nPoints, noise);
    case 'fourquadrants': return genFourQuadrants(nPoints, noise);
    default: return genCircle(nPoints, noise);
  }
}

// ─── CART Decision Tree ───────────────────────────────────────────────────────

function gini(labels: number[]): number {
  if (labels.length === 0) return 0;
  const counts: { [k: number]: number } = {};
  labels.forEach(function(l) { counts[l] = (counts[l] || 0) + 1; });
  let g = 1;
  const n = labels.length;
  Object.keys(counts).forEach(function(k) {
    const p = counts[+k] / n;
    g -= p * p;
  });
  return g;
}

function entropy(labels: number[]): number {
  if (labels.length === 0) return 0;
  const counts: { [k: number]: number } = {};
  labels.forEach(function(l) { counts[l] = (counts[l] || 0) + 1; });
  let e = 0;
  const n = labels.length;
  Object.keys(counts).forEach(function(k) {
    const p = counts[+k] / n;
    if (p > 0) e -= p * Math.log2(p);
  });
  return e;
}

function impurityFn(labels: number[]): number {
  return state.criterion === 'gini' ? gini(labels) : entropy(labels);
}

function majorityLabel(labels: number[]): number {
  const counts: { [k: number]: number } = {};
  labels.forEach(function(l) { counts[l] = (counts[l] || 0) + 1; });
  let best = 0, bestCount = -1;
  Object.keys(counts).forEach(function(k) {
    if (counts[+k] > bestCount) { bestCount = counts[+k]; best = +k; }
  });
  return best;
}

function buildTree(
  data: Sample[],
  depth: number,
  maxDepth: number,
  minSamples: number,
  features: number[]
): TreeNode {
  const labels = data.map(function(d) { return d.label; });
  const imp = impurityFn(labels);
  const majLabel = majorityLabel(labels);

  if (depth >= maxDepth || data.length < minSamples || imp < 1e-10) {
    return { isLeaf: true, label: majLabel, impurity: imp, samples: data.length, depth };
  }

  let bestFeature = -1, bestThreshold = 0, bestGain = -Infinity;
  let bestLeft: Sample[] = [], bestRight: Sample[] = [];
  const n = data.length;

  features.forEach(function(feat) {
    const vals = data.map(function(d) { return feat === 0 ? d.x : d.y; });
    const sorted = vals.slice().sort(function(a, b) { return a - b; });
    // sample up to 20 candidate thresholds for speed
    const step = Math.max(1, Math.floor(sorted.length / 20));
    for (let i = 0; i < sorted.length - 1; i += step) {
      const threshold = (sorted[i] + sorted[i + 1]) / 2;
      const left: Sample[] = [], right: Sample[] = [];
      data.forEach(function(d) {
        const v = feat === 0 ? d.x : d.y;
        if (v <= threshold) left.push(d); else right.push(d);
      });
      if (left.length === 0 || right.length === 0) continue;
      const gain = imp
        - (left.length / n) * impurityFn(left.map(function(d) { return d.label; }))
        - (right.length / n) * impurityFn(right.map(function(d) { return d.label; }));
      if (gain > bestGain) {
        bestGain = gain; bestFeature = feat; bestThreshold = threshold;
        bestLeft = left; bestRight = right;
      }
    }
  });

  if (bestFeature === -1 || bestGain <= 0) {
    return { isLeaf: true, label: majLabel, impurity: imp, samples: data.length, depth };
  }

  return {
    isLeaf: false,
    feature: bestFeature,
    threshold: bestThreshold,
    impurity: imp,
    samples: data.length,
    depth,
    left: buildTree(bestLeft, depth + 1, maxDepth, minSamples, features),
    right: buildTree(bestRight, depth + 1, maxDepth, minSamples, features),
  };
}

function predictTree(node: TreeNode, x: number, y: number): number {
  if (node.isLeaf) return node.label!;
  const val = node.feature === 0 ? x : y;
  if (val <= node.threshold!) return predictTree(node.left!, x, y);
  return predictTree(node.right!, x, y);
}

// Bootstrap sample
function bootstrap(data: Sample[]): Sample[] {
  const result: Sample[] = [];
  for (let i = 0; i < data.length; i++) {
    result.push(data[Math.floor(Math.random() * data.length)]);
  }
  return result;
}

function buildForest(data: Sample[]): TreeNode[] {
  const trees: TreeNode[] = [];
  for (let t = 0; t < state.nTrees; t++) {
    const bag = bootstrap(data);
    // random feature subset: pick 1 feature randomly per tree or use both
    const features = Math.random() < 0.5 ? [0] : [0, 1];
    trees.push(buildTree(bag, 0, state.maxDepth, state.minSamples, features));
  }
  return trees;
}

function predictForest(trees: TreeNode[], x: number, y: number): number {
  const votes: { [k: number]: number } = {};
  trees.forEach(function(t) {
    const lbl = predictTree(t, x, y);
    votes[lbl] = (votes[lbl] || 0) + 1;
  });
  let best = 0, bestCount = -1;
  Object.keys(votes).forEach(function(k) {
    if (votes[+k] > bestCount) { bestCount = votes[+k]; best = +k; }
  });
  return best;
}

function computeAccuracy(): number {
  if (!trainData.length) return 0;
  let correct = 0;
  trainData.forEach(function(d) {
    const pred = state.model === 'tree' && treeRoot
      ? predictTree(treeRoot, d.x, d.y)
      : predictForest(forest, d.x, d.y);
    if (pred === d.label) correct++;
  });
  return correct / trainData.length;
}

// ─── Colour palette ──────────────────────────────────────────────────────────

const CLASS_COLORS = ['#4e9af1', '#f1714e', '#4ef1a0', '#f1d44e', '#c14ef1'];
const CLASS_COLORS_LIGHT = ['rgba(78,154,241,0.35)', 'rgba(241,113,78,0.35)', 'rgba(78,241,160,0.35)', 'rgba(241,212,78,0.35)', 'rgba(193,78,241,0.35)'];

// ─── Canvas: Decision boundary ───────────────────────────────────────────────

const GRID_RES = 80; // grid cells per axis

function drawBoundary(canvas: HTMLCanvasElement, xScale: d3.ScaleLinear<number,number>, yScale: d3.ScaleLinear<number,number>) {
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const xMin = xScale.domain()[0], xMax = xScale.domain()[1];
  const yMin = yScale.domain()[0], yMax = yScale.domain()[1];
  const cellW = W / GRID_RES, cellH = H / GRID_RES;

  for (let gi = 0; gi < GRID_RES; gi++) {
    for (let gj = 0; gj < GRID_RES; gj++) {
      const px = xMin + (gi + 0.5) / GRID_RES * (xMax - xMin);
      const py = yMin + (gj + 0.5) / GRID_RES * (yMax - yMin);
      let label: number;
      if (state.model === 'tree' && treeRoot) {
        label = predictTree(treeRoot, px, py);
      } else if (forest.length > 0) {
        label = predictForest(forest, px, py);
      } else {
        continue;
      }
      ctx.fillStyle = CLASS_COLORS_LIGHT[label % CLASS_COLORS_LIGHT.length];
      ctx.fillRect(gi * cellW, gj * cellH, cellW + 1, cellH + 1);
    }
  }
}

// ─── SVG: Training points overlay ────────────────────────────────────────────

function drawPoints(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  xScale: d3.ScaleLinear<number,number>,
  yScale: d3.ScaleLinear<number,number>
) {
  svg.selectAll('.pt').remove();
  svg.selectAll('.pt')
    .data(trainData)
    .enter()
    .append('circle')
    .attr('class', 'pt')
    .attr('cx', function(d) { return xScale(d.x); })
    .attr('cy', function(d) { return yScale(d.y); })
    .attr('r', 4)
    .attr('fill', function(d) { return CLASS_COLORS[d.label % CLASS_COLORS.length]; })
    .attr('stroke', '#0008')
    .attr('stroke-width', 0.8);
}

// ─── SVG: Tree diagram ───────────────────────────────────────────────────────

function treeToD3Hierarchy(node: TreeNode): d3.HierarchyNode<TreeNode> {
  return d3.hierarchy<TreeNode>(node, function(d) {
    if (d.isLeaf) return null;
    const children: TreeNode[] = [];
    if (d.left) children.push(d.left);
    if (d.right) children.push(d.right);
    return children;
  });
}

function drawTree(svgEl: SVGSVGElement, root: TreeNode) {
  const svg = d3.select(svgEl);
  svg.selectAll('*').remove();

  const W = svgEl.clientWidth || 560;
  const H = svgEl.clientHeight || 340;
  const margin = { top: 20, right: 10, bottom: 10, left: 10 };
  const iw = W - margin.left - margin.right;
  const ih = H - margin.top - margin.bottom;

  const g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

  const hier = treeToD3Hierarchy(root);
  const layout = d3.tree<TreeNode>().size([iw, ih]);
  const treeData = layout(hier);

  // Links
  g.selectAll('.link')
    .data(treeData.links())
    .enter()
    .append('path')
    .attr('class', 'link')
    .attr('fill', 'none')
    .attr('stroke', '#4a4a7a')
    .attr('stroke-width', 1.5)
    .attr('d', function(d) {
      return 'M' + d.source.x + ',' + d.source.y
        + 'C' + d.source.x + ',' + (d.source.y + d.target.y) / 2
        + ' ' + d.target.x + ',' + (d.source.y + d.target.y) / 2
        + ' ' + d.target.x + ',' + d.target.y;
    });

  // Nodes
  const node = g.selectAll('.node')
    .data(treeData.descendants())
    .enter()
    .append('g')
    .attr('class', 'node')
    .attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; });

  node.append('circle')
    .attr('r', 14)
    .attr('fill', function(d) {
      if (d.data.isLeaf) return CLASS_COLORS[d.data.label! % CLASS_COLORS.length];
      return '#1e2a4a';
    })
    .attr('stroke', function(d) {
      if (d.data.isLeaf) return CLASS_COLORS[d.data.label! % CLASS_COLORS.length];
      return '#6060aa';
    })
    .attr('stroke-width', 2);

  node.append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em')
    .attr('font-size', '9px')
    .attr('fill', '#e0e0f0')
    .text(function(d) {
      if (d.data.isLeaf) return 'C' + d.data.label;
      const feat = d.data.feature === 0 ? 'x' : 'y';
      return feat + '≤' + (d.data.threshold!.toFixed(2));
    });
}

// ─── Full render ─────────────────────────────────────────────────────────────

let boundaryCanvas: HTMLCanvasElement;
let overlaySvg: d3.Selection<SVGSVGElement, unknown, null, undefined>;
let treeSvg: SVGSVGElement;
let xScale: d3.ScaleLinear<number, number>;
let yScale: d3.ScaleLinear<number, number>;

function getDataExtents(): [[number, number], [number, number]] {
  if (!trainData.length) return [[-1.5, 1.5], [-1.5, 1.5]];
  const xs = trainData.map(function(d) { return d.x; });
  const ys = trainData.map(function(d) { return d.y; });
  const xMin = Math.min.apply(null, xs), xMax = Math.max.apply(null, xs);
  const yMin = Math.min.apply(null, ys), yMax = Math.max.apply(null, ys);
  const xPad = (xMax - xMin) * 0.1 + 0.1;
  const yPad = (yMax - yMin) * 0.1 + 0.1;
  return [[xMin - xPad, xMax + xPad], [yMin - yPad, yMax + yPad]];
}

function render() {
  const [[xMin, xMax], [yMin, yMax]] = getDataExtents();
  const W = boundaryCanvas.width, H = boundaryCanvas.height;

  xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, W]);
  yScale = d3.scaleLinear().domain([yMin, yMax]).range([0, H]);

  drawBoundary(boundaryCanvas, xScale, yScale);
  drawPoints(overlaySvg, xScale, yScale);

  const acc = computeAccuracy();
  const accEl = document.getElementById('accuracy');
  if (accEl) accEl.textContent = 'Accuracy: ' + (acc * 100).toFixed(1) + '%';

  const rootToShow = state.model === 'tree' ? treeRoot : (forest.length > 0 ? forest[0] : null);
  if (rootToShow) drawTree(treeSvg, rootToShow);
  else {
    d3.select(treeSvg).selectAll('*').remove();
    d3.select(treeSvg).append('text')
      .attr('x', 20).attr('y', 30)
      .attr('fill', '#8888aa').attr('font-size', '13px')
      .text('No model trained yet.');
  }
}

function rebuild() {
  trainData = generateDataset();
  if (state.model === 'tree') {
    treeRoot = buildTree(trainData, 0, state.maxDepth, state.minSamples, [0, 1]);
    forest = [];
  } else {
    forest = buildForest(trainData);
    treeRoot = null;
  }
  render();
}

// ─── UI setup ────────────────────────────────────────────────────────────────

function numericControl(id: string, key: keyof typeof state, isFloat: boolean) {
  const el = document.getElementById(id) as HTMLInputElement | null;
  if (!el) return;
  el.value = String(state[key]);
  el.addEventListener('input', function() {
    const v = isFloat ? parseFloat(el.value) : parseInt(el.value, 10);
    if (!isNaN(v)) (state as any)[key] = v;
    updateLabels();
  });
}

function selectControl(id: string, key: keyof typeof state) {
  const el = document.getElementById(id) as HTMLSelectElement | null;
  if (!el) return;
  el.value = String(state[key]);
  el.addEventListener('change', function() {
    (state as any)[key] = el.value;
    updateForestVisibility();
  });
}

function updateLabels() {
  const ids: Array<[string, keyof typeof state]> = [
    ['lbl-maxdepth', 'maxDepth'], ['lbl-minsamples', 'minSamples'],
    ['lbl-ntrees', 'nTrees'], ['lbl-npoints', 'nPoints'], ['lbl-noise', 'noise']
  ];
  ids.forEach(function(pair) {
    const el = document.getElementById(pair[0]);
    if (el) {
      const v = state[pair[1]];
      el.textContent = pair[1] === 'noise' ? (+v).toFixed(2) : String(v);
    }
  });
}

function updateForestVisibility() {
  const forestSection = document.getElementById('forest-section');
  if (forestSection) {
    forestSection.style.display = state.model === 'forest' ? '' : 'none';
  }
}

function setupUI() {
  selectControl('sel-dataset', 'dataset');
  selectControl('sel-model', 'model');
  selectControl('sel-criterion', 'criterion');

  numericControl('sl-maxdepth', 'maxDepth', false);
  numericControl('sl-minsamples', 'minSamples', false);
  numericControl('sl-ntrees', 'nTrees', false);
  numericControl('sl-npoints', 'nPoints', false);
  numericControl('sl-noise', 'noise', true);

  updateLabels();
  updateForestVisibility();

  const btn = document.getElementById('btn-rebuild');
  if (btn) btn.addEventListener('click', rebuild);
}

// ─── Init ────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function() {
  boundaryCanvas = document.getElementById('boundary-canvas') as HTMLCanvasElement;
  const overlaySvgEl = document.getElementById('overlay-svg') as unknown as SVGSVGElement;
  overlaySvg = d3.select(overlaySvgEl);
  treeSvg = document.getElementById('tree-svg') as unknown as SVGSVGElement;

  // Sync canvas size to its container
  function sizeCanvas() {
    const container = boundaryCanvas.parentElement!;
    const size = container.clientWidth || 400;
    boundaryCanvas.width = size;
    boundaryCanvas.height = size;
    overlaySvgEl.setAttribute('width', String(size));
    overlaySvgEl.setAttribute('height', String(size));
  }
  sizeCanvas();

  setupUI();
  rebuild();
});
