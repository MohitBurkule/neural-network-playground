/**
 * Runtime smoke test for the standalone labs.
 *
 * Each lab is only type-checked at build time; this harness loads every lab's
 * built page + bundle in jsdom (with canvas / SVG / rAF stubs) and asserts the
 * module initializes without throwing. Run AFTER `npm run build-all`.
 *
 * Usage: node scripts/smoke-labs.js   (or: npm run test:labs)
 * Exits non-zero if any lab fails to initialize.
 */
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const ROOT = path.join(__dirname, '..');
const DIST = path.join(ROOT, 'dist');

const LABS = [
  'cnn', 'transformer', 'rnn', 'autoencoder', 'gan', 'diffusion', 'word2vec',
  'bayesnn', 'dtree', 'svm', 'glm', 'gp', 'clustering', 'dimred', 'som',
  'rl', 'hopfield', 'genetic', 'bandit', 'optviz', 'markov', 'mcmc',
  'ca', 'fourier', 'kalman', 'rbm', 'vae', 'pso', 'snn', 'rd',
  'boids', 'aco', 'nca', 'knn', 'pathfind', 'attractor', 'fractal', 'actfn',
  'perceptron', 'conv', 'esn', 'mdn',
  'hmm', 'flows', 'minimax', 'linalg', 'pendulum', 'nbody', 'wave',
  'perlin', 'lsystem'
];

function makeCtx() {
  const ctx = new Proxy({}, {
    get: (t, p) => {
      if (p === 'createImageData' || p === 'getImageData') {
        return (w, h) => ({
          data: new Uint8ClampedArray((Math.abs(w | 0) || 1) * (Math.abs(h | 0) || 1) * 4),
          width: (w | 0) || 1, height: (h | 0) || 1
        });
      }
      if (p === 'measureText') return () => ({ width: 0 });
      if (p === 'getContext') return () => ctx;
      if (p === 'canvas') return { width: 300, height: 150 };
      return () => {};
    }
  });
  return ctx;
}

let fails = 0;
for (const lab of LABS) {
  const htmlPath = path.join(DIST, `${lab}.html`);
  const bundlePath = path.join(DIST, `bundle${lab}.js`);
  if (!fs.existsSync(htmlPath) || !fs.existsSync(bundlePath)) {
    console.log(`MISSING ${lab} (run \`npm run build-all\` first)`);
    fails++;
    continue;
  }
  try {
    const dom = new JSDOM(fs.readFileSync(htmlPath, 'utf8'),
      { runScripts: 'outside-only', pretendToBeVisual: true, url: 'http://localhost/' });
    const { window } = dom;
    const ctx = makeCtx();
    window.HTMLCanvasElement.prototype.getContext = () => ctx;
    window.HTMLCanvasElement.prototype.toDataURL = () => 'data:,';
    window.requestAnimationFrame = () => 0;
    window.cancelAnimationFrame = () => {};
    window.matchMedia = window.matchMedia || (q => ({
      matches: false, media: q, addListener() {}, removeListener() {},
      addEventListener() {}, removeEventListener() {}
    }));
    const sp = window.SVGElement && window.SVGElement.prototype;
    if (sp) {
      sp.getTotalLength = () => 100;
      sp.getPointAtLength = () => ({ x: 0, y: 0 });
      sp.getBBox = () => ({ x: 0, y: 0, width: 100, height: 100 });
    }
    global.window = window;
    global.document = window.document;
    global.navigator = window.navigator;
    window.eval(fs.readFileSync(bundlePath, 'utf8'));
    console.log(`OK    ${lab}`);
  } catch (e) {
    console.log(`THROW ${lab}: ${(e && e.message || e).toString().split('\n')[0]}`);
    fails++;
  }
}

console.log(`\n=== ${LABS.length - fails}/${LABS.length} labs initialized OK; ${fails} failing ===`);
process.exit(fails === 0 ? 0 : 1);
