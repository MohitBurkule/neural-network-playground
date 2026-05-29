const fs = require('fs');
const { JSDOM } = require('jsdom');
const path = '/home/user/neural-network-playground/dist/';
const html = fs.readFileSync(path + 'index.html', 'utf8');
const dom = new JSDOM(html, { runScripts: 'outside-only', pretendToBeVisual: true });
const { window } = dom;
global.window = window; global.document = window.document;
global.navigator = window.navigator;
window.HTMLCanvasElement.prototype.getContext = function() {
  return new Proxy({}, { get: () => () => ({ data: [] }) });
};
window.SVGElement.prototype.getTotalLength = () => 100;
window.SVGElement.prototype.getPointAtLength = () => ({ x: 0, y: 0 });
global.THREE = window.THREE = new Proxy(function(){return new Proxy({}, h);}, h={ get: () => global.THREE, apply: () => new Proxy({}, h) });
window.requestAnimationFrame = () => 0;
const ctx = window;
function run(file){ const code = fs.readFileSync(path+file,'utf8'); new window.Function(code).call(window); }
try {
  run('lib.js');
  run('bundle.js');
} catch(e){ console.error('EXCEPTION', e); process.exit(1); }
const d = window.document;
const nodes = d.querySelectorAll('#svg g.core g.node');
console.log('node count', nodes.length);
const lossTrain = d.querySelector('#loss-train');
const accTest = d.querySelector('#acc-test');
console.log('loss-train', JSON.stringify(lossTrain && lossTrain.textContent));
console.log('acc-test', JSON.stringify(accTest && accTest.textContent));
if(nodes.length<=0) throw new Error('no nodes');
if(!lossTrain || !lossTrain.textContent.trim()) throw new Error('loss empty');
if(!accTest || !accTest.textContent.trim()) throw new Error('acc empty');
console.log('SMOKE OK');
