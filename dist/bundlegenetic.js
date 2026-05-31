(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var DOMAIN_MIN = -5.12;
var DOMAIN_MAX = 5.12;
function rastriginRaw(x, y) {
    return 20 + x * x + y * y
        - 10 * (Math.cos(2 * Math.PI * x) + Math.cos(2 * Math.PI * y));
}
function ackleyRaw(x, y) {
    var a = 20, b = 0.2, c = 2 * Math.PI;
    var sq = Math.sqrt(0.5 * (x * x + y * y));
    var cos = 0.5 * (Math.cos(c * x) + Math.cos(c * y));
    return -a * Math.exp(-b * sq) - Math.exp(cos) + a + Math.E;
}
function rosenbrockRaw(x, y) {
    return (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x);
}
function sphereRaw(x, y) {
    return x * x + y * y;
}
function himmelblauRaw(x, y) {
    return (x * x + y - 11) * (x * x + y - 11)
        + (x + y * y - 7) * (x + y * y - 7);
}
var FN_META = {
    rastrigin: { raw: rastriginRaw, dir: 'minimize', label: 'Rastrigin (minimize)' },
    ackley: { raw: ackleyRaw, dir: 'minimize', label: 'Ackley (minimize)' },
    rosenbrock: { raw: rosenbrockRaw, dir: 'minimize', label: 'Rosenbrock (minimize)' },
    sphere: { raw: sphereRaw, dir: 'minimize', label: 'Sphere (minimize)' },
    himmelblau: { raw: himmelblauRaw, dir: 'minimize', label: 'Himmelblau (minimize)' }
};
function evalFitness(fn, x, y) {
    var meta = FN_META[fn];
    var raw = meta.raw(x, y);
    return meta.dir === 'minimize' ? -raw : raw;
}
var GRID = 120;
function buildHeatmap(fn) {
    var data = new Float32Array(GRID * GRID);
    var min = Infinity, max = -Infinity;
    for (var row = 0; row < GRID; row++) {
        for (var col = 0; col < GRID; col++) {
            var x = DOMAIN_MIN + (col / (GRID - 1)) * (DOMAIN_MAX - DOMAIN_MIN);
            var y = DOMAIN_MIN + (row / (GRID - 1)) * (DOMAIN_MAX - DOMAIN_MIN);
            var v = FN_META[fn].raw(x, y);
            data[row * GRID + col] = v;
            if (v < min)
                min = v;
            if (v > max)
                max = v;
        }
    }
    return { data: data, min: min, max: max };
}
function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
}
function randn() {
    var u = Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u + 1e-12)) * Math.cos(2 * Math.PI * v);
}
function createIndividual(fn) {
    var x = DOMAIN_MIN + Math.random() * (DOMAIN_MAX - DOMAIN_MIN);
    var y = DOMAIN_MIN + Math.random() * (DOMAIN_MAX - DOMAIN_MIN);
    return { x: x, y: y, fitness: evalFitness(fn, x, y) };
}
function tournamentSelect(pop) {
    var k = 3;
    var best = pop[Math.floor(Math.random() * pop.length)];
    for (var i = 1; i < k; i++) {
        var cand = pop[Math.floor(Math.random() * pop.length)];
        if (cand.fitness > best.fitness)
            best = cand;
    }
    return best;
}
function rouletteSelect(pop, totalFit, minFit) {
    var shifted = pop.map(function (ind) { return ind.fitness - minFit + 1e-6; });
    var shiftedTotal = shifted.reduce(function (a, b) { return a + b; }, 0);
    var r = Math.random() * shiftedTotal;
    for (var i = 0; i < pop.length; i++) {
        r -= shifted[i];
        if (r <= 0)
            return pop[i];
    }
    return pop[pop.length - 1];
}
function crossover(a, b, crossRate, fn) {
    if (Math.random() > crossRate) {
        var src = Math.random() < 0.5 ? a : b;
        return { x: src.x, y: src.y, fitness: src.fitness };
    }
    var alpha = 0.3;
    var xLo = Math.min(a.x, b.x), xHi = Math.max(a.x, b.x);
    var yLo = Math.min(a.y, b.y), yHi = Math.max(a.y, b.y);
    var dx = xHi - xLo, dy = yHi - yLo;
    var cx = clamp(xLo - alpha * dx + Math.random() * (dx * (1 + 2 * alpha)), DOMAIN_MIN, DOMAIN_MAX);
    var cy = clamp(yLo - alpha * dy + Math.random() * (dy * (1 + 2 * alpha)), DOMAIN_MIN, DOMAIN_MAX);
    return { x: cx, y: cy, fitness: evalFitness(fn, cx, cy) };
}
function mutate(ind, mutRate, mutSigma, fn) {
    var x = ind.x, y = ind.y;
    if (Math.random() < mutRate)
        x = clamp(x + randn() * mutSigma, DOMAIN_MIN, DOMAIN_MAX);
    if (Math.random() < mutRate)
        y = clamp(y + randn() * mutSigma, DOMAIN_MIN, DOMAIN_MAX);
    return { x: x, y: y, fitness: evalFitness(fn, x, y) };
}
function stepGeneration(pop, params) {
    var fn = params.fn, popSize = params.popSize, mutRate = params.mutRate, mutSigma = params.mutSigma, crossRate = params.crossRate, selection = params.selection, elitism = params.elitism;
    pop.sort(function (a, b) { return b.fitness - a.fitness; });
    var next = [];
    var eliteCount = Math.min(elitism, popSize);
    for (var i = 0; i < eliteCount; i++) {
        next.push({ x: pop[i].x, y: pop[i].y, fitness: pop[i].fitness });
    }
    var minFit = pop[pop.length - 1].fitness;
    var totalFit = pop.reduce(function (s, ind) { return s + ind.fitness; }, 0);
    while (next.length < popSize) {
        var parent1 = void 0, parent2 = void 0;
        if (selection === 'tournament') {
            parent1 = tournamentSelect(pop);
            parent2 = tournamentSelect(pop);
        }
        else {
            parent1 = rouletteSelect(pop, totalFit, minFit);
            parent2 = rouletteSelect(pop, totalFit, minFit);
        }
        var child = crossover(parent1, parent2, crossRate, fn);
        var mutated = mutate(child, mutRate, mutSigma, fn);
        next.push(mutated);
    }
    return next;
}
var canvasLandscape = document.getElementById('canvas-landscape');
var canvasCurve = document.getElementById('canvas-curve');
var ctxL = canvasLandscape.getContext('2d');
var ctxC = canvasCurve.getContext('2d');
function getEl(id) {
    return document.getElementById(id);
}
var selFn = getEl('sel-fn');
var selSel = getEl('sel-sel');
var sliderPop = getEl('slider-pop');
var sliderMutRate = getEl('slider-mutrate');
var sliderMutSigma = getEl('slider-mutsigma');
var sliderCross = getEl('slider-cross');
var sliderElite = getEl('slider-elite');
var sliderSpeed = getEl('slider-speed');
var btnPlay = getEl('btn-play');
var btnPause = getEl('btn-pause');
var btnStep = getEl('btn-step');
var btnReset = getEl('btn-reset');
var spanGen = getEl('span-gen');
var spanBest = getEl('span-best');
var spanMean = getEl('span-mean');
var labelPop = getEl('label-pop');
var labelMutRate = getEl('label-mutrate');
var labelMutSigma = getEl('label-mutsigma');
var labelCross = getEl('label-cross');
var labelElite = getEl('label-elite');
var labelSpeed = getEl('label-speed');
var population = [];
var history = [];
var generation = 0;
var running = false;
var animHandle = null;
var lastFrameTime = 0;
var heatmap = null;
var heatImageData = null;
var COLORS = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
];
function colorAt(t) {
    t = clamp(t, 0, 1);
    var scaled = t * (COLORS.length - 1);
    var lo = Math.floor(scaled);
    var hi = Math.min(lo + 1, COLORS.length - 1);
    var f = scaled - lo;
    var a = COLORS[lo], b = COLORS[hi];
    return [
        Math.round(a[0] + f * (b[0] - a[0])),
        Math.round(a[1] + f * (b[1] - a[1])),
        Math.round(a[2] + f * (b[2] - a[2])),
    ];
}
function getParams() {
    return {
        fn: selFn.value,
        popSize: parseInt(sliderPop.value, 10),
        mutRate: parseFloat(sliderMutRate.value),
        mutSigma: parseFloat(sliderMutSigma.value),
        crossRate: parseFloat(sliderCross.value),
        selection: selSel.value,
        elitism: parseInt(sliderElite.value, 10),
        speed: parseInt(sliderSpeed.value, 10)
    };
}
function buildHeatImageData(fn, W, H) {
    heatmap = buildHeatmap(fn);
    var imgData = ctxL.createImageData(W, H);
    var data = heatmap.data, min = heatmap.min, max = heatmap.max;
    var range = max - min || 1;
    for (var row = 0; row < H; row++) {
        for (var col = 0; col < W; col++) {
            var gridCol = Math.round((col / (W - 1)) * (GRID - 1));
            var gridRow = Math.round((row / (H - 1)) * (GRID - 1));
            var v = data[gridRow * GRID + gridCol];
            var t = (v - min) / range;
            var tDisplay = FN_META[fn].dir === 'minimize' ? 1 - t : t;
            var _a = colorAt(tDisplay), r = _a[0], g = _a[1], b = _a[2];
            var idx = (row * W + col) * 4;
            imgData.data[idx] = r;
            imgData.data[idx + 1] = g;
            imgData.data[idx + 2] = b;
            imgData.data[idx + 3] = 255;
        }
    }
    return imgData;
}
function domainToCanvas(v, W) {
    return ((v - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * W;
}
function drawLandscape(params) {
    var W = canvasLandscape.width;
    var H = canvasLandscape.height;
    if (heatImageData) {
        ctxL.putImageData(heatImageData, 0, 0);
    }
    else {
        ctxL.fillStyle = '#0d0d1e';
        ctxL.fillRect(0, 0, W, H);
    }
    var fn = params.fn;
    var bestIdx = 0;
    population.forEach(function (ind, i) {
        if (ind.fitness > population[bestIdx].fitness)
            bestIdx = i;
    });
    population.forEach(function (ind, i) {
        var px = domainToCanvas(ind.x, W);
        var py = domainToCanvas(ind.y, H);
        var isBest = i === bestIdx;
        ctxL.beginPath();
        ctxL.arc(px, py, isBest ? 6 : 3.5, 0, 2 * Math.PI);
        if (isBest) {
            ctxL.fillStyle = '#ffffff';
            ctxL.strokeStyle = '#ff4444';
            ctxL.lineWidth = 2;
            ctxL.fill();
            ctxL.stroke();
        }
        else {
            ctxL.fillStyle = 'rgba(255,220,80,0.75)';
            ctxL.fill();
        }
    });
    if (population.length > 0) {
        var best = population[bestIdx];
        var px = domainToCanvas(best.x, W);
        var py = domainToCanvas(best.y, H);
        var rawBest = FN_META[fn].raw(best.x, best.y);
        ctxL.fillStyle = '#fff';
        ctxL.font = '11px monospace';
        ctxL.fillText("(".concat(best.x.toFixed(2), ", ").concat(best.y.toFixed(2), ") \u2192 ").concat(rawBest.toFixed(4)), px + 8, py - 6);
    }
}
function drawCurve() {
    var W = canvasCurve.width;
    var H = canvasCurve.height;
    var pad = { top: 18, right: 14, bottom: 30, left: 48 };
    ctxC.fillStyle = '#0d0d1e';
    ctxC.fillRect(0, 0, W, H);
    if (history.length < 2)
        return;
    var innerW = W - pad.left - pad.right;
    var innerH = H - pad.top - pad.bottom;
    var fn = selFn.value;
    var isMin = FN_META[fn].dir === 'minimize';
    var rawHistory = history.map(function (r) { return ({
        gen: r.gen,
        best: isMin ? -r.best : r.best,
        mean: isMin ? -r.mean : r.mean
    }); });
    var allVals = rawHistory.reduce(function (acc, r) {
        acc.push(r.best, r.mean);
        return acc;
    }, []);
    var yMin = Math.min.apply(Math, allVals);
    var yMax = Math.max.apply(Math, allVals);
    var yRange = yMax - yMin || 1;
    var xScale = function (gen) { return pad.left + (gen / Math.max(history.length - 1, 1)) * innerW; };
    var yScale = function (v) { return pad.top + innerH - ((v - yMin) / yRange) * innerH; };
    ctxC.strokeStyle = 'rgba(255,255,255,0.07)';
    ctxC.lineWidth = 1;
    for (var i = 0; i <= 4; i++) {
        var yy = pad.top + (i / 4) * innerH;
        ctxC.beginPath();
        ctxC.moveTo(pad.left, yy);
        ctxC.lineTo(W - pad.right, yy);
        ctxC.stroke();
        var label = (yMax - (i / 4) * yRange).toFixed(1);
        ctxC.fillStyle = '#778899';
        ctxC.font = '10px monospace';
        ctxC.textAlign = 'right';
        ctxC.fillText(label, pad.left - 4, yy + 3);
    }
    ctxC.fillStyle = '#778899';
    ctxC.font = '10px monospace';
    ctxC.textAlign = 'center';
    var tickCount = Math.min(6, history.length);
    for (var i = 0; i <= tickCount; i++) {
        var gIdx = Math.round((i / tickCount) * (history.length - 1));
        ctxC.fillText(String(rawHistory[gIdx].gen), xScale(gIdx), H - pad.bottom + 14);
    }
    ctxC.beginPath();
    ctxC.strokeStyle = 'rgba(100,180,255,0.6)';
    ctxC.lineWidth = 1.5;
    rawHistory.forEach(function (r, i) {
        var x = xScale(i), y = yScale(r.mean);
        if (i === 0)
            ctxC.moveTo(x, y);
        else
            ctxC.lineTo(x, y);
    });
    ctxC.stroke();
    ctxC.beginPath();
    ctxC.strokeStyle = '#ff6666';
    ctxC.lineWidth = 2;
    rawHistory.forEach(function (r, i) {
        var x = xScale(i), y = yScale(r.best);
        if (i === 0)
            ctxC.moveTo(x, y);
        else
            ctxC.lineTo(x, y);
    });
    ctxC.stroke();
    ctxC.font = '10px sans-serif';
    ctxC.textAlign = 'left';
    ctxC.fillStyle = '#ff6666';
    ctxC.fillText('Best', pad.left + 4, pad.top + 12);
    ctxC.fillStyle = 'rgba(100,180,255,0.9)';
    ctxC.fillText('Mean', pad.left + 36, pad.top + 12);
    ctxC.save();
    ctxC.translate(12, pad.top + innerH / 2);
    ctxC.rotate(-Math.PI / 2);
    ctxC.fillStyle = '#8899aa';
    ctxC.font = '10px sans-serif';
    ctxC.textAlign = 'center';
    ctxC.fillText(isMin ? 'f(x,y) — lower is better' : 'Fitness — higher is better', 0, 0);
    ctxC.restore();
}
function recordGeneration() {
    if (population.length === 0)
        return;
    var fitnesses = population.map(function (ind) { return ind.fitness; });
    var best = fitnesses.reduce(function (a, b) { return Math.max(a, b); }, -Infinity);
    var mean = fitnesses.reduce(function (a, b) { return a + b; }, 0) / fitnesses.length;
    history.push({ gen: generation, best: best, mean: mean });
    var fn = selFn.value;
    var isMin = FN_META[fn].dir === 'minimize';
    spanGen.textContent = String(generation);
    spanBest.textContent = (isMin ? -best : best).toFixed(5);
    spanMean.textContent = (isMin ? -mean : mean).toFixed(5);
}
function doStep() {
    var params = getParams();
    population = stepGeneration(population, params);
    generation++;
    recordGeneration();
    drawLandscape(params);
    drawCurve();
}
function animate(timestamp) {
    if (!running)
        return;
    var params = getParams();
    var delay = params.speed;
    if (timestamp - lastFrameTime >= delay) {
        doStep();
        lastFrameTime = timestamp;
    }
    animHandle = requestAnimationFrame(animate);
}
function reset() {
    running = false;
    if (animHandle !== null) {
        cancelAnimationFrame(animHandle);
        animHandle = null;
    }
    generation = 0;
    history = [];
    var params = getParams();
    population = Array.from({ length: params.popSize }, function () { return createIndividual(params.fn); });
    recordGeneration();
    heatImageData = buildHeatImageData(params.fn, canvasLandscape.width, canvasLandscape.height);
    drawLandscape(params);
    drawCurve();
    updateLabels();
}
function play() {
    if (running)
        return;
    running = true;
    lastFrameTime = 0;
    animHandle = requestAnimationFrame(animate);
}
function pause() {
    running = false;
    if (animHandle !== null) {
        cancelAnimationFrame(animHandle);
        animHandle = null;
    }
}
function resizeCanvases() {
    var panel = document.getElementById('canvas-panel');
    var W = Math.min(panel.clientWidth - 24, 560);
    canvasLandscape.width = W;
    canvasLandscape.height = W;
    canvasCurve.width = W;
    canvasCurve.height = Math.round(W * 0.36);
    var params = getParams();
    heatImageData = buildHeatImageData(params.fn, W, W);
    drawLandscape(params);
    drawCurve();
}
function updateLabels() {
    labelPop.textContent = sliderPop.value;
    labelMutRate.textContent = sliderMutRate.value;
    labelMutSigma.textContent = sliderMutSigma.value;
    labelCross.textContent = sliderCross.value;
    labelElite.textContent = sliderElite.value;
    labelSpeed.textContent = sliderSpeed.value + ' ms';
}
btnPlay.addEventListener('click', play);
btnPause.addEventListener('click', pause);
btnStep.addEventListener('click', function () { pause(); doStep(); });
btnReset.addEventListener('click', reset);
selFn.addEventListener('change', reset);
selSel.addEventListener('change', function () { });
[sliderPop, sliderMutRate, sliderMutSigma, sliderCross, sliderElite, sliderSpeed].forEach(function (s) {
    s.addEventListener('input', updateLabels);
});
sliderPop.addEventListener('change', reset);
window.addEventListener('resize', function () {
    pause();
    resizeCanvases();
});
resizeCanvases();
reset();

},{}]},{},[1]);
