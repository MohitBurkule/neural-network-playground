(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
function zeros(rows, cols) {
    return Array.from({ length: rows }, function () { return new Array(cols).fill(0); });
}
function randn() {
    var u = 1 - Math.random();
    var v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function makeLayer(inDim, outDim) {
    var scale = Math.sqrt(2 / inDim);
    var W = Array.from({ length: outDim }, function () {
        return Array.from({ length: inDim }, function () { return randn() * scale; });
    });
    var b = new Array(outDim).fill(0);
    return { W: W, b: b, dW: zeros(outDim, inDim), db: new Array(outDim).fill(0) };
}
function linearForward(layer, x) {
    return layer.W.map(function (row, i) { return row.reduce(function (s, w, j) { return s + w * x[j]; }, 0) + layer.b[i]; });
}
function relu(x) {
    return x.map(function (v) { return Math.max(0, v); });
}
function reluGrad(x) {
    return x.map(function (v) { return (v > 0 ? 1 : 0); });
}
function sigmoid(x) {
    return 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, x))));
}
function makeMLP(inDim, hiddenSize, outDim, numHidden) {
    var layers = [];
    var d = inDim;
    for (var i = 0; i < numHidden; i++) {
        layers.push(makeLayer(d, hiddenSize));
        d = hiddenSize;
    }
    layers.push(makeLayer(d, outDim));
    return { layers: layers, hiddenSize: hiddenSize, inDim: inDim, outDim: outDim };
}
function resetGrads(mlp) {
    for (var _i = 0, _a = mlp.layers; _i < _a.length; _i++) {
        var l = _a[_i];
        l.dW = zeros(l.W.length, l.W[0].length);
        l.db = new Array(l.b.length).fill(0);
    }
}
function mlpForward(mlp, input) {
    var preacts = [];
    var acts = [input];
    var x = input;
    for (var i = 0; i < mlp.layers.length; i++) {
        var pre = linearForward(mlp.layers[i], x);
        preacts.push(pre);
        if (i < mlp.layers.length - 1) {
            x = relu(pre);
        }
        else {
            x = pre;
        }
        acts.push(x);
    }
    return { preacts: preacts, acts: acts };
}
function mlpBackward(mlp, fwd, dLoss_dOut) {
    var delta = dLoss_dOut;
    var _loop_1 = function (i) {
        var layer = mlp.layers[i];
        var xIn = fwd.acts[i];
        for (var o = 0; o < layer.W.length; o++) {
            for (var j = 0; j < layer.W[0].length; j++) {
                layer.dW[o][j] += delta[o] * xIn[j];
            }
            layer.db[o] += delta[o];
        }
        var dX = new Array(xIn.length).fill(0);
        for (var j = 0; j < xIn.length; j++) {
            for (var o = 0; o < layer.W.length; o++) {
                dX[j] += layer.W[o][j] * delta[o];
            }
        }
        if (i > 0) {
            var preact = fwd.preacts[i - 1];
            var rg_1 = reluGrad(preact);
            delta = dX.map(function (v, j) { return v * rg_1[j]; });
        }
        else {
            delta = dX;
        }
    };
    for (var i = mlp.layers.length - 1; i >= 0; i--) {
        _loop_1(i);
    }
    return delta;
}
function sgdStep(mlp, lr) {
    for (var _i = 0, _a = mlp.layers; _i < _a.length; _i++) {
        var layer = _a[_i];
        for (var o = 0; o < layer.W.length; o++) {
            for (var j = 0; j < layer.W[0].length; j++) {
                layer.W[o][j] -= lr * layer.dW[o][j];
            }
            layer.b[o] -= lr * layer.db[o];
        }
    }
}
function sampleReal(dist) {
    switch (dist) {
        case 'ring': {
            var angle = Math.random() * 2 * Math.PI;
            var r = 0.7 + randn() * 0.05;
            return [r * Math.cos(angle), r * Math.sin(angle)];
        }
        case 'two-moons': {
            var top_1 = Math.random() > 0.5;
            var t = Math.random() * Math.PI;
            var x = (top_1 ? 1 : 0) + Math.cos(t) * 0.5 + randn() * 0.05;
            var y = (top_1 ? 0.3 : -0.3) + Math.sin(t) * 0.5 * (top_1 ? 1 : -1) + randn() * 0.05;
            return [x - 0.5, y];
        }
        case '8-gaussians': {
            var centers = [
                [0.7, 0], [-0.7, 0], [0, 0.7], [0, -0.7],
                [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]
            ];
            var c = centers[Math.floor(Math.random() * 8)];
            return [c[0] + randn() * 0.05, c[1] + randn() * 0.05];
        }
        case 'spiral': {
            var branch = Math.random() > 0.5 ? 0 : 1;
            var t = Math.random() * 3 * Math.PI;
            var r = t / (3 * Math.PI) * 0.85;
            var angle = t + branch * Math.PI;
            return [r * Math.cos(angle) + randn() * 0.03, r * Math.sin(angle) + randn() * 0.03];
        }
    }
}
function makeGAN(cfg) {
    var G = makeMLP(cfg.latentDim, cfg.hiddenSize, 2, 2);
    var D = makeMLP(2, cfg.hiddenSize, 1, 2);
    return { G: G, D: D, config: cfg, step: 0, lossD: Math.log(2), lossG: Math.log(2), lossDHistory: [], lossGHistory: [] };
}
function sampleLatent(latentDim) {
    return Array.from({ length: latentDim }, function () { return randn(); });
}
function generate(G, z) {
    var fwd = mlpForward(G, z);
    var out = fwd.acts[fwd.acts.length - 1];
    return [out[0], out[1]];
}
function discriminate(D, pt) {
    var fwd = mlpForward(D, [pt[0], pt[1]]);
    var logit = fwd.acts[fwd.acts.length - 1][0];
    return sigmoid(logit);
}
function bceLossAndGrad(logit, label) {
    var p = sigmoid(logit);
    var loss = -(label * Math.log(p + 1e-8) + (1 - label) * Math.log(1 - p + 1e-8));
    var grad = p - label;
    return [loss, grad];
}
function ganTrainStep(state) {
    var G = state.G, D = state.D, config = state.config;
    var latentDim = config.latentDim, lr = config.lr, batchSize = config.batchSize, dist = config.dist, dStepsPerG = config.dStepsPerG;
    var totalLossD = 0;
    for (var ds = 0; ds < dStepsPerG; ds++) {
        resetGrads(D);
        var ld = 0;
        for (var i = 0; i < batchSize; i++) {
            var real = sampleReal(dist);
            var fwdReal = mlpForward(D, [real[0], real[1]]);
            var logitReal = fwdReal.acts[fwdReal.acts.length - 1][0];
            var _a = bceLossAndGrad(logitReal, 1), lossReal = _a[0], gradReal = _a[1];
            mlpBackward(D, fwdReal, [gradReal / batchSize]);
            ld += lossReal;
            var z = sampleLatent(latentDim);
            var fwdG = mlpForward(G, z);
            var fake = [fwdG.acts[fwdG.acts.length - 1][0], fwdG.acts[fwdG.acts.length - 1][1]];
            var fwdFake = mlpForward(D, [fake[0], fake[1]]);
            var logitFake = fwdFake.acts[fwdFake.acts.length - 1][0];
            var _b = bceLossAndGrad(logitFake, 0), lossFake = _b[0], gradFake = _b[1];
            mlpBackward(D, fwdFake, [gradFake / batchSize]);
            ld += lossFake;
        }
        sgdStep(D, lr);
        totalLossD = ld / (2 * batchSize);
    }
    resetGrads(G);
    var lg = 0;
    for (var i = 0; i < batchSize; i++) {
        var z = sampleLatent(latentDim);
        var fwdG = mlpForward(G, z);
        var fake = [fwdG.acts[fwdG.acts.length - 1][0], fwdG.acts[fwdG.acts.length - 1][1]];
        var fwdD = mlpForward(D, [fake[0], fake[1]]);
        var logitD = fwdD.acts[fwdD.acts.length - 1][0];
        var _c = bceLossAndGrad(logitD, 1), lossG = _c[0], gradD_logit = _c[1];
        lg += lossG;
        var dFake = mlpBackward(D, fwdD, [gradD_logit / batchSize]);
        mlpBackward(G, fwdG, [dFake[0] / batchSize, dFake[1] / batchSize]);
    }
    sgdStep(G, lr);
    state.step++;
    state.lossD = totalLossD;
    state.lossG = lg / batchSize;
    state.lossDHistory.push(state.lossD);
    state.lossGHistory.push(state.lossG);
    if (state.lossDHistory.length > 300) {
        state.lossDHistory.shift();
        state.lossGHistory.shift();
    }
}
var PLOT_SIZE = 380;
var HEATMAP_RES = 40;
var VIEW_RANGE = 1.2;
function toCanvas(v) {
    return ((v + VIEW_RANGE) / (2 * VIEW_RANGE)) * PLOT_SIZE;
}
function fromCanvas(px) {
    return (px / PLOT_SIZE) * 2 * VIEW_RANGE - VIEW_RANGE;
}
function drawScene(ctx, state, nSamples) {
    var G = state.G, D = state.D, config = state.config;
    var step = PLOT_SIZE / HEATMAP_RES;
    for (var row = 0; row < HEATMAP_RES; row++) {
        for (var col = 0; col < HEATMAP_RES; col++) {
            var px = col * step + step / 2;
            var py = row * step + step / 2;
            var x = fromCanvas(px);
            var y = fromCanvas(py);
            var score = discriminate(D, [x, y]);
            var r = Math.round(score * 160);
            var b = Math.round((1 - score) * 160);
            ctx.fillStyle = "rgb(".concat(r, ",20,").concat(b, ")");
            ctx.fillRect(col * step, row * step, step, step);
        }
    }
    ctx.fillStyle = 'rgba(100,220,100,0.75)';
    for (var i = 0; i < nSamples; i++) {
        var pt = sampleReal(config.dist);
        var cx = toCanvas(pt[0]);
        var cy = toCanvas(pt[1]);
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
        ctx.fill();
    }
    ctx.fillStyle = 'rgba(255,180,30,0.75)';
    for (var i = 0; i < nSamples; i++) {
        var z = sampleLatent(config.latentDim);
        var fwd = mlpForward(G, z);
        var out = fwd.acts[fwd.acts.length - 1];
        var cx = toCanvas(out[0]);
        var cy = toCanvas(out[1]);
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
        ctx.fill();
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    var mid = toCanvas(0);
    ctx.beginPath();
    ctx.moveTo(mid, 0);
    ctx.lineTo(mid, PLOT_SIZE);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(PLOT_SIZE, mid);
    ctx.stroke();
}
function drawLossCurves(ctx, width, height, lossDHistory, lossGHistory) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);
    if (lossDHistory.length < 2)
        return;
    var allVals = __spreadArray(__spreadArray([], lossDHistory, true), lossGHistory, true);
    var maxV = Math.min(Math.max.apply(Math, allVals) * 1.1, 4);
    var minV = 0;
    var pad = { t: 10, r: 10, b: 25, l: 40 };
    var pw = width - pad.l - pad.r;
    var ph = height - pad.t - pad.b;
    function mapX(i) {
        return pad.l + (i / (lossDHistory.length - 1)) * pw;
    }
    function mapY(v) {
        return pad.t + (1 - (v - minV) / (maxV - minV)) * ph;
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    for (var v = 0; v <= 2; v += 0.5) {
        var y = mapY(v);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + pw, y);
        ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        ctx.font = '9px monospace';
        ctx.fillText(v.toFixed(1), 2, y + 3);
    }
    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    lossDHistory.forEach(function (v, i) {
        var x = mapX(i);
        var y = mapY(v);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.strokeStyle = '#ffb74d';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    lossGHistory.forEach(function (v, i) {
        var x = mapX(i);
        var y = mapY(v);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.font = '10px monospace';
    ctx.fillStyle = '#4fc3f7';
    ctx.fillText('D loss', pad.l + 4, pad.t + 12);
    ctx.fillStyle = '#ffb74d';
    ctx.fillText('G loss', pad.l + 55, pad.t + 12);
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t + ph);
    ctx.lineTo(pad.l + pw, pad.t + ph);
    ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '9px monospace';
    ctx.fillText('steps (last 300)', pad.l, height - 4);
}
function getConfig(controls) {
    return {
        latentDim: parseInt(controls.latentInput.value, 10),
        hiddenSize: parseInt(controls.hiddenInput.value, 10),
        dStepsPerG: parseInt(controls.dStepsInput.value, 10),
        lr: parseFloat(controls.lrInput.value),
        batchSize: 32,
        dist: controls.distSelect.value
    };
}
function init() {
    var mainCanvas = document.getElementById('mainCanvas');
    var lossCanvas = document.getElementById('lossCanvas');
    var ctx = mainCanvas.getContext('2d');
    var lossCtx = lossCanvas.getContext('2d');
    mainCanvas.width = PLOT_SIZE;
    mainCanvas.height = PLOT_SIZE;
    lossCanvas.width = lossCanvas.offsetWidth || 380;
    lossCanvas.height = lossCanvas.offsetHeight || 120;
    var controls = {
        playPauseBtn: document.getElementById('playPause'),
        stepBtn: document.getElementById('stepBtn'),
        resetBtn: document.getElementById('resetBtn'),
        distSelect: document.getElementById('distSelect'),
        lrInput: document.getElementById('lrInput'),
        lrVal: document.getElementById('lrVal'),
        latentInput: document.getElementById('latentInput'),
        latentVal: document.getElementById('latentVal'),
        hiddenInput: document.getElementById('hiddenInput'),
        hiddenVal: document.getElementById('hiddenVal'),
        dStepsInput: document.getElementById('dStepsInput'),
        dStepsVal: document.getElementById('dStepsVal'),
        stepCount: document.getElementById('stepCount'),
        lossDEl: document.getElementById('lossDVal'),
        lossGEl: document.getElementById('lossGVal')
    };
    var state = makeGAN(getConfig(controls));
    var running = false;
    var animId = 0;
    var stepsPerFrame = 5;
    function render() {
        drawScene(ctx, state, 150);
        drawLossCurves(lossCtx, lossCanvas.width, lossCanvas.height, state.lossDHistory, state.lossGHistory);
        controls.stepCount.textContent = String(state.step);
        controls.lossDEl.textContent = state.lossD.toFixed(4);
        controls.lossGEl.textContent = state.lossG.toFixed(4);
    }
    function loop() {
        for (var i = 0; i < stepsPerFrame; i++) {
            ganTrainStep(state);
        }
        render();
        if (running)
            animId = requestAnimationFrame(loop);
    }
    controls.playPauseBtn.addEventListener('click', function () {
        running = !running;
        controls.playPauseBtn.textContent = running ? 'Pause' : 'Play';
        if (running)
            loop();
        else
            cancelAnimationFrame(animId);
    });
    controls.stepBtn.addEventListener('click', function () {
        if (!running) {
            ganTrainStep(state);
            render();
        }
    });
    controls.resetBtn.addEventListener('click', function () {
        running = false;
        cancelAnimationFrame(animId);
        controls.playPauseBtn.textContent = 'Play';
        state = makeGAN(getConfig(controls));
        render();
    });
    function syncLabel(input, span) {
        span.textContent = input.value;
        input.addEventListener('input', function () { span.textContent = input.value; });
    }
    syncLabel(controls.lrInput, controls.lrVal);
    syncLabel(controls.latentInput, controls.latentVal);
    syncLabel(controls.hiddenInput, controls.hiddenVal);
    syncLabel(controls.dStepsInput, controls.dStepsVal);
    render();
}
window.addEventListener('DOMContentLoaded', init);

},{}]},{},[1]);
