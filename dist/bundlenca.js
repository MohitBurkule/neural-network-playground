(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var CELL_SIZE = 8;
var GRID_W = 28;
var GRID_H = 28;
var N_CHANNELS = 12;
var PERCEPTION_CHANNELS = N_CHANNELS * 3;
var HIDDEN = 64;
function zeros(n) { return new Float32Array(n); }
function randn() { var u = 0, v = 0; while (!u)
    u = Math.random(); while (!v)
    v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
function initWeights(nChannels, hidden, perceptionC) {
    var scale = Math.sqrt(2 / perceptionC);
    var W1 = new Float32Array(hidden * perceptionC).map(function () { return randn() * scale; });
    var b1 = zeros(hidden);
    var W2 = zeros(nChannels * hidden);
    var b2 = zeros(nChannels);
    return { W1: W1, b1: b1, W2: W2, b2: b2 };
}
function initAdam(weights) {
    var total = weights.W1.length + weights.b1.length + weights.W2.length + weights.b2.length;
    return { m: zeros(total), v: zeros(total), t: 0 };
}
function flatWeights(w) {
    var out = new Float32Array(w.W1.length + w.b1.length + w.W2.length + w.b2.length);
    out.set(w.W1, 0);
    out.set(w.b1, w.W1.length);
    out.set(w.W2, w.W1.length + w.b1.length);
    out.set(w.b2, w.W1.length + w.b1.length + w.W2.length);
    return out;
}
function unflatWeights(flat, nChannels, hidden, perceptionC) {
    var off = 0;
    var W1 = flat.slice(off, off + hidden * perceptionC);
    off += W1.length;
    var b1 = flat.slice(off, off + hidden);
    off += hidden;
    var W2 = flat.slice(off, off + nChannels * hidden);
    off += W2.length;
    var b2 = flat.slice(off, off + nChannels);
    return { W1: W1, b1: b1, W2: W2, b2: b2 };
}
function perceive(grid, gw, gh, nCh) {
    var out = new Float32Array(gh * gw * nCh * 3);
    var _loop_1 = function (y) {
        var _loop_2 = function (x) {
            var ci = (y * gw + x) * nCh * 3;
            var _loop_3 = function (c) {
                var gat = function (yy, xx) {
                    var ny = Math.min(gh - 1, Math.max(0, y + yy));
                    var nx = Math.min(gw - 1, Math.max(0, x + xx));
                    return grid[(ny * gw + nx) * nCh + c];
                };
                out[ci + c] = gat(0, 0);
                out[ci + nCh + c] = (-gat(-1, -1) + gat(-1, 1)
                    - 2 * gat(0, -1) + 2 * gat(0, 1)
                    - gat(1, -1) + gat(1, 1)) / 8;
                out[ci + nCh * 2 + c] = (-gat(-1, -1) - 2 * gat(-1, 0) - gat(-1, 1)
                    + gat(1, -1) + 2 * gat(1, 0) + gat(1, 1)) / 8;
            };
            for (var c = 0; c < nCh; c++) {
                _loop_3(c);
            }
        };
        for (var x = 0; x < gw; x++) {
            _loop_2(x);
        }
    };
    for (var y = 0; y < gh; y++) {
        _loop_1(y);
    }
    return out;
}
function mlpForward(perc, gw, gh, nCh, hidden, W1, b1, W2, b2) {
    var N = gw * gh;
    var pCh = nCh * 3;
    var delta = new Float32Array(N * nCh);
    var h_pre = new Float32Array(N * hidden);
    var h_act = new Float32Array(N * hidden);
    for (var i = 0; i < N; i++) {
        for (var h = 0; h < hidden; h++) {
            var s = b1[h];
            for (var p = 0; p < pCh; p++)
                s += W1[h * pCh + p] * perc[i * pCh + p];
            h_pre[i * hidden + h] = s;
            h_act[i * hidden + h] = s > 0 ? s : 0;
        }
        for (var c = 0; c < nCh; c++) {
            var s = b2[c];
            for (var h = 0; h < hidden; h++)
                s += W2[c * hidden + h] * h_act[i * hidden + h];
            delta[i * nCh + c] = s;
        }
    }
    return { delta: delta, h_pre: h_pre, h_act: h_act };
}
function ncaStep(grid, gw, gh, nCh, hidden, W1, b1, W2, b2, updateRate) {
    var perc = perceive(grid, gw, gh, nCh);
    var delta = mlpForward(perc, gw, gh, nCh, hidden, W1, b1, W2, b2).delta;
    var N = gw * gh;
    var newGrid = new Float32Array(grid);
    for (var i = 0; i < N; i++) {
        if (Math.random() > updateRate)
            continue;
        var y = Math.floor(i / gw), x = i % gw;
        var selfAlpha = grid[i * nCh + 3];
        var neighborAlive = false;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                var ny = Math.min(gh - 1, Math.max(0, y + dy));
                var nx = Math.min(gw - 1, Math.max(0, x + dx));
                if (grid[(ny * gw + nx) * nCh + 3] > 0.1) {
                    neighborAlive = true;
                    break;
                }
            }
            if (neighborAlive)
                break;
        }
        if (!neighborAlive && selfAlpha < 0.1)
            continue;
        for (var c = 0; c < nCh; c++) {
            newGrid[i * nCh + c] = grid[i * nCh + c] + delta[i * nCh + c];
        }
        newGrid[i * nCh + 3] = clamp01(newGrid[i * nCh + 3]);
    }
    return newGrid;
}
function computeGradients(grid, target, gw, gh, nCh, hidden, W1, b1, W2, b2) {
    var N = gw * gh;
    var pCh = nCh * 3;
    var perc = perceive(grid, gw, gh, nCh);
    var _a = mlpForward(perc, gw, gh, nCh, hidden, W1, b1, W2, b2), delta = _a.delta, h_pre = _a.h_pre, h_act = _a.h_act;
    var loss = 0;
    var dLoss_dDelta = new Float32Array(N * nCh);
    for (var i = 0; i < N; i++) {
        for (var c = 0; c < 4; c++) {
            var pred = grid[i * nCh + c] + delta[i * nCh + c];
            var diff = pred - target[i * nCh + c];
            loss += diff * diff;
            dLoss_dDelta[i * nCh + c] = 2 * diff / (N * 4);
        }
    }
    loss /= (N * 4);
    var gW2 = zeros(nCh * hidden);
    var gb2 = zeros(nCh);
    var dLoss_dH = zeros(N * hidden);
    for (var i = 0; i < N; i++) {
        for (var c = 0; c < nCh; c++) {
            var g = dLoss_dDelta[i * nCh + c];
            gb2[c] += g;
            for (var h = 0; h < hidden; h++) {
                gW2[c * hidden + h] += g * h_act[i * hidden + h];
                dLoss_dH[i * hidden + h] += g * W2[c * hidden + h];
            }
        }
    }
    var gW1 = zeros(hidden * pCh);
    var gb1 = zeros(hidden);
    for (var i = 0; i < N; i++) {
        for (var h = 0; h < hidden; h++) {
            var d = h_pre[i * hidden + h] > 0 ? dLoss_dH[i * hidden + h] : 0;
            gb1[h] += d;
            for (var p = 0; p < pCh; p++) {
                gW1[h * pCh + p] += d * perc[i * pCh + p];
            }
        }
    }
    return { loss: loss, gW1: gW1, gb1: gb1, gW2: gW2, gb2: gb2 };
}
function adamUpdate(weights, grads, state, lr) {
    var flat = flatWeights(weights);
    var gflat = new Float32Array(flat.length);
    var off = 0;
    gflat.set(grads.gW1, off);
    off += grads.gW1.length;
    gflat.set(grads.gb1, off);
    off += grads.gb1.length;
    gflat.set(grads.gW2, off);
    off += grads.gW2.length;
    gflat.set(grads.gb2, off);
    var beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    state.t += 1;
    var bc1 = 1 - Math.pow(beta1, state.t);
    var bc2 = 1 - Math.pow(beta2, state.t);
    for (var i = 0; i < flat.length; i++) {
        state.m[i] = beta1 * state.m[i] + (1 - beta1) * gflat[i];
        state.v[i] = beta2 * state.v[i] + (1 - beta2) * gflat[i] * gflat[i];
        flat[i] -= lr * (state.m[i] / bc1) / (Math.sqrt(state.v[i] / bc2) + eps);
    }
    return unflatWeights(flat, N_CHANNELS, HIDDEN, PERCEPTION_CHANNELS);
}
function makeTarget(name, gw, gh) {
    var t = new Float32Array(gw * gh * 4);
    var cx = gw / 2, cy = gh / 2;
    for (var y = 0; y < gh; y++) {
        for (var x = 0; x < gw; x++) {
            var i = (y * gw + x) * 4;
            var dx = x - cx, dy = y - cy;
            var r = Math.sqrt(dx * dx + dy * dy);
            var R = 0, G = 0, B = 0, A = 0;
            if (name === 'circle') {
                A = r < gw * 0.4 ? 1 : 0;
                R = 0.2;
                G = 0.6;
                B = 1.0;
            }
            else if (name === 'ring') {
                A = (r > gw * 0.25 && r < gw * 0.42) ? 1 : 0;
                R = 1.0;
                G = 0.4;
                B = 0.1;
            }
            else if (name === 'smile') {
                if (r < gw * 0.42) {
                    A = 1;
                    R = 1;
                    G = 0.85;
                    B = 0.2;
                }
                var le = Math.sqrt((x - cx + gw * 0.15) * (x - cx + gw * 0.15) + (y - cy + gh * 0.12) * (y - cy + gh * 0.12));
                var re = Math.sqrt((x - cx - gw * 0.15) * (x - cx - gw * 0.15) + (y - cy + gh * 0.12) * (y - cy + gh * 0.12));
                if (le < gw * 0.08 || re < gw * 0.08) {
                    A = 1;
                    R = 0.1;
                    G = 0.1;
                    B = 0.1;
                }
                if (r < gw * 0.42 && dy > gh * 0.05) {
                    var smileR = Math.sqrt(dx * dx + (dy - gh * 0.1) * (dy - gh * 0.1));
                    if (smileR > gw * 0.2 && smileR < gw * 0.28 && dy > 0) {
                        A = 1;
                        R = 0.1;
                        G = 0.1;
                        B = 0.1;
                    }
                }
            }
            else if (name === 'cross') {
                A = (Math.abs(dx) < gw * 0.1 || Math.abs(dy) < gh * 0.1) && r < gw * 0.45 ? 1 : 0;
                R = 0.9;
                G = 0.2;
                B = 0.2;
            }
            else if (name === 'star') {
                var angle = Math.atan2(dy, dx);
                var points = 5;
                var outerR = gw * 0.42, innerR = gw * 0.18;
                var minR = 1e9;
                for (var p = 0; p < points; p++) {
                    var a1 = (p / points) * Math.PI * 2 - Math.PI / 2;
                    var a2 = ((p + 0.5) / points) * Math.PI * 2 - Math.PI / 2;
                    var normAngle = ((angle - a1 + Math.PI * 4) % (Math.PI * 2 / points));
                    var half = Math.PI / points;
                    var frac = normAngle < half ? normAngle / half : (2 * half - normAngle) / half;
                    var threshold = innerR + (outerR - innerR) * frac;
                    minR = Math.min(minR, threshold);
                }
                var starA = Math.atan2(dy, dx) + Math.PI / 2;
                var starR2 = 0.5 * (outerR + innerR) + 0.5 * (outerR - innerR) * Math.cos(points * starA);
                A = r < starR2 ? 1 : 0;
                R = 1.0;
                G = 0.85;
                B = 0.0;
            }
            else {
                A = ((Math.floor(x / 4) + Math.floor(y / 4)) % 2 === 0 && r < gw * 0.42) ? 1 : 0;
                R = 0.4;
                G = 0.9;
                B = 0.5;
            }
            t[i] = R;
            t[i + 1] = G;
            t[i + 2] = B;
            t[i + 3] = A;
        }
    }
    return t;
}
function makeSeedGrid(gw, gh, nCh) {
    var g = new Float32Array(gw * gh * nCh);
    var cy = Math.floor(gh / 2), cx = Math.floor(gw / 2);
    var i = (cy * gw + cx) * nCh;
    g[i + 3] = 1;
    return g;
}
function gridToImageData(grid, gw, gh, nCh, imgData) {
    for (var i = 0; i < gw * gh; i++) {
        var alpha = clamp01(grid[i * nCh + 3]);
        imgData.data[i * 4] = clamp01(grid[i * nCh + 0]) * alpha * 255;
        imgData.data[i * 4 + 1] = clamp01(grid[i * nCh + 1]) * alpha * 255;
        imgData.data[i * 4 + 2] = clamp01(grid[i * nCh + 2]) * alpha * 255;
        imgData.data[i * 4 + 3] = alpha * 255;
    }
}
function targetToImageData(target, gw, gh, imgData) {
    for (var i = 0; i < gw * gh; i++) {
        var alpha = target[i * 4 + 3];
        imgData.data[i * 4] = target[i * 4 + 0] * alpha * 255;
        imgData.data[i * 4 + 1] = target[i * 4 + 1] * alpha * 255;
        imgData.data[i * 4 + 2] = target[i * 4 + 2] * alpha * 255;
        imgData.data[i * 4 + 3] = alpha * 255;
    }
}
document.addEventListener('DOMContentLoaded', function () {
    var nCh = N_CHANNELS;
    var hidden = HIDDEN;
    var percC = nCh * 3;
    var weights = initWeights(nCh, hidden, percC);
    var adamState = initAdam(weights);
    var grid = makeSeedGrid(GRID_W, GRID_H, nCh);
    var targetName = 'smile';
    var target = makeTarget(targetName, GRID_W, GRID_H);
    var lr = 2e-3;
    var rolloutLen = 8;
    var updateRate = 0.5;
    var running = false;
    var trainEvery = 1;
    var frameCount = 0;
    var speed = 1;
    var losses = [];
    var MAX_LOSS_HISTORY = 200;
    var gridCanvas = document.getElementById('grid-canvas');
    var targetCanvas = document.getElementById('target-canvas');
    var lossCanvas = document.getElementById('loss-canvas');
    var lossLabel = document.getElementById('loss-label');
    var stepCount = document.getElementById('step-count');
    gridCanvas.width = GRID_W * CELL_SIZE;
    gridCanvas.height = GRID_H * CELL_SIZE;
    targetCanvas.width = GRID_W * CELL_SIZE;
    targetCanvas.height = GRID_H * CELL_SIZE;
    var gridCtx = gridCanvas.getContext('2d');
    var targetCtx = targetCanvas.getContext('2d');
    var lossCtx = lossCanvas.getContext('2d');
    var lrInput = document.getElementById('lr');
    var lrVal = document.getElementById('lr-val');
    var rolloutInput = document.getElementById('rollout');
    var rolloutVal = document.getElementById('rollout-val');
    var speedInput = document.getElementById('speed');
    var speedVal = document.getElementById('speed-val');
    var patternSel = document.getElementById('pattern');
    var playBtn = document.getElementById('play-btn');
    var stepBtn = document.getElementById('step-btn');
    var resetBtn = document.getElementById('reset-btn');
    var seedBtn = document.getElementById('seed-btn');
    lrInput.addEventListener('input', function () { lr = Math.pow(10, parseFloat(lrInput.value)); lrVal.textContent = lr.toExponential(1); });
    rolloutInput.addEventListener('input', function () { rolloutLen = parseInt(rolloutInput.value); rolloutVal.textContent = String(rolloutLen); });
    speedInput.addEventListener('input', function () { speed = parseInt(speedInput.value); speedVal.textContent = String(speed); });
    patternSel.addEventListener('change', function () { targetName = patternSel.value; target = makeTarget(targetName, GRID_W, GRID_H); drawTarget(); });
    playBtn.addEventListener('click', function () { running = !running; playBtn.textContent = running ? 'Pause' : 'Play'; });
    stepBtn.addEventListener('click', function () { doStep(); render(); });
    resetBtn.addEventListener('click', function () {
        weights = initWeights(nCh, hidden, percC);
        adamState = initAdam(weights);
        grid = makeSeedGrid(GRID_W, GRID_H, nCh);
        losses = [];
        frameCount = 0;
        render();
    });
    seedBtn.addEventListener('click', function () {
        grid = makeSeedGrid(GRID_W, GRID_H, nCh);
        render();
    });
    gridCanvas.addEventListener('click', function (e) {
        var rect = gridCanvas.getBoundingClientRect();
        var mx = Math.floor((e.clientX - rect.left) / CELL_SIZE);
        var my = Math.floor((e.clientY - rect.top) / CELL_SIZE);
        var radius = 3;
        for (var dy = -radius; dy <= radius; dy++) {
            for (var dx = -radius; dx <= radius; dx++) {
                if (dx * dx + dy * dy > radius * radius)
                    continue;
                var nx = mx + dx, ny = my + dy;
                if (nx < 0 || ny < 0 || nx >= GRID_W || ny >= GRID_H)
                    continue;
                var base = (ny * GRID_W + nx) * nCh;
                for (var c = 0; c < nCh; c++)
                    grid[base + c] = 0;
            }
        }
        render();
    });
    var totalSteps = 0;
    function doStep() {
        grid = ncaStep(grid, GRID_W, GRID_H, nCh, hidden, weights.W1, weights.b1, weights.W2, weights.b2, updateRate);
        totalSteps++;
        stepCount.textContent = String(totalSteps);
    }
    function doTrain() {
        var g = new Float32Array(grid);
        for (var s = 0; s < rolloutLen; s++) {
            g = ncaStep(g, GRID_W, GRID_H, nCh, hidden, weights.W1, weights.b1, weights.W2, weights.b2, updateRate);
        }
        var _a = computeGradients(g, target, GRID_W, GRID_H, nCh, hidden, weights.W1, weights.b1, weights.W2, weights.b2), loss = _a.loss, gW1 = _a.gW1, gb1 = _a.gb1, gW2 = _a.gW2, gb2 = _a.gb2;
        weights = adamUpdate(weights, { gW1: gW1, gb1: gb1, gW2: gW2, gb2: gb2 }, adamState, lr);
        losses.push(loss);
        if (losses.length > MAX_LOSS_HISTORY)
            losses.shift();
        lossLabel.textContent = loss.toFixed(5);
    }
    function drawTarget() {
        var imgData = targetCtx.createImageData(GRID_W, GRID_H);
        targetToImageData(target, GRID_W, GRID_H, imgData);
        var offscreen = document.createElement('canvas');
        offscreen.width = GRID_W;
        offscreen.height = GRID_H;
        offscreen.getContext('2d').putImageData(imgData, 0, 0);
        targetCtx.imageSmoothingEnabled = false;
        targetCtx.drawImage(offscreen, 0, 0, GRID_W * CELL_SIZE, GRID_H * CELL_SIZE);
    }
    function drawGrid() {
        var imgData = gridCtx.createImageData(GRID_W, GRID_H);
        gridToImageData(grid, GRID_W, GRID_H, nCh, imgData);
        var offscreen = document.createElement('canvas');
        offscreen.width = GRID_W;
        offscreen.height = GRID_H;
        offscreen.getContext('2d').putImageData(imgData, 0, 0);
        gridCtx.imageSmoothingEnabled = false;
        gridCtx.drawImage(offscreen, 0, 0, GRID_W * CELL_SIZE, GRID_H * CELL_SIZE);
    }
    function drawLoss() {
        var W = lossCanvas.width, H = lossCanvas.height;
        lossCtx.fillStyle = '#1a1a2e';
        lossCtx.fillRect(0, 0, W, H);
        if (losses.length < 2)
            return;
        var maxL = Math.max.apply(Math, losses);
        var minL = Math.min.apply(Math, losses);
        var range = maxL - minL || 1e-8;
        lossCtx.strokeStyle = '#4fc3f7';
        lossCtx.lineWidth = 1.5;
        lossCtx.beginPath();
        for (var i = 0; i < losses.length; i++) {
            var px = (i / (MAX_LOSS_HISTORY - 1)) * W;
            var py = H - ((losses[i] - minL) / range) * (H - 4) - 2;
            if (i === 0)
                lossCtx.moveTo(px, py);
            else
                lossCtx.lineTo(px, py);
        }
        lossCtx.stroke();
        lossCtx.fillStyle = '#aaa';
        lossCtx.font = '10px monospace';
        lossCtx.fillText(maxL.toFixed(4), 2, 12);
        lossCtx.fillText(minL.toFixed(4), 2, H - 4);
    }
    function render() {
        drawGrid();
        drawLoss();
    }
    function loop() {
        requestAnimationFrame(loop);
        if (!running)
            return;
        for (var s = 0; s < speed; s++) {
            doStep();
        }
        frameCount++;
        doTrain();
        render();
    }
    drawTarget();
    drawGrid();
    drawLoss();
    requestAnimationFrame(loop);
});

},{}]},{},[1]);
