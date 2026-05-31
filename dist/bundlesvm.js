(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function kernelLinear(a, b) {
    return a[0] * b[0] + a[1] * b[1];
}
function kernelPoly(a, b, degree) {
    return Math.pow(1 + a[0] * b[0] + a[1] * b[1], degree);
}
function kernelRBF(a, b, gamma) {
    var dx = a[0] - b[0];
    var dy = a[1] - b[1];
    return Math.exp(-gamma * (dx * dx + dy * dy));
}
var SVM = (function () {
    function SVM(C, kernelType, gamma, degree) {
        this.alphas = [];
        this.bias = 0;
        this.xs = [];
        this.ys = [];
        this.C = 1;
        this.kernelType = 'rbf';
        this.gamma = 0.5;
        this.degree = 3;
        this.cache = null;
        this.n = 0;
        this.C = C;
        this.kernelType = kernelType;
        this.gamma = gamma;
        this.degree = degree;
    }
    SVM.prototype.K = function (i, j) {
        if (this.cache) {
            var idx = i * this.n + j;
            if (this.cache[idx] !== -999)
                return this.cache[idx];
            var v = this.rawK(this.xs[i], this.xs[j]);
            this.cache[idx] = v;
            return v;
        }
        return this.rawK(this.xs[i], this.xs[j]);
    };
    SVM.prototype.rawK = function (a, b) {
        if (this.kernelType === 'linear')
            return kernelLinear(a, b);
        if (this.kernelType === 'poly')
            return kernelPoly(a, b, this.degree);
        return kernelRBF(a, b, this.gamma);
    };
    SVM.prototype.decisionRaw = function (i) {
        var s = 0;
        for (var j = 0; j < this.n; j++) {
            if (this.alphas[j] > 1e-8) {
                s += this.alphas[j] * this.ys[j] * this.K(j, i);
            }
        }
        return s + this.bias;
    };
    SVM.prototype.train = function (points, maxIter) {
        if (maxIter === void 0) { maxIter = 200; }
        this.n = points.length;
        this.xs = points.map(function (p) { return [p.x, p.y]; });
        this.ys = points.map(function (p) { return p.label; });
        this.alphas = new Array(this.n).fill(0);
        this.bias = 0;
        if (this.n <= 500) {
            this.cache = new Float64Array(this.n * this.n).fill(-999);
        }
        else {
            this.cache = null;
        }
        var C = this.C;
        var tol = 1e-3;
        for (var iter = 0; iter < maxIter; iter++) {
            var numChanged = 0;
            for (var i = 0; i < this.n; i++) {
                var Ei = this.decisionRaw(i) - this.ys[i];
                var yi = this.ys[i];
                var ai = this.alphas[i];
                if ((yi * Ei < -tol && ai < C) || (yi * Ei > tol && ai > 0)) {
                    var j = Math.floor(Math.random() * (this.n - 1));
                    if (j >= i)
                        j++;
                    var Ej = this.decisionRaw(j) - this.ys[j];
                    var yj = this.ys[j];
                    var aj = this.alphas[j];
                    var L = void 0, H_1 = void 0;
                    if (yi !== yj) {
                        L = Math.max(0, aj - ai);
                        H_1 = Math.min(C, C + aj - ai);
                    }
                    else {
                        L = Math.max(0, ai + aj - C);
                        H_1 = Math.min(C, ai + aj);
                    }
                    if (L >= H_1)
                        continue;
                    var eta = 2 * this.K(i, j) - this.K(i, i) - this.K(j, j);
                    if (eta >= 0)
                        continue;
                    var ajNew = aj - yj * (Ei - Ej) / eta;
                    if (ajNew > H_1)
                        ajNew = H_1;
                    else if (ajNew < L)
                        ajNew = L;
                    if (Math.abs(ajNew - aj) < 1e-5)
                        continue;
                    var aiNew = ai + yi * yj * (aj - ajNew);
                    this.alphas[i] = aiNew;
                    this.alphas[j] = ajNew;
                    var b1 = this.bias - Ei - yi * (aiNew - ai) * this.K(i, i) - yj * (ajNew - aj) * this.K(i, j);
                    var b2 = this.bias - Ej - yi * (aiNew - ai) * this.K(i, j) - yj * (ajNew - aj) * this.K(j, j);
                    if (aiNew > 0 && aiNew < C)
                        this.bias = b1;
                    else if (ajNew > 0 && ajNew < C)
                        this.bias = b2;
                    else
                        this.bias = (b1 + b2) / 2;
                    numChanged++;
                }
            }
            if (numChanged === 0)
                break;
        }
    };
    SVM.prototype.predict = function (x, y) {
        var s = 0;
        var pt = [x, y];
        for (var i = 0; i < this.n; i++) {
            if (this.alphas[i] > 1e-8) {
                s += this.alphas[i] * this.ys[i] * this.rawK(this.xs[i], pt);
            }
        }
        return s + this.bias;
    };
    SVM.prototype.getSupportVectorIndices = function () {
        var svs = [];
        for (var i = 0; i < this.n; i++) {
            if (this.alphas[i] > 1e-8)
                svs.push(i);
        }
        return svs;
    };
    SVM.prototype.accuracy = function (points) {
        var correct = 0;
        for (var i = 0; i < points.length; i++) {
            var p = points[i];
            var pred = this.predict(p.x, p.y) >= 0 ? 1 : -1;
            if (pred === p.label)
                correct++;
        }
        return correct / points.length;
    };
    return SVM;
}());
function randn() {
    var u = Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u + 1e-10)) * Math.cos(2 * Math.PI * v);
}
function generateBlobs(n, noise) {
    var pts = [];
    var centers = [[-1.5, -1.5], [1.5, 1.5]];
    for (var i = 0; i < n; i++) {
        var label = i < n / 2 ? -1 : 1;
        var c = label === -1 ? centers[0] : centers[1];
        pts.push({ x: c[0] + randn() * (0.5 + noise), y: c[1] + randn() * (0.5 + noise), label: label });
    }
    return pts;
}
function generateMoons(n, noise) {
    var pts = [];
    for (var i = 0; i < n; i++) {
        var t = Math.PI * i / (n / 2);
        var label = i < n / 2 ? -1 : 1;
        if (i < n / 2) {
            pts.push({ x: Math.cos(t) * 2 + randn() * noise, y: Math.sin(t) * 2 + randn() * noise, label: label });
        }
        else {
            var t2 = Math.PI * (i - n / 2) / (n / 2);
            pts.push({ x: Math.cos(t2) * 2 + 1 + randn() * noise, y: -Math.sin(t2) * 2 + 0.5 + randn() * noise, label: label });
        }
    }
    return pts;
}
function generateCircles(n, noise) {
    var pts = [];
    for (var i = 0; i < n; i++) {
        var label = i < n / 2 ? -1 : 1;
        var r = label === -1 ? 0.8 : 2.0;
        var t = 2 * Math.PI * Math.random();
        pts.push({ x: Math.cos(t) * r + randn() * noise, y: Math.sin(t) * r + randn() * noise, label: label });
    }
    return pts;
}
function generateXOR(n, noise) {
    var pts = [];
    for (var i = 0; i < n; i++) {
        var x = (Math.random() * 2 - 1) * 2;
        var y = (Math.random() * 2 - 1) * 2;
        var label = x * y > 0 ? 1 : -1;
        pts.push({ x: x + randn() * noise, y: y + randn() * noise, label: label });
    }
    return pts;
}
function generateOverlapping(n, noise) {
    var pts = [];
    for (var i = 0; i < n; i++) {
        var label = i < n / 2 ? -1 : 1;
        var cx = label * 0.6;
        pts.push({ x: cx + randn() * (0.8 + noise), y: randn() * (0.8 + noise), label: label });
    }
    return pts;
}
function generateDataset(type, n, noise) {
    if (type === 'blobs')
        return generateBlobs(n, noise);
    if (type === 'moons')
        return generateMoons(n, noise);
    if (type === 'circles')
        return generateCircles(n, noise);
    if (type === 'xor')
        return generateXOR(n, noise);
    return generateOverlapping(n, noise);
}
var points = [];
var svm = null;
var trainedPoints = [];
var canvas = document.getElementById('svmCanvas');
var ctx = canvas.getContext('2d');
var btnTrain = document.getElementById('btnTrain');
var btnReset = document.getElementById('btnReset');
var selDataset = document.getElementById('selDataset');
var selKernel = document.getElementById('selKernel');
var inpC = document.getElementById('inpC');
var inpGamma = document.getElementById('inpGamma');
var inpDegree = document.getElementById('inpDegree');
var inpN = document.getElementById('inpN');
var inpNoise = document.getElementById('inpNoise');
var lblC = document.getElementById('lblC');
var lblGamma = document.getElementById('lblGamma');
var lblDegree = document.getElementById('lblDegree');
var lblN = document.getElementById('lblN');
var lblNoise = document.getElementById('lblNoise');
var statsEl = document.getElementById('stats');
var rowGamma = document.getElementById('rowGamma');
var rowDegree = document.getElementById('rowDegree');
var statusEl = document.getElementById('status');
var RANGE = 3.5;
var W = canvas.width;
var H = canvas.height;
function dataToCanvas(x, y) {
    var cx = (x + RANGE) / (2 * RANGE) * W;
    var cy = (1 - (y + RANGE) / (2 * RANGE)) * H;
    return [cx, cy];
}
function canvasToData(cx, cy) {
    var x = cx / W * 2 * RANGE - RANGE;
    var y = -(cy / H * 2 * RANGE - RANGE);
    return [x, y];
}
var GRID = 80;
function drawDecisionBoundary() {
    if (!svm)
        return;
    var imageData = ctx.createImageData(W, H);
    var data = imageData.data;
    var step = 1;
    for (var py = 0; py < H; py += step) {
        for (var px = 0; px < W; px += step) {
            var _a = canvasToData(px + 0.5, py + 0.5), x = _a[0], y = _a[1];
            var val = svm.predict(x, y);
            var r = 26, g = 26, b = 46;
            if (val > 0) {
                var t = Math.min(1, Math.abs(val) / 2);
                r = Math.round(26 + t * (100 - 26));
                g = Math.round(26 + t * (60 - 26));
                b = Math.round(46 + t * (160 - 46));
            }
            else {
                var t = Math.min(1, Math.abs(val) / 2);
                r = Math.round(26 + t * (160 - 26));
                g = Math.round(26 + t * (60 - 26));
                b = Math.round(46 + t * (80 - 46));
            }
            var idx = (py * W + px) * 4;
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = 255;
        }
    }
    ctx.putImageData(imageData, 0, 0);
    drawContour(0, '#ffffff', 2);
    drawContour(1, '#64d8cb', 1);
    drawContour(-1, '#f4a261', 1);
}
function contourInterp(va, vb, pa, pb) {
    if (Math.abs(vb - va) < 1e-10)
        return (pa + pb) / 2;
    return pa + (pb - pa) * (-va) / (vb - va);
}
function drawContour(level, color, lineWidth) {
    if (!svm)
        return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.setLineDash(level === 0 ? [] : [4, 4]);
    ctx.beginPath();
    var cols = GRID, rows = GRID;
    var vals = [];
    for (var row = 0; row <= rows; row++) {
        vals[row] = [];
        for (var col = 0; col <= cols; col++) {
            var _a = canvasToData(col / cols * W, row / rows * H), x = _a[0], y = _a[1];
            vals[row][col] = svm.predict(x, y) - level;
        }
    }
    for (var row = 0; row < rows; row++) {
        for (var col = 0; col < cols; col++) {
            var a = vals[row][col];
            var b = vals[row][col + 1];
            var c = vals[row + 1][col + 1];
            var d = vals[row + 1][col];
            var x0 = col / cols * W;
            var x1 = (col + 1) / cols * W;
            var y0 = row / rows * H;
            var y1 = (row + 1) / rows * H;
            var top_1 = a * b <= 0 ? [contourInterp(a, b, x0, x1), y0] : null;
            var right = b * c <= 0 ? [x1, contourInterp(b, c, y0, y1)] : null;
            var bottom = d * c <= 0 ? [contourInterp(d, c, x0, x1), y1] : null;
            var left = a * d <= 0 ? [x0, contourInterp(a, d, y0, y1)] : null;
            var segs = [];
            if (top_1)
                segs.push(top_1);
            if (right)
                segs.push(right);
            if (bottom)
                segs.push(bottom);
            if (left)
                segs.push(left);
            if (segs.length >= 2) {
                ctx.moveTo(segs[0][0], segs[0][1]);
                ctx.lineTo(segs[1][0], segs[1][1]);
            }
        }
    }
    ctx.stroke();
    ctx.restore();
}
function drawPoints(svIndices) {
    var radius = 5;
    trainedPoints.forEach(function (p, i) {
        var _a = dataToCanvas(p.x, p.y), cx = _a[0], cy = _a[1];
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
        ctx.fillStyle = p.label === 1 ? '#60a5fa' : '#f87171';
        ctx.fill();
        ctx.strokeStyle = '#1a1a2e';
        ctx.lineWidth = 1;
        ctx.stroke();
        if (svIndices.has(i)) {
            ctx.beginPath();
            ctx.arc(cx, cy, radius + 4, 0, 2 * Math.PI);
            ctx.strokeStyle = '#facc15';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
}
function drawUntrained() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);
    points.forEach(function (p) {
        var _a = dataToCanvas(p.x, p.y), cx = _a[0], cy = _a[1];
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
        ctx.fillStyle = p.label === 1 ? '#60a5fa' : '#f87171';
        ctx.fill();
        ctx.strokeStyle = '#1a1a2e';
        ctx.lineWidth = 1;
        ctx.stroke();
    });
}
function redraw() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);
    if (svm) {
        drawDecisionBoundary();
        var svSet = new Set(svm.getSupportVectorIndices());
        drawPoints(svSet);
        var acc = svm.accuracy(trainedPoints);
        statsEl.textContent = 'Accuracy: ' + (acc * 100).toFixed(1) + '%  |  Support Vectors: ' + svSet.size;
    }
    else {
        drawUntrained();
        statsEl.textContent = '';
    }
}
function updateKernelUI() {
    var k = selKernel.value;
    rowGamma.style.display = k === 'rbf' ? '' : 'none';
    rowDegree.style.display = k === 'poly' ? '' : 'none';
}
inpC.addEventListener('input', function () { lblC.textContent = parseFloat(inpC.value).toFixed(2); });
inpGamma.addEventListener('input', function () { lblGamma.textContent = parseFloat(inpGamma.value).toFixed(2); });
inpDegree.addEventListener('input', function () { lblDegree.textContent = inpDegree.value; });
inpN.addEventListener('input', function () { lblN.textContent = inpN.value; });
inpNoise.addEventListener('input', function () { lblNoise.textContent = parseFloat(inpNoise.value).toFixed(2); });
selKernel.addEventListener('change', updateKernelUI);
btnReset.addEventListener('click', function () {
    svm = null;
    trainedPoints = [];
    points = generateDataset(selDataset.value, parseInt(inpN.value), parseFloat(inpNoise.value));
    statusEl.textContent = '';
    redraw();
});
btnTrain.addEventListener('click', function () {
    points = generateDataset(selDataset.value, parseInt(inpN.value), parseFloat(inpNoise.value));
    trainedPoints = points.slice();
    var kernel = selKernel.value;
    var C = parseFloat(inpC.value);
    var gamma = parseFloat(inpGamma.value);
    var degree = parseInt(inpDegree.value);
    statusEl.textContent = 'Training…';
    btnTrain.disabled = true;
    setTimeout(function () {
        svm = new SVM(C, kernel, gamma, degree);
        svm.train(trainedPoints, 300);
        redraw();
        statusEl.textContent = 'Done.';
        btnTrain.disabled = false;
    }, 20);
});
updateKernelUI();
lblC.textContent = parseFloat(inpC.value).toFixed(2);
lblGamma.textContent = parseFloat(inpGamma.value).toFixed(2);
lblDegree.textContent = inpDegree.value;
lblN.textContent = inpN.value;
lblNoise.textContent = parseFloat(inpNoise.value).toFixed(2);
points = generateDataset('blobs', parseInt(inpN.value), parseFloat(inpNoise.value));
redraw();

},{}]},{},[1]);
