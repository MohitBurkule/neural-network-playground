(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var trainData = [];
var testData = [];
var k = 5;
var metric = "euclidean";
var weighted = false;
var nPoints = 200;
var noise = 0.1;
var dataset = "blobs";
var queryPoint = null;
var mainCanvas;
var mainCtx;
var curveCanvas;
var curveCtx;
var GRID = 80;
var PALETTE = [
    { fill: "#6366f1", region: "rgba(99,102,241,0.18)", border: "#818cf8" },
    { fill: "#f43f5e", region: "rgba(244,63,94,0.18)", border: "#fb7185" },
    { fill: "#10b981", region: "rgba(16,185,129,0.18)", border: "#34d399" },
    { fill: "#f59e0b", region: "rgba(245,158,11,0.18)", border: "#fbbf24" },
];
function dist(a, b, m) {
    var dx = a.x - b.x, dy = a.y - b.y;
    if (m === "euclidean")
        return Math.sqrt(dx * dx + dy * dy);
    if (m === "manhattan")
        return Math.abs(dx) + Math.abs(dy);
    return Math.max(Math.abs(dx), Math.abs(dy));
}
function classify(q, data, kk, m, w) {
    var sorted = data.slice().sort(function (a, b) { return dist(q, a, m) - dist(q, b, m); });
    var neighbors = sorted.slice(0, kk);
    var nClasses = Math.max.apply(Math, data.map(function (p) { return p.label; })) + 1;
    var votes = new Array(nClasses).fill(0);
    for (var _i = 0, neighbors_1 = neighbors; _i < neighbors_1.length; _i++) {
        var nb = neighbors_1[_i];
        var d = dist(q, nb, m);
        var weight = w ? (d < 1e-9 ? 1e9 : 1 / (d * d)) : 1;
        votes[nb.label] += weight;
    }
    var label = 0;
    for (var i = 1; i < votes.length; i++)
        if (votes[i] > votes[label])
            label = i;
    return { label: label, votes: votes, neighbors: neighbors };
}
function seededRand(seed) {
    var s = seed >>> 0;
    return function () { s = (1664525 * s + 1013904223) >>> 0; return s / 4294967296; };
}
function addNoise(v, n, rng) {
    return v + (rng() * 2 - 1) * n;
}
function generateData(type, n, ns, seed) {
    if (seed === void 0) { seed = 42; }
    var rng = seededRand(seed);
    var pts = [];
    if (type === "blobs") {
        var centers = [{ x: 0.3, y: 0.3 }, { x: 0.7, y: 0.7 }, { x: 0.3, y: 0.7 }, { x: 0.7, y: 0.3 }];
        var nc = 2;
        for (var i = 0; i < n; i++) {
            var c = i % nc;
            pts.push({ x: addNoise(centers[c].x, 0.12 + ns * 0.2, rng), y: addNoise(centers[c].y, 0.12 + ns * 0.2, rng), label: c });
        }
    }
    else if (type === "moons") {
        for (var i = 0; i < n; i++) {
            var t = (i / n) * Math.PI;
            var label = i < n / 2 ? 0 : 1;
            var cx = label === 0 ? 0.5 + 0.35 * Math.cos(t) : 0.5 - 0.35 * Math.cos(t);
            var cy = label === 0 ? 0.5 - 0.18 * Math.sin(t) : 0.5 + 0.18 * Math.sin(t);
            pts.push({ x: addNoise(cx, ns * 0.15, rng), y: addNoise(cy, ns * 0.15, rng), label: label });
        }
    }
    else if (type === "circles") {
        for (var i = 0; i < n; i++) {
            var label = i < n / 2 ? 0 : 1;
            var r = label === 0 ? 0.15 : 0.32;
            var t = rng() * 2 * Math.PI;
            pts.push({ x: addNoise(0.5 + r * Math.cos(t), ns * 0.07, rng), y: addNoise(0.5 + r * Math.sin(t), ns * 0.07, rng), label: label });
        }
    }
    else if (type === "xor") {
        for (var i = 0; i < n; i++) {
            var x = rng();
            var y = rng();
            var label = ((x > 0.5) !== (y > 0.5)) ? 1 : 0;
            pts.push({ x: addNoise(x, ns * 0.08, rng), y: addNoise(y, ns * 0.08, rng), label: label });
        }
    }
    else if (type === "spiral") {
        var nPerClass = Math.floor(n / 2);
        for (var c = 0; c < 2; c++) {
            for (var i = 0; i < nPerClass; i++) {
                var t = (i / nPerClass) * 3 * Math.PI + c * Math.PI;
                var r = 0.05 + 0.38 * (i / nPerClass);
                pts.push({ x: addNoise(0.5 + r * Math.cos(t), ns * 0.06, rng), y: addNoise(0.5 + r * Math.sin(t), ns * 0.06, rng), label: c });
            }
        }
    }
    else if (type === "three-class") {
        var centers = [{ x: 0.25, y: 0.75 }, { x: 0.75, y: 0.75 }, { x: 0.5, y: 0.25 }];
        for (var i = 0; i < n; i++) {
            var c = i % 3;
            pts.push({ x: addNoise(centers[c].x, 0.1 + ns * 0.18, rng), y: addNoise(centers[c].y, 0.1 + ns * 0.18, rng), label: c });
        }
    }
    for (var _i = 0, pts_1 = pts; _i < pts_1.length; _i++) {
        var p = pts_1[_i];
        p.x = Math.max(0.01, Math.min(0.99, p.x));
        p.y = Math.max(0.01, Math.min(0.99, p.y));
    }
    return pts;
}
function splitTrainTest(pts, trainFrac, seed) {
    if (trainFrac === void 0) { trainFrac = 0.8; }
    if (seed === void 0) { seed = 7; }
    var rng = seededRand(seed);
    var shuffled = pts.slice().sort(function () { return rng() - 0.5; });
    var n = Math.floor(pts.length * trainFrac);
    return [shuffled.slice(0, n), shuffled.slice(n)];
}
function accuracy(data, kk, m, w) {
    if (data.length === 0)
        return 0;
    var correct = 0;
    for (var _i = 0, data_1 = data; _i < data_1.length; _i++) {
        var p = data_1[_i];
        var label = classify(p, trainData, kk, m, w).label;
        if (label === p.label)
            correct++;
    }
    return correct / data.length;
}
function kAccuracyCurve(maxK) {
    var results = [];
    for (var kk = 1; kk <= maxK; kk += (maxK > 20 ? 2 : 1)) {
        results.push({ k: kk, train: accuracy(trainData, kk, metric, weighted), test: accuracy(testData, kk, metric, weighted) });
    }
    return results;
}
function drawDecisionBoundary() {
    var W = mainCanvas.width, H = mainCanvas.height;
    mainCtx.clearRect(0, 0, W, H);
    if (trainData.length === 0)
        return;
    var cw = W / GRID, ch = H / GRID;
    for (var gi = 0; gi < GRID; gi++) {
        for (var gj = 0; gj < GRID; gj++) {
            var qx = (gi + 0.5) / GRID, qy = (gj + 0.5) / GRID;
            var label = classify({ x: qx, y: qy }, trainData, k, metric, weighted).label;
            mainCtx.fillStyle = PALETTE[label % PALETTE.length].region;
            mainCtx.fillRect(gi * cw, gj * ch, cw + 1, ch + 1);
        }
    }
    for (var _i = 0, testData_1 = testData; _i < testData_1.length; _i++) {
        var p = testData_1[_i];
        var px = p.x * W, py = p.y * H;
        var col = PALETTE[p.label % PALETTE.length];
        mainCtx.beginPath();
        mainCtx.arc(px, py, 5, 0, 2 * Math.PI);
        mainCtx.fillStyle = "#1a1a2e";
        mainCtx.fill();
        mainCtx.strokeStyle = col.border;
        mainCtx.lineWidth = 1.5;
        mainCtx.setLineDash([3, 2]);
        mainCtx.stroke();
        mainCtx.setLineDash([]);
    }
    for (var _a = 0, trainData_1 = trainData; _a < trainData_1.length; _a++) {
        var p = trainData_1[_a];
        var px = p.x * W, py = p.y * H;
        var col = PALETTE[p.label % PALETTE.length];
        mainCtx.beginPath();
        mainCtx.arc(px, py, 5, 0, 2 * Math.PI);
        mainCtx.fillStyle = col.fill;
        mainCtx.fill();
        mainCtx.strokeStyle = "#ffffff33";
        mainCtx.lineWidth = 0.8;
        mainCtx.stroke();
    }
    if (queryPoint) {
        var _b = classify(queryPoint, trainData, k, metric, weighted), neighbors = _b.neighbors, label = _b.label, votes = _b.votes;
        var qpx = queryPoint.x * W, qpy = queryPoint.y * H;
        for (var _c = 0, neighbors_2 = neighbors; _c < neighbors_2.length; _c++) {
            var nb = neighbors_2[_c];
            mainCtx.beginPath();
            mainCtx.moveTo(qpx, qpy);
            mainCtx.lineTo(nb.x * W, nb.y * H);
            mainCtx.strokeStyle = "rgba(255,255,255,0.35)";
            mainCtx.lineWidth = 1;
            mainCtx.stroke();
        }
        for (var _d = 0, neighbors_3 = neighbors; _d < neighbors_3.length; _d++) {
            var nb = neighbors_3[_d];
            mainCtx.beginPath();
            mainCtx.arc(nb.x * W, nb.y * H, 8, 0, 2 * Math.PI);
            mainCtx.strokeStyle = "#ffffff88";
            mainCtx.lineWidth = 2;
            mainCtx.stroke();
        }
        mainCtx.beginPath();
        mainCtx.arc(qpx, qpy, 7, 0, 2 * Math.PI);
        mainCtx.fillStyle = PALETTE[label % PALETTE.length].fill;
        mainCtx.fill();
        mainCtx.strokeStyle = "#fff";
        mainCtx.lineWidth = 2;
        mainCtx.stroke();
        var totalVotes = votes.reduce(function (a, b) { return a + b; }, 0);
        var lines = ["Predicted: Class ".concat(label)];
        for (var i = 0; i < votes.length; i++) {
            if (votes[i] > 0)
                lines.push("  C".concat(i, ": ").concat((100 * votes[i] / totalVotes).toFixed(0), "%"));
        }
        var tw = 130, th = lines.length * 16 + 12;
        var tx = qpx + 14, ty = qpy - th / 2;
        if (tx + tw > W)
            tx = qpx - tw - 14;
        if (ty < 4)
            ty = 4;
        mainCtx.fillStyle = "rgba(20,20,40,0.88)";
        mainCtx.strokeStyle = "#ffffff22";
        mainCtx.lineWidth = 1;
        mainCtx.beginPath();
        mainCtx.roundRect(tx, ty, tw, th, 5);
        mainCtx.fill();
        mainCtx.stroke();
        mainCtx.fillStyle = "#e0e0f0";
        mainCtx.font = "12px 'Segoe UI', monospace";
        for (var i = 0; i < lines.length; i++) {
            mainCtx.fillText(lines[i], tx + 8, ty + 14 + i * 16);
        }
    }
}
function drawKCurve() {
    var W = curveCanvas.width, H = curveCanvas.height;
    curveCtx.clearRect(0, 0, W, H);
    var maxK = Math.min(trainData.length, 25);
    if (maxK < 1)
        return;
    var data = kAccuracyCurve(maxK);
    var pad = { l: 40, r: 16, t: 14, b: 32 };
    var pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    curveCtx.strokeStyle = "#ffffff22";
    curveCtx.lineWidth = 1;
    curveCtx.beginPath();
    curveCtx.moveTo(pad.l, pad.t);
    curveCtx.lineTo(pad.l, pad.t + ph);
    curveCtx.lineTo(pad.l + pw, pad.t + ph);
    curveCtx.stroke();
    curveCtx.fillStyle = "#8888aa";
    curveCtx.font = "10px 'Segoe UI'";
    curveCtx.textAlign = "right";
    for (var v = 0; v <= 100; v += 25) {
        var y = pad.t + ph - (v / 100) * ph;
        curveCtx.fillText(v + "%", pad.l - 4, y + 3);
        curveCtx.strokeStyle = "#ffffff0a";
        curveCtx.beginPath();
        curveCtx.moveTo(pad.l, y);
        curveCtx.lineTo(pad.l + pw, y);
        curveCtx.stroke();
    }
    curveCtx.textAlign = "center";
    var kVals = data.map(function (d) { return d.k; });
    var kMin = kVals[0], kMax = kVals[kVals.length - 1];
    var xOf = function (kk) { return pad.l + ((kk - kMin) / (kMax - kMin || 1)) * pw; };
    var yOf = function (v) { return pad.t + ph - v * ph; };
    for (var _i = 0, data_2 = data; _i < data_2.length; _i++) {
        var d = data_2[_i];
        if (d.k === 1 || d.k % 5 === 0 || d.k === kMax) {
            curveCtx.fillStyle = "#8888aa";
            curveCtx.fillText(String(d.k), xOf(d.k), pad.t + ph + 16);
        }
    }
    curveCtx.fillStyle = "#8888aa";
    curveCtx.fillText("k", pad.l + pw / 2, H - 2);
    curveCtx.beginPath();
    curveCtx.strokeStyle = "#6366f1";
    curveCtx.lineWidth = 2;
    for (var i = 0; i < data.length; i++) {
        var x = xOf(data[i].k), y = yOf(data[i].train);
        i === 0 ? curveCtx.moveTo(x, y) : curveCtx.lineTo(x, y);
    }
    curveCtx.stroke();
    curveCtx.beginPath();
    curveCtx.strokeStyle = "#f43f5e";
    curveCtx.lineWidth = 2;
    for (var i = 0; i < data.length; i++) {
        var x = xOf(data[i].k), y = yOf(data[i].test);
        i === 0 ? curveCtx.moveTo(x, y) : curveCtx.lineTo(x, y);
    }
    curveCtx.stroke();
    var curX = xOf(k);
    curveCtx.strokeStyle = "#ffffff55";
    curveCtx.lineWidth = 1;
    curveCtx.setLineDash([3, 3]);
    curveCtx.beginPath();
    curveCtx.moveTo(curX, pad.t);
    curveCtx.lineTo(curX, pad.t + ph);
    curveCtx.stroke();
    curveCtx.setLineDash([]);
    curveCtx.font = "10px 'Segoe UI'";
    curveCtx.textAlign = "left";
    curveCtx.fillStyle = "#6366f1";
    curveCtx.fillText("▬ Train", pad.l + 4, pad.t + 12);
    curveCtx.fillStyle = "#f43f5e";
    curveCtx.fillText("▬ Test", pad.l + 60, pad.t + 12);
}
function updateAccuracy() {
    var trAcc = accuracy(trainData, k, metric, weighted);
    var teAcc = accuracy(testData, k, metric, weighted);
    var el = document.getElementById("accuracy-display");
    if (el)
        el.textContent = "Train: ".concat((trAcc * 100).toFixed(1), "%   Test: ").concat((teAcc * 100).toFixed(1), "%");
}
function redraw() {
    drawDecisionBoundary();
    drawKCurve();
    updateAccuracy();
}
function regenerate(newSeed) {
    var _a;
    var seed = newSeed !== null && newSeed !== void 0 ? newSeed : Math.floor(Math.random() * 100000);
    var all = generateData(dataset, nPoints, noise, seed);
    _a = splitTrainTest(all, 0.8, seed + 1), trainData = _a[0], testData = _a[1];
    queryPoint = null;
    redraw();
}
function canvasCoord(e, canvas) {
    var rect = canvas.getBoundingClientRect();
    var scaleX = canvas.width / rect.width;
    var scaleY = canvas.height / rect.height;
    return { x: ((e.clientX - rect.left) * scaleX) / canvas.width, y: ((e.clientY - rect.top) * scaleY) / canvas.height };
}
function init() {
    mainCanvas = document.getElementById("main-canvas");
    curveCanvas = document.getElementById("curve-canvas");
    mainCtx = mainCanvas.getContext("2d");
    curveCtx = curveCanvas.getContext("2d");
    var datasetSel = document.getElementById("dataset-sel");
    var kSlider = document.getElementById("k-slider");
    var kVal = document.getElementById("k-val");
    var metricSel = document.getElementById("metric-sel");
    var weightToggle = document.getElementById("weight-toggle");
    var nSlider = document.getElementById("n-slider");
    var nVal = document.getElementById("n-val");
    var noiseSlider = document.getElementById("noise-slider");
    var noiseVal = document.getElementById("noise-val");
    var regenBtn = document.getElementById("regen-btn");
    datasetSel.addEventListener("change", function () { dataset = datasetSel.value; regenerate(); });
    kSlider.addEventListener("input", function () { k = parseInt(kSlider.value); kVal.textContent = String(k); redraw(); });
    metricSel.addEventListener("change", function () { metric = metricSel.value; redraw(); });
    weightToggle.addEventListener("change", function () { weighted = weightToggle.checked; redraw(); });
    nSlider.addEventListener("input", function () { nPoints = parseInt(nSlider.value); nVal.textContent = String(nPoints); regenerate(); });
    noiseSlider.addEventListener("input", function () { noise = parseFloat(noiseSlider.value); noiseVal.textContent = noise.toFixed(2); regenerate(); });
    regenBtn.addEventListener("click", function () { return regenerate(); });
    mainCanvas.addEventListener("mousemove", function (e) {
        queryPoint = canvasCoord(e, mainCanvas);
        drawDecisionBoundary();
    });
    mainCanvas.addEventListener("click", function (e) {
        queryPoint = canvasCoord(e, mainCanvas);
        drawDecisionBoundary();
    });
    mainCanvas.addEventListener("mouseleave", function () {
        queryPoint = null;
        drawDecisionBoundary();
    });
    regenerate(42);
}
document.addEventListener("DOMContentLoaded", init);

},{}]},{},[1]);
