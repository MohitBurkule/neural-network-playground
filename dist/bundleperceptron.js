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
function makeBlobs(n, seed) {
    var pts = [];
    var rng = mulberry32(seed);
    for (var i = 0; i < n; i++) {
        var angle = rng() * Math.PI * 2;
        var r = rng() * 0.25;
        pts.push({ x: -0.45 + Math.cos(angle) * r, y: 0.1 * (rng() - 0.5) + Math.sin(angle) * r, label: -1 });
    }
    for (var i = 0; i < n; i++) {
        var angle = rng() * Math.PI * 2;
        var r = rng() * 0.25;
        pts.push({ x: 0.45 + Math.cos(angle) * r, y: 0.1 * (rng() - 0.5) + Math.sin(angle) * r, label: 1 });
    }
    return pts;
}
function makeNearSeparable(n, seed) {
    var pts = [];
    var rng = mulberry32(seed);
    for (var i = 0; i < n; i++) {
        var angle = rng() * Math.PI * 2;
        var r = rng() * 0.35;
        pts.push({ x: -0.3 + Math.cos(angle) * r, y: Math.sin(angle) * r * 0.8, label: -1 });
    }
    for (var i = 0; i < n; i++) {
        var angle = rng() * Math.PI * 2;
        var r = rng() * 0.35;
        pts.push({ x: 0.3 + Math.cos(angle) * r, y: Math.sin(angle) * r * 0.8, label: 1 });
    }
    pts.push({ x: 0.28, y: 0.05, label: -1 });
    pts.push({ x: -0.28, y: -0.05, label: 1 });
    return pts;
}
function makeXOR() {
    return [
        { x: -0.5, y: -0.5, label: 1 },
        { x: 0.5, y: 0.5, label: 1 },
        { x: -0.5, y: 0.5, label: -1 },
        { x: 0.5, y: -0.5, label: -1 },
        { x: -0.6, y: -0.4, label: 1 },
        { x: 0.6, y: 0.4, label: 1 },
        { x: -0.4, y: 0.6, label: -1 },
        { x: 0.4, y: -0.6, label: -1 },
    ];
}
function makeAND() {
    return [
        { x: -0.7, y: -0.7, label: -1 },
        { x: 0.7, y: -0.7, label: -1 },
        { x: -0.7, y: 0.7, label: -1 },
        { x: 0.7, y: 0.7, label: 1 },
        { x: -0.5, y: -0.5, label: -1 },
        { x: 0.5, y: -0.5, label: -1 },
        { x: -0.5, y: 0.5, label: -1 },
        { x: 0.5, y: 0.5, label: 1 },
    ];
}
function makeOR() {
    return [
        { x: -0.7, y: -0.7, label: -1 },
        { x: 0.7, y: -0.7, label: 1 },
        { x: -0.7, y: 0.7, label: 1 },
        { x: 0.7, y: 0.7, label: 1 },
        { x: -0.5, y: -0.5, label: -1 },
        { x: 0.5, y: -0.5, label: 1 },
        { x: -0.5, y: 0.5, label: 1 },
        { x: 0.5, y: 0.5, label: 1 },
    ];
}
function getDataset(name) {
    switch (name) {
        case "blobs": return makeBlobs(15, 42);
        case "near": return makeNearSeparable(12, 77);
        case "xor": return makeXOR();
        case "and": return makeAND();
        case "or": return makeOR();
        default: return makeBlobs(15, 42);
    }
}
function mulberry32(seed) {
    var s = seed >>> 0;
    return function () {
        s += 0x6D2B79F5;
        var t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
function fitSVM(points, maxIter) {
    if (maxIter === void 0) { maxIter = 2000; }
    var w = [0.01, 0.01];
    var b = 0;
    var lr = 0.01;
    var C = 10;
    for (var iter = 0; iter < maxIter; iter++) {
        var dw = [w[0], w[1]];
        var db = 0;
        for (var _i = 0, points_1 = points; _i < points_1.length; _i++) {
            var p = points_1[_i];
            var margin = p.label * (w[0] * p.x + w[1] * p.y + b);
            if (margin < 1) {
                dw[0] -= C * p.label * p.x;
                dw[1] -= C * p.label * p.y;
                db -= C * p.label;
            }
        }
        w[0] -= lr * dw[0];
        w[1] -= lr * dw[1];
        b -= lr * db;
    }
    return { w: w, b: b };
}
function predict(w, b, x, y) {
    return w[0] * x + w[1] * y + b >= 0 ? 1 : -1;
}
function countErrors(pts, w, b) {
    var c = 0;
    for (var _i = 0, pts_1 = pts; _i < pts_1.length; _i++) {
        var p = pts_1[_i];
        if (predict(w, b, p.x, p.y) !== p.label)
            c++;
    }
    return c;
}
function perceptronStep(state) {
    var pts = state.points;
    if (pts.length === 0)
        return false;
    var idx = state.currentIdx % pts.length;
    state.currentIdx = idx;
    var p = pts[idx];
    var pred = predict(state.w, state.b, p.x, p.y);
    var misclass = pred !== p.label;
    if (misclass) {
        state.w[0] += state.lr * p.label * p.x;
        state.w[1] += state.lr * p.label * p.y;
        state.b += state.lr * p.label;
    }
    state.currentIdx = (idx + 1) % pts.length;
    if (state.currentIdx === 0) {
        state.epoch++;
        var errs = countErrors(pts, state.w, state.b);
        state.errorHistory.push(errs);
        if (state.usePocket) {
            if (errs < state.pocketErrors) {
                state.pocketErrors = errs;
                state.pocketW = [state.w[0], state.w[1]];
                state.pocketB = state.b;
            }
        }
        if (errs === 0) {
            state.converged = true;
            return false;
        }
    }
    state.iteration++;
    state.misclassified = new Set();
    for (var i = 0; i < pts.length; i++) {
        if (predict(state.w, state.b, pts[i].x, pts[i].y) !== pts[i].label) {
            state.misclassified.add(i);
        }
    }
    return true;
}
var CANVAS_W = 500;
var CANVAS_H = 500;
var PAD = 40;
function toCanvasX(x) {
    return PAD + (x + 1) / 2 * (CANVAS_W - 2 * PAD);
}
function toCanvasY(y) {
    return PAD + (1 - (y + 1) / 2) * (CANVAS_H - 2 * PAD);
}
function fromCanvasX(cx) {
    return (cx - PAD) / (CANVAS_W - 2 * PAD) * 2 - 1;
}
function fromCanvasY(cy) {
    return 1 - (cy - PAD) / (CANVAS_H - 2 * PAD) * 2;
}
function drawCanvas(ctx, state) {
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = "#0f0f1a";
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.strokeStyle = "#1e1e38";
    ctx.lineWidth = 1;
    for (var gx = -1; gx <= 1.01; gx += 0.25) {
        ctx.beginPath();
        ctx.moveTo(toCanvasX(gx), PAD);
        ctx.lineTo(toCanvasX(gx), CANVAS_H - PAD);
        ctx.stroke();
    }
    for (var gy = -1; gy <= 1.01; gy += 0.25) {
        ctx.beginPath();
        ctx.moveTo(PAD, toCanvasY(gy));
        ctx.lineTo(CANVAS_W - PAD, toCanvasY(gy));
        ctx.stroke();
    }
    ctx.strokeStyle = "#3a3a6a";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(PAD, toCanvasY(0));
    ctx.lineTo(CANVAS_W - PAD, toCanvasY(0));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), PAD);
    ctx.lineTo(toCanvasX(0), CANVAS_H - PAD);
    ctx.stroke();
    var _a = state.w, w0 = _a[0], w1 = _a[1];
    var wb = state.b;
    var dw = state.usePocket && state.pocketErrors <= countErrors(state.points, state.w, state.b)
        ? { w: state.pocketW, b: state.pocketB }
        : { w: state.w, b: state.b };
    if (state.showSVM && state.svmW[0] !== 0) {
        ctx.save();
        ctx.setLineDash([8, 5]);
        ctx.strokeStyle = "#fbbf24aa";
        ctx.lineWidth = 1.5;
        drawBoundaryLine(ctx, state.svmW[0], state.svmW[1], state.svmB);
        ctx.restore();
        ctx.fillStyle = "#fbbf24";
        ctx.font = "11px monospace";
        ctx.fillText("SVM max-margin", PAD + 4, PAD + 14);
    }
    if (state.usePocket) {
        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = "#a78bfa88";
        ctx.lineWidth = 1.5;
        drawBoundaryLine(ctx, state.pocketW[0], state.pocketW[1], state.pocketB);
        ctx.restore();
    }
    if (w0 !== 0 || w1 !== 0) {
        ctx.save();
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.5;
        ctx.shadowColor = "#38bdf8";
        ctx.shadowBlur = 8;
        drawBoundaryLine(ctx, w0, w1, wb);
        ctx.restore();
        var wmag = Math.sqrt(w0 * w0 + w1 * w1);
        if (wmag > 0) {
            var nx = w0 / wmag * 0.18;
            var ny = w1 / wmag * 0.18;
            var ox = toCanvasX(0), oy = toCanvasY(0);
            var tx = toCanvasX(nx), ty = toCanvasY(ny);
            ctx.save();
            ctx.strokeStyle = "#34d399";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(ox, oy);
            ctx.lineTo(tx, ty);
            ctx.stroke();
            var angle = Math.atan2(oy - ty, ox - tx);
            ctx.beginPath();
            ctx.moveTo(tx, ty);
            ctx.lineTo(tx + Math.cos(angle - 0.4) * 9, ty + Math.sin(angle - 0.4) * 9);
            ctx.lineTo(tx + Math.cos(angle + 0.4) * 9, ty + Math.sin(angle + 0.4) * 9);
            ctx.closePath();
            ctx.fillStyle = "#34d399";
            ctx.fill();
            ctx.restore();
            ctx.fillStyle = "#34d399";
            ctx.font = "11px monospace";
            ctx.fillText("w", toCanvasX(nx * 1.3), toCanvasY(ny * 1.3));
        }
    }
    if (w0 !== 0 || w1 !== 0) {
        var imgData = ctx.getImageData(PAD, PAD, CANVAS_W - 2 * PAD, CANVAS_H - 2 * PAD);
        var dat = imgData.data;
        var pw = CANVAS_W - 2 * PAD;
        var ph = CANVAS_H - 2 * PAD;
        for (var py = 0; py < ph; py++) {
            for (var px = 0; px < pw; px++) {
                var wx = fromCanvasX(px + PAD);
                var wy = fromCanvasY(py + PAD);
                var val = w0 * wx + w1 * wy + wb;
                var idx4 = (py * pw + px) * 4;
                if (val > 0) {
                    dat[idx4] = Math.min(255, dat[idx4] + 10);
                    dat[idx4 + 2] = Math.min(255, dat[idx4 + 2] + 28);
                }
                else {
                    dat[idx4] = Math.min(255, dat[idx4] + 28);
                    dat[idx4 + 2] = Math.min(255, dat[idx4 + 2] + 8);
                }
            }
        }
        ctx.putImageData(imgData, PAD, PAD);
    }
    for (var i = 0; i < state.points.length; i++) {
        var p = state.points[i];
        var cx = toCanvasX(p.x), cy = toCanvasY(p.y);
        var isMis = state.misclassified.has(i);
        var isCurrent = (i === ((state.currentIdx - 1 + state.points.length) % state.points.length));
        ctx.beginPath();
        ctx.arc(cx, cy, isCurrent ? 10 : 7, 0, Math.PI * 2);
        if (p.label === 1) {
            ctx.fillStyle = isMis ? "#f87171" : "#38bdf8";
            ctx.strokeStyle = isCurrent ? "#ffffff" : "#1e1e38";
        }
        else {
            ctx.fillStyle = isMis ? "#f87171" : "#fb923c";
            ctx.strokeStyle = isCurrent ? "#ffffff" : "#1e1e38";
        }
        ctx.lineWidth = isCurrent ? 2.5 : 1.5;
        ctx.fill();
        ctx.stroke();
        if (p.label === -1) {
            ctx.strokeStyle = "#0f0f1a";
            ctx.lineWidth = 1.5;
            var r = 3.5;
            ctx.beginPath();
            ctx.moveTo(cx - r, cy - r);
            ctx.lineTo(cx + r, cy + r);
            ctx.moveTo(cx + r, cy - r);
            ctx.lineTo(cx - r, cy + r);
            ctx.stroke();
        }
        else {
            ctx.beginPath();
            ctx.arc(cx, cy, 2, 0, Math.PI * 2);
            ctx.fillStyle = "#0f0f1a";
            ctx.fill();
        }
        if (isMis) {
            ctx.beginPath();
            ctx.arc(cx, cy, 11, 0, Math.PI * 2);
            ctx.strokeStyle = "#f87171aa";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([3, 2]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }
    ctx.strokeStyle = "#2a2a4a";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(PAD, PAD, CANVAS_W - 2 * PAD, CANVAS_H - 2 * PAD);
}
function drawBoundaryLine(ctx, w0, w1, b) {
    if (Math.abs(w1) < 1e-10 && Math.abs(w0) < 1e-10)
        return;
    var x1, y1, x2, y2;
    if (Math.abs(w1) > Math.abs(w0)) {
        x1 = -1.05;
        y1 = -(w0 * x1 + b) / w1;
        x2 = 1.05;
        y2 = -(w0 * x2 + b) / w1;
    }
    else {
        y1 = -1.05;
        x1 = -(w1 * y1 + b) / w0;
        y2 = 1.05;
        x2 = -(w1 * y2 + b) / w0;
    }
    ctx.beginPath();
    ctx.moveTo(toCanvasX(x1), toCanvasY(y1));
    ctx.lineTo(toCanvasX(x2), toCanvasY(y2));
    ctx.stroke();
}
function drawErrorCurve(ctx2, history, total) {
    var W = ctx2.canvas.width, H = ctx2.canvas.height;
    ctx2.clearRect(0, 0, W, H);
    ctx2.fillStyle = "#0f0f1a";
    ctx2.fillRect(0, 0, W, H);
    if (history.length < 2) {
        ctx2.fillStyle = "#555580";
        ctx2.font = "12px monospace";
        ctx2.fillText("Run to see error curve", 10, H / 2);
        return;
    }
    var px = 28, py = 12, pr = 18, pb = 22;
    var iw = W - px - pr, ih = H - py - pb;
    var maxE = Math.max.apply(Math, __spreadArray([total], history, false));
    var scaleX = function (i) { return px + (i / (history.length - 1)) * iw; };
    var scaleY = function (e) { return py + ih - (e / maxE) * ih; };
    ctx2.strokeStyle = "#1e1e38";
    ctx2.lineWidth = 1;
    for (var i = 0; i <= 4; i++) {
        var y = py + (i / 4) * ih;
        ctx2.beginPath();
        ctx2.moveTo(px, y);
        ctx2.lineTo(px + iw, y);
        ctx2.stroke();
        var val = Math.round(maxE * (1 - i / 4));
        ctx2.fillStyle = "#555580";
        ctx2.font = "9px monospace";
        ctx2.fillText(String(val), 2, y + 3);
    }
    ctx2.strokeStyle = "#3a3a6a";
    ctx2.lineWidth = 1;
    ctx2.beginPath();
    ctx2.moveTo(px, py);
    ctx2.lineTo(px, py + ih);
    ctx2.lineTo(px + iw, py + ih);
    ctx2.stroke();
    ctx2.beginPath();
    ctx2.strokeStyle = "#f87171";
    ctx2.lineWidth = 2;
    for (var i = 0; i < history.length; i++) {
        var x = scaleX(i), y = scaleY(history[i]);
        if (i === 0)
            ctx2.moveTo(x, y);
        else
            ctx2.lineTo(x, y);
    }
    ctx2.stroke();
    ctx2.fillStyle = "#8888bb";
    ctx2.font = "10px monospace";
    ctx2.fillText("epoch", px + iw / 2 - 15, H - 4);
    ctx2.fillStyle = "#8888bb";
    ctx2.save();
    ctx2.translate(8, py + ih / 2);
    ctx2.rotate(-Math.PI / 2);
    ctx2.fillText("errors", -18, 0);
    ctx2.restore();
}
function initState(dataset) {
    var points = getDataset(dataset);
    var total = points.length;
    return {
        w: [0.01, 0.01],
        b: 0,
        pocketW: [0.01, 0.01],
        pocketB: 0,
        pocketErrors: total,
        iteration: 0,
        epoch: 0,
        currentIdx: 0,
        misclassified: new Set(),
        errorHistory: [],
        converged: false,
        running: false,
        usePocket: false,
        showSVM: false,
        lr: 0.1,
        speed: 80,
        animFrame: null,
        lastStepTime: 0,
        dataset: dataset,
        points: points,
        addingClass: 1,
        svmW: [0, 0],
        svmB: 0
    };
}
function recomputeSVM(state) {
    if (state.points.length >= 2) {
        var res = fitSVM(state.points);
        state.svmW = res.w;
        state.svmB = res.b;
    }
}
function main() {
    var app = document.getElementById("app");
    app.innerHTML = buildHTML();
    var state = initState("blobs");
    recomputeSVM(state);
    var canvas = document.getElementById("main-canvas");
    var ctx = canvas.getContext("2d");
    var errCanvas = document.getElementById("err-canvas");
    var ctx2 = errCanvas.getContext("2d");
    function render() {
        drawCanvas(ctx, state);
        drawErrorCurve(ctx2, state.errorHistory, state.points.length);
        updateStats();
    }
    function updateStats() {
        var errs = countErrors(state.points, state.w, state.b);
        var pocketErrs = state.usePocket ? state.pocketErrors : errs;
        document.getElementById("stat-iter").textContent = String(state.iteration);
        document.getElementById("stat-epoch").textContent = String(state.epoch);
        document.getElementById("stat-errors").textContent = String(errs);
        document.getElementById("stat-pocket").textContent = state.usePocket ? String(pocketErrs) : "—";
        var convEl = document.getElementById("stat-conv");
        if (state.converged) {
            convEl.textContent = "CONVERGED ✓";
            convEl.style.color = "#4ade80";
        }
        else if (state.epoch > 200) {
            convEl.textContent = "NOT CONVERGING";
            convEl.style.color = "#f87171";
        }
        else {
            convEl.textContent = state.running ? "running…" : "—";
            convEl.style.color = "#8888bb";
        }
        document.getElementById("lbl-lr").textContent = state.lr.toFixed(3);
        document.getElementById("lbl-speed").textContent = state.speed + "ms";
        var wEl = document.getElementById("stat-weights");
        wEl.textContent = "w=[".concat(state.w[0].toFixed(3), ", ").concat(state.w[1].toFixed(3), "] b=").concat(state.b.toFixed(3));
    }
    function loop(ts) {
        if (!state.running)
            return;
        if (ts - state.lastStepTime >= state.speed) {
            state.lastStepTime = ts;
            var cont = perceptronStep(state);
            if (!cont) {
                state.running = false;
                updatePlayPauseBtn();
            }
        }
        render();
        if (state.running) {
            state.animFrame = requestAnimationFrame(loop);
        }
    }
    function startLoop() {
        if (state.animFrame !== null)
            cancelAnimationFrame(state.animFrame);
        state.animFrame = requestAnimationFrame(loop);
    }
    function updatePlayPauseBtn() {
        var btn = document.getElementById("btn-playpause");
        btn.textContent = state.running ? "⏸ Pause" : "▶ Play";
    }
    document.getElementById("btn-playpause").addEventListener("click", function () {
        if (state.converged)
            return;
        state.running = !state.running;
        updatePlayPauseBtn();
        if (state.running)
            startLoop();
    });
    document.getElementById("btn-step").addEventListener("click", function () {
        if (state.converged)
            return;
        state.running = false;
        updatePlayPauseBtn();
        perceptronStep(state);
        render();
    });
    document.getElementById("btn-reset").addEventListener("click", function () {
        var ds = state.dataset;
        var lr = state.lr;
        var speed = state.speed;
        var pocket = state.usePocket;
        var showSVM = state.showSVM;
        var pts = state.points.slice();
        if (state.animFrame !== null)
            cancelAnimationFrame(state.animFrame);
        state = initState(ds);
        state.lr = lr;
        state.speed = speed;
        state.usePocket = pocket;
        state.showSVM = showSVM;
        state.points = pts;
        recomputeSVM(state);
        updatePlayPauseBtn();
        render();
    });
    document.getElementById("sel-dataset").addEventListener("change", function (e) {
        var ds = e.target.value;
        if (state.animFrame !== null)
            cancelAnimationFrame(state.animFrame);
        var lr = state.lr;
        var speed = state.speed;
        var pocket = state.usePocket;
        var showSVM = state.showSVM;
        state = initState(ds);
        state.lr = lr;
        state.speed = speed;
        state.usePocket = pocket;
        state.showSVM = showSVM;
        recomputeSVM(state);
        var noteEl = document.getElementById("sep-note");
        if (ds === "xor") {
            noteEl.textContent = "⚠ XOR is NOT linearly separable — the perceptron will oscillate forever. Toggle Pocket to keep best-so-far weights.";
            noteEl.style.display = "block";
        }
        else if (ds === "near") {
            noteEl.textContent = "ℹ Near-separable: a few overlapping points — perceptron may not converge. Pocket helps.";
            noteEl.style.display = "block";
        }
        else {
            noteEl.style.display = "none";
        }
        updatePlayPauseBtn();
        render();
    });
    document.getElementById("sl-lr").addEventListener("input", function (e) {
        state.lr = parseFloat(e.target.value);
        document.getElementById("lbl-lr").textContent = state.lr.toFixed(3);
    });
    document.getElementById("sl-speed").addEventListener("input", function (e) {
        state.speed = parseInt(e.target.value);
        document.getElementById("lbl-speed").textContent = state.speed + "ms";
    });
    document.getElementById("chk-pocket").addEventListener("change", function (e) {
        state.usePocket = e.target.checked;
        render();
    });
    document.getElementById("chk-svm").addEventListener("change", function (e) {
        state.showSVM = e.target.checked;
        if (state.showSVM)
            recomputeSVM(state);
        render();
    });
    document.getElementById("btn-class1").addEventListener("click", function () {
        state.addingClass = 1;
        document.getElementById("btn-class1").classList.add("active-class");
        document.getElementById("btn-class-1").classList.remove("active-class");
    });
    document.getElementById("btn-class-1").addEventListener("click", function () {
        state.addingClass = -1;
        document.getElementById("btn-class-1").classList.add("active-class");
        document.getElementById("btn-class1").classList.remove("active-class");
    });
    canvas.addEventListener("click", function (e) {
        var rect = canvas.getBoundingClientRect();
        var scaleX = CANVAS_W / rect.width;
        var scaleY = CANVAS_H / rect.height;
        var cx = (e.clientX - rect.left) * scaleX;
        var cy = (e.clientY - rect.top) * scaleY;
        var wx = fromCanvasX(cx);
        var wy = fromCanvasY(cy);
        if (wx < -1 || wx > 1 || wy < -1 || wy > 1)
            return;
        state.points.push({ x: wx, y: wy, label: state.addingClass });
        if (state.showSVM)
            recomputeSVM(state);
        render();
    });
    render();
}
function buildHTML() {
    return "\n<div class=\"header\">\n  <h1>Perceptron Lab</h1>\n  <div class=\"subtitle\">Rosenblatt (1957) \u2014 visualize the learning algorithm, convergence theorem, and pocket algorithm</div>\n</div>\n<div class=\"main-layout\">\n  <div class=\"sidebar\">\n    <div class=\"ctrl-section\">\n      <div class=\"ctrl-label\">Dataset</div>\n      <select id=\"sel-dataset\" class=\"ctrl-select\">\n        <option value=\"blobs\">Linearly Separable Blobs</option>\n        <option value=\"near\">Near-Separable</option>\n        <option value=\"xor\">XOR (non-separable)</option>\n        <option value=\"and\">AND gate</option>\n        <option value=\"or\">OR gate</option>\n      </select>\n    </div>\n\n    <div id=\"sep-note\" class=\"sep-note\" style=\"display:none\"></div>\n\n    <div class=\"ctrl-section\">\n      <div class=\"ctrl-label\">Learning Rate</div>\n      <div class=\"ctrl-row\">\n        <label>lr</label>\n        <input id=\"sl-lr\" type=\"range\" min=\"0.001\" max=\"1\" step=\"0.001\" value=\"0.1\">\n        <span id=\"lbl-lr\">0.100</span>\n      </div>\n    </div>\n\n    <div class=\"ctrl-section\">\n      <div class=\"ctrl-label\">Animation Speed</div>\n      <div class=\"ctrl-row\">\n        <label>ms/step</label>\n        <input id=\"sl-speed\" type=\"range\" min=\"10\" max=\"800\" step=\"10\" value=\"80\">\n        <span id=\"lbl-speed\">80ms</span>\n      </div>\n    </div>\n\n    <div class=\"ctrl-section\">\n      <div class=\"ctrl-label\">Options</div>\n      <div class=\"ctrl-row toggle-row\">\n        <input id=\"chk-pocket\" type=\"checkbox\">\n        <label for=\"chk-pocket\">Pocket algorithm (best-so-far)</label>\n      </div>\n      <div class=\"ctrl-row toggle-row\" style=\"margin-top:4px\">\n        <input id=\"chk-svm\" type=\"checkbox\">\n        <label for=\"chk-svm\">Overlay SVM max-margin separator</label>\n      </div>\n    </div>\n\n    <div class=\"ctrl-section\">\n      <div class=\"ctrl-label\">Playback</div>\n      <div class=\"btn-row\">\n        <button id=\"btn-playpause\" class=\"ctrl-btn primary\">\u25B6 Play</button>\n        <button id=\"btn-step\" class=\"ctrl-btn\">\u23ED Step</button>\n        <button id=\"btn-reset\" class=\"ctrl-btn danger\">\u21BA Reset</button>\n      </div>\n    </div>\n\n    <div class=\"ctrl-section\">\n      <div class=\"ctrl-label\">Add Points (click canvas)</div>\n      <div class=\"btn-row\">\n        <button id=\"btn-class1\" class=\"ctrl-btn active-class class-btn class1\">\u25CF Class +1</button>\n        <button id=\"btn-class-1\" class=\"ctrl-btn class-btn class-neg1\">\u2715 Class \u22121</button>\n      </div>\n    </div>\n\n    <div class=\"ctrl-section\">\n      <div class=\"ctrl-label\">Stats</div>\n      <div class=\"stat-grid\">\n        <div class=\"stat-item\"><span class=\"stat-key\">Iteration</span><span id=\"stat-iter\" class=\"stat-val\">0</span></div>\n        <div class=\"stat-item\"><span class=\"stat-key\">Epoch</span><span id=\"stat-epoch\" class=\"stat-val\">0</span></div>\n        <div class=\"stat-item\"><span class=\"stat-key\">Errors</span><span id=\"stat-errors\" class=\"stat-val\">\u2014</span></div>\n        <div class=\"stat-item\"><span class=\"stat-key\">Pocket best</span><span id=\"stat-pocket\" class=\"stat-val\">\u2014</span></div>\n        <div class=\"stat-item\" style=\"grid-column:1/-1\"><span class=\"stat-key\">Status</span><span id=\"stat-conv\" class=\"stat-val\">\u2014</span></div>\n        <div class=\"stat-item\" style=\"grid-column:1/-1\"><span class=\"stat-key\">Weights</span><span id=\"stat-weights\" class=\"stat-val mono\">\u2014</span></div>\n      </div>\n    </div>\n\n    <div class=\"ctrl-section\">\n      <div class=\"ctrl-label\">Legend</div>\n      <div class=\"legend-row\"><span class=\"leg-dot dot-pos\"></span> Class +1 (blue circle)</div>\n      <div class=\"legend-row\"><span class=\"leg-dot dot-neg\"></span> Class \u22121 (orange \u2715)</div>\n      <div class=\"legend-row\"><span class=\"leg-dot dot-mis\"></span> Misclassified</div>\n      <div class=\"legend-row\"><span class=\"leg-line line-current\"></span> Perceptron boundary</div>\n      <div class=\"legend-row\"><span class=\"leg-line line-pocket\"></span> Pocket best boundary</div>\n      <div class=\"legend-row\"><span class=\"leg-line line-svm\"></span> SVM max-margin</div>\n      <div class=\"legend-row\"><span class=\"leg-line line-w\"></span> Weight vector</div>\n    </div>\n\n    <div class=\"algo-box\">\n      <div class=\"ctrl-label\">Algorithm</div>\n      <div class=\"algo-text\">For each point (x\u1D62, y\u1D62) with label y\u1D62 \u2208 {+1,\u22121}:<br>\n      If sign(w\u00B7x\u1D62 + b) \u2260 y\u1D62:<br>\n      &nbsp;&nbsp;w \u2190 w + lr\u00B7y\u1D62\u00B7x\u1D62<br>\n      &nbsp;&nbsp;b \u2190 b + lr\u00B7y\u1D62<br><br>\n      <b>Convergence theorem:</b> if data is linearly separable, the algorithm terminates in finite steps.<br><br>\n      <b>Pocket:</b> when data is NOT separable, standard perceptron cycles. Pocket keeps the weight vector with fewest errors seen so far.</div>\n    </div>\n  </div>\n\n  <div class=\"content\">\n    <div class=\"canvas-wrap\">\n      <canvas id=\"main-canvas\" width=\"".concat(CANVAS_W, "\" height=\"").concat(CANVAS_H, "\"></canvas>\n    </div>\n    <div class=\"err-panel\">\n      <div class=\"panel-title\">Misclassification Error per Epoch</div>\n      <canvas id=\"err-canvas\" width=\"460\" height=\"130\"></canvas>\n    </div>\n  </div>\n</div>\n  ");
}
document.addEventListener("DOMContentLoaded", main);

},{}]},{},[1]);
