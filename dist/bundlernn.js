(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
function mat(rows, cols) {
    return { rows: rows, cols: cols, data: new Float64Array(rows * cols) };
}
function matGet(m, r, c) {
    return m.data[r * m.cols + c];
}
function matSet(m, r, c, v) {
    m.data[r * m.cols + c] = v;
}
function matFill(m, v) {
    m.data.fill(v);
}
function matRand(m, scale) {
    for (var i = 0; i < m.data.length; i++) {
        m.data[i] = (Math.random() * 2 - 1) * scale;
    }
}
function matMulVec(A, x, bias) {
    var out = new Float64Array(A.rows);
    for (var r = 0; r < A.rows; r++) {
        var s = bias[r];
        for (var c = 0; c < A.cols; c++) {
            s += matGet(A, r, c) * x[c];
        }
        out[r] = s;
    }
    return out;
}
function tanh(x) { return Math.tanh(x); }
function dtanh(y) { return 1 - y * y; }
function makeParams(inputSize, hiddenSize, outputSize) {
    var scale = 0.1;
    var Wxh = mat(hiddenSize, inputSize);
    matRand(Wxh, scale);
    var Whh = mat(hiddenSize, hiddenSize);
    matRand(Whh, scale * 0.5);
    var bh = new Float64Array(hiddenSize);
    var Why = mat(outputSize, hiddenSize);
    matRand(Why, scale);
    var by = new Float64Array(outputSize);
    return { Wxh: Wxh, Whh: Whh, bh: bh, Why: Why, by: by };
}
function zeroGrads(p) {
    return {
        Wxh: mat(p.Wxh.rows, p.Wxh.cols),
        Whh: mat(p.Whh.rows, p.Whh.cols),
        bh: new Float64Array(p.bh.length),
        Why: mat(p.Why.rows, p.Why.cols),
        by: new Float64Array(p.by.length)
    };
}
function forward(p, xs, h0) {
    var caches = [];
    var hPrev = h0;
    for (var _i = 0, xs_1 = xs; _i < xs_1.length; _i++) {
        var x = xs_1[_i];
        var zx = matMulVec(p.Wxh, x, p.bh);
        var zh = matMulVec(p.Whh, hPrev, new Float64Array(p.bh.length));
        var z = new Float64Array(zx.length);
        for (var i = 0; i < z.length; i++)
            z[i] = zx[i] + zh[i];
        var h = z.map(tanh);
        var y = matMulVec(p.Why, h, p.by);
        caches.push({ x: x, hPrev: hPrev, z: z, h: h, y: y });
        hPrev = h;
    }
    return { caches: caches, hLast: hPrev };
}
function bptt(p, caches, targets) {
    var grads = zeroGrads(p);
    var loss = 0;
    var dh_next = new Float64Array(p.Whh.rows);
    for (var t = caches.length - 1; t >= 0; t--) {
        var _a = caches[t], x = _a.x, hPrev = _a.hPrev, _z = _a.z, h = _a.h, y = _a.y;
        var target = targets[t];
        var dy = new Float64Array(y.length);
        for (var i = 0; i < y.length; i++) {
            var diff = y[i] - target[i];
            loss += diff * diff;
            dy[i] = 2 * diff;
        }
        for (var r = 0; r < p.Why.rows; r++) {
            grads.by[r] += dy[r];
            for (var c = 0; c < p.Why.cols; c++) {
                grads.Why.data[r * p.Why.cols + c] += dy[r] * h[c];
            }
        }
        var dh = new Float64Array(h.length);
        for (var c = 0; c < p.Why.cols; c++) {
            for (var r = 0; r < p.Why.rows; r++) {
                dh[c] += matGet(p.Why, r, c) * dy[r];
            }
            dh[c] += dh_next[c];
        }
        var dz = new Float64Array(dh.length);
        for (var i = 0; i < dz.length; i++)
            dz[i] = dh[i] * dtanh(h[i]);
        for (var i = 0; i < grads.bh.length; i++)
            grads.bh[i] += dz[i];
        for (var r = 0; r < p.Wxh.rows; r++) {
            for (var c = 0; c < p.Wxh.cols; c++) {
                grads.Wxh.data[r * p.Wxh.cols + c] += dz[r] * x[c];
            }
        }
        for (var r = 0; r < p.Whh.rows; r++) {
            for (var c = 0; c < p.Whh.cols; c++) {
                grads.Whh.data[r * p.Whh.cols + c] += dz[r] * hPrev[c];
            }
        }
        dh_next = new Float64Array(p.Whh.cols);
        for (var c = 0; c < p.Whh.cols; c++) {
            for (var r = 0; r < p.Whh.rows; r++) {
                dh_next[c] += matGet(p.Whh, r, c) * dz[r];
            }
        }
    }
    loss /= caches.length;
    return { grads: grads, loss: loss };
}
function clipGrads(grads, clip) {
    var arrays = [grads.Wxh.data, grads.Whh.data, grads.bh, grads.Why.data, grads.by];
    for (var _i = 0, arrays_1 = arrays; _i < arrays_1.length; _i++) {
        var arr = arrays_1[_i];
        for (var i = 0; i < arr.length; i++) {
            arr[i] = Math.max(-clip, Math.min(clip, arr[i]));
        }
    }
}
function sgdUpdate(p, grads, lr) {
    var pairs = [
        [p.Wxh.data, grads.Wxh.data],
        [p.Whh.data, grads.Whh.data],
        [p.bh, grads.bh],
        [p.Why.data, grads.Why.data],
        [p.by, grads.by],
    ];
    for (var _i = 0, pairs_1 = pairs; _i < pairs_1.length; _i++) {
        var _a = pairs_1[_i], param = _a[0], grad = _a[1];
        for (var i = 0; i < param.length; i++) {
            param[i] -= lr * grad[i];
        }
    }
}
function generateSequence(seqLen, delay) {
    if (delay === void 0) { delay = 2; }
    var inputs = [];
    for (var i = 0; i < seqLen; i++)
        inputs.push(Math.random() > 0.5 ? 1 : 0);
    var targets = inputs.map(function (_v, i) { return (i >= delay ? inputs[i - delay] : 0); });
    return { inputs: inputs, targets: targets };
}
var state;
function initState(hiddenSize, seqLen, bpttLen, lr, delay) {
    var inputSize = 1;
    var outputSize = 1;
    var params = makeParams(inputSize, hiddenSize, outputSize);
    return {
        params: params,
        hiddenSize: hiddenSize,
        seqLen: seqLen,
        bpttLen: bpttLen,
        lr: lr,
        delay: delay,
        running: false, step: 0,
        lossHistory: [],
        lastInputs: [], lastTargets: [], lastPredictions: [], lastHiddenStates: [],
        h0: new Float64Array(hiddenSize)
    };
}
function trainStep(s) {
    var _a = generateSequence(s.seqLen, s.delay), inputs = _a.inputs, targets = _a.targets;
    s.lastInputs = inputs;
    s.lastTargets = targets;
    var h = new Float64Array(s.h0);
    var totalLoss = 0;
    var chunks = 0;
    var allCaches = [];
    for (var start = 0; start < s.seqLen; start += s.bpttLen) {
        var end = Math.min(start + s.bpttLen, s.seqLen);
        var xs = inputs.slice(start, end).map(function (v) { return new Float64Array([v]); });
        var ts = targets.slice(start, end).map(function (v) { return new Float64Array([v]); });
        var _b = forward(s.params, xs, h), caches_1 = _b.caches, hLast = _b.hLast;
        var _c = bptt(s.params, caches_1, ts), grads = _c.grads, loss = _c.loss;
        clipGrads(grads, 5);
        sgdUpdate(s.params, grads, s.lr);
        h = hLast;
        totalLoss += loss;
        chunks++;
        allCaches.push.apply(allCaches, caches_1);
    }
    s.h0 = h;
    s.lossHistory.push(totalLoss / chunks);
    if (s.lossHistory.length > 300)
        s.lossHistory.shift();
    s.lastPredictions = allCaches.map(function (c) { return c.y[0]; });
    s.lastHiddenStates = allCaches.map(function (c) { return c.h; });
    s.step++;
}
function clearCanvas(ctx, w, h) {
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, w, h);
}
function heatColor(v, vmin, vmax) {
    var t = vmax === vmin ? 0.5 : (v - vmin) / (vmax - vmin);
    var c = Math.round(t * 255);
    if (t < 0.5) {
        var r = Math.round(2 * t * 200);
        return "rgb(".concat(r, ",").concat(r, ",255)");
    }
    else {
        var g = Math.round((1 - (t - 0.5) * 2) * 200);
        return "rgb(255,".concat(g, ",").concat(g, ")");
    }
}
function drawSequence(canvas, s) {
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    clearCanvas(ctx, W, H);
    var n = s.lastInputs.length;
    if (n === 0)
        return;
    var pad = 30;
    var innerW = W - 2 * pad;
    var innerH = H - 2 * pad;
    var midY = pad + innerH / 2;
    ctx.strokeStyle = '#333';
    ctx.beginPath();
    ctx.moveTo(pad, midY);
    ctx.lineTo(W - pad, midY);
    ctx.stroke();
    ctx.fillStyle = '#aaa';
    ctx.font = '11px monospace';
    ctx.fillText('1', pad - 18, pad + 4);
    ctx.fillText('0', pad - 18, H - pad + 4);
    var dx = innerW / (n - 1 || 1);
    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    s.lastInputs.forEach(function (v, i) {
        var x = pad + i * dx;
        var y = H - pad - v * innerH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.strokeStyle = '#81c784';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    s.lastTargets.forEach(function (v, i) {
        var x = pad + i * dx;
        var y = H - pad - v * innerH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    if (s.lastPredictions.length === n) {
        ctx.strokeStyle = '#ff8a65';
        ctx.lineWidth = 2;
        ctx.beginPath();
        s.lastPredictions.forEach(function (v, i) {
            var x = pad + i * dx;
            var y = H - pad - Math.max(0, Math.min(1, v)) * innerH;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
    }
    ctx.font = '10px monospace';
    var items = [['#4fc3f7', 'input'], ['#81c784', 'target'], ['#ff8a65', 'predicted']];
    items.forEach(function (_a, i) {
        var color = _a[0], label = _a[1];
        ctx.fillStyle = color;
        ctx.fillRect(pad + i * 90, 6, 12, 8);
        ctx.fillStyle = '#ccc';
        ctx.fillText(label, pad + i * 90 + 15, 14);
    });
    ctx.fillStyle = '#888';
    ctx.fillText("step ".concat(s.step), W - 70, 14);
}
function drawHiddenStates(canvas, s) {
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    clearCanvas(ctx, W, H);
    var states = s.lastHiddenStates;
    if (!states.length)
        return;
    var T = states.length;
    var HS = states[0].length;
    var padL = 50, padT = 20, padR = 10, padB = 20;
    var cellW = (W - padL - padR) / T;
    var cellH = (H - padT - padB) / HS;
    for (var t = 0; t < T; t++) {
        for (var h = 0; h < HS; h++) {
            var v = states[t][h];
            ctx.fillStyle = heatColor(v, -1, 1);
            ctx.fillRect(padL + t * cellW, padT + h * cellH, Math.max(1, cellW - 0.5), Math.max(1, cellH - 0.5));
        }
    }
    ctx.fillStyle = '#aaa';
    ctx.font = '10px monospace';
    ctx.fillText('h units', 2, H / 2);
    ctx.fillText('time →', padL, H - 4);
    [0, Math.floor(T / 2), T - 1].forEach(function (t) {
        ctx.fillStyle = '#888';
        ctx.fillText(String(t), padL + t * cellW, padT - 4);
    });
}
function drawWhh(canvas, s) {
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    clearCanvas(ctx, W, H);
    var Whh = s.params.Whh;
    var HS = Whh.rows;
    var padL = 40, padT = 20, padR = 10, padB = 20;
    var cellW = (W - padL - padR) / HS;
    var cellH = (H - padT - padB) / HS;
    var vmin = Infinity, vmax = -Infinity;
    for (var i = 0; i < Whh.data.length; i++) {
        vmin = Math.min(vmin, Whh.data[i]);
        vmax = Math.max(vmax, Whh.data[i]);
    }
    for (var r = 0; r < HS; r++) {
        for (var c = 0; c < HS; c++) {
            ctx.fillStyle = heatColor(matGet(Whh, r, c), vmin, vmax);
            ctx.fillRect(padL + c * cellW, padT + r * cellH, Math.max(1, cellW - 0.5), Math.max(1, cellH - 0.5));
        }
    }
    ctx.fillStyle = '#aaa';
    ctx.font = '10px monospace';
    ctx.fillText('Whh', 4, H / 2);
    ctx.fillText("[".concat(vmin.toFixed(2), ", ").concat(vmax.toFixed(2), "]"), padL, H - 4);
}
function drawLoss(canvas, s) {
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    clearCanvas(ctx, W, H);
    var history = s.lossHistory;
    if (history.length < 2)
        return;
    var pad = 30;
    var innerW = W - 2 * pad;
    var innerH = H - 2 * pad;
    var maxLoss = Math.max.apply(Math, __spreadArray(__spreadArray([], history, false), [0.01], false));
    var minLoss = Math.min.apply(Math, history);
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, H - pad);
    ctx.lineTo(W - pad, H - pad);
    ctx.stroke();
    ctx.strokeStyle = '#ffd54f';
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach(function (v, i) {
        var x = pad + (i / (history.length - 1)) * innerW;
        var y = H - pad - ((v - minLoss) / (maxLoss - minLoss || 1)) * innerH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = '#aaa';
    ctx.font = '10px monospace';
    ctx.fillText('loss', 2, pad + 4);
    ctx.fillText(maxLoss.toFixed(3), pad + 2, pad + 12);
    ctx.fillText(minLoss.toFixed(3), pad + 2, H - pad - 2);
    var last = history[history.length - 1];
    ctx.fillStyle = '#ffd54f';
    ctx.fillText("cur: ".concat(last.toFixed(4)), W - 100, pad + 12);
}
var animId = null;
function render() {
    var canvSeq = document.getElementById('canvas-seq');
    var canvHidden = document.getElementById('canvas-hidden');
    var canvWhh = document.getElementById('canvas-whh');
    var canvLoss = document.getElementById('canvas-loss');
    drawSequence(canvSeq, state);
    drawHiddenStates(canvHidden, state);
    drawWhh(canvWhh, state);
    drawLoss(canvLoss, state);
}
function loop() {
    if (state.running) {
        trainStep(state);
        render();
        animId = requestAnimationFrame(loop);
    }
}
function doStep() {
    trainStep(state);
    render();
}
function resetState() {
    var hiddenSize = parseInt(document.getElementById('ctrl-hidden').value);
    var seqLen = parseInt(document.getElementById('ctrl-seqlen').value);
    var bpttLen = parseInt(document.getElementById('ctrl-bptt').value);
    var lr = parseFloat(document.getElementById('ctrl-lr').value);
    var delay = parseInt(document.getElementById('ctrl-delay').value);
    state = initState(hiddenSize, seqLen, bpttLen, lr, delay);
    state.running = false;
    render();
}
window.addEventListener('DOMContentLoaded', function () {
    var hiddenSize = parseInt(document.getElementById('ctrl-hidden').value);
    var seqLen = parseInt(document.getElementById('ctrl-seqlen').value);
    var bpttLen = parseInt(document.getElementById('ctrl-bptt').value);
    var lr = parseFloat(document.getElementById('ctrl-lr').value);
    var delay = parseInt(document.getElementById('ctrl-delay').value);
    state = initState(hiddenSize, seqLen, bpttLen, lr, delay);
    function syncLabel(id, labelId) {
        var el = document.getElementById(id);
        var lbl = document.getElementById(labelId);
        lbl.textContent = el.value;
        el.addEventListener('input', function () {
            lbl.textContent = el.value;
            if (id === 'ctrl-lr')
                state.lr = parseFloat(el.value);
        });
    }
    syncLabel('ctrl-lr', 'lbl-lr');
    syncLabel('ctrl-hidden', 'lbl-hidden');
    syncLabel('ctrl-seqlen', 'lbl-seqlen');
    syncLabel('ctrl-bptt', 'lbl-bptt');
    syncLabel('ctrl-delay', 'lbl-delay');
    document.getElementById('btn-play').addEventListener('click', function () {
        state.running = true;
        if (animId === null)
            loop();
    });
    document.getElementById('btn-pause').addEventListener('click', function () {
        state.running = false;
        if (animId !== null) {
            cancelAnimationFrame(animId);
            animId = null;
        }
    });
    document.getElementById('btn-step').addEventListener('click', function () {
        state.running = false;
        if (animId !== null) {
            cancelAnimationFrame(animId);
            animId = null;
        }
        doStep();
    });
    document.getElementById('btn-reset').addEventListener('click', function () {
        state.running = false;
        if (animId !== null) {
            cancelAnimationFrame(animId);
            animId = null;
        }
        resetState();
    });
    render();
});

},{}]},{},[1]);
