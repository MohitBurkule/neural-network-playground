(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function mat(rows, cols) {
    return { rows: rows, cols: cols, data: new Float64Array(rows * cols) };
}
function mget(m, r, c) { return m.data[r * m.cols + c]; }
function mset(m, r, c, v) { m.data[r * m.cols + c] = v; }
function mvmul(A, x, out) {
    var R = A.rows, C = A.cols;
    for (var r = 0; r < R; r++) {
        var s = 0;
        for (var c = 0; c < C; c++)
            s += mget(A, r, c) * x[c];
        out[r] = s;
    }
}
function mvmulT(A, x, out) {
    var R = A.rows, C = A.cols;
    out.fill(0);
    for (var r = 0; r < R; r++) {
        var xr = x[r];
        for (var c = 0; c < C; c++)
            out[c] += mget(A, r, c) * xr;
    }
}
function randn() {
    var u = 1 - Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function generateSignal(name, T) {
    var s = new Float64Array(T);
    if (name === 'sine') {
        for (var t = 0; t < T; t++)
            s[t] = Math.sin(2 * Math.PI * t / 40);
    }
    else if (name === 'sum-of-sines') {
        for (var t = 0; t < T; t++)
            s[t] = 0.5 * Math.sin(2 * Math.PI * t / 30) + 0.5 * Math.sin(2 * Math.PI * t / 73);
    }
    else if (name === 'mackey-glass') {
        var tau = 17, a = 0.2, b = 0.1, n = 10, dt = 1;
        var buf = new Float64Array(T + tau + 1);
        for (var i = 0; i <= tau; i++)
            buf[i] = 1.2;
        for (var t = tau; t < T + tau; t++) {
            var xt = buf[t], xtau = buf[t - tau];
            buf[t + 1] = xt + dt * (a * xtau / (1 + Math.pow(xtau, n)) - b * xt);
        }
        var mn = Infinity, mx = -Infinity;
        for (var t = 0; t < T; t++) {
            mn = Math.min(mn, buf[t + tau]);
            mx = Math.max(mx, buf[t + tau]);
        }
        var rng = mx - mn || 1;
        for (var t = 0; t < T; t++)
            s[t] = 2 * (buf[t + tau] - mn) / rng - 1;
    }
    else if (name === 'square') {
        for (var t = 0; t < T; t++)
            s[t] = Math.sin(2 * Math.PI * t / 50) >= 0 ? 1 : -1;
    }
    else if (name === 'lorenz') {
        var dt = 0.02, sigma = 10, rho = 28, beta = 8 / 3;
        var x = 1, y = 1, z = 1;
        var buf = [];
        for (var t = 0; t < T + 500; t++) {
            var dx = sigma * (y - x), dy = x * (rho - z) - y, dz = x * y - beta * z;
            x += dt * dx;
            y += dt * dy;
            z += dt * dz;
            if (t >= 500)
                buf.push(x);
        }
        var mn_1 = Infinity, mx_1 = -Infinity;
        buf.forEach(function (v) { mn_1 = Math.min(mn_1, v); mx_1 = Math.max(mx_1, v); });
        var rng = mx_1 - mn_1 || 1;
        for (var t = 0; t < T; t++)
            s[t] = 2 * (buf[t] - mn_1) / rng - 1;
    }
    return s;
}
function buildReservoir(N, sparsity, targetSR) {
    var W = mat(N, N);
    for (var r = 0; r < N; r++) {
        for (var c = 0; c < N; c++) {
            if (Math.random() < sparsity)
                mset(W, r, c, randn());
        }
    }
    var sr = spectralRadiusApprox(W, N);
    if (sr > 0) {
        var scale = targetSR / sr;
        for (var i = 0; i < W.data.length; i++)
            W.data[i] *= scale;
    }
    return W;
}
function spectralRadiusApprox(W, N) {
    var v = new Float64Array(N);
    for (var i = 0; i < N; i++)
        v[i] = randn();
    var norm = 0;
    for (var i = 0; i < N; i++)
        norm += v[i] * v[i];
    norm = Math.sqrt(norm) || 1;
    for (var i = 0; i < N; i++)
        v[i] /= norm;
    var ev = 1;
    var tmp = new Float64Array(N);
    for (var iter = 0; iter < 40; iter++) {
        mvmul(W, v, tmp);
        norm = 0;
        for (var i = 0; i < N; i++)
            norm += tmp[i] * tmp[i];
        ev = Math.sqrt(norm) || 1e-12;
        for (var i = 0; i < N; i++)
            v[i] = tmp[i] / ev;
    }
    return ev;
}
function ridgeRegression(X, y, lambda) {
    var T = X.length, N = X[0].length;
    var A = new Float64Array(N * N);
    var b = new Float64Array(N);
    for (var t = 0; t < T; t++) {
        var xt = X[t];
        for (var i = 0; i < N; i++) {
            b[i] += xt[i] * y[t];
            for (var j = 0; j < N; j++)
                A[i * N + j] += xt[i] * xt[j];
        }
    }
    for (var i = 0; i < N; i++)
        A[i * N + i] += lambda;
    return solveLinear(A, b, N);
}
function solveLinear(A, b, N) {
    var M = new Float64Array(N * (N + 1));
    for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++)
            M[i * (N + 1) + j] = A[i * N + j];
        M[i * (N + 1) + N] = b[i];
    }
    for (var col = 0; col < N; col++) {
        var pivot = col;
        var pivVal = Math.abs(M[col * (N + 1) + col]);
        for (var row = col + 1; row < N; row++) {
            var v = Math.abs(M[row * (N + 1) + col]);
            if (v > pivVal) {
                pivVal = v;
                pivot = row;
            }
        }
        if (pivot !== col) {
            for (var j = 0; j <= N; j++) {
                var tmp = M[col * (N + 1) + j];
                M[col * (N + 1) + j] = M[pivot * (N + 1) + j];
                M[pivot * (N + 1) + j] = tmp;
            }
        }
        var diag = M[col * (N + 1) + col];
        if (Math.abs(diag) < 1e-15)
            continue;
        for (var row = col + 1; row < N; row++) {
            var factor = M[row * (N + 1) + col] / diag;
            for (var j = col; j <= N; j++)
                M[row * (N + 1) + j] -= factor * M[col * (N + 1) + j];
        }
    }
    var x = new Float64Array(N);
    for (var row = N - 1; row >= 0; row--) {
        var sum = M[row * (N + 1) + N];
        for (var j = row + 1; j < N; j++)
            sum -= M[row * (N + 1) + j] * x[j];
        var d = M[row * (N + 1) + row];
        x[row] = Math.abs(d) < 1e-15 ? 0 : sum / d;
    }
    return x;
}
function esnStep(state, Wr, Win, input, leak) {
    var N = state.length;
    var pre = new Float64Array(N);
    mvmul(Wr, state, pre);
    for (var i = 0; i < N; i++) {
        pre[i] += Win[i] * input;
        pre[i] = Math.tanh(pre[i]);
        state[i] = (1 - leak) * state[i] + leak * pre[i];
    }
}
var ESN = (function () {
    function ESN(cfg) {
        this.Wout = null;
        this.cfg = cfg;
        var N = cfg.N, spectralRadius = cfg.spectralRadius, inputScale = cfg.inputScale, sparsity = cfg.sparsity;
        this.Wr = buildReservoir(N, sparsity, spectralRadius);
        this.Win = new Float64Array(N);
        for (var i = 0; i < N; i++)
            this.Win[i] = (Math.random() * 2 - 1) * inputScale;
        this.state = new Float64Array(N);
    }
    ESN.prototype.reset = function () { this.state.fill(0); };
    ESN.prototype.collectStates = function (signal, washout) {
        var _a = this.cfg, N = _a.N, leak = _a.leak;
        this.reset();
        var states = [];
        var targets = [];
        for (var t = 0; t < signal.length - 1; t++) {
            esnStep(this.state, this.Wr, this.Win, signal[t], leak);
            if (t >= washout) {
                states.push(new Float64Array(this.state));
                targets.push(signal[t + 1]);
            }
        }
        return { states: states, targets: new Float64Array(targets) };
    };
    ESN.prototype.train = function (signal, washout) {
        var _a = this.collectStates(signal, washout), states = _a.states, targets = _a.targets;
        this.Wout = ridgeRegression(states, targets, this.cfg.ridge);
    };
    ESN.prototype.readout = function () {
        if (!this.Wout)
            return 0;
        var s = 0;
        for (var i = 0; i < this.state.length; i++)
            s += this.Wout[i] * this.state[i];
        return s;
    };
    ESN.prototype.predictTeacherForced = function (signal, washout) {
        var leak = this.cfg.leak;
        this.reset();
        var preds = [];
        for (var t = 0; t < signal.length - 1; t++) {
            esnStep(this.state, this.Wr, this.Win, signal[t], leak);
            if (t >= washout)
                preds.push(this.readout());
        }
        return new Float64Array(preds);
    };
    ESN.prototype.freeRun = function (seed, steps) {
        var leak = this.cfg.leak;
        var out = new Float64Array(steps);
        var inp = seed;
        for (var t = 0; t < steps; t++) {
            esnStep(this.state, this.Wr, this.Win, inp, leak);
            inp = this.readout();
            out[t] = inp;
        }
        return out;
    };
    ESN.prototype.getActivations = function (signal, washout, nSample) {
        var leak = this.cfg.leak;
        this.reset();
        var rows = [];
        var N = this.state.length;
        var step = Math.max(1, Math.floor(N / nSample));
        var indices = [];
        for (var i = 0; i < N && indices.length < nSample; i += step)
            indices.push(i);
        for (var _ = 0; _ < indices.length; _++)
            rows.push(new Float64Array(signal.length - 1 - washout));
        for (var t = 0; t < signal.length - 1; t++) {
            esnStep(this.state, this.Wr, this.Win, signal[t], leak);
            if (t >= washout) {
                for (var k = 0; k < indices.length; k++) {
                    rows[k][t - washout] = this.state[indices[k]];
                }
            }
        }
        return rows;
    };
    return ESN;
}());
var app = {
    esn: null,
    signal: null,
    trainPreds: null,
    freeRunPreds: null,
    activations: null,
    trained: false,
    trainLen: 500,
    washout: 100
};
var TOTAL_LEN = 1200;
var FREERUN_STEPS = 400;
var ACT_SAMPLE = 20;
function getConfig() {
    return {
        N: parseInt(document.getElementById('res-size').value),
        spectralRadius: parseFloat(document.getElementById('spectral-radius').value),
        inputScale: parseFloat(document.getElementById('input-scale').value),
        leak: parseFloat(document.getElementById('leak-rate').value),
        ridge: parseFloat(document.getElementById('ridge').value),
        sparsity: 0.1
    };
}
function getSignalName() {
    return document.getElementById('signal-sel').value;
}
function rmse(a, b, start, end) {
    if (start === void 0) { start = 0; }
    var e = end !== null && end !== void 0 ? end : Math.min(a.length, b.length);
    var s = 0, n = 0;
    for (var i = start; i < e; i++) {
        var d = a[i] - b[i];
        s += d * d;
        n++;
    }
    return n > 0 ? Math.sqrt(s / n) : 0;
}
function drawSignalPlot() {
    var _a;
    var canvas = document.getElementById('signal-canvas');
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0a14';
    ctx.fillRect(0, 0, W, H);
    if (!app.signal)
        return;
    var sig = app.signal;
    var trainLen = app.trainLen;
    var washout = app.washout;
    var showTrain = trainLen - washout;
    var showFree = FREERUN_STEPS;
    var totalShow = showTrain + showFree;
    var xScale = (W - 60) / totalShow;
    var yMid = H / 2;
    var yScale = (H - 40) / 2.2;
    var toX = function (i) { return 40 + i * xScale; };
    var toY = function (v) { return yMid - v * yScale; };
    ctx.fillStyle = 'rgba(56,189,248,0.04)';
    ctx.fillRect(40, 10, showTrain * xScale, H - 20);
    ctx.fillStyle = 'rgba(251,113,133,0.06)';
    ctx.fillRect(40 + showTrain * xScale, 10, showFree * xScale, H - 20);
    ctx.strokeStyle = '#666';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(40 + showTrain * xScale, 10);
    ctx.lineTo(40 + showTrain * xScale, H - 10);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#38bdf8';
    ctx.font = '11px monospace';
    ctx.fillText('Training', 50, 22);
    ctx.fillStyle = '#fb7185';
    ctx.fillText('Free-run', 40 + showTrain * xScale + 8, 22);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (var i = 0; i < showTrain; i++) {
        var v = sig[i + washout + 1];
        if (i === 0)
            ctx.moveTo(toX(i), toY(v));
        else
            ctx.lineTo(toX(i), toY(v));
    }
    ctx.stroke();
    if (app.freeRunPreds) {
        ctx.strokeStyle = 'rgba(56,189,248,0.35)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        for (var i = 0; i < showFree; i++) {
            var v = (_a = sig[trainLen + i + 1]) !== null && _a !== void 0 ? _a : 0;
            if (i === 0)
                ctx.moveTo(toX(showTrain + i), toY(v));
            else
                ctx.lineTo(toX(showTrain + i), toY(v));
        }
        ctx.stroke();
    }
    if (app.trainPreds) {
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (var i = 0; i < Math.min(showTrain, app.trainPreds.length); i++) {
            var v = app.trainPreds[i];
            if (i === 0)
                ctx.moveTo(toX(i), toY(v));
            else
                ctx.lineTo(toX(i), toY(v));
        }
        ctx.stroke();
    }
    if (app.freeRunPreds) {
        ctx.strokeStyle = '#fb7185';
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (var i = 0; i < app.freeRunPreds.length; i++) {
            var v = app.freeRunPreds[i];
            if (i === 0)
                ctx.moveTo(toX(showTrain + i), toY(v));
            else
                ctx.lineTo(toX(showTrain + i), toY(v));
        }
        ctx.stroke();
    }
    ctx.strokeStyle = '#334';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, 10);
    ctx.lineTo(40, H - 10);
    ctx.stroke();
    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    ctx.fillText('1', 24, toY(1) + 4);
    ctx.fillText('0', 28, toY(0) + 4);
    ctx.fillText('-1', 20, toY(-1) + 4);
    ctx.beginPath();
    ctx.moveTo(36, toY(1));
    ctx.lineTo(40, toY(1));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(36, toY(0));
    ctx.lineTo(40, toY(0));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(36, toY(-1));
    ctx.lineTo(40, toY(-1));
    ctx.stroke();
    var lx = W - 200;
    ctx.fillStyle = '#38bdf8';
    ctx.fillRect(lx, H - 42, 18, 3);
    ctx.fillStyle = '#ccc';
    ctx.font = '11px sans-serif';
    ctx.fillText('Target', lx + 22, H - 35);
    ctx.fillStyle = '#facc15';
    ctx.fillRect(lx, H - 28, 18, 3);
    ctx.fillStyle = '#ccc';
    ctx.fillText('Train pred', lx + 22, H - 21);
    ctx.fillStyle = '#fb7185';
    ctx.fillRect(lx + 100, H - 42, 18, 3);
    ctx.fillStyle = '#ccc';
    ctx.fillText('Free-run', lx + 122, H - 35);
}
function drawActivationPlot() {
    var canvas = document.getElementById('act-canvas');
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0a14';
    ctx.fillRect(0, 0, W, H);
    if (!app.activations || app.activations.length === 0) {
        ctx.fillStyle = '#555';
        ctx.font = '13px sans-serif';
        ctx.fillText('Train to see reservoir activations', W / 2 - 130, H / 2);
        return;
    }
    var acts = app.activations;
    var nNeurons = acts.length;
    var T = acts[0].length;
    var rowH = (H - 20) / nNeurons;
    var xScale = (W - 40) / T;
    for (var k = 0; k < nNeurons; k++) {
        var y0 = 10 + k * rowH;
        for (var t = 0; t < T; t++) {
            var v = Math.max(-1, Math.min(1, acts[k][t]));
            var x = 40 + t * xScale;
            var r = void 0, g = void 0, b = void 0;
            if (v >= 0) {
                r = Math.round(251 * v + 20 * (1 - v));
                g = Math.round(80 * v + 20 * (1 - v));
                b = Math.round(10 * v + 20 * (1 - v));
            }
            else {
                var u = -v;
                r = Math.round(20 * (1 - u));
                g = Math.round(80 * (1 - u));
                b = Math.round(200 * u + 20 * (1 - u));
            }
            ctx.fillStyle = "rgb(".concat(r, ",").concat(g, ",").concat(b, ")");
            ctx.fillRect(x, y0, Math.ceil(xScale) + 1, Math.ceil(rowH) + 1);
        }
    }
    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    ctx.fillText('Neuron', 2, H / 2);
    ctx.fillStyle = '#555';
    ctx.font = '10px monospace';
    ctx.fillText('← time →', 40, H - 2);
    ctx.fillStyle = '#555';
    ctx.font = '11px sans-serif';
    ctx.fillText("".concat(nNeurons, " reservoir neurons  \u00B7  training window"), 50, 14);
}
function drawWeightsPlot() {
    var canvas = document.getElementById('wout-canvas');
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0a14';
    ctx.fillRect(0, 0, W, H);
    if (!app.esn || !app.esn.Wout) {
        ctx.fillStyle = '#555';
        ctx.font = '13px sans-serif';
        ctx.fillText('Train to see readout weights', W / 2 - 110, H / 2);
        return;
    }
    var w = app.esn.Wout;
    var N = w.length;
    var mx = 0;
    for (var i = 0; i < N; i++)
        mx = Math.max(mx, Math.abs(w[i]));
    if (mx === 0)
        mx = 1;
    var barW = Math.max(1, (W - 40) / N);
    var yMid = H / 2;
    var yScale = (H - 30) / 2;
    for (var i = 0; i < N; i++) {
        var v = w[i] / mx;
        var x = 40 + i * barW;
        var barH = Math.abs(v) * yScale;
        ctx.fillStyle = v >= 0 ? 'rgba(56,189,248,0.7)' : 'rgba(251,113,133,0.7)';
        ctx.fillRect(x, v >= 0 ? yMid - barH : yMid, barW - 0.5, barH);
    }
    ctx.strokeStyle = '#334';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(40, yMid);
    ctx.lineTo(W, yMid);
    ctx.stroke();
    ctx.fillStyle = '#777';
    ctx.font = '11px monospace';
    ctx.fillText("Readout weights (N=".concat(N, ")"), 44, 14);
}
function updateStats() {
    var elTrain = document.getElementById('stat-train');
    var elTest = document.getElementById('stat-test');
    if (!app.signal || !app.trainPreds) {
        elTrain.textContent = '—';
        elTest.textContent = '—';
        return;
    }
    var washout = app.washout;
    var trainLen = app.trainLen;
    var target = app.signal.slice(washout + 1, trainLen + 1);
    var tErr = rmse(target, app.trainPreds, 0, Math.min(target.length, app.trainPreds.length));
    elTrain.textContent = tErr.toFixed(5);
    if (app.freeRunPreds) {
        var freeTarget = app.signal.slice(trainLen + 1, trainLen + 1 + FREERUN_STEPS);
        var fErr = rmse(freeTarget, app.freeRunPreds, 0, Math.min(freeTarget.length, app.freeRunPreds.length));
        elTest.textContent = fErr.toFixed(5);
    }
    else {
        elTest.textContent = '—';
    }
}
function redrawAll() {
    drawSignalPlot();
    drawActivationPlot();
    drawWeightsPlot();
    updateStats();
}
function handleReset() {
    var cfg = getConfig();
    app.esn = new ESN(cfg);
    app.signal = generateSignal(getSignalName(), TOTAL_LEN);
    app.trainPreds = null;
    app.freeRunPreds = null;
    app.activations = null;
    app.trained = false;
    document.getElementById('btn-freerun').disabled = true;
    redrawAll();
    setStatus('Reservoir reset. Click "Train" to train readout weights.');
}
function handleTrain() {
    if (!app.esn || !app.signal) {
        handleReset();
        return;
    }
    var washout = app.washout;
    var trainLen = app.trainLen;
    var trainSig = app.signal.slice(0, trainLen);
    setStatus('Training…');
    setTimeout(function () {
        app.esn.train(trainSig, washout);
        app.trainPreds = app.esn.predictTeacherForced(trainSig, washout);
        app.activations = app.esn.getActivations(trainSig, washout, ACT_SAMPLE);
        app.trained = true;
        document.getElementById('btn-freerun').disabled = false;
        app.freeRunPreds = null;
        redrawAll();
        setStatus('Trained! Train RMSE updated. Click "Free-run" to generate.');
    }, 10);
}
function handleFreeRun() {
    if (!app.esn || !app.signal || !app.trained)
        return;
    setStatus('Free-running…');
    setTimeout(function () {
        var trainLen = app.trainLen;
        var washout = app.washout;
        var trainSig = app.signal.slice(0, trainLen);
        app.esn.reset();
        var leak = app.esn.cfg.leak;
        for (var t = 0; t < trainSig.length - 1; t++) {
            esnStep(app.esn.state, app.esn.Wr, app.esn.Win, trainSig[t], leak);
        }
        var seed = trainSig[trainSig.length - 1];
        app.freeRunPreds = app.esn.freeRun(seed, FREERUN_STEPS);
        redrawAll();
        setStatus('Free-run complete.');
    }, 10);
}
function setStatus(msg) {
    var el = document.getElementById('status');
    if (el)
        el.textContent = msg;
}
function bindSlider(id, labelId, fmt) {
    var el = document.getElementById(id);
    var lbl = document.getElementById(labelId);
    var update = function () {
        var v = parseFloat(el.value);
        lbl.textContent = fmt ? fmt(v) : String(v);
    };
    el.addEventListener('input', update);
    update();
}
function init() {
    bindSlider('res-size', 'lbl-res-size', function (v) { return String(Math.round(v)); });
    bindSlider('spectral-radius', 'lbl-sr');
    bindSlider('input-scale', 'lbl-is');
    bindSlider('leak-rate', 'lbl-lr');
    bindSlider('ridge', 'lbl-ridge', function (v) { return v.toExponential(0); });
    document.getElementById('btn-reset').addEventListener('click', handleReset);
    document.getElementById('btn-train').addEventListener('click', handleTrain);
    document.getElementById('btn-freerun').addEventListener('click', handleFreeRun);
    document.getElementById('signal-sel').addEventListener('change', handleReset);
    handleReset();
}
window.addEventListener('DOMContentLoaded', init);

},{}]},{},[1]);
