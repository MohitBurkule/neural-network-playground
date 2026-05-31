(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
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
    return {
        W: W,
        b: b,
        mW: zeros(outDim, inDim), vW: zeros(outDim, inDim),
        mb: new Array(outDim).fill(0), vb: new Array(outDim).fill(0),
        dW: zeros(outDim, inDim), db: new Array(outDim).fill(0)
    };
}
function linearForward(layer, x) {
    return layer.W.map(function (row, i) {
        return row.reduce(function (s, w, j) { return s + w * x[j]; }, 0) + layer.b[i];
    });
}
function relu(x) {
    return x.map(function (v) { return Math.max(0, v); });
}
function reluGrad(x) {
    return x.map(function (v) { return (v > 0 ? 1 : 0); });
}
function makeMLP(inDim, hiddenSize, outDim, numHidden) {
    var layers = [];
    var d = inDim;
    for (var i = 0; i < numHidden; i++) {
        layers.push(makeLayer(d, hiddenSize));
        d = hiddenSize;
    }
    layers.push(makeLayer(d, outDim));
    return { layers: layers, t: 0 };
}
function resetGrads(mlp) {
    for (var _i = 0, _a = mlp.layers; _i < _a.length; _i++) {
        var l = _a[_i];
        l.dW = zeros(l.W.length, l.W[0].length);
        l.db = new Array(l.b.length).fill(0);
    }
}
function forward(mlp, x) {
    var preActs = [];
    var acts = [];
    var cur = x;
    for (var i = 0; i < mlp.layers.length; i++) {
        var pre = linearForward(mlp.layers[i], cur);
        preActs.push(pre);
        if (i < mlp.layers.length - 1) {
            cur = relu(pre);
        }
        else {
            cur = pre;
        }
        acts.push(cur);
    }
    return { preActs: preActs, acts: acts };
}
function backward(mlp, x, fwd, dOut) {
    var delta = dOut;
    var _loop_1 = function (i) {
        var layer = mlp.layers[i];
        if (i < mlp.layers.length - 1) {
            delta = delta.map(function (d, k) { return d * reluGrad(fwd.preActs[i])[k]; });
        }
        var inp = i === 0 ? x : fwd.acts[i - 1];
        for (var r = 0; r < layer.W.length; r++) {
            for (var c = 0; c < layer.W[r].length; c++) {
                layer.dW[r][c] += delta[r] * inp[c];
            }
            layer.db[r] += delta[r];
        }
        if (i > 0) {
            var newDelta = new Array(layer.W[0].length).fill(0);
            for (var c = 0; c < layer.W[0].length; c++) {
                for (var r = 0; r < layer.W.length; r++) {
                    newDelta[c] += layer.W[r][c] * delta[r];
                }
            }
            delta = newDelta;
        }
    };
    for (var i = mlp.layers.length - 1; i >= 0; i--) {
        _loop_1(i);
    }
}
function adamStep(mlp, lr, beta1, beta2, eps) {
    if (beta1 === void 0) { beta1 = 0.9; }
    if (beta2 === void 0) { beta2 = 0.999; }
    if (eps === void 0) { eps = 1e-8; }
    mlp.t += 1;
    var t = mlp.t;
    var bc1 = 1 - Math.pow(beta1, t);
    var bc2 = 1 - Math.pow(beta2, t);
    for (var _i = 0, _a = mlp.layers; _i < _a.length; _i++) {
        var layer = _a[_i];
        for (var r = 0; r < layer.W.length; r++) {
            for (var c = 0; c < layer.W[r].length; c++) {
                layer.mW[r][c] = beta1 * layer.mW[r][c] + (1 - beta1) * layer.dW[r][c];
                layer.vW[r][c] = beta2 * layer.vW[r][c] + (1 - beta2) * layer.dW[r][c] * layer.dW[r][c];
                var mHat_1 = layer.mW[r][c] / bc1;
                var vHat_1 = layer.vW[r][c] / bc2;
                layer.W[r][c] -= lr * mHat_1 / (Math.sqrt(vHat_1) + eps);
            }
            layer.mb[r] = beta1 * layer.mb[r] + (1 - beta1) * layer.db[r];
            layer.vb[r] = beta2 * layer.vb[r] + (1 - beta2) * layer.db[r] * layer.db[r];
            var mHat = layer.mb[r] / bc1;
            var vHat = layer.vb[r] / bc2;
            layer.b[r] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
    }
}
function sinEmbedding(t, dim) {
    var half = dim >> 1;
    var emb = [];
    for (var i = 0; i < half; i++) {
        var freq = Math.pow(10000, -i / half);
        emb.push(Math.sin(t * freq * Math.PI * 2));
        emb.push(Math.cos(t * freq * Math.PI * 2));
    }
    return emb.slice(0, dim);
}
function makeSchedule(T, betaStart, betaEnd) {
    if (betaStart === void 0) { betaStart = 0.0001; }
    if (betaEnd === void 0) { betaEnd = 0.02; }
    var betas = [];
    var alphas = [];
    var alphaBar = [];
    var cumAlpha = 1;
    for (var i = 0; i < T; i++) {
        var beta = betaStart + (betaEnd - betaStart) * i / (T - 1);
        betas.push(beta);
        var alpha = 1 - beta;
        alphas.push(alpha);
        cumAlpha *= alpha;
        alphaBar.push(cumAlpha);
    }
    return { T: T, betas: betas, alphas: alphas, alphaBar: alphaBar };
}
function qSample(x0, t, sched) {
    var ab = sched.alphaBar[t];
    var sqrtAb = Math.sqrt(ab);
    var sqrtOneMinusAb = Math.sqrt(1 - ab);
    var eps = [randn(), randn()];
    var xt = x0.map(function (v, i) { return sqrtAb * v + sqrtOneMinusAb * eps[i]; });
    return { xt: xt, eps: eps };
}
function sampleDist(name, n) {
    var pts = [];
    if (name === 'ring') {
        for (var i = 0; i < n; i++) {
            var angle = Math.random() * 2 * Math.PI;
            var r = 0.9 + randn() * 0.08;
            pts.push([r * Math.cos(angle), r * Math.sin(angle)]);
        }
    }
    else if (name === 'two-moons') {
        for (var i = 0; i < n; i++) {
            var upper = Math.random() < 0.5;
            var angle = Math.PI * Math.random();
            var r = 0.85 + randn() * 0.07;
            if (upper) {
                pts.push([r * Math.cos(angle), r * Math.sin(angle)]);
            }
            else {
                pts.push([r * Math.cos(angle + Math.PI) + 0.05, r * Math.sin(angle + Math.PI) + 0.45]);
            }
        }
    }
    else if (name === 'spiral') {
        for (var i = 0; i < n; i++) {
            var arm = Math.floor(Math.random() * 2);
            var frac = Math.random();
            var angle = frac * 3 * Math.PI + arm * Math.PI;
            var r = 0.1 + 0.8 * frac;
            pts.push([r * Math.cos(angle) + randn() * 0.04, r * Math.sin(angle) + randn() * 0.04]);
        }
    }
    else if (name === '8-gaussians') {
        var centers = [
            [0.85, 0], [-0.85, 0], [0, 0.85], [0, -0.85],
            [0.6, 0.6], [-0.6, 0.6], [0.6, -0.6], [-0.6, -0.6],
        ];
        for (var i = 0; i < n; i++) {
            var c = centers[Math.floor(Math.random() * centers.length)];
            pts.push([c[0] + randn() * 0.1, c[1] + randn() * 0.1]);
        }
    }
    else if (name === 's-curve') {
        for (var i = 0; i < n; i++) {
            var t = Math.random() * 2 * Math.PI;
            var x = Math.sin(t) * 0.9 + randn() * 0.06;
            var y = (Math.cos(t) * 0.45 + (t > Math.PI ? 0.45 : -0.45)) * 0.9 + randn() * 0.06;
            pts.push([x, y]);
        }
    }
    return pts;
}
var T_EMB_DIM = 16;
function makeDenoiser(hiddenSize) {
    return makeMLP(2 + T_EMB_DIM, hiddenSize, 2, 2);
}
function predictNoise(mlp, xt, tNorm) {
    var emb = sinEmbedding(tNorm, T_EMB_DIM);
    var inp = xt.concat(emb);
    var fwd = forward(mlp, inp);
    return fwd.acts[fwd.acts.length - 1];
}
function predictNoiseFwd(mlp, xt, tNorm) {
    var emb = sinEmbedding(tNorm, T_EMB_DIM);
    var inp = xt.concat(emb);
    var fwd = forward(mlp, inp);
    var pred = fwd.acts[fwd.acts.length - 1];
    return { pred: pred, fwd: fwd, inp: inp };
}
function trainStep(mlp, data, sched, lr, batchSize) {
    resetGrads(mlp);
    var totalLoss = 0;
    var _loop_2 = function (b) {
        var x0 = data[Math.floor(Math.random() * data.length)];
        var t = Math.floor(Math.random() * sched.T);
        var _a = qSample(x0, t, sched), xt = _a.xt, eps = _a.eps;
        var tNorm = t / (sched.T - 1);
        var _b = predictNoiseFwd(mlp, xt, tNorm), pred = _b.pred, fwd = _b.fwd, inp = _b.inp;
        var dOut = pred.map(function (p, i) { return (p - eps[i]) / batchSize; });
        totalLoss += pred.reduce(function (s, p, i) { return s + (p - eps[i]) * (p - eps[i]); }, 0);
        backward(mlp, inp, fwd, dOut);
    };
    for (var b = 0; b < batchSize; b++) {
        _loop_2(b);
    }
    adamStep(mlp, lr);
    return totalLoss / (batchSize * 2);
}
function reverseSample(mlp, sched, n) {
    var traj = [];
    var pts = Array.from({ length: n }, function () { return [randn(), randn()]; });
    traj.push(pts.map(function (p) { return [p[0], p[1]]; }));
    var _loop_3 = function (t) {
        var tNorm = t / (sched.T - 1);
        var beta = sched.betas[t];
        var alpha = sched.alphas[t];
        var alphaBar = sched.alphaBar[t];
        var alphaBarPrev = t > 0 ? sched.alphaBar[t - 1] : 1;
        pts = pts.map(function (xt) {
            var epsTheta = predictNoise(mlp, xt, tNorm);
            var coeff = beta / Math.sqrt(1 - alphaBar);
            var mean = xt.map(function (v, i) { return (v - coeff * epsTheta[i]) / Math.sqrt(alpha); });
            if (t === 0) {
                return mean;
            }
            var variance = beta * (1 - alphaBarPrev) / (1 - alphaBar);
            var sigma = Math.sqrt(variance);
            return mean.map(function (v) { return v + sigma * randn(); });
        });
        traj.push(pts.map(function (p) { return [p[0], p[1]]; }));
    };
    for (var t = sched.T - 1; t >= 0; t--) {
        _loop_3(t);
    }
    return traj;
}
var NUM_DATA = 300;
var SAMPLE_N = 150;
var state;
function initState() {
    var dist = 'ring';
    var T = 40;
    var hiddenSize = 32;
    var data = sampleDist(dist, NUM_DATA);
    var sched = makeSchedule(T);
    return {
        dist: dist,
        T: T,
        hiddenSize: hiddenSize,
        lr: 0.005, batchSize: 32,
        data: data,
        sched: sched,
        denoiser: makeDenoiser(hiddenSize),
        running: false, step: 0, losses: [],
        fwdT: 0,
        sampling: false, sampleTraj: [], sampleFrame: 0, sampleAnimId: null
    };
}
var SCALE = 160;
var OFFSET_X = 200;
var OFFSET_Y = 200;
var CANVAS_W = 400;
var CANVAS_H = 400;
var LOSS_W = 400;
var LOSS_H = 120;
function ptToCanvas(p) {
    return [p[0] * SCALE + OFFSET_X, -p[1] * SCALE + OFFSET_Y];
}
function drawMain(canvas, st) {
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = '#0d0d1f';
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.strokeStyle = '#1a1a3a';
    ctx.lineWidth = 1;
    for (var g = -1; g <= 1; g += 0.5) {
        ctx.beginPath();
        var _a = ptToCanvas([g, 0]), gx = _a[0], gy = _a[1];
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, CANVAS_H);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(CANVAS_W, gy);
        ctx.stroke();
    }
    if (st.sampling && st.sampleTraj.length > 0) {
        var frame = Math.min(st.sampleFrame, st.sampleTraj.length - 1);
        var pts = st.sampleTraj[frame];
        var progress = frame / (st.sampleTraj.length - 1);
        var r = Math.round(100 + 155 * (1 - progress));
        var g = Math.round(200 * progress);
        var b = Math.round(255 * progress);
        var color_1 = "rgba(".concat(r, ",").concat(g, ",").concat(b, ",0.75)");
        pts.forEach(function (p) {
            var _a = ptToCanvas(p), cx = _a[0], cy = _a[1];
            ctx.beginPath();
            ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
            ctx.fillStyle = color_1;
            ctx.fill();
        });
        st.data.forEach(function (p) {
            var _a = ptToCanvas(p), cx = _a[0], cy = _a[1];
            ctx.beginPath();
            ctx.arc(cx, cy, 2, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(100,220,100,0.18)';
            ctx.fill();
        });
    }
    else {
        var ab_1 = st.sched.alphaBar[st.fwdT];
        var sqrtAb_1 = Math.sqrt(ab_1);
        var sqrtOneMinusAb_1 = Math.sqrt(1 - ab_1);
        st.data.forEach(function (x0) {
            var xt = [
                sqrtAb_1 * x0[0] + sqrtOneMinusAb_1 * randn(),
                sqrtAb_1 * x0[1] + sqrtOneMinusAb_1 * randn(),
            ];
            var _a = ptToCanvas(xt), cx = _a[0], cy = _a[1];
            var noiseLevel = 1 - ab_1;
            var rr = Math.round(100 + 155 * noiseLevel);
            var gb = Math.round(220 * (1 - noiseLevel));
            ctx.beginPath();
            ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
            ctx.fillStyle = "rgba(".concat(rr, ",").concat(gb, ",").concat(gb, ",0.7)");
            ctx.fill();
        });
    }
}
function drawLoss(canvas, losses) {
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, LOSS_W, LOSS_H);
    ctx.fillStyle = '#0d0d1f';
    ctx.fillRect(0, 0, LOSS_W, LOSS_H);
    if (losses.length < 2)
        return;
    var maxLoss = Math.max.apply(Math, losses.slice(0, 50));
    var minLoss = Math.min.apply(Math, losses);
    var range = Math.max(maxLoss - minLoss, 1e-6);
    var pad = 20;
    var w = LOSS_W - 2 * pad;
    var h = LOSS_H - 2 * pad;
    ctx.strokeStyle = '#2a2a4a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, pad + h);
    ctx.lineTo(pad + w, pad + h);
    ctx.stroke();
    ctx.strokeStyle = '#a78bfa';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    var step = Math.max(1, Math.floor(losses.length / w));
    losses.forEach(function (l, i) {
        var x = pad + (i / (losses.length - 1)) * w;
        var y = pad + h - ((l - minLoss) / range) * h;
        if (i === 0)
            ctx.moveTo(x, y);
        else
            ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = '#6b7280';
    ctx.font = '10px monospace';
    ctx.fillText("loss: ".concat(losses[losses.length - 1].toFixed(4)), pad + 4, pad + 12);
}
function main() {
    state = initState();
    var mainCanvas = document.getElementById('mainCanvas');
    var lossCanvas = document.getElementById('lossCanvas');
    mainCanvas.width = CANVAS_W;
    mainCanvas.height = CANVAS_H;
    lossCanvas.width = LOSS_W;
    lossCanvas.height = LOSS_H;
    var distSelect = document.getElementById('distSelect');
    var lrInput = document.getElementById('lrInput');
    var lrVal = document.getElementById('lrVal');
    var hiddenInput = document.getElementById('hiddenInput');
    var hiddenVal = document.getElementById('hiddenVal');
    var tInput = document.getElementById('tInput');
    var tVal = document.getElementById('tVal');
    var fwdSlider = document.getElementById('fwdSlider');
    var fwdTVal = document.getElementById('fwdTVal');
    var playPause = document.getElementById('playPause');
    var stepBtn = document.getElementById('stepBtn');
    var resetBtn = document.getElementById('resetBtn');
    var sampleBtn = document.getElementById('sampleBtn');
    var stepCount = document.getElementById('stepCount');
    var lossVal = document.getElementById('lossVal');
    var sampleStatus = document.getElementById('sampleStatus');
    function syncFwdSlider() {
        fwdSlider.max = String(state.T - 1);
        fwdSlider.value = String(state.fwdT);
        fwdTVal.textContent = String(state.fwdT);
    }
    function reset() {
        state.running = false;
        playPause.textContent = 'Play';
        state.step = 0;
        state.losses = [];
        state.data = sampleDist(state.dist, NUM_DATA);
        state.sched = makeSchedule(state.T);
        state.denoiser = makeDenoiser(state.hiddenSize);
        state.fwdT = 0;
        state.sampling = false;
        state.sampleTraj = [];
        state.sampleFrame = 0;
        if (state.sampleAnimId !== null) {
            cancelAnimationFrame(state.sampleAnimId);
            state.sampleAnimId = null;
        }
        syncFwdSlider();
        stepCount.textContent = '0';
        lossVal.textContent = '—';
        sampleStatus.textContent = '';
        draw();
    }
    function draw() {
        drawMain(mainCanvas, state);
        drawLoss(lossCanvas, state.losses);
    }
    function doTrainStep() {
        var loss = trainStep(state.denoiser, state.data, state.sched, state.lr, state.batchSize);
        state.step++;
        state.losses.push(loss);
        if (state.losses.length > 500)
            state.losses.splice(0, 1);
        stepCount.textContent = String(state.step);
        lossVal.textContent = loss.toFixed(5);
        if (!state.sampling)
            draw();
    }
    var trainRafId = null;
    var lastTrainTime = 0;
    var TRAIN_INTERVAL = 50;
    function trainLoop(now) {
        if (!state.running)
            return;
        var elapsed = now - lastTrainTime;
        var stepsPerFrame = 5;
        for (var i = 0; i < stepsPerFrame; i++) {
            doTrainStep();
        }
        if (elapsed > TRAIN_INTERVAL) {
            draw();
            lastTrainTime = now;
        }
        trainRafId = requestAnimationFrame(trainLoop);
    }
    function startSampleAnim() {
        if (state.sampleAnimId !== null) {
            cancelAnimationFrame(state.sampleAnimId);
            state.sampleAnimId = null;
        }
        state.sampling = true;
        state.sampleFrame = 0;
        sampleStatus.textContent = 'Denoising…';
        function animFrame() {
            if (!state.sampling)
                return;
            state.sampleFrame++;
            if (state.sampleFrame >= state.sampleTraj.length) {
                state.sampleFrame = state.sampleTraj.length - 1;
                sampleStatus.textContent = "Done (".concat(state.sampleTraj.length - 1, " reverse steps)");
                draw();
                state.sampleAnimId = null;
                return;
            }
            draw();
            state.sampleAnimId = requestAnimationFrame(animFrame);
        }
        state.sampleAnimId = requestAnimationFrame(animFrame);
    }
    playPause.addEventListener('click', function () {
        state.running = !state.running;
        playPause.textContent = state.running ? 'Pause' : 'Play';
        if (state.running) {
            lastTrainTime = 0;
            trainRafId = requestAnimationFrame(trainLoop);
        }
        else {
            if (trainRafId !== null) {
                cancelAnimationFrame(trainRafId);
                trainRafId = null;
            }
            draw();
        }
    });
    stepBtn.addEventListener('click', function () {
        doTrainStep();
        draw();
    });
    resetBtn.addEventListener('click', reset);
    sampleBtn.addEventListener('click', function () {
        if (state.step === 0) {
            sampleStatus.textContent = 'Train first!';
            return;
        }
        var traj = reverseSample(state.denoiser, state.sched, SAMPLE_N);
        state.sampleTraj = traj;
        state.sampleFrame = 0;
        startSampleAnim();
    });
    distSelect.addEventListener('change', function () {
        state.dist = distSelect.value;
        reset();
    });
    lrInput.addEventListener('input', function () {
        state.lr = parseFloat(lrInput.value);
        lrVal.textContent = state.lr.toFixed(4);
    });
    hiddenInput.addEventListener('change', function () {
        state.hiddenSize = parseInt(hiddenInput.value, 10);
        hiddenVal.textContent = String(state.hiddenSize);
        reset();
    });
    tInput.addEventListener('change', function () {
        state.T = parseInt(tInput.value, 10);
        tVal.textContent = String(state.T);
        reset();
    });
    fwdSlider.addEventListener('input', function () {
        state.fwdT = parseInt(fwdSlider.value, 10);
        fwdTVal.textContent = String(state.fwdT);
        if (!state.sampling)
            draw();
    });
    syncFwdSlider();
    draw();
}
document.addEventListener('DOMContentLoaded', main);

},{}]},{},[1]);
