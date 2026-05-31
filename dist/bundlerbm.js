(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
function sampleBernoulli(p) {
    return Math.random() < p ? 1 : 0;
}
function createRBM(nV, nH) {
    var W = new Float64Array(nV * nH);
    var bV = new Float64Array(nV);
    var bH = new Float64Array(nH);
    for (var i = 0; i < W.length; i++) {
        W[i] = (Math.random() * 2 - 1) * 0.05;
    }
    return { nV: nV, nH: nH, W: W, bV: bV, bH: bH };
}
function hiddenProbs(rbm, v, pH) {
    for (var j = 0; j < rbm.nH; j++) {
        var act = rbm.bH[j];
        for (var i = 0; i < rbm.nV; i++) {
            act += v[i] * rbm.W[i * rbm.nH + j];
        }
        pH[j] = sigmoid(act);
    }
}
function visibleProbs(rbm, h, pV) {
    for (var i = 0; i < rbm.nV; i++) {
        var act = rbm.bV[i];
        for (var j = 0; j < rbm.nH; j++) {
            act += h[j] * rbm.W[i * rbm.nH + j];
        }
        pV[i] = sigmoid(act);
    }
}
function sampleH(rbm, v, pH, hSample) {
    hiddenProbs(rbm, v, pH);
    for (var j = 0; j < rbm.nH; j++) {
        hSample[j] = sampleBernoulli(pH[j]);
    }
}
function sampleV(rbm, h, pV, vSample) {
    visibleProbs(rbm, h, pV);
    for (var i = 0; i < rbm.nV; i++) {
        vSample[i] = sampleBernoulli(pV[i]);
    }
}
function cdStep(rbm, v0, k, lr) {
    var nV = rbm.nV, nH = rbm.nH;
    var pH0 = new Float64Array(nH);
    var h0 = new Float64Array(nH);
    var pV = new Float64Array(nV);
    var vK = new Float64Array(nV);
    var pHK = new Float64Array(nH);
    sampleH(rbm, v0, pH0, h0);
    var hCur = h0;
    vK.set(v0);
    for (var step = 0; step < k; step++) {
        sampleV(rbm, hCur, pV, vK);
        if (step < k - 1) {
            var hNext = new Float64Array(nH);
            sampleH(rbm, vK, pHK, hNext);
            hCur = hNext;
        }
        else {
            hiddenProbs(rbm, vK, pHK);
        }
    }
    for (var i = 0; i < nV; i++) {
        for (var j = 0; j < nH; j++) {
            rbm.W[i * nH + j] += lr * (v0[i] * pH0[j] - vK[i] * pHK[j]);
        }
        rbm.bV[i] += lr * (v0[i] - vK[i]);
    }
    for (var j = 0; j < nH; j++) {
        rbm.bH[j] += lr * (pH0[j] - pHK[j]);
    }
    visibleProbs(rbm, h0, pV);
    var err = 0;
    for (var i = 0; i < nV; i++) {
        var d = v0[i] - pV[i];
        err += d * d;
    }
    return err / nV;
}
function gibbsSample(rbm, vStart, steps) {
    var nV = rbm.nV, nH = rbm.nH;
    var pV = new Float64Array(nV);
    var pH = new Float64Array(nH);
    var h = new Float64Array(nH);
    var v = new Float64Array(nV);
    v.set(vStart);
    for (var s = 0; s < steps; s++) {
        sampleH(rbm, v, pH, h);
        sampleV(rbm, h, pV, v);
    }
    visibleProbs(rbm, h, pV);
    return pV;
}
function reconstruct(rbm, v) {
    var pH = new Float64Array(rbm.nH);
    var pV = new Float64Array(rbm.nV);
    hiddenProbs(rbm, v, pH);
    visibleProbs(rbm, pH, pV);
    return pV;
}
var GLYPH_SIZE = 6;
var GLYPHS = [
    [0, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 1, 1,
        1, 0, 0, 1, 0, 1,
        1, 0, 1, 0, 0, 1,
        1, 1, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0,
        0, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 1,
        0, 0, 1, 1, 0, 0,
        0, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 1,
        0, 0, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 0,
        0, 0, 1, 0, 1, 0,
        0, 1, 0, 0, 1, 0,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0,
        1, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 0,
        0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0,
        0, 1, 0, 0, 1, 0,
        1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1,
        0, 1, 0, 0, 1, 0,
        0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1,
        1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 1,
        0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0],
];
var GLYPH_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'Cross', 'Diag', 'Checker', 'Diamond', 'H', 'U', 'Z', 'T'
];
var rbm;
var nHidden = 16;
var cdK = 1;
var learningRate = 0.02;
var epochInterval = null;
var epoch = 0;
var errorHistory = [];
var selectedExample = 0;
var dreamSteps = 100;
var dataset = GLYPHS.map(function (g) {
    var v = new Float64Array(GLYPH_SIZE * GLYPH_SIZE);
    for (var i = 0; i < g.length; i++)
        v[i] = g[i];
    return v;
});
function initRBM() {
    rbm = createRBM(GLYPH_SIZE * GLYPH_SIZE, nHidden);
    epoch = 0;
    errorHistory = [];
}
function runEpoch() {
    var idx = [];
    for (var i = 0; i < dataset.length; i++)
        idx.push(i);
    for (var i = idx.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
    var totalErr = 0;
    idx.forEach(function (i) {
        totalErr += cdStep(rbm, dataset[i], cdK, learningRate);
    });
    epoch++;
    errorHistory.push(totalErr / dataset.length);
}
function renderBinaryImage(canvas, data, cols, rows, cellSize, colorOn, colorOff) {
    if (colorOn === void 0) { colorOn = '#c084fc'; }
    if (colorOff === void 0) { colorOff = '#1a1a38'; }
    canvas.width = cols * cellSize;
    canvas.height = rows * cellSize;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
            var val = data[r * cols + c];
            ctx.fillStyle = lerpColor(colorOff, colorOn, val);
            ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
        }
    }
}
function lerpColor(hex1, hex2, t) {
    var r1 = parseInt(hex1.slice(1, 3), 16), g1 = parseInt(hex1.slice(3, 5), 16), b1 = parseInt(hex1.slice(5, 7), 16);
    var r2 = parseInt(hex2.slice(1, 3), 16), g2 = parseInt(hex2.slice(3, 5), 16), b2 = parseInt(hex2.slice(5, 7), 16);
    var r = Math.round(r1 + (r2 - r1) * t);
    var g = Math.round(g1 + (g2 - g1) * t);
    var b = Math.round(b1 + (b2 - b1) * t);
    return "rgb(".concat(r, ",").concat(g, ",").concat(b, ")");
}
function renderReceptiveFields(canvas) {
    var nH = rbm.nH;
    var nV = rbm.nV;
    var gs = GLYPH_SIZE;
    var cell = 3;
    var gap = 2;
    var cols = Math.ceil(Math.sqrt(nH));
    var rows = Math.ceil(nH / cols);
    var fw = gs * cell;
    var fh = gs * cell;
    canvas.width = cols * (fw + gap) + gap;
    canvas.height = rows * (fh + gap) + gap;
    var ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    var wMin = Infinity, wMax = -Infinity;
    for (var i = 0; i < rbm.W.length; i++) {
        if (rbm.W[i] < wMin)
            wMin = rbm.W[i];
        if (rbm.W[i] > wMax)
            wMax = rbm.W[i];
    }
    var wRange = wMax - wMin || 1;
    for (var j = 0; j < nH; j++) {
        var col = j % cols;
        var row = Math.floor(j / cols);
        var ox = gap + col * (fw + gap);
        var oy = gap + row * (fh + gap);
        for (var i = 0; i < nV; i++) {
            var t = (rbm.W[i * nH + j] - wMin) / wRange;
            ctx.fillStyle = lerpColor('#1a0a3a', '#f0abfc', t);
            var vi = i % gs;
            var vr = Math.floor(i / gs);
            ctx.fillRect(ox + vi * cell, oy + vr * cell, cell, cell);
        }
    }
}
function renderErrorCurve(canvas) {
    var W = canvas.width, H = canvas.height;
    var ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, W, H);
    if (errorHistory.length < 2)
        return;
    var maxErr = Math.max.apply(Math, errorHistory) || 1;
    var minErr = Math.min.apply(Math, errorHistory);
    ctx.strokeStyle = '#c084fc';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    errorHistory.forEach(function (e, i) {
        var x = (i / (errorHistory.length - 1)) * (W - 20) + 10;
        var y = H - 10 - ((e - minErr) / (maxErr - minErr || 1)) * (H - 20);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = '#5a5a8a';
    ctx.font = '10px monospace';
    ctx.fillText("err: ".concat(errorHistory[errorHistory.length - 1].toFixed(4)), 12, 14);
    ctx.fillText("epoch ".concat(epoch), W - 70, 14);
}
function renderDreams(canvas, count) {
    var gs = GLYPH_SIZE;
    var cell = 6;
    var gap = 4;
    var perRow = count;
    canvas.width = perRow * (gs * cell + gap) + gap;
    canvas.height = gs * cell + gap * 2;
    var ctx = canvas.getContext('2d');
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (var s = 0; s < count; s++) {
        var vStart = new Float64Array(gs * gs);
        for (var i = 0; i < vStart.length; i++)
            vStart[i] = Math.random() < 0.5 ? 1 : 0;
        var pV = gibbsSample(rbm, vStart, dreamSteps);
        var ox = gap + s * (gs * cell + gap);
        var oy = gap;
        for (var r = 0; r < gs; r++) {
            for (var c = 0; c < gs; c++) {
                var t = pV[r * gs + c];
                ctx.fillStyle = lerpColor('#1a1a38', '#c084fc', t);
                ctx.fillRect(ox + c * cell, oy + r * cell, cell, cell);
            }
        }
    }
}
function buildUI() {
    document.body.innerHTML = "\n<h1>Restricted Boltzmann Machine</h1>\n<p class=\"subtitle\">\n  Binary RBM trained with Contrastive Divergence (CD-k) on 6\u00D76 binary glyphs.\n  Watch the model learn features, reconstruct inputs, and generate new samples.\n</p>\n\n<div class=\"main-row\">\n\n  <!-- ===== LEFT: visualisations ===== -->\n  <div class=\"left-col\">\n\n    <!-- Receptive fields -->\n    <div class=\"panel\">\n      <div class=\"panel-title\">Receptive Fields \u2014 learned weights per hidden unit</div>\n      <canvas id=\"rfCanvas\"></canvas>\n    </div>\n\n    <!-- Input / Reconstruction -->\n    <div class=\"panel\">\n      <div class=\"panel-title\">Input &amp; Reconstruction (one forward\u2013backward pass)</div>\n      <div class=\"recon-row\">\n        <div class=\"recon-block\">\n          <div class=\"recon-label\">Input</div>\n          <canvas id=\"inputCanvas\"></canvas>\n        </div>\n        <div class=\"recon-block\">\n          <div class=\"recon-label\">Reconstruction</div>\n          <canvas id=\"reconCanvas\"></canvas>\n        </div>\n      </div>\n    </div>\n\n    <!-- Dreams -->\n    <div class=\"panel\">\n      <div class=\"panel-title\">Dreams \u2014 Gibbs samples from random noise</div>\n      <canvas id=\"dreamCanvas\"></canvas>\n    </div>\n\n    <!-- Error curve -->\n    <div class=\"panel\">\n      <div class=\"panel-title\">Reconstruction Error vs Epoch</div>\n      <canvas id=\"errorCanvas\" width=\"480\" height=\"110\"></canvas>\n    </div>\n\n  </div>\n\n  <!-- ===== RIGHT: controls ===== -->\n  <div class=\"controls-col\">\n\n    <div class=\"section-title\">Model</div>\n\n    <div class=\"ctrl-group\">\n      <label>Hidden Units</label>\n      <div class=\"range-row\">\n        <input type=\"range\" id=\"nHiddenSlider\" min=\"4\" max=\"64\" step=\"4\" value=\"16\" />\n        <span id=\"nHiddenVal\">16</span>\n      </div>\n    </div>\n\n    <div class=\"ctrl-group\">\n      <label>Learning Rate</label>\n      <div class=\"range-row\">\n        <input type=\"range\" id=\"lrSlider\" min=\"-4\" max=\"-1\" step=\"0.25\" value=\"-2\" />\n        <span id=\"lrVal\">0.0100</span>\n      </div>\n    </div>\n\n    <div class=\"ctrl-group\">\n      <label>CD-k Steps</label>\n      <div class=\"range-row\">\n        <input type=\"range\" id=\"cdkSlider\" min=\"1\" max=\"10\" step=\"1\" value=\"1\" />\n        <span id=\"cdkVal\">1</span>\n      </div>\n    </div>\n\n    <div class=\"section-title\">Training</div>\n\n    <div class=\"btn-row\">\n      <button id=\"playBtn\">\u25B6 Play</button>\n      <button id=\"stepBtn\">Step</button>\n    </div>\n    <div class=\"btn-row\">\n      <button id=\"resetBtn\">Reset</button>\n    </div>\n\n    <div class=\"section-title\">Example</div>\n\n    <div class=\"ctrl-group\">\n      <label>Input Glyph</label>\n      <select id=\"exampleSelect\"></select>\n    </div>\n\n    <div class=\"section-title\">Dream</div>\n\n    <div class=\"ctrl-group\">\n      <label>Gibbs Steps</label>\n      <div class=\"range-row\">\n        <input type=\"range\" id=\"dreamSlider\" min=\"10\" max=\"1000\" step=\"10\" value=\"100\" />\n        <span id=\"dreamVal\">100</span>\n      </div>\n    </div>\n\n    <div class=\"btn-row\">\n      <button id=\"dreamBtn\">Dream \u2726</button>\n    </div>\n\n    <div class=\"section-title\">Status</div>\n    <div id=\"status\">Ready. Press \u25B6 Play to train.</div>\n\n    <div class=\"section-title\">Theory</div>\n    <div class=\"theory-box\">\n      <strong>CD-k:</strong> positive phase <code>\u27E8v h\u27E9_data</code> minus negative phase <code>\u27E8v h\u27E9_k</code><br/>\n      <strong>Update:</strong> <code>\u0394W = lr \u00D7 (p_h0 v0\u1D40 \u2212 p_hk vk\u1D40)</code><br/>\n      <strong>Dream:</strong> run Gibbs chain from noise \u2192 model's prior.\n    </div>\n\n  </div>\n</div>\n";
    var sel = document.getElementById('exampleSelect');
    GLYPH_NAMES.forEach(function (name, i) {
        var opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = name;
        sel.appendChild(opt);
    });
    wireSlider('nHiddenSlider', 'nHiddenVal', function (v) {
        nHidden = v;
    }, function (v) { return String(v); });
    wireSlider('lrSlider', 'lrVal', function (v) {
        learningRate = Math.pow(10, v);
    }, function (v) { return Math.pow(10, v).toFixed(4); });
    wireSlider('cdkSlider', 'cdkVal', function (v) {
        cdK = v;
    }, function (v) { return String(v); });
    wireSlider('dreamSlider', 'dreamVal', function (v) {
        dreamSteps = v;
    }, function (v) { return String(v); });
    sel.addEventListener('change', function () {
        selectedExample = parseInt(sel.value);
        renderAll();
    });
    document.getElementById('playBtn').addEventListener('click', togglePlay);
    document.getElementById('stepBtn').addEventListener('click', function () { rbmDoStep(); renderAll(); });
    document.getElementById('resetBtn').addEventListener('click', function () {
        stopPlay();
        nHidden = parseInt(document.getElementById('nHiddenSlider').value);
        initRBM();
        renderAll();
        setStatus('Model reset.');
    });
    document.getElementById('dreamBtn').addEventListener('click', function () {
        var dc = document.getElementById('dreamCanvas');
        renderDreams(dc, 6);
    });
}
function wireSlider(sliderId, valId, onValue, fmt) {
    var slider = document.getElementById(sliderId);
    var valEl = document.getElementById(valId);
    var update = function () {
        var v = parseFloat(slider.value);
        onValue(v);
        valEl.textContent = fmt(v);
    };
    slider.addEventListener('input', update);
    update();
}
function setStatus(msg) {
    var el = document.getElementById('status');
    if (el)
        el.textContent = msg;
}
var playing = false;
function togglePlay() {
    playing = !playing;
    var btn = document.getElementById('playBtn');
    if (playing) {
        btn.textContent = '⏸ Pause';
        scheduleEpoch();
    }
    else {
        btn.textContent = '▶ Play';
        if (epochInterval !== null) {
            cancelAnimationFrame(epochInterval);
            epochInterval = null;
        }
    }
}
function stopPlay() {
    playing = false;
    var btn = document.getElementById('playBtn');
    if (btn)
        btn.textContent = '▶ Play';
    if (epochInterval !== null) {
        cancelAnimationFrame(epochInterval);
        epochInterval = null;
    }
}
function scheduleEpoch() {
    epochInterval = requestAnimationFrame(function () {
        if (!playing)
            return;
        rbmDoStep();
        renderAll();
        scheduleEpoch();
    });
}
function rbmDoStep() {
    runEpoch();
    var lastErr = errorHistory[errorHistory.length - 1];
    setStatus("Epoch ".concat(epoch, " \u2014 avg reconstruction error: ").concat(lastErr.toFixed(5)));
}
function renderAll() {
    var rfCanvas = document.getElementById('rfCanvas');
    var inputCanvas = document.getElementById('inputCanvas');
    var reconCanvas = document.getElementById('reconCanvas');
    var errorCanvas = document.getElementById('errorCanvas');
    renderReceptiveFields(rfCanvas);
    var v0 = dataset[selectedExample];
    renderBinaryImage(inputCanvas, v0, GLYPH_SIZE, GLYPH_SIZE, 10);
    var recon = reconstruct(rbm, v0);
    renderBinaryImage(reconCanvas, recon, GLYPH_SIZE, GLYPH_SIZE, 10);
    renderErrorCurve(errorCanvas);
    var dc = document.getElementById('dreamCanvas');
    renderDreams(dc, 6);
}
function injectStyles() {
    var style = document.createElement('style');
    style.textContent = "\n*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }\n\nbody {\n  background: #0a0a1a;\n  color: #d0d0e8;\n  font-family: 'Segoe UI', system-ui, sans-serif;\n  font-size: 14px;\n  min-height: 100vh;\n  display: flex;\n  flex-direction: column;\n  align-items: center;\n  padding: 20px 12px 48px;\n}\n\nh1 {\n  font-size: 1.5rem;\n  font-weight: 700;\n  color: #c084fc;\n  margin-bottom: 4px;\n  letter-spacing: 0.04em;\n}\n\n.subtitle {\n  color: #6b7280;\n  font-size: 0.82rem;\n  margin-bottom: 20px;\n  text-align: center;\n  max-width: 640px;\n}\n\n.main-row {\n  display: flex;\n  gap: 16px;\n  align-items: flex-start;\n  flex-wrap: wrap;\n  justify-content: center;\n  width: 100%;\n  max-width: 1100px;\n}\n\n.left-col {\n  display: flex;\n  flex-direction: column;\n  gap: 12px;\n}\n\n.panel {\n  background: #0d0d1f;\n  border: 1px solid #2a2a4a;\n  border-radius: 10px;\n  padding: 12px 14px;\n}\n\n.panel-title {\n  font-size: 0.70rem;\n  color: #5a5a8a;\n  text-transform: uppercase;\n  letter-spacing: 0.08em;\n  margin-bottom: 8px;\n}\n\n.recon-row {\n  display: flex;\n  gap: 20px;\n  align-items: flex-start;\n}\n\n.recon-block {\n  display: flex;\n  flex-direction: column;\n  align-items: center;\n  gap: 4px;\n}\n\n.recon-label {\n  font-size: 0.70rem;\n  color: #7070a0;\n  text-transform: uppercase;\n  letter-spacing: 0.06em;\n}\n\ncanvas { display: block; image-rendering: pixelated; border-radius: 4px; }\n\n.controls-col {\n  background: #10102a;\n  border: 1px solid #2a2a4a;\n  border-radius: 10px;\n  padding: 18px;\n  width: 240px;\n  min-width: 200px;\n  display: flex;\n  flex-direction: column;\n  gap: 12px;\n}\n\n.section-title {\n  font-size: 0.70rem;\n  color: #5a5a8a;\n  text-transform: uppercase;\n  letter-spacing: 0.08em;\n  border-bottom: 1px solid #22224a;\n  padding-bottom: 4px;\n}\n\n.ctrl-group {\n  display: flex;\n  flex-direction: column;\n  gap: 4px;\n}\n\n.ctrl-group label {\n  font-size: 0.76rem;\n  color: #8888bb;\n  text-transform: uppercase;\n  letter-spacing: 0.06em;\n}\n\nselect, input[type=range] {\n  width: 100%;\n  background: #1a1a38;\n  border: 1px solid #3a3a5c;\n  border-radius: 5px;\n  color: #d0d0e8;\n  padding: 4px 6px;\n  font-size: 0.85rem;\n  cursor: pointer;\n}\n\ninput[type=range] {\n  padding: 0;\n  accent-color: #c084fc;\n  height: 18px;\n}\n\n.range-row {\n  display: flex;\n  justify-content: space-between;\n  align-items: center;\n  gap: 6px;\n}\n\n.range-row span {\n  min-width: 40px;\n  text-align: right;\n  font-size: 0.82rem;\n  color: #c0c0e0;\n  font-variant-numeric: tabular-nums;\n}\n\n.btn-row {\n  display: flex;\n  gap: 5px;\n  flex-wrap: wrap;\n}\n\nbutton {\n  flex: 1;\n  padding: 6px 4px;\n  border: none;\n  border-radius: 6px;\n  cursor: pointer;\n  font-size: 0.78rem;\n  font-weight: 600;\n  transition: opacity 0.12s;\n  white-space: nowrap;\n}\n\nbutton:hover { opacity: 0.82; }\n\n#playBtn  { background: #7c3aed; color: #fff; }\n#stepBtn  { background: #1e3a5f; color: #7dd3fc; }\n#resetBtn { background: #3b1f1f; color: #fca5a5; }\n#dreamBtn { background: #166534; color: #86efac; flex: 2; }\n\n#status {\n  font-size: 0.78rem;\n  color: #8888cc;\n  background: #0d0d22;\n  border: 1px solid #2a2a44;\n  border-radius: 6px;\n  padding: 5px 10px;\n  min-height: 28px;\n  word-break: break-word;\n}\n\n.theory-box {\n  background: #0d0d22;\n  border-left: 3px solid #7c3aed;\n  border-radius: 0 6px 6px 0;\n  padding: 8px 12px;\n  font-size: 0.75rem;\n  color: #7070a0;\n  line-height: 1.55;\n}\n\n.theory-box code {\n  color: #c4b5fd;\n  font-family: monospace;\n}\n  ";
    document.head.appendChild(style);
}
window.addEventListener('DOMContentLoaded', function () {
    injectStyles();
    initRBM();
    buildUI();
    renderAll();
});

},{}]},{},[1]);
