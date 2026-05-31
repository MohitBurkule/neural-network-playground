(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function cadd(a, b) { return { re: a.re + b.re, im: a.im + b.im }; }
function cmul(a, b) {
    return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}
function cexp(theta) { return { re: Math.cos(theta), im: Math.sin(theta) }; }
function cabs(a) { return Math.sqrt(a.re * a.re + a.im * a.im); }
function carg(a) { return Math.atan2(a.im, a.re); }
function dft(signal) {
    var N = signal.length;
    var result = [];
    for (var k = 0; k < N; k++) {
        var re = 0, im = 0;
        for (var n = 0; n < N; n++) {
            var phi = (2 * Math.PI * k * n) / N;
            var c = cexp(-phi);
            var p = cmul(signal[n], c);
            re += p.re;
            im += p.im;
        }
        re /= N;
        im /= N;
        result.push({ freq: k, amp: Math.sqrt(re * re + im * im), phase: Math.atan2(im, re), re: re, im: im });
    }
    return result;
}
function sampleSquare(N) {
    var pts = [];
    var sides = 4;
    for (var i = 0; i < N; i++) {
        var t = (i / N) * sides;
        var side = Math.floor(t);
        var frac = t - side;
        var x = 0, y = 0;
        var s = 150;
        if (side === 0) {
            x = -s + 2 * s * frac;
            y = -s;
        }
        else if (side === 1) {
            x = s;
            y = -s + 2 * s * frac;
        }
        else if (side === 2) {
            x = s - 2 * s * frac;
            y = s;
        }
        else {
            x = -s;
            y = s - 2 * s * frac;
        }
        pts.push({ re: x, im: y });
    }
    return pts;
}
function sampleStar(N, points) {
    if (points === void 0) { points = 5; }
    var pts = [];
    var outer = 140, inner = 55;
    for (var i = 0; i < N; i++) {
        var t = (i / N) * 2 * Math.PI;
        var totalPeaks = points * 2;
        var sector = (t / (2 * Math.PI)) * totalPeaks;
        var frac = sector - Math.floor(sector);
        var r1 = Math.floor(sector) % 2 === 0 ? outer : inner;
        var r2 = Math.floor(sector) % 2 === 0 ? inner : outer;
        var r = r1 + (r2 - r1) * frac;
        pts.push({ re: r * Math.cos(t - Math.PI / 2), im: r * Math.sin(t - Math.PI / 2) });
    }
    return pts;
}
function sampleHeart(N) {
    var pts = [];
    for (var i = 0; i < N; i++) {
        var t = (i / N) * 2 * Math.PI;
        var x = 130 * (16 * Math.pow(Math.sin(t), 3)) / 16;
        var y = -130 * (13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) / 16;
        pts.push({ re: x, im: y });
    }
    return pts;
}
function sampleFigure8(N) {
    var pts = [];
    for (var i = 0; i < N; i++) {
        var t = (i / N) * 2 * Math.PI;
        pts.push({ re: 150 * Math.sin(t), im: 100 * Math.sin(2 * t) });
    }
    return pts;
}
var state = {
    mode: 'epicycles',
    preset: 'square',
    drawing: false,
    drawnPath: [],
    phasors: [],
    numTerms: 10,
    epicycleTime: 0,
    epicycleSpeed: 1,
    tracePoints: [],
    animating: false,
    waveform: 'square',
    numHarmonics: 5,
    showSpectrum: true,
    signal1dTime: 0,
    signal1dSpeed: 1
};
var epicyclesCanvas = document.getElementById('epicycles-canvas');
var signal1dCanvas = document.getElementById('signal1d-canvas');
var spectrumCanvas = document.getElementById('spectrum-canvas');
var ctxE = epicyclesCanvas.getContext('2d');
var ctx1d = signal1dCanvas.getContext('2d');
var ctxS = spectrumCanvas.getContext('2d');
var C = {
    bg: '#0f0f1a',
    grid: '#1e1e3a',
    circle: 'rgba(99,102,241,0.35)',
    circleStroke: 'rgba(99,102,241,0.8)',
    arm: '#818cf8',
    trace: '#f472b6',
    target: 'rgba(251,191,36,0.5)',
    approx: '#34d399',
    spectrum: '#60a5fa',
    text: '#e2e8f0',
    accent: '#f472b6'
};
function resizeCanvas(c) {
    c.width = c.offsetWidth;
    c.height = c.offsetHeight;
}
function clearCanvas(ctx, canvas) {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
function drawGrid(ctx, canvas) {
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    var step = 40;
    for (var x = 0; x < canvas.width; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    for (var y = 0; y < canvas.height; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
}
function preparePhasors(path) {
    if (path.length < 4)
        return;
    var N = Math.min(512, path.length);
    var sampled = [];
    for (var i = 0; i < N; i++) {
        var idx = Math.floor((i / N) * path.length);
        sampled.push(path[idx]);
    }
    state.phasors = dft(sampled);
    state.phasors.sort(function (a, b) { return b.amp - a.amp; });
    state.numTerms = Math.min(state.numTerms, state.phasors.length);
    state.tracePoints = [];
    state.epicycleTime = 0;
    var slider = document.getElementById('terms-slider');
    if (slider) {
        slider.max = String(state.phasors.length);
        slider.value = String(state.numTerms);
        var label = document.getElementById('terms-label');
        if (label)
            label.textContent = String(state.numTerms);
    }
}
function loadPreset(preset) {
    if (preset === 'draw')
        return;
    var N = 256;
    if (preset === 'square')
        state.drawnPath = sampleSquare(N);
    else if (preset === 'star')
        state.drawnPath = sampleStar(N);
    else if (preset === 'heart')
        state.drawnPath = sampleHeart(N);
    else if (preset === 'figure8')
        state.drawnPath = sampleFigure8(N);
    preparePhasors(state.drawnPath);
}
function drawEpicycles(t) {
    var canvas = epicyclesCanvas;
    var ctx = ctxE;
    clearCanvas(ctx, canvas);
    drawGrid(ctx, canvas);
    var cx = canvas.width / 2;
    var cy = canvas.height / 2;
    var terms = Math.max(1, state.numTerms);
    var usedPhasors = state.phasors.slice(0, terms);
    var x = cx, y = cy;
    usedPhasors.forEach(function (p) {
        var prevX = x, prevY = y;
        var angle = 2 * Math.PI * p.freq * t / state.phasors.length + p.phase;
        var dx = p.amp * Math.cos(angle);
        var dy = p.amp * Math.sin(angle);
        var nx = prevX + dx;
        var ny = prevY + dy;
        ctx.beginPath();
        ctx.arc(prevX, prevY, p.amp, 0, 2 * Math.PI);
        ctx.strokeStyle = C.circle;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(nx, ny);
        ctx.strokeStyle = C.arm;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        x = nx;
        y = ny;
    });
    state.tracePoints.push({ x: x, y: y });
    var maxTrace = state.phasors.length;
    if (state.tracePoints.length > maxTrace)
        state.tracePoints.shift();
    if (state.tracePoints.length > 1) {
        ctx.beginPath();
        ctx.moveTo(state.tracePoints[0].x, state.tracePoints[0].y);
        for (var i = 1; i < state.tracePoints.length; i++) {
            ctx.lineTo(state.tracePoints[i].x, state.tracePoints[i].y);
        }
        ctx.strokeStyle = C.trace;
        ctx.lineWidth = 2;
        ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fillStyle = C.trace;
    ctx.fill();
}
function targetWaveform(t, wave) {
    var x = t / (2 * Math.PI);
    if (wave === 'square')
        return x % 1 < 0.5 ? 1 : -1;
    if (wave === 'sawtooth')
        return 2 * (x % 1) - 1;
    if (wave === 'triangle') {
        var u = x % 1;
        return u < 0.5 ? 4 * u - 1 : 3 - 4 * u;
    }
    return Math.sin(t) + 0.5 * Math.sin(3 * t) + 0.25 * Math.cos(5 * t);
}
function fourierApprox(t, n, wave) {
    var sum = 0;
    if (wave === 'square') {
        for (var k = 1; k <= n; k++) {
            var h = 2 * k - 1;
            sum += (4 / Math.PI) * (1 / h) * Math.sin(h * t);
        }
    }
    else if (wave === 'sawtooth') {
        for (var k = 1; k <= n; k++) {
            sum += (2 / Math.PI) * (Math.pow(-1, k + 1) / k) * Math.sin(k * t);
        }
    }
    else if (wave === 'triangle') {
        for (var k = 1; k <= n; k++) {
            var h = 2 * k - 1;
            sum += (8 / (Math.PI * Math.PI)) * (Math.pow(-1, k + 1) / (h * h)) * Math.sin(h * t);
        }
    }
    else {
        for (var k = 1; k <= n; k++) {
            sum += (1 / k) * Math.sin(k * t) * Math.cos(k * 0.1);
        }
    }
    return sum;
}
function getSpectrumCoeffs(n, wave) {
    var coeffs = [];
    if (wave === 'square') {
        for (var k = 1; k <= n; k++) {
            var h = 2 * k - 1;
            coeffs.push({ k: h, amp: (4 / Math.PI) / h });
        }
    }
    else if (wave === 'sawtooth') {
        for (var k = 1; k <= n; k++) {
            coeffs.push({ k: k, amp: (2 / Math.PI) / k });
        }
    }
    else if (wave === 'triangle') {
        for (var k = 1; k <= n; k++) {
            var h = 2 * k - 1;
            coeffs.push({ k: h, amp: (8 / (Math.PI * Math.PI)) / (h * h) });
        }
    }
    else {
        for (var k = 1; k <= n; k++) {
            coeffs.push({ k: k, amp: 1 / k });
        }
    }
    return coeffs;
}
function draw1dSignal() {
    var canvas = signal1dCanvas;
    var ctx = ctx1d;
    clearCanvas(ctx, canvas);
    drawGrid(ctx, canvas);
    var W = canvas.width, H = canvas.height;
    var cy = H / 2;
    var scaleY = H * 0.35;
    var points = 600;
    ctx.beginPath();
    for (var i = 0; i <= points; i++) {
        var t = (i / points) * 4 * Math.PI;
        var y = targetWaveform(t, state.waveform);
        var px = (i / points) * W;
        var py = cy - y * scaleY;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.strokeStyle = C.target;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    for (var i = 0; i <= points; i++) {
        var t = (i / points) * 4 * Math.PI;
        var y = fourierApprox(t, state.numHarmonics, state.waveform);
        var px = (i / points) * W;
        var py = cy - y * scaleY;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.strokeStyle = C.approx;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    var markerX = ((state.signal1dTime % (4 * Math.PI)) / (4 * Math.PI)) * W;
    ctx.beginPath();
    ctx.moveTo(markerX, 0);
    ctx.lineTo(markerX, H);
    ctx.strokeStyle = 'rgba(244,114,182,0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.font = '13px monospace';
    ctx.fillStyle = C.target;
    ctx.fillText('● target', 16, 22);
    ctx.fillStyle = C.approx;
    ctx.fillText('● approx (' + state.numHarmonics + ' harmonics)', 16, 42);
    ctx.fillStyle = C.text;
    ctx.fillText('waveform: ' + state.waveform, W - 180, 22);
}
function drawSpectrum() {
    var canvas = spectrumCanvas;
    var ctx = ctxS;
    clearCanvas(ctx, canvas);
    if (!state.showSpectrum) {
        ctx.fillStyle = C.text;
        ctx.font = '14px monospace';
        ctx.fillText('spectrum hidden', canvas.width / 2 - 60, canvas.height / 2);
        return;
    }
    var coeffs = getSpectrumCoeffs(state.numHarmonics, state.waveform);
    if (coeffs.length === 0)
        return;
    var W = canvas.width, H = canvas.height;
    var maxAmp = Math.max.apply(null, coeffs.map(function (c) { return c.amp; }));
    var padL = 40, padR = 20, padT = 20, padB = 40;
    var plotW = W - padL - padR;
    var plotH = H - padT - padB;
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();
    ctx.fillStyle = C.text;
    ctx.font = '11px monospace';
    ctx.save();
    ctx.translate(12, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('amplitude', -30, 0);
    ctx.restore();
    ctx.fillText('harmonic k', padL + plotW / 2 - 30, H - 4);
    var barW = Math.max(4, plotW / (coeffs.length * 2));
    coeffs.forEach(function (c, i) {
        var bx = padL + (i / coeffs.length) * plotW + (plotW / coeffs.length - barW) / 2;
        var bh = (c.amp / maxAmp) * plotH;
        var by = padT + plotH - bh;
        var grad = ctx.createLinearGradient(bx, by, bx, by + bh);
        grad.addColorStop(0, '#818cf8');
        grad.addColorStop(1, C.spectrum);
        ctx.fillStyle = grad;
        ctx.fillRect(bx, by, barW, bh);
        ctx.fillStyle = C.text;
        ctx.font = '10px monospace';
        ctx.fillText(String(c.k), bx + barW / 2 - 4, padT + plotH + 16);
    });
}
var lastRaf = 0;
function animate(ts) {
    var dt = Math.min((ts - lastRaf) / 1000, 0.05);
    lastRaf = ts;
    if (state.mode === 'epicycles') {
        if (state.phasors.length > 0) {
            var step = state.epicycleSpeed * 2;
            state.epicycleTime += step;
            if (state.epicycleTime >= state.phasors.length) {
                state.epicycleTime = 0;
                state.tracePoints = [];
            }
            drawEpicycles(state.epicycleTime);
        }
        else {
            clearCanvas(ctxE, epicyclesCanvas);
            drawGrid(ctxE, epicyclesCanvas);
            ctxE.fillStyle = 'rgba(148,163,184,0.6)';
            ctxE.font = '15px monospace';
            ctxE.fillText('Draw a path or pick a preset', epicyclesCanvas.width / 2 - 120, epicyclesCanvas.height / 2);
        }
    }
    else {
        state.signal1dTime += dt * state.signal1dSpeed * 2;
        draw1dSignal();
        drawSpectrum();
    }
    requestAnimationFrame(animate);
}
function setupDrawing() {
    function toComplex(e) {
        var r = epicyclesCanvas.getBoundingClientRect();
        return {
            re: e.clientX - r.left - epicyclesCanvas.width / 2,
            im: e.clientY - r.top - epicyclesCanvas.height / 2
        };
    }
    epicyclesCanvas.addEventListener('mousedown', function (e) {
        if (state.preset !== 'draw')
            return;
        state.drawing = true;
        state.drawnPath = [];
        state.phasors = [];
        state.tracePoints = [];
        state.drawnPath.push(toComplex(e));
    });
    epicyclesCanvas.addEventListener('mousemove', function (e) {
        if (!state.drawing)
            return;
        state.drawnPath.push(toComplex(e));
        clearCanvas(ctxE, epicyclesCanvas);
        drawGrid(ctxE, epicyclesCanvas);
        ctxE.beginPath();
        var cx = epicyclesCanvas.width / 2, cy = epicyclesCanvas.height / 2;
        state.drawnPath.forEach(function (p, i) {
            i === 0 ? ctxE.moveTo(cx + p.re, cy + p.im) : ctxE.lineTo(cx + p.re, cy + p.im);
        });
        ctxE.strokeStyle = C.trace;
        ctxE.lineWidth = 2;
        ctxE.stroke();
    });
    epicyclesCanvas.addEventListener('mouseup', function () {
        if (!state.drawing)
            return;
        state.drawing = false;
        if (state.drawnPath.length > 10)
            preparePhasors(state.drawnPath);
    });
}
function setupControls() {
    var modeBtns = document.querySelectorAll('.mode-btn');
    modeBtns.forEach(function (btn) {
        btn.addEventListener('click', function () {
            state.mode = btn.dataset.mode;
            modeBtns.forEach(function (b) { return b.classList.remove('active'); });
            btn.classList.add('active');
            var ePanel = document.getElementById('epicycles-panel');
            var sPanel = document.getElementById('signal1d-panel');
            if (state.mode === 'epicycles') {
                ePanel.style.display = '';
                sPanel.style.display = 'none';
            }
            else {
                ePanel.style.display = 'none';
                sPanel.style.display = '';
            }
        });
    });
    var presetSel = document.getElementById('preset-select');
    presetSel.addEventListener('change', function () {
        state.preset = presetSel.value;
        if (state.preset !== 'draw') {
            loadPreset(state.preset);
        }
        else {
            state.drawnPath = [];
            state.phasors = [];
            state.tracePoints = [];
        }
    });
    var termsSlider = document.getElementById('terms-slider');
    var termsLabel = document.getElementById('terms-label');
    termsSlider.addEventListener('input', function () {
        state.numTerms = parseInt(termsSlider.value, 10);
        termsLabel.textContent = String(state.numTerms);
        state.tracePoints = [];
        state.epicycleTime = 0;
    });
    var speedSlider = document.getElementById('speed-slider');
    var speedLabel = document.getElementById('speed-label');
    speedSlider.addEventListener('input', function () {
        state.epicycleSpeed = parseFloat(speedSlider.value);
        speedLabel.textContent = state.epicycleSpeed.toFixed(1) + 'x';
    });
    var clearBtn = document.getElementById('clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', function () {
            state.drawnPath = [];
            state.phasors = [];
            state.tracePoints = [];
            state.preset = 'draw';
            presetSel.value = 'draw';
        });
    }
    var waveformSel = document.getElementById('waveform-select');
    waveformSel.addEventListener('change', function () {
        state.waveform = waveformSel.value;
    });
    var harmonicsSlider = document.getElementById('harmonics-slider');
    var harmonicsLabel = document.getElementById('harmonics-label');
    harmonicsSlider.addEventListener('input', function () {
        state.numHarmonics = parseInt(harmonicsSlider.value, 10);
        harmonicsLabel.textContent = String(state.numHarmonics);
    });
    var sig1dSpeedSlider = document.getElementById('signal1d-speed-slider');
    var sig1dSpeedLabel = document.getElementById('signal1d-speed-label');
    sig1dSpeedSlider.addEventListener('input', function () {
        state.signal1dSpeed = parseFloat(sig1dSpeedSlider.value);
        sig1dSpeedLabel.textContent = state.signal1dSpeed.toFixed(1) + 'x';
    });
    var spectrumToggle = document.getElementById('spectrum-toggle');
    spectrumToggle.addEventListener('change', function () {
        state.showSpectrum = spectrumToggle.checked;
        var sc = document.getElementById('spectrum-container');
        sc.style.display = state.showSpectrum ? '' : 'none';
    });
}
function init() {
    Array.from(document.querySelectorAll('canvas')).forEach(function (c) { return resizeCanvas(c); });
    window.addEventListener('resize', function () {
        Array.from(document.querySelectorAll('canvas')).forEach(function (c) { return resizeCanvas(c); });
    });
    setupDrawing();
    setupControls();
    loadPreset('square');
    requestAnimationFrame(function (ts) { lastRaf = ts; requestAnimationFrame(animate); });
}
document.addEventListener('DOMContentLoaded', init);

},{}]},{},[1]);
