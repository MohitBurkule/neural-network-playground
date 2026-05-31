(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function logBanana(x) {
    var a = 1, b = 10;
    var u = x[0] / a;
    var v = x[1] - b * (u * u - 1);
    return -0.5 * (u * u + v * v);
}
function logMixtureGaussians(x) {
    var centers = [[-2.5, -2.5], [2.5, 2.5], [-2.5, 2.5], [2.5, -2.5]];
    var sig2 = 0.7;
    var sum = 0;
    centers.forEach(function (c) {
        var dx = x[0] - c[0], dy = x[1] - c[1];
        sum += Math.exp(-0.5 * (dx * dx + dy * dy) / sig2);
    });
    return Math.log(sum + 1e-300);
}
function logCorrelatedGaussian(x) {
    var sig1 = 1.5, sig2 = 1.5, rho = 0.9;
    var z = (x[0] * x[0]) / (sig1 * sig1)
        - 2 * rho * x[0] * x[1] / (sig1 * sig2)
        + (x[1] * x[1]) / (sig2 * sig2);
    return -z / (2 * (1 - rho * rho));
}
function logRing(x) {
    var r = Math.sqrt(x[0] * x[0] + x[1] * x[1]);
    var dr = r - 3;
    return -0.5 * (dr * dr) / (0.4 * 0.4);
}
var TARGETS = {
    banana: { fn: logBanana, label: 'Banana (Rosenbrock)', range: 5 },
    mixture: { fn: logMixtureGaussians, label: 'Mixture of Gaussians', range: 5 },
    correlated: { fn: logCorrelatedGaussian, label: 'Correlated Gaussian', range: 5 },
    ring: { fn: logRing, label: 'Ring / Donut', range: 5 }
};
function randn() {
    var u = 0, v = 0;
    while (u === 0)
        u = Math.random();
    while (v === 0)
        v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function stepMH(current, logTarget, sigma) {
    var proposal = [current[0] + sigma * randn(), current[1] + sigma * randn()];
    var logA = logTarget(proposal) - logTarget(current);
    var accepted = Math.log(Math.random()) < logA;
    return { x: accepted ? proposal : current, accepted: accepted, proposal: proposal };
}
function stepGibbs(current, logTarget, sigma) {
    var state = [current[0], current[1]];
    var accepted = false;
    var prop0 = [state[0] + sigma * randn(), state[1]];
    if (Math.log(Math.random()) < logTarget(prop0) - logTarget(state)) {
        state = prop0;
        accepted = true;
    }
    var prop1 = [state[0], state[1] + sigma * randn()];
    if (Math.log(Math.random()) < logTarget(prop1) - logTarget(state)) {
        state = prop1;
        accepted = true;
    }
    return { x: state, accepted: accepted, proposal: state };
}
function stepHMC(current, logTarget, sigma) {
    var L = 10;
    var eps = sigma * 0.3;
    var grad = function (x) {
        var h = 1e-4;
        var gx = (logTarget([x[0] + h, x[1]]) - logTarget([x[0] - h, x[1]])) / (2 * h);
        var gy = (logTarget([x[0], x[1] + h]) - logTarget([x[0], x[1] - h])) / (2 * h);
        return [gx, gy];
    };
    var q = [current[0], current[1]];
    var p = [randn(), randn()];
    var H0 = -logTarget(q) + 0.5 * (p[0] * p[0] + p[1] * p[1]);
    var g = grad(q);
    p = [p[0] + 0.5 * eps * g[0], p[1] + 0.5 * eps * g[1]];
    for (var i = 0; i < L; i++) {
        q = [q[0] + eps * p[0], q[1] + eps * p[1]];
        g = grad(q);
        if (i < L - 1) {
            p = [p[0] + eps * g[0], p[1] + eps * g[1]];
        }
        else {
            p = [p[0] + 0.5 * eps * g[0], p[1] + 0.5 * eps * g[1]];
        }
    }
    var H1 = -logTarget(q) + 0.5 * (p[0] * p[0] + p[1] * p[1]);
    var accepted = Math.log(Math.random()) < H0 - H1;
    return { x: accepted ? q : current, accepted: accepted, proposal: q };
}
var SAMPLERS = {
    mh: { fn: stepMH, label: 'Metropolis-Hastings' },
    gibbs: { fn: stepGibbs, label: 'Gibbs Sampler' },
    hmc: { fn: stepHMC, label: 'HMC-lite' }
};
var targetKey = 'banana';
var samplerKey = 'mh';
var sigma = 0.5;
var speed = 5;
var burnIn = 200;
var showHistogram = true;
var playing = false;
var animId = null;
var current = [0, 0];
var chainPath = [];
var allSamples = [];
var lastSample = null;
var stepCount = 0;
var acceptCount = 0;
var MAX_PATH = 200;
var HIST_BINS = 50;
var DENSITY_SIZE = 480;
var TRACE_W = 480;
var TRACE_H = 100;
var MARGIN = 40;
var densityCanvas;
var densityCtx;
var traceXCanvas;
var traceXCtx;
var traceYCanvas;
var traceYCtx;
var heatmapCache = null;
var heatmapTargetKey = '';
function buildHeatmap(range) {
    var size = DENSITY_SIZE;
    var img = densityCtx.createImageData(size, size);
    var logFn = TARGETS[targetKey].fn;
    var maxVal = -Infinity;
    var vals = new Float32Array(size * size);
    for (var py = 0; py < size; py++) {
        for (var px = 0; px < size; px++) {
            var wx = (px / size) * 2 * range - range;
            var wy = range - (py / size) * 2 * range;
            var v = logFn([wx, wy]);
            vals[py * size + px] = v;
            if (v > maxVal)
                maxVal = v;
        }
    }
    for (var i = 0; i < vals.length; i++) {
        var t = Math.max(0, Math.min(1, (vals[i] - maxVal + 12) / 12));
        var r = Math.round(t * t * 180);
        var g = Math.round(t * 200);
        var b = Math.round(80 + t * 175);
        img.data[i * 4 + 0] = r;
        img.data[i * 4 + 1] = g;
        img.data[i * 4 + 2] = b;
        img.data[i * 4 + 3] = 255;
    }
    return img;
}
function buildHistogram(range) {
    var bins = HIST_BINS;
    var counts = new Float32Array(bins * bins);
    allSamples.forEach(function (s) {
        var bx = Math.floor(((s[0] + range) / (2 * range)) * bins);
        var by = Math.floor(((range - s[1]) / (2 * range)) * bins);
        if (bx >= 0 && bx < bins && by >= 0 && by < bins) {
            counts[by * bins + bx]++;
        }
    });
    return counts;
}
function toCanvas(wx, wy, range) {
    var px = ((wx + range) / (2 * range)) * DENSITY_SIZE;
    var py = ((range - wy) / (2 * range)) * DENSITY_SIZE;
    return [px, py];
}
function drawDensity() {
    var range = TARGETS[targetKey].range;
    if (heatmapTargetKey !== targetKey || !heatmapCache) {
        heatmapCache = buildHeatmap(range);
        heatmapTargetKey = targetKey;
    }
    densityCtx.putImageData(heatmapCache, 0, 0);
    if (showHistogram && allSamples.length > 10) {
        var counts = buildHistogram(range);
        var maxC = Math.max.apply(Math, Array.from(counts));
        if (maxC > 0) {
            var cellSize = DENSITY_SIZE / HIST_BINS;
            for (var by = 0; by < HIST_BINS; by++) {
                for (var bx = 0; bx < HIST_BINS; bx++) {
                    var c = counts[by * HIST_BINS + bx];
                    if (c > 0) {
                        var alpha = Math.min(1, c / maxC) * 0.5;
                        densityCtx.fillStyle = "rgba(255,230,100,".concat(alpha.toFixed(3), ")");
                        densityCtx.fillRect(bx * cellSize, by * cellSize, cellSize, cellSize);
                    }
                }
            }
        }
    }
    if (chainPath.length > 1) {
        densityCtx.lineWidth = 1;
        for (var i = 1; i < chainPath.length; i++) {
            var age = (i / chainPath.length);
            var alpha = age * 0.8;
            densityCtx.strokeStyle = "rgba(255,100,100,".concat(alpha.toFixed(3), ")");
            densityCtx.beginPath();
            var _a = toCanvas(chainPath[i - 1][0], chainPath[i - 1][1], range), x0 = _a[0], y0 = _a[1];
            var _b = toCanvas(chainPath[i][0], chainPath[i][1], range), x1 = _b[0], y1 = _b[1];
            densityCtx.moveTo(x0, y0);
            densityCtx.lineTo(x1, y1);
            densityCtx.stroke();
        }
    }
    if (lastSample && lastSample.proposal) {
        var _c = toCanvas(current[0], current[1], range), cx = _c[0], cy = _c[1];
        if (!lastSample.accepted && lastSample.proposal) {
            var _d = toCanvas(lastSample.proposal[0], lastSample.proposal[1], range), px = _d[0], py = _d[1];
            densityCtx.strokeStyle = 'rgba(255,60,60,0.7)';
            densityCtx.lineWidth = 1;
            densityCtx.setLineDash([4, 3]);
            densityCtx.beginPath();
            densityCtx.moveTo(cx, cy);
            densityCtx.lineTo(px, py);
            densityCtx.stroke();
            densityCtx.setLineDash([]);
            densityCtx.fillStyle = 'rgba(255,60,60,0.6)';
            densityCtx.beginPath();
            densityCtx.arc(px, py, 4, 0, 2 * Math.PI);
            densityCtx.fill();
        }
        densityCtx.fillStyle = '#fff';
        densityCtx.beginPath();
        densityCtx.arc(cx, cy, 5, 0, 2 * Math.PI);
        densityCtx.fill();
        densityCtx.strokeStyle = '#7eb8f7';
        densityCtx.lineWidth = 2;
        densityCtx.stroke();
    }
}
var MAX_TRACE = 300;
var traceX = [];
var traceY = [];
function drawTrace(ctx, data, color, label) {
    var W = TRACE_W, H = TRACE_H;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, W, H);
    if (data.length < 2)
        return;
    var min = Math.min.apply(Math, data);
    var max = Math.max.apply(Math, data);
    var range = max - min || 1;
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, H / 2);
    ctx.lineTo(W, H / 2);
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    data.forEach(function (v, i) {
        var px = (i / (data.length - 1)) * W;
        var py = H - ((v - min) / range) * (H - 10) - 5;
        if (i === 0)
            ctx.moveTo(px, py);
        else
            ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.fillStyle = '#aaa';
    ctx.font = '11px monospace';
    ctx.fillText(label, 6, 14);
}
function drawAll() {
    drawDensity();
    drawTrace(traceXCtx, traceX, '#7eb8f7', "x\u2080  (last: ".concat(current[0].toFixed(2), ")"));
    drawTrace(traceYCtx, traceY, '#f7c07e', "x\u2081  (last: ".concat(current[1].toFixed(2), ")"));
    var rate = stepCount > 0 ? (acceptCount / stepCount * 100).toFixed(1) : '—';
    var el = document.getElementById('acc-rate');
    if (el)
        el.textContent = "".concat(rate, "%");
    var stepEl = document.getElementById('step-count');
    if (stepEl)
        stepEl.textContent = "".concat(stepCount);
    var burnEl = document.getElementById('burn-status');
    if (burnEl)
        burnEl.textContent = stepCount < burnIn ? "burn-in (".concat(stepCount, "/").concat(burnIn, ")") : 'sampling';
}
function doStep() {
    var logFn = TARGETS[targetKey].fn;
    var samplerFn = SAMPLERS[samplerKey].fn;
    var s = samplerFn(current, logFn, sigma);
    lastSample = s;
    current = s.x;
    stepCount++;
    if (s.accepted)
        acceptCount++;
    if (stepCount > burnIn) {
        allSamples.push([current[0], current[1]]);
        if (allSamples.length > 5000)
            allSamples.splice(0, allSamples.length - 5000);
    }
    chainPath.push([current[0], current[1]]);
    if (chainPath.length > MAX_PATH)
        chainPath.shift();
    traceX.push(current[0]);
    traceY.push(current[1]);
    if (traceX.length > MAX_TRACE)
        traceX.shift();
    if (traceY.length > MAX_TRACE)
        traceY.shift();
}
function doReset() {
    current = [0, 0];
    chainPath = [];
    allSamples = [];
    lastSample = null;
    stepCount = 0;
    acceptCount = 0;
    traceX = [];
    traceY = [];
    heatmapCache = null;
    drawAll();
}
function tick() {
    for (var i = 0; i < speed; i++)
        doStep();
    drawAll();
    if (playing)
        animId = requestAnimationFrame(tick);
}
function play() {
    if (playing)
        return;
    playing = true;
    animId = requestAnimationFrame(tick);
}
function pause() {
    playing = false;
    if (animId !== null) {
        cancelAnimationFrame(animId);
        animId = null;
    }
}
function buildUI() {
    var root = document.getElementById('app');
    root.innerHTML = '';
    var title = document.createElement('h1');
    title.textContent = 'MCMC Sampling Lab';
    root.appendChild(title);
    var sub = document.createElement('div');
    sub.className = 'subtitle';
    sub.textContent = 'Visualizing Markov Chain Monte Carlo over 2D target distributions';
    root.appendChild(sub);
    var layout = document.createElement('div');
    layout.className = 'main-layout';
    root.appendChild(layout);
    var left = document.createElement('div');
    left.className = 'left-col';
    layout.appendChild(left);
    densityCanvas = document.createElement('canvas');
    densityCanvas.width = DENSITY_SIZE;
    densityCanvas.height = DENSITY_SIZE;
    densityCanvas.className = 'density-canvas';
    left.appendChild(densityCanvas);
    densityCtx = densityCanvas.getContext('2d');
    var tracesLabel = document.createElement('div');
    tracesLabel.className = 'section-label';
    tracesLabel.textContent = 'Trace Plots';
    left.appendChild(tracesLabel);
    traceXCanvas = document.createElement('canvas');
    traceXCanvas.width = TRACE_W;
    traceXCanvas.height = TRACE_H;
    traceXCanvas.className = 'trace-canvas';
    left.appendChild(traceXCanvas);
    traceXCtx = traceXCanvas.getContext('2d');
    traceYCanvas = document.createElement('canvas');
    traceYCanvas.width = TRACE_W;
    traceYCanvas.height = TRACE_H;
    traceYCanvas.className = 'trace-canvas';
    left.appendChild(traceYCanvas);
    traceYCtx = traceYCanvas.getContext('2d');
    var right = document.createElement('div');
    right.className = 'right-col';
    layout.appendChild(right);
    function makeSection(title) {
        var sec = document.createElement('div');
        sec.className = 'control-section';
        var t = document.createElement('div');
        t.className = 'control-title';
        t.textContent = title;
        sec.appendChild(t);
        right.appendChild(sec);
        return sec;
    }
    function makeLabel(text, forId) {
        var l = document.createElement('label');
        l.textContent = text;
        if (forId)
            l.htmlFor = forId;
        return l;
    }
    function makeSelect(id, options, val, onChange) {
        var sel = document.createElement('select');
        sel.id = id;
        Object.keys(options).forEach(function (k) {
            var opt = document.createElement('option');
            opt.value = k;
            opt.textContent = options[k];
            if (k === val)
                opt.selected = true;
            sel.appendChild(opt);
        });
        sel.addEventListener('change', function () { return onChange(sel.value); });
        return sel;
    }
    function makeSlider(id, min, max, step, value, onChange) {
        var row = document.createElement('div');
        row.className = 'slider-row';
        var inp = document.createElement('input');
        inp.type = 'range';
        inp.id = id;
        inp.min = String(min);
        inp.max = String(max);
        inp.step = String(step);
        inp.value = String(value);
        var valSpan = document.createElement('span');
        valSpan.textContent = String(value);
        inp.addEventListener('input', function () {
            valSpan.textContent = inp.value;
            onChange(parseFloat(inp.value));
        });
        row.appendChild(inp);
        row.appendChild(valSpan);
        return { row: row, valSpan: valSpan };
    }
    var tSec = makeSection('Target Distribution');
    var targetOpts = {};
    Object.keys(TARGETS).forEach(function (k) { targetOpts[k] = TARGETS[k].label; });
    var targetSel = makeSelect('target-sel', targetOpts, targetKey, function (v) {
        targetKey = v;
        doReset();
    });
    tSec.appendChild(targetSel);
    var sSec = makeSection('Sampler');
    var samplerOpts = {};
    Object.keys(SAMPLERS).forEach(function (k) { samplerOpts[k] = SAMPLERS[k].label; });
    var samplerSel = makeSelect('sampler-sel', samplerOpts, samplerKey, function (v) {
        samplerKey = v;
        doReset();
    });
    sSec.appendChild(samplerSel);
    var pSec = makeSection('Parameters');
    var sigRow = document.createElement('div');
    sigRow.className = 'param-row';
    sigRow.appendChild(makeLabel('Step size σ', 'sigma-sl'));
    var sigSlider = makeSlider('sigma-sl', 0.05, 3, 0.05, sigma, function (v) { sigma = v; });
    sigRow.appendChild(sigSlider.row);
    pSec.appendChild(sigRow);
    var speedRow = document.createElement('div');
    speedRow.className = 'param-row';
    speedRow.appendChild(makeLabel('Speed (steps/frame)', 'speed-sl'));
    var speedSlider = makeSlider('speed-sl', 1, 50, 1, speed, function (v) { speed = Math.round(v); });
    speedRow.appendChild(speedSlider.row);
    pSec.appendChild(speedRow);
    var burnRow = document.createElement('div');
    burnRow.className = 'param-row';
    burnRow.appendChild(makeLabel('Burn-in steps', 'burnin-sl'));
    var burnSlider = makeSlider('burnin-sl', 0, 1000, 50, burnIn, function (v) { burnIn = Math.round(v); });
    burnRow.appendChild(burnSlider.row);
    pSec.appendChild(burnRow);
    var histRow = document.createElement('div');
    histRow.className = 'param-row toggle-row';
    var histLabel = makeLabel('Show histogram overlay');
    var histCheck = document.createElement('input');
    histCheck.type = 'checkbox';
    histCheck.checked = showHistogram;
    histCheck.addEventListener('change', function () { showHistogram = histCheck.checked; drawAll(); });
    histRow.appendChild(histCheck);
    histRow.appendChild(histLabel);
    pSec.appendChild(histRow);
    var cSec = makeSection('Controls');
    var btnRow = document.createElement('div');
    btnRow.className = 'btn-row';
    var playBtn = document.createElement('button');
    playBtn.textContent = '▶ Play';
    playBtn.className = 'btn btn-primary';
    playBtn.addEventListener('click', function () {
        if (playing) {
            pause();
            playBtn.textContent = '▶ Play';
        }
        else {
            play();
            playBtn.textContent = '⏸ Pause';
        }
    });
    var stepBtn = document.createElement('button');
    stepBtn.textContent = 'Step';
    stepBtn.className = 'btn';
    stepBtn.addEventListener('click', function () {
        if (playing) {
            pause();
            playBtn.textContent = '▶ Play';
        }
        doStep();
        drawAll();
    });
    var resetBtn = document.createElement('button');
    resetBtn.textContent = 'Reset';
    resetBtn.className = 'btn';
    resetBtn.addEventListener('click', function () {
        pause();
        playBtn.textContent = '▶ Play';
        doReset();
    });
    btnRow.appendChild(playBtn);
    btnRow.appendChild(stepBtn);
    btnRow.appendChild(resetBtn);
    cSec.appendChild(btnRow);
    var statSec = makeSection('Statistics');
    var statGrid = document.createElement('div');
    statGrid.className = 'stat-grid';
    function makeStatRow(label, id, init) {
        var row = document.createElement('div');
        row.className = 'stat-row';
        var lbl = document.createElement('span');
        lbl.textContent = label;
        var val = document.createElement('span');
        val.id = id;
        val.className = 'stat-val';
        val.textContent = init;
        row.appendChild(lbl);
        row.appendChild(val);
        statGrid.appendChild(row);
    }
    makeStatRow('Acceptance Rate:', 'acc-rate', '—');
    makeStatRow('Total Steps:', 'step-count', '0');
    makeStatRow('Status:', 'burn-status', 'burn-in (0/200)');
    statSec.appendChild(statGrid);
    var legSec = makeSection('Legend');
    var legItems = [
        { color: '#7eb8f7', label: 'Current position' },
        { color: 'rgba(255,100,100,0.8)', label: 'Chain path (recent)' },
        { color: 'rgba(255,60,60,0.6)', label: 'Rejected proposal' },
        { color: 'rgba(255,230,100,0.7)', label: 'Sample histogram' },
    ];
    legItems.forEach(function (item) {
        var row = document.createElement('div');
        row.className = 'leg-row';
        var swatch = document.createElement('span');
        swatch.className = 'swatch';
        swatch.style.background = item.color;
        var lbl = document.createElement('span');
        lbl.textContent = item.label;
        row.appendChild(swatch);
        row.appendChild(lbl);
        legSec.appendChild(row);
    });
    drawAll();
}
function injectStyles() {
    var style = document.createElement('style');
    style.textContent = "\n    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }\n    body {\n      font-family: 'Segoe UI', Arial, sans-serif;\n      background: #0f1117;\n      color: #e0e0e0;\n      min-height: 100vh;\n      display: flex;\n      flex-direction: column;\n      align-items: center;\n      padding: 24px 16px 40px;\n    }\n    h1 {\n      font-size: 1.6rem;\n      font-weight: 600;\n      color: #7eb8f7;\n      margin-bottom: 4px;\n      letter-spacing: 0.5px;\n    }\n    .subtitle {\n      font-size: 0.85rem;\n      color: #888;\n      margin-bottom: 20px;\n      text-align: center;\n    }\n    .main-layout {\n      display: flex;\n      gap: 24px;\n      align-items: flex-start;\n      flex-wrap: wrap;\n      justify-content: center;\n      width: 100%;\n      max-width: 1100px;\n    }\n    .left-col {\n      display: flex;\n      flex-direction: column;\n      gap: 6px;\n    }\n    .density-canvas {\n      border: 1px solid #2a2a3a;\n      border-radius: 6px;\n      display: block;\n    }\n    .section-label {\n      font-size: 0.75rem;\n      color: #666;\n      text-transform: uppercase;\n      letter-spacing: 1px;\n      margin-top: 4px;\n    }\n    .trace-canvas {\n      border: 1px solid #2a2a3a;\n      border-radius: 4px;\n      display: block;\n    }\n    .right-col {\n      display: flex;\n      flex-direction: column;\n      gap: 12px;\n      min-width: 240px;\n      max-width: 280px;\n    }\n    .control-section {\n      background: #161820;\n      border: 1px solid #2a2a3a;\n      border-radius: 8px;\n      padding: 12px 14px;\n      display: flex;\n      flex-direction: column;\n      gap: 8px;\n    }\n    .control-title {\n      font-size: 0.75rem;\n      text-transform: uppercase;\n      letter-spacing: 1px;\n      color: #7eb8f7;\n      margin-bottom: 2px;\n    }\n    select {\n      background: #1e2030;\n      color: #e0e0e0;\n      border: 1px solid #3a3a5a;\n      border-radius: 4px;\n      padding: 5px 8px;\n      font-size: 0.85rem;\n      width: 100%;\n      cursor: pointer;\n    }\n    .param-row {\n      display: flex;\n      flex-direction: column;\n      gap: 3px;\n    }\n    .param-row label {\n      font-size: 0.8rem;\n      color: #aaa;\n    }\n    .slider-row {\n      display: flex;\n      align-items: center;\n      gap: 8px;\n    }\n    .slider-row input[type=range] {\n      flex: 1;\n      accent-color: #7eb8f7;\n    }\n    .slider-row span {\n      font-size: 0.8rem;\n      color: #7eb8f7;\n      min-width: 32px;\n      text-align: right;\n      font-family: monospace;\n    }\n    .toggle-row {\n      flex-direction: row;\n      align-items: center;\n      gap: 8px;\n    }\n    .toggle-row label { color: #ccc; }\n    .btn-row {\n      display: flex;\n      gap: 8px;\n      flex-wrap: wrap;\n    }\n    .btn {\n      background: #1e2030;\n      color: #e0e0e0;\n      border: 1px solid #3a3a5a;\n      border-radius: 4px;\n      padding: 6px 14px;\n      font-size: 0.85rem;\n      cursor: pointer;\n      transition: background 0.15s;\n    }\n    .btn:hover { background: #2a2d45; }\n    .btn-primary {\n      background: #1a3a5c;\n      border-color: #4a7ab5;\n      color: #7eb8f7;\n    }\n    .btn-primary:hover { background: #224d7a; }\n    .stat-grid {\n      display: flex;\n      flex-direction: column;\n      gap: 5px;\n    }\n    .stat-row {\n      display: flex;\n      justify-content: space-between;\n      font-size: 0.82rem;\n      color: #aaa;\n    }\n    .stat-val {\n      color: #7eb8f7;\n      font-family: monospace;\n    }\n    .leg-row {\n      display: flex;\n      align-items: center;\n      gap: 8px;\n      font-size: 0.8rem;\n      color: #aaa;\n    }\n    .swatch {\n      width: 14px;\n      height: 14px;\n      border-radius: 3px;\n      flex-shrink: 0;\n    }\n  ";
    document.head.appendChild(style);
}
document.addEventListener('DOMContentLoaded', function () {
    injectStyles();
    buildUI();
});

},{}]},{},[1]);
