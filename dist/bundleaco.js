(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var cities = [];
var phero = [];
var bestTour = [];
var bestDist = Infinity;
var iteration = 0;
var running = false;
var timerId = null;
var NUM_CITIES = 20;
var NUM_ANTS = 30;
var ALPHA = 1.0;
var BETA = 3.0;
var RHO = 0.1;
var Q = 100;
var SPEED = 80;
var mainCanvas = document.getElementById("main-canvas");
var chartCanvas = document.getElementById("chart-canvas");
var ctx = mainCanvas.getContext("2d");
var cctx = chartCanvas.getContext("2d");
var W = mainCanvas.width;
var H = mainCanvas.height;
var MARGIN = 40;
function dist(a, b) {
    var dx = a.x - b.x;
    var dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
}
function tourLength(tour) {
    var d = 0;
    for (var i = 0; i < tour.length; i++) {
        d += dist(cities[tour[i]], cities[tour[(i + 1) % tour.length]]);
    }
    return d;
}
function initPhero() {
    var n = cities.length;
    phero = [];
    for (var i = 0; i < n; i++) {
        phero[i] = new Array(n).fill(1.0);
    }
}
function randomCities(n) {
    var result = [];
    for (var i = 0; i < n; i++) {
        result.push({
            x: MARGIN + Math.random() * (W - 2 * MARGIN),
            y: MARGIN + Math.random() * (H - 2 * MARGIN)
        });
    }
    return result;
}
function buildTour() {
    var n = cities.length;
    var visited = new Uint8Array(n);
    var tour = [];
    var current = Math.floor(Math.random() * n);
    visited[current] = 1;
    tour.push(current);
    for (var step = 1; step < n; step++) {
        var sum = 0;
        var weights = new Float64Array(n);
        for (var j = 0; j < n; j++) {
            if (visited[j])
                continue;
            var d = dist(cities[current], cities[j]);
            if (d === 0) {
                weights[j] = 0;
                continue;
            }
            var w = Math.pow(phero[current][j], ALPHA) * Math.pow(1 / d, BETA);
            weights[j] = w;
            sum += w;
        }
        var r = Math.random() * sum;
        var next = -1;
        for (var j = 0; j < n; j++) {
            if (visited[j])
                continue;
            r -= weights[j];
            if (r <= 0) {
                next = j;
                break;
            }
        }
        if (next === -1) {
            for (var j = 0; j < n; j++) {
                if (!visited[j]) {
                    next = j;
                    break;
                }
            }
        }
        visited[next] = 1;
        tour.push(next);
        current = next;
    }
    return { tour: tour, dist: tourLength(tour) };
}
function stepACO() {
    if (cities.length < 2)
        return;
    var n = cities.length;
    var ants = [];
    for (var k = 0; k < NUM_ANTS; k++) {
        ants.push(buildTour());
    }
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
            phero[i][j] *= (1 - RHO);
            if (phero[i][j] < 1e-6)
                phero[i][j] = 1e-6;
        }
    }
    for (var _i = 0, ants_1 = ants; _i < ants_1.length; _i++) {
        var ant = ants_1[_i];
        var deposit = Q / ant.dist;
        for (var s = 0; s < ant.tour.length; s++) {
            var a = ant.tour[s];
            var b = ant.tour[(s + 1) % ant.tour.length];
            phero[a][b] += deposit;
            phero[b][a] += deposit;
        }
        if (ant.dist < bestDist) {
            bestDist = ant.dist;
            bestTour = ant.tour.slice();
        }
    }
    iteration++;
    distHistory.push(bestDist);
    updateLabels();
    draw();
    drawChart();
}
var distHistory = [];
function maxPhero() {
    var m = 0;
    for (var _i = 0, phero_1 = phero; _i < phero_1.length; _i++) {
        var row = phero_1[_i];
        for (var _a = 0, row_1 = row; _a < row_1.length; _a++) {
            var v = row_1[_a];
            if (v > m)
                m = v;
        }
    }
    return m || 1;
}
function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0a0a14";
    ctx.fillRect(0, 0, W, H);
    if (cities.length < 2) {
        drawCities();
        return;
    }
    var n = cities.length;
    var maxP = maxPhero();
    for (var i = 0; i < n; i++) {
        for (var j = i + 1; j < n; j++) {
            var p = phero[i][j] / maxP;
            if (p < 0.02)
                continue;
            var alpha = Math.min(0.9, p * 0.9);
            var width = Math.min(6, 0.5 + p * 5);
            ctx.beginPath();
            ctx.moveTo(cities[i].x, cities[i].y);
            ctx.lineTo(cities[j].x, cities[j].y);
            ctx.strokeStyle = "rgba(56,189,248,".concat(alpha.toFixed(3), ")");
            ctx.lineWidth = width;
            ctx.stroke();
        }
    }
    if (bestTour.length === n) {
        ctx.beginPath();
        ctx.moveTo(cities[bestTour[0]].x, cities[bestTour[0]].y);
        for (var i = 1; i < n; i++) {
            ctx.lineTo(cities[bestTour[i]].x, cities[bestTour[i]].y);
        }
        ctx.closePath();
        ctx.strokeStyle = "rgba(251,191,36,0.85)";
        ctx.lineWidth = 2;
        ctx.stroke();
    }
    drawCities();
}
function drawCities() {
    for (var i = 0; i < cities.length; i++) {
        var c = cities[i];
        ctx.beginPath();
        ctx.arc(c.x, c.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = "#f0f0ff";
        ctx.fill();
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = "#0a0a14";
        ctx.font = "bold 9px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        if (cities.length <= 30)
            ctx.fillText(String(i + 1), c.x, c.y);
    }
}
function drawChart() {
    var cw = chartCanvas.width;
    var ch = chartCanvas.height;
    cctx.clearRect(0, 0, cw, ch);
    cctx.fillStyle = "#0a0a14";
    cctx.fillRect(0, 0, cw, ch);
    var data = distHistory;
    if (data.length < 2)
        return;
    var pad = { top: 12, right: 12, bottom: 28, left: 48 };
    var pw = cw - pad.left - pad.right;
    var ph = ch - pad.top - pad.bottom;
    var minD = Math.min.apply(Math, data) * 0.95;
    var maxD = Math.max.apply(Math, data) * 1.05;
    var xScale = function (i) { return pad.left + (i / (data.length - 1)) * pw; };
    var yScale = function (v) { return pad.top + ph - ((v - minD) / (maxD - minD)) * ph; };
    cctx.strokeStyle = "#334";
    cctx.lineWidth = 1;
    cctx.beginPath();
    cctx.moveTo(pad.left, pad.top);
    cctx.lineTo(pad.left, pad.top + ph);
    cctx.lineTo(pad.left + pw, pad.top + ph);
    cctx.stroke();
    cctx.fillStyle = "#7777aa";
    cctx.font = "9px monospace";
    cctx.textAlign = "right";
    for (var t = 0; t <= 4; t++) {
        var v = minD + (t / 4) * (maxD - minD);
        var y = yScale(v);
        cctx.fillText(v.toFixed(0), pad.left - 4, y + 3);
        cctx.strokeStyle = "#1a1a2e";
        cctx.lineWidth = 1;
        cctx.beginPath();
        cctx.moveTo(pad.left, y);
        cctx.lineTo(pad.left + pw, y);
        cctx.stroke();
    }
    cctx.fillStyle = "#7777aa";
    cctx.textAlign = "center";
    cctx.fillText("Iteration", pad.left + pw / 2, ch - 4);
    cctx.beginPath();
    cctx.moveTo(xScale(0), yScale(data[0]));
    for (var i = 1; i < data.length; i++) {
        cctx.lineTo(xScale(i), yScale(data[i]));
    }
    cctx.strokeStyle = "#f59e0b";
    cctx.lineWidth = 1.5;
    cctx.stroke();
    var lastX = xScale(data.length - 1);
    var lastY = yScale(data[data.length - 1]);
    cctx.beginPath();
    cctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
    cctx.fillStyle = "#fbbf24";
    cctx.fill();
}
function updateLabels() {
    var iterLabel = document.getElementById("iter-label");
    iterLabel.textContent = "Iter ".concat(iteration, "  |  Best: ").concat(bestDist === Infinity ? "–" : bestDist.toFixed(1));
}
function reset() {
    pause();
    cities = randomCities(NUM_CITIES);
    initPhero();
    bestTour = [];
    bestDist = Infinity;
    iteration = 0;
    distHistory.length = 0;
    updateLabels();
    draw();
    drawChart();
}
function pause() {
    running = false;
    if (timerId !== null) {
        clearTimeout(timerId);
        timerId = null;
    }
    var btn = document.getElementById("btn-play");
    btn.textContent = "▶ Play";
}
function play() {
    if (running)
        return;
    running = true;
    var btn = document.getElementById("btn-play");
    btn.textContent = "⏸ Pause";
    function loop() {
        if (!running)
            return;
        stepACO();
        timerId = window.setTimeout(loop, SPEED);
    }
    loop();
}
function togglePlay() {
    if (running)
        pause();
    else
        play();
}
function wireSlider(id, valId, decimals, setter) {
    var inp = document.getElementById(id);
    var lbl = document.getElementById(valId);
    lbl.textContent = parseFloat(inp.value).toFixed(decimals);
    inp.addEventListener("input", function () {
        var v = parseFloat(inp.value);
        lbl.textContent = v.toFixed(decimals);
        setter(v);
    });
}
function init() {
    document.getElementById("btn-play").addEventListener("click", togglePlay);
    document.getElementById("btn-step").addEventListener("click", function () { pause(); stepACO(); });
    document.getElementById("btn-reset").addEventListener("click", reset);
    wireSlider("inp-cities", "lbl-cities", 0, function (v) { NUM_CITIES = v; reset(); });
    wireSlider("inp-ants", "lbl-ants", 0, function (v) { NUM_ANTS = v; });
    wireSlider("inp-alpha", "lbl-alpha", 1, function (v) { ALPHA = v; });
    wireSlider("inp-beta", "lbl-beta", 1, function (v) { BETA = v; });
    wireSlider("inp-rho", "lbl-rho", 2, function (v) { RHO = v; });
    wireSlider("inp-q", "lbl-q", 0, function (v) { Q = v; });
    wireSlider("inp-speed", "lbl-speed", 0, function (v) { SPEED = v; });
    mainCanvas.addEventListener("click", function (e) {
        var rect = mainCanvas.getBoundingClientRect();
        var sx = mainCanvas.width / rect.width;
        var sy = mainCanvas.height / rect.height;
        cities.push({ x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy });
        if (cities.length >= 2)
            initPhero();
        bestTour = [];
        bestDist = Infinity;
        iteration = 0;
        distHistory.length = 0;
        updateLabels();
        draw();
        drawChart();
    });
    reset();
}
document.addEventListener("DOMContentLoaded", init);

},{}]},{},[1]);
