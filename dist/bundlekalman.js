(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function kmat(rows, cols, data) {
    return { rows: rows, cols: cols, data: data ? data.slice() : new Array(rows * cols).fill(0) };
}
function mget(m, r, c) {
    return m.data[r * m.cols + c];
}
function mset(m, r, c, v) {
    m.data[r * m.cols + c] = v;
}
function mmul(a, b) {
    var out = kmat(a.rows, b.cols);
    for (var i = 0; i < a.rows; i++) {
        for (var j = 0; j < b.cols; j++) {
            var s = 0;
            for (var k = 0; k < a.cols; k++)
                s += mget(a, i, k) * mget(b, k, j);
            mset(out, i, j, s);
        }
    }
    return out;
}
function madd(a, b) {
    var out = kmat(a.rows, a.cols);
    for (var i = 0; i < a.data.length; i++)
        out.data[i] = a.data[i] + b.data[i];
    return out;
}
function msub(a, b) {
    var out = kmat(a.rows, a.cols);
    for (var i = 0; i < a.data.length; i++)
        out.data[i] = a.data[i] - b.data[i];
    return out;
}
function mscale(a, s) {
    var out = kmat(a.rows, a.cols);
    for (var i = 0; i < a.data.length; i++)
        out.data[i] = a.data[i] * s;
    return out;
}
function mtranspose(a) {
    var out = kmat(a.cols, a.rows);
    for (var r = 0; r < a.rows; r++) {
        for (var c = 0; c < a.cols; c++) {
            mset(out, c, r, mget(a, r, c));
        }
    }
    return out;
}
function keye(n) {
    var m = kmat(n, n);
    for (var i = 0; i < n; i++)
        mset(m, i, i, 1);
    return m;
}
function inv2(m) {
    var a = mget(m, 0, 0), b = mget(m, 0, 1);
    var c = mget(m, 1, 0), d = mget(m, 1, 1);
    var det = a * d - b * c;
    return kmat(2, 2, [d / det, -b / det, -c / det, a / det]);
}
function buildF(dt) {
    return kmat(4, 4, [
        1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]);
}
function buildQ(dt, q) {
    var dt2 = dt * dt;
    var dt3 = dt2 * dt;
    var dt4 = dt3 * dt;
    return kmat(4, 4, [
        q * dt4 / 4, 0, q * dt3 / 2, 0,
        0, q * dt4 / 4, 0, q * dt3 / 2,
        q * dt3 / 2, 0, q * dt2, 0,
        0, q * dt3 / 2, 0, q * dt2,
    ]);
}
var KH = kmat(2, 4, [
    1, 0, 0, 0,
    0, 1, 0, 0,
]);
function buildR(r) {
    return kmat(2, 2, [r, 0, 0, r]);
}
function kalmanPredict(state, F, Q) {
    var xp = mmul(F, state.x);
    var Pp = madd(mmul(mmul(F, state.P), mtranspose(F)), Q);
    return { x: xp, P: Pp };
}
function kalmanUpdate(state, z, R) {
    var y = msub(z, mmul(KH, state.x));
    var S = madd(mmul(mmul(KH, state.P), mtranspose(KH)), R);
    var K = mmul(mmul(state.P, mtranspose(KH)), inv2(S));
    var xu = madd(state.x, mmul(K, y));
    var I_KH = msub(keye(4), mmul(K, KH));
    var Pu = mmul(I_KH, state.P);
    return { x: xu, P: Pu };
}
function circleTraj(t, cx, cy, r, speed) {
    var angle = t * speed;
    return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
}
function fig8Traj(t, cx, cy, r, speed) {
    var angle = t * speed;
    return {
        x: cx + r * Math.cos(angle),
        y: cy + r * 0.5 * Math.sin(2 * angle)
    };
}
var KalmanApp = (function () {
    function KalmanApp() {
        this.playing = false;
        this.t = 0;
        this.dt = 0.05;
        this.speed = 1.0;
        this.qNoise = 0.1;
        this.rNoise = 10.0;
        this.mode = "circle";
        this.animId = 0;
        this.trail = [];
        this.maxTrail = 300;
        this.drawPath = [];
        this.drawPathIndex = 0;
        this.isDrawing = false;
        this.canvas = document.getElementById("mainCanvas");
        this.ctx = this.canvas.getContext("2d");
        this.plotCanvas = document.getElementById("plotCanvas");
        this.plotCtx = this.plotCanvas.getContext("2d");
        this.kalmanState = this.initKalman();
        this.bindControls();
        this.bindDrawEvents();
        this.render();
    }
    KalmanApp.prototype.initKalman = function () {
        var start = this.groundTruth(0);
        return {
            x: kmat(4, 1, [start.x, start.y, 0, 0]),
            P: mscale(keye(4), 100)
        };
    };
    KalmanApp.prototype.groundTruth = function (t) {
        var cx = this.canvas.width / 2;
        var cy = this.canvas.height / 2;
        var r = Math.min(cx, cy) * 0.55;
        if (this.mode === "circle")
            return circleTraj(t, cx, cy, r, this.speed);
        if (this.mode === "figure8")
            return fig8Traj(t, cx, cy, r, this.speed);
        if (this.drawPath.length < 2)
            return { x: cx, y: cy };
        var idx = this.drawPathIndex % this.drawPath.length;
        return this.drawPath[idx];
    };
    KalmanApp.prototype.addNoise = function (p, sigma) {
        var u1 = Math.random(), u2 = Math.random();
        var n1 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
        var n2 = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.sin(2 * Math.PI * u2);
        return { x: p.x + n1 * sigma, y: p.y + n2 * sigma };
    };
    KalmanApp.prototype.step = function () {
        this.t += this.dt;
        if (this.mode === "draw")
            this.drawPathIndex++;
        var truth = this.groundTruth(this.t);
        var meas = this.addNoise(truth, Math.sqrt(this.rNoise));
        var F = buildF(this.dt);
        var Q = buildQ(this.dt, this.qNoise);
        var R = buildR(this.rNoise);
        var state = kalmanPredict(this.kalmanState, F, Q);
        var z = kmat(2, 1, [meas.x, meas.y]);
        state = kalmanUpdate(state, z, R);
        this.kalmanState = state;
        var ex = mget(state.x, 0, 0);
        var ey = mget(state.x, 1, 0);
        this.trail.push({
            true_x: truth.x, true_y: truth.y,
            meas_x: meas.x, meas_y: meas.y,
            est_x: ex, est_y: ey
        });
        if (this.trail.length > this.maxTrail)
            this.trail.shift();
    };
    KalmanApp.prototype.reset = function () {
        this.t = 0;
        this.drawPathIndex = 0;
        this.trail = [];
        this.kalmanState = this.initKalman();
        this.render();
    };
    KalmanApp.prototype.drawCovarEllipse = function (cx, cy) {
        var P = this.kalmanState.P;
        var pxx = mget(P, 0, 0);
        var pxy = mget(P, 0, 1);
        var pyy = mget(P, 1, 1);
        var trace = pxx + pyy;
        var det = pxx * pyy - pxy * pxy;
        var disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
        var l1 = trace / 2 + disc;
        var l2 = trace / 2 - disc;
        var a = Math.sqrt(Math.max(0, l1)) * 2.45;
        var b = Math.sqrt(Math.max(0, l2)) * 2.45;
        var angle = Math.atan2(pxy, pxx - l2);
        this.ctx.save();
        this.ctx.translate(cx, cy);
        this.ctx.rotate(angle);
        this.ctx.beginPath();
        this.ctx.ellipse(0, 0, Math.max(a, 1), Math.max(b, 1), 0, 0, Math.PI * 2);
        this.ctx.strokeStyle = "rgba(100,220,255,0.6)";
        this.ctx.lineWidth = 1.5;
        this.ctx.fillStyle = "rgba(100,220,255,0.07)";
        this.ctx.fill();
        this.ctx.stroke();
        this.ctx.restore();
    };
    KalmanApp.prototype.render = function () {
        var W = this.canvas.width;
        var H2 = this.canvas.height;
        var ctx = this.ctx;
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, W, H2);
        if (this.mode === "draw") {
            if (this.drawPath.length < 2 && !this.isDrawing) {
                ctx.fillStyle = "#888";
                ctx.font = "16px monospace";
                ctx.textAlign = "center";
                ctx.fillText("Draw your path on this canvas", W / 2, H2 / 2);
                ctx.fillText("(click and drag, then press Play)", W / 2, H2 / 2 + 24);
            }
            if (this.drawPath.length > 1) {
                ctx.strokeStyle = "rgba(255,255,100,0.25)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(this.drawPath[0].x, this.drawPath[0].y);
                for (var i = 1; i < this.drawPath.length; i++) {
                    ctx.lineTo(this.drawPath[i].x, this.drawPath[i].y);
                }
                ctx.stroke();
            }
        }
        else {
            var steps = 120;
            ctx.beginPath();
            ctx.setLineDash([4, 6]);
            ctx.strokeStyle = "rgba(255,255,255,0.12)";
            ctx.lineWidth = 1;
            for (var i = 0; i <= steps; i++) {
                var pt = this.groundTruth(i / steps * (Math.PI * 2 / this.speed));
                if (i === 0)
                    ctx.moveTo(pt.x, pt.y);
                else
                    ctx.lineTo(pt.x, pt.y);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }
        for (var i = 0; i < this.trail.length; i++) {
            var alpha = (i + 1) / this.trail.length;
            var tp = this.trail[i];
            if (i > 0) {
                var prev = this.trail[i - 1];
                ctx.beginPath();
                ctx.moveTo(prev.true_x, prev.true_y);
                ctx.lineTo(tp.true_x, tp.true_y);
                ctx.strokeStyle = "rgba(80,220,80," + (alpha * 0.7) + ")";
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
            ctx.beginPath();
            ctx.arc(tp.meas_x, tp.meas_y, 2.5, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(255,120,80," + (alpha * 0.6) + ")";
            ctx.fill();
            if (i > 0) {
                var prev = this.trail[i - 1];
                ctx.beginPath();
                ctx.moveTo(prev.est_x, prev.est_y);
                ctx.lineTo(tp.est_x, tp.est_y);
                ctx.strokeStyle = "rgba(100,180,255," + (alpha * 0.9) + ")";
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }
        if (this.trail.length > 0) {
            var last = this.trail[this.trail.length - 1];
            this.drawCovarEllipse(last.est_x, last.est_y);
            ctx.beginPath();
            ctx.arc(last.est_x, last.est_y, 5, 0, Math.PI * 2);
            ctx.fillStyle = "#64b4ff";
            ctx.fill();
            ctx.beginPath();
            ctx.arc(last.true_x, last.true_y, 5, 0, Math.PI * 2);
            ctx.fillStyle = "#50dc50";
            ctx.fill();
        }
        ctx.font = "13px monospace";
        ctx.textAlign = "left";
        var legendItems = [
            ["#50dc50", "True trajectory"],
            ["rgba(255,120,80,0.9)", "Noisy measurements"],
            ["#64b4ff", "Kalman estimate"],
            ["rgba(100,220,255,0.6)", "95% covariance ellipse"],
        ];
        legendItems.forEach(function (item, i) {
            ctx.fillStyle = item[0];
            ctx.fillRect(14, 14 + i * 20, 14, 3);
            ctx.fillStyle = "#aaa";
            ctx.fillText(item[1], 34, 22 + i * 20);
        });
        this.renderPlot();
    };
    KalmanApp.prototype.renderPlot = function () {
        var W = this.plotCanvas.width;
        var H = this.plotCanvas.height;
        var ctx = this.plotCtx;
        ctx.fillStyle = "#0d0d0d";
        ctx.fillRect(0, 0, W, H);
        if (this.trail.length < 2) {
            ctx.fillStyle = "#555";
            ctx.font = "12px monospace";
            ctx.textAlign = "center";
            ctx.fillText("x(t) will appear here", W / 2, H / 2);
            return;
        }
        var minX = Infinity, maxX = -Infinity;
        this.trail.forEach(function (tp) {
            if (tp.true_x < minX)
                minX = tp.true_x;
            if (tp.meas_x < minX)
                minX = tp.meas_x;
            if (tp.est_x < minX)
                minX = tp.est_x;
            if (tp.true_x > maxX)
                maxX = tp.true_x;
            if (tp.meas_x > maxX)
                maxX = tp.meas_x;
            if (tp.est_x > maxX)
                maxX = tp.est_x;
        });
        var pad = 20;
        var xRange = maxX - minX || 1;
        var n = this.trail.length;
        var toPlot = function (val, row) {
            var px = pad + (row / (n - 1)) * (W - 2 * pad);
            var py = pad + (1 - (val - minX) / xRange) * (H - 2 * pad);
            return [px, py];
        };
        var sigma = Math.sqrt(Math.max(0, mget(this.kalmanState.P, 0, 0)));
        ctx.beginPath();
        this.trail.forEach(function (tp, i) {
            var p = toPlot(tp.est_x + sigma * 2, i);
            if (i === 0)
                ctx.moveTo(p[0], p[1]);
            else
                ctx.lineTo(p[0], p[1]);
        });
        for (var i = n - 1; i >= 0; i--) {
            var p = toPlot(this.trail[i].est_x - sigma * 2, i);
            ctx.lineTo(p[0], p[1]);
        }
        ctx.closePath();
        ctx.fillStyle = "rgba(100,180,255,0.15)";
        ctx.fill();
        ctx.beginPath();
        this.trail.forEach(function (tp, i) {
            var p = toPlot(tp.true_x, i);
            if (i === 0)
                ctx.moveTo(p[0], p[1]);
            else
                ctx.lineTo(p[0], p[1]);
        });
        ctx.strokeStyle = "rgba(80,220,80,0.7)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        var plotCtx = ctx;
        this.trail.forEach(function (tp, i) {
            var p = toPlot(tp.meas_x, i);
            plotCtx.beginPath();
            plotCtx.arc(p[0], p[1], 1.5, 0, Math.PI * 2);
            plotCtx.fillStyle = "rgba(255,120,80,0.5)";
            plotCtx.fill();
        });
        ctx.beginPath();
        this.trail.forEach(function (tp, i) {
            var p = toPlot(tp.est_x, i);
            if (i === 0)
                ctx.moveTo(p[0], p[1]);
            else
                ctx.lineTo(p[0], p[1]);
        });
        ctx.strokeStyle = "rgba(100,180,255,0.9)";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "#555";
        ctx.font = "11px monospace";
        ctx.textAlign = "left";
        ctx.fillText("x-position over time  (blue=estimate, green=true, red=meas, band=±2σ)", 6, 11);
    };
    KalmanApp.prototype.loop = function () {
        var _this = this;
        this.step();
        this.render();
        if (this.playing) {
            this.animId = requestAnimationFrame(function () { return _this.loop(); });
        }
    };
    KalmanApp.prototype.bindControls = function () {
        var _this = this;
        this.rSlider = document.getElementById("rSlider");
        this.qSlider = document.getElementById("qSlider");
        this.speedSlider = document.getElementById("speedSlider");
        this.rLabel = document.getElementById("rLabel");
        this.qLabel = document.getElementById("qLabel");
        this.speedLabel = document.getElementById("speedLabel");
        this.modeSelect = document.getElementById("modeSelect");
        this.playBtn = document.getElementById("playBtn");
        this.stepBtn = document.getElementById("stepBtn");
        this.resetBtn = document.getElementById("resetBtn");
        this.rSlider.addEventListener("input", function () {
            _this.rNoise = parseFloat(_this.rSlider.value);
            _this.rLabel.textContent = _this.rNoise.toFixed(1);
        });
        this.qSlider.addEventListener("input", function () {
            _this.qNoise = parseFloat(_this.qSlider.value);
            _this.qLabel.textContent = _this.qNoise.toFixed(3);
        });
        this.speedSlider.addEventListener("input", function () {
            _this.speed = parseFloat(_this.speedSlider.value);
            _this.speedLabel.textContent = _this.speed.toFixed(2);
        });
        this.modeSelect.addEventListener("change", function () {
            _this.mode = _this.modeSelect.value;
            _this.playing = false;
            cancelAnimationFrame(_this.animId);
            _this.playBtn.textContent = "▶ Play";
            _this.drawPath = [];
            _this.drawPathIndex = 0;
            _this.reset();
        });
        this.playBtn.addEventListener("click", function () {
            if (_this.mode === "draw" && _this.drawPath.length < 2) {
                alert("Draw a path on the canvas first!");
                return;
            }
            _this.playing = !_this.playing;
            _this.playBtn.textContent = _this.playing ? "⏸ Pause" : "▶ Play";
            if (_this.playing)
                _this.loop();
        });
        this.stepBtn.addEventListener("click", function () {
            if (_this.mode === "draw" && _this.drawPath.length < 2)
                return;
            _this.playing = false;
            _this.playBtn.textContent = "▶ Play";
            _this.step();
            _this.render();
        });
        this.resetBtn.addEventListener("click", function () {
            _this.playing = false;
            cancelAnimationFrame(_this.animId);
            _this.playBtn.textContent = "▶ Play";
            _this.reset();
        });
    };
    KalmanApp.prototype.bindDrawEvents = function () {
        var _this = this;
        this.canvas.addEventListener("mousedown", function (e) {
            if (_this.mode !== "draw")
                return;
            _this.isDrawing = true;
            _this.drawPath = [];
            var rect = _this.canvas.getBoundingClientRect();
            _this.drawPath.push({ x: e.clientX - rect.left, y: e.clientY - rect.top });
        });
        this.canvas.addEventListener("mousemove", function (e) {
            if (!_this.isDrawing || _this.mode !== "draw")
                return;
            var rect = _this.canvas.getBoundingClientRect();
            _this.drawPath.push({ x: e.clientX - rect.left, y: e.clientY - rect.top });
            _this.render();
        });
        this.canvas.addEventListener("mouseup", function () {
            if (_this.mode !== "draw")
                return;
            _this.isDrawing = false;
            _this.drawPathIndex = 0;
            _this.reset();
            _this.render();
        });
        this.canvas.addEventListener("touchstart", function (e) {
            if (_this.mode !== "draw")
                return;
            e.preventDefault();
            _this.isDrawing = true;
            _this.drawPath = [];
            var rect = _this.canvas.getBoundingClientRect();
            var touch = e.touches[0];
            _this.drawPath.push({ x: touch.clientX - rect.left, y: touch.clientY - rect.top });
        });
        this.canvas.addEventListener("touchmove", function (e) {
            if (!_this.isDrawing || _this.mode !== "draw")
                return;
            e.preventDefault();
            var rect = _this.canvas.getBoundingClientRect();
            var touch = e.touches[0];
            _this.drawPath.push({ x: touch.clientX - rect.left, y: touch.clientY - rect.top });
            _this.render();
        });
        this.canvas.addEventListener("touchend", function () {
            if (_this.mode !== "draw")
                return;
            _this.isDrawing = false;
            _this.drawPathIndex = 0;
            _this.reset();
            _this.render();
        });
    };
    return KalmanApp;
}());
window.addEventListener("load", function () {
    new KalmanApp();
});

},{}]},{},[1]);
