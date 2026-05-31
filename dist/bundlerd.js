(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var PRESETS = [
    { name: 'coral', F: 0.0545, k: 0.062, Du: 0.2097, Dv: 0.1050 },
    { name: 'mitosis', F: 0.0367, k: 0.0649, Du: 0.2097, Dv: 0.1050 },
    { name: 'maze', F: 0.029, k: 0.057, Du: 0.2097, Dv: 0.1050 },
    { name: 'spots', F: 0.025, k: 0.060, Du: 0.2097, Dv: 0.1050 },
    { name: 'waves', F: 0.014, k: 0.054, Du: 0.2097, Dv: 0.1050 },
    { name: 'stripes', F: 0.022, k: 0.051, Du: 0.2097, Dv: 0.1050 },
];
var COLORMAPS = [
    {
        name: 'plasma',
        fn: function (t) {
            var r = Math.round(255 * Math.min(1, Math.max(0, 0.05 + 1.0 * t * t + 0.3 * t)));
            var g = Math.round(255 * Math.min(1, Math.max(0, 0.02 + 0.9 * t * (1 - t) * 2)));
            var b = Math.round(255 * Math.min(1, Math.max(0, 0.53 - 0.5 * t + 0.3 * t * t)));
            return "rgb(".concat(r, ",").concat(g, ",").concat(b, ")");
        }
    },
    {
        name: 'viridis',
        fn: function (t) {
            var r = Math.round(255 * Math.min(1, Math.max(0, 0.267 + 0.004 * t - 1.263 * t * t + 2.17 * t * t * t)));
            var g = Math.round(255 * Math.min(1, Math.max(0, 0.0 + 1.418 * t - 0.5 * t * t)));
            var b = Math.round(255 * Math.min(1, Math.max(0, 0.329 + 1.498 * t - 3.0 * t * t + 2.0 * t * t * t)));
            return "rgb(".concat(r, ",").concat(g, ",").concat(b, ")");
        }
    },
    {
        name: 'fire',
        fn: function (t) {
            var r = Math.round(255 * Math.min(1, Math.max(0, t * 2.5)));
            var g = Math.round(255 * Math.min(1, Math.max(0, t * t * 2.0)));
            var b = Math.round(255 * Math.min(1, Math.max(0, t * t * t * 3.0)));
            return "rgb(".concat(r, ",").concat(g, ",").concat(b, ")");
        }
    },
    {
        name: 'ice',
        fn: function (t) {
            var r = Math.round(255 * Math.min(1, Math.max(0, t * t)));
            var g = Math.round(255 * Math.min(1, Math.max(0, t * 1.2)));
            var b = Math.round(255 * Math.min(1, Math.max(0, 0.2 + t * 0.8)));
            return "rgb(".concat(r, ",").concat(g, ",").concat(b, ")");
        }
    },
];
function buildLUT(cm) {
    var lut = new Uint8Array(256 * 3);
    for (var i = 0; i < 256; i++) {
        var c = cm.fn(i / 255);
        var m = c.match(/rgb\((\d+),(\d+),(\d+)\)/);
        if (m) {
            lut[i * 3 + 0] = parseInt(m[1]);
            lut[i * 3 + 1] = parseInt(m[2]);
            lut[i * 3 + 2] = parseInt(m[3]);
        }
    }
    return lut;
}
var RDSim = (function () {
    function RDSim(W, H) {
        this.dt = 1.0;
        this.W = W;
        this.H = H;
        var n = W * H;
        this.U = new Float32Array(n);
        this.V = new Float32Array(n);
        this.U2 = new Float32Array(n);
        this.V2 = new Float32Array(n);
        this.F = PRESETS[0].F;
        this.k = PRESETS[0].k;
        this.Du = PRESETS[0].Du;
        this.Dv = PRESETS[0].Dv;
        this.reset();
    }
    RDSim.prototype.reset = function () {
        var n = this.W * this.H;
        for (var i = 0; i < n; i++) {
            this.U[i] = 1.0;
            this.V[i] = 0.0;
        }
        var blobCount = 8;
        var blobSize = Math.max(4, Math.floor(Math.min(this.W, this.H) / 12));
        for (var b = 0; b < blobCount; b++) {
            var cx = Math.floor(Math.random() * this.W);
            var cy = Math.floor(Math.random() * this.H);
            for (var dy = -blobSize; dy <= blobSize; dy++) {
                for (var dx = -blobSize; dx <= blobSize; dx++) {
                    var nx = (cx + dx + this.W) % this.W;
                    var ny = (cy + dy + this.H) % this.H;
                    this.U[ny * this.W + nx] = 0.5 + Math.random() * 0.1;
                    this.V[ny * this.W + nx] = 0.25 + Math.random() * 0.1;
                }
            }
        }
    };
    RDSim.prototype.paint = function (gx, gy, radius) {
        var r = Math.max(1, Math.round(radius));
        for (var dy = -r; dy <= r; dy++) {
            for (var dx = -r; dx <= r; dx++) {
                if (dx * dx + dy * dy <= r * r) {
                    var nx = (gx + dx + this.W) % this.W;
                    var ny = (gy + dy + this.H) % this.H;
                    var idx = ny * this.W + nx;
                    this.U[idx] = 0.5 + Math.random() * 0.1;
                    this.V[idx] = 0.25 + Math.random() * 0.1;
                }
            }
        }
    };
    RDSim.prototype.step = function (substeps) {
        var W = this.W, H = this.H;
        var U = this.U, V = this.V;
        var U2 = this.U2, V2 = this.V2;
        var F = this.F, k = this.k;
        var Du = this.Du, Dv = this.Dv;
        var dt = this.dt;
        for (var s = 0; s < substeps; s++) {
            for (var y = 0; y < H; y++) {
                var yn = (y - 1 + H) % H;
                var yp = (y + 1) % H;
                var rowY = y * W;
                var rowYn = yn * W;
                var rowYp = yp * W;
                for (var x = 0; x < W; x++) {
                    var xn = (x - 1 + W) % W;
                    var xp = (x + 1) % W;
                    var idx = rowY + x;
                    var u = U[idx];
                    var v = V[idx];
                    var lapU = U[rowY + xn] + U[rowY + xp] + U[rowYn + x] + U[rowYp + x] - 4.0 * u;
                    var lapV = V[rowY + xn] + V[rowY + xp] + V[rowYn + x] + V[rowYp + x] - 4.0 * v;
                    var uvv = u * v * v;
                    U2[idx] = u + dt * (Du * lapU - uvv + F * (1.0 - u));
                    V2[idx] = v + dt * (Dv * lapV + uvv - (F + k) * v);
                }
            }
            var tmpU = this.U;
            this.U = this.U2;
            this.U2 = tmpU;
            var tmpV = this.V;
            this.V = this.V2;
            this.V2 = tmpV;
        }
    };
    return RDSim;
}());
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
var presetSel = document.getElementById('preset');
var colormapSel = document.getElementById('colormap');
var inF = document.getElementById('inF');
var inK = document.getElementById('inK');
var inDu = document.getElementById('inDu');
var inDv = document.getElementById('inDv');
var inSpeed = document.getElementById('inSpeed');
var inBrush = document.getElementById('inBrush');
var inGridSize = document.getElementById('inGridSize');
var btnPlay = document.getElementById('btnPlay');
var btnStep = document.getElementById('btnStep');
var btnReset = document.getElementById('btnReset');
var lblF = document.getElementById('lblF');
var lblK = document.getElementById('lblK');
var lblDu = document.getElementById('lblDu');
var lblDv = document.getElementById('lblDv');
var lblSpeed = document.getElementById('lblSpeed');
var lblBrush = document.getElementById('lblBrush');
var lblGridSize = document.getElementById('lblGridSize');
var fpsEl = document.getElementById('fps');
PRESETS.forEach(function (p, i) {
    var opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = p.name;
    presetSel.appendChild(opt);
});
COLORMAPS.forEach(function (c, i) {
    var opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = c.name;
    colormapSel.appendChild(opt);
});
var gridSize = parseInt(inGridSize.value);
var sim = new RDSim(gridSize, gridSize);
var playing = true;
var substeps = parseInt(inSpeed.value);
var brushRadius = parseInt(inBrush.value);
var cmIndex = 0;
var lut = buildLUT(COLORMAPS[0]);
var imageData = ctx.createImageData(gridSize, gridSize);
function syncParamsFromSim() {
    inF.value = String(sim.F);
    inK.value = String(sim.k);
    inDu.value = String(sim.Du);
    inDv.value = String(sim.Dv);
    lblF.textContent = sim.F.toFixed(4);
    lblK.textContent = sim.k.toFixed(4);
    lblDu.textContent = sim.Du.toFixed(4);
    lblDv.textContent = sim.Dv.toFixed(4);
}
syncParamsFromSim();
function applyPreset(idx) {
    var p = PRESETS[idx];
    sim.F = p.F;
    sim.k = p.k;
    sim.Du = p.Du;
    sim.Dv = p.Dv;
    syncParamsFromSim();
}
presetSel.addEventListener('change', function () {
    applyPreset(parseInt(presetSel.value));
});
colormapSel.addEventListener('change', function () {
    cmIndex = parseInt(colormapSel.value);
    lut = buildLUT(COLORMAPS[cmIndex]);
});
inF.addEventListener('input', function () { sim.F = parseFloat(inF.value); lblF.textContent = sim.F.toFixed(4); });
inK.addEventListener('input', function () { sim.k = parseFloat(inK.value); lblK.textContent = sim.k.toFixed(4); });
inDu.addEventListener('input', function () { sim.Du = parseFloat(inDu.value); lblDu.textContent = sim.Du.toFixed(4); });
inDv.addEventListener('input', function () { sim.Dv = parseFloat(inDv.value); lblDv.textContent = sim.Dv.toFixed(4); });
inSpeed.addEventListener('input', function () {
    substeps = parseInt(inSpeed.value);
    lblSpeed.textContent = String(substeps);
});
inBrush.addEventListener('input', function () {
    brushRadius = parseInt(inBrush.value);
    lblBrush.textContent = String(brushRadius);
});
inGridSize.addEventListener('change', function () {
    gridSize = parseInt(inGridSize.value);
    lblGridSize.textContent = String(gridSize);
    var newSim = new RDSim(gridSize, gridSize);
    newSim.F = sim.F;
    newSim.k = sim.k;
    newSim.Du = sim.Du;
    newSim.Dv = sim.Dv;
    sim = newSim;
    imageData = ctx.createImageData(gridSize, gridSize);
});
btnPlay.addEventListener('click', function () {
    playing = !playing;
    btnPlay.textContent = playing ? 'Pause' : 'Play';
});
btnStep.addEventListener('click', function () {
    playing = false;
    btnPlay.textContent = 'Play';
    sim.step(1);
    render();
});
btnReset.addEventListener('click', function () {
    sim.reset();
    render();
});
var mouseDown = false;
function canvasToGrid(cx, cy) {
    var rect = canvas.getBoundingClientRect();
    var px = cx - rect.left;
    var py = cy - rect.top;
    var gx = Math.floor((px / rect.width) * gridSize);
    var gy = Math.floor((py / rect.height) * gridSize);
    return [gx, gy];
}
canvas.addEventListener('mousedown', function (e) {
    mouseDown = true;
    var _a = canvasToGrid(e.clientX, e.clientY), gx = _a[0], gy = _a[1];
    sim.paint(gx, gy, brushRadius);
});
canvas.addEventListener('mousemove', function (e) {
    if (!mouseDown)
        return;
    var _a = canvasToGrid(e.clientX, e.clientY), gx = _a[0], gy = _a[1];
    sim.paint(gx, gy, brushRadius);
});
canvas.addEventListener('mouseup', function () { mouseDown = false; });
canvas.addEventListener('mouseleave', function () { mouseDown = false; });
canvas.addEventListener('touchstart', function (e) {
    e.preventDefault();
    mouseDown = true;
    var t = e.touches[0];
    var _a = canvasToGrid(t.clientX, t.clientY), gx = _a[0], gy = _a[1];
    sim.paint(gx, gy, brushRadius);
}, { passive: false });
canvas.addEventListener('touchmove', function (e) {
    e.preventDefault();
    var t = e.touches[0];
    var _a = canvasToGrid(t.clientX, t.clientY), gx = _a[0], gy = _a[1];
    sim.paint(gx, gy, brushRadius);
}, { passive: false });
canvas.addEventListener('touchend', function () { mouseDown = false; });
function render() {
    var V = sim.V;
    var W = sim.W, H = sim.H;
    var data = imageData.data;
    for (var i = 0; i < W * H; i++) {
        var v = V[i];
        var idx8 = Math.min(255, Math.max(0, Math.round(v * 255 * 3.5)));
        var li = idx8 * 3;
        var d = i * 4;
        data[d + 0] = lut[li + 0];
        data[d + 1] = lut[li + 1];
        data[d + 2] = lut[li + 2];
        data[d + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}
function resizeCanvas() {
    canvas.width = gridSize;
    canvas.height = gridSize;
}
resizeCanvas();
var lastTime = 0;
var frameCount = 0;
var fpsTime = 0;
function loop(ts) {
    var dt = ts - lastTime;
    lastTime = ts;
    fpsTime += dt;
    frameCount++;
    if (fpsTime >= 500) {
        fpsEl.textContent = (frameCount / (fpsTime / 1000)).toFixed(1);
        fpsTime = 0;
        frameCount = 0;
    }
    if (playing) {
        sim.step(substeps);
    }
    render();
    requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

},{}]},{},[1]);
