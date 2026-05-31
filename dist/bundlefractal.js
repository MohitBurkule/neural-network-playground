(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
Object.defineProperty(exports, "__esModule", { value: true });
function lerp(a, b, t) {
    return a + (b - a) * t;
}
function lerpColor(c1, c2, t) {
    return [
        Math.round(lerp(c1[0], c2[0], t)),
        Math.round(lerp(c1[1], c2[1], t)),
        Math.round(lerp(c1[2], c2[2], t))
    ];
}
var PALETTES = {
    ultra: [
        [0, 7, 100], [32, 107, 203], [237, 255, 255], [255, 170, 0], [0, 2, 0]
    ],
    fire: [
        [0, 0, 0], [128, 0, 0], [255, 64, 0], [255, 200, 0], [255, 255, 200], [255, 255, 255]
    ],
    ice: [
        [0, 0, 30], [0, 50, 120], [0, 160, 220], [180, 230, 255], [255, 255, 255]
    ],
    gold: [
        [10, 5, 0], [80, 40, 0], [200, 130, 0], [255, 220, 80], [255, 255, 200], [255, 255, 255]
    ],
    electric: [
        [0, 0, 0], [20, 0, 60], [100, 0, 200], [0, 150, 255], [0, 255, 200], [255, 255, 255]
    ]
};
function paletteColor(t, name) {
    var stops = PALETTES[name];
    var n = stops.length - 1;
    var scaled = t * n;
    var idx = Math.min(Math.floor(scaled), n - 1);
    var frac = scaled - idx;
    return lerpColor(stops[idx], stops[idx + 1], frac);
}
function mandelbrotIter(cr, ci, maxIter) {
    var zr = 0, zi = 0, zr2 = 0, zi2 = 0;
    for (var i = 0; i < maxIter; i++) {
        zi = 2 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        zr2 = zr * zr;
        zi2 = zi * zi;
        if (zr2 + zi2 > 256) {
            var log2abs = Math.log(zr2 + zi2) * 0.5 / Math.LN2;
            return i + 1 - Math.log(log2abs) / Math.LN2;
        }
    }
    return -1;
}
function juliaIter(zr, zi, cr, ci, maxIter) {
    var zr2 = zr * zr, zi2 = zi * zi;
    for (var i = 0; i < maxIter; i++) {
        zi = 2 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        zr2 = zr * zr;
        zi2 = zi * zi;
        if (zr2 + zi2 > 256) {
            var log2abs = Math.log(zr2 + zi2) * 0.5 / Math.LN2;
            return i + 1 - Math.log(log2abs) / Math.LN2;
        }
    }
    return -1;
}
function renderFractal(canvas, view, palette, isJulia, juliaC, scale) {
    if (scale === void 0) { scale = 1; }
    var w = Math.floor(canvas.width * scale);
    var h = Math.floor(canvas.height * scale);
    var ctx = canvas.getContext('2d');
    var buf = new Uint8ClampedArray(w * h * 4);
    var cx = view.cx, cy = view.cy, zoom = view.zoom, maxIter = view.maxIter;
    var zs = zoom * scale;
    for (var py = 0; py < h; py++) {
        var ci = cy + (py - h / 2) / zs;
        for (var px = 0; px < w; px++) {
            var cr = cx + (px - w / 2) / zs;
            var t = void 0;
            if (isJulia) {
                t = juliaIter(cr, ci, juliaC[0], juliaC[1], maxIter);
            }
            else {
                t = mandelbrotIter(cr, ci, maxIter);
            }
            var idx = (py * w + px) * 4;
            if (t < 0) {
                buf[idx] = 0;
                buf[idx + 1] = 0;
                buf[idx + 2] = 0;
                buf[idx + 3] = 255;
            }
            else {
                var nt = (t % maxIter) / maxIter;
                var smoothT = Math.pow(nt, 0.5);
                var _a = paletteColor(smoothT, palette), r = _a[0], g = _a[1], b = _a[2];
                buf[idx] = r;
                buf[idx + 1] = g;
                buf[idx + 2] = b;
                buf[idx + 3] = 255;
            }
        }
    }
    var imgData = new ImageData(buf, w, h);
    if (scale === 1) {
        ctx.putImageData(imgData, 0, 0);
    }
    else {
        var offscreen = document.createElement('canvas');
        offscreen.width = w;
        offscreen.height = h;
        offscreen.getContext('2d').putImageData(imgData, 0, 0);
        ctx.save();
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
        ctx.restore();
    }
}
var DEFAULT_MANDELBROT = { cx: -0.5, cy: 0, zoom: 200, maxIter: 100 };
var DEFAULT_JULIA = { cx: 0, cy: 0, zoom: 200, maxIter: 100 };
function autoIter(zoom) {
    return Math.min(1000, Math.max(100, Math.floor(50 + Math.log2(zoom) * 15)));
}
function initApp() {
    var modeSelect = document.getElementById('mode-select');
    var paletteSelect = document.getElementById('palette-select');
    var maxIterInput = document.getElementById('max-iter');
    var maxIterDisplay = document.getElementById('max-iter-display');
    var resetBtn = document.getElementById('reset-btn');
    var infoDiv = document.getElementById('info');
    var juliaControls = document.getElementById('julia-controls');
    var cRealInput = document.getElementById('c-real');
    var cImagInput = document.getElementById('c-imag');
    var cRealDisplay = document.getElementById('c-real-display');
    var cImagDisplay = document.getElementById('c-imag-display');
    var mainCanvas = document.getElementById('main-canvas');
    var pickerCanvas = document.getElementById('picker-canvas');
    var pickerContainer = document.getElementById('picker-container');
    var mode = 'mandelbrot';
    var palette = 'ultra';
    var mainView = __assign({}, DEFAULT_MANDELBROT);
    var juliaView = __assign({}, DEFAULT_JULIA);
    var juliaC = [-0.7, 0.27];
    var pickerView = __assign({}, DEFAULT_MANDELBROT);
    var renderTimeout = null;
    var isDragging = false;
    var dragStart = null;
    var isBoxSelecting = false;
    var boxStart = null;
    var overlayCanvas = null;
    function getMode() { return mode; }
    function scheduleRender(fast) {
        if (fast === void 0) { fast = false; }
        if (renderTimeout !== null)
            clearTimeout(renderTimeout);
        if (fast) {
            doRender(0.25);
            renderTimeout = window.setTimeout(function () {
                doRender(0.5);
                renderTimeout = window.setTimeout(function () { return doRender(1); }, 150);
            }, 80);
        }
        else {
            doRender(1);
        }
    }
    function doRender(scale) {
        var m = getMode();
        if (m === 'mandelbrot') {
            renderFractal(mainCanvas, mainView, palette, false, juliaC, scale);
        }
        else if (m === 'julia') {
            renderFractal(mainCanvas, mainView, palette, true, juliaC, scale);
        }
        else {
            renderFractal(mainCanvas, mainView, palette, false, juliaC, scale);
            renderFractal(pickerCanvas, juliaView, palette, true, juliaC, scale);
        }
        updateInfo();
    }
    function updateInfo() {
        var v = (mode === 'julia') ? mainView : mainView;
        infoDiv.textContent =
            "center: (".concat(v.cx.toFixed(6), ", ").concat(v.cy.toFixed(6), ")  zoom: ").concat(Math.round(v.zoom), "  iters: ").concat(v.maxIter);
    }
    function pixelToComplex(canvas, px, py, view) {
        var rect = canvas.getBoundingClientRect();
        var x = (px - rect.left) * (canvas.width / rect.width);
        var y = (py - rect.top) * (canvas.height / rect.height);
        return [
            view.cx + (x - canvas.width / 2) / view.zoom,
            view.cy + (y - canvas.height / 2) / view.zoom
        ];
    }
    function onModeChange() {
        mode = modeSelect.value;
        var isJuliaRelated = mode === 'julia' || mode === 'linked';
        juliaControls.style.display = isJuliaRelated ? 'flex' : 'none';
        pickerContainer.style.display = mode === 'linked' ? 'flex' : 'none';
        if (mode === 'mandelbrot') {
            mainView = __assign({}, DEFAULT_MANDELBROT);
        }
        else if (mode === 'julia') {
            mainView = __assign({}, DEFAULT_JULIA);
        }
        else {
            mainView = __assign({}, DEFAULT_MANDELBROT);
            juliaView = __assign({}, DEFAULT_JULIA);
        }
        scheduleRender();
    }
    function zoomView(view, factor, pivotX, pivotY, canvas) {
        var _a = pixelToComplex(canvas, pivotX, pivotY, view), pr = _a[0], pi = _a[1];
        view.zoom *= factor;
        view.cx = pr - (pivotX - canvas.getBoundingClientRect().left) * (canvas.width / canvas.getBoundingClientRect().width) / view.zoom +
            canvas.width / 2 / view.zoom;
        view.cy = pi - (pivotY - canvas.getBoundingClientRect().top) * (canvas.height / canvas.getBoundingClientRect().height) / view.zoom +
            canvas.height / 2 / view.zoom;
        if (autoIterCheck()) {
            view.maxIter = autoIter(view.zoom);
            maxIterInput.value = String(view.maxIter);
            maxIterDisplay.textContent = String(view.maxIter);
        }
    }
    function zoomViewCentered(view, factor, pivotCr, pivotCi) {
        view.zoom *= factor;
        if (autoIterCheck()) {
            view.maxIter = autoIter(view.zoom);
            maxIterInput.value = String(view.maxIter);
            maxIterDisplay.textContent = String(view.maxIter);
        }
    }
    function autoIterCheck() {
        var _a, _b;
        return (_b = (_a = document.getElementById('auto-iter')) === null || _a === void 0 ? void 0 : _a.checked) !== null && _b !== void 0 ? _b : true;
    }
    mainCanvas.addEventListener('wheel', function (e) {
        e.preventDefault();
        var factor = e.deltaY < 0 ? 1.3 : 1 / 1.3;
        var v = mainView;
        var _a = pixelToComplex(mainCanvas, e.clientX, e.clientY, v), pr = _a[0], pi = _a[1];
        v.zoom *= factor;
        v.cx = pr - (e.clientX - mainCanvas.getBoundingClientRect().left) * (mainCanvas.width / mainCanvas.getBoundingClientRect().width - 0) / v.zoom + mainCanvas.width / (2 * v.zoom);
        v.cy = pi - (e.clientY - mainCanvas.getBoundingClientRect().top) * (mainCanvas.height / mainCanvas.getBoundingClientRect().height) / v.zoom + mainCanvas.height / (2 * v.zoom);
        if (autoIterCheck()) {
            v.maxIter = autoIter(v.zoom);
            maxIterInput.value = String(v.maxIter);
            maxIterDisplay.textContent = String(v.maxIter);
        }
        scheduleRender(true);
    }, { passive: false });
    mainCanvas.addEventListener('mousedown', function (e) {
        if (e.button === 0) {
            if (e.shiftKey) {
                isBoxSelecting = true;
                boxStart = { x: e.clientX, y: e.clientY };
                if (!overlayCanvas) {
                    overlayCanvas = document.createElement('canvas');
                    overlayCanvas.style.position = 'absolute';
                    overlayCanvas.style.pointerEvents = 'none';
                    overlayCanvas.style.top = mainCanvas.offsetTop + 'px';
                    overlayCanvas.style.left = mainCanvas.offsetLeft + 'px';
                    mainCanvas.parentElement.appendChild(overlayCanvas);
                }
                overlayCanvas.width = mainCanvas.width;
                overlayCanvas.height = mainCanvas.height;
                overlayCanvas.style.display = 'block';
            }
            else {
                isDragging = true;
                var rect = mainCanvas.getBoundingClientRect();
                dragStart = { x: e.clientX, y: e.clientY, cx: mainView.cx, cy: mainView.cy };
            }
        }
    });
    window.addEventListener('mousemove', function (e) {
        if (isDragging && dragStart) {
            var rect = mainCanvas.getBoundingClientRect();
            var dx = (e.clientX - dragStart.x) * (mainCanvas.width / rect.width);
            var dy = (e.clientY - dragStart.y) * (mainCanvas.height / rect.height);
            mainView.cx = dragStart.cx - dx / mainView.zoom;
            mainView.cy = dragStart.cy - dy / mainView.zoom;
            scheduleRender(true);
        }
        if (isBoxSelecting && boxStart && overlayCanvas) {
            var rect = mainCanvas.getBoundingClientRect();
            var ctx = overlayCanvas.getContext('2d');
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            var x1 = (boxStart.x - rect.left) * (mainCanvas.width / rect.width);
            var y1 = (boxStart.y - rect.top) * (mainCanvas.height / rect.height);
            var x2 = (e.clientX - rect.left) * (mainCanvas.width / rect.width);
            var y2 = (e.clientY - rect.top) * (mainCanvas.height / rect.height);
            ctx.strokeStyle = 'rgba(255,255,255,0.8)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.fillStyle = 'rgba(255,255,255,0.08)';
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        }
        if (mode === 'linked') {
            var rect = mainCanvas.getBoundingClientRect();
            if (e.clientX >= rect.left && e.clientX <= rect.right &&
                e.clientY >= rect.top && e.clientY <= rect.bottom) {
                var _a = pixelToComplex(mainCanvas, e.clientX, e.clientY, mainView), cr = _a[0], ci = _a[1];
                juliaC = [cr, ci];
                cRealInput.value = String(cr.toFixed(4));
                cImagInput.value = String(ci.toFixed(4));
                cRealDisplay.textContent = cr.toFixed(4);
                cImagDisplay.textContent = ci.toFixed(4);
                drawPickerCrosshair(e.clientX, e.clientY);
                renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 0.4);
                if (renderTimeout !== null)
                    clearTimeout(renderTimeout);
                renderTimeout = window.setTimeout(function () {
                    renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1);
                }, 120);
            }
        }
    });
    function drawPickerCrosshair(clientX, clientY) {
        var rect = mainCanvas.getBoundingClientRect();
        if (!overlayCanvas)
            return;
        overlayCanvas.width = mainCanvas.width;
        overlayCanvas.height = mainCanvas.height;
        overlayCanvas.style.display = 'block';
        var ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        var x = (clientX - rect.left) * (mainCanvas.width / rect.width);
        var y = (clientY - rect.top) * (mainCanvas.height / rect.height);
        ctx.strokeStyle = 'rgba(255,255,100,0.8)';
        ctx.lineWidth = 1;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(x - 10, y);
        ctx.lineTo(x + 10, y);
        ctx.moveTo(x, y - 10);
        ctx.lineTo(x, y + 10);
        ctx.stroke();
    }
    window.addEventListener('mouseup', function (e) {
        if (isDragging) {
            isDragging = false;
            dragStart = null;
            scheduleRender(false);
        }
        if (isBoxSelecting && boxStart) {
            isBoxSelecting = false;
            if (overlayCanvas) {
                overlayCanvas.style.display = 'none';
            }
            var rect = mainCanvas.getBoundingClientRect();
            var x1 = (boxStart.x - rect.left) * (mainCanvas.width / rect.width);
            var y1 = (boxStart.y - rect.top) * (mainCanvas.height / rect.height);
            var x2 = (e.clientX - rect.left) * (mainCanvas.width / rect.width);
            var y2 = (e.clientY - rect.top) * (mainCanvas.height / rect.height);
            var w = Math.abs(x2 - x1);
            var h = Math.abs(y2 - y1);
            if (w > 5 && h > 5) {
                var cx = mainView.cx + (Math.min(x1, x2) + w / 2 - mainCanvas.width / 2) / mainView.zoom;
                var cy = mainView.cy + (Math.min(y1, y2) + h / 2 - mainCanvas.height / 2) / mainView.zoom;
                var factor = Math.min(mainCanvas.width / w, mainCanvas.height / h);
                mainView.cx = cx;
                mainView.cy = cy;
                mainView.zoom *= factor;
                if (autoIterCheck()) {
                    mainView.maxIter = autoIter(mainView.zoom);
                    maxIterInput.value = String(mainView.maxIter);
                    maxIterDisplay.textContent = String(mainView.maxIter);
                }
                scheduleRender(true);
            }
            boxStart = null;
        }
    });
    mainCanvas.addEventListener('dblclick', function (e) {
        var _a = pixelToComplex(mainCanvas, e.clientX, e.clientY, mainView), pr = _a[0], pi = _a[1];
        mainView.cx = pr;
        mainView.cy = pi;
        mainView.zoom *= 2.5;
        if (autoIterCheck()) {
            mainView.maxIter = autoIter(mainView.zoom);
            maxIterInput.value = String(mainView.maxIter);
            maxIterDisplay.textContent = String(mainView.maxIter);
        }
        scheduleRender(true);
    });
    pickerCanvas.addEventListener('wheel', function (e) {
        e.preventDefault();
        var factor = e.deltaY < 0 ? 1.3 : 1 / 1.3;
        var _a = pixelToComplex(pickerCanvas, e.clientX, e.clientY, juliaView), pr = _a[0], pi = _a[1];
        juliaView.zoom *= factor;
        juliaView.cx = pr - (e.clientX - pickerCanvas.getBoundingClientRect().left) * (pickerCanvas.width / pickerCanvas.getBoundingClientRect().width) / juliaView.zoom + pickerCanvas.width / (2 * juliaView.zoom);
        juliaView.cy = pi - (e.clientY - pickerCanvas.getBoundingClientRect().top) * (pickerCanvas.height / pickerCanvas.getBoundingClientRect().height) / juliaView.zoom + pickerCanvas.height / (2 * juliaView.zoom);
        renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 0.25);
        if (renderTimeout !== null)
            clearTimeout(renderTimeout);
        renderTimeout = window.setTimeout(function () { return renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1); }, 200);
    }, { passive: false });
    var pickerDrag = null;
    pickerCanvas.addEventListener('mousedown', function (e) {
        pickerDrag = { x: e.clientX, y: e.clientY, cx: juliaView.cx, cy: juliaView.cy };
    });
    window.addEventListener('mousemove', function (e) {
        if (!pickerDrag)
            return;
        var rect = pickerCanvas.getBoundingClientRect();
        var dx = (e.clientX - pickerDrag.x) * (pickerCanvas.width / rect.width);
        var dy = (e.clientY - pickerDrag.y) * (pickerCanvas.height / rect.height);
        juliaView.cx = pickerDrag.cx - dx / juliaView.zoom;
        juliaView.cy = pickerDrag.cy - dy / juliaView.zoom;
        renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 0.25);
    });
    window.addEventListener('mouseup', function () {
        if (pickerDrag) {
            pickerDrag = null;
            renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1);
        }
    });
    modeSelect.addEventListener('change', onModeChange);
    paletteSelect.addEventListener('change', function () {
        palette = paletteSelect.value;
        scheduleRender(false);
        if (mode === 'linked')
            renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1);
    });
    maxIterInput.addEventListener('input', function () {
        var v = parseInt(maxIterInput.value, 10);
        mainView.maxIter = v;
        juliaView.maxIter = v;
        maxIterDisplay.textContent = String(v);
        scheduleRender(false);
    });
    cRealInput.addEventListener('input', function () {
        juliaC[0] = parseFloat(cRealInput.value);
        cRealDisplay.textContent = parseFloat(cRealInput.value).toFixed(4);
        scheduleRender(false);
        if (mode === 'linked')
            renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1);
    });
    cImagInput.addEventListener('input', function () {
        juliaC[1] = parseFloat(cImagInput.value);
        cImagDisplay.textContent = parseFloat(cImagInput.value).toFixed(4);
        scheduleRender(false);
        if (mode === 'linked')
            renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1);
    });
    resetBtn.addEventListener('click', function () {
        if (mode === 'mandelbrot') {
            mainView = __assign({}, DEFAULT_MANDELBROT);
        }
        else if (mode === 'julia') {
            mainView = __assign({}, DEFAULT_JULIA);
        }
        else {
            mainView = __assign({}, DEFAULT_MANDELBROT);
            juliaView = __assign({}, DEFAULT_JULIA);
        }
        scheduleRender(false);
        if (mode === 'linked')
            renderFractal(pickerCanvas, juliaView, palette, true, juliaC, 1);
    });
    function setupOverlay() {
        if (!overlayCanvas) {
            overlayCanvas = document.createElement('canvas');
            overlayCanvas.style.position = 'absolute';
            overlayCanvas.style.pointerEvents = 'none';
            overlayCanvas.style.display = 'none';
            mainCanvas.parentElement.style.position = 'relative';
            mainCanvas.parentElement.appendChild(overlayCanvas);
        }
        overlayCanvas.width = mainCanvas.width;
        overlayCanvas.height = mainCanvas.height;
        overlayCanvas.style.top = '0px';
        overlayCanvas.style.left = '0px';
    }
    setupOverlay();
    onModeChange();
}
document.addEventListener('DOMContentLoaded', initApp);

},{}]},{},[1]);
