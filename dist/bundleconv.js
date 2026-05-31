(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
Object.defineProperty(exports, "__esModule", { value: true });
var PRESETS_3 = [
    {
        name: "Identity",
        size: 3,
        values: [0, 0, 0, 0, 1, 0, 0, 0, 0],
        divisor: 1,
        offset: 0
    },
    {
        name: "Box Blur",
        size: 3,
        values: [1, 1, 1, 1, 1, 1, 1, 1, 1],
        divisor: 9,
        offset: 0
    },
    {
        name: "Gaussian Blur",
        size: 3,
        values: [1, 2, 1, 2, 4, 2, 1, 2, 1],
        divisor: 16,
        offset: 0
    },
    {
        name: "Sharpen",
        size: 3,
        values: [0, -1, 0, -1, 5, -1, 0, -1, 0],
        divisor: 1,
        offset: 0
    },
    {
        name: "Edge Detect (Laplacian)",
        size: 3,
        values: [0, 1, 0, 1, -4, 1, 0, 1, 0],
        divisor: 1,
        offset: 128
    },
    {
        name: "Sobel X",
        size: 3,
        values: [-1, 0, 1, -2, 0, 2, -1, 0, 1],
        divisor: 1,
        offset: 128
    },
    {
        name: "Sobel Y",
        size: 3,
        values: [-1, -2, -1, 0, 0, 0, 1, 2, 1],
        divisor: 1,
        offset: 128
    },
    {
        name: "Emboss",
        size: 3,
        values: [-2, -1, 0, -1, 1, 1, 0, 1, 2],
        divisor: 1,
        offset: 128
    },
    {
        name: "Outline",
        size: 3,
        values: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        divisor: 1,
        offset: 0
    },
    {
        name: "Motion Blur",
        size: 3,
        values: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        divisor: 3,
        offset: 0
    },
];
var PRESETS_5 = [
    {
        name: "Identity (5x5)",
        size: 5,
        values: [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0,
        ],
        divisor: 1,
        offset: 0
    },
    {
        name: "Box Blur (5x5)",
        size: 5,
        values: new Array(25).fill(1),
        divisor: 25,
        offset: 0
    },
    {
        name: "Gaussian Blur (5x5)",
        size: 5,
        values: [
            1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1,
            4, 6, 4, 1,
        ],
        divisor: 256,
        offset: 0
    },
    {
        name: "Edge Detect (5x5)",
        size: 5,
        values: [
            0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0,
            0, 0, -1, 0, 0,
        ],
        divisor: 1,
        offset: 128
    },
    {
        name: "Unsharp Mask (5x5)",
        size: 5,
        values: [
            -1, -4, -6, -4, -1, -4, -16, -24, -16, -4, -6, -24, 476, -24, -6, -4,
            -16, -24, -16, -4, -1, -4, -6, -4, -1,
        ],
        divisor: 256,
        offset: 0
    },
];
var IMG_SIZE = 128;
function makeGradient() {
    var d = new ImageData(IMG_SIZE, IMG_SIZE);
    for (var y = 0; y < IMG_SIZE; y++) {
        for (var x = 0; x < IMG_SIZE; x++) {
            var i = (y * IMG_SIZE + x) * 4;
            d.data[i] = Math.round((x / (IMG_SIZE - 1)) * 255);
            d.data[i + 1] = Math.round((y / (IMG_SIZE - 1)) * 255);
            d.data[i + 2] = 128;
            d.data[i + 3] = 255;
        }
    }
    return d;
}
function makeCheckerboard() {
    var d = new ImageData(IMG_SIZE, IMG_SIZE);
    for (var y = 0; y < IMG_SIZE; y++) {
        for (var x = 0; x < IMG_SIZE; x++) {
            var i = (y * IMG_SIZE + x) * 4;
            var v = (Math.floor(x / 8) + Math.floor(y / 8)) % 2 === 0 ? 240 : 20;
            d.data[i] = v;
            d.data[i + 1] = v;
            d.data[i + 2] = v;
            d.data[i + 3] = 255;
        }
    }
    return d;
}
function makeShapes() {
    var d = new ImageData(IMG_SIZE, IMG_SIZE);
    for (var i = 0; i < IMG_SIZE * IMG_SIZE * 4; i += 4) {
        d.data[i] = 20;
        d.data[i + 1] = 20;
        d.data[i + 2] = 30;
        d.data[i + 3] = 255;
    }
    var setP = function (x, y, r, g, b) {
        if (x < 0 || x >= IMG_SIZE || y < 0 || y >= IMG_SIZE)
            return;
        var i = (y * IMG_SIZE + x) * 4;
        d.data[i] = r;
        d.data[i + 1] = g;
        d.data[i + 2] = b;
        d.data[i + 3] = 255;
    };
    for (var y = 10; y < 40; y++)
        for (var x = 10; x < 55; x++)
            setP(x, y, 220, 220, 220);
    var cx = 88, cy = 55, cr = 28;
    for (var y = cy - cr; y <= cy + cr; y++) {
        for (var x = cx - cr; x <= cx + cr; x++) {
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= cr * cr)
                setP(x, y, 80, 160, 240);
        }
    }
    for (var t = 0; t <= 100; t++) {
        var f = t / 100;
        setP(Math.round(30 + f * 40), Math.round(110 - f * 50), 240, 180, 60);
        setP(Math.round(70 - f * 40), Math.round(60 + f * 50), 240, 180, 60);
        setP(Math.round(30 + f * 0), Math.round(110 + f * 0), 240, 180, 60);
    }
    for (var i = 0; i < IMG_SIZE; i++) {
        setP(i, Math.floor((i * 70) / IMG_SIZE) + 80, 200, 80, 200);
    }
    return d;
}
function makeTextLike() {
    var d = new ImageData(IMG_SIZE, IMG_SIZE);
    for (var i = 0; i < IMG_SIZE * IMG_SIZE * 4; i += 4) {
        d.data[i] = 245;
        d.data[i + 1] = 245;
        d.data[i + 2] = 245;
        d.data[i + 3] = 255;
    }
    var setP = function (x, y, v) {
        if (x < 0 || x >= IMG_SIZE || y < 0 || y >= IMG_SIZE)
            return;
        var i = (y * IMG_SIZE + x) * 4;
        d.data[i] = v;
        d.data[i + 1] = v;
        d.data[i + 2] = v;
        d.data[i + 3] = 255;
    };
    var rows = [10, 20, 30, 42, 54, 66, 78, 90, 102, 114];
    for (var _i = 0, rows_1 = rows; _i < rows_1.length; _i++) {
        var row = rows_1[_i];
        var x = 6;
        while (x < IMG_SIZE - 6) {
            var len = 4 + Math.floor(((x * 31 + row * 17) % 7) * 1.5);
            for (var dx = 0; dx < len && x + dx < IMG_SIZE - 6; dx++) {
                setP(x + dx, row, 30);
                setP(x + dx, row + 1, 30);
            }
            x += len + 2 + ((x + row) % 4);
        }
    }
    return d;
}
function makeSyntheticPhoto() {
    var d = new ImageData(IMG_SIZE, IMG_SIZE);
    for (var y = 0; y < IMG_SIZE; y++) {
        for (var x = 0; x < IMG_SIZE; x++) {
            var i = (y * IMG_SIZE + x) * 4;
            if (y < 60) {
                d.data[i] = Math.round(50 + (y / 60) * 80);
                d.data[i + 1] = Math.round(100 + (y / 60) * 60);
                d.data[i + 2] = Math.round(200 - (y / 60) * 40);
            }
            else {
                d.data[i] = Math.round(60 + ((y - 60) / 68) * 30);
                d.data[i + 1] = Math.round(90 + ((y - 60) / 68) * 20);
                d.data[i + 2] = 40;
            }
            d.data[i + 3] = 255;
        }
    }
    var sx = 90, sy = 20, sr = 14;
    for (var y = sy - sr; y <= sy + sr; y++) {
        for (var x = sx - sr; x <= sx + sr; x++) {
            if (x >= 0 &&
                x < IMG_SIZE &&
                y >= 0 &&
                y < IMG_SIZE &&
                (x - sx) * (x - sx) + (y - sy) * (y - sy) <= sr * sr) {
                var i = (y * IMG_SIZE + x) * 4;
                d.data[i] = 255;
                d.data[i + 1] = 230;
                d.data[i + 2] = 80;
            }
        }
    }
    for (var y = 75; y < 115; y++) {
        for (var x = 30; x < 70; x++) {
            var i = (y * IMG_SIZE + x) * 4;
            if (y === 75 || y === 114 || x === 30 || x === 69) {
                d.data[i] = 180;
                d.data[i + 1] = 140;
                d.data[i + 2] = 90;
            }
            else {
                d.data[i] = 220;
                d.data[i + 1] = 190;
                d.data[i + 2] = 140;
            }
        }
    }
    for (var t = 0; t <= 40; t++) {
        var x1 = 30 - t, x2 = 69 + t, y = 74 - t;
        var mx = 49;
        if (y >= 45) {
            for (var x = Math.min(x1, mx); x <= Math.max(x2, mx); x++) {
                if (x >= x1 &&
                    x <= x2 &&
                    x >= 0 &&
                    x < IMG_SIZE &&
                    y >= 0 &&
                    y < IMG_SIZE) {
                    var xi = x >= x1 && x <= mx ? x1 + ((x - x1) * 2) / 1 : x2;
                    var i = (y * IMG_SIZE + x) * 4;
                    d.data[i] = 180;
                    d.data[i + 1] = 60;
                    d.data[i + 2] = 60;
                }
            }
        }
    }
    for (var t = 0; t < 30; t++) {
        var y = 74 - t;
        var xl = 30 - t, xr = 69 + t;
        for (var x = xl; x <= xr; x++) {
            if (x >= 0 && x < IMG_SIZE && y >= 0 && y < IMG_SIZE) {
                var i = (y * IMG_SIZE + x) * 4;
                if (d.data[i + 3] === 255 && d.data[i] > 100) {
                    d.data[i] = 190;
                    d.data[i + 1] = 70;
                    d.data[i + 2] = 70;
                }
            }
        }
    }
    return d;
}
var IMAGE_GENERATORS = [
    { name: "Gradient", fn: makeGradient },
    { name: "Checkerboard", fn: makeCheckerboard },
    { name: "Shapes", fn: makeShapes },
    { name: "Text-like", fn: makeTextLike },
    { name: "Synthetic Photo", fn: makeSyntheticPhoto },
];
function applyKernel(src, kernel, kSize, divisor, offset, grayscale) {
    var w = src.width, h = src.height;
    var out = new ImageData(w, h);
    var half = Math.floor(kSize / 2);
    for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
            var r = 0, g = 0, b = 0;
            for (var ky = 0; ky < kSize; ky++) {
                for (var kx = 0; kx < kSize; kx++) {
                    var sy = Math.min(h - 1, Math.max(0, y + ky - half));
                    var sx = Math.min(w - 1, Math.max(0, x + kx - half));
                    var si = (sy * w + sx) * 4;
                    var kv = kernel[ky * kSize + kx];
                    if (grayscale) {
                        var gray = 0.299 * src.data[si] +
                            0.587 * src.data[si + 1] +
                            0.114 * src.data[si + 2];
                        r += gray * kv;
                        g += gray * kv;
                        b += gray * kv;
                    }
                    else {
                        r += src.data[si] * kv;
                        g += src.data[si + 1] * kv;
                        b += src.data[si + 2] * kv;
                    }
                }
            }
            var div = divisor === 0 ? 1 : divisor;
            var oi = (y * w + x) * 4;
            out.data[oi] = Math.min(255, Math.max(0, Math.round(r / div + offset)));
            out.data[oi + 1] = Math.min(255, Math.max(0, Math.round(g / div + offset)));
            out.data[oi + 2] = Math.min(255, Math.max(0, Math.round(b / div + offset)));
            out.data[oi + 3] = 255;
        }
    }
    return out;
}
function computePixel(src, px, py, kernel, kSize, divisor, offset, grayscale) {
    var w = src.width, h = src.height;
    var half = Math.floor(kSize / 2);
    var r = 0, g = 0, b = 0;
    var products = [];
    for (var ky = 0; ky < kSize; ky++) {
        for (var kx = 0; kx < kSize; kx++) {
            var sy = Math.min(h - 1, Math.max(0, py + ky - half));
            var sx = Math.min(w - 1, Math.max(0, px + kx - half));
            var si = (sy * w + sx) * 4;
            var kv = kernel[ky * kSize + kx];
            if (grayscale) {
                var gray = 0.299 * src.data[si] +
                    0.587 * src.data[si + 1] +
                    0.114 * src.data[si + 2];
                r += gray * kv;
                g += gray * kv;
                b += gray * kv;
                products.push(Math.round(gray * kv));
            }
            else {
                var rv = src.data[si];
                r += rv * kv;
                products.push(Math.round(rv * kv));
                g += src.data[si + 1] * kv;
                b += src.data[si + 2] * kv;
            }
        }
    }
    var div = divisor === 0 ? 1 : divisor;
    return {
        r: Math.min(255, Math.max(0, Math.round(r / div + offset))),
        g: Math.min(255, Math.max(0, Math.round(g / div + offset))),
        b: Math.min(255, Math.max(0, Math.round(b / div + offset))),
        products: products
    };
}
function main() {
    var state = {
        kernelSize: 3,
        kernelValues: __spreadArray([], PRESETS_3[0].values, true),
        divisor: PRESETS_3[0].divisor,
        offset: PRESETS_3[0].offset,
        imageIndex: 0,
        grayscale: false,
        showWindow: true,
        animating: true,
        animFrame: null,
        animX: 30,
        animY: 30,
        animDirX: 1,
        animDirY: 1,
        uploadedImageData: null,
        usingUpload: false
    };
    var inputCanvas = document.getElementById("inputCanvas");
    var outputCanvas = document.getElementById("outputCanvas");
    var inCtx = inputCanvas.getContext("2d");
    var outCtx = outputCanvas.getContext("2d");
    var kernelGridEl = document.getElementById("kernelGrid");
    var divisorInput = document.getElementById("divisorInput");
    var offsetInput = document.getElementById("offsetInput");
    var presetSelect = document.getElementById("presetSelect");
    var imageSel = document.getElementById("imageSel");
    var grayscaleCb = document.getElementById("grayscaleCb");
    var showWindowCb = document.getElementById("showWindowCb");
    var animateCb = document.getElementById("animateCb");
    var kernelSizeRadios = document.querySelectorAll('input[name="ksize"]');
    var fileInput = document.getElementById("fileInput");
    var pixelInfoEl = document.getElementById("pixelInfo");
    var productsEl = document.getElementById("productsDisplay");
    inputCanvas.width = IMG_SIZE;
    inputCanvas.height = IMG_SIZE;
    outputCanvas.width = IMG_SIZE;
    outputCanvas.height = IMG_SIZE;
    var imageCache = new Array(IMAGE_GENERATORS.length).fill(null);
    var outputImageData = null;
    function getSourceImage() {
        if (state.usingUpload && state.uploadedImageData) {
            return state.uploadedImageData;
        }
        var idx = state.imageIndex;
        if (!imageCache[idx]) {
            imageCache[idx] = IMAGE_GENERATORS[idx].fn();
        }
        return imageCache[idx];
    }
    var kernelInputEls = [];
    function buildKernelGrid() {
        var _a;
        kernelGridEl.innerHTML = "";
        kernelGridEl.style.gridTemplateColumns = "repeat(".concat(state.kernelSize, ", 1fr)");
        kernelInputEls = [];
        var _loop_1 = function (i) {
            var inp = document.createElement("input");
            inp.type = "number";
            inp.className = "kernel-cell";
            inp.value = String((_a = state.kernelValues[i]) !== null && _a !== void 0 ? _a : 0);
            inp.step = "0.01";
            inp.addEventListener("input", function () {
                state.kernelValues[i] = parseFloat(inp.value) || 0;
                renderOutput();
            });
            kernelGridEl.appendChild(inp);
            kernelInputEls.push(inp);
        };
        for (var i = 0; i < state.kernelSize * state.kernelSize; i++) {
            _loop_1(i);
        }
    }
    function syncKernelInputs() {
        var _a;
        for (var i = 0; i < kernelInputEls.length; i++) {
            kernelInputEls[i].value = String((_a = state.kernelValues[i]) !== null && _a !== void 0 ? _a : 0);
        }
    }
    function renderInput() {
        var src = getSourceImage();
        inCtx.putImageData(src, 0, 0);
    }
    function renderOutput() {
        var src = getSourceImage();
        outputImageData = applyKernel(src, state.kernelValues, state.kernelSize, state.divisor, state.offset, state.grayscale);
        outCtx.putImageData(outputImageData, 0, 0);
        if (state.showWindow) {
            drawSlideWindow();
        }
    }
    function drawSlideWindow() {
        var src = getSourceImage();
        var px = state.animX;
        var py = state.animY;
        var half = Math.floor(state.kernelSize / 2);
        var scale = inputCanvas.clientWidth / IMG_SIZE || 1;
        inCtx.putImageData(src, 0, 0);
        var nx = (px - half) * scale;
        var ny = (py - half) * scale;
        var nw = state.kernelSize * scale;
        inCtx.save();
        inCtx.strokeStyle = "rgba(255, 220, 50, 0.9)";
        inCtx.lineWidth = 2;
        inCtx.strokeRect(px - half - 0.5, py - half - 0.5, state.kernelSize, state.kernelSize);
        inCtx.strokeStyle = "rgba(255,80,80,0.9)";
        inCtx.lineWidth = 1.5;
        inCtx.strokeRect(px - 0.5, py - 0.5, 1, 1);
        inCtx.restore();
        if (outputImageData) {
            outCtx.putImageData(outputImageData, 0, 0);
        }
        outCtx.save();
        outCtx.strokeStyle = "rgba(255, 80, 80, 0.9)";
        outCtx.lineWidth = 2;
        outCtx.strokeRect(px - 0.5, py - 0.5, 1, 1);
        outCtx.restore();
        var _a = computePixel(src, px, py, state.kernelValues, state.kernelSize, state.divisor, state.offset, state.grayscale), r = _a.r, g = _a.g, b = _a.b, products = _a.products;
        pixelInfoEl.innerHTML =
            "<span class=\"info-label\">Pixel (".concat(px, ", ").concat(py, ")</span> ") +
                "&rarr; <span class=\"out-val\">R:".concat(r, " G:").concat(g, " B:").concat(b, "</span>");
        var kSize = state.kernelSize;
        var html = "<div class=\"prod-grid\" style=\"grid-template-columns:repeat(".concat(kSize, ",1fr)\">");
        for (var i = 0; i < products.length; i++) {
            var v = products[i];
            var intensity = Math.min(1, Math.abs(v) / 255);
            var col = v >= 0 ? "rgba(80,200,120,".concat(0.3 + intensity * 0.7, ")") : "rgba(220,80,80,".concat(0.3 + intensity * 0.7, ")");
            html += "<div class=\"prod-cell\" style=\"background:".concat(col, "\">").concat(v > 999 ? "…" : v, "</div>");
        }
        html += "</div>";
        html += "<div class=\"prod-sum\">Sum = ".concat(products.reduce(function (a, b) { return a + b; }, 0), " &divide; ").concat(state.divisor, " + ").concat(state.offset, " &rarr; <b>").concat(r, "</b></div>");
        productsEl.innerHTML = html;
    }
    var lastAnimTime = 0;
    function animStep(ts) {
        if (!state.animating || !state.showWindow) {
            state.animFrame = null;
            return;
        }
        var elapsed = ts - lastAnimTime;
        if (elapsed > 60) {
            lastAnimTime = ts;
            state.animX += state.animDirX;
            state.animY += state.animDirY;
            var half = Math.floor(state.kernelSize / 2);
            if (state.animX <= half || state.animX >= IMG_SIZE - half - 1) {
                state.animDirX *= -1;
                state.animX += state.animDirX * 2;
            }
            if (state.animY <= half || state.animY >= IMG_SIZE - half - 1) {
                state.animDirY *= -1;
                state.animY += state.animDirY * 2;
            }
            drawSlideWindow();
        }
        state.animFrame = requestAnimationFrame(animStep);
    }
    function startAnim() {
        if (state.animFrame !== null)
            return;
        state.animFrame = requestAnimationFrame(animStep);
    }
    function stopAnim() {
        if (state.animFrame !== null) {
            cancelAnimationFrame(state.animFrame);
            state.animFrame = null;
        }
    }
    function populatePresets() {
        presetSelect.innerHTML = "";
        var presets = state.kernelSize === 3 ? PRESETS_3 : PRESETS_5;
        presets.forEach(function (p, i) {
            var opt = document.createElement("option");
            opt.value = String(i);
            opt.textContent = p.name;
            presetSelect.appendChild(opt);
        });
    }
    function applyPreset(idx) {
        var presets = state.kernelSize === 3 ? PRESETS_3 : PRESETS_5;
        var p = presets[idx];
        if (!p)
            return;
        state.kernelValues = __spreadArray([], p.values, true);
        state.divisor = p.divisor;
        state.offset = p.offset;
        divisorInput.value = String(p.divisor);
        offsetInput.value = String(p.offset);
        syncKernelInputs();
    }
    presetSelect.addEventListener("change", function () {
        applyPreset(parseInt(presetSelect.value));
        renderOutput();
    });
    imageSel.addEventListener("change", function () {
        state.imageIndex = parseInt(imageSel.value);
        state.usingUpload = false;
        renderInput();
        renderOutput();
    });
    grayscaleCb.addEventListener("change", function () {
        state.grayscale = grayscaleCb.checked;
        renderOutput();
    });
    showWindowCb.addEventListener("change", function () {
        state.showWindow = showWindowCb.checked;
        if (state.showWindow) {
            renderOutput();
            if (state.animating)
                startAnim();
        }
        else {
            stopAnim();
            renderInput();
            renderOutput();
            pixelInfoEl.innerHTML = "";
            productsEl.innerHTML = "";
        }
    });
    animateCb.addEventListener("change", function () {
        state.animating = animateCb.checked;
        if (state.animating && state.showWindow) {
            startAnim();
        }
        else {
            stopAnim();
        }
    });
    divisorInput.addEventListener("input", function () {
        state.divisor = parseFloat(divisorInput.value) || 1;
        renderOutput();
    });
    offsetInput.addEventListener("input", function () {
        state.offset = parseFloat(offsetInput.value) || 0;
        renderOutput();
    });
    kernelSizeRadios.forEach(function (r) {
        r.addEventListener("change", function () {
            if (r.checked) {
                state.kernelSize = parseInt(r.value);
                populatePresets();
                applyPreset(0);
                buildKernelGrid();
                renderOutput();
            }
        });
    });
    inputCanvas.addEventListener("click", function (e) {
        var rect = inputCanvas.getBoundingClientRect();
        var scaleX = IMG_SIZE / rect.width;
        var scaleY = IMG_SIZE / rect.height;
        var px = Math.round((e.clientX - rect.left) * scaleX);
        var py = Math.round((e.clientY - rect.top) * scaleY);
        var half = Math.floor(state.kernelSize / 2);
        state.animX = Math.min(IMG_SIZE - half - 1, Math.max(half, px));
        state.animY = Math.min(IMG_SIZE - half - 1, Math.max(half, py));
        if (!state.showWindow) {
            showWindowCb.checked = true;
            state.showWindow = true;
        }
        drawSlideWindow();
    });
    fileInput.addEventListener("change", function () {
        var _a;
        var file = (_a = fileInput.files) === null || _a === void 0 ? void 0 : _a[0];
        if (!file)
            return;
        var url = URL.createObjectURL(file);
        var img = new Image();
        img.onload = function () {
            var tmp = document.createElement("canvas");
            tmp.width = IMG_SIZE;
            tmp.height = IMG_SIZE;
            var tc = tmp.getContext("2d");
            tc.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);
            state.uploadedImageData = tc.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
            state.usingUpload = true;
            URL.revokeObjectURL(url);
            renderInput();
            renderOutput();
        };
        img.src = url;
    });
    populatePresets();
    buildKernelGrid();
    divisorInput.value = String(state.divisor);
    offsetInput.value = String(state.offset);
    renderInput();
    renderOutput();
    if (state.animating && state.showWindow)
        startAnim();
}
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
}
else {
    main();
}

},{}]},{},[1]);
