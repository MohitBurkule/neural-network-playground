(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var matA = [1, 0, 0, 1];
var matB = [1, 0, 0, 1];
var composeMode = false;
var animating = false;
var showEigen = true;
var showDetArea = true;
var showSVD = false;
var animT = 0;
var animDir = 1;
var animHandle = -1;
var lastTs = 0;
var ANIM_DUR = 1200;
var svg = document.getElementById('plane');
var inputA = Array.from({ length: 4 }, function (_, i) {
    return document.getElementById('ma' + i);
});
var inputB = Array.from({ length: 4 }, function (_, i) {
    return document.getElementById('mb' + i);
});
var detReadout = document.getElementById('det-readout');
var eigenReadout = document.getElementById('eigen-readout');
var svdReadout = document.getElementById('svd-readout');
var composePanel = document.getElementById('compose-panel');
var NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent) {
    if (attrs === void 0) { attrs = {}; }
    var e = document.createElementNS(NS, tag);
    for (var _i = 0, _a = Object.entries(attrs); _i < _a.length; _i++) {
        var _b = _a[_i], k = _b[0], v = _b[1];
        e.setAttribute(k, String(v));
    }
    if (parent)
        parent.appendChild(e);
    return e;
}
var W = 0, H = 0, CX = 0, CY = 0;
var GRID_CELLS = 8;
var SCALE = 50;
function resize() {
    var box = svg.parentElement.getBoundingClientRect();
    W = box.width;
    H = box.height;
    svg.setAttribute('width', W);
    svg.setAttribute('height', H);
    CX = W / 2;
    CY = H / 2;
    SCALE = Math.min(W, H) / (GRID_CELLS * 2 + 2) * 1.4;
    render();
}
function sx(x, y, m, t) {
    var _a = lerp2(m, t), a = _a[0], b = _a[1], d = _a[3];
    var c2 = lerp2(m, t)[2];
    return CX + (interp(1, a, t) * x + interp(0, b, t) * y) * SCALE;
}
function sy(x, y, m, t) {
    var lm = lerp2(m, t);
    return CY - (lm[2] * x + lm[3] * y) * SCALE;
}
function lerp2(m, t) {
    return [
        interp(1, m[0], t),
        interp(0, m[1], t),
        interp(0, m[2], t),
        interp(1, m[3], t),
    ];
}
function interp(a, b, t) {
    return a + (b - a) * t;
}
function det(m) { return m[0] * m[3] - m[1] * m[2]; }
function mul(a, b) {
    return [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ];
}
function eigenvalues(m) {
    var tr = m[0] + m[3];
    var d = det(m);
    var disc = tr * tr - 4 * d;
    if (disc < 0)
        return { real: false };
    var sq = Math.sqrt(disc);
    return { real: true, l1: (tr + sq) / 2, l2: (tr - sq) / 2 };
}
function eigenvector(m, lam) {
    var a = m[0] - lam, b = m[1];
    var c = m[2], d = m[3] - lam;
    if (Math.abs(b) > 1e-9)
        return [b, lam - m[0]];
    if (Math.abs(c) > 1e-9)
        return [lam - m[3], c];
    if (Math.abs(a) > 1e-9)
        return [0, 1];
    return [1, 0];
}
function normalise(v) {
    var len = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
    if (len < 1e-12)
        return [1, 0];
    return [v[0] / len, v[1] / len];
}
function svd2(m) {
    var ata = [
        m[0] * m[0] + m[2] * m[2], m[0] * m[1] + m[2] * m[3],
        m[0] * m[1] + m[2] * m[3], m[1] * m[1] + m[3] * m[3],
    ];
    var ev = eigenvalues(ata);
    var l1 = 1, l2 = 0;
    if (ev.real) {
        l1 = ev.l1;
        l2 = ev.l2;
    }
    var s1 = Math.sqrt(Math.max(0, l1));
    var s2 = Math.sqrt(Math.max(0, l2));
    var v1 = normalise(eigenvector(ata, l1));
    var v2 = [-v1[1], v1[0]];
    var u1raw = s1 > 1e-9 ?
        [(m[0] * v1[0] + m[1] * v1[1]) / s1, (m[2] * v1[0] + m[3] * v1[1]) / s1] : [1, 0];
    var u2raw = s2 > 1e-9 ?
        [(m[0] * v2[0] + m[1] * v2[1]) / s2, (m[2] * v2[0] + m[3] * v2[1]) / s2] : [-u1raw[1], u1raw[0]];
    return { s1: s1, s2: s2, u1: normalise(u1raw), u2: normalise(u2raw), v1: v1, v2: v2 };
}
function render() {
    while (svg.firstChild)
        svg.removeChild(svg.firstChild);
    var effMat = composeMode ? mul(matB, matA) : matA;
    var t = animT;
    el('rect', { x: 0, y: 0, width: W, height: H, fill: '#0d1117' }, svg);
    drawDeterminantArea(effMat, t);
    drawGrid(effMat, t);
    drawSampleShape(effMat, t);
    if (showSVD)
        drawSVDEllipse(effMat, t);
    drawAxes(effMat, t);
    drawBasisVectors(effMat, t);
    if (showEigen)
        drawEigenvectors(effMat, t);
    drawLabels();
    updateReadouts(effMat);
}
function drawGrid(m, t) {
    var N = GRID_CELLS;
    var g = el('g', { opacity: '0.3' }, svg);
    for (var i = -N; i <= N; i++) {
        var x1 = sx(i, -N, m, t), y1 = sy(i, -N, m, t);
        var x2 = sx(i, N, m, t), y2 = sy(i, N, m, t);
        el('line', { x1: x1, y1: y1, x2: x2, y2: y2, stroke: i === 0 ? '#8b949e' : '#30363d', 'stroke-width': i === 0 ? 1.5 : 0.8 }, g);
        var x3 = sx(-N, i, m, t), y3 = sy(-N, i, m, t);
        var x4 = sx(N, i, m, t), y4 = sy(N, i, m, t);
        el('line', { x1: x3, y1: y3, x2: x4, y2: y4, stroke: i === 0 ? '#8b949e' : '#30363d', 'stroke-width': i === 0 ? 1.5 : 0.8 }, g);
    }
}
function drawAxes(m, t) {
    var g = el('g', {}, svg);
    var N = GRID_CELLS + 1;
    el('line', {
        x1: sx(-N, 0, m, t), y1: sy(-N, 0, m, t),
        x2: sx(N, 0, m, t), y2: sy(N, 0, m, t),
        stroke: '#8b949e', 'stroke-width': 1.5, opacity: 0.5
    }, g);
    el('line', {
        x1: sx(0, -N, m, t), y1: sy(0, -N, m, t),
        x2: sx(0, N, m, t), y2: sy(0, N, m, t),
        stroke: '#8b949e', 'stroke-width': 1.5, opacity: 0.5
    }, g);
}
function drawBasisVectors(m, t) {
    var lm = lerp2(m, t);
    drawArrow(CX, CY, CX + lm[0] * SCALE, CY - lm[2] * SCALE, '#f85149', 'î', svg);
    drawArrow(CX, CY, CX + lm[1] * SCALE, CY - lm[3] * SCALE, '#3fb950', 'ĵ', svg);
}
function drawArrow(x1, y1, x2, y2, color, label, parent) {
    var g = el('g', {}, parent);
    var dx = x2 - x1, dy = y2 - y1;
    var len = Math.sqrt(dx * dx + dy * dy);
    if (len < 1)
        return;
    var ux = dx / len, uy = dy / len;
    var hs = 10;
    var ax = x2 - ux * hs, ay = y2 - uy * hs;
    var px = -uy, py = ux;
    el('line', { x1: x1, y1: y1, x2: ax, y2: ay, stroke: color, 'stroke-width': 2.5, 'stroke-linecap': 'round' }, g);
    var pts = "".concat(x2, ",").concat(y2, " ").concat(ax + px * 4, ",").concat(ay + py * 4, " ").concat(ax - px * 4, ",").concat(ay - py * 4);
    el('polygon', { points: pts, fill: color }, g);
    el('text', { x: x2 + ux * 14 + px * 6, y: y2 + uy * 14 + py * 6,
        fill: color, 'font-size': 14, 'font-weight': 'bold',
        'text-anchor': 'middle', 'dominant-baseline': 'middle' }, g).textContent = label;
}
function drawDeterminantArea(m, t) {
    if (!showDetArea)
        return;
    var lm = lerp2(m, t);
    var d = det(lm);
    var color = d >= 0 ? '#58a6ff' : '#f85149';
    var pts = [
        [CX, CY],
        [CX + lm[0] * SCALE, CY - lm[2] * SCALE],
        [CX + (lm[0] + lm[1]) * SCALE, CY - (lm[2] + lm[3]) * SCALE],
        [CX + lm[1] * SCALE, CY - lm[3] * SCALE],
    ].map(function (p) { return p.join(','); }).join(' ');
    el('polygon', { points: pts, fill: color, opacity: 0.15, stroke: color, 'stroke-width': 1, 'stroke-opacity': 0.5 }, svg);
}
var SHAPE_POINTS = [
    [1.5, 0], [2, 0.5], [2, 1], [1.5, 1.5], [1, 2], [0, 2],
    [-1, 1.5], [-1.5, 1], [-1.5, 0], [-1, -0.5], [0, -1], [1, -0.5],
];
function drawSampleShape(m, t) {
    var lm = lerp2(m, t);
    var tx = function (_a) {
        var x = _a[0], y = _a[1];
        return [
            CX + (lm[0] * x + lm[1] * y) * SCALE,
            CY - (lm[2] * x + lm[3] * y) * SCALE,
        ];
    };
    var pts = SHAPE_POINTS.map(function (p) { return tx(p).join(','); }).join(' ');
    el('polygon', { points: pts, fill: 'none', stroke: '#e3b341', 'stroke-width': 1.5, opacity: 0.7, 'stroke-linejoin': 'round' }, svg);
    for (var _i = 0, SHAPE_POINTS_1 = SHAPE_POINTS; _i < SHAPE_POINTS_1.length; _i++) {
        var p = SHAPE_POINTS_1[_i];
        var _a = tx(p), px = _a[0], py = _a[1];
        el('circle', { cx: px, cy: py, r: 2.5, fill: '#e3b341', opacity: 0.7 }, svg);
    }
}
function drawEigenvectors(m, t) {
    var lm = lerp2(m, t);
    var ev = eigenvalues(lm);
    if (!ev.real)
        return;
    for (var _i = 0, _a = [[ev.l1, '#d2a8ff'], [ev.l2, '#79c0ff']]; _i < _a.length; _i++) {
        var _b = _a[_i], lam = _b[0], color = _b[1];
        var v = normalise(eigenvector(lm, lam));
        var ext = GRID_CELLS * SCALE * 1.2;
        el('line', {
            x1: CX - v[0] * ext, y1: CY + v[1] * ext,
            x2: CX + v[0] * ext, y2: CY - v[1] * ext,
            stroke: color, 'stroke-width': 1.5, 'stroke-dasharray': '6 4', opacity: 0.8
        }, svg);
        var scale = Math.min(Math.abs(lam), 4) * Math.sign(lam);
        drawArrow(CX, CY, CX + v[0] * scale * SCALE, CY - v[1] * scale * SCALE, color, "\u03BB=".concat(lam.toFixed(2)), svg);
    }
}
function drawSVDEllipse(m, t) {
    var lm = lerp2(m, t);
    var _a = svd2(lm), s1 = _a.s1, s2 = _a.s2, u1 = _a.u1, u2 = _a.u2, v1 = _a.v1, v2 = _a.v2;
    el('circle', { cx: CX, cy: CY, r: SCALE, stroke: '#6e7681', 'stroke-width': 1,
        'stroke-dasharray': '4 4', fill: 'none', opacity: 0.6 }, svg);
    var pts = [];
    var N = 60;
    for (var i = 0; i <= N; i++) {
        var th = (i / N) * Math.PI * 2;
        var ux2 = v1[0] * Math.cos(th) + v2[0] * Math.sin(th);
        var uy2 = v1[1] * Math.cos(th) + v2[1] * Math.sin(th);
        var ex = lm[0] * ux2 + lm[1] * uy2;
        var ey = lm[2] * ux2 + lm[3] * uy2;
        pts.push("".concat(CX + ex * SCALE, ",").concat(CY - ey * SCALE));
    }
    el('polyline', { points: pts.join(' '), stroke: '#f0883e', 'stroke-width': 2, fill: 'none', opacity: 0.85 }, svg);
    drawArrow(CX, CY, CX + u1[0] * s1 * SCALE, CY - u1[1] * s1 * SCALE, '#f0883e', "\u03C3\u2081=".concat(s1.toFixed(2)), svg);
    if (s2 > 0.01) {
        drawArrow(CX, CY, CX + u2[0] * s2 * SCALE, CY - u2[1] * s2 * SCALE, '#ffa657', "\u03C3\u2082=".concat(s2.toFixed(2)), svg);
    }
}
function drawLabels() {
    var g = el('g', { 'font-size': '11', fill: '#8b949e' }, svg);
    el('text', { x: CX + GRID_CELLS * SCALE + 4, y: CY, 'dominant-baseline': 'middle' }, g).textContent = 'x';
    el('text', { x: CX + 4, y: CY - GRID_CELLS * SCALE - 4 }, g).textContent = 'y';
}
function updateReadouts(m) {
    var d = det(m);
    detReadout.textContent = "det = ".concat(d.toFixed(3));
    detReadout.style.color = d >= 0 ? '#58a6ff' : '#f85149';
    var ev = eigenvalues(m);
    if (ev.real) {
        eigenReadout.textContent = "\u03BB\u2081 = ".concat(ev.l1.toFixed(3), ",  \u03BB\u2082 = ").concat(ev.l2.toFixed(3));
        eigenReadout.style.color = '#d2a8ff';
    }
    else {
        var tr = m[0] + m[3];
        var disc = Math.abs((m[0] + m[3]) * (m[0] + m[3]) - 4 * det(m));
        var im = Math.sqrt(disc) / 2;
        eigenReadout.textContent = "\u03BB = ".concat((tr / 2).toFixed(2), " \u00B1 ").concat(im.toFixed(2), "i  (complex)");
        eigenReadout.style.color = '#8b949e';
    }
    var _a = svd2(m), s1 = _a.s1, s2 = _a.s2;
    svdReadout.textContent = "\u03C3\u2081=".concat(s1.toFixed(3), ", \u03C3\u2082=").concat(s2.toFixed(3), "  cond=").concat(s2 > 0.001 ? (s1 / s2).toFixed(2) : '∞');
}
function startAnim() {
    if (animHandle !== -1)
        cancelAnimationFrame(animHandle);
    lastTs = 0;
    animating = true;
    animHandle = requestAnimationFrame(frame);
}
function frame(ts) {
    if (!lastTs)
        lastTs = ts;
    var dt = ts - lastTs;
    lastTs = ts;
    animT = Math.max(0, Math.min(1, animT + animDir * dt / ANIM_DUR));
    render();
    if (animT > 0 && animT < 1) {
        animHandle = requestAnimationFrame(frame);
    }
    else {
        animating = false;
        animHandle = -1;
    }
}
function readMatrix(inputs) {
    return inputs.map(function (i) { return parseFloat(i.value) || 0; });
}
function writeMatrix(inputs, m) {
    inputs.forEach(function (inp, i) { inp.value = String(m[i]); });
}
function applyMatrix() {
    matA = readMatrix(inputA);
    animT = 0;
    animDir = 1;
    startAnim();
}
function resetMatrix() {
    matA = [1, 0, 0, 1];
    writeMatrix(inputA, matA);
    animT = 0;
    animDir = 1;
    startAnim();
}
function setPreset(name) {
    var presets = {
        identity: [1, 0, 0, 1],
        rotate90: [0, -1, 1, 0],
        rotate45: [Math.SQRT1_2, -Math.SQRT1_2, Math.SQRT1_2, Math.SQRT1_2],
        scale2: [2, 0, 0, 2],
        scalexy: [2, 0, 0, 0.5],
        shearx: [1, 1, 0, 1],
        sheary: [1, 0, 1, 1],
        reflectx: [1, 0, 0, -1],
        reflecty: [-1, 0, 0, 1],
        reflectdiag: [0, 1, 1, 0],
        projectx: [1, 0, 0, 0],
        rotscale: [1.5 * Math.SQRT1_2, -1.5 * Math.SQRT1_2, 1.5 * Math.SQRT1_2, 1.5 * Math.SQRT1_2],
        squeeze: [2, 0, 0, 0.5],
        singular: [1, 2, 2, 4]
    };
    var p = presets[name];
    if (!p)
        return;
    matA = p;
    writeMatrix(inputA, matA);
    animT = 0;
    animDir = 1;
    startAnim();
}
function init() {
    writeMatrix(inputA, matA);
    writeMatrix(inputB, matB);
    inputA.forEach(function (inp) { return inp.addEventListener('input', function () {
        matA = readMatrix(inputA);
        render();
    }); });
    inputB.forEach(function (inp) { return inp.addEventListener('input', function () {
        matB = readMatrix(inputB);
        render();
    }); });
    document.getElementById('btn-apply').addEventListener('click', applyMatrix);
    document.getElementById('btn-reset').addEventListener('click', resetMatrix);
    document.getElementById('btn-animate').addEventListener('click', function () {
        matA = readMatrix(inputA);
        if (animT >= 1) {
            animT = 1;
            animDir = -1;
        }
        else if (animT <= 0) {
            animT = 0;
            animDir = 1;
        }
        else {
            animDir = -animDir;
        }
        startAnim();
    });
    document.getElementById('btn-compose').addEventListener('click', function () {
        composeMode = !composeMode;
        composePanel.style.display = composeMode ? 'block' : 'none';
        document.getElementById('btn-compose').classList.toggle('active', composeMode);
        render();
    });
    document.getElementById('btn-eigen').addEventListener('click', function () {
        showEigen = !showEigen;
        document.getElementById('btn-eigen').classList.toggle('active', showEigen);
        render();
    });
    document.getElementById('btn-det').addEventListener('click', function () {
        showDetArea = !showDetArea;
        document.getElementById('btn-det').classList.toggle('active', showDetArea);
        render();
    });
    document.getElementById('btn-svd').addEventListener('click', function () {
        showSVD = !showSVD;
        document.getElementById('btn-svd').classList.toggle('active', showSVD);
        render();
    });
    document.getElementById('preset-select').addEventListener('change', function (e) {
        var v = e.target.value;
        if (v)
            setPreset(v);
    });
    setupDrag();
    window.addEventListener('resize', resize);
    resize();
}
function setupDrag() {
    var dragging = null;
    svg.addEventListener('mousedown', function (e) {
        var lm = lerp2(matA, animT);
        var tipI = [CX + lm[0] * SCALE, CY - lm[2] * SCALE];
        var tipJ = [CX + lm[1] * SCALE, CY - lm[3] * SCALE];
        var mx = e.offsetX, my = e.offsetY;
        if (dist(mx, my, tipI[0], tipI[1]) < 16)
            dragging = 'i';
        else if (dist(mx, my, tipJ[0], tipJ[1]) < 16)
            dragging = 'j';
    });
    svg.addEventListener('mousemove', function (e) {
        if (!dragging)
            return;
        var lx = (e.offsetX - CX) / SCALE;
        var ly = -(e.offsetY - CY) / SCALE;
        if (dragging === 'i') {
            matA[0] = +lx.toFixed(2);
            matA[2] = +ly.toFixed(2);
        }
        else {
            matA[1] = +lx.toFixed(2);
            matA[3] = +ly.toFixed(2);
        }
        writeMatrix(inputA, matA);
        animT = 1;
        render();
    });
    svg.addEventListener('mouseup', function () { dragging = null; });
    svg.addEventListener('mouseleave', function () { dragging = null; });
}
function dist(x1, y1, x2, y2) {
    return Math.sqrt(Math.pow((x1 - x2), 2) + Math.pow((y1 - y2), 2));
}
document.addEventListener('DOMContentLoaded', init);

},{}]},{},[1]);
