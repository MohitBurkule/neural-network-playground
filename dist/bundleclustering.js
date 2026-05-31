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
function randn() {
    var u = Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u + 1e-10)) * Math.cos(2 * Math.PI * v);
}
function randnPair(mx, my, sx, sy, rho) {
    var z1 = randn(), z2 = randn();
    var x = mx + sx * z1;
    var y = my + sy * (rho * z1 + Math.sqrt(1 - rho * rho) * z2);
    return [x, y];
}
function genBlobs(n) {
    var centers = [[-1.5, -1.5], [1.5, 1.5], [-1.5, 1.5], [1.5, -1.5]];
    var pts = [];
    for (var i = 0; i < n; i++) {
        var c = centers[i % centers.length];
        pts.push({ x: c[0] + randn() * 0.5, y: c[1] + randn() * 0.5, cluster: -1 });
    }
    return pts;
}
function genMoons(n) {
    var pts = [];
    for (var i = 0; i < n; i++) {
        var half = i < n / 2;
        var t = Math.PI * Math.random();
        if (half) {
            pts.push({ x: Math.cos(t) * 1.5, y: Math.sin(t) * 1.5 + randn() * 0.15, cluster: -1 });
        }
        else {
            pts.push({ x: 1.5 - Math.cos(t) * 1.5, y: -Math.sin(t) * 1.5 + 0.5 + randn() * 0.15, cluster: -1 });
        }
    }
    return pts;
}
function genCircles(n) {
    var pts = [];
    for (var i = 0; i < n; i++) {
        var inner = i < n / 2;
        var r = inner ? 0.6 + randn() * 0.08 : 1.5 + randn() * 0.08;
        var t = Math.random() * 2 * Math.PI;
        pts.push({ x: r * Math.cos(t), y: r * Math.sin(t), cluster: -1 });
    }
    return pts;
}
function genAniso(n) {
    var pts = [];
    var groups = [
        [-1.5, 0, 1.2, 0.15, 0.8],
        [1.5, 0, 1.2, 0.15, -0.8],
        [0, 1.5, 0.15, 1.2, 0.0],
    ];
    for (var i = 0; i < n; i++) {
        var g = groups[i % groups.length];
        var _a = randnPair(g[0], g[1], g[2], g[3], g[4]), x = _a[0], y = _a[1];
        pts.push({ x: x, y: y, cluster: -1 });
    }
    return pts;
}
function genUniformOutliers(n) {
    var pts = [];
    var nOutliers = Math.max(1, Math.floor(n * 0.1));
    var nBlob = n - nOutliers;
    var centers = [[-1, -1], [1, 1], [0, 1.5]];
    for (var i = 0; i < nBlob; i++) {
        var c = centers[i % centers.length];
        pts.push({ x: c[0] + randn() * 0.4, y: c[1] + randn() * 0.4, cluster: -1 });
    }
    for (var i = 0; i < nOutliers; i++) {
        pts.push({ x: (Math.random() - 0.5) * 5, y: (Math.random() - 0.5) * 5, cluster: -1 });
    }
    return pts;
}
function generateDataset(name, n) {
    switch (name) {
        case 'blobs': return genBlobs(n);
        case 'moons': return genMoons(n);
        case 'circles': return genCircles(n);
        case 'aniso': return genAniso(n);
        case 'uniform': return genUniformOutliers(n);
    }
}
function dist2(a, b) {
    return Math.pow((a.x - b.x), 2) + Math.pow((a.y - b.y), 2);
}
function kmeansInit(pts, k) {
    var centroids = [];
    var first = pts[Math.floor(Math.random() * pts.length)];
    centroids.push({ x: first.x, y: first.y, trail: [] });
    while (centroids.length < k) {
        var dists = pts.map(function (p) { return Math.min.apply(Math, centroids.map(function (c) { return dist2(p, c); })); });
        var sum = dists.reduce(function (a, b) { return a + b; }, 0);
        var r = Math.random() * sum;
        var idx = 0;
        for (var i = 0; i < dists.length; i++) {
            r -= dists[i];
            if (r <= 0) {
                idx = i;
                break;
            }
        }
        var chosen = pts[idx];
        centroids.push({ x: chosen.x, y: chosen.y, trail: [] });
    }
    return centroids;
}
function kmeansAssign(pts, centroids) {
    var changed = false;
    var _loop_1 = function (p) {
        var best = 0, bestD = Infinity;
        centroids.forEach(function (c, i) {
            var d = dist2(p, c);
            if (d < bestD) {
                bestD = d;
                best = i;
            }
        });
        if (p.cluster !== best) {
            changed = true;
            p.cluster = best;
        }
    };
    for (var _i = 0, pts_1 = pts; _i < pts_1.length; _i++) {
        var p = pts_1[_i];
        _loop_1(p);
    }
    return changed;
}
function kmeansUpdate(pts, centroids) {
    centroids.forEach(function (c, i) {
        var group = pts.filter(function (p) { return p.cluster === i; });
        if (group.length === 0)
            return;
        var nx = group.reduce(function (s, p) { return s + p.x; }, 0) / group.length;
        var ny = group.reduce(function (s, p) { return s + p.y; }, 0) / group.length;
        c.trail.push({ x: c.x, y: c.y });
        if (c.trail.length > 20)
            c.trail.shift();
        c.x = nx;
        c.y = ny;
    });
}
function kmeansInertia(pts, centroids) {
    return pts.reduce(function (s, p) { var _a; return s + dist2(p, (_a = centroids[p.cluster]) !== null && _a !== void 0 ? _a : centroids[0]); }, 0);
}
function dbscan(pts, eps, minPts) {
    var eps2 = eps * eps;
    pts.forEach(function (p) { p.cluster = -1; p.role = 'noise'; });
    function neighbors(idx) {
        return pts.reduce(function (acc, p, i) {
            if (dist2(pts[idx], p) <= eps2)
                acc.push(i);
            return acc;
        }, []);
    }
    var clusterId = 0;
    var visited = new Set();
    var _loop_2 = function (i) {
        if (visited.has(i))
            return "continue";
        visited.add(i);
        var nbrs = neighbors(i);
        if (nbrs.length < minPts) {
            pts[i].role = 'noise';
            return "continue";
        }
        pts[i].role = 'core';
        pts[i].cluster = clusterId;
        var queue = nbrs.filter(function (j) { return j !== i; });
        while (queue.length > 0) {
            var j = queue.shift();
            if (!visited.has(j)) {
                visited.add(j);
                var nbrs2 = neighbors(j);
                if (nbrs2.length >= minPts) {
                    pts[j].role = 'core';
                    queue.push.apply(queue, nbrs2.filter(function (x) { return !visited.has(x); }));
                }
                else {
                    pts[j].role = 'border';
                }
            }
            if (pts[j].cluster === -1)
                pts[j].cluster = clusterId;
        }
        clusterId++;
    };
    for (var i = 0; i < pts.length; i++) {
        _loop_2(i);
    }
}
function mat2Inv(m) {
    var det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    var d = det === 0 ? 1e-10 : det;
    return [
        [m[1][1] / d, -m[0][1] / d],
        [-m[1][0] / d, m[0][0] / d],
    ];
}
function mat2Det(m) {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}
function gaussian2D(x, y, mean, cov) {
    var inv = mat2Inv(cov);
    var det = mat2Det(cov);
    var dx = x - mean[0], dy = y - mean[1];
    var mah = inv[0][0] * dx * dx + (inv[0][1] + inv[1][0]) * dx * dy + inv[1][1] * dy * dy;
    var denom = 2 * Math.PI * Math.sqrt(Math.abs(det) + 1e-10);
    return Math.exp(-0.5 * mah) / denom;
}
function gmmInit(pts, k) {
    var shuffled = __spreadArray([], pts, true).sort(function () { return Math.random() - 0.5; });
    return shuffled.slice(0, k).map(function (p) { return ({
        mean: [p.x, p.y],
        cov: [[0.5, 0], [0, 0.5]],
        weight: 1 / k
    }); });
}
function gmmEStep(pts, components) {
    return pts.map(function (p) {
        var raw = components.map(function (c) { return c.weight * gaussian2D(p.x, p.y, c.mean, c.cov); });
        var sum = raw.reduce(function (a, b) { return a + b; }, 1e-300);
        return raw.map(function (r) { return r / sum; });
    });
}
function gmmMStep(pts, components, resp) {
    var N = pts.length;
    components.forEach(function (c, k) {
        var Nk = resp.reduce(function (s, r) { return s + r[k]; }, 0) + 1e-10;
        var mx = resp.reduce(function (s, r, i) { return s + r[k] * pts[i].x; }, 0) / Nk;
        var my = resp.reduce(function (s, r, i) { return s + r[k] * pts[i].y; }, 0) / Nk;
        c.mean = [mx, my];
        var c00 = 0, c01 = 0, c11 = 0;
        pts.forEach(function (p, i) {
            var dx = p.x - mx, dy = p.y - my;
            c00 += resp[i][k] * dx * dx;
            c01 += resp[i][k] * dx * dy;
            c11 += resp[i][k] * dy * dy;
        });
        c.cov = [
            [c00 / Nk + 1e-3, c01 / Nk],
            [c01 / Nk, c11 / Nk + 1e-3],
        ];
        c.weight = Nk / N;
    });
}
function gmmAssign(pts, resp) {
    pts.forEach(function (p, i) {
        p.cluster = resp[i].indexOf(Math.max.apply(Math, resp[i]));
    });
}
function silhouetteSample(pts, maxSample) {
    if (maxSample === void 0) { maxSample = 200; }
    var seenC = {};
    pts.forEach(function (p) { seenC[p.cluster] = true; });
    var clusters = Object.keys(seenC).map(Number).filter(function (c) { return c >= 0; });
    if (clusters.length < 2)
        return 0;
    var sample = pts.filter(function (p) { return p.cluster >= 0; })
        .sort(function () { return Math.random() - 0.5; })
        .slice(0, maxSample);
    var total = 0;
    var _loop_3 = function (p) {
        var intraD = sample.filter(function (q) { return q !== p && q.cluster === p.cluster; })
            .map(function (q) { return Math.sqrt(dist2(p, q)); });
        var a = intraD.length > 0 ? intraD.reduce(function (s, d) { return s + d; }, 0) / intraD.length : 0;
        var b = Infinity;
        var _loop_4 = function (c) {
            if (c === p.cluster)
                return "continue";
            var interD = sample.filter(function (q) { return q.cluster === c; })
                .map(function (q) { return Math.sqrt(dist2(p, q)); });
            if (interD.length > 0) {
                var mean = interD.reduce(function (s, d) { return s + d; }, 0) / interD.length;
                if (mean < b)
                    b = mean;
            }
        };
        for (var _a = 0, clusters_1 = clusters; _a < clusters_1.length; _a++) {
            var c = clusters_1[_a];
            _loop_4(c);
        }
        var s = b === Infinity ? 0 : (b - a) / Math.max(a, b);
        total += s;
    };
    for (var _i = 0, sample_1 = sample; _i < sample_1.length; _i++) {
        var p = sample_1[_i];
        _loop_3(p);
    }
    return total / sample.length;
}
var PALETTE = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
    '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4',
];
function clusterColor(id) {
    if (id < 0)
        return '#888888';
    return PALETTE[id % PALETTE.length];
}
function main() {
    var canvas = document.getElementById('canvas');
    var ctx = canvas.getContext('2d');
    var selAlgo = document.getElementById('sel-algo');
    var selData = document.getElementById('sel-data');
    var sliderK = document.getElementById('slider-k');
    var labelK = document.getElementById('label-k');
    var sliderEps = document.getElementById('slider-eps');
    var labelEps = document.getElementById('label-eps');
    var sliderMinPts = document.getElementById('slider-minpts');
    var labelMinPts = document.getElementById('label-minpts');
    var sliderN = document.getElementById('slider-n');
    var labelN = document.getElementById('label-n');
    var btnPlay = document.getElementById('btn-play');
    var btnPause = document.getElementById('btn-pause');
    var btnStep = document.getElementById('btn-step');
    var btnReset = document.getElementById('btn-reset');
    var spanIter = document.getElementById('span-iter');
    var spanMetric = document.getElementById('span-metric');
    var rowK = document.getElementById('row-k');
    var rowEps = document.getElementById('row-eps');
    var rowMinPts = document.getElementById('row-minpts');
    var legendDiv = document.getElementById('legend');
    var state = {
        pts: [],
        algorithm: 'kmeans',
        dataset: 'blobs',
        k: 3,
        eps: 0.5,
        minPts: 5,
        nPoints: 200,
        running: false,
        iter: 0,
        centroids: [],
        kmDone: false,
        dbDone: false,
        components: [],
        gmmResp: [],
        gmmPhase: 'E',
        gmmDone: false
    };
    var animId = -1;
    var lastStepTime = 0;
    var STEP_INTERVAL_MS = 600;
    function resizeCanvas() {
        var container = canvas.parentElement;
        var size = Math.min(container.clientWidth, container.clientHeight, 600);
        canvas.width = size;
        canvas.height = size;
    }
    resizeCanvas();
    window.addEventListener('resize', function () { resizeCanvas(); draw(); });
    var DATA_RANGE = 3.5;
    function toCanvasX(x) {
        return ((x + DATA_RANGE) / (2 * DATA_RANGE)) * canvas.width;
    }
    function toCanvasY(y) {
        return ((DATA_RANGE - y) / (2 * DATA_RANGE)) * canvas.height;
    }
    function updateParamVisibility() {
        var algo = state.algorithm;
        rowK.style.display = (algo === 'kmeans' || algo === 'gmm') ? 'block' : 'none';
        rowEps.style.display = algo === 'dbscan' ? 'block' : 'none';
        rowMinPts.style.display = algo === 'dbscan' ? 'block' : 'none';
    }
    function updateLegend() {
        legendDiv.innerHTML = '';
        if (state.algorithm === 'dbscan') {
            var roles = [
                { label: 'Core', color: '#3498db', shape: '●' },
                { label: 'Border', color: '#f39c12', shape: '◉' },
                { label: 'Noise', color: '#888', shape: '×' },
            ];
            roles.forEach(function (r) {
                var el = document.createElement('span');
                el.innerHTML = "<span style=\"color:".concat(r.color, ";font-size:1.2em\">").concat(r.shape, "</span> ").concat(r.label);
                el.className = 'legend-item';
                legendDiv.appendChild(el);
            });
            return;
        }
        var seenL = {};
        state.pts.forEach(function (p) { seenL[p.cluster] = true; });
        var clusters = Object.keys(seenL).map(Number).filter(function (c) { return c >= 0; }).sort(function (a, b) { return a - b; });
        clusters.forEach(function (c) {
            var el = document.createElement('span');
            el.innerHTML = "<span style=\"display:inline-block;width:12px;height:12px;border-radius:50%;background:".concat(clusterColor(c), ";vertical-align:middle;margin-right:4px\"></span>Cluster ").concat(c + 1);
            el.className = 'legend-item';
            legendDiv.appendChild(el);
        });
    }
    function resetState() {
        state.running = false;
        clearInterval(animId);
        state.iter = 0;
        state.kmDone = false;
        state.dbDone = false;
        state.gmmDone = false;
        state.gmmPhase = 'E';
        state.pts = generateDataset(state.dataset, state.nPoints);
        state.centroids = [];
        state.components = [];
        state.gmmResp = [];
        if (state.algorithm === 'kmeans') {
            state.centroids = kmeansInit(state.pts, state.k);
            kmeansAssign(state.pts, state.centroids);
        }
        else if (state.algorithm === 'gmm') {
            state.components = gmmInit(state.pts, state.k);
            state.gmmResp = gmmEStep(state.pts, state.components);
            gmmAssign(state.pts, state.gmmResp);
        }
        else {
        }
        updateStatus();
        updateLegend();
        draw();
    }
    function step() {
        if (state.algorithm === 'kmeans') {
            if (state.kmDone)
                return;
            kmeansUpdate(state.pts, state.centroids);
            var changed = kmeansAssign(state.pts, state.centroids);
            state.iter++;
            if (!changed) {
                state.kmDone = true;
                state.running = false;
            }
        }
        else if (state.algorithm === 'dbscan') {
            if (state.dbDone)
                return;
            dbscan(state.pts, state.eps, state.minPts);
            state.iter++;
            state.dbDone = true;
            state.running = false;
        }
        else if (state.algorithm === 'gmm') {
            if (state.gmmDone)
                return;
            if (state.gmmPhase === 'E') {
                state.gmmResp = gmmEStep(state.pts, state.components);
                gmmAssign(state.pts, state.gmmResp);
                state.gmmPhase = 'M';
            }
            else {
                gmmMStep(state.pts, state.components, state.gmmResp);
                state.gmmPhase = 'E';
                state.iter++;
                if (state.iter >= 100) {
                    state.gmmDone = true;
                    state.running = false;
                }
            }
        }
        updateStatus();
        updateLegend();
        draw();
    }
    function updateStatus() {
        spanIter.textContent = String(state.iter);
        if (state.algorithm === 'kmeans' && state.centroids.length > 0) {
            var inertia = kmeansInertia(state.pts, state.centroids).toFixed(1);
            spanMetric.textContent = "Inertia: ".concat(inertia);
        }
        else if (state.algorithm === 'dbscan' && state.dbDone) {
            var sil = silhouetteSample(state.pts).toFixed(3);
            spanMetric.textContent = "Silhouette: ".concat(sil);
        }
        else if (state.algorithm === 'gmm' && state.gmmResp.length > 0) {
            var sil = silhouetteSample(state.pts).toFixed(3);
            spanMetric.textContent = "Silhouette: ".concat(sil);
        }
        else {
            spanMetric.textContent = '—';
        }
    }
    function draw() {
        var W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth = 1;
        for (var gx = -3; gx <= 3; gx++) {
            var cx = toCanvasX(gx);
            ctx.beginPath();
            ctx.moveTo(cx, 0);
            ctx.lineTo(cx, H);
            ctx.stroke();
        }
        for (var gy = -3; gy <= 3; gy++) {
            var cy = toCanvasY(gy);
            ctx.beginPath();
            ctx.moveTo(0, cy);
            ctx.lineTo(W, cy);
            ctx.stroke();
        }
        if (state.algorithm === 'gmm' && state.components.length > 0) {
            drawGMMEllipses();
        }
        if (state.algorithm === 'kmeans') {
            state.centroids.forEach(function (c, i) {
                if (c.trail.length < 2)
                    return;
                ctx.strokeStyle = clusterColor(i);
                ctx.lineWidth = 1.5;
                ctx.setLineDash([3, 3]);
                ctx.globalAlpha = 0.4;
                ctx.beginPath();
                c.trail.forEach(function (t, ti) {
                    var tx = toCanvasX(t.x), ty = toCanvasY(t.y);
                    ti === 0 ? ctx.moveTo(tx, ty) : ctx.lineTo(tx, ty);
                });
                ctx.lineTo(toCanvasX(c.x), toCanvasY(c.y));
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalAlpha = 1;
            });
        }
        var r = Math.max(3, Math.min(6, 500 / state.pts.length));
        for (var _i = 0, _a = state.pts; _i < _a.length; _i++) {
            var p = _a[_i];
            var cx = toCanvasX(p.x), cy = toCanvasY(p.y);
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, 2 * Math.PI);
            if (state.algorithm === 'dbscan') {
                switch (p.role) {
                    case 'core':
                        ctx.fillStyle = clusterColor(p.cluster);
                        ctx.fill();
                        ctx.strokeStyle = 'rgba(255,255,255,0.6)';
                        ctx.lineWidth = 1;
                        ctx.stroke();
                        break;
                    case 'border':
                        ctx.fillStyle = clusterColor(p.cluster);
                        ctx.globalAlpha = 0.5;
                        ctx.fill();
                        ctx.globalAlpha = 1;
                        ctx.strokeStyle = clusterColor(p.cluster);
                        ctx.lineWidth = 1.5;
                        ctx.stroke();
                        break;
                    case 'noise':
                        ctx.fillStyle = '#555';
                        ctx.fill();
                        ctx.strokeStyle = '#999';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(cx - r, cy - r);
                        ctx.lineTo(cx + r, cy + r);
                        ctx.moveTo(cx + r, cy - r);
                        ctx.lineTo(cx - r, cy + r);
                        ctx.stroke();
                        break;
                }
            }
            else {
                ctx.fillStyle = clusterColor(p.cluster);
                ctx.fill();
                ctx.strokeStyle = 'rgba(0,0,0,0.3)';
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
        if (state.algorithm === 'kmeans') {
            state.centroids.forEach(function (c, i) {
                var cx = toCanvasX(c.x), cy = toCanvasY(c.y);
                ctx.beginPath();
                ctx.arc(cx, cy, 10, 0, 2 * Math.PI);
                ctx.fillStyle = clusterColor(i);
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(cx - 5, cy);
                ctx.lineTo(cx + 5, cy);
                ctx.moveTo(cx, cy - 5);
                ctx.lineTo(cx, cy + 5);
                ctx.stroke();
            });
        }
        if (state.algorithm === 'gmm' && state.components.length > 0) {
            state.components.forEach(function (c, i) {
                var cx = toCanvasX(c.mean[0]), cy = toCanvasY(c.mean[1]);
                ctx.beginPath();
                ctx.arc(cx, cy, 8, 0, 2 * Math.PI);
                ctx.fillStyle = clusterColor(i);
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
            });
        }
    }
    function drawGMMEllipses() {
        var nSigma = [1, 2];
        state.components.forEach(function (c, i) {
            var _a = eigenDecomp2x2(c.cov), angle = _a.angle, s1 = _a.s1, s2 = _a.s2;
            var cx = toCanvasX(c.mean[0]);
            var cy = toCanvasY(c.mean[1]);
            var scaleX = (2 * DATA_RANGE / canvas.width);
            var scaleY = (2 * DATA_RANGE / canvas.height);
            for (var _i = 0, nSigma_1 = nSigma; _i < nSigma_1.length; _i++) {
                var ns = nSigma_1[_i];
                var rx = Math.sqrt(s1) * ns / scaleX;
                var ry = Math.sqrt(s2) * ns / scaleY;
                ctx.save();
                ctx.translate(cx, cy);
                ctx.rotate(-angle);
                ctx.beginPath();
                ctx.ellipse(0, 0, rx, ry, 0, 0, 2 * Math.PI);
                ctx.restore();
                ctx.strokeStyle = clusterColor(i);
                ctx.lineWidth = ns === 1 ? 2 : 1;
                ctx.globalAlpha = ns === 1 ? 0.8 : 0.4;
                ctx.stroke();
                ctx.globalAlpha = 1;
            }
        });
    }
    function eigenDecomp2x2(m) {
        var a = m[0][0], b = m[0][1], d = m[1][1];
        var trace = a + d;
        var det = a * d - b * b;
        var disc = Math.sqrt(Math.max(0, Math.pow((trace / 2), 2) - det));
        var l1 = trace / 2 + disc;
        var l2 = trace / 2 - disc;
        var angle = 0;
        if (Math.abs(b) > 1e-10) {
            angle = Math.atan2(l1 - a, b);
        }
        return { angle: angle, s1: Math.max(l1, 1e-6), s2: Math.max(l2, 1e-6) };
    }
    function animLoop(now) {
        if (!state.running)
            return;
        if (now - lastStepTime >= STEP_INTERVAL_MS) {
            lastStepTime = now;
            step();
        }
        animId = requestAnimationFrame(animLoop);
    }
    selAlgo.addEventListener('change', function () {
        state.algorithm = selAlgo.value;
        updateParamVisibility();
        resetState();
    });
    selData.addEventListener('change', function () {
        state.dataset = selData.value;
        resetState();
    });
    sliderK.addEventListener('input', function () {
        state.k = parseInt(sliderK.value, 10);
        labelK.textContent = sliderK.value;
        resetState();
    });
    sliderEps.addEventListener('input', function () {
        state.eps = parseFloat(sliderEps.value);
        labelEps.textContent = sliderEps.value;
        resetState();
    });
    sliderMinPts.addEventListener('input', function () {
        state.minPts = parseInt(sliderMinPts.value, 10);
        labelMinPts.textContent = sliderMinPts.value;
        resetState();
    });
    sliderN.addEventListener('input', function () {
        state.nPoints = parseInt(sliderN.value, 10);
        labelN.textContent = sliderN.value;
        resetState();
    });
    btnPlay.addEventListener('click', function () {
        state.running = true;
        lastStepTime = 0;
        animId = requestAnimationFrame(animLoop);
    });
    btnPause.addEventListener('click', function () {
        state.running = false;
        cancelAnimationFrame(animId);
    });
    btnStep.addEventListener('click', function () {
        state.running = false;
        cancelAnimationFrame(animId);
        step();
    });
    btnReset.addEventListener('click', function () {
        resetState();
    });
    updateParamVisibility();
    resetState();
}
document.addEventListener('DOMContentLoaded', main);

},{}]},{},[1]);
