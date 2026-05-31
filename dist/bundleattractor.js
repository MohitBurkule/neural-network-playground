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
var SYSTEMS = {
    Lorenz: {
        name: "Lorenz",
        params: [
            { label: "σ", min: 1, max: 20, value: 10 },
            { label: "ρ", min: 1, max: 60, value: 28 },
            { label: "β", min: 0.1, max: 8, value: 2.667 },
        ],
        deriv: function (p, _a) {
            var s = _a[0], r = _a[1], b = _a[2];
            return ({
                x: s * (p.y - p.x),
                y: p.x * (r - p.z) - p.y,
                z: p.x * p.y - b * p.z
            });
        },
        initPos: { x: 1, y: 1, z: 1 },
        scale: 14,
        offset: { x: 0, y: 0, z: -25 }
    },
    Rössler: {
        name: "Rössler",
        params: [
            { label: "a", min: 0.01, max: 0.5, value: 0.2 },
            { label: "b", min: 0.01, max: 0.5, value: 0.2 },
            { label: "c", min: 1, max: 15, value: 5.7 },
        ],
        deriv: function (p, _a) {
            var a = _a[0], b = _a[1], c = _a[2];
            return ({
                x: -(p.y + p.z),
                y: p.x + a * p.y,
                z: b + p.z * (p.x - c)
            });
        },
        initPos: { x: 1, y: 1, z: 1 },
        scale: 8,
        offset: { x: 0, y: 0, z: -5 }
    },
    Aizawa: {
        name: "Aizawa",
        params: [
            { label: "a", min: 0.5, max: 1.5, value: 0.95 },
            { label: "b", min: 0.1, max: 1.0, value: 0.7 },
            { label: "c", min: 0.1, max: 1.0, value: 0.6 },
            { label: "d", min: 1, max: 5, value: 3.5 },
            { label: "e", min: 0.1, max: 1.0, value: 0.25 },
            { label: "f", min: 0.01, max: 0.5, value: 0.1 },
        ],
        deriv: function (p, _a) {
            var a = _a[0], b = _a[1], c = _a[2], d = _a[3], e = _a[4], f = _a[5];
            return ({
                x: (p.z - b) * p.x - d * p.y,
                y: d * p.x + (p.z - b) * p.y,
                z: c + a * p.z - (p.z * p.z * p.z) / 3 - (p.x * p.x + p.y * p.y) * (1 + e * p.z) + f * p.z * p.x * p.x * p.x
            });
        },
        initPos: { x: 0.1, y: 0, z: 0 },
        scale: 70,
        offset: { x: 0, y: 0, z: 0 }
    },
    Thomas: {
        name: "Thomas",
        params: [
            { label: "b", min: 0.1, max: 0.5, value: 0.19 },
        ],
        deriv: function (p, _a) {
            var b = _a[0];
            return ({
                x: Math.sin(p.y) - b * p.x,
                y: Math.sin(p.z) - b * p.y,
                z: Math.sin(p.x) - b * p.z
            });
        },
        initPos: { x: 0.1, y: 0, z: 0 },
        scale: 18,
        offset: { x: 0, y: 0, z: 0 }
    },
    Halvorsen: {
        name: "Halvorsen",
        params: [
            { label: "a", min: 0.5, max: 3, value: 1.4 },
        ],
        deriv: function (p, _a) {
            var a = _a[0];
            return ({
                x: -a * p.x - 4 * p.y - 4 * p.z - p.y * p.y,
                y: -a * p.y - 4 * p.z - 4 * p.x - p.z * p.z,
                z: -a * p.z - 4 * p.x - 4 * p.y - p.x * p.x
            });
        },
        initPos: { x: -5, y: 0, z: 0 },
        scale: 8,
        offset: { x: 0, y: 0, z: 0 }
    },
    Chen: {
        name: "Chen",
        params: [
            { label: "a", min: 20, max: 50, value: 35 },
            { label: "b", min: 1, max: 10, value: 3 },
            { label: "c", min: 10, max: 30, value: 28 },
        ],
        deriv: function (p, _a) {
            var a = _a[0], b = _a[1], c = _a[2];
            return ({
                x: a * (p.y - p.x),
                y: (c - a) * p.x - p.x * p.z + c * p.y,
                z: p.x * p.y - b * p.z
            });
        },
        initPos: { x: -0.1, y: 0.5, z: -0.6 },
        scale: 5,
        offset: { x: 0, y: 0, z: -5 }
    }
};
function rk4(p, dt, deriv, params) {
    var k1 = deriv(p, params);
    var p2 = { x: p.x + k1.x * dt / 2, y: p.y + k1.y * dt / 2, z: p.z + k1.z * dt / 2 };
    var k2 = deriv(p2, params);
    var p3 = { x: p.x + k2.x * dt / 2, y: p.y + k2.y * dt / 2, z: p.z + k2.z * dt / 2 };
    var k3 = deriv(p3, params);
    var p4 = { x: p.x + k3.x * dt, y: p.y + k3.y * dt, z: p.z + k3.z * dt };
    var k4 = deriv(p4, params);
    return {
        x: p.x + (k1.x + 2 * k2.x + 2 * k3.x + k4.x) * dt / 6,
        y: p.y + (k1.y + 2 * k2.y + 2 * k3.y + k4.y) * dt / 6,
        z: p.z + (k1.z + 2 * k2.z + 2 * k3.z + k4.z) * dt / 6
    };
}
function project(p, rotX, rotY, scale, cx, cy) {
    var cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    var x1 = p.x * cosY + p.z * sinY;
    var z1 = -p.x * sinY + p.z * cosY;
    var cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    var y2 = p.y * cosX - z1 * sinX;
    var z2 = p.y * sinX + z1 * cosX;
    var depth = 1 / (1 + z2 * 0.01);
    return [cx + x1 * scale * depth, cy - y2 * scale * depth];
}
function speedColor(speed, alpha) {
    var h = Math.max(0, 240 - speed * 2);
    return "hsla(".concat(h, ",90%,60%,").concat(alpha.toFixed(3), ")");
}
function altColor(speed, alpha) {
    var h = Math.max(0, 60 - speed * 1.5);
    return "hsla(".concat(h, ",90%,65%,").concat(alpha.toFixed(3), ")");
}
function drawBifurcation(canvas) {
    var ctx = canvas.getContext("2d");
    var W = canvas.width, H = canvas.height;
    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, W, H);
    var rMin = 2.5, rMax = 4.0;
    var steps = W * 2;
    var warmup = 500, plot = 300;
    ctx.fillStyle = "rgba(100,200,255,0.35)";
    for (var i = 0; i <= steps; i++) {
        var r = rMin + (rMax - rMin) * i / steps;
        var x = 0.5;
        for (var j = 0; j < warmup; j++)
            x = r * x * (1 - x);
        for (var j = 0; j < plot; j++) {
            x = r * x * (1 - x);
            var px = (r - rMin) / (rMax - rMin) * W;
            var py = (1 - x) * H;
            ctx.fillRect(px, py, 1, 1);
        }
    }
    ctx.fillStyle = "#aaa";
    ctx.font = "13px monospace";
    ctx.fillText("Logistic Map Bifurcation  x→ r·x·(1−x)", 14, 22);
    ctx.fillText("r →", W - 40, H - 8);
    ctx.fillText("x", 6, H / 2);
}
(function main() {
    var root = document.getElementById("app");
    root.innerHTML = "\n    <div id=\"sidebar\">\n      <h2>Strange Attractors</h2>\n      <label>System</label>\n      <select id=\"sysSelect\">".concat(Object.keys(SYSTEMS).map(function (k) { return "<option value=\"".concat(k, "\">").concat(k, "</option>"); }).join(""), "<option value=\"Bifurcation\">Bifurcation (2D)</option></select>\n\n      <div id=\"paramSliders\"></div>\n\n      <hr/>\n      <label>Speed / dt <span id=\"dtVal\"></span></label>\n      <input type=\"range\" id=\"dtSlider\" min=\"0.1\" max=\"5\" step=\"0.05\" value=\"1\"/>\n\n      <label>Point budget <span id=\"budgetVal\"></span></label>\n      <input type=\"range\" id=\"budgetSlider\" min=\"500\" max=\"20000\" step=\"500\" value=\"6000\"/>\n\n      <hr/>\n      <label class=\"row\"><input type=\"checkbox\" id=\"autoRotate\" checked/> Auto-rotate</label>\n      <label class=\"row\"><input type=\"checkbox\" id=\"diverge\"/> Two-trajectory divergence</label>\n\n      <button id=\"resetBtn\">Reset</button>\n      <button id=\"bifBtn\">Bifurcation diagram</button>\n\n      <div id=\"info\">\n        <small>Drag to rotate \u2022 Scroll reserved</small><br/>\n        <small id=\"statsLine\"></small>\n      </div>\n    </div>\n    <canvas id=\"canvas\"></canvas>\n  ");
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    var sysSelect = document.getElementById("sysSelect");
    var paramDiv = document.getElementById("paramSliders");
    var dtSlider = document.getElementById("dtSlider");
    var dtVal = document.getElementById("dtVal");
    var budgetSlider = document.getElementById("budgetSlider");
    var budgetVal = document.getElementById("budgetVal");
    var autoRotateCb = document.getElementById("autoRotate");
    var divergeCb = document.getElementById("diverge");
    var resetBtn = document.getElementById("resetBtn");
    var bifBtn = document.getElementById("bifBtn");
    var statsLine = document.getElementById("statsLine");
    var currentKey = "Lorenz";
    var trail1 = [];
    var trail2 = [];
    var pos1 = { x: 0, y: 0, z: 0 };
    var pos2 = { x: 0, y: 0, z: 0 };
    var rotX = 0.3, rotY = 0;
    var dragging = false, lastMX = 0, lastMY = 0;
    var bifMode = false;
    var animId = 0;
    var frameCount = 0;
    function resize() {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        if (bifMode)
            drawBifurcation(canvas);
    }
    window.addEventListener("resize", resize);
    function buildParams(key) {
        paramDiv.innerHTML = "";
        if (!(key in SYSTEMS))
            return;
        var sys = SYSTEMS[key];
        sys.params.forEach(function (p, i) {
            var lbl = document.createElement("label");
            var span = document.createElement("span");
            span.textContent = "".concat(p.label, "  ");
            var valSpan = document.createElement("span");
            valSpan.id = "pv_".concat(i);
            valSpan.textContent = p.value.toFixed(3);
            lbl.appendChild(span);
            lbl.appendChild(valSpan);
            var slider = document.createElement("input");
            slider.type = "range";
            slider.min = String(p.min);
            slider.max = String(p.max);
            slider.step = String((p.max - p.min) / 200);
            slider.value = String(p.value);
            slider.id = "ps_".concat(i);
            slider.addEventListener("input", function () {
                p.value = parseFloat(slider.value);
                valSpan.textContent = p.value.toFixed(3);
                resetTrails();
            });
            paramDiv.appendChild(lbl);
            paramDiv.appendChild(slider);
        });
    }
    function getParams() {
        if (!(currentKey in SYSTEMS))
            return [];
        return SYSTEMS[currentKey].params.map(function (p) { return p.value; });
    }
    function resetTrails() {
        if (!(currentKey in SYSTEMS))
            return;
        var sys = SYSTEMS[currentKey];
        pos1 = __assign({}, sys.initPos);
        pos2 = __assign(__assign({}, sys.initPos), { x: sys.initPos.x + 1e-5 });
        trail1 = [];
        trail2 = [];
        frameCount = 0;
    }
    canvas.addEventListener("mousedown", function (e) { dragging = true; lastMX = e.clientX; lastMY = e.clientY; });
    window.addEventListener("mouseup", function () { dragging = false; });
    window.addEventListener("mousemove", function (e) {
        if (!dragging)
            return;
        rotY += (e.clientX - lastMX) * 0.005;
        rotX += (e.clientY - lastMY) * 0.005;
        lastMX = e.clientX;
        lastMY = e.clientY;
    });
    canvas.addEventListener("touchstart", function (e) {
        dragging = true;
        lastMX = e.touches[0].clientX;
        lastMY = e.touches[0].clientY;
    }, { passive: true });
    canvas.addEventListener("touchend", function () { dragging = false; });
    canvas.addEventListener("touchmove", function (e) {
        if (!dragging)
            return;
        rotY += (e.touches[0].clientX - lastMX) * 0.005;
        rotX += (e.touches[0].clientY - lastMY) * 0.005;
        lastMX = e.touches[0].clientX;
        lastMY = e.touches[0].clientY;
    }, { passive: true });
    sysSelect.addEventListener("change", function () {
        currentKey = sysSelect.value;
        bifMode = (currentKey === "Bifurcation");
        buildParams(currentKey);
        resetTrails();
        if (bifMode) {
            setTimeout(function () { resize(); }, 50);
        }
    });
    resetBtn.addEventListener("click", resetTrails);
    bifBtn.addEventListener("click", function () {
        sysSelect.value = "Bifurcation";
        currentKey = "Bifurcation";
        bifMode = true;
        buildParams("Bifurcation");
        setTimeout(function () { resize(); }, 50);
    });
    function step() {
        if (bifMode) {
            animId = requestAnimationFrame(step);
            return;
        }
        var sys = SYSTEMS[currentKey];
        if (!sys) {
            animId = requestAnimationFrame(step);
            return;
        }
        var params = getParams();
        var dtMult = parseFloat(dtSlider.value);
        var budget = parseInt(budgetSlider.value, 10);
        dtVal.textContent = (0.005 * dtMult).toFixed(4);
        budgetVal.textContent = String(budget);
        var dt = 0.005 * dtMult;
        var stepsPerFrame = Math.max(1, Math.floor(dtMult * 3));
        for (var s = 0; s < stepsPerFrame; s++) {
            var prev1 = __assign({}, pos1);
            pos1 = rk4(pos1, dt, sys.deriv, params);
            var dx1 = pos1.x - prev1.x, dy1 = pos1.y - prev1.y, dz1 = pos1.z - prev1.z;
            var spd1 = Math.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1) / dt;
            trail1.push(__assign(__assign({}, pos1), { speed: spd1, age: 0 }));
            if (divergeCb.checked) {
                var prev2 = __assign({}, pos2);
                pos2 = rk4(pos2, dt, sys.deriv, params);
                var dx2 = pos2.x - prev2.x, dy2 = pos2.y - prev2.y, dz2 = pos2.z - prev2.z;
                var spd2 = Math.sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2) / dt;
                trail2.push(__assign(__assign({}, pos2), { speed: spd2, age: 0 }));
            }
        }
        for (var i = 0; i < trail1.length; i++)
            trail1[i].age++;
        for (var i = 0; i < trail2.length; i++)
            trail2[i].age++;
        if (trail1.length > budget)
            trail1 = trail1.slice(trail1.length - budget);
        if (trail2.length > budget)
            trail2 = trail2.slice(trail2.length - budget);
        if (autoRotateCb.checked && !dragging) {
            rotY += 0.003;
        }
        frameCount++;
        draw();
        animId = requestAnimationFrame(step);
    }
    function draw() {
        var W = canvas.width, H = canvas.height;
        ctx.fillStyle = "rgba(10,10,15,0.25)";
        ctx.fillRect(0, 0, W, H);
        var sys = SYSTEMS[currentKey];
        if (!sys)
            return;
        var scale = sys.scale * Math.min(W, H) / 600;
        var cx = W / 2 + sys.offset.x * scale;
        var cy = H / 2 - sys.offset.y * scale;
        var len1 = trail1.length;
        for (var i = 1; i < len1; i++) {
            var t = i / len1;
            var alpha = Math.pow(t, 1.5) * 0.9;
            var pt = trail1[i];
            var _a = project(pt, rotX, rotY, scale, cx, cy), sx = _a[0], sy = _a[1];
            var _b = project(trail1[i - 1], rotX, rotY, scale, cx, cy), px = _b[0], py = _b[1];
            ctx.beginPath();
            ctx.moveTo(px, py);
            ctx.lineTo(sx, sy);
            ctx.strokeStyle = speedColor(pt.speed * 0.05, alpha);
            ctx.lineWidth = t * 1.5 + 0.3;
            ctx.stroke();
        }
        if (divergeCb.checked) {
            var len2 = trail2.length;
            for (var i = 1; i < len2; i++) {
                var t = i / len2;
                var alpha = Math.pow(t, 1.5) * 0.85;
                var pt = trail2[i];
                var _c = project(pt, rotX, rotY, scale, cx, cy), sx = _c[0], sy = _c[1];
                var _d = project(trail2[i - 1], rotX, rotY, scale, cx, cy), px = _d[0], py = _d[1];
                ctx.beginPath();
                ctx.moveTo(px, py);
                ctx.lineTo(sx, sy);
                ctx.strokeStyle = altColor(pt.speed * 0.05, alpha);
                ctx.lineWidth = t * 1.5 + 0.3;
                ctx.stroke();
            }
            if (trail1.length > 0 && trail2.length > 0) {
                var p1 = trail1[trail1.length - 1];
                var p2 = trail2[trail2.length - 1];
                var dist = Math.sqrt(Math.pow((p1.x - p2.x), 2) + Math.pow((p1.y - p2.y), 2) + Math.pow((p1.z - p2.z), 2));
                statsLine.textContent = "\u0394 separation = ".concat(dist.toFixed(4), "  (started at 1e-5)");
            }
        }
        else {
            statsLine.textContent = "pts: ".concat(trail1.length, "  frame: ").concat(frameCount);
        }
    }
    function drawAxes() {
    }
    void drawAxes;
    buildParams(currentKey);
    resetTrails();
    resize();
    cancelAnimationFrame(animId);
    animId = requestAnimationFrame(step);
})();

},{}]},{},[1]);
