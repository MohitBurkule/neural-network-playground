(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
}
function randomInt(n) {
    return Math.floor(Math.random() * n);
}
var PATTERNS = [
    {
        name: "Glider",
        cells: [[0, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    },
    {
        name: "Pulsar",
        cells: [
            [0, 2], [0, 3], [0, 4], [0, 8], [0, 9], [0, 10],
            [2, 0], [2, 5], [2, 7], [2, 12],
            [3, 0], [3, 5], [3, 7], [3, 12],
            [4, 0], [4, 5], [4, 7], [4, 12],
            [5, 2], [5, 3], [5, 4], [5, 8], [5, 9], [5, 10],
            [7, 2], [7, 3], [7, 4], [7, 8], [7, 9], [7, 10],
            [8, 0], [8, 5], [8, 7], [8, 12],
            [9, 0], [9, 5], [9, 7], [9, 12],
            [10, 0], [10, 5], [10, 7], [10, 12],
            [12, 2], [12, 3], [12, 4], [12, 8], [12, 9], [12, 10]
        ]
    },
    {
        name: "Gosper Gun",
        cells: [
            [0, 24], [1, 22], [1, 24], [2, 12], [2, 13], [2, 20], [2, 21], [2, 34], [2, 35],
            [3, 11], [3, 15], [3, 20], [3, 21], [3, 34], [3, 35], [4, 0], [4, 1], [4, 10],
            [4, 16], [4, 20], [4, 21], [5, 0], [5, 1], [5, 10], [5, 14], [5, 16], [5, 17],
            [5, 22], [5, 24], [6, 10], [6, 16], [6, 24], [7, 11], [7, 15], [8, 12], [8, 13]
        ]
    },
    {
        name: "LWSS",
        cells: [
            [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 4], [2, 4], [3, 0], [3, 3]
        ]
    }
];
var GameOfLife = (function () {
    function GameOfLife(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.grid = new Uint8Array(rows * cols);
        this.next = new Uint8Array(rows * cols);
    }
    GameOfLife.prototype.idx = function (r, c) {
        return r * this.cols + c;
    };
    GameOfLife.prototype.get = function (r, c) {
        r = ((r % this.rows) + this.rows) % this.rows;
        c = ((c % this.cols) + this.cols) % this.cols;
        return this.grid[this.idx(r, c)];
    };
    GameOfLife.prototype.set = function (r, c, v) {
        r = ((r % this.rows) + this.rows) % this.rows;
        c = ((c % this.cols) + this.cols) % this.cols;
        this.grid[this.idx(r, c)] = v;
    };
    GameOfLife.prototype.clear = function () {
        this.grid.fill(0);
    };
    GameOfLife.prototype.randomize = function (density) {
        if (density === void 0) { density = 0.3; }
        for (var i = 0; i < this.grid.length; i++) {
            this.grid[i] = Math.random() < density ? 1 : 0;
        }
    };
    GameOfLife.prototype.placePattern = function (pattern, row, col) {
        pattern.cells.forEach(function (cell) {
            var r = cell[0];
            var c = cell[1];
            var tr = ((row + r) % this.rows + this.rows) % this.rows;
            var tc = ((col + c) % this.cols + this.cols) % this.cols;
            this.grid[this.idx(tr, tc)] = 1;
        }, this);
    };
    GameOfLife.prototype.population = function () {
        var count = 0;
        for (var i = 0; i < this.grid.length; i++) {
            count += this.grid[i];
        }
        return count;
    };
    GameOfLife.prototype.step = function () {
        var rows = this.rows;
        var cols = this.cols;
        for (var r = 0; r < rows; r++) {
            for (var c = 0; c < cols; c++) {
                var n = this.get(r - 1, c - 1) + this.get(r - 1, c) + this.get(r - 1, c + 1)
                    + this.get(r, c - 1) + this.get(r, c + 1)
                    + this.get(r + 1, c - 1) + this.get(r + 1, c) + this.get(r + 1, c + 1);
                var alive = this.grid[this.idx(r, c)];
                this.next[this.idx(r, c)] = (alive && (n === 2 || n === 3)) || (!alive && n === 3) ? 1 : 0;
            }
        }
        var tmp = this.grid;
        this.grid = this.next;
        this.next = tmp;
    };
    return GameOfLife;
}());
var ElementaryCA = (function () {
    function ElementaryCA(width, maxRows) {
        this.rule = 30;
        this.width = width;
        this.maxRows = maxRows;
        this.rows = [];
    }
    ElementaryCA.prototype.reset = function (singleCell) {
        this.rows = [];
        var first = new Uint8Array(this.width);
        if (singleCell) {
            first[Math.floor(this.width / 2)] = 1;
        }
        else {
            for (var i = 0; i < this.width; i++) {
                first[i] = Math.random() < 0.5 ? 1 : 0;
            }
        }
        this.rows.push(first);
    };
    ElementaryCA.prototype.computeNext = function () {
        if (this.rows.length >= this.maxRows)
            return;
        var prev = this.rows[this.rows.length - 1];
        var next = new Uint8Array(this.width);
        var w = this.width;
        var rule = this.rule;
        for (var i = 0; i < w; i++) {
            var l = prev[(i - 1 + w) % w];
            var m = prev[i];
            var r = prev[(i + 1) % w];
            var idx = (l << 2) | (m << 1) | r;
            next[i] = (rule >> idx) & 1;
        }
        this.rows.push(next);
    };
    ElementaryCA.prototype.fillAll = function () {
        while (this.rows.length < this.maxRows) {
            this.computeNext();
        }
    };
    return ElementaryCA;
}());
var CANVAS_W = 800;
var CANVAS_H = 600;
var LIFE_COLS = 80;
var LIFE_ROWS = 60;
var ECA_WIDTH = 200;
var ECA_MAX_ROWS = 150;
var COLOR_BG = "#0d1117";
var COLOR_CELL_LIFE = "#58a6ff";
var COLOR_CELL_DEAD = "#161b22";
var COLOR_GRID = "#21262d";
var COLOR_ECA_ON = "#e6c84a";
var COLOR_ECA_OFF = "#161b22";
var caState;
var caCanvas;
var caCtx;
var caAnimId = null;
var caLastTime = 0;
function initCAState() {
    var life = new GameOfLife(LIFE_ROWS, LIFE_COLS);
    life.randomize();
    var eca = new ElementaryCA(ECA_WIDTH, ECA_MAX_ROWS);
    eca.rule = 30;
    eca.reset(true);
    eca.fillAll();
    caState = {
        mode: "life",
        running: false,
        speed: 100,
        generation: 0,
        life: life,
        cellSize: Math.floor(Math.min(CANVAS_W / LIFE_COLS, CANVAS_H / LIFE_ROWS)),
        selectedPattern: 0,
        drawMode: false,
        drawValue: 1,
        eca: eca,
        ecaSingleCell: true,
        ecaRule: 30,
        ecaRunning: false
    };
}
function caDrawLife() {
    var s = caState.cellSize;
    var cols = caState.life.cols;
    var rows = caState.life.rows;
    var offX = Math.floor((CANVAS_W - cols * s) / 2);
    var offY = Math.floor((CANVAS_H - rows * s) / 2);
    caCtx.fillStyle = COLOR_BG;
    caCtx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
            var alive = caState.life.grid[caState.life.idx(r, c)];
            caCtx.fillStyle = alive ? COLOR_CELL_LIFE : COLOR_CELL_DEAD;
            caCtx.fillRect(offX + c * s, offY + r * s, s - 1, s - 1);
        }
    }
    caCtx.fillStyle = "#8b949e";
    caCtx.font = "12px monospace";
    caCtx.fillText("Gen " + caState.generation + "  Pop " + caState.life.population(), 8, 18);
}
function caDrawECA() {
    caCtx.fillStyle = COLOR_BG;
    caCtx.fillRect(0, 0, CANVAS_W, CANVAS_H);
    var eca = caState.eca;
    var numRows = eca.rows.length;
    var numCols = eca.width;
    var cellW = Math.floor(CANVAS_W / numCols);
    var cellH = Math.floor(CANVAS_H / ECA_MAX_ROWS);
    var offX = Math.floor((CANVAS_W - numCols * cellW) / 2);
    for (var r = 0; r < numRows; r++) {
        var row = eca.rows[r];
        for (var c = 0; c < numCols; c++) {
            caCtx.fillStyle = row[c] ? COLOR_ECA_ON : COLOR_ECA_OFF;
            caCtx.fillRect(offX + c * cellW, r * cellH, cellW, cellH);
        }
    }
    caCtx.fillStyle = "#8b949e";
    caCtx.font = "12px monospace";
    caCtx.fillText("Rule " + eca.rule + "  Gen " + numRows, 8, 18);
}
function caDraw() {
    if (caState.mode === "life") {
        caDrawLife();
    }
    else {
        caDrawECA();
    }
}
function caLoop(ts) {
    caAnimId = requestAnimationFrame(caLoop);
    if (!caState.running) {
        caDraw();
        return;
    }
    if (ts - caLastTime < caState.speed) {
        caDraw();
        return;
    }
    caLastTime = ts;
    if (caState.mode === "life") {
        caState.life.step();
        caState.generation++;
    }
    else {
        if (caState.eca.rows.length < caState.eca.maxRows) {
            caState.eca.computeNext();
        }
        else {
            caState.running = false;
            caUpdatePlayPauseButton();
        }
    }
    caDraw();
}
function caStartLoop() {
    if (caAnimId === null) {
        caAnimId = requestAnimationFrame(caLoop);
    }
}
function caUpdatePlayPauseButton() {
    var btn = document.getElementById("btn-playpause");
    if (btn)
        btn.textContent = caState.running ? "⏸ Pause" : "▶ Play";
}
function caUpdateModeUI() {
    var lifeControls = document.getElementById("life-controls");
    var ecaControls = document.getElementById("eca-controls");
    if (caState.mode === "life") {
        lifeControls.style.display = "flex";
        ecaControls.style.display = "none";
    }
    else {
        lifeControls.style.display = "none";
        ecaControls.style.display = "flex";
    }
}
function caCanvasToCell(mx, my) {
    var s = caState.cellSize;
    var offX = Math.floor((CANVAS_W - caState.life.cols * s) / 2);
    var offY = Math.floor((CANVAS_H - caState.life.rows * s) / 2);
    var c = Math.floor((mx - offX) / s);
    var r = Math.floor((my - offY) / s);
    if (r < 0 || r >= caState.life.rows || c < 0 || c >= caState.life.cols)
        return null;
    return [r, c];
}
function caBindUI() {
    var modeSelect = document.getElementById("mode-select");
    modeSelect.addEventListener("change", function () {
        caState.mode = modeSelect.value;
        caState.running = false;
        caUpdatePlayPauseButton();
        caUpdateModeUI();
        caDraw();
    });
    var btnPlay = document.getElementById("btn-playpause");
    btnPlay.addEventListener("click", function () {
        caState.running = !caState.running;
        caUpdatePlayPauseButton();
    });
    var btnStep = document.getElementById("btn-step");
    btnStep.addEventListener("click", function () {
        if (caState.mode === "life") {
            caState.life.step();
            caState.generation++;
        }
        else {
            caState.eca.computeNext();
        }
        caDraw();
    });
    var speedRange = document.getElementById("speed-range");
    speedRange.addEventListener("input", function () {
        var v = parseInt(speedRange.value, 10);
        caState.speed = Math.round(520 - v * 50);
        var label = document.getElementById("speed-label");
        if (label)
            label.textContent = v.toString();
    });
    var btnClear = document.getElementById("btn-clear");
    btnClear.addEventListener("click", function () {
        caState.life.clear();
        caState.generation = 0;
        caDraw();
    });
    var btnRandom = document.getElementById("btn-random");
    btnRandom.addEventListener("click", function () {
        caState.life.randomize();
        caState.generation = 0;
        caDraw();
    });
    var patternSel = document.getElementById("pattern-select");
    PATTERNS.forEach(function (p, i) {
        var opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = p.name;
        patternSel.appendChild(opt);
    });
    patternSel.addEventListener("change", function () {
        caState.selectedPattern = parseInt(patternSel.value, 10);
    });
    var btnPlace = document.getElementById("btn-place");
    btnPlace.addEventListener("click", function () {
        var pat = PATTERNS[caState.selectedPattern];
        caState.life.placePattern(pat, Math.floor(caState.life.rows / 2) - 5, Math.floor(caState.life.cols / 2) - 5);
        caDraw();
    });
    var gridSel = document.getElementById("grid-size");
    gridSel.addEventListener("change", function () {
        var val = gridSel.value.split("x");
        var cols = parseInt(val[0], 10);
        var rows = parseInt(val[1], 10);
        caState.life = new GameOfLife(rows, cols);
        caState.life.randomize();
        caState.generation = 0;
        caState.cellSize = Math.floor(Math.min(CANVAS_W / cols, CANVAS_H / rows));
        caDraw();
    });
    var mouseDown = false;
    caCanvas.addEventListener("mousedown", function (e) {
        if (caState.mode !== "life")
            return;
        mouseDown = true;
        var cell = caCanvasToCell(e.offsetX, e.offsetY);
        if (!cell)
            return;
        caState.drawValue = caState.life.get(cell[0], cell[1]) ? 0 : 1;
        caState.life.set(cell[0], cell[1], caState.drawValue);
        caDraw();
    });
    caCanvas.addEventListener("mousemove", function (e) {
        if (!mouseDown || caState.mode !== "life")
            return;
        var cell = caCanvasToCell(e.offsetX, e.offsetY);
        if (!cell)
            return;
        caState.life.set(cell[0], cell[1], caState.drawValue);
        caDraw();
    });
    window.addEventListener("mouseup", function () { mouseDown = false; });
    var ruleInput = document.getElementById("eca-rule");
    ruleInput.addEventListener("input", function () {
        var v = parseInt(ruleInput.value, 10);
        if (isNaN(v))
            v = 0;
        v = clamp(v, 0, 255);
        caState.eca.rule = v;
        caState.ecaRule = v;
        var label = document.getElementById("eca-rule-label");
        if (label)
            label.textContent = String(v);
        caState.eca.reset(caState.ecaSingleCell);
        caState.eca.fillAll();
        caDraw();
    });
    var famousRules = [30, 90, 110, 184];
    famousRules.forEach(function (r) {
        var btn = document.getElementById("eca-rule-" + r);
        if (!btn)
            return;
        btn.addEventListener("click", function () {
            ruleInput.value = String(r);
            var label = document.getElementById("eca-rule-label");
            if (label)
                label.textContent = String(r);
            caState.eca.rule = r;
            caState.ecaRule = r;
            caState.eca.reset(caState.ecaSingleCell);
            caState.eca.fillAll();
            caDraw();
        });
    });
    var ecaInitSel = document.getElementById("eca-init");
    ecaInitSel.addEventListener("change", function () {
        caState.ecaSingleCell = ecaInitSel.value === "single";
        caState.eca.reset(caState.ecaSingleCell);
        caState.eca.fillAll();
        caDraw();
    });
    var btnEcaReset = document.getElementById("btn-eca-reset");
    btnEcaReset.addEventListener("click", function () {
        caState.eca.reset(caState.ecaSingleCell);
        if (!caState.running) {
            caState.eca.fillAll();
        }
        caDraw();
    });
}
window.addEventListener("DOMContentLoaded", function () {
    caCanvas = document.getElementById("ca-caCanvas");
    caCtx = caCanvas.getContext("2d");
    caCanvas.width = CANVAS_W;
    caCanvas.height = CANVAS_H;
    initCAState();
    caBindUI();
    caUpdateModeUI();
    caUpdatePlayPauseButton();
    caStartLoop();
});

},{}]},{},[1]);
