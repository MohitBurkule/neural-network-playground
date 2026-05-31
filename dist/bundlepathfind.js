(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var ROWS = 25;
var COLS = 40;
var CELL_SIZE = 0;
var grid = [];
var startRow = 2, startCol = 2;
var goalRow = ROWS - 3, goalCol = COLS - 3;
var algorithm = "astar";
var heuristic = "manhattan";
var diagonals = false;
var speedMs = 20;
var visitedSet = new Set();
var frontierSet = new Set();
var pathCells = [];
var nodesExpanded = 0;
var pathLength = 0;
var elapsedMs = 0;
var playState = "idle";
var stepGen = null;
var animFrame = 0;
var lastStepTime = 0;
var mouseDown = false;
var drawMode = "wall";
var canvas = document.getElementById("pathfind-canvas");
var ctx = canvas.getContext("2d");
function resizeCanvas() {
    var wrap = document.getElementById("canvas-wrap");
    var W = wrap.clientWidth;
    var H = wrap.clientHeight;
    var cs = Math.floor(Math.min(W / COLS, H / ROWS));
    CELL_SIZE = Math.max(cs, 4);
    canvas.width = COLS * CELL_SIZE;
    canvas.height = ROWS * CELL_SIZE;
    draw();
}
function initGrid() {
    grid = [];
    for (var r = 0; r < ROWS; r++) {
        grid[r] = [];
        for (var c = 0; c < COLS; c++) {
            grid[r][c] = { row: r, col: c, type: "empty", weight: 1 };
        }
    }
}
function resetSearch() {
    visitedSet = new Set();
    frontierSet = new Set();
    pathCells = [];
    nodesExpanded = 0;
    pathLength = 0;
    elapsedMs = 0;
    playState = "idle";
    stepGen = null;
    cancelAnimationFrame(animFrame);
    updateStats();
    draw();
}
var COLOR = {
    bg: "#0d1117",
    empty: "#161b22",
    wall: "#58a6ff",
    weight: "#e3b341",
    start: "#3fb950",
    goal: "#f85149",
    visited: "#1c3a5c",
    frontier: "#388bfd",
    path: "#ffa657",
    grid: "#21262d"
};
function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var cs = CELL_SIZE;
    for (var r = 0; r < ROWS; r++) {
        for (var c = 0; c < COLS; c++) {
            var cell = grid[r][c];
            var key = "".concat(r, ",").concat(c);
            var fill = COLOR.empty;
            if (cell.type === "wall")
                fill = COLOR.wall;
            else if (cell.type === "weight")
                fill = COLOR.weight;
            else if (visitedSet.has(key))
                fill = COLOR.visited;
            else if (frontierSet.has(key))
                fill = COLOR.frontier;
            ctx.fillStyle = fill;
            ctx.fillRect(c * cs, r * cs, cs, cs);
            ctx.strokeStyle = COLOR.grid;
            ctx.lineWidth = 0.5;
            ctx.strokeRect(c * cs, r * cs, cs, cs);
        }
    }
    if (pathCells.length > 0) {
        ctx.fillStyle = COLOR.path;
        pathCells.forEach(function (_a) {
            var r = _a[0], c = _a[1];
            ctx.fillRect(c * cs + 1, r * cs + 1, cs - 2, cs - 2);
        });
    }
    drawSpecial(startRow, startCol, COLOR.start, "S");
    drawSpecial(goalRow, goalCol, COLOR.goal, "G");
}
function drawSpecial(r, c, color, label) {
    var cs = CELL_SIZE;
    ctx.fillStyle = color;
    ctx.fillRect(c * cs + 1, r * cs + 1, cs - 2, cs - 2);
    if (cs >= 12) {
        ctx.fillStyle = "#0d1117";
        ctx.font = "bold ".concat(Math.floor(cs * 0.55), "px monospace");
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, c * cs + cs / 2, r * cs + cs / 2);
    }
}
function heuristicFn(r1, c1, r2, c2) {
    var dr = Math.abs(r1 - r2);
    var dc = Math.abs(c1 - c2);
    if (heuristic === "manhattan")
        return dr + dc;
    if (heuristic === "euclidean")
        return Math.sqrt(dr * dr + dc * dc);
    return Math.max(dr, dc);
}
function neighbors(r, c) {
    var dirs4 = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    var dirs8 = [
        [-1, 0], [1, 0], [0, -1], [0, 1],
        [-1, -1], [-1, 1], [1, -1], [1, 1],
    ];
    var dirs = diagonals ? dirs8 : dirs4;
    var result = [];
    dirs.forEach(function (_a) {
        var dr = _a[0], dc = _a[1];
        var nr = r + dr;
        var nc = c + dc;
        if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
            var cell = grid[nr][nc];
            if (cell.type !== "wall") {
                var isDiag = dr !== 0 && dc !== 0;
                var moveCost = (isDiag ? 1.414 : 1) * cell.weight;
                result.push([nr, nc, moveCost]);
            }
        }
    });
    return result;
}
var MinHeap = (function () {
    function MinHeap() {
        this.data = [];
    }
    MinHeap.prototype.push = function (priority, value) {
        this.data.push([priority, value]);
        this._bubbleUp(this.data.length - 1);
    };
    MinHeap.prototype.pop = function () {
        if (this.data.length === 0)
            return undefined;
        var top = this.data[0][1];
        var last = this.data.pop();
        if (this.data.length > 0) {
            this.data[0] = last;
            this._sinkDown(0);
        }
        return top;
    };
    Object.defineProperty(MinHeap.prototype, "size", {
        get: function () { return this.data.length; },
        enumerable: false,
        configurable: true
    });
    MinHeap.prototype._bubbleUp = function (i) {
        var _a;
        while (i > 0) {
            var parent_1 = Math.floor((i - 1) / 2);
            if (this.data[parent_1][0] <= this.data[i][0])
                break;
            _a = [this.data[i], this.data[parent_1]], this.data[parent_1] = _a[0], this.data[i] = _a[1];
            i = parent_1;
        }
    };
    MinHeap.prototype._sinkDown = function (i) {
        var _a;
        var n = this.data.length;
        while (true) {
            var smallest = i;
            var l = 2 * i + 1, r = 2 * i + 2;
            if (l < n && this.data[l][0] < this.data[smallest][0])
                smallest = l;
            if (r < n && this.data[r][0] < this.data[smallest][0])
                smallest = r;
            if (smallest === i)
                break;
            _a = [this.data[i], this.data[smallest]], this.data[smallest] = _a[0], this.data[i] = _a[1];
            i = smallest;
        }
    };
    return MinHeap;
}());
function astarGen(isDijkstra, isGreedy) {
    var visited, frontier, open, gScore, closed, frontierKeys, startNode, expanded, current, key, path, node, frontierSnap, nbrs, i, _a, nr, nc, cost, nkey, tentativeG, prevG, h, f, node;
    var _b;
    return __generator(this, function (_c) {
        switch (_c.label) {
            case 0:
                visited = [];
                frontier = [];
                open = new MinHeap();
                gScore = new Map();
                closed = new Set();
                frontierKeys = new Set();
                startNode = {
                    row: startRow, col: startCol,
                    g: 0, h: heuristicFn(startRow, startCol, goalRow, goalCol),
                    f: 0, parent: null
                };
                startNode.f = isGreedy ? startNode.h : (isDijkstra ? startNode.g : startNode.g + startNode.h);
                open.push(startNode.f, startNode);
                gScore.set("".concat(startRow, ",").concat(startCol), 0);
                frontierKeys.add("".concat(startRow, ",").concat(startCol));
                expanded = 0;
                _c.label = 1;
            case 1:
                if (!(open.size > 0)) return [3, 3];
                current = open.pop();
                key = "".concat(current.row, ",").concat(current.col);
                if (closed.has(key))
                    return [3, 1];
                closed.add(key);
                frontierKeys.delete(key);
                expanded++;
                visited.push([current.row, current.col]);
                if (current.row === goalRow && current.col === goalCol) {
                    path = [];
                    node = current;
                    while (node !== null) {
                        path.push([node.row, node.col]);
                        node = node.parent;
                    }
                    path.reverse();
                    return [2, {
                            done: true,
                            visited: Array.from(visited),
                            frontier: Array.from(frontierKeys).map(function (k) {
                                var _a = k.split(",").map(Number), r = _a[0], c = _a[1];
                                return [r, c];
                            }),
                            path: path,
                            nodesExpanded: expanded
                        }];
                }
                frontierSnap = Array.from(frontierKeys).map(function (k) {
                    var _a = k.split(",").map(Number), r = _a[0], c = _a[1];
                    return [r, c];
                });
                return [4, {
                        done: false,
                        visited: Array.from(visited),
                        frontier: frontierSnap,
                        path: [],
                        nodesExpanded: expanded
                    }];
            case 2:
                _c.sent();
                nbrs = neighbors(current.row, current.col);
                for (i = 0; i < nbrs.length; i++) {
                    _a = nbrs[i], nr = _a[0], nc = _a[1], cost = _a[2];
                    nkey = "".concat(nr, ",").concat(nc);
                    if (closed.has(nkey))
                        continue;
                    tentativeG = isDijkstra || !isGreedy ? current.g + cost : 0;
                    prevG = (_b = gScore.get(nkey)) !== null && _b !== void 0 ? _b : Infinity;
                    if (tentativeG < prevG) {
                        gScore.set(nkey, tentativeG);
                        h = heuristicFn(nr, nc, goalRow, goalCol);
                        f = isGreedy ? h : (isDijkstra ? tentativeG : tentativeG + h);
                        node = { row: nr, col: nc, g: tentativeG, h: h, f: f, parent: current };
                        open.push(f, node);
                        frontierKeys.add(nkey);
                    }
                }
                return [3, 1];
            case 3: return [2, { done: true, visited: Array.from(visited), frontier: [], path: [], nodesExpanded: expanded }];
        }
    });
}
function bfsGen() {
    var visitedArr, queue, visited, startNode, expanded, current, path, node, frontierSnap, nbrs, i, _a, nr, nc, nkey;
    return __generator(this, function (_b) {
        switch (_b.label) {
            case 0:
                visitedArr = [];
                queue = [];
                visited = new Set();
                startNode = { row: startRow, col: startCol, g: 0, h: 0, f: 0, parent: null };
                queue.push(startNode);
                visited.add("".concat(startRow, ",").concat(startCol));
                expanded = 0;
                _b.label = 1;
            case 1:
                if (!(queue.length > 0)) return [3, 3];
                current = queue.shift();
                expanded++;
                visitedArr.push([current.row, current.col]);
                if (current.row === goalRow && current.col === goalCol) {
                    path = [];
                    node = current;
                    while (node !== null) {
                        path.push([node.row, node.col]);
                        node = node.parent;
                    }
                    path.reverse();
                    return [2, { done: true, visited: Array.from(visitedArr), frontier: [], path: path, nodesExpanded: expanded }];
                }
                frontierSnap = queue.map(function (n) { return [n.row, n.col]; });
                return [4, {
                        done: false,
                        visited: Array.from(visitedArr),
                        frontier: frontierSnap,
                        path: [],
                        nodesExpanded: expanded
                    }];
            case 2:
                _b.sent();
                nbrs = neighbors(current.row, current.col);
                for (i = 0; i < nbrs.length; i++) {
                    _a = nbrs[i], nr = _a[0], nc = _a[1];
                    nkey = "".concat(nr, ",").concat(nc);
                    if (!visited.has(nkey)) {
                        visited.add(nkey);
                        queue.push({ row: nr, col: nc, g: current.g + 1, h: 0, f: 0, parent: current });
                    }
                }
                return [3, 1];
            case 3: return [2, { done: true, visited: Array.from(visitedArr), frontier: [], path: [], nodesExpanded: expanded }];
        }
    });
}
function makeGenerator() {
    if (algorithm === "bfs")
        return bfsGen();
    if (algorithm === "dijkstra")
        return astarGen(true, false);
    if (algorithm === "greedy")
        return astarGen(false, true);
    return astarGen(false, false);
}
function applyStepResult(result) {
    visitedSet = new Set(result.visited.map(function (_a) {
        var r = _a[0], c = _a[1];
        return "".concat(r, ",").concat(c);
    }));
    frontierSet = new Set(result.frontier.map(function (_a) {
        var r = _a[0], c = _a[1];
        return "".concat(r, ",").concat(c);
    }));
    pathCells = result.path;
    nodesExpanded = result.nodesExpanded;
    if (pathCells.length > 1) {
        var len = 0;
        for (var i = 1; i < pathCells.length; i++) {
            var _a = pathCells[i], r = _a[0], c = _a[1];
            var _b = pathCells[i - 1], pr = _b[0], pc = _b[1];
            var dr = Math.abs(r - pr), dc = Math.abs(c - pc);
            var isDiag = dr + dc > 1;
            len += (isDiag ? 1.414 : 1) * grid[r][c].weight;
        }
        pathLength = Math.round(len * 10) / 10;
    }
    else {
        pathLength = 0;
    }
    updateStats();
    draw();
}
function animLoop(ts) {
    if (playState !== "running")
        return;
    if (!stepGen)
        return;
    if (ts - lastStepTime < speedMs) {
        animFrame = requestAnimationFrame(animLoop);
        return;
    }
    lastStepTime = ts;
    elapsedMs += speedMs;
    var res = stepGen.next();
    var value = res.value;
    applyStepResult(value);
    if (value.done || res.done) {
        playState = "done";
        updateButtons();
        return;
    }
    animFrame = requestAnimationFrame(animLoop);
}
function updateStats() {
    var el = document.getElementById("stats");
    el.innerHTML = "\n    <div class=\"stat\"><span>Nodes expanded</span><strong>".concat(nodesExpanded, "</strong></div>\n    <div class=\"stat\"><span>Path length</span><strong>").concat(pathLength || "—", "</strong></div>\n    <div class=\"stat\"><span>Status</span><strong>").concat(playState, "</strong></div>\n  ");
}
function updateButtons() {
    var play = document.getElementById("btn-play");
    var pause = document.getElementById("btn-pause");
    var step = document.getElementById("btn-step");
    play.disabled = playState === "running" || playState === "done";
    pause.disabled = playState !== "running";
    step.disabled = playState === "running" || playState === "done";
}
function onPlay() {
    if (playState === "idle") {
        stepGen = makeGenerator();
        playState = "running";
        lastStepTime = 0;
    }
    else if (playState === "paused") {
        playState = "running";
    }
    updateButtons();
    animFrame = requestAnimationFrame(animLoop);
}
function onPause() {
    playState = "paused";
    cancelAnimationFrame(animFrame);
    updateButtons();
}
function onStep() {
    if (playState === "idle") {
        stepGen = makeGenerator();
        playState = "paused";
    }
    if (!stepGen || playState === "done")
        return;
    var res = stepGen.next();
    var value = res.value;
    applyStepResult(value);
    if (value.done || res.done) {
        playState = "done";
    }
    updateButtons();
}
function onReset() {
    resetSearch();
    updateButtons();
}
function generateMaze() {
    resetSearch();
    for (var r = 0; r < ROWS; r++) {
        for (var c = 0; c < COLS; c++) {
            grid[r][c].type = "wall";
            grid[r][c].weight = 1;
        }
    }
    var visited = new Set();
    function carve(r, c) {
        var _a;
        visited.add("".concat(r, ",").concat(c));
        grid[r][c].type = "empty";
        var dirs = [[-2, 0], [2, 0], [0, -2], [0, 2]];
        for (var i = dirs.length - 1; i > 0; i--) {
            var j = Math.floor(Math.random() * (i + 1));
            _a = [dirs[j], dirs[i]], dirs[i] = _a[0], dirs[j] = _a[1];
        }
        dirs.forEach(function (_a) {
            var dr = _a[0], dc = _a[1];
            var nr = r + dr, nc = c + dc;
            if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && !visited.has("".concat(nr, ",").concat(nc))) {
                grid[r + dr / 2][c + dc / 2].type = "empty";
                carve(nr, nc);
            }
        });
    }
    var sr = 1, sc = 1;
    carve(sr, sc);
    grid[startRow][startCol].type = "empty";
    grid[goalRow][goalCol].type = "empty";
    draw();
}
function cellFromEvent(e) {
    var rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    var c = Math.floor(x / CELL_SIZE);
    var r = Math.floor(y / CELL_SIZE);
    if (r >= 0 && r < ROWS && c >= 0 && c < COLS)
        return [r, c];
    return null;
}
function paintCell(r, c) {
    if (r === startRow && c === startCol)
        return;
    if (r === goalRow && c === goalCol)
        return;
    if (drawMode === "wall") {
        grid[r][c].type = "wall";
        grid[r][c].weight = 1;
    }
    else if (drawMode === "weight") {
        grid[r][c].type = "weight";
        grid[r][c].weight = 5;
    }
    else if (drawMode === "erase") {
        grid[r][c].type = "empty";
        grid[r][c].weight = 1;
    }
    draw();
}
canvas.addEventListener("mousedown", function (e) {
    var pos = cellFromEvent(e);
    if (!pos)
        return;
    var r = pos[0], c = pos[1];
    if (drawMode === "start") {
        startRow = r;
        startCol = c;
        resetSearch();
        draw();
        return;
    }
    if (drawMode === "goal") {
        goalRow = r;
        goalCol = c;
        resetSearch();
        draw();
        return;
    }
    mouseDown = true;
    paintCell(r, c);
});
canvas.addEventListener("mousemove", function (e) {
    if (!mouseDown)
        return;
    var pos = cellFromEvent(e);
    if (!pos)
        return;
    paintCell(pos[0], pos[1]);
});
canvas.addEventListener("mouseup", function () { mouseDown = false; });
canvas.addEventListener("mouseleave", function () { mouseDown = false; });
canvas.addEventListener("contextmenu", function (e) { return e.preventDefault(); });
function wireControls() {
    var algoSel = document.getElementById("algo-select");
    algoSel.addEventListener("change", function () {
        algorithm = algoSel.value;
        resetSearch();
    });
    var heuristicSel = document.getElementById("heuristic-select");
    heuristicSel.addEventListener("change", function () {
        heuristic = heuristicSel.value;
        resetSearch();
    });
    var diagCheck = document.getElementById("diag-check");
    diagCheck.addEventListener("change", function () {
        diagonals = diagCheck.checked;
        resetSearch();
    });
    var speedSlider = document.getElementById("speed-slider");
    var speedLabel = document.getElementById("speed-label");
    speedSlider.addEventListener("input", function () {
        var v = parseInt(speedSlider.value);
        speedMs = v;
        speedLabel.textContent = v + "ms";
    });
    var sizeSlider = document.getElementById("size-slider");
    var sizeLabel = document.getElementById("size-label");
    sizeSlider.addEventListener("input", function () {
        var s = parseInt(sizeSlider.value);
        ROWS = s;
        COLS = Math.round(s * 1.6);
        sizeLabel.textContent = "".concat(ROWS, "\u00D7").concat(COLS);
        startRow = 2;
        startCol = 2;
        goalRow = ROWS - 3;
        goalCol = COLS - 3;
        initGrid();
        resetSearch();
        resizeCanvas();
    });
    document.getElementById("btn-play").addEventListener("click", onPlay);
    document.getElementById("btn-pause").addEventListener("click", onPause);
    document.getElementById("btn-step").addEventListener("click", onStep);
    document.getElementById("btn-reset").addEventListener("click", onReset);
    document.getElementById("btn-maze").addEventListener("click", generateMaze);
    document.getElementById("btn-clear").addEventListener("click", function () {
        initGrid();
        resetSearch();
        resizeCanvas();
    });
    var drawBtns = document.querySelectorAll(".draw-btn");
    drawBtns.forEach(function (btn) {
        btn.addEventListener("click", function () {
            drawMode = btn.dataset["mode"];
            drawBtns.forEach(function (b) { return b.classList.remove("active"); });
            btn.classList.add("active");
        });
    });
}
function init() {
    initGrid();
    wireControls();
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
    updateStats();
    updateButtons();
}
window.addEventListener("DOMContentLoaded", init);

},{}]},{},[1]);
