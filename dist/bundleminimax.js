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
var gameMode = "ttt";
var humanPlayer = "X";
var aiPlayer = "O";
var board = [];
var history = [];
var aiDepth = 5;
var gameOver = false;
var winner = null;
var isDraw = false;
var showTree = true;
var currentTree = null;
var nodeIdCounter = 0;
var nodesWithPruning = 0;
var nodesWithout = 0;
var c4Scores = [];
var aiStrength = "hard";
var TTT_SIZE = 3;
var C4_ROWS = 6;
var C4_COLS = 7;
function boardSize() {
    return gameMode === "ttt" ? TTT_SIZE * TTT_SIZE : C4_ROWS * C4_COLS;
}
function freshBoard() {
    return Array(boardSize()).fill(null);
}
var TTT_WINS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
];
function tttWinner(b) {
    for (var _i = 0, TTT_WINS_1 = TTT_WINS; _i < TTT_WINS_1.length; _i++) {
        var _a = TTT_WINS_1[_i], a = _a[0], c = _a[1], d = _a[2];
        if (b[a] && b[a] === b[c] && b[a] === b[d])
            return b[a];
    }
    return null;
}
function tttMoves(b) {
    return b.map(function (c, i) { return c === null ? i : -1; }).filter(function (i) { return i >= 0; });
}
function tttApply(b, move, p) {
    var nb = __spreadArray([], b, true);
    nb[move] = p;
    return nb;
}
function c4Col(idx) { return idx % C4_COLS; }
function c4Row(idx) { return Math.floor(idx / C4_COLS); }
function c4Idx(row, col) { return row * C4_COLS + col; }
function c4DropRow(b, col) {
    for (var r = C4_ROWS - 1; r >= 0; r--) {
        if (!b[c4Idx(r, col)])
            return r;
    }
    return -1;
}
function c4Moves(b) {
    var cols = [];
    for (var c = 0; c < C4_COLS; c++) {
        if (c4DropRow(b, c) >= 0)
            cols.push(c);
    }
    return cols;
}
function c4Apply(b, col, p) {
    var nb = __spreadArray([], b, true);
    var r = c4DropRow(nb, col);
    if (r >= 0)
        nb[c4Idx(r, col)] = p;
    return nb;
}
function c4Check4(b, p) {
    var _loop_1 = function (r) {
        var _loop_5 = function (c) {
            if ([0, 1, 2, 3].every(function (d) { return b[c4Idx(r, c + d)] === p; }))
                return { value: true };
        };
        for (var c = 0; c <= C4_COLS - 4; c++) {
            var state_5 = _loop_5(c);
            if (typeof state_5 === "object")
                return state_5;
        }
    };
    for (var r = 0; r < C4_ROWS; r++) {
        var state_1 = _loop_1(r);
        if (typeof state_1 === "object")
            return state_1.value;
    }
    var _loop_2 = function (r) {
        var _loop_6 = function (c) {
            if ([0, 1, 2, 3].every(function (d) { return b[c4Idx(r + d, c)] === p; }))
                return { value: true };
        };
        for (var c = 0; c < C4_COLS; c++) {
            var state_6 = _loop_6(c);
            if (typeof state_6 === "object")
                return state_6;
        }
    };
    for (var r = 0; r <= C4_ROWS - 4; r++) {
        var state_2 = _loop_2(r);
        if (typeof state_2 === "object")
            return state_2.value;
    }
    var _loop_3 = function (r) {
        var _loop_7 = function (c) {
            if ([0, 1, 2, 3].every(function (d) { return b[c4Idx(r + d, c + d)] === p; }))
                return { value: true };
        };
        for (var c = 0; c <= C4_COLS - 4; c++) {
            var state_7 = _loop_7(c);
            if (typeof state_7 === "object")
                return state_7;
        }
    };
    for (var r = 0; r <= C4_ROWS - 4; r++) {
        var state_3 = _loop_3(r);
        if (typeof state_3 === "object")
            return state_3.value;
    }
    var _loop_4 = function (r) {
        var _loop_8 = function (c) {
            if ([0, 1, 2, 3].every(function (d) { return b[c4Idx(r - d, c + d)] === p; }))
                return { value: true };
        };
        for (var c = 0; c <= C4_COLS - 4; c++) {
            var state_8 = _loop_8(c);
            if (typeof state_8 === "object")
                return state_8;
        }
    };
    for (var r = 3; r < C4_ROWS; r++) {
        var state_4 = _loop_4(r);
        if (typeof state_4 === "object")
            return state_4.value;
    }
    return false;
}
function c4Score(b, p) {
    var opp = p === "X" ? "O" : "X";
    var score = 0;
    function scoreWindow(window) {
        var mine = window.filter(function (c) { return c === p; }).length;
        var empty = window.filter(function (c) { return c === null; }).length;
        var theirs = window.filter(function (c) { return c === opp; }).length;
        if (mine === 4)
            return 100;
        if (mine === 3 && empty === 1)
            return 5;
        if (mine === 2 && empty === 2)
            return 2;
        if (theirs === 3 && empty === 1)
            return -4;
        return 0;
    }
    var center = Math.floor(C4_COLS / 2);
    for (var r = 0; r < C4_ROWS; r++) {
        if (b[c4Idx(r, center)] === p)
            score += 3;
    }
    var _loop_9 = function (r) {
        var _loop_13 = function (c) {
            score += scoreWindow([0, 1, 2, 3].map(function (d) { return b[c4Idx(r, c + d)]; }));
        };
        for (var c = 0; c <= C4_COLS - 4; c++) {
            _loop_13(c);
        }
    };
    for (var r = 0; r < C4_ROWS; r++) {
        _loop_9(r);
    }
    var _loop_10 = function (r) {
        var _loop_14 = function (c) {
            score += scoreWindow([0, 1, 2, 3].map(function (d) { return b[c4Idx(r + d, c)]; }));
        };
        for (var c = 0; c < C4_COLS; c++) {
            _loop_14(c);
        }
    };
    for (var r = 0; r <= C4_ROWS - 4; r++) {
        _loop_10(r);
    }
    var _loop_11 = function (r) {
        var _loop_15 = function (c) {
            score += scoreWindow([0, 1, 2, 3].map(function (d) { return b[c4Idx(r + d, c + d)]; }));
        };
        for (var c = 0; c <= C4_COLS - 4; c++) {
            _loop_15(c);
        }
    };
    for (var r = 0; r <= C4_ROWS - 4; r++) {
        _loop_11(r);
    }
    var _loop_12 = function (r) {
        var _loop_16 = function (c) {
            score += scoreWindow([0, 1, 2, 3].map(function (d) { return b[c4Idx(r - d, c + d)]; }));
        };
        for (var c = 0; c <= C4_COLS - 4; c++) {
            _loop_16(c);
        }
    };
    for (var r = 3; r < C4_ROWS; r++) {
        _loop_12(r);
    }
    return score;
}
function currentWinner(b) {
    if (gameMode === "ttt")
        return tttWinner(b);
    if (c4Check4(b, "X"))
        return "X";
    if (c4Check4(b, "O"))
        return "O";
    return null;
}
function getMoves(b) {
    return gameMode === "ttt" ? tttMoves(b) : c4Moves(b);
}
function applyMove(b, move, p) {
    return gameMode === "ttt" ? tttApply(b, move, p) : c4Apply(b, move, p);
}
function opponent(p) { return p === "X" ? "O" : "X"; }
function minimax(b, depth, isMax, alpha, beta, player, buildTree, parentNode, move) {
    nodesWithout++;
    var nodeId = nodeIdCounter++;
    var node = null;
    var w = currentWinner(b);
    var moves = getMoves(b);
    var terminal = w !== null || moves.length === 0;
    var value;
    if (w !== null) {
        value = w === aiPlayer ? 1000 : -1000;
        if (gameMode === "c4")
            value = w === aiPlayer ? 10000 : -10000;
    }
    else if (moves.length === 0) {
        value = 0;
    }
    else if (depth === 0) {
        value = gameMode === "c4" ? c4Score(b, aiPlayer) - c4Score(b, humanPlayer) : 0;
    }
    else {
        value = isMax ? -Infinity : Infinity;
        nodesWithPruning++;
        if (buildTree) {
            node = { id: nodeId, move: move, value: 0, alpha: alpha, beta: beta, depth: aiDepth - depth, player: player, children: [], pruned: false, terminal: false };
            if (parentNode)
                parentNode.children.push(node);
            if (!currentTree)
                currentTree = node;
        }
        for (var _i = 0, moves_1 = moves; _i < moves_1.length; _i++) {
            var m = moves_1[_i];
            var nb = applyMove(b, m, player);
            var childVal = minimax(nb, depth - 1, !isMax, alpha, beta, opponent(player), buildTree, node, m);
            if (isMax) {
                value = Math.max(value, childVal);
                alpha = Math.max(alpha, value);
            }
            else {
                value = Math.min(value, childVal);
                beta = Math.min(beta, value);
            }
            if (beta <= alpha) {
                if (buildTree && node) {
                    var pruneNode = {
                        id: nodeIdCounter++, move: -1, value: 0,
                        alpha: alpha,
                        beta: beta,
                        depth: aiDepth - depth + 1, player: opponent(player),
                        children: [], pruned: true, terminal: false
                    };
                    node.children.push(pruneNode);
                }
                break;
            }
        }
        if (node)
            node.value = value;
        return value;
    }
    nodesWithPruning++;
    if (buildTree && terminal) {
        node = { id: nodeId, move: move, value: value, alpha: alpha, beta: beta, depth: aiDepth - depth, player: player, children: [], pruned: false, terminal: true };
        if (parentNode)
            parentNode.children.push(node);
        if (!currentTree)
            currentTree = node;
    }
    return value;
}
function getBestMove(b) {
    nodeIdCounter = 0;
    nodesWithPruning = 0;
    nodesWithout = 0;
    currentTree = null;
    var moves = getMoves(b);
    var bestMove = moves[0];
    var bestVal = -Infinity;
    var colScores = Array(gameMode === "c4" ? C4_COLS : 9).fill(-Infinity);
    var savedNodesW = nodesWithPruning;
    nodesWithout = 0;
    var depth = aiStrength === "easy" ? 1 : aiStrength === "medium" ? 3 : aiDepth;
    var doBuildTree = gameMode === "ttt" && showTree;
    var firstMove = true;
    for (var _i = 0, moves_2 = moves; _i < moves_2.length; _i++) {
        var m = moves_2[_i];
        var nb = applyMove(b, m, aiPlayer);
        nodeIdCounter = 0;
        var doTree = doBuildTree && firstMove;
        if (doTree)
            currentTree = null;
        var val = minimax(nb, depth - 1, false, -Infinity, Infinity, humanPlayer, doTree, null, m);
        colScores[m] = val;
        if (val > bestVal) {
            bestVal = val;
            bestMove = m;
            if (doTree && currentTree) {
            }
        }
        firstMove = false;
    }
    if (gameMode === "ttt" && showTree) {
        nodeIdCounter = 0;
        nodesWithPruning = 0;
        currentTree = null;
        var rootNode = {
            id: nodeIdCounter++, move: null, value: 0, alpha: -Infinity, beta: Infinity,
            depth: 0, player: aiPlayer, children: [], pruned: false, terminal: false
        };
        currentTree = rootNode;
        var rv = -Infinity;
        var ra = -Infinity;
        for (var _a = 0, moves_3 = moves; _a < moves_3.length; _a++) {
            var m = moves_3[_a];
            var nb = applyMove(b, m, aiPlayer);
            var childNode = {
                id: nodeIdCounter++, move: m, value: 0, alpha: -Infinity, beta: Infinity,
                depth: 1, player: humanPlayer, children: [], pruned: false, terminal: false
            };
            rootNode.children.push(childNode);
            var val = minimax(nb, depth - 1, false, ra, Infinity, humanPlayer, true, childNode, m);
            childNode.value = val;
            rv = Math.max(rv, val);
            ra = Math.max(ra, rv);
        }
        rootNode.value = rv;
        nodesWithout = 0;
        for (var _b = 0, moves_4 = moves; _b < moves_4.length; _b++) {
            var m = moves_4[_b];
            var nb = applyMove(b, m, aiPlayer);
            countNodes(nb, depth - 1, false, humanPlayer);
        }
        nodesWithout += moves.length + 1;
    }
    if (gameMode === "c4") {
        c4Scores = Array(C4_COLS).fill(-Infinity);
        for (var _c = 0, moves_5 = moves; _c < moves_5.length; _c++) {
            var m = moves_5[_c];
            c4Scores[m] = colScores[m];
        }
    }
    return { move: bestMove, colScores: colScores };
}
function countNodes(b, depth, isMax, player) {
    nodesWithout++;
    var w = currentWinner(b);
    var moves = getMoves(b);
    if (w !== null || moves.length === 0 || depth === 0)
        return;
    for (var _i = 0, moves_6 = moves; _i < moves_6.length; _i++) {
        var m = moves_6[_i];
        var nb = applyMove(b, m, player);
        countNodes(nb, depth - 1, !isMax, opponent(player));
    }
}
function el(id) {
    return document.getElementById(id);
}
function svg(tag, attrs) {
    if (attrs === void 0) { attrs = {}; }
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var k in attrs)
        e.setAttribute(k, String(attrs[k]));
    return e;
}
function renderBoard() {
    var container = el("board-container");
    container.innerHTML = "";
    if (gameMode === "ttt")
        renderTTTBoard(container);
    else
        renderC4Board(container);
}
function renderTTTBoard(container) {
    var size = 300;
    var cell = size / 3;
    var s = svg("svg", { width: size, height: size });
    s.style.display = "block";
    var _loop_17 = function (i) {
        var r = Math.floor(i / 3);
        var c = i % 3;
        var rect = svg("rect", {
            x: c * cell, y: r * cell, width: cell - 4, height: cell - 4,
            rx: 8, fill: "#1e2040", stroke: "#3a3a6a", "stroke-width": 2,
            transform: "translate(2,2)"
        });
        if (!gameOver && board[i] === null) {
            rect.style.cursor = "pointer";
            rect.addEventListener("click", function () { return humanMove(i); });
            rect.addEventListener("mouseenter", function () { rect.setAttribute("fill", "#252850"); });
            rect.addEventListener("mouseleave", function () { rect.setAttribute("fill", "#1e2040"); });
        }
        s.appendChild(rect);
        if (board[i]) {
            var text = svg("text", {
                x: c * cell + cell / 2, y: r * cell + cell / 2 + 14,
                "text-anchor": "middle", "font-size": 52, "font-weight": "bold",
                fill: board[i] === "X" ? "#f87171" : "#60a5fa"
            });
            text.textContent = board[i];
            s.appendChild(text);
        }
    };
    for (var i = 0; i < 9; i++) {
        _loop_17(i);
    }
    for (var i = 1; i < 3; i++) {
        s.appendChild(svg("line", { x1: i * cell, y1: 5, x2: i * cell, y2: size - 5, stroke: "#4a4a8a", "stroke-width": 2 }));
        s.appendChild(svg("line", { x1: 5, y1: i * cell, x2: size - 5, y2: i * cell, stroke: "#4a4a8a", "stroke-width": 2 }));
    }
    if (winner) {
        for (var _i = 0, TTT_WINS_2 = TTT_WINS; _i < TTT_WINS_2.length; _i++) {
            var _a = TTT_WINS_2[_i], a = _a[0], b2 = _a[1], c = _a[2];
            if (board[a] === winner && board[b2] === winner && board[c] === winner) {
                var positions = [a, b2, c].map(function (idx) { return ({
                    x: (idx % 3) * cell + cell / 2, y: Math.floor(idx / 3) * cell + cell / 2
                }); });
                var line = svg("line", {
                    x1: positions[0].x, y1: positions[0].y,
                    x2: positions[2].x, y2: positions[2].y,
                    stroke: "#fbbf24", "stroke-width": 4, "stroke-linecap": "round", opacity: 0.85
                });
                s.appendChild(line);
                break;
            }
        }
    }
    container.appendChild(s);
}
function renderC4Board(container) {
    var cellW = 60, cellH = 60;
    var width = C4_COLS * cellW;
    var height = C4_ROWS * cellH + 50;
    var s = svg("svg", { width: width, height: height });
    if (c4Scores.length) {
        var maxAbs = Math.max.apply(Math, __spreadArray(__spreadArray([], c4Scores.filter(function (v) { return v !== -Infinity; }).map(Math.abs), false), [1], false));
        for (var c = 0; c < C4_COLS; c++) {
            var val = c4Scores[c];
            if (val === -Infinity)
                continue;
            var norm = Math.max(-1, Math.min(1, val / maxAbs));
            var barH = Math.abs(norm) * 20;
            var fill = norm > 0 ? "#60a5fa" : "#f87171";
            s.appendChild(svg("rect", {
                x: c * cellW + 8, y: norm > 0 ? 30 - barH : 30,
                width: cellW - 16, height: barH,
                fill: fill,
                rx: 3, opacity: 0.85
            }));
            var label = svg("text", {
                x: c * cellW + cellW / 2, y: 48, "text-anchor": "middle",
                "font-size": 9, fill: "#8888aa"
            });
            label.textContent = val.toFixed(0);
            s.appendChild(label);
        }
        s.appendChild(svg("line", { x1: 0, y1: 30, x2: width, y2: 30, stroke: "#3a3a6a", "stroke-width": 1 }));
    }
    var offsetY = 50;
    s.appendChild(svg("rect", { x: 0, y: offsetY, width: width, height: C4_ROWS * cellH, fill: "#1a2a5a", rx: 8 }));
    var _loop_18 = function (c) {
        var dropRow = c4DropRow(board, c);
        var colRect = svg("rect", {
            x: c * cellW, y: offsetY, width: cellW, height: C4_ROWS * cellH,
            fill: "transparent"
        });
        if (!gameOver && dropRow >= 0) {
            colRect.style.cursor = "pointer";
            colRect.addEventListener("click", function () { return humanMove(c); });
            colRect.addEventListener("mouseenter", function () {
                if (dropRow >= 0) {
                    var preview = document.getElementById("c4-preview-".concat(c));
                    if (preview)
                        preview.setAttribute("opacity", "0.35");
                }
            });
            colRect.addEventListener("mouseleave", function () {
                var preview = document.getElementById("c4-preview-".concat(c));
                if (preview)
                    preview.setAttribute("opacity", "0");
            });
        }
        s.appendChild(colRect);
        if (dropRow >= 0 && !gameOver) {
            var circ = svg("circle", {
                id: "c4-preview-".concat(c),
                cx: c * cellW + cellW / 2, cy: offsetY + dropRow * cellH + cellH / 2,
                r: cellW / 2 - 6, fill: humanPlayer === "X" ? "#f87171" : "#60a5fa", opacity: 0
            });
            s.appendChild(circ);
        }
    };
    for (var c = 0; c < C4_COLS; c++) {
        _loop_18(c);
    }
    for (var r = 0; r < C4_ROWS; r++) {
        for (var c = 0; c < C4_COLS; c++) {
            var cell = board[c4Idx(r, c)];
            var cx = c * cellW + cellW / 2;
            var cy = offsetY + r * cellH + cellH / 2;
            var fill = cell === "X" ? "#f87171" : cell === "O" ? "#60a5fa" : "#0f172a";
            s.appendChild(svg("circle", {
                cx: cx,
                cy: cy,
                r: cellW / 2 - 5,
                fill: fill,
                stroke: "#2a3a7a", "stroke-width": 2
            }));
        }
    }
    container.appendChild(s);
}
function renderTree() {
    var panel = el("tree-panel");
    panel.innerHTML = "";
    if (gameMode === "c4") {
        renderC4Info(panel);
        return;
    }
    if (!showTree || !currentTree) {
        panel.innerHTML = "<p style=\"color:#8888aa;font-size:0.85rem;margin-top:8px;\">\n      ".concat(showTree ? "Make a move to see the game tree." : "Toggle 'Show Tree' to visualize minimax.", "\n    </p>");
        return;
    }
    var stats = document.createElement("div");
    stats.className = "stats";
    stats.innerHTML = "\n    <span>Nodes with \u03B1-\u03B2 pruning: <b style=\"color:#60a5fa\">".concat(nodesWithPruning, "</b></span>\n    <span>Nodes without: <b style=\"color:#f87171\">").concat(nodesWithout, "</b></span>\n    <span>Saved: <b style=\"color:#4ade80\">").concat(Math.max(0, nodesWithout - nodesWithPruning), "</b></span>\n  ");
    panel.appendChild(stats);
    var maxDisplayDepth = 4;
    var nodeR = 16;
    var levelH = 70;
    var laid = [];
    var maxX = 0;
    function layout(n, depth, left, right, px, py) {
        if (depth > maxDisplayDepth)
            return;
        var x = (left + right) / 2;
        var y = depth * levelH + nodeR + 10;
        laid.push({ node: n, x: x, y: y, parentX: px, parentY: py });
        if (x > maxX)
            maxX = x;
        var count = n.children.length;
        if (count > 0 && depth < maxDisplayDepth) {
            var w_1 = (right - left) / count;
            n.children.forEach(function (child, i) {
                layout(child, depth + 1, left + i * w_1, left + (i + 1) * w_1, x, y);
            });
        }
    }
    var treeWidth = Math.min(800, Math.max(400, 80 * Math.pow(2, Math.min(maxDisplayDepth, 3))));
    layout(currentTree, 0, 0, treeWidth, undefined, undefined);
    var treeHeight = (Math.min(maxDisplayDepth, 4) + 1) * levelH + nodeR * 2 + 20;
    var treeSvg = svg("svg", { width: treeWidth, height: treeHeight });
    treeSvg.style.cssText = "display:block;overflow:visible;";
    for (var _i = 0, laid_1 = laid; _i < laid_1.length; _i++) {
        var ln = laid_1[_i];
        if (ln.parentX !== undefined && ln.parentY !== undefined) {
            var edge = svg("line", {
                x1: ln.parentX, y1: ln.parentY,
                x2: ln.x, y2: ln.y,
                stroke: ln.node.pruned ? "#3a3a5a" : "#4a4a7a",
                "stroke-width": ln.node.pruned ? 1 : 1.5,
                "stroke-dasharray": ln.node.pruned ? "4,3" : "none"
            });
            treeSvg.appendChild(edge);
        }
    }
    for (var _a = 0, laid_2 = laid; _a < laid_2.length; _a++) {
        var ln = laid_2[_a];
        var n = ln.node;
        var isAI = n.player !== aiPlayer;
        var pruned = n.pruned;
        var g = svg("g", { transform: "translate(".concat(ln.x, ",").concat(ln.y, ")") });
        var fill = pruned ? "#1a1a3a" :
            n.terminal ? (n.value > 0 ? "#1a3a2a" : n.value < 0 ? "#3a1a1a" : "#2a2a2a") :
                isAI ? "#1e2a4a" : "#2a1e3a";
        var stroke = pruned ? "#3a3a5a" :
            n.player === aiPlayer ? "#60a5fa" : "#f87171";
        g.appendChild(svg("circle", {
            r: nodeR,
            fill: fill,
            stroke: stroke,
            "stroke-width": pruned ? 1 : 2
        }));
        if (!pruned) {
            var valText = svg("text", {
                "text-anchor": "middle", dy: "0.35em",
                "font-size": n.value === Infinity || n.value === -Infinity ? 9 : 10,
                "font-weight": "bold",
                fill: n.value > 0 ? "#4ade80" : n.value < 0 ? "#f87171" : "#e0e0f0"
            });
            valText.textContent = n.value === Infinity ? "∞" : n.value === -Infinity ? "-∞" :
                Math.abs(n.value) >= 1000 ? (n.value > 0 ? "WIN" : "LOS") : String(n.value);
            g.appendChild(valText);
            if (n.move !== null) {
                var moveLabel = svg("text", {
                    "text-anchor": "middle", dy: "-1.6em", "font-size": 9, fill: "#8888aa"
                });
                moveLabel.textContent = gameMode === "ttt" ? "[".concat(n.move, "]") : "c".concat(n.move);
                g.appendChild(moveLabel);
            }
        }
        else {
            var x = svg("text", { "text-anchor": "middle", dy: "0.35em", "font-size": 11, fill: "#444466" });
            x.textContent = "✂";
            g.appendChild(x);
        }
        if (!pruned && !n.terminal && n.depth < maxDisplayDepth) {
            var abText = svg("text", {
                "text-anchor": "middle", dy: "".concat(nodeR + 12, "px"), "font-size": 8, fill: "#6666aa"
            });
            abText.textContent = "\u03B1".concat(n.alpha === -Infinity ? "-∞" : n.alpha, " \u03B2").concat(n.beta === Infinity ? "∞" : n.beta);
            g.appendChild(abText);
        }
        treeSvg.appendChild(g);
    }
    var legend = svg("g", { transform: "translate(4, ".concat(treeHeight - 18, ")") });
    var items = [
        ["#60a5fa", "●", "AI (max)"],
        ["#f87171", "●", "Human (min)"],
        ["#3a3a5a", "- -", "Pruned branch"]
    ];
    items.forEach(function (_a, i) {
        var color = _a[0], sym = _a[1], label = _a[2];
        var t = svg("text", { x: i * 130, "font-size": 9, fill: color });
        t.textContent = "".concat(sym, " ").concat(label);
        legend.appendChild(t);
    });
    treeSvg.appendChild(legend);
    var wrapper = document.createElement("div");
    wrapper.style.cssText = "overflow-x:auto;margin-top:8px;";
    wrapper.appendChild(treeSvg);
    panel.appendChild(wrapper);
}
function renderC4Info(panel) {
    var div = document.createElement("div");
    div.innerHTML = "\n    <div class=\"stats\">\n      <span>Nodes explored: <b style=\"color:#60a5fa\">".concat(nodesWithPruning, "</b></span>\n      <span>Search depth: <b style=\"color:#4ade80\">").concat(aiStrength === "easy" ? 1 : aiStrength === "medium" ? 3 : aiDepth, "</b></span>\n    </div>\n    <div style=\"margin-top:12px;font-size:0.82rem;color:#8888aa;\">\n      <p>Column scores shown above the board.</p>\n      <p style=\"margin-top:4px;\"><span style=\"color:#60a5fa\">Blue bars</span> = AI favors this column.</p>\n      <p style=\"margin-top:4px;\"><span style=\"color:#f87171\">Red bars</span> = AI disfavors this column.</p>\n    </div>\n  ");
    panel.appendChild(div);
}
function updateStatus() {
    var statusEl = el("status");
    if (winner) {
        statusEl.textContent = winner === humanPlayer ? "You win! 🎉" : "AI wins!";
        statusEl.style.color = winner === humanPlayer ? "#4ade80" : "#f87171";
    }
    else if (isDraw) {
        statusEl.textContent = "It's a draw!";
        statusEl.style.color = "#fbbf24";
    }
    else {
        var moves = getMoves(board);
        if (moves.length === 0) {
            statusEl.textContent = "Draw!";
            isDraw = true;
        }
        else {
            var whose = currentTurn() === humanPlayer ? "Your turn" : "AI thinking…";
            statusEl.textContent = whose;
            statusEl.style.color = currentTurn() === humanPlayer ? "#a78bfa" : "#8888aa";
        }
    }
}
function currentTurn() {
    var placed = board.filter(function (c) { return c !== null; }).length;
    var xFirst = "X";
    if (placed % 2 === 0)
        return xFirst;
    return opponent(xFirst);
}
function humanMove(move) {
    if (gameOver)
        return;
    if (currentTurn() !== humanPlayer)
        return;
    var newBoard = applyMove(board, move, humanPlayer);
    if (JSON.stringify(newBoard) === JSON.stringify(board))
        return;
    history.push(__spreadArray([], board, true));
    board = newBoard;
    checkEnd();
    renderBoard();
    updateStatus();
    if (!gameOver) {
        setTimeout(aiMove, 100);
    }
}
function aiMove() {
    if (gameOver || currentTurn() !== aiPlayer)
        return;
    var move = getBestMove(board).move;
    history.push(__spreadArray([], board, true));
    board = applyMove(board, move, aiPlayer);
    checkEnd();
    renderBoard();
    renderTree();
    updateStatus();
}
function checkEnd() {
    var w = currentWinner(board);
    if (w) {
        winner = w;
        gameOver = true;
    }
    else if (getMoves(board).length === 0) {
        isDraw = true;
        gameOver = true;
    }
}
function newGame() {
    board = freshBoard();
    history = [];
    gameOver = false;
    winner = null;
    isDraw = false;
    currentTree = null;
    c4Scores = [];
    nodesWithPruning = 0;
    nodesWithout = 0;
    renderBoard();
    renderTree();
    updateStatus();
    if (currentTurn() === aiPlayer) {
        setTimeout(aiMove, 200);
    }
}
function undoMove() {
    if (history.length < 1)
        return;
    board = history.pop();
    if (history.length >= 1 && currentTurn() === humanPlayer) {
    }
    gameOver = false;
    winner = null;
    isDraw = false;
    currentTree = null;
    c4Scores = [];
    renderBoard();
    renderTree();
    updateStatus();
}
function init() {
    var gameSelect = el("game-select");
    gameSelect.addEventListener("change", function () {
        gameMode = gameSelect.value;
        var treeToggle = el("tree-toggle-row");
        treeToggle.style.display = gameMode === "ttt" ? "flex" : "none";
        var depthRow = el("depth-row");
        depthRow.style.display = gameMode === "c4" ? "flex" : "none";
        newGame();
    });
    var whoFirst = el("who-first");
    whoFirst.addEventListener("change", function () {
        humanPlayer = whoFirst.value;
        aiPlayer = opponent(humanPlayer);
        newGame();
    });
    var showTreeChk = el("show-tree");
    showTreeChk.addEventListener("change", function () {
        showTree = showTreeChk.checked;
        renderTree();
    });
    var depthSlider = el("depth-slider");
    var depthLabel = el("depth-label");
    depthSlider.addEventListener("input", function () {
        aiDepth = parseInt(depthSlider.value);
        depthLabel.textContent = String(aiDepth);
    });
    var strengthSel = el("strength-select");
    strengthSel.addEventListener("change", function () {
        aiStrength = strengthSel.value;
    });
    el("new-game-btn").addEventListener("click", newGame);
    el("undo-btn").addEventListener("click", undoMove);
    newGame();
}
document.addEventListener("DOMContentLoaded", init);

},{}]},{},[1]);
