(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.classifyTwoGaussBlobs = classifyTwoGaussBlobs;
exports.classifyConcentricSpheres = classifyConcentricSpheres;
exports.classifyHelix = classifyHelix;
exports.classifySwissRoll = classifySwissRoll;
function randNormal(mean, stddev) {
    var u = 0, v = 0;
    while (u === 0)
        u = Math.random();
    while (v === 0)
        v = Math.random();
    return mean + stddev * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function randUniform(a, b) {
    return Math.random() * (b - a) + a;
}
function classifyTwoGaussBlobs(numSamples, noise) {
    var points = [];
    var variance = 0.5 + noise * 3;
    var half = Math.floor(numSamples / 2);
    for (var i = 0; i < half; i++) {
        points.push({
            x: randNormal(-2, variance),
            y: randNormal(-2, variance),
            z: randNormal(-2, variance),
            label: 1
        });
    }
    for (var i = half; i < numSamples; i++) {
        points.push({
            x: randNormal(2, variance),
            y: randNormal(2, variance),
            z: randNormal(2, variance),
            label: -1
        });
    }
    return points;
}
function classifyConcentricSpheres(numSamples, noise) {
    var points = [];
    for (var i = 0; i < numSamples; i++) {
        var theta = randUniform(0, 2 * Math.PI);
        var phi = Math.acos(randUniform(-1, 1));
        var isInner = i % 2 === 0;
        var r = isInner
            ? randUniform(0.3, 1.5) + randUniform(-1, 1) * noise
            : randUniform(2.5, 4.0) + randUniform(-1, 1) * noise;
        var x = r * Math.sin(phi) * Math.cos(theta);
        var y = r * Math.sin(phi) * Math.sin(theta);
        var z = r * Math.cos(phi);
        points.push({ x: x, y: y, z: z, label: isInner ? 1 : -1 });
    }
    return points;
}
function classifyHelix(numSamples, noise) {
    var points = [];
    var half = Math.floor(numSamples / 2);
    for (var i = 0; i < half; i++) {
        var t = (i / half) * 4 * Math.PI;
        var r = 2;
        points.push({
            x: r * Math.cos(t) + randNormal(0, noise),
            y: t * 0.5 - Math.PI * 2 + randNormal(0, noise),
            z: r * Math.sin(t) + randNormal(0, noise),
            label: 1
        });
    }
    for (var i = 0; i < numSamples - half; i++) {
        var t = (i / (numSamples - half)) * 4 * Math.PI;
        var r = 2;
        points.push({
            x: r * Math.cos(t + Math.PI) + randNormal(0, noise),
            y: t * 0.5 - Math.PI * 2 + randNormal(0, noise),
            z: r * Math.sin(t + Math.PI) + randNormal(0, noise),
            label: -1
        });
    }
    return points;
}
function classifySwissRoll(numSamples, noise) {
    var points = [];
    for (var i = 0; i < numSamples; i++) {
        var t = 1.5 * Math.PI * (1 + 2 * Math.random());
        var height = randUniform(-3, 3);
        var x = (t * Math.cos(t) / 5) + randNormal(0, noise);
        var y = height + randNormal(0, noise);
        var z = (t * Math.sin(t) / 5) + randNormal(0, noise);
        var label = Math.cos(t) > 0 ? 1 : -1;
        points.push({ x: x, y: y, z: z, label: label });
    }
    return points;
}

},{}],2:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.OPTIMIZER_EPSILON = exports.OPTIMIZER_BETA2 = exports.OPTIMIZER_BETA1 = exports.OptimizerType = exports.Link = exports.WeightQuantizationFunction = exports.RegularizationFunction = exports.Activations = exports.Errors = exports.Node = void 0;
exports.buildNetwork = buildNetwork;
exports.forwardProp = forwardProp;
exports.backProp = backProp;
exports.updateWeights = updateWeights;
exports.forEachNode = forEachNode;
exports.getOutputNode = getOutputNode;
exports.compileNetworkToJs = compileNetworkToJs;
var Node = (function () {
    function Node(id, activation, initZero) {
        this.inputLinks = [];
        this.bias = 0.1;
        this.outputs = [];
        this.outputDer = 0;
        this.inputDer = 0;
        this.accInputDer = 0;
        this.numAccumulatedDers = 0;
        this.biasOptimizerState = {};
        this.id = id;
        this.activation = activation;
        if (initZero) {
            this.bias = 0;
        }
    }
    Node.prototype.updateOutput = function (weightQuantizationFunction) {
        this.totalInput = this.bias;
        for (var j = 0; j < this.inputLinks.length; j++) {
            var link = this.inputLinks[j];
            var weight = weightQuantizationFunction ? weightQuantizationFunction.output(link.weight) : link.weight;
            this.totalInput += weight * link.source.output;
        }
        this.output = this.activation.output(this.totalInput);
        return this.output;
    };
    Node.prototype.compileToJs = function () {
        var js = this.bias.toPrecision(2) + "";
        for (var j = 0; j < this.inputLinks.length; j++) {
            var link = this.inputLinks[j];
            js += " + (".concat(link.weight.toPrecision(2), " * ").concat(link.source.compileToJsName(), ")");
        }
        return this.activation.compileToJs(js);
    };
    Node.prototype.compileToJsName = function () {
        return "v" + this.id;
    };
    return Node;
}());
exports.Node = Node;
var Errors = (function () {
    function Errors() {
    }
    Errors.SQUARE = {
        error: function (output, target) {
            return 0.5 * Math.pow(output - target, 2);
        },
        der: function (output, target) { return output - target; }
    };
    return Errors;
}());
exports.Errors = Errors;
Math.tanh = Math.tanh || function (x) {
    if (x === Infinity) {
        return 1;
    }
    else if (x === -Infinity) {
        return -1;
    }
    else {
        var e2x = Math.exp(2 * x);
        return (e2x - 1) / (e2x + 1);
    }
};
Math.softplus = Math.softplus || function (x) {
    var threshold = 20;
    var beta = 1;
    if (x * beta > threshold) {
        return x;
    }
    else if (x === -Infinity) {
        return 0;
    }
    else {
        return 1 / beta * Math.log(1 + Math.exp(beta * x));
    }
};
var Activations = (function () {
    function Activations() {
    }
    Activations.TANH = {
        output: function (x) { return Math.tanh(x); },
        der: function (x) {
            var output = Activations.TANH.output(x);
            return 1 - output * output;
        },
        compileToJs: function (arg) { return "Math.tanh(".concat(arg, ")"); }
    };
    Activations.RELU = {
        output: function (x) { return Math.max(0, x); },
        der: function (x) { return x <= 0 ? 0 : 1; },
        compileToJs: function (arg) { return "Math.max(0, ".concat(arg, ")"); }
    };
    Activations.SIGMOID = {
        output: function (x) { return 1 / (1 + Math.exp(-x)); },
        der: function (x) {
            var output = Activations.SIGMOID.output(x);
            return output * (1 - output);
        },
        compileToJs: function (arg) { return "(1 / (1 + Math.exp(-(".concat(arg, "))))"); }
    };
    Activations.LINEAR = {
        output: function (x) { return x; },
        der: function (x) { return 1; },
        compileToJs: function (arg) { return arg; }
    };
    Activations.SINE = {
        output: function (x) { return Math.sin(x); },
        der: function (x) { return Math.cos(x); },
        compileToJs: function (arg) { return "Math.sin(".concat(arg, ")"); }
    };
    Activations.SINC = {
        output: function (x) { return x < 0.000001 ? 1 : Math.sin(x) / x; },
        der: function (x) { return (x * x) < 0.000001 ? 0 : (x * Math.cos(x) - Math.sin(x)) / (x * x); },
        compileToJs: function (arg) { return "Math.sinc(".concat(arg, ")"); }
    };
    Activations.MISH = {
        output: function (x) { return x * Activations.TANH.output(Math.softplus(x)); },
        der: function (x) {
            var sig_x = Activations.SIGMOID.output(x);
            var tanh_sp_x = Activations.TANH.output(Math.softplus(x));
            return tanh_sp_x * x * sig_x * (1 - tanh_sp_x * tanh_sp_x);
        },
        compileToJs: function (arg) { return "mish(".concat(arg, ")"); }
    };
    Activations.GELU = {
        output: function (x) { return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))); },
        der: function (x) {
            var tanhPart = Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)));
            return 0.5 * (1 + tanhPart) + 0.5 * x * (1 - Math.pow(tanhPart, 2)) * Math.sqrt(2 / Math.PI) * (1 + 3 * 0.044715 * Math.pow(x, 2));
        },
        compileToJs: function (arg) { return "gelu(".concat(arg, ")"); }
    };
    Activations.LEAKY_RELU = {
        output: function (x) { return x >= 0 ? x : 0.01 * x; },
        der: function (x) { return x >= 0 ? 1 : 0.01; },
        compileToJs: function (arg) { return "leakyrelu(".concat(arg, ")"); }
    };
    Activations.PReLU = function (alpha) { return ({
        output: function (x) { return x >= 0 ? x : alpha * x; },
        der: function (x) { return x >= 0 ? 1 : alpha; },
        compileToJs: function (arg) { return "prelu(".concat(arg, ")"); }
    }); };
    return Activations;
}());
exports.Activations = Activations;
var RegularizationFunction = (function () {
    function RegularizationFunction() {
    }
    RegularizationFunction.L1 = {
        output: function (w) { return Math.abs(w); },
        der: function (w) { return w < 0 ? -1 : (w > 0 ? 1 : 0); }
    };
    RegularizationFunction.L2 = {
        output: function (w) { return 0.5 * w * w; },
        der: function (w) { return w; }
    };
    return RegularizationFunction;
}());
exports.RegularizationFunction = RegularizationFunction;
var WeightQuantizationFunction = (function () {
    function WeightQuantizationFunction() {
    }
    WeightQuantizationFunction.q16bit = {
        output: function (w) { return Math.round(w * 65536) / 65536; }
    };
    WeightQuantizationFunction.q8bit = {
        output: function (w) { return Math.round(w * 256) / 256; }
    };
    WeightQuantizationFunction.q4bit = {
        output: function (w) { return Math.round(w * 16) / 16; }
    };
    WeightQuantizationFunction.q2bit = {
        output: function (w) { return Math.round(w * 4) / 4; }
    };
    return WeightQuantizationFunction;
}());
exports.WeightQuantizationFunction = WeightQuantizationFunction;
var Link = (function () {
    function Link(source, dest, initZero) {
        this.weight = Math.random() - 0.5;
        this.isDead = false;
        this.errorDer = 0;
        this.accErrorDer = 0;
        this.numAccumulatedDers = 0;
        this.optimizerState = {};
        this.id = source.id + "-" + dest.id;
        this.source = source;
        this.dest = dest;
        if (initZero) {
            this.weight = 0;
        }
    }
    return Link;
}());
exports.Link = Link;
function buildNetwork(networkShape, activation, outputActivation, inputIds, initZero) {
    var numLayers = networkShape.length;
    var id = 1;
    var network = [];
    for (var layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        var isOutputLayer = layerIdx === numLayers - 1;
        var isInputLayer = layerIdx === 0;
        var currentLayer = [];
        network.push(currentLayer);
        var numNodes = networkShape[layerIdx];
        for (var i = 0; i < numNodes; i++) {
            var nodeId = id.toString();
            if (isInputLayer) {
                nodeId = inputIds[i];
            }
            else {
                id++;
            }
            var node = new Node(nodeId, isOutputLayer ? outputActivation : activation, initZero);
            currentLayer.push(node);
            if (layerIdx >= 1) {
                for (var j = 0; j < network[layerIdx - 1].length; j++) {
                    var prevNode = network[layerIdx - 1][j];
                    var link = new Link(prevNode, node, initZero);
                    prevNode.outputs.push(link);
                    node.inputLinks.push(link);
                }
            }
        }
    }
    return network;
}
function forwardProp(network, inputs, weightQuantizationFunction, layerNorm) {
    if (layerNorm === void 0) { layerNorm = false; }
    var inputLayer = network[0];
    if (inputs.length !== inputLayer.length) {
        throw new Error("The number of inputs must match the number of nodes in" +
            " the input layer");
    }
    for (var i = 0; i < inputLayer.length; i++) {
        var node = inputLayer[i];
        node.output = inputs[i];
    }
    var isOutputLayer;
    for (var layerIdx = 1; layerIdx < network.length; layerIdx++) {
        var currentLayer = network[layerIdx];
        isOutputLayer = layerIdx === network.length - 1;
        for (var i = 0; i < currentLayer.length; i++) {
            var node = currentLayer[i];
            node.totalInput = node.bias;
            for (var j = 0; j < node.inputLinks.length; j++) {
                var link = node.inputLinks[j];
                var weight = weightQuantizationFunction ?
                    weightQuantizationFunction.output(link.weight) : link.weight;
                node.totalInput += weight * link.source.output;
            }
        }
        if (layerNorm && !isOutputLayer && currentLayer.length > 1) {
            var mean = 0;
            for (var i = 0; i < currentLayer.length; i++) {
                mean += currentLayer[i].totalInput;
            }
            mean /= currentLayer.length;
            var variance = 0;
            for (var i = 0; i < currentLayer.length; i++) {
                var diff = currentLayer[i].totalInput - mean;
                variance += diff * diff;
            }
            variance /= currentLayer.length;
            var std = Math.sqrt(variance + 1e-8);
            for (var i = 0; i < currentLayer.length; i++) {
                currentLayer[i].totalInput = (currentLayer[i].totalInput - mean) / std;
            }
        }
        for (var i = 0; i < currentLayer.length; i++) {
            var node = currentLayer[i];
            node.output = node.activation.output(node.totalInput);
        }
    }
    return network[network.length - 1][0].output;
}
function backProp(network, target, errorFunc) {
    var outputNode = network[network.length - 1][0];
    outputNode.outputDer = errorFunc.der(outputNode.output, target);
    for (var layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
        var currentLayer = network[layerIdx];
        for (var i = 0; i < currentLayer.length; i++) {
            var node = currentLayer[i];
            node.inputDer = node.outputDer * node.activation.der(node.totalInput);
            node.accInputDer += node.inputDer;
            node.numAccumulatedDers++;
        }
        for (var i = 0; i < currentLayer.length; i++) {
            var node = currentLayer[i];
            for (var j = 0; j < node.inputLinks.length; j++) {
                var link = node.inputLinks[j];
                if (link.isDead) {
                    continue;
                }
                link.errorDer = node.inputDer * link.source.output;
                link.accErrorDer += link.errorDer;
                link.numAccumulatedDers++;
            }
        }
        if (layerIdx === 1) {
            continue;
        }
        var prevLayer = network[layerIdx - 1];
        for (var i = 0; i < prevLayer.length; i++) {
            var node = prevLayer[i];
            node.outputDer = 0;
            for (var j = 0; j < node.outputs.length; j++) {
                var output = node.outputs[j];
                node.outputDer += output.weight * output.dest.inputDer;
            }
        }
    }
}
var OptimizerType;
(function (OptimizerType) {
    OptimizerType["SGD"] = "sgd";
    OptimizerType["MOMENTUM"] = "momentum";
    OptimizerType["RMSPROP"] = "rmsprop";
    OptimizerType["ADAM"] = "adam";
})(OptimizerType || (exports.OptimizerType = OptimizerType = {}));
exports.OPTIMIZER_BETA1 = 0.9;
exports.OPTIMIZER_BETA2 = 0.999;
exports.OPTIMIZER_EPSILON = 1e-8;
function optimizerDelta(grad, learningRate, optimizerType, state) {
    switch (optimizerType) {
        case OptimizerType.SGD:
            return learningRate * grad;
        case OptimizerType.MOMENTUM: {
            state.m = state.m == null ? 0 : state.m;
            state.m = exports.OPTIMIZER_BETA1 * state.m + (1 - exports.OPTIMIZER_BETA1) * grad;
            return learningRate * state.m;
        }
        case OptimizerType.RMSPROP: {
            state.v = state.v == null ? 0 : state.v;
            state.v = exports.OPTIMIZER_BETA2 * state.v + (1 - exports.OPTIMIZER_BETA2) * grad * grad;
            return learningRate * grad / (Math.sqrt(state.v) + exports.OPTIMIZER_EPSILON);
        }
        case OptimizerType.ADAM: {
            state.m = state.m == null ? 0 : state.m;
            state.v = state.v == null ? 0 : state.v;
            state.t = state.t == null ? 0 : state.t;
            state.t += 1;
            state.m = exports.OPTIMIZER_BETA1 * state.m + (1 - exports.OPTIMIZER_BETA1) * grad;
            state.v = exports.OPTIMIZER_BETA2 * state.v + (1 - exports.OPTIMIZER_BETA2) * grad * grad;
            var mHat = state.m / (1 - Math.pow(exports.OPTIMIZER_BETA1, state.t));
            var vHat = state.v / (1 - Math.pow(exports.OPTIMIZER_BETA2, state.t));
            return learningRate * mHat / (Math.sqrt(vHat) + exports.OPTIMIZER_EPSILON);
        }
        default:
            return learningRate * grad;
    }
}
function updateWeights(network, learningRate, regularization, regularizationRate, optimizerType) {
    if (optimizerType === void 0) { optimizerType = OptimizerType.SGD; }
    for (var layerIdx = 1; layerIdx < network.length; layerIdx++) {
        var currentLayer = network[layerIdx];
        for (var i = 0; i < currentLayer.length; i++) {
            var node = currentLayer[i];
            if (node.numAccumulatedDers > 0) {
                var biasGrad = node.accInputDer / node.numAccumulatedDers;
                node.bias -= optimizerDelta(biasGrad, learningRate, optimizerType, node.biasOptimizerState);
                node.accInputDer = 0;
                node.numAccumulatedDers = 0;
            }
            for (var j = 0; j < node.inputLinks.length; j++) {
                var link = node.inputLinks[j];
                if (link.isDead) {
                    continue;
                }
                var regulDer = regularization ?
                    regularization.der(link.weight) : 0;
                if (link.numAccumulatedDers > 0) {
                    var grad = link.accErrorDer / link.numAccumulatedDers;
                    link.weight -= optimizerDelta(grad, learningRate, optimizerType, link.optimizerState);
                    var newLinkWeight = link.weight -
                        (learningRate * regularizationRate) * regulDer;
                    if (regularization === RegularizationFunction.L1 &&
                        link.weight * newLinkWeight < 0) {
                        link.weight = 0;
                        link.isDead = true;
                    }
                    else {
                        link.weight = newLinkWeight;
                    }
                    link.accErrorDer = 0;
                    link.numAccumulatedDers = 0;
                }
            }
        }
    }
}
function forEachNode(network, ignoreInputs, accessor) {
    for (var layerIdx = ignoreInputs ? 1 : 0; layerIdx < network.length; layerIdx++) {
        var currentLayer = network[layerIdx];
        for (var i = 0; i < currentLayer.length; i++) {
            var node = currentLayer[i];
            accessor(node);
        }
    }
}
function getOutputNode(network) {
    return network[network.length - 1][0];
}
function compileNetworkToJs(network) {
    var inputLayer = network[0];
    var js = "function(".concat(inputLayer.map(function (node) { return node.compileToJsName(); }).join(", "), ") {\n");
    for (var layerIdx = 1; layerIdx < network.length; layerIdx++) {
        var currentLayer = network[layerIdx];
        for (var i = 0; i < currentLayer.length; i++) {
            var node = currentLayer[i];
            js += "  const ".concat(node.compileToJsName(), " = ").concat(node.compileToJs(), ";\n");
        }
    }
    js += "  return ".concat(network[network.length - 1][0].compileToJsName(), ";\n");
    js += "}";
    return js;
}

},{}],3:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var nn_1 = require("./nn");
var dataset3d_1 = require("./dataset3d");
var DATASETS = {
    blobs: dataset3d_1.classifyTwoGaussBlobs,
    spheres: dataset3d_1.classifyConcentricSpheres,
    helix: dataset3d_1.classifyHelix,
    swissroll: dataset3d_1.classifySwissRoll
};
var currentDataset = 'blobs';
var numSamples = 200;
var noise = 0.1;
var learningRate = 0.03;
var numHidden = 6;
var useSquaredFeatures = true;
var isTraining = false;
var epoch = 0;
var totalLoss = 0;
var data = [];
var network = [];
var container = document.getElementById('canvas-container');
var WIDTH = container.clientWidth || 600;
var HEIGHT = container.clientHeight || 500;
var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(WIDTH, HEIGHT);
container.appendChild(renderer.domElement);
var scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
var camera = new THREE.PerspectiveCamera(50, WIDTH / HEIGHT, 0.1, 100);
camera.position.set(0, 0, 14);
var ambient = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambient);
var dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(5, 10, 7);
scene.add(dirLight);
var gridHelper = new THREE.GridHelper(12, 12, 0x333355, 0x222244);
scene.add(gridHelper);
var rotGroup = new THREE.Group();
scene.add(rotGroup);
var isDragging = false;
var prevMouse = { x: 0, y: 0 };
var rotX = 0.3;
var rotY = 0.5;
function applyRotation() {
    rotGroup.rotation.x = rotX;
    rotGroup.rotation.y = rotY;
}
applyRotation();
var canvas = renderer.domElement;
canvas.addEventListener('mousedown', function (e) {
    isDragging = true;
    prevMouse = { x: e.clientX, y: e.clientY };
});
window.addEventListener('mouseup', function () { isDragging = false; });
window.addEventListener('mousemove', function (e) {
    if (!isDragging)
        return;
    var dx = e.clientX - prevMouse.x;
    var dy = e.clientY - prevMouse.y;
    rotY += dx * 0.01;
    rotX += dy * 0.01;
    applyRotation();
    prevMouse = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener('touchstart', function (e) {
    isDragging = true;
    prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
});
window.addEventListener('touchend', function () { isDragging = false; });
window.addEventListener('touchmove', function (e) {
    if (!isDragging)
        return;
    var dx = e.touches[0].clientX - prevMouse.x;
    var dy = e.touches[0].clientY - prevMouse.y;
    rotY += dx * 0.01;
    rotX += dy * 0.01;
    applyRotation();
    prevMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
});
canvas.addEventListener('wheel', function (e) {
    camera.position.z = Math.max(4, Math.min(30, camera.position.z + e.deltaY * 0.02));
    e.preventDefault();
}, { passive: false });
var dataPointsMesh = null;
var boundaryMesh = null;
var POS_COLOR = new THREE.Color(0xff4d6d);
var NEG_COLOR = new THREE.Color(0x4d9fff);
function buildDataPoints() {
    if (dataPointsMesh) {
        rotGroup.remove(dataPointsMesh);
        dataPointsMesh.geometry.dispose();
        dataPointsMesh.material.dispose();
        dataPointsMesh = null;
    }
    var positions = [];
    var colors = [];
    for (var _i = 0, data_1 = data; _i < data_1.length; _i++) {
        var pt = data_1[_i];
        positions.push(pt.x, pt.y, pt.z);
        var c = pt.label > 0 ? POS_COLOR : NEG_COLOR;
        colors.push(c.r, c.g, c.b);
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    var mat = new THREE.PointsMaterial({
        size: 0.18,
        vertexColors: true,
        sizeAttenuation: true
    });
    dataPointsMesh = new THREE.Points(geo, mat);
    rotGroup.add(dataPointsMesh);
}
var GRID_RES = 14;
function buildBoundaryPoints() {
    if (boundaryMesh) {
        rotGroup.remove(boundaryMesh);
        boundaryMesh.geometry.dispose();
        boundaryMesh.material.dispose();
        boundaryMesh = null;
    }
    var positions = [];
    var colors = [];
    var range = 4.5;
    var step = (2 * range) / (GRID_RES - 1);
    for (var ix = 0; ix < GRID_RES; ix++) {
        for (var iy = 0; iy < GRID_RES; iy++) {
            for (var iz = 0; iz < GRID_RES; iz++) {
                var x = -range + ix * step;
                var y = -range + iy * step;
                var z = -range + iz * step;
                var inputs = buildInputs(x, y, z);
                var out = (0, nn_1.forwardProp)(network, inputs, null);
                var conf = Math.abs(out);
                if (conf < 0.6) {
                    positions.push(x, y, z);
                    var t = (out + 1) / 2;
                    var r = POS_COLOR.r * t + NEG_COLOR.r * (1 - t);
                    var g = POS_COLOR.g * t + NEG_COLOR.g * (1 - t);
                    var b = POS_COLOR.b * t + NEG_COLOR.b * (1 - t);
                    colors.push(r, g, b);
                }
            }
        }
    }
    if (positions.length === 0)
        return;
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    var mat = new THREE.PointsMaterial({
        size: 0.30,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.45,
        depthWrite: false
    });
    boundaryMesh = new THREE.Points(geo, mat);
    rotGroup.add(boundaryMesh);
}
function buildInputs(x, y, z) {
    var inputs = [x, y, z];
    if (useSquaredFeatures) {
        inputs.push(x * x, y * y, z * z);
    }
    return inputs;
}
function inputIds() {
    var ids = ['x', 'y', 'z'];
    if (useSquaredFeatures) {
        ids.push('x2', 'y2', 'z2');
    }
    return ids;
}
function rebuildNetwork() {
    var numInputs = useSquaredFeatures ? 6 : 3;
    var shape = [numInputs, numHidden, numHidden, 1];
    network = (0, nn_1.buildNetwork)(shape, nn_1.Activations.TANH, nn_1.Activations.TANH, inputIds());
}
function computeLoss() {
    var loss = 0;
    for (var _i = 0, data_2 = data; _i < data_2.length; _i++) {
        var pt = data_2[_i];
        var inputs = buildInputs(pt.x, pt.y, pt.z);
        var out = (0, nn_1.forwardProp)(network, inputs, null);
        loss += nn_1.Errors.SQUARE.error(out, pt.label);
    }
    return loss / data.length;
}
var boundaryUpdateCounter = 0;
var BOUNDARY_UPDATE_EVERY = 20;
function trainStep() {
    for (var _i = 0, data_3 = data; _i < data_3.length; _i++) {
        var pt = data_3[_i];
        var inputs = buildInputs(pt.x, pt.y, pt.z);
        (0, nn_1.forwardProp)(network, inputs, null);
        (0, nn_1.backProp)(network, pt.label, nn_1.Errors.SQUARE);
        (0, nn_1.updateWeights)(network, learningRate, null, 0);
    }
    epoch++;
    totalLoss = computeLoss();
    var lossEl = document.getElementById('loss-value');
    if (lossEl)
        lossEl.textContent = totalLoss.toFixed(4);
    var epochEl = document.getElementById('epoch-value');
    if (epochEl)
        epochEl.textContent = String(epoch);
    boundaryUpdateCounter++;
    if (boundaryUpdateCounter >= BOUNDARY_UPDATE_EVERY) {
        buildBoundaryPoints();
        boundaryUpdateCounter = 0;
    }
}
function animate() {
    requestAnimationFrame(animate);
    if (isTraining) {
        trainStep();
    }
    renderer.render(scene, camera);
}
function reset() {
    epoch = 0;
    totalLoss = 0;
    boundaryUpdateCounter = 0;
    data = DATASETS[currentDataset](numSamples, noise);
    rebuildNetwork();
    buildDataPoints();
    buildBoundaryPoints();
    var lossEl = document.getElementById('loss-value');
    if (lossEl)
        lossEl.textContent = '—';
    var epochEl = document.getElementById('epoch-value');
    if (epochEl)
        epochEl.textContent = '0';
}
function wireControls() {
    var btnPlay = document.getElementById('btn-play');
    var btnPause = document.getElementById('btn-pause');
    var btnReset = document.getElementById('btn-reset');
    var selDataset = document.getElementById('sel-dataset');
    var sliderLR = document.getElementById('slider-lr');
    var sliderNoise = document.getElementById('slider-noise');
    var sliderHidden = document.getElementById('slider-hidden');
    var chkSquared = document.getElementById('chk-squared');
    var lblLR = document.getElementById('lbl-lr');
    var lblNoise = document.getElementById('lbl-noise');
    var lblHidden = document.getElementById('lbl-hidden');
    btnPlay.addEventListener('click', function () {
        isTraining = true;
        btnPlay.disabled = true;
        btnPause.disabled = false;
    });
    btnPause.addEventListener('click', function () {
        isTraining = false;
        btnPlay.disabled = false;
        btnPause.disabled = true;
    });
    btnReset.addEventListener('click', function () {
        isTraining = false;
        btnPlay.disabled = false;
        btnPause.disabled = true;
        reset();
    });
    selDataset.addEventListener('change', function () {
        currentDataset = selDataset.value;
        reset();
    });
    sliderLR.addEventListener('input', function () {
        learningRate = parseFloat(sliderLR.value);
        lblLR.textContent = learningRate.toFixed(3);
    });
    sliderNoise.addEventListener('input', function () {
        noise = parseFloat(sliderNoise.value);
        lblNoise.textContent = noise.toFixed(2);
        reset();
    });
    sliderHidden.addEventListener('input', function () {
        numHidden = parseInt(sliderHidden.value, 10);
        lblHidden.textContent = String(numHidden);
        rebuildNetwork();
        buildBoundaryPoints();
        epoch = 0;
    });
    chkSquared.addEventListener('change', function () {
        useSquaredFeatures = chkSquared.checked;
        rebuildNetwork();
        buildBoundaryPoints();
        epoch = 0;
    });
}
window.addEventListener('resize', function () {
    var w = container.clientWidth;
    var h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
});
wireControls();
reset();
animate();

},{"./dataset3d":1,"./nn":2}]},{},[3]);
