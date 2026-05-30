/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as nn from "./nn";
import {HeatMap, reduceMatrix} from "./heatmap";
import {
  State,
  datasets,
  regDatasets,
  activations,
  optimizers,
  weightInits,
  lrSchedules,
  problems,
  regularizations,
  weightQuantizations,
  getKeyFromValue,
  Problem
} from "./state";
import {Example2D, shuffle} from "./dataset";
import {AppendingLineChart} from "./linechart";
import {ThreeView, Point3D} from "./threeview";
import {
  Example3D,
  classifyTwoGaussBlobs,
  classifyConcentricSpheres,
  classifyHelix,
  classifySwissRoll,
  classifyLinkedRings,
  classifyCheckerboardCube,
  classifyDoubleHelix,
  classifyXOR3D,
  classifyShellVsCore,
  classifySCurve3D
} from "./dataset3d";
import {parseCSV} from "./customdataset";
import * as d3 from 'd3';
import {compile} from 'mathjs';

let mainWidth

// More scrolling
d3.select(".more button").on("click", function() {
  let position = 800;
  d3.transition()
    .duration(1000)
    .tween("scroll", scrollTween(position));
});

function scrollTween(offset) {
  return function() {
    let i = d3.interpolateNumber(window.pageYOffset ||
        document.documentElement.scrollTop, offset);
    return function(t) { scrollTo(0, i(t)); };
  };
}

const RECT_SIZE = 30;
const BIAS_SIZE = 5;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const DENSITY = 100;
const HEATMAP_MIN = -6;
const HEATMAP_MAX = 6;

enum HoverType {
  BIAS, WEIGHT
}

type FN = ((x: number, y: number) => number)

interface InputFeature {
  f: FN | any;
  label?: string;
}

let INPUTS: {[name: string]: InputFeature} = {
  "x": {f: (x, y) => x, label: "x"},
  "y": {f: (x, y) => y, label: "y"},
  "xSquared": {f: (x, y) => x * x, label: "x^2"},
  "ySquared": {f: (x, y) => y * y,  label: "y^2"},
  "xTimesY": {f: (x, y) => x * y, label: "x*y"},
  "sinX": {f: (x, y) => Math.sin(x), label: "sin(x)"},
  "sinY": {f: (x, y) => Math.sin(y), label: "sin(y)"},
  "atan2YX":{f: (x,y) => Math.atan2(y,x), label: "sptheta"},
  "normXY":{f: (x,y) => Math.sqrt(x*x+y*y), label: "spradius"},
  "absx_y":{f: (x,y) => Math.abs(x-y)-3, label: "|x-y|-3"},
  "absx_y_add":{f: (x,y) => Math.abs(x+y)-3, label: "|x+y|-3"},
    
};

let HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Discretize output", "discretize"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Activation", "activation"],
  ["Regularization", "regularization"],
  ["Regularization rate", "regularizationRate"],
  ["Problem type", "problem"],
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
  ["Paint Platform", "paintPlatform"],
];

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    const timer = d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        timer.stop();  // Done.
        return;
      }
      oneStep();
    });
  }
}

let state = State.deserializeState();

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let boundary: {[id: string]: number[][]} = {};
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [HEATMAP_MIN, HEATMAP_MAX];
let heatMap =
    new HeatMap(300, DENSITY, xDomain, xDomain, d3.select("#heatmap"),
        {showAxes: true});
let linkWidthScale = d3.scaleLinear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scaleLinear<string, number>()
                     .domain([-1, 0, 1])
                     .range(["#f59322", "#e8eaeb", "#0877bd"])
                     .clamp(true);
let iter = 0;
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
    ["#777", "black"]);

function makeGUI() {
  d3.select("#reset-button").on("click", () => {
    reset();
    userHasInteracted();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
    oneStep();
  });

  d3.select("#seed").on("input", function() {
    state.seed = (this as any).value;
    Math.seedrandom(state.seed);
    state.serialize();
    userHasInteracted();
  });

  d3.select("#data-regen-button").on("click", () => {
    generateData();
    parametersChanged = true;
  });

  let dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function() {
    let newDataset = datasets[(this as any).dataset.dataset];
    if (newDataset === state.dataset) {
      return; // No-op.
    }
    state.dataset =  newDataset;
    dataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed("selected", true);

  let regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function() {
    let newDataset = regDatasets[(this as any).dataset.regdataset];
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset =  newDataset;
    regDataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed("selected", true);

  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 6) {
      return;
    }
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  // For changing state on different selections
  d3.select("#select-orange").on("change", function() {
    state.editColor = (this as any).checked ? -1 : 1
    state.serialize()
    userHasInteracted()
  });

  d3.select("#select-blue").on("change", function() {
    state.editColor = (this as any).checked ? 1 : -1
    state.serialize()
    userHasInteracted()
  });

  // On drag, we want to paint our canvas with the dots.
  let dragBehavior = d3.drag().on("drag", function(event) {
    let isVisible = d3.select("#select-platform").style("display") === "block"
    if(state.problem === Problem.CLASSIFICATION && isVisible) {
      let [x, y] = d3.pointer(event, this)
      let label = state.editColor
      let padding = 20
      let maxScale = 5.0
      let factor = 23.07
      x -= padding
      y -= padding
      x = x/factor - maxScale
      y = maxScale - y/factor
      state.trainData.push({x, y, label})
      heatMap.updatePoints(state.trainData);
    }
  });

  d3.select("#heatmap").call(dragBehavior);

  let showTestData = d3.select("#show-test-data").on("change", function() {
    state.showTestData = (this as any).checked;
    state.serialize();
    userHasInteracted();
    heatMap.updateTestPoints(state.showTestData ? state.testData : []);
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property("checked", state.showTestData);

  let discretize = d3.select("#discretize").on("change", function() {
    state.discretize = (this as any).checked;
    state.serialize();
    userHasInteracted();
    updateUI();
  });
  // Check/uncheck the checbox according to the current state.
  discretize.property("checked", state.discretize);

  let percTrain = d3.select("#percTrainData").on("input", function() {
    state.percTrainData = (this as any).value;
    d3.select("label[for='percTrainData'] .value").text((this as any).value);
    generateData();
    parametersChanged = true;
    reset();
  });
  percTrain.property("value", state.percTrainData);
  d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

  let noise = d3.select("#noise").on("input", function() {
    state.noise = (this as any).value;
    d3.select("label[for='noise'] .value").text((this as any).value);
    generateData();
    parametersChanged = true;
    reset();
  });
  let currentMax = parseInt(noise.property("max"));
  if (state.noise > currentMax) {
    if (state.noise <= 80) {
      noise.property("max", state.noise);
    } else {
      state.noise = 50;
    }
  } else if (state.noise < 0) {
    state.noise = 0;
  }
  noise.property("value", state.noise);
  d3.select("label[for='noise'] .value").text(state.noise);

  let batchSize = d3.select("#batchSize").on("input", function() {
    state.batchSize = (this as any).value;
    d3.select("label[for='batchSize'] .value").text((this as any).value);
    parametersChanged = true;
    reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  let activationDropdown = d3.select("#activations").on("change", function() {
    state.activation = activations[(this as any).value];
    parametersChanged = true;
    reset();
  });
  activationDropdown.property("value",
      getKeyFromValue(activations, state.activation));

  let learningRate = d3.select("#learningRate").on("change", function() {
    state.learningRate = +(this as any).value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  learningRate.property("value", state.learningRate);

  let optimizerDropdown = d3.select("#optimizer").on("change", function() {
    state.optimizer = (this as any).value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    reset();
  });
  optimizerDropdown.property("value", state.optimizer);

  let layerNormCheckbox = d3.select("#layer-norm").on("change", function() {
    state.layerNorm = (this as any).checked;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    reset();
  });
  layerNormCheckbox.property("checked", state.layerNorm);

  let weightInitDropdown = d3.select("#weight-init").on("change", function() {
    state.weightInit = (this as any).value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    reset();
  });
  weightInitDropdown.property("value", state.weightInit);

  let lrScheduleDropdown = d3.select("#lr-schedule").on("change", function() {
    state.lrSchedule = (this as any).value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  lrScheduleDropdown.property("value", state.lrSchedule);

  let dropoutSlider = d3.select("#dropout").on("input", function() {
    state.dropout = +(this as any).value;
    d3.select("label[for='dropout'] .value").text(state.dropout);
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  dropoutSlider.property("value", state.dropout);
  d3.select("label[for='dropout'] .value").text(state.dropout);

  let gradClipSlider = d3.select("#grad-clip").on("input", function() {
    state.gradClip = +(this as any).value;
    d3.select("label[for='grad-clip'] .value").text(state.gradClip);
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  gradClipSlider.property("value", state.gradClip);
  d3.select("label[for='grad-clip'] .value").text(state.gradClip);

  let weightDecaySlider = d3.select("#weight-decay").on("input", function() {
    state.weightDecay = +(this as any).value;
    d3.select("label[for='weight-decay'] .value").text(state.weightDecay);
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  weightDecaySlider.property("value", state.weightDecay);
  d3.select("label[for='weight-decay'] .value").text(state.weightDecay);

  let batchNormCheckbox = d3.select("#batch-norm").on("change", function() {
    state.batchNorm = (this as any).checked;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    reset();
  });
  batchNormCheckbox.property("checked", state.batchNorm);

  let regularDropdown = d3.select("#regularizations").on("change",
      function() {
    state.regularization = regularizations[(this as any).value];
    parametersChanged = true;
    state.serialize();
    userHasInteracted();
  });
  regularDropdown.property("value",
      getKeyFromValue(regularizations, state.regularization));

  let regularRate = d3.select("#regularRate").on("change", function() {
    state.regularizationRate = +(this as any).value;
    parametersChanged = true;
    state.serialize();
    userHasInteracted();
  });
  regularRate.property("value", state.regularizationRate);

  let weightQuantizationDropdown = d3.select("#weightQuantization").on("change", 
      function() {
    state.weightQuantization = weightQuantizations[(this as any).value];
    parametersChanged = true;
    state.serialize();
    userHasInteracted();
  });
  weightQuantizationDropdown.property("value",
      getKeyFromValue(weightQuantizations, state.weightQuantization));

  let problem = d3.select("#problem").on("change", function() {
    state.problem = problems[(this as any).value];
    togglePaintSelection();
    generateData();
    drawDatasetThumbnails();
    parametersChanged = true;
    reset();
  });
  problem.property("value", getKeyFromValue(problems, state.problem));

  // Add scale to the gradient color map.
  let x = d3.scaleLinear().domain([-1, 1]).range([0, 144]);
  let xAxis = d3.axisBottom(x)
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d") as any);
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Listen for css-responsive changes and redraw the svg network.

  window.addEventListener("resize", () => {
    let newWidth = document.querySelector("#main-part")
        .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      drawNetwork(network);
      updateUI(true);
    }
  });

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  }
}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
            .style("stroke-dashoffset", -iter / 3)
            .style("stroke-width", linkWidthScale(Math.abs(link.weight)))
            .style("stroke", colorScale(link.weight))
            .datum(link);
      }
    }
  }
}

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, node?: nn.Node) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr("class", "node")
    .attr("id", `node${nodeId}`)
    .attr("transform", `translate(${x},${y})`);

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", RECT_SIZE)
    .attr("height", RECT_SIZE);
  let activeOrNotClass = state[nodeId] ? "active" : "inactive";
  if (!isInput && node && node.frozen) {
    nodeGroup.classed("frozen", true);
  }
  if (isInput) {
    let label = INPUTS[nodeId].label != null ?
        INPUTS[nodeId].label : nodeId;
    // Draw the input label.
    let text = nodeGroup.append("text")
      .attr("class", "main-label")
      .attr("x", -10)
      .attr("y", RECT_SIZE / 2)
      .attr("text-anchor", "end");
    if (/[_^]/.test(label)) {
      let myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        let prefix = myArray[1];
        let sep = myArray[2];
        let suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text.append("tspan")
        .attr("baseline-shift", sep === "_" ? "sub" : "super")
        .style("font-size", "9px")
        .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
    nodeGroup.classed(activeOrNotClass, true);
  }
  if (!isInput) {
    // Draw the node's bias.
    nodeGroup.append("rect")
      .attr("id", `bias-${nodeId}`)
      .attr("x", -BIAS_SIZE - 2)
      .attr("y", RECT_SIZE - BIAS_SIZE + 3)
      .attr("width", BIAS_SIZE)
      .attr("height", BIAS_SIZE)
      .on("mouseenter", function(event) {
        updateHoverCard(HoverType.BIAS, node, d3.pointer(event, container.node()));
      }).on("mouseleave", function() {
        updateHoverCard(null);
      });
  }

  // Draw the node's canvas.
  let div = d3.select("#network").insert("div", ":first-child")
    .attr("id", `canvas-${nodeId}`)
    .attr("class", "canvas")
    .style("position", "absolute")
    .style("left", `${x + 3}px`)
    .style("top", `${y + 3}px`)
    .on("mouseenter", function() {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nodeId], state.discretize);
    })
    .on("mouseleave", function() {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nn.getOutputNode(network).id],
          state.discretize);
    });
  if (isInput) {
    div.on("click", function() {
      state[nodeId] = !state[nodeId];
      parametersChanged = true;
      reset();
    });
    div.style("cursor", "pointer");
  }
  if (isInput) {
    div.classed(activeOrNotClass, true);
  }
  let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
      xDomain, div, {noSvg: true});
  div.datum({heatmap: nodeHeatMap, id: nodeId});

}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();
  d3.select("#network").selectAll("div.plus-minus-neurons").remove();

  // Get the width of the svg container.
  let padding = 3;
  let co = d3.select(".column.output").node() as HTMLDivElement;
  let cf = d3.select(".column.features").node() as HTMLDivElement;
  let width = co.offsetLeft - cf.offsetLeft;
  svg.attr("width", width);

  // Map of all node coordinates.
  let node2coord: {[id: string]: {cx: number, cy: number}} = {};
  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);
  // Draw the network layer by layer.
  let numLayers = network.length;
  let featureWidth = 118;
  let layerScale: any = d3.scalePoint()
      .domain(d3.range(1, numLayers - 1).map(String))
      .range([featureWidth, width - RECT_SIZE])
      .padding(0.7);
  let layerX = (layerIdx: number) => layerScale(String(layerIdx)) as number;
  let nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 25);


  let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  let calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout = null;
  let targetIdWithCallout = null;

  // Draw the input layer separately.
  let cx = RECT_SIZE / 2 + 50;
  let nodeIds = Object.keys(INPUTS);
  let maxY = nodeIndexScale(nodeIds.length);
  nodeIds.forEach((nodeId, i) => {
    let cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[nodeId] = {cx, cy};
    drawNode(cx, cy, nodeId, true, container);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    let numNodes = network[layerIdx].length;
    let cx = layerX(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerX(layerIdx), layerIdx);
    for (let i = 0; i < numNodes; i++) {
      let node = network[layerIdx][i];
      let cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = {cx, cy};
      drawNode(cx, cy, node.id, false, container, node);

      // Show callout to thumbnails.
      let numNodes = network[layerIdx].length;
      let nextNumNodes = network[layerIdx + 1].length;
      if (idWithCallout == null &&
          i === numNodes - 1 &&
          nextNumNodes <= numNodes) {
        calloutThumb
          .style("display", null)
          .style("top", `${20 + 3 + cy}px`)
          .style("left", `${cx}px`);
        idWithCallout = node.id;
      }

      // Draw links.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let path: SVGPathElement = drawLink(link, node2coord, network,
            container, j === 0, j, node.inputLinks.length).node() as any;
        // Show callout to weights.
        let prevLayer = network[layerIdx - 1];
        let lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (targetIdWithCallout == null &&
            i === numNodes - 1 &&
            link.source.id === lastNodePrevLayer.id &&
            (link.source.id !== idWithCallout || numLayers <= 5) &&
            link.dest.id !== idWithCallout &&
            prevLayer.length >= numNodes) {
          let midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
          calloutWeights
            .style("display", null)
            .style("top", `${midPoint.y + 5}px`)
            .style("left", `${midPoint.x + 3}px`);
          targetIdWithCallout = link.dest.id;
        }
      }
    }
  }

  // Draw the output node separately.
  cx = width + RECT_SIZE / 2;
  let node = network[numLayers - 1][0];
  let cy = nodeIndexScale(0) + RECT_SIZE / 2;
  node2coord[node.id] = {cx, cy};
  // Draw links.
  for (let i = 0; i < node.inputLinks.length; i++) {
    let link = node.inputLinks[i];
    drawLink(link, node2coord, network, container, i === 0, i,
        node.inputLinks.length);
  }
  // Adjust the height of the svg.
  svg.attr("height", maxY);

  // Adjust the height of the features column.
  let height = Math.max(
    getRelativeHeight(calloutThumb),
    getRelativeHeight(calloutWeights),
    getRelativeHeight(d3.select("#network"))
  );
  d3.select(".column.features").style("height", height + "px");

  // Now "draw" it as JavaScript
  d3.select("#network-as-javascript").text(nn.compileNetworkToJs(network));
}

function getRelativeHeight(selection) {
  let node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

function addPlusMinusControl(x: number, layerIdx: number) {
  let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style("left", `${x - 10}px`);

  let i = layerIdx - 1;
  let firstRow = div.append("div").attr("class", `ui-numNodes${layerIdx}`);
  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons >= 800) {
          return;
        }
        state.networkShape[i]++;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("add");

  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons <= 1) {
          return;
        }
        state.networkShape[i]--;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("remove");

  let suffix = state.networkShape[i] > 1 ? "s" : "";
  div.append("div").text(
    state.networkShape[i] + " neuron" + suffix
  );
}

function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
    coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (type == null) {
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", () => {
    hovercard.select(".value").style("display", "none");
    let input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function() {
      if ((this as any).value != null && (this as any).value !== "") {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +(this as any).value;
        } else {
          (nodeOrLink as nn.Node).bias = +(this as any).value;
        }
        updateUI();
      }
    });
    input.on("keypress", (event) => {
      if ((event as any).keyCode === 13) {
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    (input.node() as HTMLInputElement).focus();
  });
  let value = (type === HoverType.WEIGHT) ?
    (nodeOrLink as nn.Link).weight :
    (nodeOrLink as nn.Node).bias;
  let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
  hovercard
    .style("left", `${coordinates[0] + 20}px`)
    .style("top", `${coordinates[1]}px`)
    .style("display", "block");
  hovercard.select(".type").text(name);
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
  hovercard.select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

/**
 * Replicates the old d3.svg.diagonal() (removed in d3 v4+) using the
 * projection d => [d.y, d.x]. Produces a cubic bezier between source and
 * target points, interpolating the control points along the y-axis.
 */
function makeDiagonal(d: {source: {x: number, y: number},
    target: {x: number, y: number}}): string {
  let project = (p: {x: number, y: number}) => [p.y, p.x];
  let m = (d.source.y + d.target.y) / 2;
  let points = [
    d.source,
    {x: d.source.x, y: m},
    {x: d.target.x, y: m},
    d.target
  ].map(project);
  return `M${points[0]}C${points[1]} ${points[2]} ${points[3]}`;
}

function drawLink(
    input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
    network: nn.Node[][], container,
    isFirst: boolean, index: number, length: number) {
  let line = container.insert("path", ":first-child");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let datum = {
    source: {
      y: source.cx + RECT_SIZE / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = makeDiagonal(datum);
  line.attr("marker-start", "url(#markerArrow)")
    .attr("class", "link")
    .attr("id", "link" + input.source.id + "-" + input.dest.id)
    .attr("d", diagonal);

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container.append("path")
    .attr("d", diagonal)
    .attr("class", "link-hover")
    .on("mouseenter", function(event) {
      updateHoverCard(HoverType.WEIGHT, input, d3.pointer(event, this));
    }).on("mouseleave", function() {
      updateHoverCard(null);
    });
  return line;
}

/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
  if (firstTime) {
    boundary = {};
    nn.forEachNode(network, true, node => {
      boundary[node.id] = new Array(DENSITY);
    });
    // Go through all predefined inputs.
    for (let nodeId in INPUTS) {
      boundary[nodeId] = new Array(DENSITY);
    }
  }
  let xScale = d3.scaleLinear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scaleLinear().domain([DENSITY - 1, 0]).range(xDomain);

  let i = 0, j = 0;
  for (i = 0; i < DENSITY; i++) {
    if (firstTime) {
      nn.forEachNode(network, true, node => {
        boundary[node.id][i] = new Array(DENSITY);
      });
      // Go through all predefined inputs.
      for (let nodeId in INPUTS) {
        boundary[nodeId][i] = new Array(DENSITY);
      }
    }
    for (j = 0; j < DENSITY; j++) {
      // 1 for points inside the circle, and 0 for points outside the circle.
      let x = xScale(i);
      let y = yScale(j);
      let input = constructInput(x, y);
      nn.forwardProp(network, input, state.weightQuantization, state.layerNorm);
      nn.forEachNode(network, true, node => {
        boundary[node.id][i][j] = node.output;
      });
      if (firstTime) {
        // Go through all predefined inputs.
        for (let nodeId in INPUTS) {
          if (INPUTS[nodeId].f instanceof Function) {
            boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
          } else {
            boundary[nodeId][i][j] = INPUTS[nodeId].f.eval({x:x,y:y});
          }
        }
      }
    }
  }
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let output = nn.forwardProp(network, input, state.weightQuantization, state.layerNorm);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

// Confusion matrix: [actualOrange][predictedOrange|predictedBlue], etc.
// Rows = actual (Orange=+1, Blue=-1), cols = predicted.
interface ClassMetrics {
  accuracy: number;
  // matrix[0] = actual Orange, matrix[1] = actual Blue.
  // each row: [predicted Orange, predicted Blue].
  matrix: number[][];
}

// Treat 0 as positive consistently.
function predSign(v: number): number {
  return v >= 0 ? 1 : -1;
}

function computeClassMetrics(network: nn.Node[][],
    dataPoints: Example2D[]): ClassMetrics {
  let matrix = [[0, 0], [0, 0]];
  let correct = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let output = nn.forwardProp(network, input, state.weightQuantization,
        state.layerNorm);
    let predicted = predSign(output);
    let actual = predSign(dataPoint.label);
    if (predicted === actual) {
      correct++;
    }
    let row = actual === 1 ? 0 : 1;
    let col = predicted === 1 ? 0 : 1;
    matrix[row][col]++;
  }
  return {
    accuracy: dataPoints.length ? correct / dataPoints.length : 0,
    matrix
  };
}

function updateConfusionMatrix(metrics: ClassMetrics): void {
  let container = d3.select("#confusion-matrix");
  container.style("display", null);
  let m = metrics.matrix;
  let maxCell = Math.max(1, m[0][0], m[0][1], m[1][0], m[1][1]);
  let labels = ["Orange", "Blue"];
  let colorFor = (actualIdx: number, count: number) => {
    let base = actualIdx === 0 ? [255, 117, 84] : [0, 124, 197];
    let t = count / maxCell;
    let r = Math.round(255 + (base[0] - 255) * t);
    let g = Math.round(255 + (base[1] - 255) * t);
    let b = Math.round(255 + (base[2] - 255) * t);
    return "rgb(" + r + "," + g + "," + b + ")";
  };

  let html = "<div class=\"cm-title\">Confusion matrix (test)</div>";
  html += "<table class=\"cm-table\"><thead><tr>" +
      "<th class=\"cm-corner\"></th>" +
      "<th colspan=\"2\" class=\"cm-predhead\">Predicted</th></tr>" +
      "<tr><th class=\"cm-corner\">Actual</th>" +
      "<th>Orange</th><th>Blue</th></tr></thead><tbody>";
  for (let r = 0; r < 2; r++) {
    html += "<tr><th>" + labels[r] + "</th>";
    for (let c = 0; c < 2; c++) {
      html += "<td style=\"background:" + colorFor(r, m[r][c]) + "\">" +
          m[r][c] + "</td>";
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  container.html(html);
}

function updateUI(firstStep = false) {
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  // Update the bias values visually.
  updateBiasesUI(network);
  // Get the decision boundary of the network.
  updateDecisionBoundary(network, firstStep);
  let selectedId = selectedNodeId != null ?
      selectedNodeId : nn.getOutputNode(network).id;
  heatMap.updateBackground(boundary[selectedId], state.discretize);

  // Update all decision boundaries.
  d3.select("#network").selectAll("div.canvas")
      .each(function(data: {heatmap: HeatMap, id: string}) {
    data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
        state.discretize);
  });

  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  d3.select("#loss-test").text(humanReadable(lossTest));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  d3.select("#effective-lr").text(effectiveLearningRate().toPrecision(3));
  lineChart.addDataPoint([lossTrain, lossTest]);
  updateClassificationMetricsUI();
  updateAnalysis();

  // Now "draw" it as JavaScript
  d3.select("#network-as-javascript").text(nn.compileNetworkToJs(network));
}

function pct(v: number): string {
  return (v * 100).toFixed(1) + "%";
}

// Refresh the accuracy readouts and confusion matrix. Only meaningful for
// classification problems; hidden entirely for regression.
function updateClassificationMetricsUI(): void {
  let accTrain = d3.select("#acc-train");
  let accTest = d3.select("#acc-test");
  let cm = d3.select("#confusion-matrix");

  if (state.problem !== Problem.CLASSIFICATION) {
    d3.selectAll(".acc-stat").style("display", "none");
    cm.style("display", "none");
    return;
  }
  d3.selectAll(".acc-stat").style("display", null);

  if (state.threeD) {
    if (threeData && threeData.length) {
      let m3d = compute3DClassMetrics(network, threeData as Example3D[]);
      accTrain.text(pct(m3d.accuracy));
      accTest.text(pct(m3d.accuracy));
      updateConfusionMatrix(m3d);
    } else {
      accTrain.text("—");
      accTest.text("—");
      cm.style("display", "none");
    }
    return;
  }

  let mTrain = computeClassMetrics(network, state.trainData);
  let mTest = computeClassMetrics(network, state.testData);
  accTrain.text(pct(mTrain.accuracy));
  accTest.text(pct(mTest.accuracy));
  updateConfusionMatrix(mTest);
}

function compute3DClassMetrics(net: nn.Node[][],
    points: Example3D[]): ClassMetrics {
  let matrix = [[0, 0], [0, 0]];
  let correct = 0;
  for (let p of points) {
    let output = nn.forwardProp(net, construct3DInput(p.x, p.y, p.z),
        state.weightQuantization, state.layerNorm);
    let predicted = predSign(output);
    let actual = predSign(p.label);
    if (predicted === actual) {
      correct++;
    }
    matrix[actual === 1 ? 0 : 1][predicted === 1 ? 0 : 1]++;
  }
  return { accuracy: points.length ? correct / points.length : 0, matrix };
}

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      if (INPUTS[inputName].f instanceof Function) {
        input.push(INPUTS[inputName].f(x, y));
      } else {
        input.push(INPUTS[inputName].f.eval({x:x,y:y}))
      }
    }
  }
  return input;
}

// ============================================================================
// Adversarial sampling (feature-agnostic, via finite differences on raw x,y).
// ============================================================================

/** Loss of the current network on a single raw (x,y) point with given label. */
function pointLoss(x: number, y: number, label: number): number {
  let out = nn.forwardProp(network, constructInput(x, y),
      state.weightQuantization, state.layerNorm);
  return nn.Errors.SQUARE.error(out, label);
}

/**
 * Gradient of loss wrt raw (x,y) via central finite differences. This is
 * feature-agnostic: features are recomputed through constructInput() each call,
 * so it works regardless of which input features are enabled.
 */
function rawInputGradient(x: number, y: number, label: number): [number, number] {
  let h = 1e-3;
  let dx = (pointLoss(x + h, y, label) - pointLoss(x - h, y, label)) / (2 * h);
  let dy = (pointLoss(x, y + h, label) - pointLoss(x, y - h, label)) / (2 * h);
  return [dx, dy];
}

function fgsm(point: Example2D, eps: number): Example2D {
  let [gx, gy] = rawInputGradient(point.x, point.y, point.label);
  return {
    x: point.x + eps * Math.sign(gx),
    y: point.y + eps * Math.sign(gy),
    label: point.label
  };
}

function pgd(point: Example2D, eps: number, steps: number,
    stepSize: number): Example2D {
  let ax = point.x;
  let ay = point.y;
  for (let s = 0; s < steps; s++) {
    let [gx, gy] = rawInputGradient(ax, ay, point.label);
    ax += stepSize * Math.sign(gx);
    ay += stepSize * Math.sign(gy);
    ax = Math.max(point.x - eps, Math.min(point.x + eps, ax));
    ay = Math.max(point.y - eps, Math.min(point.y + eps, ay));
  }
  return {x: ax, y: ay, label: point.label};
}

/** Perturb a point using the currently selected method/epsilon. */
function perturb(point: Example2D): Example2D {
  let eps = state.advEpsilon;
  if (state.advMethod === "pgd") {
    return pgd(point, eps, 10, eps / 4);
  }
  return fgsm(point, eps);
}

function accuracy(points: Example2D[]): number {
  if (points.length === 0) return 1;
  let correct = 0;
  for (let p of points) {
    let out = nn.forwardProp(network, constructInput(p.x, p.y),
        state.weightQuantization, state.layerNorm);
    if (Math.sign(out) === Math.sign(p.label)) correct++;
  }
  return correct / points.length;
}

// ============================================================================
// In-place 3D mode.
// ============================================================================

let threeView: ThreeView = null;
let threeData: Example3D[] = [];
const THREE_GENERATORS: {[k: string]: (n: number, noise: number) => Example3D[]} = {
  "blobs": classifyTwoGaussBlobs,
  "spheres": classifyConcentricSpheres,
  "helix": classifyHelix,
  "swiss-roll": classifySwissRoll,
  "linked-rings": classifyLinkedRings,
  "checkerboard-cube": classifyCheckerboardCube,
  "double-helix": classifyDoubleHelix,
  "xor3d": classifyXOR3D,
  "shell-vs-core": classifyShellVsCore,
  "s-curve-3d": classifySCurve3D
};

function construct3DInput(x: number, y: number, z: number): number[] {
  return [x, y, z];
}

function generate3DData(): void {
  Math.seedrandom(state.seed);
  let gen = THREE_GENERATORS[state.threeDDataset] || classifyTwoGaussBlobs;
  threeData = gen(NUM_SAMPLES_CLASSIFY, state.noise / 100);
  if (threeView) {
    threeView.setDataPoints(threeData as Point3D[]);
    threeView.render();
  }
}

function get3DLoss(net: nn.Node[][], points: Example3D[]): number {
  let loss = 0;
  for (let p of points) {
    let out = nn.forwardProp(net, construct3DInput(p.x, p.y, p.z),
        state.weightQuantization, state.layerNorm);
    loss += nn.Errors.SQUARE.error(out, p.label);
  }
  return points.length ? loss / points.length : 0;
}

function update3DBoundary(): void {
  if (!threeView) return;
  let N = 10;
  let voxels: {x: number, y: number, z: number, value: number}[] = [];
  let scale = (i: number) => -5 + (10 * i) / (N - 1);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      for (let k = 0; k < N; k++) {
        let x = scale(i), y = scale(j), z = scale(k);
        let v = nn.forwardProp(network, construct3DInput(x, y, z),
            state.weightQuantization, state.layerNorm);
        voxels.push({x, y, z, value: Math.max(-1, Math.min(1, v))});
      }
    }
  }
  threeView.setBoundary(voxels);
  threeView.render();
}

function oneStep3D(): void {
  let optimizerType = optimizers[state.optimizer] || nn.OptimizerType.SGD;
  threeData.forEach((point, i) => {
    nn.forwardProp(network, construct3DInput(point.x, point.y, point.z),
        state.weightQuantization, state.layerNorm);
    nn.backProp(network, point.label, nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(network, state.learningRate, state.regularization,
          state.regularizationRate, optimizerType);
    }
  });
  lossTrain = get3DLoss(network, threeData);
  lossTest = lossTrain;
  d3.select("#loss-train").text(lossTrain.toFixed(3));
  d3.select("#loss-test").text(lossTest.toFixed(3));
  updateClassificationMetricsUI();
  d3.select("#iter-number").text(iter);
  lineChart.addDataPoint([lossTrain, lossTest]);
  if (iter % 5 === 0) {
    update3DBoundary();
  }
}

function reset3D(): void {
  iter = 0;
  let shape = [3].concat(state.networkShape).concat([1]);
  network = nn.buildNetwork(shape, state.activation, nn.Activations.TANH,
      ["x", "y", "z"], state.initZero);
  generate3DData();
  lossTrain = get3DLoss(network, threeData);
  lossTest = lossTrain;
  drawNetwork(network);
  d3.select("#loss-train").text(lossTrain.toFixed(3));
  d3.select("#loss-test").text(lossTest.toFixed(3));
  updateClassificationMetricsUI();
  update3DBoundary();
}

function enterThreeD(): void {
  d3.select("#heatmap").style("display", "none");
  let container = document.getElementById("threeview");
  container.style.display = "block";
  if (!threeView) {
    threeView = new ThreeView(container, 300, 300);
    threeView.enableControls();
  }
  reset3D();
}

function exitThreeD(): void {
  if (threeView) {
    threeView.dispose();
    threeView = null;
  }
  d3.select("#threeview").style("display", "none");
  d3.select("#heatmap").style("display", null);
  reset();
}

/**
 * Computes the effective learning rate for the current iteration based on the
 * selected schedule. "constant" returns the base rate unchanged.
 */
function effectiveLearningRate(): number {
  let base = state.learningRate;
  let t = iter;
  const TOTAL = 1000;  // reference horizon for annealing schedules
  switch (state.lrSchedule) {
    case "step":
      // Halve the rate every 200 iterations.
      return base * Math.pow(0.5, Math.floor(t / 200));
    case "exponential":
      return base * Math.exp(-0.002 * t);
    case "cosine":
      return base * 0.5 * (1 + Math.cos(Math.PI * Math.min(t, TOTAL) / TOTAL));
    case "warmup-decay": {
      let warmup = 100;
      if (t < warmup) return base * (t / warmup);
      let p = Math.min(1, (t - warmup) / (TOTAL - warmup));
      return base * (1 - p);
    }
    case "onecycle": {
      let half = TOTAL / 2;
      let tt = Math.min(t, TOTAL);
      // Ramp 0.1*base -> base over first half, back down over second half.
      if (tt < half) {
        return base * (0.1 + 0.9 * (tt / half));
      }
      return base * (1 - 0.9 * ((tt - half) / half));
    }
    case "constant":
    default:
      return base;
  }
}

function oneStep(): void {
  iter++;
  if (state.threeD) {
    oneStep3D();
    return;
  }
  let optimizerType = optimizers[state.optimizer] || nn.OptimizerType.SGD;
  let lr = effectiveLearningRate();
  state.trainData.forEach((point, i) => {
    let input = constructInput(point.x, point.y);
    nn.forwardProp(network, input, state.weightQuantization, state.layerNorm,
        state.dropout, true, state.batchNorm);
    nn.backProp(network, point.label, nn.Errors.SQUARE);
    if (state.adversarialTraining) {
      // Train also on an on-the-fly adversarial perturbation of this point.
      let adv = perturb(point);
      nn.forwardProp(network, constructInput(adv.x, adv.y),
          state.weightQuantization, state.layerNorm,
          state.dropout, true, state.batchNorm);
      nn.backProp(network, point.label, nn.Errors.SQUARE);
    }
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(network, lr, state.regularization,
          state.regularizationRate, optimizerType, state.gradClip,
          state.weightDecay);
    }
  });
  // Compute the loss.
  lossTrain = getLoss(network, state.trainData);
  lossTest = getLoss(network, state.testData);
  updateUI();
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}

function reset(onStartup=false) {
  lineChart.reset();
  trainingHistory = [];
  weightMagHistory = [];
  analysisStep = 0;
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  if (state.threeD && threeView) {
    reset3D();
    return;
  }

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);

  togglePaintSelection()
  // Correct radio button on reset
  let radioColor = state.editColor === - 1 ? "#select-orange" : "#select-blue";
  d3.select(radioColor).attr("checked", "checked")

  // Make a simple network.
  iter = 0;
  let numInputs = constructInput(0 , 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  let outputActivation = (state.problem === Problem.REGRESSION) ?
      nn.Activations.LINEAR : nn.Activations.TANH;
  network = nn.buildNetwork(shape, state.activation, outputActivation,
constructInputIds(), state.initZero);
  nn.applyWeightInit(network, weightInits[state.weightInit]);
  applyFrozenLayers();
  lossTrain = getLoss(network, state.trainData);
  lossTest = getLoss(network, state.testData);
  drawNetwork(network);
  updateUI(true);
};

function initTutorial() {
  if (state.tutorial == null || state.tutorial === '' || state.hideText) {
    return;
  }
  // Remove all other text.
  d3.selectAll("article div.l--body").remove();
  let tutorial = d3.select("article").append("div")
    .attr("class", "l--body");
  // Insert tutorial text.
  d3.html(`tutorials/${state.tutorial}.html`).then(htmlFragment => {
    tutorial.node().appendChild(htmlFragment);
    // If the tutorial has a <title> tag, set the page title to that.
    let title = tutorial.select("title");
    if (title.size()) {
      d3.select("header h1")
        .style("margin-top", "20px")
        .style("margin-bottom", "20px")
        .text(title.text());
      document.title = title.text();
    }
  }).catch(err => { throw err; });
}

function drawDatasetThumbnails() {
  function renderThumbnail(canvas, dataGenerator) {
    let w = 100;
    let h = 100;
    canvas.setAttribute("width", w);
    canvas.setAttribute("height", h);
    let context = canvas.getContext("2d");
    let data = dataGenerator(200, 0);
    data.forEach(function(d) {
      context.fillStyle = colorScale(d.label);
      context.fillRect(w * (d.x + 6) / 12, h - h * (d.y + 6) / 12 - 4, 4, 4);
    });
    d3.select(canvas.parentNode).style("display", null);
  }
  d3.selectAll(".dataset").style("display", "none");

  if (state.problem === Problem.CLASSIFICATION) {
    for (let dataset in datasets) {
      let canvas: any =
          document.querySelector(`canvas[data-dataset=${dataset}]`);
      let dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
  if (state.problem === Problem.REGRESSION) {
    for (let regDataset in regDatasets) {
      let canvas: any =
          document.querySelector(`canvas[data-regDataset=${regDataset}]`);
      let dataGenerator = regDatasets[regDataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
}

function hideControls() {
  // Set display:none to all the UI elements that are hidden.
  let hiddenProps = state.getHiddenProps();
  hiddenProps.forEach(prop => {
    let controls = d3.selectAll(`.ui-${prop}`);
    if (controls.size() === 0) {
      console.warn(`0 html elements found with class .ui-${prop}`);
    }
    controls.style("display", "none");
  });

  // Also add checkbox for each hidable control in the "use it in classroom"
  // section.
  let hideControls = d3.select(".hide-controls");
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    let label = hideControls.append("label")
      .attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
    let input = label.append("input")
      .attr("type", "checkbox")
      .attr("class", "mdl-checkbox__input");
    if (hiddenProps.indexOf(id) === -1) {
      input.attr("checked", "true");
    }
    input.on("change", function() {
      state.setHideProperty(id, !(this as any).checked);
      state.serialize();
      userHasInteracted();
      d3.select(".hide-controls-link")
        .attr("href", window.location.href);
    });
    label.append("span")
      .attr("class", "mdl-checkbox__label label")
      .text(text);
  });
  d3.select(".hide-controls-link")
    .attr("href", window.location.href);
}

function togglePaintSelection() {
  let visiblity = state.problem === Problem.CLASSIFICATION ? "" : "none"
  d3.select("#select-platform").style("display", visiblity);
}

function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    userHasInteracted();
  }
  Math.seedrandom(state.seed);
  let numSamples = (state.problem === Problem.REGRESSION) ?
      NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
  if (state.dataset === datasets.three) {
    numSamples = NUM_SAMPLES_CLASSIFY * 2
  }
  let generator = state.problem === Problem.CLASSIFICATION ?
      state.dataset : state.regDataset;
  let data = generator(numSamples, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  state.trainData = data.slice(0, splitIndex);
  state.testData = data.slice(splitIndex);
  heatMap.updatePoints(state.trainData);
  heatMap.updateTestPoints(state.showTestData ? state.testData : []);
}

let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) {
    return;
  }
  firstInteraction = false;
  let page = 'index';
  if (state.tutorial != null && state.tutorial !== '') {
    page = `/v/tutorials/${state.tutorial}`;
  }
  ga('set', 'page', page);
  ga('send', 'pageview', {'sessionControl': 'start'});
}

function simulationStarted() {
  ga('send', {
    hitType: 'event',
    eventCategory: 'Starting Simulation',
    eventAction: parametersChanged ? 'changed' : 'unchanged',
    eventLabel: state.tutorial == null ? '' : state.tutorial
  });
  parametersChanged = false;
}

// ============================================================================
// Machine unlearning (gradient ascent on a forget set).
// ============================================================================

function unlearn(forgetSet: Example2D[], steps: number): void {
  if (forgetSet.length === 0) return;
  let optimizerType = optimizers[state.optimizer] || nn.OptimizerType.SGD;
  for (let s = 0; s < steps; s++) {
    forgetSet.forEach(point => {
      nn.forwardProp(network, constructInput(point.x, point.y),
          state.weightQuantization, state.layerNorm);
      nn.backProp(network, point.label, nn.Errors.SQUARE);
      // Gradient ASCENT: negative learning rate.
      nn.updateWeights(network, -state.learningRate, state.regularization,
          state.regularizationRate, optimizerType);
    });
  }
}

function misclassified(points: Example2D[]): Example2D[] {
  return points.filter(p => {
    let out = nn.forwardProp(network, constructInput(p.x, p.y),
        state.weightQuantization, state.layerNorm);
    return Math.sign(out) !== Math.sign(p.label);
  });
}

function doUnlearn(forgetSet: Example2D[], label: string): void {
  let retainSet = state.trainData.filter(p => forgetSet.indexOf(p) === -1);
  let accF0 = accuracy(forgetSet);
  let accR0 = accuracy(retainSet);
  let steps = +(d3.select("#unlearn-steps").property("value") || 100);
  unlearn(forgetSet, steps);
  let accF1 = accuracy(forgetSet);
  let accR1 = accuracy(retainSet);
  updateUI();
  d3.select("#unlearn-readout").html(
    `${label} (${forgetSet.length} pts, ${steps} steps)<br>` +
    `Forget acc: ${(accF0 * 100).toFixed(1)}% &rarr; ${(accF1 * 100).toFixed(1)}%<br>` +
    `Retain acc: ${(accR0 * 100).toFixed(1)}% &rarr; ${(accR1 * 100).toFixed(1)}%`);
}

// ============================================================================
// Layer freezing (fine-tuning) + model save/load.
// ============================================================================

/**
 * Applies state.frozenLayers to the freshly-built network by setting the
 * `frozen` flag on every node in the corresponding hidden layer. Hidden layer
 * index 1 maps to network layer index 1 (network[0] is the input layer).
 */
function applyFrozenLayers(): void {
  if (network == null) return;
  let numHidden = network.length - 2;
  // Drop any stale indices that no longer exist.
  state.frozenLayers = state.frozenLayers.filter(
      idx => idx >= 1 && idx <= numHidden);
  nn.forEachNode(network, true, node => { node.frozen = false; });
  state.frozenLayers.forEach(layerIdx => {
    let layer = network[layerIdx];
    if (layer) {
      layer.forEach(n => { n.frozen = true; });
    }
  });
}

function isLayerFrozen(layerIdx: number): boolean {
  return state.frozenLayers.indexOf(layerIdx) !== -1;
}

function setLayerFrozen(layerIdx: number, frozen: boolean): void {
  let pos = state.frozenLayers.indexOf(layerIdx);
  if (frozen && pos === -1) {
    state.frozenLayers.push(layerIdx);
  } else if (!frozen && pos !== -1) {
    state.frozenLayers.splice(pos, 1);
  }
}

/** (Re)builds the per-layer freeze checkbox list and refreshes node styles. */
function rebuildFreezeControls(): void {
  let numHidden = state.networkShape.length;
  let container = d3.select("#freeze-layers-list");
  container.selectAll("*").remove();
  if (numHidden === 0) {
    container.append("span").attr("class", "adv-help").text("No hidden layers.");
    return;
  }
  for (let layerIdx = 1; layerIdx <= numHidden; layerIdx++) {
    let label = container.append("label")
      .attr("class", "freeze-layer-item")
      .style("display", "inline-block")
      .style("margin-right", "10px");
    label.append("input")
      .attr("type", "checkbox")
      .property("checked", isLayerFrozen(layerIdx))
      .on("change", function() {
        setLayerFrozen(layerIdx, (this as any).checked);
        state.serialize();
        applyFrozenLayers();
        drawNetwork(network);
        updateUI();
      });
    label.append("span").text(" Layer " + layerIdx);
  }
}

function exportModel(): void {
  if (network == null) {
    d3.select("#model-io-readout").text("No network to export.");
    return;
  }
  let numInputs = constructInput(0, 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  let links: {[id: string]: number} = {};
  let biases: {[id: string]: number} = {};
  nn.forEachNode(network, true, node => {
    biases[node.id] = node.bias;
    node.inputLinks.forEach(link => { links[link.id] = link.weight; });
  });
  let model = {
    format: "nn-playground-model",
    version: 1,
    networkShape: shape,
    activationKey: getKeyFromActivation(state.activation),
    problem: state.problem === Problem.REGRESSION ? "regression" : "classification",
    inputIds: constructInputIds(),
    frozenLayers: state.frozenLayers.slice(),
    biases: biases,
    links: links
  };
  let json = JSON.stringify(model, null, 2);
  d3.select("#model-io-text").property("value", json);
  // Offer a download via a Blob URL anchor.
  let blob = new Blob([json], {type: "application/json"});
  let url = URL.createObjectURL(blob);
  let anchor = d3.select("#model-download-link");
  anchor.attr("href", url)
    .attr("download", "nn-playground-model.json")
    .style("display", "inline");
  d3.select("#model-io-readout").text(
    "Exported model (" + shape.join("-") + "). JSON is in the textarea; " +
    "use the download link to save it.");
}

function getKeyFromActivation(activation: nn.ActivationFunction): string {
  let key = getKeyFromValue(activations, activation);
  return key != null ? key : "tanh";
}

function importModel(): void {
  let text = d3.select("#model-io-text").property("value") as string;
  let model: any;
  try {
    model = JSON.parse(text);
  } catch (e) {
    d3.select("#model-io-readout").text("Import failed: invalid JSON.");
    return;
  }
  if (!model || model.format !== "nn-playground-model" ||
      !Array.isArray(model.networkShape) || model.networkShape.length < 2 ||
      typeof model.links !== "object" || typeof model.biases !== "object") {
    d3.select("#model-io-readout").text(
      "Import failed: not a valid playground model.");
    return;
  }
  let activation = activations[model.activationKey] || nn.Activations.TANH;
  let problem = model.problem === "regression" ?
      Problem.REGRESSION : Problem.CLASSIFICATION;
  let outputActivation = problem === Problem.REGRESSION ?
      nn.Activations.LINEAR : nn.Activations.TANH;
  let inputIds: string[] = Array.isArray(model.inputIds) ?
      model.inputIds : constructInputIds();
  let shape: number[] = model.networkShape.map(Number);
  let newNetwork: nn.Node[][];
  try {
    newNetwork = nn.buildNetwork(shape, activation, outputActivation,
        inputIds, false);
    // Assign saved biases and weights.
    nn.forEachNode(newNetwork, true, node => {
      if (model.biases[node.id] != null) {
        node.bias = +model.biases[node.id];
      }
      node.inputLinks.forEach(link => {
        if (model.links[link.id] != null) {
          link.weight = +model.links[link.id];
        }
      });
    });
  } catch (e) {
    d3.select("#model-io-readout").text(
      "Import failed: could not reconstruct network (" + e.message + ").");
    return;
  }
  // Update state to match the imported model.
  state.activation = activation;
  state.problem = problem;
  state.networkShape = shape.slice(1, shape.length - 1);
  state.numHiddenLayers = state.networkShape.length;
  if (Array.isArray(model.frozenLayers)) {
    state.frozenLayers = model.frozenLayers.map(Number);
  }
  // Reflect the imported input feature ids in the UI/INPUTS set.
  for (let key in INPUTS) {
    state[key] = inputIds.indexOf(key) !== -1;
  }
  network = newNetwork;
  applyFrozenLayers();
  iter = 0;
  state.serialize();
  // Sync activation dropdown if possible.
  let actKey = getKeyFromActivation(activation);
  d3.select("#activations").property("value", actKey);
  d3.select("#problem").property("value",
      problem === Problem.REGRESSION ? "regression" : "classification");
  lossTrain = getLoss(network, state.trainData);
  lossTest = getLoss(network, state.testData);
  drawNetwork(network);
  rebuildFreezeControls();
  updateUI(true);
  d3.select("#model-io-readout").text(
    "Imported model (" + shape.join("-") + "). Weights kept; iter reset to 0.");
}

function makeFineTuneGUI(): void {
  rebuildFreezeControls();

  d3.select("#freeze-all-but-last").on("click", () => {
    let numHidden = state.networkShape.length;
    state.frozenLayers = [];
    // Freeze every hidden layer except the last one (classic fine-tuning:
    // train only the head, i.e. the last hidden layer + output).
    for (let layerIdx = 1; layerIdx < numHidden; layerIdx++) {
      state.frozenLayers.push(layerIdx);
    }
    state.serialize();
    applyFrozenLayers();
    drawNetwork(network);
    rebuildFreezeControls();
    updateUI();
  });

  d3.select("#unfreeze-all").on("click", () => {
    state.frozenLayers = [];
    state.serialize();
    applyFrozenLayers();
    drawNetwork(network);
    rebuildFreezeControls();
    updateUI();
  });

  d3.select("#export-model").on("click", () => { exportModel(); });
  d3.select("#import-model").on("click", () => { importModel(); });
}

function makeAdvancedGUI() {
  // ---- 3D mode ----
  let threeDToggle = d3.select("#threeD-toggle").on("change", function() {
    state.threeD = (this as any).checked;
    state.serialize();
    if (state.threeD) {
      enterThreeD();
    } else {
      exitThreeD();
    }
  });
  threeDToggle.property("checked", state.threeD);

  let threeDDataset = d3.select("#threeD-dataset").on("change", function() {
    state.threeDDataset = (this as any).value;
    state.serialize();
    if (state.threeD) {
      reset3D();
    }
  });
  threeDDataset.property("value", state.threeDDataset);

  // ---- Adversarial ----
  let advEps = d3.select("#adv-epsilon").on("input", function() {
    state.advEpsilon = +(this as any).value;
    d3.select("#adv-epsilon-val").text((this as any).value);
    state.serialize();
  });
  advEps.property("value", state.advEpsilon);
  d3.select("#adv-epsilon-val").text(state.advEpsilon);

  let advMethod = d3.select("#adv-method").on("change", function() {
    state.advMethod = (this as any).value;
    state.serialize();
  });
  advMethod.property("value", state.advMethod);

  let advTraining = d3.select("#adv-training").on("change", function() {
    state.adversarialTraining = (this as any).checked;
    state.serialize();
  });
  advTraining.property("checked", state.adversarialTraining);

  d3.select("#adv-generate").on("click", () => {
    let clean = state.testData;
    let cleanAcc = accuracy(clean);
    let perturbed = clean.map(p => perturb(p));
    let advAcc = accuracy(perturbed);
    let flipped = 0;
    for (let i = 0; i < clean.length; i++) {
      let oc = nn.forwardProp(network, constructInput(clean[i].x, clean[i].y),
          state.weightQuantization, state.layerNorm);
      let op = nn.forwardProp(network,
          constructInput(perturbed[i].x, perturbed[i].y),
          state.weightQuantization, state.layerNorm);
      if (Math.sign(oc) !== Math.sign(op)) flipped++;
    }
    heatMap.updateTestPoints(perturbed);
    d3.select("#adv-readout").html(
      `Method: ${state.advMethod.toUpperCase()}, &epsilon;=${state.advEpsilon}<br>` +
      `Clean acc: ${(cleanAcc * 100).toFixed(1)}%<br>` +
      `Adversarial acc: ${(advAcc * 100).toFixed(1)}%<br>` +
      `Predictions flipped: ${flipped}/${clean.length}`);
  });

  // ---- Unlearning ----
  let unlearnSteps = d3.select("#unlearn-steps").on("input", function() {
    d3.select("#unlearn-steps-val").text((this as any).value);
  });
  unlearnSteps.property("value", 100);
  d3.select("#unlearn-steps-val").text(100);

  d3.select("#forget-orange").on("click", () => {
    doUnlearn(state.trainData.filter(p => p.label < 0), "Forgot Orange");
  });
  d3.select("#forget-blue").on("click", () => {
    doUnlearn(state.trainData.filter(p => p.label > 0), "Forgot Blue");
  });
  d3.select("#forget-misclassified").on("click", () => {
    doUnlearn(misclassified(state.trainData), "Forgot misclassified");
  });
  d3.select("#retrain-without").on("click", () => {
    let forget = misclassified(state.trainData);
    let retain = state.trainData.filter(p => forget.indexOf(p) === -1);
    state.trainData = retain;
    heatMap.updatePoints(state.trainData);
    reset();
    // Train the fresh network on the retained set only.
    for (let e = 0; e < 100; e++) oneStep();
    player.pause();
    d3.select("#unlearn-readout").html(
      `Retrained from scratch without ${forget.length} forgotten points ` +
      `(100 epochs).`);
  });

  // ---- Custom data ----
  d3.select("#custom-data-load").on("click", () => {
    let text = d3.select("#custom-data-text").property("value") as string;
    let result = parseCSV(text);
    if (result.examples.length === 0) {
      d3.select("#custom-data-readout").html(
        `No valid examples parsed.` +
        (result.errors.length ? `<br>${result.errors.slice(0, 5).join("<br>")}` : ""));
      return;
    }
    let split = Math.floor(result.examples.length * state.percTrainData / 100);
    state.trainData = result.examples.slice(0, split);
    state.testData = result.examples.slice(split);
    heatMap.updatePoints(state.trainData);
    heatMap.updateTestPoints(state.showTestData ? state.testData : []);
    reset();
    d3.select("#custom-data-readout").html(
      `Loaded ${result.examples.length} examples ` +
      `(${state.trainData.length} train / ${state.testData.length} test).` +
      (result.errors.length ?
        `<br>${result.errors.length} bad line(s) skipped.` : ""));
  });
}

// ============================================================================
// Analysis: evaluation & visualization. All panels live in #analysis-section
// and are refreshed (cheaply) from updateUI() / via buttons. Classification-
// only panels are hidden in regression mode.
// ============================================================================

// State accumulated for export / time-series charts.
let analysisStep = 0;
let lastStepTime = (typeof performance !== "undefined" ? performance.now() : Date.now());
let lastStepsPerSec = 0;
let trainingHistory: Array<{iter: number; lossTrain: number; lossTest: number;
    accTrain: number; accTest: number}> = [];
// Per-layer mean |weight| time series (one array per hidden layer + output).
let weightMagHistory: number[][] = [];
let lastAuc = NaN;
let lastAp = NaN;
let lastConfusion: number[][] = [[0, 0], [0, 0]];

const analysisOpts = {curves: true, hist: true, grad: true, landscape: true};

function nowMs(): number {
  return (typeof performance !== "undefined" ? performance.now() : Date.now());
}

/** Raw network outputs + labels over a dataset (classification scoring). */
function scoreDataset(net: nn.Node[][], data: Example2D[]):
    Array<{score: number; label: number}> {
  let out: Array<{score: number; label: number}> = [];
  for (let p of data) {
    let o = nn.forwardProp(net, constructInput(p.x, p.y),
        state.weightQuantization, state.layerNorm);
    out.push({score: o, label: p.label});
  }
  return out;
}

/** ROC points (sweep threshold over raw output) + AUC. Positive = label +1. */
function computeRoc(scored: Array<{score: number; label: number}>):
    {points: Array<[number, number]>; auc: number} {
  let pos = scored.filter(s => s.label > 0).length;
  let neg = scored.length - pos;
  if (pos === 0 || neg === 0) {
    return {points: [[0, 0], [1, 1]], auc: 0.5};
  }
  let sorted = scored.slice().sort((a, b) => b.score - a.score);
  let points: Array<[number, number]> = [[0, 0]];
  let tp = 0, fp = 0;
  let auc = 0;
  let prevFpr = 0, prevTpr = 0;
  for (let s of sorted) {
    if (s.label > 0) { tp++; } else { fp++; }
    let tpr = tp / pos;
    let fpr = fp / neg;
    // Trapezoidal area increment.
    auc += (fpr - prevFpr) * (tpr + prevTpr) / 2;
    points.push([fpr, tpr]);
    prevFpr = fpr; prevTpr = tpr;
  }
  return {points, auc};
}

/** Precision-Recall curve + average precision. Positive = label +1. */
function computePr(scored: Array<{score: number; label: number}>):
    {points: Array<[number, number]>; ap: number} {
  let pos = scored.filter(s => s.label > 0).length;
  if (pos === 0) { return {points: [[0, 1]], ap: 0}; }
  let sorted = scored.slice().sort((a, b) => b.score - a.score);
  let tp = 0, fp = 0;
  let points: Array<[number, number]> = [];
  let ap = 0;
  let prevRecall = 0;
  for (let s of sorted) {
    if (s.label > 0) { tp++; } else { fp++; }
    let precision = tp / (tp + fp);
    let recall = tp / pos;
    ap += (recall - prevRecall) * precision;
    points.push([recall, precision]);
    prevRecall = recall;
  }
  return {points, ap};
}

/** Precision/recall/F1/specificity at threshold 0 (predict +1 if output>=0). */
function computeThresholdMetrics(scored: Array<{score: number; label: number}>):
    {precision: number; recall: number; f1: number; specificity: number} {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let s of scored) {
    let pred = s.score >= 0 ? 1 : -1;
    if (s.label > 0) {
      if (pred > 0) { tp++; } else { fn++; }
    } else {
      if (pred > 0) { fp++; } else { tn++; }
    }
  }
  let precision = tp + fp ? tp / (tp + fp) : 0;
  let recall = tp + fn ? tp / (tp + fn) : 0;
  let f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
  let specificity = tn + fp ? tn / (tn + fp) : 0;
  return {precision, recall, f1, specificity};
}

/** Generic line/curve plot into an svg id. data = list of [x,y] in [0,1]. */
function drawCurve(svgId: string, data: Array<[number, number]>,
    color: string, diagonal: boolean): void {
  let svg = d3.select<SVGSVGElement, unknown>("#" + svgId);
  if (svg.empty()) { return; }
  let W = +svg.attr("width");
  let H = +svg.attr("height");
  let m = {t: 8, r: 8, b: 22, l: 28};
  let x = d3.scaleLinear().domain([0, 1]).range([m.l, W - m.r]);
  let y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
  svg.selectAll("*").remove();
  // Axes.
  svg.append("g").attr("class", "an-axis")
      .attr("transform", `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).ticks(4));
  svg.append("g").attr("class", "an-axis")
      .attr("transform", `translate(${m.l},0)`)
      .call(d3.axisLeft(y).ticks(4));
  if (diagonal) {
    svg.append("line")
        .attr("x1", x(0)).attr("y1", y(0))
        .attr("x2", x(1)).attr("y2", y(1))
        .attr("stroke", "#ccc").attr("stroke-dasharray", "3,3");
  }
  let line = d3.line<[number, number]>()
      .x(d => x(d[0])).y(d => y(d[1]));
  svg.append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", color)
      .attr("stroke-width", 1.5)
      .attr("d", line);
}

/** Histogram of values into an svg id. */
function drawHistogram(svgId: string, values: number[], color: string,
    domain?: [number, number]): void {
  let svg = d3.select<SVGSVGElement, unknown>("#" + svgId);
  if (svg.empty()) { return; }
  let W = +svg.attr("width");
  let H = +svg.attr("height");
  let m = {t: 8, r: 8, b: 22, l: 28};
  svg.selectAll("*").remove();
  if (!values.length) { return; }
  let dom = domain || [d3.min(values), d3.max(values)];
  if (dom[0] === dom[1]) { dom = [dom[0] - 1, dom[1] + 1]; }
  let x = d3.scaleLinear().domain(dom).range([m.l, W - m.r]);
  let bins = d3.bin().domain(dom as [number, number]).thresholds(16)(values);
  let maxCount = d3.max(bins, b => b.length) || 1;
  let y = d3.scaleLinear().domain([0, maxCount]).range([H - m.b, m.t]);
  svg.append("g").attr("class", "an-axis")
      .attr("transform", `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).ticks(4));
  svg.append("g").attr("class", "an-axis")
      .attr("transform", `translate(${m.l},0)`)
      .call(d3.axisLeft(y).ticks(3));
  svg.selectAll("rect").data(bins).enter().append("rect")
      .attr("x", d => x(d.x0) + 1)
      .attr("y", d => y(d.length))
      .attr("width", d => Math.max(0, x(d.x1) - x(d.x0) - 1))
      .attr("height", d => (H - m.b) - y(d.length))
      .attr("fill", color);
}

/** Bar chart from labeled values into an svg id. */
function drawBars(svgId: string, labels: string[], values: number[],
    color: string): void {
  let svg = d3.select<SVGSVGElement, unknown>("#" + svgId);
  if (svg.empty()) { return; }
  let W = +svg.attr("width");
  let H = +svg.attr("height");
  let m = {t: 8, r: 8, b: 22, l: 32};
  svg.selectAll("*").remove();
  let x = d3.scaleBand().domain(labels).range([m.l, W - m.r]).padding(0.2);
  let maxV = d3.max(values) || 1;
  let y = d3.scaleLinear().domain([0, maxV]).range([H - m.b, m.t]);
  svg.append("g").attr("class", "an-axis")
      .attr("transform", `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x));
  svg.append("g").attr("class", "an-axis")
      .attr("transform", `translate(${m.l},0)`)
      .call(d3.axisLeft(y).ticks(3));
  svg.selectAll("rect").data(values).enter().append("rect")
      .attr("x", (d, i) => x(labels[i]))
      .attr("y", d => y(d))
      .attr("width", x.bandwidth())
      .attr("height", d => (H - m.b) - y(d))
      .attr("fill", color);
}

/** Multi-line chart (per-layer series) into svg id. */
function drawMultiLine(svgId: string, series: number[][],
    colors: string[]): void {
  let svg = d3.select<SVGSVGElement, unknown>("#" + svgId);
  if (svg.empty()) { return; }
  let W = +svg.attr("width");
  let H = +svg.attr("height");
  let m = {t: 8, r: 8, b: 22, l: 32};
  svg.selectAll("*").remove();
  let maxLen = d3.max(series, s => s.length) || 1;
  if (maxLen < 2) { return; }
  let allVals: number[] = [];
  series.forEach(s => s.forEach(v => allVals.push(v)));
  let maxV = d3.max(allVals) || 1;
  let x = d3.scaleLinear().domain([0, maxLen - 1]).range([m.l, W - m.r]);
  let y = d3.scaleLinear().domain([0, maxV]).range([H - m.b, m.t]);
  svg.append("g").attr("class", "an-axis")
      .attr("transform", `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).ticks(4));
  svg.append("g").attr("class", "an-axis")
      .attr("transform", `translate(${m.l},0)`)
      .call(d3.axisLeft(y).ticks(3));
  let line = d3.line<number>()
      .x((d, i) => x(i)).y(d => y(d));
  series.forEach((s, idx) => {
    svg.append("path")
        .datum(s)
        .attr("fill", "none")
        .attr("stroke", colors[idx % colors.length])
        .attr("stroke-width", 1.3)
        .attr("d", line);
  });
}

/** Reliability/calibration diagram. */
function drawCalibration(svgId: string,
    scored: Array<{score: number; label: number}>): void {
  let nBins = 10;
  let binSum = new Array(nBins).fill(0);
  let binPos = new Array(nBins).fill(0);
  let binCnt = new Array(nBins).fill(0);
  for (let s of scored) {
    let p = (s.score + 1) / 2;  // map [-1,1] -> [0,1]
    p = Math.max(0, Math.min(1, p));
    let b = Math.min(nBins - 1, Math.floor(p * nBins));
    binSum[b] += p;
    binCnt[b]++;
    if (s.label > 0) { binPos[b]++; }
  }
  let points: Array<[number, number]> = [];
  for (let b = 0; b < nBins; b++) {
    if (binCnt[b] > 0) {
      points.push([binSum[b] / binCnt[b], binPos[b] / binCnt[b]]);
    }
  }
  drawCurve(svgId, points, "#9b59b6", true);
}

/** Count total trainable parameters (weights + biases). */
function countParameters(net: nn.Node[][]): number {
  let count = 0;
  for (let layerIdx = 1; layerIdx < net.length; layerIdx++) {
    for (let node of net[layerIdx]) {
      count++;  // bias
      count += node.inputLinks.length;  // weights
    }
  }
  return count;
}

/** Collect all link weights / node biases. */
function collectWeights(net: nn.Node[][]): number[] {
  let w: number[] = [];
  for (let layerIdx = 1; layerIdx < net.length; layerIdx++) {
    for (let node of net[layerIdx]) {
      for (let link of node.inputLinks) { w.push(link.weight); }
    }
  }
  return w;
}
function collectBiases(net: nn.Node[][]): number[] {
  let b: number[] = [];
  nn.forEachNode(net, true, node => b.push(node.bias));
  return b;
}

/** Hidden-node activations over the test set. */
function collectActivations(net: nn.Node[][], data: Example2D[]): number[] {
  let acts: number[] = [];
  let sample = data.slice(0, 200);
  for (let p of sample) {
    nn.forwardProp(net, constructInput(p.x, p.y),
        state.weightQuantization, state.layerNorm);
    for (let layerIdx = 1; layerIdx < net.length - 1; layerIdx++) {
      for (let node of net[layerIdx]) { acts.push(node.output); }
    }
  }
  return acts;
}

/** Mean |weight| per hidden+output layer. */
function meanWeightMagPerLayer(net: nn.Node[][]): number[] {
  let result: number[] = [];
  for (let layerIdx = 1; layerIdx < net.length; layerIdx++) {
    let sum = 0, cnt = 0;
    for (let node of net[layerIdx]) {
      for (let link of node.inputLinks) { sum += Math.abs(link.weight); cnt++; }
    }
    result.push(cnt ? sum / cnt : 0);
  }
  return result;
}

/** Run a backprop pass on a sample to populate gradients; return mean |grad|
 * per hidden layer (using link.errorDer of incoming links). */
function gradientFlowPerLayer(net: nn.Node[][], data: Example2D[]): number[] {
  let sample = data.slice(0, 50);
  // Reset accumulators by recomputing errorDer per example then averaging.
  let sums: number[] = new Array(net.length).fill(0);
  let counts: number[] = new Array(net.length).fill(0);
  for (let p of sample) {
    nn.forwardProp(net, constructInput(p.x, p.y),
        state.weightQuantization, state.layerNorm);
    nn.backProp(net, p.label, nn.Errors.SQUARE);
    for (let layerIdx = 1; layerIdx < net.length; layerIdx++) {
      for (let node of net[layerIdx]) {
        for (let link of node.inputLinks) {
          sums[layerIdx] += Math.abs(link.errorDer);
          counts[layerIdx]++;
        }
      }
    }
  }
  let out: number[] = [];
  for (let layerIdx = 1; layerIdx < net.length; layerIdx++) {
    out.push(counts[layerIdx] ? sums[layerIdx] / counts[layerIdx] : 0);
  }
  return out;
}

const LAYER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e"];

function updateAnalysis(): void {
  if (!network) { return; }
  let isClass = state.problem === Problem.CLASSIFICATION && !state.threeD;

  // Show/hide classification-only panels & readouts.
  d3.selectAll(".classification-only").style("display", isClass ? null : "none");

  // --- Cheap readouts (every step) ---
  d3.select("#an-params").text(countParameters(network).toString());

  let t = nowMs();
  let dt = t - lastStepTime;
  if (dt > 0) {
    let sps = 1000 / dt;
    lastStepsPerSec = lastStepsPerSec ? lastStepsPerSec * 0.7 + sps * 0.3 : sps;
  }
  lastStepTime = t;
  d3.select("#an-sps").text(lastStepsPerSec.toFixed(1));

  // Decision margin (mean |out|) over test set.
  let testScored = scoreDataset(network, state.testData);
  let margin = testScored.length ?
      d3.mean(testScored, s => Math.abs(s.score)) : 0;
  d3.select("#an-margin").text((margin || 0).toFixed(3));

  if (isClass) {
    // Class balance.
    let trPos = state.trainData.filter(p => p.label > 0).length;
    let trNeg = state.trainData.length - trPos;
    let tePos = state.testData.filter(p => p.label > 0).length;
    let teNeg = state.testData.length - tePos;
    d3.select("#an-balance").text(
        `tr ${trPos}/${trNeg} · te ${tePos}/${teNeg}`);

    let thr = computeThresholdMetrics(testScored);
    d3.select("#an-prec-rec").text(
        `${(thr.precision * 100).toFixed(1)}% / ${(thr.recall * 100).toFixed(1)}%`);
    d3.select("#an-f1-spec").text(
        `${thr.f1.toFixed(3)} / ${(thr.specificity * 100).toFixed(1)}%`);

    let roc = computeRoc(testScored);
    let pr = computePr(testScored);
    lastAuc = roc.auc;
    lastAp = pr.ap;
    lastConfusion = computeClassMetrics(network, state.testData).matrix;
    d3.select("#an-auc").text(roc.auc.toFixed(3));
    d3.select("#an-ap").text(pr.ap.toFixed(3));

    // --- Curves (every K steps, if enabled) ---
    if (analysisOpts.curves && analysisStep % 5 === 0) {
      drawCurve("an-roc", roc.points, "#e74c3c", true);
      drawCurve("an-pr", pr.points, "#3498db", false);
      drawCalibration("an-calib", testScored);
    }
  } else {
    lastAuc = NaN; lastAp = NaN;
  }

  // --- Histograms (every step is fine but throttle heavy ones) ---
  if (analysisOpts.hist) {
    drawHistogram("an-whist", collectWeights(network), "#3498db");
    drawHistogram("an-bhist", collectBiases(network), "#e67e22");
    drawHistogram("an-chist", testScored.map(s => Math.abs(s.score)),
        "#2ecc71", [0, 1]);
    if (analysisStep % 5 === 0) {
      drawHistogram("an-ahist", collectActivations(network, state.testData),
          "#9b59b6");
    }
  }

  // --- Gradient flow & per-layer weight magnitude (every K steps) ---
  if (analysisOpts.grad && analysisStep % 5 === 0) {
    let gf = gradientFlowPerLayer(network, state.trainData);
    let labels = gf.map((_, i) => "L" + (i + 1));
    drawBars("an-grad", labels, gf, "#16a085");

    let mags = meanWeightMagPerLayer(network);
    if (weightMagHistory.length !== mags.length) {
      weightMagHistory = mags.map(() => []);
    }
    mags.forEach((mg, i) => {
      weightMagHistory[i].push(mg);
      if (weightMagHistory[i].length > 200) { weightMagHistory[i].shift(); }
    });
    drawMultiLine("an-wmag", weightMagHistory, LAYER_COLORS);
  }

  // --- Training history accumulation (for CSV export) ---
  let accTrain = isClass ?
      computeClassMetrics(network, state.trainData).accuracy : NaN;
  let accTest = isClass ?
      computeClassMetrics(network, state.testData).accuracy : NaN;
  trainingHistory.push({iter, lossTrain, lossTest, accTrain, accTest});
  if (trainingHistory.length > 5000) { trainingHistory.shift(); }

  analysisStep++;
}

/** Loss landscape 1D slice: perturb weights along a random direction. */
function computeLossLandscape(): void {
  if (!network) { return; }
  // Save weights & biases.
  let savedW: number[] = [];
  let dirW: number[] = [];
  let linkRefs: nn.Link[] = [];
  let savedB: number[] = [];
  let dirB: number[] = [];
  let nodeRefs: nn.Node[] = [];
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    for (let node of network[layerIdx]) {
      nodeRefs.push(node);
      savedB.push(node.bias);
      dirB.push(Math.random() - 0.5);
      for (let link of node.inputLinks) {
        linkRefs.push(link);
        savedW.push(link.weight);
        dirW.push(Math.random() - 0.5);
      }
    }
  }
  // Normalize direction.
  let norm = Math.sqrt(
      dirW.reduce((a, b) => a + b * b, 0) + dirB.reduce((a, b) => a + b * b, 0));
  if (norm === 0) { norm = 1; }
  dirW = dirW.map(v => v / norm);
  dirB = dirB.map(v => v / norm);

  let r = 1.0;
  let steps = 41;
  let points: Array<[number, number]> = [];
  let minLoss = Infinity, maxLoss = -Infinity;
  for (let k = 0; k < steps; k++) {
    let alpha = -r + (2 * r) * k / (steps - 1);
    for (let i = 0; i < linkRefs.length; i++) {
      linkRefs[i].weight = savedW[i] + alpha * dirW[i];
    }
    for (let i = 0; i < nodeRefs.length; i++) {
      nodeRefs[i].bias = savedB[i] + alpha * dirB[i];
    }
    let loss = getLoss(network, state.trainData);
    minLoss = Math.min(minLoss, loss);
    maxLoss = Math.max(maxLoss, loss);
    points.push([(alpha + r) / (2 * r), loss]);
  }
  // Restore.
  for (let i = 0; i < linkRefs.length; i++) { linkRefs[i].weight = savedW[i]; }
  for (let i = 0; i < nodeRefs.length; i++) { nodeRefs[i].bias = savedB[i]; }

  // Normalize y for the [0,1] drawCurve helper.
  let range = maxLoss - minLoss || 1;
  let norm2: Array<[number, number]> =
      points.map(p => [p[0], (p[1] - minLoss) / range]);
  drawCurve("an-landscape", norm2, "#c0392b", false);
}

function downloadText(filename: string, text: string, mime: string): void {
  let blob = new Blob([text], {type: mime});
  let url = URL.createObjectURL(blob);
  let a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportHistoryCsv(): void {
  let rows = ["iter,lossTrain,lossTest,accTrain,accTest"];
  for (let h of trainingHistory) {
    rows.push(`${h.iter},${h.lossTrain},${h.lossTest},` +
        `${isNaN(h.accTrain) ? "" : h.accTrain},` +
        `${isNaN(h.accTest) ? "" : h.accTest}`);
  }
  downloadText("training-history.csv", rows.join("\n"), "text/csv");
}

function exportMetricsJson(): void {
  let snapshot = {
    iter,
    lossTrain,
    lossTest,
    auc: isNaN(lastAuc) ? null : lastAuc,
    averagePrecision: isNaN(lastAp) ? null : lastAp,
    confusionMatrix: lastConfusion,
    parameters: network ? countParameters(network) : 0,
    problem: getKeyFromValue(problems, state.problem)
  };
  downloadText("metrics-snapshot.json", JSON.stringify(snapshot, null, 2),
      "application/json");
}

function exportBoundaryPng(): void {
  let canvasEl = document.querySelector("#heatmap canvas") as HTMLCanvasElement;
  if (!canvasEl) { return; }
  let dataUrl = canvasEl.toDataURL("image/png");
  let a = document.createElement("a");
  a.href = dataUrl;
  a.download = "decision-boundary.png";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function makeAnalysisGUI(): void {
  // Collapsible toggle.
  d3.select("#analysis-toggle").on("click", () => {
    let sec = document.getElementById("analysis-section");
    if (sec) { sec.classList.toggle("collapsed"); }
  });

  // Panel visibility checkboxes.
  let bind = (id: string, key: keyof typeof analysisOpts,
      panelSelector: string) => {
    let cb = document.getElementById(id) as HTMLInputElement;
    if (!cb) { return; }
    let apply = () => {
      analysisOpts[key] = cb.checked;
      d3.selectAll(panelSelector).style("display", cb.checked ? null : "none");
      // Re-apply classification gating.
      if (state.problem !== Problem.CLASSIFICATION || state.threeD) {
        d3.selectAll(".classification-only").style("display", "none");
      }
    };
    cb.addEventListener("change", apply);
  };
  bind("ao-curves", "curves", "#an-panel-roc,#an-panel-pr,#an-panel-calib");
  bind("ao-hist", "hist",
      "#an-panel-whist,#an-panel-bhist,#an-panel-ahist,#an-panel-chist");
  bind("ao-grad", "grad", "#an-panel-grad,#an-panel-wmag");
  bind("ao-landscape", "landscape", "#an-panel-landscape");

  d3.select("#an-export-png").on("click", () => exportBoundaryPng());
  d3.select("#an-export-csv").on("click", () => exportHistoryCsv());
  d3.select("#an-export-json").on("click", () => exportMetricsJson());
  d3.select("#an-landscape-btn").on("click", () => computeLossLandscape());
}

drawDatasetThumbnails();
initTutorial();
makeGUI();
makeAdvancedGUI();
makeFineTuneGUI();
makeAnalysisGUI();
generateData(true);
reset(true);
hideControls();


function makeid(length) {
    let result = '';
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    const charactersLength = characters.length;
    let counter = 0;
    while (counter < length) {
      result += characters.charAt(Math.floor(Math.random() * charactersLength));
      counter += 1;
    }
    return result;
}

document.querySelector("#addinput").addEventListener("click", () => {
  const form = prompt("enter formula:")
  if(!form) return;
  INPUTS[makeid(8)] = {f: compile(form), label: form}
  reset()
})

document.querySelector("#add-activation").addEventListener("click", () => {
  const formula = prompt("Enter activation formula in terms of x (e.g. tanh(x)*x):");
  if (!formula) return;
  let compiled: any;
  try {
    compiled = compile(formula);
  } catch (e) {
    alert("Could not compile formula: " + e.message);
    return;
  }
  // Build activation using compiled mathjs expression; derivative via finite differences.
  const h = 1e-4;
  const customActivation: nn.ActivationFunction = {
    output: (x: number) => compiled.evaluate({x}),
    der: (x: number) => (compiled.evaluate({x: x + h}) - compiled.evaluate({x: x - h})) / (2 * h),
    compileToJs: (arg: string) => `/* custom: ${formula} */ (function(x){return ${formula};})(${arg})`
  };
  // Register in global activations map.
  const key = "custom_" + makeid(4);
  activations[key] = customActivation;
  // Add option to the activations dropdown.
  const sel = document.querySelector("#activations") as HTMLSelectElement;
  const opt = document.createElement("option");
  opt.value = key;
  opt.text = formula;
  sel.appendChild(opt);
  // Select the new activation.
  state.activation = customActivation;
  sel.value = key;
  parametersChanged = true;
  reset();
})
