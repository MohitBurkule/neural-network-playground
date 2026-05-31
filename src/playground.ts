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
  lossFunctions,
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
  classifySCurve3D,
  classifyTrefoilKnot,
  classifyMobiusBand,
  classifyStackedPlanes,
  classifySpiralTower,
  classifyOctantChecker,
  classifySphereGrid,
  classifyNestedCubes,
  classifyCubeShell,
  classifyDoubleTorus,
  classifySpiralCone,
  classifyLatticePoints,
  classify3DMoons,
  classifyOctahedronVsSphere,
  classifyHelixPair,
  classifyPlaneStack5,
  classifySphereSpiral,
  classifyCubeLattice,
  classifyTwoHelices,
  classifyConeStack,
  classifyTorusKnot2,
  classifyPlaneVsBlob,
  classifyOctantSpheres,
  classifySwissRoll3Class,
  classifySphericalShell2,
  classifyGridXor3D,
  classifyHelixTriple,
  classifyCubeEdges,
  classifySphereClusters,
  classifyTwistedTorus,
  classifyChecker3D8,
  classifyParaboloidShell
} from "./dataset3d";
import {parseCSV} from "./customdataset";
import {
  randomNoiseAttack, targetedFgsm, deepFoolLite, perturbationBudget,
  attackSuccessRate, robustnessCurve, GradFn
} from "./adversarial";
import {forgetQualityScore} from "./unlearning";
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

  isActive(): boolean {
    return this.isPlaying;
  }

  private start(localTimerIndex: number) {
    const timer = d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        timer.stop();  // Done.
        return;
      }
      // Run a configurable number of steps per animation tick so training can
      // proceed faster. Defaults to 1 to preserve the original behavior.
      let steps = Math.max(1, uxStepsPerTick | 0);
      // "Fast training" scheduler (feature 8): run a larger batch of steps per
      // frame off the main render path. Combined with throttled redraw this
      // keeps the UI responsive at high speed without a real Web Worker
      // (the nn library is not worker-portable here).
      if (uxFastTraining) {
        steps = Math.max(steps, steps * 8);
      }
      for (let s = 0; s < steps; s++) {
        oneStep();
        if (!this.isPlaying) {
          break;  // A hook (NaN / convergence / run-N) paused us mid-batch.
        }
      }
    });
  }
}

// ===== UX features: shared module-level state (see initUXFeatures) =====
/** Number of training steps executed per animation tick (speed control). */
let uxStepsPerTick = 1;
/** Redraw the heavy SVG/heatmap only every K training steps (feature 10).
 *  Default 1 preserves the original behavior (redraw every step). */
let uxRedrawEvery = 1;
/** Performance: when true, the player batches many steps per frame and only
 *  repaints periodically (the "fast training" worker-style scheduler). */
let uxFastTraining = false;
/** Hook invoked after each oneStep() to power convergence / run-N / NaN logic. */
let uxAfterStep: (() => void) | null = null;
/** Early-stopping monitor state (training-methodology feature 2). */
let tmEarlyStopBest = Infinity;
let tmEarlyStopWait = 0;

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
// ML-security feature state.
let lastClickedPoint: Example2D = null;
let selectedForget: Example2D[] = [];
let brushSelectActive = false;
let lastAdvClean: Example2D[] = [];
let lastAdvPerturbed: Example2D[] = [];
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
      lastClickedPoint = {x, y, label};
      heatMap.updatePoints(state.trainData);
    }
  });

  d3.select("#heatmap").call(dragBehavior);

  // Record the last clicked location on the heatmap (for saliency & nearest-N).
  d3.select("#heatmap").on("click", function(event) {
    let [px, py] = d3.pointer(event, this);
    let padding = 20, maxScale = 5.0, factor = 23.07;
    let x = (px - padding) / factor - maxScale;
    let y = maxScale - (py - padding) / factor;
    lastClickedPoint = {x, y, label: state.editColor || 1};
  });

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

/** Returns the currently selected error/loss function (defaults to SQUARE). */
function currentErrorFunc(): nn.ErrorFunction {
  return lossFunctions[state.lossFunction] || nn.Errors.SQUARE;
}

/**
 * Computes inverse-frequency class weights for the +1 / -1 classes in a
 * dataset. Returns a function mapping a label to its weight (mean weight 1).
 * When class weighting is disabled, every weight is 1.
 */
function classWeightFor(dataPoints: Example2D[]): (label: number) => number {
  if (!state.classWeighting || dataPoints.length === 0) {
    return () => 1;
  }
  let pos = 0, neg = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    if (dataPoints[i].label >= 0) { pos++; } else { neg++; }
  }
  if (pos === 0 || neg === 0) { return () => 1; }
  let n = dataPoints.length;
  // Inverse frequency, normalized so the average weight is ~1.
  let wPos = n / (2 * pos);
  let wNeg = n / (2 * neg);
  return (label: number) => label >= 0 ? wPos : wNeg;
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let errFunc = currentErrorFunc();
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let output = nn.forwardProp(network, input, state.weightQuantization, state.layerNorm);
    loss += errFunc.error(output, dataPoint.label);
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
  updateTrainingConfigSummary();
  interpAfterUpdate();
  // Experiments: record downsampled per-epoch history + refresh live readouts.
  if (typeof xpRecordLiveHistory === "function") {
    xpRecordLiveHistory();
    xpRenderBadgeAndDelta();
  }

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

/** Predict scalar output for raw (x, y) via the playground feature pipeline. */
function predictXY(x: number, y: number): number {
  return nn.forwardProp(network, constructInput(x, y),
      state.weightQuantization, state.layerNorm);
}

/** Feature-agnostic loss gradient wrt raw (x, y), for module attack helpers. */
const advGrad: GradFn = (x, y, label) => rawInputGradient(x, y, label);

/** Perturb a point using the currently selected method/epsilon. */
function perturb(point: Example2D, epsOverride?: number): Example2D {
  let eps = epsOverride != null ? epsOverride : state.advEpsilon;
  switch (state.advMethod) {
    case "pgd":
      return pgd(point, eps, 10, eps / 4);
    case "random":
      return randomNoiseAttack(point, eps);
    case "targeted":
      return targetedFgsm(advGrad, point, eps, 10, eps / 4);
    case "deepfool": {
      let r = deepFoolLite(predictXY, advGrad, point, Math.max(eps / 4, 0.05),
          50);
      return {x: r.x, y: r.y, label: r.label};
    }
    default:
      return fgsm(point, eps);
  }
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
  "s-curve-3d": classifySCurve3D,
  "trefoil-knot": classifyTrefoilKnot,
  "mobius-band": classifyMobiusBand,
  "stacked-planes": classifyStackedPlanes,
  "spiral-tower": classifySpiralTower,
  "octant-checker": classifyOctantChecker,
  "sphere-grid": classifySphereGrid,
  "nested-cubes": classifyNestedCubes,
  "cube-shell": classifyCubeShell,
  "double-torus": classifyDoubleTorus,
  "spiral-cone": classifySpiralCone,
  "lattice-points": classifyLatticePoints,
  "3d-moons": classify3DMoons,
  "octahedron-vs-sphere": classifyOctahedronVsSphere,
  "helix-pair": classifyHelixPair,
  "plane-stack-5": classifyPlaneStack5,
  "sphere-spiral": classifySphereSpiral,
  "cube-lattice": classifyCubeLattice,
  "two-helices": classifyTwoHelices,
  "cone-stack": classifyConeStack,
  "torus-knot-2": classifyTorusKnot2,
  "plane-vs-blob": classifyPlaneVsBlob,
  "octant-spheres": classifyOctantSpheres,
  "swiss-roll-3class": classifySwissRoll3Class,
  "spherical-shell-2": classifySphericalShell2,
  "3d-grid-xor": classifyGridXor3D,
  "helix-triple": classifyHelixTriple,
  "cube-edges": classifyCubeEdges,
  "sphere-clusters": classifySphereClusters,
  "twisted-torus": classifyTwistedTorus,
  "3d-checker-8": classifyChecker3D8,
  "paraboloid-shell": classifyParaboloidShell
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
  if (!container) { return; }
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

/** Standard normal sample via Box-Muller. */
function gaussianNoise(): number {
  let u = 0, v = 0;
  while (u === 0) { u = Math.random(); }
  while (v === 0) { v = Math.random(); }
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Returns an error function whose derivative is scaled by a constant (used for
 * class weighting and mixup mixing coefficients). error() is unscaled.
 */
function scaledErrorFunc(base: nn.ErrorFunction, scale: number): nn.ErrorFunction {
  if (scale === 1) { return base; }
  return {
    error: (o, t) => base.error(o, t),
    der: (o, t) => scale * base.der(o, t)
  };
}

/**
 * Adds annealed Gaussian noise to every accumulated gradient in the network.
 * Variance decays with iteration count (a known regularizer; Neelakantan 2015).
 */
function addGradientNoise(net: nn.Node[][], eta: number, t: number): void {
  if (eta <= 0) { return; }
  let std = Math.sqrt(eta / Math.pow(1 + t, 0.55));
  nn.forEachNode(net, true, (node) => {
    if (node.numAccumulatedDers > 0) {
      node.accInputDer += gaussianNoise() * std * node.numAccumulatedDers;
    }
    for (let j = 0; j < node.inputLinks.length; j++) {
      let link = node.inputLinks[j];
      if (link.numAccumulatedDers > 0) {
        link.accErrorDer += gaussianNoise() * std * link.numAccumulatedDers;
      }
    }
  });
}

function oneStep(): void {
  iter++;
  if (state.threeD) {
    oneStep3D();
    return;
  }
  let optimizerType = optimizers[state.optimizer] || nn.OptimizerType.SGD;
  let lr = effectiveLearningRate();
  let errFunc = currentErrorFunc();
  let weightOf = classWeightFor(state.trainData);
  // Epoch-wise shuffling of training data (off by default to preserve order).
  let order = state.trainData;
  if (state.epochShuffle) {
    order = state.trainData.slice();
    shuffle(order);
  }
  order.forEach((point, i) => {
    // Input jitter (data augmentation): perturb raw inputs during training.
    let px = point.x, py = point.y;
    if (state.inputJitter > 0) {
      px += gaussianNoise() * state.inputJitter;
      py += gaussianNoise() * state.inputJitter;
    }
    let w = weightOf(point.label);
    let input = constructInput(px, py);
    nn.forwardProp(network, input, state.weightQuantization, state.layerNorm,
        state.dropout, true, state.batchNorm);
    nn.backProp(network, point.label, scaledErrorFunc(errFunc, w));
    // Mixup-style augmentation: occasionally train on a convex combination of
    // this point and another random training point.
    if (state.mixup && state.trainData.length > 1 && Math.random() < 0.5) {
      let other = state.trainData[Math.floor(Math.random() * state.trainData.length)];
      let lam = Math.random();
      let mx = lam * point.x + (1 - lam) * other.x;
      let my = lam * point.y + (1 - lam) * other.y;
      let mTarget = lam * point.label + (1 - lam) * other.label;
      nn.forwardProp(network, constructInput(mx, my),
          state.weightQuantization, state.layerNorm,
          state.dropout, true, state.batchNorm);
      nn.backProp(network, mTarget, errFunc);
    }
    if (state.adversarialTraining && state.problem === Problem.CLASSIFICATION) {
      // Train also on an on-the-fly adversarial perturbation of this point.
      let adv = perturb(point);
      nn.forwardProp(network, constructInput(adv.x, adv.y),
          state.weightQuantization, state.layerNorm,
          state.dropout, true, state.batchNorm);
      nn.backProp(network, point.label, errFunc);
    }
    if ((i + 1) % state.batchSize === 0) {
      addGradientNoise(network, state.gradientNoise, iter);
      nn.updateWeights(network, lr, state.regularization,
          state.regularizationRate, optimizerType, state.gradClip,
          state.weightDecay);
    }
  });
  // Compute the loss.
  lossTrain = getLoss(network, state.trainData);
  lossTest = getLoss(network, state.testData);
  // Lazy/throttled redraw (feature 10): at high speeds only repaint the heavy
  // SVG/heatmap every K steps. Default K=1 preserves original behavior.
  let k = Math.max(1, uxRedrawEvery | 0);
  if (k <= 1 || iter % k === 0 || !player.isActive()) {
    updateUI();
  } else {
    updateLightUI();
  }
  uxFrameAccountStep();
  if (uxAfterStep) {
    uxAfterStep();
  }
}

/** Lightweight per-step UI refresh used when heavy redraw is throttled.
 *  Updates only cheap text readouts; the network/heatmap are left untouched
 *  until the next full updateUI(). */
function updateLightUI(): void {
  d3.select("#loss-train").text(lossTrain.toFixed(3));
  d3.select("#loss-test").text(lossTest.toFixed(3));
  let pad = "000000";
  d3.select("#iter-number").text((pad + iter).slice(-pad.length)
      .replace(/\B(?=(\d{3})+(?!\d))/g, ","));
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
  // Clear early-stopping state/badge on every reset.
  tmEarlyStopBest = Infinity;
  tmEarlyStopWait = 0;
  let esBadge = (typeof document !== "undefined") ?
      document.getElementById("tm-earlystop-badge") : null;
  if (esBadge) { esBadge.style.display = "none"; }
  lineChart.reset();
  trainingHistory = [];
  weightMagHistory = [];
  xpResetLiveHistory();
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
  // Label-noise injection: randomly flip a fraction of TRAIN labels only.
  if (state.labelNoise > 0 && state.problem === Problem.CLASSIFICATION) {
    let frac = state.labelNoise / 100;
    state.trainData.forEach((p) => {
      if (Math.random() < frac) {
        p.label = -p.label;
      }
    });
  }
  heatMap.updatePoints(state.trainData);
  heatMap.updateTestPoints(state.showTestData ? state.testData : []);
}

/**
 * Re-shuffles the existing data into a fresh train/test split without
 * regenerating the underlying points (keeps the same sample distribution).
 */
function reshuffleSplit(): void {
  let all = state.trainData.concat(state.testData);
  if (all.length === 0) { return; }
  shuffle(all);
  let splitIndex = Math.floor(all.length * state.percTrainData / 100);
  state.trainData = all.slice(0, splitIndex);
  state.testData = all.slice(splitIndex);
  heatMap.updatePoints(state.trainData);
  heatMap.updateTestPoints(state.showTestData ? state.testData : []);
  reset();
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

/** Snapshot the discretized predicted class for every heatmap grid cell. */
function snapshotBoundary(): number[] {
  let xScale = d3.scaleLinear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scaleLinear().domain([DENSITY - 1, 0]).range(xDomain);
  let grid: number[] = [];
  for (let i = 0; i < DENSITY; i++) {
    for (let j = 0; j < DENSITY; j++) {
      grid.push(Math.sign(predictXY(xScale(i), yScale(j))) || 1);
    }
  }
  return grid;
}

/** Fraction of grid cells whose predicted class flipped between two snapshots. */
function boundaryChangePct(before: number[], after: number[]): number {
  let flipped = 0;
  for (let i = 0; i < before.length; i++) {
    if (before[i] !== after[i]) flipped++;
  }
  return before.length ? flipped / before.length : 0;
}

/** Fine-tune-on-retain unlearning using the playground feature pipeline. */
function fineTuneRetain(retainSet: Example2D[], steps: number): void {
  if (retainSet.length === 0) return;
  let optimizerType = optimizers[state.optimizer] || nn.OptimizerType.SGD;
  for (let s = 0; s < steps; s++) {
    retainSet.forEach(point => {
      nn.forwardProp(network, constructInput(point.x, point.y),
          state.weightQuantization, state.layerNorm);
      nn.backProp(network, point.label, nn.Errors.SQUARE);
      nn.updateWeights(network, state.learningRate, state.regularization,
          state.regularizationRate, optimizerType);
    });
  }
}

function meanLoss(points: Example2D[]): number {
  if (points.length === 0) return 0;
  let total = 0;
  for (let p of points) {
    let out = predictXY(p.x, p.y);
    total += nn.Errors.SQUARE.error(out, p.label);
  }
  return total / points.length;
}

/** Sample up to k items uniformly without bias toward order. */
function sampleSubset(arr: Example2D[], k: number): Example2D[] {
  if (arr.length <= k) return arr;
  let copy = arr.slice();
  let out: Example2D[] = [];
  for (let i = 0; i < k; i++) {
    out.push(copy.splice(Math.floor(Math.random() * copy.length), 1)[0]);
  }
  return out;
}

/** Membership-inference proxy: loss gap forget vs random retain subset. */
function miaGap(forgetSet: Example2D[], retainSet: Example2D[]):
    {forgetLoss: number; retainLoss: number; gap: number} {
  let subset = sampleSubset(retainSet, Math.max(1, forgetSet.length));
  let fl = meanLoss(forgetSet);
  let rl = meanLoss(subset);
  return {forgetLoss: fl, retainLoss: rl, gap: rl - fl};
}

function doUnlearn(forgetSet: Example2D[], label: string): void {
  let retainSet = state.trainData.filter(p => forgetSet.indexOf(p) === -1);
  let method = (d3.select("#unlearn-method").property("value") as string)
      || "ascent";
  let accF0 = accuracy(forgetSet);
  let accR0 = accuracy(retainSet);
  let mia0 = miaGap(forgetSet, retainSet);
  let boundaryBefore = snapshotBoundary();
  let steps = +(d3.select("#unlearn-steps").property("value") || 100);
  if (method === "finetune") {
    fineTuneRetain(retainSet, steps);
  } else {
    unlearn(forgetSet, steps);
  }
  let accF1 = accuracy(forgetSet);
  let accR1 = accuracy(retainSet);
  let mia1 = miaGap(forgetSet, retainSet);
  let quality = forgetQualityScore(
    {forgetLoss: 0, forgetAccuracy: accF0, retainLoss: 0, retainAccuracy: accR0},
    {forgetLoss: 0, forgetAccuracy: accF1, retainLoss: 0, retainAccuracy: accR1});
  updateUI();
  let boundaryAfter = snapshotBoundary();
  let changePct = boundaryChangePct(boundaryBefore, boundaryAfter);
  d3.select("#unlearn-readout").html(
    `${label} (${forgetSet.length} pts, ${steps} steps, ${method})<br>` +
    `Forget acc: ${(accF0 * 100).toFixed(1)}% &rarr; ${(accF1 * 100).toFixed(1)}%<br>` +
    `Retain acc: ${(accR0 * 100).toFixed(1)}% &rarr; ${(accR1 * 100).toFixed(1)}%<br>` +
    `MIA loss gap: ${mia0.gap.toFixed(4)} &rarr; ${mia1.gap.toFixed(4)}<br>` +
    `Forget-quality score: ${quality}/100<br>` +
    `Boundary changed: ${(changePct * 100).toFixed(1)}% of grid cells`);
}

function dist2(a: Example2D, b: Example2D): number {
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

/** Non-destructive relearn-time probe (playground feature pipeline). */
function relearnProbe(forgetSet: Example2D[], maxSteps: number,
    target: number): {steps: number; accuracy: number; reached: boolean} {
  // Snapshot weights & biases.
  let weights: number[][] = [];
  let biases: number[] = [];
  for (let l = 1; l < network.length; l++) {
    for (let node of network[l]) {
      biases.push(node.bias);
      weights.push(node.inputLinks.map(lk => lk.weight));
    }
  }
  let optimizerType = optimizers[state.optimizer] || nn.OptimizerType.SGD;
  let steps = maxSteps;
  let acc = 0;
  for (let s = 1; s <= maxSteps; s++) {
    forgetSet.forEach(point => {
      nn.forwardProp(network, constructInput(point.x, point.y),
          state.weightQuantization, state.layerNorm);
      nn.backProp(network, point.label, nn.Errors.SQUARE);
      nn.updateWeights(network, state.learningRate, state.regularization,
          state.regularizationRate, optimizerType);
    });
    acc = accuracy(forgetSet);
    if (acc >= target) { steps = s; break; }
  }
  // Restore.
  let bi = 0, wi = 0;
  for (let l = 1; l < network.length; l++) {
    for (let node of network[l]) {
      node.bias = biases[bi++];
      let ws = weights[wi++];
      node.inputLinks.forEach((lk, i) => { lk.weight = ws[i]; });
    }
  }
  updateUI();
  return {steps, accuracy: acc, reached: acc >= target};
}

// ----------------------------------------------------------------------------
// Heatmap overlay helpers (saliency arrow, brush selection, robustness plot).
// ----------------------------------------------------------------------------

const HM_PADDING = 20;
const HM_FACTOR = 23.07;
const HM_MAXSCALE = 5.0;

function dataToPx(x: number, y: number): [number, number] {
  return [(x + HM_MAXSCALE) * HM_FACTOR + HM_PADDING,
          (HM_MAXSCALE - y) * HM_FACTOR + HM_PADDING];
}
function pxToData(px: number, py: number): [number, number] {
  return [(px - HM_PADDING) / HM_FACTOR - HM_MAXSCALE,
          HM_MAXSCALE - (py - HM_PADDING) / HM_FACTOR];
}

/** Return (and lazily create) an SVG overlay group on the heatmap container. */
function heatmapOverlay() {
  let svg = d3.select("#heatmap").select("svg");
  if (svg.empty()) {
    svg = d3.select("#heatmap").append("svg")
      .attr("class", "ml-sec-overlay")
      .style("position", "absolute")
      .style("left", "0").style("top", "0")
      .style("pointer-events", "none")
      .attr("width", 340).attr("height", 340);
  }
  let g = svg.select("g.ml-sec");
  if (g.empty()) {
    g = svg.append("g").attr("class", "ml-sec");
  }
  return g;
}

function drawSaliencyArrow(p: Example2D, gx: number, gy: number): void {
  let g = heatmapOverlay();
  g.selectAll("*").remove();
  let [x0, y0] = dataToPx(p.x, p.y);
  let norm = Math.hypot(gx, gy) || 1;
  let len = 30;
  // Screen y is inverted relative to data y.
  let x1 = x0 + (gx / norm) * len;
  let y1 = y0 - (gy / norm) * len;
  g.append("line")
    .attr("x1", x0).attr("y1", y0).attr("x2", x1).attr("y2", y1)
    .attr("stroke", "#c00").attr("stroke-width", 2);
  g.append("circle")
    .attr("cx", x1).attr("cy", y1).attr("r", 3).attr("fill", "#c00");
}

function enableBrushSelect(): void {
  let svg = d3.select("#heatmap").select("svg");
  if (svg.empty()) {
    svg = d3.select("#heatmap").append("svg")
      .attr("class", "ml-sec-overlay")
      .style("position", "absolute").style("left", "0").style("top", "0")
      .attr("width", 340).attr("height", 340);
  }
  svg.style("pointer-events", "all");
  let brush = d3.brush()
    .extent([[HM_PADDING, HM_PADDING], [HM_PADDING + 300, HM_PADDING + 300]])
    .on("end", (event: any) => {
      if (!event.selection) { selectedForget = []; highlightSelected(); return; }
      let [[x0, y0], [x1, y1]] = event.selection;
      let [dx0, dy0] = pxToData(x0, y0);
      let [dx1, dy1] = pxToData(x1, y1);
      let xmin = Math.min(dx0, dx1), xmax = Math.max(dx0, dx1);
      let ymin = Math.min(dy0, dy1), ymax = Math.max(dy0, dy1);
      selectedForget = state.trainData.filter(p =>
        p.x >= xmin && p.x <= xmax && p.y >= ymin && p.y <= ymax);
      highlightSelected();
      d3.select("#unlearn-readout").html(
        `Selected ${selectedForget.length} points. Click "Forget selected".`);
    });
  svg.append("g").attr("class", "ml-brush").call(brush as any);
}

function disableBrushSelect(): void {
  let svg = d3.select("#heatmap").select("svg.ml-sec-overlay");
  svg.select("g.ml-brush").remove();
  svg.style("pointer-events", "none");
  selectedForget = [];
  highlightSelected();
}

function highlightSelected(): void {
  let g = heatmapOverlay();
  g.selectAll("circle.sel").remove();
  g.selectAll("circle.sel").data(selectedForget).enter()
    .append("circle").attr("class", "sel")
    .attr("cx", (d: Example2D) => dataToPx(d.x, d.y)[0])
    .attr("cy", (d: Example2D) => dataToPx(d.x, d.y)[1])
    .attr("r", 5).attr("fill", "none")
    .attr("stroke", "#000").attr("stroke-width", 1.5);
}

function drawRobustnessCurve(
    curve: {epsilon: number; accuracy: number}[]): void {
  let svg = d3.select("#adv-robustness-plot");
  svg.style("display", "block");
  svg.selectAll("*").remove();
  let w = 240, h = 140, m = 28;
  let maxEps = d3.max(curve, d => d.epsilon) || 1;
  let xs = d3.scaleLinear().domain([0, maxEps]).range([m, w - 8]);
  let ys = d3.scaleLinear().domain([0, 1]).range([h - m, 8]);
  svg.append("g").attr("transform", `translate(0,${h - m})`)
    .call(d3.axisBottom(xs).ticks(4) as any);
  svg.append("g").attr("transform", `translate(${m},0)`)
    .call(d3.axisLeft(ys).ticks(4) as any);
  let line = d3.line<{epsilon: number; accuracy: number}>()
    .x(d => xs(d.epsilon)).y(d => ys(d.accuracy));
  svg.append("path").datum(curve)
    .attr("fill", "none").attr("stroke", "#0877bd").attr("stroke-width", 2)
    .attr("d", line as any);
  svg.selectAll("circle.pt").data(curve).enter().append("circle")
    .attr("class", "pt")
    .attr("cx", d => xs(d.epsilon)).attr("cy", d => ys(d.accuracy))
    .attr("r", 2).attr("fill", "#0877bd");
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
    lastAdvClean = clean;
    lastAdvPerturbed = perturbed;
    let budget = perturbationBudget(clean, perturbed);
    let success = attackSuccessRate(predictXY, clean, perturbed);
    d3.select("#adv-readout").html(
      `Method: ${state.advMethod.toUpperCase()}, &epsilon;=${state.advEpsilon}<br>` +
      `Clean acc: ${(cleanAcc * 100).toFixed(1)}%<br>` +
      `Adversarial acc: ${(advAcc * 100).toFixed(1)}%<br>` +
      `Predictions flipped: ${flipped}/${clean.length}<br>` +
      `Attack success rate: ${(success * 100).toFixed(1)}%<br>` +
      `Budget: mean L2=${budget.meanL2.toFixed(3)}, ` +
      `mean L&infin;=${budget.meanLinf.toFixed(3)}, ` +
      `max L&infin;=${budget.maxLinf.toFixed(3)}`);
  });

  // Robustness curve: sweep epsilon and plot adversarial accuracy.
  d3.select("#adv-robustness").on("click", () => {
    let epsilons: number[] = [];
    for (let e = 0; e <= 3.0001; e += 0.25) epsilons.push(+e.toFixed(3));
    let curve = robustnessCurve(predictXY, (p, eps) => perturb(p, eps),
        state.testData, epsilons);
    drawRobustnessCurve(curve);
    let worst = curve[curve.length - 1];
    d3.select("#adv-readout").html(
      `Robustness sweep over &epsilon;&isin;[0,3]<br>` +
      `Acc @&epsilon;=0: ${(curve[0].accuracy * 100).toFixed(1)}%, ` +
      `@&epsilon;=${worst.epsilon}: ${(worst.accuracy * 100).toFixed(1)}%`);
  });

  // Saliency readout: dLoss/dx, dLoss/dy as an arrow at the last point.
  d3.select("#adv-saliency").on("click", () => {
    let p = lastClickedPoint ||
        (state.testData.length ? state.testData[0] : null);
    if (p == null) {
      d3.select("#adv-readout").html("Click a point on the plot first.");
      return;
    }
    let [gx, gy] = rawInputGradient(p.x, p.y, p.label);
    drawSaliencyArrow(p, gx, gy);
    d3.select("#adv-readout").html(
      `Saliency at (${p.x.toFixed(2)}, ${p.y.toFixed(2)})<br>` +
      `dLoss/dx = ${gx.toFixed(4)}<br>dLoss/dy = ${gy.toFixed(4)}`);
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

  // Nearest-N slider.
  let nearestN = d3.select("#forget-nearest-n").on("input", function() {
    d3.select("#forget-nearest-n-val").text((this as any).value);
  });
  nearestN.property("value", 10);
  d3.select("#forget-nearest-n-val").text(10);

  // Brush-to-select toggle.
  d3.select("#unlearn-brush-toggle").on("click", () => {
    brushSelectActive = !brushSelectActive;
    d3.select("#unlearn-brush-toggle").text(
      "Brush-select: " + (brushSelectActive ? "on" : "off"));
    if (brushSelectActive) {
      enableBrushSelect();
    } else {
      disableBrushSelect();
    }
  });

  d3.select("#forget-selected").on("click", () => {
    if (selectedForget.length === 0) {
      d3.select("#unlearn-readout").html(
        "No points selected. Toggle brush-select and drag a rectangle.");
      return;
    }
    doUnlearn(selectedForget.slice(), "Forgot selected");
    selectedForget = [];
    highlightSelected();
  });

  d3.select("#forget-nearest").on("click", () => {
    if (lastClickedPoint == null) {
      d3.select("#unlearn-readout").html("Click a point on the plot first.");
      return;
    }
    let n = +(d3.select("#forget-nearest-n").property("value") || 10);
    let sorted = state.trainData.slice().sort((a, b) =>
      dist2(a, lastClickedPoint) - dist2(b, lastClickedPoint));
    doUnlearn(sorted.slice(0, n), `Forgot nearest ${n}`);
  });

  d3.select("#unlearn-relearn").on("click", () => {
    let forget = selectedForget.length ? selectedForget.slice() :
        misclassified(state.trainData);
    if (forget.length === 0) {
      d3.select("#unlearn-readout").html("No forget set for relearn probe.");
      return;
    }
    let probe = relearnProbe(forget, 300, 1);
    d3.select("#unlearn-readout").html(
      `Relearn-time probe on ${forget.length} pts:<br>` +
      `${probe.reached ? probe.steps : "&ge;" + probe.steps} steps to ` +
      `recover accuracy (reached ${(probe.accuracy * 100).toFixed(1)}%).<br>` +
      `Weights restored after probe.`);
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

// ============================================================================
// Interpretability (2D classification only; degrades gracefully otherwise).
// ============================================================================

let ablatedNeuronId: string = null;          // id of currently ablated neuron
let ablatedSavedWeights: number[] = null;    // its outgoing weights, for restore
let confidenceContourOn = false;

/** True when interpretability tools are meaningful. */
function interpEnabled(): boolean {
  return state.problem === Problem.CLASSIFICATION && !state.threeD;
}

/** List hidden-layer nodes as {id, label}. */
function hiddenNodes(): {id: string; label: string}[] {
  let out: {id: string; label: string}[] = [];
  if (!network) { return out; }
  for (let l = 1; l < network.length - 1; l++) {
    network[l].forEach((node, i) => {
      out.push({id: node.id, label: `L${l} N${i + 1} (${node.id})`});
    });
  }
  return out;
}

function findNode(id: string): nn.Node {
  let found: nn.Node = null;
  nn.forEachNode(network, true, n => { if (n.id === id) { found = n; } });
  return found;
}

/** Output prediction for raw (x, y), respecting current ablation. */
function interpPredict(x: number, y: number): number {
  return nn.forwardProp(network, constructInput(x, y),
      state.weightQuantization, state.layerNorm);
}

/** Simple horizontal/vertical bar chart of labelled values into an svg. */
function drawBarChart(selector: string, items: {label: string; value: number}[],
    color = "#0877bd"): void {
  let svg = d3.select(selector);
  svg.selectAll("*").remove();
  if (!items.length) { return; }
  let w = +svg.attr("width"), h = +svg.attr("height");
  let m = {l: 70, r: 10, t: 6, b: 6};
  let maxAbs = d3.max(items, d => Math.abs(d.value)) || 1;
  let x = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([m.l, w - m.r]);
  let y = d3.scaleBand().domain(items.map(d => d.label))
      .range([m.t, h - m.b]).padding(0.2);
  let zero = x(0);
  svg.append("line").attr("x1", zero).attr("x2", zero)
    .attr("y1", m.t).attr("y2", h - m.b).attr("stroke", "#ccc");
  svg.selectAll("rect.bar").data(items).enter().append("rect")
    .attr("class", "bar")
    .attr("x", d => d.value >= 0 ? zero : x(d.value))
    .attr("y", d => y(d.label))
    .attr("width", d => Math.abs(x(d.value) - zero))
    .attr("height", y.bandwidth())
    .attr("fill", d => d.value >= 0 ? color : "#f59322");
  svg.selectAll("text.lbl").data(items).enter().append("text")
    .attr("class", "lbl")
    .attr("x", 2).attr("y", d => y(d.label) + y.bandwidth() / 2 + 3)
    .attr("font-size", "9px").attr("fill", "#666")
    .text(d => d.label);
}

/** Small line chart (x,y points) into an svg with given domains. */
function drawLineChart(selector: string, pts: {x: number; y: number}[],
    yDom: [number, number], title?: string): void {
  let svg = d3.select(selector);
  svg.selectAll("*").remove();
  if (!pts.length) { return; }
  let w = +svg.attr("width"), h = +svg.attr("height");
  let m = 24;
  let xs = d3.scaleLinear()
      .domain(d3.extent(pts, d => d.x) as [number, number]).range([m, w - 6]);
  let ys = d3.scaleLinear().domain(yDom).range([h - m, 6]);
  svg.append("g").attr("transform", `translate(0,${h - m})`)
    .call(d3.axisBottom(xs).ticks(4) as any);
  svg.append("g").attr("transform", `translate(${m},0)`)
    .call(d3.axisLeft(ys).ticks(3) as any);
  let line = d3.line<{x: number; y: number}>().x(d => xs(d.x)).y(d => ys(d.y));
  svg.append("path").datum(pts).attr("fill", "none")
    .attr("stroke", "#0877bd").attr("stroke-width", 2).attr("d", line as any);
  if (title) {
    svg.append("text").attr("x", m).attr("y", 12).attr("font-size", "9px")
      .attr("fill", "#666").text(title);
  }
}

// ---- 1. Saliency field ----
function computeSaliencyField(): void {
  let g = heatmapOverlay();
  g.selectAll("g.ip-saliency").remove();
  let grp = g.append("g").attr("class", "ip-saliency");
  let N = 9, h = 1e-2;
  let step = (2 * HM_MAXSCALE) / (N + 1);
  let maxMag = 0;
  let vecs: {x: number; y: number; gx: number; gy: number; mag: number}[] = [];
  for (let i = 1; i <= N; i++) {
    for (let j = 1; j <= N; j++) {
      let x = -HM_MAXSCALE + i * step;
      let y = -HM_MAXSCALE + j * step;
      let gx = (interpPredict(x + h, y) - interpPredict(x - h, y)) / (2 * h);
      let gy = (interpPredict(x, y + h) - interpPredict(x, y - h)) / (2 * h);
      let mag = Math.hypot(gx, gy);
      maxMag = Math.max(maxMag, mag);
      vecs.push({x, y, gx, gy, mag});
    }
  }
  let L = step * HM_FACTOR * 0.45;
  for (let v of vecs) {
    let [px, py] = dataToPx(v.x, v.y);
    let n = v.mag || 1;
    let x1 = px + (v.gx / n) * L;
    let y1 = py - (v.gy / n) * L;
    let op = maxMag ? 0.25 + 0.75 * (v.mag / maxMag) : 0.5;
    grp.append("line").attr("x1", px).attr("y1", py).attr("x2", x1).attr("y2", y1)
      .attr("stroke", "#333").attr("stroke-width", 1).attr("opacity", op);
    grp.append("circle").attr("cx", x1).attr("cy", y1).attr("r", 1.5)
      .attr("fill", "#333").attr("opacity", op);
  }
  d3.select("#ip-saliency-out").text(
    `Drew ${vecs.length} gradient arrows. Max |grad|=${maxMag.toFixed(3)}.`);
}

// ---- 2. Occlusion sensitivity ----
function computeOcclusion(): void {
  let ids = constructInputIds();
  let data = sampleSubset(state.testData.concat(state.trainData), 150);
  if (!data.length) { return; }
  // Mean of each feature across data, for "zeroing" semantics we set to mean.
  let base = data.map(p => interpPredict(p.x, p.y));
  let items: {label: string; value: number}[] = [];
  for (let fi = 0; fi < ids.length; fi++) {
    let sum = 0;
    for (let k = 0; k < data.length; k++) {
      let full = constructInput(data[k].x, data[k].y);
      full[fi] = 0;
      let out = nn.forwardProp(network, full,
          state.weightQuantization, state.layerNorm);
      sum += Math.abs(out - base[k]);
    }
    items.push({label: INPUTS[ids[fi]].label, value: sum / data.length});
  }
  drawBarChart("#ip-occlusion-chart", items, "#0877bd");
}

// ---- 3. Drop-feature importance ----
function computeDropFeature(): void {
  let ids = constructInputIds();
  let data = state.trainData.concat(state.testData);
  if (!data.length) { return; }
  let means = ids.map((_, fi) => d3.mean(data,
      p => constructInput(p.x, p.y)[fi]) || 0);
  let baseAcc = 0;
  for (let p of data) {
    if (Math.sign(interpPredict(p.x, p.y)) === Math.sign(p.label)) { baseAcc++; }
  }
  baseAcc /= data.length;
  let items: {label: string; value: number}[] = [];
  for (let fi = 0; fi < ids.length; fi++) {
    let correct = 0;
    for (let p of data) {
      let full = constructInput(p.x, p.y);
      full[fi] = means[fi];
      let out = nn.forwardProp(network, full,
          state.weightQuantization, state.layerNorm);
      if (Math.sign(out) === Math.sign(p.label)) { correct++; }
    }
    items.push({label: INPUTS[ids[fi]].label, value: baseAcc - correct / data.length});
  }
  drawBarChart("#ip-dropfeat-chart", items, "#0877bd");
}

// ---- 4. Partial dependence ----
function computePDP(): void {
  let data = sampleSubset(state.trainData.concat(state.testData), 120);
  if (!data.length) { return; }
  let N = 21;
  let mk = (vary: "x" | "y") => {
    let pts: {x: number; y: number}[] = [];
    for (let i = 0; i < N; i++) {
      let v = -HM_MAXSCALE + (2 * HM_MAXSCALE) * i / (N - 1);
      let sum = 0;
      for (let p of data) {
        let px = vary === "x" ? v : p.x;
        let py = vary === "y" ? v : p.y;
        sum += interpPredict(px, py);
      }
      pts.push({x: v, y: sum / data.length});
    }
    return pts;
  };
  drawLineChart("#ip-pdp-x", mk("x"), [-1, 1], "PDP over x");
  drawLineChart("#ip-pdp-y", mk("y"), [-1, 1], "PDP over y");
}

// ---- 5. Neuron ablation ----
function setAblation(id: string | null): void {
  // Restore previous ablation if any.
  if (ablatedNeuronId != null && ablatedSavedWeights != null) {
    let prev = findNode(ablatedNeuronId);
    if (prev) {
      prev.outputs.forEach((lk, i) => { lk.weight = ablatedSavedWeights[i]; });
    }
    ablatedNeuronId = null;
    ablatedSavedWeights = null;
  }
  if (id != null) {
    let node = findNode(id);
    if (node) {
      ablatedSavedWeights = node.outputs.map(lk => lk.weight);
      node.outputs.forEach(lk => { lk.weight = 0; });
      ablatedNeuronId = id;
    }
  }
}

function applyAblationToggle(): void {
  let on = (document.getElementById("ip-ablate-toggle") as HTMLInputElement).checked;
  let id = (d3.select("#ip-ablate-neuron").property("value") as string) || null;
  let accBefore = accuracy(state.testData);
  if (on && id) {
    setAblation(id);
  } else {
    setAblation(null);
  }
  let accAfter = accuracy(state.testData);
  updateUI();
  d3.select("#ip-ablate-out").text(
    (on && id ? `Ablated ${id}. ` : `Restored. `) +
    `Test acc ${(accBefore * 100).toFixed(1)}% -> ${(accAfter * 100).toFixed(1)}%`);
}

// ---- 6. Activation maximization ----
function computeActMax(): void {
  let id = (d3.select("#ip-actmax-neuron").property("value") as string);
  let node = findNode(id);
  if (!node) { return; }
  let best = {x: 0, y: 0, act: -Infinity};
  let N = 40;
  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= N; j++) {
      let x = -HM_MAXSCALE + (2 * HM_MAXSCALE) * i / N;
      let y = -HM_MAXSCALE + (2 * HM_MAXSCALE) * j / N;
      nn.forwardProp(network, constructInput(x, y),
          state.weightQuantization, state.layerNorm);
      if (node.output > best.act) { best = {x, y, act: node.output}; }
    }
  }
  // Local gradient-ascent refinement.
  let step = (2 * HM_MAXSCALE) / N, h = 1e-2;
  let x = best.x, y = best.y;
  for (let s = 0; s < 30; s++) {
    let f = (xx: number, yy: number) => {
      nn.forwardProp(network, constructInput(xx, yy),
          state.weightQuantization, state.layerNorm);
      return node.output;
    };
    let gx = (f(x + h, y) - f(x - h, y)) / (2 * h);
    let gy = (f(x, y + h) - f(x, y - h)) / (2 * h);
    x = Math.max(-HM_MAXSCALE, Math.min(HM_MAXSCALE, x + step * 0.3 * Math.sign(gx)));
    y = Math.max(-HM_MAXSCALE, Math.min(HM_MAXSCALE, y + step * 0.3 * Math.sign(gy)));
  }
  let act = (function() {
    nn.forwardProp(network, constructInput(x, y),
        state.weightQuantization, state.layerNorm);
    return node.output;
  })();
  let g = heatmapOverlay();
  g.selectAll("g.ip-actmax").remove();
  let grp = g.append("g").attr("class", "ip-actmax");
  let [px, py] = dataToPx(x, y);
  grp.append("circle").attr("cx", px).attr("cy", py).attr("r", 7)
    .attr("fill", "none").attr("stroke", "#7b2").attr("stroke-width", 3);
  d3.select("#ip-actmax-out").text(
    `Max activation ${act.toFixed(3)} at (${x.toFixed(2)}, ${y.toFixed(2)}).`);
}

// ---- 7. Counterfactual ----
function computeCounterfactual(): void {
  let p = lastClickedPoint ||
      (state.testData.length ? state.testData[0] : null);
  if (!p) { d3.select("#ip-counterfactual-out").text("Click a point first."); return; }
  let origClass = Math.sign(interpPredict(p.x, p.y)) || 1;
  let best: {x: number; y: number; d: number} = null;
  for (let r = 0.2; r <= 4 && !best; r += 0.2) {
    for (let a = 0; a < 24; a++) {
      let ang = (a / 24) * 2 * Math.PI;
      let x = p.x + r * Math.cos(ang);
      let y = p.y + r * Math.sin(ang);
      if (Math.abs(x) > HM_MAXSCALE || Math.abs(y) > HM_MAXSCALE) { continue; }
      if ((Math.sign(interpPredict(x, y)) || 1) !== origClass) {
        let d = Math.hypot(x - p.x, y - p.y);
        if (!best || d < best.d) { best = {x, y, d}; }
      }
    }
    if (best) { break; }
  }
  let g = heatmapOverlay();
  g.selectAll("g.ip-cf").remove();
  if (!best) { d3.select("#ip-counterfactual-out").text("No counterfactual found nearby."); return; }
  let grp = g.append("g").attr("class", "ip-cf");
  let [x0, y0] = dataToPx(p.x, p.y);
  let [x1, y1] = dataToPx(best.x, best.y);
  grp.append("line").attr("x1", x0).attr("y1", y0).attr("x2", x1).attr("y2", y1)
    .attr("stroke", "#b07").attr("stroke-width", 2);
  grp.append("circle").attr("cx", x1).attr("cy", y1).attr("r", 4).attr("fill", "#b07");
  d3.select("#ip-counterfactual-out").text(
    `From (${p.x.toFixed(2)},${p.y.toFixed(2)}) to ` +
    `(${best.x.toFixed(2)},${best.y.toFixed(2)}), dist ${best.d.toFixed(2)}.`);
}

// ---- 8. PCA of hidden activations ----
function lastHiddenActivations(p: Example2D): number[] {
  nn.forwardProp(network, constructInput(p.x, p.y),
      state.weightQuantization, state.layerNorm);
  let l = network.length - 2;          // last hidden layer index
  if (l < 1) { return []; }
  return network[l].map(n => n.output);
}

function computePCA(): void {
  let data = sampleSubset(state.testData.concat(state.trainData), 200);
  let rows = data.map(p => ({a: lastHiddenActivations(p), label: p.label}))
      .filter(r => r.a.length > 0);
  let svg = d3.select("#ip-pca-chart");
  svg.selectAll("*").remove();
  if (rows.length < 2 || rows[0].a.length < 1) {
    svg.append("text").attr("x", 8).attr("y", 20).attr("font-size", "10px")
      .text("Need >=1 hidden neuron.");
    return;
  }
  let dim = rows[0].a.length;
  let mean = new Array(dim).fill(0);
  for (let r of rows) { for (let k = 0; k < dim; k++) { mean[k] += r.a[k]; } }
  for (let k = 0; k < dim; k++) { mean[k] /= rows.length; }
  let X = rows.map(r => r.a.map((v, k) => v - mean[k]));
  // Covariance matrix.
  let cov: number[][] = [];
  for (let i = 0; i < dim; i++) {
    cov[i] = new Array(dim).fill(0);
    for (let j = 0; j < dim; j++) {
      let s = 0;
      for (let r of X) { s += r[i] * r[j]; }
      cov[i][j] = s / rows.length;
    }
  }
  let powerIter = (m: number[][]): number[] => {
    let v = new Array(dim).fill(0).map(() => Math.random());
    for (let it = 0; it < 100; it++) {
      let nv = new Array(dim).fill(0);
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) { nv[i] += m[i][j] * v[j]; }
      }
      let norm = Math.hypot(...nv) || 1;
      v = nv.map(x => x / norm);
    }
    return v;
  };
  let pc1 = powerIter(cov);
  let lam1 = 0;
  { let mv = new Array(dim).fill(0);
    for (let i = 0; i < dim; i++) { for (let j = 0; j < dim; j++) { mv[i] += cov[i][j] * pc1[j]; } }
    for (let i = 0; i < dim; i++) { lam1 += pc1[i] * mv[i]; } }
  // Deflate.
  let cov2 = cov.map((row, i) => row.map((val, j) => val - lam1 * pc1[i] * pc1[j]));
  let pc2 = dim > 1 ? powerIter(cov2) : new Array(dim).fill(0);
  let proj = X.map((r, idx) => ({
    x: r.reduce((s, v, k) => s + v * pc1[k], 0),
    y: r.reduce((s, v, k) => s + v * pc2[k], 0),
    label: rows[idx].label
  }));
  let w = +svg.attr("width"), h = +svg.attr("height"), m = 8;
  let xs = d3.scaleLinear().domain(d3.extent(proj, d => d.x) as [number, number])
      .range([m, w - m]);
  let ys = d3.scaleLinear().domain(d3.extent(proj, d => d.y) as [number, number])
      .range([h - m, m]);
  svg.selectAll("circle").data(proj).enter().append("circle")
    .attr("cx", d => xs(d.x)).attr("cy", d => ys(d.y)).attr("r", 2.5)
    .attr("fill", d => d.label > 0 ? "#0877bd" : "#f59322").attr("opacity", 0.7);
}

// ---- 9. Decision-tree surrogate (CART, depth<=3) ----
interface TreeNode {
  leaf?: boolean; cls?: number;
  feat?: number; thr?: number; left?: TreeNode; right?: TreeNode;
}
function fitSurrogate(): void {
  let data = sampleSubset(state.trainData.concat(state.testData), 200);
  let samples = data.map(p => ({
    f: [p.x, p.y], cls: Math.sign(interpPredict(p.x, p.y)) || 1
  }));
  if (!samples.length) { return; }
  let gini = (s: typeof samples) => {
    if (!s.length) { return 0; }
    let pos = s.filter(d => d.cls > 0).length / s.length;
    return 1 - pos * pos - (1 - pos) * (1 - pos);
  };
  let majority = (s: typeof samples) =>
    s.filter(d => d.cls > 0).length >= s.length / 2 ? 1 : -1;
  let build = (s: typeof samples, depth: number): TreeNode => {
    if (depth >= 3 || s.length < 4 || gini(s) === 0) {
      return {leaf: true, cls: majority(s)};
    }
    let best: {gain: number; feat: number; thr: number} = null;
    let parent = gini(s);
    for (let feat = 0; feat < 2; feat++) {
      let vals = s.map(d => d.f[feat]).sort((a, b) => a - b);
      for (let i = 1; i < vals.length; i++) {
        let thr = (vals[i - 1] + vals[i]) / 2;
        let L = s.filter(d => d.f[feat] <= thr);
        let R = s.filter(d => d.f[feat] > thr);
        if (!L.length || !R.length) { continue; }
        let g = parent - (L.length * gini(L) + R.length * gini(R)) / s.length;
        if (!best || g > best.gain) { best = {gain: g, feat, thr}; }
      }
    }
    if (!best || best.gain <= 1e-9) { return {leaf: true, cls: majority(s)}; }
    return {
      feat: best.feat, thr: best.thr,
      left: build(s.filter(d => d.f[best.feat] <= best.thr), depth + 1),
      right: build(s.filter(d => d.f[best.feat] > best.thr), depth + 1)
    };
  };
  let tree = build(samples, 0);
  let predict = (t: TreeNode, f: number[]): number => {
    while (!t.leaf) { t = f[t.feat] <= t.thr ? t.left : t.right; }
    return t.cls;
  };
  let agree = samples.filter(d => predict(tree, d.f) === d.cls).length;
  let render = (t: TreeNode, ind: string): string => {
    if (t.leaf) { return `${ind}-> class ${t.cls > 0 ? "+1" : "-1"}\n`; }
    let name = t.feat === 0 ? "x" : "y";
    return `${ind}${name} <= ${t.thr.toFixed(2)}?\n` +
      render(t.left, ind + "  ") + render(t.right, ind + "  ");
  };
  d3.select("#ip-surrogate-out").text(
    `Fidelity: ${(100 * agree / samples.length).toFixed(1)}%\n` + render(tree, ""));
}

// ---- 10/11. Confidence contours + entropy ----
function drawConfidenceContours(): void {
  let g = heatmapOverlay();
  g.selectAll("g.ip-contour").remove();
  if (!confidenceContourOn || !interpEnabled()) { return; }
  let grp = g.append("g").attr("class", "ip-contour");
  let N = 60;
  let grid: number[] = [];
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      let x = -HM_MAXSCALE + (2 * HM_MAXSCALE) * i / (N - 1);
      let y = HM_MAXSCALE - (2 * HM_MAXSCALE) * j / (N - 1);
      grid.push(Math.abs(interpPredict(x, y)));
    }
  }
  let cellW = (HM_FACTOR * 2 * HM_MAXSCALE) / (N - 1);
  let levels = [0.25, 0.5, 0.75];
  let colors = ["#999", "#666", "#333"];
  // Marching-squares-lite: draw segment midpoints where a level crossing occurs.
  for (let li = 0; li < levels.length; li++) {
    let lev = levels[li];
    for (let j = 0; j < N - 1; j++) {
      for (let i = 0; i < N - 1; i++) {
        let a = grid[j * N + i], b = grid[j * N + i + 1];
        let c = grid[(j + 1) * N + i];
        let px = HM_PADDING + i * cellW, py = HM_PADDING + j * cellW;
        if ((a - lev) * (b - lev) < 0) {
          let t = (lev - a) / (b - a);
          grp.append("circle").attr("cx", px + t * cellW).attr("cy", py)
            .attr("r", 0.8).attr("fill", colors[li]);
        }
        if ((a - lev) * (c - lev) < 0) {
          let t = (lev - a) / (c - a);
          grp.append("circle").attr("cx", px).attr("cy", py + t * cellW)
            .attr("r", 0.8).attr("fill", colors[li]);
        }
      }
    }
  }
}

function computeGridEntropy(): void {
  let N = 50, total = 0, confTotal = 0;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let x = -HM_MAXSCALE + (2 * HM_MAXSCALE) * i / (N - 1);
      let y = -HM_MAXSCALE + (2 * HM_MAXSCALE) * j / (N - 1);
      let out = interpPredict(x, y);
      // Map output in [-1,1] to probability p in [0,1].
      let p = Math.min(1, Math.max(0, (out + 1) / 2));
      let e = (p === 0 || p === 1) ? 0 :
        -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
      total += e;
      confTotal += Math.abs(out);
    }
  }
  let n = N * N;
  d3.select("#ip-entropy-out").text(
    `Mean entropy: ${(total / n).toFixed(3)} bits\n` +
    `Mean confidence |out|: ${(confTotal / n).toFixed(3)}`);
}

// ---- 12. What-if inspector ----
function whatIfInspect(): void {
  let p = lastClickedPoint ||
      (state.testData.length ? state.testData[0] : null);
  if (!p) { d3.select("#ip-whatif-out").text("Click a point first."); return; }
  let out = interpPredict(p.x, p.y);
  let lines = [`(${p.x.toFixed(2)}, ${p.y.toFixed(2)}) -> ` +
    `${out.toFixed(3)} [${out >= 0 ? "+1 blue" : "-1 orange"}]`];
  for (let l = 1; l < network.length; l++) {
    let vals = network[l].map(n => n.output.toFixed(2)).join(", ");
    lines.push(`L${l}: ${vals}`);
  }
  d3.select("#ip-whatif-out").text(lines.join("\n"));
}

// ---- 13. k-NN baseline ----
function computeKNN(): void {
  let train = state.trainData, test = state.testData;
  if (!train.length || !test.length) {
    d3.select("#ip-knn-out").text("Need train and test data."); return;
  }
  let k = Math.min(5, train.length);
  let correct = 0;
  for (let q of test) {
    let nbrs = train.map(t => ({d: dist2(q, t), label: t.label}))
      .sort((a, b) => a.d - b.d).slice(0, k);
    let s = nbrs.reduce((acc, n) => acc + Math.sign(n.label), 0);
    let pred = s >= 0 ? 1 : -1;
    if (pred === Math.sign(q.label)) { correct++; }
  }
  let knnAcc = correct / test.length;
  let netAcc = accuracy(test);
  d3.select("#ip-knn-out").text(
    `k=${k}-NN test acc: ${(knnAcc * 100).toFixed(1)}%\n` +
    `Network test acc: ${(netAcc * 100).toFixed(1)}%`);
}

// ---- 14. Per-feature contribution at a point ----
function computeContributions(): void {
  let p = lastClickedPoint ||
      (state.testData.length ? state.testData[0] : null);
  if (!p) { return; }
  let ids = constructInputIds();
  let base = interpPredict(p.x, p.y);
  let baseInput = constructInput(p.x, p.y);
  let means = ids.map((_, fi) => d3.mean(
      state.trainData.concat(state.testData), q => constructInput(q.x, q.y)[fi]) || 0);
  let items: {label: string; value: number}[] = [];
  for (let fi = 0; fi < ids.length; fi++) {
    let perturbed = baseInput.slice();
    perturbed[fi] = means[fi];
    let out = nn.forwardProp(network, perturbed,
        state.weightQuantization, state.layerNorm);
    items.push({label: INPUTS[ids[fi]].label, value: base - out});
  }
  drawBarChart("#ip-contrib-chart", items, "#0877bd");
}

/** Populate the two neuron-select dropdowns from current hidden layers. */
function refreshNeuronSelects(): void {
  let nodes = hiddenNodes();
  for (let sel of ["#ip-ablate-neuron", "#ip-actmax-neuron"]) {
    let prev = d3.select(sel).property("value");
    let s = d3.select(sel);
    s.selectAll("option").remove();
    s.selectAll("option").data(nodes).enter().append("option")
      .attr("value", d => d.id).text(d => d.label);
    if (nodes.some(n => n.id === prev)) { s.property("value", prev); }
  }
}

/** Per-step hook: keep dropdowns current and redraw contours if enabled. */
function interpAfterUpdate(): void {
  let note = document.getElementById("interp-nonclass-note");
  let grid = document.querySelector(".interp-grid") as HTMLElement;
  if (note && grid) {
    let ok = interpEnabled();
    note.style.display = ok ? "none" : "block";
    grid.style.display = ok ? "flex" : "none";
  }
  if (!interpEnabled()) { return; }
  refreshNeuronSelects();
  drawConfidenceContours();
}

function makeInterpGUI(): void {
  d3.select("#interp-toggle").on("click", () => {
    let sec = document.getElementById("interp-section");
    if (sec) { sec.classList.toggle("collapsed"); }
  });
  d3.select("#ip-saliency").on("click", computeSaliencyField);
  d3.select("#ip-occlusion").on("click", computeOcclusion);
  d3.select("#ip-dropfeat").on("click", computeDropFeature);
  d3.select("#ip-pdp").on("click", computePDP);
  d3.select("#ip-ablate-neuron").on("change", () => {
    if ((document.getElementById("ip-ablate-toggle") as HTMLInputElement).checked) {
      applyAblationToggle();
    }
  });
  d3.select("#ip-ablate-toggle").on("change", applyAblationToggle);
  d3.select("#ip-actmax").on("click", computeActMax);
  d3.select("#ip-counterfactual").on("click", computeCounterfactual);
  d3.select("#ip-pca").on("click", computePCA);
  d3.select("#ip-surrogate").on("click", fitSurrogate);
  d3.select("#ip-contour-toggle").on("change", function() {
    confidenceContourOn = (this as any).checked;
    drawConfidenceContours();
  });
  d3.select("#ip-entropy").on("click", computeGridEntropy);
  d3.select("#ip-whatif").on("click", whatIfInspect);
  d3.select("#ip-knn").on("click", computeKNN);
  d3.select("#ip-contrib").on("click", computeContributions);
  refreshNeuronSelects();
}

// ============================================================================
// Experiments: run tracking & comparison (features 1-12).
// ============================================================================

interface SavedRun {
  id: string;
  name: string;
  hash: string;            // serialized State (config) for reproduction.
  summary: string;         // human-readable config summary.
  iter: number;
  lossTrain: number;
  lossTest: number;
  accTrain: number;
  accTest: number;
  problem: string;
  // Downsampled per-epoch metric history for overlay comparison.
  history: Array<{iter: number; lossTrain: number; lossTest: number;
      accTrain: number; accTest: number}>;
}

const XP_STORAGE_KEY = "nn-playground-runs";
let xpRuns: SavedRun[] = [];
let xpSortBy = "accTest";
let xpOverlayMetric = "loss";
// Live per-epoch history of the currently active run (downsampled on save).
let xpLiveHistory: Array<{iter: number; lossTrain: number; lossTest: number;
    accTrain: number; accTest: number}> = [];

const XP_PALETTE = ["#e8710a", "#0877bd", "#16a085", "#8e44ad", "#c0392b",
  "#27ae60", "#2980b9", "#d35400"];

function xpIsClassification(): boolean {
  return state.problem === Problem.CLASSIFICATION;
}

/** Current run's accuracy (NaN for regression). */
function xpCurrentAcc(data: Example2D[]): number {
  if (!xpIsClassification() || !network) { return NaN; }
  return computeClassMetrics(network, data).accuracy;
}

/** Record one downsampled live-history sample for the active run. */
function xpRecordLiveHistory(): void {
  let accTrain = xpCurrentAcc(state.trainData);
  let accTest = xpCurrentAcc(state.testData);
  xpLiveHistory.push({iter, lossTrain, lossTest, accTrain, accTest});
  // Downsample: keep at most ~120 evenly-spaced samples.
  if (xpLiveHistory.length > 240) {
    let kept: typeof xpLiveHistory = [];
    for (let i = 0; i < xpLiveHistory.length; i += 2) {
      kept.push(xpLiveHistory[i]);
    }
    xpLiveHistory = kept;
  }
}

function xpResetLiveHistory(): void {
  xpLiveHistory = [];
}

/** Feature 10: auto-name from key hyperparameters. */
function xpAutoName(): string {
  let act = getKeyFromValue(activations, state.activation) || "act";
  let opt = state.optimizer;
  let lr = state.learningRate;
  let shape = state.networkShape.length ? state.networkShape.join("x") : "0";
  return `${act}·${opt}·lr${lr}·${shape}`;
}

function xpConfigSummary(): string {
  let act = getKeyFromValue(activations, state.activation) || "?";
  let ds = xpIsClassification()
    ? (getKeyFromValue(datasets, state.dataset) || "?")
    : (getKeyFromValue(regDatasets, state.regDataset) || "?");
  let shape = state.networkShape.length ? state.networkShape.join("-") : "none";
  return `${ds} | ${shape} | ${act}/${state.optimizer} | lr ${state.learningRate}` +
    ` | bs ${state.batchSize}`;
}

function xpLoadFromStorage(): void {
  try {
    if (typeof localStorage === "undefined") { return; }
    let raw = localStorage.getItem(XP_STORAGE_KEY);
    if (!raw) { return; }
    let parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) { xpRuns = parsed; }
  } catch (e) { /* ignore corrupt storage */ }
}

function xpPersist(): void {
  try {
    if (typeof localStorage === "undefined") { return; }
    localStorage.setItem(XP_STORAGE_KEY, JSON.stringify(xpRuns));
  } catch (e) { /* ignore quota errors */ }
}

function xpDownsampleHistory(): SavedRun["history"] {
  let src = xpLiveHistory.length ? xpLiveHistory : trainingHistory;
  let max = 60;
  if (src.length <= max) { return src.map(h => ({...h})); }
  let step = Math.ceil(src.length / max);
  let out: SavedRun["history"] = [];
  for (let i = 0; i < src.length; i += step) {
    out.push({...src[i]});
  }
  return out;
}

function xpSaveRun(): void {
  let suggested = xpAutoName();
  let name = suggested;
  if (typeof window !== "undefined" && typeof window.prompt === "function") {
    let entered = window.prompt("Name this run:", suggested);
    if (entered == null) { return; }       // cancelled
    if (entered.trim() !== "") { name = entered.trim(); }
  }
  // Capture the serialized config without disturbing the URL permanently:
  // state.serialize() writes to the hash, which already reflects current state.
  state.serialize();
  let hash = (typeof window !== "undefined") ? window.location.hash.slice(1) : "";
  let run: SavedRun = {
    id: "run-" + Date.now() + "-" + Math.floor(Math.random() * 1e6),
    name,
    hash,
    summary: xpConfigSummary(),
    iter,
    lossTrain,
    lossTest,
    accTrain: xpCurrentAcc(state.trainData),
    accTest: xpCurrentAcc(state.testData),
    problem: getKeyFromValue(problems, state.problem),
    history: xpDownsampleHistory()
  };
  xpRuns.push(run);
  xpPersist();
  xpRenderAll();
}

function xpDeleteRun(id: string): void {
  xpRuns = xpRuns.filter(r => r.id !== id);
  xpPersist();
  xpRenderAll();
}

function xpClearRuns(): void {
  if (typeof window !== "undefined" && typeof window.confirm === "function") {
    if (!window.confirm("Delete all saved runs?")) { return; }
  }
  xpRuns = [];
  xpPersist();
  xpRenderAll();
}

/** Feature 3: restore a saved run's CONFIG (apply state, reset). */
function xpLoadRun(id: string): void {
  let run = xpRuns.filter(r => r.id === id)[0];
  if (!run || typeof window === "undefined") { return; }
  window.location.hash = run.hash;
  state = State.deserializeState();
  // Re-filter hidden inputs (mirrors startup behavior).
  state.getHiddenProps().forEach(prop => {
    if (prop in INPUTS) { delete INPUTS[prop]; }
  });
  // Re-sync controls (d3 .on() replaces handlers, so no double-binding).
  makeGUI();
  generateData(false);
  reset();
}

function xpBestRun(): SavedRun {
  let best: SavedRun = null;
  xpRuns.forEach(r => {
    if (isNaN(r.accTest)) { return; }
    if (best == null || r.accTest > best.accTest) { best = r; }
  });
  return best;
}

function xpFmtPct(v: number): string {
  return isNaN(v) ? "—" : (v * 100).toFixed(1) + "%";
}
function xpFmtNum(v: number): string {
  return isNaN(v) ? "—" : v.toFixed(3);
}

function xpSortedRuns(): SavedRun[] {
  let arr = xpRuns.slice();
  let key = xpSortBy;
  arr.sort((a, b) => {
    let av = (a as any)[key];
    let bv = (b as any)[key];
    if (isNaN(av)) { av = -Infinity; }
    if (isNaN(bv)) { bv = -Infinity; }
    // Loss: lower is better (ascending); others: higher is better.
    if (key === "lossTest" || key === "lossTrain") { return av - bv; }
    return bv - av;
  });
  return arr;
}

function xpEscape(s: string): string {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/** Features 2 & 11: leaderboard + best-run highlight. */
function xpRenderLeaderboard(): void {
  let wrap = d3.select("#xp-leaderboard");
  if (wrap.empty()) { return; }
  if (!xpRuns.length) {
    wrap.html("<div class=\"xp-empty\">No saved runs yet. Click \"Save run\".</div>");
    return;
  }
  let best = xpBestRun();
  let rows = xpSortedRuns();
  let html = "<table class=\"xp-table\"><thead><tr>" +
    "<th>Name</th><th>Config</th><th>Epoch</th>" +
    "<th>Train loss</th><th>Test loss</th>" +
    "<th>Train acc</th><th>Test acc</th><th></th></tr></thead><tbody>";
  rows.forEach(r => {
    let isBest = best && r.id === best.id;
    html += "<tr class=\"" + (isBest ? "xp-best-row" : "") + "\">" +
      "<td>" + xpEscape(r.name) + (isBest ? " ★" : "") + "</td>" +
      "<td>" + xpEscape(r.summary) + "</td>" +
      "<td>" + r.iter + "</td>" +
      "<td>" + xpFmtNum(r.lossTrain) + "</td>" +
      "<td>" + xpFmtNum(r.lossTest) + "</td>" +
      "<td>" + xpFmtPct(r.accTrain) + "</td>" +
      "<td>" + xpFmtPct(r.accTest) + "</td>" +
      "<td class=\"xp-row-actions\">" +
        "<button data-xp-load=\"" + r.id + "\">Load</button>" +
        "<button data-xp-del=\"" + r.id + "\">Del</button>" +
      "</td></tr>";
  });
  html += "</tbody></table>";
  wrap.html(html);
  wrap.selectAll("button[data-xp-load]").on("click", function() {
    xpLoadRun((this as HTMLElement).getAttribute("data-xp-load"));
  });
  wrap.selectAll("button[data-xp-del]").on("click", function() {
    xpDeleteRun((this as HTMLElement).getAttribute("data-xp-del"));
  });
}

/** Feature 11/12: best badge + live current-vs-best delta. */
function xpRenderBadgeAndDelta(): void {
  let best = xpBestRun();
  let badge = d3.select("#xp-best-badge");
  if (!badge.empty()) {
    if (best) {
      badge.classed("has-best", true)
        .text("Best run: " + best.name + " (" + xpFmtPct(best.accTest) + ")");
    } else {
      badge.classed("has-best", false).text("Best run: —");
    }
  }
  let deltaEl = d3.select("#xp-current-delta");
  if (!deltaEl.empty()) {
    if (best && xpIsClassification() && network) {
      let cur = xpCurrentAcc(state.testData);
      let d = cur - best.accTest;
      let sign = d >= 0 ? "+" : "";
      deltaEl.text("Current vs best: " + xpFmtPct(cur) +
        " (" + sign + (d * 100).toFixed(1) + " pts)");
    } else if (best) {
      let d = lossTest - best.lossTest;
      let sign = d >= 0 ? "+" : "";
      deltaEl.text("Current vs best (test loss): " + xpFmtNum(lossTest) +
        " (" + sign + d.toFixed(3) + ")");
    } else {
      deltaEl.text("Current vs best: —");
    }
  }
}

/** Feature 5: overlay comparison chart of saved runs' curves. */
function xpRenderOverlay(): void {
  let svgSel = d3.select("#xp-overlay-chart");
  if (svgSel.empty()) { return; }
  let svg = svgSel as any;
  svg.selectAll("*").remove();
  let legend = d3.select("#xp-overlay-legend");
  if (!legend.empty()) { legend.html(""); }
  let runs = xpRuns.filter(r => r.history && r.history.length > 1);
  if (!runs.length) {
    svg.append("text").attr("x", 12).attr("y", 24)
      .attr("fill", "#999").attr("font-size", "12px")
      .text("Save runs to compare their curves here.");
    return;
  }
  let W = +svg.attr("width");
  let H = +svg.attr("height");
  let m = {top: 12, right: 12, bottom: 26, left: 40};
  let iw = W - m.left - m.right;
  let ih = H - m.top - m.bottom;
  let useAcc = xpOverlayMetric === "acc";
  let valueOf = (h: SavedRun["history"][0]) =>
    useAcc ? h.accTest : h.lossTest;
  let maxIter = 1, maxVal = useAcc ? 1 : 1e-9, minVal = 0;
  runs.forEach(r => {
    r.history.forEach(h => {
      if (h.iter > maxIter) { maxIter = h.iter; }
      let v = valueOf(h);
      if (!isNaN(v) && v > maxVal) { maxVal = v; }
    });
  });
  if (useAcc) { maxVal = 1; }
  let xScale = d3.scaleLinear().domain([0, maxIter]).range([0, iw]);
  let yScale = d3.scaleLinear().domain([minVal, maxVal]).range([ih, 0]);
  let g = svg.append("g")
    .attr("transform", "translate(" + m.left + "," + m.top + ")");
  // Axes (lightweight).
  g.append("line").attr("x1", 0).attr("y1", ih).attr("x2", iw).attr("y2", ih)
    .attr("stroke", "#ccc");
  g.append("line").attr("x1", 0).attr("y1", 0).attr("x2", 0).attr("y2", ih)
    .attr("stroke", "#ccc");
  g.append("text").attr("x", -4).attr("y", 0).attr("text-anchor", "end")
    .attr("font-size", "9px").attr("fill", "#999").text(maxVal.toFixed(2));
  g.append("text").attr("x", -4).attr("y", ih).attr("text-anchor", "end")
    .attr("font-size", "9px").attr("fill", "#999").text(minVal.toFixed(2));
  let line = d3.line<SavedRun["history"][0]>()
    .defined(h => !isNaN(valueOf(h)))
    .x(h => xScale(h.iter))
    .y(h => yScale(valueOf(h)));
  runs.forEach((r, i) => {
    let color = XP_PALETTE[i % XP_PALETTE.length];
    g.append("path").datum(r.history)
      .attr("fill", "none").attr("stroke", color).attr("stroke-width", 1.5)
      .attr("d", line as any);
    if (!legend.empty()) {
      let item = legend.append("div").attr("class", "xp-legend-item");
      item.append("span").attr("class", "xp-swatch")
        .style("background", color);
      item.append("span").text(r.name);
    }
  });
}

/** Feature 6: compare A/B configs (diff-highlight) + final metrics. */
function xpPopulateComparePickers(): void {
  let a = d3.select("#xp-compare-a");
  let b = d3.select("#xp-compare-b");
  if (a.empty() || b.empty()) { return; }
  let prevA = a.property("value");
  let prevB = b.property("value");
  let opts = "<option value=\"\">—</option>" + xpRuns.map(r =>
    "<option value=\"" + r.id + "\">" + xpEscape(r.name) + "</option>").join("");
  a.html(opts);
  b.html(opts);
  a.property("value", prevA);
  b.property("value", prevB);
}

function xpRunConfigMap(run: SavedRun): {[k: string]: string} {
  let map: {[k: string]: string} = {};
  (run.hash || "").split("&").forEach(kv => {
    let idx = kv.indexOf("=");
    if (idx > 0) { map[kv.slice(0, idx)] = kv.slice(idx + 1); }
  });
  return map;
}

function xpRenderCompare(): void {
  let wrap = d3.select("#xp-compare-table");
  if (wrap.empty()) { return; }
  let aId = d3.select("#xp-compare-a").property("value");
  let bId = d3.select("#xp-compare-b").property("value");
  let a = xpRuns.filter(r => r.id === aId)[0];
  let b = xpRuns.filter(r => r.id === bId)[0];
  if (!a || !b) {
    wrap.html("<div class=\"xp-empty\">Pick two runs to compare.</div>");
    return;
  }
  let ma = xpRunConfigMap(a);
  let mb = xpRunConfigMap(b);
  let keys: string[] = [];
  let seen: {[k: string]: boolean} = {};
  Object.keys(ma).concat(Object.keys(mb)).forEach(k => {
    if (!seen[k]) { seen[k] = true; keys.push(k); }
  });
  keys.sort();
  let html = "<table class=\"xp-table\"><thead><tr><th>Hyperparameter</th>" +
    "<th>" + xpEscape(a.name) + "</th><th>" + xpEscape(b.name) +
    "</th></tr></thead><tbody>";
  keys.forEach(k => {
    let va = ma[k] == null ? "—" : ma[k];
    let vb = mb[k] == null ? "—" : mb[k];
    let diff = va !== vb;
    let cls = diff ? " class=\"xp-diff\"" : "";
    html += "<tr><td>" + xpEscape(k) + "</td><td" + cls + ">" +
      xpEscape(va) + "</td><td" + cls + ">" + xpEscape(vb) + "</td></tr>";
  });
  // Final metrics rows.
  let metricRow = (label: string, fa: string, fb: string) =>
    "<tr><td><b>" + label + "</b></td><td>" + fa + "</td><td>" + fb + "</td></tr>";
  html += metricRow("epoch", String(a.iter), String(b.iter));
  html += metricRow("test loss", xpFmtNum(a.lossTest), xpFmtNum(b.lossTest));
  html += metricRow("test acc", xpFmtPct(a.accTest), xpFmtPct(b.accTest));
  html += "</tbody></table>";
  wrap.html(html);
}

/** Feature 8: export/import all runs as JSON. */
function xpExportRuns(): void {
  let json = JSON.stringify(xpRuns, null, 2);
  let ta = d3.select("#xp-io-text");
  if (!ta.empty()) { ta.property("value", json); }
  downloadText("nn-playground-runs.json", json, "application/json");
}

function xpImportRuns(): void {
  let ta = d3.select("#xp-io-text");
  if (ta.empty()) { return; }
  let text = ta.property("value") as string;
  try {
    let parsed = JSON.parse(text);
    if (!Array.isArray(parsed)) { throw new Error("not an array"); }
    // Merge by id (imported wins).
    let byId: {[k: string]: SavedRun} = {};
    xpRuns.forEach(r => { byId[r.id] = r; });
    parsed.forEach((r: SavedRun) => { if (r && r.id) { byId[r.id] = r; } });
    xpRuns = Object.keys(byId).map(k => byId[k]);
    xpPersist();
    xpRenderAll();
  } catch (e) {
    if (!ta.empty()) {
      d3.select("#xp-io-text").property("value",
        "Import failed: invalid JSON.\n\n" + text);
    }
  }
}

/** Feature 9: export the current run's per-epoch metric history as CSV. */
function xpExportLiveHistoryCsv(): void {
  let src = xpLiveHistory.length ? xpLiveHistory : trainingHistory;
  let rows = ["iter,lossTrain,lossTest,accTrain,accTest"];
  src.forEach(h => {
    rows.push(h.iter + "," + h.lossTrain + "," + h.lossTest + "," +
      (isNaN(h.accTrain) ? "" : h.accTrain) + "," +
      (isNaN(h.accTest) ? "" : h.accTest));
  });
  downloadText("run-history.csv", rows.join("\n"), "text/csv");
}

/** Feature 7: mini grid search over lr x hidden-size on throwaway networks. */
function xpGridSearch(): void {
  let resWrap = d3.select("#xp-grid-results");
  if (resWrap.empty()) { return; }
  if (!xpIsClassification() || state.threeD) {
    resWrap.html("<div class=\"xp-empty\">Grid search requires 2D " +
      "classification mode.</div>");
    return;
  }
  let lrs = [0.01, 0.03, 0.1];
  let sizes = [2, 4, 8];        // single hidden layer of N units.
  let steps = 30;
  let inputIds = constructInputIds();
  let numInputs = constructInput(0, 0).length;
  let errFunc = currentErrorFunc();
  let train = state.trainData;
  let test = state.testData;
  // Results[sizeIdx][lrIdx] = test accuracy.
  let results: number[][] = [];
  let bestAcc = -1, bestCell = "";
  sizes.forEach((size, si) => {
    results[si] = [];
    lrs.forEach((lr, li) => {
      let shape = [numInputs, size, 1];
      let net = nn.buildNetwork(shape, state.activation, nn.Activations.TANH,
        inputIds, false);
      nn.applyWeightInit(net, weightInits[state.weightInit]);
      let optType = optimizers[state.optimizer] || nn.OptimizerType.SGD;
      for (let s = 0; s < steps; s++) {
        train.forEach((point, i) => {
          nn.forwardProp(net, constructInput(point.x, point.y), null, false);
          nn.backProp(net, point.label, errFunc);
          if ((i + 1) % state.batchSize === 0) {
            nn.updateWeights(net, lr, state.regularization,
              state.regularizationRate, optType, 0, 0);
          }
        });
      }
      // Test accuracy on the throwaway net.
      let correct = 0;
      test.forEach(p => {
        let o = nn.forwardProp(net, constructInput(p.x, p.y), null, false);
        if (Math.sign(o) === Math.sign(p.label)) { correct++; }
      });
      let acc = test.length ? correct / test.length : 0;
      results[si][li] = acc;
      if (acc > bestAcc) {
        bestAcc = acc;
        bestCell = "size " + size + ", lr " + lr;
      }
    });
  });
  // Render as a heatmap-styled table.
  let heat = (v: number) => {
    let t = Math.max(0, Math.min(1, v));
    let r = Math.round(255 + (39 - 255) * t);
    let g = Math.round(255 + (174 - 255) * t);
    let b = Math.round(255 + (96 - 255) * t);
    return "rgb(" + r + "," + g + "," + b + ")";
  };
  let html = "<div class=\"xp-grid-caption\">Best: " + xpEscape(bestCell) +
    " → " + xpFmtPct(bestAcc) + "</div>";
  html += "<table class=\"xp-table\"><thead><tr><th>hidden \\ lr</th>";
  lrs.forEach(lr => { html += "<th>" + lr + "</th>"; });
  html += "</tr></thead><tbody>";
  sizes.forEach((size, si) => {
    html += "<tr><th>" + size + "</th>";
    lrs.forEach((lr, li) => {
      let acc = results[si][li];
      html += "<td class=\"xp-heat\" style=\"background:" + heat(acc) + "\">" +
        xpFmtPct(acc) + "</td>";
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  resWrap.html(html);
  // Throwaway nets used local vars only; main model (network) is untouched.
}

function xpRenderAll(): void {
  xpRenderLeaderboard();
  xpRenderBadgeAndDelta();
  xpRenderOverlay();
  xpPopulateComparePickers();
  xpRenderCompare();
}

function initExperimentsGUI(): void {
  xpLoadFromStorage();
  d3.select("#experiments-toggle").on("click", () => {
    let sec = document.getElementById("experiments-section");
    if (sec) { sec.classList.toggle("collapsed"); }
  });
  d3.select("#xp-save-run").on("click", () => xpSaveRun());
  d3.select("#xp-clear-runs").on("click", () => xpClearRuns());
  d3.select("#xp-export-runs").on("click", () => xpExportRuns());
  d3.select("#xp-import-runs").on("click", () => xpImportRuns());
  d3.select("#xp-export-history-csv").on("click", () => xpExportLiveHistoryCsv());
  d3.select("#xp-grid-search").on("click", () => xpGridSearch());
  d3.select("#xp-sort-by").on("change", function() {
    xpSortBy = (this as HTMLSelectElement).value;
    xpRenderLeaderboard();
  });
  d3.selectAll("input[name='xp-overlay-metric']").on("change", function() {
    xpOverlayMetric = (this as HTMLInputElement).value;
    xpRenderOverlay();
  });
  d3.select("#xp-compare-a").on("change", () => xpRenderCompare());
  d3.select("#xp-compare-b").on("change", () => xpRenderCompare());
  xpRenderAll();
}

drawDatasetThumbnails();
initTutorial();
makeGUI();
makeAdvancedGUI();
makeFineTuneGUI();
makeAnalysisGUI();
makeInterpGUI();
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

// ===========================================================================
// UX / sharing / quality-of-life features.
// All wiring lives here to avoid double-binding controls set up in makeGUI().
// ===========================================================================

/** Lightweight toast/notification helper (feature 18). */
function uxToast(message: string, isWarning = false): void {
  let container = document.getElementById("ux-toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "ux-toast-container";
    document.body.appendChild(container);
  }
  let toast = document.createElement("div");
  toast.className = "ux-toast" + (isWarning ? " ux-toast-warn" : "");
  toast.textContent = message;
  container.appendChild(toast);
  // Force reflow so the transition runs.
  void toast.offsetWidth;
  toast.classList.add("ux-toast-show");
  setTimeout(() => {
    toast.classList.remove("ux-toast-show");
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 350);
  }, isWarning ? 4000 : 2000);
}

const UX_PREFS_KEY = "nnpg-ux-prefs";

function uxLoadPrefs(): any {
  try {
    let raw = window.localStorage.getItem(UX_PREFS_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch (e) {
    return {};
  }
}

function uxSavePrefs(patch: any): void {
  try {
    let prefs = uxLoadPrefs();
    for (let k in patch) {
      prefs[k] = patch[k];
    }
    window.localStorage.setItem(UX_PREFS_KEY, JSON.stringify(prefs));
  } catch (e) { /* localStorage unavailable */ }
}

/** Feature 1: copy a full share link (state lives in the URL hash). */
function uxCopyShareLink(): void {
  try { state.serialize(); } catch (e) { /* ignore */ }
  let url = window.location.href;
  let done = () => uxToast("Share link copied!");
  let fallback = () => {
    try {
      let ta = document.createElement("textarea");
      ta.value = url;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      done();
    } catch (e) {
      uxToast("Copy failed; URL: " + url, true);
    }
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(url).then(done, fallback);
  } else {
    fallback();
  }
}

/** Feature 2: dark mode toggle, persisted in localStorage. */
function uxApplyDarkMode(on: boolean): void {
  let dark = document.getElementById("dark-style") as HTMLLinkElement;
  let light = document.getElementById("light-style") as HTMLLinkElement;
  if (dark) { (dark as any).disabled = !on; }
  if (light) { (light as any).disabled = on; }
  document.body.classList.toggle("ux-dark", on);
  let toggle = document.getElementById("dark-mode-toggle") as HTMLInputElement;
  if (toggle) { toggle.checked = on; }
}

/** Feature 3: curated preset configurations ("Model zoo"). */
interface UxPreset {
  name: string;
  apply: () => void;
}

function uxSetFeatures(active: string[]): void {
  for (let key in INPUTS) {
    if (state[key] !== undefined) {
      state[key] = active.indexOf(key) !== -1;
    }
  }
  // The canonical input toggles.
  ["x", "y", "xSquared", "ySquared", "xTimesY", "sinX", "sinY"].forEach(k => {
    state[k] = active.indexOf(k) !== -1;
  });
}

const UX_PRESETS: UxPreset[] = [
  {
    name: "Spiral solver (tanh, 4x8)",
    apply: () => {
      state.problem = Problem.CLASSIFICATION;
      if (datasets["spiral"]) { state.dataset = datasets["spiral"]; }
      state.networkShape = [8, 8, 8, 8];
      state.numHiddenLayers = 4;
      state.activation = activations["tanh"];
      state.learningRate = 0.03;
      state.regularization = null;
      state.regularizationRate = 0;
      state.batchSize = 10;
      state.noise = 0;
      uxSetFeatures(["x", "y", "xSquared", "ySquared", "xTimesY", "sinX", "sinY"]);
    }
  },
  {
    name: "XOR minimal",
    apply: () => {
      state.problem = Problem.CLASSIFICATION;
      if (datasets["xor"]) { state.dataset = datasets["xor"]; }
      state.networkShape = [2];
      state.numHiddenLayers = 1;
      state.activation = activations["tanh"];
      state.learningRate = 0.1;
      state.regularization = null;
      state.regularizationRate = 0;
      state.batchSize = 10;
      uxSetFeatures(["xTimesY"]);
    }
  },
  {
    name: "Circle (relu)",
    apply: () => {
      state.problem = Problem.CLASSIFICATION;
      if (datasets["circle"]) { state.dataset = datasets["circle"]; }
      state.networkShape = [4, 2];
      state.numHiddenLayers = 2;
      state.activation = activations["relu"];
      state.learningRate = 0.03;
      uxSetFeatures(["x", "y"]);
    }
  },
  {
    name: "Deep & narrow",
    apply: () => {
      state.problem = Problem.CLASSIFICATION;
      if (datasets["spiral"]) { state.dataset = datasets["spiral"]; }
      state.networkShape = [3, 3, 3, 3, 3, 3];
      state.numHiddenLayers = 6;
      state.activation = activations["relu"];
      state.learningRate = 0.03;
      uxSetFeatures(["x", "y"]);
    }
  },
  {
    name: "Wide & shallow",
    apply: () => {
      state.problem = Problem.CLASSIFICATION;
      if (datasets["circle"]) { state.dataset = datasets["circle"]; }
      state.networkShape = [12];
      state.numHiddenLayers = 1;
      state.activation = activations["tanh"];
      state.learningRate = 0.03;
      uxSetFeatures(["x", "y"]);
    }
  },
  {
    name: "Regression plane",
    apply: () => {
      state.problem = Problem.REGRESSION;
      if (regDatasets["reg-plane"]) { state.regDataset = regDatasets["reg-plane"]; }
      state.networkShape = [3];
      state.numHiddenLayers = 1;
      state.activation = activations["tanh"];
      state.learningRate = 0.03;
      uxSetFeatures(["x", "y"]);
    }
  },
  {
    name: "Robust (adv training on)",
    apply: () => {
      state.problem = Problem.CLASSIFICATION;
      if (datasets["spiral"]) { state.dataset = datasets["spiral"]; }
      state.networkShape = [8, 8];
      state.numHiddenLayers = 2;
      state.activation = activations["relu"];
      state.learningRate = 0.03;
      state.adversarialTraining = true;
      state.advEpsilon = 0.5;
      state.noise = 20;
      uxSetFeatures(["x", "y"]);
    }
  },
  {
    name: "Overfit demo (no reg, tiny train)",
    apply: () => {
      state.problem = Problem.CLASSIFICATION;
      if (datasets["spiral"]) { state.dataset = datasets["spiral"]; }
      state.networkShape = [8, 8, 8];
      state.numHiddenLayers = 3;
      state.activation = activations["relu"];
      state.learningRate = 0.1;
      state.regularization = null;
      state.regularizationRate = 0;
      state.percTrainData = 10;
      state.noise = 25;
      uxSetFeatures(["x", "y", "xSquared", "ySquared", "xTimesY"]);
    }
  }
];

function uxApplyPreset(index: number): void {
  let preset = UX_PRESETS[index];
  if (!preset) { return; }
  preset.apply();
  state.serialize();
  // Sync the controls that reset() does not itself reflect.
  d3.select("#activations").property("value",
      getKeyFromValue(activations, state.activation));
  d3.select("#learningRate").property("value", state.learningRate);
  d3.select("#problem").property("value",
      state.problem === Problem.REGRESSION ? "regression" : "classification");
  let dsKey = getKeyFromValue(datasets, state.dataset);
  d3.selectAll("canvas[data-dataset]").classed("selected", false);
  if (dsKey) {
    d3.select(`canvas[data-dataset=${dsKey}]`).classed("selected", true);
  }
  generateData();
  reset();
  uxToast("Applied preset: " + preset.name);
}

/** Feature 14/15: snapshot & weight helpers (in memory). */
let uxSnapshot: {biases: {[id: string]: number}; links: {[id: string]: number}} | null = null;

function uxTakeSnapshot(): void {
  if (network == null) { uxToast("No network to snapshot.", true); return; }
  let biases: {[id: string]: number} = {};
  let links: {[id: string]: number} = {};
  nn.forEachNode(network, true, node => {
    biases[node.id] = node.bias;
    node.inputLinks.forEach(link => { links[link.id] = link.weight; });
  });
  uxSnapshot = {biases, links};
  uxToast("Snapshot saved.");
}

function uxRestoreSnapshot(): void {
  if (network == null || uxSnapshot == null) {
    uxToast("No snapshot to restore.", true);
    return;
  }
  nn.forEachNode(network, true, node => {
    if (uxSnapshot.biases[node.id] != null) { node.bias = uxSnapshot.biases[node.id]; }
    node.inputLinks.forEach(link => {
      if (uxSnapshot.links[link.id] != null) { link.weight = uxSnapshot.links[link.id]; }
    });
  });
  lossTrain = getLoss(network, state.trainData);
  lossTest = getLoss(network, state.testData);
  drawNetwork(network);
  updateUI(true);
  uxToast("Snapshot restored.");
}

function uxRandomizeWeights(): void {
  if (network == null) { uxToast("No network.", true); return; }
  Math.seedrandom(Math.random().toFixed(8));
  nn.applyWeightInit(network, weightInits[state.weightInit]);
  applyFrozenLayers();
  iter = 0;
  lossTrain = getLoss(network, state.trainData);
  lossTest = getLoss(network, state.testData);
  drawNetwork(network);
  updateUI(true);
  uxToast("Weights re-initialized.");
}

// ===========================================================================
// Training-methodology features (k-fold CV, LR finder, ensemble, weight noise).
// All operate on throwaway networks (except weight-noise, which can restore the
// real model) and reuse the existing build/train primitives.
// ===========================================================================

/** Builds a fresh network matching the current architecture & init scheme. */
function buildFreshNetwork(): nn.Node[][] {
  let inputIds = constructInputIds();
  let shape = [inputIds.length].concat(state.networkShape).concat([1]);
  let net = nn.buildNetwork(shape, state.activation, nn.Activations.TANH,
      inputIds, state.initZero);
  nn.applyWeightInit(net, weightInits[state.weightInit]);
  return net;
}

/** Loss for an arbitrary network/dataset using the active error function. */
function lossOf(net: nn.Node[][], data: Example2D[]): number {
  let errFunc = currentErrorFunc();
  let loss = 0;
  for (let i = 0; i < data.length; i++) {
    let out = nn.forwardProp(net, constructInput(data[i].x, data[i].y),
        state.weightQuantization, state.layerNorm);
    loss += errFunc.error(out, data[i].label);
  }
  return data.length ? loss / data.length : 0;
}

/** Accuracy for an arbitrary network/dataset. */
function accuracyOf(net: nn.Node[][], data: Example2D[]): number {
  let correct = 0;
  for (let i = 0; i < data.length; i++) {
    let out = nn.forwardProp(net, constructInput(data[i].x, data[i].y),
        state.weightQuantization, state.layerNorm);
    if ((out >= 0 ? 1 : -1) === (data[i].label >= 0 ? 1 : -1)) { correct++; }
  }
  return data.length ? correct / data.length : 0;
}

/** Trains a network in place for a number of epochs over the given data. */
function trainNetwork(net: nn.Node[][], data: Example2D[], epochs: number,
    lr: number): void {
  let optimizerType = optimizers[state.optimizer] || nn.OptimizerType.SGD;
  let errFunc = currentErrorFunc();
  for (let e = 0; e < epochs; e++) {
    data.forEach((point, i) => {
      nn.forwardProp(net, constructInput(point.x, point.y),
          state.weightQuantization, state.layerNorm, state.dropout, true,
          state.batchNorm);
      nn.backProp(net, point.label, errFunc);
      if ((i + 1) % state.batchSize === 0) {
        nn.updateWeights(net, lr, state.regularization,
            state.regularizationRate, optimizerType, state.gradClip,
            state.weightDecay);
      }
    });
  }
}

/** Returns a bootstrap (sample-with-replacement) copy of the data. */
function bootstrapSample(data: Example2D[]): Example2D[] {
  let out: Example2D[] = [];
  for (let i = 0; i < data.length; i++) {
    out.push(data[Math.floor(Math.random() * data.length)]);
  }
  return out;
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
}
function std(xs: number[]): number {
  let m = mean(xs);
  return Math.sqrt(mean(xs.map(x => (x - m) * (x - m))));
}

/** Feature 3: k-fold cross-validation on the training data. */
function runKFoldCV(k: number, epochs: number): void {
  let data = state.trainData.slice();
  if (data.length < k) {
    uxToast("Not enough training data for " + k + " folds.", true);
    return;
  }
  shuffle(data);
  let accs: number[] = [];
  let losses: number[] = [];
  let foldSize = Math.floor(data.length / k);
  let lr = state.learningRate;
  for (let f = 0; f < k; f++) {
    let start = f * foldSize;
    let end = f === k - 1 ? data.length : start + foldSize;
    let valFold = data.slice(start, end);
    let trainFold = data.slice(0, start).concat(data.slice(end));
    let net = buildFreshNetwork();
    trainNetwork(net, trainFold, epochs, lr);
    accs.push(accuracyOf(net, valFold));
    losses.push(lossOf(net, valFold));
  }
  let readout = document.getElementById("tm-cv-readout");
  if (readout) {
    readout.innerHTML =
        "k=" + k + " folds, " + epochs + " epochs/fold<br>" +
        "Val accuracy: " + (mean(accs) * 100).toFixed(1) + "% ± " +
        (std(accs) * 100).toFixed(1) + "%<br>" +
        "Val loss: " + mean(losses).toFixed(3) + " ± " + std(losses).toFixed(3);
  }
  uxToast("k-fold CV done: " + (mean(accs) * 100).toFixed(1) + "% mean acc.");
}

/** Feature 4: learning-rate finder — sweep LR and plot loss vs LR. */
function runLRFinder(): void {
  let data = state.trainData;
  if (data.length === 0) { uxToast("No training data.", true); return; }
  let lrMin = 1e-4, lrMax = 3;
  let steps = 25;
  let results: {lr: number; loss: number}[] = [];
  let net = buildFreshNetwork();
  for (let s = 0; s < steps; s++) {
    let lr = lrMin * Math.pow(lrMax / lrMin, s / (steps - 1));
    // A few mini-batches per LR on the evolving throwaway net.
    trainNetwork(net, data, 1, lr);
    let loss = lossOf(net, data);
    results.push({lr, loss});
    if (!isFinite(loss) || loss > 1e3) { break; }
  }
  // Suggest LR at steepest descent (min loss with some margin before blow-up).
  let best = results.reduce((a, b) => b.loss < a.loss ? b : a, results[0]);
  let suggested = best.lr / 10;  // common heuristic: an order below the min.
  plotLRFinder(results, suggested);
  let readout = document.getElementById("tm-lrf-readout");
  if (readout) {
    readout.textContent = "Suggested LR ~ " + suggested.toPrecision(2) +
        " (min loss " + best.loss.toFixed(3) + " at LR " +
        best.lr.toPrecision(2) + ")";
  }
  uxToast("LR finder done. Suggested ~ " + suggested.toPrecision(2));
}

function plotLRFinder(results: {lr: number; loss: number}[],
    suggested: number): void {
  let svg = d3.select<SVGSVGElement, unknown>("#tm-lrf-plot");
  if (svg.empty()) { return; }
  svg.selectAll("*").remove();
  let W = 260, H = 140, m = {t: 8, r: 8, b: 24, l: 36};
  svg.attr("width", W).attr("height", H);
  let x = d3.scaleLog()
      .domain([results[0].lr, results[results.length - 1].lr])
      .range([m.l, W - m.r]);
  let maxLoss = d3.max(results, d => d.loss) || 1;
  let y = d3.scaleLinear().domain([0, maxLoss]).range([H - m.b, m.t]);
  let line = d3.line<{lr: number; loss: number}>()
      .x(d => x(d.lr)).y(d => y(d.loss));
  svg.append("path").datum(results)
      .attr("fill", "none").attr("stroke", "#f59322").attr("stroke-width", 2)
      .attr("d", line);
  svg.append("line")
      .attr("x1", x(suggested)).attr("x2", x(suggested))
      .attr("y1", m.t).attr("y2", H - m.b)
      .attr("stroke", "#0877bd").attr("stroke-dasharray", "3,3");
  svg.append("g").attr("transform", "translate(0," + (H - m.b) + ")")
      .call(d3.axisBottom(x).ticks(4, "~g"));
  svg.append("g").attr("transform", "translate(" + m.l + ",0)")
      .call(d3.axisLeft(y).ticks(4));
}

/** Feature 9/10: train an ensemble and draw the averaged decision boundary. */
function runEnsemble(n: number, bagging: boolean): void {
  if (state.trainData.length === 0) { uxToast("No training data.", true); return; }
  let epochs = 30;
  let lr = state.learningRate;
  let members: nn.Node[][][] = [];
  for (let i = 0; i < n; i++) {
    let net = buildFreshNetwork();
    let trainSet = bagging ? bootstrapSample(state.trainData) : state.trainData;
    trainNetwork(net, trainSet, epochs, lr);
    members.push(net);
  }
  // Averaged-output boundary on the heatmap.
  let xScale = d3.scaleLinear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scaleLinear().domain([DENSITY - 1, 0]).range(xDomain);
  let mat: number[][] = new Array(DENSITY);
  for (let i = 0; i < DENSITY; i++) {
    mat[i] = new Array(DENSITY);
    for (let j = 0; j < DENSITY; j++) {
      let xv = xScale(i), yv = yScale(j);
      let sum = 0;
      for (let mIdx = 0; mIdx < members.length; mIdx++) {
        sum += nn.forwardProp(members[mIdx], constructInput(xv, yv),
            state.weightQuantization, state.layerNorm);
      }
      mat[i][j] = sum / members.length;
    }
  }
  heatMap.updateBackground(mat, state.discretize);
  // Ensemble vs single accuracy on the test set.
  let single = accuracyOf(members[0], state.testData);
  let ensCorrect = 0;
  for (let t = 0; t < state.testData.length; t++) {
    let p = state.testData[t];
    let sum = 0;
    for (let mIdx = 0; mIdx < members.length; mIdx++) {
      sum += nn.forwardProp(members[mIdx], constructInput(p.x, p.y),
          state.weightQuantization, state.layerNorm);
    }
    let avg = sum / members.length;
    if ((avg >= 0 ? 1 : -1) === (p.label >= 0 ? 1 : -1)) { ensCorrect++; }
  }
  let ensAcc = state.testData.length ? ensCorrect / state.testData.length : 0;
  let readout = document.getElementById("tm-ensemble-readout");
  if (readout) {
    readout.innerHTML = "N=" + n + (bagging ? " (bagging)" : "") + "<br>" +
        "Single test acc: " + (single * 100).toFixed(1) + "%<br>" +
        "Ensemble test acc: " + (ensAcc * 100).toFixed(1) + "%";
  }
  uxToast("Ensemble drawn (avg of " + n + "). Reset to restore the live model.");
}

/** Feature 12: perturb the trained weights with Gaussian noise, show delta. */
let tmWeightNoiseSnapshot: {biases: {[id: string]: number};
    links: {[id: string]: number}} | null = null;

function applyWeightNoise(sigma: number): void {
  if (network == null) { uxToast("No network.", true); return; }
  // Snapshot so it can be restored.
  let biases: {[id: string]: number} = {};
  let links: {[id: string]: number} = {};
  nn.forEachNode(network, true, node => {
    biases[node.id] = node.bias;
    node.inputLinks.forEach(link => { links[link.id] = link.weight; });
  });
  tmWeightNoiseSnapshot = {biases, links};
  let accBefore = accuracyOf(network, state.testData);
  nn.forEachNode(network, true, node => {
    node.bias += gaussianNoise() * sigma;
    node.inputLinks.forEach(link => { link.weight += gaussianNoise() * sigma; });
  });
  let accAfter = accuracyOf(network, state.testData);
  lossTrain = getLoss(network, state.trainData);
  lossTest = getLoss(network, state.testData);
  drawNetwork(network);
  updateUI(true);
  let readout = document.getElementById("tm-weightnoise-readout");
  if (readout) {
    readout.innerHTML = "σ=" + sigma + "<br>" +
        "Test acc: " + (accBefore * 100).toFixed(1) + "% → " +
        (accAfter * 100).toFixed(1) + "% (Δ " +
        ((accAfter - accBefore) * 100).toFixed(1) + " pts)";
  }
}

function restoreWeightNoise(): void {
  if (network == null || tmWeightNoiseSnapshot == null) {
    uxToast("Nothing to restore.", true);
    return;
  }
  nn.forEachNode(network, true, node => {
    if (tmWeightNoiseSnapshot.biases[node.id] != null) {
      node.bias = tmWeightNoiseSnapshot.biases[node.id];
    }
    node.inputLinks.forEach(link => {
      if (tmWeightNoiseSnapshot.links[link.id] != null) {
        link.weight = tmWeightNoiseSnapshot.links[link.id];
      }
    });
  });
  lossTrain = getLoss(network, state.trainData);
  lossTest = getLoss(network, state.testData);
  drawNetwork(network);
  updateUI(true);
  uxToast("Weights restored.");
}

/** Feature 15: training-config summary readout. */
function updateTrainingConfigSummary(): void {
  let el = document.getElementById("tm-config-summary");
  if (!el) { return; }
  let lossKey = getKeyFromValue(lossFunctions, currentErrorFunc()) ||
      state.lossFunction;
  let regKey = getKeyFromValue(regularizations, state.regularization) || "none";
  let parts = [
    "loss: " + lossKey,
    "optimizer: " + state.optimizer,
    "lr: " + state.learningRate,
    "lr-schedule: " + state.lrSchedule,
    "batch: " + state.batchSize,
    "reg: " + regKey + " (" + state.regularizationRate + ")",
    "dropout: " + state.dropout,
    "weight-decay: " + state.weightDecay,
    "class-weighting: " + (state.classWeighting ? "on" : "off"),
    "label-noise: " + state.labelNoise + "%",
    "input-jitter: " + state.inputJitter,
    "mixup: " + (state.mixup ? "on" : "off"),
    "grad-noise: " + state.gradientNoise,
    "epoch-shuffle: " + (state.epochShuffle ? "on" : "off")
  ];
  el.innerHTML = parts.join("<br>");
}

/** Feature 2: early-stopping monitor (called from the after-step hook). */
function tmEarlyStopCheck(): void {
  let chk = document.getElementById("tm-earlystop-enable") as HTMLInputElement;
  if (!chk || !chk.checked) { tmEarlyStopBest = Infinity; tmEarlyStopWait = 0; return; }
  let patienceInput = document.getElementById("tm-earlystop-patience") as HTMLInputElement;
  let patience = patienceInput ? Math.max(1, +patienceInput.value || 10) : 10;
  if (lossTest < tmEarlyStopBest - 1e-6) {
    tmEarlyStopBest = lossTest;
    tmEarlyStopWait = 0;
  } else {
    tmEarlyStopWait++;
    if (tmEarlyStopWait >= patience && player.isActive()) {
      player.pause();
      tmEarlyStopWait = 0;
      tmEarlyStopBest = Infinity;
      let badge = document.getElementById("tm-earlystop-badge");
      if (badge) { badge.style.display = "inline"; }
      uxToast("Stopped early (no test-loss improvement for " + patience + " evals).");
    }
  }
}

/** Wires up all training-methodology controls. */
function initTrainingMethodologyGUI(): void {
  // ---- Feature 1: loss-function dropdown ----
  let lossSel = d3.select("#loss-function");
  if (!lossSel.empty()) {
    lossSel.property("value", state.lossFunction);
    lossSel.on("change.tm", function() {
      state.lossFunction = (this as HTMLSelectElement).value;
      state.serialize();
      parametersChanged = true;
      updateTrainingConfigSummary();
      reset();
    });
  }

  // ---- Feature 5: class weighting ----
  let cw = d3.select("#tm-class-weighting");
  if (!cw.empty()) {
    cw.property("checked", state.classWeighting);
    cw.on("change.tm", function() {
      state.classWeighting = (this as HTMLInputElement).checked;
      state.serialize();
      updateTrainingConfigSummary();
    });
  }

  // ---- Feature 6: label noise ----
  let ln = d3.select("#tm-label-noise");
  if (!ln.empty()) {
    ln.property("value", state.labelNoise);
    d3.select("#tm-label-noise-val").text(String(state.labelNoise));
    ln.on("input.tm", function() {
      state.labelNoise = +(this as HTMLInputElement).value;
      d3.select("#tm-label-noise-val").text(String(state.labelNoise));
      state.serialize();
      updateTrainingConfigSummary();
      generateData();
      reset();
    });
  }

  // ---- Feature 7: mixup ----
  let mx = d3.select("#tm-mixup");
  if (!mx.empty()) {
    mx.property("checked", state.mixup);
    mx.on("change.tm", function() {
      state.mixup = (this as HTMLInputElement).checked;
      state.serialize();
      updateTrainingConfigSummary();
    });
  }

  // ---- Feature 8: input jitter ----
  let ij = d3.select("#tm-input-jitter");
  if (!ij.empty()) {
    ij.property("value", state.inputJitter);
    d3.select("#tm-input-jitter-val").text(String(state.inputJitter));
    ij.on("input.tm", function() {
      state.inputJitter = +(this as HTMLInputElement).value;
      d3.select("#tm-input-jitter-val").text(String(state.inputJitter));
      state.serialize();
      updateTrainingConfigSummary();
    });
  }

  // ---- Feature 11: gradient noise ----
  let gnoise = d3.select("#tm-grad-noise");
  if (!gnoise.empty()) {
    gnoise.property("value", state.gradientNoise);
    d3.select("#tm-grad-noise-val").text(String(state.gradientNoise));
    gnoise.on("input.tm", function() {
      state.gradientNoise = +(this as HTMLInputElement).value;
      d3.select("#tm-grad-noise-val").text(String(state.gradientNoise));
      state.serialize();
      updateTrainingConfigSummary();
    });
  }

  // ---- Feature 14: epoch-wise shuffling ----
  let es = d3.select("#tm-epoch-shuffle");
  if (!es.empty()) {
    es.property("checked", state.epochShuffle);
    es.on("change.tm", function() {
      state.epochShuffle = (this as HTMLInputElement).checked;
      state.serialize();
      updateTrainingConfigSummary();
    });
  }

  // ---- Feature 2: early stopping ----
  let esBadge = document.getElementById("tm-earlystop-badge");
  let esChk = d3.select("#tm-earlystop-enable");
  if (!esChk.empty()) {
    esChk.on("change.tm", () => {
      tmEarlyStopBest = Infinity;
      tmEarlyStopWait = 0;
      if (esBadge) { esBadge.style.display = "none"; }
    });
  }

  // ---- Feature 3: k-fold CV ----
  let cvBtn = document.getElementById("tm-cv-btn");
  if (cvBtn) {
    cvBtn.addEventListener("click", () => {
      let kInput = document.getElementById("tm-cv-k") as HTMLInputElement;
      let k = kInput ? Math.max(2, +kInput.value || 5) : 5;
      runKFoldCV(k, 30);
    });
  }

  // ---- Feature 4: LR finder ----
  let lrfBtn = document.getElementById("tm-lrf-btn");
  if (lrfBtn) { lrfBtn.addEventListener("click", () => runLRFinder()); }

  // ---- Feature 9/10: ensemble + bagging ----
  let ensBtn = document.getElementById("tm-ensemble-btn");
  if (ensBtn) {
    ensBtn.addEventListener("click", () => {
      let nInput = document.getElementById("tm-ensemble-n") as HTMLInputElement;
      let n = nInput ? Math.max(2, +nInput.value || 5) : 5;
      let bagChk = document.getElementById("tm-ensemble-bagging") as HTMLInputElement;
      runEnsemble(n, bagChk ? bagChk.checked : false);
    });
  }

  // ---- Feature 12: weight noise / perturbation ----
  let wnBtn = document.getElementById("tm-weightnoise-btn");
  if (wnBtn) {
    wnBtn.addEventListener("click", () => {
      let sInput = document.getElementById("tm-weightnoise-sigma") as HTMLInputElement;
      let sigma = sInput ? Math.max(0, +sInput.value || 0.1) : 0.1;
      applyWeightNoise(sigma);
    });
  }
  let wnRestore = document.getElementById("tm-weightnoise-restore");
  if (wnRestore) { wnRestore.addEventListener("click", () => restoreWeightNoise()); }

  // ---- Feature 13: re-shuffle split ----
  let reshufBtn = document.getElementById("tm-reshuffle-split");
  if (reshufBtn) { reshufBtn.addEventListener("click", () => reshuffleSplit()); }

  updateTrainingConfigSummary();
}

/** Main entry point for all UX features. */
function initUXFeatures(): void {
  let prefs = uxLoadPrefs();

  // ---- Feature 18 already defined (uxToast) ----

  // ---- Feature 2: dark mode ----
  let darkToggle = document.getElementById("dark-mode-toggle") as HTMLInputElement;
  let darkOn = prefs.darkMode === true;
  uxApplyDarkMode(darkOn);
  if (darkToggle) {
    darkToggle.addEventListener("change", () => {
      uxApplyDarkMode(darkToggle.checked);
      uxSavePrefs({darkMode: darkToggle.checked});
    });
  }

  // ---- Feature 1: copy share link ----
  let copyBtn = document.getElementById("ux-copy-link");
  if (copyBtn) { copyBtn.addEventListener("click", uxCopyShareLink); }

  // ---- Feature 3: presets dropdown ----
  let presetSel = document.getElementById("ux-preset-select") as HTMLSelectElement;
  if (presetSel) {
    UX_PRESETS.forEach((p, i) => {
      let opt = document.createElement("option");
      opt.value = String(i);
      opt.text = p.name;
      presetSel.appendChild(opt);
    });
    presetSel.addEventListener("change", () => {
      let v = presetSel.value;
      if (v === "") { return; }
      uxApplyPreset(+v);
      presetSel.value = "";
    });
  }

  // ---- Feature 5: steps-per-tick speed control ----
  let speedSlider = document.getElementById("ux-speed") as HTMLInputElement;
  if (speedSlider) {
    if (typeof prefs.stepsPerTick === "number") {
      speedSlider.value = String(prefs.stepsPerTick);
    }
    uxStepsPerTick = +speedSlider.value || 1;
    let speedLabel = document.getElementById("ux-speed-value");
    if (speedLabel) { speedLabel.textContent = String(uxStepsPerTick); }
    speedSlider.addEventListener("input", () => {
      uxStepsPerTick = Math.max(1, +speedSlider.value || 1);
      if (speedLabel) { speedLabel.textContent = String(uxStepsPerTick); }
      uxSavePrefs({stepsPerTick: uxStepsPerTick});
    });
  }

  // ---- Feature 7: auto-stop on convergence ----
  let convChk = document.getElementById("ux-conv-enable") as HTMLInputElement;
  let convThreshInput = document.getElementById("ux-conv-threshold") as HTMLInputElement;
  let convWindow: number[] = [];

  // ---- Feature 6: run N epochs then stop ----
  let runNRemaining = 0;

  // ---- Feature 8/6/7: after-step hook ----
  uxAfterStep = () => {
    // Pause-on-NaN (feature 8).
    if (!isFinite(lossTrain) || !isFinite(lossTest)) {
      if (player.isActive()) {
        player.pause();
        uxToast("Loss became NaN/Infinity — training paused.", true);
      }
      runNRemaining = 0;
      convWindow = [];
      return;
    }
    // Run-N (feature 6).
    if (runNRemaining > 0) {
      runNRemaining--;
      if (runNRemaining === 0) {
        player.pause();
        uxToast("Finished requested epochs.");
      }
    }
    // Convergence auto-stop (feature 7).
    if (convChk && convChk.checked) {
      let thr = convThreshInput ? +convThreshInput.value : 0.0001;
      convWindow.push(lossTrain);
      if (convWindow.length > 20) { convWindow.shift(); }
      if (convWindow.length >= 20) {
        let max = Math.max.apply(null, convWindow);
        let min = Math.min.apply(null, convWindow);
        if (max - min < thr && player.isActive()) {
          player.pause();
          convWindow = [];
          uxToast("Converged (loss change < " + thr + ") — paused.");
        }
      }
    }
    // Early stopping (training-methodology feature 2).
    tmEarlyStopCheck();
    // Status bar steps/sec (feature 12).
    uxTickStatus();
  };

  // ---- Feature 6 button ----
  let runNBtn = document.getElementById("ux-run-n-btn");
  let runNInput = document.getElementById("ux-run-n") as HTMLInputElement;
  if (runNBtn && runNInput) {
    runNBtn.addEventListener("click", () => {
      let n = Math.max(1, parseInt(runNInput.value, 10) || 0);
      runNRemaining = n;
      if (iter === 0) { simulationStarted(); }
      if (!player.isActive()) { player.playOrPause(); }
    });
  }

  // ---- Feature 13: numerical LR input ----
  let lrNum = document.getElementById("ux-lr-num") as HTMLInputElement;
  if (lrNum) {
    lrNum.value = String(state.learningRate);
    lrNum.addEventListener("change", () => {
      let v = parseFloat(lrNum.value);
      if (isFinite(v) && v > 0) {
        state.learningRate = v;
        state.serialize();
        parametersChanged = true;
        d3.select("#learningRate").property("value", v);
        uxToast("Learning rate set to " + v);
      }
    });
  }

  // ---- Feature 14: randomize weights ----
  let randBtn = document.getElementById("ux-randomize-weights");
  if (randBtn) { randBtn.addEventListener("click", uxRandomizeWeights); }

  // ---- Feature 15: snapshot / restore ----
  let snapBtn = document.getElementById("ux-snapshot");
  let restoreBtn = document.getElementById("ux-restore");
  if (snapBtn) { snapBtn.addEventListener("click", uxTakeSnapshot); }
  if (restoreBtn) { restoreBtn.addEventListener("click", uxRestoreSnapshot); }

  // ---- Feature 10: reset view / clear localStorage ----
  let clearBtn = document.getElementById("ux-clear-storage");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      try {
        window.localStorage.removeItem(UX_PREFS_KEY);
        window.localStorage.removeItem("nnpg-onboard-dismissed");
      } catch (e) { /* ignore */ }
      window.location.hash = "";
      window.location.reload();
    });
  }

  // ---- Feature 11: fullscreen toggle for output area ----
  let fsBtn = document.getElementById("ux-fullscreen");
  if (fsBtn) {
    fsBtn.addEventListener("click", () => {
      let el = document.querySelector(".column.output") as any;
      if (!el) { return; }
      let doc = document as any;
      if (!doc.fullscreenElement) {
        if (el.requestFullscreen) { el.requestFullscreen().catch(() => {}); }
      } else {
        if (doc.exitFullscreen) { doc.exitFullscreen().catch(() => {}); }
      }
    });
  }

  // ---- Feature 16: persisted UI prefs for analysis panels ----
  // Restore open/closed <details> panels and remember changes.
  let savedPanels = prefs.openPanels || {};
  d3.selectAll("details[id]").each(function() {
    let el = this as HTMLDetailsElement;
    if (savedPanels[el.id] !== undefined) {
      el.open = !!savedPanels[el.id];
    }
    el.addEventListener("toggle", () => {
      let cur = uxLoadPrefs().openPanels || {};
      cur[el.id] = el.open;
      uxSavePrefs({openPanels: cur});
    });
  });

  // ---- Feature 17: onboarding hint banner ----
  let banner = document.getElementById("ux-onboard");
  let dismissed = false;
  try { dismissed = window.localStorage.getItem("nnpg-onboard-dismissed") === "1"; } catch (e) {}
  if (banner) {
    if (dismissed) {
      banner.style.display = "none";
    }
    let dismissBtn = document.getElementById("ux-onboard-dismiss");
    if (dismissBtn) {
      dismissBtn.addEventListener("click", () => {
        banner.style.display = "none";
        try { window.localStorage.setItem("nnpg-onboard-dismissed", "1"); } catch (e) {}
      });
    }
  }

  // ---- Feature 4: keyboard shortcuts + help overlay ----
  let helpOverlay = document.getElementById("ux-help-overlay");
  let helpBtn = document.getElementById("ux-help-btn");
  let helpClose = document.getElementById("ux-help-close");
  let toggleHelp = (show?: boolean) => {
    if (!helpOverlay) { return; }
    let visible = helpOverlay.style.display !== "none";
    let next = show === undefined ? !visible : show;
    helpOverlay.style.display = next ? "flex" : "none";
  };
  if (helpBtn) { helpBtn.addEventListener("click", () => toggleHelp()); }
  if (helpClose) { helpClose.addEventListener("click", () => toggleHelp(false)); }

  document.addEventListener("keydown", (ev: KeyboardEvent) => {
    let target = ev.target as HTMLElement;
    let tag = target && target.tagName ? target.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "textarea" || tag === "select" ||
        (target && target.isContentEditable)) {
      return;  // Don't hijack typing.
    }
    if (ev.metaKey || ev.ctrlKey || ev.altKey) { return; }
    let key = ev.key;
    if (key === " " || key === "Spacebar") {
      ev.preventDefault();
      if (iter === 0) { simulationStarted(); }
      player.playOrPause();
    } else if (key === "s" || key === "S") {
      player.pause();
      if (iter === 0) { simulationStarted(); }
      oneStep();
    } else if (key === "r" || key === "R") {
      reset();
      userHasInteracted();
    } else if (key === "d" || key === "D") {
      generateData();
      parametersChanged = true;
    } else if (key === "?") {
      toggleHelp();
    } else if (key === "Escape") {
      toggleHelp(false);
    }
  });

  // Initialise the status bar once.
  uxTickStatus();
}

// ---- Feature 12: compact status bar (reads existing readouts) ----
let uxLastStatusTime = 0;
let uxLastStatusIter = 0;
let uxStepsPerSec = 0;
function uxTickStatus(): void {
  let now = (typeof performance !== "undefined" && performance.now) ?
      performance.now() : Date.now();
  if (uxLastStatusTime !== 0) {
    let dt = (now - uxLastStatusTime) / 1000;
    if (dt > 0) {
      let inst = (iter - uxLastStatusIter) / dt;
      // Exponential smoothing for a stable readout.
      uxStepsPerSec = uxStepsPerSec === 0 ? inst : uxStepsPerSec * 0.8 + inst * 0.2;
    }
  }
  uxLastStatusTime = now;
  uxLastStatusIter = iter;
  let set = (id: string, txt: string) => {
    let el = document.getElementById(id);
    if (el) { el.textContent = txt; }
  };
  set("ux-status-epoch", String(iter));
  set("ux-status-losstrain", isFinite(lossTrain) ? lossTrain.toFixed(3) : "NaN");
  set("ux-status-losstest", isFinite(lossTest) ? lossTest.toFixed(3) : "NaN");
  let at = document.getElementById("acc-train");
  let ate = document.getElementById("acc-test");
  set("ux-status-acctrain", at ? (at.textContent || "—") : "—");
  set("ux-status-acctest", ate ? (ate.textContent || "—") : "—");
  set("ux-status-sps", uxStepsPerSec ? uxStepsPerSec.toFixed(1) : "0");
}

// ===========================================================================
// Accessibility / i18n / performance / responsiveness features.
// ===========================================================================

/** Step counter consumed by the FPS / step-rate meter. */
let perfStepCount = 0;
function uxFrameAccountStep(): void {
  perfStepCount++;
}

const A11Y_PREFS_KEY = "nnpg-a11y-prefs";
function a11yLoadPrefs(): any {
  try {
    let raw = window.localStorage.getItem(A11Y_PREFS_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch (e) { return {}; }
}
function a11ySavePrefs(patch: any): void {
  try {
    let prefs = a11yLoadPrefs();
    for (let k in patch) { prefs[k] = patch[k]; }
    window.localStorage.setItem(A11Y_PREFS_KEY, JSON.stringify(prefs));
  } catch (e) { /* ignore */ }
}

// ---- Feature 1: i18n dictionary. Keys map to data-i18n attributes. ----
const I18N: {[lang: string]: {[key: string]: string}} = {
  en: {
    lang: "Language", highContrast: "High contrast", reducedMotion: "Reduce motion",
    compactMode: "Compact mode", uiScale: "UI scale", fastTraining: "Fast training",
    redrawEvery: "Redraw every", steps: "steps", stepsPerSec: "steps/s",
    data: "Data", features: "Features", output: "Output", epoch: "Epoch",
    learningRate: "Learning rate", activation: "Activation", optimizer: "Optimizer",
    regularization: "Regularization", problemType: "Problem type",
    testLoss: "Test loss", trainingLoss: "Training loss", trainAcc: "Train acc", testAcc: "Test acc"
  },
  es: {
    lang: "Idioma", highContrast: "Alto contraste", reducedMotion: "Reducir movimiento",
    compactMode: "Modo compacto", uiScale: "Escala de interfaz", fastTraining: "Entrenamiento rápido",
    redrawEvery: "Redibujar cada", steps: "pasos", stepsPerSec: "pasos/s",
    data: "Datos", features: "Características", output: "Salida", epoch: "Época",
    learningRate: "Tasa de aprendizaje", activation: "Activación", optimizer: "Optimizador",
    regularization: "Regularización", problemType: "Tipo de problema",
    testLoss: "Pérdida de prueba", trainingLoss: "Pérdida de entrenamiento", trainAcc: "Precisión entren.", testAcc: "Precisión prueba"
  },
  fr: {
    lang: "Langue", highContrast: "Contraste élevé", reducedMotion: "Réduire le mouvement",
    compactMode: "Mode compact", uiScale: "Échelle de l'interface", fastTraining: "Entraînement rapide",
    redrawEvery: "Redessiner tous les", steps: "pas", stepsPerSec: "pas/s",
    data: "Données", features: "Caractéristiques", output: "Sortie", epoch: "Époque",
    learningRate: "Taux d'apprentissage", activation: "Activation", optimizer: "Optimiseur",
    regularization: "Régularisation", problemType: "Type de problème",
    testLoss: "Perte de test", trainingLoss: "Perte d'entraînement", trainAcc: "Précision entr.", testAcc: "Précision test"
  },
  hi: {
    lang: "भाषा", highContrast: "उच्च कंट्रास्ट", reducedMotion: "गति घटाएँ",
    compactMode: "संक्षिप्त मोड", uiScale: "यूआई स्केल", fastTraining: "तेज़ प्रशिक्षण",
    redrawEvery: "हर बार फिर बनाएँ", steps: "चरण", stepsPerSec: "चरण/से",
    data: "डेटा", features: "विशेषताएँ", output: "आउटपुट", epoch: "युग",
    learningRate: "सीखने की दर", activation: "सक्रियण", optimizer: "ऑप्टिमाइज़र",
    regularization: "नियमितीकरण", problemType: "समस्या का प्रकार",
    testLoss: "परीक्षण हानि", trainingLoss: "प्रशिक्षण हानि", trainAcc: "प्रशिक्षण सटीकता", testAcc: "परीक्षण सटीकता"
  },
  zh: {
    lang: "语言", highContrast: "高对比度", reducedMotion: "减少动态效果",
    compactMode: "紧凑模式", uiScale: "界面缩放", fastTraining: "快速训练",
    redrawEvery: "每隔多少步重绘", steps: "步", stepsPerSec: "步/秒",
    data: "数据", features: "特征", output: "输出", epoch: "轮次",
    learningRate: "学习率", activation: "激活函数", optimizer: "优化器",
    regularization: "正则化", problemType: "问题类型",
    testLoss: "测试损失", trainingLoss: "训练损失", trainAcc: "训练准确率", testAcc: "测试准确率"
  }
};

function setLanguage(lang: string): void {
  let dict = I18N[lang] || I18N["en"];
  let nodes = document.querySelectorAll("[data-i18n]");
  Array.prototype.forEach.call(nodes, (el: HTMLElement) => {
    let key = el.getAttribute("data-i18n");
    if (key && dict[key] != null) { el.textContent = dict[key]; }
  });
  try { document.documentElement.setAttribute("lang", lang); } catch (e) {}
  a11ySavePrefs({language: lang});
}

function initA11yFeatures(): void {
  let prefs = a11yLoadPrefs();

  // ---- Feature 1: language selector ----
  let langSel = document.getElementById("a11y-language") as HTMLSelectElement;
  let initialLang = (prefs.language && I18N[prefs.language]) ? prefs.language : "en";
  if (langSel) {
    langSel.value = initialLang;
    langSel.addEventListener("change", () => setLanguage(langSel.value));
  }
  if (initialLang !== "en") { setLanguage(initialLang); }

  // ---- Feature 5: high-contrast theme ----
  let hcChk = document.getElementById("a11y-high-contrast") as HTMLInputElement;
  let applyHC = (on: boolean) => document.body.classList.toggle("ux-high-contrast", on);
  if (hcChk) {
    hcChk.checked = !!prefs.highContrast;
    applyHC(hcChk.checked);
    hcChk.addEventListener("change", () => {
      applyHC(hcChk.checked);
      a11ySavePrefs({highContrast: hcChk.checked});
    });
  }

  // ---- Feature 4: reduced motion (manual toggle + OS preference) ----
  let rmChk = document.getElementById("a11y-reduced-motion") as HTMLInputElement;
  let osReduce = false;
  try {
    osReduce = !!(window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  } catch (e) {}
  let applyRM = (on: boolean) => document.body.classList.toggle("ux-reduced-motion", on);
  if (rmChk) {
    rmChk.checked = prefs.reducedMotion != null ? !!prefs.reducedMotion : osReduce;
    applyRM(rmChk.checked);
    rmChk.addEventListener("change", () => {
      applyRM(rmChk.checked);
      a11ySavePrefs({reducedMotion: rmChk.checked});
    });
  }

  // ---- Feature 11: compact mode ----
  let cmChk = document.getElementById("a11y-compact") as HTMLInputElement;
  let applyCM = (on: boolean) => document.body.classList.toggle("ux-compact", on);
  if (cmChk) {
    cmChk.checked = !!prefs.compact;
    applyCM(cmChk.checked);
    cmChk.addEventListener("change", () => {
      applyCM(cmChk.checked);
      a11ySavePrefs({compact: cmChk.checked});
    });
  }

  // ---- Feature 6: UI scale / font size ----
  let scaleSlider = document.getElementById("a11y-ui-scale") as HTMLInputElement;
  let scaleVal = document.getElementById("a11y-ui-scale-val");
  let applyScale = (pct: number) => {
    document.documentElement.style.fontSize = (16 * pct / 100) + "px";
    if (scaleVal) { scaleVal.textContent = String(pct); }
  };
  if (scaleSlider) {
    let p = typeof prefs.uiScale === "number" ? prefs.uiScale : 100;
    scaleSlider.value = String(p);
    if (p !== 100) { applyScale(p); } else if (scaleVal) { scaleVal.textContent = "100"; }
    scaleSlider.addEventListener("input", () => {
      let v = Math.max(80, Math.min(150, +scaleSlider.value || 100));
      applyScale(v);
      a11ySavePrefs({uiScale: v});
    });
  }

  // ---- Feature 8: fast training toggle ----
  let workerChk = document.getElementById("perf-worker") as HTMLInputElement;
  if (workerChk) {
    workerChk.checked = !!prefs.fastTraining;
    uxFastTraining = workerChk.checked;
    workerChk.addEventListener("change", () => {
      uxFastTraining = workerChk.checked;
      a11ySavePrefs({fastTraining: workerChk.checked});
    });
  }

  // ---- Feature 10: throttled redraw (redraw every K steps) ----
  let kInput = document.getElementById("perf-redraw-k") as HTMLInputElement;
  if (kInput) {
    let k = typeof prefs.redrawEvery === "number" ? prefs.redrawEvery : 1;
    kInput.value = String(k);
    uxRedrawEvery = Math.max(1, k | 0);
    kInput.addEventListener("change", () => {
      uxRedrawEvery = Math.max(1, parseInt(kInput.value, 10) || 1);
      a11ySavePrefs({redrawEvery: uxRedrawEvery});
    });
  }

  // ---- Feature 3: keyboard navigation for dataset thumbnails ----
  Array.prototype.forEach.call(
      document.querySelectorAll(".dataset"), (el: HTMLElement) => {
    if (!el.hasAttribute("tabindex")) { el.setAttribute("tabindex", "0"); }
    el.setAttribute("role", "button");
    let title = el.getAttribute("title");
    if (title && !el.hasAttribute("aria-label")) {
      el.setAttribute("aria-label", "Dataset: " + title);
    }
    el.addEventListener("keydown", (ev: KeyboardEvent) => {
      if (ev.key === "Enter" || ev.key === " " || ev.key === "Spacebar") {
        ev.preventDefault();
        let canvas = el.querySelector("canvas") as HTMLElement;
        if (canvas) { canvas.dispatchEvent(new MouseEvent("click", {bubbles: true})); }
      }
    });
  });

  // ---- Feature 2/3: focus + redraw the boundary when training stops ----
  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
    let btn = document.getElementById("play-pause-button");
    if (btn) { btn.setAttribute("aria-pressed", String(isPlaying)); }
    // When training stops while redraw was throttled, force a final full redraw
    // so the decision boundary is up to date.
    if (!isPlaying && network && uxRedrawEvery > 1) {
      try { updateUI(); } catch (e) { /* ignore */ }
    }
  });

  // ---- Feature 9: FPS / step-rate meter + frame-budget indicator ----
  let spsEl = document.getElementById("perf-sps");
  let fpsEl = document.getElementById("perf-fps");
  let budgetEl = document.getElementById("perf-budget");
  let lastT = (typeof performance !== "undefined" && performance.now)
      ? performance.now() : Date.now();
  let lastSteps = 0;
  let frames = 0;
  let smoothSps = 0;
  let tickMeter = () => {
    frames++;
    let now = (typeof performance !== "undefined" && performance.now)
        ? performance.now() : Date.now();
    let dt = now - lastT;
    if (dt >= 500) {
      let sps = (perfStepCount - lastSteps) * 1000 / dt;
      smoothSps = smoothSps === 0 ? sps : smoothSps * 0.7 + sps * 0.3;
      let fps = frames * 1000 / dt;
      if (spsEl) { spsEl.textContent = smoothSps.toFixed(0); }
      if (fpsEl) { fpsEl.textContent = fps.toFixed(0); }
      if (budgetEl) {
        // Frame budget: green if we comfortably hit ~60fps, else warn/bad.
        let cls = fps >= 45 ? "good" : (fps >= 25 ? "warn" : "bad");
        budgetEl.className = cls;
        budgetEl.textContent = "●";
      }
      lastT = now;
      lastSteps = perfStepCount;
      frames = 0;
    }
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(tickMeter);
    }
  };
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(tickMeter);
  }
}

initTrainingMethodologyGUI();
initUXFeatures();
initA11yFeatures();
initExperimentsGUI();
