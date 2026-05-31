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

import * as d3 from 'd3';

/**
 * A two dimensional example: x and y coordinates with the label.
 */
export type Example2D = {
  x: number,
  y: number,
  label: number
};

type Point = {
  x: number,
  y: number
};

/**
 * Shuffles the array using Fisher-Yates algorithm. Uses the seedrandom
 * library as the random generator.
 */
export function shuffle(array: any[]): void {
  let counter = array.length;
  let temp = 0;
  let index = 0;
  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = Math.floor(Math.random() * counter);
    // Decrease counter by 1
    counter--;
    // And swap the last element with it
    temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

/**
 * Generates a hash-like dataset for classification.
 */
export function classifyHashData(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let gridSize = 3; // Define the size of the grid

  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-5, 5);
    let y = randUniform(-5, 5);
    let noiseX = randUniform(-1, 1) * noise;
    let noiseY = randUniform(-1, 1) * noise;
    let label = ((Math.floor((x + noiseX) / gridSize) + Math.floor((y + noiseY) / gridSize)) % 2 === 0) ? 1 : -1;
    points.push({x, y, label});
  }
  return points;
}

export type DataGenerator = (numSamples: number, noise: number) => Example2D[];

export function classifyTwoGaussData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];

  let varianceScale = d3.scaleLinear().domain([0, .5]).range([0.5, 4]);
  let variance = varianceScale(noise);

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      let x = normalRandom(cx, variance);
      let y = normalRandom(cy, variance);
      points.push({x, y, label});
    }
  }

  genGauss(2, 2, 1); // Gaussian with positive examples.
  genGauss(-2, -2, -1); // Gaussian with negative examples.
  return points;
}

export function regressArgMax(numSamples: number, noise: number):
  Example2D[] {
  let radius = 6;
  let labelScale = d3.scaleLinear()
    .domain([-10, 10])
    .range([-1, 1]);
  let getLabel = (x, y) => Math.max(x,y)==x?0:1;

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = randUniform(-radius, radius);
    let label = getLabel(x , y);
    points.push({x, y, label});
  }
  return points;
}

export function regressMaximum(numSamples: number, noise: number):
  Example2D[] {
  let radius = 6;
  let labelScale = d3.scaleLinear()
    .domain([-10, 10])
    .range([-1, 1]); 
  let getLabel = (x, y) => Math.max(x,y);

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = randUniform(-radius, radius);
    let label = getLabel(x , y);
    points.push({x, y, label});
  }
  return points;
}

export function regressPlane(numSamples: number, noise: number):
  Example2D[] {
  let radius = 6;
  let labelScale = d3.scaleLinear()
    .domain([-10, 10])
    .range([-1, 1]);
  let getLabel = (x, y) => labelScale(x + y);

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = randUniform(-radius, radius);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getLabel(x + noiseX, y + noiseY);
    points.push({x, y, label});
  }
  return points;
}

export function regressGaussian(numSamples: number, noise: number):
  Example2D[] {
  let points: Example2D[] = [];

  let labelScale = d3.scaleLinear()
    .domain([0, 2])
    .range([1, 0])
    .clamp(true);

  let gaussians = [
    [-4, 2.5, 1],
    [0, 2.5, -1],
    [4, 2.5, 1],
    [-4, -2.5, -1],
    [0, -2.5, 1],
    [4, -2.5, -1]
  ];

  function getLabel(x, y) {
    // Choose the one that is maximum in abs value.
    let label = 0;
    gaussians.forEach(([cx, cy, sign]) => {
      let newLabel = sign * labelScale(dist({x, y}, {x: cx, y: cy}));
      if (Math.abs(newLabel) > Math.abs(label)) {
        label = newLabel;
      }
    });
    return label;
  }
  let radius = 6;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = randUniform(-radius, radius);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getLabel(x + noiseX, y + noiseY);
    points.push({x,y, label});
  };
  return points;
}

export function classifySpiralData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;

  function genSpiral(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      let r = i / n * 5;
      let t = 1.75 * i / n * 2 * Math.PI + deltaT;
      let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      let y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push({x, y, label});
    }
  }

  genSpiral(0, 1); // Positive examples.
  genSpiral(Math.PI, -1); // Negative examples.
  return points;
}

export function classifyCircleData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];
  let radius = 5;
  function getCircleLabel(p: Point, center: Point) {
    return (dist(p, center) < (radius * 0.6)) ? 1 : -1;
  }

  // Generate positive points inside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = radius*0.5;//randUniform(0, radius * 0.5);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }

  // Generate negative points outside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = radius*0.7//randUniform(radius * 0.7, radius);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }
  return points;
}

export function classifyXORData(numSamples: number, noise: number):
    Example2D[] {
  function getXORLabel(p: Point) { return p.x * p.y >= 0 ? 1 : -1; }

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-5, 5);
    let padding = 0.3;
    x += x > 0 ? padding : -padding;  // Padding.
    let y = randUniform(-5, 5);
    y += y > 0 ? padding : -padding;
    let noiseX = randUniform(-5, 5) * noise;
    let noiseY = randUniform(-5, 5) * noise;
    let label = getXORLabel({x: x + noiseX, y: y + noiseY});
    points.push({x, y, label});
  }
  return points;
}

export function classifyMNISTThreeData(numSamples: number, noise: number):
    Example2D[] {
  let three = [[0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 197, 234, 234, 234, 234, 234, 234, 196, 197, 219, 97, 97, 97, 13, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 213, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 218, 179, 64, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 93, 210, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 243, 177, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 18, 82, 82, 82, 82, 82, 82, 82, 171, 219, 219, 233, 253, 253, 253, 235, 16, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 111, 253, 253, 253, 253, 124, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 130, 247, 253, 253, 253, 242, 123, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 111, 200, 248, 253, 253, 253, 253, 253, 173, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 203, 234, 253, 253, 253, 253, 253, 253, 253, 167, 10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 227, 253, 253, 253, 253, 253, 253, 193, 77, 14, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 201, 253, 253, 253, 253, 253, 177, 9, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 89, 189, 229, 253, 253, 161, 19, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 71, 144, 142, 33, 0, 0, 0, 0, 0, 0, 0, 28, 253, 253, 253, 93, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 254, 253, 177, 0, 0, 0, 0, 0, 0, 0, 0, 28, 253, 253, 253, 34, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 198, 253, 208, 81, 0, 0, 0, 0, 0, 0, 0, 17, 209, 253, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 192, 253, 253, 246, 201, 83, 83, 47, 75, 83, 83, 172, 239, 253, 253, 52, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 25, 204, 253, 253, 253, 253, 253, 230, 248, 253, 253, 253, 253, 253, 253, 34, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 29, 75, 179, 212, 236, 253, 253, 253, 253, 253, 253, 253, 253, 236, 28, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 96, 229, 232, 232, 232, 232, 232, 165, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
  function getMNISTLabel(p: Point) { return three[p.y][p.x] > 0 ? 1 : -1; }
  
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = Math.floor(randUniform(0, 28));
    let y = Math.floor(randUniform(0, 28));
    let noiseX = 0//randUniform(-5, 5) * noise;
    let noiseY = 0//randUniform(-5, 5) * noise;
    let label = getMNISTLabel({x: x + noiseX, y: y + noiseY});
    x = x * 12 / 28 - 6;
    y = y * 12 / 28 - 6;
    points.push({x, y, label});
  }
  return points;
}

/**
 * Generates a concentric circles dataset for classification.
 */
export function classifyConcentricCircles(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;

  function genCircle(radius: number, label: number) {
    for (let i = 0; i < n; i++) {
      let angle = randUniform(0, 2 * Math.PI);
      let x = radius * Math.cos(angle) + randUniform(-1, 1) * noise;
      let y = radius * Math.sin(angle) + randUniform(-1, 1) * noise;
      points.push({x, y, label});
    }
  }

  genCircle(2, 1);  // Inner circle
  genCircle(4, -1); // Outer circle
  return points;
}

/**
 * Generates a sine wave dataset for regression.
 */
export function regressSineWave(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let frequency = 0.5;
  let amplitude = 1;

  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-10, 10);
    let y = amplitude * Math.sin(frequency * x) + randUniform(-1, 1) * noise;
    points.push({x, y, label: y});
  }
  return points;
}

/**
 * Generates a biclusters dataset for classification.
 */
export function classifyBiclusters(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;

  for (let i = 0; i < n; i++) {
    let x = randUniform(-5, 5);
    let y = x + randUniform(-1, 1) * noise;
    points.push({x, y, label: 1});
  }

  for (let i = 0; i < n; i++) {
    let x = randUniform(-5, 5);
    let y = -x + randUniform(-1, 1) * noise;
    points.push({x, y, label: -1});
  }

  return points;
}

/**
 * Generates a moons dataset for classification.
 */
export function classifyMoons(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;

  for (let i = 0; i < n; i++) {
    let angle = Math.PI * i / n;
    let x = Math.cos(angle) + randUniform(-1, 1) * noise;
    let y = Math.sin(angle) + randUniform(-1, 1) * noise;
    points.push({x, y, label: 1});
  }

  for (let i = 0; i < n; i++) {
    let angle = Math.PI * i / n;
    let x = 1 - Math.cos(angle) + randUniform(-1, 1) * noise;
    let y = 1 - Math.sin(angle) - 0.5 + randUniform(-1, 1) * noise;
    points.push({x, y, label: -1});
  }

  return points;
}

/**
 * Generates a Friedman 1 dataset for regression.
 */
export function regressFriedman1(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x1 = randUniform(0, 1);
    let x2 = randUniform(0, 1);
    let x3 = randUniform(0, 1);
    let x4 = randUniform(0, 1);
    let x5 = randUniform(0, 1);
    let y = 10 * Math.sin(Math.PI * x1 * x2) + 20 * Math.pow(x3 - 0.5, 2) + 10 * x4 + 5 * x5 + randUniform(-1, 1) * noise;
    points.push({x: x1, y: y, label: y});
  }
  return points;
}

/**
 * Generates a Friedman 2 dataset for regression.
 */
export function regressFriedman2(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x1 = randUniform(0, 100);
    let x2 = randUniform(40 * Math.PI, 560 * Math.PI);
    let x3 = randUniform(0, 1);
    let x4 = randUniform(1, 11);
    let y = Math.sqrt(x1 * x1 + (x2 * x3 - 1 / (x2 * x4)) ** 2) + randUniform(-1, 1) * noise;
    points.push({x: x1, y: y, label: y});
  }
  return points;
}

/**
 * Generates a Friedman 3 dataset for regression.
 */
export function regressFriedman3(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x1 = randUniform(0, 100);
    let x2 = randUniform(40 * Math.PI, 560 * Math.PI);
    let x3 = randUniform(0, 1);
    let x4 = randUniform(1, 11);
    let y = Math.atan((x2 * x3 - 1 / (x2 * x4)) / x1) + randUniform(-1, 1) * noise;
    points.push({x: x1, y: y, label: y});
  }
  return points;
}

/**
 * Checkerboard: NxN alternating grid of squares.
 */
export function classifyCheckerboard(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let cells = 4;
  let size = 12 / cells;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let cx = Math.floor((x + 6 + nx) / size);
    let cy = Math.floor((y + 6 + ny) / size);
    let label = (cx + cy) % 2 === 0 ? 1 : -1;
    points.push({x, y, label});
  }
  return points;
}

/**
 * Four-quadrant XOR blobs: a gaussian blob in each quadrant, XOR labelled.
 */
export function classifyQuadrantBlobs(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let variance = 0.5 + noise * 2;
  let centers = [
    [3, 3, 1], [-3, -3, 1], [3, -3, -1], [-3, 3, -1]
  ];
  centers.forEach(([cx, cy, label]) => {
    for (let i = 0; i < numSamples / 4; i++) {
      let x = normalRandom(cx, variance);
      let y = normalRandom(cy, variance);
      points.push({x, y, label});
    }
  });
  return points;
}

/**
 * Three (or more) concentric rings with alternating labels.
 */
export function classifyConcentricRings(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let radii = [1.5, 3, 4.5];
  let per = numSamples / radii.length;
  radii.forEach((radius, idx) => {
    let label = idx % 2 === 0 ? 1 : -1;
    for (let i = 0; i < per; i++) {
      let angle = randUniform(0, 2 * Math.PI);
      let x = radius * Math.cos(angle) + randUniform(-1, 1) * noise;
      let y = radius * Math.sin(angle) + randUniform(-1, 1) * noise;
      points.push({x, y, label});
    }
  });
  return points;
}

/**
 * Generic multi-arm spiral generator.
 */
function genMultiSpiral(numSamples: number, noise: number, arms: number): Example2D[] {
  let points: Example2D[] = [];
  let per = numSamples / arms;
  for (let a = 0; a < arms; a++) {
    let deltaT = (a / arms) * 2 * Math.PI;
    let label = a % 2 === 0 ? 1 : -1;
    for (let i = 0; i < per; i++) {
      let r = i / per * 5;
      let t = 1.75 * i / per * 2 * Math.PI + deltaT;
      let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      let y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push({x, y, label});
    }
  }
  return points;
}

/** Spiral with three arms. */
export function classifyThreeArmSpiral(numSamples: number, noise: number): Example2D[] {
  return genMultiSpiral(numSamples, noise, 3);
}

/** Spiral with four arms. */
export function classifyFourArmSpiral(numSamples: number, noise: number): Example2D[] {
  return genMultiSpiral(numSamples, noise, 4);
}

/** Two tightly-wound interleaved spirals. */
export function classifyTightSpiral(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;
  function genSpiral(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      let r = i / n * 5;
      let t = 3.0 * i / n * 2 * Math.PI + deltaT;
      let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      let y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push({x, y, label});
    }
  }
  genSpiral(0, 1);
  genSpiral(Math.PI, -1);
  return points;
}

/** Clean two-moons (low base noise). */
export function classifyCleanMoons(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;
  for (let i = 0; i < n; i++) {
    let angle = Math.PI * i / n;
    let x = 3 * Math.cos(angle) + randUniform(-1, 1) * noise;
    let y = 3 * Math.sin(angle) + randUniform(-1, 1) * noise;
    points.push({x, y, label: 1});
  }
  for (let i = 0; i < n; i++) {
    let angle = Math.PI * i / n;
    let x = 3 - 3 * Math.cos(angle) + randUniform(-1, 1) * noise;
    let y = -3 * Math.sin(angle) + 1.5 + randUniform(-1, 1) * noise;
    points.push({x, y, label: -1});
  }
  return points;
}

/** Two nested U / horseshoe shapes. */
export function classifyNestedU(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;
  function genU(radius: number, label: number, yShift: number) {
    for (let i = 0; i < n; i++) {
      let angle = Math.PI + Math.PI * i / n; // bottom half
      let x = radius * Math.cos(angle) + randUniform(-1, 1) * noise;
      let y = radius * Math.sin(angle) + yShift + randUniform(-1, 1) * noise;
      points.push({x, y, label});
    }
  }
  genU(2.5, 1, 1);
  genU(4.5, -1, 1);
  return points;
}

/** K gaussian blobs in a row with alternating labels. */
export function classifyGaussianMixture(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let k = 6;
  let variance = 0.3 + noise;
  let per = numSamples / k;
  for (let j = 0; j < k; j++) {
    let cx = -5 + (10 / (k - 1)) * j;
    let cy = (j % 2 === 0 ? 2.5 : -2.5);
    let label = j % 2 === 0 ? 1 : -1;
    for (let i = 0; i < per; i++) {
      points.push({
        x: normalRandom(cx, variance),
        y: normalRandom(cy, variance),
        label
      });
    }
  }
  return points;
}

/** Diagonal stripes pattern. */
export function classifyDiagonalStripes(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let width = 2;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let label = Math.floor((x + y + 12 + nx) / width) % 2 === 0 ? 1 : -1;
    points.push({x, y, label});
  }
  return points;
}

/** Label by whether y is above or below a sine boundary. */
export function classifySineBoundary(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let ny = randUniform(-1, 1) * noise;
    let boundary = 3 * Math.sin(x);
    let label = (y + ny) > boundary ? 1 : -1;
    points.push({x, y, label});
  }
  return points;
}

/** Circle inscribed in a square: inside circle vs corners. */
export function classifyCircleInSquare(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let radius = 3.5;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let label = dist({x: x + nx, y: y + ny}, {x: 0, y: 0}) < radius ? 1 : -1;
    points.push({x, y, label});
  }
  return points;
}

/** Plus-sign / cross shape. */
export function classifyCross(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let arm = 2;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let inCross = Math.abs(x + nx) < arm || Math.abs(y + ny) < arm;
    points.push({x, y, label: inCross ? 1 : -1});
  }
  return points;
}

/** S-curve boundary classification. */
export function classifySCurve(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    // S-shaped boundary x = f(y)
    let boundary = 3 * Math.tanh(y);
    let label = (x + nx) > boundary ? 1 : -1;
    points.push({x, y, label});
  }
  return points;
}

/** Pinwheel: rotating blobs spreading from center. */
export function classifyPinwheel(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let blades = 5;
  let per = numSamples / blades;
  for (let b = 0; b < blades; b++) {
    let label = b % 2 === 0 ? 1 : -1;
    let baseAngle = (b / blades) * 2 * Math.PI;
    for (let i = 0; i < per; i++) {
      let r = randUniform(0.5, 5);
      let angle = baseAngle + r * 0.4 + normalRandom(0, 0.05 + noise * 0.1);
      let x = r * Math.cos(angle);
      let y = r * Math.sin(angle);
      points.push({x, y, label});
    }
  }
  return points;
}

/** Island clusters: many small blobs with random labels grouped by region. */
export function classifyIslands(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let islands = [
    [-4, -4, 1], [-4, 4, -1], [4, -4, -1], [4, 4, 1],
    [0, 0, 1], [-4, 0, -1], [4, 0, -1], [0, 4, 1], [0, -4, 1]
  ];
  let variance = 0.3 + noise;
  let per = numSamples / islands.length;
  islands.forEach(([cx, cy, label]) => {
    for (let i = 0; i < per; i++) {
      points.push({
        x: normalRandom(cx, variance),
        y: normalRandom(cy, variance),
        label
      });
    }
  });
  return points;
}

/** Ring vs center: a central blob surrounded by a ring. */
export function classifyRingVsCenter(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let half = numSamples / 2;
  for (let i = 0; i < half; i++) {
    points.push({
      x: normalRandom(0, 0.6 + noise),
      y: normalRandom(0, 0.6 + noise),
      label: 1
    });
  }
  for (let i = 0; i < half; i++) {
    let angle = randUniform(0, 2 * Math.PI);
    let r = 4.5;
    points.push({
      x: r * Math.cos(angle) + randUniform(-1, 1) * noise,
      y: r * Math.sin(angle) + randUniform(-1, 1) * noise,
      label: -1
    });
  }
  return points;
}

/** Gaussian quantiles: label by radial distance band from a single gaussian. */
export function classifyGaussianQuantiles(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = normalRandom(0, 2);
    let y = normalRandom(0, 2);
    let r = dist({x, y}, {x: 0, y: 0});
    let nx = randUniform(-1, 1) * noise;
    let label = (r + nx) < 2.2 ? 1 : -1;
    points.push({x, y, label});
  }
  return points;
}

/** Anisotropic (stretched/rotated) blobs. */
export function classifyAnisotropicBlobs(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let half = numSamples / 2;
  let theta = Math.PI / 6;
  function genBlob(cx: number, cy: number, label: number) {
    for (let i = 0; i < half; i++) {
      let lx = normalRandom(0, 3 + noise);
      let ly = normalRandom(0, 0.5 + noise);
      let x = cx + lx * Math.cos(theta) - ly * Math.sin(theta);
      let y = cy + lx * Math.sin(theta) + ly * Math.cos(theta);
      points.push({x, y, label});
    }
  }
  genBlob(-1.5, -1.5, 1);
  genBlob(1.5, 1.5, -1);
  return points;
}

/** Random-label blobs: blobs at random positions with random labels. */
export function classifyRandomLabelBlobs(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let numBlobs = 8;
  let per = numSamples / numBlobs;
  let variance = 0.4 + noise;
  for (let b = 0; b < numBlobs; b++) {
    let cx = randUniform(-4.5, 4.5);
    let cy = randUniform(-4.5, 4.5);
    let label = Math.random() < 0.5 ? 1 : -1;
    for (let i = 0; i < per; i++) {
      points.push({
        x: normalRandom(cx, variance),
        y: normalRandom(cy, variance),
        label
      });
    }
  }
  return points;
}

/** Regression: ripple surface sin(r)/r. */
export function regressRipple(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let r = Math.sqrt(x * x + y * y) + 1e-6;
    let label = Math.sin(r * 1.5) / r * 4 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: saddle x*y. */
export function regressSaddle(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = (x * y) / 9 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: single gaussian bump. */
export function regressGaussianBump(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let r2 = x * x + y * y;
    let label = Math.exp(-r2 / 8) + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: staircase surface. */
export function regressStaircase(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = (Math.floor(x / 2) + Math.floor(y / 2)) * 0.4 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Target with 4 concentric rings, alternating labels. */
export function classifyTargetRings(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let radii = [1, 2.25, 3.5, 4.75];
  let per = numSamples / radii.length;
  radii.forEach((radius, idx) => {
    let label = idx % 2 === 0 ? 1 : -1;
    for (let i = 0; i < per; i++) {
      let angle = randUniform(0, 2 * Math.PI);
      let x = radius * Math.cos(angle) + randUniform(-1, 1) * noise;
      let y = radius * Math.sin(angle) + randUniform(-1, 1) * noise;
      points.push({x, y, label});
    }
  });
  return points;
}

/** Spiral galaxy: two logarithmic arms of blobs. */
export function classifySpiralGalaxy(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;
  function genArm(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      let r = 0.5 + i / n * 4.5;
      let t = 2.5 * Math.log(r + 1) * Math.PI + deltaT;
      let x = r * Math.cos(t) + normalRandom(0, 0.15 + noise * 0.2);
      let y = r * Math.sin(t) + normalRandom(0, 0.15 + noise * 0.2);
      points.push({x, y, label});
    }
  }
  genArm(0, 1);
  genArm(Math.PI, -1);
  return points;
}

/** Yin-yang: classic two-teardrop pattern. */
export function classifyYinYang(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let R = 5;
  function label(x: number, y: number): number {
    // Top half base = 1, bottom = -1, swap inside the two small circles.
    let base = y >= 0 ? 1 : -1;
    // Two small circles centered at (0, R/2) and (0, -R/2), radius R/2.
    if (dist({x, y}, {x: 0, y: R / 2}) < R / 2) base = -1;
    if (dist({x, y}, {x: 0, y: -R / 2}) < R / 2) base = 1;
    // Tiny dots.
    if (dist({x, y}, {x: 0, y: R / 2}) < R / 8) base = 1;
    if (dist({x, y}, {x: 0, y: -R / 2}) < R / 8) base = -1;
    return base;
  }
  for (let i = 0; i < numSamples; i++) {
    let angle = randUniform(0, 2 * Math.PI);
    let r = Math.sqrt(Math.random()) * R;
    let x = r * Math.cos(angle);
    let y = r * Math.sin(angle);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    points.push({x, y, label: label(x + nx, y + ny)});
  }
  return points;
}

/** Smiley face: face region vs eyes/mouth features. */
export function classifySmiley(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  function label(x: number, y: number): number {
    if (dist({x, y}, {x: 0, y: 0}) > 5) return -1;
    // Eyes.
    if (dist({x, y}, {x: -1.8, y: 1.8}) < 0.8) return -1;
    if (dist({x, y}, {x: 1.8, y: 1.8}) < 0.8) return -1;
    // Mouth: arc band.
    let r = dist({x, y}, {x: 0, y: 0});
    if (y < -0.5 && r > 2.2 && r < 3.2) return -1;
    return 1;
  }
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    points.push({x, y, label: label(x + nx, y + ny)});
  }
  return points;
}

/** NxN grid of gaussian blobs with checkerboard labels. */
export function classifyGridBlobs(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let n = 4;
  let variance = 0.15 + noise * 0.5;
  let centers: number[][] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let cx = -4.5 + (9 / (n - 1)) * i;
      let cy = -4.5 + (9 / (n - 1)) * j;
      centers.push([cx, cy, (i + j) % 2 === 0 ? 1 : -1]);
    }
  }
  let per = numSamples / centers.length;
  centers.forEach(([cx, cy, lbl]) => {
    for (let i = 0; i < per; i++) {
      points.push({x: normalRandom(cx, variance), y: normalRandom(cy, variance), label: lbl});
    }
  });
  return points;
}

/** Interleaving sine waves. */
export function classifyInterleavingWaves(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;
  function genWave(shift: number, label: number) {
    for (let i = 0; i < n; i++) {
      let x = randUniform(-6, 6);
      let y = 2 * Math.sin(x) + shift + randUniform(-1, 1) * (0.4 + noise);
      points.push({x, y, label});
    }
  }
  genWave(1.8, 1);
  genWave(-1.8, -1);
  return points;
}

/** Blob inside a ring. */
export function classifyBlobInRing(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let half = numSamples / 2;
  for (let i = 0; i < half; i++) {
    points.push({x: normalRandom(0, 0.8 + noise), y: normalRandom(0, 0.8 + noise), label: 1});
  }
  for (let i = 0; i < half; i++) {
    let angle = randUniform(0, 2 * Math.PI);
    let r = 4.5 + randUniform(-1, 1) * noise;
    points.push({x: r * Math.cos(angle), y: r * Math.sin(angle), label: -1});
  }
  return points;
}

/** Triangle region vs circle region. */
export function classifyTriangleVsCircle(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  function inTriangle(x: number, y: number): boolean {
    // Triangle on the left half pointing up.
    let cx = -2.5;
    return y > -3 && y < 3 && Math.abs(x - cx) < (3 - y) * 0.7;
  }
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let inCircle = dist({x: x + nx - 2.5, y: y + ny}, {x: 0, y: 0}) < 2.5;
    let label = (inTriangle(x + nx, y + ny) || inCircle) ? 1 : -1;
    points.push({x, y, label});
  }
  return points;
}

/** Gaussian cross: two overlapping anisotropic gaussians forming a cross. */
export function classifyGaussianCross(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let half = numSamples / 2;
  for (let i = 0; i < half; i++) {
    points.push({x: normalRandom(0, 3.5 + noise), y: normalRandom(0, 0.5 + noise * 0.5), label: 1});
  }
  for (let i = 0; i < half; i++) {
    points.push({x: normalRandom(0, 0.5 + noise * 0.5), y: normalRandom(0, 3.5 + noise), label: -1});
  }
  return points;
}

/** Noisy 4-quadrant XOR (region based). */
export function classifyNoisyXor4(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-5, 5);
    let y = randUniform(-5, 5);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let label = ((x + nx) >= 0) === ((y + ny) >= 0) ? 1 : -1;
    points.push({x, y, label});
  }
  return points;
}

/** Crescent pair: two facing crescents. */
export function classifyCrescentPair(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;
  for (let i = 0; i < n; i++) {
    let angle = Math.PI * (i / n) - Math.PI / 2;
    let x = 3 * Math.cos(angle) - 1.5 + randUniform(-1, 1) * noise;
    let y = 3 * Math.sin(angle) + randUniform(-1, 1) * noise;
    points.push({x, y, label: 1});
  }
  for (let i = 0; i < n; i++) {
    let angle = Math.PI * (i / n) + Math.PI / 2;
    let x = 3 * Math.cos(angle) + 1.5 + randUniform(-1, 1) * noise;
    let y = 3 * Math.sin(angle) + randUniform(-1, 1) * noise;
    points.push({x, y, label: -1});
  }
  return points;
}

/** Dartboard: concentric rings sliced into angular sectors. */
export function classifyDartboard(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let sectors = 8;
  for (let i = 0; i < numSamples; i++) {
    let angle = randUniform(0, 2 * Math.PI);
    let r = Math.sqrt(Math.random()) * 5;
    let x = r * Math.cos(angle) + randUniform(-1, 1) * noise;
    let y = r * Math.sin(angle) + randUniform(-1, 1) * noise;
    let ang = Math.atan2(y, x) + Math.PI;
    let sector = Math.floor(ang / (2 * Math.PI) * sectors);
    let ring = Math.floor(dist({x, y}, {x: 0, y: 0}) / (5 / 3));
    let label = (sector + ring) % 2 === 0 ? 1 : -1;
    points.push({x, y, label});
  }
  return points;
}

/** Comb: vertical stripes. */
export function classifyComb(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let width = 1.5;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let label = Math.floor((x + 6 + nx) / width) % 2 === 0 ? 1 : -1;
    points.push({x, y, label});
  }
  return points;
}

/** Diagonal checker: fine diagonal checkerboard. */
export function classifyDiagonalChecker(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let size = 2;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let u = (x + nx) + (y + ny);
    let v = (x + nx) - (y + ny);
    let label = (Math.floor(u / size) + Math.floor(v / size)) % 2 === 0 ? 1 : -1;
    points.push({x, y, label});
  }
  return points;
}

/** Cluster chain: a chain of alternating blobs along a sine path. */
export function classifyClusterChain(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let k = 7;
  let variance = 0.25 + noise * 0.5;
  let per = numSamples / k;
  for (let j = 0; j < k; j++) {
    let cx = -5 + (10 / (k - 1)) * j;
    let cy = 3 * Math.sin(cx);
    let label = j % 2 === 0 ? 1 : -1;
    for (let i = 0; i < per; i++) {
      points.push({x: normalRandom(cx, variance), y: normalRandom(cy, variance), label});
    }
  }
  return points;
}

/** Regression: sin(x)*cos(y). */
export function regressSinCos(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = Math.sin(x) * Math.cos(y) + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: radial distance f = r. */
export function regressRadial(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = Math.sqrt(x * x + y * y) / 2 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: f = |x| + |y|. */
export function regressAbs(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = (Math.abs(x) + Math.abs(y)) / 3 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: step inside a circle. */
export function regressStepCircle(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = (dist({x, y}, {x: 0, y: 0}) < 3 ? 1 : -1) + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: f = sin(x) + sin(y). */
export function regressWaves(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = (Math.sin(x) + Math.sin(y)) / 2 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Nested squares: inside small square=1, ring between=-1, outside=1. */
export function classifyNestedSquares(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let m = Math.max(Math.abs(x + nx), Math.abs(y + ny));
    let band = Math.floor(m / 2);
    points.push({x, y, label: band % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Polygon boundary: inside a regular pentagon vs outside. */
export function classifyPolygonBoundary(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let sides = 5;
  let R = 4;
  function inPoly(x: number, y: number): boolean {
    let ang = Math.atan2(y, x);
    let r = Math.sqrt(x * x + y * y);
    let sector = 2 * Math.PI / sides;
    let a = ((ang % sector) + sector) % sector - sector / 2;
    let bound = R * Math.cos(sector / 2) / Math.cos(a);
    return r < bound;
  }
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    points.push({x, y, label: inPoly(x + nx, y + ny) ? 1 : -1});
  }
  return points;
}

/** Voronoi regions: label by parity of nearest seed. */
export function classifyVoronoiRegions(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let seeds = [
    [-4, -3, 1], [3, -4, -1], [4, 3, 1], [-3, 4, -1],
    [0, 0, 1], [-5, 1, -1], [2, 1, -1], [1, -2, 1]
  ];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let best = Infinity;
    let label = 1;
    seeds.forEach(([sx, sy, l]) => {
      let d = dist({x: x + nx, y: y + ny}, {x: sx, y: sy});
      if (d < best) { best = d; label = l; }
    });
    points.push({x, y, label});
  }
  return points;
}

/** 3x3 grid of gaussian blobs, labelled by parity. */
export function classifyGaussianGrid9(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let variance = 0.2 + noise * 0.5;
  let centers: number[][] = [];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      centers.push([-4 + i * 4, -4 + j * 4, (i + j) % 2 === 0 ? 1 : -1]);
    }
  }
  let per = numSamples / centers.length;
  centers.forEach(([cx, cy, lbl]) => {
    for (let i = 0; i < per; i++) {
      points.push({x: normalRandom(cx, variance), y: normalRandom(cy, variance), label: lbl});
    }
  });
  return points;
}

/** Two rings arranged in an XOR pattern. */
export function classifyTwoRingsXor(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let centers = [[-2.5, -2.5, 1], [2.5, 2.5, 1], [-2.5, 2.5, -1], [2.5, -2.5, -1]];
  let per = numSamples / centers.length;
  centers.forEach(([cx, cy, lbl]) => {
    for (let i = 0; i < per; i++) {
      let a = randUniform(0, 2 * Math.PI);
      let r = 1.5;
      points.push({x: cx + r * Math.cos(a) + randUniform(-1, 1) * noise,
        y: cy + r * Math.sin(a) + randUniform(-1, 1) * noise, label: lbl});
    }
  });
  return points;
}

/** Finer 4x4 checkerboard. */
export function classifyChecker4(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let size = 3;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let cx = Math.floor((x + 6 + nx) / size);
    let cy = Math.floor((y + 6 + ny) / size);
    points.push({x, y, label: (cx + cy) % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Radial petals (flower): label by angular petals. */
export function classifyRadialPetals(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let petals = 6;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let r = Math.sqrt((x + nx) * (x + nx) + (y + ny) * (y + ny));
    let a = Math.atan2(y + ny, x + nx);
    let bound = 4 * Math.abs(Math.cos(petals / 2 * a));
    points.push({x, y, label: r < bound ? 1 : -1});
  }
  return points;
}

/** Heart shape boundary. */
export function classifyHeartShape(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  function inHeart(x: number, y: number): boolean {
    let xs = x / 3.5;
    let ys = y / 3.5;
    let v = Math.pow(xs * xs + ys * ys - 1, 3) - xs * xs * ys * ys * ys;
    return v < 0;
  }
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    points.push({x, y, label: inHeart(x + nx, y + ny) ? 1 : -1});
  }
  return points;
}

/** Wave interference: product of two sine waves. */
export function classifyWaveInterference(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let v = Math.sin(1.2 * (x + nx)) * Math.sin(1.2 * (y + ny));
    points.push({x, y, label: v >= 0 ? 1 : -1});
  }
  return points;
}

/** Gradient blobs: probability of label varies with x. */
export function classifyGradientBlobs(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let p = (x + 6) / 12;
    let nx = randUniform(-1, 1) * noise;
    points.push({x, y, label: (Math.random() + nx * 0.3) < p ? 1 : -1});
  }
  return points;
}

/** Three Gaussian classes collapsed to binary labels. */
export function classifyThreeClassBinary(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let variance = 0.5 + noise * 1.5;
  let centers = [[0, 4, 1], [-4, -3, -1], [4, -3, 1]];
  let per = numSamples / centers.length;
  centers.forEach(([cx, cy, lbl]) => {
    for (let i = 0; i < per; i++) {
      points.push({x: normalRandom(cx, variance), y: normalRandom(cy, variance), label: lbl});
    }
  });
  return points;
}

/** Noisy three concentric circles. */
export function classifyNoisyConcentric3(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let radii = [1.5, 3.5, 5];
  let per = numSamples / radii.length;
  radii.forEach((radius, idx) => {
    let label = idx % 2 === 0 ? 1 : -1;
    for (let i = 0; i < per; i++) {
      let a = randUniform(0, 2 * Math.PI);
      points.push({x: radius * Math.cos(a) + randUniform(-1, 1) * (0.4 + noise),
        y: radius * Math.sin(a) + randUniform(-1, 1) * (0.4 + noise), label});
    }
  });
  return points;
}

/** Five diagonal bands. */
export function classifyDiagonalBands5(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let width = 12 / 5;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let band = Math.floor((x + y + 12 + nx) / width);
    points.push({x, y, label: band % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Blob constellation: many small blobs scattered, random labels. */
export function classifyBlobConstellation(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let numBlobs = 12;
  let per = numSamples / numBlobs;
  let variance = 0.25 + noise * 0.4;
  for (let b = 0; b < numBlobs; b++) {
    let cx = randUniform(-5, 5);
    let cy = randUniform(-5, 5);
    let label = (b % 2 === 0) ? 1 : -1;
    for (let i = 0; i < per; i++) {
      points.push({x: normalRandom(cx, variance), y: normalRandom(cy, variance), label});
    }
  }
  return points;
}

/** Sparse vs dense: dense cluster vs sparse spread. */
export function classifySparseVsDense(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let half = numSamples / 2;
  for (let i = 0; i < half; i++) {
    points.push({x: normalRandom(-2, 0.7 + noise), y: normalRandom(-2, 0.7 + noise), label: 1});
  }
  for (let i = 0; i < half; i++) {
    points.push({x: randUniform(-6, 6), y: randUniform(-6, 6), label: -1});
  }
  return points;
}

/** Half plane with noisy boundary. */
export function classifyHalfPlaneNoisy(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * (noise + 0.3);
    let ny = randUniform(-1, 1) * (noise + 0.3);
    points.push({x, y, label: (0.7 * (x + nx) + 0.7 * (y + ny)) > 0 ? 1 : -1});
  }
  return points;
}

/** Lens: intersection of two circles. */
export function classifyLens(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let inA = dist({x: x + nx, y: y + ny}, {x: -1.8, y: 0}) < 3.5;
    let inB = dist({x: x + nx, y: y + ny}, {x: 1.8, y: 0}) < 3.5;
    points.push({x, y, label: (inA && inB) ? 1 : -1});
  }
  return points;
}

/** Hourglass: two opposing triangles meeting at center. */
export function classifyHourglass(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let inside = Math.abs(x + nx) < Math.abs(y + ny) * 0.9 && Math.abs(y + ny) < 5;
    points.push({x, y, label: inside ? 1 : -1});
  }
  return points;
}

/** Zigzag boundary classification. */
export function classifyZigzagBoundary(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  function tri(x: number): number {
    let p = 4;
    let t = ((x % p) + p) % p;
    return (t < p / 2 ? t : p - t) * 2 - 2;
  }
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let ny = randUniform(-1, 1) * noise;
    points.push({x, y, label: (y + ny) > tri(x) ? 1 : -1});
  }
  return points;
}

/** Comb teeth: vertical rectangular teeth from the bottom. */
export function classifyCombTeeth(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let width = 1.5;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let tooth = Math.floor((x + 6 + nx) / width) % 2 === 0;
    let inTooth = tooth && y < 2;
    points.push({x, y, label: inTooth ? 1 : -1});
  }
  return points;
}

/** Target with three rings. */
export function classifyTarget3Rings(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let r = dist({x: x + nx, y: y + ny}, {x: 0, y: 0});
    let ring = Math.floor(r / 1.8);
    points.push({x, y, label: ring % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Very tight double spiral. */
export function classifyDoubleSpiralTight(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;
  function genSpiral(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      let r = i / n * 5;
      let t = 4.0 * i / n * 2 * Math.PI + deltaT;
      points.push({x: r * Math.sin(t) + randUniform(-1, 1) * noise,
        y: r * Math.cos(t) + randUniform(-1, 1) * noise, label});
    }
  }
  genSpiral(0, 1);
  genSpiral(Math.PI, -1);
  return points;
}

/** Quadrant stripes: stripes within each quadrant. */
export function classifyQuadrantStripes(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let q = ((x >= 0) ? 1 : 0) + ((y >= 0) ? 1 : 0);
    let stripe = Math.floor(Math.abs(x + nx) / 1.5);
    points.push({x, y, label: (q + stripe) % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Gaussian ring: gaussian-distributed radius around a circle. */
export function classifyGaussianRing(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let half = numSamples / 2;
  for (let i = 0; i < half; i++) {
    let a = randUniform(0, 2 * Math.PI);
    let r = normalRandom(3.5, 0.3 + noise);
    points.push({x: r * Math.cos(a), y: r * Math.sin(a), label: 1});
  }
  for (let i = 0; i < half; i++) {
    let a = randUniform(0, 2 * Math.PI);
    let r = normalRandom(1.2, 0.3 + noise);
    points.push({x: r * Math.cos(a), y: r * Math.sin(a), label: -1});
  }
  return points;
}

/** Regression: 2D sine sin(x)+sin(y) variant (distinct from reg-waves freq). */
export function regressSin2D(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = Math.sin(0.8 * x + 0.8 * y) + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: normalized product x*y. */
export function regressProduct(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = (x * y) / 18 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: Euclidean distance to origin normalized. */
export function regressDistance(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = Math.sqrt(x * x + y * y) / 4 - 1 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: a slice of the Rosenbrock function. */
export function regressRosenbrockSlice(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let xs = x / 3;
    let ys = y / 3;
    let v = (1 - xs) * (1 - xs) + 100 * (ys - xs * xs) * (ys - xs * xs);
    let label = Math.tanh(v / 200) + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: smooth checkerboard surface. */
export function regressCheckerboardSmooth(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = Math.sin(x) * Math.sin(y) * 0.9 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: MATLAB peaks surface. */
export function regressPeaks(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let xs = x / 2;
    let ys = y / 2;
    let z = 3 * Math.pow(1 - xs, 2) * Math.exp(-(xs * xs) - Math.pow(ys + 1, 2))
      - 10 * (xs / 5 - Math.pow(xs, 3) - Math.pow(ys, 5)) * Math.exp(-xs * xs - ys * ys)
      - (1 / 3) * Math.exp(-Math.pow(xs + 1, 2) - ys * ys);
    let label = z / 6 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: Mexican hat (ricker) wavelet. */
export function regressMexicanHat(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let r2 = (x * x + y * y) / 4;
    let label = (1 - r2) * Math.exp(-r2 / 2) + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

export function classifyTrefoil2D(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let label = i % 2 === 0 ? 1 : -1;
    let t = randUniform(0, 2 * Math.PI);
    let r = label === 1 ? 4 : 4.5;
    let x = Math.sin(t) + 2 * Math.sin(2 * t);
    let y = Math.cos(t) - 2 * Math.cos(2 * t);
    let s = label === 1 ? 1.1 : 1.6;
    points.push({x: x * s + normalRandom(0, noise), y: y * s + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyGear(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let r = Math.sqrt(x * x + y * y);
    let a = Math.atan2(y, x);
    let teeth = 2 + Math.sin(a * 8);
    let label = r < teeth * 1.6 ? 1 : -1;
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyStar6(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let r = Math.sqrt(x * x + y * y);
    let a = Math.atan2(y, x);
    let edge = 3 + 1.8 * Math.cos(6 * a);
    let label = r < edge ? 1 : -1;
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyCrescentMoons3(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let centers = [{x: -3, y: 0}, {x: 0, y: 1.5}, {x: 3, y: 0}];
  for (let i = 0; i < numSamples; i++) {
    let g = i % 3;
    let t = randUniform(0, Math.PI);
    let x = centers[g].x + 3 * Math.cos(t);
    let y = centers[g].y + 3 * Math.sin(t) * (g === 1 ? -1 : 1);
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label: g === 1 ? -1 : 1});
  }
  return points;
}

export function classifyBlobLattice16(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let gx = Math.floor(randUniform(0, 4));
    let gy = Math.floor(randUniform(0, 4));
    let cx = -4.5 + gx * 3;
    let cy = -4.5 + gy * 3;
    let label = (gx + gy) % 2 === 0 ? 1 : -1;
    points.push({x: cx + normalRandom(0, 0.5 + noise), y: cy + normalRandom(0, 0.5 + noise), label});
  }
  return points;
}

export function classifyNoisyChecker6(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let cx = Math.floor((x + 6) / 2);
    let cy = Math.floor((y + 6) / 2);
    let label = (cx + cy) % 2 === 0 ? 1 : -1;
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label});
  }
  return points;
}

export function classifySpiral3ArmTight(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let arm = i % 3;
    let r = randUniform(0.5, 6);
    let t = 2.5 * r + arm * (2 * Math.PI / 3);
    let x = r * Math.cos(t);
    let y = r * Math.sin(t);
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label: arm === 0 ? 1 : -1});
  }
  return points;
}

export function classifyRingSegments(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let t = randUniform(0, 2 * Math.PI);
    let r = randUniform(3, 4.5);
    let seg = Math.floor(t / (Math.PI / 3));
    let label = seg % 2 === 0 ? 1 : -1;
    points.push({x: r * Math.cos(t) + normalRandom(0, noise), y: r * Math.sin(t) + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyPieSlices(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let a = Math.atan2(y, x) + Math.PI;
    let slice = Math.floor(a / (Math.PI / 4));
    let label = slice % 2 === 0 ? 1 : -1;
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyDroplet(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let edge = 3 * (1 - Math.sin(Math.atan2(y, x)));
    let r = Math.sqrt(x * x + y * y);
    let label = r < Math.abs(edge) + 0.5 ? 1 : -1;
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyInfinitySymbol(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let label = i % 2 === 0 ? 1 : -1;
    let t = randUniform(0, 2 * Math.PI);
    let scale = label === 1 ? 4 : 5.2;
    let x = scale * Math.cos(t);
    let y = scale * Math.sin(t) * Math.cos(t);
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyBowtie(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = Math.abs(y) < Math.abs(x) * 0.7 ? 1 : -1;
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyParallelSines(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let band = Math.floor((y - 1.5 * Math.sin(x) + 6) / 2);
    let label = band % 2 === 0 ? 1 : -1;
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyRadialGradientClass(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let r = Math.sqrt(x * x + y * y);
    let label = Math.sin(r * 1.5) > 0 ? 1 : -1;
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyConcentricArcs(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let t = randUniform(-Math.PI / 2, Math.PI / 2);
    let band = i % 4;
    let r = 1.5 + band * 1.3;
    let label = band % 2 === 0 ? 1 : -1;
    points.push({x: r * Math.cos(t) + normalRandom(0, noise), y: r * Math.sin(t) + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyClusteredOutliers(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    if (i % 5 === 0) {
      points.push({x: randUniform(-6, 6), y: randUniform(-6, 6), label: -1});
    } else {
      points.push({x: normalRandom(0, 1 + noise), y: normalRandom(0, 1 + noise), label: 1});
    }
  }
  return points;
}

export function classifyTwoBlobsOverlap(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let label = i % 2 === 0 ? 1 : -1;
    let cx = label === 1 ? -1.2 : 1.2;
    points.push({x: normalRandom(cx, 2 + noise), y: normalRandom(0, 2 + noise), label});
  }
  return points;
}

export function classifyDiamondGrid(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let u = Math.floor((x + y + 12) / 2);
    let v = Math.floor((x - y + 12) / 2);
    let label = (u + v) % 2 === 0 ? 1 : -1;
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyWavyStripes(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let stripe = Math.floor((x + 1.5 * Math.sin(y) + 6) / 1.5);
    let label = stripe % 2 === 0 ? 1 : -1;
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label});
  }
  return points;
}

export function classifyPlusMinusGrid(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let cx = Math.floor((x + 6) / 3);
    let cy = Math.floor((y + 6) / 3);
    let inCross = Math.abs((x % 3) - 1.5) < 0.6 || Math.abs((y % 3) - 1.5) < 0.6;
    let label = inCross ? 1 : -1;
    points.push({x: x + normalRandom(0, noise), y: y + normalRandom(0, noise), label});
  }
  return points;
}

export function regressRipple2(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let r = Math.sqrt(x * x + y * y);
    let label = Math.cos(r * 2) * Math.exp(-r / 6) + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

export function regressSaddle2(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = (x * x - y * y) / 18 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

export function regressGaussianHill(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = Math.exp(-((x - 1) * (x - 1) + (y + 1) * (y + 1)) / 8) * 2 - 1 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

export function regressSinProduct(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = Math.sin(x) * Math.sin(y) + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

export function regressAbsDiff(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = (Math.abs(x) - Math.abs(y)) / 6 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

export function regressLog(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = Math.log(1 + x * x + y * y) / 4 - 1 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

export function regressTanhWave(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = Math.tanh(x - Math.sin(y)) + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

export function regressCrossRidge(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = Math.exp(-x * x / 4) + Math.exp(-y * y / 4) - 1 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Eight-petal flower: angular petals labelled inside/outside. */
export function classifyPetal8(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let r = Math.sqrt((x + nx) * (x + nx) + (y + ny) * (y + ny));
    let a = Math.atan2(y + ny, x + nx);
    let bound = 4.5 * Math.abs(Math.cos(4 * a));
    points.push({x, y, label: r < bound ? 1 : -1});
  }
  return points;
}

/** Sun rays: alternating angular wedges radiating from center. */
export function classifySunRays(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let rays = 12;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let a = Math.atan2(y + ny, x + nx) + Math.PI;
    let wedge = Math.floor(a / (2 * Math.PI) * rays);
    points.push({x, y, label: wedge % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Brick wall: offset rows of rectangular bricks. */
export function classifyBrickWall(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let rowH = 2;
  let brickW = 3;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let row = Math.floor((y + 6 + ny) / rowH);
    let offset = (row % 2) * (brickW / 2);
    let col = Math.floor((x + 6 + nx + offset) / brickW);
    points.push({x, y, label: (row + col) % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Polka dots: grid of circular dots vs background. */
export function classifyPolkaDots(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let spacing = 3;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let dx = ((x + nx + 1.5) % spacing + spacing) % spacing - 1.5;
    let dy = ((y + ny + 1.5) % spacing + spacing) % spacing - 1.5;
    points.push({x, y, label: (dx * dx + dy * dy) < 1 ? 1 : -1});
  }
  return points;
}

/** Swirl pair: two offset swirls. */
export function classifySwirlPair(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let half = numSamples / 2;
  function genSwirl(cx: number, dir: number, label: number) {
    for (let i = 0; i < half; i++) {
      let r = i / half * 3;
      let t = dir * 2.5 * r;
      points.push({
        x: cx + r * Math.cos(t) + randUniform(-1, 1) * noise,
        y: r * Math.sin(t) + randUniform(-1, 1) * noise,
        label
      });
    }
  }
  genSwirl(-2.5, 1, 1);
  genSwirl(2.5, -1, -1);
  return points;
}

/** Archipelago: scattered island clusters with regional labels. */
export function classifyArchipelago(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let islands = [
    [-4.5, 3, 1], [-2, 4.5, 1], [0, 2, -1], [3.5, 4, -1],
    [4.5, 0, 1], [2, -2, -1], [-3, -3, 1], [-4.5, -1, -1],
    [0.5, -4.5, 1], [4, -4, -1]
  ];
  let variance = 0.35 + noise * 0.5;
  let per = numSamples / islands.length;
  islands.forEach(([cx, cy, label]) => {
    for (let i = 0; i < per; i++) {
      points.push({x: normalRandom(cx, variance), y: normalRandom(cy, variance), label});
    }
  });
  return points;
}

/** Ripple rings: many fine concentric rings via radial sine. */
export function classifyRippleRings(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let r = Math.sqrt((x + nx) * (x + nx) + (y + ny) * (y + ny));
    points.push({x, y, label: Math.sin(r * 2.2) >= 0 ? 1 : -1});
  }
  return points;
}

/** Lemniscate: inside a figure-eight (Bernoulli) curve vs outside. */
export function classifyLemniscate(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let a = 4;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let xs = x + nx;
    let ys = y + ny;
    let lhs = Math.pow(xs * xs + ys * ys, 2);
    let rhs = a * a * (xs * xs - ys * ys);
    points.push({x, y, label: lhs < rhs ? 1 : -1});
  }
  return points;
}

/** Clover: four-leaf rose curve interior. */
export function classifyClover(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let r = Math.sqrt((x + nx) * (x + nx) + (y + ny) * (y + ny));
    let a = Math.atan2(y + ny, x + nx);
    let bound = 4.5 * Math.abs(Math.sin(2 * a));
    points.push({x, y, label: r < bound ? 1 : -1});
  }
  return points;
}

/** Hex grid: hexagonal lattice cells labelled by parity. */
export function classifyHexGrid(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let size = 1.8;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let row = Math.round((y + ny) / (size * 1.5));
    let offset = (row % 2) * (size * Math.sqrt(3) / 2);
    let col = Math.round((x + nx - offset) / (size * Math.sqrt(3)));
    points.push({x, y, label: (row + col) % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Noisy sine band: band of width around a sine curve. */
export function classifyNoisySineBand(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let ny = randUniform(-1, 1) * (noise + 0.3);
    let center = 3 * Math.sin(x);
    points.push({x, y, label: Math.abs((y + ny) - center) < 1.5 ? 1 : -1});
  }
  return points;
}

/** Blobs in corners: a gaussian blob in each of the four corners. */
export function classifyBlobsInCorners(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let corners = [[-4.5, -4.5, 1], [4.5, 4.5, 1], [-4.5, 4.5, -1], [4.5, -4.5, -1]];
  let variance = 0.5 + noise;
  let per = numSamples / corners.length;
  corners.forEach(([cx, cy, label]) => {
    for (let i = 0; i < per; i++) {
      points.push({x: normalRandom(cx, variance), y: normalRandom(cy, variance), label});
    }
  });
  return points;
}

/** Dual crescents: two large facing crescents (distinct from crescent-pair). */
export function classifyDualCrescents(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;
  for (let i = 0; i < n; i++) {
    let angle = Math.PI * (i / n);
    let x = 4 * Math.cos(angle) + randUniform(-1, 1) * noise;
    let y = 4 * Math.sin(angle) - 1 + randUniform(-1, 1) * noise;
    points.push({x, y, label: 1});
  }
  for (let i = 0; i < n; i++) {
    let angle = Math.PI * (i / n);
    let x = -4 * Math.cos(angle) + randUniform(-1, 1) * noise;
    let y = -4 * Math.sin(angle) + 1 + randUniform(-1, 1) * noise;
    points.push({x, y, label: -1});
  }
  return points;
}

/** Dartboard 5: 5 concentric rings sliced into sectors. */
export function classifyDartboard5(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let sectors = 10;
  for (let i = 0; i < numSamples; i++) {
    let angle = randUniform(0, 2 * Math.PI);
    let r = Math.sqrt(Math.random()) * 5.5;
    let x = r * Math.cos(angle) + randUniform(-1, 1) * noise;
    let y = r * Math.sin(angle) + randUniform(-1, 1) * noise;
    let ang = Math.atan2(y, x) + Math.PI;
    let sector = Math.floor(ang / (2 * Math.PI) * sectors);
    let ring = Math.floor(dist({x, y}, {x: 0, y: 0}) / (5.5 / 5));
    points.push({x, y, label: (sector + ring) % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Radial checker: polar checkerboard over radius and angle. */
export function classifyRadialChecker(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let rings = 5;
  let sectors = 8;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let r = Math.sqrt((x + nx) * (x + nx) + (y + ny) * (y + ny));
    let a = Math.atan2(y + ny, x + nx) + Math.PI;
    let ri = Math.floor(r / (6 / rings));
    let si = Math.floor(a / (2 * Math.PI) * sectors);
    points.push({x, y, label: (ri + si) % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Maze stripes: long horizontal stripes broken by phase shift. */
export function classifyMazeStripes(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let band = Math.floor((y + 6 + ny) / 1.5);
    let phase = (band % 2 === 0) ? 0 : 1.5;
    let cell = Math.floor((x + 6 + nx + phase) / 3);
    points.push({x, y, label: (band + cell) % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Concentric spiral squares: square-norm rings. */
export function classifySpiralSquares(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let nx = randUniform(-1, 1) * noise;
    let ny = randUniform(-1, 1) * noise;
    let m = Math.abs(x + nx) + Math.abs(y + ny);
    points.push({x, y, label: Math.floor(m / 1.7) % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Regression: product of two sinc functions. */
export function regressSincProduct(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  function sinc(v: number) { return v === 0 ? 1 : Math.sin(v) / v; }
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = sinc(x) * sinc(y) + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: volcano (ring-shaped ridge). */
export function regressVolcano(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let r = Math.sqrt(x * x + y * y);
    let label = Math.exp(-Math.pow(r - 3, 2) / 2) - 0.3 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: stepped terraces. */
export function regressTerraces(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let r = Math.sqrt(x * x + y * y);
    let label = Math.floor(r) * 0.3 - 1 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: monkey saddle (third-order saddle). */
export function regressSaddle3(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let label = (x * x * x - 3 * x * y * y) / 80 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: multiple gaussian bumps. */
export function regressBumps(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  let centers = [[-3, -3], [3, 3], [-3, 3], [3, -3], [0, 0]];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let v = 0;
    centers.forEach(([cx, cy]) => {
      v += Math.exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / 3);
    });
    let label = v - 0.5 + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/** Regression: spiral height (value follows a spiral phase). */
export function regressSpiralHeight(numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-6, 6);
    let y = randUniform(-6, 6);
    let r = Math.sqrt(x * x + y * y);
    let a = Math.atan2(y, x);
    let label = Math.sin(a + r) + randUniform(-1, 1) * noise;
    points.push({x, y, label});
  }
  return points;
}

/**
 * Returns a sample from a uniform [a, b] distribution.
 * Uses the seedrandom library as the random generator.
 */
function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}

/**
 * Samples from a normal distribution. Uses the seedrandom library as the
 * random generator.
 *
 * @param mean The mean. Default is 0.
 * @param variance The variance. Default is 1.
 */
function normalRandom(mean = 0, variance = 1): number {
  let v1: number, v2: number, s: number;
  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1);

  let result = Math.sqrt(-2 * Math.log(s) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}

/** Returns the Euclidean distance between two points. */
function dist(a: Point, b: Point): number {
  let dx = a.x - b.x;
  let dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}
