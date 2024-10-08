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

  let varianceScale = d3.scale.linear().domain([0, .5]).range([0.5, 4]);
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
  let labelScale = d3.scale.linear()
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
  let labelScale = d3.scale.linear()
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
  let labelScale = d3.scale.linear()
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

  let labelScale = d3.scale.linear()
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
