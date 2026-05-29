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

/**
 * A three-dimensional example point with a label.
 */
export type Example3D = {
  x: number;
  y: number;
  z: number;
  label: number;
};

export type DataGenerator3D = (numSamples: number, noise: number) => Example3D[];

function randNormal(mean: number, stddev: number): number {
  // Box-Muller transform
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return mean + stddev * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function randUniform(a: number, b: number): number {
  return Math.random() * (b - a) + a;
}

/**
 * Two 3D Gaussian blobs, one centered at (-2,-2,-2) and one at (2,2,2).
 */
export function classifyTwoGaussBlobs(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const variance = 0.5 + noise * 3;
  const half = Math.floor(numSamples / 2);

  for (let i = 0; i < half; i++) {
    points.push({
      x: randNormal(-2, variance),
      y: randNormal(-2, variance),
      z: randNormal(-2, variance),
      label: 1
    });
  }
  for (let i = half; i < numSamples; i++) {
    points.push({
      x: randNormal(2, variance),
      y: randNormal(2, variance),
      z: randNormal(2, variance),
      label: -1
    });
  }
  return points;
}

/**
 * Concentric spheres: inner sphere label=1, outer shell label=-1.
 */
export function classifyConcentricSpheres(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    // Random point on unit sphere
    const theta = randUniform(0, 2 * Math.PI);
    const phi = Math.acos(randUniform(-1, 1));
    const isInner = i % 2 === 0;
    const r = isInner
      ? randUniform(0.3, 1.5) + randUniform(-1, 1) * noise
      : randUniform(2.5, 4.0) + randUniform(-1, 1) * noise;
    const x = r * Math.sin(phi) * Math.cos(theta);
    const y = r * Math.sin(phi) * Math.sin(theta);
    const z = r * Math.cos(phi);
    points.push({ x, y, z, label: isInner ? 1 : -1 });
  }
  return points;
}

/**
 * 3D double helix / spiral.
 * Two interleaved helices with opposite labels.
 */
export function classifyHelix(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const half = Math.floor(numSamples / 2);

  for (let i = 0; i < half; i++) {
    const t = (i / half) * 4 * Math.PI;
    const r = 2;
    points.push({
      x: r * Math.cos(t) + randNormal(0, noise),
      y: t * 0.5 - Math.PI * 2 + randNormal(0, noise),
      z: r * Math.sin(t) + randNormal(0, noise),
      label: 1
    });
  }
  for (let i = 0; i < numSamples - half; i++) {
    const t = (i / (numSamples - half)) * 4 * Math.PI;
    const r = 2;
    points.push({
      x: r * Math.cos(t + Math.PI) + randNormal(0, noise),
      y: t * 0.5 - Math.PI * 2 + randNormal(0, noise),
      z: r * Math.sin(t + Math.PI) + randNormal(0, noise),
      label: -1
    });
  }
  return points;
}

/**
 * Swiss roll dataset in 3D.
 */
export function classifySwissRoll(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const t = 1.5 * Math.PI * (1 + 2 * Math.random());
    const height = randUniform(-3, 3);
    const x = (t * Math.cos(t) / 5) + randNormal(0, noise);
    const y = height + randNormal(0, noise);
    const z = (t * Math.sin(t) / 5) + randNormal(0, noise);
    // Label by which half of the roll
    const label = Math.cos(t) > 0 ? 1 : -1;
    points.push({ x, y, z, label });
  }
  return points;
}
