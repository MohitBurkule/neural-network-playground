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

/**
 * Two interlocking rings (Hopf-link style): one ring in the xy-plane,
 * one in the xz-plane offset so they interlink.
 */
export function classifyLinkedRings(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const half = Math.floor(numSamples / 2);
  const r = 3;
  for (let i = 0; i < half; i++) {
    const t = randUniform(0, 2 * Math.PI);
    points.push({
      x: r * Math.cos(t) + randNormal(0, noise),
      y: r * Math.sin(t) + randNormal(0, noise),
      z: randNormal(0, noise),
      label: 1
    });
  }
  for (let i = half; i < numSamples; i++) {
    const t = randUniform(0, 2 * Math.PI);
    points.push({
      x: r + r * Math.cos(t) + randNormal(0, noise),
      y: randNormal(0, noise),
      z: r * Math.sin(t) + randNormal(0, noise),
      label: -1
    });
  }
  return points;
}

/**
 * 3D checkerboard cube: alternating labels across a 3x3x3 voxel grid.
 */
export function classifyCheckerboardCube(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const size = 2;
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-3, 3);
    const y = randUniform(-3, 3);
    const z = randUniform(-3, 3);
    const cx = Math.floor((x + 3 + randNormal(0, noise)) / size);
    const cy = Math.floor((y + 3 + randNormal(0, noise)) / size);
    const cz = Math.floor((z + 3 + randNormal(0, noise)) / size);
    const label = (cx + cy + cz) % 2 === 0 ? 1 : -1;
    points.push({ x, y, z, label });
  }
  return points;
}

/**
 * Double helix pair, two intertwined helices wound around the y-axis.
 */
export function classifyDoubleHelix(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const half = Math.floor(numSamples / 2);
  const r = 2.5;
  function genHelix(phase: number, label: number, n: number) {
    for (let i = 0; i < n; i++) {
      const t = (i / n) * 6 * Math.PI;
      points.push({
        x: r * Math.cos(t + phase) + randNormal(0, noise),
        y: (t / (6 * Math.PI)) * 8 - 4 + randNormal(0, noise),
        z: r * Math.sin(t + phase) + randNormal(0, noise),
        label
      });
    }
  }
  genHelix(0, 1, half);
  genHelix(Math.PI, -1, numSamples - half);
  return points;
}

/**
 * 3D XOR: label by parity of signs of x, y, z (8 octants).
 */
export function classifyXOR3D(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-4, 4);
    const y = randUniform(-4, 4);
    const z = randUniform(-4, 4);
    const sx = (x + randNormal(0, noise)) >= 0 ? 1 : 0;
    const sy = (y + randNormal(0, noise)) >= 0 ? 1 : 0;
    const sz = (z + randNormal(0, noise)) >= 0 ? 1 : 0;
    const label = (sx + sy + sz) % 2 === 0 ? 1 : -1;
    points.push({ x, y, z, label });
  }
  return points;
}

/**
 * Sphere shell vs solid core.
 */
export function classifyShellVsCore(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const theta = randUniform(0, 2 * Math.PI);
    const phi = Math.acos(randUniform(-1, 1));
    const isCore = i % 2 === 0;
    const r = isCore
      ? randUniform(0, 1.5) + randNormal(0, noise)
      : 4 + randNormal(0, noise * 0.5);
    points.push({
      x: r * Math.sin(phi) * Math.cos(theta),
      y: r * Math.sin(phi) * Math.sin(theta),
      z: r * Math.cos(phi),
      label: isCore ? 1 : -1
    });
  }
  return points;
}

/**
 * 3D S-curve: points along an S-shaped manifold, labelled by branch.
 */
export function classifySCurve3D(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const t = randUniform(-1.5 * Math.PI, 1.5 * Math.PI);
    const x = Math.sin(t) * 3 + randNormal(0, noise);
    const y = randUniform(-3, 3) + randNormal(0, noise);
    const z = Math.sign(t) * (Math.cos(t) - 1) * 3 + randNormal(0, noise);
    const label = t >= 0 ? 1 : -1;
    points.push({ x, y, z, label });
  }
  return points;
}

/**
 * Trefoil knot: points along a trefoil curve, labelled by which third of
 * the parameter they fall in (alternating).
 */
export function classifyTrefoilKnot(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const t = randUniform(0, 2 * Math.PI);
    const x = (Math.sin(t) + 2 * Math.sin(2 * t)) + randNormal(0, noise);
    const y = (Math.cos(t) - 2 * Math.cos(2 * t)) + randNormal(0, noise);
    const z = (-Math.sin(3 * t)) * 2 + randNormal(0, noise);
    const label = Math.floor(t / (2 * Math.PI) * 6) % 2 === 0 ? 1 : -1;
    points.push({ x, y, z, label });
  }
  return points;
}

/**
 * Mobius-like band: a twisted strip, labelled by side along the strip width.
 */
export function classifyMobiusBand(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const u = randUniform(0, 2 * Math.PI);
    const v = randUniform(-1, 1);
    const r = 3 + v * Math.cos(u / 2);
    const x = r * Math.cos(u) + randNormal(0, noise);
    const y = r * Math.sin(u) + randNormal(0, noise);
    const z = v * Math.sin(u / 2) * 2 + randNormal(0, noise);
    const label = v >= 0 ? 1 : -1;
    points.push({ x, y, z, label });
  }
  return points;
}

/**
 * Stacked planes: several horizontal planes at different heights with
 * alternating labels.
 */
export function classifyStackedPlanes(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const levels = [-3, -1, 1, 3];
  const per = Math.floor(numSamples / levels.length);
  levels.forEach((h, idx) => {
    const label = idx % 2 === 0 ? 1 : -1;
    for (let i = 0; i < per; i++) {
      points.push({
        x: randUniform(-4, 4) + randNormal(0, noise),
        y: h + randNormal(0, noise),
        z: randUniform(-4, 4) + randNormal(0, noise),
        label
      });
    }
  });
  return points;
}

/**
 * 3D spiral tower: a helix that climbs in y, labelled by height band.
 */
export function classifySpiralTower(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const r = 3;
  for (let i = 0; i < numSamples; i++) {
    const t = (i / numSamples) * 6 * Math.PI;
    const y = (i / numSamples) * 8 - 4;
    points.push({
      x: r * Math.cos(t) + randNormal(0, noise),
      y: y + randNormal(0, noise),
      z: r * Math.sin(t) + randNormal(0, noise),
      label: Math.floor((y + 4) / 2) % 2 === 0 ? 1 : -1
    });
  }
  return points;
}

/**
 * Octant checker: 8 octants of the cube, labelled by parity of octant.
 */
export function classifyOctantChecker(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-4, 4);
    const y = randUniform(-4, 4);
    const z = randUniform(-4, 4);
    const sx = (x + randNormal(0, noise)) >= 0 ? 1 : 0;
    const sy = (y + randNormal(0, noise)) >= 0 ? 1 : 0;
    const sz = (z + randNormal(0, noise)) >= 0 ? 1 : 0;
    const label = (sx ^ sy ^ sz) === 0 ? 1 : -1;
    points.push({ x, y, z, label });
  }
  return points;
}

/**
 * Sphere grid: points on a sphere surface, labelled in a checkerboard
 * pattern over latitude/longitude.
 */
export function classifySphereGrid(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const r = 4;
  const bands = 6;
  for (let i = 0; i < numSamples; i++) {
    const theta = randUniform(0, 2 * Math.PI);
    const phi = Math.acos(randUniform(-1, 1));
    const x = r * Math.sin(phi) * Math.cos(theta) + randNormal(0, noise);
    const y = r * Math.sin(phi) * Math.sin(theta) + randNormal(0, noise);
    const z = r * Math.cos(phi) + randNormal(0, noise);
    const latBand = Math.floor(phi / Math.PI * bands);
    const lonBand = Math.floor(theta / (2 * Math.PI) * bands);
    const label = (latBand + lonBand) % 2 === 0 ? 1 : -1;
    points.push({ x, y, z, label });
  }
  return points;
}
