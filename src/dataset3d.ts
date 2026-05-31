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

/** Nested cubes: alternating shells by Chebyshev distance. */
export function classifyNestedCubes(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-4.5, 4.5);
    const y = randUniform(-4.5, 4.5);
    const z = randUniform(-4.5, 4.5);
    const m = Math.max(Math.abs(x + randNormal(0, noise)),
      Math.abs(y + randNormal(0, noise)), Math.abs(z + randNormal(0, noise)));
    const band = Math.floor(m / 1.5);
    points.push({ x, y, z, label: band % 2 === 0 ? 1 : -1 });
  }
  return points;
}

/** Cube shell vs interior. */
export function classifyCubeShell(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const isShell = i % 2 === 0;
    let x: number, y: number, z: number;
    if (isShell) {
      // Point on the surface of a cube of half-size 4.
      const face = Math.floor(randUniform(0, 3));
      const s = randUniform(-1, 1) >= 0 ? 4 : -4;
      const a = randUniform(-4, 4);
      const b = randUniform(-4, 4);
      if (face === 0) { x = s; y = a; z = b; }
      else if (face === 1) { x = a; y = s; z = b; }
      else { x = a; y = b; z = s; }
    } else {
      x = randUniform(-1.5, 1.5);
      y = randUniform(-1.5, 1.5);
      z = randUniform(-1.5, 1.5);
    }
    points.push({
      x: x + randNormal(0, noise),
      y: y + randNormal(0, noise),
      z: z + randNormal(0, noise),
      label: isShell ? 1 : -1
    });
  }
  return points;
}

/** Double torus: two tori, labelled by which one. */
export function classifyDoubleTorus(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const half = Math.floor(numSamples / 2);
  function genTorus(cx: number, R: number, r: number, label: number, n: number) {
    for (let i = 0; i < n; i++) {
      const u = randUniform(0, 2 * Math.PI);
      const v = randUniform(0, 2 * Math.PI);
      points.push({
        x: cx + (R + r * Math.cos(v)) * Math.cos(u) + randNormal(0, noise),
        y: (R + r * Math.cos(v)) * Math.sin(u) + randNormal(0, noise),
        z: r * Math.sin(v) + randNormal(0, noise),
        label
      });
    }
  }
  genTorus(-2, 2.5, 0.8, 1, half);
  genTorus(2, 2.5, 0.8, -1, numSamples - half);
  return points;
}

/** Spiral cone: a helix whose radius grows with height. */
export function classifySpiralCone(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const t = (i / numSamples) * 6 * Math.PI;
    const h = (i / numSamples) * 8 - 4;
    const r = (h + 4) / 8 * 4;
    points.push({
      x: r * Math.cos(t) + randNormal(0, noise),
      y: h + randNormal(0, noise),
      z: r * Math.sin(t) + randNormal(0, noise),
      label: Math.floor(t / Math.PI) % 2 === 0 ? 1 : -1
    });
  }
  return points;
}

/** Lattice points: 3D grid of blobs labelled by parity. */
export function classifyLatticePoints(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const coords = [-3, 0, 3];
  const centers: number[][] = [];
  for (const cx of coords)
    for (const cy of coords)
      for (const cz of coords) {
        const parity = ((cx + cy + cz) / 3) ;
        centers.push([cx, cy, cz, Math.round(parity) % 2 === 0 ? 1 : -1]);
      }
  const per = Math.max(1, Math.floor(numSamples / centers.length));
  centers.forEach(([cx, cy, cz, label]) => {
    for (let i = 0; i < per; i++) {
      points.push({
        x: cx + randNormal(0, 0.4 + noise),
        y: cy + randNormal(0, 0.4 + noise),
        z: cz + randNormal(0, 0.4 + noise),
        label
      });
    }
  });
  return points;
}

/** 3D moons: two interleaving half-shells. */
export function classify3DMoons(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const half = Math.floor(numSamples / 2);
  for (let i = 0; i < half; i++) {
    const t = randUniform(0, Math.PI);
    const p = randUniform(0, Math.PI);
    points.push({
      x: 3 * Math.sin(p) * Math.cos(t) + randNormal(0, noise),
      y: 3 * Math.cos(p) + randNormal(0, noise),
      z: 3 * Math.sin(p) * Math.sin(t) + randNormal(0, noise),
      label: 1
    });
  }
  for (let i = half; i < numSamples; i++) {
    const t = randUniform(0, Math.PI);
    const p = randUniform(0, Math.PI);
    points.push({
      x: 3 - 3 * Math.sin(p) * Math.cos(t) + randNormal(0, noise),
      y: -3 * Math.cos(p) + 1.5 + randNormal(0, noise),
      z: 3 * Math.sin(p) * Math.sin(t) + randNormal(0, noise),
      label: -1
    });
  }
  return points;
}

/** Octahedron region vs surrounding sphere. */
export function classifyOctahedronVsSphere(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const inOcta = i % 2 === 0;
    let x: number, y: number, z: number;
    if (inOcta) {
      do {
        x = randUniform(-3, 3); y = randUniform(-3, 3); z = randUniform(-3, 3);
      } while (Math.abs(x) + Math.abs(y) + Math.abs(z) > 3);
    } else {
      const theta = randUniform(0, 2 * Math.PI);
      const phi = Math.acos(randUniform(-1, 1));
      const r = 4.5;
      x = r * Math.sin(phi) * Math.cos(theta);
      y = r * Math.sin(phi) * Math.sin(theta);
      z = r * Math.cos(phi);
    }
    points.push({
      x: x + randNormal(0, noise),
      y: y + randNormal(0, noise),
      z: z + randNormal(0, noise),
      label: inOcta ? 1 : -1
    });
  }
  return points;
}

/** Helix pair offset along z, labelled by helix. */
export function classifyHelixPair(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const half = Math.floor(numSamples / 2);
  function genHelix(zOff: number, label: number, n: number) {
    for (let i = 0; i < n; i++) {
      const t = (i / n) * 5 * Math.PI;
      points.push({
        x: 3 * Math.cos(t) + randNormal(0, noise),
        y: (t / (5 * Math.PI)) * 8 - 4 + randNormal(0, noise),
        z: 3 * Math.sin(t) + zOff + randNormal(0, noise),
        label
      });
    }
  }
  genHelix(1.5, 1, half);
  genHelix(-1.5, -1, numSamples - half);
  return points;
}

/** Five stacked planes (more levels than stacked-planes). */
export function classifyPlaneStack5(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const levels = [-4, -2, 0, 2, 4];
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

export function classifySphereSpiral(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const t = (i / numSamples) * 12 * Math.PI;
    const phi = Math.acos(1 - 2 * (i / numSamples));
    const r = 4;
    const x = r * Math.sin(phi) * Math.cos(t) + randNormal(0, noise);
    const y = r * Math.sin(phi) * Math.sin(t) + randNormal(0, noise);
    const z = r * Math.cos(phi) + randNormal(0, noise);
    points.push({x, y, z, label: i % 2 === 0 ? 1 : -1});
  }
  return points;
}

export function classifyCubeLattice(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const gx = Math.floor(randUniform(0, 3));
    const gy = Math.floor(randUniform(0, 3));
    const gz = Math.floor(randUniform(0, 3));
    const label = (gx + gy + gz) % 2 === 0 ? 1 : -1;
    points.push({
      x: -4 + gx * 4 + randNormal(0, 0.5 + noise),
      y: -4 + gy * 4 + randNormal(0, 0.5 + noise),
      z: -4 + gz * 4 + randNormal(0, 0.5 + noise),
      label
    });
  }
  return points;
}

export function classifyTwoHelices(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const label = i % 2 === 0 ? 1 : -1;
    const t = randUniform(0, 4 * Math.PI);
    const r = label === 1 ? 2 : 3.5;
    points.push({
      x: r * Math.cos(t) + randNormal(0, noise),
      y: r * Math.sin(t) + randNormal(0, noise),
      z: (t / (4 * Math.PI)) * 8 - 4 + randNormal(0, noise),
      label
    });
  }
  return points;
}

export function classifyConeStack(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const label = i % 2 === 0 ? 1 : -1;
    const h = randUniform(-4, 4);
    const radius = (label === 1 ? 0.5 : 1.5) * (4 - Math.abs(h)) / 4 + 0.3;
    const t = randUniform(0, 2 * Math.PI);
    points.push({
      x: radius * 3 * Math.cos(t) + randNormal(0, noise),
      y: radius * 3 * Math.sin(t) + randNormal(0, noise),
      z: h + randNormal(0, noise),
      label
    });
  }
  return points;
}

export function classifyTorusKnot2(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const label = i % 2 === 0 ? 1 : -1;
    const t = randUniform(0, 2 * Math.PI);
    const p = 3, q = label === 1 ? 2 : 4;
    const r = 2 + Math.cos(q * t);
    points.push({
      x: r * Math.cos(p * t) + randNormal(0, noise),
      y: r * Math.sin(p * t) + randNormal(0, noise),
      z: Math.sin(q * t) * 2 + randNormal(0, noise),
      label
    });
  }
  return points;
}

export function classifyPlaneVsBlob(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    if (i % 2 === 0) {
      points.push({
        x: randUniform(-4, 4) + randNormal(0, noise),
        y: randUniform(-4, 4) + randNormal(0, noise),
        z: randNormal(0, 0.3 + noise),
        label: 1
      });
    } else {
      points.push({
        x: randNormal(0, 1 + noise),
        y: randNormal(0, 1 + noise),
        z: randNormal(3, 1 + noise),
        label: -1
      });
    }
  }
  return points;
}

export function classifyOctantSpheres(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const ox = randUniform(-1, 1) >= 0 ? 1 : -1;
    const oy = randUniform(-1, 1) >= 0 ? 1 : -1;
    const oz = randUniform(-1, 1) >= 0 ? 1 : -1;
    const theta = randUniform(0, 2 * Math.PI);
    const phi = Math.acos(randUniform(-1, 1));
    const r = randUniform(0, 1.3);
    const label = ox * oy * oz > 0 ? 1 : -1;
    points.push({
      x: ox * 2.5 + r * Math.sin(phi) * Math.cos(theta) + randNormal(0, noise),
      y: oy * 2.5 + r * Math.sin(phi) * Math.sin(theta) + randNormal(0, noise),
      z: oz * 2.5 + r * Math.cos(phi) + randNormal(0, noise),
      label
    });
  }
  return points;
}

export function classifySwissRoll3Class(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const cls = i % 3;
    const t = 1.5 * Math.PI * (1 + 2 * (cls / 3 + randUniform(0, 0.33)));
    points.push({
      x: t * Math.cos(t) * 0.4 + randNormal(0, noise),
      y: randUniform(-4, 4) + randNormal(0, noise),
      z: t * Math.sin(t) * 0.4 + randNormal(0, noise),
      label: cls === 1 ? -1 : 1
    });
  }
  return points;
}

export function classifySphericalShell2(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const label = i % 2 === 0 ? 1 : -1;
    const theta = randUniform(0, 2 * Math.PI);
    const phi = Math.acos(randUniform(-1, 1));
    const r = label === 1 ? 2 : 4;
    points.push({
      x: r * Math.sin(phi) * Math.cos(theta) + randNormal(0, noise),
      y: r * Math.sin(phi) * Math.sin(theta) + randNormal(0, noise),
      z: r * Math.cos(phi) + randNormal(0, noise),
      label
    });
  }
  return points;
}

export function classifyGridXor3D(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-4, 4);
    const y = randUniform(-4, 4);
    const z = randUniform(-4, 4);
    const cx = Math.floor((x + 4) / 2.67);
    const cy = Math.floor((y + 4) / 2.67);
    const cz = Math.floor((z + 4) / 2.67);
    const label = (cx + cy + cz) % 2 === 0 ? 1 : -1;
    points.push({x: x + randNormal(0, noise), y: y + randNormal(0, noise), z: z + randNormal(0, noise), label});
  }
  return points;
}

/** Triple helix: three interleaved helices, label by helix parity. */
export function classifyHelixTriple(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const per = Math.floor(numSamples / 3);
  for (let h = 0; h < 3; h++) {
    const phase = (h / 3) * 2 * Math.PI;
    const label = h % 2 === 0 ? 1 : -1;
    const n = h === 2 ? numSamples - 2 * per : per;
    for (let i = 0; i < n; i++) {
      const t = (i / n) * 5 * Math.PI;
      points.push({
        x: 3 * Math.cos(t + phase) + randNormal(0, noise),
        y: (t / (5 * Math.PI)) * 8 - 4 + randNormal(0, noise),
        z: 3 * Math.sin(t + phase) + randNormal(0, noise),
        label
      });
    }
  }
  return points;
}

/** Cube edges: points along the 12 edges of a cube, labelled by edge axis. */
export function classifyCubeEdges(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const c = 4;
  const corners = [-c, c];
  const edges: number[][] = [];
  // Edges along x.
  for (const y of corners) for (const z of corners) edges.push([0, y, z]);
  // Edges along y.
  for (const x of corners) for (const z of corners) edges.push([1, x, z]);
  // Edges along z.
  for (const x of corners) for (const y of corners) edges.push([2, x, y]);
  for (let i = 0; i < numSamples; i++) {
    const e = edges[i % edges.length];
    const axis = e[0];
    const t = randUniform(-c, c);
    let x: number, y: number, z: number;
    if (axis === 0) { x = t; y = e[1]; z = e[2]; }
    else if (axis === 1) { y = t; x = e[1]; z = e[2]; }
    else { z = t; x = e[1]; y = e[2]; }
    points.push({
      x: x + randNormal(0, noise),
      y: y + randNormal(0, noise),
      z: z + randNormal(0, noise),
      label: axis === 0 ? 1 : -1
    });
  }
  return points;
}

/** Sphere clusters: gaussian blobs placed on a sphere, alternating labels. */
export function classifySphereClusters(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const centers: number[][] = [];
  const k = 8;
  for (let j = 0; j < k; j++) {
    const theta = (j / k) * 2 * Math.PI;
    const phi = Math.acos(1 - 2 * ((j + 0.5) / k));
    const r = 4;
    centers.push([
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.sin(phi) * Math.sin(theta),
      r * Math.cos(phi),
      j % 2 === 0 ? 1 : -1
    ]);
  }
  const per = Math.max(1, Math.floor(numSamples / centers.length));
  centers.forEach(([cx, cy, cz, label]) => {
    for (let i = 0; i < per; i++) {
      points.push({
        x: cx + randNormal(0, 0.5 + noise),
        y: cy + randNormal(0, 0.5 + noise),
        z: cz + randNormal(0, 0.5 + noise),
        label
      });
    }
  });
  return points;
}

/** Twisted torus: torus with a twist in the tube, labelled by twist side. */
export function classifyTwistedTorus(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  const R = 3, r = 1.2;
  for (let i = 0; i < numSamples; i++) {
    const u = randUniform(0, 2 * Math.PI);
    const v = randUniform(0, 2 * Math.PI);
    const tw = v + 2 * u;
    points.push({
      x: (R + r * Math.cos(v)) * Math.cos(u) + randNormal(0, noise),
      y: (R + r * Math.cos(v)) * Math.sin(u) + randNormal(0, noise),
      z: r * Math.sin(v) + randNormal(0, noise),
      label: Math.sin(tw) >= 0 ? 1 : -1
    });
  }
  return points;
}

/** 3D checker (8 cells): 2x2x2 voxel checkerboard. */
export function classifyChecker3D8(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-4, 4);
    const y = randUniform(-4, 4);
    const z = randUniform(-4, 4);
    const cx = (x + randNormal(0, noise)) >= 0 ? 1 : 0;
    const cy = (y + randNormal(0, noise)) >= 0 ? 1 : 0;
    const cz = (z + randNormal(0, noise)) >= 0 ? 1 : 0;
    points.push({x, y, z, label: (cx + cy + cz) % 2 === 0 ? 1 : -1});
  }
  return points;
}

/** Paraboloid shell: nested paraboloids labelled by which one. */
export function classifyParaboloidShell(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const label = i % 2 === 0 ? 1 : -1;
    const theta = randUniform(0, 2 * Math.PI);
    const rho = randUniform(0, 3);
    const k = label === 1 ? 0.4 : 0.7;
    points.push({
      x: rho * Math.cos(theta) + randNormal(0, noise),
      y: k * rho * rho - 3 + randNormal(0, noise),
      z: rho * Math.sin(theta) + randNormal(0, noise),
      label
    });
  }
  return points;
}
