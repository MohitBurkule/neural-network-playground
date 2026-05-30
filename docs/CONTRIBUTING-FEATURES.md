# Contributing New Features

This guide explains where to add code when extending the playground with a new activation function, optimizer, dataset, or input feature. All registration points follow the same pattern: implement the logic in `src/nn.ts` or `src/dataset.ts`, register it in `src/state.ts`, and add the UI control in `index.html`.

---

## Adding a New Activation Function

### 1. Implement in `src/nn.ts`

Add a new static property to the `Activations` class. An `ActivationFunction` must implement three methods:

```typescript
// src/nn.ts — inside class Activations { … }
public static MY_ACT: ActivationFunction = {
  output: (x: number) => /* forward formula */,
  der:    (x: number) => /* derivative of output w.r.t. x */,
  compileToJs: (arg: string) => `/* JS expression using ${arg} */`
};
```

`compileToJs` is used by `compileNetworkToJs` to produce an exportable JavaScript function string. If your activation has no simple closed-form JS expression, return a call to a helper function name and document that the consumer must supply that helper.

### 2. Register in `src/state.ts`

Add a key/value entry to the `activations` map:

```typescript
// src/state.ts
export let activations: {[key: string]: nn.ActivationFunction} = {
  // … existing entries …
  "my-act": nn.Activations.MY_ACT,
};
```

The string key is both the URL hash value and the dropdown label.

### 3. Add to the UI in `index.html`

Find the `<select>` element with `id="activations"` and add an `<option>`:

```html
<select id="activations">
  <!-- existing options -->
  <option value="my-act">My Activation</option>
</select>
```

The `value` attribute must match the key you added in `state.ts`.

---

## Adding a New Optimizer

### 1. Implement in `src/nn.ts`

Add a new variant to the `OptimizerType` enum and handle it in `optimizerDelta`:

```typescript
// src/nn.ts — OptimizerType enum
export enum OptimizerType {
  SGD      = "sgd",
  MOMENTUM = "momentum",
  RMSPROP  = "rmsprop",
  ADAM     = "adam",
  MY_OPT   = "my-opt",   // <-- add here
}

// src/nn.ts — optimizerDelta function
function optimizerDelta(
    grad: number, learningRate: number,
    optimizerType: OptimizerType, state: OptimizerState): number {
  switch (optimizerType) {
    // … existing cases …
    case OptimizerType.MY_OPT: {
      // Use state.m / state.v / state.t as scratch storage.
      // Return the step delta (positive = subtract from weight).
      return learningRate * grad; // example: just SGD
    }
  }
}
```

`OptimizerState` already provides `m` (first moment), `v` (second moment), and `t` (step count) fields. If you need additional state, extend the `OptimizerState` interface.

### 2. Register in `src/state.ts`

```typescript
// src/state.ts
export let optimizers: {[key: string]: nn.OptimizerType} = {
  // … existing entries …
  "my-opt": nn.OptimizerType.MY_OPT,
};
```

### 3. Add to the UI in `index.html`

Find the `<select>` element with `id="optimizers"` and add an `<option>`:

```html
<select id="optimizers">
  <!-- existing options -->
  <option value="my-opt">My Optimizer</option>
</select>
```

---

## Adding a New 2D Dataset

### 1. Implement the generator in `src/dataset.ts`

A dataset generator must satisfy the `DataGenerator` type: `(numSamples: number, noise: number) => Example2D[]`. Labels must be `1` or `−1`.

```typescript
// src/dataset.ts
export function classifyMyDataset(numSamples: number, noise: number): Example2D[] {
  const points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-5, 5);
    const y = randUniform(-5, 5);
    const noiseVal = randUniform(-1, 1) * noise;
    const label = /* some rule */ ? 1 : -1;
    points.push({ x, y, label });
  }
  return points;
}
```

`randUniform` and `normalRandom` helpers are already defined in `dataset.ts`. The playground uses `Math.seedrandom` so generators will be reproducible as long as they only use `Math.random()`.

### 2. Register in `src/state.ts`

For a classification dataset add to the `datasets` map; for a regression dataset add to `regDatasets`:

```typescript
// src/state.ts
import * as dataset from "./dataset";

export let datasets: {[key: string]: dataset.DataGenerator} = {
  // … existing entries …
  "my-dataset": dataset.classifyMyDataset,
};
```

### 3. Add a thumbnail canvas in `index.html`

Dataset thumbnails are small `<canvas>` elements in the data panel. Find the block containing `data-dataset` attributes and add a new entry:

```html
<canvas class="data-thumbnail" data-dataset="my-dataset"></canvas>
```

The playground automatically draws a preview on each canvas and wires the click handler to switch datasets.

---

## Adding a New 3D Dataset

### 1. Implement the generator in `src/dataset3d.ts`

Satisfy the `DataGenerator3D` type: `(numSamples: number, noise: number) => Example3D[]`.

```typescript
// src/dataset3d.ts
export function classifyMyShape(numSamples: number, noise: number): Example3D[] {
  const points: Example3D[] = [];
  for (let i = 0; i < numSamples; i++) {
    // generate x, y, z and label ∈ {1, -1}
    points.push({ x, y, z, label });
  }
  return points;
}
```

### 2. Register in `src/playground.ts`

Find the `THREE_GENERATORS` map near the top of the 3D data section and add your entry:

```typescript
// src/playground.ts
const THREE_GENERATORS: {[key: string]: dataset3d.DataGenerator3D} = {
  "blobs":     classifyTwoGaussBlobs,
  "spheres":   classifyConcentricSpheres,
  "helix":     classifyHelix,
  "swissroll": classifySwissRoll,
  "my-shape":  classifyMyShape,  // <-- add here
};
```

Also add the import at the top of the file:

```typescript
import {
  Example3D,
  classifyTwoGaussBlobs,
  // …
  classifyMyShape          // <-- add here
} from "./dataset3d";
```

### 3. Add to the 3D dataset dropdown in `index.html`

Find the `<select>` element for the 3D dataset (look for `id="three-d-dataset"` or similar) and add an `<option>`:

```html
<option value="my-shape">My Shape</option>
```

The `value` must match the key in `THREE_GENERATORS`.

---

## Adding a New Input Feature

### 1. Add to the `INPUTS` map in `src/playground.ts`

```typescript
// src/playground.ts
let INPUTS: {[name: string]: InputFeature} = {
  // … existing features …
  "myFeature": { f: (x, y) => /* formula */, label: "myLabel" },
};
```

The `label` string is displayed on the node in the network diagram.

### 2. Add to `State.PROPS` in `src/state.ts`

So the feature's enabled/disabled state is preserved in the URL:

```typescript
// src/state.ts — inside State.PROPS array
{ name: "myFeature", type: Type.BOOLEAN },
```

And add a default value in the `State` class body:

```typescript
// src/state.ts — inside class State { … }
myFeature = false;
```

### 3. Add a checkbox in `index.html`

Find the features panel (the checkboxes labelled x, y, x², etc.) and add:

```html
<label>
  <input type="checkbox" id="myFeature">
  <span>myLabel</span>
</label>
```

The `id` must match the key in `INPUTS` and the name in `State.PROPS`.

---

## General Tips

- **Keep the registration maps as the single source of truth.** `state.ts` drives the dropdowns; `playground.ts` reads from them. Adding to both the implementation file and `state.ts` is always sufficient — you should not need to modify `nn.ts` internal dispatch logic for datasets or features.
- **URL-hash serialization is automatic** for any property listed in `State.PROPS`. Choose short, lowercase, hyphen-separated key names to keep URLs readable.
- **Tests.** Unit tests live in `tests/` (Jest) and end-to-end tests in `e2e/` (Playwright). Add a unit test for any new activation derivative or dataset generator that has non-obvious mathematical properties.
- **Build and verify.** After any source change run `npm run build` and open `dist/index.html` in a browser (or `npm run serve`) to confirm the new option appears and behaves correctly.
