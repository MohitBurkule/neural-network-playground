# Contributing New Features

This guide explains where to add code when extending the playground with a new activation function, optimizer, loss function, dataset, input feature, or standalone lab. All registration points follow the same pattern: implement the logic in `src/nn.ts` or `src/dataset.ts`, register it in `src/state.ts`, and add the UI control in `index.html`.

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

The string key is both the URL hash value and the dropdown `value` attribute.

### 3. Add to the UI in `index.html`

Find the `<select>` element with `id="activations"` and add an `<option>`:

```html
<select id="activations">
  <!-- existing options -->
  <option value="my-act">My Activation</option>
</select>
```

The `value` must match the key you added in `state.ts`.

---

## Adding a New Optimizer

### 1. Implement in `src/nn.ts`

Add a new variant to the `OptimizerType` enum and handle it in `optimizerDelta`:

```typescript
// src/nn.ts — OptimizerType enum
export enum OptimizerType {
  // … existing …
  MY_OPT = "my-opt",
}

// src/nn.ts — optimizerDelta function
function optimizerDelta(
    grad: number, learningRate: number,
    optimizerType: OptimizerType, state: OptimizerState): number {
  switch (optimizerType) {
    // … existing cases …
    case OptimizerType.MY_OPT: {
      // Use state.m / state.v / state.t / state.acc / state.accDelta as scratch.
      return learningRate * grad; // example: plain SGD step
    }
  }
}
```

`OptimizerState` provides `m`, `v`, `t`, `vMax`, `acc`, and `accDelta`. Extend the interface if you need additional fields.

### 2. Register in `src/state.ts`

```typescript
export let optimizers: {[key: string]: nn.OptimizerType} = {
  // … existing entries …
  "my-opt": nn.OptimizerType.MY_OPT,
};
```

### 3. Add to the UI in `index.html`

```html
<select id="optimizer">
  <!-- existing options -->
  <option value="my-opt">My Optimizer</option>
</select>
```

---

## Adding a New Loss Function

### 1. Implement in `src/nn.ts`

Add a new static property to the `Errors` class:

```typescript
// src/nn.ts — inside class Errors { … }
public static MY_LOSS: ErrorFunction = {
  error: (output: number, target: number) => /* scalar loss */,
  der:   (output: number, target: number) => /* dLoss/dOutput */,
};
```

### 2. Register in `src/state.ts`

```typescript
export let lossFunctions: {[key: string]: nn.ErrorFunction} = {
  // … existing entries …
  "my-loss": nn.Errors.MY_LOSS,
};
```

### 3. Add to the UI in `index.html`

```html
<select id="loss-function">
  <!-- existing options -->
  <option value="my-loss">My Loss</option>
</select>
```

---

## Adding a New 2D Dataset

### 1. Implement the generator in `src/dataset.ts`

A dataset generator must satisfy `DataGenerator`: `(numSamples: number, noise: number) => Example2D[]`. Labels must be `1` or `−1`.

```typescript
// src/dataset.ts
export function classifyMyDataset(numSamples: number, noise: number): Example2D[] {
  const points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-5, 5);
    const y = randUniform(-5, 5);
    const noiseVal = randUniform(-1, 1) * noise;
    const label = (/* some rule */ + noiseVal) > 0 ? 1 : -1;
    points.push({ x, y, label });
  }
  return points;
}
```

`randUniform` and `normalRandom` helpers are already defined in `dataset.ts`. The playground uses `Math.seedrandom` so generators are reproducible as long as they only call `Math.random()`.

### 2. Register in `src/state.ts`

For a classification dataset add to `datasets`; for a regression dataset add to `regDatasets`:

```typescript
export let datasets: {[key: string]: dataset.DataGenerator} = {
  // … existing entries …
  "my-dataset": dataset.classifyMyDataset,
};
```

### 3. Add a thumbnail canvas in `index.html`

Dataset thumbnails are small `<canvas>` elements in the data panel. Find the block containing `data-dataset` attributes and add:

```html
<canvas class="data-thumbnail" data-dataset="my-dataset"></canvas>
```

The playground automatically draws a preview on each canvas and wires the click handler.

---

## Adding a New 3D Dataset

### 1. Implement the generator in `src/dataset3d.ts`

Satisfy `DataGenerator3D`: `(numSamples: number, noise: number) => Example3D[]`.

```typescript
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

Find the `THREE_GENERATORS` map and add your entry, plus the import at the top of the file:

```typescript
import { /* … */ classifyMyShape } from "./dataset3d";

const THREE_GENERATORS: {[key: string]: dataset3d.DataGenerator3D} = {
  // … existing entries …
  "my-shape": classifyMyShape,
};
```

### 3. Add to the 3D dataset dropdown in `index.html`

Find the `<select id="threeD-dataset">` element and add:

```html
<option value="my-shape">My Shape</option>
```

The `value` must match the key in `THREE_GENERATORS`.

---

## Adding a New Input Feature

### 1. Add to the `INPUTS` map in `src/playground.ts`

```typescript
let INPUTS: {[name: string]: InputFeature} = {
  // … existing features …
  "myFeature": { f: (x, y) => /* formula */, label: "myLabel" },
};
```

### 2. Add to `State.PROPS` in `src/state.ts`

```typescript
// inside State.PROPS array
{ name: "myFeature", type: Type.BOOLEAN },
```

And add a default value in the `State` class body:

```typescript
myFeature = false;
```

### 3. Add a checkbox in `index.html`

Find the features panel (checkboxes labelled x, y, x², etc.) and add:

```html
<label>
  <input type="checkbox" id="myFeature">
  <span>myLabel</span>
</label>
```

The `id` must match the key in `INPUTS` and the name in `State.PROPS`.

---

## Adding a New Standalone Lab

Labs are self-contained TypeScript modules with their own HTML page and browser bundle. Follow these steps:

### 1. Create `src/<name>.ts`

Write the algorithm and D3-based visualization in a single TypeScript file (or multiple files imported from it). The entry point runs immediately when the script loads — no shared state with `playground.ts`.

Conventions:
- Import D3 via `import * as d3 from 'd3';` (the UMD alias in `package.json` handles the rest).
- Do not import from `playground.ts`, `state.ts`, or `heatmap.ts`; labs are independent.
- Keep the bundle self-contained. If you need three.js, add a CDN `<script>` tag in the HTML template.

### 2. Create `labs/<name>.html`

Write an HTML template for the lab. Reference the bundle using a sibling path:

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>My Lab</title>
  <link rel="stylesheet" href="bundle.css">
</head>
<body>
  <div id="app"></div>
  <script src="bundle<name>.js"></script>
</body>
</html>
```

The bundle path `bundle<name>.js` matches the output filename produced by step 3 below.

### 3. Add a `build-<name>` script to `package.json`

```json
"build-<name>": "browserify src/<name>.ts -p [tsify] > dist/bundle<name>.js"
```

Then chain it into the `build-labs` script by appending `&& npm run build-<name>` before the `copyfiles` step:

```json
"build-labs": "npm run build-cnn && … && npm run build-<name> && copyfiles -f \"labs/*.html\" dist"
```

`build-all` calls `build-labs` automatically, so `npm run build-all` will now compile your lab.

### 4. Add a nav link in `index.html`

Find the "Advanced labs" link row near the bottom of the output panel:

```html
<div class="adv-labs-link" …>
  Advanced labs: <a href="cnn.html">CNN</a> · … · <a href="<name>.html">My Lab</a>
</div>
```

### 5. (Optional) Add unit tests

Add a Jest test file in `tests/<name>.test.ts` if the lab contains pure algorithmic logic (e.g., an exact solver, a dataset generator). Playwright E2E tests can be added to `e2e/` to smoke-test the lab page.

---

## General Tips

- **Registration maps are the single source of truth.** `state.ts` drives the dropdowns; `playground.ts` reads from them. Adding to the implementation file and `state.ts` is always sufficient for main-page features — you should not need to touch internal dispatch logic.
- **URL-hash serialization is automatic** for any property listed in `State.PROPS`. Use short, lowercase, hyphen-separated key names to keep URLs readable.
- **Reproducibility.** All dataset generators and weight initializers must use `Math.random()` only (seeded by `Math.seedrandom` at startup). Avoid `Date.now()` or `performance.now()` as sources of randomness.
- **Tests.** Add unit tests for any new activation derivative or dataset generator that has non-obvious mathematical properties. End-to-end tests in `e2e/` should smoke-test any new control that changes observable UI state.
- **Build and verify.** After any source change run `npm run build` (or `npm run build-all` for labs) and open `dist/index.html` in a browser (or `npm run serve`) to confirm the new option appears and works correctly.
- **TypeScript strict mode** is enabled in `tsconfig.json`. Avoid `any` casts; prefer explicit types and the existing helper interfaces.
