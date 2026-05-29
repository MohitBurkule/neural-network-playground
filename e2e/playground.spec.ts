import { test, expect } from '@playwright/test';

test.describe('Neural Network Playground', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('page title is present', async ({ page }) => {
    const title = await page.title();
    expect(title).toBeTruthy();
  });

  test('#heatmap element renders and contains a canvas', async ({ page }) => {
    const heatmap = page.locator('#heatmap');
    await expect(heatmap).toBeVisible();
    // The heatmap SVG or canvas is rendered inside this container
    // Wait for the app to boot and render something inside #heatmap
    await page.waitForFunction(() => {
      const el = document.getElementById('heatmap');
      return el && el.children.length > 0;
    }, { timeout: 10000 });
    const children = await heatmap.locator('*').count();
    expect(children).toBeGreaterThan(0);
  });

  test('play/pause button exists and is clickable', async ({ page }) => {
    const playBtn = page.locator('#play-pause-button');
    await expect(playBtn).toBeVisible();
    // Click play to start training
    await playBtn.click();
    // Click again to pause
    await playBtn.click();
  });

  test('iteration number updates after clicking play then step', async ({ page }) => {
    const iterEl = page.locator('#iter-number');
    await expect(iterEl).toBeVisible();

    // Click "next step" button a few times to advance iterations
    const nextStepBtn = page.locator('#next-step-button');
    await expect(nextStepBtn).toBeVisible();

    await nextStepBtn.click();
    await nextStepBtn.click();
    await nextStepBtn.click();

    // After stepping, the iter number should be non-empty and > 0
    const iterText = await iterEl.textContent();
    const iterVal = parseInt(iterText || '0', 10);
    expect(iterVal).toBeGreaterThan(0);
  });

  test('loss values appear in the output panel', async ({ page }) => {
    // Step the network so loss gets computed
    const nextStepBtn = page.locator('#next-step-button');
    await nextStepBtn.click();

    const lossTest = page.locator('#loss-test');
    const lossTrain = page.locator('#loss-train');
    await expect(lossTest).toBeVisible();
    await expect(lossTrain).toBeVisible();

    const testText = await lossTest.textContent();
    const trainText = await lossTrain.textContent();
    expect(testText).toBeTruthy();
    expect(trainText).toBeTruthy();
  });

  test('switching dataset thumbnail changes selection', async ({ page }) => {
    // Click on the "Exclusive or" dataset thumbnail
    const xorThumb = page.locator('canvas.data-thumbnail[data-dataset="xor"]');
    await expect(xorThumb).toBeVisible();
    await xorThumb.click();

    // After clicking, the parent .dataset div should have the selected class
    const xorDataset = page.locator('.dataset:has(canvas[data-dataset="xor"])');
    // The app adds a "selected" class to active dataset
    await expect(xorDataset).toHaveClass(/selected/, { timeout: 3000 });
  });

  test('add hidden layer button increases the network SVG node count', async ({ page }) => {
    const addLayerBtn = page.locator('#add-layers');
    await expect(addLayerBtn).toBeVisible();

    // Count nodes in the SVG before adding a layer
    const svgEl = page.locator('#svg');
    await expect(svgEl).toBeVisible();

    const beforeCount = await svgEl.locator('circle').count();

    await addLayerBtn.click();

    // After adding a layer, there should be more circles in the network SVG
    await page.waitForFunction(
      (before) => {
        const svg = document.getElementById('svg');
        if (!svg) return false;
        return svg.querySelectorAll('circle').length > before;
      },
      beforeCount,
      { timeout: 5000 }
    );

    const afterCount = await svgEl.locator('circle').count();
    expect(afterCount).toBeGreaterThan(beforeCount);
  });

  test('remove hidden layer button decreases the network SVG node count', async ({ page }) => {
    // First add a layer so we can remove one
    const addLayerBtn = page.locator('#add-layers');
    await addLayerBtn.click();

    const svgEl = page.locator('#svg');
    const countAfterAdd = await svgEl.locator('circle').count();

    const removeLayerBtn = page.locator('#remove-layers');
    await expect(removeLayerBtn).toBeVisible();
    await removeLayerBtn.click();

    await page.waitForFunction(
      (before) => {
        const svg = document.getElementById('svg');
        if (!svg) return false;
        return svg.querySelectorAll('circle').length < before;
      },
      countAfterAdd,
      { timeout: 5000 }
    );

    const countAfterRemove = await svgEl.locator('circle').count();
    expect(countAfterRemove).toBeLessThan(countAfterAdd);
  });
});
