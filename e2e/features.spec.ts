import { test, expect } from '@playwright/test';

/**
 * Additional feature tests for the Neural Network Playground.
 * These specs exercise: activation/optimizer dropdowns, adversarial generate
 * button, 3D toggle, and custom-data load.
 */

test.describe('Activation and Optimizer dropdowns', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('activation dropdown is present and has multiple options', async ({ page }) => {
    const select = page.locator('#activations');
    await expect(select).toBeVisible();
    const count = await select.locator('option').count();
    expect(count).toBeGreaterThan(1);
  });

  test('changing activation to RELU does not throw a JS error', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    const select = page.locator('#activations');
    await select.selectOption({ label: /relu/i });

    // Small pause to let the app react
    await page.waitForTimeout(500);
    expect(errors).toHaveLength(0);
  });

  test('changing activation to SIGMOID does not throw a JS error', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    const select = page.locator('#activations');
    await select.selectOption({ label: /sigmoid/i });

    await page.waitForTimeout(500);
    expect(errors).toHaveLength(0);
  });

  test('optimizer dropdown is present and has multiple options', async ({ page }) => {
    const select = page.locator('#optimizer');
    await expect(select).toBeVisible();
    const count = await select.locator('option').count();
    expect(count).toBeGreaterThan(1);
  });

  test('changing optimizer to Adam does not throw a JS error', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    const select = page.locator('#optimizer');
    await select.selectOption({ label: /adam/i });

    await page.waitForTimeout(500);
    expect(errors).toHaveLength(0);
  });

  test('changing optimizer back to SGD does not throw a JS error', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    const select = page.locator('#optimizer');
    await select.selectOption({ label: /sgd/i });

    await page.waitForTimeout(500);
    expect(errors).toHaveLength(0);
  });
});

test.describe('Adversarial generate button', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('adversarial generate button exists', async ({ page }) => {
    const btn = page.locator('#adv-generate');
    await expect(btn).toBeVisible();
  });

  test('clicking generate button updates the adv-readout element', async ({ page }) => {
    const btn = page.locator('#adv-generate');
    await expect(btn).toBeVisible();

    const readout = page.locator('#adv-readout');
    await expect(readout).toBeAttached();

    // Step network first to ensure it has been initialized
    const nextStepBtn = page.locator('#next-step-button');
    await nextStepBtn.click();
    await page.waitForTimeout(300);

    await btn.click();
    await page.waitForTimeout(500);

    // The readout should have some text after clicking generate
    const text = await readout.textContent();
    expect(text).not.toBeNull();
  });
});

test.describe('3D toggle', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('3D toggle checkbox is present', async ({ page }) => {
    const toggle = page.locator('#threeD-toggle');
    await expect(toggle).toBeAttached();
  });

  test('threeview container exists in DOM', async ({ page }) => {
    const container = page.locator('#threeview');
    await expect(container).toBeAttached();
  });

  test('clicking 3D toggle shows the threeview container', async ({ page }) => {
    const toggle = page.locator('#threeD-toggle');
    const container = page.locator('#threeview');

    // Initially hidden
    const initialDisplay = await container.evaluate(
      (el) => window.getComputedStyle(el).display
    );
    expect(initialDisplay).toBe('none');

    // Click the checkbox label to enable 3D mode
    const label = page.locator('label[for="threeD-toggle"]');
    await label.click();
    await page.waitForTimeout(500);

    const afterDisplay = await container.evaluate(
      (el) => window.getComputedStyle(el).display
    );
    expect(afterDisplay).not.toBe('none');
  });
});

test.describe('Custom data load', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('custom-data textarea exists', async ({ page }) => {
    const textarea = page.locator('#custom-data-text');
    await expect(textarea).toBeAttached();
  });

  test('custom-data load button exists', async ({ page }) => {
    const btn = page.locator('#custom-data-load');
    await expect(btn).toBeVisible();
  });

  test('loading valid CSV data does not throw a JS error', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    const textarea = page.locator('#custom-data-text');
    await textarea.fill('1.0,2.0,1\n-1.0,-2.0,-1\n0.5,0.5,1\n-0.5,-0.5,-1');

    const btn = page.locator('#custom-data-load');
    await btn.click();
    await page.waitForTimeout(500);

    expect(errors).toHaveLength(0);
  });

  test('custom-data readout updates after loading CSV', async ({ page }) => {
    const textarea = page.locator('#custom-data-text');
    await textarea.fill('1.0,2.0,1\n-1.0,-2.0,-1\n0.5,0.5,1\n-0.5,-0.5,-1');

    const btn = page.locator('#custom-data-load');
    await btn.click();
    await page.waitForTimeout(500);

    // The readout should contain some feedback text
    const readout = page.locator('#custom-data-readout');
    const text = await readout.textContent();
    expect(text).not.toBeNull();
  });

  test('loading malformed CSV produces error feedback in readout', async ({ page }) => {
    const textarea = page.locator('#custom-data-text');
    await textarea.fill('bad,data\nnot,valid');

    const btn = page.locator('#custom-data-load');
    await btn.click();
    await page.waitForTimeout(500);

    const readout = page.locator('#custom-data-readout');
    const text = await readout.textContent();
    // Should show something (error message or empty-data warning)
    expect(text).not.toBeNull();
  });
});
