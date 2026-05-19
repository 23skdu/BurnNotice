import { test, expect } from '@playwright/test';

test.describe('Model Profile Detail Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/model/test-model-123');
  });

  test('should display the model profile title with model ID', async ({ page }) => {
    await expect(page).toHaveTitle(/Model Profile - test-model-123/);
    await expect(page.getByText('Model Profile:')).toBeVisible();
    await expect(page.locator('.font-mono')).toContainText('test-model-123');
  });

  test('should display extracted directions table', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Extracted Directions' })).toBeVisible();

    await expect(page.getByText('Deception')).toBeVisible();
    await expect(page.getByText('Truth')).toBeVisible();
    await expect(page.locator('tbody td').nth(1)).toContainText('32');
  });

  test('should have visualization links for each direction', async ({ page }) => {
    const heatmapLinks = page.getByRole('link', { name: 'Heatmap' });
    await expect(heatmapLinks).toHaveCount(2);

    const subspaceLinks = page.getByRole('link', { name: 'Subspace' });
    await expect(subspaceLinks).toHaveCount(2);

    const circuitsLinks = page.getByRole('link', { name: 'Circuits' });
    await expect(circuitsLinks).toHaveCount(2);
  });

  test('should navigate to heatmap page', async ({ page }) => {
    await page.getByRole('link', { name: 'Heatmap' }).first().click();
    await expect(page).toHaveURL('/model/test-model-123/heatmap');
    await expect(page.getByText('Layer x Token Heatmap')).toBeVisible();
  });

  test('should navigate to subspace page', async ({ page }) => {
    await page.getByRole('link', { name: 'Subspace' }).first().click();
    await expect(page).toHaveURL('/model/test-model-123/subspace');
    await expect(page.getByText('Subspace Projection')).toBeVisible();
  });

  test('should navigate to circuits page', async ({ page }) => {
    await page.getByRole('link', { name: 'Circuits' }).first().click();
    await expect(page).toHaveURL('/model/test-model-123/circuits');
    await expect(page.getByText('Neural Circuits')).toBeVisible();
  });

  test('should have correct table headers', async ({ page }) => {
    const headers = page.locator('thead th');
    await expect(headers).toHaveCount(3);

    await expect(headers.nth(0)).toContainText('Direction');
    await expect(headers.nth(1)).toContainText('Layers');
    await expect(headers.nth(2)).toContainText('Visualizations');
  });

  test('should have proper table row count', async ({ page }) => {
    await expect(page.locator('tbody tr')).toHaveCount(2);
  });

  test('should navigate back to dashboard', async ({ page }) => {
    await page.getByRole('link', { name: 'Dashboard' }).click();
    await expect(page).toHaveURL('/');
  });

  test('should navigate to profiler', async ({ page }) => {
    await page.getByRole('link', { name: 'Profiler' }).click();
    await expect(page).toHaveURL('/profiler');
  });
});
