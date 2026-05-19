import { test, expect } from '@playwright/test';

test.describe('Layer x Token Heatmap', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/model/test-model/heatmap');
  });

  test('should display the heatmap title with model ID', async ({ page }) => {
    await expect(page).toHaveTitle(/Heatmap - test-model/);
    await expect(page.getByText('Layer x Token Heatmap:')).toBeVisible();
    await expect(page.locator('.font-mono')).toContainText('test-model');
  });

  test('should render SVG heatmap', async ({ page }) => {
    const svg = page.locator('svg');
    await expect(svg).toBeVisible();
    await expect(svg).toHaveAttribute('viewBox', '0 0 800 400');
  });

  test('should render heatmap cells', async ({ page }) => {
    const cells = page.locator('svg rect');
    const count = await cells.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should have proper heatmap dimensions', async ({ page }) => {
    const svg = page.locator('svg');
    const width = await svg.getAttribute('width');
    const height = await svg.getAttribute('height');
    expect(width).toBe('800');
    expect(height).toBe('400');
  });

  test('should have colored cells with RGB values', async ({ page }) => {
    const firstCell = page.locator('svg rect').first();
    const fill = await firstCell.getAttribute('fill');
    expect(fill).toMatch(/^rgb\(\d+,\d+,\d+\)$/);
  });

  test('should navigate back to dashboard', async ({ page }) => {
    await page.getByRole('link', { name: 'Dashboard' }).click();
    await expect(page).toHaveURL('/');
  });
});

test.describe('Subspace Projection', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/model/test-model/subspace');
  });

  test('should display the subspace title with model ID', async ({ page }) => {
    await expect(page).toHaveTitle(/Subspace - test-model/);
    await expect(page.getByText('Subspace Projection:')).toBeVisible();
    await expect(page.locator('.font-mono')).toContainText('test-model');
  });

  test('should render SVG scatter plot', async ({ page }) => {
    const svg = page.locator('svg');
    await expect(svg).toBeVisible();
    await expect(svg).toHaveAttribute('viewBox', '0 0 600 400');
  });

  test('should display axis labels', async ({ page }) => {
    await expect(page.getByText('Truth')).toBeVisible();
    await expect(page.getByText('Deception')).toBeVisible();
  });

  test('should render axis lines', async ({ page }) => {
    const lines = page.locator('svg line');
    const count = await lines.count();
    expect(count).toBeGreaterThan(2);
  });

  test('should render scatter points', async ({ page }) => {
    const points = page.locator('svg circle');
    const count = await points.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should have decision boundary line', async ({ page }) => {
    const dashedLine = page.locator('svg line[stroke-dasharray]');
    await expect(dashedLine).toBeVisible();
    await expect(dashedLine).toHaveAttribute('stroke', '#f44');
  });

  test('should have mixed color points', async ({ page }) => {
    const greenPoints = page.locator('svg circle[fill="#4ade80"]');
    const redPoints = page.locator('svg circle[fill="#f87171"]');

    const greenCount = await greenPoints.count();
    const redCount = await redPoints.count();

    expect(greenCount).toBeGreaterThan(0);
    expect(redCount).toBeGreaterThan(0);
  });

  test('should navigate back to dashboard', async ({ page }) => {
    await page.getByRole('link', { name: 'Dashboard' }).click();
    await expect(page).toHaveURL('/');
  });
});

test.describe('Neural Circuit Graph', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/model/test-model/circuits');
  });

  test('should display the circuits title with model ID', async ({ page }) => {
    await expect(page).toHaveTitle(/Circuits - test-model/);
    await expect(page.getByText('Neural Circuits:')).toBeVisible();
    await expect(page.locator('.font-mono')).toContainText('test-model');
  });

  test('should render SVG circuit graph', async ({ page }) => {
    const svg = page.locator('svg');
    await expect(svg).toBeVisible();
    await expect(svg).toHaveAttribute('viewBox', '0 0 800 500');
  });

  test('should render circuit nodes', async ({ page }) => {
    const nodes = page.locator('svg circle');
    const count = await nodes.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should render circuit edges', async ({ page }) => {
    const edges = page.locator('svg line');
    const count = await edges.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should have mixed color nodes', async ({ page }) => {
    const greenNodes = page.locator('svg circle[fill="#4ade80"]');
    const redNodes = page.locator('svg circle[fill="#f87171"]');

    const greenCount = await greenNodes.count();
    const redCount = await redNodes.count();

    expect(greenCount).toBeGreaterThan(0);
    expect(redCount).toBeGreaterThan(0);
  });

  test('should have proper node sizing', async ({ page }) => {
    const firstNode = page.locator('svg circle').first();
    const r = await firstNode.getAttribute('r');
    expect(r).toBe('20');
  });

  test('should have edge styling', async ({ page }) => {
    const firstEdge = page.locator('svg line').first();
    const stroke = await firstEdge.getAttribute('stroke');
    const strokeWidth = await firstEdge.getAttribute('stroke-width');
    expect(stroke).toBe('#666');
    expect(strokeWidth).toBe('2');
  });

  test('should navigate back to dashboard', async ({ page }) => {
    await page.getByRole('link', { name: 'Dashboard' }).click();
    await expect(page).toHaveURL('/');
  });
});
