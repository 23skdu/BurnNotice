import { test, expect } from '@playwright/test';

test.describe('Live Monitoring Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/monitor/sess-abc-456');
  });

  test('should display the monitor title with session ID', async ({ page }) => {
    await expect(page).toHaveTitle(/Monitor sess-abc-456/);
    await expect(page.getByText('Live Monitor:')).toBeVisible();
    await expect(page.locator('.font-mono')).toContainText('sess-abc-456');
  });

  test('should display all four score cards', async ({ page }) => {
    await expect(page.getByText('Deception')).toBeVisible();
    await expect(page.getByText('Conflict')).toBeVisible();
    await expect(page.getByText('Truth')).toBeVisible();
    await expect(page.getByText('Compliance')).toBeVisible();
  });

  test('should display initial score values', async ({ page }) => {
    await expect(page.locator('#deception-score')).toContainText('0.12');
    await expect(page.locator('#conflict-score')).toContainText('0.34');
    await expect(page.locator('#truth-score')).toContainText('0.89');
    await expect(page.locator('#compliance-score')).toContainText('0.95');
  });

  test('should display progress bars for each score', async ({ page }) => {
    const progressBars = page.locator('.bg-gray-700 .h-2');
    await expect(progressBars).toHaveCount(4);
  });

  test('should have SSE extension configured for live updates', async ({ page }) => {
    const sseContainer = page.locator('[hx-ext="sse"]');
    await expect(sseContainer).toBeVisible();
    const sseConnect = await sseContainer.getAttribute('sse-connect');
    expect(sseConnect).toBe('/sse');
  });

  test('should have color-coded score displays', async ({ page }) => {
    const deceptionScore = page.locator('#deception-score');
    await expect(deceptionScore).toHaveClass(/text-green-400/);

    const conflictScore = page.locator('#conflict-score');
    await expect(conflictScore).toHaveClass(/text-yellow-400/);

    const truthScore = page.locator('#truth-score');
    await expect(truthScore).toHaveClass(/text-green-400/);

    const complianceScore = page.locator('#compliance-score');
    await expect(complianceScore).toHaveClass(/text-green-400/);
  });

  test('should have correct grid layout', async ({ page }) => {
    const grid = page.locator('.grid-cols-4');
    await expect(grid).toBeVisible();
  });

  test('should navigate back to dashboard', async ({ page }) => {
    await page.getByRole('link', { name: 'Dashboard' }).click();
    await expect(page).toHaveURL('/');
  });

  test('should navigate to profiler', async ({ page }) => {
    await page.getByRole('link', { name: 'Profiler' }).click();
    await expect(page).toHaveURL('/profiler');
  });

  test('should navigate to alerts', async ({ page }) => {
    await page.getByRole('link', { name: 'Alerts' }).click();
    await expect(page).toHaveURL('/alerts');
  });

  test('should navigate to policies', async ({ page }) => {
    await page.getByRole('link', { name: 'Policies' }).click();
    await expect(page).toHaveURL('/policies');
  });

  test('should have responsive layout on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.getByText('Live Monitor:')).toBeVisible();
    await expect(page.getByText('Deception')).toBeVisible();
  });

  test('should have proper score card styling', async ({ page }) => {
    const scoreCards = page.locator('.bg-gray-800.rounded-lg');
    await expect(scoreCards).toHaveCount(4);

    for (let i = 0; i < 4; i++) {
      const card = scoreCards.nth(i);
      await expect(card).toHaveClass(/border-gray-700/);
      await expect(card).toHaveClass(/p-4/);
    }
  });
});
