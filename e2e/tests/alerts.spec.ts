import { test, expect } from '@playwright/test';

test.describe('Alert History Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/alerts');
  });

  test('should display the alerts title', async ({ page }) => {
    await expect(page).toHaveTitle(/Alerts - BurnNotice/);
    await expect(page.getByText('Alert History')).toBeVisible();
  });

  test('should display the alerts table', async ({ page }) => {
    const table = page.locator('table');
    await expect(table).toBeVisible();
  });

  test('should have correct table headers', async ({ page }) => {
    const headers = page.locator('thead th');
    await expect(headers).toHaveCount(6);

    await expect(headers.nth(0)).toContainText('Timestamp');
    await expect(headers.nth(1)).toContainText('Metric');
    await expect(headers.nth(2)).toContainText('Score');
    await expect(headers.nth(3)).toContainText('Threshold');
    await expect(headers.nth(4)).toContainText('Action');
    await expect(headers.nth(5)).toContainText('Session');
  });

  test('should display sample alert rows', async ({ page }) => {
    await expect(page.getByText('deception_score')).toBeVisible();
    await expect(page.getByText('conflict_score')).toBeVisible();
  });

  test('should display alert scores with color coding', async ({ page }) => {
    const redScore = page.locator('td.text-red-400');
    await expect(redScore).toContainText('0.85');

    const yellowScore = page.locator('td.text-yellow-400');
    await expect(yellowScore).toContainText('0.72');
  });

  test('should display action badges', async ({ page }) => {
    const steerBadge = page.locator('.bg-blue-900').first();
    await expect(steerBadge).toBeVisible();
    await expect(steerBadge).toContainText('steer');

    const logBadge = page.locator('.bg-gray-600');
    await expect(logBadge).toBeVisible();
    await expect(logBadge).toContainText('log');
  });

  test('should display session IDs in monospace font', async ({ page }) => {
    await expect(page.getByText('sess-001')).toBeVisible();
    await expect(page.getByText('sess-002')).toBeVisible();
  });

  test('should display timestamps', async ({ page }) => {
    await expect(page.getByText('2026-05-18T10:30:00Z')).toBeVisible();
    await expect(page.getByText('2026-05-18T10:31:00Z')).toBeVisible();
  });

  test('should display threshold values', async ({ page }) => {
    const rows = page.locator('tbody tr');
    await expect(rows).toHaveCount(2);

    const firstRowThreshold = rows.nth(0).locator('td').nth(3);
    await expect(firstRowThreshold).toContainText('0.7');

    const secondRowThreshold = rows.nth(1).locator('td').nth(3);
    await expect(secondRowThreshold).toContainText('0.8');
  });

  test('should have correct number of alert rows', async ({ page }) => {
    await expect(page.locator('tbody tr')).toHaveCount(2);
  });

  test('should navigate back to dashboard', async ({ page }) => {
    await page.getByRole('link', { name: 'Dashboard' }).click();
    await expect(page).toHaveURL('/');
  });

  test('should navigate to policies', async ({ page }) => {
    await page.getByRole('link', { name: 'Policies' }).click();
    await expect(page).toHaveURL('/policies');
  });

  test('should navigate to profiler', async ({ page }) => {
    await page.getByRole('link', { name: 'Profiler' }).click();
    await expect(page).toHaveURL('/profiler');
  });

  test('should have proper table styling', async ({ page }) => {
    const table = page.locator('table');
    const className = await table.getAttribute('class');
    expect(className).toContain('w-full');

    const thead = page.locator('thead');
    const theadClass = await thead.getAttribute('class');
    expect(theadClass).toContain('bg-gray-700');
  });

  test('should have responsive layout on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.getByText('Alert History')).toBeVisible();
    await expect(page.locator('table')).toBeVisible();
  });
});
