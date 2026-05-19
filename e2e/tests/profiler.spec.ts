import { test, expect } from '@playwright/test';

test.describe('Profiler Status Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/profiler');
  });

  test('should display the profiler title', async ({ page }) => {
    await expect(page).toHaveTitle(/Profiler - BurnNotice/);
    await expect(page.getByText('Profiler Status')).toBeVisible();
  });

  test('should display the job queue table', async ({ page }) => {
    const table = page.locator('table');
    await expect(table).toBeVisible();

    await expect(page.getByText('Model ID')).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Status' })).toBeVisible();
    await expect(page.getByText('Progress')).toBeVisible();
    await expect(page.getByText('Actions')).toBeVisible();
  });

  test('should display sample job row', async ({ page }) => {
    await expect(page.getByText('llama-3-8b')).toBeVisible();
    await expect(page.getByText('complete')).toBeVisible();
  });

  test('should display progress bar for completed job', async ({ page }) => {
    const progressBar = page.locator('.bg-green-500');
    await expect(progressBar).toBeVisible();
    const style = await progressBar.getAttribute('style');
    expect(style).toContain('width:100%');
  });

  test('should have View link for completed job', async ({ page }) => {
    const viewLink = page.getByRole('link', { name: 'View' });
    await expect(viewLink).toBeVisible();
    await expect(viewLink).toHaveAttribute('href', '/model/llama-3-8b');
  });

  test('should navigate to model profile when clicking View', async ({ page }) => {
    await page.getByRole('link', { name: 'View' }).click();
    await expect(page).toHaveURL('/model/llama-3-8b');
    await expect(page.getByText('Model Profile:')).toBeVisible();
  });

  test('should have SSE extension configured', async ({ page }) => {
    const sseContainer = page.locator('[hx-ext="sse"]');
    await expect(sseContainer).toBeVisible();
    const sseConnect = await sseContainer.getAttribute('sse-connect');
    expect(sseConnect).toBe('/sse');
  });

  test('should have correct table structure', async ({ page }) => {
    await expect(page.locator('thead')).toBeVisible();
    await expect(page.locator('tbody')).toBeVisible();
    await expect(page.locator('tbody tr')).toHaveCount(1);
    await expect(page.locator('tbody td')).toHaveCount(4);
  });

  test('should have status badge styling', async ({ page }) => {
    const badge = page.locator('.bg-green-900');
    await expect(badge).toBeVisible();
    await expect(badge).toHaveClass(/text-green-300/);
    await expect(badge).toHaveClass(/text-sm/);
  });

  test('should navigate back to dashboard', async ({ page }) => {
    await page.getByRole('link', { name: 'Dashboard' }).click();
    await expect(page).toHaveURL('/');
  });

  test('should have proper responsive layout', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.getByText('Profiler Status')).toBeVisible();
    await expect(page.locator('table')).toBeVisible();
  });
});
