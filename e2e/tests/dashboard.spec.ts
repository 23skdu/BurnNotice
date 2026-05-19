import { test, expect } from '@playwright/test';

test.describe('Dashboard Home Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display the dashboard title', async ({ page }) => {
    await expect(page).toHaveTitle(/BurnNotice Dashboard/);
  });

  test('should display the navigation bar with all links', async ({ page }) => {
    const nav = page.locator('nav');
    await expect(nav).toBeVisible();

    const brand = nav.locator('a').filter({ hasText: 'BurnNotice' });
    await expect(brand).toBeVisible();

    const links = nav.locator('a[href]');
    await expect(links).toHaveCount(5);

    const linkTexts = await links.allTextContents();
    expect(linkTexts).toContain('BurnNotice');
    expect(linkTexts).toContain('Dashboard');
    expect(linkTexts).toContain('Profiler');
    expect(linkTexts).toContain('Alerts');
    expect(linkTexts).toContain('Policies');
  });

  test('should display summary cards with correct labels', async ({ page }) => {
    await expect(page.getByText('Active Models')).toBeVisible();
    await expect(page.getByText('Total Profiles')).toBeVisible();
    await expect(page.getByText('System Status')).toBeVisible();
    await expect(page.getByText('Active Sessions')).toBeVisible();
  });

  test('should display initial values in summary cards', async ({ page }) => {
    await expect(page.locator('text="0"').first()).toBeVisible();
  });

  test('should display quick links section', async ({ page }) => {
    await expect(page.getByText('Quick Links')).toBeVisible();
    await expect(page.getByRole('link', { name: 'Run Profiler' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'View Alerts' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Edit Policies' })).toBeVisible();
  });

  test('should navigate to profiler page when clicking Run Profiler', async ({ page }) => {
    await page.getByRole('link', { name: 'Run Profiler' }).click();
    await expect(page).toHaveURL('/profiler');
    await expect(page.getByText('Profiler Status')).toBeVisible();
  });

  test('should navigate to alerts page when clicking View Alerts', async ({ page }) => {
    await page.getByRole('link', { name: 'View Alerts' }).click();
    await expect(page).toHaveURL('/alerts');
    await expect(page.getByText('Alert History')).toBeVisible();
  });

  test('should navigate to policies page when clicking Edit Policies', async ({ page }) => {
    await page.getByRole('link', { name: 'Edit Policies' }).click();
    await expect(page).toHaveURL('/policies');
    await expect(page.getByText('Policy Configuration')).toBeVisible();
  });

  test('should navigate via nav links', async ({ page }) => {
    await page.getByRole('link', { name: 'Profiler', exact: true }).click();
    await expect(page).toHaveURL('/profiler');

    await page.getByRole('link', { name: 'Dashboard', exact: true }).click();
    await expect(page).toHaveURL('/');

    await page.getByRole('link', { name: 'Alerts', exact: true }).click();
    await expect(page).toHaveURL('/alerts');

    await page.getByRole('link', { name: 'Dashboard', exact: true }).click();
    await expect(page).toHaveURL('/');

    await page.getByRole('link', { name: 'Policies', exact: true }).click();
    await expect(page).toHaveURL('/policies');
  });

  test('should fetch summary API endpoint', async ({ page }) => {
    const response = await page.request.get('/api/summary');
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body).toHaveProperty('active_models');
    expect(body).toHaveProperty('total_profiles');
    expect(body).toHaveProperty('status', 'healthy');
    expect(body).toHaveProperty('timestamp');
  });

  test('should have proper HTML structure', async ({ page }) => {
    await expect(page.locator('html')).toBeAttached();
    await expect(page.locator('head')).toBeAttached();
    await expect(page.locator('body')).toBeAttached();
    await expect(page.locator('main')).toBeVisible();
  });

  test('should load HTMX script', async ({ page }) => {
    const htmxLoaded = await page.evaluate(() => typeof (window as any).htmx !== 'undefined');
    expect(htmxLoaded).toBeTruthy();
  });

  test('should have dark theme styling', async ({ page }) => {
    const body = page.locator('body');
    await expect(body).toHaveClass(/bg-gray-900/);
    await expect(body).toHaveClass(/text-gray-100/);
  });
});
