import { test, expect } from '@playwright/test';

test.describe('Navigation Flow', () => {
  test('should navigate from dashboard to all pages and back', async ({ page }) => {
    await page.goto('/');

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
    await page.getByRole('link', { name: 'Dashboard', exact: true }).click();
    await expect(page).toHaveURL('/');
  });

  test('should navigate from profiler to model profile and visualizations', async ({ page }) => {
    await page.goto('/profiler');
    await page.getByRole('link', { name: 'View' }).click();
    await expect(page).toHaveURL('/model/llama-3-8b');

    await page.getByRole('link', { name: 'Heatmap' }).first().click();
    await expect(page).toHaveURL('/model/llama-3-8b/heatmap');

    await page.goto('/model/llama-3-8b');
    await page.getByRole('link', { name: 'Subspace' }).first().click();
    await expect(page).toHaveURL('/model/llama-3-8b/subspace');

    await page.goto('/model/llama-3-8b');
    await page.getByRole('link', { name: 'Circuits' }).first().click();
    await expect(page).toHaveURL('/model/llama-3-8b/circuits');
  });

  test('should navigate from policies to alerts and back', async ({ page }) => {
    await page.goto('/policies');
    await page.getByRole('link', { name: 'Alerts' }).click();
    await expect(page).toHaveURL('/alerts');
    await page.getByRole('link', { name: 'Policies' }).click();
    await expect(page).toHaveURL('/policies');
  });

  test('should maintain navigation bar on all pages', async ({ page }) => {
    const pages = ['/', '/profiler', '/alerts', '/policies', '/monitor/test', '/model/test'];

    for (const path of pages) {
      await page.goto(path);
      const nav = page.locator('nav');
      await expect(nav).toBeVisible();
      const linkTexts = await nav.locator('a[href]').allTextContents();
      expect(linkTexts).toContain('BurnNotice');
      expect(linkTexts).toContain('Dashboard');
      expect(linkTexts).toContain('Profiler');
      expect(linkTexts).toContain('Alerts');
      expect(linkTexts).toContain('Policies');
    }
  });
});

test.describe('API Endpoints', () => {
  test('GET /api/summary should return valid JSON', async ({ request }) => {
    const response = await request.get('/api/summary');
    expect(response.ok()).toBeTruthy();
    expect(response.headers()['content-type']).toContain('application/json');

    const body = await response.json();
    expect(body).toHaveProperty('active_models');
    expect(body).toHaveProperty('total_profiles');
    expect(body).toHaveProperty('status');
    expect(body).toHaveProperty('timestamp');
  });

  test('POST /api/policies should save policy', async ({ request }) => {
    const response = await request.post('/api/policies');
    expect(response.ok()).toBeTruthy();

    const body = await response.text();
    expect(body).toContain('Policy saved successfully');
  });

  test('GET /api/policies should return method not allowed', async ({ request }) => {
    const response = await request.get('/api/policies');
    expect(response.status()).toBe(405);
  });

  test('POST /job/test-job/cancel should cancel job', async ({ request }) => {
    const response = await request.post('/job/test-job/cancel');
    expect(response.ok()).toBeTruthy();

    const body = await response.text();
    expect(body).toContain('Job test-job cancelled');
  });
});

test.describe('SSE Endpoint', () => {
  test.skip('should accept SSE connection', async ({ request }) => {
    const response = await request.get('/sse', { timeout: 5000 });
    expect(response.status()).toBe(200);
    expect(response.headers()['content-type']).toContain('text/event-stream');
  });
});

test.describe('Static Files', () => {
  test('should serve CSS files', async ({ request }) => {
    const response = await request.get('/static/css/dashboard.css');
    expect(response.status()).toBeLessThan(500);
  });
});

test.describe('Error Handling', () => {
  test('should return 404 for unknown routes', async ({ request }) => {
    const response = await request.get('/nonexistent');
    expect(response.status()).toBe(404);
  });

  test('should return 405 for wrong method on POST endpoints', async ({ request }) => {
    const response = await request.get('/api/policies');
    expect(response.status()).toBe(405);
  });
});

test.describe('Cross-Browser Compatibility', () => {
  test('should render correctly on desktop viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/');
    await expect(page.getByText('BurnNotice Dashboard')).toBeVisible();
  });

  test('should render correctly on tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');
    await expect(page.getByText('BurnNotice Dashboard')).toBeVisible();
  });

  test('should render correctly on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    await expect(page.getByText('BurnNotice Dashboard')).toBeVisible();
  });
});
