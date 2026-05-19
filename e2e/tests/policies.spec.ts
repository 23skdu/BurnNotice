import { test, expect } from '@playwright/test';

test.describe('Policy Configuration Editor', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/policies');
  });

  test('should display the policies title', async ({ page }) => {
    await expect(page).toHaveTitle(/Policies - BurnNotice/);
    await expect(page.getByText('Policy Configuration')).toBeVisible();
  });

  test('should display the add policy form', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Add Policy' })).toBeVisible();
    await expect(page.getByPlaceholder('Policy name')).toBeVisible();
    await expect(page.getByPlaceholder('Threshold')).toBeVisible();
  });

  test('should have metric dropdown with all options', async ({ page }) => {
    const metricSelect = page.locator('select[name="metric"]');
    await expect(metricSelect).toBeVisible();

    const options = await metricSelect.locator('option').allTextContents();
    expect(options).toContain('Deception Score');
    expect(options).toContain('Conflict Score');
    expect(options).toContain('Truth Score');
    expect(options).toContain('Compliance');
  });

  test('should have action dropdown with all options', async ({ page }) => {
    const actionSelect = page.locator('select[name="action"]');
    await expect(actionSelect).toBeVisible();

    const options = await actionSelect.locator('option').allTextContents();
    expect(options).toContain('Steer');
    expect(options).toContain('Block');
    expect(options).toContain('Log');
    expect(options).toContain('Terminate');
  });

  test('should have Add Policy submit button', async ({ page }) => {
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeVisible();
    await expect(submitBtn).toContainText('Add Policy');
  });

  test('should have HTMX form submission configured', async ({ page }) => {
    const form = page.locator('form');
    const hxPost = await form.getAttribute('hx-post');
    const hxSwap = await form.getAttribute('hx-swap');
    expect(hxPost).toBe('/api/policies');
    expect(hxSwap).toBe('outerHTML');
  });

  test('should submit policy form successfully', async ({ page }) => {
    await page.getByPlaceholder('Policy name').fill('test_policy');
    await page.getByPlaceholder('Threshold').fill('0.5');
    await page.locator('button[type="submit"]').click();

    await expect(page.getByText('Policy saved successfully')).toBeVisible();
  });

  test('should display existing policies table', async ({ page }) => {
    await expect(page.getByText('deception_threshold')).toBeVisible();
    await expect(page.getByText('deception_score')).toBeVisible();
    await expect(page.getByText('0.7')).toBeVisible();
  });

  test('should display policy action badge', async ({ page }) => {
    const badge = page.locator('.bg-blue-900');
    await expect(badge).toBeVisible();
    await expect(badge).toContainText('steer');
  });

  test('should have delete button for existing policies', async ({ page }) => {
    const deleteBtn = page.getByRole('button', { name: 'Delete' });
    await expect(deleteBtn).toBeVisible();
  });

  test('should have correct table headers', async ({ page }) => {
    const headers = page.locator('thead th');
    await expect(headers).toHaveCount(5);

    await expect(headers.nth(0)).toContainText('Name');
    await expect(headers.nth(1)).toContainText('Metric');
    await expect(headers.nth(2)).toContainText('Threshold');
    await expect(headers.nth(3)).toContainText('Action');
    await expect(headers.nth(4)).toContainText('Actions');
  });

  test('should navigate back to dashboard', async ({ page }) => {
    await page.getByRole('link', { name: 'Dashboard' }).click();
    await expect(page).toHaveURL('/');
  });

  test('should navigate to alerts', async ({ page }) => {
    await page.getByRole('link', { name: 'Alerts' }).click();
    await expect(page).toHaveURL('/alerts');
  });

  test('should navigate to profiler', async ({ page }) => {
    await page.getByRole('link', { name: 'Profiler' }).click();
    await expect(page).toHaveURL('/profiler');
  });

  test('should have proper form input styling', async ({ page }) => {
    const inputs = page.locator('input, select');
    const count = await inputs.count();
    expect(count).toBeGreaterThanOrEqual(4);

    for (let i = 0; i < count; i++) {
      const input = inputs.nth(i);
      const className = await input.getAttribute('class');
      expect(className).toContain('bg-gray-700');
      expect(className).toContain('rounded');
    }
  });

  test('should have threshold input with correct attributes', async ({ page }) => {
    const thresholdInput = page.getByPlaceholder('Threshold');
    const type = await thresholdInput.getAttribute('type');
    const step = await thresholdInput.getAttribute('step');
    const min = await thresholdInput.getAttribute('min');
    const max = await thresholdInput.getAttribute('max');

    expect(type).toBe('number');
    expect(step).toBe('0.1');
    expect(min).toBe('0');
    expect(max).toBe('1');
  });
});
