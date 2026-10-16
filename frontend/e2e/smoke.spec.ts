import { expect, test, type Page } from "@playwright/test";

const hud = (page: Page) => page.locator("#hud");

async function applyPreset(page: Page, preset: Record<string, unknown>) {
  await page.evaluate((p) => (window as unknown as { __app: { applyPreset(p: unknown): void } }).__app.applyPreset(p), preset);
}

test("renders the aquarium with 1,500 fish and reports FPS / step time", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.locator("canvas")).toBeVisible();
  await expect(hud(page)).toContainText("Agents 1500/1500", { timeout: 15_000 });
  await expect(hud(page)).toContainText(/Step \d+\.\d+ ms/);
  await expect.poll(async () => Number((await hud(page).textContent())?.match(/FPS\s+(\d+)/)?.[1] ?? 0)).toBeGreaterThan(5);
  expect(errors).toEqual([]);
});

test("Predator Attack preset runs in rules mode", async ({ page }) => {
  await page.goto("/");
  await applyPreset(page, { name: "Predator Attack", count: 800, seed: 7, mode: "rules", predators: 3, food: 0, weights: {} });
  await expect(hud(page)).toContainText("Mode rules");
  await expect(hud(page)).toContainText("Agents");
  await expect(hud(page)).toContainText("Survival");
});

test("ML mode without a backend degrades gracefully to rules", async ({ page }) => {
  await page.goto("/?experiment=e2e");
  await applyPreset(page, { name: "ML vs Rules", count: 600, seed: 1, mode: "ml", predators: 1, food: 2, weights: {} });
  await expect(hud(page)).toContainText("Mode ml");
  await expect(hud(page)).toContainText(/Backend (closed|connecting) \| exp e2e/);
  await expect(hud(page)).toContainText(/\| waiting|-> rules/);
  await expect(hud(page)).toContainText("Agents 600/600");
});
