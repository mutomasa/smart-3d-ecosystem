/**
 * Records the Streamlit dashboard (embedded scene + Boids / Rules / ML comparison).
 *   DASHBOARD_URL=http://localhost:8502 npx tsx scripts/record-dashboard.ts <outDir>
 */
import { chromium } from "@playwright/test";
import { mkdirSync, renameSync } from "node:fs";
import { join } from "node:path";

const URL = process.env.DASHBOARD_URL ?? "http://localhost:8502";
const OUT = process.argv[2] ?? "../docs/demo/raw";
const SIZE = { width: 1280, height: 800 };

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ args: ["--ignore-gpu-blocklist"] });
  const context = await browser.newContext({ viewport: SIZE, recordVideo: { dir: OUT, size: SIZE } });
  const page = await context.newPage();
  await page.goto(`${URL}/?compare=cmp`);
  await page.getByText("Time-series comparison").waitFor({ timeout: 60_000 });
  await selectMetric(page, "Food Collected");
  await page.waitForTimeout(4000);
  await scrollToText(page, "Time-series comparison", 3000);
  await page.waitForTimeout(3000);
  await page.screenshot({ path: join(OUT, "..", "dashboard-timeseries.png") });
  await scrollToText(page, "Final value per run", 2500);
  await page.waitForTimeout(3000);
  await page.screenshot({ path: join(OUT, "..", "dashboard-charts.png") });
  await scrollToText(page, "Offline evaluation", 2500);
  await page.waitForTimeout(3000);
  await page.screenshot({ path: join(OUT, "..", "dashboard-offline.png") });
  await scrollToText(page, "SMART 3D ECOSYSTEM", 2500);
  await page.waitForTimeout(2000);
  await page.screenshot({ path: join(OUT, "..", "dashboard-top.png") });
  const video = page.video();
  await context.close();
  renameSync(await video!.path(), join(OUT, "dashboard.webm"));
  await browser.close();
}

async function selectMetric(page: import("@playwright/test").Page, label: string): Promise<void> {
  await page.locator('[data-testid="stSelectbox"]').filter({ hasText: "Metric" }).click();
  await page.getByRole("option", { name: label }).click();
}

/** Smoothly scrolls so that the heading containing `text` is near the top of the view. */
async function scrollToText(page: import("@playwright/test").Page, text: string, ms: number): Promise<void> {
  const dy = await page.evaluate((t) => {
    const main = document.querySelector<HTMLElement>('[data-testid="stMain"]')!;
    const el = [...main.querySelectorAll<HTMLElement>("h1, h2, h3")].find((h) => h.textContent?.includes(t));
    return el ? el.getBoundingClientRect().top - main.getBoundingClientRect().top - 60 : 0;
  }, text);
  await smoothScroll(page, dy, ms);
}

/** Streamlit scrolls inside its main container (and the wheel would hit the 3D iframe). */
async function smoothScroll(page: import("@playwright/test").Page, dy: number, ms: number): Promise<void> {
  const steps = Math.round(ms / 50);
  for (let k = 0; k < steps; k++) {
    await page.evaluate((d) => {
      const el = document.querySelector<HTMLElement>('[data-testid="stMain"]');
      (el ?? document.scrollingElement!).scrollBy(0, d);
    }, dy / steps);
    await page.waitForTimeout(50);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
