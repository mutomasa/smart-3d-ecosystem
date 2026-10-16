/**
 * Generates comparison runs (same scenario and seeds, one experiment per mode/seed) by driving
 * the real browser scene, so the dashboard shows Boids / Rules / ML side by side.
 *   BASE_URL=http://localhost:8010 npx tsx scripts/generate-runs.ts [seconds] [seeds]
 */
import { chromium, type Browser } from "@playwright/test";

const BASE_URL = process.env.BASE_URL ?? "http://localhost:8010";
const SECONDS = Number(process.argv[2] ?? 25);
const SEEDS = (process.argv[3] ?? "1,2,3").split(",").map(Number);
const MODES = ["boids", "rules", "ml"] as const;
const SCENARIO = "preset=ML%20vs%20Rules&count=1500&predators=2&food=4";

async function run(browser: Browser, mode: string, seed: number): Promise<string> {
  const page = await browser.newPage({ viewport: { width: 800, height: 500 } });
  // The scenario comes from the URL so no default-scene telemetry is recorded first (#22).
  await page.goto(`${BASE_URL}/?experiment=cmp-${mode}-${seed}&${SCENARIO}&mode=${mode}&seed=${seed}`);
  await page.waitForTimeout(SECONDS * 1000);
  const hud = (await page.locator("#hud").textContent()) ?? "";
  await page.close();
  return `${mode} seed=${seed}: ${hud.split("\n").find((l) => l.startsWith("Survival"))}`;
}

async function main(): Promise<void> {
  const browser = await chromium.launch({ args: ["--ignore-gpu-blocklist"] });
  for (const seed of SEEDS) {
    // The three modes of one seed run in parallel (separate pages => separate worlds).
    const lines = await Promise.all(MODES.map((m) => run(browser, m, seed)));
    lines.forEach((l) => console.log(l));
  }
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
