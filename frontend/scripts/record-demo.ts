/**
 * Records the README demo clips with Playwright (headless Chromium, WebGL).
 *   BASE_URL=http://localhost:8010 npx tsx scripts/record-demo.ts <outDir>
 * The backend must serve frontend/dist (FastAPI) so that ML mode works.
 * Each scene is saved as <outDir>/<name>.webm; scripts/make_gifs.sh turns them into GIFs.
 */
import { chromium, type Browser, type Page } from "@playwright/test";
import { mkdirSync, renameSync } from "node:fs";
import { join } from "node:path";

const BASE_URL = process.env.BASE_URL ?? "http://localhost:8010";
const OUT = process.argv[2] ?? "../docs/demo/raw";
const SIZE = { width: 960, height: 600 };

interface Scene {
  name: string;
  preset: Record<string, unknown>;
  camera: [number, number, number];
  seconds: number;
  autoRotate?: boolean;
  warmup?: number;
}

const SCENES: Scene[] = [
  { name: "calm-school", preset: { name: "Calm School", count: 2000, seed: 42, mode: "boids", predators: 0, food: 0, weights: {} }, camera: [0, 18, 68], seconds: 9, autoRotate: true },
  { name: "predator-boids", preset: { name: "Predator Attack", count: 1500, seed: 7, mode: "boids", predators: 3, food: 0, weights: {} }, camera: [0, 30, 75], seconds: 12, warmup: 4 },
  { name: "predator-rules", preset: { name: "Predator Attack", count: 1500, seed: 7, mode: "rules", predators: 3, food: 0, weights: {} }, camera: [0, 30, 75], seconds: 12, warmup: 4 },
  { name: "food-rush", preset: { name: "Food Rush", count: 1500, seed: 11, mode: "rules", predators: 0, food: 6, weights: {} }, camera: [0, 40, 80], seconds: 12, warmup: 2 },
  { name: "ml-policy", preset: { name: "ML vs Rules", count: 1500, seed: 2024, mode: "ml", predators: 2, food: 4, weights: {} }, camera: [0, 30, 75], seconds: 12, warmup: 4 },
];

type AppHandle = {
  applyPreset(p: unknown): void;
  controls: { gui: { close(): void } };
  scene: { camera: { position: { set(x: number, y: number, z: number): void } }; controls: { autoRotate: boolean; autoRotateSpeed: number } };
};

async function setupScene(page: Page, scene: Scene): Promise<void> {
  await page.waitForFunction(() => "__app" in window);
  await page.evaluate(({ preset, camera, autoRotate }) => {
    const app = (window as unknown as { __app: AppHandle }).__app;
    app.controls.gui.close();
    app.applyPreset(preset);
    app.scene.camera.position.set(...camera);
    app.scene.controls.autoRotate = !!autoRotate;
    app.scene.controls.autoRotateSpeed = 1.2;
  }, { preset: scene.preset, camera: scene.camera, autoRotate: scene.autoRotate });
}

async function record(browser: Browser, scene: Scene): Promise<void> {
  const context = await browser.newContext({ viewport: SIZE, recordVideo: { dir: OUT, size: SIZE } });
  const page = await context.newPage();
  await page.goto(`${BASE_URL}/?experiment=demo-${scene.name}`);
  await setupScene(page, scene);
  await page.waitForTimeout(((scene.warmup ?? 1) + scene.seconds) * 1000);
  const hud = await page.locator("#hud").textContent();
  const video = page.video();
  await context.close();
  renameSync(await video!.path(), join(OUT, `${scene.name}.webm`));
  console.log(`${scene.name}: warmup ${scene.warmup ?? 1}s\n${hud}\n`);
}

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ args: ["--use-angle=default", "--enable-gpu", "--ignore-gpu-blocklist"] });
  const only = process.env.SCENES?.split(",");
  for (const scene of SCENES.filter((s) => !only || only.includes(s.name))) await record(browser, scene);
  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
