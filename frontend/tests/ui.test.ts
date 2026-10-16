import { describe, expect, it } from "vitest";
import { FpsMeter } from "../src/ui/fpsMeter";
import { MetricsLog } from "../src/ui/metricsLog";
import { formatHud } from "../src/ui/hud";
import { PRESETS } from "../src/presets";

describe("FpsMeter", () => {
  it("estimates 60 fps from 16.67ms frames", () => {
    const m = new FpsMeter();
    for (let i = 0; i <= 60; i++) m.frame(i * (1000 / 60));
    expect(m.fps()).toBeCloseTo(60, 0);
  });
});

describe("MetricsLog", () => {
  const sample = (t: number) => ({ t, tick: t, fps: 60, stepMs: 1, dt: 1 / 60, agents: 10, alive: 10, mode: "boids" });

  it("records at most one sample per interval and exports JSON", () => {
    const log = new MetricsLog(1000);
    for (let t = 0; t < 3000; t += 100) log.maybeRecord(sample(t));
    expect(log.samples).toHaveLength(3);
    const parsed = JSON.parse(log.toJSON({ seed: 1 }));
    expect(parsed.meta.seed).toBe(1);
    expect(parsed.samples[1].t).toBe(1000);
  });

  it("caps the number of samples", () => {
    const log = new MetricsLog(0, 5);
    for (let t = 0; t < 10; t++) log.maybeRecord(sample(t));
    expect(log.samples).toHaveLength(5);
  });
});

describe("HUD / presets", () => {
  it("formats FPS, agents, mode and step time", () => {
    const text = formatHud({ fps: 59.6, agents: 1000, alive: 990, mode: "rules", stepMs: 1.234, dt: 1 / 60 });
    expect(text).toContain("FPS  60");
    expect(text).toContain("Agents 990/1000");
    expect(text).toContain("Step 1.23 ms");
    expect(text).toContain("dt 1/60");
  });

  it("defines the four demo presets", () => {
    expect(Object.keys(PRESETS)).toEqual(["Calm School", "Predator Attack", "Food Rush", "ML vs Rules"]);
  });
});

describe("presetFromUrl (#22)", () => {
  it("starts directly in the requested scenario", async () => {
    const { presetFromUrl } = await import("../src/presets");
    const p = presetFromUrl("?preset=Food%20Rush&mode=ml&seed=3&count=1500&predators=2&food=4");
    expect(p).toMatchObject({ name: "Food Rush", mode: "ml", seed: 3, count: 1500, predators: 2, food: 4 });
  });

  it("falls back to the preset values on invalid input", async () => {
    const { presetFromUrl, PRESETS } = await import("../src/presets");
    const p = presetFromUrl("?preset=nope&mode=evil&seed=-1&count=99999&predators=x");
    expect(p).toEqual(PRESETS["Calm School"]);
  });
});
