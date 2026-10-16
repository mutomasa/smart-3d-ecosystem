import { describe, expect, it } from "vitest";
import { AdaptiveTimestep, FixedStepClock } from "../src/sim/fixedStep";

describe("FixedStepClock", () => {
  it("runs one step per dt of wall time", () => {
    const c = new FixedStepClock(10);
    expect(c.advance(0)).toBe(0);
    expect(c.advance(25)).toBe(2);
    expect(c.advance(30)).toBe(1); // 5ms carried over + 5ms
  });

  it("caps steps per tick and drops the backlog after a stall", () => {
    const c = new FixedStepClock(10, 4);
    c.advance(0);
    expect(c.advance(1000)).toBe(4);
    expect(c.droppedMs).toBeGreaterThan(900);
    expect(c.advance(1010)).toBeLessThanOrEqual(2);
  });
});

describe("AdaptiveTimestep", () => {
  const opts = { budget: 0.6, warmup: 10, patience: 5 };

  it("ignores JIT warm-up spikes at start-up (regression #6)", () => {
    const a = new AdaptiveTimestep(opts);
    for (let i = 0; i < 10; i++) expect(a.observe(50)).toBe(false);
    for (let i = 0; i < 100; i++) a.observe(2);
    expect(a.dt).toBe(AdaptiveTimestep.FAST);
  });

  it("needs a sustained overload before switching, and switches back with hysteresis", () => {
    const a = new AdaptiveTimestep(opts);
    for (let i = 0; i < 10; i++) a.observe(1);
    for (let i = 0; i < 4; i++) a.observe(15);
    a.observe(1); // a single good sample resets the streak
    expect(a.dt).toBe(AdaptiveTimestep.FAST);
    for (let i = 0; i < 5; i++) a.observe(15);
    expect(a.dt).toBe(AdaptiveTimestep.SLOW);
    for (let i = 0; i < 50; i++) a.observe(6); // inside hysteresis band
    expect(a.dt).toBe(AdaptiveTimestep.SLOW);
    for (let i = 0; i < 5; i++) a.observe(1);
    expect(a.dt).toBe(AdaptiveTimestep.FAST);
  });

  it("never changes dt when disabled", () => {
    const a = new AdaptiveTimestep({ enabled: false });
    for (let i = 0; i < 500; i++) expect(a.observe(100)).toBe(false);
    expect(a.dt).toBe(AdaptiveTimestep.FAST);
  });
});
