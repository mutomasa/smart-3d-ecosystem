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
  it("switches to 1/30 under load and back with hysteresis", () => {
    const a = new AdaptiveTimestep(true, 0.6);
    for (let i = 0; i < 200; i++) a.observe(15);
    expect(a.dt).toBe(AdaptiveTimestep.SLOW);
    for (let i = 0; i < 200; i++) a.observe(6);
    expect(a.dt).toBe(AdaptiveTimestep.SLOW); // within hysteresis band
    for (let i = 0; i < 200; i++) a.observe(1);
    expect(a.dt).toBe(AdaptiveTimestep.FAST);
  });

  it("never changes dt when disabled", () => {
    const a = new AdaptiveTimestep(false);
    for (let i = 0; i < 100; i++) expect(a.observe(100)).toBe(false);
    expect(a.dt).toBe(AdaptiveTimestep.FAST);
  });
});
