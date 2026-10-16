import { describe, expect, it } from "vitest";
import { Rng } from "../src/sim/rng";

describe("Rng", () => {
  it("is reproducible for the same seed", () => {
    const a = new Rng(7), b = new Rng(7);
    for (let i = 0; i < 100; i++) expect(a.next()).toBe(b.next());
  });

  it("differs across seeds and stays in [0,1)", () => {
    const a = new Rng(1), b = new Rng(2);
    const xs = Array.from({ length: 1000 }, () => a.next());
    expect(xs.every((x) => x >= 0 && x < 1)).toBe(true);
    expect(xs[0]).not.toBe(b.next());
  });

  it("int() stays within bounds", () => {
    const r = new Rng(3);
    for (let i = 0; i < 1000; i++) {
      const v = r.int(5);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(5);
    }
  });
});
