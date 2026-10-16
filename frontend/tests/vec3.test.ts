import { describe, expect, it } from "vitest";
import { clampMagnitude3, steerTowards3 } from "../src/sim/vec3";
import { limitTurn } from "../src/sim/motion";
import { axisPush } from "../src/sim/steering";

describe("vector helpers", () => {
  it("clampMagnitude3 limits length and keeps zero vectors", () => {
    const v = new Float32Array([3, 4, 0]);
    clampMagnitude3(v, 1);
    expect(Math.hypot(v[0], v[1], v[2])).toBeCloseTo(1);
    const z = new Float32Array(3);
    clampMagnitude3(z, 1);
    expect(Array.from(z)).toEqual([0, 0, 0]);
  });

  it("steerTowards3 returns zero for a zero direction", () => {
    const out = new Float32Array(3);
    steerTowards3(out, 0, 0, 0, 1, 2, 3, 10, 5);
    expect(Array.from(out)).toEqual([0, 0, 0]);
  });

  it("limitTurn caps the heading change", () => {
    const d = limitTurn(1, 0, 0, 0, 1, 0, 0.1);
    expect(Math.acos(d[0])).toBeCloseTo(0.1, 4);
  });

  it("limitTurn keeps heading on antiparallel input and handles zeros", () => {
    expect(Array.from(limitTurn(1, 0, 0, -1, 0, 0, 0.1))).toEqual([1, 0, 0]);
    expect(Array.from(limitTurn(0, 0, 0, 0, 0, 0, 0.1))).toEqual([1, 0, 0]);
  });

  it("axisPush is zero in the interior and bounded at the wall", () => {
    expect(axisPush(0, 50, 8)).toBe(0);
    expect(axisPush(60, 50, 8)).toBe(-1);
    expect(axisPush(-46, 50, 8)).toBeCloseTo(0.5);
  });
});
