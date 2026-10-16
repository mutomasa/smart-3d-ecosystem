import { describe, expect, it } from "vitest";
import { GroupStrategies, W_COHESION, W_FOOD, W_PREDATOR } from "../src/sim/strategy";
import { RulesPolicy, BoidsOnlyPolicy, rulesAction, RULES } from "../src/sim/policies";
import { OBS_DIM, FEATURE_NAMES } from "../src/sim/groups";

describe("GroupStrategies", () => {
  it("clamps directions to unit length, clamps weights and handles zero / NaN safely", () => {
    const s = new GroupStrategies(2);
    s.setTarget(0, 3, 4, 0, 2, -1, NaN);
    expect(Array.from(s.targetDir.slice(0, 3))).toEqual([expect.closeTo(0.6), expect.closeTo(0.8), 0]);
    expect(Array.from(s.targetW.slice(0, 3))).toEqual([1, 0, 0]);
    s.setTarget(1, 0, 0, 0, 0.5, 0.5, 0.5);
    expect(Array.from(s.targetDir.slice(3, 6))).toEqual([0, 0, 0]);
    s.setTarget(1, 0.25, 0, 0, 0.5, 0.5, 0.5); // weak preference keeps its magnitude
    expect(s.targetDir[3]).toBeCloseTo(0.25);
    s.setTarget(7, 1, 0, 0, 1, 1, 1); // out of range: ignored
  });

  it("smooths towards the target with a ~0.2 s time constant", () => {
    const s = new GroupStrategies(1, 0.2);
    s.setTarget(0, 1, 0, 0, 1, 1, 1);
    s.smooth(0.2);
    expect(s.weight(0, W_PREDATOR)).toBeCloseTo(1 - Math.exp(-1), 5);
    for (let k = 0; k < 120; k++) s.smooth(1 / 60);
    expect(s.weight(0, W_PREDATOR)).toBeGreaterThan(0.99);
  });
});

describe("policies", () => {
  const obs = (patch: Record<string, number>) => {
    const f = new Float32Array(OBS_DIM);
    f[FEATURE_NAMES.indexOf("nearest_predator_distance")] = 150;
    f[FEATURE_NAMES.indexOf("nearest_food_distance")] = 150;
    f[FEATURE_NAMES.indexOf("mean_energy")] = 1;
    for (const [k, v] of Object.entries(patch)) f[FEATURE_NAMES.indexOf(k)] = v;
    return f;
  };

  it("rules flee from a close predator", () => {
    const a = rulesAction(obs({ predator_present: 1, nearest_predator_relative_x: 5, nearest_predator_distance: 5 }));
    expect(a[0]).toBe(-5);
    expect(a.slice(3)).toEqual([...RULES.flee]);
  });

  it("rules forage when hungry and food exists, otherwise cruise", () => {
    const hungry = rulesAction(obs({ food_present: 1, nearest_food_relative_z: 9, nearest_food_distance: 9, mean_energy: 0.2 }));
    expect(hungry[2]).toBe(9);
    expect(hungry.slice(3)).toEqual([...RULES.forage]);
    const full = rulesAction(obs({ food_present: 1, nearest_food_relative_z: 9, mean_energy: 0.9 }));
    expect(full.slice(0, 3)).toEqual([0, 0, 0]);
  });

  it("boids-only disables predator / food weights and keeps neutral cohesion", () => {
    const s = new GroupStrategies(3);
    new BoidsOnlyPolicy().decide(new Float32Array(3 * OBS_DIM), new Uint32Array([1, 1, 1]), s);
    s.snap();
    expect(s.weight(1, W_PREDATOR)).toBe(0);
    expect(s.weight(1, W_FOOD)).toBe(0);
    expect(s.weight(1, W_COHESION)).toBe(0.5);
  });

  it("rules policy skips empty groups", () => {
    const s = new GroupStrategies(2);
    const o = new Float32Array(2 * OBS_DIM);
    o.set(obs({ predator_present: 1, nearest_predator_relative_x: 1, nearest_predator_distance: 1 }), OBS_DIM);
    new RulesPolicy().decide(o, new Uint32Array([0, 5]), s);
    expect(s.targetW[0]).toBe(0);
    expect(s.targetW[3]).toBe(RULES.flee[0]);
  });
});
