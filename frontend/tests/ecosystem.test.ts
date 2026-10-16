import { describe, expect, it } from "vitest";
import { World } from "../src/sim/world";
import { FEATURE_NAMES, OBS_DIM } from "../src/sim/groups";
import { allFinite } from "../src/sim/analysis";
import schema from "../../shared/schemas/policy.json";

const run = (world: World, seconds: number) => {
  for (let s = 0; s < seconds * 60; s++) world.step(1 / 60);
  return world;
};

describe("predators", () => {
  it("eat fish and mark them dead (count stays, alive decreases)", () => {
    const w = run(World.create({ count: 300, seed: 1, predators: 3, mode: "boids" }), 15);
    const m = w.metricsSnapshot();
    expect(m.predation).toBeGreaterThan(0);
    expect(m.alive).toBe(300 - m.predation);
    expect(w.sim.state.count).toBe(300);
    expect(m.survivalRate).toBeCloseTo(m.alive / 300);
  });

  it("are limited to 0..5", () => {
    const w = World.create({ count: 10, predators: 9 });
    expect(w.predators.count).toBe(5);
    w.setPredators(-1);
    expect(w.predators.count).toBe(0);
  });

  it("rules mode survives better than boids-only on the same seed", () => {
    const boids = run(World.create({ count: 500, seed: 3, predators: 3, mode: "boids" }), 20).metricsSnapshot();
    const rules = run(World.create({ count: 500, seed: 3, predators: 3, mode: "rules" }), 20).metricsSnapshot();
    expect(rules.survivalRate).toBeGreaterThan(boids.survivalRate);
  }, 30_000);
});

describe("food", () => {
  it("rules mode collects more food than boids-only and raises energy", () => {
    const boids = run(World.create({ count: 500, seed: 4, food: 6, mode: "boids" }), 20).metricsSnapshot();
    const rules = run(World.create({ count: 500, seed: 4, food: 6, mode: "rules" }), 20).metricsSnapshot();
    expect(rules.foodCollected).toBeGreaterThan(boids.foodCollected);
    expect(rules.meanEnergy).toBeGreaterThan(boids.meanEnergy);
  }, 30_000);

  it("depleted patches respawn and energy stays within [0, 1]", () => {
    const w = World.create({ count: 200, seed: 2, food: 1, mode: "rules" });
    w.food.amount[0] = 0.001;
    run(w, 6);
    expect(w.food.amount[0]).toBeGreaterThan(0);
    const e = w.sim.state.energy.slice(0, 200);
    expect(e.every((v) => v >= 0 && v <= 1)).toBe(true);
  });

  it("manual spawn adds a patch up to the limit of 10", () => {
    const w = World.create({ count: 10, food: 9 });
    w.spawnFood();
    w.spawnFood();
    expect(w.food.count).toBe(10);
  });
});

describe("observations", () => {
  it("follow the shared schema order and use the absent sentinel", () => {
    expect(FEATURE_NAMES).toEqual(schema.observation.features);
    const w = World.create({ count: 200, seed: 5 });
    const obs = w.buildObservations();
    expect(obs.length).toBe(w.config.groupCount * OBS_DIM);
    const at = (k: string) => obs[FEATURE_NAMES.indexOf(k)];
    expect(at("predator_present")).toBe(0);
    expect(at("nearest_predator_distance")).toBe(schema.normalization.max_distance);
    expect(at("nearest_food_relative_x")).toBe(0);
    expect(obs.every(Number.isFinite)).toBe(true);
  });

  it("report the nearest predator relative to the group centre", () => {
    const w = World.create({ count: 200, seed: 5, predators: 1 });
    w.buildObservations();
    const obs = w.buildObservations();
    const rel = [6, 7, 8].map((k) => obs[k]);
    const expected = [0, 1, 2].map((a) => w.predators.pos[a] - w.groups.center[a]);
    rel.forEach((v, a) => expect(v).toBeCloseTo(expected[a], 3));
    expect(obs[18]).toBe(1);
  });
});

describe("world determinism", () => {
  it("same seed + scenario => identical metrics; stays finite", () => {
    const a = run(World.create({ count: 300, seed: 9, predators: 2, food: 3, mode: "rules" }), 10);
    const b = run(World.create({ count: 300, seed: 9, predators: 2, food: 3, mode: "rules" }), 10);
    expect(a.metricsSnapshot()).toEqual(b.metricsSnapshot());
    expect(allFinite(a.sim.state)).toBe(true);
  });

  it("food placement does not depend on the fish count (independent RNG streams)", () => {
    const a = World.create({ count: 100, seed: 1, food: 3 });
    const b = World.create({ count: 900, seed: 1, food: 3 });
    expect(Array.from(a.food.pos.slice(0, 9))).toEqual(Array.from(b.food.pos.slice(0, 9)));
  });
});
