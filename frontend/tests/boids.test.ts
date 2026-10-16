import { describe, expect, it } from "vitest";
import { BoidsSimulation } from "../src/sim/boids";
import { makeConfig } from "../src/sim/config";
import { allFinite, localAlignment } from "../src/sim/analysis";

describe("BoidsSimulation", () => {
  it("does not produce NaN when agents share the exact same position", () => {
    const sim = new BoidsSimulation(makeConfig({ count: 50 }));
    sim.state.pos.fill(0);
    sim.state.vel.fill(0);
    for (let s = 0; s < 10; s++) sim.step();
    expect(allFinite(sim.state)).toBe(true);
  });

  it("keeps positions and velocities finite and bounded after 10,000 steps", () => {
    const cfg = makeConfig({ count: 120 });
    const sim = new BoidsSimulation(cfg);
    for (let s = 0; s < 10_000; s++) sim.step();
    expect(allFinite(sim.state)).toBe(true);
    for (let i = 0; i < sim.state.count; i++) {
      for (let a = 0; a < 3; a++) {
        expect(Math.abs(sim.state.pos[i * 3 + a])).toBeLessThanOrEqual(cfg.halfExtents[a] * 1.1 + 1e-3);
      }
      const v = sim.state.vel.subarray(i * 3, i * 3 + 3);
      expect(Math.hypot(v[0], v[1], v[2])).toBeLessThanOrEqual(cfg.maxSpeed + 1e-3);
    }
  });

  it("is deterministic for a fixed seed and dt", () => {
    const run = () => {
      const sim = new BoidsSimulation(makeConfig({ count: 200, seed: 123 }));
      for (let s = 0; s < 300; s++) sim.step();
      return sim.state.pos.slice(0, sim.state.count * 3);
    };
    expect(run()).toEqual(run());
  });

  it("produces different trajectories for different seeds", () => {
    const a = new BoidsSimulation(makeConfig({ count: 50, seed: 1 }));
    const b = new BoidsSimulation(makeConfig({ count: 50, seed: 2 }));
    expect(a.state.pos.slice(0, 150)).not.toEqual(b.state.pos.slice(0, 150));
  });

  it("keeps state arrays consistent when the population changes", () => {
    const sim = new BoidsSimulation(makeConfig({ count: 100 }));
    sim.setCount(500);
    expect(sim.state.count).toBe(500);
    expect(sim.state.pos.length).toBeGreaterThanOrEqual(500 * 3);
    expect(sim.state.aliveCount()).toBe(500);
    sim.step();
    sim.setCount(50);
    sim.step();
    expect(sim.state.count).toBe(50);
    expect(allFinite(sim.state)).toBe(true);
  });

  it("forms locally aligned schools over time", () => {
    const sim = new BoidsSimulation(makeConfig({ count: 400, seed: 5 }));
    const before = localAlignment(sim.state, 5);
    for (let s = 0; s < 600; s++) sim.step();
    const after = localAlignment(sim.state, 5);
    expect(after).toBeGreaterThan(0.9);
    expect(after).toBeGreaterThan(before + 0.2);
  });

  it("dead agents stay frozen", () => {
    const sim = new BoidsSimulation(makeConfig({ count: 10 }));
    sim.state.alive[3] = 0;
    const p = Array.from(sim.state.pos.slice(9, 12));
    for (let s = 0; s < 20; s++) sim.step();
    expect(Array.from(sim.state.pos.slice(9, 12))).toEqual(p);
  });
});
