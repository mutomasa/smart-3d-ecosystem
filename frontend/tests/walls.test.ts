import { describe, expect, it } from "vitest";
import { World } from "../src/sim/world";
import { makeConfig } from "../src/sim/config";
import { removeOutward } from "../src/sim/steering";

function wallFraction(world: World, seconds: number): number {
  let frac = 0, samples = 0;
  for (let s = 0; s < 60 * seconds; s++) {
    world.step(1 / 60);
    if (s % 60 !== 0) continue;
    const st = world.sim.state;
    let near = 0, alive = 0;
    for (let i = 0; i < st.count; i++) {
      if (!st.alive[i]) continue;
      alive++;
      for (let k = 0; k < 3; k++) {
        if (Math.abs(st.pos[i * 3 + k]) > world.config.halfExtents[k] - 2) { near++; break; }
      }
    }
    frac += near / alive;
    samples++;
  }
  return frac / samples;
}

describe("boundary-aware fleeing (regression #9)", () => {
  it("removeOutward clips only the wall-ward component inside the margin", () => {
    const cfg = makeConfig();
    const v = new Float32Array([1, 1, -1]);
    removeOutward(v, 0, cfg.halfExtents[1], 0, cfg); // touching the ceiling
    expect(Array.from(v)).toEqual([1, 0, -1]);
    const inward = new Float32Array([0, -1, 0]);
    removeOutward(inward, 0, cfg.halfExtents[1], 0, cfg);
    expect(inward[1]).toBe(-1);
    const interior = new Float32Array([1, 1, 1]);
    removeOutward(interior, 0, 0, 0, cfg);
    expect(Array.from(interior)).toEqual([1, 1, 1]);
  });

  it("rules-mode schools are not pinned against the walls while fleeing", () => {
    const fraction = wallFraction(World.create({ count: 1000, seed: 7, predators: 3, mode: "rules" }), 30);
    // Was 0.122 before #9. Pursuing predators (they chase the nearest fish within 12 units)
    // corner some fish legitimately, so the bound leaves room for that.
    expect(fraction).toBeLessThan(0.05);
  }, 60_000);
});
