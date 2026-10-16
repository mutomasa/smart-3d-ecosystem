import { describe, expect, it } from "vitest";
import { SpatialGrid } from "../src/sim/spatialGrid";
import { Rng } from "../src/sim/rng";

function randomCloud(n: number, seed: number): Float32Array {
  const rng = new Rng(seed);
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) pos[i] = rng.range(-20, 20);
  return pos;
}

describe("SpatialGrid", () => {
  it("finds every neighbour within the cell size (matches brute force)", () => {
    const n = 400, r = 5;
    const pos = randomCloud(n, 11);
    const alive = new Uint8Array(n).fill(1);
    const grid = new SpatialGrid([20, 20, 20], r);
    grid.rebuild(pos, alive, n);
    const out = new Int32Array(n);
    for (let i = 0; i < n; i += 37) {
      const m = grid.query(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2], out);
      const found = new Set(Array.from(out.subarray(0, m)));
      for (let j = 0; j < n; j++) {
        const d = Math.hypot(pos[j * 3] - pos[i * 3], pos[j * 3 + 1] - pos[i * 3 + 1], pos[j * 3 + 2] - pos[i * 3 + 2]);
        if (d < r) expect(found.has(j)).toBe(true);
      }
    }
  });

  it("skips dead agents", () => {
    const pos = new Float32Array([0, 0, 0, 0.1, 0, 0]);
    const alive = new Uint8Array([1, 0]);
    const grid = new SpatialGrid([10, 10, 10], 5);
    grid.rebuild(pos, alive, 2);
    const out = new Int32Array(8);
    expect(grid.query(0, 0, 0, out)).toBe(1);
    expect(out[0]).toBe(0);
  });

  it("clamps positions outside the world to border cells", () => {
    const grid = new SpatialGrid([10, 10, 10], 5);
    expect(grid.cellOfPosition(1000, 1000, 1000)).toBe(grid.cellTotal - 1);
    expect(grid.cellOfPosition(-1000, -1000, -1000)).toBe(0);
  });

  it("respects the output buffer limit", () => {
    const n = 100;
    const pos = new Float32Array(n * 3); // all at origin
    const grid = new SpatialGrid([10, 10, 10], 5);
    grid.rebuild(pos, new Uint8Array(n).fill(1), n);
    expect(grid.query(0, 0, 0, new Int32Array(10))).toBe(10);
    expect(grid.maxOccupancy).toBe(n);
  });
});
