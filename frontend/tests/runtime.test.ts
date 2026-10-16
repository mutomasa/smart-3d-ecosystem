import { describe, expect, it } from "vitest";
import { SimulationRuntime } from "../src/sim/runtime";
import type { WorkerEvent } from "../src/sim/protocol";
import { FISH_STRIDE, BufferPool } from "../src/sim/frame";

type Frame = Extract<WorkerEvent, { type: "frame" }>;

function setup() {
  const events: WorkerEvent[] = [];
  const rt = new SimulationRuntime((e) => events.push(e));
  rt.handle({ type: "init", settings: { count: 100, seed: 3, adaptiveDt: false } });
  const frames = () => events.filter((e): e is Frame => e.type === "frame");
  return { rt, events, frames };
}

describe("SimulationRuntime", () => {
  it("publishes frames with packed fish data sized to the population", () => {
    const { rt, frames } = setup();
    rt.tick(0);
    rt.tick(20);
    const f = frames()[0];
    expect(f.count).toBe(100);
    expect(f.buffer.byteLength).toBeGreaterThanOrEqual(100 * FISH_STRIDE * 4);
    expect(new Float32Array(f.buffer)[6]).toBe(1); // alive flag of fish 0
  });

  it("stops publishing when all buffers are in flight and resumes when one is returned", () => {
    const { rt, frames } = setup();
    let t = 0;
    for (let k = 0; k < 10; k++) rt.tick((t += 20));
    expect(frames()).toHaveLength(3);
    rt.handle({ type: "returnBuffer", buffer: frames()[0].buffer });
    rt.tick((t += 20));
    expect(frames()).toHaveLength(4);
  });

  it("keeps frame buffer and state consistent after population changes", () => {
    const { rt, frames } = setup();
    rt.tick(0);
    rt.handle({ type: "setCount", count: 400 });
    rt.tick(20);
    const f = frames().at(-1)!;
    expect(f.count).toBe(400);
    expect(f.buffer.byteLength).toBeGreaterThanOrEqual(400 * FISH_STRIDE * 4);
    expect(f.alive).toBe(400);
  });

  it("reports dt changes in telemetry (0 when adaptive dt is off)", () => {
    const { rt, frames } = setup();
    rt.tick(0);
    rt.tick(20);
    expect(frames()[0].telemetry.dtChanges).toBe(0);
    expect(frames()[0].telemetry.dt).toBeCloseTo(1 / 60);
  });

  it("does not advance while paused", () => {
    const { rt, frames } = setup();
    rt.handle({ type: "pause", paused: true });
    rt.tick(0);
    rt.tick(100);
    expect(frames()).toHaveLength(0);
    expect(rt.world!.tick).toBe(0);
  });

  it("reset with the same seed reproduces the same state", () => {
    const { rt } = setup();
    const a = rt.world!.sim.state.pos.slice(0, 300);
    rt.handle({ type: "reset", settings: { count: 100, seed: 3 } });
    expect(rt.world!.sim.state.pos.slice(0, 300)).toEqual(a);
  });
});

describe("BufferPool", () => {
  it("reuses released buffers that are large enough", () => {
    const pool = new BufferPool();
    const b = pool.acquire(64, 2, 0)!;
    pool.release(b);
    expect(pool.acquire(32, 2, 0)).toBe(b);
    expect(pool.acquire(64, 1, 1)).toBeNull();
  });
});
