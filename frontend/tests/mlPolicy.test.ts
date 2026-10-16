import { describe, expect, it } from "vitest";
import { ExternalMlPolicy, ML_HOLD_S } from "../src/sim/mlPolicy";
import { GroupStrategies, W_FOOD } from "../src/sim/strategy";
import { OBS_DIM } from "../src/sim/groups";
import { World } from "../src/sim/world";
import { SimulationRuntime } from "../src/sim/runtime";
import type { WorkerEvent } from "../src/sim/protocol";

const action = (g: number, food: number) => ({ group_id: g, direction: [0, 0, 1], weights: [0.2, food, 0.5] });

function setup() {
  let t = 0;
  const policy = new ExternalMlPolicy(() => t);
  const strategies = new GroupStrategies(2);
  const obs = new Float32Array(2 * OBS_DIM);
  const sizes = new Uint32Array([1, 1]);
  const decide = () => policy.decide(obs, sizes, strategies);
  return { policy, strategies, decide, advance: (s: number) => (t += s) };
}

describe("ExternalMlPolicy", () => {
  it("applies fresh actions", () => {
    const { policy, strategies, decide } = setup();
    policy.receive(10, [action(0, 0.9), action(1, 0.9)], "bc-v1");
    decide();
    expect(policy.status).toBe("ml");
    expect(strategies.targetW[W_FOOD]).toBeCloseTo(0.9);
    expect(policy.modelVersion).toBe("bc-v1");
  });

  it("drops responses to older requests (stale tick)", () => {
    const { policy } = setup();
    expect(policy.receive(20, [action(0, 0.1)], "v")).toBe(true);
    expect(policy.receive(10, [action(0, 0.9)], "v")).toBe(false);
    expect(policy.staleDropped).toBe(1);
  });

  it("holds the last action briefly, then falls back to rules", () => {
    const { policy, decide, advance } = setup();
    policy.receive(1, [action(0, 0.9)], "v");
    advance(0.6);
    decide();
    expect(policy.status).toBe("holding");
    advance(ML_HOLD_S);
    decide();
    expect(policy.status).toBe("fallback");
  });

  it("uses rules while waiting for the first response", () => {
    const { policy, strategies, decide } = setup();
    decide();
    expect(policy.status).toBe("waiting");
    expect(strategies.targetW[W_FOOD]).toBeCloseTo(0.3); // rules "cruise" weights
  });
});

describe("ML mode in the worker runtime", () => {
  it("emits observations of non-empty groups at the policy rate and applies actions", async () => {
    const events: WorkerEvent[] = [];
    // Behave like the main thread: return every frame buffer right away.
    const rt: SimulationRuntime = new SimulationRuntime((e) => {
      events.push(e);
      if (e.type === "frame") queueMicrotask(() => rt.handle({ type: "returnBuffer", buffer: e.buffer }));
    });
    rt.handle({ type: "init", settings: { count: 160, seed: 1, mode: "ml", adaptiveDt: false } });
    let t = 0;
    for (let k = 0; k < 30; k++) { rt.tick((t += 1000 / 60)); await Promise.resolve(); }
    const obs = events.filter((e): e is Extract<WorkerEvent, { type: "observations" }> => e.type === "observations");
    expect(obs.length).toBeGreaterThanOrEqual(3);
    expect(obs[0].groupIds).toHaveLength(16);
    expect(obs[0].features.length).toBe(16 * OBS_DIM);
    rt.handle({ type: "policyActions", requestTick: obs.at(-1)!.tick, modelVersion: "bc-v1", actions: [action(3, 1)] });
    for (let k = 0; k < 7; k++) { rt.tick((t += 1000 / 60)); await Promise.resolve(); }
    const frame = events.filter((e): e is Extract<WorkerEvent, { type: "frame" }> => e.type === "frame").at(-1)!;
    expect(frame.policy.status).toBe("ml");
    expect(frame.policy.modelVersion).toBe("bc-v1");
  });

  it("is never emitted in boids / rules mode", () => {
    const events: WorkerEvent[] = [];
    const rt = new SimulationRuntime((e) => events.push(e));
    rt.handle({ type: "init", settings: { count: 50, mode: "rules" } });
    for (let k = 0, t = 0; k < 30; k++) rt.tick((t += 17));
    expect(events.some((e) => e.type === "observations")).toBe(false);
  });

  it("falls back to rules deterministically when the backend is silent", () => {
    const run = () => {
      const w = World.create({ count: 200, seed: 4, predators: 2, mode: "ml" });
      for (let s = 0; s < 300; s++) w.step(1 / 60);
      return w.metricsSnapshot();
    };
    const rules = World.create({ count: 200, seed: 4, predators: 2, mode: "rules" });
    for (let s = 0; s < 300; s++) rules.step(1 / 60);
    expect(run()).toEqual(rules.metricsSnapshot());
  });
});
