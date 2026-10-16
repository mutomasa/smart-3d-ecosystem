import { describe, expect, it } from "vitest";
import { SimulationRuntime } from "../src/sim/runtime";
import type { WorkerEvent } from "../src/sim/protocol";
import { FakeSocket } from "./fakeSocket";

type Frame = Extract<WorkerEvent, { type: "frame" }>;

async function setup() {
  const events: WorkerEvent[] = [];
  const sockets: FakeSocket[] = [];
  const rt: SimulationRuntime = new SimulationRuntime(
    (e) => {
      events.push(e);
      if (e.type === "frame") queueMicrotask(() => rt.handle({ type: "returnBuffer", buffer: e.buffer }));
    },
    () => { const s = new FakeSocket(); sockets.push(s); return s; },
  );
  rt.handle({ type: "init", settings: { count: 160, seed: 3, mode: "ml", adaptiveDt: false } });
  rt.handle({ type: "connectBackend", url: "ws://test/ws/simulation/x" });
  sockets[0].open();
  let t = 0;
  const run = async (n: number) => {
    for (let k = 0; k < n; k++) { rt.tick((t += 1000 / 60)); await Promise.resolve(); }
  };
  return { rt, events, socket: sockets[0], run, frames: () => events.filter((e): e is Frame => e.type === "frame") };
}

describe("backend link inside the worker (#13)", () => {
  it("sends observations straight from the worker, never via the main thread", async () => {
    const { events, socket, run } = await setup();
    await run(30);
    expect(socket.messages("policy_request").length).toBeGreaterThanOrEqual(3);
    expect(events.some((e) => e.type === "observations")).toBe(false);
  });

  it("applies responses in the worker and reports backend state in frames", async () => {
    const { socket, run, frames } = await setup();
    await run(10);
    const req = socket.messages("policy_request").at(-1);
    socket.push({
      type: "policy_response", request_id: req.request_id, tick: req.tick, policy: "ml", model_version: "bc-v1",
      latency_ms: 1, actions: [{ group_id: 0, direction: [1, 0, 0], weights: [0.1, 0.9, 0.5] }],
    });
    await run(7);
    const f = frames().at(-1)!;
    expect(f.policy.status).toBe("ml");
    expect(f.policy.modelVersion).toBe("bc-v1");
    expect(f.backend?.connection).toBe("open");
    expect(f.backend?.latencyP50).not.toBeNull();
  });

  it("sends 1 Hz telemetry with the client FPS and forwards settings to the main thread", async () => {
    const { rt, events, socket, run } = await setup();
    rt.handle({ type: "clientStats", fps: 58 });
    await run(130);
    const telemetry = socket.messages("telemetry");
    expect(telemetry.length).toBeGreaterThanOrEqual(2);
    expect(telemetry.at(-1).metrics.fps).toBe(58);
    expect(telemetry.at(-1).metrics.seed).toBe(3);
    socket.push({ type: "settings", version: 4, settings: { food: 2 } });
    expect(events.at(-1)).toEqual({ type: "settings", version: 4, settings: { food: 2 } });
  });
});
