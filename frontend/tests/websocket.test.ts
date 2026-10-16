import { describe, expect, it } from "vitest";
import { PolicyClient, experimentIdFromUrl, type SocketLike, type PolicyClientHandlers } from "../src/api/websocket";
import { SettingsVersionGate, planSettings } from "../src/app/settingsSync";
import { DEFAULT_WEIGHTS } from "../src/sim/config";

class FakeSocket implements SocketLike {
  readyState = 0;
  sent: string[] = [];
  onopen: SocketLike["onopen"] = null;
  onclose: SocketLike["onclose"] = null;
  onerror: SocketLike["onerror"] = null;
  onmessage: SocketLike["onmessage"] = null;
  send(d: string) { this.sent.push(d); }
  close() { this.readyState = 3; this.onclose?.({}); }
  open() { this.readyState = 1; this.onopen?.({}); }
  push(msg: object) { this.onmessage?.({ data: JSON.stringify(msg) }); }
}

function setup() {
  const sockets: FakeSocket[] = [];
  const scheduled: { fn: () => void; ms: number }[] = [];
  const got = { actions: [] as unknown[], settings: [] as unknown[], errors: [] as unknown[], states: [] as string[] };
  const handlers: PolicyClientHandlers = {
    onActions: (m) => got.actions.push(m),
    onSettings: (m) => got.settings.push(m),
    onError: (m) => got.errors.push(m),
    onState: (s) => got.states.push(s),
  };
  let now = 0;
  const client = new PolicyClient("ws://x", handlers, () => { const s = new FakeSocket(); sockets.push(s); return s; },
    (fn, ms) => scheduled.push({ fn, ms }), () => now);
  return { client, sockets, scheduled, got, tick: (ms: number) => (now += ms) };
}

describe("PolicyClient", () => {
  it("sends requests only while open and measures round-trip latency", () => {
    const { client, sockets, got, tick } = setup();
    client.connect();
    expect(client.request(1, [0], new Float32Array(20), 20)).toBe(false);
    sockets[0].open();
    expect(client.request(5, [0, 3], new Float32Array(40), 20)).toBe(true);
    const req = JSON.parse(sockets[0].sent[0]);
    expect(req).toMatchObject({ type: "policy_request", tick: 5, policy: "ml" });
    expect(req.observations.map((o: { group_id: number }) => o.group_id)).toEqual([0, 3]);
    expect(req.observations[1].features).toHaveLength(20);
    tick(12);
    sockets[0].push({ type: "policy_response", request_id: req.request_id, tick: 5, policy: "ml", model_version: "v", latency_ms: 1, actions: [] });
    expect(got.actions).toHaveLength(1);
    expect(client.latencyPercentile(50)).toBe(12);
  });

  it("reconnects with exponential backoff after a disconnect", () => {
    const { client, sockets, scheduled, got } = setup();
    client.connect();
    sockets[0].open();
    sockets[0].close();
    expect(got.states).toEqual(["connecting", "open", "closed"]);
    expect(scheduled[0].ms).toBe(500);
    scheduled[0].fn();
    sockets[1].close();
    expect(scheduled[1].ms).toBe(1000);
    expect(sockets).toHaveLength(2);
  });

  it("does not reconnect after close() and counts server errors", () => {
    const { client, sockets, scheduled, got } = setup();
    client.connect();
    sockets[0].open();
    sockets[0].push({ type: "error", request_id: "req-1", code: "timeout", message: "", fallback: "rules" });
    expect(client.errors).toBe(1);
    expect(got.errors).toHaveLength(1);
    client.close();
    expect(scheduled).toHaveLength(0);
  });

  it("forwards settings and ignores malformed messages", () => {
    const { client, sockets, got } = setup();
    client.connect();
    sockets[0].open();
    sockets[0].onmessage?.({ data: "{oops" });
    sockets[0].push({ type: "settings", version: 2, settings: { food: 3 } });
    expect(got.settings).toEqual([{ type: "settings", version: 2, settings: { food: 3 } }]);
  });
});

describe("settings sync", () => {
  const state = () => ({ preset: "Calm School", count: 100, seed: 1, paused: false, adaptiveDt: true, mode: "boids" as const, predators: 0, food: 0, weights: { ...DEFAULT_WEIGHTS } });

  it("accepts only newer versions", () => {
    const gate = new SettingsVersionGate();
    expect(gate.accept(1)).toBe(true);
    expect(gate.accept(1)).toBe(false);
    expect(gate.accept(3)).toBe(true);
    expect(gate.accept(2)).toBe(false);
  });

  it("maps live settings to worker commands", () => {
    const s = state();
    const plan = planSettings({ mode: "ml", predators: 2, weights: { food: 2 } }, s);
    expect(plan.reset).toBe(false);
    expect(plan.commands.map((c) => c.type)).toEqual(["setMode", "setPredators", "setWeights"]);
    expect(s.mode).toBe("ml");
    expect(s.weights.food).toBe(2);
  });

  it("requires a reset when the seed or population changes", () => {
    const s = state();
    expect(planSettings({ seed: 9 }, s).reset).toBe(true);
    expect(s.seed).toBe(9);
    expect(planSettings({ preset: "Food Rush" }, s).preset).toBe("Food Rush");
  });

  it("parses and validates the experiment id from the URL", () => {
    expect(experimentIdFromUrl("?experiment=run-1")).toBe("run-1");
    expect(experimentIdFromUrl("?experiment=../../x")).toBe("default");
    expect(experimentIdFromUrl("")).toBe("default");
  });
});
