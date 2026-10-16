import { PolicyClient, type SettingsMessage, type SocketFactory } from "../api/websocket";
import { OBS_DIM } from "./groups";
import type { BackendState } from "./protocol";
import type { World } from "./world";

const TELEMETRY_INTERVAL_MS = 1000;

/**
 * Worker-side connection to the backend. Observations, actions and telemetry never touch the
 * main thread, so ML latency is not affected by rendering load (#13).
 */
export class BackendLink {
  readonly client: PolicyClient;
  private lastTelemetry = -Infinity;
  private world: World | null = null;
  clientFps = 0;

  constructor(url: string, onSettings: (msg: SettingsMessage) => void, factory?: SocketFactory) {
    this.client = new PolicyClient(url, {
      onActions: (msg) => this.world?.mlPolicy?.receive(msg.tick, msg.actions, msg.model_version),
      onSettings,
      onError: () => {},
      onState: () => {},
    }, factory);
  }

  attach(world: World): void {
    this.world = world;
  }

  connect(): void {
    this.client.connect();
  }

  request(tick: number, groupIds: number[], features: Float32Array): void {
    this.client.request(tick, groupIds, features, OBS_DIM);
  }

  maybeSendTelemetry(nowMs: number, world: World, stepMs: number, seed: number): void {
    if (nowMs - this.lastTelemetry < TELEMETRY_INTERVAL_MS) return;
    this.lastTelemetry = nowMs;
    const p50 = this.client.latencyPercentile(50), p95 = this.client.latencyPercentile(95);
    const ml = world.mlPolicy;
    this.client.sendTelemetry(world.tick, world.mode, {
      ...world.metricsSnapshot(), fps: this.clientFps, stepMs, seed,
      mlFallback: ml?.status === "fallback" ? 1 : 0,
      ...(p50 === null ? {} : { mlLatencyP50: p50, mlLatencyP95: p95! }),
    });
  }

  state(): BackendState {
    return {
      connection: this.client.state,
      latencyP50: this.client.latencyPercentile(50),
      latencyP95: this.client.latencyPercentile(95),
      errors: this.client.errors,
    };
  }
}
