import { PolicyClient, type ConnectionState, type SettingsMessage } from "../api/websocket";
import { OBS_DIM } from "../sim/groups";
import type { ObservationsEvent, SimulationClient } from "./SimulationClient";

export interface PolicyInfo {
  name: string;
  available: boolean;
  kind: string;
  model_version: string;
  label: string;
}

/**
 * Connects the worker's ML observations to the backend and the backend's actions back to the
 * worker. Also forwards low-rate telemetry and dashboard settings.
 */
export class PolicyBridge {
  readonly client: PolicyClient;
  connection: ConnectionState = "closed";
  mlLabel = "ML (not connected)";
  lastError = "";

  constructor(url: string, private readonly sim: SimulationClient, onSettings: (msg: SettingsMessage) => void) {
    this.client = new PolicyClient(url, {
      onActions: (msg) => this.sim.send({ type: "policyActions", requestTick: msg.tick, modelVersion: msg.model_version, actions: msg.actions }),
      onSettings,
      onError: (msg) => { this.lastError = msg.code; },
      onState: (state) => { this.connection = state; },
    });
    sim.onObservations = (event) => this.forward(event);
  }

  start(): void {
    this.client.connect();
    void this.loadPolicyLabel();
  }

  telemetry(tick: number, mode: string, metrics: Record<string, number>): void {
    this.client.sendTelemetry(tick, mode, metrics);
  }

  private forward(event: ObservationsEvent): void {
    this.client.request(event.tick, event.groupIds, event.features, OBS_DIM);
  }

  private async loadPolicyLabel(): Promise<void> {
    try {
      const policies = (await (await fetch("api/policies")).json()) as PolicyInfo[];
      const ml = policies.find((p) => p.name === "ml");
      this.mlLabel = ml?.available ? ml.label : "ML (not loaded)";
    } catch {
      this.mlLabel = "ML (backend unreachable)";
    }
  }
}
