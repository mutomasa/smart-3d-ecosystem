import type { BoidsWeights } from "./config";
import type { EcosystemMetrics } from "./metrics";
import type { GroupActionInput, MlStatus } from "./mlPolicy";
import type { PolicyMode, WorldSettings } from "./world";

/** Main thread -> worker. */
export type WorkerCommand =
  | { type: "init"; settings: Partial<WorldSettings> }
  | { type: "reset"; settings: Partial<WorldSettings> }
  | { type: "setWeights"; weights: Partial<BoidsWeights> }
  | { type: "setCount"; count: number }
  | { type: "setMode"; mode: PolicyMode }
  | { type: "setPredators"; count: number }
  | { type: "setFood"; count: number }
  | { type: "spawnFood" }
  | { type: "policyActions"; requestTick: number; modelVersion: string; actions: GroupActionInput[] }
  | { type: "connectBackend"; url: string }
  | { type: "clientStats"; fps: number }
  | { type: "pause"; paused: boolean }
  | { type: "returnBuffer"; buffer: ArrayBuffer };

export interface StepTelemetry {
  stepMs: number;
  dt: number;
  neighborChecks: number;
  collisions: number;
  maxCellOccupancy: number;
  droppedMs: number;
  /** Number of adaptive dt switches since reset (0 => the run used a single fixed dt). */
  dtChanges: number;
}

/** Worker -> main thread. `buffer` is transferred and must be returned with `returnBuffer`. */
export type WorkerEvent =
  | {
      type: "frame";
      tick: number;
      count: number;
      alive: number;
      mode: PolicyMode;
      buffer: ArrayBuffer;
      telemetry: StepTelemetry;
      metrics: EcosystemMetrics;
      policy: PolicyState;
      backend: BackendState | null;
      /** Predator positions/velocities (x,y,z,vx,vy,vz per predator), copied. */
      predators: Float32Array;
      /** Food patches (x,y,z,amount per patch; amount 0 = depleted), copied. */
      food: Float32Array;
    }
  | { type: "log"; level: "info" | "warn"; message: string }
  | { type: "settings"; version: number; settings: Record<string, unknown> }
  | {
      /** ML mode only, when no backend link is attached: raw observations of the non-empty groups. */
      type: "observations";
      tick: number;
      groupIds: number[];
      features: Float32Array;
    };

export interface BackendState {
  connection: "connecting" | "open" | "closed";
  latencyP50: number | null;
  latencyP95: number | null;
  errors: number;
}

export interface PolicyState {
  status: MlStatus | "local";
  modelVersion: string;
  staleDropped: number;
}
