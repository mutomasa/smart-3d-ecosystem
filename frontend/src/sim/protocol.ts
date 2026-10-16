import type { BoidsWeights } from "./config";
import type { EcosystemMetrics } from "./metrics";
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
      /** Predator positions/velocities (x,y,z,vx,vy,vz per predator), copied. */
      predators: Float32Array;
      /** Food patches (x,y,z,amount per patch; amount 0 = depleted), copied. */
      food: Float32Array;
    }
  | { type: "log"; level: "info" | "warn"; message: string };
