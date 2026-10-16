import type { BoidsWeights } from "./config";
import type { PolicyMode, WorldSettings } from "./world";

/** Main thread -> worker. */
export type WorkerCommand =
  | { type: "init"; settings: Partial<WorldSettings> }
  | { type: "reset"; settings: Partial<WorldSettings> }
  | { type: "setWeights"; weights: Partial<BoidsWeights> }
  | { type: "setCount"; count: number }
  | { type: "setMode"; mode: PolicyMode }
  | { type: "pause"; paused: boolean }
  | { type: "returnBuffer"; buffer: ArrayBuffer };

export interface StepTelemetry {
  stepMs: number;
  dt: number;
  neighborChecks: number;
  collisions: number;
  maxCellOccupancy: number;
  droppedMs: number;
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
    }
  | { type: "log"; level: "info" | "warn"; message: string };
