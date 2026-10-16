import type { BoidsWeights } from "../sim/config";
import type { WorkerCommand } from "../sim/protocol";
import type { ControlState } from "../ui/controls";

/** Settings pushed by the dashboard (all optional). */
export interface RemoteSettings {
  preset?: string;
  count?: number;
  seed?: number;
  mode?: ControlState["mode"];
  predators?: number;
  food?: number;
  paused?: boolean;
  weights?: Partial<BoidsWeights>;
}

/**
 * Applies only versions newer than the last one seen (the server is last-writer-wins, so a
 * replayed or reordered push can never roll settings back).
 */
export class SettingsVersionGate {
  private last = 0;

  accept(version: number): boolean {
    if (version <= this.last) return false;
    this.last = version;
    return true;
  }
}

export interface SettingsPlan {
  /** Seed / population changes need a reset of the world (reproducible runs). */
  reset: boolean;
  preset?: string;
  commands: WorkerCommand[];
}

/** Merges remote settings into the control state and lists the worker commands to send. */
export function planSettings(s: RemoteSettings, state: ControlState): SettingsPlan {
  if (s.preset) return { reset: false, preset: s.preset, commands: [] };
  const reset = (s.seed !== undefined && s.seed !== state.seed) || (s.count !== undefined && s.count !== state.count);
  Object.assign(state, pick(s, ["count", "seed", "mode", "predators", "food", "paused"]));
  if (s.weights) Object.assign(state.weights, s.weights);
  if (reset) return { reset, commands: [] };
  const commands: WorkerCommand[] = [];
  if (s.mode) commands.push({ type: "setMode", mode: s.mode });
  if (s.predators !== undefined) commands.push({ type: "setPredators", count: s.predators });
  if (s.food !== undefined) commands.push({ type: "setFood", count: s.food });
  if (s.paused !== undefined) commands.push({ type: "pause", paused: s.paused });
  if (s.weights) commands.push({ type: "setWeights", weights: s.weights });
  return { reset, commands };
}

function pick<T extends object, K extends keyof T>(obj: T, keys: K[]): Partial<T> {
  return Object.fromEntries(keys.filter((k) => obj[k] !== undefined).map((k) => [k, obj[k]])) as unknown as Partial<T>;
}
