import GUI from "lil-gui";
import { DEFAULT_WEIGHTS, type BoidsWeights } from "../sim/config";
import { PRESET_NAMES } from "../presets";
import type { PolicyMode } from "../sim/policies";

export interface ControlState {
  preset: string;
  count: number;
  seed: number;
  paused: boolean;
  /** Off => dt fixed at 1/60 for reproducible comparison runs. Applied on reset. */
  adaptiveDt: boolean;
  mode: PolicyMode;
  predators: number;
  food: number;
  weights: BoidsWeights;
}

export interface ControlHandlers {
  onCount(count: number): void;
  onWeights(weights: Partial<BoidsWeights>): void;
  onPause(paused: boolean): void;
  onReset(state: ControlState): void;
  onPreset(name: string): void;
  onMode(mode: PolicyMode): void;
  onPredators(n: number): void;
  onFood(n: number): void;
  onSpawnFood(): void;
  onDownloadLog(): void;
}

/** lil-gui panel. Only translates UI events to handler calls (no simulation logic here). */
export class Controls {
  readonly gui = new GUI({ title: "Controls" });
  readonly state: ControlState;

  constructor(initial: Omit<ControlState, "weights">, private readonly h: ControlHandlers) {
    this.state = { ...initial, weights: { ...DEFAULT_WEIGHTS } };
    this.buildScenario();
    this.buildEcosystem();
    this.buildWeights();
    this.buildActions();
  }

  refresh(): void {
    this.gui.controllersRecursive().forEach((c) => c.updateDisplay());
  }

  private buildScenario(): void {
    const f = this.gui.addFolder("Scenario");
    f.add(this.state, "preset", PRESET_NAMES).name("Preset").onChange((n: string) => this.h.onPreset(n));
    f.add(this.state, "count", 0, 5000, 50).name("Population").onFinishChange((n: number) => this.h.onCount(n));
    f.add(this.state, "seed", 0, 99999, 1).name("Seed");
  }

  private buildEcosystem(): void {
    const f = this.gui.addFolder("Ecosystem");
    f.add(this.state, "mode", ["boids", "rules", "ml"]).name("Policy").onChange((m: PolicyMode) => this.h.onMode(m));
    f.add(this.state, "predators", 0, 5, 1).name("Predators").onFinishChange((n: number) => this.h.onPredators(n));
    f.add(this.state, "food", 0, 10, 1).name("Food patches").onFinishChange((n: number) => this.h.onFood(n));
    f.add({ spawn: () => this.h.onSpawnFood() }, "spawn").name("Drop food");
  }

  private buildWeights(): void {
    const f = this.gui.addFolder("Boids gains");
    const keys: (keyof BoidsWeights)[] = ["separation", "alignment", "cohesion", "boundary", "predator", "food", "ml"];
    for (const k of keys) {
      f.add(this.state.weights, k, 0, 5, 0.05).onChange((v: number) => this.h.onWeights({ [k]: v }));
    }
  }

  private buildActions(): void {
    const f = this.gui.addFolder("Run");
    f.add(this.state, "adaptiveDt").name("Adaptive dt (on reset)");
    f.add(this.state, "paused").name("Pause").onChange((p: boolean) => this.h.onPause(p));
    f.add({ reset: () => this.h.onReset(this.state) }, "reset").name("Reset (same seed)");
    f.add({ log: () => this.h.onDownloadLog() }, "log").name("Download metrics log");
  }
}
