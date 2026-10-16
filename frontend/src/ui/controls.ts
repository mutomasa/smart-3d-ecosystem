import GUI from "lil-gui";
import { DEFAULT_WEIGHTS, type BoidsWeights } from "../sim/config";
import { PRESET_NAMES } from "../presets";

export interface ControlState {
  preset: string;
  count: number;
  seed: number;
  paused: boolean;
  weights: BoidsWeights;
}

export interface ControlHandlers {
  onCount(count: number): void;
  onWeights(weights: Partial<BoidsWeights>): void;
  onPause(paused: boolean): void;
  onReset(state: ControlState): void;
  onPreset(name: string): void;
  onDownloadLog(): void;
}

/** lil-gui panel. Only translates UI events to handler calls (no simulation logic here). */
export class Controls {
  readonly gui = new GUI({ title: "Controls" });
  readonly state: ControlState;

  constructor(initial: Omit<ControlState, "weights">, private readonly h: ControlHandlers) {
    this.state = { ...initial, weights: { ...DEFAULT_WEIGHTS } };
    this.buildScenario();
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

  private buildWeights(): void {
    const f = this.gui.addFolder("Boids gains");
    const keys: (keyof BoidsWeights)[] = ["separation", "alignment", "cohesion", "boundary"];
    for (const k of keys) {
      f.add(this.state.weights, k, 0, 5, 0.05).onChange((v: number) => this.h.onWeights({ [k]: v }));
    }
  }

  private buildActions(): void {
    const f = this.gui.addFolder("Run");
    f.add(this.state, "paused").name("Pause").onChange((p: boolean) => this.h.onPause(p));
    f.add({ reset: () => this.h.onReset(this.state) }, "reset").name("Reset (same seed)");
    f.add({ log: () => this.h.onDownloadLog() }, "log").name("Download metrics log");
  }
}
