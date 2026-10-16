import { DEFAULT_CONFIG, type BoidsWeights } from "../sim/config";
import type { WorkerEvent } from "../sim/protocol";
import { PRESETS, type Preset } from "../presets";
import { AquariumScene } from "../scene/Scene";
import { FishRenderer } from "../scene/FishRenderer";
import { Controls, type ControlState } from "../ui/controls";
import { FpsMeter } from "../ui/fpsMeter";
import { Hud } from "../ui/hud";
import { MetricsLog, downloadText } from "../ui/metricsLog";
import { SimulationClient } from "./SimulationClient";

type FrameEvent = Extract<WorkerEvent, { type: "frame" }>;

/** Composition root: wires the worker, the scene, the HUD and the controls together. */
export class App {
  private readonly scene: AquariumScene;
  private readonly fish: FishRenderer;
  private readonly client: SimulationClient;
  private readonly hud: Hud;
  private readonly fps = new FpsMeter();
  private readonly log = new MetricsLog();
  private readonly controls: Controls;
  private lastFrame: FrameEvent | null = null;
  private preset: Preset = PRESETS["Calm School"];

  constructor(container: HTMLElement, hudEl: HTMLElement) {
    this.scene = new AquariumScene(container, DEFAULT_CONFIG.halfExtents);
    this.fish = new FishRenderer(this.scene.scene, this.preset.count, DEFAULT_CONFIG.groupCount);
    this.hud = new Hud(hudEl);
    this.client = new SimulationClient((level, msg) => console[level === "warn" ? "warn" : "info"](msg));
    this.controls = new Controls(
      { preset: this.preset.name, count: this.preset.count, seed: this.preset.seed, paused: false },
      {
        onCount: (count) => this.client.send({ type: "setCount", count }),
        onWeights: (weights) => this.client.send({ type: "setWeights", weights }),
        onPause: (paused) => this.client.send({ type: "pause", paused }),
        onReset: (state) => this.reset(state),
        onPreset: (name) => this.applyPreset(PRESETS[name]),
        onDownloadLog: () => this.downloadLog(),
      },
    );
    this.applyPreset(this.preset);
  }

  start(): void {
    const loop = (now: number) => {
      this.frame(now);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  private frame(now: number): void {
    this.client.consume((frame, data) => {
      this.fish.update(data, frame.count);
      this.lastFrame = frame;
    });
    this.scene.render();
    this.fps.frame(now);
    this.report(now);
  }

  private report(now: number): void {
    const f = this.lastFrame;
    if (!f) return;
    const state = {
      fps: this.fps.fps(), agents: f.count, alive: f.alive, mode: f.mode,
      stepMs: f.telemetry.stepMs, dt: f.telemetry.dt,
    };
    this.hud.update({ ...state, extra: [`Max cell ${f.telemetry.maxCellOccupancy} | Collisions ${f.telemetry.collisions}`] });
    this.log.maybeRecord({ t: now, tick: f.tick, ...state, maxCell: f.telemetry.maxCellOccupancy, collisions: f.telemetry.collisions });
  }

  private applyPreset(preset: Preset | undefined): void {
    if (!preset) return;
    this.preset = preset;
    Object.assign(this.controls.state, { preset: preset.name, count: preset.count, seed: preset.seed });
    this.controls.refresh();
    this.reset(this.controls.state, preset.weights);
  }

  private reset(state: ControlState, presetWeights: Partial<BoidsWeights> = {}): void {
    const weights = { ...state.weights, ...presetWeights };
    Object.assign(state.weights, weights);
    this.controls.refresh();
    this.log.clear();
    this.client.send({ type: "reset", settings: { count: state.count, seed: state.seed, weights } });
    this.client.send({ type: "setMode", mode: this.preset.mode });
    this.client.send({ type: "pause", paused: state.paused });
  }

  private downloadLog(): void {
    const meta = { preset: this.preset.name, seed: this.controls.state.seed, gpu: this.scene.gpuName(), userAgent: navigator.userAgent };
    downloadText(`metrics-${Date.now()}.json`, this.log.toJSON(meta));
  }
}
