import { DEFAULT_CONFIG, type BoidsWeights } from "../sim/config";
import type { WorkerEvent } from "../sim/protocol";
import { PRESETS, type Preset } from "../presets";
import { AquariumScene } from "../scene/Scene";
import { FishRenderer } from "../scene/FishRenderer";
import { FoodRenderer, PredatorRenderer } from "../scene/EntityRenderers";
import { Controls, type ControlState } from "../ui/controls";
import { FpsMeter } from "../ui/fpsMeter";
import { Hud } from "../ui/hud";
import { MetricsLog, downloadText } from "../ui/metricsLog";
import { SimulationClient } from "./SimulationClient";
import { fetchMlLabel } from "./PolicyBridge";
import { SettingsVersionGate, planSettings, type RemoteSettings } from "./settingsSync";
import { experimentIdFromUrl, wsUrl } from "../api/websocket";

type FrameEvent = Extract<WorkerEvent, { type: "frame" }>;

/** Composition root: wires the worker, the scene, the HUD and the controls together. */
export class App {
  private readonly scene: AquariumScene;
  private readonly fish: FishRenderer;
  private readonly predators: PredatorRenderer;
  private readonly foodPatches: FoodRenderer;
  private readonly client: SimulationClient;
  private readonly hud: Hud;
  private readonly fps = new FpsMeter();
  private readonly log = new MetricsLog();
  private readonly controls: Controls;
  private lastFrame: FrameEvent | null = null;
  private mlLabel = "ML";
  private readonly settingsGate = new SettingsVersionGate();
  private lastStats = -Infinity;
  readonly experimentId = experimentIdFromUrl(location.search);
  private preset: Preset = PRESETS["Calm School"];

  constructor(container: HTMLElement, hudEl: HTMLElement) {
    this.scene = new AquariumScene(container, DEFAULT_CONFIG.halfExtents);
    this.fish = new FishRenderer(this.scene.scene, this.preset.count, DEFAULT_CONFIG.groupCount);
    this.predators = new PredatorRenderer(this.scene.scene);
    this.foodPatches = new FoodRenderer(this.scene.scene);
    this.hud = new Hud(hudEl);
    this.client = new SimulationClient((level, msg) => console[level === "warn" ? "warn" : "info"](msg));
    this.controls = new Controls(
      {
        preset: this.preset.name, count: this.preset.count, seed: this.preset.seed, paused: false, adaptiveDt: true,
        mode: this.preset.mode, predators: this.preset.predators, food: this.preset.food,
      },
      {
        onCount: (count) => this.client.send({ type: "setCount", count }),
        onWeights: (weights) => this.client.send({ type: "setWeights", weights }),
        onPause: (paused) => this.client.send({ type: "pause", paused }),
        onReset: (state) => this.reset(state),
        onPreset: (name) => this.applyPreset(PRESETS[name]),
        onDownloadLog: () => this.downloadLog(),
        onMode: (mode) => this.client.send({ type: "setMode", mode }),
        onPredators: (count) => this.client.send({ type: "setPredators", count }),
        onFood: (count) => this.client.send({ type: "setFood", count }),
        onSpawnFood: () => this.client.send({ type: "spawnFood" }),
      },
    );
    this.client.onSettings = (m) => this.applyRemoteSettings(m.version, m.settings as RemoteSettings);
    this.client.send({ type: "connectBackend", url: wsUrl(location, this.experimentId) });
    void fetchMlLabel().then((label) => { this.mlLabel = label; });
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
      this.predators.update(frame.predators);
      this.foodPatches.update(frame.food);
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
    this.hud.update({ ...state, metrics: f.metrics, extra: [
      `Max cell ${f.telemetry.maxCellOccupancy} | Collisions ${f.telemetry.collisions} | dt changes ${f.telemetry.dtChanges}`,
      ...this.policyLines(f),
    ] });
    this.sendTelemetry(now);
    this.log.maybeRecord({ t: now, tick: f.tick, ...state, maxCell: f.telemetry.maxCellOccupancy, dtChanges: f.telemetry.dtChanges,
      survivalRate: f.metrics.survivalRate, foodCollected: f.metrics.foodCollected, meanEnergy: f.metrics.meanEnergy,
      dispersion: f.metrics.dispersion, collisions: f.metrics.collisions, predation: f.metrics.predation });
  }

  private policyLines(f: FrameEvent): string[] {
    const b = f.backend;
    if (!b) return [];
    const latency = b.latencyP50 === null ? "-" : `${b.latencyP50.toFixed(1)}/${b.latencyP95!.toFixed(1)} ms`;
    const lines = [`Backend ${b.connection} | exp ${this.experimentId}`];
    if (f.mode === "ml") {
      lines.push(`Policy ${this.mlLabel} | ${f.policy.status}${f.policy.status === "fallback" ? " -> rules" : ""}`);
      lines.push(`ML latency p50/p95 ${latency} | errors ${b.errors} | stale ${f.policy.staleDropped}`);
    }
    return lines;
  }

  /** The worker sends telemetry itself; it only needs the render FPS from this thread. */
  private sendTelemetry(now: number): void {
    if (now - this.lastStats < 1000) return;
    this.lastStats = now;
    this.client.send({ type: "clientStats", fps: this.fps.fps() });
  }

  private applyRemoteSettings(version: number, settings: RemoteSettings): void {
    if (!this.settingsGate.accept(version)) return;
    const plan = planSettings(settings, this.controls.state);
    if (plan.preset) return this.applyPreset(PRESETS[plan.preset]);
    this.controls.refresh();
    if (plan.reset) return this.reset(this.controls.state);
    plan.commands.forEach((c) => this.client.send(c));
  }

  private applyPreset(preset: Preset | undefined): void {
    if (!preset) return;
    this.preset = preset;
    Object.assign(this.controls.state, {
      preset: preset.name, count: preset.count, seed: preset.seed,
      mode: preset.mode, predators: preset.predators, food: preset.food,
    });
    this.controls.refresh();
    this.reset(this.controls.state, preset.weights);
  }

  private reset(state: ControlState, presetWeights: Partial<BoidsWeights> = {}): void {
    const weights = { ...state.weights, ...presetWeights };
    Object.assign(state.weights, weights);
    this.controls.refresh();
    this.log.clear();
    this.client.send({
      type: "reset",
      settings: {
        count: state.count, seed: state.seed, weights, adaptiveDt: state.adaptiveDt,
        mode: state.mode, predators: state.predators, food: state.food,
      },
    });
    this.client.send({ type: "pause", paused: state.paused });
  }

  private downloadLog(): void {
    const meta = { preset: this.preset.name, seed: this.controls.state.seed, gpu: this.scene.gpuName(), userAgent: navigator.userAgent };
    downloadText(`metrics-${Date.now()}.json`, this.log.toJSON(meta));
  }
}
