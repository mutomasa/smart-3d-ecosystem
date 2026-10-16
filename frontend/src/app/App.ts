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
import { PolicyBridge } from "./PolicyBridge";
import { SettingsVersionGate, planSettings, type RemoteSettings } from "./settingsSync";
import { experimentIdFromUrl, wsUrl, type SettingsMessage } from "../api/websocket";

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
  private readonly bridge: PolicyBridge;
  private readonly settingsGate = new SettingsVersionGate();
  private lastTelemetry = -Infinity;
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
    this.bridge = new PolicyBridge(wsUrl(location, this.experimentId), this.client, (m) => this.applyRemoteSettings(m));
    this.bridge.start();
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
    this.sendTelemetry(now, f);
    this.log.maybeRecord({ t: now, tick: f.tick, ...state, maxCell: f.telemetry.maxCellOccupancy, dtChanges: f.telemetry.dtChanges,
      survivalRate: f.metrics.survivalRate, foodCollected: f.metrics.foodCollected, meanEnergy: f.metrics.meanEnergy,
      dispersion: f.metrics.dispersion, collisions: f.metrics.collisions, predation: f.metrics.predation });
  }

  private policyLines(f: FrameEvent): string[] {
    const b = this.bridge;
    const p50 = b.client.latencyPercentile(50), p95 = b.client.latencyPercentile(95);
    const latency = p50 === null ? "-" : `${p50.toFixed(1)}/${p95!.toFixed(1)} ms`;
    const lines = [`Backend ${b.connection} | exp ${this.experimentId}`];
    if (f.mode === "ml") {
      lines.push(`Policy ${b.mlLabel} | ${f.policy.status}${f.policy.status === "fallback" ? " -> rules" : ""}`);
      lines.push(`ML latency p50/p95 ${latency} | errors ${b.client.errors} | stale ${f.policy.staleDropped}`);
    }
    return lines;
  }

  private sendTelemetry(now: number, f: FrameEvent): void {
    if (now - this.lastTelemetry < 1000) return;
    this.lastTelemetry = now;
    const p50 = this.bridge.client.latencyPercentile(50), p95 = this.bridge.client.latencyPercentile(95);
    this.bridge.telemetry(f.tick, f.mode, {
      ...f.metrics, fps: this.fps.fps(), stepMs: f.telemetry.stepMs, seed: this.controls.state.seed,
      mlFallback: f.policy.status === "fallback" ? 1 : 0,
      ...(p50 === null ? {} : { mlLatencyP50: p50, mlLatencyP95: p95! }),
    });
  }

  private applyRemoteSettings(msg: SettingsMessage): void {
    if (!this.settingsGate.accept(msg.version)) return;
    const plan = planSettings(msg.settings as RemoteSettings, this.controls.state);
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
