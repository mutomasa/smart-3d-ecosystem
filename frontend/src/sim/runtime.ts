import { BufferPool, frameFloats } from "./frame";
import { AdaptiveTimestep, FixedStepClock } from "./fixedStep";
import { OBS_DIM } from "./groups";
import type { PolicyState, StepTelemetry, WorkerCommand, WorkerEvent } from "./protocol";
import { World, type WorldSettings } from "./world";

export type PostFn = (event: WorkerEvent, transfer: Transferable[]) => void;

const MAX_BUFFERS = 3;

/**
 * Worker-side runtime: owns the World, the fixed-step clock and the ping-pong buffers.
 * Independent of the worker global so it can be unit-tested with a fake `post` and clock.
 */
export class SimulationRuntime {
  world: World | null = null;
  paused = false;
  private readonly pool = new BufferPool();
  private inFlight = 0;
  private timestep = new AdaptiveTimestep({ enabled: false });
  private clock = new FixedStepClock(this.timestep.dt * 1000);
  private lastStepMs = 0;

  private adaptive = true;
  private dtChanges = 0;
  private emittedDecisions = 0;

  constructor(private readonly post: PostFn) {}

  handle(cmd: WorkerCommand): void {
    switch (cmd.type) {
      case "init":
      case "reset":
        return this.reset(cmd.settings);
      case "setWeights":
        return this.world?.setWeights(cmd.weights);
      case "setCount":
        return this.world?.setCount(cmd.count);
      case "setMode":
        return this.world?.setMode(cmd.mode);
      case "setPredators":
        return this.world?.setPredators(cmd.count);
      case "setFood":
        return this.world?.setFood(cmd.count);
      case "spawnFood":
        return this.world?.spawnFood();
      case "policyActions":
        this.world?.mlPolicy?.receive(cmd.requestTick, cmd.actions, cmd.modelVersion);
        return;
      case "pause":
        return this.setPaused(cmd.paused);
      case "returnBuffer":
        this.inFlight = Math.max(0, this.inFlight - 1);
        return this.pool.release(cmd.buffer);
    }
  }

  /** Advances the simulation to wall time `nowMs` and publishes a frame if anything changed. */
  tick(nowMs: number): void {
    if (!this.world || this.paused) return;
    const steps = this.clock.advance(nowMs);
    if (steps === 0) return;
    this.runSteps(this.world, steps);
    this.publishObservations(this.world);
    this.publishFrame(this.world);
  }

  /** Emits the observation batch once per new policy decision while in ML mode. */
  private publishObservations(world: World): void {
    if (!world.mlPolicy || world.decisions === this.emittedDecisions) return;
    this.emittedDecisions = world.decisions;
    const groupIds: number[] = [];
    for (let g = 0; g < world.groups.groupCount; g++) if (world.groups.size[g] > 0) groupIds.push(g);
    const features = new Float32Array(groupIds.length * OBS_DIM);
    groupIds.forEach((g, k) => features.set(world.observations.subarray(g * OBS_DIM, (g + 1) * OBS_DIM), k * OBS_DIM));
    this.post({ type: "observations", tick: world.tick, groupIds, features }, [features.buffer]);
  }

  private reset(settings: Partial<WorldSettings>): void {
    this.world = World.create(settings);
    this.emittedDecisions = 0;
    this.adaptive = settings.adaptiveDt ?? true;
    this.dtChanges = 0;
    this.timestep = new AdaptiveTimestep({ enabled: this.adaptive });
    this.clock = new FixedStepClock(this.timestep.dt * 1000);
  }

  private setPaused(paused: boolean): void {
    this.paused = paused;
    if (!paused) this.clock.reset();
  }

  private runSteps(world: World, steps: number): void {
    const t0 = performance.now();
    for (let s = 0; s < steps; s++) world.step(this.timestep.dt);
    this.lastStepMs = (performance.now() - t0) / steps;
    if (!this.timestep.observe(this.lastStepMs)) return;
    this.clock.dtMs = this.timestep.dt * 1000;
    this.dtChanges++;
    const message = `adaptive dt -> 1/${Math.round(1 / this.timestep.dt)} s (step ${this.lastStepMs.toFixed(2)} ms)`;
    this.post({ type: "log", level: "warn", message }, []);
  }

  private publishFrame(world: World): void {
    const count = world.sim.state.count;
    const buffer = this.pool.acquire(frameFloats(count) * 4, MAX_BUFFERS, this.inFlight);
    if (!buffer) return; // main thread is behind; skip this frame
    world.packFish(new Float32Array(buffer, 0, frameFloats(count)));
    this.inFlight++;
    const event: WorkerEvent = {
      type: "frame", tick: world.tick, count, alive: world.aliveCount(), mode: world.mode,
      buffer, telemetry: this.telemetry(world), metrics: world.metricsSnapshot(), policy: policyState(world),
      predators: packPredators(world), food: packFood(world),
    };
    this.post(event, [buffer]);
  }

  private telemetry(world: World): StepTelemetry {
    const { stats } = world.sim;
    return {
      stepMs: this.lastStepMs,
      dt: this.timestep.dt,
      neighborChecks: stats.neighborChecks,
      collisions: stats.collisions,
      maxCellOccupancy: stats.maxCellOccupancy,
      droppedMs: this.clock.droppedMs,
      dtChanges: this.dtChanges,
    };
  }
}

function policyState(world: World): PolicyState {
  const ml = world.mlPolicy;
  if (!ml) return { status: "local", modelVersion: world.mode, staleDropped: 0 };
  return { status: ml.status, modelVersion: ml.modelVersion, staleDropped: ml.staleDropped };
}

function packPredators(world: World): Float32Array {
  const { predators } = world;
  const out = new Float32Array(predators.count * 6);
  for (let p = 0; p < predators.count; p++) {
    out.set(predators.pos.subarray(p * 3, p * 3 + 3), p * 6);
    out.set(predators.vel.subarray(p * 3, p * 3 + 3), p * 6 + 3);
  }
  return out;
}

function packFood(world: World): Float32Array {
  const { food } = world;
  const out = new Float32Array(food.count * 4);
  for (let f = 0; f < food.count; f++) {
    out.set(food.pos.subarray(f * 3, f * 3 + 3), f * 4);
    out[f * 4 + 3] = food.amount[f];
  }
  return out;
}
