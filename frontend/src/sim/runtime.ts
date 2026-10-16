import { BufferPool, frameFloats } from "./frame";
import { AdaptiveTimestep, FixedStepClock } from "./fixedStep";
import type { StepTelemetry, WorkerCommand, WorkerEvent } from "./protocol";
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
        if (this.world) this.world.mode = cmd.mode;
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
    this.publishFrame(this.world);
  }

  private reset(settings: Partial<WorldSettings>): void {
    this.world = World.create(settings);
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
      buffer, telemetry: this.telemetry(world),
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
