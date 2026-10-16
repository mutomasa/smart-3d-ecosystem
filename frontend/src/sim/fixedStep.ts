/**
 * Fixed-timestep accumulator. `advance(now)` returns how many simulation steps to run so that
 * simulated time follows wall time, capped by `maxStepsPerTick` so that a stall does not cause
 * a spiral of death (the backlog is dropped instead).
 */
export class FixedStepClock {
  private accumulator = 0;
  private last: number | null = null;
  droppedMs = 0;

  constructor(public dtMs: number, private readonly maxStepsPerTick = 4) {}

  advance(nowMs: number): number {
    if (this.last === null) {
      this.last = nowMs;
      return 0;
    }
    this.accumulator += nowMs - this.last;
    this.last = nowMs;
    const steps = Math.min(this.maxStepsPerTick, Math.floor(this.accumulator / this.dtMs));
    this.accumulator -= steps * this.dtMs;
    this.dropBacklog();
    return steps;
  }

  reset(): void {
    this.accumulator = 0;
    this.last = null;
  }

  private dropBacklog(): void {
    if (this.accumulator <= this.dtMs * this.maxStepsPerTick) return;
    this.droppedMs += this.accumulator - this.dtMs;
    this.accumulator = this.dtMs;
  }
}

/**
 * Chooses dt = 1/60 normally and 1/30 when the measured step cost stays above `budget` of the
 * 1/60 frame budget (and back again with hysteresis).
 */
export class AdaptiveTimestep {
  static readonly FAST = 1 / 60;
  static readonly SLOW = 1 / 30;
  private ema = 0;
  dt = AdaptiveTimestep.FAST;

  constructor(private readonly enabled = true, private readonly budget = 0.6) {}

  /** Feeds the cost of one tick (ms); returns true when dt changed. */
  observe(stepMs: number): boolean {
    this.ema = this.ema === 0 ? stepMs : this.ema * 0.95 + stepMs * 0.05;
    if (!this.enabled) return false;
    const fastBudget = AdaptiveTimestep.FAST * 1000 * this.budget;
    const next = this.ema > fastBudget ? AdaptiveTimestep.SLOW
      : this.ema < fastBudget * 0.5 ? AdaptiveTimestep.FAST : this.dt;
    const changed = next !== this.dt;
    this.dt = next;
    return changed;
  }
}
