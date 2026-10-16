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

export interface AdaptiveOptions {
  enabled: boolean;
  /** Fraction of the 1/60 frame budget above which dt is relaxed to 1/30. */
  budget: number;
  /** Observations ignored at start-up (JIT warm-up). */
  warmup: number;
  /** Consecutive over/under-budget observations required before switching. */
  patience: number;
}

const DEFAULT_ADAPTIVE: AdaptiveOptions = { enabled: true, budget: 0.6, warmup: 120, patience: 60 };

/**
 * Chooses dt = 1/60 normally and 1/30 when the measured step cost stays above `budget` of the
 * 1/60 frame budget. Start-up samples are ignored and a switch needs `patience` consecutive
 * observations, so JIT warm-up spikes never change dt. Disable it for reproducible comparisons.
 */
export class AdaptiveTimestep {
  static readonly FAST = 1 / 60;
  static readonly SLOW = 1 / 30;
  private readonly opts: AdaptiveOptions;
  private seen = 0;
  private streak = 0;
  dt = AdaptiveTimestep.FAST;

  constructor(options: Partial<AdaptiveOptions> = {}) {
    this.opts = { ...DEFAULT_ADAPTIVE, ...options };
  }

  /** Feeds the cost of one tick (ms); returns true when dt changed. */
  observe(stepMs: number): boolean {
    this.seen++;
    if (!this.opts.enabled || this.seen <= this.opts.warmup) return false;
    const wanted = this.wantedDt(stepMs);
    this.streak = wanted === this.dt ? 0 : this.streak + 1;
    if (this.streak < this.opts.patience) return false;
    this.dt = wanted;
    this.streak = 0;
    return true;
  }

  private wantedDt(stepMs: number): number {
    const fastBudget = AdaptiveTimestep.FAST * 1000 * this.opts.budget;
    if (stepMs > fastBudget) return AdaptiveTimestep.SLOW;
    if (stepMs < fastBudget * 0.5) return AdaptiveTimestep.FAST;
    return this.dt;
  }
}
