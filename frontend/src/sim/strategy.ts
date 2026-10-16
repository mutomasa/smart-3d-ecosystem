import type { SimConfig } from "./config";
import type { BoidState } from "./state";
import { removeOutward } from "./steering";
import { EPSILON, length3 } from "./vec3";

/** Index of each weight in a group strategy (matches the action contract order). */
export const W_PREDATOR = 0;
export const W_FOOD = 1;
export const W_COHESION = 2;

/**
 * Per-group high-level strategy chosen by a policy: desired direction (unit or zero) and
 * three bounded weights in [0, 1]. `target*` is what the policy asked for; `current*` follows it
 * with a first-order low-pass filter (time constant `tau`) to avoid abrupt switches.
 */
export class GroupStrategies {
  readonly targetDir: Float32Array;
  readonly targetW: Float32Array;
  readonly dir: Float32Array;
  readonly w: Float32Array;

  constructor(readonly groupCount: number, private readonly tau = 0.2) {
    this.targetDir = new Float32Array(groupCount * 3);
    this.targetW = new Float32Array(groupCount * 3);
    this.dir = new Float32Array(groupCount * 3);
    this.w = new Float32Array(groupCount * 3);
  }

  /**
   * Sets the target for group `g`. The direction is clamped to at most unit length (its magnitude
   * is the preference strength; zero = none) and the weights are clamped to [0, 1].
   */
  setTarget(g: number, dx: number, dy: number, dz: number, wPred: number, wFood: number, wCoh: number): void {
    if (g < 0 || g >= this.groupCount) return;
    const len = length3(dx, dy, dz);
    const s = !Number.isFinite(len) ? 0 : len > 1 ? 1 / len : 1;
    this.targetDir.set([dx * s, dy * s, dz * s], g * 3);
    this.targetW.set([clamp01(wPred), clamp01(wFood), clamp01(wCoh)], g * 3);
  }

  /** Jumps immediately to the targets (used on reset / mode change). */
  snap(): void {
    this.dir.set(this.targetDir);
    this.w.set(this.targetW);
  }

  smooth(dt: number): void {
    const a = 1 - Math.exp(-dt / this.tau);
    for (let k = 0; k < this.dir.length; k++) {
      this.dir[k] += (this.targetDir[k] - this.dir[k]) * a;
      this.w[k] += (this.targetW[k] - this.w[k]) * a;
    }
  }

  weight(g: number, which: number): number {
    return this.w[g * 3 + which];
  }
}

function clamp01(v: number): number {
  return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;
}

/**
 * Applies the group direction as an extra steering term (weight `w.ml`) and exposes the
 * per-group cohesion multiplier (cohesion weight 0.5 => x1).
 */
export class StrategySteering {
  private readonly desired = new Float32Array(3);

  constructor(private readonly strategies: GroupStrategies) {}

  contribute(i: number, state: BoidState, cfg: SimConfig, out: Float32Array): void {
    const g = state.group[i] * 3;
    const s = this.strategies.dir;
    const len = length3(s[g], s[g + 1], s[g + 2]);
    if (len < 1e-3) return;
    const o = i * 3;
    const k = cfg.weights.ml * cfg.maxForce * Math.min(1, len);
    const desired = this.desired;
    desired.set([s[g] / len, s[g + 1] / len, s[g + 2] / len]);
    removeOutward(desired, state.pos[o], state.pos[o + 1], state.pos[o + 2], cfg);
    // Steer the velocity towards the desired direction at max speed.
    const dx = desired[0] * cfg.maxSpeed - state.vel[o];
    const dy = desired[1] * cfg.maxSpeed - state.vel[o + 1];
    const dz = desired[2] * cfg.maxSpeed - state.vel[o + 2];
    const dl = length3(dx, dy, dz);
    if (dl < EPSILON) return;
    out[0] += (dx / dl) * k;
    out[1] += (dy / dl) * k;
    out[2] += (dz / dl) * k;
  }

  cohesionScale(group: number): number {
    return 2 * this.strategies.weight(group, W_COHESION);
  }
}
