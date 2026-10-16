import type { SimConfig } from "./config";
import type { Rng } from "./rng";
import type { BoidState } from "./state";
import { W_PREDATOR, type GroupStrategies } from "./strategy";
import { EPSILON, length3 } from "./vec3";

export interface PredatorConfig {
  speed: number;
  turnAccel: number;
  dangerRadius: number;
  catchRadius: number;
  /** Seconds a predator rests after a catch. */
  digestTime: number;
}

export const DEFAULT_PREDATOR: PredatorConfig = {
  speed: 11, turnAccel: 12, dangerRadius: 18, catchRadius: 1.4, digestTime: 1.0,
};

export const MAX_PREDATORS = 5;

/** 1-5 rule-based predators chasing the nearest group centre and eating fish on contact. */
export class PredatorSystem {
  readonly pos = new Float32Array(MAX_PREDATORS * 3);
  readonly vel = new Float32Array(MAX_PREDATORS * 3);
  private readonly digest = new Float32Array(MAX_PREDATORS);
  count = 0;
  /** Fish eaten during the last step (reset every step). */
  caughtThisStep = 0;
  totalCaught = 0;

  constructor(readonly cfg: PredatorConfig = DEFAULT_PREDATOR) {}

  setCount(n: number, world: SimConfig, rng: Rng): void {
    const next = Math.max(0, Math.min(MAX_PREDATORS, Math.floor(n)));
    for (let p = this.count; p < next; p++) this.spawn(p, world, rng);
    this.count = next;
  }

  /** Moves predators towards the nearest non-empty group centre. */
  move(centers: Float32Array, groupSizes: Uint32Array, world: SimConfig, dt: number): void {
    for (let p = 0; p < this.count; p++) {
      this.digest[p] = Math.max(0, this.digest[p] - dt);
      const g = nearestGroup(this.pos, p * 3, centers, groupSizes);
      if (g >= 0) this.steer(p, centers[g * 3], centers[g * 3 + 1], centers[g * 3 + 2], dt);
      this.integrate(p, world, dt);
    }
  }

  /** Marks fish inside the catch radius of a hungry predator as eaten (one per predator per step). */
  eat(state: BoidState): void {
    this.caughtThisStep = 0;
    const r2 = this.cfg.catchRadius ** 2;
    for (let p = 0; p < this.count; p++) {
      if (this.digest[p] > 0) continue;
      const victim = closestFish(state, this.pos, p * 3, r2);
      if (victim < 0) continue;
      state.alive[victim] = 0;
      this.digest[p] = this.cfg.digestTime;
      this.caughtThisStep++;
      this.totalCaught++;
    }
  }

  private spawn(p: number, world: SimConfig, rng: Rng): void {
    const [hx, hy, hz] = world.halfExtents;
    this.pos.set([rng.range(-hx, hx) * 0.8, rng.range(-hy, hy) * 0.8, rng.range(-hz, hz) * 0.8], p * 3);
    this.vel.set([0, 0, this.cfg.speed * 0.5], p * 3);
    this.digest[p] = 0;
  }

  private steer(p: number, tx: number, ty: number, tz: number, dt: number): void {
    const o = p * 3;
    const dx = tx - this.pos[o], dy = ty - this.pos[o + 1], dz = tz - this.pos[o + 2];
    const len = length3(dx, dy, dz);
    if (len < EPSILON) return;
    const k = (this.cfg.speed / len);
    const a = this.cfg.turnAccel * dt;
    this.vel[o] += clampAbs(dx * k - this.vel[o], a);
    this.vel[o + 1] += clampAbs(dy * k - this.vel[o + 1], a);
    this.vel[o + 2] += clampAbs(dz * k - this.vel[o + 2], a);
  }

  private integrate(p: number, world: SimConfig, dt: number): void {
    const o = p * 3;
    const slow = this.digest[p] > 0 ? 0.4 : 1;
    for (let a = 0; a < 3; a++) {
      const h = world.halfExtents[a];
      this.pos[o + a] = Math.min(h, Math.max(-h, this.pos[o + a] + this.vel[o + a] * dt * slow));
    }
  }
}

function clampAbs(v: number, max: number): number {
  return Math.min(max, Math.max(-max, v));
}

export function nearestGroup(pos: Float32Array, o: number, centers: Float32Array, sizes: Uint32Array): number {
  let best = -1, bestD = Infinity;
  for (let g = 0; g < sizes.length; g++) {
    if (sizes[g] === 0) continue;
    const d = (centers[g * 3] - pos[o]) ** 2 + (centers[g * 3 + 1] - pos[o + 1]) ** 2 + (centers[g * 3 + 2] - pos[o + 2]) ** 2;
    if (d < bestD) { bestD = d; best = g; }
  }
  return best;
}

function closestFish(state: BoidState, pos: Float32Array, o: number, r2: number): number {
  let best = -1, bestD = r2;
  for (let i = 0; i < state.count; i++) {
    if (!state.alive[i]) continue;
    const q = i * 3;
    const d = (state.pos[q] - pos[o]) ** 2 + (state.pos[q + 1] - pos[o + 1]) ** 2 + (state.pos[q + 2] - pos[o + 2]) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

/** Fish flee from predators inside the danger radius; strength scales with the group strategy. */
export class PredatorAvoidance {
  constructor(private readonly predators: PredatorSystem, private readonly strategies: GroupStrategies) {}

  contribute(i: number, state: BoidState, cfg: SimConfig, out: Float32Array): void {
    const w = cfg.weights.predator * this.strategies.weight(state.group[i], W_PREDATOR);
    if (w <= 0 || this.predators.count === 0) return;
    const o = i * 3;
    const r = this.predators.cfg.dangerRadius;
    for (let p = 0; p < this.predators.count; p++) {
      const q = p * 3;
      const dx = state.pos[o] - this.predators.pos[q];
      const dy = state.pos[o + 1] - this.predators.pos[q + 1];
      const dz = state.pos[o + 2] - this.predators.pos[q + 2];
      const d = length3(dx, dy, dz);
      if (d >= r) continue;
      const k = (w * cfg.maxForce * (1 - d / r)) / Math.max(d, EPSILON);
      out[0] += d < EPSILON ? w * cfg.maxForce : dx * k;
      out[1] += dy * k;
      out[2] += dz * k;
    }
  }
}
