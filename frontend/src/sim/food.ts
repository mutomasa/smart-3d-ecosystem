import type { SimConfig } from "./config";
import type { Rng } from "./rng";
import type { BoidState } from "./state";
import { W_FOOD, type GroupStrategies } from "./strategy";
import { EPSILON, length3 } from "./vec3";

export interface FoodConfig {
  /** Food units per source when spawned. */
  amount: number;
  eatRadius: number;
  senseRadius: number;
  /** Energy gained per second while eating (also the amount consumed). */
  eatRate: number;
  /** Energy lost per second by every fish. */
  energyDecay: number;
  /** Seconds before an empty source re-appears (autoRespawn only). */
  respawnDelay: number;
  autoRespawn: boolean;
}

export const DEFAULT_FOOD: FoodConfig = {
  amount: 40, eatRadius: 3, senseRadius: 35, eatRate: 0.5, energyDecay: 0.02, respawnDelay: 4, autoRespawn: true,
};

export const MAX_FOOD = 10;

/** 1-10 food patches. Fish inside a patch regain energy and deplete it. */
export class FoodSystem {
  readonly pos = new Float32Array(MAX_FOOD * 3);
  readonly amount = new Float32Array(MAX_FOOD);
  private readonly respawnIn = new Float32Array(MAX_FOOD);
  count = 0;
  totalCollected = 0;

  constructor(readonly cfg: FoodConfig = { ...DEFAULT_FOOD }) {}

  setCount(n: number, world: SimConfig, rng: Rng): void {
    const next = Math.max(0, Math.min(MAX_FOOD, Math.floor(n)));
    for (let f = this.count; f < next; f++) this.spawn(f, world, rng);
    this.count = next;
  }

  /** Manually drops food at a random place (or refills the emptiest slot when full). */
  spawnOne(world: SimConfig, rng: Rng): void {
    if (this.count < MAX_FOOD) return this.setCount(this.count + 1, world, rng);
    this.spawn(this.emptiest(), world, rng);
  }

  isActive(f: number): boolean {
    return f < this.count && this.amount[f] > 0;
  }

  step(state: BoidState, world: SimConfig, rng: Rng, dt: number): void {
    decayEnergy(state, this.cfg.energyDecay * dt);
    for (let f = 0; f < this.count; f++) {
      if (this.amount[f] > 0) this.feed(f, state, dt);
      else this.tickRespawn(f, world, rng, dt);
    }
  }

  private feed(f: number, state: BoidState, dt: number): void {
    const r2 = this.cfg.eatRadius ** 2;
    const o = f * 3;
    for (let i = 0; i < state.count && this.amount[f] > 0; i++) {
      if (!state.alive[i] || state.energy[i] >= 1) continue;
      const q = i * 3;
      const d2 = (state.pos[q] - this.pos[o]) ** 2 + (state.pos[q + 1] - this.pos[o + 1]) ** 2 + (state.pos[q + 2] - this.pos[o + 2]) ** 2;
      if (d2 > r2) continue;
      const bite = Math.min(this.cfg.eatRate * dt, 1 - state.energy[i], this.amount[f]);
      state.energy[i] += bite;
      this.amount[f] -= bite;
      this.totalCollected += bite;
    }
    if (this.amount[f] <= 0) this.respawnIn[f] = this.cfg.respawnDelay;
  }

  private tickRespawn(f: number, world: SimConfig, rng: Rng, dt: number): void {
    if (!this.cfg.autoRespawn) return;
    this.respawnIn[f] -= dt;
    if (this.respawnIn[f] <= 0) this.spawn(f, world, rng);
  }

  private spawn(f: number, world: SimConfig, rng: Rng): void {
    const [hx, hy, hz] = world.halfExtents;
    this.pos.set([rng.range(-hx, hx) * 0.75, rng.range(-hy, hy) * 0.75, rng.range(-hz, hz) * 0.75], f * 3);
    this.amount[f] = this.cfg.amount;
    this.respawnIn[f] = 0;
  }

  private emptiest(): number {
    let best = 0;
    for (let f = 1; f < this.count; f++) if (this.amount[f] < this.amount[best]) best = f;
    return best;
  }
}

function decayEnergy(state: BoidState, amount: number): void {
  for (let i = 0; i < state.count; i++) {
    if (state.alive[i]) state.energy[i] = Math.max(0, state.energy[i] - amount);
  }
}

/** Hungry fish steer towards the nearest active food patch within the sense radius. */
export class FoodAttraction {
  constructor(private readonly food: FoodSystem, private readonly strategies: GroupStrategies) {}

  contribute(i: number, state: BoidState, cfg: SimConfig, out: Float32Array): void {
    const w = cfg.weights.food * this.strategies.weight(state.group[i], W_FOOD);
    if (w <= 0) return;
    const f = this.nearest(i, state);
    if (f < 0) return;
    const o = i * 3, q = f * 3;
    const dx = this.food.pos[q] - state.pos[o];
    const dy = this.food.pos[q + 1] - state.pos[o + 1];
    const dz = this.food.pos[q + 2] - state.pos[o + 2];
    const d = length3(dx, dy, dz);
    if (d < EPSILON) return;
    const hunger = 1 - state.energy[i];
    const k = (w * cfg.maxForce * (0.25 + 0.75 * hunger)) / d;
    out[0] += dx * k;
    out[1] += dy * k;
    out[2] += dz * k;
  }

  private nearest(i: number, state: BoidState): number {
    let best = -1, bestD = this.food.cfg.senseRadius ** 2;
    const o = i * 3;
    for (let f = 0; f < this.food.count; f++) {
      if (!this.food.isActive(f)) continue;
      const q = f * 3;
      const d = (this.food.pos[q] - state.pos[o]) ** 2 + (this.food.pos[q + 1] - state.pos[o + 1]) ** 2 + (this.food.pos[q + 2] - state.pos[o + 2]) ** 2;
      if (d < bestD) { bestD = d; best = f; }
    }
    return best;
  }
}
