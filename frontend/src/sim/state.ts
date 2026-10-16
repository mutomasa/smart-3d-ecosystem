import type { SimConfig } from "./config";
import type { Rng } from "./rng";

/**
 * Struct-of-arrays state for all fish. Positions / velocities are double buffered:
 * a step reads `pos`/`vel` and writes `nextPos`/`nextVel`, then `swap()` is called.
 * Slots `[0, count)` are in use; `alive[i] === 0` marks an eaten fish.
 */
export class BoidState {
  capacity = 0;
  count = 0;
  pos = new Float32Array(0);
  vel = new Float32Array(0);
  nextPos = new Float32Array(0);
  nextVel = new Float32Array(0);
  energy = new Float32Array(0);
  alive = new Uint8Array(0);
  group = new Uint16Array(0);

  constructor(capacity: number) {
    this.ensureCapacity(capacity);
  }

  ensureCapacity(capacity: number): void {
    if (capacity <= this.capacity) return;
    this.pos = grow(this.pos, capacity * 3);
    this.vel = grow(this.vel, capacity * 3);
    this.nextPos = grow(this.nextPos, capacity * 3);
    this.nextVel = grow(this.nextVel, capacity * 3);
    this.energy = grow(this.energy, capacity);
    this.alive = grow(this.alive, capacity);
    this.group = grow(this.group, capacity);
    this.capacity = capacity;
  }

  swap(): void {
    [this.pos, this.nextPos] = [this.nextPos, this.pos];
    [this.vel, this.nextVel] = [this.nextVel, this.vel];
  }

  aliveCount(): number {
    let n = 0;
    for (let i = 0; i < this.count; i++) n += this.alive[i];
    return n;
  }
}

type TypedArray = Float32Array | Uint8Array | Uint16Array;

function grow<T extends TypedArray>(src: T, length: number): T {
  const out = new (src.constructor as new (n: number) => T)(length);
  out.set(src);
  return out;
}

/** Spawns fish `[from, to)` around per-group cluster centers. */
export function spawnFish(state: BoidState, from: number, to: number, cfg: SimConfig, rng: Rng): void {
  const centers = groupCenters(cfg, rng);
  for (let i = from; i < to; i++) {
    const g = i % cfg.groupCount;
    spawnOne(state, i, g, centers[g], cfg, rng);
  }
}

function groupCenters(cfg: SimConfig, rng: Rng): number[][] {
  const [hx, hy, hz] = cfg.halfExtents;
  const centers: number[][] = [];
  for (let g = 0; g < cfg.groupCount; g++) {
    centers.push([rng.range(-hx, hx) * 0.6, rng.range(-hy, hy) * 0.6, rng.range(-hz, hz) * 0.6]);
  }
  return centers;
}

function spawnOne(state: BoidState, i: number, g: number, c: number[], cfg: SimConfig, rng: Rng): void {
  const o = i * 3;
  const spread = cfg.perceptionRadius * 1.5;
  for (let a = 0; a < 3; a++) {
    state.pos[o + a] = c[a] + rng.range(-spread, spread);
    state.vel[o + a] = rng.range(-1, 1) * cfg.minSpeed;
  }
  state.vel[o] += cfg.minSpeed; // avoid zero initial velocity
  state.energy[i] = 1;
  state.alive[i] = 1;
  state.group[i] = g;
}
