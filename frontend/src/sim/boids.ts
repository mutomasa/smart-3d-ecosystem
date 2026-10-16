import type { SimConfig } from "./config";
import { integrate } from "./motion";
import { Rng } from "./rng";
import { SpatialGrid } from "./spatialGrid";
import { BoidState, spawnFish } from "./state";
import {
  NeighborSums,
  accumulateNeighbors,
  alignmentForce,
  boundaryForce,
  cohesionForce,
  separationForce,
} from "./steering";
import { addScaled3, clampMagnitude3 } from "./vec3";

/**
 * Extension point for extra steering terms (predators, food, ML group direction ...).
 * Implementations add their (already weighted) contribution for agent `i` into `out`.
 */
export interface SteeringContributor {
  prepare?(state: BoidState, cfg: SimConfig): void;
  contribute(i: number, state: BoidState, cfg: SimConfig, out: Float32Array): void;
}

export interface StepStats {
  neighborChecks: number;
  collisions: number;
  maxCellOccupancy: number;
}

/** Deterministic Boids core: fixed dt, seeded spawn, double-buffered batch update. */
export class BoidsSimulation {
  readonly state: BoidState;
  readonly config: SimConfig;
  readonly stats: StepStats = { neighborChecks: 0, collisions: 0, maxCellOccupancy: 0 };
  tick = 0;
  readonly grid: SpatialGrid;
  /** Optional per-group multiplier of the cohesion weight (set by the strategy layer). */
  cohesionScale: ((group: number) => number) | null = null;
  private rng: Rng;
  private contributors: SteeringContributor[] = [];
  private readonly ranges = new Int32Array(54);
  private readonly sums = new NeighborSums();
  private readonly acc = new Float32Array(3);
  private readonly force = new Float32Array(3);

  constructor(config: SimConfig) {
    this.config = config;
    this.rng = new Rng(config.seed);
    this.state = new BoidState(config.count);
    this.grid = new SpatialGrid(config.halfExtents, config.perceptionRadius);
    this.state.count = config.count;
    spawnFish(this.state, 0, config.count, config, this.rng);
  }

  addContributor(c: SteeringContributor): void {
    this.contributors.push(c);
  }

  /** Changes the number of slots. New fish are spawned from the seeded RNG; removed slots are dropped. */
  setCount(count: number): void {
    const n = Math.max(0, Math.floor(count));
    const prev = this.state.count;
    this.state.ensureCapacity(n);
    this.state.count = n;
    if (n > prev) spawnFish(this.state, prev, n, this.config, this.rng);
    this.config.count = n;
  }

  step(): void {
    const { state, config } = this;
    this.grid.rebuild(state.pos, state.alive, state.count);
    for (const c of this.contributors) c.prepare?.(state, config);
    this.resetStats();
    for (let i = 0; i < state.count; i++) {
      if (state.alive[i]) this.updateAgent(i);
      else this.copyDead(i);
    }
    state.swap();
    this.tick++;
  }

  private resetStats(): void {
    this.stats.neighborChecks = 0;
    this.stats.collisions = 0;
    this.stats.maxCellOccupancy = this.grid.maxOccupancy;
  }

  private updateAgent(i: number): void {
    const { state, config } = this;
    const o = i * 3;
    const r = this.grid.queryRanges(state.pos[o], state.pos[o + 1], state.pos[o + 2], this.ranges);
    accumulateNeighbors(i, this.grid.sortedIndices, this.ranges, r, state.pos, state.vel, config, this.sums);
    this.stats.neighborChecks += this.sums.candidates;
    this.stats.collisions += this.sums.collisions;
    this.computeAcceleration(i);
    integrate(i * 3, this.acc, state.pos, state.vel, state.nextPos, state.nextVel, config);
  }

  private computeAcceleration(i: number): void {
    const { state, config, acc, force, sums } = this;
    const w = config.weights;
    const o = i * 3;
    const px = state.pos[o], py = state.pos[o + 1], pz = state.pos[o + 2];
    const vx = state.vel[o], vy = state.vel[o + 1], vz = state.vel[o + 2];
    acc.fill(0);
    separationForce(force, sums, vx, vy, vz, config);
    addScaled3(acc, force, w.separation);
    alignmentForce(force, sums, vx, vy, vz, config);
    addScaled3(acc, force, w.alignment);
    cohesionForce(force, sums, px, py, pz, vx, vy, vz, config);
    addScaled3(acc, force, w.cohesion * (this.cohesionScale ? this.cohesionScale(state.group[i]) : 1));
    boundaryForce(force, px, py, pz, vx, vy, vz, config);
    addScaled3(acc, force, w.boundary);
    for (const c of this.contributors) c.contribute(i, state, config, acc);
    clampMagnitude3(acc, config.maxForce);
  }

  private copyDead(i: number): void {
    const { state } = this;
    const o = i * 3;
    for (let a = 0; a < 3; a++) {
      state.nextPos[o + a] = state.pos[o + a];
      state.nextVel[o + a] = 0;
    }
  }
}
