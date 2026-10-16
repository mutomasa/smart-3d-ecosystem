import { BoidsSimulation } from "./boids";
import { makeConfig, type BoidsWeights, type SimConfig } from "./config";
import { packFish } from "./frame";

export type PolicyMode = "boids" | "rules" | "ml";

export interface WorldSettings {
  count: number;
  seed: number;
  weights: Partial<BoidsWeights>;
}

/** Facade over the simulation subsystems; owned by the Web Worker (or by tests / benchmarks). */
export class World {
  sim: BoidsSimulation;
  mode: PolicyMode = "boids";

  constructor(config: SimConfig) {
    this.sim = new BoidsSimulation(config);
  }

  static create(settings: Partial<WorldSettings> = {}): World {
    return new World(makeConfig({ count: settings.count, seed: settings.seed, weights: settings.weights as BoidsWeights }));
  }

  get config(): SimConfig {
    return this.sim.config;
  }

  get tick(): number {
    return this.sim.tick;
  }

  step(dt: number): void {
    this.sim.config.dt = dt;
    this.sim.step();
  }

  setWeights(weights: Partial<BoidsWeights>): void {
    Object.assign(this.sim.config.weights, weights);
  }

  setCount(count: number): void {
    this.sim.setCount(count);
  }

  packFish(out: Float32Array): void {
    packFish(this.sim.state, out);
  }

  aliveCount(): number {
    return this.sim.state.aliveCount();
  }
}
