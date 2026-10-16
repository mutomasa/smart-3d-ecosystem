import { BoidsSimulation } from "./boids";
import { makeConfig, type BoidsWeights, type SimConfig } from "./config";
import { FoodAttraction, FoodSystem } from "./food";
import { packFish } from "./frame";
import { GroupSummaries, OBS_DIM } from "./groups";
import { MetricsTracker, type EcosystemMetrics } from "./metrics";
import { ExternalMlPolicy } from "./mlPolicy";
import { BoidsOnlyPolicy, RulesPolicy, type PolicyMode, type StrategyPolicy } from "./policies";
import { PredatorAvoidance, PredatorSystem } from "./predator";
import { Rng } from "./rng";
import { GroupStrategies, StrategySteering } from "./strategy";

export type { PolicyMode } from "./policies";

export interface WorldSettings {
  count: number;
  seed: number;
  weights: Partial<BoidsWeights>;
  /** false => keep dt = 1/60 for the whole run (reproducible comparisons). */
  adaptiveDt: boolean;
  predators: number;
  food: number;
  mode: PolicyMode;
}

/** Policy decisions per second (design: 5-10 Hz). */
export const POLICY_HZ = 10;

/** Facade over the simulation subsystems; owned by the Web Worker (or by tests / benchmarks). */
export class World {
  readonly sim: BoidsSimulation;
  readonly predators = new PredatorSystem();
  readonly food = new FoodSystem();
  readonly groups: GroupSummaries;
  readonly strategies: GroupStrategies;
  readonly observations: Float32Array;
  private readonly metrics: MetricsTracker;
  // Independent RNG streams so that e.g. fish count does not shift food placement.
  private readonly predatorRng: Rng;
  private readonly foodRng: Rng;
  private policy: StrategyPolicy = new BoidsOnlyPolicy();
  private sinceDecision = Infinity;
  /** Simulated seconds since reset. */
  simTime = 0;
  /** Incremented on every policy decision (the worker uses it to emit ML observations). */
  decisions = 0;

  constructor(config: SimConfig, settings: Partial<WorldSettings> = {}) {
    this.sim = new BoidsSimulation(config);
    this.groups = new GroupSummaries(config.groupCount);
    this.strategies = new GroupStrategies(config.groupCount);
    this.observations = new Float32Array(config.groupCount * OBS_DIM);
    this.metrics = new MetricsTracker(config.count);
    this.predatorRng = new Rng(config.seed ^ 0x9e3779b9);
    this.foodRng = new Rng(config.seed ^ 0x85ebca6b);
    this.wireSteering();
    this.predators.setCount(settings.predators ?? 0, config, this.predatorRng);
    this.food.setCount(settings.food ?? 0, config, this.foodRng);
    this.setMode(settings.mode ?? "boids");
  }

  static create(settings: Partial<WorldSettings> = {}): World {
    const config = makeConfig({ count: settings.count, seed: settings.seed, weights: settings.weights as BoidsWeights });
    return new World(config, settings);
  }

  get config(): SimConfig {
    return this.sim.config;
  }

  get tick(): number {
    return this.sim.tick;
  }

  get mode(): PolicyMode {
    return this.policy.mode;
  }

  /** In "ml" mode strategies come from the backend (rules fallback when it is silent). */
  get mlPolicy(): ExternalMlPolicy | null {
    return this.policy instanceof ExternalMlPolicy ? this.policy : null;
  }

  setMode(mode: PolicyMode): void {
    if (mode === "ml") return this.setPolicy(new ExternalMlPolicy(() => this.simTime));
    this.setPolicy(mode === "boids" ? new BoidsOnlyPolicy() : new RulesPolicy());
  }

  setPolicy(policy: StrategyPolicy): void {
    this.policy = policy;
    this.sinceDecision = Infinity;
    this.decide();
    this.strategies.snap();
  }

  step(dt: number): void {
    this.config.dt = dt;
    this.simTime += dt;
    this.groups.compute(this.sim.state);
    this.sinceDecision += dt;
    if (this.sinceDecision >= 1 / POLICY_HZ - 1e-9) this.decide();
    this.strategies.smooth(dt);
    this.predators.move(this.groups.center, this.groups.size, this.config, dt);
    this.sim.step();
    this.predators.eat(this.sim.state);
    this.food.step(this.sim.state, this.config, this.foodRng, dt);
    this.metrics.addCollisions(this.sim.stats.collisions);
  }

  setWeights(weights: Partial<BoidsWeights>): void {
    Object.assign(this.config.weights, weights);
  }

  setCount(count: number): void {
    this.sim.setCount(count);
    this.metrics.rebase(this.sim.state.aliveCount());
  }

  setPredators(n: number): void {
    this.predators.setCount(n, this.config, this.predatorRng);
  }

  setFood(n: number): void {
    this.food.setCount(n, this.config, this.foodRng);
  }

  spawnFood(): void {
    this.food.spawnOne(this.config, this.foodRng);
  }

  packFish(out: Float32Array): void {
    packFish(this.sim.state, out);
  }

  aliveCount(): number {
    return this.sim.state.aliveCount();
  }

  metricsSnapshot(): EcosystemMetrics {
    return this.metrics.snapshot(this.sim.state, this.groups, this.food, this.predators);
  }

  /** Rebuilds the observation matrix (raw units, schema order) from the latest group summaries. */
  buildObservations(): Float32Array {
    const row = new Float32Array(OBS_DIM);
    for (let g = 0; g < this.groups.groupCount; g++) {
      this.groups.features(g, this.predators, this.food, row);
      this.observations.set(row, g * OBS_DIM);
    }
    return this.observations;
  }

  private decide(): void {
    this.sinceDecision = 0;
    this.decisions++;
    this.groups.compute(this.sim.state);
    this.policy.decide(this.buildObservations(), this.groups.size, this.strategies);
  }

  private wireSteering(): void {
    const steering = new StrategySteering(this.strategies);
    this.sim.addContributor(steering);
    this.sim.addContributor(new PredatorAvoidance(this.predators, this.strategies));
    this.sim.addContributor(new FoodAttraction(this.food, this.strategies));
    this.sim.cohesionScale = (g) => steering.cohesionScale(g);
  }
}
