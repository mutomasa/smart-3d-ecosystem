import type { FoodSystem } from "./food";
import type { GroupSummaries } from "./groups";
import type { PredatorSystem } from "./predator";
import type { BoidState } from "./state";

/** Evaluation metrics of section 9 that can be computed inside the simulation. */
export interface EcosystemMetrics {
  survivalRate: number;
  alive: number;
  initial: number;
  foodCollected: number;
  meanEnergy: number;
  dispersion: number;
  collisions: number;
  predation: number;
}

export class MetricsTracker {
  private initial: number;
  private collisionsTotal = 0;

  constructor(initial: number) {
    this.initial = Math.max(1, initial);
  }

  /** Population changes from the UI redefine the baseline (otherwise survival would jump). */
  rebase(population: number): void {
    this.initial = Math.max(1, population);
  }

  addCollisions(n: number): void {
    this.collisionsTotal += n;
  }

  snapshot(state: BoidState, groups: GroupSummaries, food: FoodSystem, predators: PredatorSystem): EcosystemMetrics {
    const alive = state.aliveCount();
    return {
      survivalRate: alive / this.initial,
      alive,
      initial: this.initial,
      foodCollected: food.totalCollected,
      meanEnergy: meanEnergy(state),
      dispersion: groups.meanDispersion(),
      collisions: this.collisionsTotal,
      predation: predators.totalCaught,
    };
  }
}

function meanEnergy(state: BoidState): number {
  let sum = 0, n = 0;
  for (let i = 0; i < state.count; i++) {
    if (!state.alive[i]) continue;
    sum += state.energy[i];
    n++;
  }
  return n === 0 ? 0 : sum / n;
}
