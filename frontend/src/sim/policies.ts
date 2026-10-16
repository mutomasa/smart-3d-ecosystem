import { OBS_DIM } from "./groups";
import type { GroupStrategies } from "./strategy";

export type PolicyMode = "boids" | "rules" | "ml";

/** Writes a strategy for every group from the observation matrix (groupCount x OBS_DIM). */
export interface StrategyPolicy {
  readonly mode: PolicyMode;
  decide(observations: Float32Array, sizes: Uint32Array, strategies: GroupStrategies): void;
}

/** Plain flocking: no predator avoidance, no food seeking, neutral cohesion. */
export class BoidsOnlyPolicy implements StrategyPolicy {
  readonly mode = "boids" as const;

  decide(_obs: Float32Array, _sizes: Uint32Array, strategies: GroupStrategies): void {
    for (let g = 0; g < strategies.groupCount; g++) strategies.setTarget(g, 0, 0, 0, 0, 0, 0.5);
  }
}

/** Thresholds of the handcrafted heuristic. Mirrored in Python (`backend/policy.py`). */
export const RULES = {
  fleeDistance: 25,
  hungryEnergy: 0.7,
  flee: [1.0, 0.1, 0.8],
  forage: [0.6, 1.0, 0.4],
  cruise: [0.6, 0.3, 0.5],
} as const;

/**
 * Handcrafted heuristic: flee when a predator is close, forage when hungry and food exists,
 * otherwise cruise. Inputs are raw (world-unit) features in schema order.
 */
export function rulesAction(f: ArrayLike<number>, at = 0): [number, number, number, number, number, number] {
  const predatorPresent = f[at + 18] > 0.5, foodPresent = f[at + 19] > 0.5;
  if (predatorPresent && f[at + 9] < RULES.fleeDistance) {
    return [-f[at + 6], -f[at + 7], -f[at + 8], ...RULES.flee];
  }
  if (foodPresent && f[at + 14] < RULES.hungryEnergy) {
    return [f[at + 10], f[at + 11], f[at + 12], ...RULES.forage];
  }
  return [0, 0, 0, ...RULES.cruise];
}

export class RulesPolicy implements StrategyPolicy {
  readonly mode = "rules" as const;

  decide(obs: Float32Array, sizes: Uint32Array, strategies: GroupStrategies): void {
    for (let g = 0; g < strategies.groupCount; g++) {
      if (sizes[g] === 0) continue;
      const [dx, dy, dz, wp, wf, wc] = rulesAction(obs, g * OBS_DIM);
      strategies.setTarget(g, dx, dy, dz, wp, wf, wc);
    }
  }
}
