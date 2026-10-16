import type { BoidsWeights } from "./sim/config";
import type { PolicyMode } from "./sim/world";

export interface Preset {
  name: string;
  count: number;
  seed: number;
  mode: PolicyMode;
  predators: number;
  food: number;
  weights: Partial<BoidsWeights>;
}

export const PRESETS: Record<string, Preset> = {
  "Calm School": { name: "Calm School", count: 1500, seed: 42, mode: "boids", predators: 0, food: 0, weights: {} },
  "Predator Attack": { name: "Predator Attack", count: 2000, seed: 7, mode: "rules", predators: 3, food: 0, weights: { predator: 3.5 } },
  "Food Rush": { name: "Food Rush", count: 2000, seed: 11, mode: "rules", predators: 0, food: 6, weights: { food: 1.5 } },
  "ML vs Rules": { name: "ML vs Rules", count: 2000, seed: 2024, mode: "ml", predators: 2, food: 4, weights: {} },
};

export const PRESET_NAMES = Object.keys(PRESETS);
