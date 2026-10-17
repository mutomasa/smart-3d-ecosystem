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

const MODES = new Set(["boids", "rules", "ml"]);

/**
 * Initial scenario from the URL, e.g. `?preset=Food%20Rush&mode=ml&seed=3&count=1500&predators=2&food=4`.
 * Starting directly in the requested scenario means no telemetry is ever sent for a default
 * scene that is replaced a second later (#22). Invalid values fall back to the preset's own.
 */
export function presetFromUrl(search: string, fallback: Preset = PRESETS["Calm School"]): Preset {
  const q = new URLSearchParams(search);
  const base = PRESETS[q.get("preset") ?? ""] ?? fallback;
  const num = (key: string, min: number, max: number, def: number) => {
    const v = Number(q.get(key));
    return q.has(key) && Number.isInteger(v) && v >= min && v <= max ? v : def;
  };
  const mode = q.get("mode");
  return {
    ...base,
    mode: mode && MODES.has(mode) ? (mode as Preset["mode"]) : base.mode,
    seed: num("seed", 0, 99_999, base.seed),
    count: num("count", 0, 5000, base.count),
    predators: num("predators", 0, 5, base.predators),
    food: num("food", 0, 10, base.food),
  };
}
