/** Headless comparison of policy modes on the same seeds / scenario (no rendering, no backend). */
import { World, type PolicyMode } from "../src/sim/world";

const SECONDS = Number(process.argv[2] ?? 30);
const SEEDS = (process.argv[3] ?? "1,2,3").split(",").map(Number);
const SCENARIOS = {
  "Predator Attack": { count: 1000, predators: 3, food: 0 },
  "Food Rush": { count: 1000, predators: 0, food: 6 },
  Mixed: { count: 1000, predators: 2, food: 4 },
};

function run(mode: PolicyMode, seed: number, scenario: (typeof SCENARIOS)[keyof typeof SCENARIOS]) {
  const world = World.create({ ...scenario, seed, mode });
  const steps = Math.round(SECONDS * 60);
  for (let s = 0; s < steps; s++) world.step(1 / 60);
  return world.metricsSnapshot();
}

const rows = [];
for (const [name, scenario] of Object.entries(SCENARIOS)) {
  for (const mode of ["boids", "rules"] as PolicyMode[]) {
    const results = SEEDS.map((seed) => run(mode, seed, scenario));
    const mean = (k: "survivalRate" | "foodCollected" | "meanEnergy" | "dispersion") =>
      +(results.reduce((a, r) => a + r[k], 0) / results.length).toFixed(3);
    rows.push({ scenario: name, mode, survival: mean("survivalRate"), food: mean("foodCollected"), energy: mean("meanEnergy"), dispersion: mean("dispersion") });
  }
}
console.table(rows);
