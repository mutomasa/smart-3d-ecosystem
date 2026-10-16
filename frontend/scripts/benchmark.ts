/** Headless Boids benchmark: mean / p95 step time for several population sizes. */
import { BoidsSimulation } from "../src/sim/boids";
import { makeConfig } from "../src/sim/config";

const SIZES = (process.argv[2] ?? "500,1000,3000,5000").split(",").map(Number);
const WARMUP = 120;
const STEPS = 600;

function percentile(sorted: number[], p: number): number {
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}

function measure(count: number) {
  const sim = new BoidsSimulation(makeConfig({ count, seed: 1 }));
  for (let s = 0; s < WARMUP; s++) sim.step();
  const times: number[] = [];
  let maxOcc = 0;
  for (let s = 0; s < STEPS; s++) {
    const t0 = performance.now();
    sim.step();
    times.push(performance.now() - t0);
    maxOcc = Math.max(maxOcc, sim.stats.maxCellOccupancy);
  }
  times.sort((a, b) => a - b);
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  return { count, meanMs: +mean.toFixed(3), p95Ms: +percentile(times, 0.95).toFixed(3), maxCellOccupancy: maxOcc };
}

const results = SIZES.map(measure);
console.table(results);
console.log(JSON.stringify({ node: process.version, results }));
