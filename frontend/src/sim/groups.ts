import schema from "../../../shared/schemas/policy.json";
import type { FoodSystem } from "./food";
import type { PredatorSystem } from "./predator";
import type { BoidState } from "./state";
import { length3 } from "./vec3";

export const FEATURE_NAMES: readonly string[] = schema.observation.features;
export const OBS_DIM = FEATURE_NAMES.length;
const MAX_DISTANCE = schema.normalization.max_distance;
const NEARBY_GROUP_RADIUS = 20;

/**
 * Per-group summaries computed from the fish state in O(N). They feed the predators, the metrics
 * and the policy observations (raw world units; normalization happens in Python).
 */
export class GroupSummaries {
  readonly center: Float32Array;
  readonly velocity: Float32Array;
  readonly size: Uint32Array;
  readonly energy: Float32Array;
  readonly dispersion: Float32Array;

  constructor(readonly groupCount: number) {
    this.center = new Float32Array(groupCount * 3);
    this.velocity = new Float32Array(groupCount * 3);
    this.size = new Uint32Array(groupCount);
    this.energy = new Float32Array(groupCount);
    this.dispersion = new Float32Array(groupCount);
  }

  compute(state: BoidState): void {
    this.accumulate(state);
    this.average();
    this.computeDispersion(state);
  }

  /** Mean dispersion over non-empty groups, weighted by group size. */
  meanDispersion(): number {
    let sum = 0, n = 0;
    for (let g = 0; g < this.groupCount; g++) {
      sum += this.dispersion[g] * this.size[g];
      n += this.size[g];
    }
    return n === 0 ? 0 : sum / n;
  }

  density(g: number): number {
    const r = this.dispersion[g] + 1;
    return Math.min(1, this.size[g] / ((4 / 3) * Math.PI * r * r * r));
  }

  nearbyGroups(g: number): number {
    let n = 0;
    for (let h = 0; h < this.groupCount; h++) {
      if (h === g || this.size[h] === 0) continue;
      if (distance(this.center, g * 3, this.center, h * 3) < NEARBY_GROUP_RADIUS) n++;
    }
    return this.groupCount > 1 ? n / (this.groupCount - 1) : 0;
  }

  /** Writes the observation vector of group `g` (schema order, raw units) into `out`. */
  features(g: number, predators: PredatorSystem, food: FoodSystem, out: Float32Array): void {
    const o = g * 3;
    out.set(this.center.subarray(o, o + 3), 0);
    out.set(this.velocity.subarray(o, o + 3), 3);
    const pred = nearestPoint(this.center, o, predators.pos, predators.count, () => true);
    writeRelative(out, 6, this.center, o, predators.pos, pred);
    const fd = nearestPoint(this.center, o, food.pos, food.count, (f) => food.isActive(f));
    writeRelative(out, 10, this.center, o, food.pos, fd);
    out[14] = this.energy[g];
    out[15] = this.density(g);
    out[16] = this.dispersion[g];
    out[17] = this.nearbyGroups(g);
    out[18] = pred >= 0 ? 1 : 0;
    out[19] = fd >= 0 ? 1 : 0;
  }

  private accumulate(state: BoidState): void {
    this.center.fill(0); this.velocity.fill(0); this.size.fill(0); this.energy.fill(0);
    for (let i = 0; i < state.count; i++) {
      if (!state.alive[i]) continue;
      const g = state.group[i], o = i * 3, q = g * 3;
      for (let a = 0; a < 3; a++) {
        this.center[q + a] += state.pos[o + a];
        this.velocity[q + a] += state.vel[o + a];
      }
      this.energy[g] += state.energy[i];
      this.size[g]++;
    }
  }

  private average(): void {
    for (let g = 0; g < this.groupCount; g++) {
      const n = this.size[g];
      if (n === 0) continue;
      for (let a = 0; a < 3; a++) {
        this.center[g * 3 + a] /= n;
        this.velocity[g * 3 + a] /= n;
      }
      this.energy[g] /= n;
    }
  }

  private computeDispersion(state: BoidState): void {
    this.dispersion.fill(0);
    for (let i = 0; i < state.count; i++) {
      if (!state.alive[i]) continue;
      const g = state.group[i];
      this.dispersion[g] += distance(state.pos, i * 3, this.center, g * 3);
    }
    for (let g = 0; g < this.groupCount; g++) if (this.size[g] > 0) this.dispersion[g] /= this.size[g];
  }
}

function distance(a: Float32Array, i: number, b: Float32Array, j: number): number {
  return length3(a[i] - b[j], a[i + 1] - b[j + 1], a[i + 2] - b[j + 2]);
}

function nearestPoint(
  from: Float32Array, o: number, points: Float32Array, count: number, active: (k: number) => boolean,
): number {
  let best = -1, bestD = Infinity;
  for (let k = 0; k < count; k++) {
    if (!active(k)) continue;
    const d = distance(points, k * 3, from, o);
    if (d < bestD) { bestD = d; best = k; }
  }
  return best;
}

/** Relative vector + distance, or the absent sentinel (zero vector, max distance). */
function writeRelative(out: Float32Array, at: number, from: Float32Array, o: number, points: Float32Array, k: number): void {
  if (k < 0) {
    out.set([0, 0, 0, MAX_DISTANCE], at);
    return;
  }
  const dx = points[k * 3] - from[o], dy = points[k * 3 + 1] - from[o + 1], dz = points[k * 3 + 2] - from[o + 2];
  out.set([dx, dy, dz, Math.min(MAX_DISTANCE, length3(dx, dy, dz))], at);
}
