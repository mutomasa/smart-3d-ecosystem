import type { SimConfig } from "./config";
import { EPSILON, set3, steerTowards3 } from "./vec3";

/** Accumulated neighbourhood sums for one agent. */
export class NeighborSums {
  readonly sep = new Float32Array(3);
  readonly align = new Float32Array(3);
  readonly coh = new Float32Array(3);
  neighbors = 0;
  /** Agents found inside the perception radius (before the topological limit). */
  candidates = 0;
  separated = 0;
  collisions = 0;

  reset(): void {
    this.candidates = 0;
    this.sep.fill(0);
    this.align.fill(0);
    this.coh.fill(0);
    this.neighbors = 0;
    this.separated = 0;
    this.collisions = 0;
  }
}

/**
 * Scans candidate neighbours of agent `i` and accumulates separation / alignment / cohesion sums.
 * Separation always uses every agent inside the separation radius; alignment and cohesion use at
 * most `cfg.maxNeighbors` agents (topological limit). Coincident agents (distance 0) get a
 * deterministic index-based push instead of a division by zero.
 */
export function accumulateNeighbors(
  i: number,
  indices: Int32Array,
  ranges: Int32Array,
  rangeCount: number,
  pos: Float32Array,
  vel: Float32Array,
  cfg: SimConfig,
  sums: NeighborSums,
): void {
  const o = i * 3;
  const px = pos[o], py = pos[o + 1], pz = pos[o + 2];
  const r2 = cfg.perceptionRadius * cfg.perceptionRadius;
  const s2 = cfg.separationRadius * cfg.separationRadius;
  const c2 = cfg.collisionRadius * cfg.collisionRadius;
  const limit = cfg.maxNeighbors;
  // Hot loop: accumulate into locals and write `sums` once at the end.
  let ax = 0, ay = 0, az = 0, cx = 0, cy = 0, cz = 0, sx = 0, sy = 0, sz = 0;
  let neighbors = 0, candidates = 0, separated = 0, collisions = 0;
  for (let r = 0; r < rangeCount; r++) {
    const end = ranges[r * 2 + 1];
    for (let k = ranges[r * 2]; k < end; k++) {
      const j = indices[k];
      const q = j * 3;
      const dx = pos[q] - px, dy = pos[q + 1] - py, dz = pos[q + 2] - pz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > r2 || j === i) continue;
      candidates++;
      if (neighbors < limit) {
        ax += vel[q]; ay += vel[q + 1]; az += vel[q + 2];
        cx += pos[q]; cy += pos[q + 1]; cz += pos[q + 2];
        neighbors++;
      }
      if (d2 >= s2) continue;
      separated++;
      if (d2 < c2) collisions++;
      if (d2 < EPSILON) {
        sx += i < j ? -1 : 1; // deterministic push apart when exactly overlapping
      } else {
        // Push away with weight 1/d (vector / d^2).
        sx -= dx / d2; sy -= dy / d2; sz -= dz / d2;
      }
    }
  }
  set3(sums.align, ax, ay, az);
  set3(sums.coh, cx, cy, cz);
  set3(sums.sep, sx, sy, sz);
  sums.neighbors = neighbors;
  sums.candidates = candidates;
  sums.separated = separated;
  sums.collisions = collisions;
}

export function separationForce(
  out: Float32Array, sums: NeighborSums, vx: number, vy: number, vz: number, cfg: SimConfig,
): void {
  if (sums.separated === 0) return set3(out, 0, 0, 0);
  steerTowards3(out, sums.sep[0], sums.sep[1], sums.sep[2], vx, vy, vz, cfg.maxSpeed, cfg.maxForce);
}

export function alignmentForce(
  out: Float32Array, sums: NeighborSums, vx: number, vy: number, vz: number, cfg: SimConfig,
): void {
  if (sums.neighbors === 0) return set3(out, 0, 0, 0);
  steerTowards3(out, sums.align[0], sums.align[1], sums.align[2], vx, vy, vz, cfg.maxSpeed, cfg.maxForce);
}

export function cohesionForce(
  out: Float32Array, sums: NeighborSums,
  px: number, py: number, pz: number,
  vx: number, vy: number, vz: number,
  cfg: SimConfig,
): void {
  if (sums.neighbors === 0) return set3(out, 0, 0, 0);
  const inv = 1 / sums.neighbors;
  const dx = sums.coh[0] * inv - px, dy = sums.coh[1] * inv - py, dz = sums.coh[2] * inv - pz;
  steerTowards3(out, dx, dy, dz, vx, vy, vz, cfg.maxSpeed, cfg.maxForce);
}

/**
 * Smooth boundary steering: inside the margin the agent is pushed back with a strength that
 * grows linearly with penetration depth (no hard reflection).
 */
export function boundaryForce(
  out: Float32Array,
  px: number, py: number, pz: number,
  vx: number, vy: number, vz: number,
  cfg: SimConfig,
): void {
  const dx = axisPush(px, cfg.halfExtents[0], cfg.boundaryMargin);
  const dy = axisPush(py, cfg.halfExtents[1], cfg.boundaryMargin);
  const dz = axisPush(pz, cfg.halfExtents[2], cfg.boundaryMargin);
  if (dx === 0 && dy === 0 && dz === 0) return set3(out, 0, 0, 0);
  steerTowards3(out, dx, dy, dz, vx, vy, vz, cfg.maxSpeed, cfg.maxForce);
  const depth = Math.min(1, Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)));
  out[0] *= depth;
  out[1] *= depth;
  out[2] *= depth;
}

/** Returns a push in [-1, 1]; 0 when outside the margin band. */
export function axisPush(p: number, half: number, margin: number): number {
  const inner = half - margin;
  if (p > inner) return -Math.min(1, (p - inner) / margin);
  if (p < -inner) return Math.min(1, (-inner - p) / margin);
  return 0;
}
