import type { BoidState } from "./state";
import { EPSILON, length3 } from "./vec3";

/** Polarization order parameter in [0, 1]: |mean of unit velocities|. 1 = everybody aligned. */
export function polarization(state: BoidState): number {
  let sx = 0, sy = 0, sz = 0, n = 0;
  for (let i = 0; i < state.count; i++) {
    if (!state.alive[i]) continue;
    const o = i * 3;
    const len = length3(state.vel[o], state.vel[o + 1], state.vel[o + 2]);
    if (len < EPSILON) continue;
    sx += state.vel[o] / len;
    sy += state.vel[o + 1] / len;
    sz += state.vel[o + 2] / len;
    n++;
  }
  return n === 0 ? 0 : length3(sx, sy, sz) / n;
}

/** True when every position and velocity component is a finite number. */
export function allFinite(state: BoidState): boolean {
  for (let k = 0; k < state.count * 3; k++) {
    if (!Number.isFinite(state.pos[k]) || !Number.isFinite(state.vel[k])) return false;
  }
  return true;
}

/**
 * Mean cosine between each agent's velocity and the mean velocity of its neighbours within
 * `radius`. ~1 means locally aligned schools. Brute force O(N^2): use for tests / offline analysis.
 */
export function localAlignment(state: BoidState, radius: number): number {
  let total = 0, n = 0;
  for (let i = 0; i < state.count; i++) {
    if (!state.alive[i]) continue;
    const c = neighborCosine(state, i, radius * radius);
    if (c === null) continue;
    total += c;
    n++;
  }
  return n === 0 ? 0 : total / n;
}

function neighborCosine(state: BoidState, i: number, r2: number): number | null {
  const { pos, vel } = state;
  const o = i * 3;
  let ax = 0, ay = 0, az = 0, m = 0;
  for (let j = 0; j < state.count; j++) {
    if (j === i || !state.alive[j]) continue;
    const q = j * 3;
    const dx = pos[q] - pos[o], dy = pos[q + 1] - pos[o + 1], dz = pos[q + 2] - pos[o + 2];
    if (dx * dx + dy * dy + dz * dz >= r2) continue;
    ax += vel[q];
    ay += vel[q + 1];
    az += vel[q + 2];
    m++;
  }
  const la = length3(ax, ay, az);
  const lv = length3(vel[o], vel[o + 1], vel[o + 2]);
  if (m === 0 || la < EPSILON || lv < EPSILON) return null;
  return (ax * vel[o] + ay * vel[o + 1] + az * vel[o + 2]) / (la * lv);
}
