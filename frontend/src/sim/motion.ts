import type { SimConfig } from "./config";
import { EPSILON, length3 } from "./vec3";

/**
 * Integrates one agent: v' = clamp(v + a dt), with a turn-rate limit and min/max speed,
 * then p' = p + v' dt. Reads from `pos`/`vel`, writes to `outPos`/`outVel` at offset `o`.
 */
export function integrate(
  o: number,
  acc: Float32Array,
  pos: Float32Array,
  vel: Float32Array,
  outPos: Float32Array,
  outVel: Float32Array,
  cfg: SimConfig,
): void {
  const vx = vel[o], vy = vel[o + 1], vz = vel[o + 2];
  let nx = vx + acc[0] * cfg.dt;
  let ny = vy + acc[1] * cfg.dt;
  let nz = vz + acc[2] * cfg.dt;
  const speed = clampSpeed(length3(nx, ny, nz), cfg);
  const dir = limitTurn(vx, vy, vz, nx, ny, nz, cfg.maxTurnRate * cfg.dt);
  nx = dir[0] * speed;
  ny = dir[1] * speed;
  nz = dir[2] * speed;
  outVel[o] = nx;
  outVel[o + 1] = ny;
  outVel[o + 2] = nz;
  outPos[o] = clampAxis(pos[o] + nx * cfg.dt, cfg.halfExtents[0]);
  outPos[o + 1] = clampAxis(pos[o + 1] + ny * cfg.dt, cfg.halfExtents[1]);
  outPos[o + 2] = clampAxis(pos[o + 2] + nz * cfg.dt, cfg.halfExtents[2]);
}

function clampSpeed(speed: number, cfg: SimConfig): number {
  return Math.min(cfg.maxSpeed, Math.max(cfg.minSpeed, speed));
}

/** Hard safety wall slightly outside the soft boundary. */
function clampAxis(v: number, half: number): number {
  const limit = half * 1.1;
  return Math.min(limit, Math.max(-limit, v));
}

const turnScratch = new Float32Array(3);

/**
 * Returns the unit direction obtained by rotating from `v` towards `n` by at most `maxAngle`.
 * Degenerate inputs (zero vectors, antiparallel) fall back to a safe direction.
 */
export function limitTurn(
  vx: number, vy: number, vz: number,
  nx: number, ny: number, nz: number,
  maxAngle: number,
): Float32Array {
  const lv = length3(vx, vy, vz);
  const ln = length3(nx, ny, nz);
  if (ln < EPSILON) return unitOr(turnScratch, vx, vy, vz, lv);
  if (lv < EPSILON) return unitOr(turnScratch, nx, ny, nz, ln);
  const ux = vx / lv, uy = vy / lv, uz = vz / lv;
  const wx = nx / ln, wy = ny / ln, wz = nz / ln;
  const cos = Math.min(1, Math.max(-1, ux * wx + uy * wy + uz * wz));
  const angle = Math.acos(cos);
  const sin = Math.sin(angle);
  if (angle <= maxAngle) return unitOr(turnScratch, wx, wy, wz, 1);
  if (sin < 1e-4) return unitOr(turnScratch, ux, uy, uz, 1); // antiparallel: keep heading
  const a = Math.sin(angle - maxAngle) / sin;
  const b = Math.sin(maxAngle) / sin;
  const rx = ux * a + wx * b, ry = uy * a + wy * b, rz = uz * a + wz * b;
  return unitOr(turnScratch, rx, ry, rz, length3(rx, ry, rz));
}

function unitOr(out: Float32Array, x: number, y: number, z: number, len: number): Float32Array {
  if (len < EPSILON) {
    out[0] = 1;
    out[1] = 0;
    out[2] = 0;
    return out;
  }
  out[0] = x / len;
  out[1] = y / len;
  out[2] = z / len;
  return out;
}
