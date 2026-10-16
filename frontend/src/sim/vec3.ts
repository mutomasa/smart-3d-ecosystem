/** Allocation-free vector helpers. Vectors are passed as scalars and written into `out[0..2]`. */

export const EPSILON = 1e-6;

export function length3(x: number, y: number, z: number): number {
  return Math.sqrt(x * x + y * y + z * z);
}

export function set3(out: Float32Array, x: number, y: number, z: number): void {
  out[0] = x;
  out[1] = y;
  out[2] = z;
}

export function addScaled3(out: Float32Array, src: Float32Array, scale: number): void {
  out[0] += src[0] * scale;
  out[1] += src[1] * scale;
  out[2] += src[2] * scale;
}

/** Scales `out` down so that its magnitude is at most `max`. Zero vectors are left untouched. */
export function clampMagnitude3(out: Float32Array, max: number): void {
  const len = length3(out[0], out[1], out[2]);
  if (len <= max || len < EPSILON) return;
  const s = max / len;
  out[0] *= s;
  out[1] *= s;
  out[2] *= s;
}

/**
 * Reynolds steering: desired = normalize(d) * maxSpeed, steer = desired - v, clamped to maxForce.
 * A zero direction yields a zero steering vector (never NaN).
 */
export function steerTowards3(
  out: Float32Array,
  dx: number, dy: number, dz: number,
  vx: number, vy: number, vz: number,
  maxSpeed: number,
  maxForce: number,
): void {
  const len = length3(dx, dy, dz);
  if (len < EPSILON) {
    set3(out, 0, 0, 0);
    return;
  }
  const s = maxSpeed / len;
  set3(out, dx * s - vx, dy * s - vy, dz * s - vz);
  clampMagnitude3(out, maxForce);
}
