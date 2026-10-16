import type { BoidState } from "./state";

/** Floats per fish in a frame buffer: px, py, pz, vx, vy, vz, alive. */
export const FISH_STRIDE = 7;

export function frameFloats(count: number): number {
  return count * FISH_STRIDE;
}

/** Packs the current fish state into `out` (must hold `frameFloats(state.count)` floats). */
export function packFish(state: BoidState, out: Float32Array): void {
  for (let i = 0; i < state.count; i++) {
    const o = i * 3;
    const f = i * FISH_STRIDE;
    out[f] = state.pos[o];
    out[f + 1] = state.pos[o + 1];
    out[f + 2] = state.pos[o + 2];
    out[f + 3] = state.vel[o];
    out[f + 4] = state.vel[o + 1];
    out[f + 5] = state.vel[o + 2];
    out[f + 6] = state.alive[i];
  }
}

/** Reusable pool of transferable buffers (ping-pong between worker and main thread). */
export class BufferPool {
  private free: ArrayBuffer[] = [];

  /** Returns a buffer of at least `bytes`, or null when every buffer is still on the main thread. */
  acquire(bytes: number, maxBuffers: number, inFlight: number): ArrayBuffer | null {
    while (this.free.length > 0) {
      const b = this.free.pop()!;
      if (b.byteLength >= bytes) return b;
    }
    return inFlight < maxBuffers ? new ArrayBuffer(bytes) : null;
  }

  release(buffer: ArrayBuffer): void {
    this.free.push(buffer);
  }
}
