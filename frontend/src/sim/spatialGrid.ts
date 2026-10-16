/**
 * Uniform 3D grid built with a counting sort every step.
 * A query visits the agent's own cell and its 26 neighbours only, so the cost depends on
 * local density instead of O(N^2). Positions outside the world are clamped to border cells.
 */
export class SpatialGrid {
  readonly dims: [number, number, number];
  private readonly cellSize: number;
  private readonly halfExtents: [number, number, number];
  private cellStart: Int32Array;
  private cellCount: Int32Array;
  private cellOf = new Int32Array(0);
  private sorted = new Int32Array(0);
  /** Largest number of agents found in a single cell during the last rebuild. */
  maxOccupancy = 0;

  constructor(halfExtents: [number, number, number], cellSize: number) {
    this.cellSize = cellSize;
    this.halfExtents = halfExtents;
    this.dims = halfExtents.map((h) => Math.max(1, Math.ceil((2 * h) / cellSize))) as [number, number, number];
    const cells = this.dims[0] * this.dims[1] * this.dims[2];
    this.cellStart = new Int32Array(cells + 1);
    this.cellCount = new Int32Array(cells);
  }

  get cellTotal(): number {
    return this.cellCount.length;
  }

  rebuild(pos: Float32Array, alive: Uint8Array, count: number): void {
    this.ensureCapacity(count);
    this.countCells(pos, alive, count);
    this.prefixSum();
    this.scatter(alive, count);
  }

  /** Agent indices sorted by cell; index ranges returned by `queryRanges` point into this array. */
  get sortedIndices(): Int32Array {
    return this.sorted;
  }

  /**
   * Writes `[start, end)` pairs into `ranges` (length >= 54) for the agent's own cell followed by
   * its 26 neighbours, and returns the number of ranges. Nothing is truncated.
   */
  queryRanges(x: number, y: number, z: number, ranges: Int32Array): number {
    const cx = this.axisCell(x, 0);
    const cy = this.axisCell(y, 1);
    const cz = this.axisCell(z, 2);
    let r = this.pushRange(this.cellIndex(cx, cy, cz), ranges, 0);
    for (let dz = -1; dz <= 1; dz++) {
      const iz = cz + dz;
      if (iz < 0 || iz >= this.dims[2]) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const iy = cy + dy;
        if (iy < 0 || iy >= this.dims[1]) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const ix = cx + dx;
          if (ix < 0 || ix >= this.dims[0] || (dx === 0 && dy === 0 && dz === 0)) continue;
          r = this.pushRange(this.cellIndex(ix, iy, iz), ranges, r);
        }
      }
    }
    return r;
  }

  /** Convenience: copies candidate indices (own cell first) into `out`, up to its length. */
  query(x: number, y: number, z: number, out: Int32Array): number {
    const ranges = new Int32Array(54);
    const r = this.queryRanges(x, y, z, ranges);
    let n = 0;
    for (let k = 0; k < r && n < out.length; k++) {
      for (let s = ranges[k * 2]; s < ranges[k * 2 + 1] && n < out.length; s++) out[n++] = this.sorted[s];
    }
    return n;
  }

  cellOfPosition(x: number, y: number, z: number): number {
    return this.cellIndex(this.axisCell(x, 0), this.axisCell(y, 1), this.axisCell(z, 2));
  }

  private ensureCapacity(count: number): void {
    if (this.sorted.length >= count) return;
    this.sorted = new Int32Array(count);
    this.cellOf = new Int32Array(count);
  }

  private countCells(pos: Float32Array, alive: Uint8Array, count: number): void {
    this.cellCount.fill(0);
    for (let i = 0; i < count; i++) {
      if (!alive[i]) continue;
      const c = this.cellOfPosition(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
      this.cellOf[i] = c;
      this.cellCount[c]++;
    }
  }

  private prefixSum(): void {
    let sum = 0;
    let max = 0;
    for (let c = 0; c < this.cellCount.length; c++) {
      this.cellStart[c] = sum;
      sum += this.cellCount[c];
      if (this.cellCount[c] > max) max = this.cellCount[c];
    }
    this.cellStart[this.cellCount.length] = sum;
    this.maxOccupancy = max;
  }

  private scatter(alive: Uint8Array, count: number): void {
    this.cellCount.fill(0);
    for (let i = 0; i < count; i++) {
      if (!alive[i]) continue;
      const c = this.cellOf[i];
      this.sorted[this.cellStart[c] + this.cellCount[c]++] = i;
    }
  }

  private pushRange(cell: number, ranges: Int32Array, r: number): number {
    const start = this.cellStart[cell];
    const end = this.cellStart[cell + 1];
    if (end === start) return r;
    ranges[r * 2] = start;
    ranges[r * 2 + 1] = end;
    return r + 1;
  }

  private axisCell(v: number, axis: 0 | 1 | 2): number {
    const c = Math.floor((v + this.halfExtents[axis]) / this.cellSize);
    return Math.min(this.dims[axis] - 1, Math.max(0, c));
  }

  private cellIndex(ix: number, iy: number, iz: number): number {
    return ix + this.dims[0] * (iy + this.dims[1] * iz);
  }
}
