/** Rolling frames-per-second estimate over a time window. */
export class FpsMeter {
  private stamps: number[] = [];

  constructor(private readonly windowMs = 1000) {}

  frame(nowMs: number): number {
    this.stamps.push(nowMs);
    while (this.stamps.length > 0 && nowMs - this.stamps[0] > this.windowMs) this.stamps.shift();
    return this.fps();
  }

  fps(): number {
    if (this.stamps.length < 2) return 0;
    const span = this.stamps[this.stamps.length - 1] - this.stamps[0];
    return span > 0 ? ((this.stamps.length - 1) * 1000) / span : 0;
  }
}
