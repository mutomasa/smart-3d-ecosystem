/** One sample of runtime measurements. */
export interface MetricsSample {
  t: number;
  tick: number;
  fps: number;
  stepMs: number;
  dt: number;
  agents: number;
  alive: number;
  mode: string;
  [key: string]: number | string;
}

/** Collects samples at a fixed interval and exports them as JSON (for the 計測ログ requirement). */
export class MetricsLog {
  readonly samples: MetricsSample[] = [];
  private lastT = -Infinity;

  constructor(private readonly intervalMs = 1000, private readonly maxSamples = 36_000) {}

  /** Records `sample` if the interval has elapsed. Returns true when recorded. */
  maybeRecord(sample: MetricsSample): boolean {
    if (sample.t - this.lastT < this.intervalMs) return false;
    this.lastT = sample.t;
    this.samples.push(sample);
    if (this.samples.length > this.maxSamples) this.samples.shift();
    return true;
  }

  clear(): void {
    this.samples.length = 0;
    this.lastT = -Infinity;
  }

  toJSON(meta: Record<string, unknown> = {}): string {
    return JSON.stringify({ meta, samples: this.samples }, null, 2);
  }
}

export function downloadText(filename: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  a.click();
  URL.revokeObjectURL(url);
}
