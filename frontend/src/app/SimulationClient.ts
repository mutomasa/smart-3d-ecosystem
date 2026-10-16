import type { WorkerCommand, WorkerEvent } from "../sim/protocol";

type FrameEvent = Extract<WorkerEvent, { type: "frame" }>;
export type SettingsEvent = Extract<WorkerEvent, { type: "settings" }>;
export type FrameHandler = (frame: FrameEvent, data: Float32Array) => void;

/** Main-thread side of the worker: sends commands and hands frames to the renderer. */
export class SimulationClient {
  private readonly worker: Worker;
  private latest: FrameEvent | null = null;
  onSettings: (event: SettingsEvent) => void = () => {};

  constructor(private readonly onLog: (level: string, message: string) => void = console.info) {
    this.worker = new Worker(new URL("../sim/boids.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent<WorkerEvent>) => this.receive(e.data);
  }

  send(cmd: WorkerCommand): void {
    this.worker.postMessage(cmd);
  }

  /** Consumes the most recent frame (if any) and returns its buffer to the worker. */
  consume(handler: FrameHandler): FrameEvent | null {
    const frame = this.latest;
    if (!frame) return null;
    this.latest = null;
    handler(frame, new Float32Array(frame.buffer, 0, frame.count * 7));
    this.giveBack(frame);
    return frame;
  }

  private receive(event: WorkerEvent): void {
    if (event.type === "log") return this.onLog(event.level, event.message);
    if (event.type === "settings") return this.onSettings(event);
    if (event.type === "observations") return; // only emitted when no backend link exists
    if (this.latest) this.giveBack(this.latest); // drop the older, unrendered frame
    this.latest = event;
  }

  private giveBack(frame: FrameEvent): void {
    this.worker.postMessage({ type: "returnBuffer", buffer: frame.buffer } satisfies WorkerCommand, [frame.buffer]);
  }
}
