/// <reference lib="webworker" />
import type { WorkerCommand } from "./protocol";
import { SimulationRuntime } from "./runtime";

const TICK_INTERVAL_MS = 4;
const ctx = self as unknown as DedicatedWorkerGlobalScope;
const runtime = new SimulationRuntime((event, transfer) => ctx.postMessage(event, transfer));

ctx.onmessage = (e: MessageEvent<WorkerCommand>) => runtime.handle(e.data);

function loop(): void {
  runtime.tick(performance.now());
  setTimeout(loop, TICK_INTERVAL_MS);
}
loop();
