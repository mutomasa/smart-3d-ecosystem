import { RulesPolicy, type StrategyPolicy } from "./policies";
import type { GroupStrategies } from "./strategy";

export interface GroupActionInput {
  group_id: number;
  direction: number[];
  weights: number[];
}

export type MlStatus = "waiting" | "ml" | "holding" | "fallback";

/** Seconds an ML action stays authoritative, then is held, then replaced by rules. */
export const ML_FRESH_S = 0.35;
export const ML_HOLD_S = 1.0;

/**
 * Strategy source fed by the backend. Responses are applied only if they answer a request at
 * least as new as the last applied one (stale ticks are dropped). Without fresh responses the
 * last actions are held for `ML_HOLD_S`, after which the deterministic rules take over.
 */
export class ExternalMlPolicy implements StrategyPolicy {
  readonly mode = "ml" as const;
  status: MlStatus = "waiting";
  modelVersion = "-";
  staleDropped = 0;
  private readonly fallback = new RulesPolicy();
  private actions: GroupActionInput[] = [];
  private lastRequestTick = -1;
  private receivedAt = -Infinity;

  constructor(private readonly now: () => number) {}

  /** Returns false when the response was stale and ignored. */
  receive(requestTick: number, actions: GroupActionInput[], modelVersion: string): boolean {
    if (requestTick < this.lastRequestTick) {
      this.staleDropped++;
      return false;
    }
    this.lastRequestTick = requestTick;
    this.actions = actions;
    this.modelVersion = modelVersion;
    this.receivedAt = this.now();
    return true;
  }

  decide(obs: Float32Array, sizes: Uint32Array, strategies: GroupStrategies): void {
    const age = this.now() - this.receivedAt;
    if (age > ML_HOLD_S) {
      this.status = this.receivedAt === -Infinity ? "waiting" : "fallback";
      this.fallback.decide(obs, sizes, strategies);
      return;
    }
    this.status = age <= ML_FRESH_S ? "ml" : "holding";
    for (const a of this.actions) applyAction(a, strategies);
  }
}

function applyAction(a: GroupActionInput, strategies: GroupStrategies): void {
  const [dx = 0, dy = 0, dz = 0] = a.direction;
  const [wp = 0, wf = 0, wc = 0.5] = a.weights;
  strategies.setTarget(a.group_id, dx, dy, dz, wp, wf, wc);
}
