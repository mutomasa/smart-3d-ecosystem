import type { GroupActionInput } from "../sim/mlPolicy";

export type ConnectionState = "connecting" | "open" | "closed";

export interface PolicyResponseMessage {
  type: "policy_response";
  request_id: string;
  tick: number;
  policy: string;
  model_version: string;
  latency_ms: number;
  actions: GroupActionInput[];
}

export interface SettingsMessage {
  type: "settings";
  version: number;
  settings: Record<string, unknown>;
}

export interface ErrorMessage {
  type: "error";
  request_id?: string | null;
  code: string;
  message: string;
  fallback: "rules";
}

type ServerMessage = PolicyResponseMessage | SettingsMessage | ErrorMessage;

export interface PolicyClientHandlers {
  onActions(msg: PolicyResponseMessage, roundTripMs: number): void;
  onSettings(msg: SettingsMessage): void;
  onError(msg: ErrorMessage): void;
  onState(state: ConnectionState): void;
}

/** Minimal WebSocket surface so tests can inject a fake. */
export interface SocketLike {
  readyState: number;
  onopen: ((ev: unknown) => void) | null;
  onclose: ((ev: unknown) => void) | null;
  onerror: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  send(data: string): void;
  close(): void;
}

export type SocketFactory = (url: string) => SocketLike;
const OPEN = 1;
const MAX_BACKOFF_MS = 8000;
const MAX_PENDING = 64;

/**
 * WebSocket client for `/ws/simulation/{experiment_id}` with automatic reconnection
 * (exponential backoff). It never blocks the simulation: when the socket is down, requests are
 * dropped and the worker falls back to rules on its own.
 */
export class PolicyClient {
  state: ConnectionState = "closed";
  readonly latencies: number[] = [];
  errors = 0;
  private socket: SocketLike | null = null;
  private backoffMs = 500;
  private seq = 0;
  private readonly pending = new Map<string, number>();
  private stopped = false;

  constructor(
    private readonly url: string,
    private readonly handlers: PolicyClientHandlers,
    private readonly factory: SocketFactory = (u) => new WebSocket(u) as unknown as SocketLike,
    private readonly schedule: (fn: () => void, ms: number) => void = (fn, ms) => void setTimeout(fn, ms),
    private readonly now: () => number = () => performance.now(),
  ) {}

  connect(): void {
    this.stopped = false;
    this.setState("connecting");
    const socket = this.factory(this.url);
    this.socket = socket;
    socket.onopen = () => {
      this.backoffMs = 500;
      this.setState("open");
    };
    socket.onmessage = (ev) => this.receive(String(ev.data));
    socket.onerror = () => socket.close();
    socket.onclose = () => this.handleClose(socket);
  }

  close(): void {
    this.stopped = true;
    this.socket?.close();
  }

  /** Sends a batched policy request; returns false when the socket is not open. */
  request(tick: number, groupIds: number[], features: Float32Array, dim: number, policy = "ml"): boolean {
    if (!this.isOpen()) return false;
    const requestId = `req-${++this.seq}`;
    const observations = groupIds.map((g, k) => ({ group_id: g, features: Array.from(features.subarray(k * dim, (k + 1) * dim)) }));
    this.trackPending(requestId);
    this.socket!.send(JSON.stringify({ type: "policy_request", request_id: requestId, tick, policy, observations }));
    return true;
  }

  sendTelemetry(tick: number, mode: string, metrics: Record<string, number>): void {
    if (this.isOpen()) this.socket!.send(JSON.stringify({ type: "telemetry", tick, mode, metrics }));
  }

  /** Round-trip latency percentile (ms) over the recent window, or null without samples. */
  latencyPercentile(p: number): number | null {
    if (this.latencies.length === 0) return null;
    const sorted = [...this.latencies].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
  }

  private isOpen(): boolean {
    return this.state === "open" && this.socket?.readyState === OPEN;
  }

  private trackPending(requestId: string): void {
    this.pending.set(requestId, this.now());
    if (this.pending.size > MAX_PENDING) this.pending.delete(this.pending.keys().next().value!);
  }

  private receive(text: string): void {
    let msg: ServerMessage;
    try {
      msg = JSON.parse(text) as ServerMessage;
    } catch {
      return;
    }
    if (msg.type === "policy_response") return this.handleResponse(msg);
    if (msg.type === "settings") return this.handlers.onSettings(msg);
    if (msg.type === "error") {
      this.errors++;
      if (msg.request_id) this.pending.delete(msg.request_id);
      this.handlers.onError(msg);
    }
  }

  private handleResponse(msg: PolicyResponseMessage): void {
    const sentAt = this.pending.get(msg.request_id);
    this.pending.delete(msg.request_id);
    const rtt = sentAt === undefined ? NaN : this.now() - sentAt;
    if (Number.isFinite(rtt)) {
      this.latencies.push(rtt);
      if (this.latencies.length > 500) this.latencies.shift();
    }
    this.handlers.onActions(msg, rtt);
  }

  private handleClose(socket: SocketLike): void {
    if (socket !== this.socket) return;
    this.pending.clear();
    this.setState("closed");
    if (this.stopped) return;
    this.schedule(() => this.connect(), this.backoffMs);
    this.backoffMs = Math.min(MAX_BACKOFF_MS, this.backoffMs * 2);
  }

  private setState(state: ConnectionState): void {
    this.state = state;
    this.handlers.onState(state);
  }
}

export function experimentIdFromUrl(search: string): string {
  const id = new URLSearchParams(search).get("experiment") ?? "default";
  return /^[A-Za-z0-9_.-]{1,64}$/.test(id) ? id : "default";
}

export function wsUrl(location: Location, experimentId: string): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/simulation/${experimentId}`;
}
