import type { SocketLike } from "../src/api/websocket";

export class FakeSocket implements SocketLike {
  readyState = 0;
  sent: string[] = [];
  onopen: SocketLike["onopen"] = null;
  onclose: SocketLike["onclose"] = null;
  onerror: SocketLike["onerror"] = null;
  onmessage: SocketLike["onmessage"] = null;
  send(d: string) { this.sent.push(d); }
  close() { this.readyState = 3; this.onclose?.({}); }
  open() { this.readyState = 1; this.onopen?.({}); }
  push(msg: object) { this.onmessage?.({ data: JSON.stringify(msg) }); }
  messages(type: string) { return this.sent.map((s) => JSON.parse(s)).filter((m) => m.type === type); }
}
