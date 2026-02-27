import type { HebeEvent, ClientMsg } from "./types";

type OnEvent = (ev: HebeEvent) => void;
type OnConn = (connected: boolean) => void;

export class WSClient {
  private ws: WebSocket | null = null;
  private url: string;
  private onEvent: OnEvent;
  private onConn: OnConn;
  private shouldReconnect = true;
  private reconnectMs = 800;

  constructor(opts: { url: string; onEvent: OnEvent; onConn: OnConn }) {
    this.url = opts.url;
    this.onEvent = opts.onEvent;
    this.onConn = opts.onConn;
  }

  connect() {
    this.shouldReconnect = true;
    this._connect();
  }

  disconnect() {
    this.shouldReconnect = false;
    try { this.ws?.close(); } catch {}
    this.ws = null;
    this.onConn(false);
  }

  send(msg: ClientMsg) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  private _connect() {
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.onConn(true);
        this.reconnectMs = 800;
      };

      this.ws.onmessage = (e) => {
        try {
          const obj = JSON.parse(e.data);
          // El backend ya manda {type,data,ts}
          if (obj?.type && typeof obj.ts === "number") {
            this.onEvent(obj as HebeEvent);
          } else {
            // tolerante
            this.onEvent({ type: "log", data: obj, ts: Date.now() / 1000 });
          }
        } catch (err) {
          this.onEvent({ type: "error", data: { message: String(err) }, ts: Date.now() / 1000 });
        }
      };

      this.ws.onclose = () => {
        this.onConn(false);
        if (this.shouldReconnect) {
          setTimeout(() => this._connect(), this.reconnectMs);
          this.reconnectMs = Math.min(5000, Math.round(this.reconnectMs * 1.5));
        }
      };

      this.ws.onerror = () => {
        // onclose se encargará del resto
      };
    } catch (err) {
      this.onConn(false);
      if (this.shouldReconnect) {
        setTimeout(() => this._connect(), this.reconnectMs);
      }
    }
  }
}
