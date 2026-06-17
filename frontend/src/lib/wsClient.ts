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
  private reconnectTimer: number | null = null;

  constructor(opts: { url: string; onEvent: OnEvent; onConn: OnConn }) {
    this.url = opts.url;
    this.onEvent = opts.onEvent;
    this.onConn = opts.onConn;
  }

  connect() {
    this.shouldReconnect = true;
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    this._connect();
  }

  disconnect() {
    this.shouldReconnect = false;
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    const ws = this.ws;
    if (ws) {
      console.log("[HEBE][UI][WS] listener cleanup");
      ws.onopen = null;
      ws.onmessage = null;
      ws.onclose = null;
      ws.onerror = null;
    }
    try { ws?.close(); } catch {}
    this.ws = null;
    console.log("[HEBE][UI][WS] disconnected");
    this.onConn(false);
  }

  send(msg: ClientMsg) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  private _connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      const ws = new WebSocket(this.url);
      this.ws = ws;

      ws.onopen = () => {
        if (this.ws !== ws) return;
        console.log("[HEBE][UI][WS] connected");
        this.onConn(true);
        this.reconnectMs = 800;
      };

      ws.onmessage = (e) => {
        if (this.ws !== ws) return;
        try {
          console.log("[HEBE][UI][WS][RAW]", e.data);
          const obj = JSON.parse(e.data);
          console.log("[HEBE][UI][WS][PARSED]", obj);
          if (obj?.type) {
            const parsedTs = Number(obj.ts);
            this.onEvent({
              ...obj,
              ts: Number.isFinite(parsedTs) ? parsedTs : Date.now() / 1000,
            } as HebeEvent);
          } else {
            this.onEvent({ type: "log", data: obj, ts: Date.now() / 1000 });
          }
        } catch (err) {
          this.onEvent({ type: "error", data: { message: String(err) }, ts: Date.now() / 1000 });
        }
      };

      ws.onclose = () => {
        if (this.ws !== ws) return;
        this.ws = null;
        console.log("[HEBE][UI][WS] disconnected");
        this.onConn(false);
        if (this.shouldReconnect) {
          console.log("[HEBE][UI][WS] reconnecting");
          this.reconnectTimer = window.setTimeout(() => {
            this.reconnectTimer = null;
            this._connect();
          }, this.reconnectMs);
          this.reconnectMs = Math.min(5000, Math.round(this.reconnectMs * 1.5));
        }
      };

      ws.onerror = () => {
        // onclose handles reconnect.
      };
      console.log("[HEBE][UI][WS] listener attached");
    } catch (err) {
      this.onConn(false);
      if (this.shouldReconnect) {
        console.log("[HEBE][UI][WS] reconnecting");
        this.reconnectTimer = window.setTimeout(() => {
          this.reconnectTimer = null;
          this._connect();
        }, this.reconnectMs);
      }
    }
  }
}
