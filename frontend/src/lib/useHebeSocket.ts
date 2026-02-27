import { useEffect, useRef, useState } from "react";
import type { ClientMsg, ServerEvent } from "../type.ts";

export function useHebeSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<ServerEvent[]>([]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as ServerEvent;
      setEvents((prev) => [...prev.slice(-500), msg]); // cap
    };

    return () => ws.close();
  }, []);

  const send = (msg: ClientMsg) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(msg));
  };

  return { connected, events, send };
}
