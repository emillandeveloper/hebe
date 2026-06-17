import { useEffect, useRef, useState } from "react";
import type { ClientMsg, ServerEvent } from "../type.ts";
import { WSClient } from "./wsClient";

export function useHebeSocket() {
  const clientRef = useRef<WSClient | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<ServerEvent[]>([]);

  useEffect(() => {
    const client = new WSClient({
      url: "ws://localhost:8000/ws",
      onConn: setConnected,
      onEvent: (msg) => setEvents((prev) => [...prev.slice(-500), msg as ServerEvent]),
    });
    clientRef.current = client;
    client.connect();
    return () => client.disconnect();
  }, []);

  const send = (msg: ClientMsg) => {
    clientRef.current?.send(msg);
  };

  return { connected, events, send };
}
