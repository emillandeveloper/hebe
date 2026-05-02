import { useEffect, useMemo, useRef, useState } from "react";
import type { HebeEvent } from "./lib/types";
import { WSClient } from "./lib/wsClient";
import { clamp, fmtTime, uid } from "./lib/utils";
import VtuberPreview from "./components/VtuberPreview";


type MsgRole = "user" | "assistant" | "system";
type CurationStatus = "ok" | "no_ok" | "needs_enhancement" | null;

type ChatMsg = {
  id: string;
  role: MsgRole;
  text: string;
  ts: number;
  partial?: boolean;
  traceId?: string;
  sourceMessage?: string;
  sourceUser?: string;
  curation?: CurationStatus;
};

type LangMode = "auto" | "es" | "en";

const LS_KEY = "hebe.ui.settings.v1";

function readSettings(): { volume: number; speed: number; lang: LangMode; showLogs: boolean } {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { volume: 0.9, speed: 1.0, lang: "auto", showLogs: false };
    const j = JSON.parse(raw);
    return {
      volume: clamp(Number(j.volume ?? 0.9), 0, 1),
      speed: clamp(Number(j.speed ?? 1.0), 0.75, 1.25),
      lang: (j.lang === "es" || j.lang === "en" || j.lang === "auto") ? j.lang : "auto",
      showLogs: Boolean(j.showLogs ?? false),
    };
  } catch {
    return { volume: 0.9, speed: 1.0, lang: "auto", showLogs: false };
  }
}

function writeSettings(s: { volume: number; speed: number; lang: LangMode; showLogs: boolean }) {
  localStorage.setItem(LS_KEY, JSON.stringify(s));
}

export default function App() {
  const [connected, setConnected] = useState(false);
  const [backendRunning, setBackendRunning] = useState<boolean | null>(null);
  const [engineStage, setEngineStage] = useState<string>("");
  const [engineReady, setEngineReady] = useState<boolean>(false);

  const [ttsState, setTtsState] = useState<"idle" | "speaking">("idle");
  const [sttLive, setSttLive] = useState<string>("");

  const [messages, setMessages] = useState<ChatMsg[]>(() => ([]));
  const [logs, setLogs] = useState<{ id: string; ev: HebeEvent }[]>([]);
  const [draft, setDraft] = useState<string>("");

  const settings0 = useMemo(() => readSettings(), []);
  const [volume, setVolume] = useState(settings0.volume);
  const [speed, setSpeed] = useState(settings0.speed);
  const [lang, setLang] = useState<LangMode>(settings0.lang);
  const [showLogs, setShowLogs] = useState(settings0.showLogs);

  const listRef = useRef<HTMLDivElement | null>(null);
  const clientRef = useRef<WSClient | null>(null);
  const lastUserRef = useRef<{ text: string; ts: number } | null>(null);

  function pushUser(text: string, ts: number) {
    const t = text.trim();
    if (!t) return;

    const last = lastUserRef.current;
    if (last && last.text === t && Math.abs(ts - last.ts) < 2.0) return;

    lastUserRef.current = { text: t, ts };

    setMessages((prev) => [
      ...prev,
      { id: uid(), role: "user", text: t, ts },
      { id: uid(), role: "assistant", text: "", ts: Date.now() / 1000, partial: true },
    ]);
  }

  const wsUrl = (import.meta as any).env?.VITE_WS_URL || "ws://127.0.0.1:8000/ws";

  function pushLog(ev: HebeEvent) {
    setLogs((prev) => {
      const next = [...prev, { id: uid(), ev }];
      return next.length > 250 ? next.slice(next.length - 250) : next;
    });
  }

  function ensureScrollBottom() {
    const el = listRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }

  function upsertAssistantDraft(deltaOrFinal: string, isFinal: boolean) {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last?.role === "assistant" && last.partial) {
        const updated = { ...last, text: isFinal ? deltaOrFinal : (last.text + deltaOrFinal), partial: !isFinal };
        return [...prev.slice(0, -1), updated];
      }
      const newMsg: ChatMsg = { id: uid(), role: "assistant", text: deltaOrFinal, ts: Date.now()/1000, partial: !isFinal };
      return [...prev, newMsg];
    });
  }

  function attachDatasetExample(data: any, ts: number) {
    const traceId = String(data?.trace_id ?? "").trim();
    const response = String(data?.response ?? "").trim();
    if (!traceId || !response) return;

    const sourceMessage = String(data?.message ?? "").trim();
    const sourceUser = String(data?.chatter_clean || data?.display_name || data?.user_login || "").trim();
    const status = (data?.curation?.status ?? null) as CurationStatus;

    setMessages((prev) => {
      const next = [...prev];

      // 1) Evitar duplicados si llega dos veces el evento.
      const existingIdx = next.findIndex((m) => m.traceId === traceId);
      if (existingIdx >= 0) {
        next[existingIdx] = {
          ...next[existingIdx],
          sourceMessage,
          sourceUser,
          curation: status,
        };
        return next;
      }

      // 2) Encontrar la última respuesta de Hebe con el mismo texto.
      for (let i = next.length - 1; i >= 0; i--) {
        const m = next[i];
        if (m.role === "assistant" && !m.partial && m.text.trim() === response) {
          next[i] = {
            ...m,
            traceId,
            sourceMessage,
            sourceUser,
            curation: status,
          };
          return next;
        }
      }

      // 3) Si no hay burbuja previa porque no llegó llm.final, crearla.
      next.push({
        id: uid(),
        role: "assistant",
        text: response,
        ts,
        traceId,
        sourceMessage,
        sourceUser,
        curation: status,
      });
      return next;
    });
  }

  function markCuration(messageId: string, traceId: string, status: Exclude<CurationStatus, null>) {
    const label = status === "ok" ? "ok" : status === "no_ok" ? "no ok" : "needs enhancement";

    setMessages((prev) => prev.map((m) => (
      m.id === messageId ? { ...m, curation: status } : m
    )));

    const ok = clientRef.current?.send({
      type: "client.command",
      data: {
        name: "dataset_curate",
        payload: {
          trace_id: traceId,
          status,
          tags: [label],
        },
      },
    }) ?? false;

    if (!ok) {
      pushLog({
        type: "error",
        data: { message: "WebSocket no conectado (no pude curar el dataset)" },
        ts: Date.now()/1000,
      });
    }
  }

  function handleEvent(ev: HebeEvent) {
    pushLog(ev);

    switch (ev.type) {
      case "status": {
        if (typeof ev.data?.connected === "boolean") setConnected(ev.data.connected);
        if (typeof ev.data?.running === "boolean") setBackendRunning(ev.data.running);
        if (typeof ev.data?.stage === "string") setEngineStage(ev.data.stage);
        if (typeof ev.data?.engine === "string") setEngineReady(ev.data.engine === "ready");
        break;
      }
      case "stt.partial": {
        setSttLive(String(ev.data?.text ?? ""));
        break;
      }
      case "stt.final": {
        setSttLive("");
        break;
      }
      case "chat.user": {
        const txt = String(ev.data?.text ?? "").trim();
        if (txt) pushUser(txt, ev.ts);
        break;
      }
      case "llm.partial": {
        const d = String(ev.data?.delta ?? "");
        if (d) upsertAssistantDraft(d, false);
        break;
      }
      case "llm.final": {
        const txt = String(ev.data?.text ?? "").trim();
        if (txt) upsertAssistantDraft(txt, true);
        break;
      }
      case "chat.assistant": {
        const txt = String(ev.data?.text ?? "").trim();
        if (txt) {
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last?.role === "assistant" && last.partial) {
              const updated = { ...last, text: txt, ts: ev.ts, partial: false };
              return [...prev.slice(0, -1), updated];
            }
            if (last?.role === "assistant" && !last.partial && last.text === txt) return prev;
            return [...prev, { id: uid(), role: "assistant", text: txt, ts: ev.ts }];
          });
        }
        break;
      }
      case "dataset.example": {
        attachDatasetExample(ev.data, ev.ts);
        break;
      }
      case "dataset.curation.updated": {
        const traceId = String(ev.data?.trace_id ?? "");
        const status = String(ev.data?.status ?? "") as CurationStatus;
        if (traceId && status) {
          setMessages((prev) => prev.map((m) => (
            m.traceId === traceId ? { ...m, curation: status } : m
          )));
        }
        break;
      }
      case "tts.start":
        setTtsState("speaking");
        break;
      case "tts.end":
        setTtsState("idle");
        break;
      case "error":
      default:
        break;
    }

    setTimeout(ensureScrollBottom, 0);
  }

  function sendCommand(name: string, payload?: Record<string, any>) {
    const ok = clientRef.current?.send({ type: "client.command", data: { name, payload } }) ?? false;
    if (!ok) {
      pushLog({ type: "error", data: { message: "WebSocket no conectado (no pude enviar comando)" }, ts: Date.now()/1000 });
    }
  }

  function sendText(text: string) {
    const trimmed = text.trim();
    if (!trimmed) return;

    const ok = clientRef.current?.send({ type: "client.message", data: { text: trimmed } }) ?? false;
    if (!ok) {
      pushLog({ type: "error", data: { message: "WebSocket no conectado (no pude enviar mensaje)" }, ts: Date.now()/1000 });
      return;
    }

    setTimeout(ensureScrollBottom, 0);
  }

  useEffect(() => {
    const client = new WSClient({
      url: wsUrl,
      onEvent: handleEvent,
      onConn: (c) => setConnected(c),
    });
    clientRef.current = client;
    client.connect();
    return () => client.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const s = { volume, speed, lang, showLogs };
    writeSettings(s);
    if (!connected) return;
    sendCommand("set_tts", { volume, speed });
    sendCommand("set_lang", { lang });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volume, speed, lang, showLogs]);

  const [input, setInput] = useState("");
  const startDisabled = backendRunning === true;
  const stopDisabled = backendRunning === false;

  return (
    <div className="app">
      <div className="bgGlow" />

      <div className="shell">
        <header className="topbar glass">
          <div className="brand">
            <div className="avatar">
              <span className="avatarLetter">H</span>
            </div>
            <div className="brandText">
              <div className="brandTitle">Hebe UI</div>
              <div className="brandSub">Chat + STT + TTS + VTuber bridge</div>
            </div>
          </div>

          <div className="pills">
            <div className={"pill " + (connected ? "ok" : "bad")}>
              <span className="dot" />
              {connected ? "Backend conectado" : "Sin conexión"}
            </div>
            <div className={"pill " + (engineReady ? "ok" : "warn")}>
              <span className="dot" />
              {engineReady ? "Hebe lista" : `Arrancando…${engineStage ? " " + engineStage : ""}`}
            </div>
            <div className={"pill " + (ttsState === "idle" ? "" : "warn")}>
              <span className="dot" />
              {ttsState === "idle" ? "TTS: idle" : "TTS: speaking"}
            </div>
          </div>
        </header>

        <main className="grid">
          <section className="glass panel chat">
            <div className="panelHeader">
              <div className="panelTitle">Conversación</div>
              <div className="panelMeta">
                <span className="muted">🎙️ STT live:</span>{" "}
                <span className="mono">{sttLive ? sttLive : "..."}</span>
              </div>
            </div>

            <div className="chatList" ref={listRef}>
              {messages.map((m) => (
                <div key={m.id} className={"bubbleRow " + (m.role === "user" ? "right" : "left")}>
                  <div className={"bubble " + (m.role === "user" ? "user" : "assistant") + (m.traceId ? " datasetLinked" : "")}>
                    <div className="bubbleTop">
                      <span className="bubbleName">{m.role === "user" ? "Tú" : "Hebe"}</span>
                      <span className="bubbleTime">{fmtTime(m.ts)}</span>
                    </div>

                    {m.role === "assistant" && m.sourceMessage && (
                      <div className="replyContext">
                        <div className="replyContextLabel">Responde a {m.sourceUser || "chat"}</div>
                        <div className="replyContextText">{m.sourceMessage}</div>
                      </div>
                    )}

                    <div className={"bubbleText " + (m.partial ? "partial" : "")}>
                      {m.role === "assistant" && m.partial && !m.text ? (
                        <span className="thinkingDots" aria-label="Hebe está pensando">...</span>
                      ) : (
                        m.text
                      )}
                    </div>

                    {m.role === "assistant" && m.traceId && !m.partial && (
                      <div className="curationBar">
                        <button
                          className={"curationBtn ok " + (m.curation === "ok" ? "active" : "")}
                          onClick={() => markCuration(m.id, m.traceId!, "ok")}
                          title="Guardar como ejemplo bueno"
                        >
                          OK
                        </button>
                        <button
                          className={"curationBtn bad " + (m.curation === "no_ok" ? "active" : "")}
                          onClick={() => markCuration(m.id, m.traceId!, "no_ok")}
                          title="Marcar como mal ejemplo"
                        >
                          No OK
                        </button>
                        <button
                          className={"curationBtn warn " + (m.curation === "needs_enhancement" ? "active" : "")}
                          onClick={() => markCuration(m.id, m.traceId!, "needs_enhancement")}
                          title="La idea sirve, pero necesita mejorar"
                        >
                          Mejorar
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <div className="composer">
              <input
                className="input"
                placeholder="Escribe a Hebe…"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                    sendText(input);
                    setInput("");
                  }
                }}
              />
              <button
                className="btn primary"
                onClick={() => {
                  sendText(input);
                  setInput("");
                }}
              >
                Enviar
              </button>
            </div>
            <div className="hint muted">
              Tip: <span className="mono">Ctrl+Enter</span> para enviar.
            </div>
          </section>

          <aside className="glass panel controlPanel">
            <div className="panelHeader slim">
              <div className="panelTitle">Control</div>
              <div className="panelMeta mono">{logs.length} eventos</div>
            </div>

            <div className="controlScroll">
              <div className="btnStack">
                <button className="btn" disabled={startDisabled} onClick={() => sendCommand("start")} title="Arranca pipeline / escucha">▶ Start</button>
                <button className="btn danger" disabled={stopDisabled} onClick={() => sendCommand("stop")} title="Para pipeline / escucha">■ Stop</button>
                <button className="btn" onClick={() => sendCommand("stop_speaking")} title="Corta el audio en reproducción">🔇 Stop Speaking</button>
              </div>

              <div className="card">
                <div className="cardTitle">Voz</div>

                <div className="field">
                  <div className="fieldTop"><span>Volumen</span><span className="mono">{Math.round(volume * 100)}%</span></div>
                  <input type="range" min={0} max={1} step={0.01} value={volume} onChange={(e) => setVolume(Number(e.target.value))} />
                </div>

                <div className="field">
                  <div className="fieldTop"><span>Velocidad</span><span className="mono">{speed.toFixed(2)}x</span></div>
                  <input type="range" min={0.75} max={1.25} step={0.01} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
                </div>

                <div className="field">
                  <div className="fieldTop"><span>Idioma (STT/TTS)</span></div>
                  <select className="select" value={lang} onChange={(e) => setLang(e.target.value as LangMode)}>
                    <option value="auto">Auto</option>
                    <option value="es">Español</option>
                    <option value="en">English</option>
                  </select>
                  <div className="muted small">Si el backend no soporta el comando, no pasa nada.</div>
                </div>
              </div>

              <div className="card">
                <div className="cardTitle row">
                  <span>Estado</span>
                  <label className="toggle">
                    <input type="checkbox" checked={showLogs} onChange={(e) => setShowLogs(e.target.checked)} />
                    <span className="toggleLabel">Logs</span>
                  </label>
                </div>
                <div className="kv">
                  <div className="k">Conexión</div><div className="v">{connected ? "OK" : "OFF"}</div>
                  <div className="k">Hebe</div><div className="v">{engineReady ? "lista" : "arrancando"}</div>
                  <div className="k">TTS</div><div className="v">{ttsState}</div>
                </div>

                {showLogs && (
                  <div className="logBox">
                    {logs.slice(-80).reverse().map((l) => (
                      <div key={l.id} className="logLine">
                        <span className="mono muted">{fmtTime(l.ev.ts)}</span>{" "}
                        <span className={"badge " + (l.ev.type === "error" ? "bad" : l.ev.type.startsWith("tts") ? "warn" : "")}>{l.ev.type}</span>{" "}
                        <span className="mono logMsg">{safeString(l.ev.data)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </aside>

          <aside className="glass panel modelPanel">
            <div className="panelHeader slim">
              <div className="panelTitle">Modelo VTuber</div>
              <div className="panelMeta">preview vertical</div>
            </div>
            <VtuberPreview />
          </aside>
        </main>
      </div>
    </div>
  );
}

function safeString(x: any) {
  if (x == null) return "";
  if (typeof x === "string") return x;
  if (typeof x === "number" || typeof x === "boolean") return String(x);
  try {
    if (x?.message) return String(x.message);
    if (x?.text) return String(x.text);
    return JSON.stringify(x);
  } catch {
    return String(x);
  }
}
