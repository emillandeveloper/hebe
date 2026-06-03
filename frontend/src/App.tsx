import type { MouseEvent as ReactMouseEvent } from "react";
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
type ViewMode = "chat" | "database" | "logs";
type LogFilter = "all" | "chat.assistant" | "chat.user" | "twitch" | "stream_context" | "stt" | "tts" | "db" | "errors";

type DbTableInfo = {
  name: string;
  row_count: number;
  column_count: number;
};

type DbColumnInfo = {
  cid: number;
  name: string;
  type: string;
  notnull: boolean;
  default_value: unknown;
  pk: boolean;
  sensitive?: boolean;
};

type DbRowsPayload = {
  table: string;
  total: number;
  limit: number;
  offset: number;
  columns: string[];
  rows: Record<string, unknown>[];
};

type DbCellSelection = {
  table: string;
  column: string;
  value: unknown;
} | null;

const LS_KEY = "hebe.ui.settings.v1";

function readSettings(): { volume: number; speed: number; lang: LangMode } {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { volume: 0.9, speed: 1.0, lang: "auto" };
    const j = JSON.parse(raw);
    return {
      volume: clamp(Number(j.volume ?? 0.9), 0, 1),
      speed: clamp(Number(j.speed ?? 1.0), 0.75, 1.25),
      lang: (j.lang === "es" || j.lang === "en" || j.lang === "auto") ? j.lang : "auto",
    };
  } catch {
    return { volume: 0.9, speed: 1.0, lang: "auto" };
  }
}

function writeSettings(s: { volume: number; speed: number; lang: LangMode }) {
  localStorage.setItem(LS_KEY, JSON.stringify(s));
}

export default function App() {
  const [view, setView] = useState<ViewMode>("chat");
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
  const apiBase = useMemo(() => wsUrl.replace(/^ws/, "http").replace(/\/ws$/, ""), [wsUrl]);

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
    const s = { volume, speed, lang };
    writeSettings(s);
    if (!connected) return;
    sendCommand("set_tts", { volume, speed });
    sendCommand("set_lang", { lang });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volume, speed, lang]);

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

          <div className="viewTabs" role="tablist" aria-label="Vista principal">
            <button
              className={"tabBtn " + (view === "chat" ? "active" : "")}
              onClick={() => setView("chat")}
              role="tab"
              aria-selected={view === "chat"}
            >
              Chat
            </button>
            <button
              className={"tabBtn " + (view === "database" ? "active" : "")}
              onClick={() => setView("database")}
              role="tab"
              aria-selected={view === "database"}
            >
              BBDD
            </button>
            <button
              className={"tabBtn " + (view === "logs" ? "active" : "")}
              onClick={() => setView("logs")}
              role="tab"
              aria-selected={view === "logs"}
            >
              Logs
            </button>
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

        {view === "database" ? (
          <DatabaseInspector apiBase={apiBase} />
        ) : view === "logs" ? (
          <LogsView logs={logs} onClearVisible={(ids) => setLogs((prev) => prev.filter((item) => !ids.has(item.id)))} />
        ) : (
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

            <QuickControlToolbar
              disabled={!connected}
              onCommand={(command) => sendText(command)}
            />

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
                  <button className="btn compact" onClick={() => setView("logs")}>Abrir Logs</button>
                </div>
                <div className="kv">
                  <div className="k">Conexión</div><div className="v">{connected ? "OK" : "OFF"}</div>
                  <div className="k">Hebe</div><div className="v">{engineReady ? "lista" : "arrancando"}</div>
                  <div className="k">TTS</div><div className="v">{ttsState}</div>
                </div>

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
        )}
      </div>
    </div>
  );
}

function QuickControlToolbar({ disabled, onCommand }: { disabled: boolean; onCommand: (command: string) => void }) {
  const groups = [
    {
      title: "Stream",
      items: [
        ["🔄", "Actualizar contexto", "Hebe, actualiza contexto de stream"],
        ["📡", "Estado stream", "Hebe, qué contexto de stream tienes"],
        ["🧠", "Contexto partida", "Hebe, qué contexto de partida tienes"],
        ["💾", "Snapshot stream", "Hebe, resume este stream"],
        ["🏁", "Finalizar stream", "Hebe, finaliza stream"],
      ],
    },
    {
      title: "Espontaneidad",
      items: [
        ["💬", "Estado", "Hebe, estado de espontaneidad"],
        ["⏸️", "Pausar", "Hebe, pausa espontaneidad"],
        ["▶️", "Activar", "Hebe, activa espontaneidad"],
        ["🤫", "Reactiva", "Hebe, modo reactiva"],
        ["🧍", "Compañera", "Hebe, modo compañera"],
        ["🎭", "Show", "Hebe, modo show"],
        ["🔇", "Idle voz OFF", "Hebe, espontaneidad en texto"],
        ["🔊", "Idle voz ON", "Hebe, espontaneidad con voz"],
      ],
    },
    {
      title: "Voz",
      items: [
        ["🎙️", "STT amb ON", "Hebe, activa STT ambiental"],
        ["🎙️", "STT amb OFF", "Hebe, desactiva STT ambiental"],
        ["👂", "Qué ha oído", "Hebe, qué ha oído del stream"],
        ["🧹", "Limpiar oído", "Hebe, limpia oído del stream"],
        ["🔊", "Activar voz", "Hebe, activa la voz"],
        ["🔇", "Solo texto", "Hebe, desactiva la voz"],
        ["🛑", "Stop voz", "Hebe, desactiva la voz"],
      ],
    },
    {
      title: "Twitch",
      items: [
        ["📣", "Probar SO", "Hebe, previsualiza shoutout a tester"],
        ["🧪", "Probar raid", "Hebe, prueba raid"],
        ["👥", "Chat", "Hebe, qué está pasando en chat"],
      ],
    },
  ] as const;

  return (
    <div className="quickToolbar">
      {groups.map((group) => (
        <div className="quickGroup" key={group.title}>
          <div className="quickGroupTitle">{group.title}</div>
          <div className="quickButtons">
            {group.items.map(([icon, label, command]) => (
              <button
                className="quickBtn"
                key={command}
                disabled={disabled}
                onClick={() => onCommand(command)}
                title={command}
              >
                <span className="quickIcon">{icon}</span>
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function LogsView({ logs, onClearVisible }: { logs: { id: string; ev: HebeEvent }[]; onClearVisible: (ids: Set<string>) => void }) {
  const [filter, setFilter] = useState<LogFilter>("all");
  const [search, setSearch] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [wrap, setWrap] = useState(true);
  const listRef = useRef<HTMLDivElement | null>(null);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return logs.filter(({ ev }) => {
      const type = ev.type || "";
      const text = `${type} ${safeString(ev.data)}`.toLowerCase();
      const matchesSearch = !needle || text.includes(needle);
      const matchesFilter =
        filter === "all" ||
        type === filter ||
        (filter === "twitch" && type.includes("twitch")) ||
        (filter === "stream_context" && (type.includes("stream") || text.includes("stream_context"))) ||
        (filter === "stt" && type.startsWith("stt")) ||
        (filter === "tts" && type.startsWith("tts")) ||
        (filter === "db" && text.includes("db")) ||
        (filter === "errors" && type === "error");
      return matchesSearch && matchesFilter;
    });
  }, [logs, filter, search]);

  useEffect(() => {
    if (autoScroll && listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [filtered.length, autoScroll]);

  async function copyLogs() {
    const text = filtered.map(({ ev }) => `${fmtTime(ev.ts)} ${ev.type} ${safeString(ev.data)}`).join("\n");
    await navigator.clipboard.writeText(text);
  }

  const filters: LogFilter[] = ["all", "chat.assistant", "chat.user", "twitch", "stream_context", "stt", "tts", "db", "errors"];

  return (
    <main className="logsLayout">
      <section className="glass panel logsPanel">
        <div className="panelHeader">
          <div>
            <div className="panelTitle">Logs</div>
            <div className="panelMeta">{filtered.length} visibles / {logs.length} eventos</div>
          </div>
          <div className="logsActions">
            <input className="input logsSearch" placeholder="Buscar logs..." value={search} onChange={(e) => setSearch(e.target.value)} />
            <select className="select logsFilter" value={filter} onChange={(e) => setFilter(e.target.value as LogFilter)}>
              {filters.map((item) => <option value={item} key={item}>{item}</option>)}
            </select>
            <label className="toggle mini"><input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} /><span className="toggleLabel">Auto</span></label>
            <label className="toggle mini"><input type="checkbox" checked={wrap} onChange={(e) => setWrap(e.target.checked)} /><span className="toggleLabel">Wrap</span></label>
            <button className="btn compact" onClick={copyLogs}>Copiar</button>
            <button className="btn compact danger" onClick={() => onClearVisible(new Set(filtered.map((item) => item.id)))}>Limpiar visibles</button>
          </div>
        </div>
        <div className={"logsFullBox " + (wrap ? "wrap" : "nowrap")} ref={listRef}>
          {filtered.map(({ id, ev }) => (
            <div className="logFullLine" key={id}>
              <span className="mono muted">{fmtTime(ev.ts)}</span>
              <span className={"badge " + (ev.type === "error" ? "bad" : ev.type.startsWith("tts") ? "warn" : "")}>{ev.type}</span>
              <span className="mono logMsg">{safeString(ev.data)}</span>
            </div>
          ))}
          {filtered.length === 0 && <div className="emptyState">No hay logs con ese filtro.</div>}
        </div>
      </section>
    </main>
  );
}

function DatabaseInspector({ apiBase }: { apiBase: string }) {
  const [tables, setTables] = useState<DbTableInfo[]>([]);
  const [selectedTable, setSelectedTable] = useState<string>("");
  const [schema, setSchema] = useState<DbColumnInfo[]>([]);
  const [rowsPayload, setRowsPayload] = useState<DbRowsPayload | null>(null);
  const [limit, setLimit] = useState(50);
  const [offset, setOffset] = useState(0);
  const [tableFilter, setTableFilter] = useState("");
  const [rowFilter, setRowFilter] = useState("");
  const [columnWidths, setColumnWidths] = useState<Record<string, number>>({});
  const [selectedRow, setSelectedRow] = useState<Record<string, unknown> | null>(null);
  const [selectedCell, setSelectedCell] = useState<DbCellSelection>(null);
  const [loadingTables, setLoadingTables] = useState(false);
  const [loadingRows, setLoadingRows] = useState(false);
  const [error, setError] = useState("");

  async function readJson<T>(url: string): Promise<T> {
    const res = await fetch(url);
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = String(payload?.detail || payload?.message || "Read failed");
      throw new Error(detail);
    }
    return payload as T;
  }

  async function loadTables() {
    setLoadingTables(true);
    setError("");
    try {
      const payload = await readJson<{ tables: DbTableInfo[] }>(`${apiBase}/debug/db/tables`);
      setTables(payload.tables || []);
      setSelectedTable((prev) => {
        if (prev && payload.tables?.some((table) => table.name === prev)) return prev;
        return payload.tables?.[0]?.name || "";
      });
    } catch (exc) {
      setTables([]);
      setSelectedTable("");
      setSchema([]);
      setRowsPayload(null);
      setError(exc instanceof Error ? exc.message : "Database read failed");
    } finally {
      setLoadingTables(false);
    }
  }

  async function loadSelectedTable(tableName: string, nextLimit = limit, nextOffset = offset) {
    if (!tableName) return;
    setLoadingRows(true);
    setError("");
    try {
      const [schemaPayload, rows] = await Promise.all([
        readJson<{ columns: DbColumnInfo[] }>(`${apiBase}/debug/db/tables/${encodeURIComponent(tableName)}/schema`),
        readJson<DbRowsPayload>(
          `${apiBase}/debug/db/tables/${encodeURIComponent(tableName)}/rows?limit=${nextLimit}&offset=${nextOffset}`
        ),
      ]);
      setSchema(schemaPayload.columns || []);
      setRowsPayload(rows);
    } catch (exc) {
      setSchema([]);
      setRowsPayload(null);
      setError(exc instanceof Error ? exc.message : "Database read failed");
    } finally {
      setLoadingRows(false);
    }
  }

  function chooseTable(tableName: string) {
    setSelectedTable(tableName);
    setOffset(0);
    setRowFilter("");
    setSelectedRow(null);
    setSelectedCell(null);
    try {
      const raw = localStorage.getItem(`hebe.dbInspector.widths.${tableName}`);
      setColumnWidths(raw ? JSON.parse(raw) : {});
    } catch {
      setColumnWidths({});
    }
  }

  function refreshAll() {
    loadTables();
    if (selectedTable) loadSelectedTable(selectedTable);
  }

  useEffect(() => {
    loadTables();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase]);

  useEffect(() => {
    if (selectedTable) loadSelectedTable(selectedTable);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTable, limit, offset]);

  const filteredTables = useMemo(() => {
    const needle = tableFilter.trim().toLowerCase();
    if (!needle) return tables;
    return tables.filter((table) => table.name.toLowerCase().includes(needle));
  }, [tables, tableFilter]);

  const visibleRows = useMemo(() => {
    const rows = rowsPayload?.rows || [];
    const needle = rowFilter.trim().toLowerCase();
    if (!needle) return rows;
    return rows.filter((row) => Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(needle)));
  }, [rowsPayload, rowFilter]);

  const selectedInfo = tables.find((table) => table.name === selectedTable);
  const total = rowsPayload?.total ?? selectedInfo?.row_count ?? 0;
  const canPrev = offset > 0;
  const canNext = offset + limit < total;
  const visibleColumns = rowsPayload?.columns || schema.map((column) => column.name);

  function setColumnWidth(column: string, width: number) {
    const next = { ...columnWidths, [column]: Math.max(80, Math.min(720, Math.round(width))) };
    setColumnWidths(next);
    if (selectedTable) {
      localStorage.setItem(`hebe.dbInspector.widths.${selectedTable}`, JSON.stringify(next));
    }
  }

  function beginColumnResize(column: string, event: ReactMouseEvent<HTMLSpanElement>) {
    event.preventDefault();
    event.stopPropagation();
    const startX = event.clientX;
    const startWidth = columnWidths[column] || 180;
    const onMove = (moveEvent: MouseEvent) => {
      setColumnWidth(column, startWidth + moveEvent.clientX - startX);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  async function copyText(value: unknown) {
    const text = formatDbValue(value, true);
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      document.body.removeChild(area);
    }
  }

  return (
    <main className="dbLayout">
      <section className="glass panel dbTablesPanel">
        <div className="panelHeader slim">
          <div>
            <div className="panelTitle">BBDD</div>
            <div className="panelMeta">{tables.length} tablas</div>
          </div>
          <button className="btn compact" onClick={refreshAll} disabled={loadingTables || loadingRows}>
            Refresh
          </button>
        </div>

        <input
          className="input dbSearch"
          placeholder="Filtrar tablas..."
          value={tableFilter}
          onChange={(e) => setTableFilter(e.target.value)}
        />

        <div className="dbTableList">
          {filteredTables.map((table) => (
            <button
              key={table.name}
              className={"dbTableItem " + (table.name === selectedTable ? "active" : "")}
              onClick={() => chooseTable(table.name)}
            >
              <span className="dbTableName">{table.name}</span>
              <span className="dbTableMeta">{table.row_count} filas · {table.column_count} cols</span>
            </button>
          ))}
          {!loadingTables && filteredTables.length === 0 && (
            <div className="emptyState">No hay tablas que coincidan.</div>
          )}
        </div>
      </section>

      <section className="glass panel dbDataPanel">
        <div className="panelHeader">
          <div>
            <div className="panelTitle">{selectedTable || "Database Inspector"}</div>
            <div className="panelMeta">
              {selectedTable ? `${total} filas totales` : "Selecciona una tabla"}
            </div>
          </div>
          <div className="dbActions">
            <input
              className="input dbRowFilter"
              placeholder="Filtrar filas cargadas..."
              value={rowFilter}
              onChange={(e) => setRowFilter(e.target.value)}
            />
            <select
              className="select dbPageSize"
              value={limit}
              onChange={(e) => {
                setLimit(Number(e.target.value));
                setOffset(0);
              }}
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={250}>250</option>
            </select>
          </div>
        </div>

        {error && <div className="errorBox">{error}</div>}

        {selectedTable && (
          <>
            <div className="dbSchema">
              {schema.map((column) => (
                <div key={column.cid} className="dbColumn">
                  <span className="dbColumnName">{column.name}</span>
                  <span className="dbColumnType">{column.type || "ANY"}</span>
                  {column.pk && <span className="badge warn">PK</span>}
                  {column.sensitive && <span className="badge bad">masked</span>}
                </div>
              ))}
            </div>

            <div className="dbRowsWrap">
              <table className="dbDataTable">
                <thead>
                  <tr>
                    {(rowsPayload?.columns || schema.map((column) => column.name)).map((column) => (
                      <th key={column} style={{ width: columnWidths[column] || 180, minWidth: columnWidths[column] || 180 }}>
                        <span>{column}</span>
                        <span
                          className="dbResizeHandle"
                          onMouseDown={(event) => beginColumnResize(column, event)}
                          title="Resize column"
                        />
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {visibleRows.map((row, idx) => (
                    <tr key={`${offset}-${idx}`} className={selectedRow === row ? "selected" : ""} onClick={() => setSelectedRow(row)}>
                      {visibleColumns.map((column) => (
                        <td
                          key={column}
                          style={{ width: columnWidths[column] || 180, minWidth: columnWidths[column] || 180 }}
                          onClick={(event) => {
                            event.stopPropagation();
                            setSelectedRow(row);
                            setSelectedCell({ table: selectedTable, column, value: row[column] });
                          }}
                          title="Click para ver el valor completo"
                        >
                          {formatDbValue(row[column])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>

              {!loadingRows && selectedTable && total === 0 && (
                <div className="emptyState dbEmpty">Table is empty</div>
              )}
              {!loadingRows && total > 0 && visibleRows.length === 0 && (
                <div className="emptyState dbEmpty">No hay filas cargadas que coincidan.</div>
              )}
            </div>

            <div className="dbPagination">
              <button className="btn compact" disabled={!canPrev || loadingRows} onClick={() => setOffset(Math.max(0, offset - limit))}>
                Previous
              </button>
              <span className="mono muted">
                {total === 0 ? "0-0" : `${offset + 1}-${Math.min(offset + limit, total)}`} / {total}
              </span>
              <button className="btn compact" disabled={!canNext || loadingRows} onClick={() => setOffset(offset + limit)}>
                Next
              </button>
            </div>

            {selectedRow && (
              <div className="dbDetailPanel">
                <div className="dbDetailHeader">
                  <div className="panelTitle">Detalle de fila</div>
                  <div className="dbDetailActions">
                    <button className="btn compact" onClick={() => copyText(selectedRow)}>Copiar JSON</button>
                    <button className="btn compact" onClick={() => setSelectedRow(null)}>Cerrar</button>
                  </div>
                </div>
                <div className="dbDetailGrid">
                  {visibleColumns.map((column) => (
                    <div className="dbDetailItem" key={column}>
                      <div className="dbDetailKey">{column}</div>
                      <pre className="dbDetailValue">{formatDbValue(selectedRow[column], true)}</pre>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {!selectedTable && !error && (
          <div className="emptyState">Database not found or no tables available.</div>
        )}
      </section>

      {selectedCell && (
        <div className="dbModalBackdrop" onClick={() => setSelectedCell(null)}>
          <div className="dbModal" onClick={(event) => event.stopPropagation()}>
            <div className="dbModalHeader">
              <div>
                <div className="panelTitle">{selectedCell.column}</div>
                <div className="panelMeta">{selectedCell.table}</div>
              </div>
              <div className="dbDetailActions">
                <button className="btn compact" onClick={() => copyText(selectedCell.value)}>Copiar</button>
                <button className="btn compact" onClick={() => setSelectedCell(null)}>Cerrar</button>
              </div>
            </div>
            <pre className="dbFullValue">{formatDbValue(selectedCell.value, true)}</pre>
          </div>
        </div>
      )}
    </main>
  );
}

function formatDbValue(value: unknown, pretty = false) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value, null, pretty ? 2 : 0);
  const text = String(value);
  if (!pretty) return text;
  const trimmed = text.trim();
  if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
    try {
      return JSON.stringify(JSON.parse(trimmed), null, 2);
    } catch {
      return text;
    }
  }
  return text;
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
