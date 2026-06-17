import type { MouseEvent as ReactMouseEvent, ReactNode } from "react";
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
type ViewMode = "chat" | "session" | "audio" | "dev" | "logs" | "database";
type LogFilter = "all" | "chat.assistant" | "chat.user" | "twitch" | "stream_context" | "stt" | "tts" | "memory" | "routing" | "dev" | "spontaneity" | "db" | "errors";

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

type AudioInputDevice = {
  id: string;
  index?: number;
  name: string;
  host_api?: string;
  is_default?: boolean;
  is_default_input?: boolean;
  is_loopback?: boolean;
  channels?: number;
  sample_rate?: number;
  max_input_channels?: number;
  max_output_channels?: number;
  default_sample_rate?: number;
  display_label?: string;
  signature?: string;
  host_api_warning?: string;
};

type VoiceCommandDebug = {
  raw_text?: string;
  normalized_text?: string;
  detected_script?: string;
  script?: string;
  retry_attempted?: boolean;
  retry_transcript?: string;
  final_decision?: string;
  intent?: string;
  target?: string;
  confidence?: number;
  status?: string;
  reason?: string;
};

type DevBackendStatus = {
  devEnabled?: boolean;
  running?: boolean;
  pid?: number | null;
  uptimeMs?: number;
  lastRestartTime?: string | null;
  lastError?: string;
  status?: string;
  ok?: boolean;
  error?: string;
};

type HebeDevBridge = {
  enabled: boolean;
  reloadUi?: () => Promise<DevBackendStatus>;
  restartBackend?: () => Promise<DevBackendStatus>;
  fullReset?: () => Promise<DevBackendStatus>;
  getBackendStatus?: () => Promise<DevBackendStatus>;
  onBackendStatus?: (callback: (status: DevBackendStatus) => void) => () => void;
};

declare global {
  interface Window {
    hebeDev?: HebeDevBridge;
  }
}

const LS_KEY = "hebe.ui.settings.v1";
const FULL_RESET_PENDING_KEY = "hebe.dev.fullResetPending";

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
  const [hebeSleeping, setHebeSleeping] = useState<boolean>(false);
  const [wakeRequired, setWakeRequired] = useState<boolean>(false);
  const [wakeLoopAlive, setWakeLoopAlive] = useState<boolean | null>(null);
  const [wakeLoopError, setWakeLoopError] = useState<string>("");

  const [ttsState, setTtsState] = useState<"idle" | "speaking">("idle");
  const [ttsEnabled, setTtsEnabled] = useState<boolean | null>(null);
  const [sttStatus, setSttStatus] = useState<string>("off");
  const [sttLive, setSttLive] = useState<string>("");
  const [sttLevel, setSttLevel] = useState<number>(0);
  const [sttRms, setSttRms] = useState<number>(0);
  const [sttPeak, setSttPeak] = useState<number>(0);
  const [lastSttFinal, setLastSttFinal] = useState<string>("");
  const [lastSttLevelAt, setLastSttLevelAt] = useState<number>(Date.now());
  const [uiTick, setUiTick] = useState<number>(Date.now());
  const [micDevices, setMicDevices] = useState<AudioInputDevice[]>([]);
  const [selectedMicId, setSelectedMicId] = useState<string>("");
  const [selectedMicName, setSelectedMicName] = useState<string>("");
  const [selectedMicHostApi, setSelectedMicHostApi] = useState<string>("");
  const [micTestResult, setMicTestResult] = useState<any>(null);
  const [micError, setMicError] = useState<string>("");
  const [voiceCommandDebug, setVoiceCommandDebug] = useState<VoiceCommandDebug | null>(null);
  const [devStatus, setDevStatus] = useState<DevBackendStatus>(() => ({
    devEnabled: Boolean(window.hebeDev?.enabled),
    status: "unknown",
  }));
  const [devBusy, setDevBusy] = useState<"" | "reload" | "restart" | "full">("");
  const [sessionDebug, setSessionDebug] = useState<any | null>(null);
  const [sessionLoading, setSessionLoading] = useState(false);
  const [sessionError, setSessionError] = useState("");

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
        if (typeof ev.data?.tts_enabled === "boolean") setTtsEnabled(ev.data.tts_enabled);
        if (typeof ev.data?.hebe_sleeping === "boolean") setHebeSleeping(ev.data.hebe_sleeping);
        if (typeof ev.data?.wake_required === "boolean") setWakeRequired(ev.data.wake_required);
        if (typeof ev.data?.wake_loop_alive === "boolean") setWakeLoopAlive(ev.data.wake_loop_alive);
        if (typeof ev.data?.wake_loop_error === "string") setWakeLoopError(ev.data.wake_loop_error);
        if (typeof ev.data?.stt_enabled === "boolean") setSttStatus(ev.data.stt_enabled ? "listening" : "off");
        if (typeof ev.data?.stt === "string") setSttStatus(ev.data.stt);
        if (typeof ev.data?.last_stt_error === "string" && ev.data.last_stt_error) setMicError(ev.data.last_stt_error);
        if (ev.data?.stt_input_device && typeof ev.data.stt_input_device === "object") {
          const device = ev.data.stt_input_device as any;
          if (typeof device.device_id === "string") setSelectedMicId(device.device_id);
          if (typeof device.device_name === "string") setSelectedMicName(device.device_name);
          if (typeof device.host_api === "string") setSelectedMicHostApi(device.host_api);
          if (typeof device.error === "string") setMicError(device.error);
          if (typeof device.failed_error === "string" && device.failed_error) setMicError(device.failed_error);
        }
        break;
      }
      case "stt.partial": {
        const text = String(ev.data?.text ?? "");
        const parsedLevel = Number((text.match(/lvl\s+([0-9.]+)/i) || [])[1] ?? 0);
        const parsedRms = Number((text.match(/rms\s+([0-9.]+)/i) || [])[1] ?? parsedLevel);
        const parsedPeak = Number((text.match(/peak\s+([0-9.]+)/i) || [])[1] ?? parsedLevel);
        const level = typeof ev.data?.level === "number" ? Number(ev.data.level) : parsedLevel;
        const rms = typeof ev.data?.rms === "number" ? Number(ev.data.rms) : parsedRms;
        const peak = typeof ev.data?.peak === "number" ? Number(ev.data.peak) : parsedPeak;
        setSttLive(text);
        if (Number.isFinite(level)) {
          setSttLevel(level);
          setSttRms(Number.isFinite(rms) ? rms : level);
          setSttPeak(Number.isFinite(peak) ? peak : level);
          if (level > 0.001 || rms > 0.003) setLastSttLevelAt(Date.now());
        }
        break;
      }
      case "stt.final": {
        const text = String(ev.data?.text ?? "").trim();
        setSttLive("");
        if (text) setLastSttFinal(text);
        break;
      }
      case "voice.command": {
        setVoiceCommandDebug({
          raw_text: String(ev.data?.raw_text ?? ""),
          normalized_text: String(ev.data?.normalized_text ?? ""),
          detected_script: ev.data?.detected_script ? String(ev.data.detected_script) : ev.data?.script ? String(ev.data.script) : "",
          script: ev.data?.script ? String(ev.data.script) : "",
          retry_attempted: typeof ev.data?.retry_attempted === "boolean" ? Boolean(ev.data.retry_attempted) : undefined,
          retry_transcript: ev.data?.retry_transcript ? String(ev.data.retry_transcript) : "",
          final_decision: ev.data?.final_decision ? String(ev.data.final_decision) : "",
          intent: ev.data?.intent ? String(ev.data.intent) : "",
          target: ev.data?.target ? String(ev.data.target) : "",
          confidence: typeof ev.data?.confidence === "number" ? Number(ev.data.confidence) : undefined,
          status: ev.data?.status ? String(ev.data.status) : "",
          reason: ev.data?.reason ? String(ev.data.reason) : "",
        });
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

  async function refreshDevStatus() {
    const bridge = window.hebeDev;
    if (!bridge?.enabled || !bridge.getBackendStatus) return;
    try {
      setDevStatus(await bridge.getBackendStatus());
    } catch (error) {
      setDevStatus((prev) => ({ ...prev, status: "failed", lastError: error instanceof Error ? error.message : String(error) }));
    }
  }

  async function runDevAction(action: "reload" | "restart" | "full") {
    const bridge = window.hebeDev;
    if (!bridge?.enabled) return;
    setDevBusy(action);
    try {
      if (action === "reload") {
        await bridge.reloadUi?.();
        return;
      }
      if (action === "full") {
        sessionStorage.setItem(FULL_RESET_PENDING_KEY, "1");
        const result = await bridge.fullReset?.();
        if (result) setDevStatus(result);
        if (result && result.ok === false) sessionStorage.removeItem(FULL_RESET_PENDING_KEY);
        return;
      }
      const result = await bridge.restartBackend?.();
      if (result) setDevStatus(result);
      if (result?.ok) {
        clientRef.current?.disconnect();
        clientRef.current?.connect();
      }
    } catch (error) {
      setDevStatus((prev) => ({ ...prev, status: "failed", lastError: error instanceof Error ? error.message : String(error) }));
    } finally {
      if (action !== "reload" && action !== "full") setDevBusy("");
    }
  }

  async function refreshSessionDebug() {
    setSessionLoading(true);
    setSessionError("");
    try {
      const res = await fetch(`${apiBase}/debug/live-session`);
      const payload = await res.json().catch(() => ({}));
      if (!res.ok || payload?.ok === false) throw new Error(payload?.reason || payload?.detail || "No live session data yet");
      setSessionDebug(payload);
    } catch (error) {
      setSessionError(error instanceof Error ? error.message : String(error));
    } finally {
      setSessionLoading(false);
    }
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
    const bridge = window.hebeDev;
    if (!bridge?.enabled) return;
    refreshDevStatus();
    const unsubscribe = bridge.onBackendStatus?.((status) => setDevStatus(status));
    const timer = window.setInterval(refreshDevStatus, 2000);
    return () => {
      unsubscribe?.();
      window.clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (view !== "session") return;
    refreshSessionDebug();
    const timer = window.setInterval(refreshSessionDebug, 5000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, apiBase, connected]);

  useEffect(() => {
    if (!connected || !sessionStorage.getItem(FULL_RESET_PENDING_KEY)) return;
    sessionStorage.removeItem(FULL_RESET_PENDING_KEY);
    pushLog({ type: "backend.log", data: { raw: "[HEBE][DEV] websocket reconnected", category: "backend" }, ts: Date.now()/1000 });
    console.log("[HEBE][DEV] websocket reconnected");
    sendText("Hebe, actualiza contexto de stream");
    refreshMicDevices();
    fetch(`${apiBase}/debug/memory`).catch(() => undefined);
    pushLog({ type: "backend.log", data: { raw: "[HEBE][DEV] full_reset complete", category: "backend" }, ts: Date.now()/1000 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected]);

  useEffect(() => {
    const s = { volume, speed, lang };
    writeSettings(s);
    if (!connected) return;
    sendCommand("set_tts", { volume, speed });
    sendCommand("set_lang", { lang });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volume, speed, lang]);

  async function refreshMicDevices() {
    setMicError("");
    try {
      const [devicesRes, selectedRes] = await Promise.all([
        fetch(`${apiBase}/audio/input-devices`),
        fetch(`${apiBase}/audio/input-device`),
      ]);
      if (!devicesRes.ok) throw new Error(await devicesRes.text());
      const devicesPayload = await devicesRes.json();
      const selectedPayload = selectedRes.ok ? await selectedRes.json() : {};
      const devices = Array.isArray(devicesPayload?.devices) ? devicesPayload.devices : [];
      setMicDevices(devices);
      setSelectedMicId(String(selectedPayload?.device_id ?? ""));
      setSelectedMicName(String(selectedPayload?.device_name ?? ""));
      setSelectedMicHostApi(String(selectedPayload?.host_api ?? ""));
      if (selectedPayload?.error) setMicError(String(selectedPayload.error));
    } catch (err: any) {
      setMicError(err?.message || "No he podido listar micrófonos.");
    }
  }

  async function selectMic(deviceId: string) {
    const device = micDevices.find((item) => String(item.id) === String(deviceId));
    const deviceName = device?.name ?? "";
    const hostApi = device?.host_api ?? "";
    setSelectedMicId(deviceId);
    setSelectedMicName(deviceName);
    setSelectedMicHostApi(hostApi);
    setMicTestResult(null);
    setMicError("");
    try {
      const res = await fetch(`${apiBase}/audio/input-device`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          device_id: deviceId,
          device_name: deviceName,
          host_api: hostApi,
          sample_rate: device?.default_sample_rate ?? device?.sample_rate ?? null,
          channels: device?.max_input_channels ?? device?.channels ?? null,
          signature: device?.signature ?? "",
        }),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok || payload?.ok === false) {
        throw new Error(payload?.error || payload?.detail || "No he podido cambiar el micrófono.");
      }
      pushLog({ type: "status", data: { message: `Micrófono STT seleccionado: ${deviceName || "default"}` }, ts: Date.now() / 1000 });
    } catch (err: any) {
      setMicError(err?.message || "No he podido cambiar el micrófono.");
    }
  }

  async function testSelectedMic() {
    setMicError("");
    setMicTestResult(null);
    const device = micDevices.find((item) => String(item.id) === String(selectedMicId));
    try {
      const res = await fetch(`${apiBase}/audio/input-device/test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          device_id: selectedMicId,
          device_name: selectedMicName || device?.name || "",
          host_api: selectedMicHostApi || device?.host_api || "",
          seconds: 4,
        }),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok || payload?.ok === false) {
        throw new Error(payload?.detail || payload?.error || "No he podido probar el micro.");
      }
      setMicTestResult(payload);
      setSttRms(Number(payload?.rms ?? 0));
      setSttPeak(Number(payload?.peak ?? 0));
      setSttLevel(Number(payload?.peak ?? 0));
      if (payload?.signal_detected) setLastSttLevelAt(Date.now());
    } catch (err: any) {
      setMicError(err?.message || "No he podido probar el micro.");
    }
  }

  useEffect(() => {
    refreshMicDevices();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase]);

  useEffect(() => {
    const timer = window.setInterval(() => setUiTick(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const [input, setInput] = useState("");
  const startDisabled = backendRunning === true;
  const stopDisabled = backendRunning === false;
  const devControlsEnabled = Boolean(window.hebeDev?.enabled);
  const liveSession = sessionDebug?.live_session || {};
  const streamMetadata = sessionDebug?.stream_metadata || {};
  const streamStatus = String(streamMetadata.stream_status || streamMetadata.live_status || liveSession.stream_status || "unknown");
  const streamTitle = String(streamMetadata.title || liveSession.current_title || "no title");
  const streamGame = String(streamMetadata.game || liveSession.current_game || streamMetadata.category || liveSession.current_category || "unknown");
  const sttTone = sttStatus === "off" || sttStatus === "idle" ? "idle" : sttStatus === "recording" || sttStatus === "listening" ? "warn" : "ok";
  const ttsTone = ttsEnabled === false ? "idle" : ttsState === "speaking" ? "warn" : "ok";
  const hebeTone = hebeSleeping ? "idle" : engineReady ? "ok" : "warn";
  const streamTone = streamStatus === "live" ? "ok" : streamStatus === "offline" ? "idle" : "warn";
  const twitchTone = streamStatus === "live" ? "ok" : connected ? "warn" : "bad";
  const micWarning = sttStatus !== "off" && sttStatus !== "idle" && uiTick - lastSttLevelAt > 10000 && sttRms <= 0.003 && (sttPeak || sttLevel) <= 0.001;

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
            <button className={"tabBtn " + (view === "chat" ? "active" : "")} onClick={() => setView("chat")} role="tab" aria-selected={view === "chat"}>Chat</button>
            <button className={"tabBtn " + (view === "session" ? "active" : "")} onClick={() => setView("session")} role="tab" aria-selected={view === "session"}>Session</button>
            <button className={"tabBtn " + (view === "audio" ? "active" : "")} onClick={() => setView("audio")} role="tab" aria-selected={view === "audio"}>Audio</button>
            {devControlsEnabled && <button className={"tabBtn " + (view === "dev" ? "active" : "")} onClick={() => setView("dev")} role="tab" aria-selected={view === "dev"}>Dev</button>}
            <button className={"tabBtn " + (view === "logs" ? "active" : "")} onClick={() => setView("logs")} role="tab" aria-selected={view === "logs"}>Logs</button>
            <button className={"tabBtn " + (view === "database" ? "active" : "")} onClick={() => setView("database")} role="tab" aria-selected={view === "database"}>BBDD / Memory</button>
          </div>

          <div className="pills statusBar" aria-label="Estado global">
            <StatusChip label="Backend" value={connected ? "WS" : "off"} tone={connected ? "ok" : "bad"} detail={connected ? "WebSocket conectado" : "WebSocket desconectado"} />
            <StatusChip label="Hebe" value={hebeSleeping ? "sleep" : engineReady ? "ready" : "boot"} tone={hebeTone} detail={engineStage || (hebeSleeping ? "Dormida" : "Estado de motor")} />
            <StatusChip label="STT" value={sttStatus || "unknown"} tone={sttTone} detail={sttLive || lastSttFinal || "Entrada de voz"} />
            <StatusChip label="TTS" value={ttsEnabled === false ? "off" : ttsState} tone={ttsTone} detail={ttsEnabled === false ? "TTS desactivado" : "Salida de voz"} />
            <StatusChip label="Twitch" value={streamStatus === "live" ? "live" : connected ? "check" : "off"} tone={twitchTone} detail={streamTitle} />
            <StatusChip label="Stream" value={streamStatus} tone={streamTone} detail={streamGame} />
          </div>
        </header>

        {view === "database" ? (
          <DatabaseInspector apiBase={apiBase} />
        ) : view === "logs" ? (
          <LogsView apiBase={apiBase} logs={logs} onClearVisible={(ids) => setLogs((prev) => prev.filter((item) => !ids.has(item.id)))} />
        ) : view === "session" ? (
          <SessionView data={sessionDebug} loading={sessionLoading} error={sessionError} onRefresh={refreshSessionDebug} onCommand={sendText} />
        ) : view === "audio" ? (
          <AudioView connected={connected} devices={micDevices} selectedId={selectedMicId} selectedName={selectedMicName} selectedHostApi={selectedMicHostApi} rms={sttRms} peak={sttPeak || sttLevel} sttStatus={sttStatus} lastPartial={sttLive} lastFinal={lastSttFinal} testResult={micTestResult} warning={micWarning} error={micError} volume={volume} speed={speed} lang={lang} ttsEnabled={ttsEnabled} ttsState={ttsState} onRefresh={refreshMicDevices} onSelect={selectMic} onTestMic={testSelectedMic} onVolume={setVolume} onSpeed={setSpeed} onLang={setLang} onStopSpeaking={() => sendCommand("stop_speaking")} onCommand={sendText} />
        ) : view === "dev" ? (
          <DevView enabled={devControlsEnabled} status={devStatus} websocketConnected={connected} busy={devBusy} wakeLoopAlive={wakeLoopAlive} wakeLoopError={wakeLoopError} onReloadUi={() => runDevAction("reload")} onRestartBackend={() => runDevAction("restart")} onFullReset={() => runDevAction("full")} onRefresh={refreshDevStatus} />
        ) : (
          <main className="grid chatGrid">
            <section className="glass panel chat chatMainPanel">
              <div className="panelHeader"><div><div className="panelTitle">Conversacion</div><div className="panelMeta">{messages.length} mensajes</div></div></div>
              <div className="chatList" ref={listRef}>{messages.map((m) => (<div key={m.id} className={"bubbleRow " + (m.role === "user" ? "right" : "left")}><div className={"bubble " + (m.role === "user" ? "user" : "assistant") + (m.traceId ? " datasetLinked" : "")}><div className="bubbleTop"><span className="bubbleName">{m.role === "user" ? "Tu" : "Hebe"}</span><span className="bubbleTime">{fmtTime(m.ts)}</span></div>{m.role === "assistant" && m.sourceMessage && <div className="replyContext"><div className="replyContextLabel">Responde a {m.sourceUser || "chat"}</div><div className="replyContextText">{m.sourceMessage}</div></div>}<div className={"bubbleText " + (m.partial ? "partial" : "")}>{m.role === "assistant" && m.partial && !m.text ? <span className="thinkingDots" aria-label="Hebe esta pensando">...</span> : m.text}</div>{m.role === "assistant" && m.traceId && !m.partial && <div className="curationBar"><button className={"curationBtn ok " + (m.curation === "ok" ? "active" : "")} onClick={() => markCuration(m.id, m.traceId!, "ok")} title="Guardar como ejemplo bueno">OK</button><button className={"curationBtn bad " + (m.curation === "no_ok" ? "active" : "")} onClick={() => markCuration(m.id, m.traceId!, "no_ok")} title="Marcar como mal ejemplo">No OK</button><button className={"curationBtn warn " + (m.curation === "needs_enhancement" ? "active" : "")} onClick={() => markCuration(m.id, m.traceId!, "needs_enhancement")} title="La idea sirve, pero necesita mejorar">Mejorar</button></div>}</div></div>))}</div>
              <div className="composer"><input className="input" placeholder="Escribe a Hebe..." value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { sendText(input); setInput(""); } }} /><button className="btn primary" onClick={() => { sendText(input); setInput(""); }}>Enviar</button></div>
              <div className="hint muted">Tip: <span className="mono">Ctrl+Enter</span> para enviar.</div>
            </section>
            <aside className="glass panel controlPanel liveControlPanel"><div className="panelHeader slim"><div><div className="panelTitle">Live control center</div><div className="panelMeta mono">{logs.length} eventos</div></div></div><LiveControlColumn disabled={!connected} startDisabled={startDisabled} stopDisabled={stopDisabled} onCommand={sendText} onControl={sendCommand} onStopSpeaking={() => sendCommand("stop_speaking")} onOpenAudio={() => setView("audio")} ttsEnabled={ttsEnabled} sttStatus={sttStatus} sttLive={sttLive} hebeSleeping={hebeSleeping} selectedMicName={selectedMicName} selectedMicHostApi={selectedMicHostApi} rms={sttRms} peak={sttPeak || sttLevel} micWarning={micWarning} micError={micError} sessionData={sessionDebug} /></aside>
            <aside className="glass panel modelPanel"><div className="panelHeader slim"><div className="panelTitle">Modelo VTuber</div><div className="panelMeta">preview vertical</div></div><VtuberPreview /></aside>
          </main>
        )}
      </div>
    </div>
  );
}

function StatusChip({ label, value, tone, detail }: { label: string; value: string; tone: "ok" | "warn" | "bad" | "idle"; detail?: string }) { return <div className={"pill statusChip " + tone} title={detail || value}><span className="dot" /><span className="statusChipLabel">{label}</span><span className="statusChipValue">{value}</span></div>; }
function displayValue(value: unknown, fallback = "-"): string { if (value === null || value === undefined || value === "") return fallback; if (Array.isArray(value)) return value.length ? value.map((item) => displayValue(item, "")).filter(Boolean).join(" | ") : fallback; if (typeof value === "object") { const data = value as Record<string, unknown>; return displayValue(data.text || data.raw_text || data.summary_text || data.message || data.topic || JSON.stringify(data), fallback); } return String(value); }
function DevView({ enabled, status, websocketConnected, busy, wakeLoopAlive, wakeLoopError, onReloadUi, onRestartBackend, onFullReset, onRefresh }: { enabled: boolean; status: DevBackendStatus; websocketConnected: boolean; busy: "" | "reload" | "restart" | "full"; wakeLoopAlive: boolean | null; wakeLoopError: string; onReloadUi: () => void; onRestartBackend: () => void; onFullReset: () => void; onRefresh: () => void }) { return <main className="tabLayout devLayout"><section className="glass panel devMainPanel"><div className="panelHeader"><div><div className="panelTitle">Dev maintenance</div><div className="panelMeta">Controles separados del flujo normal</div></div><button className="btn compact" onClick={onRefresh}>Refresh</button></div>{enabled ? <DevControlPanel status={status} websocketConnected={websocketConnected} busy={busy} onReloadUi={onReloadUi} onRestartBackend={onRestartBackend} onFullReset={onFullReset} /> : <div className="emptyState">Dev controls disabled in this build.</div>}</section><section className="glass panel devHealthPanel"><div className="panelHeader slim"><div className="panelTitle">Backend health</div><div className="panelMeta">runtime</div></div><div className="statusList"><StatusLine label="Backend running" value={status.running ? "yes" : "no"} tone={status.running ? "ok" : "bad"} /><StatusLine label="PID" value={status.pid ? String(status.pid) : "-"} tone={status.pid ? "ok" : "idle"} /><StatusLine label="Uptime" value={formatDuration(status.uptimeMs || 0)} tone={status.running ? "ok" : "idle"} /><StatusLine label="Last restart" value={formatRestartTime(status.lastRestartTime)} tone={status.lastRestartTime ? "ok" : "idle"} /><StatusLine label="WebSocket" value={websocketConnected ? "yes" : "no"} tone={websocketConnected ? "ok" : "bad"} /><StatusLine label="Wake/STT loop" value={wakeLoopAlive === false ? "crashed" : wakeLoopAlive === true ? "alive" : "unknown"} tone={wakeLoopAlive === false ? "bad" : wakeLoopAlive === true ? "ok" : "warn"} /></div>{(wakeLoopAlive === false && wakeLoopError) && <div className="devError mono">Wake/STT loop crashed: {wakeLoopError}</div>}{(status.lastError || status.error) && <div className="devError mono">{status.lastError || status.error}</div>}</section></main>; }
function SessionView({ data, loading, error, onRefresh, onCommand }: { data: any | null; loading: boolean; error: string; onRefresh: () => void; onCommand: (command: string) => void }) { const meta = data?.stream_metadata || {}; const live = data?.live_session || {}; const rag = data?.memory_rag || {}; const timeline = Array.isArray(rag.recent_timeline_events) ? rag.recent_timeline_events : Array.isArray(data?.recent_events) ? data.recent_events : []; const summaries = Array.isArray(rag.rolling_summaries) ? rag.rolling_summaries : Array.isArray(data?.rolling_summaries) ? data.rolling_summaries : []; const chatters = Array.isArray(live.recent_chatters) ? live.recent_chatters : Array.isArray(live.active_chatters) ? live.active_chatters : []; const lastSummary = summaries[0]?.summary_text || summaries[summaries.length - 1]?.summary_text; const lastHebe = live.last_hebe_utterance || live.last_spontaneous_message || {}; return <main className="sessionLayout"><section className="glass panel sessionPanel wide"><div className="panelHeader"><div><div className="panelTitle">Live session brain</div><div className="panelMeta">{loading ? "refreshing" : data ? "debug snapshot" : "waiting for session"}</div></div><button className="btn compact" onClick={onRefresh}>Refresh</button></div>{error && <div className="micWarn">{error}</div>}<div className="sessionCardGrid"><SessionCard title="Stream metadata"><InfoRow label="Status" value={displayValue(meta.stream_status || meta.live_status || live.stream_status)} /><InfoRow label="Game" value={displayValue(meta.game || live.current_game)} /><InfoRow label="Category" value={displayValue(meta.category || live.current_category)} /><InfoRow label="Title" value={displayValue(meta.title || live.current_title)} /></SessionCard><SessionCard title="Live session state"><InfoRow label="Phase" value={displayValue(live.current_phase)} /><InfoRow label="Objective" value={displayValue(live.current_objective)} /><InfoRow label="Progress" value={displayValue((live.recent_progress_markers || []).slice(-1)[0])} /><InfoRow label="Boss/combat" value={displayValue(live.latest_boss_state || live.latest_strategy_topic)} /><InfoRow label="Correction" value={displayValue(live.latest_correction_from_leo)} /></SessionCard><SessionCard title="Hebe memory/RAG"><InfoRow label="Events" value={displayValue(rag.meaningful_events)} /><InfoRow label="Context updates" value={displayValue(rag.session_context_updates)} /><InfoRow label="Last retrieval" value={displayValue(rag.last_retrieved_context_used?.query)} /><InfoRow label="Last memory update" value={displayValue(live.last_updated_at || rag.latest_rolling_summary_time)} /></SessionCard><SessionCard title="Interaction anchors"><InfoRow label="Chat topic" value={displayValue(live.current_chat_topic)} /><InfoRow label="Last Hebe" value={displayValue(lastHebe.text || lastHebe.raw_text)} /><InfoRow label="Last anchor" value={displayValue(live.last_hebe_anchor || lastHebe.anchor_id)} /><InfoRow label="Last direct" value={displayValue(live.last_direct_interaction_with_leo)} /></SessionCard></div></section><section className="glass panel sessionPanel"><div className="panelHeader slim"><div className="panelTitle">Recent timeline</div><div className="panelMeta">{timeline.length} events</div></div><div className="timelineList">{timeline.slice(0, 12).map((item: any, idx: number) => <div className="timelineItem" key={item.id || idx}><span className="timelineType">{displayValue(item.event_type || item.topic)}</span><span>{displayValue(item.raw_text || item.summary_text || item.output_target)}</span></div>)}{!timeline.length && <div className="emptyState">No timeline yet.</div>}</div></section><section className="glass panel sessionPanel"><div className="panelHeader slim"><div className="panelTitle">Chat participants</div><div className="panelMeta">{chatters.length} recent</div></div><div className="chatterList">{chatters.slice(0, 12).map((item: any, idx: number) => <div className="chatterItem" key={item.username || item.display_name || idx}><span>{displayValue(item.display_name || item.username)}</span><span className="muted">{displayValue(item.last_message || item.recent_topics?.[0])}</span></div>)}{!chatters.length && <div className="emptyState">No chat participants in the current snapshot.</div>}</div></section><section className="glass panel sessionPanel wide"><div className="panelHeader slim"><div className="panelTitle">Rolling summary preview</div><div className="panelMeta">session memory</div></div><div className="summaryPreview">{displayValue(lastSummary, "No rolling summary yet.")}</div><div className="sessionActions"><button className="btn compact" onClick={() => onCommand("Hebe, actualiza contexto de stream")}>Refresh stream context</button><button className="btn compact" onClick={() => onCommand("Hebe, que recuerdas de este directo")}>Ask memory</button></div></section></main>; }
function SessionCard({ title, children }: { title: string; children: ReactNode }) { return <div className="sessionCard"><div className="sessionCardTitle">{title}</div>{children}</div>; }
function InfoRow({ label, value }: { label: string; value: string }) { return <div className="infoRow"><span>{label}</span><strong title={value}>{value}</strong></div>; }
function AudioView({ connected, devices, selectedId, selectedName, selectedHostApi, rms, peak, sttStatus, lastPartial, lastFinal, testResult, warning, error, volume, speed, lang, ttsEnabled, ttsState, onRefresh, onSelect, onTestMic, onVolume, onSpeed, onLang, onStopSpeaking, onCommand }: { connected: boolean; devices: AudioInputDevice[]; selectedId: string; selectedName: string; selectedHostApi: string; rms: number; peak: number; sttStatus: string; lastPartial: string; lastFinal: string; testResult: any; warning: boolean; error: string; volume: number; speed: number; lang: LangMode; ttsEnabled: boolean | null; ttsState: "idle" | "speaking"; onRefresh: () => void; onSelect: (deviceId: string) => void; onTestMic: () => void; onVolume: (value: number) => void; onSpeed: (value: number) => void; onLang: (value: LangMode) => void; onStopSpeaking: () => void; onCommand: (command: string) => void }) { return <main className="audioLayout"><section className="glass panel audioPanel wide"><div className="panelHeader"><div><div className="panelTitle">Audio / STT</div><div className="panelMeta">input device and voice controls</div></div></div><MicSelector devices={devices} selectedId={selectedId} selectedName={selectedName} selectedHostApi={selectedHostApi} rms={rms} peak={peak} sttStatus={sttStatus} lastPartial={lastPartial} lastFinal={lastFinal} testResult={testResult} warning={warning} error={error} disabled={!connected} onRefresh={onRefresh} onSelect={onSelect} onTest={onTestMic} /></section><section className="glass panel audioPanel"><div className="panelHeader slim"><div className="panelTitle">TTS</div><div className="panelMeta">backend config</div></div><div className="field"><div className="fieldTop"><span>Volume</span><span className="mono">{Math.round(volume * 100)}%</span></div><input type="range" min={0} max={1} step={0.01} value={volume} onChange={(e) => onVolume(Number(e.target.value))} /></div><div className="field"><div className="fieldTop"><span>Speed</span><span className="mono">{speed.toFixed(2)}x</span></div><input type="range" min={0.75} max={1.25} step={0.01} value={speed} onChange={(e) => onSpeed(Number(e.target.value))} /></div><div className="field"><div className="fieldTop"><span>Language</span></div><select className="select" value={lang} onChange={(e) => onLang(e.target.value as LangMode)}><option value="auto">Auto</option><option value="es">Espanol</option><option value="en">English</option></select></div><div className="statusList"><StatusLine label="TTS state" value={ttsEnabled === false ? "off" : ttsState} tone={ttsEnabled === false ? "idle" : ttsState === "speaking" ? "warn" : "ok"} /><StatusLine label="Engine" value="backend/default" tone="idle" /><StatusLine label="Output route" value="config" tone="idle" /></div><div className="audioActions"><button className="btn compact" onClick={() => onCommand("Hebe, estado voz")}>Voice status</button><button className="btn compact danger" onClick={onStopSpeaking}>Stop voice</button></div></section></main>; }
function LiveControlColumn({ disabled, startDisabled, stopDisabled, onCommand, onControl, onStopSpeaking, onOpenAudio, ttsEnabled, sttStatus, sttLive, hebeSleeping, selectedMicName, selectedMicHostApi, rms, peak, micWarning, micError, sessionData }: { disabled: boolean; startDisabled: boolean; stopDisabled: boolean; onCommand: (command: string) => void; onControl: (name: string, payload?: Record<string, any>) => void; onStopSpeaking: () => void; onOpenAudio: () => void; ttsEnabled: boolean | null; sttStatus: string; sttLive: string; hebeSleeping: boolean; selectedMicName: string; selectedMicHostApi: string; rms: number; peak: number; micWarning: boolean; micError: string; sessionData: any | null }) { const sttOn = sttStatus !== "off" && sttStatus !== "idle"; const live = sessionData?.live_session || {}; return <div className="controlScroll controlAccordion"><details className="controlSection" open><summary>Stream <span>{displayValue(live.current_game || sessionData?.stream_metadata?.game, "context")}</span></summary><div className="controlActions"><ControlButton label="Prep" onClick={() => onCommand("Hebe, prepara el stream de hoy")} disabled={disabled} primary /><ControlButton label="Title" onClick={() => onCommand("Hebe, sugiere titulo para hoy")} disabled={disabled} /><ControlButton label="Context" onClick={() => onCommand("Hebe, actualiza contexto de stream")} disabled={disabled} primary /><ControlButton label="Status" onClick={() => onCommand("Hebe, que contexto de stream tienes")} disabled={disabled} /><ControlButton label="Start" onClick={() => onControl("start")} disabled={startDisabled} /><ControlButton label="Stop" onClick={() => onControl("stop")} disabled={stopDisabled} danger /></div></details><details className="controlSection" open><summary>Hebe <span>{hebeSleeping ? "sleeping" : "awake"}</span></summary><div className="controlActions"><ControlButton label="Wake" onClick={() => onCommand("Hebe, despierta")} disabled={disabled} primary={hebeSleeping} /><ControlButton label="Sleep" onClick={() => onCommand("Hebe, duerme")} disabled={disabled} /><ControlButton label="Session" onClick={() => onCommand("Hebe, que contexto de partida tienes")} disabled={disabled} /><ControlButton label="Memory" onClick={() => onCommand("Hebe, que recuerdas de este directo")} disabled={disabled} /></div></details><details className="controlSection" open><summary>Voice / STT <span>{sttStatus}</span></summary><CompactMicStatus selectedMicName={selectedMicName} selectedMicHostApi={selectedMicHostApi} rms={rms} peak={peak} sttStatus={sttStatus} sttLive={sttLive} warning={micWarning} error={micError} onOpenAudio={onOpenAudio} /><div className="controlActions"><ControlButton label="STT ON" onClick={() => onCommand("Hebe, activa STT ambiental")} disabled={disabled} primary={!sttOn} /><ControlButton label="STT OFF" onClick={() => onCommand("Hebe, desactiva STT ambiental")} disabled={disabled} primary={sttOn} /><ControlButton label="Voice ON" onClick={() => onCommand("Hebe, activa la voz")} disabled={disabled} primary={ttsEnabled === false} /><ControlButton label="Text only" onClick={() => onCommand("Hebe, solo texto")} disabled={disabled} /><ControlButton label="Stop voice" onClick={onStopSpeaking} disabled={disabled} danger /></div></details><details className="controlSection"><summary>Spontaneity <span>{displayValue(live.last_hebe_anchor, "anchors")}</span></summary><div className="controlActions"><ControlButton label="Status" onClick={() => onCommand("Hebe, estado de espontaneidad")} disabled={disabled} /><ControlButton label="Pause" onClick={() => onCommand("Hebe, pausa espontaneidad")} disabled={disabled} /><ControlButton label="Enable" onClick={() => onCommand("Hebe, activa espontaneidad")} disabled={disabled} primary /><ControlButton label="Companion" onClick={() => onCommand("Hebe, modo companera")} disabled={disabled} /><ControlButton label="Show" onClick={() => onCommand("Hebe, modo show")} disabled={disabled} /></div></details><details className="controlSection"><summary>Twitch <span>{displayValue(live.current_chat_topic, "chat")}</span></summary><div className="controlActions"><ControlButton label="Shoutout" onClick={() => onCommand("Hebe, prueba SO")} disabled={disabled} /><ControlButton label="Raid" onClick={() => onCommand("Hebe, prueba raid")} disabled={disabled} /><ControlButton label="Chat" onClick={() => onCommand("Hebe, que esta pasando en chat")} disabled={disabled} primary /></div></details></div>; }
function ControlButton({ label, onClick, disabled, primary, danger }: { label: string; onClick: () => void; disabled: boolean; primary?: boolean; danger?: boolean }) { return <button className={"quickBtn controlActionBtn " + (primary ? "active " : "") + (danger ? "danger" : "")} onClick={onClick} disabled={disabled}>{label}</button>; }
function CompactMicStatus({ selectedMicName, selectedMicHostApi, rms, peak, sttStatus, sttLive, warning, error, onOpenAudio }: { selectedMicName: string; selectedMicHostApi: string; rms: number; peak: number; sttStatus: string; sttLive: string; warning: boolean; error: string; onOpenAudio: () => void }) { const pct = Math.max(0, Math.min(100, Math.round(Math.max(rms * 500, peak * 100)))); return <div className="compactMic"><div className="compactMicTop"><div><div className="compactMicName" title={selectedMicName || "Default system input"}>{selectedMicName || "Default system input"}</div><div className="compactMicMeta">{selectedMicHostApi || sttStatus}</div></div><button className="miniBtn" onClick={onOpenAudio}>Audio</button></div><div className="meter"><div className="meterFill" style={{ width: String(pct) + "%" }} /></div>{sttLive && <div className="muted small mono">{sttLive}</div>}{warning && <div className="micWarn">No input signal.</div>}{error && <div className="micError">{error}</div>}</div>; }

function DevControlPanel({
  status,
  websocketConnected,
  busy,
  onReloadUi,
  onRestartBackend,
  onFullReset,
}: {
  status: DevBackendStatus;
  websocketConnected: boolean;
  busy: "" | "reload" | "restart" | "full";
  onReloadUi: () => void;
  onRestartBackend: () => void;
  onFullReset: () => void;
}) {
  const state = status.status || "unknown";
  const failed = state === "failed" || Boolean(status.lastError || status.error);
  const tone = failed ? "bad" : state === "healthy" ? "ok" : state === "restarting" || state === "starting" || state === "stopping" ? "warn" : "idle";
  return (
    <div className="card devCard">
      <div className="cardTitle row">
        <span>Dev</span>
        <span className={`devState ${tone}`}>{state}</span>
      </div>
      <div className="devButtons">
        <button className="btn compact" disabled={Boolean(busy)} onClick={onReloadUi}>Reload UI</button>
        <button className="btn compact warning" disabled={Boolean(busy)} onClick={onRestartBackend}>{busy === "restart" ? "Restarting..." : "Restart Backend"}</button>
        <button className="btn compact danger" disabled={Boolean(busy)} onClick={onFullReset}>{busy === "full" ? "Resetting..." : "Full Dev Reset"}</button>
      </div>
      <div className="statusList devStatusList">
        <StatusLine label="Backend" value={status.running ? "yes" : "no"} tone={status.running ? "ok" : "bad"} />
        <StatusLine label="PID" value={status.pid ? String(status.pid) : "-"} tone={status.pid ? "ok" : "idle"} />
        <StatusLine label="Uptime" value={formatDuration(status.uptimeMs || 0)} tone={status.running ? "ok" : "idle"} />
        <StatusLine label="Restart" value={formatRestartTime(status.lastRestartTime)} tone={status.lastRestartTime ? "ok" : "idle"} />
        <StatusLine label="WebSocket" value={websocketConnected ? "yes" : "no"} tone={websocketConnected ? "ok" : "bad"} />
      </div>
      {(status.lastError || status.error) && <div className="devError mono">{status.lastError || status.error}</div>}
    </div>
  );
}

function formatDuration(ms: number) {
  if (!ms) return "0s";
  const seconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  if (minutes <= 0) return `${rest}s`;
  const hours = Math.floor(minutes / 60);
  if (hours <= 0) return `${minutes}m ${rest}s`;
  return `${hours}h ${minutes % 60}m`;
}

function formatRestartTime(value?: string | null) {
  if (!value) return "never";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function QuickControlToolbar({ disabled, onCommand }: { disabled: boolean; onCommand: (command: string) => void }) {
  const [expanded, setExpanded] = useState(false);
  const groups = [
    {
      title: "Stream",
      items: [
        ["Prep", "Preparar hoy", "Hebe, prepara el stream de hoy"],
        ["Title", "T�tulo", "Hebe, sugiere t�tulo para hoy"],
        ["Start", "Guardar inicio", "Hebe, guarda que empezamos por confirmar"],
        ["End", "Guardar final", "Hebe, guarda que terminamos por confirmar"],
        ["View", "Ver sesi�n", "Hebe, prepara el stream de hoy"],
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
        ["🧹", "Limpiar", "Hebe, limpia contexto oído"],
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
  const items = groups.flatMap((group) => group.items.map((item) => ({ group: group.title, icon: item[0], label: item[1], command: item[2] })));
  const visibleItems = expanded ? items : items.slice(0, 12);

  return (
    <div className={"quickToolbar compact " + (expanded ? "expanded" : "")}>
      <div className="quickToolbarTop">
        <span className="quickToolbarTitle">Live controls</span>
        <button className="quickMoreBtn" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "Menos" : "Más"}
        </button>
      </div>
      <div className="quickButtons">
        {visibleItems.map((item) => (
          <button
            className="quickBtn"
            key={item.command}
            disabled={disabled}
            onClick={() => onCommand(item.command)}
            title={`${item.group}: ${item.command}`}
          >
            <span className="quickIcon">{item.icon}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function StatusLine({ label, value, tone }: { label: string; value: string; tone: "ok" | "warn" | "bad" | "idle" }) {
  return (
    <div className="statusLine">
      <span className="statusLabel">{label}</span>
      <span className={"statusValue " + tone}>
        <span className="statusDot" />
        {value}
      </span>
    </div>
  );
}

type QuickItem = {
  icon: string;
  label: string;
  command: string;
  featured?: boolean;
};

function LiveControlToolbar({
  disabled,
  onCommand,
  onStopSpeaking,
  ttsEnabled,
  sttStatus,
  hebeSleeping,
}: {
  disabled: boolean;
  onCommand: (command: string) => void;
  onStopSpeaking: () => void;
  ttsEnabled: boolean | null;
  sttStatus: string;
  hebeSleeping: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const sttOn = sttStatus !== "off" && sttStatus !== "idle";
  const groups: { title: string; items: QuickItem[] }[] = [
    {
      title: "STREAM",
      items: [
        { icon: "Prep", label: "Preparar", command: "Hebe, prepara el stream de hoy", featured: true },
        { icon: "Title", label: "T�tulo", command: "Hebe, sugiere t�tulo para hoy", featured: true },
        { icon: "Start", label: "Inicio", command: "Hebe, guarda que empezamos por confirmar" },
        { icon: "End", label: "Final", command: "Hebe, guarda que terminamos por confirmar" },
        { icon: "View", label: "Sesi�n", command: "Hebe, prepara el stream de hoy" },
        { icon: "🔄", label: "Contexto", command: "Hebe, actualiza contexto de stream", featured: true },
        { icon: "📡", label: "Estado", command: "Hebe, qué contexto de stream tienes", featured: true },
        { icon: "🧠", label: "Partida", command: "Hebe, qué contexto de partida tienes", featured: true },
        { icon: "💾", label: "Snapshot", command: "Hebe, guarda snapshot del stream" },
        { icon: "🏁", label: "Finalizar", command: "Hebe, finaliza stream" },
      ],
    },
    {
      title: "ESPONTANEIDAD",
      items: [
        { icon: "💬", label: "Estado", command: "Hebe, estado de espontaneidad", featured: true },
        { icon: "⏸", label: "Pausar", command: "Hebe, pausa espontaneidad", featured: true },
        { icon: "▶", label: "Activar", command: "Hebe, activa espontaneidad", featured: true },
        { icon: "🤫", label: "Reactiva", command: "Hebe, modo reactiva" },
        { icon: "🧍", label: "Compañera", command: "Hebe, modo compañera" },
        { icon: "🎭", label: "Show", command: "Hebe, modo show" },
        { icon: "🔇", label: "Idle voz OFF", command: "Hebe, espontaneidad en texto" },
        { icon: "🔊", label: "Idle voz ON", command: "Hebe, espontaneidad con voz" },
      ],
    },
    {
      title: "HEBE",
      items: [
        { icon: "☀️", label: "Despertar", command: "Hebe, despierta", featured: hebeSleeping },
        { icon: "🌙", label: "Dormir", command: "Hebe, duerme", featured: !hebeSleeping },
      ],
    },
    {
      title: "VOZ / STT",
      items: [
        { icon: "🎙", label: "STT ON", command: "Hebe, activa STT ambiental", featured: !sttOn },
        { icon: "🎙", label: "STT OFF", command: "Hebe, desactiva STT ambiental", featured: sttOn },
        { icon: "👂", label: "Oído", command: "Hebe, qué ha oído del stream" },
        { icon: "🧹", label: "Limpiar", command: "Hebe, limpia contexto oído" },
        { icon: "🔊", label: "Voz ON", command: "Hebe, activa la voz", featured: ttsEnabled === false },
        { icon: "🔇", label: "Solo texto", command: "Hebe, solo texto", featured: ttsEnabled !== false },
        { icon: "🛑", label: "Stop voz", command: "__stop_speaking__", featured: true },
      ],
    },
    {
      title: "TWITCH",
      items: [
        { icon: "📣", label: "SO", command: "Hebe, prueba SO", featured: true },
        { icon: "🧪", label: "Raid", command: "Hebe, prueba raid", featured: true },
        { icon: "👥", label: "Chat", command: "Hebe, qué está pasando en chat" },
      ],
    },
  ];

  return (
    <div className={"quickToolbar compact grouped " + (expanded ? "expanded" : "")}>
      <div className="quickToolbarTop">
        <span className="quickToolbarTitle">Live control dashboard</span>
        <button className="quickMoreBtn" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "Menos" : "Más"}
        </button>
      </div>
      <div className="quickGroupGrid">
        {groups.map((group) => {
          const items = expanded ? group.items : group.items.filter((item) => item.featured);
          return (
            <div className="quickGroup" key={group.title}>
              <div className="quickGroupTitle">{group.title}</div>
              <div className="quickButtons">
                {items.map((item) => {
                  const active = (item.label === "Voz ON" && ttsEnabled === true)
                    || (item.label === "Solo texto" && ttsEnabled === false)
                    || (item.label === "STT OFF" && sttOn);
                  return (
                    <button
                      className={"quickBtn " + (active ? "active" : "")}
                      key={item.command}
                      disabled={disabled}
                      onClick={() => item.command === "__stop_speaking__" ? onStopSpeaking() : onCommand(item.command)}
                      title={`${group.title}: ${item.command === "__stop_speaking__" ? "stop speaking" : item.command}`}
                    >
                      <span className="quickIcon">{item.icon}</span>
                      <span>{item.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MicSelector({
  devices,
  selectedId,
  selectedName,
  selectedHostApi,
  rms,
  peak,
  sttStatus,
  lastPartial,
  lastFinal,
  testResult,
  warning,
  error,
  disabled,
  onRefresh,
  onSelect,
  onTest,
}: {
  devices: AudioInputDevice[];
  selectedId: string;
  selectedName: string;
  selectedHostApi: string;
  rms: number;
  peak: number;
  sttStatus: string;
  lastPartial: string;
  lastFinal: string;
  testResult: any;
  warning: boolean;
  error: string;
  disabled: boolean;
  onRefresh: () => void;
  onSelect: (deviceId: string) => void;
  onTest: () => void;
}) {
  const pct = Math.max(0, Math.min(100, Math.round(Math.max(rms * 500, peak * 100))));
  const selectedDevice = devices.find((item) => String(item.id) === String(selectedId));
  const defaultDevice = devices.find((item) => item.is_default_input || item.is_default);
  const currentName = selectedDevice?.display_label || selectedName || defaultDevice?.display_label || "Dispositivo por defecto";
  const defaultLabel = defaultDevice?.display_label || "no detectado";
  const selectedIsDefault = selectedDevice && defaultDevice && String(selectedDevice.id) === String(defaultDevice.id);
  return (
    <div className="card micCard">
      <div className="cardTitle row">
        <span>Micrófono STT</span>
        <div className="miniActions">
          <button className="miniBtn" disabled={disabled} onClick={onRefresh}>Actualizar</button>
          <button className="miniBtn" disabled={disabled} onClick={onTest}>Probar micro</button>
        </div>
      </div>
      <select className="select" value={selectedId} disabled={disabled} onChange={(e) => onSelect(e.target.value)}>
        <option value="">Default del sistema</option>
        {devices.map((device) => (
          <option key={device.id} value={device.id}>
            {device.display_label || `${device.name} - ${device.host_api || "API ?"} - id ${device.id}`}{device.is_default_input || device.is_default ? " (default)" : ""}
          </option>
        ))}
      </select>
      <div className="micCurrent" title={currentName}>{currentName}</div>
      <div className="muted small">Default Windows: {defaultLabel}</div>
      {selectedDevice && !selectedIsDefault && <div className="muted small">Seleccionado distinto del default.</div>}
      {selectedHostApi && <div className="muted small">Ruta: {selectedHostApi}</div>}
      {selectedDevice?.host_api_warning && <div className="micWarn">{selectedDevice.host_api_warning}</div>}
      <div className="meter" aria-label="Nivel de entrada STT">
        <div className="meterFill" style={{ width: `${pct}%` }} />
      </div>
      <div className="micMeta">
        <span>STT: {sttStatus}</span>
        <span>rms {rms.toFixed(4)} / peak {peak.toFixed(4)}</span>
      </div>
      {lastPartial && <div className="muted small">Parcial: <span className="mono">{lastPartial}</span></div>}
      {lastFinal && <div className="muted small">Último final: {lastFinal}</div>}
      {testResult && (
        <div className={testResult.signal_detected ? "micOk" : "micWarn"}>
          {testResult.signal_detected ? "Micro OK: entra señal." : "No entra señal en este dispositivo. Prueba otro Yeti GX / host API."}
          <div className="mono">RMS {Number(testResult.rms || 0).toFixed(5)} / Peak {Number(testResult.peak || 0).toFixed(5)}</div>
          <div className="mono">{testResult.sample_rate}Hz / {testResult.channels}ch</div>
        </div>
      )}
      {warning && <div className="micWarn">No entra señal del micrófono.</div>}
      {error && <div className="micError">{error}</div>}
      {!devices.length && !error && <div className="muted small">No hay micrófonos detectados.</div>}
    </div>
  );
}

function VoiceCommandDebugPanel({ debug }: { debug: VoiceCommandDebug | null }) {
  if (!debug || (!debug.raw_text && !debug.normalized_text)) return null;
  const status = debug.status || "unknown";
  const statusClass = status === "accepted" ? "ok" : status === "needs_confirmation" ? "warn" : "bad";
  return (
    <div className="voiceCommandDebug">
      <div className="voiceCommandDebugTop">
        <span>Comando de voz</span>
        <span className={`voiceCommandBadge ${statusClass}`}>{status}</span>
      </div>
      <div className="voiceCommandGrid">
        <span>Raw</span><code>{debug.raw_text || "-"}</code>
        <span>Script</span><code>{debug.detected_script || debug.script || "-"}</code>
        <span>Retry</span><code>{typeof debug.retry_attempted === "boolean" ? (debug.retry_attempted ? "yes" : "no") : "-"}</code>
        <span>Retry raw</span><code>{debug.retry_transcript || "-"}</code>
        <span>Final</span><code>{debug.final_decision || status}</code>
        <span>Normalizado</span><code>{debug.normalized_text || "-"}</code>
        <span>Intent</span><code>{debug.intent || "-"}</code>
        <span>Target</span><code>{debug.target || "-"}</code>
        <span>Confianza</span><code>{typeof debug.confidence === "number" ? debug.confidence.toFixed(2) : "-"}</code>
        <span>Razón</span><code>{debug.reason || "-"}</code>
      </div>
    </div>
  );
}

function LogsView({ apiBase, logs, onClearVisible }: { apiBase: string; logs: { id: string; ev: HebeEvent }[]; onClearVisible: (ids: Set<string>) => void }) {
  const [backendBuffer, setBackendBuffer] = useState<{ id: string; ev: HebeEvent }[]>([]);
  const [filter, setFilter] = useState<LogFilter>("all");
  const [search, setSearch] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [wrap, setWrap] = useState(true);
  const [hiddenKeys, setHiddenKeys] = useState<Set<string>>(() => new Set());
  const listRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadBackendLogs() {
      try {
        const res = await fetch(`${apiBase}/debug/logs?limit=1500`);
        const payload = await res.json();
        if (cancelled) return;
        const entries = Array.isArray(payload?.logs) ? payload.logs : [];
        setBackendBuffer(entries.map((entry: any, idx: number) => ({
          id: `backend-buffer-${entry.ts || idx}-${idx}`,
          ev: { type: "backend.log", data: entry, ts: Number(entry.ts || Date.now() / 1000) },
        })));
      } catch {
        if (!cancelled) setBackendBuffer([]);
      }
    }
    loadBackendLogs();
    const timer = window.setInterval(loadBackendLogs, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [apiBase]);

  const mergedLogs = useMemo(() => {
    const seen = new Set<string>();
    return [...backendBuffer, ...logs].filter((item) => {
      const key = logEntryKey(item);
      if (seen.has(key)) return false;
      if (hiddenKeys.has(key)) return false;
      seen.add(key);
      return true;
    }).sort((a, b) => a.ev.ts - b.ev.ts);
  }, [backendBuffer, logs, hiddenKeys]);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return mergedLogs.filter(({ ev }) => {
      const type = ev.type || "";
      const category = String(ev.data?.category || "");
      const level = String(ev.data?.level || "");
      const raw = String(ev.data?.raw || ev.data?.message || safeString(ev.data));
      const text = `${type} ${category} ${level} ${raw}`.toLowerCase();
      const matchesSearch = !needle || text.includes(needle);
      const matchesFilter =
        filter === "all" ||
        type === filter ||
        category === filter ||
        (filter === "twitch" && (type.includes("twitch") || category === "twitch")) ||
        (filter === "stream_context" && (type.includes("stream") || category === "stream_context" || text.includes("stream_context"))) ||
        (filter === "stt" && (type.startsWith("stt") || category === "stt")) ||
        (filter === "tts" && (type.startsWith("tts") || category === "tts")) ||
        (filter === "memory" && (category === "memory" || text.includes("memory") || text.includes("rag"))) ||
        (filter === "routing" && (text.includes("routing") || text.includes("output_target") || category === "routing")) ||
        (filter === "dev" && (category === "dev" || text.includes("[hebe][dev]") || text.includes("dev"))) ||
        (filter === "spontaneity" && (category === "spontaneity" || text.includes("spontaneity") || text.includes("espontaneidad"))) ||
        (filter === "db" && (category === "db" || text.includes("[hebe][db") || text.includes("database"))) ||
        (filter === "errors" && (type === "error" || level === "error" || category === "errors" || text.includes("error") || text.includes("failed")));
      return matchesSearch && matchesFilter;
    });
  }, [mergedLogs, filter, search]);

  useEffect(() => {
    if (autoScroll && listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [filtered.length, autoScroll]);

  async function copyLogs(entries = filtered) {
    const text = entries.map(({ ev }) => `${fmtTime(ev.ts)} ${formatLogKind(ev)} ${formatLogMessage(ev)}`).join("\n");
    await navigator.clipboard.writeText(text);
  }

  const filters: LogFilter[] = ["all", "chat.assistant", "chat.user", "twitch", "stream_context", "stt", "tts", "memory", "routing", "dev", "spontaneity", "db", "errors"];

  return (
    <main className="logsLayout">
      <section className="glass panel logsPanel">
        <div className="panelHeader">
          <div>
            <div className="panelTitle">Logs</div>
            <div className="panelMeta">{filtered.length} visibles / {mergedLogs.length} eventos</div>
          </div>
          <div className="logsActions">
            <input className="input logsSearch" placeholder="Buscar logs..." value={search} onChange={(e) => setSearch(e.target.value)} />
            <select className="select logsFilter" value={filter} onChange={(e) => setFilter(e.target.value as LogFilter)}>
              {filters.map((item) => <option value={item} key={item}>{item}</option>)}
            </select>
            <label className="toggle mini"><input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} /><span className="toggleLabel">Auto</span></label>
            <label className="toggle mini"><input type="checkbox" checked={wrap} onChange={(e) => setWrap(e.target.checked)} /><span className="toggleLabel">Wrap</span></label>
            <button className="btn compact" onClick={() => copyLogs()}>Copiar visibles</button>
            <button className="btn compact" onClick={() => copyLogs(filtered.slice(-200))}>Copiar últimos 200</button>
            <button
              className="btn compact danger"
              onClick={() => {
                setHiddenKeys((prev) => new Set([...prev, ...filtered.map(logEntryKey)]));
                onClearVisible(new Set(filtered.map((item) => item.id)));
              }}
            >
              Limpiar visibles
            </button>
          </div>
        </div>
        <div className={"logsFullBox " + (wrap ? "wrap" : "nowrap")} ref={listRef}>
          {filtered.map(({ id, ev }) => (
            <div className={"logFullLine " + (logBadgeClass(ev) === "bad" ? "error" : "")} key={id}>
              <span className="mono muted">{fmtTime(ev.ts)}</span>
              <span className={"badge " + logBadgeClass(ev)}>{formatLogKind(ev)}</span>
              <span className="mono logMsg">{formatLogMessage(ev)}</span>
            </div>
          ))}
          {filtered.length === 0 && <div className="emptyState">No hay logs visibles con este filtro.</div>}
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

function formatLogKind(ev: HebeEvent) {
  const category = String(ev.data?.category || "").trim();
  const source = String(ev.data?.source || "").trim();
  if (ev.type === "backend.log") return category ? `backend:${category}` : `backend:${source || "log"}`;
  return ev.type;
}

function logEntryKey(item: { id: string; ev: HebeEvent }) {
  return `${item.ev.type}:${item.ev.ts}:${formatLogMessage(item.ev)}`;
}

function formatLogMessage(ev: HebeEvent) {
  if (ev.type === "backend.log") {
    return String(ev.data?.raw || ev.data?.message || "");
  }
  return safeString(ev.data);
}

function logBadgeClass(ev: HebeEvent) {
  const level = String(ev.data?.level || "");
  const category = String(ev.data?.category || "");
  if (ev.type === "error" || level === "error" || category === "errors") return "bad";
  if (ev.type.startsWith("tts") || category === "tts" || category === "stream_context") return "warn";
  if (category === "twitch" || category === "db") return "ok";
  return "";
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
