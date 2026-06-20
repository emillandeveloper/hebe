import asyncio
import time
import inspect
import logging
import os
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .ws import WSManager
from .events import Event, ClientMsg
from .hebe_adapter import HebeAdapter
from .api.debug import router as debug_router
from .api.audio import router as audio_router
from .api.capabilities import router as capabilities_router
from .core.log_bus import get_recent_logs, install_log_capture

from dotenv import load_dotenv
load_dotenv()

last_status: dict | None = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en local/Electron es lo más cómodo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(debug_router)
app.include_router(audio_router)
app.include_router(capabilities_router)

ws_manager = WSManager()
event_q: asyncio.Queue[Event] = asyncio.Queue()
hebe = HebeAdapter(event_q)
app.state.adapter = hebe


def _get_stream_stats():
    engine = getattr(hebe, "_engine", None)
    synth = getattr(engine, "response_synthesizer", None)
    return getattr(synth, "_stream_stats", None)


def _try_dump_stream_summary(reason: str = "") -> None:
    try:
        stats = _get_stream_stats()
        if stats and stats.total > 0:
            tag = f" reason={reason!r}" if reason else ""
            print(f"[HEBE][STREAM_SUMMARY]{tag} (triggered by shutdown/disconnect)", flush=True)
            stats.log_summary()
    except Exception:
        pass

async def maybe_await(x):
    # Si HebeAdapter es async -> await
    # Si es sync y devuelve dict/None -> devuelve tal cual
    if inspect.isawaitable(x):
        return await x
    return x


def _prepare_ws_payload(payload: dict) -> dict:
    event = dict(payload or {})
    data = event.get("data")
    if isinstance(data, dict):
        data = dict(data)
    else:
        data = {} if data is None else data

    event_id = str(event.get("event_id") or (data.get("event_id") if isinstance(data, dict) else "") or "").strip()
    if not event_id:
        event_id = f"evt_{uuid.uuid4().hex}"
    event["event_id"] = event_id
    if isinstance(data, dict):
        data.setdefault("event_id", event_id)
        event["data"] = data

    message_event = _chat_message_event(event, data, event_id)
    if message_event is not None:
        message = message_event["message"]
        print(
            "[HEBE][UI_EVENT] "
            f"broadcast type=chat_message event_id={message_event['event_id']} "
            f"message_id={message['message_id']} role={message['role']} text={message['text']!r}",
            flush=True,
        )
        return message_event
    return event


def _chat_message_event(event: dict, data: dict, event_id: str) -> dict | None:
    event_type = str(event.get("type") or "")
    if event_type not in {"chat.user", "chat.assistant", "llm.final"}:
        return None

    if event_type == "chat.user":
        role = "user"
        speaker = str(data.get("speaker") or "Leo")
        source = str(data.get("source") or "ui")
        text = str(data.get("text") or "").strip()
    else:
        role = "assistant"
        speaker = str(data.get("speaker") or "Hebe")
        source = str(data.get("source") or ("llm" if event_type == "llm.final" else "system"))
        text = str(data.get("text") or "").strip()

    if not text:
        return None

    message_id = str(data.get("message_id") or data.get("id") or "").strip()
    if not message_id:
        message_id = f"msg_{event_id}"
    ts = float(event.get("ts") or time.time())
    created_at = str(
        data.get("created_at")
        or datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    )
    message = {
        "message_id": message_id,
        "role": role,
        "speaker": speaker,
        "text": text,
        "source": source,
        "created_at": created_at,
        "output_target": str(data.get("output_target") or data.get("target") or "local_ui"),
        "metadata": {
            key: value
            for key, value in data.items()
            if key not in {"message_id", "id", "role", "speaker", "text", "source", "created_at", "output_target", "target"}
        },
    }
    return {
        "type": "chat_message",
        "event_id": event_id,
        "message": message,
        "data": {
            **data,
            "event_id": event_id,
            "message_id": message_id,
            "legacy_type": event_type,
            "message": message,
        },
        "ts": ts,
    }

@app.get("/health")
def health():
    engine = getattr(hebe, "_engine", None)
    wake_loop = {}
    if engine is not None:
        try:
            wake_loop = engine.wake_loop_health()
        except Exception as exc:
            wake_loop = {"alive": False, "last_error": str(exc), "thread_alive": False}
    ok = not wake_loop or bool(wake_loop.get("alive") or not getattr(engine, "use_wakeword", False))
    return {"ok": ok, "ts": time.time(), "wake_loop": wake_loop}


@app.post("/dev/shutdown")
async def dev_shutdown():
    """Best-effort cleanup before Electron restarts the owned backend process."""
    if not _dev_controls_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    print("[HEBE][DEV] shutdown requested", flush=True)
    try:
        await hebe.stop()
    except Exception as exc:
        print(f"[HEBE][DEV] hebe stop failed: {type(exc).__name__}: {exc}", flush=True)
    try:
        await ws_manager.broadcast(
            {"type": "status", "data": {"backend": "stopping", "running": False}, "ts": time.time()}
        )
    except Exception:
        pass
    return {"ok": True, "ts": time.time()}


def _dev_controls_enabled() -> bool:
    return (
        os.getenv("ELECTRON_DEV", "0").strip() == "1"
        or os.getenv("HEBE_DEV_CONTROLS", "0").strip() == "1"
        or os.getenv("HEBE_DEV_SHUTDOWN_ENABLED", "0").strip() == "1"
    )


def _require_dev_engine():
    if not _dev_controls_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    engine = getattr(hebe, "_engine", None)
    if engine is None or not hebe.running:
        raise HTTPException(status_code=503, detail="engine not running")
    return engine


@app.post("/dev/simulate/twitch-message")
async def dev_simulate_twitch_message(body: dict):
    engine = _require_dev_engine()
    viewer = str((body or {}).get("viewer_name") or (body or {}).get("user_login") or (body or {}).get("username") or "viewer").strip()
    text = str((body or {}).get("text") or (body or {}).get("message_text") or "").strip()
    print(f"[HEBE][SIM] twitch_message viewer={viewer} text={text!r}", flush=True)
    if not text:
        raise HTTPException(status_code=400, detail="missing text")
    return engine.simulate_twitch_message(body or {})


@app.post("/dev/simulate/leo-message")
async def dev_simulate_leo_message(body: dict):
    engine = _require_dev_engine()
    source = str((body or {}).get("source") or "ui").strip()
    text = str((body or {}).get("text") or "").strip()
    print(f"[HEBE][SIM] leo_message source={source} text={text!r}", flush=True)
    if not text:
        raise HTTPException(status_code=400, detail="missing text")
    return engine.simulate_leo_message(
        text,
        source=source,
        pending_kind=str((body or {}).get("pending_kind") or "").strip() or None,
    )


@app.post("/dev/simulate/ambient-stt")
async def dev_simulate_ambient_stt(body: dict):
    engine = _require_dev_engine()
    text = str((body or {}).get("text") or "").strip()
    print(f"[HEBE][SIM] ambient_stt text={text!r}", flush=True)
    if not text:
        raise HTTPException(status_code=400, detail="missing text")
    return engine.simulate_ambient_stt(text)


@app.post("/dev/policy/behavior-blocks/clear")
async def dev_clear_behavior_blocks():
    engine = _require_dev_engine()
    blocks = engine.clear_active_behavior_blocks()
    return {
        "ok": True,
        "behavior_blocks": blocks,
        "last_policy_decision": engine.get_last_policy_trace(),
    }


@app.post("/dev/stream-output-mode")
async def dev_set_stream_output_mode(body: dict):
    engine = _require_dev_engine()
    mode = str((body or {}).get("mode") or "").strip()
    reason = str((body or {}).get("reason") or "heavy_game_or_user_setting").strip()
    try:
        return {"ok": True, **engine.set_stream_output_mode(mode, reason=reason)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.api_route("/dev/test-ui-message", methods=["GET", "POST"])
async def dev_test_ui_message():
    if not _dev_controls_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    event_id = f"evt_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    text = "Hebe UI test message"
    await event_q.put(
        Event(
            type="chat.assistant",
            event_id=event_id,
            ts=time.time(),
            data={
                "event_id": event_id,
                "message_id": message_id,
                "text": text,
                "speaker": "Hebe",
                "source": "system",
                "output_target": "local_ui",
                "metadata": {"dev_test": True},
            },
        )
    )
    return {"ok": True, "event_id": event_id, "message_id": message_id, "text": text}


@app.on_event("shutdown")
async def shutdown():
    try:
        await hebe.stop()
    except Exception:
        pass
    _try_dump_stream_summary("sigterm")


@app.get("/debug/stream-summary")
def debug_stream_summary():
    """Vuelca y devuelve las métricas acumuladas del stream actual."""
    stats = _get_stream_stats()
    if stats is None:
        return {"ok": False, "reason": "engine not running or no stats"}
    stats.log_summary()
    return {
        "ok": True,
        "total": stats.total,
        "retried": stats.retried,
        "salvaged": stats.salvaged,
        "published_with_helper": stats.published_with_helper,
        "patterns": dict(stats.pattern_count.most_common()),
        "top_chatters": dict(stats.by_chatter.most_common(8)),
    }


@app.get("/debug/memory")
def debug_memory():
    """Inspect persistent memory and last private chat turns."""
    from app.services import db_sqlite
    from app.cognitive.memory.memory_store import count_chunks, get_recent_chunks
    from app.stream.live_session import latest_live_session_debug

    last_turns = db_sqlite.get_recent_chat_log(source="ui", limit=10)
    last_input = ""
    for turn in last_turns:
        if turn.get("role") == "user":
            last_input = str(turn.get("text") or "")
            break

    retrieval = {"facts": [], "chunks": []}
    if last_input:
        retrieval["facts"] = db_sqlite.search_memory_facts(
            query_text=last_input,
            active_only=True,
            limit=5,
        )
        try:
            from app.cognitive.memory.memory_store import search_chunks

            retrieval["chunks"] = search_chunks(
                query=last_input,
                top_k=5,
                min_similarity=0.3,
            )
        except Exception as exc:
            retrieval["chunks_error"] = repr(exc)

    engine = getattr(hebe, "_engine", None)
    runtime = getattr(engine, "runtime", None)
    state = getattr(runtime, "state", None)
    stream = getattr(state, "stream", None)
    policies = getattr(stream, "policies", None) if stream is not None else None
    live_session = None
    if engine is not None:
        try:
            live_session = engine._live_session_debug_snapshot()
        except Exception:
            live_session = None
    live_session = live_session or latest_live_session_debug()

    return {
        "db_path": db_sqlite.DB_PATH,
        "tts_enabled": bool(getattr(state, "tts_enabled", False)),
        "stream_tts_enabled": bool(getattr(policies, "allow_tts_replies", False)),
        "stream_output_mode": str(getattr(stream, "stream_output_mode", "tts_enabled") if stream is not None else "tts_enabled"),
        "stt_enabled": bool(getattr(runtime, "stt_enabled", False)),
        "facts_count": db_sqlite.count_memory_facts(active_only=True),
        "chunks_count": count_chunks(active_only=True),
        "last_facts": db_sqlite.get_recent_memory_facts(limit=10, active_only=True),
        "last_chunks": get_recent_chunks(limit=10, active_only=True),
        "last_chat_turns": last_turns,
        "retrieval_for_last_input": retrieval,
        "live_session": live_session,
    }


@app.on_event("startup")
async def startup():
    # NO arrancar el motor aquí (XTTS/Whisper pueden tardar bastante)
    loop = asyncio.get_running_loop()

    def _broadcast_backend_log(entry: dict) -> None:
        try:
            loop.call_soon_threadsafe(
                event_q.put_nowait,
                Event(type="backend.log", data=entry, ts=float(entry.get("ts") or time.time())),
            )
        except Exception:
            pass

    install_log_capture(_broadcast_backend_log)
    print("[HEBE][LOG_BUS] capture installed", flush=True)
    asyncio.create_task(event_pump())
    await event_q.put(Event(type="status", data={"backend": "up", "running": False}, ts=time.time()))


async def event_pump():
    global last_status
    while True:
        ev = await event_q.get()
        payload = _prepare_ws_payload(ev.model_dump())
        if payload.get("type") == "status":
            last_status = payload
        await ws_manager.broadcast(payload)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        await ws.send_json(_prepare_ws_payload({"type": "status", "data": {"connected": True}, "ts": time.time()}))
        for entry in get_recent_logs(limit=1000):
            await ws.send_json(_prepare_ws_payload({"type": "backend.log", "data": entry, "ts": float(entry.get("ts") or time.time())}))
        
        if not hebe.running:
            asyncio.create_task(hebe.start())

        if last_status:
            await ws.send_json(_prepare_ws_payload(last_status))

        while True:
            msg = await ws.receive_json()
            cm = ClientMsg(**msg)

            if cm.type == "client.message":
                text = (cm.data.get("text") or "").strip()
                if text:
                    await maybe_await(hebe.send_text(text))

            elif cm.type == "client.command":
                name = cm.data.get("name")
                payload = cm.data.get("payload", {}) or {}

                if name == "start":
                    await maybe_await(hebe.start())
                elif name == "stop":
                    await maybe_await(hebe.stop())
                else:
                    await maybe_await(hebe.command(name, payload))

    except WebSocketDisconnect:
        # normal cuando cierras la ventana o recargas
        _try_dump_stream_summary("ws_disconnect")
    except Exception as e:
        logging.exception("WS error: %s", e)
        try:
            await ws.send_json(_prepare_ws_payload({"type": "error", "data": {"message": str(e)}, "ts": time.time()}))
        except Exception:
            pass
    finally:
        await ws_manager.disconnect(ws)
