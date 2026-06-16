import asyncio
import time
import inspect
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .ws import WSManager
from .events import Event, ClientMsg
from .hebe_adapter import HebeAdapter
from .api.debug import router as debug_router
from .api.audio import router as audio_router
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

@app.get("/health")
def health():
    return {"ok": True, "ts": time.time()}


@app.on_event("shutdown")
async def shutdown():
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
        payload = ev.model_dump()
        if payload.get("type") == "status":
            last_status = payload
        await ws_manager.broadcast(payload)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        await ws.send_json({"type": "status", "data": {"connected": True}, "ts": time.time()})
        for entry in get_recent_logs(limit=1000):
            await ws.send_json({"type": "backend.log", "data": entry, "ts": float(entry.get("ts") or time.time())})
        
        if not hebe.running:
            asyncio.create_task(hebe.start())

        if last_status:
            await ws.send_json(last_status)

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
            await ws.send_json({"type": "error", "data": {"message": str(e)}, "ts": time.time()})
        except Exception:
            pass
    finally:
        await ws_manager.disconnect(ws)
