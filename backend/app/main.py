import asyncio
import time
import inspect
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .ws import WSManager
from .events import Event, ClientMsg
from .hebe_adapter import HebeAdapter

last_status: dict | None = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en local/Electron es lo más cómodo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ws_manager = WSManager()
event_q: asyncio.Queue[Event] = asyncio.Queue()
hebe = HebeAdapter(event_q)

async def maybe_await(x):
    # Si HebeAdapter es async -> await
    # Si es sync y devuelve dict/None -> devuelve tal cual
    if inspect.isawaitable(x):
        return await x
    return x

@app.get("/health")
def health():
    return {"ok": True, "ts": time.time()}

@app.on_event("startup")
async def startup():
    # NO arrancar el motor aquí (XTTS/Whisper pueden tardar bastante)
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
        pass
    except Exception as e:
        logging.exception("WS error: %s", e)
        try:
            await ws.send_json({"type": "error", "data": {"message": str(e)}, "ts": time.time()})
        except Exception:
            pass
    finally:
        await ws_manager.disconnect(ws)
