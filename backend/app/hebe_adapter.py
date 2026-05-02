import asyncio
import time

from .events import Event
from .hebe_engine import HebeEngine
from .core.ui_bridge import set_emitter
from .core.runtime import build_runtime
from .cognitive.persona.stream_dataset_logger import StreamDatasetLogger


class _AsyncEmitter:
    """Convierte callbacks desde hilos (STT/TTS) en eventos AsyncIO para el WebSocket."""

    def __init__(self, loop: asyncio.AbstractEventLoop, q: "asyncio.Queue[Event]"):
        self.loop = loop
        self.q = q

    def __call__(self, event_type: str, data: dict):
        ev = Event(type=event_type, data=data, ts=time.time())
        try:
            self.loop.call_soon_threadsafe(self.q.put_nowait, ev)
        except Exception:
            pass


class HebeAdapter:
    def __init__(self, event_q: "asyncio.Queue[Event]"):
        self.event_q = event_q
        self._engine: HebeEngine | None = None
        self._emitter: _AsyncEmitter | None = None
        self.running = False
        self._dataset_logger = StreamDatasetLogger()

    async def start(self):
        if self.running:
            return

        loop = asyncio.get_running_loop()
        self._emitter = _AsyncEmitter(loop, self.event_q)

        set_emitter(self._emitter)

        self._engine = HebeEngine(
            runtime=build_runtime(),
            use_wakeword=True,
            say_hello=True,
        )
        self._engine.start()

        self.running = True
        await self.event_q.put(Event(type="status", data={"running": True}, ts=time.time()))

    async def stop(self):
        if self._engine:
            self._engine.stop()
        self.running = False
        await self.event_q.put(Event(type="status", data={"running": False}, ts=time.time()))

    async def send_text(self, text: str):
        if not self.running:
            await self.start()
        if self._engine:
            self._engine.submit_text(text)

    async def command(self, name: str, payload: dict):
        payload = payload or {}

        if name == "dataset_curate":
            trace_id = str(payload.get("trace_id") or "").strip()
            status = str(payload.get("status") or "").strip()
            corrected_response = payload.get("corrected_response")
            notes = payload.get("notes")
            tags = payload.get("tags")
            if not isinstance(tags, list):
                tags = []

            ok = self._dataset_logger.update_curation(
                trace_id=trace_id,
                status=status,
                corrected_response=corrected_response,
                notes=notes,
                tags=[str(t) for t in tags if t],
            )

            await self.event_q.put(
                Event(
                    type="dataset.curation.updated" if ok else "dataset.curation.error",
                    data={
                        "trace_id": trace_id,
                        "status": status,
                        "ok": ok,
                    },
                    ts=time.time(),
                )
            )
            return

        await self.event_q.put(Event(type="status", data={"command": name, "payload": payload}, ts=time.time()))
