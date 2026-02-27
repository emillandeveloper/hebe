import asyncio
import time

from .events import Event
from .hebe_engine import HebeEngine, set_emitter


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
            # No reventamos el motor por un fallo de UI
            pass


class HebeAdapter:
    def __init__(self, event_q: "asyncio.Queue[Event]"):
        self.event_q = event_q
        self._engine: HebeEngine | None = None
        self._emitter: _AsyncEmitter | None = None
        self.running = False

    async def start(self):
        if self.running:
            # Ya está arrancada
            return

        loop = asyncio.get_running_loop()
        self._emitter = _AsyncEmitter(loop, self.event_q)

        # Inyectamos emisor (para speak/listen/llm → UI)
        set_emitter(self._emitter)

        # Arrancamos el motor en hilo (no bloquea uvicorn)
        self._engine = HebeEngine(use_wakeword=True, say_hello=True)
        self._engine.start()

        self.running = True
        await self.event_q.put(Event(type="status", data={"running": True}, ts=time.time()))

    async def stop(self):
        if self._engine:
            self._engine.stop()
        self.running = False
        await self.event_q.put(Event(type="status", data={"running": False}, ts=time.time()))

    async def send_text(self, text: str):
        # Mensaje escrito desde la UI
        if not self.running:
            await self.start()
        if self._engine:
            self._engine.submit_text(text)

    async def command(self, name: str, payload: dict):
        # De momento, solo informamos a la UI (puedes mapear comandos reales luego)
        await self.event_q.put(Event(type="status", data={"command": name, "payload": payload}, ts=time.time()))
