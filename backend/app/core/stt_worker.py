# backend/app/core/stt_worker.py
from __future__ import annotations

import threading
from typing import Optional

from app.core.input_bus import submit_text_from_voice
from app.services.stt_whisper import STTDeviceOpenFailure


class STTWorker:
    def __init__(self, stt, stop_event: threading.Event):
        self.stt = stt
        self.stop_event = stop_event
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True

        def run():
            while not self.stop_event.is_set():
                try:
                    text = self.stt.listen()
                    if text:
                        print(f"[STT_WORKER] voice -> {text!r}", flush=True)
                        submit_text_from_voice(text)
                except STTDeviceOpenFailure as e:
                    print(f"[STT_WORKER] paused: {e}", flush=True)
                    break
                except Exception as e:
                    print(f"[STT_WORKER] error: {e}", flush=True)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
