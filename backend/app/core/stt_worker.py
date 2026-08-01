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
                        metadata = dict(getattr(self.stt, "last_result_metadata", {}) or {})
                        if not self._language_metadata_allows_submission(metadata):
                            print(
                                "[HEBE][STT_REJECTED] "
                                "reason=unsupported_language_recovery_failed "
                                f"initial_language={metadata.get('detected_language') or ''}",
                                flush=True,
                            )
                            continue
                        print(f"[STT_WORKER] voice -> {text!r}", flush=True)
                        submit_text_from_voice(text, metadata)
                except STTDeviceOpenFailure as e:
                    print(f"[STT_WORKER] paused: {e}", flush=True)
                    break
                except Exception as e:
                    print(f"[STT_WORKER] error: {e}", flush=True)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    @staticmethod
    def _language_metadata_allows_submission(metadata: dict | None) -> bool:
        data = dict(metadata or {})
        language = str(data.get("detected_language") or "").lower()
        if not language:
            return True
        if language in {"es", "en"} and data.get("language_allowed", True):
            return True
        recovery = dict(data.get("language_recovery") or {})
        return bool(
            recovery.get("accepted")
            and str(recovery.get("selected_language") or "").lower() in {"es", "en"}
        )

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
