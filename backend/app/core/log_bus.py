from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import Callable, TextIO

from app.core.persistent_logs import record_console_log


MAX_LOG_LINES = 5000
_buffer: deque[dict] = deque(maxlen=MAX_LOG_LINES)
_callback: Callable[[dict], None] | None = None
_installed = False
_lock = threading.RLock()


def classify_log_line(line: str, source: str = "stdout") -> tuple[str, str]:
    text = str(line or "")
    lower = text.lower()
    level = "error" if source == "stderr" or any(token in lower for token in ("error", "exception", "traceback", "failed", "fatal")) else "info"
    if "[hebe][twitch][chatbot]" in lower:
        category = "twitch"
    elif "[hebe][twitch]" in lower:
        category = "twitch"
    elif "[hebe][stream_context]" in lower or "stream_context" in lower:
        category = "stream_context"
    elif "[hebe][scheduler]" in lower:
        category = "scheduler"
    elif "[hebe][stt]" in lower or "stt." in lower:
        category = "stt"
    elif "[hebe][tts]" in lower or "tts." in lower:
        category = "tts"
    elif "[hebe][db]" in lower or "[hebe][db_inspector]" in lower:
        category = "db"
    elif "uvicorn" in lower:
        category = "status"
    elif "[hebe]" in lower:
        category = "status"
    else:
        category = "backend"
    if level == "error":
        category = "errors" if category == "backend" else category
    return level, category


def get_recent_logs(limit: int = 1000) -> list[dict]:
    with _lock:
        return list(_buffer)[-max(0, min(int(limit), MAX_LOG_LINES)) :]


def install_log_capture(callback: Callable[[dict], None] | None = None) -> None:
    global _callback, _installed
    with _lock:
        _callback = callback
        if _installed:
            return
        sys.stdout = _TeeStream(sys.stdout, "stdout")  # type: ignore[assignment]
        sys.stderr = _TeeStream(sys.stderr, "stderr")  # type: ignore[assignment]
        _installed = True


def _publish(line: str, source: str) -> None:
    text = str(line or "").rstrip("\r\n")
    if not text:
        return
    level, category = classify_log_line(text, source)
    entry = {
        "ts": time.time(),
        "source": source,
        "level": level,
        "category": category,
        "message": text,
        "raw": text,
    }
    with _lock:
        _buffer.append(entry)
        callback = _callback
    try:
        record_console_log(entry)
    except Exception:
        pass
    if callback is not None:
        try:
            callback(entry)
        except Exception:
            pass


class _TeeStream:
    def __init__(self, wrapped: TextIO, source: str) -> None:
        self._wrapped = wrapped
        self._source = source
        self._partial = ""
        self._lock = threading.RLock()

    def write(self, data: str) -> int:
        written = self._wrapped.write(data)
        self._wrapped.flush()
        with self._lock:
            self._partial += str(data)
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                _publish(line, self._source)
        return written

    def flush(self) -> None:
        self._wrapped.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._wrapped, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._wrapped.fileno()

    @property
    def encoding(self) -> str | None:
        return getattr(self._wrapped, "encoding", None)

    def __getattr__(self, name: str):
        return getattr(self._wrapped, name)
