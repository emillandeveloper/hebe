# app/core/ui_bridge.py
from typing import Callable, Optional

_EMIT: Optional[Callable[[str, dict], None]] = None


def set_emitter(fn: Callable[[str, dict], None] | None):
    global _EMIT
    _EMIT = fn


def emit(event_type: str, data: dict | None = None):
    if _EMIT:
        try:
            _EMIT(event_type, data or {})
        except Exception:
            pass