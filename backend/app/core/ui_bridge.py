# app/core/ui_bridge.py
from __future__ import annotations

import hashlib
import threading
import time
import unicodedata
import uuid
from typing import Any, Callable, Optional

_EMIT: Optional[Callable[[str, dict], None]] = None
_MESSAGE_ID_LOCK = threading.Lock()
_RECENT_MESSAGE_IDS: dict[str, tuple[str, float]] = {}
_MESSAGE_ID_TTL_SECONDS = 45.0

_ASSISTANT_EVENT_TYPES = {"chat.assistant", "llm.final", "dataset.example"}
_USER_EVENT_TYPES = {"chat.user"}
_MESSAGE_EVENT_TYPES = _ASSISTANT_EVENT_TYPES | _USER_EVENT_TYPES


def set_emitter(fn: Callable[[str, dict], None] | None):
    global _EMIT
    _EMIT = fn


def _normalize_message_text(text: Any) -> str:
    raw = str(text or "").strip().lower()
    without_accents = "".join(
        char for char in unicodedata.normalize("NFKD", raw)
        if not unicodedata.combining(char)
    )
    cleaned = "".join(char if char.isalnum() or char.isspace() else " " for char in without_accents)
    return " ".join(cleaned.split())


def _message_role_for_event(event_type: str) -> str:
    return "user" if event_type in _USER_EVENT_TYPES else "assistant"


def _message_text_for_event(event_type: str, data: dict[str, Any]) -> str:
    if event_type == "dataset.example":
        return str(data.get("response") or "").strip()
    return str(data.get("text") or "").strip()


def _message_id_for(event_type: str, data: dict[str, Any]) -> str | None:
    explicit = str(data.get("message_id") or data.get("id") or "").strip()
    if explicit:
        return explicit

    text = _message_text_for_event(event_type, data)
    normalized = _normalize_message_text(text)
    if not normalized:
        return None

    role = _message_role_for_event(event_type)
    cache_key = f"{role}:{normalized}"
    now = time.time()
    with _MESSAGE_ID_LOCK:
        expired = [
            key for key, (_, ts) in _RECENT_MESSAGE_IDS.items()
            if now - ts > _MESSAGE_ID_TTL_SECONDS
        ]
        for key in expired:
            _RECENT_MESSAGE_IDS.pop(key, None)

        existing = _RECENT_MESSAGE_IDS.get(cache_key)
        if existing:
            message_id, _ = existing
            _RECENT_MESSAGE_IDS[cache_key] = (message_id, now)
            return message_id

        digest = hashlib.sha1(f"{cache_key}:{int(now * 1000)}".encode("utf-8")).hexdigest()[:16]
        message_id = f"msg_{digest}"
        _RECENT_MESSAGE_IDS[cache_key] = (message_id, now)
        return message_id


def _with_event_metadata(event_type: str, data: dict | None) -> dict:
    payload = dict(data or {})
    event_id = str(payload.get("event_id") or "").strip() or f"evt_{uuid.uuid4().hex}"
    payload["event_id"] = event_id
    if event_type in _MESSAGE_EVENT_TYPES:
        message_id = _message_id_for(event_type, payload)
        if message_id:
            payload["message_id"] = message_id
    return payload


def emit(event_type: str, data: dict | None = None):
    if _EMIT:
        try:
            _EMIT(event_type, _with_event_metadata(event_type, data))
        except Exception:
            pass
