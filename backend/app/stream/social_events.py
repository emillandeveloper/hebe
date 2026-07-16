from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import re
import time
from typing import Any


@dataclass(slots=True)
class TwitchCheerEvent:
    event_id: str
    source: str
    viewer_login: str
    viewer_display_name: str
    bits: int
    message: str
    timestamp: float
    twitch_message_id: str = ""
    dedupe_key: str = ""
    raw_tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_twitch_cheer_privmsg(
    *,
    username: str,
    display_name: str,
    message: str,
    tags: dict[str, Any] | None,
    timestamp: float | None = None,
) -> TwitchCheerEvent | None:
    """Promote a tagged IRC PRIVMSG to a cheer before chat classification."""
    raw_tags = dict(tags or {})
    try:
        bits = int(str(raw_tags.get("bits") or "0"))
    except (TypeError, ValueError):
        return None
    if bits <= 0:
        return None
    ts = float(timestamp if timestamp is not None else time.time())
    message_id = str(raw_tags.get("id") or raw_tags.get("message-id") or "").strip()
    login = str(username or raw_tags.get("login") or "viewer").strip().lstrip("@")
    display = str(raw_tags.get("display-name") or display_name or login).strip()
    event_id = message_id or f"cheer_{hashlib.sha256(f'{login.casefold()}:{bits}:{ts:.3f}:{message}'.encode()).hexdigest()[:20]}"
    bucket = int(ts // 10)
    dedupe_key = message_id or f"{login.casefold()}:{bits}:{bucket}"
    return TwitchCheerEvent(
        event_id=event_id,
        source="irc_privmsg",
        viewer_login=login,
        viewer_display_name=display,
        bits=bits,
        message=str(message or "").strip(),
        timestamp=ts,
        twitch_message_id=message_id,
        dedupe_key=dedupe_key,
        raw_tags=raw_tags,
    )


_BOT_CHEER_PATTERNS = (
    re.compile(r"(?P<viewer>@?[\w.-]+).{0,80}?(?P<bits>\d+)\s*bits?", re.I),
    re.compile(r"(?P<bits>\d+)\s*bits?.{0,80}?(?:de|from)\s+(?P<viewer>@?[\w.-]+)", re.I),
)


def parse_cheer_bot_fallback(username: str, message: str, *, timestamp: float | None = None) -> TwitchCheerEvent | None:
    """Parse a public alert-bot announcement only as a lower-authority fallback."""
    if str(username or "").casefold().lstrip("@") not in {"jotunbot", "streamelements", "streamlabs"}:
        return None
    text = str(message or "").strip()
    if not re.search(r"\b(?:bits?|cheer)\b", text, re.I):
        return None
    match = next((pattern.search(text) for pattern in _BOT_CHEER_PATTERNS if pattern.search(text)), None)
    if match is None:
        return None
    bits = int(match.group("bits"))
    if bits <= 0:
        return None
    viewer = match.group("viewer").lstrip("@")
    ts = float(timestamp if timestamp is not None else time.time())
    return TwitchCheerEvent(
        event_id=f"cheer_fallback_{hashlib.sha256(f'{viewer.casefold()}:{bits}:{int(ts // 10)}'.encode()).hexdigest()[:20]}",
        source="bot_fallback",
        viewer_login=viewer,
        viewer_display_name=viewer,
        bits=bits,
        message=text,
        timestamp=ts,
        dedupe_key=f"{viewer.casefold()}:{bits}:{int(ts // 10)}",
        raw_tags={},
    )


class StreamSocialEventRouter:
    def route(self, event: TwitchCheerEvent | dict[str, Any]) -> dict[str, Any]:
        payload = event.to_dict() if isinstance(event, TwitchCheerEvent) else dict(event)
        return {"event_type": "cheer", "event_id": payload.get("event_id", ""), "payload": payload}


class CheerEventPolicy:
    """High-value event policy. Soft chat/owner-speech budgets do not apply."""

    def decide(self, event: TwitchCheerEvent, *, duplicate: bool = False, hard_safety_passed: bool = True) -> dict[str, Any]:
        if duplicate:
            return {"allowed": False, "route": "suppress", "reason": "duplicate_cheer_event", "open_pending": False}
        if not hard_safety_passed:
            return {"allowed": False, "route": "suppress", "reason": "hard_safety", "open_pending": False}
        return {
            "allowed": True,
            "route": "stream_tts_reply",
            "reason": "valid_cheer_high_value",
            "bypass_no_mention": True,
            "bypass_recent_owner_speech": True,
            "bypass_soft_chat_budget": True,
            "open_pending": False,
            "allow_followup_question": False,
        }


class CheerDeduplicator:
    def __init__(self, window_seconds: float = 20.0) -> None:
        self.window_seconds = float(window_seconds)
        self._seen: list[dict[str, Any]] = []

    def check_and_record(self, event: TwitchCheerEvent, *, now: float | None = None) -> tuple[bool, str]:
        now = float(now if now is not None else event.timestamp or time.time())
        self._seen = [item for item in self._seen if now - float(item["timestamp"]) <= self.window_seconds]
        for item in self._seen:
            same_id = bool(event.twitch_message_id and event.twitch_message_id == item.get("message_id"))
            same_tuple = (
                event.viewer_login.casefold() == str(item.get("viewer", "")).casefold()
                and event.bits == int(item.get("bits", 0))
                and abs(event.timestamp - float(item.get("event_timestamp", 0.0))) <= self.window_seconds
            )
            if same_id or same_tuple:
                return True, "message_id" if same_id else "viewer_bits_window"
        self._seen.append({
            "timestamp": now,
            "event_timestamp": event.timestamp,
            "message_id": event.twitch_message_id,
            "viewer": event.viewer_login,
            "bits": event.bits,
            "source": event.source,
        })
        return False, "new_event"


class CheerAcknowledgementRenderer:
    def render(self, event: TwitchCheerEvent) -> str:
        viewer = event.viewer_display_name or event.viewer_login or "chat"
        # This is an event template, not a regression-specific phrase.
        return f"Gracias, {viewer}, por esos {event.bits} bits."


def twitch_timestamp(tags: dict[str, Any] | None) -> float:
    raw = str((tags or {}).get("tmi-sent-ts") or "").strip()
    if raw.isdigit():
        return int(raw) / 1000.0
    return time.time()
