from __future__ import annotations

import re
import time
from urllib.parse import unquote


def parse_irc_tags(line: str) -> dict[str, str]:
    value = str(line or "").strip()
    if not value.startswith("@"):
        return {}
    tag_text = value.split(" ", 1)[0][1:]
    tags: dict[str, str] = {}
    for part in tag_text.split(";"):
        if not part:
            continue
        key, _, raw = part.partition("=")
        if key:
            tags[key] = _unescape_irc_tag(raw)
    return tags


def parse_raid_usernotice(line: str, *, now: float | None = None) -> dict | None:
    value = str(line or "").strip()
    if " USERNOTICE " not in value:
        return None
    tags = parse_irc_tags(value)
    if str(tags.get("msg-id") or "").lower() != "raid":
        return None
    login = tags.get("msg-param-login") or tags.get("login") or ""
    display = tags.get("msg-param-displayName") or tags.get("display-name") or login
    viewers = _safe_int(tags.get("msg-param-viewerCount") or tags.get("msg-param-viewerCount".lower()))
    event_id = tags.get("id") or f"raid:{login or display}:{viewers}:{int(now or time.time())}"
    return {
        "event_id": event_id,
        "source": "irc_usernotice",
        "user_login": login or display,
        "display_name": display or login or "alguien",
        "viewer_count": viewers,
        "raw_irc": value,
        "ts": float(now or time.time()),
    }


def parse_raid_bot_message(username: str, message: str, *, now: float | None = None) -> dict | None:
    text = str(message or "").strip()
    if not text:
        return None
    patterns = (
        r"^(?P<name>.+?)\s+just\s+raided\s+the\s+channel\s+with\s+(?P<count>\d+)\s+viewers?!?$",
        r"^(?P<name>.+?)\s+has\s+raided\s+the\s+channel\s+with\s+(?P<count>\d+)\s+viewers?!?$",
        r"^raid(?:\s+entrante)?\s+de\s+(?P<name>.+?)\s+con\s+(?P<count>\d+)\s+(?:viewers?|espectadores?)!?$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        display = str(match.group("name") or "").strip(" @,.;:")
        viewers = _safe_int(match.group("count"))
        if not display:
            return None
        login = re.sub(r"[^A-Za-z0-9_]", "", display) or display
        return {
            "event_id": f"raid_bot:{login}:{viewers}:{int(now or time.time())}",
            "source": "bot_fallback",
            "bot_username": str(username or "").strip(),
            "user_login": login,
            "display_name": display,
            "viewer_count": viewers,
            "message_text": text,
            "ts": float(now or time.time()),
        }
    return None


def _safe_int(value: object) -> int:
    try:
        return max(0, int(str(value or "0").strip()))
    except (TypeError, ValueError):
        return 0


def _unescape_irc_tag(value: str) -> str:
    # Twitch IRCv3 tags use slash escapes, while a few clients percent-encode spaces.
    decoded = unquote(str(value or ""))
    return (
        decoded.replace(r"\s", " ")
        .replace(r"\:", ";")
        .replace(r"\\", "\\")
        .replace(r"\r", "\r")
        .replace(r"\n", "\n")
    )
