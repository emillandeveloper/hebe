from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable


ACTION_LOCAL_REPLY = "local_reply"
ACTION_LOCAL_UI_MESSAGE = "local_ui_message"
ACTION_LOCAL_TTS = "local_tts"
ACTION_MEMORY_WRITE = "memory_write"
ACTION_SESSION_CONTEXT_UPDATE = "session_context_update"
ACTION_TWITCH_REPLY = "twitch_reply"
ACTION_TWITCH_ACTION = "twitch_action"
ACTION_PROMOTION_SHOUTOUT = "promotion_shoutout"
ACTION_APP_CONTROL = "app_control"
ACTION_SCHEDULER = "scheduler"
ACTION_DESTRUCTIVE_ACTION = "destructive_action"

ACTION_CATEGORIES = (
    ACTION_LOCAL_REPLY,
    ACTION_LOCAL_UI_MESSAGE,
    ACTION_LOCAL_TTS,
    ACTION_MEMORY_WRITE,
    ACTION_SESSION_CONTEXT_UPDATE,
    ACTION_TWITCH_REPLY,
    ACTION_TWITCH_ACTION,
    ACTION_PROMOTION_SHOUTOUT,
    ACTION_APP_CONTROL,
    ACTION_SCHEDULER,
    ACTION_DESTRUCTIVE_ACTION,
)

KNOWN_BOT_USERNAMES = {
    "jotunbot",
    "streamelements",
    "nightbot",
    "streamlabs",
    "moobot",
    "fossabot",
    "wizebot",
}

OWNER_SOURCES = {"owner_ui", "owner_stt_direct"}
FOLLOWUP_SOURCES = {"owner_stt_followup"}
TWITCH_SOURCES = {"twitch_viewer", "twitch_bot", "twitch_system"}

WAKE_NAMES = {"hebe", "ebe", "eve", "jebe", "heve", "ehbe"}
COMMAND_MARKERS = {
    "abre",
    "abrir",
    "actualiza",
    "actualizar",
    "cambia",
    "cambiar",
    "configura",
    "configurar",
    "dime",
    "di",
    "haz",
    "hacer",
    "lee",
    "leer",
    "lanza",
    "lanzar",
    "manda",
    "mandar",
    "promociona",
    "promocionar",
    "responde",
    "responder",
    "shoutout",
    "so",
    "stop",
}

ENGLISH_FUNCTION_WORDS = {
    "a",
    "am",
    "and",
    "are",
    "be",
    "but",
    "do",
    "for",
    "from",
    "gonna",
    "i",
    "im",
    "in",
    "is",
    "it",
    "me",
    "my",
    "not",
    "of",
    "on",
    "please",
    "so",
    "the",
    "to",
    "we",
    "will",
    "with",
    "you",
    "your",
}

SPANISH_FUNCTION_WORDS = {
    "a",
    "al",
    "con",
    "de",
    "del",
    "el",
    "en",
    "la",
    "lo",
    "los",
    "me",
    "mi",
    "no",
    "para",
    "que",
    "se",
    "si",
    "te",
    "un",
    "una",
    "y",
    "yo",
}


def normalize_firewall_text(value: str | None) -> str:
    raw = str(value or "").casefold()
    raw = "".join(
        char for char in unicodedata.normalize("NFKD", raw)
        if not unicodedata.combining(char)
    )
    raw = re.sub(r"[^a-z0-9_@' ]+", " ", raw)
    raw = raw.replace("'", "")
    return " ".join(raw.split())


def _clean_username(username: str | None) -> str:
    return normalize_firewall_text(username).lstrip("@")


def is_known_bot_username(username: str | None, extra_bot_usernames: Iterable[str] | None = None) -> bool:
    user = _clean_username(username)
    if not user:
        return True
    bot_names = set(KNOWN_BOT_USERNAMES)
    bot_names.update(_clean_username(item) for item in (extra_bot_usernames or []) if item)
    return user in bot_names


def _looks_directly_addressed(normalized: str) -> bool:
    tokens = set(normalized.split())
    if tokens & WAKE_NAMES:
        return True
    return any(re.search(rf"(?<![a-z0-9_])@?{re.escape(name)}(?![a-z0-9_])", normalized) for name in WAKE_NAMES)


def _has_command_structure(normalized: str) -> bool:
    tokens = set(normalized.split())
    return bool(tokens & COMMAND_MARKERS) or _looks_directly_addressed(normalized)


def looks_like_media_or_singing(text: str | None) -> tuple[bool, str]:
    normalized = normalize_firewall_text(text)
    if not normalized:
        return False, ""
    if _has_command_structure(normalized):
        return False, ""

    tokens = normalized.split()
    if len(tokens) < 3:
        return False, ""

    repeated_token_count = len(tokens) - len(set(tokens))
    if len(tokens) >= 5 and repeated_token_count >= 2:
        return True, "repeated_fragment"

    english_hits = sum(1 for token in tokens if token in ENGLISH_FUNCTION_WORDS)
    spanish_hits = sum(1 for token in tokens if token in SPANISH_FUNCTION_WORDS)
    ascii_word_count = sum(1 for token in tokens if re.fullmatch(r"[a-z0-9_]+", token))
    english_ratio = english_hits / max(1, len(tokens))
    mostly_ascii = ascii_word_count >= max(2, len(tokens) - 1)
    if mostly_ascii and english_hits >= 2 and english_ratio >= 0.25 and spanish_hits <= 1:
        return True, "singing_or_lyrics"

    short_loose_fragment = len(tokens) <= 7 and english_hits >= 1 and spanish_hits == 0 and mostly_ascii
    if short_loose_fragment:
        return True, "background_media_fragment"

    return False, ""


@dataclass(frozen=True)
class InputFirewallDecision:
    source: str
    authority: str
    input_trust: str
    firewall_decision: str
    reason: str
    allowed_actions: list[str] = field(default_factory=list)
    blocked_actions: list[str] = field(default_factory=list)
    media_or_singing_detected: bool = False
    media_reason: str = ""
    bot_detected: bool = False
    stream_is_live: bool = False
    followup_window_used: bool = False
    would_call_llm: bool = False
    would_send_twitch: bool = False
    username: str = ""
    event_type: str = ""
    is_simulation: bool = False

    def allows_action(self, action: str) -> bool:
        return action in set(self.allowed_actions)

    def blocks_action(self, action: str) -> bool:
        return action in set(self.blocked_actions)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "authority": self.authority,
            "input_trust": self.input_trust,
            "firewall_decision": self.firewall_decision,
            "reason": self.reason,
            "allowed_actions": list(self.allowed_actions),
            "blocked_actions": list(self.blocked_actions),
            "media_or_singing_detected": bool(self.media_or_singing_detected),
            "media_reason": self.media_reason,
            "bot_detected": bool(self.bot_detected),
            "stream_is_live": bool(self.stream_is_live),
            "followup_window_used": bool(self.followup_window_used),
            "would_call_llm": bool(self.would_call_llm),
            "would_send_twitch": bool(self.would_send_twitch),
            "username": self.username,
            "event_type": self.event_type,
            "is_simulation": bool(self.is_simulation),
        }


class InputAuthorityFirewall:
    def __init__(self, *, extra_bot_usernames: Iterable[str] | None = None):
        self.extra_bot_usernames = tuple(extra_bot_usernames or ())

    def is_bot_user(self, username: str | None) -> bool:
        return is_known_bot_username(username, self.extra_bot_usernames)

    def decide(
        self,
        *,
        source: str,
        text: str | None = "",
        username: str | None = "",
        stream_is_live: bool = False,
        is_simulation: bool = False,
        addressed_to_hebe: bool = False,
        pending_followup: bool = False,
        has_action_intent: bool = False,
        event_type: str | None = "",
    ) -> InputFirewallDecision:
        source = self._normalize_source(source, event_type=event_type)
        normalized_text = normalize_firewall_text(text)
        media_detected, media_reason = looks_like_media_or_singing(text)
        bot_detected = source in {"twitch_viewer", "twitch_bot"} and self.is_bot_user(username)
        if bot_detected:
            source = "twitch_bot"

        if is_simulation:
            return self._decision(
                source="simulation",
                authority="bot" if bot_detected else ("viewer" if source in TWITCH_SOURCES else "system"),
                input_trust="simulation_only",
                firewall_decision="allow",
                reason="simulation_mode",
                allowed_actions=[ACTION_LOCAL_UI_MESSAGE],
                media_detected=media_detected,
                media_reason=media_reason,
                bot_detected=bot_detected,
                stream_is_live=stream_is_live,
                followup_window_used=pending_followup,
                would_call_llm=bool(addressed_to_hebe and not bot_detected),
                would_send_twitch=False,
                username=username,
                event_type=event_type or "",
                is_simulation=True,
            )

        if bot_detected:
            return self._ignore(
                source="twitch_bot",
                authority="bot",
                input_trust="untrusted_bot",
                reason="bot_message",
                media_detected=media_detected,
                media_reason=media_reason,
                bot_detected=True,
                stream_is_live=stream_is_live,
                followup_window_used=pending_followup,
                username=username,
                event_type=event_type,
            )

        if source == "media_or_music" or (source == "ambient_stt" and media_detected):
            return self._ignore(
                source=source,
                authority="ambient",
                input_trust="untrusted_ambient",
                reason="media_or_singing_stt",
                media_detected=True,
                media_reason=media_reason or "media_or_singing",
                stream_is_live=stream_is_live,
                followup_window_used=pending_followup,
                username=username,
                event_type=event_type,
            )

        if source == "ambient_stt":
            if pending_followup:
                return self._ignore(
                    source=source,
                    authority="ambient",
                    input_trust="untrusted_ambient",
                    reason="ambient_followup_rejected",
                    stream_is_live=stream_is_live,
                    followup_window_used=True,
                    username=username,
                    event_type=event_type,
                )
            if not stream_is_live:
                return self._ignore(
                    source=source,
                    authority="ambient",
                    input_trust="untrusted_offline_stream",
                    reason="offline_stream",
                    stream_is_live=False,
                    username=username,
                    event_type=event_type,
                )
            return self._decision(
                source=source,
                authority="ambient",
                input_trust="untrusted_ambient",
                firewall_decision="allow_context_only",
                reason="ambient_context_only",
                allowed_actions=[ACTION_SESSION_CONTEXT_UPDATE],
                stream_is_live=True,
                username=username,
                event_type=event_type or "",
            )

        if source in OWNER_SOURCES:
            allowed = [
                ACTION_LOCAL_REPLY,
                ACTION_LOCAL_UI_MESSAGE,
                ACTION_LOCAL_TTS,
                ACTION_MEMORY_WRITE,
                ACTION_SESSION_CONTEXT_UPDATE,
                ACTION_APP_CONTROL,
                ACTION_SCHEDULER,
            ]
            if stream_is_live and has_action_intent:
                allowed.extend([ACTION_TWITCH_ACTION, ACTION_PROMOTION_SHOUTOUT])
            return self._decision(
                source=source,
                authority="owner",
                input_trust="trusted_direct",
                firewall_decision="allow",
                reason="owner_direct",
                allowed_actions=allowed,
                media_detected=media_detected,
                media_reason=media_reason,
                stream_is_live=stream_is_live,
                followup_window_used=pending_followup,
                would_call_llm=True,
                would_send_twitch=stream_is_live and has_action_intent,
                username=username,
                event_type=event_type or "",
            )

        if source in FOLLOWUP_SOURCES:
            return self._decision(
                source=source,
                authority="owner",
                input_trust="trusted_followup",
                firewall_decision="allow",
                reason="owner_related_followup",
                allowed_actions=[
                    ACTION_LOCAL_REPLY,
                    ACTION_LOCAL_UI_MESSAGE,
                    ACTION_LOCAL_TTS,
                    ACTION_SESSION_CONTEXT_UPDATE,
                ],
                stream_is_live=stream_is_live,
                followup_window_used=True,
                would_call_llm=True,
                would_send_twitch=False,
                username=username,
                event_type=event_type or "",
            )

        if source == "twitch_viewer":
            if not stream_is_live:
                return self._decision(
                    source=source,
                    authority="viewer",
                    input_trust="untrusted_offline_stream",
                    firewall_decision="block_reply",
                    reason="offline_stream",
                    allowed_actions=[ACTION_LOCAL_UI_MESSAGE],
                    media_detected=media_detected,
                    media_reason=media_reason,
                    stream_is_live=False,
                    username=username,
                    event_type=event_type or "",
                )
            return self._decision(
                source=source,
                authority="viewer",
                input_trust="trusted_direct",
                firewall_decision="allow",
                reason="live_viewer_message",
                allowed_actions=[ACTION_LOCAL_UI_MESSAGE, ACTION_TWITCH_REPLY],
                media_detected=media_detected,
                media_reason=media_reason,
                stream_is_live=True,
                would_call_llm=bool(addressed_to_hebe),
                would_send_twitch=bool(addressed_to_hebe),
                username=username,
                event_type=event_type or "",
            )

        if source in {"twitch_system", "internal_event"}:
            is_twitch_event = str(event_type or "").startswith("twitch_") or source == "twitch_system"
            if is_twitch_event and not stream_is_live:
                return self._decision(
                    source=source,
                    authority="system",
                    input_trust="untrusted_offline_stream",
                    firewall_decision="block_action",
                    reason="offline_stream",
                    allowed_actions=[ACTION_LOCAL_UI_MESSAGE],
                    media_detected=media_detected,
                    media_reason=media_reason,
                    stream_is_live=False,
                    username=username,
                    event_type=event_type or "",
                )
            return self._decision(
                source=source,
                authority="system",
                input_trust="trusted_direct",
                firewall_decision="allow",
                reason="system_event",
                allowed_actions=[ACTION_LOCAL_UI_MESSAGE, ACTION_SESSION_CONTEXT_UPDATE, ACTION_TWITCH_REPLY],
                media_detected=media_detected,
                media_reason=media_reason,
                stream_is_live=stream_is_live,
                would_call_llm=is_twitch_event,
                would_send_twitch=is_twitch_event and stream_is_live,
                username=username,
                event_type=event_type or "",
            )

        if normalized_text:
            return self._decision(
                source=source or "internal_event",
                authority="none",
                input_trust="untrusted_ambient",
                firewall_decision="allow_context_only",
                reason="unknown_source_context_only",
                allowed_actions=[ACTION_LOCAL_UI_MESSAGE],
                media_detected=media_detected,
                media_reason=media_reason,
                stream_is_live=stream_is_live,
                username=username,
                event_type=event_type or "",
            )
        return self._ignore(
            source=source or "internal_event",
            authority="none",
            input_trust="untrusted_ambient",
            reason="empty_input",
            media_detected=media_detected,
            media_reason=media_reason,
            stream_is_live=stream_is_live,
            username=username,
            event_type=event_type,
        )

    def _normalize_source(self, source: str | None, *, event_type: str | None = "") -> str:
        value = str(source or "").strip().lower()
        if value in {"ui", "typed_ui", "owner"}:
            return "owner_ui"
        if value in {"stt_direct", "direct_stt", "owner_stt_direct"}:
            return "owner_stt_direct"
        if value in {"stt_followup", "direct_stt_followup", "owner_stt_followup"}:
            return "owner_stt_followup"
        if value in {"stt_voice", "voice", "ambient", "ambient_stt"}:
            return "ambient_stt"
        if value in {"twitch_chat", "twitch_chat_observe", "twitch", "twitch_viewer"}:
            return "twitch_viewer"
        if value in {"twitch_bot"}:
            return "twitch_bot"
        if value in {"twitch_event", "twitch_system"}:
            return "twitch_system"
        if value in {"simulation", "dev_simulation"}:
            return "simulation"
        if value in {"media", "music", "media_or_music"}:
            return "media_or_music"
        if str(event_type or "").startswith("twitch_"):
            return "twitch_system"
        return value or "internal_event"

    def _ignore(self, **kwargs) -> InputFirewallDecision:
        return self._decision(
            firewall_decision="ignore",
            allowed_actions=[],
            would_call_llm=False,
            would_send_twitch=False,
            **kwargs,
        )

    def _decision(
        self,
        *,
        source: str,
        authority: str,
        input_trust: str,
        firewall_decision: str,
        reason: str,
        allowed_actions: list[str],
        media_detected: bool = False,
        media_reason: str = "",
        bot_detected: bool = False,
        stream_is_live: bool = False,
        followup_window_used: bool = False,
        would_call_llm: bool = False,
        would_send_twitch: bool = False,
        username: str | None = "",
        event_type: str | None = "",
        is_simulation: bool = False,
    ) -> InputFirewallDecision:
        allowed = list(dict.fromkeys(action for action in allowed_actions if action in ACTION_CATEGORIES))
        blocked = [action for action in ACTION_CATEGORIES if action not in set(allowed)]
        return InputFirewallDecision(
            source=source,
            authority=authority,
            input_trust=input_trust,
            firewall_decision=firewall_decision,
            reason=reason,
            allowed_actions=allowed,
            blocked_actions=blocked,
            media_or_singing_detected=bool(media_detected),
            media_reason=media_reason,
            bot_detected=bool(bot_detected),
            stream_is_live=bool(stream_is_live),
            followup_window_used=bool(followup_window_used),
            would_call_llm=bool(would_call_llm),
            would_send_twitch=bool(would_send_twitch),
            username=str(username or ""),
            event_type=str(event_type or ""),
            is_simulation=bool(is_simulation),
        )
