from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import asdict, dataclass


def _norm(value: str) -> str:
    raw = "".join(
        char for char in unicodedata.normalize("NFKD", str(value or "").casefold())
        if not unicodedata.combining(char)
    )
    return " ".join(re.sub(r"[^a-z]+", " ", raw).split())


@dataclass(frozen=True, slots=True)
class StreamOutputLanguageDecision:
    text: str
    expected_language: str
    detected_language: str
    action: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class StreamOutputLanguagePolicy:
    """Keeps autonomous speech in the configured language; viewer replies may mirror."""

    _ES = {
        "que", "esto", "esa", "ese", "vaya", "vamos", "bien", "pero", "ahora", "tension",
        "casi", "tienes", "queda", "vida", "enemigo", "madre", "menudo", "momento", "aqui",
    }
    _EN = {
        "the", "this", "that", "what", "wow", "nice", "but", "now", "tense", "almost",
        "you", "have", "health", "enemy", "here", "moment", "close", "oof", "was", "is",
    }

    def __init__(self, configured_language: str | None = None) -> None:
        self.configured_language = self._allowed(
            configured_language or os.getenv("HEBE_STREAM_OUTPUT_LANGUAGE", "es")
        )

    def set_owner_preference(self, language: str) -> str:
        self.configured_language = self._allowed(language)
        return self.configured_language

    def expected_language(
        self,
        *,
        event_type: str = "spontaneous_stream_comment",
        source_language: str = "",
        owner_requested_language: str = "",
    ) -> str:
        if owner_requested_language:
            return self._allowed(owner_requested_language)
        if str(event_type or "").lower() in {"twitch_chat", "direct_viewer_reply", "viewer_reply"}:
            source = str(source_language or "").strip().lower()
            if source in {"es", "en"}:
                return source
        return self.configured_language

    def enforce(
        self,
        text: str,
        *,
        event_type: str = "spontaneous_stream_comment",
        source_language: str = "",
        owner_requested_language: str = "",
        fallback: str = "",
    ) -> StreamOutputLanguageDecision:
        expected = self.expected_language(
            event_type=event_type,
            source_language=source_language,
            owner_requested_language=owner_requested_language,
        )
        detected = self.detect(text)
        if not str(text or "").strip():
            return StreamOutputLanguageDecision("", expected, detected, "suppress", "empty_output")
        if detected in {expected, "neutral"}:
            return StreamOutputLanguageDecision(str(text).strip(), expected, detected, "allow", "language_matches")
        safe = str(fallback or "").strip()
        if safe and self.detect(safe) in {expected, "neutral"}:
            return StreamOutputLanguageDecision(safe, expected, detected, "rewrite", "configured_language_mismatch")
        safe = "Uf, qué tensión." if expected == "es" else "Oof, that was tense."
        return StreamOutputLanguageDecision(safe, expected, detected, "rewrite", "configured_language_mismatch")

    def detect(self, text: str) -> str:
        words = set(_norm(text).split())
        if not words:
            return "neutral"
        es = len(words & self._ES)
        en = len(words & self._EN)
        if es == en:
            return "neutral"
        return "es" if es > en else "en"

    @staticmethod
    def _allowed(language: str) -> str:
        value = str(language or "es").strip().lower()
        if value not in {"es", "en"}:
            raise ValueError(f"unsupported_stream_output_language:{value}")
        return value


__all__ = ["StreamOutputLanguageDecision", "StreamOutputLanguagePolicy"]
