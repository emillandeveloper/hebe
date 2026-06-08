from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class StreamIntentCandidate:
    intent: str
    confidence: float
    entities: dict[str, str] = field(default_factory=dict)
    reason: str = ""


class StreamIntentParser:
    """Small semantic parser for stream commands using concepts, not sentence lists."""

    shoutout_concepts = {"promo", "promocion", "promociona", "promocionar", "shoutout", "so", "recomienda"}
    ambient_concepts = {"stt", "ambiental", "ambiente"}
    enable_concepts = {"activa", "enciende", "reanuda", "pon", "enable", "resume", "on"}
    disable_concepts = {"desactiva", "apaga", "pausa", "quita", "disable", "pause", "off"}
    target_prepositions = {"a", "al", "para", "to"}
    filler = {"haz", "hazle", "dale", "manda", "pon", "un", "una", "el", "la", "de", "del", "give"}

    def parse(self, text: str, *, raw_text: str | None = None) -> list[StreamIntentCandidate]:
        normalized = self.normalize(text)
        raw = str(raw_text if raw_text is not None else text or "").strip()
        candidates: list[StreamIntentCandidate] = []
        ambient = self._parse_ambient_stt(normalized)
        if ambient:
            candidates.append(ambient)
        shoutout = self._parse_shoutout(normalized, raw)
        if shoutout:
            candidates.append(shoutout)
        return candidates

    def _parse_ambient_stt(self, normalized: str) -> StreamIntentCandidate | None:
        tokens = normalized.split()
        token_set = set(tokens)
        if "stt" not in token_set or not (token_set & self.ambient_concepts):
            return None
        if token_set & self.disable_concepts:
            return StreamIntentCandidate("stream_ambient_stt_disabled", 0.93, reason="disable_stt_ambient")
        if token_set & self.enable_concepts:
            return StreamIntentCandidate("stream_ambient_stt_enabled", 0.93, reason="enable_stt_ambient")
        return None

    def _parse_shoutout(self, normalized: str, raw_text: str) -> StreamIntentCandidate | None:
        tokens = normalized.split()
        if not tokens:
            return None
        concept_indexes = [idx for idx, token in enumerate(tokens) if token in self.shoutout_concepts]
        if not concept_indexes:
            return None
        concept_index = concept_indexes[0]
        target_text = self._extract_target(tokens, concept_index, raw_text)
        confidence = 0.92 if target_text else 0.88
        return StreamIntentCandidate(
            "twitch_shoutout",
            confidence,
            entities={"target_text": target_text or ""},
            reason="shoutout_concept",
        )

    def _extract_target(self, tokens: list[str], concept_index: int, raw_text: str) -> str:
        tail = tokens[concept_index + 1 :]
        for idx, token in enumerate(tail):
            if token in self.target_prepositions:
                return self._raw_tail_after_token(raw_text, tail[idx + 1 :]) or " ".join(tail[idx + 1 :]).strip()
        if tail and tail[0] not in self.filler:
            return self._raw_tail_after_token(raw_text, tail) or " ".join(tail).strip()
        if len(tail) > 1:
            cleaned = [token for token in tail if token not in self.filler]
            return self._raw_tail_after_token(raw_text, cleaned) or " ".join(cleaned).strip()
        return ""

    def _raw_tail_after_token(self, raw_text: str, normalized_tail: list[str]) -> str:
        if not raw_text or not normalized_tail:
            return ""
        first = normalized_tail[0]
        raw_tokens = str(raw_text or "").strip().split()
        for index, token in enumerate(raw_tokens):
            if self.normalize(token) == first:
                return " ".join(raw_tokens[index:]).strip(" ,.;:")
        return ""

    def normalize(self, text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() or ch == "_" else " " for ch in str(text or "").strip().lower())
        cleaned = cleaned.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        cleaned = cleaned.replace("ü", "u").replace("ñ", "n")
        return re.sub(r"\s+", " ", cleaned).strip()
