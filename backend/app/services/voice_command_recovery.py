from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
import unicodedata


@dataclass
class TranscriptNormalizationResult:
    raw_text: str
    normalized_text: str
    normalized_candidates: list[str] = field(default_factory=list)
    confidence: float = 1.0
    reason: str = "normalized"
    metadata: dict = field(default_factory=dict)

    def as_event(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "normalized_candidates": self.normalized_candidates,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
            "metadata": self.metadata,
        }


def normalize_for_voice(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").strip().lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in value)
    return " ".join(value.split())


def normalize_stt_transcript(raw_text: str, *, known_targets: list[str] | None = None) -> TranscriptNormalizationResult:
    raw = str(raw_text or "").strip()
    base = normalize_for_voice(raw)
    if not base:
        return TranscriptNormalizationResult(raw, "", [], 0.0, "empty")

    tokens = base.split()
    notes: list[str] = []
    candidates: list[str] = [base]

    tokens, wake_conf = _fix_wakeword(tokens)
    if wake_conf < 1.0:
        notes.append("wakeword_fuzzy")

    fixed_tokens = _fix_command_tokens(tokens)
    if fixed_tokens != tokens:
        notes.append("command_words")
    normalized = " ".join(fixed_tokens)

    target_fixed = _fix_attached_known_target(normalized, known_targets or [])
    if target_fixed != normalized:
        notes.append("known_target_spacing")
        normalized = target_fixed

    if normalized not in candidates:
        candidates.append(normalized)
    confidence = 0.75 if notes else 1.0
    return TranscriptNormalizationResult(
        raw_text=raw,
        normalized_text=normalized,
        normalized_candidates=candidates,
        confidence=confidence,
        reason=";".join(notes) if notes else "no_changes",
        metadata={"normalization_only": True},
    )


def _similar(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _fix_wakeword(tokens: list[str]) -> tuple[list[str], float]:
    if not tokens:
        return tokens, 0.0
    first = tokens[0]
    if first == "hebe":
        return tokens, 1.0
    if first in {"ebe", "jebe", "heve"}:
        return ["hebe", *tokens[1:]], 0.9
    if first == "eve":
        return ["hebe", *tokens[1:]], 0.72
    if _similar(first, "hebe") >= 0.72:
        return ["hebe", *tokens[1:]], 0.7
    return tokens, 1.0


def _fix_command_tokens(tokens: list[str]) -> list[str]:
    fixed: list[str] = []
    command_words = {"promo", "promocion", "promociona", "promocioname", "shoutout", "so"}
    for i, token in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if token == "az":
            fixed.append("haz")
        elif token == "as" and nxt in command_words:
            fixed.append("haz")
        elif token in {"prommo", "pormo", "promosion", "promocioname"}:
            fixed.append("promo")
        elif token in {"promocion", "promociona"}:
            fixed.append("promociona")
        else:
            fixed.append(token)
    return fixed


def _compact_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", normalize_for_voice(value).lstrip("@"))


def _fix_attached_known_target(normalized: str, known_targets: list[str]) -> str:
    words = normalized.split()
    if not words:
        return normalized
    known = [(target, _compact_name(target)) for target in known_targets if _compact_name(target)]
    if not known:
        return normalized
    fixed = list(words)
    for i, word in enumerate(words):
        compact = _compact_name(word)
        if not compact.startswith("a") or len(compact) <= 4:
            continue
        without_a = compact[1:]
        for original, candidate in known:
            if without_a == candidate or _similar(without_a, candidate) >= 0.86:
                fixed[i] = f"a {candidate}"
                return " ".join(fixed)
    return normalized
