from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Iterable


@dataclass(frozen=True)
class WakeNameResolution:
    addressed_to_hebe: bool = False
    wake_command: bool = False
    sleep_command: bool = False
    matched_name: str | None = None
    stripped_text: str = ""
    confidence: float = 0.0
    reason: str = "no_match"


class WakeNameResolver:
    """Resolve whether a transcript is addressing Hebe without phrase tables."""

    canonical_names = ("hebe", "ebe", "eve", "eb", "e b", "jebe", "heve")
    wake_concepts = {"despierta", "levanta", "wake", "awake"}
    sleep_concepts = {"duerme", "descansa", "dormir", "sleep", "espera", "standby"}

    def resolve(
        self,
        *,
        raw_text: str,
        normalized_text: str,
        source: str = "",
        is_sleeping: bool = False,
        command_markers: Iterable[str] | None = None,
    ) -> WakeNameResolution:
        normalized = self._normalize(normalized_text or raw_text)
        tokens = normalized.split()
        if not tokens:
            return WakeNameResolution(stripped_text="")

        name_index, matched_name, name_score = self._find_name(tokens)
        stripped = self._strip_name(tokens, name_index)
        stripped_tokens = stripped.split()
        has_wake = any(token in self.wake_concepts for token in stripped_tokens)
        has_sleep = any(token in self.sleep_concepts for token in stripped_tokens)
        has_command_context = self._has_command_context(stripped_tokens, command_markers)

        if matched_name:
            confidence = name_score
            if matched_name == "eve" and not (has_wake or has_sleep or has_command_context or is_sleeping):
                return WakeNameResolution(
                    addressed_to_hebe=False,
                    matched_name=matched_name,
                    stripped_text=stripped,
                    confidence=min(confidence, 0.45),
                    reason="weak_eve_context",
                )
            return WakeNameResolution(
                addressed_to_hebe=True,
                wake_command=has_wake,
                sleep_command=has_sleep,
                matched_name=matched_name,
                stripped_text=stripped,
                confidence=confidence,
                reason="name_with_command_context" if has_command_context else "name_match",
            )

        if is_sleeping:
            return WakeNameResolution(stripped_text=normalized, reason="sleeping_without_name")

        trusted_source = source in {"stt_voice", "voice", "ui", "typed_ui"}
        if trusted_source and has_command_context:
            return WakeNameResolution(
                addressed_to_hebe=True,
                wake_command=False,
                sleep_command=has_sleep,
                matched_name=None,
                stripped_text=normalized,
                confidence=0.72,
                reason="trusted_source_command_context",
            )

        return WakeNameResolution(stripped_text=normalized)

    def _find_name(self, tokens: list[str]) -> tuple[int | None, str | None, float]:
        if not tokens:
            return None, None, 0.0
        candidates = [(0, tokens[0])]
        if len(tokens) >= 2:
            candidates.append((0, f"{tokens[0]} {tokens[1]}"))
        if len(tokens) > 1:
            candidates.append((len(tokens) - 1, tokens[-1]))
        best: tuple[int | None, str | None, float] = (None, None, 0.0)
        for index, token in candidates:
            compact = token.replace(" ", "")
            for name in self.canonical_names:
                name_compact = name.replace(" ", "")
                score = 1.0 if compact == name_compact else SequenceMatcher(None, compact, name_compact).ratio()
                if score > best[2] and score >= 0.78:
                    best = (index, name_compact, score)
        return best

    def _strip_name(self, tokens: list[str], name_index: int | None) -> str:
        if name_index is None:
            return " ".join(tokens)
        values = list(tokens)
        if name_index == 0:
            if len(values) >= 2 and f"{values[0]}{values[1]}" == "eb":
                values = values[2:]
            else:
                values = values[1:]
        elif name_index == len(values) - 1:
            values = values[:-1]
        return " ".join(values).strip()

    def _has_command_context(self, tokens: list[str], command_markers: Iterable[str] | None) -> bool:
        markers = {self._normalize(item) for item in (command_markers or []) if str(item or "").strip()}
        marker_tokens = {part for marker in markers for part in marker.split()}
        return bool(set(tokens) & marker_tokens)

    def _normalize(self, text: str) -> str:
        value = str(text or "").lower()
        value = value.replace(".", " ")
        value = value.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        value = value.replace("ü", "u").replace("ñ", "n")
        value = re.sub(r"[^a-z0-9_ ]+", " ", value)
        return " ".join(value.split())
