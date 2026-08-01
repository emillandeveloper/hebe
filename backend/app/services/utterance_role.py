from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import StrEnum
import re
import time


class UtteranceRole(StrEnum):
    DIRECT_COMMAND = "direct_command"
    OWNER_COMMENTARY = "owner_commentary"
    OWNER_QUESTION_TO_STREAM = "owner_question_to_stream"
    QUOTED_OR_READ_DIALOGUE = "quoted_or_read_dialogue"
    GAME_AUDIO_BLEED = "game_audio_bleed"
    CONVERSATIONAL_FILLER = "conversational_filler"
    UNCERTAIN = "uncertain"


@dataclass(slots=True)
class UtteranceRoleDecision:
    role: UtteranceRole
    confidence: float
    signals: list[str]
    context_allowed: bool
    discourse_allowed: bool
    action_allowed: bool

    def to_dict(self) -> dict:
        result = asdict(self)
        result["role"] = self.role.value
        return result


class UtteranceRoleClassifier:
    """Conservative post-STT role classifier; it never grants owner authority."""

    def __init__(self, *, history_size: int = 8, dialogue_window_seconds: float = 45.0) -> None:
        self.history: deque[dict] = deque(maxlen=history_size)
        self.dialogue_window_seconds = float(dialogue_window_seconds)

    def classify(
        self,
        *,
        raw_transcript: str,
        detected_language: str | None = None,
        wake_detected: bool = False,
        wake_confidence: float = 0.0,
        command_structure: bool = False,
        current_game_language: str | None = None,
        audio_metadata: dict | None = None,
        now: float | None = None,
    ) -> UtteranceRoleDecision:
        now = float(now if now is not None else time.time())
        raw = str(raw_transcript or "").strip()
        normalized = " ".join(re.findall(r"[a-z0-9']+", raw.casefold()))
        language = str(detected_language or "").lower()
        audio = dict(audio_metadata or {})
        signals: list[str] = []

        if audio.get("loopback") or audio.get("game_audio_bleed") or audio.get("speaker_distance") == "far":
            decision = self._decision(UtteranceRole.GAME_AUDIO_BLEED, 0.96, ["audio_bleed_metadata"])
            return self._remember(decision, language, now)

        if wake_detected and wake_confidence >= 0.75 and command_structure:
            decision = self._decision(UtteranceRole.DIRECT_COMMAND, 0.98, ["wake", "command_structure"])
            return self._remember(decision, language, now)

        recent_dialogue = sum(
            1 for item in self.history
            if now - item["timestamp"] <= self.dialogue_window_seconds
            and item["language"] == "en"
            and item["role"] in {
                UtteranceRole.QUOTED_OR_READ_DIALOGUE.value,
                UtteranceRole.GAME_AUDIO_BLEED.value,
            }
        )
        recent_english_non_owner = sum(
            1 for item in self.history
            if now - item["timestamp"] <= self.dialogue_window_seconds
            and item["language"] == "en"
            and not item.get("owner_framing", False)
        )
        dialogue_syntax = bool(
            re.search(
                r"\b(?:my lord|your highness|castle|kingdom|warrior|commander|we must|"
                r"you shall|you will|i am the|i won't let|our kingdom|the gate|the crystal|"
                r"our people|your majesty|captain|chosen one|ancient power)\b",
                normalized,
            )
            or re.match(r"^(?:[a-z][a-z0-9_'-]{2,16})\s*:", raw.casefold())
            or '"' in raw or "“" in raw or "”" in raw
        )
        narrative_syntax = bool(
            re.search(r"\b(?:he said|she said|they told us|back in|through the|meanwhile|at last|we have come)\b", normalized)
        )
        owner_framing = bool(
            re.search(r"\b(?:i think|i feel|in my opinion|for me|me parece|yo creo|opino|pienso que|a mi me)\b", normalized)
        )
        subtitle_cadence = bool(
            language == "en"
            and 4 <= len(normalized.split()) <= 18
            and recent_english_non_owner >= 2
            and not re.search(r"\b(?:chat|stream|guys|gameplay|i think|for me)\b", normalized)
        )
        if language == "en" and not wake_detected and (
            dialogue_syntax or narrative_syntax or recent_dialogue >= 2 or subtitle_cadence
        ) and not owner_framing:
            if dialogue_syntax:
                signals.append("dialogue_syntax")
            if narrative_syntax:
                signals.append("narrative_syntax")
            if recent_dialogue:
                signals.append("consecutive_english_dialogue")
            if subtitle_cadence:
                signals.append("subtitle_sequence_cadence")
            decision = self._decision(
                UtteranceRole.QUOTED_OR_READ_DIALOGUE,
                min(0.96, 0.76 + recent_dialogue * 0.08),
                signals,
            )
            return self._remember(decision, language, now, owner_framing=False)

        if owner_framing:
            decision = self._decision(UtteranceRole.OWNER_COMMENTARY, 0.9, ["owner_opinion_framing"])
            return self._remember(decision, language, now, owner_framing=True)

        question = raw.endswith("?") or bool(re.match(r"^(?:what|why|how|where|when|que|como|por que|donde)\b", normalized))
        if question and not wake_detected:
            decision = self._decision(UtteranceRole.OWNER_QUESTION_TO_STREAM, 0.72, ["question_without_wake"])
            return self._remember(decision, language, now)

        if len(normalized.split()) <= 3:
            decision = self._decision(UtteranceRole.CONVERSATIONAL_FILLER, 0.76, ["short_fragment"])
            return self._remember(decision, language, now)

        if language in {"es", "en", ""}:
            decision = self._decision(UtteranceRole.OWNER_COMMENTARY, 0.68, ["supported_language", "no_dialogue_signal"])
            return self._remember(decision, language, now)
        decision = self._decision(UtteranceRole.UNCERTAIN, 0.4, ["unsupported_or_uncertain_language"])
        return self._remember(decision, language, now)

    @staticmethod
    def _decision(role: UtteranceRole, confidence: float, signals: list[str]) -> UtteranceRoleDecision:
        dialogue = role in {UtteranceRole.QUOTED_OR_READ_DIALOGUE, UtteranceRole.GAME_AUDIO_BLEED}
        discourse = role in {UtteranceRole.OWNER_COMMENTARY, UtteranceRole.OWNER_QUESTION_TO_STREAM}
        return UtteranceRoleDecision(
            role=role,
            confidence=confidence,
            signals=signals,
            context_allowed=role not in {UtteranceRole.GAME_AUDIO_BLEED, UtteranceRole.CONVERSATIONAL_FILLER, UtteranceRole.UNCERTAIN},
            discourse_allowed=discourse,
            action_allowed=role == UtteranceRole.DIRECT_COMMAND,
        )

    def _remember(
        self, decision: UtteranceRoleDecision, language: str, now: float,
        owner_framing: bool = False,
    ) -> UtteranceRoleDecision:
        self.history.append({
            "role": decision.role.value, "language": language,
            "timestamp": now, "owner_framing": owner_framing,
        })
        print(
            "[HEBE][UTTERANCE_ROLE] "
            f"role={decision.role.value} confidence={decision.confidence:.3f} signals={decision.signals!r} "
            f"context_allowed={str(decision.context_allowed).lower()} "
            f"discourse_allowed={str(decision.discourse_allowed).lower()} "
            f"action_allowed={str(decision.action_allowed).lower()}",
            flush=True,
        )
        return decision
