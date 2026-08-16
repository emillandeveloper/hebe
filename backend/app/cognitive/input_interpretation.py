from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import re
import unicodedata
from typing import Any

from app.core.persistent_logs import log_jsonl_event
from app.services.direct_stt_command import (
    DirectSTTCommandResult,
    DirectUtteranceIntentFamily,
    parse_direct_stt_command,
)


class InputSpeechAct(StrEnum):
    OWNER_COMMAND = "owner_command"
    OWNER_FEEDBACK = "owner_feedback"
    OWNER_COMMENTARY = "owner_commentary"
    OWNER_ANSWER_FOLLOWUP = "owner_answer_followup"
    VIEWER_DIRECTED_TO_HEBE = "viewer_directed_to_hebe"
    VIEWER_CONTEXT = "viewer_context"
    AMBIENT_CONTEXT = "ambient_context"
    SYSTEM_EVENT = "system_event"


@dataclass(frozen=True, slots=True)
class FeedbackSignal:
    target: str
    polarity: str
    strength: float
    explicitness: str
    referent: str = ""
    meta_about_hebe: bool = True


@dataclass(frozen=True, slots=True)
class InputInterpretation:
    """Canonical communicative meaning consumed by routing and domain gates."""

    speech_act: InputSpeechAct
    source: str
    authority: str
    addressed_to_hebe: bool
    confidence: float
    possible_command_syntax: bool
    authorized_action_command: bool
    context_eligible: bool
    feedback: FeedbackSignal | None = None
    feedback_text: str = ""
    context_text: str = ""
    excluded_context_spans: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""

    @property
    def action_eligible(self) -> bool:
        return self.authorized_action_command

    @property
    def meta_about_hebe(self) -> bool:
        return bool(self.feedback and self.feedback.meta_about_hebe)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["speech_act"] = self.speech_act.value
        value["excluded_context_spans"] = list(self.excluded_context_spans)
        return value


_OWNER_SOURCES = {
    "ui", "typed_ui", "owner_ui", "voice", "stt_voice", "button",
    "owner_stt_direct", "owner_stt_command", "owner_stt_followup",
}
_VIEWER_SOURCES = {"twitch", "twitch_chat", "twitch_viewer"}
_AMBIENT_SOURCES = {"ambient", "ambient_stt", "ambient_context"}
_SYSTEM_SOURCES = {
    "system", "twitch_system", "twitch_event", "scheduler/spontaneity",
    "system/tool_result", "scheduler", "internal_event",
}

_NEGATIVE_FEEDBACK = (
    r"\bdeja(?:\s+ya)?\s+de\b",
    r"\bno\s+me\s+vuelvas\s+a\b",
    r"\bno\s+hace\s+falta\s+que\s+me\b",
    r"\bte\s+quieres\s+callar\b",
    r"\botra\s+vez\s+con\s+(?:lo\s+de(?:l)?|la|el)\b",
    r"\b(?:todo\s+el\s+rato|todo\s+el\s+(?:puto\s+)?stream)\b.*\b(?:diciendome|recordandome|recomendandome)\b",
    r"\bno\s+vuelvas\s+a\b",
    r"\bno\s+quiero\s+que\s+vuelvas\s+a\b",
    r"\bdeja\s+(?:ese|esa)\s+(?:tema|broma)\b",
)
_POSITIVE_FEEDBACK = (
    r"\beso\s+si\s+me\s+ha\s+hecho\s+gracia\b",
    r"\b(?:esa|esta|tu)\s+respuesta\s+(?:estuvo|ha\s+estado|esta)\s+bien\b",
    r"\bme\s+ha\s+gustado\s+(?:eso|esa|tu\s+respuesta)\b",
    r"\b(?:puedes|puede)\s+(?:volver|seguir)\b",
    r"\bya\s+puedes\b.*\b(?:seguir|volver|hacer)\b",
    r"\b(?:esa|eso)\s+vuelve\s+a\s+tener\s+gracia\b",
)
_CONTEXT_CLAUSE = re.compile(
    r"\b(?:\d+\s*hp|hp|vida|mana|jefe|boss|enemig[oa]|combate|"
    r"me\s+ha(?:n)?\s+matado|he\s+muerto|game\s+over|wipe)\b"
)
_GENERIC_COMMAND_PREFIX = re.compile(
    r"^(?:(?:hebe|ebe|eve|jebe|heve)\s*[,;:!?-]*\s*)?"
    r"(?:(?:oye|mira|vale|ok(?:ay)?|por\s+favor|puedes|podrias)\s+)*"
    r"(?:abre|abrir|inicia|iniciar|arranca|arrancar|lanza|lanzar|ejecuta|ejecutar|"
    r"pon|ponme|cierra|cerrar|haz|hazle|dale|manda|envia|dile|di|activa|desactiva|"
    r"pausa|reanuda|cambia|quita|deja|recuerdame|avisame|agenda|agendame|apuntame|apaontame|"
    r"cancela|descarta|anula|despierta|duerme|calla|callate|silencio|"
    r"open|start|launch|run|close|send|enable|disable|stop|pause|resume|cancel)\b"
)
_TEMPORAL_REMINDER_COMMAND = re.compile(
    r"^(?:(?:dentro\s+de|en)\s+(?:\d+|un[ao]?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s+"
    r"(?:minut[oa]s?|horas?|dias?)\s+)(?:recuerdame|avisame)\b"
)


class InputInterpreter:
    """Single owner for primary input meaning and semantic scope."""

    def interpret_event(
        self,
        event: Any,
        *,
        authority: str | None = None,
        addressed_to_hebe: bool | None = None,
        explicit_command_mode: bool | None = None,
        pending_valid: bool = False,
        recent_hebe_utterance: str = "",
        direct_result: DirectSTTCommandResult | None = None,
    ) -> InputInterpretation:
        existing = getattr(event, "interpretation", None)
        if isinstance(existing, InputInterpretation) and not pending_valid:
            return existing
        metadata = dict(getattr(event, "stt_metadata", {}) or {})
        direct = direct_result or DirectSTTCommandResult.from_dict(metadata.get("direct_stt_command"))
        if not direct.command_text:
            direct = parse_direct_stt_command(
                getattr(event, "raw_text", "") or getattr(event, "normalized_text", "")
            )
        source = str(getattr(event, "source", "") or "ambient")
        resolved_authority = authority or self.authority_for_source(source)
        addressed = (
            bool(addressed_to_hebe)
            if addressed_to_hebe is not None
            else bool(direct.wake_detected or source in {"ui", "typed_ui", "owner_ui", "button"})
        )
        command_mode = (
            bool(explicit_command_mode)
            if explicit_command_mode is not None
            else bool(metadata.get("command_mode", source in {"ui", "typed_ui", "owner_ui", "button"}))
        )
        result = self.interpret(
            raw_text=getattr(event, "raw_text", "") or getattr(event, "normalized_text", ""),
            source=source,
            authority=resolved_authority,
            addressed_to_hebe=addressed,
            explicit_command_mode=command_mode,
            pending_valid=pending_valid,
            recent_hebe_utterance=recent_hebe_utterance,
            direct_result=direct,
        )
        event.interpretation = result
        return result

    def interpret(
        self,
        *,
        raw_text: str,
        source: str,
        authority: str,
        addressed_to_hebe: bool,
        explicit_command_mode: bool = False,
        pending_valid: bool = False,
        recent_hebe_utterance: str = "",
        direct_result: DirectSTTCommandResult | None = None,
    ) -> InputInterpretation:
        raw = str(raw_text or "").strip()
        direct = direct_result or parse_direct_stt_command(raw)
        interpreted_text = str(direct.command_text or raw).strip()
        normalized = self.normalize(interpreted_text)
        possible_command = bool(
            direct.detected_intent_family in {
                DirectUtteranceIntentFamily.APPLICATION_ACTION.value,
            }
            or (
                direct.detected_intent_family == DirectUtteranceIntentFamily.INCOMPLETE_COMMAND.value
                and direct.wake_detected
            )
            or _GENERIC_COMMAND_PREFIX.match(normalized)
            or _TEMPORAL_REMINDER_COMMAND.match(normalized)
        )

        silence_control = bool(re.match(
            r"^(?:(?:hebe|ebe|eve|jebe|heve)\s+)?(?:deja(?: ya)? de hablar|"
            r"no hables sola|calla|callate|silencio|quieta)\s*$",
            normalized,
        ))
        feedback = None if silence_control else self._feedback_signal(
            normalized,
            recent_hebe_utterance=recent_hebe_utterance,
        )
        if feedback is not None and authority == "owner":
            feedback_text, context_text = self._semantic_scopes(interpreted_text)
            result = InputInterpretation(
                speech_act=InputSpeechAct.OWNER_FEEDBACK,
                source=source,
                authority=authority,
                addressed_to_hebe=bool(addressed_to_hebe or self._feedback_is_directed(normalized)),
                confidence=0.97 if feedback.polarity == "negative" else 0.9,
                possible_command_syntax=possible_command,
                authorized_action_command=False,
                context_eligible=bool(context_text),
                feedback=feedback,
                feedback_text=feedback_text,
                context_text=context_text,
                excluded_context_spans=(feedback_text,) if feedback_text else (),
                reason="explicit_owner_feedback",
            )
            return self._log(result)

        if authority == "system" or source in _SYSTEM_SOURCES:
            return self._log(InputInterpretation(
                InputSpeechAct.SYSTEM_EVENT, source, "system", addressed_to_hebe,
                1.0, possible_command, False, False, reason="system_source",
            ))
        if authority == "viewer" or source in _VIEWER_SOURCES:
            act = (
                InputSpeechAct.VIEWER_DIRECTED_TO_HEBE
                if addressed_to_hebe or self._mentions_hebe(normalized)
                else InputSpeechAct.VIEWER_CONTEXT
            )
            return self._log(InputInterpretation(
                act, source, "viewer", act == InputSpeechAct.VIEWER_DIRECTED_TO_HEBE,
                0.94 if act == InputSpeechAct.VIEWER_DIRECTED_TO_HEBE else 0.84,
                possible_command, False, act == InputSpeechAct.VIEWER_CONTEXT,
                context_text=interpreted_text if act == InputSpeechAct.VIEWER_CONTEXT else "",
                reason="viewer_addressing" if act == InputSpeechAct.VIEWER_DIRECTED_TO_HEBE else "viewer_context",
            ))
        if authority == "ambient" or source in _AMBIENT_SOURCES:
            return self._log(InputInterpretation(
                InputSpeechAct.AMBIENT_CONTEXT, source, "ambient", False, 0.9,
                possible_command, False, True, context_text=interpreted_text, reason="ambient_source",
            ))
        if pending_valid and authority == "owner":
            return self._log(InputInterpretation(
                InputSpeechAct.OWNER_ANSWER_FOLLOWUP, source, authority, addressed_to_hebe,
                0.96, possible_command, False, False, reason="valid_pending_followup",
            ))

        authorized = bool(
            authority == "owner"
            and possible_command
            and (
                addressed_to_hebe
                or explicit_command_mode
                or source in {
                    "ui", "typed_ui", "owner_ui", "button", "stt_voice", "voice",
                    "owner_stt_direct", "owner_stt_command",
                }
            )
        )
        if authorized:
            return self._log(InputInterpretation(
                InputSpeechAct.OWNER_COMMAND, source, authority, addressed_to_hebe,
                0.98 if addressed_to_hebe else 0.88, possible_command, True, False,
                reason="authorized_owner_command",
            ))
        return self._log(InputInterpretation(
            InputSpeechAct.OWNER_COMMENTARY, source, authority, addressed_to_hebe,
            0.78, possible_command, False, True, context_text=interpreted_text,
            reason="owner_non_action_input",
        ))

    @staticmethod
    def authority_for_source(source: str) -> str:
        source = str(source or "").lower()
        if source in _OWNER_SOURCES:
            return "owner"
        if source in _VIEWER_SOURCES:
            return "viewer"
        if source in _AMBIENT_SOURCES:
            return "ambient"
        return "system"

    @staticmethod
    def normalize(text: str) -> str:
        value = str(text or "").casefold()
        value = "".join(
            character for character in unicodedata.normalize("NFKD", value)
            if not unicodedata.combining(character)
        )
        value = re.sub(r"[^a-z0-9\s;:'\"-]", " ", value)
        return " ".join(value.split())

    def _feedback_signal(self, normalized: str, *, recent_hebe_utterance: str) -> FeedbackSignal | None:
        negative = any(re.search(pattern, normalized) for pattern in _NEGATIVE_FEEDBACK)
        positive = any(re.search(pattern, normalized) for pattern in _POSITIVE_FEEDBACK)
        contextual_positive = bool(
            recent_hebe_utterance.strip()
            and re.search(r"\bbuenisima\s+esa\b", normalized)
        )
        if not (negative or positive or contextual_positive):
            return None
        polarity = "negative" if negative else "positive"
        repeated = bool(re.search(r"\b(?:otra vez|repetir|insistir|todo el (?:puto )?(?:rato|stream))\b", normalized))
        target = (
            "repeated_hebe_behavior" if repeated
            else "hebe_response" if positive or contextual_positive
            else "hebe_behavior"
        )
        referent = self._extract_referent(normalized)
        if contextual_positive and not referent:
            referent = "previous_hebe_utterance"
        strong_language = bool(re.search(r"\b(?:put[oa]|joder|mierda)\b", normalized))
        strength = 0.98 if strong_language else 0.9 if negative else 0.78
        return FeedbackSignal(
            target=target,
            polarity=polarity,
            strength=strength,
            explicitness="explicit" if not contextual_positive else "contextual",
            referent=referent,
        )

    @staticmethod
    def _extract_referent(normalized: str) -> str:
        patterns = (
            r"\botra vez con lo de(?:l)?\s+(.+?)(?:[;.]|$)",
            r"\bdeja(?: ya)? de\s+(.+?)(?:[;.]|$)",
            r"\bno me vuelvas a\s+(.+?)(?:[;.]|$)",
            r"\bno vuelvas a\s+(.+?)(?:[;.]|$)",
            r"\bno quiero que vuelvas a\s+(.+?)(?:[;.]|$)",
            r"\bdeja\s+(?:ese|esa)\s+(?:tema|broma)\s*(.*?)(?:[;.]|$)",
            r"\b(?:puedes|puede)\s+(?:volver|seguir)(?:\s+a)?\s*(.*?)(?:[;.]|$)",
            r"\b(?:esa|esta|tu) respuesta\b",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            if match.lastindex:
                value = match.group(1).strip(" ,:-")
                return value[:160]
            return "response"
        return ""

    @staticmethod
    def _semantic_scopes(raw: str) -> tuple[str, str]:
        clauses = [part.strip() for part in re.split(r"\s*;\s*", raw) if part.strip()]
        if len(clauses) <= 1:
            return raw, ""
        context_clauses = [part for part in clauses[1:] if _CONTEXT_CLAUSE.search(InputInterpreter.normalize(part))]
        if not context_clauses:
            return raw, ""
        context_text = "; ".join(context_clauses)
        feedback_text = clauses[0]
        return feedback_text, context_text

    @staticmethod
    def _feedback_is_directed(normalized: str) -> bool:
        return bool(
            InputInterpreter._mentions_hebe(normalized)
            or re.search(r"\b(?:te quieres callar|deja(?: ya)? de|no me vuelvas a|no hace falta que me)\b", normalized)
        )

    @staticmethod
    def _mentions_hebe(normalized: str) -> bool:
        return bool(re.search(r"\b(?:hebe|ebe|eve|jebe|heve)\b", normalized))

    @staticmethod
    def _log(result: InputInterpretation) -> InputInterpretation:
        feedback = result.feedback
        payload = {
            "source": result.source,
            "authority": result.authority,
            "speech_act": result.speech_act.value,
            "addressed_to_hebe": result.addressed_to_hebe,
            "command_eligible": result.authorized_action_command,
            "possible_command_syntax": result.possible_command_syntax,
            "context_eligible": result.context_eligible,
            "feedback_polarity": feedback.polarity if feedback else "",
            "feedback_target": feedback.target if feedback else "",
            "feedback_referent": feedback.referent if feedback else "",
            "confidence": result.confidence,
        }
        print(
            "[HEBE][INPUT_INTERPRETATION] "
            + " ".join(f"{key}={value!r}" for key, value in payload.items()),
            flush=True,
        )
        log_jsonl_event("input_interpretation", payload)
        return result
