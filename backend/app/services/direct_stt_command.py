from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import re
import unicodedata
import uuid


class DirectUtteranceIntentFamily(StrEnum):
    APPLICATION_ACTION = "application_action"
    STREAM_OPERATION = "stream_operation"
    SYSTEM_COMMAND = "system_command"
    DIRECT_QUESTION = "direct_question"
    CASUAL_CONVERSATION = "casual_conversation"
    INCOMPLETE_COMMAND = "incomplete_command"
    UNCERTAIN = "uncertain"


@dataclass(slots=True)
class DirectSTTCommandResult:
    event_id: str = field(default_factory=lambda: f"stt_{uuid.uuid4().hex}")
    ambient_text: str = ""
    command_text: str = ""
    normalized_command_text: str = ""
    wake_detected: bool = False
    wake_confidence: float = 0.0
    agreement_score: float = 0.0
    detected_intent_family: str = DirectUtteranceIntentFamily.UNCERTAIN.value
    action_verb: str = ""
    raw_target: str = ""
    target_candidates: list[str] = field(default_factory=list)
    final_outcome: str = ""
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict | None) -> "DirectSTTCommandResult":
        allowed = cls.__dataclass_fields__
        return cls(**{key: item for key, item in dict(value or {}).items() if key in allowed})


_WAKE_PREFIX = re.compile(
    r"^\s*(?P<wake>hebe|ebe|eve|e\s*[.\-]?\s*b\s*\.?|e\s*[.\-]?\s*v\s*\.?)"
    r"(?=\s|[,;:!?]|\b)\s*[,;:!?\-]*\s*",
    re.IGNORECASE,
)
_FILLER_PREFIX = re.compile(
    r"^(?:(?:oye|mira|vale|ok(?:ay)?|por\s+favor|puedes|podrias|podrías)\b[\s,;:]*)+",
    re.IGNORECASE,
)
_OPEN_VERB = re.compile(
    r"\b(?P<verb>abre|abrir|inicia|iniciar|lanza|lanzar|ejecuta|ejecutar|pon|open|start|launch|run)\b",
    re.IGNORECASE,
)
_STREAM_TERMS = {
    "stream", "directo", "twitch", "chat", "promo", "promocion", "shoutout", "raid",
}
_SYSTEM_TERMS = {"duerme", "descansa", "despierta", "silencio", "callate", "cállate"}
_QUESTION_PREFIXES = {
    "que", "qué", "como", "cómo", "cuando", "cuándo", "donde", "dónde", "quien",
    "quién", "estas", "estás", "eres", "puedes", "sabes", "tienes", "hay",
}


def normalize_direct_command_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).strip()
    value = re.sub(r"\s+", " ", value)
    return value.casefold()


def _clean_target(value: str) -> str:
    target = re.sub(r"^[\s,;:!?\-]+|[\s,;:!?\-\.]+$", "", str(value or "")).strip()
    dotted = re.findall(r"[A-Za-z]", target)
    non_letters = re.sub(r"[A-Za-z.\s]", "", target)
    pieces = re.findall(r"[A-Za-z]+", target)
    if not non_letters and len(pieces) >= 2 and all(len(piece) == 1 for piece in pieces):
        return "".join(dotted).upper()
    return re.sub(r"\s+", " ", target)


def parse_direct_stt_command(
    command_text: str,
    *,
    ambient_text: str = "",
    agreement_score: float = 0.0,
    event_id: str | None = None,
) -> DirectSTTCommandResult:
    exact = str(command_text or "").strip()
    normalized = normalize_direct_command_text(exact)
    wake_match = _WAKE_PREFIX.match(exact)
    addressed = bool(wake_match)
    body = exact[wake_match.end():] if wake_match else exact
    body = _FILLER_PREFIX.sub("", body).strip()
    verb_match = _OPEN_VERB.search(body)
    action_verb = "open" if verb_match else ""
    raw_target = _clean_target(body[verb_match.end():]) if verb_match else ""
    body_normalized = normalize_direct_command_text(body)
    tokens = set(re.findall(r"[a-z0-9áéíóúüñ]+", body_normalized))
    first_word = next(iter(re.findall(r"[a-záéíóúüñ]+", body_normalized)), "")
    is_direct_question = bool(addressed and ("?" in exact or first_word in _QUESTION_PREFIXES))

    recognized_target = False
    if addressed and not verb_match and body and not is_direct_question:
        try:
            from app.services.app_registry import resolve_whitelisted_app
            recognized_target = resolve_whitelisted_app(_clean_target(body)) is not None
        except Exception:
            recognized_target = False
        if recognized_target:
            raw_target = _clean_target(body)

    if verb_match and raw_target:
        family = DirectUtteranceIntentFamily.APPLICATION_ACTION
    elif verb_match or recognized_target:
        family = DirectUtteranceIntentFamily.INCOMPLETE_COMMAND
    elif tokens & _STREAM_TERMS and tokens & {"promo", "promocion", "shoutout", "raid", "stream", "directo"}:
        family = DirectUtteranceIntentFamily.STREAM_OPERATION
    elif tokens & _SYSTEM_TERMS:
        family = DirectUtteranceIntentFamily.SYSTEM_COMMAND
    elif is_direct_question:
        family = DirectUtteranceIntentFamily.DIRECT_QUESTION
    elif addressed and body_normalized:
        family = DirectUtteranceIntentFamily.CASUAL_CONVERSATION
    elif addressed:
        family = DirectUtteranceIntentFamily.INCOMPLETE_COMMAND
    else:
        family = DirectUtteranceIntentFamily.UNCERTAIN

    return DirectSTTCommandResult(
        event_id=event_id or f"stt_{uuid.uuid4().hex}",
        ambient_text=str(ambient_text or ""),
        command_text=exact,
        normalized_command_text=normalized,
        wake_detected=addressed,
        wake_confidence=1.0 if addressed else 0.0,
        agreement_score=float(agreement_score or 0.0),
        detected_intent_family=family.value,
        action_verb=action_verb,
        raw_target=raw_target,
        target_candidates=[raw_target] if raw_target else [],
    )
