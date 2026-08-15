from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ConversationContext(str, Enum):
    OWNER_LOCAL = "owner_local"
    OWNER_LIVE_CONTROL = "owner_live_control"
    STREAM_PUBLIC = "stream_public"
    PRIVATE_UI = "private_ui"


class AttentionState(str, Enum):
    ACQUIRED = "ACQUIRED"
    HANDED_OFF = "HANDED_OFF"
    RELEASED = "RELEASED"


class ConversationStatus(str, Enum):
    OPEN = "OPEN"
    WAITING_ON_LEO = "WAITING_ON_LEO"
    WAITING_ON_HEBE = "WAITING_ON_HEBE"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class ExpectedReplyType(str, Enum):
    YES_NO = "yes_no"
    ENTITY_SELECTION = "entity_selection"
    VALUE = "value"
    CORRECTION = "correction"
    FREE_RESPONSE = "free_response"
    DATETIME = "datetime"
    GAME_PROGRESS_STATE = "game_progress_state"
    GAME_PARTY_OR_CHARACTER = "game_party_or_character"
    TWITCH_USERNAME_OR_VIEWER_ALIAS = "twitch_username_or_viewer_alias"
    CASUAL_ANSWER = "casual_answer"
    CLARIFICATION = "clarification"
    ACTION_CONFIRMATION = "action_confirmation"


class ConversationalAct(str, Enum):
    AFFIRM = "AFFIRM"
    DENY = "DENY"
    CANCEL = "CANCEL"
    SELECT = "SELECT"
    CORRECT = "CORRECT"
    FREE_RESPONSE = "FREE_RESPONSE"
    UNKNOWN = "UNKNOWN"


class OpenThreadStatus(str, Enum):
    OPEN = "OPEN"
    WAITING_ON_LEO = "WAITING_ON_LEO"
    WAITING_ON_HEBE = "WAITING_ON_HEBE"
    SNOOZED = "SNOOZED"
    RESOLVED = "RESOLVED"
    EXPIRED = "EXPIRED"
    ARCHIVED = "ARCHIVED"


def normalize_reply_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9_]+", " ", value).strip()


@dataclass(frozen=True, slots=True)
class ExpectedReply:
    type: ExpectedReplyType
    allowed_sources: tuple[str, ...] = ("owner_stt",)
    allowed_participant: str = "leo"
    semantic_constraints: dict[str, Any] = field(default_factory=dict)
    candidate_refs: tuple[str, ...] = ()
    expires_at: float = 0.0
    consume_policy: str = "once"

    def classify(self, text: str) -> tuple[ConversationalAct, dict[str, Any], str]:
        normalized = normalize_reply_text(text)
        if not normalized:
            return ConversationalAct.UNKNOWN, {}, "empty"
        tokens = normalized.split()
        affirmative = {"si", "sip", "sep", "correcto", "exacto", "claro", "dale", "hazlo", "venga"}
        negative = {"no", "nop"}
        cancel = {"dejalo", "nada", "olvidalo"}
        deictic_yes = {"ese", "esa", "ese mismo", "esa misma"}
        deictic_no = {"ese no", "esa no", "el otro", "la otra"}
        phrase = normalized
        if phrase in cancel:
            return ConversationalAct.CANCEL, {}, "deterministic_cancel"
        if self.type == ExpectedReplyType.YES_NO:
            if phrase in affirmative or (
                phrase in deictic_yes and bool(self.semantic_constraints.get("allow_deictic", True))
            ):
                return ConversationalAct.AFFIRM, {"value": True}, "deterministic_affirm"
            if phrase in negative or phrase in deictic_no:
                return ConversationalAct.DENY, {"value": False}, "deterministic_deny"
            return ConversationalAct.UNKNOWN, {}, "not_bounded_yes_no"
        if self.type == ExpectedReplyType.ENTITY_SELECTION:
            ordinals = {"el primero": 0, "la primera": 0, "primero": 0, "primera": 0,
                        "el segundo": 1, "la segunda": 1, "segundo": 1, "segunda": 1}
            if phrase in ordinals:
                index = ordinals[phrase]
                if index < len(self.candidate_refs):
                    return ConversationalAct.SELECT, {"index": index, "candidate": self.candidate_refs[index]}, "ordinal_selection"
            for index, candidate in enumerate(self.candidate_refs):
                if normalize_reply_text(candidate) == phrase:
                    return ConversationalAct.SELECT, {"index": index, "candidate": candidate}, "candidate_selection"
            if phrase in affirmative | deictic_yes and len(self.candidate_refs) == 1:
                return ConversationalAct.SELECT, {"index": 0, "candidate": self.candidate_refs[0]}, "single_candidate_affirm"
            return ConversationalAct.UNKNOWN, {}, "selection_not_resolved"
        if self.type == ExpectedReplyType.VALUE:
            numeric = str(text or "").strip().replace(" ", "")
            match = re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", numeric)
            if match:
                return ConversationalAct.FREE_RESPONSE, {"value": numeric.replace(",", ".")}, "bounded_value"
            return ConversationalAct.UNKNOWN, {}, "value_not_resolved"
        if self.type == ExpectedReplyType.CORRECTION:
            if phrase in cancel:
                return ConversationalAct.CANCEL, {}, "deterministic_cancel"
            if phrase.startswith("no ") or phrase.startswith("no, ") or any(token in tokens for token in ("segundo", "tercero", "primero")):
                return ConversationalAct.CORRECT, {"correction_text": normalized}, "bounded_correction"
            return ConversationalAct.UNKNOWN, {}, "correction_not_resolved"
        if self.type in {
            ExpectedReplyType.FREE_RESPONSE,
            ExpectedReplyType.DATETIME,
            ExpectedReplyType.GAME_PROGRESS_STATE,
            ExpectedReplyType.GAME_PARTY_OR_CHARACTER,
            ExpectedReplyType.TWITCH_USERNAME_OR_VIEWER_ALIAS,
            ExpectedReplyType.CASUAL_ANSWER,
            ExpectedReplyType.CLARIFICATION,
            ExpectedReplyType.ACTION_CONFIRMATION,
        }:
            maximum = int(self.semantic_constraints.get("max_words") or 40)
            minimum = int(self.semantic_constraints.get("min_words") or 1)
            if minimum <= len(tokens) <= maximum:
                return ConversationalAct.FREE_RESPONSE, {"response_text": normalized}, "bounded_free_response"
        return ConversationalAct.UNKNOWN, {}, "incompatible_reply"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["type"] = self.type.value
        value["allowed_sources"] = list(self.allowed_sources)
        value["candidate_refs"] = list(self.candidate_refs)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExpectedReply":
        raw = dict(value or {})
        return cls(
            type=ExpectedReplyType(str(raw.get("type"))),
            allowed_sources=tuple(str(item) for item in raw.get("allowed_sources") or ("owner_stt",)),
            allowed_participant=str(raw.get("allowed_participant") or "leo"),
            semantic_constraints=dict(raw.get("semantic_constraints") or {}),
            candidate_refs=tuple(str(item) for item in raw.get("candidate_refs") or ()),
            expires_at=float(raw.get("expires_at") or 0.0),
            consume_policy=str(raw.get("consume_policy") or "once"),
        )


@dataclass(frozen=True, slots=True)
class CurrentConversation:
    id: str
    context_kind: ConversationContext
    context_id: str
    participants: tuple[str, ...]
    attention_state: AttentionState
    turn_owner: str
    expected_reply: ExpectedReply | None
    topic: str
    origin_event_id: str
    last_event_id: str
    opened_at: float
    last_turn_at: float
    expires_at: float
    status: ConversationStatus
    closure_reason: str = ""
    version: int = 1
    domain_payload: dict[str, Any] = field(default_factory=dict)
    consumed_event_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["context_kind"] = self.context_kind.value
        value["attention_state"] = self.attention_state.value
        value["status"] = self.status.value
        value["participants"] = list(self.participants)
        value["consumed_event_ids"] = list(self.consumed_event_ids)
        value["expected_reply"] = self.expected_reply.to_dict() if self.expected_reply else None
        return value


@dataclass(frozen=True, slots=True)
class OpenThread:
    id: str
    thread_type: str
    scope_kind: str
    scope_id: str
    participant_ids: tuple[str, ...]
    subject_ref: str
    summary: str
    origin_event_id: str
    latest_event_id: str
    status: OpenThreadStatus
    priority: int
    created_at: float
    relevance_until: float
    valid_until: float
    resolved_at: float = 0.0
    resolution_event_id: str = ""
    sensitivity: str = "normal"
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["participant_ids"] = list(self.participant_ids)
        value["status"] = self.status.value
        return value
