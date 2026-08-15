from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.cognitive.input_event import InputEvent


@dataclass(frozen=True)
class InputClassification:
    source: str
    input_type: str
    purpose: str = ""
    addressed_to_hebe: bool = False
    confidence: float = 0.0
    reason: str = ""
    voice_event_type: str = ""
    has_action_intent: bool = False


@dataclass(frozen=True)
class ConversationState:
    active: bool = False
    topic: str = ""
    source: str = ""
    last_direct_user_input: str = ""
    last_assistant_reply: str = ""
    expected_reply_type: str = ""
    expires_at: float = 0.0
    allow_no_wakeword: bool = False
    output_target: list[str] = field(default_factory=list)
    confidence: float = 0.0
    matched: bool = False
    reason: str = "no_active_conversation"


@dataclass(frozen=True)
class ContextRelevance:
    useful: bool = False
    category: str = "none"
    confidence: float = 0.0
    reason: str = ""
    facts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class KnowledgeResolution:
    game: str = ""
    current_session_known: bool = False
    profile_found: bool = False
    lookup_used: bool = False
    confidence: str = "unknown"
    provenance: str = "unknown"
    unresolved_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResponseDecision:
    should_reply: bool
    reason: str
    output_target: list[str] = field(default_factory=list)
    allow_question: bool = False
    max_questions: int = 0
    max_sentences: int = 1


@dataclass(frozen=True)
class ResponseFrame:
    input_type: str
    source: str
    current_game: str = ""
    current_session_context: dict[str, Any] = field(default_factory=dict)
    conversation_state: ConversationState = field(default_factory=ConversationState)
    intent: str = ""
    action_plan: dict[str, Any] | None = None
    output_target: list[str] = field(default_factory=list)
    allow_question: bool = False
    max_questions: int = 0
    max_sentences: int = 1
    tone: str = "stream_companion"
    should_reply: bool = False
    forbidden_patterns: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_type": self.input_type,
            "source": self.source,
            "current_game": self.current_game,
            "current_session_context": self.current_session_context,
            "conversation_state": self.conversation_state.__dict__,
            "intent": self.intent,
            "action_plan": self.action_plan,
            "output_target": list(self.output_target),
            "allow_question": self.allow_question,
            "max_questions": self.max_questions,
            "max_sentences": self.max_sentences,
            "tone": self.tone,
            "should_reply": self.should_reply,
            "forbidden_patterns": list(self.forbidden_patterns),
        }


class InputClassifier:
    def classify(
        self,
        event: InputEvent,
        *,
        voice_event_type: str = "",
        addressed_to_hebe: bool = False,
        has_action_intent: bool = False,
        pending_followup: bool = False,
        valid: bool = True,
    ) -> InputClassification:
        envelope = getattr(event, "envelope", None)
        if envelope is not None:
            return InputClassification(
                source=envelope.source,
                input_type=envelope.input_type,
                purpose="pending_resolution" if envelope.pending_compatible else "local_command" if envelope.app_target else "",
                addressed_to_hebe=envelope.addressed_to_hebe,
                confidence=float((envelope.app_plan_result or {}).get("confidence") or (
                    .95 if envelope.pending_compatible else .55 if envelope.source == "ambient_stt" else .9
                )),
                reason=envelope.reason,
                voice_event_type=voice_event_type,
                has_action_intent=has_action_intent,
            )
        source = _canonical_source(event.source)
        if not valid:
            return InputClassification(source=source, input_type="noise/rejected", reason="failed_validity_gate")
        if source == "twitch_chat":
            return InputClassification(
                source=source,
                input_type="twitch_chat_mention" if addressed_to_hebe else "twitch_chat_observed",
                addressed_to_hebe=addressed_to_hebe,
                confidence=0.9 if addressed_to_hebe else 0.65,
                reason="chat_mention" if addressed_to_hebe else "observe_only",
            )
        if source in {"twitch_event", "scheduler/spontaneity", "system/tool_result"}:
            return InputClassification(source=source, input_type="system_event", confidence=1.0, reason=source)
        if pending_followup:
            return InputClassification(
                source=source,
                input_type="active_conversation_followup",
                confidence=0.86,
                reason="active_conversation_state",
                voice_event_type=voice_event_type,
                has_action_intent=has_action_intent,
            )
        if has_action_intent:
            return InputClassification(
                source=source,
                input_type="explicit_command",
                addressed_to_hebe=addressed_to_hebe,
                confidence=0.86,
                reason="structured_action_intent",
                voice_event_type=voice_event_type,
                has_action_intent=True,
            )
        if addressed_to_hebe:
            input_type = "explicit_question" if _looks_question_like(event.raw_text, event.normalized_text) else "direct_to_hebe"
            return InputClassification(
                source=source,
                input_type=input_type,
                addressed_to_hebe=True,
                confidence=0.9,
                reason="addressed_to_hebe",
                voice_event_type=voice_event_type,
            )
        if source == "stt_voice":
            return InputClassification(
                source=source,
                input_type="ambient_stream_context",
                confidence=0.55,
                reason=voice_event_type or "not_addressed",
                voice_event_type=voice_event_type,
            )
        if source == "ui_text":
            return InputClassification(source=source, input_type="direct_to_hebe", confidence=0.8, reason="typed_ui")
        return InputClassification(source=source, input_type="ambient_stream_context", confidence=0.4, reason="default_observe")


class ConversationStateResolver:
    def from_pending_turn(self, pending_turn: dict | None, *, matched: bool = False, reason: str = "") -> ConversationState:
        if not isinstance(pending_turn, dict) or pending_turn.get("status") != "pending":
            return ConversationState(active=False, matched=False, reason=reason or "no_active_conversation")
        return ConversationState(
            active=True,
            topic=str(pending_turn.get("topic") or pending_turn.get("expected_type") or ""),
            source=str(pending_turn.get("source") or ""),
            last_direct_user_input=str(pending_turn.get("last_direct_user_input") or ""),
            last_assistant_reply=str(pending_turn.get("previous_assistant_message") or ""),
            expected_reply_type=str(pending_turn.get("expected_type") or ""),
            expires_at=float(pending_turn.get("expires_at", 0.0) or 0.0),
            allow_no_wakeword=bool(pending_turn.get("allow_without_wakeword", False)),
            output_target=list(pending_turn.get("output_target") or []),
            confidence=float(pending_turn.get("confidence", 0.76) or 0.76),
            matched=matched,
            reason=reason or ("matched" if matched else "active_not_matched"),
        )


class KnowledgePolicyResolver:
    def resolve(self, *, stream: Any | None, profile_store: Any | None = None) -> KnowledgeResolution:
        game = ""
        if stream is not None:
            game = str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or "").strip()
        profile_found = False
        if profile_store is not None and game:
            try:
                profile_found = bool(profile_store.has_specific_profile(current_category=game, current_game=game, current_title=game))
            except Exception:
                profile_found = False
        current_session_known = bool(
            stream is not None
            and (
                getattr(stream, "current_run_objective", None)
                or getattr(stream, "current_run_location", None)
                or getattr(stream, "recent_run_context_facts", None)
            )
        )
        if current_session_known:
            confidence = "confirmed_by_leo"
            provenance = "leo_stt"
        elif profile_found:
            confidence = "known_from_game_profile"
            provenance = "game_profile"
        elif game:
            confidence = "inferred_from_context"
            provenance = "twitch_title"
        else:
            confidence = "unknown"
            provenance = "unknown"
        return KnowledgeResolution(
            game=game,
            current_session_known=current_session_known,
            profile_found=profile_found,
            lookup_used=False,
            confidence=confidence,
            provenance=provenance,
        )


class ResponseDecisionResolver:
    def decide(
        self,
        *,
        classification: InputClassification,
        conversation_state: ConversationState,
        relevance: ContextRelevance | None,
        output_targets: list[str],
    ) -> ResponseDecision:
        input_type = classification.input_type
        if input_type == "noise/rejected":
            return ResponseDecision(False, "rejected_noise", ["none"])
        if input_type == "ambient_stream_context":
            if relevance and relevance.useful:
                return ResponseDecision(False, "no_context_only", output_targets)
            return ResponseDecision(False, "no_ignore", output_targets)
        if input_type == "twitch_chat_observed":
            return ResponseDecision(False, "no_ignore", output_targets)
        if input_type == "twitch_chat_mention":
            return ResponseDecision(True, "yes_twitch_mention", output_targets, max_sentences=1)
        if input_type == "active_conversation_followup" and conversation_state.active and conversation_state.matched:
            return ResponseDecision(True, "yes_contextual_followup", output_targets, max_sentences=1)
        if input_type in {"direct_to_hebe", "explicit_command", "explicit_question"}:
            allow_question = input_type in {"explicit_command", "explicit_question"} and classification.has_action_intent
            return ResponseDecision(
                True,
                "direct_command" if input_type == "explicit_command" else "direct_question",
                output_targets,
                allow_question=allow_question,
                max_questions=1 if allow_question else 0,
                max_sentences=1,
            )
        if input_type == "system_event":
            return ResponseDecision(True, "yes_spontaneous", output_targets, max_sentences=1)
        return ResponseDecision(False, "no_ignore", ["none"])


def _canonical_source(source: str) -> str:
    value = str(source or "").strip()
    return {
        "ui": "ui_text",
        "typed_ui": "ui_text",
        "voice": "stt_voice",
        "stt_voice": "stt_voice",
        "twitch_irc": "twitch_chat",
        "twitch_chat": "twitch_chat",
    }.get(value, value or "system/tool_result")


def _looks_question_like(raw_text: str, normalized_text: str) -> bool:
    text = f"{raw_text or ''} {normalized_text or ''}".casefold()
    if "?" in text or "¿" in text:
        return True
    first = (normalized_text or "").split(" ", 1)[0] if normalized_text else ""
    return first in {"que", "qué", "como", "cómo", "cuando", "cuándo", "donde", "dónde", "cual", "cuál"}
