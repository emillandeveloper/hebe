from __future__ import annotations

from typing import Any

from app.continuity.models import ConversationContext, ConversationStatus, ExpectedReply, ExpectedReplyType
from app.continuity.service import ConversationContinuityService


class LegacyPendingAdapter:
    """The only compatibility seam between legacy pending dicts and v2."""

    def __init__(self, service: ConversationContinuityService, state: Any) -> None:
        self.service = service
        self.state = state
        self.last_projection: dict[str, Any] = {}

    def project_legacy_pending(
        self, pending: dict[str, Any], *, context_kind: ConversationContext,
        context_id: str, event_id: str,
    ):
        kind = str(pending.get("kind") or "legacy_clarification")
        candidates = tuple(str(item) for item in pending.get("candidates") or () if str(item).strip())
        legacy_type = str(pending.get("expected_reply_type") or "free_response")
        if kind == "promotion_target_clarification":
            expected_type = ExpectedReplyType.YES_NO if len(candidates) == 1 else ExpectedReplyType.ENTITY_SELECTION
        else:
            expected_type = {
                "yes_no": ExpectedReplyType.YES_NO,
                "entity_selection": ExpectedReplyType.ENTITY_SELECTION,
                "value": ExpectedReplyType.VALUE,
                "correction": ExpectedReplyType.CORRECTION,
                "free_response": ExpectedReplyType.FREE_RESPONSE,
                "casual_answer": ExpectedReplyType.FREE_RESPONSE,
            }.get(legacy_type, ExpectedReplyType.FREE_RESPONSE)
        expected = ExpectedReply(
            type=expected_type,
            allowed_sources=("owner_stt",) if context_kind != ConversationContext.PRIVATE_UI else ("owner_ui",),
            allowed_participant="leo",
            semantic_constraints={"allow_deictic": True, "min_words": 1, "max_words": 40},
            candidate_refs=candidates,
            expires_at=float(pending.get("expires_at") or 0.0),
        )
        conversation = self.service.open_conversation(
            context_kind=context_kind, context_id=context_id, topic=kind,
            origin_event_id=event_id or str(pending.get("opened_by_event_id") or pending.get("id") or "legacy"),
            expected_reply=expected,
            domain_payload={"legacy_pending_id": pending.get("id"), "legacy_kind": kind, "pending": dict(pending)},
            reason="legacy_pending_shadow_projection",
        )
        pending["conversation_id"] = conversation.id
        self.state.pending_clarification = pending
        self.last_projection = {"direction": "legacy_to_v2", "pending_id": pending.get("id"), "conversation_id": conversation.id}
        return conversation

    def close_for_legacy(self, pending: dict[str, Any] | None, *, reason: str, event_id: str = "") -> None:
        if not isinstance(pending, dict):
            return
        conversation_id = str(pending.get("conversation_id") or "")
        if not conversation_id:
            return
        conversation = self.service.conversations.get(conversation_id)
        if conversation is None or conversation.status not in {
            ConversationStatus.OPEN, ConversationStatus.WAITING_ON_LEO, ConversationStatus.WAITING_ON_HEBE
        }:
            return
        status = {
            "expired": ConversationStatus.EXPIRED,
            "consumed": ConversationStatus.CLOSED,
            "superseded_by_fresh_promotion": ConversationStatus.INTERRUPTED,
            "new_owner_command_interrupted": ConversationStatus.INTERRUPTED,
        }.get(reason, ConversationStatus.CANCELLED)
        self.service.conversations.transition(
            conversation.id, expected_version=conversation.version, status=status, reason=reason,
            last_event_id=event_id or str(pending.get("id") or "legacy"), now=self.service.now_fn(),
        )
        self.last_projection = {"direction": "legacy_close", "pending_id": pending.get("id"), "conversation_id": conversation.id, "reason": reason}
