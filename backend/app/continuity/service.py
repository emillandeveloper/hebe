from __future__ import annotations

import statistics
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from app.continuity.models import (
    AttentionState,
    ConversationContext,
    ConversationStatus,
    ConversationalAct,
    CurrentConversation,
    ExpectedReply,
    ExpectedReplyType,
    OpenThread,
    OpenThreadStatus,
)
from app.continuity.repository import ConversationRepository, ConversationVersionConflict, OpenThreadRepository


@dataclass(frozen=True, slots=True)
class ContinuationResolution:
    consumed: bool
    decision: str
    reason: str
    conversation_id: str = ""
    reply_act: ConversationalAct = ConversationalAct.UNKNOWN
    payload: dict[str, Any] = field(default_factory=dict)
    conversation: CurrentConversation | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumed": self.consumed, "decision": self.decision, "reason": self.reason,
            "conversation_id": self.conversation_id, "reply_act": self.reply_act.value,
            "payload": self.payload, "conversation": self.conversation.to_dict() if self.conversation else None,
            "latency_ms": self.latency_ms,
        }


class ConversationContinuityService:
    """Single owner of conversation attention, compatibility and consumption."""

    def __init__(
        self, conversations: ConversationRepository, threads: OpenThreadRepository, *,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.conversations = conversations
        self.threads = threads
        self.now_fn = now_fn
        self.shadow_diffs: list[dict[str, Any]] = []
        self.latencies_ms: list[float] = []
        self.last_resolution: ContinuationResolution | None = None

    def open_conversation(
        self, *, context_kind: ConversationContext, context_id: str, topic: str,
        origin_event_id: str, expected_reply: ExpectedReply, domain_payload: dict[str, Any] | None = None,
        participants: tuple[str, ...] = ("leo", "hebe"), reason: str = "hebe_handed_turn_to_leo",
    ) -> CurrentConversation:
        now = self.now_fn()
        expires_at = expected_reply.expires_at or now + 60.0
        item = CurrentConversation(
            id=f"conv_{uuid.uuid4().hex}", context_kind=context_kind, context_id=context_id,
            participants=participants, attention_state=AttentionState.HANDED_OFF, turn_owner="leo",
            expected_reply=expected_reply, topic=topic, origin_event_id=origin_event_id,
            last_event_id=origin_event_id, opened_at=now, last_turn_at=now, expires_at=expires_at,
            status=ConversationStatus.WAITING_ON_LEO, domain_payload=dict(domain_payload or {}),
        )
        opened = self.conversations.open(item)
        self.threads.archive_interrupted_clarifications(
            event_id=origin_event_id, now=now,
        )
        thread = OpenThread(
            id=f"thread_{uuid.uuid4().hex}", thread_type="clarification",
            scope_kind=context_kind.value, scope_id=context_id, participant_ids=participants,
            subject_ref=opened.id, summary=f"Unresolved {topic} clarification",
            origin_event_id=origin_event_id, latest_event_id=origin_event_id,
            status=OpenThreadStatus.WAITING_ON_LEO, priority=50, created_at=now,
            relevance_until=expires_at, valid_until=expires_at,
        )
        self.threads.create(thread)
        print(
            f"[HEBE][CONVERSATION_OPEN] conversation_id={opened.id} context={context_kind.value} "
            f"participants={list(participants)} origin_event={origin_event_id} reason={reason}", flush=True,
        )
        print(
            f"[HEBE][CONVERSATION_HANDOFF] conversation_id={opened.id} from=hebe to=leo "
            f"expected_reply={expected_reply.type.value} expires_at={expires_at}", flush=True,
        )
        return opened

    def resolve_input(
        self, *, context_kind: str, context_id: str, source: str, participant: str,
        authority: str, text: str, event_id: str, wake: bool = False,
        consume: bool = True,
    ) -> ContinuationResolution:
        started = time.perf_counter()

        def finish(**kwargs: Any) -> ContinuationResolution:
            latency = (time.perf_counter() - started) * 1000.0
            result = ContinuationResolution(latency_ms=latency, **kwargs)
            self.latencies_ms.append(latency)
            self.last_resolution = result
            return result

        try:
            active = self.conversations.get_active(context_kind, context_id)
        except Exception as exc:
            return finish(consumed=False, decision="reject", reason=f"repository_error:{type(exc).__name__}")
        if active is None:
            return finish(consumed=False, decision="no_conversation", reason="no_compatible_active_conversation")
        if wake:
            if not consume:
                return finish(consumed=False, decision="interrupt_shadow", reason="new_owner_command_interrupted", conversation_id=active.id, conversation=active)
            updated, _ = self.conversations.transition(
                active.id, expected_version=active.version, status=ConversationStatus.INTERRUPTED,
                reason="new_owner_command_interrupted", last_event_id=event_id, now=self.now_fn(),
            )
            self.threads.transition_for_subject(
                active.id, status=OpenThreadStatus.ARCHIVED, event_id=event_id, now=self.now_fn(),
            )
            print(f"[HEBE][CONVERSATION_CLOSE] conversation_id={active.id} reason=new_owner_command_interrupted", flush=True)
            return finish(consumed=False, decision="interrupt", reason="new_owner_command_interrupted", conversation_id=active.id, conversation=updated)
        now = self.now_fn()
        if active.expires_at <= now:
            updated, _ = self.conversations.transition(
                active.id, expected_version=active.version, status=ConversationStatus.EXPIRED,
                reason="ttl", last_event_id=event_id, now=now,
            )
            self.threads.transition_for_subject(
                active.id, status=OpenThreadStatus.EXPIRED, event_id=event_id, now=now,
            )
            print(f"[HEBE][CONVERSATION_EXPIRE] conversation_id={active.id} reason=ttl", flush=True)
            return finish(consumed=False, decision="reject", reason="expired", conversation_id=active.id, conversation=updated)
        expected = active.expected_reply
        if active.status != ConversationStatus.WAITING_ON_LEO or active.turn_owner != "leo" or expected is None:
            return finish(consumed=False, decision="reject", reason="turn_owner_mismatch", conversation_id=active.id, conversation=active)
        if participant != expected.allowed_participant or authority != "owner":
            reason = "participant_mismatch" if participant != expected.allowed_participant else "authority_mismatch"
            print(f"[HEBE][CONVERSATION_REJECT_REPLY] conversation_id={active.id} source={source} reason={reason}", flush=True)
            return finish(consumed=False, decision="reject", reason=reason, conversation_id=active.id, conversation=active)
        if source not in set(expected.allowed_sources):
            print(f"[HEBE][CONVERSATION_REJECT_REPLY] conversation_id={active.id} source={source} reason=source_mismatch", flush=True)
            return finish(consumed=False, decision="reject", reason="source_mismatch", conversation_id=active.id, conversation=active)
        act, payload, reason = expected.classify(text)
        if act == ConversationalAct.UNKNOWN:
            print(f"[HEBE][CONVERSATION_REJECT_REPLY] conversation_id={active.id} source={source} reason=incompatible_reply", flush=True)
            return finish(consumed=False, decision="reject", reason="incompatible_reply", conversation_id=active.id, conversation=active)
        terminal = ConversationStatus.CANCELLED if act == ConversationalAct.CANCEL else ConversationStatus.CLOSED
        closure = "cancelled_by_owner" if act == ConversationalAct.CANCEL else "reply_consumed"
        payload = {**payload, "domain": dict(active.domain_payload), "expected_reply_type": expected.type.value}
        if not consume:
            return finish(
                consumed=True, decision="compatible_reply_shadow", reason=reason,
                conversation_id=active.id, reply_act=act, payload=payload, conversation=active,
            )
        try:
            updated, consumed = self.conversations.transition(
                active.id, expected_version=active.version, status=terminal, reason=closure,
                last_event_id=event_id, now=now, consumed_event_id=event_id,
            )
        except ConversationVersionConflict:
            return finish(consumed=False, decision="reject", reason="version_conflict", conversation_id=active.id)
        if not consumed:
            return finish(consumed=False, decision="duplicate", reason="event_already_consumed", conversation_id=active.id, conversation=updated)
        self.threads.transition_for_subject(
            active.id,
            status=OpenThreadStatus.ARCHIVED if terminal == ConversationStatus.CANCELLED else OpenThreadStatus.RESOLVED,
            event_id=event_id, now=now,
        )
        print(
            f"[HEBE][CONVERSATION_RESOLVE] conversation_id={active.id} event_id={event_id} source={source} "
            f"wake=false decision=compatible_reply reply_act={act.value}", flush=True,
        )
        print(f"[HEBE][CONVERSATION_CLOSE] conversation_id={active.id} reason={closure}", flush=True)
        return finish(
            consumed=True, decision="compatible_reply", reason=reason, conversation_id=active.id,
            reply_act=act, payload=payload, conversation=updated,
        )

    def expire_due(self) -> int:
        now = self.now_fn()
        expired = self.conversations.expire_due(now)
        if expired:
            self.threads.expire_closed_clarifications(event_id="ttl", now=now)
        return expired

    def record_shadow(self, *, legacy_result: bool, v2_result: ContinuationResolution) -> dict[str, Any]:
        item = {
            "legacy_result": bool(legacy_result), "v2_result": bool(v2_result.consumed),
            "match": bool(legacy_result) == bool(v2_result.consumed),
            "difference_reason": "" if bool(legacy_result) == bool(v2_result.consumed) else v2_result.reason,
        }
        self.shadow_diffs.append(item)
        print(
            f"[HEBE][CONVERSATION_SHADOW] legacy_result={str(item['legacy_result']).lower()} "
            f"v2_result={str(item['v2_result']).lower()} match={str(item['match']).lower()} "
            f"difference_reason={item['difference_reason'] or 'none'}", flush=True,
        )
        return item

    def performance(self) -> dict[str, float | int]:
        values = sorted(self.latencies_ms)
        if not values:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0}
        p50 = statistics.median(values)
        p95 = values[min(len(values) - 1, max(0, math.ceil(len(values) * 0.95) - 1))]
        return {"count": len(values), "p50_ms": round(p50, 6), "p95_ms": round(p95, 6)}

    def shadow_metrics(self) -> dict[str, Any]:
        total = len(self.shadow_diffs)
        matches = sum(bool(item.get("match")) for item in self.shadow_diffs)
        reasons: dict[str, int] = {}
        for item in self.shadow_diffs:
            reason = str(item.get("difference_reason") or "")
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "total": total,
            "matches": matches,
            "differences": total - matches,
            "match_rate": round(matches / total, 6) if total else 1.0,
            "difference_reasons": reasons,
        }
