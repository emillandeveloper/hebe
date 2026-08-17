from __future__ import annotations

import os
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable


class SpeechIntentType(str, Enum):
    REACTION = "REACTION"
    OPINION = "OPINION"
    BANTER = "BANTER"
    CALLBACK = "CALLBACK"
    QUESTION = "QUESTION"
    GAME_COMMENT = "GAME_COMMENT"
    SOCIAL_FOLLOWUP = "SOCIAL_FOLLOWUP"
    OWNER_MEMORY_REFERENCE = "OWNER_MEMORY_REFERENCE"
    SELF_INITIATED_TOPIC = "SELF_INITIATED_TOPIC"
    RAID_FAREWELL = "RAID_FAREWELL"
    IDLE_CHATTER = "IDLE_CHATTER"


class SpeechIntentStatus(str, Enum):
    PENDING = "PENDING"
    TURN_RESERVED = "TURN_RESERVED"
    TTS_COMMITTED = "TTS_COMMITTED"
    EMITTED = "EMITTED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"
    SUPPRESSED = "SUPPRESSED"


@dataclass(frozen=True, slots=True)
class IntentTiming:
    minimum_turn_gap: float
    maximum_turn_delay: float


@dataclass(frozen=True, slots=True)
class SpeechIntentTimingConfig:
    reaction: IntentTiming = IntentTiming(1.2, 8.0)
    banter: IntentTiming = IntentTiming(1.2, 10.0)
    game_comment: IntentTiming = IntentTiming(1.8, 14.0)
    opinion: IntentTiming = IntentTiming(2.5, 25.0)
    callback: IntentTiming = IntentTiming(3.0, 45.0)
    question: IntentTiming = IntentTiming(2.5, 40.0)
    social_followup: IntentTiming = IntentTiming(1.5, 30.0)
    owner_memory_reference: IntentTiming = IntentTiming(3.0, 45.0)
    self_initiated_topic: IntentTiming = IntentTiming(5.0, 90.0)
    raid_farewell: IntentTiming = IntentTiming(0.8, 12.0)
    idle_chatter: IntentTiming = IntentTiming(30.0, 120.0)

    @classmethod
    def from_env(cls) -> "SpeechIntentTimingConfig":
        defaults = cls()

        def configured(name: str, timing: IntentTiming) -> IntentTiming:
            prefix = f"HEBE_SPEECH_INTENT_{name}"
            return IntentTiming(
                float(os.getenv(f"{prefix}_MIN_GAP_SECONDS", str(timing.minimum_turn_gap))),
                float(os.getenv(f"{prefix}_TTL_SECONDS", str(timing.maximum_turn_delay))),
            )

        return cls(**{
            field_name: configured(field_name.upper(), getattr(defaults, field_name))
            for field_name in defaults.__dataclass_fields__
        })

    def for_type(self, intent_type: SpeechIntentType) -> IntentTiming:
        return getattr(self, intent_type.value.lower())


@dataclass(slots=True)
class SpeechIntent:
    id: str
    type: SpeechIntentType
    source_event_ids: list[str]
    anchor_ids: list[str]
    topic: str
    subject_ref: str
    semantic_material: str
    value: float
    urgency: float
    freshness: float
    created_at: float
    expires_at: float
    interruptibility: str
    minimum_turn_gap: float
    maximum_turn_delay: float
    scene_relevance: dict[str, Any]
    status: SpeechIntentStatus = SpeechIntentStatus.PENDING
    suppression_reason: str = ""
    reserved_at: float = 0.0
    emitted_at: float = 0.0
    contribution_material: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["type"] = self.type.value
        value["status"] = self.status.value
        return value


@dataclass(slots=True)
class TurnArbitration:
    intent: SpeechIntent | None
    reason: str
    owner_voice_active: bool
    owner_silence_ms: int
    pending_intents: int
    ranked_candidates: list[dict[str, Any]] = field(default_factory=list)


class SpeechIntentManager:
    """Bounded, in-memory conversational intention and turn-taking state."""

    TERMINAL = {
        SpeechIntentStatus.EMITTED,
        SpeechIntentStatus.EXPIRED,
        SpeechIntentStatus.SUPERSEDED,
        SpeechIntentStatus.SUPPRESSED,
    }

    def __init__(
        self,
        *,
        timing: SpeechIntentTimingConfig | None = None,
        now_fn: Callable[[], float] | None = None,
        max_pending: int = 5,
    ) -> None:
        self.timing = timing or SpeechIntentTimingConfig.from_env()
        self._now_fn = now_fn or time.time
        self.max_pending = max(1, int(max_pending))
        self._intents: list[SpeechIntent] = []
        self.metrics: Counter[str] = Counter()
        self.created_to_emit_ms: list[float] = []
        self.turn_gap_ms: list[float] = []
        self.creation_latency_ms: list[float] = []
        self.arbitration_latency_ms: list[float] = []
        self.presence_turn_decision_latency_ms: list[float] = []

    def reset_session(self) -> None:
        self._intents.clear()
        self.metrics.clear()
        self.created_to_emit_ms.clear()
        self.turn_gap_ms.clear()
        self.creation_latency_ms.clear()
        self.arbitration_latency_ms.clear()
        self.presence_turn_decision_latency_ms.clear()

    def create(
        self,
        *,
        intent_type: SpeechIntentType | str,
        source_event_ids: list[str] | None = None,
        anchor_ids: list[str] | None = None,
        topic: str = "",
        subject_ref: str = "",
        semantic_material: str = "",
        value: float,
        urgency: float = 0.5,
        freshness: float = 1.0,
        scene_relevance: dict[str, Any] | None = None,
        contribution_material: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> SpeechIntent:
        started = time.perf_counter()
        now = float(self._now_fn() if now is None else now)
        kind = intent_type if isinstance(intent_type, SpeechIntentType) else SpeechIntentType(str(intent_type))
        timing = self.timing.for_type(kind)
        topic = str(topic or kind.value.lower())
        anchor_ids = [str(item) for item in (anchor_ids or []) if str(item)]
        existing = self._coalescing_match(kind, topic, subject_ref, scene_relevance or {})
        if existing is not None:
            if float(value) <= existing.value and float(freshness) <= existing.freshness:
                existing.source_event_ids = list(dict.fromkeys([*existing.source_event_ids, *(source_event_ids or [])]))
                existing.anchor_ids = list(dict.fromkeys([*existing.anchor_ids, *anchor_ids]))
                self.metrics["intents_coalesced"] += 1
                self.creation_latency_ms.append((time.perf_counter() - started) * 1000.0)
                return existing
            self._finish(existing, SpeechIntentStatus.SUPERSEDED, "stronger_current_intent", now)
        intent = SpeechIntent(
            id=f"intent_{uuid.uuid4().hex}", type=kind,
            source_event_ids=[str(item) for item in (source_event_ids or []) if str(item)],
            anchor_ids=anchor_ids, topic=topic, subject_ref=str(subject_ref or ""),
            semantic_material=str(semantic_material or ""),
            value=max(0.0, min(1.0, float(value))),
            urgency=max(0.0, min(1.0, float(urgency))),
            freshness=max(0.0, min(1.0, float(freshness))),
            created_at=now, expires_at=now + timing.maximum_turn_delay,
            interruptibility="yield_before_tts_commit", minimum_turn_gap=timing.minimum_turn_gap,
            maximum_turn_delay=timing.maximum_turn_delay,
            scene_relevance=dict(scene_relevance or {}),
            contribution_material=dict(contribution_material or {}),
        )
        self._intents.append(intent)
        self.metrics["intents_created"] += 1
        self.metrics[f"created:{kind.value}"] += 1
        print(
            "[HEBE][SPEECH_INTENT_CREATE] "
            f"id={intent.id} type={kind.value} anchor={','.join(anchor_ids) or 'none'} "
            f"value={intent.value:.3f} ttl={timing.maximum_turn_delay:g}",
            flush=True,
        )
        self._enforce_bound(now)
        self.creation_latency_ms.append((time.perf_counter() - started) * 1000.0)
        return intent

    def arbitrate(
        self,
        *,
        owner_voice_active: bool,
        owner_utterance_ended_at: float,
        tts_active: bool,
        current_scene: dict[str, Any] | None = None,
        now: float | None = None,
        rank_policy: Callable[[SpeechIntent], dict[str, Any]] | None = None,
    ) -> TurnArbitration:
        started = time.perf_counter()
        now = float(self._now_fn() if now is None else now)
        self.expire(now=now, current_scene=current_scene)
        pending = self.pending()
        silence = max(0.0, now - float(owner_utterance_ended_at or 0.0)) if owner_utterance_ended_at else float("inf")
        silence_ms = int(silence * 1000) if silence != float("inf") else -1
        if owner_voice_active:
            yielded = False
            for item in self._intents:
                if item.status == SpeechIntentStatus.TURN_RESERVED:
                    item.status = SpeechIntentStatus.PENDING
                    item.reserved_at = 0.0
                    item.suppression_reason = "owner_resumed"
                    yielded = True
            if yielded:
                self.metrics["yield_due_owner_resume"] += 1
                print("[HEBE][TURN_YIELD] reason=owner_resumed", flush=True)
            self.metrics["pending_due_owner_voice_active"] += len(pending)
            if pending:
                print("[HEBE][SPEECH_INTENT_PENDING] reason=owner_voice_active", flush=True)
            return self._arbitration_result(None, "owner_voice_active", True, silence_ms, started)
        if tts_active:
            return self._arbitration_result(None, "tts_active", False, silence_ms, started)
        available = [item for item in pending if silence >= item.minimum_turn_gap]
        if not available:
            reason = "turn_gap_too_short" if pending else "no_pending_intent"
            if pending:
                print(
                    f"[HEBE][SPEECH_INTENT_PENDING] reason={reason} owner_silence_ms={silence_ms}",
                    flush=True,
                )
            return self._arbitration_result(None, reason, False, silence_ms, started)
        print(
            f"[HEBE][TURN_WINDOW_OPEN] owner_silence_ms={silence_ms} pending_intents={len(pending)}",
            flush=True,
        )
        ranked: list[dict[str, Any]] = []
        eligible: list[tuple[float, SpeechIntent]] = []
        for item in available:
            base_score = self._priority(item)
            policy = dict(rank_policy(item) or {}) if rank_policy is not None else {}
            multiplier = max(0.0, min(1.0, float(policy.get("score_multiplier", 1.0) or 0.0)))
            adjusted_score = base_score * multiplier
            allowed = multiplier > 0.0 and str(policy.get("action") or "allow") not in {"cooldown", "suppress"}
            ranked.append({
                "intent_id": item.id,
                "topic": item.topic,
                "base_score": round(base_score, 6),
                "score_multiplier": round(multiplier, 6),
                "adjusted_score": round(adjusted_score, 6),
                "eligible": allowed,
                "policy": policy,
            })
            if allowed:
                eligible.append((adjusted_score, item))
            else:
                item.status = SpeechIntentStatus.SUPPRESSED
                item.reserved_at = 0.0
                item.suppression_reason = str(policy.get("reason") or "behavior_policy_suppressed")
                self.metrics["intents_suppressed"] += 1
                self.metrics[f"behavior_policy_suppressed:{item.suppression_reason}"] += 1
        if not eligible:
            return self._arbitration_result(
                None, "behavior_policy_no_candidate", False, silence_ms, started,
                ranked_candidates=ranked,
            )
        _, selected = max(eligible, key=lambda pair: pair[0])
        selected.status = SpeechIntentStatus.TURN_RESERVED
        selected.reserved_at = now
        selected.suppression_reason = ""
        self.metrics["turns_reserved"] += 1
        print(
            f"[HEBE][SPEECH_INTENT_SELECT] id={selected.id} reason=highest_value_current",
            flush=True,
        )
        return self._arbitration_result(
            selected, "turn_reserved", False, silence_ms, started,
            ranked_candidates=ranked,
        )

    def release(self, intent_id: str, reason: str, *, suppress: bool = False) -> None:
        intent = self.get(intent_id)
        if intent is None or intent.status in self.TERMINAL:
            return
        intent.status = SpeechIntentStatus.SUPPRESSED if suppress else SpeechIntentStatus.PENDING
        intent.reserved_at = 0.0
        intent.suppression_reason = str(reason or "presence_rejected")
        self.metrics["intents_suppressed"] += 1
        self.metrics[f"presence_rejected:{intent.suppression_reason}"] += 1

    def yield_reserved(self, intent_id: str, *, reason: str = "owner_resumed") -> bool:
        intent = self.get(intent_id)
        if intent is None or intent.status != SpeechIntentStatus.TURN_RESERVED:
            return False
        intent.status = SpeechIntentStatus.PENDING
        intent.reserved_at = 0.0
        intent.suppression_reason = reason
        self.metrics["yield_due_owner_resume"] += 1
        print(f"[HEBE][TURN_YIELD] id={intent.id} reason={reason}", flush=True)
        return True

    def mark_tts_committed(self, intent_id: str) -> None:
        intent = self.get(intent_id)
        if intent is not None and intent.status == SpeechIntentStatus.TURN_RESERVED:
            intent.status = SpeechIntentStatus.TTS_COMMITTED

    def mark_emitted(self, intent_id: str, *, owner_utterance_ended_at: float = 0.0, now: float | None = None) -> None:
        now = float(self._now_fn() if now is None else now)
        intent = self.get(intent_id)
        if intent is None or intent.status in self.TERMINAL:
            return
        intent.status = SpeechIntentStatus.EMITTED
        intent.emitted_at = now
        intent.suppression_reason = ""
        self.metrics["intents_emitted"] += 1
        self.metrics[f"emitted:{intent.type.value}"] += 1
        self.created_to_emit_ms.append(max(0.0, now - intent.created_at) * 1000.0)
        if owner_utterance_ended_at:
            self.turn_gap_ms.append(max(0.0, now - owner_utterance_ended_at) * 1000.0)
        print(f"[HEBE][SPEECH_INTENT_EMIT] id={intent.id} type={intent.type.value}", flush=True)

    def expire(self, *, now: float | None = None, current_scene: dict[str, Any] | None = None) -> None:
        now = float(self._now_fn() if now is None else now)
        for intent in self._intents:
            if intent.status in self.TERMINAL or intent.status == SpeechIntentStatus.TTS_COMMITTED:
                continue
            reason = ""
            if now >= intent.expires_at:
                reason = "ttl"
            elif not self._scene_compatible(intent.scene_relevance, current_scene or {}):
                reason = "stale_scene"
            if reason:
                self._finish(intent, SpeechIntentStatus.EXPIRED, reason, now)

    def pending(self) -> list[SpeechIntent]:
        return [item for item in self._intents if item.status in {SpeechIntentStatus.PENDING, SpeechIntentStatus.TURN_RESERVED}]

    def get(self, intent_id: str) -> SpeechIntent | None:
        return next((item for item in self._intents if item.id == intent_id), None)

    def has_seen_anchor(self, anchor_id: str) -> bool:
        anchor_id = str(anchor_id or "")
        return bool(anchor_id and any(anchor_id in item.anchor_ids for item in self._intents))

    def has_seen_source(self, source_event_id: str) -> bool:
        source_event_id = str(source_event_id or "")
        return bool(source_event_id and any(source_event_id in item.source_event_ids for item in self._intents))

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": [item.to_dict() for item in self.pending()],
            "all": [item.to_dict() for item in self._intents[-50:]],
            "metrics": self.metrics_snapshot(),
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            **dict(self.metrics),
            "pending": len(self.pending()),
            "time_created_to_emit": self._distribution(self.created_to_emit_ms),
            "turn_gap_before_emit": self._distribution(self.turn_gap_ms),
            "intent_creation": self._distribution(self.creation_latency_ms),
            "pending_queue_operation": self._distribution(self.creation_latency_ms),
            "turn_arbitration": self._distribution(self.arbitration_latency_ms),
            "presence_turn_decision": self._distribution(self.presence_turn_decision_latency_ms),
        }

    def observe_presence_turn_decision(self, elapsed_ms: float) -> None:
        self.presence_turn_decision_latency_ms.append(max(0.0, float(elapsed_ms)))

    def _coalescing_match(
        self,
        kind: SpeechIntentType,
        topic: str,
        subject_ref: str,
        scene: dict[str, Any],
    ) -> SpeechIntent | None:
        scene_id = str(scene.get("scene_id") or "")
        for item in reversed(self.pending()):
            same_scene = not scene_id or not item.scene_relevance.get("scene_id") or scene_id == str(item.scene_relevance.get("scene_id"))
            if item.type == kind and item.topic.casefold() == topic.casefold() and item.subject_ref == str(subject_ref or "") and same_scene:
                return item
        return None

    def _enforce_bound(self, now: float) -> None:
        pending = self.pending()
        while len(pending) > self.max_pending:
            weakest = min(pending, key=self._priority)
            self._finish(weakest, SpeechIntentStatus.SUPERSEDED, "pending_bound", now)
            pending = self.pending()

    def _finish(self, intent: SpeechIntent, status: SpeechIntentStatus, reason: str, now: float) -> None:
        intent.status = status
        intent.suppression_reason = reason
        if status == SpeechIntentStatus.EXPIRED:
            self.metrics["intents_expired"] += 1
            print(f"[HEBE][SPEECH_INTENT_EXPIRE] id={intent.id} reason={reason}", flush=True)
        elif status == SpeechIntentStatus.SUPERSEDED:
            self.metrics["intents_superseded"] += 1
            print(f"[HEBE][SPEECH_INTENT_EXPIRE] id={intent.id} reason=superseded", flush=True)

    @staticmethod
    def _scene_compatible(expected: dict[str, Any], current: dict[str, Any]) -> bool:
        expected_scene = str(expected.get("scene_id") or "")
        current_scene = str(current.get("scene_id") or "")
        if expected_scene and current_scene and expected_scene != current_scene:
            return False
        expected_topic = str(expected.get("topic_id") or "")
        current_topic = str(current.get("topic_id") or "")
        return not (expected_topic and current_topic and expected_topic != current_topic)

    @staticmethod
    def _priority(intent: SpeechIntent) -> float:
        class_bonus = {
            SpeechIntentType.RAID_FAREWELL: 0.25,
            SpeechIntentType.REACTION: 0.22,
            SpeechIntentType.BANTER: 0.16,
            SpeechIntentType.GAME_COMMENT: 0.14,
            SpeechIntentType.QUESTION: 0.10,
            SpeechIntentType.OPINION: 0.08,
            SpeechIntentType.SOCIAL_FOLLOWUP: 0.08,
            SpeechIntentType.CALLBACK: 0.05,
            SpeechIntentType.OWNER_MEMORY_REFERENCE: 0.05,
            SpeechIntentType.SELF_INITIATED_TOPIC: 0.0,
            SpeechIntentType.IDLE_CHATTER: -0.15,
        }[intent.type]
        return intent.value + intent.urgency * 0.25 + intent.freshness * 0.15 + class_bonus

    def _arbitration_result(
        self,
        intent: SpeechIntent | None,
        reason: str,
        owner_active: bool,
        silence_ms: int,
        started: float,
        ranked_candidates: list[dict[str, Any]] | None = None,
    ) -> TurnArbitration:
        self.arbitration_latency_ms.append((time.perf_counter() - started) * 1000.0)
        return TurnArbitration(
            intent, reason, owner_active, silence_ms, len(self.pending()),
            list(ranked_candidates or []),
        )

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0}
        ordered = sorted(values)
        percentile = lambda fraction: ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))]
        return {
            "count": len(ordered),
            "p50_ms": round(percentile(0.50), 3),
            "p95_ms": round(percentile(0.95), 3),
        }
