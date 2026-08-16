from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from app.cognitive.core_loop import PerceivedEvent, PolicyContract, PresenceEngine, UnderstandingResult
from app.cognitive.scheduler import InternalEvent
from app.stream.proactive import ProactiveDecision, log_proactive_decision, semantic_cooldown_key
from app.stream.behavior_adaptation import AdaptationAction, BehaviorAdaptationService
from app.stream.speech_intents import (
    SpeechIntent,
    SpeechIntentManager,
    SpeechIntentStatus,
    SpeechIntentType,
)
from app.stream.spontaneity import StreamSpontaneityService
from app.stream.scene_timeline import SceneTimelineManager, SpontaneousOpportunityManager
from app.stream.state import StreamSessionState


@dataclass
class StreamCompanionTick:
    should_speak: bool
    reason: str
    blocked_reason: str
    decision: ProactiveDecision
    event: InternalEvent | None = None
    route: str = "observe_only"
    presence: dict[str, Any] = field(default_factory=dict)
    readiness: dict[str, Any] = field(default_factory=dict)
    speech_intent: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CompanionAnchorEvidence:
    anchor_id: str
    anchor_type: str
    raw_owner_fragments: list[str]
    exact_supported_claims: list[str]
    timestamps: list[float]
    topic_id: str
    currentness: float
    confidence: float
    allowed_contribution_types: list[str]
    forbidden_claims: list[str]
    expires_at: float
    scene_id: str = ""
    state_version: int = 0
    current_state: str = "active"
    terminal: bool = False
    extracted_subject: str = ""
    extracted_object: str = ""
    extracted_predicate: str = ""
    supported_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "anchor_type": self.anchor_type,
            "raw_owner_fragments": self.raw_owner_fragments,
            "exact_supported_claims": self.exact_supported_claims,
            "timestamps": self.timestamps,
            "topic_id": self.topic_id,
            "currentness": self.currentness,
            "confidence": self.confidence,
            "allowed_contribution_types": self.allowed_contribution_types,
            "forbidden_claims": self.forbidden_claims,
            "expires_at": self.expires_at,
            "scene_id": self.scene_id,
            "state_version": self.state_version,
            "current_state": self.current_state,
            "terminal": self.terminal,
            "extracted_subject": self.extracted_subject,
            "extracted_object": self.extracted_object,
            "extracted_predicate": self.extracted_predicate,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
        }


class StreamCompanionLoop:
    """Live-stream heartbeat for safe solo companion comments.

    The loop owns health, structured logging, and presence routing. Existing
    StreamSpontaneityService still owns anchor/timing/cooldown judgment.
    """

    def __init__(
        self,
        *,
        spontaneity: StreamSpontaneityService | None = None,
        presence_engine: PresenceEngine | None = None,
        scene_timeline: SceneTimelineManager | None = None,
        opportunities: SpontaneousOpportunityManager | None = None,
        now_fn: Callable[[], float] | None = None,
        owner_voice_active_fn: Callable[[], bool] | None = None,
        tts_active_fn: Callable[[], bool] | None = None,
        intent_manager: SpeechIntentManager | None = None,
        behavior_adaptation: BehaviorAdaptationService | None = None,
    ):
        self.spontaneity = spontaneity or StreamSpontaneityService()
        self.presence_engine = presence_engine or PresenceEngine()
        self.scene_timeline = scene_timeline or SceneTimelineManager(now_fn=now_fn or time.time)
        self.opportunities = opportunities or SpontaneousOpportunityManager(now_fn=now_fn or time.time)
        self._now_fn = now_fn
        self.owner_voice_active_fn = owner_voice_active_fn or (lambda: False)
        self.tts_active_fn = tts_active_fn or (lambda: False)
        self.intent_manager = intent_manager or SpeechIntentManager(now_fn=now_fn or time.time)
        self.behavior_adaptation = behavior_adaptation or BehaviorAdaptationService()
        self.last_tick_ts = 0.0
        self.last_summary_ts = 0.0
        self._used_anchor_ids: set[str] = set()
        self.ticks = 0
        self.should_speak_count = 0
        self.emitted_count = 0
        self.blocked_reasons: Counter[str] = Counter()

    def reset_session(self) -> None:
        self.last_tick_ts = 0.0
        self.last_summary_ts = 0.0
        self._used_anchor_ids.clear()
        self.opportunities.reset()
        self.ticks = 0
        self.should_speak_count = 0
        self.emitted_count = 0
        self.blocked_reasons.clear()
        self.intent_manager.reset_session()

    def due(self, *, last_poll_ts: float, interval_sec: float) -> bool:
        return self._now() - float(last_poll_ts or 0.0) >= float(interval_sec or 45.0)

    def evaluate(
        self,
        stream: StreamSessionState | None,
        *,
        stream_tts_enabled: bool,
        output_mode: str,
        backend_running: bool = True,
    ) -> StreamCompanionTick | None:
        now = self._now()
        if not self._eligible_to_tick(
            stream,
            stream_tts_enabled=stream_tts_enabled,
            output_mode=output_mode,
            backend_running=backend_running,
            now=now,
        ):
            self.maybe_log_health(stream, now=now)
            return None

        assert stream is not None
        decision_started = time.perf_counter()
        self.last_tick_ts = now
        self.ticks += 1
        mode = str(getattr(stream, "presence_mode", "reactive") or "reactive")
        print(f"[HEBE][PRESENCE_MODE] mode={mode}", flush=True)
        last_public = float(
            getattr(stream, "last_public_reply_ts", 0.0)
            or getattr(stream, "last_hebe_stream_speak_ts", 0.0)
            or 0.0
        )
        last_owner = float(getattr(stream, "last_voice_event_ts", 0.0) or 0.0)
        print(
            "[HEBE][STREAM_COMPANION_TICK] "
            f"live={str(bool(getattr(stream, 'is_live', False))).lower()} "
            f"stream_id={self._twitch_stream_id(stream) or 'unknown'} "
            f"session_id={getattr(stream, 'active_stream_session_id', None)} "
            f"mode={mode} last_public_hebe={last_public:.3f} last_owner_stt={last_owner:.3f}",
            flush=True,
        )

        readiness = self.spontaneity.evaluate(stream, now=now, live_override=False, mutate_baseline=True)
        anchor = self._selected_anchor(stream, readiness, now)
        print(
            "[HEBE][COMPANION_ANCHORS] "
            f"candidates={readiness.get('specific_context_anchors') or []} "
            f"selected={anchor.get('type') or 'none'} reason={anchor.get('reason') or readiness.get('blocked_reason')}",
            flush=True,
        )

        self._create_speech_intent(stream, readiness, anchor, now=now)
        owner_voice_active = bool(self.owner_voice_active_fn())
        stream.owner_voice_active = owner_voice_active
        if owner_voice_active:
            stream.owner_voice_started_ts = now
        owner_ended_at = float(
            getattr(stream, "last_owner_utterance_end_ts", 0.0)
            or getattr(stream, "last_voice_event_ts", 0.0)
            or 0.0
        )
        arbitration = self.intent_manager.arbitrate(
            owner_voice_active=owner_voice_active,
            owner_utterance_ended_at=owner_ended_at,
            tts_active=bool(self.tts_active_fn()),
            current_scene=dict(getattr(stream, "current_scene_timeline", None) or {}),
            now=now,
        )
        stream.speech_intent_state = self.intent_manager.snapshot()
        selected_intent = arbitration.intent
        if selected_intent is None:
            no_candidate_reason = str(readiness.get("blocked_reason") or "")
            wait_reason = (
                no_candidate_reason
                if arbitration.reason == "no_pending_intent" and no_candidate_reason not in {"", "ready"}
                else arbitration.reason
            )
            presence = {
                "should_intervene": False,
                "intervention_level": "observe_only",
                "reason": wait_reason,
                "social_value_score": 0.0,
                "interruption_cost": 1.0 if owner_voice_active else 0.0,
                "channel_cost": 0.0,
            }
        else:
            candidate_ref = " ".join(
                value for value in (
                    selected_intent.subject_ref,
                    selected_intent.topic,
                    str(readiness.get("candidate_topic") or ""),
                ) if value
            )
            adaptation = self.behavior_adaptation.evaluate_candidate(
                stream,
                candidate_ref,
                topic=selected_intent.topic,
                mode="proactive",
                now=now,
            )
            readiness["behavior_adaptation"] = adaptation.to_dict()
            if adaptation.action == AdaptationAction.DOWNRANK:
                selected_intent.value *= 0.55
            presence = self._presence_decision(stream, readiness, anchor, selected_intent, now=now)
            if adaptation.action in {AdaptationAction.COOLDOWN, AdaptationAction.SUPPRESS}:
                presence["should_intervene"] = False
                presence["reason"] = f"behavior_adaptation_{adaptation.action.value}"
            if not presence.get("should_intervene"):
                self.intent_manager.release(selected_intent.id, str(presence.get("reason") or "presence_rejected"))
                stream.speech_intent_state = self.intent_manager.snapshot()
        route = self._route_for_presence(stream, presence, stream_tts_enabled=stream_tts_enabled, output_mode=output_mode)
        should_speak = bool(selected_intent and presence.get("should_intervene") and route != "observe_only")
        blocked_reason = "" if should_speak else str(presence.get("reason") or arbitration.reason or "observe_only")
        reason = "presence_value" if should_speak else blocked_reason
        if should_speak:
            print(
                f"[HEBE][STREAM_COMPANION_ALLOWED] mode={mode} anchor={anchor.get('type') or 'none'}",
                flush=True,
            )
            self.should_speak_count += 1
            event = self.spontaneity.build_event(
                stream,
                mode=str(readiness.get("presence_mode") or mode),
                topic=str(readiness.get("candidate_topic") or ""),
            )
            if event.payload is not None:
                event.payload["source"] = "stream_companion_tick"
                event.payload["anchor_type"] = anchor.get("type")
                event.payload["anchor_quality"] = anchor.get("quality")
                event.payload["anchor_evidence"] = anchor.get("evidence") or {}
                event.payload["speech_intent_id"] = selected_intent.id
                event.payload["speech_intent_type"] = selected_intent.type.value
                event.payload["speech_intent"] = selected_intent.to_dict()
                scene_guard = dict(anchor.get("scene_guard") or self.scene_timeline.snapshot())
                event.payload["scene_guard"] = scene_guard
                opportunity = self.opportunities.open(
                    str(anchor.get("id") or ""),
                    expires_at=float((anchor.get("evidence") or {}).get("expires_at", 0.0) or 0.0),
                )
                if opportunity is not None:
                    event.payload["opportunity_id"] = opportunity.opportunity_id
                    stream.spontaneous_opportunities = self.opportunities.all_states()
                event.payload["core_loop"] = {"intervention": presence}
        else:
            self.blocked_reasons[blocked_reason] += 1
            public_reason = "reactive_mode" if blocked_reason == "presence mode is reactive" else blocked_reason
            print(f"[HEBE][STREAM_COMPANION_BLOCKED] reason={public_reason}", flush=True)
            event = None

        decision = self._build_decision(
            stream,
            readiness=readiness,
            anchor=anchor,
            presence=presence,
            should_speak=should_speak,
            blocked_reason=blocked_reason,
            reason=reason,
            selected_route=route,
            speech_intent=selected_intent,
            now=now,
        )
        stream.last_proactive_decision = decision.to_dict()
        log_proactive_decision(decision)
        print(
            "[HEBE][STREAM_COMPANION_DECISION] "
            f"should_speak={str(should_speak).lower()} "
            f"{'anchor=' + str(anchor.get('type')) + ' route=' + route if should_speak else 'reason=' + blocked_reason}",
            flush=True,
        )
        self.maybe_log_health(stream, now=now)
        self.intent_manager.observe_presence_turn_decision(
            (time.perf_counter() - decision_started) * 1000.0
        )
        stream.speech_intent_state = self.intent_manager.snapshot()
        return StreamCompanionTick(
            should_speak=should_speak,
            reason=reason,
            blocked_reason=blocked_reason,
            decision=decision,
            event=event,
            route=route,
            presence=presence,
            readiness=readiness,
            speech_intent=selected_intent.to_dict() if selected_intent else {},
        )

    def record_emitted(
        self,
        stream: StreamSessionState | None,
        final_response: str,
        *,
        route: str = "stream_tts_reply",
        intent_id: str = "",
    ) -> None:
        self.emitted_count += 1
        if stream is None:
            return
        intent_id = intent_id or str((getattr(stream, "last_proactive_decision", {}) or {}).get("speech_intent_id") or "")
        if intent_id:
            self.intent_manager.mark_emitted(
                intent_id,
                owner_utterance_ended_at=float(getattr(stream, "last_owner_utterance_end_ts", 0.0) or 0.0),
            )
            stream.speech_intent_state = self.intent_manager.snapshot()
        anchor_id = str((getattr(stream, "last_proactive_decision", {}) or {}).get("anchor_id") or "")
        if anchor_id:
            self._used_anchor_ids.add(anchor_id)
            opportunity = self.opportunities.for_anchor(anchor_id)
            if opportunity is not None:
                self.opportunities.mark(opportunity.opportunity_id, "emitted")
                stream.spontaneous_opportunities = self.opportunities.all_states()
        decision = getattr(stream, "last_proactive_decision", None)
        if isinstance(decision, dict) and decision.get("trigger") == "stream_companion_tick":
            decision = dict(decision)
            decision["final_response"] = str(final_response or "")
            decision["selected_route"] = route
            stream.last_proactive_decision = decision

    def maybe_log_health(self, stream: StreamSessionState | None, *, now: float | None = None, force: bool = False) -> None:
        now = self._now() if now is None else float(now)
        live = bool(stream and getattr(stream, "is_live", False))
        if live and not self.last_tick_ts:
            live_since = float(getattr(stream, "last_stream_live_transition_ts", 0.0) or 0.0)
            if live_since and now - live_since > 120:
                print(
                    "[HEBE][STREAM_COMPANION_HEALTH] "
                    f"status=not_running live=true last_tick=never reason=no_tick_after_live_transition",
                    flush=True,
                )
        if not force and now - self.last_summary_ts < 600:
            return
        self.last_summary_ts = now
        top = ",".join(f"{reason}:{count}" for reason, count in self.blocked_reasons.most_common(5))
        print(
            "[HEBE][STREAM_COMPANION_HEALTH] "
            f"ticks={self.ticks} should_speak={self.should_speak_count} emitted={self.emitted_count} "
            f"top_blocked_reasons={top or 'none'} intent_metrics={self.intent_manager.metrics_snapshot()}",
            flush=True,
        )

    def _eligible_to_tick(
        self,
        stream: StreamSessionState | None,
        *,
        stream_tts_enabled: bool,
        output_mode: str,
        backend_running: bool,
        now: float,
    ) -> bool:
        if not backend_running or stream is None:
            return False
        if not bool(getattr(stream, "enabled", False)):
            return False
        live = bool(getattr(stream, "is_live", False) or getattr(stream, "live_test_override", False))
        if not live:
            return False
        if output_mode in {"silent", "twitch_chat_only"}:
            return False
        mode = str(getattr(stream, "stream_voice_mode", "normal") or "normal")
        if mode == "muted" and float(getattr(stream, "muted_until", 0.0) or 0.0) > now:
            return False
        if mode == "wake_only" and float(getattr(stream, "wake_only_until", 0.0) or 0.0) > now:
            return False
        eligible = bool(stream_tts_enabled or output_mode == "ui_only")
        if not eligible:
            reason = "stream_tts_disabled"
            self.blocked_reasons[reason] += 1
            print(f"[HEBE][STREAM_COMPANION_BLOCKED] reason={reason}", flush=True)
        return eligible

    def _presence_decision(
        self,
        stream: StreamSessionState,
        readiness: dict[str, Any],
        anchor: dict[str, Any],
        intent: SpeechIntent,
        *,
        now: float,
    ) -> dict[str, Any]:
        quality = float(anchor.get("quality") or 0.0)
        pressure = max(0.0, min(1.0, max(intent.value, 0.42 + quality * 0.45)))
        readiness_reason = str(readiness.get("blocked_reason") or "ready")
        turn_only_reasons = {
            "Leo spoke recently", "recent_owner_speech", "recent_chat_activity",
            "chat_active", "chat activity baseline not ready",
        }
        anchored = intent.type not in {SpeechIntentType.SELF_INITIATED_TOPIC, SpeechIntentType.IDLE_CHATTER}
        cognitive_material = bool(intent.contribution_material.get("cognitive_candidate"))
        material_replaces_anchor = readiness_reason in {"no_high_quality_anchor", "weak_anchor", "no_specific_context"}
        budget_allowed = (
            bool(readiness.get("would_send"))
            or bool(anchored and readiness_reason in turn_only_reasons)
            or bool(cognitive_material and material_replaces_anchor)
        )
        budget = {
            "allowed": budget_allowed,
            "reason": "anchored_intent_turn_arbitrated" if budget_allowed and not readiness.get("would_send") else readiness_reason,
            "presence_rejection_reason": "" if budget_allowed else readiness_reason,
        }
        perception = PerceivedEvent(
            event_id=intent.id,
            timestamp=now,
            source="stream_companion_tick",
            source_type="system",
            speaker="Hebe",
            speaker_type="assistant",
            output_context="stream",
            stream_live=bool(getattr(stream, "is_live", False)),
            current_game=str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or ""),
            current_activity=str(getattr(stream, "current_activity", "") or "unknown"),
            confidence=max(quality, intent.value),
        )
        understanding = UnderstandingResult(
            intent=f"stream_companion_{intent.type.value.lower()}",
            confidence=max(quality, intent.value),
            authority="system",
            reply_pressure=pressure,
            social_context=str(intent.topic or anchor.get("type") or "stream_silence"),
        )
        decision = self.presence_engine.decide(
            perception=perception,
            understanding=understanding,
            policy=PolicyContract(risk_level="low"),
            budget_result=budget,
        ).to_dict()
        positive = [intent.type.value, anchor.get("type") or "stream_context"]
        negative = []
        if not readiness.get("would_send"):
            negative.append(str(readiness.get("blocked_reason") or "not_due"))
        print(
            "[HEBE][PRESENCE_ENGINE] "
            f"source=stream_companion_tick should_intervene={str(bool(decision.get('should_intervene'))).lower()} "
            f"level={decision.get('intervention_level')} social_value={decision.get('social_value_score')} "
            f"interruption_cost={decision.get('interruption_cost')} reason={decision.get('reason')}",
            flush=True,
        )
        print(f"[HEBE][PRESENCE_FACTORS] positive={positive} negative={negative}", flush=True)
        return decision

    def owner_voice_active(self) -> bool:
        return bool(self.owner_voice_active_fn())

    def yield_intent(self, intent_id: str, *, reason: str = "owner_resumed") -> bool:
        return self.intent_manager.yield_reserved(intent_id, reason=reason)

    def mark_tts_committed(self, intent_id: str) -> None:
        self.intent_manager.mark_tts_committed(intent_id)

    def _create_speech_intent(
        self,
        stream: StreamSessionState,
        readiness: dict[str, Any],
        anchor: dict[str, Any],
        *,
        now: float,
    ) -> SpeechIntent | None:
        candidates = list(getattr(stream, "speech_intent_candidates", []) or [])
        created: SpeechIntent | None = None
        anchor_id = str(anchor.get("id") or "")
        anchor_type = str(anchor.get("type") or "stream_silence")
        quality = float(anchor.get("quality") or 0.0)
        if (
            quality >= 0.55
            and anchor_type != "stream_silence"
            and (not anchor_id or not self.intent_manager.has_seen_anchor(anchor_id))
        ):
            intent_type = self._intent_type_for_anchor(anchor_type, anchor)
            created = self.intent_manager.create(
                intent_type=intent_type,
                source_event_ids=list((anchor.get("evidence") or {}).get("source_event_ids") or []),
                anchor_ids=[anchor_id] if anchor_id else [],
                topic=anchor_type,
                subject_ref=str((anchor.get("evidence") or {}).get("extracted_subject") or ""),
                value=quality,
                urgency=0.85 if intent_type in {SpeechIntentType.REACTION, SpeechIntentType.BANTER} else 0.55,
                freshness=float((anchor.get("evidence") or {}).get("currentness", 1.0) or 1.0),
                scene_relevance=dict(anchor.get("scene_guard") or {}),
                contribution_material={"anchor": anchor, "readiness_topic": readiness.get("candidate_topic")},
                now=now,
            )
        elif readiness.get("would_send"):
            created = self.intent_manager.create(
                intent_type=SpeechIntentType.IDLE_CHATTER,
                topic=str(readiness.get("candidate_topic") or "stream_context"),
                value=float((readiness.get("spontaneity_score") or {}).get("total") or 0.62),
                urgency=0.1,
                scene_relevance=dict(getattr(stream, "current_scene_timeline", None) or {}),
                contribution_material={"readiness_topic": readiness.get("candidate_topic")},
                now=now,
            )
        for candidate in candidates:
            if not isinstance(candidate, dict) or candidate.get("consumed"):
                continue
            try:
                created = self.intent_manager.create(
                    intent_type=str(candidate.get("type") or SpeechIntentType.SELF_INITIATED_TOPIC.value),
                    source_event_ids=list(candidate.get("source_event_ids") or []),
                    anchor_ids=list(candidate.get("anchor_ids") or []),
                    topic=str(candidate.get("topic") or "cognitive_candidate"),
                    subject_ref=str(candidate.get("subject_ref") or ""),
                    value=float(candidate.get("value") or 0.0),
                    urgency=float(candidate.get("urgency") or 0.3),
                    freshness=float(candidate.get("freshness") or 1.0),
                    scene_relevance=dict(candidate.get("scene_relevance") or getattr(stream, "current_scene_timeline", None) or {}),
                    contribution_material={"cognitive_candidate": True, **dict(candidate.get("material") or {})},
                    now=now,
                )
                candidate["consumed"] = True
            except (TypeError, ValueError):
                candidate["consumed"] = True
                candidate["rejected_reason"] = "invalid_speech_intent_candidate"
        stream.speech_intent_candidates = candidates[-20:]
        return created

    @staticmethod
    def _intent_type_for_anchor(anchor_type: str, anchor: dict[str, Any]) -> SpeechIntentType:
        normalized = anchor_type.casefold()
        if any(token in normalized for token in ("rng", "luck", "failure", "death", "victory", "frustration")):
            return SpeechIntentType.REACTION
        if any(token in normalized for token in ("joke", "laughter", "banter")):
            return SpeechIntentType.BANTER
        if any(token in normalized for token in ("opinion", "discourse")):
            return SpeechIntentType.OPINION
        if any(token in normalized for token in ("social", "viewer", "chat", "callback")):
            return SpeechIntentType.SOCIAL_FOLLOWUP
        return SpeechIntentType.GAME_COMMENT

    def _route_for_presence(
        self,
        stream: StreamSessionState,
        presence: dict[str, Any],
        *,
        stream_tts_enabled: bool,
        output_mode: str,
    ) -> str:
        if not bool(presence.get("should_intervene")):
            return "observe_only"
        if output_mode == "ui_only":
            return "local_ui_debug_only"
        if stream_tts_enabled:
            return "stream_tts_reply"
        return "observe_only"

    def _selected_anchor(self, stream: StreamSessionState, readiness: dict[str, Any], now: float) -> dict[str, Any]:
        current_topic = str((getattr(stream, "current_discourse_topic", {}) or {}).get("topic_id") or "")
        facts = [
            fact for fact in list(getattr(stream, "recent_run_context_facts", []) or [])
            if float(fact.get("expires_at", 0.0) or 0.0) > now
            and str(fact.get("utterance_role") or "owner_commentary") not in {
                "quoted_or_read_dialogue", "game_audio_bleed",
            }
            and bool(fact.get("proactive_eligible", True))
            and str(fact.get("id") or fact.get("fact_id") or "") not in self._used_anchor_ids
            and self.opportunities.eligible(str(fact.get("id") or fact.get("fact_id") or ""))
        ]
        facts = self.scene_timeline.filter_current_facts(
            facts,
            topic_id=current_topic,
            now=now,
            anchor_relevant=True,
        )
        high = sorted(
            facts,
            key=lambda item: (float(item.get("confidence", 0.0) or 0.0), float(item.get("timestamp", 0.0) or 0.0)),
        )
        if high:
            fact = high[-1]
            fact_topic = str(fact.get("topic_id") or "")
            topic_match = not (current_topic and fact_topic) or current_topic == fact_topic
            age = max(0.0, now - float(fact.get("timestamp", now) or now))
            allowed = bool(topic_match and float(fact.get("expires_at", 0.0) or 0.0) > now)
            print(
                "[HEBE][ANCHOR_FRESHNESS] "
                f"anchor={fact.get('id') or fact.get('fact_id') or ''} age_seconds={age:.3f} "
                f"topic_match={str(topic_match).lower()} allowed={str(allowed).lower()}",
                flush=True,
            )
            if not allowed:
                return {"id": "", "type": "stream_silence", "quality": 0.0, "reason": "stale_or_topic_mismatch"}
            raw = str(fact.get("raw_evidence") or fact.get("raw_text") or fact.get("text") or "")
            evidence = CompanionAnchorEvidence(
                anchor_id=str(fact.get("id") or fact.get("fact_id") or ""),
                anchor_type=str(fact.get("category") or fact.get("kind") or "ambient_context"),
                raw_owner_fragments=[raw] if raw else [],
                exact_supported_claims=[raw] if raw else [],
                timestamps=[float(fact.get("timestamp", now) or now)],
                topic_id=fact_topic,
                currentness=max(0.0, 1.0 - age / max(1.0, float(fact.get("ttl_sec", 60) or 60))),
                confidence=float(fact.get("confidence", 0.0) or 0.0),
                allowed_contribution_types=["contextual_reaction", "emotional_banter", "concise_observation"],
                forbidden_claims=["unsupported strategy", "save instruction", "unrelated mechanic", "stale topic fusion"],
                expires_at=float(fact.get("expires_at", 0.0) or 0.0),
                scene_id=str(fact.get("scene_id") or ""),
                state_version=int(fact.get("state_version", 0) or 0),
                current_state=str(fact.get("current_state") or "active"),
                terminal=bool(fact.get("terminal")),
                extracted_subject=str(fact.get("extracted_subject") or ""),
                extracted_object=str(fact.get("extracted_object") or ""),
                extracted_predicate=str(fact.get("extracted_predicate") or ""),
                supported_claims=[str(item) for item in fact.get("supported_claims") or []],
                unsupported_claims=[str(item) for item in fact.get("unsupported_claims") or []],
            )
            return {
                "id": fact.get("id") or "",
                "type": fact.get("category") or fact.get("kind") or "ambient_context",
                "quality": float(fact.get("confidence", 0.0) or 0.0),
                "reason": "recent_ambient_context",
                "evidence": evidence.to_dict(),
                "scene_guard": {
                    "scene_id": evidence.scene_id,
                    "state_version": evidence.state_version,
                    "current_state": evidence.current_state,
                    "terminal": evidence.terminal,
                },
            }
        anchors = list(readiness.get("specific_context_anchors") or [])
        quality = float((readiness.get("spontaneity_score") or {}).get("anchor_quality") or 0.0)
        if not quality:
            quality = 0.58 if {"game", "title"} & set(anchors) else 0.25
        if anchors:
            return {"id": "", "type": ",".join(anchors), "quality": quality, "reason": "stream_context"}
        return {"id": "", "type": "stream_silence", "quality": quality, "reason": "long_stream_silence"}

    def _build_decision(
        self,
        stream: StreamSessionState,
        *,
        readiness: dict[str, Any],
        anchor: dict[str, Any],
        presence: dict[str, Any],
        should_speak: bool,
        blocked_reason: str,
        reason: str,
        selected_route: str,
        speech_intent: SpeechIntent | None,
        now: float,
    ) -> ProactiveDecision:
        topic = str(readiness.get("candidate_topic") or anchor.get("type") or "stream_silence")
        return ProactiveDecision(
            id=f"pro_{uuid.uuid4().hex}",
            proactive_type="stream_companion",
            trigger="stream_companion_tick",
            anchor_type=str(anchor.get("type") or "none"),
            anchor_quality=float(anchor.get("quality") or 0.0),
            current_game=str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or ""),
            current_activity=str(getattr(stream, "current_activity", None) or "unknown"),
            stream_state={
                "enabled": bool(getattr(stream, "enabled", False)),
                "is_live": bool(getattr(stream, "is_live", False)),
                "live_status_known": bool(getattr(stream, "live_status_known", False)),
                "presence_mode": str(getattr(stream, "presence_mode", "") or ""),
                "stream_output_mode": str(getattr(stream, "stream_output_mode", "") or ""),
                "stream_voice_mode": str(getattr(stream, "stream_voice_mode", "") or ""),
            },
            schedule_slot=None,
            action_available=False,
            suggested_action="",
            knowledge_source="stream_context+ambient_context+presence_engine",
            confidence=float(presence.get("social_value_score") or readiness.get("confidence") or anchor.get("quality") or 0.0),
            cooldown_key=semantic_cooldown_key("", topic),
            should_speak=should_speak,
            reason=reason if should_speak else "",
            blocked_reason="" if should_speak else blocked_reason,
            stream_session_id=getattr(stream, "active_stream_session_id", None),
            twitch_stream_id=self._twitch_stream_id(stream),
            anchor_id=str(anchor.get("id") or ""),
            selected_route=selected_route,
            social_value_score=float(presence.get("social_value_score") or 0.0),
            interruption_cost=float(presence.get("interruption_cost") or 0.0),
            channel_cost=float(presence.get("channel_cost") or 0.0),
            speech_intent_id=speech_intent.id if speech_intent else "",
            speech_intent_type=speech_intent.type.value if speech_intent else "",
            speech_intent_status=speech_intent.status.value if speech_intent else "",
            score={
                **{k: float(v) for k, v in (readiness.get("spontaneity_score") or {}).items() if isinstance(v, (int, float))},
                "social_value_score": float(presence.get("social_value_score") or 0.0),
                "interruption_cost": float(presence.get("interruption_cost") or 0.0),
                "channel_cost": float(presence.get("channel_cost") or 0.0),
            },
        )

    def _twitch_stream_id(self, stream: StreamSessionState) -> str:
        return str(
            getattr(stream, "twitch_stream_id", "")
            or getattr(stream, "current_twitch_stream_id", "")
            or getattr(stream, "active_twitch_stream_id", "")
            or ""
        )

    def _now(self) -> float:
        return float(self._now_fn() if self._now_fn is not None else time.time())
