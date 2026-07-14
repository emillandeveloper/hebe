from __future__ import annotations

import time
import uuid
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from app.cognitive.core_loop import PerceivedEvent, PolicyContract, PresenceEngine, UnderstandingResult
from app.cognitive.scheduler import InternalEvent
from app.stream.proactive import ProactiveDecision, log_proactive_decision, semantic_cooldown_key
from app.stream.spontaneity import StreamSpontaneityService
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
        now_fn: Callable[[], float] | None = None,
    ):
        self.spontaneity = spontaneity or StreamSpontaneityService()
        self.presence_engine = presence_engine or PresenceEngine()
        self._now_fn = now_fn
        self.min_seconds_after_owner_speech = float(
            os.getenv("HEBE_COMPANION_MIN_SECONDS_AFTER_OWNER_SPEECH", "30")
        )
        self.last_tick_ts = 0.0
        self.last_summary_ts = 0.0
        self.ticks = 0
        self.should_speak_count = 0
        self.emitted_count = 0
        self.blocked_reasons: Counter[str] = Counter()

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

        presence = self._presence_decision(stream, readiness, anchor, now=now)
        route = self._route_for_presence(stream, presence, stream_tts_enabled=stream_tts_enabled, output_mode=output_mode)
        should_speak = bool(readiness.get("would_send") and presence.get("should_intervene") and route != "observe_only")
        blocked_reason = "" if should_speak else str(presence.get("reason") or readiness.get("blocked_reason") or "observe_only")
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
        return StreamCompanionTick(
            should_speak=should_speak,
            reason=reason,
            blocked_reason=blocked_reason,
            decision=decision,
            event=event,
            route=route,
            presence=presence,
            readiness=readiness,
        )

    def record_emitted(self, stream: StreamSessionState | None, final_response: str, *, route: str = "stream_tts_reply") -> None:
        self.emitted_count += 1
        if stream is None:
            return
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
            f"top_blocked_reasons={top or 'none'}",
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
        *,
        now: float,
    ) -> dict[str, Any]:
        recent_owner = float(getattr(stream, "last_voice_event_ts", 0.0) or 0.0)
        owner_speech_age = (now - recent_owner) if recent_owner else float("inf")
        recent_owner_block = bool(recent_owner and owner_speech_age < self.min_seconds_after_owner_speech)
        print(
            "[HEBE][RECENT_OWNER_SPEECH_GATE] "
            f"blocked={str(recent_owner_block).lower()} "
            f"age_seconds={owner_speech_age if recent_owner else 'never'} "
            f"threshold={self.min_seconds_after_owner_speech:g}",
            flush=True,
        )
        quality = float(anchor.get("quality") or 0.0)
        pressure = max(0.0, min(1.0, 0.42 + quality * 0.45))
        budget = {"allowed": bool(readiness.get("would_send")) and not recent_owner_block, "reason": readiness.get("blocked_reason") or "ready"}
        if recent_owner_block:
            budget["reason"] = "recent_owner_speech"
        perception = PerceivedEvent(
            event_id=f"stream_companion_tick:{int(now)}",
            timestamp=now,
            source="stream_companion_tick",
            source_type="system",
            speaker="Hebe",
            speaker_type="assistant",
            output_context="stream",
            stream_live=bool(getattr(stream, "is_live", False)),
            current_game=str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or ""),
            current_activity=str(getattr(stream, "current_activity", "") or "unknown"),
            confidence=quality,
        )
        understanding = UnderstandingResult(
            intent="stream_companion_anchor",
            confidence=quality,
            authority="system",
            reply_pressure=pressure,
            social_context=str(anchor.get("type") or "stream_silence"),
        )
        decision = self.presence_engine.decide(
            perception=perception,
            understanding=understanding,
            policy=PolicyContract(risk_level="low"),
            budget_result=budget,
        ).to_dict()
        positive = [anchor.get("type") or "stream_context"] if quality >= 0.55 else []
        negative = []
        if recent_owner_block:
            negative.append("recent_owner_speech")
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
        facts = [
            fact for fact in list(getattr(stream, "recent_run_context_facts", []) or [])
            if float(fact.get("expires_at", 0.0) or 0.0) > now
        ]
        high = sorted(
            facts,
            key=lambda item: (float(item.get("confidence", 0.0) or 0.0), float(item.get("timestamp", 0.0) or 0.0)),
        )
        if high:
            fact = high[-1]
            return {
                "id": fact.get("id") or "",
                "type": fact.get("category") or fact.get("kind") or "ambient_context",
                "quality": float(fact.get("confidence", 0.0) or 0.0),
                "reason": "recent_ambient_context",
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
