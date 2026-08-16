from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.persistent_logs import log_jsonl_event


@dataclass(frozen=True)
class ProactiveDecision:
    id: str
    proactive_type: str
    trigger: str
    anchor_type: str
    anchor_quality: float
    current_game: str
    current_activity: str
    stream_state: dict[str, Any]
    schedule_slot: dict[str, Any] | None
    action_available: bool
    suggested_action: str
    knowledge_source: str
    confidence: float
    cooldown_key: str
    should_speak: bool
    reason: str
    blocked_reason: str = ""
    proposed_response: str = ""
    final_response: str = ""
    stream_session_id: int | None = None
    twitch_stream_id: str = ""
    anchor_id: str = ""
    selected_route: str = ""
    social_value_score: float = 0.0
    interruption_cost: float = 0.0
    channel_cost: float = 0.0
    speech_intent_id: str = ""
    speech_intent_type: str = ""
    speech_intent_status: str = ""
    game_advice_validation: dict[str, Any] | None = None
    score: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def log_proactive_decision(decision: ProactiveDecision) -> None:
    payload = decision.to_dict()
    log_jsonl_event("proactive_decisions", payload)
    print(
        "[HEBE][PROACTIVE_DECISION] "
        f"type={decision.proactive_type} trigger={decision.trigger} "
        f"anchor={decision.anchor_type} quality={decision.anchor_quality:.2f} "
        f"should_speak={str(decision.should_speak).lower()} "
        f"reason={decision.reason or decision.blocked_reason}",
        flush=True,
    )


class StreamPreparationRoutine:
    def evaluate(
        self,
        *,
        stream: Any,
        schedule_slot: dict | None,
        obs_running: bool | None,
        expected_game_running: bool | None = None,
        twitch_connected: bool | None = None,
        chat_connected: bool | None = None,
        stt_listening: bool | None = None,
        tts_ready: bool | None = None,
        vtube_connected: bool | None = None,
        title_category_known: bool | None = None,
        game_run_state_ready: bool | None = None,
        trigger: str = "stream_schedule_window",
    ) -> ProactiveDecision:
        stream_mode = bool(getattr(stream, "enabled", False))
        live_known = bool(getattr(stream, "live_status_known", False))
        live = bool(getattr(stream, "is_live", False)) if live_known else None
        expected_game = str((schedule_slot or {}).get("game") or getattr(stream, "current_game", "") or "").strip()
        current_activity = str(getattr(stream, "current_activity", "") or "pre_stream")
        stream_state = {
            "stream_mode": stream_mode,
            "is_live": live,
            "live_status_known": live_known,
            "obs_running": obs_running,
            "expected_game_running": expected_game_running,
            "twitch_connected": twitch_connected,
            "chat_connected": chat_connected,
            "stt_listening": stt_listening,
            "tts_ready": tts_ready,
            "vtube_connected": vtube_connected,
            "title_category_known": title_category_known,
            "game_run_state_ready": game_run_state_ready,
        }
        missing: list[str] = []
        actions: list[str] = []
        if not stream_mode:
            missing.append("stream_mode")
            actions.append("enable_stream_mode")
        if obs_running is False:
            missing.append("obs")
            actions.append("open_obs")
        if expected_game and expected_game_running is False:
            missing.append("expected_game")
        if twitch_connected is False:
            missing.append("twitch_connection")
        if chat_connected is False:
            missing.append("chat_connection")
        if stt_listening is False:
            missing.append("stt")
        if tts_ready is False:
            missing.append("tts")
        if vtube_connected is False:
            missing.append("vtube_studio")
        if title_category_known is False:
            missing.append("title_category")
        if expected_game and game_run_state_ready is False:
            missing.append("game_run_state")
            actions.append("sync_game_run_state")

        action_available = bool(actions)
        already_ready = bool(schedule_slot) and not missing and stream_mode and (obs_running is True or obs_running is None)
        should_speak = bool(schedule_slot) and not already_ready and (action_available or missing)
        reason = "stream_prep_checklist" if should_speak else "already_prepared" if already_ready else "no_schedule_slot"
        blocked = "" if should_speak else reason
        confidence = 0.88 if schedule_slot else 0.4
        anchor_quality = 0.92 if schedule_slot else 0.25

        decision = ProactiveDecision(
            id=f"pro_{uuid.uuid4().hex}",
            proactive_type="actionable_routine",
            trigger=trigger,
            anchor_type="stream_schedule_event" if schedule_slot else "none",
            anchor_quality=anchor_quality,
            current_game=expected_game,
            current_activity=current_activity,
            stream_state=stream_state,
            schedule_slot=dict(schedule_slot or {}) or None,
            action_available=action_available,
            suggested_action=",".join(actions),
            knowledge_source="stream_schedule+runtime_state",
            confidence=confidence,
            cooldown_key="stream_preparation",
            should_speak=should_speak,
            reason=reason if should_speak else "",
            blocked_reason=blocked,
            score={
                "anchor_quality": anchor_quality,
                "usefulness": 0.9 if missing else 0.2,
                "confidence": confidence,
            },
        )
        print(
            "[HEBE][STREAM_PREP] "
            f"schedule={str(bool(schedule_slot)).lower()} obs={obs_running} "
            f"game={expected_game or 'unknown'} stream_mode={str(stream_mode).lower()} "
            f"action_available={str(action_available).lower()} missing={missing}",
            flush=True,
        )
        log_proactive_decision(decision)
        return decision


def scheduled_reminder_decision(*, trigger: str, schedule_slot: dict | None, current_game: str = "") -> ProactiveDecision:
    decision = ProactiveDecision(
        id=f"pro_{uuid.uuid4().hex}",
        proactive_type="scheduled_reminder",
        trigger=trigger,
        anchor_type="stream_schedule_time",
        anchor_quality=0.72 if schedule_slot else 0.2,
        current_game=current_game,
        current_activity="pre_stream",
        stream_state={},
        schedule_slot=dict(schedule_slot or {}) or None,
        action_available=False,
        suggested_action="",
        knowledge_source="stream_schedule",
        confidence=0.75 if schedule_slot else 0.35,
        cooldown_key="stream_preparation",
        should_speak=bool(schedule_slot),
        reason="scheduled_stream_window" if schedule_slot else "",
        blocked_reason="" if schedule_slot else "no_schedule_slot",
    )
    log_proactive_decision(decision)
    return decision


def technical_cooldown_active(stream: Any, key: str, *, now: float | None = None) -> bool:
    now = time.time() if now is None else float(now)
    cooldowns = getattr(stream, "cooldowns", None)
    if not isinstance(cooldowns, dict):
        return False
    until = float(cooldowns.get(str(key), 0.0) or 0.0)
    return bool(until and now < until)


def mark_technical_cooldown(stream: Any, key: str, *, now: float | None = None, seconds: float = 45 * 60) -> None:
    now = time.time() if now is None else float(now)
    cooldowns = getattr(stream, "cooldowns", None)
    if not isinstance(cooldowns, dict):
        stream.cooldowns = {}
        cooldowns = stream.cooldowns
    cooldowns[str(key)] = now + seconds
