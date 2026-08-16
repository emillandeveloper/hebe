from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable


VOLATILE_STREAM_DEFAULTS: dict[str, Any] = {
    "companion_tick_count": 0,
    "idle_prompts_sent_stream": 0,
    "recent_idle_messages": [],
    "recent_chat_messages": [],
    "recent_active_users": [],
    "recent_chat_topics": [],
    "recent_chat_summary": None,
    "last_chat_activity_ts": 0.0,
    "last_hebe_stream_speak_ts": 0.0,
    "public_reply_timestamps": [],
    "public_reply_viewer_timestamps": {},
    "public_reply_thread_counts": {},
    "public_reply_boundary_cooldowns": {},
    "public_reply_no_mention_timestamps": [],
    "consecutive_public_replies": 0,
    "last_public_reply_ts": 0.0,
    "human_messages_since_last_public_reply": 0,
    "last_no_mention_reply_ts": 0.0,
    "consumed_spontaneity_anchors": [],
    "invalidated_anchors": [],
    "recent_progress_markers": [],
    "recent_run_context_facts": [],
    "current_scene_timeline": None,
    "spontaneous_opportunities": [],
    "completed_run_markers": [],
    "last_raid_event": None,
    "recent_raid_contexts": [],
    "last_raid_ack_result": None,
    "last_raid_ack_error": None,
    "last_cheer_event": None,
    "last_cheer_ack_result": None,
    "last_cheer_dedupe_result": None,
    "shoutout_cooldowns": {},
    "promotion_executions_this_session": set(),
    "social_event_dedupe": set(),
    "processed_event_ids": set(),
    "cooldowns": {},
    "last_spontaneous_message": None,
    "last_stream_spontaneity_blocked_reason": None,
    "last_proactive_decision": None,
    "behavior_adaptation_state": {"entries": []},
    "last_feedback_application": None,
    "last_behavior_adaptation_decision": None,
    "last_behavior_correlation_id": "",
    "proposed_discourse_contribution": None,
    "current_stream_turn": None,
    "last_discourse_contribution": None,
    "last_discourse_blocked_reason": None,
    "discourse_contribution_timestamps": [],
    "last_promo_parse": None,
    "last_promo_rejected_reason": None,
    "last_promo_execution_decision": None,
    "last_promotion_outcome": None,
}

PERSISTENT_LIVE_FIELDS = (
    "viewer_identities",
    "viewer_linguistic_profiles",
    "promotion_preferences",
    "confirmed_memories",
)


@dataclass(slots=True)
class LiveSessionResetResult:
    session_id: str
    changed: bool
    reset_fields: list[str] = field(default_factory=list)
    persistent_fields_loaded: list[str] = field(default_factory=list)


class LiveSessionStateManager:
    """Starts a clean volatile live session without touching persistent stores."""

    def __init__(self, *, logger: Callable[[str], None] | None = None) -> None:
        self.current_session_id = ""
        self.logger = logger or (lambda message: print(message, flush=True))

    def begin_session(
        self,
        stream: Any,
        session_id: str | int,
        *,
        persistent_fields_loaded: list[str] | None = None,
        force: bool = False,
    ) -> LiveSessionResetResult:
        key = str(session_id or "").strip()
        if not key:
            raise ValueError("session_id is required")
        loaded = list(persistent_fields_loaded or PERSISTENT_LIVE_FIELDS)
        if key == self.current_session_id and not force:
            return LiveSessionResetResult(key, False, [], loaded)
        reset = []
        for name, default in VOLATILE_STREAM_DEFAULTS.items():
            setattr(stream, name, copy.deepcopy(default))
            reset.append(name)
        blocks = list(getattr(stream, "active_behavior_blocks", []) or [])
        stream.active_behavior_blocks = [
            item for item in blocks
            if isinstance(item, dict) and str(item.get("scope") or "current_stream") == "durable"
        ]
        self.current_session_id = key
        self.logger(
            "[HEBE][LIVE_SESSION_STATE] "
            f"session_id={key} reset_fields={','.join(reset)} "
            f"persistent_fields_loaded={','.join(loaded)}"
        )
        return LiveSessionResetResult(key, True, reset, loaded)

    def reset_for_replay(self, stream: Any, replay_session_id: str) -> LiveSessionResetResult:
        self.current_session_id = ""
        return self.begin_session(stream, replay_session_id, force=True)


__all__ = [
    "LiveSessionResetResult",
    "LiveSessionStateManager",
    "PERSISTENT_LIVE_FIELDS",
    "VOLATILE_STREAM_DEFAULTS",
]
