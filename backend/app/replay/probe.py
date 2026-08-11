from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, set):
        return sorted(_plain(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return str(value)


@dataclass(slots=True)
class CognitiveStateSnapshot:
    runtime: dict[str, Any] = field(default_factory=dict)
    stream_session: dict[str, Any] = field(default_factory=dict)
    current_scene: dict[str, Any] = field(default_factory=dict)
    pending: dict[str, Any] = field(default_factory=dict)
    open_threads: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    beliefs: list[dict[str, Any]] = field(default_factory=list)
    game_state: dict[str, Any] = field(default_factory=dict)
    social_state: dict[str, Any] = field(default_factory=dict)
    promotion_profiles: list[dict[str, Any]] = field(default_factory=list)
    actions: dict[str, Any] = field(default_factory=dict)
    receipts: list[dict[str, Any]] = field(default_factory=list)
    emitted_outputs: list[dict[str, Any]] = field(default_factory=list)
    final_emission_results: list[dict[str, Any]] = field(default_factory=list)
    database_watermarks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


class CognitiveStateProbe:
    """Read-only state projection used by assertions and reports."""

    SAFE_COUNT_TABLES = (
        "chat_log", "memory_facts", "memory_chunks", "stream_sessions",
        "stream_chat_messages", "stream_events", "live_session_timeline",
        "promotion_events", "viewer_promotion_profiles", "schema_migrations",
    )

    def __init__(
        self,
        *,
        connection_factory: Callable[[], sqlite3.Connection],
        actions: list[dict[str, Any]],
        speech_requests: list[dict[str, Any]],
        final_emissions: list[dict[str, Any]],
        model_calls: list[dict[str, Any]],
        research_calls: list[dict[str, Any]],
    ) -> None:
        self.connection_factory = connection_factory
        self.actions = actions
        self.speech_requests = speech_requests
        self.final_emissions = final_emissions
        self.model_calls = model_calls
        self.research_calls = research_calls

    def snapshot(self, engine: Any) -> CognitiveStateSnapshot:
        state = getattr(getattr(engine, "runtime", None), "state", None)
        stream = getattr(state, "stream", None)
        pending = getattr(state, "pending_clarification", None)
        pending_turn = getattr(state, "pending_conversation_turn", None)
        rows = self._db_rows()
        trace = dict(getattr(engine, "_last_policy_trace", {}) or {})
        cognitive = dict(getattr(engine, "_last_cognitive_trace", {}) or {})
        runtime = {
            "mode": getattr(state, "mode", None),
            "hebe_sleeping": bool(getattr(state, "hebe_sleeping", False)),
            "is_running": bool(getattr(state, "is_running", False)),
            "last_input_source": getattr(state, "last_input_source", None),
            "last_intent": getattr(state, "last_intent", None),
            "last_firewall": {
                "source": dict(getattr(engine, "_last_input_firewall", {}) or {}).get("source"),
                "authority": dict(getattr(engine, "_last_input_firewall", {}) or {}).get("authority"),
                "decision": dict(getattr(engine, "_last_input_firewall", {}) or {}).get("firewall_decision"),
                "reason": dict(getattr(engine, "_last_input_firewall", {}) or {}).get("reason"),
            },
            "last_policy": {
                "source": trace.get("source"),
                "authority": trace.get("authority"),
                "decision": trace.get("policy_decision"),
                "reason": trace.get("reason"),
            },
        }
        stream_session = {
            "enabled": bool(getattr(stream, "enabled", False)),
            "is_live": bool(getattr(stream, "is_live", False)),
            "live_status_known": bool(getattr(stream, "live_status_known", False)),
            "active_stream_session_id": getattr(stream, "active_stream_session_id", None),
            "last_transition": getattr(stream, "last_stream_live_transition", None),
            "title": getattr(stream, "current_stream_title", None),
            "game": getattr(stream, "current_game", None),
            "category": getattr(stream, "current_category", None),
        }
        scene_value = getattr(stream, "current_scene_timeline", None)
        scene = _plain(scene_value) if scene_value else {}
        game_run = getattr(state, "game_run_state", None)
        game_state = _plain(game_run) if game_run is not None else {}
        game_state.update({
            "current_game": getattr(stream, "current_game", None),
            "current_objective": getattr(stream, "current_run_objective", None),
            "current_location": getattr(stream, "current_run_location", None),
            "recent_run_context_facts": _plain(list(getattr(stream, "recent_run_context_facts", []) or [])),
        })
        social_state = {
            "recent_active_users": list(getattr(stream, "recent_active_users", []) or []),
            "recent_chat_count": len(list(getattr(stream, "recent_chat_messages", []) or [])),
            "last_raid": _plain(getattr(stream, "last_raid_event", None) or {}),
            "last_cheer": _plain(getattr(stream, "last_cheer_event", None) or {}),
        }
        final_response = ""
        final_response = str(cognitive.get("final_response") or trace.get("final_response") or trace.get("hebe_response") or "")
        emitted = [self._minimal_emission(item) for item in self.final_emissions]
        return CognitiveStateSnapshot(
            runtime=runtime,
            stream_session=stream_session,
            current_scene=scene,
            pending={"clarification": _plain(pending), "conversation_turn": _plain(pending_turn)},
            open_threads=[],
            memory={"facts_count": rows["counts"].get("memory_facts", 0), "chunks_count": rows["counts"].get("memory_chunks", 0)},
            beliefs=[],
            game_state=game_state,
            social_state=social_state,
            promotion_profiles=rows["promotion_profiles"],
            actions={
                "attempts": _plain(self.actions),
                "speech_requests": [{"language": item.get("language"), "text_digest": _digest(item.get("text"))} for item in self.speech_requests],
                "model_calls": _plain(self.model_calls),
                "research_calls": _plain(self.research_calls),
            },
            receipts=rows["promotion_events"],
            emitted_outputs=emitted,
            final_emission_results=emitted,
            database_watermarks={
                "counts": rows["counts"],
                "schema_migrations": rows["schema_migrations"],
                "final_response_digest": _digest(final_response),
                "final_response_present": bool(final_response),
            },
        )

    def _db_rows(self) -> dict[str, Any]:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            existing = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            counts = {
                table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
                for table in self.SAFE_COUNT_TABLES if table in existing
            }
            profiles = []
            if "viewer_promotion_profiles" in existing:
                profiles = [dict(row) for row in conn.execute(
                    "SELECT twitch_user_id, current_login, display_name, auto_promo_mode, created_by, last_promoted_stream_id, owner_locked, active FROM viewer_promotion_profiles ORDER BY twitch_user_id"
                )]
            promotions = []
            if "promotion_events" in existing:
                promotions = [dict(row) for row in conn.execute(
                    "SELECT id, stream_session_id, source_event_id, requested_by, resolved_twitch_user_id, resolved_login, trigger_type, execution_status, twitch_message_id, failure_reason FROM promotion_events ORDER BY created_at, id"
                )]
            migrations = []
            if "schema_migrations" in existing:
                migrations = [dict(row) for row in conn.execute(
                    "SELECT component, version, name, checksum, applied_at FROM schema_migrations ORDER BY component, version"
                )]
            return {"counts": counts, "promotion_profiles": profiles, "promotion_events": promotions, "schema_migrations": migrations}
        finally:
            conn.close()

    @staticmethod
    def _minimal_emission(item: dict[str, Any]) -> dict[str, Any]:
        data = dict(item or {})
        return {
            "event_id": data.get("event_id"),
            "emitted": bool(data.get("emitted")),
            "route": data.get("output_route") or data.get("route"),
            "targets": list(data.get("output_targets") or data.get("targets") or []),
            "reason": data.get("reason") or data.get("suppress_reason") or "",
            "text_digest": _digest(data.get("final_response") or data.get("text")),
            "text_present": bool(data.get("final_response") or data.get("text")),
        }


def _digest(value: Any) -> str:
    import hashlib

    text = str(value or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] if text else ""
