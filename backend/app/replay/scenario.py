from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.replay.clock import ScenarioClock, resolve_event_time
from app.replay.feature_flags import CognitiveFeatureFlags


SCENARIO_SCHEMA_VERSION = 1
EVENT_TYPES = {
    "owner_stt", "ambient_stt", "twitch_chat", "twitch_follow", "twitch_sub",
    "twitch_resub", "twitch_raid", "twitch_cheer", "stream_started",
    "stream_ended", "stream_metadata_changed", "advance_time", "restart_hebe",
    "maintenance", "configure_external_outcome", "game_research",
    "open_conversation",
    "propose_belief", "seed_known_belief", "correct_belief", "retrieve_beliefs",
    "add_legacy_memory_fact", "project_legacy_memory_fact", "add_vector_context",
    "resolve_game_run", "pause_game_run", "finish_game_run", "record_run_fact",
    "infer_run_fact", "correct_run_fact", "add_game_knowledge", "build_game_context",
    "resolve_person", "record_social_episode", "propose_social_hypothesis",
    "open_social_thread", "resolve_social_thread", "expire_social", "retrieve_social_context",
    "create_culture_candidate", "reinforce_culture", "use_culture", "select_culture", "social_opportunity",
    "consolidate_session", "learn_owner_preference", "learn_hebe_opinion", "observe_leo_language",
    "project_action_receipt", "validate_action_claim", "outgoing_raid", "incoming_raid",
    "build_continuity_context", "observe_schedule",
    "owner_voice_state", "speech_intent_candidate", "stream_scene", "companion_tick",
}


@dataclass(frozen=True, slots=True)
class ScenarioAssertion:
    assertion: str
    path: str = ""
    expected: Any = None
    matching: dict[str, Any] = field(default_factory=dict)
    count: int | None = None
    after_event: str = ""
    description: str = ""
    future_phase: str = ""

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> "ScenarioAssertion":
        data = dict(value or {})
        kind = str(data.get("assertion") or data.get("op") or "equals").strip()
        return cls(
            assertion=kind,
            path=str(data.get("path") or ""),
            expected=data.get("expected", data.get("equals")),
            matching=dict(data.get("matching") or data.get("matches") or {}),
            count=int(data["count"]) if data.get("count") is not None else None,
            after_event=str(data.get("after_event") or ""),
            description=str(data.get("description") or ""),
            future_phase=str(data.get("future_phase") or ""),
        )


@dataclass(frozen=True, slots=True)
class CognitiveReplayEvent:
    event_id: str
    event_type: str
    timestamp: float
    payload: dict[str, Any]
    assertions: tuple[ScenarioAssertion, ...] = ()


@dataclass(frozen=True, slots=True)
class CognitiveReplayScenario:
    schema_version: int
    scenario_id: str
    initial_time: float
    initial_database_fixture: str = ""
    seed: int = 0
    feature_flags: CognitiveFeatureFlags = field(default_factory=CognitiveFeatureFlags)
    model_fixtures: dict[str, Any] = field(default_factory=dict)
    research_fixtures: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    external_outcomes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    twitch_resolution_fixtures: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: tuple[CognitiveReplayEvent, ...] = ()
    final_assertions: tuple[ScenarioAssertion, ...] = ()
    required_layers: tuple[str, ...] = ("integration", "replay")
    source_path: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "CognitiveReplayScenario":
        scenario_path = Path(path).resolve()
        if scenario_path.suffix.lower() != ".json":
            raise ValueError("Cognitive Replay currently accepts versioned JSON scenarios")
        value = json.loads(scenario_path.read_text(encoding="utf-8"))
        return cls.from_value(value, source_path=str(scenario_path))

    @classmethod
    def from_value(cls, value: dict[str, Any], *, source_path: str = "") -> "CognitiveReplayScenario":
        data = dict(value or {})
        version = int(data.get("schema_version") or 0)
        if version != SCENARIO_SCHEMA_VERSION:
            raise ValueError(f"unsupported scenario schema_version: {version}")
        scenario_id = str(data.get("scenario_id") or "").strip()
        if not scenario_id:
            raise ValueError("scenario_id is required")
        clock = ScenarioClock.from_value(data.get("initial_time"))
        initial = clock.now()
        previous = initial
        seen: set[str] = set()
        events: list[CognitiveReplayEvent] = []
        for index, raw in enumerate(data.get("events") or []):
            row = dict(raw or {})
            event_type = str(row.pop("type", row.pop("event_type", ""))).strip()
            if event_type not in EVENT_TYPES:
                raise ValueError(f"unsupported replay event type: {event_type}")
            event_id = str(row.pop("event_id", "") or f"event-{index + 1:03d}")
            if event_id in seen:
                raise ValueError(f"duplicate event_id: {event_id}")
            seen.add(event_id)
            at = row.pop("at", row.pop("timestamp", previous))
            timestamp = resolve_event_time(at, initial=initial, previous=previous)
            previous = timestamp
            event_assertions = tuple(ScenarioAssertion.from_value(item) for item in row.pop("assertions", []) or [])
            events.append(CognitiveReplayEvent(event_id, event_type, timestamp, row, event_assertions))
        final_assertions = tuple(ScenarioAssertion.from_value(item) for item in data.get("final_assertions") or [])
        return cls(
            schema_version=version,
            scenario_id=scenario_id,
            initial_time=initial,
            initial_database_fixture=str(data.get("initial_database_fixture") or ""),
            seed=int(data.get("seed") or 0),
            feature_flags=CognitiveFeatureFlags.from_value(data.get("feature_flags")),
            model_fixtures=dict(data.get("model_fixtures") or {}),
            research_fixtures={str(k): [dict(row) for row in rows] for k, rows in dict(data.get("research_fixtures") or {}).items()},
            external_outcomes={str(k): [dict(row) for row in rows] for k, rows in dict(data.get("external_outcomes") or {}).items()},
            twitch_resolution_fixtures={
                str(k).lower(): dict(row) for k, row in dict(data.get("twitch_resolution_fixtures") or {}).items()
            },
            events=tuple(events),
            final_assertions=final_assertions,
            required_layers=tuple(str(item) for item in data.get("required_layers") or ("integration", "replay")),
            source_path=source_path,
        )
