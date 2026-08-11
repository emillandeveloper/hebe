from __future__ import annotations

import gc
import json
import os
import random
import sqlite3
import time
import uuid
import weakref
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from contextlib import ExitStack, contextmanager

from app.core.runtime import HebeRuntime
from app.core.state import HebeState
from app.hebe_engine import HebeEngine
from app.replay.assertions import AssertionResult, evaluate
from app.replay.clock import ScenarioClock
from app.replay.fakes import (
    DeterministicOutcomeQueue,
    DeterministicEmbedder,
    FakeSTT,
    FakeTwitch,
    FakeWinAutomation,
    FixtureModel,
    FixtureResearchProvider,
    RecordingSpeech,
    UnexpectedFixtureCall,
)
from app.replay.probe import CognitiveStateProbe
from app.replay.report import STATUS_FAILED, STATUS_INCOMPLETE, STATUS_VERIFIED
from app.replay.scenario import CognitiveReplayEvent, CognitiveReplayScenario, ScenarioAssertion
from app.replay.workspace import ScenarioWorkspace
from app.stream.context_sync import StreamContextSyncService
from app.continuity.models import ConversationContext, ExpectedReply, ExpectedReplyType


_REAL_PERF_COUNTER = time.perf_counter


@dataclass(slots=True)
class EventRunResult:
    event_id: str
    event_type: str
    timestamp: float
    duration_seconds: float
    assertions: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScenarioRunResult:
    scenario_id: str
    status: str
    scenario_schema_version: int
    seed: int
    feature_flags: dict[str, bool]
    events_processed: int
    duration_seconds: float
    restart_count: int
    event_results: list[dict[str, Any]]
    final_assertions: list[dict[str, Any]]
    assertion_summary: dict[str, int]
    failures: list[dict[str, Any]]
    final_state: dict[str, Any]
    checkpoint_states: dict[str, dict[str, Any]]
    database: dict[str, Any]
    external_boundaries: dict[str, str]
    limitations: list[str]
    restart_evidence: list[dict[str, Any]]
    expected_future_gap: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CognitiveReplayRunner:
    """Runs scenario data through the production HebeEngine seams.

    Only external adapters are fake. Policy, cognition, domain transactions,
    persistence and FinalEmissionGate remain production implementations.
    """

    def __init__(self, *, workspace_root: str | Path | None = None, retain_workspace: bool = True) -> None:
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None
        self.retain_workspace = retain_workspace
        self.workspace: ScenarioWorkspace | None = None
        self.clock: ScenarioClock | None = None
        self.engine: HebeEngine | None = None
        self.outcomes: DeterministicOutcomeQueue | None = None
        self.model: FixtureModel | None = None
        self.intent_model: FixtureModel | None = None
        self.research_provider: FixtureResearchProvider | None = None
        self.twitch: FakeTwitch | None = None
        self.speech_requests: list[dict[str, Any]] = []
        self.final_emissions: list[dict[str, Any]] = []
        self.restart_evidence: list[dict[str, Any]] = []
        self._instance_generation = 0
        self._active_scenario: CognitiveReplayScenario | None = None
        self._old_embedder: Any = None
        self._id_counter = 0

    def run(self, scenario: CognitiveReplayScenario | str | Path) -> ScenarioRunResult:
        if not isinstance(scenario, CognitiveReplayScenario):
            scenario = CognitiveReplayScenario.load(scenario)
        self._active_scenario = scenario
        started = _REAL_PERF_COUNTER()
        random.seed(scenario.seed)
        self.clock = ScenarioClock(scenario.initial_time)
        self.outcomes = DeterministicOutcomeQueue(scenario.external_outcomes)
        self.model = FixtureModel(scenario.model_fixtures, label="conversation_model")
        self.intent_model = FixtureModel(scenario.model_fixtures, label="intent_model")
        self.research_provider = FixtureResearchProvider(scenario.research_fixtures)
        root = self.workspace_root / scenario.scenario_id if self.workspace_root else None
        self.workspace = ScenarioWorkspace(
            scenario.scenario_id,
            root=root,
            database_fixture=self._resolve_database_fixture(scenario),
        )
        self.workspace.activate()
        from app.cognitive.memory import embeddings
        self._old_embedder = embeddings._default_embedder
        embeddings.set_default_embedder(DeterministicEmbedder())
        checkpoints: dict[str, dict[str, Any]] = {}
        event_results: list[EventRunResult] = []
        final_results: list[AssertionResult] = []
        limitations = [
            "datetime.now() reads in legacy persistence remain wall-clock based; behavioral TTL/cooldown time.time() reads are controlled during replay dispatch",
            "faster-whisper audio decoding is outside the cognitive replay boundary and requires its separate integration suite",
        ]
        try:
            self._record_replay_metadata(scenario)
            self._create_engine()
            for event in scenario.events:
                event_started = _REAL_PERF_COUNTER()
                error = ""
                assertion_results: list[AssertionResult] = []
                try:
                    self.clock.move_to(event.timestamp)
                    with self._deterministic_context():
                        self._dispatch(event)
                    snapshot = self._probe().snapshot(self.engine).to_dict()
                    checkpoints[event.event_id] = snapshot
                    assertion_results.extend(evaluate(item, snapshot) for item in event.assertions)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    snapshot = self._safe_snapshot()
                    checkpoints[event.event_id] = snapshot
                event_results.append(EventRunResult(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    timestamp=event.timestamp,
                    duration_seconds=round(_REAL_PERF_COUNTER() - event_started, 6),
                    assertions=[item.to_dict() for item in assertion_results],
                    error=error,
                    state=snapshot,
                ))
                if error:
                    break
            final_state = self._safe_snapshot()
            for assertion in scenario.final_assertions:
                target = checkpoints.get(assertion.after_event, final_state) if assertion.after_event else final_state
                final_results.append(evaluate(assertion, target))
            all_assertions = [
                item for event in event_results for item in event.assertions
            ] + [item.to_dict() for item in final_results]
            failed_assertions = [item for item in all_assertions if not item.get("passed") and not item.get("skipped")]
            skipped_assertions = [item for item in all_assertions if item.get("skipped")]
            errors = [item for item in event_results if item.error]
            expected_future_gap = bool(skipped_assertions and not failed_assertions and not errors)
            if errors or failed_assertions:
                status = STATUS_FAILED
            elif skipped_assertions:
                status = STATUS_INCOMPLETE
            else:
                status = STATUS_VERIFIED
            failures = [
                {"event_id": item.event_id, "path": "", "reason": item.error}
                for item in errors
            ] + [
                {"event_id": item.get("after_event") or "final", "path": item.get("path"), "reason": item.get("reason")}
                for item in failed_assertions
            ]
            summary = {
                "passed": sum(1 for item in all_assertions if item.get("passed") and not item.get("skipped")),
                "failed": len(failed_assertions),
                "skipped": len(skipped_assertions),
            }
            return ScenarioRunResult(
                scenario_id=scenario.scenario_id,
                status=status,
                scenario_schema_version=scenario.schema_version,
                seed=scenario.seed,
                feature_flags=scenario.feature_flags.to_dict(),
                events_processed=len(event_results),
                duration_seconds=round(_REAL_PERF_COUNTER() - started, 6),
                restart_count=len(self.restart_evidence),
                event_results=[asdict(item) for item in event_results],
                final_assertions=[item.to_dict() for item in final_results],
                assertion_summary=summary,
                failures=failures,
                final_state=final_state,
                checkpoint_states=checkpoints,
                database={
                    "path": str(self.workspace.db_path),
                    "type": "fresh" if not scenario.initial_database_fixture else "copied_fixture",
                    "schema_migrations": self._applied_schema_migrations(),
                },
                external_boundaries={
                    "twitch": "fake", "tts_audio": "fake", "desktop": "fake",
                    "research": "fixture", "model": "fixture", "network": "none",
                },
                limitations=limitations,
                restart_evidence=list(self.restart_evidence),
                expected_future_gap=expected_future_gap,
            )
        finally:
            self._dispose_engine()
            if not self.retain_workspace:
                self.workspace.cleanup()
            else:
                self.workspace.deactivate()
            from app.cognitive.memory import embeddings
            embeddings._default_embedder = self._old_embedder

    def _create_engine(self) -> None:
        assert self.outcomes and self.model and self.intent_model and self.clock
        self._instance_generation += 1
        twitch = FakeTwitch(
            self.outcomes,
            self._active_scenario.twitch_resolution_fixtures if self._active_scenario else None,
        )
        speech = RecordingSpeech(self.outcomes, self.speech_requests)
        win = FakeWinAutomation(self.outcomes)
        stt = FakeSTT()
        state = HebeState()
        state.tts_enabled = False
        actions = SimpleNamespace(
            open_app_from_text=lambda text: win.open_app(str(text)),
            store_memory_from_text=lambda _text: False,
        )
        runtime = HebeRuntime(
            stt=stt,
            llm=self.model,
            intent_llm=self.intent_model,
            win=win,
            actions=actions,
            tools=SimpleNamespace(),
            speak=speech,
            state=state,
            twitch=twitch,
            twitch_events=SimpleNamespace(push_event_callback=None, start=lambda: False, stop=lambda: None),
            twitch_chat_bot=SimpleNamespace(
                enabled=False,
                bot_username=twitch.bot_username,
                is_connected=True,
                ambient_message_callback=None,
                message_callback=None,
                social_event_callback=None,
                start=lambda: False,
                stop=lambda: None,
            ),
            stt_enabled=False,
        )
        env = {
            "HEBE_TTS_ENABLED": "false",
            "HEBE_STT_ENABLED": "false",
            "HEBE_GAME_RESEARCH_ENABLED": "false",
            "HEBE_WEB_LOOKUP_ENABLED": "false",
            "HEBE_STREAM_AMBIENT_STT_ENABLED": "true",
            "HEBE_AUTO_ENABLE_STREAM_WHEN_LIVE": "true",
            "HEBE_STREAM_OUTPUT_MODE": "twitch_chat_only",
            "HEBE_COGNITIVE_REPLAY_ENABLED": "true",
            "HEBE_CONVERSATION_CONTINUITY_V2": str(
                bool(self._active_scenario and self._active_scenario.feature_flags.conversation_continuity_v2)
            ).lower(),
            "HEBE_CONVERSATION_CONTINUITY_SHADOW": "true",
        }
        with patch.dict(os.environ, env, clear=False), self._deterministic_context():
            engine = HebeEngine(runtime=runtime, use_wakeword=True, say_hello=False)
        engine.stream_context_sync = StreamContextSyncService(twitch_api=twitch, now_fn=self.clock.now)
        self._install_research_fixture(engine)
        original_emit = engine._emit_final_response

        def recorded_emit(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = original_emit(*args, **kwargs)
            self.final_emissions.append(dict(result or {}))
            return result

        engine._emit_final_response = recorded_emit  # type: ignore[method-assign]
        engine._replay_instance_generation = self._instance_generation
        self.engine = engine
        self.twitch = twitch

    def _install_research_fixture(self, engine: HebeEngine) -> None:
        if not (self._active_scenario and self._active_scenario.research_fixtures):
            return
        provider = self.research_provider
        intelligence = getattr(engine, "game_intelligence", None)
        if intelligence is not None:
            intelligence.provider = provider
            intelligence.provider_name = "replay_fixture"
            intelligence.provider_configured = True
        legacy = getattr(engine, "game_research", None)
        if legacy is not None:
            legacy.search_provider = provider

    def _dispose_engine(self) -> None:
        engine = self.engine
        self.engine = None
        self.twitch = None
        if engine is not None:
            try:
                engine.stop()
            finally:
                research = getattr(engine, "game_intelligence", None)
                executor = getattr(research, "_executor", None)
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                del engine
                gc.collect()

    def _restart(self, event: CognitiveReplayEvent) -> None:
        assert self.engine is not None and self.workspace is not None
        before = self._probe().snapshot(self.engine).to_dict()
        old_id = id(self.engine)
        old_ref = weakref.ref(self.engine)
        self._dispose_engine()
        self._create_engine()
        gc.collect()
        after = self._probe().snapshot(self.engine).to_dict()
        self.restart_evidence.append({
            "event_id": event.event_id,
            "old_engine_id": old_id,
            "new_engine_id": id(self.engine),
            "old_engine_collected": old_ref() is None,
            "same_database": str(self.workspace.db_path),
            "before_persisted_counts": before.get("database_watermarks", {}).get("counts", {}),
            "after_persisted_counts": after.get("database_watermarks", {}).get("counts", {}),
            "volatile_state_recreated": before.get("runtime") != after.get("runtime") or old_id != id(self.engine),
        })

    def _dispatch(self, event: CognitiveReplayEvent) -> None:
        assert self.engine is not None and self.clock is not None and self.twitch is not None and self.outcomes is not None
        payload = dict(event.payload)
        event_type = event.event_type
        if event_type == "owner_stt":
            self.engine.ingest_owner_stt(str(payload.get("text") or ""), stt_metadata={
                "replay_event_id": str(payload.get("input_event_id") or event.event_id),
                "authority": str(payload.get("authority") or "owner"),
                "source_context": str(payload.get("source_context") or "owner_live_control"),
            })
            return
        if event_type == "open_conversation":
            context_kind = ConversationContext(str(payload.get("context_kind") or "owner_live_control"))
            if payload.get("context_id"):
                context_id = str(payload["context_id"])
            elif context_kind in {ConversationContext.OWNER_LOCAL, ConversationContext.OWNER_LIVE_CONTROL}:
                _resolved_kind, context_id = self.engine._conversation_context_for_owner_stt()
            else:
                context_id = "leo_ui" if context_kind == ConversationContext.PRIVATE_UI else "stream_public"
            expected = ExpectedReply(
                type=ExpectedReplyType(str(payload.get("expected_reply_type") or "free_response")),
                allowed_sources=tuple(payload.get("allowed_sources") or ("owner_stt",)),
                allowed_participant=str(payload.get("allowed_participant") or "leo"),
                semantic_constraints=dict(payload.get("semantic_constraints") or {}),
                candidate_refs=tuple(str(item) for item in payload.get("candidate_refs") or ()),
                expires_at=self.clock.now() + float(payload.get("ttl_seconds") or 60.0),
            )
            self.engine.conversation_continuity.open_conversation(
                context_kind=context_kind, context_id=context_id,
                topic=str(payload.get("topic") or "replay_handoff"),
                origin_event_id=event.event_id, expected_reply=expected,
                domain_payload=dict(payload.get("domain_payload") or {}),
            )
            return
        if event_type == "ambient_stt":
            self.engine.ingest_ambient_stt(str(payload.get("text") or ""))
            return
        if event_type == "twitch_chat":
            login = str(payload.get("login") or payload.get("user_login") or "viewer")
            display = str(payload.get("display_name") or login)
            user_id = str(payload.get("user_id") or "")
            message_id = str(payload.get("message_id") or event.event_id)
            self.twitch.remember_identity(user_id=user_id, login=login, display_name=display)
            tags = {
                "id": message_id,
                "user-id": user_id,
                "display-name": display,
                **dict(payload.get("irc_tags") or {}),
            }
            reply = dict(payload.get("reply") or {})
            tags.update({
                "reply-parent-user-login": reply.get("parent_login") or "",
                "reply-parent-display-name": reply.get("parent_display_name") or "",
                "reply-parent-msg-id": reply.get("parent_message_id") or "",
                "reply-parent-msg-body": reply.get("parent_message") or "",
            })
            self.engine.ingest_normalized_twitch_chat(
                username=login,
                display_name=display,
                text=str(payload.get("text") or ""),
                channel=str(payload.get("channel") or self.twitch.channel_name),
                irc_tags=tags,
                normalized_fields={
                    "event_id": message_id,
                    "message_id": message_id,
                    "twitch_user_id": user_id,
                    "mentions_hebe": bool(payload.get("mentions_hebe")),
                    "source": "replay_normalized_irc",
                },
            )
            return
        if event_type in {"twitch_follow", "twitch_sub", "twitch_resub", "twitch_raid", "twitch_cheer"}:
            login = str(payload.get("login") or payload.get("user_login") or "viewer")
            display = str(payload.get("display_name") or login)
            user_id = str(payload.get("user_id") or "")
            self.twitch.remember_identity(user_id=user_id, login=login, display_name=display)
            normalized = {
                "event_id": str(payload.get("message_id") or event.event_id),
                "message_id": str(payload.get("message_id") or event.event_id),
                "user_id": user_id,
                "twitch_user_id": user_id,
                "user_login": login,
                "username": login,
                "display_name": display,
                "source": str(payload.get("source") or "eventsub"),
                "passive_eventsub": bool(payload.get("passive_eventsub", True)),
                "visible_public": bool(payload.get("visible_public", False)),
                **payload,
            }
            mapped = {
                "twitch_follow": "twitch_follow",
                "twitch_sub": "twitch_sub",
                "twitch_resub": "twitch_sub",
                "twitch_raid": "twitch_raid",
                "twitch_cheer": "twitch_cheer",
            }[event_type]
            if event_type == "twitch_resub":
                normalized["is_resub"] = True
                normalized.setdefault("months", payload.get("cumulative_months", 1))
            from app.cognitive.scheduler import InternalEvent

            self.engine.process_internal_event(InternalEvent(mapped, normalized, self.clock.iso()))
            return
        if event_type == "stream_started":
            metadata = {
                "is_live": True,
                "stream_id": payload.get("session_id") or payload.get("stream_id") or event.event_id,
                "started_at": payload.get("started_at") or self.clock.iso(),
                **payload,
            }
            self.twitch.configure_stream_metadata(metadata)
            self.engine.ingest_stream_lifecycle("stream_started", metadata, created_at=self.clock.iso())
            return
        if event_type == "stream_ended":
            self.twitch.configure_stream_metadata({**payload, "is_live": False})
            self.engine.ingest_stream_lifecycle("stream_ended", payload, created_at=self.clock.iso())
            return
        if event_type == "stream_metadata_changed":
            self.engine.ingest_stream_metadata(payload)
            return
        if event_type == "advance_time":
            seconds = float(payload.get("seconds") or 0.0)
            if seconds:
                self.clock.advance(seconds)
            self._run_maintenance()
            return
        if event_type == "maintenance":
            self._run_maintenance()
            return
        if event_type == "restart_hebe":
            self._restart(event)
            return
        if event_type == "configure_external_outcome":
            operation = str(payload.pop("operation", ""))
            if not operation:
                raise ValueError("configure_external_outcome requires operation")
            self.outcomes.configure_next(operation, payload)
            return
        if event_type == "game_research":
            title = str(payload.get("game") or payload.get("game_title") or "").strip()
            if not title:
                raise ValueError("game_research requires game")
            self.engine.game_intelligence.get_or_build_dossier(
                game_title=title,
                platform=str(payload.get("platform") or ""),
                version=str(payload.get("version") or ""),
                force_refresh=bool(payload.get("force_refresh", True)),
            )
            return
        raise ValueError(f"unsupported event: {event_type}")

    def _run_maintenance(self) -> None:
        assert self.engine is not None
        self.engine._active_pending_clarification()
        self.engine._get_pending_conversation_turn()
        continuity = getattr(self.engine, "conversation_continuity", None)
        if continuity is not None:
            continuity.expire_due()
        self.engine.poll_internal_events()
        self.engine.poll_stream_presence()

    def _probe(self) -> CognitiveStateProbe:
        assert self.workspace is not None and self.outcomes is not None and self.model is not None and self.intent_model is not None and self.research_provider is not None
        return CognitiveStateProbe(
            connection_factory=lambda: self.workspace.connection(readonly=True),
            actions=self.outcomes.attempts,
            speech_requests=self.speech_requests,
            final_emissions=self.final_emissions,
            model_calls=[*self.model.calls, *self.intent_model.calls],
            research_calls=self.research_provider.calls,
        )

    def _safe_snapshot(self) -> dict[str, Any]:
        if self.engine is None:
            return {}
        try:
            return self._probe().snapshot(self.engine).to_dict()
        except Exception as exc:
            return {"probe_error": f"{type(exc).__name__}: {exc}"}

    def _record_replay_metadata(self, scenario: CognitiveReplayScenario) -> None:
        assert self.workspace is not None
        conn = self.workspace.connection()
        try:
            conn.execute(
                """
                INSERT INTO cognitive_replay_metadata(scenario_id, schema_version, seed, last_run_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(scenario_id) DO UPDATE SET schema_version=excluded.schema_version, seed=excluded.seed, last_run_at=excluded.last_run_at
                """,
                (scenario.scenario_id, scenario.schema_version, scenario.seed, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def _applied_schema_migrations(self) -> list[dict[str, Any]]:
        assert self.workspace is not None
        conn = self.workspace.connection(readonly=True)
        conn.row_factory = sqlite3.Row
        try:
            existing = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
            ).fetchone()
            if not existing:
                return []
            return [dict(row) for row in conn.execute(
                "SELECT component,version,name,checksum,applied_at FROM schema_migrations ORDER BY component,version"
            )]
        finally:
            conn.close()

    @staticmethod
    def _resolve_database_fixture(scenario: CognitiveReplayScenario) -> str:
        if not scenario.initial_database_fixture:
            return ""
        path = Path(scenario.initial_database_fixture)
        if not path.is_absolute() and scenario.source_path:
            path = Path(scenario.source_path).parent / path
        return str(path.resolve())

    @contextmanager
    def _deterministic_context(self):
        assert self.clock is not None
        with ExitStack() as stack:
            stack.enter_context(patch("time.time", self.clock.now))
            stack.enter_context(patch("time.time_ns", self._deterministic_time_ns))
            stack.enter_context(patch("time.monotonic", self.clock.monotonic))
            stack.enter_context(patch("uuid.uuid4", self._deterministic_uuid4))
            yield

    def _deterministic_uuid4(self) -> uuid.UUID:
        self._id_counter += 1
        scenario = self._active_scenario
        identity = f"{getattr(scenario, 'scenario_id', 'replay')}:{getattr(scenario, 'seed', 0)}:{self._id_counter}"
        return uuid.uuid5(uuid.NAMESPACE_URL, identity)

    def _deterministic_time_ns(self) -> int:
        assert self.clock is not None
        self._id_counter += 1
        return int(self.clock.now() * 1_000_000_000) + self._id_counter
