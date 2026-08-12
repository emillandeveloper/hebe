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
from app.epistemics.models import BeliefStatus, EvidenceRef, EvidenceRelation, RetrievalRequest


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
        self._belief_aliases: dict[str, str] = {}
        self._game_aliases: dict[str, str] = {}
        self._person_aliases: dict[str, str] = {}
        self._episode_aliases: dict[str, str] = {}
        self._culture_aliases: dict[str, str] = {}

    def run(self, scenario: CognitiveReplayScenario | str | Path) -> ScenarioRunResult:
        if not isinstance(scenario, CognitiveReplayScenario):
            scenario = CognitiveReplayScenario.load(scenario)
        self._active_scenario = scenario
        self._belief_aliases = {}
        self._game_aliases = {}
        self._person_aliases = {}
        self._episode_aliases = {}
        self._culture_aliases = {}
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
            "HEBE_BELIEF_V2_READS": str(bool(self._active_scenario and self._active_scenario.feature_flags.belief_v2_reads)).lower(),
            "HEBE_BELIEF_V2_WRITES": str(bool(self._active_scenario and self._active_scenario.feature_flags.belief_v2_writes)).lower(),
            "HEBE_GAME_CONTEXT_V2": str(bool(self._active_scenario and self._active_scenario.feature_flags.game_context_v2)).lower(),
            "HEBE_GAME_RUN_V2_READS": str(bool(self._active_scenario and self._active_scenario.feature_flags.game_run_v2_reads)).lower(),
            "HEBE_GAME_RUN_V2_WRITES": str(bool(self._active_scenario and self._active_scenario.feature_flags.game_run_v2_writes)).lower(),
            "HEBE_GAME_KNOWLEDGE_V2_READS": str(bool(self._active_scenario and self._active_scenario.feature_flags.game_knowledge_v2_reads)).lower(),
            "HEBE_GAME_KNOWLEDGE_V2_WRITES": str(bool(self._active_scenario and self._active_scenario.feature_flags.game_knowledge_v2_writes)).lower(),
            "HEBE_GAME_RESEARCH_MEMORY_FIRST": str(bool(self._active_scenario and self._active_scenario.feature_flags.game_research_memory_first)).lower(),
            "HEBE_SOCIAL_WORLD_V2": str(bool(self._active_scenario and self._active_scenario.feature_flags.social_world_v2)).lower(),
            "HEBE_SOCIAL_IDENTITY_V2": str(bool(self._active_scenario and self._active_scenario.feature_flags.social_identity_v2)).lower(),
            "HEBE_SOCIAL_EPISODE_WRITES_V2": str(bool(self._active_scenario and self._active_scenario.feature_flags.social_episode_writes_v2)).lower(),
            "HEBE_SOCIAL_RETRIEVAL_V2": str(bool(self._active_scenario and self._active_scenario.feature_flags.social_retrieval_v2)).lower(),
            "HEBE_SHARED_CULTURE_V2": str(bool(self._active_scenario and self._active_scenario.feature_flags.shared_culture_v2)).lower(),
            "HEBE_SOCIAL_THREAD_OPPORTUNITIES_V2": str(bool(self._active_scenario and self._active_scenario.feature_flags.social_thread_opportunities_v2)).lower(),
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
        resolver = getattr(engine,"game_context_resolver",None)
        if resolver is not None:
            resolver.research_service = intelligence
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

    def _record_canonical_belief_evidence(self, event: CognitiveReplayEvent, payload: dict[str, Any]) -> tuple[str,str,str]:
        """Materialize replay evidence in the same canonical timeline projection used by live experience."""
        assert self.workspace is not None and self.clock is not None
        source_event_id=str(payload.get("source_event_id") or event.event_id)
        source_record_type=str(payload.get("source_record_type") or "live_session_timeline")
        source_record_id=str(payload.get("source_record_id") or source_event_id)
        if source_record_type!="live_session_timeline":
            return source_event_id,source_record_type,source_record_id
        literal=dict(payload.get("literal_span") or {})
        raw_text=str(payload.get("text") or literal.get("excerpt") or "")
        conn=self.workspace.connection()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO live_session_timeline(
                    session_id,event_uid,event_type,event_ts,source,raw_text,normalized_text,speaker,
                    confidence,provenance,index_for_rag,payload_json,created_at,context_kind,
                    source_record_type,source_record_id,authority,literal_evidence_json,valid_from,
                    valid_until,schema_version
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    "cognitive-replay",source_event_id,"belief_evidence",self.clock.iso(),"cognitive_replay",
                    raw_text,raw_text,"leo" if str(payload.get("authority_class") or "")=="owner" else "",
                    float(payload.get("confidence") or 1.0),"deterministic_replay",0,"{}",self.clock.iso(),
                    str(payload.get("context_kind") or "owner_local"),"cognitive_replay_event",event.event_id,
                    str(payload.get("authority_class") or "extractor"),json.dumps(literal,ensure_ascii=False),
                    self.clock.now(),float(payload.get("valid_until") or 0),1,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return source_event_id,source_record_type,source_record_id

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
            domain_payload=dict(payload.get("domain_payload") or {})
            target_ref=str(domain_payload.pop("target_belief_ref","") or "")
            if target_ref:domain_payload["target_belief_id"]=self._belief_aliases.get(target_ref,target_ref)
            self.engine.conversation_continuity.open_conversation(
                context_kind=context_kind, context_id=context_id,
                topic=str(payload.get("topic") or "replay_handoff"),
                origin_event_id=event.event_id, expected_reply=expected,
                domain_payload=domain_payload,
            )
            return
        if event_type == "resolve_person":
            person,identity=self.engine.social_world.resolve_person(platform=str(payload.get("platform") or "twitch"),platform_user_id=str(payload.get("platform_user_id") or payload.get("user_id") or ""),login=str(payload.get("login") or ""),display_name=str(payload.get("display_name") or ""),source="cognitive_replay",stream_session_id=str(payload.get("stream_session_id") or ""));self._person_aliases[event.event_id]=person.person_id;return
        if event_type == "record_social_episode":
            refs=payload.get("participant_refs") or ();people=tuple(self._person_aliases.get(str(x),str(x)) for x in refs)
            result=self.engine.social_world.record_episode(episode_type=str(payload.get("episode_type") or "meaningful_interaction"),participant_ids=people,origin_event_id=event.event_id,summary=str(payload.get("summary") or payload.get("text") or ""),salience_reason=str(payload.get("salience_reason") or ""),relevance_seconds=float(payload.get("relevance_seconds") or 86400),retention_seconds=float(payload.get("retention_seconds") or 2592000),sensitivity=str(payload.get("sensitivity") or "normal"),retention_class=str(payload.get("retention_class") or "bounded"),retrieval_scope=str(payload.get("retrieval_scope") or "stream_public"),tone_observations=tuple(payload.get("tone_observations") or ()))
            if result:self._episode_aliases[event.event_id]=result.id
            return
        if event_type == "propose_social_hypothesis":
            ref=str(payload.get("person_ref") or "");person_id=self._person_aliases.get(ref,ref);payload.setdefault("authority_class","extractor");payload.setdefault("literal_span",{"excerpt":str(payload.get("text") or "")[:120]})
            source_event_id,source_record_type,source_record_id=self._record_canonical_belief_evidence(event,payload);evidence=EvidenceRef(source_event_id=source_event_id,source_record_type=source_record_type,source_record_id=source_record_id,relation=EvidenceRelation.SUPPORTS,observed_at=self.clock.now(),extractor="social_hypothesis_validator",extractor_version="v1",literal_span=dict(payload["literal_span"]))
            result=self.engine.social_world.propose_hypothesis(person_id,predicate=str(payload.get("predicate") or "interest.topic"),object_value=payload.get("object"),confidence=float(payload.get("confidence") or .6),evidence=evidence,sensitivity=str(payload.get("sensitivity") or "normal"),relevance_seconds=float(payload.get("relevance_seconds") or 2592000))
            if result:self._belief_aliases[event.event_id]=result.id
            return
        if event_type == "open_social_thread":
            ref=str(payload.get("person_ref") or "");person_id=self._person_aliases.get(ref,ref);self.engine.social_world.open_social_thread(person_id,thread_type=str(payload.get("thread_type") or "question_followup"),subject_ref=str(payload.get("subject_ref") or event.event_id),summary=str(payload.get("summary") or ""),origin_event_id=event.event_id,relevance_seconds=float(payload.get("relevance_seconds") or 86400),valid_seconds=float(payload.get("valid_seconds") or payload.get("relevance_seconds") or 86400),sensitivity=str(payload.get("sensitivity") or "normal"),priority=int(payload.get("priority") or 40));return
        if event_type == "resolve_social_thread":
            from app.continuity.models import OpenThreadStatus
            self.engine.social_world.resolve_social_thread(str(payload.get("subject_ref") or ""),event_id=event.event_id,status=OpenThreadStatus(str(payload.get("status") or "RESOLVED")));return
        if event_type == "expire_social":self.engine.social_world.expire_social_threads(event.event_id);return
        if event_type == "retrieve_social_context":
            ref=str(payload.get("person_ref") or "");person_id=self._person_aliases.get(ref,ref);self.engine.social_world.retrieve_social_context(person_id,purpose=str(payload.get("purpose") or "social_greeting"),retrieval_scope=str(payload.get("retrieval_scope") or "stream_public"),topic=str(payload.get("topic") or ""),scene_tone=str(payload.get("scene_tone") or "casual"));return
        if event_type == "create_culture_candidate":
            refs=payload.get("participant_refs") or ();people=[self._person_aliases.get(str(x),str(x)) for x in refs];episode_ref=str(payload.get("episode_ref") or "");item=self.engine.social_world.create_culture_candidate(label=str(payload.get("label") or "callback"),meaning=str(payload.get("meaning") or ""),participant_ids=people,origin_episode_id=self._episode_aliases.get(episode_ref,episode_ref),event_id=event.event_id,tone=str(payload.get("tone") or "playful"),owner_confirmed=bool(payload.get("owner_confirmed")));self._culture_aliases[event.event_id]=item["id"];return
        if event_type == "reinforce_culture":
            ref=str(payload.get("culture_ref") or "");item_id=self._culture_aliases.get(ref,ref);episode_ref=str(payload.get("episode_ref") or "");self.engine.social_world.reinforce_culture(item_id,event_id=event.event_id,episode_id=self._episode_aliases.get(episode_ref,episode_ref),reaction=str(payload.get("reaction") or "positive"),weight=float(payload.get("weight") or 1),authority=str(payload.get("authority") or "interaction"));return
        if event_type == "use_culture":
            ref=str(payload.get("culture_ref") or "");self.engine.social_world.use_culture(self._culture_aliases.get(ref,ref),event_id=event.event_id,cooldown_seconds=float(payload.get("cooldown_seconds") or 3600));return
        if event_type == "select_culture":
            ref=str(payload.get("person_ref") or "");self.engine.social_world.select_culture(self._person_aliases.get(ref,ref),topic=str(payload.get("topic") or ""),scene_tone=str(payload.get("scene_tone") or "casual"));return
        if event_type == "social_opportunity":
            ref=str(payload.get("person_ref") or "");self.engine.social_world.opportunities(self._person_aliases.get(ref,ref),scene_suitable=bool(payload.get("scene_suitable",True)));return
        if event_type == "resolve_game_run":
            result=self.engine.game_run_service.resolve(game=str(payload.get("game") or "Unknown Game"),stream_session_id=str(payload.get("stream_session_id") or event.event_id),source_event_id=str(payload.get("source_event_id") or event.event_id),owner_id=str(payload.get("owner_id") or "leo"),run_kind=str(payload.get("run_kind") or "unknown"),rules=dict(payload.get("rules") or {}),explicit_new=bool(payload.get("explicit_new")),explicit_continue=bool(payload.get("explicit_continue")))
            if result.active_run:self._game_aliases[event.event_id]=result.active_run.id
            return
        if event_type == "pause_game_run":
            ref=str(payload.get("run_ref") or "");run_id=self._game_aliases.get(ref,ref);self.engine.game_run_service.pause(run_id,stream_session_id=str(payload.get("stream_session_id") or "session"),event_id=event.event_id);return
        if event_type == "finish_game_run":
            from app.game_context_v2.models import GameRunStatus
            ref=str(payload.get("run_ref") or "");run_id=self._game_aliases.get(ref,ref);self.engine.game_run_service.finish(run_id,status=GameRunStatus(str(payload.get("status") or "COMPLETED")),event_id=event.event_id);return
        if event_type in {"record_run_fact","infer_run_fact"}:
            payload.setdefault("authority_class","owner" if event_type=="record_run_fact" else "extractor")
            source_event_id,source_record_type,source_record_id=self._record_canonical_belief_evidence(event,payload)
            evidence=EvidenceRef(source_event_id=source_event_id,source_record_type=source_record_type,source_record_id=source_record_id,relation=EvidenceRelation.SUPPORTS,weight=float(payload.get("weight") or 1),observed_at=self.clock.now(),extractor="owner_run_statement" if event_type=="record_run_fact" else str(payload.get("extractor") or "run_extractor"),extractor_version=str(payload.get("extractor_version") or "v1"),literal_span=dict(payload.get("literal_span") or {}))
            ref=str(payload.get("run_ref") or "");run_id=self._game_aliases.get(ref,ref)
            result=self.engine.game_run_service.record_fact(run_id,subject_ref=str(payload.get("subject_ref") or "run"),predicate=str(payload.get("predicate") or "state"),object_value=payload.get("object"),evidence=evidence,event_type=str(payload.get("run_event_type") or "notable_run_event"),owner_confirmed=event_type=="record_run_fact",confidence=float(payload.get("confidence") or .6),entailment_valid=bool(payload.get("entailment_valid",True)))
            if result:self._belief_aliases[event.event_id]=result.id
            return
        if event_type == "correct_run_fact":
            payload.setdefault("authority_class","owner");payload.setdefault("literal_span",{"start":0,"end":len(str(payload.get("text") or "")),"excerpt":str(payload.get("text") or "")[:80]})
            source_event_id,source_record_type,source_record_id=self._record_canonical_belief_evidence(event,payload);old_ref=str(payload.get("belief_ref") or "");belief_id=self._belief_aliases.get(old_ref,old_ref)
            evidence=EvidenceRef(source_event_id=source_event_id,source_record_type=source_record_type,source_record_id=source_record_id,relation=EvidenceRelation.CORRECTS,observed_at=self.clock.now(),extractor="game_run_domain_correction",extractor_version="v1",literal_span=dict(payload["literal_span"]))
            if bool(payload.get("via_conversation")):
                resolution=self.engine.conversation_continuity.resolve_input(context_kind="owner_local",context_id="leo_local",source="owner_stt",participant="leo",authority="owner",text=str(payload.get("text") or ""),event_id=event.event_id,wake=False,consume=True)
                self.engine._last_continuity_resolution=resolution.to_dict();self.engine._apply_game_run_correction_continuation(resolution,event_id=event.event_id,text=str(payload.get("text") or ""));return
            result=self.engine.game_run_service.correct_fact(belief_id,object_value=payload.get("object"),evidence=evidence);self._belief_aliases[event.event_id]=result.id;return
        if event_type == "add_game_knowledge":
            identity=self.engine.game_v2_repository.resolve_identity(str(payload.get("game") or "Unknown Game"));payload.setdefault("authority_class","domain_validator")
            source_event_id,source_record_type,source_record_id=self._record_canonical_belief_evidence(event,payload)
            evidence=EvidenceRef(source_event_id=source_event_id,source_record_type=source_record_type,source_record_id=source_record_id,observed_at=self.clock.now(),extractor="validated_fixture",extractor_version="v1",literal_span=dict(payload.get("literal_span") or {"excerpt":str(payload.get("text") or "validated fixture")}))
            result=self.engine.game_knowledge_v2_service.add_validated(game_id=identity.game_id,subject_ref=str(payload.get("subject_ref") or "game"),predicate=str(payload.get("predicate") or "fact"),object_value=payload.get("object"),confidence=float(payload.get("confidence") or .9),evidence=evidence,source_type=str(payload.get("source_type") or "curated"),source_quality=str(payload.get("source_quality") or "validated"),spoiler_class=str(payload.get("spoiler_class") or "safe_general_mechanic"),version_tag=str(payload.get("version_tag") or ""))
            if result:self._belief_aliases[event.event_id]=result.id
            return
        if event_type == "build_game_context":
            ref=str(payload.get("run_ref") or "");run_id=self._game_aliases.get(ref,ref)
            self.engine.game_context_resolver.build(game=str(payload.get("game") or "Unknown Game"),purpose=str(payload.get("purpose") or "run_context"),stream_session_id=str(payload.get("stream_session_id") or ""),run_id=run_id,subject_ref=str(payload.get("subject_ref") or ""),predicate=str(payload.get("predicate") or ""),question_type=str(payload.get("question_type") or ""),query_intent=str(payload.get("query_intent") or ""),spoiler_ceiling=str(payload.get("spoiler_ceiling") or "strict"),required_confidence=float(payload.get("required_confidence") or .6),event_id=event.event_id,allow_research=bool(payload.get("allow_research")),historical_run=bool(payload.get("historical_run")),scene_assertions=tuple(payload.get("scene_assertions") or ()));return
        if event_type in {"propose_belief", "seed_known_belief"}:
            payload.setdefault("authority_class","owner" if event_type=="seed_known_belief" else "extractor")
            source_event_id,source_record_type,source_record_id=self._record_canonical_belief_evidence(event,payload)
            relation = EvidenceRelation(str(payload.get("relation") or "SUPPORTS"))
            evidence = EvidenceRef(
                source_event_id=source_event_id,
                source_record_type=source_record_type,
                source_record_id=source_record_id,
                relation=relation, weight=float(payload.get("weight") or 1.0), observed_at=self.clock.now(),
                extractor=str(payload.get("extractor") or "replay_fixture"), extractor_version="v1",
                literal_span=dict(payload.get("literal_span") or {}),
            )
            common=dict(namespace=str(payload.get("namespace") or "verification"),scope_kind=str(payload.get("scope_kind") or "owner_local"),scope_id=str(payload.get("scope_id") or "leo"),subject_ref=str(payload.get("subject_ref") or "subject"),predicate=str(payload.get("predicate") or "value"),object_value=payload.get("object"),authority_class=str(payload.get("authority_class") or ("owner" if event_type=="seed_known_belief" else "extractor")),evidence=evidence)
            if event_type == "seed_known_belief": result=self.engine.belief_lifecycle.seed_known(**common)
            else: result=self.engine.belief_lifecycle.propose(**common,confidence=float(payload.get("confidence") or 0.5),status=BeliefStatus(str(payload.get("status") or "INFERRED")),sensitivity=str(payload.get("sensitivity") or "normal"))
            if result is not None:self._belief_aliases[event.event_id]=result.id
            return
        if event_type == "correct_belief":
            old_ref=str(payload.get("old_belief_ref") or ""); old_id=self._belief_aliases.get(old_ref,old_ref)
            payload.setdefault("authority_class","owner"); payload.setdefault("literal_span",{"start":0,"end":len(str(payload.get("text") or "")),"excerpt":str(payload.get("text") or "")[:80]})
            source_event_id,source_record_type,source_record_id=self._record_canonical_belief_evidence(event,payload)
            evidence=EvidenceRef(source_event_id=source_event_id,source_record_type=source_record_type,source_record_id=source_record_id,relation=EvidenceRelation.CORRECTS,weight=1.0,observed_at=self.clock.now(),extractor="owner_correction",extractor_version="v1",literal_span=dict(payload["literal_span"]))
            result=self.engine.belief_lifecycle.correct(old_id,object_value=payload.get("object"),evidence=evidence,authority_class=str(payload.get("authority_class") or "owner"));self._belief_aliases[event.event_id]=result.id;return
        if event_type == "retrieve_beliefs":
            self.engine.memory_retrieval.retrieve(RetrievalRequest(context_kind=str(payload.get("context_kind") or "owner_local"),purpose=str(payload.get("purpose") or "current_context"),subject=str(payload.get("subject_ref") or ""),allowed_scopes=tuple(payload.get("allowed_scopes") or ()),allowed_sensitivity=tuple(payload.get("allowed_sensitivity") or ("normal",)),temporal_intent=str(payload.get("temporal_intent") or "current"),max_results=int(payload.get("max_results") or 10)));return
        if event_type == "add_legacy_memory_fact":
            conn=self.workspace.connection();now=self.clock.iso();cur=conn.execute("INSERT INTO memory_facts(kind,subject,payload_json,source_text,confidence,created_at,updated_at,active) VALUES(?,?,?,?,?,?,?,1)",(str(payload.get("kind") or "fact"),str(payload.get("subject_ref") or "legacy"),json.dumps(payload.get("payload") or {},ensure_ascii=False),str(payload.get("source_text") or ""),float(payload.get("confidence") or .5),now,now));conn.commit();conn.close();self._belief_aliases[event.event_id]=str(cur.lastrowid);return
        if event_type == "project_legacy_memory_fact":
            fact_ref=str(payload.get("fact_ref") or "");result=self.engine.legacy_memory_fact_adapter.shadow_project(int(self._belief_aliases.get(fact_ref,fact_ref)));self._belief_aliases[event.event_id]=getattr(result,"id","");return
        if event_type == "add_vector_context":
            conn=self.workspace.connection();conn.execute("INSERT INTO memory_chunks(text,kind,subject,source_session,embedding,embedding_model,embedding_dim,importance,created_at,tags,active) VALUES(?,?,?,?,?,?,?,?,?,?,1)",(str(payload.get("text") or "vector context"),"misc",str(payload.get("subject_ref") or "subject"),"replay",b'0',"replay",1,.5,self.clock.iso(),"{}"));conn.commit();conn.close();return
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
        social=getattr(self.engine,"social_world",None)
        if social is not None:social.expire_social_threads()
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
