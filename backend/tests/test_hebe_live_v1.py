from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.cognitive.final_emission_gate import FinalEmissionGate, OutputRoute
from app.stream.game_intelligence import (
    CommentKnowledgePolicy,
    GameAssistanceGuard,
    GameAssistanceMode,
    GameDossier,
    GameIntelligenceStore,
    GameProgressState,
    GameResearchService,
    KnowledgeGapTracker,
    RetrievedGameFact,
    SpoilerClassification,
    SpoilerFirewall,
)
from app.stream.intent_parser import StreamIntentParser
from app.stream.live_runtime import LiveSessionStateManager
from app.stream.promotions import (
    AutomaticPromotionService,
    AutoPromoMode,
    PromotionExecutionStatus,
    PromotionProfileManager,
    PromotionStore,
    ViewerPromotionProfile,
    parse_promotion_profile_command,
    record_manual_promotion,
)
from app.stream.replay import (
    ReplayDecision,
    ReplayFixtureResearchProvider,
    ReplayMode,
    ReplayResult,
    ReplaySession,
    StreamReplayLab,
)
from app.stream.runtime_context import HebeLiveContextPolicy, HebeLiveRuntimeContext


class _Provider:
    def __init__(self, rows=None, error: Exception | None = None):
        self.rows = list(rows or [])
        self.error = error
        self.calls = []

    def search(self, query):
        self.calls.append(query)
        if self.error:
            raise self.error
        return [dict(row) for row in self.rows]


def _safe_rows():
    return [
        {
            "claim": "Guard reduces incoming combat damage.",
            "title": "Official manual",
            "url": "https://official.invalid/guard",
            "content": "The Guard command reduces incoming combat damage.",
            "confidence": 0.92,
            "general_mechanic": True,
            "source_type": "official",
        },
        {
            "claim": "Guard reduces incoming combat damage.",
            "title": "Established wiki",
            "url": "https://wiki.invalid/guard",
            "content": "Guard reduces damage received until the next turn.",
            "confidence": 0.84,
            "general_mechanic": True,
            "source_type": "wiki",
        },
    ]


def _fact(**overrides):
    values = {
        "fact_id": "fact_1",
        "claim": "Guard reduces incoming combat damage.",
        "source_title": "Manual",
        "source_location": "https://official.invalid/guard",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "confidence": 0.9,
        "corroboration_count": 2,
        "spoiler_classification": SpoilerClassification.SAFE_GENERAL_MECHANIC.value,
        "progress_compatibility": "compatible",
        "exact_supporting_excerpt_internal": "Guard reduces damage.",
        "usable_for_comment": True,
        "usable_for_advice": True,
        "source_type": "official",
    }
    values.update(overrides)
    return RetrievedGameFact(**values)


def _progress(**overrides):
    values = {
        "game_id": "test_game",
        "stream_session_id": "session_1",
        "playthrough_type": "first_playthrough",
        "spoiler_policy": "strict",
        "confidence": 0.9,
    }
    values.update(overrides)
    return GameProgressState(**values)


class RuntimeContextTests(unittest.TestCase):
    def setUp(self):
        self.policy = HebeLiveContextPolicy()

    def test_owner_local_never_writes_twitch(self):
        result = self.policy.authorize_output(HebeLiveRuntimeContext.OWNER_LOCAL, ["twitch_chat"])
        self.assertFalse(result.allowed)
        sent = []
        emission = FinalEmissionGate().emit(
            event_id="ctx1",
            source="ui",
            final_response="must stay private",
            output_route=OutputRoute.TWITCH_TEXT_REPLY,
            output_targets=["twitch_chat"],
            guard_result={"passed": True},
            debug_payload={"response_stage": "final"},
            runtime_context="owner_local",
            send_twitch=sent.append,
        )
        self.assertFalse(emission.emitted)
        self.assertEqual(sent, [])

    def test_owner_live_control_can_execute_authorized_stream_operation(self):
        result = self.policy.authorize_action("owner_live_control", "promotion.owner_manual")
        self.assertTrue(result.allowed)

    def test_stream_public_cannot_execute_owner_desktop_action(self):
        result = self.policy.authorize_action("stream_public", "desktop.open_app")
        self.assertFalse(result.allowed)

    def test_shared_identity_does_not_merge_output_routes(self):
        shared_memory_id = "hebe-persistent-identity"
        local = {"identity": shared_memory_id, "context": "owner_local"}
        public = {"identity": shared_memory_id, "context": "stream_public"}
        self.assertEqual(local["identity"], public["identity"])
        self.assertFalse(self.policy.authorize_output(local["context"], ["twitch_chat"]).allowed)
        self.assertTrue(self.policy.authorize_output(public["context"], ["twitch_chat"]).allowed)

    def test_public_discourse_source_resolves_to_stream_context(self):
        self.assertEqual(
            self.policy.from_source("owner_discourse_opportunity"),
            HebeLiveRuntimeContext.STREAM_PUBLIC,
        )

    def test_public_automatic_promotion_requires_trusted_automation(self):
        self.assertFalse(
            self.policy.authorize_action("stream_public", "promotion.automatic_first_message").allowed
        )
        self.assertTrue(
            self.policy.authorize_action(
                "stream_public", "promotion.automatic_first_message", trusted_automation=True
            ).allowed
        )


class LiveSessionIsolationTests(unittest.TestCase):
    def setUp(self):
        self.stream = SimpleNamespace(
            companion_tick_count=8,
            idle_prompts_sent_stream=4,
            recent_chat_messages=[{"text": "old"}],
            recent_idle_messages=[{"text": "old comment"}],
            public_reply_timestamps=[1.0],
            processed_event_ids={"old-event"},
        )
        self.manager = LiveSessionStateManager(logger=lambda _message: None)

    def test_new_stream_has_zero_session_counters(self):
        self.manager.begin_session(self.stream, "new")
        self.assertEqual(self.stream.companion_tick_count, 0)
        self.assertEqual(self.stream.idle_prompts_sent_stream, 0)

    def test_old_comment_not_recent_in_new_stream(self):
        self.manager.begin_session(self.stream, "new")
        self.assertEqual(self.stream.recent_idle_messages, [])
        self.assertEqual(self.stream.recent_chat_messages, [])

    def test_promotion_preference_survives_new_stream(self):
        persistent = {"alice": "first_message_each_stream"}
        self.stream.promotion_preferences = persistent
        self.manager.begin_session(self.stream, "new")
        self.assertIs(self.stream.promotion_preferences, persistent)

    def test_game_dossier_survives_new_stream(self):
        dossier = {"test_game": {"facts": ["guard"]}}
        self.stream.game_dossiers = dossier
        self.manager.begin_session(self.stream, "new")
        self.assertIs(self.stream.game_dossiers, dossier)

    def test_old_emission_dedupe_does_not_cross_stream(self):
        gate = FinalEmissionGate()
        sent = []
        arguments = {
            "source": "spontaneity",
            "final_response": "That timing was close.",
            "output_route": OutputRoute.TWITCH_TEXT_REPLY,
            "output_targets": ["twitch_chat"],
            "guard_result": {"passed": True},
            "debug_payload": {"response_stage": "final"},
            "send_twitch": sent.append,
        }
        self.assertTrue(gate.emit(**arguments).emitted)
        self.assertFalse(gate.emit(**arguments).emitted)
        gate.reset_session()
        self.assertTrue(gate.emit(**arguments).emitted)
        self.assertEqual(len(sent), 2)


class ReplayLabTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            {"event_id": "second", "event_type": "stt", "timestamp": 2, "payload": {"text": "b"}},
            {"event_id": "first", "event_type": "stream_online", "timestamp": 1, "payload": {}},
        ]

    @staticmethod
    def processor(event, runtime):
        runtime.state.last_event = event.event_id
        return ReplayDecision(
            event_id=event.event_id,
            proposed_output=f"out:{event.event_id}",
            output_targets=["twitch_chat"],
            should_emit=True,
            presence_allowed=True,
            final_guard_allowed=True,
        )

    def test_replay_never_writes_real_twitch(self):
        def attempted_writer(event, runtime):
            runtime.io.send_twitch("unsafe")
            return self.processor(event, runtime)

        result = StreamReplayLab().run(self.events, attempted_writer)
        self.assertIsInstance(result, ReplayResult)
        self.assertTrue(result.blocked_real_writes)
        self.assertTrue(all(item["blocked"] for item in result.blocked_real_writes))

    def test_accelerated_replay_preserves_event_order(self):
        result = StreamReplayLab().run(self.events, self.processor, mode=ReplayMode.ACCELERATED)
        self.assertEqual([item.event_id for item in result.decisions], ["first", "second"])

    def test_same_fixture_same_decisions(self):
        lab = StreamReplayLab()
        first = lab.run(self.events, self.processor)
        second = lab.run(self.events, self.processor)
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_compare_versions_reports_changed_outputs(self):
        def candidate(event, runtime):
            value = self.processor(event, runtime)
            value.proposed_output += ":new"
            return value

        report = StreamReplayLab().compare_versions(self.events, self.processor, candidate)
        self.assertEqual(len(report.changed_outputs), 2)

    def test_stream_state_resets_at_replay_start(self):
        lab = StreamReplayLab(state_factory=lambda: SimpleNamespace(companion_tick_count=99, recent_chat_messages=["old"]))
        result = lab.run(self.events[:1], self.processor)
        snapshot = result.state_snapshots[0]["state"]
        self.assertEqual(snapshot["companion_tick_count"], 0)
        self.assertEqual(snapshot["recent_chat_messages"], [])

    def test_step_by_step_only_advances_one_event(self):
        session = StreamReplayLab().run(self.events, self.processor, mode=ReplayMode.STEP_BY_STEP)
        self.assertIsInstance(session, ReplaySession)
        self.assertEqual(session.step().event_id, "first")
        self.assertFalse(session.done)

    def test_replay_uses_research_fixture(self):
        provider = ReplayFixtureResearchProvider({"guard": [{"claim": "safe"}]})
        self.assertEqual(provider.search("ignored", cache_key="guard")[0]["claim"], "safe")

    def test_replay_without_fixture_fails_closed(self):
        with self.assertRaises(LookupError):
            ReplayFixtureResearchProvider().search("missing")


class PromotionPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.store = PromotionStore(connection=self.connection)

    def tearDown(self):
        self.connection.close()

    def test_failed_so_not_recorded_as_executed(self):
        event = record_manual_promotion(
            self.store,
            stream_session_id="s1",
            source_event_id="cmd1",
            requested_by="leo",
            raw_target_text="alice",
            resolved_twitch_user_id="1",
            resolved_login="alice",
            resolution_confidence=1,
            send_shoutout=lambda _login: False,
        )
        self.assertEqual(event.execution_status, "failed")
        self.assertIsNone(event.executed_at)

    def test_twitch_send_success_records_sent(self):
        event = record_manual_promotion(
            self.store,
            stream_session_id="s1",
            source_event_id="cmd2",
            requested_by="leo",
            raw_target_text="alice",
            resolved_twitch_user_id="1",
            resolved_login="alice",
            resolution_confidence=1,
            send_shoutout=lambda _login: {"success": True, "message_id": "tw1"},
        )
        self.assertEqual(event.execution_status, "sent")
        self.assertEqual(event.twitch_message_id, "tw1")

    def test_complaint_about_missing_promo_not_parsed_as_executed_command(self):
        parsed = StreamIntentParser().parse("Hebe, no hiciste la promo que te pedi ayer")
        self.assertFalse(any(item.intent == "twitch_shoutout" for item in parsed))

    def test_one_terminal_promotion_status(self):
        event = record_manual_promotion(
            self.store,
            stream_session_id="s1",
            source_event_id="cmd3",
            requested_by="leo",
            raw_target_text="alice",
            resolved_twitch_user_id="1",
            resolved_login="alice",
            resolution_confidence=1,
            send_shoutout=lambda _login: True,
        )
        with self.assertRaises(ValueError):
            self.store.transition(event.id, PromotionExecutionStatus.FAILED)

    def test_successful_owner_promo_can_create_profile(self):
        manager = PromotionProfileManager(self.store, default_auto_after_success=True)
        profile = manager.learn_after_success(twitch_user_id="1", login="alice", stream_session_id="s1")
        self.assertEqual(profile.auto_promo_mode, "first_message_each_stream")

    def test_only_this_time_does_not_enable_auto(self):
        manager = PromotionProfileManager(self.store, default_auto_after_success=True)
        profile = manager.learn_after_success(
            twitch_user_id="1", login="alice", owner_command="haz promo a Alice solo esta vez"
        )
        self.assertIsNone(profile)

    def test_viewer_request_cannot_enable_auto_promo(self):
        self.assertIsNone(parse_promotion_profile_command("Hebe me haces promo a mi porfa"))
        self.assertEqual(self.store.list_profiles(), [])

    def test_owner_can_disable_auto_promo(self):
        manager = PromotionProfileManager(self.store)
        manager.learn_after_success(twitch_user_id="1", login="alice")
        command = parse_promotion_profile_command("deja de promocionar a alice")
        result = manager.apply_command(command)
        self.assertEqual(result.auto_promo_mode, AutoPromoMode.DISABLED.value)

    def test_profile_persists_by_twitch_user_id(self):
        profile = ViewerPromotionProfile(
            twitch_user_id="42", current_login="old_login", auto_promo_mode=AutoPromoMode.FIRST_MESSAGE_EACH_STREAM.value
        )
        self.store.upsert_profile(profile)
        profile.current_login = "new_login"
        self.store.upsert_profile(profile)
        self.assertEqual(self.store.get_profile(twitch_user_id="42").current_login, "new_login")


class AutomaticPromotionTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.store = PromotionStore(connection=self.connection)
        self.store.upsert_profile(ViewerPromotionProfile(
            twitch_user_id="42",
            current_login="alice",
            auto_promo_mode=AutoPromoMode.FIRST_MESSAGE_EACH_STREAM.value,
        ))
        self.service = AutomaticPromotionService(self.store, spacing_seconds=0, bot_usernames={"nightbot"})

    def tearDown(self):
        self.connection.close()

    def observe(self, **overrides):
        values = {
            "stream_session_id": "s1",
            "twitch_user_id": "42",
            "login": "alice",
            "display_name": "Alice",
            "message_text": "hola",
            "message_id": "m1",
            "channel_live": True,
        }
        values.update(overrides)
        return self.service.observe_chat_message(**values)

    def test_configured_viewer_first_message_triggers_so(self):
        self.assertEqual(self.observe().decision, "queue")
        sent = []
        result = self.service.drain_ready(lambda login: sent.append(login) or True)
        self.assertEqual(result.execution_status, "sent")
        self.assertEqual(sent, ["alice"])

    def test_second_message_same_stream_does_not_repeat_so(self):
        self.observe()
        self.service.drain_ready(lambda _login: True)
        second = self.observe(message_id="m2", message_text="otra")
        self.assertEqual(second.reason, "not_first_message")

    def test_duplicate_observed_message_does_not_duplicate_so(self):
        self.observe()
        duplicate = self.observe()
        self.assertEqual(duplicate.reason, "duplicate_observed_message")
        self.assertEqual(self.service.queued_count, 1)

    def test_new_stream_allows_new_so(self):
        self.observe()
        self.service.drain_ready(lambda _login: True)
        decision = self.observe(stream_session_id="s2", message_id="m2")
        self.assertEqual(decision.decision, "queue")

    def test_bot_message_never_triggers_auto_promo(self):
        decision = self.observe(twitch_user_id="bot", login="nightbot", is_bot=True)
        self.assertEqual(decision.reason, "bot_or_self")

    def test_offline_message_never_triggers_auto_promo(self):
        decision = self.observe(channel_live=False)
        self.assertEqual(decision.reason, "channel_offline")


class GameIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.store = GameIntelligenceStore(connection=self.connection)

    def tearDown(self):
        self.connection.close()

    def service(self, rows=None, error=None, now_fn=None):
        provider = _Provider(_safe_rows() if rows is None else rows, error=error)
        return GameResearchService(store=self.store, provider=provider, now_fn=now_fn or (lambda: 1000)), provider

    def test_known_game_dossier_reused(self):
        service, provider = self.service()
        first, status = service.get_or_build_dossier(game_title="Test Game")
        second, second_status = service.get_or_build_dossier(game_title="Test Game")
        self.assertEqual(status, "created")
        self.assertEqual(second_status, "loaded")
        self.assertEqual(first.game_id, second.game_id)
        self.assertEqual(len(provider.calls), 1)

    def test_missing_game_dossier_can_be_built(self):
        service, _ = self.service()
        dossier, status = service.get_or_build_dossier(game_title="Test Game")
        self.assertEqual(status, "created")
        self.assertTrue(dossier.confirmed_general_mechanics)

    def test_dossier_failure_does_not_block_stream(self):
        service, _ = self.service(error=RuntimeError("offline"))
        dossier, status = service.get_or_build_dossier(game_title="Test Game")
        self.assertIsNone(dossier)
        self.assertEqual(status, "failed")

    def test_dossier_contains_provenance(self):
        service, _ = self.service()
        dossier, _ = service.get_or_build_dossier(game_title="Test Game")
        self.assertTrue(dossier.sources[0]["location"].startswith("https://"))
        self.assertTrue(dossier.sources[0]["fact_id"])

    def test_dossier_excludes_future_story_summary(self):
        rows = _safe_rows() + [{
            "claim": "A future party member is secretly the villain.",
            "title": "Spoiler wiki",
            "url": "https://wiki.invalid/spoiler",
            "content": "The future party member is secretly the villain.",
            "confidence": .9,
            "spoiler_classification": "identity_spoiler",
        }]
        service, _ = self.service(rows=rows)
        dossier, _ = service.get_or_build_dossier(game_title="Test Game")
        self.assertNotIn("villain", " ".join(dossier.confirmed_general_mechanics).lower())

    def test_future_boss_fact_blocked(self):
        fact = _fact(spoiler_classification="future_mechanic", claim="Future boss weakness is fire")
        self.assertFalse(SpoilerFirewall().evaluate(fact, _progress()).allowed)

    def test_future_party_member_blocked(self):
        fact = _fact(spoiler_classification="identity_spoiler", claim="X joins later")
        self.assertFalse(SpoilerFirewall().evaluate(fact, _progress()).allowed)

    def test_general_combat_mechanic_allowed(self):
        self.assertTrue(SpoilerFirewall().evaluate(_fact(), _progress()).allowed)

    def test_encountered_boss_mechanic_allowed_if_safe(self):
        progress = _progress(encountered_bosses=["Dragon"])
        fact = _fact(spoiler_classification="safe_current_progress", progress_compatibility="compatible")
        self.assertTrue(SpoilerFirewall().evaluate(fact, progress).allowed)

    def test_unknown_progress_fact_blocked_under_strict_mode(self):
        fact = _fact(spoiler_classification="uncertain_progress")
        self.assertFalse(SpoilerFirewall().evaluate(fact, _progress()).allowed)

    def test_repeated_unknown_mechanic_triggers_lookup(self):
        service, _ = self.service()
        first = service.trigger_engine.decide(
            game_id="test", text="HP does not decrease", entity="enemy hp", owner_uncertainty=True, unknown_mechanic=True
        )
        second = service.trigger_engine.decide(
            game_id="test", text="HP still does not decrease", entity="enemy hp", owner_uncertainty=True, unknown_mechanic=True
        )
        self.assertFalse(first.should_research)
        self.assertTrue(second.should_research)

    def test_one_uncertain_fragment_does_not_search(self):
        service, provider = self.service()
        decision = service.trigger_engine.decide(
            game_id="test", text="maybe heal", entity="heal", unknown_mechanic=True, confidence=.4
        )
        self.assertFalse(decision.should_research)
        self.assertEqual(provider.calls, [])

    def test_owner_direct_game_question_triggers_lookup(self):
        service, _ = self.service()
        decision = service.trigger_engine.decide(
            game_id="test", text="How does guard work?", entity="guard", explicit_direct_question=True
        )
        self.assertTrue(decision.should_research)
        self.assertEqual(decision.mode, "owner_explicit_question")

    def test_quoted_game_dialogue_does_not_trigger_lookup(self):
        service, _ = self.service()
        decision = service.trigger_engine.decide(
            game_id="test", text="Where is the crystal?", quoted_dialogue=True, explicit_direct_question=True
        )
        self.assertFalse(decision.should_research)

    def test_cached_query_not_repeated(self):
        service, provider = self.service()
        plan = service.plan_search(
            game_title="Test Game", game_id="test", entity="guard", question_type="mechanic", expected_fact_type="mechanic"
        )
        service.research(plan, progress=_progress())
        service.research(plan, progress=_progress())
        self.assertEqual(len(provider.calls), 1)

    def test_weak_single_snippet_not_specific_advice(self):
        rows = [{"claim": "Use fire", "title": "search", "url": "https://search.invalid", "snippet": "Use fire", "confidence": .95}]
        service, _ = self.service(rows=rows)
        plan = service.plan_search(game_title="Test", game_id="test", entity="boss", question_type="advice", expected_fact_type="advice")
        fact = service.research(plan, progress=_progress())[0]
        self.assertFalse(fact.usable_for_advice)
        self.assertFalse(fact.usable_for_comment)

    def test_conflicting_sources_mark_fact_uncertain(self):
        rows = [{
            "claim": "Guard heals HP", "title": "A", "url": "https://a.invalid", "content": "Guard heals.",
            "conflict_group": "guard", "conflicting": True, "general_mechanic": True, "confidence": .9,
        }]
        service, _ = self.service(rows=rows)
        plan = service.plan_search(game_title="Test", game_id="test", entity="guard", question_type="mechanic", expected_fact_type="mechanic")
        fact = service.research(plan, progress=_progress())[0]
        self.assertEqual(fact.spoiler_classification, "uncertain_progress")
        self.assertFalse(fact.usable_for_comment)

    def test_corroborated_mechanic_may_be_used(self):
        service, _ = self.service()
        plan = service.plan_search(game_title="Test", game_id="test", entity="guard", question_type="mechanic", expected_fact_type="mechanic")
        fact = service.research(plan, progress=_progress())[0]
        self.assertTrue(fact.usable_for_comment)
        self.assertTrue(fact.usable_for_advice)

    def test_failed_search_does_not_generate_fake_fact(self):
        service, _ = self.service(error=RuntimeError("network"))
        plan = service.plan_search(game_title="Test", game_id="test", entity="guard", question_type="mechanic", expected_fact_type="mechanic")
        with self.assertRaises(RuntimeError):
            service.research(plan, progress=_progress())
        self.assertEqual(service.diagnostics.facts_accepted, [])

    def test_repeated_unknown_term_creates_gap(self):
        tracker = KnowledgeGapTracker(self.store)
        self.assertIsNone(tracker.observe(game_id="test", term="Zantetsu", raw_evidence="heard once"))
        gap = tracker.observe(game_id="test", term="Zantetsu", raw_evidence="heard twice")
        self.assertEqual(gap.occurrence_count, 2)

    def test_resolved_gap_updates_dossier(self):
        tracker = KnowledgeGapTracker(self.store)
        tracker.observe(game_id="test", term="Guard", raw_evidence="one")
        gap = tracker.observe(game_id="test", term="Guard", raw_evidence="two")
        dossier = GameDossier(game_id="test", canonical_title="Test")
        tracker.resolve(gap, [_fact()], dossier)
        self.assertIn("Guard reduces incoming combat damage.", dossier.confirmed_general_mechanics)

    def test_failed_gap_does_not_invent_knowledge(self):
        tracker = KnowledgeGapTracker(self.store)
        tracker.observe(game_id="test", term="Mystery", raw_evidence="one")
        gap = tracker.observe(game_id="test", term="Mystery", raw_evidence="two")
        dossier = GameDossier(game_id="test", canonical_title="Test")
        tracker.resolve(gap, [], dossier)
        self.assertEqual(gap.status, "failed")
        self.assertEqual(dossier.confirmed_general_mechanics, [])

    def test_low_value_single_term_not_persisted(self):
        tracker = KnowledgeGapTracker(self.store)
        self.assertIsNone(tracker.observe(game_id="test", term="x", raw_evidence="x"))
        self.assertIsNone(self.store.get_gap("test", "x"))

    def test_unsolicited_puzzle_solution_blocked(self):
        allowed, _ = GameAssistanceGuard().allow(
            "La solución del puzzle es pulsar rojo", mode=GameAssistanceMode.MECHANICS_WITHOUT_SOLUTIONS
        )
        self.assertFalse(allowed)

    def test_unsolicited_exact_route_blocked(self):
        allowed, _ = GameAssistanceGuard().allow(
            "Ve primero a la derecha por la ruta exacta", mode=GameAssistanceMode.HINTS_ON_REQUEST
        )
        self.assertFalse(allowed)

    def test_general_mechanic_observation_allowed(self):
        allowed, _ = GameAssistanceGuard().allow(
            "Parece que Guard reduce daño", mode=GameAssistanceMode.MECHANICS_WITHOUT_SOLUTIONS
        )
        self.assertTrue(allowed)

    def test_owner_explicit_request_can_receive_spoiler_safe_hint(self):
        allowed, _ = GameAssistanceGuard().allow(
            "La ruta exacta empieza por la derecha",
            mode=GameAssistanceMode.FULL_HELP_ON_REQUEST,
            explicit_owner_request=True,
        )
        self.assertTrue(allowed)

    def test_research_does_not_block_twitch_queue(self):
        release = threading.Event()

        class BlockingProvider:
            def search(self, _query):
                release.wait(timeout=2)
                return _safe_rows()

        service = GameResearchService(store=self.store, provider=BlockingProvider(), now_fn=lambda: 1000)
        plan = service.plan_search(
            game_title="Test", game_id="test", entity="guard",
            question_type="mechanic", expected_fact_type="mechanic",
        )
        started = time.monotonic()
        job = service.queue_research(plan, progress=_progress(), scene_id="scene-1")
        elapsed = time.monotonic() - started
        try:
            self.assertLess(elapsed, 0.25)
            self.assertIn(job.status, {"queued", "running"})
        finally:
            release.set()
            service._jobs[job.job_id][1].result(timeout=2)

    def test_stale_research_result_not_used_for_old_scene(self):
        clock = [1000.0]
        service, _ = self.service(now_fn=lambda: clock[0])
        plan = service.plan_search(
            game_title="Test", game_id="test", entity="guard",
            question_type="mechanic", expected_fact_type="mechanic",
        )
        job = service.queue_research(plan, progress=_progress(), scene_id="old-scene", ttl_seconds=0.1)
        service._jobs[job.job_id][1].result(timeout=2)
        clock[0] = 1001.0
        completed, facts = service.collect_job(job.job_id)
        self.assertEqual(completed.status, "stale")
        self.assertEqual(facts, [])

    def test_direct_owner_question_can_wait_with_explicit_status(self):
        release = threading.Event()

        class BlockingProvider:
            def search(self, _query):
                release.wait(timeout=2)
                return _safe_rows()

        service = GameResearchService(store=self.store, provider=BlockingProvider(), now_fn=lambda: 1000)
        plan = service.plan_search(
            game_title="Test", game_id="test", entity="How does guard work?",
            question_type="owner_question", expected_fact_type="mechanic",
        )
        job = service.queue_research(plan, progress=_progress(), scene_id="owner-question", ttl_seconds=45)
        try:
            pending, facts = service.collect_job(job.job_id)
            self.assertIn(pending.status, {"queued", "running"})
            self.assertEqual(facts, [])
            self.assertEqual(service.diagnostics.active_research_job["job_id"], job.job_id)
        finally:
            release.set()
            service._jobs[job.job_id][1].result(timeout=2)

    def test_spontaneous_scene_expires_cleanly(self):
        service, _ = self.service()
        plan = service.plan_search(
            game_title="Test", game_id="test", entity="combat moment",
            question_type="spontaneous", expected_fact_type="mechanic",
        )
        job = service.queue_research(plan, progress=_progress(), scene_id="moment-1")
        service._jobs[job.job_id][1].result(timeout=2)
        completed, facts = service.collect_job(job.job_id, scene_still_current=False)
        self.assertEqual(completed.status, "stale")
        self.assertEqual(facts, [])

    def test_comment_contract_defaults_to_safe_reaction(self):
        contract = CommentKnowledgePolicy().build_contract(
            scene_evidence=["enemy HP is not decreasing"], facts=[], progress=_progress()
        )
        self.assertEqual(contract.contribution_mode, "contextual_reaction")
        self.assertEqual(contract.allowed_claims, [])

    def test_comment_diagnostics_include_mode_and_fact_provenance(self):
        service, _ = self.service()
        service.get_or_build_dossier(game_title="Test Game")
        payload = service.record_final_comment(
            comment_id="comment-1",
            text="Guard reduces incoming combat damage.",
            game="Test Game",
        )
        snapshot = service.debug_snapshot()
        self.assertEqual(payload["mode"], "informed_observation")
        self.assertEqual(snapshot["current_comment_mode"], "informed_observation")
        self.assertTrue(snapshot["current_comment_provenance"])

    def test_changed_progress_rechecks_spoiler_compatibility(self):
        rows = [{
            "claim": "Dragon guard timing is visible after meeting Dragon.",
            "title": "Wiki", "url": "https://wiki.invalid/dragon", "content": "Guard timing is visible.",
            "confidence": .9, "spoiler_classification": "safe_current_progress", "required_boss": "Dragon",
        }]
        service, provider = self.service(rows=rows)
        plan = service.plan_search(game_title="Test", game_id="test", entity="dragon", question_type="mechanic", expected_fact_type="mechanic")
        first = service.research(plan, progress=_progress(encountered_bosses=["Dragon"]))[0]
        changed = service.research(plan, progress=_progress(encountered_bosses=[]))[0]
        self.assertTrue(first.usable_for_comment)
        self.assertFalse(changed.usable_for_comment)
        self.assertEqual(len(provider.calls), 1)

    def test_low_level_enemies_do_not_imply_level_one_challenge(self):
        service, provider = self.service()
        decision = service.trigger_engine.decide(
            game_id="test", text="These enemies are low level", entity="low-level enemies", confidence=.9
        )
        self.assertFalse(decision.should_research)
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
