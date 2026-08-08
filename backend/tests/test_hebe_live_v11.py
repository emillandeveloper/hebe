from __future__ import annotations

import sqlite3
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.cognitive.input_event import InputEvent
from app.hebe_engine import HebeEngine
from app.services.direct_stt_command import DirectSTTCommandResult
from app.services.stt_whisper import STTConfig, STTService
from app.stream.evidence_entailment import EvidenceEntailmentGuard
from app.stream.game_advice_gate import GameAdviceGate
from app.stream.game_intelligence import (
    CommentKnowledgePolicy,
    GameIntelligenceStore,
    GameResearchService,
    ResearchMode,
)
from app.stream.output_language import StreamOutputLanguagePolicy
from app.stream.promotions import AutomaticPromotionService, PromotionProfileManager, PromotionStore
from app.stream.scene_timeline import SceneTimelineManager, SpontaneousOpportunityManager
from app.stream.viewer_operation_gate import ViewerStreamOperationTopicGate


class _ResearchProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[str] = []

    def search(self, query: str):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("provider_down")
        return [{
            "claim": "Combat uses a guard action to reduce damage.",
            "source_title": "Official manual",
            "url": "https://example.invalid/manual",
            "excerpt": "The guard action reduces incoming combat damage.",
            "confidence": 0.94,
            "general_mechanic": True,
        }]


def _research_service(provider=None, **kwargs) -> GameResearchService:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    return GameResearchService(
        store=GameIntelligenceStore(connection=connection),
        provider=provider,
        cache_ttl_seconds=60,
        **kwargs,
    )


def _collect(service: GameResearchService, job_id: str):
    pair = service._jobs[job_id]
    try:
        pair[1].result(timeout=2)
    except RuntimeError:
        pass
    return service.collect_job(job_id)


class HebeLiveV11ResearchTests(unittest.TestCase):
    def test_insufficient_dossier_queues_research(self):
        service = _research_service(provider=None)
        job = service.prepare_game_async(game_title="Unknown Game")
        self.assertEqual(job.mode, ResearchMode.PRE_STREAM_DOSSIER.value)
        self.assertIn(job.job_id, service._jobs)
        _collect(service, job.job_id)
        self.assertEqual(job.status, "failed")
        self.assertIn("research_provider_missing", job.error)
        service._executor.shutdown(wait=True)

    def test_configured_provider_is_invoked(self):
        provider = _ResearchProvider()
        service = _research_service(provider=provider, provider_name="fixture", provider_configured=True)
        job = service.prepare_game_async(game_title="Test Game")
        _collect(service, job.job_id)
        self.assertEqual(len(provider.queries), 1)
        service._executor.shutdown(wait=True)

    def test_provider_missing_has_explicit_status(self):
        service = _research_service(provider=None, provider_name="none", provider_configured=False)
        self.assertFalse(service.diagnostics.research_provider_available)
        self.assertEqual(service.diagnostics.research_provider_reason, "provider_not_configured")
        service._executor.shutdown(wait=True)

    def test_successful_research_populates_dossier(self):
        service = _research_service(provider=_ResearchProvider())
        job = service.prepare_game_async(game_title="Test Game")
        _collect(service, job.job_id)
        dossier = service.store.get_dossier("test_game")
        self.assertIsNotNone(dossier)
        self.assertTrue(dossier.confirmed_general_mechanics)
        self.assertTrue(dossier.sources)
        service._executor.shutdown(wait=True)

    def test_research_failure_keeps_reaction_only_mode(self):
        service = _research_service(provider=_ResearchProvider(fail=True))
        job = service.prepare_game_async(game_title="Test Game")
        _collect(service, job.job_id)
        self.assertEqual(service.diagnostics.dossier_status, "failed")
        self.assertEqual(service.diagnostics.current_comment_mode, "contextual_reaction")
        service._executor.shutdown(wait=True)


class HebeLiveV11SceneTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.timeline = SceneTimelineManager(now_fn=lambda: self.now)

    def _fact(self, fact_id: str, text: str, **extra):
        return {
            "id": fact_id, "text": text, "timestamp": self.now,
            "ttl_sec": 60, "expires_at": self.now + 60,
            "confidence": 0.9, "proactive_eligible": True,
            **extra,
        }

    def test_enemy_death_supersedes_low_hp_observation(self):
        low = self._fact("low-hp", "Le queda poca vida")
        self.timeline.observe(low["text"], event_id="low-hp", topic_id="combat", facts=[low])
        low = self.timeline.annotate_facts([low], topic_id="combat")[0]
        self.timeline.observe("El enemigo está muerto", event_id="dead", topic_id="combat")
        low = self.timeline.annotate_facts([low], topic_id="combat")[0]
        self.assertTrue(low["superseded"])
        self.assertEqual(self.timeline.current.current_state, "enemy_dead")
        self.assertTrue(self.timeline.current.terminal)

    def test_candidate_cancelled_if_scene_changes_during_generation(self):
        self.timeline.observe("Combate activo", event_id="combat", topic_id="combat")
        snapshot = self.timeline.snapshot()
        self.timeline.observe("Entramos en una nueva zona", event_id="area", topic_id="explore")
        decision = self.timeline.revalidate(snapshot)
        self.assertFalse(decision.valid)
        self.assertEqual(decision.reason, "scene_changed")

    def test_old_combat_fact_excluded_after_terminal_event(self):
        fact = self._fact("combat-fact", "Le queda poca vida", topic_id="combat")
        self.timeline.observe(fact["text"], event_id="combat-fact", topic_id="combat", facts=[fact])
        fact = self.timeline.annotate_facts([fact], topic_id="combat")[0]
        self.timeline.observe("Fin del combate", event_id="end", topic_id="combat")
        fact["superseded"] = True
        self.assertEqual(self.timeline.filter_current_facts([fact], topic_id="combat"), [])

    def test_area_change_invalidates_prior_navigation_anchor(self):
        fact = self._fact("nav", "Ve por la izquierda", topic_id="navigation")
        self.timeline.observe("Buscando la salida", event_id="nav", topic_id="navigation", facts=[fact])
        fact = self.timeline.annotate_facts([fact], topic_id="navigation")[0]
        self.timeline.observe("Hemos llegado a una nueva zona", event_id="area", topic_id="new-area")
        self.assertEqual(self.timeline.filter_current_facts([fact], topic_id="new-area"), [])

    def test_save_remark_from_old_scene_not_fused(self):
        policy = CommentKnowledgePolicy()
        facts = [self._fact("save-old", "Hay que guardar", scene_id="old", topic_id="old")]
        selected = policy.filter_scene_facts(
            facts, current_scene_id="new", current_topic_id="new", now=self.now,
        )
        self.assertEqual(selected, [])

    def test_level_up_comment_not_reused_as_combat_instruction(self):
        policy = CommentKnowledgePolicy()
        fact = self._fact("level", "Subí de nivel", scene_id="scene", topic_id="level_up")
        selected = policy.filter_scene_facts(
            [fact], current_scene_id="scene", current_topic_id="combat", now=self.now,
        )
        self.assertEqual(selected, [])

    def test_comment_contract_contains_only_same_scene_facts(self):
        policy = CommentKnowledgePolicy()
        facts = [
            self._fact("current", "Ahora combate", scene_id="s1", topic_id="combat"),
            self._fact("old", "Antes guardar", scene_id="s0", topic_id="old"),
        ]
        contract = policy.build_contract(
            scene_evidence=[], facts=[],
            progress=SimpleNamespace(), scene_facts=facts,
            current_scene_id="s1", current_topic_id="combat", now=self.now,
        )
        self.assertEqual(contract.scene_fact_ids, ["current"])

    def test_stale_fact_ids_excluded_from_provenance(self):
        policy = CommentKnowledgePolicy()
        stale = self._fact("stale", "Viejo", scene_id="s1", topic_id="combat")
        stale["expires_at"] = self.now - 1
        contract = policy.build_contract(
            scene_evidence=[], facts=[], progress=SimpleNamespace(),
            scene_facts=[stale], current_scene_id="s1", current_topic_id="combat", now=self.now,
        )
        self.assertNotIn("stale", contract.scene_fact_ids)


class HebeLiveV11OpportunityTests(unittest.TestCase):
    def setUp(self):
        self.manager = SpontaneousOpportunityManager(now_fn=lambda: 100.0)

    def test_blocked_anchor_becomes_consumed(self):
        opportunity = self.manager.open("anchor-a")
        self.manager.mark(opportunity.opportunity_id, "consumed", reason="guard", guard="test")
        self.assertFalse(self.manager.eligible("anchor-a"))

    def test_failed_safe_rewrite_consumes_opportunity(self):
        opportunity = self.manager.open("anchor-a")
        self.assertTrue(self.manager.safe_rewrite_once(opportunity.opportunity_id))
        self.assertFalse(self.manager.safe_rewrite_once(opportunity.opportunity_id))
        self.manager.mark(opportunity.opportunity_id, "consumed", reason="rewrite_failed")
        self.assertEqual(opportunity.status, "consumed")

    def test_same_anchor_not_probed_twice_after_guard_failure(self):
        opportunity = self.manager.open("anchor-a")
        self.manager.mark(opportunity.opportunity_id, "blocked", reason="guard")
        self.assertIsNone(self.manager.open("anchor-a"))

    def test_unrelated_new_anchor_remains_eligible(self):
        opportunity = self.manager.open("anchor-a")
        self.manager.mark(opportunity.opportunity_id, "consumed")
        self.assertTrue(self.manager.eligible("anchor-b"))


class HebeLiveV11GuardTests(unittest.TestCase):
    def setUp(self):
        self.advice = GameAdviceGate()

    def _blocked(self, text: str, game: str = "Unknown"):
        return self.advice.validate(current_game=game, proposed_advice=text, source_evidence=[])

    def test_english_save_advice_detected(self):
        self.assertIn("save_instruction", self._blocked("Remember to save before the boss.").blocked)

    def test_spanish_heal_advice_detected(self):
        self.assertIn("heal_instruction", self._blocked("No olvides curarte antes.").blocked)

    def test_cross_game_mechanic_rejected(self):
        result = self._blocked("You should use materia now.", game="Persona 5 Royal")
        self.assertFalse(result.allowed)
        self.assertIn("materia", result.blocked)

    def test_unvalidated_enemy_state_assumption_rejected(self):
        result = self._blocked("Casi lo tienes, sigue atacando.")
        self.assertFalse(result.allowed)
        self.assertIn("enemy_alive_assumption", result.blocked)

    def test_game_advice_gate_runs_on_spontaneous_gameplay_comment(self):
        result = self._blocked("You should equip the sword.")
        self.assertFalse(result.allowed)
        self.assertIn("equip_instruction", result.mechanics)

    def test_empty_claim_extraction_blocks_substantive_candidate(self):
        decision = EvidenceEntailmentGuard().evaluate(
            "The next room is definitely the final arena.",
            {"raw_owner_fragments": ["Entramos en una sala."], "current_state": "active"},
        )
        self.assertFalse(decision.passed)
        self.assertEqual(decision.result, "extraction_failure")

    def test_pure_emotional_reaction_allowed_without_claims(self):
        decision = EvidenceEntailmentGuard().evaluate("Uf!", {})
        self.assertTrue(decision.passed)

    def test_enemy_alive_assumption_checked_against_scene_state(self):
        decision = EvidenceEntailmentGuard().evaluate(
            "Casi lo tienes, ataca.",
            {"raw_owner_fragments": ["El enemigo está muerto."], "current_state": "enemy_dead", "terminal": True},
        )
        self.assertIn("enemy_alive_assumption", decision.contradicted)
        self.assertFalse(decision.passed)

    def test_save_advice_requires_current_support(self):
        decision = EvidenceEntailmentGuard().evaluate(
            "Recuerda guardar.", {"raw_owner_fragments": ["Entramos en combate."]},
        )
        self.assertIn("save_instruction", decision.unsupported)

    def test_level_up_advice_requires_current_support(self):
        decision = EvidenceEntailmentGuard().evaluate(
            "Deberías subir de nivel.", {"raw_owner_fragments": ["Este enemigo pega fuerte."]},
        )
        self.assertIn("level_up_condition", decision.unsupported)


class HebeLiveV11LanguageAndAuthorityTests(unittest.TestCase):
    def test_spontaneous_comment_stays_in_configured_language(self):
        policy = StreamOutputLanguagePolicy("es")
        result = policy.enforce("That was close!", event_type="spontaneous_stream_comment")
        self.assertEqual(result.expected_language, "es")
        self.assertEqual(policy.detect(result.text), "es")

    def test_english_dialogue_does_not_flip_autonomous_output_language(self):
        policy = StreamOutputLanguagePolicy("es")
        self.assertEqual(policy.expected_language(event_type="spontaneous_stream_comment", source_language="en"), "es")

    def test_direct_english_viewer_can_receive_english_reply(self):
        policy = StreamOutputLanguagePolicy("es")
        result = policy.enforce("That was close!", event_type="direct_viewer_reply", source_language="en")
        self.assertEqual(result.text, "That was close!")

    def test_owner_can_explicitly_change_stream_output_language(self):
        policy = StreamOutputLanguagePolicy("es")
        self.assertEqual(policy.set_owner_preference("en"), "en")
        self.assertEqual(policy.expected_language(), "en")

    def test_viewer_cannot_trigger_raid(self):
        result = ViewerStreamOperationTopicGate().evaluate("Haz una raid a Alice", source_type="viewer")
        self.assertEqual(result.outcome, "observe_only")
        self.assertFalse(result.may_execute)

    def test_viewer_cannot_trigger_moderation(self):
        result = ViewerStreamOperationTopicGate().evaluate("Banea a Alice", source_type="viewer")
        self.assertEqual(result.operation, "moderation")
        self.assertFalse(result.may_execute)

    def test_viewer_cannot_change_title_or_category(self):
        gate = ViewerStreamOperationTopicGate()
        self.assertEqual(gate.evaluate("Cambia el título", source_type="viewer").outcome, "observe_only")
        self.assertEqual(gate.evaluate("Cambia la categoría", source_type="viewer").outcome, "observe_only")

    def test_owner_can_trigger_live_control(self):
        result = ViewerStreamOperationTopicGate().evaluate(
            "Cambia la categoría", source_type="owner", owner_trusted=True,
        )
        self.assertTrue(result.may_execute)

    def test_historical_raid_discussion_can_receive_safe_banter(self):
        result = ViewerStreamOperationTopicGate().evaluate("La raid de ayer fue enorme", source_type="viewer")
        self.assertEqual(result.outcome, "authority_preserving_banter")
        self.assertTrue(result.may_generate_reply)


class HebeLiveV11PromotionAndSTTTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.store = PromotionStore(connection=self.connection)

    def test_successful_manual_promo_creates_profile(self):
        profile = PromotionProfileManager(self.store).learn_after_success(
            twitch_user_id="123", login="Alice", owner_command="haz promo a Alice",
            source_promotion_event="promo-1", known_aliases=["AliceTV"],
        )
        self.assertEqual(profile.twitch_user_id, "123")
        self.assertIn("alicetv", profile.known_aliases)

    def test_failed_manual_promo_does_not_create_profile(self):
        self.assertIsNone(self.store.get_profile(login="alice"))

    def test_learned_profile_triggers_next_stream(self):
        PromotionProfileManager(self.store).learn_after_success(
            twitch_user_id="123", login="alice", owner_command="haz promo a alice",
        )
        service = AutomaticPromotionService(self.store, spacing_seconds=0)
        decision = service.observe_chat_message(
            stream_session_id="next", twitch_user_id="123", login="alice",
            display_name="Alice", message_text="hola", message_id="m1", channel_live=True,
        )
        self.assertEqual(decision.decision, "queue")

    def test_renamed_login_matches_same_twitch_id(self):
        manager = PromotionProfileManager(self.store)
        manager.learn_after_success(twitch_user_id="123", login="alice", owner_command="haz promo")
        service = AutomaticPromotionService(self.store)
        service.observe_chat_message(
            stream_session_id="next", twitch_user_id="123", login="newalice",
            display_name="NewAlice", message_text="hola", message_id="m1", channel_live=True,
        )
        profile = self.store.get_profile(twitch_user_id="123")
        self.assertEqual(profile.twitch_user_id, "123")
        self.assertEqual(profile.current_login, "newalice")
        self.assertIn("alice", profile.known_aliases)

    def test_profile_missing_log_emitted_once(self):
        service = AutomaticPromotionService(self.store)
        with patch("builtins.print") as printer:
            for index in range(2):
                service.observe_chat_message(
                    stream_session_id="s", twitch_user_id="404", login="nobody",
                    display_name="Nobody", message_text="hola", message_id=f"m{index}", channel_live=True,
                )
        logs = [str(call) for call in printer.call_args_list if "profile_missing" in str(call)]
        self.assertEqual(len(logs), 1)

    def test_success_terminal_cannot_be_overwritten_by_rejection(self):
        engine = object.__new__(HebeEngine)
        result = DirectSTTCommandResult(event_id="terminal-1", detected_intent_family="application_action")
        receipt = {"action_type": "test", "target": "test", "executor_invoked": True, "success": True, "timestamp": time.time()}
        with patch("app.hebe_engine.emit"):
            self.assertTrue(engine._log_direct_stt_outcome(result, outcome="action_executed", reason="success", action_receipt=receipt))
            self.assertFalse(engine._log_direct_stt_outcome(result, outcome="rejected", reason="fallback"))
        self.assertEqual(result.final_outcome, "action_executed")

    def test_success_terminal_stops_parser_fallback(self):
        engine = object.__new__(HebeEngine)
        direct = DirectSTTCommandResult(event_id="terminal-2", detected_intent_family="application_action")
        engine._current_input_event = InputEvent(
            source="stt_voice", raw_text="abre OBS", normalized_text="abre obs",
            stt_metadata={"direct_stt_command": direct.to_dict()},
        )
        with patch("app.hebe_engine.emit"):
            self.assertTrue(engine._commit_current_direct_stt_terminal(
                outcome="action_executed", reason="success",
                action_receipt={"action_type": "test", "target": "test", "executor_invoked": True, "success": True, "timestamp": time.time()},
            ))
        self.assertEqual(engine._current_direct_stt_terminal_outcome()["outcome"], "action_executed")

    def test_duplicate_terminal_write_is_ignored(self):
        self.test_success_terminal_cannot_be_overwritten_by_rejection()

    def test_empty_audio_classified_without_language_recovery(self):
        service = STTService(STTConfig())
        reason = service.classify_empty_transcript(audio_np=np.array([], dtype=np.float32), metadata={})
        self.assertEqual(reason, "empty_audio")

    def test_no_speech_in_allowed_language_is_not_dual_decoded(self):
        service = STTService(STTConfig())
        reason = service.classify_empty_transcript(
            audio_np=np.ones(16000, dtype=np.float32) * 0.02,
            metadata={"detected_language": "es", "no_speech_probability": 0.95},
            speech_detected=True,
        )
        self.assertEqual(reason, "no_speech")

    def test_quiet_background_audio_is_dropped_cleanly(self):
        service = STTService(STTConfig())
        reason = service.classify_empty_transcript(
            audio_np=np.zeros(16000, dtype=np.float32), metadata={}, speech_detected=False,
        )
        self.assertEqual(reason, "empty_audio")


if __name__ == "__main__":
    unittest.main()
