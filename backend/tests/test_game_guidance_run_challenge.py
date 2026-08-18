from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.cognitive.cognitive_router import CognitiveRouter
from app.cognitive.context_builder import ContextBuilder
from app.cognitive.deliberation_service import DeliberationService
from app.cognitive.game_guidance import GameGuidanceCapability, GameRunState
from app.cognitive.speech_act_pipeline import _fallback_for_guard, action_claim_guard, build_universal_speech_act_bundle
from app.epistemics.repository import BeliefRepository
from app.epistemics.service import BeliefLifecycleService
from app.game_context_v2.challenge import ChallengeContextService
from app.game_context_v2.repository import GameV2Repository
from app.game_context_v2.service import GameRunService
from app.replay.migrations import MigrationRunner, belief_v2_migrations, game_context_v2_migrations
from app.core.state import HebeState
from tests.test_game_guidance_routing import game_pending, routing_context


class GroundedSearchProvider:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.queries: list[str] = []

    def search(self, query: str):
        self.queries.append(query)
        if self.fail:
            raise TimeoutError("fixture provider unavailable")
        return [{
            "claim": "Continue from the named milestone using the cited route.",
            "excerpt": "The guide identifies the next destination after this milestone.",
            "url": "https://example.test/game-guide",
            "source_type": "fixture_reference",
        }]


class GameQuestionContractTests(unittest.TestCase):
    def evaluate(self, text: str, *, run: GameRunState | None = None, provider=None, snapshot=None):
        capability = GameGuidanceCapability(search_provider=provider)
        state = dict(snapshot or {})
        state.setdefault("game_run_state", (run or GameRunState()).to_dict())
        context = SimpleNamespace(input_text=text, state_snapshot=state, relevant_chunks=[])
        return capability.evaluate(context)

    def test_current_game_is_reused_for_general_totk_question_and_lookup(self):
        provider = GroundedSearchProvider()
        result = self.evaluate(
            "¿Dónde consigo la capucha hyliana?",
            provider=provider,
            snapshot={
                "current_game_context": {
                    "game": "The Legend of Zelda: Tears of the Kingdom",
                    "source": "twitch_category",
                    "confidence": .92,
                }
            },
        )

        self.assertEqual(result.context.game, "The Legend of Zelda: Tears of the Kingdom")
        self.assertEqual(result.context.current_game_source, "twitch_category")
        self.assertEqual(result.context.game_question_type, "GAME_GENERAL")
        self.assertFalse(result.context.needs_clarification)
        self.assertTrue(result.context.lookup_attempted)
        self.assertEqual(result.context.lookup_outcome, "success")
        self.assertEqual(result.context.answer_grounding, "web_research")

    def test_progression_milestone_from_input_is_sufficient(self):
        provider = GroundedSearchProvider()
        result = self.evaluate(
            "¿Dónde voy después de conseguir el barco volador?",
            run=GameRunState(game="Final Fantasy V", confidence=.95), provider=provider,
        )

        self.assertEqual(result.context.game_question_type, "PROGRESSION_DEPENDENT")
        self.assertEqual(result.context.milestone, "conseguir el barco volador")
        self.assertFalse(result.context.needs_clarification)
        self.assertIn("barco volador", provider.queries[0].casefold())

    def test_run_specific_question_requests_level_and_party_once(self):
        result = self.evaluate(
            "¿Puedo vencer a Ifrit con mi party actual?",
            run=GameRunState(game="Final Fantasy V", confidence=.95),
        )

        self.assertEqual(result.context.game_question_type, "RUN_SPECIFIC")
        self.assertTrue(result.context.needs_clarification)
        self.assertEqual(result.context.missing_required_fields, ["level", "party_jobs", "party_members"])

    def test_real_ffv_recommended_level_question_is_general_and_researched(self):
        provider = GroundedSearchProvider()
        result = self.evaluate(
            "¿Cuál es el nivel recomendado para vencer a Ifrit en Final Fantasy V?",
            provider=provider,
        )

        self.assertEqual(result.context.game_question_type, "GAME_GENERAL")
        self.assertEqual(result.context.query_target, "ifrit")
        self.assertFalse(result.context.needs_clarification)
        self.assertEqual(result.context.lookup_outcome, "success")

    def test_real_library_progression_question_uses_named_milestone(self):
        provider = GroundedSearchProvider()
        result = self.evaluate(
            "¿Dónde voy después de Library of the Ancients?",
            run=GameRunState(game="Final Fantasy V", confidence=.95), provider=provider,
        )

        self.assertEqual(result.context.game_question_type, "PROGRESSION_DEPENDENT")
        self.assertEqual(result.context.milestone, "library of the ancients")
        self.assertFalse(result.context.needs_clarification)
        self.assertEqual(result.context.lookup_outcome, "success")

    def test_context_builder_resolves_game_from_title_without_challenge_suffix(self):
        state = HebeState()
        state.stream.current_stream_title = "Final Fantasy V | Crystal Roulette Fiesta"
        snapshot = ContextBuilder(memory_store=None)._build_state_snapshot(state)

        self.assertEqual(snapshot["current_game_context"]["game"], "Final Fantasy V")
        self.assertEqual(snapshot["current_game_context"]["source"], "stream_title")

    def test_challenge_rules_and_run_overrides_are_applied_to_guidance_context(self):
        provider = GroundedSearchProvider()
        rule = {"rule_id": "rule-1", "text": "No se puede cambiar de job", "status": "ACTIVE"}
        override = {"rule_id": "override-1", "text": "Esta vez se permite repetir", "status": "ACTIVE"}
        result = self.evaluate(
            "¿Cuál es el nivel recomendado para vencer a Ifrit?",
            run=GameRunState(
                game="Final Fantasy V", confidence=.95, playthrough_type="challenge",
                challenge="Reto persistente", challenge_definition_id="challenge-1",
                challenge_rules=[rule], challenge_overrides=[override],
            ), provider=provider,
        )

        self.assertEqual(result.context.challenge_definition_id, "challenge-1")
        self.assertEqual(result.context.challenge_rules, [rule])
        self.assertEqual(result.context.challenge_overrides, [override])
        self.assertFalse(result.context.needs_clarification)

    def test_ffv_party_followup_is_typed_and_continues_original_question(self):
        provider = GroundedSearchProvider()
        capability = GameGuidanceCapability(search_provider=provider)
        router = CognitiveRouter(game_guidance=capability)
        service = DeliberationService(intent_model=None, reasoning_model=None)
        service.cognitive_router = router
        service.game_guidance = capability
        service.goal_extractor.game_guidance = capability
        original = "¿Puedo vencer a Ifrit con mi party actual?"
        first = capability.evaluate(routing_context(
            original, run_state=GameRunState(game="Final Fantasy V", confidence=.95).to_dict(),
        ))
        pending = game_pending(domain_payload={
            "game": "Final Fantasy V",
            "missing_fields": first.context.missing_required_fields,
            "original_question": original,
            "spoiler_policy": "spoiler_safe_hints",
        })
        context = routing_context(
            "nivel 16, Beastmaster, Geomancer, Berserker y Ninja",
            run_state=GameRunState(game="Final Fantasy V", confidence=.95).to_dict(), pending=pending,
        )
        context.cognitive_decision = router.route(context)
        plan = service.deliberate(context).plan

        updates = plan.steps[0].data["updates"]
        self.assertEqual(updates["level"], 16)
        self.assertEqual(updates["party_jobs"], ["Beastmaster", "Geomancer", "Berserker", "Ninja"])
        self.assertNotIn("current_character", updates)
        self.assertNotIn("current_location", updates)
        guidance = plan.steps[1].data["game_guidance"]["context"]
        self.assertFalse(guidance["needs_clarification"])
        self.assertEqual(guidance["lookup_outcome"], "success")
        self.assertEqual(len(provider.queries), 1)

    def test_unknown_game_fact_lookup_failure_is_observable_and_not_grounded(self):
        result = self.evaluate(
            "¿Dónde consigo el objeto antiguo?",
            run=GameRunState(game="Juego desconocido", confidence=.9),
            provider=GroundedSearchProvider(fail=True),
        )

        self.assertTrue(result.context.lookup_attempted)
        self.assertTrue(result.context.lookup_outcome.startswith("failed:"))
        self.assertEqual(result.context.answer_grounding, "none")
        self.assertEqual(result.web_results, [])
        self.assertFalse(result.context.needs_clarification)

    def test_technical_pending_fallback_is_never_user_facing(self):
        bundle = build_universal_speech_act_bundle(
            route="owner_private_chat", speech_act_type="pending_task_followup",
            input_text="continúa", execution_result=None,
        )
        guard = action_claim_guard("Hecho, ya está completado.", bundle)
        reply = _fallback_for_guard(bundle, guard)

        self.assertNotIn("tarea pendiente", reply.casefold())
        self.assertNotIn("no se ejecutó ninguna operación", reply.casefold())
        self.assertIn("no voy a fingir", reply.casefold())


class ChallengeContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "challenge.sqlite3"
        self.connect = lambda: sqlite3.connect(self.path)
        runner = MigrationRunner(self.connect)
        runner.migrate(belief_v2_migrations())
        runner.migrate(game_context_v2_migrations())
        self.repo = GameV2Repository(self.connect)
        lifecycle = BeliefLifecycleService(BeliefRepository(self.connect), now_fn=lambda: 1000.0)
        self.runs = GameRunService(self.repo, lifecycle, now_fn=lambda: 1000.0)
        self.service = ChallengeContextService(self.repo, now_fn=lambda: 1000.0)
        self.run = self.runs.resolve(
            game="Final Fantasy V", stream_session_id="stream-1", source_event_id="start",
        ).active_run

    def tearDown(self):
        self.tmp.cleanup()

    def mention(self):
        return self.service.observe_owner_utterance(
            "Final Fantasy V con el desafío Crystal Roulette Fiesta",
            game="Final Fantasy V", run_id=self.run.id, source_event_id="mention",
        )

    def test_explicit_challenge_name_persists_and_marks_run(self):
        event = self.mention()
        state = self.runs.state(self.run.id)

        self.assertEqual(event["playthrough_type"], "challenge")
        self.assertEqual(state["playthrough_type"], "challenge")
        self.assertEqual(state["challenge"], "Crystal Roulette Fiesta")
        self.assertTrue(state["challenge_definition_id"])

    def test_rule_capture_adds_multiple_owner_rules_with_provenance(self):
        self.mention()
        started = self.service.observe_owner_utterance(
            "Hacemos un repaso a las reglas", game="Final Fantasy V", run_id=self.run.id, source_event_id="capture",
        )
        first = self.service.observe_owner_utterance(
            "No se puede cambiar la clase elegida", game="Final Fantasy V", run_id=self.run.id, source_event_id="rule-1",
        )
        second = self.service.observe_owner_utterance(
            "Cada vez hay que aceptar el resultado", game="Final Fantasy V", run_id=self.run.id, source_event_id="rule-2",
        )
        definition = self.service.definition_for_run(self.run.id)

        self.assertTrue(started["challenge_capture_started"])
        self.assertTrue(first["challenge_rule_added"])
        self.assertTrue(second["challenge_rule_added"])
        self.assertEqual(len(definition.rules), 2)
        self.assertTrue(all(rule["provenance"] == "owner_explicit" for rule in definition.rules))

    def test_capture_closes_explicitly_and_emits_observable_outcome(self):
        self.mention()
        self.service.observe_owner_utterance(
            "Hacemos un repaso a las reglas", game="Final Fantasy V", run_id=self.run.id, source_event_id="capture",
        )
        closed = self.service.observe_owner_utterance(
            "Eso es todo", game="Final Fantasy V", run_id=self.run.id, source_event_id="close",
        )

        self.assertTrue(closed["challenge_capture_closed"])
        self.assertEqual(closed["capture_close_reason"], "explicit_close")
        self.assertIsNone(self.service.capture)

    def test_owner_correction_supersedes_inference_instead_of_appending_contradiction(self):
        self.mention()
        definition = self.service.definition_for_run(self.run.id)
        self.service.add_rule(
            definition.challenge_id, "La selección aleatoria decide el camino",
            provenance="model_inference", confidence=.55,
        )
        self.service.observe_owner_utterance(
            "Te explico las reglas del reto", game="Final Fantasy V", run_id=self.run.id, source_event_id="capture",
        )
        corrected = self.service.observe_owner_utterance(
            "La selección aleatoria no sirve para decidir el camino",
            game="Final Fantasy V", run_id=self.run.id, source_event_id="correction",
        )
        rules = self.service.definition_for_run(self.run.id).rules

        self.assertTrue(corrected["challenge_rule_corrected"])
        self.assertEqual(sum(rule["status"] == "ACTIVE" for rule in rules), 1)
        self.assertEqual(sum(rule["status"] == "SUPERSEDED" for rule in rules), 1)
        self.assertEqual(next(rule for rule in rules if rule["status"] == "ACTIVE")["provenance"], "owner_explicit")

    def test_definition_is_reused_by_next_run_and_stream_title(self):
        first = self.mention()
        self.runs.finish(self.run.id, event_id="done")
        next_run = self.runs.resolve(
            game="FFV", stream_session_id="stream-2", source_event_id="new", explicit_new=True,
        ).active_run
        reused = self.service.apply_known_definition_from_metadata(
            title="Final Fantasy V | Crystal Roulette Fiesta", game="Final Fantasy V", run_id=next_run.id,
        )

        self.assertEqual(reused.challenge_id, first["challenge_id"])
        self.assertEqual(self.runs.state(next_run.id)["challenge_definition_id"], first["challenge_id"])

    def test_run_override_does_not_modify_global_definition(self):
        self.mention()
        self.service.observe_owner_utterance(
            "Te explico las reglas del reto", game="Final Fantasy V", run_id=self.run.id, source_event_id="capture",
        )
        before = self.service.definition_for_run(self.run.id)
        event = self.service.observe_owner_utterance(
            "En esta run se permite repetir una tirada", game="Final Fantasy V", run_id=self.run.id, source_event_id="override",
        )
        after = self.service.definition_for_run(self.run.id)
        context = self.service.context_for_run(self.run.id)

        self.assertTrue(event["run_override"])
        self.assertEqual(before.rules, after.rules)
        self.assertEqual(len(context["run_overrides"]), 1)


if __name__ == "__main__":
    unittest.main()
