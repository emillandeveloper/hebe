from __future__ import annotations

import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch

from app.cognitive.cognitive_router import CognitiveRouter
from app.cognitive.deliberation_service import DeliberationService
from app.cognitive.game_guidance import GameGuidanceCapability, GameRunState
from app.cognitive.context_builder import BuiltContext
from app.cognitive.models import DeliberationResult, ExecutionResult, Plan, StepExecutionResult
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.hebe_engine import HebeEngine
from app.cognitive.input_event import InputEvent


def game_pending(**overrides):
    pending = {
        "id": "game-pending",
        "kind": "game_guidance_clarification",
        "game": "Final Fantasy VII",
        "location_or_area": "Midgar",
        "expected_reply_type": "game_party_or_character",
        "missing_fields": ["current_character", "party_members", "story_phase", "recent_event"],
        "original_question": "Hebe, en FFVII acabo de llegar a Midgar; ¿cuál es el siguiente objetivo?",
        "authority": "owner",
        "spoiler_policy": "no_story_spoilers",
        "expires_at": time.time() + 300,
    }
    pending.update(overrides)
    return pending


def routing_context(text: str, *, run_state=None, chunks=None, pending=None):
    return SimpleNamespace(
        input_text=text, internal_event=None,
        state_snapshot={"game_run_state": run_state or {}, "pending_clarification": pending},
        relevant_chunks=list(chunks or []), relevant_facts=[], recent_appointments=[],
        pending_reminders=[], conversation_history=[], resolved_entities=[],
        message_type="direct_question", context_policy={"memory": "relevant"},
        source="ui", authority="owner", addressed_to_hebe=True,
        message_id="game-test", cognitive_decision=None,
    )


class ClaimingModel:
    def chat(self, messages, **kwargs):
        return "Ve al castillo, habla con la autoridad local y busca el objeto clave."


class SearchProvider:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [{"title": "mechanics reference", "content": "verified strategy notes", "url": "https://example.test/guide"}]


class GameGuidanceRoutingTests(unittest.TestCase):
    def setUp(self):
        self.router = CognitiveRouter()
        self.service = DeliberationService(intent_model=None, reasoning_model=None)

    def test_ambiguous_location_routes_to_clarification_not_fallback(self):
        value = routing_context(
            "Hebe, en FFIX ando en Alexandria y no sé cuál es el siguiente objetivo"
        )
        value.cognitive_decision = self.router.route(value)
        plan = self.service.deliberate(value).plan
        guidance = plan.steps[0].data["game_guidance"]["context"]

        self.assertEqual(value.cognitive_decision.intent, "game_guidance_query")
        self.assertTrue(value.cognitive_decision.allows_capability("game.guidance"))
        self.assertFalse(value.cognitive_decision.allows_capability("hebe.chat_reply"))
        self.assertEqual(plan.steps[0].data["mode"], "game_guidance_clarification")
        self.assertEqual(guidance["game"], "Final Fantasy IX")
        self.assertEqual(guidance["location_or_area"], "Alexandria")
        self.assertTrue(guidance["needs_clarification"])
        self.assertFalse(guidance["should_search_web"])

    def test_concrete_item_question_selects_rag_then_web_when_local_empty(self):
        value = routing_context("Hebe, en FFIX dónde consigo este objeto raro?")
        value.cognitive_decision = self.router.route(value)
        plan = self.service.deliberate(value).plan
        guidance = plan.steps[0].data["game_guidance"]["context"]

        self.assertEqual(value.cognitive_decision.intent, "game_guidance_query")
        self.assertTrue(guidance["should_use_rag"])
        self.assertTrue(guidance["should_search_web"])
        self.assertFalse(guidance["needs_clarification"])

    def test_game_run_state_supplies_game_challenge_and_character(self):
        run = GameRunState(
            game="Final Fantasy IX", playthrough_type="level_1_challenge",
            spoiler_policy="mechanics_ok_story_avoid", current_location="Alexandria",
            current_character="Vivi", current_objective="inspect the current objective marker",
            challenge="level one", provenance="Leo said", confidence=.95,
        )
        value = routing_context("Hebe, qué toca ahora?", run_state=run.to_dict())
        value.cognitive_decision = self.router.route(value)
        plan = self.service.deliberate(value).plan
        guidance = plan.steps[0].data["game_guidance"]["context"]

        self.assertEqual(value.cognitive_decision.intent, "game_guidance_query")
        self.assertEqual(guidance["current_character"], "Vivi")
        self.assertEqual(guidance["playthrough_type"], "level_1_challenge")
        self.assertEqual(guidance["spoiler_policy"], "mechanics_ok_story_avoid")
        self.assertFalse(guidance["needs_clarification"])

    def test_first_playthrough_policy_limits_answer_depth(self):
        capability = GameGuidanceCapability()
        value = routing_context(
            "Hebe, qué hago hoy en Persona 5?",
            run_state=GameRunState(
                game="Persona 5 Royal", playthrough_type="first_playthrough",
                spoiler_policy="no_story_spoilers", confidence=.9,
            ).to_dict(),
        )
        result = capability.evaluate(value).context

        self.assertEqual(result.spoiler_policy, "no_story_spoilers")
        self.assertEqual(result.allowed_answer_depth, "hint_only")
        self.assertIn("unrequested_story_reveals", result.forbidden_content)
        self.assertTrue(result.needs_clarification)

    def test_concrete_boss_query_uses_run_game_and_precise_search(self):
        provider = SearchProvider()
        capability = GameGuidanceCapability(search_provider=provider)
        value = routing_context(
            "Hebe, cómo planteo el combate contra Myrkul?",
            run_state=GameRunState(
                game="Baldur's Gate 3", playthrough_type="strategy_mode",
                spoiler_policy="mechanics_ok_story_avoid", confidence=.9,
            ).to_dict(),
        )
        result = capability.evaluate(value)

        self.assertEqual(result.context.game, "Baldur's Gate 3")
        self.assertEqual(result.context.query_kind, "boss")
        self.assertTrue(result.context.should_search_web)
        self.assertTrue(result.web_results)
        self.assertIn("Baldur's Gate 3", provider.queries[0])
        self.assertIn("spoiler safe mechanics", provider.queries[0])

    def test_story_sensitive_first_playthrough_requires_spoiler_permission(self):
        capability = GameGuidanceCapability()
        value = routing_context(
            "Hebe, explícame el giro final de Persona 5",
            run_state=GameRunState(
                game="Persona 5 Royal", playthrough_type="first_playthrough",
                spoiler_policy="no_story_spoilers", confidence=.9,
            ).to_dict(),
        )
        result = capability.evaluate(value).context

        self.assertTrue(result.needs_clarification)
        self.assertFalse(result.should_search_web)
        self.assertIn("major_spoiler_permission_required", result.ambiguity_reasons)

    def test_fallback_chat_blocks_ungrounded_walkthrough_claim(self):
        context = BuiltContext(
            input_text="Hebe, sigo en Alexandria dentro de FFIX; ¿cuál es el próximo objetivo?",
            internal_event=None, relevant_facts=[], recent_appointments=[], pending_reminders=[],
            state_snapshot={}, relevant_chunks=[], conversation_history=[], message_type="direct_question",
            inject_memory=True, context_policy={"memory": "relevant"},
        )
        context.cognitive_decision = SimpleNamespace(intent="direct_question")
        execution = ExecutionResult([StepExecutionResult("reply", True, {"mode": "chat"})])
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            reply = ResponseSynthesizer(ClaimingModel()).synthesize(
                context=context, deliberation=DeliberationResult(Plan()), execution=execution,
            )

        self.assertNotIn("castillo", reply.casefold())
        self.assertIn("contexto", reply.casefold())
        self.assertIn("blocked_game_walkthrough=true", "\n".join(logs))

    def test_game_guidance_clarification_creates_real_pending(self):
        value = routing_context("Hebe, en FFVII he llegado a Midgar y no ubico el siguiente paso")
        value.cognitive_decision = self.router.route(value)
        plan = self.service.deliberate(value).plan
        engine = HebeEngine.__new__(HebeEngine)
        engine.runtime = SimpleNamespace(state=SimpleNamespace(pending_clarification=None))
        engine.deliberation_service = self.service

        engine._apply_game_guidance_reply_state(
            plan.steps[0].data, value.cognitive_decision, value, "owner_ui",
        )

        pending = engine.runtime.state.pending_clarification
        self.assertEqual(pending["kind"], "game_guidance_clarification")
        self.assertEqual(pending["expected_reply_type"], "game_party_or_character")
        self.assertIn("current_character", pending["missing_fields"])

    def test_clarification_answer_routes_and_updates_run_state_plan(self):
        value = routing_context("Voy con Cloud", pending=game_pending())
        value.cognitive_decision = self.router.route(value)
        plan = self.service.deliberate(value).plan

        self.assertEqual(value.cognitive_decision.intent, "game_guidance_clarification_answer")
        self.assertTrue(value.cognitive_decision.uses_pending_task)
        self.assertTrue(value.cognitive_decision.pending_compatible)
        update = plan.steps[0].data["updates"]
        self.assertEqual(update["current_character"], "Cloud")
        self.assertEqual(update["party_members"], ["Cloud"])
        self.assertEqual(update["provenance"], "leo_clarification")
        next_guidance = plan.steps[1].data["game_guidance"]["context"]
        self.assertNotIn("character_unknown", next_guidance["ambiguity_reasons"])

    def test_wake_aliases_are_stripped_and_character_aliases_normalized(self):
        capability = GameGuidanceCapability()
        updates = capability.parse_clarification_answer(
            game_pending(), "Heba, Tifa, Cloud, Yufi, Eve",
        )

        self.assertEqual(updates["party_members"], ["Tifa", "Cloud", "Yuffie"])
        self.assertEqual(updates["current_character"], "Tifa")
        self.assertNotIn("Eve", updates["party_members"])
        self.assertNotIn("Heba", updates["party_members"])

    def test_addressed_clarification_answer_still_consumes_pending(self):
        value = routing_context("Hebe, estoy con Cloud, Eve", pending=game_pending())
        decision = self.router.route(value)

        self.assertEqual(decision.intent, "game_guidance_clarification_answer")
        self.assertTrue(decision.pending_compatible)

    def test_fuzzy_location_variants_normalize_conservatively(self):
        capability = GameGuidanceCapability()
        for variant in ("Mikdar", "Migdar", "Milgar", "Migar", "Midgard"):
            with self.subTest(variant=variant):
                self.assertEqual(capability.normalize_entity(variant, "location", "Final Fantasy VII"), "Midgar")

    def test_unrelated_rag_chunks_are_not_guidance_sources(self):
        value = routing_context(
            "Hebe, en FFVII dónde encuentro este objeto?",
            chunks=[
                {"subject": "Baldur's Gate 3", "text": "boss route notes"},
                {"subject": "Final Fantasy VII", "text": "local mechanics note"},
            ],
        )
        result = GameGuidanceCapability().evaluate(value)

        self.assertEqual(len(result.rag_chunks), 1)
        self.assertIn("Final Fantasy VII", result.rag_chunks[0]["subject"])

    def test_fallback_chat_is_blocked_while_game_pending_is_active(self):
        context = BuiltContext(
            input_text="Cloud, Tifa", internal_event=None,
            relevant_facts=[], recent_appointments=[], pending_reminders=[],
            state_snapshot={"pending_clarification": game_pending()}, relevant_chunks=[],
            conversation_history=[], message_type="unknown", inject_memory=False,
            context_policy={"memory": "limited"},
        )
        context.cognitive_decision = SimpleNamespace(intent="unknown_chat")
        execution = ExecutionResult([StepExecutionResult("reply", True, {"mode": "chat"})])
        logs = []
        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            reply = ResponseSynthesizer(ClaimingModel()).synthesize(
                context=context, deliberation=DeliberationResult(Plan()), execution=execution,
            )

        self.assertIn("aclaración", reply)
        self.assertIn("active_game_guidance_pending", "\n".join(logs))

    def test_stt_answer_without_wake_is_owner_followup(self):
        from tests.test_voice_command_pipeline import make_engine

        engine = make_engine(live=False)
        engine.runtime.state.pending_clarification = game_pending()
        event = InputEvent(
            source="stt_voice", raw_text="Voy con Cloud", normalized_text="voy con cloud",
            stt_metadata={"command_mode": True},
        )
        envelope = engine._build_stt_input_envelope(
            event, voice_type="unknown", conversation_followup=False,
        )

        self.assertEqual(envelope.source, "owner_stt_followup")
        self.assertTrue(envelope.pending_compatible)
        self.assertEqual(envelope.expected_reply_type, "game_party_or_character")

    def test_character_clarification_is_not_requested_twice(self):
        run = GameRunState(
            game="Final Fantasy VII", current_location="Midgar",
            current_character="Cloud", party_members=["Cloud"],
            spoiler_policy="no_story_spoilers", provenance="leo_clarification", confidence=.92,
        )
        value = routing_context("Hebe, ¿cuál es el siguiente paso?", run_state=run.to_dict())
        value.cognitive_decision = self.router.route(value)
        plan = self.service.deliberate(value).plan
        guidance = plan.steps[-1].data["game_guidance"]["context"]

        self.assertNotIn("character_unknown", guidance["ambiguity_reasons"])
        self.assertNotIn("current_character", self.service.game_guidance.missing_fields(guidance))

    def test_location_answer_does_not_become_party_member(self):
        updates = self.router.game_guidance.parse_clarification_answer(
            game_pending(), "Acabo de llegar a Mikdar",
        )

        self.assertEqual(updates["current_location"], "Midgar")
        self.assertNotIn("current_character", updates)
        self.assertNotIn("party_members", updates)

    def test_successful_state_update_mutates_runtime_game_run_state(self):
        from tests.test_voice_command_pipeline import make_engine

        engine = make_engine(live=False)
        result = StepExecutionResult("state_update", True, {
            "kind": "game_run_state", "pending_id": "game-pending",
            "updates": {"game": "Final Fantasy VII", "current_character": "Cloud", "party_members": ["Cloud"]},
        })
        engine._apply_game_run_state_execution(result)

        run = engine.runtime.state.game_run_state
        self.assertEqual(run.game, "Final Fantasy VII")
        self.assertEqual(run.current_character, "Cloud")
        self.assertEqual(run.party_members, ["Cloud"])


if __name__ == "__main__":
    unittest.main()
