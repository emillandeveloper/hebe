from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.cognitive.interaction_history import (
    RecentInteractionDecisionHistory,
    detect_self_explanation_query,
    render_grounded_self_explanation,
)
from app.cognitive.models import ExecutionResult, StepExecutionResult
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.cognitive.speech_act_pipeline import (
    build_universal_speech_act_bundle,
    final_response_guard,
)
from app.stream.game_advice_gate import GameAdviceGate
from app.stream.game_knowledge import (
    GameKnowledgeConfig,
    GameKnowledgeResolver,
    classify_game_knowledge_query,
)
from app.stream.game_profiles import GameProfile, GameProfileStore
from app.stream.game_research import GameKnowledgeResearchConfig, GameKnowledgeResearchService
from app.stream.state import StreamSessionState
from tests.test_voice_command_pipeline import (
    FakeContextBuilder,
    FakeDeliberationService,
    FakePlanExecutor,
    make_engine,
)


class RecordingModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else ""


class SearchProvider:
    def __init__(self, rows=None, error: Exception | None = None):
        self.rows = list(rows or [])
        self.error = error
        self.calls = []

    def search(self, query):
        self.calls.append(query)
        if self.error:
            raise self.error
        return list(self.rows)


class UnknownGameGroundingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, profiles=()):
        root = Path(self.tmp.name)
        path = root / "profiles.json"
        cache = root / "cache.json"
        path.write_text(json.dumps({"profiles": [item.to_dict() for item in profiles]}), encoding="utf-8")
        return GameProfileStore(path=path, cache_path=cache)

    @staticmethod
    def context(message, knowledge, *, current_game="Current Fixture Game", source="ui"):
        return SimpleNamespace(
            internal_event=None,
            input_text=message,
            message_type="direct_question",
            source=source,
            state_snapshot={},
            resolved_entities=[],
            relevant_chunks=[],
            relevant_facts=[],
            inject_memory=False,
            context_policy={"memory": "limited"},
            response_frame={
                "current_game": current_game,
                "current_session_context": {"game_intelligence": dict(knowledge)},
            },
        )

    @staticmethod
    def chat_execution():
        return ExecutionResult([StepExecutionResult(
            step_type="reply",
            success=True,
            data={"mode": "chat"},
        )])

    def test_a_current_game_known_dates_remain_grounded(self):
        query = classify_game_knowledge_query(
            "¿De cuándo es el juego que estoy jugando, Hebe?",
            current_game="Current Fixture Game",
        )
        self.assertEqual(query.game_title, "Current Fixture Game")
        model = RecordingModel(["Salió en 2008 en Japón y en 2009 en Occidente."])
        synth = ResponseSynthesizer(conversation_model=model)
        context = self.context(
            "¿De cuándo es el juego que estoy jugando, Hebe?",
            {
                "query_detected": True,
                "query_intent": "factual",
                "queried_game": "Current Fixture Game",
                "game_knowledge_status": "PARTIAL",
                "allowed_claims": ["release_japan=2008", "release_west=2009"],
                "evidence_count": 2,
                "claim_count": 2,
                "lookup_used": False,
            },
        )

        reply = synth.synthesize(context=context, deliberation=SimpleNamespace(plan=SimpleNamespace()), execution=self.chat_execution())

        self.assertIn("2008", reply)
        self.assertIn("2009", reply)
        validation = synth.last_response_debug_contract["guard_result"]["game_knowledge_validation"]
        self.assertEqual(validation["claims_grounded"], ["year"])
        self.assertFalse(validation["ungrounded_claim_blocked"])

    def test_b_unknown_without_lookup_blocks_genre_mechanics_and_year(self):
        bundle = build_universal_speech_act_bundle(
            route="unknown_game",
            speech_act_type="direct_answer",
            input_text="¿Qué sabes de Fictional Meridian?",
            current_game="Fictional Meridian",
            memory={"game_knowledge": {
                "query_detected": True,
                "queried_game": "Fictional Meridian",
                "game_knowledge_status": "UNKNOWN",
                "allowed_claims": [],
            }},
        )

        result = final_response_guard(
            "Es un JRPG de combate por turnos lanzado en 2024.",
            bundle,
            game_advice_gate=GameAdviceGate(),
        )

        self.assertFalse(result.passed)
        self.assertEqual(
            set(result.game_knowledge_validation["claims_ungrounded"]),
            {"genre", "gameplay", "year"},
        )
        self.assertTrue(result.game_knowledge_validation["ungrounded_claim_blocked"])
        hedged = final_response_guard(
            "No tengo datos fiables, pero parece un JRPG de combate por turnos.",
            bundle,
            game_advice_gate=GameAdviceGate(),
        )
        self.assertFalse(hedged.passed)
        self.assertTrue({"genre", "gameplay"} <= set(hedged.game_knowledge_validation["claims_ungrounded"]))

    def test_c_unknown_recommendation_is_honest_and_skips_model(self):
        model = RecordingModel(["Sí, es un RPG táctico excelente."])
        synth = ResponseSynthesizer(conversation_model=model)
        context = self.context(
            "¿Me recomiendas Fictional Meridian?",
            {
                "query_detected": True,
                "query_intent": "recommendation",
                "recommendation_requested": True,
                "queried_game": "Fictional Meridian",
                "game_knowledge_status": "UNKNOWN",
                "allowed_claims": [],
                "evidence_count": 0,
                "lookup_used": False,
                "claim_count": 0,
            },
        )

        reply = synth.synthesize(context=context, deliberation=SimpleNamespace(plan=SimpleNamespace()), execution=self.chat_execution())

        self.assertEqual(model.calls, [])
        self.assertIn("no tengo base fiable suficiente", reply.casefold())
        self.assertNotIn("rpg", reply.casefold())
        self.assertEqual(synth.last_response_debug_contract["game_knowledge_status"], "UNKNOWN")

    def test_factual_guard_covers_non_advice_game_claim_categories(self):
        bundle = build_universal_speech_act_bundle(
            route="unknown_game_categories",
            speech_act_type="direct_answer",
            input_text="¿Qué sabes de Fictional Meridian?",
            current_game="Fictional Meridian",
            memory={"game_knowledge": {
                "query_detected": True,
                "queried_game": "Fictional Meridian",
                "game_knowledge_status": "UNKNOWN",
                "allowed_claims": [],
            }},
        )
        cases = (
            ("Está disponible en Nintendo Switch.", {"availability", "platform"}),
            ("Fue desarrollado por Studio Fixture.", {"developer"}),
            ("La trama sigue a una heroína perdida.", {"plot"}),
            ("Fue aclamado por la crítica.", {"reception"}),
            ("Es un juego muy difícil.", {"difficulty"}),
            ("Tiene multijugador cooperativo.", {"features"}),
        )

        for candidate, expected in cases:
            with self.subTest(candidate=candidate):
                result = final_response_guard(candidate, bundle, game_advice_gate=GameAdviceGate())
                self.assertFalse(result.passed, result.to_dict())
                self.assertTrue(expected <= set(result.game_knowledge_validation["claims_ungrounded"]))

    def test_d_partial_knowledge_allows_only_available_claim_categories(self):
        bundle = build_universal_speech_act_bundle(
            route="partial_game",
            speech_act_type="direct_answer",
            input_text="¿De cuándo es Partial Fixture?",
            current_game="Partial Fixture",
            memory={"game_knowledge": {
                "query_detected": True,
                "queried_game": "Partial Fixture",
                "game_knowledge_status": "PARTIAL",
                "allowed_claims": ["release_year=2012"],
            }},
        )

        grounded = final_response_guard("Salió en 2012.", bundle, game_advice_gate=GameAdviceGate())
        invented = final_response_guard("Salió en 2012 y es un JRPG táctico.", bundle, game_advice_gate=GameAdviceGate())

        self.assertTrue(grounded.passed)
        self.assertFalse(invented.passed)
        self.assertEqual(invented.game_knowledge_validation["claims_ungrounded"], ["genre"])

    def test_e_successful_lookup_propagates_only_sourced_claims(self):
        store = self.store()
        provider = SearchProvider([{
            "claim": "Verified Fixture es una aventura de puzles centrada en exploración.",
            "excerpt": "Una aventura de puzles centrada en exploración.",
            "url": "https://example.test/verified-fixture",
        }])
        research = GameKnowledgeResearchService(
            store=store,
            config=GameKnowledgeResearchConfig(enabled=True),
            search_provider=provider,
        )
        resolver = GameKnowledgeResolver(
            profile_store=store,
            research_service=research,
            config=GameKnowledgeConfig(web_lookup_enabled=True, game_profile_web_lookup_enabled=True),
        )

        result = resolver.resolve(game="Verified Fixture", stream=StreamSessionState())

        self.assertEqual(result.game_knowledge_status, "LOOKUP_SUCCEEDED")
        self.assertTrue(result.lookup_used)
        self.assertTrue(provider.calls)
        self.assertTrue(result.claims)
        self.assertTrue(all(item["provenance"] for item in result.evidence))
        contract = result.to_state_changes()
        contract.update({"query_detected": True, "queried_game": "Verified Fixture"})
        bundle = build_universal_speech_act_bundle(
            route="lookup_success",
            speech_act_type="direct_answer",
            input_text="¿Cómo es Verified Fixture?",
            current_game="Verified Fixture",
            memory={"game_knowledge": contract},
        )
        grounded = final_response_guard(
            "Es una aventura de puzles con exploración.",
            bundle,
            game_advice_gate=GameAdviceGate(),
        )
        self.assertTrue(grounded.passed, grounded.to_dict())

    def test_f_failed_lookup_returns_no_profile_or_claims(self):
        store = self.store()
        provider = SearchProvider([])
        research = GameKnowledgeResearchService(
            store=store,
            config=GameKnowledgeResearchConfig(enabled=True),
            search_provider=provider,
        )
        resolver = GameKnowledgeResolver(
            profile_store=store,
            research_service=research,
            config=GameKnowledgeConfig(web_lookup_enabled=True, game_profile_web_lookup_enabled=True),
        )

        result = resolver.resolve(game="Empty Lookup Fixture", stream=StreamSessionState())

        self.assertEqual(result.game_knowledge_status, "LOOKUP_FAILED")
        self.assertTrue(result.lookup_used)
        self.assertEqual(result.claims, [])
        self.assertFalse(store.has_specific_profile(current_game="Empty Lookup Fixture"))
        self.assertIn("no he obtenido información fiable", result.fallback_text.casefold())
        contract = result.to_state_changes()
        contract.update({
            "query_detected": True,
            "query_intent": "factual",
            "queried_game": "Empty Lookup Fixture",
        })
        synth = ResponseSynthesizer(conversation_model=RecordingModel(["Es un RPG."]))
        reply = synth.synthesize(
            context=self.context("¿Qué sabes de Empty Lookup Fixture?", contract),
            deliberation=SimpleNamespace(plan=SimpleNamespace()),
            execution=self.chat_execution(),
        )
        self.assertIn("he intentado consultar", reply.casefold())
        self.assertNotIn("rpg", reply.casefold())

    def test_g_ambiguous_title_requests_clarification(self):
        profiles = [
            GameProfile(game_slug="saga_one", canonical_title="Saga One", aliases=["Saga"]),
            GameProfile(game_slug="saga_two", canonical_title="Saga Two", aliases=["Saga"]),
        ]
        resolver = GameKnowledgeResolver(profile_store=self.store(profiles), config=GameKnowledgeConfig())

        result = resolver.resolve(game="Saga", stream=StreamSessionState())

        self.assertEqual(result.game_knowledge_status, "AMBIGUOUS")
        self.assertEqual(result.response_mode, "ambiguous")
        self.assertEqual(set(result.ambiguous_candidates), {"Saga One", "Saga Two"})
        self.assertIn("dime cuál", result.fallback_text.casefold())

    def test_h_plausible_invented_name_gets_no_description(self):
        synth = ResponseSynthesizer(conversation_model=RecordingModel(["Es una aventura espacial."]))
        context = self.context(
            "¿Qué sabes de Chronicles of the Silver Eclipse?",
            {
                "query_detected": True,
                "queried_game": "Chronicles of the Silver Eclipse",
                "game_knowledge_status": "UNKNOWN",
                "allowed_claims": [],
            },
        )

        reply = synth.synthesize(context=context, deliberation=SimpleNamespace(plan=SimpleNamespace()), execution=self.chat_execution())

        self.assertIn("no tengo datos fiables", reply.casefold())
        self.assertNotIn("aventura espacial", reply.casefold())

    def test_i_unknown_gameplay_question_does_not_invent_mechanics(self):
        synth = ResponseSynthesizer(conversation_model=RecordingModel(["Tiene combate por turnos."]))
        context = self.context(
            "¿Cómo es el gameplay de Fictional Meridian?",
            {
                "query_detected": True,
                "query_intent": "gameplay",
                "gameplay_requested": True,
                "queried_game": "Fictional Meridian",
                "game_knowledge_status": "UNKNOWN",
                "allowed_claims": [],
            },
        )

        reply = synth.synthesize(context=context, deliberation=SimpleNamespace(plan=SimpleNamespace()), execution=self.chat_execution())

        self.assertIn("no tengo datos fiables", reply.casefold())
        self.assertNotIn("combate", reply.casefold())

    def test_j_owner_authority_does_not_relax_factual_grounding(self):
        model = RecordingModel(["Sí, es un RPG táctico."])
        synth = ResponseSynthesizer(conversation_model=model)
        context = self.context(
            "¿Me recomiendas Fictional Meridian?",
            {
                "query_detected": True,
                "recommendation_requested": True,
                "queried_game": "Fictional Meridian",
                "game_knowledge_status": "UNKNOWN",
                "allowed_claims": [],
            },
            source="ui",
        )

        reply = synth.synthesize(context=context, deliberation=SimpleNamespace(plan=SimpleNamespace()), execution=self.chat_execution())

        self.assertEqual(model.calls, [])
        self.assertIn("no tengo base fiable", reply.casefold())

    def test_k_game_advice_gate_still_blocks_unknown_mechanics(self):
        result = GameAdviceGate().validate(
            current_game="Unknown Fixture",
            proposed_advice="Baton Pass aumenta el daño.",
            source_evidence=[],
        )

        self.assertFalse(result.allowed)
        self.assertIn("baton_pass", result.blocked)

    def test_l_nonfactual_creative_opinion_is_not_blocked(self):
        bundle = build_universal_speech_act_bundle(
            route="creative_game_reaction",
            speech_act_type="direct_answer",
            input_text="Ese título suena dramático",
            current_game="Fictional Meridian",
            memory={"game_knowledge": {
                "query_detected": True,
                "queried_game": "Fictional Meridian",
                "game_knowledge_status": "UNKNOWN",
                "allowed_claims": [],
            }},
        )

        result = final_response_guard(
            "El título tiene energía de tragedia con presupuesto, y eso me hace gracia.",
            bundle,
            game_advice_gate=GameAdviceGate(),
        )

        self.assertTrue(result.passed, result.to_dict())

    def test_self_explanation_uses_unknown_knowledge_outcome(self):
        stream = StreamSessionState()
        history = RecentInteractionDecisionHistory()
        history.upsert(stream, {
            "trace_id": "unknown-recommendation",
            "actor": "ViewerFixture",
            "actor_identities": ["ViewerFixture", "viewer_fixture"],
            "requested_effect": "game_recommendation",
            "reply_authorized": True,
            "reason_code": "game_knowledge_unknown",
            "game_knowledge_status": "UNKNOWN",
            "evidence_count": 0,
            "lookup_used": False,
            "claim_count": 0,
            "ungrounded_claim_blocked": False,
            "emission_outcome": "emitted",
        })
        query = detect_self_explanation_query(
            "¿Por qué no me recomendaste el juego?",
            requester="ViewerFixture",
            known_identities=["ViewerFixture"],
        )

        decision = history.resolve(stream, query)
        explanation = render_grounded_self_explanation(query, decision, requester="ViewerFixture")

        self.assertTrue(query.detected)
        self.assertEqual(explanation.source_trace_id, "unknown-recommendation")
        self.assertIn("datos fiables suficientes", explanation.text.casefold())

    def test_viewer_flow_emits_one_honest_reply_and_structured_outcome(self):
        engine = make_engine(["viewer_fixture"])
        engine.runtime.state.stream.is_live = True
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.game_context_resolver = None
        engine.game_profiles = self.store()
        engine.game_knowledge = GameKnowledgeResolver(
            profile_store=engine.game_profiles,
            config=GameKnowledgeConfig(web_lookup_enabled=False, game_profile_web_lookup_enabled=False),
        )
        model = RecordingModel(["Sí, es un JRPG táctico de 2025."])
        engine.response_synthesizer = ResponseSynthesizer(conversation_model=model)
        engine.response_synthesizer._dataset_logger.log_twitch_chat_react = lambda **kwargs: None
        event = SimpleNamespace(event_type="twitch_chat_react", payload={
            "event_id": "unknown-viewer-flow",
            "user_login": "viewer_fixture",
            "display_name": "ViewerFixture",
            "message_text": "Hebe, ¿me recomiendas Fictional Meridian?",
        })

        with patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            engine.process_internal_event(event)

        self.assertEqual(model.calls, [])
        self.assertEqual(len(engine.runtime.twitch.sent), 1)
        self.assertIn("no tengo base fiable suficiente", engine.runtime.twitch.sent[0].casefold())
        outcome = engine.runtime.state.stream.last_game_knowledge_outcome
        self.assertEqual(outcome["game_knowledge_status"], "UNKNOWN")
        self.assertEqual(outcome["evidence_count"], 0)
        self.assertFalse(outcome["lookup_used"])
        self.assertEqual(outcome["claim_count"], 0)
        self.assertFalse(outcome["ungrounded_claim_blocked"])
        self.assertEqual(outcome["emission_outcome"], "emitted")
        self.assertEqual(outcome["requested_effect"], "game_recommendation")
        self.assertEqual(outcome["reason_code"], "game_knowledge_unknown")


if __name__ == "__main__":
    unittest.main()
