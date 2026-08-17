from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.cognitive.cognitive_router import CAP_GAME_GUIDANCE, CAP_OPEN_APP, CognitiveRouter
from app.cognitive.deliberation_service import DeliberationService
from app.cognitive.input_event import InputEvent
from app.cognitive.input_interpretation import InputInterpreter, InputSpeechAct
from app.cognitive.local_app_planner import LocalAppActionPlanner
from app.hebe_engine import HebeEngine
from app.stream.ambient_context import AmbientContextExtractor
from app.stream.behavior_adaptation import BehaviorAdaptationService
from tests.test_voice_command_pipeline import make_engine, wire_canonical_app_pipeline


DOOR_FEEDBACK = (
    "Mira, otra vez con lo de abre la puerta esa. Lleva todo el puto stream "
    "diciéndome que abra la puerta, pero te quieres callar con la puta puerta "
    "que no hay ninguna puerta"
)


class CanonicalInputInterpretationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.interpreter = InputInterpreter()

    def owner(
        self,
        text: str,
        *,
        addressed: bool = False,
        recent: str = "",
        social_identities: list[str] | None = None,
    ):
        return self.interpreter.interpret(
            raw_text=text,
            source="stt_voice",
            authority="owner",
            addressed_to_hebe=addressed,
            recent_hebe_utterance=recent,
            social_identities=social_identities,
        )

    @staticmethod
    def route(text: str, interpretation, *, source: str = "stt_voice", authority: str = "owner"):
        return CognitiveRouter().route(SimpleNamespace(
            input_text=text,
            source=source,
            authority=authority,
            addressed_to_hebe=interpretation.addressed_to_hebe,
            input_interpretation=interpretation,
            state_snapshot={},
            firewall_decision="allow",
            stream_is_live=True,
            route_hints=[],
            internal_event=None,
        ))

    def test_authorized_owner_app_command_is_distinct_from_possible_syntax(self):
        result = self.interpreter.interpret(
            raw_text="Hebe, abre Steam",
            source="ui",
            authority="owner",
            addressed_to_hebe=True,
        )

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_COMMAND)
        self.assertTrue(result.possible_command_syntax)
        self.assertTrue(result.authorized_action_command)
        self.assertEqual(self.route("Hebe, abre Steam", result, source="ui").intent, "command_open_app")

    def test_reported_app_command_is_feedback_and_cannot_plan_an_action(self):
        text = "He dicho 'abre Steam' tres veces, deja ya de insistir."
        result = self.owner(text)
        event = InputEvent("stt_voice", text, text, interpretation=result)

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_FEEDBACK)
        self.assertFalse(result.authorized_action_command)
        self.assertIsNone(LocalAppActionPlanner().plan(event))

    def test_real_door_incident_is_negative_owner_feedback_with_no_domain_side_effect(self):
        result = self.owner(DOOR_FEEDBACK)
        decision = self.route(DOOR_FEEDBACK, result)
        extraction = AmbientContextExtractor().extract(
            DOOR_FEEDBACK,
            input_interpretation=result,
        )
        service = DeliberationService(intent_model=None, reasoning_model=None)
        service.local_app_planner.plan = Mock(side_effect=AssertionError("app planner must not run"))
        context = SimpleNamespace(
            input_text=DOOR_FEEDBACK,
            internal_event=None,
            state_snapshot={},
            resolved_entities=[],
            message_type="owner_feedback",
            source="stt_voice",
            authority="owner",
            addressed_to_hebe=result.addressed_to_hebe,
            input_interpretation=result,
            cognitive_decision=decision,
        )
        plan = service.deliberate(context).plan

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_FEEDBACK)
        self.assertEqual(result.authority, "owner")
        self.assertTrue(result.meta_about_hebe)
        self.assertEqual(result.feedback.polarity, "negative")
        self.assertIn("puerta", result.feedback.referent)
        self.assertFalse(result.authorized_action_command)
        self.assertEqual(decision.intent, "owner_feedback")
        self.assertNotIn(CAP_OPEN_APP, decision.allowed_capabilities)
        self.assertNotIn(CAP_GAME_GUIDANCE, decision.allowed_capabilities)
        self.assertFalse(extraction.useful)
        self.assertEqual(extraction.reason, "canonical_feedback_scope_excluded")
        self.assertTrue(all(step.type != "action" for step in plan.steps))
        service.local_app_planner.plan.assert_not_called()

    def test_real_door_incident_replay_never_reaches_app_execution_or_game_failure(self):
        engine = wire_canonical_app_pipeline(make_engine())
        engine._deliver_manual_reply = lambda _text, *, source: None
        planner = engine.deliberation_service.local_app_planner
        planner.plan = Mock(wraps=planner.plan)

        with patch("app.hebe_engine.log_chat"):
            outcome = engine._process_stt_voice_transcript(DOOR_FEEDBACK)

        self.assertEqual(outcome, "continue")
        self.assertEqual(engine.runtime.win.opened, [])
        planner.plan.assert_not_called()
        self.assertEqual(engine._last_cognitive_trace["intent"], "owner_feedback")
        self.assertEqual(engine.runtime.state.stream.last_voice_event, "owner_feedback")
        self.assertEqual(
            engine.runtime.state.stream.last_owner_feedback["feedback"]["polarity"],
            "negative",
        )
        categories = {
            fact.get("category")
            for fact in engine.runtime.state.stream.recent_run_context_facts
        }
        self.assertNotIn("failure_or_death", categories)

    def test_narrated_door_action_is_commentary_not_command(self):
        text = "y entonces abre la puerta y aparece el jefe"
        result = self.owner(text)

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_COMMENTARY)
        self.assertFalse(result.possible_command_syntax)
        self.assertFalse(result.authorized_action_command)

    def test_explicit_negative_feedback(self):
        result = self.owner("Hebe, deja de repetir eso.", addressed=True)

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_FEEDBACK)
        self.assertEqual(result.feedback.polarity, "negative")
        self.assertEqual(result.feedback.target, "repeated_hebe_behavior")

    def test_explicit_positive_feedback(self):
        result = self.owner("Eso sí me ha hecho gracia, Hebe.")

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_FEEDBACK)
        self.assertEqual(result.feedback.polarity, "positive")
        self.assertEqual(result.feedback.target, "hebe_response")

    def test_elliptical_praise_requires_recent_hebe_context(self):
        without_context = self.owner("Buenísima esa.")
        with_context = self.owner("Buenísima esa.", recent="Respuesta anterior de Hebe")

        self.assertEqual(without_context.speech_act, InputSpeechAct.OWNER_COMMENTARY)
        self.assertEqual(with_context.speech_act, InputSpeechAct.OWNER_FEEDBACK)
        self.assertEqual(with_context.feedback.referent, "previous_hebe_utterance")

    def test_viewer_directed_command_never_acquires_owner_pc_authority(self):
        text = "Hebe, abre OBS"
        result = self.interpreter.interpret(
            raw_text=text,
            source="twitch_viewer",
            authority="viewer",
            addressed_to_hebe=True,
        )
        decision = self.route(text, result, source="twitch_viewer", authority="viewer")

        self.assertEqual(result.speech_act, InputSpeechAct.VIEWER_DIRECTED_TO_HEBE)
        self.assertTrue(result.possible_command_syntax)
        self.assertFalse(result.authorized_action_command)
        self.assertNotIn(CAP_OPEN_APP, decision.allowed_capabilities)

    def test_viewer_report_about_owner_is_context_only(self):
        text = "Leo abre OBS siempre al empezar"
        result = self.interpreter.interpret(
            raw_text=text,
            source="twitch_viewer",
            authority="viewer",
            addressed_to_hebe=False,
        )

        self.assertEqual(result.speech_act, InputSpeechAct.VIEWER_CONTEXT)
        self.assertFalse(result.authorized_action_command)
        self.assertTrue(result.context_eligible)

    def test_valid_pending_owner_answer_has_its_own_primary_act(self):
        result = self.interpreter.interpret(
            raw_text="A las cinco",
            source="owner_stt_followup",
            authority="owner",
            addressed_to_hebe=False,
            pending_valid=True,
        )

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_ANSWER_FOLLOWUP)
        self.assertFalse(result.authorized_action_command)

    def test_real_repeated_game_failure_remains_game_context(self):
        text = "otra vez me ha matado el jefe"
        result = self.owner(text)
        engine = HebeEngine.__new__(HebeEngine)
        event_type, mood = engine._classify_voice_event(text, interpretation=result)
        extraction = AmbientContextExtractor().extract(text, event_type=event_type, input_interpretation=result)

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_COMMENTARY)
        self.assertEqual(event_type, "gameplay_failure")
        self.assertEqual(mood, "frustrated")
        self.assertTrue(extraction.useful)
        self.assertIn("failure_or_death", {fact["category"] for fact in extraction.facts})

    def test_mixed_feedback_and_game_state_preserves_only_separate_context_clause(self):
        text = "Hebe, deja de decirme que cure; estoy a 1 HP."
        result = self.owner(text, addressed=True)
        extraction = AmbientContextExtractor().extract(text, input_interpretation=result)

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_FEEDBACK)
        self.assertEqual(result.feedback.polarity, "negative")
        self.assertEqual(result.context_text, "estoy a 1 HP.")
        self.assertTrue(result.context_eligible)
        self.assertNotEqual(extraction.reason, "canonical_feedback_scope_excluded")
        self.assertTrue(all("deja de decirme" not in fact.get("raw_text", "").lower() for fact in extraction.facts))

    def test_negated_completed_death_is_not_promoted_to_failure(self):
        text = "Pero si no he muerto todavía."
        result = self.owner(text)
        engine = HebeEngine.__new__(HebeEngine)

        event_type, _ = engine._classify_voice_event(text, interpretation=result)
        extraction = AmbientContextExtractor().extract(
            text, event_type=event_type, input_interpretation=result,
        )

        predicate = result.semantic_clauses[0].predicates[0]
        self.assertEqual(predicate.predicate, "completed_death")
        self.assertEqual(predicate.polarity, "negative")
        self.assertIn("completed_death", result.negated_predicates)
        self.assertNotEqual(event_type, "gameplay_failure")
        self.assertNotIn("failure_or_death", {fact["category"] for fact in extraction.facts})
        self.assertEqual(extraction.reason, "negated_gameplay_predicate")

    def test_asserted_completed_death_remains_failure(self):
        text = "Me han matado otra vez."
        result = self.owner(text)
        engine = HebeEngine.__new__(HebeEngine)

        event_type, _ = engine._classify_voice_event(text, interpretation=result)
        extraction = AmbientContextExtractor().extract(
            text, event_type=event_type, input_interpretation=result,
        )

        self.assertEqual(result.semantic_clauses[0].predicates[0].polarity, "positive")
        self.assertEqual(event_type, "gameplay_failure")
        self.assertIn("failure_or_death", {fact["category"] for fact in extraction.facts})

    def test_near_death_is_risk_not_completed_failure(self):
        text = "Estoy a punto de morir."
        result = self.owner(text)
        engine = HebeEngine.__new__(HebeEngine)

        event_type, _ = engine._classify_voice_event(text, interpretation=result)
        extraction = AmbientContextExtractor().extract(
            text, event_type=event_type, input_interpretation=result,
        )

        predicate = result.semantic_clauses[0].predicates[0]
        self.assertEqual((predicate.predicate, predicate.polarity), ("death_risk", "uncertain"))
        self.assertEqual(event_type, "combat_risk")
        self.assertIn("combat_risk", {fact["category"] for fact in extraction.facts})
        self.assertNotIn("failure_or_death", {fact["category"] for fact in extraction.facts})

    def test_aspectual_no_deja_de_is_not_owner_feedback(self):
        result = self.owner("No deja de ser un capricho.")

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_COMMENTARY)
        self.assertIsNone(result.feedback)
        self.assertEqual(result.semantic_clauses[0].semantic_role, "descriptive_clause")
        self.assertIn("behavior_feedback", result.semantic_clauses[0].excluded_domains)

    def test_weather_stop_surface_is_not_assistant_feedback_or_command(self):
        result = self.owner("Deja de llover.", recent="Respuesta reciente de Hebe")

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_COMMENTARY)
        self.assertIsNone(result.feedback)
        self.assertFalse(result.possible_command_syntax)
        self.assertEqual(result.semantic_clauses[0].subject, "weather")
        self.assertIn("behavior_feedback", result.semantic_clauses[0].excluded_domains)

    def test_death_predicate_polarity_is_clause_bounded(self):
        cases = (
            ("Me he muerto.", "completed_death", "positive"),
            ("No me he muerto.", "completed_death", "negative"),
            ("Casi me muero.", "death_risk", "uncertain"),
            ("Pensé que había muerto.", "completed_death", "uncertain"),
        )
        for text, predicate_name, polarity in cases:
            with self.subTest(text=text):
                result = self.owner(text)
                predicate = next(
                    item for item in result.semantic_clauses[0].predicates
                    if item.predicate == predicate_name
                )
                self.assertEqual(predicate.polarity, polarity)

        reported = self.owner("Pensé que había muerto.")
        extraction = AmbientContextExtractor().extract(
            "Pensé que había muerto.", input_interpretation=reported,
        )
        self.assertNotIn("failure_or_death", {fact["category"] for fact in extraction.facts})
        self.assertEqual(extraction.reason, "uncertain_gameplay_predicate")

    def test_resolved_other_vocative_prevents_hebe_feedback(self):
        result = self.owner(
            "Natti, deja de liarla.",
            social_identities=["Natti"],
        )

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_COMMENTARY)
        self.assertIsNone(result.feedback)
        self.assertEqual(result.resolved_addressee, "Natti")
        self.assertEqual(
            result.semantic_clauses[0].addressee_provenance,
            "resolved_social_identity_vocative",
        )
        self.assertIn("behavior_feedback", result.semantic_clauses[0].excluded_domains)

    def test_real_long_social_address_stays_out_of_hebe_feedback(self):
        result = self.owner(
            "Como siempre liándola en el chat... Natti, eres la hostia, deja de liarla.",
            social_identities=["Natti"],
        )

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_COMMENTARY)
        self.assertIsNone(result.feedback)
        social_clause = next(
            clause for clause in result.semantic_clauses
            if clause.resolved_addressee == "Natti"
        )
        self.assertEqual(social_clause.semantic_role, "social_addressed_clause")
        self.assertIn("behavior_feedback", social_clause.excluded_domains)

    def test_engine_supplies_recent_canonical_social_identities_to_interpreter(self):
        engine = make_engine()
        engine.social_world = SimpleNamespace(
            recent_identity_names=Mock(return_value=["Natti"]),
        )

        event = engine._build_input_event(
            source="stt_voice",
            raw_text="Natti, deja de liarla.",
            normalized_text="natti deja de liarla",
        )

        self.assertEqual(event.interpretation.resolved_addressee, "Natti")
        self.assertEqual(event.interpretation.speech_act, InputSpeechAct.OWNER_COMMENTARY)
        engine.social_world.recent_identity_names.assert_called_once_with(limit=80)

    def test_proxy_stop_request_targets_third_party_not_hebe_behavior(self):
        result = self.owner(
            "Hebe, dile a Natti que deje de liarla.",
            addressed=True,
            social_identities=["Natti"],
        )

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_COMMAND)
        self.assertIsNone(result.feedback)
        self.assertEqual(result.semantic_clauses[0].subject, "Natti")
        self.assertEqual(result.semantic_clauses[0].semantic_role, "owner_command")
        self.assertIn("behavior_feedback", result.semantic_clauses[0].excluded_domains)

    def test_game_subject_aspectual_stop_is_gameplay_not_feedback(self):
        text = "El jefe no deja de atacar."
        result = self.owner(text)
        engine = HebeEngine.__new__(HebeEngine)
        event_type, _ = engine._classify_voice_event(text, interpretation=result)
        extraction = AmbientContextExtractor().extract(
            text, event_type=event_type, input_interpretation=result,
        )

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_COMMENTARY)
        self.assertIsNone(result.feedback)
        self.assertEqual(result.semantic_clauses[0].subject, "game_entity")
        self.assertIn("enemy_attack_pattern", {fact["category"] for fact in extraction.facts})

    def test_medial_hebe_vocative_resolves_explicit_recent_behavior_feedback(self):
        result = self.owner("Eso que acabas de decir, Hebe, deja de hacerlo.")

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_FEEDBACK)
        self.assertEqual(result.feedback_target, "Hebe")
        self.assertEqual(result.feedback_target_provenance, "explicit_hebe_vocative")
        self.assertEqual(result.feedback.referent, "hacerlo")

    def test_third_party_subject_request_does_not_create_hebe_feedback_state(self):
        result = self.owner(
            "Eso que acaba de hacer Natti, que deje de hacerlo.",
            social_identities=["Natti"],
        )
        stream = SimpleNamespace(
            active_behavior_blocks=[], behavior_adaptation_state={"entries": []},
            recent_idle_messages=[], current_discourse_topic="",
        )

        application = BehaviorAdaptationService().apply_feedback(stream, result, now=1000.0)

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_COMMENTARY)
        self.assertIsNone(result.feedback)
        self.assertEqual(result.semantic_clauses[0].subject, "Natti")
        self.assertIn("behavior_feedback", result.semantic_clauses[0].excluded_domains)
        self.assertFalse(application.applied)
        self.assertEqual(stream.behavior_adaptation_state["entries"], [])

    def test_mixed_feedback_excludes_feedback_clause_and_keeps_negated_death_plus_hp(self):
        text = "Hebe, deja de decirme que cure; no me he muerto, estoy a 1 HP."
        result = self.owner(text, addressed=True)
        extraction = AmbientContextExtractor().extract(text, input_interpretation=result)
        categories = {fact["category"] for fact in extraction.facts}

        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_FEEDBACK)
        self.assertEqual(result.context_text, "no me he muerto, estoy a 1 HP.")
        self.assertIn("completed_death", result.negated_predicates)
        self.assertIn("combat_risk", categories)
        self.assertNotIn("failure_or_death", categories)
        self.assertTrue(all("deja de decirme" not in fact.get("raw_text", "").lower() for fact in extraction.facts))

    def test_viewer_social_target_never_acquires_owner_feedback_authority(self):
        result = self.interpreter.interpret(
            raw_text="@Natti deja de liarla",
            source="twitch_viewer",
            authority="viewer",
            addressed_to_hebe=False,
            social_identities=["Natti"],
        )
        stream = SimpleNamespace(
            active_behavior_blocks=[], behavior_adaptation_state={"entries": []},
            recent_idle_messages=[], current_discourse_topic="",
        )
        application = BehaviorAdaptationService().apply_feedback(stream, result, now=1000.0)

        self.assertEqual(result.speech_act, InputSpeechAct.VIEWER_CONTEXT)
        self.assertIsNone(result.feedback)
        self.assertFalse(result.meta_about_hebe)
        self.assertEqual(result.resolved_addressee, "Natti")
        self.assertFalse(application.applied)
        self.assertEqual(stream.behavior_adaptation_state["entries"], [])

    def test_semantic_scope_observability_explains_negation_and_target(self):
        logged = []
        with patch(
            "app.cognitive.input_interpretation.log_jsonl_event",
            lambda kind, payload: logged.append((kind, payload)),
        ):
            negated = self.owner("No me he muerto.")
            social = self.owner(
                "Natti, deja de liarla.", social_identities=["Natti"],
            )

        negated_payload = logged[0][1]
        social_payload = logged[1][1]
        self.assertIn("completed_death", negated_payload["negated_predicates"])
        self.assertEqual(
            negated_payload["semantic_clauses"][0]["predicates"][0]["reason"],
            "clause_local_negation",
        )
        self.assertEqual(social_payload["resolved_addressee"], "Natti")
        self.assertIn("behavior_feedback", social_payload["semantic_clauses"][0]["excluded_domains"])


if __name__ == "__main__":
    unittest.main()
