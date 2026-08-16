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
from tests.test_voice_command_pipeline import make_engine, wire_canonical_app_pipeline


DOOR_FEEDBACK = (
    "Mira, otra vez con lo de abre la puerta esa. Lleva todo el puto stream "
    "diciéndome que abra la puerta, pero te quieres callar con la puta puerta "
    "que no hay ninguna puerta"
)


class CanonicalInputInterpretationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.interpreter = InputInterpreter()

    def owner(self, text: str, *, addressed: bool = False, recent: str = ""):
        return self.interpreter.interpret(
            raw_text=text,
            source="stt_voice",
            authority="owner",
            addressed_to_hebe=addressed,
            recent_hebe_utterance=recent,
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


if __name__ == "__main__":
    unittest.main()
