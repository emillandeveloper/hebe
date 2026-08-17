from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.cognitive.interaction_history import (
    RecentInteractionDecisionHistory,
    detect_self_explanation_query,
)
from app.cognitive.models import ExecutionResult, StepExecutionResult
from app.cognitive.models import ActionResult
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.stream.policy import ViewerIntentPolicy
from app.stream.state import StreamSessionState
from tests.test_voice_command_pipeline import (
    FakeContextBuilder,
    FakeDeliberationService,
    FakePlanExecutor,
    SequentialResponseModel,
    make_engine,
)


def twitch_event(event_id: str, *, user: str, display: str, text: str):
    return SimpleNamespace(
        event_type="twitch_chat_react",
        payload={
            "event_id": event_id,
            "user_login": user,
            "display_name": display,
            "message_text": text,
        },
    )


def process_twitch(engine, event) -> None:
    with patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
        engine.process_internal_event(event)


class ViewerPolicySelfExplanationTests(unittest.TestCase):
    def make_live_engine(self, model=None):
        engine = make_engine(["viewer_one", "viewer_two"])
        engine.runtime.state.stream.is_live = True
        engine.response_synthesizer = ResponseSynthesizer(conversation_model=model)
        engine.response_synthesizer._dataset_logger.log_twitch_chat_react = lambda **kwargs: None
        return engine

    def ask_owner(self, engine, text: str) -> str:
        delivered = []
        engine._deliver_manual_reply = lambda reply, *, source: delivered.append(reply)
        with patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            result = engine.cognitive_flow(text, source="ui")
        self.assertEqual(result, "continue")
        self.assertEqual(len(delivered), 1)
        return delivered[0]

    def test_a_viewer_owner_behavior_request_denies_effect_and_emits_boundary(self):
        engine = self.make_live_engine()

        process_twitch(engine, twitch_event(
            "evt-boundary",
            user="viewer_one",
            display="ViewerOne",
            text="Hebe, la próxima le dices a Leo que beba agua",
        ))

        self.assertEqual(len(engine.runtime.twitch.sent), 1)
        self.assertNotIn("policy", engine.runtime.twitch.sent[0].casefold())
        history = engine.get_recent_interaction_decisions()
        decision = next(item for item in history if item["trace_id"] == "evt-boundary")
        self.assertEqual(decision["interaction_decision"], "deny_action_reply")
        self.assertFalse(decision["effect_authorized"])
        self.assertTrue(decision["reply_authorized"])
        self.assertEqual(decision["reason_code"], "viewer_repeat_to_leo_request")
        self.assertEqual(decision["response_intent"], "hebe_playful_boundary")
        self.assertEqual(decision["emission_outcome"], "emitted")
        self.assertEqual(engine.runtime.state.stream.viewer_policy_cooldowns["viewer_one:viewer_repeat_to_leo_request"]["count"], 1)

    def test_b_repeated_policy_requests_never_gain_authority_or_disable_reply_contract(self):
        stream = StreamSessionState()
        policy = ViewerIntentPolicy()

        decisions = [
            policy.decide(
                stream,
                username="viewer_one",
                display_name="ViewerOne",
                text="Hebe, recuérdale a Leo que beba agua",
                now=1000.0 + index,
            )
            for index in range(5)
        ]

        self.assertTrue(all(decision.allow_reply for decision in decisions))
        self.assertTrue(all(not decision.execute_as_command for decision in decisions))
        self.assertTrue(all(decision.reason == "viewer_repeat_to_leo_request" for decision in decisions))
        self.assertEqual([decision.boundary_repeat_count for decision in decisions], [1, 2, 3, 4, 5])

    def test_c_owner_why_question_uses_viewer_authority_trace(self):
        engine = self.make_live_engine()
        process_twitch(engine, twitch_event(
            "evt-owner-why-source",
            user="viewer_one",
            display="ViewerOne",
            text="Hebe, dile a Leo que pare ahora",
        ))

        reply = self.ask_owner(engine, "¿Por qué no le hiciste caso a ViewerOne?")

        self.assertIn("marqué el límite", reply)
        self.assertIn("ViewerOne", reply)
        self.assertNotIn("modo ninja", reply.casefold())
        trace = engine.runtime.state.stream.last_self_explanation
        self.assertEqual(trace["explanation_source_trace"], "evt-owner-why-source")
        self.assertEqual(trace["explanation_reason_code"], "viewer_repeat_to_leo_request")

    def test_d_viewer_why_me_question_uses_own_recent_outcome(self):
        engine = self.make_live_engine()
        process_twitch(engine, twitch_event(
            "evt-viewer-source",
            user="viewer_one",
            display="ViewerOne",
            text="Hebe, dile a Leo que mire el chat",
        ))

        process_twitch(engine, twitch_event(
            "evt-viewer-why",
            user="viewer_one",
            display="ViewerOne",
            text="Hebe, ¿por qué me ignoraste?",
        ))

        self.assertEqual(len(engine.runtime.twitch.sent), 2)
        self.assertIn("No te ignoré", engine.runtime.twitch.sent[-1])
        self.assertEqual(
            engine.runtime.state.stream.last_self_explanation["explanation_source_trace"],
            "evt-viewer-source",
        )

    def test_e_missing_recent_outcome_does_not_invent_cause(self):
        engine = self.make_live_engine()

        reply = self.ask_owner(engine, "¿Por qué no contestaste?")

        self.assertIn("no tengo suficiente contexto reciente", reply.casefold())
        self.assertEqual(
            engine.runtime.state.stream.last_self_explanation["explanation_reason_code"],
            "insufficient_recent_context",
        )

    def test_f_app_not_found_explains_only_the_observed_failure(self):
        engine = self.make_live_engine()
        engine._current_input_event = SimpleNamespace(
            source="ui",
            stt_metadata={"interaction_trace_id": "input-app-not-found"},
        )
        action = ActionResult(
            success=False,
            error="app_not_found",
            data={"error_code": "app_not_found", "app_id": "sample_app", "app_name": "Sample App"},
        )
        execution = ExecutionResult([StepExecutionResult(
            step_type="action",
            success=False,
            data={
                "action_name": "open_application",
                "params": {"requested_target": "sample_app"},
                "action_result": action,
            },
        )])
        engine._record_canonical_open_app_execution(execution)

        reply = self.ask_owner(engine, "¿Por qué no abriste la app?")

        self.assertIn("no encontré una instalación válida", reply.casefold())
        self.assertNotIn("permiso", reply.casefold())
        self.assertEqual(
            engine.runtime.state.stream.last_self_explanation["explanation_reason_code"],
            "app_not_found",
        )

    def test_g_behavior_suppression_explains_repetition_without_new_motive(self):
        engine = self.make_live_engine()
        engine._record_interaction_decision({
            "trace_id": "behavior-repeat",
            "actor": "Hebe",
            "actor_identities": ["Hebe"],
            "target": "stream",
            "interaction_decision": "behavior_candidate_suppressed",
            "authority": "system",
            "requested_effect": "behavior_expression",
            "effect_authorized": False,
            "reply_authorized": False,
            "reason_code": "behavior_repetition_fatigue_suppressed",
            "response_intent": "proactive_gag",
            "generation_outcome": "generated",
            "emission_outcome": "suppressed",
        })

        reply = self.ask_owner(engine, "¿Por qué no dijiste ese gag?")

        self.assertIn("repetido", reply.casefold())
        self.assertNotIn("me apetecía", reply.casefold())
        self.assertEqual(
            engine.runtime.state.stream.last_self_explanation["explanation_source_trace"],
            "behavior-repeat",
        )

    def test_h_terminal_generation_failure_is_explainable_after_public_fallback(self):
        model = SequentialResponseModel(["", "", "", "", "", ""])
        engine = self.make_live_engine(model)
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        process_twitch(engine, twitch_event(
            "evt-generation-failure",
            user="viewer_one",
            display="ViewerOne",
            text="Hebe, ¿qué harías para salir de un bloqueo creativo?",
        ))

        reply = self.ask_owner(engine, "¿Por qué no contestaste bien?")

        self.assertIn("respuesta suficientemente buena", reply.casefold())
        self.assertIn("humo", reply.casefold())
        self.assertEqual(
            engine.runtime.state.stream.last_self_explanation["explanation_reason_code"],
            "directed_viewer_generation_failed",
        )

    def test_i_policy_boundary_generation_failure_uses_one_deterministic_fallback(self):
        model = SequentialResponseModel(["", "", ""])
        engine = self.make_live_engine(model)

        process_twitch(engine, twitch_event(
            "evt-boundary-generation-failure",
            user="viewer_one",
            display="ViewerOne",
            text="Hebe, dile a Leo que lea esto",
        ))

        self.assertEqual(len(model.calls), 3)
        self.assertEqual(len(engine.runtime.twitch.sent), 1)
        self.assertTrue(engine.runtime.twitch.sent[0].strip())
        decision = next(
            item for item in engine.get_recent_interaction_decisions()
            if item["trace_id"] == "evt-boundary-generation-failure"
        )
        self.assertEqual(decision["generation_outcome"], "fallback_template")
        self.assertEqual(decision["emission_outcome"], "emitted")

    def test_j_named_viewer_resolution_does_not_confuse_two_identities(self):
        stream = StreamSessionState()
        history = RecentInteractionDecisionHistory()
        for trace_id, actor, reason in (
            ("trace-one", "ViewerOne", "viewer_repeat_to_leo_request"),
            ("trace-two", "ViewerTwo", "directed_viewer_generation_failed"),
        ):
            history.upsert(stream, {
                "trace_id": trace_id,
                "actor": actor,
                "actor_identities": [actor, actor.casefold()],
                "requested_effect": "reply",
                "reply_authorized": True,
                "reason_code": reason,
                "emission_outcome": "emitted",
            })
        query = detect_self_explanation_query(
            "¿Por qué ignoraste a ViewerOne?",
            requester="Leo",
            known_identities=["ViewerOne", "ViewerTwo"],
        )

        resolved = history.resolve(stream, query)

        self.assertEqual(resolved["trace_id"], "trace-one")

    def test_golden_current_game_factual_answer_uses_grounded_release_claims(self):
        model = SequentialResponseModel([
            "Salió en 2008 en Japón y en 2009 en Occidente.",
        ])
        synth = ResponseSynthesizer(conversation_model=model)
        context = SimpleNamespace(
            internal_event=None,
            input_text="¿De cuándo es el juego que estoy jugando, Hebe?",
            message_type="direct_question",
            source="stt_voice",
            state_snapshot={},
            resolved_entities=[],
            relevant_chunks=[],
            relevant_facts=[],
            inject_memory=False,
            context_policy={"memory": "limited"},
            response_frame={
                "current_game": "Super Robot Taisen OG Saga: Endless Frontier",
                "current_session_context": {
                    "game_intelligence": {
                        "allowed_claims": [
                            "release_japan=2008",
                            "release_west=2009",
                        ],
                        "source_provenance": ["fixture:release_history"],
                    },
                },
            },
        )
        execution = ExecutionResult([StepExecutionResult(
            step_type="reply",
            success=True,
            data={"mode": "chat"},
        )])

        reply = synth.synthesize(
            context=context,
            deliberation=SimpleNamespace(plan=SimpleNamespace()),
            execution=execution,
        )

        self.assertIn("2008", reply)
        self.assertIn("2009", reply)
        self.assertEqual(synth.last_response_source, "persona_generated")
        prompt = "\n".join(message["content"] for message in model.calls[0][0])
        self.assertIn("Super Robot Taisen OG Saga: Endless Frontier", prompt)
        self.assertIn("release_japan=2008", prompt)
        self.assertIn("release_west=2009", prompt)
        self.assertNotIn("grounded_self_explanation", prompt)


if __name__ == "__main__":
    unittest.main()
