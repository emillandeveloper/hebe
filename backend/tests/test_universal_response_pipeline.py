import unittest
from types import SimpleNamespace

from app.cognitive.models import ExecutionResult, StepExecutionResult
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.cognitive.speech_act_pipeline import (
    TestFakeProvider,
    action_claim_guard,
    build_universal_speech_act_bundle,
)


class CapturingModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(messages)
        return self.replies.pop(0) if self.replies else ""


def context(text="Hebe, abre OBS", source="stt_voice", message_type="task_request"):
    return SimpleNamespace(
        input_text=text,
        source=source,
        message_type=message_type,
        relevant_facts=[],
        relevant_chunks=[],
        conversation_history=[],
        response_frame={},
        state_snapshot={},
        cognitive_decision=SimpleNamespace(intent="command_open_app"),
        internal_event=None,
    )


class UniversalResponsePipelineTests(unittest.TestCase):
    def test_owner_open_obs_uses_universal_pipeline(self):
        model = CapturingModel(["OBS queda abierto, Leo."])
        synth = ResponseSynthesizer(conversation_model=model)
        execution = ExecutionResult([
            StepExecutionResult(
                step_type="action",
                success=True,
                data={"action_name": "open_application", "action_result": SimpleNamespace(data={"app_name": "OBS Studio"})},
            )
        ])

        reply = synth._generate_confirm_action(context(), execution)

        self.assertIn("OBS", reply)
        self.assertEqual(synth.last_response_source, "persona_generated")
        self.assertEqual(synth.last_response_debug_contract["speech_act_plan"]["speech_act_type"], "action_confirmation")
        self.assertTrue(synth.last_response_debug_contract["execution_result"]["success"])

    def test_action_confirmation_requires_execution_success(self):
        bundle = build_universal_speech_act_bundle(
            route="confirm_action",
            speech_act_type="action_failure",
            input_text="abre OBS",
            execution_result={"action": "open_application", "success": False},
        )

        result = action_claim_guard("Hecho, OBS abierto.", bundle)

        self.assertFalse(result.passed)
        self.assertIn("action_claim_without_execution_success", [v.type for v in result.violations])

    def test_action_failure_rendered_in_hebe_voice(self):
        model = CapturingModel(["No ha abierto, Leo. Me quedo con el fallo a la vista."])
        synth = ResponseSynthesizer(conversation_model=model)
        execution = ExecutionResult([
            StepExecutionResult(
                step_type="action",
                success=False,
                data={"action_name": "open_application", "action_result": SimpleNamespace(data={"app_name": "OBS Studio"})},
                error="missing executable",
            )
        ])

        reply = synth._generate_confirm_action(context(), execution)

        self.assertIn("Leo", reply)
        self.assertNotIn("como IA", reply.lower())
        self.assertEqual(synth.last_response_debug_contract["speech_act_plan"]["speech_act_type"], "action_failure")

    def test_current_time_uses_universal_pipeline(self):
        model = CapturingModel(["Son las 10:30, Leo."])
        synth = ResponseSynthesizer(conversation_model=model)
        execution = ExecutionResult([
            StepExecutionResult(step_type="reply", success=True, data={"mode": "time_answer", "time": "10:30"})
        ])

        reply = synth.synthesize(context("Hebe, que hora es"), SimpleNamespace(plan=SimpleNamespace(steps=[])), execution)

        self.assertIn("10:30", reply)
        self.assertEqual(synth.last_response_debug_contract["speech_act_plan"]["speech_act_type"], "direct_answer")

    def test_owner_personal_state_uses_universal_pipeline(self):
        model = CapturingModel(["Te leo, Leo. Baja una marcha y seguimos con cabeza."])
        synth = ResponseSynthesizer(conversation_model=model)

        reply = synth._generate_personal_state_reply(context("Hebe, estoy cansado", message_type="small_talk"), {"state": "tired"})

        self.assertIn("Leo", reply)
        self.assertEqual(synth.last_response_debug_contract["speech_act_plan"]["speech_act_type"], "owner_supportive_reaction")

    def test_game_guidance_uses_universal_pipeline(self):
        model = CapturingModel(["Con esa fuente, mira recursos antes de avanzar."])
        synth = ResponseSynthesizer(conversation_model=model)

        reply = synth._generate_game_guidance_reply(
            context("Hebe, que hago ahora"),
            {
                "game_guidance": {
                    "context": {"game": "Persona 5 Royal"},
                    "rag_chunks": ["SP management is relevant here."],
                }
            },
        )

        self.assertTrue(reply)
        self.assertEqual(synth.last_response_debug_contract["speech_act_plan"]["speech_act_type"], "game_guidance_answer")

    def test_renderer_provider_is_provider_agnostic(self):
        provider = TestFakeProvider(["Linea de prueba."])

        self.assertEqual(provider.render(system="s", user="u"), "Linea de prueba.")
        self.assertEqual(provider.provider_name, "test_fake")


if __name__ == "__main__":
    unittest.main()
