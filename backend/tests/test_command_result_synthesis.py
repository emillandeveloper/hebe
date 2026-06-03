import unittest

from app.cognitive.command_result import CommandResult
from app.cognitive.response_synthesizer import ResponseSynthesizer


class FakeModel:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.text


class CommandResultSynthesisTests(unittest.TestCase):
    def test_generated_reply_is_used_when_valid(self):
        model = FakeModel("Listo, mi señor. Voz local encendida y el directo sigue en texto.")
        synth = ResponseSynthesizer(conversation_model=model)
        result = CommandResult(
            action_type="tts_scope_resolved",
            user_visible_summary="Voice is enabled locally only; stream remains text-only.",
            state_changes={"tts_enabled": True, "stream_idle_tts": False},
            fallback_text="Perfecto, voz activada solo aquí. En stream seguiré en texto salvo que me digas lo contrario.",
            metadata={"scope": "local", "message_goal": "Confirm local voice only."},
        )

        reply = synth.synthesize_command_result(result, input_text="local")

        self.assertEqual(reply, "Listo, mi señor. Voz local encendida y el directo sigue en texto.")
        self.assertTrue(model.calls)

    def test_question_after_resolved_scope_uses_fallback(self):
        model = FakeModel("¿La quieres también para stream?")
        synth = ResponseSynthesizer(conversation_model=model)
        result = CommandResult(
            action_type="tts_scope_resolved",
            user_visible_summary="Voice is enabled locally only; stream remains text-only.",
            state_changes={"tts_enabled": True, "stream_idle_tts": False},
            fallback_text="Perfecto, voz activada solo aquí. En stream seguiré en texto salvo que me digas lo contrario.",
            metadata={"scope": "local", "message_goal": "Confirm local voice only."},
        )

        reply = synth.synthesize_command_result(result, input_text="local")

        self.assertEqual(reply, result.fallback_text)

    def test_stream_enabled_claim_for_local_scope_uses_fallback(self):
        model = FakeModel("Perfecto, también para el stream queda activado.")
        synth = ResponseSynthesizer(conversation_model=model)
        result = CommandResult(
            action_type="tts_scope_resolved",
            user_visible_summary="Voice is enabled locally only; stream remains text-only.",
            state_changes={"tts_enabled": True, "stream_idle_tts": False},
            fallback_text="Perfecto, voz activada solo aquí. En stream seguiré en texto salvo que me digas lo contrario.",
            metadata={"scope": "local", "message_goal": "Confirm local voice only."},
        )

        reply = synth.synthesize_command_result(result, input_text="local")

        self.assertEqual(reply, result.fallback_text)


if __name__ == "__main__":
    unittest.main()
