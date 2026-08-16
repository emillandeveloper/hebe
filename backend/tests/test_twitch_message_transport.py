import unittest

from app.cognitive.final_emission_gate import FinalEmissionGate, OutputRoute
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.cognitive.speech_act_pipeline import (
    HebeResponsePipeline,
    TestFakeProvider as FakeProvider,
    build_universal_speech_act_bundle,
)
from app.integrations.twitch.message_transport import split_twitch_message
from app.integrations.twitch.chat_client import TwitchChatClient
from app.integrations.twitch.service import TwitchService


class CapturingChatClient:
    def __init__(self, *, fail_at=None):
        self.sent = []
        self.fail_at = fail_at

    def send_message(self, text):
        attempt = len(self.sent) + 1
        if attempt == self.fail_at:
            return False
        self.sent.append(text)
        return True


class SequentialModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.replies.pop(0) if self.replies else ""


class TwitchMessageTransportTests(unittest.TestCase):
    def test_short_response_is_one_identical_message(self):
        chat = CapturingChatClient()
        service = TwitchService(chat_client=chat)
        text = "Una respuesta breve, sin tocar una coma."

        self.assertTrue(service.send_message(text))
        self.assertEqual(chat.sent, [text])

    def test_response_just_below_limit_is_one_identical_message(self):
        chat = CapturingChatClient()
        service = TwitchService(chat_client=chat, message_max_chars=50)
        text = "x" * 49

        self.assertTrue(service.send_message(text))
        self.assertEqual(chat.sent, [text])

    def test_low_level_chat_client_rejects_unplanned_over_limit_message(self):
        client = TwitchChatClient(enabled=True)

        self.assertFalse(client.send_message("x" * 501))

    def test_over_limit_chunks_reconstruct_original(self):
        text = "Primera idea completa. Segunda idea, con detalle y contexto. Tercera idea que cierra."
        plan = split_twitch_message(text, max_chars=35)

        self.assertGreater(len(plan.chunks), 1)
        self.assertTrue(all(len(chunk) <= 35 for chunk in plan.chunks))
        self.assertEqual(plan.reconstruct(), text)
        self.assertEqual(len(plan.separators), len(plan.chunks) - 1)

    def test_split_priority_is_sentence_then_punctuation_then_whitespace(self):
        sentence = split_twitch_message("Uno dos tres. Cuatro cinco seis siete.", max_chars=20)
        punctuation = split_twitch_message("Uno dos, tres cuatro cinco seis", max_chars=18)
        whitespace = split_twitch_message("uno dos tres cuatro cinco seis", max_chars=16)

        self.assertEqual(sentence.chunks[0], "Uno dos tres.")
        self.assertEqual(punctuation.chunks[0], "Uno dos,")
        self.assertTrue(whitespace.chunks[0].endswith("tres"))
        self.assertEqual(sentence.reconstruct(), "Uno dos tres. Cuatro cinco seis siete.")
        self.assertEqual(punctuation.reconstruct(), "Uno dos, tres cuatro cinco seis")

    def test_splitter_avoids_tiny_tail_when_an_earlier_boundary_can_redistribute(self):
        text = "alpha beta gamma delta epsilon zeta eta theta"
        plan = split_twitch_message(text, max_chars=25)

        self.assertEqual(plan.reconstruct(), text)
        self.assertGreaterEqual(len(plan.chunks[-1].split()), 3)

    def test_unicode_accents_em_dash_and_emoji_are_preserved(self):
        text = "Ánimo — prueba café ☕ y código 👩‍💻; después sonríe 🙂."
        plan = split_twitch_message(text, max_chars=14)

        self.assertEqual(plan.reconstruct(), text)
        self.assertTrue(all(len(chunk) <= 14 for chunk in plan.chunks))
        self.assertFalse(any(chunk.endswith("\u200d") for chunk in plan.chunks))
        self.assertFalse(any(chunk and ord(chunk[0]) in range(0x1F3FB, 0x1F400) for chunk in plan.chunks))

    def test_real_regression_is_not_cut_after_twenty_four_words(self):
        text = (
            "Natti, prueba cosas pequeñas — cambia rutina, escucha música distinta, pasea, "
            "copia a quien te inspire y no te exijas perfección. Cuando menos lo esperas, "
            "la idea suele encontrarte en movimiento."
        )
        synth = ResponseSynthesizer(conversation_model=None)

        guarded = synth._guard_twitch_reply(
            text,
            chatter="Viewer",
            message="Hebe, ¿cómo recupero la creatividad?",
            is_broadcaster=False,
        )

        self.assertEqual(guarded, text)
        self.assertNotEqual(guarded[-17:], "Cuando menos lo.")

    def test_style_too_long_repairs_but_transport_does_not_slice(self):
        long_reply = "Esta respuesta contiene demasiadas palabras para el objetivo de estilo del directo. " * 3
        concise_reply = "La creatividad vuelve mejor sin exigirle puntualidad."
        bundle = build_universal_speech_act_bundle(
            route="test_stream_style",
            speech_act_type="direct_answer",
            input_text="Hebe, ¿cómo recupero la creatividad?",
            source="twitch_viewer",
            output_target="twitch_chat",
            speaker="Viewer",
            authority="viewer",
            mode="stream",
            max_length_chars=80,
        )
        pipeline = HebeResponsePipeline(FakeProvider([long_reply, concise_reply]))

        result = pipeline.render(bundle, route="test_stream_style")
        chat = CapturingChatClient()
        service = TwitchService(chat_client=chat, message_max_chars=30)

        self.assertEqual(result.response_source, "persona_repair_generated")
        self.assertTrue(service.send_message(result.text))
        self.assertEqual(split_twitch_message(result.text, max_chars=30).reconstruct(), result.text)

    def test_chunk_two_failure_is_observable_and_not_full_success(self):
        chat = CapturingChatClient(fail_at=2)
        service = TwitchService(chat_client=chat, message_max_chars=24)
        gate = FinalEmissionGate()

        result = gate.emit(
            event_id="evt-partial",
            source="twitch",
            final_response="Primera oración completa. Segunda oración completa.",
            output_route=OutputRoute.TWITCH_TEXT_REPLY,
            output_targets=["twitch_chat"],
            guard_result={"passed": True},
            send_twitch=service.send_message,
            debug_payload={"response_stage": "final"},
        )

        self.assertFalse(result.emitted)
        self.assertEqual(result.reason, "twitch_delivery_failed")
        self.assertFalse(service.last_delivery_outcome["success"])
        self.assertEqual(service.last_delivery_outcome["sent_chunks"], 1)
        self.assertEqual(service.last_delivery_outcome["failed_chunk"], 2)


class DirectedViewerFallbackTests(unittest.TestCase):
    @staticmethod
    def _synth(replies):
        model = SequentialModel(replies)
        synth = ResponseSynthesizer(conversation_model=model)
        synth._dataset_logger.log_twitch_chat_react = lambda **kwargs: None
        return synth, model

    def test_directed_casual_question_retries_generic_fallback_contextually(self):
        synth, model = self._synth([
            "",
            "",
            "",
            "La creatividad suele volver cuando cambias una rutina pequeña y pruebas sin exigirte perfección.",
        ])

        reply = synth._generate_twitch_chat_react({
            "user_login": "viewer",
            "display_name": "Viewer",
            "message_text": "Hebe, ¿cómo puedo recuperar la creatividad?",
            "direct_address_to_hebe": True,
            "recent_chat": [],
        })

        self.assertIn("creatividad", reply.casefold())
        self.assertEqual(len(model.calls), 4)
        self.assertNotEqual(synth.last_response_source, "local_safe_fallback")
        self.assertEqual(
            synth.last_response_debug_contract["directed_viewer_recovery"]["outcome"],
            "regenerated",
        )

    def test_directed_generation_failure_uses_one_public_safe_terminal_fallback(self):
        synth, model = self._synth(["", "", "", "", "", ""])

        reply = synth._generate_twitch_chat_react({
            "user_login": "viewer",
            "display_name": "Viewer",
            "message_text": "Hebe, ¿qué harías para salir de un bloqueo creativo?",
            "direct_address_to_hebe": True,
            "recent_chat": [],
        })

        self.assertEqual(
            reply,
            "Viewer, no tengo una buena respuesta para eso; prefiero no improvisarte humo.",
        )
        self.assertEqual(len(model.calls), 6)
        self.assertEqual(synth.last_response_source, "directed_viewer_terminal_fallback")
        self.assertEqual(
            synth.last_response_debug_contract["directed_viewer_recovery"]["outcome"],
            "terminal_fallback",
        )
        self.assertEqual(
            synth.last_response_debug_contract["directed_viewer_recovery"]["generation_outcome"],
            "failed",
        )


if __name__ == "__main__":
    unittest.main()
