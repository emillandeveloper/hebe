import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.hebe_engine import HebeEngine
from app.stream.state import StreamSessionState


class FakeTwitch:
    def __init__(self):
        self.sent = []

    def is_available(self):
        return True

    def send_message(self, text):
        self.sent.append(text)


def make_engine(*, tts_enabled=True):
    engine = HebeEngine.__new__(HebeEngine)
    stream = StreamSessionState()
    engine.runtime = SimpleNamespace(
        state=SimpleNamespace(
            tts_enabled=tts_enabled,
            stream=stream,
            pending_tts_scope=None,
            pending_clarification=None,
            pending_reminder=None,
        ),
        speak=Mock(),
        twitch=FakeTwitch(),
    )
    return engine


class TTSControlTests(unittest.TestCase):
    def test_global_tts_off_command_disables_voice(self):
        engine = make_engine(tts_enabled=True)

        reply = engine._handle_tts_manual_command("Hebe, desactiva tu voz")

        self.assertFalse(engine.runtime.state.tts_enabled)
        self.assertEqual(reply, "Vale, Leo. Me quedo en texto.")

    def test_global_tts_on_command_enables_voice(self):
        engine = make_engine(tts_enabled=False)

        reply = engine._handle_tts_manual_command("Hebe, activa tu voz")

        self.assertTrue(engine.runtime.state.tts_enabled)
        self.assertIsNotNone(engine.runtime.state.pending_tts_scope)
        self.assertIn("solo", reply.lower())

    def test_global_tts_on_command_with_suffix_hebe_enables_voice(self):
        engine = make_engine(tts_enabled=False)

        reply = engine._handle_tts_manual_command("activa la voz hebe")

        self.assertTrue(engine.runtime.state.tts_enabled)
        self.assertIsNotNone(engine.runtime.state.pending_tts_scope)
        self.assertIn("stream", reply.lower())

    def test_tts_scope_followup_local_does_not_trigger_reminder(self):
        engine = make_engine(tts_enabled=False)
        engine._handle_tts_manual_command("activa la voz hebe")

        reply = engine._handle_pending_manual_intent("solo por ahora para poder escucharte")

        self.assertTrue(engine.runtime.state.tts_enabled)
        self.assertIsNone(engine.runtime.state.pending_tts_scope)
        self.assertFalse(engine.runtime.state.stream.policies.allow_tts_idle_prompts)
        self.assertIn("solo", reply.lower())

    def test_cancel_pending_reminder_clears_state(self):
        engine = make_engine(tts_enabled=False)
        engine.runtime.state.pending_clarification = {"kind": "appointment_datetime"}
        engine.runtime.state.pending_reminder = {"kind": "appointment_datetime"}

        reply = engine._handle_pending_manual_intent("no quiero que guardes nada")

        self.assertIsNone(engine.runtime.state.pending_clarification)
        self.assertIsNone(engine.runtime.state.pending_reminder)
        self.assertEqual(reply, "Vale, no guardo nada.")

    def test_global_tts_disabled_skips_runtime_speak_and_emits_text(self):
        engine = make_engine(tts_enabled=False)

        with patch("app.hebe_engine.emit") as emit:
            engine._deliver_voice_reply("Texto visible.")

        engine.runtime.speak.assert_not_called()
        emit.assert_called_with("chat.assistant", {"text": "Texto visible."})

    def test_stream_tts_off_command_keeps_text_chat(self):
        engine = make_engine(tts_enabled=True)

        reply = engine._handle_tts_manual_command("Hebe, responde solo por chat")

        self.assertFalse(engine.runtime.state.stream.policies.allow_tts_replies)
        self.assertEqual(reply, "Entendido. En stream responderé solo por chat.")

        engine._deliver_twitch_reply("Mensaje de Twitch.")

        self.assertEqual(engine.runtime.twitch.sent, ["Mensaje de Twitch."])
        engine.runtime.speak.assert_not_called()

    def test_stream_tts_on_command_enables_stream_voice_policy(self):
        engine = make_engine(tts_enabled=True)
        engine.runtime.state.stream.policies.allow_tts_replies = False

        reply = engine._handle_tts_manual_command("Hebe, puedes hablar en stream")

        self.assertTrue(engine.runtime.state.stream.policies.allow_tts_replies)
        self.assertEqual(reply, "Vale. Si toca, también hablaré en stream.")

    def test_global_tts_disabled_blocks_stream_tts_even_when_policy_allows(self):
        engine = make_engine(tts_enabled=False)
        engine.runtime.state.stream.policies.allow_tts_replies = True

        engine._deliver_twitch_reply("Texto al chat.")

        self.assertEqual(engine.runtime.twitch.sent, ["Texto al chat."])
        engine.runtime.speak.assert_not_called()


if __name__ == "__main__":
    unittest.main()
