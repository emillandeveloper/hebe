import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.cognitive.command_result import CommandResult
from app.hebe_engine import HebeEngine
from app.stream.state import StreamSessionState
from tests.test_voice_command_pipeline import install_test_continuity, open_test_conversation


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
        ),
        speak=Mock(),
        twitch=FakeTwitch(),
    )
    capabilities = {"audio.tts_control", "pending.cancel", "stream.local_state_control", "twitch_action"}
    engine._active_cognitive_decision = SimpleNamespace(
        authority="owner", source="ui", should_stop_pipeline=False,
        allowed_step_types=["state_update", "action", "reply"],
        action_permission_summary={"stream_live": True},
        allows_capability=lambda capability: capability in capabilities,
    )
    install_test_continuity(engine)
    return engine


class TTSControlTests(unittest.TestCase):
    def test_global_tts_off_command_disables_voice(self):
        engine = make_engine(tts_enabled=True)

        reply = engine._handle_tts_manual_command("Hebe, desactiva tu voz")

        self.assertFalse(engine.runtime.state.tts_enabled)
        self.assertIsInstance(reply, CommandResult)
        self.assertEqual(reply.action_type, "tts_disabled")
        self.assertEqual(reply.fallback_text, "Vale, Leo. Me quedo en texto.")

    def test_global_tts_on_command_enables_voice(self):
        engine = make_engine(tts_enabled=False)

        reply = engine._handle_tts_manual_command("Hebe, activa tu voz")

        self.assertTrue(engine.runtime.state.tts_enabled)
        self.assertEqual(engine._active_current_conversation().topic, "tts_scope")
        self.assertIsInstance(reply, CommandResult)
        self.assertIn("solo", reply.lower())

    def test_global_tts_on_command_with_suffix_hebe_enables_voice(self):
        engine = make_engine(tts_enabled=False)

        reply = engine._handle_tts_manual_command("activa la voz hebe")

        self.assertTrue(engine.runtime.state.tts_enabled)
        self.assertEqual(engine._active_current_conversation().topic, "tts_scope")
        self.assertIsInstance(reply, CommandResult)
        self.assertIn("stream", reply.lower())

    def test_tts_scope_followup_local_does_not_trigger_reminder(self):
        engine = make_engine(tts_enabled=False)
        engine._handle_tts_manual_command("activa la voz hebe")

        reply = engine._handle_pending_manual_intent("solo por ahora para poder escucharte")

        self.assertTrue(engine.runtime.state.tts_enabled)
        self.assertIsNone(engine._active_current_conversation())
        self.assertFalse(engine.runtime.state.stream.policies.allow_tts_idle_prompts)
        self.assertIsInstance(reply, CommandResult)
        self.assertIn("solo", reply.lower())

    def test_tts_scope_followup_short_local_resolves(self):
        engine = make_engine(tts_enabled=False)
        engine._handle_tts_manual_command("hebe activa la voz")

        reply = engine._handle_pending_manual_intent("local")

        self.assertTrue(engine.runtime.state.tts_enabled)
        self.assertIsNone(engine._active_current_conversation())
        self.assertFalse(engine.runtime.state.stream.policies.allow_tts_idle_prompts)
        self.assertIsInstance(reply, CommandResult)
        self.assertIn("stream", reply.lower())

    def test_tts_scope_followup_aqui_resolves_local(self):
        engine = make_engine(tts_enabled=False)
        engine._handle_tts_manual_command("activa la voz")

        reply = engine._handle_pending_manual_intent("aquí")

        self.assertIsNone(engine._active_current_conversation())
        self.assertIsInstance(reply, CommandResult)
        self.assertIn("stream", reply.lower())

    def test_tts_scope_followup_stream_resolves_stream(self):
        engine = make_engine(tts_enabled=False)
        engine.runtime.state.stream.policies.allow_tts_replies = False
        engine._handle_tts_manual_command("activa la voz")

        reply = engine._handle_pending_manual_intent("stream")

        self.assertIsNone(engine._active_current_conversation())
        self.assertTrue(engine.runtime.state.stream.policies.allow_tts_replies)
        self.assertTrue(engine.runtime.state.stream.policies.allow_tts_event_replies)
        self.assertTrue(engine.runtime.state.stream.policies.allow_tts_raid_thanks)
        self.assertIsInstance(reply, CommandResult)
        self.assertIn("stream", reply.lower())

    def test_tts_scope_followup_tambien_en_directo_resolves_stream(self):
        engine = make_engine(tts_enabled=False)
        engine.runtime.state.stream.policies.allow_tts_replies = False
        engine._handle_tts_manual_command("activa la voz")

        reply = engine._handle_pending_manual_intent("también en directo")

        self.assertIsNone(engine._active_current_conversation())
        self.assertTrue(engine.runtime.state.stream.policies.allow_tts_replies)
        self.assertIsInstance(reply, CommandResult)
        self.assertIn("eventos", reply.lower())

    def test_pending_tts_scope_does_not_hijack_new_explicit_text_command(self):
        engine = make_engine(tts_enabled=False)
        engine._handle_tts_manual_command("activa la voz")

        reply = engine._handle_pending_manual_intent("solo texto")

        self.assertIsNone(reply)
        self.assertIsNone(engine._active_current_conversation())

    def test_pending_tts_scope_does_not_hijack_new_explicit_stt_command(self):
        engine = make_engine(tts_enabled=False)
        engine.stream_ambient_stt_enabled = True
        engine._handle_tts_manual_command("activa la voz")

        pending_reply = engine._handle_pending_manual_intent("Hebe, desactiva STT ambiental")
        action = engine._handle_stream_manual_command("Hebe, desactiva STT ambiental")

        self.assertIsNone(pending_reply)
        self.assertIsNone(engine._active_current_conversation())
        self.assertIsInstance(action, CommandResult)
        self.assertEqual(action.action_type, "stream_ambient_stt_disabled")
        self.assertFalse(engine.stream_ambient_stt_enabled)

    def test_pending_tts_scope_unclear_asks_once_then_defaults_local(self):
        engine = make_engine(tts_enabled=False)
        engine._handle_tts_manual_command("activa la voz")

        first = engine._handle_pending_manual_intent("patata")
        second = engine._handle_pending_manual_intent("patata otra vez")

        self.assertIn("local", first.lower())
        self.assertIsNone(engine._active_current_conversation())
        self.assertIsInstance(second, CommandResult)
        self.assertEqual(second.action_type, "tts_scope_resolved")
        self.assertEqual(second.metadata["scope"], "local")

    def test_synthesizer_receives_command_result(self):
        engine = make_engine(tts_enabled=False)
        synth = Mock()
        synth.synthesize_command_result.return_value = "Hecho con estilo."
        engine.response_synthesizer = synth

        result = engine._handle_pending_manual_intent("no guardes nada")
        self.assertIsNone(result)
        command_result = engine._handle_tts_manual_command("activa la voz")
        text = engine._synthesize_command_result(command_result, input_text="activa la voz")

        self.assertEqual(text, "Hecho con estilo.")
        synth.synthesize_command_result.assert_called_once()
        self.assertIs(synth.synthesize_command_result.call_args.args[0], command_result)

    def test_synthesizer_failure_uses_fallback(self):
        engine = make_engine(tts_enabled=False)
        synth = Mock()
        synth.synthesize_command_result.side_effect = RuntimeError("boom")
        engine.response_synthesizer = synth
        command_result = engine._handle_tts_manual_command("desactiva la voz")

        text = engine._synthesize_command_result(command_result, input_text="desactiva la voz")

        self.assertEqual(text, command_result.fallback_text)

    def test_cancel_pending_reminder_clears_state(self):
        engine = make_engine(tts_enabled=False)
        open_test_conversation(engine, kind="appointment_datetime")

        reply = engine._handle_pending_manual_intent("no quiero que guardes nada")

        self.assertIsNone(engine._active_current_conversation())
        self.assertIsInstance(reply, CommandResult)
        self.assertEqual(reply.action_type, "pending_reminder_cancelled")
        self.assertEqual(reply.fallback_text, "Vale, no guardo nada.")

    def test_global_tts_disabled_skips_runtime_speak_and_emits_text(self):
        engine = make_engine(tts_enabled=False)

        with patch("app.hebe_engine.emit") as emit:
            engine._deliver_voice_reply("Texto visible.")

        engine.runtime.speak.assert_not_called()
        emit.assert_called_with(
            "chat.assistant",
            {"text": "Texto visible.", "source": "direct_stt", "output_target": "local_ui"},
        )

    def test_stream_tts_off_command_keeps_text_chat(self):
        engine = make_engine(tts_enabled=True)
        engine.runtime.state.stream.is_live = True

        reply = engine._handle_tts_manual_command("Hebe, responde solo por chat")

        self.assertFalse(engine.runtime.state.stream.policies.allow_tts_replies)
        self.assertIsInstance(reply, CommandResult)
        self.assertEqual(reply.action_type, "stream_tts_disabled")
        self.assertEqual(reply.fallback_text, "Entendido. En stream responderé solo por chat.")

        engine._deliver_twitch_reply("Mensaje de Twitch.")

        self.assertEqual(engine.runtime.twitch.sent, ["Mensaje de Twitch."])
        engine.runtime.speak.assert_not_called()

    def test_stream_tts_on_command_enables_stream_voice_policy(self):
        engine = make_engine(tts_enabled=True)
        engine.runtime.state.stream.policies.allow_tts_replies = False

        reply = engine._handle_tts_manual_command("Hebe, puedes hablar en stream")

        self.assertTrue(engine.runtime.state.stream.policies.allow_tts_replies)
        self.assertIsInstance(reply, CommandResult)
        self.assertEqual(reply.action_type, "stream_tts_enabled")
        self.assertEqual(reply.fallback_text, "Vale. Si toca, también hablaré en stream.")

    def test_global_tts_disabled_blocks_stream_tts_even_when_policy_allows(self):
        engine = make_engine(tts_enabled=False)
        engine.runtime.state.stream.is_live = True
        engine.runtime.state.stream.policies.allow_tts_replies = True

        engine._deliver_twitch_reply("Texto al chat.")

        self.assertEqual(engine.runtime.twitch.sent, ["Texto al chat."])
        engine.runtime.speak.assert_not_called()


if __name__ == "__main__":
    unittest.main()
