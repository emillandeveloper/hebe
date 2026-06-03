import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.cognitive.input_event import InputEvent
from app.hebe_engine import HebeEngine
from app.services.voice_command_recovery import normalize_stt_transcript
from app.stream.action_planner import StreamActionPlanner
from app.stream.state import StreamSessionState


class FakeTwitch:
    def __init__(self):
        self.sent = []
        self.channel_name = "leonifelheim"
        self.bot_username = "HebeNifelheim"
        self.shoutout_command_template = "!so {username}"

    def is_available(self):
        return True

    def normalize_twitch_username(self, username):
        return str(username or "").strip().lstrip("@").replace(" ", "")

    def build_shoutout_command(self, username):
        return self.shoutout_command_template.format(username=self.normalize_twitch_username(username))

    def shoutout(self, username):
        self.sent.append(self.build_shoutout_command(username))
        return True

    def resolve_user(self, raw_target):
        return None


class FakeSynth:
    def synthesize_command_result(self, result, **kwargs):
        return f"modelo:{result.action_type}:{result.state_changes.get('target', '')}"


def make_engine(chatters=None, *, live=True):
    stream = StreamSessionState(enabled=True, presence_mode="reactive")
    stream.is_live = live
    stream.live_status_known = True
    stream.recent_chat_messages = [
        {"username": name, "display_name": name.capitalize(), "text": "hola", "ts": 1.0}
        for name in (chatters or ["nuria", "charlie", "totodile", "alguien_del_chat"])
    ]
    stream.recent_active_users = list(chatters or ["nuria", "charlie", "totodile", "alguien_del_chat"])
    engine = HebeEngine.__new__(HebeEngine)
    engine.runtime = SimpleNamespace(
        state=SimpleNamespace(stream=stream),
        twitch=FakeTwitch(),
        twitch_chat_bot=None,
        speak=Mock(),
    )
    engine.response_synthesizer = FakeSynth()
    engine.voice_command_confirm_ambiguous = True
    engine.shoutout_cooldown_seconds = 120
    engine.shoutout_allow_bots = False
    engine.shoutout_blocked_users = {"hebenifelheim", "jotunbot", "streamelements", "nightbot"}
    engine._manual_reply_ui_only = False
    engine._current_input_event = None
    engine.stream_action_planner = engine._build_stream_action_planner()
    return engine


class VoiceCommandPipelineTests(unittest.TestCase):
    def test_stt_normalization_only_does_not_execute(self):
        result = normalize_stt_transcript("ebe az promo anuria", known_targets=["nuria"])

        self.assertEqual(result.normalized_text, "hebe haz promo a nuria")
        self.assertTrue(result.metadata["normalization_only"])
        self.assertFalse(hasattr(result, "action_type"))

    def test_planner_detects_generic_shoutout_phrasings(self):
        engine = make_engine()
        planner = engine._get_stream_action_planner()
        cases = {
            "Hebe haz promo a Nuria": "Nuria",
            "Haz una promo a Charlie": "Charlie",
            "Dale SO a Totodile": "Totodile",
            "Promociona a alguien_del_chat": "alguien_del_chat",
            "Shoutout to randomUser": "randomUser",
        }
        for text, target in cases.items():
            with self.subTest(text=text):
                plan = planner.plan(InputEvent(source="typed_ui", raw_text=text, normalized_text=text))
                self.assertIsNotNone(plan)
                self.assertEqual(plan.action_type, "twitch_shoutout")
                self.assertEqual(plan.status, "complete")
                self.assertEqual(plan.target, target)

    def test_noisy_stt_enters_planner_as_normalized_candidate(self):
        engine = make_engine(["nuria"])
        normalization = normalize_stt_transcript("ebe az promo anuria", known_targets=engine._known_voice_command_targets())
        plan = engine._get_stream_action_planner().plan(
            InputEvent(source="stt_voice", raw_text=normalization.raw_text, normalized_text=normalization.normalized_text, is_voice=True)
        )

        self.assertEqual(normalization.normalized_text, "hebe haz promo a nuria")
        self.assertEqual(plan.action_type, "twitch_shoutout")
        self.assertEqual(plan.status, "complete")
        self.assertEqual(plan.target, "nuria")

    def test_action_executor_sends_shoutout_and_result_goes_to_synthesizer(self):
        engine = make_engine(["nuria"])
        engine._current_input_event = InputEvent(
            source="stt_voice",
            raw_text="ebe az promo anuria",
            normalized_text="hebe haz promo a nuria",
            is_voice=True,
        )

        result = engine._handle_stream_manual_command("hebe haz promo a nuria")
        reply = engine._synthesize_command_result(result, input_text="hebe haz promo a nuria")

        self.assertEqual(result.action_type, "twitch_shoutout")
        self.assertTrue(result.success)
        self.assertEqual(engine.runtime.twitch.sent, ["!so nuria"])
        self.assertEqual(reply, "modelo:twitch_shoutout:nuria")

    def test_missing_target_asks_for_followup_without_executing(self):
        engine = make_engine()

        result = engine._handle_stream_manual_command("Hebe haz SO")

        self.assertEqual(result.action_type, "twitch_shoutout_clarify")
        self.assertFalse(result.success)
        self.assertEqual(engine.runtime.twitch.sent, [])
        self.assertIn("A quién", result.fallback_text)

    def test_ambiguous_target_asks_for_clarification(self):
        engine = make_engine(["nuria", "muria"])

        result = engine._handle_stream_manual_command("Hebe haz promo a uria")

        self.assertEqual(result.action_type, "twitch_shoutout_clarify")
        self.assertFalse(result.success)
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_offline_context_is_visible_in_action_plan(self):
        engine = make_engine(["nuria"], live=False)
        plan = engine._get_stream_action_planner().plan(
            InputEvent(source="typed_ui", raw_text="Hebe haz promo a Nuria", normalized_text="Hebe haz promo a Nuria")
        )

        self.assertEqual(plan.action_type, "twitch_shoutout")
        self.assertFalse(plan.context_checks["is_live"])
        self.assertEqual(plan.status, "complete")

    def test_pipeline_logs_full_flow(self):
        engine = make_engine(["nuria"])
        logs = []
        emits = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emits.append((event_type, data or {}))):
            normalization = engine._normalize_stt_input("ebe az promo anuria")
            engine._current_input_event = engine._build_input_event(
                source="stt_voice",
                raw_text=normalization.raw_text,
                normalized_text=normalization.normalized_text,
                stt_metadata=normalization.as_event(),
            )
            result = engine._handle_stream_manual_command(normalization.normalized_text)
            engine._synthesize_command_result(result, input_text=normalization.normalized_text)

        joined = "\n".join(logs)
        self.assertIn("[HEBE][INPUT]", joined)
        self.assertIn("[HEBE][STT][NORMALIZED]", joined)
        self.assertIn("[HEBE][COGNITION]", joined)
        self.assertIn("[HEBE][ACTION_PLAN]", joined)
        self.assertIn("[HEBE][ACTION_EXECUTOR]", joined)
        self.assertIn("[HEBE][RESPONSE_SYNTH]", joined)
        self.assertTrue(any(event_type == "voice.command" for event_type, _ in emits))

    def test_stt_ambient_command_uses_input_event_action_plan_executor(self):
        engine = make_engine(["nuria"])
        engine.stream_ambient_stt_enabled = True
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            normalization = engine._normalize_stt_input("Hebe, desactiva STT ambiental")
            engine._current_input_event = engine._build_input_event(
                source="stt_voice",
                raw_text=normalization.raw_text,
                normalized_text=normalization.normalized_text,
                stt_metadata=normalization.as_event(),
            )
            result = engine._handle_stream_manual_command(normalization.normalized_text)
            reply = engine._synthesize_command_result(result, input_text=normalization.normalized_text)

        joined = "\n".join(logs)
        self.assertEqual(result.action_type, "stream_ambient_stt_disabled")
        self.assertFalse(engine.stream_ambient_stt_enabled)
        self.assertEqual(reply, "modelo:stream_ambient_stt_disabled:")
        self.assertIn("[HEBE][INPUT] source=stt_voice", joined)
        self.assertIn("[HEBE][COGNITION]", joined)
        self.assertIn("[HEBE][ACTION_PLAN] action_type=stream_ambient_stt_disabled", joined)
        self.assertIn("[HEBE][ACTION_EXECUTOR] executing action_type=stream_ambient_stt_disabled", joined)

    def test_stt_command_enters_cognitive_flow_and_executes_action_plan(self):
        engine = make_engine(["nuria"])
        engine.stream_ambient_stt_enabled = True
        delivered = []
        logs = []
        normalization = engine._normalize_stt_input("Hebe, desactiva STT ambiental")
        engine._current_input_event = engine._build_input_event(
            source="stt_voice",
            raw_text=normalization.raw_text,
            normalized_text=normalization.normalized_text,
            stt_metadata=normalization.as_event(),
        )
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            result = engine.cognitive_flow(normalization.normalized_text, source="stt_voice")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertFalse(engine.stream_ambient_stt_enabled)
        self.assertEqual(delivered, [("stt_voice", "modelo:stream_ambient_stt_disabled:")])
        self.assertIn("[HEBE][COG] incoming source='stt_voice'", joined)
        self.assertIn("[HEBE][ACTION_PLAN] action_type=stream_ambient_stt_disabled", joined)
        self.assertIn("[HEBE][ACTION_EXECUTOR] executing action_type=stream_ambient_stt_disabled", joined)
        self.assertIn("[HEBE][RESPONSE_SYNTH]", joined)

    def test_stt_shoutout_without_wakeword_still_has_action_intent(self):
        engine = make_engine(["nuria"])
        normalization = engine._normalize_stt_input("Haz una promo a Nuria")
        event = engine._build_input_event(
            source="stt_voice",
            raw_text=normalization.raw_text,
            normalized_text=normalization.normalized_text,
            stt_metadata=normalization.as_event(),
        )

        self.assertTrue(engine._input_event_has_action_intent(event))
        plan = engine._get_stream_action_planner().plan(event)
        self.assertEqual(plan.action_type, "twitch_shoutout")
        self.assertEqual(plan.target, "nuria")


if __name__ == "__main__":
    unittest.main()
