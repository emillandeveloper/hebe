import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.hebe_engine import HebeEngine
from app.stream.state import StreamSessionState


class FakeTwitch:
    def __init__(self):
        self.sent = []

    def is_available(self):
        return True

    def send_message(self, text):
        self.sent.append(text)


class FakeSynth:
    def generate_stream_presence(self, **kwargs):
        return "El chat se ha quedado más quieto que un NPC sin quest."


def make_engine(stream=None):
    engine = HebeEngine.__new__(HebeEngine)
    stream = stream or StreamSessionState()
    engine.runtime = SimpleNamespace(
        state=SimpleNamespace(stream=stream),
        twitch=FakeTwitch(),
        speak=Mock(),
    )
    engine.response_synthesizer = FakeSynth()
    engine._last_presence_poll_ts = 0.0
    engine.presence_poll_interval_sec = 0.0
    engine._last_routine_poll_ts = 0.0
    engine.routine_poll_interval_sec = 30.0
    return engine


class StreamPresenceTests(unittest.TestCase):
    def test_stream_presence_default_is_reactive(self):
        stream = StreamSessionState()

        self.assertEqual(stream.presence_mode, "reactive")

    def test_presence_mode_command_sets_companion(self):
        engine = make_engine()

        reply = engine._handle_stream_manual_command("Hebe, modo compañera")

        self.assertEqual(engine.runtime.state.stream.presence_mode, "companion")
        self.assertIn("compañera", reply)

    def test_manual_no_stream_suppresses_today(self):
        engine = make_engine()

        reply = engine._handle_stream_manual_command("Hebe, hoy no hay stream")

        self.assertFalse(engine.runtime.state.stream.enabled)
        self.assertIsNotNone(engine.runtime.state.stream.no_stream_today_date)
        self.assertIn("no insisto", reply)

    def test_voice_comment_updates_context_but_is_not_command(self):
        engine = make_engine(StreamSessionState(enabled=True))

        event_type, mood = engine._classify_voice_event("me han matado otra vez")
        engine._record_voice_event("me han matado otra vez", event_type, mood)

        stream = engine.runtime.state.stream
        self.assertEqual(event_type, "gameplay_failure")
        self.assertEqual(stream.last_voice_event, "gameplay_failure")
        self.assertEqual(stream.leo_mood_hint, "frustrated")

    def test_direct_voice_command_to_hebe_is_detected(self):
        engine = make_engine(StreamSessionState(enabled=True))

        event_type, mood = engine._classify_voice_event("hebe prepara stream")

        self.assertEqual(event_type, "direct_command_to_hebe")
        self.assertIsNone(mood)

    def test_presence_speaks_when_chat_is_silent_and_mode_companion(self):
        stream = StreamSessionState(enabled=True, presence_mode="companion")
        stream.last_chat_activity_ts = time.time() - 10 * 60
        stream.last_hebe_stream_speak_ts = time.time() - 20 * 60
        stream.last_voice_event_ts = time.time() - 60
        engine = make_engine(stream)

        engine.poll_stream_presence()

        self.assertEqual(
            engine.runtime.twitch.sent,
            ["El chat se ha quedado más quieto que un NPC sin quest."],
        )

    def test_presence_skips_reactive_mode(self):
        stream = StreamSessionState(enabled=True, presence_mode="reactive")
        stream.last_chat_activity_ts = time.time() - 10 * 60
        engine = make_engine(stream)

        engine.poll_stream_presence()

        self.assertEqual(engine.runtime.twitch.sent, [])


if __name__ == "__main__":
    unittest.main()
