import unittest
from unittest.mock import patch

from app.stream.companion_loop import StreamCompanionLoop
from app.cognitive.input_interpretation import InputInterpreter
from app.stream.behavior_adaptation import BehaviorAdaptationService
from app.stream.spontaneity import StreamSpontaneityConfig, StreamSpontaneityService
from app.stream.state import StreamSessionState


class StreamCompanionLoopTests(unittest.TestCase):
    def make_stream(self, *, now=1_000_000.0):
        stream = StreamSessionState(enabled=True, presence_mode="companion")
        stream.is_live = True
        stream.live_status_known = True
        stream.stream_context_updated_ts = now
        stream.last_chat_activity_ts = now - 60 * 60
        stream.current_game = "Test RPG"
        stream.policies.allow_tts_idle_prompts = True
        return stream

    def make_loop(self, *, now=1_000_000.0):
        service = StreamSpontaneityService(
            config=StreamSpontaneityConfig(
                companion_silence_sec=60,
                companion_jitter_sec=0,
                global_stream_cooldown_sec=60,
                startup_grace_sec=0,
                companion_max_per_hour=3,
            ),
            now_fn=lambda: now,
        )
        return StreamCompanionLoop(spontaneity=service, now_fn=lambda: now)

    def test_silent_tick_still_logged(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now)
        stream.presence_mode = "reactive"
        loop = self.make_loop(now=now)
        logged = []

        with patch("app.stream.companion_loop.log_proactive_decision", lambda decision: logged.append(decision.to_dict())):
            tick = loop.evaluate(stream, stream_tts_enabled=True, output_mode="tts_enabled")

        self.assertIsNotNone(tick)
        self.assertFalse(tick.should_speak)
        self.assertEqual(logged[-1]["trigger"], "stream_companion_tick")
        self.assertFalse(logged[-1]["should_speak"])
        self.assertEqual(stream.last_proactive_decision["blocked_reason"], "presence mode is reactive")

    def test_stream_companion_tick_writes_proactive_decision(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now)
        loop = self.make_loop(now=now)
        logged = []

        with patch("app.stream.companion_loop.log_proactive_decision", lambda decision: logged.append(decision.to_dict())):
            loop.evaluate(stream, stream_tts_enabled=True, output_mode="tts_enabled")

        self.assertEqual(len(logged), 1)
        decision = logged[0]
        self.assertEqual(decision["proactive_type"], "stream_companion")
        self.assertEqual(decision["trigger"], "stream_companion_tick")
        self.assertIn("stream_state", decision)
        self.assertIn("selected_route", decision)
        self.assertIn("social_value_score", decision)
        self.assertIn("interruption_cost", decision)

    def test_ambient_failure_context_can_create_anchor(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now)
        stream.run_context_updated_ts = now
        stream.recent_run_context_facts = [{
            "id": "ambient:failure_or_death:1",
            "kind": "failure_or_death",
            "category": "failure_or_death",
            "text": "Leo mentioned death, game over, a wipe, or a failed attempt.",
            "summary": "Leo mentioned death, game over, a wipe, or a failed attempt.",
            "confidence": 0.84,
            "timestamp": now - 45,
            "expires_at": now + 600,
        }]
        stream.last_voice_event_ts = now - 45
        loop = self.make_loop(now=now)

        with patch("app.stream.companion_loop.log_proactive_decision", lambda decision: None):
            tick = loop.evaluate(stream, stream_tts_enabled=True, output_mode="tts_enabled")

        self.assertIsNotNone(tick)
        self.assertTrue(tick.should_speak)
        self.assertIsNotNone(tick.event)
        self.assertEqual(tick.decision.anchor_type, "failure_or_death")
        self.assertEqual(tick.route, "stream_tts_reply")

    def test_recent_owner_speech_is_not_a_blanket_veto_after_turn_gap(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now)
        stream.last_voice_event_ts = now - 10
        loop = self.make_loop(now=now)

        with patch("app.stream.companion_loop.log_proactive_decision", lambda decision: None):
            tick = loop.evaluate(stream, stream_tts_enabled=True, output_mode="tts_enabled")

        self.assertIsNotNone(tick)
        self.assertTrue(tick.should_speak)
        self.assertNotEqual(tick.blocked_reason, "recent_owner_speech")

    def test_recent_owner_speech_expires_after_window(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now)
        stream.last_voice_event_ts = now - 31
        loop = self.make_loop(now=now)

        with patch("app.stream.companion_loop.log_proactive_decision", lambda decision: None):
            tick = loop.evaluate(stream, stream_tts_enabled=True, output_mode="tts_enabled")

        self.assertIsNotNone(tick)
        self.assertTrue(tick.should_speak)

    def test_emitted_tick_logged_with_final_response(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now)
        loop = self.make_loop(now=now)

        with patch("app.stream.companion_loop.log_proactive_decision", lambda decision: None):
            loop.evaluate(stream, stream_tts_enabled=True, output_mode="tts_enabled")
        loop.record_emitted(stream, "short final line", route="stream_tts_reply")

        self.assertEqual(stream.last_proactive_decision["final_response"], "short final line")
        self.assertEqual(stream.last_proactive_decision["selected_route"], "stream_tts_reply")
        self.assertEqual(loop.emitted_count, 1)

    def test_proactive_hot_path_consults_behavior_adaptation_before_event(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now)
        stream.run_context_updated_ts = now
        stream.recent_idle_messages = [{
            "text": "Otra reacción sobre failure death.",
            "topic": "failure_or_death",
            "timestamp": now - 90,
        }]
        stream.recent_run_context_facts = [{
            "id": "ambient:failure_or_death:feedback",
            "kind": "failure_or_death",
            "category": "failure_or_death",
            "text": "Leo mentioned a failure or death.",
            "confidence": 0.9,
            "timestamp": now - 30,
            "expires_at": now + 600,
        }]
        interpretation = InputInterpreter().interpret(
            raw_text="Otra vez con lo de failure death, ya cansa.",
            source="stt_voice",
            authority="owner",
            addressed_to_hebe=True,
        )
        adaptation = BehaviorAdaptationService()
        adaptation.apply_feedback(stream, interpretation, now=now)
        loop = self.make_loop(now=now)
        loop.behavior_adaptation = adaptation

        with patch("app.stream.companion_loop.log_proactive_decision", lambda decision: None):
            tick = loop.evaluate(stream, stream_tts_enabled=True, output_mode="tts_enabled")

        self.assertIsNotNone(tick)
        self.assertFalse(tick.should_speak)
        self.assertEqual(tick.blocked_reason, "behavior_policy_no_candidate")
        ranked = tick.readiness["behavior_ranked_candidates"]
        self.assertEqual(ranked[0]["policy"]["action"], "suppress")
        self.assertFalse(ranked[0]["eligible"])

    def test_companion_health_summary_logged(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now)
        loop = self.make_loop(now=now)
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            loop.maybe_log_health(stream, now=now, force=True)

        self.assertIn("[HEBE][STREAM_COMPANION_HEALTH]", "\n".join(logs))


if __name__ == "__main__":
    unittest.main()
