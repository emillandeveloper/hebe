import unittest

from app.stream.spontaneity import StreamSpontaneityConfig, StreamSpontaneityService
from app.stream.state import StreamSessionState


class StreamSpontaneityTests(unittest.TestCase):
    def make_service(self, now=1_000_000.0):
        return StreamSpontaneityService(
            config=StreamSpontaneityConfig(
                companion_silence_sec=10 * 60,
                show_silence_sec=5 * 60,
                companion_jitter_sec=0,
                show_jitter_sec=0,
                global_stream_cooldown_sec=4 * 60,
            ),
            now_fn=lambda: now,
        )

    def make_stream(self, *, now, presence_mode="show", enabled=True):
        stream = StreamSessionState(enabled=enabled, presence_mode=presence_mode)
        stream.is_live = True
        stream.live_status_known = True
        stream.stream_context_updated_ts = now
        return stream

    def test_silent_mode_never_emits(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="silent")
        stream.last_chat_activity_ts = now - 60 * 60

        event = self.make_service(now).build_due_event(stream)

        self.assertIsNone(event)

    def test_reactive_mode_does_not_emit_idle_prompts(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="reactive")
        stream.last_chat_activity_ts = now - 60 * 60

        event = self.make_service(now).build_due_event(stream)

        self.assertIsNone(event)

    def test_companion_emits_only_after_enough_silence(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="companion")
        stream.last_chat_activity_ts = now - (10 * 60 - 1)

        self.assertIsNone(self.make_service(now).build_due_event(stream))

        stream.last_chat_activity_ts = now - 10 * 60
        event = self.make_service(now).build_due_event(stream)

        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, "twitch_idle_prompt")
        self.assertEqual(event.payload["presence_mode"], "companion")

    def test_show_emits_earlier_than_companion(self):
        now = 1_000_000.0
        companion = self.make_stream(now=now, presence_mode="companion")
        companion.last_chat_activity_ts = now - 6 * 60
        show = self.make_stream(now=now, presence_mode="show")
        show.last_chat_activity_ts = now - 6 * 60

        self.assertIsNone(self.make_service(now).build_due_event(companion))
        self.assertIsNotNone(self.make_service(now).build_due_event(show))

    def test_recent_chat_activity_blocks_idle_prompt(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.last_chat_activity_ts = now - 60

        event = self.make_service(now).build_due_event(stream)

        self.assertIsNone(event)

    def test_recent_hebe_message_blocks_idle_prompt(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.last_chat_activity_ts = now - 60 * 60
        stream.last_hebe_stream_speak_ts = now - 60

        event = self.make_service(now).build_due_event(stream)

        self.assertIsNone(event)

    def test_stream_must_be_enabled(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show", enabled=False)
        stream.last_chat_activity_ts = now - 60 * 60

        event = self.make_service(now).build_due_event(stream)

        self.assertIsNone(event)

    def test_offline_stream_blocks_idle_prompt(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.is_live = False
        stream.last_chat_activity_ts = now - 60 * 60

        event = self.make_service(now).build_due_event(stream)

        self.assertIsNone(event)

    def test_unknown_or_stale_live_status_blocks_idle_prompt(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.live_status_known = False
        stream.last_chat_activity_ts = now - 60 * 60

        self.assertIsNone(self.make_service(now).build_due_event(stream))

        stream.live_status_known = True
        stream.stream_context_updated_ts = now - 10 * 60

        self.assertIsNone(self.make_service(now).build_due_event(stream))

    def test_active_chat_suppresses_idle_prompt(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.last_chat_activity_ts = now - 60 * 60
        stream.recent_chat_messages = [
            {"username": "viewer", "text": "linux ram", "ts": now - 30, "topic": "tech_pc"},
            {"username": "viewer", "text": "server", "ts": now - 20, "topic": "tech_pc"},
            {"username": "viewer", "text": "obs", "ts": now - 10, "topic": "tech_pc"},
        ]

        readiness = self.make_service(now).evaluate(stream, now=now)

        self.assertFalse(readiness["would_send"])
        self.assertEqual(readiness["blocked_reason"], "chat_active")

    def test_title_marker_becomes_stale_after_ttl(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.last_chat_activity_ts = now - 60 * 60
        stream.title_context_markers = ["Ramuh"]
        stream.title_context_updated_ts = now - 2 * 60 * 60

        event = self.make_service(now).build_due_event(stream)

        self.assertIsNotNone(event)
        self.assertIn("Ramuh", event.payload["run_context"]["title_markers_stale"])
        self.assertNotIn("Ramuh", event.payload["run_context"]["title_markers_fresh"])

    def test_completed_marker_is_not_fresh_title_context(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.last_chat_activity_ts = now - 60 * 60
        stream.title_context_markers = ["Ramuh"]
        stream.title_context_updated_ts = now
        stream.completed_run_markers = ["Ramuh"]

        event = self.make_service(now).build_due_event(stream)

        self.assertIsNotNone(event)
        self.assertIn("Ramuh", event.payload["run_context"]["title_markers_stale"])

    def test_same_idle_topic_cannot_repeat_twice(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.last_chat_activity_ts = now - 60 * 60
        stream.recent_idle_messages = [{"topic": "challenge_comment", "timestamp": now - 100, "text": "x"}]

        event = self.make_service(now).build_due_event(stream)

        self.assertIsNotNone(event)
        self.assertNotEqual(event.payload["idle_topic"], "challenge_comment")

    def test_companion_hourly_limit_is_enforced(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="companion")
        stream.last_chat_activity_ts = now - 60 * 60
        stream.recent_idle_messages = [
            {"topic": "jrpg_trope", "timestamp": now - 1200, "text": "x"},
            {"topic": "game_vibe", "timestamp": now - 600, "text": "y"},
        ]

        readiness = self.make_service(now).evaluate(stream, now=now)

        self.assertEqual(readiness["blocked_reason"], "hourly_limit")

    def test_specificity_gate_skips_without_context_anchor(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.last_chat_activity_ts = now - 60 * 60
        service = StreamSpontaneityService(
            config=StreamSpontaneityConfig(
                show_silence_sec=5 * 60,
                show_jitter_sec=0,
                require_specific_context=True,
            ),
            now_fn=lambda: now,
        )

        readiness = service.evaluate(stream, now=now)

        self.assertFalse(readiness["would_send"])
        self.assertEqual(readiness["blocked_reason"], "no_session_primer_or_run_context")

    def test_specificity_gate_blocks_current_game_without_primer_or_run_context(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.current_category = "Zwei!!: The Arges Adventure"
        stream.last_chat_activity_ts = now - 60 * 60
        service = StreamSpontaneityService(
            config=StreamSpontaneityConfig(
                show_silence_sec=5 * 60,
                show_jitter_sec=0,
                require_specific_context=True,
            ),
            now_fn=lambda: now,
        )

        event = service.build_due_event(stream)

        self.assertIsNone(event)
        self.assertEqual(stream.last_stream_spontaneity_blocked_reason, "no_session_primer_or_run_context")

    def test_specificity_gate_uses_recent_run_context_fact(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.last_chat_activity_ts = now - 60 * 60
        stream.recent_run_context_facts = [{
            "id": "ambient:healing_item_effectiveness:1",
            "kind": "healing_item_effectiveness",
            "category": "healing_item_effectiveness",
            "text": "Leo complained that a healing item barely restores enough HP.",
            "summary": "Leo complained that a healing item barely restores enough HP.",
            "confidence": 0.84,
            "expires_at": now + 60,
        }]
        stream.run_context_updated_ts = now
        service = StreamSpontaneityService(
            config=StreamSpontaneityConfig(
                show_silence_sec=5 * 60,
                show_jitter_sec=0,
                require_specific_context=True,
            ),
            now_fn=lambda: now,
        )

        event = service.build_due_event(stream)

        self.assertIsNotNone(event)
        self.assertIn("run_context", event.payload["specific_context_anchors"])
        self.assertEqual(event.payload["idle_topic"], "resource_management")
        self.assertEqual(event.payload["used_fact_id"], "ambient:healing_item_effectiveness:1")

    def test_spontaneity_skips_when_only_weak_context_exists(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.last_chat_activity_ts = now - 60 * 60
        stream.run_context_updated_ts = now
        stream.recent_run_context_facts = [{
            "id": "ambient:objective:1",
            "kind": "objective",
            "category": "objective",
            "text": "Vamos a ver.",
            "summary": "Vamos a ver.",
            "confidence": 0.4,
            "timestamp": now,
            "expires_at": now + 60,
        }]

        readiness = self.make_service(now).evaluate(stream, now=now)

        self.assertFalse(readiness["would_send"])
        self.assertEqual(readiness["blocked_reason"], "no_high_quality_anchor")

    def test_spontaneity_uses_high_quality_rng_anchor(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        stream.last_chat_activity_ts = now - 60 * 60
        stream.run_context_updated_ts = now
        stream.recent_run_context_facts = [{
            "id": "ambient:rng_dependency:1",
            "kind": "rng_dependency",
            "category": "rng_dependency",
            "text": "Leo framed the current situation as dependent on RNG or luck.",
            "summary": "Leo framed the current situation as dependent on RNG or luck.",
            "confidence": 0.86,
            "timestamp": now,
            "expires_at": now + 60,
        }]

        event = self.make_service(now).build_due_event(stream)

        self.assertIsNotNone(event)
        self.assertEqual(event.payload["idle_topic"], "challenge_comment")
        self.assertEqual(event.payload["used_fact_id"], "ambient:rng_dependency:1")

    def test_motif_cooldown_blocks_repeated_coffee(self):
        now = 1_000_000.0
        stream = self.make_stream(now=now, presence_mode="show")
        service = self.make_service(now)
        service.record_idle_message(stream, "Esto pide café antes del boss.", topic="game_vibe")

        motif = service.motif_on_cooldown(stream, "Otro comentario de café, Leo.", now=now + 60)

        self.assertEqual(motif, "cafe")

    def test_idle_tts_is_disabled_by_default(self):
        stream = StreamSessionState()

        self.assertFalse(stream.policies.allow_tts_idle_prompts)


if __name__ == "__main__":
    unittest.main()
