import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.stream.policy import classify_viewer_semantic_intent
from tests.test_voice_command_pipeline import make_engine


class StreamAntiTrollRegressionTests(unittest.TestCase):
    def test_viewer_affection_to_leo_is_observed_without_boundary(self):
        engine = make_engine(["viewer"], live=True)
        text = "Holaaaaaa Leo guapo, hermoso <3"
        self.assertEqual(engine._classify_twitch_viewer_message(text), "viewer_to_leo_affection")
        self.assertNotEqual(classify_viewer_semantic_intent(text).intent, "viewer_behavior_request")
        route = engine._pre_generation_twitch_route_decision(
            payload={"user_login": "viewer", "display_name": "Viewer", "message_text": text},
            event_type="twitch_chat_react",
            stream=engine.runtime.state.stream,
        )
        self.assertFalse(route["should_generate"])
        self.assertEqual(route["route"], "observe_only")

    def test_explicit_relay_to_leo_remains_boundary(self):
        engine = make_engine(["viewer"], live=True)
        text = "Hebe dile a Leo que es guapo"
        self.assertEqual(engine._classify_twitch_viewer_message(text), "viewer_relay_attempt")
        self.assertEqual(classify_viewer_semantic_intent(text).intent, "viewer_repeat_to_leo_request")

    def test_degrading_identity_adoption_is_suppressed(self):
        engine = make_engine(["viewer"], live=True)
        result = engine._anti_troll_frame_guard(
            "Vale, soy la becaria del chat.",
            category="viewer_talks_about_hebe",
            event_type="twitch_chat_react",
            payload={"message_text": "Hebe es la becaria"},
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["action"], "suppress")

        observed_variant = engine._anti_troll_frame_guard(
            "La becaria oficial del caos; trae café.",
            category="viewer_talks_about_hebe",
            event_type="twitch_chat_react",
            payload={"message_text": "Hebe es la becaria"},
        )
        self.assertFalse(observed_variant["passed"])

    def test_playful_self_respect_deflection_is_allowed(self):
        engine = make_engine(["viewer"], live=True)
        result = engine._anti_troll_frame_guard(
            "Becaria no; criterio externo con horario propio.",
            category="viewer_talks_about_hebe",
            event_type="twitch_chat_react",
            payload={"message_text": "Hebe es la becaria"},
        )
        self.assertTrue(result["passed"])

    def test_viewer_promo_candidate_cannot_negotiate_or_execute(self):
        engine = make_engine(["viewer"], live=True)
        payload = {"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe porque no me haces promo"}
        decision = engine._viewer_policy_decision(payload)
        self.assertEqual(decision.intent, "promo_request_from_viewer")
        self.assertFalse(decision.allow_free_llm)
        self.assertFalse(decision.execute_as_command)
        self.assertIsNone(engine.runtime.state.pending_clarification)
        guard = engine._anti_troll_frame_guard(
            "Paga la tarifa VIP y me lo pienso.",
            category="promo_request_from_viewer",
            event_type="twitch_chat_react",
            payload=payload,
        )
        self.assertFalse(guard["passed"])
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_promotion_pending_accepts_a_lache_and_resolves_login(self):
        engine = make_engine(["lache_bg12"], live=True)
        engine.runtime.twitch.remember_chat_message(
            username="lache_bg12", display_name="Lache", text="hola"
        )
        engine.runtime.state.stream.recent_active_users = ["lache_bg12"]
        engine.runtime.state.pending_clarification = engine._make_pending_task(
            id="promo-lache",
            kind="promotion_target_clarification",
            expected_reply_type="twitch_username_or_viewer_alias",
            explicit_question_asked=True,
            can_accept_no_wake_followup=True,
            ttl_seconds=60,
            max_attempts=1,
        )
        result = engine._resolve_pending_promotion_target("a Lache", "a lache", engine.runtime.state.stream)
        self.assertTrue(result.success)
        self.assertEqual(engine.runtime.twitch.sent, ["!so lache_bg12"])
        self.assertIsNone(engine.runtime.state.pending_clarification)

    def test_ambient_stt_does_not_consume_promotion_attempt(self):
        engine = make_engine(["lache_bg12"], live=True)
        engine.runtime.state.stream.enabled = True
        pending = engine._make_pending_task(
            id="promo-ambient",
            kind="promotion_target_clarification",
            expected_reply_type="twitch_username_or_viewer_alias",
            explicit_question_asked=True,
            can_accept_no_wake_followup=True,
            ttl_seconds=60,
            max_attempts=1,
        )
        engine.runtime.state.pending_clarification = pending
        with patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            engine._process_stt_voice_transcript("No, pero porque me coge la letra H")
        self.assertEqual(pending["attempts"], 0)
        self.assertIs(engine.runtime.state.pending_clarification, pending)

    def test_pending_attempts_hold_at_max_without_paused_state(self):
        engine = make_engine(["viewer"], live=True)
        pending = engine._make_pending_task(
            kind="promotion_target_clarification",
            expected_reply_type="twitch_username_or_viewer_alias",
            max_attempts=1,
        )
        engine.runtime.state.pending_clarification = pending
        engine._increment_pending_attempt(pending, reason="invalid_target")
        engine._increment_pending_attempt(pending, reason="invalid_target")
        self.assertEqual(pending["attempts"], 1)
        self.assertEqual(pending["status"], "active")

    def test_raid_renderer_uses_defined_event_context(self):
        synth = ResponseSynthesizer.__new__(ResponseSynthesizer)
        synth._build_stream_style_block = lambda: "style"
        pipeline = Mock()
        pipeline.render.return_value = SimpleNamespace(
            text="raid ack", debug_contract={}, response_source="test"
        )
        synth._universal_pipeline = lambda: pipeline
        reply = synth._generate_twitch_raid({
            "display_name": "er_tito_xarly", "user_login": "er_tito_xarly", "viewer_count": 3
        })
        self.assertEqual(reply, "raid ack")
        self.assertIn("er_tito_xarly acaba de hacer raid", pipeline.render.call_args.args[0].envelope.raw_text)

    def test_cheer_guard_repairs_harmful_spend_bait(self):
        engine = make_engine(["viewer"], live=True)
        guard = engine._cheer_anti_bait_guard("Paga y que Leo se quede ciego.")
        self.assertFalse(guard["passed"])
        self.assertIn("encourages_more_spend", guard["violations"])
        self.assertIn("unsafe_challenge_amplified", guard["violations"])


if __name__ == "__main__":
    unittest.main()
