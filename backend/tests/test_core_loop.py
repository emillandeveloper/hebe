import unittest

from app.cognitive.core_loop import (
    HebeCoreLoop,
    PerceivedEvent,
    PolicyContract,
    PresenceEngine,
    UnderstandingResult,
)


class CoreLoopTests(unittest.TestCase):
    def test_pure_emote_observe_only(self):
        decision = PresenceEngine().decide(
            perception=PerceivedEvent(
                event_id="evt-1",
                source_type="twitch_chat",
                speaker_type="viewer",
                raw_text="Kappa",
                is_emote_only=True,
                is_low_value_chat=True,
            ),
            understanding=UnderstandingResult(
                intent="viewer_emote_only",
                confidence=0.9,
                authority="viewer",
                reply_pressure=0.02,
                social_context="meme_or_emote",
            ),
            policy=PolicyContract(result="allow"),
        )

        self.assertFalse(decision.should_intervene)
        self.assertEqual(decision.intervention_level, "observe_only")

    def test_talks_about_hebe_can_trigger_self_banter(self):
        decision = PresenceEngine().decide(
            perception=PerceivedEvent(
                event_id="evt-2",
                source_type="twitch_chat",
                speaker_type="viewer",
                raw_text="Hebe is present in the scene",
                mentions_hebe=True,
                talks_about_hebe=True,
            ),
            understanding=UnderstandingResult(
                intent="viewer_talks_about_hebe",
                confidence=0.86,
                authority="viewer",
                reply_pressure=0.56,
                social_context="viewer_talks_about_hebe",
            ),
            policy=PolicyContract(result="allow"),
            budget_result={"allowed": True, "reason": "allowed"},
        )

        self.assertTrue(decision.should_intervene)
        self.assertEqual(decision.intervention_level, "twitch_text_reply")
        self.assertEqual(decision.speech_act_type, "self_banter_reply")

    def test_social_budget_blocks_noncritical_thread(self):
        decision = PresenceEngine().decide(
            perception=PerceivedEvent(
                event_id="evt-3",
                source_type="twitch_chat",
                speaker_type="viewer",
                mentions_hebe=True,
                direct_address_to_hebe=True,
            ),
            understanding=UnderstandingResult(
                intent="viewer_direct_question_to_hebe",
                confidence=0.9,
                authority="viewer",
                reply_pressure=0.64,
            ),
            policy=PolicyContract(result="allow"),
            budget_result={"allowed": False, "reason": "thread_closed"},
        )

        self.assertFalse(decision.should_intervene)
        self.assertEqual(decision.reason, "thread_closed")
        self.assertEqual(decision.intervention_level, "observe_only")

    def test_boundary_bypasses_low_social_value(self):
        decision = PresenceEngine().decide(
            perception=PerceivedEvent(
                event_id="evt-4",
                source_type="twitch_chat",
                speaker_type="viewer",
                is_low_value_chat=True,
            ),
            understanding=UnderstandingResult(
                intent="viewer_proxy_request",
                confidence=0.9,
                authority="viewer",
                reply_pressure=0.1,
                risk_flags=["viewer_authority"],
            ),
            policy=PolicyContract(
                result="redirect",
                reason="viewer_proxy_request",
                boundary_required=True,
                blocked_behavior="message_to_leo",
                risk_level="medium",
            ),
            budget_result={"allowed": True, "reason": "allowed"},
        )

        self.assertTrue(decision.should_intervene)
        self.assertEqual(decision.speech_act_type, "viewer_boundary")

    def test_core_loop_returns_full_chain(self):
        result = HebeCoreLoop().process(
            perception=PerceivedEvent(event_id="evt-5", source_type="twitch_chat", speaker_type="viewer"),
            understanding=UnderstandingResult(intent="viewer_low_value_banter", confidence=0.8, authority="viewer"),
            policy=PolicyContract(result="allow"),
        )

        self.assertIn("perception", result)
        self.assertIn("understanding", result)
        self.assertIn("policy", result)
        self.assertIn("intervention", result)


if __name__ == "__main__":
    unittest.main()
