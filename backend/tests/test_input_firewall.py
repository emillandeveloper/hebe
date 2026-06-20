import unittest

from app.stream.input_firewall import (
    ACTION_PROMOTION_SHOUTOUT,
    ACTION_TWITCH_ACTION,
    ACTION_TWITCH_REPLY,
    InputAuthorityFirewall,
    is_known_bot_username,
    looks_like_media_or_singing,
)


class InputAuthorityFirewallTests(unittest.TestCase):
    def setUp(self):
        self.firewall = InputAuthorityFirewall(extra_bot_usernames={"hebenifelheim"})

    def test_singing_background_stt_is_ignored(self):
        decision = self.firewall.decide(
            source="ambient_stt",
            text="and im still on the road with you tonight",
            stream_is_live=False,
        )

        self.assertEqual(decision.source, "ambient_stt")
        self.assertTrue(decision.media_or_singing_detected)
        self.assertEqual(decision.firewall_decision, "ignore")
        self.assertFalse(decision.would_call_llm)
        self.assertFalse(decision.would_send_twitch)
        self.assertTrue(decision.blocks_action(ACTION_TWITCH_ACTION))
        self.assertTrue(decision.blocks_action(ACTION_PROMOTION_SHOUTOUT))

    def test_ambient_channel_like_word_cannot_promote(self):
        decision = self.firewall.decide(
            source="ambient_stt",
            text="randomchannelname",
            stream_is_live=True,
        )

        self.assertEqual(decision.firewall_decision, "allow_context_only")
        self.assertTrue(decision.blocks_action(ACTION_PROMOTION_SHOUTOUT))
        self.assertTrue(decision.blocks_action(ACTION_TWITCH_REPLY))

    def test_offline_twitch_viewer_is_blocked(self):
        decision = self.firewall.decide(
            source="twitch_viewer",
            username="viewer",
            text="Hebe do a stream action",
            stream_is_live=False,
            addressed_to_hebe=True,
        )

        self.assertEqual(decision.firewall_decision, "block_reply")
        self.assertEqual(decision.reason, "offline_stream")
        self.assertFalse(decision.would_send_twitch)
        self.assertTrue(decision.blocks_action(ACTION_TWITCH_REPLY))

    def test_known_bot_message_is_ignored(self):
        decision = self.firewall.decide(
            source="twitch_viewer",
            username="Jotunbot",
            text="automated stream helper message",
            stream_is_live=True,
            addressed_to_hebe=True,
        )

        self.assertEqual(decision.source, "twitch_bot")
        self.assertEqual(decision.authority, "bot")
        self.assertEqual(decision.input_trust, "untrusted_bot")
        self.assertEqual(decision.firewall_decision, "ignore")
        self.assertTrue(decision.bot_detected)
        self.assertFalse(decision.would_call_llm)

    def test_followup_window_rejects_media_like_stt(self):
        decision = self.firewall.decide(
            source="ambient_stt",
            text="we are running through the night again",
            stream_is_live=True,
            pending_followup=True,
        )

        self.assertEqual(decision.firewall_decision, "ignore")
        self.assertEqual(decision.reason, "media_or_singing_stt")
        self.assertTrue(decision.media_or_singing_detected)

    def test_owner_direct_stt_is_allowed_offline(self):
        decision = self.firewall.decide(
            source="owner_stt_direct",
            text="Hebe status",
            stream_is_live=False,
            addressed_to_hebe=True,
        )

        self.assertEqual(decision.firewall_decision, "allow")
        self.assertEqual(decision.authority, "owner")
        self.assertTrue(decision.allows_action("local_reply"))
        self.assertTrue(decision.blocks_action(ACTION_TWITCH_REPLY))

    def test_owner_stt_command_is_local_offline_and_twitch_remains_blocked(self):
        decision = self.firewall.decide(
            source="owner_stt_command",
            text="abre obs",
            stream_is_live=False,
            addressed_to_hebe=False,
            has_action_intent=True,
        )

        self.assertEqual(decision.firewall_decision, "allow")
        self.assertEqual(decision.authority, "owner")
        self.assertTrue(decision.allows_action("app_control"))
        self.assertTrue(decision.blocks_action(ACTION_TWITCH_REPLY))

    def test_owner_ui_local_chat_is_allowed_offline(self):
        decision = self.firewall.decide(
            source="owner_ui",
            text="local check",
            stream_is_live=False,
            addressed_to_hebe=True,
        )

        self.assertEqual(decision.firewall_decision, "allow")
        self.assertEqual(decision.input_trust, "trusted_direct")
        self.assertTrue(decision.allows_action("local_ui_message"))

    def test_viewer_message_offline_does_not_call_llm_or_twitch(self):
        decision = self.firewall.decide(
            source="twitch_viewer",
            username="viewer",
            text="Hebe can you reply",
            stream_is_live=False,
            addressed_to_hebe=True,
        )

        self.assertEqual(decision.firewall_decision, "block_reply")
        self.assertFalse(decision.would_call_llm)
        self.assertFalse(decision.would_send_twitch)

    def test_owner_promotion_intent_allows_promotion_only_when_live(self):
        decision = self.firewall.decide(
            source="owner_ui",
            text="Hebe promote target",
            stream_is_live=True,
            addressed_to_hebe=True,
            has_action_intent=True,
        )

        self.assertEqual(decision.firewall_decision, "allow")
        self.assertTrue(decision.allows_action(ACTION_PROMOTION_SHOUTOUT))

    def test_promotion_from_ambient_stt_is_blocked(self):
        decision = self.firewall.decide(
            source="ambient_stt",
            text="channelname",
            stream_is_live=True,
            has_action_intent=True,
        )

        self.assertEqual(decision.firewall_decision, "allow_context_only")
        self.assertTrue(decision.blocks_action(ACTION_PROMOTION_SHOUTOUT))

    def test_wizebot_is_known_bot(self):
        self.assertTrue(is_known_bot_username("WizeBot"))

    def test_media_classifier_is_not_exact_phrase_only(self):
        detected, reason = looks_like_media_or_singing("we will stay in the light again")

        self.assertTrue(detected)
        self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
