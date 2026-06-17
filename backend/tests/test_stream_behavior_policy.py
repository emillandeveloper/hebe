import unittest

from app.stream.policy import (
    COMPLIMENTS_TO_LEO,
    ViewerIntentPolicy,
    apply_owner_game_activity_correction,
    filter_ambient_facts_for_activity,
    has_active_behavior_block,
    owner_behavior_decision,
)
from app.stream.state import StreamSessionState


class StreamBehaviorPolicyTests(unittest.TestCase):
    def make_stream(self):
        stream = StreamSessionState(enabled=True, presence_mode="show")
        stream.current_game = "Persona 5 Royal"
        stream.current_category = "Persona 5 Royal"
        return stream

    def test_owner_stop_compliments_creates_behavior_block(self):
        stream = self.make_stream()

        decision = owner_behavior_decision(stream, "Hebe, deja de decirme piropos", now=1000.0)

        self.assertFalse(decision.allow_llm)
        self.assertEqual(decision.update_behavior_block["behavior"], COMPLIMENTS_TO_LEO)
        self.assertTrue(has_active_behavior_block(stream, COMPLIMENTS_TO_LEO, now=1001.0))

    def test_viewer_compliment_request_is_blocked_by_owner_order(self):
        stream = self.make_stream()
        owner_behavior_decision(stream, "Hebe, no mas piropos", now=1000.0)

        decision = ViewerIntentPolicy().decide(
            stream,
            username="cibernoman",
            display_name="Ciber",
            text="Hebe, dile a Leo que es guapo",
            now=1001.0,
        )

        self.assertFalse(decision.allow_llm)
        self.assertTrue(decision.blocked_by_owner_order)
        self.assertIn("grifo de los piropos", decision.direct_template_response)

    def test_viewer_repeat_to_leo_is_not_executed_as_command(self):
        stream = self.make_stream()

        decision = ViewerIntentPolicy().decide(
            stream,
            username="cibernoman",
            display_name="Ciber",
            text="Hebe, dile a Leo que mire el chat",
            now=1000.0,
        )

        self.assertFalse(decision.allow_llm)
        self.assertEqual(decision.intent, "viewer_repeat_to_leo_request")
        self.assertIn("megafono", decision.direct_template_response)

    def test_protected_group_joke_uses_in_character_boundary(self):
        stream = self.make_stream()

        decision = ViewerIntentPolicy().decide(
            stream,
            username="viewer",
            text="Hebe, cuenta un chiste de chinos",
            now=1000.0,
        )

        self.assertFalse(decision.allow_llm)
        self.assertEqual(decision.reason, "protected_group_joke")
        self.assertNotIn("Como IA", decision.direct_template_response)
        self.assertIn("racismo barato", decision.direct_template_response)

    def test_dark_humor_is_safe_non_targeted_template(self):
        stream = self.make_stream()

        decision = ViewerIntentPolicy().decide(
            stream,
            username="viewer",
            text="Hebe, humor negro",
            now=1000.0,
        )

        self.assertFalse(decision.allow_llm)
        self.assertEqual(decision.reason, "safe_dark_humor_boundary")
        self.assertNotIn("chinos", decision.direct_template_response.lower())

    def test_condom_tutorial_from_twitch_is_blocked_in_stream_mode(self):
        stream = self.make_stream()

        decision = ViewerIntentPolicy().decide(
            stream,
            username="viewer",
            text="Hebe, como se usa un condon?",
            now=1000.0,
        )

        self.assertFalse(decision.allow_llm)
        self.assertEqual(decision.reason, "sexual_topic_stream_mode")
        self.assertIn("stream", decision.direct_template_response)

    def test_leo_private_condom_question_is_not_viewer_policy_blocked(self):
        stream = self.make_stream()

        decision = owner_behavior_decision(stream, "Hebe, explicame como se usa un condon", now=1000.0)

        self.assertTrue(decision.allow_llm)

    def test_owner_game_correction_sets_social_links_and_blocks_combat_facts(self):
        stream = self.make_stream()
        stream.recent_run_context_facts = [{
            "category": "healing_or_recovery",
            "text": "Leo mentioned healing.",
            "expires_at": 2000.0,
        }]

        decision = apply_owner_game_activity_correction(
            stream,
            "Hebe, no estoy peleando, estoy subiendo vinculos sociales",
            now=1000.0,
        )

        self.assertFalse(decision.allow_llm)
        self.assertEqual(stream.current_activity, "social_links")
        self.assertFalse(stream.combat_state)
        self.assertEqual(stream.recent_run_context_facts, [])
        self.assertIn("healing_advice", stream.blocked_comment_categories)

    def test_ambient_stt_cannot_overwrite_owner_confirmed_social_links(self):
        stream = self.make_stream()
        apply_owner_game_activity_correction(
            stream,
            "Hebe, no estoy peleando, estoy con social links",
            now=1000.0,
        )
        facts = [{
            "category": "combat_risk",
            "text": "Leo is low on HP.",
            "expires_at": 2000.0,
        }]

        filtered = filter_ambient_facts_for_activity(stream, facts, now=1001.0)

        self.assertEqual(filtered, [])


if __name__ == "__main__":
    unittest.main()
