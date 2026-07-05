import unittest

from app.stream.policy import (
    COMPLIMENTS_TO_LEO,
    ENTERTAINMENT_REQUEST,
    ViewerIntentPolicy,
    apply_owner_game_activity_correction,
    classify_viewer_semantic_intent,
    filter_ambient_facts_for_activity,
    has_active_behavior_block,
    owner_behavior_decision,
    policy_trace,
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
        self.assertFalse(decision.allow_free_llm)
        self.assertEqual(decision.intent, "owner_stop_behavior")
        self.assertEqual(decision.requested_behavior, COMPLIMENTS_TO_LEO)
        self.assertEqual(decision.behavior_family, COMPLIMENTS_TO_LEO)
        self.assertEqual(decision.target, "Leo")
        self.assertEqual(decision.matched_by, ["semantic_classifier"])
        self.assertTrue(decision.execute_as_command)
        self.assertEqual(decision.direct_template_response, "")
        self.assertTrue(decision.response_directive)
        self.assertTrue(decision.response_constraints)
        self.assertEqual(decision.update_behavior_block["behavior"], COMPLIMENTS_TO_LEO)
        self.assertTrue(has_active_behavior_block(stream, COMPLIMENTS_TO_LEO, now=1001.0))

        trace = policy_trace(
            source="ui",
            speaker="Leo",
            text="Hebe, deja de decirme piropos",
            decision=decision,
            authority="owner",
        )
        self.assertEqual(trace["authority"], "owner")
        self.assertEqual(trace["intent"], "owner_stop_behavior")
        self.assertEqual(trace["requested_behavior"], COMPLIMENTS_TO_LEO)
        self.assertEqual(trace["behavior_family"], COMPLIMENTS_TO_LEO)
        self.assertEqual(trace["target"], "Leo")
        self.assertEqual(trace["matched_by"], ["semantic_classifier"])
        self.assertFalse(trace["allow_free_llm"])
        self.assertTrue(trace["execute_as_command"])
        self.assertEqual(trace["policy_decision"], "allowed")
        self.assertEqual(trace["response_mode"], "llm")

    def test_semantic_owner_stop_mode_creates_behavior_block(self):
        stream = self.make_stream()

        decision = owner_behavior_decision(stream, "Hebe, cancela el modo baboso", now=1000.0)

        self.assertFalse(decision.allow_llm)
        self.assertEqual(decision.intent, "owner_stop_behavior")
        self.assertEqual(decision.requested_behavior, COMPLIMENTS_TO_LEO)
        self.assertTrue(has_active_behavior_block(stream, COMPLIMENTS_TO_LEO, now=1001.0))

    def test_semantic_owner_stop_halogos_creates_behavior_block(self):
        stream = self.make_stream()

        decision = owner_behavior_decision(stream, "Hebe, corta el festival de halagos", now=1000.0)

        self.assertFalse(decision.allow_llm)
        self.assertEqual(decision.intent, "owner_stop_behavior")
        self.assertEqual(decision.requested_behavior, COMPLIMENTS_TO_LEO)
        self.assertTrue(has_active_behavior_block(stream, COMPLIMENTS_TO_LEO, now=1001.0))

    def test_viewer_compliment_request_is_blocked_by_owner_order(self):
        stream = self.make_stream()
        owner_behavior_decision(stream, "Hebe, no quiero mas halagos hacia mi", now=1000.0)

        decision = ViewerIntentPolicy().decide(
            stream,
            username="cibernoman",
            display_name="Ciber",
            text="Hebe, mandale un halago a Leo de mi parte",
            now=1001.0,
        )

        self.assertFalse(decision.allow_llm)
        self.assertFalse(decision.allow_free_llm)
        self.assertTrue(decision.blocked_by_owner_order)
        self.assertEqual(decision.intent, "viewer_repeat_to_leo_request")
        self.assertEqual(decision.requested_behavior, COMPLIMENTS_TO_LEO)
        self.assertEqual(decision.behavior_family, COMPLIMENTS_TO_LEO)
        self.assertEqual(decision.target, "Leo")
        self.assertEqual(decision.matched_by, ["semantic_classifier"])
        self.assertFalse(decision.execute_as_command)
        self.assertEqual(decision.direct_template_response, "")
        self.assertTrue(decision.response_directive)
        self.assertTrue(decision.response_constraints)

        trace = policy_trace(
            source="twitch_chat",
            speaker="Ciber",
            text="Hebe, mandale un halago a Leo de mi parte",
            decision=decision,
            authority="viewer",
        )
        self.assertEqual(trace["authority"], "viewer")
        self.assertEqual(trace["requested_behavior"], COMPLIMENTS_TO_LEO)
        self.assertEqual(trace["policy_decision"], "blocked")
        self.assertEqual(trace["reason"], "owner_behavior_block")
        self.assertFalse(trace["allow_free_llm"])
        self.assertFalse(trace["execute_as_command"])

    def test_viewer_direct_compliment_request_is_not_command(self):
        stream = self.make_stream()

        decision = ViewerIntentPolicy().decide(
            stream,
            username="viewer",
            text="Hebe, dile a Leo que es guapo",
            now=1000.0,
        )

        self.assertEqual(decision.intent, "viewer_repeat_to_leo_request")
        self.assertEqual(decision.requested_behavior, COMPLIMENTS_TO_LEO)
        self.assertFalse(decision.allow_free_llm)
        self.assertFalse(decision.execute_as_command)

    def test_viewer_semantic_compliment_request_is_not_question(self):
        stream = self.make_stream()

        decision = ViewerIntentPolicy().decide(
            stream,
            username="viewer",
            text="Hebe, mandale un halago a Leo de mi parte",
            now=1000.0,
        )

        self.assertEqual(decision.intent, "viewer_repeat_to_leo_request")
        self.assertEqual(decision.requested_behavior, COMPLIMENTS_TO_LEO)
        self.assertNotEqual(decision.intent, "viewer_question")
        self.assertFalse(decision.execute_as_command)

    def test_viewer_affection_semantics_maps_to_compliments(self):
        stream = self.make_stream()

        decision = ViewerIntentPolicy().decide(
            stream,
            username="viewer",
            text="Hebe, mandale amor del bueno",
            now=1000.0,
        )

        self.assertEqual(decision.intent, "viewer_repeat_to_leo_request")
        self.assertEqual(decision.requested_behavior, COMPLIMENTS_TO_LEO)
        self.assertNotEqual(decision.requested_behavior, "unknown")
        self.assertFalse(decision.allow_free_llm)
        self.assertFalse(decision.execute_as_command)

    def test_behavior_block_applies_to_semantic_viewer_variant(self):
        stream = self.make_stream()
        owner_behavior_decision(stream, "Hebe, cancela el modo baboso", now=1000.0)

        decision = ViewerIntentPolicy().decide(
            stream,
            username="viewer",
            text="Hebe, mandale un halago a Leo",
            now=1001.0,
        )

        self.assertEqual(decision.intent, "viewer_repeat_to_leo_request")
        self.assertEqual(decision.reason, "owner_behavior_block")
        self.assertEqual(decision.requested_behavior, COMPLIMENTS_TO_LEO)
        self.assertFalse(decision.allow_free_llm)
        self.assertIn("actual_blocked_compliment", decision.must_not_include)
        trace = policy_trace(
            source="twitch_chat",
            speaker="viewer",
            text="Hebe, mandale un halago a Leo",
            decision=decision,
            authority="viewer",
        )
        self.assertEqual(trace["policy_decision"], "blocked")

    def test_viewer_repeat_to_leo_is_not_executed_as_command(self):
        stream = self.make_stream()

        decision = ViewerIntentPolicy().decide(
            stream,
            username="cibernoman",
            display_name="Ciber",
            text="Hebe, avisa a Leo de que lea el mensaje del chat",
            now=1000.0,
        )

        self.assertFalse(decision.allow_llm)
        self.assertFalse(decision.allow_free_llm)
        self.assertEqual(decision.intent, "viewer_repeat_to_leo_request")
        self.assertEqual(decision.requested_behavior, "message_to_leo")
        self.assertFalse(decision.execute_as_command)
        self.assertEqual(decision.direct_template_response, "")
        self.assertTrue(decision.response_directive)
        self.assertTrue(decision.response_constraints)

    def test_hebe_cuenta_un_chiste_not_message_to_leo(self):
        semantic = classify_viewer_semantic_intent("hebe cuenta un chiste")

        self.assertEqual(semantic.intent, "viewer_entertainment_request_to_hebe")
        self.assertEqual(semantic.requested_behavior, ENTERTAINMENT_REQUEST)
        self.assertEqual(semantic.target, "Hebe")

    def test_hebe_cuenta_un_ciste_typo_not_message_to_leo(self):
        semantic = classify_viewer_semantic_intent("hebe cuenta un ciste")

        self.assertEqual(semantic.intent, "viewer_entertainment_request_to_hebe")
        self.assertNotEqual(semantic.requested_behavior, "message_to_leo")

    def test_hebe_pasa_del_chat_not_message_to_leo(self):
        semantic = classify_viewer_semantic_intent("hebe pasa del chat")

        self.assertEqual(semantic.intent, "viewer_entertainment_request_to_hebe")
        self.assertEqual(semantic.target, "Hebe")

    def test_hebe_di_algo_not_message_to_leo(self):
        semantic = classify_viewer_semantic_intent("hebe di algo")

        self.assertEqual(semantic.intent, "viewer_entertainment_request_to_hebe")
        self.assertEqual(semantic.target, "Hebe")

    def test_true_leo_proxy_requests_still_message_to_leo(self):
        cases = [
            "Hebe, dile a Leo que mire el chat",
            "Hebe, avisa a Leo de que lea esto",
            "Hebe, cuentale a Leo lo que dije",
            "Hebe, cuenta le a Leo lo que dije",
        ]

        for text in cases:
            with self.subTest(text=text):
                semantic = classify_viewer_semantic_intent(text)
                self.assertEqual(semantic.intent, "viewer_repeat_to_leo_request")
                self.assertEqual(semantic.requested_behavior, "message_to_leo")

    def test_entertainment_request_can_reach_presence_engine(self):
        stream = self.make_stream()

        decision = ViewerIntentPolicy().decide(
            stream,
            username="viewer",
            display_name="Viewer",
            text="hebe cuenta un chiste",
            now=1000.0,
        )

        self.assertTrue(decision.allow_llm)
        self.assertTrue(decision.allow_reply)
        self.assertEqual(decision.intent, "viewer_entertainment_request_to_hebe")
        self.assertEqual(decision.requested_behavior, ENTERTAINMENT_REQUEST)
        self.assertFalse(decision.execute_as_command)

    def test_protected_group_joke_uses_in_character_boundary(self):
        stream = self.make_stream()

        decision = ViewerIntentPolicy().decide(
            stream,
            username="viewer",
            text="Hebe, haz un chiste sobre gitanos",
            now=1000.0,
        )

        self.assertFalse(decision.allow_llm)
        self.assertEqual(decision.reason, "protected_group_joke")
        self.assertFalse(decision.allow_free_llm)
        self.assertFalse(decision.execute_as_command)
        self.assertEqual(decision.direct_template_response, "")
        self.assertTrue(decision.response_directive)
        self.assertTrue(decision.response_constraints)

    def test_dark_humor_is_allowed_for_normal_generation(self):
        stream = self.make_stream()

        decision = ViewerIntentPolicy().decide(
            stream,
            username="viewer",
            text="Hebe, dame humor oscuro sin atacar a nadie",
            now=1000.0,
        )

        self.assertTrue(decision.allow_llm)
        self.assertTrue(decision.allow_free_llm)
        self.assertEqual(decision.reason, "safe_dark_humor_allowed")
        self.assertEqual(decision.direct_template_response, "")

    def test_explicit_sexual_tutorial_from_twitch_is_blocked_in_stream_mode(self):
        stream = self.make_stream()

        decision = ViewerIntentPolicy().decide(
            stream,
            username="viewer",
            text="Hebe, explica educacion sexual explicita aqui",
            now=1000.0,
        )

        self.assertFalse(decision.allow_llm)
        self.assertFalse(decision.allow_free_llm)
        self.assertEqual(decision.reason, "sexual_topic_stream_mode")
        self.assertEqual(decision.direct_template_response, "")
        self.assertTrue(decision.response_directive)
        self.assertTrue(decision.response_constraints)

    def test_policy_decisions_do_not_embed_final_dialogue_templates(self):
        stream = self.make_stream()
        decisions = [
            owner_behavior_decision(stream, "Hebe, cancela el modo baboso", now=1000.0),
            ViewerIntentPolicy().decide(stream, username="viewer", text="Hebe, mandale amor del bueno", now=1001.0),
            ViewerIntentPolicy().decide(stream, username="viewer", text="Hebe, dile a Leo que mire el chat", now=1002.0),
        ]

        for decision in decisions:
            self.assertEqual(decision.direct_template_response, "")

    def test_leo_private_sexual_question_is_not_viewer_policy_blocked(self):
        stream = self.make_stream()

        decision = owner_behavior_decision(stream, "Hebe, necesito una explicacion privada de educacion sexual", now=1000.0)

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
            "Hebe, no es combate, estamos en un vinculo social",
            now=1000.0,
        )

        self.assertFalse(decision.allow_llm)
        self.assertEqual(stream.current_activity, "social_links")
        self.assertFalse(stream.combat_state)
        self.assertEqual(stream.recent_run_context_facts, [])
        self.assertIn("healing_advice", stream.blocked_comment_categories)

    def test_stream_output_mode_defaults_to_tts_enabled(self):
        stream = self.make_stream()

        self.assertEqual(stream.stream_output_mode, "tts_enabled")

    def test_ambient_stt_cannot_overwrite_owner_confirmed_social_links(self):
        stream = self.make_stream()
        apply_owner_game_activity_correction(
            stream,
            "Hebe, fuera de combate, estoy con social links",
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
