import unittest

from app.cognitive.core_loop import PerceivedEvent, PresenceEngine, UnderstandingResult
from app.integrations.twitch.chat_bot import TwitchChatBot
from app.stream.audio_state import EffectiveStreamAudioState
from app.stream.discourse import (
    DiscourseContributionPlanner,
    DiscourseGroundingGuard,
    DiscourseParticipationBudget,
    DiscourseTopicTracker,
    OwnerDiscourseBuffer,
    StreamTurnDetector,
)
from app.stream.social_events import (
    CheerDeduplicator,
    CheerEventPolicy,
    TwitchCheerEvent,
    parse_twitch_cheer_privmsg,
)


DISCUSSION = [
    "Cada vez hay más lanzamientos digitales porque las editoras están probando qué acepta el mercado.",
    "Si desaparece el formato físico, pierdes propiedad y dependes de la plataforma.",
    "Además desaparece la reventa, así que el consumidor pierde elección y control.",
    "Creo que esa transición puede extenderse por toda la industria.",
]


def build_topic():
    buffer = OwnerDiscourseBuffer()
    topic = None
    for index, text in enumerate(DISCUSSION):
        topic = buffer.add_fragment(text, timestamp=1000 + index * 12, confidence=.9)
    return buffer, topic


class CheerIntakeTests(unittest.TestCase):
    def test_bits_tag_creates_twitch_cheer_event(self):
        event = parse_twitch_cheer_privmsg(username="ismael_3452", display_name="Ismael", message="Cheer100 genial", tags={"bits": "100", "id": "m1"}, timestamp=1000)
        self.assertIsInstance(event, TwitchCheerEvent)
        self.assertEqual(event.bits, 100)

    def test_cheer_not_classified_as_normal_no_mention_chat(self):
        chats, socials = [], []
        bot = TwitchChatBot(channel_name="test", bot_username="HebeNifelheim", oauth_token="x", message_callback=lambda *args: chats.append(args), social_event_callback=lambda *args: socials.append(args))
        bot._handle_privmsg("@bits=100;id=m1;display-name=Ismael :ismael_3452!u@h PRIVMSG #test :Cheer100 genial")
        self.assertFalse(chats)
        self.assertEqual(socials[0][0], "twitch_cheer")

    def test_cheer_bypasses_no_mention_gate(self):
        event = parse_twitch_cheer_privmsg(username="viewer", display_name="Viewer", message="Cheer10", tags={"bits": "10"}, timestamp=1)
        result = CheerEventPolicy().decide(event)
        self.assertTrue(result["bypass_no_mention"])

    def test_cheer_acknowledged_once_and_bot_fallback_deduped(self):
        event = parse_twitch_cheer_privmsg(username="viewer", display_name="Viewer", message="Cheer10", tags={"bits": "10", "id": "real"}, timestamp=100)
        fallback = TwitchCheerEvent(**{**event.to_dict(), "event_id": "fallback", "source": "bot_fallback", "twitch_message_id": ""})
        dedupe = CheerDeduplicator()
        self.assertFalse(dedupe.check_and_record(event)[0])
        self.assertTrue(dedupe.check_and_record(fallback)[0])

    def test_cheer_bypasses_recent_owner_speech_and_soft_chat_budget(self):
        event = parse_twitch_cheer_privmsg(username="viewer", display_name="Viewer", message="Cheer10", tags={"bits": "10"}, timestamp=1)
        result = CheerEventPolicy().decide(event)
        self.assertTrue(result["bypass_recent_owner_speech"])
        self.assertTrue(result["bypass_soft_chat_budget"])
        self.assertFalse(result["open_pending"])


class OwnerDiscourseTests(unittest.TestCase):
    def test_owner_monologue_fragments_added_to_discourse_buffer(self):
        buffer, topic = build_topic()
        self.assertEqual(len(buffer.current_session.fragments), 4)
        self.assertTrue(topic.stable)

    def test_fragment_does_not_immediately_trigger_reply(self):
        buffer = OwnerDiscourseBuffer()
        topic = buffer.add_fragment(DISCUSSION[0], timestamp=1)
        self.assertFalse(DiscourseContributionPlanner().plan(topic).should_contribute)

    def test_multiple_related_fragments_form_one_discourse_session(self):
        buffer, _ = build_topic()
        self.assertEqual(len(buffer.sessions), 1)

    def test_unrelated_topic_change_starts_new_topic_segment(self):
        buffer, _ = build_topic()
        buffer.add_fragment("El mapa del boss tiene una ruta y un combate muy difícil.", timestamp=1100)
        buffer.add_fragment("No encuentro la ruta del mapa ni el inventario para el jefe.", timestamp=1110)
        self.assertEqual(len(buffer.sessions), 2)

    def test_physical_format_discussion_detected_as_industry_opinion(self):
        _, topic = build_topic()
        self.assertEqual(topic.family, "industry_opinion")
        self.assertTrue(topic.non_game_discussion)

    def test_sustained_topic_not_reduced_to_last_stt_fragment(self):
        _, topic = build_topic()
        self.assertEqual(len(topic.fragments), 4)
        self.assertIn("reventa", " ".join(topic.topic_keywords))

    def test_contribution_adds_value_instead_of_paraphrasing_only(self):
        _, topic = build_topic()
        plan = DiscourseContributionPlanner().plan(topic)
        guard = DiscourseGroundingGuard().evaluate(plan, topic, candidate=DISCUSSION[-1])
        self.assertFalse(guard["passed"])
        self.assertIn("paraphrase_only", guard["violations"])


class TurnAndBudgetTests(unittest.TestCase):
    def test_discourse_waits_while_owner_speaking(self):
        turn = StreamTurnDetector(natural_pause_seconds=3)
        turn.record_owner_fragment("Sigo hablando", timestamp=10)
        self.assertFalse(turn.detect(now=11).turn_available)

    def test_natural_pause_allows_contribution(self):
        turn = StreamTurnDetector(natural_pause_seconds=3)
        turn.record_owner_fragment("Termino esta idea.", timestamp=10)
        self.assertTrue(turn.detect(now=14).turn_available)

    def test_owner_resuming_cancels_pending_contribution(self):
        turn = StreamTurnDetector(natural_pause_seconds=3)
        turn.record_owner_fragment("Termino.", timestamp=10)
        self.assertTrue(turn.detect(now=14).turn_available)
        turn.record_owner_fragment("Un momento, sigo", timestamp=14.1)
        self.assertFalse(turn.detect(now=14.2).turn_available)

    def test_one_contribution_per_topic(self):
        _, topic = build_topic()
        budget = DiscourseParticipationBudget(min_between_seconds=0)
        budget.record(topic, contribution_type="synthesis", now=2000)
        self.assertFalse(budget.allows(topic, now=2001)["allowed"])

    def test_hourly_discourse_limit_enforced(self):
        _, topic = build_topic()
        budget = DiscourseParticipationBudget(min_between_seconds=0, max_per_hour=3, max_per_topic=5)
        for index in range(3):
            clone = type(topic)(**{**{key: value for key, value in topic.__dict__.items()} }) if hasattr(topic, "__dict__") else topic
            budget.contributions.append({"topic_id": f"other{index}", "timestamp": 2000 + index, "topic_keywords": [f"unique{index}"], "contribution_type": "synthesis"})
        self.assertFalse(budget.allows(topic, now=2010)["allowed"])

    def test_direct_mention_and_cheer_not_blocked_by_discourse_budget(self):
        _, topic = build_topic()
        budget = DiscourseParticipationBudget(max_per_hour=0)
        self.assertTrue(budget.allows(topic, direct_priority=True)["allowed"])
        self.assertTrue(budget.allows(topic, event_type="cheer")["allowed"])

    def test_effective_tts_state_consistent(self):
        ready = EffectiveStreamAudioState.resolve(configured=True, engine_ready=True, route_enabled=True)
        blocked = EffectiveStreamAudioState.resolve(configured=True, engine_ready=False, route_enabled=True)
        self.assertTrue(ready.actual_can_speak)
        self.assertEqual(blocked.blocked_reason, "tts_engine_not_ready")


if __name__ == "__main__":
    unittest.main()
