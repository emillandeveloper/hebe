import os
import tempfile
import time
import unittest
from types import SimpleNamespace

from app.cognitive.wake_name_resolver import WakeNameResolver
from app.services import db_sqlite
from app.stream.live_session import LiveSessionBrain, init_live_session_schema


class LiveSessionBrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db_path = db_sqlite.DB_PATH
        db_sqlite.DB_PATH = os.path.join(self.tmp.name, "hebe_test.sqlite3")
        db_sqlite.init_db()
        init_live_session_schema()

    def tearDown(self):
        db_sqlite.DB_PATH = self.old_db_path
        self.tmp.cleanup()

    def make_stream(self):
        return SimpleNamespace(
            active_stream_session_id=42,
            is_live=True,
            live_status_known=True,
            current_game="Persona 5 Royal",
            current_category="Persona 5 Royal",
            current_stream_title="Tercera dungeon",
            language_mode="ESP",
            spoiler_policy="no_spoilers",
            current_run_phase=None,
            current_run_location=None,
            current_run_objective=None,
            completed_run_markers=[],
        )

    def test_live_session_tracks_progress_and_retrieves_context(self):
        brain = LiveSessionBrain(self.make_stream(), session_id="test-session")

        event_id = brain.observe_leo_stt(
            "Bien, estamos ya terminando la tercera dungeon.",
            "bien estamos ya terminando la tercera dungeon",
            addressed_to_hebe=False,
            voice_event_type="progress_update",
        )
        brain.update_from_voice_relevance(
            "Bien, estamos ya terminando la tercera dungeon.",
            "progress_update",
            SimpleNamespace(category="progress_marker", confidence=0.8, facts=[]),
        )
        context = brain.retrieve_context("que recuerdas", limit_events=5, limit_summaries=2)

        self.assertGreater(event_id, 0)
        self.assertEqual(brain.state.current_phase, "Bien, estamos ya terminando la tercera dungeon.")
        self.assertTrue(any(item["event_type"] == "session_context_update" for item in context["recent_events"]))
        self.assertEqual(context["live_state"]["current_game"], "Persona 5 Royal")

    def test_correction_invalidates_last_anchor_and_sets_boss_defeated(self):
        brain = LiveSessionBrain(self.make_stream(), session_id="test-session")
        anchor_id = brain.create_spontaneity_anchor(anchor_id="boss-risk", anchor_type="combat_risk", topic="boss")
        brain.observe_hebe_utterance(
            "Cuidado con ese boss, que huele a susto.",
            output_target=["twitch_chat"],
            input_type="spontaneity",
            anchor_id=anchor_id,
            topic="boss",
        )

        self.assertTrue(brain.is_possible_reply_to_hebe("gracias hebe pero ya vencimos al boss"))
        brain.observe_leo_stt(
            "Gracias Hebe, pero ya vencimos al boss.",
            "gracias hebe pero ya vencimos al boss",
            addressed_to_hebe=False,
            voice_event_type="casual_comment",
        )

        self.assertEqual(brain.state.latest_boss_state, "defeated")
        self.assertIn("boss-risk", brain.state.invalidated_anchors)
        self.assertTrue(brain.is_anchor_consumed_or_invalidated("boss-risk"))

    def test_recent_chatters_and_topics_are_lurk_aware(self):
        brain = LiveSessionBrain(self.make_stream(), session_id="test-session")

        brain.observe_chat_message("noise", "Noise", "estoy mirando mientras como", topic="chat_topic", mention=False)

        self.assertEqual(brain.state.current_chat_topic, "chat_topic")
        self.assertEqual(brain.state.recent_chatters[0]["username"], "noise")
        self.assertTrue(brain.state.recent_chatters[0]["likely_still_around"])

    def test_wake_aliases_without_se_ve_false_positive(self):
        resolver = WakeNameResolver()
        direct = resolver.resolve(
            raw_text="Eh ve, gracias, pero ya lo hicimos",
            normalized_text="eh ve gracias pero ya lo hicimos",
            source="stt_voice",
            command_markers={"gracias", "corrige", "mira"},
        )
        false_positive = resolver.resolve(
            raw_text="se ve bien",
            normalized_text="se ve bien",
            source="stt_voice",
            command_markers={"gracias", "corrige", "mira"},
        )

        self.assertTrue(direct.addressed_to_hebe)
        self.assertEqual(direct.canonical, "hebe")
        self.assertFalse(false_positive.addressed_to_hebe)


if __name__ == "__main__":
    unittest.main()
