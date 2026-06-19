import os
import tempfile
import unittest
from contextlib import closing

from app.services import db_sqlite
from app.stream import memory as stream_memory
from app.stream.live_session import init_live_session_schema
from app.stream.state import StreamSessionState


class StreamDataIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db_path = db_sqlite.DB_PATH
        db_sqlite.DB_PATH = os.path.join(self.tmp.name, "hebe_stream_test.sqlite3")
        stream_memory.init_stream_memory_schema()

    def tearDown(self):
        db_sqlite.DB_PATH = self.old_db_path
        self.tmp.cleanup()

    def _conn(self):
        return db_sqlite.get_db_connection()

    def _live_stream(self):
        stream = StreamSessionState(enabled=True)
        stream.is_live = True
        stream.live_status_known = True
        stream.current_stream_title = "First Playthrough Night"
        stream.current_category = "Final Fantasy IX"
        stream.current_game = "Final Fantasy IX"
        stream.stream_started_at = "2026-06-18T18:00:00Z"
        stream.language_mode = "bilingual"
        stream.spoiler_policy = "no_spoilers"
        stream.twitch_stream_id = "tw-123"
        return stream

    def test_real_stream_lifecycle_links_chat_dedupes_events_and_summarizes_metadata(self):
        stream = self._live_stream()

        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        reused_id = stream_memory.ensure_active_stream_session(stream, source="engine")

        self.assertEqual(session_id, reused_id)
        with closing(self._conn()) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM stream_sessions").fetchone()[0], 1)
            session = conn.execute("SELECT * FROM stream_sessions WHERE id = ?", (session_id,)).fetchone()
            self.assertEqual(session["status"], "live")
            self.assertEqual(session["is_real_stream"], 1)
            self.assertEqual(session["title"], "First Playthrough Night")

        message_id = stream_memory.record_chat_message(
            username="viewer",
            display_name="Viewer",
            message_text="great fight",
            source="twitch_irc",
            topic_hint="game",
        )
        self.assertGreater(message_id, 0)

        raid_payload = {"user_login": "raider", "display_name": "Raider", "viewer_count": 7}
        first_event_id = stream_memory.record_stream_event("twitch_raid", raid_payload, stream=stream)
        second_event_id = stream_memory.record_stream_event("twitch_raid", raid_payload, stream=stream)

        self.assertGreater(first_event_id, 0)
        self.assertEqual(second_event_id, 0)

        summary = stream_memory.close_active_stream_session(stream, reason="stream_offline_event")

        self.assertIsNotNone(summary)
        self.assertIn("Final Fantasy IX", summary["summary_text"])
        self.assertIn("First Playthrough Night", summary["summary_text"])
        self.assertNotIn("sin categoria", summary["summary_text"].casefold())
        self.assertNotIn("sin titulo", summary["summary_text"].casefold())

        with closing(self._conn()) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM stream_summaries").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM stream_events").fetchone()[0], 1)
            chat = conn.execute("SELECT * FROM stream_chat_messages WHERE id = ?", (message_id,)).fetchone()
            self.assertEqual(chat["stream_session_id"], session_id)
            session = conn.execute("SELECT * FROM stream_sessions WHERE id = ?", (session_id,)).fetchone()
            self.assertEqual(session["status"], "ended")

        health = stream_memory.stream_data_health()
        self.assertEqual(health["real_sessions"], 1)
        self.assertEqual(health["sessions_missing_metadata"], 0)
        self.assertEqual(health["sessions_without_summary"], 0)
        self.assertEqual(health["possible_duplicate_events"], 0)

    def test_offline_and_simulation_inputs_do_not_create_real_sessions(self):
        stream = StreamSessionState(enabled=True)
        stream.is_live = False
        stream.current_stream_title = "Offline title"
        stream.current_category = "JRPG"

        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        message_id = stream_memory.record_chat_message(
            username="viewer",
            display_name="Viewer",
            message_text="simulated chat",
            source="simulation",
        )
        event_id = stream_memory.record_stream_event("twitch_raid", {"_simulated": True, "user_login": "raider"}, stream=stream)

        self.assertIsNone(session_id)
        self.assertEqual(message_id, 0)
        self.assertEqual(event_id, 0)
        with closing(self._conn()) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM stream_sessions").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM stream_chat_messages").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM stream_events").fetchone()[0], 0)

    def test_repair_dry_run_reports_and_execute_backfills_without_fake_real_metadata(self):
        init_live_session_schema()
        now = "2026-06-18T19:00:00+00:00"
        with closing(self._conn()) as conn:
            conn.execute(
                """
                INSERT INTO stream_sessions (
                    title, category, game, started_at, ended_at, duration_seconds,
                    status, source, is_real_stream, created_at, updated_at
                )
                VALUES (NULL, NULL, NULL, ?, ?, 3600, 'ended', 'twitch', 1, ?, ?)
                """,
                (now, "2026-06-18T20:00:00+00:00", now, now),
            )
            session_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            conn.execute(
                """
                INSERT INTO live_session_state (
                    session_id, stream_session_id, stream_status, current_game,
                    current_category, current_title, language_mode, spoiler_policy,
                    state_json, created_at, updated_at
                )
                VALUES ('live-test', ?, 'ended', 'Persona 5 Royal', 'Persona 5 Royal',
                        'Palace cleanup', 'bilingual', 'no_spoilers', '{}', ?, ?)
                """,
                (session_id, now, now),
            )
            payload = '{"user_login":"raider","display_name":"Raider","viewer_count":9}'
            for _ in range(2):
                conn.execute(
                    """
                    INSERT INTO stream_events (stream_session_id, event_type, event_ts, payload_json, created_at)
                    VALUES (?, 'twitch_raid', ?, ?, ?)
                    """,
                    (session_id, now, payload, now),
                )
            conn.commit()

        dry_run = stream_memory.repair_stream_data(dry_run=True)
        self.assertEqual(dry_run["sessions_checked"], 1)
        self.assertGreaterEqual(dry_run["sessions_repaired"], 1)
        self.assertEqual(dry_run["duplicate_events_found"], 1)

        with closing(self._conn()) as conn:
            session = conn.execute("SELECT * FROM stream_sessions WHERE id = ?", (session_id,)).fetchone()
            self.assertIsNone(session["title"])
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM stream_events").fetchone()[0], 2)

        result = stream_memory.repair_stream_data(dry_run=False)
        self.assertGreaterEqual(result["sessions_repaired"], 1)
        self.assertEqual(result["duplicate_events_removed_or_marked"], 1)
        self.assertGreaterEqual(result["summaries_regenerated"], 1)

        with closing(self._conn()) as conn:
            session = conn.execute("SELECT * FROM stream_sessions WHERE id = ?", (session_id,)).fetchone()
            self.assertEqual(session["title"], "Palace cleanup")
            self.assertEqual(session["game"], "Persona 5 Royal")
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM stream_events").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM stream_summaries WHERE stream_session_id = ?", (session_id,)).fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
