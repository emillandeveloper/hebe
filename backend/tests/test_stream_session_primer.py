import os
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from app.cognitive.command_result import CommandResult
from app.hebe_engine import HebeEngine
from app.services import db_sqlite
from app.stream import session_primer
from app.stream.spontaneity import StreamSpontaneityConfig, StreamSpontaneityService
from app.stream.state import StreamSessionState


class StreamSessionPrimerTests(unittest.TestCase):
    def setUp(self):
        self.original_db_path = db_sqlite.DB_PATH
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        self.tmp_path = tmp.name
        db_sqlite.DB_PATH = self.tmp_path
        db_sqlite.init_db()
        session_primer.init_session_primer_schema()

    def tearDown(self):
        db_sqlite.DB_PATH = self.original_db_path
        try:
            os.unlink(self.tmp_path)
        except OSError:
            pass

    def test_today_weekday_resolves_to_scheduled_game(self):
        dt = datetime(2026, 6, 9, 18, 0, tzinfo=ZoneInfo("Europe/Madrid"))

        schedule = session_primer.get_schedule_for_date(dt)

        self.assertEqual(schedule["weekday"], "tuesday")
        self.assertEqual(schedule["game"], "Persona 5 Royal")

    def test_prepare_stream_today_creates_primer(self):
        dt = datetime(2026, 6, 9, 18, 0, tzinfo=ZoneInfo("Europe/Madrid"))

        primer = session_primer.build_stream_session_primer(dt=dt)

        self.assertEqual(primer.game, "Persona 5 Royal")
        self.assertEqual(primer.playthrough_type, "First Playthrough")
        self.assertIn("previous_session", primer.missing_info)
        self.assertTrue(primer.title_suggestions)

    def test_persona_title_uses_leo_standard_format(self):
        session = {
            "end_summary": "terminamos después del museo",
            "next_time_plan": "ver qué viene después del museo",
        }

        titles = session_primer.generate_title_suggestions(
            "Persona 5 Royal",
            playthrough_type="First Playthrough",
            last_session=session,
            count=1,
        )

        self.assertRegex(titles[0], r"^\[ENG/ESP\] .+ — Persona 5 Royal \| First Playthrough$")
        self.assertIn("Museum", titles[0])

    def test_last_game_session_is_loaded_when_available(self):
        primer = session_primer.build_stream_session_primer(
            game="Persona 5 Royal",
            dt=datetime(2026, 6, 9, 18, 0, tzinfo=ZoneInfo("Europe/Madrid")),
            canonical_run_state={
                "run_id": "game_run_persona",
                "last_confirmed_progress": "terminamos después del museo",
                "current_objective": "ver qué viene después del museo",
                "spoiler_policy": "no_spoilers",
            },
        )

        self.assertNotIn("previous_session", primer.missing_info)
        self.assertIn("museo", primer.last_session_summary)
        self.assertIn("museo", primer.likely_objective)

    def test_manual_correction_updates_schedule_and_session_state(self):
        engine = HebeEngine.__new__(HebeEngine)
        stream = StreamSessionState(enabled=True, presence_mode="reactive")
        engine.runtime = SimpleNamespace(state=SimpleNamespace(stream=stream))

        def command_result(action_type, fallback_text, **kwargs):
            return CommandResult(
                action_type=action_type,
                success=kwargs.get("success", True),
                user_visible_summary=kwargs.get("message_goal") or fallback_text,
                state_changes=kwargs.get("state_changes") or {},
                fallback_text=fallback_text,
                requires_model_response=True,
                metadata={"message_goal": kwargs.get("message_goal") or fallback_text},
            )

        result = engine._handle_stream_session_primer_command(
            "hoy toca Persona 5 Royal",
            "hoy toca persona 5 royal",
            stream,
            command_result,
        )

        self.assertIsInstance(result, CommandResult)
        self.assertEqual(result.action_type, "set_today_stream_game")
        self.assertTrue(result.requires_model_response)
        self.assertEqual(stream.current_game, "Persona 5 Royal")
        self.assertTrue(getattr(stream, "session_primer", None))

    def test_spontaneity_uses_primer_context_when_available(self):
        stream = StreamSessionState(enabled=True, presence_mode="show")
        stream.is_live = True
        stream.live_status_known = True
        stream.last_chat_activity_ts = 1_000_000 - 60 * 60
        stream.stream_context_updated_ts = 1_000_000
        primer = session_primer.build_stream_session_primer(
            game="Persona 5 Royal",
            dt=datetime(2026, 6, 9, 18, 0, tzinfo=ZoneInfo("Europe/Madrid")),
        )
        session_primer.apply_primer_to_stream(stream, primer)
        service = StreamSpontaneityService(
            config=StreamSpontaneityConfig(
                require_specific_context=True,
                show_silence_sec=5 * 60,
                show_jitter_sec=0,
                startup_grace_sec=0,
            ),
            now_fn=lambda: 1_000_000,
        )

        event = service.build_due_event(stream)

        self.assertIsNotNone(event)
        self.assertIn("session_primer", event.payload["specific_context_anchors"])
        self.assertEqual(event.payload["session_primer"]["game"], "Persona 5 Royal")

    def test_generic_spontaneity_is_skipped_without_primer_or_run_context(self):
        stream = StreamSessionState(enabled=True, presence_mode="show")
        stream.is_live = True
        stream.live_status_known = True
        stream.last_chat_activity_ts = 1_000_000 - 60 * 60
        stream.stream_context_updated_ts = 1_000_000
        stream.current_game = "Persona 5 Royal"
        service = StreamSpontaneityService(
            config=StreamSpontaneityConfig(
                require_specific_context=True,
                show_silence_sec=5 * 60,
                show_jitter_sec=0,
                startup_grace_sec=0,
            ),
            now_fn=lambda: 1_000_000,
        )

        event = service.build_due_event(stream)

        self.assertIsNone(event)
        self.assertEqual(stream.last_stream_spontaneity_blocked_reason, "no_session_primer_or_run_context")


if __name__ == "__main__":
    unittest.main()
