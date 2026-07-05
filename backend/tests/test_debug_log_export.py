import json
import io
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.debug import router
from app.core import persistent_logs
from app.services import db_sqlite
from app.stream import memory as stream_memory
from app.stream.state import StreamSessionState


class DebugLogExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.tmp.name) / "logs"
        self.bundle_dir = self.log_dir / "debug_bundles"
        self.session_dir = self.log_dir / "sessions"
        self.old_db_path = db_sqlite.DB_PATH
        db_sqlite.DB_PATH = str(Path(self.tmp.name) / "debug_export.sqlite3")
        patches = [
            patch.object(persistent_logs, "LOG_DIR", self.log_dir),
            patch.object(persistent_logs, "DEBUG_BUNDLE_DIR", self.bundle_dir),
            patch.object(persistent_logs, "SESSION_LOG_DIR", self.session_dir),
            patch.object(persistent_logs, "MAX_BYTES", 160),
            patch.object(persistent_logs, "BACKUP_COUNT", 2),
        ]
        self._patches = patches
        for item in patches:
            item.start()
        persistent_logs.ensure_log_dirs()

    def tearDown(self):
        for item in reversed(self._patches):
            item.stop()
        db_sqlite.DB_PATH = self.old_db_path
        self.tmp.cleanup()

    def test_logs_are_written_to_backend_logs(self):
        persistent_logs.record_console_log({
            "ts": 1,
            "source": "stdout",
            "level": "info",
            "category": "stt",
            "message": "[HEBE][STT] accepted",
        })

        self.assertTrue((self.log_dir / "current.log").exists())
        self.assertTrue((self.log_dir / "stt.log").exists())
        self.assertIn("accepted", (self.log_dir / "current.log").read_text(encoding="utf-8"))

    def test_jsonl_events_are_valid_json(self):
        persistent_logs.log_jsonl_event("cognitive_router", {"intent": "owner_personal_state"})

        line = (self.log_dir / "cognitive_router.jsonl").read_text(encoding="utf-8").strip()
        payload = json.loads(line)
        self.assertEqual(payload["intent"], "owner_personal_state")

    def test_secrets_are_redacted(self):
        persistent_logs.log_jsonl_event("input_firewall", {
            "authorization": "Bearer abcdefghijklmnop",
            "text": "api_key=sk-abcdefghijklmnopqrstuvwxyz",
        })

        text = (self.log_dir / "input_firewall.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("abcdefghijklmnop", text)
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", text)
        self.assertIn("[REDACTED]", text)

    def test_log_rotation_does_not_crash(self):
        for idx in range(8):
            persistent_logs.record_console_log({
                "ts": idx,
                "source": "stdout",
                "level": "info",
                "category": "backend",
                "message": "x" * 120,
            })

        self.assertTrue((self.log_dir / "current.log").exists())
        self.assertTrue(any(path.name.startswith("current.log.") for path in self.log_dir.iterdir()))

    def test_export_logs_returns_zip_with_required_files(self):
        persistent_logs.record_console_log({"ts": 1, "source": "stdout", "level": "info", "category": "backend", "message": "hello"})
        persistent_logs.record_console_log({"ts": 2, "source": "stderr", "level": "error", "category": "errors", "message": "boom"})
        persistent_logs.log_jsonl_event("cognitive_router", {"intent": "direct_question"})
        persistent_logs.log_jsonl_event("stt", {"status": "passed"})

        bundle = persistent_logs.create_debug_bundle(minutes=30)

        with zipfile.ZipFile(bundle) as zf:
            names = set(zf.namelist())
            self.assertIn("logs/current.log", names)
            self.assertIn("logs/errors.log", names)
            self.assertIn("logs/cognitive_router.jsonl", names)
            self.assertIn("logs/stt.jsonl", names)
            self.assertIn("state/capability_backlog_summary.json", names)
            self.assertIn("state/current_state.json", names)

    def test_debug_export_default_is_5_hours(self):
        bundle = persistent_logs.create_debug_bundle()

        with zipfile.ZipFile(bundle) as zf:
            metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        self.assertEqual(metadata["export_mode"], "last_5_hours")
        self.assertEqual(metadata["requested_minutes"], 300)
        self.assertEqual(metadata["minutes"], 300)

    def test_debug_export_metadata_reports_300_minutes(self):
        bundle = persistent_logs.create_debug_bundle(mode="last_5_hours")

        with zipfile.ZipFile(bundle) as zf:
            metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        self.assertEqual(metadata["requested_minutes"], 300)
        self.assertEqual(metadata["export_mode"], "last_5_hours")
        self.assertIn("actual_start_time", metadata)
        self.assertIn("actual_end_time", metadata)
        self.assertIn("included_logs", metadata)

    def test_debug_export_includes_rotated_logs(self):
        now = datetime.now(timezone.utc).isoformat()
        (self.log_dir / "current.log.1").write_text(f"{now} INFO backend stdout rotated early stream line\n", encoding="utf-8")

        bundle = persistent_logs.create_debug_bundle(mode="last_5_hours")

        with zipfile.ZipFile(bundle) as zf:
            names = set(zf.namelist())
            metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        self.assertIn("logs/current.log.1", names)
        self.assertIn({"path": "logs/current.log.1", "size": (self.log_dir / "current.log.1").stat().st_size}, metadata["included_logs"])

    def test_export_works_if_some_logs_are_missing(self):
        bundle = persistent_logs.create_debug_bundle(minutes=5)
        with zipfile.ZipFile(bundle) as zf:
            self.assertIn("metadata.json", zf.namelist())

    def test_export_current_stream_session_uses_session_window(self):
        stream_memory.init_stream_memory_schema()
        stream = StreamSessionState(enabled=True)
        stream.is_live = True
        stream.live_status_known = True
        stream.twitch_stream_id = "stream-123"
        stream.current_stream_title = "Live test"
        stream.current_category = "JRPG"
        stream.stream_started_at = "2026-07-04T18:00:00+00:00"
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")

        bundle = persistent_logs.create_debug_bundle(mode="current_stream_session", minutes=30)

        with zipfile.ZipFile(bundle) as zf:
            metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        self.assertEqual(metadata["mode"], "current_stream_session")
        self.assertEqual(metadata["session_id"], session_id)
        self.assertEqual(metadata["stream_id"], "stream-123")
        self.assertIn("log_window", metadata)

    def test_export_does_not_include_raw_secrets(self):
        persistent_logs.record_console_log({
            "ts": 1,
            "source": "stdout",
            "level": "info",
            "category": "backend",
            "message": "authorization: Bearer abcdefghijklmnop",
        })

        bundle = persistent_logs.create_debug_bundle(minutes=30, include_config=True)
        with zipfile.ZipFile(bundle) as zf:
            combined = "\n".join(zf.read(name).decode("utf-8", errors="replace") for name in zf.namelist())
        self.assertNotIn("abcdefghijklmnop", combined)
        self.assertIn("[REDACTED]", combined)

    def test_debug_export_endpoint_returns_zip(self):
        app = FastAPI()
        app.include_router(router)
        app.state.adapter = None
        client = TestClient(app)

        response = client.get("/debug/export-logs?minutes=5")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")
        self.assertGreater(len(response.content), 20)

    def test_debug_export_accepts_minutes_300(self):
        app = FastAPI()
        app.include_router(router)
        app.state.adapter = None
        client = TestClient(app)

        response = client.get("/debug/export-logs?minutes=300&mode=last_5_hours")

        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        self.assertEqual(metadata["requested_minutes"], 300)
        self.assertEqual(metadata["export_mode"], "last_5_hours")

    def test_recent_logs_endpoint_returns_preview(self):
        persistent_logs.log_jsonl_event("cognitive_router", {"intent": "direct_question"})
        persistent_logs.log_jsonl_event("stt", {"status": "passed"})
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/debug/logs/recent?minutes=10")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("cognitive_router", payload)
        self.assertIn("stt", payload)

    def test_debug_export_ui_default_button_last_5h(self):
        app_path = Path(__file__).resolve().parents[2] / "frontend" / "src" / "App.tsx"
        source = app_path.read_text(encoding="utf-8")

        self.assertIn('useState("last_5_hours")', source)
        self.assertIn("Export last 5h", source)
        self.assertIn("minutes: 300", source)


if __name__ == "__main__":
    unittest.main()
