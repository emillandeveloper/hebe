import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.debug import router
from app.core import persistent_logs


class DebugLogExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.tmp.name) / "logs"
        self.bundle_dir = self.log_dir / "debug_bundles"
        self.session_dir = self.log_dir / "sessions"
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

    def test_export_works_if_some_logs_are_missing(self):
        bundle = persistent_logs.create_debug_bundle(minutes=5)
        with zipfile.ZipFile(bundle) as zf:
            self.assertIn("metadata.json", zf.namelist())

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


if __name__ == "__main__":
    unittest.main()
