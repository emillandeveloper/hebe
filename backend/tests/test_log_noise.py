import os
import unittest
from unittest.mock import patch

from app.cognitive.scheduler import SchedulerService
from app.services.stt_whisper import STTConfig


class FakeMemoryStore:
    def __init__(self, due=None):
        self.due = due or []

    def list_due_reminders(self, limit=20):
        return self.due[:limit]

    def list_pending_reminders(self, limit=1000):
        return []


class LogNoiseTests(unittest.TestCase):
    def test_scheduler_poll_is_quiet_without_due_when_not_verbose(self):
        scheduler = SchedulerService(FakeMemoryStore())
        logs = []

        with patch.dict(os.environ, {"HEBE_VERBOSE_SCHEDULER_LOGS": "false"}), \
             patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            scheduler.poll_due_events()

        self.assertFalse(any("[HEBE][SCHEDULER] poll" in item for item in logs))

    def test_scheduler_poll_logs_when_verbose(self):
        scheduler = SchedulerService(FakeMemoryStore())
        logs = []

        with patch.dict(os.environ, {"HEBE_VERBOSE_SCHEDULER_LOGS": "true"}), \
             patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            scheduler.poll_due_events()

        self.assertTrue(any("[HEBE][SCHEDULER] poll pending=0 due=0" in item for item in logs))

    def test_stt_silence_log_cooldown_uses_new_env(self):
        with patch.dict(os.environ, {"HEBE_STT_SILENCE_LOG_COOLDOWN_SECONDS": "123"}):
            cfg = STTConfig()

        self.assertEqual(cfg.silence_warning_rate_limit_seconds, 123.0)

    def test_stt_device_logs_default_to_quiet(self):
        with patch.dict(os.environ, {}, clear=False):
            cfg = STTConfig()

        self.assertFalse(cfg.verbose_device_logs)


if __name__ == "__main__":
    unittest.main()
