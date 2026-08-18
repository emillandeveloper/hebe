import os
import time
import unittest
from unittest.mock import patch

from app import main


class BackendHealthTests(unittest.TestCase):
    def test_health_publishes_real_process_identity_and_uptime(self):
        with patch.object(main.hebe, "_engine", None):
            first = main.health()
            time.sleep(0.002)
            second = main.health()

        self.assertTrue(first["ok"])
        self.assertEqual(first["pid"], os.getpid())
        self.assertEqual(first["parent_pid"], os.getppid())
        self.assertGreater(first["started_at"], 0)
        self.assertGreaterEqual(second["uptime_ms"], first["uptime_ms"])
