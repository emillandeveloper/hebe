import unittest

from app.core.log_bus import classify_log_line


class LogBusTests(unittest.TestCase):
    def test_twitch_log_is_categorized(self):
        level, category = classify_log_line("[HEBE][TWITCH][CHATBOT] incoming message", "stdout")

        self.assertEqual(level, "info")
        self.assertEqual(category, "twitch")

    def test_stream_context_log_is_categorized(self):
        level, category = classify_log_line("[HEBE][STREAM_CONTEXT] refresh started", "stdout")

        self.assertEqual(level, "info")
        self.assertEqual(category, "stream_context")

    def test_stderr_log_is_error(self):
        level, category = classify_log_line("Traceback: boom", "stderr")

        self.assertEqual(level, "error")
        self.assertIn(category, {"errors", "backend", "status"})


if __name__ == "__main__":
    unittest.main()
