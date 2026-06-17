import unittest

from fastapi.testclient import TestClient

from app.main import app


class CapabilityApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_next_todo_endpoint_returns_stream_chat_activity_report(self):
        response = self.client.get("/capabilities/backlog/next")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        todo = payload["next_recommended_todo"]
        self.assertEqual(todo["id"], "stream.chat_activity_report")
        self.assertEqual(todo["status"], "partial")
        self.assertFalse(todo["enabled"])
        self.assertEqual(todo["priority"], "P0")
        self.assertTrue(todo["recommended_next"])
        self.assertIn("reason", todo)

    def test_backlog_collection_endpoints_exist(self):
        for path in (
            "/capabilities",
            "/capabilities/summary",
            "/capabilities/backlog",
            "/capabilities/backlog/planned",
            "/capabilities/backlog/partial",
            "/capabilities/backlog/implemented-disabled",
            "/capabilities/stream.chat_activity_report",
            "/debug/capabilities/backlog/next",
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
