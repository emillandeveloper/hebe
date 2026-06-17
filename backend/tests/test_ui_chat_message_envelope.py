import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import _prepare_ws_payload, app


class UiChatMessageEnvelopeTests(unittest.TestCase):
    def test_chat_assistant_legacy_event_becomes_stable_chat_message(self):
        payload = _prepare_ws_payload(
            {
                "type": "chat.assistant",
                "event_id": "evt_test",
                "ts": 123.0,
                "data": {
                    "message_id": "msg_test",
                    "text": "Hola Leo",
                    "source": "ui",
                    "output_target": "local_ui",
                },
            }
        )

        self.assertEqual(payload["type"], "chat_message")
        self.assertEqual(payload["event_id"], "evt_test")
        self.assertEqual(payload["message"]["message_id"], "msg_test")
        self.assertEqual(payload["message"]["role"], "assistant")
        self.assertEqual(payload["message"]["speaker"], "Hebe")
        self.assertEqual(payload["message"]["text"], "Hola Leo")
        self.assertEqual(payload["data"]["legacy_type"], "chat.assistant")

    def test_chat_message_ids_are_created_when_missing(self):
        payload = _prepare_ws_payload(
            {
                "type": "chat.user",
                "ts": 123.0,
                "data": {
                    "text": "Hebe, me escuchas?",
                    "source": "ui",
                },
            }
        )

        self.assertEqual(payload["type"], "chat_message")
        self.assertTrue(payload["event_id"].startswith("evt_"))
        self.assertTrue(payload["message"]["message_id"].startswith("msg_"))
        self.assertEqual(payload["message"]["role"], "user")

    def test_dev_test_ui_message_endpoint_is_dev_gated(self):
        client = TestClient(app)

        with patch.dict("os.environ", {"HEBE_DEV_CONTROLS": "1"}):
            response = client.post("/dev/test-ui-message")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["event_id"].startswith("evt_"))
        self.assertTrue(payload["message_id"].startswith("msg_"))
        self.assertEqual(payload["text"], "Hebe UI test message")


if __name__ == "__main__":
    unittest.main()
