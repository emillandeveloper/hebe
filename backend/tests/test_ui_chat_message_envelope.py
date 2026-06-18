import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import _prepare_ws_payload, app, hebe


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

    def test_dev_simulate_twitch_message_uses_engine_simulation(self):
        client = TestClient(app)

        class FakeEngine:
            def simulate_twitch_message(self, body):
                return {
                    "ok": True,
                    "last_policy_decision": {
                        "source": "twitch_chat",
                        "authority": "viewer",
                        "policy_decision": "blocked",
                    },
                }

        previous_engine = hebe._engine
        previous_running = hebe.running
        hebe._engine = FakeEngine()
        hebe.running = True
        try:
            with patch.dict("os.environ", {"HEBE_DEV_CONTROLS": "1"}):
                response = client.post(
                    "/dev/simulate/twitch-message",
                    json={"viewer_name": "cibernoman", "display_name": "Ciber", "text": "Hebe, envia una flor verbal para Leo"},
                )
        finally:
            hebe._engine = previous_engine
            hebe.running = previous_running

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["last_policy_decision"]["authority"], "viewer")

    def test_debug_policy_endpoints_return_engine_payloads(self):
        client = TestClient(app)

        class FakeEngine:
            def get_last_policy_trace(self):
                return {"policy_decision": "blocked", "reason": "owner_behavior_block"}

            def get_active_behavior_blocks(self):
                return [{"behavior": "compliments_to_leo"}]

        previous_engine = hebe._engine
        hebe._engine = FakeEngine()
        try:
            last = client.get("/debug/policy/last")
            blocks = client.get("/debug/policy/behavior-blocks")
        finally:
            hebe._engine = previous_engine

        self.assertEqual(last.status_code, 200)
        self.assertEqual(last.json()["last_policy_decision"]["policy_decision"], "blocked")
        self.assertEqual(blocks.status_code, 200)
        self.assertEqual(blocks.json()["behavior_blocks"][0]["behavior"], "compliments_to_leo")


if __name__ == "__main__":
    unittest.main()
