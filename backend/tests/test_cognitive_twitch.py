import unittest
from types import SimpleNamespace

from app.cognitive.deliberation_service import DeliberationService
from app.cognitive.scheduler import SchedulerService, InternalEvent
from app.integrations.twitch.chat_bot import TwitchChatBot


class DummyMemoryStore:
    def __init__(self):
        self.logged_events = []

    def log_internal_event(self, event_type: str, payload: dict) -> None:
        self.logged_events.append((event_type, payload))

    def list_due_reminders(self, limit: int):
        return []

    def mark_reminder_fired(self, reminder_id):
        raise AssertionError("mark_reminder_fired should not be called in this test")


class DummyModel:
    pass


class CognitiveTwitchTests(unittest.TestCase):
    def test_scheduler_push_event_enqueues_manual_internal_event(self):
        memory_store = DummyMemoryStore()
        scheduler = SchedulerService(memory_store)

        event = scheduler.push_event("twitch_sub", {"display_name": "StreamerFan"})

        self.assertIsInstance(event, InternalEvent)
        self.assertEqual(event.event_type, "twitch_sub")
        self.assertEqual(event.payload, {"display_name": "StreamerFan"})
        self.assertEqual(len(scheduler._pending), 1)
        self.assertEqual(memory_store.logged_events, [("twitch_sub", {"display_name": "StreamerFan"})])

        drained = scheduler.poll_due_events(limit=1)
        self.assertEqual(drained, [event])
        self.assertEqual(len(scheduler._pending), 0)

    def test_twitch_chat_bot_parses_privmsg_with_tags(self):
        received = []

        def callback(username, display_name, text, channel):
            received.append({
                "username": username,
                "display_name": display_name,
                "text": text,
                "channel": channel,
            })

        bot = TwitchChatBot(
            channel_name="testchannel",
            bot_username="TestBot",
            oauth_token="oauth:dummy",
            enabled=True,
            message_callback=callback,
        )

        line = (
            "@badge-info=;badges=;color=;display-name=Viewer;emotes=;flags=;id=1234; "
            ":viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #testchannel :Hola Hebe"
        )
        bot._handle_privmsg(line)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["username"], "viewer")
        self.assertEqual(received[0]["display_name"], "viewer")
        self.assertEqual(received[0]["text"], "Hola Hebe")
        self.assertEqual(received[0]["channel"], "#testchannel")

    def test_deliberation_service_plans_twitch_event_as_reply(self):
        deliberation_service = DeliberationService(
            intent_model=DummyModel(),
            reasoning_model=DummyModel(),
        )

        event = InternalEvent(
            event_type="twitch_raid",
            payload={"display_name": "Broadcaster"},
            created_at="2026-04-26T12:00:00Z",
        )
        context = SimpleNamespace(internal_event=event, input_text=None)

        result = deliberation_service.deliberate(context)

        self.assertTrue(result.plan.steps, "Expected at least one plan step")
        step = result.plan.steps[0]
        self.assertEqual(step.type, "reply")
        self.assertEqual(step.data["mode"], "twitch_raid")
        self.assertEqual(step.data["payload"], {"display_name": "Broadcaster"})


if __name__ == "__main__":
    unittest.main()
