import unittest
from types import SimpleNamespace

from app.cognitive.deliberation_service import DeliberationService
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.cognitive.scheduler import SchedulerService, InternalEvent
from app.integrations.twitch.chat_bot import TwitchChatBot
from app.integrations.twitch.service import TwitchService


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


class CapturingModel:
    def __init__(self, reply):
        self.reply = reply
        self.messages = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        return self.reply


class CapturingChatClient:
    def __init__(self):
        self.sent = []

    def send_message(self, text):
        self.sent.append(text)


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

    def test_twitch_chat_bot_detects_at_bot_mentions(self):
        received = []
        bot = TwitchChatBot(
            channel_name="testchannel",
            bot_username="HebeNifelheim",
            oauth_token="oauth:dummy",
            enabled=True,
            message_callback=lambda *args: received.append(args),
        )

        bot._handle_privmsg(":viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #testchannel :@HebeNifelheim que opinas?")
        bot._handle_privmsg(":viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #testchannel :@hebenifelheim despierta")
        bot._handle_privmsg(":viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #testchannel :HebeNifelheim mira esto")

        self.assertEqual(len(received), 3)

    def test_twitch_chat_bot_uses_configured_bot_username_for_mentions(self):
        received = []
        bot = TwitchChatBot(
            channel_name="testchannel",
            bot_username="OtherHebe",
            oauth_token="oauth:dummy",
            enabled=True,
            message_callback=lambda *args: received.append(args),
        )

        bot._handle_privmsg(":viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #testchannel :@otherhebe hola")

        self.assertEqual(len(received), 1)

    def test_twitch_esquirola_reply_is_short_in_character(self):
        synth = ResponseSynthesizer(conversation_model=CapturingModel("alguien tiene que recordarles que no son el centro del universo."))

        reply = synth._generate_twitch_chat_react({
            "user_login": "viewer",
            "display_name": "Viewer",
            "message_text": "hebe la esquirola",
            "recent_chat": [],
        })

        self.assertEqual(reply, "esquirola no, superviviente sindical del caos.")
        self.assertLessEqual(len(reply.split()), 10)

    def test_twitch_sexual_or_aggressive_mention_deflects_without_escalation(self):
        synth = ResponseSynthesizer(conversation_model=CapturingModel("vete a la mierda con esa imagen sexual explicita y larga."))

        reply = synth._generate_twitch_chat_react({
            "user_login": "viewer",
            "display_name": "Viewer",
            "message_text": "hebe puta",
            "recent_chat": [],
        })

        self.assertIn("bonito vocabulario", reply)
        self.assertLessEqual(len(reply.split()), 10)

    def test_twitch_service_shoutout_uses_configurable_template(self):
        chat = CapturingChatClient()
        twitch = TwitchService(chat_client=chat, shoutout_command_template="!promo {username}")

        ok = twitch.shoutout("@Totodile")

        self.assertTrue(ok)
        self.assertEqual(chat.sent, ["!promo Totodile"])

    def test_twitch_chat_bot_ignores_own_bot_messages_and_unrelated_words(self):
        received = []
        bot = TwitchChatBot(
            channel_name="testchannel",
            bot_username="HebeNifelheim",
            oauth_token="oauth:dummy",
            enabled=True,
            message_callback=lambda *args: received.append(args),
        )

        bot._handle_privmsg(":HebeNifelheim!bot@bot.tmi.twitch.tv PRIVMSG #testchannel :@HebeNifelheim hola")
        bot._handle_privmsg(":viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #testchannel :alhebedo no cuenta")

        self.assertEqual(received, [])

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
        context = SimpleNamespace(
            internal_event=event, input_text=None, state_snapshot={},
            source="twitch_system", authority="system", addressed_to_hebe=False,
            firewall_decision="allow", stream_is_live=True,
        )

        result = deliberation_service.deliberate(context)

        self.assertTrue(result.plan.steps, "Expected at least one plan step")
        step = result.plan.steps[0]
        self.assertEqual(step.type, "reply")
        self.assertEqual(step.data["mode"], "twitch_raid")
        self.assertEqual(step.data["payload"], {"display_name": "Broadcaster"})

    def test_response_synthesizer_handles_twitch_idle_prompt(self):
        event = InternalEvent(
            event_type="twitch_idle_prompt",
            payload={
                "reason": "stream_companion_prompt",
                "presence_mode": "show",
                "silence_seconds": 360,
            },
            created_at="2026-05-31T12:00:00Z",
        )
        event.payload["specific_context_anchors"] = ["game"]
        event.payload["current_category"] = "JRPG"
        context = SimpleNamespace(internal_event=event, input_text=None)

        reply = ResponseSynthesizer(conversation_model=CapturingModel("La run pide mirar recursos antes del siguiente susto."))._handle_internal_event(
            context,
            execution=SimpleNamespace(),
        )

        self.assertTrue(reply)
        self.assertLessEqual(len(reply), 220)

    def test_response_synthesizer_rejects_dot_only_spontaneous_reply(self):
        synth = ResponseSynthesizer(conversation_model=None)

        reply = synth._safe_spontaneous_stream_reply(".", "fallback", payload={"specific_context_anchors": ["game"]})

        self.assertEqual(reply, "")

    def test_response_synthesizer_rejects_repeated_overused_motif(self):
        synth = ResponseSynthesizer(conversation_model=None)

        reply = synth._safe_spontaneous_stream_reply(
            "Otro café antes de entrar, Leo.",
            "fallback",
            payload={
                "specific_context_anchors": ["game"],
                "recent_style_motifs": ["cafe"],
            },
        )

        self.assertEqual(reply, "")

    def test_spontaneous_prompt_includes_stream_context(self):
        model = CapturingModel("Mi senor, revisa recursos antes de avanzar.")
        event = InternalEvent(
            event_type="twitch_idle_prompt",
            payload={
                "title": "Challenge Playthrough Level 1",
                "current_category": "Final Fantasy X",
                "playthrough_type": "challenge",
                "challenge": "level_1",
                "spoiler_policy": "no_spoilers",
                "last_voice_event": "menu/equipment",
                "leo_mood_hint": "focused",
                "presence_mode": "show",
                "game_profile": {
                    "title": "Final Fantasy X",
                    "genres": ["JRPG", "turn-based"],
                    "tone_vibe": "pilgrimage fantasy",
                    "gameplay_systems_non_spoiler": ["turn-based combat", "resource management"],
                    "channel_context": "Final Fantasy stream with strong no-spoiler needs.",
                    "safe_comment_topics": ["resource checks", "save spheres"],
                    "spoiler_policy": "no_spoilers",
                    "unsafe_comment_topics": ["story twists", "boss names"],
                    "challenge_hooks": ["No Sphere Grid is about patience, not route advice"],
                },
            },
            created_at="2026-05-31T12:00:00Z",
        )
        context = SimpleNamespace(internal_event=event, input_text=None)

        reply = ResponseSynthesizer(conversation_model=model)._handle_internal_event(
            context,
            execution=SimpleNamespace(),
        )

        prompt_text = "\n".join(message["content"] for message in model.messages)
        self.assertEqual(reply, "Mi senor, revisa recursos antes de avanzar.")
        self.assertIn("Challenge Playthrough Level 1", prompt_text)
        self.assertIn("Final Fantasy X", prompt_text)
        self.assertIn("challenge", prompt_text)
        self.assertIn("level_1", prompt_text)
        self.assertIn("game_profile:", prompt_text)
        self.assertIn("tone_vibe: pilgrimage fantasy", prompt_text)
        self.assertIn("turn-based combat", prompt_text)
        self.assertIn("save spheres", prompt_text)
        self.assertIn("story twists", prompt_text)
        self.assertIn("No Sphere Grid is about patience", prompt_text)
        self.assertIn("No spoilers", prompt_text)

    def test_spontaneous_reply_filters_forbidden_silence_viewer_phrases(self):
        forbidden_replies = [
            "Silencio en la sala, no hay viewers.",
            "Si alguien esta lurking, que vote.",
            "Aunque no haya nadie, seguimos.",
            "Chat esta muerto hoy.",
            "Esta esto tranquilo.",
        ]
        for raw_reply in forbidden_replies:
            with self.subTest(raw_reply=raw_reply):
                model = CapturingModel(raw_reply)
                event = InternalEvent(
                    event_type="twitch_idle_prompt",
                    payload={"current_category": "JRPG", "presence_mode": "companion"},
                    created_at="2026-05-31T12:00:00Z",
                )
                context = SimpleNamespace(internal_event=event, input_text=None)

                reply = ResponseSynthesizer(conversation_model=model)._handle_internal_event(
                    context,
                    execution=SimpleNamespace(),
                )

                self.assertNotEqual(reply, raw_reply)
                lowered = reply.lower()
                self.assertNotIn("silencio", lowered)
                self.assertNotIn("viewer", lowered)
                self.assertNotIn("lurking", lowered)
                self.assertNotIn("aunque no haya nadie", lowered)

    def test_ff9_idle_prompt_rejects_spheres(self):
        model = CapturingModel("Revisa las esferas antes de avanzar.")
        event = InternalEvent(
            event_type="twitch_idle_prompt",
            payload={
                "current_category": "Final Fantasy IX",
                "presence_mode": "companion",
                "game_profile": {"title": "Final Fantasy IX"},
            },
            created_at="2026-06-01T12:00:00Z",
        )
        context = SimpleNamespace(internal_event=event, input_text=None)

        reply = ResponseSynthesizer(conversation_model=model)._handle_internal_event(
            context,
            execution=SimpleNamespace(),
        )

        self.assertNotIn("esfera", reply.lower())


if __name__ == "__main__":
    unittest.main()
