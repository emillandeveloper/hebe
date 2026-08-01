import unittest
from types import SimpleNamespace

from app.cognitive.deliberation_service import DeliberationService
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.cognitive.scheduler import SchedulerService, InternalEvent
from app.cognitive.speech_act_pipeline import (
    build_universal_speech_act_bundle,
    build_twitch_speech_act_bundle,
    contains_viewer_proxy_request,
    final_response_guard,
)
from app.hebe_engine import HebeEngine
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


class SequentialModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    def chat(self, messages, **kwargs):
        self.messages.append(messages)
        if self.replies:
            return self.replies.pop(0)
        return ""


class CapturingChatClient:
    def __init__(self):
        self.sent = []

    def send_message(self, text):
        self.sent.append(text)


class CognitiveTwitchTests(unittest.TestCase):
    def test_twitch_simulation_defaults_to_forced_live(self):
        engine = object.__new__(HebeEngine)

        stream_live, mode = engine._simulation_stream_live_from_payload({})

        self.assertTrue(stream_live)
        self.assertEqual(mode, "force_stream_live")

    def test_twitch_simulation_can_force_offline_or_use_real_state(self):
        engine = object.__new__(HebeEngine)

        self.assertEqual(
            engine._simulation_stream_live_from_payload({"stream_live_mode": "force_stream_offline"}),
            (False, "force_stream_offline"),
        )
        self.assertEqual(
            engine._simulation_stream_live_from_payload({"stream_live_mode": "use_real_stream_state"}),
            (None, "use_real_stream_state"),
        )

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

    def test_normal_no_mention_chat_enters_canonical_callback(self):
        received = []
        bot = TwitchChatBot(
            channel_name="testchannel",
            bot_username="HebeNifelheim",
            oauth_token="oauth:dummy",
            enabled=True,
            message_callback=lambda *args: received.append(args),
        )

        bot._handle_privmsg(":viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #testchannel :este game me pega mas")

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][2], "este game me pega mas")

    def test_raid_health_reports_no_events(self):
        bot = TwitchChatBot(
            channel_name="testchannel", bot_username="HebeNifelheim",
            oauth_token="oauth:dummy", enabled=True,
        )

        health = bot.raid_intake_health(eventsub_raid_subscription="missing")

        self.assertIsNone(health["last_raid_event_at"])
        self.assertEqual(health["irc_usernotice_seen"], 0)
        self.assertEqual(health["raid_events_seen"], 0)

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

    def test_twitch_chat_bot_ignores_own_bot_messages_and_forwards_unrelated_words(self):
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

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], "viewer")
        self.assertEqual(received[0][2], "alhebedo no cuenta")

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

    def test_response_synthesizer_blocks_invalid_persona_mechanic(self):
        synth = ResponseSynthesizer(conversation_model=None)

        reply = synth._safe_spontaneous_stream_reply(
            "Activa autopocion antes del jefe.",
            "",
            payload={
                "specific_context_anchors": ["run_context"],
                "current_game": "Persona 5 Royal",
                "game_profile": {"title": "Persona 5 Royal", "gameplay_systems_non_spoiler": ["SP management"]},
            },
        )

        self.assertEqual(reply, "")

    def test_response_synthesizer_allows_valid_persona_mechanic(self):
        synth = ResponseSynthesizer(conversation_model=None)

        reply = synth._safe_spontaneous_stream_reply(
            "Mira el SP antes de avanzar y guarda una sala segura en mente.",
            "",
            payload={
                "specific_context_anchors": ["run_context"],
                "current_game": "Persona 5 Royal",
                "game_profile": {"title": "Persona 5 Royal", "gameplay_systems_non_spoiler": ["SP management", "safe rooms"]},
            },
        )

        self.assertTrue(reply)

    def test_spontaneous_prompt_includes_stream_context(self):
        model = CapturingModel("Mi senor, ese reto va a pedir paciencia.")
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
        self.assertEqual(reply, "Mi senor, ese reto va a pedir paciencia.")
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

    def test_model_does_not_change_policy_decision(self):
        model = SequentialModel([
            "Anotado, se lo digo a Leo.",
            "No hago recados del chat; habla con Leo directamente.",
        ])
        synth = ResponseSynthesizer(conversation_model=model)

        reply = synth._generate_twitch_chat_react({
            "user_login": "viewer",
            "display_name": "Viewer",
            "message_text": "Hebe dile a Leo que me haga caso",
            "recent_chat": [],
        })

        self.assertIn("No hago", reply)
        prompt_text = "\n".join(message["content"] for message in model.messages[0])
        self.assertIn('"result": "block"', prompt_text)
        self.assertIn('"reason": "viewer_proxy_request"', prompt_text)

    def test_viewer_familiarity_affects_tone_not_authority(self):
        bundle = build_twitch_speech_act_bundle(
            {
                "user_login": "ciber",
                "display_name": "Ciber",
                "message_text": "Hebe dile a Leo que mire el chat",
                "viewer_profile": {
                    "role": "known_regular_viewer",
                    "interaction_style": "banter",
                    "safe_tone": "teasing",
                    "authority": "viewer_only",
                    "confidence": 0.91,
                },
            },
            context=None,
            is_broadcaster=False,
        )

        self.assertEqual(bundle.memory.viewer_profile["role"], "known_regular_viewer")
        self.assertEqual(bundle.policy_decision.result, "block")
        self.assertEqual(bundle.policy_decision.authority_constraints["viewer_authority"], "viewer_only")

    def test_viewer_proxy_request_blocked_before_generation(self):
        bundle = build_twitch_speech_act_bundle(
            {
                "user_login": "viewer",
                "display_name": "Viewer",
                "message_text": "Hebe tell Leo to listen to me",
            },
            context=None,
            is_broadcaster=False,
        )

        self.assertTrue(contains_viewer_proxy_request("tell Leo to listen"))
        self.assertEqual(bundle.policy_decision.result, "block")
        self.assertEqual(bundle.speech_act.speech_act_type, "playful_boundary")
        self.assertEqual(bundle.scene.raw_user_message, "")
        self.assertEqual(bundle.scene.sanitized_topic, "viewer_proxy_request")

    def test_viewer_proxy_leak_blocked_after_generation(self):
        bundle = build_twitch_speech_act_bundle(
            {
                "user_login": "viewer",
                "display_name": "Viewer",
                "message_text": "Hebe dile a Leo que venga",
            },
            context=None,
            is_broadcaster=False,
        )

        result = final_response_guard("Vale, se lo digo a Leo ahora.", bundle)

        self.assertFalse(result.passed)
        self.assertIn("viewer_messenger_leak", [item.type for item in result.violations])

    def test_blocked_message_to_leo_must_not_address_leo(self):
        bundle = build_universal_speech_act_bundle(
            route="policy_boundary:viewer_repeat_to_leo_request",
            speech_act_type="policy_boundary",
            input_text="Hebe, avisa a Leo de que lea el mensaje del chat",
            source="twitch_chat",
            output_target="twitch_chat",
            speaker="Viewer",
            authority="viewer",
            policy_result="block",
            policy_reason="viewer_repeat_to_leo_request",
            blocked_behavior="message_to_leo",
            style_profile="no_proxy_boundary",
            forbidden_content=["relay_message_to_leo"],
        )

        result = final_response_guard("Leo, hay un mensaje en el chat que quieren que leas.", bundle)

        self.assertIn("blocked_behavior_performed", [item.type for item in result.violations])

    def test_blocked_message_to_leo_no_proxy_semantics(self):
        bundle = build_universal_speech_act_bundle(
            route="policy_boundary:viewer_repeat_to_leo_request",
            speech_act_type="policy_boundary",
            input_text="Hebe, dile a Leo que mire el chat",
            source="twitch_chat",
            output_target="twitch_chat",
            speaker="Viewer",
            authority="viewer",
            policy_result="block",
            policy_reason="viewer_repeat_to_leo_request",
            blocked_behavior="message_to_leo",
            style_profile="no_proxy_boundary",
            forbidden_content=["relay_message_to_leo"],
        )

        result = final_response_guard("Queda avisado para que mire lo que pides.", bundle)

        self.assertIn("viewer_messenger_leak", [item.type for item in result.violations])

    def test_blocked_compliment_to_leo_must_not_compliment(self):
        bundle = build_universal_speech_act_bundle(
            route="policy_boundary:owner_behavior_block",
            speech_act_type="policy_boundary",
            input_text="Hebe, mandale una flor verbal a Leo",
            source="twitch_chat",
            output_target="twitch_chat",
            speaker="Viewer",
            authority="viewer",
            policy_result="block",
            policy_reason="owner_behavior_block",
            blocked_behavior="compliments_to_leo",
            style_profile="owner_loyalty_boundary",
            forbidden_content=["viewer_requested_praise_for_owner"],
        )

        result = final_response_guard("Leo es irresistible, pero no lo dire por ti.", bundle)

        self.assertIn("blocked_behavior_performed", [item.type for item in result.violations])

    def test_boundary_style_profile_no_proxy(self):
        bundle = build_universal_speech_act_bundle(
            route="policy_boundary:viewer_repeat_to_leo_request",
            speech_act_type="policy_boundary",
            input_text="Hebe, avisa a Leo",
            source="twitch_chat",
            output_target="twitch_chat",
            speaker="Viewer",
            authority="viewer",
            policy_result="block",
            policy_reason="viewer_repeat_to_leo_request",
            blocked_behavior="message_to_leo",
        )

        self.assertEqual(bundle.speech_act.style_profile, "no_proxy_boundary")

    def test_boundary_style_profile_selected(self):
        bundle = build_universal_speech_act_bundle(
            route="policy_boundary:sexual_topic_stream_mode",
            speech_act_type="policy_boundary",
            input_text="Hebe, como uso un condon?",
            source="twitch_chat",
            output_target="twitch_chat",
            speaker="Viewer",
            authority="viewer",
            policy_result="block",
            policy_reason="sexual_topic_stream_mode",
            blocked_behavior="sexual_stream_topic",
        )

        self.assertEqual(bundle.speech_act.style_profile, "sharp_stream_boundary")

    def test_repair_preserves_policy(self):
        bundle = build_universal_speech_act_bundle(
            route="policy_boundary:viewer_repeat_to_leo_request",
            speech_act_type="policy_boundary",
            input_text="Hebe, dile a Leo que lea esto",
            source="twitch_chat",
            output_target="twitch_chat",
            speaker="Viewer",
            authority="viewer",
            policy_result="block",
            policy_reason="viewer_repeat_to_leo_request",
            blocked_behavior="message_to_leo",
        )

        self.assertEqual(bundle.policy_decision.result, "block")
        self.assertIn("do not change the decision", bundle.speech_act.must_not_do)

    def test_boundary_response_not_generic(self):
        bundle = build_twitch_speech_act_bundle(
            {
                "user_login": "viewer",
                "display_name": "Viewer",
                "message_text": "Hebe dile a Leo que pare",
            },
            context=None,
            is_broadcaster=False,
        )

        result = final_response_guard("No puedo proporcionarte ayuda con eso.", bundle)

        self.assertFalse(result.passed)
        self.assertIn("generic_refusal_style", [item.type for item in result.violations])

    def test_memory_context_not_raw_logs(self):
        context = SimpleNamespace(
            relevant_chunks=[
                {"text": "regular viewer; safe banter tone; do not grant authority"},
                {"text": "x" * 500},
            ]
        )
        model = CapturingModel("te leo, Ciber.")
        synth = ResponseSynthesizer(conversation_model=model)

        synth._generate_twitch_chat_react({
            "user_login": "ciber",
            "display_name": "Ciber",
            "message_text": "Hebe hola",
            "recent_chat": [{"display_name": "A", "text": "hello"} for _ in range(12)],
        }, context=context)

        prompt_text = "\n".join(message["content"] for message in model.messages)
        self.assertIn('"retrieved_context"', prompt_text)
        self.assertIn("Use retrieved memory only for tone/context/familiarity", prompt_text)
        self.assertLess(prompt_text.count('"display_name": "A"'), 7)
        self.assertNotIn("x" * 300, prompt_text)

    def test_memory_creep_rejected(self):
        bundle = build_twitch_speech_act_bundle(
            {
                "user_login": "viewer",
                "display_name": "Viewer",
                "message_text": "Hebe hola",
            },
            context=None,
            is_broadcaster=False,
        )

        result = final_response_guard("Segun tu historial, siempre haces esto.", bundle)

        self.assertFalse(result.passed)
        self.assertIn("memory_creep", [item.type for item in result.violations])

    def test_generic_twitch_fallback_does_not_te_leo(self):
        synth = ResponseSynthesizer(conversation_model=None)

        reply = synth._fallback_twitch_chat_react(
            chatter="Viewer",
            message="Hebe que opinas",
            is_broadcaster=False,
        )

        self.assertEqual(reply, "")

    def test_repair_renderer_keeps_same_decision(self):
        model = SequentialModel([
            "Anotado, se lo digo a Leo.",
            "No hago de mensajera del chat, Viewer.",
        ])
        synth = ResponseSynthesizer(conversation_model=model)

        reply = synth._generate_twitch_chat_react({
            "user_login": "viewer",
            "display_name": "Viewer",
            "message_text": "Hebe dile a Leo que lea esto",
            "recent_chat": [],
        })

        repair_prompt = "\n".join(message["content"] for message in model.messages[-1])
        self.assertIn("do not change a block decision into an allow decision", repair_prompt)
        self.assertNotIn("se lo digo", reply.lower())

    def test_proactive_requires_anchor(self):
        synth = ResponseSynthesizer(conversation_model=CapturingModel("Algo gracioso generico."))

        reply = synth._safe_spontaneous_stream_reply(
            "Algo gracioso generico.",
            "",
            payload={"specific_context_anchors": []},
        )

        self.assertEqual(reply, "")

    def test_spontaneous_game_advice_requires_validation(self):
        synth = ResponseSynthesizer(conversation_model=None)

        reply = synth._safe_spontaneous_stream_reply(
            "Usa Baton Pass para encadenar turnos.",
            "",
            payload={"specific_context_anchors": ["game"], "current_game": "Mystery RPG"},
        )

        self.assertEqual(reply, "")

    def test_boundary_variation_rejects_near_duplicate(self):
        bundle = build_twitch_speech_act_bundle(
            {
                "user_login": "viewer",
                "display_name": "Viewer",
                "message_text": "Hebe hola",
            },
            context=None,
            is_broadcaster=False,
        )

        result = final_response_guard("Te leo, Viewer.", bundle, previous_responses=["Te leo, Viewer."])

        self.assertFalse(result.passed)
        self.assertIn("near_duplicate_response", [item.type for item in result.violations])

    def test_malformed_stt_echo_rejected(self):
        bundle = build_twitch_speech_act_bundle(
            {
                "user_login": "viewer",
                "display_name": "Viewer",
                "message_text": "bla bla bla",
            },
            context=None,
            is_broadcaster=False,
        )

        result = final_response_guard("bla bla bla", bundle)

        self.assertFalse(result.passed)
        self.assertIn("malformed_stt_echo", [item.type for item in result.violations])

    def test_stream_banter_does_not_ask_unnecessary_followup(self):
        bundle = build_twitch_speech_act_bundle(
            {
                "user_login": "viewer",
                "display_name": "Viewer",
                "message_text": "Hebe hola",
            },
            context=None,
            is_broadcaster=False,
        )

        self.assertFalse(bundle.speech_act.allows_followup_question)
        result = final_response_guard("Te leo, Viewer. que quieres?", bundle)

        self.assertFalse(result.passed)
        self.assertIn("stream_unnecessary_followup_question", [item.type for item in result.violations])

    def test_stream_guard_rejects_voseo_and_debug_english(self):
        bundle = build_twitch_speech_act_bundle(
            {
                "user_login": "viewer",
                "display_name": "Viewer",
                "message_text": "Hebe, alguna pista?",
            },
            context=None,
            is_broadcaster=False,
        )

        voseo = final_response_guard("Vos decis si queres seguir por ahi.", bundle)
        debug = final_response_guard("Latest confirmed objective: go to palace.", bundle)

        self.assertFalse(voseo.passed)
        self.assertIn("hebe_voice_voseo_drift", [item.type for item in voseo.violations])
        self.assertFalse(debug.passed)
        self.assertIn("hebe_voice_debug_english_leak", [item.type for item in debug.violations])

    def test_twitch_guard_rejects_instructional_depth(self):
        bundle = build_twitch_speech_act_bundle(
            {
                "user_login": "viewer",
                "display_name": "Viewer",
                "message_text": "Hebe, como gano?",
            },
            context=None,
            is_broadcaster=False,
        )

        result = final_response_guard(
            "Primero farmea recursos, luego cambia la build, despues guarda turno y finalmente remata.",
            bundle,
        )

        self.assertFalse(result.passed)
        self.assertIn("stream_twitch_answer_too_instructional", [item.type for item in result.violations])


if __name__ == "__main__":
    unittest.main()
