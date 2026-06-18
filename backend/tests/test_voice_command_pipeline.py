import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.cognitive.input_event import InputEvent
from app.cognitive.action_runtime import ActionRuntime
from app.cognitive.models import DeliberationResult, ExecutionResult, Plan, PlanStep, StepExecutionResult
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.cognitive.wake_name_resolver import WakeNameResolver
from app.core.state import HebeState
from app.hebe_engine import HebeEngine
from app.integrations.twitch.chat_cache import TwitchChatCache
from app.integrations.twitch.target_resolver import TwitchTargetResolver
from app.services.voice_command_recovery import normalize_stt_transcript
from app.stream.action_planner import StreamActionPlanner
from app.stream.state import StreamSessionState


class FakeTwitch:
    def __init__(self):
        self.sent = []
        self.channel_name = "leonifelheim"
        self.bot_username = "HebeNifelheim"
        self.shoutout_command_template = "!so {username}"
        self.chat_cache = TwitchChatCache()
        self.target_resolver = TwitchTargetResolver(self.chat_cache, event_memory=None, aliases={})

    def is_available(self):
        return True

    def normalize_twitch_username(self, username):
        return str(username or "").strip().lstrip("@").replace(" ", "")

    def build_shoutout_command(self, username):
        return self.shoutout_command_template.format(username=self.normalize_twitch_username(username))

    def send_message(self, text):
        self.sent.append(text)
        return True

    def shoutout(self, username):
        self.sent.append(self.build_shoutout_command(username))
        return True

    def resolve_user(self, raw_target):
        return self.target_resolver.resolve_user(raw_target)

    def resolve_user_details(self, raw_target, intent=""):
        return self.target_resolver.resolve_user_details(raw_target, intent=intent)

    def remember_user_alias(self, alias, username):
        return self.target_resolver.remember_alias(alias, username)

    def remember_chat_message(self, *, username, display_name="", text=""):
        self.chat_cache.add_message(username=username, display_name=display_name, text=text)


class FakeSynth:
    def __init__(self):
        self.results = []

    def synthesize_command_result(self, result, **kwargs):
        self.results.append((result, kwargs))
        return f"modelo:{result.action_type}:{result.state_changes.get('target', result.state_changes.get('app_id', ''))}"


class FakeWin:
    def __init__(self, ok=True):
        self.ok = ok
        self.opened = []

    def open_app(self, app):
        self.opened.append(app)
        return self.ok


class RetrySTT:
    def __init__(self, retry_text, *, speech_detected=True):
        self.last_speech_detected = speech_detected
        self.retry_text = retry_text
        self.calls = []

    def retry_last_command_transcript(self, *, language=None):
        self.calls.append({"language": language})
        return {"text": self.retry_text, "attempted": True, "speech_detected": self.last_speech_detected}


class FakeContextBuilder:
    def __init__(self):
        self.inputs = []

    def build(self, state, input_text=None, internal_event=None):
        self.inputs.append(input_text)
        return SimpleNamespace(
            input_text=input_text,
            internal_event=internal_event,
            state_snapshot={},
            relevant_facts=[],
            relevant_chunks=[],
            conversation_history=[],
            message_type="small_talk",
            context_policy={"memory": "limited"},
            resolved_entities=[],
        )


class FakeDeliberationService:
    def deliberate(self, context):
        return DeliberationResult(
            plan=Plan(
                steps=[PlanStep(type="reply", data={"mode": "chat"})],
                reasoning="Fallback chat",
            )
        )


class FakePlanExecutor:
    def execute(self, plan):
        return ExecutionResult([StepExecutionResult(step_type="reply", success=True, data={"mode": "chat"})])


class FixedResponseSynth:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def synthesize(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply

    def synthesize_command_result(self, result, **kwargs):
        return f"modelo:{result.action_type}:{result.state_changes.get('target', result.state_changes.get('app_id', ''))}"


def make_engine(chatters=None, *, live=True):
    stream = StreamSessionState(enabled=True, presence_mode="reactive")
    stream.is_live = live
    stream.live_status_known = True
    stream.recent_chat_messages = [
        {"username": name, "display_name": name.capitalize(), "text": "hola", "ts": 1.0}
        for name in (chatters or ["nuria", "charlie", "totodile", "alguien_del_chat"])
    ]
    stream.recent_active_users = list(chatters or ["nuria", "charlie", "totodile", "alguien_del_chat"])
    engine = HebeEngine.__new__(HebeEngine)
    engine.runtime = SimpleNamespace(
        state=SimpleNamespace(stream=stream, hebe_sleeping=False, mode="active", pending_clarification=None),
        twitch=FakeTwitch(),
        twitch_chat_bot=SimpleNamespace(is_connected=True),
        speak=Mock(),
        stt=None,
        win=FakeWin(),
    )
    engine.response_synthesizer = FakeSynth()
    engine.action_runtime = ActionRuntime(engine.runtime)
    engine.voice_command_confirm_ambiguous = True
    engine.shoutout_cooldown_seconds = 120
    engine.shoutout_allow_bots = False
    engine.shoutout_blocked_users = {"hebenifelheim", "jotunbot", "streamelements", "nightbot"}
    engine._manual_reply_ui_only = False
    engine._current_input_event = None
    engine.spontaneous_twitch_chat_enabled = False
    engine.stream_action_planner = engine._build_stream_action_planner()
    return engine


def pending_marker(expected="casual_answer"):
    return SimpleNamespace(last_opens_conversation_turn=True, last_expected_reply_type=expected)


class VoiceCommandPipelineTests(unittest.TestCase):
    def test_start_awake_default_initializes_not_sleeping(self):
        runtime = SimpleNamespace(
            state=HebeState(),
            llm=None,
            intent_llm=None,
            twitch=FakeTwitch(),
            twitch_chat_bot=None,
            twitch_events=None,
            speak=Mock(),
        )

        engine = HebeEngine(runtime=runtime, use_wakeword=True)

        self.assertTrue(engine.start_awake)
        self.assertFalse(runtime.state.hebe_sleeping)
        self.assertEqual(runtime.state.mode, "active")

    def test_stt_normalization_only_does_not_execute(self):
        result = normalize_stt_transcript("ebe az promo anuria", known_targets=["nuria"])

        self.assertEqual(result.normalized_text, "hebe haz promo a nuria")
        self.assertTrue(result.metadata["normalization_only"])
        self.assertFalse(hasattr(result, "action_type"))

    def test_ui_hebe_abre_obs_creates_open_application_action_plan(self):
        engine = make_engine()
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": r"C:\Tools\OBS\obs64.exe"}):
            result = engine.cognitive_flow("hebe abre obs", source="ui")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.win.opened[0]["app_id"], "obs")
        self.assertEqual(delivered, [("ui", "modelo:open_application:obs")])
        synth_result = engine.response_synthesizer.results[-1][0]
        self.assertEqual(synth_result.action_type, "open_application")
        self.assertTrue(synth_result.success)

    def test_ui_abre_obs_creates_open_application_when_awake_and_whitelisted(self):
        engine = make_engine()
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": r"C:\Tools\OBS\obs64.exe"}):
            result = engine.cognitive_flow("abre obs", source="ui")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.win.opened[0]["app_id"], "obs")
        self.assertEqual(delivered, [("ui", "modelo:open_application:obs")])

    def test_stt_hebe_abre_obs_uses_same_open_application_pipeline(self):
        engine = make_engine()
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": r"C:\Tools\OBS\obs64.exe"}):
            result = engine._process_stt_voice_transcript("Hebe abre OBS")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.win.opened[0]["app_id"], "obs")
        self.assertEqual(delivered, [("stt_voice", "modelo:open_application:obs")])

    def test_obs_path_missing_returns_structured_action_result_not_generic_advice(self):
        engine = make_engine()
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": ""}, clear=False), \
             patch("pathlib.Path.exists", lambda self: False):
            result = engine.cognitive_flow("hebe abre obs", source="ui")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.win.opened, [])
        synth_result = engine.response_synthesizer.results[-1][0]
        self.assertEqual(synth_result.action_type, "open_application")
        self.assertFalse(synth_result.success)
        self.assertEqual(synth_result.metadata["error_code"], "app_path_missing")
        self.assertIn("HEBE_APP_OBS_PATH", synth_result.fallback_text)

    def test_unknown_app_does_not_execute_or_call_command_synth(self):
        engine = make_engine()

        result = engine._plan_and_execute_local_app_action("hebe abre paint raro", "ui")

        self.assertIsNone(result)
        self.assertEqual(engine.runtime.win.opened, [])
        self.assertEqual(engine.response_synthesizer.results, [])

    def test_non_whitelisted_app_does_not_execute_even_with_command_words(self):
        engine = make_engine()

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": r"C:\Tools\OBS\obs64.exe"}):
            result = engine._plan_and_execute_local_app_action("abre calculadora", "ui")

        self.assertIsNone(result)
        self.assertEqual(engine.runtime.win.opened, [])

    def test_model_is_not_called_before_open_application_action_plan(self):
        class GuardSynth(FakeSynth):
            def synthesize_command_result(self, result, **kwargs):
                self.results.append((result, kwargs))
                assert result.action_type == "open_application"
                assert result.state_changes.get("app_id")
                return "ok"

        engine = make_engine()
        engine.response_synthesizer = GuardSynth()
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": r"C:\Tools\OBS\obs64.exe"}):
            result = engine.cognitive_flow("hebe inicia obs", source="ui")

        self.assertEqual(result, "continue")
        self.assertEqual(delivered, [("ui", "ok")])

    def test_planner_detects_generic_shoutout_phrasings(self):
        engine = make_engine()
        planner = engine._get_stream_action_planner()
        cases = {
            "Hebe haz promo a Nuria": "Nuria",
            "Haz una promo a Charlie": "Charlie",
            "Dale SO a Totodile": "Totodile",
            "Promociona a alguien_del_chat": "alguien_del_chat",
            "Shoutout to randomUser": "randomUser",
        }
        for text, target in cases.items():
            with self.subTest(text=text):
                plan = planner.plan(InputEvent(source="typed_ui", raw_text=text, normalized_text=text))
                self.assertIsNotNone(plan)
                self.assertEqual(plan.action_type, "twitch_shoutout")
                self.assertEqual(plan.status, "complete")
                self.assertEqual(plan.target, target)

    def test_noisy_stt_enters_planner_as_normalized_candidate(self):
        engine = make_engine(["nuria"])
        normalization = normalize_stt_transcript("ebe az promo anuria", known_targets=engine._known_voice_command_targets())
        plan = engine._get_stream_action_planner().plan(
            InputEvent(source="stt_voice", raw_text=normalization.raw_text, normalized_text=normalization.normalized_text, is_voice=True)
        )

        self.assertEqual(normalization.normalized_text, "hebe haz promo a nuria")
        self.assertEqual(plan.action_type, "twitch_shoutout")
        self.assertEqual(plan.status, "complete")
        self.assertEqual(plan.target, "nuria")

    def test_action_executor_sends_shoutout_and_result_goes_to_synthesizer(self):
        engine = make_engine(["nuria"])
        engine._current_input_event = InputEvent(
            source="stt_voice",
            raw_text="ebe az promo anuria",
            normalized_text="hebe haz promo a nuria",
            is_voice=True,
        )

        result = engine._handle_stream_manual_command("hebe haz promo a nuria")
        reply = engine._synthesize_command_result(result, input_text="hebe haz promo a nuria")

        self.assertEqual(result.action_type, "twitch_shoutout")
        self.assertTrue(result.success)
        self.assertEqual(engine.runtime.twitch.sent, ["!so nuria"])
        self.assertEqual(reply, "modelo:twitch_shoutout:nuria")

    def test_recent_chatter_can_resolve_spoken_promo_target(self):
        engine = make_engine([])

        engine.observe_twitch_chat_message("er_tito_xarly", "er_tito_xarly", "leo^^", "#chan")
        plan = engine._get_stream_action_planner().plan(
            InputEvent(source="stt_voice", raw_text="hazle una promo a Charlie", normalized_text="hazle una promo a Charlie", is_voice=True)
        )

        self.assertEqual(plan.action_type, "twitch_shoutout")
        self.assertEqual(plan.target, "er_tito_xarly")
        self.assertEqual(plan.status, "complete")

    def test_manual_alias_resolves_spoken_promo_target(self):
        engine = make_engine([])

        alias_result = engine._handle_stream_manual_command("Hebe, Charlie es er_tito_xarly")
        plan = engine._get_stream_action_planner().plan(
            InputEvent(source="stt_voice", raw_text="hazle una promo a Charlie", normalized_text="hazle una promo a Charlie", is_voice=True)
        )

        self.assertEqual(alias_result.action_type, "chatter_alias_stored")
        self.assertTrue(alias_result.success)
        self.assertEqual(plan.action_type, "twitch_shoutout")
        self.assertEqual(plan.target, "er_tito_xarly")

    def test_missing_target_asks_for_followup_without_executing(self):
        engine = make_engine()

        result = engine._handle_stream_manual_command("Hebe haz SO")

        self.assertEqual(result.action_type, "twitch_shoutout_clarify")
        self.assertFalse(result.success)
        self.assertEqual(engine.runtime.twitch.sent, [])
        self.assertIn("A quién", result.fallback_text)

    def test_ambiguous_target_asks_for_clarification(self):
        engine = make_engine(["nuria", "muria"])

        result = engine._handle_stream_manual_command("Hebe haz promo a uria")

        self.assertEqual(result.action_type, "twitch_shoutout_clarify")
        self.assertFalse(result.success)
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_offline_context_is_visible_in_action_plan(self):
        engine = make_engine(["nuria"], live=False)
        plan = engine._get_stream_action_planner().plan(
            InputEvent(source="typed_ui", raw_text="Hebe haz promo a Nuria", normalized_text="Hebe haz promo a Nuria")
        )

        self.assertEqual(plan.action_type, "twitch_shoutout")
        self.assertFalse(plan.context_checks["is_live"])
        self.assertEqual(plan.status, "complete")

    def test_pipeline_logs_full_flow(self):
        engine = make_engine(["nuria"])
        logs = []
        emits = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emits.append((event_type, data or {}))):
            normalization = engine._normalize_stt_input("ebe az promo anuria")
            engine._current_input_event = engine._build_input_event(
                source="stt_voice",
                raw_text=normalization.raw_text,
                normalized_text=normalization.normalized_text,
                stt_metadata=normalization.as_event(),
            )
            result = engine._handle_stream_manual_command(normalization.normalized_text)
            engine._synthesize_command_result(result, input_text=normalization.normalized_text)

        joined = "\n".join(logs)
        self.assertIn("[HEBE][INPUT]", joined)
        self.assertIn("[HEBE][STT][NORMALIZED]", joined)
        self.assertIn("[HEBE][COGNITION]", joined)
        self.assertIn("[HEBE][ACTION_PLAN]", joined)
        self.assertIn("[HEBE][ACTION_EXECUTOR]", joined)
        self.assertIn("[HEBE][RESPONSE_SYNTH]", joined)
        self.assertTrue(any(event_type == "voice.command" for event_type, _ in emits))

    def test_stt_ambient_command_uses_input_event_action_plan_executor(self):
        engine = make_engine(["nuria"])
        engine.stream_ambient_stt_enabled = True
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            normalization = engine._normalize_stt_input("Hebe, desactiva STT ambiental")
            engine._current_input_event = engine._build_input_event(
                source="stt_voice",
                raw_text=normalization.raw_text,
                normalized_text=normalization.normalized_text,
                stt_metadata=normalization.as_event(),
            )
            result = engine._handle_stream_manual_command(normalization.normalized_text)
            reply = engine._synthesize_command_result(result, input_text=normalization.normalized_text)

        joined = "\n".join(logs)
        self.assertEqual(result.action_type, "stream_ambient_stt_disabled")
        self.assertFalse(engine.stream_ambient_stt_enabled)
        self.assertEqual(reply, "modelo:stream_ambient_stt_disabled:")
        self.assertIn("[HEBE][INPUT] source=stt_voice", joined)
        self.assertIn("[HEBE][COGNITION]", joined)
        self.assertIn("[HEBE][ACTION_PLAN] action_type=stream_ambient_stt_disabled", joined)
        self.assertIn("[HEBE][ACTION_EXECUTOR] executing action_type=stream_ambient_stt_disabled", joined)

    def test_stt_command_enters_cognitive_flow_and_executes_action_plan(self):
        engine = make_engine(["nuria"])
        engine.stream_ambient_stt_enabled = True
        delivered = []
        logs = []
        normalization = engine._normalize_stt_input("Hebe, desactiva STT ambiental")
        engine._current_input_event = engine._build_input_event(
            source="stt_voice",
            raw_text=normalization.raw_text,
            normalized_text=normalization.normalized_text,
            stt_metadata=normalization.as_event(),
        )
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            result = engine.cognitive_flow(normalization.normalized_text, source="stt_voice")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertFalse(engine.stream_ambient_stt_enabled)
        self.assertEqual(delivered, [("stt_voice", "modelo:stream_ambient_stt_disabled:")])
        self.assertIn("[HEBE][COG] incoming source='stt_voice'", joined)
        self.assertIn("[HEBE][ACTION_PLAN] action_type=stream_ambient_stt_disabled", joined)
        self.assertIn("[HEBE][ACTION_EXECUTOR] executing action_type=stream_ambient_stt_disabled", joined)
        self.assertIn("[HEBE][RESPONSE_SYNTH]", joined)

    def test_stt_worker_transcript_process_helper_routes_to_cognition(self):
        engine = make_engine(["nuria"])
        engine.stream_ambient_stt_enabled = True
        delivered = []
        logs = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            result = engine._process_stt_voice_transcript("Hebe, desactiva STT ambiental")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertFalse(engine.stream_ambient_stt_enabled)
        self.assertEqual(delivered, [("stt_voice", "modelo:stream_ambient_stt_disabled:")])
        self.assertIn("[HEBE][INPUT] source=stt_voice raw='Hebe, desactiva STT ambiental'", joined)
        self.assertIn("[HEBE][STT][NORMALIZED] raw='Hebe, desactiva STT ambiental' normalized='hebe desactiva stt ambiental'", joined)
        self.assertIn("[HEBE][COG] incoming source='stt_voice'", joined)
        self.assertIn("[HEBE][COG] decision=command", joined)
        self.assertIn("[HEBE][ACTION_PLAN] action_type=stream_ambient_stt_disabled", joined)
        self.assertIn("[HEBE][ACTION_EXECUTOR] success=true", joined)
        self.assertIn("[HEBE][RESPONSE_SYNTH]", joined)

    def test_unsupported_script_transcript_does_not_enter_cognition_without_retry_audio(self):
        engine = make_engine(["nuria"])
        handled = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            result = engine._process_stt_voice_transcript("à¤¯à¤¬ à¤†à¤¬à¤°à¥‡ à¤…à¤¬à¥‡ à¤¯à¤¶à¥‡")

        self.assertEqual(result, "continue")
        self.assertEqual(handled, [])
        self.assertIn("reason=unsupported_script script=devanagari", "\n".join(logs))

    def test_repeated_hotword_prompt_transcript_is_rejected_before_input_event(self):
        engine = make_engine(["nuria"])
        bad = "Hebe, Ebe, Ebe, Zwei, Persona, Final Fantasy."
        handled = []
        emitted = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"

        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))), \
             patch("app.hebe_engine.log_chat") as log_chat_mock:
            result = engine._process_stt_voice_transcript(bad)

        self.assertEqual(result, "continue")
        self.assertEqual(handled, [])
        self.assertIsNone(engine._current_input_event)
        log_chat_mock.assert_not_called()
        rejected = [data for event_type, data in emitted if event_type == "voice.command" and data.get("status") == "rejected"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[-1]["reason"], "stt_prompt_echo_or_hotword_list")
        self.assertEqual(rejected[-1]["raw_text"], bad)
        self.assertFalse(any(event_type == "chat.user" for event_type, _ in emitted))

    def test_known_name_prompt_list_transcript_is_rejected_before_cognition(self):
        engine = make_engine(["xarly", "totodile", "charlie", "zwei"])
        bad = "Hebe, Ebe, Xarly, Totodile, Charlie, Zwei, Persona, Final Fantasy."
        handled = []
        emitted = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"

        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))), \
             patch("app.hebe_engine.log_chat") as log_chat_mock:
            result = engine._process_stt_voice_transcript(bad)

        self.assertEqual(result, "continue")
        self.assertEqual(handled, [])
        self.assertIsNone(engine._current_input_event)
        log_chat_mock.assert_not_called()
        rejected = [data for event_type, data in emitted if event_type == "voice.command" and data.get("status") == "rejected"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[-1]["reason"], "stt_prompt_echo_or_hotword_list")
        self.assertFalse(any(event_type == "chat.user" for event_type, _ in emitted))

    def test_rejected_stt_artifact_never_enters_main_conversation_or_cognition(self):
        engine = make_engine(["xarly", "totodile", "charlie", "zwei"])
        bad = "Xarly, Xarly, Zwei, Totodile..."
        handled = []
        emitted = []
        logs = []
        delivered = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"
        engine._deliver_voice_reply = lambda text: delivered.append(text)
        engine.stt_log_rejected_raw = False

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))), \
             patch("app.hebe_engine.log_chat") as log_chat_mock:
            result = engine._process_stt_voice_transcript(bad)

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(handled, [])
        self.assertEqual(delivered, [])
        self.assertIsNone(engine._current_input_event)
        log_chat_mock.assert_not_called()
        self.assertIn("[HEBE][STT][REJECTED] reason=stt_prompt_echo_or_hotword_list", joined)
        self.assertNotIn("[HEBE][STT][RAW]", joined)
        self.assertNotIn("[HEBE][INPUT]", joined)
        self.assertNotIn("[HEBE][COG]", joined)
        self.assertNotIn("[HEBE][JARVIS][CHAT]", joined)
        self.assertFalse(any(event_type == "chat.user" for event_type, _ in emitted))
        self.assertTrue(any(event_type == "voice.command" and data.get("status") == "rejected" for event_type, data in emitted))

    def test_repeated_engine_prompt_echo_disables_stt_prompt_for_session(self):
        class PromptSwitch:
            def __init__(self):
                self.disabled = 0

            def disable_command_prompt_for_session(self):
                self.disabled += 1
                return True

        engine = make_engine(["xarly", "totodile", "zwei"])
        engine.runtime.stt = PromptSwitch()
        engine.stt_prompt_echo_window_seconds = 300
        engine.stt_prompt_echo_disable_threshold = 2
        engine.stt_auto_disable_prompt_on_echo = True
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), \
             patch("app.hebe_engine.log_chat"):
            first = engine._process_stt_voice_transcript("Xarly, Xarly, Zwei, Totodile...")
            second = engine._process_stt_voice_transcript("Hebe, Ebe, Xarly, Totodile, Persona, Final Fantasy.")

        self.assertEqual(first, "continue")
        self.assertEqual(second, "continue")
        self.assertEqual(engine.runtime.stt.disabled, 1)
        self.assertIn("[HEBE][STT][PROMPT] auto_disabled reason=repeated_prompt_echo", "\n".join(logs))

    def test_hotword_only_transcript_does_not_trigger_chat_fallback_or_memory(self):
        engine = make_engine(["nuria"])
        bad = "Hebe, Ebe, Ebe, Zwei, Persona, Final Fantasy."
        engine.memory_extractor = Mock()
        handled = []
        emitted = []
        logs = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            result = engine._process_stt_voice_transcript(bad)

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(handled, [])
        engine.memory_extractor.extract_and_store.assert_not_called()
        self.assertNotIn("[HEBE][INPUT]", joined)
        self.assertNotIn("[HEBE][COG] incoming", joined)
        self.assertNotIn("[HEBE][JARVIS][CHAT]", joined)
        self.assertFalse(any(event_type == "chat.user" for event_type, _ in emitted))

    def test_recent_tts_text_heard_by_stt_is_rejected_as_self_echo(self):
        engine = make_engine(["nuria"])
        engine.stt_tts_echo_window_seconds = 10
        engine.stt_tts_echo_similarity_threshold = 0.82
        engine._remember_tts_text("Ya estoy aquÃ­, Leo.")
        handled = []
        emitted = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"

        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            result = engine._process_stt_voice_transcript("Ya estoy aquÃ­, Leo.")

        self.assertEqual(result, "continue")
        self.assertEqual(handled, [])
        rejected = [data for event_type, data in emitted if event_type == "voice.command" and data.get("status") == "rejected"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[-1]["reason"], "self_tts_echo")
        self.assertGreaterEqual(rejected[-1]["similarity"], 0.82)
        self.assertFalse(any(event_type == "chat.user" for event_type, _ in emitted))

    def test_tts_echo_variation_does_not_consume_followup_window(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = False
        engine.runtime.state.pending_conversation_turn = {
            "expected_type": "casual_answer",
            "previous_assistant_message_id": "assistant-test",
            "previous_assistant_message": "Pues bien, dormida a ratos pero lista para tus locuras, cabron.",
            "created_at": 1.0,
            "expires_at": time.time() + 30,
            "source": "assistant_question",
            "allowed_sources": ["stt_voice", "ui"],
            "allow_without_wakeword": True,
            "status": "pending",
            "followups_used": 0,
            "max_followups": 1,
            "reply_source": "stt_voice",
        }
        engine.stt_tts_echo_window_seconds = 10
        engine.stt_tts_echo_similarity_threshold = 0.82
        engine._remember_tts_text("Pues bien, dormida a ratos pero lista para tus locuras, cabron.")
        handled = []
        emitted = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"

        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            result = engine._process_stt_voice_transcript("Pues bien, dormia ratos, pero lista para tus locuras, cabron.")

        self.assertEqual(result, "continue")
        self.assertEqual(handled, [])
        self.assertEqual(engine.runtime.state.pending_conversation_turn["status"], "pending")
        self.assertEqual(engine.runtime.state.pending_conversation_turn["followups_used"], 0)
        rejected = [data for event_type, data in emitted if event_type == "voice.command" and data.get("status") == "rejected"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[-1]["reason"], "self_tts_echo")
        self.assertFalse(any(event_type == "chat.user" for event_type, _ in emitted))

    def test_user_speech_is_allowed_during_tts_when_not_echo(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = False
        engine.runtime.tts = SimpleNamespace(is_speaking=True)
        engine.stt_ignore_while_tts_speaking = True
        engine._remember_tts_text("Ya estoy aqui, Leo.")
        handled = []
        emitted = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"

        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            result = engine._process_stt_voice_transcript("Hebe, como estas?")

        self.assertEqual(result, "continue")
        self.assertEqual(handled, [("stt_voice", "hebe como estas")])
        rejected = [data for event_type, data in emitted if event_type == "voice.command" and data.get("status") == "rejected"]
        self.assertFalse(rejected)
        self.assertTrue(any(event_type == "chat.user" for event_type, _ in emitted))

    def test_valid_stt_question_still_enters_cognition_and_user_bubble(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = False
        handled = []
        emitted = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"

        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))), \
             patch("app.hebe_engine.log_chat") as log_chat_mock:
            result = engine._process_stt_voice_transcript("Hebe, como estas?")

        self.assertEqual(result, "continue")
        self.assertEqual(handled, [("stt_voice", "hebe como estas")])
        log_chat_mock.assert_called_once()
        self.assertTrue(any(event_type == "chat.user" and data.get("text") == "Hebe, como estas?" for event_type, data in emitted))

    def test_eve_stt_question_enters_direct_cognition_and_stays_local(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = True
        engine.runtime.state.tts_enabled = True
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("Hoy toca seguir la run.")
        engine.memory_extractor = Mock()
        delivered = []
        logs = []
        engine._deliver_voice_reply = lambda text, **kwargs: delivered.append((text, kwargs))

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), \
             patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("Eve, que toca hoy?")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(delivered[0][0], "modelo:stream_schedule_lookup:")
        self.assertIn("[HEBE][RESPONSE_DECISION] should_reply=true reason=direct_question", joined)
        self.assertIn("input_type=direct_stt output_target=local_ui+stream_tts", joined)
        self.assertNotIn("output_target=twitch_chat", joined)

    def test_hola_eve_como_estas_is_addressed_to_hebe(self):
        resolver = WakeNameResolver()

        result = resolver.resolve(
            raw_text="Hola, Eve, como estas?",
            normalized_text="hola eve como estas",
            source="stt_voice",
            command_markers={"abre", "haz", "pon"},
        )

        self.assertTrue(result.addressed_to_hebe)
        self.assertEqual(result.matched_name, "eve")
        self.assertEqual(result.canonical, "hebe")
        self.assertEqual(result.reason, "stt_alias_vocative")
        self.assertGreaterEqual(result.confidence, 0.78)

    def test_assistant_question_creates_pending_conversation_turn(self):
        engine = make_engine(["nuria"])
        engine.pending_conversation_ttl_seconds = 120

        engine._record_assistant_reply_for_conversation("AquÃ­ sobreviviendo. Â¿tÃº quÃ© tal?", source="stt_voice", synthesizer=pending_marker())

        turn = engine.runtime.state.pending_conversation_turn
        self.assertEqual(turn["source"], "assistant_question")
        self.assertEqual(turn["expected_type"], "casual_answer")
        self.assertEqual(turn["allowed_sources"], ["stt_voice", "ui"])
        self.assertFalse(turn.get("requires_wakeword", False))

    def test_stt_followup_after_assistant_question_enters_conversation_without_wakeword(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = False
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("Eso me vale, Leo.")
        engine.memory_extractor = Mock()
        delivered = []
        emitted = []
        logs = []
        engine._deliver_voice_reply = lambda text: delivered.append(text)
        engine._record_assistant_reply_for_conversation("Â¿tÃº quÃ© tal?", source="stt_voice", synthesizer=pending_marker())

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))), \
             patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("Yo bien, sorprendido por tu respuesta.")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(delivered, ["Eso me vale, Leo."])
        self.assertEqual(engine.context_builder.inputs, ["yo bien sorprendido por tu respuesta"])
        self.assertIn("[HEBE][CONVERSATION] pending_turn matched source=stt_voice", joined)
        self.assertIn("[HEBE][COG] decision=conversation_followup", joined)
        self.assertNotIn("reason=not_direct_command", joined)
        self.assertTrue(any(event_type == "chat.user" for event_type, _ in emitted))

    def test_pending_conversation_does_not_capture_filler_mumble(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = False
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("No deberia sonar.")
        engine.memory_extractor = Mock()
        delivered = []
        emitted = []
        logs = []
        engine._deliver_voice_reply = lambda text: delivered.append(text)
        engine._record_assistant_reply_for_conversation("Â¿Quieres que responda al chat o genero una lÃ­nea?", source="stt_voice", synthesizer=pending_marker("clarification"))

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))), \
             patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("Mmm...")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(delivered, [])
        self.assertEqual(engine.context_builder.inputs, [])
        self.assertNotIn("decision=conversation_followup", joined)
        self.assertFalse(any(event_type == "chat.user" for event_type, _ in emitted))

    def test_ambient_stt_without_wakeword_does_not_enter_jarvis_chat(self):
        engine = make_engine(["ciber"])
        engine.runtime.state.stream.enabled = True
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("No deberia sonar.")
        delivered = []
        emitted = []
        logs = []
        engine._deliver_voice_reply = lambda text: delivered.append(text)

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))), \
             patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("10 no es nada personal, Ciber")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(delivered, [])
        self.assertEqual(engine.context_builder.inputs, [])
        self.assertIn("[HEBE][STT_GATE] ambient_only reason=no_wake_no_valid_pending", joined)
        self.assertIn("input_type=ambient_stt output_target=silent_context_update", joined)
        self.assertNotIn("[HEBE][JARVIS][CHAT]", joined)
        self.assertFalse(any(event_type == "chat.user" for event_type, _ in emitted))

    def test_ambient_progress_updates_session_context_without_reply(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = True
        engine.context_builder = FakeContextBuilder()
        delivered = []
        emitted = []
        logs = []
        engine._deliver_voice_reply = lambda text: delivered.append(text)

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))), \
             patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("Ahora toca salir de la ciudad vieja")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(delivered, [])
        self.assertEqual(engine.context_builder.inputs, [])
        self.assertEqual(engine.runtime.state.stream.current_run_objective, "ahora toca salir de la ciudad vieja")
        self.assertIn("[HEBE][INPUT_CLASSIFY] source=stt_voice input_type=ambient_stream_context", joined)
        self.assertIn("[HEBE][CONTEXT_RELEVANCE] useful=true", joined)
        self.assertIn("[HEBE][SESSION_CONTEXT] updated=true", joined)
        self.assertIn("[HEBE][RESPONSE_DECISION] should_reply=false reason=no_context_only", joined)
        self.assertFalse(any(event_type == "chat.user" for event_type, _ in emitted))

    def test_unknown_game_terms_are_stored_as_leo_context_without_invention(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = True
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), \
             patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("Hemos pasado el palacio raro de Kamoshida")

        self.assertEqual(result, "continue")
        facts = engine.runtime.state.stream.recent_run_context_facts
        self.assertTrue(any("kamoshida" in str(fact.get("raw_text") or "").lower() for fact in facts))
        self.assertNotIn("chapter", "\n".join(logs).lower())

    def test_direct_stt_declares_local_targets_and_does_not_post_to_twitch_chat(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.tts_enabled = True
        engine.handle_command = lambda command, source="voice": "continue"
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), \
             patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("Hebe, que sabes de Persona 5 Royal?")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.twitch.sent, [])
        self.assertIn("input_type=direct_stt output_target=local_ui+stream_tts", joined)
        self.assertIn("[HEBE][INPUT_CLASSIFY] source=stt_voice input_type=explicit_question", joined)
        self.assertIn("[HEBE][RESPONSE_DECISION] should_reply=true reason=direct_question", joined)

    def test_direct_stt_can_post_to_twitch_only_via_stream_chat_action_plan(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.tts_enabled = False
        delivered = []
        logs = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("Hebe, dile al chat que vuelvo en 5 minutos")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.twitch.sent, ["vuelvo en 5 minutos"])
        self.assertIn("[HEBE][ACTION_PLAN] action_type=stream_chat_message target=twitch_chat", joined)
        self.assertIn("output_target=twitch_chat", joined)
        self.assertEqual(delivered, [("stt_voice", "modelo:stream_chat_message:twitch_chat")])

    def test_spontaneity_reply_is_voice_first_when_twitch_chat_disabled(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.tts_enabled = True
        engine.runtime.state.stream.policies.allow_tts_idle_prompts = True
        emitted = []
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            engine._deliver_twitch_reply(
                "Mira recursos antes de avanzar.",
                event_type="twitch_idle_prompt",
                payload={"idle_topic": "resource_management"},
            )

        joined = "\n".join(logs)
        self.assertEqual(engine.runtime.twitch.sent, [])
        engine.runtime.speak.assert_called_once_with("Mira recursos antes de avanzar.", emit_chat=False)
        self.assertIn("input_type=spontaneity output_target=stream_tts", joined)
        self.assertIn("skipped reason=twitch_spontaneous_disabled", joined)
        self.assertTrue(any(
            event_type == "chat.assistant"
            and data.get("source") == "spontaneity"
            and data.get("output_target") == "local_ui"
            for event_type, data in emitted
        ))

    def test_spontaneity_reply_posts_to_twitch_chat_when_enabled_and_anchored(self):
        engine = make_engine(["nuria"])
        engine.spontaneous_twitch_chat_enabled = True
        engine.runtime.state.tts_enabled = True
        engine.runtime.state.stream.last_spontaneous_twitch_chat_ts = 0.0
        emitted = []
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            engine._deliver_twitch_reply(
                "Mira recursos antes de avanzar.",
                event_type="twitch_idle_prompt",
                payload={
                    "idle_topic": "resource_management",
                    "specific_context_anchors": ["run_context"],
                    "used_fact_id": "fact-1",
                },
            )

        joined = "\n".join(logs)
        self.assertEqual(engine.runtime.twitch.sent, ["Mira recursos antes de avanzar."])
        engine.runtime.speak.assert_not_called()
        self.assertIn("input_type=spontaneity output_target=twitch_chat", joined)
        self.assertIn("reason=spontaneous_twitch_enabled", joined)
        self.assertIn("[HEBE][TWITCH][CHATBOT] send_message reason=spontaneity", joined)
        self.assertFalse(any(event_type == "chat.assistant" for event_type, _ in emitted))

    def test_twitch_mention_reply_posts_to_twitch_chat_and_no_private_pending_turn(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        engine.runtime.state.tts_enabled = False
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"):
            engine._deliver_twitch_reply(
                "Corta y al pie.",
                event_type="twitch_chat_react",
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe"},
            )
            engine._record_assistant_reply_for_conversation("¿tú qué tal?", source="twitch_chat_react", synthesizer=pending_marker())

        self.assertEqual(engine.runtime.twitch.sent, ["Corta y al pie."])
        self.assertFalse(hasattr(engine.runtime.state, "pending_conversation_turn"))
        self.assertIn("input_type=twitch_mention_or_event output_target=twitch_chat", "\n".join(logs))

    def test_ambient_game_commentary_does_not_enter_jarvis_unless_addressed(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = True
        engine.context_builder = FakeContextBuilder()
        delivered = []
        logs = []
        engine._deliver_voice_reply = lambda text: delivered.append(text)

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), \
             patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("me han pillado eso como si se les creyo")

        self.assertEqual(result, "continue")
        self.assertEqual(delivered, [])
        self.assertEqual(engine.context_builder.inputs, [])
        self.assertIn("[HEBE][STT_GATE] ambient_only reason=no_wake_no_valid_pending", "\n".join(logs))

    def test_pending_turn_not_created_for_twitch_source(self):
        engine = make_engine(["nuria"])

        engine._record_assistant_reply_for_conversation("¿tú qué tal?", source="twitch_chat_react", synthesizer=pending_marker())

        self.assertFalse(hasattr(engine.runtime.state, "pending_conversation_turn"))

    def test_pending_conversation_expires_after_ttl(self):
        engine = make_engine(["nuria"])
        engine.pending_conversation_ttl_seconds = 1

        with patch("time.time", return_value=1000.0):
            engine._record_assistant_reply_for_conversation("Â¿tÃº quÃ© tal?", source="stt_voice", synthesizer=pending_marker())
        with patch("time.time", return_value=1002.0):
            self.assertFalse(engine._pending_conversation_matches(source="stt_voice"))

        self.assertEqual(engine.runtime.state.pending_conversation_turn["status"], "expired")

    def test_unrelated_action_during_pending_conversation_still_uses_action_flow(self):
        engine = make_engine(["nuria"])
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))
        engine._record_assistant_reply_for_conversation("Â¿tÃº quÃ© tal?", source="stt_voice", synthesizer=pending_marker())

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": r"C:\Tools\OBS\obs64.exe"}), \
             patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("Hebe abre OBS")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.win.opened[0]["app_id"], "obs")
        self.assertEqual(delivered, [("stt_voice", "modelo:open_application:obs")])

    def test_duplicate_stt_followup_is_processed_only_once(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = False
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("Te sigo.")
        engine.memory_extractor = Mock()
        emitted = []
        engine._deliver_voice_reply = lambda text: None
        engine._record_assistant_reply_for_conversation("Â¿tÃº quÃ© tal?", source="stt_voice", synthesizer=pending_marker())

        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))), \
             patch("app.hebe_engine.log_chat"):
            first = engine._process_stt_voice_transcript("Yo bien, sorprendido por tu respuesta.")
            second = engine._process_stt_voice_transcript("Yo bien sorprendido por tu respuesta.")

        self.assertEqual(first, "continue")
        self.assertEqual(second, "continue")
        self.assertEqual(engine.context_builder.inputs, ["yo bien sorprendido por tu respuesta"])
        rejected = [data for event_type, data in emitted if event_type == "voice.command" and data.get("status") == "rejected"]
        self.assertEqual(rejected[-1]["reason"], "duplicate_recent_transcript")

    def test_response_tone_guard_removes_hostile_direct_insult_greeting(self):
        ctx = SimpleNamespace(message_type="small_talk")
        synth = ResponseSynthesizer(conversation_model=None)

        guarded = synth._guard_hostile_direct_insult_greeting(
            "Hija de puta, aquÃ­ sobreviviendo. Â¿tÃº quÃ© tal, jefe?",
            ctx,
        )

        self.assertNotIn("Hija de puta", guarded)
        self.assertIn("aquÃ­ sobreviviendo", guarded)

    def test_unsupported_script_retries_forced_spanish_and_routes_valid_latin_result(self):
        engine = make_engine(["nuria"])
        engine.runtime.stt = RetrySTT("Hebe abre OBS")
        routed = []
        engine.handle_command = lambda command, source="voice": routed.append((source, command, engine._current_input_event)) or "continue"
        emitted = []
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            result = engine._process_stt_voice_transcript("à¤¯à¤¬ à¤†à¤¬à¤°à¥‡ à¤…à¤¬à¥‡ à¤¯à¤¶à¥‡")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.stt.calls, [{"language": "es"}])
        self.assertEqual(routed[0][0], "stt_voice")
        self.assertEqual(routed[0][1], "abre obs")
        self.assertEqual(routed[0][2].raw_text, "à¤¯à¤¬ à¤†à¤¬à¤°à¥‡ à¤…à¤¬à¥‡ à¤¯à¤¶à¥‡")
        self.assertEqual(routed[0][2].normalized_text, "hebe abre obs")
        self.assertIn("[HEBE][STT][RETRY] reason=unsupported_script forcing_language=es", joined)
        self.assertIn("[HEBE][STT][RETRY_RESULT] raw='Hebe abre OBS' accepted=true", joined)
        debug_events = [data for event_type, data in emitted if event_type == "voice.command"]
        self.assertTrue(any(data.get("retry_attempted") is True and data.get("retry_transcript") == "Hebe abre OBS" for data in debug_events))
        self.assertTrue(any(data.get("final_decision") == "accepted" for data in debug_events))

    def test_unsupported_script_retry_prompt_injection_is_rejected_without_visible_prompt(self):
        engine = make_engine(["nuria"])
        engine.runtime.stt = RetrySTT("Hebe, Ebe, OBS, Twitch, chat, promo, shoutout, OBS, stream, chat, promo, shoutout")
        handled = []
        emitted = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"

        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            result = engine._process_stt_voice_transcript("à¤¯à¤¬ à¤†à¤¬à¤°à¥‡ à¤…à¤¬à¥‡ à¤¯à¤¶à¥‡")

        self.assertEqual(result, "continue")
        self.assertEqual(handled, [])
        rejected = [data for event_type, data in emitted if event_type == "voice.command" and data.get("status") == "rejected"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[-1]["reason"], "stt_prompt_echo_or_hotword_list")
        self.assertEqual(rejected[-1]["raw_text"], "à¤¯à¤¬ à¤†à¤¬à¤°à¥‡ à¤…à¤¬à¥‡ à¤¯à¤¶à¥‡")
        self.assertEqual(rejected[-1]["retry_transcript"], "")

    def test_unsupported_script_retry_still_unsupported_is_rejected(self):
        engine = make_engine(["nuria"])
        engine.runtime.stt = RetrySTT("à¤¯à¤¬ à¤†à¤¬à¤°à¥‡ à¤…à¤¬à¥‡ à¤¯à¤¶à¥‡")
        handled = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"
        emitted = []
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            result = engine._process_stt_voice_transcript("à¤¯à¤¬ à¤†à¤¬à¤°à¥‡ à¤…à¤¬à¥‡ à¤¯à¤¶à¥‡")

        self.assertEqual(result, "continue")
        self.assertEqual(handled, [])
        self.assertIn("reason=unsupported_script_after_retry script=devanagari", "\n".join(logs))
        rejected = [data for event_type, data in emitted if event_type == "voice.command" and data.get("status") == "rejected"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[-1]["reason"], "unsupported_script_after_retry")
        self.assertTrue(rejected[-1]["retry_attempted"])
        self.assertEqual(rejected[-1]["retry_transcript"], "à¤¯à¤¬ à¤†à¤¬à¤°à¥‡ à¤…à¤¬à¥‡ à¤¯à¤¶à¥‡")

    def test_retry_valid_eve_transcript_is_handled_by_wake_resolver(self):
        engine = make_engine(["nuria"])
        engine.runtime.stt = RetrySTT("Eve despierta")
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        result = engine._process_stt_voice_transcript("à¤¯à¤¬ à¤†à¤¬à¤°à¥‡ à¤…à¤¬à¥‡ à¤¯à¤¶à¥‡")

        self.assertEqual(result, "continue")
        self.assertEqual(delivered, [("stt_voice", "modelo:already_awake:")])

    def test_retry_route_does_not_add_command_specific_execution_hardcoding(self):
        engine = make_engine(["nuria"])
        engine.runtime.stt = RetrySTT("Hebe abre OBS")
        captured = []
        engine.handle_command = lambda command, source="voice": captured.append(command) or "continue"

        engine._process_stt_voice_transcript("à¤¯à¤¬ à¤†à¤¬à¤°à¥‡ à¤…à¤¬à¥‡ à¤¯à¤¶à¥‡")

        self.assertEqual(captured, ["abre obs"])

    def test_stt_worker_transcript_process_helper_logs_rejected_decision(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = False
        engine.stream_ambient_stt_enabled = False
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            result = engine._process_stt_voice_transcript("Estoy hablando solo")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertIn("[HEBE][INPUT] source=stt_voice raw='Estoy hablando solo'", joined)
        self.assertIn("[HEBE][INPUT_FIREWALL] source=ambient_stt", joined)
        self.assertIn("[HEBE][INPUT_CLASSIFY] source=stt_voice input_type=ambient_stream_context", joined)
        self.assertIn("[HEBE][RESPONSE_DECISION] should_reply=false reason=no_ignore", joined)
        self.assertIn("[HEBE][COG] decision=ambient_ignored_low_value reason=ambient_context_only", joined)

    def test_stt_shoutout_without_wakeword_still_has_action_intent(self):
        engine = make_engine(["nuria"])
        normalization = engine._normalize_stt_input("Haz una promo a Nuria")
        event = engine._build_input_event(
            source="stt_voice",
            raw_text=normalization.raw_text,
            normalized_text=normalization.normalized_text,
            stt_metadata=normalization.as_event(),
        )

        self.assertTrue(engine._input_event_has_action_intent(event))
        plan = engine._get_stream_action_planner().plan(event)
        self.assertEqual(plan.action_type, "twitch_shoutout")
        self.assertEqual(plan.target, "nuria")

    def test_awake_shoutout_without_despierta_enters_cognition(self):
        engine = make_engine(["nuria"])
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        result = engine.cognitive_flow("haz promo a nuria", source="stt_voice")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.twitch.sent, ["!so nuria"])
        self.assertEqual(delivered, [("stt_voice", "modelo:twitch_shoutout:nuria")])

    def test_awake_hebe_despierta_returns_already_awake(self):
        engine = make_engine(["nuria"])
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        result = engine.cognitive_flow("hebe despierta", source="stt_voice")

        self.assertEqual(result, "continue")
        self.assertFalse(engine.runtime.state.hebe_sleeping)
        self.assertEqual(delivered, [("stt_voice", "modelo:already_awake:")])

    def test_sleep_command_sets_sleeping_and_pauses_idle(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.idle_spontaneity_enabled = True
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        result = engine.cognitive_flow("hebe duerme", source="ui")

        self.assertEqual(result, "continue")
        self.assertTrue(engine.runtime.state.hebe_sleeping)
        self.assertFalse(engine.runtime.state.stream.idle_spontaneity_enabled)
        self.assertEqual(delivered, [("ui", "modelo:sleep_mode:")])

    def test_sleeping_ignores_non_wake_command(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.hebe_sleeping = True
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        result = engine.cognitive_flow("haz promo a nuria", source="stt_voice")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.twitch.sent, [])
        self.assertEqual(delivered, [])

    def test_sleeping_wake_command_wakes_then_commands_work(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.hebe_sleeping = True
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        wake = engine.cognitive_flow("hebe despierta", source="stt_voice")
        command = engine.cognitive_flow("haz promo a nuria", source="stt_voice")

        self.assertEqual(wake, "continue")
        self.assertEqual(command, "continue")
        self.assertFalse(engine.runtime.state.hebe_sleeping)
        self.assertEqual(engine.runtime.twitch.sent, ["!so nuria"])
        self.assertEqual(delivered[0], ("stt_voice", "modelo:wake_from_sleep:"))


if __name__ == "__main__":
    unittest.main()


