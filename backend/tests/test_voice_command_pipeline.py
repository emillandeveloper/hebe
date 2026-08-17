import os
import time
import sys
import unittest
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.cognitive.input_event import InputEvent
from app.cognitive.action_runtime import ActionRuntime
from app.cognitive.local_app_planner import LocalAppActionPlanner
from app.cognitive.deliberation_service import DeliberationService
from app.cognitive.models import DeliberationResult, ExecutionResult, Plan, PlanStep, StepExecutionResult
from app.cognitive.plan_executor import PlanExecutor
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.cognitive.cognitive_router import CognitiveRouter
from app.cognitive.game_guidance import GameRunState
from app.cognitive.wake_name_resolver import WakeNameResolver
from app.core.state import HebeState
from app.hebe_engine import HebeEngine
from app.integrations.twitch.chat_cache import TwitchChatCache
from app.integrations.twitch.target_resolver import TwitchTargetResolver
from app.services.voice_command_recovery import normalize_stt_transcript
from app.services.direct_stt_command import parse_direct_stt_command
from app.services.local_capability import (
    ApplicationCandidate,
    ApplicationDiscoveryService,
    LocalCapabilityResolver,
)
from app.stream.action_planner import StreamActionPlanner
from app.stream.state import StreamSessionState
from app.continuity import ConversationContinuityService, ConversationRepository, OpenThreadRepository
from app.replay.migrations import (
    MigrationRunner, belief_v2_migrations, conversation_continuity_migrations,
    game_context_v2_migrations,
)
from app.epistemics.repository import BeliefRepository
from app.epistemics.service import BeliefLifecycleService
from app.game_context_v2.repository import GameV2Repository
from app.game_context_v2.service import GameRunService


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
        if not result.requires_model_response:
            return result.fallback_text
        return f"modelo:{result.action_type}:{result.state_changes.get('target', result.state_changes.get('app_id', ''))}"

    def synthesize(self, **kwargs):
        execution = kwargs["execution"]
        step = execution.first_result_of_type("action")
        result = (step.data or {}).get("action_result") if step else None
        payload = getattr(result, "data", {}) or {}
        error_code = payload.get("error_code") or getattr(result, "error", None)
        command_result = SimpleNamespace(
            action_type=(step.data or {}).get("action_name") if step else "",
            success=bool(step and step.success),
            state_changes={
                "app_id": payload.get("app_id") or payload.get("requested_target"),
                "app_name": payload.get("app_name"),
                "error_code": error_code,
            },
            metadata={"error_code": error_code},
        )
        self.results.append((command_result, kwargs))
        return f"modelo:{command_result.action_type}:{command_result.state_changes.get('app_id') or ''}"


class FakeWin:
    def __init__(self, ok=True):
        self.ok = ok
        self.opened = []

    def open_app(self, app):
        self.opened.append(app)
        return self.ok


class RetrySTT:
    def __init__(self, retry_text, *, speech_detected=True, accepted=True, selected_language="es"):
        self.last_speech_detected = speech_detected
        self.retry_text = retry_text
        self.accepted = accepted
        self.selected_language = selected_language
        self.calls = []

    def retry_last_language_recovery(self, *, initial_language=None):
        self.calls.append({"initial_language": initial_language})
        return {
            "text": self.retry_text,
            "selected_text": self.retry_text if self.accepted else "",
            "selected_language": self.selected_language if self.accepted else "",
            "accepted": self.accepted,
            "attempted": True,
            "speech_detected": self.last_speech_detected,
        }


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


class SequentialResponseModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.replies.pop(0) if self.replies else ""


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
        state=HebeState(stream=stream),
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
    engine._last_cognitive_trace = {}
    engine._last_input_envelope = None
    engine.spontaneous_twitch_chat_enabled = False
    engine.stream_action_planner = engine._build_stream_action_planner()
    capabilities = {"audio.tts_control", "pending.cancel", "stream.local_state_control", "twitch_action", "hebe.wake_control"}
    engine._active_cognitive_decision = SimpleNamespace(
        authority="owner", source="ui", should_stop_pipeline=False,
        allowed_step_types=["state_update", "action", "reply"],
        action_permission_summary={"stream_live": bool(live)},
        allows_capability=lambda capability: capability in capabilities,
    )
    install_test_continuity(engine)
    return engine


def install_test_continuity(engine):
    engine._continuity_test_tmp = tempfile.mkdtemp(prefix="hebe-continuity-test-")
    db = Path(engine._continuity_test_tmp) / "continuity.sqlite3"
    connect = lambda: sqlite3.connect(db)
    MigrationRunner(connect).migrate(conversation_continuity_migrations())
    engine.conversation_continuity = ConversationContinuityService(
        ConversationRepository(connect), OpenThreadRepository(connect), now_fn=time.time,
    )
    engine._last_continuity_resolution = {}
    return engine


def open_test_conversation(engine, kind="appointment_datetime", expected_reply_type="datetime", **metadata):
    source = metadata.pop("conversation_source", "stt_voice")
    return engine._open_pending_conversation(
        kind=kind, expected_reply_type=expected_reply_type,
        can_accept_no_wake_followup=True, ttl_seconds=metadata.pop("ttl_seconds", 300),
        conversation_source=source, **metadata,
    )


def wire_canonical_app_pipeline(engine):
    engine.context_builder = FakeContextBuilder()
    engine.deliberation_service = DeliberationService(None, None)
    engine.deliberation_service.local_app_planner = engine._get_local_app_planner()
    engine.plan_executor = PlanExecutor(Mock(), engine.action_runtime)
    return engine


def wire_canonical_game_pipeline(engine):
    tmp=tempfile.mkdtemp(prefix="hebe-game-test-")
    db=Path(tmp)/"game.sqlite3"
    connect=lambda:sqlite3.connect(db)
    runner=MigrationRunner(connect);runner.migrate(belief_v2_migrations());runner.migrate(game_context_v2_migrations())
    repository=GameV2Repository(connect)
    lifecycle=BeliefLifecycleService(BeliefRepository(connect),now_fn=time.time)
    engine._game_test_tmp=tmp
    engine.game_v2_repository=repository
    engine.game_run_service=GameRunService(repository,lifecycle,now_fn=time.time)
    engine.runtime.state.stream.active_stream_session_id="game-test-session"
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

        self.assertEqual(result.normalized_text, "ebe az promo anuria")
        self.assertIn("hebe haz promo a nuria", result.alternative_candidates)
        self.assertTrue(result.metadata["normalization_only"])
        self.assertFalse(hasattr(result, "action_type"))

    def test_ui_hebe_abre_obs_creates_open_application_action_plan(self):
        engine = wire_canonical_app_pipeline(make_engine())
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": sys.executable}):
            result = engine.cognitive_flow("hebe abre obs", source="ui")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.win.opened[0]["app_id"], "obs")
        self.assertEqual(delivered, [("ui", "modelo:open_application:obs")])
        synth_result = engine.response_synthesizer.results[-1][0]
        self.assertEqual(synth_result.action_type, "open_application")
        self.assertTrue(synth_result.success)

    def test_ui_abre_obs_creates_open_application_when_awake_and_whitelisted(self):
        engine = wire_canonical_app_pipeline(make_engine())
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": sys.executable}):
            result = engine.cognitive_flow("abre obs", source="ui")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.win.opened[0]["app_id"], "obs")
        self.assertEqual(delivered, [("ui", "modelo:open_application:obs")])

    def test_stt_hebe_abre_obs_uses_same_open_application_pipeline(self):
        engine = wire_canonical_app_pipeline(make_engine())
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": sys.executable}):
            result = engine._process_stt_voice_transcript("Hebe abre OBS")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.win.opened[0]["app_id"], "obs")
        self.assertEqual(delivered, [("stt_voice", "modelo:open_application:obs")])

    def test_stt_portable_app_discovery_executes_once(self):
        transcript = "Hebe, abre Melón DS"
        engine = wire_canonical_app_pipeline(make_engine())
        emitted = []
        engine._deliver_manual_reply = lambda text, *, source: None
        direct = parse_direct_stt_command(transcript, event_id="portable-app-regression")
        metadata = {
            "command_hypothesis": {
                "wake_detected": True,
                "wake_score": 1.0,
                "hypothesis_agreement": 1.0,
                "action_structure_score": 1.0,
            },
            "direct_stt_command": direct.to_dict(),
            "command_mode": True,
            "action_eligible": True,
            "detected_language": "es",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = (
                Path(temp_dir)
                / "Users/Public/Documents/WinDS PRO/emu/melonds/melonDS.exe"
            )
            executable.parent.mkdir(parents=True)
            executable.touch()
            item_url = "file:" + str(executable).replace(os.sep, "/")
            discovery = ApplicationDiscoveryService()
            discovery._iter_windows_index_rows = Mock(side_effect=lambda term: iter([
                ("melonDS.exe", item_url, "melonDS.exe"),
            ]) if term == "melonds" else iter([]))
            discovery._search_registry_app_paths = Mock(return_value=[])
            discovery._search_installed_registry = Mock(return_value=[])
            discovery._search_shortcuts = Mock(return_value=[])
            discovery._search_executables = Mock(return_value=[])
            discovery._search_persisted_db_entries = Mock(return_value=[])
            engine.action_runtime.local_capability = LocalCapabilityResolver(discovery)
            with patch("app.services.local_capability.resolve_whitelisted_app", return_value=None), \
                 patch("app.services.local_capability.persist_learned_app", return_value=None), \
                 patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
                result = engine._process_stt_voice_transcript(transcript, stt_metadata=metadata)

        self.assertEqual(result, "continue")
        self.assertEqual(len(engine.runtime.win.opened), 1)
        self.assertEqual(engine.runtime.win.opened[0]["app_id"], "melonds")
        self.assertEqual(engine.runtime.win.opened[0]["executable_path"], str(executable))
        outcomes = [
            data for event_type, data in emitted
            if event_type == "voice.command" and data.get("status") == "outcome"
        ]
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["outcome"], "action_executed")

    def test_addressed_question_bypasses_app_resolver_and_replies(self):
        engine = make_engine()
        engine.runtime.state.stream.enabled = False
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("Aquí estoy.")
        engine._deliver_voice_reply = lambda _text: None
        planner = Mock()
        planner.command_markers.return_value = {"abre"}
        planner.plan.side_effect = AssertionError("App resolver must not run for direct conversation")
        engine.local_app_planner = planner
        emitted = []

        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))), \
             patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("Ebe, ¿estás ahí?")

        self.assertEqual(result, "continue")
        planner.plan.assert_not_called()
        self.assertEqual(engine.context_builder.inputs, ["ebe estas ahi"])
        outcomes = [data for event_type, data in emitted if event_type == "voice.command" and data.get("status") == "outcome"]
        self.assertEqual(outcomes[-1]["outcome"], "conversational_reply")

    def test_incomplete_direct_app_command_clarifies(self):
        engine = make_engine()
        engine.runtime.state.stream.enabled = False
        delivered = []
        emitted = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            result = engine._process_stt_voice_transcript("Hebe, abre")

        self.assertEqual(result, "continue")
        self.assertEqual(delivered, [("stt_voice", "¿Qué aplicación quieres que abra?")])
        outcomes = [data for event_type, data in emitted if event_type == "voice.command" and data.get("status") == "outcome"]
        self.assertEqual(outcomes[-1]["outcome"], "clarification")

    def test_direct_command_pipeline_failure_is_not_silent(self):
        engine = make_engine()
        engine.runtime.state.stream.enabled = False
        delivered = []
        emitted = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))
        engine.handle_command = Mock(side_effect=RuntimeError("parser exploded"))

        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            result = engine._process_stt_voice_transcript("Hebe, ¿estás ahí?")

        self.assertEqual(result, "continue")
        self.assertTrue(delivered)
        outcomes = [data for event_type, data in emitted if event_type == "voice.command" and data.get("status") == "outcome"]
        self.assertEqual(outcomes[-1]["outcome"], "action_failed")

    def test_obs_path_missing_returns_structured_action_result_not_generic_advice(self):
        engine = wire_canonical_app_pipeline(make_engine())
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
        execution = engine.response_synthesizer.results[-1][1]["execution"]
        action_result = execution.first_result_of_type("action").data["action_result"]
        self.assertEqual(action_result.error, "app_path_missing")
        self.assertEqual(action_result.data["error_code"], "app_path_missing")

    def test_unknown_app_does_not_execute_or_call_command_synth(self):
        engine = wire_canonical_app_pipeline(make_engine())
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        result = engine.cognitive_flow("hebe abre paint raro", source="ui")

        self.assertEqual(result, "continue")
        synth_result = engine.response_synthesizer.results[-1][0]
        self.assertFalse(synth_result.success)
        self.assertEqual(synth_result.state_changes.get("error_code"), "app_not_found")
        self.assertEqual(engine.runtime.win.opened, [])

    def test_non_whitelisted_app_does_not_execute_even_with_command_words(self):
        engine = wire_canonical_app_pipeline(make_engine())
        engine._deliver_manual_reply = lambda text, *, source: None

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": sys.executable}):
            result = engine.cognitive_flow("abre calculadora", source="ui")

        self.assertEqual(result, "continue")
        synth_result = engine.response_synthesizer.results[-1][0]
        self.assertFalse(synth_result.success)
        self.assertEqual(synth_result.state_changes.get("error_code"), "app_not_found")
        self.assertEqual(engine.runtime.win.opened, [])

    def test_local_app_planner_does_not_resolve_registry_before_runtime(self):
        event = InputEvent(source="ui", raw_text="abre obs", normalized_text="abre obs")

        planner = LocalAppActionPlanner()
        plan = planner.plan(event, is_awake=True)

        self.assertIsNotNone(plan)
        self.assertEqual(plan.action_type, "open_application")
        self.assertEqual(plan.status, "complete")
        self.assertEqual(plan.target, "obs")
        self.assertEqual(plan.slots.get("application_target"), "obs")
        self.assertNotIn("app_id", plan.slots)
        self.assertIsNone(plan.slots.get("app_record"))

    def test_model_is_not_called_before_open_application_action_plan(self):
        class GuardSynth(FakeSynth):
            def synthesize(self, **kwargs):
                step = kwargs["execution"].first_result_of_type("action")
                result = (step.data or {}).get("action_result") if step else None
                self.results.append((result, kwargs))
                assert (step.data or {}).get("action_name") == "open_application"
                assert (getattr(result, "data", {}) or {}).get("app_id")
                return "ok"

        engine = wire_canonical_app_pipeline(make_engine())
        engine.response_synthesizer = GuardSynth()
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": sys.executable}):
            result = engine.cognitive_flow("hebe inicia obs", source="ui")

        self.assertEqual(result, "continue")
        self.assertEqual(delivered, [("ui", "ok")])

    def test_planner_detects_generic_shoutout_phrasings(self):
        engine = make_engine()
        planner = engine._get_stream_action_planner()
        cases = {
            "Hebe haz promo a Nuria": "nuriiia___",
            "Haz una promo a Charlie": "er_tito_xarly",
            "Dale SO a Totodile": "Totodile",
            "Promociona a alguien_del_chat": "alguien_del_chat",
        }
        for text, target in cases.items():
            with self.subTest(text=text):
                plan = planner.plan(InputEvent(source="typed_ui", raw_text=text, normalized_text=text))
                self.assertIsNotNone(plan)
                self.assertEqual(plan.action_type, "twitch_shoutout")
                self.assertEqual(plan.status, "complete")
                self.assertEqual(plan.target, target)

    def test_promo_laurichu_extracts_laurichu(self):
        engine = make_engine([])
        planner = engine._get_stream_action_planner()

        plan = planner.plan(InputEvent(source="typed_ui", raw_text="Eve, haz una promo al Laurichu.", normalized_text="Eve, haz una promo al Laurichu."))

        self.assertIsNotNone(plan)
        self.assertEqual(plan.target, "laurichutv")
        self.assertNotIn("hazunapromo", plan.command.lower())

    def test_promo_angelo_noctis_extracts_angelo_noctis(self):
        engine = make_engine([])
        planner = engine._get_stream_action_planner()

        plan = planner.plan(InputEvent(source="typed_ui", raw_text="Hebe haz una promo a Angelo Noctis", normalized_text="Hebe haz una promo a Angelo Noctis"))

        self.assertIsNotNone(plan)
        self.assertEqual(plan.target, "angelo_noctis")

    def test_promo_command_words_not_in_target(self):
        engine = make_engine([])
        planner = engine._get_stream_action_planner()

        plan = planner.plan(InputEvent(source="typed_ui", raw_text="haz promo a haz una promo al laurichu", normalized_text="haz promo a haz una promo al laurichu"))

        self.assertIsNotNone(plan)
        self.assertNotIn("hazunapromo", (plan.command or "").lower())

    def test_promotion_guard_rejects_hazunapromoallaurichu(self):
        engine = make_engine([])
        planner = engine._get_stream_action_planner()

        accepted, reason, _target = planner._promotion_target_guard("hazunapromoallaurichu", resolved=True)

        self.assertFalse(accepted)
        self.assertEqual(reason, "command_words_in_target")

    def test_so_um_is_not_parsed_as_shoutout(self):
        engine = make_engine([])
        planner = engine._get_stream_action_planner()

        plan = planner.plan(InputEvent(source="typed_ui", raw_text="So, um...", normalized_text="So, um..."))

        self.assertIsNone(plan)

    def test_promotion_guard_rejects_filler_target_um(self):
        engine = make_engine([])
        planner = engine._get_stream_action_planner()

        accepted, reason, _target = planner._promotion_target_guard("um", resolved=False)

        self.assertFalse(accepted)
        self.assertEqual(reason, "filler_target")

    def test_valid_twitch_login_allowed(self):
        engine = make_engine([])
        planner = engine._get_stream_action_planner()

        accepted, reason, target = planner._promotion_target_guard("laurichutv", resolved=True)

        self.assertTrue(accepted, reason)
        self.assertEqual(target, "laurichutv")

    def test_promo_nuria_alias(self):
        engine = make_engine([])

        result = engine._handle_stream_manual_command("haz promo a Nuria")

        self.assertEqual(result.action_type, "twitch_shoutout")
        self.assertTrue(result.success)
        self.assertEqual(result.state_changes["target"], "nuriiia___")
        self.assertEqual(engine.runtime.twitch.sent, ["!so nuriiia___"])

    def test_promo_charlie_alias_strip_suffix(self):
        engine = make_engine([])

        result = engine._handle_stream_manual_command("haz promo a Charlie, a ver si ahora lo hace")

        self.assertEqual(result.action_type, "twitch_shoutout")
        self.assertTrue(result.success)
        self.assertEqual(result.state_changes["target"], "er_tito_xarly")
        self.assertEqual(engine.runtime.twitch.sent, ["!so er_tito_xarly"])

    def test_owner_promo_with_filler_executes(self):
        engine = make_engine(["superdamu"])

        result = engine._handle_stream_manual_command("de hecho Eve, haz una promo a Super Dammu.")

        self.assertEqual(result.action_type, "twitch_shoutout")
        self.assertTrue(result.success)
        self.assertEqual(engine.runtime.twitch.sent, ["!so superdamu"])
        self.assertNotIn("charlieaversiahoralohace", engine.runtime.twitch.sent[0].lower())

    def test_promo_superdamu_active_viewer_fuzzy(self):
        engine = make_engine([])
        engine.observe_twitch_chat_message("superdamu", "SUPERDAMU", "hola", "#chan")

        result = engine._handle_stream_manual_command("E-B, haz una promo a Super Dammu, a ver si lo hace")

        self.assertEqual(result.action_type, "twitch_shoutout")
        self.assertTrue(result.success)
        self.assertEqual(result.state_changes["target"], "superdamu")
        self.assertEqual(engine.runtime.twitch.sent, ["!so superdamu"])

    def test_promo_superdamu_no_wake_followup_after_pending(self):
        engine = make_engine([])
        engine.observe_twitch_chat_message("superdamu", "SUPERDAMU", "hola", "#chan")

        first = engine._handle_stream_manual_command("haz promo a usuario_inventado")
        followup = engine._handle_stream_manual_command("Super Damu")

        self.assertEqual(first.action_type, "twitch_shoutout_clarify")
        self.assertEqual(followup.action_type, "twitch_shoutout")
        self.assertTrue(followup.success)
        self.assertEqual(followup.state_changes["target"], "superdamu")
        self.assertEqual(engine.runtime.twitch.sent, ["!so superdamu"])

    def test_promotion_pending_followup_allows_promotion_action(self):
        engine = make_engine([])
        open_test_conversation(
            engine, kind="promotion_target_clarification",
            expected_reply_type="twitch_username_or_viewer_alias", candidates=[], ttl_seconds=60,
        )
        engine.observe_twitch_chat_message("superdamu", "SUPERDAMU", "hola", "#chan")

        result = engine._process_stt_voice_transcript("Super Damu")

        self.assertEqual(result, "continue")
        firewall = engine._last_input_firewall
        self.assertIn("promotion_shoutout", firewall.get("allowed_actions", []))
        self.assertIn("twitch_action", firewall.get("allowed_actions", []))
        self.assertEqual(engine.runtime.twitch.sent, ["!so superdamu"])

    def test_promo_pending_contextual_recent_chatter_followup(self):
        engine = make_engine([])
        engine.observe_twitch_chat_message("superdamu", "SUPERDAMU", "hola", "#chan")

        first = engine._handle_stream_manual_command("haz promo")
        followup = engine._handle_stream_manual_command("el que acaba de hablar")

        self.assertEqual(first.action_type, "twitch_shoutout_clarify")
        self.assertEqual(followup.action_type, "twitch_shoutout")
        self.assertTrue(followup.success)
        self.assertEqual(followup.state_changes["target"], "superdamu")
        self.assertEqual(engine.runtime.twitch.sent, ["!so superdamu"])

    def test_owner_mute_command_sets_wake_only_and_suppresses_tts(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        engine.runtime.state.tts_enabled = True
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("deja de hablar")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.state.stream.stream_voice_mode, "wake_only")
        self.assertGreater(engine.runtime.state.stream.wake_only_until, time.time())
        self.assertIn("[HEBE][OWNER_MUTE_COMMAND] mode=wake_only", joined)
        self.assertIn("[HEBE][TTS_CANCEL] reason=owner_mute", joined)
        self.assertIn("[HEBE][VOICE_MODE] mode=wake_only", joined)

    def test_monologue_with_hablar_does_not_mute(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True

        self.assertIsNone(engine._owner_mute_command_mode("vamos a hablar de la build del juego"))

    def test_monologue_with_callado_does_not_mute(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True

        self.assertIsNone(engine._owner_mute_command_mode("hoy me quedo callado un rato"))

    def test_direct_hebe_deja_de_hablar_sets_wake_only(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True

        with patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("Hebe, deja de hablar")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.state.stream.stream_voice_mode, "wake_only")
        self.assertEqual(engine.runtime.state.stream.voice_mode_activated_by_text, "Hebe, deja de hablar")

    def test_wake_only_expires_after_ttl(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        engine.runtime.state.stream.stream_voice_mode = "wake_only"
        engine.runtime.state.stream.wake_only_until = time.time() - 1
        engine.runtime.state.stream.voice_mode_expires_at = time.time() - 1

        mode, reason = engine._stream_voice_mode_active()

        self.assertEqual(mode, "normal")
        self.assertEqual(reason, "")
        self.assertEqual(engine.runtime.state.stream.stream_voice_mode, "normal")

    def test_wake_only_allows_direct_command(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        engine.runtime.state.stream.stream_voice_mode = "wake_only"
        engine.runtime.state.stream.wake_only_until = time.time() + 300
        handled = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"

        with patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("Hebe, que hora es")

        self.assertEqual(result, "continue")
        self.assertEqual(handled, [("stt_voice", "que hora es")])

    def test_muted_suppresses_proactive_idle_prompt(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        engine.runtime.state.stream.stream_voice_mode = "muted"
        engine.runtime.state.stream.muted_until = time.time() + 300
        engine.runtime.state.stream.wake_only_until = time.time() + 300
        engine.runtime.state.stream.mute_reason = "owner_mute"
        engine.context_builder = Mock()
        logs = []
        event = SimpleNamespace(event_type="twitch_idle_prompt", payload={"idle_topic": "resource_management"})

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            engine.process_internal_event(event)

        self.assertIn("[HEBE][PROACTIVE_SUPPRESSED] reason=owner_mute", "\n".join(logs))
        engine.context_builder.build.assert_not_called()

    def test_output_dedupe_suppresses_same_manual_reply(self):
        engine = make_engine(["nuria"])
        emitted = []
        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))), \
             patch("app.hebe_engine.log_chat"):
            engine._deliver_manual_reply("Hecho.", source="ui")
            engine._deliver_manual_reply("Hecho.", source="ui")

        assistant_events = [event for event in emitted if event[0] == "chat.assistant"]
        self.assertEqual(len(assistant_events), 1)

    def test_ui_cognitive_flow_emits_one_final_assistant_message(self):
        engine = make_engine(["nuria"], live=False)
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("Respuesta final.")
        engine.memory_extractor = Mock()
        emitted = []
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))), \
             patch("app.hebe_engine.log_chat"):
            result = engine.cognitive_flow("hola", source="ui")

        assistant_events = [event for event in emitted if event[0] == "chat.assistant"]
        self.assertEqual(result, "continue")
        self.assertEqual(len(assistant_events), 1)
        self.assertEqual(assistant_events[0][1]["text"], "Respuesta final.")
        self.assertIn("[HEBE][FINAL_EMISSION_GATE] emitted=true route=local_owner_reply", "\n".join(logs))

    def test_twitch_viewer_reply_addressed_to_owner_is_suppressed(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True

        engine._deliver_twitch_reply(
            "Te leo, Leo. Recalibro.",
            event_type="twitch_chat_react",
            payload={"message_text": "Hebe?", "username": "yulawild", "user_login": "yulawild"},
        )

        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_translate_previous_response_uses_last_assistant_text(self):
        engine = make_engine(["nuria"], live=False)
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append(text)
        engine._remember_assistant_text("Mira recursos antes de avanzar.", source="stt_voice")

        result = engine.cognitive_flow("Ahora me lo dices en ingles", source="ui")

        self.assertEqual(result, "continue")
        self.assertEqual(delivered, ["In English: Mira recursos antes de avanzar."])
        self.assertNotIn("Now say", delivered[0])

    def test_promo_low_confidence_asks_clarification(self):
        engine = make_engine([])

        result = engine._handle_stream_manual_command("shoutout a usuario_inventado")

        self.assertEqual(result.action_type, "twitch_shoutout_clarify")
        self.assertFalse(result.success)
        self.assertEqual(engine.runtime.twitch.sent, [])
        self.assertEqual(engine._active_current_conversation().topic, "promotion_target_clarification")

    def test_no_debug_metadata_in_spoken_promo_response(self):
        engine = make_engine([])
        result = engine._handle_stream_manual_command("haz promo a Nuria")
        reply = engine._synthesize_command_result(result, input_text="haz promo a Nuria")

        self.assertEqual(reply, "Promo hecha para nuriiia___.")
        self.assertNotIn("confidence", reply.lower())
        self.assertNotIn("!so", reply.lower())
        self.assertNotIn("command", reply.lower())

    def test_owner_promotion_command_not_dropped(self):
        decision = CognitiveRouter().route(SimpleNamespace(
            input_text="haz promo a Nuria",
            source="owner_stt_direct",
            authority="owner",
            addressed_to_hebe=True,
            firewall_decision="allow",
            stream_is_live=True,
            route_hints=[],
            state_snapshot={},
        ))

        self.assertEqual(decision.intent, "promotion_shoutout")
        self.assertTrue(decision.allows_capability("twitch.promotion"))
        self.assertEqual(decision.response_mode, "command_result")
        self.assertTrue(decision.should_reply)

    def test_noisy_stt_enters_planner_as_normalized_candidate(self):
        engine = make_engine(["nuria"])
        normalization = normalize_stt_transcript("ebe az promo anuria", known_targets=engine._known_voice_command_targets())
        command_hypothesis = normalization.alternative_candidates[-1]
        plan = engine._get_stream_action_planner().plan(
            InputEvent(source="stt_voice", raw_text=normalization.raw_text, normalized_text=command_hypothesis, is_voice=True)
        )

        self.assertEqual(normalization.normalized_text, "ebe az promo anuria")
        self.assertEqual(command_hypothesis, "hebe haz promo a nuria")
        self.assertEqual(plan.action_type, "twitch_shoutout")
        self.assertEqual(plan.status, "complete")
        self.assertEqual(plan.target, "nuriiia___")

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
        self.assertEqual(engine.runtime.twitch.sent, ["!so nuriiia___"])
        self.assertEqual(reply, "Promo hecha para nuriiia___.")

    def test_narrow_fuzzy_wake_recovers_anuria_promotion(self):
        engine = make_engine(["nuriiia___"])
        engine.runtime.twitch.remember_user_alias("anuria", "nuriiia___")

        handled, command = engine._extract_stream_command("A ver, espera, Efe, haz una promo anuria.")
        result = engine._handle_stream_manual_command(command)

        self.assertTrue(handled)
        self.assertEqual(command, "haz una promo anuria")
        self.assertTrue(result.success)
        self.assertEqual(engine.runtime.twitch.sent, ["!so nuriiia___"])
        self.assertEqual(engine.runtime.state.stream.last_promotion_outcome["outcome"], "executed")

    def test_narrow_fuzzy_wake_recovers_ismael_promotion(self):
        engine = make_engine(["ismael"])

        handled, command = engine._extract_stream_command("Y ve, haz una promo Ismael.")
        with patch.object(engine, "_record_promotion_outcome", wraps=engine._record_promotion_outcome) as record:
            result = engine._handle_stream_manual_command(command)

        self.assertTrue(handled)
        self.assertTrue(result.success)
        self.assertEqual(engine.runtime.twitch.sent, ["!so ismael"])
        self.assertEqual(record.call_count, 1)
        self.assertEqual(engine.runtime.state.stream.last_promotion_outcome["outcome"], "executed")

    def test_fuzzy_wake_recovery_does_not_expand_to_non_promotion_speech(self):
        engine = make_engine(["ismael"])

        handled, command = engine._extract_stream_command("Efe, dime que tal va todo")

        self.assertTrue(handled)
        self.assertIsNone(command)
        self.assertEqual(engine.runtime.twitch.sent, [])

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
            command_hypothesis = normalization.alternative_candidates[-1]
            engine._current_input_event = engine._build_input_event(
                source="stt_voice",
                raw_text=normalization.raw_text,
                normalized_text=command_hypothesis,
                stt_metadata=normalization.as_event(),
            )
            result = engine._handle_stream_manual_command(command_hypothesis)
            engine._synthesize_command_result(result, input_text=command_hypothesis)

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
        pending = open_test_conversation(
            engine, kind="assistant_followup", expected_reply_type="casual_answer",
            ttl_seconds=30, previous_assistant_message_id="assistant-test",
            previous_assistant_message="Pues bien, dormida a ratos pero lista para tus locuras, cabron.",
            source="assistant_question", max_attempts=1, reply_source="stt_voice",
        )
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
        current = engine._active_current_conversation()
        self.assertEqual((current.id, current.version), (pending.id, pending.version))
        self.assertEqual(current.domain_payload["attempts"], 0)
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

    def test_owner_direct_personal_state_reaches_cognitive_router(self):
        samples = (
            ("Hebe, tengo hambre", "hunger"),
            ("Hebe, estoy cansado", "fatigue"),
        )
        for spoken, expected_state in samples:
            with self.subTest(spoken=spoken):
                engine = make_engine(["nuria"])
                engine.runtime.state.stream.enabled = False
                engine.context_builder = FakeContextBuilder()
                engine.deliberation_service = FakeDeliberationService()
                engine.plan_executor = FakePlanExecutor()
                engine.response_synthesizer = FixedResponseSynth("Te escucho, Leo.")
                engine.memory_extractor = Mock()
                delivered = []
                logs = []
                engine._deliver_voice_reply = lambda text, **kwargs: delivered.append(text)

                with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
                     patch("app.hebe_engine.emit"), \
                     patch("app.hebe_engine.log_chat"):
                    result = engine._process_stt_voice_transcript(spoken)

                joined = "\n".join(logs)
                trace = engine._last_cognitive_trace
                self.assertEqual(result, "continue")
                self.assertEqual(delivered, ["Te escucho, Leo."])
                self.assertIn("[HEBE][STT_GATE] passed reason=owner_direct_addressed_to_hebe", joined)
                self.assertNotIn("[HEBE][STT_REJECTED]", joined)
                self.assertEqual(trace["source"], "owner_stt_direct")
                self.assertEqual(trace["authority"], "owner")
                self.assertTrue(trace["addressed_to_hebe"])
                self.assertEqual(trace["intent"], "owner_personal_state")
                self.assertEqual(trace["personal_state"], expected_state)
                self.assertFalse(trace["uses_pending_task"])
                self.assertTrue(trace["should_reply"])
                self.assertEqual(trace["response_mode"], "companion_reaction")

    def test_pending_appointment_datetime_without_wake_is_unified_owner_followup(self):
        engine = make_engine()
        engine.runtime.state.stream.enabled = False
        open_test_conversation(engine, kind="appointment_datetime", ttl_seconds=300)
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("Fecha recibida.")
        engine.memory_extractor = Mock()
        delivered = []
        logs = []
        engine._deliver_voice_reply = lambda text, **kwargs: delivered.append(text)

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("el 16 de septiembre a las 10")

        trace = engine._last_cognitive_trace
        envelope = engine._last_input_envelope
        self.assertEqual(result, "continue")
        self.assertEqual(delivered, ["Fecha recibida."])
        self.assertEqual(envelope.source, "owner_stt_followup")
        self.assertTrue(envelope.pending_compatible)
        self.assertEqual(trace["intent"], "pending_datetime_answer")
        self.assertTrue(trace["uses_pending_task"])
        self.assertTrue(trace["should_reply"])
        self.assertIn("[HEBE][CONVERSATION_STATE] active=true matched=true", "\n".join(logs))
        self.assertIn("reason=pending_compatible_input_envelope", "\n".join(logs))

    def test_pending_appointment_with_wake_remains_pending_followup(self):
        engine = make_engine()
        engine.runtime.state.stream.enabled = False
        open_test_conversation(engine, kind="appointment_datetime", ttl_seconds=300)
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("Fecha recibida.")
        engine.memory_extractor = Mock()
        engine._deliver_voice_reply = lambda text, **kwargs: None

        with patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            engine._process_stt_voice_transcript("Hebe, el 16 de septiembre a las 10")

        self.assertTrue(engine._last_input_envelope.addressed_to_hebe)
        self.assertEqual(engine._last_input_envelope.source, "owner_stt_followup")
        self.assertEqual(engine._last_cognitive_trace["intent"], "pending_datetime_answer")

    def test_current_time_with_pending_appointment_is_new_direct_request(self):
        engine = make_engine()
        engine.runtime.state.stream.enabled = False
        open_test_conversation(engine, kind="appointment_datetime", ttl_seconds=300)
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("Son las doce.")
        engine.memory_extractor = Mock()
        engine._deliver_voice_reply = lambda text, **kwargs: None

        with patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            engine._process_stt_voice_transcript("Hebe, que hora es")

        envelope = engine._last_input_envelope
        trace = engine._last_cognitive_trace
        self.assertEqual(envelope.source, "owner_stt_direct")
        self.assertFalse(envelope.pending_compatible)
        self.assertEqual(trace["intent"], "current_time_query")
        self.assertFalse(trace["uses_pending_task"])

    def test_no_wake_whitelisted_app_command_routes_while_stream_offline(self):
        engine = wire_canonical_app_pipeline(make_engine(live=False))
        engine.runtime.state.stream.enabled = False
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": sys.executable}), \
             patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("Abre OBS")

        envelope = engine._last_input_envelope
        trace = engine._last_cognitive_trace
        self.assertEqual(result, "continue")
        self.assertEqual(envelope.source, "owner_stt_command")
        self.assertEqual(envelope.app_target, "OBS")
        self.assertEqual(trace["intent"], "command_open_app")
        self.assertIn("pc.open_application", trace["allowed_capabilities"])
        self.assertEqual(engine.runtime.win.opened[0]["app_id"], "obs")
        self.assertTrue(delivered)

    def test_ambient_personal_state_without_wake_does_not_reach_cognition(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = False
        handled = []
        logs = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"):
            result = engine._process_stt_voice_transcript("tengo hambre")

        self.assertEqual(result, "continue")
        self.assertEqual(handled, [])
        self.assertIn("[HEBE][STT_GATE] ambient_only reason=no_wake_no_valid_pending", "\n".join(logs))

    def test_random_unaddressed_noise_remains_safely_ignored(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = False
        handled = []
        engine.handle_command = lambda command, source="voice": handled.append((source, command)) or "continue"

        with patch("app.hebe_engine.emit"):
            result = engine._process_stt_voice_transcript("blargh krzzzt mmm")

        self.assertEqual(result, "continue")
        self.assertEqual(handled, [])

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

    def test_assistant_question_creates_canonical_conversation(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = False
        engine.pending_conversation_ttl_seconds = 120

        engine._record_assistant_reply_for_conversation("AquÃ­ sobreviviendo. Â¿tÃº quÃ© tal?", source="stt_voice", synthesizer=pending_marker())

        conversation = engine._active_current_conversation()
        self.assertEqual(conversation.topic, "assistant_followup")
        self.assertEqual(conversation.domain_payload["source"], "assistant_question")
        self.assertEqual(conversation.expected_reply.type.value, "casual_answer")
        self.assertEqual(list(conversation.expected_reply.allowed_sources), ["owner_stt", "owner_ui"])
        self.assertTrue(conversation.domain_payload["can_accept_no_wake_followup"])

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

    def test_live_owner_monologue_does_not_reply_as_conversation_followup(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("No deberia sonar.")
        engine.memory_extractor = Mock()
        delivered = []
        logs = []
        engine._deliver_voice_reply = lambda text, **kwargs: delivered.append(text)
        engine._record_assistant_reply_for_conversation("Que tal vas?", source="stt_voice", synthesizer=pending_marker())

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("un RPG pero con combate por combos")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(delivered, [])
        self.assertEqual(engine.context_builder.inputs, [])
        self.assertIn("[HEBE][LIVE_OWNER_SPEECH_GATE] action=context_only", joined)
        self.assertNotIn("decision=conversation_followup", joined)

    def test_expired_pending_is_purged_before_stt_classification(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        expired = open_test_conversation(engine, kind="appointment_datetime", ttl_seconds=-1)
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("el jueves")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertIsNone(engine._active_current_conversation())
        stored = engine.conversation_continuity.conversations.get(expired.id)
        self.assertEqual(stored.status.value, "EXPIRED")
        self.assertIn(f"[HEBE][PENDING_EXPIRED] kind=appointment_datetime id={expired.id}", joined)

    def test_appointment_pending_rejects_stream_planning_weekday_chatter(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        open_test_conversation(engine, kind="appointment_datetime", ttl_seconds=300)
        engine.context_builder = FakeContextBuilder()
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("estar atentos porque voy a traer si terminamos la partida el jueves")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(engine.context_builder.inputs, [])
        self.assertIn("[HEBE][LIVE_OWNER_SPEECH_GATE] action=context_only", joined)
        self.assertIn("pending_compatible=false", joined)

    def test_live_owner_monologue_not_reply_even_with_unrelated_pending(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        open_test_conversation(
            engine, kind="appointment_datetime", expected_reply_type="datetime",
            explicit_question_asked=True, ttl_seconds=300, max_attempts=1,
        )
        engine.context_builder = FakeContextBuilder()
        delivered = []
        logs = []
        engine._deliver_voice_reply = lambda text, **kwargs: delivered.append(text)

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("el jueves hacemos stream largo si terminamos esta partida")

        self.assertEqual(result, "continue")
        self.assertEqual(delivered, [])
        self.assertEqual(engine.context_builder.inputs, [])
        self.assertIsNotNone(engine._active_current_conversation())
        joined = "\n".join(logs)
        self.assertIn("[HEBE][LIVE_OWNER_SPEECH_GATE] action=context_only", joined)
        self.assertIn("pending_compatible=false", joined)

    def test_simulated_owner_stt_monologue_is_context_only(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        engine.context_builder = FakeContextBuilder()

        result = engine.simulate_leo_message(
            "voy a farmear un poco antes del jefe",
            source="stt_voice",
            stream_live_mode="force_stream_live",
        )

        self.assertTrue(result["ok"])
        self.assertIn(result.get("policy_decision"), {"ignored", "allowed"})
        self.assertEqual(result.get("final_response") or "", "")
        self.assertEqual(engine.context_builder.inputs, [])

    def test_promotion_pending_rejects_stream_words_without_resolver_target(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        open_test_conversation(
            engine, kind="promotion_target_clarification",
            expected_reply_type="twitch_username_or_viewer_alias",
            explicit_question_asked=True, ttl_seconds=60, max_attempts=1,
        )
        engine.context_builder = FakeContextBuilder()
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("este combate del jueves va cerca")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.context_builder.inputs, [])
        self.assertEqual(engine.runtime.twitch.sent, [])
        joined = "\n".join(logs)
        self.assertIn("kind=promotion_target_clarification", joined)
        self.assertIn("compatible=false", joined)

    def test_promo_rejects_insult_as_target(self):
        engine = make_engine(["nuria"], live=True)

        compatible, reason, resolution = engine._promotion_pending_reply_compatible("un tonto", "un tonto", {})

        self.assertFalse(compatible)
        self.assertEqual(reason, "invalid_target")
        self.assertEqual(resolution.get("target") or "", "")

    def test_promo_rejects_normal_sentence_as_target(self):
        engine = make_engine(["nuria"], live=True)

        compatible, reason, _resolution = engine._promotion_pending_reply_compatible(
            "este combate del jueves va cerca",
            "este combate del jueves va cerca",
            {},
        )

        self.assertFalse(compatible)
        self.assertIn(reason, {"stream_monologue", "sentence_fragment", "low_confidence_target"})

    def test_promo_h_single_letter_asks_clarification(self):
        engine = make_engine(["nuria"], live=True)

        compatible, reason, _resolution = engine._promotion_pending_reply_compatible("h", "h", {})

        self.assertFalse(compatible)
        self.assertIn(reason, {"single_letter_target", "ambiguous_single_letter_target"})

    def test_promo_hache_asks_clarification(self):
        engine = make_engine(["nuria"], live=True)

        compatible, reason, _resolution = engine._promotion_pending_reply_compatible("hache", "hache", {})

        self.assertFalse(compatible)
        self.assertEqual(reason, "ambiguous_single_letter_target")

    def test_promo_lache_resolves_active_chatter(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.recent_active_users = ["lache"]

        compatible, reason, resolution = engine._promotion_pending_reply_compatible("lache", "lache", {})

        self.assertTrue(compatible)
        self.assertEqual(reason, "recent_active_chatter")
        self.assertEqual(str(resolution.get("target")).lower(), "lache")

    def test_viewer_promo_request_blocked(self):
        engine = make_engine(["nuria"], live=True)

        category = engine._classify_twitch_viewer_message("Hebe haz promo a alguien")
        contract = engine._policy_contract_for_twitch_category(category=category)

        self.assertEqual(category, "promo_request_from_viewer")
        self.assertEqual(contract.result, "redirect")
        self.assertTrue(contract.boundary_required)

    def test_structured_game_pending_has_one_live_attempt(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        context = SimpleNamespace(
            input_text="Hebe ayuda con el juego",
            stream_is_live=True,
        )
        decision = SimpleNamespace(intent="game_guidance_query")
        reply_data = {
            "mode": "game_guidance_clarification",
            "game_guidance": {
                "context": {
                    "game": "Persona 5 Royal",
                    "source_context": {"user_input": "Hebe ayuda"},
                }
            },
        }
        engine.deliberation_service = SimpleNamespace(
            game_guidance=SimpleNamespace(missing_fields=lambda guidance: ["current_location"])
        )

        engine._apply_game_guidance_reply_state(reply_data, decision, context, "owner_stt_direct")

        pending = engine._active_current_conversation()
        self.assertEqual(pending.topic, "game_guidance_clarification")
        self.assertEqual(pending.status.value, "WAITING_ON_LEO")
        self.assertEqual(pending.domain_payload["max_attempts"], 1)
        self.assertTrue(pending.domain_payload["explicit_question_asked"])
        self.assertTrue(pending.domain_payload["can_accept_no_wake_followup"])

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
        self.assertIn("[HEBE][INPUT_CLASSIFY] source=ambient_stt input_type=ambient_stream_context", joined)
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
        self.assertIn("[HEBE][INPUT_CLASSIFY] source=owner_stt_direct input_type=explicit_question", joined)
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

    def test_twitch_mention_reply_posts_to_twitch_chat_and_no_private_conversation(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        engine.runtime.state.tts_enabled = False
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"):
            engine._deliver_twitch_reply(
                "Corta y al pie.",
                event_type="twitch_chat_react",
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe, que opinas de esto?"},
            )
            engine._record_assistant_reply_for_conversation("¿tú qué tal?", source="twitch_chat_react", synthesizer=pending_marker())

        self.assertEqual(engine.runtime.twitch.sent, ["Corta y al pie."])
        self.assertIsNone(engine._active_current_conversation())
        self.assertIn("input_type=twitch_mention_or_event output_target=twitch_chat", "\n".join(logs))

    def test_twitch_mention_not_always_public_reply(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
            patch("app.hebe_engine.emit"):
            engine._deliver_twitch_reply(
                "Visto.",
                event_type="twitch_chat_react",
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe xd"},
            )

        joined = "\n".join(logs)
        self.assertEqual(engine.runtime.twitch.sent, [])
        self.assertIn("[HEBE][OUTPUT_ROUTE_DECISION] route=observe_only", joined)

    def test_high_value_question_writes_to_twitch(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True

        engine._deliver_twitch_reply(
            "FFIX tiene una melancolia muy fina.",
            event_type="twitch_chat_react",
            payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe, que opinas de Final Fantasy IX?"},
        )

        self.assertEqual(engine.runtime.twitch.sent, ["FFIX tiene una melancolia muy fina."])

    def test_emote_only_no_reply(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True

        engine._deliver_twitch_reply(
            "Te leo.",
            event_type="twitch_chat_react",
            payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Kappa Kappa"},
        )

        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_repeated_viewer_mentions_budgeted(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True

        for idx in range(3):
            engine._deliver_twitch_reply(
                f"Respuesta {idx}.",
                event_type="twitch_chat_react",
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": f"Hebe, que opinas de tema {idx}?"},
            )

        self.assertEqual(engine.runtime.twitch.sent, ["Respuesta 0.", "Respuesta 1."])

    def test_thread_followup_limited(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        thread_id = engine._twitch_thread_id(username="viewer", text="Hebe, que opinas de esto?", category="high_value_question")
        engine.runtime.state.stream.public_reply_thread_counts[thread_id] = 2

        engine._deliver_twitch_reply(
            "Otra respuesta.",
            event_type="twitch_chat_react",
            payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe, que opinas de esto?"},
        )

        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_reply_parent_to_hebe_detected_and_bypasses_viewer_cooldown(self):
        engine = make_engine(["viewer"])
        stream = engine.runtime.state.stream
        stream.is_live = True
        now = time.time()
        stream.public_reply_viewer_timestamps = {"viewer": [now - 10, now - 20]}

        category = engine._classify_twitch_viewer_message(
            "@HebeNifelheim Respete a los mayores",
            payload={"reply_to_hebe_message": True, "mentions_hebe": True},
        )
        budget = engine._twitch_reply_budget_allows(
            stream=stream,
            username="viewer",
            category=category,
            thread_id="viewer:reply_to_hebe_message:respete",
            payload={"reply_to_hebe_message": True, "direct_priority_reason": "reply_to_hebe_message"},
        )

        self.assertEqual(category, "reply_to_hebe_message")
        self.assertTrue(budget["allowed"])
        self.assertTrue(budget["direct_priority_applied"])

    def test_mention_hebe_never_normal_no_mention(self):
        engine = make_engine(["viewer"])

        category = engine._classify_twitch_viewer_message("Hebenifelheim oye no te metas en conversaciones ajenas Kappa")

        self.assertIn(category, {"direct_hebe_prompt", "direct_hebe_banter", "viewer_talks_about_hebe"})
        self.assertNotEqual(category, "normal_no_mention_chat")

    def test_greeting_not_high_value_question(self):
        engine = make_engine(["viewer"])

        category = engine._classify_twitch_viewer_message("que ondaaaaa")

        self.assertNotEqual(category, "high_value_question")

    def test_human_chat_between_replies_does_not_reset_viewer_cooldown(self):
        engine = make_engine(["viewer"])
        stream = engine.runtime.state.stream
        now = time.time()
        stream.public_reply_viewer_timestamps = {"viewer": [now - 10, now - 20]}
        stream.consecutive_public_replies = 1

        engine.observe_twitch_chat_message("other", "Other", "hola normal", "#chan")
        budget = engine._twitch_reply_budget_allows(
            stream=stream,
            username="viewer",
            category="high_value_question",
            thread_id="thread",
            payload={},
        )

        self.assertFalse(budget["allowed"])
        self.assertEqual(budget["reason"], "viewer_cooldown")

    def test_third_person_leo_mention_allowed(self):
        engine = make_engine(["nuria"])

        allowed, reason = engine._target_speaker_guard(
            "yo me quedo con Leo en este desastre.",
            source="twitch_viewer",
            speaker="Viewer",
        )

        self.assertTrue(allowed, reason)

    def test_direct_leo_address_from_viewer_blocked(self):
        engine = make_engine(["nuria"])

        allowed, reason = engine._target_speaker_guard(
            "Leo, mira el chat ahora.",
            source="twitch_viewer",
            speaker="Viewer",
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "viewer_answer_addressed_to_owner")

    def test_viewer_relay_to_leo_blocked(self):
        engine = make_engine(["nuria"])

        allowed, reason = engine._target_speaker_guard(
            "Dile a Leo que mire esto.",
            source="twitch_viewer",
            speaker="Viewer",
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "viewer_answer_addressed_to_owner")

    def test_loyalty_statement_about_leo_allowed(self):
        engine = make_engine(["nuria"])

        allowed, reason = engine._target_speaker_guard(
            "Me quedo con Leo en este caos.",
            source="twitch_viewer",
            speaker="Viewer",
        )

        self.assertTrue(allowed, reason)

    def test_boundary_can_bypass_budget(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        now = time.time()
        engine.runtime.state.stream.public_reply_timestamps = [now - 5, now - 7, now - 9, now - 11, now - 13]
        engine.runtime.state.stream.consecutive_public_replies = 3
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            decision = engine._twitch_reply_budget_allows(
                stream=engine.runtime.state.stream,
                username="viewer",
                category="viewer_command_attempt",
                thread_id="viewer:viewer_command_attempt:test",
                now=now,
            )

        self.assertTrue(decision["allowed"])
        self.assertEqual(decision["reason"], "allowed")
        self.assertIn("[HEBE][TWITCH_REPLY_BUDGET] allowed=true", "\n".join(logs))

    def test_followup_question_not_allowed_for_banter(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            engine._deliver_twitch_reply(
                "Te vi. Qué quieres ahora?",
                event_type="twitch_chat_react",
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe esta muy callada"},
            )

        self.assertEqual(engine.runtime.twitch.sent, ["Te vi."])
        self.assertIn("[HEBE][FOLLOWUP_QUESTION_GUARD] allowed=false action=repair", "\n".join(logs))

    def test_twitch_depth_guard_suppresses_instead_of_truncating_unrepaired_content(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        long_reply = "Primero corta todo. Segundo mezcla ingredientes. Tercero hornea media hora. Cuarto deja reposar."
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            engine._deliver_twitch_reply(
                long_reply,
                event_type="twitch_chat_react",
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe, receta rapida para cocinar algo?"},
            )

        self.assertEqual(engine.runtime.twitch.sent, [])
        self.assertIn("[HEBE][TWITCH_ANSWER_DEPTH] action=repair", "\n".join(logs))

    def test_candidate_not_broadcast_before_route(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        emitted = []

        with patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            engine._deliver_twitch_reply(
                "Te leo.",
                event_type="twitch_chat_react",
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe xd"},
            )

        self.assertEqual(engine.runtime.twitch.sent, [])
        self.assertFalse(any(event_type == "chat.assistant" for event_type, _ in emitted))

    def test_observe_only_twitch_message_no_model_call(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        engine.context_builder = Mock()
        engine.response_synthesizer = Mock()
        logs = []
        event = SimpleNamespace(
            event_type="twitch_chat_react",
            payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe xd"},
        )

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            engine.process_internal_event(event)

        joined = "\n".join(logs)
        engine.context_builder.build.assert_not_called()
        engine.response_synthesizer.synthesize.assert_not_called()
        self.assertIn("[HEBE][PRE_GENERATION_ROUTE] should_generate=false route=observe_only reason=low_value_banter", joined)
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_twitch_normal_no_mention_chat_reaches_presence_observe(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        engine.context_builder = Mock()
        engine.response_synthesizer = Mock()
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            engine.handle_twitch_chat_event(
                username="viewer",
                display_name="Viewer",
                text="estoy mirando la partida tranquilamente",
                channel="#chan",
            )

        joined = "\n".join(logs)
        engine.context_builder.build.assert_not_called()
        engine.response_synthesizer.synthesize.assert_not_called()
        self.assertIn("[HEBE][TWITCH_PIPELINE_START]", joined)
        self.assertIn("[HEBE][TWITCH_PIPELINE_CLASSIFY] category=normal_no_mention_chat", joined)
        self.assertIn("[HEBE][PRESENCE_ENGINE] source=twitch_chat", joined)
        self.assertIn("[HEBE][TWITCH_PIPELINE_FINAL] route=observe_only emitted=false reason=normal_no_mention_chat", joined)
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_high_value_game_tip_can_reply_without_hebe_mention(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("buena pista; eso encaja con la ruta")
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), \
             patch("app.hebe_engine.log_chat"):
            engine.handle_twitch_chat_event(
                username="viewer",
                display_name="Viewer",
                text="pista de juego: revisa la cueva antes del jefe",
                channel="#chan",
            )

        joined = "\n".join(logs)
        self.assertEqual(len(engine.response_synthesizer.calls), 1)
        self.assertEqual(engine.runtime.twitch.sent, ["buena pista; eso encaja con la ruta"])
        self.assertIn("[HEBE][TWITCH_PIPELINE_CLASSIFY] category=high_value_game_tip", joined)
        self.assertIn("[HEBE][PRESENCE_ENGINE] source=twitch_chat should_intervene=true", joined)
        self.assertIn("[HEBE][TWITCH_PIPELINE_FINAL] route=twitch_text_reply emitted=true", joined)

    def test_twitch_pipeline_health_counts_messages(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        engine.context_builder = Mock()
        engine.response_synthesizer = Mock()

        with patch("builtins.print"):
            engine.handle_twitch_chat_event(
                username="viewer",
                display_name="Viewer",
                text="estoy mirando la partida tranquilamente",
                channel="#chan",
            )

        health = engine.runtime.state.stream.twitch_pipeline_health
        self.assertEqual(health["twitch_messages_received"], 1)
        self.assertEqual(health["twitch_messages_presence_evaluated"], 1)
        self.assertEqual(health["twitch_messages_observe_only"], 1)
        self.assertEqual(health["twitch_messages_final_emitted"], 0)

    def test_pre_generation_observe_no_model_call_logs_route_gate(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        engine.context_builder = Mock()
        engine.response_synthesizer = Mock()
        logs = []
        event = SimpleNamespace(
            event_type="twitch_chat_react",
            payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe xd"},
        )

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            engine.process_internal_event(event)

        joined = "\n".join(logs)
        engine.context_builder.build.assert_not_called()
        engine.response_synthesizer.synthesize.assert_not_called()
        self.assertIn("[HEBE][PRE_GENERATION_ROUTE_GATE] should_generate=false route=observe_only reason=low_value_banter", joined)

    def test_hebe_own_twitch_message_ignored(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        engine.context_builder = Mock()
        engine.response_synthesizer = Mock()
        logs = []
        event = SimpleNamespace(
            event_type="twitch_chat_react",
            payload={"user_login": "HebeNifelheim", "display_name": "Hebe", "message_text": "mensaje propio"},
        )

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            engine.process_internal_event(event)

        engine.context_builder.build.assert_not_called()
        engine.response_synthesizer.synthesize.assert_not_called()
        self.assertEqual(engine.runtime.twitch.sent, [])
        self.assertIn("[HEBE][SELF_MESSAGE_IGNORED] username=HebeNifelheim", "\n".join(logs))

    def test_owner_typed_so_command_no_llm_reply(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        engine.context_builder = Mock()
        engine.response_synthesizer = Mock()
        event = SimpleNamespace(
            event_type="twitch_chat_react",
            payload={"user_login": "leonifelheim", "display_name": "Leo", "message_text": "!so @laurichutv"},
        )

        engine.process_internal_event(event)

        engine.context_builder.build.assert_not_called()
        engine.response_synthesizer.synthesize.assert_not_called()
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_owner_typed_so_command_not_viewer_react(self):
        engine = make_engine(["nuria"])

        decision = engine._pre_generation_twitch_route_decision(
            payload={"user_login": "leonifelheim", "display_name": "Leo", "message_text": "!so @laurichutv"},
            event_type="twitch_chat_react",
            stream=engine.runtime.state.stream,
        )

        self.assertFalse(decision["should_generate"])
        self.assertEqual(decision["route"], "twitch_action_observed")
        self.assertEqual(decision["reason"], "owner_manual_twitch_command")

    def test_hebe_own_message_ignored(self):
        engine = make_engine(["nuria"])

        decision = engine._pre_generation_twitch_route_decision(
            payload={"user_login": "HebeNifelheim", "display_name": "HebeNifelheim", "message_text": "texto del bot"},
            event_type="twitch_chat_react",
            stream=engine.runtime.state.stream,
        )

        self.assertFalse(decision["should_generate"])
        self.assertEqual(decision["reason"], "bot_message")

    def test_jotunbot_so_response_ignored(self):
        engine = make_engine(["nuria"])

        decision = engine._pre_generation_twitch_route_decision(
            payload={"user_login": "JotunBot", "display_name": "JotunBot", "message_text": "Shoutout enviado"},
            event_type="twitch_chat_react",
            stream=engine.runtime.state.stream,
        )

        self.assertFalse(decision["should_generate"])
        self.assertEqual(decision["reason"], "bot_message")

    def test_hebe_voice_guard_rejects_generic_hype(self):
        engine = make_engine(["nuria"])

        result = engine._stream_persona_quality_guard(
            "sending love and raid energy to laurichutv, go make that dinner",
            category="viewer_talks_about_hebe",
            event_type="twitch_chat_react",
            payload={},
        )

        self.assertFalse(result["passed"])
        self.assertIn("generic_english_hype", result["violations"])

    def test_budget_blocked_twitch_message_no_model_call(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        text = "Hebe, que opinas de esto?"
        thread_id = engine._twitch_thread_id(username="viewer", text=text, category="direct_open_prompt_to_hebe")
        engine.runtime.state.stream.public_reply_thread_counts[thread_id] = 2
        engine.context_builder = Mock()
        engine.response_synthesizer = Mock()
        logs = []
        event = SimpleNamespace(
            event_type="twitch_chat_react",
            payload={"user_login": "viewer", "display_name": "Viewer", "message_text": text},
        )

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            engine.process_internal_event(event)

        joined = "\n".join(logs)
        engine.context_builder.build.assert_not_called()
        engine.response_synthesizer.synthesize.assert_not_called()
        self.assertIn("[HEBE][PRE_GENERATION_ROUTE] should_generate=false route=observe_only reason=thread_closed", joined)
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_talks_about_hebe_can_intervene_if_social_value_high(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            decision = engine._pre_generation_twitch_route_decision(
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe esta muy callada"},
                event_type="twitch_chat_react",
                stream=engine.runtime.state.stream,
            )

        joined = "\n".join(logs)
        self.assertTrue(decision["should_generate"])
        self.assertEqual(decision["category"], "viewer_talks_about_hebe")
        self.assertEqual(decision["route"], "twitch_text_reply")
        self.assertIn("[HEBE][PRESENCE_ENGINE] source=twitch_chat should_intervene=true", joined)

    def test_direct_hebe_open_prompt_reaches_presence_and_generation(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            decision = engine._pre_generation_twitch_route_decision(
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "hebe que opinas"},
                event_type="twitch_chat_react",
                stream=engine.runtime.state.stream,
            )

        joined = "\n".join(logs)
        self.assertTrue(decision["should_generate"])
        self.assertEqual(decision["category"], "direct_open_prompt_to_hebe")
        self.assertEqual(decision["route"], "twitch_text_reply")
        self.assertEqual(decision["twitch_message_category"], "direct_open_prompt_to_hebe")
        self.assertIn("[HEBE][PRESENCE_ENGINE] source=twitch_chat should_intervene=true", joined)
        self.assertIn("[HEBE][INTERVENTION_DECISION] source=twitch_chat", joined)

    def test_direct_hebe_prompt_dispatches_after_cognitive_router(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        engine.response_synthesizer = FixedResponseSynth("opinion corta con filo")
        logs = []
        event = SimpleNamespace(
            event_type="twitch_chat_react",
            payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "hebe que opinas"},
        )

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), \
             patch("app.hebe_engine.log_chat"):
            engine.process_internal_event(event)

        joined = "\n".join(logs)
        self.assertEqual(len(engine.response_synthesizer.calls), 1)
        self.assertEqual(engine.runtime.twitch.sent, ["opinion corta con filo"])
        self.assertIn("[HEBE][POST_ROUTER_DISPATCH]", joined)
        self.assertIn("[HEBE][PRESENCE_ENGINE] source=twitch_chat", joined)
        self.assertIn("[HEBE][OUTPUT_ROUTE_DECISION] route=twitch_text_reply", joined)

    def test_directed_casual_question_recovers_after_generic_fallback_and_is_public(self):
        engine = make_engine(["viewer"])
        engine.runtime.state.stream.is_live = True
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        model = SequentialResponseModel([
            "",
            "",
            "",
            "La creatividad suele volver cuando cambias una rutina pequeña y pruebas sin exigirte perfección.",
        ])
        engine.response_synthesizer = ResponseSynthesizer(conversation_model=model)
        engine.response_synthesizer._dataset_logger.log_twitch_chat_react = lambda **kwargs: None
        logs = []
        event = SimpleNamespace(
            event_type="twitch_chat_react",
            payload={
                "event_id": "evt-directed-recovery",
                "user_login": "viewer",
                "display_name": "Viewer",
                "message_text": "Hebe, ¿cómo puedo recuperar la creatividad?",
            },
        )

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), \
             patch("app.hebe_engine.log_chat"):
            engine.process_internal_event(event)

        firewall = engine._firewall_payload()
        joined = "\n".join(logs)
        self.assertEqual(firewall["firewall_decision"], "allow")
        self.assertTrue(firewall["addressed_to_hebe"])
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(len(engine.runtime.twitch.sent), 1)
        self.assertIn("creatividad", engine.runtime.twitch.sent[0].casefold())
        self.assertIn("[HEBE][POST_ROUTER_DISPATCH]", joined)
        self.assertIn("[HEBE][DIRECTED_VIEWER_RESPONSE_OUTCOME] outcome=regenerated", joined)
        self.assertNotIn("generic_ack_twitch_fallback", joined)

    def test_directed_generation_failure_sends_one_terminal_fallback_publicly(self):
        engine = make_engine(["viewer"])
        engine.runtime.state.stream.is_live = True
        engine.context_builder = FakeContextBuilder()
        engine.deliberation_service = FakeDeliberationService()
        engine.plan_executor = FakePlanExecutor()
        model = SequentialResponseModel(["", "", "", "", "", ""])
        engine.response_synthesizer = ResponseSynthesizer(conversation_model=model)
        engine.response_synthesizer._dataset_logger.log_twitch_chat_react = lambda **kwargs: None
        logs = []
        event = SimpleNamespace(
            event_type="twitch_chat_react",
            payload={
                "event_id": "evt-directed-terminal-fallback",
                "user_login": "viewer",
                "display_name": "Viewer",
                "message_text": "Hebe, ¿qué harías para salir de un bloqueo creativo?",
            },
        )

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), \
             patch("app.hebe_engine.log_chat"):
            engine.process_internal_event(event)

        joined = "\n".join(logs)
        self.assertEqual(len(model.calls), 6)
        self.assertEqual(
            engine.runtime.twitch.sent,
            ["Viewer, no tengo una buena respuesta para eso; prefiero no improvisarte humo."],
        )
        self.assertIn("generation=failed outcome=terminal_fallback", joined)
        self.assertIn("[HEBE][TWITCH_PIPELINE_FINAL] route=twitch_text_reply emitted=true", joined)
        self.assertNotIn("generic_ack_twitch_fallback", joined)

    def test_input_firewall_twitch_preserves_raw_text_and_addressing(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True

        engine._input_firewall_decision(
            source="twitch_viewer",
            text="hebe que opinas",
            username="viewer",
            event_type="twitch_chat_react",
            addressed_to_hebe=True,
        )

        payload = engine._firewall_payload()
        self.assertEqual(payload["raw_text"], "hebe que opinas")
        self.assertEqual(payload["normalized_text"], "hebe que opinas")
        self.assertTrue(payload["addressed_to_hebe"])
        self.assertTrue(payload["mentions_hebe"])

    def test_repeated_hebe_talk_observed_after_saturation(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        text = "Hebe esta muy callada"
        thread_id = engine._twitch_thread_id(username="viewer", text=text, category="viewer_talks_about_hebe")
        engine.runtime.state.stream.public_reply_thread_counts[thread_id] = 2
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            decision = engine._pre_generation_twitch_route_decision(
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": text},
                event_type="twitch_chat_react",
                stream=engine.runtime.state.stream,
            )

        joined = "\n".join(logs)
        self.assertFalse(decision["should_generate"])
        self.assertEqual(decision["route"], "observe_only")
        self.assertEqual(decision["reason"], "thread_closed")
        self.assertIn("[HEBE][SOCIAL_BUDGET] allowed=false reason=thread_closed", joined)

    def test_generic_reply_repaired_or_suppressed(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            engine._deliver_twitch_reply(
                "perfecto, sigue asi",
                event_type="twitch_chat_react",
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe, que opinas de esto?"},
            )

        self.assertEqual(engine.runtime.twitch.sent, [])
        self.assertIn("[HEBE][STREAM_PERSONA_QUALITY_GUARD] passed=false", "\n".join(logs))

    def test_public_twitch_generic_te_leo_ack_suppressed(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            engine._deliver_twitch_reply(
                "te leo, Viewer.",
                event_type="twitch_chat_react",
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe, que opinas?"},
            )

        joined = "\n".join(logs)
        self.assertEqual(engine.runtime.twitch.sent, [])
        self.assertIn("generic_ack_twitch_fallback", joined)
        self.assertIn("[HEBE][OUTPUT_ROUTE_DECISION] route=suppress", joined)

    def test_text_only_route_split(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.is_live = True
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            engine._deliver_twitch_reply(
                "Respuesta con valor.",
                event_type="twitch_chat_react",
                payload={"user_login": "viewer", "display_name": "Viewer", "message_text": "Hebe, que opinas de esto?"},
            )

        joined = "\n".join(logs)
        self.assertIn("[HEBE][OUTPUT_ROUTE_DECISION] route=twitch_text_reply", joined)
        self.assertNotIn("route=text_only", joined)

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

    def test_casual_monologue_not_game_guidance_followup(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        open_test_conversation(
            engine, kind="game_guidance_clarification", expected_reply_type="game_progress_state",
            game="Persona 5 Royal", missing_fields=["current_location"], ttl_seconds=300,
        )
        engine.context_builder = FakeContextBuilder()
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit"), \
             patch("app.hebe_engine.log_chat"):
            result = engine._process_stt_voice_transcript("En plan tranquilamente podiamos ser 40")

        joined = "\n".join(logs)
        self.assertEqual(result, "continue")
        self.assertEqual(engine.context_builder.inputs, [])
        self.assertIsNotNone(engine._active_current_conversation())
        self.assertIn("[HEBE][GAME_PENDING_COMPAT] compatible=false", joined)
        self.assertIn("ordinary_stream_or_real_life_talk", joined)

    def test_compatible_game_progress_followup_accepted(self):
        engine = make_engine(["nuria"], live=True)
        engine.runtime.state.stream.enabled = True
        open_test_conversation(
            engine, kind="game_guidance_clarification", expected_reply_type="game_progress_state",
            game="Persona 5 Royal", missing_fields=["current_location"], ttl_seconds=300,
        )
        event = engine._build_input_event(
            source="stt_voice",
            raw_text="Hebe, estoy en el palacio de Kamoshida",
            normalized_text=engine._normalize_text("Hebe, estoy en el palacio de Kamoshida"),
            stt_metadata={"command_mode": True},
        )
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            envelope = engine._build_stt_input_envelope(event, voice_type="direct_command_to_hebe", conversation_followup=False)

        self.assertTrue(envelope.pending_compatible)
        self.assertEqual(envelope.source, "owner_stt_followup")
        self.assertIn("[HEBE][GAME_PENDING_COMPAT] compatible=true", "\n".join(logs))

    def test_game_run_state_write_guard_rejects_stt_junk_and_keeps_previous_state(self):
        engine = wire_canonical_game_pipeline(make_engine(["nuria"]))
        engine._apply_game_run_state_execution(StepExecutionResult(
            step_type="state_update",success=True,data={
                "kind":"game_run_state","pending_id":"pending-game-seed",
                "updates":{"game":"Persona 5 Royal","current_location":"Palacio de Kamoshida",
                           "provenance":"leo_clarification","confidence":0.92},
            },
        ))
        state_update = StepExecutionResult(
            step_type="state_update",
            success=True,
            data={
                "kind": "game_run_state",
                "pending_id": "pending-game",
                "updates": {
                    "current_location": "Hacer artes marciales ver",
                    "current_character": "Rango eh",
                    "provenance": "leo_clarification",
                    "confidence": 0.92,
                },
            },
        )
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            engine._apply_game_run_state_execution(state_update)

        run = engine.runtime.state.game_run_state
        self.assertEqual(run.current_location, "Palacio de Kamoshida")
        self.assertEqual(run.current_character, "")
        joined = "\n".join(logs)
        self.assertIn("[HEBE][GAME_RUN_STATE_WRITE_GUARD] accepted=false field=current_location", joined)
        self.assertIn("[HEBE][GAME_PENDING] state_update_rejected", joined)

    def test_game_run_state_write_guard_accepts_known_persona_location(self):
        engine = wire_canonical_game_pipeline(make_engine(["nuria"]))
        state_update = StepExecutionResult(
            step_type="state_update",
            success=True,
            data={
                "kind": "game_run_state",
                "pending_id": "pending-game",
                "updates": {
                    "game": "Persona 5 Royal",
                    "current_location": "Palacio de Kamoshida",
                    "provenance": "leo_clarification",
                    "confidence": 0.92,
                },
            },
        )

        engine._apply_game_run_state_execution(state_update)

        self.assertEqual(engine.runtime.state.game_run_state.current_location, "Palacio de Kamoshida")

    def test_single_letter_shoutout_target_requires_clarification_without_alias(self):
        engine = make_engine(["nuria", "charlie"])

        result = engine._handle_stream_manual_command("shoutout a c")

        self.assertEqual(result.action_type, "twitch_shoutout_clarify")
        self.assertEqual(result.metadata["action_plan"]["reason"], "ambiguous_single_letter_target")
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_conversation_not_created_for_twitch_source(self):
        engine = make_engine(["nuria"])

        engine._record_assistant_reply_for_conversation("¿tú qué tal?", source="twitch_chat_react", synthesizer=pending_marker())

        self.assertIsNone(engine._active_current_conversation())

    def test_pending_conversation_expires_after_ttl(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = False
        engine.pending_conversation_ttl_seconds = 1

        with patch("time.time", return_value=1000.0):
            engine._record_assistant_reply_for_conversation("Â¿tÃº quÃ© tal?", source="stt_voice", synthesizer=pending_marker())
            conversation = engine._active_current_conversation()
        with patch("time.time", return_value=1002.0):
            self.assertFalse(engine._pending_conversation_matches(source="stt_voice"))

        stored = engine.conversation_continuity.conversations.get(conversation.id)
        self.assertEqual(stored.status.value, "EXPIRED")

    def test_unrelated_action_during_pending_conversation_still_uses_action_flow(self):
        engine = wire_canonical_app_pipeline(make_engine(["nuria"]))
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))
        engine._record_assistant_reply_for_conversation("Â¿tÃº quÃ© tal?", source="stt_voice", synthesizer=pending_marker())

        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": sys.executable}), \
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

    def test_unsupported_script_uses_dual_recovery_and_routes_valid_latin_result(self):
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
        self.assertEqual(engine.runtime.stt.calls, [{"initial_language": "devanagari"}])
        self.assertEqual(routed[0][0], "stt_voice")
        self.assertEqual(routed[0][1], "abre obs")
        self.assertEqual(routed[0][2].raw_text, "à¤¯à¤¬ à¤†à¤¬à¤°à¥‡ à¤…à¤¬à¥‡ à¤¯à¤¶à¥‡")
        self.assertEqual(routed[0][2].normalized_text, "hebe abre obs")
        self.assertIn("[HEBE][STT][RETRY] reason=unsupported_script policy=dual_decode_then_drop", joined)
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
        engine.runtime.stt = RetrySTT(
            "à¤¯à¤¬ à¤†à¤¬à¤°à¥‡ à¤…à¤¬à¥‡ à¤¯à¤¶à¥‡",
            accepted=False,
        )
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
        self.assertIn("[HEBE][INPUT_CLASSIFY] source=ambient_stt input_type=ambient_stream_context", joined)
        self.assertIn("[HEBE][RESPONSE_DECISION] should_reply=false reason=no_ignore", joined)
        self.assertIn("[HEBE][COG] decision=ambient_ignored_low_value reason=ambient_context_only", joined)

    def test_ambient_stt_ignored_by_firewall_never_enters_cognitive_flow_or_actions(self):
        engine = make_engine(["nuria"])
        engine.runtime.state.stream.enabled = False
        engine.stream_ambient_stt_enabled = False
        engine.cognitive_flow = Mock()
        engine.action_runtime.execute = Mock()

        result = engine._process_stt_voice_transcript("Estoy comentando algo al fondo")

        self.assertEqual(result, "continue")
        engine.cognitive_flow.assert_not_called()
        engine.action_runtime.execute.assert_not_called()

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
        self.assertEqual(plan.target, "nuriiia___")

    def test_awake_shoutout_without_despierta_enters_cognition(self):
        engine = make_engine(["nuria"])
        delivered = []
        engine._deliver_manual_reply = lambda text, *, source: delivered.append((source, text))

        result = engine.cognitive_flow("haz promo a nuria", source="stt_voice")

        self.assertEqual(result, "continue")
        self.assertEqual(engine.runtime.twitch.sent, ["!so nuriiia___"])
        self.assertEqual(delivered, [("stt_voice", "Promo hecha para nuriiia___.")])

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
        self.assertEqual(engine.runtime.twitch.sent, ["!so nuriiia___"])
        self.assertEqual(delivered[0], ("stt_voice", "modelo:wake_from_sleep:"))


if __name__ == "__main__":
    unittest.main()
