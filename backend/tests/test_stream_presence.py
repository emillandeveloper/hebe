import os
import time
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.hebe_engine import HebeEngine
from app.services import db_sqlite
from app.stream.context_sync import StreamContextSyncService
from app.stream.game_research import GameKnowledgeResearchConfig, GameKnowledgeResearchService
from app.stream.game_profiles import GameProfileStore
from app.stream import memory as stream_memory
from app.stream.state import StreamSessionState
from app.stream.spontaneity import StreamSpontaneityConfig, StreamSpontaneityService
from app.cognitive.scheduler import InternalEvent


class FakeTwitch:
    def __init__(self):
        self.sent = []
        self.channel_name = "leonifelheim"
        self.bot_username = "HebeNifelheim"
        self.shoutout_command_template = "!so {username}"
        self.helix_client = SimpleNamespace(
            broadcaster_id="124070929",
            client_id="client",
            channel_name="leonifelheim",
        )
        self.chat_client = SimpleNamespace(sender_id="1480877711")

    def is_available(self):
        return True

    def send_message(self, text):
        self.sent.append(text)

    def normalize_twitch_username(self, username):
        value = str(username or "").strip().lstrip("@").strip()
        return value.replace(" ", "") if value else ""

    def build_shoutout_command(self, username):
        return self.shoutout_command_template.format(username=self.normalize_twitch_username(username))

    def shoutout(self, username):
        self.send_message(self.build_shoutout_command(username))
        return True

    def resolve_user(self, raw_target):
        return None


class FailingShoutoutTwitch(FakeTwitch):
    def shoutout(self, username):
        raise RuntimeError("boom")


class FakeSynth:
    def generate_stream_presence(self, **kwargs):
        return "Antes de avanzar, mira equipo. La epica tambien paga facturas."

    def generate_twitch_idle_prompt_preview(self, payload):
        return "First playthrough significa sospechar de todo NPC amable. Es ley."


class FakeSearchProvider:
    def __init__(self):
        self.calls = []

    def search(self, query):
        self.calls.append(query)
        return [{
            "title": "Spoiler-free gameplay overview",
            "snippet": "Spoiler-free overview with turn-based combat, equipment abilities, whimsical theatrical fantasy, and party resources.",
            "url": "https://example.com/ff9",
        }]


def make_engine(stream=None):
    engine = HebeEngine.__new__(HebeEngine)
    stream = stream or StreamSessionState()
    engine.runtime = SimpleNamespace(
        state=SimpleNamespace(stream=stream),
        twitch=FakeTwitch(),
        twitch_chat_bot=SimpleNamespace(is_connected=True),
        twitch_events=SimpleNamespace(is_connected=False),
        speak=Mock(),
    )
    engine.game_profiles = GameProfileStore()
    engine.game_research = GameKnowledgeResearchService(
        store=engine.game_profiles,
        config=GameKnowledgeResearchConfig(enabled=False),
    )
    engine._last_game_research_category = None
    engine.response_synthesizer = FakeSynth()
    engine.stream_spontaneity = StreamSpontaneityService(
        config=StreamSpontaneityConfig(companion_jitter_sec=0, show_jitter_sec=0),
        game_profiles=engine.game_profiles,
    )
    engine._last_presence_poll_ts = 0.0
    engine.presence_poll_interval_sec = 0.0
    engine._last_stream_context_poll_ts = 0.0
    engine.stream_context_poll_interval_sec = 90.0
    engine.stream_context_sync = None
    engine.stream_ambient_stt_enabled = True
    engine.stream_observe_chat = True
    engine.chat_activity_window_sec = 180
    engine.chat_active_message_threshold = 3
    engine.chat_active_user_threshold = 1
    engine.idle_suppress_when_chat_active = True
    engine._manual_reply_ui_only = False
    engine._last_routine_poll_ts = 0.0
    engine.routine_poll_interval_sec = 30.0
    engine.auto_enable_stream_when_live = True
    engine.default_live_presence_mode = "companion"
    engine.auto_shoutout_raiders = True
    engine.shoutout_cooldown_seconds = 120
    engine.shoutout_allow_bots = False
    engine.shoutout_blocked_users = engine._load_shoutout_blocked_users()
    capabilities = {"audio.tts_control", "pending.cancel", "stream.local_state_control", "twitch_action", "hebe.wake_control"}
    engine._active_cognitive_decision = SimpleNamespace(
        authority="owner", source="ui", should_stop_pipeline=False,
        allowed_step_types=["state_update", "action", "reply"],
        action_permission_summary={"stream_live": True},
        allows_capability=lambda capability: capability in capabilities,
    )
    return engine


class FakeContextSync:
    def __init__(self, ok=True, live=None):
        self.ok = ok
        self.live = live
        self.calls = []

    def sync(self, stream):
        self.calls.append(stream)
        if not self.ok:
            stream.last_stream_context_error = "Helix get_streams failed: 401 Unauthorized."
        if self.live is not None:
            stream.is_live = bool(self.live)
            stream.live_status_known = True
            stream.stream_context_updated_ts = time.time()
        return self.ok


class StreamPresenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp_db = tempfile.TemporaryDirectory()
        self.old_db_path = db_sqlite.DB_PATH
        db_sqlite.DB_PATH = os.path.join(self.tmp_db.name, "hebe_stream_presence.sqlite3")
        stream_memory._READY_DB_PATH = None
        stream_memory.init_stream_memory_schema()

    def tearDown(self):
        db_sqlite.DB_PATH = self.old_db_path
        stream_memory._READY_DB_PATH = None
        self.tmp_db.cleanup()

    def test_stream_presence_default_is_reactive(self):
        stream = StreamSessionState()

        self.assertEqual(stream.presence_mode, "reactive")

    def test_presence_mode_command_sets_companion(self):
        engine = make_engine()

        reply = engine._handle_stream_manual_command("Hebe, modo compañera")

        self.assertEqual(engine.runtime.state.stream.presence_mode, "companion")
        self.assertIn("compañera", reply)

    def test_manual_no_stream_suppresses_today(self):
        engine = make_engine()

        reply = engine._handle_stream_manual_command("Hebe, hoy no hay stream")

        self.assertFalse(engine.runtime.state.stream.enabled)
        self.assertIsNotNone(engine.runtime.state.stream.no_stream_today_date)
        self.assertIn("no insisto", reply)

    def test_voice_comment_updates_context_but_is_not_command(self):
        engine = make_engine(StreamSessionState(enabled=True))

        event_type, mood = engine._classify_voice_event("me han matado otra vez")
        engine._record_voice_event("me han matado otra vez", event_type, mood)

        stream = engine.runtime.state.stream
        self.assertEqual(event_type, "gameplay_failure")
        self.assertEqual(stream.last_voice_event, "gameplay_failure")
        self.assertEqual(stream.leo_mood_hint, "frustrated")

    def test_direct_voice_command_to_hebe_is_detected(self):
        engine = make_engine(StreamSessionState(enabled=True))

        event_type, mood = engine._classify_voice_event("hebe prepara stream")

        self.assertEqual(event_type, "direct_command_to_hebe")
        self.assertIsNone(mood)

    def test_performance_profile_replaces_frozen_spontaneity_config(self):
        stream = StreamSessionState(enabled=True)
        stream.current_game = "Baldur's Gate 3"
        engine = make_engine(stream)

        engine._apply_stream_performance_profile()

        self.assertGreaterEqual(engine.stream_spontaneity.config.global_stream_cooldown_sec, 10 * 60)

    def test_direct_voice_tts_uses_stream_target_when_live(self):
        stream = StreamSessionState(enabled=False)
        stream.is_live = True
        stream.live_status_known = True
        engine = make_engine(stream)

        self.assertEqual(engine._direct_voice_tts_target(), "stream_tts")

    def test_ambient_voice_without_wakeword_does_not_become_command(self):
        engine = make_engine(StreamSessionState(enabled=True))

        handled, command = engine._extract_stream_command("me han matado otra vez")

        self.assertTrue(handled)
        self.assertIsNone(command)

    def test_ambient_voice_updates_lightweight_context_only(self):
        engine = make_engine(StreamSessionState(enabled=True))
        event_type, mood = engine._classify_voice_event("no entiendo donde voy ahora")

        engine._record_voice_event("no entiendo donde voy ahora", event_type, mood)

        stream = engine.runtime.state.stream
        self.assertEqual(stream.last_voice_event, "confusion/lost")
        self.assertEqual(stream.leo_mood_hint, "confused")
        self.assertEqual(stream.last_voice_summary, "no entiendo donde voy ahora")

    def test_ambient_stt_level_gap_updates_run_context(self):
        engine = make_engine(StreamSessionState(enabled=True))
        event_type, mood = engine._classify_voice_event("ahora mismo son cuatro niveles más que yo")

        engine._record_voice_event("ahora mismo son cuatro niveles más que yo", event_type, mood)

        stream = engine.runtime.state.stream
        facts = stream.recent_run_context_facts
        self.assertTrue(any(item.get("kind") == "level_gap" for item in facts))
        self.assertEqual(stream.run_context_source, "stt_voice")
        self.assertIn("4 levels", stream.current_run_phase)

    def test_unsupported_script_stt_is_rejected_before_cognition(self):
        engine = make_engine(StreamSessionState(enabled=True))
        logs = []
        emitted = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))), \
             patch("app.hebe_engine.emit", lambda event_type, data=None: emitted.append((event_type, data or {}))):
            result = engine._process_stt_voice_transcript("これはテストです")

        self.assertEqual(result, "continue")
        self.assertIn("[HEBE][STT][REJECTED] reason=unsupported_script script=japanese", "\n".join(logs))
        self.assertTrue(any(data.get("message") == "Raw STT rejected: unsupported language/script." for _, data in emitted))

    def test_unsupported_script_detector_covers_non_latin_families(self):
        engine = make_engine(StreamSessionState(enabled=True))

        self.assertEqual(engine._unsupported_stt_script("привет тест"), "cyrillic")
        self.assertEqual(engine._unsupported_stt_script("δοκιμή"), "greek")
        self.assertEqual(engine._unsupported_stt_script("தமிழ்"), "tamil")

    def test_chat_message_without_mention_updates_activity_without_reply(self):
        stream = StreamSessionState(enabled=True)
        stream.is_live = True
        engine = make_engine(stream)

        engine.observe_twitch_chat_message("viewer", "Viewer", "linux y ram hoy", "#chan")

        stream = engine.runtime.state.stream
        self.assertGreater(stream.last_chat_activity_ts, 0)
        self.assertEqual(len(stream.recent_chat_messages), 1)
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_chat_rng_topic_links_to_recent_run_context_without_reply(self):
        stream = StreamSessionState(enabled=True)
        stream.is_live = True
        now = time.time()
        stream.recent_run_context_facts = [{
            "id": "ambient:rng_dependency:1",
            "kind": "rng_dependency",
            "category": "rng_dependency",
            "text": "Leo framed the current situation as dependent on RNG or luck.",
            "summary": "Leo framed the current situation as dependent on RNG or luck.",
            "confidence": 0.86,
            "timestamp": now,
            "expires_at": now + 600,
        }]
        engine = make_engine(stream)

        engine.observe_twitch_chat_message("nuriiia___", "Nuria", "tan genial el RNG, como tirar dados en el parchis", "#chan")

        entry = stream.recent_chat_messages[-1]
        self.assertEqual(entry["topic"], "rng_dependency")
        self.assertEqual(entry["linked_to_recent_run_context"], "rng_dependency")
        self.assertIn("Nuria", entry["summary"])
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_bot_messages_do_not_count_as_chat_activity(self):
        engine = make_engine(StreamSessionState(enabled=True))

        engine.observe_twitch_chat_message("Nightbot", "Nightbot", "mensaje automatico", "#chan")

        self.assertEqual(engine.runtime.state.stream.last_chat_activity_ts, 0.0)
        self.assertEqual(engine.runtime.state.stream.recent_chat_messages, [])

    def test_bot_reaction_event_stops_before_context_or_model_pipeline(self):
        stream = StreamSessionState(enabled=True)
        stream.is_live = True
        stream.live_status_known = True
        engine = make_engine(stream)
        engine._last_input_firewall = {}
        engine._last_policy_trace = {}
        engine.input_authority_firewall = engine._build_input_firewall()
        engine.context_builder = Mock()
        engine.deliberation_service = Mock()

        engine.process_internal_event(InternalEvent(
            event_type="twitch_chat_react",
            payload={
                "user_login": "Nightbot",
                "display_name": "Nightbot",
                "message_text": "automated channel status",
            },
            created_at="2026-06-20T00:00:00Z",
        ))

        engine.context_builder.build.assert_not_called()
        engine.deliberation_service.deliberate.assert_not_called()

    def test_manual_completed_marker_updates_run_context(self):
        engine = make_engine(StreamSessionState(enabled=True))

        reply = engine._handle_stream_manual_command("Hebe, ya hemos pasado Ramuh")

        self.assertIn("Ramuh", engine.runtime.state.stream.completed_run_markers)
        self.assertIn("Marcador completado", reply)

    def test_manual_objective_updates_run_context(self):
        engine = make_engine(StreamSessionState(enabled=True))

        reply = engine._handle_stream_manual_command("Hebe, objetivo actual: avanzar hasta Burmecia")

        self.assertEqual(engine.runtime.state.stream.current_run_objective, "avanzar hasta Burmecia")
        self.assertIn("Objetivo actual", reply)

    def test_ambient_stt_completed_marker_does_not_reply(self):
        engine = make_engine(StreamSessionState(enabled=True))
        event_type, mood = engine._classify_voice_event("Ya hemos pasado Ramuh")

        engine._record_voice_event("Ya hemos pasado Ramuh", event_type, mood)

        self.assertEqual(event_type, "completed_marker")
        self.assertIn("Ramuh", engine.runtime.state.stream.completed_run_markers)
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_ambient_stt_failure_updates_mood_only(self):
        engine = make_engine(StreamSessionState(enabled=True))
        event_type, mood = engine._classify_voice_event("Joder, me han matado otra vez")

        engine._record_voice_event("Joder, me han matado otra vez", event_type, mood)

        stream = engine.runtime.state.stream
        self.assertEqual(stream.last_voice_event, "gameplay_failure")
        self.assertEqual(stream.leo_mood_hint, "frustrated")
        self.assertIsNone(stream.current_run_objective)

    def test_pausing_idle_spontaneity_does_not_block_raid_thanks(self):
        stream = StreamSessionState(enabled=True, presence_mode="reactive")
        stream.idle_spontaneity_enabled = False
        stream.is_live = True
        engine = make_engine(stream)
        engine._synthesize_internal_event_reply = Mock(return_value="Gracias por la raid.")

        engine._handle_twitch_raid_event(engine._build_local_internal_event("twitch_raid", {
            "display_name": "Raider",
            "viewer_count": 3,
        }))

        self.assertEqual(engine.runtime.twitch.sent, ["Gracias por la raid.", "!so Raider"])

    def test_readiness_reports_chat_active_blocked_reason(self):
        now = time.time()
        stream = StreamSessionState(enabled=True, presence_mode="show")
        stream.is_live = True
        stream.live_status_known = True
        stream.stream_context_updated_ts = now
        stream.last_chat_activity_ts = now - 60 * 60
        stream.recent_chat_messages = [
            {"username": "viewer", "text": "linux", "ts": now - 20, "topic": "tech_pc"},
            {"username": "viewer", "text": "ram", "ts": now - 10, "topic": "tech_pc"},
            {"username": "viewer", "text": "server", "ts": now - 5, "topic": "tech_pc"},
        ]
        engine = make_engine(stream)

        reply = engine._build_spontaneity_readiness_reply(stream)

        self.assertIn("Chat active: yes", reply)
        self.assertIn("Reason if blocked: chat_active", reply)

    def test_presence_speaks_when_context_allows_companion(self):
        stream = StreamSessionState(enabled=True, presence_mode="companion")
        stream.last_chat_activity_ts = time.time() - 25 * 60
        stream.last_hebe_stream_speak_ts = time.time() - 20 * 60
        stream.last_voice_event_ts = time.time() - 60
        stream.is_live = True
        stream.live_status_known = True
        stream.stream_context_updated_ts = time.time()
        engine = make_engine(stream)
        engine.process_internal_event = Mock(
            side_effect=lambda event: engine._deliver_twitch_reply("Antes de avanzar, mira equipo.")
        )

        engine.poll_stream_presence()

        self.assertEqual(
            engine.runtime.twitch.sent,
            ["Antes de avanzar, mira equipo."],
        )

    def test_presence_skips_reactive_mode(self):
        stream = StreamSessionState(enabled=True, presence_mode="reactive")
        stream.last_chat_activity_ts = time.time() - 10 * 60
        stream.is_live = True
        stream.live_status_known = True
        stream.stream_context_updated_ts = time.time()
        engine = make_engine(stream)
        engine.process_internal_event = Mock()

        engine.poll_stream_presence()

        self.assertEqual(engine.runtime.twitch.sent, [])
        engine.process_internal_event.assert_not_called()

    def test_manual_refresh_command_calls_context_sync(self):
        stream = StreamSessionState(enabled=False)
        engine = make_engine(stream)
        sync = FakeContextSync(ok=True)
        engine.stream_context_sync = sync

        reply = engine._handle_stream_manual_command("Hebe, actualiza contexto de stream")

        self.assertEqual(str(reply), "Contexto de stream actualizado.")
        self.assertEqual(sync.calls, [stream])

    def test_context_query_includes_last_stream_context_error(self):
        stream = StreamSessionState()
        stream.last_stream_context_error = "Helix get_streams failed: 401 Unauthorized."
        engine = make_engine(stream)

        reply = engine._handle_stream_manual_command("Hebe, que contexto de stream tienes")

        self.assertIn("Esto tengo ahora mismo:", reply)
        self.assertIn("Ultimo error: Helix get_streams failed: 401 Unauthorized.", reply)

    def test_context_query_includes_parsed_stream_fields(self):
        stream = StreamSessionState()
        stream.live_status_known = True
        stream.is_live = False
        stream.current_category = "Zwei!!: The Arges Adventure"
        stream.current_stream_title = "[ENG/ESP] Retro Weekend: Food for Leveling! That's Zwei | First Playthrough"
        stream.current_playthrough_type = "first_playthrough"
        stream.current_stream_slot = "retro_weekend"
        stream.current_challenge = None
        stream.bilingual_mode = True
        stream.language_mode = "ENG/ESP"
        stream.spoiler_policy = "no_spoilers"
        stream.stream_context_updated_ts = time.time() - 12
        engine = make_engine(stream)

        reply = engine._handle_stream_manual_command("Hebe, que contexto de stream tienes")

        self.assertIn("Estado: offline, comprobado con Twitch.", reply)
        self.assertIn("Juego/categoria: Zwei!!: The Arges Adventure.", reply)
        self.assertIn("Tipo de directo detectado: first_playthrough.", reply)
        self.assertIn("Slot/tema detectado: retro_weekend.", reply)
        self.assertIn("Challenge detectado: ninguno.", reply)
        self.assertIn("Idioma/modo: ENG/ESP.", reply)
        self.assertIn("Spoilers: no_spoilers.", reply)
        self.assertIn("Contexto actualizado: hace", reply)
        self.assertIn("Ultimo error: ninguno.", reply)

    def test_context_query_unknown_fields_are_friendly(self):
        stream = StreamSessionState()
        stream.live_status_known = False
        engine = make_engine(stream)

        reply = engine._handle_stream_manual_command("Hebe, que contexto de stream tienes")

        self.assertIn("Estado: desconocido. No he podido confirmar si el stream esta online.", reply)
        self.assertIn("Tipo de directo detectado: no detectado.", reply)
        self.assertIn("Slot/tema detectado: no detectado.", reply)
        self.assertIn("Challenge detectado: ninguno.", reply)
        self.assertIn("Idioma/modo: no detectado.", reply)
        self.assertIn("Contexto actualizado: nunca.", reply)
        self.assertNotIn("None", reply)
        self.assertNotIn("null", reply)

    def test_context_sync_missing_service_records_error_when_no_runtime_twitch(self):
        stream = StreamSessionState(enabled=True)
        engine = make_engine(stream)
        engine.runtime.twitch = None
        engine.stream_context_sync = StreamContextSyncService(twitch_api=None)

        ok = engine.poll_stream_context(force=True, require_enabled=False)

        self.assertFalse(ok)
        self.assertEqual(stream.last_stream_context_error, "Context sync service not initialized")

    def test_diagnostica_twitch_does_not_expose_tokens(self):
        stream = StreamSessionState()
        engine = make_engine(stream)

        with patch.dict(os.environ, {"TWITCH_OAUTH_TOKEN": "secret-token", "TWITCH_BROADCASTER_OAUTH_TOKEN": "secret-broadcaster"}):
            reply = engine._handle_stream_manual_command("Hebe, diagnostica twitch")

        self.assertIn("TWITCH_OAUTH_TOKEN loaded: yes", reply)
        self.assertIn("TWITCH_BROADCASTER_OAUTH_TOKEN loaded: yes", reply)
        self.assertNotIn("secret-token", reply)
        self.assertNotIn("secret-broadcaster", reply)

    def test_preview_spontaneous_message_works_offline_without_twitch_or_tts(self):
        stream = StreamSessionState(enabled=True, presence_mode="companion")
        stream.is_live = False
        stream.live_status_known = True
        stream.current_stream_title = "First Playthrough"
        stream.current_category = "JRPG"
        engine = make_engine(stream)

        reply = engine._handle_stream_manual_command("Hebe, prueba espontaneidad")

        self.assertIn("Prueba de espontaneidad:", reply)
        self.assertIn("First playthrough", reply)
        self.assertEqual(engine.runtime.twitch.sent, [])
        engine.runtime.speak.assert_not_called()
        self.assertEqual(stream.last_hebe_stream_speak_ts, 0.0)

    def test_readiness_checklist_explains_offline_block(self):
        stream = StreamSessionState(enabled=True, presence_mode="companion")
        stream.is_live = False
        stream.live_status_known = True
        stream.stream_context_updated_ts = time.time()
        engine = make_engine(stream)

        reply = engine._handle_stream_manual_command("Hebe, comprueba espontaneidad")

        self.assertIn("Estado de espontaneidad:", reply)
        self.assertIn("* Twitch live: no", reply)
        self.assertIn("Reason if blocked: stream is offline", reply)

    def test_readiness_checklist_explains_reactive_block(self):
        stream = StreamSessionState(enabled=True, presence_mode="reactive")
        stream.is_live = True
        stream.live_status_known = True
        stream.stream_context_updated_ts = time.time()
        stream.last_chat_activity_ts = time.time() - 60 * 60
        engine = make_engine(stream)

        reply = engine._handle_stream_manual_command("Hebe, estado de espontaneidad")

        self.assertIn("* Presence mode: reactive", reply)
        self.assertIn("Reason if blocked: presence mode is reactive", reply)

    def test_live_test_override_appears_in_readiness_output(self):
        stream = StreamSessionState(enabled=True, presence_mode="companion")
        stream.is_live = False
        stream.live_status_known = True
        stream.live_test_override = True
        stream.stream_context_updated_ts = time.time()
        engine = make_engine(stream)

        reply = engine._handle_stream_manual_command("Hebe, estado de espontaneidad")

        self.assertIn("Simulacion de directo: activada", reply)

    def test_reset_spontaneity_cooldowns_clears_only_spontaneity_keys(self):
        stream = StreamSessionState()
        stream.cooldowns = {
            "stream_idle_prompt_next_ts": 123.0,
            "companion_idle_silence_sec": 1.0,
            "unrelated": 999.0,
        }
        engine = make_engine(stream)

        reply = engine._handle_stream_manual_command("Hebe, resetea cooldowns de espontaneidad")

        self.assertIn("2", reply)
        self.assertEqual(stream.cooldowns, {"unrelated": 999.0})

    def test_stream_online_event_sets_live_and_refreshes_context(self):
        stream = StreamSessionState(enabled=True)
        engine = make_engine(stream)
        engine.poll_stream_context = Mock(return_value=True)

        engine._handle_stream_lifecycle_event(SimpleNamespace(event_type="stream_online", payload={"started_at": "2026-05-31T18:00:00Z"}))

        self.assertTrue(stream.is_live)
        self.assertTrue(stream.live_status_known)
        self.assertEqual(stream.stream_started_at, "2026-05-31T18:00:00Z")
        self.assertGreater(stream.stream_spontaneity_grace_until_ts, time.time())
        engine.poll_stream_context.assert_called_once_with(force=True, require_enabled=False)

    def test_stream_offline_event_sets_offline_and_blocks_spontaneity(self):
        stream = StreamSessionState(enabled=True, presence_mode="companion")
        stream.is_live = True
        stream.live_status_known = True
        stream.stream_context_updated_ts = time.time()
        stream.last_chat_activity_ts = time.time() - 60 * 60
        engine = make_engine(stream)

        engine._handle_stream_lifecycle_event(SimpleNamespace(event_type="stream_offline", payload={}))

        self.assertFalse(stream.is_live)
        event = engine.stream_spontaneity.build_due_event(stream)
        self.assertIsNone(event)

    def test_grace_period_blocks_immediate_spontaneous_message(self):
        stream = StreamSessionState(enabled=True, presence_mode="companion")
        stream.is_live = True
        stream.live_status_known = True
        stream.stream_context_updated_ts = time.time()
        stream.last_chat_activity_ts = time.time() - 60 * 60
        stream.stream_spontaneity_grace_until_ts = time.time() + 60
        engine = make_engine(stream)

        event = engine.stream_spontaneity.build_due_event(stream)

        self.assertIsNone(event)

    def test_context_sync_live_auto_enables_stream_mode(self):
        stream = StreamSessionState(enabled=False, presence_mode="reactive")
        engine = make_engine(stream)
        engine.stream_context_sync = FakeContextSync(ok=True, live=True)

        ok = engine.poll_stream_context(force=True, require_enabled=False)

        self.assertTrue(ok)
        self.assertTrue(stream.enabled)
        self.assertEqual(stream.presence_mode, "companion")
        self.assertGreater(stream.stream_spontaneity_grace_until_ts, time.time())

    def test_live_auto_enable_does_not_override_explicit_silent(self):
        stream = StreamSessionState(enabled=False, presence_mode="silent")
        stream.presence_mode_explicit = True
        engine = make_engine(stream)
        engine.stream_context_sync = FakeContextSync(ok=True, live=True)

        engine.poll_stream_context(force=True, require_enabled=False)

        self.assertTrue(stream.enabled)
        self.assertEqual(stream.presence_mode, "silent")

    def test_ignored_stream_prompt_does_not_prevent_live_auto_enable(self):
        stream = StreamSessionState(enabled=False, presence_mode="reactive")
        engine = make_engine(stream)
        engine.stream_context_sync = FakeContextSync(ok=True, live=True)

        ok = engine.poll_stream_context(force=True, require_enabled=False)

        self.assertTrue(ok)
        self.assertTrue(stream.enabled)

    def test_raid_event_sends_thank_you_in_reactive_mode(self):
        stream = StreamSessionState(enabled=True, presence_mode="reactive")
        stream.is_live = True
        engine = make_engine(stream)
        engine._synthesize_internal_event_reply = Mock(return_value="Gracias por la raid, Raider.")

        engine._handle_twitch_raid_event(SimpleNamespace(
            event_type="twitch_raid",
            payload={"display_name": "Raider", "viewer_count": 5},
        ))

        self.assertEqual(engine.runtime.twitch.sent, ["Gracias por la raid, Raider.", "!so Raider"])
        self.assertEqual(stream.last_raid_event["display_name"], "Raider")

    def test_raid_thank_you_not_blocked_by_idle_cooldown(self):
        stream = StreamSessionState(enabled=True, presence_mode="reactive")
        stream.is_live = True
        stream.cooldowns["stream_idle_prompt_next_ts"] = time.time() + 9999
        engine = make_engine(stream)
        engine._synthesize_internal_event_reply = Mock(return_value="Gracias por la raid, Raider.")

        engine._handle_twitch_raid_event(SimpleNamespace(
            event_type="twitch_raid",
            payload={"display_name": "Raider", "viewer_count": 5},
        ))

        self.assertEqual(engine.runtime.twitch.sent, ["Gracias por la raid, Raider.", "!so Raider"])

    def test_raid_text_sends_when_global_tts_off(self):
        stream = StreamSessionState(enabled=True, presence_mode="reactive")
        stream.is_live = True
        engine = make_engine(stream)
        engine.runtime.state.tts_enabled = False
        engine._synthesize_internal_event_reply = Mock(return_value="Gracias por la raid, Raider.")

        engine._handle_twitch_raid_event(SimpleNamespace(
            event_type="twitch_raid",
            payload={"display_name": "Raider", "viewer_count": 5},
        ))

        self.assertEqual(engine.runtime.twitch.sent, ["Gracias por la raid, Raider.", "!so Raider"])
        engine.runtime.speak.assert_not_called()

    def test_raid_duplicate_does_not_spam_shoutout_within_cooldown(self):
        stream = StreamSessionState(enabled=True, presence_mode="reactive")
        stream.is_live = True
        engine = make_engine(stream)
        engine._synthesize_internal_event_reply = Mock(return_value="Gracias por la raid, Raider.")
        event = SimpleNamespace(
            event_type="twitch_raid",
            payload={"display_name": "Raider", "user_login": "Raider", "viewer_count": 5},
        )

        engine._handle_twitch_raid_event(event)
        engine._handle_twitch_raid_event(event)

        self.assertEqual(engine.runtime.twitch.sent, [
            "Gracias por la raid, Raider.",
            "!so Raider",
            "Gracias por la raid, Raider.",
        ])

    def test_auto_shoutout_uses_configured_template(self):
        stream = StreamSessionState(enabled=True, presence_mode="reactive")
        stream.is_live = True
        engine = make_engine(stream)
        engine.runtime.twitch.shoutout_command_template = "!promo {username}"
        engine._synthesize_internal_event_reply = Mock(return_value="Gracias.")

        engine._handle_twitch_raid_event(SimpleNamespace(
            event_type="twitch_raid",
            payload={"display_name": "Raider", "user_login": "Raider", "viewer_count": 5},
        ))

        self.assertIn("!promo Raider", engine.runtime.twitch.sent)

    def test_manual_shoutout_commands_send_so(self):
        phrases = [
            "Hebe, haz SO a Totodile",
            "Hebe, haz shoutout a Totodile",
            "Hebe, promociona a Totodile",
            "Hebe, haz promo a Totodile",
            "Hebe, give a shoutout to Totodile",
        ]
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                stream = StreamSessionState(enabled=True)
                stream.is_live = True
                engine = make_engine(stream)
                reply = engine._handle_stream_manual_command(phrase)
                self.assertEqual(reply.action_type, "twitch_shoutout")
                self.assertTrue(reply.success)
                self.assertIn("Promo hecha", reply.fallback_text)
                self.assertEqual(engine.runtime.twitch.sent, ["!so Totodile"])

    def test_manual_shoutout_normalizes_at_target(self):
        stream = StreamSessionState(enabled=True)
        stream.is_live = True
        engine = make_engine(stream)

        engine._handle_stream_manual_command("Hebe, haz SO a @Totodile")

        self.assertEqual(engine.runtime.twitch.sent, ["!so Totodile"])

    def test_manual_shoutout_uses_last_raider(self):
        stream = StreamSessionState(enabled=True)
        stream.is_live = True
        stream.last_raid_event = {"user_login": "LastRaider", "display_name": "LastRaider", "ts": time.time()}
        engine = make_engine(stream)

        engine._handle_stream_manual_command("Hebe, haz SO al ultimo raider")

        self.assertEqual(engine.runtime.twitch.sent, ["!so LastRaider"])

    def test_manual_shoutout_without_target_asks_for_clarification(self):
        engine = make_engine(StreamSessionState(enabled=True))

        reply = engine._handle_stream_manual_command("Hebe, haz SO")

        self.assertIn("A quién", reply)
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_blocked_bot_users_do_not_receive_shoutout(self):
        stream = StreamSessionState(enabled=True)
        stream.is_live = True
        engine = make_engine(stream)

        reply = engine._handle_stream_manual_command("Hebe, haz SO a Nightbot")

        self.assertIn("No le hago SO", reply)
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_hebe_does_not_shoutout_herself(self):
        stream = StreamSessionState(enabled=True)
        stream.is_live = True
        engine = make_engine(stream)

        reply = engine._handle_stream_manual_command("Hebe, haz SO a HebeNifelheim")

        self.assertIn("No le hago SO", reply)
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_preview_shoutout_does_not_send_to_twitch(self):
        engine = make_engine(StreamSessionState(enabled=True))

        reply = engine._handle_stream_manual_command("Hebe, previsualiza shoutout a Totodile")

        self.assertIn("!so Totodile", reply)
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_shoutout_command_failure_sets_error(self):
        stream = StreamSessionState(enabled=True)
        stream.is_live = True
        engine = make_engine(stream)
        engine.runtime.twitch = FailingShoutoutTwitch()

        reply = engine._handle_stream_manual_command("Hebe, haz SO a Totodile")

        self.assertIn("No he podido", reply)
        self.assertIn("send_failed", stream.last_shoutout_error)

    def test_simulated_raid_sends_thank_you_and_shoutout(self):
        stream = StreamSessionState(enabled=True, presence_mode="reactive")
        stream.is_live = True
        engine = make_engine(stream)
        engine._synthesize_internal_event_reply = Mock(return_value="Gracias por la raid.")

        reply = engine._handle_stream_manual_command("Hebe, simula raid de Totodile")

        self.assertIn("Raid simulado", reply)
        self.assertEqual(engine.runtime.twitch.sent, [])

    def test_shoutout_status_reports_debug_fields(self):
        engine = make_engine(StreamSessionState(enabled=True))

        reply = engine._handle_stream_manual_command("Hebe, estado de shoutouts")

        self.assertIn("Auto shoutout raiders enabled", reply)
        self.assertIn("!so {username}", reply)

    def test_game_profile_command_returns_readable_info(self):
        stream = StreamSessionState()
        stream.current_category = "Zwei!!: The Arges Adventure"
        engine = make_engine(stream)

        reply = engine._handle_stream_manual_command("Hebe, que perfil de juego estas usando")

        self.assertIn("Perfil spoiler-safe de juego:", reply)
        self.assertIn("Zwei!!: The Arges Adventure", reply)
        self.assertIn("no_spoilers", reply)
        self.assertIn("food leveling", reply)
        self.assertIn("Tono/vibe:", reply)

    def test_manual_research_disabled_uses_local_profile(self):
        stream = StreamSessionState()
        stream.current_category = "Final Fantasy IX"
        engine = make_engine(stream)

        reply = engine._handle_stream_manual_command("Hebe, investiga este juego sin spoilers")

        self.assertIn("Investigacion de juegos desactivada", reply)
        self.assertIn("Final Fantasy IX", reply)

    def test_manual_research_enabled_calls_research_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = StreamSessionState()
            stream.current_category = "One Off Research RPG"
            engine = make_engine(stream)
            engine.game_profiles = GameProfileStore(cache_path=Path(tmp) / "cache.json")
            provider = FakeSearchProvider()
            engine.game_research = GameKnowledgeResearchService(
                store=engine.game_profiles,
                config=GameKnowledgeResearchConfig(enabled=True, provider="fake"),
                search_provider=provider,
                now_fn=lambda: 1_000_000.0,
            )

            reply = engine._handle_stream_manual_command("Hebe, investiga este juego sin spoilers")

        self.assertIn("Conocimiento spoiler-safe actualizado", reply)
        self.assertEqual(len(provider.calls), 1)

    def test_idle_prompt_does_not_trigger_internet_research(self):
        stream = StreamSessionState(enabled=True, presence_mode="show")
        stream.current_category = "Imaginary RPG"
        stream.is_live = True
        stream.live_status_known = True
        stream.stream_context_updated_ts = time.time()
        stream.last_chat_activity_ts = time.time() - 60 * 60
        engine = make_engine(stream)
        provider = FakeSearchProvider()
        engine.game_research = GameKnowledgeResearchService(
            store=engine.game_profiles,
            config=GameKnowledgeResearchConfig(enabled=True, provider="fake"),
            search_provider=provider,
        )

        event = engine.stream_spontaneity.build_due_event(stream)

        self.assertIsNotNone(event)
        self.assertEqual(provider.calls, [])

    def test_reload_game_profiles_command(self):
        engine = make_engine(StreamSessionState())

        reply = engine._handle_stream_manual_command("Hebe, recarga perfiles de juegos")

        self.assertIn("Perfiles de juegos recargados:", reply)


if __name__ == "__main__":
    unittest.main()
