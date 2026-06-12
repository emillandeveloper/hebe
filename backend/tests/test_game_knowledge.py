import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.hebe_engine import HebeEngine
from app.services import db_sqlite
from app.stream import session_primer
from app.stream.context_sync import StreamContextSyncService
from app.stream.game_knowledge import GameKnowledgeConfig, GameKnowledgeResolver
from app.stream.game_profiles import GameProfile, GameProfileStore
from app.stream.game_research import GameKnowledgeResearchConfig, GameKnowledgeResearchService
from app.stream.state import StreamSessionState


class FakeSearchProvider:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [
            {
                "title": "Test Game overview",
                "snippet": "Test Game is a turn-based RPG about exploration, party management, and stylish resource decisions.",
                "url": "https://example.test/test-game",
            }
        ]


class EchoModel:
    def __init__(self):
        self.calls = []

    def complete(self, prompt, **kwargs):
        self.calls.append(prompt)
        return "Persona 5 Royal: tengo perfil spoiler-safe, pero no memoria de la run de Leo."


class GameKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.original_db_path = db_sqlite.DB_PATH
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        self.tmp_path = tmp.name
        db_sqlite.DB_PATH = self.tmp_path
        db_sqlite.init_db()
        session_primer.init_session_primer_schema()
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        db_sqlite.DB_PATH = self.original_db_path
        self.tmp_dir.cleanup()
        try:
            os.unlink(self.tmp_path)
        except OSError:
            pass

    def _store(self, profiles=None):
        path = Path(self.tmp_dir.name) / "profiles.json"
        cache_path = Path(self.tmp_dir.name) / "cache.json"
        rows = [profile.to_dict() if hasattr(profile, "to_dict") else profile for profile in (profiles or [])]
        path.write_text(json.dumps({"profiles": rows}), encoding="utf-8")
        return GameProfileStore(path=path, cache_path=cache_path)

    def _persona_store(self):
        return self._store(
            [
                GameProfile(
                    game_slug="persona_5_royal",
                    canonical_title="Persona 5 Royal",
                    aliases=["Persona 5", "P5R"],
                    genres=["JRPG", "social sim"],
                    general_non_spoiler_summary="Spoiler-safe profile for a stylish JRPG/social sim.",
                    gameplay_systems_non_spoiler=["turn-based combat", "time management"],
                    safe_comment_topics=["calendar pressure", "party preparation"],
                    sources_used=["local_seed_spoiler_safe"],
                    updated_at="2026-06-09T00:00:00+00:00",
                    last_updated_ts=time.time(),
                    confidence=0.9,
                )
            ]
        )

    def test_persona_query_routes_to_game_knowledge_not_chatter_only(self):
        engine = HebeEngine.__new__(HebeEngine)
        stream = StreamSessionState(enabled=True)
        engine.runtime = SimpleNamespace(state=SimpleNamespace(stream=stream))
        engine.game_profiles = self._persona_store()
        engine.game_research = GameKnowledgeResearchService(
            store=engine.game_profiles,
            config=GameKnowledgeResearchConfig(enabled=False),
        )
        engine.game_knowledge = GameKnowledgeResolver(profile_store=engine.game_profiles, research_service=engine.game_research)

        result = engine._handle_game_knowledge_query("Que sabes de Persona 5 Royal?", "que sabes de persona 5 royal", stream)

        self.assertEqual(result.action_type, "game_knowledge_query")
        self.assertEqual(result.state_changes["response_mode"], "profile_only")
        self.assertNotEqual(result.fallback_text.strip().lower(), "no tengo memoria")
        self.assertIn("Spoiler-safe", result.fallback_text)

    def test_profile_only_includes_public_info_and_missing_run_memory(self):
        resolver = GameKnowledgeResolver(profile_store=self._persona_store(), config=GameKnowledgeConfig())

        result = resolver.resolve(game="Persona 5 Royal", stream=StreamSessionState())

        self.assertEqual(result.response_mode, "profile_only")
        self.assertIn("game_profile", result.to_state_changes())
        self.assertIn("personal_session_memory", result.missing)
        self.assertIn("No tengo todavia memoria", result.fallback_text)

    def test_missing_profile_with_web_disabled_offers_seed_or_lookup(self):
        resolver = GameKnowledgeResolver(
            profile_store=self._store(),
            config=GameKnowledgeConfig(web_lookup_enabled=False, game_profile_web_lookup_enabled=False),
        )

        result = resolver.resolve(game="Unknown RPG", stream=StreamSessionState())

        self.assertEqual(result.response_mode, "missing")
        self.assertIn("local_game_profile", result.missing)
        self.assertIn("activar lookup web", result.fallback_text)

    def test_web_enabled_fetches_and_caches_safe_profile(self):
        store = self._store()
        provider = FakeSearchProvider()
        research = GameKnowledgeResearchService(
            store=store,
            config=GameKnowledgeResearchConfig(enabled=True, cache_days=30),
            search_provider=provider,
        )
        resolver = GameKnowledgeResolver(
            profile_store=store,
            research_service=research,
            config=GameKnowledgeConfig(web_lookup_enabled=True, game_profile_web_lookup_enabled=True),
        )

        result = resolver.resolve(game="Test Game", stream=StreamSessionState())

        self.assertEqual(result.response_mode, "profile_only")
        self.assertEqual(result.profile_source, "web_cache")
        self.assertTrue(provider.queries)
        self.assertTrue(store.has_specific_profile(current_game="Test Game"))

    def test_manual_today_game_override_beats_stale_twitch_before_live(self):
        engine = HebeEngine.__new__(HebeEngine)
        stream = StreamSessionState(enabled=True)
        stream.current_game = "FINAL FANTASY IX"
        stream.current_category = "FINAL FANTASY IX"
        engine.runtime = SimpleNamespace(state=SimpleNamespace(stream=stream))

        def command_result(action_type, fallback_text, **kwargs):
            from app.cognitive.command_result import CommandResult

            return CommandResult(
                action_type=action_type,
                success=True,
                user_visible_summary=fallback_text,
                state_changes=kwargs.get("state_changes") or {},
                fallback_text=fallback_text,
                requires_model_response=True,
                metadata={"message_goal": kwargs.get("message_goal") or fallback_text},
            )

        result = engine._handle_stream_session_primer_command(
            "No, hoy toca Persona 5 Royal, Eve.",
            "no hoy toca persona 5 royal eve",
            stream,
            command_result,
        )

        self.assertEqual(result.action_type, "set_today_stream_game")
        self.assertEqual(stream.current_game, "Persona 5 Royal")
        self.assertEqual(getattr(stream, "user_today_game_override"), "Persona 5 Royal")

    def test_offline_twitch_title_does_not_override_manual_correction(self):
        class Twitch:
            def get_stream(self):
                return None

            def get_channel_info(self):
                return {"title": "Iifa Tree Nightmare! Will We Beat It?", "game_name": "FINAL FANTASY IX", "tags": []}

        stream = StreamSessionState(enabled=True)
        stream.user_today_game_override = "Persona 5 Royal"
        service = StreamContextSyncService(twitch_api=Twitch(), now_fn=lambda: 1_000_000)

        ok = service.sync(stream)

        self.assertTrue(ok)
        self.assertFalse(stream.is_live)
        self.assertEqual(stream.current_game, "Persona 5 Royal")
        self.assertEqual(stream.current_category, "Persona 5 Royal")

    def test_title_parser_ignores_will_we_beat_it_marker(self):
        service = StreamContextSyncService(twitch_api=None)

        markers = service._extract_title_markers("Iifa Tree Nightmare! Will We Beat It?", "FINAL FANTASY IX")

        self.assertIn("Iifa Tree Nightmare", markers)
        self.assertNotIn("Will", markers)
        self.assertFalse(any("Will" in marker for marker in markers))

    def test_memory_extraction_guard_blocks_unconfirmed_game_knowledge_reply(self):
        engine = HebeEngine.__new__(HebeEngine)
        engine._current_input_event = SimpleNamespace(
            normalized_text="que sabes de persona 5 royal",
            stt_metadata={"block_memory_extraction": True, "block_memory_extraction_reason": "model_invented_or_unconfirmed"},
        )

        self.assertFalse(engine._should_extract_memory(source="stt_voice", execution=SimpleNamespace(first_result_of_type=lambda kind: object())))

    def test_user_correction_invalidates_wrong_stored_fact(self):
        session_primer.save_game_session_note(
            "FINAL FANTASY IX",
            end_summary="Will parece ser el objetivo de hoy",
            next_time_plan="buscar a Will",
        )

        changed = session_primer.invalidate_game_session_term("FINAL FANTASY IX", "Will", source="user_correction")
        latest = session_primer.latest_game_session("FINAL FANTASY IX")

        self.assertEqual(changed, 1)
        self.assertNotIn("Will", latest["end_summary"])
        self.assertNotIn("Will", latest["next_time_plan"])

    def test_response_synthesizer_handles_game_knowledge_command_result(self):
        model = EchoModel()
        synth = ResponseSynthesizer(conversation_model=model)
        engine = HebeEngine.__new__(HebeEngine)
        stream = StreamSessionState(enabled=True)
        engine.runtime = SimpleNamespace(state=SimpleNamespace(stream=stream))
        engine.game_profiles = self._persona_store()
        engine.game_research = GameKnowledgeResearchService(store=engine.game_profiles, config=GameKnowledgeResearchConfig(enabled=False))
        engine.game_knowledge = GameKnowledgeResolver(profile_store=engine.game_profiles, research_service=engine.game_research)
        result = engine._handle_game_knowledge_query("Que sabes de Persona 5 Royal?", "que sabes de persona 5 royal", stream)

        reply = synth.synthesize_command_result(result, input_text="Que sabes de Persona 5 Royal?")

        self.assertTrue(model.calls)
        self.assertIn("Persona 5 Royal", reply)


if __name__ == "__main__":
    unittest.main()
