import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from app.cognitive.scheduler import InternalEvent
from app.hebe_engine import HebeEngine
from app.services import db_sqlite
from app.stream import session_primer
from app.stream.ambient_context import AmbientContextExtractor
from app.stream.game_intelligence import GameIntelligenceStore, GameResearchService
from app.stream.spontaneity import StreamSpontaneityService
from app.stream.state import StreamSessionState
from backend.tests.test_stream_presence import make_engine
from tests.test_voice_command_pipeline import install_test_continuity, open_test_conversation


SAFE_ROWS = [{
    "claim": "Combat uses guard to reduce incoming damage.",
    "source_title": "Official manual",
    "url": "https://example.invalid/manual",
    "excerpt": "Guard reduces incoming damage.",
    "confidence": 0.94,
    "general_mechanic": True,
}]


def collect(service, job):
    try:
        service._jobs[job.job_id][1].result(timeout=2)
    except Exception:
        pass
    return service.collect_job(job.job_id)[0]


class TemporalScheduleRegressionTests(unittest.TestCase):
    def setUp(self):
        self.old_path = db_sqlite.DB_PATH
        tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        tmp.close()
        self.path = tmp.name
        db_sqlite.DB_PATH = self.path
        db_sqlite.init_db()
        session_primer.init_session_primer_schema()

    def tearDown(self):
        db_sqlite.DB_PATH = self.old_path
        os.unlink(self.path)

    def test_repeated_real_sundays_supersede_old_retro_slot(self):
        tz = ZoneInfo("Europe/Madrid")
        dates = (datetime(2026, 8, 2, 19, tzinfo=tz), datetime(2026, 8, 9, 19, tzinfo=tz))
        first = session_primer.record_schedule_observation(
            stream_session_id="sunday-1", canonical_content="Super Robot Taisen OG Saga: Endless Frontier", dt=dates[0],
        )
        self.assertTrue(first["recorded"])
        weakening = session_primer.get_schedule_for_date(dates[0])
        self.assertTrue(weakening["schedule_uncertain"])

        second = session_primer.record_schedule_observation(
            stream_session_id="sunday-2", canonical_content="Super Robot Taisen OG Saga: Endless Frontier", dt=dates[1],
        )
        self.assertEqual(second["hypothesis"]["status"], "tentative")
        current = session_primer.get_schedule_for_date(dates[1])
        self.assertEqual(current["game"], "Super Robot Taisen OG Saga: Endless Frontier")
        self.assertEqual(current["hypothesis_status"], "tentative")
        conn = db_sqlite.get_db_connection()
        old = conn.execute(
            "SELECT status FROM schedule_hypotheses WHERE weekday='sunday' AND source='owner_declared'"
        ).fetchone()
        conn.close()
        self.assertEqual(old["status"], "superseded")


class PartialResearchRegressionTests(unittest.TestCase):
    def test_successful_units_survive_one_subquery_timeout_and_only_it_retries(self):
        clock = [100.0]
        calls = []
        combat_attempts = [0]

        class Provider:
            available = True

            def search(self, query, constraints=None, **kwargs):
                calls.append(query)
                if "core combat mechanics" in query:
                    combat_attempts[0] += 1
                    if combat_attempts[0] == 1:
                        raise TimeoutError("combat timed out")
                return SAFE_ROWS

        connection = sqlite3.connect(":memory:", check_same_thread=False)
        service = GameResearchService(
            store=GameIntelligenceStore(connection=connection), provider=Provider(),
            now_fn=lambda: clock[0], retry_base_seconds=1,
        )
        try:
            identity = service.prepare_game_async(game_title="Test Game", session_id="53")
            self.assertEqual(identity.timeout_seconds, 40.0)
            self.assertEqual(collect(service, identity).status, "completed")
            systems = service.prepare_game_async(game_title="Test Game", session_id="53")
            self.assertEqual(collect(service, systems).status, "completed")
            combat = service.prepare_game_async(game_title="Test Game", session_id="53")
            failed = collect(service, combat)
            self.assertEqual(failed.status, "failed")
            dossier = service.store.get_dossier("test_game")
            self.assertEqual(dossier.status, "partial")
            self.assertEqual(dossier.research_sections["identity_premise"], "completed")
            self.assertEqual(dossier.research_sections["core_systems"], "completed")
            self.assertNotIn("combat_mechanics", dossier.research_sections)
            clock[0] = failed.next_retry_at
            retry = service.retry_due_jobs()[0]
            self.assertEqual(retry.metadata["research_unit"], "combat_mechanics")
            self.assertEqual(collect(service, retry).status, "completed")
            ready = service.store.get_dossier("test_game")
            self.assertEqual(ready.status, "ready")
            self.assertEqual(len(calls), 4)
        finally:
            service._executor.shutdown(wait=True)
            connection.close()


class EvidenceAndOpportunityRegressionTests(unittest.TestCase):
    def test_ground_counter_fact_is_literal_and_scene_bound(self):
        result = AmbientContextExtractor().extract(
            "si toca el suelo luego te puede hacer contraataque", now=10, scene_id="scene-9",
        )
        facts = [fact for fact in result.facts if fact["category"] == "enemy_mechanic"]
        self.assertEqual(len(facts), 1)
        fact = facts[0]
        self.assertEqual(fact["scene_id"], "scene-9")
        self.assertEqual(fact["inferred_claims"], [])
        semantic = " ".join([fact["summary"], *fact["directly_supported_claims"]]).lower()
        self.assertIn("ground", semantic)
        self.assertIn("counterattack", semantic)
        for unsupported in ("low hp", "survival", "healing", "auto-healing"):
            self.assertNotIn(unsupported, semantic)

    def test_consumed_opportunity_rate_limit_is_scoped_to_same_scene(self):
        service = StreamSpontaneityService(now_fn=lambda: 100.0)
        stream = StreamSessionState()
        stream.current_scene_timeline = {"scene_id": "scene-a", "topic_id": "boss"}
        payload = {
            "idle_topic": "challenge_comment",
            "opportunity_rate_limit_key": service.opportunity_rate_limit_key(
                stream, topic="challenge_comment", fact={"category": "combat_risk"},
            ),
        }
        consumed = service.consume_opportunity(stream, payload, reason="generated_output_suppressed")
        self.assertGreater(stream.cooldowns[consumed], 100.0)
        stream.current_scene_timeline = {"scene_id": "scene-b", "topic_id": "boss"}
        reopened = service.opportunity_rate_limit_key(
            stream, topic="challenge_comment", fact={"category": "combat_risk"},
        )
        self.assertNotEqual(consumed, reopened)
        self.assertNotIn(reopened, stream.cooldowns)


class RoutingRegressionTests(unittest.TestCase):
    def test_duplicate_raid_is_dropped_before_second_coordinator_submit(self):
        engine = HebeEngine.__new__(HebeEngine)
        stream = StreamSessionState(enabled=True, is_live=True)
        engine._get_stream_state = lambda: stream

        class Coordinator:
            def __init__(self):
                self.events = []

            def submit(self, event, processor):
                self.events.append(event)

        coordinator = Coordinator()
        engine._get_twitch_interaction_coordinator = lambda: coordinator
        first = InternalEvent("twitch_raid", {"source": "irc_usernotice", "user_login": "raider", "viewer_count": 7, "event_id": "irc-1"}, "2026-08-09T19:00:00Z")
        second = InternalEvent("twitch_raid", {"source": "eventsub", "user_login": "raider", "viewer_count": 7, "event_id": "es-1"}, "2026-08-09T19:00:01Z")
        engine.process_internal_event(first)
        engine.process_internal_event(second)
        self.assertEqual(len(coordinator.events), 1)

    def test_fresh_complete_promo_executes_after_expired_clarification(self):
        stream = StreamSessionState(enabled=True, is_live=True)
        engine = make_engine(stream)
        install_test_continuity(engine)
        open_test_conversation(
            engine, kind="promotion_target_clarification",
            expected_reply_type="twitch_username_or_viewer_alias", ttl_seconds=-1,
        )
        result = engine._handle_stream_manual_command("Ebe, hazle una promo a Charlie")
        self.assertTrue(result.success)
        self.assertEqual(engine.runtime.twitch.sent, ["!so Charlie"])
        self.assertIsNone(engine._active_current_conversation())


if __name__ == "__main__":
    unittest.main()
