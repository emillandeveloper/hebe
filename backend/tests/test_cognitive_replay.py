from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.replay.assertions import evaluate
from app.replay.clock import ScenarioClock, resolve_event_time
from app.replay.cognitive import CognitiveReplayRunner
from app.replay.migrations import MigrationRunner, replay_foundation_migrations
from app.replay.report import (
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    STATUS_VERIFIED,
    build_report,
    sanitize,
    write_report,
)
from app.replay.scenario import CognitiveReplayScenario, ScenarioAssertion
from app.replay.workspace import ScenarioWorkspace
from app.core.state import HebeState
from app.hebe_engine import HebeEngine


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cognitive_replay"


class ScenarioSchemaTests(unittest.TestCase):
    def test_versioned_scenario_resolves_relative_times_from_start(self):
        scenario = CognitiveReplayScenario.from_value({
            "schema_version": 1,
            "scenario_id": "schema",
            "initial_time": "2026-08-11T18:00:00+02:00",
            "events": [
                {"event_id": "one", "at": "+5s", "type": "owner_stt", "text": "Hebe"},
                {"event_id": "two", "at": "+1m", "type": "advance_time"},
            ],
        })
        self.assertEqual(scenario.events[0].timestamp - scenario.initial_time, 5)
        self.assertEqual(scenario.events[1].timestamp - scenario.initial_time, 60)

    def test_unknown_event_and_future_schema_fail_closed(self):
        with self.assertRaises(ValueError):
            CognitiveReplayScenario.from_value({"schema_version": 2, "scenario_id": "bad", "initial_time": 1})
        with self.assertRaises(ValueError):
            CognitiveReplayScenario.from_value({
                "schema_version": 1, "scenario_id": "bad", "initial_time": 1,
                "events": [{"type": "invented", "at": 1}],
            })

    def test_scenario_clock_advances_days_without_sleep(self):
        clock = ScenarioClock.from_value("2026-08-11T00:00:00+00:00")
        before = clock.now()
        clock.advance(7 * 86400)
        self.assertEqual(clock.now() - before, 7 * 86400)
        self.assertEqual(clock.monotonic(), 7 * 86400)


class MigrationRunnerTests(unittest.TestCase):
    def test_migration_is_restart_safe_and_checksum_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "migration.sqlite3"
            runner = MigrationRunner(lambda: sqlite3.connect(path))
            first = runner.migrate(replay_foundation_migrations())
            second = runner.migrate(replay_foundation_migrations())
            self.assertFalse(first[0]["already_applied"])
            self.assertTrue(second[0]["already_applied"])

    def test_workspace_supports_copied_legacy_database_and_clean_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "legacy.sqlite3"
            conn = sqlite3.connect(fixture)
            try:
                conn.execute("CREATE TABLE legacy_marker(value TEXT NOT NULL)")
                conn.execute("INSERT INTO legacy_marker(value) VALUES ('preserved')")
                conn.commit()
            finally:
                conn.close()
            root = Path(tmp) / "workspace"
            workspace = ScenarioWorkspace("copy", root=root, database_fixture=str(fixture))
            workspace.activate()
            conn = workspace.connection(readonly=True)
            try:
                self.assertEqual(conn.execute("SELECT value FROM legacy_marker").fetchone()[0], "preserved")
            finally:
                conn.close()
            workspace.deactivate()
            conn = sqlite3.connect(workspace.db_path)
            try:
                conn.execute("INSERT INTO legacy_marker(value) VALUES ('stale-run')")
                conn.commit()
            finally:
                conn.close()
            workspace.activate()
            conn = workspace.connection(readonly=True)
            try:
                values = [row[0] for row in conn.execute("SELECT value FROM legacy_marker")]
            finally:
                conn.close()
            workspace.deactivate()
            self.assertEqual(values, ["preserved"])


class AssertionTests(unittest.TestCase):
    def test_collection_and_exactly_once_assertions(self):
        state = {"actions": [{"operation": "twitch.shoutout", "ok": True}]}
        result = evaluate(ScenarioAssertion("exactly_once", "actions", matching={"operation": "twitch.shoutout"}), state)
        self.assertTrue(result.passed)

    def test_future_assertion_is_explicitly_skipped(self):
        result = evaluate(ScenarioAssertion("exists", "beliefs.x", future_phase="2"), {})
        self.assertTrue(result.skipped)
        self.assertIn("pending_future_phase", result.reason)


class ReportGeneratorTests(unittest.TestCase):
    def _scenario_result(self, status):
        return {
            "scenario_id": "known",
            "status": status,
            "scenario_schema_version": 1,
            "seed": 1,
            "feature_flags": {"cognitive_replay_enabled": True},
            "events_processed": 1,
            "restart_count": 0,
            "duration_seconds": 0.01,
            "assertion_summary": {"passed": 1, "failed": 0, "skipped": 0},
            "failures": [],
            "database": {"path": "synthetic.sqlite3", "schema_migrations": []},
        }

    def test_pass_fail_incomplete_statuses(self):
        self.assertEqual(build_report(scenario_results=[self._scenario_result(STATUS_VERIFIED)]).overall_status, STATUS_VERIFIED)
        self.assertEqual(build_report(scenario_results=[self._scenario_result(STATUS_FAILED)]).overall_status, STATUS_FAILED)
        self.assertEqual(
            build_report(scenario_results=[self._scenario_result(STATUS_VERIFIED)], tests={"required_layer_missing": True}).overall_status,
            STATUS_INCOMPLETE,
        )

    def test_inherited_baseline_failures_do_not_become_phase_regressions(self):
        report = build_report(
            scenario_results=[self._scenario_result(STATUS_VERIFIED)],
            tests={"failed": 13, "required_layer_missing": False},
            baseline_differential={"pre_existing_failures": 13, "new_regressions": 0},
        )
        self.assertEqual(report.overall_status, STATUS_VERIFIED)

    def test_new_baseline_regression_still_fails_verification(self):
        report = build_report(
            scenario_results=[self._scenario_result(STATUS_VERIFIED)],
            tests={"failed": 1, "required_layer_missing": False},
            baseline_differential={"pre_existing_failures": 0, "new_regressions": 1},
        )
        self.assertEqual(report.overall_status, STATUS_FAILED)

    def test_expected_future_scenario_gap_does_not_block_foundation_status(self):
        future = self._scenario_result(STATUS_INCOMPLETE)
        future["expected_future_gap"] = True
        self.assertEqual(build_report(scenario_results=[future]).overall_status, STATUS_VERIFIED)

    def test_markdown_and_json_agree_and_sensitive_values_are_redacted(self):
        report = build_report(scenario_results=[self._scenario_result(STATUS_VERIFIED)])
        report.scenarios[0]["oauth_token"] = "secret-value"
        report.scenarios[0]["raw_transcript"] = "unrestricted words"
        with tempfile.TemporaryDirectory() as tmp:
            json_path, markdown_path = write_report(report, tmp)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertEqual(data["overall_status"], STATUS_VERIFIED)
            self.assertIn("**VERIFIED**", markdown)
            self.assertNotIn("secret-value", json_path.read_text(encoding="utf-8"))
            self.assertNotIn("unrestricted words", json_path.read_text(encoding="utf-8"))


class CanonicalIngressParityTests(unittest.TestCase):
    def test_owner_and_ambient_wrappers_share_normalized_stt_seam(self):
        engine = HebeEngine.__new__(HebeEngine)
        engine.ingest_normalized_stt = Mock(return_value="continue")
        engine._last_policy_trace = {"intent": "ambient_stt"}
        engine._last_input_firewall = {"source": "ambient_stt", "authority": "ambient"}

        engine.ingest_owner_stt("Hebe, prueba")
        engine.ingest_ambient_stt("ruido de fondo")

        self.assertEqual(engine.ingest_normalized_stt.call_count, 2)
        self.assertEqual(engine.ingest_normalized_stt.call_args_list[0].args, ("Hebe, prueba",))
        self.assertEqual(engine.ingest_normalized_stt.call_args_list[1].args, ("ruido de fondo",))
        self.assertTrue(engine.ingest_normalized_stt.call_args_list[1].kwargs["force_ambient"])

    def test_live_chat_adapter_calls_the_same_normalized_ingress(self):
        chat_bot = SimpleNamespace(
            enabled=False,
            bot_username="hebe",
            ambient_message_callback=None,
            message_callback=None,
            social_event_callback=None,
        )
        runtime = SimpleNamespace(
            state=HebeState(),
            llm=None,
            intent_llm=None,
            twitch=None,
            twitch_chat_bot=chat_bot,
            twitch_events=None,
            speak=Mock(),
        )
        with patch.object(HebeEngine, "ingest_normalized_twitch_chat") as ingress:
            HebeEngine(runtime=runtime, use_wakeword=True)
            chat_bot.message_callback("alice", "Alice", "hola", "#test", {"id": "m1"})
        ingress.assert_called_once_with(
            username="alice", display_name="Alice", text="hola", channel="#test", irc_tags={"id": "m1"}
        )

    def test_replay_owner_stt_dispatch_calls_shared_engine_ingress(self):
        scenario = CognitiveReplayScenario.from_value({
            "schema_version": 1, "scenario_id": "owner-parity", "initial_time": 1,
            "events": [{"event_id": "owner", "at": 1, "type": "owner_stt", "text": "Hebe"}],
        })
        runner = CognitiveReplayRunner()
        runner.clock = ScenarioClock(1)
        runner.engine = SimpleNamespace(ingest_owner_stt=Mock())
        runner.twitch = SimpleNamespace()
        runner.outcomes = SimpleNamespace()
        runner._dispatch(scenario.events[0])
        runner.engine.ingest_owner_stt.assert_called_once()

    def test_replay_twitch_dispatch_calls_shared_normalized_ingress(self):
        scenario = CognitiveReplayScenario.from_value({
            "schema_version": 1, "scenario_id": "chat-parity", "initial_time": 1,
            "events": [{"event_id": "chat", "at": 1, "type": "twitch_chat", "user_id": "1", "login": "alice", "text": "hola"}],
        })
        engine = SimpleNamespace(ingest_normalized_twitch_chat=Mock())
        twitch = SimpleNamespace(channel_name="test", remember_identity=Mock())
        runner = CognitiveReplayRunner()
        runner.clock = ScenarioClock(1)
        runner.engine = engine
        runner.twitch = twitch
        runner.outcomes = SimpleNamespace()
        runner._dispatch(scenario.events[0])
        engine.ingest_normalized_twitch_chat.assert_called_once()

    def test_replay_lifecycle_dispatch_calls_shared_engine_ingress(self):
        scenario = CognitiveReplayScenario.from_value({
            "schema_version": 1, "scenario_id": "lifecycle-parity", "initial_time": 1,
            "events": [{"event_id": "start", "at": 1, "type": "stream_started", "session_id": "s1"}],
        })
        engine = SimpleNamespace(ingest_stream_lifecycle=Mock())
        twitch = SimpleNamespace(configure_stream_metadata=Mock())
        runner = CognitiveReplayRunner()
        runner.clock = ScenarioClock(1)
        runner.engine = engine
        runner.twitch = twitch
        runner.outcomes = SimpleNamespace()
        runner._dispatch(scenario.events[0])
        engine.ingest_stream_lifecycle.assert_called_once()

    def test_replay_metadata_dispatch_calls_shared_engine_ingress(self):
        scenario = CognitiveReplayScenario.from_value({
            "schema_version": 1, "scenario_id": "metadata-parity", "initial_time": 1,
            "events": [{"event_id": "meta", "at": 1, "type": "stream_metadata_changed", "title": "Synthetic", "game": "FFV"}],
        })
        engine = SimpleNamespace(ingest_stream_metadata=Mock())
        runner = CognitiveReplayRunner()
        runner.clock = ScenarioClock(1)
        runner.engine = engine
        runner.twitch = SimpleNamespace()
        runner.outcomes = SimpleNamespace()
        runner._dispatch(scenario.events[0])
        engine.ingest_stream_metadata.assert_called_once_with({"title": "Synthetic", "game": "FFV"})


class CognitiveReplayIntegrationTests(unittest.TestCase):
    def test_ambient_scenario_uses_firewall_and_does_not_emit(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = CognitiveReplayRunner(workspace_root=tmp, retain_workspace=True).run(
                FIXTURES / "ambient_false_positive_foundation.json"
            )
        self.assertEqual(result.status, STATUS_VERIFIED, result.failures)
        before = result.checkpoint_states["owner"]["final_emission_results"]
        after = result.checkpoint_states["ambient"]["final_emission_results"]
        self.assertEqual(len(after), len(before))
        self.assertEqual(result.checkpoint_states["ambient"]["runtime"]["last_firewall"]["source"], "ambient_stt")
        self.assertEqual(result.checkpoint_states["ambient"]["runtime"]["last_firewall"]["authority"], "ambient")

    def test_ivanxi_scenario_uses_receipt_and_survives_real_engine_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = CognitiveReplayRunner(workspace_root=tmp, retain_workspace=True)
            result = runner.run(FIXTURES / "ivanxi_resub_promo_restart.json")
        self.assertEqual(result.status, STATUS_VERIFIED, result.failures)
        self.assertEqual(result.restart_count, 1)
        self.assertTrue(result.restart_evidence[0]["old_engine_collected"])
        self.assertTrue(result.restart_evidence[0]["volatile_state_recreated"])
        receipts = result.final_state["receipts"]
        self.assertTrue(any(item["execution_status"] == "sent" for item in receipts))
        profiles = result.final_state["promotion_profiles"]
        self.assertTrue(any(item["current_login"] == "ivanxi_kun" for item in profiles))
        attempts = result.final_state["actions"]["attempts"]
        self.assertEqual(sum(1 for item in attempts if item["operation"] == "twitch.shoutout"), 1)
        self.assertTrue(result.final_state["final_emission_results"])


if __name__ == "__main__":
    unittest.main()
