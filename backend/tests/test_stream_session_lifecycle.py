import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.hebe_engine import HebeEngine
from app.core import persistent_logs
from app.integrations.twitch.event_adapter import TwitchEventAdapter
from app.services import db_sqlite
from app.stream import memory as stream_memory
from app.stream.context_sync import StreamContextSyncService
from app.stream.behavior_observability import BehaviorObservability, GLOBAL_BEHAVIOR_OBSERVABILITY
from app.stream.state import StreamSessionState


class StreamSessionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db_path = db_sqlite.DB_PATH
        self.old_session_dir = stream_memory.persistent_logs.SESSION_LOG_DIR
        db_sqlite.DB_PATH = os.path.join(self.tmp.name, "hebe.sqlite3")
        stream_memory.persistent_logs.SESSION_LOG_DIR = Path(self.tmp.name) / "sessions"
        stream_memory.init_stream_memory_schema()

    def tearDown(self):
        db_sqlite.DB_PATH = self.old_db_path
        stream_memory.persistent_logs.SESSION_LOG_DIR = self.old_session_dir
        self.tmp.cleanup()

    def live_stream(self, stream_id="stream-1"):
        return StreamSessionState(
            enabled=True,
            is_live=True,
            live_status_known=True,
            twitch_stream_id=stream_id,
            current_stream_title="QA stream",
            current_game="QA game",
            stream_started_at="2026-08-16T18:00:00Z",
        )

    def row(self, session_id):
        conn = db_sqlite.get_db_connection()
        try:
            return dict(conn.execute("SELECT * FROM stream_sessions WHERE id = ?", (session_id,)).fetchone())
        finally:
            conn.close()

    def engine(self, stream):
        engine = HebeEngine.__new__(HebeEngine)
        engine.runtime = SimpleNamespace(
            state=SimpleNamespace(stream=stream, is_running=True, is_processing=False, mode="active"),
            twitch=SimpleNamespace(),
            twitch_events=SimpleNamespace(stop=Mock()),
            twitch_chat_bot=SimpleNamespace(stop=Mock()),
        )
        engine._stop_event = threading.Event()
        engine._persist_canonical_chatter_summaries = Mock()
        def deliver(*_args, **_kwargs):
            engine.runtime.twitch.last_delivery_outcome = {"success": True}
        engine._deliver_twitch_reply = Mock(side_effect=deliver)
        engine.response_synthesizer = SimpleNamespace(
            _generate_twitch_outgoing_raid=Mock(return_value="Gracias por acompañarme. Nos vamos de raid.")
        )
        engine.stream_spontaneity = SimpleNamespace(config=SimpleNamespace(cooldown_key="stream_idle_prompt_next_ts"))
        engine._get_live_session_brain = Mock(return_value=SimpleNamespace(
            observe_stream_metadata=Mock(), retrieve_context=Mock()
        ))
        return engine

    def test_live_offline_finalizes_exactly_once(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        stream_memory.mark_stream_session_ending(stream, reason="offline", source_signal="eventsub_offline")
        first = stream_memory.finalize_stream_session(stream, reason="offline", source_signal="eventsub_offline")
        second = stream_memory.finalize_stream_session(stream, reason="offline", source_signal="poll_offline")
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(self.row(session_id)["finalize_count"], 1)
        self.assertIsNone(stream_memory.get_active_stream_session())

    def test_raid_then_shutdown_replay_finalizes_once_and_does_not_reuse(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        stream_memory.record_chat_message(username="viewer", display_name="Viewer", message_text="bye")
        engine = self.engine(stream)
        event = SimpleNamespace(
            event_type="twitch_outgoing_raid",
            payload={"target_channel": "friends", "source_signal": "eventsub_outgoing_raid"},
        )
        engine.process_internal_event(event)
        engine.stop()

        row = self.row(session_id)
        self.assertEqual(row["lifecycle_state"], "FINALIZED")
        self.assertEqual(row["closure_reason"], "raid")
        self.assertEqual(row["finalize_count"], 1)
        self.assertEqual(row["farewell_status"], "emitted")
        engine._deliver_twitch_reply.assert_called_once()
        self.assertTrue((Path(self.tmp.name) / "sessions" / f"stream-session-{session_id}.json").exists())
        next_stream = self.live_stream("stream-2")
        next_id = stream_memory.ensure_active_stream_session(next_stream, source="engine")
        self.assertNotEqual(next_id, session_id)

    def test_clean_shutdown_without_raid_finalizes_and_flushes(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        self.engine(stream).stop()
        row = self.row(session_id)
        self.assertEqual(row["closure_reason"], "normal_shutdown")
        self.assertEqual(row["finalize_count"], 1)

    def test_restart_offline_recovers_stale_open_session(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        restarted = StreamSessionState(enabled=True, is_live=False, live_status_known=True)
        result = stream_memory.recover_incomplete_stream_session(
            restarted, live_evidence=False, current_stream_id=None
        )
        self.assertEqual(result["action"], "finalized")
        self.assertEqual(self.row(session_id)["closure_reason"], "recovered_after_restart")

    def test_restart_same_stream_resumes_without_duplicate(self):
        original = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(original, source="engine")
        restarted = self.live_stream()
        restarted.active_stream_session_id = None
        result = stream_memory.recover_incomplete_stream_session(
            restarted, live_evidence=True, current_stream_id="stream-1"
        )
        reused = stream_memory.ensure_active_stream_session(restarted, source="context_sync")
        self.assertEqual(result["action"], "resumed")
        self.assertEqual(reused, session_id)

    def test_restart_different_live_stream_finalizes_old_instead_of_reusing_it(self):
        original = self.live_stream("stream-old")
        session_id = stream_memory.ensure_active_stream_session(original, source="engine")
        restarted = self.live_stream("stream-new")
        restarted.active_stream_session_id = None
        result = stream_memory.recover_incomplete_stream_session(
            restarted, live_evidence=True, current_stream_id="stream-new"
        )
        self.assertEqual(result["action"], "finalized")
        self.assertEqual(self.row(session_id)["closure_reason"], "recovered_after_restart")
        next_id = stream_memory.ensure_active_stream_session(restarted, source="context_sync")
        self.assertNotEqual(next_id, session_id)

    def test_raid_then_offline_does_not_finalize_or_farewell_twice(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        engine = self.engine(stream)
        event = SimpleNamespace(event_type="twitch_outgoing_raid", payload={"source_signal": "eventsub_outgoing_raid"})
        engine.process_internal_event(event)
        engine._handle_stream_lifecycle_event(SimpleNamespace(event_type="stream_offline", payload={}))
        self.assertEqual(self.row(session_id)["finalize_count"], 1)
        engine._deliver_twitch_reply.assert_called_once()

    def test_farewell_failure_never_blocks_technical_finalization(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        engine = self.engine(stream)
        engine.response_synthesizer._generate_twitch_outgoing_raid.side_effect = RuntimeError("model down")
        engine.process_internal_event(SimpleNamespace(
            event_type="twitch_outgoing_raid", payload={"source_signal": "eventsub_outgoing_raid"}
        ))
        row = self.row(session_id)
        self.assertEqual(row["farewell_status"], "skipped")
        self.assertEqual(row["finalize_count"], 1)

    def test_farewell_transport_failure_is_not_reported_as_emitted(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        engine = self.engine(stream)
        engine._deliver_twitch_reply.side_effect = lambda *_args, **_kwargs: setattr(
            engine.runtime.twitch, "last_delivery_outcome", {"success": False, "reason": "chunk_failed"}
        )
        engine.process_internal_event(SimpleNamespace(
            event_type="twitch_outgoing_raid", payload={"source_signal": "eventsub_outgoing_raid"}
        ))
        row = self.row(session_id)
        self.assertEqual(row["farewell_status"], "skipped")
        self.assertEqual(row["farewell_reason"], "chunk_failed")
        self.assertEqual(row["finalize_count"], 1)

    def test_incremental_artifact_exists_and_contains_bounded_references(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        path = Path(self.tmp.name) / "sessions" / f"stream-session-{session_id}.json"
        self.assertTrue(path.exists())
        stream_memory.record_stream_event("session_action", {"id": "action-1"}, stream=stream)
        stream_memory.record_chat_message(username="viewer", display_name="Viewer", message_text="hello")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["counts"]["inputs"], 1)
        self.assertEqual(artifact["counts"]["events"], 1)
        self.assertEqual(artifact["correlation_ranges"]["events"]["first_id"], 1)
        self.assertNotIn("hello", artifact)

    def test_interrupted_atomic_write_preserves_previous_valid_artifact(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        path = Path(self.tmp.name) / "sessions" / f"stream-session-{session_id}.json"
        before = json.loads(path.read_text(encoding="utf-8"))
        with patch.object(stream_memory.os, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                stream_memory.checkpoint_stream_session(session_id)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), before)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_api_failure_preserves_open_live_session_without_offline_evidence(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        stream.stream_context_updated_ts = 0
        service = StreamContextSyncService(
            twitch_api=SimpleNamespace(get_stream=Mock(side_effect=RuntimeError("temporary")))
        )
        self.assertFalse(service.sync(stream))
        result = stream_memory.recover_incomplete_stream_session(
            stream, live_evidence=None, current_stream_id=stream.twitch_stream_id
        )
        self.assertEqual(result["action"], "preserved")
        self.assertEqual(self.row(session_id)["lifecycle_state"], "LIVE")

    def test_eventsub_outgoing_raid_is_distinct_from_incoming(self):
        pushed = []
        adapter = TwitchEventAdapter(
            client_id="client", user_oauth_token="token", broadcaster_user_id="123",
            bot_user_id="456", twitch_service=Mock(), push_event_callback=lambda kind, payload: pushed.append((kind, payload)),
        )
        adapter._handle_event("channel.raid", {
            "from_broadcaster_user_id": "123", "to_broadcaster_user_id": "999",
            "to_broadcaster_user_login": "friends", "viewers": 7,
        })
        self.assertEqual(pushed[0][0], "twitch_outgoing_raid")
        self.assertEqual(pushed[0][1]["source_signal"], "eventsub_outgoing_raid")

    def test_eventsub_subscribes_to_both_raid_directions(self):
        adapter = TwitchEventAdapter(
            client_id="client", user_oauth_token="token", broadcaster_user_id="123",
            bot_user_id="456", twitch_service=Mock(),
        )
        adapter._session_id = "eventsub-session"
        adapter._create_subscription = Mock(return_value=True)
        adapter._subscribe_defaults()
        raid_conditions = [
            call.kwargs["condition"] for call in adapter._create_subscription.call_args_list
            if call.kwargs["sub_type"] == "channel.raid"
        ]
        self.assertIn({"to_broadcaster_user_id": "123"}, raid_conditions)
        self.assertIn({"from_broadcaster_user_id": "123"}, raid_conditions)

    def test_retention_never_prunes_open_artifact_and_is_separate_from_log_rotation(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        path = Path(self.tmp.name) / "sessions" / f"stream-session-{session_id}.json"
        os.utime(path, (0, 0))
        self.assertEqual(
            stream_memory.prune_session_artifacts(retention_days=1, now=datetime.now(timezone.utc)), 0
        )
        stream_memory.finalize_stream_session(stream, reason="offline", source_signal="eventsub_offline")
        os.utime(path, (0, 0))
        self.assertEqual(
            stream_memory.prune_session_artifacts(retention_days=1, now=datetime.now(timezone.utc)), 1
        )
        self.assertFalse(path.exists())

    def test_behavior_historical_retention_is_configurable_independently(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        persistent_logs.log_behavior_session_event(session_id, {
            "event": "candidate_policy", "trace_id": "retention-trace",
            "timestamp": 1.0, "evaluation_count": 1, "evaluation_delta": 1,
        })
        stream_memory.finalize_stream_session(stream, reason="offline", source_signal="eventsub_offline")
        artifact = Path(self.tmp.name) / "sessions" / f"stream-session-{session_id}.json"
        behavior_path, behavior_index = persistent_logs.behavior_session_paths(session_id)
        compressed = behavior_path.with_suffix(behavior_path.suffix + ".gz")
        os.utime(behavior_index, (0, 0))
        os.utime(compressed, (0, 0))
        with patch.dict(os.environ, {
            "HEBE_SESSION_ARTIFACT_RETENTION_DAYS": "365",
            "HEBE_BEHAVIOR_SESSION_RETENTION_DAYS": "1",
        }):
            stream_memory.prune_session_artifacts(now=datetime.now(timezone.utc))
        self.assertTrue(artifact.exists())
        self.assertFalse(behavior_index.exists())
        self.assertFalse(compressed.exists())

    def test_session_artifact_references_complete_coalesced_behavior_telemetry(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        observability = BehaviorObservability(
            log_fn=lambda *_args: None,
            session_log_fn=persistent_logs.log_behavior_session_event,
            coalesce_checkpoint_seconds=15,
            coalesce_checkpoint_evaluations=25,
        )
        for index in range(55):
            observability.record(
                "candidate_policy", trace_id="trace-first", timestamp=10_000 + index * 0.5,
                stream_session_id=str(session_id), candidate_id="intent-repeat",
                normalized_motif_identity="motif-repeat", usage_count=8, fatigue=0.5,
                policy_decision="DOWNRANK", reason_code="motif_repetition_downrank",
            )
        observability.flush_session(session_id)
        stream_memory.checkpoint_stream_session(session_id)

        artifact_path = Path(self.tmp.name) / "sessions" / f"stream-session-{session_id}.json"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        reference = artifact["behavior_telemetry"]
        telemetry_path = Path(reference["telemetry_file"])
        lines = telemetry_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(reference["policy_evaluation_count"], 55)
        self.assertLessEqual(reference["event_count"], 4)
        self.assertEqual(json.loads(lines[0])["trace_id"], "trace-first")

        stream_memory.finalize_stream_session(stream, reason="offline", source_signal="eventsub_offline")
        finalized = json.loads(artifact_path.read_text(encoding="utf-8"))["behavior_telemetry"]
        self.assertTrue(finalized["compressed"])
        self.assertTrue(Path(finalized["telemetry_file"]).exists())
        self.assertFalse(telemetry_path.exists())

    def test_global_rotation_cannot_destroy_active_session_behavior_trace(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        global_dir = Path(self.tmp.name) / "global-logs"
        with patch.object(persistent_logs, "LOG_DIR", global_dir), \
             patch.object(persistent_logs, "MAX_BYTES", 500), \
             patch.object(persistent_logs, "BACKUP_COUNT", 2):
            observability = BehaviorObservability(
                coalesce_checkpoint_seconds=999,
                coalesce_checkpoint_evaluations=999,
            )
            for index in range(60):
                observability.record(
                    "candidate_policy", trace_id="trace-active", timestamp=20_000 + index,
                    stream_session_id=str(session_id), candidate_id="intent-active",
                    normalized_motif_identity="motif-active", usage_count=1,
                    policy_decision="ALLOW", reason_code="allowed",
                )
            observability.flush_session(session_id)

        global_variants = list(global_dir.glob("behavior_calibration.jsonl*"))
        reference = persistent_logs.behavior_session_reference(session_id)
        self.assertLessEqual(len(global_variants), 3)
        self.assertEqual(reference["policy_evaluation_count"], 60)
        self.assertEqual(reference["candidate_trace_count"], 1)
        self.assertTrue(Path(reference["telemetry_file"]).exists())

    def test_session_finalization_flushes_global_behavior_counters_before_artifact(self):
        stream = self.live_stream()
        session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
        with patch.object(GLOBAL_BEHAVIOR_OBSERVABILITY, "_log_fn", lambda *_args: None), \
             patch.object(
                 GLOBAL_BEHAVIOR_OBSERVABILITY,
                 "_session_log_fn",
                 persistent_logs.log_behavior_session_event,
             ):
            for index in range(7):
                GLOBAL_BEHAVIOR_OBSERVABILITY.record(
                    "candidate_policy", trace_id="trace-finalize", timestamp=30_000 + index,
                    stream_session_id=str(session_id), candidate_id="intent-finalize",
                    normalized_motif_identity="motif-finalize", usage_count=1,
                    policy_decision="ALLOW", reason_code="allowed",
                )
            stream_memory.finalize_stream_session(stream, reason="offline", source_signal="eventsub_offline")

        artifact_path = Path(self.tmp.name) / "sessions" / f"stream-session-{session_id}.json"
        reference = json.loads(artifact_path.read_text(encoding="utf-8"))["behavior_telemetry"]
        self.assertEqual(reference["policy_evaluation_count"], 7)
        self.assertTrue(reference["compressed"])


if __name__ == "__main__":
    unittest.main()
