from __future__ import annotations

import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from app.cognitive.input_interpretation import InputInterpreter
from app.replay.behavior_policy import BehaviorTraceReplayCurator
from app.replay.migrations import MigrationRunner
from app.stream.behavior_adaptation import (
    BehaviorAdaptationService,
    semantic_similarity,
    semantic_similarity_evidence,
)
from app.stream.behavior_constraint_store import (
    BehaviorConstraintRepository,
    behavior_constraint_migrations,
)
from app.stream.behavior_constraints import BehaviorConstraint
from app.stream.behavior_observability import BehaviorObservability
from app.stream.live_runtime import LiveSessionStateManager
from app.stream.spontaneity import StreamSpontaneityService


NOW = 50_000.0


def make_stream() -> SimpleNamespace:
    return SimpleNamespace(
        active_stream_session_id="session-shadow",
        recent_idle_messages=[],
        active_behavior_blocks=[],
        behavior_adaptation_state={"entries": []},
        last_behavior_adaptation_decision=None,
        last_feedback_application=None,
        current_discourse_topic="",
        idle_prompts_sent_stream=0,
    )


class BehaviorObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.logged = []
        self.observability = BehaviorObservability(
            max_recent=64,
            max_labels=16,
            clock=lambda: NOW,
            log_fn=lambda kind, payload: self.logged.append((kind, deepcopy(payload))),
        )
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "behavior.sqlite3"
        self.connection_factory = lambda: sqlite3.connect(self.db_path)
        MigrationRunner(self.connection_factory).migrate(behavior_constraint_migrations())
        self.repository = BehaviorConstraintRepository(self.connection_factory)
        self.service = BehaviorAdaptationService(
            repository=self.repository,
            observability=self.observability,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def owner_feedback(self, text: str, *, recent: str = ""):
        return InputInterpreter().interpret(
            raw_text=text,
            source="stt_voice",
            authority="owner",
            addressed_to_hebe=True,
            recent_hebe_utterance=recent,
        )

    def test_candidate_trace_contains_correlation_and_similarity_evidence_without_raw_text(self):
        stream = make_stream()
        stream.recent_idle_messages = [{
            "text": "El jefe se esconde tras esa puerta.",
            "topic": "door_joke",
            "timestamp": NOW - 30,
        }]

        decision = self.service.evaluate_candidate(
            stream,
            "La puerta vuelve a cobrar entrada.",
            topic="door_joke",
            now=NOW,
            observation={
                "trace_id": "trace-door",
                "candidate_id": "intent-door",
                "speech_intent_id": "intent-door",
                "speech_intent": "BANTER",
            },
        )

        event = self.observability.snapshot()["recent_events"][-1]
        self.assertEqual(decision.trace_id, "trace-door")
        self.assertEqual(event["trace_id"], "trace-door")
        self.assertEqual(event["candidate_id"], "intent-door")
        self.assertEqual(event["speech_intent"], "BANTER")
        self.assertTrue(event["semantic_terms"])
        comparison = event["recent_comparable_motifs"][0]
        self.assertIn("shared_terms", comparison)
        self.assertIn("containment", comparison)
        self.assertIn("jaccard", comparison)
        self.assertNotIn("candidate_text", event)
        self.assertNotIn("raw_text", event)

    def test_similarity_explanation_is_observational_and_matches_existing_algorithm(self):
        false_positive = semantic_similarity_evidence(
            "jefe detras de la puerta",
            "configura la puerta de enlace de red",
        )
        false_negative = semantic_similarity_evidence("culpa al rng", "ha sido cosa del azar")

        self.assertEqual(
            false_positive["similarity"],
            round(semantic_similarity("jefe detras de la puerta", "configura la puerta de enlace de red"), 6),
        )
        self.assertGreaterEqual(false_positive["similarity"], 0.25)
        self.assertEqual(false_negative["similarity"], 0.0)
        self.assertEqual(false_negative["shared_terms"], [])

    def test_ranking_post_generation_and_emission_share_trace_id(self):
        stream = make_stream()
        decision = self.service.evaluate_candidate(
            stream,
            "La musica acompana bien esta zona.",
            now=NOW,
            observation={"trace_id": "trace-chain", "candidate_id": "intent-1"},
        )
        self.service.record_ranking(
            stream,
            [{
                "intent_id": "intent-1",
                "topic": "music",
                "base_score": 0.8,
                "adjusted_score": 0.8,
                "policy": decision.to_dict(),
            }],
            selected_intent_id="intent-1",
            generation_attempted=True,
            timestamp=NOW,
        )
        post = self.service.validate_generated_output(
            stream,
            "La musica acompana bien esta zona.",
            now=NOW,
            observation={"trace_id": "trace-chain", "candidate_id": "intent-1"},
        )
        self.service.record_emission(
            trace_id=post.trace_id,
            stream=stream,
            event_id="intent-1",
            emitted=True,
            timestamp=NOW,
        )

        events = [item for item in self.observability.snapshot()["recent_events"] if item["trace_id"] == "trace-chain"]
        self.assertEqual(
            [item["event"] for item in events],
            ["candidate_policy", "candidate_ranking", "post_generation", "emission"],
        )
        self.assertTrue(events[1]["candidate_selected"])
        self.assertTrue(events[1]["generation_attempted"])
        self.assertEqual(events[0]["candidate_semantic_material"], ["musica", "acompana", "bien", "zona"])
        self.assertEqual(events[0]["pre_generation_motif"], decision.motif_id)
        self.assertEqual(events[2]["generated_semantic_material"], ["musica", "acompana", "bien", "zona"])
        self.assertEqual(events[2]["post_generation_motif"], post.motif_id)
        self.assertEqual(events[2]["comparison_decision"], "ALLOW")
        self.assertEqual(events[2]["comparison_reason"], post.reason)
        self.assertTrue(events[-1]["emitted"])

    def test_structural_topic_is_observable_but_not_semantic_material(self):
        stream = make_stream()

        self.service.evaluate_candidate(
            stream,
            "tensión en el combate",
            topic="game,title,playthrough_type,recent_voice_event",
            now=NOW,
            observation={"trace_id": "clean-semantic-material"},
        )

        event = next(
            item for item in self.observability.snapshot()["recent_events"]
            if item["trace_id"] == "clean-semantic-material"
        )
        self.assertEqual(event["topic"], "game,title,playthrough_type,recent_voice_event")
        self.assertEqual(event["candidate_semantic_material"], ["tension", "combate"])
        self.assertTrue({"game", "title", "playthrough", "type", "recent", "voice", "event"}.isdisjoint(event["semantic_terms"]))

    def test_feedback_trace_is_minimal_and_metrics_are_observational(self):
        stream = make_stream()
        utterance = "Otra broma sobre el RNG."
        stream.recent_idle_messages = [{"text": utterance, "topic": "rng", "timestamp": NOW - 10}]
        result = self.service.apply_feedback(
            stream,
            self.owner_feedback("Otra vez con lo del RNG.", recent=utterance),
            recent_hebe_utterance=utterance,
            source_event_id="feedback-1",
            now=NOW,
        )

        feedback = next(item for item in self.observability.snapshot()["recent_events"] if item["event"] == "feedback")
        self.assertTrue(result.applied)
        self.assertEqual(feedback["trace_id"], "feedback-1")
        self.assertTrue(feedback["referent_resolved"])
        self.assertEqual(feedback["polarity"], "negative")
        self.assertNotIn("feedback_text", feedback)
        self.assertNotIn("referent_text", feedback)

    def test_read_only_inspector_separates_constraint_lifecycle_and_fatigue(self):
        stream = make_stream()
        current = BehaviorConstraint(
            id="current-1", actor="Hebe", behavior_family="semantic_motif",
            behavior_variants=["motif:current", "puerta"], recipient_scope="everyone",
            scope="current_stream", source_event_id="event-current", created_at=NOW,
        )
        durable = BehaviorConstraint(
            id="durable-1", actor="Hebe", behavior_family="semantic_motif",
            behavior_variants=["motif:durable", "rng"], recipient_scope="everyone",
            scope="durable", source_event_id="event-durable", created_at=NOW,
        )
        retired = BehaviorConstraint(
            id="durable-retired", actor="Hebe", behavior_family="semantic_motif",
            behavior_variants=["motif:retired", "canto"], recipient_scope="everyone",
            scope="durable", source_event_id="event-retired", created_at=NOW - 100,
        )
        self.service.register_explicit_constraint(stream, current)
        self.service.register_explicit_constraint(stream, durable)
        self.repository.save_durable(retired)
        self.repository.retire("durable-retired", reason="manual_reversal", now=NOW)
        stream.behavior_adaptation_state["entries"] = [{
            "motif_id": "fatigue-1", "motif_terms": ["puerta"],
            "negative_weight": 0.5, "positive_weight": 0.1,
            "negative_applications": 1, "positive_applications": 0,
            "created_at": NOW - 60, "updated_at": NOW - 60,
            "suppress_until": NOW + 60, "provenance": "recent_hebe_utterance",
            "last_kind": "episodic_negative",
        }]
        before = deepcopy(stream.behavior_adaptation_state)

        snapshot = self.service.inspection_snapshot(stream, now=NOW)

        self.assertEqual([item["id"] for item in snapshot["active_current_stream"]], ["current-1"])
        self.assertEqual([item["id"] for item in snapshot["active_durable"]], ["durable-1"])
        self.assertEqual([item["id"] for item in snapshot["retired_durable_recent"]], ["durable-retired"])
        self.assertEqual(snapshot["episodic_fatigue"][0]["motif_id"], "fatigue-1")
        self.assertEqual(stream.behavior_adaptation_state, before)
        self.assertNotIn("source_text", snapshot["active_durable"][0])

    def test_manual_label_is_bounded_and_does_not_change_runtime(self):
        stream = make_stream()
        before = deepcopy(stream.__dict__)
        label = self.observability.label("trace-review", "FALSE_POSITIVE")

        self.assertEqual(label["label"], "FALSE_POSITIVE")
        self.assertEqual(stream.__dict__, before)
        with self.assertRaises(ValueError):
            self.observability.label("trace-review", "TRAIN_AUTOMATICALLY")

    def test_calibration_metrics_cover_decisions_feedback_blocks_and_lifecycle(self):
        for action in ("ALLOW", "DOWNRANK", "COOLDOWN", "SUPPRESS"):
            self.observability.record(
                "candidate_policy",
                trace_id=f"metric-{action}",
                policy_decision=action,
                reason_code=(
                    "negative_feedback_and_recent_repetition"
                    if action == "SUPPRESS"
                    else "observed"
                ),
            )
        self.observability.record(
            "feedback", trace_id="metric-feedback", referent_resolved=False,
        )
        self.observability.record(
            "post_generation", trace_id="metric-post",
            post_generation_decision="SUPPRESS",
            reason_code="generated_output_matches_constraint",
        )
        self.observability.record(
            "constraint_created", trace_id="metric-create", scope="durable",
        )
        self.observability.record(
            "constraint_reverted", trace_id="metric-revert", scope="durable",
        )

        metrics = self.observability.snapshot()["metrics"]
        self.assertEqual(metrics["candidates_evaluated"], 4)
        self.assertEqual(metrics["ALLOW"], 1)
        self.assertEqual(metrics["DOWNRANK"], 1)
        self.assertEqual(metrics["COOLDOWN"], 1)
        self.assertEqual(metrics["SUPPRESS"], 1)
        self.assertEqual(metrics["unresolved_feedback_referents"], 1)
        self.assertEqual(metrics["post_generation_blocks"], 1)
        self.assertEqual(metrics["suppressions_owner_feedback"], 1)
        self.assertEqual(metrics["suppressions_repetition_fatigue"], 1)
        self.assertEqual(metrics["suppressions_explicit_constraint"], 1)
        self.assertEqual(metrics["durable_constraints_created"], 1)
        self.assertEqual(metrics["durable_constraints_reverted"], 1)

    def test_identical_half_second_evaluations_are_coalesced_with_final_count(self):
        written = []
        observability = BehaviorObservability(
            clock=lambda: NOW,
            log_fn=lambda kind, payload: written.append(deepcopy(payload)),
            coalesce_checkpoint_seconds=15,
            coalesce_checkpoint_evaluations=25,
        )
        for index in range(55):
            observability.record(
                "candidate_policy", trace_id="trace-repeat", timestamp=NOW + index * 0.5,
                stream_session_id="session-soak", candidate_id="intent-repeat",
                normalized_motif_identity="motif-repeat", usage_count=11,
                fatigue=0.598 - index * 0.0001, active_constraint="",
                policy_decision="DOWNRANK", reason_code="motif_repetition_downrank",
                similarity_score=0.89,
            )
        observability.flush_session("session-soak")

        self.assertEqual(observability.snapshot()["metrics"]["candidates_evaluated"], 55)
        self.assertLessEqual(len(written), 4)
        self.assertEqual(written[-1]["evaluation_count"], 55)
        self.assertEqual(written[-1]["first_seen"], NOW)
        self.assertEqual(written[-1]["last_seen"], NOW + 27)

    def test_policy_decision_changes_are_never_coalesced(self):
        written = []
        observability = BehaviorObservability(
            log_fn=lambda kind, payload: written.append(deepcopy(payload)),
            coalesce_checkpoint_seconds=999,
            coalesce_checkpoint_evaluations=999,
        )
        for index, decision in enumerate(("ALLOW", "ALLOW", "DOWNRANK", "DOWNRANK", "SUPPRESS")):
            observability.record(
                "candidate_policy", trace_id="trace-transition", timestamp=NOW + index,
                stream_session_id="session-transition", candidate_id="intent-transition",
                normalized_motif_identity="motif-transition", usage_count=1,
                policy_decision=decision, reason_code=decision.casefold(), similarity_score=0.5,
            )
        observability.flush_session("session-transition")

        self.assertEqual([item["policy_decision"] for item in written], ["ALLOW", "DOWNRANK", "SUPPRESS"])
        self.assertEqual([item["evaluation_count"] for item in written], [1, 3, 5])

    def test_coalescing_never_skips_policy_execution(self):
        stream = make_stream()
        decisions = []
        for index in range(20):
            decisions.append(self.service.evaluate_candidate(
                stream,
                "La misma observacion sobre la puerta.",
                topic="door",
                now=NOW + index * 0.5,
                observation={
                    "trace_id": "trace-policy-rerun",
                    "candidate_id": "intent-policy-rerun",
                    "speech_intent_id": "intent-policy-rerun",
                },
            ))
        policy_writes = [item for _, item in self.logged if item["event"] == "candidate_policy"]
        self.assertEqual(len(decisions), 20)
        self.assertEqual(self.observability.snapshot()["metrics"]["candidates_evaluated"], 20)
        self.assertLess(len(policy_writes), len(decisions))

    def _run_multihour_observability_soak(self, hours: int) -> tuple[int, int, dict]:
        written = []
        observability = BehaviorObservability(
            max_recent=1000,
            log_fn=lambda kind, payload: written.append(payload),
            coalesce_checkpoint_seconds=999,
            coalesce_checkpoint_evaluations=999,
            max_coalesce_keys=20_000,
        )
        candidate_count = hours * 60 * 20  # at least the real ~20 candidates/minute
        raw_evaluations = 0
        for candidate_index in range(candidate_count):
            candidate_id = f"soak-{hours}h-{candidate_index}"
            first_seen = NOW + candidate_index * 3
            for evaluation_index in range(3):
                raw_evaluations += 1
                observability.record(
                    "candidate_policy", trace_id=f"trace-{candidate_id}",
                    timestamp=first_seen + evaluation_index * 0.5,
                    stream_session_id=f"soak-{hours}h", candidate_id=candidate_id,
                    normalized_motif_identity=f"motif-{candidate_index % 31}",
                    usage_count=candidate_index % 12, fatigue=0.4,
                    policy_decision="ALLOW", reason_code="allowed", similarity_score=0.2,
                )
        observability.flush_session(f"soak-{hours}h")
        return raw_evaluations, len(written), observability.snapshot()

    def test_four_hour_soak_keeps_first_trace_and_bounds_memory(self):
        raw, written, snapshot = self._run_multihour_observability_soak(4)
        self.assertEqual(raw, 14_400)
        self.assertEqual(written, 9_600)
        self.assertEqual(snapshot["retention"]["coalesce_keys"], 0)
        self.assertLessEqual(snapshot["retention"]["recent_events"], 1000)

    def test_six_hour_soak_keeps_first_trace_and_bounds_memory(self):
        raw, written, snapshot = self._run_multihour_observability_soak(6)
        self.assertEqual(raw, 21_600)
        self.assertEqual(written, 14_400)
        self.assertEqual(snapshot["retention"]["coalesce_keys"], 0)
        self.assertLessEqual(snapshot["retention"]["recent_events"], 1000)

    def test_trace_to_replay_requires_explicit_human_curation(self):
        trace = {"trace_id": "trace-curated", "topic": "rng", "reason_code": "motif_repetition_downrank"}
        case = BehaviorTraceReplayCurator.curate(
            trace,
            name="rng wording",
            candidate="Otra vez ha decidido el azar.",
            expected_decision="downrank",
            calibration_label="FALSE_NEGATIVE",
        )

        self.assertEqual(case.source_trace_id, "trace-curated")
        self.assertEqual(case.to_fixture_row()["candidate"], "Otra vez ha decidido el azar.")
        with self.assertRaises(ValueError):
            BehaviorTraceReplayCurator.curate(
                trace,
                name="invalid",
                candidate="",
                expected_decision="allow",
                calibration_label="UNCERTAIN",
            )

    def test_bounded_soak_covers_candidates_feedback_session_and_durable_reload(self):
        stream = make_stream()
        service = StreamSpontaneityService(now_fn=lambda: NOW)
        for index in range(200):
            service.record_idle_message(stream, f"Mensaje {index}", topic=f"topic-{index % 7}")
        utterance = "La puerta vuelve a juzgarte."
        stream.recent_idle_messages[-1].update({"text": utterance, "topic": "door", "timestamp": NOW})
        feedback = self.owner_feedback("Otra vez con lo de la puerta.", recent=utterance)
        for index in range(80):
            self.service.apply_feedback(
                stream,
                feedback,
                recent_hebe_utterance=utterance,
                source_event_id=f"feedback-{index}",
                now=NOW + index,
            )
        for index in range(400):
            self.service.evaluate_candidate(
                stream,
                f"Candidate {index % 23} sobre la puerta.",
                now=NOW + index,
                observation={"trace_id": f"soak-{index}", "candidate_id": f"intent-{index}"},
            )
        durable = BehaviorConstraint(
            id="soak-durable", actor="Hebe", behavior_family="semantic_motif",
            behavior_variants=["motif:soak", "rng"], recipient_scope="everyone",
            scope="durable", source_event_id="soak-create", created_at=NOW,
        )
        self.service.register_explicit_constraint(stream, durable)
        self.service.load_durable_constraints(stream)
        self.service.load_durable_constraints(stream)

        self.assertLessEqual(len(stream.recent_idle_messages), 30)
        self.assertLessEqual(len(stream.behavior_adaptation_state["entries"]), 50)
        self.assertEqual(
            len([item for item in stream.active_behavior_blocks if item["id"] == "soak-durable"]),
            1,
        )
        retention = self.observability.snapshot()["retention"]
        self.assertEqual(retention["recent_events"], 64)
        self.assertEqual(self.observability.snapshot()["metrics"]["candidates_evaluated"], 400)
        decayed = self.service.inspection_snapshot(stream, now=NOW + 12 * 60 * 60)
        self.assertEqual(decayed["episodic_fatigue"], [])
        LiveSessionStateManager(logger=lambda _message: None).begin_session(stream, "next-session")
        self.assertEqual(stream.behavior_adaptation_state, {"entries": []})
        self.assertEqual([item["id"] for item in stream.active_behavior_blocks], ["soak-durable"])

    def test_store_failures_have_explicit_reason_codes_without_ram_fallback(self):
        stream = make_stream()
        unavailable = BehaviorAdaptationService(repository=None, observability=self.observability)
        self.assertEqual(unavailable.load_durable_constraints(stream), [])
        self.assertEqual(
            self.observability.snapshot()["recent_events"][-1]["reason_code"],
            "behavior_constraint_store_unavailable",
        )

        class BrokenRepository:
            def list_active(self):
                raise sqlite3.OperationalError("offline")

            def save_durable(self, _constraint):
                raise sqlite3.OperationalError("readonly")

        broken = BehaviorAdaptationService(repository=BrokenRepository(), observability=self.observability)
        with self.assertRaises(sqlite3.OperationalError):
            broken.load_durable_constraints(stream)
        self.assertEqual(
            self.observability.snapshot()["recent_events"][-1]["reason_code"],
            "durable_constraint_load_failed",
        )
        durable = BehaviorConstraint(
            id="broken-durable", actor="Hebe", behavior_family="semantic_motif",
            behavior_variants=["motif:broken", "rng"], recipient_scope="everyone",
            scope="durable", source_event_id="broken-write", created_at=NOW,
        )
        with self.assertRaises(sqlite3.OperationalError):
            broken.register_explicit_constraint(stream, durable)
        self.assertEqual(
            self.observability.snapshot()["recent_events"][-1]["reason_code"],
            "durable_constraint_write_failed",
        )
        self.assertFalse(any(item.get("id") == "broken-durable" for item in stream.active_behavior_blocks))

    def test_policy_constants_and_direct_response_behavior_remain_unchanged(self):
        self.assertEqual(BehaviorAdaptationService.NEGATIVE_HALF_LIFE_SEC, 30 * 60)
        self.assertEqual(BehaviorAdaptationService.POSITIVE_HALF_LIFE_SEC, 20 * 60)
        self.assertEqual(BehaviorAdaptationService.USE_WINDOW_SEC, 45 * 60)
        direct = self.service.evaluate_candidate(
            make_stream(),
            "La puerta requiere una llave.",
            mode="direct_response",
            now=NOW,
            observation={"trace_id": "direct-required"},
        )
        self.assertEqual(direct.action.value, "allow")
        self.assertEqual(direct.reason, "direct_required_response")

    def test_telemetry_failure_cannot_block_or_change_direct_policy(self):
        broken_observability = BehaviorObservability(
            max_recent=8,
            log_fn=lambda _kind, _payload: (_ for _ in ()).throw(OSError("disk full")),
        )
        service = BehaviorAdaptationService(observability=broken_observability)
        stream = make_stream()
        stream.recent_idle_messages = [{
            "text": "historical",
            "topic": "door",
            "timestamp": "not-a-number",
        }]

        direct = service.evaluate_candidate(
            stream,
            "La puerta requiere una llave.",
            mode="direct_response",
            now=NOW,
            observation={"trace_id": "telemetry-failure-direct"},
        )

        self.assertEqual(direct.action.value, "allow")
        self.assertEqual(direct.reason, "direct_required_response")
        metrics = broken_observability.snapshot()["metrics"]
        self.assertGreaterEqual(metrics["telemetry_write_failed"], 1)
        self.assertEqual(metrics["event:telemetry_failure"], 1)


if __name__ == "__main__":
    unittest.main()
