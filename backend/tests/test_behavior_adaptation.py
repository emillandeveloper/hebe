from __future__ import annotations

from types import SimpleNamespace
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.cognitive.input_interpretation import InputInterpreter, InputSpeechAct
from app.stream.behavior_adaptation import (
    AdaptationAction,
    BehaviorAdaptationService,
    FeedbackKind,
    ReferentProvenance,
    semantic_similarity,
)
from app.stream.live_runtime import LiveSessionStateManager
from app.stream.speech_intents import SpeechIntentManager
from app.stream import proactive
from app.stream.spontaneity import StreamSpontaneityService
from app.replay.behavior_policy import BehaviorCalibrationCase, BehaviorPolicyCalibrationReplay
from app.replay.migrations import MigrationRunner
from app.stream.behavior_constraint_store import (
    BehaviorConstraintRepository,
    behavior_constraint_migrations,
)


NOW = 10_000.0


def stream_with(*messages: str) -> SimpleNamespace:
    return SimpleNamespace(
        recent_idle_messages=[
            {"text": text, "topic": "optional_banter", "timestamp": NOW - (len(messages) - index) * 90}
            for index, text in enumerate(messages)
        ],
        active_behavior_blocks=[],
        behavior_adaptation_state={"entries": []},
        current_discourse_topic="",
    )


class BehaviorAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "behavior_constraints.sqlite3"
        self.connection_factory = lambda: sqlite3.connect(self.db_path)
        MigrationRunner(self.connection_factory).migrate(behavior_constraint_migrations())
        self.repository = BehaviorConstraintRepository(self.connection_factory)
        self.service = BehaviorAdaptationService(repository=self.repository)
        self.interpreter = InputInterpreter()

    def tearDown(self):
        self.tempdir.cleanup()

    def owner_feedback(self, text: str, *, recent: str = ""):
        result = self.interpreter.interpret(
            raw_text=text,
            source="stt_voice",
            authority="owner",
            addressed_to_hebe=True,
            recent_hebe_utterance=recent,
        )
        self.assertEqual(result.speech_act, InputSpeechAct.OWNER_FEEDBACK)
        return result

    def test_a_isolated_criticism_temporarily_suppresses_recent_motif(self):
        stream = stream_with("Mira esa puerta, seguro que quiere que la abras.")
        result = self.service.apply_feedback(
            stream,
            self.owner_feedback("Otra vez con lo de abre la puerta, ya cansa."),
            now=NOW,
        )
        decision = self.service.evaluate_candidate(stream, "Dale a esa puerta, Leo.", now=NOW + 1)
        self.assertTrue(result.applied)
        self.assertEqual(result.kind, FeedbackKind.EPISODIC_NEGATIVE)
        self.assertEqual(decision.action, AdaptationAction.SUPPRESS)

    def test_b_repeated_criticism_increases_weight_and_suppression_duration(self):
        stream = stream_with("Culpa al RNG de esa tirada.")
        first = self.service.apply_feedback(stream, self.owner_feedback("Otra vez con lo del RNG."), now=NOW)
        first_entry = dict(stream.behavior_adaptation_state["entries"][0])
        second = self.service.apply_feedback(stream, self.owner_feedback("Otra vez con lo del RNG, ya cansa."), now=NOW + 30)
        second_entry = stream.behavior_adaptation_state["entries"][0]
        self.assertTrue(first.applied and second.applied)
        self.assertGreater(second_entry["negative_weight"], first_entry["negative_weight"])
        self.assertGreater(second_entry["suppress_until"], first_entry["suppress_until"])

    def test_c_positive_feedback_does_not_cause_immediate_repetition(self):
        utterance = "Me vas a cobrar entrada por fallar así."
        stream = stream_with(utterance)
        result = self.service.apply_feedback(
            stream,
            self.owner_feedback("Buenísima esa.", recent=utterance),
            now=NOW,
            recent_hebe_utterance=utterance,
        )
        decision = self.service.evaluate_candidate(stream, "¿Vas a cobrar entrada otra vez?", now=NOW + 15)
        self.assertEqual(result.kind, FeedbackKind.EPISODIC_POSITIVE)
        self.assertNotEqual(decision.action, AdaptationAction.ALLOW)

    def test_d_clear_positive_feedback_relaxes_matching_negative_state(self):
        utterance = "No cantes esa victoria todavía."
        stream = stream_with(utterance)
        self.service.apply_feedback(stream, self.owner_feedback("Otra vez con lo de no cantes."), now=NOW)
        before = stream.behavior_adaptation_state["entries"][0]["negative_weight"]
        result = self.service.apply_feedback(
            stream,
            self.owner_feedback("Buenísima esa.", recent=utterance),
            recent_hebe_utterance=utterance,
            now=NOW + 10,
        )
        after = stream.behavior_adaptation_state["entries"][0]["negative_weight"]
        self.assertTrue(result.applied)
        self.assertLess(after, before)

    def test_e_explicit_today_instruction_creates_session_constraint(self):
        utterance = "Abre esa puerta imaginaria."
        stream = stream_with(utterance)
        result = self.service.apply_feedback(
            stream,
            self.owner_feedback("No vuelvas a hacer esa broma hoy."),
            recent_hebe_utterance=utterance,
            now=NOW,
        )
        self.assertEqual(result.kind, FeedbackKind.EXPLICIT_TEMPORARY_INSTRUCTION)
        self.assertEqual(stream.active_behavior_blocks[0]["scope"], "current_stream")

    def test_f_explicit_durable_preference_creates_durable_constraint(self):
        utterance = "Otra broma sobre cobrar entrada."
        stream = stream_with(utterance)
        result = self.service.apply_feedback(
            stream,
            self.owner_feedback("No quiero que vuelvas a hacer ese tipo de bromas sobre cobrar entrada."),
            recent_hebe_utterance=utterance,
            now=NOW,
        )
        self.assertTrue(result.applied)
        self.assertEqual(result.kind, FeedbackKind.EXPLICIT_DURABLE_PREFERENCE)
        self.assertEqual(stream.active_behavior_blocks[0]["scope"], "durable")

    def test_g_unambiguous_reversal_removes_matching_constraint(self):
        utterance = "No cantes todavía."
        stream = stream_with(utterance)
        self.service.apply_feedback(
            stream,
            self.owner_feedback("No vuelvas a hacer esa broma hoy."),
            recent_hebe_utterance=utterance,
            now=NOW,
        )
        result = self.service.apply_feedback(
            stream,
            self.owner_feedback("Bah, puedes volver a hacerla."),
            recent_hebe_utterance=utterance,
            now=NOW + 30,
        )
        self.assertEqual(result.kind, FeedbackKind.CORRECTION_REVERSAL)
        self.assertEqual(stream.active_behavior_blocks, [])

    def test_h_unresolved_feedback_is_observable_but_does_not_mutate_state(self):
        stream = stream_with()
        result = self.service.apply_feedback(
            stream,
            self.owner_feedback("Me ha gustado esa respuesta."),
            now=NOW,
        )
        self.assertFalse(result.applied)
        self.assertEqual(result.referent.provenance, ReferentProvenance.UNRESOLVED)
        self.assertEqual(stream.behavior_adaptation_state["entries"], [])
        self.assertEqual(stream.last_feedback_application["reason"], "referent_unresolved")

    def test_i_viewer_feedback_cannot_modify_owner_constraints(self):
        stream = stream_with("Mira esa puerta.")
        viewer = self.interpreter.interpret(
            raw_text="Otra vez con lo de la puerta.",
            source="twitch_viewer",
            authority="viewer",
            addressed_to_hebe=True,
        )
        result = self.service.apply_feedback(stream, viewer, now=NOW)
        self.assertFalse(result.applied)
        self.assertEqual(stream.active_behavior_blocks, [])

    def test_j_ambiguous_sarcasm_does_not_persist_durable_preference(self):
        stream = stream_with("Una broma sobre el RNG.")
        result = self.service.apply_feedback(
            stream,
            self.owner_feedback("Sí, claro, no quiero que vuelvas a hacer bromas sobre el RNG, supongo."),
            now=NOW,
        )
        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "durable_evidence_insufficient")
        self.assertEqual(stream.active_behavior_blocks, [])

    def test_k_unrelated_proactive_topic_remains_available(self):
        stream = stream_with("Abre esa puerta.")
        self.service.apply_feedback(stream, self.owner_feedback("Otra vez con lo de la puerta."), now=NOW)
        decision = self.service.evaluate_candidate(stream, "Qué música tan tranquila tiene esta zona.", now=NOW + 1)
        self.assertEqual(decision.action, AdaptationAction.ALLOW)

    def test_l_direct_answer_bypasses_optional_behavior_adaptation(self):
        stream = stream_with("Abre esa puerta.")
        self.service.apply_feedback(stream, self.owner_feedback("Otra vez con lo de la puerta."), now=NOW)
        decision = self.service.evaluate_candidate(
            stream,
            "La puerta no se abre hasta activar el interruptor.",
            mode="direct_response",
            now=NOW + 1,
        )
        self.assertEqual(decision.action, AdaptationAction.ALLOW)
        self.assertEqual(decision.reason, "direct_required_response")

    def test_m_session_restart_clears_episodic_state_and_session_constraint(self):
        stream = stream_with("Abre esa puerta.")
        self.service.apply_feedback(stream, self.owner_feedback("No vuelvas a hacer esa broma hoy."), now=NOW)
        LiveSessionStateManager(logger=lambda _message: None).begin_session(stream, "next")
        self.assertEqual(stream.behavior_adaptation_state, {"entries": []})
        self.assertEqual(stream.active_behavior_blocks, [])
        row = BehaviorPolicyCalibrationReplay(self.service).run(
            stream,
            [BehaviorCalibrationCase("session_restart", "Otra broma sobre la puerta.")],
            now=NOW + 1,
        )[0]
        self.assertEqual(row["decision"], "allow")
        self.assertEqual(row["constraint"], "")

    def test_episodic_negative_feedback_decays_and_recovers_with_time(self):
        stream = stream_with("Abre esa puerta.")
        self.service.apply_feedback(stream, self.owner_feedback("Otra vez con lo de la puerta."), now=NOW)

        decision = self.service.evaluate_candidate(stream, "Esa puerta vuelve a aparecer.", now=NOW + 2 * 60 * 60)

        self.assertEqual(decision.action, AdaptationAction.ALLOW)
        self.assertLess(decision.negative_weight, 0.1)

    def test_n_durable_behavior_constraint_survives_process_restart(self):
        utterance = "Otra broma sobre el RNG."
        stream = stream_with(utterance)
        self.service.apply_feedback(
            stream,
            self.owner_feedback("No quiero que vuelvas a hacer bromas sobre el RNG."),
            recent_hebe_utterance=utterance,
            now=NOW,
        )
        restarted_stream = stream_with()
        restarted_service = BehaviorAdaptationService(
            repository=BehaviorConstraintRepository(self.connection_factory),
        )
        loaded = restarted_service.load_durable_constraints(restarted_stream)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(restarted_stream.active_behavior_blocks[0]["scope"], "durable")
        decision = restarted_service.evaluate_candidate(
            restarted_stream, "El RNG vuelve a tener la culpa.", now=NOW + 1,
        )
        self.assertEqual(decision.action, AdaptationAction.SUPPRESS)
        row = BehaviorPolicyCalibrationReplay(restarted_service).run(
            restarted_stream,
            [BehaviorCalibrationCase("durable_restart", "El RNG vuelve a tener la culpa.")],
            now=NOW + 1,
        )[0]
        self.assertTrue(row["constraint"])
        self.assertFalse(row["final_emission"])

    def test_durable_reversal_is_retired_and_survives_process_restart(self):
        utterance = "Otra broma sobre cantar victoria."
        stream = stream_with(utterance)
        created = self.service.apply_feedback(
            stream,
            self.owner_feedback("No quiero que vuelvas a hacer bromas sobre cantar victoria."),
            recent_hebe_utterance=utterance,
            source_event_id="feedback-create",
            now=NOW,
        )
        restarted_stream = stream_with(utterance)
        restarted = BehaviorAdaptationService(
            repository=BehaviorConstraintRepository(self.connection_factory),
        )
        restarted.load_durable_constraints(restarted_stream)
        reversed_result = restarted.apply_feedback(
            restarted_stream,
            self.owner_feedback("Bah, puedes volver a hacerlas."),
            recent_hebe_utterance=utterance,
            source_event_id="feedback-reverse",
            now=NOW + 60,
        )
        after_second_restart = BehaviorConstraintRepository(self.connection_factory)
        self.assertEqual(restarted_stream.active_behavior_blocks, [])
        self.assertIn(created.constraint_id, reversed_result.constraint_id)
        self.assertEqual(after_second_restart.list_active(), [])
        self.assertEqual(after_second_restart.list_all()[0].status, "RETIRED")
        self.assertEqual(
            [item["event_type"] for item in after_second_restart.events(created.constraint_id)],
            ["CREATED", "RETIRED"],
        )
        final_stream = stream_with()
        BehaviorAdaptationService(repository=after_second_restart).load_durable_constraints(final_stream)
        self.assertEqual(final_stream.active_behavior_blocks, [])

    def test_semantic_similarity_groups_variants_without_named_registry(self):
        self.assertGreaterEqual(
            semantic_similarity("Mira esa puerta", "El jefe estará detrás de la puerta"),
            0.34,
        )
        self.assertLess(
            semantic_similarity("Mira esa puerta", "La música de esta zona es preciosa"),
            0.34,
        )

    def test_full_door_replay_suppresses_fourth_variant_only(self):
        stream = stream_with(
            "Abre la puerta como si escondiera un jefe.",
            "Mira esa puerta, te está juzgando.",
            "Dale a la puerta antes de que cobre entrada.",
        )
        feedback = self.owner_feedback(
            "Mira, otra vez con lo de abre la puerta esa. Lleva todo el puto stream diciéndome "
            "que abra la puerta, pero te quieres callar con la puta puerta que no hay ninguna puerta."
        )
        applied = self.service.apply_feedback(stream, feedback, now=NOW)
        fourth = self.service.evaluate_candidate(stream, "El jefe seguro que está detrás de esa puerta.", now=NOW + 1)
        unrelated = self.service.evaluate_candidate(stream, "Ese combate ha tenido un timing perfecto.", now=NOW + 1)
        direct = self.service.evaluate_candidate(
            stream,
            "Esa puerta requiere una llave del inventario.",
            mode="direct_response",
            now=NOW + 1,
        )
        self.assertTrue(applied.applied)
        self.assertIn(applied.referent.provenance, {ReferentProvenance.EXPLICIT_TEXT, ReferentProvenance.RECENT_HEBE_UTTERANCE})
        self.assertEqual(fourth.action, AdaptationAction.SUPPRESS)
        self.assertEqual(unrelated.action, AdaptationAction.ALLOW)
        self.assertEqual(direct.action, AdaptationAction.ALLOW)

    def test_downrank_changes_selection_between_optional_candidates(self):
        stream = stream_with(
            "Mira esa puerta.",
            "La puerta vuelve a juzgarte.",
            "Esa puerta te cobra entrada.",
            "Hay otro jefe detrás de la puerta.",
        )
        manager = SpeechIntentManager(now_fn=lambda: NOW)
        fatigued = manager.create(
            intent_type="BANTER", topic="puerta", subject_ref="broma puerta",
            value=0.95, urgency=0.5,
        )
        alternative = manager.create(
            intent_type="BANTER", topic="musica", subject_ref="música de la zona",
            value=0.75, urgency=0.5,
        )

        arbitration = manager.arbitrate(
            owner_voice_active=False,
            owner_utterance_ended_at=NOW - 5,
            tts_active=False,
            now=NOW,
            rank_policy=lambda intent: self.service.evaluate_candidate(
                stream,
                f"{intent.subject_ref} {intent.topic}",
                topic=intent.topic,
                now=NOW,
            ).to_dict(),
        )

        self.assertEqual(arbitration.intent.id, alternative.id)
        ranked = {item["intent_id"]: item for item in arbitration.ranked_candidates}
        self.assertEqual(ranked[fatigued.id]["policy"]["action"], "downrank")
        self.assertLess(ranked[fatigued.id]["adjusted_score"], ranked[alternative.id]["adjusted_score"])

    def test_post_generation_check_blocks_drift_into_suppressed_motif(self):
        stream = stream_with("Mira esa puerta.")
        self.service.apply_feedback(stream, self.owner_feedback("Otra vez con lo de la puerta."), now=NOW)
        candidate = self.service.evaluate_candidate(stream, "La música de esta zona.", now=NOW + 1)
        generated = self.service.validate_generated_output(
            stream, "El jefe seguro que está detrás de esa puerta.", now=NOW + 1,
        )
        self.assertEqual(candidate.action, AdaptationAction.ALLOW)
        self.assertEqual(generated.action, AdaptationAction.SUPPRESS)
        self.assertEqual(generated.stage, "generated_output")

    def test_old_semantic_cooldown_classifier_is_absent(self):
        self.assertFalse(hasattr(proactive, "semantic_cooldown_key"))
        self.assertFalse(hasattr(proactive, "cooldown_active"))

    def test_manual_style_registry_is_not_a_second_suppression_source(self):
        stream = stream_with()
        decision = self.service.evaluate_candidate(stream, "Un café antes de continuar.", now=NOW)
        self.assertEqual(decision.action, AdaptationAction.ALLOW)
        self.assertFalse(hasattr(StreamSpontaneityService(), "motif_on_cooldown"))
        self.assertFalse(hasattr(stream, "recent_style_motifs"))

    def test_calibration_replay_records_complete_decision_matrix(self):
        stream = stream_with(
            "Culpa al RNG de esa tirada.",
            "Otra vez el RNG haciendo de las suyas.",
            "El RNG ha decidido el combate.",
            "La puerta te vuelve a juzgar.",
            "No cantes victoria todavía.",
        )
        self.service.apply_feedback(
            stream,
            self.owner_feedback("Otra vez con lo del RNG, ya cansa."),
            now=NOW,
        )
        self.service.apply_feedback(
            stream,
            self.owner_feedback("Buenísima esa.", recent="No cantes victoria todavía."),
            recent_hebe_utterance="No cantes victoria todavía.",
            now=NOW + 1,
        )
        rows = BehaviorPolicyCalibrationReplay(self.service).run(
            stream,
            [
                BehaviorCalibrationCase("door", "La puerta vuelve a cobrar entrada."),
                BehaviorCalibrationCase("rng", "El RNG vuelve a tener la culpa."),
                BehaviorCalibrationCase("singing", "No cantes victoria todavía."),
                BehaviorCalibrationCase("healing", "Quizá toque curarse después del combate."),
                BehaviorCalibrationCase("repeated_joke", "Otra broma con el RNG."),
                BehaviorCalibrationCase("distinct_topic", "La música de esta zona funciona muy bien."),
                BehaviorCalibrationCase("direct_rng", "¿Qué significa RNG?", mode="direct_response"),
            ],
            now=NOW,
        )
        self.assertEqual(
            {row["case"] for row in rows},
            {"door", "rng", "singing", "healing", "repeated_joke", "distinct_topic", "direct_rng"},
        )
        self.assertTrue(all({
            "candidate", "motif", "uses", "fatigue", "negative_weight",
            "positive_weight", "constraint", "decision", "final_emission",
        } <= set(row) for row in rows))
        self.assertEqual(next(row for row in rows if row["case"] == "direct_rng")["decision"], "allow")

    def test_lexical_similarity_known_false_positive_shared_word(self):
        self.assertGreaterEqual(
            semantic_similarity("jefe detrás de la puerta", "configura la puerta de enlace de red"),
            0.25,
        )

    def test_lexical_similarity_known_false_negative_synonym_without_overlap(self):
        self.assertEqual(semantic_similarity("culpa al rng", "ha sido cosa del azar"), 0.0)


if __name__ == "__main__":
    unittest.main()
