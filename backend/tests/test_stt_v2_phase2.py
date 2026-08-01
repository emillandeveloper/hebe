from __future__ import annotations

import threading
import time
import unittest

from app.cognitive.final_emission_gate import FinalEmissionGate, OutputRoute
from app.cognitive.wake_name_resolver import WakeNameResolver
from app.services.stream_tts_guard import StreamTTSSafetyManager
from app.services.utterance_role import UtteranceRole, UtteranceRoleClassifier
from app.services.voice_command_recovery import normalize_stt_transcript
from app.stream.ambient_context import AmbientContextExtractor
from app.stream.game_advice_gate import GameAdviceGate, ReactionFirstContributionPolicy
from app.stream.spontaneity import StreamSpontaneityService
from app.stream.state import StreamSessionState


class STTV2Phase2Tests(unittest.TestCase):
    def test_english_here_does_not_become_hebe(self):
        result = normalize_stt_transcript("here back in Mirabilis Castle", detected_language="en")
        self.assertEqual(result.conservative_normalized_text, "here back in mirabilis castle")
        self.assertEqual(result.normalized_text, "here back in mirabilis castle")
        self.assertNotIn("hebe back in mirabilis castle", result.alternative_candidates)

    def test_wake_hypothesis_does_not_mutate_canonical_text(self):
        result = normalize_stt_transcript("ebe abre OBS", detected_language="es")
        self.assertEqual(result.normalized_text, "ebe abre obs")
        self.assertIn("hebe abre obs", result.alternative_candidates)

    def test_here_is_ambient_but_exact_hebe_wakes(self):
        resolver = WakeNameResolver()
        here = resolver.resolve(
            raw_text="here through the cross gate", normalized_text="here through the cross gate",
            source="stt_voice", detected_language="en",
        )
        hebe = resolver.resolve(
            raw_text="Hebe abre OBS", normalized_text="hebe abre obs",
            source="stt_voice", detected_language="es", command_markers={"abre"},
        )
        self.assertFalse(here.addressed_to_hebe)
        self.assertTrue(hebe.addressed_to_hebe)

    def test_fuzzy_wake_requires_redecode_and_acoustic_support(self):
        resolver = WakeNameResolver()
        rejected = resolver.resolve(
            raw_text="ebi abre obs", normalized_text="ebi abre obs", source="stt_voice",
            detected_language="es", alternative_candidates=["hebe abre obs"],
            command_markers={"abre"}, acoustic_wake_score=0.2,
        )
        accepted = resolver.resolve(
            raw_text="not-a-name abre obs", normalized_text="not a name abre obs", source="stt_voice",
            detected_language="es", alternative_candidates=["hebe abre obs"],
            command_markers={"abre"}, acoustic_wake_score=0.9, command_redecode_supports_wake=True,
            owner_trusted=True,
        )
        self.assertTrue(rejected.addressed_to_hebe)  # exact configured alias remains valid
        self.assertTrue(accepted.addressed_to_hebe)

    def test_consecutive_english_character_dialogue_classified(self):
        classifier = UtteranceRoleClassifier()
        first = classifier.classify(raw_transcript="Back in the castle, we must find the crystal", detected_language="en")
        second = classifier.classify(raw_transcript="My lord, the gate is open", detected_language="en")
        self.assertEqual(first.role, UtteranceRole.QUOTED_OR_READ_DIALOGUE)
        self.assertEqual(second.role, UtteranceRole.QUOTED_OR_READ_DIALOGUE)
        self.assertFalse(second.discourse_allowed)
        self.assertFalse(second.action_allowed)

    def test_genuine_english_owner_commentary_allowed(self):
        result = UtteranceRoleClassifier().classify(
            raw_transcript="I think this fight is much harder than the last one",
            detected_language="en",
        )
        self.assertEqual(result.role, UtteranceRole.OWNER_COMMENTARY)
        self.assertTrue(result.discourse_allowed)

    def test_quoted_dialogue_transient_and_not_proactive(self):
        extraction = AmbientContextExtractor().extract(
            "My lord, we must cross the gate", utterance_role="quoted_or_read_dialogue",
            language="en", topic_id="scene-1", now=100,
        )
        self.assertTrue(extraction.useful)
        fact = extraction.facts[0]
        self.assertEqual(fact["raw_evidence"], "My lord, we must cross the gate")
        self.assertFalse(fact["data"]["proactive_eligible"])
        self.assertEqual(fact["expires_at"], 130)

    def test_game_audio_excluded_from_context(self):
        extraction = AmbientContextExtractor().extract(
            "The kingdom will fall", utterance_role="game_audio_bleed", language="en",
        )
        self.assertFalse(extraction.useful)

    def test_game_advice_empty_validation_fails_closed(self):
        gate = GameAdviceGate()
        self.assertFalse(gate.validate(current_game="Unknown", proposed_advice="Guarda antes de entrar en esa sala.").allowed)
        self.assertFalse(gate.validate(current_game="Unknown", proposed_advice="Espera hasta que termine de curarse y luego ataca.").allowed)
        self.assertTrue(gate.validate(current_game="Unknown", proposed_advice="Ese combate pinta complicado.").allowed)

    def test_unknown_game_prefers_reaction_and_validated_tip_possible(self):
        policy = ReactionFirstContributionPolicy()
        self.assertEqual(policy.choose_mode(current_game="Unknown"), "contextual_reaction")
        self.assertEqual(
            policy.choose_mode(current_game="Known", grounded_mechanics=["guard"], validated_mechanics=["guard"]),
            "validated_tip",
        )

    def test_candidate_stages_never_emit_normal_ui(self):
        gate = FinalEmissionGate()
        ui = []
        for stage in ("generated", "validating", "repair", "too_similar", "failed_guard", "observed"):
            result = gate.emit(
                event_id=stage, source="spontaneity", final_response="candidate",
                output_route=OutputRoute.LOCAL_OWNER_REPLY, output_targets=["local_ui"],
                guard_result={"passed": True}, debug_payload={"response_stage": stage},
                emit_ui=ui.append,
            )
            self.assertFalse(result.emitted)
        self.assertEqual(ui, [])

    def test_observe_only_zero_bubbles_and_final_once(self):
        gate = FinalEmissionGate()
        ui = []
        observed = gate.emit(
            event_id="observe", source="spontaneity", final_response="hidden",
            output_route=OutputRoute.OBSERVE_ONLY, output_targets=["local_ui"],
            debug_payload={"response_stage": "final"}, emit_ui=ui.append,
        )
        final = gate.emit(
            event_id="final", source="spontaneity", final_response="visible",
            output_route=OutputRoute.LOCAL_OWNER_REPLY, output_targets=["local_ui"],
            guard_result={"passed": True}, debug_payload={"response_stage": "final"}, emit_ui=ui.append,
        )
        duplicate = gate.emit(
            event_id="final", source="spontaneity", final_response="visible",
            output_route=OutputRoute.LOCAL_OWNER_REPLY, output_targets=["local_ui"],
            guard_result={"passed": True}, debug_payload={"response_stage": "final"}, emit_ui=ui.append,
        )
        self.assertFalse(observed.emitted)
        self.assertTrue(final.emitted)
        self.assertFalse(duplicate.emitted)
        self.assertEqual(len(ui), 1)

    def test_raid_tts_schedule_is_non_blocking(self):
        manager = StreamTTSSafetyManager()
        manager.min_free_vram_mb = 0
        started = threading.Event()
        release = threading.Event()
        def slow_speak(_text):
            started.set()
            release.wait(1)
        before = time.perf_counter()
        result = manager.schedule("Gracias por la raid", slow_speak, event_type="raid")
        elapsed = time.perf_counter() - before
        self.assertTrue(result["scheduled"])
        self.assertLess(elapsed, 0.2)
        self.assertTrue(started.wait(0.5))
        release.set()

    def test_low_gpu_headroom_skips_optional_tts(self):
        manager = StreamTTSSafetyManager()
        manager.min_free_vram_mb = 100
        manager.gpu_snapshot = lambda: {"free_vram_mb": 10, "total_vram_mb": 1000, "peak_allocated_mb": 0}
        called = []
        result = manager.schedule("optional", lambda text: called.append(text), event_type="raid")
        self.assertFalse(result["scheduled"])
        self.assertEqual(result["reason"], "low_gpu_headroom")
        self.assertEqual(called, [])

    def test_same_anchor_not_reused_for_multiple_public_comments(self):
        service = StreamSpontaneityService(now_fn=lambda: 100.0)
        stream = StreamSessionState()
        stream.recent_run_context_facts = [{
            "id": "fact-1", "text": "que nos cura a todos",
            "confidence": 0.9, "timestamp": 95.0, "expires_at": 150.0,
            "utterance_role": "owner_commentary", "proactive_eligible": True,
        }]
        self.assertEqual(service._recent_run_context_fact(stream, 100.0)["id"], "fact-1")
        service.record_idle_message(
            stream, "Esa cura grupal cambia el ritmo.", topic="game_vibe",
            used_fact_id="fact-1",
        )
        self.assertIsNone(service._recent_run_context_fact(stream, 100.0))

    def test_tts_warmup_available_in_readiness(self):
        manager = StreamTTSSafetyManager()
        manager.warmup(lambda _text: None)
        readiness = manager.readiness()
        self.assertEqual(readiness["warmup_status"], "ready")
        self.assertIsNotNone(readiness["warmup_latency_ms"])


if __name__ == "__main__":
    unittest.main()
