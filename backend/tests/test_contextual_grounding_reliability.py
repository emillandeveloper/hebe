import io
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from app.cognitive.final_emission_gate import FinalEmissionGate, OutputRoute
from app.hebe_engine import HebeEngine
from app.services.direct_stt_command import DirectSTTCommandResult, parse_direct_stt_command
from app.services.stt_whisper import STTService
from app.services.utterance_role import UtteranceRole, UtteranceRoleClassifier
from app.stream.ambient_context import AmbientContextExtractor
from app.stream.companion_loop import StreamCompanionLoop
from app.stream.evidence_entailment import EvidenceEntailmentGuard
from app.stream.game_advice_gate import GameAdviceGate
from app.stream.intent_parser import StreamIntentParser


class ContextualGroundingReliabilityTests(unittest.TestCase):
    def test_guard_suppress_overrides_route_and_all_outputs(self):
        calls = {"ui": 0, "twitch": 0, "tts": 0}
        result = FinalEmissionGate().emit(
            event_id="suppressed", source="spontaneous", final_response="¿Seguimos?",
            output_route=OutputRoute.STREAM_TTS_REPLY,
            output_targets=["local_ui", "twitch_chat", "stream_tts"],
            guard_result={
                "passed": False, "action": "suppress",
                "violations": ["unnecessary_followup_question"],
                "source_guards": ["persona_quality"],
                "final_route_override": "suppress",
            },
            debug_payload={"response_stage": "final"},
            emit_ui=lambda _: calls.__setitem__("ui", calls["ui"] + 1),
            send_twitch=lambda _: calls.__setitem__("twitch", calls["twitch"] + 1),
            speak=lambda _: calls.__setitem__("tts", calls["tts"] + 1),
        )
        self.assertTrue(result.suppressed)
        self.assertEqual(result.route, "suppress")
        self.assertEqual(calls, {"ui": 0, "twitch": 0, "tts": 0})

    def test_ambient_referents_and_semantic_categories(self):
        extractor = AmbientContextExtractor()
        enemy = extractor.extract("Bueno, a ver, estos son nivel bajo.", now=100).facts
        self.assertEqual(enemy[0]["extracted_subject"], "enemies")
        self.assertEqual(enemy[0]["extracted_predicate"], "low_level")
        self.assertNotIn("challenge_constraint", {fact["category"] for fact in enemy})
        hp = extractor.extract("No se le baja la barra de vida.", now=101).facts
        self.assertEqual({fact["category"] for fact in hp}, {"uncertain_combat_observation"})
        self.assertIn("autopotion", hp[0]["unsupported_claims"])
        owner = extractor.extract("Estoy a nivel uno.", now=102).facts
        self.assertEqual(owner[0]["category"], "challenge_constraint")
        self.assertEqual(owner[0]["extracted_subject"], "owner_player")
        for fact in enemy + hp + owner:
            self.assertTrue(fact["raw_text"])
            self.assertIn("conservative_normalized_text", fact)

    def test_spontaneous_scene_contains_raw_and_structured_evidence(self):
        now = time.time()
        fact = AmbientContextExtractor().extract(
            "Estos son nivel bajo.", now=now, topic_id="fight",
        ).facts[0]
        state = SimpleNamespace(
            current_discourse_topic={"topic_id": "fight"},
            recent_run_context_facts=[fact],
        )
        anchor = StreamCompanionLoop()._selected_anchor(state, {}, now + 1)
        evidence = anchor["evidence"]
        self.assertEqual(evidence["raw_owner_fragments"], ["Estos son nivel bajo."])
        self.assertEqual(evidence["extracted_subject"], "enemies")
        self.assertTrue(evidence["supported_claims"])

    def test_heuristic_summary_cannot_self_validate_claim(self):
        result = GameAdviceGate().validate(
            current_game="Unknown Game",
            proposed_advice="La autopoción se ha activado.",
            source_evidence=[{
                "evidence_type": "heuristic_summary",
                "evidence_id": "summary:1",
                "exact_supporting_text": "autopotion",
                "confidence": 0.9,
            }],
        )
        self.assertFalse(result.allowed)
        explicit = GameAdviceGate().validate(
            current_game="Unknown Game",
            proposed_advice="La autopoción se ha activado.",
            source_evidence=[{
                "evidence_type": "raw_owner_evidence",
                "evidence_id": "ambient:1",
                "exact_supporting_text": "Se activó la autopoción.",
                "confidence": 0.95,
            }],
        )
        self.assertTrue(explicit.allowed)
        self.assertEqual(explicit.validated_claims[0]["evidence_id"], "ambient:1")

    def test_evidence_entailment_rejects_invention_and_wrong_referent(self):
        guard = EvidenceEntailmentGuard()
        hp = guard.evaluate("Autopoción activada.", {
            "raw_owner_fragments": ["No se le baja la barra de vida."],
            "extracted_subject": "enemies",
            "unsupported_claims": ["autopotion", "healing"],
        })
        self.assertEqual(hp.result, "unsupported")
        self.assertEqual(hp.action, "repair")
        referent = guard.evaluate("Leo está bajo de nivel.", {
            "raw_owner_fragments": ["Estos son nivel bajo."],
            "extracted_subject": "enemies",
        })
        self.assertEqual(referent.result, "wrong_referent")
        self.assertEqual(referent.action, "suppress")
        reaction = guard.evaluate("Eso tiene pinta de ser durísimo.", {
            "raw_owner_fragments": ["No se le baja la barra de vida."],
            "extracted_subject": "enemies",
        })
        self.assertEqual(reaction.result, "reasonable_low-risk_reaction")
        self.assertTrue(reaction.passed)

    def test_direct_stt_terminal_is_compare_and_set(self):
        engine = object.__new__(HebeEngine)
        result = DirectSTTCommandResult(event_id="obs-one", detected_intent_family="application_action")
        with patch("app.hebe_engine.emit"):
            self.assertTrue(engine._log_direct_stt_outcome(
                result, outcome="action_executed", reason="application_launch_succeeded",
                action_receipt={"action_type": "open_application", "target": "obs", "executor_invoked": True, "success": True, "timestamp": time.time()},
            ))
            self.assertFalse(engine._log_direct_stt_outcome(
                result, outcome="rejected", reason="application_parser_or_resolver_failed",
            ))
        self.assertEqual(result.final_outcome, "action_executed")

    def test_app_variants_preserve_the_target_for_canonical_discovery(self):
        for raw in (
            "Hebe, abre melonDS",
            "Ebe, abre Melón DS",
            "Eve, abre Melón de Ese",
        ):
            parsed = parse_direct_stt_command(raw)
            self.assertEqual(parsed.action_verb, "open")
            self.assertTrue(parsed.raw_target)

    def test_command_candidate_requires_wake_or_known_actionable_target(self):
        stt = object.__new__(STTService)
        for ordinary in (
            "Hi hi, hello everyone and welcome to another stream",
            "y me manda a tomar por culo",
            "seguramente fuera de stream intenté subir de nivel",
        ):
            self.assertFalse(stt._looks_like_command_candidate(ordinary))
        self.assertTrue(stt._looks_like_command_candidate("abre OBS"))
        self.assertTrue(stt._looks_like_command_candidate("haz promo a Nuria"))

    def test_sustained_dialogue_and_owner_commentary_are_discriminated(self):
        classifier = UtteranceRoleClassifier()
        classifier.classify(raw_transcript="The road ahead is dark and cold.", detected_language="en")
        classifier.classify(raw_transcript="We cannot turn back after all this.", detected_language="en")
        third = classifier.classify(
            raw_transcript="The captain will meet us at the gate.", detected_language="en",
        )
        self.assertEqual(third.role, UtteranceRole.QUOTED_OR_READ_DIALOGUE)
        owner = classifier.classify(
            raw_transcript="I think this game looks fantastic.", detected_language="en",
        )
        self.assertEqual(owner.role, UtteranceRole.OWNER_COMMENTARY)

    def test_promotion_prefilter_skips_ordinary_speech(self):
        parser = StreamIntentParser()
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertIsNone(parser.parse_promotion_request("The captain waits at the gate."))
            self.assertIsNone(parser.parse_promotion_request("Creo que esta zona es difícil."))
            self.assertIsNotNone(parser.parse_promotion_request("haz promo a Nuria"))
        logged = output.getvalue()
        self.assertEqual(logged.count("[HEBE][PROMOTION_PARSE_ATTEMPT]"), 1)
        self.assertEqual(logged.count("[HEBE][PROMOTION_PARSE_SKIP]"), 2)
        self.assertTrue(parser.promotion_prefilter("Nuria", pending_active=True))


if __name__ == "__main__":
    unittest.main()
