import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from app.cognitive.twitch_interaction_coordinator import TrollEngagementBudget
from app.stream.behavior_constraints import (
    BehaviorConstraintCompiler, BehaviorConstraintOutputGuard, constraint_matches,
    persist_constraint, render_constraint_confirmation,
)
from app.stream.intent_parser import StreamIntentParser
from app.stream.promotion_recovery import PendingAnswerCapture, PromotionSTTRecovery, stream_ops_no_generic_fallback
from app.stream.viewer_profiles import GrammaticalAgreementGuard, ViewerLinguisticProfile, ViewerLinguisticProfileStore


ALIASES = {"ciber": "cibernoman", "nuria": "nuriiia___", "ivanxi": "ivanxi_kun", "ismael": "ismael_3452"}


def resolve(name):
    login = ALIASES.get(str(name).casefold(), "")
    return {"username": login, "display_name": name, "user_id": f"id:{login}" if login else "",
            "confidence": .99 if login else 0, "candidates": [login] if login else [],
            "reason": "alias" if login else "not_found"}


class Stream:
    active_behavior_blocks = []


class BehaviorConstraintTests(unittest.TestCase):
    def setUp(self):
        self.compiler = BehaviorConstraintCompiler(resolve)

    def test_no_compliments_to_ciber_creates_specific_recipient_constraint(self):
        item = self.compiler.compile("Eve, no le hagas más cumplidos a Ciber").constraint
        self.assertEqual((item.behavior_family, item.recipient_scope, item.recipient_login),
                         ("compliment", "specific_viewer", "cibernoman"))

    def test_no_compliments_to_leo_creates_owner_recipient_constraint(self):
        item = self.compiler.compile("No hagas cumplidos a Leo").constraint
        self.assertEqual(item.recipient_scope, "owner")

    def test_no_compliments_requested_by_ciber_creates_requester_constraint(self):
        item = self.compiler.compile("No hagas cumplidos pedidos por Ciber").constraint
        self.assertEqual((item.requester_scope, item.requester_login, item.recipient_scope),
                         ("specific_viewer", "cibernoman", "any_viewer"))

    def test_no_compliments_to_anyone_creates_global_constraint(self):
        self.assertEqual(self.compiler.compile("No des cumplidos a nadie").constraint.recipient_scope, "everyone")

    def test_unresolved_recipient_does_not_create_wrong_constraint(self):
        result = self.compiler.compile("No hagas cumplidos a Desconocido")
        self.assertTrue(result.needs_clarification)
        self.assertIsNone(result.constraint)

    def test_confirmation_matches_persisted_constraint(self):
        stream = Stream(); stream.active_behavior_blocks = []
        item = self.compiler.compile("No hagas cumplidos a Ciber").constraint
        stored = persist_constraint(stream, item)
        text, invariant = render_constraint_confirmation(item)
        self.assertTrue(invariant["passed"])
        self.assertIn(stored["recipient_display_name"], text)

    def test_direct_ciber_request_does_not_bypass_no_compliments_constraint(self):
        item = self.compiler.compile("No hagas cumplidos a Ciber").constraint
        self.assertTrue(constraint_matches(item, behavior_family="compliment", recipient_login="cibernoman", requester_login="cibernoman"))

    def test_compliment_to_ciber_output_blocked_and_neutral_allowed(self):
        item = self.compiler.compile("No hagas cumplidos a Ciber").constraint.to_dict()
        guard = BehaviorConstraintOutputGuard()
        bad = guard.evaluate([item], intended_recipient="cibernoman", source_viewer="cibernoman",
                             generated_response="Ciber, tienes buen gusto y aguantas como un campeón.")
        good = guard.evaluate([item], intended_recipient="cibernoman", source_viewer="cibernoman",
                              generated_response="Ciber, ese tema queda cerrado.")
        self.assertFalse(bad["passed"]); self.assertTrue(good["passed"])


class BaitTests(unittest.TestCase):
    def test_owner_stop_closes_compliment_topic_and_unrelated_question_allowed(self):
        budget = TrollEngagementBudget()
        self.assertEqual(budget.evaluate(viewer="cibernoman", text="Hebe dime un cumplido")["action"], "allow")
        budget.close_topic_by_owner(viewer="cibernoman", topic="compliment_fishing")
        self.assertEqual(budget.evaluate(viewer="cibernoman", text="otro cumplido")["action"], "boundary")
        self.assertEqual(budget.evaluate(viewer="cibernoman", text="qué juego es este")["action"], "allow")


class PromotionTests(unittest.TestCase):
    def test_haz_una_promo_detects_command_missing_target(self):
        parsed = StreamIntentParser().parse_promotion_request("Eve, haz una promo")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.reason, "missing_target")

    def test_fused_promotion_recovery(self):
        recovery = PromotionSTTRecovery(resolve)
        for text, target in (("Eve haz una promanuria", "nuriiia___"),
                             ("Eve haz una promonuria", "nuriiia___"),
                             ("Eve haz una promoivanxi", "ivanxi_kun")):
            result = recovery.recover(text, trusted_owner=True, addressed_to_hebe=True)
            self.assertTrue(result.recovered, text); self.assertEqual(result.resolved_target, target)

    def test_unknown_fused_suffix_clarifies_and_never_generic(self):
        result = PromotionSTTRecovery(resolve).recover("Eve haz una promowhatever", trusted_owner=True, addressed_to_hebe=True)
        self.assertTrue(result.command_candidate); self.assertFalse(result.recovered)
        self.assertFalse(stream_ops_no_generic_fallback(result, routed_to_generic=True)["passed"])

    def test_pending_answer_capture_starts_after_tts_and_buffers(self):
        capture = PendingAnswerCapture("p1", starts_after_tts_end=100)
        self.assertTrue(capture.buffer("A Ivanxi", source="stt_voice", timestamp=99))
        capture.mark_tts_completed(101)
        self.assertEqual(capture.next_answer(now=102), "A Ivanxi")
        self.assertFalse(capture.buffer("esto es un monólogo demasiado largo para ser objetivo", source="stt_voice"))
        self.assertFalse(capture.buffer("Ivanxi", source="ambient_stt"))


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = str(Path(self.tmp.name) / "profiles.db")
        self.store = ViewerLinguisticProfileStore(lambda: sqlite3.connect(path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_viewer_profile_persists_by_twitch_user_id_and_rename(self):
        self.store.apply_evidence(twitch_user_id="42", login="nuria_old", candidate_gender="feminine",
                                  confidence=1, source_type="owner_confirmed")
        profile = self.store.get(twitch_user_id="42")
        profile.login = "nuria_new"; self.store.save(profile)
        self.assertEqual(self.store.get(twitch_user_id="42").preferred_grammatical_gender, "feminine")
        self.assertEqual(self.store.get(twitch_user_id="42").login, "nuria_new")

    def test_unknown_profile_defaults_neutral_guard(self):
        profile = self.store.get(twitch_user_id="99", login="unknown")
        result = GrammaticalAgreementGuard().evaluate("Tranquila, campeona", viewer="unknown", profile=profile)
        self.assertEqual(result["action"], "neutralize")
        self.assertNotIn("tranquila", result["text"].casefold())

    def test_confirmed_agreement_and_hebe_self_unchanged(self):
        guard = GrammaticalAgreementGuard()
        feminine = ViewerLinguisticProfile("1", "nuria", preferred_grammatical_gender="feminine")
        masculine = ViewerLinguisticProfile("2", "ismael", preferred_grammatical_gender="masculine")
        self.assertIn("tranquila", guard.evaluate("Nuria, tranquilo", viewer="nuria", profile=feminine)["text"].casefold())
        self.assertIn("tranquilo", guard.evaluate("Ismael, tranquila", viewer="ismael", profile=masculine)["text"].casefold())
        self.assertEqual(guard.evaluate("Estoy tranquila", viewer="nuria", profile=masculine, refers_to_hebe=True)["text"], "Estoy tranquila")

    def test_low_conflict_does_not_overwrite_confirmed(self):
        self.store.apply_evidence(twitch_user_id="42", login="nuria", candidate_gender="feminine", confidence=1, source_type="owner_confirmed")
        profile, action = self.store.apply_evidence(twitch_user_id="42", login="nuria", candidate_gender="masculine", confidence=.6, source_type="heuristic")
        self.assertEqual(action, "conflict"); self.assertEqual(profile.preferred_grammatical_gender, "feminine")


if __name__ == "__main__":
    unittest.main()
