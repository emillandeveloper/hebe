from __future__ import annotations

import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.integrations.twitch.target_resolver import TwitchTargetResolver
from app.hebe_engine import HebeEngine
from app.services.direct_stt_command import DirectSTTCommandResult
from app.services.stream_tts_guard import StreamTTSSafetyManager
from app.stream.ambient_context import AmbientContextExtractor
from app.stream.game_advice_gate import GameAdviceGate
from app.stream.promotions import PromotionProfileManager, PromotionStore, ViewerPromotionProfile
from app.stream.social_response_guards import ChannelRetentionGuard, SocialAuthorityCommitmentGuard
from app.stream.viewer_profiles import GrammaticalAgreementGuard, ViewerLinguisticProfile


class _Chat:
    def recent_users(self):
        return [("dksgaminges", "DKSGamingES")]


class HebeLiveV12EvidenceTests(unittest.TestCase):
    def test_solo_queda_ese_does_not_invent_mechanics(self):
        result = AmbientContextExtractor().extract("solo queda ese", now=10)
        joined = repr(result.facts).casefold()
        self.assertIn("one_remaining", joined)
        supported = " ".join(item for fact in result.facts for item in fact.get("supported_claims", []))
        for forbidden in ("auto_healing", "counterattack", "healing", "regeneration"):
            self.assertNotIn(forbidden, supported)

    def test_substantive_mechanic_nouns_are_extracted(self):
        claims = GameAdviceGate().extract_substantive_claims(
            "Cuidado con contraataques y curaciones automaticas."
        )
        self.assertIn("counterattack", claims)
        self.assertIn("automatic_healing", claims)


class HebeLiveV12PromotionTests(unittest.TestCase):
    def test_action_executed_requires_success_receipt(self):
        engine = object.__new__(HebeEngine)
        result = DirectSTTCommandResult(event_id="no-receipt", detected_intent_family="stream_operation")
        with patch("app.hebe_engine.emit"):
            engine._log_direct_stt_outcome(result, outcome="action_executed", reason="stream_operation_handled")
        self.assertEqual(result.final_outcome, "action_failed")
        self.assertEqual(result.rejection_reason, "missing_success_action_receipt")

    def test_adk_prefers_active_dksgaminges(self):
        resolved = TwitchTargetResolver(_Chat(), SimpleNamespace()).resolve_user_details("ADK")
        self.assertEqual(resolved.username, "dksgaminges")
        self.assertEqual(resolved.reason, "phonetic_active_chatter")

    def test_orphan_profile_is_invalidated(self):
        connection = sqlite3.connect(":memory:")
        store = PromotionStore(connection=connection)
        connection.execute(
            "INSERT INTO viewer_promotion_profiles(twitch_user_id,current_login,display_name,known_aliases_json,auto_promo_mode,created_by,created_at,updated_at,cooldown_hours,owner_locked,active) VALUES ('login:adk','adk','adk','[]','first_message_each_stream','owner_command','','',0,1,1)"
        )
        connection.commit()
        self.assertEqual(store.invalidate_orphaned_profiles(), 1)
        self.assertFalse(store.get_profile(login="adk").active)

    def test_unconfirmed_identity_cannot_be_learned(self):
        store = PromotionStore(connection=sqlite3.connect(":memory:"))
        profile = PromotionProfileManager(store).learn_after_success(
            twitch_user_id="login:adk", login="adk", source_promotion_event="promo-x"
        )
        self.assertIsNone(profile)


class HebeLiveV12SocialAndTTSTests(unittest.TestCase):
    def test_viewer_commitment_is_rewritten(self):
        result = SocialAuthorityCommitmentGuard().evaluate("No lo diré.", requester_is_owner=False)
        self.assertEqual(result.action, "rewrite")

    def test_owner_commitment_is_allowed(self):
        result = SocialAuthorityCommitmentGuard().evaluate("No lo diré.", requester_is_owner=True)
        self.assertTrue(result.passed)

    def test_channel_leave_instruction_is_rewritten(self):
        self.assertEqual(ChannelRetentionGuard().evaluate("Cambia de canal si te aburres.").action, "rewrite")

    def test_neutralization_punctuation_is_grammatical(self):
        profile = ViewerLinguisticProfile(twitch_user_id="1", login="ismael")
        guard = GrammaticalAgreementGuard()
        for text in ("Ismael, tranquilo: empezó ahora.", "tranquilo, Ismael — seguimos."):
            repaired = guard.evaluate(text, viewer="ismael", profile=profile)["text"]
            self.assertNotIn("Ismael,:", repaired)
            self.assertFalse(repaired.startswith(","))

    def test_disabled_raid_tts_never_invokes_speaker(self):
        manager = StreamTTSSafetyManager()
        speak = Mock()
        with patch("builtins.print") as printer:
            result = manager.schedule("hola", speak, event_type="raid", output_enabled=False)
        self.assertFalse(result["scheduled"])
        speak.assert_not_called()
        self.assertIn("stream_tts_disabled", repr(printer.call_args_list))
        manager.shutdown()


if __name__ == "__main__":
    unittest.main()
