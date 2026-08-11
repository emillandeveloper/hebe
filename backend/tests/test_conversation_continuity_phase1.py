from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.continuity.legacy_adapter import LegacyPendingAdapter
from app.continuity.models import (
    ConversationContext, ConversationStatus, ConversationalAct, ExpectedReply,
    ExpectedReplyType, OpenThreadStatus,
)
from app.continuity.repository import ConversationRepository, OpenThreadRepository
from app.continuity.service import ConversationContinuityService
from app.replay.migrations import MigrationRunner, conversation_continuity_migrations
from app.replay.cognitive import CognitiveReplayRunner
from app.replay.report import STATUS_VERIFIED


PHASE1_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "cognitive_replay_phase1"


class ContinuityFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "continuity.sqlite3"
        self.connect = lambda: sqlite3.connect(self.db)
        MigrationRunner(self.connect).migrate(conversation_continuity_migrations())
        self.now = 1000.0
        self.conversations = ConversationRepository(self.connect)
        self.threads = OpenThreadRepository(self.connect)
        self.service = ConversationContinuityService(
            self.conversations, self.threads, now_fn=lambda: self.now,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def open(self, reply_type=ExpectedReplyType.YES_NO, **kwargs):
        return self.service.open_conversation(
            context_kind=kwargs.pop("context_kind", ConversationContext.OWNER_LIVE_CONTROL),
            context_id=kwargs.pop("context_id", "stream-1"), topic="test",
            origin_event_id="question", expected_reply=ExpectedReply(
                type=reply_type, candidate_refs=tuple(kwargs.pop("candidate_refs", ("ivanxi_kun",))),
                expires_at=self.now + kwargs.pop("ttl", 60),
                semantic_constraints=kwargs.pop("semantic_constraints", {"allow_deictic": True}),
            ), domain_payload=kwargs.pop("domain_payload", {}), **kwargs,
        )


class ExpectedReplyTests(unittest.TestCase):
    def test_yes_no_variants_are_bounded(self):
        reply = ExpectedReply(ExpectedReplyType.YES_NO)
        for text in ("sí", "sip", "correcto", "ese", "dale", "hazlo"):
            self.assertEqual(reply.classify(text)[0], ConversationalAct.AFFIRM)
        for text in ("no", "nop", "ese no", "el otro"):
            self.assertEqual(reply.classify(text)[0], ConversationalAct.DENY)
        self.assertEqual(reply.classify("sí, es que este boss es horrible")[0], ConversationalAct.UNKNOWN)

    def test_entity_value_correction_and_free_response(self):
        entity = ExpectedReply(ExpectedReplyType.ENTITY_SELECTION, candidate_refs=("ivan", "ivanxi"))
        self.assertEqual(entity.classify("el segundo")[1]["candidate"], "ivanxi")
        self.assertEqual(ExpectedReply(ExpectedReplyType.VALUE).classify("12,5")[1]["value"], "12.5")
        self.assertEqual(ExpectedReply(ExpectedReplyType.CORRECTION).classify("no, el tercero")[0], ConversationalAct.CORRECT)
        self.assertEqual(ExpectedReply(ExpectedReplyType.FREE_RESPONSE).classify("porque me gusta")[0], ConversationalAct.FREE_RESPONSE)


class ConversationServiceTests(ContinuityFixture):
    def test_owner_reply_consumes_atomically_and_resolves_thread(self):
        opened = self.open()
        result = self.service.resolve_input(
            context_kind="owner_live_control", context_id="stream-1", source="owner_stt",
            participant="leo", authority="owner", text="sí", event_id="reply-1",
        )
        self.assertTrue(result.consumed)
        self.assertEqual(result.reply_act, ConversationalAct.AFFIRM)
        stored = self.conversations.get(opened.id)
        self.assertEqual(stored.status, ConversationStatus.CLOSED)
        self.assertEqual(stored.consumed_event_ids, ("reply-1",))
        self.assertEqual(self.threads.list_open(), [])

    def test_authority_matrix_rejects_ambient_viewer_mod_system_and_self(self):
        for source, participant, authority in (
            ("ambient_stt", "ambient", "ambient"),
            ("twitch_chat", "viewer", "viewer"),
            ("twitch_chat", "moderator", "mod"),
            ("internal_event", "system", "system"),
            ("twitch_chat", "hebe", "self"),
        ):
            with self.subTest(source=source, authority=authority):
                opened = self.open()
                result = self.service.resolve_input(
                    context_kind="owner_live_control", context_id="stream-1", source=source,
                    participant=participant, authority=authority, text="sí", event_id=f"{source}-{authority}",
                )
                self.assertFalse(result.consumed)
                self.assertEqual(self.conversations.get(opened.id).status, ConversationStatus.WAITING_ON_LEO)

    def test_wrong_context_expiry_interruption_and_restart_are_explicit(self):
        ui = self.open(context_kind=ConversationContext.PRIVATE_UI, context_id="leo-ui")
        wrong = self.service.resolve_input(
            context_kind="owner_live_control", context_id="stream-1", source="owner_stt",
            participant="leo", authority="owner", text="sí", event_id="wrong",
        )
        self.assertFalse(wrong.consumed)
        self.assertEqual(self.conversations.get(ui.id).status, ConversationStatus.WAITING_ON_LEO)
        live = self.open(ttl=1)
        self.now += 2
        expired = self.service.resolve_input(
            context_kind="owner_live_control", context_id="stream-1", source="owner_stt",
            participant="leo", authority="owner", text="sí", event_id="late",
        )
        self.assertEqual(expired.reason, "expired")
        self.assertEqual(self.conversations.get(live.id).status, ConversationStatus.EXPIRED)
        active = self.open()
        interrupted = self.service.resolve_input(
            context_kind="owner_live_control", context_id="stream-1", source="owner_stt",
            participant="leo", authority="owner", text="Hebe abre OBS", event_id="new", wake=True,
        )
        self.assertEqual(interrupted.decision, "interrupt")
        self.assertEqual(self.conversations.get(active.id).status, ConversationStatus.INTERRUPTED)
        stale = self.open()
        self.assertEqual(self.conversations.interrupt_active_on_start(), 2)
        self.assertEqual(self.conversations.get(stale.id).closure_reason, "runtime_restart")

    def test_shadow_resolution_does_not_mutate(self):
        opened = self.open()
        result = self.service.resolve_input(
            context_kind="owner_live_control", context_id="stream-1", source="owner_stt",
            participant="leo", authority="owner", text="sí", event_id="shadow", consume=False,
        )
        self.assertTrue(result.consumed)
        self.assertEqual(result.decision, "compatible_reply_shadow")
        self.assertEqual(self.conversations.get(opened.id).status, ConversationStatus.WAITING_ON_LEO)

    def test_legacy_adapter_projects_one_candidate_as_yes_no(self):
        state = SimpleNamespace(pending_clarification=None)
        adapter = LegacyPendingAdapter(self.service, state)
        pending = {
            "id": "pending-1", "kind": "promotion_target_clarification",
            "expected_reply_type": "twitch_username_or_viewer_alias",
            "candidates": ["ivanxi_kun"], "expires_at": self.now + 60,
        }
        conversation = adapter.project_legacy_pending(
            pending, context_kind=ConversationContext.OWNER_LIVE_CONTROL,
            context_id="stream-1", event_id="question",
        )
        self.assertEqual(conversation.expected_reply.type, ExpectedReplyType.YES_NO)
        self.assertEqual(pending["conversation_id"], conversation.id)


class Phase1ReplayIntegrationTests(unittest.TestCase):
    def test_required_phase1_scenarios_are_deterministic_and_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in sorted(PHASE1_FIXTURES.glob("*.json")):
                with self.subTest(scenario=path.stem):
                    result = CognitiveReplayRunner(workspace_root=tmp, retain_workspace=False).run(path)
                    self.assertEqual(result.status, STATUS_VERIFIED, result.failures)

    def test_ivanxi_uses_receipt_truth_and_exactly_one_external_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = CognitiveReplayRunner(workspace_root=tmp, retain_workspace=False).run(
                PHASE1_FIXTURES / "a_ivanxi_wake_free.json"
            )
        attempts = result.final_state["actions"]["attempts"]
        self.assertEqual(sum(item["operation"] == "twitch.shoutout" for item in attempts), 1)
        self.assertTrue(any(item["execution_status"] == "sent" for item in result.final_state["receipts"]))
        self.assertEqual(result.checkpoint_states["yes"]["conversation"]["latest"]["status"], "CLOSED")


if __name__ == "__main__":
    unittest.main()
