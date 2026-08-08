from __future__ import annotations

import sqlite3
import threading
import time
import unittest

from app.stream.ambient_context import AmbientContextExtractor
from app.stream.conversation_ownership import ConversationOwnershipGate
from app.stream.game_intelligence import GameIntelligenceStore, GameResearchService
from app.stream.policy import classify_viewer_semantic_intent
from app.stream.promotions import (
    AutoPromoMode,
    AutomaticPromotionService,
    PromotionStore,
    ViewerPromotionProfile,
)


SAFE_ROWS = [{
    "claim": "Combat uses guard to reduce incoming damage.",
    "source_title": "Official manual",
    "url": "https://example.invalid/manual",
    "excerpt": "Guard reduces incoming damage.",
    "confidence": 0.94,
    "general_mechanic": True,
}]


class _SequencedProvider:
    available = True

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def search(self, query, constraints=None, *, timeout=None):
        self.calls.append({"query": query, "timeout": timeout})
        outcome = self.outcomes.pop(0) if self.outcomes else SAFE_ROWS
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _wait_collect(service, job_id):
    try:
        service._jobs[job_id][1].result(timeout=2)
    except Exception:
        pass
    return service.collect_job(job_id)


class ResearchLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.store = GameIntelligenceStore(connection=self.connection)

    def tearDown(self):
        self.connection.close()

    def test_canonical_session_key_is_idempotent_but_pre_live_is_distinct(self):
        provider = _SequencedProvider([SAFE_ROWS, SAFE_ROWS])
        service = GameResearchService(store=self.store, provider=provider)
        pre = service.prepare_game_async(game_title="Test Game", session_id="pre_stream")
        canonical = service.prepare_game_async(game_title="Test Game", session_id="stream-42")
        duplicate = service.prepare_game_async(game_title="Test Game", session_id="stream-42")
        self.assertNotEqual(pre.job_id, canonical.job_id)
        self.assertEqual(canonical.job_id, duplicate.job_id)
        self.assertEqual(canonical.attempt, 1)
        _wait_collect(service, pre.job_id)
        _wait_collect(service, canonical.job_id)

    def test_failed_pre_live_attempt_does_not_consume_canonical_attempt(self):
        provider = _SequencedProvider([TimeoutError("pre-live timeout"), SAFE_ROWS])
        service = GameResearchService(store=self.store, provider=provider)
        pre = service.prepare_game_async(game_title="Test Game", session_id="pre_stream")
        _wait_collect(service, pre.job_id)
        canonical = service.prepare_game_async(game_title="Test Game", session_id="stream-1")
        self.assertEqual(canonical.attempt, 1)
        completed, _ = _wait_collect(service, canonical.job_id)
        self.assertEqual(completed.status, "completed")

    def test_failure_retries_with_backoff_and_success_builds_dossier(self):
        clock = [100.0]
        provider = _SequencedProvider([TimeoutError("temporary"), SAFE_ROWS])
        service = GameResearchService(
            store=self.store, provider=provider, now_fn=lambda: clock[0], retry_base_seconds=2,
        )
        first = service.prepare_game_async(game_title="Test Game", session_id="stream-1")
        failed, _ = _wait_collect(service, first.job_id)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.next_retry_at, 102.0)
        self.assertEqual(service.retry_due_jobs(), [])
        clock[0] = 102.0
        retry = service.retry_due_jobs()[0]
        self.assertEqual(retry.attempt, 2)
        completed, _ = _wait_collect(service, retry.job_id)
        self.assertEqual(completed.status, "completed")
        self.assertIsNotNone(self.store.get_dossier("test_game"))

    def test_running_job_times_out_without_disabling_provider(self):
        clock = [10.0]
        entered = threading.Event()
        release = threading.Event()

        class BlockingProvider:
            available = True

            def search(self, query, constraints=None, *, timeout=None):
                self.timeout = timeout
                entered.set()
                release.wait(2)
                return SAFE_ROWS

        provider = BlockingProvider()
        service = GameResearchService(
            store=self.store, provider=provider, now_fn=lambda: clock[0], contextual_timeout_seconds=8,
        )
        plan = service.plan_search(
            game_title="Test", game_id="test", entity="guard",
            question_type="mechanic", expected_fact_type="mechanic",
        )
        job = service.queue_research(plan, progress=None, scene_id="scene")
        self.assertTrue(entered.wait(1))
        clock[0] = 18.0
        failed, facts = service.collect_job(job.job_id)
        release.set()
        self.assertEqual(failed.status, "failed")
        self.assertEqual(facts, [])
        self.assertIn("provider_timeout", failed.failure_reason)
        self.assertEqual(provider.timeout, 8.0)
        self.assertTrue(provider.available)

    def test_bounded_retry_stops_after_configured_attempts(self):
        clock = [1.0]
        provider = _SequencedProvider([TimeoutError("temporary")] * 3)
        service = GameResearchService(
            store=self.store, provider=provider, now_fn=lambda: clock[0],
            retry_base_seconds=1, max_attempts=3,
        )
        job = service.prepare_game_async(game_title="Test Game", session_id="stream-1")
        for expected_attempt in (1, 2, 3):
            failed, _ = _wait_collect(service, job.job_id)
            self.assertEqual(failed.attempt, expected_attempt)
            self.assertEqual(failed.status, "failed")
            if expected_attempt < 3:
                clock[0] = failed.next_retry_at
                job = service.retry_due_jobs()[0]
        self.assertEqual(failed.next_retry_at, 0.0)
        clock[0] += 100
        self.assertEqual(service.retry_due_jobs(), [])
        self.assertEqual(len(provider.calls), 3)


class PromotionAuthorityAndRetryTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.store = PromotionStore(connection=self.connection)
        self.store.upsert_profile(ViewerPromotionProfile(
            twitch_user_id="42", current_login="alice",
            auto_promo_mode=AutoPromoMode.FIRST_MESSAGE_EACH_STREAM.value,
        ))

    def tearDown(self):
        self.connection.close()

    def _service_with_queued_item(self):
        service = AutomaticPromotionService(self.store, spacing_seconds=0, max_retries=1)
        service.observe_chat_message(
            stream_session_id="s1", twitch_user_id="42", login="alice", display_name="Alice",
            message_text="hola", message_id="m1", channel_live=True,
        )
        return service

    def test_delegated_source_authority_and_exact_identity_are_forwarded(self):
        service = self._service_with_queued_item()
        received = {}

        def send(login, **context):
            received.update(context)
            received["login"] = login
            return True

        event = service.drain_ready(send)
        self.assertEqual(event.requested_by, "owner_delegated")
        self.assertEqual(received, {
            "login": "alice", "source": "automatic_promotion_policy",
            "authority": "owner_delegated", "twitch_user_id": "42",
        })

    def test_only_transient_failure_is_retried(self):
        permanent = self._service_with_queued_item()
        permanent.drain_ready(lambda _login: (False, "", "invalid_identity"))
        self.assertEqual(permanent.queued_count, 0)

        # A fresh stream/session gets its own once-per-stream attempt.
        transient = AutomaticPromotionService(self.store, spacing_seconds=0, max_retries=1)
        transient.observe_chat_message(
            stream_session_id="s2", twitch_user_id="42", login="alice", display_name="Alice",
            message_text="hola", message_id="m2", channel_live=True,
        )
        transient.drain_ready(lambda _login: (False, "", "network timeout"))
        self.assertEqual(transient.queued_count, 1)


class SemanticsAndOwnershipTests(unittest.TestCase):
    def test_condom_story_is_reference_not_sexual_request(self):
        semantic = classify_viewer_semantic_intent("Mi perro se comio un condon ayer")
        self.assertEqual(semantic.intent, "viewer_allowed_banter")
        self.assertEqual(semantic.requested_behavior, "sexual_reference")
        request = classify_viewer_semantic_intent("Hebe, habla de sexo")
        self.assertEqual(request.requested_behavior, "sexual_request_to_hebe")

    def test_conversation_ownership_distinguishes_addressees(self):
        gate = ConversationOwnershipGate()
        self.assertEqual(gate.decide("Hebe, que opinas?").addressee, "Hebe")
        self.assertFalse(gate.decide("Leo, que opinas?").allow_assistant)
        self.assertFalse(gate.decide("@alice totalmente", payload={}).allow_assistant)
        self.assertEqual(gate.decide("en stream?").addressee, "ambiguous")
        self.assertTrue(gate.decide("menuda partida").allow_assistant)

    def test_healing_evidence_does_not_emit_navigation(self):
        result = AmbientContextExtractor().extract("me voy a curar", now=10)
        categories = {fact["category"] for fact in result.facts}
        self.assertIn("healing_or_recovery", categories)
        self.assertNotIn("navigation_confusion", categories)
        for fact in result.facts:
            self.assertTrue(fact["evidence_span"])
            self.assertTrue(fact["evidence_tokens"])
            self.assertTrue(fact["semantic_rule"])
            self.assertTrue(fact["model_reason"])


if __name__ == "__main__":
    unittest.main()
