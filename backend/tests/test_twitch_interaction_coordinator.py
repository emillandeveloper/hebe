import threading
import unittest
from types import SimpleNamespace

from app.cognitive.twitch_interaction_coordinator import (
    TrollEngagementBudget,
    TwitchInteractionCoordinator,
)


def twitch_event(event_id, text, *, viewer="viewer", direct=False, idle=False):
    return SimpleNamespace(
        event_type="twitch_idle_prompt" if idle else "twitch_chat_react",
        payload={
            "event_id": event_id,
            "user_login": viewer,
            "message_text": text,
            "direct_address_to_hebe": direct,
        },
    )


class TwitchInteractionCoordinatorTests(unittest.TestCase):
    def test_direct_question_preempts_active_and_waiting_banter(self):
        coordinator = TwitchInteractionCoordinator()
        started = threading.Event()
        release = threading.Event()
        processed = []

        def processor(event):
            processed.append(event.payload["event_id"])
            if event.payload["event_id"] == "banter-active":
                started.set()
                release.wait(2)
            else:
                coordinator.record_candidate(event.payload["event_id"], "respuesta")
                coordinator.record_emission(
                    event.payload["event_id"],
                    {"emitted": True, "route": "twitch_text_reply", "event_id": event.payload["event_id"]},
                )

        worker = threading.Thread(
            target=lambda: coordinator.submit(twitch_event("banter-active", "ambiente", idle=True), processor)
        )
        worker.start()
        self.assertTrue(started.wait(1))
        waiting = coordinator.submit(twitch_event("banter-waiting", "otro ambiente", idle=True), processor)
        direct = coordinator.submit(twitch_event("question", "Hebe, como estas?", direct=True), processor)
        release.set()
        worker.join(2)

        self.assertEqual(waiting.status, "cancelled")
        self.assertEqual(coordinator.jobs["banter-active"].status, "cancelled")
        self.assertEqual(direct.status, "emitted")
        self.assertEqual(processed, ["banter-active", "question"])
        self.assertEqual(coordinator.max_generation_in_flight, 1)

    def test_generated_but_not_emitted_is_failed_and_retried(self):
        coordinator = TwitchInteractionCoordinator()

        first = coordinator.submit(
            twitch_event("q1", "Hebe, que opinas?", viewer="ana", direct=True),
            lambda event: coordinator.record_candidate(event.payload["event_id"], "Borrador"),
        )
        calls = []

        def retry_processor(event):
            calls.append(event.payload["event_id"])
            coordinator.record_candidate(event.payload["event_id"], "Final")
            coordinator.record_emission(
                event.payload["event_id"],
                {"emitted": True, "route": "twitch_text_reply", "event_id": "response-2"},
            )

        retry = coordinator.submit(
            twitch_event("q2", "Ebe, que opinas?", viewer="ana", direct=True), retry_processor
        )

        self.assertEqual(first.status, "failed")
        self.assertEqual(first.failure_reason, "generated_but_not_emitted")
        self.assertEqual(calls, ["q2"])
        self.assertEqual(retry.status, "emitted")

    def test_style_guard_failure_is_failed_not_answered(self):
        coordinator = TwitchInteractionCoordinator()

        def processor(event):
            event_id = event.payload["event_id"]
            coordinator.record_candidate(event_id, "Ismael: candidate")
            coordinator.record_emission(
                event_id,
                {"emitted": False, "route": "suppress"},
                reason="stream_response_quality_guard:hebe_voice_report_prefix",
            )

        job = coordinator.submit(
            twitch_event("q1", "Hebe, quien gana?", viewer="ismael", direct=True), processor
        )

        self.assertEqual(job.status, "failed")
        self.assertEqual(job.response_outcome, "failed")

    def test_answered_semantic_repeat_is_deduped(self):
        coordinator = TwitchInteractionCoordinator()

        def processor(event):
            event_id = event.payload["event_id"]
            coordinator.record_candidate(event_id, "Bien")
            coordinator.record_emission(
                event_id, {"emitted": True, "route": "twitch_text_reply", "event_id": "final-1"}
            )

        coordinator.submit(twitch_event("q1", "Hebe, como estas?", viewer="ana", direct=True), processor)
        repeat = coordinator.submit(
            twitch_event("q2", "Ebe, como estas?", viewer="ana", direct=True),
            lambda event: self.fail("an answered repeat must not regenerate"),
        )

        self.assertEqual(repeat.status, "observed")
        self.assertEqual(repeat.response_outcome, "already_answered_repeat")

    def test_policy_blocked_repeat_remains_blocked_without_generation(self):
        coordinator = TwitchInteractionCoordinator()
        first = coordinator.submit(
            twitch_event("q1", "Hebe, haz algo prohibido?", viewer="ana", direct=True),
            lambda event: coordinator.record_policy_suppression(event.payload["event_id"], "safety_policy"),
        )
        repeat = coordinator.submit(
            twitch_event("q2", "Ebe, haz algo prohibido?", viewer="ana", direct=True),
            lambda event: self.fail("policy-blocked repeat must not regenerate"),
        )

        self.assertEqual(first.response_outcome, "policy_blocked")
        self.assertEqual(repeat.status, "suppressed")
        self.assertEqual(repeat.response_outcome, "policy_blocked")

    def test_suppressed_unanswered_repeat_is_not_deduped(self):
        coordinator = TwitchInteractionCoordinator()

        def suppress(event):
            coordinator.record_candidate(event.payload["event_id"], "candidate")
            coordinator.record_emission(
                event.payload["event_id"],
                {"emitted": False, "route": "suppress", "reason": "flood_soft_limit"},
            )

        first = coordinator.submit(
            twitch_event("q1", "Hebe, que piensas?", viewer="ana", direct=True), suppress
        )
        processed = []
        retry = coordinator.submit(
            twitch_event("q2", "Ebe, que piensas?", viewer="ana", direct=True),
            lambda event: processed.append(event.payload["event_id"]),
        )

        self.assertEqual(first.status, "suppressed")
        self.assertEqual(processed, ["q2"])
        self.assertNotEqual(retry.response_outcome, "already_answered_repeat")

    def test_job_uses_immutable_context_snapshot(self):
        coordinator = TwitchInteractionCoordinator()
        event = twitch_event("q1", "Hebe, que ves?", viewer="ana", direct=True)
        observed = []

        def processor(snapshot):
            observed.append(snapshot.payload["message_text"])
            event.payload["message_text"] = "mutated externally"
            coordinator.record_policy_suppression(snapshot.payload["event_id"], "safety_policy")

        job = coordinator.submit(event, processor)

        self.assertEqual(observed, ["Hebe, que ves?"])
        self.assertEqual(job.context_snapshot["raw_text"], "Hebe, que ves?")
        self.assertEqual(
            coordinator.direct_outcomes[("ana", job.semantic_key)].final_outcome,
            "policy_blocked",
        )

    def test_owner_operation_cancels_low_value_before_emission(self):
        coordinator = TwitchInteractionCoordinator()
        started = threading.Event()
        release = threading.Event()

        def banter_processor(_event):
            started.set()
            release.wait(2)

        worker = threading.Thread(
            target=lambda: coordinator.submit(twitch_event("banter", "ambiente", idle=True), banter_processor)
        )
        worker.start()
        self.assertTrue(started.wait(1))
        result = coordinator.submit_owner_stream_operation(
            event_id="promo", text="haz promo", processor=lambda: "executed"
        )
        self.assertFalse(coordinator.allows_final_emission("banter"))
        release.set()
        worker.join(2)

        self.assertEqual(result, "executed")
        self.assertEqual(coordinator.jobs["banter"].status, "cancelled")


class TrollEngagementBudgetTests(unittest.TestCase):
    def test_bait_topic_allows_once_closes_once_then_observes(self):
        budget = TrollEngagementBudget()

        self.assertEqual(budget.evaluate(viewer="ana", text="salami")["action"], "allow")
        budget.record_engagement(viewer="ana", text="salami")
        self.assertEqual(budget.evaluate(viewer="ana", text="chorizo")["action"], "close")
        budget.record_engagement(viewer="ana", text="chorizo")
        self.assertEqual(budget.evaluate(viewer="ana", text="mandanga")["action"], "observe")

    def test_unrelated_direct_question_is_not_charged_to_bait_topic(self):
        budget = TrollEngagementBudget()
        budget.record_engagement(viewer="ana", text="porros")
        budget.record_engagement(viewer="ana", text="canuto")

        result = budget.evaluate(viewer="ana", text="Hebe, como estas?")

        self.assertEqual(result["action"], "allow")
        self.assertEqual(result["reason"], "not_bait_topic")


if __name__ == "__main__":
    unittest.main()
