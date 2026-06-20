from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.cognitive.cognitive_router import CognitiveRouter
from app.cognitive.deliberation_service import DeliberationService


def context(text: str, pending: dict | None = None):
    return SimpleNamespace(
        input_text=text,
        internal_event=None,
        state_snapshot={"pending_clarification": pending} if pending else {},
        resolved_entities=[],
        message_type="direct_question",
        source="ui",
        authority="owner",
        addressed_to_hebe=True,
        message_id="test-message",
        cognitive_decision=None,
    )


def active_pending(**overrides):
    value = {
        "id": "pending-test",
        "kind": "appointment_datetime",
        "authority": "owner",
        "expires_at": time.time() + 300,
        "draft": {"title": "Consulta", "source_text": "appointment source"},
    }
    value.update(overrides)
    return value


class CognitiveRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = CognitiveRouter()
        self.service = DeliberationService(intent_model=None, reasoning_model=None)

    def route(self, value):
        value.cognitive_decision = self.router.route(value)
        return value.cognitive_decision

    def test_current_time_is_high_priority_without_pending(self):
        value = context("Hebe, Â¿quÃ© hora es?")
        decision = self.route(value)
        plan = self.service.deliberate(value).plan
        self.assertEqual(decision.intent, "current_time_query")
        self.assertFalse(decision.uses_pending_task)
        self.assertIn("appointment.create", decision.blocked_capability_ids)
        self.assertFalse(decision.allows_capability("pending.cancel"))
        self.assertFalse(decision.allows_capability("audio.tts_control"))
        self.assertEqual(plan.steps[0].data["mode"], "time_answer")

    def test_current_time_overrides_active_pending(self):
        pending = active_pending()
        value = context("Hebe, Â¿quÃ© hora es?", pending)
        decision = self.route(value)
        plan = self.service.deliberate(value).plan
        self.assertTrue(decision.is_new_request)
        self.assertFalse(decision.uses_pending_task)
        self.assertEqual(decision.pending_reason, "new_request_override")
        self.assertEqual(plan.steps[0].data["mode"], "time_answer")
        self.assertIs(value.state_snapshot["pending_clarification"], pending)

    def test_semantic_datetime_answer_may_resolve_pending(self):
        value = context("A las cinco", active_pending())
        decision = self.route(value)
        self.assertEqual(decision.intent, "pending_datetime_answer")
        self.assertTrue(decision.uses_pending_task)
        self.assertTrue(decision.pending_resolution_allowed)

    def test_owner_personal_state_blocks_creators(self):
        value = context("Hebe, tengo hambre")
        decision = self.route(value)
        plan = self.service.deliberate(value).plan
        self.assertEqual(decision.intent, "owner_personal_state")
        self.assertEqual(decision.personal_state, "hunger")
        self.assertEqual(plan.steps[0].data["mode"], "companion_reaction")
        self.assertIn("scheduler.create", decision.blocked_capability_ids)
        self.assertFalse(decision.allows_capability("pending.cancel"))
        self.assertFalse(decision.allows_capability("audio.tts_control"))
        self.assertFalse(decision.allows_capability("reminder.create"))

    def test_generic_time_term_does_not_mean_appointment(self):
        decision = self.route(context("Hebe, la hora actual"))
        self.assertEqual(decision.intent, "current_time_query")
        self.assertNotEqual(decision.intent, "appointment_create_request")

    def test_explicit_appointment_without_datetime_clarifies(self):
        value = context("Hebe, apÃºntame una cita con el mÃ©dico")
        decision = self.route(value)
        plan = self.service.deliberate(value).plan
        self.assertEqual(decision.intent, "appointment_create_request")
        self.assertEqual(plan.steps[-1].data["mode"], "clarify_appointment_datetime")

    def test_explicit_appointment_with_datetime_resolves(self):
        value = context("Hebe agenda una cita manana a las 17:00")
        decision = self.route(value)
        plan = self.service.deliberate(value).plan
        self.assertEqual(decision.intent, "appointment_create_request")
        self.assertEqual([step.type for step in plan.steps], ["memory", "reminder", "reply"])
        self.assertEqual(plan.steps[-1].data["mode"], "confirm_appointment")

    def test_current_date_has_its_own_route(self):
        value = context("Hebe, que fecha es hoy")
        decision = self.route(value)
        plan = self.service.deliberate(value).plan
        self.assertEqual(decision.intent, "current_date_query")
        self.assertEqual(plan.steps[0].data["mode"], "date_answer")

    def test_known_structured_routes_never_call_fallback_chat_planner(self):
        samples = (
            "Hebe, dime la hora actual",
            "Hebe, inicia Discord",
            "Hebe, tengo hambre",
        )
        with patch.object(
            self.service,
            "_plan_with_llm",
            side_effect=AssertionError("structured intent reached fallback chat"),
        ):
            for text in samples:
                with self.subTest(text=text):
                    value = context(text)
                    self.route(value)
                    plan = self.service.deliberate(value).plan
                    self.assertNotEqual(plan.steps[0].data.get("mode"), "chat")

    def test_open_app_is_not_stolen_by_pending(self):
        value = context("Hebe, abre Discord", active_pending())
        decision = self.route(value)
        plan = self.service.deliberate(value).plan
        self.assertEqual(decision.intent, "command_open_app")
        self.assertFalse(decision.uses_pending_task)
        self.assertEqual(plan.steps[0].type, "action")

    def test_expired_pending_is_never_consumed(self):
        value = context("A las cinco", active_pending(expires_at=time.time() - 1))
        decision = self.route(value)
        plan = self.service.deliberate(value).plan
        self.assertFalse(decision.uses_pending_task)
        self.assertEqual(decision.pending_reason, "expired")
        self.assertEqual(plan.steps[0].data["mode"], "chat")


if __name__ == "__main__":
    unittest.main()
