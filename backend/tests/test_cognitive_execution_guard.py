from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.cognitive.cognitive_router import CognitiveRouter
from app.cognitive.models import Plan, PlanStep
from app.cognitive.plan_executor import PlanExecutor


class CognitiveExecutionGuardTests(unittest.TestCase):
    def setUp(self):
        self.memory = Mock()
        self.runtime = Mock()
        self.executor = PlanExecutor(self.memory, self.runtime)

    def test_action_without_decision_is_blocked(self):
        result = self.executor.execute(Plan(steps=[PlanStep(
            type="action", data={"name": "open_application", "params": {"app_name": "Discord"}}
        )]))
        self.assertFalse(result.ok)
        self.assertEqual(result.results[0].error, "missing_cognitive_decision")
        self.runtime.execute.assert_not_called()

    def test_reminder_without_capability_grant_is_blocked(self):
        decision = self._decision("direct_question", ["hebe.chat_reply"], ["reply"])
        plan = Plan(
            steps=[PlanStep(type="reminder", data={"title": "x", "due_at": "2099-01-01T00:00:00Z"})],
            metadata={"cognitive_decision": decision},
        )
        result = self.executor.execute(plan)
        self.assertEqual(result.results[0].error, "step_type_not_authorized")
        self.memory.create_reminder.assert_not_called()

    def test_twitch_reply_requires_live_stream(self):
        decision = self._decision("twitch_internal_event", ["twitch.reply"], ["reply"], authority="system")
        decision["action_permission_summary"] = {"stream_live": False}
        plan = Plan(
            steps=[PlanStep(type="reply", capability_id="twitch.reply", data={"mode": "twitch_raid"})],
            metadata={"cognitive_decision": decision},
        )
        result = self.executor.execute(plan)
        self.assertEqual(result.results[0].error, "stream_not_live")

    def test_offline_internal_twitch_event_stops_at_router(self):
        context = SimpleNamespace(
            input_text="", internal_event=SimpleNamespace(event_type="twitch_raid"),
            state_snapshot={}, source="twitch_system", authority="system",
            addressed_to_hebe=False, firewall_decision="allow", stream_is_live=False,
        )
        decision = CognitiveRouter().route(context)
        self.assertTrue(decision.should_stop_pipeline)
        self.assertFalse(decision.should_reply)
        self.assertNotIn("twitch.reply", decision.allowed_capabilities)

    def test_live_internal_twitch_event_carries_reply_grant(self):
        context = SimpleNamespace(
            input_text="", internal_event=SimpleNamespace(event_type="twitch_raid"),
            state_snapshot={}, source="twitch_system", authority="system",
            addressed_to_hebe=False, firewall_decision="allow", stream_is_live=True,
        )
        decision = CognitiveRouter().route(context)
        self.assertEqual(decision.intent, "twitch_internal_event")
        self.assertFalse(decision.should_stop_pipeline)
        self.assertTrue(decision.allows_capability("twitch.reply"))

    @staticmethod
    def _decision(intent, capabilities, step_types, *, authority="owner"):
        return {
            "intent": intent, "source": "twitch_system" if authority == "system" else "ui", "authority": authority,
            "allowed_capabilities": capabilities, "blocked_capabilities": [],
            "allowed_step_types": step_types, "blocked_step_types": [],
            "should_stop_pipeline": False, "action_permission_summary": {},
        }


if __name__ == "__main__":
    unittest.main()
