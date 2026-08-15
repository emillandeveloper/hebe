import unittest
from types import SimpleNamespace

from app.cognitive.capabilities import CapabilityMatcher, CapabilityRegistry, GoalExtractor
from app.cognitive.deliberation_service import DeliberationService


def make_context(text: str, state_snapshot: dict | None = None):
    return SimpleNamespace(
        input_text=text,
        internal_event=None,
        state_snapshot=state_snapshot or {},
        resolved_entities=[],
        message_type="direct_question",
    )


class GoalCapabilityPlanningTests(unittest.TestCase):
    def setUp(self):
        self.registry = CapabilityRegistry.default()
        self.extractor = GoalExtractor()
        self.matcher = CapabilityMatcher(self.registry)

    def test_chat_activity_request_maps_to_partial_capability_without_execution(self):
        goal = self.extractor.extract(make_context("quien hablo mas en el chat de Twitch"))
        match = self.matcher.match(goal, current_mode="private")

        self.assertEqual(goal.goal_type, "analyze_chat_activity")
        self.assertEqual(goal.slots["time_range"], "all_recorded_history")
        self.assertIn(
            "stream.chat_activity_report",
            {item["capability_id"] for item in match.rejected_capabilities},
        )
        self.assertEqual(match.selected_capabilities, [])

    def test_game_strategy_request_keeps_spoiler_policy_and_rejects_partial(self):
        goal = self.extractor.extract(make_context("quiero romper Kingdom Hearts Chain of Memories"))
        match = self.matcher.match(goal, current_mode="private")

        self.assertEqual(goal.goal_type, "research_game_strategy")
        self.assertEqual(goal.slots["strategy_mode"], "break_the_game")
        self.assertEqual(goal.slots["game"], "kingdom hearts chain of memories")
        self.assertEqual(goal.spoiler_sensitivity, "mechanics_ok_story_avoid")
        self.assertIn(
            "game.strategy_research",
            {item["capability_id"] for item in match.rejected_capabilities},
        )
        self.assertEqual(match.selected_capabilities, [])

    def test_open_application_plan_records_capability_metadata(self):
        service = DeliberationService(intent_model=None, reasoning_model=None)
        plan = service.deliberate(make_context("Hebe abre OBS")).plan

        self.assertEqual(plan.goal["goal_type"], "control_pc")
        self.assertIn("pc.open_application", plan.selected_capabilities)
        self.assertEqual(plan.steps[0].type, "action")
        self.assertEqual(plan.steps[0].capability_id, "pc.open_application")
        self.assertEqual(plan.steps[0].data["params"]["requested_target"], "OBS")

    def test_catalogue_backlog_question_returns_query_plan(self):
        service = DeliberationService(intent_model=None, reasoning_model=None)
        plan = service.deliberate(
            make_context("what capabilities are planned but not implemented?")
        ).plan

        self.assertEqual(plan.goal["goal_type"], "analyze_data")
        self.assertIn("hebe.capability_backlog_query", plan.selected_capabilities)
        self.assertEqual(plan.steps[0].data["mode"], "capability_catalogue_query")
        self.assertEqual(plan.steps[0].data["query_type"], "planned_not_implemented")
        self.assertGreater(plan.steps[0].data["payload"]["count"], 0)

    def test_next_recommended_todo_question_returns_stream_chat_report(self):
        service = DeliberationService(intent_model=None, reasoning_model=None)
        plan = service.deliberate(make_context("what is the next recommended TODO?")).plan
        payload = plan.steps[0].data["payload"]

        self.assertEqual(plan.steps[0].data["query_type"], "next_todo")
        self.assertEqual(payload["next_recommended_todo"]["id"], "stream.chat_activity_report")

    def test_spanish_capability_backlog_questions_use_catalogue_capability(self):
        service = DeliberationService(intent_model=None, reasoning_model=None)
        examples = {
            "Hebe, cual es el siguiente TODO?": "next_todo",
            "Hebe, que capabilities estan planned?": "planned_not_implemented",
            "Hebe, que capabilities estan partial?": "partial_needs_completion",
            "Hebe, que falta por implementar?": "planned_not_implemented",
            "Hebe, cual es el siguiente capability recomendado?": "next_todo",
        }

        for text, query_type in examples.items():
            with self.subTest(text=text):
                plan = service.deliberate(make_context(text)).plan
                self.assertIn("hebe.capability_backlog_query", plan.selected_capabilities)
                self.assertEqual(plan.steps[0].data["query_type"], query_type)

    def test_catalogue_unavailable_returns_explicit_reply_payload(self):
        service = DeliberationService(intent_model=None, reasoning_model=None)
        service.capability_registry = None
        service.capability_matcher = None
        service.capability_catalogue_error = "missing catalogue"

        plan = service.deliberate(make_context("Hebe, cual es el siguiente TODO?")).plan
        payload = plan.steps[0].data["payload"]

        self.assertTrue(payload["catalogue_unavailable"])
        self.assertEqual(payload["error"], "missing catalogue")


if __name__ == "__main__":
    unittest.main()
