import unittest

from app.cognitive.capabilities import CapabilityRegistry


REQUIRED_CAPABILITY_FIELDS = {
    "id",
    "category",
    "name",
    "description",
    "status",
    "enabled",
    "risk_level",
    "requires_confirmation",
    "available_in_modes",
    "input_schema",
    "output_schema",
    "dependencies",
    "implemented_by",
    "tests",
    "notes",
    "examples_semantic",
    "output_policy",
    "memory_policy",
    "spoiler_policy",
    "goal_types",
    "backlog",
}


class CapabilityCatalogueTests(unittest.TestCase):
    def setUp(self):
        self.registry = CapabilityRegistry.default()

    def test_catalogue_loads_with_required_fields(self):
        capabilities = self.registry.list_all_capabilities()

        self.assertGreaterEqual(len(capabilities), 10)
        for capability in capabilities:
            data = capability.to_dict()
            self.assertTrue(REQUIRED_CAPABILITY_FIELDS.issubset(data.keys()), capability.id)
            self.assertTrue(capability.id)
            self.assertIn(capability.status, {"implemented", "partial", "planned"})

    def test_planned_not_implemented_backlog(self):
        planned_ids = {capability.id for capability in self.registry.list_planned_not_implemented()}

        self.assertIn("db.analytics_query", planned_ids)
        self.assertIn("obs.start_stream", planned_ids)

    def test_high_priority_unblocked_backlog(self):
        capability_ids = {capability.id for capability in self.registry.list_high_priority_unblocked()}

        self.assertIn("stream.chat_activity_report", capability_ids)
        self.assertIn("game.strategy_research", capability_ids)
        self.assertNotIn("obs.start_stream", capability_ids)

    def test_next_recommended_todo(self):
        capability = self.registry.next_recommended_todo()

        self.assertIsNotNone(capability)
        self.assertEqual(capability.id, "stream.chat_activity_report")

    def test_implemented_disabled_backlog(self):
        capability_ids = {capability.id for capability in self.registry.list_implemented_disabled()}

        self.assertIn("dev.full_dev_reset", capability_ids)

    def test_partial_needing_completion_backlog(self):
        capability_ids = {capability.id for capability in self.registry.list_partial_needing_completion()}

        self.assertIn("stream.chat_activity_report", capability_ids)
        self.assertIn("diagnostics.no_voice_response", capability_ids)

    def test_executable_excludes_planned_and_disabled(self):
        executable_ids = {capability.id for capability in self.registry.list_executable_capabilities()}

        self.assertIn("pc.open_application", executable_ids)
        self.assertNotIn("db.analytics_query", executable_ids)
        self.assertNotIn("dev.full_dev_reset", executable_ids)

    def test_check_available_rejects_planned_only(self):
        planned = self.registry.check_capability_available("db.analytics_query", current_mode="private")
        implemented = self.registry.check_capability_available("pc.open_application", current_mode="private")

        self.assertFalse(planned["available"])
        self.assertTrue(implemented["available"])

    def test_answer_backlog_query_returns_structured_data(self):
        response = self.registry.answer_backlog_query("partial_needs_completion")

        self.assertEqual(response["query_type"], "partial_needs_completion")
        self.assertGreater(response["count"], 0)
        self.assertIn("items", response)


if __name__ == "__main__":
    unittest.main()
