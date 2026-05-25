import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.cognitive.deliberation_service import DeliberationService
from app.cognitive.memory_store import MemoryStore
from app.cognitive.plan_executor import PlanExecutor
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.services import db_sqlite


def make_context(text: str):
    return SimpleNamespace(
        input_text=text,
        internal_event=None,
        state_snapshot={},
    )


class DeterministicReminderTests(unittest.TestCase):
    def setUp(self):
        self._old_db_path = db_sqlite.DB_PATH
        self._tmp = tempfile.TemporaryDirectory()
        db_sqlite.DB_PATH = os.path.join(self._tmp.name, "hebe_test.sqlite3")
        db_sqlite.init_db()

    def tearDown(self):
        db_sqlite.DB_PATH = self._old_db_path
        self._tmp.cleanup()

    def _plan_for(self, text: str):
        service = DeliberationService(intent_model=None, reasoning_model=None)
        return service.deliberate(make_context(text)).plan

    def test_recuerdame_en_un_minuto_creates_real_reminder_plan(self):
        plan = self._plan_for("recuérdame en 1 minuto que activé la voz")

        self.assertEqual([step.type for step in plan.steps], ["reminder", "reply"])
        self.assertEqual(plan.steps[1].data["mode"], "confirm_reminder")
        self.assertNotEqual(plan.steps[1].data.get("mode"), "clarify_appointment_datetime")
        self.assertIn("activé la voz", plan.steps[0].data["message"])
        self.assertEqual(plan.steps[1].data["relative_label"], "1 minuto")

    def test_avisame_en_tres_minutos_executes_and_confirms_without_followup(self):
        plan = self._plan_for("avísame en 3 minutos que te revise")
        executor = PlanExecutor(
            memory_store=MemoryStore(),
            action_runtime=Mock(),
        )
        execution = executor.execute(plan)
        reply = ResponseSynthesizer(conversation_model=None).synthesize(
            context=make_context("avísame en 3 minutos que te revise"),
            deliberation=SimpleNamespace(plan=plan),
            execution=execution,
        )

        pending = db_sqlite.list_pending_reminders(limit=10)
        self.assertEqual(len(pending), 1)
        self.assertIn("te revise", pending[0]["message"])
        self.assertEqual(reply, "Vale, Leo. Te aviso en 3 minutos.")
        self.assertNotIn("quieres", reply.lower())

    def test_dentro_de_diez_minutos_recuerdame_extracts_message(self):
        plan = self._plan_for("dentro de 10 minutos recuérdame tomar agua")

        self.assertEqual(plan.steps[0].type, "reminder")
        self.assertIn("tomar agua", plan.steps[0].data["message"])
        self.assertNotIn("10 minutos", plan.steps[0].data["message"])
        self.assertEqual(plan.steps[1].data["relative_label"], "10 minutos")

    def test_ponme_un_recordatorio_extracts_message(self):
        plan = self._plan_for("ponme un recordatorio en 1 minuto para revisar OBS")

        self.assertEqual(plan.steps[0].type, "reminder")
        self.assertIn("revisar OBS", plan.steps[0].data["message"])
        self.assertEqual(plan.steps[1].data["relative_label"], "1 minuto")


if __name__ == "__main__":
    unittest.main()
