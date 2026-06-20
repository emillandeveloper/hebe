import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.cognitive.context_builder import ContextBuilder, BuiltContext
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.cognitive.models import ExecutionResult, StepExecutionResult, DeliberationResult, Plan


class DummyMemoryStore:
    def __init__(self):
        self.search_calls = []

    def search_facts(self, **kwargs):
        self.search_calls.append(kwargs)
        return []

    def get_recent_appointments(self, limit=3):
        return []

    def list_pending_reminders(self, limit=5):
        return []


class CapturingModel:
    def __init__(self, reply="Voy bien, Leo. Nada ardiendo todavía."):
        self.reply = reply
        self.messages = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        return self.reply


class ContextMessageTypeTests(unittest.TestCase):
    def test_small_talk_does_not_inject_memory(self):
        store = DummyMemoryStore()
        builder = ContextBuilder(store)

        with (
            patch("app.services.db_sqlite.get_recent_chat_turns", return_value=[]),
            patch.object(builder, "_retrieve_memory_for_jarvis", return_value=[]),
        ):
            ctx = builder.build(
                state=SimpleNamespace(stream=None),
                input_text="¿cómo lo llevas, Hebe?",
                internal_event=None,
            )

        self.assertEqual(ctx.message_type, "small_talk")
        self.assertFalse(ctx.inject_memory)
        self.assertEqual(ctx.relevant_facts, [])
        self.assertEqual(ctx.relevant_chunks, [])
        self.assertEqual(store.search_calls, [])

    def test_greeting_with_how_are_you_is_small_talk_policy_limited(self):
        store = DummyMemoryStore()
        builder = ContextBuilder(store)

        with (
            patch("app.services.db_sqlite.get_recent_chat_turns", return_value=[]),
            patch.object(builder, "_retrieve_memory_for_jarvis", return_value=[]),
        ):
            ctx = builder.build(
                state=SimpleNamespace(stream=None),
                input_text="buenas hebe, como lo llevas?",
                internal_event=None,
            )

        self.assertEqual(ctx.message_type, "small_talk")
        self.assertEqual(ctx.context_policy["memory"], "limited")
        self.assertFalse(ctx.context_policy["schedule"])
        self.assertEqual(ctx.context_policy["history_turns"], 2)
        self.assertFalse(ctx.inject_memory)

    def test_memory_query_allows_memory(self):
        store = DummyMemoryStore()
        builder = ContextBuilder(store)

        with (
            patch("app.services.db_sqlite.get_recent_chat_turns", return_value=[]),
            patch.object(builder, "_retrieve_memory_for_jarvis", return_value=[]),
        ):
            ctx = builder.build(
                state=SimpleNamespace(stream=None),
                input_text="¿qué hemos hablado antes?",
                internal_event=None,
            )

        self.assertEqual(ctx.message_type, "memory_query")
        self.assertTrue(ctx.inject_memory)

    def test_banter_policy_does_not_retrieve_schedule_memory(self):
        store = DummyMemoryStore()
        builder = ContextBuilder(store)

        with (
            patch("app.services.db_sqlite.get_recent_chat_turns", return_value=[]),
            patch.object(builder, "_retrieve_memory_for_jarvis", return_value=[]) as retrieve,
        ):
            ctx = builder.build(
                state=SimpleNamespace(stream=None),
                input_text="sigo en modo zombie jajaja",
                internal_event=None,
            )

        self.assertEqual(ctx.message_type, "banter")
        self.assertEqual(ctx.context_policy["memory"], "limited")
        self.assertFalse(ctx.context_policy["schedule"])
        self.assertEqual(ctx.context_policy["history_turns"], 2)
        self.assertFalse(ctx.inject_memory)
        retrieve.assert_not_called()

    def test_planning_request_allows_schedule_memory(self):
        store = DummyMemoryStore()
        builder = ContextBuilder(store)

        with (
            patch("app.services.db_sqlite.get_recent_chat_turns", return_value=[]),
            patch.object(builder, "_retrieve_memory_for_jarvis", return_value=[]) as retrieve,
        ):
            ctx = builder.build(
                state=SimpleNamespace(stream=None),
                input_text="que toca hoy en stream?",
                internal_event=None,
            )

        self.assertEqual(ctx.message_type, "planning_request")
        self.assertEqual(ctx.context_policy["memory"], "full")
        self.assertTrue(ctx.context_policy["schedule"])
        self.assertEqual(ctx.context_policy["history_turns"], 10)
        self.assertTrue(ctx.inject_memory)
        retrieve.assert_called_once()
        self.assertTrue(store.search_calls)

    def test_what_do_you_remember_allows_memory(self):
        store = DummyMemoryStore()
        builder = ContextBuilder(store)

        with (
            patch("app.services.db_sqlite.get_recent_chat_turns", return_value=[]),
            patch.object(builder, "_retrieve_memory_for_jarvis", return_value=[]),
        ):
            ctx = builder.build(
                state=SimpleNamespace(stream=None),
                input_text="¿qué recuerdas de mí?",
                internal_event=None,
            )

        self.assertEqual(ctx.message_type, "memory_query")
        self.assertTrue(ctx.inject_memory)

    def test_small_talk_prompt_blocks_recaps_and_memory(self):
        model = CapturingModel()
        synthesizer = ResponseSynthesizer(conversation_model=model)
        context = BuiltContext(
            input_text="¿cómo lo llevas, Hebe?",
            internal_event=None,
            relevant_facts=[],
            recent_appointments=[],
            pending_reminders=[],
            state_snapshot={},
            relevant_chunks=[{"subject": "old", "text": "We talked about old project context."}],
            conversation_history=[],
            message_type="small_talk",
            inject_memory=False,
            context_policy={
                "memory": "limited",
                "schedule": False,
                "history_turns": 2,
                "max_sentences": 2,
            },
        )
        execution = ExecutionResult(
            results=[
                StepExecutionResult(
                    step_type="reply",
                    success=True,
                    data={"mode": "chat"},
                )
            ]
        )

        reply = synthesizer.synthesize(
            context=context,
            deliberation=DeliberationResult(plan=Plan(steps=[])),
            execution=execution,
        )

        self.assertLessEqual(len([s for s in reply.split(".") if s.strip()]), 2)
        prompt_text = "\n".join(m["content"] for m in model.messages)
        self.assertIn("casual small talk", prompt_text)
        self.assertIn("Do not recap previous conversation", prompt_text)
        self.assertNotIn("We talked about old project context", prompt_text)
        self.assertNotIn("hablamos de", reply.lower())

    def test_fallback_chat_cannot_claim_pending_action_without_execution(self):
        model = CapturingModel(reply="Apuntado: cita el 16 de septiembre.")
        synthesizer = ResponseSynthesizer(conversation_model=model)
        context = BuiltContext(
            input_text="el 16 de septiembre",
            internal_event=None,
            relevant_facts=[],
            recent_appointments=[],
            pending_reminders=[],
            state_snapshot={"pending_clarification": {"kind": "appointment_datetime"}},
            relevant_chunks=[],
            conversation_history=[],
            message_type="small_talk",
            inject_memory=False,
            context_policy={"memory": "limited"},
        )
        context.cognitive_decision = SimpleNamespace(intent="unknown_chat")
        execution = ExecutionResult(results=[
            StepExecutionResult(step_type="reply", success=True, data={"mode": "chat"})
        ])
        logs = []

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            reply = synthesizer.synthesize(
                context=context,
                deliberation=DeliberationResult(plan=Plan(steps=[])),
                execution=execution,
            )

        self.assertNotIn("apuntado", reply.casefold())
        self.assertIn("no se ejecut", reply.casefold())
        self.assertIn("[HEBE][FALLBACK_GUARD] blocked_action_claim=true", "\n".join(logs))

    def test_banter_prompt_blocks_planning_topic_shift(self):
        model = CapturingModel(reply="Perfecto, zombi creativo. Sufrimiento premium, pero con estilo.")
        synthesizer = ResponseSynthesizer(conversation_model=model)
        context = BuiltContext(
            input_text="sigo en modo zombie jajaja",
            internal_event=None,
            relevant_facts=[],
            recent_appointments=[],
            pending_reminders=[],
            state_snapshot={},
            relevant_chunks=[
                {"subject": "stream", "text": "Leo may play FFIX Level 1 or Persona 5 Royal on stream."}
            ],
            conversation_history=[],
            message_type="banter",
            inject_memory=False,
            context_policy={
                "memory": "limited",
                "schedule": False,
                "history_turns": 2,
                "max_sentences": 2,
            },
        )
        execution = ExecutionResult(
            results=[
                StepExecutionResult(
                    step_type="reply",
                    success=True,
                    data={"mode": "chat"},
                )
            ]
        )

        reply = synthesizer.synthesize(
            context=context,
            deliberation=DeliberationResult(plan=Plan(steps=[])),
            execution=execution,
        )

        self.assertLessEqual(len([s for s in reply.split(".") if s.strip()]), 2)
        prompt_text = "\n".join(m["content"] for m in model.messages)
        self.assertIn("do not ask planning questions", prompt_text)
        self.assertNotIn("FFIX", prompt_text)
        self.assertNotIn("Persona", prompt_text)
        self.assertNotIn("stream", reply.lower())
        self.assertNotIn("FFIX", reply)
        self.assertNotIn("Persona", reply)


if __name__ == "__main__":
    unittest.main()
