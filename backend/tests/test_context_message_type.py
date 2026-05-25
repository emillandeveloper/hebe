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
        self.assertTrue(store.search_calls)

    def test_what_do_you_remember_allows_memory(self):
        store = DummyMemoryStore()
        builder = ContextBuilder(store)

        with patch("app.services.db_sqlite.get_recent_chat_turns", return_value=[]):
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


if __name__ == "__main__":
    unittest.main()
