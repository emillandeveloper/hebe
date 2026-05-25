import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.cognitive.context_builder import ContextBuilder, BuiltContext
from app.cognitive.entity_resolver import EntityResolver, entity_prompt_lines
from app.cognitive.memory_store import MemoryFact
from app.cognitive.models import DeliberationResult, ExecutionResult, Plan, StepExecutionResult
from app.cognitive.response_synthesizer import ResponseSynthesizer


def fact(memory_id: int, entity_id: str, subject: str, text: str) -> MemoryFact:
    return MemoryFact(
        id=memory_id,
        kind="leo_fact",
        subject=subject,
        payload={
            "text": text,
            "entity_id": entity_id,
            "entity_type": "dog" if entity_id == "jotun_dog" else "channel_bot",
        },
        source_text=text,
        confidence=1.0,
        created_at="2026-05-25T00:00:00Z",
        updated_at="2026-05-25T00:00:00Z",
        last_used_at=None,
        active=True,
    )


class DummyMemoryStore:
    def __init__(self):
        self.facts = [
            fact(1, "jotun_dog", "Jotun", "Jotun es el perro de Leo."),
            fact(2, "jotun_bot", "JotunBot", "JotunBot es el bot del canal con comandos."),
        ]

    def search_facts(self, **kwargs):
        return list(self.facts)

    def get_recent_appointments(self, limit=3):
        return []

    def list_pending_reminders(self, limit=5):
        return []


class CapturingModel:
    def __init__(self, reply: str):
        self.reply = reply
        self.messages = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        return self.reply


class EntityDisambiguationTests(unittest.TestCase):
    def test_jotun_private_default_prefers_dog_and_allows_bot_second_for_broad_query(self):
        builder = ContextBuilder(DummyMemoryStore())

        with (
            patch("app.services.db_sqlite.get_recent_chat_turns", return_value=[]),
            patch.object(builder, "_retrieve_memory_for_jarvis", return_value=[]),
        ):
            ctx = builder.build(
                state=SimpleNamespace(stream=None),
                input_text="¿y quién es Jotun?",
                internal_event=None,
            )

        self.assertEqual(ctx.resolved_entities[0]["selected"], "jotun_dog")
        self.assertEqual(ctx.resolved_entities[0]["reason"], "private_chat_default")
        self.assertTrue(ctx.resolved_entities[0]["broad_query"])
        self.assertEqual([f.payload["entity_id"] for f in ctx.relevant_facts], ["jotun_dog", "jotun_bot"])

    def test_jotunbot_explicit_alias_prefers_bot_only(self):
        builder = ContextBuilder(DummyMemoryStore())

        with (
            patch("app.services.db_sqlite.get_recent_chat_turns", return_value=[]),
            patch.object(builder, "_retrieve_memory_for_jarvis", return_value=[]),
        ):
            ctx = builder.build(
                state=SimpleNamespace(stream=None),
                input_text="¿qué es JotunBot?",
                internal_event=None,
            )

        self.assertEqual(ctx.resolved_entities[0]["selected"], "jotun_bot")
        self.assertEqual(ctx.resolved_entities[0]["reason"], "explicit_alias")
        self.assertEqual([f.payload["entity_id"] for f in ctx.relevant_facts], ["jotun_bot"])

    def test_jotun_commands_prefers_bot_context(self):
        resolver = EntityResolver()
        result = resolver.resolve("qué comandos tiene Jotun?", source_context="private")

        self.assertEqual(result[0].selected, "jotun_bot")
        self.assertEqual(result[0].reason, "context_keyword")

    def test_jotun_health_prefers_dog_context(self):
        resolver = EntityResolver()
        result = resolver.resolve("cómo está Jotun?", source_context="private")

        self.assertEqual(result[0].selected, "jotun_dog")
        self.assertEqual(result[0].reason, "context_keyword")

    def test_hebe_prompt_identity_disambiguation(self):
        lines = entity_prompt_lines(
            [
                {
                    "mention": "Hebe",
                    "candidates": ("hebe_ai",),
                    "selected": "hebe_ai",
                    "reason": "explicit_alias",
                    "broad_query": True,
                }
            ]
        )

        self.assertTrue(lines)
        self.assertIn("Leo's companion", lines[0])
        self.assertIn("not a generic assistant", lines[0])

    def test_jotun_prompt_tells_model_dog_first(self):
        model = CapturingModel(
            "Jotun es tu perro, Leo. También forma parte de la identidad del canal."
        )
        synthesizer = ResponseSynthesizer(conversation_model=model)
        context = BuiltContext(
            input_text="¿y quién es Jotun?",
            internal_event=None,
            relevant_facts=[],
            recent_appointments=[],
            pending_reminders=[],
            state_snapshot={},
            relevant_chunks=[],
            conversation_history=[],
            message_type="direct_question",
            inject_memory=True,
            context_policy={
                "memory": "relevant",
                "schedule": False,
                "history_turns": 6,
                "max_sentences": None,
            },
            resolved_entities=[
                {
                    "mention": "Jotun",
                    "candidates": ("jotun_dog", "jotun_bot"),
                    "selected": "jotun_dog",
                    "reason": "private_chat_default",
                    "broad_query": True,
                }
            ],
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

        prompt_text = "\n".join(m["content"] for m in model.messages)
        self.assertIn("primarily means Leo's dog", prompt_text)
        self.assertIn("mention the dog first", prompt_text)
        self.assertIn("perro", reply.lower())


if __name__ == "__main__":
    unittest.main()
