import unittest

from app.cognitive.memory.memory_extractor import MemoryExtractor
from app.cognitive.persona.hebe_identity import (
    build_hebe_core_identity,
    build_private_mode_style,
    build_stream_mode_style,
)
from app.cognitive.persona.hebe_voice import build_stream_style_block, build_chat_react_examples


class HebeIdentityMemoryTests(unittest.TestCase):
    def test_identity_is_shared_by_private_and_stream_prompts(self):
        core = build_hebe_core_identity()
        private = f"{core}\n\n{build_private_mode_style()}"
        stream = build_stream_style_block()

        self.assertIn("Hebe is female", core)
        self.assertIn("Leo's companion", private)
        self.assertIn("Leo's companion", stream)
        self.assertIn("Maximum 240 characters", build_stream_mode_style())

    def test_stream_examples_include_spanish_and_english(self):
        examples = build_chat_react_examples()

        self.assertIn("[chatter]: hola hebe", examples)
        self.assertIn("[chatter]: hello hebe", examples)
        self.assertIn("[chatter]: hebe who are you", examples)

    def test_rule_extractor_captures_feminine_and_peninsular_spanish(self):
        extractor = MemoryExtractor(intent_model=None)
        result = extractor.extract(
            user_text=(
                "Recuerda que prefiero que hables en femenino y que si hablas "
                "español uses español de España."
            ),
            assistant_reply="Claro, Leo. Me quedo con eso.",
        )

        memories = result["memories"]
        texts = " ".join(m["text"] for m in memories)
        self.assertIn("feminine grammatical form", texts)
        self.assertIn("Spanish from Spain", texts)


if __name__ == "__main__":
    unittest.main()
