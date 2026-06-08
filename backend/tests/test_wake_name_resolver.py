import unittest

from app.cognitive.wake_name_resolver import WakeNameResolver


class WakeNameResolverTests(unittest.TestCase):
    def setUp(self):
        self.resolver = WakeNameResolver()
        self.command_markers = {"despierta", "duerme", "pon", "stt", "promo"}

    def resolve(self, text: str):
        return self.resolver.resolve(
            raw_text=text,
            normalized_text=text,
            source="stt_voice",
            command_markers=self.command_markers,
        )

    def test_resolves_common_wake_name_variants(self):
        cases = [
            ("Hebe despierta", "hebe", "despierta"),
            ("Ebe pon STT ambiental", "ebe", "pon stt ambiental"),
            ("E.B. duerme", "eb", "duerme"),
            ("EB haz promo a Totodile", "eb", "haz promo a totodile"),
        ]

        for text, matched_name, stripped in cases:
            with self.subTest(text=text):
                result = self.resolve(text)
                self.assertTrue(result.addressed_to_hebe)
                self.assertEqual(result.matched_name, matched_name)
                self.assertEqual(result.stripped_text, stripped)

    def test_eve_needs_command_context_when_awake(self):
        weak = self.resolve("Eve mira esto")
        strong = self.resolve("Eve despierta")

        self.assertFalse(weak.addressed_to_hebe)
        self.assertEqual(weak.reason, "weak_eve_context")
        self.assertTrue(strong.addressed_to_hebe)
        self.assertTrue(strong.wake_command)


if __name__ == "__main__":
    unittest.main()
