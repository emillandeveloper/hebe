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
            ("Eve, que toca hoy?", "eve", "que toca hoy"),
            ("lista para darlo todo, Eve", "eve", "lista para darlo todo"),
            ("Ebi responde", "ebi", "responde"),
            ("Heb habla por Twitch", "heb", "habla por twitch"),
            ("E.B. duerme", "eb", "duerme"),
            ("EB haz promo a Totodile", "eb", "haz promo a totodile"),
        ]

        for text, matched_name, stripped in cases:
            with self.subTest(text=text):
                result = self.resolve(text)
                self.assertTrue(result.addressed_to_hebe)
                self.assertEqual(result.matched_name, matched_name)
                self.assertEqual(result.canonical, "hebe")
                self.assertEqual(result.stripped_text, stripped)

    def test_eve_is_direct_when_vocative_or_command_shaped(self):
        vocative = self.resolve("Eve mira esto")
        command = self.resolve("Eve despierta")

        self.assertTrue(vocative.addressed_to_hebe)
        self.assertEqual(vocative.reason, "stt_alias_vocative")
        self.assertTrue(command.addressed_to_hebe)
        self.assertTrue(command.wake_command)

    def test_se_ve_does_not_false_wake(self):
        for text in ("como se ve", "no se ve", "se ve bien"):
            with self.subTest(text=text):
                result = self.resolve(text)
                self.assertFalse(result.addressed_to_hebe)

    def test_eve_needs_direct_context_when_not_vocative(self):
        weak = self.resolve("Eve mira esto")
        middle = self.resolve("esto Eve no tiene forma directa")

        self.assertTrue(weak.addressed_to_hebe)
        self.assertFalse(middle.addressed_to_hebe)
        self.assertEqual(middle.reason, "weak_eve_context")

    def test_direct_question_aliases_address_hebe(self):
        cases = [
            "Hebe, me escuchas?",
            "Ebe, me escuchas?",
            "Eve, me escuchas?",
            "Me escuchas, Ebe?",
            "Que tal estas, Eve?",
        ]

        for text in cases:
            with self.subTest(text=text):
                result = self.resolve(text)
                self.assertTrue(result.addressed_to_hebe)
                self.assertEqual(result.canonical, "hebe")

    def test_trusted_command_without_name_is_command_evidence_not_addressing(self):
        result = self.resolve("pon stt ambiental")

        self.assertFalse(result.addressed_to_hebe)
        self.assertIsNone(result.matched_name)
        self.assertEqual(result.reason, "trusted_source_command_evidence_without_wake")

    def test_single_h_is_not_a_wake_name_or_stripped_target(self):
        result = self.resolve("haz promo al h")

        self.assertFalse(result.addressed_to_hebe)
        self.assertIsNone(result.matched_name)
        self.assertIn("h", result.stripped_text)

    def test_eve_wake_preserves_h_promo_target(self):
        result = self.resolve("Eve haz promo al h")

        self.assertTrue(result.addressed_to_hebe)
        self.assertEqual(result.matched_name, "eve")
        self.assertEqual(result.stripped_text, "haz promo al h")


if __name__ == "__main__":
    unittest.main()
