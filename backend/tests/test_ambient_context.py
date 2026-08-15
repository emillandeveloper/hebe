import unittest

from app.stream.ambient_context import AmbientContextExtractor


class AmbientContextExtractorTests(unittest.TestCase):
    def setUp(self):
        self.extractor = AmbientContextExtractor()
        self.now = 1_000_000.0

    def category_for(self, text: str):
        result = self.extractor.extract(text, now=self.now)
        self.assertTrue(result.useful, result.reason)
        return result.facts[0]

    def categories_for(self, text: str):
        result = self.extractor.extract(text, now=self.now)
        self.assertTrue(result.useful, result.reason)
        return {fact["category"] for fact in result.facts}

    def test_healing_item_effectiveness_category(self):
        result = self.extractor.extract("Estos limones no curan nada, apenas sube la vida", now=self.now)
        self.assertTrue(result.useful, result.reason)
        categories = {fact["category"] for fact in result.facts}

        self.assertIn("healing_or_recovery", categories)
        fact = next(fact for fact in result.facts if fact["category"] == "healing_or_recovery")
        self.assertEqual(fact["source"], "stt_voice")
        self.assertIn("healing", fact["summary"])
        self.assertEqual(fact["raw_text"], "Estos limones no curan nada, apenas sube la vida")

    def test_unexpected_attack_category(self):
        fact = self.category_for("Joder que ataque ha sido ese, me ha borrado")

        self.assertEqual(fact["category"], "unexpected_attack")

    def test_guide_strategy_category(self):
        fact = self.category_for("La guia dice que este boss era facil y que use magia de luz")

        self.assertEqual(fact["category"], "guide_strategy")

    def test_enemy_mechanic_category(self):
        categories = self.categories_for("Parece que el boss se queda a 1 HP y luego se cura")

        self.assertIn("enemy_mechanic", categories)

    def test_low_value_navigation_filler_is_ignored(self):
        result = self.extractor.extract("por aqui", now=self.now)

        self.assertFalse(result.useful)
        self.assertEqual(result.reason, "generic_filler")

    def test_combat_risk_sentence_extracts_multiple_useful_categories(self):
        categories = self.categories_for("No me fio de que use Autopocion ni Counter y me quede a 15 de vida")

        self.assertIn("combat_risk", categories)
        self.assertIn("enemy_mechanic", categories)
        self.assertIn("healing_or_recovery", categories)

    def test_rng_dependency_in_challenge_context_is_specific(self):
        categories = self.categories_for("Lo jodido de este desafio es depender del RNG")

        self.assertIn("rng_dependency", categories)
        self.assertIn("challenge_constraint", categories)

    def test_death_and_game_over_extract_failure_not_navigation(self):
        categories = self.categories_for("Creo que he muerto")

        self.assertIn("failure_or_death", categories)
        self.assertNotIn("navigation_confusion", categories)

        categories = self.categories_for("No me voy a cansar de escuchar el Game Over")
        self.assertIn("failure_or_death", categories)
        self.assertNotIn("navigation_confusion", categories)

    def test_generic_filler_is_not_objective(self):
        for text in ("Vamos a ver", "Eso hay que hacerse mirar"):
            with self.subTest(text=text):
                result = self.extractor.extract(text, now=self.now)
                self.assertFalse(result.useful)
                self.assertEqual(result.reason, "generic_filler")

    def test_otra_vez_does_not_auto_classify_failure(self):
        text = "Mira, otra vez con lo de abre la puerta esa. Lleva todo el puto stream diciéndome que abra la puerta, pero te quieres callar con la puta puerta que no hay ninguna puerta."
        result = self.extractor.extract(text, now=self.now)
        self.assertTrue(result.useful or result.reason == "low_value")
        categories = {fact["category"] for fact in result.facts}
        self.assertNotIn("failure_or_death", categories)
        self.assertNotIn("navigation_confusion", categories)

    def test_otra_vez_by_itself_is_not_gameplay_failure(self):
        result = self.extractor.extract("otra vez", now=self.now)
        self.assertFalse(result.useful)


if __name__ == "__main__":
    unittest.main()
