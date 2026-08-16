import unittest

from app.cognitive.speech_act_pipeline import (
    build_universal_speech_act_bundle,
    final_response_guard,
)
from app.stream.game_advice_gate import (
    COMMON_LANGUAGE_COLLISION,
    CONTEXTUAL_MECHANIC,
    ENTITY_COLLISION,
    UNAMBIGUOUS_MECHANIC,
    GameAdviceGate,
)


class GameAdviceSemanticGroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = GameAdviceGate()

    def test_references_do_not_become_mechanic_claims(self) -> None:
        cases = (
            (
                "Persona 5 Royal: tengo perfil spoiler-safe, pero no memoria de la run de Leo.",
                ["Persona 5 Royal"],
            ),
            ("Es una persona amable.", []),
            ("Hoy es un buen día.", []),
            ("La media fue alta.", []),
            ("Me gusta Mario Party.", ["Mario Party"]),
            ("El equipo está preparado.", []),
            ("Materia: entrada del glosario.", ["Materia"]),
        )

        for text, entity_spans in cases:
            with self.subTest(text=text):
                result = self.gate.validate(
                    current_game="Unknown Game",
                    proposed_advice=text,
                    entity_spans=entity_spans,
                )
                self.assertTrue(result.allowed, result.to_dict())
                self.assertEqual(result.mechanics, [])
                self.assertEqual(result.mechanic_assertions, [])
                self.assertEqual(result.mechanic_instructions, [])

    def test_instructions_and_assertions_remain_fail_closed(self) -> None:
        cases = (
            ("Fusiona una Persona antes del jefe.", "instruction", "personas"),
            ("Las Personas se fusionan para crear otras.", "assertion", "personas"),
            ("Usa Dia para curarte.", "instruction", "healing_skills"),
            ("Baton Pass aumenta el daño.", "assertion", "baton_pass"),
        )

        for text, semantic_kind, mechanic in cases:
            with self.subTest(text=text):
                result = self.gate.validate(
                    current_game="Unknown Game",
                    proposed_advice=text,
                    source_evidence=[],
                )
                self.assertFalse(result.allowed, result.to_dict())
                self.assertIn(mechanic, result.blocked)
                semantic_claims = (
                    result.mechanic_instructions
                    if semantic_kind == "instruction"
                    else result.mechanic_assertions
                )
                self.assertIn(mechanic, semantic_claims)

    def test_mixed_title_and_mechanic_claim_are_classified_independently(self) -> None:
        persona = self.gate.validate(
            current_game="Unknown Game",
            proposed_advice="En Persona 5 Royal, Baton Pass aumenta el daño.",
            entity_spans=["Persona 5 Royal"],
            source_evidence=[],
        )
        self.assertFalse(persona.allowed)
        self.assertIn("Persona 5 Royal", persona.entity_references)
        self.assertIn("baton_pass", persona.mechanic_assertions)
        self.assertNotIn("personas", persona.mechanics)

        mario = self.gate.validate(
            current_game="Unknown Game",
            proposed_advice="En Mario Party, el party member aumenta el ataque.",
            entity_spans=["Mario Party"],
            source_evidence=[],
        )
        self.assertFalse(mario.allowed)
        self.assertIn("Mario Party", mario.entity_references)
        self.assertIn("party_members", mario.mechanic_assertions)

        single_word_entity_claim = self.gate.validate(
            current_game="Unknown Game",
            proposed_advice="Materia aumenta la magia.",
            entity_spans=["Materia"],
            source_evidence=[],
        )
        self.assertFalse(single_word_entity_claim.allowed)
        self.assertIn("Materia", single_word_entity_claim.entity_references)
        self.assertIn("materia", single_word_entity_claim.mechanic_assertions)

    def test_command_result_game_title_is_a_reference_not_evidence(self) -> None:
        bundle = build_universal_speech_act_bundle(
            route="command_result:game_knowledge_query",
            speech_act_type="action_confirmation",
            input_text="Que sabes de Persona 5 Royal?",
            execution_result={
                "step_type": "command_result",
                "action": "game_knowledge_query",
                "success": True,
                "data": {
                    "game_title": "Persona 5 Royal",
                    "game_profile": {"title": "Persona 5 Royal"},
                },
            },
        )
        self.assertEqual(bundle.scene.entity_references, ["Persona 5 Royal"])

        reference = final_response_guard(
            "Persona 5 Royal: tengo perfil spoiler-safe, pero no memoria de la run de Leo.",
            bundle,
            game_advice_gate=self.gate,
        )
        self.assertTrue(reference.passed, reference.to_dict())

        invented_claim = final_response_guard(
            "En Persona 5 Royal, Baton Pass aumenta el daño.",
            bundle,
            game_advice_gate=self.gate,
        )
        self.assertFalse(invented_claim.passed)
        self.assertIn(
            "unvalidated_game_mechanics",
            [violation.type for violation in invented_claim.violations],
        )
        self.assertIn("baton_pass", invented_claim.game_advice_validation["blocked"])
        self.assertNotIn("personas", invented_claim.game_advice_validation["mechanics"])

    def test_known_registry_titles_are_data_driven_reference_regressions(self) -> None:
        collisions = []
        for profile in self.gate.registry._profiles:
            for title in (profile.canonical_title, *profile.aliases):
                if self.gate.detect_mechanics(title):
                    collisions.append(title)
                    with self.subTest(title=title):
                        result = self.gate.validate(
                            current_game="Unknown Game",
                            proposed_advice=f"{title}: perfil disponible.",
                            entity_spans=[title],
                        )
                        self.assertTrue(result.allowed, result.to_dict())
                        self.assertEqual(result.mechanics, [])
        self.assertTrue(collisions, "registry fixture must exercise at least one title/alias collision")

    def test_common_language_collision_corpus_is_data_driven(self) -> None:
        corpus = {
            "Es una persona amable.": "personas",
            "Hoy es un buen día.": "healing_skills",
            "La media fue alta.": "healing_skills",
            "Me gusta Mario Party.": "party_members",
            "El equipo está preparado.": "equipment_stats",
            "La materia es complicada.": "materia",
            "Guard es un título.": "guard",
            "Trance es un título.": "trance",
            "Las cartas llegaron ayer.": "card_deck",
        }
        for text, lexical_mechanic in corpus.items():
            with self.subTest(text=text):
                self.assertIn(lexical_mechanic, self.gate.detect_mechanics(text))
                semantic = self.gate.analyze_semantics(text)
                self.assertEqual(semantic.mechanics, [])

    def test_alias_collision_classes_are_explicit(self) -> None:
        self.assertEqual(self.gate.alias_classification("persona"), ENTITY_COLLISION)
        self.assertEqual(self.gate.alias_classification("materia"), ENTITY_COLLISION)
        self.assertEqual(self.gate.alias_classification("día"), COMMON_LANGUAGE_COLLISION)
        self.assertEqual(self.gate.alias_classification("media"), COMMON_LANGUAGE_COLLISION)
        self.assertEqual(self.gate.alias_classification("poción"), CONTEXTUAL_MECHANIC)
        self.assertEqual(self.gate.alias_classification("baton pass"), UNAMBIGUOUS_MECHANIC)


if __name__ == "__main__":
    unittest.main()
