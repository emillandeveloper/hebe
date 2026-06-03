import unittest
import tempfile
from pathlib import Path

from app.stream.game_profiles import GameProfileStore
from app.stream.game_research import GameKnowledgeResearchConfig, GameKnowledgeResearchService


class FakeSearchProvider:
    def __init__(self):
        self.calls = []

    def search(self, query):
        self.calls.append(query)
        return [
            {
                "title": "Spoiler-free overview",
                "snippet": "A spoiler-free gameplay overview with turn-based combat, equipment abilities, party resources, and whimsical fantasy tone.",
                "url": "https://example.com/spoiler-free",
            }
        ]


class GameProfileTests(unittest.TestCase):
    def setUp(self):
        self.store = GameProfileStore()

    def test_zwei_category_matches_zwei_profile(self):
        profile = self.store.lookup(current_category="Zwei!!: The Arges Adventure")

        self.assertEqual(profile.game_slug, "zwei_arges_adventure")
        self.assertEqual(profile.spoiler_policy, "no_spoilers")
        self.assertIn("food leveling", profile.safe_comment_topics)

    def test_ff9_title_matches_final_fantasy_ix_profile(self):
        profile = self.store.lookup(current_title="FF9 Level 1 Challenge Playthrough")

        self.assertEqual(profile.game_slug, "final_fantasy_ix")

    def test_uppercase_ff9_category_matches_final_fantasy_ix_profile(self):
        profile = self.store.lookup(current_category="FINAL FANTASY IX")

        self.assertEqual(profile.game_slug, "final_fantasy_ix")

    def test_unknown_game_falls_back_to_generic_profile(self):
        profile = self.store.lookup(current_category="Some Unknown Thing")

        self.assertEqual(profile.game_slug, "generic_jrpg_rpg")

    def test_no_profile_file_causes_no_crash(self):
        store = GameProfileStore(path="missing_profiles_file.json")
        profile = store.lookup(current_category="Zwei!!: The Arges Adventure")

        self.assertEqual(profile.game_slug, "generic_jrpg_rpg")

    def test_research_disabled_uses_local_fallback_without_crash(self):
        provider = FakeSearchProvider()
        service = GameKnowledgeResearchService(
            store=self.store,
            config=GameKnowledgeResearchConfig(enabled=False),
            search_provider=provider,
        )

        ok, profile, reason = service.research_current_game(current_category="Unknown Game")

        self.assertFalse(ok)
        self.assertEqual(reason, "research_disabled")
        self.assertEqual(profile.game_slug, "generic_jrpg_rpg")
        self.assertEqual(provider.calls, [])

    def test_research_result_is_cached_and_prevents_repeated_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "profiles.cache.json"
            store = GameProfileStore(cache_path=cache_path)
            provider = FakeSearchProvider()
            service = GameKnowledgeResearchService(
                store=store,
                config=GameKnowledgeResearchConfig(enabled=True, provider="fake", cache_days=30),
                search_provider=provider,
                now_fn=lambda: 1_000_000.0,
            )

            ok, profile, reason = service.research_current_game(current_category="Imaginary RPG")
            ok2, profile2, reason2 = service.research_current_game(current_category="Imaginary RPG")

            self.assertTrue(ok)
            self.assertEqual(reason, "researched")
            self.assertTrue(cache_path.exists())
            self.assertTrue(ok2)
            self.assertEqual(reason2, "cached_profile")
            self.assertEqual(profile2.game_slug, profile.game_slug)
            self.assertEqual(len(provider.calls), 1)


if __name__ == "__main__":
    unittest.main()
