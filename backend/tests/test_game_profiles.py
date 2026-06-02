import unittest

from app.stream.game_profiles import GameProfileStore


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

    def test_unknown_game_falls_back_to_generic_profile(self):
        profile = self.store.lookup(current_category="Some Unknown Thing")

        self.assertEqual(profile.game_slug, "generic_jrpg_rpg")

    def test_no_profile_file_causes_no_crash(self):
        store = GameProfileStore(path="missing_profiles_file.json")
        profile = store.lookup(current_category="Zwei!!: The Arges Adventure")

        self.assertEqual(profile.game_slug, "generic_jrpg_rpg")


if __name__ == "__main__":
    unittest.main()
