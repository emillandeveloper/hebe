from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.continuity.repository import OpenThreadRepository
from app.epistemics.repository import BeliefRepository
from app.epistemics.service import BeliefLifecycleService
from app.replay.migrations import (
    MigrationRunner,
    belief_v2_migrations,
    conversation_continuity_migrations,
    social_world_v2_migrations,
)
from app.social_world_v2 import SocialIdentityConflict, SocialWorldRepository, SocialWorldService
from app.social_world_v2.migration import (
    social_identity_canonicalization_migrations,
    social_summary_canonicalization_migrations,
)


class SocialCanonicalizationPhase1DTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "social.sqlite3"
        self.connect = lambda: sqlite3.connect(self.path)
        self._base_schema()
        self.runner = MigrationRunner(self.connect)
        self.runner.migrate(social_identity_canonicalization_migrations())
        self.runner.migrate(social_summary_canonicalization_migrations())
        self.repository = SocialWorldRepository(self.connect)
        beliefs = BeliefLifecycleService(BeliefRepository(self.connect), now_fn=lambda: 1000.0)
        self.social = SocialWorldService(
            self.repository,
            beliefs,
            OpenThreadRepository(self.connect),
            now_fn=lambda: 1000.0,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _base_schema(self) -> None:
        runner = MigrationRunner(self.connect)
        for migrations in (
            conversation_continuity_migrations(),
            belief_v2_migrations(),
            social_world_v2_migrations(),
        ):
            runner.migrate(migrations)

    def _legacy_database(self) -> tuple[Path, callable]:
        path = Path(self.tmp.name) / f"legacy-{len(list(Path(self.tmp.name).glob('legacy-*')))}.sqlite3"
        connect = lambda: sqlite3.connect(path)
        runner = MigrationRunner(connect)
        for migrations in (
            conversation_continuity_migrations(),
            belief_v2_migrations(),
            social_world_v2_migrations(),
        ):
            runner.migrate(migrations)
        conn = connect()
        conn.executescript(
            """
            CREATE TABLE chatter_profiles(
              username TEXT PRIMARY KEY,display_name TEXT,aliases_json TEXT,
              first_seen_at TEXT,last_seen_at TEXT,updated_at TEXT
            );
            CREATE TABLE chatter_presence(
              id INTEGER PRIMARY KEY,stream_session_id INTEGER,username TEXT,
              first_seen_at TEXT,last_seen_at TEXT,first_message_at TEXT,last_message_at TEXT,
              last_direct_interaction_at TEXT,message_count INTEGER,direct_interaction_count INTEGER,
              presence_source_json TEXT
            );
            CREATE TABLE stream_chatter_summaries(
              id INTEGER PRIMARY KEY,stream_session_id INTEGER,username TEXT,summary_text TEXT,
              topics_json TEXT,message_count INTEGER,direct_interaction_count INTEGER,
              created_at TEXT,notable_quotes_json TEXT,inferred_facts_json TEXT
            );
            CREATE TABLE viewer_promotion_profiles(
              twitch_user_id TEXT PRIMARY KEY,current_login TEXT,known_aliases_json TEXT,
              created_by TEXT,owner_locked INTEGER
            );
            """
        )
        conn.commit()
        conn.close()
        return path, connect

    @staticmethod
    def _legacy_profile(conn: sqlite3.Connection, login: str, *, session_id: int = 7) -> None:
        conn.execute(
            "INSERT INTO chatter_profiles VALUES(?,?,?, ?,?,?)",
            (login, login.title(), "[]", "2026-08-01T10:00:00+00:00", "2026-08-01T11:00:00+00:00", "2026-08-01T11:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO chatter_presence VALUES(1,?,?, ?,?,?,?,?,?,?,?)",
            (
                session_id, login, "2026-08-01T10:00:00+00:00", "2026-08-01T11:00:00+00:00",
                "2026-08-01T10:00:00+00:00", "2026-08-01T11:00:00+00:00", "",
                4, 1, '["chat"]',
            ),
        )

    def test_a_known_twitch_stable_id_resolves_one_person(self) -> None:
        first, _ = self.social.resolve_person(platform_user_id="42", login="viewer")
        second, _ = self.social.resolve_person(platform_user_id="42", login="viewer")
        self.assertEqual(first.person_id, second.person_id)
        self.assertEqual(len(self.repository.people()), 1)

    def test_b_same_stable_id_rename_updates_alias_without_new_person(self) -> None:
        first, _ = self.social.resolve_person(platform_user_id="42", login="old_name")
        second, identity = self.social.resolve_person(platform_user_id="42", login="new_name")
        self.assertEqual(first.person_id, second.person_id)
        self.assertEqual(identity.login, "new_name")
        self.assertEqual(identity.aliases, ("old_name", "new_name"))

    def test_c_display_name_change_does_not_change_identity(self) -> None:
        first, _ = self.social.resolve_person(platform_user_id="42", login="viewer", display_name="Viewer")
        second, identity = self.social.resolve_person(platform_user_id="42", login="viewer", display_name="VIEWER!")
        self.assertEqual(first.person_id, second.person_id)
        self.assertEqual(identity.display_name, "VIEWER!")

    def test_d_same_username_with_different_stable_ids_stays_distinct(self) -> None:
        first, _ = self.social.resolve_person(platform_user_id="42", login="shared")
        second, _ = self.social.resolve_person(platform_user_id="99", login="shared")
        self.assertNotEqual(first.person_id, second.person_id)
        self.assertEqual(len(self.repository.people()), 2)

    def test_e_owner_verified_legacy_profile_migrates_to_stable_identity(self) -> None:
        _path, connect = self._legacy_database()
        conn = connect();self._legacy_profile(conn, "verified")
        conn.execute("INSERT INTO viewer_promotion_profiles VALUES('42','verified','[]','owner_command',1)")
        conn.commit();conn.close()
        runner = MigrationRunner(connect);runner.migrate(social_identity_canonicalization_migrations())
        conn = connect();audit = conn.execute("SELECT classification,target_person_id FROM legacy_social_identity_migration_audit").fetchone()
        identity = conn.execute("SELECT platform_user_id FROM person_identities WHERE person_id=?", (audit[1],)).fetchone();conn.close()
        self.assertEqual(audit[0], "SAFE_TO_MIGRATE")
        self.assertEqual(identity[0], "42")

    def test_f_same_event_modern_identity_is_deterministic_mapping(self) -> None:
        _path, connect = self._legacy_database();conn = connect();self._legacy_profile(conn, "linked")
        at = 1785578400.0
        conn.execute("INSERT INTO people VALUES('person-linked',?,?, 'stream_public',1)", (at, at))
        conn.execute("INSERT INTO person_identities VALUES('identity-linked','person-linked','twitch','42','linked','Linked','[\"linked\"]',?,?,1,'twitch_chat',1)", (at, at))
        conn.execute("INSERT INTO person_sessions(person_id,stream_session_id,first_seen_at,last_seen_at) VALUES('person-linked','7',?,?)", (at, at))
        conn.commit();conn.close()
        MigrationRunner(connect).migrate(social_identity_canonicalization_migrations())
        conn = connect();audit = conn.execute("SELECT classification,target_person_id FROM legacy_social_identity_migration_audit").fetchone();conn.close()
        self.assertEqual(audit, ("ALREADY_CANONICAL", "person-linked"))

    def test_g_ambiguous_name_match_is_not_fused(self) -> None:
        _path, connect = self._legacy_database();conn = connect();self._legacy_profile(conn, "uncertain")
        conn.execute("INSERT INTO people VALUES('p',1,1,'stream_public',1)")
        conn.execute("INSERT INTO person_identities VALUES('i','p','twitch','42','uncertain','Uncertain','[]',1,1,1,'old',1)")
        conn.commit();conn.close();MigrationRunner(connect).migrate(social_identity_canonicalization_migrations())
        conn = connect();audit = conn.execute("SELECT classification,outcome FROM legacy_social_identity_migration_audit").fetchone();conn.close()
        self.assertEqual(audit, ("AMBIGUOUS", "ambiguous"))

    def test_h_conflicting_modern_people_are_audited_without_overwrite(self) -> None:
        _path, connect = self._legacy_database();conn = connect();self._legacy_profile(conn, "conflict")
        for suffix, stable in (("a", ""), ("b", "42")):
            conn.execute("INSERT INTO people VALUES(?,?,?,'stream_public',1)", (f"p-{suffix}", 1, 1))
            conn.execute("INSERT INTO person_identities VALUES(?,?,?,?,?,?, '[]',1,1,?, 'test',1)", (f"i-{suffix}", f"p-{suffix}", "twitch", stable, "conflict", "Conflict", 1 if stable else .6))
        conn.commit();conn.close();MigrationRunner(connect).migrate(social_identity_canonicalization_migrations())
        conn = connect();audit = conn.execute("SELECT classification,outcome FROM legacy_social_identity_migration_audit").fetchone();people = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0];conn.close()
        self.assertEqual(audit, ("CONFLICT", "conflict"));self.assertEqual(people, 2)

    def test_i_verified_profile_summary_gets_canonical_person_link(self) -> None:
        _path, connect = self._legacy_database();conn = connect();self._legacy_profile(conn, "verified")
        conn.execute("INSERT INTO viewer_promotion_profiles VALUES('42','verified','[]','owner_command',1)")
        conn.execute("INSERT INTO stream_chatter_summaries VALUES(1,7,'verified','context only','[\"game\"]',4,1,'2026-08-01T11:00:00+00:00','[\"raw omitted\"]','[]')")
        conn.commit();conn.close();runner=MigrationRunner(connect);runner.migrate(social_identity_canonicalization_migrations());runner.migrate(social_summary_canonicalization_migrations())
        conn=connect();row=conn.execute("SELECT summary_text,topics_json FROM social_summaries").fetchone();audit=conn.execute("SELECT classification,outcome,details_json FROM legacy_social_summary_migration_audit").fetchone();conn.close()
        self.assertEqual(row, ("context only", '["game"]'));self.assertEqual(audit[:2], ("MIGRATABLE", "migrated"));self.assertFalse(json.loads(audit[2])["raw_quotes_copied"])

    def test_j_conflicting_profile_summary_is_not_assigned(self) -> None:
        _path, connect = self._legacy_database();conn=connect();self._legacy_profile(conn,"conflict")
        for suffix, stable in (("a", ""), ("b", "42")):
            conn.execute("INSERT INTO people VALUES(?,?,?,'stream_public',1)",(f"p-{suffix}",1,1));conn.execute("INSERT INTO person_identities VALUES(?,?,?,?,?,?, '[]',1,1,?, 'test',1)",(f"i-{suffix}",f"p-{suffix}","twitch",stable,"conflict","Conflict",1 if stable else .6))
        conn.execute("INSERT INTO stream_chatter_summaries VALUES(1,7,'conflict','context','[]',2,0,'2026-08-01T11:00:00+00:00','[]','[]')");conn.commit();conn.close()
        runner=MigrationRunner(connect);runner.migrate(social_identity_canonicalization_migrations());runner.migrate(social_summary_canonicalization_migrations());conn=connect();audit=conn.execute("SELECT classification FROM legacy_social_summary_migration_audit").fetchone()[0];count=conn.execute("SELECT COUNT(*) FROM social_summaries").fetchone()[0];conn.close()
        self.assertEqual(audit,"AMBIGUOUS_OWNER");self.assertEqual(count,0)

    def test_k_migrations_are_idempotent(self) -> None:
        first_identity=self.runner.migrate(social_identity_canonicalization_migrations())[0]
        first_summary=self.runner.migrate(social_summary_canonicalization_migrations())[0]
        self.assertTrue(first_identity["already_applied"]);self.assertTrue(first_summary["already_applied"])

    def test_incompatible_legacy_schema_does_not_mark_completion(self) -> None:
        path=Path(self.tmp.name)/"bad-schema.sqlite3";connect=lambda:sqlite3.connect(path);runner=MigrationRunner(connect)
        for migrations in (conversation_continuity_migrations(),belief_v2_migrations(),social_world_v2_migrations()):runner.migrate(migrations)
        conn=connect();conn.execute("CREATE TABLE chatter_profiles(username TEXT PRIMARY KEY)");conn.commit();conn.close()
        with self.assertRaisesRegex(RuntimeError,"unsupported chatter_profiles schema"):
            runner.migrate(social_identity_canonicalization_migrations())
        conn=connect();marked=conn.execute("SELECT COUNT(*) FROM schema_migrations WHERE component='social_identity_canonicalization'").fetchone()[0];conn.close()
        self.assertEqual(marked,0)

    def test_l_post_cutover_viewer_writes_only_canonical_tables(self) -> None:
        conn=self.connect();conn.execute("CREATE TABLE chatter_profiles(username TEXT PRIMARY KEY)");conn.execute("CREATE TABLE stream_chatter_summaries(id INTEGER PRIMARY KEY)");conn.commit();conn.close()
        person,_,_=self.social.observe_presence(observation_id="message-1",platform_user_id="42",login="new",stream_session_id="s1",source="twitch_chat",message_seen=True)
        self.social.record_summary_for_login(login="new",stream_session_id="s1",source_record_id="summary-1",summary_text="context",message_count=1)
        conn=self.connect();legacy=(conn.execute("SELECT COUNT(*) FROM chatter_profiles").fetchone()[0],conn.execute("SELECT COUNT(*) FROM stream_chatter_summaries").fetchone()[0]);modern=(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0],conn.execute("SELECT COUNT(*) FROM social_summaries").fetchone()[0]);conn.close()
        self.assertEqual(legacy,(0,0));self.assertEqual(modern,(1,1));self.assertTrue(person.person_id)

    def test_m_rename_preserves_sessions_and_familiarity(self) -> None:
        person,_,_=self.social.observe_presence(observation_id="m1",platform_user_id="42",login="old",stream_session_id="s1",source="chat",message_seen=True)
        self.social.observe_presence(observation_id="m2",platform_user_id="42",login="new",stream_session_id="s2",source="chat",message_seen=True)
        self.assertEqual(self.repository.familiarity(person.person_id)["distinct_sessions"],2)
        self.assertEqual(len(self.repository.people()),1)

    def test_n_missing_stable_id_is_explicitly_unverified_and_conflicts_fail_closed(self) -> None:
        first,identity=self.social.resolve_person(platform_user_id="",login="unverified")
        second,_=self.social.resolve_person(platform_user_id="",login="unverified")
        self.assertEqual(first.person_id,second.person_id);self.assertEqual(identity.platform_user_id,"");self.assertLess(identity.confidence,1)
        self.social.resolve_person(platform_user_id="42",login="shared");self.social.resolve_person(platform_user_id="99",login="shared")
        with self.assertRaises(SocialIdentityConflict):self.social.resolve_person(platform_user_id="",login="shared")

    def test_o_presence_observation_is_applied_once(self) -> None:
        first=self.social.observe_presence(observation_id="same",platform_user_id="42",login="viewer",stream_session_id="s1",source="chat",message_seen=True)
        second=self.social.observe_presence(observation_id="same",platform_user_id="42",login="viewer",stream_session_id="s1",source="chat",message_seen=True)
        conn=self.connect();counts=(conn.execute("SELECT COUNT(*) FROM social_presence_events").fetchone()[0],conn.execute("SELECT message_count FROM person_sessions").fetchone()[0]);conn.close()
        self.assertTrue(first[2]);self.assertFalse(second[2]);self.assertEqual(counts,(1,1))

    def test_p_familiarity_never_grants_owner_authority(self) -> None:
        person,_=self.social.resolve_person(platform_user_id="42",login="regular")
        for index in range(5):self.social.observe_presence(observation_id=f"m{index}",platform_user_id="42",login="regular",stream_session_id=f"s{index}",source="chat",message_seen=True)
        context=self.social.retrieve_social_context(person.person_id)
        self.assertEqual(context.familiarity["band"],"regular");self.assertEqual(context.domain_authority_refs,())


if __name__ == "__main__":
    unittest.main()
