from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app.integrity.hygiene import HygienePlanner
from app.integrity.migration import ALL_MIGRATIONS, migrate, verify_copied_database
from app.integrity.ownership import inventory
from app.integrity.production_defaults import production_defaults
from app.integrity.scanner import IntegrityScanner
from app.replay.migrations import MigrationRunner


class Phase6Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/"phase6.sqlite3"
        self.connect=lambda:sqlite3.connect(self.path)
        runner=MigrationRunner(self.connect)
        for factory in ALL_MIGRATIONS:runner.migrate(factory())
        self.now=10_000.0
    def tearDown(self):self.tmp.cleanup()
    def scan(self):return IntegrityScanner(self.path,now=self.now).scan()
    def codes(self):return {x["check_id"] for x in self.scan()["findings"]}
    def insert_belief(self,ident="b1",namespace="test",scope_kind="owner",scope_id="leo",predicate="p",status="INFERRED",superseded_by="",valid_until=0,sensitivity="normal",owner=0,authority="extractor"):
        with closing(self.connect()) as c:c.execute("INSERT INTO beliefs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(ident,namespace,scope_kind,scope_id,"subject",predicate,json.dumps("value"),status,.7,authority,1,1,1,valid_until,0,superseded_by,owner,sensitivity,1,"LONG",1));c.commit()


class Phase6DeterministicScenarios(Phase6Fixture):
    def test_scenario_a_legacy_pending_cutover_archives_expired(self):
        with closing(self.connect()) as c:c.execute("INSERT INTO conversations(id,context_kind,context_id,participants_json,attention_state,turn_owner,expected_reply_type,expected_reply_json,topic,origin_event_id,last_event_id,opened_at,last_turn_at,expires_at,status) VALUES('c','owner_local','leo','[]','FOREGROUND','LEO','yes_no','{}','promo','e','e',1,1,2,'WAITING_ON_LEO')");c.commit()
        plan=HygienePlanner(self.path,now=self.now).plan();self.assertTrue(any(x["store"]=="conversations" and x["classification"]=="ARCHIVE" for x in plan["records"]))
        HygienePlanner(self.path,now=self.now).apply_safe(plan)
        with closing(self.connect()) as c:self.assertEqual(c.execute("SELECT status FROM conversations WHERE id='c'").fetchone()[0],"ARCHIVED")

    def test_scenario_b_dirty_belief_db_is_blocking(self):
        self.insert_belief();self.assertIn("belief.provenance",self.codes())

    def test_scenario_c_action_memory_corruption_is_blocking(self):
        with closing(self.connect()) as c:c.execute("INSERT INTO action_ledger VALUES('a','promotion','ivanxi','SUCCEEDED','memories','7',1,2,'{}',1)");c.commit()
        self.assertIn("action.receipt_backing",self.codes())

    def test_scenario_d_game_legacy_conflict_is_isolated_from_v2(self):
        self.insert_belief("gk","game_knowledge","game","ffv","mechanic")
        with closing(self.connect()) as c:
            c.execute("INSERT INTO belief_evidence VALUES('e','gk','event','timeline','event','SUPPORTS',1,1,'test','1','{}','k')")
            c.execute("INSERT INTO game_identities VALUES('ffv','Final Fantasy V','[]','{}','Final Fantasy',1)")
            c.execute("INSERT INTO game_knowledge_facts VALUES('k','ffv','gk','validated','high','safe','', '',1,1)")
            c.commit()
        self.assertEqual(self.scan()["blocking_error_count"],0)

    def test_scenario_e_old_run_leakage_detects_multiple_active_runs(self):
        with closing(self.connect()) as c:
            c.execute("INSERT INTO game_identities VALUES('ffv','FFV','[]','{}','',1)")
            for ident in ('r1','r2'):c.execute("INSERT INTO game_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(ident,'ffv','leo','challenge','{}','ACTIVE',1,1,0,1,'event',1))
            c.commit()
        self.assertIn("game.multiple_active_runs",self.codes())

    def test_scenario_f_social_garbage_duplicate_identity(self):
        with closing(self.connect()) as c:
            for person in ('p1','p2'):c.execute("INSERT INTO people VALUES(?,?,?,'stream_public',1)",(person,1,1))
            c.execute("INSERT INTO person_identities VALUES(?,?,?,?,?,?,?,?,?,?,?,1)",('ip1','p1','twitch','42','p1','p1','[]',1,1,1,'test'))
            with self.assertRaises(sqlite3.IntegrityError):c.execute("INSERT INTO person_identities VALUES(?,?,?,?,?,?,?,?,?,?,?,1)",('ip2','p2','twitch','42','p2','p2','[]',1,1,1,'test'))
            c.commit()

    def test_scenario_g_sensitive_stale_social_is_blocking(self):
        with closing(self.connect()) as c:c.execute("INSERT INTO social_episodes VALUES('s','followup','[]','event','[]','private','[]',1,2,3,'sensitive','bounded','stream_public','test',1)");c.commit()
        self.assertIn("social.sensitive_public",self.codes())

    def test_scenario_h_shared_culture_survives_migration_restart(self):
        with closing(self.connect()) as c:
            c.execute("INSERT INTO people VALUES('p',1,1,'stream_public',1)");c.execute("INSERT INTO shared_culture_items VALUES('i','label','meaning','','[\"p\"]','participants','playful','ACTIVE',1,1,1,0,0,0,1)")
            c.commit()
        migrate(self.path)
        with closing(self.connect()) as c:self.assertEqual(c.execute("SELECT status FROM shared_culture_items WHERE id='i'").fetchone()[0],"ACTIVE")

    def test_scenario_i_owner_preference_conflict_is_detected(self):
        for ident in ('p1','p2'):
            self.insert_belief(ident,'owner_preference','owner','leo','promo.style',owner=1,authority='owner')
            with closing(self.connect()) as c:c.execute("INSERT INTO belief_evidence VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",('e'+ident,ident,'event'+ident,'owner_stt','event'+ident,'SUPPORTS',1,1,'test','1','{}','k'+ident));c.commit()
        self.assertIn("belief.duplicate_current",self.codes())

    def test_scenario_j_schedule_conflict_is_needs_review(self):
        with closing(self.connect()) as c:c.execute("CREATE TABLE schedule_hypotheses(id INTEGER PRIMARY KEY,weekday TEXT,time_window TEXT,canonical_content TEXT,content_key TEXT,stream_format TEXT,source TEXT,confidence REAL,evidence_count INTEGER,consecutive_matches INTEGER,consecutive_misses INTEGER,first_observed_at TEXT,last_observed_at TEXT,status TEXT,superseded_by INTEGER)");c.execute("INSERT INTO schedule_hypotheses VALUES(1,'mon','20','FFV','ffv','game','legacy',.5,1,1,0,'a','b','ACTIVE',NULL)");c.commit()
        plan=HygienePlanner(self.path,now=self.now).plan();self.assertEqual([x for x in plan["records"] if x["store"]=="schedule_hypotheses"][0]["classification"],"NEEDS_REVIEW")

    def test_scenario_k_consolidation_duplicate_prevented_by_schema(self):
        with closing(self.connect()) as c:
            c.execute("INSERT INTO consolidation_runs VALUES('r','s','a','b','v','v','DONE',1,2,'key')")
            values=('d1','r','GAME','x','{}','[]','ACCEPTED','','same','',1);c.execute("INSERT INTO consolidation_deltas VALUES(?,?,?,?,?,?,?,?,?,?,?)",values)
            with self.assertRaises(sqlite3.IntegrityError):c.execute("INSERT INTO consolidation_deltas VALUES(?,?,?,?,?,?,?,?,?,?,?)",('d2',*values[1:]))
            c.commit()

    def test_scenario_l_copied_production_db(self):
        copy=Path(self.tmp.name)/"copy.sqlite3";report=verify_copied_database(self.path,copy)
        self.assertTrue(report["source_untouched"]);self.assertEqual(report["integrity_status"],"PASS")

    def test_scenario_m_full_restart_preserves_cognitive_state(self):
        self.insert_belief();
        with closing(self.connect()) as c:c.execute("INSERT INTO belief_evidence VALUES('e','b1','event','timeline','event','SUPPORTS',1,1,'test','1','{}','k')");c.commit()
        self.assertEqual(IntegrityScanner(self.path,now=self.now).scan()["status"],"PASS")

    def test_scenario_n_external_authority_invariants(self):
        with closing(self.connect()) as c:
            c.execute("CREATE TABLE viewer_promotion_profiles(twitch_user_id TEXT PRIMARY KEY,current_login TEXT,display_name TEXT,known_aliases_json TEXT,auto_promo_mode TEXT,created_by TEXT,created_at TEXT,updated_at TEXT,last_promoted_at TEXT,last_promoted_stream_id TEXT,cooldown_hours REAL,owner_locked INTEGER,active INTEGER)")
            c.execute("INSERT INTO viewer_promotion_profiles VALUES('42','viewer','','[]','always','viewer','a','a',NULL,NULL,0,0,1)")
            c.commit()
        self.assertIn("action.promotion_authority",self.codes())

    def test_scenario_o_production_defaults_are_canonical_with_kill_switch(self):
        defaults=production_defaults(environ={});self.assertTrue(all(value for key,value in defaults.items() if key!='HEBE_CONVERSATION_CONTINUITY_SHADOW'));self.assertFalse(defaults['HEBE_CONVERSATION_CONTINUITY_SHADOW'])
        disabled=production_defaults(environ={'HEBE_COGNITIVE_V2_ENABLED':'false'});self.assertFalse(any(value for key,value in disabled.items() if key!='HEBE_CONVERSATION_CONTINUITY_SHADOW'))

    def test_scenario_p_migration_restart_safety(self):
        first,second=migrate(self.path);self.assertTrue(all(x["already_applied"] for x in first));self.assertTrue(all(x["already_applied"] for x in second));self.assertEqual(len(inventory(before=False)),len(inventory(before=True)))


if __name__=="__main__":unittest.main()
