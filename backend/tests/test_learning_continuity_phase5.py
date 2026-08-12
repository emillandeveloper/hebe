from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.continuity.repository import OpenThreadRepository
from app.epistemics.models import EvidenceRef
from app.epistemics.repository import BeliefRepository
from app.learning_v2.repository import LearningRepository
from app.learning_v2.service import (
    ContinuityContextBuilder, HebeSelfModel, HistoricalActionLedger, LeoLanguageModel,
    OwnerProceduralPreferences, SceneConsequenceReducer, SessionConsolidator,
    StableHebeCore, TemporalRelevanceService,
)
from app.replay.migrations import (
    MigrationRunner, belief_v2_migrations, conversation_continuity_migrations,
    learning_v2_migrations, social_world_v2_migrations,
)
from app.social_world_v2.repository import SocialWorldRepository


class Phase5Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/"phase5.sqlite3";self.connect=lambda:sqlite3.connect(self.path)
        runner=MigrationRunner(self.connect)
        for migrations in (belief_v2_migrations(),conversation_continuity_migrations(),social_world_v2_migrations(),learning_v2_migrations()):runner.migrate(migrations)
        self.now=1000.;self.beliefs=BeliefRepository(self.connect);self.learning=LearningRepository(self.connect);self.threads=OpenThreadRepository(self.connect);self.social=SocialWorldRepository(self.connect)
        self.core=StableHebeCore();self.self_model=HebeSelfModel(self.beliefs,self.learning,now_fn=lambda:self.now);self.preferences=OwnerProceduralPreferences(self.beliefs,self.learning,now_fn=lambda:self.now);self.language=LeoLanguageModel(self.beliefs,self.learning,now_fn=lambda:self.now);self.ledger=HistoricalActionLedger(self.learning,now_fn=lambda:self.now);self.scene=SceneConsequenceReducer(self.learning,self.preferences,now_fn=lambda:self.now);self.consolidator=SessionConsolidator(self.learning,self.core,self.self_model,self.preferences,self.language,now_fn=lambda:self.now)
    def tearDown(self):self.tmp.cleanup()
    def evidence(self,event):return EvidenceRef(event,"live_session_timeline",event,observed_at=self.now)


class LearningModelsTests(Phase5Fixture):
    def test_stable_core_is_versioned_and_mutation_is_rejected(self):
        version=self.core.version
        result=self.consolidator.consolidate(session_id="s",start_event="a",end_event="b",candidates=[{"domain":"STABLE_CORE","delta_type":"change_owner","payload":{"owner":"viewer"},"evidence_ids":["e1"]}])
        self.assertEqual(result["rejected_deltas"],1);self.assertEqual(self.core.version,version);self.assertEqual(self.beliefs.list(namespace="hebe_self"),[])

    def test_stable_core_nested_mutation_and_inferred_owner_preference_fail_closed(self):
        candidates=[{"domain":"THREAD","delta_type":"changed","payload":{"proposal":{"predicate":"disable_owner_authority"}},"evidence_ids":["e1"]},{"domain":"OWNER_PREFERENCE","delta_type":"added","payload":{"predicate":"promo.style","value":"short"},"evidence_ids":["e2"]}]
        result=self.consolidator.consolidate(session_id="adversarial",start_event="a",end_event="b",candidates=candidates)
        self.assertEqual(result["rejected_deltas"],2);self.assertFalse(self.beliefs.list(namespace="owner_preference"))

    def test_hebe_opinion_evolves_with_history_and_stays_namespaced(self):
        old,_=self.self_model.learn(subject="p5.x",predicate="opinion.character",value="distrust",evidence=self.evidence("e1"));self.now+=10
        new,_=self.self_model.learn(subject="p5.x",predicate="opinion.character",value="warming_up",evidence=self.evidence("e2"))
        self.assertEqual(self.beliefs.get(old.id).epistemic_status.value,"SUPERSEDED");self.assertEqual(self.beliefs.get(old.id).superseded_by,new.id);self.assertFalse(self.beliefs.list(namespace="game_knowledge"))

    def test_owner_preference_is_authoritative_persistent_and_renderable(self):
        result,_=self.preferences.learn(subject="leo",predicate="raid_ack.omit_viewer_count",value=True,evidence=self.evidence("e1"))
        restarted=OwnerProceduralPreferences(BeliefRepository(self.connect),LearningRepository(self.connect),now_fn=lambda:self.now)
        self.assertTrue(result.owner_confirmed);self.assertEqual(result.retention_policy,"LONG");self.assertTrue(restarted.rendering_policy("raid_ack")["omit_viewer_count"])

    def test_leo_language_requires_repetition_and_does_not_touch_hebe_self(self):
        self.assertIsNone(self.language.observe(predicate="lexical.confirmation",value="sip",event_id="e1",evidence=self.evidence("e1"))[0])
        learned,_=self.language.observe(predicate="lexical.confirmation",value="sip",event_id="e2",evidence=self.evidence("e2"))
        self.assertEqual(self.language.interpretation_aliases()["sip"],"affirmative");self.assertEqual(learned.namespace,"leo_language");self.assertFalse(self.beliefs.list(namespace="hebe_self"))

    def test_cross_domain_contamination_is_rejected(self):
        candidates=[
          {"domain":"HEBE_SELF","delta_type":"learn","payload":{"predicate":"opinion.character","value":"likes","source_domain":"leo_opinion"},"evidence_ids":["a"]},
          {"domain":"GAME","delta_type":"learn","payload":{"source_domain":"hebe_opinion"},"evidence_ids":["b"]},
          {"domain":"SOCIAL","delta_type":"learn","payload":{"objective_truth":True},"evidence_ids":["c"]},
        ]
        result=self.consolidator.consolidate(session_id="cross",start_event="a",end_event="z",candidates=candidates)
        self.assertEqual(result["rejected_deltas"],3)


class ConsolidationTests(Phase5Fixture):
    def test_candidate_provider_is_domain_keyed_and_only_proposes(self):
        calls=[]
        def provider(**kwargs):calls.append((kwargs["domain"],kwargs["schema_version"],kwargs["session_id"]));return []
        service=SessionConsolidator(self.learning,self.core,self.self_model,self.preferences,self.language,now_fn=lambda:self.now,candidate_provider=provider)
        result=service.consolidate(session_id="provider",start_event="a",end_event="b")
        self.assertEqual(result["accepted_deltas"],0);self.assertEqual({x[0] for x in calls},service.DOMAINS-{"STABLE_CORE"});self.assertTrue(all(x[1:]==(1,"provider") for x in calls))

    def test_no_change_and_same_watermark_are_successful_and_idempotent(self):
        first=self.consolidator.consolidate(session_id="empty",start_event="a",end_event="b",candidates=[]);second=self.consolidator.consolidate(session_id="empty",start_event="a",end_event="b",candidates=[])
        self.assertEqual(first["accepted_deltas"],0);self.assertEqual(second["accepted_deltas"],0);self.assertEqual(len(self.learning.rows("consolidation_runs")),1)

    def test_already_committed_game_delta_is_audit_only(self):
        result=self.consolidator.consolidate(session_id="game",start_event="a",end_event="b",candidates=[{"domain":"GAME","delta_type":"already_committed","payload":{"committed_object_ref":"belief_run_white_mage"},"evidence_ids":["run-e1"]}])
        self.assertEqual(result["accepted_deltas"],1);self.assertEqual(self.learning.rows("consolidation_deltas")[0]["committed_object_ref"],"belief_run_white_mage");self.assertFalse(self.beliefs.list(namespace="game_run"))

    def test_owner_preference_immediate_write_is_not_duplicated_by_consolidation(self):
        belief,_=self.preferences.learn(subject="leo",predicate="promo.style",value="short",evidence=self.evidence("e1"))
        self.consolidator.consolidate(session_id="pref",start_event="a",end_event="b",candidates=[{"domain":"OWNER_PREFERENCE","delta_type":"already_committed","payload":{"predicate":"promo.style","value":"short","explicit_owner_feedback":True},"evidence_ids":["e1"]}])
        self.assertEqual(len(self.beliefs.list(namespace="owner_preference")),1);self.assertEqual(self.preferences.current("leo")[0].id,belief.id)


class TemporalActionSceneTests(Phase5Fixture):
    def test_action_ledger_preserves_success_failure_unknown_truth(self):
        for status,target in (("SUCCEEDED","a"),("FAILED","b"),("UNKNOWN","c")):self.ledger.project(source_store="test",source_record_id=target,action_type="desktop",target=target,status=status)
        self.assertTrue(self.ledger.validate_claim(action_type="desktop",target="a").allowed);self.assertFalse(self.ledger.validate_claim(action_type="desktop",target="b").allowed);self.assertEqual(self.ledger.validate_claim(action_type="desktop",target="c").claim_strength,"uncertain")

    def test_outgoing_and_incoming_raids_have_distinct_consequences(self):
        self.preferences.learn(subject="leo",predicate="raid_ack.omit_viewer_count",value=True,evidence=self.evidence("p"))
        outgoing=self.scene.outgoing_raid(event_id="r1",destination="ivanxi",receipt_status="SUCCEEDED",viewer_count=200);incoming=self.scene.incoming_raid(event_id="r2",source="nuria",viewer_count=40)
        self.assertTrue(outgoing["stream_ending"]);self.assertTrue(outgoing["farewell_opportunity"]["presence_gated"]);self.assertTrue(outgoing["rendering_policy"]["omit_viewer_count"]);self.assertFalse(incoming["stream_ending"])

    def test_temporal_service_expires_bounded_belief_and_audits(self):
        from app.epistemics.models import Belief
        belief=Belief("b","social","person","p","p","interest.topic","x",__import__('app.epistemics.models',fromlist=['BeliefStatus']).BeliefStatus.INFERRED,.6,"extractor",self.now,self.now,self.now,0,self.now+10,"",False,"normal",1,"SHORT",1)
        self.beliefs.propose(belief,self.evidence("e"));self.now+=11
        temporal=TemporalRelevanceService(self.connect,self.beliefs,self.threads,self.social,self.learning,now_fn=lambda:self.now);actions=temporal.maintain()
        self.assertEqual(self.beliefs.get("b").epistemic_status.value,"HISTORICAL");self.assertEqual(actions[0]["reason"],"relevance_elapsed")

    def test_context_keeps_claim_types_and_bounded_manifest(self):
        self.self_model.learn(subject="ffv",predicate="preference.game",value="likes",evidence=self.evidence("s"));self.preferences.learn(subject="leo",predicate="game_advice.preference",value="only_when_confident",evidence=self.evidence("p"))
        context=ContinuityContextBuilder(self.self_model,self.preferences,self.language,self.ledger,self.scene).build(purpose="reply")
        self.assertEqual(context["self"][0]["claim_type"],"self_opinion");self.assertEqual(context["owner_preferences"][0]["claim_type"],"owner_preference");self.assertLessEqual(len(context["provenance_manifest"]),15)


class MigrationTests(unittest.TestCase):
    def test_phase5_migration_is_additive_checksummed_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"old.sqlite3";connect=lambda:sqlite3.connect(path);runner=MigrationRunner(connect);runner.migrate(belief_v2_migrations());first=runner.migrate(learning_v2_migrations())[0];second=runner.migrate(learning_v2_migrations())[0]
            self.assertFalse(first["already_applied"]);self.assertTrue(second["already_applied"]);self.assertEqual(len(first["checksum"]),64)


if __name__ == "__main__":unittest.main()
