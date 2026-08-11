from __future__ import annotations
import sqlite3,tempfile,unittest
from pathlib import Path
from app.epistemics.models import BeliefStatus,EvidenceRef,EvidenceRelation,RetrievalRequest
from app.epistemics.repository import BeliefRepository,InvalidBeliefTransition
from app.epistemics.service import BeliefLifecycleService
from app.epistemics.retrieval import MemoryRetrievalCoordinator
from app.replay.migrations import MigrationRunner,belief_v2_migrations
from app.replay.cognitive import CognitiveReplayRunner
from app.replay.scenario import CognitiveReplayScenario

class Phase2Fixture(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/"b.sqlite3";self.connect=lambda:sqlite3.connect(self.path);MigrationRunner(self.connect).migrate(belief_v2_migrations());self.now=1000.;self.repo=BeliefRepository(self.connect);self.service=BeliefLifecycleService(self.repo,now_fn=lambda:self.now);self.retrieve=MemoryRetrievalCoordinator(self.repo,now_fn=lambda:self.now)
 def tearDown(self):self.tmp.cleanup()
 def ev(self,event="e1",relation=EvidenceRelation.SUPPORTS,span=True,extractor="test"):
  return EvidenceRef(event,"owner_stt","record-"+event,relation,1,self.now,extractor,"v1",{"start":0,"end":2,"excerpt":"ok"} if span else {})
 def propose(self,obj=3,event="e1",status=BeliefStatus.INFERRED,scope="owner_local",authority="extractor"):
  return self.service.propose(namespace="test",scope_kind=scope,scope_id="s1",subject_ref="progress",predicate="crystal",object_value=obj,confidence=.74,authority_class=authority,evidence=self.ev(event,extractor=authority),status=status)

class BeliefLifecycleTests(Phase2Fixture):
 def test_status_and_confidence_are_independent(self):
  b=self.propose(status=BeliefStatus.HISTORICAL)
  self.assertEqual(b.epistemic_status,BeliefStatus.HISTORICAL);self.assertEqual(b.confidence,.74)
 def test_owner_correction_supersedes_without_deleting_history(self):
  old=self.propose();self.now+=1;new=self.service.correct(old.id,object_value=2,evidence=self.ev("correct",EvidenceRelation.CORRECTS),authority_class="owner")
  old2=self.repo.get(old.id);self.assertEqual(old2.epistemic_status,BeliefStatus.SUPERSEDED);self.assertEqual(old2.superseded_by,new.id);self.assertTrue(new.owner_confirmed);self.assertEqual(new.epistemic_status,BeliefStatus.KNOWN);self.assertEqual(self.repo.evidence_for(new.id)[0]["relation"],"CORRECTS")
 def test_weak_contradiction_cannot_override_owner_truth(self):
  known=self.service.seed_known(namespace="test",scope_kind="owner_local",scope_id="s1",subject_ref="progress",predicate="crystal",object_value=2,authority_class="owner",evidence=self.ev("owner"))
  rejected=self.service.propose(namespace="test",scope_kind="owner_local",scope_id="s1",subject_ref="progress",predicate="crystal",object_value=3,confidence=.55,authority_class="extractor",evidence=self.ev("weak",extractor="extractor"))
  self.assertIsNone(rejected);self.assertEqual(self.repo.active_for_identity(namespace="test",scope_kind="owner_local",scope_id="s1",subject_ref="progress",predicate="crystal")[0].id,known.id);self.assertTrue(any(row["relation"]=="CONTRADICTS" for row in self.repo.evidence_for(known.id)))
 def test_evidence_and_literal_span_required(self):
  result=self.service.propose(namespace="test",scope_kind="owner_local",scope_id="s1",subject_ref="x",predicate="p",object_value=1,confidence=.5,authority_class="extractor",evidence=self.ev("bad",span=False,extractor="extractor"))
  self.assertIsNone(result);self.assertEqual(self.service.last_transition["reason"],"no_literal_span")
 def test_duplicate_evidence_is_idempotent(self):
  a=self.propose(event="same");b=self.propose(obj=3,event="same");self.assertEqual(a.id,b.id);self.assertEqual(len(self.repo.evidence_for(a.id)),1)
 def test_model_cannot_set_known_owner_truth(self):
  result=self.service.propose(namespace="test",scope_kind="owner_local",scope_id="s1",subject_ref="x",predicate="p",object_value=1,confidence=1,authority_class="model",evidence=self.ev("model",extractor="model"),status=BeliefStatus.KNOWN,owner_confirmed=True)
  self.assertIsNone(result);self.assertEqual(self.service.last_transition["reason"],"invalid_transition")
 def test_explicit_lifecycle_operations_keep_evidence_and_rows(self):
  belief=self.propose(event="initial")
  supported=self.service.support(belief.id,evidence=self.ev("support"));self.assertEqual(len(supported.evidence_ids),2)
  confirmed=self.service.confirm(belief.id,evidence=self.ev("confirm"));self.assertEqual(confirmed.epistemic_status,BeliefStatus.KNOWN);self.assertTrue(confirmed.owner_confirmed)
  archived=self.service.archive_relevance(belief.id,at=1005);self.assertEqual(archived.relevance_until,1005)
  historical=self.service.mark_historical(belief.id,valid_until=1006);self.assertEqual(historical.epistemic_status,BeliefStatus.HISTORICAL);self.assertEqual(len(self.repo.evidence_for(belief.id)),3)
  self.assertGreaterEqual(self.repo.performance()["sqlite_write"]["count"],4)

class RetrievalTests(Phase2Fixture):
 def test_current_historical_and_privacy_ranking(self):
  old=self.propose(obj="FFIX",event="old",status=BeliefStatus.HISTORICAL);new=self.service.seed_known(namespace="test",scope_kind="owner_local",scope_id="s1",subject_ref="monday",predicate="game",object_value="FFV",authority_class="owner",evidence=self.ev("new"))
  current=self.retrieve.retrieve(RetrievalRequest("owner_local","current",subject="monday"));self.assertEqual(current.selected_claims[0]["object"],"FFV")
  hist=self.retrieve.retrieve(RetrievalRequest("owner_local","history",subject="progress",temporal_intent="historical"));self.assertEqual(hist.selected_claims[0]["id"],old.id)
  public=self.retrieve.retrieve(RetrievalRequest("stream_public","public",subject="monday"));self.assertEqual(public.selected_claims,());self.assertEqual(public.rejection_reasons["scope_violation"],1)

class MigrationTests(unittest.TestCase):
 def test_fresh_and_restart_idempotent(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/"db.sqlite3";runner=MigrationRunner(lambda:sqlite3.connect(path));self.assertFalse(runner.migrate(belief_v2_migrations())[0]["already_applied"]);self.assertTrue(runner.migrate(belief_v2_migrations())[0]["already_applied"])
 def test_representative_old_schema_is_additive_and_preserves_unicode(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/"old.sqlite3";conn=sqlite3.connect(path)
   conn.execute("CREATE TABLE live_session_timeline(id INTEGER PRIMARY KEY,event_uid TEXT,raw_text TEXT)")
   conn.execute("INSERT INTO live_session_timeline(event_uid,raw_text) VALUES('old','Mago Blanco â€” cura')")
   conn.execute("CREATE TABLE memory_facts(id INTEGER PRIMARY KEY)");conn.execute("CREATE TABLE memory_chunks(id INTEGER PRIMARY KEY)");conn.commit();conn.close()
   applied=MigrationRunner(lambda:sqlite3.connect(path)).migrate(belief_v2_migrations());self.assertEqual(len(applied[0]["checksum"]),64)
   conn=sqlite3.connect(path);self.assertEqual(conn.execute("SELECT raw_text FROM live_session_timeline WHERE event_uid='old'").fetchone()[0],"Mago Blanco â€” cura")
   self.assertIn("context_kind",{row[1] for row in conn.execute("PRAGMA table_info(live_session_timeline)")});self.assertIn("belief_id",{row[1] for row in conn.execute("PRAGMA table_info(memory_facts)")});conn.close()

class ReplayPhase2IntegrationTests(unittest.TestCase):
 def test_owner_correction_uses_engine_services_canonical_evidence_and_restart(self):
  fixture=Path(__file__).parent/"fixtures"/"cognitive_replay_phase2"/"a_owner_correction.json"
  with tempfile.TemporaryDirectory() as tmp:
   result=CognitiveReplayRunner(workspace_root=Path(tmp),retain_workspace=True).run(CognitiveReplayScenario.load(fixture))
   self.assertEqual(result.status,"VERIFIED");self.assertEqual(result.restart_count,1)
   conn=sqlite3.connect(result.database["path"]);rows=conn.execute("SELECT event_uid,raw_text,source_record_type,authority FROM live_session_timeline WHERE event_type='belief_evidence' ORDER BY id").fetchall();conn.close()
   self.assertEqual([row[0] for row in rows],["old","correction"]);self.assertEqual(rows[1][1],"No, es el segundo.");self.assertEqual(rows[1][2],"cognitive_replay_event");self.assertEqual(rows[1][3],"owner")

if __name__=="__main__":unittest.main()
