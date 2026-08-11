from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.epistemics.models import BeliefStatus, EvidenceRef, EvidenceRelation
from app.epistemics.repository import BeliefRepository
from app.epistemics.retrieval import MemoryRetrievalCoordinator
from app.epistemics.service import BeliefLifecycleService
from app.game_context_v2.context import GameContextResolver
from app.game_context_v2.models import GameRunStatus
from app.game_context_v2.repository import GameV2Repository
from app.game_context_v2.service import GameKnowledgeService, GameRunService
from app.replay.migrations import MigrationRunner, belief_v2_migrations, game_context_v2_migrations
from app.replay.cognitive import CognitiveReplayRunner
from app.replay.scenario import CognitiveReplayScenario


class Provider:
    def __init__(self):self.calls=[]
    def plan_search(self,**kwargs):return SimpleNamespace(**kwargs,query="fixture query",cache_key="fixture")
    def research(self,plan,**kwargs):
        self.calls.append((plan,kwargs));return [SimpleNamespace(claim="Restores HP",source_location="https://example.test/item",exact_supporting_excerpt_internal="The item restores HP.",confidence=.9,source_type="primary",spoiler_classification="safe_general_mechanic")]


class Phase3Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/"game.sqlite3";self.connect=lambda:sqlite3.connect(self.path)
        runner=MigrationRunner(self.connect);runner.migrate(belief_v2_migrations());runner.migrate(game_context_v2_migrations())
        self.now=1000.;self.belief_repo=BeliefRepository(self.connect);self.lifecycle=BeliefLifecycleService(self.belief_repo,now_fn=lambda:self.now)
        self.repo=GameV2Repository(self.connect);self.runs=GameRunService(self.repo,self.lifecycle,now_fn=lambda:self.now);self.knowledge=GameKnowledgeService(self.repo,self.lifecycle,now_fn=lambda:self.now)
    def tearDown(self):self.tmp.cleanup()
    def evidence(self,event,text="evidence",relation=EvidenceRelation.SUPPORTS):
        return EvidenceRef(event,"live_session_timeline",event,relation,1,self.now,"test","v1",{"start":0,"end":len(text),"excerpt":text})


class GameRunTests(Phase3Fixture):
    def test_alias_identity_and_same_run_resume_across_service_restart(self):
        first=self.runs.resolve(game="FFV",stream_session_id="s1",source_event_id="start1",run_kind="crystal_roulette")
        self.now+=86400
        restarted=GameRunService(GameV2Repository(self.connect),BeliefLifecycleService(BeliefRepository(self.connect),now_fn=lambda:self.now),now_fn=lambda:self.now)
        second=restarted.resolve(game="Final Fantasy 5",stream_session_id="s2",source_event_id="start2",explicit_continue=True)
        self.assertEqual(first.game_identity.game_id,"final_fantasy_v");self.assertEqual(first.active_run.id,second.active_run.id);self.assertEqual(len(restarted.repository.session_links(first.active_run.id)),2)

    def test_explicit_new_same_game_isolated(self):
        a=self.runs.resolve(game="FFV",stream_session_id="a",source_event_id="a",run_kind="challenge").active_run
        self.runs.record_fact(a.id,subject_ref="job",predicate="rolled",object_value="white_mage",evidence=self.evidence("wm"),owner_confirmed=True)
        self.runs.finish(a.id,status=GameRunStatus.COMPLETED,event_id="done");self.now+=1
        b=self.runs.resolve(game="FFV",stream_session_id="b",source_event_id="b",run_kind="challenge",explicit_new=True).active_run
        self.assertNotEqual(a.id,b.id);self.assertEqual(self.runs.facts(b.id),[]);self.assertEqual(self.runs.facts(a.id,historical=True)[0]["object"],"white_mage")

    def test_owner_correction_and_weak_inference_precedence(self):
        run=self.runs.resolve(game="FFV",stream_session_id="s",source_event_id="start").active_run
        old=self.runs.record_fact(run.id,subject_ref="progress",predicate="current_crystal",object_value=3,evidence=self.evidence("infer","third"),confidence=.74)
        new=self.runs.correct_fact(old.id,object_value=2,evidence=self.evidence("correct","No, second",EvidenceRelation.CORRECTS))
        weak=self.runs.record_fact(run.id,subject_ref="progress",predicate="current_crystal",object_value=3,evidence=self.evidence("weak","third"),confidence=.55)
        self.assertIsNone(weak);self.assertEqual(self.belief_repo.get(old.id).epistemic_status,BeliefStatus.SUPERSEDED);self.assertTrue(new.owner_confirmed);self.assertEqual(self.runs.facts(run.id)[0]["object"],2)

    def test_unsupported_semantic_inference_is_rejected(self):
        run=self.runs.resolve(game="FFV",stream_session_id="s",source_event_id="start").active_run
        result=self.runs.record_fact(run.id,subject_ref="leo",predicate="navigation_state",object_value="lost",evidence=self.evidence("raw","Aquí es donde estuvimos encerrados."),confidence=.4,entailment_valid=False)
        self.assertIsNone(result);self.assertEqual(self.lifecycle.last_transition["reason"],"unsupported_run_inference")


class GameKnowledgeTests(Phase3Fixture):
    def test_scoped_memory_and_scene_precede_research(self):
        identity=self.repo.resolve_identity("FFV");provider=Provider()
        self.lifecycle.propose(namespace="memory",scope_kind="game",scope_id=identity.game_id,subject_ref="potion",predicate="effect",object_value="restores_hp",authority_class="domain_validator",evidence=self.evidence("memory"),confidence=.95,status=BeliefStatus.INFERRED)
        retrieval=MemoryRetrievalCoordinator(self.belief_repo,now_fn=lambda:self.now)
        resolver=GameContextResolver(self.repo,self.runs,self.knowledge,research_service=provider,memory_retrieval=retrieval,now_fn=lambda:self.now)
        scene={"id":"scene_1","namespace":"scene","scope_kind":"stream_session","scope_id":"s1","epistemic_status":"KNOWN","confidence":1.0,"authority_class":"deterministic","evidence_ids":["scene_event"]}
        context=resolver.build(game="FFV",purpose="game_fact",subject_ref="potion",predicate="effect",question_type="item_effect",allow_research=True,scene_assertions=(scene,))
        self.assertEqual(context.research_status,"memory_available");self.assertEqual(context.rag_context[0]["object"],"restores_hp");self.assertEqual(context.scene_assertions[0]["id"],"scene_1");self.assertEqual(len(provider.calls),0)

    def test_general_knowledge_is_separate_from_run_scope(self):
        identity=self.repo.resolve_identity("Persona 5 Royal")
        claim=self.knowledge.add_validated(game_id=identity.game_id,subject_ref="maruki",predicate="role",object_value="confidant",confidence=.95,evidence=self.evidence("known"),source_type="curated",source_quality="validated")
        run=self.runs.resolve(game="P5R",stream_session_id="p1",source_event_id="p1").active_run
        self.runs.record_fact(run.id,subject_ref="maruki",predicate="rank",object_value=6,evidence=self.evidence("rank"),owner_confirmed=True)
        general,_=self.knowledge.find(identity.game_id,subject_ref="maruki",predicate="role")
        self.assertEqual(general[0]["object"],"confidant");self.assertEqual(claim.scope_kind,"game");self.assertEqual(self.runs.facts(run.id)[0]["scope_kind"],"game_run")

    def test_memory_first_research_once_then_restart_hit(self):
        provider=Provider();resolver=GameContextResolver(self.repo,self.runs,self.knowledge,research_service=provider,now_fn=lambda:self.now)
        first=resolver.build(game="FFV",purpose="game_fact",subject_ref="potion",predicate="effect",question_type="item_effect",query_intent="potion effect",allow_research=True,event_id="q1")
        self.assertEqual(first.research_status,"research_completed");self.assertEqual(len(provider.calls),1)
        restarted=GameContextResolver(GameV2Repository(self.connect),GameRunService(GameV2Repository(self.connect),self.lifecycle,now_fn=lambda:self.now),GameKnowledgeService(GameV2Repository(self.connect),self.lifecycle,now_fn=lambda:self.now),research_service=provider,now_fn=lambda:self.now)
        second=restarted.build(game="Final Fantasy V",purpose="game_fact",subject_ref="potion",predicate="effect",question_type="item_effect",query_intent="potion effect",allow_research=True,event_id="q2")
        self.assertEqual(second.research_status,"knowledge_available");self.assertEqual(len(provider.calls),1)

    def test_known_fact_prevents_research_and_spoiler_is_filtered(self):
        identity=self.repo.resolve_identity("P5R");provider=Provider()
        self.knowledge.add_validated(game_id=identity.game_id,subject_ref="combat",predicate="system",object_value="turn_based",confidence=.9,evidence=self.evidence("safe"),source_type="curated",source_quality="validated")
        self.knowledge.add_validated(game_id=identity.game_id,subject_ref="ending",predicate="identity",object_value="forbidden",confidence=.9,evidence=self.evidence("spoiler"),source_type="curated",source_quality="validated",spoiler_class="ending_spoiler")
        resolver=GameContextResolver(self.repo,self.runs,self.knowledge,research_service=provider,now_fn=lambda:self.now)
        known=resolver.build(game="P5R",purpose="game_fact",subject_ref="combat",predicate="system",allow_research=True,question_type="system_rule");blocked=resolver.build(game="P5R",purpose="game_advice",subject_ref="ending",predicate="identity",allow_research=True,question_type="character_identity",spoiler_ceiling="strict")
        self.assertEqual(len(provider.calls),0);self.assertEqual(known.knowledge_claims[0]["object"],"turn_based");self.assertTrue(any(item["rejection_reason"]=="spoiler_blocked" for item in blocked.rejected_knowledge));self.assertFalse(blocked.advice_allowed);self.assertTrue(blocked.reaction_allowed)


class MigrationTests(unittest.TestCase):
    def test_phase2_database_upgrade_and_restart_are_additive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"old.sqlite3";connect=lambda:sqlite3.connect(path);runner=MigrationRunner(connect);runner.migrate(belief_v2_migrations())
            conn=connect();conn.execute("CREATE TABLE game_sessions(id INTEGER PRIMARY KEY,title TEXT)");conn.execute("INSERT INTO game_sessions(title) VALUES('Partida ñ')");conn.commit();conn.close()
            first=runner.migrate(game_context_v2_migrations());second=runner.migrate(game_context_v2_migrations())
            conn=connect();self.assertEqual(conn.execute("SELECT title FROM game_sessions").fetchone()[0],"Partida ñ");self.assertIn("game_run_id",{row[1] for row in conn.execute("PRAGMA table_info(game_sessions)")});conn.close()
            self.assertFalse(first[0]["already_applied"]);self.assertTrue(second[0]["already_applied"]);self.assertEqual(len(first[0]["checksum"]),64)


class ReplayIntegrationTests(unittest.TestCase):
    def test_durable_run_scenario_uses_engine_restart_and_canonical_services(self):
        fixture=Path(__file__).parent/"fixtures"/"cognitive_replay_phase3"/"a_ffv_white_mage_durable.json"
        with tempfile.TemporaryDirectory() as tmp:
            result=CognitiveReplayRunner(workspace_root=Path(tmp),retain_workspace=True).run(CognitiveReplayScenario.load(fixture))
            self.assertEqual(result.status,"VERIFIED");self.assertEqual(result.restart_count,1)
            state=result.final_state["game_state"];self.assertEqual(len(state["runs"]),1);self.assertEqual(len(state["session_links"]),2);self.assertEqual(state["context"]["run_facts"][0]["object"],"white_mage")


if __name__=="__main__":unittest.main()
