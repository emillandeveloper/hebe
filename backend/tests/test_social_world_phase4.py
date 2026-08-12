from __future__ import annotations
import sqlite3,tempfile,unittest
from pathlib import Path
from app.continuity.repository import OpenThreadRepository
from app.epistemics.models import BeliefStatus,EvidenceRef,EvidenceRelation
from app.epistemics.repository import BeliefRepository
from app.epistemics.service import BeliefLifecycleService
from app.replay.cognitive import CognitiveReplayRunner
from app.replay.migrations import MigrationRunner,belief_v2_migrations,conversation_continuity_migrations,game_context_v2_migrations,social_world_v2_migrations
from app.replay.scenario import CognitiveReplayScenario
from app.social_world_v2 import SocialWorldRepository,SocialWorldService

class Phase4Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/"social.sqlite3";self.connect=lambda:sqlite3.connect(self.path);runner=MigrationRunner(self.connect)
        for migrations in (conversation_continuity_migrations(),belief_v2_migrations(),game_context_v2_migrations(),social_world_v2_migrations()):runner.migrate(migrations)
        self.now=1000.;self.belief_repo=BeliefRepository(self.connect);self.beliefs=BeliefLifecycleService(self.belief_repo,now_fn=lambda:self.now);self.repo=SocialWorldRepository(self.connect);self.threads=OpenThreadRepository(self.connect);self.social=SocialWorldService(self.repo,self.beliefs,self.threads,now_fn=lambda:self.now)
    def tearDown(self):self.tmp.cleanup()
    def evidence(self,event,text="evidence",relation=EvidenceRelation.SUPPORTS):return EvidenceRef(event,"live_session_timeline",event,relation,1,self.now,"test","v1",{"excerpt":text})

class IdentityTests(Phase4Fixture):
    def test_stable_id_rename_and_similar_names_isolate(self):
        a,old=self.social.resolve_person(platform_user_id="42",login="ivanxi",stream_session_id="s1");self.now+=1;a2,new=self.social.resolve_person(platform_user_id="42",login="ivanxi_new",stream_session_id="s2");b,_=self.social.resolve_person(platform_user_id="99",login="ivanxii",stream_session_id="s2")
        self.assertEqual(a.person_id,a2.person_id);self.assertNotEqual(a.person_id,b.person_id);self.assertIn("ivanxi",new.aliases);self.assertEqual(len(self.repo.people()),2)
    def test_familiarity_uses_distinct_sessions(self):
        person,_=self.social.resolve_person(platform_user_id="42",login="regular",stream_session_id="s1")
        for _ in range(20):self.social.resolve_person(platform_user_id="42",login="regular",stream_session_id="s1")
        self.social.resolve_person(platform_user_id="42",login="regular",stream_session_id="s2");self.assertEqual(self.repo.familiarity(person.person_id)["distinct_sessions"],2)

class SocialContinuityTests(Phase4Fixture):
    def test_low_salience_and_sensitive_persistence_fail_conservative(self):
        person,_=self.social.resolve_person(platform_user_id="1",login="one")
        self.assertIsNone(self.social.record_episode(episode_type="chat",participant_ids=(person.person_id,),origin_event_id="low",summary="hola",salience_reason="chat"))
        self.assertIsNone(self.social.record_episode(episode_type="update",participant_ids=(person.person_id,),origin_event_id="sensitive",summary="diagnosis",salience_reason="personal_update",sensitivity="medical_diagnosis"));self.assertEqual(self.repo.episodes(),[])
    def test_temporary_followup_expires_but_episode_remains(self):
        person,_=self.social.resolve_person(platform_user_id="1",login="one");self.social.record_episode(episode_type="personal_update",participant_ids=(person.person_id,),origin_event_id="ill",summary="sick today",salience_reason="explicit_personal_update",relevance_seconds=86400,sensitivity="low");self.social.open_social_thread(person.person_id,thread_type="wellbeing_followup",subject_ref="illness",summary="check recovery",origin_event_id="ill",relevance_seconds=86400)
        self.now+=90000;self.assertEqual(self.social.expire_social_threads(),1);self.assertEqual(self.threads.list_open(scope_kind="person",scope_id=person.person_id),[]);self.assertEqual(len(self.repo.episodes(person.person_id)),1)
    def test_social_hypothesis_remains_inferred(self):
        person,_=self.social.resolve_person(platform_user_id="1",login="one");b=self.social.propose_hypothesis(person.person_id,predicate="interest.book_topic",object_value="book",confidence=.72,evidence=self.evidence("book","my book"));self.assertEqual(b.epistemic_status,BeliefStatus.INFERRED);self.assertFalse(b.owner_confirmed)
    def test_public_privacy_rejects_private_episode(self):
        person,_=self.social.resolve_person(platform_user_id="1",login="one");self.social.record_episode(episode_type="project",participant_ids=(person.person_id,),origin_event_id="private",summary="private project",salience_reason="explicit_update",retrieval_scope="private_owner")
        context=self.social.retrieve_social_context(person.person_id,retrieval_scope="stream_public");self.assertEqual(context.recent_episodes,());self.assertEqual(context.reasons["privacy_scope"],1)

class CultureTests(Phase4Fixture):
    def test_candidate_activation_cooldown_and_owner_retirement(self):
        person,_=self.social.resolve_person(platform_user_id="1",login="one");item=self.social.create_culture_candidate(label="peach",meaning="peach callback",participant_ids=(person.person_id,),origin_episode_id="e",event_id="origin");self.assertEqual(item["status"],"CANDIDATE")
        self.social.reinforce_culture(item["id"],event_id="p1");active=self.social.reinforce_culture(item["id"],event_id="p2");self.assertEqual(active["status"],"ACTIVE");self.assertIsNotNone(self.social.use_culture(item["id"],event_id="use",cooldown_seconds=100));selected,rejected=self.social.select_culture(person.person_id,topic="peach");self.assertEqual(selected,[]);self.assertEqual(rejected[0]["rejection_reason"],"cooldown")
        retired=self.social.reinforce_culture(item["id"],event_id="no",reaction="owner_reject",authority="owner");self.assertEqual(retired["status"],"RETIRED")
    def test_context_required_for_callback(self):
        person,_=self.social.resolve_person(platform_user_id="1",login="one");item=self.social.create_culture_candidate(label="peach",meaning="peach callback",participant_ids=(person.person_id,),origin_episode_id="e",event_id="origin",owner_confirmed=True)
        self.assertEqual(self.social.select_culture(person.person_id,topic="boss")[0],[]);self.assertEqual(self.social.select_culture(person.person_id,topic="peach")[0][0]["id"],item["id"])

class MigrationAndReplayTests(unittest.TestCase):
    def test_phase3_upgrade_is_additive_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"old.sqlite3";connect=lambda:sqlite3.connect(path);runner=MigrationRunner(connect)
            for migrations in (conversation_continuity_migrations(),belief_v2_migrations(),game_context_v2_migrations()):runner.migrate(migrations)
            c=connect();c.execute("INSERT INTO game_identities(game_id,canonical_name,aliases_json,platform_ids_json) VALUES('x','X','[]','{}')");c.commit();c.close();first=runner.migrate(social_world_v2_migrations());second=runner.migrate(social_world_v2_migrations());c=connect();self.assertEqual(c.execute("SELECT canonical_name FROM game_identities").fetchone()[0],"X");c.close();self.assertFalse(first[0]["already_applied"]);self.assertTrue(second[0]["already_applied"])
    def test_rename_replay_uses_real_restart(self):
        fixture=Path(__file__).parent/"fixtures"/"cognitive_replay_phase4"/"a_stable_identity_rename.json"
        with tempfile.TemporaryDirectory() as tmp:
            result=CognitiveReplayRunner(workspace_root=Path(tmp),retain_workspace=True).run(CognitiveReplayScenario.load(fixture));self.assertEqual(result.status,"VERIFIED");self.assertEqual(result.restart_count,1);self.assertEqual(len(result.final_state["social_state"]["people"]),1)

if __name__=="__main__":unittest.main()
