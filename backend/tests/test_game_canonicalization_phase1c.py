from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest

from app.epistemics.models import EvidenceRef
from app.epistemics.repository import BeliefRepository
from app.epistemics.service import BeliefLifecycleService
from app.game_context_v2.context import GameContextResolver
from app.game_context_v2.migration import (
    game_knowledge_canonicalization_migrations,
    game_run_state_canonicalization_migrations,
)
from app.game_context_v2.repository import GameV2Repository
from app.game_context_v2.service import GameKnowledgeService, GameRunService
from app.replay.migrations import MigrationRunner, belief_v2_migrations, game_context_v2_migrations
from app.stream.game_intelligence import GameIntelligenceStore, GameResearchService


class GameCanonicalizationPhase1CTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "game.sqlite3")
        conn = self.connect()
        conn.executescript(
            """
            CREATE TABLE game_progress_states(
              game_id TEXT NOT NULL,stream_session_id TEXT NOT NULL,state_json TEXT NOT NULL,
              updated_at TEXT NOT NULL,PRIMARY KEY(game_id,stream_session_id));
            CREATE TABLE game_dossiers(
              game_id TEXT PRIMARY KEY,canonical_title TEXT NOT NULL,dossier_json TEXT NOT NULL,
              dossier_version INTEGER NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
            CREATE TABLE game_sessions(
              id INTEGER PRIMARY KEY AUTOINCREMENT,game TEXT NOT NULL,game_key TEXT NOT NULL,
              updated_at TEXT NOT NULL);
            """
        )
        conn.commit()
        conn.close()
        self.runner = MigrationRunner(self.connect)
        self.runner.migrate(belief_v2_migrations())
        self.runner.migrate(game_context_v2_migrations())
        self.repo = GameV2Repository(self.connect)
        self.lifecycle = BeliefLifecycleService(BeliefRepository(self.connect), now_fn=lambda: 1000.0)
        self.runs = GameRunService(self.repo, self.lifecycle, now_fn=lambda: 1000.0)
        self.knowledge = GameKnowledgeService(self.repo, self.lifecycle, now_fn=lambda: 1000.0)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def evidence(event_id: str = "owner:1", text: str = "Leo confirms the run state") -> EvidenceRef:
        return EvidenceRef(
            source_event_id=event_id,
            source_record_type="owner_stt",
            source_record_id=event_id,
            observed_at=1000.0,
            literal_span={"excerpt": text},
        )

    def resolve(self, game: str = "P5R", session: str = "s1", **kwargs):
        return self.runs.resolve(
            game=game,
            stream_session_id=session,
            source_event_id=f"stream_session:{session}",
            run_kind=kwargs.pop("run_kind", "first_playthrough"),
            **kwargs,
        )

    def test_a_new_run_has_one_canonical_identity(self) -> None:
        first = self.resolve("P5R")
        second = self.resolve("Persona 5 Royal", session="s2")
        self.assertEqual(first.game_identity.game_id, "persona_5_royal")
        self.assertEqual(first.active_run.id, second.active_run.id)
        conn = self.connect()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM game_identities").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM game_runs").fetchone()[0], 1)
        conn.close()

    def test_b_current_game_update_is_one_canonical_write(self) -> None:
        run = self.resolve().active_run
        self.assertIsNotNone(run)
        self.runs.resolve(game="P5R", stream_session_id="s1", source_event_id="stream_session:s1")
        conn = self.connect()
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM game_runs").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM game_progress_states").fetchone()[0], 0)
        conn.close()

    def test_c_valid_party_provenance_updates_canonical_state(self) -> None:
        run = self.resolve().active_run
        result = self.runs.update_state(
            run.id,
            updates={"party_members": ["Joker", "Morgana"], "current_character": "Joker"},
            provenance="leo_clarification",
            confidence=0.95,
            evidence=self.evidence(),
        )
        self.assertEqual(result["rejected"], {})
        state = self.runs.state(run.id)
        self.assertEqual(state["party_members"], ["Joker", "Morgana"])
        self.assertEqual(state["current_character"], "Joker")

    def test_d_weak_party_provenance_is_rejected(self) -> None:
        run = self.resolve().active_run
        result = self.runs.update_state(
            run.id,
            updates={"party_members": ["A sentence fragment from STT"]},
            provenance="inferred",
            confidence=0.4,
            evidence=self.evidence(),
        )
        self.assertEqual(result["accepted"], {})
        self.assertEqual(result["rejected"]["party_members"], "low_confidence_or_missing_provenance")
        self.assertEqual(self.runs.state(run.id)["party_members"], [])

    def test_e_progress_update_persists_once_and_never_legacy(self) -> None:
        run = self.resolve().active_run
        for event_id in ("owner:progress:1", "owner:progress:2"):
            self.runs.update_state(
                run.id,
                updates={"last_confirmed_progress": "second crystal"},
                provenance="leo_clarification",
                confidence=0.9,
                evidence=self.evidence(event_id, "We are at the second crystal"),
            )
        conn = self.connect()
        active = conn.execute(
            "SELECT COUNT(*) FROM beliefs WHERE namespace='game_run' AND scope_id=? AND predicate='last_confirmed_progress' AND superseded_by=''",
            (run.id,),
        ).fetchone()[0]
        self.assertEqual(active, 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM game_progress_states").fetchone()[0], 0)
        conn.close()

    def test_f_location_and_objective_are_run_state_not_game_knowledge(self) -> None:
        run = self.resolve().active_run
        self.runs.update_state(
            run.id,
            updates={"current_location": "Kamoshida Palace", "current_objective": "reach the treasure"},
            provenance="leo_clarification",
            confidence=0.9,
            evidence=self.evidence(),
        )
        conn = self.connect()
        namespaces = {row[0] for row in conn.execute("SELECT DISTINCT namespace FROM beliefs")}
        conn.close()
        self.assertEqual(namespaces, {"game_run"})

    def test_g_challenge_is_scoped_to_its_run(self) -> None:
        first = self.resolve("FFV", run_kind="challenge", rules={"challenge": "Crystal Roulette"}).active_run
        second = self.resolve("FFV", session="s2", explicit_new=True, run_kind="casual").active_run
        self.assertEqual(self.runs.state(first.id)["challenge"], "Crystal Roulette")
        self.assertEqual(self.runs.state(second.id)["challenge"], "")

    def test_h_restart_reconstructs_the_modern_contract(self) -> None:
        run = self.resolve("FFV", run_kind="challenge", rules={"challenge": "No shops"}).active_run
        self.runs.update_state(
            run.id,
            updates={"current_location": "Karnak", "party_members": ["Bartz", "Lenna"]},
            provenance="leo_clarification",
            confidence=0.9,
            evidence=self.evidence(),
        )
        restarted = GameRunService(
            GameV2Repository(self.connect),
            BeliefLifecycleService(BeliefRepository(self.connect), now_fn=lambda: 1001.0),
            now_fn=lambda: 1001.0,
        )
        state = restarted.state(run.id)
        self.assertEqual((state["current_location"], state["party_members"], state["challenge"]),
                         ("Karnak", ["Bartz", "Lenna"], "No shops"))

    def test_i_linked_legacy_progress_migrates_without_new_run(self) -> None:
        run = self.resolve("P5R", session="56").active_run
        self._insert_progress("persona_5_royal", "56", current_area="Kamoshida Palace")
        self.runner.migrate(game_run_state_canonicalization_migrations())
        self.assertEqual(self.runs.state(run.id)["current_location"], "Kamoshida Palace")
        conn = self.connect()
        audit = conn.execute("SELECT * FROM legacy_game_run_state_migration_audit").fetchone()
        self.assertEqual((audit["classification"], audit["outcome"], audit["target_run_id"]),
                         ("CURRENT_RUN_STATE", "migrated", run.id))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM game_runs").fetchone()[0], 1)
        conn.close()

    def test_j_ambiguous_and_orphaned_progress_never_get_invented_runs(self) -> None:
        self._insert_progress("final_fantasy_v", "current", current_area="Karnak")
        self._insert_progress("persona_5_royal", "47", current_area="Mementos")
        self.runner.migrate(game_run_state_canonicalization_migrations())
        conn = self.connect()
        rows = conn.execute(
            "SELECT source_record_id,classification,outcome,target_run_id FROM legacy_game_run_state_migration_audit ORDER BY source_record_id"
        ).fetchall()
        self.assertEqual([(r["classification"], r["outcome"], r["target_run_id"]) for r in rows],
                         [("AMBIGUOUS", "ambiguous", ""), ("ORPHANED", "skipped", "")])
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM game_runs").fetchone()[0], 0)
        conn.close()

    def test_k_dossier_only_migrates_supported_semantic_claims(self) -> None:
        self._insert_dossier(
            claims=["Jobs can be changed outside battle"],
            sources=[{
                "claim": "Jobs can be changed outside battle",
                "location": "https://example.test/ffv/jobs",
                "excerpt": "Jobs can be changed outside battle.",
                "source_type": "curated",
            }],
        )
        self.runner.migrate(game_knowledge_canonicalization_migrations())
        selected, _ = self.knowledge.find("final_fantasy_v")
        self.assertEqual([item["object"] for item in selected], ["Jobs can be changed outside battle"])
        conn = self.connect()
        self.assertEqual(conn.execute("SELECT v2_projection_version FROM game_dossiers").fetchone()[0], 1)
        conn.close()

    def test_l_post_cutover_writes_do_not_touch_legacy_tables(self) -> None:
        self._insert_progress("persona_5_royal", "old")
        before = self._legacy_progress_json()
        run = self.resolve().active_run
        self.runs.update_state(
            run.id,
            updates={"current_objective": "send the calling card"},
            provenance="leo_clarification",
            confidence=0.9,
            evidence=self.evidence(),
        )
        self.assertEqual(self._legacy_progress_json(), before)

    def test_m_knowledge_retrieval_does_not_depend_on_legacy_run_state(self) -> None:
        identity = self.repo.resolve_identity("FFV")
        self.knowledge.add_validated(
            game_id=identity.game_id,
            subject_ref="jobs",
            predicate="system",
            object_value="job_system",
            confidence=0.9,
            evidence=self.evidence("curated:1", "FFV has a job system"),
            source_type="curated",
            source_quality="validated",
        )
        self._insert_progress("final_fantasy_v", "current", current_area="wrong legacy area")
        context = GameContextResolver(self.repo, self.runs, self.knowledge).build(
            game="FFV", purpose="game_fact", subject_ref="jobs", predicate="system"
        )
        self.assertEqual(context.knowledge_claims[0]["object"], "job_system")
        self.assertEqual(context.run_facts, ())

    def test_n_game_switch_does_not_mix_previous_run_state(self) -> None:
        ffv = self.resolve("FFV").active_run
        self.runs.update_state(
            ffv.id,
            updates={"current_location": "Karnak"},
            provenance="leo_clarification",
            confidence=0.9,
            evidence=self.evidence(),
        )
        persona = self.resolve("P5R", session="p1").active_run
        self.assertEqual(self.runs.state(persona.id)["current_location"], "")
        self.assertEqual(self.runs.state(ffv.id)["current_location"], "Karnak")

    def test_game_research_worker_owner_closes_idempotently(self) -> None:
        connection=sqlite3.connect(":memory:",check_same_thread=False)
        service=GameResearchService(store=GameIntelligenceStore(connection=connection))
        service.close();service.close()
        plan=service.plan_search(
            game_title="FFV",game_id="final_fantasy_v",entity="jobs",
            question_type="mechanic",expected_fact_type="general_mechanic",
        )
        with self.assertRaisesRegex(RuntimeError,"game_research_service_closed"):
            service.queue_research(plan,progress=None,scene_id="closed")
        connection.close()

    def _insert_progress(self, game_id: str, session_id: str, **updates) -> None:
        state = {
            "game_id": game_id,
            "stream_session_id": session_id,
            "playthrough_type": "first_playthrough",
            "spoiler_policy": "strict",
            "current_chapter": "",
            "current_area": "",
            "known_party_members": [],
            "encountered_characters": [],
            "encountered_bosses": [],
            "unlocked_mechanics": [],
            "recent_progress_markers": [],
            "confidence": 0.9,
            "last_updated_at": "2026-08-12T17:09:09+00:00",
            **updates,
        }
        conn = self.connect()
        conn.execute(
            "INSERT INTO game_progress_states(game_id,stream_session_id,state_json,updated_at,game_run_id) VALUES(?,?,?,?,NULL)",
            (game_id, session_id, json.dumps(state), state["last_updated_at"]),
        )
        conn.commit()
        conn.close()

    def _insert_dossier(self, *, claims: list[str], sources: list[dict]) -> None:
        payload = {
            "game_id": "final_fantasy_v",
            "canonical_title": "FINAL FANTASY V",
            "confirmed_general_mechanics": claims,
            "sources": sources,
            "dossier_version": 2,
        }
        conn = self.connect()
        conn.execute(
            "INSERT INTO game_dossiers(game_id,canonical_title,dossier_json,dossier_version,created_at,updated_at,v2_projection_version) VALUES(?,?,?,?,?,?,0)",
            ("final_fantasy_v", "FINAL FANTASY V", json.dumps(payload), 2, "2026-08-10", "2026-08-10"),
        )
        conn.commit()
        conn.close()

    def _legacy_progress_json(self) -> str:
        conn = self.connect()
        value = conn.execute("SELECT state_json FROM game_progress_states ORDER BY game_id,stream_session_id").fetchall()
        conn.close()
        return json.dumps([row[0] for row in value])


if __name__ == "__main__":
    unittest.main()
