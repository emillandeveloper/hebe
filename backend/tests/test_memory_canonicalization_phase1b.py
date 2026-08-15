from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest

from app.cognitive.memory_store import MemoryStore
from app.epistemics.memory_migration import (
    LEGACY_MEMORY_MIGRATION_COMPONENT,
    legacy_memory_fact_migrations,
)
from app.epistemics.models import Belief, BeliefStatus, EvidenceRef
from app.epistemics.repository import BeliefRepository
from app.replay.migrations import MigrationRunner, belief_v2_migrations


LEGACY_SCHEMA = """
CREATE TABLE memory_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    subject TEXT,
    payload_json TEXT,
    source_text TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_used_at TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    belief_id TEXT,
    epistemic_status TEXT
);
"""


class MemoryCanonicalizationPhase1BTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "historical.sqlite3")
        conn = self.connection()
        conn.executescript(LEGACY_SCHEMA)
        conn.commit()
        conn.close()
        MigrationRunner(self.connection).migrate(belief_v2_migrations())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert_legacy(
        self,
        *,
        kind: str,
        subject: str = "leo",
        payload: object = None,
        source_text: str = "legacy source",
        confidence: object = 0.8,
        created_at: str = "2024-01-02T03:04:05+00:00",
        updated_at: str = "2024-02-03T04:05:06+00:00",
        active: int = 1,
        raw_payload: str | None = None,
    ) -> int:
        conn = self.connection()
        cur = conn.execute(
            """INSERT INTO memory_facts
               (kind,subject,payload_json,source_text,confidence,created_at,updated_at,active)
               VALUES(?,?,?,?,?,?,?,?)""",
            (
                kind,
                subject,
                raw_payload if raw_payload is not None else json.dumps(payload),
                source_text,
                confidence,
                created_at,
                updated_at,
                active,
            ),
        )
        conn.commit()
        fact_id = int(cur.lastrowid)
        conn.close()
        return fact_id

    def migrate(self) -> list[dict]:
        return MigrationRunner(self.connection).migrate(legacy_memory_fact_migrations())

    def beliefs(self) -> list[sqlite3.Row]:
        conn = self.connection()
        rows = conn.execute("SELECT * FROM beliefs ORDER BY id").fetchall()
        conn.close()
        return rows

    def audits(self) -> list[sqlite3.Row]:
        conn = self.connection()
        rows = conn.execute(
            "SELECT * FROM legacy_memory_fact_migration_audit ORDER BY source_fact_id"
        ).fetchall()
        conn.close()
        return rows

    def test_a_direct_mapping_creates_canonical_preference(self) -> None:
        self.insert_legacy(
            kind="preference",
            subject="leo.language.spanish",
            payload={"text": "Leo prefers Spanish from Spain"},
        )
        self.migrate()
        row = self.beliefs()[0]
        self.assertEqual((row["namespace"], row["scope_kind"], row["scope_id"]),
                         ("memory.preference", "owner_local", "leo"))
        self.assertEqual(row["subject_ref"], "leo.language.spanish")

    def test_b_provenance_and_confidence_are_preserved(self) -> None:
        fact_id = self.insert_legacy(
            kind="leo_fact", payload={"text": "Leo likes tea"}, confidence=0.73,
            source_text="Leo said: I like tea",
        )
        self.migrate()
        row = self.beliefs()[0]
        self.assertAlmostEqual(float(row["confidence"]), 0.73)
        conn = self.connection()
        evidence = conn.execute("SELECT * FROM belief_evidence").fetchone()
        conn.close()
        self.assertEqual(evidence["source_record_type"], "memory_facts")
        self.assertEqual(evidence["source_record_id"], str(fact_id))
        self.assertEqual(json.loads(evidence["literal_span_json"])["text"], "Leo said: I like tea")

    def test_c_created_and_updated_timestamps_are_preserved(self) -> None:
        self.insert_legacy(kind="habit", payload={"text": "Leo streams on Fridays"})
        self.migrate()
        row = self.beliefs()[0]
        self.assertEqual(float(row["created_at"]), 1704164645.0)
        self.assertEqual(float(row["last_confirmed_at"]), 1706933106.0)
        self.assertEqual(float(row["valid_from"]), 1704164645.0)

    def test_d_running_the_migration_twice_does_not_duplicate(self) -> None:
        self.insert_legacy(kind="project_fact", subject="hebe", payload={"text": "Hebe is local"})
        first = self.migrate()
        second = self.migrate()
        self.assertFalse(first[0]["already_applied"])
        self.assertTrue(second[0]["already_applied"])
        self.assertEqual(len(self.beliefs()), 1)
        self.assertEqual(len(self.audits()), 1)

    def test_e_equivalent_existing_belief_is_deduplicated(self) -> None:
        fact_id = self.insert_legacy(
            kind="preference", subject="leo.language", payload={"value": "es-ES", "predicate": "locale"}
        )
        repository = BeliefRepository(self.connection)
        existing = Belief(
            id="belief_existing", namespace="memory.preference", scope_kind="owner_local", scope_id="leo",
            subject_ref="leo.language", predicate="locale", object_value="es-ES",
            epistemic_status=BeliefStatus.INFERRED, confidence=0.9, authority_class="owner",
            created_at=1.0, last_confirmed_at=1.0, valid_from=1.0, valid_until=0.0,
            relevance_until=0.0,
        )
        repository.propose(existing, EvidenceRef(
            source_event_id="existing", source_record_type="test", source_record_id="existing",
            observed_at=1.0,
        ))
        self.migrate()
        self.assertEqual(len(self.beliefs()), 1)
        audit = self.audits()[0]
        self.assertEqual((audit["source_fact_id"], audit["outcome"], audit["target_belief_id"]),
                         (str(fact_id), "deduplicated", "belief_existing"))
        self.assertEqual(len(repository.evidence_for("belief_existing")), 2)

    def test_f_transformable_generic_fact_uses_explicit_predicate_and_value(self) -> None:
        self.insert_legacy(
            kind="fact", subject="leo", payload={"predicate": "favorite_drink", "value": "tea"}
        )
        self.migrate()
        row = self.beliefs()[0]
        self.assertEqual((row["namespace"], row["predicate"], json.loads(row["object_json"])),
                         ("memory.fact", "favorite_drink", "tea"))

    def test_g_obsolete_task_is_audited_without_polluting_beliefs(self) -> None:
        self.insert_legacy(kind="task", subject="old todo", payload={"done": False})
        self.migrate()
        self.assertEqual(self.beliefs(), [])
        self.assertEqual((self.audits()[0]["outcome"], self.audits()[0]["reason"]),
                         ("skipped", "obsolete_kind"))

    def test_h_unknown_kind_is_visible_and_does_not_invent_semantics(self) -> None:
        self.insert_legacy(kind="telepathic_note", payload={"value": "unsafe assumption"})
        self.migrate()
        self.assertEqual(self.beliefs(), [])
        audit = self.audits()[0]
        self.assertEqual((audit["outcome"], audit["reason"]), ("unsupported", "unknown_kind"))

    def test_i_corrupt_row_does_not_abort_other_rows(self) -> None:
        self.insert_legacy(kind="preference", raw_payload="{not-json")
        self.insert_legacy(kind="habit", subject="sleep", payload={"text": "Leo sleeps late"})
        self.migrate()
        self.assertEqual(len(self.beliefs()), 1)
        self.assertEqual([row["outcome"] for row in self.audits()], ["error", "migrated"])

    def test_j_retrieval_after_cutover_uses_only_canonical_memory(self) -> None:
        self.insert_legacy(kind="preference", subject="canonical", payload={"text": "canonical tea"})
        self.migrate()
        self.insert_legacy(kind="preference", subject="late legacy", payload={"text": "legacy coffee"})
        store = MemoryStore.from_connection_factory(self.connection, run_legacy_migration=False)
        results = store.search_facts(query_text="coffee", limit=20)
        self.assertEqual(results, [])
        self.assertEqual([item.subject for item in store.search_facts(query_text="tea", limit=20)], ["canonical"])

    def test_k_new_memory_write_never_writes_memory_facts(self) -> None:
        self.migrate()
        store = MemoryStore.from_connection_factory(self.connection, run_legacy_migration=False)
        fact = store.create_fact(
            kind="preference", subject="leo.language", payload={"text": "Natural English"},
            source_text="Remember I prefer natural English", confidence=0.95,
        )
        self.assertTrue(str(fact.id).startswith("belief_"))
        conn = self.connection()
        count = conn.execute("SELECT COUNT(*) FROM memory_facts").fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

    def test_l_completed_startup_does_not_require_the_legacy_adapter(self) -> None:
        first = self.migrate()
        store = MemoryStore.from_connection_factory(self.connection)
        conn = self.connection()
        markers = conn.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE component=?",
            (LEGACY_MEMORY_MIGRATION_COMPONENT,),
        ).fetchone()[0]
        conn.close()
        self.assertFalse(first[0]["already_applied"])
        self.assertEqual(markers, 1)
        self.assertFalse(hasattr(store, "legacy_adapter"))


if __name__ == "__main__":
    unittest.main()
