from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable


@dataclass(frozen=True, slots=True)
class Migration:
    component: str
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]

    @property
    def checksum(self) -> str:
        identity = f"{self.component}:{self.version}:{self.name}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class MigrationRunner:
    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self.connection_factory = connection_factory

    def migrate(self, migrations: Iterable[Migration]) -> list[dict]:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    component TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    PRIMARY KEY(component, version)
                )
                """
            )
            conn.commit()
            applied: list[dict] = []
            ordered = sorted(migrations, key=lambda item: (item.component, item.version))
            for migration in ordered:
                row = conn.execute(
                    "SELECT checksum, name, applied_at FROM schema_migrations WHERE component=? AND version=?",
                    (migration.component, migration.version),
                ).fetchone()
                if row is not None:
                    if row["checksum"] != migration.checksum:
                        raise RuntimeError(
                            f"migration checksum mismatch: {migration.component}:{migration.version}"
                        )
                    applied.append({
                        "component": migration.component,
                        "version": migration.version,
                        "name": row["name"],
                        "checksum": row["checksum"],
                        "applied_at": row["applied_at"],
                        "already_applied": True,
                    })
                    continue
                try:
                    conn.execute("BEGIN")
                    migration.apply(conn)
                    applied_at = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        "INSERT INTO schema_migrations(component, version, name, checksum, applied_at) VALUES (?, ?, ?, ?, ?)",
                        (migration.component, migration.version, migration.name, migration.checksum, applied_at),
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                applied.append({
                    "component": migration.component,
                    "version": migration.version,
                    "name": migration.name,
                    "checksum": migration.checksum,
                    "applied_at": applied_at,
                    "already_applied": False,
                })
            return applied
        finally:
            conn.close()


def replay_foundation_migrations() -> tuple[Migration, ...]:
    def create_metadata(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cognitive_replay_metadata (
                scenario_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                last_run_at TEXT NOT NULL
            )
            """
        )

    return (Migration("cognitive_replay", 1, "replay_metadata", create_metadata),)


def conversation_continuity_migrations() -> tuple[Migration, ...]:
    def create_phase1_tables(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                context_kind TEXT NOT NULL,
                context_id TEXT NOT NULL,
                participants_json TEXT NOT NULL,
                attention_state TEXT NOT NULL,
                turn_owner TEXT NOT NULL,
                expected_reply_type TEXT,
                expected_reply_json TEXT,
                topic TEXT NOT NULL,
                origin_event_id TEXT NOT NULL,
                last_event_id TEXT NOT NULL,
                opened_at REAL NOT NULL,
                last_turn_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                status TEXT NOT NULL,
                closure_reason TEXT NOT NULL DEFAULT '',
                version INTEGER NOT NULL DEFAULT 1,
                domain_payload_json TEXT NOT NULL DEFAULT '{}',
                consumed_event_ids_json TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_conversations_context_status
                ON conversations(context_kind, context_id, status);
            CREATE INDEX IF NOT EXISTS idx_conversations_status_expiry
                ON conversations(status, expires_at);

            CREATE TABLE IF NOT EXISTS open_threads (
                id TEXT PRIMARY KEY,
                thread_type TEXT NOT NULL,
                scope_kind TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                participant_ids_json TEXT NOT NULL,
                subject_ref TEXT NOT NULL,
                summary TEXT NOT NULL,
                origin_event_id TEXT NOT NULL,
                latest_event_id TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                relevance_until REAL NOT NULL,
                valid_until REAL NOT NULL,
                resolved_at REAL NOT NULL DEFAULT 0,
                resolution_event_id TEXT NOT NULL DEFAULT '',
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                version INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_open_threads_scope_status
                ON open_threads(scope_kind, scope_id, status);
            CREATE INDEX IF NOT EXISTS idx_open_threads_status_validity
                ON open_threads(status, relevance_until, valid_until);
            CREATE INDEX IF NOT EXISTS idx_open_threads_subject
                ON open_threads(subject_ref, status);
            """
        )

    return (Migration("conversation_continuity", 1, "conversation_and_open_threads", create_phase1_tables),)


def belief_v2_migrations() -> tuple[Migration, ...]:
    def phase2(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS beliefs (
              id TEXT PRIMARY KEY, namespace TEXT NOT NULL, scope_kind TEXT NOT NULL, scope_id TEXT NOT NULL,
              subject_ref TEXT NOT NULL, predicate TEXT NOT NULL, object_json TEXT NOT NULL,
              epistemic_status TEXT NOT NULL, confidence REAL NOT NULL, authority_class TEXT NOT NULL,
              created_at REAL NOT NULL, last_confirmed_at REAL NOT NULL, valid_from REAL NOT NULL,
              valid_until REAL NOT NULL DEFAULT 0, relevance_until REAL NOT NULL DEFAULT 0,
              superseded_by TEXT NOT NULL DEFAULT '', owner_confirmed INTEGER NOT NULL DEFAULT 0,
              sensitivity TEXT NOT NULL DEFAULT 'normal', schema_version INTEGER NOT NULL DEFAULT 1,
              retention_policy TEXT NOT NULL DEFAULT 'retain_history', version INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_beliefs_identity_active ON beliefs(namespace,scope_kind,scope_id,subject_ref,predicate,epistemic_status);
            CREATE INDEX IF NOT EXISTS idx_beliefs_validity ON beliefs(epistemic_status,valid_until,relevance_until);
            CREATE INDEX IF NOT EXISTS idx_beliefs_scope_sensitivity ON beliefs(scope_kind,scope_id,sensitivity);
            CREATE TABLE IF NOT EXISTS belief_evidence (
              id TEXT PRIMARY KEY, belief_id TEXT NOT NULL, source_event_id TEXT NOT NULL,
              source_record_type TEXT NOT NULL, source_record_id TEXT NOT NULL, relation TEXT NOT NULL,
              weight REAL NOT NULL, observed_at REAL NOT NULL, extractor TEXT NOT NULL,
              extractor_version TEXT NOT NULL, literal_span_json TEXT NOT NULL DEFAULT '{}',
              subject_key TEXT NOT NULL, FOREIGN KEY(belief_id) REFERENCES beliefs(id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_belief_evidence_idempotency ON belief_evidence(source_event_id,relation,subject_key);
            CREATE INDEX IF NOT EXISTS idx_belief_evidence_belief ON belief_evidence(belief_id,observed_at);
            CREATE INDEX IF NOT EXISTS idx_belief_evidence_source ON belief_evidence(source_record_type,source_record_id);
            CREATE TABLE IF NOT EXISTS scene_assertions (
              id TEXT PRIMARY KEY, subject_ref TEXT NOT NULL, predicate TEXT NOT NULL, object_json TEXT NOT NULL,
              epistemic_status TEXT NOT NULL, confidence REAL NOT NULL, evidence_ids_json TEXT NOT NULL,
              referent_data_json TEXT NOT NULL, observed_at REAL NOT NULL, valid_from REAL NOT NULL,
              valid_until REAL NOT NULL DEFAULT 0, provenance TEXT NOT NULL, extractor TEXT NOT NULL,
              extractor_version TEXT NOT NULL, schema_version INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        def add(table: str, column: str, declaration: str) -> None:
            existing={str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
            if table in {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")} and column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
        for column,declaration in (
            ("context_kind","TEXT"),("source_record_type","TEXT"),("source_record_id","TEXT"),
            ("authority","TEXT"),("literal_evidence_json","TEXT"),("valid_from","REAL"),
            ("valid_until","REAL"),("supersedes_event_uid","TEXT"),("schema_version","INTEGER NOT NULL DEFAULT 1"),
        ): add("live_session_timeline",column,declaration)
        add("memory_facts","belief_id","TEXT")
        add("memory_facts","epistemic_status","TEXT")
        add("memory_chunks","belief_id","TEXT")
        add("memory_chunks","episode_id","TEXT")
    return (Migration("belief_v2",1,"beliefs_evidence_and_compatibility_columns",phase2),)


def game_context_v2_migrations() -> tuple[Migration, ...]:
    def phase3(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS game_identities (
              game_id TEXT PRIMARY KEY, canonical_name TEXT NOT NULL, aliases_json TEXT NOT NULL,
              platform_ids_json TEXT NOT NULL DEFAULT '{}', series TEXT NOT NULL DEFAULT '',
              schema_version INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS game_runs (
              id TEXT PRIMARY KEY, game_id TEXT NOT NULL, owner_id TEXT NOT NULL, run_kind TEXT NOT NULL,
              rules_json TEXT NOT NULL, status TEXT NOT NULL, started_at REAL NOT NULL,
              last_active_at REAL NOT NULL, ended_at REAL NOT NULL DEFAULT 0,
              current_checkpoint_version INTEGER NOT NULL DEFAULT 1,
              created_from_event_id TEXT NOT NULL, schema_version INTEGER NOT NULL DEFAULT 1,
              FOREIGN KEY(game_id) REFERENCES game_identities(game_id)
            );
            CREATE INDEX IF NOT EXISTS idx_game_runs_resolution ON game_runs(game_id,owner_id,status,last_active_at);
            CREATE TABLE IF NOT EXISTS game_run_sessions (
              id TEXT PRIMARY KEY, game_run_id TEXT NOT NULL, stream_session_id TEXT NOT NULL,
              started_at REAL NOT NULL, ended_at REAL NOT NULL DEFAULT 0, evidence_event_id TEXT NOT NULL,
              source TEXT NOT NULL, schema_version INTEGER NOT NULL DEFAULT 1,
              FOREIGN KEY(game_run_id) REFERENCES game_runs(id), UNIQUE(game_run_id,stream_session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_game_run_sessions_stream ON game_run_sessions(stream_session_id,game_run_id);
            CREATE TABLE IF NOT EXISTS game_run_events (
              id TEXT PRIMARY KEY, game_run_id TEXT NOT NULL, event_type TEXT NOT NULL,
              subject_ref TEXT NOT NULL, predicate TEXT NOT NULL, object_json TEXT NOT NULL,
              evidence_event_id TEXT NOT NULL, belief_id TEXT NOT NULL DEFAULT '', observed_at REAL NOT NULL,
              epistemic_status TEXT NOT NULL, schema_version INTEGER NOT NULL DEFAULT 1,
              FOREIGN KEY(game_run_id) REFERENCES game_runs(id)
            );
            CREATE INDEX IF NOT EXISTS idx_game_run_events_run ON game_run_events(game_run_id,observed_at,event_type);
            CREATE TABLE IF NOT EXISTS game_knowledge_facts (
              id TEXT PRIMARY KEY, game_id TEXT NOT NULL, belief_id TEXT NOT NULL UNIQUE,
              source_type TEXT NOT NULL, source_quality TEXT NOT NULL, spoiler_class TEXT NOT NULL,
              dossier_link TEXT NOT NULL DEFAULT '', version_tag TEXT NOT NULL DEFAULT '',
              created_at REAL NOT NULL, schema_version INTEGER NOT NULL DEFAULT 1,
              FOREIGN KEY(game_id) REFERENCES game_identities(game_id), FOREIGN KEY(belief_id) REFERENCES beliefs(id)
            );
            CREATE INDEX IF NOT EXISTS idx_game_knowledge_lookup ON game_knowledge_facts(game_id,spoiler_class,created_at);
            CREATE TABLE IF NOT EXISTS game_knowledge_v2_gaps (
              id TEXT PRIMARY KEY, game_id TEXT NOT NULL, run_id TEXT NOT NULL DEFAULT '',
              subject_ref TEXT NOT NULL, question_type TEXT NOT NULL, query_intent TEXT NOT NULL,
              spoiler_ceiling TEXT NOT NULL, required_confidence REAL NOT NULL,
              created_from_event_id TEXT NOT NULL, normalized_gap_key TEXT NOT NULL UNIQUE,
              status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
              resolved_fact_ids_json TEXT NOT NULL DEFAULT '[]', schema_version INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_game_v2_gaps_status ON game_knowledge_v2_gaps(game_id,status,updated_at);
            """
        )
        tables={str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        def add(table: str, column: str, declaration: str) -> None:
            if table in tables and column not in {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
        add("game_dossiers","v2_projection_version","INTEGER NOT NULL DEFAULT 0")
        add("game_progress_states","game_run_id","TEXT")
        add("game_sessions","game_run_id","TEXT")
    return (Migration("game_context_v2",1,"durable_runs_knowledge_and_gaps",phase3),)
