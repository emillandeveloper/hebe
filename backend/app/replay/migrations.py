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


def social_world_v2_migrations() -> tuple[Migration, ...]:
    def phase4(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS people (
              person_id TEXT PRIMARY KEY, created_at REAL NOT NULL, last_seen_at REAL NOT NULL,
              scope TEXT NOT NULL DEFAULT 'stream_public', schema_version INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS person_identities (
              id TEXT PRIMARY KEY, person_id TEXT NOT NULL, platform TEXT NOT NULL,
              platform_user_id TEXT NOT NULL DEFAULT '', login TEXT NOT NULL DEFAULT '',
              display_name TEXT NOT NULL DEFAULT '', aliases_json TEXT NOT NULL DEFAULT '[]',
              first_seen_at REAL NOT NULL, last_seen_at REAL NOT NULL, confidence REAL NOT NULL,
              source TEXT NOT NULL, schema_version INTEGER NOT NULL DEFAULT 1,
              FOREIGN KEY(person_id) REFERENCES people(person_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_person_identity_stable ON person_identities(platform,platform_user_id) WHERE platform_user_id<>'';
            CREATE INDEX IF NOT EXISTS idx_person_identity_login ON person_identities(platform,lower(login));
            CREATE TABLE IF NOT EXISTS person_sessions (
              person_id TEXT NOT NULL, stream_session_id TEXT NOT NULL, first_seen_at REAL NOT NULL,
              last_seen_at REAL NOT NULL, PRIMARY KEY(person_id,stream_session_id),
              FOREIGN KEY(person_id) REFERENCES people(person_id)
            );
            CREATE TABLE IF NOT EXISTS social_episodes (
              id TEXT PRIMARY KEY, episode_type TEXT NOT NULL, participant_ids_json TEXT NOT NULL,
              origin_event_id TEXT NOT NULL UNIQUE, related_event_ids_json TEXT NOT NULL DEFAULT '[]',
              summary TEXT NOT NULL, tone_observations_json TEXT NOT NULL DEFAULT '[]',
              created_at REAL NOT NULL, relevance_until REAL NOT NULL, retention_until REAL NOT NULL,
              sensitivity TEXT NOT NULL, retention_class TEXT NOT NULL, retrieval_scope TEXT NOT NULL,
              salience_reason TEXT NOT NULL, schema_version INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_social_episode_relevance ON social_episodes(relevance_until,retention_until,episode_type);
            CREATE TABLE IF NOT EXISTS shared_culture_items (
              id TEXT PRIMARY KEY, label TEXT NOT NULL, meaning TEXT NOT NULL, origin_episode_id TEXT NOT NULL,
              participant_ids_json TEXT NOT NULL, scope TEXT NOT NULL, tone TEXT NOT NULL,
              status TEXT NOT NULL, confidence REAL NOT NULL, created_at REAL NOT NULL,
              last_reinforced_at REAL NOT NULL, last_used_at REAL NOT NULL DEFAULT 0,
              reuse_count INTEGER NOT NULL DEFAULT 0, cooldown_until REAL NOT NULL DEFAULT 0,
              schema_version INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_shared_culture_status ON shared_culture_items(status,cooldown_until,last_reinforced_at);
            CREATE TABLE IF NOT EXISTS shared_culture_evidence (
              id TEXT PRIMARY KEY, culture_item_id TEXT NOT NULL, event_id TEXT NOT NULL,
              episode_id TEXT NOT NULL DEFAULT '', reaction TEXT NOT NULL, polarity TEXT NOT NULL,
              weight REAL NOT NULL, observed_at REAL NOT NULL, authority TEXT NOT NULL,
              FOREIGN KEY(culture_item_id) REFERENCES shared_culture_items(id), UNIQUE(culture_item_id,event_id)
            );
            CREATE INDEX IF NOT EXISTS idx_culture_evidence_item ON shared_culture_evidence(culture_item_id,observed_at);
            """
        )
    return (Migration("social_world_v2",1,"people_episodes_and_shared_culture",phase4),)


def learning_v2_migrations() -> tuple[Migration, ...]:
    def phase5(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS consolidation_runs (
              id TEXT PRIMARY KEY, session_id TEXT NOT NULL, input_start_event TEXT NOT NULL,
              input_end_event TEXT NOT NULL, pre_state_version TEXT NOT NULL,
              consolidator_version TEXT NOT NULL, status TEXT NOT NULL,
              started_at REAL NOT NULL, completed_at REAL NOT NULL DEFAULT 0,
              idempotency_key TEXT NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS idx_consolidation_session ON consolidation_runs(session_id,input_end_event,status);
            CREATE TABLE IF NOT EXISTS consolidation_deltas (
              id TEXT PRIMARY KEY, consolidation_run_id TEXT NOT NULL, domain TEXT NOT NULL,
              delta_type TEXT NOT NULL, payload_json TEXT NOT NULL, evidence_ids_json TEXT NOT NULL,
              validator_result TEXT NOT NULL, committed_object_ref TEXT NOT NULL DEFAULT '',
              idempotency_key TEXT NOT NULL UNIQUE, rejection_reason TEXT NOT NULL DEFAULT '',
              created_at REAL NOT NULL, FOREIGN KEY(consolidation_run_id) REFERENCES consolidation_runs(id)
            );
            CREATE INDEX IF NOT EXISTS idx_consolidation_delta_run ON consolidation_deltas(consolidation_run_id,validator_result);
            CREATE TABLE IF NOT EXISTS action_ledger (
              id TEXT PRIMARY KEY, action_type TEXT NOT NULL, target TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL, source_store TEXT NOT NULL, source_record_id TEXT NOT NULL,
              requested_at REAL NOT NULL, completed_at REAL NOT NULL DEFAULT 0,
              evidence_json TEXT NOT NULL DEFAULT '{}', schema_version INTEGER NOT NULL DEFAULT 1,
              UNIQUE(source_store,source_record_id)
            );
            CREATE INDEX IF NOT EXISTS idx_action_ledger_claim ON action_ledger(action_type,target,requested_at,status);
            CREATE TABLE IF NOT EXISTS temporal_maintenance_audit (
              id TEXT PRIMARY KEY, object_ref TEXT NOT NULL, object_type TEXT NOT NULL,
              old_status TEXT NOT NULL, new_status TEXT NOT NULL, reason TEXT NOT NULL,
              changed_at REAL NOT NULL, policy_version TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_temporal_audit_object ON temporal_maintenance_audit(object_ref,changed_at);
            CREATE TABLE IF NOT EXISTS learning_observations (
              id TEXT PRIMARY KEY, model TEXT NOT NULL, subject TEXT NOT NULL, value TEXT NOT NULL,
              event_id TEXT NOT NULL UNIQUE, observed_at REAL NOT NULL, explicit INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_learning_observation_pattern ON learning_observations(model,subject,value,observed_at);
            CREATE TABLE IF NOT EXISTS scene_transitions (
              id TEXT PRIMARY KEY, source_event_id TEXT NOT NULL, transition_type TEXT NOT NULL,
              destination_ref TEXT NOT NULL DEFAULT '', payload_json TEXT NOT NULL,
              created_at REAL NOT NULL, UNIQUE(source_event_id,transition_type)
            );
            """
        )
    return (Migration("learning_v2",1,"consolidation_temporal_action_and_scene",phase5),)


def architecture_consolidation_migrations() -> tuple[Migration, ...]:
    """Phase 6 audit tables. These preserve history and never delete domain data."""

    def phase6(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cognitive_migration_audit (
              id TEXT PRIMARY KEY, run_id TEXT NOT NULL, operation TEXT NOT NULL,
              source_store TEXT NOT NULL, source_record_id TEXT NOT NULL,
              target_store TEXT NOT NULL DEFAULT '', target_record_id TEXT NOT NULL DEFAULT '',
              classification TEXT NOT NULL, reason TEXT NOT NULL,
              provenance_json TEXT NOT NULL DEFAULT '{}', applied_at TEXT NOT NULL,
              schema_version INTEGER NOT NULL DEFAULT 1,
              UNIQUE(run_id,source_store,source_record_id,operation)
            );
            CREATE INDEX IF NOT EXISTS idx_cognitive_migration_source
              ON cognitive_migration_audit(source_store,source_record_id,classification);
            CREATE TABLE IF NOT EXISTS cognitive_cutover_state (
              concern TEXT PRIMARY KEY, canonical_owner TEXT NOT NULL,
              legacy_mode TEXT NOT NULL, compatibility_write_enabled INTEGER NOT NULL DEFAULT 0,
              decided_at TEXT NOT NULL, reason TEXT NOT NULL,
              schema_version INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS cognitive_hygiene_runs (
              run_id TEXT PRIMARY KEY, db_fingerprint_before TEXT NOT NULL,
              mode TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT NOT NULL DEFAULT '',
              classification_counts_json TEXT NOT NULL DEFAULT '{}',
              destructive_changes INTEGER NOT NULL DEFAULT 0,
              schema_version INTEGER NOT NULL DEFAULT 1
            );
            """
        )

    return (
        Migration(
            "architecture_consolidation",
            1,
            "audit_hygiene_and_cutover_state",
            phase6,
        ),
    )
