from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from app.replay.migrations import Migration


RUN_MIGRATION_COMPONENT = "game_run_state_canonicalization"
RUN_MIGRATION_VERSION = 1
KNOWLEDGE_MIGRATION_COMPONENT = "game_knowledge_canonicalization"
KNOWLEDGE_MIGRATION_VERSION = 1


def game_run_state_canonicalization_migrations() -> tuple[Migration, ...]:
    return (
        Migration(
            RUN_MIGRATION_COMPONENT,
            RUN_MIGRATION_VERSION,
            "linked_legacy_game_state_to_run_beliefs",
            _migrate_run_state,
        ),
    )


def game_knowledge_canonicalization_migrations() -> tuple[Migration, ...]:
    return (
        Migration(
            KNOWLEDGE_MIGRATION_COMPONENT,
            KNOWLEDGE_MIGRATION_VERSION,
            "supported_dossier_claims_to_game_knowledge",
            _migrate_game_knowledge,
        ),
    )


def _migrate_run_state(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    _create_run_audit(conn)
    tables = _tables(conn)
    if "game_progress_states" in tables:
        _require_columns(
            conn,
            "game_progress_states",
            {"game_id", "stream_session_id", "state_json", "updated_at", "game_run_id"},
        )
        for row in conn.execute(
            "SELECT * FROM game_progress_states ORDER BY game_id,stream_session_id"
        ).fetchall():
            _migrate_progress_row(conn, row)
    if "game_sessions" in tables:
        _require_columns(
            conn,
            "game_sessions",
            {"id", "game", "game_key", "updated_at", "game_run_id"},
        )
        for row in conn.execute("SELECT * FROM game_sessions ORDER BY id").fetchall():
            _migrate_session_row(conn, row)


def _migrate_progress_row(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    game_id = str(row["game_id"] or "").strip()
    session_id = str(row["stream_session_id"] or "").strip()
    source_id = f"{game_id}|{session_id}"
    try:
        payload = json.loads(row["state_json"] or "{}")
        if not isinstance(payload, dict):
            raise ValueError("state_json_not_object")
        if str(payload.get("game_id") or game_id) != game_id:
            raise ValueError("game_id_mismatch")
        if str(payload.get("stream_session_id") or session_id) != session_id:
            raise ValueError("stream_session_id_mismatch")
        target, classification, reason = _resolve_run(
            conn,
            game_id=game_id,
            stream_session_id=session_id,
            explicit_run_id=str(row["game_run_id"] or ""),
        )
        if not target:
            outcome = "ambiguous" if classification == "AMBIGUOUS" else "skipped"
            _audit_run(conn, "game_progress_states", source_id, classification, outcome, reason, "", {})
            return
        claims = _progress_claims(payload)
        migrated, deduplicated = _migrate_run_claims(
            conn,
            run_id=target,
            source_store="game_progress_states",
            source_id=source_id,
            observed_at=_epoch(row["updated_at"]),
            confidence=_confidence(payload.get("confidence", 0.0)),
            claims=claims,
        )
        conn.execute(
            "UPDATE game_progress_states SET game_run_id=? WHERE game_id=? AND stream_session_id=?",
            (target, game_id, session_id),
        )
        outcome = "migrated" if migrated else "deduplicated"
        _audit_run(
            conn,
            "game_progress_states",
            source_id,
            classification,
            outcome,
            "verified_session_link",
            target,
            {"migrated_fields": migrated, "deduplicated_fields": deduplicated},
        )
    except Exception as exc:
        _audit_run(
            conn,
            "game_progress_states",
            source_id,
            "AMBIGUOUS",
            "error",
            str(exc)[:160] or type(exc).__name__,
            "",
            {},
        )


def _migrate_session_row(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    source_id = str(row["id"])
    game_id = _normalize_game(str(row["game_key"] or row["game"] or ""))
    try:
        target, classification, reason = _resolve_run(
            conn,
            game_id=game_id,
            stream_session_id="",
            explicit_run_id=str(row["game_run_id"] or ""),
        )
        if not target:
            _audit_run(
                conn,
                "game_sessions",
                source_id,
                classification,
                "ambiguous" if classification == "AMBIGUOUS" else "skipped",
                reason,
                "",
                {},
            )
            return
        claims = []
        location = str(row["current_location"] or "").strip() if "current_location" in row.keys() else ""
        objective = next((str(row[column] or "").strip() for column in ("current_objective","next_time_plan") if column in row.keys() and str(row[column] or "").strip()),"")
        progress = next((str(row[column] or "").strip() for column in ("end_summary","start_summary") if column in row.keys() and str(row[column] or "").strip()),"")
        if location:claims.append(("current_location",location))
        if objective:claims.append(("current_objective",objective))
        if progress:claims.append(("last_confirmed_progress",progress))
        migrated, deduplicated = _migrate_run_claims(
            conn,
            run_id=target,
            source_store="game_sessions",
            source_id=source_id,
            observed_at=_epoch(row["updated_at"]),
            confidence=0.6,
            claims=claims,
        )
        _audit_run(
            conn,
            "game_sessions",
            source_id,
            classification,
            "migrated" if migrated else "deduplicated",
            "explicit_run_link",
            target,
            {"migrated_fields": migrated, "deduplicated_fields": deduplicated},
        )
    except Exception as exc:
        _audit_run(conn, "game_sessions", source_id, "AMBIGUOUS", "error", str(exc)[:160], "", {})


def _resolve_run(
    conn: sqlite3.Connection,
    *,
    game_id: str,
    stream_session_id: str,
    explicit_run_id: str,
) -> tuple[str, str, str]:
    if explicit_run_id:
        row = conn.execute(
            "SELECT id,status FROM game_runs WHERE id=? AND game_id=?",
            (explicit_run_id, game_id),
        ).fetchone()
        if row is None:
            return "", "ORPHANED", "invalid_explicit_run_link"
        return explicit_run_id, _run_classification(str(row["status"])), "explicit_run_link"
    if not stream_session_id:
        return "", "ORPHANED", "no_run_or_session_link"
    if stream_session_id.casefold() == "current":
        return "", "AMBIGUOUS", "pseudo_session_current_has_no_stable_identity"
    rows = conn.execute(
        """SELECT DISTINCT runs.id,runs.status
           FROM game_run_sessions links
           JOIN game_runs runs ON runs.id=links.game_run_id
           WHERE links.stream_session_id=? AND runs.game_id=?""",
        (stream_session_id, game_id),
    ).fetchall()
    if not rows:
        return "", "ORPHANED", "no_matching_canonical_run_session"
    if len(rows) > 1:
        return "", "AMBIGUOUS", "multiple_matching_canonical_runs"
    return str(rows[0]["id"]), _run_classification(str(rows[0]["status"])), "verified_session_link"


def _run_classification(status: str) -> str:
    return "CURRENT_RUN_STATE" if status == "ACTIVE" else "HISTORICAL_RUN_STATE"


def _progress_claims(payload: dict[str, Any]) -> list[tuple[str, Any]]:
    claims: list[tuple[str, Any]] = []
    scalar = (
        ("current_chapter", "current_chapter"),
        ("current_area", "current_location"),
    )
    arrays = (
        ("known_party_members", "party_members"),
        ("encountered_characters", "encountered_characters"),
        ("encountered_bosses", "encountered_bosses"),
        ("unlocked_mechanics", "unlocked_mechanics"),
        ("recent_progress_markers", "progress_markers"),
    )
    for source, predicate in scalar:
        value = str(payload.get(source) or "").strip()
        if value:
            claims.append((predicate, value))
    for source, predicate in arrays:
        value = payload.get(source)
        if isinstance(value, list) and value:
            claims.append((predicate, value))
    return claims


def _migrate_run_claims(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    source_store: str,
    source_id: str,
    observed_at: float,
    confidence: float,
    claims: list[tuple[str, Any]],
) -> tuple[list[str], list[str]]:
    migrated: list[str] = []
    deduplicated: list[str] = []
    for predicate, value in claims:
        object_json = json.dumps(value, ensure_ascii=False, sort_keys=True)
        existing = conn.execute(
            """SELECT id FROM beliefs
               WHERE namespace='game_run' AND scope_kind='game_run' AND scope_id=?
                 AND subject_ref='run_state' AND predicate=? AND object_json=?
                 AND epistemic_status IN ('KNOWN','INFERRED','SUSPECTED') AND superseded_by=''
               LIMIT 1""",
            (run_id, predicate, object_json),
        ).fetchone()
        belief_id = str(existing["id"]) if existing else _stable_id(
            "belief_legacy_game_run_v1", source_store, source_id, predicate
        )
        if existing:
            deduplicated.append(predicate)
        else:
            conn.execute(
                """INSERT INTO beliefs(
                   id,namespace,scope_kind,scope_id,subject_ref,predicate,object_json,
                   epistemic_status,confidence,authority_class,created_at,last_confirmed_at,
                   valid_from,valid_until,relevance_until,superseded_by,owner_confirmed,
                   sensitivity,schema_version,retention_policy,version)
                   VALUES(?,'game_run','game_run',?,'run_state',?,?,'SUSPECTED',?,'legacy',?,?,?,0,0,'',0,'normal',1,'retain_history',1)""",
                (belief_id, run_id, predicate, object_json, confidence, observed_at, observed_at, observed_at),
            )
            migrated.append(predicate)
        subject_key = f"game_run|game_run|{run_id}|run_state|{predicate}"
        evidence_id = _stable_id("evidence_legacy_game_run_v1", source_store, source_id, predicate)
        conn.execute(
            """INSERT OR IGNORE INTO belief_evidence(
               id,belief_id,source_event_id,source_record_type,source_record_id,relation,weight,
               observed_at,extractor,extractor_version,literal_span_json,subject_key)
               VALUES(?,?,?,?,?,'SUPPORTS',?,?, 'game_run_state_canonicalization','v1',?,?)""",
            (
                evidence_id,
                belief_id,
                f"{source_store}:{source_id}",
                source_store,
                source_id,
                confidence,
                observed_at,
                json.dumps({"legacy_value": value}, ensure_ascii=False, sort_keys=True),
                subject_key,
            ),
        )
        if predicate in migrated:
            conn.execute(
                """INSERT OR IGNORE INTO game_run_events(
                   id,game_run_id,event_type,subject_ref,predicate,object_json,evidence_event_id,
                   belief_id,observed_at,epistemic_status,schema_version)
                   VALUES(?,?,'legacy_state_migrated','run_state',?,?,?,? ,?,'SUSPECTED',1)""",
                (
                    _stable_id("game_event_legacy_v1", source_store, source_id, predicate),
                    run_id,
                    predicate,
                    object_json,
                    f"{source_store}:{source_id}",
                    belief_id,
                    observed_at,
                ),
            )
    return migrated, deduplicated


def _migrate_game_knowledge(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS legacy_game_knowledge_migration_audit(
           migration_version INTEGER NOT NULL,source_game_id TEXT NOT NULL,classification TEXT NOT NULL,
           outcome TEXT NOT NULL,reason TEXT NOT NULL,target_fact_ids_json TEXT NOT NULL DEFAULT '[]',
           details_json TEXT NOT NULL DEFAULT '{}',migrated_at TEXT NOT NULL,
           PRIMARY KEY(migration_version,source_game_id))"""
    )
    if "game_dossiers" not in _tables(conn):
        return
    _require_columns(
        conn,
        "game_dossiers",
        {"game_id", "canonical_title", "dossier_json", "dossier_version", "v2_projection_version"},
    )
    for row in conn.execute("SELECT * FROM game_dossiers ORDER BY game_id").fetchall():
        _migrate_dossier(conn, row)


def _migrate_dossier(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    game_id = str(row["game_id"] or "").strip()
    try:
        payload = json.loads(row["dossier_json"] or "{}")
        if not isinstance(payload, dict):
            raise ValueError("dossier_json_not_object")
        claims = [str(value).strip() for value in payload.get("confirmed_general_mechanics") or [] if str(value).strip()]
        sources = [dict(value) for value in payload.get("sources") or [] if isinstance(value, dict)]
        supported: list[tuple[str, dict[str, Any]]] = []
        unsupported: list[str] = []
        for claim in claims:
            source = next((item for item in sources if str(item.get("claim") or "").strip() == claim), None)
            location = str((source or {}).get("url") or (source or {}).get("location") or (source or {}).get("source_location") or "").strip()
            excerpt = str((source or {}).get("excerpt") or (source or {}).get("supporting_excerpt") or "").strip()
            if source is None or not location or not excerpt:
                unsupported.append(claim)
            else:
                supported.append((claim, source))
        if unsupported:
            _audit_knowledge(
                conn,
                game_id,
                "GAME_KNOWLEDGE",
                "ambiguous",
                "claims_missing_claim_level_provenance",
                [],
                {"unsupported_claims": unsupported},
            )
            return
        fact_ids: list[str] = []
        canonical_name = str(row["canonical_title"] or game_id)
        conn.execute(
            """INSERT OR IGNORE INTO game_identities(
               game_id,canonical_name,aliases_json,platform_ids_json,series,schema_version)
               VALUES(?,?,?,'{}','',1)""",
            (game_id, canonical_name, json.dumps(payload.get("aliases") or [canonical_name], ensure_ascii=False)),
        )
        for index, (claim, source) in enumerate(supported):
            belief_id = _stable_id("belief_legacy_game_knowledge_v1", game_id, str(index), claim)
            fact_id = _stable_id("game_fact_legacy_dossier_v1", game_id, str(index), claim)
            observed_at = _epoch(row["updated_at"] if "updated_at" in row.keys() else 0)
            conn.execute(
                """INSERT OR IGNORE INTO beliefs(
                   id,namespace,scope_kind,scope_id,subject_ref,predicate,object_json,
                   epistemic_status,confidence,authority_class,created_at,last_confirmed_at,
                   valid_from,valid_until,relevance_until,superseded_by,owner_confirmed,
                   sensitivity,schema_version,retention_policy,version)
                   VALUES(?,'game_knowledge','game',?,'game','general_mechanic',?,'INFERRED',?,'legacy_validated',?,?,?,0,0,'',0,'normal',1,'retain_history',1)""",
                (
                    belief_id,
                    game_id,
                    json.dumps(claim, ensure_ascii=False),
                    _confidence(source.get("confidence", 0.7)),
                    observed_at,
                    observed_at,
                    observed_at,
                ),
            )
            location = str(source.get("url") or source.get("location") or source.get("source_location"))
            excerpt = str(source.get("excerpt") or source.get("supporting_excerpt"))
            conn.execute(
                """INSERT OR IGNORE INTO belief_evidence(
                   id,belief_id,source_event_id,source_record_type,source_record_id,relation,weight,
                   observed_at,extractor,extractor_version,literal_span_json,subject_key)
                   VALUES(?,?,?,?,?,'SUPPORTS',?,?, 'game_knowledge_canonicalization','v1',?,?)""",
                (
                    _stable_id("evidence_legacy_game_knowledge_v1", game_id, str(index), claim),
                    belief_id,
                    f"game_dossier:{game_id}:{index}",
                    "game_dossiers",
                    game_id,
                    _confidence(source.get("confidence", 0.7)),
                    observed_at,
                    json.dumps({"source_url": location, "excerpt": excerpt}, ensure_ascii=False),
                    f"game_knowledge|game|{game_id}|game|general_mechanic",
                ),
            )
            conn.execute(
                """INSERT OR IGNORE INTO game_knowledge_facts(
                   id,game_id,belief_id,source_type,source_quality,spoiler_class,dossier_link,
                   version_tag,created_at,schema_version)
                   VALUES(?,?,?,?,?,'safe_general_mechanic',?,?,?,1)""",
                (
                    fact_id,
                    game_id,
                    belief_id,
                    str(source.get("source_type") or "legacy_dossier"),
                    str(source.get("source_quality") or "legacy_validated"),
                    f"game_dossiers:{game_id}",
                    f"dossier_v{int(row['dossier_version'] or 0)}",
                    observed_at,
                ),
            )
            fact_ids.append(fact_id)
        conn.execute("UPDATE game_dossiers SET v2_projection_version=1 WHERE game_id=?", (game_id,))
        _audit_knowledge(
            conn,
            game_id,
            "GAME_KNOWLEDGE",
            "migrated" if fact_ids else "skipped",
            "supported_claims_migrated" if fact_ids else "no_semantic_claims",
            fact_ids,
            {
                "dossier_version": int(row["dossier_version"] or 0),
                "non_claim_metadata_preserved_in_legacy_row": True,
            },
        )
    except Exception as exc:
        _audit_knowledge(conn, game_id, "AMBIGUOUS", "error", str(exc)[:160], [], {})


def _create_run_audit(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS legacy_game_run_state_migration_audit(
           migration_version INTEGER NOT NULL,source_store TEXT NOT NULL,source_record_id TEXT NOT NULL,
           classification TEXT NOT NULL,outcome TEXT NOT NULL,reason TEXT NOT NULL,
           target_run_id TEXT NOT NULL DEFAULT '',details_json TEXT NOT NULL DEFAULT '{}',
           migrated_at TEXT NOT NULL,PRIMARY KEY(migration_version,source_store,source_record_id))"""
    )


def _audit_run(
    conn: sqlite3.Connection,
    source_store: str,
    source_id: str,
    classification: str,
    outcome: str,
    reason: str,
    target_run_id: str,
    details: dict[str, Any],
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO legacy_game_run_state_migration_audit(
           migration_version,source_store,source_record_id,classification,outcome,reason,
           target_run_id,details_json,migrated_at) VALUES(?,?,?,?,?,?,?,?,?)""",
        (
            RUN_MIGRATION_VERSION,
            source_store,
            source_id,
            classification,
            outcome,
            reason,
            target_run_id,
            json.dumps(details, ensure_ascii=False, sort_keys=True),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _audit_knowledge(
    conn: sqlite3.Connection,
    game_id: str,
    classification: str,
    outcome: str,
    reason: str,
    fact_ids: list[str],
    details: dict[str, Any],
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO legacy_game_knowledge_migration_audit(
           migration_version,source_game_id,classification,outcome,reason,target_fact_ids_json,
           details_json,migrated_at) VALUES(?,?,?,?,?,?,?,?)""",
        (
            KNOWLEDGE_MIGRATION_VERSION,
            game_id,
            classification,
            outcome,
            reason,
            json.dumps(fact_ids),
            json.dumps(details, ensure_ascii=False, sort_keys=True),
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _require_columns(conn: sqlite3.Connection, table: str, required: set[str]) -> None:
    columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(f"unsupported {table} schema; missing columns: {', '.join(missing)}")


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _epoch(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()


def _normalize_game(value: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()
    return "_".join(re.findall(r"[a-z0-9]+", text)) or "unknown_game"


def _stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, '|'.join(parts)).hex}"
