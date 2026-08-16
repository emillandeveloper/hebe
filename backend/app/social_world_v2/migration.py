from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.replay.migrations import Migration


IDENTITY_COMPONENT = "social_identity_canonicalization"
IDENTITY_VERSION = 1
SUMMARY_COMPONENT = "social_summary_canonicalization"
SUMMARY_VERSION = 1


def social_identity_canonicalization_migrations() -> tuple[Migration, ...]:
    return (
        Migration(
            IDENTITY_COMPONENT,
            IDENTITY_VERSION,
            "verified_legacy_profiles_to_social_world",
            _migrate_identities,
        ),
    )


def social_summary_canonicalization_migrations() -> tuple[Migration, ...]:
    return (
        Migration(
            SUMMARY_COMPONENT,
            SUMMARY_VERSION,
            "verified_legacy_summaries_to_social_world",
            _migrate_summaries,
        ),
    )


def _migrate_identities(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    _require_columns(conn, "people", {"person_id", "created_at", "last_seen_at"})
    _require_columns(
        conn,
        "person_identities",
        {
            "id", "person_id", "platform", "platform_user_id", "login",
            "display_name", "aliases_json", "first_seen_at", "last_seen_at",
            "confidence", "source",
        },
    )
    _ensure_presence_schema(conn)
    _create_identity_audit(conn)
    if "chatter_profiles" not in _tables(conn):
        return
    _require_columns(
        conn,
        "chatter_profiles",
        {
            "username", "display_name", "aliases_json", "first_seen_at",
            "last_seen_at", "updated_at",
        },
    )
    if "chatter_presence" in _tables(conn):
        _require_columns(
            conn,
            "chatter_presence",
            {
                "id", "stream_session_id", "username", "first_seen_at",
                "last_seen_at", "last_message_at", "last_direct_interaction_at",
                "message_count", "direct_interaction_count", "presence_source_json",
            },
        )
    if "viewer_promotion_profiles" in _tables(conn):
        _require_columns(
            conn,
            "viewer_promotion_profiles",
            {
                "twitch_user_id", "current_login", "known_aliases_json",
                "created_by", "owner_locked",
            },
        )
    for row in conn.execute("SELECT * FROM chatter_profiles ORDER BY username").fetchall():
        _migrate_profile(conn, row)


def _migrate_profile(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    login = _login(row["username"])
    source_id = _profile_source_id(login)
    try:
        candidates = _identity_candidates(conn, login)
        verified_ids = _verified_promotion_ids(conn, login)
        classification = ""
        reason = ""
        target: sqlite3.Row | None = None

        if len(candidates) > 1:
            classification, reason = "CONFLICT", "multiple_modern_people_for_login_or_alias"
        elif len(verified_ids) > 1:
            classification, reason = "CONFLICT", "multiple_owner_verified_stable_ids"
        elif candidates:
            candidate = candidates[0]
            stable_id = str(candidate["platform_user_id"] or "")
            verified_id = verified_ids[0] if verified_ids else ""
            if stable_id and verified_id and stable_id != verified_id:
                classification, reason = "CONFLICT", "modern_and_owner_verified_stable_id_disagree"
            elif not stable_id and verified_id:
                classification, reason = "CONFLICT", "unverified_modern_person_requires_manual_merge"
            elif stable_id and (
                verified_id == stable_id or _same_observation_evidence(conn, login, candidate)
            ):
                target = candidate
                exact = _login(candidate["login"]) == login
                classification = "ALREADY_CANONICAL" if exact else "ALIAS_UPDATE"
                reason = "owner_verified_stable_id" if verified_id else "same_session_event_and_stable_id"
            elif stable_id:
                classification, reason = "AMBIGUOUS", "name_match_without_deterministic_event_evidence"
            else:
                classification, reason = "ORPHANED", "modern_candidate_has_no_stable_identity"
        elif verified_ids:
            stable_id = verified_ids[0]
            target = conn.execute(
                "SELECT * FROM person_identities WHERE platform='twitch' AND platform_user_id=?",
                (stable_id,),
            ).fetchone()
            classification = "ALIAS_UPDATE" if target else "SAFE_TO_MIGRATE"
            reason = "owner_locked_promotion_identity"
        else:
            classification, reason = "ORPHANED", "no_verified_stable_identity"

        if classification in {"CONFLICT", "AMBIGUOUS", "ORPHANED"}:
            outcome = "conflict" if classification == "CONFLICT" else "ambiguous" if classification == "AMBIGUOUS" else "skipped"
            _audit_identity(conn, source_id, classification, outcome, reason, "", {})
            return

        if target is None:
            target = _create_verified_identity(
                conn,
                stable_id=verified_ids[0],
                login=login,
                display_name=str(row["display_name"] or login),
                first_seen=_epoch(row["first_seen_at"]),
                last_seen=_epoch(row["last_seen_at"]),
            )
        _merge_profile_history(conn, target, row)
        presence = _migrate_profile_presence(conn, login, str(target["person_id"]))
        _audit_identity(
            conn,
            source_id,
            classification,
            "migrated" if classification in {"SAFE_TO_MIGRATE", "ALIAS_UPDATE"} else "deduplicated",
            reason,
            str(target["person_id"]),
            {
                "presence_rows_migrated": presence[0],
                "presence_rows_deduplicated": presence[1],
                "stable_identity_preserved": True,
            },
        )
    except Exception as exc:
        _audit_identity(conn, source_id, "AMBIGUOUS", "error", str(exc)[:180], "", {})


def _migrate_summaries(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    _require_columns(conn, "people", {"person_id"})
    _ensure_summary_schema(conn)
    _create_summary_audit(conn)
    if "stream_chatter_summaries" not in _tables(conn):
        return
    _require_columns(
        conn,
        "stream_chatter_summaries",
        {
            "id", "stream_session_id", "username", "summary_text", "topics_json",
            "message_count", "direct_interaction_count", "created_at",
            "notable_quotes_json", "inferred_facts_json",
        },
    )
    if "legacy_social_identity_migration_audit" not in _tables(conn):
        raise RuntimeError("social identity migration audit is required before summary migration")
    for row in conn.execute("SELECT * FROM stream_chatter_summaries ORDER BY id").fetchall():
        _migrate_summary(conn, row)


def _migrate_summary(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    source_id = str(row["id"])
    profile_id = _profile_source_id(_login(row["username"]))
    try:
        identity_audit = conn.execute(
            """SELECT classification,target_person_id
               FROM legacy_social_identity_migration_audit
               WHERE migration_version=? AND source_profile_id=?""",
            (IDENTITY_VERSION, profile_id),
        ).fetchone()
        if identity_audit is None:
            _audit_summary(conn, source_id, "ORPHANED", "skipped", "profile_not_audited", "", {})
            return
        classification = str(identity_audit["classification"])
        person_id = str(identity_audit["target_person_id"] or "")
        if classification in {"CONFLICT", "AMBIGUOUS"}:
            _audit_summary(conn, source_id, "AMBIGUOUS_OWNER", "ambiguous", "profile_identity_not_safe", "", {})
            return
        if not person_id:
            _audit_summary(conn, source_id, "KEEP_LEGACY_REFERENCE", "skipped", "profile_has_no_canonical_identity", "", {})
            return
        summary_id = _stable_id("social_summary_legacy_v1", source_id)
        topics = _json_list(row["topics_json"])
        cursor = conn.execute(
            """INSERT OR IGNORE INTO social_summaries(
               id,person_id,stream_session_id,source_type,source_record_id,summary_text,
               topics_json,message_count,direct_interaction_count,created_at,schema_version)
               VALUES(?,?,?,'legacy_stream_chatter_summary',?,?,?,?,?,?,1)""",
            (
                summary_id,
                person_id,
                str(row["stream_session_id"]),
                source_id,
                str(row["summary_text"] or "")[:500],
                json.dumps(topics, ensure_ascii=False),
                max(0, int(row["message_count"] or 0)),
                max(0, int(row["direct_interaction_count"] or 0)),
                _epoch(row["created_at"]),
            ),
        )
        _audit_summary(
            conn,
            source_id,
            "MIGRATABLE",
            "migrated" if cursor.rowcount == 1 else "deduplicated",
            "canonical_profile_link",
            person_id,
            {
                "target_summary_id": summary_id,
                "raw_quotes_copied": False,
                "inferred_facts_copied": False,
            },
        )
    except Exception as exc:
        _audit_summary(conn, source_id, "AMBIGUOUS_OWNER", "error", str(exc)[:180], "", {})


def _ensure_presence_schema(conn: sqlite3.Connection) -> None:
    columns = _columns(conn, "person_sessions")
    additions = {
        "first_message_at": "REAL NOT NULL DEFAULT 0",
        "last_message_at": "REAL NOT NULL DEFAULT 0",
        "last_direct_interaction_at": "REAL NOT NULL DEFAULT 0",
        "message_count": "INTEGER NOT NULL DEFAULT 0",
        "direct_interaction_count": "INTEGER NOT NULL DEFAULT 0",
        "presence_sources_json": "TEXT NOT NULL DEFAULT '[]'",
    }
    for name, definition in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE person_sessions ADD COLUMN {name} {definition}")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS social_presence_events(
          id TEXT PRIMARY KEY, person_id TEXT NOT NULL, stream_session_id TEXT NOT NULL,
          observed_at REAL NOT NULL, source TEXT NOT NULL, message_count INTEGER NOT NULL DEFAULT 0,
          direct_interaction_count INTEGER NOT NULL DEFAULT 0, schema_version INTEGER NOT NULL DEFAULT 1,
          FOREIGN KEY(person_id) REFERENCES people(person_id)
        );
        CREATE INDEX IF NOT EXISTS idx_social_presence_person_session
          ON social_presence_events(person_id,stream_session_id,observed_at);
        """
    )


def _ensure_summary_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS social_summaries(
          id TEXT PRIMARY KEY, person_id TEXT NOT NULL, stream_session_id TEXT NOT NULL,
          source_type TEXT NOT NULL, source_record_id TEXT NOT NULL, summary_text TEXT NOT NULL,
          topics_json TEXT NOT NULL DEFAULT '[]', message_count INTEGER NOT NULL DEFAULT 0,
          direct_interaction_count INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
          schema_version INTEGER NOT NULL DEFAULT 1,
          FOREIGN KEY(person_id) REFERENCES people(person_id),
          UNIQUE(source_type,source_record_id)
        );
        CREATE INDEX IF NOT EXISTS idx_social_summary_person_created
          ON social_summaries(person_id,created_at DESC);
        """
    )


def _create_identity_audit(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS legacy_social_identity_migration_audit(
           migration_version INTEGER NOT NULL,source_profile_id TEXT NOT NULL,
           classification TEXT NOT NULL,outcome TEXT NOT NULL,reason TEXT NOT NULL,
           target_person_id TEXT NOT NULL DEFAULT '',details_json TEXT NOT NULL DEFAULT '{}',
           migrated_at TEXT NOT NULL,PRIMARY KEY(migration_version,source_profile_id))"""
    )


def _create_summary_audit(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS legacy_social_summary_migration_audit(
           migration_version INTEGER NOT NULL,source_summary_id TEXT NOT NULL,
           classification TEXT NOT NULL,outcome TEXT NOT NULL,reason TEXT NOT NULL,
           target_person_id TEXT NOT NULL DEFAULT '',details_json TEXT NOT NULL DEFAULT '{}',
           migrated_at TEXT NOT NULL,PRIMARY KEY(migration_version,source_summary_id))"""
    )


def _identity_candidates(conn: sqlite3.Connection, login: str) -> list[sqlite3.Row]:
    result: list[sqlite3.Row] = []
    for row in conn.execute("SELECT * FROM person_identities WHERE platform='twitch'").fetchall():
        aliases = {_login(value) for value in _json_list(row["aliases_json"])}
        if _login(row["login"]) == login or login in aliases:
            result.append(row)
    return result


def _verified_promotion_ids(conn: sqlite3.Connection, login: str) -> list[str]:
    if "viewer_promotion_profiles" not in _tables(conn):
        return []
    result: set[str] = set()
    for row in conn.execute(
        "SELECT * FROM viewer_promotion_profiles WHERE owner_locked=1 AND created_by='owner_command'"
    ).fetchall():
        names = {_login(row["current_login"]), *(_login(x) for x in _json_list(row["known_aliases_json"]))}
        stable_id = str(row["twitch_user_id"] or "").strip()
        if login in names and stable_id:
            result.add(stable_id)
    return sorted(result)


def _same_observation_evidence(conn: sqlite3.Connection, login: str, identity: sqlite3.Row) -> bool:
    if "chatter_presence" not in _tables(conn):
        return False
    sessions = {
        str(row[0])
        for row in conn.execute(
            "SELECT stream_session_id FROM person_sessions WHERE person_id=?",
            (identity["person_id"],),
        ).fetchall()
    }
    if not sessions:
        return False
    first_seen = float(identity["first_seen_at"] or 0)
    for row in conn.execute(
        "SELECT stream_session_id,first_seen_at FROM chatter_presence WHERE lower(username)=lower(?)",
        (login,),
    ).fetchall():
        if str(row["stream_session_id"]) in sessions and abs(_epoch(row["first_seen_at"]) - first_seen) <= 2.0:
            return True
    return False


def _create_verified_identity(
    conn: sqlite3.Connection,
    *,
    stable_id: str,
    login: str,
    display_name: str,
    first_seen: float,
    last_seen: float,
) -> sqlite3.Row:
    person_id = _stable_id("person_legacy_social_v1", "twitch", stable_id)
    identity_id = _stable_id("identity_legacy_social_v1", "twitch", stable_id)
    conn.execute(
        "INSERT OR IGNORE INTO people(person_id,created_at,last_seen_at,scope,schema_version) VALUES(?,?,?,'stream_public',1)",
        (person_id, first_seen, last_seen),
    )
    conn.execute(
        """INSERT OR IGNORE INTO person_identities(
           id,person_id,platform,platform_user_id,login,display_name,aliases_json,
           first_seen_at,last_seen_at,confidence,source,schema_version)
           VALUES(?,?,'twitch',?,?,?,?,?,?,1.0,'legacy_owner_verified',1)""",
        (identity_id, person_id, stable_id, login, display_name or login, json.dumps([login]), first_seen, last_seen),
    )
    return conn.execute("SELECT * FROM person_identities WHERE platform='twitch' AND platform_user_id=?", (stable_id,)).fetchone()


def _merge_profile_history(conn: sqlite3.Connection, target: sqlite3.Row, profile: sqlite3.Row) -> None:
    login = _login(profile["username"])
    aliases = {_login(value) for value in _json_list(target["aliases_json"]) if _login(value)}
    aliases.update(_login(value) for value in _json_list(profile["aliases_json"]) if _login(value))
    aliases.add(login)
    first_seen = min(float(target["first_seen_at"] or 0), _epoch(profile["first_seen_at"]))
    last_seen = max(float(target["last_seen_at"] or 0), _epoch(profile["last_seen_at"]))
    display_name = (
        str(profile["display_name"] or target["display_name"] or login)
        if _login(target["login"]) == login
        else str(target["display_name"] or target["login"])
    )
    conn.execute(
        """UPDATE person_identities SET display_name=?,aliases_json=?,first_seen_at=?,last_seen_at=?
           WHERE id=?""",
        (
            display_name,
            json.dumps(sorted(aliases), ensure_ascii=False),
            first_seen,
            last_seen,
            target["id"],
        ),
    )
    conn.execute(
        "UPDATE people SET created_at=min(created_at,?),last_seen_at=max(last_seen_at,?) WHERE person_id=?",
        (first_seen, last_seen, target["person_id"]),
    )


def _migrate_profile_presence(conn: sqlite3.Connection, login: str, person_id: str) -> tuple[int, int]:
    if "chatter_presence" not in _tables(conn):
        return 0, 0
    migrated = 0
    deduplicated = 0
    for row in conn.execute(
        "SELECT * FROM chatter_presence WHERE lower(username)=lower(?) ORDER BY id",
        (login,),
    ).fetchall():
        created = _record_presence(
            conn,
            event_id=_stable_id("social_presence_legacy_v1", str(row["id"])),
            person_id=person_id,
            stream_session_id=str(row["stream_session_id"] or ""),
            first_seen=_epoch(row["first_seen_at"]),
            last_seen=_epoch(row["last_seen_at"]),
            first_message=_epoch(row["first_message_at"]),
            last_message=_epoch(row["last_message_at"]),
            last_direct=_epoch(row["last_direct_interaction_at"]),
            message_count=max(0, int(row["message_count"] or 0)),
            direct_count=max(0, int(row["direct_interaction_count"] or 0)),
            sources=_json_list(row["presence_source_json"]),
        )
        migrated += int(created)
        deduplicated += int(not created)
    return migrated, deduplicated


def _record_presence(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    person_id: str,
    stream_session_id: str,
    first_seen: float,
    last_seen: float,
    first_message: float,
    last_message: float,
    last_direct: float,
    message_count: int,
    direct_count: int,
    sources: list[Any],
) -> bool:
    source_names = sorted({str(value) for value in sources if str(value).strip()}) or ["legacy_presence"]
    cursor = conn.execute(
        """INSERT OR IGNORE INTO social_presence_events(
           id,person_id,stream_session_id,observed_at,source,message_count,direct_interaction_count,schema_version)
           VALUES(?,?,?,?,?,?,?,1)""",
        (event_id, person_id, stream_session_id, last_seen or first_seen, ",".join(source_names), message_count, direct_count),
    )
    if cursor.rowcount != 1:
        return False
    existing = conn.execute(
        "SELECT * FROM person_sessions WHERE person_id=? AND stream_session_id=?",
        (person_id, stream_session_id),
    ).fetchone()
    if existing is None:
        conn.execute(
            """INSERT INTO person_sessions(
               person_id,stream_session_id,first_seen_at,last_seen_at,first_message_at,last_message_at,
               last_direct_interaction_at,message_count,direct_interaction_count,presence_sources_json)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                person_id, stream_session_id, first_seen, last_seen, first_message, last_message,
                last_direct, message_count, direct_count, json.dumps(source_names, ensure_ascii=False),
            ),
        )
    else:
        combined = sorted(set(_json_list(existing["presence_sources_json"])) | set(source_names))
        conn.execute(
            """UPDATE person_sessions SET
               first_seen_at=?,last_seen_at=?,first_message_at=?,last_message_at=?,
               last_direct_interaction_at=?,message_count=message_count+?,
               direct_interaction_count=direct_interaction_count+?,presence_sources_json=?
               WHERE person_id=? AND stream_session_id=?""",
            (
                _min_positive(float(existing["first_seen_at"] or 0), first_seen),
                max(float(existing["last_seen_at"] or 0), last_seen),
                _min_positive(float(existing["first_message_at"] or 0), first_message),
                max(float(existing["last_message_at"] or 0), last_message),
                max(float(existing["last_direct_interaction_at"] or 0), last_direct),
                message_count,
                direct_count,
                json.dumps(combined, ensure_ascii=False),
                person_id,
                stream_session_id,
            ),
        )
    return True


def _audit_identity(
    conn: sqlite3.Connection,
    source_id: str,
    classification: str,
    outcome: str,
    reason: str,
    target_person_id: str,
    details: dict[str, Any],
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO legacy_social_identity_migration_audit(
           migration_version,source_profile_id,classification,outcome,reason,target_person_id,
           details_json,migrated_at) VALUES(?,?,?,?,?,?,?,?)""",
        (
            IDENTITY_VERSION, source_id, classification, outcome, reason, target_person_id,
            json.dumps(details, ensure_ascii=False, sort_keys=True), _now_iso(),
        ),
    )


def _audit_summary(
    conn: sqlite3.Connection,
    source_id: str,
    classification: str,
    outcome: str,
    reason: str,
    target_person_id: str,
    details: dict[str, Any],
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO legacy_social_summary_migration_audit(
           migration_version,source_summary_id,classification,outcome,reason,target_person_id,
           details_json,migrated_at) VALUES(?,?,?,?,?,?,?,?)""",
        (
            SUMMARY_VERSION, source_id, classification, outcome, reason, target_person_id,
            json.dumps(details, ensure_ascii=False, sort_keys=True), _now_iso(),
        ),
    )


def _profile_source_id(login: str) -> str:
    return "legacy_profile_" + hashlib.sha256(login.encode("utf-8")).hexdigest()[:20]


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join((prefix, *map(str, parts))).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _login(value: Any) -> str:
    return str(value or "").strip().casefold().lstrip("@")


def _json_list(raw: Any) -> list[Any]:
    try:
        value = json.loads(str(raw or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _epoch(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def _min_positive(a: float, b: float) -> float:
    values = [value for value in (a, b) if value > 0]
    return min(values) if values else 0.0


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if table not in _tables(conn):
        raise RuntimeError(f"required table missing: {table}")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError("invalid table name")
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _require_columns(conn: sqlite3.Connection, table: str, required: set[str]) -> None:
    missing = required - _columns(conn, table)
    if missing:
        raise RuntimeError(f"unsupported {table} schema; missing {sorted(missing)}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
