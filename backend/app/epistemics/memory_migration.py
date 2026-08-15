from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.epistemics.models import BeliefStatus
from app.replay.migrations import Migration


LEGACY_MEMORY_MIGRATION_COMPONENT = "memory_canonicalization"
LEGACY_MEMORY_MIGRATION_VERSION = 1


class LegacyMemoryMappingError(ValueError):
    def __init__(self, reason: str, *, outcome: str = "error") -> None:
        super().__init__(reason)
        self.reason = reason
        self.outcome = outcome


@dataclass(frozen=True, slots=True)
class CanonicalMemoryClaim:
    namespace: str
    scope_kind: str
    scope_id: str
    subject_ref: str
    predicate: str
    object_value: Any
    mapping_class: str


DIRECT_KINDS = {
    "preference": ("memory.preference", "owner_local", "leo", "preference"),
    "leo_fact": ("memory.owner_fact", "owner_local", "leo", "fact"),
    "hebe_identity": ("memory.hebe_identity", "assistant_local", "hebe", "identity"),
    "project_fact": ("memory.project", "owner_local", "leo", "fact"),
    "habit": ("memory.habit", "owner_local", "leo", "habit"),
    "viewer_fact": ("memory.viewer", "viewer_local", "", "fact"),
}
TRANSFORM_KINDS = {"appointment", "fact", "person", "stream_fact"}
OBSOLETE_KINDS = {"task"}
AMBIGUOUS_KINDS = {"misc"}
KNOWN_LEGACY_KINDS = frozenset(DIRECT_KINDS) | TRANSFORM_KINDS | OBSOLETE_KINDS | AMBIGUOUS_KINDS


def map_memory_fact(
    *, kind: str, subject: str | None, payload: Any, source_text: str | None,
) -> CanonicalMemoryClaim:
    normalized_kind = str(kind or "").strip().casefold()
    normalized_subject = str(subject or "").strip()
    text = str(source_text or "").strip()
    if normalized_kind in OBSOLETE_KINDS:
        raise LegacyMemoryMappingError("obsolete_kind", outcome="skipped")
    if normalized_kind in AMBIGUOUS_KINDS:
        raise LegacyMemoryMappingError("ambiguous_kind", outcome="unsupported")
    if normalized_kind not in KNOWN_LEGACY_KINDS:
        raise LegacyMemoryMappingError("unknown_kind", outcome="unsupported")
    if not isinstance(payload, dict):
        raise LegacyMemoryMappingError("payload_not_object")

    if normalized_kind == "appointment":
        due_at = str(payload.get("due_at") or "").strip()
        title = str(payload.get("title") or normalized_subject).strip()
        if not due_at or not title:
            raise LegacyMemoryMappingError("appointment_missing_title_or_due_at", outcome="unsupported")
        object_value = dict(payload)
        object_value["title"] = title
        object_value["due_at"] = due_at
        return CanonicalMemoryClaim(
            namespace="memory.appointment", scope_kind="owner_local", scope_id="leo",
            subject_ref=normalized_subject or title, predicate="scheduled_appointment",
            object_value=object_value, mapping_class="TRANSFORM_REQUIRED",
        )

    if normalized_kind in {"fact", "person"}:
        predicate = str(payload.get("predicate") or "").strip()
        if not predicate or "value" not in payload or not normalized_subject:
            raise LegacyMemoryMappingError("explicit_predicate_value_required", outcome="unsupported")
        namespace = "memory.fact" if normalized_kind == "fact" else "memory.person"
        return CanonicalMemoryClaim(
            namespace=namespace, scope_kind="owner_local", scope_id="leo",
            subject_ref=normalized_subject, predicate=predicate, object_value=payload["value"],
            mapping_class="TRANSFORM_REQUIRED",
        )

    if normalized_kind == "stream_fact":
        source_context = str(payload.get("source_context") or payload.get("source") or "").strip().casefold()
        channel = str(payload.get("channel") or "").strip()
        if not channel and source_context not in {"stream", "twitch", "stream_public"}:
            raise LegacyMemoryMappingError("stream_scope_evidence_required", outcome="unsupported")
        if not normalized_subject:
            raise LegacyMemoryMappingError("subject_required")
        predicate = str(payload.get("predicate") or "fact").strip()
        object_value = payload.get("value") if payload.get("predicate") and "value" in payload else payload
        return CanonicalMemoryClaim(
            namespace="memory.stream", scope_kind="stream_public",
            scope_id=channel or "legacy_stream", subject_ref=normalized_subject,
            predicate=predicate, object_value=object_value,
            mapping_class="TRANSFORM_REQUIRED",
        )

    namespace, scope_kind, default_scope_id, default_predicate = DIRECT_KINDS[normalized_kind]
    if not normalized_subject:
        raise LegacyMemoryMappingError("subject_required")
    scope_id = normalized_subject if normalized_kind == "viewer_fact" else default_scope_id
    predicate = str(payload.get("predicate") or default_predicate).strip()
    if "value" in payload and payload.get("predicate"):
        object_value = payload["value"]
    elif payload:
        object_value = payload
    elif text:
        object_value = {"text": text}
    else:
        raise LegacyMemoryMappingError("empty_memory")
    return CanonicalMemoryClaim(
        namespace=namespace, scope_kind=scope_kind, scope_id=scope_id,
        subject_ref=normalized_subject, predicate=predicate, object_value=object_value,
        mapping_class="DIRECT_MAPPING",
    )


def legacy_memory_fact_migrations() -> tuple[Migration, ...]:
    return (
        Migration(
            LEGACY_MEMORY_MIGRATION_COMPONENT,
            LEGACY_MEMORY_MIGRATION_VERSION,
            "memory_facts_to_canonical_beliefs",
            _migrate_memory_facts,
        ),
    )


def _migrate_memory_facts(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS legacy_memory_fact_migration_audit (
            migration_version INTEGER NOT NULL,
            source_fact_id TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            outcome TEXT NOT NULL,
            reason TEXT NOT NULL,
            target_belief_id TEXT NOT NULL DEFAULT '',
            details_json TEXT NOT NULL DEFAULT '{}',
            migrated_at TEXT NOT NULL,
            PRIMARY KEY(migration_version, source_fact_id)
        )
        """
    )
    tables = {str(row[0]) for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if "memory_facts" not in tables:
        print("[HEBE][MEMORY_MIGRATION] version=1 status=no_legacy_table", flush=True)
        return

    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(memory_facts)")}
    required = {"id", "kind", "subject", "payload_json", "source_text", "confidence", "created_at", "updated_at", "active"}
    missing = sorted(required - columns)
    if missing:
        print(f"[HEBE][MEMORY_MIGRATION] version=1 status=unsupported_schema missing={missing!r}", flush=True)
        raise RuntimeError(f"unsupported memory_facts schema; missing columns: {', '.join(missing)}")

    counters = {"migrated": 0, "deduplicated": 0, "skipped": 0, "unsupported": 0, "error": 0}
    rows = conn.execute("SELECT * FROM memory_facts ORDER BY id").fetchall()
    for row in rows:
        fact_id = str(row["id"])
        kind = str(row["kind"] or "")
        try:
            linked_belief_id = str(row["belief_id"] or "") if "belief_id" in columns else ""
            if linked_belief_id and conn.execute(
                "SELECT 1 FROM beliefs WHERE id=?", (linked_belief_id,),
            ).fetchone():
                outcome = "deduplicated"
                _audit(conn, fact_id, kind, outcome, "existing_belief_link", linked_belief_id, {})
                counters[outcome] += 1
                continue
            payload = json.loads(row["payload_json"] or "{}")
            confidence = _confidence(row["confidence"])
            created_at = _epoch(row["created_at"], field="created_at")
            updated_at = _epoch(row["updated_at"], field="updated_at")
            claim = map_memory_fact(
                kind=kind, subject=row["subject"], payload=payload, source_text=row["source_text"],
            )
            status = _legacy_status(row, columns)
            existing_id = _find_equivalent(conn, claim)
            if existing_id:
                _insert_evidence(conn, existing_id, row, claim, updated_at, payload)
                outcome, reason, belief_id = "deduplicated", "equivalent_belief_exists", existing_id
            else:
                belief_id = f"belief_legacy_memory_fact_v1_{fact_id}"
                valid_until = updated_at if status == BeliefStatus.HISTORICAL else 0.0
                conn.execute(
                    """INSERT INTO beliefs(
                       id,namespace,scope_kind,scope_id,subject_ref,predicate,object_json,
                       epistemic_status,confidence,authority_class,created_at,last_confirmed_at,
                       valid_from,valid_until,relevance_until,superseded_by,owner_confirmed,
                       sensitivity,schema_version,retention_policy,version)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        belief_id, claim.namespace, claim.scope_kind, claim.scope_id,
                        claim.subject_ref, claim.predicate,
                        json.dumps(claim.object_value, ensure_ascii=False, sort_keys=True),
                        status.value, confidence, "legacy", created_at, updated_at, created_at,
                        valid_until, 0.0, "", 0, "normal", 1, "retain_history", 1,
                    ),
                )
                _insert_evidence(conn, belief_id, row, claim, updated_at, payload)
                outcome, reason = "migrated", claim.mapping_class.casefold()
            _audit(conn, fact_id, kind, outcome, reason, belief_id, {"mapping_class": claim.mapping_class})
        except LegacyMemoryMappingError as exc:
            outcome = exc.outcome
            _audit(conn, fact_id, kind, outcome, exc.reason, "", {})
        except Exception as exc:
            outcome = "error"
            _audit(conn, fact_id, kind, outcome, type(exc).__name__, "", {"message": str(exc)[:300]})
        counters[outcome] += 1
    print(
        "[HEBE][MEMORY_MIGRATION] version=1 "
        + " ".join(f"{key}={value}" for key, value in counters.items()),
        flush=True,
    )


def _legacy_status(row: sqlite3.Row, columns: set[str]) -> BeliefStatus:
    if not bool(row["active"]):
        return BeliefStatus.HISTORICAL
    if "epistemic_status" in columns:
        raw = str(row["epistemic_status"] or "").strip().upper()
        if raw in {BeliefStatus.INFERRED.value, BeliefStatus.SUSPECTED.value}:
            return BeliefStatus(raw)
    return BeliefStatus.SUSPECTED


def _confidence(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise LegacyMemoryMappingError("invalid_confidence")
    return number


def _epoch(value: Any, *, field: str) -> float:
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value or "").strip()
        if not text:
            raise LegacyMemoryMappingError(f"missing_{field}")
        try:
            number = float(text)
        except ValueError:
            normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            number = parsed.timestamp()
    if not math.isfinite(number):
        raise LegacyMemoryMappingError(f"invalid_{field}")
    return number


def _find_equivalent(conn: sqlite3.Connection, claim: CanonicalMemoryClaim) -> str:
    rows = conn.execute(
        """SELECT id,object_json FROM beliefs
           WHERE namespace=? AND scope_kind=? AND scope_id=? AND subject_ref=? AND predicate=?
             AND epistemic_status IN ('KNOWN','INFERRED','SUSPECTED') AND superseded_by=''""",
        (claim.namespace, claim.scope_kind, claim.scope_id, claim.subject_ref, claim.predicate),
    ).fetchall()
    return next(
        (str(row["id"]) for row in rows if json.loads(row["object_json"]) == claim.object_value),
        "",
    )


def _insert_evidence(
    conn: sqlite3.Connection, belief_id: str, row: sqlite3.Row,
    claim: CanonicalMemoryClaim, observed_at: float, payload: dict[str, Any],
) -> None:
    subject_key = "|".join((claim.namespace, claim.scope_kind, claim.scope_id, claim.subject_ref, claim.predicate))
    literal = {
        "text": str(row["source_text"] or ""),
        "legacy_payload": payload,
        "legacy_last_used_at": row["last_used_at"] if "last_used_at" in row.keys() else None,
        "legacy_active": bool(row["active"]),
    }
    conn.execute(
        """INSERT OR IGNORE INTO belief_evidence(
           id,belief_id,source_event_id,source_record_type,source_record_id,relation,weight,
           observed_at,extractor,extractor_version,literal_span_json,subject_key)
           VALUES(?,?,?,?,?,'SUPPORTS',?,?,?,?,?,?)""",
        (
            f"evidence_legacy_memory_fact_v1_{row['id']}", belief_id,
            f"memory_fact:{row['id']}", "memory_facts", str(row["id"]),
            _confidence(row["confidence"]), observed_at, "memory_canonicalization",
            "v1", json.dumps(literal, ensure_ascii=False, sort_keys=True), subject_key,
        ),
    )


def _audit(
    conn: sqlite3.Connection, fact_id: str, kind: str, outcome: str, reason: str,
    belief_id: str, details: dict[str, Any],
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO legacy_memory_fact_migration_audit
           (migration_version,source_fact_id,source_kind,outcome,reason,target_belief_id,details_json,migrated_at)
           VALUES(?,?,?,?,?,?,?,?)""",
        (
            LEGACY_MEMORY_MIGRATION_VERSION, fact_id, kind, outcome, reason, belief_id,
            json.dumps(details, ensure_ascii=False, sort_keys=True), datetime.now(timezone.utc).isoformat(),
        ),
    )
