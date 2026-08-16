from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Callable

from app.replay.migrations import Migration
from app.stream.behavior_constraints import BehaviorConstraint


def behavior_constraint_migrations() -> tuple[Migration, ...]:
    def create_tables(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS behavior_constraints (
                id TEXT PRIMARY KEY,
                actor TEXT NOT NULL,
                behavior_family TEXT NOT NULL,
                behavior_variants_json TEXT NOT NULL,
                recipient_scope TEXT NOT NULL,
                recipient_user_id TEXT NOT NULL DEFAULT '',
                recipient_login TEXT NOT NULL DEFAULT '',
                recipient_display_name TEXT NOT NULL DEFAULT '',
                requester_scope TEXT NOT NULL DEFAULT 'any',
                requester_user_id TEXT NOT NULL DEFAULT '',
                requester_login TEXT NOT NULL DEFAULT '',
                source_event_id TEXT NOT NULL DEFAULT '',
                created_by TEXT NOT NULL,
                authority TEXT NOT NULL,
                priority TEXT NOT NULL,
                scope TEXT NOT NULL,
                explicitness TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                retired_at REAL NOT NULL DEFAULT 0,
                retirement_reason TEXT NOT NULL DEFAULT '',
                version INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_behavior_constraints_active
                ON behavior_constraints(status, scope, behavior_family);
            CREATE TABLE IF NOT EXISTS behavior_constraint_events (
                id TEXT PRIMARY KEY,
                constraint_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                source_event_id TEXT NOT NULL DEFAULT '',
                authority TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                constraint_version INTEGER NOT NULL,
                FOREIGN KEY(constraint_id) REFERENCES behavior_constraints(id)
            );
            CREATE INDEX IF NOT EXISTS idx_behavior_constraint_events_constraint
                ON behavior_constraint_events(constraint_id, created_at);
            """
        )

    return (Migration("behavior_constraints", 1, "durable_owner_constraints", create_tables),)


class BehaviorConstraintRepository:
    """Canonical persistence for explicit durable owner behavior policy only."""

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self.connection_factory = connection_factory

    def save_durable(self, constraint: BehaviorConstraint) -> BehaviorConstraint:
        if constraint.scope != "durable":
            raise ValueError("only_durable_constraints_are_persisted")
        if constraint.authority != "owner" or constraint.explicitness != "explicit":
            raise ValueError("durable_constraint_requires_explicit_owner_authority")
        conn = self.connection_factory()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT version FROM behavior_constraints WHERE id=?", (constraint.id,),
            ).fetchone()
            version = int(existing[0]) + 1 if existing else max(1, int(constraint.version or 1))
            conn.execute(
                """
                INSERT INTO behavior_constraints (
                    id, actor, behavior_family, behavior_variants_json,
                    recipient_scope, recipient_user_id, recipient_login, recipient_display_name,
                    requester_scope, requester_user_id, requester_login, source_event_id,
                    created_by, authority, priority, scope, explicitness, confidence,
                    created_at, expires_at, status, reason, retired_at, retirement_reason, version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    behavior_variants_json=excluded.behavior_variants_json,
                    status='ACTIVE', retired_at=0, retirement_reason='', version=excluded.version
                """,
                (
                    constraint.id, constraint.actor, constraint.behavior_family,
                    json.dumps(constraint.behavior_variants, ensure_ascii=False),
                    constraint.recipient_scope, constraint.recipient_user_id,
                    constraint.recipient_login, constraint.recipient_display_name,
                    constraint.requester_scope, constraint.requester_user_id,
                    constraint.requester_login, constraint.source_event_id,
                    constraint.created_by, constraint.authority, constraint.priority,
                    constraint.scope, constraint.explicitness, float(constraint.confidence),
                    float(constraint.created_at), float(constraint.expires_at), "ACTIVE",
                    constraint.reason, 0.0, "", version,
                ),
            )
            self._event(
                conn, constraint.id, "CREATED" if existing is None else "REACTIVATED",
                constraint.source_event_id, constraint.authority, constraint.reason, version,
                now=constraint.created_at,
            )
            conn.commit()
            constraint.status = "ACTIVE"
            constraint.version = version
            constraint.active = True
            return constraint
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def list_active(self) -> list[BehaviorConstraint]:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM behavior_constraints WHERE status='ACTIVE' AND scope='durable' ORDER BY created_at, id"
            ).fetchall()
            return [self._from_row(row) for row in rows]
        finally:
            conn.close()

    def list_all(self) -> list[BehaviorConstraint]:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            return [self._from_row(row) for row in conn.execute(
                "SELECT * FROM behavior_constraints ORDER BY created_at, id"
            ).fetchall()]
        finally:
            conn.close()

    def events(self, constraint_id: str) -> list[dict]:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM behavior_constraint_events WHERE constraint_id=? ORDER BY created_at, rowid",
                (constraint_id,),
            ).fetchall()]
        finally:
            conn.close()

    def retire(
        self,
        constraint_id: str,
        *,
        reason: str,
        source_event_id: str = "",
        authority: str = "owner",
        now: float | None = None,
    ) -> bool:
        now = time.time() if now is None else float(now)
        conn = self.connection_factory()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status,version FROM behavior_constraints WHERE id=?", (constraint_id,),
            ).fetchone()
            if row is None or str(row[0]) != "ACTIVE":
                conn.rollback()
                return False
            version = int(row[1]) + 1
            conn.execute(
                "UPDATE behavior_constraints SET status='RETIRED',retired_at=?,retirement_reason=?,version=? WHERE id=?",
                (now, reason, version, constraint_id),
            )
            self._event(conn, constraint_id, "RETIRED", source_event_id, authority, reason, version, now=now)
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _event(
        conn: sqlite3.Connection,
        constraint_id: str,
        event_type: str,
        source_event_id: str,
        authority: str,
        reason: str,
        version: int,
        *,
        now: float | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO behavior_constraint_events(id,constraint_id,event_type,source_event_id,authority,reason,created_at,constraint_version) VALUES(?,?,?,?,?,?,?,?)",
            (
                f"constraint_event_{uuid.uuid4().hex}", constraint_id, event_type,
                source_event_id, authority, reason, time.time() if now is None else now, version,
            ),
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> BehaviorConstraint:
        return BehaviorConstraint(
            id=str(row["id"]), actor=str(row["actor"]),
            behavior_family=str(row["behavior_family"]),
            behavior_variants=list(json.loads(row["behavior_variants_json"] or "[]")),
            recipient_scope=str(row["recipient_scope"]),
            recipient_user_id=str(row["recipient_user_id"]),
            recipient_login=str(row["recipient_login"]),
            recipient_display_name=str(row["recipient_display_name"]),
            requester_scope=str(row["requester_scope"]),
            requester_user_id=str(row["requester_user_id"]),
            requester_login=str(row["requester_login"]),
            source_event_id=str(row["source_event_id"]), source_text="",
            created_by=str(row["created_by"]), authority=str(row["authority"]),
            priority=str(row["priority"]), scope=str(row["scope"]),
            explicitness=str(row["explicitness"]), confidence=float(row["confidence"]),
            created_at=float(row["created_at"]), expires_at=float(row["expires_at"]),
            active=str(row["status"]) == "ACTIVE", status=str(row["status"]),
            reason=str(row["reason"]), retired_at=float(row["retired_at"]),
            retirement_reason=str(row["retirement_reason"]), version=int(row["version"]),
        )


__all__ = ["BehaviorConstraintRepository", "behavior_constraint_migrations"]
