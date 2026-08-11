from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from typing import Callable

from app.continuity.models import (
    AttentionState,
    ConversationContext,
    ConversationStatus,
    CurrentConversation,
    ExpectedReply,
    OpenThread,
    OpenThreadStatus,
)


ACTIVE_CONVERSATION_STATUSES = {
    ConversationStatus.OPEN.value,
    ConversationStatus.WAITING_ON_LEO.value,
    ConversationStatus.WAITING_ON_HEBE.value,
}


class ConversationVersionConflict(RuntimeError):
    pass


class ConversationRepository:
    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self.connection_factory = connection_factory

    def open(self, conversation: CurrentConversation, *, interrupt_reason: str = "replaced_by_new_conversation") -> CurrentConversation:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in ACTIVE_CONVERSATION_STATUSES)
            conn.execute(
                f"""UPDATE conversations SET status=?, closure_reason=?, version=version+1
                    WHERE context_kind=? AND context_id=? AND status IN ({placeholders})""",
                (
                    ConversationStatus.INTERRUPTED.value, interrupt_reason,
                    conversation.context_kind.value, conversation.context_id,
                    *sorted(ACTIVE_CONVERSATION_STATUSES),
                ),
            )
            self._insert(conn, conversation)
            conn.commit()
            return conversation
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get(self, conversation_id: str) -> CurrentConversation | None:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
            return self._from_row(row) if row else None
        finally:
            conn.close()

    def get_active(self, context_kind: str, context_id: str) -> CurrentConversation | None:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" for _ in ACTIVE_CONVERSATION_STATUSES)
            row = conn.execute(
                f"""SELECT * FROM conversations WHERE context_kind=? AND context_id=?
                    AND status IN ({placeholders}) ORDER BY last_turn_at DESC LIMIT 1""",
                (context_kind, context_id, *sorted(ACTIVE_CONVERSATION_STATUSES)),
            ).fetchone()
            return self._from_row(row) if row else None
        finally:
            conn.close()

    def list_all(self, *, limit: int = 100) -> list[CurrentConversation]:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            return [self._from_row(row) for row in conn.execute(
                "SELECT * FROM conversations ORDER BY opened_at DESC LIMIT ?", (limit,)
            ).fetchall()]
        finally:
            conn.close()

    def transition(
        self, conversation_id: str, *, expected_version: int, status: ConversationStatus,
        reason: str, last_event_id: str, now: float, consumed_event_id: str = "",
    ) -> tuple[CurrentConversation, bool]:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
            if row is None:
                raise KeyError(conversation_id)
            current = self._from_row(row)
            if consumed_event_id and consumed_event_id in current.consumed_event_ids:
                conn.rollback()
                return current, False
            if current.version != expected_version:
                raise ConversationVersionConflict(conversation_id)
            consumed = list(current.consumed_event_ids)
            if consumed_event_id:
                consumed.append(consumed_event_id)
            cursor = conn.execute(
                """UPDATE conversations SET status=?, closure_reason=?, last_event_id=?, last_turn_at=?,
                    consumed_event_ids_json=?, attention_state=?, version=version+1
                    WHERE id=? AND version=?""",
                (
                    status.value, reason, last_event_id, now, json.dumps(consumed),
                    AttentionState.RELEASED.value if status not in {
                        ConversationStatus.OPEN, ConversationStatus.WAITING_ON_LEO, ConversationStatus.WAITING_ON_HEBE
                    } else current.attention_state.value,
                    conversation_id, expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise ConversationVersionConflict(conversation_id)
            conn.commit()
            updated = self.get(conversation_id)
            if updated is None:
                raise KeyError(conversation_id)
            return updated, True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def expire_due(self, now: float) -> int:
        conn = self.connection_factory()
        try:
            placeholders = ",".join("?" for _ in ACTIVE_CONVERSATION_STATUSES)
            cursor = conn.execute(
                f"""UPDATE conversations SET status=?, closure_reason='ttl', attention_state=?, version=version+1
                    WHERE status IN ({placeholders}) AND expires_at<=?""",
                (ConversationStatus.EXPIRED.value, AttentionState.RELEASED.value, *sorted(ACTIVE_CONVERSATION_STATUSES), now),
            )
            conn.commit()
            return int(cursor.rowcount or 0)
        finally:
            conn.close()

    def interrupt_active_on_start(self, *, reason: str = "runtime_restart") -> int:
        conn = self.connection_factory()
        try:
            placeholders = ",".join("?" for _ in ACTIVE_CONVERSATION_STATUSES)
            cursor = conn.execute(
                f"""UPDATE conversations SET status=?, closure_reason=?, attention_state=?, version=version+1
                    WHERE status IN ({placeholders})""",
                (
                    ConversationStatus.INTERRUPTED.value, reason, AttentionState.RELEASED.value,
                    *sorted(ACTIVE_CONVERSATION_STATUSES),
                ),
            )
            conn.commit()
            return int(cursor.rowcount or 0)
        finally:
            conn.close()

    @staticmethod
    def _insert(conn: sqlite3.Connection, item: CurrentConversation) -> None:
        expected = item.expected_reply.to_dict() if item.expected_reply else None
        conn.execute(
            """INSERT INTO conversations(
                id,context_kind,context_id,participants_json,attention_state,turn_owner,
                expected_reply_type,expected_reply_json,topic,origin_event_id,last_event_id,
                opened_at,last_turn_at,expires_at,status,closure_reason,version,domain_payload_json,
                consumed_event_ids_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                item.id, item.context_kind.value, item.context_id, json.dumps(item.participants),
                item.attention_state.value, item.turn_owner,
                item.expected_reply.type.value if item.expected_reply else None,
                json.dumps(expected, ensure_ascii=False) if expected else None,
                item.topic, item.origin_event_id, item.last_event_id, item.opened_at,
                item.last_turn_at, item.expires_at, item.status.value, item.closure_reason,
                item.version, json.dumps(item.domain_payload, ensure_ascii=False),
                json.dumps(item.consumed_event_ids),
            ),
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> CurrentConversation:
        expected_raw = json.loads(row["expected_reply_json"] or "null")
        return CurrentConversation(
            id=row["id"], context_kind=ConversationContext(row["context_kind"]), context_id=row["context_id"],
            participants=tuple(json.loads(row["participants_json"] or "[]")),
            attention_state=AttentionState(row["attention_state"]), turn_owner=row["turn_owner"],
            expected_reply=ExpectedReply.from_dict(expected_raw) if expected_raw else None,
            topic=row["topic"], origin_event_id=row["origin_event_id"], last_event_id=row["last_event_id"],
            opened_at=float(row["opened_at"]), last_turn_at=float(row["last_turn_at"]),
            expires_at=float(row["expires_at"]), status=ConversationStatus(row["status"]),
            closure_reason=row["closure_reason"], version=int(row["version"]),
            domain_payload=json.loads(row["domain_payload_json"] or "{}"),
            consumed_event_ids=tuple(json.loads(row["consumed_event_ids_json"] or "[]")),
        )


class OpenThreadRepository:
    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self.connection_factory = connection_factory

    def create(self, item: OpenThread) -> OpenThread:
        conn = self.connection_factory()
        try:
            conn.execute(
                """INSERT INTO open_threads(
                    id,thread_type,scope_kind,scope_id,participant_ids_json,subject_ref,summary,
                    origin_event_id,latest_event_id,status,priority,created_at,relevance_until,
                    valid_until,resolved_at,resolution_event_id,sensitivity,version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    item.id,item.thread_type,item.scope_kind,item.scope_id,json.dumps(item.participant_ids),
                    item.subject_ref,item.summary,item.origin_event_id,item.latest_event_id,item.status.value,
                    item.priority,item.created_at,item.relevance_until,item.valid_until,item.resolved_at,
                    item.resolution_event_id,item.sensitivity,item.version,
                ),
            )
            conn.commit()
            return item
        finally:
            conn.close()

    def list_open(self, *, scope_kind: str = "", scope_id: str = "", now: float | None = None) -> list[OpenThread]:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            where = ["status IN ('OPEN','WAITING_ON_LEO','WAITING_ON_HEBE','SNOOZED')"]
            params: list[object] = []
            if scope_kind:
                where.append("scope_kind=?"); params.append(scope_kind)
            if scope_id:
                where.append("scope_id=?"); params.append(scope_id)
            if now is not None:
                where.append("valid_until>?"); params.append(now)
            rows = conn.execute(
                "SELECT * FROM open_threads WHERE " + " AND ".join(where) + " ORDER BY priority DESC, created_at ASC",
                params,
            ).fetchall()
            return [self._from_row(row) for row in rows]
        finally:
            conn.close()

    def transition(self, thread_id: str, *, expected_version: int, status: OpenThreadStatus, event_id: str, now: float) -> OpenThread:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """UPDATE open_threads SET status=?, latest_event_id=?, resolved_at=?, resolution_event_id=?, version=version+1
                    WHERE id=? AND version=?""",
                (status.value,event_id,now if status == OpenThreadStatus.RESOLVED else 0.0,
                 event_id if status == OpenThreadStatus.RESOLVED else "",thread_id,expected_version),
            )
            if cursor.rowcount != 1:
                conn.rollback(); raise ConversationVersionConflict(thread_id)
            conn.commit()
            row = conn.execute("SELECT * FROM open_threads WHERE id=?", (thread_id,)).fetchone()
            return self._from_row(row)
        finally:
            conn.close()

    def transition_for_subject(
        self, subject_ref: str, *, status: OpenThreadStatus, event_id: str, now: float,
    ) -> int:
        conn = self.connection_factory()
        try:
            cursor = conn.execute(
                """UPDATE open_threads SET status=?, latest_event_id=?, resolved_at=?,
                    resolution_event_id=?, version=version+1
                    WHERE subject_ref=? AND status IN ('OPEN','WAITING_ON_LEO','WAITING_ON_HEBE','SNOOZED')""",
                (
                    status.value, event_id,
                    now if status in {OpenThreadStatus.RESOLVED, OpenThreadStatus.EXPIRED, OpenThreadStatus.ARCHIVED} else 0.0,
                    event_id, subject_ref,
                ),
            )
            conn.commit()
            return int(cursor.rowcount or 0)
        finally:
            conn.close()

    def archive_interrupted_clarifications(self, *, event_id: str, now: float) -> int:
        conn = self.connection_factory()
        try:
            cursor = conn.execute(
                """UPDATE open_threads SET status='ARCHIVED', latest_event_id=?, resolved_at=?,
                    resolution_event_id=?, version=version+1
                    WHERE thread_type='clarification'
                    AND status IN ('OPEN','WAITING_ON_LEO','WAITING_ON_HEBE','SNOOZED')
                    AND subject_ref IN (SELECT id FROM conversations WHERE status='INTERRUPTED')""",
                (event_id, now, event_id),
            )
            conn.commit()
            return int(cursor.rowcount or 0)
        finally:
            conn.close()

    def expire_closed_clarifications(self, *, event_id: str, now: float) -> int:
        conn = self.connection_factory()
        try:
            cursor = conn.execute(
                """UPDATE open_threads SET status='EXPIRED', latest_event_id=?, resolved_at=?,
                    resolution_event_id=?, version=version+1
                    WHERE thread_type='clarification'
                    AND status IN ('OPEN','WAITING_ON_LEO','WAITING_ON_HEBE','SNOOZED')
                    AND subject_ref IN (SELECT id FROM conversations WHERE status='EXPIRED')""",
                (event_id, now, event_id),
            )
            conn.commit()
            return int(cursor.rowcount or 0)
        finally:
            conn.close()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> OpenThread:
        return OpenThread(
            id=row["id"],thread_type=row["thread_type"],scope_kind=row["scope_kind"],scope_id=row["scope_id"],
            participant_ids=tuple(json.loads(row["participant_ids_json"] or "[]")),subject_ref=row["subject_ref"],
            summary=row["summary"],origin_event_id=row["origin_event_id"],latest_event_id=row["latest_event_id"],
            status=OpenThreadStatus(row["status"]),priority=int(row["priority"]),created_at=float(row["created_at"]),
            relevance_until=float(row["relevance_until"]),valid_until=float(row["valid_until"]),
            resolved_at=float(row["resolved_at"]),resolution_event_id=row["resolution_event_id"],
            sensitivity=row["sensitivity"],version=int(row["version"]),
        )
