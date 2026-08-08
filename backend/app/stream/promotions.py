from __future__ import annotations

import json
import hashlib
import inspect
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable

from app.services import db_sqlite


def _now_iso(now: float | None = None) -> str:
    return datetime.fromtimestamp(now if now is not None else time.time(), timezone.utc).isoformat()


class PromotionTriggerType(StrEnum):
    OWNER_MANUAL = "owner_manual"
    OWNER_LEARN_AND_EXECUTE = "owner_learn_and_execute"
    AUTOMATIC_FIRST_MESSAGE = "automatic_first_message"
    RAID_POLICY = "raid_policy"


class PromotionExecutionStatus(StrEnum):
    PROPOSED = "proposed"
    RESOLVING = "resolving"
    QUEUED = "queued"
    SENT = "sent"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    CLARIFICATION_REQUIRED = "clarification_required"
    TARGET_NOT_FOUND = "target_not_found"


class AutoPromoMode(StrEnum):
    DISABLED = "disabled"
    FIRST_MESSAGE_EACH_STREAM = "first_message_each_stream"
    FIRST_GREETING_EACH_STREAM = "first_greeting_each_stream"
    MANUAL_ONLY = "manual_only"


class PromotionProfileCreator(StrEnum):
    OWNER_COMMAND = "owner_command"
    MANUAL_UI = "manual_ui"


TERMINAL_PROMOTION_STATUSES = {
    PromotionExecutionStatus.SENT,
    PromotionExecutionStatus.FAILED,
    PromotionExecutionStatus.BLOCKED,
    PromotionExecutionStatus.CANCELLED,
    PromotionExecutionStatus.CLARIFICATION_REQUIRED,
    PromotionExecutionStatus.TARGET_NOT_FOUND,
}

_ALLOWED_TRANSITIONS = {
    PromotionExecutionStatus.PROPOSED: {
        PromotionExecutionStatus.RESOLVING,
        PromotionExecutionStatus.QUEUED,
        PromotionExecutionStatus.BLOCKED,
        PromotionExecutionStatus.CANCELLED,
    },
    PromotionExecutionStatus.RESOLVING: {
        PromotionExecutionStatus.QUEUED,
        PromotionExecutionStatus.FAILED,
        PromotionExecutionStatus.BLOCKED,
        PromotionExecutionStatus.CANCELLED,
    },
    PromotionExecutionStatus.QUEUED: TERMINAL_PROMOTION_STATUSES,
}


@dataclass(slots=True)
class PromotionEvent:
    id: str
    stream_session_id: str
    source_event_id: str
    requested_by: str
    raw_target_text: str
    resolved_twitch_user_id: str
    resolved_login: str
    resolution_confidence: float
    trigger_type: str
    execution_status: str
    twitch_message_id: str = ""
    created_at: str = ""
    executed_at: str | None = None
    failure_reason: str = ""


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    action_type: str
    target: str
    executor_invoked: bool
    success: bool
    external_confirmation: str = ""
    timestamp: str = field(default_factory=_now_iso)


@dataclass(slots=True)
class PromotionCommandTransaction:
    transaction_id: str
    source_event_id: str
    raw_owner_text: str
    parsed_target: str = ""
    candidate_targets: list[str] = field(default_factory=list)
    resolved_twitch_user_id: str = ""
    resolved_login: str = ""
    resolution_confidence: float = 0.0
    execution_requested: bool = False
    execution_receipt: ActionReceipt | None = None
    public_command_message_id: str = ""
    final_status: str = PromotionExecutionStatus.CLARIFICATION_REQUIRED.value
    profile_learning_status: str = "not_attempted"
    correction_of_transaction_id: str = ""


@dataclass(slots=True)
class ViewerPromotionProfile:
    twitch_user_id: str
    current_login: str
    display_name: str = ""
    known_aliases: list[str] = field(default_factory=list)
    auto_promo_mode: str = AutoPromoMode.DISABLED.value
    created_by: str = PromotionProfileCreator.OWNER_COMMAND.value
    created_at: str = ""
    updated_at: str = ""
    last_promoted_at: str | None = None
    last_promoted_stream_id: str | None = None
    cooldown_hours: float = 0.0
    owner_locked: bool = True
    active: bool = True


@dataclass(frozen=True, slots=True)
class AutoPromotionDecision:
    viewer: str
    profile: str
    first_message: bool
    decision: str
    reason: str
    event_id: str = ""


@dataclass(frozen=True, slots=True)
class PromotionProfileCommand:
    action: str
    target: str = ""
    mode: str = ""


class PromotionStore:
    def __init__(self, *, connection: sqlite3.Connection | None = None) -> None:
        self._connection = connection
        if self._connection is not None:
            self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.init_schema()
        self.invalidate_orphaned_profiles()

    def _connect(self) -> tuple[sqlite3.Connection, bool]:
        if self._connection is not None:
            return self._connection, False
        return db_sqlite.get_db_connection(), True

    def init_schema(self) -> None:
        conn, close = self._connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS promotion_events (
                id TEXT PRIMARY KEY,
                stream_session_id TEXT NOT NULL,
                source_event_id TEXT NOT NULL,
                requested_by TEXT NOT NULL,
                raw_target_text TEXT,
                resolved_twitch_user_id TEXT,
                resolved_login TEXT,
                resolution_confidence REAL NOT NULL DEFAULT 0.0,
                trigger_type TEXT NOT NULL,
                execution_status TEXT NOT NULL,
                twitch_message_id TEXT,
                created_at TEXT NOT NULL,
                executed_at TEXT,
                failure_reason TEXT
            );

            CREATE TABLE IF NOT EXISTS viewer_promotion_profiles (
                twitch_user_id TEXT PRIMARY KEY,
                current_login TEXT NOT NULL,
                display_name TEXT,
                known_aliases_json TEXT NOT NULL DEFAULT '[]',
                auto_promo_mode TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_promoted_at TEXT,
                last_promoted_stream_id TEXT,
                cooldown_hours REAL NOT NULL DEFAULT 0.0,
                owner_locked INTEGER NOT NULL DEFAULT 1,
                active INTEGER NOT NULL DEFAULT 1
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_promotion_source_event
            ON promotion_events(stream_session_id, source_event_id, trigger_type);
            CREATE INDEX IF NOT EXISTS idx_promotion_session_login
            ON promotion_events(stream_session_id, resolved_login, execution_status);
            CREATE INDEX IF NOT EXISTS idx_promotion_profile_login
            ON viewer_promotion_profiles(current_login);
            """
        )
        conn.commit()
        if close:
            conn.close()

    def create_event(
        self,
        *,
        stream_session_id: str | int,
        source_event_id: str,
        requested_by: str,
        raw_target_text: str,
        resolved_twitch_user_id: str = "",
        resolved_login: str = "",
        resolution_confidence: float = 0.0,
        trigger_type: PromotionTriggerType | str = PromotionTriggerType.OWNER_MANUAL,
        status: PromotionExecutionStatus | str = PromotionExecutionStatus.PROPOSED,
        now: float | None = None,
    ) -> PromotionEvent:
        trigger = PromotionTriggerType(str(trigger_type))
        current = PromotionExecutionStatus(str(status))
        event = PromotionEvent(
            id=f"promo_{uuid.uuid4().hex}",
            stream_session_id=str(stream_session_id or ""),
            source_event_id=str(source_event_id or f"source_{uuid.uuid4().hex}"),
            requested_by=str(requested_by or "owner"),
            raw_target_text=str(raw_target_text or ""),
            resolved_twitch_user_id=str(resolved_twitch_user_id or ""),
            resolved_login=_login(resolved_login),
            resolution_confidence=float(resolution_confidence or 0.0),
            trigger_type=trigger.value,
            execution_status=current.value,
            created_at=_now_iso(now),
        )
        conn, close = self._connect()
        with self._lock:
            try:
                conn.execute(
                    """
                    INSERT INTO promotion_events (
                        id, stream_session_id, source_event_id, requested_by,
                        raw_target_text, resolved_twitch_user_id, resolved_login,
                        resolution_confidence, trigger_type, execution_status,
                        twitch_message_id, created_at, executed_at, failure_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, NULL, '')
                    """,
                    (
                        event.id,
                        event.stream_session_id,
                        event.source_event_id,
                        event.requested_by,
                        event.raw_target_text,
                        event.resolved_twitch_user_id,
                        event.resolved_login,
                        event.resolution_confidence,
                        event.trigger_type,
                        event.execution_status,
                        event.created_at,
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                row = conn.execute(
                    """
                    SELECT * FROM promotion_events
                    WHERE stream_session_id = ? AND source_event_id = ? AND trigger_type = ?
                    """,
                    (event.stream_session_id, event.source_event_id, event.trigger_type),
                ).fetchone()
                if close:
                    conn.close()
                if row is None:
                    raise
                return self._event(row)
        if close:
            conn.close()
        return event

    def transition(
        self,
        event_id: str,
        status: PromotionExecutionStatus | str,
        *,
        twitch_message_id: str = "",
        failure_reason: str = "",
        now: float | None = None,
    ) -> PromotionEvent:
        target = PromotionExecutionStatus(str(status))
        conn, close = self._connect()
        with self._lock:
            row = conn.execute("SELECT * FROM promotion_events WHERE id = ?", (event_id,)).fetchone()
            if row is None:
                if close:
                    conn.close()
                raise KeyError(event_id)
            current = PromotionExecutionStatus(str(row["execution_status"]))
            if current in TERMINAL_PROMOTION_STATUSES:
                if close:
                    conn.close()
                if current is target:
                    return self._event(row)
                raise ValueError(f"terminal promotion event cannot transition: {current.value}->{target.value}")
            if target not in _ALLOWED_TRANSITIONS.get(current, set()):
                if close:
                    conn.close()
                raise ValueError(f"invalid promotion transition: {current.value}->{target.value}")
            executed_at = _now_iso(now) if target is PromotionExecutionStatus.SENT else None
            conn.execute(
                """
                UPDATE promotion_events
                SET execution_status = ?, twitch_message_id = ?, executed_at = ?, failure_reason = ?
                WHERE id = ?
                """,
                (target.value, str(twitch_message_id or ""), executed_at, str(failure_reason or ""), event_id),
            )
            conn.commit()
            updated = conn.execute("SELECT * FROM promotion_events WHERE id = ?", (event_id,)).fetchone()
        if close:
            conn.close()
        assert updated is not None
        return self._event(updated)

    def get_event(self, event_id: str) -> PromotionEvent | None:
        conn, close = self._connect()
        row = conn.execute("SELECT * FROM promotion_events WHERE id = ?", (event_id,)).fetchone()
        if close:
            conn.close()
        return self._event(row) if row is not None else None

    def events_for_session(self, session_id: str | int) -> list[PromotionEvent]:
        conn, close = self._connect()
        rows = conn.execute(
            "SELECT * FROM promotion_events WHERE stream_session_id = ? ORDER BY created_at, id",
            (str(session_id or ""),),
        ).fetchall()
        if close:
            conn.close()
        return [self._event(row) for row in rows]

    def was_sent(self, session_id: str | int, *, twitch_user_id: str = "", login: str = "") -> bool:
        conn, close = self._connect()
        row = conn.execute(
            """
            SELECT id FROM promotion_events
            WHERE stream_session_id = ? AND execution_status = 'sent'
              AND ((? != '' AND resolved_twitch_user_id = ?) OR (? != '' AND lower(resolved_login) = lower(?)))
            LIMIT 1
            """,
            (str(session_id or ""), twitch_user_id, twitch_user_id, _login(login), _login(login)),
        ).fetchone()
        if close:
            conn.close()
        return row is not None

    def upsert_profile(self, profile: ViewerPromotionProfile) -> ViewerPromotionProfile:
        now = profile.updated_at or _now_iso()
        created = profile.created_at or now
        login = _login(profile.current_login)
        user_id = str(profile.twitch_user_id or "").strip()
        if not _confirmed_twitch_user_id(user_id):
            raise ValueError("confirmed_twitch_identity_required")
        aliases = sorted({_login(item) for item in [*profile.known_aliases, login] if _login(item)})
        conn, close = self._connect()
        with self._lock:
            conn.execute(
                """
                INSERT INTO viewer_promotion_profiles (
                    twitch_user_id, current_login, display_name, known_aliases_json,
                    auto_promo_mode, created_by, created_at, updated_at,
                    last_promoted_at, last_promoted_stream_id, cooldown_hours,
                    owner_locked, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(twitch_user_id) DO UPDATE SET
                    current_login=excluded.current_login,
                    display_name=COALESCE(NULLIF(excluded.display_name, ''), viewer_promotion_profiles.display_name),
                    known_aliases_json=excluded.known_aliases_json,
                    auto_promo_mode=excluded.auto_promo_mode,
                    updated_at=excluded.updated_at,
                    last_promoted_at=COALESCE(excluded.last_promoted_at, viewer_promotion_profiles.last_promoted_at),
                    last_promoted_stream_id=COALESCE(excluded.last_promoted_stream_id, viewer_promotion_profiles.last_promoted_stream_id),
                    cooldown_hours=excluded.cooldown_hours,
                    owner_locked=excluded.owner_locked,
                    active=excluded.active
                """,
                (
                    user_id,
                    login,
                    str(profile.display_name or ""),
                    json.dumps(aliases, ensure_ascii=False),
                    AutoPromoMode(str(profile.auto_promo_mode)).value,
                    PromotionProfileCreator(str(profile.created_by)).value,
                    created,
                    now,
                    profile.last_promoted_at,
                    profile.last_promoted_stream_id,
                    float(profile.cooldown_hours or 0.0),
                    1 if profile.owner_locked else 0,
                    1 if profile.active else 0,
                ),
            )
            conn.commit()
        if close:
            conn.close()
        stored = self.get_profile(twitch_user_id=user_id)
        assert stored is not None
        return stored

    def get_profile(self, *, twitch_user_id: str = "", login: str = "") -> ViewerPromotionProfile | None:
        conn, close = self._connect()
        row = None
        if twitch_user_id:
            row = conn.execute(
                "SELECT * FROM viewer_promotion_profiles WHERE twitch_user_id = ?",
                (str(twitch_user_id),),
            ).fetchone()
        if row is None and login:
            row = conn.execute(
                """
                SELECT * FROM viewer_promotion_profiles
                WHERE lower(current_login) = lower(?)
                   OR EXISTS (
                       SELECT 1 FROM json_each(known_aliases_json)
                       WHERE lower(json_each.value) = lower(?)
                   )
                LIMIT 1
                """,
                (_login(login), _login(login)),
            ).fetchone()
        if close:
            conn.close()
        return self._profile(row) if row is not None else None

    def bind_twitch_user_id(self, profile: ViewerPromotionProfile, twitch_user_id: str) -> ViewerPromotionProfile:
        stable_id = str(twitch_user_id or "").strip()
        if not stable_id or stable_id == profile.twitch_user_id:
            return profile
        conn, close = self._connect()
        with self._lock:
            existing = conn.execute(
                "SELECT twitch_user_id FROM viewer_promotion_profiles WHERE twitch_user_id = ?",
                (stable_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "UPDATE viewer_promotion_profiles SET twitch_user_id = ?, updated_at = ? WHERE twitch_user_id = ?",
                    (stable_id, _now_iso(), profile.twitch_user_id),
                )
                conn.commit()
        if close:
            conn.close()
        return self.get_profile(twitch_user_id=stable_id) or profile

    def list_profiles(self, *, active_only: bool = False) -> list[ViewerPromotionProfile]:
        conn, close = self._connect()
        where = "WHERE active = 1" if active_only else ""
        rows = conn.execute(
            f"SELECT * FROM viewer_promotion_profiles {where} ORDER BY lower(current_login)"
        ).fetchall()
        if close:
            conn.close()
        return [self._profile(row) for row in rows]

    def set_mode(self, target: str, mode: AutoPromoMode | str, *, active: bool = True) -> ViewerPromotionProfile | None:
        profile = self.get_profile(twitch_user_id=target) or self.get_profile(login=target)
        if profile is None:
            return None
        profile.auto_promo_mode = AutoPromoMode(str(mode)).value
        profile.active = active
        profile.updated_at = _now_iso()
        return self.upsert_profile(profile)

    def delete_profile(self, target: str) -> bool:
        conn, close = self._connect()
        cur = conn.execute(
            "DELETE FROM viewer_promotion_profiles WHERE twitch_user_id = ? OR lower(current_login) = lower(?)",
            (str(target or ""), _login(target)),
        )
        conn.commit()
        changed = cur.rowcount > 0
        if close:
            conn.close()
        return changed

    def invalidate_orphaned_profiles(self) -> int:
        """Automatically disable legacy profiles backed only by an arbitrary login."""
        conn, close = self._connect()
        cur = conn.execute(
            "UPDATE viewer_promotion_profiles SET active = 0, auto_promo_mode = 'disabled' "
            "WHERE twitch_user_id = '' OR twitch_user_id LIKE 'login:%'"
        )
        conn.commit()
        changed = max(0, int(cur.rowcount or 0))
        if close:
            conn.close()
        if changed:
            print(f"[HEBE][PROMOTION_PROFILE_MIGRATION] orphaned_invalidated={changed}", flush=True)
        return changed

    def mark_profile_promoted(self, profile: ViewerPromotionProfile, session_id: str | int, *, now: float | None = None) -> None:
        profile.last_promoted_at = _now_iso(now)
        profile.last_promoted_stream_id = str(session_id or "")
        profile.updated_at = _now_iso(now)
        self.upsert_profile(profile)

    @staticmethod
    def _event(row: sqlite3.Row) -> PromotionEvent:
        return PromotionEvent(**{key: row[key] for key in PromotionEvent.__dataclass_fields__})

    @staticmethod
    def _profile(row: sqlite3.Row) -> ViewerPromotionProfile:
        aliases = json.loads(row["known_aliases_json"] or "[]")
        return ViewerPromotionProfile(
            twitch_user_id=row["twitch_user_id"],
            current_login=row["current_login"],
            display_name=row["display_name"] or "",
            known_aliases=list(aliases or []),
            auto_promo_mode=row["auto_promo_mode"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_promoted_at=row["last_promoted_at"],
            last_promoted_stream_id=row["last_promoted_stream_id"],
            cooldown_hours=float(row["cooldown_hours"] or 0.0),
            owner_locked=bool(row["owner_locked"]),
            active=bool(row["active"]),
        )


class PromotionProfileManager:
    def __init__(self, store: PromotionStore, *, default_auto_after_success: bool | None = None) -> None:
        self.store = store
        self.default_auto_after_success = (
            os.getenv("HEBE_PROMOTION_LEARN_AFTER_SUCCESS", "true").strip().lower() in {"1", "true", "yes", "on"}
            if default_auto_after_success is None
            else bool(default_auto_after_success)
        )

    def learn_after_success(
        self,
        *,
        twitch_user_id: str,
        login: str,
        display_name: str = "",
        owner_command: str = "",
        stream_session_id: str | int = "",
        known_aliases: list[str] | None = None,
        source_promotion_event: str = "",
        now: float | None = None,
    ) -> ViewerPromotionProfile | None:
        if not _confirmed_twitch_user_id(twitch_user_id) or not _login(login):
            print(
                f"[HEBE][PROMOTION_PROFILE_LEARN] viewer={_login(login)} persisted=false "
                "reason=confirmed_twitch_identity_and_success_event_required",
                flush=True,
            )
            return None
        source_event = self.store.get_event(source_promotion_event) if source_promotion_event else None
        if source_event is not None and source_event.execution_status != PromotionExecutionStatus.SENT.value:
            return None
        command = _normalize(owner_command)
        if _only_this_time(command):
            print(
                "[HEBE][PROMOTION_PROFILE_LEARN] "
                f"viewer={_login(login)} source_promotion_event={source_promotion_event or 'none'} "
                "auto_promo_mode=manual_only persisted=false",
                flush=True,
            )
            return None
        explicit_always = bool(re.search(r"\b(?:siempre|automaticamente|automatico|cada directo|cuando aparezca|always|every stream)\b", command))
        if not explicit_always and not self.default_auto_after_success:
            return None
        existing = self.store.get_profile(twitch_user_id=twitch_user_id, login=login)
        profile = existing or ViewerPromotionProfile(
            twitch_user_id=str(twitch_user_id or "").strip(),
            current_login=_login(login),
            display_name=display_name or login,
            created_by=PromotionProfileCreator.OWNER_COMMAND.value,
            created_at=_now_iso(now),
        )
        profile.current_login = _login(login)
        profile.display_name = display_name or profile.display_name or login
        profile.known_aliases = sorted(set([
            *profile.known_aliases,
            _login(login),
            *(_login(alias) for alias in (known_aliases or []) if _login(alias)),
        ]))
        profile.auto_promo_mode = AutoPromoMode.FIRST_MESSAGE_EACH_STREAM.value
        profile.active = True
        profile.owner_locked = True
        profile.last_promoted_at = _now_iso(now)
        profile.last_promoted_stream_id = str(stream_session_id or "") or None
        profile.updated_at = _now_iso(now)
        persisted = self.store.upsert_profile(profile)
        print(
            "[HEBE][PROMOTION_PROFILE_LEARN] "
            f"viewer={profile.current_login} source_promotion_event={source_promotion_event or 'none'} "
            f"auto_promo_mode={profile.auto_promo_mode} persisted={str(persisted is not None).lower()}",
            flush=True,
        )
        return persisted

    def apply_command(self, command: PromotionProfileCommand) -> Any:
        if command.action == "list":
            return self.store.list_profiles(active_only=True)
        if command.action == "disable":
            return self.store.set_mode(command.target, AutoPromoMode.DISABLED, active=True)
        if command.action == "clear":
            return self.store.delete_profile(command.target)
        if command.action == "enable":
            profile = self.store.get_profile(login=command.target)
            if profile is None:
                return None
            profile.auto_promo_mode = command.mode or AutoPromoMode.FIRST_MESSAGE_EACH_STREAM.value
            profile.active = True
            profile.updated_at = _now_iso()
            return self.store.upsert_profile(profile)
        return None


class AutomaticPromotionService:
    def __init__(
        self,
        store: PromotionStore,
        *,
        spacing_seconds: float = 8.0,
        max_retries: int = 1,
        bot_usernames: set[str] | None = None,
        self_usernames: set[str] | None = None,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.spacing_seconds = max(0.0, float(spacing_seconds))
        self.max_retries = max(0, int(max_retries))
        self.bot_usernames = {_login(item) for item in (bot_usernames or set())}
        self.self_usernames = {_login(item) for item in (self_usernames or set())}
        self.now_fn = now_fn
        self.session_id = ""
        self._seen_viewers: set[str] = set()
        self._greeted_viewers: set[str] = set()
        self._seen_message_ids: set[str] = set()
        self._profile_miss_logged: set[str] = set()
        self._queue: deque[dict[str, Any]] = deque()
        self._last_send_at = float("-inf")
        self._lock = threading.RLock()

    def start_session(self, session_id: str | int) -> None:
        key = str(session_id or "")
        if key == self.session_id:
            return
        with self._lock:
            self.session_id = key
            self._seen_viewers.clear()
            self._greeted_viewers.clear()
            self._seen_message_ids.clear()
            self._profile_miss_logged.clear()
            self._queue.clear()
            self._last_send_at = float("-inf")

    def observe_chat_message(
        self,
        *,
        stream_session_id: str | int,
        twitch_user_id: str,
        login: str,
        display_name: str,
        message_text: str,
        message_id: str,
        channel_live: bool,
        is_bot: bool = False,
        is_self: bool = False,
    ) -> AutoPromotionDecision:
        self.start_session(stream_session_id)
        user_login = _login(login)
        viewer_key = str(twitch_user_id or "").strip() or f"login:{user_login}"
        fallback_digest = hashlib.sha256(str(message_text or "").encode("utf-8")).hexdigest()[:16]
        source_id = str(message_id or "").strip() or f"chat:{viewer_key}:{fallback_digest}"
        if source_id in self._seen_message_ids:
            return self._decision(user_login, "", False, "skip", "duplicate_observed_message")
        self._seen_message_ids.add(source_id)
        first_message = viewer_key not in self._seen_viewers
        self._seen_viewers.add(viewer_key)
        if not channel_live:
            return self._decision(user_login, "", first_message, "skip", "channel_offline")
        if is_bot or is_self or user_login in self.bot_usernames or user_login in self.self_usernames:
            return self._decision(user_login, "", first_message, "skip", "bot_or_self")
        profile = self.store.get_profile(twitch_user_id=twitch_user_id, login=user_login)
        if profile is None:
            return self._decision(user_login, "", first_message, "skip", "profile_missing")
        if twitch_user_id and profile.twitch_user_id.startswith("login:"):
            profile = self.store.bind_twitch_user_id(profile, twitch_user_id)
        elif twitch_user_id and profile.twitch_user_id == twitch_user_id and user_login != profile.current_login:
            profile.known_aliases = sorted(set([
                *profile.known_aliases, profile.current_login, user_login,
            ]))
            profile.current_login = user_login
            profile.display_name = display_name or profile.display_name or user_login
            profile.updated_at = _now_iso(self.now_fn())
            profile = self.store.upsert_profile(profile)
        if not profile.active or profile.auto_promo_mode in {AutoPromoMode.DISABLED.value, AutoPromoMode.MANUAL_ONLY.value}:
            return self._decision(user_login, profile.auto_promo_mode, first_message, "skip", "profile_disabled")
        if profile.auto_promo_mode == AutoPromoMode.FIRST_GREETING_EACH_STREAM.value:
            if not _looks_greeting(message_text):
                return self._decision(user_login, profile.auto_promo_mode, first_message, "skip", "message_not_greeting")
            if viewer_key in self._greeted_viewers:
                return self._decision(user_login, profile.auto_promo_mode, first_message, "skip", "not_first_greeting")
            self._greeted_viewers.add(viewer_key)
        elif not first_message:
            return self._decision(user_login, profile.auto_promo_mode, False, "skip", "not_first_message")
        if self.store.was_sent(self.session_id, twitch_user_id=twitch_user_id, login=user_login):
            return self._decision(user_login, profile.auto_promo_mode, True, "skip", "already_sent_this_stream")
        if self._cooldown_active(profile):
            return self._decision(user_login, profile.auto_promo_mode, True, "skip", "profile_cooldown")
        event = self.store.create_event(
            stream_session_id=self.session_id,
            source_event_id=source_id,
            requested_by="owner_delegated",
            raw_target_text=user_login,
            resolved_twitch_user_id=twitch_user_id,
            resolved_login=user_login,
            resolution_confidence=1.0,
            trigger_type=PromotionTriggerType.AUTOMATIC_FIRST_MESSAGE,
        )
        if event.execution_status == PromotionExecutionStatus.PROPOSED.value:
            event = self.store.transition(event.id, PromotionExecutionStatus.QUEUED)
            with self._lock:
                self._queue.append({"event": event, "profile": profile, "retry": 0})
        return self._decision(user_login, profile.auto_promo_mode, True, "queue", "configured_first_message", event.id)

    def drain_ready(self, send_shoutout: Callable[[str], Any], *, now: float | None = None) -> PromotionEvent | None:
        current = self.now_fn() if now is None else float(now)
        with self._lock:
            if not self._queue or current - self._last_send_at < self.spacing_seconds:
                return None
            item = self._queue.popleft()
        event: PromotionEvent = item["event"]
        profile: ViewerPromotionProfile = item["profile"]
        success, twitch_message_id, reason = _send_result(
            send_shoutout,
            event.resolved_login,
            source="automatic_promotion_policy",
            authority="owner_delegated",
            twitch_user_id=profile.twitch_user_id,
        )
        if success:
            updated = self.store.transition(
                event.id,
                PromotionExecutionStatus.SENT,
                twitch_message_id=twitch_message_id,
                now=current,
            )
            self.store.mark_profile_promoted(profile, self.session_id, now=current)
            self._last_send_at = current
            self._log_outcome(event.resolved_login, "sent", "twitch_send_success")
            return updated
        updated = self.store.transition(
            event.id,
            PromotionExecutionStatus.FAILED,
            failure_reason=reason or "twitch_send_failed",
            now=current,
        )
        retry = int(item.get("retry") or 0)
        if retry < self.max_retries and _is_transient_promotion_failure(reason):
            retry_event = self.store.create_event(
                stream_session_id=self.session_id,
                source_event_id=f"{event.source_event_id}:retry:{retry + 1}",
                requested_by=event.requested_by,
                raw_target_text=event.raw_target_text,
                resolved_twitch_user_id=event.resolved_twitch_user_id,
                resolved_login=event.resolved_login,
                resolution_confidence=event.resolution_confidence,
                trigger_type=PromotionTriggerType.AUTOMATIC_FIRST_MESSAGE,
                now=current,
            )
            retry_event = self.store.transition(retry_event.id, PromotionExecutionStatus.QUEUED)
            with self._lock:
                self._queue.append({"event": retry_event, "profile": profile, "retry": retry + 1})
        self._last_send_at = current
        self._log_outcome(event.resolved_login, "failed", reason or "twitch_send_failed")
        return updated

    @property
    def queued_count(self) -> int:
        return len(self._queue)

    def _cooldown_active(self, profile: ViewerPromotionProfile) -> bool:
        if not profile.last_promoted_at or profile.cooldown_hours <= 0:
            return False
        try:
            then = datetime.fromisoformat(profile.last_promoted_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return False
        return self.now_fn() - then < profile.cooldown_hours * 3600

    def _decision(self, viewer: str, profile: str, first_message: bool, decision: str, reason: str, event_id: str = "") -> AutoPromotionDecision:
        result = AutoPromotionDecision(viewer, profile, first_message, decision, reason, event_id)
        if reason == "profile_missing":
            if viewer in self._profile_miss_logged:
                return result
            self._profile_miss_logged.add(viewer)
        print(
            "[HEBE][AUTO_PROMO_TRIGGER] "
            f"viewer={viewer} profile={profile or 'none'} first_message={str(first_message).lower()} "
            f"decision={decision} reason={reason}",
            flush=True,
        )
        return result

    @staticmethod
    def _log_outcome(viewer: str, status: str, reason: str) -> None:
        print(f"[HEBE][AUTO_PROMO_OUTCOME] viewer={viewer} status={status} reason={reason}", flush=True)


def parse_promotion_profile_command(text: str) -> PromotionProfileCommand | None:
    command = _normalize(text)
    command = re.sub(r"^(?:hebe|ebe|eve|jebe)\s+", "", command).strip()
    if re.search(r"\b(?:lista|muestra|ensena|ver|show|list)\b.*\b(?:promociones|promos|promotion)\b", command):
        return PromotionProfileCommand("list")
    patterns = (
        ("disable", r"^(?:deja de|para de|no vuelvas a|stop)\s+(?:promocionar|hacer promo|automatic promotions? for)\s+(?:a\s+)?(.+)$"),
        ("clear", r"^(?:borra|limpia|olvida|clear)\s+(?:la\s+)?(?:preferencia de )?(?:promo|promocion|promotion)\s+(?:de|para|for|a)\s+(.+)$"),
        ("enable", r"^(?:promociona|haz promo a|promote)\s+(.+?)\s+(?:siempre|cuando aparezca|cada directo|always|every stream)$"),
        ("enable", r"^(?:siempre|always)\s+(?:promociona|promote)\s+(?:a\s+)?(.+)$"),
    )
    for action, pattern in patterns:
        match = re.match(pattern, command)
        if match:
            target = _login(match.group(1))
            return PromotionProfileCommand(
                action,
                target,
                AutoPromoMode.FIRST_MESSAGE_EACH_STREAM.value if action == "enable" else "",
            )
    return None


def record_manual_promotion(
    store: PromotionStore,
    *,
    stream_session_id: str | int,
    source_event_id: str,
    requested_by: str,
    raw_target_text: str,
    resolved_twitch_user_id: str,
    resolved_login: str,
    resolution_confidence: float,
    send_shoutout: Callable[[str], Any],
    trigger_type: PromotionTriggerType | str = PromotionTriggerType.OWNER_MANUAL,
) -> PromotionEvent:
    event = store.create_event(
        stream_session_id=stream_session_id,
        source_event_id=source_event_id,
        requested_by=requested_by,
        raw_target_text=raw_target_text,
        resolved_twitch_user_id=resolved_twitch_user_id,
        resolved_login=resolved_login,
        resolution_confidence=resolution_confidence,
        trigger_type=trigger_type,
    )
    if event.execution_status != PromotionExecutionStatus.PROPOSED.value:
        return event
    store.transition(event.id, PromotionExecutionStatus.RESOLVING)
    store.transition(event.id, PromotionExecutionStatus.QUEUED)
    success, twitch_message_id, reason = _send_result(send_shoutout, resolved_login)
    if success:
        return store.transition(event.id, PromotionExecutionStatus.SENT, twitch_message_id=twitch_message_id)
    return store.transition(event.id, PromotionExecutionStatus.FAILED, failure_reason=reason or "twitch_send_failed")


def _send_result(send_shoutout: Callable[[str], Any], login: str, **context: Any) -> tuple[bool, str, str]:
    try:
        signature = inspect.signature(send_shoutout)
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        accepted = context if accepts_kwargs else {key: value for key, value in context.items() if key in signature.parameters}
        raw = send_shoutout(login, **accepted)
    except Exception as exc:
        return False, "", f"{type(exc).__name__}: {exc}"
    if isinstance(raw, dict):
        return bool(raw.get("success") or raw.get("sent")), str(raw.get("message_id") or ""), str(raw.get("reason") or raw.get("error") or "")
    if isinstance(raw, tuple):
        return bool(raw[0] if raw else False), str(raw[1] if len(raw) > 1 else ""), str(raw[2] if len(raw) > 2 else "")
    return bool(raw), "", "" if raw else "twitch_send_returned_false"


def _is_transient_promotion_failure(reason: str) -> bool:
    normalized = _normalize(reason)
    permanent = {
        "untrusted_source", "unauthorized", "profile_disabled", "invalid_identity",
        "invalid_target", "policy_blocked", "offline_stream", "wrong_runtime_context",
        "blocked_bot_user", "own_channel", "ambient_stt_not_allowed", "cooldown_active",
        "profile_cooldown", "authentication", "auth_failed", "forbidden",
    }
    if any(marker in normalized for marker in permanent):
        return False
    transient = {
        "timeout", "timed out", "connection", "network", "temporar", "rate limit",
        "429", "502", "503", "504", "service unavailable", "send failed", "send_failed",
        "twitch send returned false",
    }
    return any(marker in normalized for marker in transient)


def _login(value: str) -> str:
    cleaned = str(value or "").strip().lower().lstrip("@").strip()
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    return cleaned[:25]


def _confirmed_twitch_user_id(value: str) -> bool:
    candidate = str(value or "").strip()
    return bool(candidate and not candidate.casefold().startswith("login:") and re.fullmatch(r"[0-9]+", candidate))


def _normalize(value: str) -> str:
    text = str(value or "").casefold()
    text = text.translate(str.maketrans("áéíóúüñ", "aeiouun"))
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    return " ".join(text.split())


def _only_this_time(command: str) -> bool:
    return bool(re.search(r"\b(?:solo esta vez|solo ahora|una sola vez|only this time|just this once)\b", command))


def _looks_greeting(text: str) -> bool:
    return bool(re.match(r"^\s*(?:hola|buenas|hey|hi|hello|wenas|saludos)\b", _normalize(text)))


__all__ = [
    "AutoPromoMode",
    "AutoPromotionDecision",
    "AutomaticPromotionService",
    "ActionReceipt",
    "PromotionEvent",
    "PromotionCommandTransaction",
    "PromotionExecutionStatus",
    "PromotionProfileCommand",
    "PromotionProfileCreator",
    "PromotionProfileManager",
    "PromotionStore",
    "PromotionTriggerType",
    "ViewerPromotionProfile",
    "parse_promotion_profile_command",
    "record_manual_promotion",
]
