from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.services import db_sqlite


BOT_USERNAMES = {
    "hebenifelheim",
    "jotunbot",
    "streamelements",
    "nightbot",
    "moobot",
    "fossabot",
    "streamlabs",
}

_READY_DB_PATH: str | None = None
_INITIALIZING_SCHEMA = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _loads(value: str | None, fallback: Any = None) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _norm_user(username: str | None) -> str:
    return re.sub(r"[^a-z0-9_]", "", str(username or "").strip().lower().lstrip("@"))[:25]


def _display_name(username: str | None, display_name: str | None = None) -> str:
    return (str(display_name or "").strip() or str(username or "").strip())[:80]


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _seconds_between(start: str | None, end: str | None) -> int | None:
    a = _parse_iso(start)
    b = _parse_iso(end)
    if not a or not b:
        return None
    return max(0, int((b - a).total_seconds()))


def _session_stale_reason(active: dict | None, metadata: dict[str, Any], now: str) -> str | None:
    if not active:
        return None
    active_stream_id = str(active.get("twitch_stream_id") or "").strip()
    current_stream_id = str(metadata.get("twitch_stream_id") or "").strip()
    if active_stream_id and current_stream_id and active_stream_id != current_stream_id:
        return "stream_id_mismatch"
    age = _seconds_between(active.get("started_at"), now) or 0
    max_age = int(float(os.getenv("HEBE_STREAM_SESSION_MAX_SECONDS", "64800") or 64800))
    if age > max_age:
        return "stale_or_too_old"
    started = _parse_iso(active.get("started_at"))
    current = _parse_iso(now)
    if started and current and started.date() != current.date() and age > 4 * 3600:
        return "stale_day_boundary"
    return None


def _stream_references_active_session(stream: Any, active: dict | None) -> bool:
    if stream is None or not active:
        return False
    active_stream_id = str(active.get("twitch_stream_id") or "").strip()
    current_stream_id = str(_stream_metadata(stream).get("twitch_stream_id") or "").strip()
    if active_stream_id and current_stream_id and active_stream_id != current_stream_id:
        return False
    try:
        return int(getattr(stream, "active_stream_session_id", 0) or 0) == int(active.get("id") or 0)
    except (TypeError, ValueError):
        return False


def _stale_close_session(conn: sqlite3.Connection, active: dict, *, now: str, reason: str) -> None:
    max_age = int(float(os.getenv("HEBE_STREAM_SESSION_MAX_SECONDS", "64800") or 64800))
    duration = min(_seconds_between(active.get("started_at"), now) or 0, max_age)
    conn.execute(
        """
        UPDATE stream_sessions
        SET ended_at = ?, duration_seconds = ?, status = 'stale_closed', updated_at = ?
        WHERE id = ?
        """,
        (now, duration, now, active["id"]),
    )
    print(f"[HEBE][STREAM_SESSION] stale_closed id={active['id']} reason={reason}", flush=True)


def _is_bot(username: str | None) -> bool:
    configured = os.getenv("HEBE_TWITCH_BOT_USERNAMES", "")
    bots = set(BOT_USERNAMES)
    bots.update(part.strip().lower().lstrip("@") for part in configured.split(",") if part.strip())
    return _norm_user(username) in bots


def _clean_text(value: Any, *, max_len: int = 240) -> str | None:
    text = str(value or "").strip()
    return text[:max_len] if text else None


def _stream_metadata(stream: Any = None) -> dict[str, Any]:
    if stream is None:
        return {}
    title = _clean_text(getattr(stream, "current_stream_title", None), max_len=300)
    category = _clean_text(getattr(stream, "current_category", None), max_len=160)
    game = _clean_text(getattr(stream, "current_game", None), max_len=160) or category
    started_at = getattr(stream, "stream_started_at", None)
    twitch_stream_id = (
        getattr(stream, "twitch_stream_id", None)
        or getattr(stream, "current_twitch_stream_id", None)
        or getattr(stream, "stream_id", None)
    )
    started_at = db_sqlite.normalize_iso(started_at) if started_at else None
    return {
        "twitch_stream_id": _clean_text(twitch_stream_id, max_len=80),
        "title": title,
        "category": category,
        "game": game,
        "started_at": started_at,
        "playthrough_type": _clean_text(getattr(stream, "current_playthrough_type", None), max_len=80),
        "challenge": _clean_text(getattr(stream, "current_challenge", None), max_len=160),
        "language_mode": _clean_text(getattr(stream, "language_mode", None), max_len=60),
        "spoiler_policy": _clean_text(getattr(stream, "spoiler_policy", None), max_len=80) or "no_spoilers",
    }


def _source_is_dev_or_simulation(source: str | None, payload: dict | None = None) -> bool:
    text = str(source or "").strip().lower()
    if any(token in text for token in ("sim", "dev", "test", "fixture")):
        return True
    payload = payload or {}
    return bool(payload.get("_simulated") or payload.get("simulated") or payload.get("source") in {"simulation", "dev"})


def _stream_is_live(stream: Any = None) -> bool:
    return bool(stream is not None and getattr(stream, "is_live", False))


def _metadata_missing(row: sqlite3.Row | dict | None) -> bool:
    if not row:
        return True
    return not (_clean_text(row["title"] if isinstance(row, sqlite3.Row) else row.get("title")) and (
        _clean_text(row["game"] if isinstance(row, sqlite3.Row) else row.get("game"))
        or _clean_text(row["category"] if isinstance(row, sqlite3.Row) else row.get("category"))
    ))


def _ensure_stream_memory_columns(conn: sqlite3.Connection) -> None:
    migrations = {
        "stream_sessions": {
            "twitch_stream_id": "TEXT",
            "title": "TEXT",
            "category": "TEXT",
            "game": "TEXT",
            "started_at": "TEXT",
            "ended_at": "TEXT",
            "duration_seconds": "INTEGER",
            "playthrough_type": "TEXT",
            "challenge": "TEXT",
            "language_mode": "TEXT",
            "spoiler_policy": "TEXT",
            "status": "TEXT NOT NULL DEFAULT 'unknown'",
            "source": "TEXT NOT NULL DEFAULT 'unknown'",
            "is_real_stream": "INTEGER NOT NULL DEFAULT 1",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        },
        "stream_events": {
            "dedupe_key": "TEXT",
        },
        "stream_summaries": {
            "metadata_json": "TEXT",
        },
    }
    for table, columns in migrations.items():
        for column, definition in columns.items():
            db_sqlite.ensure_column(conn, table, column, definition)


def init_stream_memory_schema() -> None:
    global _READY_DB_PATH, _INITIALIZING_SCHEMA
    _INITIALIZING_SCHEMA = True
    conn = db_sqlite.get_db_connection()
    try:
        cur = conn.cursor()

        existing = {
            row["name"]
            for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }

        cur.executescript(
            """
        CREATE TABLE IF NOT EXISTS stream_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            twitch_stream_id TEXT,
            title TEXT,
            category TEXT,
            game TEXT,
            started_at TEXT,
            ended_at TEXT,
            duration_seconds INTEGER,
            playthrough_type TEXT,
            challenge TEXT,
            language_mode TEXT,
            spoiler_policy TEXT,
            status TEXT NOT NULL DEFAULT 'unknown',
            source TEXT NOT NULL DEFAULT 'unknown',
            is_real_stream INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS stream_chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_session_id INTEGER,
            twitch_message_id TEXT,
            username TEXT NOT NULL,
            display_name TEXT,
            message_text TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            is_mention_to_hebe INTEGER NOT NULL DEFAULT 0,
            is_direct_reply_to_hebe INTEGER NOT NULL DEFAULT 0,
            is_bot INTEGER NOT NULL DEFAULT 0,
            is_mod INTEGER NOT NULL DEFAULT 0,
            is_vip INTEGER NOT NULL DEFAULT 0,
            is_subscriber INTEGER NOT NULL DEFAULT 0,
            badges_json TEXT,
            source TEXT NOT NULL DEFAULT 'twitch_irc',
            replied_by_hebe INTEGER NOT NULL DEFAULT 0,
            hebe_reply_message_id INTEGER,
            topic_hint TEXT,
            language_hint TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(stream_session_id) REFERENCES stream_sessions(id)
        );

        CREATE TABLE IF NOT EXISTS chatter_profiles (
            username TEXT PRIMARY KEY,
            display_name TEXT,
            aliases_json TEXT,
            first_seen_at TEXT,
            last_seen_at TEXT,
            last_message_at TEXT,
            last_direct_interaction_at TEXT,
            last_lurk_seen_at TEXT,
            last_raid_at TEXT,
            last_follow_at TEXT,
            last_sub_at TEXT,
            streams_seen_count INTEGER NOT NULL DEFAULT 0,
            streams_chatted_count INTEGER NOT NULL DEFAULT 0,
            total_messages INTEGER NOT NULL DEFAULT 0,
            total_direct_interactions INTEGER NOT NULL DEFAULT 0,
            total_lurk_sessions INTEGER NOT NULL DEFAULT 0,
            viewer_status TEXT,
            preferred_language TEXT,
            relationship_level TEXT,
            notes_summary TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chatter_presence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_session_id INTEGER,
            username TEXT NOT NULL,
            display_name TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            first_message_at TEXT,
            last_message_at TEXT,
            first_direct_interaction_at TEXT,
            last_direct_interaction_at TEXT,
            message_count INTEGER NOT NULL DEFAULT 0,
            direct_interaction_count INTEGER NOT NULL DEFAULT 0,
            was_present INTEGER NOT NULL DEFAULT 1,
            was_active_chatter INTEGER NOT NULL DEFAULT 0,
            was_passive_viewer INTEGER NOT NULL DEFAULT 0,
            was_raider INTEGER NOT NULL DEFAULT 0,
            was_new_chatter INTEGER NOT NULL DEFAULT 0,
            was_returning_after_absence INTEGER NOT NULL DEFAULT 0,
            presence_source_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(stream_session_id) REFERENCES stream_sessions(id)
        );

        CREATE TABLE IF NOT EXISTS chatter_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            fact_text TEXT NOT NULL,
            fact_type TEXT,
            confidence TEXT NOT NULL DEFAULT 'low',
            source_message_id INTEGER,
            source_stream_session_id INTEGER,
            evidence_count INTEGER NOT NULL DEFAULT 1,
            first_observed_at TEXT NOT NULL,
            last_confirmed_at TEXT NOT NULL,
            expires_at TEXT,
            public_reference_allowed INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_message_id) REFERENCES stream_chat_messages(id),
            FOREIGN KEY(source_stream_session_id) REFERENCES stream_sessions(id)
        );

        CREATE TABLE IF NOT EXISTS stream_chatter_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_session_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            display_name TEXT,
            message_count INTEGER NOT NULL DEFAULT 0,
            direct_interaction_count INTEGER NOT NULL DEFAULT 0,
            summary_text TEXT,
            topics_json TEXT,
            notable_quotes_json TEXT,
            mood_tone TEXT,
            inferred_facts_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(stream_session_id) REFERENCES stream_sessions(id)
        );

        CREATE TABLE IF NOT EXISTS stream_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_session_id INTEGER NOT NULL,
            summary_text TEXT,
            key_events_json TEXT,
            game_progress_json TEXT,
            chat_topics_json TEXT,
            chatter_highlights_json TEXT,
            raids_json TEXT,
            shoutouts_json TEXT,
            next_stream_context TEXT,
            metadata_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(stream_session_id) REFERENCES stream_sessions(id)
        );

        CREATE TABLE IF NOT EXISTS stream_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_session_id INTEGER,
            event_type TEXT NOT NULL,
            event_ts TEXT NOT NULL,
            payload_json TEXT,
            dedupe_key TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(stream_session_id) REFERENCES stream_sessions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_stream_chat_messages_session ON stream_chat_messages(stream_session_id);
        CREATE INDEX IF NOT EXISTS idx_stream_chat_messages_username ON stream_chat_messages(username);
        CREATE INDEX IF NOT EXISTS idx_stream_chat_messages_observed_at ON stream_chat_messages(observed_at);
        CREATE INDEX IF NOT EXISTS idx_stream_chat_messages_username_observed_at ON stream_chat_messages(username, observed_at);
        CREATE INDEX IF NOT EXISTS idx_stream_chat_messages_session_username ON stream_chat_messages(stream_session_id, username);
        CREATE INDEX IF NOT EXISTS idx_chatter_profiles_last_seen ON chatter_profiles(last_seen_at);
        CREATE INDEX IF NOT EXISTS idx_chatter_profiles_last_message ON chatter_profiles(last_message_at);
        CREATE INDEX IF NOT EXISTS idx_chatter_profiles_status ON chatter_profiles(viewer_status);
        CREATE INDEX IF NOT EXISTS idx_chatter_presence_session ON chatter_presence(stream_session_id);
        CREATE INDEX IF NOT EXISTS idx_chatter_presence_username ON chatter_presence(username);
        CREATE INDEX IF NOT EXISTS idx_chatter_presence_last_seen ON chatter_presence(last_seen_at);
        CREATE INDEX IF NOT EXISTS idx_chatter_presence_last_message ON chatter_presence(last_message_at);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_chatter_presence_session_username ON chatter_presence(stream_session_id, username);
        CREATE INDEX IF NOT EXISTS idx_chatter_facts_username ON chatter_facts(username);
        CREATE INDEX IF NOT EXISTS idx_chatter_facts_type ON chatter_facts(fact_type);
        CREATE INDEX IF NOT EXISTS idx_chatter_facts_confidence ON chatter_facts(confidence);
        CREATE INDEX IF NOT EXISTS idx_chatter_facts_last_confirmed ON chatter_facts(last_confirmed_at);
        CREATE INDEX IF NOT EXISTS idx_stream_sessions_status ON stream_sessions(status);
        CREATE INDEX IF NOT EXISTS idx_stream_sessions_started_at ON stream_sessions(started_at);
        CREATE INDEX IF NOT EXISTS idx_stream_sessions_game_category ON stream_sessions(game, category);
        CREATE INDEX IF NOT EXISTS idx_stream_events_session ON stream_events(stream_session_id);
        CREATE INDEX IF NOT EXISTS idx_stream_events_type ON stream_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_stream_events_ts ON stream_events(event_ts);
        CREATE INDEX IF NOT EXISTS idx_stream_summaries_session ON stream_summaries(stream_session_id);
        CREATE INDEX IF NOT EXISTS idx_stream_summaries_created_at ON stream_summaries(created_at);
            """
        )
        _ensure_stream_memory_columns(conn)
        cur.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_stream_sessions_real_status ON stream_sessions(is_real_stream, status);
            CREATE INDEX IF NOT EXISTS idx_stream_events_dedupe ON stream_events(stream_session_id, event_type, dedupe_key);
            """
        )

        conn.commit()
        created = [
            name
            for name in (
                "stream_sessions",
                "stream_chat_messages",
                "chatter_profiles",
                "chatter_presence",
                "chatter_facts",
                "stream_chatter_summaries",
                "stream_summaries",
                "stream_events",
            )
            if name not in existing
        ]
        print(
            "[HEBE][STREAM_MEMORY] schema checked "
            f"existing_reused={sorted(existing & {'chat_log','internal_events_log','memory_chunks','memory_facts','memories','reminders'})} "
            f"new_tables_created={created} indexes_created_or_verified=true "
            "reuse=chat_log/general_conversation,memory_facts/general_facts,memory_chunks/stream_summary_rag",
            flush=True,
        )
        _READY_DB_PATH = db_sqlite.DB_PATH
    finally:
        conn.close()
        _INITIALIZING_SCHEMA = False


def ensure_stream_memory_ready() -> None:
    if _INITIALIZING_SCHEMA:
        return
    if _READY_DB_PATH != db_sqlite.DB_PATH:
        init_stream_memory_schema()


def get_active_stream_session(conn: sqlite3.Connection | None = None, *, real_only: bool = True) -> dict | None:
    ensure_stream_memory_ready()
    close = conn is None
    conn = conn or db_sqlite.get_db_connection()
    if real_only:
        row = conn.execute(
            """
            SELECT * FROM stream_sessions
            WHERE status = 'live' AND COALESCE(is_real_stream, 1) = 1
            ORDER BY started_at DESC, id DESC LIMIT 1
            """
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM stream_sessions WHERE status = 'live' ORDER BY started_at DESC, id DESC LIMIT 1"
        ).fetchone()
    if close:
        conn.close()
    return _row(row)


def ensure_active_stream_session(stream: Any = None, *, source: str = "unknown") -> int | None:
    ensure_stream_memory_ready()
    conn = db_sqlite.get_db_connection()
    now = _now_iso()
    metadata = _stream_metadata(stream)
    active = get_active_stream_session(conn)
    stored_source = "twitch" if source in {"engine", "context_sync", "stream_online"} else source
    if _source_is_dev_or_simulation(source):
        print(f"[HEBE][STREAM_SESSION] skipped reason=simulation source={source}", flush=True)
        conn.close()
        return getattr(stream, "active_stream_session_id", None) if stream is not None else None
    if not _stream_is_live(stream):
        print("[HEBE][STREAM_SESSION] skipped reason=offline", flush=True)
        conn.close()
        return getattr(stream, "active_stream_session_id", None) if stream is not None else None
    started_at = metadata.get("started_at") or now

    if active and not _stream_references_active_session(stream, active):
        stale_reason = _session_stale_reason(active, metadata, now)
        if stale_reason:
            _stale_close_session(conn, active, now=now, reason=stale_reason)
            conn.commit()
            active = None

    if active:
        session_id = int(active["id"])
        conn.execute(
            """
            UPDATE stream_sessions
            SET twitch_stream_id = COALESCE(?, twitch_stream_id),
                title = COALESCE(?, title),
                category = COALESCE(?, category),
                game = COALESCE(?, game),
                started_at = COALESCE(started_at, ?),
                playthrough_type = COALESCE(?, playthrough_type),
                challenge = COALESCE(?, challenge),
                language_mode = COALESCE(?, language_mode),
                spoiler_policy = COALESCE(?, spoiler_policy),
                status = 'live',
                source = COALESCE(NULLIF(source, 'unknown'), ?),
                is_real_stream = 1,
                updated_at = ?
            WHERE id = ?
            """,
            (
                metadata.get("twitch_stream_id"),
                metadata.get("title"),
                metadata.get("category"),
                metadata.get("game"),
                started_at,
                metadata.get("playthrough_type"),
                metadata.get("challenge"),
                metadata.get("language_mode"),
                metadata.get("spoiler_policy"),
                stored_source,
                now,
                session_id,
            ),
        )
        print(f"[HEBE][STREAM_SESSION] reused_active id={session_id} reason=same_twitch_stream_id", flush=True)
    else:
        cur = conn.execute(
            """
            INSERT INTO stream_sessions (
                twitch_stream_id, title, category, game, started_at,
                playthrough_type, challenge, language_mode, spoiler_policy,
                status, source, is_real_stream, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'live', ?, 1, ?, ?)
            """,
            (
                metadata.get("twitch_stream_id"),
                metadata.get("title"),
                metadata.get("category"),
                metadata.get("game"),
                started_at,
                metadata.get("playthrough_type"),
                metadata.get("challenge"),
                metadata.get("language_mode"),
                metadata.get("spoiler_policy"),
                stored_source,
                now,
                now,
            ),
        )
        session_id = int(cur.lastrowid)
        print(
            "[HEBE][STREAM_SESSION] "
            f"created id={session_id} twitch_stream_id={metadata.get('twitch_stream_id')!r} title={metadata.get('title')!r} "
            f"game={(metadata.get('game') or metadata.get('category'))!r} source=twitch",
            flush=True,
        )

    conn.commit()
    conn.close()
    if stream is not None:
        setattr(stream, "active_stream_session_id", session_id)
    return session_id


def close_active_stream_session(stream: Any = None, *, reason: str = "offline") -> dict | None:
    ensure_stream_memory_ready()
    conn = db_sqlite.get_db_connection()
    active = get_active_stream_session(conn)
    if not active:
        conn.close()
        return None
    metadata = _stream_metadata(stream)
    active_stream_id = str(active.get("twitch_stream_id") or "").strip()
    current_stream_id = str(metadata.get("twitch_stream_id") or "").strip()
    if current_stream_id and active_stream_id and current_stream_id != active_stream_id:
        conn.close()
        print(
            f"[HEBE][STREAM_SESSION] close_skipped id={active['id']} reason=stream_id_mismatch",
            flush=True,
        )
        return None
    now = _now_iso()
    max_age = int(float(os.getenv("HEBE_STREAM_SESSION_MAX_SECONDS", "64800") or 64800))
    duration = min(_seconds_between(active.get("started_at"), now) or 0, max_age)
    conn.execute(
        """
        UPDATE stream_sessions
        SET ended_at = ?, duration_seconds = ?, status = 'ended', updated_at = ?
        WHERE id = ?
        """,
        (now, duration, now, active["id"]),
    )
    conn.commit()
    conn.close()
    if stream is not None:
        setattr(stream, "active_stream_session_id", None)
    summary = summarize_stream_session(int(active["id"]), reason=reason)
    print(f"[HEBE][STREAM_SESSION] closed id={active['id']} duration={duration or 0}", flush=True)
    return summary


def _event_identity_payload(payload: dict | None) -> dict[str, Any]:
    payload = payload or {}
    keys = (
        "id",
        "event_id",
        "twitch_event_id",
        "message_id",
        "user_login",
        "display_name",
        "viewer_count",
        "target_channel",
        "target_user_login",
        "raid_id",
        "started_at",
        "ended_at",
    )
    return {key: payload.get(key) for key in keys if payload.get(key) not in (None, "")}


def _stream_event_dedupe_key(event_type: str, payload: dict | None, event_ts: str) -> str:
    parsed = _parse_iso(event_ts) or datetime.now(timezone.utc)
    rounded = parsed.replace(microsecond=0, second=(parsed.second // 10) * 10)
    identity = _event_identity_payload(payload)
    if not identity:
        identity = {"payload": payload or {}}
    return _json({"type": event_type, "window": rounded.isoformat(), "identity": identity})


def _stream_event_is_duplicate(
    conn: sqlite3.Connection,
    *,
    session_id: int | None,
    event_type: str,
    payload: dict | None,
    event_ts: str,
    dedupe_key: str,
) -> bool:
    exact = conn.execute(
        """
        SELECT id FROM stream_events
        WHERE COALESCE(stream_session_id, 0) = COALESCE(?, 0)
          AND event_type = ?
          AND dedupe_key = ?
        LIMIT 1
        """,
        (session_id, event_type, dedupe_key),
    ).fetchone()
    if exact:
        return True
    event_time = _parse_iso(event_ts)
    if event_time is None:
        return False
    identity = _event_identity_payload(payload)
    if not identity:
        return False
    candidates = conn.execute(
        """
        SELECT payload_json, event_ts
        FROM stream_events
        WHERE COALESCE(stream_session_id, 0) = COALESCE(?, 0)
          AND event_type = ?
        ORDER BY event_ts DESC
        LIMIT 50
        """,
        (session_id, event_type),
    ).fetchall()
    for row in candidates:
        other_time = _parse_iso(row["event_ts"])
        if other_time is None or abs((event_time - other_time).total_seconds()) > 120:
            continue
        other_identity = _event_identity_payload(_loads(row["payload_json"], {}))
        if identity and other_identity == identity:
            return True
    return False


def record_stream_event(event_type: str, payload: dict | None = None, *, stream: Any = None) -> int:
    ensure_stream_memory_ready()
    payload = payload or {}
    if _source_is_dev_or_simulation(getattr(stream, "source", None), payload):
        print(f"[HEBE][STREAM_SESSION] skipped reason=simulation event_type={event_type}", flush=True)
        return 0
    session_id = getattr(stream, "active_stream_session_id", None) if stream is not None else None
    if not session_id:
        active = get_active_stream_session()
        session_id = active["id"] if active else None
    conn = db_sqlite.get_db_connection()
    event_ts = db_sqlite.normalize_iso(str(payload.get("event_ts") or payload.get("timestamp") or "")) or _now_iso()
    dedupe_key = _stream_event_dedupe_key(event_type, payload, event_ts)
    if _stream_event_is_duplicate(
        conn,
        session_id=int(session_id) if session_id else None,
        event_type=event_type,
        payload=payload,
        event_ts=event_ts,
        dedupe_key=dedupe_key,
    ):
        conn.close()
        print(f"[HEBE][STREAM_EVENT] duplicate_ignored type={event_type}", flush=True)
        return 0
    cur = conn.execute(
        """
        INSERT INTO stream_events (stream_session_id, event_type, event_ts, payload_json, dedupe_key, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, event_type, event_ts, _json(payload), dedupe_key, _now_iso()),
    )
    conn.commit()
    event_id = int(cur.lastrowid)
    conn.close()
    print(f"[HEBE][STREAM_EVENT] inserted id={event_id} type={event_type} session_id={session_id}", flush=True)
    return event_id


def observe_presence(
    username: str,
    display_name: str | None = None,
    *,
    stream_session_id: int | None = None,
    source: str = "chat",
    message_seen: bool = False,
    direct_interaction: bool = False,
    passive: bool = False,
) -> None:
    ensure_stream_memory_ready()
    user = _norm_user(username)
    if not user:
        return
    now = _now_iso()
    display = _display_name(user, display_name)
    conn = db_sqlite.get_db_connection()
    profile = conn.execute("SELECT * FROM chatter_profiles WHERE username = ?", (user,)).fetchone()
    was_new = profile is None
    returning = False
    if profile is not None:
        last_seen = _parse_iso(profile["last_seen_at"])
        if last_seen:
            days = (datetime.now(timezone.utc) - last_seen).days
            returning = days >= int(os.getenv("HEBE_RETURNING_AFTER_DAYS", "60") or 60)

    if profile is None:
        conn.execute(
            """
            INSERT INTO chatter_profiles (
                username, display_name, aliases_json, first_seen_at, last_seen_at,
                last_message_at, last_direct_interaction_at, last_lurk_seen_at,
                streams_seen_count, streams_chatted_count, total_messages,
                total_direct_interactions, total_lurk_sessions, viewer_status, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
            """,
            (
                user,
                display,
                _json([display]) if display and display.lower() != user else _json([]),
                now,
                now,
                now if message_seen else None,
                now if direct_interaction else None,
                now if passive and not message_seen else None,
                1 if message_seen else 0,
                1 if message_seen else 0,
                1 if direct_interaction else 0,
                1 if passive and not message_seen else 0,
                "new" if not _is_bot(user) else "bot",
                now,
            ),
        )
    else:
        aliases = _loads(profile["aliases_json"], [])
        if display and display not in aliases and display.lower() != user:
            aliases.append(display)
        conn.execute(
            """
            UPDATE chatter_profiles
            SET display_name = ?,
                aliases_json = ?,
                last_seen_at = ?,
                last_message_at = CASE WHEN ? THEN ? ELSE last_message_at END,
                last_direct_interaction_at = CASE WHEN ? THEN ? ELSE last_direct_interaction_at END,
                last_lurk_seen_at = CASE WHEN ? THEN ? ELSE last_lurk_seen_at END,
                total_messages = total_messages + ?,
                total_direct_interactions = total_direct_interactions + ?,
                total_lurk_sessions = total_lurk_sessions + ?,
                viewer_status = ?,
                updated_at = ?
            WHERE username = ?
            """,
            (
                display,
                _json(aliases),
                now,
                1 if message_seen else 0,
                now,
                1 if direct_interaction else 0,
                now,
                1 if passive and not message_seen else 0,
                now,
                1 if message_seen else 0,
                1 if direct_interaction else 0,
                1 if passive and not message_seen else 0,
                _viewer_status(profile, returning=returning, message_seen=message_seen, direct_interaction=direct_interaction),
                now,
                user,
            ),
        )

    if stream_session_id:
        current = conn.execute(
            "SELECT * FROM chatter_presence WHERE stream_session_id = ? AND username = ?",
            (stream_session_id, user),
        ).fetchone()
        if current is None:
            conn.execute(
                """
                INSERT INTO chatter_presence (
                    stream_session_id, username, display_name, first_seen_at, last_seen_at,
                    first_message_at, last_message_at, first_direct_interaction_at,
                    last_direct_interaction_at, message_count, direct_interaction_count,
                    was_present, was_active_chatter, was_passive_viewer, was_new_chatter,
                    was_returning_after_absence, presence_source_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stream_session_id,
                    user,
                    display,
                    now,
                    now,
                    now if message_seen else None,
                    now if message_seen else None,
                    now if direct_interaction else None,
                    now if direct_interaction else None,
                    1 if message_seen else 0,
                    1 if direct_interaction else 0,
                    1 if message_seen else 0,
                    1 if passive and not message_seen else 0,
                    1 if was_new else 0,
                    1 if returning else 0,
                    _json([source]),
                    now,
                    now,
                ),
            )
        else:
            sources = _loads(current["presence_source_json"], [])
            if source not in sources:
                sources.append(source)
            conn.execute(
                """
                UPDATE chatter_presence
                SET display_name = ?,
                    last_seen_at = ?,
                    first_message_at = COALESCE(first_message_at, CASE WHEN ? THEN ? ELSE NULL END),
                    last_message_at = CASE WHEN ? THEN ? ELSE last_message_at END,
                    first_direct_interaction_at = COALESCE(first_direct_interaction_at, CASE WHEN ? THEN ? ELSE NULL END),
                    last_direct_interaction_at = CASE WHEN ? THEN ? ELSE last_direct_interaction_at END,
                    message_count = message_count + ?,
                    direct_interaction_count = direct_interaction_count + ?,
                    was_active_chatter = CASE WHEN ? THEN 1 ELSE was_active_chatter END,
                    was_passive_viewer = CASE WHEN ? THEN 1 ELSE was_passive_viewer END,
                    was_returning_after_absence = CASE WHEN ? THEN 1 ELSE was_returning_after_absence END,
                    presence_source_json = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    display,
                    now,
                    1 if message_seen else 0,
                    now,
                    1 if message_seen else 0,
                    now,
                    1 if direct_interaction else 0,
                    now,
                    1 if direct_interaction else 0,
                    now,
                    1 if message_seen else 0,
                    1 if direct_interaction else 0,
                    1 if message_seen else 0,
                    1 if passive and not message_seen else 0,
                    1 if returning else 0,
                    _json(sources),
                    now,
                    current["id"],
                ),
            )
    conn.commit()
    conn.close()


def _viewer_status(profile: sqlite3.Row, *, returning: bool, message_seen: bool, direct_interaction: bool) -> str:
    if returning:
        return "returning_after_long_absence"
    total = int(profile["total_messages"] or 0) + (1 if message_seen else 0)
    interactions = int(profile["total_direct_interactions"] or 0) + (1 if direct_interaction else 0)
    streams_seen = int(profile["streams_seen_count"] or 0)
    if interactions >= 20 or total >= 120:
        return "core_regular"
    if streams_seen >= int(os.getenv("HEBE_REGULAR_MIN_STREAMS_30D", "3") or 3) or total >= 25:
        return "regular"
    if message_seen:
        return "active_chatter"
    if streams_seen >= int(os.getenv("HEBE_PASSIVE_VIEWER_MIN_SEEN_STREAMS", "3") or 3):
        return "passive_viewer"
    return profile["viewer_status"] or "occasional"


def record_chat_message(
    *,
    username: str,
    display_name: str | None,
    message_text: str,
    stream_session_id: int | None = None,
    is_mention_to_hebe: bool = False,
    is_direct_reply_to_hebe: bool = False,
    is_bot: bool | None = None,
    badges: dict | None = None,
    source: str = "twitch_irc",
    topic_hint: str | None = None,
    language_hint: str | None = None,
) -> int:
    ensure_stream_memory_ready()
    user = _norm_user(username)
    if not user or not str(message_text or "").strip():
        return 0
    if _source_is_dev_or_simulation(source):
        print("[HEBE][STREAM_SESSION] skipped reason=simulation chat_message=true", flush=True)
        return 0
    bot = _is_bot(user) if is_bot is None else bool(is_bot)
    if not stream_session_id:
        active = get_active_stream_session()
        stream_session_id = int(active["id"]) if active else None

    observed_at = _now_iso()
    conn = db_sqlite.get_db_connection()
    cur = conn.execute(
        """
        INSERT INTO stream_chat_messages (
            stream_session_id, username, display_name, message_text, observed_at,
            is_mention_to_hebe, is_direct_reply_to_hebe, is_bot,
            badges_json, source, topic_hint, language_hint, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            stream_session_id,
            user,
            _display_name(user, display_name),
            str(message_text).strip(),
            observed_at,
            1 if is_mention_to_hebe else 0,
            1 if is_direct_reply_to_hebe else 0,
            1 if bot else 0,
            _json(badges or {}),
            source,
            topic_hint,
            language_hint,
            observed_at,
        ),
    )
    message_id = int(cur.lastrowid)
    conn.commit()
    conn.close()

    observe_presence(
        user,
        display_name,
        stream_session_id=stream_session_id,
        source=source,
        message_seen=not bot,
        direct_interaction=is_mention_to_hebe or is_direct_reply_to_hebe,
    )
    if not bot:
        maybe_record_chatter_fact_from_message(
            username=user,
            message_text=message_text,
            source_message_id=message_id,
            stream_session_id=stream_session_id,
        )
    return message_id


def maybe_record_chatter_fact_from_message(
    *,
    username: str,
    message_text: str,
    source_message_id: int | None = None,
    stream_session_id: int | None = None,
) -> int | None:
    ensure_stream_memory_ready()
    user = _norm_user(username)
    text = str(message_text or "").strip()
    normalized = text.lower()
    if not user or not text:
        return None

    fact_text = None
    fact_type = None
    confidence = "low"
    if "linux" in normalized and ("windows 11" in normalized or "win11" in normalized):
        fact_text = f"{user} comentó que usa Linux porque su PC no acepta Windows 11."
        fact_type = "setup_pc"
        confidence = "medium"

    if not fact_text:
        return None

    now = _now_iso()
    conn = db_sqlite.get_db_connection()
    existing = conn.execute(
        """
        SELECT * FROM chatter_facts
        WHERE username = ? AND fact_type = ? AND fact_text = ?
        ORDER BY id DESC LIMIT 1
        """,
        (user, fact_type, fact_text),
    ).fetchone()
    if existing:
        evidence_count = int(existing["evidence_count"] or 1) + 1
        next_conf = "high" if evidence_count >= 3 else confidence
        conn.execute(
            """
            UPDATE chatter_facts
            SET evidence_count = ?, confidence = ?, last_confirmed_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (evidence_count, next_conf, now, now, existing["id"]),
        )
        fact_id = int(existing["id"])
    else:
        cur = conn.execute(
            """
            INSERT INTO chatter_facts (
                username, fact_text, fact_type, confidence, source_message_id,
                source_stream_session_id, evidence_count, first_observed_at,
                last_confirmed_at, public_reference_allowed, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, 1, ?, ?)
            """,
            (user, fact_text, fact_type, confidence, source_message_id, stream_session_id, now, now, now, now),
        )
        fact_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return fact_id


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table_name,)).fetchone()
    return row is not None


def _session_metadata_for_summary(conn: sqlite3.Connection, session: sqlite3.Row) -> tuple[dict[str, Any], list[str]]:
    metadata = {
        "title": _clean_text(session["title"], max_len=300),
        "category": _clean_text(session["category"], max_len=160),
        "game": _clean_text(session["game"], max_len=160),
        "twitch_stream_id": _clean_text(session["twitch_stream_id"], max_len=80),
        "started_at": session["started_at"],
        "ended_at": session["ended_at"],
        "duration_seconds": session["duration_seconds"],
        "playthrough_type": _clean_text(session["playthrough_type"], max_len=80),
        "challenge": _clean_text(session["challenge"], max_len=160),
        "language_mode": _clean_text(session["language_mode"], max_len=60),
        "spoiler_policy": _clean_text(session["spoiler_policy"], max_len=80),
        "source": _clean_text(session["source"], max_len=80) or "unknown",
        "is_real_stream": bool(session["is_real_stream"]),
    }
    if not metadata["game"] and metadata["category"]:
        metadata["game"] = metadata["category"]

    if _metadata_missing(metadata) and _table_exists(conn, "live_session_state"):
        live_row = conn.execute(
            """
            SELECT current_title, current_category, current_game, language_mode, spoiler_policy
            FROM live_session_state
            WHERE stream_session_id = ?
            ORDER BY updated_at DESC LIMIT 1
            """,
            (session["id"],),
        ).fetchone()
        if live_row:
            metadata["title"] = metadata["title"] or _clean_text(live_row["current_title"], max_len=300)
            metadata["category"] = metadata["category"] or _clean_text(live_row["current_category"], max_len=160)
            metadata["game"] = metadata["game"] or _clean_text(live_row["current_game"], max_len=160) or metadata["category"]
            metadata["language_mode"] = metadata["language_mode"] or _clean_text(live_row["language_mode"], max_len=60)
            metadata["spoiler_policy"] = metadata["spoiler_policy"] or _clean_text(live_row["spoiler_policy"], max_len=80)

    if _metadata_missing(metadata):
        try:
            from app.stream import session_primer

            started = _parse_iso(str(session["started_at"] or "")) or datetime.now(timezone.utc)
            schedule = session_primer.get_schedule_for_date(started)
        except Exception:
            schedule = None
        if schedule:
            metadata["category"] = metadata["category"] or _clean_text(schedule.get("category"), max_len=160)
            metadata["game"] = metadata["game"] or _clean_text(schedule.get("game"), max_len=160) or metadata["category"]
            metadata["playthrough_type"] = metadata["playthrough_type"] or _clean_text(schedule.get("playthrough_type"), max_len=80)
            metadata["source"] = metadata["source"] or "schedule"

    missing = []
    if not metadata["title"]:
        missing.append("title")
    if not (metadata["game"] or metadata["category"]):
        missing.append("game_or_category")
    print(
        "[HEBE][STREAM_SUMMARY] using_metadata "
        f"session_id={session['id']} title={metadata.get('title')!r} "
        f"category={metadata.get('category')!r} game={metadata.get('game')!r}",
        flush=True,
    )
    if missing:
        print(f"[HEBE][STREAM_SUMMARY] metadata_missing session_id={session['id']} missing={missing}", flush=True)
    return metadata, missing


def _live_timeline_for_summary(conn: sqlite3.Connection, stream_session_id: int) -> list[dict[str, Any]]:
    if not _table_exists(conn, "live_session_timeline"):
        return []
    rows = conn.execute(
        """
        SELECT event_type, event_ts, raw_text, topic, category, payload_json
        FROM live_session_timeline
        WHERE stream_session_id = ?
        ORDER BY event_ts ASC
        LIMIT 200
        """,
        (stream_session_id,),
    ).fetchall()
    return [
        {
            "type": row["event_type"],
            "ts": row["event_ts"],
            "text": row["raw_text"],
            "topic": row["topic"],
            "category": row["category"],
            "payload": _loads(row["payload_json"], {}),
        }
        for row in rows
    ]


def summarize_stream_session(stream_session_id: int, *, reason: str = "manual") -> dict | None:
    ensure_stream_memory_ready()
    conn = db_sqlite.get_db_connection()
    session = conn.execute("SELECT * FROM stream_sessions WHERE id = ?", (stream_session_id,)).fetchone()
    if not session:
        conn.close()
        return None
    existing = conn.execute(
        "SELECT * FROM stream_summaries WHERE stream_session_id = ? ORDER BY id DESC LIMIT 1",
        (stream_session_id,),
    ).fetchone()
    messages = conn.execute(
        """
        SELECT * FROM stream_chat_messages
        WHERE stream_session_id = ? AND is_bot = 0
        ORDER BY observed_at ASC
        """,
        (stream_session_id,),
    ).fetchall()
    events = conn.execute(
        """
        SELECT event_type, payload_json, event_ts FROM stream_events
        WHERE stream_session_id = ?
        ORDER BY event_ts ASC
        """,
        (stream_session_id,),
    ).fetchall()
    timeline = _live_timeline_for_summary(conn, stream_session_id)
    by_user: dict[str, list[sqlite3.Row]] = {}
    topics: dict[str, int] = {}
    for msg in messages:
        by_user.setdefault(msg["username"], []).append(msg)
        if msg["topic_hint"]:
            topics[msg["topic_hint"]] = topics.get(msg["topic_hint"], 0) + 1

    chatter_highlights = []
    now = _now_iso()
    conn.execute("DELETE FROM stream_chatter_summaries WHERE stream_session_id = ?", (stream_session_id,))
    for user, user_msgs in sorted(by_user.items(), key=lambda item: len(item[1]), reverse=True):
        if len(user_msgs) < 2:
            continue
        sample = [m["message_text"] for m in user_msgs[:3]]
        summary = f"{user} participó con {len(user_msgs)} mensajes. Temas: {', '.join(sorted({m['topic_hint'] for m in user_msgs if m['topic_hint']}) or ['general'])}."
        conn.execute(
            """
            INSERT INTO stream_chatter_summaries (
                stream_session_id, username, display_name, message_count,
                direct_interaction_count, summary_text, topics_json,
                notable_quotes_json, inferred_facts_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stream_session_id,
                user,
                user_msgs[-1]["display_name"],
                len(user_msgs),
                sum(int(m["is_mention_to_hebe"] or 0) + int(m["is_direct_reply_to_hebe"] or 0) for m in user_msgs),
                summary,
                _json(sorted({m["topic_hint"] for m in user_msgs if m["topic_hint"]})),
                _json(sample),
                _json([]),
                now,
            ),
        )
        chatter_highlights.append({"username": user, "message_count": len(user_msgs), "summary": summary})

    metadata, missing = _session_metadata_for_summary(conn, session)
    title = metadata.get("title") or "unknown title"
    game = metadata.get("game") or metadata.get("category") or "unknown category"
    raids = [_loads(e["payload_json"], {}) for e in events if e["event_type"] == "twitch_raid"]
    shoutouts = [_loads(e["payload_json"], {}) for e in events if e["event_type"] == "twitch_shoutout"]
    key_events = [{"type": e["event_type"], "ts": e["event_ts"], "payload": _event_identity_payload(_loads(e["payload_json"], {}))} for e in events]
    key_events.extend({"type": item["type"], "ts": item["ts"], "topic": item["topic"], "category": item["category"]} for item in timeline)
    summary_text = (
        f"Stream de {game}. Título: {title}. "
        f"Mensajes reales observados: {len(messages)}. "
        f"Chatters activos: {len(by_user)}. "
        f"Eventos registrados: {len(events)}. "
        f"Timeline RAG: {len(timeline)}. Finalizado por: {reason}."
    )

    payload = {
        "summary_text": summary_text,
        "key_events_json": _json(key_events),
        "game_progress_json": _json({
            "title": title,
            "game": game,
            "category": metadata.get("category"),
            "playthrough_type": metadata.get("playthrough_type"),
            "challenge": metadata.get("challenge"),
            "timeline_event_count": len(timeline),
            "metadata_missing": missing,
        }),
        "chat_topics_json": _json(topics),
        "chatter_highlights_json": _json(chatter_highlights[:10]),
        "raids_json": _json(raids),
        "shoutouts_json": _json(shoutouts),
        "next_stream_context": "",
        "metadata_json": _json(metadata),
    }
    if existing:
        conn.execute(
            """
            UPDATE stream_summaries
            SET summary_text = ?, key_events_json = ?, game_progress_json = ?,
                chat_topics_json = ?, chatter_highlights_json = ?, raids_json = ?,
                shoutouts_json = ?, next_stream_context = ?, metadata_json = ?, created_at = ?
            WHERE id = ?
            """,
            (
                payload["summary_text"],
                payload["key_events_json"],
                payload["game_progress_json"],
                payload["chat_topics_json"],
                payload["chatter_highlights_json"],
                payload["raids_json"],
                payload["shoutouts_json"],
                payload["next_stream_context"],
                payload["metadata_json"],
                now,
                existing["id"],
            ),
        )
        summary_id = int(existing["id"])
    else:
        cur = conn.execute(
            """
            INSERT INTO stream_summaries (
                stream_session_id, summary_text, key_events_json, game_progress_json,
                chat_topics_json, chatter_highlights_json, raids_json, shoutouts_json,
                next_stream_context, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stream_session_id,
                payload["summary_text"],
                payload["key_events_json"],
                payload["game_progress_json"],
                payload["chat_topics_json"],
                payload["chatter_highlights_json"],
                payload["raids_json"],
                payload["shoutouts_json"],
                payload["next_stream_context"],
                payload["metadata_json"],
                now,
            ),
        )
        summary_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    if existing:
        print(f"[HEBE][STREAM_SUMMARY] regenerated session_id={stream_session_id}", flush=True)
    else:
        print(f"[HEBE][STREAM_SUMMARY] generated session_id={stream_session_id}", flush=True)
    return {"id": summary_id, "stream_session_id": stream_session_id, **payload}


def get_latest_stream_summary() -> dict | None:
    ensure_stream_memory_ready()
    conn = db_sqlite.get_db_connection()
    row = conn.execute(
        """
        SELECT ss.*, s.summary_text, s.chat_topics_json, s.chatter_highlights_json, s.created_at AS summary_created_at
        FROM stream_summaries s
        JOIN stream_sessions ss ON ss.id = s.stream_session_id
        ORDER BY s.id DESC
        LIMIT 1
        """
    ).fetchone()
    conn.close()
    return _row(row)


def _count(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    return int(conn.execute(sql, params).fetchone()[0] or 0)


def _latest_session(conn: sqlite3.Connection) -> dict | None:
    return _row(conn.execute("SELECT * FROM stream_sessions ORDER BY COALESCE(started_at, created_at) DESC, id DESC LIMIT 1").fetchone())


def _latest_summary(conn: sqlite3.Connection) -> dict | None:
    return _row(
        conn.execute(
            """
            SELECT s.*, ss.title, ss.game, ss.category, ss.status, ss.source, ss.is_real_stream
            FROM stream_summaries s
            LEFT JOIN stream_sessions ss ON ss.id = s.stream_session_id
            ORDER BY s.created_at DESC, s.id DESC LIMIT 1
            """
        ).fetchone()
    )


def _duplicate_event_groups(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT COALESCE(stream_session_id, 0) AS stream_session_id,
               event_type,
               COALESCE(dedupe_key, payload_json, '') AS identity_key,
               COUNT(*) AS count,
               GROUP_CONCAT(id) AS ids
        FROM stream_events
        GROUP BY COALESCE(stream_session_id, 0), event_type, COALESCE(dedupe_key, payload_json, '')
        HAVING COUNT(*) > 1 AND COALESCE(dedupe_key, payload_json, '') != ''
        ORDER BY count DESC
        LIMIT 100
        """
    ).fetchall()
    return [
        {
            "stream_session_id": row["stream_session_id"],
            "event_type": row["event_type"],
            "count": int(row["count"] or 0),
            "ids": [int(part) for part in str(row["ids"] or "").split(",") if part],
        }
        for row in rows
    ]


def stream_data_health() -> dict[str, Any]:
    init_stream_memory_schema()
    conn = db_sqlite.get_db_connection()
    sessions = conn.execute("SELECT * FROM stream_sessions").fetchall()
    summaries = conn.execute("SELECT * FROM stream_summaries").fetchall()
    missing_sessions = [int(row["id"]) for row in sessions if bool(row["is_real_stream"]) and _metadata_missing(row)]
    duplicate_groups = _duplicate_event_groups(conn)
    active = get_active_stream_session(conn)
    latest_session = _latest_session(conn)
    latest_summary = _latest_summary(conn)
    warnings: list[str] = []
    if active and _metadata_missing(active):
        warnings.append("active stream session is missing metadata")
    if duplicate_groups:
        warnings.append("possible duplicate stream events detected")
    summaries_without_session = _count(
        conn,
        """
        SELECT COUNT(*)
        FROM stream_summaries s
        LEFT JOIN stream_sessions ss ON ss.id = s.stream_session_id
        WHERE ss.id IS NULL
        """,
    )
    if summaries_without_session:
        warnings.append("summaries without session links detected")
    payload = {
        "sessions_total": len(sessions),
        "real_sessions": sum(1 for row in sessions if bool(row["is_real_stream"])),
        "active_session": active,
        "sessions_missing_metadata": len(missing_sessions),
        "sessions_missing_metadata_ids": missing_sessions[:50],
        "summaries_total": len(summaries),
        "summaries_missing_metadata": sum(
            1
            for row in summaries
            if not _clean_text(row["metadata_json"], max_len=20)
            or row["stream_session_id"] in set(missing_sessions)
        ),
        "sessions_without_summary": _count(
            conn,
            """
            SELECT COUNT(*)
            FROM stream_sessions ss
            LEFT JOIN stream_summaries s ON s.stream_session_id = ss.id
            WHERE COALESCE(ss.is_real_stream, 1) = 1
              AND ss.status = 'ended'
              AND s.id IS NULL
            """,
        ),
        "summaries_without_session": summaries_without_session,
        "chat_messages_without_session": _count(conn, "SELECT COUNT(*) FROM stream_chat_messages WHERE stream_session_id IS NULL"),
        "events_without_session": _count(conn, "SELECT COUNT(*) FROM stream_events WHERE stream_session_id IS NULL"),
        "possible_duplicate_events": sum(max(0, int(group["count"]) - 1) for group in duplicate_groups),
        "possible_duplicate_event_groups": duplicate_groups[:20],
        "dev_simulation_sessions": _count(
            conn,
            """
            SELECT COUNT(*)
            FROM stream_sessions
            WHERE COALESCE(is_real_stream, 1) = 0
               OR lower(COALESCE(source, '')) LIKE '%sim%'
               OR lower(COALESCE(source, '')) LIKE '%dev%'
               OR lower(COALESCE(source, '')) LIKE '%test%'
            """,
        ),
        "latest_session": latest_session,
        "latest_summary": latest_summary,
        "warnings": warnings,
    }
    conn.close()
    print(
        "[HEBE][STREAM_DATA_HEALTH] "
        f"sessions_total={payload['sessions_total']} real_sessions={payload['real_sessions']} "
        f"missing_metadata={payload['sessions_missing_metadata']} duplicate_events={payload['possible_duplicate_events']}",
        flush=True,
    )
    return payload


def _session_evidence_counts(conn: sqlite3.Connection, session_id: int) -> dict[str, int]:
    return {
        "chat_messages": _count(conn, "SELECT COUNT(*) FROM stream_chat_messages WHERE stream_session_id = ?", (session_id,)),
        "events": _count(conn, "SELECT COUNT(*) FROM stream_events WHERE stream_session_id = ?", (session_id,)),
        "chatter_summaries": _count(conn, "SELECT COUNT(*) FROM stream_chatter_summaries WHERE stream_session_id = ?", (session_id,)),
    }


def _repair_event_dedupe_keys(conn: sqlite3.Connection, *, dry_run: bool) -> int:
    rows = conn.execute("SELECT id, event_type, event_ts, payload_json, dedupe_key FROM stream_events").fetchall()
    changed = 0
    for row in rows:
        payload = _loads(row["payload_json"], {})
        next_key = _stream_event_dedupe_key(row["event_type"], payload, row["event_ts"] or _now_iso())
        if row["dedupe_key"] == next_key:
            continue
        changed += 1
        if not dry_run:
            conn.execute("UPDATE stream_events SET dedupe_key = ? WHERE id = ?", (next_key, row["id"]))
    return changed


def repair_stream_data(*, dry_run: bool = True, regenerate_summaries: bool = True) -> dict[str, Any]:
    init_stream_memory_schema()
    conn = db_sqlite.get_db_connection()
    sessions = conn.execute("SELECT * FROM stream_sessions ORDER BY id ASC").fetchall()
    sessions_repaired = 0
    sessions_marked_unknown = 0
    warnings: list[str] = []
    regenerate_ids: set[int] = set()

    for session in sessions:
        session_id = int(session["id"])
        metadata, missing = _session_metadata_for_summary(conn, session)
        updates: dict[str, Any] = {}
        for column in ("title", "category", "game", "playthrough_type", "challenge", "language_mode", "spoiler_policy"):
            if not _clean_text(session[column]) and metadata.get(column):
                updates[column] = metadata[column]
        if updates:
            sessions_repaired += 1
            regenerate_ids.add(session_id)
            if not dry_run:
                assignments = ", ".join(f"{column} = ?" for column in updates)
                conn.execute(
                    f"UPDATE stream_sessions SET {assignments}, updated_at = ? WHERE id = ?",
                    (*updates.values(), _now_iso(), session_id),
                )
        if missing and bool(session["is_real_stream"]) and session["status"] != "live":
            evidence = _session_evidence_counts(conn, session_id)
            if sum(evidence.values()) == 0 and not _clean_text(session["twitch_stream_id"]):
                sessions_marked_unknown += 1
                sessions_repaired += 1
                if not dry_run:
                    conn.execute(
                        """
                        UPDATE stream_sessions
                        SET source = 'unknown', is_real_stream = 0, status = 'unknown', updated_at = ?
                        WHERE id = ?
                        """,
                        (_now_iso(), session_id),
                    )
            else:
                warnings.append(f"session {session_id} missing metadata but has evidence; left as real")

    dedupe_keys_repaired = _repair_event_dedupe_keys(conn, dry_run=dry_run)
    duplicate_groups = _duplicate_event_groups(conn)
    duplicate_ids: list[int] = []
    for group in duplicate_groups:
        ids = sorted(group["ids"])
        duplicate_ids.extend(ids[1:])

    if duplicate_ids and not dry_run:
        placeholders = ",".join("?" for _ in duplicate_ids)
        conn.execute(f"DELETE FROM stream_events WHERE id IN ({placeholders})", tuple(duplicate_ids))

    ended_without_summary = [
        int(row["id"])
        for row in conn.execute(
            """
            SELECT ss.id
            FROM stream_sessions ss
            LEFT JOIN stream_summaries s ON s.stream_session_id = ss.id
            WHERE COALESCE(ss.is_real_stream, 1) = 1
              AND ss.status = 'ended'
              AND s.id IS NULL
            """
        ).fetchall()
    ]
    regenerate_ids.update(ended_without_summary)
    if not regenerate_summaries:
        regenerate_ids.clear()

    if dry_run:
        conn.close()
    else:
        conn.commit()
        conn.close()
        for session_id in sorted(regenerate_ids):
            summarize_stream_session(session_id, reason="repair")

    result = {
        "dry_run": bool(dry_run),
        "sessions_checked": len(sessions),
        "sessions_repaired": sessions_repaired,
        "sessions_marked_unknown": sessions_marked_unknown,
        "summaries_regenerated": len(regenerate_ids),
        "duplicate_events_found": len(duplicate_ids),
        "duplicate_events_removed_or_marked": 0 if dry_run else len(duplicate_ids),
        "dedupe_keys_repaired": dedupe_keys_repaired,
        "warnings": warnings,
    }
    print(
        "[HEBE][STREAM_DATA_REPAIR] "
        f"dry_run={dry_run} sessions_checked={result['sessions_checked']} "
        f"sessions_repaired={sessions_repaired} duplicates={len(duplicate_ids)} "
        f"summaries_regenerated={len(regenerate_ids)}",
        flush=True,
    )
    return result


def get_chatter_profile(username: str) -> dict | None:
    ensure_stream_memory_ready()
    user = _norm_user(username)
    if not user:
        return None
    conn = db_sqlite.get_db_connection()
    profile = _row(conn.execute("SELECT * FROM chatter_profiles WHERE username = ?", (user,)).fetchone())
    if profile:
        profile["facts"] = [
            _row(row)
            for row in conn.execute(
                """
                SELECT * FROM chatter_facts
                WHERE username = ?
                ORDER BY last_confirmed_at DESC, id DESC
                LIMIT 10
                """,
                (user,),
            ).fetchall()
        ]
    conn.close()
    return profile


def list_recent_chatter_names(limit: int = 80) -> list[str]:
    ensure_stream_memory_ready()
    conn = db_sqlite.get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT username, display_name
            FROM chatter_profiles
            ORDER BY COALESCE(last_message_at, last_seen_at, updated_at, first_seen_at) DESC
            LIMIT ?
            """,
            (max(1, min(int(limit or 80), 250)),),
        ).fetchall()
    except Exception:
        conn.close()
        return []
    conn.close()

    names: list[str] = []
    for row in rows:
        for key in ("username", "display_name"):
            value = str(row[key] or "").strip()
            if value and value.lower() not in {item.lower() for item in names}:
                names.append(value)
    return names


def get_last_chatter_summary(username: str) -> dict | None:
    ensure_stream_memory_ready()
    user = _norm_user(username)
    if not user:
        return None
    conn = db_sqlite.get_db_connection()
    row = conn.execute(
        """
        SELECT scs.*, ss.title, ss.game, ss.category, ss.started_at
        FROM stream_chatter_summaries scs
        JOIN stream_sessions ss ON ss.id = scs.stream_session_id
        WHERE scs.username = ?
        ORDER BY scs.id DESC
        LIMIT 1
        """,
        (user,),
    ).fetchone()
    conn.close()
    return _row(row)


def format_chatter_profile_reply(username: str) -> str:
    profile = get_chatter_profile(username)
    user = _norm_user(username)
    if not profile:
        return f"No tengo memoria de {username} todavia."
    facts = profile.get("facts") or []
    fact_lines = [
        f"- {fact['fact_text']} (confianza: {fact['confidence']}, evidencias: {fact['evidence_count']})"
        for fact in facts
        if fact and int(fact.get("public_reference_allowed") or 0)
    ]
    return (
        f"Esto se de {profile.get('display_name') or user}:\n\n"
        f"* Estado: {profile.get('viewer_status') or 'sin clasificar'}.\n"
        f"* Ultima vez visto: {profile.get('last_seen_at') or 'nunca'}.\n"
        f"* Ultima vez que hablo: {profile.get('last_message_at') or 'nunca'}.\n"
        f"* Ultima interaccion conmigo: {profile.get('last_direct_interaction_at') or 'nunca'}.\n"
        f"* Mensajes totales: {profile.get('total_messages') or 0}.\n"
        f"* Hechos recordables:\n{chr(10).join(fact_lines) if fact_lines else '- ninguno con evidencia suficiente'}"
    )


def format_last_seen_reply(username: str, *, kind: str) -> str:
    profile = get_chatter_profile(username)
    if not profile:
        return f"No tengo registro de {username} todavia."
    field = "last_message_at" if kind == "message" else "last_seen_at"
    label = "hablo" if kind == "message" else "lo vi por aqui"
    value = profile.get(field)
    if not value:
        return f"Tengo a {username} en memoria, pero no tengo fecha de cuando {label}."
    return f"La ultima vez que {label} {username} fue: {value}."
