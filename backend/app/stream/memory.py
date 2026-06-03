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


def _is_bot(username: str | None) -> bool:
    configured = os.getenv("HEBE_TWITCH_BOT_USERNAMES", "")
    bots = set(BOT_USERNAMES)
    bots.update(part.strip().lower().lstrip("@") for part in configured.split(",") if part.strip())
    return _norm_user(username) in bots


def init_stream_memory_schema() -> None:
    conn = db_sqlite.get_db_connection()
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
            created_at TEXT NOT NULL,
            FOREIGN KEY(stream_session_id) REFERENCES stream_sessions(id)
        );

        CREATE TABLE IF NOT EXISTS stream_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_session_id INTEGER,
            event_type TEXT NOT NULL,
            event_ts TEXT NOT NULL,
            payload_json TEXT,
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
    conn.close()


def get_active_stream_session(conn: sqlite3.Connection | None = None) -> dict | None:
    close = conn is None
    conn = conn or db_sqlite.get_db_connection()
    row = conn.execute(
        "SELECT * FROM stream_sessions WHERE status = 'live' ORDER BY started_at DESC, id DESC LIMIT 1"
    ).fetchone()
    if close:
        conn.close()
    return _row(row)


def ensure_active_stream_session(stream: Any = None, *, source: str = "unknown") -> int:
    conn = db_sqlite.get_db_connection()
    now = _now_iso()
    active = get_active_stream_session(conn)
    title = getattr(stream, "current_stream_title", None) if stream is not None else None
    category = getattr(stream, "current_category", None) if stream is not None else None
    game = getattr(stream, "current_game", None) if stream is not None else None
    started_at = getattr(stream, "stream_started_at", None) if stream is not None else None
    started_at = db_sqlite.normalize_iso(started_at) if started_at else now

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
                updated_at = ?
            WHERE id = ?
            """,
            (
                getattr(stream, "twitch_stream_id", None) if stream is not None else None,
                title,
                category,
                game,
                started_at,
                getattr(stream, "current_playthrough_type", None) if stream is not None else None,
                getattr(stream, "current_challenge", None) if stream is not None else None,
                getattr(stream, "language_mode", None) if stream is not None else None,
                getattr(stream, "spoiler_policy", None) if stream is not None else None,
                now,
                session_id,
            ),
        )
    else:
        cur = conn.execute(
            """
            INSERT INTO stream_sessions (
                twitch_stream_id, title, category, game, started_at,
                playthrough_type, challenge, language_mode, spoiler_policy,
                status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'live', ?, ?)
            """,
            (
                getattr(stream, "twitch_stream_id", None) if stream is not None else None,
                title,
                category,
                game,
                started_at,
                getattr(stream, "current_playthrough_type", None) if stream is not None else None,
                getattr(stream, "current_challenge", None) if stream is not None else None,
                getattr(stream, "language_mode", None) if stream is not None else None,
                getattr(stream, "spoiler_policy", "no_spoilers") if stream is not None else "no_spoilers",
                now,
                now,
            ),
        )
        session_id = int(cur.lastrowid)
        print(f"[HEBE][STREAM_MEMORY] created stream_session id={session_id} source={source}", flush=True)

    conn.commit()
    conn.close()
    if stream is not None:
        setattr(stream, "active_stream_session_id", session_id)
    return session_id


def close_active_stream_session(stream: Any = None, *, reason: str = "offline") -> dict | None:
    conn = db_sqlite.get_db_connection()
    active = get_active_stream_session(conn)
    if not active:
        conn.close()
        return None
    now = _now_iso()
    duration = _seconds_between(active.get("started_at"), now)
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
    print(f"[HEBE][STREAM_MEMORY] closed stream_session id={active['id']} reason={reason}", flush=True)
    return summary


def record_stream_event(event_type: str, payload: dict | None = None, *, stream: Any = None) -> int:
    session_id = getattr(stream, "active_stream_session_id", None) if stream is not None else None
    if not session_id:
        active = get_active_stream_session()
        session_id = active["id"] if active else None
    conn = db_sqlite.get_db_connection()
    cur = conn.execute(
        """
        INSERT INTO stream_events (stream_session_id, event_type, event_ts, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, event_type, _now_iso(), _json(payload or {}), _now_iso()),
    )
    conn.commit()
    event_id = int(cur.lastrowid)
    conn.close()
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
    user = _norm_user(username)
    if not user or not str(message_text or "").strip():
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


def summarize_stream_session(stream_session_id: int, *, reason: str = "manual") -> dict | None:
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
    by_user: dict[str, list[sqlite3.Row]] = {}
    topics: dict[str, int] = {}
    for msg in messages:
        by_user.setdefault(msg["username"], []).append(msg)
        if msg["topic_hint"]:
            topics[msg["topic_hint"]] = topics.get(msg["topic_hint"], 0) + 1

    chatter_highlights = []
    now = _now_iso()
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

    title = session["title"] or "sin título"
    game = session["game"] or session["category"] or "sin categoría"
    summary_text = (
        f"Stream de {game}. Título: {title}. "
        f"Mensajes reales observados: {len(messages)}. "
        f"Chatters activos: {len(by_user)}. Finalizado por: {reason}."
    )

    payload = {
        "summary_text": summary_text,
        "key_events_json": _json([{"type": e["event_type"], "ts": e["event_ts"]} for e in events]),
        "game_progress_json": _json({"title": title, "game": game}),
        "chat_topics_json": _json(topics),
        "chatter_highlights_json": _json(chatter_highlights[:10]),
        "raids_json": _json([_loads(e["payload_json"], {}) for e in events if e["event_type"] == "twitch_raid"]),
        "shoutouts_json": _json([_loads(e["payload_json"], {}) for e in events if e["event_type"] == "twitch_shoutout"]),
        "next_stream_context": "",
    }
    if existing:
        conn.execute(
            """
            UPDATE stream_summaries
            SET summary_text = ?, key_events_json = ?, game_progress_json = ?,
                chat_topics_json = ?, chatter_highlights_json = ?, raids_json = ?,
                shoutouts_json = ?, next_stream_context = ?, created_at = ?
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
                next_stream_context, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                now,
            ),
        )
        summary_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return {"id": summary_id, "stream_session_id": stream_session_id, **payload}


def get_latest_stream_summary() -> dict | None:
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


def get_chatter_profile(username: str) -> dict | None:
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
