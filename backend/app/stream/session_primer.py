from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.services import db_sqlite


DEFAULT_TIMEZONE = os.getenv("HEBE_STREAM_TIMEZONE", "Europe/Madrid")


WEEKDAY_KEYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


DEFAULT_SCHEDULE = [
    {
        "weekday": "monday",
        "slot_name": "Monday Challenge",
        "game": "FINAL FANTASY IX",
        "category": "FINAL FANTASY IX",
        "playthrough_type": "Lv. 1 Challenge",
        "tags": ["ENG/ESP", "Challenge"],
        "title_style": "leo_standard",
        "enabled": True,
    },
    {
        "weekday": "tuesday",
        "slot_name": "Persona Week",
        "game": "Persona 5 Royal",
        "category": "Persona 5 Royal",
        "playthrough_type": "First Playthrough",
        "tags": ["ENG/ESP", "No spoilers"],
        "title_style": "leo_standard",
        "enabled": True,
    },
    {
        "weekday": "wednesday",
        "slot_name": "Persona Week",
        "game": "Persona 5 Royal",
        "category": "Persona 5 Royal",
        "playthrough_type": "First Playthrough",
        "tags": ["ENG/ESP", "No spoilers"],
        "title_style": "leo_standard",
        "enabled": True,
    },
    {
        "weekday": "thursday",
        "slot_name": "Chat Playthrough",
        "game": "Baldur's Gate 3",
        "category": "Baldur's Gate 3",
        "playthrough_type": "Chat Playthrough",
        "tags": ["ENG/ESP", "Chat"],
        "title_style": "leo_standard",
        "enabled": True,
    },
    {
        "weekday": "friday",
        "slot_name": "Chat Playthrough",
        "game": "Baldur's Gate 3",
        "category": "Baldur's Gate 3",
        "playthrough_type": "Chat Playthrough",
        "tags": ["ENG/ESP", "Chat"],
        "title_style": "leo_standard",
        "enabled": True,
    },
    {
        "weekday": "saturday",
        "slot_name": "Retro Weekend",
        "game": "Retro Weekend",
        "category": "Retro",
        "playthrough_type": "Retro Weekend",
        "tags": ["ENG/ESP", "Retro"],
        "title_style": "leo_standard",
        "enabled": True,
    },
    {
        "weekday": "sunday",
        "slot_name": "Retro Weekend",
        "game": "Retro Weekend",
        "category": "Retro",
        "playthrough_type": "Retro Weekend",
        "tags": ["ENG/ESP", "Retro"],
        "title_style": "leo_standard",
        "enabled": True,
    },
]


@dataclass
class StreamSessionPrimer:
    local_now: str
    weekday: str
    timezone: str
    slot_name: str
    game: str
    category: str
    playthrough_type: str
    challenge_type: str
    last_session_summary: str
    starting_point: str
    likely_objective: str
    spoiler_policy: str
    safe_context_for_spontaneity: list[str]
    title_suggestions: list[str]
    missing_info: list[str]
    schedule: dict[str, Any]
    last_session: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScheduleHypothesis:
    weekday: str
    time_window: str
    canonical_content: str
    stream_format: str
    source: str
    confidence: float
    evidence_count: int
    consecutive_matches: int
    consecutive_misses: int
    first_observed_at: str
    last_observed_at: str
    status: str
    superseded_by: int | None = None


@dataclass(frozen=True)
class ScheduleObservation:
    stream_session_id: str
    weekday: str
    time_window: str
    canonical_content: str
    stream_format: str
    source: str
    observed_at: str


def _now_iso() -> str:
    return datetime.now(ZoneInfo(DEFAULT_TIMEZONE)).isoformat()


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


def normalize_game_key(game: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(game or "").casefold()).strip()


def init_session_primer_schema() -> None:
    conn = db_sqlite.get_db_connection()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS stream_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            weekday TEXT NOT NULL,
            slot_name TEXT,
            game TEXT NOT NULL,
            category TEXT,
            playthrough_type TEXT,
            tags_json TEXT,
            title_style TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL,
            UNIQUE(weekday, slot_name)
        );

        CREATE TABLE IF NOT EXISTS game_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game TEXT NOT NULL,
            game_key TEXT NOT NULL,
            stream_date TEXT,
            title TEXT,
            started_at TEXT,
            ended_at TEXT,
            start_summary TEXT,
            end_summary TEXT,
            current_location TEXT,
            current_objective TEXT,
            important_events_json TEXT,
            unresolved_threads_json TEXT,
            next_time_plan TEXT,
            spoiler_policy TEXT,
            source TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_stream_schedule_weekday ON stream_schedule(weekday, enabled);
        CREATE INDEX IF NOT EXISTS idx_game_sessions_game_key ON game_sessions(game_key, stream_date, updated_at);

        CREATE TABLE IF NOT EXISTS schedule_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream_session_id TEXT NOT NULL UNIQUE,
            weekday TEXT NOT NULL,
            time_window TEXT NOT NULL,
            canonical_content TEXT NOT NULL,
            content_key TEXT NOT NULL,
            stream_format TEXT NOT NULL,
            source TEXT NOT NULL,
            observed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS schedule_hypotheses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            weekday TEXT NOT NULL,
            time_window TEXT NOT NULL,
            canonical_content TEXT NOT NULL,
            content_key TEXT NOT NULL,
            stream_format TEXT NOT NULL,
            source TEXT NOT NULL,
            confidence REAL NOT NULL,
            evidence_count INTEGER NOT NULL,
            consecutive_matches INTEGER NOT NULL,
            consecutive_misses INTEGER NOT NULL,
            first_observed_at TEXT NOT NULL,
            last_observed_at TEXT NOT NULL,
            status TEXT NOT NULL,
            superseded_by INTEGER,
            UNIQUE(weekday, time_window, content_key, stream_format, source)
        );
        CREATE INDEX IF NOT EXISTS idx_schedule_observation_pattern
            ON schedule_observations(weekday, time_window, content_key, stream_format, observed_at);
        CREATE INDEX IF NOT EXISTS idx_schedule_hypothesis_current
            ON schedule_hypotheses(weekday, status, confidence);
        """
    )
    now = _now_iso()
    count = cur.execute("SELECT COUNT(*) AS c FROM stream_schedule").fetchone()["c"]
    if int(count or 0) == 0:
        for item in DEFAULT_SCHEDULE:
            cur.execute(
                """
                INSERT INTO stream_schedule (
                    weekday, slot_name, game, category, playthrough_type,
                    tags_json, title_style, enabled, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["weekday"],
                    item["slot_name"],
                    item["game"],
                    item["category"],
                    item["playthrough_type"],
                    _json(item["tags"]),
                    item["title_style"],
                    1 if item["enabled"] else 0,
                    now,
                ),
            )
    # Legacy schedule rows become temporally revisable owner hypotheses.  The
    # row remains for compatibility and history; it is no longer eternal truth.
    rows = cur.execute("SELECT * FROM stream_schedule WHERE enabled = 1").fetchall()
    for row in rows:
        content = str(row["game"] or "").strip()
        if not content:
            continue
        cur.execute(
            """
            INSERT OR IGNORE INTO schedule_hypotheses(
                weekday, time_window, canonical_content, content_key, stream_format,
                source, confidence, evidence_count, consecutive_matches,
                consecutive_misses, first_observed_at, last_observed_at, status, superseded_by
            ) VALUES (?, 'any', ?, ?, ?, 'owner_declared', 0.90, 1, 1, 0, ?, ?, 'active', NULL)
            """,
            (
                row["weekday"], content, normalize_game_key(content),
                infer_stream_format(content, str(row["playthrough_type"] or "")),
                row["updated_at"], row["updated_at"],
            ),
        )
    conn.commit()
    conn.close()


def local_now(timezone_name: str = DEFAULT_TIMEZONE) -> datetime:
    return datetime.now(ZoneInfo(timezone_name))


def weekday_key_for(dt: datetime) -> str:
    return WEEKDAY_KEYS[int(dt.weekday())]


def schedule_time_window(dt: datetime) -> str:
    hour = int(dt.hour)
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 24:
        return "evening"
    return "night"


def infer_stream_format(content: str, playthrough_type: str = "") -> str:
    value = normalize_game_key(f"{content} {playthrough_type}")
    if "retro" in value:
        return "retro"
    if any(marker in value for marker in ("challenge", "desafio", "level 1", "lv 1", "no exp")):
        return "challenge_run"
    if any(marker in value for marker in ("charlando", "just chatting", "podcast", "talk")):
        return "talk_show"
    return "game_playthrough"


def _hypothesis_row_to_schedule(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    content = str(row["canonical_content"] or "").strip()
    stream_format = str(row["stream_format"] or "game_playthrough")
    confidence = float(row["confidence"] or 0.0)
    status = str(row["status"] or "tentative")
    return {
        "id": int(row["id"]),
        "weekday": row["weekday"],
        "slot_name": content,
        "game": content,
        "category": content,
        "playthrough_type": infer_playthrough_type(content),
        "tags": [],
        "title_style": "leo_standard",
        "enabled": 1,
        "updated_at": row["last_observed_at"],
        "time_window": row["time_window"],
        "stream_format": stream_format,
        "source": row["source"],
        "confidence": confidence,
        "evidence_count": int(row["evidence_count"] or 0),
        "hypothesis_status": status,
        "schedule_uncertain": status != "active" or confidence < 0.8,
        "superseded_by": row["superseded_by"],
    }


def record_schedule_observation(
    *, stream_session_id: str | int, canonical_content: str, dt: datetime | None = None,
    timezone_name: str = DEFAULT_TIMEZONE, stream_format: str = "", source: str = "observed",
) -> dict:
    """Record one canonical live session and update revisable weekday hypotheses."""
    init_session_primer_schema()
    dt = dt or local_now(timezone_name)
    session_key = str(stream_session_id or "").strip()
    content = str(canonical_content or "").strip()
    if not session_key or not content:
        return {"recorded": False, "reason": "missing_session_or_content"}
    weekday = weekday_key_for(dt)
    window = schedule_time_window(dt)
    content_key = normalize_game_key(content)
    format_key = str(stream_format or infer_stream_format(content)).strip() or "game_playthrough"
    observed_at = dt.isoformat()
    conn = db_sqlite.get_db_connection()
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO schedule_observations(
            stream_session_id, weekday, time_window, canonical_content,
            content_key, stream_format, source, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_key, weekday, window, content, content_key, format_key, source, observed_at),
    )
    if not cur.rowcount:
        conn.close()
        return {"recorded": False, "reason": "duplicate_session"}

    observations = conn.execute(
        """
        SELECT * FROM schedule_observations
        WHERE weekday = ? AND time_window = ?
        ORDER BY observed_at DESC, id DESC
        """,
        (weekday, window),
    ).fetchall()
    matching = [row for row in observations if row["content_key"] == content_key and row["stream_format"] == format_key]
    consecutive_matches = 0
    for row in observations:
        if row["content_key"] == content_key and row["stream_format"] == format_key:
            consecutive_matches += 1
        else:
            break
    hypothesis_id = None
    if len(matching) >= 2:
        status = "active" if len(matching) >= 3 else "tentative"
        confidence = min(0.94, 0.62 + 0.10 * len(matching))
        conn.execute(
            """
            INSERT INTO schedule_hypotheses(
                weekday, time_window, canonical_content, content_key, stream_format,
                source, confidence, evidence_count, consecutive_matches,
                consecutive_misses, first_observed_at, last_observed_at, status, superseded_by
            ) VALUES (?, ?, ?, ?, ?, 'observed', ?, ?, ?, 0, ?, ?, ?, NULL)
            ON CONFLICT(weekday, time_window, content_key, stream_format, source) DO UPDATE SET
                canonical_content=excluded.canonical_content,
                confidence=excluded.confidence,
                evidence_count=excluded.evidence_count,
                consecutive_matches=excluded.consecutive_matches,
                consecutive_misses=0,
                last_observed_at=excluded.last_observed_at,
                status=excluded.status,
                superseded_by=NULL
            """,
            (
                weekday, window, content, content_key, format_key, confidence,
                len(matching), consecutive_matches, matching[-1]["observed_at"],
                observed_at, status,
            ),
        )
        row = conn.execute(
            """SELECT id FROM schedule_hypotheses WHERE weekday=? AND time_window=?
               AND content_key=? AND stream_format=? AND source='observed'""",
            (weekday, window, content_key, format_key),
        ).fetchone()
        hypothesis_id = int(row["id"]) if row else None

    competing = conn.execute(
        """
        SELECT * FROM schedule_hypotheses
        WHERE weekday = ? AND time_window IN (?, 'any')
          AND content_key != ? AND status IN ('active', 'tentative', 'weakening')
        """,
        (weekday, window, content_key),
    ).fetchall()
    for row in competing:
        misses = int(row["consecutive_misses"] or 0) + 1
        if misses >= 2:
            new_status = "superseded"
            superseded_by = hypothesis_id
        else:
            new_status = "weakening"
            superseded_by = None
        conn.execute(
            """UPDATE schedule_hypotheses SET consecutive_matches=0, consecutive_misses=?,
               confidence=?, status=?, superseded_by=?, last_observed_at=? WHERE id=?""",
            (
                misses, max(0.2, float(row["confidence"] or 0.0) - 0.14),
                new_status, superseded_by, observed_at, row["id"],
            ),
        )
    conn.commit()
    current = conn.execute(
        """SELECT * FROM schedule_hypotheses WHERE id = ?""", (hypothesis_id,),
    ).fetchone() if hypothesis_id else None
    conn.close()
    print(
        f"[HEBE][SCHEDULE_OBSERVATION] session={session_key} weekday={weekday} window={window} "
        f"content={content!r} format={format_key} matches={len(matching)}",
        flush=True,
    )
    return {
        "recorded": True,
        "observation": asdict(ScheduleObservation(
            session_key, weekday, window, content, format_key, source, observed_at,
        )),
        "hypothesis": asdict(ScheduleHypothesis(
            weekday=current["weekday"], time_window=current["time_window"],
            canonical_content=current["canonical_content"], stream_format=current["stream_format"],
            source=current["source"], confidence=float(current["confidence"]),
            evidence_count=int(current["evidence_count"]), consecutive_matches=int(current["consecutive_matches"]),
            consecutive_misses=int(current["consecutive_misses"]), first_observed_at=current["first_observed_at"],
            last_observed_at=current["last_observed_at"], status=current["status"],
            superseded_by=current["superseded_by"],
        )) if current else None,
    }


def get_schedule_for_date(dt: datetime | None = None, *, timezone_name: str = DEFAULT_TIMEZONE) -> dict | None:
    init_session_primer_schema()
    dt = dt or local_now(timezone_name)
    weekday = weekday_key_for(dt)
    conn = db_sqlite.get_db_connection()
    window = schedule_time_window(dt)
    hypothesis = conn.execute(
        """
        SELECT * FROM schedule_hypotheses
        WHERE weekday = ? AND time_window IN (?, 'any')
          AND status IN ('active', 'tentative', 'weakening')
        ORDER BY confidence DESC,
          CASE source WHEN 'owner_declared' THEN 0 WHEN 'observed' THEN 1 ELSE 2 END,
          last_observed_at DESC
        LIMIT 1
        """,
        (weekday, window),
    ).fetchone()
    hypothesis_schedule = _hypothesis_row_to_schedule(hypothesis)
    if hypothesis_schedule is not None:
        conn.close()
        return hypothesis_schedule
    row = conn.execute(
        """
        SELECT * FROM stream_schedule
        WHERE weekday = ? AND enabled = 1
        ORDER BY id ASC
        LIMIT 1
        """,
        (weekday,),
    ).fetchone()
    conn.close()
    item = _row(row)
    if item:
        item["tags"] = _loads(item.pop("tags_json", ""), [])
    return item


def update_schedule_for_weekday(weekday: str, game: str, *, category: str | None = None, playthrough_type: str | None = None) -> dict:
    init_session_primer_schema()
    weekday = str(weekday or "").strip().lower()
    if weekday not in WEEKDAY_KEYS:
        raise ValueError(f"Unsupported weekday: {weekday}")
    game = str(game or "").strip()
    if not game:
        raise ValueError("Game is required")
    now = _now_iso()
    category = category or game
    playthrough_type = playthrough_type or infer_playthrough_type(game)
    conn = db_sqlite.get_db_connection()
    conn.execute(
        """
        INSERT INTO stream_schedule (
            weekday, slot_name, game, category, playthrough_type,
            tags_json, title_style, enabled, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
        ON CONFLICT(weekday, slot_name) DO UPDATE SET
            game = excluded.game,
            category = excluded.category,
            playthrough_type = excluded.playthrough_type,
            tags_json = excluded.tags_json,
            title_style = excluded.title_style,
            enabled = 1,
            updated_at = excluded.updated_at
        """,
        (weekday, "Manual Override", game, category, playthrough_type, _json(["ENG/ESP"]), "leo_standard", now),
    )
    conn.commit()
    content_key = normalize_game_key(game)
    stream_format = infer_stream_format(game, playthrough_type)
    conn.execute(
        """UPDATE schedule_hypotheses SET status='historical', superseded_by=NULL
           WHERE weekday=? AND status IN ('active','tentative','weakening')""",
        (weekday,),
    )
    conn.execute(
        """
        INSERT INTO schedule_hypotheses(
            weekday,time_window,canonical_content,content_key,stream_format,source,
            confidence,evidence_count,consecutive_matches,consecutive_misses,
            first_observed_at,last_observed_at,status,superseded_by
        ) VALUES (?, 'any', ?, ?, ?, 'owner_declared', 0.98, 1, 1, 0, ?, ?, 'active', NULL)
        ON CONFLICT(weekday,time_window,content_key,stream_format,source) DO UPDATE SET
          canonical_content=excluded.canonical_content, confidence=0.98,
          consecutive_matches=1, consecutive_misses=0, last_observed_at=excluded.last_observed_at,
          status='active', superseded_by=NULL
        """,
        (weekday, game, content_key, stream_format, now, now),
    )
    conn.commit()
    row = conn.execute(
        """
        SELECT * FROM stream_schedule
        WHERE weekday = ? AND enabled = 1
        ORDER BY id DESC
        LIMIT 1
        """,
        (weekday,),
    ).fetchone()
    conn.close()
    item = _row(row) or {}
    if item:
        item["tags"] = _loads(item.pop("tags_json", ""), [])
    return item


def infer_playthrough_type(game: str) -> str:
    value = normalize_game_key(game)
    if "persona 5" in value:
        return "First Playthrough"
    if "final fantasy ix" in value or "ff9" in value:
        return "Lv. 1 Challenge"
    if "baldur" in value:
        return "Chat Playthrough"
    if "retro" in value:
        return "Retro Weekend"
    return "Playthrough"


def latest_game_session(game: str) -> dict | None:
    init_session_primer_schema()
    key = normalize_game_key(game)
    conn = db_sqlite.get_db_connection()
    row = conn.execute(
        """
        SELECT * FROM game_sessions
        WHERE game_key = ?
        ORDER BY COALESCE(stream_date, ended_at, updated_at) DESC, id DESC
        LIMIT 1
        """,
        (key,),
    ).fetchone()
    conn.close()
    item = _row(row)
    if item:
        item["important_events"] = _loads(item.pop("important_events_json", ""), [])
        item["unresolved_threads"] = _loads(item.pop("unresolved_threads_json", ""), [])
    return item


def save_game_session_note(
    game: str,
    *,
    stream_date: str | None = None,
    start_summary: str = "",
    end_summary: str = "",
    current_location: str = "",
    current_objective: str = "",
    next_time_plan: str = "",
    spoiler_policy: str = "no_spoilers",
    source: str = "manual",
) -> dict:
    init_session_primer_schema()
    now = _now_iso()
    game = str(game or "").strip()
    key = normalize_game_key(game)
    conn = db_sqlite.get_db_connection()
    cur = conn.execute(
        """
        INSERT INTO game_sessions (
            game, game_key, stream_date, title, started_at, ended_at,
            start_summary, end_summary, current_location, current_objective,
            important_events_json, unresolved_threads_json, next_time_plan,
            spoiler_policy, source, created_at, updated_at
        )
        VALUES (?, ?, ?, '', '', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            game,
            key,
            stream_date or now[:10],
            now,
            start_summary,
            end_summary,
            current_location,
            current_objective,
            _json([]),
            _json([]),
            next_time_plan,
            spoiler_policy,
            source,
            now,
            now,
        ),
    )
    conn.commit()
    session_id = int(cur.lastrowid)
    row = conn.execute("SELECT * FROM game_sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    item = _row(row) or {}
    item["important_events"] = _loads(item.pop("important_events_json", ""), [])
    item["unresolved_threads"] = _loads(item.pop("unresolved_threads_json", ""), [])
    return item


def invalidate_game_session_term(game: str, term: str, *, source: str = "manual_correction") -> int:
    init_session_primer_schema()
    key = normalize_game_key(game)
    needle = normalize_game_key(term)
    if not key or not needle:
        return 0
    conn = db_sqlite.get_db_connection()
    rows = conn.execute(
        """
        SELECT * FROM game_sessions
        WHERE game_key = ?
        """,
        (key,),
    ).fetchall()
    changed = 0
    now = _now_iso()
    text_fields = ("start_summary", "end_summary", "current_location", "current_objective", "next_time_plan")
    for row in rows:
        updates: dict[str, str] = {}
        for field in text_fields:
            value = str(row[field] or "")
            if needle in normalize_game_key(value):
                updates[field] = _remove_term_phrase(value, term)
        important = _loads(row["important_events_json"], [])
        unresolved = _loads(row["unresolved_threads_json"], [])
        new_important = [item for item in important if needle not in normalize_game_key(item)]
        new_unresolved = [item for item in unresolved if needle not in normalize_game_key(item)]
        if len(new_important) != len(important):
            updates["important_events_json"] = _json(new_important)
        if len(new_unresolved) != len(unresolved):
            updates["unresolved_threads_json"] = _json(new_unresolved)
        if not updates:
            continue
        assignments = ", ".join(f"{field} = ?" for field in updates) + ", source = ?, updated_at = ?"
        conn.execute(
            f"UPDATE game_sessions SET {assignments} WHERE id = ?",
            (*updates.values(), source, now, row["id"]),
        )
        changed += 1
    conn.commit()
    conn.close()
    if changed:
        print(
            f"[HEBE][MEMORY_EXTRACT] invalidated game={game!r} term={term!r} rows={changed}",
            flush=True,
        )
    return changed


def _remove_term_phrase(text: str, term: str) -> str:
    cleaned = re.sub(rf"\b{re.escape(str(term or '').strip())}\b", "", str(text or ""), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip(" ,.;:")


def build_stream_session_primer(
    *,
    game: str | None = None,
    dt: datetime | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
    twitch_context: dict | None = None,
) -> StreamSessionPrimer:
    init_session_primer_schema()
    dt = dt or local_now(timezone_name)
    weekday = weekday_key_for(dt)
    schedule = get_schedule_for_date(dt, timezone_name=timezone_name) or {}
    scheduled_game = str(game or schedule.get("game") or "").strip()
    if not scheduled_game:
        scheduled_game = "Unknown game"
    playthrough_type = str(schedule.get("playthrough_type") or infer_playthrough_type(scheduled_game))
    category = str((twitch_context or {}).get("category") or schedule.get("category") or scheduled_game)
    last = latest_game_session(scheduled_game)
    missing: list[str] = []
    if not schedule:
        missing.append("schedule")
    if not last:
        missing.append("previous_session")
    last_summary = _session_summary(last) if last else ""
    starting_point = (last or {}).get("next_time_plan") or (last or {}).get("current_location") or ""
    likely_objective = (last or {}).get("next_time_plan") or (last or {}).get("current_objective") or ""
    spoiler_policy = (last or {}).get("spoiler_policy") or "no_spoilers"
    safe_context = _safe_context(last, scheduled_game, playthrough_type, spoiler_policy)
    titles = generate_title_suggestions(
        scheduled_game,
        playthrough_type=playthrough_type,
        last_session=last,
        count=5,
    )
    print(
        f"[HEBE][SESSION_PRIMER] today={dt.date().isoformat()} weekday={weekday} scheduled_game={scheduled_game!r}",
        flush=True,
    )
    print(f"[HEBE][SESSION_PRIMER] last_session_found={str(bool(last)).lower()}", flush=True)
    print(f"[HEBE][SESSION_PRIMER] title_suggestions={titles!r}", flush=True)
    return StreamSessionPrimer(
        local_now=dt.isoformat(),
        weekday=weekday,
        timezone=timezone_name,
        slot_name=str(schedule.get("slot_name") or ""),
        game=scheduled_game,
        category=category,
        playthrough_type=playthrough_type,
        challenge_type=playthrough_type if "challenge" in playthrough_type.casefold() else "",
        last_session_summary=last_summary,
        starting_point=starting_point,
        likely_objective=likely_objective,
        spoiler_policy=spoiler_policy,
        safe_context_for_spontaneity=safe_context,
        title_suggestions=titles,
        missing_info=missing,
        schedule=schedule,
        last_session=last,
    )


def _session_summary(session: dict | None) -> str:
    if not session:
        return ""
    return (
        session.get("end_summary")
        or session.get("start_summary")
        or session.get("next_time_plan")
        or session.get("current_objective")
        or session.get("current_location")
        or ""
    )


def _safe_context(session: dict | None, game: str, playthrough_type: str, spoiler_policy: str) -> list[str]:
    items = [f"{game} - {playthrough_type}", f"spoiler_policy={spoiler_policy}"]
    if session:
        for key in ("end_summary", "current_location", "current_objective", "next_time_plan"):
            value = str(session.get(key) or "").strip()
            if value:
                items.append(value)
    return items[:8]


def generate_title_suggestions(
    game: str,
    *,
    playthrough_type: str = "Playthrough",
    last_session: dict | None = None,
    count: int = 5,
) -> list[str]:
    game = str(game or "Unknown Game").strip()
    playthrough_type = str(playthrough_type or infer_playthrough_type(game)).strip()
    context = " ".join(
        str((last_session or {}).get(key) or "")
        for key in ("next_time_plan", "end_summary", "current_objective", "current_location")
    ).casefold()
    hooks: list[str]
    if "museum" in context or "museo" in context:
        hooks = [
            "Museum case closed! What comes next?",
            "The Museum case is closing, what comes next?",
            "Back after the Museum case",
            "Picking up after the Museum",
            "What comes after the Museum?",
        ]
    elif "retro" in playthrough_type.casefold() or "retro" in game.casefold():
        hooks = [
            "Retro Weekend is back!",
            "Retro Weekend continues",
            "Old-school chaos, fresh coffee",
            "Retro Weekend roulette",
            "Back to the classics",
        ]
    elif "challenge" in playthrough_type.casefold():
        hooks = [
            "Challenge run continues",
            "One more careful step",
            "Low level, high stress",
            "Back to the challenge",
            "No levels, no mercy",
        ]
    elif "persona 5" in normalize_game_key(game):
        hooks = [
            "Back with the Phantom Thieves!",
            "One more night with the Phantom Thieves",
            "Palaces, plans, and questionable confidence",
            "The next move begins",
            "No spoilers, just trouble",
        ]
    else:
        hooks = [
            "Back to the run",
            "Picking up where we left off",
            "One more session, one more bad idea",
            "The next chapter begins",
            "No spoilers, just vibes",
        ]
    titles = [f"[ENG/ESP] {hook} — {game} | {playthrough_type}" for hook in hooks]
    print("[HEBE][TITLE] generated format=leo_standard", flush=True)
    return titles[: max(1, int(count or 1))]


def apply_primer_to_stream(stream: Any, primer: StreamSessionPrimer) -> None:
    if stream is None:
        return
    data = primer.to_dict()
    stream.current_game = primer.game
    stream.current_category = primer.category
    stream.current_playthrough_type = primer.playthrough_type
    stream.current_challenge = primer.challenge_type
    stream.current_stream_slot = primer.slot_name
    stream.spoiler_policy = primer.spoiler_policy
    stream.current_run_objective = primer.likely_objective or getattr(stream, "current_run_objective", None)
    stream.current_run_location = primer.starting_point or getattr(stream, "current_run_location", None)
    stream.session_primer = data
    stream.stream_context_updated_ts = datetime.now().timestamp()
    stream.run_context_updated_ts = datetime.now().timestamp()
    stream.run_context_source = "session_primer"
    facts = list(getattr(stream, "recent_run_context_facts", []) or [])
    for text in primer.safe_context_for_spontaneity[:5]:
        facts.append({
            "id": f"session_primer:{len(facts) + 1}",
            "kind": "session_primer",
            "category": "session_primer",
            "text": text,
            "summary": text,
            "confidence": 0.86,
            "source": "session_primer",
        })
    stream.recent_run_context_facts = facts[-20:]
    print(f"[HEBE][STREAM_CONTEXT] primer_loaded=true game={primer.game!r}", flush=True)
