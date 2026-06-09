# backend/app/services/db_sqlite.py
from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Any


def _default_db_path() -> str:
    # Mantiene compat con tu código actual: DB en cwd/hebe.db
    return os.path.join(os.getcwd(), "hebe.db")


DB_PATH = os.getenv("HEBE_DB_PATH", _default_db_path())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_iso(dt_value: Optional[str]) -> Optional[str]:
    if not dt_value:
        return None

    text = dt_value.strip()
    if not text:
        return None

    # Soporta "Z"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc).isoformat()


def dumps_json(value: Optional[dict]) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def loads_json(value: Optional[str]) -> Optional[dict]:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def row_to_dict(row: sqlite3.Row | None) -> Optional[dict]:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [row_to_dict(r) for r in rows if r is not None]


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(conn: sqlite3.Connection, table: str, column: str, col_def: str) -> None:
    """Add a column to a table if it doesn't exist (safe migration)."""
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        rows = cur.fetchall()
        existing = set()
        for r in rows:
            try:
                existing.add(r[1])       # tuple
            except Exception:
                existing.add(r["name"])  # Row
        if column not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
    except Exception as e:
        print(f"⚠️ No se pudo asegurar la columna {table}.{column}: {e}")


def init_db() -> None:
    print(f"[HEBE][DB] DB_PATH={DB_PATH}", flush=True)
    conn = get_db_connection()
    cur = conn.cursor()

    # Conversaciones
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            role TEXT NOT NULL,      -- 'user' | 'assistant' | 'system'
            source TEXT NOT NULL,    -- 'voice' | 'tts' | 'llm' | 'wiki' | 'ui'
            text TEXT NOT NULL
        )
        """
    )

    # Apps que Hebe puede abrir
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            command TEXT NOT NULL,
            description TEXT,
            aliases TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            usage_count INTEGER NOT NULL DEFAULT 0,
            last_used_at TEXT,
            process_name TEXT,
            window_title TEXT
        );
        """
    )

    # Memorias legacy
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT,
            text TEXT NOT NULL,
            category TEXT,
            importance INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            active INTEGER NOT NULL DEFAULT 1
        )
        """
    )

    # Memoria estructurada v1
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,                -- appointment | preference | person | task | fact
            subject TEXT,
            payload_json TEXT,
            source_text TEXT,
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_used_at TEXT,
            active INTEGER NOT NULL DEFAULT 1
        )
        """
    )

    # Recordatorios / scheduler v1
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL DEFAULT 'generic',     -- appointment | task | generic
            title TEXT NOT NULL,
            message TEXT,
            due_at TEXT NOT NULL,
            timezone TEXT DEFAULT 'UTC',
            status TEXT NOT NULL DEFAULT 'pending',   -- pending | fired | done | cancelled
            source_memory_id INTEGER,
            payload_json TEXT,
            created_at TEXT NOT NULL,
            fired_at TEXT,
            FOREIGN KEY(source_memory_id) REFERENCES memory_facts(id)
        )
        """
    )

    # Log opcional de eventos internos
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS internal_events_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            payload_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )

    # Índices útiles
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_facts_kind_active ON memory_facts(kind, active)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_facts_subject ON memory_facts(subject)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_reminders_status_due_at ON reminders(status, due_at)"
    )

    # Safe migrations (older DBs might miss columns)
    ensure_column(conn, "app_commands", "process_name", "TEXT")
    ensure_column(conn, "app_commands", "window_title", "TEXT")

    conn.commit()
    conn.close()
    cleanup_stt_prompt_injection_rows()

    try:
        from app.stream.memory import init_stream_memory_schema

        init_stream_memory_schema()
    except Exception as exc:
        print(f"[HEBE][STREAM_MEMORY] init failed: {exc!r}", flush=True)


def _table_exists(cur: sqlite3.Cursor, table: str) -> bool:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table,))
    return cur.fetchone() is not None


def cleanup_stt_prompt_injection_rows() -> dict[str, int]:
    """Remove already-persisted STT hotword prompt loops from chat/memory stores."""
    conn = get_db_connection()
    cur = conn.cursor()
    deleted: dict[str, int] = {}
    predicates = """
        lower({column}) LIKE '%obs%stream%chat%promo%shoutout%'
        OR lower({column}) LIKE 'hebe, ebe,%obs%twitch%promo%shoutout%'
        OR lower({column}) LIKE '%obs, twitch, stream, chat, promo, shoutout%'
    """

    targets = [
        ("chat_log", "text", "DELETE FROM chat_log WHERE role = 'user' AND (" + predicates.format(column="text") + ")"),
        ("memories", "text", "DELETE FROM memories WHERE " + predicates.format(column="text")),
        ("memory_facts", "source_text", "UPDATE memory_facts SET active = 0 WHERE " + predicates.format(column="source_text")),
        ("memory_chunks", "text", "UPDATE memory_chunks SET active = 0 WHERE " + predicates.format(column="text")),
    ]
    for table, _column, sql in targets:
        if not _table_exists(cur, table):
            continue
        try:
            cur.execute(sql)
            deleted[table] = int(cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0)
        except Exception as exc:
            print(f"[HEBE][STT][CLEANUP] table={table} failed={exc!r}", flush=True)
    conn.commit()
    conn.close()
    total = sum(deleted.values())
    if total:
        print(f"[HEBE][STT][CLEANUP] removed_prompt_injection_rows={deleted}", flush=True)
    return deleted


def get_recent_chat_turns(
    source: str = "ui",
    limit: int = 10,
) -> list[dict]:
    """
    Devuelve los últimos `limit` turnos de chat_log con source=`source`,
    ordenados cronológicamente ASCENDENTE (los más antiguos primero), listos
    para usar como historial conversacional en messages[] de OpenAI.

    Cada turno: {"role": "user"|"assistant", "content": str}
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, text
        FROM chat_log
        WHERE source = ?
          AND role IN ('user', 'assistant')
        ORDER BY id DESC
        LIMIT ?
        """,
        (source, limit),
    )
    rows = cur.fetchall()
    conn.close()
    # Reverse: la query saca DESC para limitar a los N más recientes, pero
    # el resultado va ASC (antiguo → reciente) para usar como historial.
    return [
        {"role": r["role"], "content": r["text"]}
        for r in reversed(rows)
    ]


def log_chat(role: str, text: str, source: str = "voice") -> None:
    """Guarda una línea de conversación en la BD."""
    if not text:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_log (timestamp, role, source, text)
        VALUES (?, ?, ?, ?)
        """,
        (utc_now_iso(), role, source, text),
    )
    conn.commit()
    conn.close()


def log_internal_event(event_type: str, payload: Optional[dict] = None) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO internal_events_log (event_type, payload_json, created_at)
        VALUES (?, ?, ?)
        """,
        (event_type, dumps_json(payload), utc_now_iso()),
    )
    event_id = cur.lastrowid
    conn.commit()
    conn.close()
    return event_id


def ensure_app_settings_table() -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    ensure_app_settings_table()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return default
    return row["value"]


def set_setting(key: str, value: Optional[str]) -> None:
    ensure_app_settings_table()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO app_settings (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        (key, value, utc_now_iso()),
    )
    conn.commit()
    conn.close()


def seed_default_apps() -> None:
    """Rellena app_commands con algunas apps por defecto si no existen."""
    defaults = [
        ("chrome", "start chrome", "Navegador Chrome", "navegador,google"),
        ("discord", "start discord", "Discord", "dc"),
        ("obs", r"C:\Program Files\obs-studio\bin\64bit\obs64.exe", "OBS Studio", ""),
        ("explorador", "explorer", "Explorador de archivos", "explorer,archivos"),
        ("notas", "notepad", "Bloc de notas", "bloc,notepad"),
        ("calculadora", "calc", "Calculadora", "calc,calculadora"),
        (
            "final",
            r"C:\Program Files (x86)\SquareEnix\FINAL FANTASY XIV - A Realm Reborn\boot\ffxivboot.exe",
            "Final Fantasy XIV",
            "ff14,ffxiv",
        ),
    ]

    conn = get_db_connection()
    cur = conn.cursor()
    for name, cmd, desc, aliases in defaults:
        cur.execute(
            """
            INSERT OR IGNORE INTO app_commands (name, command, description, aliases)
            VALUES (?, ?, ?, ?)
            """,
            (name, cmd, desc, aliases),
        )
    conn.commit()
    conn.close()


def find_app_for_command(command_text: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM app_commands WHERE enabled = 1")
    apps = cur.fetchall()
    conn.close()

    text = (command_text or "").lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))

    best = None
    best_score = -1.0

    for app in apps:
        names = [app["name"]]
        if app["aliases"]:
            names += [a.strip() for a in app["aliases"].split(",") if a.strip()]

        for alias in names:
            a = (alias or "").strip().lower()
            if not a:
                continue

            a_tokens = set(re.findall(r"[a-z0-9]+", a))
            if not a_tokens:
                continue

            hit = len(a_tokens & tokens)
            if hit == 0:
                continue

            bonus = 0.0
            if re.search(rf"\b{re.escape(a)}\b", text):
                bonus += 3.0

            bonus += min(len(a), 30) / 10.0

            usage = int(app["usage_count"] or 0)
            bonus += min(usage, 50) / 25.0

            score = hit * 5 + bonus
            if score > best_score:
                best_score = score
                best = app

    return best


def register_app_usage(app_id: int) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE app_commands
        SET usage_count = usage_count + 1,
            last_used_at = ?
        WHERE id = ?
        """,
        (utc_now_iso(), app_id),
    )
    conn.commit()
    conn.close()


# =========================
# Legacy memories
# =========================

def add_memory(
    text: str,
    key: Optional[str] = None,
    category: Optional[str] = None,
    importance: int = 1,
) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO memories (key, text, category, importance, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (key, text, category, importance, utc_now_iso()),
    )
    conn.commit()
    conn.close()


def get_active_memories(limit: int = 5):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, key, text, category, importance, created_at, last_used_at
        FROM memories
        WHERE active = 1
        ORDER BY importance DESC, created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


# =========================
# Structured memory v1
# =========================

def insert_memory_fact(
    kind: str,
    subject: Optional[str] = None,
    payload: Optional[dict] = None,
    source_text: Optional[str] = None,
    confidence: float = 1.0,
    active: bool = True,
) -> int:
    now = utc_now_iso()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO memory_facts (
            kind, subject, payload_json, source_text, confidence,
            created_at, updated_at, active
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            kind,
            subject,
            dumps_json(payload),
            source_text,
            float(confidence),
            now,
            now,
            1 if active else 0,
        ),
    )
    memory_id = cur.lastrowid
    conn.commit()
    conn.close()
    return memory_id


def get_memory_fact(memory_id: int) -> Optional[dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM memory_facts
        WHERE id = ?
        """,
        (memory_id,),
    )
    row = cur.fetchone()
    conn.close()

    data = row_to_dict(row)
    if data and data.get("payload_json"):
        data["payload"] = loads_json(data["payload_json"])
    return data


def search_memory_facts(
    query_text: Optional[str] = None,
    kind: Optional[str] = None,
    active_only: bool = True,
    limit: int = 10,
) -> list[dict]:
    conn = get_db_connection()
    cur = conn.cursor()

    where = []
    params: list[Any] = []

    if active_only:
        where.append("active = 1")

    if kind:
        where.append("kind = ?")
        params.append(kind)

    if query_text:
        where.append("(subject LIKE ? OR source_text LIKE ? OR payload_json LIKE ?)")
        like = f"%{query_text}%"
        params.extend([like, like, like])

    sql = """
        SELECT *
        FROM memory_facts
    """
    if where:
        sql += " WHERE " + " AND ".join(where)

    sql += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
    params.append(limit)

    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    items = rows_to_dicts(rows)
    for item in items:
        item["payload"] = loads_json(item.get("payload_json"))
    return items


def find_memory_fact(
    *,
    kind: str,
    subject: Optional[str],
    active_only: bool = True,
) -> Optional[dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    where = ["kind = ?"]
    params: list[Any] = [kind]

    if subject is None:
        where.append("subject IS NULL")
    else:
        where.append("subject = ?")
        params.append(subject)

    if active_only:
        where.append("active = 1")

    cur.execute(
        "SELECT * FROM memory_facts WHERE "
        + " AND ".join(where)
        + " ORDER BY updated_at DESC LIMIT 1",
        params,
    )
    row = cur.fetchone()
    conn.close()

    item = row_to_dict(row)
    if item:
        item["payload"] = loads_json(item.get("payload_json"))
    return item


def upsert_memory_fact(
    kind: str,
    subject: Optional[str] = None,
    payload: Optional[dict] = None,
    source_text: Optional[str] = None,
    confidence: float = 1.0,
    active: bool = True,
) -> tuple[int, bool]:
    """
    Insert or update the active fact for the same kind + subject.

    Returns (memory_id, created). This intentionally replaces the active fact's
    readable text/payload so corrections from Leo win over older wording.
    """
    existing = find_memory_fact(kind=kind, subject=subject, active_only=True)
    now = utc_now_iso()

    if existing:
        memory_id = int(existing["id"])
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE memory_facts
            SET payload_json = ?,
                source_text = ?,
                confidence = ?,
                updated_at = ?,
                active = ?
            WHERE id = ?
            """,
            (
                dumps_json(payload),
                source_text,
                float(confidence),
                now,
                1 if active else 0,
                memory_id,
            ),
        )
        conn.commit()
        conn.close()
        return memory_id, False

    return (
        insert_memory_fact(
            kind=kind,
            subject=subject,
            payload=payload,
            source_text=source_text,
            confidence=confidence,
            active=active,
        ),
        True,
    )


def count_memory_facts(active_only: bool = True) -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    sql = "SELECT COUNT(*) AS c FROM memory_facts"
    params: list[Any] = []
    if active_only:
        sql += " WHERE active = ?"
        params.append(1)
    cur.execute(sql, params)
    n = cur.fetchone()["c"]
    conn.close()
    return int(n)


def get_recent_memory_facts(limit: int = 10, active_only: bool = True) -> list[dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    sql = "SELECT * FROM memory_facts"
    params: list[Any] = []
    if active_only:
        sql += " WHERE active = ?"
        params.append(1)
    sql += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
    params.append(limit)
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    items = rows_to_dicts(rows)
    for item in items:
        item["payload"] = loads_json(item.get("payload_json"))
    return items


def get_recent_chat_log(source: Optional[str] = None, limit: int = 10) -> list[dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    params: list[Any] = []
    sql = "SELECT id, timestamp, role, source, text FROM chat_log"
    if source:
        sql += " WHERE source = ?"
        params.append(source)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return rows_to_dicts(rows)


def touch_memory_fact(memory_id: int) -> None:
    now = utc_now_iso()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE memory_facts
        SET last_used_at = ?, updated_at = ?
        WHERE id = ?
        """,
        (now, now, memory_id),
    )
    conn.commit()
    conn.close()


def deactivate_memory_fact(memory_id: int) -> None:
    now = utc_now_iso()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE memory_facts
        SET active = 0, updated_at = ?
        WHERE id = ?
        """,
        (now, memory_id),
    )
    conn.commit()
    conn.close()


# =========================
# Reminders v1
# =========================

def create_reminder(
    title: str,
    due_at: str,
    message: Optional[str] = None,
    kind: str = "generic",
    timezone_name: str = "UTC",
    source_memory_id: Optional[int] = None,
    payload: Optional[dict] = None,
) -> int:
    due_at_norm = normalize_iso(due_at)
    if not due_at_norm:
        raise ValueError(f"due_at inválido: {due_at}")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO reminders (
            kind, title, message, due_at, timezone, status,
            source_memory_id, payload_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
        """,
        (
            kind,
            title,
            message,
            due_at_norm,
            timezone_name,
            source_memory_id,
            dumps_json(payload),
            utc_now_iso(),
        ),
    )
    reminder_id = cur.lastrowid
    conn.commit()
    conn.close()
    return reminder_id


def get_reminder(reminder_id: int) -> Optional[dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM reminders
        WHERE id = ?
        """,
        (reminder_id,),
    )
    row = cur.fetchone()
    conn.close()

    data = row_to_dict(row)
    if data and data.get("payload_json"):
        data["payload"] = loads_json(data["payload_json"])
    return data


def list_due_reminders(now_iso: Optional[str] = None, limit: int = 20) -> list[dict]:
    now_norm = normalize_iso(now_iso) if now_iso else utc_now_iso()

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM reminders
        WHERE status = 'pending'
          AND due_at <= ?
        ORDER BY due_at ASC
        LIMIT ?
        """,
        (now_norm, limit),
    )
    rows = cur.fetchall()
    conn.close()

    items = rows_to_dicts(rows)
    for item in items:
        item["payload"] = loads_json(item.get("payload_json"))
    return items


def list_pending_reminders(limit: int = 20) -> list[dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM reminders
        WHERE status = 'pending'
        ORDER BY due_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    items = rows_to_dicts(rows)
    for item in items:
        item["payload"] = loads_json(item.get("payload_json"))
    return items


def mark_reminder_fired(reminder_id: int) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE reminders
        SET status = 'fired',
            fired_at = ?
        WHERE id = ?
        """,
        (utc_now_iso(), reminder_id),
    )
    conn.commit()
    conn.close()


def mark_reminder_done(reminder_id: int) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE reminders
        SET status = 'done'
        WHERE id = ?
        """,
        (reminder_id,),
    )
    conn.commit()
    conn.close()


def cancel_reminder(reminder_id: int) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE reminders
        SET status = 'cancelled'
        WHERE id = ?
        """,
        (reminder_id,),
    )
    conn.commit()
    conn.close()


# =========================
# Apps
# =========================

def save_app_command(name: str, command: str, description: str = "", aliases: str = ""):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO app_commands (name, command, description, aliases)
        VALUES (?, ?, ?, ?)
        """,
        (name, command, description, aliases),
    )
    conn.commit()

    cur.execute("SELECT id FROM app_commands WHERE name = ?", (name,))
    row = cur.fetchone()
    conn.close()

    return row["id"] if row else None


def update_app_process_name(app_id: int, process_name: str) -> None:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE app_commands SET process_name = ? WHERE id = ?",
            (process_name, app_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ No se pudo guardar process_name en DB: {e}")
