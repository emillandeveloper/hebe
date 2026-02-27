# backend/app/services/db_sqlite.py
from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Any


def _default_db_path() -> str:
    # Mantiene compat con tu código actual: DB en cwd/hebe.db
    return os.path.join(os.getcwd(), "hebe.db")


DB_PATH = os.getenv("HEBE_DB_PATH", _default_db_path())


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
                existing.add(r[1])      # tuple
            except Exception:
                existing.add(r["name"])  # Row
        if column not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
    except Exception as e:
        print(f"⚠️ No se pudo asegurar la columna {table}.{column}: {e}")


def init_db() -> None:
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

    # Memorias
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

    # Safe migrations (older DBs might miss columns)
    ensure_column(conn, "app_commands", "process_name", "TEXT")
    ensure_column(conn, "app_commands", "window_title", "TEXT")

    conn.commit()
    conn.close()


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
        (datetime.utcnow().isoformat(), role, source, text),
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
        (datetime.utcnow().isoformat(), app_id),
    )
    conn.commit()
    conn.close()


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
        (key, text, category, importance, datetime.utcnow().isoformat()),
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
