from __future__ import annotations
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List

# backend/app/tools/registry.py  -> parents[2] = backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BACKEND_DIR / "hebe.db"

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def find_app_for_command(command_text: str) -> Optional[Dict[str, Any]]:
    """
    Busca apps en app_commands cuyo nombre o alias aparezca en el texto.
    """
    text = (command_text or "").lower()

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM app_commands WHERE enabled = 1")
    apps = cur.fetchall()
    conn.close()

    for app in apps:
        names = [app["name"]]
        if app["aliases"]:
            names += [a.strip() for a in app["aliases"].split(",") if a.strip()]

        for alias in names:
            if alias and alias.lower() in text:
                return dict(app)

    return None

def register_app_usage(app_id: int) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE app_commands
        SET usage_count = COALESCE(usage_count, 0) + 1,
            last_used_at = ?
        WHERE id = ?
        """,
        (now, app_id),
    )
    conn.commit()
    conn.close()

def update_app_fields(
    app_id: int,
    process_name: Optional[str] = None,
    window_title: Optional[str] = None,
) -> None:
    sets = []
    params: List[Any] = []

    if process_name is not None:
        sets.append("process_name = ?")
        params.append(process_name)

    if window_title is not None:
        sets.append("window_title = ?")
        params.append(window_title)

    if not sets:
        return

    params.append(app_id)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"UPDATE app_commands SET {', '.join(sets)} WHERE id = ?", params)
    conn.commit()
    conn.close()

def get_app_by_id(app_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM app_commands WHERE id = ?", (app_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None
