import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "hebe.db"  # ajusta a tu ruta real

def column_exists(conn, table: str, col: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == col for row in cur.fetchall())

with sqlite3.connect(DB_PATH) as conn:
    conn.row_factory = sqlite3.Row

    if not column_exists(conn, "app_commands", "process_name"):
        conn.execute("ALTER TABLE app_commands ADD COLUMN process_name TEXT")
        print("✅ added process_name")

    if not column_exists(conn, "app_commands", "window_title"):
        conn.execute("ALTER TABLE app_commands ADD COLUMN window_title TEXT")
        print("✅ added window_title")

    conn.commit()

print("Done.")

