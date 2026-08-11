from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable


@dataclass(frozen=True, slots=True)
class Migration:
    component: str
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]

    @property
    def checksum(self) -> str:
        identity = f"{self.component}:{self.version}:{self.name}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class MigrationRunner:
    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self.connection_factory = connection_factory

    def migrate(self, migrations: Iterable[Migration]) -> list[dict]:
        conn = self.connection_factory()
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    component TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    PRIMARY KEY(component, version)
                )
                """
            )
            conn.commit()
            applied: list[dict] = []
            ordered = sorted(migrations, key=lambda item: (item.component, item.version))
            for migration in ordered:
                row = conn.execute(
                    "SELECT checksum, name, applied_at FROM schema_migrations WHERE component=? AND version=?",
                    (migration.component, migration.version),
                ).fetchone()
                if row is not None:
                    if row["checksum"] != migration.checksum:
                        raise RuntimeError(
                            f"migration checksum mismatch: {migration.component}:{migration.version}"
                        )
                    applied.append({
                        "component": migration.component,
                        "version": migration.version,
                        "name": row["name"],
                        "checksum": row["checksum"],
                        "applied_at": row["applied_at"],
                        "already_applied": True,
                    })
                    continue
                try:
                    conn.execute("BEGIN")
                    migration.apply(conn)
                    applied_at = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        "INSERT INTO schema_migrations(component, version, name, checksum, applied_at) VALUES (?, ?, ?, ?, ?)",
                        (migration.component, migration.version, migration.name, migration.checksum, applied_at),
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                applied.append({
                    "component": migration.component,
                    "version": migration.version,
                    "name": migration.name,
                    "checksum": migration.checksum,
                    "applied_at": applied_at,
                    "already_applied": False,
                })
            return applied
        finally:
            conn.close()


def replay_foundation_migrations() -> tuple[Migration, ...]:
    def create_metadata(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cognitive_replay_metadata (
                scenario_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                last_run_at TEXT NOT NULL
            )
            """
        )

    return (Migration("cognitive_replay", 1, "replay_metadata", create_metadata),)
