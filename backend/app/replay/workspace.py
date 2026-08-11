from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from app.cognitive.memory.memory_store import init_memory_chunks_schema
from app.replay.migrations import MigrationRunner, replay_foundation_migrations
from app.services import db_sqlite
from app.stream import memory as stream_memory
from app.stream.live_session import init_live_session_schema


class ScenarioWorkspace:
    """Owns an isolated DB and artifacts for one scenario run."""

    def __init__(self, scenario_id: str, *, root: str | Path | None = None, database_fixture: str = "") -> None:
        self._temporary = None if root else tempfile.TemporaryDirectory(prefix=f"hebe-replay-{scenario_id}-")
        self.root = Path(root or self._temporary.name).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "hebe-replay.sqlite3"
        self.report_dir = self.root / "reports"
        self.report_dir.mkdir(exist_ok=True)
        self.database_fixture = str(database_fixture or "")
        self._old_db_path = db_sqlite.DB_PATH
        self._old_env_db_path = os.environ.get("HEBE_DB_PATH")
        self.migrations: list[dict[str, Any]] = []

    def activate(self) -> None:
        # A retained workspace is an artifact, not an input. Always start it
        # clean so rerunning the same named scenario stays deterministic.
        if self.db_path.exists():
            self.db_path.unlink()
        if self.database_fixture:
            source = Path(self.database_fixture).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            if source == self.db_path:
                raise ValueError("database fixture and scenario database must differ")
            shutil.copy2(source, self.db_path)
        os.environ["HEBE_DB_PATH"] = str(self.db_path)
        db_sqlite.DB_PATH = str(self.db_path)
        stream_memory._READY_DB_PATH = None
        db_sqlite.init_db()
        init_memory_chunks_schema()
        init_live_session_schema()
        stream_memory.init_stream_memory_schema()
        runner = MigrationRunner(lambda: sqlite3.connect(str(self.db_path)))
        self.migrations = runner.migrate(replay_foundation_migrations())

    def connection(self, *, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            uri = self.db_path.as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        else:
            conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def deactivate(self) -> None:
        db_sqlite.DB_PATH = self._old_db_path
        stream_memory._READY_DB_PATH = None
        if self._old_env_db_path is None:
            os.environ.pop("HEBE_DB_PATH", None)
        else:
            os.environ["HEBE_DB_PATH"] = self._old_env_db_path

    def cleanup(self) -> None:
        self.deactivate()
        if self._temporary is not None:
            self._temporary.cleanup()

    def __enter__(self) -> "ScenarioWorkspace":
        self.activate()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cleanup()
