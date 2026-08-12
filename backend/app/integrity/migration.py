from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from app.replay.migrations import (
    MigrationRunner, architecture_consolidation_migrations, belief_v2_migrations,
    conversation_continuity_migrations, game_context_v2_migrations,
    learning_v2_migrations, replay_foundation_migrations, social_world_v2_migrations,
)

from .hygiene import HygienePlanner
from .scanner import IntegrityScanner


ALL_MIGRATIONS = (
    replay_foundation_migrations,
    conversation_continuity_migrations,
    belief_v2_migrations,
    game_context_v2_migrations,
    social_world_v2_migrations,
    learning_v2_migrations,
    architecture_consolidation_migrations,
)


def schema_snapshot(path: Path) -> dict:
    conn=sqlite3.connect(f"file:{path.as_posix()}?mode=ro",uri=True)
    try:
        tables=[]
        for row in conn.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
            name=str(row[0]);columns=[str(c[1]) for c in conn.execute(f'PRAGMA table_info("{name}")')]
            tables.append({"name":name,"columns":columns,"row_count":int(conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0])})
        indexes=[{"name":str(r[0]),"table":str(r[1]),"sql":str(r[2] or "")} for r in conn.execute("SELECT name,tbl_name,sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {"table_count":len(tables),"index_count":len(indexes),"total_rows":sum(t["row_count"] for t in tables),"tables":tables,"indexes":indexes}
    finally:conn.close()


def migrate(path: Path) -> tuple[list[dict], list[dict]]:
    runner=MigrationRunner(lambda:sqlite3.connect(path));first=[];second=[]
    for factory in ALL_MIGRATIONS:first.extend(runner.migrate(factory()))
    for factory in ALL_MIGRATIONS:second.extend(runner.migrate(factory()))
    return first,second


def verify_copied_database(source: str | Path, copy_path: str | Path, *, apply_safe: bool = True) -> dict:
    source=Path(source).resolve();copy_path=Path(copy_path).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    if source==copy_path:raise ValueError("source and copy must differ")
    copy_path.parent.mkdir(parents=True,exist_ok=True)
    source_before=IntegrityScanner.fingerprint(source);shutil.copy2(source,copy_path)
    copied_initial=IntegrityScanner.fingerprint(copy_path);before=schema_snapshot(copy_path)
    first,second=migrate(copy_path)
    dry=HygienePlanner(copy_path).plan();applied=HygienePlanner(copy_path).apply_safe(dry) if apply_safe else None
    integrity=IntegrityScanner(copy_path).scan();after=schema_snapshot(copy_path)
    # Restart safety is represented by closing/reopening the DB and rerunning every migration.
    restart_migrations=[];runner=MigrationRunner(lambda:sqlite3.connect(copy_path))
    for factory in ALL_MIGRATIONS:restart_migrations.extend(runner.migrate(factory()))
    with closing(sqlite3.connect(copy_path)) as conn:
        startup_status="PASS" if str(conn.execute("PRAGMA quick_check").fetchone()[0])=="ok" else "FAIL"
    source_after=IntegrityScanner.fingerprint(source)
    return {
      "schema_version":1,"source_db":str(source),"copied_db":str(copy_path),
      "source_fingerprint_before":source_before,"source_fingerprint_after":source_after,
      "source_untouched":source_before==source_after,"copied_initial_fingerprint":copied_initial,
      "copy_matches_source":copied_initial==source_before,"schema_before":before,"schema_after":after,
      "migrations_first_pass":first,"migrations_second_pass":second,
      "all_second_pass_already_applied":all(x["already_applied"] for x in second),
      "hygiene_counts":dry["classification_counts"],"safe_apply":applied,
      "integrity_status":integrity["status"],"integrity_blocking_errors":integrity["blocking_error_count"],
      "restart_migrations_idempotent":all(x["already_applied"] for x in restart_migrations),
      "application_startup_status":startup_status,
    }


def write_report(report: dict, path: str | Path) -> None:
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
