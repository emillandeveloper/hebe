import os
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.capabilities import (
    capability_backlog_payload,
    capability_detail_payload,
    capability_list_payload,
    capability_summary_payload,
    implemented_disabled_capabilities_payload,
    next_capability_payload,
    partial_capabilities_payload,
    planned_capabilities_payload,
)
from app.services import db_sqlite
from app.core.log_bus import get_recent_logs
from app.core.persistent_logs import (
    create_debug_bundle,
    prune_debug_bundles,
    read_jsonl_recent,
    read_text_recent,
)
from app.stream.live_session import latest_live_session_debug
from app.stream import memory as stream_memory
from app.stream.viewer_profiles import ViewerLinguisticProfileStore

router = APIRouter(prefix="/debug", tags=["debug"])


class ViewerProfileUpdate(BaseModel):
    twitch_user_id: str = ""
    display_name: str = ""
    preferred_grammatical_gender: str = "unknown"
    pronouns: dict[str, str] = Field(default_factory=dict)


@router.get("/viewer-profiles")
def get_viewer_profiles():
    return {"ok": True, "profiles": ViewerLinguisticProfileStore().list_profiles()}


@router.put("/viewer-profiles/{login}")
def update_viewer_profile(login: str, body: ViewerProfileUpdate):
    store = ViewerLinguisticProfileStore()
    profile, action = store.apply_evidence(
        twitch_user_id=body.twitch_user_id or f"login:{login.casefold()}", login=login,
        display_name=body.display_name or login,
        candidate_gender=body.preferred_grammatical_gender, confidence=1.0,
        source_type="manual", evidence_summary="manual debug/settings correction",
    )
    profile.pronouns = dict(body.pronouns or {})
    profile.owner_locked = True
    store.save(profile)
    return {"ok": True, "action": action, "profile": profile.to_dict()}


@router.delete("/viewer-profiles/{login}")
def clear_viewer_profile(login: str):
    return {"ok": True, "cleared": ViewerLinguisticProfileStore().clear(login=login)}

SENSITIVE_COLUMN_PARTS = (
    "token",
    "api_key",
    "secret",
    "password",
    "oauth",
    "authorization",
    "bearer",
)


def _stream_data_repair_enabled() -> bool:
    return (
        os.getenv("ELECTRON_DEV", "0").strip() == "1"
        or os.getenv("HEBE_DEV_CONTROLS", "0").strip() == "1"
        or os.getenv("HEBE_STREAM_DATA_REPAIR_ENABLED", "0").strip() == "1"
    )


def _log_db_error(message: str) -> None:
    print(f"[HEBE][DB_INSPECTOR] {message}", flush=True)


def _db_path() -> Path:
    return Path(db_sqlite.DB_PATH).expanduser().resolve()


def _connect_readonly() -> sqlite3.Connection:
    path = _db_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    uri_path = quote(path.as_posix(), safe="/:")
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True, timeout=1.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name COLLATE NOCASE
        """
    ).fetchall()
    return [str(row["name"]) for row in rows]


def _require_table(conn: sqlite3.Connection, table_name: str) -> str:
    if not table_name or table_name not in set(_table_names(conn)):
        raise HTTPException(status_code=404, detail="Unknown table")
    return table_name


def _column_count(conn: sqlite3.Connection, table_name: str) -> int:
    return len(conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall())


def _is_sensitive_column(column_name: str) -> bool:
    lowered = column_name.lower()
    return any(part in lowered for part in SENSITIVE_COLUMN_PARTS)


def _mask_value(value):
    if value is None or value == "":
        return value
    text = str(value)
    if len(text) <= 8:
        return "[masked]"
    return f"{text[:4]}********{text[-4:]}"


def _mask_row(row: sqlite3.Row) -> dict:
    result = {}
    for key in row.keys():
        value = row[key]
        result[key] = _mask_value(value) if _is_sensitive_column(key) else value
    return result

@router.post("/push-event")
async def push_event(request: Request, body: dict):
    """
    Inyecta un InternalEvent manual en el scheduler.
    Body: {"event_type": "twitch_sub", "payload": {...}}
    """
    event_type = body.get("event_type")
    payload = body.get("payload") or {}

    if not event_type:
        raise HTTPException(400, "missing event_type")

    # Acceso al engine — adapta a tu DI real (probablemente request.app.state.adapter)
    adapter = request.app.state.adapter
    engine = getattr(adapter, "_engine", None)

    if engine is None or not adapter.running:
        raise HTTPException(503, "engine not running")

    event = engine.scheduler.push_event(event_type, payload)
    return {"ok": True, "event_type": event.event_type, "created_at": event.created_at}


@router.get("/db/tables")
def list_db_tables():
    try:
        with closing(_connect_readonly()) as conn:
            tables = []
            for name in _table_names(conn):
                quoted = _quote_identifier(name)
                row_count = conn.execute(f"SELECT COUNT(*) AS count FROM {quoted}").fetchone()["count"]
                tables.append(
                    {
                        "name": name,
                        "row_count": int(row_count),
                        "column_count": _column_count(conn, name),
                    }
                )
            return {"db_path": str(_db_path()), "tables": tables}
    except HTTPException:
        raise
    except Exception as exc:
        _log_db_error(f"list tables failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Database read failed")


@router.get("/logs")
def list_backend_logs(limit: int = Query(1000, ge=1, le=5000)):
    return {"logs": get_recent_logs(limit=limit)}


@router.get("/logs/recent")
def preview_recent_logs(minutes: int = Query(10, ge=1, le=720)):
    return {
        "minutes": minutes,
        "errors": read_text_recent("errors.log", minutes=minutes, max_lines=120),
        "cognitive_router": read_jsonl_recent("cognitive_router", minutes=minutes, limit=120),
        "stt": read_jsonl_recent("stt", minutes=minutes, limit=120),
        "proactive_decisions": read_jsonl_recent("proactive_decisions", minutes=minutes, limit=120),
    }


@router.get("/export-logs")
def export_logs(
    request: Request,
    minutes: int = Query(300, ge=1, le=720),
    mode: str = Query("last_5_hours"),
    session_id: int | None = Query(None),
    include_db_snapshot: bool = Query(False),
    include_config: bool = Query(False),
    include_recent_state: bool = Query(True),
    include_recent_ui: bool = Query(True),
):
    adapter = getattr(request.app.state, "adapter", None)
    engine = getattr(adapter, "_engine", None) if adapter is not None else None
    try:
        bundle_path = create_debug_bundle(
            minutes=minutes,
            mode=mode,
            session_id=session_id,
            include_db_snapshot=include_db_snapshot,
            include_config=include_config,
            include_recent_state=include_recent_state,
            include_recent_ui=include_recent_ui,
            engine=engine,
        )
        prune_debug_bundles()
    except Exception as exc:
        _log_db_error(f"export logs failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Debug bundle export failed")
    return FileResponse(
        str(bundle_path),
        media_type="application/zip",
        filename=bundle_path.name,
    )


@router.get("/stream-data/health")
def get_stream_data_health():
    try:
        return {"ok": True, **stream_memory.stream_data_health()}
    except Exception as exc:
        _log_db_error(f"stream data health failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Stream data health failed")


@router.post("/stream-data/repair")
def repair_stream_data(dry_run: bool = Query(True)):
    if not _stream_data_repair_enabled():
        raise HTTPException(status_code=404, detail="Not found")
    try:
        return {"ok": True, **stream_memory.repair_stream_data(dry_run=bool(dry_run))}
    except Exception as exc:
        _log_db_error(f"stream data repair failed: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Stream data repair failed")


@router.get("/capabilities")
def list_capabilities(
    status: str | None = Query(None),
    category: str | None = Query(None),
    executable: bool | None = Query(None),
):
    return capability_list_payload(status=status, category=category, executable=executable)


@router.get("/capabilities/summary")
def get_capability_summary():
    return capability_summary_payload()


@router.get("/capabilities/backlog")
def get_capability_backlog():
    return capability_backlog_payload()


@router.get("/capabilities/backlog/next")
def get_next_capability_todo():
    return next_capability_payload()


@router.get("/capabilities/backlog/planned")
def get_planned_capability_backlog():
    return planned_capabilities_payload()


@router.get("/capabilities/backlog/partial")
def get_partial_capability_backlog():
    return partial_capabilities_payload()


@router.get("/capabilities/backlog/implemented-disabled")
def get_implemented_disabled_capability_backlog():
    return implemented_disabled_capabilities_payload()


@router.get("/capabilities/{capability_id}")
def get_capability(capability_id: str):
    return capability_detail_payload(capability_id)


@router.get("/live-session")
def get_live_session_debug(request: Request):
    adapter = getattr(request.app.state, "adapter", None)
    engine = getattr(adapter, "_engine", None) if adapter is not None else None
    if engine is not None:
        try:
            snapshot = engine._live_session_debug_snapshot()
            if snapshot:
                return {"ok": True, **snapshot}
        except Exception as exc:
            _log_db_error(f"live session engine snapshot failed: {type(exc).__name__}: {exc}")
    snapshot = latest_live_session_debug()
    if snapshot:
        return {"ok": True, **snapshot}
    return {"ok": False, "reason": "no live session state yet"}


@router.get("/game-intelligence")
def get_game_intelligence_debug(request: Request):
    adapter = getattr(request.app.state, "adapter", None)
    engine = getattr(adapter, "_engine", None) if adapter is not None else None
    service = getattr(engine, "game_intelligence", None) if engine is not None else None
    if service is None:
        return {"ok": False, "reason": "game intelligence not running"}
    try:
        return {"ok": True, **service.debug_snapshot()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Game intelligence debug failed: {type(exc).__name__}: {exc}")


@router.get("/policy/last")
def get_last_policy_decision(request: Request):
    adapter = getattr(request.app.state, "adapter", None)
    engine = getattr(adapter, "_engine", None) if adapter is not None else None
    if engine is None:
        return {"ok": False, "reason": "engine not running", "last_policy_decision": None}
    try:
        return {"ok": True, "last_policy_decision": engine.get_last_policy_trace()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Policy debug failed: {type(exc).__name__}: {exc}")


@router.get("/policy/behavior-blocks")
def get_policy_behavior_blocks(request: Request):
    adapter = getattr(request.app.state, "adapter", None)
    engine = getattr(adapter, "_engine", None) if adapter is not None else None
    if engine is None:
        return {"ok": False, "reason": "engine not running", "behavior_blocks": []}
    try:
        return {"ok": True, "behavior_blocks": engine.get_active_behavior_blocks()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Behavior block debug failed: {type(exc).__name__}: {exc}")


@router.get("/db/tables/{table_name}/schema")
def get_db_table_schema(table_name: str):
    try:
        with closing(_connect_readonly()) as conn:
            table = _require_table(conn, table_name)
            columns = [
                {
                    "cid": int(row["cid"]),
                    "name": row["name"],
                    "type": row["type"],
                    "notnull": bool(row["notnull"]),
                    "default_value": row["dflt_value"],
                    "pk": bool(row["pk"]),
                    "sensitive": _is_sensitive_column(str(row["name"])),
                }
                for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
            ]
            return {"table": table, "columns": columns}
    except HTTPException:
        raise
    except Exception as exc:
        _log_db_error(f"schema read failed for {table_name!r}: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Database read failed")


@router.get("/db/tables/{table_name}/rows")
def get_db_table_rows(
    table_name: str,
    limit: int = Query(50, ge=1, le=250),
    offset: int = Query(0, ge=0),
):
    try:
        with closing(_connect_readonly()) as conn:
            table = _require_table(conn, table_name)
            quoted = _quote_identifier(table)
            total = int(conn.execute(f"SELECT COUNT(*) AS count FROM {quoted}").fetchone()["count"])
            rows = conn.execute(f"SELECT * FROM {quoted} LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            columns = list(rows[0].keys()) if rows else [
                row["name"] for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
            ]
            return {
                "table": table,
                "total": total,
                "limit": limit,
                "offset": offset,
                "columns": columns,
                "rows": [_mask_row(row) for row in rows],
            }
    except HTTPException:
        raise
    except Exception as exc:
        _log_db_error(f"rows read failed for {table_name!r}: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Database read failed")
