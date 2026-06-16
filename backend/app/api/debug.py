import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request

from app.services import db_sqlite
from app.core.log_bus import get_recent_logs
from app.stream.live_session import latest_live_session_debug

router = APIRouter(prefix="/debug", tags=["debug"])

SENSITIVE_COLUMN_PARTS = (
    "token",
    "api_key",
    "secret",
    "password",
    "oauth",
    "authorization",
    "bearer",
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
