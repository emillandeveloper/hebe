from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from app.services import db_sqlite


BACKEND_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BACKEND_DIR / "logs"
DEBUG_BUNDLE_DIR = LOG_DIR / "debug_bundles"
SESSION_LOG_DIR = LOG_DIR / "sessions"

TEXT_LOGS = {
    "current": "current.log",
    "errors": "errors.log",
    "stt": "stt.log",
    "routing": "routing.log",
    "stream": "stream.log",
    "tts": "tts.log",
    "ui_events": "ui_events.log",
}

JSONL_LOGS = {
    "stt": "stt.jsonl",
    "input_firewall": "input_firewall.jsonl",
    "cognitive_router": "cognitive_router.jsonl",
    "pending": "pending_tasks.jsonl",
    "game_guidance": "game_guidance.jsonl",
    "plan_executor": "plan_executor.jsonl",
    "proactive_decisions": "proactive_decisions.jsonl",
    "tts": "tts.jsonl",
    "ui_events": "ui_events.jsonl",
}

MAX_BYTES = int(os.getenv("HEBE_LOG_MAX_BYTES", str(2 * 1024 * 1024)))
BACKUP_COUNT = int(os.getenv("HEBE_LOG_BACKUP_COUNT", "5"))
_lock = threading.RLock()

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?([A-Za-z0-9._~+/=-]{10,})"),
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._~+/=-]{10,})"),
    re.compile(r"(?i)(oauth[:_ -]?token\s*[:=]\s*)([A-Za-z0-9._~+/=-]{8,})"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([A-Za-z0-9._~+/=-]{8,})"),
    re.compile(r"(?i)(client[_-]?secret\s*[:=]\s*)([A-Za-z0-9._~+/=-]{8,})"),
    re.compile(r"(?i)(password\s*[:=]\s*)([^\s,;]{4,})"),
    re.compile(r"(?i)(sk-[A-Za-z0-9]{12,})"),
)

SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "authorization",
    "api_key",
    "apikey",
    "oauth",
    "bearer",
)


def ensure_log_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)


def redact_text(value: Any) -> str:
    text = str(value if value is not None else "")
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 3:
            text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2) or ''}[REDACTED]", text)
        elif pattern.groups == 2:
            text = pattern.sub(lambda m: f"{m.group(1)}[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def redact_value(value: Any, key: str = "") -> Any:
    lowered = str(key or "").lower()
    if any(part in lowered for part in SENSITIVE_KEY_PARTS):
        if value in (None, ""):
            return value
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(item, key) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item, key) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _rotate_if_needed(path: Path) -> None:
    try:
        if not path.exists() or path.stat().st_size < MAX_BYTES:
            return
        for idx in range(BACKUP_COUNT - 1, 0, -1):
            src = path.with_name(f"{path.name}.{idx}")
            dst = path.with_name(f"{path.name}.{idx + 1}")
            if dst.exists():
                dst.unlink()
            if src.exists():
                src.rename(dst)
        first = path.with_name(f"{path.name}.1")
        if first.exists():
            first.unlink()
        path.rename(first)
    except Exception:
        pass


def _append_line(path: Path, line: str) -> None:
    ensure_log_dirs()
    with _lock:
        _rotate_if_needed(path)
        with path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(line.rstrip("\r\n") + "\n")


def record_console_log(entry: dict) -> None:
    ts = datetime.fromtimestamp(float(entry.get("ts") or time.time()), tz=timezone.utc).isoformat()
    message = redact_text(entry.get("message") or entry.get("raw") or "")
    source = redact_text(entry.get("source") or "")
    level = redact_text(entry.get("level") or "info")
    category = redact_text(entry.get("category") or "backend")
    line = f"{ts} {level.upper()} {category} {source} {message}"

    _append_line(LOG_DIR / TEXT_LOGS["current"], line)
    if level == "error":
        _append_line(LOG_DIR / TEXT_LOGS["errors"], line)
    if category in {"stt", "tts"}:
        _append_line(LOG_DIR / TEXT_LOGS[category], line)
    if category in {"twitch", "stream_context"} or "[HEBE][STREAM" in message:
        _append_line(LOG_DIR / TEXT_LOGS["stream"], line)
    if category == "routing" or any(marker in message for marker in ("[HEBE][COGNITIVE", "[HEBE][INPUT_ENVELOPE]", "[HEBE][PENDING")):
        _append_line(LOG_DIR / TEXT_LOGS["routing"], line)
    if category == "ui_events" or "[HEBE][UI_EVENT]" in message:
        _append_line(LOG_DIR / TEXT_LOGS["ui_events"], line)


def log_jsonl_event(kind: str, payload: dict | None = None) -> None:
    filename = JSONL_LOGS.get(kind)
    if not filename:
        return
    event = redact_value(dict(payload or {}))
    event.setdefault("event_kind", kind)
    event.setdefault("ts", time.time())
    line = json.dumps(event, ensure_ascii=False, default=str, sort_keys=True)
    _append_line(LOG_DIR / filename, line)


def _log_file_variants(filename: str) -> list[Path]:
    base = LOG_DIR / filename
    paths = []
    if base.exists():
        paths.append(base)
    paths.extend(sorted(LOG_DIR.glob(f"{filename}.*"), key=lambda item: item.name))
    return paths


def read_jsonl_recent(kind: str, *, minutes: int = 10, limit: int = 200) -> list[dict]:
    filename = JSONL_LOGS.get(kind, kind)
    cutoff = time.time() - max(0, int(minutes)) * 60
    rows: list[dict] = []
    paths = _log_file_variants(filename)
    if not paths:
        return rows
    for path in paths:
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    if float(item.get("ts") or 0) < cutoff:
                        continue
                except (TypeError, ValueError):
                    pass
                rows.append(redact_value(item))
        except Exception:
            continue
    rows.sort(key=lambda item: float(item.get("ts") or 0) if isinstance(item, dict) else 0)
    return rows[-max(1, int(limit)) :]


def read_text_recent(filename: str, *, minutes: int = 10, max_lines: int = 300) -> list[str]:
    paths = _log_file_variants(filename)
    if not paths:
        return []
    cutoff = time.time() - max(0, int(minutes)) * 60
    lines: list[str] = []
    for path in paths:
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                include = True
                prefix = line[:32]
                try:
                    if "T" in prefix:
                        dt = datetime.fromisoformat(prefix.split(" ")[0].replace("Z", "+00:00"))
                        include = dt.timestamp() >= cutoff
                except Exception:
                    include = True
                if include:
                    lines.append(redact_text(line))
        except Exception:
            continue
    return lines[-max(1, int(max_lines)) :]


def _write_json(zip_file: zipfile.ZipFile, name: str, payload: Any) -> None:
    zip_file.writestr(name, json.dumps(redact_value(payload), ensure_ascii=False, indent=2, default=str))


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BACKEND_DIR.parent),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def sanitized_config_summary() -> dict:
    interesting = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("HEBE_") or key.startswith("TWITCH_") or key.startswith("OPENAI_")
    }
    return redact_value(interesting)


def engine_state_summary(engine: Any) -> dict:
    state = getattr(getattr(engine, "runtime", None), "state", None)
    stream = getattr(state, "stream", None) if state is not None else None
    game_run_state = getattr(state, "game_run_state", None) if state is not None else None
    if hasattr(game_run_state, "to_dict"):
        game_run_state = game_run_state.to_dict()
    summary = {
        "game_run_state": game_run_state,
        "stream": {
            "enabled": getattr(stream, "enabled", None),
            "is_live": getattr(stream, "is_live", None),
            "session_id": getattr(stream, "session_id", None),
            "last_session_id": getattr(stream, "last_session_id", None),
            "stream_output_mode": getattr(stream, "stream_output_mode", None),
            "current_game": getattr(stream, "current_game", None),
            "schedule_slot": getattr(stream, "schedule_slot", None),
            "last_proactive_decision": getattr(stream, "last_proactive_decision", None),
        } if stream is not None else None,
        "last_policy_decision": None,
        "active_behavior_blocks": [],
    }
    if engine is not None:
        for name, key in (("get_last_policy_trace", "last_policy_decision"), ("get_active_behavior_blocks", "active_behavior_blocks")):
            method = getattr(engine, name, None)
            if callable(method):
                try:
                    summary[key] = method()
                except Exception as exc:
                    summary[key] = {"error": repr(exc)}
    return redact_value(summary)


def _capability_summary() -> dict:
    try:
        from app.api.capabilities import capability_summary_payload, capability_backlog_payload

        return {
            "summary": capability_summary_payload(),
            "backlog": capability_backlog_payload(),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def _recent_db_snapshot() -> dict:
    try:
        path = Path(db_sqlite.DB_PATH)
        if not path.exists():
            return {"ok": False, "reason": "database_not_found"}
        conn = sqlite3.connect(str(path), timeout=1.0)
        conn.row_factory = sqlite3.Row
        try:
            tables = [row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
            payload: dict[str, Any] = {}
            for table in tables[:30]:
                safe_table = '"' + str(table).replace('"', '""') + '"'
                rows = conn.execute(f"SELECT * FROM {safe_table} LIMIT 25").fetchall()
                payload[str(table)] = [redact_value(dict(row)) for row in rows]
            return {"ok": True, "tables": payload}
        finally:
            conn.close()
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def create_debug_bundle(
    *,
    minutes: int = 300,
    mode: str = "last_5_hours",
    session_id: int | None = None,
    include_db_snapshot: bool = False,
    include_config: bool = False,
    include_recent_state: bool = True,
    include_recent_ui: bool = True,
    engine: Any = None,
) -> Path:
    ensure_log_dirs()
    window = _debug_bundle_window(mode=mode, minutes=minutes, session_id=session_id)
    minutes = int(window.get("minutes") or minutes)
    created_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = DEBUG_BUNDLE_DIR / f"hebe_debug_{stamp}_{int(time.time() * 1000)}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        included_logs: list[dict[str, Any]] = []
        for filename in set(TEXT_LOGS.values()) | set(JSONL_LOGS.values()):
            for source in _log_file_variants(filename):
                if not source.exists():
                    continue
                arcname = f"logs/{source.name}"
                bundle.write(source, arcname)
                try:
                    size = source.stat().st_size
                except OSError:
                    size = None
                included_logs.append({"path": arcname, "size": size})
        metadata = {
            "export_mode": window.get("mode"),
            "requested_minutes": minutes,
            "actual_start_time": window.get("actual_start_time") or window.get("start_time"),
            "actual_end_time": window.get("actual_end_time") or window.get("end_time"),
            "created_at": created_at,
            "included_logs": included_logs,
            "stream_session_id": window.get("session_id"),
            "twitch_stream_id": window.get("stream_id"),
            "app_version": "",
            "git_commit": _git_commit(),
            "mode": window.get("mode"),
            "minutes": minutes,
            "session_id": window.get("session_id"),
            "stream_id": window.get("stream_id"),
            "start_time": window.get("start_time"),
            "end_time": window.get("end_time"),
            "duration": window.get("duration"),
            "log_window": window,
            "commit": _git_commit(),
            "include_db_snapshot": include_db_snapshot,
            "include_config": include_config,
            "include_recent_state": include_recent_state,
            "include_recent_ui": include_recent_ui,
        }
        _write_json(bundle, "metadata.json", metadata)
        _write_json(bundle, "recent/cognitive_router.json", read_jsonl_recent("cognitive_router", minutes=minutes, limit=500))
        _write_json(bundle, "recent/stt.json", read_jsonl_recent("stt", minutes=minutes, limit=500))
        _write_json(bundle, "recent/input_firewall.json", read_jsonl_recent("input_firewall", minutes=minutes, limit=500))
        _write_json(bundle, "recent/game_guidance.json", read_jsonl_recent("game_guidance", minutes=minutes, limit=500))
        _write_json(bundle, "recent/plan_executor.json", read_jsonl_recent("plan_executor", minutes=minutes, limit=500))
        _write_json(bundle, "recent/proactive_decisions.json", read_jsonl_recent("proactive_decisions", minutes=minutes, limit=500))
        _write_json(bundle, "recent/errors.json", read_text_recent("errors.log", minutes=minutes, max_lines=500))
        if include_recent_ui:
            _write_json(bundle, "recent/ui_events.json", read_jsonl_recent("ui_events", minutes=minutes, limit=500))
        if include_recent_state:
            _write_json(bundle, "state/current_state.json", engine_state_summary(engine))
        if include_config:
            _write_json(bundle, "state/sanitized_config.json", sanitized_config_summary())
        _write_json(bundle, "state/capability_backlog_summary.json", _capability_summary())
        if include_db_snapshot:
            _write_json(bundle, "state/sanitized_db_snapshot.json", _recent_db_snapshot())
    return path


def _debug_bundle_window(*, mode: str, minutes: int, session_id: int | None = None) -> dict:
    selected_mode = str(mode or "last_5_hours").strip().lower()
    now = datetime.now(timezone.utc)
    fixed = {
        "last_30_minutes": 30,
        "last_2_hours": 120,
        "last_5_hours": 300,
    }
    if selected_mode in fixed:
        requested_minutes = fixed[selected_mode]
        start = now - timedelta(minutes=requested_minutes)
        return {
            "mode": selected_mode,
            "minutes": requested_minutes,
            "actual_start_time": start.isoformat(),
            "actual_end_time": now.isoformat(),
            "start_time": start.isoformat(),
            "end_time": now.isoformat(),
        }
    if selected_mode not in {"current_stream_session", "by_session_id"}:
        requested_minutes = max(1, int(minutes or 300))
        start = now - timedelta(minutes=requested_minutes)
        return {
            "mode": f"last_{requested_minutes}_minutes",
            "minutes": requested_minutes,
            "actual_start_time": start.isoformat(),
            "actual_end_time": now.isoformat(),
            "start_time": start.isoformat(),
            "end_time": now.isoformat(),
        }
    try:
        conn = db_sqlite.get_db_connection()
        try:
            if selected_mode == "by_session_id" and session_id:
                row = conn.execute("SELECT * FROM stream_sessions WHERE id = ?", (int(session_id),)).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT * FROM stream_sessions
                    WHERE status IN ('live', 'ended', 'stale_closed')
                    ORDER BY COALESCE(started_at, created_at) DESC, id DESC LIMIT 1
                    """
                ).fetchone()
        finally:
            conn.close()
    except Exception:
        row = None
    if not row:
        requested_minutes = max(1, int(minutes or 300))
        start = now - timedelta(minutes=requested_minutes)
        return {
            "mode": selected_mode,
            "minutes": requested_minutes,
            "actual_start_time": start.isoformat(),
            "actual_end_time": now.isoformat(),
            "start_time": start.isoformat(),
            "end_time": now.isoformat(),
        }
    start = row["started_at"] or row["created_at"]
    end = row["ended_at"] or datetime.now(timezone.utc).isoformat()
    duration = _seconds_between_iso(start, end)
    derived_minutes = max(1, min(24 * 60, int((duration or 0) / 60) + 10))
    return {
        "mode": selected_mode,
        "minutes": derived_minutes,
        "session_id": row["id"],
        "stream_id": row["twitch_stream_id"],
        "actual_start_time": start,
        "actual_end_time": end,
        "start_time": start,
        "end_time": end,
        "duration": duration,
    }


def _seconds_between_iso(start: str | None, end: str | None) -> int | None:
    def parse(value: str | None) -> datetime | None:
        if not value:
            return None
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    a = parse(start)
    b = parse(end)
    if not a or not b:
        return None
    return max(0, int((b - a).total_seconds()))


def prune_debug_bundles(max_files: int = 12) -> None:
    ensure_log_dirs()
    bundles = sorted(DEBUG_BUNDLE_DIR.glob("hebe_debug_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in bundles[max_files:]:
        try:
            old.unlink()
        except Exception:
            pass


def reset_logs_for_tests() -> None:
    if LOG_DIR.exists():
        shutil.rmtree(LOG_DIR)
    ensure_log_dirs()
