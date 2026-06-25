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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


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


def read_jsonl_recent(kind: str, *, minutes: int = 10, limit: int = 200) -> list[dict]:
    filename = JSONL_LOGS.get(kind, kind)
    path = LOG_DIR / filename
    cutoff = time.time() - max(0, int(minutes)) * 60
    rows: list[dict] = []
    if not path.exists():
        return rows
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
        return []
    return rows[-max(1, int(limit)) :]


def read_text_recent(filename: str, *, minutes: int = 10, max_lines: int = 300) -> list[str]:
    path = LOG_DIR / filename
    if not path.exists():
        return []
    cutoff = time.time() - max(0, int(minutes)) * 60
    lines: list[str] = []
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
        return []
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
        "pending_clarification": getattr(state, "pending_clarification", None),
        "pending_tts_scope": getattr(state, "pending_tts_scope", None),
        "pending_reminder": getattr(state, "pending_reminder", None),
        "pending_confirmation": getattr(state, "pending_confirmation", None),
        "game_run_state": game_run_state,
        "stream": {
            "enabled": getattr(stream, "enabled", None),
            "is_live": getattr(stream, "is_live", None),
            "session_id": getattr(stream, "session_id", None),
            "last_session_id": getattr(stream, "last_session_id", None),
            "stream_output_mode": getattr(stream, "stream_output_mode", None),
            "current_game": getattr(stream, "current_game", None),
            "schedule_slot": getattr(stream, "schedule_slot", None),
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
        from app.services import db_sqlite

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
    minutes: int = 30,
    include_db_snapshot: bool = False,
    include_config: bool = False,
    include_recent_state: bool = True,
    include_recent_ui: bool = True,
    engine: Any = None,
) -> Path:
    ensure_log_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = DEBUG_BUNDLE_DIR / f"hebe_debug_{stamp}_{int(time.time() * 1000)}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "minutes": minutes,
            "commit": _git_commit(),
            "include_db_snapshot": include_db_snapshot,
            "include_config": include_config,
            "include_recent_state": include_recent_state,
            "include_recent_ui": include_recent_ui,
        }
        _write_json(bundle, "metadata.json", metadata)
        for filename in set(TEXT_LOGS.values()) | set(JSONL_LOGS.values()):
            source = LOG_DIR / filename
            if source.exists():
                bundle.write(source, f"logs/{filename}")
        _write_json(bundle, "recent/cognitive_router.json", read_jsonl_recent("cognitive_router", minutes=minutes, limit=500))
        _write_json(bundle, "recent/stt.json", read_jsonl_recent("stt", minutes=minutes, limit=500))
        _write_json(bundle, "recent/input_firewall.json", read_jsonl_recent("input_firewall", minutes=minutes, limit=500))
        _write_json(bundle, "recent/game_guidance.json", read_jsonl_recent("game_guidance", minutes=minutes, limit=500))
        _write_json(bundle, "recent/plan_executor.json", read_jsonl_recent("plan_executor", minutes=minutes, limit=500))
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
