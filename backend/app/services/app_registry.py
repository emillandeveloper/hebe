from __future__ import annotations

import os
import sqlite3
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _get_db_path() -> Optional[Path]:
    here = Path(__file__).resolve()
    backend_root = here.parents[2]

    candidates = [
        Path(os.getenv("HEBE_DB_PATH")) if os.getenv("HEBE_DB_PATH") else None,
        backend_root / "hebe.db",
        backend_root / "data" / "hebe.db",
    ]

    for path in candidates:
        if path and path.exists():
            return path

    return None


def _connect():
    db_path = _get_db_path()
    if not db_path:
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


@dataclass(frozen=True)
class AppRegistryEntry:
    app_id: str
    display_name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    executable_path: str = ""
    working_directory: str | None = None
    launch_args: tuple[str, ...] = field(default_factory=tuple)
    enabled: bool = True
    requires_confirmation: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "builtin"

    def as_dict(self) -> dict[str, Any]:
        command = self.executable_path
        if self.launch_args:
            command = " ".join([command, *self.launch_args]).strip()
        return {
            "app_id": self.app_id,
            "display_name": self.display_name,
            "aliases": list(self.aliases),
            "executable_path": self.executable_path,
            "working_directory": self.working_directory,
            "launch_args": list(self.launch_args),
            "enabled": self.enabled,
            "requires_confirmation": self.requires_confirmation,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            # Compatibility with the existing Windows launcher.
            "name": self.display_name,
            "command": command,
            "process_name": Path(self.executable_path).name if self.executable_path else "",
            "window_title": self.display_name,
        }


def _normalize_alias(value: str) -> str:
    normalized = "".join(
        ch for ch in unicodedata.normalize("NFKD", str(value or "").strip().casefold())
        if not unicodedata.combining(ch)
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _alias_keys(value: str) -> set[str]:
    normalized = _normalize_alias(value)
    return {normalized, normalized.replace(" ", "")} if normalized else set()


def _safe_builtin_registry() -> list[AppRegistryEntry]:
    obs_env = (os.getenv("HEBE_APP_OBS_PATH") or "").strip()
    default_obs = r"C:\Program Files\obs-studio\bin\64bit\obs64.exe"
    obs_path = obs_env or (default_obs if Path(default_obs).exists() else "")
    now = datetime.now(timezone.utc).isoformat()
    return [
        AppRegistryEntry(
            app_id="obs",
            display_name="OBS Studio",
            aliases=("obs", "obs studio"),
            executable_path=obs_path,
            enabled=True,
            requires_confirmation=False,
            created_at=now,
            updated_at=now,
            source="env" if obs_env else "builtin",
        ),
    ]


def list_whitelisted_apps() -> list[dict[str, Any]]:
    return [entry.as_dict() for entry in _safe_builtin_registry() if entry.enabled]


def resolve_whitelisted_app(name: str) -> Optional[Dict[str, Any]]:
    normalized = _normalize_alias(name)
    if not normalized:
        return None

    for entry in _safe_builtin_registry():
        if not entry.enabled:
            continue
        aliases: set[str] = set()
        for alias in (entry.app_id, entry.display_name, *entry.aliases):
            aliases.update(_alias_keys(alias))
        if _alias_keys(normalized) & aliases:
            app = entry.as_dict()
            print(
                "[HEBE][APP_RESOLVER] "
                f"target={normalized} resolved_app_id={entry.app_id} confidence=1.000",
                flush=True,
            )
            return app

    print(
        "[HEBE][APP_RESOLVER] "
        f"target={normalized} resolved_app_id=None confidence=0.000",
        flush=True,
    )
    return None


def _learned_apps_table_exists(conn: sqlite3.Connection) -> bool:
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name='learned_apps'
            LIMIT 1
            """
        )
        return cur.fetchone() is not None
    except Exception:
        return False


def _ensure_learned_apps_table(conn: sqlite3.Connection) -> None:
    if _learned_apps_table_exists(conn):
        return
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS learned_apps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            app_id TEXT NOT NULL UNIQUE,
            canonical_name TEXT NOT NULL,
            aliases TEXT,
            executable_path TEXT NOT NULL,
            launch_arguments TEXT,
            source TEXT,
            confidence REAL,
            validated_at TEXT,
            learned_at TEXT NOT NULL,
            last_successful_launch TEXT,
            last_failed_launch TEXT,
            validation_version TEXT,
            process_name TEXT,
            window_title TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def persist_learned_app(
    app_id: str,
    canonical_name: str,
    aliases: Sequence[str] | str,
    executable_path: str,
    launch_arguments: str = "",
    source: str = "discovered",
    confidence: float = 0.5,
    process_name: str = "",
    window_title: str = "",
    validation_version: str = "",
) -> Optional[Dict[str, Any]]:
    conn = _connect()
    if not conn:
        return None

    try:
        _ensure_learned_apps_table(conn)
        alias_text = ",".join([alias.strip() for alias in aliases]) if isinstance(aliases, (list, tuple)) else str(aliases or "")
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO learned_apps
            (app_id, canonical_name, aliases, executable_path, launch_arguments, source, confidence,
             validated_at, learned_at, last_successful_launch, last_failed_launch, validation_version,
             process_name, window_title, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(app_id) DO UPDATE SET
                canonical_name=excluded.canonical_name,
                aliases=excluded.aliases,
                executable_path=excluded.executable_path,
                launch_arguments=excluded.launch_arguments,
                source=excluded.source,
                confidence=excluded.confidence,
                validated_at=excluded.validated_at,
                learned_at=COALESCE(learned_apps.learned_at, excluded.learned_at),
                last_successful_launch=COALESCE(learned_apps.last_successful_launch, excluded.last_successful_launch),
                last_failed_launch=COALESCE(learned_apps.last_failed_launch, excluded.last_failed_launch),
                validation_version=excluded.validation_version,
                process_name=excluded.process_name,
                window_title=excluded.window_title,
                updated_at=excluded.updated_at
            """,
            (
                app_id,
                canonical_name,
                alias_text,
                executable_path,
                launch_arguments,
                source,
                float(confidence),
                now,
                now,
                None,
                None,
                validation_version,
                process_name,
                window_title,
                now,
            ),
        )
        conn.commit()
        return lookup_learned_app_record(app_id)
    except Exception as e:
        print(f"[HEBE][APP_REGISTRY] persist_learned_app failed: {e}")
    finally:
        conn.close()
    return None


def lookup_learned_app_record(app_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    if not conn:
        return None

    try:
        if not _learned_apps_table_exists(conn):
            return None
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM learned_apps WHERE lower(app_id) = ? LIMIT 1",
            (app_id.strip().lower(),),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    except Exception as e:
        print(f"[HEBE][APP_REGISTRY] lookup_learned_app_record failed: {e}")
        return None
    finally:
        conn.close()


def _resolve_db_candidates(normalized: str) -> list[Dict[str, Any]]:
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name='app_commands'
            LIMIT 1
            """
        )
        return cur.fetchone() is not None
    except Exception:
        return False


def resolve_app(name: str) -> Optional[Dict[str, Any]]:
    candidates = resolve_candidates(name)
    return candidates[0] if candidates else None


def resolve_candidates(name: str) -> list[Dict[str, Any]]:
    normalized = name.strip().lower()
    if not normalized:
        return []

    out: list[Dict[str, Any]] = []
    seen: set[str] = set()

    # 1. Learned registry
    learned_candidates = _resolve_learned_app_candidates(normalized)
    for candidate in learned_candidates:
        key = _candidate_key(candidate)
        if key not in seen:
            out.append(candidate)
            seen.add(key)

    # 2. BD
    db_candidates = _resolve_db_candidates(normalized)
    for candidate in db_candidates:
        key = _candidate_key(candidate)
        if key not in seen:
            out.append(candidate)
            seen.add(key)

    # 3. Start Menu
    start_menu_candidates = discover_from_start_menu(normalized)
    for candidate in start_menu_candidates:
        key = _candidate_key(candidate)
        if key not in seen:
            out.append(candidate)
            seen.add(key)

    # 4. EXE scan
    exe_candidates = discover_exe(normalized)
    for candidate in exe_candidates:
        key = _candidate_key(candidate)
        if key not in seen:
            out.append(candidate)
            seen.add(key)

    return out


def _resolve_learned_app_candidates(normalized: str) -> list[Dict[str, Any]]:
    conn = _connect()
    if not conn:
        return []

    try:
        if not _learned_apps_table_exists(conn):
            return []

        cur = conn.cursor()
        like_term = f"%{normalized}%"
        cur.execute(
            """
            SELECT *
            FROM learned_apps
            WHERE lower(app_id) = ?
               OR lower(canonical_name) = ?
               OR lower(aliases) LIKE ?
            LIMIT 50
            """,
            (normalized, normalized, like_term),
        )
        rows = cur.fetchall()

        candidates: list[Dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            if not row_dict.get("executable_path"):
                continue
            candidates.append(
                {
                    "name": row_dict.get("canonical_name") or row_dict.get("app_id"),
                    "alias": row_dict.get("app_id"),
                    "aliases": row_dict.get("aliases"),
                    "command": row_dict.get("executable_path"),
                    "process_name": row_dict.get("process_name"),
                    "window_title": row_dict.get("window_title"),
                    "enabled": 1,
                    "source": "learned_db",
                }
            )
        return candidates
    except Exception as e:
        print(f"[app_registry] resolve_learned_app_candidates error: {e}")
        return []
    finally:
        conn.close()


def register_app(app: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conn = _connect()
    if not conn:
        return None

    try:
        if not _app_commands_table_exists(conn):
            print("[app_registry] register skipped: table 'app_commands' does not exist")
            return None

        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO app_commands
            (name, command, description, aliases, enabled, process_name, window_title)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                app.get("name"),
                app.get("command"),
                app.get("description", ""),
                app.get("aliases", ""),
                int(app.get("enabled", 1)),
                app.get("process_name"),
                app.get("window_title"),
            ),
        )
        conn.commit()

        cur.execute(
            """
            SELECT *
            FROM app_commands
            WHERE lower(name) = ?
            LIMIT 1
            """,
            ((app.get("name") or "").strip().lower(),),
        )
        row = cur.fetchone()
        if row:
            saved = _normalize_app_record(dict(row))
            print(f"[app_registry] registered {saved.get('name')} (id={saved.get('id')})")
            return saved

    except Exception as e:
        print(f"[app_registry] register_app error: {e}")

    finally:
        conn.close()

    return None


def _resolve_db_candidates(normalized: str) -> list[Dict[str, Any]]:
    conn = _connect()
    if not conn:
        return []

    try:
        if not _app_commands_table_exists(conn):
            return []

        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM app_commands
            WHERE lower(name) = ?
               OR lower(command) LIKE ?
               OR lower(aliases) LIKE ?
            LIMIT 50
            """,
            (normalized, f"%{normalized}%", f"%{normalized}%"),
        )
        rows = cur.fetchall()

        exact: list[Dict[str, Any]] = []
        partial: list[Dict[str, Any]] = []

        for row in rows:
            row_dict = dict(row)
            normalized_row = _normalize_app_record(row_dict)

            aliases = (row_dict.get("aliases") or "").strip().lower()
            alias_list = [a.strip() for a in aliases.split(",") if a.strip()]
            name = (row_dict.get("name") or "").strip().lower()

            if normalized == name or normalized in alias_list:
                exact.append(normalized_row)
            else:
                partial.append(normalized_row)

        return exact + partial

    except Exception as e:
        print(f"[app_registry] resolve_app error: {e}")
        return []

    finally:
        conn.close()


def discover_from_start_menu(name: str) -> list[Dict[str, Any]]:
    normalized = name.strip().lower()
    if not normalized:
        return []

    start_menu_roots = [
        Path(os.environ.get("PROGRAMDATA", "")) / r"Microsoft\Windows\Start Menu\Programs",
        Path(os.environ.get("APPDATA", "")) / r"Microsoft\Windows\Start Menu\Programs",
    ]

    candidates: list[Path] = []

    for root in start_menu_roots:
        if not root.exists():
            continue

        try:
            for path in root.rglob("*.lnk"):
                stem = path.stem.lower()
                full = str(path).lower()

                if "uninstall" in stem or "desinstal" in stem:
                    continue

                if normalized in stem or normalized in full:
                    candidates.append(path)
        except Exception:
            continue

    candidates.sort(key=lambda p: len(p.stem))

    out: list[Dict[str, Any]] = []
    for selected in candidates[:10]:
        print(f"[app_registry] discovered from start menu {name} -> {selected}")
        out.append(
            {
                "name": selected.stem,
                "alias": normalized,
                "aliases": normalized,
                "command": str(selected),
                "process_name": None,
                "window_title": selected.stem,
                "enabled": 1,
                "source": "start_menu",
            }
        )
    return out


def discover_exe(name: str) -> list[Dict[str, Any]]:
    normalized = name.strip().lower()
    if not normalized:
        return []

    search_roots = [
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMFILES(X86)"),
        os.environ.get("LOCALAPPDATA"),
    ]

    reject_terms = {
        "uninstall",
        "unins",
        "setup",
        "updater",
        "update",
        "helper",
        "crash",
        "report",
        "test",
        "tests",
        "ffmpeg",
        "browser",
        "stream-elements",
        "streamelements",
        "cef",
        "service",
        "renderer",
        "launcherhelper",
        "machine-config",
        "node",
    }

    candidates: list[str] = []

    for root in search_roots:
        if not root or not os.path.exists(root):
            continue

        try:
            for dirpath, _, filenames in os.walk(root):
                dir_lower = dirpath.lower()

                # poda básica de rutas claramente malas
                if any(bad in dir_lower for bad in ["uninstall", "cache", "temp", "logs", "crash"]):
                    continue

                for file in filenames:
                    lower = file.lower()

                    if not lower.endswith(".exe"):
                        continue

                    if normalized not in lower:
                        continue

                    if any(bad in lower for bad in reject_terms):
                        continue

                    full_path = os.path.join(dirpath, file)
                    candidates.append(full_path)
        except Exception:
            continue

    def score(path: str) -> tuple[int, int, int]:
        lower = path.lower()
        filename = Path(path).name.lower()

        score_value = 0

        # preferir match exacto del exe
        if filename == f"{normalized}.exe":
            score_value += 100

        # preferir nombres que empiecen por el nombre buscado
        if filename.startswith(normalized):
            score_value += 30

        # penalizar rutas sospechosas
        suspicious = [
            "uninstall",
            "cache",
            "temp",
            "logs",
            "crash",
            "streamelements",
            "streamlabs obs\\resources",
            "node_modules",
            "plugin",
            "plugins",
            "obs-plugins",
        ]
        for bad in suspicious:
            if bad in lower:
                score_value -= 40

        # preferir rutas más cortas
        return (-score_value, len(path), len(filename))

    candidates = sorted(set(candidates), key=score)

    out: list[Dict[str, Any]] = []
    for selected in candidates[:10]:
        print(f"[app_registry] discovered exe {name} -> {selected}")
        out.append(
            {
                "name": Path(selected).stem,
                "alias": normalized,
                "aliases": normalized,
                "command": selected,
                "process_name": Path(selected).name,
                "window_title": Path(selected).stem,
                "enabled": 1,
                "source": "exe_scan",
            }
        )
    return out


def _normalize_app_record(row: Dict[str, Any]) -> Dict[str, Any]:
    aliases = row.get("aliases", "")

    primary_alias = ""
    if isinstance(aliases, str) and aliases.strip():
        primary_alias = aliases.split(",")[0].strip()

    return {
        "id": row.get("id"),
        "name": row.get("name"),
        "alias": primary_alias,
        "aliases": aliases,
        "command": row.get("command"),
        "process_name": row.get("process_name"),
        "window_title": row.get("window_title"),
        "enabled": row.get("enabled", 1),
        "source": "db",
    }


def _candidate_key(candidate: Dict[str, Any]) -> str:
    return f"{candidate.get('command','')}|{candidate.get('name','')}"
