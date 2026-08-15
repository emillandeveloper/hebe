from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from app.services import app_registry
from app.services.app_registry import persist_learned_app, resolve_whitelisted_app


@dataclass
class ApplicationCandidate:
    canonical_name: str
    display_name: str
    executable_path: str
    arguments: str = ""
    source_type: str = "unknown"
    source_location: str = ""
    alias_matches: list[str] = field(default_factory=list)
    executable_name_match: str = ""
    confidence: float = 0.0
    exists: bool = False
    executable: bool = False
    validation_notes: list[str] = field(default_factory=list)

    @property
    def command(self) -> str:
        if self.arguments:
            return f"{self.executable_path} {self.arguments}".strip()
        return self.executable_path


@dataclass
class CapabilityResolution:
    capability: str
    requested_target: str
    canonical_target: str
    status: str
    implementation: Optional[ApplicationCandidate] = None
    candidate_count: int = 0
    confidence: float = 0.0
    provenance: str = ""
    persisted: bool = False
    clarification_question: Optional[str] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    app_record: Optional[dict[str, Any]] = None


@dataclass
class LocalActionReceipt:
    action: str
    app_id: str
    executable_path: str
    requested_at: str
    launched: bool
    process_id: Optional[int] = None
    verification: Optional[str] = None
    failure_reason: Optional[str] = None
    persisted: bool = False


class ApplicationDiscoveryService:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, Optional[str]], dict[str, Any]] = {}
        self._cache_ttl_s = 300.0

    def search(self, target: str, canonical_app_id: Optional[str] = None, aliases: Sequence[str] | None = None) -> list[ApplicationCandidate]:
        normalized_target = self._normalize_target(target)
        primary_id = canonical_app_id or normalized_target
        now = time.time()
        key = (normalized_target, primary_id)
        cached = self._cache.get(key)
        if cached and now - cached["timestamp"] < self._cache_ttl_s:
            return cached["candidates"]

        candidates: list[ApplicationCandidate] = []
        candidates.extend(self._search_registry_app_paths(normalized_target))
        candidates.extend(self._search_installed_registry(normalized_target))
        candidates.extend(self._search_shortcuts(normalized_target))
        candidates.extend(self._search_executables(normalized_target))
        candidates.extend(self._search_persisted_db_entries(primary_id))
        candidates = self._deduplicate_candidates(candidates)
        candidates = self._validate_candidates(candidates, target, aliases)
        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)

        self._cache[key] = {"timestamp": now, "candidates": candidates}
        return candidates

    def _normalize_target(self, target: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(target or "").strip().casefold())

    def _search_persisted_db_entries(self, canonical_app_id: Optional[str]) -> list[ApplicationCandidate]:
        if not canonical_app_id:
            return []

        candidates: list[ApplicationCandidate] = []
        try:
            persisted = app_registry.lookup_learned_app_record(canonical_app_id)
            if persisted and persisted.get("command"):
                path = str(persisted["command"]).strip()
                candidate = ApplicationCandidate(
                    canonical_name=canonical_app_id,
                    display_name=str(persisted.get("description") or canonical_app_id),
                    executable_path=path,
                    arguments="",
                    source_type="learned_db",
                    source_location=path,
                    alias_matches=[canonical_app_id],
                    executable_name_match=Path(path).name,
                )
                candidates.append(candidate)
        except Exception:
            pass
        return candidates

    def _search_registry_app_paths(self, target: str) -> list[ApplicationCandidate]:
        candidates: list[ApplicationCandidate] = []
        if os.name != "nt":
            return candidates

        try:
            import winreg

            roots = [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]
            for root in roots:
                base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
                with winreg.OpenKey(root, base) as base_key:
                    for i in range(0, winreg.QueryInfoKey(base_key)[0]):
                        try:
                            sub_name = winreg.EnumKey(base_key, i)
                            normalized = re.sub(r"[^a-z0-9]+", "", sub_name.casefold())
                            if target not in normalized:
                                continue
                            with winreg.OpenKey(base_key, sub_name) as sub_key:
                                path, _ = winreg.QueryValueEx(sub_key, None)
                                if path and isinstance(path, str):
                                    candidates.append(ApplicationCandidate(
                                        canonical_name=sub_name,
                                        display_name=sub_name,
                                        executable_path=path,
                                        source_type="app_paths_registry",
                                        source_location=f"{root}:{base}/{sub_name}",
                                        alias_matches=[target],
                                        executable_name_match=Path(path).name,
                                    ))
                        except OSError:
                            continue
        except Exception:
            pass

        return candidates

    def _search_installed_registry(self, target: str) -> list[ApplicationCandidate]:
        candidates: list[ApplicationCandidate] = []
        if os.name != "nt":
            return candidates

        uninstall_roots = [
            ("HKLM", r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            ("HKLM", r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            ("HKCU", r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        ]
        try:
            import winreg

            for hive_name, subkey in uninstall_roots:
                root = winreg.HKEY_CURRENT_USER if hive_name == "HKCU" else winreg.HKEY_LOCAL_MACHINE
                with winreg.OpenKey(root, subkey) as root_key:
                    for i in range(0, winreg.QueryInfoKey(root_key)[0]):
                        try:
                            key_name = winreg.EnumKey(root_key, i)
                            with winreg.OpenKey(root_key, key_name) as app_key:
                                display_name = self._read_registry_value(app_key, "DisplayName") or ""
                                install_location = self._read_registry_value(app_key, "InstallLocation") or ""
                                display_icon = self._read_registry_value(app_key, "DisplayIcon") or ""
                                if not display_name:
                                    continue
                                normalized = re.sub(r"[^a-z0-9]+", "", display_name.casefold())
                                if target not in normalized:
                                    continue
                                path = display_icon or install_location
                                if path and isinstance(path, str):
                                    candidates.append(ApplicationCandidate(
                                        canonical_name=display_name,
                                        display_name=display_name,
                                        executable_path=path,
                                        source_type="installed_registry",
                                        source_location=f"{hive_name}:{subkey}/{key_name}",
                                        alias_matches=[target],
                                        executable_name_match=Path(path).name,
                                    ))
                        except OSError:
                            continue
        except Exception:
            pass

        return candidates

    def _read_registry_value(self, key: Any, value_name: str) -> Optional[str]:
        try:
            value, _type = key.QueryValueEx(value_name)
            return str(value) if isinstance(value, str) else None
        except Exception:
            return None

    def _search_shortcuts(self, target: str) -> list[ApplicationCandidate]:
        candidates: list[ApplicationCandidate] = []
        if os.name != "nt":
            return candidates

        roots = [
            Path(os.environ.get("PROGRAMDATA", "")) / r"Microsoft\Windows\Start Menu\Programs",
            Path(os.environ.get("APPDATA", "")) / r"Microsoft\Windows\Start Menu\Programs",
            Path(os.environ.get("USERPROFILE", "")) / "Desktop",
            Path(os.environ.get("PUBLIC", "")) / "Desktop",
        ]
        for root in roots:
            if not root.exists():
                continue
            try:
                for path in root.rglob("*.lnk"):
                    normalized = re.sub(r"[^a-z0-9]+", "", path.stem.casefold())
                    if target in normalized:
                        candidates.append(ApplicationCandidate(
                            canonical_name=path.stem,
                            display_name=path.stem,
                            executable_path=str(path),
                            source_type="shortcut",
                            source_location=str(path),
                            alias_matches=[target],
                            executable_name_match=path.name,
                        ))
            except Exception:
                continue
        return candidates

    def _search_executables(self, target: str) -> list[ApplicationCandidate]:
        candidates: list[ApplicationCandidate] = []
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
            "service",
            "renderer",
            "launcherhelper",
            "machine-config",
            "node",
        }
        for root in search_roots:
            if not root:
                continue
            root_path = Path(root)
            if not root_path.exists():
                continue
            for path in self._walk_bounded(root_path, max_depth=4):
                if path.suffix.lower() != ".exe":
                    continue
                lower = path.name.casefold()
                if target not in lower:
                    continue
                if any(bad in lower for bad in reject_terms):
                    continue
                candidates.append(ApplicationCandidate(
                    canonical_name=path.stem,
                    display_name=path.stem,
                    executable_path=str(path),
                    source_type="exe_scan",
                    source_location=str(path),
                    alias_matches=[target],
                    executable_name_match=path.name,
                ))
        return candidates

    def _walk_bounded(self, root: Path, max_depth: int) -> Iterable[Path]:
        try:
            root_depth = len(root.parts)
            for path, dirs, files in os.walk(root):
                current_depth = len(Path(path).parts) - root_depth
                if current_depth > max_depth:
                    dirs.clear()
                    continue
                for file in files:
                    yield Path(path) / file
        except Exception:
            return

    def _deduplicate_candidates(self, candidates: list[ApplicationCandidate]) -> list[ApplicationCandidate]:
        seen: dict[str, ApplicationCandidate] = {}
        for candidate in candidates:
            key = candidate.command.lower().strip()
            if not key:
                continue
            existing = seen.get(key)
            if existing is None or candidate.confidence > existing.confidence:
                seen[key] = candidate
        return list(seen.values())

    def _validate_candidates(self, candidates: list[ApplicationCandidate], target: str, aliases: Sequence[str] | None = None) -> list[ApplicationCandidate]:
        normalized_target = self._normalize_target(target)
        alias_keys = {self._normalize_target(alias) for alias in (aliases or []) if alias}

        for candidate in candidates:
            candidate.exists = bool(candidate.executable_path and Path(candidate.executable_path).exists())
            candidate.executable = candidate.exists and Path(candidate.executable_path).is_file()
            candidate.validation_notes = []
            if not candidate.executable:
                candidate.validation_notes.append("path_missing_or_invalid")
            candidate_name = self._normalize_target(candidate.executable_name_match)
            if normalized_target and candidate_name == f"{normalized_target}.exe":
                candidate.confidence += 0.25
            if normalized_target and candidate_name.startswith(normalized_target):
                candidate.confidence += 0.10
            if target and target in candidate.source_location.casefold():
                candidate.confidence += 0.10
            if candidate.source_type == "learned_db":
                candidate.confidence += 0.40
            if candidate.source_type == "app_paths_registry":
                candidate.confidence += 0.30
            if candidate.source_type == "shortcut":
                candidate.confidence += 0.10
            if candidate.executable and candidate.exists:
                candidate.confidence = min(candidate.confidence + 0.05, 1.0)
            candidate.confidence = max(0.0, min(candidate.confidence, 1.0))

        return [candidate for candidate in candidates if candidate.confidence > 0.0]


class LocalCapabilityResolver:
    def __init__(self, discovery_service: ApplicationDiscoveryService | None = None) -> None:
        self.discovery = discovery_service or ApplicationDiscoveryService()

    def resolve_open_application(self, requested_target: str) -> CapabilityResolution:
        app_record = resolve_whitelisted_app(requested_target)
        registered = app_record is not None
        if app_record is None:
            app_id = re.sub(r"[^a-z0-9]+", "", requested_target.casefold())
            app_record = {
                "app_id": app_id or requested_target,
                "display_name": requested_target,
                "aliases": [requested_target],
                "executable_path": "",
                "source": "unregistered",
            }
        app_id = str(app_record.get("app_id") or app_record.get("name") or requested_target).strip()
        display_name = str(app_record.get("display_name") or app_record.get("name") or app_id).strip()
        current_candidate = self._build_candidate_from_record(app_record)
        if current_candidate and self._is_valid_executable(current_candidate):
            return CapabilityResolution(
                capability="open_application",
                requested_target=requested_target,
                canonical_target=display_name,
                status="known",
                implementation=current_candidate,
                candidate_count=1,
                confidence=1.0,
                provenance="existing_registration",
                persisted=False,
                app_record=app_record,
            )

        search_aliases = self._extract_aliases(app_record)
        candidates = self.discovery.search(display_name, canonical_app_id=app_id, aliases=search_aliases)
        if current_candidate and current_candidate.executable_path:
            invalid_reason = "invalid_existing_registration"
        else:
            invalid_reason = "missing_existing_registration"

        valid_candidates = [c for c in candidates if c.executable]
        if not valid_candidates:
            question = self._build_clarification_question(display_name)
            return CapabilityResolution(
                capability="open_application",
                requested_target=requested_target,
                canonical_target=display_name,
                status="not_found",
                implementation=None,
                candidate_count=len(candidates),
                confidence=0.0,
                provenance=invalid_reason,
                persisted=False,
                clarification_question=question,
                diagnostics={
                    "reason": invalid_reason,
                    "registered": registered,
                    "checked_sources": [c.source_type for c in candidates],
                },
                app_record=app_record if registered else None,
            )

        if len(valid_candidates) == 1:
            selected = valid_candidates[0]
            persisted = self._persist_candidate(app_id, display_name, search_aliases, selected)
            return CapabilityResolution(
                capability="open_application",
                requested_target=requested_target,
                canonical_target=display_name,
                status="discovered",
                implementation=selected,
                candidate_count=1,
                confidence=selected.confidence,
                provenance=selected.source_type,
                persisted=persisted,
                app_record=self._record_for_candidate(app_id, display_name, search_aliases, selected),
            )

        top = valid_candidates[0]
        second = valid_candidates[1]
        if top.confidence >= 0.80 and top.confidence - second.confidence >= 0.2:
            persisted = self._persist_candidate(app_id, display_name, search_aliases, top)
            return CapabilityResolution(
                capability="open_application",
                requested_target=requested_target,
                canonical_target=display_name,
                status="discovered",
                implementation=top,
                candidate_count=len(valid_candidates),
                confidence=top.confidence,
                provenance=top.source_type,
                persisted=persisted,
                app_record=self._record_for_candidate(app_id, display_name, search_aliases, top),
            )

        options = [c.source_location or c.executable_path for c in valid_candidates[:2] if c.source_location or c.executable_path]
        if len(options) == 2:
            question = (
                f"Me salen dos {display_name}. ¿Quieres el de {self._shorten_path(options[0])} o el de {self._shorten_path(options[1])}?"
            )
        else:
            question = f"Me salen varias opciones para {display_name}. ¿Cuál quieres?"
        return CapabilityResolution(
            capability="open_application",
            requested_target=requested_target,
            canonical_target=display_name,
            status="ambiguous",
            implementation=None,
            candidate_count=len(valid_candidates),
            confidence=top.confidence,
            provenance="multiple_candidates",
            persisted=False,
            clarification_question=question,
            diagnostics={"candidate_paths": [c.source_location or c.executable_path for c in valid_candidates]},
        )

    @staticmethod
    def _record_for_candidate(
        app_id: str,
        display_name: str,
        aliases: list[str],
        candidate: ApplicationCandidate,
    ) -> dict[str, Any]:
        return {
            "app_id": app_id,
            "display_name": display_name,
            "aliases": aliases,
            "executable_path": candidate.executable_path,
            "command": candidate.command,
            "source": candidate.source_type,
            "name": display_name,
        }

    def _build_candidate_from_record(self, app_record: dict[str, Any]) -> Optional[ApplicationCandidate]:
        path = str(app_record.get("executable_path") or app_record.get("command") or "").strip()
        if not path:
            return None
        display_name = str(app_record.get("display_name") or app_record.get("name") or "").strip()
        return ApplicationCandidate(
            canonical_name=str(app_record.get("app_id") or display_name),
            display_name=display_name,
            executable_path=path,
            arguments="",
            source_type=str(app_record.get("source") or "existing_registration"),
            source_location=path,
            alias_matches=self._extract_aliases(app_record),
            executable_name_match=Path(path).name,
            confidence=0.95,
        )

    def _extract_aliases(self, app_record: dict[str, Any]) -> list[str]:
        aliases: list[str] = []
        if app_record.get("aliases"):
            if isinstance(app_record["aliases"], str):
                aliases.extend([alias.strip() for alias in str(app_record["aliases"]).split(",") if alias.strip()])
            elif isinstance(app_record["aliases"], list):
                aliases.extend([str(alias).strip() for alias in app_record["aliases"] if str(alias).strip()])
        if app_record.get("app_id"):
            aliases.append(str(app_record["app_id"]))
        if app_record.get("display_name"):
            aliases.append(str(app_record["display_name"]))
        return list(dict.fromkeys(alias for alias in aliases if alias))

    def _is_valid_executable(self, candidate: ApplicationCandidate) -> bool:
        if not candidate.executable_path:
            return False
        path = Path(candidate.executable_path)
        if candidate.executable_path.lower().endswith(".lnk"):
            return path.exists() and path.is_file()
        return path.exists() and path.is_file() and path.suffix.lower() == ".exe"

    def _persist_candidate(self, app_id: str, display_name: str, aliases: list[str], candidate: ApplicationCandidate) -> bool:
        try:
            saved = persist_learned_app(
                app_id=app_id,
                canonical_name=display_name,
                aliases=aliases,
                executable_path=candidate.executable_path,
                launch_arguments=candidate.arguments,
                source=candidate.source_type,
                confidence=candidate.confidence,
                process_name=candidate.executable_name_match,
                window_title=candidate.display_name,
            )
            return saved is not None
        except Exception:
            return False

    def _build_clarification_question(self, display_name: str) -> str:
        normalized = display_name.lower()
        return (
            f"No encuentro todavía dónde tienes {display_name}. "
            "¿Lo tienes instalado o es una versión portable?"
        )

    def _shorten_path(self, path: str) -> str:
        if not path:
            return "esta ruta"
        path = str(path)
        if len(path) > 40:
            return path[:40].rstrip("\\/ ") + "..."
        return path
