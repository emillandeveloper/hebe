from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


STATUS_VERIFIED = "VERIFIED"
STATUS_FAILED = "FAILED"
STATUS_INCOMPLETE = "VERIFICATION_INCOMPLETE"

_SENSITIVE_KEY = re.compile(r"(token|secret|api[_-]?key|oauth|authorization|password|raw[_-]?(chat|text|transcript)|source_text)", re.I)


@dataclass(slots=True)
class CommandVerification:
    command: str
    exit_code: int
    duration_seconds: float


@dataclass(slots=True)
class VerificationReport:
    overall_status: str
    phase_result: str
    repository: dict[str, Any]
    environment: dict[str, Any]
    commands: list[dict[str, Any]] = field(default_factory=list)
    tests: dict[str, Any] = field(default_factory=dict)
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    external_boundaries: dict[str, str] = field(default_factory=dict)
    persistence: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    baseline_differential: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return sanitize(asdict(self))


def build_report(
    *,
    scenario_results: list[dict[str, Any]],
    commands: list[CommandVerification] | None = None,
    tests: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
    baseline_differential: dict[str, Any] | None = None,
    workdir: str | Path = ".",
) -> VerificationReport:
    failed = any(str(item.get("status")) == STATUS_FAILED for item in scenario_results)
    incomplete = any(
        str(item.get("status")) == STATUS_INCOMPLETE and not bool(item.get("expected_future_gap"))
        for item in scenario_results
    )
    limitations = list(dict.fromkeys(str(item) for item in (limitations or []) if str(item).strip()))
    differential = dict(baseline_differential or {})
    required_test_failure = (
        int(differential.get("new_regressions") or 0) > 0
        if differential else bool((tests or {}).get("failed"))
    )
    required_test_missing = bool((tests or {}).get("required_layer_missing"))
    if failed or required_test_failure:
        status = STATUS_FAILED
    elif incomplete or required_test_missing:
        status = STATUS_INCOMPLETE
    else:
        status = STATUS_VERIFIED
    repo = _repository_identity(workdir)
    first = scenario_results[0] if scenario_results else {}
    is_phase1 = any(
        bool((item.get("feature_flags") or {}).get("conversation_continuity_v2"))
        or str(item.get("scenario_id") or "").startswith("phase1_")
        for item in scenario_results
    )
    phase_result = (
        f"PHASE 1 {status.replace('_', ' ')}" if is_phase1 else status
    )
    return VerificationReport(
        overall_status=status,
        phase_result=phase_result,
        repository=repo,
        environment={
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "scenario_schema_versions": sorted({item.get("scenario_schema_version") for item in scenario_results if item.get("scenario_schema_version") is not None}),
            "deterministic_seeds": sorted({item.get("seed") for item in scenario_results if item.get("seed") is not None}),
            "feature_flags": first.get("feature_flags") or {},
        },
        commands=[asdict(item) for item in commands or []],
        tests=dict(tests or {}),
        scenarios=scenario_results,
        external_boundaries={
            "twitch": "fake",
            "tts_audio": "fake",
            "desktop": "fake",
            "game_research_web": "fixture",
            "llm_model": "fixture",
            "network": "blocked_by_design",
        },
        persistence={
            "database_type": "isolated_sqlite",
            "database_paths": [item.get("database", {}).get("path") for item in scenario_results],
            "restart_points": sum(int(item.get("restart_count") or 0) for item in scenario_results),
            "schema_migrations": [item.get("database", {}).get("schema_migrations", []) for item in scenario_results],
        },
        limitations=limitations,
        baseline_differential=differential,
    )


def write_report(report: VerificationReport, output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "verification-report.json"
    markdown_path = directory / "verification-report.md"
    data = report.to_dict()
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(data), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Cognitive Replay Verification Report",
        "",
        f"Overall status: **{data.get('overall_status')}**",
        f"Phase result: **{data.get('phase_result', data.get('overall_status'))}**",
        "",
        "## Repository and environment",
        "",
        f"- Commit: `{data.get('repository', {}).get('commit', 'unknown')}`",
        f"- Working tree: `{data.get('repository', {}).get('working_tree', 'unknown')}`",
        f"- Platform: `{data.get('environment', {}).get('platform', 'unknown')}`",
        f"- Python: `{data.get('environment', {}).get('python', 'unknown')}`",
        "",
        "## Commands",
        "",
    ]
    commands = data.get("commands") or []
    if commands:
        for item in commands:
            lines.append(f"- `{item.get('command')}` → exit {item.get('exit_code')} ({item.get('duration_seconds')}s)")
    else:
        lines.append("- No external test commands were attached to this scenario run.")
    lines.extend(["", "## Tests", "", f"```json\n{json.dumps(data.get('tests') or {}, ensure_ascii=False, indent=2)}\n```", "", "## Replay scenarios", ""])
    for scenario in data.get("scenarios") or []:
        lines.extend([
            f"### {scenario.get('scenario_id')}",
            "",
            f"- Status: **{scenario.get('status')}**",
            f"- Events: {scenario.get('events_processed', 0)}",
            f"- Restarts: {scenario.get('restart_count', 0)}",
            f"- Duration: {scenario.get('duration_seconds', 0)}s",
            f"- Assertions passed/failed/skipped: {scenario.get('assertion_summary', {}).get('passed', 0)}/{scenario.get('assertion_summary', {}).get('failed', 0)}/{scenario.get('assertion_summary', {}).get('skipped', 0)}",
            "",
        ])
        for failure in scenario.get("failures") or []:
            lines.append(f"- Failure: event `{failure.get('event_id', 'final')}`, path `{failure.get('path', '')}`, reason `{failure.get('reason', '')}`")
        final_state = scenario.get("final_state") or {}
        lines.extend([
            "",
            "#### Checkpoint state",
            "",
            f"```json\n{json.dumps(scenario.get('checkpoint_states') or {}, ensure_ascii=False, indent=2)}\n```",
            "",
            "#### Final state and side effects",
            "",
            f"```json\n{json.dumps(final_state, ensure_ascii=False, indent=2)}\n```",
            "",
            "#### Restart evidence",
            "",
            f"```json\n{json.dumps(scenario.get('restart_evidence') or [], ensure_ascii=False, indent=2)}\n```",
            "",
        ])
    lines.extend(["## External boundaries", ""])
    for key, value in (data.get("external_boundaries") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Persistence", "", f"```json\n{json.dumps(data.get('persistence') or {}, ensure_ascii=False, indent=2)}\n```", "", "## Limitations", ""])
    if data.get("limitations"):
        lines.extend(f"- {item}" for item in data["limitations"])
    else:
        lines.append("- None reported.")
    lines.extend(["", "## Baseline differential", ""])
    differential = data.get("baseline_differential") or {}
    if differential:
        lines.append(f"```json\n{json.dumps(differential, ensure_ascii=False, indent=2)}\n```")
    else:
        lines.append("- No baseline differential was attached.")
    lines.extend([
        "",
        "## Human evaluation boundary",
        "",
        "This harness verifies cognitive/state prerequisites. Naturalness, personality, comedic timing, and social appropriateness still require human judgment.",
        "",
    ])
    return "\n".join(lines)


def sanitize(value: Any, *, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        return _redacted(value)
    if isinstance(value, dict):
        return {str(k): sanitize(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(item, key=key) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item, key=key) for item in value]
    text = str(value) if isinstance(value, str) else ""
    if text and re.search(r"(?i)(bearer\s+[a-z0-9._-]+|sk-[a-z0-9_-]{12,})", text):
        return _redacted(text)
    return value


def _redacted(value: Any) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
    return f"<redacted:{digest}>"


def _repository_identity(workdir: str | Path) -> dict[str, str]:
    root = str(Path(workdir).resolve())
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, timeout=5, check=True).stdout.strip()
        porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, timeout=5, check=True).stdout
        dirty = hashlib.sha256(porcelain.encode()).hexdigest()[:12] if porcelain else "clean"
        return {"commit": commit, "working_tree": dirty}
    except Exception:
        return {"commit": "unknown", "working_tree": "unknown"}
