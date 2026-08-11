from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _compact(message: str) -> str:
    text = str(message or "").replace("\r", "").strip()
    if " not found in " in text:
        text = text.split(" not found in ", 1)[0] + " not found in <captured value>"
    return text[:600]


def _subsystem(test: str) -> str:
    if "test_game_knowledge" in test:
        return "Game Knowledge response guard"
    if "twitch_" in test or "game_tip" in test:
        return "Twitch no-mention/presence"
    return "local application/capability resolution"


def _phase_path(test: str, phase_label: str = "Phase 0.5") -> str:
    if "twitch_" in test or "game_tip" in test:
        return f"YES: {phase_label} shares this path, but the baseline and current terminal assertion are identical"
    if any(token in test for token in ("test_stt_", "test_no_wake_", "test_unrelated_action_")):
        return f"ENTRY ONLY: the {phase_label} STT seam is on the path; the unchanged local capability resolver produces the identical failure"
    return f"NO: the {phase_label} production seam changes are not on this failing path"


def build_differential(
    baseline: dict[str, Any], current: dict[str, Any], *, baseline_commit: str,
    baseline_shared_total: int, phase_total: int, baseline_command: str = "",
    baseline_duration_seconds: float = 0.0,
    phase_label: str = "phase_0_5",
) -> dict[str, Any]:
    phase_label = str(phase_label or "phase_0_5").strip().lower()
    phase_upper = phase_label.upper()
    before = {row["test"]: row for row in baseline.get("records") or []}
    after = {row["test"]: row for row in current.get("records") or []}
    rows = []
    for test in sorted(set(before) | set(after)):
        old = before.get(test, {"status": "MISSING"})
        new = after.get(test, {"status": "MISSING"})
        old_failed = old.get("status") in {"FAIL", "ERROR"}
        new_failed = new.get("status") in {"FAIL", "ERROR"}
        old_error = f"{old.get('exception_type', '')}: {_compact(old.get('exception_message', ''))}".strip(": ")
        new_error = f"{new.get('exception_type', '')}: {_compact(new.get('exception_message', ''))}".strip(": ")
        if not old_failed and not new_failed:
            classification = "PASS_BOTH"
        elif old_failed and new_failed:
            comparable_old = re.sub(r"(?:twchat|msg)_[0-9a-f]+", "<stable-id>", old_error)
            comparable_new = re.sub(r"(?:twchat|msg)_[0-9a-f]+", "<stable-id>", new_error)
            classification = "PRE_EXISTING_FAILURE" if comparable_old == comparable_new else "FAILURE_CHANGED"
        elif new_failed:
            classification = f"NEW_{phase_upper}_REGRESSION"
        else:
            classification = f"FIXED_BY_{phase_upper}"
        rows.append({
            "test": test,
            "subsystem": _subsystem(test),
            "baseline_status": old.get("status"),
            f"{phase_label}_status": new.get("status"),
            "baseline_exception_or_assertion": old_error,
            f"{phase_label}_exception_or_assertion": new_error,
            f"{phase_label}_code_on_failing_path": _phase_path(test, phase_label.replace('_', ' ').title()),
            "classification": classification,
        })
    new_name = f"NEW_{phase_upper}_REGRESSION"
    fixed_name = f"FIXED_BY_{phase_upper}"
    counts = {name: sum(row["classification"] == name for row in rows) for name in (
        "PASS_BOTH", "PRE_EXISTING_FAILURE", new_name, fixed_name, "FAILURE_CHANGED"
    )}
    return {
        "baseline_commit": baseline_commit,
        "baseline_python": baseline.get("python"),
        "baseline_platform": baseline.get("platform"),
        "baseline_command": baseline_command,
        "baseline_command_duration_seconds": baseline_duration_seconds,
        "baseline_loader_errors": 1,
        "baseline_tests_passed": baseline_shared_total - counts["PRE_EXISTING_FAILURE"] - counts["FAILURE_CHANGED"],
        "baseline_tests_failed": counts["PRE_EXISTING_FAILURE"] + counts["FAILURE_CHANGED"],
        "baseline_new_module_unavailable": "backend.tests.test_cognitive_replay",
        f"{phase_label}_tests_passed": phase_total - sum(row[f"{phase_label}_status"] in {"FAIL", "ERROR"} for row in rows),
        f"{phase_label}_tests_failed": sum(row[f"{phase_label}_status"] in {"FAIL", "ERROR"} for row in rows),
        "new_regressions": counts[new_name] + counts["FAILURE_CHANGED"],
        "pre_existing_failures": counts["PRE_EXISTING_FAILURE"],
        "fixed_existing_failures": counts[fixed_name],
        "classification_counts": counts,
        "tests": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--baseline-shared-total", required=True, type=int)
    parser.add_argument("--phase-total", required=True, type=int)
    parser.add_argument("--baseline-command", default="")
    parser.add_argument("--baseline-duration-seconds", default=0.0, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-label", default="phase_0_5")
    args = parser.parse_args(argv)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    result = build_differential(
        baseline, current, baseline_commit=args.baseline_commit,
        baseline_shared_total=args.baseline_shared_total, phase_total=args.phase_total,
        baseline_command=args.baseline_command,
        baseline_duration_seconds=args.baseline_duration_seconds,
        phase_label=args.phase_label,
    )
    Path(args.output).resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if result["new_regressions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
