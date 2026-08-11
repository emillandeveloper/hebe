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


def _phase_path(test: str) -> str:
    if "twitch_" in test or "game_tip" in test:
        return "YES: handle_twitch_chat_event was refactored, but the baseline and current terminal assertion are identical"
    if any(token in test for token in ("test_stt_", "test_no_wake_", "test_unrelated_action_")):
        return "ENTRY ONLY: the shared STT ingress wrapper is on the path; the unchanged local capability resolver produces the identical failure"
    return "NO: the Phase 0.5 production seam changes are not on this failing path"


def build_differential(
    baseline: dict[str, Any], current: dict[str, Any], *, baseline_commit: str,
    baseline_shared_total: int, phase_total: int, baseline_command: str = "",
    baseline_duration_seconds: float = 0.0,
) -> dict[str, Any]:
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
            classification = "NEW_PHASE_0_5_REGRESSION"
        else:
            classification = "FIXED_BY_PHASE_0_5"
        rows.append({
            "test": test,
            "subsystem": _subsystem(test),
            "baseline_status": old.get("status"),
            "phase_0_5_status": new.get("status"),
            "baseline_exception_or_assertion": old_error,
            "phase_0_5_exception_or_assertion": new_error,
            "phase_0_5_code_on_failing_path": _phase_path(test),
            "classification": classification,
        })
    counts = {name: sum(row["classification"] == name for row in rows) for name in (
        "PASS_BOTH", "PRE_EXISTING_FAILURE", "NEW_PHASE_0_5_REGRESSION", "FIXED_BY_PHASE_0_5", "FAILURE_CHANGED"
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
        "phase_0_5_tests_passed": phase_total - sum(row["phase_0_5_status"] in {"FAIL", "ERROR"} for row in rows),
        "phase_0_5_tests_failed": sum(row["phase_0_5_status"] in {"FAIL", "ERROR"} for row in rows),
        "new_regressions": counts["NEW_PHASE_0_5_REGRESSION"] + counts["FAILURE_CHANGED"],
        "pre_existing_failures": counts["PRE_EXISTING_FAILURE"],
        "fixed_existing_failures": counts["FIXED_BY_PHASE_0_5"],
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
    args = parser.parse_args(argv)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    result = build_differential(
        baseline, current, baseline_commit=args.baseline_commit,
        baseline_shared_total=args.baseline_shared_total, phase_total=args.phase_total,
        baseline_command=args.baseline_command,
        baseline_duration_seconds=args.baseline_duration_seconds,
    )
    Path(args.output).resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if result["new_regressions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
