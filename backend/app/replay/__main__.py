from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from app.replay.cognitive import CognitiveReplayRunner
from app.replay.report import (
    CommandVerification,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    build_report,
    write_report,
)
from app.replay.scenario import CognitiveReplayScenario


PHASE_05_TEST_MODULES = (
    "backend.tests.test_cognitive_replay",
    "backend.tests.test_voice_command_pipeline",
    "backend.tests.test_cognitive_twitch",
    "backend.tests.test_stream_presence",
    "backend.tests.test_hebe_live_v1",
    "backend.tests.test_hebe_live_v11",
    "backend.tests.test_hebe_live_v12",
    "backend.tests.test_hebe_live_v12_followup",
    "backend.tests.test_hebe_live_20260809_followup",
    "backend.tests.test_final_emission_gate",
    "backend.tests.test_cognitive_execution_guard",
    "backend.tests.test_game_knowledge",
    "backend.tests.test_stream_session_primer",
    "backend.tests.test_live_session_brain",
)
PHASE_1_TEST_MODULES = (*PHASE_05_TEST_MODULES, "backend.tests.test_conversation_continuity_phase1")


def _default_scenario_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "cognitive_replay"


def _resolve_scenarios(values: list[str], suite: str) -> list[Path]:
    directory = _default_scenario_dir()
    if suite:
        if suite == "cognitive-v2-phase1":
            phase1 = directory.parent / "cognitive_replay_phase1"
            return sorted(directory.glob("*.json")) + sorted(phase1.glob("*.json"))
        if suite != "cognitive-v2":
            raise ValueError(f"unknown suite: {suite}")
        return sorted(directory.glob("*.json"))
    resolved: list[Path] = []
    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            resolved.extend(sorted(candidate.glob("*.json")))
            continue
        if not candidate.exists():
            named = directory / (value if value.endswith(".json") else f"{value}.json")
            candidate = named
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        resolved.append(candidate.resolve())
    if not resolved:
        raise ValueError("provide --scenario or --suite cognitive-v2")
    return resolved


def _run_phase_tests(workdir: Path, *, phase1: bool = False) -> tuple[CommandVerification, dict[str, object]]:
    command = [sys.executable, "-m", "unittest", *(PHASE_1_TEST_MODULES if phase1 else PHASE_05_TEST_MODULES)]
    started = time.perf_counter()
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(workdir / "backend") + (__import__("os").pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(command, cwd=workdir, env=env, text=True, capture_output=True, check=False)
    duration = round(time.perf_counter() - started, 6)
    output = "\n".join((completed.stdout, completed.stderr))
    ran = re.search(r"Ran\s+(\d+)\s+tests?", output)
    skipped = re.search(r"skipped=(\d+)", output)
    failures = len(re.findall(r"^(?:FAIL|ERROR):", output, re.MULTILINE))
    failing_tests = sorted(set(re.findall(r"^(?:FAIL|ERROR):\s+([^\r\n]+)", output, re.MULTILINE)))
    total = int(ran.group(1)) if ran else 0
    skipped_count = int(skipped.group(1)) if skipped else 0
    if completed.returncode and failures == 0:
        failures = 1
    summary: dict[str, object] = {
        "passed": max(0, total - failures - skipped_count),
        "failed": failures,
        "skipped": skipped_count,
        "total": total,
        "duration_seconds": duration,
        "expected_failures": 0,
        "failing_tests": failing_tests,
        "required_layer_missing": False,
        "output_digest": hashlib.sha256(output.encode("utf-8")).hexdigest()[:16],
    }
    return CommandVerification(" ".join(command), completed.returncode, duration), summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Hebe Cognitive Replay scenarios")
    parser.add_argument("--scenario", action="append", default=[], help="scenario JSON, directory, or fixture name")
    parser.add_argument("--suite", default="", help="named scenario suite (cognitive-v2 or cognitive-v2-phase1)")
    parser.add_argument("--output", default="artifacts/cognitive-replay/latest", help="verification artifact directory")
    parser.add_argument(
        "--run-phase-tests",
        action="store_true",
        help="run the Phase 0.5 unit/integration/regression contract and include it in the report",
    )
    parser.add_argument(
        "--baseline-differential",
        default="",
        help="machine-generated baseline/current regression differential JSON",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    output = Path(args.output).resolve()
    workspace = output / "workspaces"
    paths = _resolve_scenarios(args.scenario, args.suite)
    results = []
    limitations: list[str] = []
    for path in paths:
        scenario = CognitiveReplayScenario.load(path)
        runner = CognitiveReplayRunner(workspace_root=workspace, retain_workspace=True)
        result = runner.run(scenario)
        results.append(result.to_dict())
        limitations.extend(result.limitations)
    failed = sum(1 for item in results if item["status"] == STATUS_FAILED)
    all_incomplete = sum(1 for item in results if item["status"] == STATUS_INCOMPLETE)
    incomplete = sum(
        1 for item in results
        if item["status"] == STATUS_INCOMPLETE and not item.get("expected_future_gap")
    )
    commands: list[CommandVerification] = []
    if args.run_phase_tests:
        test_command, test_summary = _run_phase_tests(repo_root, phase1=args.suite == "cognitive-v2-phase1")
        commands.append(test_command)
    else:
        test_summary = {"passed": 0, "failed": 0, "skipped": 0, "total": 0, "required_layer_missing": True}
    differential = {}
    if args.baseline_differential:
        differential = json.loads(Path(args.baseline_differential).resolve().read_text(encoding="utf-8"))
        differential_current_failures = differential.get(
            "phase_1_tests_failed", differential.get("phase_0_5_tests_failed", -1)
        )
        if int(differential_current_failures) != int(test_summary["failed"]):
            raise ValueError("baseline differential does not match the current regression failure count")
    regression_failed = int(differential.get("new_regressions") or 0) > 0 if differential else bool(test_summary["failed"])
    exit_code = 1 if failed or regression_failed else 2 if incomplete or test_summary["required_layer_missing"] else 0
    command = "python -m app.replay " + " ".join(sys.argv[1:] if argv is None else argv)
    commands.append(CommandVerification(command, exit_code, round(time.perf_counter() - started, 6)))
    report = build_report(
        scenario_results=results,
        commands=commands,
        tests={
            "unit_integration_regression": test_summary,
            "replay": {
                "passed": len(results) - failed - all_incomplete,
                "failed": failed,
                "skipped": all_incomplete,
                "expected_future_gaps": all_incomplete - incomplete,
                "expected_failures": 0,
                "duration_seconds": round(sum(float(item.get("duration_seconds") or 0.0) for item in results), 6),
            },
            "failed": int(test_summary["failed"]),
            "required_layer_missing": bool(test_summary["required_layer_missing"]),
        },
        limitations=limitations,
        baseline_differential=differential,
        workdir=repo_root,
    )
    json_path, markdown_path = write_report(report, output)
    print(f"Cognitive Replay: {report.overall_status}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
