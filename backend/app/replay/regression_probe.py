"""Stdlib-only unittest result probe used for baseline differential evidence."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
import time
import traceback
import unittest
from pathlib import Path
from typing import Any


DEFAULT_FAILURE_METHODS = (
    "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_no_wake_whitelisted_app_command_routes_while_stream_offline",
    "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_hebe_abre_obs_uses_same_open_application_pipeline",
    "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_ui_abre_obs_creates_open_application_when_awake_and_whitelisted",
    "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_ui_hebe_abre_obs_creates_open_application_action_plan",
    "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_unrelated_action_during_pending_conversation_still_uses_action_flow",
    "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_high_value_game_tip_can_reply_without_hebe_mention",
    "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_obs_path_missing_returns_structured_action_result_not_generic_advice",
    "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_stt_canonical_melonds_command_executes_once",
    "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_twitch_normal_no_mention_chat_reaches_presence_observe",
    "backend.tests.test_voice_command_pipeline.VoiceCommandPipelineTests.test_twitch_pipeline_health_counts_messages",
    "backend.tests.test_game_knowledge.GameKnowledgeTests.test_response_synthesizer_handles_game_knowledge_command_result",
)


class DifferentialResult(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def _record(self, test: Any, status: str, err: tuple[type[BaseException], BaseException, Any] | None = None) -> None:
        record: dict[str, Any] = {"test": test.id(), "status": status}
        if err is not None:
            record.update({
                "exception_type": err[0].__name__,
                "exception_message": str(err[1]),
                "traceback_tail": traceback.format_exception(*err)[-1].strip(),
            })
        self.records.append(record)

    def addSuccess(self, test: Any) -> None:
        super().addSuccess(test)
        self._record(test, "PASS")

    def addFailure(self, test: Any, err: tuple[type[BaseException], BaseException, Any]) -> None:
        super().addFailure(test, err)
        self._record(test, "FAIL", err)

    def addError(self, test: Any, err: tuple[type[BaseException], BaseException, Any]) -> None:
        super().addError(test, err)
        self._record(test, "ERROR", err)

    def addSkip(self, test: Any, reason: str) -> None:
        super().addSkip(test, reason)
        self.records.append({"test": test.id(), "status": "SKIP", "reason": reason})

    def addSubTest(self, test: Any, subtest: Any, err: tuple[type[BaseException], BaseException, Any] | None) -> None:
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._record(subtest, "FAIL" if issubclass(err[0], test.failureException) else "ERROR", err)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.source_root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "backend"))
    os.environ["PYTHONPATH"] = str(root / "backend")
    os.chdir(root)
    suite = unittest.defaultTestLoader.loadTestsFromNames(DEFAULT_FAILURE_METHODS)
    result = DifferentialResult()
    captured = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        suite.run(result)
    data = {
        "source_root": str(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "duration_seconds": round(time.perf_counter() - started, 6),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "records": result.records,
    }
    Path(args.output).resolve().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if result.failures or result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
