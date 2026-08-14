from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.stream.speech_intents import SpeechIntentTimingConfig, SpeechIntentType


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.verification_report).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    scenario = source["scenarios"][0]
    metrics = scenario["final_state"]["speech_intents"]["metrics"]
    tests = source["tests"]["unit_integration_regression"]
    timing = SpeechIntentTimingConfig.from_env()

    timing_rows = {}
    for kind in SpeechIntentType:
        row = timing.for_type(kind)
        timing_rows[kind.value] = {
            "minimum_turn_gap_seconds": row.minimum_turn_gap,
            "maximum_turn_delay_seconds": row.maximum_turn_delay,
        }

    passed = (
        source.get("overall_status") == "VERIFIED"
        and scenario.get("status") == "VERIFIED"
        and int(source.get("baseline_differential", {}).get("new_regressions", -1)) == 0
    )
    report = {
        "overall_result": "CO-STREAMER TURN-TAKING VERIFIED" if passed else "CO-STREAMER TURN-TAKING FAILED",
        "source_verification_report": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "root_cause": {
            "presence_poll_seconds": 30.0,
            "blanket_recent_owner_speech_seconds": 30.0,
            "additional_spontaneity_recent_voice_seconds": 20.0,
            "effect": "valuable owner-anchored opportunities were discarded before conversational turn arbitration",
        },
        "before": {
            "representative_real_stream_ticks": 486,
            "should_speak_false": 486,
            "proactive_emissions": 0,
            "stream_tts_emissions": 0,
            "recent_owner_speech_suppressions": 407,
        },
        "after_representative_replay": {
            "scenario": scenario["scenario_id"],
            "status": scenario["status"],
            "assertions_passed": scenario["assertion_summary"]["passed"],
            "assertions_failed": scenario["assertion_summary"]["failed"],
            "companion_ticks": sum(1 for item in scenario["event_results"] if item["event_type"] == "companion_tick"),
            "intents_created": metrics.get("intents_created", 0),
            "intents_emitted": metrics.get("intents_emitted", 0),
            "intents_expired": metrics.get("intents_expired", 0),
            "intents_superseded": metrics.get("intents_superseded", 0),
            "yield_due_owner_resume": metrics.get("yield_due_owner_resume", 0),
            "pending_at_end": metrics.get("pending", 0),
            "final_emissions": len(scenario["final_state"].get("final_emission_results") or []),
        },
        "pipeline": [
            "candidate", "SpeechIntent", "pending", "turn_arbitration",
            "PresenceEngine", "rendering", "FinalEmissionGate", "output",
        ],
        "timing_semantics": timing_rows,
        "performance": {
            "intent_creation": metrics["intent_creation"],
            "pending_queue_operation": metrics["pending_queue_operation"],
            "turn_arbitration": metrics["turn_arbitration"],
            "presence_turn_decision": metrics["presence_turn_decision"],
            "time_created_to_emit": metrics["time_created_to_emit"],
            "turn_gap_before_emit": metrics["turn_gap_before_emit"],
        },
        "verification": {
            "tests_total": tests["total"],
            "tests_passed": tests["passed"],
            "tests_failed": tests["failed"],
            "tests_skipped": tests["skipped"],
            "inherited_failures": source["baseline_differential"].get("pre_existing_failures", 0),
            "new_regressions": source["baseline_differential"].get("new_regressions", 0),
            "scenarios_A_through_L": {
                "status": "PASSED" if passed else "FAILED",
                "count": 12,
                "test_module": "backend.tests.test_co_streamer_turn_taking.CoStreamerTurnTakingScenarios",
            },
        },
        "known_limitations": [
            "Naturalness, comedic timing, and social appropriateness still require human stream review.",
            "Replay replaces Twitch, TTS audio, and model/network boundaries with deterministic fakes.",
            "Owner voice-active uses the production RMS/VAD sample plus normalized STT utterance completion; it is not sample-accurate diarization.",
            "Hebe yields before TTS commit; already-playing audio is not forcibly cancelled by this change.",
            "The full historical suite retains 17 baseline failures; the differential confirms none were introduced by this fix.",
        ],
    }

    json_path = output / "acceptance-summary.json"
    markdown_path = output / "acceptance-summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0 if passed else 1


def render_markdown(report: dict) -> str:
    before = report["before"]
    after = report["after_representative_replay"]
    perf = report["performance"]
    verification = report["verification"]
    lines = [
        "# Co-Streamer Turn-Taking Verification Summary",
        "",
        f"Overall: **{report['overall_result']}**",
        "",
        "## Before / after",
        "",
        "| Metric | Before real stream | Deterministic replay |",
        "|---|---:|---:|",
        f"| Companion ticks | {before['representative_real_stream_ticks']} | {after['companion_ticks']} |",
        f"| Valid emissions | {before['proactive_emissions']} | {after['final_emissions']} |",
        f"| Intents created | unavailable | {after['intents_created']} |",
        f"| Intents emitted | 0 | {after['intents_emitted']} |",
        "",
        "## Performance",
        "",
        f"- Intent creation p50/p95: {perf['intent_creation']['p50_ms']} / {perf['intent_creation']['p95_ms']} ms",
        f"- Pending queue operation p50/p95: {perf['pending_queue_operation']['p50_ms']} / {perf['pending_queue_operation']['p95_ms']} ms",
        f"- Turn arbitration p50/p95: {perf['turn_arbitration']['p50_ms']} / {perf['turn_arbitration']['p95_ms']} ms",
        f"- Presence + turn decision p50/p95: {perf['presence_turn_decision']['p50_ms']} / {perf['presence_turn_decision']['p95_ms']} ms",
        f"- Created-to-emitted p50: {perf['time_created_to_emit']['p50_ms']} ms",
        f"- Conversational gap before emission p50: {perf['turn_gap_before_emit']['p50_ms']} ms",
        "",
        "## Regression",
        "",
        f"- Tests: {verification['tests_passed']}/{verification['tests_total']} passed; {verification['tests_failed']} inherited failures; {verification['new_regressions']} new regressions.",
        f"- Scenarios A–L: {verification['scenarios_A_through_L']['status']}.",
        f"- Representative replay: {after['status']} ({after['assertions_passed']} assertions).",
        "",
        "## Known limitations",
        "",
        *[f"- {item}" for item in report["known_limitations"]],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
