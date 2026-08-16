from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.stream.behavior_adaptation import AdaptationAction, BehaviorAdaptationService


@dataclass(frozen=True, slots=True)
class BehaviorCalibrationCase:
    name: str
    candidate: str
    topic: str = ""
    mode: str = "proactive"
    stage: str = "candidate"


@dataclass(frozen=True, slots=True)
class CuratedBehaviorReplayCase:
    """A human-curated bridge from one shadow trace to a replay fixture."""

    source_trace_id: str
    name: str
    candidate: str
    topic: str
    mode: str
    expected_decision: str
    calibration_label: str

    def to_fixture_row(self) -> dict[str, str]:
        return {
            "source_trace_id": self.source_trace_id,
            "name": self.name,
            "candidate": self.candidate,
            "topic": self.topic,
            "mode": self.mode,
            "expected_decision": self.expected_decision,
            "calibration_label": self.calibration_label,
        }


class BehaviorTraceReplayCurator:
    """Requires explicit human text and expectation; never converts logs automatically."""

    @staticmethod
    def curate(
        trace: dict[str, Any],
        *,
        name: str,
        candidate: str,
        expected_decision: str,
        calibration_label: str,
    ) -> CuratedBehaviorReplayCase:
        trace_id = str(trace.get("trace_id") or "").strip()
        if not trace_id:
            raise ValueError("source_trace_id_required")
        if not str(candidate or "").strip():
            raise ValueError("curated_candidate_required")
        return CuratedBehaviorReplayCase(
            source_trace_id=trace_id,
            name=str(name or trace_id),
            candidate=str(candidate).strip(),
            topic=str(trace.get("topic") or ""),
            mode="direct_response" if str(trace.get("reason_code") or "") == "direct_required_response" else "proactive",
            expected_decision=str(expected_decision or "").lower(),
            calibration_label=str(calibration_label or "").upper(),
        )


class BehaviorPolicyCalibrationReplay:
    """Deterministic shadow replay; records policy decisions without tuning them."""

    def __init__(self, service: BehaviorAdaptationService) -> None:
        self.service = service

    def run(
        self,
        stream: Any,
        cases: Iterable[BehaviorCalibrationCase],
        *,
        now: float,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for case in cases:
            if case.stage == "generated_output":
                decision = self.service.validate_generated_output(
                    stream, case.candidate, topic=case.topic, mode=case.mode, now=now,
                )
            else:
                decision = self.service.evaluate_candidate(
                    stream, case.candidate, topic=case.topic, mode=case.mode, now=now,
                )
            rows.append({
                "case": case.name,
                "candidate": case.candidate,
                "topic": case.topic,
                "stage": case.stage,
                "motif": decision.motif_id,
                "uses": decision.recent_uses,
                "fatigue": decision.fatigue,
                "negative_weight": decision.negative_weight,
                "positive_weight": decision.positive_weight,
                "constraint": decision.constraint_id,
                "decision": decision.action.value,
                "final_emission": decision.action in {AdaptationAction.ALLOW, AdaptationAction.DOWNRANK},
                "reason": decision.reason,
            })
        return rows


__all__ = [
    "BehaviorCalibrationCase",
    "BehaviorPolicyCalibrationReplay",
    "BehaviorTraceReplayCurator",
    "CuratedBehaviorReplayCase",
]
