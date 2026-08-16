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


__all__ = ["BehaviorCalibrationCase", "BehaviorPolicyCalibrationReplay"]
