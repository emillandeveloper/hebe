from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.cognitive.capabilities.goal import Goal
from app.cognitive.capabilities.models import Capability
from app.cognitive.capabilities.registry import CapabilityRegistry


_RISK_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "danger": 4}
_CONTROL_ACTION_CAPABILITIES = {
    "open_application": {"pc.open_application"},
    "stop_speaking": {"audio.stop_speaking"},
    "full_dev_reset": {"dev.full_dev_reset"},
}


@dataclass(slots=True)
class CapabilityMatchResult:
    goal: Goal
    selected_capabilities: list[Capability] = field(default_factory=list)
    rejected_capabilities: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    missing_slots: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    output_policy: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal.to_dict(),
            "selected_capabilities": [capability.to_dict() for capability in self.selected_capabilities],
            "rejected_capabilities": self.rejected_capabilities,
            "confidence": self.confidence,
            "missing_slots": self.missing_slots,
            "requires_confirmation": self.requires_confirmation,
            "output_policy": self.output_policy,
            "risk_level": self.risk_level,
        }


class CapabilityMatcher:
    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry

    def match(self, goal: Goal, current_mode: str = "private") -> CapabilityMatchResult:
        candidates = self.registry.find_capabilities_for_goal(goal.goal_type)
        selected: list[Capability] = []
        rejected: list[dict[str, Any]] = []

        for capability in candidates:
            action_reason = self._action_mismatch_reason(capability, goal)
            if action_reason:
                rejected.append({
                    "capability_id": capability.id,
                    "status": capability.status,
                    "enabled": capability.enabled,
                    "reasons": [action_reason],
                })
                continue
            available = self.registry.check_capability_available(capability.id, current_mode=current_mode)
            input_validation = self.registry.validate_capability_inputs(capability.id, goal.slots)
            if available["available"] and input_validation["ok"]:
                selected.append(capability)
                continue
            reasons = list(available["reasons"])
            if not input_validation["ok"]:
                reasons.append(f"missing_inputs={','.join(input_validation['missing'])}")
            rejected.append({
                "capability_id": capability.id,
                "status": capability.status,
                "enabled": capability.enabled,
                "reasons": reasons,
            })

        requires_confirmation = goal.requires_confirmation or any(
            capability.requires_confirmation for capability in selected
        )
        risk_level = self._highest_risk([goal.risk_level] + [capability.risk_level for capability in selected])
        output_policy = self._merge_output_policy(selected)
        result = CapabilityMatchResult(
            goal=goal,
            selected_capabilities=selected,
            rejected_capabilities=rejected,
            confidence=goal.confidence if selected or rejected else 0.0,
            missing_slots=list(goal.missing_slots),
            requires_confirmation=requires_confirmation,
            output_policy=output_policy,
            risk_level=risk_level,
        )
        print(
            "[HEBE][CAPABILITY] "
            f"selected={[capability.id for capability in selected]!r} "
            f"rejected={[item['capability_id'] for item in rejected]!r}",
            flush=True,
        )
        return result

    def _action_mismatch_reason(self, capability: Capability, goal: Goal) -> str | None:
        if goal.goal_type != "control_pc":
            return None
        action = str(goal.slots.get("action") or "")
        if not action:
            return None
        allowed = _CONTROL_ACTION_CAPABILITIES.get(action)
        if not allowed:
            return None
        if capability.id in allowed:
            return None
        return f"action_mismatch={action}"

    def _merge_output_policy(self, capabilities: list[Capability]) -> dict[str, Any]:
        output_policy: dict[str, Any] = {}
        for capability in capabilities:
            output_policy.update(capability.output_policy or {})
        return output_policy

    def _highest_risk(self, risk_levels: list[str]) -> str:
        return max(
            (risk for risk in risk_levels if risk),
            key=lambda risk: _RISK_RANK.get(risk, 1),
            default="low",
        )
