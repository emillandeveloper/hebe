from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionPlan:
    action_type: str
    status: str = "rejected"  # complete | needs_confirmation | rejected
    confidence: float = 0.0
    target: str | None = None
    command: str | None = None
    requires_stream: bool = False
    reason: str = ""
    slots: dict[str, Any] = field(default_factory=dict)
    context_checks: dict[str, Any] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "status": self.status,
            "confidence": round(float(self.confidence), 3),
            "target": self.target,
            "command": self.command,
            "requires_stream": self.requires_stream,
            "reason": self.reason,
            "slots": self.slots,
            "context_checks": self.context_checks,
            "missing_slots": self.missing_slots,
            "candidates": self.candidates,
        }
