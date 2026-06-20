from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CognitiveDecision:
    """The single routing contract consumed by downstream cognitive services."""

    message_id: str
    source: str
    authority: str
    addressed_to_hebe: bool
    raw_text: str
    normalized_text: str
    intent: str
    intent_confidence: float
    is_new_request: bool
    uses_pending_task: bool
    active_pending_task: dict[str, Any] | None = None
    pending_task_kind: str | None = None
    pending_resolution_allowed: bool = False
    pending_compatible: bool = False
    pending_reason: str = "no_active_pending"
    goal_type: str = "answer_question"
    allowed_capabilities: list[str] = field(default_factory=list)
    blocked_capabilities: list[str] = field(default_factory=list)
    allowed_step_types: list[str] = field(default_factory=list)
    blocked_step_types: list[str] = field(default_factory=list)
    should_reply: bool = True
    should_stop_pipeline: bool = False
    response_mode: str = "chat"
    response_intent: str = "answer_user"
    reason: str = "fallback_chat"
    personal_state: str | None = None
    action_permission_summary: dict[str, Any] = field(default_factory=dict)
    debug_trace: list[str] = field(default_factory=list)

    def allows_capability(self, capability_id: str) -> bool:
        if capability_id in self.blocked_capabilities:
            return False
        return capability_id in self.allowed_capabilities

    @property
    def input_text(self) -> str:
        return self.raw_text

    @property
    def pending_task_id(self) -> str | None:
        pending = self.active_pending_task or {}
        return str(pending.get("id") or pending.get("task_id") or "") or None

    @property
    def required_capability_ids(self) -> list[str]:
        return self.allowed_capabilities

    @property
    def blocked_capability_ids(self) -> list[str]:
        return self.blocked_capabilities

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
