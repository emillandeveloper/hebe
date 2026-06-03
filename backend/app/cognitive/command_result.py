from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommandResult:
    action_type: str
    success: bool = True
    user_visible_summary: str = ""
    state_changes: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    suggested_tone: str = "hebe_concise"
    fallback_text: str = ""
    requires_model_response: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.fallback_text or self.user_visible_summary

    def lower(self) -> str:
        return str(self).lower()

    def __contains__(self, item: str) -> bool:
        return item in str(self)
