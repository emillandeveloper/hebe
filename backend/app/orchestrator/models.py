from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class InputSource(str, Enum):
    VOICE = "voice"
    TEXT = "text"
    SYSTEM = "system"


@dataclass(slots=True)
class OrchestratorInput:
    """Input DTO retained only for the developer intent-evaluation tool."""

    text: str
    source: InputSource = InputSource.VOICE
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IntentResult:
    """Result DTO retained only for the developer intent-evaluation tool."""

    intent: Optional[str] = None
    confidence: float = 0.0
    slots: dict[str, Any] = field(default_factory=dict)
    source: str = "none"
    raw: Optional[dict[str, Any]] = None

    @property
    def has_intent(self) -> bool:
        return bool(self.intent)
