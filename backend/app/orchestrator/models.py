# backend/app/orchestrator/models.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# =========================
# Enums
# =========================

class InputSource(str, Enum):
    VOICE = "voice"
    TEXT = "text"
    SYSTEM = "system"


class DecisionKind(str, Enum):
    IGNORE = "ignore"
    CHAT = "chat"
    TOOL = "tool"
    CLARIFY = "clarify"
    CONFIRM = "confirm"


class PendingActionType(str, Enum):
    CLARIFY = "clarify"
    CONFIRM = "confirm"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    IGNORED = "ignored"


# =========================
# Input del orquestador
# =========================

@dataclass(slots=True)
class OrchestratorInput:
    text: str
    source: InputSource = InputSource.VOICE
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# =========================
# Resultado de resolución de intent
# =========================

@dataclass(slots=True)
class IntentResult:
    intent: Optional[str] = None
    confidence: float = 0.0
    slots: dict[str, Any] = field(default_factory=dict)
    source: str = "none"  # rules | classifier | llm | none
    raw: Optional[dict[str, Any]] = None

    @property
    def has_intent(self) -> bool:
        return bool(self.intent)


# =========================
# Acción pendiente (clarify / confirm)
# =========================

@dataclass(slots=True)
class PendingAction:
    type: PendingActionType
    intent: str
    prompt: str

    # Para tools
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)

    # Para completar información
    known_slots: dict[str, Any] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)

    # Metadata libre
    metadata: dict[str, Any] = field(default_factory=dict)


# =========================
# Decisión del policy
# =========================

@dataclass(slots=True)
class Decision:
    kind: DecisionKind

    # Contexto lógico
    intent: Optional[str] = None
    reason: Optional[str] = None

    # Texto que Hebe podría decir
    response: Optional[str] = None

    # Tool execution
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)

    # Clarify / confirm
    missing_slots: list[str] = field(default_factory=list)
    requires_confirmation: bool = False

    # Debug / trazabilidad
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.kind in {
            DecisionKind.IGNORE,
            DecisionKind.CHAT,
            DecisionKind.TOOL,
        }

    @property
    def opens_pending_action(self) -> bool:
        return self.kind in {
            DecisionKind.CLARIFY,
            DecisionKind.CONFIRM,
        }


# =========================
# Resultado de ejecución
# =========================

@dataclass(slots=True)
class ExecutionResult:
    status: ExecutionStatus
    success: bool

    output_text: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    # Referencia útil para logging/debug
    decision_kind: Optional[DecisionKind] = None
    intent: Optional[str] = None


# =========================
# Helpers opcionales
# =========================

def make_success(
    output_text: Optional[str] = None,
    *,
    data: Optional[dict[str, Any]] = None,
    decision_kind: Optional[DecisionKind] = None,
    intent: Optional[str] = None,
) -> ExecutionResult:
    return ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        success=True,
        output_text=output_text,
        data=data or {},
        error=None,
        decision_kind=decision_kind,
        intent=intent,
    )


def make_error(
    error: str,
    *,
    output_text: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
    decision_kind: Optional[DecisionKind] = None,
    intent: Optional[str] = None,
) -> ExecutionResult:
    return ExecutionResult(
        status=ExecutionStatus.ERROR,
        success=False,
        output_text=output_text,
        data=data or {},
        error=error,
        decision_kind=decision_kind,
        intent=intent,
    )


def make_ignored(
    output_text: Optional[str] = None,
    *,
    decision_kind: Optional[DecisionKind] = DecisionKind.IGNORE,
    intent: Optional[str] = None,
) -> ExecutionResult:
    return ExecutionResult(
        status=ExecutionStatus.IGNORED,
        success=True,
        output_text=output_text,
        data={},
        error=None,
        decision_kind=decision_kind,
        intent=intent,
    )


def make_cancelled(
    output_text: Optional[str] = None,
    *,
    decision_kind: Optional[DecisionKind] = None,
    intent: Optional[str] = None,
) -> ExecutionResult:
    return ExecutionResult(
        status=ExecutionStatus.CANCELLED,
        success=True,
        output_text=output_text,
        data={},
        error=None,
        decision_kind=decision_kind,
        intent=intent,
    )