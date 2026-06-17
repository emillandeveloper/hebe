# backend/app/cognitive/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any


# =========================
# Core cognitive models
# =========================

@dataclass(slots=True)
class PlanStep:
    """
    Paso individual de un plan cognitivo.
    type: memory | reminder | action | reply | ask | suggest | tool | query | state_update | diagnostic | noop
    """
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    capability_id: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    result_key: Optional[str] = None
    risk_level: str = "low"
    requires_confirmation: bool = False

@dataclass(slots=True)
class Plan:
    """
    Plan deliberado por Hebe.
    """
    steps: list[PlanStep] = field(default_factory=list)
    reasoning: Optional[str] = None
    goal: Optional[dict[str, Any]] = None
    selected_capabilities: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    output_policy: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    reasoning_summary: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: Optional[str] = None


@dataclass(slots=True)
class DeliberationResult:
    """
    Salida estructurada del deliberation service.
    """
    plan: Plan


@dataclass(slots=True)
class StepExecutionResult:
    """
    Resultado de ejecutar un step del plan.
    """
    step_type: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass(slots=True)
class ExecutionResult:
    """
    Resultado agregado de la ejecución de un plan.
    """
    results: list[StepExecutionResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.success for r in self.results)

    def first_result_of_type(self, step_type: str) -> Optional[StepExecutionResult]:
        for result in self.results:
            if result.step_type == step_type:
                return result
        return None
    
# =========================
# Actions (core of agency)
# =========================

@dataclass(slots=True)
class ActionStep:
    """
    Acción que Hebe puede ejecutar en el mundo real.
    """
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionResult:
    """
    Resultado de ejecutar una acción.
    """
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
