# backend/app/cognitive/plan_executor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

from app.cognitive.memory_store import MemoryStore, MemoryFact, Reminder
from app.cognitive.deliberation_service import Plan, PlanStep


@dataclass(slots=True)
class StepExecutionResult:
    step_type: str
    success: bool
    data: dict
    error: Optional[str] = None


@dataclass(slots=True)
class ExecutionResult:
    results: list[StepExecutionResult]

    @property
    def ok(self) -> bool:
        return all(r.success for r in self.results)

    def first_result_of_type(self, step_type: str) -> Optional[StepExecutionResult]:
        for result in self.results:
            if result.step_type == step_type:
                return result
        return None


class PlanExecutor:
    """
    Ejecuta los pasos de un plan cognitivo.

    Responsabilidades:
    - ejecutar escrituras en memoria
    - crear reminders
    - disparar tools usando la arquitectura actual
    - recopilar resultados estructurados

    Importante:
    - NO sintetiza lenguaje natural final
    - NO decide qué hacer
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        tool_dispatcher: Any | None = None,
        tool_executor: Any | None = None,
    ):
        self.memory_store = memory_store
        self.tool_dispatcher = tool_dispatcher
        self.tool_executor = tool_executor

    # =========================
    # Entry point
    # =========================

    def execute(self, plan: Plan) -> ExecutionResult:
        results: list[StepExecutionResult] = []
        context: dict[str, Any] = {}

        for step in plan.steps:
            result = self._execute_step(step, context)
            results.append(result)

            if result.success:
                self._merge_result_into_context(step, result, context)

        return ExecutionResult(results=results)

    # =========================
    # Step dispatch
    # =========================

    def _execute_step(
        self,
        step: PlanStep,
        context: dict[str, Any],
    ) -> StepExecutionResult:
        try:
            if step.type == "memory":
                return self._execute_memory_step(step, context)

            if step.type == "reminder":
                return self._execute_reminder_step(step, context)

            if step.type == "tool":
                return self._execute_tool_step(step, context)

            if step.type == "reply":
                return StepExecutionResult(
                    step_type="reply",
                    success=True,
                    data=step.data,
                )

            return StepExecutionResult(
                step_type=step.type,
                success=False,
                data={},
                error=f"Tipo de step no soportado: {step.type}",
            )

        except Exception as e:
            return StepExecutionResult(
                step_type=step.type,
                success=False,
                data={},
                error=str(e),
            )

    # =========================
    # Memory
    # =========================

    def _execute_memory_step(
        self,
        step: PlanStep,
        context: dict[str, Any],
    ) -> StepExecutionResult:
        data = step.data

        kind = data.get("kind", "fact")
        title = data.get("title")
        due_at = data.get("due_at")
        source_text = data.get("source_text")
        confidence = float(data.get("confidence", 1.0))
        payload = data.get("payload")

        # Caso especial v1: cita
        if kind == "appointment":
            fact = self.memory_store.create_fact(
                kind="appointment",
                subject=title or "Cita",
                payload=payload or {
                    "title": title or "Cita",
                    "due_at": due_at,
                },
                source_text=source_text,
                confidence=confidence,
                active=True,
            )
        else:
            fact = self.memory_store.create_fact(
                kind=kind,
                subject=title,
                payload=payload,
                source_text=source_text,
                confidence=confidence,
                active=True,
            )

        return StepExecutionResult(
            step_type="memory",
            success=True,
            data={
                "memory_id": fact.id,
                "fact": fact,
            },
        )

    # =========================
    # Reminder
    # =========================

    def _execute_reminder_step(
        self,
        step: PlanStep,
        context: dict[str, Any],
    ) -> StepExecutionResult:
        data = step.data

        memory_result = context.get("memory_result")
        source_memory_id = None
        if memory_result:
            source_memory_id = memory_result.get("memory_id")

        reminder = self.memory_store.create_reminder(
            title=data["title"],
            due_at=data["due_at"],
            message=data.get("message"),
            kind=data.get("kind", "generic"),
            timezone_name=data.get("timezone_name", "Europe/Madrid"),
            source_memory_id=source_memory_id,
            payload=data.get("payload"),
        )

        return StepExecutionResult(
            step_type="reminder",
            success=True,
            data={
                "reminder_id": reminder.id,
                "reminder": reminder,
            },
        )

    # =========================
    # Tools
    # =========================

    def _execute_tool_step(
        self,
        step: PlanStep,
        context: dict[str, Any],
    ) -> StepExecutionResult:
        data = step.data
        action = data.get("action")
        input_text = data.get("input_text", "")

        tool_result = None

        # Variante 1: dispatcher directo
        if self.tool_dispatcher is not None:
            if hasattr(self.tool_dispatcher, "dispatch"):
                tool_result = self.tool_dispatcher.dispatch(
                    action=action,
                    payload=data,
                )
            elif hasattr(self.tool_dispatcher, "handle"):
                tool_result = self.tool_dispatcher.handle(
                    action=action,
                    payload=data,
                )

        # Variante 2: executor directo
        elif self.tool_executor is not None:
            if hasattr(self.tool_executor, "execute"):
                tool_result = self.tool_executor.execute(
                    action=action,
                    payload=data,
                )
            elif hasattr(self.tool_executor, "run"):
                tool_result = self.tool_executor.run(
                    action=action,
                    payload=data,
                )

        # Variante 3: fallback muy simple para open_app usando texto
        if tool_result is None and action == "open_app":
            tool_result = {
                "ok": True,
                "action": "open_app",
                "input_text": input_text,
                "message": f"Intento de abrir app con input: {input_text}",
            }

        if tool_result is None:
            return StepExecutionResult(
                step_type="tool",
                success=False,
                data={},
                error=f"No hay dispatcher/executor compatible para action={action}",
            )

        success = self._tool_result_success(tool_result)

        return StepExecutionResult(
            step_type="tool",
            success=success,
            data={
                "tool_result": tool_result,
                "action": action,
            },
            error=None if success else self._tool_result_error(tool_result),
        )

    # =========================
    # Context propagation
    # =========================

    def _merge_result_into_context(
        self,
        step: PlanStep,
        result: StepExecutionResult,
        context: dict[str, Any],
    ) -> None:
        if step.type == "memory":
            context["memory_result"] = result.data

        elif step.type == "reminder":
            context["reminder_result"] = result.data

        elif step.type == "tool":
            context["tool_result"] = result.data

        elif step.type == "reply":
            context["reply_result"] = result.data

    # =========================
    # Tool result helpers
    # =========================

    def _tool_result_success(self, tool_result: Any) -> bool:
        if isinstance(tool_result, bool):
            return tool_result

        if isinstance(tool_result, dict):
            if "ok" in tool_result:
                return bool(tool_result["ok"])
            if "success" in tool_result:
                return bool(tool_result["success"])
            return True

        return True

    def _tool_result_error(self, tool_result: Any) -> str:
        if isinstance(tool_result, dict):
            if tool_result.get("error"):
                return str(tool_result["error"])
            if tool_result.get("message"):
                return str(tool_result["message"])
        return "Tool execution failed"