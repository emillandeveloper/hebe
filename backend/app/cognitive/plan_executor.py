from __future__ import annotations

from app.cognitive.memory_store import MemoryStore
from app.cognitive.models import (
    Plan,
    PlanStep,
    ExecutionResult,
    StepExecutionResult,
)
from app.cognitive.action_runtime import ActionRuntime


class PlanExecutor:
    """
    Ejecuta los pasos de un plan cognitivo.

    Responsabilidades:
    - ejecutar escrituras en memoria
    - crear reminders
    - ejecutar acciones reales mediante ActionRuntime
    - recopilar resultados estructurados

    Importante:
    - NO sintetiza lenguaje natural final
    - NO decide qué hacer
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        action_runtime: ActionRuntime,
    ):
        self.memory_store = memory_store
        self.action_runtime = action_runtime

    # =========================
    # Entry point
    # =========================

    def execute(self, plan: Plan) -> ExecutionResult:
        results: list[StepExecutionResult] = []
        context: dict = {}

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
        context: dict,
    ) -> StepExecutionResult:
        try:
            if step.type == "memory":
                return self._execute_memory_step(step, context)

            if step.type == "reminder":
                return self._execute_reminder_step(step, context)

            if step.type == "action":
                return self._execute_action_step(step, context)

            if step.type == "reply":
                return StepExecutionResult(
                    step_type="reply",
                    success=True,
                    data=step.data or {},
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
        context: dict,
    ) -> StepExecutionResult:
        data = step.data or {}

        kind = data.get("kind", "fact")
        title = data.get("title") or data.get("subject")
        due_at = data.get("due_at")
        source_text = data.get("source_text")
        confidence = float(data.get("confidence", 1.0))
        payload = data.get("payload") or {}

        # Caso especial v1: appointment
        if kind == "appointment":
            if title and "title" not in payload:
                payload["title"] = title
            if due_at and "due_at" not in payload:
                payload["due_at"] = due_at

            fact = self.memory_store.create_fact(
                kind="appointment",
                subject=title or "Cita",
                payload=payload,
                source_text=source_text,
                confidence=confidence,
                active=True,
            )
        else:
            fact = self.memory_store.create_fact(
                kind=kind,
                subject=title,
                payload=payload or None,
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
        context: dict,
    ) -> StepExecutionResult:
        data = step.data or {}

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
    # Actions
    # =========================

    def _execute_action_step(
        self,
        step: PlanStep,
        context: dict,
    ) -> StepExecutionResult:
        data = step.data or {}

        action_name = data.get("name")
        params = data.get("params") or {}

        if not action_name:
            return StepExecutionResult(
                step_type="action",
                success=False,
                data={},
                error="Falta action name",
            )

        print(
            f"[HEBE][PLAN_EXEC] action step name={action_name!r} params={params!r}",
            flush=True,
        )

        action_result = self.action_runtime.execute(action_name, params)

        return StepExecutionResult(
            step_type="action",
            success=bool(action_result.success),
            data={
                "action_name": action_name,
                "params": params,
                "action_result": action_result,
            },
            error=action_result.error,
        )

    # =========================
    # Context propagation
    # =========================

    def _merge_result_into_context(
        self,
        step: PlanStep,
        result: StepExecutionResult,
        context: dict,
    ) -> None:
        if step.type == "memory":
            context["memory_result"] = result.data

        elif step.type == "reminder":
            context["reminder_result"] = result.data

        elif step.type == "action":
            context["action_result"] = result.data

        elif step.type == "reply":
            context["reply_result"] = result.data