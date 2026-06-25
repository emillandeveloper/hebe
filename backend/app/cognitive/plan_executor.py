from __future__ import annotations

from app.cognitive.memory_store import MemoryStore
from app.cognitive.models import (
    Plan,
    PlanStep,
    ExecutionResult,
    StepExecutionResult,
)
from app.cognitive.action_runtime import ActionRuntime
from app.cognitive.cognitive_router import (
    CAP_APPOINTMENT, CAP_OPEN_APP, CAP_REMINDER, CAP_TWITCH_ACTION,
    CAP_TWITCH_PROMOTION, CAP_TWITCH_REPLY,
)
from app.core.persistent_logs import log_jsonl_event


PASSIVE_STEP_TYPES = {
    "ask",
    "suggest",
    "tool",
    "query",
    "state_update",
    "diagnostic",
    "noop",
}


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
        self.last_guard_results: list[dict] = []

    # =========================
    # Entry point
    # =========================

    def execute(self, plan: Plan) -> ExecutionResult:
        results: list[StepExecutionResult] = []
        context: dict = {}
        self.last_guard_results = []
        decision = (plan.metadata or {}).get("cognitive_decision")

        for step in plan.steps:
            blocked_reason = self._guard_step(step, decision, plan)
            if blocked_reason:
                capability = self._step_capability(step)
                guard = {"step_type": step.type, "capability_id": capability, "allowed": False, "reason": blocked_reason}
                self.last_guard_results.append(guard)
                log_jsonl_event("plan_executor", {
                    "plan_id": plan.message_id or (plan.metadata or {}).get("message_id") or "",
                    "message_id": (decision or {}).get("message_id") if isinstance(decision, dict) else "",
                    "step_type": step.type,
                    "capability_id": capability,
                    "authorized": False,
                    "execution_success": False,
                    "guard_reason": blocked_reason,
                    "error": blocked_reason,
                })
                print(
                    f"[HEBE][PLAN_EXEC_GUARD] blocked step={step.type} capability={capability or 'none'} reason={blocked_reason}",
                    flush=True,
                )
                result = StepExecutionResult(
                    step_type=step.type, success=False,
                    data={"blocked": True, "guard_reason": blocked_reason, "capability_id": capability},
                    error=blocked_reason,
                )
            else:
                capability = self._step_capability(step)
                self.last_guard_results.append({"step_type": step.type, "capability_id": capability, "allowed": True})
                result = self._execute_step(step, context)
                log_jsonl_event("plan_executor", {
                    "plan_id": plan.message_id or (plan.metadata or {}).get("message_id") or "",
                    "message_id": (decision or {}).get("message_id") if isinstance(decision, dict) else "",
                    "step_type": step.type,
                    "capability_id": capability,
                    "authorized": True,
                    "execution_success": bool(result.success),
                    "error": result.error,
                    "guard_reason": "",
                })
            results.append(result)

            if result.success:
                self._merge_result_into_context(step, result, context)

        return ExecutionResult(results=results)

    def _guard_step(self, step: PlanStep, decision: dict | None, plan: Plan) -> str | None:
        if step.type == "noop":
            return None
        if not isinstance(decision, dict):
            return "missing_cognitive_decision"
        if bool(decision.get("should_stop_pipeline")):
            return "pipeline_stopped_by_decision"
        allowed_types = set(decision.get("allowed_step_types") or [])
        blocked_types = set(decision.get("blocked_step_types") or [])
        if step.type in blocked_types or (allowed_types and step.type not in allowed_types):
            return "step_type_not_authorized"
        capability = self._step_capability(step)
        allowed_caps = set(decision.get("allowed_capabilities") or decision.get("required_capability_ids") or [])
        blocked_caps = set(decision.get("blocked_capabilities") or decision.get("blocked_capability_ids") or [])
        if capability and (capability in blocked_caps or capability not in allowed_caps):
            return "capability_not_authorized"
        if step.type in {"action", "memory", "reminder", "tool"} and not capability:
            return "risky_step_missing_capability"
        authority = str(decision.get("authority") or "")
        source = str(decision.get("source") or "")
        if capability in {CAP_OPEN_APP, CAP_REMINDER, CAP_APPOINTMENT, CAP_TWITCH_ACTION} and authority != "owner":
            return "authority_not_authorized"
        if capability in {CAP_OPEN_APP, CAP_REMINDER, CAP_APPOINTMENT, CAP_TWITCH_ACTION} and source not in {
            "ui", "typed_ui", "owner_ui", "voice", "stt_voice", "owner_stt_direct", "owner_stt_command", "owner_stt_followup",
        }:
            return "source_not_authorized"
        if capability == CAP_TWITCH_REPLY and authority not in {"viewer", "system"}:
            return "authority_not_authorized"
        if capability in {CAP_TWITCH_REPLY, CAP_TWITCH_PROMOTION} and not source.startswith("twitch"):
            return "source_not_authorized"
        if capability == CAP_TWITCH_PROMOTION and authority != "system":
            return "authority_not_authorized"
        if capability in {CAP_TWITCH_REPLY, CAP_TWITCH_ACTION, CAP_TWITCH_PROMOTION}:
            permission = decision.get("action_permission_summary") or {}
            if not bool(permission.get("stream_live")):
                return "stream_not_live"
        risk = str(step.risk_level or plan.risk_level or "low").lower()
        if risk in {"high", "critical"} and not (step.requires_confirmation or plan.requires_confirmation):
            return "confirmation_required"
        return None

    @staticmethod
    def _step_capability(step: PlanStep) -> str | None:
        if step.capability_id:
            return step.capability_id
        data = step.data or {}
        if step.type == "reminder":
            return CAP_APPOINTMENT if data.get("kind") == "appointment" else CAP_REMINDER
        if step.type == "memory":
            return CAP_APPOINTMENT if data.get("kind") == "appointment" else None
        if step.type == "action":
            name = str(data.get("name") or "").lower()
            if name in {"open_application", "open_app", "launch_application"}:
                return CAP_OPEN_APP
            if "shoutout" in name or "twitch" in name:
                return CAP_TWITCH_ACTION
        if step.type == "reply":
            mode = str(data.get("mode") or "")
            if mode.startswith("twitch_"):
                return CAP_TWITCH_REPLY
        return None

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
                    data=self._step_data(step),
                )

            if step.type in PASSIVE_STEP_TYPES:
                return StepExecutionResult(
                    step_type=step.type,
                    success=True,
                    data=self._step_data(step),
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

    def _step_data(self, step: PlanStep) -> dict:
        data = dict(step.data or {})
        if step.capability_id:
            data["capability_id"] = step.capability_id
        if step.result_key:
            data["result_key"] = step.result_key
        data["risk_level"] = step.risk_level
        data["requires_confirmation"] = step.requires_confirmation
        return data

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
        safe_message = str(reminder.message or reminder.title or "").replace('"', '\\"')
        print(
            f"[HEBE][REMINDER] created id={reminder.id} "
            f"due_at={reminder.due_at} message=\"{safe_message}\"",
            flush=True,
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
