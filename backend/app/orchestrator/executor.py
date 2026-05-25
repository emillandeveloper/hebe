# backend/app/orchestrator/executor.py

from __future__ import annotations

from typing import Any

from .models import (
    Decision,
    DecisionKind,
    ExecutionResult,
    make_error,
    make_ignored,
    make_success,
)


class OrchestratorExecutor:
    def __init__(
        self,
        *,
        chat_runtime: Any | None = None,
        dispatcher: Any | None = None,
    ) -> None:
        self.chat_runtime = chat_runtime
        self.dispatcher = dispatcher

    def execute(self, decision: Decision) -> ExecutionResult:
        try:
            if decision.kind == DecisionKind.IGNORE:
                return self._handle_ignore(decision)

            if decision.kind == DecisionKind.CLARIFY:
                return self._handle_clarify(decision)

            if decision.kind == DecisionKind.CONFIRM:
                return self._handle_confirm(decision)

            if decision.kind == DecisionKind.CHAT:
                return self._handle_chat(decision)

            if decision.kind == DecisionKind.TOOL:
                return self._handle_tool(decision)

            return make_error(
                error=f"Unsupported decision kind: {decision.kind}",
                output_text="No he podido ejecutar esa decisión.",
                decision_kind=decision.kind,
                intent=decision.intent,
            )

        except Exception as exc:
            print(
                f"[HEBE][EXECUTOR] execute failed "
                f"kind={decision.kind} intent={decision.intent} error={exc!r}",
                flush=True,
            )
            return make_error(
                error=str(exc),
                output_text="Ha fallado la ejecución.",
                decision_kind=decision.kind,
                intent=decision.intent,
            )

    def _handle_ignore(self, decision: Decision) -> ExecutionResult:
        return make_ignored(
            output_text=decision.response,
            decision_kind=decision.kind,
            intent=decision.intent,
        )

    def _handle_clarify(self, decision: Decision) -> ExecutionResult:
        return make_success(
            output_text=decision.response,
            data={
                "kind": decision.kind.value,
                "missing_slots": list(decision.missing_slots),
            },
            decision_kind=decision.kind,
            intent=decision.intent,
        )

    def _handle_confirm(self, decision: Decision) -> ExecutionResult:
        return make_success(
            output_text=decision.response,
            data={
                "kind": decision.kind.value,
                "requires_confirmation": decision.requires_confirmation,
            },
            decision_kind=decision.kind,
            intent=decision.intent,
        )

    def _handle_chat(self, decision: Decision) -> ExecutionResult:
        if self.chat_runtime is None:
            return make_error(
                error="chat_runtime is not configured",
                output_text="No tengo disponible el motor de conversación.",
                decision_kind=decision.kind,
                intent=decision.intent,
            )

        response_text = self._call_chat_runtime(decision)

        return make_success(
            output_text=response_text,
            data={"kind": decision.kind.value},
            decision_kind=decision.kind,
            intent=decision.intent,
        )

    def _handle_tool(self, decision: Decision) -> ExecutionResult:
        if self.dispatcher is None:
            return make_error(
                error="dispatcher is not configured",
                output_text="No tengo disponible el sistema de herramientas.",
                decision_kind=decision.kind,
                intent=decision.intent,
            )

        dispatch_result = self._call_dispatcher(decision)

        if isinstance(dispatch_result, ExecutionResult):
            return dispatch_result

        if isinstance(dispatch_result, dict):
            success = bool(dispatch_result.get("success", True))
            output_text = dispatch_result.get("output_text")
            error = dispatch_result.get("error")
            data = dict(dispatch_result.get("data", {}))

            if success:
                return make_success(
                    output_text=output_text,
                    data={"kind": decision.kind.value, **data},
                    decision_kind=decision.kind,
                    intent=decision.intent,
                )

            return make_error(
                error=error or "Tool execution failed",
                output_text=output_text or "No he podido completar esa acción.",
                data={"kind": decision.kind.value, **data},
                decision_kind=decision.kind,
                intent=decision.intent,
            )

        if isinstance(dispatch_result, str):
            return make_success(
                output_text=dispatch_result,
                data={"kind": decision.kind.value},
                decision_kind=decision.kind,
                intent=decision.intent,
            )

        return make_success(
            output_text="Acción ejecutada.",
            data={
                "kind": decision.kind.value,
                "raw_result": dispatch_result,
            },
            decision_kind=decision.kind,
            intent=decision.intent,
        )

    def _call_chat_runtime(self, decision: Decision) -> str:
        user_text = str(decision.metadata.get("user_text", "")).strip()

        if hasattr(self.chat_runtime, "ask_stateless"):
            # Legacy fallback only. Normal UI/private conversation is routed by
            # HebeEngine.cognitive_flow through ContextBuilder ->
            # DeliberationService -> PlanExecutor -> ResponseSynthesizer, so it
            # receives identity, memory retrieval, and memory extraction.
            print(
                "[HEBE][EXECUTOR][LEGACY_CHAT] ask_stateless path used; "
                "normal chat should use cognitive_flow",
                flush=True,
            )
            return self.chat_runtime.ask_stateless(user_text, temperature=0.7)

        if hasattr(self.chat_runtime, "generate"):
            try:
                return self.chat_runtime.generate(decision=decision)
            except TypeError:
                return self.chat_runtime.generate(user_text)

        if hasattr(self.chat_runtime, "chat"):
            return self.chat_runtime.chat(user_text)

        if hasattr(self.chat_runtime, "ask"):
            return self.chat_runtime.ask(user_text)

        if callable(self.chat_runtime):
            return self.chat_runtime(user_text)

        raise RuntimeError("Unsupported chat_runtime interface")

    def _resolve_tool_name(self, decision: Decision) -> str:
        # Si ya viene explícito desde policy/orchestrator, respetarlo.
        if decision.tool_name:
            return decision.tool_name

        intent = str(decision.intent or "").strip()

        stream_tool_map = {
            "stream_chat_message": "twitch_send_message",
            "stream_shoutout": "twitch_shoutout",
            "stream_enable": "stream_enable",
            "stream_disable": "stream_disable",
        }

        return stream_tool_map.get(intent, "")

    def _call_dispatcher(self, decision: Decision) -> Any:
        tool_name = self._resolve_tool_name(decision)
        tool_args = dict(decision.tool_args)

        # Si no vienen tool_args pero la policy ha dejado los slots en metadata,
        # usamos esos slots como fallback.
        if not tool_args:
            slots = decision.metadata.get("slots")
            if isinstance(slots, dict):
                tool_args = dict(slots)

        if not tool_name:
            raise RuntimeError("Decision TOOL without tool_name")

        source = str(decision.metadata.get("source", "voice")).strip() or "voice"
        metadata = dict(decision.metadata)

        if hasattr(self.dispatcher, "dispatch"):
            return self.dispatcher.dispatch(
                tool_name=tool_name,
                args=tool_args,
                source=source,
                metadata=metadata,
            )

        if hasattr(self.dispatcher, "execute"):
            return self.dispatcher.execute(
                tool_name=tool_name,
                args=tool_args,
                source=source,
                metadata=metadata,
            )

        if callable(self.dispatcher):
            return self.dispatcher(
                tool_name=tool_name,
                args=tool_args,
                source=source,
                metadata=metadata,
            )

        raise RuntimeError("Unsupported dispatcher interface")
