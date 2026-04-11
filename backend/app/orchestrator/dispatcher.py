# backend/app/orchestrator/dispatcher.py

from __future__ import annotations

from typing import Any, Callable, Optional

from .models import ExecutionResult, make_error, make_success


class OrchestratorDispatcher:
    """
    Dispatcher nuevo del orquestador.

    No depende de IntentFrame legacy.
    Recibe tool_name + tool_args directamente.
    """

    def __init__(
        self,
        *,
        tools: Optional[dict[str, Callable[..., Any]]] = None,
        runtime: Any | None = None,
    ) -> None:
        self.tools = tools or {}
        self.runtime = runtime

    def dispatch(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        source: str = "voice",
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult | dict[str, Any] | str:
        metadata = metadata or {}

        if not tool_name:
            return make_error(
                error="Missing tool_name",
                output_text="No sé qué herramienta ejecutar.",
            )

        handler = self.tools.get(tool_name)
        if handler is None:
            return make_error(
                error=f"Unknown tool: {tool_name}",
                output_text=f"No conozco la herramienta '{tool_name}'.",
            )

        try:
            result = handler(
                args=args,
                source=source,
                metadata=metadata,
            )
            return self._normalize_result(tool_name, result)

        except Exception as exc:
            return make_error(
                error=str(exc),
                output_text=f"Ha fallado la acción '{tool_name}'.",
            )

    def _normalize_result(
        self,
        tool_name: str,
        result: Any,
    ) -> ExecutionResult | dict[str, Any] | str:
        if isinstance(result, ExecutionResult):
            return result

        if isinstance(result, dict):
            return result

        if isinstance(result, str):
            return result

        return make_success(
            output_text=f"Acción '{tool_name}' ejecutada.",
            data={"raw_result": result},
        )