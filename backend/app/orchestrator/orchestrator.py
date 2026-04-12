# backend/app/orchestrator/orchestrator.py

from __future__ import annotations

import time
from typing import Any, Optional

from .gates import OrchestratorGates
from .models import (
    Decision,
    DecisionKind,
    ExecutionResult,
    InputSource,
    OrchestratorInput,
    PendingAction,
    PendingActionType,
    make_cancelled,
    make_error,
    make_ignored,
)
from .policy import OrchestratorPolicy
from app.llm.ollama_intent_client import OllamaIntentClient
from app.orchestrator.intents.resolver import IntentResolver

intent_llm = OllamaIntentClient(
    model="hebe-intent",
    base_url="http://127.0.0.1:11434",
    timeout=20.0,
)

intent_resolver = IntentResolver(llm=intent_llm)

class Orchestrator:
    """
    Coordina el flujo principal:

    1. Construye el input
    2. Consulta gates
    3. Si no intercepta nada, resuelve intent
    4. Pasa por policy
    5. Ejecuta
    6. Actualiza state
    """

    def __init__(
        self,
        *,
        state: Any,
        intent_resolver: Any,
        executor: Any,
        policy: Optional[OrchestratorPolicy] = None,
        gates: Optional[OrchestratorGates] = None,
    ) -> None:
        self.state = state
        self.intent_resolver = intent_resolver
        self.executor = executor
        self.policy = policy or OrchestratorPolicy()
        self.gates = gates or OrchestratorGates()

    def _attach_runtime_metadata(self, decision: Decision, user_input: OrchestratorInput) -> Decision:
        decision.metadata.setdefault("user_text", user_input.text)
        decision.metadata.setdefault("source", user_input.source.value)
        return decision
    # =========================
    # Public API
    # =========================

    def handle(
        self,
        text: str,
        *,
        source: str = "voice",
        metadata: Optional[dict[str, Any]] = None,
    ) -> ExecutionResult:
        user_input = OrchestratorInput(
            text=text,
            source=self._parse_input_source(source),
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self._update_input_state(user_input)

        if getattr(self.state, "is_processing", False):
            # Si quieres permitir concurrencia más adelante, aquí cambiará.
            pass

        self.state.is_processing = True

        try:
            # 1. Gates
            gate_decision = self._check_gates(user_input)
            if gate_decision is not None:
                gate_decision = self._attach_runtime_metadata(gate_decision, user_input)
                result = self._execute_decision(gate_decision)
                self._after_turn(user_input, gate_decision, result)
                return result

            # 2. Resolver intent
            intent_result = self.intent_resolver.resolve(user_input, self.state)
            print(
                f"[HEBE][ORCH] intent_result "
                f"intent={intent_result.intent!r} "
                f"confidence={intent_result.confidence!r} "
                f"slots={intent_result.slots!r} "
                f"source={intent_result.source!r}",
                flush=True,
            )
            # 3. Policy
            decision = self.policy.decide(user_input, intent_result, self.state)

            # 4. Execute
            result = self._execute_decision(decision)

            # 5. Update state
            decision = self.policy.decide(user_input, intent_result, self.state)
            decision = self._attach_runtime_metadata(decision, user_input)

            print(
                f"[HEBE][ORCH] decision "
                f"kind={decision.kind.value!r} "
                f"intent={decision.intent!r} "
                f"tool_name={decision.tool_name!r} "
                f"tool_args={decision.tool_args!r} "
                f"missing_slots={decision.missing_slots!r} "
                f"response={decision.response!r}",
                flush=True,
            )

            result = self._execute_decision(decision)
            self._after_turn(user_input, decision, result)
            return result
        except Exception as exc:
            result = make_error(
                error=str(exc),
                output_text="Ha fallado algo al procesar la petición.",
            )
            self._clear_processing_flag()
            return result

    # =========================
    # Gates integration
    # =========================

    def _check_gates(self, user_input: OrchestratorInput) -> Optional[Decision]:
        return self.gates.check(user_input, self.state)

    # =========================
    # Execution
    # =========================

    def _execute_decision(self, decision: Decision) -> ExecutionResult:
        """
        Executor externo:
        - CHAT -> generar respuesta con LLM
        - TOOL -> ejecutar tool
        - CLARIFY/CONFIRM -> devolver frase
        - IGNORE -> no hacer nada
        """

        if decision.kind == DecisionKind.IGNORE:
            return make_ignored(
                output_text=decision.response,
                decision_kind=decision.kind,
                intent=decision.intent,
            )

        if decision.kind in {DecisionKind.CLARIFY, DecisionKind.CONFIRM}:
            # No ejecuta tool. Solo devuelve el texto.
            return self.executor.execute(decision)

        if decision.kind == DecisionKind.CHAT:
            return self.executor.execute(decision)

        if decision.kind == DecisionKind.TOOL:
            return self.executor.execute(decision)

        return make_error(
            error=f"Unsupported decision kind: {decision.kind}",
            output_text="No he podido decidir qué hacer.",
            decision_kind=decision.kind,
            intent=decision.intent,
        )

    # =========================
    # State updates
    # =========================

    def _after_turn(
        self,
        user_input: OrchestratorInput,
        decision: Decision,
        result: ExecutionResult,
    ) -> None:
        self.state.last_intent = decision.intent

        if decision.metadata.get("wake_requested"):
            self.state.mode = "active"

        self._sync_pending_state(decision)

        if decision.metadata.get("clear_pending_confirmation"):
            self.state.pending_confirmation = None

        if decision.metadata.get("clear_pending_clarification"):
            self.state.pending_clarification = None

        if decision.kind == DecisionKind.TOOL:
            self.state.current_task = decision.tool_name
        elif decision.kind in {DecisionKind.CHAT, DecisionKind.IGNORE}:
            self.state.current_task = None

        self._clear_processing_flag()

    def _sync_pending_state(self, decision: Decision) -> None:
        if decision.kind == DecisionKind.CONFIRM:
            self.state.pending_confirmation = {
                "intent": decision.intent,
                "prompt": decision.response,
                "tool_name": decision.tool_name,
                "tool_args": dict(decision.tool_args),
                "known_slots": dict(decision.tool_args),
                "missing_slots": [],
                "metadata": dict(decision.metadata),
            }

        elif decision.kind == DecisionKind.CLARIFY:
            self.state.pending_clarification = {
                "intent": decision.intent,
                "prompt": decision.response,
                "tool_name": decision.tool_name,
                "tool_args": dict(decision.tool_args),
                "known_slots": dict(decision.tool_args),
                "missing_slots": list(decision.missing_slots),
                "metadata": dict(decision.metadata),
            }

    def _update_input_state(self, user_input: OrchestratorInput) -> None:
        self.state.last_input_text = user_input.text
        self.state.last_input_source = user_input.source.value

    def _clear_processing_flag(self) -> None:
        self.state.is_processing = False

    # =========================
    # Helpers
    # =========================

    def _parse_input_source(self, source: str) -> InputSource:
        source = (source or "voice").strip().lower()

        if source == "text":
            return InputSource.TEXT
        if source == "system":
            return InputSource.SYSTEM
        return InputSource.VOICE