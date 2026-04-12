# backend/app/orchestrator/policy.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .models import (
    Decision,
    DecisionKind,
    IntentResult,
    OrchestratorInput,
)


# =========================
# Config de política
# =========================

@dataclass(slots=True)
class IntentPolicyRule:
    tool_name: Optional[str] = None
    required_slots: list[str] = field(default_factory=list)
    confirm: bool = False
    clarify_prompts: dict[str, str] = field(default_factory=dict)
    confirm_prompt: Optional[str] = None


DEFAULT_INTENT_POLICY: dict[str, IntentPolicyRule] = {
    "open_app": IntentPolicyRule(
        tool_name="open_app",
        required_slots=["app_name"],
        confirm=False,
        clarify_prompts={
            "app_name": "¿Qué aplicación quieres que abra?",
        },
    ),
    "close_window": IntentPolicyRule(
        tool_name="close_window",
        required_slots=[],
        confirm=False,
        clarify_prompts={
            "target": "¿Quieres que cierre la ventana activa o una aplicación concreta?",
        },
    ),
    "set_volume": IntentPolicyRule(
        tool_name="set_volume",
        required_slots=[],  # 🔥 ya no obligamos value
        confirm=False,
        clarify_prompts={
            "value": "¿Qué volumen quieres que ponga?",
        },
    ),
    "play_music": IntentPolicyRule(
        tool_name="play_music",
        required_slots=[],
        confirm=False,
        clarify_prompts={
            "query": "¿Qué quieres que ponga?",
        },
    ),
    "shutdown_pc": IntentPolicyRule(
        tool_name="shutdown_pc",
        required_slots=[],
        confirm=True,
        confirm_prompt="¿Seguro que quieres apagar el ordenador?",
    ),
    "restart_pc": IntentPolicyRule(
        tool_name="restart_pc",
        required_slots=[],
        confirm=True,
        confirm_prompt="¿Seguro que quieres reiniciar el ordenador?",
    ),
    "sleep_pc": IntentPolicyRule(
        tool_name="sleep_pc",
        required_slots=[],
        confirm=True,
        confirm_prompt="¿Seguro que quieres poner el ordenador en suspensión?",
    ),
}


# =========================
# Policy principal
# =========================

class OrchestratorPolicy:
    def __init__(
        self,
        *,
        intent_policy: Optional[dict[str, IntentPolicyRule]] = None,
        low_confidence_threshold: float = 0.45,
        tool_confidence_threshold: float = 0.60,
        ignore_empty_input: bool = True,
    ) -> None:
        self.intent_policy = intent_policy or DEFAULT_INTENT_POLICY
        self.low_confidence_threshold = low_confidence_threshold
        self.tool_confidence_threshold = tool_confidence_threshold
        self.ignore_empty_input = ignore_empty_input

    def decide(
        self,
        user_input: OrchestratorInput,
        intent_result: IntentResult,
        state: Any = None,
    ) -> Decision:
        text = (user_input.text or "").strip()

        # 1. Ignorar input vacío
        if self.ignore_empty_input and not text:
            return Decision(
                kind=DecisionKind.IGNORE,
                reason="empty_input",
            )

        # 2. Sin intent claro -> chat
        if not intent_result.has_intent:
            return Decision(
                kind=DecisionKind.CHAT,
                reason="no_intent_detected",
                metadata={
                    "confidence": intent_result.confidence,
                    "resolver_source": intent_result.source,
                },
            )

        # 3. Intent muy débil -> chat
        if intent_result.confidence < self.low_confidence_threshold:
            return Decision(
                kind=DecisionKind.CHAT,
                intent=intent_result.intent,
                reason="low_confidence_intent",
                metadata={
                    "confidence": intent_result.confidence,
                    "resolver_source": intent_result.source,
                },
            )

        # 4. Si el intent no está mapeado como tool -> chat
        rule = self.intent_policy.get(intent_result.intent)
        if rule is None:
            return Decision(
                kind=DecisionKind.CHAT,
                intent=intent_result.intent,
                reason="intent_not_mapped_as_tool",
                metadata={
                    "confidence": intent_result.confidence,
                    "resolver_source": intent_result.source,
                },
            )

        # 5. Si es tool pero la confianza aún no da seguridad -> chat
        if intent_result.confidence < self.tool_confidence_threshold:
            return Decision(
                kind=DecisionKind.CHAT,
                intent=intent_result.intent,
                reason="tool_confidence_too_low",
                metadata={
                    "confidence": intent_result.confidence,
                    "resolver_source": intent_result.source,
                },
            )

        # 6. Validar slots obligatorios
        missing_slots = self._find_missing_slots(rule.required_slots, intent_result.slots)

        if missing_slots:
            prompt = self._build_clarify_prompt(
                intent=intent_result.intent,
                missing_slots=missing_slots,
                clarify_prompts=rule.clarify_prompts,
            )

            return Decision(
                kind=DecisionKind.CLARIFY,
                intent=intent_result.intent,
                response=prompt,
                tool_name=rule.tool_name,
                tool_args=dict(intent_result.slots),
                missing_slots=missing_slots,
                reason="missing_required_slots",
                metadata={
                    "confidence": intent_result.confidence,
                    "resolver_source": intent_result.source,
                },
            )

        # 7. Confirmación si la acción es sensible
        if rule.confirm:
            confirm_prompt = rule.confirm_prompt or self._default_confirm_prompt(intent_result.intent)

            return Decision(
                kind=DecisionKind.CONFIRM,
                intent=intent_result.intent,
                response=confirm_prompt,
                tool_name=rule.tool_name,
                tool_args=dict(intent_result.slots),
                requires_confirmation=True,
                reason="confirmation_required",
                metadata={
                    "confidence": intent_result.confidence,
                    "resolver_source": intent_result.source,
                },
            )

        # 8. Tool directa
        return Decision(
            kind=DecisionKind.TOOL,
            intent=intent_result.intent,
            tool_name=rule.tool_name,
            tool_args=dict(intent_result.slots),
            reason="tool_ready",
            metadata={
                "confidence": intent_result.confidence,
                "resolver_source": intent_result.source,
            },
        )

    # =========================
    # Helpers internos
    # =========================

    def _find_missing_slots(
        self,
        required_slots: list[str],
        provided_slots: dict[str, Any],
    ) -> list[str]:
        missing: list[str] = []

        for slot_name in required_slots:
            value = provided_slots.get(slot_name)

            if value is None:
                missing.append(slot_name)
                continue

            if isinstance(value, str) and not value.strip():
                missing.append(slot_name)
                continue

        return missing

    def _build_clarify_prompt(
        self,
        *,
        intent: str,
        missing_slots: list[str],
        clarify_prompts: dict[str, str],
    ) -> str:
        if len(missing_slots) == 1:
            slot_name = missing_slots[0]
            if slot_name in clarify_prompts:
                return clarify_prompts[slot_name]

        known_prompts = [
            clarify_prompts[slot_name]
            for slot_name in missing_slots
            if slot_name in clarify_prompts
        ]

        if known_prompts:
            return " ".join(known_prompts)

        return self._default_clarify_prompt(intent, missing_slots)

    def _default_clarify_prompt(self, intent: str, missing_slots: list[str]) -> str:
        joined = ", ".join(missing_slots)
        return f"Necesito un poco más de información para '{intent}'. Faltan: {joined}."

    def _default_confirm_prompt(self, intent: str) -> str:
        return f"¿Seguro que quieres ejecutar '{intent}'?"