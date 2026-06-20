# backend/app/orchestrator/gates.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .models import (
    Decision,
    DecisionKind,
    OrchestratorInput,
)


@dataclass(slots=True)
class GateConfig:
    ignore_empty_input: bool = True
    allow_system_input_in_sleep: bool = True
    allow_wake_commands_in_sleep: bool = True
    block_while_processing: bool = False


DEFAULT_WAKE_WORDS = {
    "hebe",
    "despierta",
    "wake up",
    "wake",
    "resume",
    "reactiva",
    "activate",
    "actívate",
}

DEFAULT_CONFIRM_YES = {
    "sí",
    "si",
    "yes",
    "y",
    "vale",
    "ok",
    "okay",
    "adelante",
    "hazlo",
    "confirma",
    "confirmo",
}

DEFAULT_CONFIRM_NO = {
    "no",
    "cancela",
    "cancela",
    "stop",
    "para",
    "déjalo",
    "dejalo",
    "olvídalo",
    "olvidalo",
    "abort",
    "cancel",
}


class OrchestratorGates:
    def __init__(
        self,
        *,
        config: Optional[GateConfig] = None,
        wake_words: Optional[set[str]] = None,
        confirm_yes_words: Optional[set[str]] = None,
        confirm_no_words: Optional[set[str]] = None,
    ) -> None:
        self.config = config or GateConfig()
        self.wake_words = {w.lower().strip() for w in (wake_words or DEFAULT_WAKE_WORDS)}
        self.confirm_yes_words = {w.lower().strip() for w in (confirm_yes_words or DEFAULT_CONFIRM_YES)}
        self.confirm_no_words = {w.lower().strip() for w in (confirm_no_words or DEFAULT_CONFIRM_NO)}

    def check(
        self,
        user_input: OrchestratorInput,
        state: Any,
    ) -> Optional[Decision]:
        """
        Si algún gate intercepta el input, devuelve Decision.
        Si devuelve None, el flujo sigue hacia intent_resolver/policy.
        """
        text = (user_input.text or "").strip()
        normalized = self._normalize(text)

        if self.config.ignore_empty_input and not normalized:
            return Decision(
                kind=DecisionKind.IGNORE,
                reason="empty_input_gate",
            )

        pending_confirmation = getattr(state, "pending_confirmation", None)
        if pending_confirmation:
            return self._handle_pending_confirmation(
                normalized_text=normalized,
                pending_confirmation=pending_confirmation,
            )

        pending_clarification = getattr(state, "pending_clarification", None)
        if pending_clarification:
            # TODO(CognitiveRouter): this subsystem must not act before CognitiveDecision authorizes it.
            # legacy_flow can otherwise consume an unrelated new request as a slot value.
            return self._handle_pending_clarification(
                user_input=user_input,
                normalized_text=normalized,
                pending_clarification=pending_clarification,
            )

        mode = getattr(state, "mode", "active")
        if mode == "sleep":
            return self._handle_sleep_mode(
                user_input=user_input,
                normalized_text=normalized,
            )

        is_processing = bool(getattr(state, "is_processing", False))
        if is_processing and self.config.block_while_processing:
            return Decision(
                kind=DecisionKind.IGNORE,
                reason="processing_gate",
                response="Ahora mismo sigo ocupada con otra cosa.",
            )

        return None

    # =========================
    # Pending confirmation
    # =========================

    def _handle_pending_confirmation(
        self,
        *,
        normalized_text: str,
        pending_confirmation: dict[str, Any],
    ) -> Decision:
        intent = pending_confirmation.get("intent")
        prompt = pending_confirmation.get("prompt")
        tool_name = pending_confirmation.get("tool_name")
        tool_args = dict(pending_confirmation.get("tool_args", {}))
        metadata = dict(pending_confirmation.get("metadata", {}))

        if normalized_text in self.confirm_yes_words:
            return Decision(
                kind=DecisionKind.TOOL,
                intent=intent,
                tool_name=tool_name,
                tool_args=tool_args,
                reason="pending_confirmation_accepted",
                metadata={
                    **metadata,
                    "from_pending_confirmation": True,
                    "clear_pending_confirmation": True,
                },
            )

        if normalized_text in self.confirm_no_words:
            return Decision(
                kind=DecisionKind.IGNORE,
                intent=intent,
                response="Vale, cancelado.",
                reason="pending_confirmation_rejected",
                metadata={
                    **metadata,
                    "from_pending_confirmation": True,
                    "clear_pending_confirmation": True,
                },
            )

        return Decision(
            kind=DecisionKind.CONFIRM,
            intent=intent,
            response=prompt or "¿Seguro?",
            tool_name=tool_name,
            tool_args=tool_args,
            requires_confirmation=True,
            reason="pending_confirmation_still_waiting",
            metadata={
                **metadata,
                "from_pending_confirmation": True,
                "keep_pending_confirmation": True,
            },
        )

    # =========================
    # Pending clarification
    # =========================

    def _handle_pending_clarification(
        self,
        *,
        user_input: OrchestratorInput,
        normalized_text: str,
        pending_clarification: dict[str, Any],
    ) -> Decision:
        intent = pending_clarification.get("intent")
        prompt = pending_clarification.get("prompt")
        tool_name = pending_clarification.get("tool_name")
        tool_args = dict(pending_clarification.get("tool_args", {}))
        missing_slots = list(pending_clarification.get("missing_slots", []))
        metadata = dict(pending_clarification.get("metadata", {}))

        if normalized_text in self.confirm_no_words:
            return Decision(
                kind=DecisionKind.IGNORE,
                intent=intent,
                response="Vale, lo dejo.",
                reason="pending_clarification_cancelled",
                metadata={
                    **metadata,
                    "from_pending_clarification": True,
                    "clear_pending_clarification": True,
                },
            )

        if len(missing_slots) == 1:
            slot_name = missing_slots[0]
            merged_args = dict(tool_args)
            merged_args[slot_name] = user_input.text.strip()

            return Decision(
                kind=DecisionKind.TOOL,
                intent=intent,
                tool_name=tool_name,
                tool_args=merged_args,
                reason="pending_clarification_resolved_single_slot",
                metadata={
                    **metadata,
                    "from_pending_clarification": True,
                    "resolved_slot": slot_name,
                    "clear_pending_clarification": True,
                },
            )

        return Decision(
            kind=DecisionKind.CLARIFY,
            intent=intent,
            response=prompt or "Necesito un poco más de información.",
            tool_name=tool_name,
            tool_args=tool_args,
            missing_slots=missing_slots,
            reason="pending_clarification_still_waiting",
            metadata={
                **metadata,
                "from_pending_clarification": True,
                "keep_pending_clarification": True,
            },
        )

    # =========================
    # Sleep mode
    # =========================

    def _handle_sleep_mode(
        self,
        *,
        user_input: OrchestratorInput,
        normalized_text: str,
    ) -> Optional[Decision]:
        if user_input.source.value == "system" and self.config.allow_system_input_in_sleep:
            return None

        if self.config.allow_wake_commands_in_sleep and self._looks_like_wake_command(normalized_text):
            return Decision(
                kind=DecisionKind.CHAT,
                reason="wake_command_detected",
                response="Aquí estoy.",
                metadata={
                    "wake_requested": True,
                },
            )

        return Decision(
            kind=DecisionKind.IGNORE,
            reason="sleep_mode_gate",
        )

    # =========================
    # Helpers
    # =========================

    def _looks_like_wake_command(self, normalized_text: str) -> bool:
        if not normalized_text:
            return False

        if normalized_text in self.wake_words:
            return True

        return any(wake_word in normalized_text for wake_word in self.wake_words)

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().strip().split())
