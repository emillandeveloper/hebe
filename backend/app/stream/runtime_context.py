from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class HebeLiveRuntimeContext(StrEnum):
    """Execution contexts that share Hebe's identity, but never her routes."""

    OWNER_LOCAL = "owner_local"
    OWNER_LIVE_CONTROL = "owner_live_control"
    STREAM_PUBLIC = "stream_public"


@dataclass(frozen=True, slots=True)
class ContextAuthorization:
    allowed: bool
    context: str
    operation: str
    reason: str


class HebeLiveContextPolicy:
    """Hard capability boundary for Hebe Core and Hebe Live execution.

    Persistent identity and memory can be shared by callers. Output routes and
    executable capabilities are intentionally evaluated here, independently.
    """

    _OUTPUTS = {
        HebeLiveRuntimeContext.OWNER_LOCAL: frozenset({"local_ui", "local_tts"}),
        HebeLiveRuntimeContext.OWNER_LIVE_CONTROL: frozenset({"local_ui", "local_tts"}),
        HebeLiveRuntimeContext.STREAM_PUBLIC: frozenset({"twitch_chat", "stream_tts", "local_ui"}),
    }
    _ACTION_PREFIXES = {
        HebeLiveRuntimeContext.OWNER_LOCAL: (
            "desktop.",
            "local.",
            "memory.",
            "owner.",
        ),
        HebeLiveRuntimeContext.OWNER_LIVE_CONTROL: (
            "stream.",
            "twitch.shoutout",
            "twitch.send_message",
            "promotion.",
        ),
        HebeLiveRuntimeContext.STREAM_PUBLIC: (
            "stream.observe",
            "stream.respond",
        ),
    }

    def authorize_output(
        self,
        context: HebeLiveRuntimeContext | str,
        targets: Iterable[str],
    ) -> ContextAuthorization:
        resolved = self.resolve(context)
        requested = {str(target or "").strip() for target in targets if str(target or "").strip()}
        forbidden = requested - set(self._OUTPUTS[resolved])
        if forbidden:
            return ContextAuthorization(
                False,
                resolved.value,
                "output",
                f"route_not_allowed:{','.join(sorted(forbidden))}",
            )
        return ContextAuthorization(True, resolved.value, "output", "context_route_allowed")

    def authorize_action(
        self,
        context: HebeLiveRuntimeContext | str,
        action: str,
        *,
        trusted_automation: bool = False,
    ) -> ContextAuthorization:
        resolved = self.resolve(context)
        operation = str(action or "").strip().lower()
        if resolved is HebeLiveRuntimeContext.STREAM_PUBLIC and trusted_automation:
            allowed = operation in {"promotion.automatic_first_message", "promotion.raid_policy"}
        else:
            allowed = any(operation.startswith(prefix) for prefix in self._ACTION_PREFIXES[resolved])
        return ContextAuthorization(
            allowed,
            resolved.value,
            operation,
            "context_action_allowed" if allowed else "context_action_forbidden",
        )

    @staticmethod
    def resolve(context: HebeLiveRuntimeContext | str) -> HebeLiveRuntimeContext:
        if isinstance(context, HebeLiveRuntimeContext):
            return context
        return HebeLiveRuntimeContext(str(context or "").strip().lower())

    @staticmethod
    def from_source(
        source: str,
        *,
        owner_trusted: bool = False,
        live_control_evidence: bool = False,
    ) -> HebeLiveRuntimeContext:
        normalized = str(source or "").strip().lower()
        if (
            normalized.startswith("twitch")
            or normalized in {
                "spontaneity",
                "simulation",
                "owner_discourse_opportunity",
                "stream_companion_tick",
            }
        ):
            return HebeLiveRuntimeContext.STREAM_PUBLIC
        if owner_trusted and live_control_evidence:
            return HebeLiveRuntimeContext.OWNER_LIVE_CONTROL
        return HebeLiveRuntimeContext.OWNER_LOCAL


__all__ = [
    "ContextAuthorization",
    "HebeLiveContextPolicy",
    "HebeLiveRuntimeContext",
]
