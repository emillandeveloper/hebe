from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass


def _norm(value: str) -> str:
    raw = "".join(
        char for char in unicodedata.normalize("NFKD", str(value or "").casefold())
        if not unicodedata.combining(char)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", raw).split())


@dataclass(frozen=True, slots=True)
class ViewerOperationDecision:
    detected: bool
    operation: str
    outcome: str
    may_generate_reply: bool
    may_execute: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class ViewerStreamOperationTopicGate:
    """Separates discussion of stream operations from authority to perform them."""

    _OPERATIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("raid", re.compile(r"\b(?:raid|raidear|raideo|raided|hacer una raid)\b")),
        ("promotion", re.compile(r"\b(?:shoutout|shout out|promo|promociona|so)\b")),
        ("moderation", re.compile(r"\b(?:ban|banea|banear|timeout|time out|silencia|expulsa)\b")),
        ("title", re.compile(r"\b(?:titulo|title)\b")),
        ("category", re.compile(r"\b(?:categoria|category|cambia de juego|change game)\b")),
        ("stream_state", re.compile(r"\b(?:inicia|empieza|start|deten|para|stop|termina) (?:el )?(?:directo|stream)\b")),
    )
    _HISTORICAL = re.compile(
        r"\b(?:ayer|antes|la ultima vez|hiciste|hicimos|vino|llego|"
        r"yesterday|last time|previously|you did|we did|came from)\b"
    )

    def evaluate(
        self,
        text: str,
        *,
        source_type: str = "viewer",
        owner_trusted: bool = False,
    ) -> ViewerOperationDecision:
        normalized = f" {_norm(text)} "
        operation = next((name for name, pattern in self._OPERATIONS if pattern.search(normalized)), "")
        if not operation:
            return ViewerOperationDecision(False, "", "not_applicable", True, False, "no_stream_operation_topic")
        if owner_trusted or str(source_type or "").lower() == "owner":
            return ViewerOperationDecision(True, operation, "owner_acknowledgement", True, True, "trusted_owner_live_control")
        historical = bool(self._HISTORICAL.search(normalized))
        if historical:
            return ViewerOperationDecision(
                True, operation, "authority_preserving_banter", True, False,
                "viewer_historical_discussion_only",
            )
        return ViewerOperationDecision(
            True, operation, "observe_only", False, False,
            "viewer_cannot_coordinate_stream_operations",
        )


__all__ = ["ViewerOperationDecision", "ViewerStreamOperationTopicGate"]
