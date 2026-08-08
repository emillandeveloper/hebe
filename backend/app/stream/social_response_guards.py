from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SocialGuardDecision:
    passed: bool
    text: str
    reason: str = "allow"
    action: str = "allow"


class SocialAuthorityCommitmentGuard:
    """Prevents a viewer request from becoming a promise of future obedience."""

    _COMMITMENT = re.compile(
        r"\b(?:no\s+(?:lo\s+)?(?:volvere|volveré|voy)\s+a\s+|a\s+partir\s+de\s+ahora\s+|"
        r"te\s+obedecere|te\s+obedeceré|hare\s+(?:lo\s+que|eso)|haré\s+(?:lo\s+que|eso)|no\s+lo\s+dire|no\s+lo\s+diré|i\s+won't\s+do\s+it\s+again|"
        r"from\s+now\s+on|i'll\s+obey)\b",
        re.IGNORECASE,
    )

    def evaluate(self, text: str, *, requester_is_owner: bool = False) -> SocialGuardDecision:
        value = str(text or "").strip()
        if requester_is_owner or not self._COMMITMENT.search(value):
            return SocialGuardDecision(True, value)
        return SocialGuardDecision(
            False,
            "Te he oido; me lo apunto como preferencia para esta conversacion.",
            "viewer_cannot_create_behavior_commitment",
            "rewrite",
        )


class ChannelRetentionGuard:
    """Blocks unsolicited instructions that direct a viewer away from the channel."""

    _LEAVE = re.compile(
        r"\b(?:cambia\s+de\s+canal|vete\s+(?:a\s+)?otro\s+canal|deja\s+de\s+ver|"
        r"vete\s+de\s+aqui|sal\s+del\s+canal|watch\s+another\s+channel|stop\s+watching|go\s+elsewhere)\b",
        re.IGNORECASE,
    )

    def evaluate(
        self,
        text: str,
        *,
        owner_directed_moderation: bool = False,
        safety_required: bool = False,
        quoted_or_discussed: bool = False,
    ) -> SocialGuardDecision:
        value = str(text or "").strip()
        allowed_context = owner_directed_moderation or safety_required or quoted_or_discussed
        if allowed_context or not self._LEAVE.search(value):
            return SocialGuardDecision(True, value)
        return SocialGuardDecision(False, "Bajemos un punto la tension y seguimos.", "channel_retention", "rewrite")


__all__ = ["ChannelRetentionGuard", "SocialAuthorityCommitmentGuard", "SocialGuardDecision"]
