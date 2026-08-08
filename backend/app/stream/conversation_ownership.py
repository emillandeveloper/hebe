from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Any


@dataclass(frozen=True)
class ConversationOwnershipDecision:
    addressee: str
    allow_assistant: bool
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConversationOwnershipGate:
    """Conservatively prevents Hebe from hijacking chatter addressed elsewhere."""

    HEBE_NAMES = {"hebe", "ebe", "eve", "jebe"}
    OWNER_NAMES = {"leo"}

    @staticmethod
    def _normalize(value: str) -> str:
        value = unicodedata.normalize("NFKD", str(value or "").casefold())
        value = "".join(ch for ch in value if not unicodedata.combining(ch))
        return " ".join(re.sub(r"[^a-z0-9_@ ]+", " ", value).split())

    def decide(self, text: str, *, payload: dict[str, Any] | None = None) -> ConversationOwnershipDecision:
        data = dict(payload or {})
        normalized = self._normalize(text)
        if bool(data.get("mentions_hebe") or data.get("direct_address_to_hebe")) or any(
            re.search(rf"(?:^|\s)@?{re.escape(name)}(?:\s|$)", normalized) for name in self.HEBE_NAMES
        ):
            return ConversationOwnershipDecision("Hebe", True, "explicit_hebe_address", 0.98)

        reply_target = self._normalize(str(
            data.get("reply_parent_user_login") or data.get("reply-parent-user-login")
            or data.get("reply_to_login") or ""
        )).lstrip("@")
        owner_login = self._normalize(str(data.get("owner_login") or "leo")).lstrip("@")
        if reply_target:
            if reply_target in self.HEBE_NAMES:
                return ConversationOwnershipDecision("Hebe", True, "reply_to_hebe", 0.99)
            if reply_target == owner_login or reply_target in self.OWNER_NAMES:
                return ConversationOwnershipDecision("Leo", False, "reply_to_owner", 0.99)
            return ConversationOwnershipDecision("other", False, "reply_to_other_viewer", 0.99)

        mentions = [item.casefold() for item in re.findall(r"@([a-z0-9_]{2,25})", normalized)]
        if mentions:
            if owner_login in mentions or any(item in self.OWNER_NAMES for item in mentions):
                return ConversationOwnershipDecision("Leo", False, "owner_mentioned", 0.95)
            return ConversationOwnershipDecision("other", False, "other_viewer_mentioned", 0.95)
        if any(
            re.search(rf"^(?:oye\s+)?{re.escape(name)}\b", normalized)
            or re.search(rf"\b(?:que opinas|tu que dices|como lo ves)\s+{re.escape(name)}$", normalized)
            for name in self.OWNER_NAMES
        ):
            return ConversationOwnershipDecision("Leo", False, "owner_named", 0.86)

        if "?" in str(text or "") and (
            len(normalized.split()) <= 6
            or re.search(r"\b(?:que opinas|tu que|como ves|que dices|puedes|podrias|te parece)\b", normalized)
        ):
            return ConversationOwnershipDecision("ambiguous", False, "ambiguous_unaddressed_question", 0.62)

        # Unaddressed public banter remains eligible for the normal Presence rules.
        return ConversationOwnershipDecision("general", True, "general_chat", 0.72)


__all__ = ["ConversationOwnershipDecision", "ConversationOwnershipGate"]
