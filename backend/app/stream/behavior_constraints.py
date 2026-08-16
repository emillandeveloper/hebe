from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
import time
import unicodedata
import uuid
from typing import Any, Callable


BEHAVIOR_FAMILIES = {
    "compliment", "flirtation", "affectionate_message", "relay_message",
    "promotion", "mention", "topic_engagement", "semantic_motif",
}


def _normalize(value: str) -> str:
    text = "".join(ch for ch in unicodedata.normalize("NFKD", str(value or "").casefold()) if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9_]+", " ", text).split())


@dataclass(slots=True)
class BehaviorConstraint:
    id: str
    actor: str
    behavior_family: str
    behavior_variants: list[str]
    recipient_scope: str
    recipient_user_id: str = ""
    recipient_login: str = ""
    recipient_display_name: str = ""
    requester_scope: str = "any"
    requester_user_id: str = ""
    requester_login: str = ""
    source_event_id: str = ""
    source_text: str = ""
    created_by: str = "owner"
    authority: str = "owner"
    priority: str = "owner_absolute"
    scope: str = "current_stream"
    explicitness: str = "explicit"
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    active: bool = True
    status: str = "ACTIVE"
    reason: str = ""
    retired_at: float = 0.0
    retirement_reason: str = ""
    version: int = 1

    @property
    def behavior(self) -> str:
        return self.behavior_family

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update({
            "behavior": self.behavior_family,
            "applies_to": self.recipient_scope,
            "source": "owner_direct_command",
            "ordered_by": "Leo",
            "blocked_patterns": list(self.behavior_variants),
            "created_at_iso": datetime.fromtimestamp(self.created_at, timezone.utc).isoformat(),
        })
        return value

    @classmethod
    def from_value(cls, value: "BehaviorConstraint | dict[str, Any]") -> "BehaviorConstraint":
        if isinstance(value, cls):
            return value
        data = dict(value or {})
        legacy_scope = "everyone" if data.get("applies_to") == "all_viewers" else str(data.get("applies_to") or "everyone")
        return cls(
            id=str(data.get("id") or f"constraint_{uuid.uuid4().hex[:12]}"),
            actor=str(data.get("actor") or "Hebe"),
            behavior_family=str(data.get("behavior_family") or data.get("behavior") or "topic_engagement"),
            behavior_variants=list(data.get("behavior_variants") or data.get("blocked_patterns") or []),
            recipient_scope=str(data.get("recipient_scope") or legacy_scope),
            recipient_user_id=str(data.get("recipient_user_id") or ""),
            recipient_login=str(data.get("recipient_login") or ""),
            recipient_display_name=str(data.get("recipient_display_name") or ""),
            requester_scope=str(data.get("requester_scope") or "any"),
            requester_user_id=str(data.get("requester_user_id") or ""),
            requester_login=str(data.get("requester_login") or ""),
            source_event_id=str(data.get("source_event_id") or ""), source_text=str(data.get("source_text") or data.get("reason") or ""),
            created_by=str(data.get("created_by") or "owner"), priority=str(data.get("priority") or "owner_absolute"),
            authority=str(data.get("authority") or "owner"),
            scope=str(data.get("scope") or "current_stream"), created_at=float(data.get("created_at") or time.time()) if not isinstance(data.get("created_at"), str) else time.time(),
            explicitness=str(data.get("explicitness") or "explicit"), confidence=float(data.get("confidence") or 1.0),
            expires_at=float(data.get("expires_at") or 0.0), active=bool(data.get("active", True)),
            status=str(data.get("status") or "ACTIVE"), reason=str(data.get("reason") or ""),
            retired_at=float(data.get("retired_at") or 0.0), retirement_reason=str(data.get("retirement_reason") or ""),
            version=int(data.get("version") or 1),
        )


@dataclass(slots=True)
class ConstraintCompilation:
    command_detected: bool
    constraint: BehaviorConstraint | None = None
    needs_clarification: bool = False
    reason: str = ""
    recipient_text: str = ""
    requester_text: str = ""
    candidates: list[str] = field(default_factory=list)


class BehaviorConstraintCompiler:
    COMPLIMENT_VARIANTS = ["compliment", "praise", "flirtatious_praise"]

    def __init__(self, resolver: Callable[[str], Any] | None = None) -> None:
        self.resolver = resolver

    def compile(self, text: str, *, source_event_id: str = "", now: float | None = None) -> ConstraintCompilation:
        raw = str(text or "").strip()
        normalized = _normalize(raw)
        if not self._is_stop(normalized) or not self._behavior(normalized):
            return ConstraintCompilation(False, reason="not_behavior_constraint")
        behavior = self._behavior(normalized)
        requester_text = self._extract_requester(normalized)
        recipient_text, recipient_scope = self._extract_recipient(normalized, requester_text=requester_text)
        print(f"[HEBE][BEHAVIOR_CONSTRAINT_PARSE] behavior={behavior} recipient_text={recipient_text!r} requester_text={requester_text!r}", flush=True)
        recipient = self._resolve_party(recipient_text, role="recipient") if recipient_scope == "specific_viewer" else {}
        requester = self._resolve_party(requester_text, role="requester") if requester_text else {}
        unresolved = (recipient_scope == "specific_viewer" and not recipient.get("login")) or (requester_text and not requester.get("login"))
        ambiguous = bool(recipient.get("ambiguous") or requester.get("ambiguous"))
        candidates = list(recipient.get("candidates") or requester.get("candidates") or [])
        if unresolved or ambiguous:
            reason = "ambiguous_recipient" if ambiguous else "unresolved_recipient"
            return ConstraintCompilation(True, needs_clarification=True, reason=reason, recipient_text=recipient_text, requester_text=requester_text, candidates=candidates)
        ts = float(now if now is not None else time.time())
        constraint = BehaviorConstraint(
            id=f"constraint_{uuid.uuid4().hex[:12]}", actor="Hebe", behavior_family=behavior,
            behavior_variants=self.COMPLIMENT_VARIANTS if behavior == "compliment" else [behavior],
            recipient_scope=recipient_scope,
            recipient_user_id=str(recipient.get("user_id") or ""), recipient_login=str(recipient.get("login") or ""),
            recipient_display_name=str(recipient.get("display_name") or recipient_text or ""),
            requester_scope="specific_viewer" if requester_text else "any",
            requester_user_id=str(requester.get("user_id") or ""), requester_login=str(requester.get("login") or ""),
            source_event_id=source_event_id, source_text=raw, created_at=ts,
            reason="explicit owner behavior constraint",
        )
        print(f"[HEBE][BEHAVIOR_CONSTRAINT_CREATED] id={constraint.id} behavior={behavior} recipient={constraint.recipient_login or constraint.recipient_scope} priority={constraint.priority}", flush=True)
        return ConstraintCompilation(True, constraint=constraint, reason="compiled", recipient_text=recipient_text, requester_text=requester_text)

    def _resolve_party(self, text: str, *, role: str) -> dict[str, Any]:
        if not text:
            return {}
        if self.resolver is None:
            result = {"login": "", "display_name": text, "confidence": 0.0, "candidates": []}
        else:
            resolved = self.resolver(text)
            result = self._resolution_dict(resolved)
        print(f"[HEBE][BEHAVIOR_CONSTRAINT_RESOLVE] {role}={result.get('login') or text} confidence={float(result.get('confidence') or 0.0):.2f}", flush=True)
        return result

    @staticmethod
    def _resolution_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        get = value.get if isinstance(value, dict) else lambda key, default=None: getattr(value, key, default)
        reason = str(get("reason", "") or "")
        candidates = list(get("candidates", []) or [])
        return {
            "login": str(get("username", "") or get("login", "") or ""),
            "display_name": str(get("display_name", "") or ""), "user_id": str(get("user_id", "") or ""),
            "confidence": float(get("confidence", 0.0) or 0.0), "candidates": candidates,
            "ambiguous": reason == "ambiguous_target" or len(candidates) > 1 and float(get("confidence", 0.0) or 0.0) < .9,
        }

    @staticmethod
    def _is_stop(text: str) -> bool:
        return bool(re.search(r"\b(?:no|deja|para|basta|corta|cancela|evita|nunca)\b", text))

    @staticmethod
    def _behavior(text: str) -> str:
        if re.search(r"\b(?:cumplid\w*|pirop\w*|halag\w*|elog\w*|alab\w*)\b", text): return "compliment"
        if re.search(r"\b(?:flirt\w*|coquet\w*)\b", text): return "flirtation"
        if re.search(r"\b(?:mensaje\s+carinoso|afecto|carino)\b", text): return "affectionate_message"
        if re.search(r"\b(?:retransmit|recado|mensaje\s+a)\b", text): return "relay_message"
        if re.search(r"\b(?:promo|promocion|shoutout|so)\b", text): return "promotion"
        if "mencion" in text: return "mention"
        if re.search(r"\b(?:tema|hablar|responder)\b", text): return "topic_engagement"
        return ""

    @staticmethod
    def _extract_requester(text: str) -> str:
        match = re.search(r"(?:pedid[oa]s?|solicitad[oa]s?|que\s+pida|que\s+me\s+pida)\s+(?:por|de)\s+([a-z0-9_]+)", text)
        if not match:
            match = re.search(r"(?:cumplidos?|piropos?|halagos?)\s+(?:que\s+)?(?:pida|pide)\s+([a-z0-9_]+)", text)
        return str(match.group(1) if match else "")

    @staticmethod
    def _extract_recipient(text: str, *, requester_text: str) -> tuple[str, str]:
        if re.search(r"\b(?:a\s+nadie|a\s+ningun[oa]|a\s+cualquiera|cumplidos?\s+a\s+todos?)\b", text):
            return "", "everyone"
        if re.search(r"\b(?:a\s+leo|al\s+dueno|al\s+owner|hacia\s+mi|para\s+mi|decirme|me\s+(?:digas|hagas|des))\b", text):
            return "Leo", "owner"
        if not requester_text and re.search(r"\b(?:modo\s+baboso|festival\s+de\s+halagos)\b", text):
            return "Leo", "owner"
        if requester_text:
            return "", "any_viewer"
        match = re.search(r"\b(?:cumplid\w*|pirop\w*|halag\w*|elog\w*|alab\w*)\s+(?:mas\s+)?a\s+([a-z0-9_]+)", text)
        if not match:
            match = re.search(r"\b(?:a|al)\s+([a-z0-9_]+)\s+(?:mas\s+)?(?:cumplid\w*|pirop\w*|halag\w*)", text)
        return (str(match.group(1)), "specific_viewer") if match else ("", "everyone")


def persist_constraint(stream: Any, constraint: BehaviorConstraint) -> dict[str, Any]:
    values = [BehaviorConstraint.from_value(item) for item in list(getattr(stream, "active_behavior_blocks", []) or []) if isinstance(item, (dict, BehaviorConstraint))]
    motif_keys = {item for item in constraint.behavior_variants if item.startswith("motif:")}
    values = [
        item for item in values
        if not (
            item.behavior_family == constraint.behavior_family
            and item.recipient_scope == constraint.recipient_scope
            and item.recipient_login.casefold() == constraint.recipient_login.casefold()
            and item.requester_login.casefold() == constraint.requester_login.casefold()
            and (
                constraint.behavior_family != "semantic_motif"
                or bool(motif_keys & {value for value in item.behavior_variants if value.startswith("motif:")})
            )
        )
    ]
    values.append(constraint)
    setattr(stream, "active_behavior_blocks", [item.to_dict() for item in values])
    return constraint.to_dict()


def constraint_matches(constraint: BehaviorConstraint | dict[str, Any], *, behavior_family: str, recipient_login: str = "", requester_login: str = "") -> bool:
    item = BehaviorConstraint.from_value(constraint)
    if not item.active or item.behavior_family != behavior_family:
        return False
    recipient = _normalize(recipient_login)
    requester = _normalize(requester_login)
    recipient_match = item.recipient_scope == "everyone" or item.recipient_scope == "any_viewer" or (item.recipient_scope == "owner" and recipient in {"leo", "owner"}) or (item.recipient_scope == "specific_viewer" and recipient == _normalize(item.recipient_login or item.recipient_display_name))
    requester_match = item.requester_scope == "any" or requester == _normalize(item.requester_login)
    return recipient_match and requester_match


class BehaviorConstraintOutputGuard:
    PRAISE = re.compile(r"\b(?:buen\s+gusto|campeon[ao]?|guap[ao]|list[ao]|increible|genial|maravillos[ao]|mejor|imprescindible|sin\s+ti|admirable|crack)\b", re.I)

    def evaluate(self, constraints: list[dict[str, Any]], *, intended_recipient: str, generated_response: str, source_viewer: str = "", speech_act: str = "", scene_context: dict[str, Any] | None = None) -> dict[str, Any]:
        text = str(generated_response or "")
        for raw in constraints:
            constraint = BehaviorConstraint.from_value(raw)
            if constraint.behavior_family not in {"compliment", "flirtation"}:
                continue
            if not constraint_matches(constraint, behavior_family=constraint.behavior_family, recipient_login=intended_recipient, requester_login=source_viewer):
                continue
            violations = []
            if self.PRAISE.search(_normalize(text)):
                violations = ["compliment_to_blocked_recipient", "praise_to_blocked_recipient"]
            if violations:
                result = {"passed": False, "constraint_id": constraint.id, "violations": violations, "action": "repair", "repaired_response": self._neutral_boundary(intended_recipient)}
                print(f"[HEBE][BEHAVIOR_CONSTRAINT_OUTPUT_GUARD] passed=false constraint_id={constraint.id} violations={violations!r} action=repair", flush=True)
                return result
        print("[HEBE][BEHAVIOR_CONSTRAINT_OUTPUT_GUARD] passed=true constraint_id=none violations=[] action=allow", flush=True)
        return {"passed": True, "constraint_id": "", "violations": [], "action": "allow", "repaired_response": text}

    @staticmethod
    def _neutral_boundary(viewer: str) -> str:
        name = str(viewer or "chat").strip().lstrip("@")
        return f"Ese hilo queda cerrado, {name}." if name else "Ese hilo queda cerrado."


def render_constraint_confirmation(constraint: BehaviorConstraint) -> tuple[str, dict[str, Any]]:
    recipient = "Leo" if constraint.recipient_scope == "owner" else "nadie" if constraint.recipient_scope == "everyone" else constraint.recipient_display_name or constraint.recipient_login or "cualquier viewer"
    requester = constraint.requester_login if constraint.requester_scope == "specific_viewer" else "cualquiera"
    text = f"Entendido: bloqueo {constraint.behavior_family} hacia {recipient}, lo pida {requester}, durante este stream."
    invariant = {"passed": bool(constraint.behavior_family in text and recipient in text), "differences": []}
    print(f"[HEBE][CONSTRAINT_CONFIRMATION_INVARIANT] passed={str(invariant['passed']).lower()} differences={invariant['differences']!r}", flush=True)
    return text, invariant
