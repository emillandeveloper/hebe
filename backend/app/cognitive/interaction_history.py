from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import re
import time
import unicodedata
from typing import Any, Iterable


VIEWER_AUTHORITY_REASONS = {
    "owner_behavior_block",
    "owner_behavior_constraint",
    "viewer_behavior_request",
    "viewer_not_authority",
    "viewer_not_authorized",
    "viewer_repeat_to_leo_request",
    "viewer_proxy_request",
}


def _normalize(value: str) -> str:
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", str(value or "").casefold())
        if not unicodedata.combining(char)
    )
    return " ".join(re.sub(r"[^a-z0-9_]+", " ", text).split())


@dataclass(frozen=True, slots=True)
class SelfExplanationQuery:
    detected: bool
    requested_kind: str = ""
    target_actor: str = ""
    asks_about_silence: bool = False
    asks_about_previous_reply: bool = False


@dataclass(frozen=True, slots=True)
class GroundedSelfExplanation:
    detected: bool
    text: str = ""
    source_trace_id: str = ""
    reason_code: str = ""
    matched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_self_explanation_query(
    text: str,
    *,
    requester: str = "",
    known_identities: Iterable[str] = (),
) -> SelfExplanationQuery:
    normalized = _normalize(text)
    if not re.search(r"\b(?:por qu(?:e)?|porque)\b", normalized):
        return SelfExplanationQuery(False)
    self_action = bool(re.search(
        r"\b(?:ignor(?:aste|abas|as)|contest(?:aste|abas|as)|respond(?:iste|ias|es)|"
        r"hiciste|haces|dijiste|dices|abriste|abres|callaste|callas|salio|salio)\b",
        normalized,
    ))
    omitted_action = bool(re.search(
        r"\bno\s+(?:le\s+)?(?:hiciste|contestaste|respondiste|dijiste|abriste|salio|salio)\b",
        normalized,
    ))
    if not (self_action or omitted_action):
        return SelfExplanationQuery(False)

    if re.search(r"\b(?:abriste|abres)\b", normalized):
        requested_kind = "open_application"
    elif re.search(r"\b(?:dijiste|dices|no\s+dijiste)\b", normalized):
        requested_kind = "behavior_or_reply"
    elif re.search(r"\b(?:ignor|contest|respond|callaste|callas|salio)\w*\b", normalized):
        requested_kind = "reply"
    else:
        requested_kind = "action_or_policy"

    target_actor = ""
    if re.search(r"\b(?:me|a mi)\b", normalized):
        target_actor = str(requester or "").strip()
    else:
        identities = sorted(
            {str(item or "").strip() for item in known_identities if str(item or "").strip()},
            key=len,
            reverse=True,
        )
        for identity in identities:
            identity_norm = _normalize(identity)
            if identity_norm and re.search(rf"(?<!\w){re.escape(identity_norm)}(?!\w)", normalized):
                target_actor = identity
                break

    return SelfExplanationQuery(
        True,
        requested_kind=requested_kind,
        target_actor=target_actor,
        asks_about_silence=bool(re.search(r"\b(?:ignor|contest|respond|callaste|callas|no\s+salio)\w*\b", normalized)),
        asks_about_previous_reply=bool(re.search(r"\b(?:dijiste|dices|eso)\b", normalized)),
    )


class RecentInteractionDecisionHistory:
    """Small session-scoped projection of observable decisions, never chain-of-thought."""

    def __init__(self, *, max_items: int = 32, ttl_seconds: float = 20 * 60) -> None:
        self.max_items = max(4, int(max_items))
        self.ttl_seconds = max(60.0, float(ttl_seconds))

    def all(self, stream: Any, *, now: float | None = None) -> list[dict[str, Any]]:
        if stream is None:
            return []
        now = time.time() if now is None else float(now)
        recent = [
            dict(item)
            for item in list(getattr(stream, "recent_interaction_decisions", []) or [])
            if isinstance(item, dict)
            and now - float(item.get("timestamp", 0.0) or 0.0) <= self.ttl_seconds
        ]
        recent = recent[-self.max_items :]
        stream.recent_interaction_decisions = recent
        return recent

    def upsert(self, stream: Any, record: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        clean = dict(record or {})
        trace_id = str(clean.get("trace_id") or clean.get("event_id") or "").strip()
        if not trace_id:
            raise ValueError("interaction decision requires trace_id")
        clean["trace_id"] = trace_id
        clean.setdefault("event_id", trace_id)
        clean.setdefault("timestamp", now)
        clean.setdefault("created_at", datetime.fromtimestamp(float(clean["timestamp"]), timezone.utc).isoformat())
        clean.setdefault("actor", "unknown")
        clean.setdefault("target", "Hebe")
        clean.setdefault("interaction_decision", "observed")
        clean.setdefault("authority", "unknown")
        clean.setdefault("requested_effect", "")
        clean.setdefault("effect_authorized", False)
        clean.setdefault("reply_authorized", False)
        clean.setdefault("reason_code", "none")
        clean.setdefault("response_intent", "")
        clean.setdefault("generation_outcome", "not_attempted")
        clean.setdefault("emission_outcome", "pending")
        clean.setdefault("actor_identities", [str(clean.get("actor") or "unknown")])

        items = self.all(stream, now=now)
        existing = next((item for item in items if str(item.get("trace_id") or "") == trace_id), None)
        if existing is not None:
            existing.update(clean)
            result = existing
        else:
            items.append(clean)
            result = clean
        stream.recent_interaction_decisions = items[-self.max_items :]
        return dict(result)

    def update(self, stream: Any, trace_id: str, **updates: Any) -> dict[str, Any] | None:
        trace_key = str(trace_id or "").strip()
        if not trace_key or stream is None:
            return None
        items = self.all(stream)
        for item in reversed(items):
            if str(item.get("trace_id") or item.get("event_id") or "") == trace_key:
                item.update({key: value for key, value in updates.items() if value is not None})
                stream.recent_interaction_decisions = items[-self.max_items :]
                return dict(item)
        return None

    def resolve(
        self,
        stream: Any,
        query: SelfExplanationQuery,
        *,
        exclude_trace_id: str = "",
    ) -> dict[str, Any] | None:
        if not query.detected:
            return None
        target = _normalize(query.target_actor)
        ranked: list[tuple[float, int, dict[str, Any]]] = []
        for index, item in enumerate(self.all(stream)):
            trace_id = str(item.get("trace_id") or item.get("event_id") or "")
            if exclude_trace_id and trace_id == exclude_trace_id:
                continue
            score = float(index) / 1000.0
            identities = {
                _normalize(value)
                for value in list(item.get("actor_identities") or []) + [item.get("actor")]
                if _normalize(value)
            }
            if target:
                if target not in identities:
                    continue
                score += 8.0
            effect = _normalize(str(item.get("requested_effect") or ""))
            reason = _normalize(str(item.get("reason_code") or ""))
            intent = _normalize(str(item.get("response_intent") or ""))
            if query.requested_kind == "open_application":
                if "open application" not in effect and "open_application" not in effect:
                    continue
                score += 6.0
            elif query.requested_kind == "reply":
                if item.get("reply_authorized") or item.get("emission_outcome") not in {"", "pending", "not_applicable"}:
                    score += 3.0
            elif query.requested_kind == "behavior_or_reply":
                if any(marker in " ".join((effect, reason, intent)) for marker in ("behavior", "motif", "repeat", "reply", "boundary", "generation")):
                    score += 3.0
            elif query.requested_kind == "action_or_policy":
                if not item.get("effect_authorized") or "policy" in str(item.get("interaction_decision") or ""):
                    score += 2.0
            ranked.append((score, index, item))
        return dict(max(ranked, key=lambda row: (row[0], row[1]))[2]) if ranked else None


def render_grounded_self_explanation(
    query: SelfExplanationQuery,
    decision: dict[str, Any] | None,
    *,
    requester: str = "",
) -> GroundedSelfExplanation:
    if not query.detected:
        return GroundedSelfExplanation(False)
    if not decision:
        return GroundedSelfExplanation(
            True,
            "No tengo suficiente contexto reciente para explicarlo sin inventarme una causa.",
            reason_code="insufficient_recent_context",
            matched=False,
        )

    trace_id = str(decision.get("trace_id") or decision.get("event_id") or "")
    reason = str(decision.get("reason_code") or "none")
    emission = str(decision.get("emission_outcome") or "")
    generation = str(decision.get("generation_outcome") or "")
    actor = str(decision.get("actor") or query.target_actor or "").strip()
    requester_matches_actor = bool(actor and requester and _normalize(actor) == _normalize(requester))

    if reason in VIEWER_AUTHORITY_REASONS:
        if emission in {"emitted", "public_reply_emitted", "local_reply_emitted"}:
            prefix = "No te ignoré" if requester_matches_actor else f"No ignoré a {actor}" if actor else "No fue por ignorar a nadie"
            text = f"{prefix}: marqué el límite porque el chat puede hablar conmigo, pero no decidir lo que hago con Leo."
        elif reason in {"boundary_cooldown", "repeated_boundary"} or emission == "boundary_cooldown":
            text = "Ese límite ya estaba marcado y no quise convertir la conversación en el mismo bucle una y otra vez."
        else:
            text = "No seguí esa petición porque el chat puede hablar conmigo, pero no decidir lo que hago con Leo."
    elif reason in {"app_not_found", "app_path_missing"}:
        text = "No la abrí porque no encontré una instalación válida de esa aplicación."
    elif reason in {"ambiguous_app_selection"}:
        text = "No la abrí porque encontré más de una opción válida y faltaba saber cuál querías."
    elif "behavior" in reason and any(marker in reason for marker in ("repeat", "similar", "fatigue", "suppress")):
        text = "No lo dije porque se estaba pareciendo demasiado a algo ya repetido y preferí no quemar el mismo gag."
    elif "generation" in reason or generation in {"failed", "failed_terminal_fallback", "terminal_fallback"}:
        text = "No salió una respuesta suficientemente buena y preferí reconocer el hueco antes que rellenarlo con humo."
    elif emission and emission not in {"emitted", "public_reply_emitted", "local_reply_emitted", "pending"}:
        text = "La respuesta no llegó a salir; lo que tengo registrado no apunta a una decisión personal contra nadie."
    else:
        text = "Tengo registrado lo que ocurrió, pero no una causa suficientemente concreta como para explicarla sin inventar."
        reason = "insufficient_structured_cause"

    return GroundedSelfExplanation(
        True,
        text,
        source_trace_id=trace_id,
        reason_code=reason,
        matched=True,
    )
