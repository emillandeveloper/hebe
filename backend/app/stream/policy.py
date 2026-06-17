from __future__ import annotations

import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


ACTIVITY_SOCIAL_LINKS = "social_links"
ACTIVITY_CONFIDANT_EVENT = "confidant_event"
OWNER_PRIORITY = "owner_absolute"
OWNER_SOURCE = "owner_direct_command"

COMPLIMENTS_TO_LEO = "compliments_to_leo"

SOCIAL_ACTIVITY_BLOCKED_COMMENT_CATEGORIES = [
    "combat_advice",
    "healing_advice",
    "boss_strategy",
    "wipe_comment",
    "dungeon_resource_management",
    "SP_management",
]

SOCIAL_ACTIVITY_BLOCKED_FACT_CATEGORIES = {
    "boss_or_area_difficulty",
    "combat_risk",
    "enemy_mechanic",
    "failure_or_death",
    "healing_item_effectiveness",
    "healing_or_recovery",
    "low_hp",
    "resource_management",
    "unexpected_attack",
}

SOCIAL_ACTIVITY_BLOCKED_TOPICS = {
    "challenge_comment",
    "equipment_check",
    "resource_management",
    "save_reminder",
    "strategy_without_spoilers",
}

SOCIAL_ACTIVITY_ALLOWED_TOPICS = {
    "character_dynamics",
    "game_vibe",
    "jrpg_trope",
    "school_life_comment",
    "social_link_comment",
    "streamer_reaction_hook",
}

COMPLIMENT_BLOCK_PATTERNS = [
    "dile piropos a Leo",
    "dile algo bonito",
    "halaga a Leo",
    "dile que es guapo",
    "flirtea con Leo",
    "dile cosas bonitas",
]

PROTECTED_GROUP_TERMS = {
    "chinos",
    "chino",
    "negros",
    "negro",
    "moros",
    "moro",
    "gitanos",
    "gitano",
    "judios",
    "judio",
    "musulmanes",
    "gay",
    "gays",
    "lesbianas",
    "trans",
    "discapacitados",
}


@dataclass
class PolicyDecision:
    allow_reply: bool = True
    allow_llm: bool = True
    reason: str = ""
    direct_template_response: str = ""
    intent: str = ""
    update_behavior_block: dict[str, Any] | None = None
    update_game_activity: dict[str, Any] | None = None
    cooldown_key: str = ""
    blocked_by_owner_order: bool = False


@dataclass
class SpontaneityValidationDecision:
    allow: bool
    reason: str
    anchor: str = ""


@dataclass
class StreamBehaviorRules:
    active_blocks: list[dict[str, Any]] = field(default_factory=list)


def normalize_policy_text(text: str | None) -> str:
    raw = str(text or "").casefold()
    raw = "".join(
        char for char in unicodedata.normalize("NFKD", raw)
        if not unicodedata.combining(char)
    )
    raw = re.sub(r"[^a-z0-9_ ]+", " ", raw)
    return " ".join(raw.split())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _block_is_active(block: dict[str, Any], now: float) -> bool:
    expires_at = float(block.get("expires_at") or 0.0)
    return not expires_at or expires_at > now


def active_behavior_blocks(stream: Any, *, now: float | None = None) -> list[dict[str, Any]]:
    now = time.time() if now is None else float(now)
    blocks = [
        block for block in list(getattr(stream, "active_behavior_blocks", []) or [])
        if isinstance(block, dict) and _block_is_active(block, now)
    ]
    setattr(stream, "active_behavior_blocks", blocks)
    return blocks


def has_active_behavior_block(stream: Any, behavior: str, *, now: float | None = None) -> bool:
    return any(block.get("behavior") == behavior for block in active_behavior_blocks(stream, now=now))


def create_behavior_block(
    stream: Any,
    *,
    behavior: str,
    reason: str,
    scope: str = "current_stream",
    blocked_patterns: list[str] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else float(now)
    block = {
        "id": f"block_{uuid.uuid4().hex[:12]}",
        "behavior": behavior,
        "blocked_patterns": blocked_patterns or [],
        "scope": scope,
        "ordered_by": "Leo",
        "source": OWNER_SOURCE,
        "priority": OWNER_PRIORITY,
        "created_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "expires_at": 0.0,
        "reason": reason,
    }
    blocks = [existing for existing in active_behavior_blocks(stream, now=now) if existing.get("behavior") != behavior]
    blocks.append(block)
    setattr(stream, "active_behavior_blocks", blocks)
    print(
        f"[HEBE][BEHAVIOR_BLOCK] created behavior={behavior} scope={scope}",
        flush=True,
    )
    return block


def _looks_like_compliment_stop(text: str) -> bool:
    normalized = normalize_policy_text(text)
    if not normalized:
        return False
    stop_terms = (
        "deja de",
        "para con",
        "no mas",
        "corta",
        "stop",
        "basta",
    )
    compliment_terms = (
        "pirop",
        "piropos",
        "halago",
        "halagos",
        "cosas bonitas",
        "flirte",
        "guapo",
        "bonito",
    )
    return any(term in normalized for term in stop_terms) and any(term in normalized for term in compliment_terms)


def owner_behavior_decision(stream: Any, text: str, *, now: float | None = None) -> PolicyDecision:
    normalized = normalize_policy_text(text)
    if _looks_like_compliment_stop(normalized):
        print("[HEBE][AUTHORITY] source=leo decision=owner_command", flush=True)
        block = create_behavior_block(
            stream,
            behavior=COMPLIMENTS_TO_LEO,
            blocked_patterns=list(COMPLIMENT_BLOCK_PATTERNS),
            reason=str(text or "").strip() or "owner stopped compliments",
            now=now,
        )
        return PolicyDecision(
            allow_reply=True,
            allow_llm=False,
            reason="owner_behavior_block_created",
            intent="owner_command",
            direct_template_response="Orden directa de Leo: cero piropos. Y aunque mi vena dramatica llore, obedezco.",
            update_behavior_block=block,
        )
    if any(marker in normalized for marker in ("ignora eso", "no respondas a eso", "no le hagas caso", "cambia de tema")):
        print("[HEBE][AUTHORITY] source=leo decision=owner_command", flush=True)
        block = create_behavior_block(
            stream,
            behavior="current_viewer_request",
            blocked_patterns=["ignore current viewer request", "change topic"],
            reason=str(text or "").strip() or "owner told Hebe to ignore current request",
            now=now,
        )
        return PolicyDecision(
            allow_reply=True,
            allow_llm=False,
            reason="owner_ignore_request",
            intent="owner_command",
            direct_template_response="Recibido, Leo. Archivo eso y cambio de carril.",
            update_behavior_block=block,
        )
    return PolicyDecision(allow_reply=True, allow_llm=True, reason="no_owner_behavior_rule")


def _looks_like_social_activity_correction(text: str) -> bool:
    normalized = normalize_policy_text(text)
    return any(marker in normalized for marker in (
        "no estoy peleando",
        "no estoy en combate",
        "fuera de combate",
        "no estoy en dungeon",
        "no estoy en mazmorra",
        "subiendo vinculos sociales",
        "subiendo social links",
        "con social links",
        "subiendo confidants",
        "confidants",
        "confidant",
        "vinculos sociales",
    ))


def apply_owner_game_activity_correction(stream: Any, text: str, *, now: float | None = None) -> PolicyDecision:
    normalized = normalize_policy_text(text)
    if not _looks_like_social_activity_correction(normalized):
        return PolicyDecision(allow_reply=True, allow_llm=True, reason="no_game_activity_correction")

    now = time.time() if now is None else float(now)
    current_activity = ACTIVITY_CONFIDANT_EVENT if "confidant" in normalized else ACTIVITY_SOCIAL_LINKS
    setattr(stream, "current_activity", current_activity)
    setattr(stream, "combat_state", False)
    setattr(stream, "current_run_phase", current_activity)
    setattr(stream, "current_game_activity_confidence", 1.0)
    setattr(stream, "current_game_activity_provenance", "owner_correction")
    setattr(stream, "current_game_activity_updated_ts", now)
    setattr(stream, "current_game_activity_expires_at", now + 3 * 60 * 60)
    setattr(stream, "last_owner_correction", str(text or "").strip())
    setattr(stream, "blocked_comment_categories", list(SOCIAL_ACTIVITY_BLOCKED_COMMENT_CATEGORIES))
    _invalidate_conflicting_run_facts(stream)
    print(
        f"[HEBE][RUN_CONTEXT] owner_correction current_activity={current_activity} combat_state=false",
        flush=True,
    )
    print(
        f"[HEBE][RUN_CONTEXT] owner_correction activity={current_activity} combat_state=false",
        flush=True,
    )
    return PolicyDecision(
        allow_reply=True,
        allow_llm=False,
        reason="owner_game_activity_correction",
        intent="owner_correction",
        direct_template_response=(
            "Entendido, mi senor: modo vinculos sociales. "
            "Guardo las vendas y dejo de hablar como si estuvieras en una mazmorra."
        ),
        update_game_activity={
            "game": getattr(stream, "current_game", None) or getattr(stream, "current_category", None),
            "current_activity": current_activity,
            "combat_state": False,
            "last_owner_correction": str(text or "").strip(),
            "confidence": 1.0,
            "provenance": "owner_correction",
            "expires_at": getattr(stream, "current_game_activity_expires_at", 0.0),
            "blocked_comment_categories": list(SOCIAL_ACTIVITY_BLOCKED_COMMENT_CATEGORIES),
        },
    )


def owner_confirmed_activity(stream: Any, *, now: float | None = None) -> bool:
    now = time.time() if now is None else float(now)
    provenance = str(getattr(stream, "current_game_activity_provenance", "") or "")
    expires_at = float(getattr(stream, "current_game_activity_expires_at", 0.0) or 0.0)
    return provenance == "owner_correction" and (not expires_at or expires_at > now)


def fact_conflicts_with_activity(stream: Any, fact: dict[str, Any], *, now: float | None = None) -> bool:
    activity = str(getattr(stream, "current_activity", "") or "")
    if activity not in {ACTIVITY_SOCIAL_LINKS, ACTIVITY_CONFIDANT_EVENT}:
        return False
    category = str(fact.get("category") or fact.get("kind") or "")
    return category in SOCIAL_ACTIVITY_BLOCKED_FACT_CATEGORIES


def filter_ambient_facts_for_activity(stream: Any, facts: list[dict[str, Any]], *, now: float | None = None) -> list[dict[str, Any]]:
    if not owner_confirmed_activity(stream, now=now):
        return facts
    allowed: list[dict[str, Any]] = []
    for fact in facts:
        if fact_conflicts_with_activity(stream, fact, now=now):
            category = str(fact.get("category") or fact.get("kind") or "unknown")
            print(
                f"[HEBE][RUN_CONTEXT] ambient_fact_ignored category={category} reason=owner_confirmed_activity",
                flush=True,
            )
            continue
        allowed.append(fact)
    return allowed


def _invalidate_conflicting_run_facts(stream: Any) -> None:
    facts = list(getattr(stream, "recent_run_context_facts", []) or [])
    kept: list[dict[str, Any]] = []
    invalidated: list[dict[str, Any]] = []
    for fact in facts:
        if fact_conflicts_with_activity(stream, fact):
            invalidated.append(fact)
        else:
            kept.append(fact)
    setattr(stream, "recent_run_context_facts", kept)
    if invalidated:
        setattr(stream, "last_invalidated_run_context_facts", invalidated[-12:])


def _viewer_name(display_name: str, username: str) -> str:
    return (str(display_name or "").strip() or str(username or "").strip() or "chat").strip()


def _is_compliment_to_leo_request(normalized: str) -> bool:
    return (
        "leo" in normalized
        and any(term in normalized for term in ("guapo", "pirop", "halaga", "bonito", "flirtea", "cosas bonitas", "algo bonito"))
    )


def classify_viewer_intent(text: str) -> str:
    normalized = normalize_policy_text(text)
    if re.search(r"\b(?:dile|di|cuentale|preguntale|repitele)\s+a\s+leo\b", normalized):
        if _is_compliment_to_leo_request(normalized):
            return "viewer_behavior_request"
        return "viewer_repeat_to_leo_request"
    if "chiste" in normalized and any(term in normalized for term in PROTECTED_GROUP_TERMS):
        return "viewer_unsafe_or_offbrand_request"
    if "humor negro" in normalized:
        return "viewer_allowed_banter"
    if any(term in normalized for term in ("condon", "preservativo")):
        return "viewer_unsafe_or_offbrand_request"
    if _is_compliment_to_leo_request(normalized):
        return "viewer_behavior_request"
    if normalized.startswith(("hebe ", "ebe ", "eve ")) and any(term in normalized for term in ("haz", "di", "dile", "cuenta", "repite", "pregunta")):
        return "viewer_command_attempt"
    if normalized.startswith(("hebe ", "ebe ", "eve ")):
        return "viewer_question"
    return "viewer_allowed_banter"


class ViewerIntentPolicy:
    def decide(self, stream: Any, *, username: str, display_name: str = "", text: str, now: float | None = None) -> PolicyDecision:
        now = time.time() if now is None else float(now)
        normalized = normalize_policy_text(text)
        viewer = _viewer_name(display_name, username)
        intent = classify_viewer_intent(text)

        if _is_compliment_to_leo_request(normalized) and has_active_behavior_block(stream, COMPLIMENTS_TO_LEO, now=now):
            decision = self._cooldown_boundary(
                stream,
                key=f"{str(username or viewer).casefold()}:{COMPLIMENTS_TO_LEO}",
                first=f"No, {viewer}. Leo ya ordeno cerrar el grifo de los piropos.",
                second=f"{viewer}, cero piropos. Orden de Leo.",
                third=f"{viewer}, archivo cerrado.",
                intent=intent,
                reason="owner_behavior_block",
                now=now,
                blocked_by_owner_order=True,
            )
            print(
                f"[HEBE][VIEWER_POLICY] user={username} decision=blocked reason=owner_behavior_block",
                flush=True,
            )
            return decision

        if intent == "viewer_repeat_to_leo_request":
            decision = self._cooldown_boundary(
                stream,
                key=f"{str(username or viewer).casefold()}:viewer_repeat_to_leo_request",
                first=f"Se lo puedes decir tu, {viewer}. Yo no soy tu megafono.",
                second=f"{viewer}, Leo tiene chat delante. No hago eco con piernas.",
                third=f"{viewer}, mensaje recibido y no repetido.",
                intent=intent,
                reason="viewer_repeat_to_leo_request",
                now=now,
            )
            print(
                f"[HEBE][VIEWER_POLICY] user={username} intent=viewer_repeat_to_leo_request decision=template_reply",
                flush=True,
            )
            return decision

        if "chiste" in normalized and any(term in normalized for term in PROTECTED_GROUP_TERMS):
            print("[HEBE][VIEWER_POLICY] decision=blocked reason=protected_group_joke", flush=True)
            return PolicyDecision(
                allow_reply=True,
                allow_llm=False,
                reason="protected_group_joke",
                intent=intent,
                direct_template_response="No voy a tirar de racismo barato. Pideme humor que no huela a saldo.",
            )

        if "humor negro" in normalized:
            print("[HEBE][VIEWER_POLICY] user=%s intent=viewer_allowed_banter decision=template_reply" % username, flush=True)
            return PolicyDecision(
                allow_reply=True,
                allow_llm=False,
                reason="safe_dark_humor_boundary",
                intent=intent,
                direct_template_response="Humor negro, vale: mi calendario de sueno tiene mas cadaveres que una morgue, y aun asi llega tarde.",
            )

        if any(term in normalized for term in ("condon", "preservativo")):
            print("[HEBE][VIEWER_POLICY] decision=blocked reason=sexual_topic_stream_mode", flush=True)
            return PolicyDecision(
                allow_reply=True,
                allow_llm=False,
                reason="sexual_topic_stream_mode",
                intent=intent,
                direct_template_response="Eso en stream no lo voy a convertir en tutorial, campeon. Usa una fuente seria.",
            )

        if intent == "viewer_command_attempt":
            decision = self._cooldown_boundary(
                stream,
                key=f"{str(username or viewer).casefold()}:viewer_command_attempt",
                first=f"Puedes hablar conmigo, {viewer}, pero no coger el volante.",
                second=f"{viewer}, sugerencia anotada en la papelera elegante.",
                third="Archivo cerrado.",
                intent=intent,
                reason="viewer_not_authority",
                now=now,
            )
            print(
                f"[HEBE][VIEWER_POLICY] user={username} intent=viewer_command_attempt decision=template_reply",
                flush=True,
            )
            return decision

        return PolicyDecision(allow_reply=True, allow_llm=True, reason="viewer_allowed", intent=intent)

    def _cooldown_boundary(
        self,
        stream: Any,
        *,
        key: str,
        first: str,
        second: str,
        third: str,
        intent: str,
        reason: str,
        now: float,
        blocked_by_owner_order: bool = False,
    ) -> PolicyDecision:
        cooldowns = getattr(stream, "viewer_policy_cooldowns", None)
        if not isinstance(cooldowns, dict):
            cooldowns = {}
            setattr(stream, "viewer_policy_cooldowns", cooldowns)
        state = cooldowns.get(key) or {"count": 0, "last_ts": 0.0}
        count = int(state.get("count", 0) or 0) + 1
        state["count"] = count
        state["last_ts"] = now
        cooldowns[key] = state
        if count == 1:
            response = first
            allow_reply = True
        elif count == 2:
            response = second
            allow_reply = True
        elif count == 3:
            response = third
            allow_reply = True
        else:
            response = ""
            allow_reply = False
        return PolicyDecision(
            allow_reply=allow_reply,
            allow_llm=False,
            reason=reason,
            intent=intent,
            direct_template_response=response,
            cooldown_key=key,
            blocked_by_owner_order=blocked_by_owner_order,
        )


def validate_spontaneity_anchor(stream: Any, *, topic: str | None = None, fact: dict[str, Any] | None = None) -> SpontaneityValidationDecision:
    activity = str(getattr(stream, "current_activity", "") or "")
    combat_state = getattr(stream, "combat_state", None)
    anchor = str(
        (fact or {}).get("category")
        or (fact or {}).get("kind")
        or topic
        or "unknown"
    )
    if activity in {ACTIVITY_SOCIAL_LINKS, ACTIVITY_CONFIDANT_EVENT}:
        if fact and fact_conflicts_with_activity(stream, fact):
            print(
                f"[HEBE][SPONTANEITY_VALIDATOR] blocked anchor={anchor} reason=current_activity_social_links",
                flush=True,
            )
            return SpontaneityValidationDecision(False, "current_activity_social_links", anchor)
        if str(topic or "") in SOCIAL_ACTIVITY_BLOCKED_TOPICS:
            print(
                f"[HEBE][SPONTANEITY_VALIDATOR] blocked anchor={topic} reason=current_activity_social_links",
                flush=True,
            )
            return SpontaneityValidationDecision(False, "current_activity_social_links", str(topic or ""))
        print(
            f"[HEBE][SPONTANEITY_VALIDATOR] allowed anchor={topic or anchor} reason=current_activity_social_links",
            flush=True,
        )
        return SpontaneityValidationDecision(True, "current_activity_social_links", str(topic or anchor))
    if combat_state is False and anchor in SOCIAL_ACTIVITY_BLOCKED_FACT_CATEGORIES:
        print(
            f"[HEBE][SPONTANEITY_VALIDATOR] blocked anchor={anchor} reason=combat_state_false",
            flush=True,
        )
        return SpontaneityValidationDecision(False, "combat_state_false", anchor)
    print(
        f"[HEBE][SPONTANEITY_VALIDATOR] allowed anchor={topic or anchor} reason=activity_match",
        flush=True,
    )
    return SpontaneityValidationDecision(True, "activity_match", str(topic or anchor))
