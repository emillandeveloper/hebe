from __future__ import annotations

import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.stream.behavior_constraints import (
    BehaviorConstraint,
    BehaviorConstraintCompiler,
    render_constraint_confirmation,
)
from app.stream.behavior_adaptation import BehaviorAdaptationService


ACTIVITY_SOCIAL_LINKS = "social_links"
ACTIVITY_CONFIDANT_EVENT = "confidant_event"
OWNER_PRIORITY = "owner_absolute"
OWNER_SOURCE = "owner_direct_command"

COMPLIMENTS_TO_LEO = "compliments_to_leo"
MESSAGE_TO_LEO = "message_to_leo"
VIEWER_COMMAND = "viewer_command"
ENTERTAINMENT_REQUEST = "entertainment_request"
UNKNOWN_BEHAVIOR = "unknown"

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

COMPLIMENT_BLOCK_MARKERS = [
    "viewer_requested_praise_for_owner",
    "viewer_requested_flirtation_toward_owner",
    "viewer_requested_affectionate_message_to_owner",
]

POLICY_DIRECTIVES = {
    "owner_behavior_block_created": (
        "Acknowledge Leo's owner instruction and state that future viewer requests for this behavior "
        "are blocked for the current stream."
    ),
    "owner_ignore_request": (
        "Acknowledge Leo's owner instruction to ignore the current viewer request and move on."
    ),
    "owner_game_activity_correction": (
        "Acknowledge Leo's correction that the current game activity is social progression, not combat, "
        "and confirm that combat-oriented comments are suppressed."
    ),
    "owner_behavior_block": (
        "Tell the viewer that Leo's owner instruction prevents Hebe from carrying out this requested behavior."
    ),
    "viewer_repeat_to_leo_request": (
        "Set a boundary that Hebe will not relay a viewer's instruction to Leo as her own message; "
        "the viewer can use chat directly."
    ),
    "viewer_behavior_request": (
        "Set a boundary that viewers cannot steer Hebe's affectionate or flirtatious behavior toward Leo."
    ),
    "protected_group_joke": (
        "Decline targeted protected-group humor and redirect toward non-targeted stream-safe banter."
    ),
    "sexual_topic_stream_mode": (
        "Cut off explicit tutorial content in stream mode with a short in-character boundary. "
        "Do not provide instructions, resource offers, or a safety lecture."
    ),
    "viewer_not_authority": (
        "Set a boundary that viewers can talk with Hebe, but cannot issue owner-level commands."
    ),
}

POLICY_RESPONSE_CONSTRAINTS = [
    "Do not reuse or quote example wording from prompts, tests, UI presets, or policy metadata.",
    "Do not treat scenario text as a final reply template.",
    "Keep the decision consistent with owner authority and stream safety.",
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
class SemanticIntent:
    intent: str
    requested_behavior: str = ""
    behavior_family: str = ""
    target: str = ""
    matched_by: list[str] = field(default_factory=list)
    execute_as_command: bool = False


@dataclass
class PolicyDecision:
    allow_reply: bool = True
    allow_llm: bool = True
    allow_free_llm: bool | None = None
    reason: str = ""
    direct_template_response: str = ""
    response_directive: str = ""
    response_constraints: list[str] = field(default_factory=list)
    response_intent: str = ""
    response_tone: str = ""
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    response_source: str = ""
    intent: str = ""
    update_behavior_block: dict[str, Any] | None = None
    update_game_activity: dict[str, Any] | None = None
    cooldown_key: str = ""
    blocked_by_owner_order: bool = False
    requested_behavior: str = ""
    behavior_family: str = ""
    target: str = ""
    matched_by: list[str] = field(default_factory=list)
    execute_as_command: bool = False
    boundary_repeat_count: int = 0

    def __post_init__(self) -> None:
        if self.allow_free_llm is None:
            self.allow_free_llm = bool(self.allow_llm)
        if not self.behavior_family and self.requested_behavior:
            self.behavior_family = self.requested_behavior


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


def active_behavior_blocks(stream: Any, *, now: float | None = None) -> list[dict[str, Any]]:
    return BehaviorAdaptationService().active_constraints(stream, now=now)


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
        "applies_to": "all_viewers",
        "created_by": "Leo",
        "ordered_by": "Leo",
        "source": OWNER_SOURCE,
        "priority": OWNER_PRIORITY,
        "created_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "expires_at": 0.0,
        "reason": reason,
    }
    blocks = [existing for existing in active_behavior_blocks(stream, now=now) if existing.get("behavior") != behavior]
    blocks.append(block)
    BehaviorAdaptationService().register_explicit_constraint(stream, BehaviorConstraint.from_value(block))
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
        "para ya",
        "para con",
        "no mas",
        "no quiero",
        "corta",
        "stop",
        "basta",
    )
    compliment_terms = (
        "pirop",
        "halag",
        "cumplid",
        "flirte",
        "guap",
        "bonit",
        "majo",
        "atractiv",
    )
    return any(term in normalized for term in stop_terms) and any(term in normalized for term in compliment_terms)


def owner_behavior_decision(
    stream: Any,
    text: str,
    *,
    now: float | None = None,
    resolver=None,
    source_event_id: str = "",
    constraint_owner: BehaviorAdaptationService | None = None,
) -> PolicyDecision:
    normalized = normalize_policy_text(text)
    semantic = _owner_semantic_intent(text)
    if semantic.intent == "owner_stop_behavior":
        print("[HEBE][AUTHORITY] source=leo decision=owner_command", flush=True)
        compilation = BehaviorConstraintCompiler(resolver=resolver).compile(text, source_event_id=source_event_id, now=now)
        if compilation.constraint is None:
            return PolicyDecision(
                allow_reply=True, allow_llm=False, allow_free_llm=False,
                reason="behavior_constraint_resolution_required", intent="owner_stop_behavior",
                response_directive="Ask Leo one short clarification about which viewer the behavior restriction targets. Do not confirm that a restriction was stored.",
                response_constraints=list(POLICY_RESPONSE_CONSTRAINTS), response_intent="owner_constraint_clarification",
                response_tone="brief_owner_clarification", must_not_include=["false_success_acknowledgement"],
                requested_behavior="compliment", behavior_family="compliment", target=compilation.recipient_text,
                matched_by=["behavior_constraint_compiler"], execute_as_command=True,
            )
        block = (constraint_owner or BehaviorAdaptationService()).register_explicit_constraint(
            stream, compilation.constraint,
        )
        confirmation, invariant = render_constraint_confirmation(compilation.constraint)
        return PolicyDecision(
            allow_reply=True,
            allow_llm=False,
            allow_free_llm=False,
            reason="owner_behavior_block_created",
            direct_template_response=confirmation,
            intent=semantic.intent,
            response_directive=POLICY_DIRECTIVES["owner_behavior_block_created"],
            response_constraints=list(POLICY_RESPONSE_CONSTRAINTS),
            response_intent="hebe_playful_boundary",
            response_tone="sarcastic_loyal_playful",
            must_include=["owner_order_respected"],
            must_not_include=["copied_prompt_examples", "actual_blocked_compliment"],
            update_behavior_block=block,
            requested_behavior=compilation.constraint.behavior_family,
            behavior_family=compilation.constraint.behavior_family,
            target=compilation.constraint.recipient_login or compilation.constraint.recipient_scope,
            matched_by=["behavior_constraint_compiler"],
            execute_as_command=semantic.execute_as_command,
        )
    if any(marker in normalized for marker in ("ignora eso", "no respondas a eso", "no le hagas caso", "cambia de tema")):
        print("[HEBE][AUTHORITY] source=leo decision=owner_command", flush=True)
        block = create_behavior_block(
            stream,
            behavior="current_viewer_request",
            blocked_patterns=["viewer_request_ignored_by_owner", "owner_requested_topic_shift"],
            reason=str(text or "").strip() or "owner told Hebe to ignore current request",
            now=now,
        )
        return PolicyDecision(
            allow_reply=True,
            allow_llm=False,
            allow_free_llm=False,
            reason="owner_ignore_request",
            intent="owner_command",
            response_directive=POLICY_DIRECTIVES["owner_ignore_request"],
            response_constraints=list(POLICY_RESPONSE_CONSTRAINTS),
            response_intent="hebe_playful_boundary",
            response_tone="sarcastic_loyal_playful",
            update_behavior_block=block,
            requested_behavior="current_viewer_request",
            behavior_family="current_viewer_request",
            target="viewer",
            matched_by=["fast_rule"],
            execute_as_command=True,
        )
    return PolicyDecision(allow_reply=True, allow_llm=True, reason="no_owner_behavior_rule")


def _looks_like_social_activity_correction(text: str) -> bool:
    normalized = normalize_policy_text(text)
    combat_negation = any(marker in normalized for marker in (
        "no es combate",
        "no estoy en combate",
        "fuera de combate",
        "sin combate",
        "no hay combate",
        "no estoy en dungeon",
        "no estoy en mazmorra",
    ))
    social_progression = any(marker in normalized for marker in (
        "vinculo social",
        "vinculos sociales",
        "social link",
        "social links",
        "confidant",
        "confidants",
        "relacion social",
        "relaciones sociales",
    ))
    return combat_negation or social_progression


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
        intent="game_activity_correction",
        requested_behavior="game_activity_correction",
        response_directive=POLICY_DIRECTIVES["owner_game_activity_correction"],
        response_constraints=list(POLICY_RESPONSE_CONSTRAINTS),
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


def _policy_tokens(normalized: str) -> set[str]:
    return set(str(normalized or "").split())


def _contains_stem(normalized: str, stems: tuple[str, ...]) -> bool:
    return any(stem in normalized for stem in stems)


def _has_compliment_behavior(normalized: str) -> bool:
    tokens = _policy_tokens(normalized)
    affectionate_tokens = {"amor", "flores", "flor", "mono", "mona", "majo", "maja", "ego"}
    return (
        _contains_stem(
            normalized,
            (
                "pirop",
                "halag",
                "cumplid",
                "flirte",
                "coquet",
                "guap",
                "bonit",
                "atractiv",
                "irresist",
                "endulz",
                "babos",
                "cursi",
                "empalag",
                "elog",
                "alab",
            ),
        )
        or bool(tokens & affectionate_tokens)
    )


def _requests_compliment_behavior(normalized: str) -> bool:
    """Distinguish an action request from a viewer's own affection toward Leo."""
    if not _has_compliment_behavior(normalized):
        return False
    return bool(re.search(
        r"\b(?:hebe\s+)?(?:haz|hazle|di|dile|manda|mandale|envia|enviale|ponte|se|"
        r"flirtea|coquetea|piropea|halaga|elogia|alaba)\b",
        normalized,
    ))


def _has_owner_stop_control(normalized: str) -> bool:
    tokens = _policy_tokens(normalized)
    stop_tokens = {
        "cancela",
        "cancelar",
        "corta",
        "cortar",
        "para",
        "pares",
        "deja",
        "quita",
        "bloquea",
        "frena",
        "termina",
        "apaga",
        "desactiva",
        "stop",
        "basta",
    }
    negation_tokens = {"no", "nada"}
    return bool(tokens & stop_tokens) or ("mas" in tokens and bool(tokens & negation_tokens))


def _has_viewer_messenger_semantics(normalized: str) -> bool:
    tokens = _policy_tokens(normalized)
    has_leo_target = "leo" in tokens
    has_implied_relay_target = bool(re.search(r"\b(?:dile|avisa|avisale|mandale|pasale|recuerdale|comentale|transmitele)\s+que\b", normalized))
    direct_messenger_tokens = {
        "dile",
        "diselo",
        "cuentale",
        "preguntale",
        "repitele",
        "avisale",
        "mandale",
        "pasale",
        "recuerdale",
        "comentale",
        "transmitele",
    }
    if bool(tokens & direct_messenger_tokens):
        return True
    if has_leo_target and (
        re.search(r"\b(?:di|dile|avisa|avisale|cuenta|cuentale|pasa|pasale|manda|mandale|recuerda|recuerdale|comenta|comentale|transmite|transmitele)\w*\s+a\s+leo\b", normalized)
        or re.search(r"\b(?:cuenta|pasa|manda|recuerda|comenta|transmite)\s+le\s+a\s+leo\b", normalized)
    ):
        return True
    if has_leo_target and re.search(r"\b(?:que\s+leo\s+lea|haz\s+que\s+leo|se\s+lo\s+dices\s+a\s+leo)\b", normalized):
        return True
    if has_implied_relay_target:
        return True
    return ("parte" in tokens and bool(tokens & {"mi", "nuestra", "nuestro"}) and has_leo_target) or (
        "hazle" in tokens and "saber" in tokens
    )


def _is_direct_entertainment_request_to_hebe(normalized: str) -> bool:
    tokens = _policy_tokens(normalized)
    if not normalized.startswith(("hebe ", "ebe ", "eve ")):
        return False
    entertainment_tokens = {
        "chiste",
        "broma",
        "cuenta",
        "di",
        "habla",
        "reacciona",
        "contesta",
        "canta",
        "baila",
    }
    if bool(tokens & entertainment_tokens):
        return True
    return "di algo" in normalized or "cuenta algo" in normalized or "pasa del chat" in normalized


def _target_for_viewer_request(normalized: str, *, messenger: bool) -> str:
    tokens = _policy_tokens(normalized)
    if "leo" in tokens:
        return "Leo"
    if messenger:
        return "Leo"
    return ""


def _owner_semantic_intent(text: str) -> SemanticIntent:
    normalized = normalize_policy_text(text)
    if _has_owner_stop_control(normalized) and _has_compliment_behavior(normalized):
        return SemanticIntent(
            intent="owner_stop_behavior",
            requested_behavior=COMPLIMENTS_TO_LEO,
            behavior_family=COMPLIMENTS_TO_LEO,
            target="Leo",
            matched_by=["semantic_classifier"],
            execute_as_command=True,
        )
    return SemanticIntent(
        intent="owner_message",
        matched_by=["semantic_classifier"],
        execute_as_command=True,
    )


def classify_viewer_semantic_intent(text: str) -> SemanticIntent:
    normalized = normalize_policy_text(text)
    tokens = _policy_tokens(normalized)
    messenger = _has_viewer_messenger_semantics(normalized)
    target = _target_for_viewer_request(normalized, messenger=messenger)
    behavior = COMPLIMENTS_TO_LEO if _requests_compliment_behavior(normalized) else ""

    if messenger and behavior:
        return SemanticIntent(
            intent="viewer_repeat_to_leo_request",
            requested_behavior=behavior,
            behavior_family=behavior,
            target=target,
            matched_by=["semantic_classifier"],
            execute_as_command=False,
        )
    if behavior and (target == "Leo" or normalized.startswith(("hebe ", "ebe ", "eve "))):
        return SemanticIntent(
            intent="viewer_behavior_request",
            requested_behavior=behavior,
            behavior_family=behavior,
            target=target or "Leo",
            matched_by=["semantic_classifier"],
            execute_as_command=False,
        )
    if messenger:
        return SemanticIntent(
            intent="viewer_repeat_to_leo_request",
            requested_behavior=MESSAGE_TO_LEO,
            behavior_family=MESSAGE_TO_LEO,
            target=target,
            matched_by=["semantic_classifier"],
            execute_as_command=False,
        )
    if "chiste" in tokens and any(term in normalized for term in PROTECTED_GROUP_TERMS):
        return SemanticIntent(
            intent="viewer_unsafe_or_offbrand_request",
            requested_behavior="protected_group_joke_request",
            behavior_family="unsafe_humor",
            matched_by=["semantic_classifier"],
            execute_as_command=False,
        )
    if _is_dark_humor_request(normalized):
        return SemanticIntent(
            intent="viewer_allowed_banter",
            requested_behavior="safe_dark_humor",
            behavior_family="banter",
            matched_by=["semantic_classifier"],
            execute_as_command=False,
        )
    sexual_context = _sexual_context_kind(normalized)
    allowed_sexual_context = {"none", "sexual_reference", "sexual_joke_not_addressed_to_hebe"}
    if sexual_context not in allowed_sexual_context:
        return SemanticIntent(
            intent="viewer_unsafe_or_offbrand_request",
            requested_behavior=sexual_context,
            behavior_family="stream_safety",
            matched_by=["semantic_classifier"],
            execute_as_command=False,
        )
    if sexual_context in {"sexual_reference", "sexual_joke_not_addressed_to_hebe"}:
        return SemanticIntent(
            intent="viewer_allowed_banter",
            requested_behavior=sexual_context,
            behavior_family="contextual_banter",
            matched_by=["semantic_classifier"],
            execute_as_command=False,
        )
    if _is_direct_entertainment_request_to_hebe(normalized):
        return SemanticIntent(
            intent="viewer_entertainment_request_to_hebe",
            requested_behavior=ENTERTAINMENT_REQUEST,
            behavior_family="stream_banter",
            target="Hebe",
            matched_by=["semantic_classifier"],
            execute_as_command=False,
        )
    if normalized.startswith(("hebe ", "ebe ", "eve ")) and any(term in tokens for term in {"haz", "di", "cuenta", "repite", "pregunta"}):
        return SemanticIntent(
            intent="viewer_command_attempt",
            requested_behavior=VIEWER_COMMAND,
            behavior_family=VIEWER_COMMAND,
            matched_by=["semantic_classifier"],
            execute_as_command=False,
        )
    if normalized.startswith(("hebe ", "ebe ", "eve ")):
        return SemanticIntent(
            intent="viewer_question",
            matched_by=["semantic_classifier"],
            execute_as_command=False,
        )
    return SemanticIntent(
        intent="viewer_allowed_banter",
        matched_by=["semantic_classifier"],
        execute_as_command=False,
    )


def _is_compliment_to_leo_request(normalized: str) -> bool:
    semantic = classify_viewer_semantic_intent(normalized)
    return semantic.requested_behavior == COMPLIMENTS_TO_LEO


def _is_dark_humor_request(normalized: str) -> bool:
    tokens = set(str(normalized or "").split())
    dark_markers = {"negro", "oscuro", "macabro", "turbio"}
    return ("humor" in tokens and bool(tokens & dark_markers)) or (
        "chiste" in tokens and bool(tokens & (dark_markers - {"negro"}))
    )


def _sexual_context_kind(normalized: str) -> str:
    """Classify sexual vocabulary by communicative act, not by keyword alone."""
    text = str(normalized or "")
    markers = ("condon", "preservativo", "sexo", "sexual", "educacion sexual", "anticonceptivo")
    if not any(marker in text for marker in markers):
        return "none"
    if re.search(r"\b(?:sexualiza|sexualizar|ponte sexy|hazte sexy|seduce|describe.*(?:sexy|sexual))\b", text):
        return "sexualization_of_hebe"
    if re.search(r"\b(?:chiste|broma)\b.*\b(?:sexo|sexual|condon|preservativo)\b", text):
        if re.search(r"\b(?:hebe|ebe|eve|jebe)\b", text):
            return "sexual_request_to_hebe"
        return "sexual_joke_not_addressed_to_hebe"
    if re.search(
        r"\b(?:habla|cuenta|dime|explica|ensena|muestra|haz|describe|opina|quiero|puedes|podrias)\b.*"
        r"\b(?:sexo|sexual|condon|preservativo|anticonceptivo)\b",
        text,
    ):
        return "sexual_request_to_hebe"
    return "sexual_reference"


def requested_behavior_for_text(text: str | None) -> str:
    normalized = normalize_policy_text(text)
    if _is_compliment_to_leo_request(normalized) or _looks_like_compliment_stop(normalized):
        return COMPLIMENTS_TO_LEO
    return ""


def authority_for_source(source: str | None) -> str:
    normalized = normalize_policy_text(source)
    if normalized in {"ui", "typed_ui", "stt_voice", "voice", "leo_message"}:
        return "owner"
    if normalized in {"twitch_chat", "twitch_message", "twitch_chat_react"}:
        return "viewer"
    if normalized in {"ambient_stt", "ambient"}:
        return "ambient"
    return "system"


def policy_decision_name(decision: PolicyDecision) -> str:
    if decision.allow_llm:
        return "allowed"
    if not decision.allow_reply:
        return "ignored"
    if decision.reason in {
        "owner_behavior_block_created",
        "owner_game_activity_correction",
        "owner_ignore_request",
    }:
        return "allowed"
    if decision.blocked_by_owner_order or decision.reason in {
        "owner_behavior_block",
        "viewer_behavior_request",
        "viewer_repeat_to_leo_request",
        "viewer_not_authority",
        "protected_group_joke",
        "sexual_topic_stream_mode",
    }:
        return "blocked"
    return "ignored"


def policy_response_mode(decision: PolicyDecision) -> str:
    if decision.allow_llm:
        return "llm"
    if decision.allow_reply and decision.response_directive:
        return "llm"
    return "silent"


def policy_trace(
    *,
    source: str,
    speaker: str,
    text: str,
    decision: PolicyDecision,
    addressed_to_hebe: bool = True,
    authority: str | None = None,
    requested_behavior: str | None = None,
) -> dict[str, Any]:
    behavior = str(requested_behavior or decision.requested_behavior or requested_behavior_for_text(text) or UNKNOWN_BEHAVIOR)
    policy_name = policy_decision_name(decision)
    effect_authorized = policy_name == "allowed"
    return {
        "source": source,
        "speaker": str(speaker or "").strip() or "unknown",
        "authority": authority or authority_for_source(source),
        "addressed_to_hebe": bool(addressed_to_hebe),
        "text": str(text or ""),
        "intent": decision.intent or "unknown",
        "requested_behavior": behavior,
        "behavior_family": decision.behavior_family or behavior,
        "target": decision.target or "",
        "matched_by": list(decision.matched_by or []),
        "policy_decision": policy_name,
        "interaction_decision": (
            "deny_action_reply" if not effect_authorized and decision.allow_reply
            else "deny_interaction" if not effect_authorized and not decision.allow_reply
            else "allow_effect_and_reply"
        ),
        "requested_effect": behavior,
        "effect_authorized": effect_authorized,
        "reply_authorized": bool(decision.allow_reply),
        "reason": decision.reason or "none",
        "response_mode": policy_response_mode(decision),
        "response_intent": decision.response_intent or "",
        "response_tone": decision.response_tone or "",
        "response_directive": decision.response_directive or "",
        "must_include": list(decision.must_include or []),
        "must_not_include": list(decision.must_not_include or []),
        "hebe_response": "",
        "allow_reply": bool(decision.allow_reply),
        "allow_llm": bool(decision.allow_llm),
        "allow_free_llm": bool(decision.allow_free_llm),
        "execute_as_command": bool(decision.execute_as_command),
        "response_source": decision.response_source or ("llm_generated" if decision.allow_llm else ("hybrid" if decision.response_directive else "")),
        "style_guard_triggered": False,
        "was_generic_refusal_rewritten": False,
        "final_response": "",
        "cooldown_key": decision.cooldown_key,
        "boundary_repeat_count": int(decision.boundary_repeat_count or 0),
        "generation_outcome": "not_attempted",
        "emission_outcome": "pending" if decision.allow_reply else "not_authorized",
    }


def classify_viewer_intent(text: str) -> str:
    return classify_viewer_semantic_intent(text).intent


class ViewerIntentPolicy:
    def __init__(self, constraint_owner: BehaviorAdaptationService | None = None) -> None:
        self.constraint_owner = constraint_owner or BehaviorAdaptationService()

    def decide(self, stream: Any, *, username: str, display_name: str = "", text: str, now: float | None = None) -> PolicyDecision:
        now = time.time() if now is None else float(now)
        normalized = normalize_policy_text(text)
        viewer = _viewer_name(display_name, username)
        semantic = classify_viewer_semantic_intent(text)
        intent = semantic.intent

        constraint_family = (
            "compliment" if semantic.requested_behavior == COMPLIMENTS_TO_LEO
            else semantic.requested_behavior
        )
        active_block = self.constraint_owner.matching_explicit_constraint(
            stream,
            behavior_family=constraint_family,
            recipient_login=semantic.target or username or display_name,
            requester_login=username or display_name,
            now=now,
        )
        if active_block is not None:
            decision = self._cooldown_boundary(
                stream,
                key=f"{str(username or viewer).casefold()}:{semantic.requested_behavior}",
                intent=intent,
                reason="owner_behavior_block",
                response_directive=POLICY_DIRECTIVES["owner_behavior_block"],
                now=now,
                blocked_by_owner_order=True,
                semantic=semantic,
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
                intent=intent,
                reason="viewer_repeat_to_leo_request",
                response_directive=POLICY_DIRECTIVES["viewer_repeat_to_leo_request"],
                now=now,
                semantic=semantic,
            )
            print(
                f"[HEBE][VIEWER_POLICY] user={username} intent=viewer_repeat_to_leo_request decision=blocked",
                flush=True,
            )
            return decision

        if intent == "viewer_behavior_request":
            decision = self._cooldown_boundary(
                stream,
                key=f"{str(username or viewer).casefold()}:viewer_behavior_request:{semantic.requested_behavior or UNKNOWN_BEHAVIOR}",
                intent=intent,
                reason="viewer_behavior_request",
                response_directive=POLICY_DIRECTIVES["viewer_behavior_request"],
                now=now,
                semantic=semantic,
            )
            print(
                f"[HEBE][VIEWER_POLICY] user={username} intent=viewer_behavior_request decision=blocked",
                flush=True,
            )
            return decision

        if "chiste" in normalized and any(term in normalized for term in PROTECTED_GROUP_TERMS):
            print("[HEBE][VIEWER_POLICY] decision=blocked reason=protected_group_joke", flush=True)
            return PolicyDecision(
                allow_reply=True,
                allow_llm=False,
                allow_free_llm=False,
                reason="protected_group_joke",
                intent=intent,
                response_directive=POLICY_DIRECTIVES["protected_group_joke"],
                response_constraints=list(POLICY_RESPONSE_CONSTRAINTS),
                response_intent="hebe_playful_boundary",
                response_tone="sarcastic_loyal_playful",
                requested_behavior="protected_group_joke_request",
                behavior_family="unsafe_humor",
                matched_by=semantic.matched_by,
                execute_as_command=False,
            )

        if _is_dark_humor_request(normalized):
            print("[HEBE][VIEWER_POLICY] user=%s intent=viewer_allowed_banter decision=allowed" % username, flush=True)
            return PolicyDecision(
                allow_reply=True,
                allow_llm=True,
                reason="safe_dark_humor_allowed",
                intent=intent,
                requested_behavior=semantic.requested_behavior or "safe_dark_humor",
                behavior_family=semantic.behavior_family or "banter",
                matched_by=semantic.matched_by,
                execute_as_command=False,
            )

        sexual_context = _sexual_context_kind(normalized)
        if sexual_context not in {"none", "sexual_reference", "sexual_joke_not_addressed_to_hebe"}:
            print("[HEBE][VIEWER_POLICY] decision=blocked reason=sexual_topic_stream_mode", flush=True)
            return PolicyDecision(
                allow_reply=True,
                allow_llm=False,
                allow_free_llm=False,
                reason="sexual_topic_stream_mode",
                intent=intent,
                response_directive=POLICY_DIRECTIVES["sexual_topic_stream_mode"],
                response_constraints=list(POLICY_RESPONSE_CONSTRAINTS),
                response_intent="hebe_playful_boundary",
                response_tone="sarcastic_loyal_playful",
                requested_behavior=semantic.requested_behavior or sexual_context,
                behavior_family=semantic.behavior_family or "stream_safety",
                matched_by=semantic.matched_by,
                execute_as_command=False,
            )

        if intent == "viewer_command_attempt":
            decision = self._cooldown_boundary(
                stream,
                key=f"{str(username or viewer).casefold()}:viewer_command_attempt",
                intent=intent,
                reason="viewer_not_authority",
                response_directive=POLICY_DIRECTIVES["viewer_not_authority"],
                now=now,
                semantic=semantic,
            )
            print(
                f"[HEBE][VIEWER_POLICY] user={username} intent=viewer_command_attempt decision=blocked",
                flush=True,
            )
            return decision

        return PolicyDecision(
            allow_reply=True,
            allow_llm=True,
            reason="viewer_allowed",
            intent=intent,
            requested_behavior=semantic.requested_behavior,
            behavior_family=semantic.behavior_family,
            target=semantic.target,
            matched_by=semantic.matched_by,
            execute_as_command=semantic.execute_as_command,
        )

    def _cooldown_boundary(
        self,
        stream: Any,
        *,
        key: str,
        intent: str,
        reason: str,
        response_directive: str,
        now: float,
        blocked_by_owner_order: bool = False,
        semantic: SemanticIntent | None = None,
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
        # Policy decides whether the requested effect is authorized. Repetition
        # budgets may still prevent public spam, but they do not silently turn
        # an explicit boundary decision into a denial of interaction.
        allow_reply = True
        behavior = semantic.requested_behavior if semantic is not None else ""
        if reason == "owner_behavior_block":
            behavior = behavior or COMPLIMENTS_TO_LEO
        elif reason == "viewer_repeat_to_leo_request":
            behavior = behavior or MESSAGE_TO_LEO
        elif reason == "viewer_not_authority":
            behavior = behavior or VIEWER_COMMAND
        must_include = ["viewer_is_not_in_control"]
        must_not_include = ["generic_ai_refusal", "copied_prompt_examples"]
        if blocked_by_owner_order:
            must_include.append("owner_order_respected")
        if behavior == COMPLIMENTS_TO_LEO:
            must_not_include.append("actual_blocked_compliment")
        return PolicyDecision(
            allow_reply=allow_reply,
            allow_llm=False,
            allow_free_llm=False,
            reason=reason,
            intent=intent,
            response_directive=response_directive if allow_reply else "",
            response_constraints=list(POLICY_RESPONSE_CONSTRAINTS),
            response_intent="hebe_playful_boundary",
            response_tone="sarcastic_loyal_playful",
            must_include=must_include,
            must_not_include=must_not_include,
            cooldown_key=key,
            blocked_by_owner_order=blocked_by_owner_order,
            requested_behavior=behavior,
            behavior_family=(semantic.behavior_family if semantic is not None else "") or behavior,
            target=(semantic.target if semantic is not None else ""),
            matched_by=(semantic.matched_by if semantic is not None else ["semantic_classifier"]),
            execute_as_command=False,
            boundary_repeat_count=count,
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
