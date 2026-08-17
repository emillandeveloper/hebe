from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.stream.game_profiles import GameProfile, GameProfileStore
from app.stream.game_research import GameKnowledgeResearchConfig, GameKnowledgeResearchService


@dataclass(frozen=True)
class GameKnowledgeConfig:
    web_lookup_enabled: bool = False
    game_profile_web_lookup_enabled: bool = False
    profile_cache_days: int = 30
    no_spoilers: bool = True

    @classmethod
    def from_env(cls) -> "GameKnowledgeConfig":
        return cls(
            web_lookup_enabled=_env_bool("HEBE_WEB_LOOKUP_ENABLED", False),
            game_profile_web_lookup_enabled=_env_bool(
                "HEBE_GAME_PROFILE_WEB_LOOKUP_ENABLED",
                _env_bool("HEBE_GAME_RESEARCH_ENABLED", False),
            ),
            profile_cache_days=int(os.getenv("HEBE_GAME_PROFILE_CACHE_DAYS", os.getenv("HEBE_GAME_RESEARCH_CACHE_DAYS", "30")) or "30"),
            no_spoilers=_env_bool("HEBE_GAME_PROFILE_NO_SPOILERS", True),
        )

    @property
    def effective_web_lookup_enabled(self) -> bool:
        return self.web_lookup_enabled and self.game_profile_web_lookup_enabled


class GameKnowledgeStatus(StrEnum):
    KNOWN = "KNOWN"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"
    LOOKUP_SUCCEEDED = "LOOKUP_SUCCEEDED"
    LOOKUP_FAILED = "LOOKUP_FAILED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class GameKnowledgeQuery:
    detected: bool
    game_title: str = ""
    intent: str = ""
    asks_for_recommendation: bool = False
    asks_for_gameplay: bool = False


@dataclass(frozen=True)
class GameFactualGroundingValidation:
    passed: bool
    game_knowledge_status: str
    claims_grounded: list[str] = field(default_factory=list)
    claims_ungrounded: list[str] = field(default_factory=list)
    ungrounded_claim_blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "game_knowledge_status": self.game_knowledge_status,
            "claims_grounded": list(self.claims_grounded),
            "claims_ungrounded": list(self.claims_ungrounded),
            "ungrounded_claim_blocked": self.ungrounded_claim_blocked,
        }


@dataclass(frozen=True)
class GameKnowledgeResult:
    game_title: str
    response_mode: str
    personal_memory: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    profile_source: str = "missing"
    web_lookup_enabled: bool = False
    web_lookup_reason: str = ""
    missing: list[str] = field(default_factory=list)
    fallback_text: str = ""
    game_knowledge_status: str = GameKnowledgeStatus.UNKNOWN.value
    claims: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    lookup_used: bool = False
    lookup_attempted: bool = False
    ambiguous_candidates: list[str] = field(default_factory=list)

    def to_state_changes(self) -> dict[str, Any]:
        return {
            "game_title": self.game_title,
            "response_mode": self.response_mode,
            "personal_memory": self.personal_memory,
            "game_profile": self.profile,
            "profile_source": self.profile_source,
            "web_lookup_enabled": self.web_lookup_enabled,
            "web_lookup_reason": self.web_lookup_reason,
            "missing": self.missing,
            "game_knowledge_status": self.game_knowledge_status,
            "allowed_claims": list(self.claims),
            "evidence": list(self.evidence),
            "evidence_count": len(self.evidence),
            "lookup_used": self.lookup_used,
            "lookup_attempted": self.lookup_attempted,
            "claim_count": len(self.claims),
            "ungrounded_claim_blocked": False,
            "ambiguous_candidates": list(self.ambiguous_candidates),
        }


class GameKnowledgeResolver:
    def __init__(
        self,
        *,
        profile_store: GameProfileStore | None = None,
        research_service: GameKnowledgeResearchService | None = None,
        config: GameKnowledgeConfig | None = None,
        run_service: Any | None = None,
    ) -> None:
        self.profile_store = profile_store or GameProfileStore()
        self.research_service = research_service
        self.config = config or GameKnowledgeConfig.from_env()
        self.run_service = run_service

    def resolve(self, *, game: str | None = None, stream: Any | None = None, force_web: bool = False) -> GameKnowledgeResult:
        title = self._resolve_game_title(game=game, stream=stream)
        candidates = self.profile_store.matching_profiles(title)
        exact_candidates = [
            profile for profile in candidates
            if _normalize(profile.canonical_title) == _normalize(title)
            or _normalize(title) in {_normalize(alias) for alias in profile.aliases}
        ]
        if len(exact_candidates) != 1 and len(candidates) > 1:
            names = sorted({profile.canonical_title for profile in candidates})
            return GameKnowledgeResult(
                game_title=title,
                response_mode="ambiguous",
                web_lookup_enabled=self.config.effective_web_lookup_enabled or bool(force_web),
                web_lookup_reason="ambiguous_title",
                missing=["unambiguous_game_title"],
                fallback_text="Ese título puede referirse a más de un juego; dime cuál tienes en mente.",
                game_knowledge_status=GameKnowledgeStatus.AMBIGUOUS.value,
                ambiguous_candidates=names,
            )
        personal = self._personal_memory_for(title, stream=stream)
        profile = self.profile_store.lookup(
            current_category=title,
            current_game=title,
            current_title=getattr(stream, "current_stream_title", None) if stream is not None else None,
        )
        profile_source = "local_cache" if self._has_specific_profile(profile) else "missing"
        web_reason = "not_needed"
        lookup_attempted = False

        if not self._has_specific_profile(profile) and (force_web or self.config.effective_web_lookup_enabled):
            service = self._research_service()
            if service is not None and service.search_provider is not None:
                search_count_before = int(getattr(service, "search_count", 0) or 0)
                ok, researched, reason = service.research_current_game(
                    current_category=title,
                    current_game=title,
                    current_title=getattr(stream, "current_stream_title", None) if stream is not None else None,
                    force=True,
                )
                profile = researched
                profile_source = "web_cache" if ok and self._has_specific_profile(profile) else "missing"
                web_reason = reason
                lookup_attempted = int(getattr(service, "search_count", 0) or 0) > search_count_before
            else:
                web_reason = "web_lookup_not_configured"
        elif not self._has_specific_profile(profile):
            web_reason = "web_lookup_disabled"

        has_personal = bool(personal)
        has_profile = self._has_specific_profile(profile)
        if has_personal and has_profile:
            mode = "memory_plus_profile"
        elif has_profile:
            mode = "profile_only"
        else:
            mode = "missing"

        missing = []
        if not has_personal:
            missing.append("personal_session_memory")
        if not has_profile:
            missing.append("local_game_profile")

        profile_payload = self._profile_payload(profile) if has_profile else {}
        claims, evidence = self._profile_claims(profile_payload, profile_source=profile_source)
        if has_profile:
            if profile_source == "web_cache" and lookup_attempted:
                knowledge_status = GameKnowledgeStatus.LOOKUP_SUCCEEDED.value
            elif len(claims) >= 3:
                knowledge_status = GameKnowledgeStatus.KNOWN.value
            else:
                knowledge_status = GameKnowledgeStatus.PARTIAL.value
        elif lookup_attempted:
            knowledge_status = GameKnowledgeStatus.LOOKUP_FAILED.value
        else:
            knowledge_status = GameKnowledgeStatus.UNKNOWN.value
        fallback = self._fallback_text(
            title=title,
            mode=mode,
            personal=personal,
            profile=profile_payload,
            web_reason=web_reason,
        )
        return GameKnowledgeResult(
            game_title=title,
            response_mode=mode,
            personal_memory=personal,
            profile=profile_payload,
            profile_source=profile_source,
            web_lookup_enabled=self.config.effective_web_lookup_enabled or bool(force_web),
            web_lookup_reason=web_reason,
            missing=missing,
            fallback_text=fallback,
            game_knowledge_status=knowledge_status,
            claims=claims,
            evidence=evidence,
            lookup_used=lookup_attempted,
            lookup_attempted=lookup_attempted,
        )

    def _research_service(self) -> GameKnowledgeResearchService | None:
        if self.research_service is not None:
            self.research_service.config = GameKnowledgeResearchConfig(
                enabled=self.config.effective_web_lookup_enabled,
                provider=self.research_service.config.provider,
                api_key=self.research_service.config.api_key,
                cache_days=self.config.profile_cache_days,
            )
            return self.research_service
        return GameKnowledgeResearchService(
            store=self.profile_store,
            config=GameKnowledgeResearchConfig(
                enabled=self.config.effective_web_lookup_enabled,
                provider=os.getenv("HEBE_GAME_RESEARCH_PROVIDER", "").strip(),
                api_key=os.getenv("HEBE_GAME_RESEARCH_API_KEY", "").strip(),
                cache_days=self.config.profile_cache_days,
            ),
        )

    def _resolve_game_title(self, *, game: str | None, stream: Any | None) -> str:
        explicit = _clean_game_title(game)
        if explicit and explicit not in {"este juego", "el juego"}:
            return explicit
        if stream is not None:
            primer = getattr(stream, "session_primer", None) or {}
            for value in (
                getattr(stream, "current_game", None),
                getattr(stream, "current_category", None),
                primer.get("game") if isinstance(primer, dict) else None,
            ):
                cleaned = _clean_game_title(value)
                if cleaned:
                    return cleaned
        return "este juego"

    def _personal_memory_for(self, title: str, *, stream: Any | None) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if stream is not None:
            objective = str(getattr(stream, "current_run_objective", "") or "").strip()
            location = str(getattr(stream, "current_run_location", "") or "").strip()
            facts = [item for item in (getattr(stream, "recent_run_context_facts", []) or []) if isinstance(item, dict)]
            if objective:
                data["current_objective"] = objective
            if location:
                data["current_location"] = location
            if facts:
                data["run_context_facts"] = facts[:5]
        runs = self.run_service
        if runs is not None and title and title != "este juego":
            identity = runs.repository.resolve_identity(title)
            run_id = str(getattr(stream, "active_game_run_id", "") or "") if stream is not None else ""
            run = runs.repository.get_run(run_id) if run_id else None
            if run is None or run.game_id != identity.game_id:
                run = next(iter(runs.repository.list_runs(
                    game_id=identity.game_id,owner_id="leo",statuses=("ACTIVE",),
                )),None)
            if run is not None:
                state = runs.state(run.id)
                durable = {
                    "current_location": state.get("current_location") or "",
                    "current_objective": state.get("current_objective") or "",
                    "last_confirmed_progress": state.get("last_confirmed_progress") or "",
                    "party_members": state.get("party_members") or [],
                    "challenge": state.get("challenge") or "",
                    "run_id": run.id,
                }
                if any(value for key,value in durable.items() if key != "run_id"):
                    data["canonical_run"] = durable
        return {key: value for key, value in data.items() if value}

    def _has_specific_profile(self, profile: GameProfile) -> bool:
        return bool(profile and profile.game_slug != "generic_jrpg_rpg")

    def _profile_payload(self, profile: GameProfile) -> dict[str, Any]:
        payload = profile.compact_prompt_context()
        payload.update(
            {
                "game_id": profile.game_slug,
                "title": profile.canonical_title,
                "safe_summary": profile.general_non_spoiler_summary,
                "spoiler_policy": profile.spoiler_policy,
                "known_terms": sorted(set(profile.safe_comment_topics + profile.gameplay_systems_non_spoiler))[:12],
                "source": ", ".join(profile.sources_used) or "local_profile_store",
                "last_updated_at": profile.updated_at,
            }
        )
        return payload

    def _profile_claims(self, profile: dict[str, Any], *, profile_source: str) -> tuple[list[str], list[dict[str, Any]]]:
        if not profile:
            return [], []
        claims: list[str] = []
        claims.extend(f"genre={value}" for value in profile.get("genres") or [] if str(value).strip())
        claims.extend(
            f"gameplay_system={value}"
            for value in profile.get("gameplay_systems_non_spoiler") or []
            if str(value).strip()
        )
        summary = str(profile.get("safe_summary") or profile.get("general_non_spoiler_summary") or "").strip()
        if summary:
            claims.append(f"summary={summary}")
        sources = [item.strip() for item in str(profile.get("source") or "").split(",") if item.strip()]
        evidence = [
            {
                "evidence_type": "confirmed_game_knowledge",
                "evidence_id": f"{profile_source}:{index}",
                "exact_supporting_text": claim,
                "confidence": float(profile.get("confidence", 0.0) or 0.0),
                "provenance": list(sources),
            }
            for index, claim in enumerate(claims)
        ]
        return claims, evidence

    def _fallback_text(self, *, title: str, mode: str, personal: dict[str, Any], profile: dict[str, Any], web_reason: str) -> str:
        if mode == "memory_plus_profile":
            summary = profile.get("safe_summary") or profile.get("general_non_spoiler_summary") or "tengo un perfil spoiler-safe."
            personal_bits = _personal_summary(personal)
            return f"De {title}: {summary} En memoria del canal tengo: {personal_bits}"
        if mode == "profile_only":
            genres = ", ".join(profile.get("genres") or [])
            summary = profile.get("safe_summary") or profile.get("general_non_spoiler_summary") or "tengo un perfil spoiler-safe."
            genre_text = f" ({genres})" if genres else ""
            return f"De {title}{genre_text}: {summary} No tengo todavia memoria de donde lo dejamos en stream."
        if web_reason == "web_lookup_not_configured":
            return f"No tengo datos fiables de {title} ahora mismo y la consulta disponible no ha podido ejecutarse."
        if web_reason.startswith("research_failed") or web_reason == "research_no_supported_evidence":
            return f"He intentado consultar {title}, pero no he obtenido información fiable suficiente."
        return f"No tengo datos fiables de {title} ahora mismo; prefiero no describirlo ni recomendarlo a ciegas."


def classify_game_knowledge_query(text: str, *, current_game: str = "") -> GameKnowledgeQuery:
    raw = str(text or "").strip()
    normalized = _normalize(raw)
    recommendation = bool(re.search(
        r"\b(?:recomiend\w*|recomendari\w*|merece la pena|vale la pena|que tal esta)\b",
        normalized,
    ))
    gameplay = bool(re.search(
        r"\b(?:gameplay|jugabilidad|mecanicas?|combate|como se juega|sistema de juego)\b",
        normalized,
    ))
    factual = bool(re.search(
        r"\b(?:de cuando|que sabes|de que trata|genero|ano|fecha|plataforma|desarrollador|"
        r"quien lo hizo|argumento|historia|recepcion|criticas|disponible|dificultad|salio|lanzamiento)\b",
        normalized,
    ))
    if not (recommendation or gameplay or factual):
        return GameKnowledgeQuery(False)

    cleaned = re.sub(r"^\s*@?(?:hebe(?:nifelheim)?|ebe|eve|jebe|heve)\s*[,;:]?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*[,;:]?\s*(?:hebe|ebe|eve|jebe|heve)\s*[?.!]*\s*$", "", cleaned, flags=re.IGNORECASE)
    title = ""
    patterns = (
        r"(?:me\s+)?recomiend\w*\s+(?:el\s+juego\s+|el\s+|la\s+)?(.+)$",
        r"(?:que\s+sabes\s+(?:de|sobre)|de\s+que\s+trata|como\s+es)\s+(.+)$",
        r"(?:gameplay|jugabilidad|mecanicas?|combate|genero|dificultad|recepcion)\s+(?:de|del)\s+(.+)$",
        r"(?:de\s+cuando\s+es|que\s+genero\s+(?:es|tiene)|cuando\s+(?:salio|se\s+lanzo)|"
        r"quien\s+(?:desarrollo|hizo)|que\s+dificultad\s+tiene)\s+(?:el\s+juego\s+)?(.+)$",
        r"(?:en\s+que\s+plataformas?\s+(?:esta|salio)|donde\s+esta\s+disponible)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            title = _clean_game_title(match.group(1))
            break
    current_reference = bool(re.search(
        r"\b(?:este juego|el juego|juego que estoy jugando|juego actual)\b",
        normalized,
    ))
    if current_reference:
        title = str(current_game or "").strip()
    intent = "recommendation" if recommendation else "gameplay" if gameplay else "factual"
    return GameKnowledgeQuery(True, title, intent, recommendation, gameplay)


_FACTUAL_CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "genre": (r"\b(?:jrpg|rpg|shooter|plataformas?|aventura|estrategia|terror|rogueli\w*|metroidvania|simulador)\b",),
    "year": (r"\b(?:19|20)\d{2}\b", r"\b(?:salio|lanzado|publicado)\b"),
    "platform": (r"\b(?:pc|windows|switch|playstation|ps[1-5]|xbox|steam|android|ios|nintendo)\b",),
    "developer": (r"\b(?:desarrollad[oa]|creado por|estudio|developer)\b",),
    "plot": (r"\b(?:trata de|historia|argumento|protagonista|la trama)\b",),
    "gameplay": (r"\b(?:combate|gameplay|jugabilidad|mecanica|mecanicas|turnos|tiempo real|exploracion)\b",),
    "reception": (r"\b(?:critica|criticas|recepcion|aclamad|valorad|resenas)\b",),
    "availability": (r"\b(?:disponible|se puede jugar|esta en)\b",),
    "difficulty": (r"\b(?:dificil|facil|exigente|accesible|dificultad)\b",),
    "features": (r"\b(?:multijugador|cooperativo|modo online|mundo abierto|crafting|construccion|personalizacion)\b",),
}


def validate_game_factual_grounding(text: str, contract: dict[str, Any] | None) -> GameFactualGroundingValidation:
    knowledge = dict(contract or {})
    status = str(knowledge.get("game_knowledge_status") or "").upper()
    if not knowledge.get("query_detected"):
        return GameFactualGroundingValidation(True, status or GameKnowledgeStatus.UNKNOWN.value)
    allowed_claims = [str(item) for item in knowledge.get("allowed_claims") or [] if str(item).strip()]
    allowed_norm = [_normalize(item) for item in allowed_claims]
    candidate = str(text or "").strip()
    candidate_norm = _normalize(candidate)
    uncertainty = bool(re.search(
        r"\b(?:no tengo (?:datos|informacion|base)|no conozco|no puedo recomendar|no me basta|"
        r"sin datos|sin informacion|prefiero no inventar|necesito confirmar|dime cual|no he obtenido)\b",
        candidate_norm,
    ))
    grounded: list[str] = []
    ungrounded: list[str] = []
    for category, patterns in _FACTUAL_CATEGORY_PATTERNS.items():
        if not any(re.search(pattern, candidate_norm) for pattern in patterns):
            continue
        supported_rows = [claim for claim in allowed_norm if category in _claim_categories(claim)]
        if supported_rows and _claim_value_supported(candidate_norm, supported_rows, category=category):
            grounded.append(category)
        elif not uncertainty or _claim_survives_uncertainty(candidate_norm, patterns):
            ungrounded.append(category)
    recommendation_patterns = (r"\b(?:te lo recomiendo|si te lo recomiendo|merece la pena|vale la pena)\b",)
    recommendation = any(re.search(pattern, candidate_norm) for pattern in recommendation_patterns)
    if recommendation and (not uncertainty or _claim_survives_uncertainty(candidate_norm, recommendation_patterns)):
        supported_categories = {
            category
            for claim in allowed_norm
            for category in _claim_categories(claim)
        }
        if status in {GameKnowledgeStatus.KNOWN.value, GameKnowledgeStatus.LOOKUP_SUCCEEDED.value} and len(supported_categories - {"other"}) >= 2:
            grounded.append("recommendation")
        else:
            ungrounded.append("recommendation")
    return GameFactualGroundingValidation(
        passed=not ungrounded,
        game_knowledge_status=status or GameKnowledgeStatus.UNKNOWN.value,
        claims_grounded=sorted(set(grounded)),
        claims_ungrounded=sorted(set(ungrounded)),
        ungrounded_claim_blocked=bool(ungrounded),
    )


def _claim_category(claim: str) -> str:
    normalized = _normalize(claim)
    prefix = normalized.split(" ", 1)[0] if normalized else ""
    if prefix in {"genre", "genres"}:
        return "genre"
    if prefix in {"release", "release_date", "release_year", "release_japan", "release_west", "year", "date"} or re.search(r"\b(?:19|20)\d{2}\b", normalized):
        return "year"
    if prefix in {"platform", "platforms", "platform_ids"}:
        return "platform"
    if prefix in {"developer", "developer_name", "developed_by", "studio"}:
        return "developer"
    if prefix in {"summary", "plot", "story", "synopsis"}:
        return "plot"
    if prefix in {"gameplay_system", "mechanic", "mechanics", "combat", "combat_type", "gameplay"}:
        return "gameplay"
    if prefix in {"reception", "review", "review_score"}:
        return "reception"
    if prefix in {"availability"}:
        return "availability"
    if prefix in {"difficulty"}:
        return "difficulty"
    if prefix in {"feature", "features", "characteristic"}:
        return "features"
    return "other"


def _claim_categories(claim: str) -> set[str]:
    categories = {_claim_category(claim)}
    for category, patterns in _FACTUAL_CATEGORY_PATTERNS.items():
        if any(re.search(pattern, claim) for pattern in patterns):
            categories.add(category)
    return categories


def _claim_value_supported(candidate: str, claims: list[str], *, category: str) -> bool:
    combined = " ".join(claims)
    if category == "year":
        candidate_years = set(re.findall(r"\b(?:19|20)\d{2}\b", candidate))
        evidence_years = set(re.findall(r"\b(?:19|20)\d{2}\b", combined))
        return bool(candidate_years and candidate_years <= evidence_years)
    stop = {
        "genre", "genres", "summary", "gameplay", "system", "mechanic", "combat", "plot", "story",
        "es", "un", "una", "de", "del", "la", "el", "y", "con", "que", "juego", "tiene", "ofrece",
    }
    candidate_tokens = {item for item in candidate.split() if len(item) > 2 and item not in stop}
    evidence_tokens = {item for item in combined.split() if len(item) > 2 and item not in stop}
    return bool(candidate_tokens & evidence_tokens)


def _claim_survives_uncertainty(candidate: str, patterns: tuple[str, ...]) -> bool:
    for marker in ("pero", "aunque", "sin embargo", "parece", "diria que"):
        match = re.search(rf"\b{marker}\b", candidate)
        if match and any(re.search(pattern, candidate[match.end():]) for pattern in patterns):
            return True
    return False


def _normalize(value: str | None) -> str:
    raw = "".join(
        char for char in unicodedata.normalize("NFKD", str(value or "").casefold())
        if not unicodedata.combining(char)
    )
    return " ".join(re.sub(r"[^a-z0-9_]+", " ", raw).split())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _clean_game_title(value: str | None) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\s*(?:de|sobre)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:hebe|eve|ebe|jebe)\s*$", "", text, flags=re.IGNORECASE).strip(" ,.;:")
    return " ".join(text.split()).strip(" ,.;:?!")


def _personal_summary(personal: dict[str, Any]) -> str:
    latest = personal.get("latest_session") or {}
    for key in ("next_time_plan", "current_objective", "end_summary", "current_location"):
        value = str(latest.get(key) or personal.get(key) or "").strip()
        if value:
            return value
    facts = personal.get("run_context_facts") or []
    if facts:
        return str(facts[0].get("summary") or facts[0].get("text") or "").strip() or "contexto reciente de la run"
    return "contexto reciente de stream"
