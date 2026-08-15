from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
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
        personal = self._personal_memory_for(title, stream=stream)
        profile = self.profile_store.lookup(
            current_category=title,
            current_game=title,
            current_title=getattr(stream, "current_stream_title", None) if stream is not None else None,
        )
        profile_source = "local_cache" if self._has_specific_profile(profile) else "missing"
        web_reason = "not_needed"

        if not self._has_specific_profile(profile) and (force_web or self.config.effective_web_lookup_enabled):
            service = self._research_service()
            if service is not None and service.search_provider is not None:
                ok, researched, reason = service.research_current_game(
                    current_category=title,
                    current_game=title,
                    current_title=getattr(stream, "current_stream_title", None) if stream is not None else None,
                    force=True,
                )
                profile = researched
                profile_source = "web_cache" if ok and self._has_specific_profile(profile) else "missing"
                web_reason = reason
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
            return f"No tengo memoria personal de {title} ni perfil local. La busqueda web esta activada, pero no hay proveedor configurado; puedo usar una semilla manual o configurar lookup."
        return f"No tengo memoria personal de {title} ni perfil local todavia. Puedo usar una semilla manual o activar lookup web spoiler-safe."


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
