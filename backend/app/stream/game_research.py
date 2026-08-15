from __future__ import annotations

import json
import os
import re
import time
import urllib.request
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from app.stream.game_profiles import GameProfile, GameProfileStore


FORBIDDEN_RESEARCH_TERMS = (
    "walkthrough",
    "boss weakness",
    "boss weaknesses",
    "puzzle solution",
    "ending",
    "final boss",
    "death",
    "dies",
    "plot twist",
    "spoiler",
)


class GameSearchProvider(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class GameKnowledgeResearchConfig:
    enabled: bool = False
    provider: str = ""
    api_key: str = ""
    cache_days: int = 30

    @classmethod
    def from_env(cls) -> "GameKnowledgeResearchConfig":
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        configured_provider = os.getenv("HEBE_GAME_RESEARCH_PROVIDER", "").strip()
        if not configured_provider and openai_key:
            configured_provider = "openai_responses_web_search"
        explicit_enabled = os.getenv("HEBE_GAME_RESEARCH_ENABLED", "").strip().lower()
        return cls(
            enabled=(explicit_enabled in ("1", "true", "yes", "on")) if explicit_enabled else bool(configured_provider),
            provider=configured_provider,
            api_key=os.getenv("HEBE_GAME_RESEARCH_API_KEY", "").strip() or openai_key,
            cache_days=int(os.getenv("HEBE_GAME_RESEARCH_CACHE_DAYS", "30") or "30"),
        )


class HttpJsonSearchProvider:
    """Small configurable JSON search adapter.

    HEBE_GAME_RESEARCH_PROVIDER can point to an HTTP endpoint that accepts
    {"query": "..."} and returns either {"results": [...]} or a list.
    """

    def __init__(self, endpoint: str, api_key: str = "") -> None:
        self.endpoint = endpoint
        self.api_key = api_key

    def search(self, query: str) -> list[dict[str, Any]]:
        body = json.dumps({"query": query}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("results", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, dict)]


class OpenAIResponsesWebSearchProvider:
    """Provider-neutral adapter over Responses API web search.

    It returns fact rows in the same format as every other game research
    provider, so no cognition component depends on OpenAI response objects.
    """

    provider_name = "openai_responses_web_search"

    def __init__(self, api_key: str, *, model: str | None = None, base_url: str | None = None, timeout: float | None = None) -> None:
        self.api_key = str(api_key or "").strip()
        self.model = model or os.getenv("HEBE_GAME_RESEARCH_OPENAI_MODEL", os.getenv("HEBE_OPENAI_MODEL", "gpt-5-mini"))
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = float(timeout or os.getenv(
            "HEBE_GAME_RESEARCH_PROVIDER_TIMEOUT_SECONDS",
            os.getenv("HEBE_GAME_RESEARCH_TIMEOUT_SECONDS", "10"),
        ))
        self.available = bool(self.api_key)

    def search(
        self,
        query: str,
        constraints: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        cancellation: threading.Event | None = None,
    ) -> list[dict[str, Any]]:
        if not self.available:
            raise RuntimeError("openai_api_key_missing")
        if cancellation is not None and cancellation.is_set():
            raise RuntimeError("research_cancelled")
        policy = dict(constraints or {})
        prompt = (
            "Research the named video game on the public web. Return JSON only as an array of independent, "
            "spoiler-safe facts. Each object must contain claim, source_title, url, excerpt, confidence, "
            "source_type, general_mechanic, and spoiler_classification. Use only facts directly supported by "
            "the cited page. Do not include future story events, identities, bosses, solutions, endings, or "
            "future unlocks. Prefer official pages/manuals and established reference sources. "
            f"Constraints: {json.dumps(policy, ensure_ascii=False)}. Query: {query}"
        )
        body = json.dumps({
            "model": self.model,
            "tools": [{"type": "web_search"}],
            "input": prompt,
            "max_output_tokens": 1800,
            "include": ["web_search_call.action.sources"],
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=float(timeout or self.timeout)) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if cancellation is not None and cancellation.is_set():
            raise RuntimeError("research_cancelled")
        output_text = str(payload.get("output_text") or "")
        cited_urls: set[str] = set()
        content_text = ""
        for item in payload.get("output") or []:
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"}:
                    content_text += str(content.get("text") or "")
                    for annotation in content.get("annotations") or []:
                        if annotation.get("type") == "url_citation" and annotation.get("url"):
                            cited_urls.add(str(annotation["url"]))
        output_text = output_text or content_text
        match = re.search(r"\[[\s\S]*\]", output_text)
        if not match:
            return []
        try:
            rows = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        accepted = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("claim") or not row.get("url"):
                continue
            # A model-authored URL is not provenance. It must also occur in the
            # Responses API's url_citation annotations.
            if cited_urls and str(row.get("url")) not in cited_urls:
                continue
            if not cited_urls:
                continue
            row["provider"] = self.provider_name
            row["citation_verified"] = True
            accepted.append(row)
        return accepted


class GameKnowledgeResearchService:
    def __init__(
        self,
        *,
        store: GameProfileStore,
        config: GameKnowledgeResearchConfig | None = None,
        search_provider: GameSearchProvider | None = None,
        now_fn=None,
    ) -> None:
        self.store = store
        self.config = config or GameKnowledgeResearchConfig.from_env()
        self.search_provider = search_provider or self._provider_from_config(self.config)
        self._now_fn = now_fn
        self.search_count = 0

    def research_current_game(
        self,
        *,
        current_category: str | None,
        current_title: str | None = None,
        current_game: str | None = None,
        force: bool = False,
    ) -> tuple[bool, GameProfile, str]:
        existing = self.store.lookup(
            current_category=current_category,
            current_title=current_title,
            current_game=current_game,
        )
        if not force and existing.game_slug != "generic_jrpg_rpg" and self._profile_is_fresh(existing):
            return True, existing, "cached_profile"

        if not self.config.enabled:
            return False, existing, "research_disabled"
        if self.search_provider is None:
            return False, existing, "research_provider_missing"

        title = self._best_title(current_category=current_category, current_title=current_title, current_game=current_game)
        if not title:
            return False, existing, "game_unknown"

        query = self._build_query(title)
        try:
            self.search_count += 1
            results = self.search_provider.search(query)
            profile = self._profile_from_results(title, results, existing=existing)
            self.store.upsert_profile(profile)
            return True, profile, "researched"
        except Exception as exc:
            print(f"[HEBE][GAME_RESEARCH] failed title={title!r} error={exc!r}", flush=True)
            return False, existing, f"research_failed: {type(exc).__name__}: {exc}"

    def maybe_research_on_category_change(
        self,
        *,
        current_category: str | None,
        current_title: str | None = None,
        current_game: str | None = None,
    ) -> tuple[bool, GameProfile, str]:
        if not self.config.enabled:
            profile = self.store.lookup(current_category=current_category, current_title=current_title, current_game=current_game)
            return False, profile, "research_disabled"
        if self.store.has_specific_profile(current_category=current_category, current_title=current_title, current_game=current_game):
            profile = self.store.lookup(current_category=current_category, current_title=current_title, current_game=current_game)
            return True, profile, "local_or_cached_profile_exists"
        return self.research_current_game(
            current_category=current_category,
            current_title=current_title,
            current_game=current_game,
            force=False,
        )

    def _provider_from_config(self, config: GameKnowledgeResearchConfig) -> GameSearchProvider | None:
        provider = (config.provider or "").strip()
        if provider.casefold() in {"openai", "openai_responses", "openai_responses_web_search"}:
            candidate = OpenAIResponsesWebSearchProvider(config.api_key)
            return candidate if candidate.available else None
        if provider.startswith("http://") or provider.startswith("https://"):
            return HttpJsonSearchProvider(provider, api_key=config.api_key)
        return None

    def _profile_is_fresh(self, profile: GameProfile) -> bool:
        if not profile.sources_used:
            return False
        if "local_seed_spoiler_safe" in set(profile.sources_used):
            return False
        max_age = max(1, self.config.cache_days) * 86400
        return self._now() - float(profile.last_updated_ts or 0.0) <= max_age

    def _best_title(self, *, current_category: str | None, current_title: str | None, current_game: str | None) -> str:
        for value in (current_category, current_game):
            if str(value or "").strip():
                return str(value).strip()
        title = str(current_title or "").strip()
        if "|" in title:
            tail = title.split("|")[-1].strip()
            if tail:
                return tail
        return title

    def _build_query(self, title: str) -> str:
        return f"{title} no spoilers spoiler-free gameplay overview systems review"

    def _profile_from_results(self, title: str, results: list[dict[str, Any]], *, existing: GameProfile) -> GameProfile:
        safe_rows = [row for row in results if not self._looks_spoilery(row)]
        snippets = " ".join(
            str(row.get("snippet") or row.get("content") or row.get("description") or "")[:400]
            for row in safe_rows[:5]
        )
        sources = [
            str(row.get("url") or row.get("link") or row.get("source") or "").strip()
            for row in safe_rows[:5]
            if str(row.get("url") or row.get("link") or row.get("source") or "").strip()
        ]
        now = self._now()
        now_iso = datetime.fromtimestamp(now, timezone.utc).isoformat()
        genres = existing.genres if existing.game_slug != "generic_jrpg_rpg" else self._infer_genres(title, snippets)
        systems = self._infer_systems(snippets)
        safe_topics = self._safe_topics(existing, systems, snippets)
        unsafe = sorted(set(existing.unsafe_comment_topics + [
            "walkthrough steps",
            "exact boss weaknesses",
            "puzzle solutions",
            "future story beats",
            "character deaths",
            "endings",
            "future party members",
            "future location order",
        ]))
        slug = existing.game_slug if existing.game_slug != "generic_jrpg_rpg" else _slugify(title)
        return GameProfile(
            game_slug=slug,
            canonical_title=existing.canonical_title if existing.game_slug != "generic_jrpg_rpg" else title,
            aliases=sorted(set(existing.aliases + [title])),
            source_category_name=title,
            genres=genres,
            tone_vibe=self._infer_tone(snippets),
            general_non_spoiler_summary=self._safe_summary(title, snippets),
            gameplay_systems_non_spoiler=systems,
            channel_context=existing.channel_context or "spoiler-safe researched stream profile",
            leo_relationship=existing.leo_relationship or "Use for safer game-flavored stream commentary.",
            spoiler_policy="no_spoilers",
            safe_comment_topics=safe_topics,
            unsafe_comment_topics=unsafe,
            stream_hooks=sorted(set(existing.stream_hooks + ["spoiler-free flavor", "stream companion commentary"])),
            challenge_notes=existing.challenge_notes,
            challenge_hooks=existing.challenge_hooks or existing.challenge_notes,
            common_jokes=existing.common_jokes,
            sources_used=sources or ["configured_search_provider"],
            created_at=existing.created_at or now_iso,
            updated_at=now_iso,
            confidence=0.7 if safe_rows else 0.45,
            last_updated_ts=now,
        )

    def _looks_spoilery(self, row: dict[str, Any]) -> bool:
        text = " ".join(str(row.get(key) or "") for key in ("title", "snippet", "content", "description")).lower()
        return any(term in text for term in FORBIDDEN_RESEARCH_TERMS if term != "spoiler")

    def _infer_genres(self, title: str, text: str) -> list[str]:
        lowered = f"{title} {text}".lower()
        genres = []
        if "jrpg" in lowered or "final fantasy" in lowered or "persona" in lowered or "tales of" in lowered:
            genres.append("JRPG")
        if "strategy" in lowered or "tactical" in lowered:
            genres.append("strategy RPG")
        if "action" in lowered:
            genres.append("action RPG")
        if "rpg" in lowered and "RPG" not in genres:
            genres.append("RPG")
        return genres or ["RPG"]

    def _infer_systems(self, text: str) -> list[str]:
        lowered = text.lower()
        systems = []
        checks = {
            "turn-based combat": ("turn-based", "turn based"),
            "equipment and abilities": ("equipment", "abilities", "gear"),
            "party management": ("party", "companions"),
            "resource management": ("items", "resources", "inventory"),
            "exploration": ("exploration", "dungeons", "world"),
            "tactical positioning": ("tactical", "grid", "strategy"),
        }
        for label, needles in checks.items():
            if any(needle in lowered for needle in needles):
                systems.append(label)
        return systems or ["broad gameplay systems", "resource awareness", "exploration"]

    def _safe_topics(self, existing: GameProfile, systems: list[str], text: str) -> list[str]:
        topics = set(existing.safe_comment_topics)
        for system in systems:
            if "equipment" in system:
                topics.add("equipment-linked decisions")
            if "resource" in system:
                topics.add("resource management")
            if "exploration" in system:
                topics.add("exploration vibe")
            if "turn-based" in system:
                topics.add("turn planning")
        topics.update(["tone/vibe", "challenge run tension", "no-spoiler caution"])
        return sorted(topics)[:10]

    def _infer_tone(self, text: str) -> str:
        lowered = text.lower()
        bits = []
        if "whimsical" in lowered or "storybook" in lowered:
            bits.append("storybook and whimsical")
        if "dark" in lowered:
            bits.append("dark fantasy")
        if "comedy" in lowered or "humor" in lowered:
            bits.append("playful comedy")
        if "classic" in lowered or "retro" in lowered:
            bits.append("classic/retro")
        return ", ".join(bits) if bits else "spoiler-safe game flavor and broad gameplay mood"

    def _safe_summary(self, title: str, text: str) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return f"Spoiler-safe high-level profile for {title}."
        return clean[:500]

    def _now(self) -> float:
        if self._now_fn is not None:
            return float(self._now_fn())
        return time.time()


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return text or "unknown_game"


# Hebe Live v1 fact-level intelligence. The legacy profile service above stays
# available for existing callers while new live/replay code imports these
# provider-neutral contracts from the same module.
from app.stream.game_intelligence import (  # noqa: E402
    CommentKnowledgeContract,
    CommentKnowledgePolicy,
    GameAssistanceGuard,
    GameAssistanceMode,
    GameDossier,
    GameIntelligenceDiagnostics,
    GameIntelligenceStore,
    GameProgressState,
    GameResearchJob,
    GameResearchService,
    GameSearchPlan,
    KnowledgeGap,
    KnowledgeGapTracker,
    ResearchMode,
    ResearchTriggerDecision,
    ResearchTriggerEngine,
    RetrievedGameFact,
    SpoilerClassification,
    SpoilerFirewall,
    SpoilerGuardResult,
    default_assistance_mode,
)

__all__ = ["GameKnowledgeResearchConfig", "GameKnowledgeResearchService", "HttpJsonSearchProvider", "OpenAIResponsesWebSearchProvider"]
