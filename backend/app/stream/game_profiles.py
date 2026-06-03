from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROFILE_PATH = Path(__file__).with_name("game_profiles.seed.json")
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "game_knowledge_profiles.cache.json"


@dataclass(frozen=True)
class GameProfile:
    game_slug: str
    canonical_title: str
    aliases: list[str] = field(default_factory=list)
    source_category_name: str = ""
    genres: list[str] = field(default_factory=list)
    tone_vibe: str = ""
    general_non_spoiler_summary: str = ""
    gameplay_systems_non_spoiler: list[str] = field(default_factory=list)
    channel_context: str = ""
    leo_relationship: str = ""
    spoiler_policy: str = "no_spoilers"
    safe_comment_topics: list[str] = field(default_factory=list)
    unsafe_comment_topics: list[str] = field(default_factory=list)
    stream_hooks: list[str] = field(default_factory=list)
    challenge_notes: list[str] = field(default_factory=list)
    challenge_hooks: list[str] = field(default_factory=list)
    common_jokes: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    confidence: float = 0.0
    last_updated_ts: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GameProfile":
        updated_ts = float(raw.get("last_updated_ts") or time.time())
        updated_at = str(raw.get("updated_at") or "").strip()
        if not updated_at:
            updated_at = datetime.fromtimestamp(updated_ts, timezone.utc).isoformat()
        created_at = str(raw.get("created_at") or updated_at).strip()
        return cls(
            game_slug=str(raw.get("game_slug") or "").strip(),
            canonical_title=str(raw.get("canonical_title") or "").strip(),
            aliases=_list(raw.get("aliases")),
            source_category_name=str(raw.get("source_category_name") or raw.get("canonical_title") or "").strip(),
            genres=_list(raw.get("genres")),
            tone_vibe=str(raw.get("tone_vibe") or "").strip(),
            general_non_spoiler_summary=str(raw.get("general_non_spoiler_summary") or raw.get("channel_context") or "").strip(),
            gameplay_systems_non_spoiler=_list(raw.get("gameplay_systems_non_spoiler")),
            channel_context=str(raw.get("channel_context") or "").strip(),
            leo_relationship=str(raw.get("leo_relationship") or "").strip(),
            spoiler_policy=str(raw.get("spoiler_policy") or "no_spoilers").strip(),
            safe_comment_topics=_list(raw.get("safe_comment_topics")),
            unsafe_comment_topics=_list(raw.get("unsafe_comment_topics")),
            stream_hooks=_list(raw.get("stream_hooks")),
            challenge_notes=_list(raw.get("challenge_notes")),
            challenge_hooks=_list(raw.get("challenge_hooks") or raw.get("challenge_notes")),
            common_jokes=_list(raw.get("common_jokes")),
            sources_used=_list(raw.get("sources_used")),
            created_at=created_at,
            updated_at=updated_at,
            confidence=float(raw.get("confidence") or 0.75),
            last_updated_ts=updated_ts,
        )

    def compact_prompt_context(self) -> dict[str, Any]:
        return {
            "title": self.canonical_title,
            "source_category_name": self.source_category_name,
            "genres": self.genres[:5],
            "tone_vibe": self.tone_vibe,
            "general_non_spoiler_summary": self.general_non_spoiler_summary[:300],
            "gameplay_systems_non_spoiler": self.gameplay_systems_non_spoiler[:8],
            "channel_context": self.channel_context,
            "leo_relationship": self.leo_relationship,
            "safe_comment_topics": self.safe_comment_topics[:6],
            "spoiler_policy": self.spoiler_policy,
            "unsafe_comment_topics": self.unsafe_comment_topics[:6],
            "stream_hooks": self.stream_hooks[:5],
            "challenge_notes": self.challenge_notes[:5],
            "challenge_hooks": self.challenge_hooks[:5],
            "common_jokes": self.common_jokes[:5],
            "confidence": self.confidence,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_slug": self.game_slug,
            "canonical_title": self.canonical_title,
            "aliases": self.aliases,
            "source_category_name": self.source_category_name,
            "genres": self.genres,
            "tone_vibe": self.tone_vibe,
            "general_non_spoiler_summary": self.general_non_spoiler_summary,
            "gameplay_systems_non_spoiler": self.gameplay_systems_non_spoiler,
            "channel_context": self.channel_context,
            "leo_relationship": self.leo_relationship,
            "spoiler_policy": self.spoiler_policy,
            "safe_comment_topics": self.safe_comment_topics,
            "unsafe_comment_topics": self.unsafe_comment_topics,
            "stream_hooks": self.stream_hooks,
            "challenge_notes": self.challenge_notes,
            "challenge_hooks": self.challenge_hooks,
            "common_jokes": self.common_jokes,
            "sources_used": self.sources_used,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
            "last_updated_ts": self.last_updated_ts,
        }


class GameProfileStore:
    def __init__(self, path: str | Path | None = None, cache_path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_PROFILE_PATH
        self.cache_path = Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH
        self.profiles: list[GameProfile] = []
        self.reload()

    def reload(self) -> int:
        profiles: list[GameProfile] = []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            rows = raw.get("profiles", raw if isinstance(raw, list) else [])
            profiles = [
                profile
                for profile in (GameProfile.from_dict(row) for row in rows)
                if profile.game_slug and profile.canonical_title
            ]
        except Exception as exc:
            print(f"[HEBE][GAME_PROFILE] load failed path={self.path} error={exc!r}", flush=True)
            profiles = [self._generic_profile()]
        profiles = self._merge_profiles(profiles, self._load_cache_profiles())
        self.profiles = profiles
        if not any(profile.game_slug == "generic_jrpg_rpg" for profile in self.profiles):
            self.profiles.append(self._generic_profile())
        return len(self.profiles)

    def lookup(
        self,
        *,
        current_category: str | None = None,
        current_title: str | None = None,
        current_game: str | None = None,
    ) -> GameProfile:
        haystacks = [
            _normalize(current_category),
            _normalize(current_game),
            _normalize(current_title),
        ]
        category_norm, game_norm, title_norm = haystacks

        for profile in self.profiles:
            names = [_normalize(profile.canonical_title)] + [_normalize(alias) for alias in profile.aliases]
            if category_norm and category_norm in names:
                return profile
            if game_norm and game_norm in names:
                return profile

        for profile in self.profiles:
            names = [_normalize(profile.canonical_title)] + [_normalize(alias) for alias in profile.aliases]
            for name in names:
                if not name:
                    continue
                if title_norm and _contains_name(title_norm, name):
                    return profile
                if category_norm and _contains_name(category_norm, name):
                    return profile

        return self._generic()

    def has_specific_profile(
        self,
        *,
        current_category: str | None = None,
        current_title: str | None = None,
        current_game: str | None = None,
    ) -> bool:
        return self.lookup(
            current_category=current_category,
            current_title=current_title,
            current_game=current_game,
        ).game_slug != "generic_jrpg_rpg"

    def upsert_profile(self, profile: GameProfile) -> None:
        profiles = [item for item in self.profiles if item.game_slug != profile.game_slug]
        profiles.append(profile)
        self.profiles = self._merge_profiles(profiles, [])
        self._write_cache_profiles([item for item in self.profiles if self._is_cache_profile(item)])

    def forget_profile(
        self,
        *,
        current_category: str | None = None,
        current_title: str | None = None,
        current_game: str | None = None,
    ) -> GameProfile:
        profile = self.lookup(
            current_category=current_category,
            current_title=current_title,
            current_game=current_game,
        )
        if profile.game_slug == "generic_jrpg_rpg":
            return profile
        self.profiles = [item for item in self.profiles if item.game_slug != profile.game_slug]
        self._write_cache_profiles([item for item in self.profiles if self._is_cache_profile(item)])
        return self._generic()

    def _generic(self) -> GameProfile:
        for profile in self.profiles:
            if profile.game_slug == "generic_jrpg_rpg":
                return profile
        return self._generic_profile()

    def _generic_profile(self) -> GameProfile:
        return GameProfile(
            game_slug="generic_jrpg_rpg",
            canonical_title="Generic JRPG/RPG",
            aliases=["jrpg", "rpg", "role playing game"],
            source_category_name="Generic JRPG/RPG",
            genres=["JRPG", "RPG"],
            tone_vibe="broad fantasy or role-playing adventure",
            general_non_spoiler_summary="Fallback spoiler-safe RPG profile when the current game is unknown.",
            gameplay_systems_non_spoiler=["party preparation", "resource management", "exploration", "menus"],
            channel_context="generic safe RPG stream context",
            leo_relationship="fallback profile when the current game is unknown",
            spoiler_policy="no_spoilers",
            safe_comment_topics=["save points", "equipment checks", "resources", "exploration", "menu preparation"],
            unsafe_comment_topics=["story spoilers", "boss identities", "puzzle solutions", "exact walkthrough steps"],
            stream_hooks=["first playthrough", "challenge run", "exploration", "preparation matters"],
            challenge_hooks=["challenge runs reward planning without exact guide advice"],
            common_jokes=["JRPG doors lead to geopolitical problems"],
            sources_used=[],
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            confidence=0.5,
            last_updated_ts=time.time(),
        )

    def _load_cache_profiles(self) -> list[GameProfile]:
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            rows = raw.get("profiles", raw if isinstance(raw, list) else [])
            return [
                profile
                for profile in (GameProfile.from_dict(row) for row in rows)
                if profile.game_slug and profile.canonical_title
            ]
        except FileNotFoundError:
            return []
        except Exception as exc:
            print(f"[HEBE][GAME_PROFILE] cache load failed path={self.cache_path} error={exc!r}", flush=True)
            return []

    def _write_cache_profiles(self, profiles: list[GameProfile]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"profiles": [profile.to_dict() for profile in profiles if profile.game_slug != "generic_jrpg_rpg"]}
            self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[HEBE][GAME_PROFILE] cache write failed path={self.cache_path} error={exc!r}", flush=True)

    def _merge_profiles(self, base: list[GameProfile], extra: list[GameProfile]) -> list[GameProfile]:
        merged: dict[str, GameProfile] = {}
        for profile in base + extra:
            merged[profile.game_slug] = profile
        return list(merged.values())

    def _is_cache_profile(self, profile: GameProfile) -> bool:
        return bool(profile.sources_used) and "local_seed_spoiler_safe" not in set(profile.sources_used)


def _list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return [str(value).strip()]


def _normalize(value: str | None) -> str:
    text = str(value or "").lower()
    text = text.replace("é", "e").replace("á", "a").replace("í", "i").replace("ó", "o").replace("ú", "u")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _contains_name(haystack: str, name: str) -> bool:
    if not haystack or not name:
        return False
    return bool(re.search(rf"(^|\s){re.escape(name)}($|\s)", haystack))
