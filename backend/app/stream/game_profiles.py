from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PROFILE_PATH = Path(__file__).with_name("game_profiles.seed.json")


@dataclass(frozen=True)
class GameProfile:
    game_slug: str
    canonical_title: str
    aliases: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    channel_context: str = ""
    leo_relationship: str = ""
    spoiler_policy: str = "no_spoilers"
    safe_comment_topics: list[str] = field(default_factory=list)
    unsafe_comment_topics: list[str] = field(default_factory=list)
    stream_hooks: list[str] = field(default_factory=list)
    challenge_notes: list[str] = field(default_factory=list)
    common_jokes: list[str] = field(default_factory=list)
    last_updated_ts: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GameProfile":
        return cls(
            game_slug=str(raw.get("game_slug") or "").strip(),
            canonical_title=str(raw.get("canonical_title") or "").strip(),
            aliases=_list(raw.get("aliases")),
            genres=_list(raw.get("genres")),
            channel_context=str(raw.get("channel_context") or "").strip(),
            leo_relationship=str(raw.get("leo_relationship") or "").strip(),
            spoiler_policy=str(raw.get("spoiler_policy") or "no_spoilers").strip(),
            safe_comment_topics=_list(raw.get("safe_comment_topics")),
            unsafe_comment_topics=_list(raw.get("unsafe_comment_topics")),
            stream_hooks=_list(raw.get("stream_hooks")),
            challenge_notes=_list(raw.get("challenge_notes")),
            common_jokes=_list(raw.get("common_jokes")),
            last_updated_ts=float(raw.get("last_updated_ts") or time.time()),
        )

    def compact_prompt_context(self) -> dict[str, Any]:
        return {
            "title": self.canonical_title,
            "genres": self.genres[:5],
            "channel_context": self.channel_context,
            "leo_relationship": self.leo_relationship,
            "safe_comment_topics": self.safe_comment_topics[:6],
            "spoiler_policy": self.spoiler_policy,
            "unsafe_comment_topics": self.unsafe_comment_topics[:6],
            "stream_hooks": self.stream_hooks[:5],
            "challenge_notes": self.challenge_notes[:5],
            "common_jokes": self.common_jokes[:5],
        }


class GameProfileStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_PROFILE_PATH
        self.profiles: list[GameProfile] = []
        self.reload()

    def reload(self) -> int:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            rows = raw.get("profiles", raw if isinstance(raw, list) else [])
            self.profiles = [
                profile
                for profile in (GameProfile.from_dict(row) for row in rows)
                if profile.game_slug and profile.canonical_title
            ]
        except Exception as exc:
            print(f"[HEBE][GAME_PROFILE] load failed path={self.path} error={exc!r}", flush=True)
            self.profiles = [self._generic_profile()]
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
            genres=["JRPG", "RPG"],
            channel_context="generic safe RPG stream context",
            leo_relationship="fallback profile when the current game is unknown",
            spoiler_policy="no_spoilers",
            safe_comment_topics=["save points", "equipment checks", "resources", "exploration", "menu preparation"],
            unsafe_comment_topics=["story spoilers", "boss identities", "puzzle solutions", "exact walkthrough steps"],
            stream_hooks=["first playthrough", "challenge run", "exploration", "preparation matters"],
            common_jokes=["JRPG doors lead to geopolitical problems"],
            last_updated_ts=time.time(),
        )


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
