from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
import os
import re
import unicodedata
from typing import Any


DEFAULT_VIEWER_ALIASES = {
    "charlie": "er_tito_xarly",
    "xarly": "er_tito_xarly",
    "nuria": "nuriiia___",
    "superdamu": "superdamu",
    "super damu": "superdamu",
    "super dammu": "superdamu",
    "super damo": "superdamu",
    "superdammu": "superdamu",
    "superdamo": "superdamu",
}


@dataclass(frozen=True)
class TargetResolution:
    username: str | None = None
    confidence: float = 0.0
    candidates: list[str] = field(default_factory=list)
    reason: str = "target_unclear"
    source: str = ""


class TwitchTargetResolver:
    def __init__(
        self,
        chat_cache: Any | None,
        event_memory: Any | None,
        aliases: dict[str, str] | None = None,
    ):
        self.chat_cache = chat_cache
        self.event_memory = event_memory
        configured_aliases = dict(DEFAULT_VIEWER_ALIASES)
        configured_aliases.update(self._load_aliases_from_env())
        configured_aliases.update(aliases or {})
        self.aliases = {
            self._normalize(k): str(v).strip().lstrip("@")
            for k, v in configured_aliases.items()
            if str(k).strip() and str(v).strip()
        }

    def remember_alias(self, alias: str, username: str) -> bool:
        alias_key = self._normalize(alias)
        target = str(username or "").strip().lstrip("@")
        if not alias_key or not target:
            return False
        self.aliases[alias_key] = target
        return True

    def resolve_user(self, raw_target: str | None, intent: str = "") -> str | None:
        return self.resolve_user_details(raw_target, intent=intent).username

    def resolve_user_details(self, raw_target: str | None, intent: str = "") -> TargetResolution:
        if not raw_target:
            contextual = self._resolve_from_context(intent)
            if contextual:
                return TargetResolution(contextual, 0.82, [contextual], "contextual_target")
            return TargetResolution(reason="missing_target")

        normalized = self._normalize(raw_target)
        if not normalized:
            contextual = self._resolve_from_context(intent)
            if contextual:
                return TargetResolution(contextual, 0.82, [contextual], "contextual_target")
            return TargetResolution(reason="missing_target")

        alias_hit = self.aliases.get(normalized)
        if alias_hit:
            return TargetResolution(alias_hit, 0.99, [alias_hit], "alias", "alias")

        scored: list[tuple[float, str]] = []
        for username, display_name, source in self._collect_candidates():
            username_norm = self._normalize(username)
            display_name_norm = self._normalize(display_name)
            compact_target = self._compact(normalized)
            compact_username = self._compact(username_norm)
            compact_display = self._compact(display_name_norm)
            if normalized in {username_norm, display_name_norm} or compact_target in {compact_username, compact_display}:
                return TargetResolution(username, 1.0, [username], "exact_target", source or "exact")
            score = max(
                self._score(normalized, username_norm),
                self._score(normalized, display_name_norm),
                self._score(compact_target, compact_username),
                self._score(compact_target, compact_display),
                self._score(self._squash_repeats(compact_target), self._squash_repeats(compact_username)),
                self._score(self._squash_repeats(compact_target), self._squash_repeats(compact_display)),
            )

            if normalized and username_norm and normalized in username_norm:
                score = max(score, 0.90)
            if normalized and display_name_norm and normalized in display_name_norm:
                score = max(score, 0.90)
            if self._loose_alias_match(normalized, username_norm) or self._loose_alias_match(normalized, display_name_norm):
                score = max(score, 0.88)

            if score >= 0.58:
                scored.append((score, username, source or "fuzzy"))

        scored.sort(reverse=True, key=lambda item: item[0])
        if scored:
            best_score = scored[0][0]
            close = [name for score, name, _source in scored if best_score - score <= 0.04]
            if len(close) > 1:
                return TargetResolution(close[0], best_score, close[:4], "ambiguous_target", scored[0][2])
            if best_score >= 0.86:
                return TargetResolution(scored[0][1], best_score, [scored[0][1]], "fuzzy_known_target", scored[0][2])
            if best_score >= 0.70:
                return TargetResolution(scored[0][1], best_score, [scored[0][1]], "medium_confidence", scored[0][2])

        return TargetResolution(reason="target_unclear")

    def _resolve_from_context(self, intent: str) -> str | None:
        lowered_intent = str(intent or "").lower()

        last_raid = getattr(self.event_memory, "last_raid", None)
        if "raid" in lowered_intent and last_raid is not None:
            username = getattr(last_raid, "username", None)
            if username:
                return username

        last_user_fn = getattr(self.chat_cache, "last_user", None)
        if callable(last_user_fn):
            return last_user_fn()

        return None

    def _collect_candidates(self) -> list[tuple[str, str, str]]:
        candidates: list[tuple[str, str, str]] = []
        seen: set[str] = set()

        last_raid = getattr(self.event_memory, "last_raid", None)
        if last_raid is not None:
            raid_username = getattr(last_raid, "username", None)
            if raid_username:
                key = self._normalize(raid_username)
                if key and key not in seen:
                    seen.add(key)
                    candidates.append((raid_username, raid_username, "recent_raid"))

        for attr in ("last_follow_username", "last_sub_username"):
            username = getattr(self.event_memory, attr, None)
            if username:
                key = self._normalize(username)
                if key and key not in seen:
                    seen.add(key)
                    candidates.append((username, username, "recent_follower" if "follow" in attr else "recent_event"))

        recent_users_fn = getattr(self.chat_cache, "recent_users", None)
        if callable(recent_users_fn):
            for item in recent_users_fn() or []:
                if not isinstance(item, tuple) or len(item) != 2:
                    continue
                username, display_name = item
                username = str(username or "").strip()
                display_name = str(display_name or "").strip()
                if not username:
                    continue
                key = self._normalize(username)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((username, display_name, "active_chatter"))

        return candidates

    def _normalize(self, value: str) -> str:
        lowered = str(value or "").lower().replace("_", " ").replace("-", " ").strip()
        lowered = "".join(ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch))
        return re.sub(r"\s+", " ", lowered).strip()

    def _compact(self, value: str) -> str:
        return "".join(ch for ch in self._normalize(value) if ch.isalnum() or ch == "_")

    def _loose_alias_match(self, raw: str, candidate: str) -> bool:
        raw_key = self._phonetic_key(raw)
        candidate_key = self._phonetic_key(candidate)
        if not raw_key or not candidate_key:
            return False
        return raw_key in candidate_key or candidate_key in raw_key

    def _phonetic_key(self, value: str) -> str:
        key = self._squash_repeats(self._compact(value)).replace("ch", "x")
        if key.endswith("ie"):
            key = key[:-2] + "y"
        if key.endswith("i"):
            key = key[:-1] + "y"
        return key

    def _squash_repeats(self, value: str) -> str:
        return re.sub(r"(.)\1{1,}", r"\1", str(value or ""))

    def _score(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _load_aliases_from_env(self) -> dict[str, str]:
        raw = os.getenv("HEBE_TWITCH_VIEWER_ALIASES_JSON", "").strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v).strip()}
