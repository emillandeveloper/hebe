from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


@dataclass(frozen=True)
class TargetResolution:
    username: str | None = None
    confidence: float = 0.0
    candidates: list[str] = field(default_factory=list)
    reason: str = "target_unclear"


class TwitchTargetResolver:
    def __init__(
        self,
        chat_cache: Any | None,
        event_memory: Any | None,
        aliases: dict[str, str] | None = None,
    ):
        self.chat_cache = chat_cache
        self.event_memory = event_memory
        self.aliases = {
            self._normalize(k): str(v).strip().lstrip("@")
            for k, v in (aliases or {}).items()
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
            return TargetResolution(alias_hit, 0.98, [alias_hit], "alias")

        scored: list[tuple[float, str]] = []
        for username, display_name in self._collect_candidates():
            username_norm = self._normalize(username)
            display_name_norm = self._normalize(display_name)
            score = max(
                self._score(normalized, username_norm),
                self._score(normalized, display_name_norm),
                self._score(self._compact(normalized), self._compact(username_norm)),
                self._score(self._compact(normalized), self._compact(display_name_norm)),
            )

            if normalized and username_norm and normalized in username_norm:
                score = max(score, 0.90)
            if normalized and display_name_norm and normalized in display_name_norm:
                score = max(score, 0.90)
            if self._loose_alias_match(normalized, username_norm) or self._loose_alias_match(normalized, display_name_norm):
                score = max(score, 0.88)

            if score >= 0.58:
                scored.append((score, username))

        scored.sort(reverse=True, key=lambda item: item[0])
        if scored:
            best_score = scored[0][0]
            close = [name for score, name in scored if best_score - score <= 0.04]
            if len(close) > 1:
                return TargetResolution(close[0], best_score, close[:4], "ambiguous_target")
            if best_score >= 0.82:
                return TargetResolution(scored[0][1], best_score, [scored[0][1]], "fuzzy_known_target")
            if best_score >= 0.68:
                return TargetResolution(scored[0][1], best_score, [scored[0][1]], "medium_confidence")

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

    def _collect_candidates(self) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()

        last_raid = getattr(self.event_memory, "last_raid", None)
        if last_raid is not None:
            raid_username = getattr(last_raid, "username", None)
            if raid_username:
                key = self._normalize(raid_username)
                if key and key not in seen:
                    seen.add(key)
                    candidates.append((raid_username, raid_username))

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
                candidates.append((username, display_name))

        return candidates

    def _normalize(self, value: str) -> str:
        return (
            str(value or "")
            .lower()
            .replace("_", " ")
            .replace("-", " ")
            .strip()
        )

    def _compact(self, value: str) -> str:
        return "".join(ch for ch in self._normalize(value) if ch.isalnum() or ch == "_")

    def _loose_alias_match(self, raw: str, candidate: str) -> bool:
        raw_key = self._phonetic_key(raw)
        candidate_key = self._phonetic_key(candidate)
        if not raw_key or not candidate_key:
            return False
        return raw_key in candidate_key or candidate_key in raw_key

    def _phonetic_key(self, value: str) -> str:
        key = self._compact(value).replace("ch", "x")
        if key.endswith("ie"):
            key = key[:-2] + "y"
        if key.endswith("i"):
            key = key[:-1] + "y"
        return key

    def _score(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()
