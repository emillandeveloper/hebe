# backend/app/integrations/twitch/target_resolver.py

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


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
            self._normalize(k): v
            for k, v in (aliases or {}).items()
            if str(k).strip() and str(v).strip()
        }

    def resolve_user(self, raw_target: str | None, intent: str = "") -> str | None:
        if not raw_target:
            return self._resolve_from_context(intent)

        normalized = self._normalize(raw_target)
        if not normalized:
            return self._resolve_from_context(intent)

        # Alias manual
        alias_hit = self.aliases.get(normalized)
        if alias_hit:
            return alias_hit

        candidates = self._collect_candidates()

        best_username = None
        best_score = 0.0

        for username, display_name in candidates:
            username_norm = self._normalize(username)
            display_name_norm = self._normalize(display_name)

            score_username = self._score(normalized, username_norm) if username_norm else 0.0
            score_display = self._score(normalized, display_name_norm) if display_name_norm else 0.0

            score = max(score_username, score_display)

            # Pequeño bonus si coincide como substring
            if normalized and username_norm and normalized in username_norm:
                score = max(score, 0.90)
            if normalized and display_name_norm and normalized in display_name_norm:
                score = max(score, 0.90)

            if score > best_score:
                best_score = score
                best_username = username

        if best_username and best_score >= 0.82:
            return best_username

        return None

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

        # Último raid como candidato prioritario
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

    def _score(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()