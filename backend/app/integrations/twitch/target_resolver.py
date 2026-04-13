from difflib import SequenceMatcher


class TwitchTargetResolver:
    def __init__(self, chat_cache, event_memory, aliases: dict[str, str] | None = None):
        self.chat_cache = chat_cache
        self.event_memory = event_memory
        self.aliases = aliases or {}

    def resolve_user(self, raw_target: str | None, intent: str = "") -> str | None:
        if not raw_target:
            return self._resolve_from_context(intent)

        normalized = self._normalize(raw_target)

        # alias manual
        if normalized in self.aliases:
            return self.aliases[normalized]

        candidates = []

        if self.event_memory.last_raid:
            candidates.append(self.event_memory.last_raid.username)

        for username, display_name in self.chat_cache.recent_users():
            candidates.append(username)
            if display_name and display_name.lower() != username.lower():
                candidates.append(display_name)

        best = None
        best_score = 0.0

        for candidate in candidates:
            score = self._score(normalized, self._normalize(candidate))
            if score > best_score:
                best = candidate
                best_score = score

        if best_score >= 0.82:
            return self._canonical_username(best)

        return None

    def _resolve_from_context(self, intent: str) -> str | None:
        if "raid" in intent and self.event_memory.last_raid:
            return self.event_memory.last_raid.username
        return self.chat_cache.last_user()

    def _normalize(self, value: str) -> str:
        return value.lower().replace("_", " ").replace("-", " ").strip()

    def _score(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _canonical_username(self, value: str) -> str:
        norm = self._normalize(value)
        for username, display_name in self.chat_cache.recent_users():
            if self._normalize(username) == norm or self._normalize(display_name) == norm:
                return username
        return value