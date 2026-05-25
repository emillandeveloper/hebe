from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True, slots=True)
class EntityDefinition:
    canonical_id: str
    display_name: str
    entity_type: str
    aliases: tuple[str, ...]
    context_keywords: tuple[str, ...]
    priority_private: int
    priority_stream: int


@dataclass(frozen=True, slots=True)
class EntityResolution:
    mention: str
    candidates: tuple[str, ...]
    selected: str
    reason: str
    broad_query: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ENTITY_CATALOG: dict[str, EntityDefinition] = {
    "jotun_dog": EntityDefinition(
        canonical_id="jotun_dog",
        display_name="Jotun",
        entity_type="dog",
        aliases=("jotun",),
        context_keywords=("perro", "dog", "mascota", "como esta", "como está"),
        priority_private=100,
        priority_stream=70,
    ),
    "jotun_bot": EntityDefinition(
        canonical_id="jotun_bot",
        display_name="JotunBot",
        entity_type="channel_bot",
        aliases=("jotunbot", "jotun bot"),
        context_keywords=("bot", "comando", "comandos", "follow", "chat", "twitch", "canal"),
        priority_private=60,
        priority_stream=100,
    ),
    "hebe_ai": EntityDefinition(
        canonical_id="hebe_ai",
        display_name="Hebe",
        entity_type="ai_companion",
        aliases=("hebe",),
        context_keywords=("quien eres", "quién eres", "compañera", "ai", "ia"),
        priority_private=100,
        priority_stream=100,
    ),
    "leo": EntityDefinition(
        canonical_id="leo",
        display_name="Leo",
        entity_type="person",
        aliases=("leo", "leonifelheim", "leo nifelheim"),
        context_keywords=("broadcaster", "streamer", "compañero"),
        priority_private=100,
        priority_stream=100,
    ),
    "stream_channel": EntityDefinition(
        canonical_id="stream_channel",
        display_name="canal de Leo",
        entity_type="stream_channel",
        aliases=("canal", "stream", "directo"),
        context_keywords=("twitch", "chat", "follow", "vod", "comandos"),
        priority_private=55,
        priority_stream=100,
    ),
}


class EntityResolver:
    def resolve(
        self,
        text: str | None,
        *,
        source_context: str = "private",
    ) -> list[EntityResolution]:
        normalized = normalize_text(text)
        if not normalized:
            return []

        resolutions: list[EntityResolution] = []
        if self._mentions_jotun(normalized):
            resolutions.append(self._resolve_jotun(normalized, source_context=source_context))

        for entity in ("hebe_ai", "leo", "stream_channel"):
            definition = ENTITY_CATALOG[entity]
            if any(self._alias_in_text(alias, normalized) for alias in definition.aliases):
                resolutions.append(
                    EntityResolution(
                        mention=definition.display_name,
                        candidates=(definition.canonical_id,),
                        selected=definition.canonical_id,
                        reason="explicit_alias",
                        broad_query=False,
                    )
                )

        return self._dedupe(resolutions)

    def _resolve_jotun(self, text: str, *, source_context: str) -> EntityResolution:
        candidates = ("jotun_dog", "jotun_bot")
        broad_query = self._is_broad_identity_query(text)

        if "jotunbot" in text or "jotun bot" in text:
            return EntityResolution(
                mention="JotunBot",
                candidates=("jotun_bot",),
                selected="jotun_bot",
                reason="explicit_alias",
                broad_query=broad_query,
            )

        bot_keywords = ENTITY_CATALOG["jotun_bot"].context_keywords
        if any(keyword in text for keyword in bot_keywords):
            return EntityResolution(
                mention="Jotun",
                candidates=candidates,
                selected="jotun_bot",
                reason="context_keyword",
                broad_query=broad_query,
            )

        dog_keywords = ENTITY_CATALOG["jotun_dog"].context_keywords
        if any(keyword in text for keyword in dog_keywords):
            return EntityResolution(
                mention="Jotun",
                candidates=candidates,
                selected="jotun_dog",
                reason="context_keyword",
                broad_query=broad_query,
            )

        if source_context == "stream":
            return EntityResolution(
                mention="Jotun",
                candidates=candidates,
                selected="jotun_bot",
                reason="stream_context_default",
                broad_query=broad_query,
            )

        return EntityResolution(
            mention="Jotun",
            candidates=candidates,
            selected="jotun_dog",
            reason="private_chat_default",
            broad_query=broad_query,
        )

    def _mentions_jotun(self, text: str) -> bool:
        return self._alias_in_text("jotun", text) or self._alias_in_text("jotunbot", text)

    def _alias_in_text(self, alias: str, text: str) -> bool:
        normalized_alias = normalize_text(alias)
        return bool(re.search(rf"(^|\s){re.escape(normalized_alias)}($|\s)", text))

    def _is_broad_identity_query(self, text: str) -> bool:
        return any(
            marker in text
            for marker in (
                "quien es",
                "quién es",
                "que es",
                "qué es",
                "who is",
                "what is",
            )
        )

    def _dedupe(self, resolutions: list[EntityResolution]) -> list[EntityResolution]:
        seen: set[str] = set()
        out: list[EntityResolution] = []
        for resolution in resolutions:
            key = resolution.selected
            if key in seen:
                continue
            seen.add(key)
            out.append(resolution)
        return out


def normalize_text(text: str | None) -> str:
    raw = (text or "").strip().lower()
    without_accents = "".join(
        ch for ch in unicodedata.normalize("NFKD", raw)
        if not unicodedata.combining(ch)
    )
    cleaned = re.sub(r"[^a-z0-9ñ\s']", " ", without_accents)
    return " ".join(cleaned.split())


def entity_prompt_lines(resolutions: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for resolution in resolutions:
        selected = resolution.get("selected")
        if selected == "jotun_dog":
            if resolution.get("broad_query"):
                lines.append(
                    "Jotun resolution: in private chat, 'Jotun' primarily means Leo's dog. "
                    "If answering broadly, mention the dog first; then mention that Jotun/JotunBot also has channel/bot identity if relevant."
                )
            else:
                lines.append("Jotun resolution: prefer Leo's dog. Do not answer as if Jotun is only JotunBot.")
        elif selected == "jotun_bot":
            lines.append("Jotun resolution: this asks about JotunBot / channel bot or commands. Keep bot context primary.")
        elif selected == "hebe_ai":
            lines.append("Hebe resolution: answer as Hebe, Leo's companion, not a generic assistant.")
    return lines
