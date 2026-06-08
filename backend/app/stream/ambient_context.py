from __future__ import annotations

from dataclasses import dataclass, field
import re
import time


@dataclass(frozen=True)
class AmbientContextExtraction:
    useful: bool
    facts: list[dict] = field(default_factory=list)
    mood: str | None = None
    reason: str = ""


class AmbientContextExtractor:
    """Extract lightweight stream/run facts from accepted ambient STT."""

    def extract(self, text: str, *, event_type: str | None = None, now: float | None = None) -> AmbientContextExtraction:
        raw = str(text or "").strip()
        normalized = self._normalize(raw)
        if not normalized:
            return AmbientContextExtraction(useful=False, reason="empty")

        now = time.time() if now is None else float(now)
        facts: list[dict] = []
        mood: str | None = None

        level_fact = self._extract_level_gap(raw, normalized, now)
        if level_fact:
            facts.append(level_fact)
            mood = "underleveled / challenge tension"

        relation_fact = self._extract_creator_relation(raw, normalized, now)
        if relation_fact:
            facts.append(relation_fact)

        objective_fact = self._extract_objective_or_location(raw, normalized, event_type, now)
        if objective_fact:
            facts.append(objective_fact)

        if facts:
            return AmbientContextExtraction(useful=True, facts=facts, mood=mood, reason="facts_extracted")

        if event_type in {"gameplay_failure", "victory", "boss_attempt", "grinding", "confusion/lost"}:
            return AmbientContextExtraction(
                useful=True,
                facts=[self._fact("phase", raw[:140], 0.45, now, ttl_sec=10 * 60)],
                mood=mood,
                reason="event_type_context",
            )

        if len(normalized.split()) <= 4 or normalized in {"vamos por aqui", "vamos por ahi", "por aqui", "por ahi"}:
            return AmbientContextExtraction(useful=False, reason="low_value")

        if any(word in normalized for word in ("juego", "creadores", "saga", "nivel", "zona", "boss", "jefe")):
            return AmbientContextExtraction(
                useful=True,
                facts=[self._fact("ambient_note", raw[:180], 0.4, now, ttl_sec=20 * 60)],
                mood=mood,
                reason="topic_hint",
            )

        return AmbientContextExtraction(useful=False, reason="low_value")

    def _extract_level_gap(self, raw: str, normalized: str, now: float) -> dict | None:
        words_to_numbers = {
            "uno": 1,
            "una": 1,
            "dos": 2,
            "tres": 3,
            "cuatro": 4,
            "cinco": 5,
            "seis": 6,
            "siete": 7,
            "ocho": 8,
            "nueve": 9,
            "diez": 10,
        }
        number_pattern = "|".join(words_to_numbers)
        match = re.search(
            rf"(?:son|estan|están|van)?\s*(\d+|{number_pattern})\s+(?:niveles?|levels?)\s+(?:mas|más)\s+que\s+yo",
            normalized,
        )
        if not match:
            return None
        value_raw = match.group(1)
        value = int(value_raw) if value_raw.isdigit() else words_to_numbers.get(value_raw, 0)
        if value <= 0:
            return None
        return self._fact(
            "level_gap",
            f"Enemies/current area are about {value} levels above Leo.",
            0.82,
            now,
            ttl_sec=25 * 60,
            data={"level_gap": value, "source_text": raw},
        )

    def _extract_creator_relation(self, raw: str, normalized: str, now: float) -> dict | None:
        if "creadores de" not in normalized and "creador de" not in normalized:
            return None
        relation = raw[:180]
        return self._fact(
            "game_relation",
            f"Leo mentioned a game/developer relationship: {relation}",
            0.68,
            now,
            ttl_sec=30 * 60,
            data={"source_text": raw},
        )

    def _extract_objective_or_location(self, raw: str, normalized: str, event_type: str | None, now: float) -> dict | None:
        if event_type == "objective_update":
            return self._fact("objective", raw[:140], 0.7, now, ttl_sec=35 * 60)
        if event_type == "location_update":
            return self._fact("location", raw[:100], 0.7, now, ttl_sec=35 * 60)
        return None

    def _fact(
        self,
        kind: str,
        text: str,
        confidence: float,
        now: float,
        *,
        ttl_sec: float,
        data: dict | None = None,
    ) -> dict:
        return {
            "kind": kind,
            "text": text,
            "source": "stt_voice",
            "timestamp": now,
            "confidence": confidence,
            "ttl_sec": ttl_sec,
            "expires_at": now + ttl_sec,
            "data": data or {},
        }

    def _normalize(self, text: str) -> str:
        normalized = str(text or "").lower()
        normalized = normalized.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        normalized = normalized.replace("ü", "u").replace("ñ", "n")
        normalized = re.sub(r"[^a-z0-9_ ]+", " ", normalized)
        return " ".join(normalized.split())
