from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
import unicodedata


@dataclass(frozen=True)
class AmbientContextExtraction:
    useful: bool
    facts: list[dict] = field(default_factory=list)
    mood: str | None = None
    reason: str = ""


class AmbientContextExtractor:
    """Extract current stream/run facts from accepted ambient STT."""

    def extract(self, text: str, *, event_type: str | None = None, now: float | None = None) -> AmbientContextExtraction:
        raw = str(text or "").strip()
        normalized = self._normalize(raw)
        if not normalized:
            return AmbientContextExtraction(useful=False, reason="empty")

        now = time.time() if now is None else float(now)
        facts: list[dict] = []
        mood: str | None = None

        gameplay_facts = self._extract_gameplay_facts(raw, normalized, now)
        if gameplay_facts:
            facts.extend(gameplay_facts)
            mood = next((fact.get("data", {}).get("mood") for fact in gameplay_facts if fact.get("data", {}).get("mood")), mood)

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

        if self._is_generic_filler(normalized):
            return AmbientContextExtraction(useful=False, reason="generic_filler")

        if any(word in normalized for word in ("juego", "creadores", "saga", "nivel", "zona", "boss", "jefe")):
            return AmbientContextExtraction(
                useful=True,
                facts=[self._fact("ambient_note", raw[:180], 0.4, now, ttl_sec=20 * 60)],
                mood=mood,
                reason="topic_hint",
            )

        return AmbientContextExtraction(useful=False, reason="low_value")

    def _extract_gameplay_facts(self, raw: str, normalized: str, now: float) -> list[dict]:
        tokens = set(normalized.split())
        if not tokens:
            return []

        facts: list[dict] = []
        healing_terms = {
            "cura", "curan", "curar", "curarse", "heal", "healing", "hp", "vida",
            "pocion", "pociones", "autopocion", "autopotion", "limon", "limones",
        }
        combat_risk_terms = {
            "vida", "hp", "muero", "morir", "muriendo", "rojo", "oneshot",
            "counter", "contraataque", "counterattack", "boss", "jefe", "enemigo",
        }
        enemy_mechanic_terms = {
            "counter", "contraataque", "counterattack", "autopocion", "autopotion",
            "aguanta", "sobrevive", "queda", "cura", "curarse", "autoheal", "autocura",
        }
        rng_terms = {"rng", "suerte", "azar", "random", "aleatorio", "dados", "dado", "parchis", "depender"}
        challenge_terms = {
            "desafio", "challenge", "level", "nivel", "exp", "experiencia", "forzado",
            "obligatorio", "sacrificio", "recursos", "recurso", "cinematica", "unskippable",
        }
        failure_terms = {"muerto", "morir", "matado", "mataron", "game", "over", "wipe", "fallado", "intento"}
        progress_terms = {"pasado", "pasamos", "derrotado", "avance", "avanzamos", "llegamos", "conseguido", "victoria"}

        if tokens & combat_risk_terms and (
            tokens & {"vida", "hp", "15", "poca", "poco", "rojo", "muero", "counter", "contraataque", "counterattack"}
        ):
            facts.append(self._category_fact(
                "combat_risk",
                "Leo is weighing a dangerous combat state with low HP or counterattack risk.",
                raw,
                normalized,
                0.86,
                now,
                mood="combat tension",
            ))
        if tokens & healing_terms:
            facts.append(self._category_fact(
                "healing_or_recovery",
                "Leo mentioned healing, HP recovery, or autopotion behavior in the fight.",
                raw,
                normalized,
                0.82,
                now,
                mood="resource tension",
            ))
        if tokens & enemy_mechanic_terms:
            facts.append(self._category_fact(
                "enemy_mechanic",
                "Leo mentioned a combat mechanic such as counters, survival, or auto-healing.",
                raw,
                normalized,
                0.84,
                now,
                mood="mechanic tension",
            ))
        if tokens & rng_terms:
            facts.append(self._category_fact(
                "rng_dependency",
                "Leo framed the current situation as dependent on RNG or luck.",
                raw,
                normalized,
                0.86,
                now,
                mood="rng tension",
            ))
        if tokens & challenge_terms and (tokens & rng_terms or tokens & {"level", "nivel", "exp", "recursos", "forzado", "obligatorio"}):
            facts.append(self._category_fact(
                "challenge_constraint",
                "Leo described a challenge constraint such as Level 1, no EXP, forced fights, or limited resources.",
                raw,
                normalized,
                0.8,
                now,
                mood="challenge tension",
            ))
        if ("game" in tokens and "over" in tokens) or tokens & {"muerto", "matado", "mataron", "wipe"}:
            facts.append(self._category_fact(
                "failure_or_death",
                "Leo mentioned death, game over, a wipe, or a failed attempt.",
                raw,
                normalized,
                0.84,
                now,
                mood="frustrated",
            ))
        if tokens & progress_terms and len(tokens) >= 4:
            facts.append(self._category_fact("progress_marker", "Leo marked concrete progress in the run.", raw, normalized, 0.7, now))

        legacy = self._extract_gameplay_category(raw, normalized, now)
        if legacy and legacy.get("category") == "navigation_confusion" and any(
            fact.get("category") in {"failure_or_death", "rng_dependency", "combat_risk"}
            for fact in facts
        ):
            legacy = None
        if legacy and not any(fact.get("category") == legacy.get("category") for fact in facts):
            facts.append(legacy)
        return facts

    def _extract_gameplay_category(self, raw: str, normalized: str, now: float) -> dict | None:
        tokens = set(normalized.split())
        if not tokens:
            return None

        healing = {"cura", "curan", "curar", "heal", "healing", "hp", "vida", "pocion", "pociones", "limon", "limones", "item", "objeto"}
        weak = {"poco", "poquisimo", "apenas", "nada", "insuficiente", "inutil", "useless", "weak", "barely"}
        attack = {"ataque", "attack", "golpe", "hostia", "dano", "damage", "critico", "crit", "oneshot"}
        surprise = {"que", "wtf", "hell", "diablos", "cono", "joder", "sorpresa", "inesperado"}
        guide = {"guia", "guide", "walkthrough", "recomienda", "dice", "decia", "said", "sugiere", "aconseja"}
        strategy = {"magia", "magic", "luz", "light", "oscura", "hielo", "fuego", "hechizo", "spell", "estrategia", "strategy", "usar", "use"}
        enemy = {"enemigo", "enemy", "boss", "jefe", "bicho", "monstruo"}
        mechanic = {"queda", "quedan", "aguanta", "sobrevive", "sobrevivir", "stays", "survive", "hp", "cura", "curarse", "heals", "heal"}
        low_hp = {"muero", "muriendo", "morir", "vida", "hp", "rojo", "red", "oneshot", "one"}
        resource = {"comida", "food", "exp", "experiencia", "recurso", "recursos", "dinero", "oro", "mana", "mp", "farmear", "farm"}
        progress = {"pasado", "pasamos", "derrotado", "avance", "avanzamos", "llegamos", "conseguido", "success", "victoria"}
        confusion = {"donde", "perdido", "perdida", "confuso", "confundido", "voy", "ir"}
        failure = {"otra", "vez", "again", "muerto", "matado", "fallado", "fallo", "intento"}
        difficulty = {"facil", "dificil", "hard", "easy", "imposible", "complicado"}

        if tokens & healing and tokens & weak:
            return self._category_fact(
                "healing_or_recovery",
                "Leo complained that a healing item barely restores enough HP.",
                raw,
                normalized,
                0.84,
                now,
                mood="resource frustration",
            )
        if tokens & guide and (tokens & strategy or tokens & difficulty):
            return self._category_fact(
                "guide_strategy",
                "A guide suggested something relevant about the current strategy or boss difficulty.",
                raw,
                normalized,
                0.82,
                now,
                mood="strategy planning",
            )
        if ({"1", "uno", "one"} & tokens) and tokens & {"hp", "vida"} and tokens & mechanic and tokens & (enemy | {"se", "lo"}):
            return self._category_fact(
                "enemy_mechanic",
                "An enemy or boss seems to survive at 1 HP and then heal.",
                raw,
                normalized,
                0.86,
                now,
                mood="mechanic confusion",
            )
        if tokens & attack and tokens & surprise:
            return self._category_fact(
                "unexpected_attack",
                "Leo was surprised by a strong or confusing enemy attack.",
                raw,
                normalized,
                0.78,
                now,
                mood="surprised",
            )
        if tokens & low_hp and tokens & {"casi", "almost", "poca", "poco", "rojo", "red"}:
            return self._category_fact("combat_risk", "Leo is low on HP or close to dying.", raw, normalized, 0.76, now, mood="danger")
        if tokens & resource and tokens & {"falta", "poco", "necesito", "sin", "gastar", "guardar", "farmear", "farm"}:
            return self._category_fact(
                "resource_management",
                "Leo is thinking about food, EXP, or resource management.",
                raw,
                normalized,
                0.72,
                now,
                mood="resource planning",
            )
        if tokens & difficulty and tokens & (enemy | {"zona", "area", "nivel"}):
            return self._category_fact(
                "boss_or_area_difficulty",
                "Leo commented on the current boss or area difficulty.",
                raw,
                normalized,
                0.72,
                now,
                mood="difficulty read",
            )
        if tokens & confusion and len(tokens) >= 4:
            return self._category_fact("navigation_confusion", "Leo is unsure where to go next.", raw, normalized, 0.7, now, mood="confused")
        if tokens & failure and len(tokens) >= 4:
            return self._category_fact("failure_or_death", "Leo mentioned repeated failure or another death.", raw, normalized, 0.68, now, mood="frustrated")
        if tokens & progress and len(tokens) >= 4:
            return self._category_fact("progress_marker", "Leo marked recent progress in the run.", raw, normalized, 0.68, now)
        return None

    def _is_generic_filler(self, normalized: str) -> bool:
        if not normalized:
            return True
        useful_terms = {
            "hp", "vida", "muerto", "matado", "game", "over", "counter", "contraataque",
            "autopocion", "autopotion", "cura", "curarse", "rng", "suerte", "dados",
            "boss", "jefe", "enemigo", "ataque", "nivel", "level", "desafio", "challenge",
            "guardar", "recargar", "forzado", "obligatorio", "cinematica", "unskippable",
        }
        if set(normalized.split()) & useful_terms:
            return False
        filler_exact = {
            "uf", "bueno", "vamos a ver", "no", "jotun", "pues nada volvemos",
            "eso hay que hacerse mirar", "pues nada", "vale", "ok",
        }
        if normalized in filler_exact:
            return True
        words = normalized.split()
        return len(words) <= 3

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
            rf"(?:son|estan|van)?\s*(\d+|{number_pattern})\s+(?:niveles?|levels?)\s+(?:mas)\s+que\s+yo",
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
            data={"level_gap": value, "source_text": raw, "raw_text": raw, "normalized_text": normalized},
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
            data={"source_text": raw, "raw_text": raw, "normalized_text": normalized},
        )

    def _extract_objective_or_location(self, raw: str, normalized: str, event_type: str | None, now: float) -> dict | None:
        if event_type == "objective_update":
            return self._fact("objective", raw[:140], 0.7, now, ttl_sec=35 * 60, data={"raw_text": raw, "normalized_text": normalized})
        if event_type == "location_update":
            return self._fact("location", raw[:100], 0.7, now, ttl_sec=35 * 60, data={"raw_text": raw, "normalized_text": normalized})
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
        data = data or {}
        category = data.get("category", kind)
        return {
            "kind": kind,
            "text": text,
            "category": category,
            "summary": text,
            "id": f"ambient:{category}:{int(now)}",
            "game": data.get("game"),
            "source": "stt_voice",
            "raw_text": data.get("raw_text"),
            "normalized_text": data.get("normalized_text"),
            "timestamp": now,
            "confidence": confidence,
            "ttl_sec": ttl_sec,
            "expires_at": now + ttl_sec,
            "data": data,
        }

    def _category_fact(
        self,
        category: str,
        summary: str,
        raw: str,
        normalized: str,
        confidence: float,
        now: float,
        *,
        ttl_sec: float = 25 * 60,
        mood: str | None = None,
    ) -> dict:
        return self._fact(
            category,
            summary,
            confidence,
            now,
            ttl_sec=ttl_sec,
            data={
                "category": category,
                "raw_text": raw,
                "normalized_text": normalized,
                "mood": mood,
            },
        )

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(text or "").lower())
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = re.sub(r"[^a-z0-9_ ]+", " ", normalized)
        return " ".join(normalized.split())
