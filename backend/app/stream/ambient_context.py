from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
import unicodedata
from typing import Any


@dataclass(frozen=True)
class GameplayReferentResolution:
    subject: str = "unknown"
    object: str = ""
    predicate: str = ""
    confidence: float = 0.0
    decision: str = "ambiguous"


class GameplayReferentResolver:
    def resolve(self, raw: str, normalized: str, *, recent_fragments: list[str] | None = None) -> GameplayReferentResolution:
        text = str(normalized or "")
        if re.search(r"\b(?:solo\s+)?queda\s+(?:ese|esa|uno|una)\b", text):
            result = GameplayReferentResolution("unknown_game_entity", "remaining_count", "one_remaining", 0.72, "uncertain_singular_game_entity")
        elif re.search(r"\b(?:estos|esos|aquellos)\s+(?:son|estan)\s+(?:de\s+)?nivel\s+bajo\b", text):
            result = GameplayReferentResolution("enemies", "level", "low_level", 0.78, "plural_game_entity")
        elif re.search(r"\b(?:yo\s+)?(?:estoy|voy|mi personaje esta)\s+(?:a\s+|en\s+)?nivel\s+(?:1|uno)\b", text):
            result = GameplayReferentResolution("owner_player", "level", "level_one", 0.96, "explicit_first_person")
        elif re.search(r"\bno\s+se\s+le\s+baja\s+(?:la\s+)?(?:barra\s+de\s+)?vida\b", text):
            result = GameplayReferentResolution("enemies", "health_bar", "health_not_decreasing", 0.82, "indirect_enemy_object")
        elif re.search(r"\b(?:enemigo|boss|jefe|bicho).*\b(?:se\s+cura|recupera\s+(?:hp|vida)|regenera)\b|\bse\s+cura\b", text):
            result = GameplayReferentResolution("enemies", "health", "heals", 0.88, "explicit_enemy_recovery")
        elif re.search(r"\b(?:me|mi personaje|yo).*\b(?:vida|hp)\b|\b(?:me curo|recupero vida)\b", text):
            result = GameplayReferentResolution("owner_player", "health", "owner_health_state", 0.86, "explicit_first_person")
        elif re.search(r"\b(?:ellos|estos|esos|enemigos|boss|jefe)\b", text):
            result = GameplayReferentResolution("enemies", "", "observation", 0.68, "game_entity_reference")
        else:
            result = GameplayReferentResolution()
        print(
            "[HEBE][GAMEPLAY_REFERENT] "
            f"raw={raw!r} subject={result.subject} predicate={result.predicate or 'unknown'} "
            f"confidence={result.confidence:.3f} decision={result.decision}",
            flush=True,
        )
        return result


@dataclass(frozen=True)
class AmbientFact:
    fact_id: str
    raw_text: str
    conservative_normalized_text: str
    utterance_role: str
    timestamp: float
    topic_id: str
    category: str
    extracted_subject: str
    extracted_object: str
    extracted_predicate: str
    confidence: float
    inference_level: str
    supported_claims: list[str]
    inferred_claims: list[str]
    unsupported_claims: list[str]
    evidence_span: str
    evidence_tokens: list[str]
    semantic_rule: str
    model_reason: str
    expires_at: float
    scene_id: str = ""

    @property
    def subject(self) -> str:
        return self.extracted_subject

    @property
    def predicate(self) -> str:
        return self.extracted_predicate

    @property
    def object(self) -> str:
        return self.extracted_object

    @property
    def referent_confidence(self) -> float:
        return self.confidence

    @property
    def directly_supported_claims(self) -> list[str]:
        return self.supported_claims

    @property
    def heuristic_category(self) -> str:
        return self.category

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "raw_text": self.raw_text,
            "conservative_normalized_text": self.conservative_normalized_text,
            "utterance_role": self.utterance_role,
            "timestamp": self.timestamp,
            "topic_id": self.topic_id,
            "category": self.category,
            "heuristic_category": self.category,
            "extracted_subject": self.extracted_subject,
            "subject": self.extracted_subject,
            "extracted_object": self.extracted_object,
            "object": self.extracted_object,
            "extracted_predicate": self.extracted_predicate,
            "predicate": self.extracted_predicate,
            "confidence": self.confidence,
            "referent_confidence": self.confidence,
            "inference_level": self.inference_level,
            "supported_claims": list(self.supported_claims),
            "directly_supported_claims": list(self.supported_claims),
            "inferred_claims": list(self.inferred_claims),
            "unsupported_claims": list(self.unsupported_claims),
            "evidence_span": self.evidence_span,
            "evidence_tokens": list(self.evidence_tokens),
            "semantic_rule": self.semantic_rule,
            "model_reason": self.model_reason,
            "expires_at": self.expires_at,
            "scene_id": self.scene_id,
        }


@dataclass(frozen=True)
class AmbientContextExtraction:
    useful: bool
    facts: list[dict] = field(default_factory=list)
    mood: str | None = None
    reason: str = ""


class AmbientContextExtractor:
    """Extract current stream/run facts from accepted ambient STT."""

    def __init__(self, referent_resolver: GameplayReferentResolver | None = None) -> None:
        self.referent_resolver = referent_resolver or GameplayReferentResolver()

    def extract(
        self,
        text: str,
        *,
        event_type: str | None = None,
        now: float | None = None,
        utterance_role: str = "owner_commentary",
        language: str | None = None,
        topic_id: str | None = None,
        scene_id: str | None = None,
    ) -> AmbientContextExtraction:
        raw = str(text or "").strip()
        normalized = self._normalize(raw)
        if not normalized:
            return AmbientContextExtraction(useful=False, reason="empty")
        if utterance_role in {"game_audio_bleed", "conversational_filler", "uncertain"}:
            return AmbientContextExtraction(useful=False, reason=f"role_excluded:{utterance_role}")
        if utterance_role == "quoted_or_read_dialogue":
            now = time.time() if now is None else float(now)
            return AmbientContextExtraction(
                useful=True,
                facts=[self._fact(
                    "transient_game_narrative", raw[:180], 0.3, now, ttl_sec=30,
                    data={
                        "raw_text": raw, "normalized_text": normalized,
                        "utterance_role": utterance_role, "language": language,
                        "topic_id": topic_id, "proactive_eligible": False,
                    },
                )],
                reason="quoted_dialogue_transient_context",
            )

        now = time.time() if now is None else float(now)
        facts: list[dict] = []
        mood: str | None = None
        referent = self.referent_resolver.resolve(raw, normalized)

        gameplay_facts = self._extract_gameplay_facts(raw, normalized, now, referent=referent)
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
            for fact in facts:
                fact["utterance_role"] = utterance_role
                fact["language"] = language
                fact["topic_id"] = topic_id
                fact["raw_evidence"] = fact.get("raw_text") or raw
                fact["normalized_evidence"] = fact.get("normalized_text") or normalized
                fact.setdefault("conservative_normalized_text", normalized)
                fact.setdefault("utterance_role", utterance_role)
                fact.setdefault("topic_id", topic_id or "")
                fact["scene_id"] = str(scene_id or fact.get("scene_id") or "")
                fact.setdefault("inferred_claims", [])
                fact.setdefault("unsupported_claims", [])
                fact.setdefault("supported_claims", [raw])
                if isinstance(fact.get("data"), dict):
                    fact["data"]["scene_id"] = fact["scene_id"]
                    fact["data"]["inferred_claims"] = list(fact.get("inferred_claims") or [])
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

    def _extract_gameplay_facts(
        self, raw: str, normalized: str, now: float, *,
        referent: GameplayReferentResolution,
    ) -> list[dict]:
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
            "aguanta", "sobrevive", "cura", "curarse", "autoheal", "autocura",
        }
        rng_terms = {"rng", "suerte", "azar", "random", "aleatorio", "dados", "dado", "parchis", "depender"}
        challenge_terms = {
            "desafio", "challenge", "level", "nivel", "exp", "experiencia", "forzado",
            "obligatorio", "sacrificio", "recursos", "recurso", "cinematica", "unskippable",
        }
        failure_terms = {"muerto", "morir", "matado", "mataron", "game", "over", "wipe", "fallado", "intento"}
        progress_terms = {"pasado", "pasamos", "derrotado", "avance", "avanzamos", "llegamos", "conseguido", "victoria"}

        first_person_health = referent.subject == "owner_player" and bool(
            tokens & {"vida", "hp", "15", "poca", "poco", "rojo", "muero"}
        )
        ground_counter = bool(re.search(
            r"\b(?:si|cuando|al)\s+(?:toca|toque|cae|caiga|llega|llegue)\s+(?:el\s+)?suelo\b.*\b(?:contraataca|contraataque|counterattack|counter)\b",
            normalized,
        ))
        if first_person_health:
            facts.append(self._category_fact(
                "combat_risk",
                "Leo said his character's health is at risk.",
                raw,
                normalized,
                0.86,
                now,
                mood="combat tension",
                referent=referent,
                supported_claims=["owner health is at risk"],
            ))
        if ground_counter:
            facts.append(self._category_fact(
                "enemy_mechanic",
                "Leo said touching the ground may trigger a counterattack.",
                raw, normalized, 0.9, now, mood="mechanic tension", referent=referent,
                supported_claims=["touching the ground may trigger a counterattack"],
                inferred_claims=[],
                unsupported_claims=["the counterattack always hits", "the counterattack causes death", "the enemy heals"],
            ))
        explicit_healing = bool(
            re.search(r"\b(?:cura|curan|curar|curarse|se cura|recupera|recuperar|regenera|heal|healing|autopocion|autopotion|pocion|pociones)\b", normalized)
        )
        hp_not_decreasing = referent.predicate == "health_not_decreasing"
        if referent.predicate == "one_remaining":
            facts.append(self._category_fact(
                "remaining_entity_observation",
                "One referenced game entity remains; the referent is uncertain.",
                raw,
                normalized,
                referent.confidence,
                now,
                mood="combat tension",
                referent=referent,
                supported_claims=["one referenced game entity remains"],
                unsupported_claims=["auto_healing", "counterattack", "healing", "regeneration"],
            ))
        if explicit_healing and not hp_not_decreasing:
            healing_summary = (
                "Leo mentioned automatic potion behavior in the fight."
                if tokens & {"autopocion", "autopotion"}
                else "Leo mentioned healing or health recovery in the fight."
            )
            facts.append(self._category_fact(
                "healing_or_recovery",
                healing_summary,
                raw,
                normalized,
                0.82,
                now,
                mood="resource tension",
                referent=referent,
                supported_claims=[raw],
            ))
        if tokens & enemy_mechanic_terms and not hp_not_decreasing and not ground_counter:
            if tokens & {"counter", "contraataque", "counterattack"}:
                mechanic_summary = "Leo mentioned a counterattack mechanic."
            elif tokens & {"autopocion", "autopotion", "autoheal", "autocura"}:
                mechanic_summary = "Leo mentioned an automatic recovery mechanic."
            elif tokens & {"aguanta", "sobrevive"}:
                mechanic_summary = "Leo mentioned that an enemy survives the current attack."
            else:
                mechanic_summary = "Leo mentioned a healing mechanic."
            facts.append(self._category_fact(
                "enemy_mechanic",
                mechanic_summary,
                raw,
                normalized,
                0.84,
                now,
                mood="mechanic tension",
                referent=referent,
                supported_claims=[raw],
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
                referent=referent,
            ))
        explicit_owner_constraint = bool(
            referent.subject == "owner_player" and referent.predicate == "level_one"
            or re.search(r"\b(?:sin exp|no exp|no gano exp|desafio|challenge|limitado a|sin usar|no puedo usar|forzado|obligatorio)\b", normalized)
        )
        if explicit_owner_constraint and tokens & challenge_terms:
            facts.append(self._category_fact(
                "challenge_constraint",
                "Leo described a challenge constraint such as Level 1, no EXP, forced fights, or limited resources.",
                raw,
                normalized,
                0.8,
                now,
                mood="challenge tension",
                referent=referent,
                supported_claims=["owner is explicitly constrained to level one"] if referent.predicate == "level_one" else [raw],
            ))
        if hp_not_decreasing:
            facts.append(self._category_fact(
                "uncertain_combat_observation",
                "The referenced enemy health bar is not decreasing.",
                raw,
                normalized,
                0.78,
                now,
                mood="mechanic uncertainty",
                referent=referent,
                supported_claims=["enemy health is not decreasing"],
                unsupported_claims=["autopotion", "healing", "regeneration", "player low HP"],
            ))
        if referent.predicate == "low_level":
            facts.append(self._category_fact(
                "entity_level_observation",
                "The referenced enemies are low level.",
                raw,
                normalized,
                0.78,
                now,
                referent=referent,
                supported_claims=["the referenced enemies are low level"],
                unsupported_claims=["Leo is low level", "Level 1 challenge", "no EXP run"],
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
        confusion = {"donde", "adonde", "perdido", "perdida", "confuso", "confundido"}
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
        explicit_navigation_confusion = bool(
            tokens & confusion
            or re.search(r"\bno se (?:por )?donde (?:ir|voy|seguir)\b", normalized)
        )
        if explicit_navigation_confusion and len(tokens) >= 4:
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
        fact_id = f"ambient:{category}:{int(now)}"
        raw_text = str(data.get("raw_text") or "")
        normalized_text = str(data.get("normalized_text") or "")
        ambient_fact = AmbientFact(
            fact_id=fact_id,
            raw_text=raw_text,
            conservative_normalized_text=normalized_text,
            utterance_role=str(data.get("utterance_role") or "owner_commentary"),
            timestamp=now,
            topic_id=str(data.get("topic_id") or ""),
            category=str(category),
            extracted_subject=str(data.get("extracted_subject") or "unknown"),
            extracted_object=str(data.get("extracted_object") or ""),
            extracted_predicate=str(data.get("extracted_predicate") or ""),
            confidence=confidence,
            inference_level=str(data.get("inference_level") or "direct_observation"),
            supported_claims=[str(item) for item in data.get("supported_claims") or ([raw_text] if raw_text else [])],
            inferred_claims=[str(item) for item in data.get("inferred_claims") or []],
            unsupported_claims=[str(item) for item in data.get("unsupported_claims") or []],
            evidence_span=str(data.get("evidence_span") or raw_text),
            evidence_tokens=[str(item) for item in data.get("evidence_tokens") or normalized_text.split()],
            semantic_rule=str(data.get("semantic_rule") or f"category:{category}"),
            model_reason=str(data.get("model_reason") or f"Observed utterance supports {category}."),
            expires_at=now + ttl_sec,
            scene_id=str(data.get("scene_id") or ""),
        )
        return {
            "kind": kind,
            "text": text,
            "category": category,
            "summary": text,
            "id": fact_id,
            **ambient_fact.to_dict(),
            "game": data.get("game"),
            "source": "stt_voice",
            "raw_evidence": raw_text,
            "normalized_text": normalized_text,
            "normalized_evidence": normalized_text,
            "utterance_role": data.get("utterance_role", "owner_commentary"),
            "topic_id": data.get("topic_id"),
            "language": data.get("language"),
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
        ttl_sec: float | None = None,
        mood: str | None = None,
        referent: GameplayReferentResolution | None = None,
        supported_claims: list[str] | None = None,
        inferred_claims: list[str] | None = None,
        unsupported_claims: list[str] | None = None,
    ) -> dict:
        if ttl_sec is None:
            ttl_sec = {
                "combat_risk": 60,
                "healing_or_recovery": 120,
                "enemy_mechanic": 120,
                "failure_or_death": 30,
                "navigation_confusion": 60,
                "progress_marker": 60,
                "rng_dependency": 120,
                "challenge_constraint": 120,
            }.get(category, 60)
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
                "extracted_subject": (referent or GameplayReferentResolution()).subject,
                "extracted_object": (referent or GameplayReferentResolution()).object,
                "extracted_predicate": (referent or GameplayReferentResolution()).predicate,
                "inference_level": "direct_observation" if (referent or GameplayReferentResolution()).confidence >= 0.75 else "heuristic",
                "supported_claims": supported_claims or [raw],
                "inferred_claims": inferred_claims or [],
                "unsupported_claims": unsupported_claims or [],
                "evidence_span": raw,
                "evidence_tokens": normalized.split(),
                "semantic_rule": f"ambient_category:{category}",
                "model_reason": summary,
            },
        )

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(text or "").lower())
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = re.sub(r"[^a-z0-9_ ]+", " ", normalized)
        return " ".join(normalized.split())
