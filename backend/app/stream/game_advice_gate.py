from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path(__file__).with_name("game_mechanics_registry.json")


MECHANIC_ALIASES: dict[str, tuple[str, ...]] = {
    "autopotion": ("autopotion", "autopocion", "auto pocion", "auto-pocion", "auto potion", "autopoción"),
    "healing_items": ("potion", "pocion", "pociones", "item de cura", "objeto de cura", "healing item", "curas grandes"),
    "healing_skills": ("cura", "curar", "healing skill", "media", "diarama", "dia"),
    "hp_management": (" hp", "vida", "salud", "puntos de vida"),
    "sp_management": (" sp", "mana", "puntos de magia", "gestiona sp", "recuperar sp"),
    "mp_management": (" mp", "pm", "magia"),
    "weakness_system": ("debilidad", "weakness", "weaknesses", "punto debil", "punto débil"),
    "baton_pass": ("baton pass", "relevo"),
    "guard": ("guardia", "guard", "defenderse"),
    "persona_fusion": ("fusion", "fusión", "fusionar persona", "persona fusion"),
    "confidants": ("confidant", "confidente", "confidants"),
    "social_links": ("social link", "vinculo social", "vínculo social"),
    "palace_deadlines": ("deadline", "fecha limite", "fecha límite", "palace deadline"),
    "safe_rooms": ("safe room", "safe rooms", "sala segura", "salas seguras"),
    "stealth_security": ("sigilo", "security level", "nivel de seguridad"),
    "personas": ("persona", "personas"),
    "equipment_stats": ("defensa", "ataque", "equipo", "armadura", "weapon", "arma"),
    "party_members": ("party", "grupo", "compañero", "companero", "party member"),
    "turn_economy": ("turno", "turnos", "turn economy"),
    "materia": ("materia",),
    "limit_breaks": ("limit break", "limite", "límite"),
    "trance": ("trance",),
    "steal": ("robar", "steal"),
    "spell_slots": ("spell slot", "slot de conjuro", "espacio de conjuro"),
    "action_economy": ("accion", "acción", "action economy"),
    "bonus_actions": ("bonus action", "accion adicional", "acción adicional"),
    "saving_throws": ("saving throw", "tirada de salvacion", "tirada de salvación"),
    "dice_rolls": ("dados", "dice", "tirada"),
    "positioning": ("posicionamiento", "altura", "high ground"),
    "resting": ("descanso", "long rest", "short rest"),
    "card_deck": ("deck", "baraja", "cartas"),
    "card_values": ("valor de carta", "card value"),
    "sleights": ("sleight", "truco", "combo de cartas"),
    "enemy_cards": ("enemy card", "carta enemiga"),
    "room_cards": ("room card", "carta de sala"),
    "reload_counter": ("reload counter", "contador de recarga"),
}


UNAMBIGUOUS_MECHANIC = "UNAMBIGUOUS_MECHANIC"
CONTEXTUAL_MECHANIC = "CONTEXTUAL_MECHANIC"
ENTITY_COLLISION = "ENTITY_COLLISION"
COMMON_LANGUAGE_COLLISION = "COMMON_LANGUAGE_COLLISION"

# These categories describe lexical ambiguity only. Every alias still needs a
# mechanic assertion or instruction context before it becomes a claim.
_ENTITY_COLLISION_ALIASES = {
    "guard", "materia", "party", "persona", "personas", "trance",
}
_COMMON_LANGUAGE_COLLISION_ALIASES = {
    "accion", "action", "altura", "arma", "ataque", "baraja", "cartas",
    "compañero", "companero", "cura", "dados", "defensa", "descanso", "dia",
    "equipo", "grupo", "guardia", "limite", "magia", "media", "salud",
    "tirada", "truco", "turno", "turnos", "vida",
}
_CONTEXTUAL_MECHANIC_ALIASES = {
    "deck", "fusion", "guardia", "mana", "pocion", "pociones", "potion",
    "relevo", "robar", "steal", "weapon",
}


@dataclass(frozen=True)
class GameMechanicsProfile:
    canonical_title: str
    aliases: list[str] = field(default_factory=list)
    allowed_mechanics: set[str] = field(default_factory=set)
    forbidden_mechanics: set[str] = field(default_factory=set)
    spoiler_policy_default: str = "no_story_spoilers"
    stream_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GameAdviceValidation:
    allowed: bool
    game: str
    mechanics: list[str]
    validated: list[str]
    blocked: list[str]
    reason: str
    confidence: float = 0.0
    validated_claims: list[dict[str, Any]] = field(default_factory=list)
    entity_references: list[str] = field(default_factory=list)
    mechanic_assertions: list[str] = field(default_factory=list)
    mechanic_instructions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimSupport:
    claim: str
    evidence_type: str
    evidence_id: str
    exact_supporting_text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Backwards-compatible name used by v1.1 callers.
ValidatedClaim = ClaimSupport


@dataclass(frozen=True)
class MechanicSemanticAnalysis:
    entity_references: list[str] = field(default_factory=list)
    mechanic_assertions: list[str] = field(default_factory=list)
    mechanic_instructions: list[str] = field(default_factory=list)
    advice_detected: bool = False

    @property
    def mechanics(self) -> list[str]:
        return sorted(dict.fromkeys([
            *self.mechanic_assertions,
            *self.mechanic_instructions,
        ]))


def _normalize(value: str) -> str:
    raw = str(value or "").casefold()
    raw = "".join(
        char for char in unicodedata.normalize("NFKD", raw)
        if not unicodedata.combining(char)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", raw).split())


class GameMechanicsRegistry:
    def __init__(self, path: Path | None = None):
        self.path = path or REGISTRY_PATH
        self._profiles = self._load()

    def lookup(self, game: str | None) -> GameMechanicsProfile | None:
        key = _normalize(game or "")
        if not key:
            return None
        for profile in self._profiles:
            keys = [_normalize(profile.canonical_title), *(_normalize(alias) for alias in profile.aliases)]
            if key in keys or any(alias and alias in key for alias in keys):
                return profile
        return None

    def _load(self) -> list[GameMechanicsProfile]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            data = {"games": []}
        profiles: list[GameMechanicsProfile] = []
        for item in data.get("games") or []:
            profiles.append(GameMechanicsProfile(
                canonical_title=str(item.get("canonical_title") or ""),
                aliases=[str(alias) for alias in item.get("aliases") or []],
                allowed_mechanics={str(mech) for mech in item.get("allowed_mechanics") or []},
                forbidden_mechanics={str(mech) for mech in item.get("forbidden_mechanics") or []},
                spoiler_policy_default=str(item.get("spoiler_policy_default") or "no_story_spoilers"),
                stream_notes=[str(note) for note in item.get("stream_notes") or []],
            ))
        return profiles


class GameAdviceGate:
    _SUBSTANTIVE_ASSERTION_PATTERNS: dict[str, tuple[str, ...]] = {
        "counterattack": (r"\b(?:contraataques?|counterattacks?)\b",),
        "automatic_healing": (r"\b(?:curacion(?:es)? automatica(?:s)?|autohealing|auto heal)\b",),
        "regeneration": (r"\b(?:regeneracion|regeneration|regenera)\b",),
        "weakness": (r"\b(?:debilidad(?:es)?|weakness(?:es)?)\b",),
        "invulnerability": (r"\b(?:invulnerabilidad|invulnerability|invulnerable)\b",),
        "status_effect": (r"\b(?:efecto(?:s)? de estado|status effects?)\b",),
    }
    _CLAIM_PATTERNS: dict[str, tuple[str, ...]] = {
        "save_instruction": (
            r"\b(?:guarda|guardar|guardalo|salva|salvar|haz un guardado|save|save it)\b",
        ),
        "heal_instruction": (
            r"\b(?:cura|curar|curate|curarte|curalo|recupera vida|heal|heal up|restore health)\b",
        ),
        "buy_instruction": (r"\b(?:compra|compralo|buy|purchase)\b",),
        "equip_instruction": (r"\b(?:equipa|equipate|ponte|equip|wear)\b",),
        "use_instruction": (r"\b(?:usa|utiliza|gastalo|use|spend it)\b",),
        "attack_instruction": (r"\b(?:ataca|golpealo|dale|attack|hit it|strike)\b",),
        "movement_instruction": (r"\b(?:ve a|vete a|entra en|sal de|gira|go to|head to|enter|leave|turn left|turn right)\b",),
        "wait_instruction": (r"\b(?:espera|aguanta|no ataques aun|wait|hold on|do not attack yet|dont attack yet)\b",),
        "stash_instruction": (r"\b(?:reserva|guardate|almacena|stash|keep it for later|hold onto)\b",),
        "level_up_condition": (r"\b(?:sube de nivel|subir de nivel|farmea|farmear|entrena|entrenar|level up|grind|gain levels)\b",),
        "enemy_alive_assumption": (
            r"\b(?:le queda poca vida|esta casi muerto|casi lo tienes|sigue vivo|"
            r"low hp|almost dead|nearly dead|you almost have it|still alive)\b",
        ),
    }
    _RECOMMENDATION = re.compile(
        r"\b(?:deberias|debes|tienes que|te conviene|mejor|recuerda|no olvides|"
        r"you should|you need to|you had better|remember|do not forget|dont forget)\b"
    )
    _MECHANIC_ASSERTION_PREDICATE = re.compile(
        r"\b(?:"
        r"aument[a-z0-9]*|increment[a-z0-9]*|reduc[a-z0-9]*|disminu[a-z0-9]*|"
        r"restaur[a-z0-9]*|recuper[a-z0-9]*|consum[a-z0-9]*|gast[a-z0-9]*|"
        r"otorg[a-z0-9]*|conced[a-z0-9]*|permit[a-z0-9]*|bloque[a-z0-9]*|"
        r"duplic[a-z0-9]*|triplic[a-z0-9]*|mejor[a-z0-9]*|empeor[a-z0-9]*|"
        r"activ[a-z0-9]*|desactiv[a-z0-9]*|desencaden[a-z0-9]*|provoc[a-z0-9]*|"
        r"caus[a-z0-9]*|requier[a-z0-9]*|fusion[a-z0-9]*|combin[a-z0-9]*|"
        r"increase[a-z0-9]*|reduce[a-z0-9]*|restore[a-z0-9]*|recover[a-z0-9]*|"
        r"consume[a-z0-9]*|grant[a-z0-9]*|allow[a-z0-9]*|block[a-z0-9]*|"
        r"trigger[a-z0-9]*|activate[a-z0-9]*|deactivate[a-z0-9]*|require[a-z0-9]*|"
        r"fuse[a-z0-9]*|combine[a-z0-9]*|heal[a-z0-9]*"
        r")\b|\bcura\s+(?:el\s+|los\s+)?(?:hp|vida|salud|puntos de vida)\b"
    )
    _MECHANIC_IMPERATIVE = re.compile(
        r"^(?:por favor\s+)?(?:fusiona|fusionad|fuse|activa|activate|combina|combine)\b"
    )

    def __init__(self, registry: GameMechanicsRegistry | None = None):
        self.registry = registry or GameMechanicsRegistry()

    def detect_mechanics(self, text: str) -> list[str]:
        normalized = f" {_normalize(text)} "
        found: list[str] = []
        for mechanic, aliases in MECHANIC_ALIASES.items():
            for alias in aliases:
                alias_norm = _normalize(alias)
                if alias_norm and f" {alias_norm} " in normalized:
                    found.append(mechanic)
                    break
        return sorted(dict.fromkeys(found))

    def alias_classification(self, alias: str) -> str:
        normalized = _normalize(alias)
        if normalized in _ENTITY_COLLISION_ALIASES:
            return ENTITY_COLLISION
        if normalized in _COMMON_LANGUAGE_COLLISION_ALIASES:
            return COMMON_LANGUAGE_COLLISION
        if normalized in _CONTEXTUAL_MECHANIC_ALIASES:
            return CONTEXTUAL_MECHANIC
        return UNAMBIGUOUS_MECHANIC

    def analyze_semantics(
        self,
        text: str,
        *,
        entity_spans: list[str] | None = None,
    ) -> MechanicSemanticAnalysis:
        references: list[str] = []
        assertions: list[str] = []
        instructions: list[str] = []
        advice_detected = False
        clauses = [part for part in re.split(r"(?:[.!?;:]+|\s*,\s*)", str(text or "")) if _normalize(part)]

        for clause in clauses:
            normalized = _normalize(clause)
            substantive = self.extract_substantive_claims(normalized)
            clause_advice = bool(
                self.detects_specific_advice(normalized)
                or self._MECHANIC_IMPERATIVE.search(normalized)
            )
            mechanical_predicate = bool(self._MECHANIC_ASSERTION_PREDICATE.search(normalized))
            semantic_context = bool(clause_advice or substantive or mechanical_predicate)
            masked, found_references = self._mask_entity_references(
                normalized,
                entity_spans or [],
                semantic_context=semantic_context,
            )
            references.extend(found_references)

            substantive = self.extract_substantive_claims(masked)
            clause_advice = bool(
                self.detects_specific_advice(masked)
                or self._MECHANIC_IMPERATIVE.search(masked)
            )
            mechanical_predicate = bool(self._MECHANIC_ASSERTION_PREDICATE.search(masked))
            lexical_mechanics = self.detect_mechanics(masked)
            advice_detected = advice_detected or clause_advice

            if clause_advice:
                instructions.extend(substantive)
                instructions.extend(lexical_mechanics)
            else:
                assertions.extend(substantive)
                if mechanical_predicate:
                    assertions.extend(lexical_mechanics)

        return MechanicSemanticAnalysis(
            entity_references=list(dict.fromkeys(references)),
            mechanic_assertions=sorted(dict.fromkeys(assertions)),
            mechanic_instructions=sorted(dict.fromkeys(instructions)),
            advice_detected=advice_detected,
        )

    def _mask_entity_references(
        self,
        normalized_clause: str,
        entity_spans: list[str],
        *,
        semantic_context: bool,
    ) -> tuple[str, list[str]]:
        masked = normalized_clause
        references: list[str] = []
        ordered = sorted(
            (str(span).strip() for span in entity_spans if str(span).strip()),
            key=lambda span: len(_normalize(span).split()),
            reverse=True,
        )
        for span in ordered:
            normalized_span = _normalize(span)
            if not normalized_span:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(normalized_span)}(?![a-z0-9])"
            if not re.search(pattern, masked):
                continue
            references.append(span)
            # Multi-token titles are protected as a unit. A single ambiguous
            # token is protected only for a pure reference; in a mechanic
            # proposition it remains available to claim detection.
            if len(normalized_span.split()) > 1 or not semantic_context:
                masked = re.sub(pattern, " ", masked)
                masked = " ".join(masked.split())
        return masked, references

    def detects_specific_advice(self, text: str) -> bool:
        normalized = _normalize(text)
        prescription = bool(re.search(
            r"\b(?:deberias|debes|tienes que|te conviene|mejor|recuerda|no olvides|haz|usa|utiliza|"
            r"guarda|salva|espera|aguanta|vende|compra|equipa|ataca|cura|lanza|reserva|ve a|vete a|"
            r"you should|you need to|you had better|remember|do not forget|dont forget|save|wait|"
            r"use|sell|buy|equip|attack|heal|stash|go to|head to|hold onto)\b",
            normalized,
        ))
        sequence = bool(re.search(r"\b(?:antes de|despues de|cuando termine|luego|then|before|after|until)\b", normalized))
        return prescription or sequence

    def extract_substantive_claims(self, text: str) -> list[str]:
        normalized = _normalize(text)
        claims = [
            claim
            for claim, patterns in self._CLAIM_PATTERNS.items()
            if any(re.search(pattern, normalized) for pattern in patterns)
        ]
        claims.extend(
            claim
            for claim, patterns in self._SUBSTANTIVE_ASSERTION_PATTERNS.items()
            if any(re.search(pattern, normalized) for pattern in patterns)
        )
        if "save_instruction" in claims and (
            re.search(r"\bguarda(?:r)?\s+(?:sp|mp|mana|pm|recursos?|objetos?|items?)\b", normalized)
            or re.search(r"\bguarda\b.*\ben mente\b", normalized)
        ):
            claims.remove("save_instruction")
        # An attack instruction necessarily assumes that the target/combat is still active.
        if "attack_instruction" in claims and "enemy_alive_assumption" not in claims:
            claims.append("enemy_alive_assumption")
        return sorted(dict.fromkeys(claims))

    def validate(
        self,
        *,
        current_game: str | None,
        proposed_advice: str,
        game_run_state: dict | None = None,
        known_game_mechanics: list[str] | None = None,
        source_evidence: list[str | dict[str, Any]] | None = None,
        entity_spans: list[str] | None = None,
    ) -> GameAdviceValidation:
        semantic = self.analyze_semantics(proposed_advice, entity_spans=entity_spans)
        mechanics = semantic.mechanics
        game = str(current_game or (game_run_state or {}).get("game") or "").strip()
        advice_detected = semantic.advice_detected
        if not mechanics:
            allowed = not advice_detected
            result = GameAdviceValidation(
                allowed, game, [], [], ["unvalidated_specific_advice"] if advice_detected else [],
                "empty_validation_specific_advice" if advice_detected else "generic_reaction",
                confidence=0.2 if advice_detected else 0.88,
                entity_references=semantic.entity_references,
                mechanic_assertions=semantic.mechanic_assertions,
                mechanic_instructions=semantic.mechanic_instructions,
            )
            print(
                "[HEBE][GAME_ADVICE_GATE] "
                f"advice_detected={str(advice_detected).lower()} claims={mechanics!r} "
                f"validated=[] decision={'allow' if allowed else 'rewrite_reaction'}",
                flush=True,
            )
            return result

        profile = self.registry.lookup(game)
        explicit = {str(item) for item in (known_game_mechanics or [])}
        evidence_mechanics: set[str] = set()
        provenance: dict[str, ClaimSupport] = {}
        allowed_evidence_types = {
            "raw_owner_evidence", "confirmed_game_knowledge",
            "current_structured_game_state", "external_validated_mechanic",
        }
        for index, item in enumerate(source_evidence or []):
            if isinstance(item, dict):
                evidence_type = str(item.get("evidence_type") or item.get("type") or "")
                evidence_id = str(item.get("evidence_id") or item.get("fact_id") or f"evidence:{index}")
                exact_text = str(item.get("exact_supporting_text") or item.get("raw_text") or "")
                confidence = float(item.get("confidence", 0.0) or 0.0)
                if evidence_type not in allowed_evidence_types:
                    continue
            else:
                evidence_type = "raw_owner_evidence"
                evidence_id = f"raw:{index}"
                exact_text = str(item or "")
                confidence = 0.8
            for mechanic in [*self.detect_mechanics(exact_text), *self.extract_substantive_claims(exact_text)]:
                evidence_mechanics.add(mechanic)
                provenance[mechanic] = ClaimSupport(
                    mechanic, evidence_type, evidence_id, exact_text, confidence,
                )
        for mechanic in explicit:
            provenance.setdefault(mechanic, ClaimSupport(
                mechanic, "confirmed_game_knowledge", f"game:{_normalize(game)}:{mechanic}",
                mechanic, 0.9,
            ))

        if profile is None:
            validated = sorted(set(mechanics) & (explicit | evidence_mechanics))
            blocked = [mechanic for mechanic in mechanics if mechanic not in set(validated)]
            result = GameAdviceValidation(
                allowed=not blocked,
                game=game or "unknown",
                mechanics=mechanics,
                validated=validated,
                blocked=blocked,
                reason="unknown_game_requires_source" if blocked else "validated_by_source",
                confidence=0.55 if blocked else 0.82,
                validated_claims=[provenance[item].to_dict() for item in validated if item in provenance],
                entity_references=semantic.entity_references,
                mechanic_assertions=semantic.mechanic_assertions,
                mechanic_instructions=semantic.mechanic_instructions,
            )
            print(
                "[HEBE][GAME_ADVICE_GATE] "
                f"advice_detected={str(advice_detected).lower()} claims={mechanics!r} "
                f"validated={validated!r} decision={'allow' if result.allowed else 'rewrite_reaction'}",
                flush=True,
            )
            return result

        allowed = set(profile.allowed_mechanics) | explicit | evidence_mechanics
        forbidden = set(profile.forbidden_mechanics)
        blocked = [mechanic for mechanic in mechanics if mechanic in forbidden or mechanic not in allowed]
        validated = [mechanic for mechanic in mechanics if mechanic not in blocked]
        for mechanic in validated:
            provenance.setdefault(mechanic, ClaimSupport(
                mechanic, "confirmed_game_knowledge", f"profile:{_normalize(profile.canonical_title)}:{mechanic}",
                mechanic, 0.92,
            ))
        result = GameAdviceValidation(
            allowed=not blocked,
            game=profile.canonical_title,
            mechanics=mechanics,
            validated=validated,
            blocked=blocked,
            reason="mechanic_not_validated" if blocked else "validated_for_game",
            confidence=0.92 if not blocked else 0.35,
            validated_claims=[provenance[item].to_dict() for item in validated if item in provenance],
            entity_references=semantic.entity_references,
            mechanic_assertions=semantic.mechanic_assertions,
            mechanic_instructions=semantic.mechanic_instructions,
        )
        print(
            "[HEBE][GAME_ADVICE_GATE] "
            f"advice_detected={str(advice_detected).lower()} claims={mechanics!r} "
            f"validated={validated!r} decision={'allow' if result.allowed else 'rewrite_reaction'}",
            flush=True,
        )
        return result


class ReactionFirstContributionPolicy:
    def choose_mode(
        self, *, current_game: str | None, grounded_mechanics: list[str] | None = None,
        validated_mechanics: list[str] | None = None, spoiler_safe: bool = True,
    ) -> str:
        grounded = list(grounded_mechanics or [])
        validated = list(validated_mechanics or [])
        if grounded and validated and spoiler_safe and set(grounded) <= set(validated):
            return "validated_tip"
        if str(current_game or "").strip():
            return "contextual_reaction"
        return "contextual_reaction"
