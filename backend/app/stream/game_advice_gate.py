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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatedClaim:
    claim: str
    evidence_type: str
    evidence_id: str
    exact_supporting_text: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def detects_specific_advice(self, text: str) -> bool:
        normalized = _normalize(text)
        prescription = bool(re.search(
            r"\b(?:deberias|debes|haz|usa|guarda|espera|vende|equipa|ataca|cura|lanza|reserva|"
            r"you should|save before|wait until|use the|sell the|equip|attack when|heal before)\b",
            normalized,
        ))
        sequence = bool(re.search(r"\b(?:antes de|despues de|cuando termine|luego|then|before|after|until)\b", normalized))
        return prescription or sequence

    def validate(
        self,
        *,
        current_game: str | None,
        proposed_advice: str,
        game_run_state: dict | None = None,
        known_game_mechanics: list[str] | None = None,
        source_evidence: list[str | dict[str, Any]] | None = None,
    ) -> GameAdviceValidation:
        mechanics = self.detect_mechanics(proposed_advice)
        game = str(current_game or (game_run_state or {}).get("game") or "").strip()
        advice_detected = self.detects_specific_advice(proposed_advice)
        if not mechanics:
            allowed = not advice_detected
            result = GameAdviceValidation(
                allowed, game, [], [], ["unvalidated_specific_advice"] if advice_detected else [],
                "empty_validation_specific_advice" if advice_detected else "generic_reaction",
                confidence=0.2 if advice_detected else 0.88,
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
        provenance: dict[str, ValidatedClaim] = {}
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
            for mechanic in self.detect_mechanics(exact_text):
                evidence_mechanics.add(mechanic)
                provenance[mechanic] = ValidatedClaim(
                    mechanic, evidence_type, evidence_id, exact_text, confidence,
                )
        for mechanic in explicit:
            provenance.setdefault(mechanic, ValidatedClaim(
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
            provenance.setdefault(mechanic, ValidatedClaim(
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
