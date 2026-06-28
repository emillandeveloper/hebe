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

    def validate(
        self,
        *,
        current_game: str | None,
        proposed_advice: str,
        game_run_state: dict | None = None,
        known_game_mechanics: list[str] | None = None,
        source_evidence: list[str] | None = None,
    ) -> GameAdviceValidation:
        mechanics = self.detect_mechanics(proposed_advice)
        game = str(current_game or (game_run_state or {}).get("game") or "").strip()
        if not mechanics:
            return GameAdviceValidation(True, game, [], [], [], "no_specific_mechanics", confidence=0.88)

        profile = self.registry.lookup(game)
        explicit = {str(item) for item in (known_game_mechanics or [])}
        evidence_text = " ".join(str(item or "") for item in (source_evidence or []))
        evidence_mechanics = set(self.detect_mechanics(evidence_text))

        if profile is None:
            validated = sorted(set(mechanics) & (explicit | evidence_mechanics))
            blocked = [mechanic for mechanic in mechanics if mechanic not in set(validated)]
            return GameAdviceValidation(
                allowed=not blocked,
                game=game or "unknown",
                mechanics=mechanics,
                validated=validated,
                blocked=blocked,
                reason="unknown_game_requires_source" if blocked else "validated_by_source",
                confidence=0.55 if blocked else 0.82,
            )

        allowed = set(profile.allowed_mechanics) | explicit | evidence_mechanics
        forbidden = set(profile.forbidden_mechanics)
        blocked = [mechanic for mechanic in mechanics if mechanic in forbidden or mechanic not in allowed]
        validated = [mechanic for mechanic in mechanics if mechanic not in blocked]
        return GameAdviceValidation(
            allowed=not blocked,
            game=profile.canonical_title,
            mechanics=mechanics,
            validated=validated,
            blocked=blocked,
            reason="mechanic_not_validated" if blocked else "validated_for_game",
            confidence=0.92 if not blocked else 0.35,
        )
