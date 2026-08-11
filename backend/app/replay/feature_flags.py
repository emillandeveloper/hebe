from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CognitiveFeatureFlags:
    cognitive_v2_enabled: bool = False
    cognitive_replay_enabled: bool = False
    conversation_continuity_v2: bool = False
    belief_v2_reads: bool = False
    belief_v2_writes: bool = False
    game_context_v2: bool = False
    social_world_v2: bool = False
    consolidation_v2: bool = False

    @classmethod
    def from_value(cls, value: dict[str, Any] | None) -> "CognitiveFeatureFlags":
        raw = dict(value or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown feature flags: {', '.join(unknown)}")
        return cls(**{key: bool(raw.get(key, False)) for key in allowed})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)
