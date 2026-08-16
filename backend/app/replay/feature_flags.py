from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CognitiveFeatureFlags:
    cognitive_v2_enabled: bool = False
    cognitive_replay_enabled: bool = False
    conversation_continuity_v2: bool = False
    # Historical fixture metadata. Game is canonical unconditionally; these
    # values no longer gate production or replay behavior.
    game_context_v2: bool = False
    game_run_v2_reads: bool = False
    game_run_v2_writes: bool = False
    game_knowledge_v2_reads: bool = False
    game_knowledge_v2_writes: bool = False
    game_research_memory_first: bool = False
    # Historical fixture metadata only. Social runtime no longer branches on
    # these values after the Phase 1D ownership cutover.
    social_world_v2: bool = False
    social_identity_v2: bool = False
    social_episode_writes_v2: bool = False
    social_retrieval_v2: bool = False
    shared_culture_v2: bool = False
    social_thread_opportunities_v2: bool = False
    consolidation_v2: bool = False
    consolidation_commits_v2: bool = False
    hebe_self_v2: bool = False
    owner_preferences_v2: bool = False
    leo_language_v2: bool = False
    temporal_relevance_v2: bool = False
    schedule_learning_v2: bool = False
    scene_consequence_v2: bool = False
    historical_action_ledger_v2: bool = False

    @classmethod
    def from_value(cls, value: dict[str, Any] | None) -> "CognitiveFeatureFlags":
        raw = dict(value or {})
        # Historical replay fixtures may still carry these now-retired cutover
        # flags. Canonical beliefs are unconditional, so the values are ignored.
        raw.pop("belief_v2_reads", None)
        raw.pop("belief_v2_writes", None)
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown feature flags: {', '.join(unknown)}")
        return cls(**{key: bool(raw.get(key, False)) for key in allowed})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)
