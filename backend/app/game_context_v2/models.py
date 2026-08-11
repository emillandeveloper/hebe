from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class GameRunStatus(StrEnum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"
    HISTORICAL = "HISTORICAL"


@dataclass(frozen=True, slots=True)
class GameIdentity:
    game_id: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    platform_ids: dict[str, str] = field(default_factory=dict)
    series: str = ""
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        value=asdict(self);value["aliases"]=list(self.aliases);return value


@dataclass(frozen=True, slots=True)
class GameRun:
    id: str
    game_id: str
    owner_id: str
    run_kind: str
    rules: dict[str, Any]
    status: GameRunStatus
    started_at: float
    last_active_at: float
    ended_at: float
    current_checkpoint_version: int
    created_from_event_id: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        value=asdict(self);value["status"]=self.status.value;return value


@dataclass(frozen=True, slots=True)
class GameRunResolution:
    game_identity: GameIdentity
    active_run: GameRun | None
    confidence: float
    decision: str
    evidence_ids: tuple[str, ...]
    reason: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {"game_identity":self.game_identity.to_dict(),"active_run":self.active_run.to_dict() if self.active_run else {},"confidence":self.confidence,"decision":self.decision,"evidence_ids":list(self.evidence_ids),"reason":self.reason,"latency_ms":self.latency_ms}


@dataclass(frozen=True, slots=True)
class GameKnowledgeGap:
    id: str
    game_id: str
    run_id: str
    subject_ref: str
    question_type: str
    query_intent: str
    spoiler_ceiling: str
    required_confidence: float
    created_from_event_id: str
    normalized_gap_key: str
    status: str
    created_at: float
    updated_at: float
    resolved_fact_ids: tuple[str, ...] = ()
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        value=asdict(self);value["resolved_fact_ids"]=list(self.resolved_fact_ids);return value


@dataclass(frozen=True, slots=True)
class GameContext:
    game_identity: dict[str, Any]
    scene_assertions: tuple[dict[str, Any], ...]
    active_run: dict[str, Any]
    run_facts: tuple[dict[str, Any], ...]
    run_hypotheses: tuple[dict[str, Any], ...]
    knowledge_claims: tuple[dict[str, Any], ...]
    rejected_knowledge: tuple[dict[str, Any], ...]
    rag_context: tuple[dict[str, Any], ...]
    knowledge_gaps: tuple[dict[str, Any], ...]
    research_status: str
    provenance_manifest: tuple[dict[str, Any], ...]
    advice_allowed: bool
    reaction_allowed: bool
    manifest_size_bytes: int
    latency_ms: float

    def to_dict(self) -> dict[str, Any]: return asdict(self)
