from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class BeliefStatus(str, Enum):
    KNOWN = "KNOWN"
    INFERRED = "INFERRED"
    SUSPECTED = "SUSPECTED"
    HISTORICAL = "HISTORICAL"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"  # Valid proposal that failed admission; never historical truth.


class EvidenceRelation(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    CORRECTS = "CORRECTS"


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    source_event_id: str
    source_record_type: str
    source_record_id: str
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS
    weight: float = 1.0
    observed_at: float = 0.0
    extractor: str = "deterministic"
    extractor_version: str = "v1"
    literal_span: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self); value["relation"] = self.relation.value; return value


@dataclass(frozen=True, slots=True)
class Belief:
    id: str
    namespace: str
    scope_kind: str
    scope_id: str
    subject_ref: str
    predicate: str
    object_value: Any
    epistemic_status: BeliefStatus
    confidence: float
    authority_class: str
    created_at: float
    last_confirmed_at: float
    valid_from: float
    valid_until: float
    relevance_until: float
    superseded_by: str = ""
    owner_confirmed: bool = False
    sensitivity: str = "normal"
    schema_version: int = 1
    retention_policy: str = "retain_history"
    version: int = 1
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["epistemic_status"] = self.epistemic_status.value
        value["object"] = value.pop("object_value")
        value["evidence_ids"] = list(self.evidence_ids)
        return value


@dataclass(frozen=True, slots=True)
class SceneAssertion:
    id: str
    subject_ref: str
    predicate: str
    object_value: Any
    epistemic_status: BeliefStatus
    confidence: float
    evidence_ids: tuple[str, ...]
    referent_data: dict[str, Any]
    observed_at: float
    valid_from: float
    valid_until: float
    provenance: str
    extractor: str
    extractor_version: str
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    context_kind: str
    purpose: str
    subject: str = ""
    topic: str = ""
    allowed_scopes: tuple[str, ...] = ()
    allowed_sensitivity: tuple[str, ...] = ("normal",)
    epistemic_statuses: tuple[BeliefStatus, ...] = ()
    temporal_intent: str = "current"
    max_age: float = 0.0
    max_results: int = 10
    provenance_required: bool = True


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    selected_claims: tuple[dict[str, Any], ...]
    rejected_claims: tuple[dict[str, Any], ...]
    rejection_reasons: dict[str, int]
    manifest_size_bytes: int
    latency_ms: float

    def to_dict(self) -> dict[str, Any]: return asdict(self)
