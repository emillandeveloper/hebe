from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class InputEnvelope:
    """Shared source/authority evidence produced once before firewall and routing."""

    raw_text: str
    normalized_text: str
    source: str
    authority: str
    trust: str
    addressed_to_hebe: bool
    matched_wake_name: str | None = None
    wake_evidence: dict[str, Any] = field(default_factory=dict)
    command_mode: bool = False
    intent_candidates: list[str] = field(default_factory=list)
    app_target: str | None = None
    app_plan_result: dict[str, Any] = field(default_factory=dict)
    active_pending: dict[str, Any] | None = None
    pending_compatible: bool = False
    expected_reply_type: str = ""
    is_followup_candidate: bool = False
    input_type: str = "ambient_stream_context"
    reason: str = "unclassified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "source": self.source,
            "authority": self.authority,
            "trust": self.trust,
            "addressed_to_hebe": self.addressed_to_hebe,
            "matched_wake_name": self.matched_wake_name,
            "wake_evidence": dict(self.wake_evidence),
            "command_mode": self.command_mode,
            "intent_candidates": list(self.intent_candidates),
            "app_target": self.app_target,
            "app_plan_result": dict(self.app_plan_result),
            "active_pending": dict(self.active_pending) if self.active_pending else None,
            "pending_compatible": self.pending_compatible,
            "expected_reply_type": self.expected_reply_type,
            "is_followup_candidate": self.is_followup_candidate,
            "input_type": self.input_type,
            "reason": self.reason,
        }


@dataclass
class InputEvent:
    source: str
    raw_text: str
    normalized_text: str
    user_id: str | None = None
    username: str | None = None
    is_voice: bool = False
    is_stream_context: bool = False
    timestamp: float = field(default_factory=time.time)
    stt_metadata: dict[str, Any] = field(default_factory=dict)
    envelope: InputEnvelope | None = None

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "username": self.username,
            "is_voice": self.is_voice,
            "is_stream_context": self.is_stream_context,
            "timestamp": self.timestamp,
            "stt_metadata": self.stt_metadata,
            "envelope": self.envelope.as_dict() if self.envelope else None,
        }
