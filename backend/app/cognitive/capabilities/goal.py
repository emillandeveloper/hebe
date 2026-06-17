from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


GOAL_TYPES = (
    "answer_question",
    "analyze_chat_activity",
    "research_game_strategy",
    "update_session_state",
    "correct_assumption",
    "control_pc",
    "schedule_task",
    "diagnose_problem",
    "analyze_data",
    "unknown",
)


@dataclass(slots=True)
class Goal:
    goal_id: str
    message_id: str
    goal_type: str
    raw_text: str
    normalized_text: str
    source: str = "ui"
    target_audience: str = "leo"
    entities: list[dict[str, Any]] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    confidence: float = 0.0
    urgency: str = "normal"
    risk_level: str = "low"
    requires_confirmation: bool = False
    spoiler_sensitivity: str = "normal"
    memory_relevance: str = "relevant"
    reasoning_summary: str = ""
    related_event_ids: list[str] = field(default_factory=list)
    related_last_hebe_utterance_id: str | None = None
    should_reply_candidate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
