from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import time


INTERVENTION_LEVELS = {
    "observe_only",
    "avatar_reaction_only",
    "local_ui_debug_only",
    "local_owner_reply",
    "twitch_text_reply",
    "stream_tts_reply",
    "twitch_action_only",
    "action_execution",
    "suppress",
}


@dataclass(slots=True)
class PerceivedEvent:
    """Perception: what happened, without deciding whether Hebe should answer."""

    event_id: str
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    source_type: str = ""
    speaker: str = ""
    speaker_type: str = ""
    raw_text: str = ""
    normalized_text: str = ""
    output_context: str = "stream"
    stream_live: bool = False
    current_game: str = ""
    current_activity: str = ""
    wake_detected: bool = False
    direct_address_to_hebe: bool = False
    talks_about_hebe: bool = False
    mentions_hebe: bool = False
    talks_to_leo: bool = False
    talks_to_viewer: bool = False
    is_emote_only: bool = False
    is_low_value_chat: bool = False
    is_owner_monologue: bool = False
    confidence: float = 1.0
    stt_confidence: float | None = None
    twitch_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UnderstandingResult:
    """Understanding: what this likely means, still not an output decision."""

    intent: str
    confidence: float
    authority: str
    reply_pressure: float = 0.0
    requires_policy: bool = True
    possible_capability: str = ""
    possible_pending_kind: str = ""
    social_context: str = ""
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PolicyContract:
    """Decision/Policy contract. The model may render it, not reinterpret it."""

    result: str = "allow"
    reason: str = "ok"
    blocked_behavior: str = ""
    allowed_action: str = ""
    forbidden_actions: list[str] = field(default_factory=list)
    capability_allowed: list[str] = field(default_factory=list)
    capability_blocked: list[str] = field(default_factory=list)
    authority_constraints: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    boundary_required: bool = False
    safe_alternatives: list[str] = field(default_factory=list)
    risk_level: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InterventionDecision:
    """Intervention: whether entering the scene improves it, and where."""

    should_intervene: bool
    intervention_level: str
    reason: str
    social_value_score: float = 0.0
    interruption_cost: float = 0.0
    channel_cost: float = 0.0
    urgency: float = 0.0
    risk: str = "low"
    speech_act_type: str = "no_output"
    output_budget_result: dict[str, Any] = field(default_factory=dict)
    thread_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryCandidate:
    subject_type: str
    subject_id: str
    observation: str
    source: str
    trust_level: str
    confidence: float
    recency: float = field(default_factory=time.time)
    allowed_use: list[str] = field(default_factory=list)
    privacy_level: str = "channel"
    expires_at: float | None = None
    requires_owner_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PresenceEngine:
    """Social intervention model. Cooldowns are inputs, not the whole brain."""

    def decide(
        self,
        *,
        perception: PerceivedEvent,
        understanding: UnderstandingResult,
        policy: PolicyContract | None = None,
        budget_result: dict[str, Any] | None = None,
        thread_result: dict[str, Any] | None = None,
    ) -> InterventionDecision:
        policy = policy or PolicyContract()
        budget = dict(budget_result or {"allowed": True, "reason": "not_checked"})
        thread = dict(thread_result or {})
        social_value = self._social_value(perception, understanding, policy)
        interruption_cost = self._interruption_cost(perception)
        channel_cost = self._channel_cost(perception, understanding)
        urgency = self._urgency(understanding, policy)
        critical = bool(policy.boundary_required or understanding.intent in {"viewer_boundary_needed", "viewer_proxy_request", "viewer_command_attempt"})

        if perception.source == "owner_discourse_opportunity" or understanding.intent == "owner_discourse_opportunity":
            if not bool(budget.get("allowed", True)):
                return self._decision(False, "observe_only", str(budget.get("reason") or "discourse_budget"), social_value, interruption_cost, channel_cost, urgency, policy, budget, thread)
            turn_available = bool(perception.twitch_metadata.get("turn_available"))
            topic_stable = bool(perception.twitch_metadata.get("topic_stable"))
            contribution_value = float(perception.twitch_metadata.get("contribution_value", 0.0) or 0.0)
            novelty = float(perception.twitch_metadata.get("novelty_score", 0.0) or 0.0)
            if not topic_stable or not turn_available or contribution_value < 0.62 or novelty < 0.5:
                reason = "owner_still_speaking" if not turn_available else "discourse_not_ready"
                return self._decision(False, "observe_only", reason, contribution_value, interruption_cost, 0.05, urgency, policy, budget, thread)
            return InterventionDecision(
                should_intervene=True, intervention_level="stream_tts_reply", reason="validated_discourse_opportunity",
                social_value_score=round(contribution_value, 3), interruption_cost=round(interruption_cost, 3),
                channel_cost=0.05, urgency=0.3, risk=policy.risk_level,
                speech_act_type="stream_discourse_contribution", output_budget_result=budget, thread_result=thread,
            )

        if policy.result in {"block", "redirect"} and not policy.boundary_required:
            return self._decision(False, "suppress", policy.reason or "policy_block", social_value, interruption_cost, channel_cost, urgency, policy, budget, thread)
        if understanding.intent in {"viewer_emote_only", "viewer_low_value_banter"}:
            return self._decision(False, "observe_only", understanding.social_context or understanding.intent, social_value, interruption_cost, channel_cost, urgency, policy, budget, thread)
        if not bool(budget.get("allowed", True)) and not critical:
            return self._decision(False, "observe_only", str(budget.get("reason") or "social_budget"), social_value, interruption_cost, channel_cost, urgency, policy, budget, thread)
        if social_value - interruption_cost - channel_cost < 0.35 and not critical:
            return self._decision(False, "observe_only", "low_social_value", social_value, interruption_cost, channel_cost, urgency, policy, budget, thread)

        level = "local_owner_reply" if perception.speaker_type == "owner" else "twitch_text_reply"
        speech_act = self._speech_act_type(understanding, policy)
        return InterventionDecision(
            should_intervene=True,
            intervention_level=level,
            reason="presence_value",
            social_value_score=round(social_value, 3),
            interruption_cost=round(interruption_cost, 3),
            channel_cost=round(channel_cost, 3),
            urgency=round(urgency, 3),
            risk=policy.risk_level,
            speech_act_type=speech_act,
            output_budget_result=budget,
            thread_result=thread,
        )

    def _decision(
        self,
        should: bool,
        level: str,
        reason: str,
        social_value: float,
        interruption_cost: float,
        channel_cost: float,
        urgency: float,
        policy: PolicyContract,
        budget: dict[str, Any],
        thread: dict[str, Any],
    ) -> InterventionDecision:
        return InterventionDecision(
            should_intervene=should,
            intervention_level=level if level in INTERVENTION_LEVELS else "observe_only",
            reason=reason,
            social_value_score=round(social_value, 3),
            interruption_cost=round(interruption_cost, 3),
            channel_cost=round(channel_cost, 3),
            urgency=round(urgency, 3),
            risk=policy.risk_level,
            speech_act_type="no_output" if not should else self._speech_act_type(UnderstandingResult(reason, 0.0, "viewer"), policy),
            output_budget_result=budget,
            thread_result=thread,
        )

    def _social_value(self, perception: PerceivedEvent, understanding: UnderstandingResult, policy: PolicyContract) -> float:
        value = float(understanding.reply_pressure or 0.0)
        if perception.direct_address_to_hebe:
            value += 0.28
        elif perception.talks_about_hebe:
            value += 0.18
        elif perception.mentions_hebe:
            value += 0.08
        if policy.boundary_required:
            value += 0.45
        if understanding.intent in {"viewer_direct_question_to_hebe", "game_guidance_request", "promotion_request", "high_value_game_tip"}:
            value += 0.25
        if perception.is_low_value_chat:
            value -= 0.35
        if perception.is_emote_only:
            value -= 0.55
        return max(0.0, min(1.0, value))

    def _interruption_cost(self, perception: PerceivedEvent) -> float:
        cost = 0.0
        if perception.is_owner_monologue:
            cost += 0.55
        if perception.current_activity in {"combat", "boss", "cutscene"}:
            cost += 0.18
        return min(1.0, cost)

    def _channel_cost(self, perception: PerceivedEvent, understanding: UnderstandingResult) -> float:
        if perception.source_type != "twitch_chat":
            return 0.05
        if understanding.intent in {"viewer_emote_only", "viewer_low_value_banter"}:
            return 0.42
        return 0.18

    def _urgency(self, understanding: UnderstandingResult, policy: PolicyContract) -> float:
        if policy.boundary_required:
            return 0.95
        if understanding.intent.endswith("_attempt"):
            return 0.7
        return min(1.0, max(0.0, understanding.reply_pressure))

    def _speech_act_type(self, understanding: UnderstandingResult, policy: PolicyContract) -> str:
        if policy.boundary_required:
            return "viewer_boundary"
        if understanding.intent in {"viewer_talks_about_hebe", "viewer_banter_about_hebe"}:
            return "self_banter_reply"
        if understanding.intent == "viewer_direct_question_to_hebe":
            return "stream_banter"
        if understanding.intent == "high_value_game_tip":
            return "game_guidance_clarification"
        if understanding.intent == "owner_direct_question":
            return "direct_answer"
        return "stream_banter"


class HebeCoreLoop:
    """Official loop contract: Perception -> Understanding -> Decision -> Intervention.

    The full engine still owns execution/rendering/emission while the migration is
    incremental. This object formalizes the first controlled stages so new output
    paths can be routed through the same contract instead of raw text shortcuts.
    """

    def __init__(self, *, presence_engine: PresenceEngine | None = None):
        self.presence_engine = presence_engine or PresenceEngine()

    def process(
        self,
        *,
        perception: PerceivedEvent,
        understanding: UnderstandingResult,
        policy: PolicyContract | None = None,
        budget_result: dict[str, Any] | None = None,
        thread_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        intervention = self.presence_engine.decide(
            perception=perception,
            understanding=understanding,
            policy=policy,
            budget_result=budget_result,
            thread_result=thread_result,
        )
        return {
            "perception": perception.to_dict(),
            "understanding": understanding.to_dict(),
            "policy": (policy or PolicyContract()).to_dict(),
            "intervention": intervention.to_dict(),
        }
