from __future__ import annotations

import re
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


def _norm(value: str) -> str:
    raw = "".join(
        char for char in unicodedata.normalize("NFKD", str(value or "").casefold())
        if not unicodedata.combining(char)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", raw).split())


@dataclass(slots=True)
class SceneTimelineState:
    scene_id: str
    topic_id: str
    entity: str
    current_state: str
    state_version: int
    supporting_event_ids: list[str] = field(default_factory=list)
    superseded_event_ids: list[str] = field(default_factory=list)
    terminal: bool = False
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SceneRevalidation:
    valid: bool
    reason: str
    current_scene_id: str
    current_state_version: int


class SceneTimelineManager:
    """Versioned current-scene truth used by context selection and final emission."""

    _AREA_CHANGE = re.compile(
        r"\b(?:nueva zona|otra zona|cambio de zona|hemos llegado|llegamos a|"
        r"new area|changed area|entered|arrived at|left the area)\b"
    )
    _ENEMY_DEAD = re.compile(
        r"\b(?:esta muerto|ha muerto|murio|lo mate|lo matamos|ha caido|enemigo derrotado|"
        r"enemy (?:is )?dead|enemy died|killed it|defeated|it is down)\b"
    )
    _BATTLE_ENDED = re.compile(
        r"\b(?:combate terminado|batalla terminada|fin del combate|victoria|"
        r"battle (?:is )?over|combat ended|victory)\b"
    )
    _ITEM_OBTAINED = re.compile(
        r"\b(?:he conseguido|consegui|hemos conseguido|obtuve|objeto obtenido|"
        r"got the item|item obtained|picked up)\b"
    )
    _PUZZLE_COMPLETED = re.compile(
        r"\b(?:puzle resuelto|puzzle resuelto|ya esta resuelto|lo resolvi|"
        r"puzzle solved|solved it)\b"
    )
    _CORRECTION = re.compile(
        r"\b(?:me corrijo|no era|quiero decir|correccion|actually|i mean|correction)\b"
    )
    _LOW_HP = re.compile(
        r"\b(?:le queda poca vida|esta a poca vida|casi muerto|"
        r"low hp|low health|almost dead|nearly dead)\b"
    )

    def __init__(self, *, now_fn: Callable[[], float] = time.time) -> None:
        self.now_fn = now_fn
        self.current: SceneTimelineState | None = None
        self._states: dict[str, SceneTimelineState] = {}
        self._event_scene: dict[str, str] = {}

    def reset(self) -> None:
        self.current = None
        self._states.clear()
        self._event_scene.clear()

    def snapshot(self) -> dict[str, Any]:
        return self.current.to_dict() if self.current is not None else {}

    def observe(
        self,
        text: str,
        *,
        event_id: str = "",
        topic_id: str = "",
        entity: str = "",
        facts: list[dict[str, Any]] | None = None,
        now: float | None = None,
    ) -> SceneTimelineState:
        now = self.now_fn() if now is None else float(now)
        raw = str(text or "")
        normalized = _norm(raw)
        event_id = str(event_id or f"scene_event:{uuid.uuid4().hex}")
        topic_id = str(topic_id or "")
        detected_entity = str(entity or self._entity_from_facts(facts) or "current_scene")
        source_facts = facts or []
        transition = self._classify(normalized, source_facts)
        incoming_event_ids = list(dict.fromkeys([
            event_id,
            *(
                str(fact.get("id") or fact.get("fact_id") or "")
                for fact in source_facts
                if fact.get("id") or fact.get("fact_id")
            ),
        ]))

        if self.current is None or transition == "area_changed":
            old = self.current
            if old is not None:
                old.terminal = True
                old.current_state = "area_changed"
                old.state_version += 1
                old.updated_at = now
                old.superseded_event_ids = list(dict.fromkeys([
                    *old.superseded_event_ids, *old.supporting_event_ids,
                ]))
            self.current = SceneTimelineState(
                scene_id=f"scene_{uuid.uuid4().hex}",
                topic_id=topic_id,
                entity=detected_entity,
                current_state="area_changed" if old is not None else transition or "active",
                state_version=1,
                supporting_event_ids=incoming_event_ids,
                terminal=False,
                updated_at=now,
            )
            self._states[self.current.scene_id] = self.current
        else:
            state = self.current
            assert state is not None
            if topic_id and state.topic_id and topic_id != state.topic_id:
                state.superseded_event_ids = list(dict.fromkeys([
                    *state.superseded_event_ids, *state.supporting_event_ids,
                ]))
                state.supporting_event_ids = []
                state.state_version += 1
            state.topic_id = topic_id or state.topic_id
            state.entity = detected_entity or state.entity
            if transition:
                if transition in {"enemy_dead", "battle_ended", "item_obtained", "puzzle_completed", "correction"}:
                    state.superseded_event_ids = list(dict.fromkeys([
                        *state.superseded_event_ids, *state.supporting_event_ids,
                    ]))
                    state.supporting_event_ids = []
                state.current_state = transition
                state.state_version += 1
                state.terminal = transition in {"enemy_dead", "battle_ended", "puzzle_completed"}
            state.supporting_event_ids = list(dict.fromkeys([
                *state.supporting_event_ids,
                *incoming_event_ids,
            ]))
            state.updated_at = now
        for incoming_event_id in incoming_event_ids:
            self._event_scene[incoming_event_id] = self.current.scene_id
        return self.current

    def annotate_facts(
        self,
        facts: list[dict[str, Any]],
        *,
        topic_id: str = "",
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        now = self.now_fn() if now is None else float(now)
        state = self.current
        result: list[dict[str, Any]] = []
        for source in facts:
            fact = dict(source)
            timestamp = float(fact.get("timestamp", now) or now)
            event_id = str(fact.get("id") or fact.get("fact_id") or "")
            fact["scene_id"] = state.scene_id if state else ""
            fact["topic_id"] = str(fact.get("topic_id") or topic_id or (state.topic_id if state else ""))
            fact["age_seconds"] = max(0.0, now - timestamp)
            fact["superseded"] = bool(
                state and event_id and event_id in set(state.superseded_event_ids)
            )
            fact["state_version"] = state.state_version if state else 0
            fact["current_state"] = state.current_state if state else "active"
            fact["terminal"] = bool(state and state.terminal)
            fact["currentness_score"] = self._currentness(fact, now=now)
            result.append(fact)
        return result

    def filter_current_facts(
        self,
        facts: list[dict[str, Any]],
        *,
        topic_id: str = "",
        now: float | None = None,
        anchor_relevant: bool = True,
    ) -> list[dict[str, Any]]:
        now = self.now_fn() if now is None else float(now)
        current = self.current
        selected: list[dict[str, Any]] = []
        for source in facts:
            fact = dict(source)
            fact["age_seconds"] = max(0.0, now - float(fact.get("timestamp", now) or now))
            fact["currentness_score"] = self._currentness(fact, now=now)
            if float(fact.get("expires_at", now + 1) or 0.0) <= now:
                continue
            if bool(fact.get("superseded")):
                continue
            if current and str(fact.get("scene_id") or "") != current.scene_id:
                continue
            fact_topic = str(fact.get("topic_id") or "")
            active_topic = str(topic_id or (current.topic_id if current else ""))
            if active_topic and active_topic != fact_topic:
                continue
            if anchor_relevant and not bool(fact.get("proactive_eligible", True)):
                continue
            selected.append(fact)
        return selected

    def revalidate(self, snapshot: dict[str, Any] | None) -> SceneRevalidation:
        expected = dict(snapshot or {})
        current = self.current
        if not expected:
            return SceneRevalidation(True, "no_scene_guard", current.scene_id if current else "", current.state_version if current else 0)
        if current is None:
            return SceneRevalidation(False, "scene_missing", "", 0)
        if str(expected.get("scene_id") or "") != current.scene_id:
            return SceneRevalidation(False, "scene_changed", current.scene_id, current.state_version)
        if int(expected.get("state_version", 0) or 0) != current.state_version:
            return SceneRevalidation(False, "scene_version_changed", current.scene_id, current.state_version)
        if current.terminal and not bool(expected.get("terminal")):
            return SceneRevalidation(False, "scene_became_terminal", current.scene_id, current.state_version)
        return SceneRevalidation(True, "scene_current", current.scene_id, current.state_version)

    def _classify(self, text: str, facts: list[dict[str, Any]]) -> str:
        fact_text = " ".join(
            _norm(str(fact.get("summary") or fact.get("text") or fact.get("raw_evidence") or ""))
            for fact in facts
        )
        combined = f"{text} {fact_text}".strip()
        for pattern, state in (
            (self._AREA_CHANGE, "area_changed"),
            (self._ENEMY_DEAD, "enemy_dead"),
            (self._BATTLE_ENDED, "battle_ended"),
            (self._PUZZLE_COMPLETED, "puzzle_completed"),
            (self._ITEM_OBTAINED, "item_obtained"),
            (self._CORRECTION, "correction"),
            (self._LOW_HP, "enemy_low_hp"),
        ):
            if pattern.search(combined):
                return state
        return ""

    @staticmethod
    def _entity_from_facts(facts: list[dict[str, Any]] | None) -> str:
        for fact in facts or []:
            value = fact.get("extracted_subject") or fact.get("entity")
            if value:
                return str(value)
        return ""

    @staticmethod
    def _currentness(fact: dict[str, Any], *, now: float) -> float:
        timestamp = float(fact.get("timestamp", now) or now)
        ttl = max(1.0, float(fact.get("ttl_sec", 60.0) or 60.0))
        age = max(0.0, now - timestamp)
        if bool(fact.get("superseded")):
            return 0.0
        return max(0.0, min(1.0, 1.0 - age / ttl))


@dataclass(slots=True)
class SpontaneousOpportunityState:
    opportunity_id: str
    anchor_id: str
    status: str = "pending"
    blocked_reason: str = ""
    blocked_guard: str = ""
    retry_count: int = 0
    expires_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpontaneousOpportunityManager:
    TERMINAL = {"emitted", "consumed", "invalidated", "blocked", "reaction_only"}

    def __init__(self, *, now_fn: Callable[[], float] = time.time) -> None:
        self.now_fn = now_fn
        self._by_id: dict[str, SpontaneousOpportunityState] = {}
        self._by_anchor: dict[str, str] = {}

    def reset(self) -> None:
        self._by_id.clear()
        self._by_anchor.clear()

    def open(self, anchor_id: str, *, expires_at: float = 0.0) -> SpontaneousOpportunityState | None:
        anchor = str(anchor_id or "")
        if not anchor or not self.eligible(anchor):
            return None
        state = SpontaneousOpportunityState(
            opportunity_id=f"opportunity_{uuid.uuid4().hex}",
            anchor_id=anchor,
            expires_at=float(expires_at or self.now_fn() + 60.0),
        )
        self._by_id[state.opportunity_id] = state
        self._by_anchor[anchor] = state.opportunity_id
        return state

    def eligible(self, anchor_id: str) -> bool:
        state = self.for_anchor(anchor_id)
        if state is None:
            return True
        if state.expires_at and state.expires_at <= self.now_fn():
            state.status = "invalidated"
        # A pending opportunity already owns this anchor for the current tick;
        # later ticks must not create another probe for it.
        return False

    def for_anchor(self, anchor_id: str) -> SpontaneousOpportunityState | None:
        opportunity_id = self._by_anchor.get(str(anchor_id or ""))
        return self._by_id.get(opportunity_id or "")

    def get(self, opportunity_id: str) -> SpontaneousOpportunityState | None:
        return self._by_id.get(str(opportunity_id or ""))

    def mark(
        self,
        opportunity_id: str,
        status: str,
        *,
        reason: str = "",
        guard: str = "",
    ) -> SpontaneousOpportunityState | None:
        state = self.get(opportunity_id)
        if state is None:
            return None
        if state.status in self.TERMINAL:
            return state
        state.status = str(status)
        state.blocked_reason = str(reason or "")
        state.blocked_guard = str(guard or "")
        if status in {"blocked", "reaction_only", "consumed"}:
            state.retry_count = min(1, state.retry_count + 1)
        return state

    def safe_rewrite_once(self, opportunity_id: str) -> bool:
        state = self.get(opportunity_id)
        if state is None or state.status in self.TERMINAL or state.retry_count >= 1:
            return False
        state.retry_count += 1
        return True

    def all_states(self) -> list[dict[str, Any]]:
        return [state.to_dict() for state in self._by_id.values()]


__all__ = [
    "SceneRevalidation",
    "SceneTimelineManager",
    "SceneTimelineState",
    "SpontaneousOpportunityManager",
    "SpontaneousOpportunityState",
]
