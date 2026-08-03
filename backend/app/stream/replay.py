from __future__ import annotations

import hashlib
import json
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Protocol

from app.stream.live_runtime import LiveSessionStateManager


class ReplayMode(StrEnum):
    REAL_TIME = "real_time"
    ACCELERATED = "accelerated"
    STEP_BY_STEP = "step_by_step"
    SHADOW = "shadow"
    COMPARE_VERSIONS = "compare_versions"


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    event_id: str
    event_type: str
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)
    sequence: int = 0
    previous_hebe_outputs: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: dict[str, Any], *, sequence: int = 0) -> "ReplayEvent":
        event_type = str(value.get("event_type") or value.get("type") or "unknown")
        payload = dict(value.get("payload") or {})
        timestamp = _timestamp(value.get("timestamp") or value.get("created_at") or payload.get("timestamp"))
        identity = str(value.get("event_id") or value.get("id") or payload.get("event_id") or "").strip()
        if not identity:
            stable = json.dumps(
                {"event_type": event_type, "timestamp": timestamp, "payload": payload, "sequence": sequence},
                ensure_ascii=False,
                sort_keys=True,
            )
            identity = f"replay_{hashlib.sha256(stable.encode('utf-8')).hexdigest()[:16]}"
        previous = tuple(str(item) for item in value.get("previous_hebe_outputs") or ())
        return cls(identity, event_type, timestamp, payload, sequence, previous)


@dataclass(slots=True)
class ReplayDecision:
    event_id: str = ""
    proposed_output: str = ""
    output_targets: list[str] = field(default_factory=list)
    should_emit: bool = False
    presence_allowed: bool = False
    final_guard_allowed: bool = False
    suppress_reason: str = ""
    action_type: str = ""
    action_status: str = ""
    research_calls: list[dict[str, Any]] = field(default_factory=list)
    promotion_decisions: list[dict[str, Any]] = field(default_factory=list)
    guard_decisions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: "ReplayDecision | dict[str, Any] | None", event_id: str) -> "ReplayDecision":
        if isinstance(value, cls):
            if not value.event_id:
                value.event_id = event_id
            return value
        data = dict(value or {})
        allowed = bool(data.get("should_emit", False))
        return cls(
            event_id=str(data.get("event_id") or event_id),
            proposed_output=str(data.get("proposed_output") or data.get("text") or ""),
            output_targets=list(data.get("output_targets") or data.get("targets") or []),
            should_emit=allowed,
            presence_allowed=bool(data.get("presence_allowed", allowed)),
            final_guard_allowed=bool(data.get("final_guard_allowed", allowed)),
            suppress_reason=str(data.get("suppress_reason") or data.get("reason") or ""),
            action_type=str(data.get("action_type") or ""),
            action_status=str(data.get("action_status") or ""),
            research_calls=list(data.get("research_calls") or []),
            promotion_decisions=list(data.get("promotion_decisions") or []),
            guard_decisions=list(data.get("guard_decisions") or []),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class ReplayResult:
    replay_id: str
    mode: str
    version: str
    decisions: list[ReplayDecision] = field(default_factory=list)
    simulated_final_emissions: list[dict[str, Any]] = field(default_factory=list)
    suppressed_messages: list[dict[str, Any]] = field(default_factory=list)
    action_outcomes: list[dict[str, Any]] = field(default_factory=list)
    state_snapshots: list[dict[str, Any]] = field(default_factory=list)
    research_calls: list[dict[str, Any]] = field(default_factory=list)
    promotion_decisions: list[dict[str, Any]] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    blocked_real_writes: list[dict[str, Any]] = field(default_factory=list)

    def deterministic_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "version": self.version,
            "decisions": [asdict(item) for item in self.decisions],
            "simulated_final_emissions": self.simulated_final_emissions,
            "suppressed_messages": self.suppressed_messages,
            "action_outcomes": self.action_outcomes,
            "state_snapshots": self.state_snapshots,
            "research_calls": self.research_calls,
            "promotion_decisions": self.promotion_decisions,
            "blocked_real_writes": self.blocked_real_writes,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.deterministic_payload(), ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplayVersionChange:
    event_id: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None


@dataclass(slots=True)
class ReplayComparison:
    baseline_version: str
    candidate_version: str
    changed_outputs: list[ReplayVersionChange] = field(default_factory=list)
    baseline_fingerprint: str = ""
    candidate_fingerprint: str = ""


class ReplayProcessor(Protocol):
    def __call__(self, event: ReplayEvent, runtime: "ReplayRuntime") -> ReplayDecision | dict[str, Any] | None:
        ...


class ReplayIOBoundary:
    """Records attempted side effects and never touches production I/O."""

    def __init__(self, *, allow_recorded_tts: bool = False) -> None:
        self.allow_recorded_tts = allow_recorded_tts
        self.blocked_real_writes: list[dict[str, Any]] = []
        self.recorded_tts: list[str] = []

    def send_twitch(self, text: str) -> bool:
        self.blocked_real_writes.append({"kind": "twitch", "text": str(text or ""), "blocked": True})
        return False

    def execute_desktop(self, action: str, payload: dict[str, Any] | None = None) -> bool:
        self.blocked_real_writes.append(
            {"kind": "desktop", "action": str(action or ""), "payload": dict(payload or {}), "blocked": True}
        )
        return False

    def speak(self, text: str) -> bool:
        if self.allow_recorded_tts:
            self.recorded_tts.append(str(text or ""))
        else:
            self.blocked_real_writes.append({"kind": "tts", "text": str(text or ""), "blocked": True})
        return self.allow_recorded_tts


class ReplayFixtureResearchProvider:
    """Deterministic research source. Missing fixtures always fail closed."""

    def __init__(self, fixtures: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.fixtures = {str(key): [dict(row) for row in rows] for key, rows in (fixtures or {}).items()}
        self.calls: list[str] = []

    def search(self, query: str, *, cache_key: str = "") -> list[dict[str, Any]]:
        key = str(cache_key or query).strip()
        self.calls.append(key)
        if key not in self.fixtures:
            raise LookupError(f"research_fixture_missing:{key}")
        return [dict(row) for row in self.fixtures[key]]


@dataclass(slots=True)
class ReplayRuntime:
    replay_id: str
    mode: str
    state: Any
    io: ReplayIOBoundary
    research_provider: ReplayFixtureResearchProvider
    simulated_time: float = 0.0
    previous_hebe_outputs: tuple[str, ...] = ()


class ReplaySession:
    def __init__(self, lab: "StreamReplayLab", events: list[ReplayEvent], processor: ReplayProcessor, runtime: ReplayRuntime, result: ReplayResult):
        self.lab = lab
        self.events = events
        self.processor = processor
        self.runtime = runtime
        self.result = result
        self.index = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.events)

    def step(self) -> ReplayDecision | None:
        if self.done:
            return None
        event = self.events[self.index]
        self.index += 1
        return self.lab._process_event(event, self.processor, self.runtime, self.result, shadow=False)


class StreamReplayLab:
    def __init__(
        self,
        *,
        state_factory: Callable[[], Any] | None = None,
        monotonic: Callable[[], float] = time.perf_counter,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.state_factory = state_factory or (lambda: SimpleNamespace())
        self.monotonic = monotonic
        self.sleep_fn = sleep_fn
        self.session_state_manager = LiveSessionStateManager()

    def run(
        self,
        events: Iterable[ReplayEvent | dict[str, Any]],
        processor: ReplayProcessor,
        *,
        mode: ReplayMode | str = ReplayMode.ACCELERATED,
        version: str = "current",
        research_fixtures: dict[str, list[dict[str, Any]]] | None = None,
        allow_recorded_tts: bool = False,
    ) -> ReplayResult | ReplaySession:
        resolved_mode = ReplayMode(str(mode))
        ordered = self._ordered_events(events)
        replay_id = self._replay_id(ordered, version)
        state = self.state_factory()
        self.session_state_manager.reset_for_replay(state, replay_id)
        io = ReplayIOBoundary(allow_recorded_tts=allow_recorded_tts)
        runtime = ReplayRuntime(
            replay_id=replay_id,
            mode=resolved_mode.value,
            state=state,
            io=io,
            research_provider=ReplayFixtureResearchProvider(research_fixtures),
        )
        result = ReplayResult(replay_id, resolved_mode.value, version)
        if resolved_mode is ReplayMode.STEP_BY_STEP:
            return ReplaySession(self, ordered, processor, runtime, result)
        previous_ts: float | None = None
        for event in ordered:
            if resolved_mode is ReplayMode.REAL_TIME and previous_ts is not None:
                self.sleep_fn(max(0.0, event.timestamp - previous_ts))
            previous_ts = event.timestamp
            self._process_event(
                event,
                processor,
                runtime,
                result,
                shadow=resolved_mode is ReplayMode.SHADOW,
            )
        result.blocked_real_writes = list(io.blocked_real_writes)
        return result

    def compare_versions(
        self,
        events: Iterable[ReplayEvent | dict[str, Any]],
        baseline_processor: ReplayProcessor,
        candidate_processor: ReplayProcessor,
        *,
        baseline_version: str = "baseline",
        candidate_version: str = "candidate",
        research_fixtures: dict[str, list[dict[str, Any]]] | None = None,
    ) -> ReplayComparison:
        materialized = list(events)
        before = self.run(materialized, baseline_processor, version=baseline_version, research_fixtures=research_fixtures)
        after = self.run(materialized, candidate_processor, version=candidate_version, research_fixtures=research_fixtures)
        assert isinstance(before, ReplayResult) and isinstance(after, ReplayResult)
        left = {item.event_id: asdict(item) for item in before.decisions}
        right = {item.event_id: asdict(item) for item in after.decisions}
        changed = [
            ReplayVersionChange(key, left.get(key), right.get(key))
            for key in sorted(set(left) | set(right))
            if left.get(key) != right.get(key)
        ]
        return ReplayComparison(
            baseline_version,
            candidate_version,
            changed,
            before.fingerprint,
            after.fingerprint,
        )

    def load_jsonl(self, path: str | Path) -> list[ReplayEvent]:
        rows = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    rows.append(ReplayEvent.from_value(json.loads(line), sequence=index))
        return self._ordered_events(rows)

    def save_golden(self, path: str | Path, result: ReplayResult) -> None:
        payload = {**result.deterministic_payload(), "fingerprint": result.fingerprint}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _process_event(
        self,
        event: ReplayEvent,
        processor: ReplayProcessor,
        runtime: ReplayRuntime,
        result: ReplayResult,
        *,
        shadow: bool,
    ) -> ReplayDecision:
        runtime.simulated_time = event.timestamp
        runtime.previous_hebe_outputs = event.previous_hebe_outputs
        started = self.monotonic()
        decision = ReplayDecision.from_value(processor(event, runtime), event.event_id)
        result.latencies_ms.append(round(max(0.0, self.monotonic() - started) * 1000, 3))
        result.decisions.append(decision)
        result.research_calls.extend(decision.research_calls)
        result.promotion_decisions.extend(decision.promotion_decisions)
        if decision.action_type:
            result.action_outcomes.append(
                {"event_id": event.event_id, "action_type": decision.action_type, "status": decision.action_status}
            )
        can_emit = (
            decision.should_emit
            and decision.presence_allowed
            and decision.final_guard_allowed
            and bool(decision.proposed_output.strip())
            and not shadow
        )
        if can_emit:
            result.simulated_final_emissions.append(
                {
                    "event_id": event.event_id,
                    "text": decision.proposed_output,
                    "targets": list(decision.output_targets),
                }
            )
        elif decision.proposed_output:
            result.suppressed_messages.append(
                {
                    "event_id": event.event_id,
                    "text": decision.proposed_output,
                    "reason": "shadow_mode" if shadow else decision.suppress_reason or "presence_or_final_gate",
                }
            )
        result.state_snapshots.append({"event_id": event.event_id, "state": _snapshot(runtime.state)})
        result.blocked_real_writes = list(runtime.io.blocked_real_writes)
        return decision

    @staticmethod
    def _ordered_events(events: Iterable[ReplayEvent | dict[str, Any]]) -> list[ReplayEvent]:
        materialized = [
            deepcopy(value) if isinstance(value, ReplayEvent) else ReplayEvent.from_value(value, sequence=index)
            for index, value in enumerate(events)
        ]
        return sorted(materialized, key=lambda item: (item.timestamp, item.sequence))

    @staticmethod
    def _replay_id(events: list[ReplayEvent], version: str) -> str:
        stable = json.dumps(
            [(event.event_id, event.event_type, event.timestamp, event.sequence) for event in events],
            ensure_ascii=False,
            sort_keys=True,
        )
        return f"replay_{hashlib.sha256((version + stable).encode('utf-8')).hexdigest()[:16]}"


def _timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        pass
    try:
        from datetime import datetime

        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _snapshot(state: Any) -> dict[str, Any]:
    if hasattr(state, "as_dict") and callable(state.as_dict):
        return dict(state.as_dict())
    if hasattr(state, "__dict__"):
        data = vars(state)
    else:
        return {"value": str(state)}
    return json.loads(json.dumps(data, ensure_ascii=False, sort_keys=True, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


__all__ = [
    "ReplayComparison",
    "ReplayDecision",
    "ReplayEvent",
    "ReplayFixtureResearchProvider",
    "ReplayIOBoundary",
    "ReplayMode",
    "ReplayResult",
    "ReplayRuntime",
    "ReplaySession",
    "ReplayVersionChange",
    "StreamReplayLab",
]
