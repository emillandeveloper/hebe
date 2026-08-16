from __future__ import annotations

import threading
import time
import uuid
from collections import Counter, deque
from enum import StrEnum
from typing import Any, Callable

from app.core.persistent_logs import log_jsonl_event


class CalibrationLabel(StrEnum):
    CORRECT = "CORRECT"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    UNCERTAIN = "UNCERTAIN"


_DISALLOWED_TEXT_KEYS = {
    "candidate_text",
    "feedback_text",
    "full_text",
    "raw_stt",
    "raw_text",
    "referent_text",
    "source_text",
    "transcript",
}


def _privacy_filter(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _privacy_filter(item)
            for key, item in value.items()
            if str(key).casefold() not in _DISALLOWED_TEXT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_privacy_filter(item) for item in value]
    return value


class BehaviorObservability:
    """Bounded shadow telemetry. It never feeds decisions back into policy."""

    def __init__(
        self,
        *,
        max_recent: int = 1000,
        max_labels: int = 500,
        clock: Callable[[], float] | None = None,
        log_fn: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.max_recent = max(1, int(max_recent))
        self.max_labels = max(1, int(max_labels))
        self._clock = clock or time.time
        self._log_fn = log_fn or log_jsonl_event
        self._recent: deque[dict[str, Any]] = deque(maxlen=self.max_recent)
        self._labels: deque[dict[str, Any]] = deque(maxlen=self.max_labels)
        self._metrics: Counter[str] = Counter()
        self._lock = threading.RLock()

    def record(self, event: str, *, trace_id: str = "", timestamp: float | None = None, **payload: Any) -> dict[str, Any]:
        value = _privacy_filter({
            "event": str(event),
            "trace_id": str(trace_id or f"behavior_trace_{uuid.uuid4().hex}"),
            "timestamp": float(self._clock() if timestamp is None else timestamp),
            **payload,
        })
        with self._lock:
            self._recent.append(value)
            self._count(value)
        try:
            self._log_fn("behavior_calibration", value)
        except Exception as exc:
            with self._lock:
                self._metrics["telemetry_write_failed"] += 1
                self._metrics[f"telemetry_write_error:{type(exc).__name__}"] += 1
        return value

    def label(self, trace_id: str, label: CalibrationLabel | str) -> dict[str, Any]:
        normalized = CalibrationLabel(str(label).upper())
        value = {
            "label_id": f"behavior_label_{uuid.uuid4().hex}",
            "trace_id": str(trace_id or "").strip(),
            "label": normalized.value,
            "timestamp": float(self._clock()),
        }
        if not value["trace_id"]:
            raise ValueError("trace_id_required")
        with self._lock:
            self._labels.append(value)
            self._metrics["manual_labels"] += 1
            self._metrics[f"manual_label:{normalized.value}"] += 1
        try:
            self._log_fn("behavior_calibration_labels", value)
        except Exception as exc:
            with self._lock:
                self._metrics["telemetry_write_failed"] += 1
                self._metrics[f"telemetry_write_error:{type(exc).__name__}"] += 1
        return value

    def snapshot(self, *, recent_limit: int = 100, label_limit: int = 100) -> dict[str, Any]:
        with self._lock:
            return {
                "metrics": dict(sorted(self._metrics.items())),
                "recent_events": list(self._recent)[-max(1, int(recent_limit)):],
                "recent_labels": list(self._labels)[-max(1, int(label_limit)):],
                "retention": {
                    "recent_events": len(self._recent),
                    "recent_events_limit": self.max_recent,
                    "labels": len(self._labels),
                    "labels_limit": self.max_labels,
                },
            }

    def _count(self, value: dict[str, Any]) -> None:
        event = str(value.get("event") or "")
        self._metrics[f"event:{event}"] += 1
        if event == "candidate_policy":
            self._metrics["candidates_evaluated"] += 1
            decision = str(value.get("policy_decision") or "").upper()
            if decision:
                self._metrics[decision] += 1
            reason = str(value.get("reason_code") or "")
            if decision == "SUPPRESS":
                if reason in {"explicit_behavior_constraint", "generated_output_matches_constraint"}:
                    self._metrics["suppressions_explicit_constraint"] += 1
                elif reason in {"negative_owner_feedback", "negative_feedback_and_recent_repetition"}:
                    self._metrics["suppressions_owner_feedback"] += 1
                if "repetition" in reason or "fatigue" in reason:
                    self._metrics["suppressions_repetition_fatigue"] += 1
        elif event == "feedback":
            if not bool(value.get("referent_resolved")):
                self._metrics["unresolved_feedback_referents"] += 1
        elif event == "post_generation":
            if str(value.get("post_generation_decision") or "").upper() == "SUPPRESS":
                self._metrics["post_generation_blocks"] += 1
                reason = str(value.get("reason_code") or "")
                if reason == "generated_output_matches_constraint":
                    self._metrics["suppressions_explicit_constraint"] += 1
                elif reason == "generated_output_reincides_in_suppressed_motif":
                    self._metrics["suppressions_owner_feedback"] += 1
                    self._metrics["suppressions_repetition_fatigue"] += 1
        elif event == "constraint_created" and value.get("scope") == "durable":
            self._metrics["durable_constraints_created"] += 1
        elif event == "constraint_reverted" and value.get("scope") == "durable":
            self._metrics["durable_constraints_reverted"] += 1


GLOBAL_BEHAVIOR_OBSERVABILITY = BehaviorObservability()


__all__ = [
    "BehaviorObservability",
    "CalibrationLabel",
    "GLOBAL_BEHAVIOR_OBSERVABILITY",
]
