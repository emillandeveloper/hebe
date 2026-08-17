from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter, deque
from enum import StrEnum
from typing import Any, Callable

from app.core.persistent_logs import log_behavior_session_event, log_jsonl_event


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
        session_log_fn: Callable[[str, dict], None] | None = None,
        coalesce_checkpoint_seconds: float = 15.0,
        coalesce_checkpoint_evaluations: int = 25,
        max_coalesce_keys: int = 20_000,
    ) -> None:
        self.max_recent = max(1, int(max_recent))
        self.max_labels = max(1, int(max_labels))
        self._clock = clock or time.time
        self._log_fn = log_fn or log_jsonl_event
        self._session_log_fn = session_log_fn if session_log_fn is not None else (
            log_behavior_session_event if log_fn is None else None
        )
        self.coalesce_checkpoint_seconds = max(1.0, float(coalesce_checkpoint_seconds))
        self.coalesce_checkpoint_evaluations = max(2, int(coalesce_checkpoint_evaluations))
        self.max_coalesce_keys = max(100, int(max_coalesce_keys))
        self._recent: deque[dict[str, Any]] = deque(maxlen=self.max_recent)
        self._labels: deque[dict[str, Any]] = deque(maxlen=self.max_labels)
        self._metrics: Counter[str] = Counter()
        self._lock = threading.RLock()
        self._coalesced: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    @staticmethod
    def _coalesce_signature(value: dict[str, Any]) -> str:
        event = str(value.get("event") or "")
        if event == "candidate_policy":
            comparable = [
                {
                    key: item.get(key)
                    for key in ("motif_identity", "source", "matched", "similarity", "related_event_id")
                }
                for item in list(value.get("recent_comparable_motifs") or [])
                if isinstance(item, dict)
            ]
            state = {
                "motif": value.get("normalized_motif_identity"),
                "terms": value.get("semantic_terms"),
                "comparables": comparable,
                "usage_count": value.get("usage_count"),
                "fatigue_bucket": round(float(value.get("fatigue") or 0.0), 2),
                "positive_weight": round(float(value.get("positive_weight") or 0.0), 3),
                "negative_weight": round(float(value.get("negative_weight") or 0.0), 3),
                "constraint": value.get("active_constraint"),
                "decision": value.get("policy_decision"),
                "reason": value.get("reason_code"),
                "similarity": round(float(value.get("similarity_score") or 0.0), 3),
            }
        else:
            state = {
                key: value.get(key)
                for key in (
                    "policy_decision", "reason_code", "base_score", "adjusted_score",
                    "candidate_selected", "generation_attempted", "topic",
                )
            }
        return json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _coalesce_key(value: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(value.get("stream_session_id") or ""),
            str(value.get("event") or ""),
            str(value.get("trace_id") or ""),
            str(value.get("candidate_id") or value.get("speech_intent_id") or ""),
        )

    def _write(self, value: dict[str, Any]) -> None:
        try:
            self._log_fn("behavior_calibration", value)
        except Exception as exc:
            with self._lock:
                self._metrics["telemetry_write_failed"] += 1
                self._metrics[f"telemetry_write_error:{type(exc).__name__}"] += 1
        session_id = str(value.get("stream_session_id") or "").strip()
        if session_id and self._session_log_fn is not None:
            try:
                self._session_log_fn(session_id, value)
            except Exception as exc:
                with self._lock:
                    self._metrics["session_telemetry_write_failed"] += 1
                    self._metrics[f"session_telemetry_write_error:{type(exc).__name__}"] += 1

    def record(self, event: str, *, trace_id: str = "", timestamp: float | None = None, **payload: Any) -> dict[str, Any]:
        value = _privacy_filter({
            "event": str(event),
            "trace_id": str(trace_id or f"behavior_trace_{uuid.uuid4().hex}"),
            "timestamp": float(self._clock() if timestamp is None else timestamp),
            **payload,
        })
        should_write = True
        evicted_value: dict[str, Any] | None = None
        emitted_value = value
        with self._lock:
            self._count(value)
            if value["event"] in {"candidate_policy", "candidate_ranking"}:
                key = self._coalesce_key(value)
                signature = self._coalesce_signature(value)
                state = self._coalesced.get(key)
                if state is None:
                    state = {
                        "signature": signature,
                        "evaluation_count": 1,
                        "first_seen": value["timestamp"],
                        "last_seen": value["timestamp"],
                        "last_written_count": 0,
                        "last_written_at": 0.0,
                        "latest": value,
                    }
                    self._coalesced[key] = state
                else:
                    state["evaluation_count"] += 1
                    state["last_seen"] = value["timestamp"]
                    state["latest"] = value
                    unchanged = signature == state["signature"]
                    checkpoint_due = (
                        state["evaluation_count"] - state["last_written_count"] >= self.coalesce_checkpoint_evaluations
                        or value["timestamp"] - state["last_written_at"] >= self.coalesce_checkpoint_seconds
                    )
                    should_write = not unchanged or checkpoint_due
                    if unchanged and not should_write:
                        self._metrics["telemetry_evaluations_coalesced"] += 1
                    if not unchanged:
                        self._metrics["telemetry_state_changes"] += 1
                        state["signature"] = signature
                emitted_value = {
                    **value,
                    "evaluation_count": state["evaluation_count"],
                    "evaluation_delta": state["evaluation_count"] - state["last_written_count"],
                    "first_seen": state["first_seen"],
                    "last_seen": state["last_seen"],
                    "coalesced": state["evaluation_count"] > 1,
                }
                if should_write:
                    state["last_written_count"] = state["evaluation_count"]
                    state["last_written_at"] = value["timestamp"]
                if len(self._coalesced) > self.max_coalesce_keys:
                    oldest_key = next(iter(self._coalesced))
                    if oldest_key != key:
                        oldest = self._coalesced.pop(oldest_key)
                        if oldest["evaluation_count"] > oldest["last_written_count"]:
                            evicted_value = {
                                **oldest["latest"],
                                "evaluation_count": oldest["evaluation_count"],
                                "evaluation_delta": oldest["evaluation_count"] - oldest["last_written_count"],
                                "first_seen": oldest["first_seen"],
                                "last_seen": oldest["last_seen"],
                                "coalesced": True,
                                "coalesced_flush": True,
                            }
                        self._metrics["telemetry_coalesce_keys_evicted"] += 1
            if should_write:
                self._recent.append(emitted_value)
        if should_write:
            self._write(emitted_value)
        if evicted_value is not None:
            self._write(evicted_value)
        return emitted_value

    def flush_session(self, stream_session_id: str | int) -> int:
        """Persist final counters for unchanged evaluations, then release memory."""
        session_id = str(stream_session_id or "").strip()
        pending: list[dict[str, Any]] = []
        with self._lock:
            for key, state in list(self._coalesced.items()):
                if key[0] != session_id:
                    continue
                if state["evaluation_count"] > state["last_written_count"]:
                    value = {
                        **state["latest"],
                        "evaluation_count": state["evaluation_count"],
                        "evaluation_delta": state["evaluation_count"] - state["last_written_count"],
                        "first_seen": state["first_seen"],
                        "last_seen": state["last_seen"],
                        "coalesced": True,
                        "coalesced_flush": True,
                    }
                    pending.append(value)
                    self._recent.append(value)
                self._coalesced.pop(key, None)
            self._metrics["telemetry_session_flushes"] += 1
        for value in pending:
            self._write(value)
        return len(pending)

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
                    "coalesce_keys": len(self._coalesced),
                    "coalesce_keys_limit": self.max_coalesce_keys,
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
