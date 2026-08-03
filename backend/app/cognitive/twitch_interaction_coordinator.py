from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import heapq
import re
import threading
import time
import unicodedata
from typing import Any, Callable


FINAL_JOB_STATES = {"emitted", "suppressed", "failed", "expired", "cancelled", "observed"}
TECHNICAL_FAILURE_REASONS = {
    "final_emission_gate", "stream_response_quality_guard", "stream_persona_quality_guard",
    "context_grounding_guard", "target_speaker_guard", "empty_response_after_generation",
    "missing_final_response", "guard_failed", "deterministic_repair_failed",
}


def _normalize(value: str) -> str:
    lowered = "".join(
        char for char in unicodedata.normalize("NFKD", str(value or "").casefold())
        if not unicodedata.combining(char)
    )
    return " ".join(re.sub(r"[^a-z0-9_]+", " ", lowered).split())


def semantic_key(text: str) -> str:
    tokens = [
        token for token in _normalize(text).split()
        if token not in {"hebe", "ebe", "eve", "oye", "tu", "por", "favor"}
    ]
    return " ".join(tokens[:32])


@dataclass(slots=True)
class DirectInteractionOutcome:
    semantic_key: str
    viewer: str
    original_event_id: str
    received_at: float
    generated_candidate: str = ""
    final_outcome: str = "queued"
    final_response_id: str = ""
    failure_reason: str = ""


@dataclass(slots=True)
class TwitchInteractionJob:
    event_id: str
    viewer: str
    normalized_text: str
    semantic_key: str
    category: str
    priority: int
    created_at: float
    expires_at: float
    status: str = "queued"
    response_outcome: str = ""
    final_route: str = ""
    failure_reason: str = ""
    is_direct_interaction: bool = False
    is_stream_operation: bool = False
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    event: Any = None
    processor: Callable[[Any], None] | None = field(default=None, repr=False)


class TwitchInteractionCoordinator:
    """Priority/single-flight boundary for potential public Twitch interactions."""

    PRIORITIES = {
        "owner_stream_operation": 1,
        "important_stream_event": 2,
        "direct_reply": 3,
        "direct_question": 4,
        "explicit_thread": 5,
        "high_value_no_mention": 6,
        "spontaneous_banter": 7,
        "low_value": 8,
    }

    def __init__(self, *, repeat_window_seconds: float = 120.0) -> None:
        self.repeat_window_seconds = float(repeat_window_seconds)
        self._lock = threading.RLock()
        self._queue: list[tuple[int, float, int, TwitchInteractionJob]] = []
        self._sequence = 0
        self._active: TwitchInteractionJob | None = None
        self._draining = False
        self._drain_token = 0
        self.jobs: dict[str, TwitchInteractionJob] = {}
        self.direct_outcomes: dict[tuple[str, str], DirectInteractionOutcome] = {}
        self.max_generation_in_flight = 0
        self._generation_in_flight = 0

    def reset_session(self) -> None:
        """Drop queued work and semantic outcomes from the previous stream."""
        with self._lock:
            self._drain_token += 1
            self._queue.clear()
            self._active = None
            self._draining = False
            self.jobs.clear()
            self.direct_outcomes.clear()
            self.max_generation_in_flight = 0
            self._generation_in_flight = 0

    @property
    def active_job(self) -> TwitchInteractionJob | None:
        with self._lock:
            return self._active

    def classify_event(self, event: Any) -> tuple[str, int, bool]:
        event_type = str(getattr(event, "event_type", "") or "")
        payload = dict(getattr(event, "payload", {}) or {})
        text = str(payload.get("message_text") or payload.get("text") or "")
        direct_reply = bool(payload.get("reply_to_hebe_message"))
        direct = bool(
            direct_reply or payload.get("direct_address_to_hebe") or payload.get("mentions_hebe")
            or re.search(r"\b(?:hebe|ebe|eve|jebe|heve)\b", _normalize(text))
        )
        if event_type in {"twitch_raid", "twitch_sub", "twitch_cheer", "twitch_follow_batch"}:
            return "important_stream_event", 2, False
        if direct_reply:
            return "direct_reply", 3, True
        if direct and ("?" in text or re.search(r"\b(?:que|quien|como|cuando|donde|cual|por que|crees|sabes)\b", _normalize(text))):
            return "direct_question", 4, True
        if direct:
            return "direct_question", 4, True
        if payload.get("explicit_hebe_thread") or payload.get("thread_match"):
            return "explicit_thread", 5, False
        if float(payload.get("reply_value_score") or payload.get("social_value") or 0.0) >= 0.75:
            return "high_value_no_mention", 6, False
        if event_type == "twitch_idle_prompt":
            return "spontaneous_banter", 7, False
        return "low_value", 8, False

    def submit(self, event: Any, processor: Callable[[Any], None]) -> TwitchInteractionJob:
        now = time.time()
        payload = deepcopy(dict(getattr(event, "payload", {}) or {}))
        event_id = str(payload.get("event_id") or payload.get("message_id") or f"twitch_job_{int(now * 1_000_000)}")
        payload.setdefault("event_id", event_id)
        event.payload = payload
        category, priority, direct = self.classify_event(event)
        viewer = str(payload.get("user_login") or payload.get("username") or payload.get("display_name") or "viewer").casefold()
        raw = str(payload.get("message_text") or payload.get("text") or "")
        key = semantic_key(raw)
        ttl = 15.0 if priority >= 7 else 120.0 if direct else 45.0
        job = TwitchInteractionJob(
            event_id=event_id,
            viewer=viewer,
            normalized_text=_normalize(raw),
            semantic_key=key,
            category=category,
            priority=priority,
            created_at=now,
            expires_at=now + ttl,
            is_direct_interaction=direct,
            context_snapshot={
                "viewer": viewer,
                "raw_text": raw,
                "normalized_text": _normalize(raw),
                "reply_parent": deepcopy({k: v for k, v in payload.items() if k.startswith("reply_")}),
                "recent_relevant_thread": deepcopy(payload.get("recent_chat") or []),
                "game_context": deepcopy(payload.get("game_run_state") or {}),
                "stream_moment": deepcopy({k: payload.get(k) for k in ("current_game", "current_activity", "title")}),
                "authority": "viewer",
                "intended_speech_act": category,
            },
            event=deepcopy(event),
            processor=processor,
        )
        with self._lock:
            repeat = self._semantic_repeat(job, now=now)
            if repeat == "dedupe":
                job.status = "observed"
                job.response_outcome = "already_answered_repeat"
                self.jobs[event_id] = job
                self._log_outcome(job, "observed", "already_answered_repeat")
                return job
            if repeat == "preserve_block":
                job.status = "suppressed"
                job.response_outcome = "policy_blocked"
                job.failure_reason = "repeated_policy_block"
                self.jobs[event_id] = job
                self.direct_outcomes[(viewer, key)] = DirectInteractionOutcome(
                    key, viewer, event_id, now,
                    final_outcome="policy_blocked",
                    failure_reason="repeated_policy_block",
                )
                self._log_outcome(job, "suppressed", "repeated_policy_block")
                return job
            if direct:
                self._cancel_waiting_low_value(reason="preempted_by_direct_question")
                if self._active is not None and self._active.priority >= 7:
                    self._active.status = "cancelled"
                    self._active.failure_reason = "preempted_by_direct_question"
                    print(
                        f"[HEBE][TWITCH_INTERACTION_QUEUE] event_id={self._active.event_id} "
                        f"priority={self._active.priority} action=cancel_active_before_emission",
                        flush=True,
                    )
            self._sequence += 1
            self.jobs[event_id] = job
            heapq.heappush(self._queue, (priority, now, self._sequence, job))
            print(f"[HEBE][TWITCH_INTERACTION_QUEUE] event_id={event_id} priority={priority} action=enqueue", flush=True)
            if direct:
                self.direct_outcomes[(viewer, key)] = DirectInteractionOutcome(key, viewer, event_id, now)
            should_drain = not self._draining
            if should_drain:
                self._draining = True
                self._drain_token += 1
                drain_token = self._drain_token
        if should_drain:
            self._drain(drain_token)
        return job

    def submit_owner_stream_operation(self, *, event_id: str, text: str, processor: Callable[[], Any]) -> Any:
        with self._lock:
            self._cancel_waiting_low_value(reason="preempted_by_owner_stream_op")
            if self._active is not None and self._active.priority >= 7:
                self._active.status = "cancelled"
                self._active.failure_reason = "preempted_by_owner_stream_op"
                print(
                    f"[HEBE][TWITCH_INTERACTION_QUEUE] event_id={self._active.event_id} "
                    f"priority={self._active.priority} action=cancel_active_before_emission",
                    flush=True,
                )
            print(f"[HEBE][TWITCH_INTERACTION_QUEUE] event_id={event_id} priority=1 action=replace", flush=True)
        return processor()

    def allows_final_emission(self, event_id: str) -> bool:
        """Return false when a higher-priority interaction cancelled this job."""
        with self._lock:
            job = self.jobs.get(str(event_id or ""))
            return job is None or job.status != "cancelled"

    def _drain(self, drain_token: int) -> None:
        try:
            while True:
                with self._lock:
                    job = self._next_job()
                    if job is None:
                        self._active = None
                        self._draining = False
                        return
                    self._active = job
                    job.status = "generating"
                    self._generation_in_flight += 1
                    self.max_generation_in_flight = max(self.max_generation_in_flight, self._generation_in_flight)
                    print(f"[HEBE][TWITCH_INTERACTION_ACTIVE] event_id={job.event_id} stage=generating", flush=True)
                try:
                    self._assert_context_ownership(job)
                    if job.processor is None:
                        raise RuntimeError("missing_twitch_job_processor")
                    job.processor(job.event)
                except Exception as exc:
                    job.status = "failed"
                    job.failure_reason = f"{type(exc).__name__}: {exc}"
                    self._finish_direct(job, "failed", job.failure_reason)
                    self._log_outcome(job, "failed", job.failure_reason)
                finally:
                    with self._lock:
                        self._generation_in_flight = max(0, self._generation_in_flight - 1)
                        if job.status not in FINAL_JOB_STATES:
                            outcome = "failed" if job.is_direct_interaction else "observed"
                            reason = (
                                "generated_but_not_emitted"
                                if job.response_outcome == "generated_candidate"
                                else "direct_interaction_not_finalized"
                            ) if job.is_direct_interaction else "no_public_response"
                            job.status = outcome
                            job.failure_reason = reason
                            self._finish_direct(job, outcome, reason)
                            self._log_outcome(job, outcome, reason)
                        self._active = None
        finally:
            with self._lock:
                if self._drain_token == drain_token:
                    self._draining = False

    def _next_job(self) -> TwitchInteractionJob | None:
        now = time.time()
        while self._queue:
            _, _, _, job = heapq.heappop(self._queue)
            if job.status == "cancelled":
                continue
            if job.expires_at <= now:
                job.status = "expired"
                self._finish_direct(job, "expired", "queue_ttl")
                print(f"[HEBE][TWITCH_INTERACTION_QUEUE] event_id={job.event_id} priority={job.priority} action=expire", flush=True)
                continue
            return job
        return None

    def _cancel_waiting_low_value(self, *, reason: str) -> None:
        for _, _, _, queued in self._queue:
            if queued.priority >= 7 and queued.status == "queued":
                queued.status = "cancelled"
                queued.failure_reason = reason
                print(f"[HEBE][TWITCH_INTERACTION_QUEUE] event_id={queued.event_id} priority={queued.priority} action=cancel", flush=True)

    def _semantic_repeat(self, job: TwitchInteractionJob, *, now: float) -> str:
        if not job.is_direct_interaction or not job.semantic_key:
            return ""
        previous = self.direct_outcomes.get((job.viewer, job.semantic_key))
        if previous is None or now - previous.received_at > self.repeat_window_seconds:
            return ""
        if previous.final_outcome == "emitted":
            print(f"[HEBE][SEMANTIC_REPEAT] viewer={job.viewer} previous_outcome=emitted action=dedupe", flush=True)
            return "dedupe"
        if previous.final_outcome == "policy_blocked":
            print(f"[HEBE][SEMANTIC_REPEAT] viewer={job.viewer} previous_outcome=policy_blocked action=preserve_block", flush=True)
            return "preserve_block"
        if previous.final_outcome in {"failed", "suppressed", "expired"}:
            print(f"[HEBE][SEMANTIC_REPEAT] viewer={job.viewer} previous_outcome={previous.final_outcome} action=retry_unanswered", flush=True)
            job.response_outcome = "unanswered_retry"
        return ""

    def record_candidate(self, event_id: str, text: str) -> None:
        with self._lock:
            job = self.jobs.get(str(event_id or "")) or self._active
            if job is None or job.status == "cancelled":
                return
            job.status = "validating"
            job.response_outcome = "generated_candidate"
            if job.is_direct_interaction:
                outcome = self.direct_outcomes.get((job.viewer, job.semantic_key))
                if outcome:
                    outcome.generated_candidate = str(text or "")

    def record_emission(self, event_id: str, result: dict[str, Any], *, reason: str = "") -> None:
        with self._lock:
            job = self.jobs.get(str(event_id or "")) or self._active
            if job is None or job.status == "cancelled":
                return
            emitted = bool(result.get("emitted"))
            route = str(result.get("route") or "")
            # The engine's explicit reason carries the guard/policy cause.  The
            # gate's generic ``suppressed_route`` must not erase that detail.
            result_reason = str(reason or result.get("reason") or "")
            if emitted:
                outcome = "emitted"
            elif (
                any(marker in result_reason for marker in TECHNICAL_FAILURE_REASONS)
                or result_reason.startswith("stage_")
            ):
                outcome = "failed"
            else:
                outcome = "suppressed"
            job.status = outcome
            job.response_outcome = outcome
            job.final_route = route
            job.failure_reason = result_reason
            self._finish_direct(job, outcome, result_reason, final_response_id=str(result.get("event_id") or ""))
            self._log_outcome(job, outcome, result_reason or route)

    def record_policy_suppression(self, event_id: str, reason: str) -> None:
        """Finalize an intentional policy/safety refusal without claiming a reply."""
        with self._lock:
            job = self.jobs.get(str(event_id or "")) or self._active
            if job is None:
                return
            job.status = "suppressed"
            job.response_outcome = "policy_blocked"
            job.failure_reason = str(reason or "policy_blocked")
            self._finish_direct(job, "policy_blocked", job.failure_reason)
            self._log_outcome(job, "suppressed", job.failure_reason)

    def _finish_direct(self, job: TwitchInteractionJob, outcome: str, reason: str, *, final_response_id: str = "") -> None:
        if not job.is_direct_interaction:
            return
        direct = self.direct_outcomes.get((job.viewer, job.semantic_key))
        if direct is None:
            return
        direct.final_outcome = outcome
        direct.failure_reason = reason
        direct.final_response_id = final_response_id
        print(
            f"[HEBE][DIRECT_INTERACTION_OUTCOME] viewer={job.viewer} semantic_key={job.semantic_key!r} "
            f"outcome={outcome} reason={reason}",
            flush=True,
        )

    def _assert_context_ownership(self, job: TwitchInteractionJob) -> None:
        payload = dict(getattr(job.event, "payload", {}) or {})
        event_id = str(payload.get("event_id") or payload.get("message_id") or "")
        viewer = str(payload.get("user_login") or payload.get("username") or payload.get("display_name") or "viewer").casefold()
        if event_id != job.event_id or viewer != job.context_snapshot.get("viewer"):
            raise RuntimeError("context_ownership_mismatch")

    def _log_outcome(self, job: TwitchInteractionJob, outcome: str, reason: str) -> None:
        print(
            f"[HEBE][TWITCH_INTERACTION_OUTCOME] event_id={job.event_id} outcome={outcome} reason={reason}",
            flush=True,
        )


class TrollEngagementBudget:
    """Small topic-level budget; it does not alter PresenceEngine thresholds."""

    TOPICS = {
        "salami": {"salami", "chorizo", "mandanga", "embutido"},
        "porros": {"porro", "porros", "marihuana", "canuto"},
        "picnic": {"picnic", "merienda"},
        "unknown_bait": {"crotolamo", "padalustro", "permanganato"},
        "compliment_fishing": {"cumplido", "cumplidos", "piropo", "piropos", "halago", "halagos"},
        "flirt_bait": {"flirtea", "coquetea", "ligar", "ligue", "guapo", "guapa"},
        "jealousy_bait": {"celoso", "celosa", "celos"},
        "obedience_test": {"obedece", "desobedece", "atreves", "mandato"},
        "degrading_identity_bait": {"esclava", "sirvienta", "sumisa", "mascota"},
        "sexual_innuendo_loop": {"sexual", "sexy", "innuendo", "doble", "sentido"},
        "repeated_command_fishing": {"hazlo", "repitelo", "otra", "vez"},
    }

    def __init__(self, *, window_seconds: float = 300.0) -> None:
        self.window_seconds = float(window_seconds)
        self._engagements: dict[tuple[str, str], list[float]] = {}
        self._topic_state: dict[tuple[str, str], dict[str, Any]] = {}

    def reset_session(self) -> None:
        self._engagements.clear()
        self._topic_state.clear()

    def topic_for(self, text: str) -> str:
        tokens = set(_normalize(text).split())
        for topic, terms in self.TOPICS.items():
            if tokens & terms:
                return topic
        return ""

    def evaluate(self, *, viewer: str, text: str) -> dict[str, Any]:
        topic = self.topic_for(text)
        if not topic:
            return {"topic": "", "engagements": 0, "action": "allow", "reason": "not_bait_topic"}
        now = time.time()
        key = (str(viewer or "viewer").casefold(), topic)
        recent = [ts for ts in self._engagements.get(key, []) if now - ts <= self.window_seconds]
        self._engagements[key] = recent
        state = dict(self._topic_state.get(key) or {})
        if state.get("closed_by_owner"):
            print(f"[HEBE][BAIT_TOPIC_STATE] viewer={viewer} topic={topic} engagements={len(recent)} closed_by_owner=true", flush=True)
            print(f"[HEBE][BAIT_LOOP_GUARD] viewer={viewer} topic={topic} engagements={len(recent)} action=boundary reason=owner_closed_topic", flush=True)
            return {"topic": topic, "engagements": len(recent), "action": "boundary", "reason": "owner_closed_topic", "closed_by_owner": True}
        action = "allow" if len(recent) == 0 else "close" if len(recent) == 1 else "observe"
        reason = "first_safe_bait" if action == "allow" else "second_closes_topic" if action == "close" else "bait_topic_budget_exhausted"
        print(f"[HEBE][BAIT_LOOP_GUARD] viewer={viewer} topic={topic} engagements={len(recent)} action={action} reason={reason}", flush=True)
        state.update({"viewer": key[0], "topic": topic, "engagements": len(recent), "closed": action in {"close", "observe"}, "closed_by_owner": False, "updated_at": now})
        self._topic_state[key] = state
        print(f"[HEBE][BAIT_TOPIC_STATE] viewer={viewer} topic={topic} engagements={len(recent)} closed_by_owner=false", flush=True)
        return {"topic": topic, "engagements": len(recent), "action": action, "reason": reason, "closed_by_owner": False}

    def record_engagement(self, *, viewer: str, text: str) -> None:
        topic = self.topic_for(text)
        if not topic:
            return
        key = (str(viewer or "viewer").casefold(), topic)
        self._engagements.setdefault(key, []).append(time.time())

    def close_topic_by_owner(self, *, viewer: str, topic: str) -> None:
        key = (str(viewer or "viewer").casefold(), str(topic or "").strip())
        state = dict(self._topic_state.get(key) or {})
        state.update({"viewer": key[0], "topic": key[1], "closed": True, "closed_by_owner": True, "owner_intervention": True, "updated_at": time.time()})
        self._topic_state[key] = state
        print(f"[HEBE][BAIT_TOPIC_STATE] viewer={viewer} topic={topic} engagements={len(self._engagements.get(key, []))} closed_by_owner=true", flush=True)
