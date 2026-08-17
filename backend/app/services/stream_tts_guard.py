from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import sys
import threading
import time
import uuid
from typing import Any, Callable

from app.services.speech_output import TTSCancelled, TTSPlaybackTimeout
from app.services.tts_worker import TTSSynthesisTimeout, TTSWarmupInProgress


@dataclass(slots=True)
class _TTSJob:
    trace_id: str
    text: str
    speak: Callable[[str], Any]
    event_type: str
    priority: str
    optional: bool
    created_at: float
    stale_after_seconds: float
    receipt: dict[str, Any]
    done: threading.Event
    on_complete: Callable[[dict[str, Any]], None] | None = None


class StreamTTSSafetyManager:
    """Lazy, bounded best-effort TTS delivery queue."""

    def __init__(
        self,
        *,
        cancel_active: Callable[[], None] | None = None,
        abort_active: Callable[[], None] | None = None,
    ) -> None:
        self.warn_seconds = float(os.getenv("HEBE_SOCIAL_TTS_WARN_SECONDS", "5") or 5)
        self.timeout_seconds = float(os.getenv("HEBE_SOCIAL_TTS_TIMEOUT_SECONDS", "10") or 10)
        self.min_free_vram_mb = int(os.getenv("HEBE_OPTIONAL_TTS_MIN_FREE_VRAM_MB", "1800") or 1800)
        self.slow_limit = int(os.getenv("HEBE_TTS_CIRCUIT_SLOW_LIMIT", "2") or 2)
        self.circuit_open_seconds = float(os.getenv("HEBE_TTS_CIRCUIT_OPEN_SECONDS", "300") or 300)
        self.max_queue_size = max(1, int(os.getenv("HEBE_TTS_QUEUE_MAX_SIZE", "2") or 2))
        self.max_receipt_history = max(16, int(os.getenv("HEBE_TTS_RECEIPT_HISTORY", "256") or 256))
        self.default_stale_seconds = float(os.getenv("HEBE_TTS_STALE_SECONDS", "8") or 8)
        self.shutdown_timeout_seconds = float(os.getenv("HEBE_TTS_SHUTDOWN_SECONDS", "1.5") or 1.5)
        self._slow_count = 0
        self._open_until = 0.0
        self._queue: deque[_TTSJob] = deque()
        self._receipts: dict[str, tuple[dict[str, Any], threading.Event]] = {}
        self._condition = threading.Condition()
        self._stop = False
        self._worker: threading.Thread | None = None
        self._active: _TTSJob | None = None
        self._cancel_active = cancel_active
        self._abort_active = abort_active
        self.warmup_latency_ms: float | None = None
        self.warmup_status = "not_run"
        self.current_gpu_task = ""

    def configure(
        self,
        *,
        cancel_active: Callable[[], None] | None = None,
        abort_active: Callable[[], None] | None = None,
    ) -> None:
        if cancel_active is not None:
            self._cancel_active = cancel_active
        if abort_active is not None:
            self._abort_active = abort_active

    @property
    def queue_depth(self) -> int:
        with self._condition:
            return len(self._queue)

    @property
    def worker_alive(self) -> bool:
        return bool(self._worker is not None and self._worker.is_alive())

    def gpu_snapshot(self) -> dict:
        result = {"free_vram_mb": None, "total_vram_mb": None, "peak_allocated_mb": None}
        try:
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                result.update({
                    "free_vram_mb": int(free / (1024 * 1024)),
                    "total_vram_mb": int(total / (1024 * 1024)),
                    "peak_allocated_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
                })
        except Exception:
            pass
        return result

    def can_schedule_optional(self) -> tuple[bool, str, dict]:
        now = time.time()
        gpu = self.gpu_snapshot()
        if now < self._open_until:
            return False, "circuit_open", gpu
        free = gpu.get("free_vram_mb")
        if free is not None and int(free) < self.min_free_vram_mb:
            return False, "low_gpu_headroom", gpu
        return True, "safe", gpu

    @staticmethod
    def _log(outcome: str, receipt: dict[str, Any]) -> None:
        print(
            "[HEBE][TTS_DELIVERY] "
            f"outcome={outcome} trace_id={receipt.get('trace_id')} stage={receipt.get('stage')} "
            f"latency_ms={float(receipt.get('latency_ms') or 0):.0f} "
            f"queue_depth={receipt.get('queue_depth', 0)} reason={receipt.get('reason') or 'none'}",
            flush=True,
        )

    def _finish(self, job: _TTSJob, *, status: str, stage: str, reason: str = "", **details: Any) -> None:
        if self._active is job:
            self.current_gpu_task = ""
            with self._condition:
                self._active = None
        job.receipt.update({
            "status": status,
            "outcome": status,
            "stage": stage,
            "reason": reason,
            "latency_ms": (time.perf_counter() - job.created_at) * 1000,
            "queue_depth": self.queue_depth,
            **details,
        })
        job.done.set()
        self._log(status, job.receipt)
        if job.on_complete is not None:
            try:
                job.on_complete(job.receipt)
            except Exception as exc:
                print(f"[HEBE][TTS_DELIVERY] callback_failed={type(exc).__name__}", flush=True)

    def _drop(self, job: _TTSJob, *, status: str, reason: str) -> None:
        self._finish(job, status=status, stage="queue", reason=reason)

    def _start_worker_locked(self) -> None:
        if self.worker_alive:
            return
        self._stop = False
        self._worker = threading.Thread(target=self._run, name="hebe-tts-delivery", daemon=True)
        self._worker.start()

    def schedule(
        self,
        text: str,
        speak: Callable[[str], Any],
        *,
        event_type: str = "social",
        output_enabled: bool = True,
        disabled_reason: str = "stream_tts_disabled",
        trace_id: str = "",
        priority: str = "normal",
        optional: bool = True,
        stale_after_seconds: float | None = None,
        on_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict:
        trace_id = str(trace_id or f"tts_{uuid.uuid4().hex}")
        now = time.perf_counter()
        receipt: dict[str, Any] = {
            "trace_id": trace_id,
            "status": "tts_queued",
            "outcome": "tts_queued",
            "stage": "queue",
            "latency_ms": 0.0,
            "queue_depth": 0,
            "reason": "queued",
            "event_type": event_type,
            "priority": priority,
        }
        done = threading.Event()
        if len(self._receipts) >= self.max_receipt_history:
            for old_trace_id, (_old_receipt, old_done) in list(self._receipts.items()):
                if old_done.is_set():
                    self._receipts.pop(old_trace_id, None)
                    break
        self._receipts[trace_id] = (receipt, done)
        if not output_enabled:
            receipt.update({"status": "tts_cancelled", "outcome": "tts_cancelled", "reason": disabled_reason})
            done.set()
            self._log("tts_cancelled", receipt)
            return {"scheduled": False, "reason": disabled_reason, "gpu_before": {}, "receipt": receipt}
        allowed, reason, gpu_before = self.can_schedule_optional() if optional else (True, "required", self.gpu_snapshot())
        if not allowed:
            receipt.update({"status": "tts_cancelled", "outcome": "tts_cancelled", "reason": reason})
            done.set()
            self._log("tts_cancelled", receipt)
            return {"scheduled": False, "reason": reason, "gpu_before": gpu_before, "receipt": receipt}
        job = _TTSJob(
            trace_id=trace_id,
            text=str(text or ""),
            speak=speak,
            event_type=event_type,
            priority=priority,
            optional=optional,
            created_at=now,
            stale_after_seconds=float(stale_after_seconds or self.default_stale_seconds),
            receipt=receipt,
            done=done,
            on_complete=on_complete,
        )
        cancel_active = False
        with self._condition:
            if self._stop:
                self._drop(job, status="tts_cancelled", reason="manager_stopped")
                return {"scheduled": False, "reason": "manager_stopped", "gpu_before": gpu_before, "receipt": receipt}
            current = time.perf_counter()
            kept: deque[_TTSJob] = deque()
            while self._queue:
                queued = self._queue.popleft()
                if queued.optional and current - queued.created_at >= queued.stale_after_seconds:
                    self._drop(queued, status="tts_dropped_stale", reason="stale_before_start")
                else:
                    kept.append(queued)
            self._queue = kept
            if len(self._queue) >= self.max_queue_size:
                replace_index = next((i for i, queued in enumerate(self._queue) if queued.optional), None)
                if replace_index is not None:
                    replaced = self._queue[replace_index]
                    del self._queue[replace_index]
                    self._drop(replaced, status="tts_dropped_stale", reason="superseded_by_newer_speech")
                else:
                    self._drop(job, status="tts_cancelled", reason="queue_full")
                    return {"scheduled": False, "reason": "queue_full", "gpu_before": gpu_before, "receipt": receipt}
            if priority in {"direct", "farewell", "required"}:
                insert_at = 0
                while insert_at < len(self._queue) and self._queue[insert_at].priority in {
                    "direct", "farewell", "required"
                }:
                    insert_at += 1
                self._queue.insert(insert_at, job)
                cancel_active = bool(self._active is not None and self._active.optional)
            else:
                self._queue.append(job)
            receipt["queue_depth"] = len(self._queue)
            self._start_worker_locked()
            self._condition.notify()
        self._log("tts_queued", receipt)
        if cancel_active and self._cancel_active is not None:
            try:
                self._cancel_active()
            except Exception as exc:
                print(f"[HEBE][TTS_DELIVERY] cancel_active_failed={type(exc).__name__}", flush=True)
        return {"scheduled": True, "reason": "queued", "gpu_before": gpu_before, "receipt": receipt}

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stop:
                    self._condition.wait()
                if self._stop and not self._queue:
                    return
                job = self._queue.popleft()
                self._active = job
            queue_ms = (time.perf_counter() - job.created_at) * 1000
            if job.optional and queue_ms >= job.stale_after_seconds * 1000:
                self._finish(job, status="tts_dropped_stale", stage="queue", reason="stale_before_start", queue_wait_ms=queue_ms)
                with self._condition:
                    self._active = None
                continue
            self.current_gpu_task = f"tts:{job.event_type}"
            job.receipt.update({"status": "tts_started", "outcome": "tts_started", "stage": "synthesis", "queue_wait_ms": queue_ms})
            self._log("tts_started", job.receipt)
            started = time.perf_counter()
            try:
                result = job.speak(job.text)
                elapsed_ms = (time.perf_counter() - started) * 1000
                details = dict(result or {}) if isinstance(result, dict) else {}
                status = str(details.pop("status", "tts_delivered") or "tts_delivered")
                if status not in {"tts_delivered", "tts_timed_out", "tts_failed", "tts_cancelled"}:
                    status = "tts_delivered"
                synthesis_ms = float(details.get("synthesis_ms") or elapsed_ms)
                if status == "tts_delivered" and synthesis_ms > self.warn_seconds * 1000:
                    self._slow_count += 1
                elif status == "tts_delivered":
                    self._slow_count = max(0, self._slow_count - 1)
                self._finish(
                    job,
                    status=status,
                    stage=str(details.pop("stage", "completion")),
                    reason=str(details.pop("reason", "")),
                    execution_ms=elapsed_ms,
                    **details,
                )
            except (TTSSynthesisTimeout, TTSPlaybackTimeout) as exc:
                stage = "synthesis" if isinstance(exc, TTSSynthesisTimeout) else "playback"
                self._slow_count += 1
                self._finish(job, status="tts_timed_out", stage=stage, reason=type(exc).__name__)
            except TTSWarmupInProgress:
                self._finish(job, status="tts_cancelled", stage="warmup", reason="warmup_in_progress")
            except TTSCancelled as exc:
                self._finish(job, status="tts_cancelled", stage="cleanup", reason=type(exc).__name__)
            except Exception as exc:
                self._finish(job, status="tts_failed", stage="synthesis", reason=type(exc).__name__)
            finally:
                if self._slow_count >= self.slow_limit:
                    self._open_until = time.time() + self.circuit_open_seconds
                    print("[HEBE][TTS_CIRCUIT_BREAKER] state=open reason=repeated_slow_generation", flush=True)
                self.current_gpu_task = ""
                with self._condition:
                    self._active = None

    def wait(self, trace_id: str, *, timeout_seconds: float = 1.0) -> dict | None:
        item = self._receipts.get(str(trace_id or ""))
        if item is None:
            return None
        receipt, done = item
        done.wait(max(0.0, float(timeout_seconds)))
        return receipt

    def warmup(self, warm: Callable[[], Any], *, background: bool = False) -> dict:
        def run() -> None:
            started = time.perf_counter()
            try:
                try:
                    result = warm()
                except TypeError:
                    # Compatibility for the former warmup(speak, text=...) hook.
                    result = warm("Hebe lista.")
                self.warmup_latency_ms = (time.perf_counter() - started) * 1000
                self.warmup_status = str((result or {}).get("status") or "ready") if isinstance(result, dict) else "ready"
                if self.warmup_status == "ready":
                    self._slow_count = 0
                    self._open_until = 0.0
            except Exception as exc:
                self.warmup_latency_ms = (time.perf_counter() - started) * 1000
                self.warmup_status = f"error:{type(exc).__name__}"
        if background:
            threading.Thread(target=run, name="hebe-tts-warmup", daemon=True).start()
            return {"status": "scheduled", "latency_ms": None}
        run()
        return {"status": self.warmup_status, "latency_ms": self.warmup_latency_ms}

    def shutdown(self, *, timeout_seconds: float | None = None) -> dict:
        timeout = self.shutdown_timeout_seconds if timeout_seconds is None else max(0.0, float(timeout_seconds))
        with self._condition:
            self._stop = True
            while self._queue:
                self._drop(self._queue.popleft(), status="tts_cancelled", reason="shutdown")
            active = self._active
            self._condition.notify_all()
        abort = self._abort_active or self._cancel_active
        if active is not None and abort is not None:
            try:
                abort()
            except Exception:
                pass
        worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout)
        return {
            "stopped": not self.worker_alive,
            "active_cancelled": active is not None,
            "queue_depth": self.queue_depth,
        }

    def readiness(self) -> dict:
        allowed, reason, gpu = self.can_schedule_optional()
        return {
            "optional_tts_allowed": allowed,
            "reason": reason,
            "gpu": gpu,
            "current_gpu_task": self.current_gpu_task,
            "circuit_state": "open" if time.time() < self._open_until else "closed",
            "warmup_status": self.warmup_status,
            "warmup_latency_ms": self.warmup_latency_ms,
            "queue_depth": self.queue_depth,
            "queue_capacity": self.max_queue_size,
            "worker_alive": self.worker_alive,
        }
