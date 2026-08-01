from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import sys
import threading
import time
from typing import Callable


class StreamTTSSafetyManager:
    def __init__(self) -> None:
        self.warn_seconds = float(os.getenv("HEBE_SOCIAL_TTS_WARN_SECONDS", "5") or 5)
        self.timeout_seconds = float(os.getenv("HEBE_SOCIAL_TTS_TIMEOUT_SECONDS", "10") or 10)
        self.min_free_vram_mb = int(os.getenv("HEBE_OPTIONAL_TTS_MIN_FREE_VRAM_MB", "1800") or 1800)
        self.slow_limit = int(os.getenv("HEBE_TTS_CIRCUIT_SLOW_LIMIT", "2") or 2)
        self.circuit_open_seconds = float(os.getenv("HEBE_TTS_CIRCUIT_OPEN_SECONDS", "300") or 300)
        self._slow_count = 0
        self._open_until = 0.0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hebe-social-tts")
        self.warmup_latency_ms: float | None = None
        self.warmup_status = "not_run"
        self.current_gpu_task = ""

    def gpu_snapshot(self) -> dict:
        result = {"free_vram_mb": None, "total_vram_mb": None, "peak_allocated_mb": None}
        try:
            # Never import/load a GPU framework from a raid handler. Use an
            # already-warm runtime only; unavailable telemetry fails neutral.
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

    def schedule(self, text: str, speak: Callable[[str], None], *, event_type: str = "social") -> dict:
        allowed, reason, gpu_before = self.can_schedule_optional()
        if not allowed:
            print(f"[HEBE][RAID_ACK_TTS] status=skipped reason={reason}", flush=True)
            return {"scheduled": False, "reason": reason, "gpu_before": gpu_before}
        queued_at = time.perf_counter()
        print("[HEBE][RAID_ACK_TTS] status=queued generation_ms=0 playback_ms=0", flush=True)

        def run() -> None:
            started = time.perf_counter()
            queue_ms = (started - queued_at) * 1000
            self.current_gpu_task = f"tts:{event_type}"
            print("[HEBE][RAID_ACK_TTS] status=started generation_ms=0 playback_ms=0", flush=True)
            try:
                speak(text)
                total_ms = (time.perf_counter() - started) * 1000
                status = "timed_out" if total_ms > self.timeout_seconds * 1000 else "completed"
                if total_ms > self.warn_seconds * 1000:
                    self._slow_count += 1
                else:
                    self._slow_count = max(0, self._slow_count - 1)
                if self._slow_count >= self.slow_limit:
                    self._open_until = time.time() + self.circuit_open_seconds
                    print("[HEBE][TTS_CIRCUIT_BREAKER] state=open reason=repeated_slow_generation", flush=True)
                print(
                    f"[HEBE][RAID_ACK_TTS] status={status} generation_ms={total_ms:.0f} playback_ms=0",
                    flush=True,
                )
                print(
                    "[HEBE][TTS_PERF] "
                    f"queue_wait_ms={queue_ms:.0f} synthesis_ms={total_ms:.0f} playback_ms=0 "
                    f"total_ms={total_ms + queue_ms:.0f} text_length={len(text)} backend=configured "
                    f"device=cuda gpu_free_mb={gpu_before.get('free_vram_mb')} status={status}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[HEBE][RAID_ACK_TTS] status=skipped error={type(exc).__name__}:{exc}", flush=True)
            finally:
                self.current_gpu_task = ""

        self._executor.submit(run)
        return {"scheduled": True, "reason": "queued", "gpu_before": gpu_before}

    def warmup(self, speak: Callable[[str], None], *, text: str = "Hebe lista.") -> dict:
        started = time.perf_counter()
        try:
            speak(text)
            self.warmup_latency_ms = (time.perf_counter() - started) * 1000
            self.warmup_status = "ready"
        except Exception as exc:
            self.warmup_latency_ms = (time.perf_counter() - started) * 1000
            self.warmup_status = f"error:{type(exc).__name__}"
        return {"status": self.warmup_status, "latency_ms": self.warmup_latency_ms}

    def readiness(self) -> dict:
        allowed, reason, gpu = self.can_schedule_optional()
        return {
            "optional_tts_allowed": allowed, "reason": reason, "gpu": gpu,
            "current_gpu_task": self.current_gpu_task,
            "circuit_state": "open" if time.time() < self._open_until else "closed",
            "warmup_status": self.warmup_status, "warmup_latency_ms": self.warmup_latency_ms,
        }
