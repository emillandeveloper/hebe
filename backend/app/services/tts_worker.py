from __future__ import annotations

import multiprocessing
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any


class TTSSynthesisTimeout(TimeoutError):
    pass


class TTSSynthesisFailure(RuntimeError):
    pass


class TTSWarmupInProgress(RuntimeError):
    pass


def _pick_backend() -> str:
    mode = os.getenv("HEBE_TTS_MODE", "auto").strip().lower()
    if mode in {"piper", "xtts"}:
        return mode
    try:
        import torch

        minimum = float(os.getenv("HEBE_TTS_MIN_VRAM_GB", "12") or 12)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if props.total_memory / (1024**3) >= minimum:
                return "xtts"
    except Exception:
        pass
    piper_exe = os.getenv("HEBE_PIPER_EXE", "")
    return "piper" if piper_exe and os.path.exists(piper_exe) else "xtts"


def _synthesis_worker(requests, responses) -> None:
    backend = _pick_backend()
    while True:
        request = requests.get()
        if not isinstance(request, dict):
            continue
        kind = str(request.get("kind") or "")
        if kind == "shutdown":
            return
        request_id = str(request.get("request_id") or "")
        started = time.perf_counter()
        try:
            if kind == "warmup":
                if backend == "xtts":
                    from app.services.tts_xtts import ensure_xtts_loaded

                    ensure_xtts_loaded()
            elif kind == "synthesize":
                text = str(request.get("text") or "")
                wav_path = str(request.get("wav_path") or "")
                language = str(request.get("language") or "es")
                if backend == "piper":
                    from app.services.tts_piper import piper_to_wav

                    piper_to_wav(text=text, wav_path=wav_path, language=language)
                else:
                    from app.services.tts_xtts import xtts_to_wav

                    xtts_to_wav(text=text, wav_path=wav_path, language=language)
            else:
                raise ValueError(f"unsupported TTS worker request: {kind}")
            responses.put({
                "request_id": request_id,
                "ok": True,
                "backend": backend,
                "latency_ms": (time.perf_counter() - started) * 1000,
            })
        except BaseException as exc:
            responses.put({
                "request_id": request_id,
                "ok": False,
                "backend": backend,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })


@dataclass(slots=True)
class SynthesisReceipt:
    backend: str
    latency_ms: float


class TTSSynthesisWorker:
    """Persistent, killable synthesis process.

    Coqui/XTTS does not expose cooperative cancellation. Keeping inference in a
    child process lets a deadline terminate the actual model work and release
    its CUDA/process resources instead of abandoning a live Python thread.
    """

    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._process = None
        self._requests = None
        self._responses = None
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._current_kind = ""

    @property
    def is_alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def _start_locked(self) -> None:
        if self.is_alive:
            return
        self._close_queues_locked()
        self._requests = self._context.Queue(maxsize=1)
        self._responses = self._context.Queue(maxsize=1)
        self._process = self._context.Process(
            target=_synthesis_worker,
            args=(self._requests, self._responses),
            name="hebe-tts-synthesis",
            daemon=True,
        )
        self._process.start()

    def _close_queues_locked(self) -> None:
        for channel in (self._requests, self._responses):
            if channel is None:
                continue
            try:
                channel.close()
                channel.cancel_join_thread()
            except Exception:
                pass
        self._requests = None
        self._responses = None

    def _terminate_locked(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)
        self._close_queues_locked()

    def _request(self, payload: dict[str, Any], *, timeout_seconds: float) -> SynthesisReceipt:
        timeout_seconds = max(0.05, float(timeout_seconds))
        with self._request_lock:
            with self._lock:
                self._start_locked()
                requests = self._requests
                responses = self._responses
                process = self._process
                self._current_kind = str(payload.get("kind") or "")
            request_id = uuid.uuid4().hex
            request = {**payload, "request_id": request_id}
            try:
                assert requests is not None
                assert responses is not None
                requests.put(request, timeout=min(timeout_seconds, 1.0))
                deadline = time.monotonic() + timeout_seconds
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        with self._lock:
                            if self._process is process:
                                self._terminate_locked()
                        raise TTSSynthesisTimeout(f"TTS synthesis exceeded {timeout_seconds:.2f}s")
                    try:
                        response = responses.get(timeout=min(remaining, 0.1))
                    except queue.Empty:
                        if process is None or not process.is_alive():
                            with self._lock:
                                if self._process is process:
                                    self._terminate_locked()
                            raise TTSSynthesisFailure("TTS synthesis worker exited")
                        continue
                    except (EOFError, OSError, ValueError):
                        raise TTSSynthesisFailure("TTS synthesis was cancelled")
                    if str(response.get("request_id") or "") != request_id:
                        continue
                    if not response.get("ok"):
                        detail = str(response.get("error") or response.get("error_type") or "provider failed")
                        raise TTSSynthesisFailure(detail)
                    return SynthesisReceipt(
                        backend=str(response.get("backend") or "unknown"),
                        latency_ms=float(response.get("latency_ms") or 0.0),
                    )
            finally:
                with self._lock:
                    self._current_kind = ""

    def synthesize(
        self,
        *,
        text: str,
        wav_path: str,
        language: str = "es",
        timeout_seconds: float,
    ) -> SynthesisReceipt:
        with self._lock:
            warmup_active = self._current_kind == "warmup"
        if warmup_active:
            raise TTSWarmupInProgress("TTS model warmup is still in progress")
        return self._request(
            {"kind": "synthesize", "text": text, "wav_path": wav_path, "language": language},
            timeout_seconds=timeout_seconds,
        )

    def warmup(self, *, timeout_seconds: float) -> SynthesisReceipt:
        return self._request({"kind": "warmup"}, timeout_seconds=timeout_seconds)

    def cancel(self) -> None:
        with self._lock:
            self._terminate_locked()

    def shutdown(self, *, timeout_seconds: float = 1.0) -> None:
        with self._lock:
            process = self._process
            if process is None:
                self._close_queues_locked()
                return
            if process.is_alive() and self._requests is not None:
                try:
                    self._requests.put_nowait({"kind": "shutdown"})
                except Exception:
                    pass
                process.join(timeout=max(0.0, float(timeout_seconds)))
            if process.is_alive():
                self._terminate_locked()
            else:
                self._process = None
                self._close_queues_locked()
