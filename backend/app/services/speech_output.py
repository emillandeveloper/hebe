from __future__ import annotations

import os
import threading
import time
import wave
from typing import Callable, Optional

import pygame

from app.services.tts_service import (
    cancel as cancel_synthesis,
    shutdown as shutdown_synthesis,
    speak as tts_to_wav,
    warmup as warmup_synthesis,
)
from app.services.tts_worker import TTSSynthesisFailure, TTSSynthesisTimeout
from app.services.vts_client import vts_hotkey

HEBE_TTS_VOLUME = float(os.getenv("HEBE_TTS_VOLUME", "0.9"))
HEBE_TTS_PLAYBACK_GRACE_SECONDS = float(os.getenv("HEBE_TTS_PLAYBACK_GRACE_SECONDS", "3") or 3)
HEBE_TTS_PLAYBACK_MAX_SECONDS = float(os.getenv("HEBE_TTS_PLAYBACK_MAX_SECONDS", "90") or 90)


class TTSPlaybackTimeout(TimeoutError):
    pass


class TTSCancelled(RuntimeError):
    pass


class SpeechOutputController:
    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._playback_lock = threading.Lock()

    @staticmethod
    def _wav_duration(path: str) -> float:
        try:
            with wave.open(path, "rb") as wav:
                rate = wav.getframerate()
                return wav.getnframes() / float(rate) if rate else 0.0
        except Exception:
            return 0.0

    def speak(
        self,
        text: str,
        language: str = "es",
        emit: Optional[Callable[[str, dict], None]] = None,
        log_chat: Optional[Callable[[str, str, str], None]] = None,
        emit_chat: bool = True,
        trace_id: str = "",
    ) -> dict:
        if not text:
            return {"status": "tts_cancelled", "reason": "empty_text"}
        if emit and emit_chat:
            emit("debug.tts_candidate", {
                "text_length": len(text),
                "response_stage": "generated",
                "trace_id": trace_id,
            })

        audio_path = ""
        self._cancelled.clear()
        total_started = time.perf_counter()
        try:
            if emit:
                emit("tts.status", {"outcome": "tts_started", "stage": "synthesis", "trace_id": trace_id})
            audio_path, synthesis = tts_to_wav(
                text=text,
                language=language,
                emit=emit,
                log_chat=log_chat,
            )
            if self._cancelled.is_set():
                raise TTSCancelled("TTS cancelled after synthesis")
            if emit:
                emit("tts.status", {
                    "outcome": "tts_synthesis_completed",
                    "stage": "synthesis",
                    "trace_id": trace_id,
                    "latency_ms": synthesis.latency_ms,
                })

            try:
                vts_hotkey("HebeTalking")
            except Exception:
                pass

            expected_duration = self._wav_duration(audio_path)
            playback_deadline = min(
                HEBE_TTS_PLAYBACK_MAX_SECONDS,
                max(HEBE_TTS_PLAYBACK_GRACE_SECONDS, expected_duration + HEBE_TTS_PLAYBACK_GRACE_SECONDS),
            )
            playback_started = time.perf_counter()
            if emit:
                emit("tts.status", {
                    "outcome": "tts_playback_started",
                    "stage": "playback",
                    "trace_id": trace_id,
                    "expected_duration_ms": expected_duration * 1000,
                })
            with self._playback_lock:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.set_volume(float(HEBE_TTS_VOLUME))
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    if self._cancelled.is_set():
                        pygame.mixer.music.stop()
                        raise TTSCancelled("TTS playback cancelled")
                    if time.perf_counter() - playback_started > playback_deadline:
                        pygame.mixer.music.stop()
                        raise TTSPlaybackTimeout(
                            f"TTS playback exceeded {playback_deadline:.2f}s"
                        )
                    time.sleep(0.05)
            playback_ms = (time.perf_counter() - playback_started) * 1000
            result = {
                "status": "tts_delivered",
                "stage": "completion",
                "synthesis_ms": synthesis.latency_ms,
                "playback_ms": playback_ms,
                "latency_ms": (time.perf_counter() - total_started) * 1000,
                "backend": synthesis.backend,
                "trace_id": trace_id,
            }
            if emit:
                emit("tts.status", result)
            return result
        except TTSSynthesisTimeout:
            if emit:
                emit("tts.status", {"outcome": "tts_timed_out", "stage": "synthesis", "trace_id": trace_id})
            raise
        except TTSPlaybackTimeout:
            if emit:
                emit("tts.status", {"outcome": "tts_timed_out", "stage": "playback", "trace_id": trace_id})
            raise
        except TTSCancelled:
            if emit:
                emit("tts.status", {"outcome": "tts_cancelled", "stage": "cleanup", "trace_id": trace_id})
            raise
        except TTSSynthesisFailure as exc:
            if self._cancelled.is_set():
                if emit:
                    emit("tts.status", {"outcome": "tts_cancelled", "stage": "cleanup", "trace_id": trace_id})
                raise TTSCancelled("TTS synthesis cancelled") from exc
            raise
        finally:
            try:
                vts_hotkey("HebeIdle")
            except Exception:
                pass
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass
            try:
                if audio_path:
                    os.remove(audio_path)
            except OSError as exc:
                print(f"[HEBE][TTS_CLEANUP] wav_delete_failed error={type(exc).__name__}", flush=True)

    def warmup(self, *, timeout_seconds: float | None = None) -> dict:
        started = time.perf_counter()
        receipt = warmup_synthesis(timeout_seconds=timeout_seconds)
        return {
            "status": "ready",
            "backend": receipt.backend,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    def cancel(self) -> None:
        self.cancel_playback()
        cancel_synthesis()

    def cancel_playback(self) -> None:
        self._cancelled.set()
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def shutdown(self, *, timeout_seconds: float = 1.0) -> None:
        self.cancel()
        shutdown_synthesis(timeout_seconds=timeout_seconds)


controller = SpeechOutputController()


def speak(
    text: str,
    language: str = "es",
    emit: Optional[Callable[[str, dict], None]] = None,
    log_chat: Optional[Callable[[str, str, str], None]] = None,
    emit_chat: bool = True,
    trace_id: str = "",
) -> dict:
    """
    High-level speech output:
    - Emits chat.assistant
    - Generates WAV via tts_service
    - Triggers VTS talking/idle
    - Plays WAV via pygame
    - Cleans up temp file
    """
    return controller.speak(
        text,
        language=language,
        emit=emit,
        log_chat=log_chat,
        emit_chat=emit_chat,
        trace_id=trace_id,
    )


def warmup(*, timeout_seconds: float | None = None) -> dict:
    return controller.warmup(timeout_seconds=timeout_seconds)


def cancel() -> None:
    controller.cancel()


def shutdown(*, timeout_seconds: float = 1.0) -> None:
    controller.shutdown(timeout_seconds=timeout_seconds)
