# backend/app/services/stt_whisper.py
from __future__ import annotations

import os
import inspect
import math
import re
import statistics
import time
import unicodedata
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum
from difflib import SequenceMatcher
from typing import Callable, Optional, Tuple

import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from app.services.direct_stt_command import (
    DirectUtteranceIntentFamily,
    parse_direct_stt_command,
)


class STTMode(str, Enum):
    AMBIENT_DISCOURSE = "ambient_discourse"
    COMMAND_CANDIDATE = "command_candidate"
    DIRECT_COMMAND = "direct_command"
    PENDING_SHORT_ANSWER = "pending_short_answer"
    DIAGNOSTIC = "diagnostic"


class UnsupportedLanguagePolicy(str, Enum):
    DUAL_DECODE_THEN_DROP = "dual_decode_then_drop"


@dataclass(slots=True)
class STTCommandHypothesis:
    ambient_text: str
    command_text: str
    ambient_language: str | None = None
    command_language: str | None = None
    wake_detected: bool = False
    wake_score: float = 0.0
    action_structure_score: float = 0.0
    hypothesis_agreement: float = 0.0
    target_candidates: tuple[str, ...] = ()
    final_command_text: str = ""
    command_confidence: float = 0.0
    action_eligible: bool = False
    decision: str = "reject"

    def as_dict(self) -> dict:
        return {
            "ambient_text": self.ambient_text,
            "command_text": self.command_text,
            "ambient_language": self.ambient_language,
            "command_language": self.command_language,
            "wake_detected": self.wake_detected,
            "wake_score": self.wake_score,
            "action_structure_score": self.action_structure_score,
            "hypothesis_agreement": self.hypothesis_agreement,
            "target_candidates": list(self.target_candidates),
            "final_command_text": self.final_command_text,
            "command_confidence": self.command_confidence,
            "action_eligible": self.action_eligible,
            "decision": self.decision,
        }


DEFAULT_COMMAND_PROMPT_WORDS = [
    "Hebe",
    "Ebe",
    "Eve",
    "E.B.",
    "E.V.",
    "abre",
    "abrir",
    "inicia",
    "lanza",
    "ejecuta",
    "pon",
    "OBS",
    "OBS Studio",
]

_PROMPT_LOOP_WORDS = {
    "hebe",
    "ebe",
    "eve",
    "leo",
    "obs",
    "abre",
    "abrir",
    "inicia",
    "lanza",
    "ejecuta",
    "pon",
    "stream",
    "chat",
    "promo",
    "shoutout",
    "twitch",
    "so",
    "nuria",
    "charlie",
    "xarly",
    "totodile",
    "jotun",
    "zwei",
    "persona",
    "final",
    "fantasy",
}


def build_stt_prompt_profile(profile: str = "app_command", *, max_chars: int | None = None) -> str:
    profile = str(profile or "app_command").strip().lower()
    if profile == "app_command":
        entries = list(DEFAULT_COMMAND_PROMPT_WORDS)
        try:
            from app.services.app_registry import list_whitelisted_apps
            for app in list_whitelisted_apps():
                entries.extend([
                    str(app.get("display_name") or ""),
                    str(app.get("app_id") or ""),
                    *[str(alias) for alias in app.get("aliases") or []],
                ])
        except Exception:
            pass
    elif profile == "promotion_command":
        entries = ["Hebe", "Ebe", "Eve", "promo", "promociona", "shoutout", "SO"]
    elif profile == "stream_operation":
        entries = ["Hebe", "Ebe", "Eve", "stream", "directo", "OBS", "chat", "activa", "desactiva"]
    else:
        entries = ["sí", "no", "cancela", "continúa"]
    return build_stt_command_prompt(", ".join(entries), max_chars=max_chars, log=False)


def _norm_prompt_token(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).strip(" ,.;")


def build_stt_command_prompt(raw_prompt: str | None = None, *, max_chars: int | None = None, log: bool = True) -> str:
    source = raw_prompt if raw_prompt is not None else ", ".join(DEFAULT_COMMAND_PROMPT_WORDS)
    configured_max = int(os.getenv("HEBE_STT_COMMAND_PROMPT_MAX_CHARS", "180") or "180")
    limit = max(40, int(max_chars or configured_max))
    seen: set[str] = set()
    words: list[str] = []
    for piece in re.split(r"[,;\n]+", str(source or "")):
        token = _norm_prompt_token(piece)
        if not token:
            continue
        key = token.casefold()
        if key in seen:
            continue
        candidate = ", ".join([*words, token]) + "."
        if len(candidate) > limit:
            break
        seen.add(key)
        words.append(token)
    prompt = ", ".join(words).strip()
    if prompt and not prompt.endswith("."):
        prompt += "."
    if log:
        print(f"[HEBE][STT][PROMPT] built length={len(prompt)} words={len(words)} deduped=true", flush=True)
    return prompt


def is_stt_prompt_injection(text: str, *, command_prompt: str | None = None) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    normalized = re.sub(r"[^a-z0-9, ]+", " ", value.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    prompt = build_stt_command_prompt(command_prompt, log=False)
    prompt_norm = re.sub(r"[^a-z0-9, ]+", " ", prompt.casefold())
    prompt_norm = re.sub(r"\s+", " ", prompt_norm).strip()
    if prompt_norm and normalized == prompt_norm:
        return True
    if prompt_norm and len(normalized) >= 40 and normalized in prompt_norm:
        return True

    return is_stt_prompt_hotword_list(value)


def is_stt_prompt_hotword_list(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    normalized = re.sub(r"[^a-z0-9, ]+", " ", value.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    comma_parts = [_norm_prompt_token(part).casefold() for part in value.split(",")]
    comma_parts = [part for part in comma_parts if part]
    if len(comma_parts) >= 4:
        promptish = []
        for part in comma_parts:
            part_tokens = re.findall(r"[a-z0-9]+", part)
            if part in _PROMPT_LOOP_WORDS or (
                part_tokens and all(token in _PROMPT_LOOP_WORDS for token in part_tokens)
            ):
                promptish.append(part)
        if len(promptish) >= 4 and len(promptish) / max(1, len(comma_parts)) >= 0.75:
            return True

    tokens = re.findall(r"[a-z0-9]+", normalized)
    if len(tokens) >= 5:
        vocab_hits = [token for token in tokens if token in _PROMPT_LOOP_WORDS]
        if len(vocab_hits) >= 5 and len(vocab_hits) / max(1, len(tokens)) >= 0.75:
            return True
        for i in range(0, max(0, len(tokens) - 4)):
            window = tokens[i : i + 5]
            if window == ["obs", "stream", "chat", "promo", "shoutout"]:
                return True
    return False


@dataclass
class STTConfig:
    rate: int = 16000
    channels: int = 1
    chunk: int = 1024
    input_device_index: Optional[int] = (
        int(os.getenv("HEBE_STT_INPUT_DEVICE") or os.getenv("HEBE_INPUT_DEVICE_INDEX"))
        if (os.getenv("HEBE_STT_INPUT_DEVICE") or os.getenv("HEBE_INPUT_DEVICE_INDEX") or "").isdigit()
        else None
    )
    input_device_name: Optional[str] = os.getenv("HEBE_STT_INPUT_DEVICE_NAME") or None
    input_device_host_api: Optional[str] = os.getenv("HEBE_STT_INPUT_DEVICE_HOST_API") or None
    input_device_signature: Optional[str] = os.getenv("HEBE_STT_INPUT_DEVICE_SIGNATURE") or None
    input_device_sample_rate: Optional[int] = (
        int(os.getenv("HEBE_STT_INPUT_DEVICE_SAMPLE_RATE"))
        if (os.getenv("HEBE_STT_INPUT_DEVICE_SAMPLE_RATE") or "").isdigit()
        else None
    )
    input_device_channels: Optional[int] = (
        int(os.getenv("HEBE_STT_INPUT_DEVICE_CHANNELS"))
        if (os.getenv("HEBE_STT_INPUT_DEVICE_CHANNELS") or "").isdigit()
        else None
    )

    silence_threshold: float = 0.01
    silence_rms_threshold: float = float(os.getenv("HEBE_STT_SILENCE_RMS_THRESHOLD", "0.003") or "0.003")
    silence_warning_after_seconds: float = float(os.getenv("HEBE_STT_SILENCE_WARNING_AFTER_SECONDS", "10") or "10")
    silence_warning_rate_limit_seconds: float = float(
        os.getenv("HEBE_STT_SILENCE_LOG_COOLDOWN_SECONDS")
        or os.getenv("HEBE_STT_SILENCE_WARNING_RATE_LIMIT_SECONDS", "60")
        or "60"
    )
    verbose_device_logs: bool = os.getenv("HEBE_VERBOSE_STT_DEVICE_LOGS", "false").strip().lower() in ("1", "true", "yes", "on")
    max_device_open_retries: int = int(os.getenv("HEBE_STT_MAX_DEVICE_OPEN_RETRIES", "1") or "1")
    retry_backoff_seconds: float = float(os.getenv("HEBE_STT_RETRY_BACKOFF_SECONDS", "30") or "30")
    disable_on_device_open_failure: bool = os.getenv(
        "HEBE_STT_DISABLE_ON_DEVICE_OPEN_FAILURE",
        "true",
    ).strip().lower() in ("1", "true", "yes", "on")
    max_record_seconds: float = 8.0
    min_record_seconds: float = 0.5
    silence_end_seconds: float = 0.8

    model_size: str = os.getenv("HEBE_WHISPER_MODEL", "medium")
    device: str = os.getenv("HEBE_WHISPER_DEVICE", "cuda")
    compute_type: str = os.getenv("HEBE_WHISPER_COMPUTE", "int8_float16")
    explicit_cpu_fallback: bool = os.getenv("HEBE_WHISPER_CPU_FALLBACK", "false").strip().lower() in ("1", "true", "yes", "on")
    ambient_language: str = os.getenv("HEBE_STT_AMBIENT_LANGUAGE", "auto").strip().lower() or "auto"
    preroll_seconds: float = float(os.getenv("HEBE_STT_PREROLL_SECONDS", "1.0") or "1.0")
    tts_echo_tail_ms: int = int(os.getenv("HEBE_STT_TTS_ECHO_TAIL_MS", "750") or "750")
    allowed_languages: tuple[str, ...] = tuple(
        part.strip().lower()
        for part in os.getenv("HEBE_STT_ALLOWED_LANGUAGES", "es,en").split(",")
        if part.strip()
    ) or ("es", "en")
    unsupported_language_policy: str = os.getenv(
        "HEBE_STT_UNSUPPORTED_LANGUAGE_POLICY", "dual_decode_then_drop",
    ).strip().lower() or "dual_decode_then_drop"
    short_audio_language_threshold_seconds: float = float(
        os.getenv("HEBE_STT_SHORT_LANGUAGE_THRESHOLD_SECONDS", "2.0") or "2.0"
    )
    language_recovery_min_score: float = float(
        os.getenv("HEBE_STT_LANGUAGE_RECOVERY_MIN_SCORE", "0.62") or "0.62"
    )
    language_recovery_min_margin: float = float(
        os.getenv("HEBE_STT_LANGUAGE_RECOVERY_MIN_MARGIN", "0.06") or "0.06"
    )
    default_language: str = os.getenv("HEBE_STT_DEFAULT_LANGUAGE", "es").strip().lower() or "es"
    command_language: str = os.getenv("HEBE_STT_COMMAND_LANGUAGE", "es").strip().lower() or "es"
    restrict_auto_language: bool = os.getenv("HEBE_STT_RESTRICT_AUTO_LANGUAGE", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    force_language_for_commands: bool = os.getenv("HEBE_STT_FORCE_LANGUAGE_FOR_COMMANDS", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    task: str = "transcribe"
    command_beam_size: int = int(os.getenv("HEBE_STT_COMMAND_BEAM_SIZE", "5") or "5")
    command_temperature: float = float(os.getenv("HEBE_STT_COMMAND_TEMPERATURE", "0") or "0")
    command_prompt: str = os.getenv("HEBE_STT_COMMAND_PROMPT", ", ".join(DEFAULT_COMMAND_PROMPT_WORDS))
    command_prompt_max_chars: int = int(os.getenv("HEBE_STT_COMMAND_PROMPT_MAX_CHARS", "180") or "180")
    command_prompt_enabled: bool = os.getenv("HEBE_STT_COMMAND_PROMPT_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    log_rejected_raw: bool = os.getenv("HEBE_STT_LOG_REJECTED_RAW", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    auto_disable_prompt_on_echo: bool = os.getenv("HEBE_STT_AUTO_DISABLE_PROMPT_ON_ECHO", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    prompt_echo_window_seconds: float = float(os.getenv("HEBE_STT_PROMPT_ECHO_WINDOW_SECONDS", "300") or "300")
    prompt_echo_disable_threshold: int = int(os.getenv("HEBE_STT_PROMPT_ECHO_DISABLE_THRESHOLD", "2") or "2")

    def __post_init__(self) -> None:
        self.model_size = os.getenv("HEBE_WHISPER_MODEL", self.model_size)
        self.device = os.getenv("HEBE_WHISPER_DEVICE", self.device)
        self.compute_type = os.getenv("HEBE_WHISPER_COMPUTE", self.compute_type)
        self.explicit_cpu_fallback = os.getenv("HEBE_WHISPER_CPU_FALLBACK", str(self.explicit_cpu_fallback)).strip().lower() in ("1", "true", "yes", "on")
        self.ambient_language = os.getenv("HEBE_STT_AMBIENT_LANGUAGE", self.ambient_language).strip().lower() or "auto"
        self.preroll_seconds = max(0.0, min(2.0, float(os.getenv("HEBE_STT_PREROLL_SECONDS", str(self.preroll_seconds)) or self.preroll_seconds)))
        self.tts_echo_tail_ms = max(0, int(os.getenv("HEBE_STT_TTS_ECHO_TAIL_MS", str(self.tts_echo_tail_ms)) or self.tts_echo_tail_ms))
        self.silence_warning_rate_limit_seconds = float(
            os.getenv("HEBE_STT_SILENCE_LOG_COOLDOWN_SECONDS")
            or os.getenv("HEBE_STT_SILENCE_WARNING_RATE_LIMIT_SECONDS", str(self.silence_warning_rate_limit_seconds))
            or self.silence_warning_rate_limit_seconds
        )
        self.verbose_device_logs = os.getenv("HEBE_VERBOSE_STT_DEVICE_LOGS", str(self.verbose_device_logs)).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.allowed_languages = tuple(
            part.strip().lower()
            for part in os.getenv("HEBE_STT_ALLOWED_LANGUAGES", ",".join(self.allowed_languages)).split(",")
            if part.strip()
        ) or ("es", "en")
        self.unsupported_language_policy = os.getenv(
            "HEBE_STT_UNSUPPORTED_LANGUAGE_POLICY", self.unsupported_language_policy,
        ).strip().lower() or UnsupportedLanguagePolicy.DUAL_DECODE_THEN_DROP.value
        self.short_audio_language_threshold_seconds = max(
            0.0,
            float(os.getenv(
                "HEBE_STT_SHORT_LANGUAGE_THRESHOLD_SECONDS",
                str(self.short_audio_language_threshold_seconds),
            ) or self.short_audio_language_threshold_seconds),
        )
        self.language_recovery_min_score = min(1.0, max(
            0.0,
            float(os.getenv(
                "HEBE_STT_LANGUAGE_RECOVERY_MIN_SCORE",
                str(self.language_recovery_min_score),
            ) or self.language_recovery_min_score),
        ))
        self.language_recovery_min_margin = min(1.0, max(
            0.0,
            float(os.getenv(
                "HEBE_STT_LANGUAGE_RECOVERY_MIN_MARGIN",
                str(self.language_recovery_min_margin),
            ) or self.language_recovery_min_margin),
        ))
        self.default_language = os.getenv("HEBE_STT_DEFAULT_LANGUAGE", self.default_language).strip().lower() or "es"
        self.command_language = os.getenv("HEBE_STT_COMMAND_LANGUAGE", self.command_language).strip().lower() or "es"
        self.restrict_auto_language = os.getenv("HEBE_STT_RESTRICT_AUTO_LANGUAGE", str(self.restrict_auto_language)).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.force_language_for_commands = os.getenv("HEBE_STT_FORCE_LANGUAGE_FOR_COMMANDS", str(self.force_language_for_commands)).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.command_prompt_enabled = os.getenv("HEBE_STT_COMMAND_PROMPT_ENABLED", str(self.command_prompt_enabled)).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.log_rejected_raw = os.getenv("HEBE_STT_LOG_REJECTED_RAW", str(self.log_rejected_raw)).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.auto_disable_prompt_on_echo = os.getenv(
            "HEBE_STT_AUTO_DISABLE_PROMPT_ON_ECHO",
            str(self.auto_disable_prompt_on_echo),
        ).strip().lower() in ("1", "true", "yes", "on")
        self.task = "transcribe"
        self.command_prompt = build_stt_command_prompt(
            os.getenv("HEBE_STT_COMMAND_PROMPT", self.command_prompt),
            max_chars=self.command_prompt_max_chars,
        )


DEFAULT_BLACKLIST = [
    "subtítulos por la comunidad de amara.org",
    "subtitulos por la comunidad de amara.org",
    "suscríbete",
    "suscribete",
]


class STTDeviceOpenFailure(RuntimeError):
    def __init__(self, message: str, *, device: dict | None = None, attempts: int = 0):
        super().__init__(message)
        self.device = device or {}
        self.attempts = attempts


class STTService:
    def __init__(
        self,
        config: STTConfig | None = None,
        emit: Optional[Callable[[str, dict], None]] = None,
        log_chat: Optional[Callable[[str, str, str], None]] = None,
        blacklist: Optional[list[str]] = None,
    ):
        self.cfg = config or STTConfig()
        self.emit = emit
        self.log_chat = log_chat
        self.blacklist = blacklist or DEFAULT_BLACKLIST
        self.cfg.command_prompt = build_stt_command_prompt(self.cfg.command_prompt, max_chars=self.cfg.command_prompt_max_chars)
        self._model: WhisperModel | None = None
        self.selected_input_device_id = str(self.cfg.input_device_index) if self.cfg.input_device_index is not None else ""
        self.selected_input_device_name = self.cfg.input_device_name or ""
        self.selected_input_host_api = self.cfg.input_device_host_api or ""
        self.selected_input_signature = self.cfg.input_device_signature or ""
        self.selected_input_sample_rate = self.cfg.input_device_sample_rate or self.cfg.rate
        self.selected_input_channels = self.cfg.input_device_channels or self.cfg.channels
        self.last_input_level = 0.0
        self.last_input_rms = 0.0
        self.last_input_peak = 0.0
        self.last_input_device_error: str | None = None
        self.status = "idle"
        self.failed_input_signature = ""
        self.failed_input_device_id = ""
        self.failed_input_error = ""
        self.failed_input_ts = 0.0
        self._open_fail_counts: dict[str, int] = {}
        self._last_silence_warning_ts = 0.0
        self.last_audio_np: np.ndarray | None = None
        self.last_speech_detected = False
        self.last_transcription_language: str | None = None
        self.last_transcription_task = self.cfg.task
        self.last_transcription_options: dict = {}
        self.last_rejected_stt: dict = {}
        self._prompt_echo_rejection_ts: list[float] = []
        self.engine_status = "idle"
        self.engine_error = ""
        self.engine_load_seconds = 0.0
        self.effective_model = self.cfg.model_size
        self.effective_device = self.cfg.device
        self.effective_compute_type = self.cfg.compute_type
        self.cuda_available: bool | None = None
        self.active_mode = STTMode.AMBIENT_DISCOURSE
        self.last_ambient_transcript = ""
        self.last_command_redecode: dict = {}
        self.last_direct_stt_result: dict = {}
        self.last_command_confidence = 0.0
        self.last_wake_decision = ""
        self.last_application_target_resolution: dict = {}
        self.last_rejection_reason = ""
        self.last_detected_language: str | None = None
        self.last_detected_language_probability: float | None = None
        self._latencies: deque[float] = deque(maxlen=120)
        self._transcription_timestamps: deque[float] = deque(maxlen=240)
        self._perf_last: dict = {}
        self.rejected_transcripts = 0
        self.command_success_count = 0
        self.command_failure_count = 0
        self.hallucination_reject_count = 0
        self.prompt_echo_reject_count = 0
        self._tts_active = False
        self._tts_ended_at = 0.0
        self._recent_tts_text = ""
        self._speech_start_offset = 0.0
        self.last_result_metadata: dict = {}
        self.last_gpu_snapshot: dict = {}

        self._silence_frames_needed = int(self.cfg.silence_end_seconds / (self.cfg.chunk / self.cfg.rate))

    def init(self) -> None:
        if self._model is None:
            started = time.perf_counter()
            self.engine_status = "loading"
            try:
                import ctranslate2
                self.cuda_available = bool(ctranslate2.get_cuda_device_count() > 0)
            except Exception:
                self.cuda_available = None
            try:
                self._model = WhisperModel(self.cfg.model_size, device=self.cfg.device, compute_type=self.cfg.compute_type)
                self.engine_status = "ready"
            except Exception as exc:
                self.engine_error = f"{type(exc).__name__}: {exc}"
                self.engine_status = "error"
                if not self.cfg.explicit_cpu_fallback or self.cfg.device == "cpu":
                    self.engine_load_seconds = time.perf_counter() - started
                    self._log_engine_profile()
                    raise
                print(f"[HEBE][STT_ENGINE] configured_profile_failed error={self.engine_error!r}", flush=True)
                self._model = WhisperModel(self.cfg.model_size, device="cpu", compute_type="int8")
                self.effective_device = "cpu"
                self.effective_compute_type = "int8"
                self.engine_status = "fallback"
            self.engine_load_seconds = time.perf_counter() - started
            self._log_engine_profile()

    def _log_engine_profile(self) -> None:
        print(
            "[HEBE][STT_ENGINE] "
            f"engine=faster_whisper model={self.effective_model} device={self.effective_device} "
            f"compute_type={self.effective_compute_type} cuda_available={self.cuda_available} "
            f"load_seconds={self.engine_load_seconds:.3f} status={self.engine_status}",
            flush=True,
        )

    def _is_blacklisted(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return True
        for bad in self.blacklist:
            if bad in t:
                return True
        return False

    def _emit(self, event_type: str, data: dict | None = None) -> None:
        if self.emit:
            try:
                self.emit(event_type, data or {})
            except Exception:
                pass

    def _command_language(self) -> str | None:
        if self.cfg.force_language_for_commands:
            return self.cfg.command_language
        if self.cfg.restrict_auto_language:
            return self.cfg.default_language
        return None

    def _transcribe_audio(
        self,
        audio_np: np.ndarray,
        *,
        language: str | None = None,
        mode: STTMode | str | None = None,
        command_mode: bool | None = None,
        force_prompt: bool | None = None,
        expected_vocabulary: list[str] | tuple[str, ...] | None = None,
        prompt_profile: str = "app_command",
        queue_wait_seconds: float = 0.0,
        final_decision: str = "decoded",
    ) -> tuple[str, dict]:
        self.init()
        assert self._model is not None
        if mode is None:
            mode = STTMode.DIRECT_COMMAND if command_mode is not False else STTMode.AMBIENT_DISCOURSE
        mode = STTMode(mode)
        command_mode = mode == STTMode.DIRECT_COMMAND
        prompt_enabled = command_mode if force_prompt is None else bool(force_prompt and command_mode)
        selected_language = language
        if mode == STTMode.AMBIENT_DISCOURSE and selected_language is None:
            selected_language = None if self.cfg.ambient_language == "auto" else self.cfg.ambient_language
        elif selected_language is None and command_mode:
            selected_language = self._command_language()
        elif mode == STTMode.PENDING_SHORT_ANSWER and selected_language is None:
            selected_language = None

        options = {
            "language": selected_language,
            "task": "transcribe",
            "beam_size": max(1, int(self.cfg.command_beam_size if command_mode else 5)),
            "vad_filter": True,
        }
        if command_mode:
            options["temperature"] = float(self.cfg.command_temperature)
            if prompt_enabled and self.cfg.command_prompt_enabled:
                focused_prompt = build_stt_prompt_profile(
                    prompt_profile,
                    max_chars=self.cfg.command_prompt_max_chars,
                )
                if focused_prompt:
                    options["initial_prompt"] = focused_prompt
                    options["hotwords"] = focused_prompt
        elif mode == STTMode.PENDING_SHORT_ANSWER and expected_vocabulary:
            bounded = build_stt_command_prompt(", ".join(expected_vocabulary), max_chars=120, log=False)
            if bounded:
                options["initial_prompt"] = bounded
                options["hotwords"] = bounded

        prompt_entries = [p.strip() for p in str(options.get("initial_prompt") or "").strip(".").split(",") if p.strip()]
        print(
            "[HEBE][STT_PROMPT_PROFILE] "
            f"mode={mode.value} profile={prompt_profile if command_mode else 'pending_answer' if mode == STTMode.PENDING_SHORT_ANSWER else 'none'} "
            f"enabled={str(bool(prompt_entries)).lower()} entries={len(prompt_entries)} "
            f"entries_source={'expected_candidates' if mode == STTMode.PENDING_SHORT_ANSWER else 'focused_command' if command_mode else 'none'}",
            flush=True,
        )

        try:
            supported = set(inspect.signature(self._model.transcribe).parameters)
            options = {key: value for key, value in options.items() if key in supported}
        except Exception:
            options.pop("hotwords", None)

        self.last_transcription_language = selected_language
        self.last_transcription_task = "transcribe"
        self.last_transcription_options = dict(options)
        self.active_mode = mode
        print(
            "[HEBE][STT][TRANSCRIBE] "
            f"using initial_prompt={str(bool(options.get('initial_prompt'))).lower()} "
            f"language={selected_language!r} task='transcribe' mode={mode.value} command_mode={command_mode}",
            flush=True,
        )
        self.last_gpu_snapshot = self._gpu_snapshot()
        print(
            "[HEBE][GPU_SAFETY] "
            f"task=whisper_decode free_vram_mb={self.last_gpu_snapshot.get('free_vram_mb')} "
            f"peak_allocated_mb={self.last_gpu_snapshot.get('peak_allocated_mb')}",
            flush=True,
        )
        decode_started = time.perf_counter()
        segments_iter, info = self._model.transcribe(audio_np, **options)
        segments = list(segments_iter)
        text = "".join(seg.text for seg in segments).strip()
        decode_seconds = time.perf_counter() - decode_started
        model_detected_language = getattr(info, "language", None)
        detected_language = (
            selected_language
            if selected_language in self.cfg.allowed_languages
            else model_detected_language
        )
        language_probability = getattr(info, "language_probability", None)
        self.last_detected_language = detected_language
        self.last_detected_language_probability = language_probability
        language_allowed = not detected_language or detected_language in self.cfg.allowed_languages
        language_decision = "allow" if language_allowed else "context_only"
        if mode == STTMode.AMBIENT_DISCOURSE:
            print(
                "[HEBE][STT_LANGUAGE] "
                f"mode={mode.value} configured={self.cfg.ambient_language} detected={detected_language} "
                f"allowed={','.join(self.cfg.allowed_languages)} decision={language_decision}",
                flush=True,
            )
        metadata = {
            "language": selected_language,
            "detected_language": detected_language,
            "model_detected_language": model_detected_language,
            "language_probability": language_probability,
            "language_allowed": language_allowed,
            "task": "transcribe",
            "command_mode": command_mode,
            "mode": mode.value,
            "action_eligible": bool(mode == STTMode.DIRECT_COMMAND and language_allowed),
            "options": dict(options),
            "avg_log_probability": self._segment_metric(segments, "avg_logprob", default=-1.25),
            "no_speech_probability": self._segment_metric(segments, "no_speech_prob", default=0.5),
            "compression_ratio": self._segment_metric(segments, "compression_ratio", default=1.0),
        }
        self._record_performance(
            mode=mode,
            audio_np=audio_np,
            decode_seconds=decode_seconds,
            queue_wait_seconds=queue_wait_seconds,
            text=text,
            detected_language=detected_language,
            final_decision=final_decision,
        )
        metadata.update(self._perf_last)
        return text, metadata

    @staticmethod
    def _segment_metric(segments: list, name: str, *, default: float) -> float:
        values: list[float] = []
        for segment in segments:
            value = getattr(segment, name, None)
            if value is not None:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass
        return statistics.fmean(values) if values else float(default)

    @staticmethod
    def _script_compatibility(text: str, language: str) -> float:
        value = str(text or "")
        letters = [char for char in value if char.isalpha()]
        if not letters:
            return 0.0
        latin = sum(
            1 for char in letters
            if "LATIN" in unicodedata.name(char, "")
        )
        # Both accepted languages use Latin script. Foreign scripts are never
        # promoted merely because a forced decode produced command vocabulary.
        return latin / max(1, len(letters)) if language in {"es", "en"} else 0.0

    def _language_candidate_quality(
        self, text: str, metadata: dict, *, language: str, audio_np: np.ndarray,
    ) -> dict:
        normalized = self._normalize_guard_text(text)
        tokens = normalized.split()
        avg_logprob = float(metadata.get("avg_log_probability", -1.25) or -1.25)
        no_speech = min(1.0, max(0.0, float(metadata.get("no_speech_probability", 0.5) or 0.0)))
        compression = max(0.0, float(metadata.get("compression_ratio", 1.0) or 1.0))
        logprob_score = 1.0 / (1.0 + math.exp(-3.0 * (avg_logprob + 0.75)))
        no_speech_score = 1.0 - no_speech
        compression_score = max(0.0, 1.0 - max(0.0, compression - 1.8) / 1.2)
        repetition_ratio = 0.0
        if tokens:
            repetition_ratio = 1.0 - len(set(tokens)) / len(tokens)
        repetition_score = max(0.0, 1.0 - repetition_ratio * 1.6)
        script_score = self._script_compatibility(text, language)
        duration = len(audio_np) / max(1, self.cfg.rate)
        chars_per_second = len(str(text or "").strip()) / max(0.25, duration)
        duration_score = 1.0 if 1.0 <= chars_per_second <= 28.0 else 0.5 if chars_per_second <= 38.0 else 0.0
        prompt_tokens = set(self._normalize_guard_text(self.cfg.command_prompt).split())
        prompt_overlap = (
            len(set(tokens) & prompt_tokens) / max(1, len(set(tokens)))
            if tokens else 0.0
        )
        prompt_score = max(0.0, 1.0 - prompt_overlap)
        score = (
            0.34 * logprob_score
            + 0.22 * no_speech_score
            + 0.10 * compression_score
            + 0.10 * repetition_score
            + 0.12 * script_score
            + 0.07 * duration_score
            + 0.05 * prompt_score
        )
        language_probability = float(metadata.get("language_probability", 0.0) or 0.0)
        score *= 0.85 + 0.15 * min(1.0, max(0.0, language_probability))
        if prompt_overlap >= 0.75:
            score *= 0.72
        if not normalized or is_stt_prompt_hotword_list(text) or is_stt_prompt_injection(text):
            score = 0.0
        return {
            "text": str(text or "").strip(),
            "language": language,
            "score": round(score, 4),
            "avg_log_probability": avg_logprob,
            "no_speech_probability": no_speech,
            "compression_ratio": compression,
            "repetition_ratio": round(repetition_ratio, 4),
            "script_compatibility": round(script_score, 4),
            "prompt_overlap": round(prompt_overlap, 4),
            "audio_duration_seconds": duration,
            "language_probability": language_probability,
        }

    def _dual_decode_language_recovery(
        self,
        audio_np: np.ndarray,
        *,
        initial_language: str | None,
        initial_text: str = "",
        short_audio: bool = False,
    ) -> dict:
        candidates: list[dict] = []
        raw_results: dict[str, tuple[str, dict]] = {}
        for language in ("es", "en"):
            text, metadata = self._transcribe_audio(
                audio_np,
                language=language,
                mode=STTMode.AMBIENT_DISCOURSE,
                force_prompt=False,
                final_decision="language_recovery",
            )
            raw_results[language] = (text, metadata)
            candidates.append(self._language_candidate_quality(
                text, metadata, language=language, audio_np=audio_np,
            ))
        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
        best, runner_up = ranked[0], ranked[1]
        initial_allowed = str(initial_language or "") in self.cfg.allowed_languages
        same_text = (
            bool(self._normalize_guard_text(best["text"]))
            and self._normalize_guard_text(best["text"]) == self._normalize_guard_text(runner_up["text"])
        )
        margin = float(best["score"]) - float(runner_up["score"])
        if short_audio and initial_allowed:
            matching = next(
                (item for item in candidates if item["language"] == initial_language),
                None,
            )
            if matching and float(matching["score"]) >= self.cfg.language_recovery_min_score:
                best = matching
                margin = max(margin, self.cfg.language_recovery_min_margin)
        required_score = self.cfg.language_recovery_min_score
        if not initial_allowed and float(best.get("prompt_overlap", 0.0)) >= 0.5:
            required_score = max(required_score, 0.78)
        accepted = bool(
            best["text"]
            and float(best["score"]) >= required_score
            and (
                margin >= self.cfg.language_recovery_min_margin
                or (short_audio and initial_allowed)
                or (same_text and initial_allowed)
            )
        )
        selected_language = str(best["language"]) if accepted else ""
        selected_text = str(best["text"]) if accepted else ""
        selected_metadata = dict(raw_results.get(selected_language, ("", {}))[1]) if accepted else {}
        recovery = {
            "policy": UnsupportedLanguagePolicy.DUAL_DECODE_THEN_DROP.value,
            "initial_language": str(initial_language or ""),
            "initial_text": str(initial_text or ""),
            "spanish_text": raw_results["es"][0],
            "spanish_score": next(item["score"] for item in candidates if item["language"] == "es"),
            "english_text": raw_results["en"][0],
            "english_score": next(item["score"] for item in candidates if item["language"] == "en"),
            "selected_language": selected_language,
            "selected_text": selected_text,
            "score_margin": round(margin, 4),
            "decision": "accept" if accepted else "reject",
            "accepted": accepted,
            "short_audio": short_audio,
            "candidates": candidates,
        }
        print(
            "[HEBE][STT_LANGUAGE_RECOVERY] "
            f"initial_language={initial_language or ''} "
            f"spanish_text={raw_results['es'][0]!r} spanish_score={recovery['spanish_score']:.3f} "
            f"english_text={raw_results['en'][0]!r} english_score={recovery['english_score']:.3f} "
            f"selected_language={selected_language or 'none'} decision={recovery['decision']}",
            flush=True,
        )
        rejection_class = ""
        if not accepted:
            if not best["text"] and not runner_up["text"]:
                rejection_class = "no_speech" if max(
                    float(best.get("no_speech_probability", 0.0)),
                    float(runner_up.get("no_speech_probability", 0.0)),
                ) >= 0.6 else "low_quality_audio"
            elif short_audio and initial_allowed:
                rejection_class = "bilingual_recovery_conflict"
            elif not initial_allowed:
                rejection_class = "unsupported_language"
            else:
                rejection_class = "low_quality_audio"
        return {
            **recovery,
            "metadata": {
                **selected_metadata,
                "detected_language": selected_language,
                "language_allowed": accepted,
                "language_recovery": recovery,
                "action_eligible": False,
                "command_mode": False,
            } if accepted else {
                "detected_language": str(initial_language or ""),
                "language_allowed": False,
                "language_recovery": recovery,
                "action_eligible": False,
                "command_mode": False,
                "rejection_reason": "unsupported_language_recovery_failed",
                "rejection_class": rejection_class,
            },
        }

    def classify_empty_transcript(
        self,
        *,
        metadata: dict | None = None,
        audio_np: np.ndarray | None = None,
        speech_detected: bool | None = None,
    ) -> str:
        data = dict(metadata or {})
        if audio_np is None or len(audio_np) == 0:
            return "empty_audio"
        rms = float(np.sqrt(np.mean(np.square(audio_np)))) if len(audio_np) else 0.0
        detected = bool(rms >= self.cfg.silence_rms_threshold) if speech_detected is None else bool(speech_detected)
        if not detected:
            return "empty_audio"
        no_speech = float(data.get("no_speech_probability", 0.5) or 0.0)
        if no_speech >= 0.6:
            return "no_speech"
        return "low_quality_audio"

    def retry_last_language_recovery(self, *, initial_language: str | None = None) -> dict:
        if self.last_audio_np is None or not self.last_speech_detected:
            return {
                "attempted": False, "accepted": False, "text": "",
                "reason": "no_recent_speech_audio",
            }
        result = self._dual_decode_language_recovery(
            self.last_audio_np,
            initial_language=initial_language,
            initial_text=self.last_ambient_transcript,
            short_audio=(
                len(self.last_audio_np) / max(1, self.cfg.rate)
                < self.cfg.short_audio_language_threshold_seconds
            ),
        )
        return {
            **result,
            "attempted": True,
            "text": result.get("selected_text", ""),
        }

    @staticmethod
    def _gpu_snapshot() -> dict:
        result = {"free_vram_mb": None, "total_vram_mb": None, "peak_allocated_mb": None}
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                result.update({
                    "free_vram_mb": int(free / (1024 * 1024)),
                    "total_vram_mb": int(total / (1024 * 1024)),
                    "peak_allocated_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
                })
        except Exception:
            pass
        return result

    def _record_performance(
        self, *, mode: STTMode, audio_np: np.ndarray, decode_seconds: float,
        queue_wait_seconds: float, text: str, detected_language: str | None,
        final_decision: str,
    ) -> None:
        audio_seconds = len(audio_np) / max(1, self.cfg.rate)
        rtf = decode_seconds / audio_seconds if audio_seconds > 0 else 0.0
        total = max(0.0, queue_wait_seconds) + decode_seconds
        self._latencies.append(total)
        self._transcription_timestamps.append(time.time())
        self._perf_last = {
            "mode": mode.value, "audio_duration_seconds": audio_seconds,
            "transcription_duration_seconds": decode_seconds, "real_time_factor": rtf,
            "queue_wait_seconds": queue_wait_seconds, "total_latency_seconds": total,
            "model": self.effective_model, "device": self.effective_device,
            "compute_type": self.effective_compute_type, "text_length": len(text),
            "detected_language": detected_language, "final_decision": final_decision,
        }
        print(
            "[HEBE][STT_PERF] "
            f"mode={mode.value} audio_seconds={audio_seconds:.3f} decode_seconds={decode_seconds:.3f} "
            f"rtf={rtf:.3f} total_latency={total:.3f} model={self.effective_model} "
            f"device={self.effective_device} compute={self.effective_compute_type} "
            f"text_length={len(text)} detected_language={detected_language} decision={final_decision}",
            flush=True,
        )

    def retry_last_command_transcript(self, *, language: str | None = None) -> dict:
        if self.last_audio_np is None or not self.last_speech_detected:
            return {
                "text": "",
                "speech_detected": bool(self.last_speech_detected),
                "language": language or self.cfg.command_language,
                "task": "transcribe",
                "attempted": False,
                "reason": "no_recent_speech_audio",
            }
        text, metadata = self._transcribe_audio(
            self.last_audio_np,
            language=language or self.cfg.command_language,
            mode=STTMode.DIRECT_COMMAND,
            force_prompt=True,
        )
        return {
            "text": text,
            "speech_detected": True,
            "attempted": True,
            **metadata,
        }

    def set_tts_playback(self, active: bool, text: str = "") -> None:
        self._tts_active = bool(active)
        if text:
            self._recent_tts_text = str(text)
        if not active:
            self._tts_ended_at = time.time()

    def _tts_echo_decision(self, text: str) -> tuple[bool, float, bool]:
        tail_active = bool(self._tts_ended_at and (time.time() - self._tts_ended_at) * 1000 < self.cfg.tts_echo_tail_ms)
        overlap = SequenceMatcher(None, self._normalize_guard_text(text), self._normalize_guard_text(self._recent_tts_text)).ratio() if text and self._recent_tts_text else 0.0
        reject = self._tts_active or (tail_active and overlap >= 0.72) or overlap >= 0.9
        print(
            "[HEBE][STT_TTS_ECHO_GUARD] "
            f"tts_active={str(self._tts_active).lower()} recent_tts_overlap={overlap:.3f} "
            f"tail_active={str(tail_active).lower()} decision={'reject' if reject else 'allow'}",
            flush=True,
        )
        return reject, overlap, tail_active

    @staticmethod
    def _normalize_guard_text(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))

    def _hallucination_reason(self, text: str, audio_np: np.ndarray | None = None) -> str:
        tokens = re.findall(r"[a-z0-9]+", self._normalize_guard_text(text))
        if len(tokens) < 5:
            return ""
        counts = {token: tokens.count(token) for token in set(tokens)}
        dominant = max(counts.values(), default=0)
        unique_ratio = len(counts) / max(1, len(tokens))
        if dominant >= 8 and dominant / len(tokens) >= 0.45:
            return "extreme_repeated_token_loop"
        ngrams = [tuple(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
        if ngrams and max(ngrams.count(item) for item in set(ngrams)) >= 6:
            return "repeated_phrase_loop"
        if len(tokens) >= 20 and unique_ratio < 0.2:
            return "absurd_repetition_ratio"
        if audio_np is not None and len(audio_np):
            rms = float(np.sqrt(np.mean(np.square(audio_np))))
            duration = len(audio_np) / max(1, self.cfg.rate)
            if rms < self.cfg.silence_rms_threshold and len(tokens) > max(8, int(duration * 5)):
                return "long_transcript_low_energy_audio"
        return ""

    def _looks_like_command_candidate(self, text: str) -> bool:
        raw = str(text or "").strip()
        direct = parse_direct_stt_command(raw)
        if direct.wake_detected:
            return True
        if (
            direct.detected_intent_family == DirectUtteranceIntentFamily.APPLICATION_ACTION.value
            and direct.action_verb and direct.raw_target
        ):
            return True
        normalized = self._normalize_guard_text(raw)
        return bool(re.match(
            r"^\s*(?:haz(?:le)?|dale|tira|pon)\s+(?:una?\s+)?"
            r"(?:promo|promocion|shoutout|s\s*o)\b",
            normalized,
        ))

    def _build_command_hypothesis(self, ambient_text: str, command_text: str, ambient_meta: dict, command_meta: dict) -> STTCommandHypothesis:
        ambient = self._normalize_guard_text(ambient_text)
        command = self._normalize_guard_text(command_text)
        agreement = SequenceMatcher(None, ambient, command).ratio() if ambient and command else 0.0
        direct = parse_direct_stt_command(
            command_text,
            ambient_text=ambient_text,
            agreement_score=agreement,
            event_id=f"stt_{uuid.uuid4().hex}",
        )
        wake = direct.wake_detected
        family = direct.detected_intent_family
        structure = 1.0 if family == DirectUtteranceIntentFamily.APPLICATION_ACTION.value else 0.55 if family == DirectUtteranceIntentFamily.INCOMPLETE_COMMAND.value else 0.35 if wake else 0.0
        confidence = min(1.0, 0.45 * structure + 0.3 * agreement + (0.2 if wake else 0.0) + (0.1 if direct.raw_target else 0.0))
        eligible = bool(
            family == DirectUtteranceIntentFamily.APPLICATION_ACTION.value
            and direct.action_verb and direct.raw_target
            and command_meta.get("language_allowed", True)
        )
        decision = (
            "execute" if eligible
            else "clarify" if family == DirectUtteranceIntentFamily.INCOMPLETE_COMMAND.value
            else "conversation" if family in {
                DirectUtteranceIntentFamily.DIRECT_QUESTION.value,
                DirectUtteranceIntentFamily.CASUAL_CONVERSATION.value,
            }
            else "route" if family in {
                DirectUtteranceIntentFamily.STREAM_OPERATION.value,
                DirectUtteranceIntentFamily.SYSTEM_COMMAND.value,
            }
            else "reject"
        )
        targets = tuple(direct.target_candidates)
        result = STTCommandHypothesis(
            ambient_text=ambient_text, command_text=command_text,
            ambient_language=ambient_meta.get("detected_language"), command_language=command_meta.get("detected_language"),
            wake_detected=wake, wake_score=1.0 if wake else 0.0,
            action_structure_score=structure, hypothesis_agreement=agreement,
            target_candidates=targets,
            final_command_text=command_text, command_confidence=confidence,
            action_eligible=eligible, decision=decision,
        )
        direct.wake_confidence = result.wake_score
        direct.final_outcome = "pending_routing"
        direct.rejection_reason = "" if decision != "reject" else "uncertain_direct_utterance"
        self.last_direct_stt_result = direct.to_dict()
        self.last_command_redecode = result.as_dict()
        self.last_command_confidence = confidence
        self.last_wake_decision = "detected" if wake else "not_detected"
        raw_target = direct.raw_target
        target_confidence = confidence if raw_target else 0.0
        self.last_application_target_resolution = {
            "verb": direct.action_verb, "raw_target": raw_target,
            "resolved_app": "",
            "confidence": target_confidence,
            "decision": decision,
        }
        print(
            "[HEBE][APP_COMMAND_RESOLVE] "
            f"verb={direct.action_verb} raw_target={raw_target!r} "
            "resolved_app= "
            f"confidence={target_confidence:.3f} decision={self.last_application_target_resolution['decision']}",
            flush=True,
        )
        print(
            "[HEBE][STT_COMMAND_REDECODE] "
            f"ambient={ambient_text!r} command={command_text!r} agreement={agreement:.3f} "
            f"wake={str(wake).lower()} target={','.join(result.target_candidates)} "
            f"confidence={confidence:.3f} decision={decision}", flush=True,
        )
        if decision == "reject":
            print(
                "[HEBE][DIRECT_STT_OUTCOME] "
                f"event_id={direct.event_id} intent_family={family} "
                "outcome=rejected reason=uncertain_direct_utterance",
                flush=True,
            )
        return result

    def _recover_prompt_echo_command(
        self,
        *,
        audio_np: np.ndarray,
        ambient_text: str,
        ambient_meta: dict,
        command_text: str,
        command_meta: dict,
        prompt_profile: str,
    ) -> tuple[str, dict, dict]:
        second_pass_invalid = bool(
            is_stt_prompt_hotword_list(command_text)
            or is_stt_prompt_injection(
                command_text,
                command_prompt=command_meta.get("options", {}).get("initial_prompt"),
            )
        )
        retry_text = ""
        ambient_fallback_used = False
        if second_pass_invalid:
            retry_text, retry_meta = self._transcribe_audio(
                audio_np,
                language=self.cfg.command_language,
                mode=STTMode.DIRECT_COMMAND,
                force_prompt=False,
                prompt_profile=prompt_profile,
                final_decision="prompt_echo_retry",
            )
            retry_invalid = bool(
                is_stt_prompt_hotword_list(retry_text)
                or is_stt_prompt_injection(retry_text, command_prompt="")
            )
            if retry_text and not retry_invalid:
                command_text, command_meta = retry_text, retry_meta
            else:
                ambient_direct = parse_direct_stt_command(ambient_text, ambient_text=ambient_text)
                strong_ambient = ambient_direct.detected_intent_family in {
                    DirectUtteranceIntentFamily.APPLICATION_ACTION.value,
                    DirectUtteranceIntentFamily.DIRECT_QUESTION.value,
                    DirectUtteranceIntentFamily.CASUAL_CONVERSATION.value,
                    DirectUtteranceIntentFamily.SYSTEM_COMMAND.value,
                    DirectUtteranceIntentFamily.STREAM_OPERATION.value,
                }
                if strong_ambient:
                    command_text = ambient_text
                    command_meta = dict(ambient_meta)
                    ambient_fallback_used = True
                else:
                    command_text = ""
            print(
                "[HEBE][STT_COMMAND_REDECODE_FALLBACK] "
                "second_pass_invalid=true reason=prompt_echo "
                f"retry_text={retry_text!r} ambient_fallback_used={str(ambient_fallback_used).lower()} "
                f"selected_text={command_text!r}",
                flush=True,
            )
        return command_text, command_meta, {
            "second_pass_invalid": second_pass_invalid,
            "retry_text": retry_text,
            "ambient_fallback_used": ambient_fallback_used,
            "selected_text": command_text,
        }

    def health_snapshot(self) -> dict:
        values = sorted(self._latencies)
        def percentile(q: float) -> float:
            if not values:
                return 0.0
            idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
            return float(values[idx])
        cutoff = time.time() - 60
        return {
            "engine": "faster_whisper", "configured_model": self.cfg.model_size,
            "effective_model": self.effective_model, "device": self.effective_device,
            "compute_type": self.effective_compute_type, "cuda_available": self.cuda_available,
            "model_load_seconds": self.engine_load_seconds, "engine_status": self.engine_status,
            "engine_error": self.engine_error, "active_mode": self.active_mode.value,
            "input_device": self.get_selected_input_device(), "detected_language": self.last_detected_language,
            "detected_language_probability": self.last_detected_language_probability,
            "last_ambient_transcript": self.last_ambient_transcript,
            "last_command_redecode": self.last_command_redecode,
            "last_direct_stt_result": dict(self.last_direct_stt_result),
            "command_confidence": self.last_command_confidence, "last_wake_decision": self.last_wake_decision,
            "last_application_target_resolution": self.last_application_target_resolution,
            "last_rejection_reason": self.last_rejection_reason,
            "tts_echo_state": {"active": self._tts_active, "tail_active": bool(self._tts_ended_at and (time.time()-self._tts_ended_at)*1000 < self.cfg.tts_echo_tail_ms)},
            "last_latency_seconds": values[-1] if values else 0.0,
            "average_latency_seconds": statistics.fmean(values) if values else 0.0,
            "p50_latency_seconds": percentile(0.5), "p95_latency_seconds": percentile(0.95),
            "max_latency_seconds": max(values) if values else 0.0,
            "transcriptions_per_minute": sum(1 for ts in self._transcription_timestamps if ts >= cutoff),
            "rejected_transcripts": self.rejected_transcripts,
            "command_success_count": self.command_success_count, "command_failure_count": self.command_failure_count,
            "hallucination_reject_count": self.hallucination_reject_count,
            "prompt_echo_reject_count": self.prompt_echo_reject_count,
            "pre_roll_seconds": self.cfg.preroll_seconds, "last_performance": dict(self._perf_last),
            "gpu": dict(self.last_gpu_snapshot),
        }

    def disable_command_prompt_for_session(self) -> bool:
        if not self.cfg.command_prompt_enabled:
            return False
        self.cfg.command_prompt_enabled = False
        print("[HEBE][STT][PROMPT] auto_disabled reason=repeated_prompt_echo", flush=True)
        return True

    def _record_prompt_echo_rejection(self) -> None:
        if not self.cfg.auto_disable_prompt_on_echo:
            return
        now = time.time()
        window = max(1.0, float(self.cfg.prompt_echo_window_seconds or 300))
        self._prompt_echo_rejection_ts = [
            ts for ts in self._prompt_echo_rejection_ts
            if now - float(ts or 0.0) <= window
        ]
        self._prompt_echo_rejection_ts.append(now)
        if len(self._prompt_echo_rejection_ts) >= max(1, int(self.cfg.prompt_echo_disable_threshold or 2)):
            self.disable_command_prompt_for_session()

    def _publish_transcript_or_reject(self, text: str, metadata: dict | None = None) -> str:
        texto = str(text or "").strip()
        metadata = dict(metadata or {})
        audio_rejection = str(metadata.get("audio_rejection_reason") or "")
        if audio_rejection in {
            "empty_audio", "no_speech", "low_quality_audio",
            "unsupported_language", "bilingual_recovery_conflict",
        }:
            print(
                f"[HEBE][STT_REJECTED] reason={audio_rejection} level=debug",
                flush=True,
            )
            self.last_rejected_stt = {
                "raw_text": "",
                "status": "rejected",
                "reason": audio_rejection,
                "ts": time.time(),
            }
            if audio_rejection != "no_speech":
                self.rejected_transcripts += 1
            self.last_rejection_reason = audio_rejection
            self._emit("voice.command", {
                "raw_text": "",
                "normalized_text": "",
                "status": "rejected",
                "reason": audio_rejection,
                "final_decision": "rejected",
            })
            self._emit("status", {"stt": "listening", "last_rejected_stt": self.last_rejected_stt})
            self._emit("stt.partial", {"text": ""})
            return ""
        detected_language = str(metadata.get("detected_language") or "")
        recovery = dict(metadata.get("language_recovery") or {})
        language_rejected = bool(
            metadata.get("rejection_reason") == "unsupported_language_recovery_failed"
            or (
                detected_language
                and detected_language not in self.cfg.allowed_languages
                and not recovery.get("accepted")
            )
        )
        if language_rejected:
            reason = "unsupported_language_recovery_failed"
            print(
                "[HEBE][STT_REJECTED] "
                f"reason={reason} initial_language={recovery.get('initial_language') or detected_language}",
                flush=True,
            )
            self.last_rejected_stt = {
                "raw_text": "",
                "status": "rejected",
                "reason": reason,
                "initial_language": recovery.get("initial_language") or detected_language,
                "ts": time.time(),
            }
            self.rejected_transcripts += 1
            self.last_rejection_reason = reason
            self._emit("voice.command", {
                "raw_text": "",
                "normalized_text": "",
                "status": "rejected",
                "reason": reason,
                "initial_language": recovery.get("initial_language") or detected_language,
                "final_decision": "rejected",
            })
            self._emit("status", {"stt": "listening", "last_rejected_stt": self.last_rejected_stt})
            self._emit("stt.partial", {"text": ""})
            return ""
        hotword_list = is_stt_prompt_hotword_list(texto)
        configured_prompt_echo = bool(
            self._normalize_guard_text(texto)
            and self._normalize_guard_text(texto) == self._normalize_guard_text(self.cfg.command_prompt)
        )
        if hotword_list or configured_prompt_echo or is_stt_prompt_injection(texto, command_prompt=self.cfg.command_prompt):
            reason = "stt_prompt_echo_or_hotword_list" if hotword_list or configured_prompt_echo else "stt_prompt_injection"
            if hotword_list and self.cfg.log_rejected_raw:
                print(f"[HEBE][STT][RAW] text={ascii(texto)}", flush=True)
            print(f"[HEBE][STT][REJECTED] reason={reason}", flush=True)
            if reason == "stt_prompt_echo_or_hotword_list":
                self._record_prompt_echo_rejection()
            self.last_rejected_stt = {
                "raw_text": texto if hotword_list else "",
                "status": "rejected",
                "reason": reason,
                "ts": time.time(),
            }
            self.rejected_transcripts += 1
            self.prompt_echo_reject_count += 1
            self.last_rejection_reason = reason
            self._emit(
                "voice.command",
                {
                    "raw_text": texto if hotword_list else "",
                    "normalized_text": "",
                    "status": "rejected",
                    "reason": reason,
                    "retry_attempted": False,
                    "final_decision": "rejected",
                },
            )
            self._emit("status", {"stt": "listening", "last_rejected_stt": self.last_rejected_stt})
            self._emit("stt.partial", {"text": ""})
            return ""

        if self._is_blacklisted(texto):
            self._emit("status", {"stt": "listening"})
            self._emit("stt.partial", {"text": ""})
            return ""

        if texto:
            self._emit("stt.final", {"text": texto, **metadata})
        return texto

    def clear_device_error(self) -> dict:
        self.status = "idle"
        self.failed_input_signature = ""
        self.failed_input_device_id = ""
        self.failed_input_error = ""
        self.failed_input_ts = 0.0
        self.last_input_device_error = None
        self._open_fail_counts.clear()
        self._emit("status", {"stt": "idle", "stt_input_device": self.get_selected_input_device()})
        print("[HEBE][STT][DEVICE] error state cleared", flush=True)
        return self.get_selected_input_device()
    
    def _resolve_input_device(self, p: pyaudio.PyAudio) -> int:
        """
        Decide qué dispositivo usar:
        1. Si hay índice en config → usarlo
        2. Si no → usar el default del sistema
        """

        devices = _list_audio_devices_with_instance(p)
        default_device = next((d for d in devices if d.get("is_default_input")), None)
        if default_device and self.cfg.verbose_device_logs:
            print(f"[HEBE][STT][DEVICE] default input={default_device.get('display_label')}", flush=True)

        selected: dict | None = None
        reason = "default_input"

        if self.cfg.input_device_index is not None:
            selected = next((d for d in devices if int(d.get("index", -1)) == int(self.cfg.input_device_index)), None)
            if selected and self.cfg.input_device_name and str(selected.get("name") or "") != self.cfg.input_device_name:
                selected = None
            if selected and self.cfg.input_device_host_api and str(selected.get("host_api") or "") != self.cfg.input_device_host_api:
                selected = None
            if selected:
                reason = "exact_id"

        if selected is None and self.cfg.input_device_signature:
            selected = next((d for d in devices if str(d.get("signature") or "") == self.cfg.input_device_signature), None)
            if selected:
                reason = "signature"

        if selected is None and self.cfg.input_device_name and self.cfg.input_device_host_api:
            selected = next(
                (
                    d for d in devices
                    if str(d.get("name") or "") == self.cfg.input_device_name
                    and str(d.get("host_api") or "") == self.cfg.input_device_host_api
                ),
                None,
            )
            if selected:
                reason = "name_host_api"

        if selected is None and self.cfg.input_device_name:
            candidates = [d for d in devices if str(d.get("name") or "") == self.cfg.input_device_name]
            selected = _sort_by_host_api_preference(candidates)[0] if candidates else None
            if selected:
                reason = "name_only"

        if selected is None:
            selected = default_device
            reason = "default_input"

        if selected is None:
            raise RuntimeError("No input audio device available")

        device_index = int(selected["index"])
        self._remember_selected_device(selected)
        self.last_input_device_error = None

        if self.cfg.verbose_device_logs:
            print(
                f"[HEBE][STT][DEVICE] selected input={selected.get('display_label')} "
                f"reason={reason}",
                flush=True,
            )
        self._log_input_device_diagnostic(selected, default_device=default_device)

        return device_index

    def _remember_selected_device(self, device: dict) -> None:
        self.selected_input_device_id = str(device.get("id") or device.get("index") or "")
        self.selected_input_device_name = str(device.get("name") or "")
        self.selected_input_host_api = str(device.get("host_api") or "")
        self.selected_input_signature = str(device.get("signature") or "")
        self.selected_input_sample_rate = int(device.get("default_sample_rate") or device.get("sample_rate") or self.cfg.rate)
        self.selected_input_channels = max(1, min(int(device.get("max_input_channels") or device.get("channels") or self.cfg.channels), 2))

    def set_input_device(
        self,
        device_id: str | int | None = None,
        device_name: str | None = None,
        host_api: str | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        signature: str | None = None,
    ) -> dict:
        device_index: int | None = None
        if device_id not in (None, ""):
            try:
                device_index = int(str(device_id))
            except ValueError as exc:
                self.last_input_device_error = f"Invalid STT input device id: {device_id}"
                raise ValueError(self.last_input_device_error) from exc

        self.cfg.input_device_index = device_index
        self.cfg.input_device_name = device_name or None
        self.cfg.input_device_host_api = host_api or None
        self.cfg.input_device_sample_rate = int(sample_rate) if sample_rate else None
        self.cfg.input_device_channels = int(channels) if channels else None
        self.cfg.input_device_signature = signature or None
        self.selected_input_device_id = str(device_index) if device_index is not None else ""
        self.selected_input_device_name = device_name or ""
        self.selected_input_host_api = host_api or ""
        self.selected_input_sample_rate = int(sample_rate) if sample_rate else self.cfg.rate
        self.selected_input_channels = int(channels) if channels else self.cfg.channels
        self.selected_input_signature = signature or ""
        self.clear_device_error()
        self.last_input_device_error = None
        print(
            f"[HEBE][STT][DEVICE] selected id={self.selected_input_device_id or '(default)'} "
            f"name={self.selected_input_device_name or '(default)'} "
            f"host_api={self.selected_input_host_api or '(unknown)'} "
            f"sample_rate={self.selected_input_sample_rate} channels={self.selected_input_channels}",
            flush=True,
        )
        self._log_input_device_diagnostic(self.get_selected_input_device())
        self._emit("status", {"stt_input_device": self.get_selected_input_device()})
        return self.get_selected_input_device()

    def _log_input_device_diagnostic(self, device: dict | None, *, default_device: dict | None = None) -> None:
        device = device or {}
        name = str(device.get("name") or device.get("device_name") or device.get("display_label") or "").strip()
        default_name = str((default_device or {}).get("name") or (default_device or {}).get("device_name") or (default_device or {}).get("display_label") or "").strip()
        selected_label = str(device.get("display_label") or name or "(default)").strip()
        default_label = str((default_device or {}).get("display_label") or default_name or "(unknown)").strip()
        actual_capture = name or selected_label or "(default)"
        lowered = actual_capture.lower()
        warning = ""
        if any(marker in lowered for marker in ("out ", " output", "voicemeeter out", "desktop", "stereo mix", "what u hear", "loopback", "cable output", "bus")):
            warning = "possible_output_mix"
        print(
            f"[HEBE][STT_DEVICE_DIAGNOSTIC] selected={name or '(default)'} warning={warning or 'none'}",
            flush=True,
        )
        print(
            "[HEBE][STT_DEVICE_ACTIVE] "
            f"selected={selected_label} default={default_label} actual_capture={actual_capture} "
            f"warning={warning or 'none'}",
            flush=True,
        )
        print(
            f"[HEBE][STT_AUDIO_SOURCE] source={actual_capture}",
            flush=True,
        )

    def get_selected_input_device(self) -> dict:
        return {
            "device_id": self.selected_input_device_id,
            "device_name": self.selected_input_device_name,
            "host_api": self.selected_input_host_api,
            "signature": self.selected_input_signature,
            "sample_rate": self.selected_input_sample_rate,
            "channels": self.selected_input_channels,
            "last_level": self.last_input_level,
            "last_rms": self.last_input_rms,
            "last_peak": self.last_input_peak,
            "error": self.last_input_device_error,
            "status": self.status,
            "failed_device_id": self.failed_input_device_id,
            "failed_signature": self.failed_input_signature,
            "failed_error": self.failed_input_error,
        }

    def _open_input_stream(self, p: pyaudio.PyAudio, device_index: int):
        device = p.get_device_info_by_index(device_index)
        default_rate = int(float(device.get("defaultSampleRate") or self.selected_input_sample_rate or self.cfg.rate))
        channels = max(1, min(int(device.get("maxInputChannels") or self.cfg.channels), self.cfg.channels or 1))
        attempts: list[tuple[int, int]] = []
        for rate, ch in [
            (self.cfg.rate, self.cfg.channels),
            (default_rate, 1),
            (48000, 1),
            (44100, 1),
            (16000, 1),
        ]:
            rate = int(rate)
            ch = max(1, min(int(ch), max(1, int(device.get("maxInputChannels") or ch))))
            if (rate, ch) not in attempts:
                attempts.append((rate, ch))

        last_exc: Exception | None = None
        for rate, ch in attempts:
            try:
                if self.cfg.verbose_device_logs:
                    print(
                        f"[HEBE][STT][DEVICE] opening input index={device_index} "
                        f"name={device.get('name')} rate={rate} channels={ch} block={self.cfg.chunk} "
                        f"engine=whisper/{self.cfg.model_size}",
                        flush=True,
                    )
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=ch,
                    rate=rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.cfg.chunk,
                )
                if self.cfg.verbose_device_logs:
                    print("[HEBE][STT][DEVICE] input stream opened successfully", flush=True)
                self.selected_input_sample_rate = rate
                self.selected_input_channels = ch
                return stream, rate, ch
            except Exception as exc:
                last_exc = exc
                print(
                    f"[HEBE][STT][ERROR] open input failed index={device_index} rate={rate} channels={ch}: {exc!r}",
                    flush=True,
                )
        assert last_exc is not None
        device_record = {
            "id": str(device_index),
            "index": device_index,
            "name": str(device.get("name") or self.selected_input_device_name or ""),
            "host_api": self.selected_input_host_api,
            "display_label": f"{device.get('name') or self.selected_input_device_name} — {self.selected_input_host_api or 'API ?'} — id {device_index}",
            "signature": self.selected_input_signature or _device_signature(
                str(device.get("name") or ""),
                self.selected_input_host_api,
                int(float(device.get("defaultSampleRate") or 0)),
                int(device.get("maxInputChannels") or 0),
            ),
        }
        self._mark_device_open_failed(device_record, len(attempts), last_exc)
        raise STTDeviceOpenFailure(self.failed_input_error, device=device_record, attempts=len(attempts)) from last_exc

    def _mark_device_open_failed(self, device: dict, attempts: int, exc: Exception) -> None:
        key = str(device.get("signature") or device.get("id") or "")
        self._open_fail_counts[key] = self._open_fail_counts.get(key, 0) + 1
        self.failed_input_signature = key
        self.failed_input_device_id = str(device.get("id") or self.selected_input_device_id)
        self.failed_input_ts = time.time()
        self.failed_input_error = f"{type(exc).__name__}: {exc}"
        self.last_input_device_error = self.failed_input_error
        self.status = "error"
        print(
            "[HEBE][STT][ERROR] selected input failed after "
            f"{attempts} sample-rate attempts device={device.get('name')!r} "
            f"host_api={device.get('host_api')!r} id={device.get('id')} "
            f"error={self.failed_input_error!r}",
            flush=True,
        )
        self._emit("status", {
            "stt": "error",
            "last_stt_error": self.failed_input_error,
            "stt_input_device": self.get_selected_input_device(),
        })
        if self.cfg.disable_on_device_open_failure and self._open_fail_counts[key] >= max(1, self.cfg.max_device_open_retries):
            print("[HEBE][STT] paused due to device_open_failure; waiting for user action", flush=True)

    def test_input_device(self, seconds: float = 4.0) -> dict:
        return test_audio_input_device(
            device_id=self.selected_input_device_id,
            device_name=self.selected_input_device_name,
            host_api=self.selected_input_host_api,
            seconds=seconds,
        )

    def listen(self) -> str:
        """
        Graba hasta detectar voz y silencio final, transcribe con Whisper.
        Devuelve texto normalizado (puede ser "").
        """
        if self.status == "error" and self.cfg.disable_on_device_open_failure:
            raise STTDeviceOpenFailure(
                self.failed_input_error or "STT input device is blocked after open failure",
                device=self.get_selected_input_device(),
                attempts=0,
            )
        self.init()
        assert self._model is not None

        p = pyaudio.PyAudio()
        device_index = self._resolve_input_device(p)

        try:
            stream, opened_rate, opened_channels = self._open_input_stream(p, device_index)
        except STTDeviceOpenFailure:
            p.terminate()
            raise

        self.status = "listening"
        self._emit("status", {"stt": "listening"})
        frames: list[bytes] = []
        preroll_frames = max(0, min(int(round(self.cfg.preroll_seconds * opened_rate / self.cfg.chunk)), int(round(2.0 * opened_rate / self.cfg.chunk))))
        preroll: deque[bytes] = deque(maxlen=preroll_frames)
        recording = False
        silence_frames = 0
        start_time = time.time()
        tick = 0

        try:
            while True:
                data = stream.read(self.cfg.chunk, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                peak = float(np.max(np.abs(audio_chunk))) if len(audio_chunk) > 0 else 0.0
                rms = float(np.sqrt(np.mean(np.square(audio_chunk)))) if len(audio_chunk) > 0 else 0.0
                self.last_input_peak = peak
                self.last_input_rms = rms
                self.last_input_level = peak
                tick += 1

                if tick % 10 == 0:
                    now = time.time()
                    if (
                        rms <= self.cfg.silence_rms_threshold
                        and now - start_time >= self.cfg.silence_warning_after_seconds
                        and now - self._last_silence_warning_ts >= self.cfg.silence_warning_rate_limit_seconds
                    ):
                        self._last_silence_warning_ts = now
                        print(
                            f"[HEBE][STT][WARN] selected input opened but signal is silent rms={rms:.4f} peak={peak:.4f}",
                            flush=True,
                        )
                    self._emit("stt.partial", {
                        "text": f"rms {rms:.3f} peak {peak:.3f}",
                        "level": peak,
                        "rms": rms,
                        "peak": peak,
                    })

                if not recording:
                    if preroll_frames:
                        preroll.append(data)
                    if peak > self.cfg.silence_threshold:
                        recording = True
                        frames.extend(preroll)
                        if not preroll or preroll[-1] is not data:
                            frames.append(data)
                        self._speech_start_offset = max(0.0, (len(frames) - 1) * self.cfg.chunk / opened_rate)
                        start_time = time.time()
                        silence_frames = 0
                        self._emit("status", {"stt": "recording"})
                else:
                    frames.append(data)
                    if peak < self.cfg.silence_threshold:
                        silence_frames += 1
                    else:
                        silence_frames = 0

                    elapsed = len(frames) * (self.cfg.chunk / self.cfg.rate)

                    if (elapsed >= self.cfg.min_record_seconds and silence_frames >= self._silence_frames_needed) or elapsed >= self.cfg.max_record_seconds:
                        break

                    if time.time() - start_time > self.cfg.max_record_seconds + 2:
                        break

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        if not frames:
            self._emit("status", {"stt": "listening"})
            self._emit("stt.partial", {"text": ""})
            return ""

        self._emit("status", {"stt": "transcribing"})

        audio_bytes = b"".join(frames)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        max_abs = float(np.max(np.abs(audio_np))) if len(audio_np) > 0 else 0.0
        self.last_audio_np = audio_np
        self.last_speech_detected = bool(max_abs >= self.cfg.silence_threshold)

        if max_abs < self.cfg.silence_threshold:
            reason = self.classify_empty_transcript(
                metadata={}, audio_np=audio_np, speech_detected=False,
            )
            self.last_result_metadata = {"audio_rejection_reason": reason, "speech_detected": False}
            self._publish_transcript_or_reject("", self.last_result_metadata)
            self._emit("status", {"stt": "listening"})
            self._emit("stt.partial", {"text": ""})
            return ""

        duration = len(frames) * self.cfg.chunk / max(1, opened_rate)
        endpoint_reason = "max_duration" if duration >= self.cfg.max_record_seconds else "silence"
        print(
            "[HEBE][STT_SEGMENT] "
            f"duration={duration:.3f} preroll={min(self.cfg.preroll_seconds, self._speech_start_offset):.3f} "
            f"speech_start={self._speech_start_offset:.3f} endpoint_reason={endpoint_reason}", flush=True,
        )

        texto, metadata = self._transcribe_audio(audio_np, mode=STTMode.AMBIENT_DISCOURSE, force_prompt=False)
        self.last_ambient_transcript = texto
        initial_language = str(metadata.get("detected_language") or "")
        short_audio = duration < self.cfg.short_audio_language_threshold_seconds
        if not str(texto or "").strip():
            reason = self.classify_empty_transcript(
                metadata=metadata,
                audio_np=audio_np,
                speech_detected=self.last_speech_detected,
            )
            metadata = {**metadata, "audio_rejection_reason": reason, "action_eligible": False, "command_mode": False}
            self.last_result_metadata = metadata
            self._publish_transcript_or_reject("", metadata)
            self._emit("status", {"stt": "listening"})
            self._emit("stt.partial", {"text": ""})
            return ""
        if (
            initial_language not in self.cfg.allowed_languages
            or short_audio
        ):
            recovery = self._dual_decode_language_recovery(
                audio_np,
                initial_language=initial_language,
                initial_text=texto,
                short_audio=short_audio,
            )
            if recovery.get("accepted"):
                texto = str(recovery.get("selected_text") or "")
                metadata = dict(recovery.get("metadata") or {})
            else:
                metadata = dict(recovery.get("metadata") or {})
                metadata["audio_rejection_reason"] = str(
                    metadata.get("rejection_class") or "unsupported_language"
                )
                texto = ""
                self.last_result_metadata = metadata
                self._publish_transcript_or_reject("", metadata)
                self._emit("status", {"stt": "listening"})
                self._emit("stt.partial", {"text": ""})
                return ""
        reason = self._hallucination_reason(texto, audio_np)
        if reason:
            retry_text, retry_meta = self._transcribe_audio(
                audio_np, mode=STTMode.DIAGNOSTIC, language=None, force_prompt=False,
                final_decision="hallucination_retry",
            )
            retry_reason = self._hallucination_reason(retry_text, audio_np)
            if retry_reason:
                self.hallucination_reject_count += 1
                self.rejected_transcripts += 1
                self.last_rejection_reason = retry_reason
                print(f"[HEBE][STT][REJECTED] reason={retry_reason}", flush=True)
                texto = ""
            else:
                texto, metadata = retry_text, retry_meta

        if texto:
            echo_reject, _, _ = self._tts_echo_decision(texto)
            if echo_reject:
                self.rejected_transcripts += 1
                self.last_rejection_reason = "tts_echo"
                texto = ""

        if texto and self._looks_like_command_candidate(texto):
            self.active_mode = STTMode.COMMAND_CANDIDATE
            ambient_tokens = set(re.findall(r"[a-z0-9]+", self._normalize_guard_text(texto)))
            prompt_profile = (
                "promotion_command"
                if ambient_tokens & {"promo", "promociona", "shoutout"}
                else "stream_operation"
                if ambient_tokens & {"stream", "directo", "chat"}
                else "app_command"
            )
            command_text, command_meta = self._transcribe_audio(
                audio_np,
                mode=STTMode.DIRECT_COMMAND,
                force_prompt=True,
                prompt_profile=prompt_profile,
            )
            command_text, command_meta, fallback_debug = self._recover_prompt_echo_command(
                audio_np=audio_np,
                ambient_text=texto,
                ambient_meta=metadata,
                command_text=command_text,
                command_meta=command_meta,
                prompt_profile=prompt_profile,
            )
            hypothesis = self._build_command_hypothesis(texto, command_text, metadata, command_meta)
            # Partial commands may enter cognition only to request clarification;
            # only a structured direct decode is action eligible.
            texto = hypothesis.final_command_text if hypothesis.decision in {
                "execute", "clarify", "conversation", "route",
            } else ""
            metadata = {
                **command_meta, "command_hypothesis": hypothesis.as_dict(),
                "direct_stt_command": dict(self.last_direct_stt_result),
                "action_eligible": hypothesis.action_eligible,
                "command_mode": hypothesis.decision in {"execute", "clarify", "route"},
                "prompt_echo_fallback": {**fallback_debug, "selected_text": texto},
            }
            if hypothesis.decision == "execute":
                self.command_success_count += 1
            elif hypothesis.decision != "reject":
                self.command_failure_count += 1
        elif texto:
            metadata["action_eligible"] = False

        if texto and not metadata.get("language_allowed", True):
            metadata["action_eligible"] = False

        texto = self._publish_transcript_or_reject(texto, metadata)
        self.last_result_metadata = dict(metadata)

        self._emit("status", {"stt": "listening"})
        self._emit("stt.partial", {"text": ""})
        return texto


def _device_signature(name: str, host_api: str, sample_rate: int, max_input_channels: int) -> str:
    return f"{name}|{host_api}|{sample_rate}|{max_input_channels}".lower()


def _device_display_label(device: dict) -> str:
    return (
        f"{device.get('name') or '(sin nombre)'} — {device.get('host_api') or 'API ?'} — "
        f"id {device.get('id')} — {device.get('default_sample_rate') or 0}Hz — "
        f"{device.get('max_input_channels') or 0}ch"
    )


def _host_api_score(host_api: str) -> int:
    normalized = (host_api or "").lower()
    if "wasapi" in normalized:
        return 0
    if "mme" in normalized:
        return 1
    if "directsound" in normalized:
        return 2
    if "wdm" in normalized or "ks" in normalized:
        return 9
    return 4


def _sort_by_host_api_preference(devices: list[dict]) -> list[dict]:
    return sorted(devices, key=lambda d: (_host_api_score(str(d.get("host_api") or "")), int(d.get("index") or 0)))


def _list_audio_devices_with_instance(p: pyaudio.PyAudio) -> list[dict]:
    devices = []
    default_index = None
    try:
        default_info = p.get_default_input_device_info()
        default_index = int(default_info.get("index"))
        print(
            f"[HEBE][STT][DEVICE] default input={default_info.get('name')} id={default_index}",
            flush=True,
        )
    except Exception:
        default_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        max_input = int(info.get("maxInputChannels") or 0)
        if max_input <= 0:
            continue
        max_output = int(info.get("maxOutputChannels") or 0)
        host_api_name = ""
        try:
            host_api = p.get_host_api_info_by_index(int(info.get("hostApi", 0)))
            host_api_name = str(host_api.get("name") or "")
        except Exception:
            host_api_name = ""
        name = str(info.get("name") or "")
        sample_rate = int(float(info.get("defaultSampleRate") or 0))
        lower_name = name.lower()
        is_loopback = "loopback" in lower_name or ("wasapi" in host_api_name.lower() and max_input > 0 and max_output > 0 and "output" in lower_name)
        device = {
            "id": str(i),
            "index": i,
            "name": name,
            "host_api": host_api_name,
            "host_api_index": int(info.get("hostApi", 0)),
            "is_default": default_index == i,
            "is_default_input": default_index == i,
            "is_loopback": is_loopback,
            "channels": max_input,
            "sample_rate": sample_rate,
            "max_input_channels": max_input,
            "max_output_channels": max_output,
            "default_sample_rate": sample_rate,
            "signature": _device_signature(name, host_api_name, sample_rate, max_input),
            "host_api_warning": "Puede fallar en Windows; prueba WASAPI/MME si no hay señal." if _host_api_score(host_api_name) >= 9 else "",
        }
        device["display_label"] = _device_display_label(device)
        devices.append(device)
    return devices


def list_audio_devices() -> list[dict]:
    """
    Devuelve lista de dispositivos de entrada disponibles.
    """
    p = pyaudio.PyAudio()
    try:
        devices = _list_audio_devices_with_instance(p)
    finally:
        p.terminate()

    print(f"[HEBE][STT][DEVICE] available devices={len(devices)}", flush=True)
    return devices


def test_audio_input_device(
    *,
    device_id: str | int | None = None,
    device_name: str | None = None,
    host_api: str | None = None,
    seconds: float = 4.0,
) -> dict:
    p = pyaudio.PyAudio()
    stream = None
    try:
        devices = _list_audio_devices_with_instance(p)
        selected: dict | None = None
        if device_id not in (None, ""):
            selected = next((d for d in devices if str(d.get("id")) == str(device_id)), None)
        if selected is None and device_name and host_api:
            selected = next((d for d in devices if d.get("name") == device_name and d.get("host_api") == host_api), None)
        if selected is None and device_name:
            selected = next((d for d in devices if d.get("name") == device_name), None)
        if selected is None:
            selected = next((d for d in devices if d.get("is_default_input")), None)
        if selected is None:
            raise RuntimeError("No input audio device available")

        device_index = int(selected["index"])
        device_rate = int(selected.get("default_sample_rate") or 48000)
        max_channels = max(1, int(selected.get("max_input_channels") or 1))
        attempts: list[tuple[int, int]] = []
        for rate, channels in [
            (device_rate, 1),
            (48000, 1),
            (44100, 1),
            (16000, 1),
            (device_rate, min(max_channels, 2)),
        ]:
            candidate = (int(rate), max(1, min(int(channels), max_channels)))
            if candidate not in attempts:
                attempts.append(candidate)

        chunks: list[np.ndarray] = []
        opened = None
        last_exc: Exception | None = None
        chunk = 1024
        for rate, channels in attempts:
            try:
                print(
                    f"[HEBE][STT][DEVICE] raw test opening id={device_index} "
                    f"name={selected.get('name')} host_api={selected.get('host_api')} rate={rate} channels={channels}",
                    flush=True,
                )
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=chunk,
                )
                opened = {"sample_rate": rate, "channels": channels}
                break
            except Exception as exc:
                last_exc = exc
                print(f"[HEBE][STT][ERROR] raw test open failed rate={rate} channels={channels}: {exc!r}", flush=True)

        if stream is None or opened is None:
            assert last_exc is not None
            raise last_exc

        deadline = time.time() + max(0.5, min(float(seconds or 4.0), 8.0))
        while time.time() < deadline:
            data = stream.read(chunk, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio):
                chunks.append(audio)

        audio_np = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(audio_np)))) if len(audio_np) else 0.0
        peak = float(np.max(np.abs(audio_np))) if len(audio_np) else 0.0
        signal = bool(rms >= float(os.getenv("HEBE_STT_SILENCE_RMS_THRESHOLD", "0.003") or "0.003") or peak >= 0.02)
        result = {
            "ok": True,
            "signal_detected": signal,
            "rms": rms,
            "peak": peak,
            "device": selected,
            "sample_rate": opened["sample_rate"],
            "channels": opened["channels"],
            "seconds": seconds,
        }
        print(
            f"[HEBE][STT][DEVICE] raw test rms={rms:.5f} peak={peak:.5f} signal={signal} "
            f"device={selected.get('display_label')}",
            flush=True,
        )
        if not signal:
            print("[HEBE][STT][WARN] selected input opened but signal is silent", flush=True)
        return result
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        p.terminate()
