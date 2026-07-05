# backend/app/services/stt_whisper.py
from __future__ import annotations

import os
import inspect
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pyaudio
from faster_whisper import WhisperModel


DEFAULT_COMMAND_PROMPT_WORDS = [
    "Hebe",
    "Ebe",
    "Eve",
    "Leo",
    "OBS",
    "Twitch",
    "stream",
    "chat",
    "promo",
    "shoutout",
    "SO",
    "Nuria",
    "Charlie",
    "Xarly",
    "Totodile",
    "Jotun",
    "Zwei",
    "Persona",
    "Final Fantasy",
]

_PROMPT_LOOP_WORDS = {
    "hebe",
    "ebe",
    "eve",
    "leo",
    "obs",
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

    model_size: str = os.getenv("HEBE_WHISPER_MODEL", "small")
    device: str = os.getenv("HEBE_WHISPER_DEVICE", "cpu")
    compute_type: str = os.getenv("HEBE_WHISPER_COMPUTE", "int8")
    allowed_languages: tuple[str, ...] = tuple(
        part.strip().lower()
        for part in os.getenv("HEBE_STT_ALLOWED_LANGUAGES", "es,en").split(",")
        if part.strip()
    ) or ("es", "en")
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

        self._silence_frames_needed = int(self.cfg.silence_end_seconds / (self.cfg.chunk / self.cfg.rate))

    def init(self) -> None:
        if self._model is None:
            self._model = WhisperModel(
                self.cfg.model_size,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
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
        command_mode: bool = True,
        force_prompt: bool = True,
    ) -> tuple[str, dict]:
        self.init()
        assert self._model is not None
        selected_language = language
        if selected_language is None and command_mode:
            selected_language = self._command_language()

        options = {
            "language": selected_language,
            "task": "transcribe",
            "beam_size": max(1, int(self.cfg.command_beam_size if command_mode else 5)),
            "vad_filter": True,
        }
        if command_mode:
            options["temperature"] = float(self.cfg.command_temperature)
            if force_prompt and self.cfg.command_prompt and self.cfg.command_prompt_enabled:
                options["initial_prompt"] = self.cfg.command_prompt
                options["hotwords"] = self.cfg.command_prompt

        try:
            supported = set(inspect.signature(self._model.transcribe).parameters)
            options = {key: value for key, value in options.items() if key in supported}
        except Exception:
            options.pop("hotwords", None)

        self.last_transcription_language = selected_language
        self.last_transcription_task = "transcribe"
        self.last_transcription_options = dict(options)
        print(
            "[HEBE][STT][TRANSCRIBE] "
            f"using initial_prompt={str(bool(options.get('initial_prompt'))).lower()} "
            f"language={selected_language!r} task='transcribe' command_mode={command_mode}",
            flush=True,
        )
        segments, info = self._model.transcribe(audio_np, **options)
        text = "".join(seg.text for seg in segments).strip()
        detected_language = getattr(info, "language", None)
        return text, {
            "language": selected_language,
            "detected_language": detected_language,
            "task": "transcribe",
            "command_mode": command_mode,
            "options": dict(options),
        }

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
            command_mode=True,
            force_prompt=True,
        )
        return {
            "text": text,
            "speech_detected": True,
            "attempted": True,
            **metadata,
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
        hotword_list = is_stt_prompt_hotword_list(texto)
        if hotword_list or is_stt_prompt_injection(texto, command_prompt=self.cfg.command_prompt):
            reason = "stt_prompt_echo_or_hotword_list" if hotword_list else "stt_prompt_injection"
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
            self._emit("stt.final", {"text": texto, **(metadata or {})})
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
        self._log_input_device_diagnostic(selected)

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

    def _log_input_device_diagnostic(self, device: dict | None) -> None:
        device = device or {}
        name = str(device.get("name") or device.get("device_name") or device.get("display_label") or "").strip()
        lowered = name.lower()
        warning = ""
        if any(marker in lowered for marker in ("out ", " output", "voicemeeter out", "desktop", "stereo mix", "what u hear", "loopback", "cable output", "bus")):
            warning = "possible_output_mix"
        print(
            f"[HEBE][STT_DEVICE_DIAGNOSTIC] selected={name or '(default)'} warning={warning or 'none'}",
            flush=True,
        )
        print(
            f"[HEBE][STT_AUDIO_SOURCE] source={name or '(default)'}",
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
                    if peak > self.cfg.silence_threshold:
                        recording = True
                        frames.append(data)
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
            self._emit("status", {"stt": "listening"})
            self._emit("stt.partial", {"text": ""})
            return ""

        texto, metadata = self._transcribe_audio(audio_np, command_mode=True, force_prompt=True)

        texto = self._publish_transcript_or_reject(texto, metadata)

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
