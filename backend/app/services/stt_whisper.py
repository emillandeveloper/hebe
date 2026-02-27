# backend/app/services/stt_whisper.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pyaudio
from faster_whisper import WhisperModel


@dataclass
class STTConfig:
    rate: int = 16000
    channels: int = 1
    chunk: int = 1024
    input_device_index: int = int(os.getenv("HEBE_INPUT_DEVICE_INDEX", "9"))

    silence_threshold: float = 0.01
    max_record_seconds: float = 8.0
    min_record_seconds: float = 0.5
    silence_end_seconds: float = 0.8

    model_size: str = os.getenv("HEBE_WHISPER_MODEL", "small")
    device: str = os.getenv("HEBE_WHISPER_DEVICE", "cpu")
    compute_type: str = os.getenv("HEBE_WHISPER_COMPUTE", "int8")


DEFAULT_BLACKLIST = [
    "subtítulos por la comunidad de amara.org",
    "subtitulos por la comunidad de amara.org",
    "suscríbete",
    "suscribete",
]


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
        self._model: WhisperModel | None = None

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

    def listen(self) -> str:
        """
        Graba hasta detectar voz y silencio final, transcribe con Whisper.
        Devuelve texto normalizado (puede ser "").
        """
        self.init()
        assert self._model is not None

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.cfg.channels,
            rate=self.cfg.rate,
            input=True,
            input_device_index=self.cfg.input_device_index,
            frames_per_buffer=self.cfg.chunk,
        )

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
                level = float(np.max(np.abs(audio_chunk))) if len(audio_chunk) > 0 else 0.0
                tick += 1

                if tick % 10 == 0:
                    self._emit("stt.partial", {"text": f"lvl {level:.3f}"})

                if not recording:
                    if level > self.cfg.silence_threshold:
                        recording = True
                        frames.append(data)
                        start_time = time.time()
                        silence_frames = 0
                        self._emit("status", {"stt": "recording"})
                else:
                    frames.append(data)
                    if level < self.cfg.silence_threshold:
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

        if max_abs < self.cfg.silence_threshold:
            self._emit("status", {"stt": "listening"})
            self._emit("stt.partial", {"text": ""})
            return ""

        segments, _info = self._model.transcribe(
            audio_np,
            language=None,
            beam_size=5,
            vad_filter=True,
        )

        texto = "".join(seg.text for seg in segments).strip()

        if self._is_blacklisted(texto):
            self._emit("status", {"stt": "listening"})
            self._emit("stt.partial", {"text": ""})
            return ""

        if texto:
            self._emit("stt.final", {"text": texto})
            self._emit("chat.user", {"text": texto})
            if self.log_chat:
                self.log_chat("user", texto, source="voice")

        self._emit("status", {"stt": "listening"})
        self._emit("stt.partial", {"text": ""})
        return texto
def list_audio_devices() -> list[dict]:
    """
    Devuelve lista de dispositivos de entrada disponibles.
    """
    devices = []
    p = pyaudio.PyAudio()

    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append({
                    "index": i,
                    "name": info.get("name"),
                    "channels": info.get("maxInputChannels"),
                })
    finally:
        p.terminate()

    return devices
