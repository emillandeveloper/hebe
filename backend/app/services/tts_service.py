from __future__ import annotations

import os
import tempfile
from typing import Callable, Optional

from app.services.tts_worker import SynthesisReceipt, TTSSynthesisWorker


HEBE_TTS_MODE = os.getenv("HEBE_TTS_MODE", "auto").lower()  # auto | piper | xtts
HEBE_TTS_MIN_VRAM_GB = float(os.getenv("HEBE_TTS_MIN_VRAM_GB", "12"))
HEBE_TTS_SYNTHESIS_TIMEOUT_SECONDS = float(os.getenv("HEBE_TTS_SYNTHESIS_TIMEOUT_SECONDS", "15") or 15)
HEBE_TTS_WARMUP_TIMEOUT_SECONDS = float(os.getenv("HEBE_TTS_WARMUP_TIMEOUT_SECONDS", "60") or 60)
_synthesis_worker = TTSSynthesisWorker()


def _has_cuda_vram(min_gb: float) -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        return vram_gb >= float(min_gb)
    except Exception:
        return False


def pick_tts_backend() -> str:
    mode = (HEBE_TTS_MODE or "auto").lower()
    if mode in ("piper", "xtts"):
        return mode

    # auto
    if _has_cuda_vram(HEBE_TTS_MIN_VRAM_GB):
        return "xtts"

    # Si piper está configurado, úsalo. Si no, cae en xtts (CPU)
    # No hacemos checks heavy aquí: piper_to_wav ya valida.
    piper_exe = os.getenv("HEBE_PIPER_EXE", "")
    if piper_exe and os.path.exists(piper_exe):
        return "piper"

    return "xtts"


def speak(
    text: str,
    language: str = "es",
    emit: Optional[Callable[[str, dict], None]] = None,
    log_chat: Optional[Callable[[str, str, str], None]] = None,
) -> tuple[str, SynthesisReceipt]:
    """
    Genera un wav en audio_tmp/ y devuelve la ruta.
    El engine decide si reproducirlo o solo emitirlo a la UI.
    """
    if not text:
        return "", SynthesisReceipt(backend="none", latency_ms=0.0)

    if emit:
        emit("tts.start", {"text_length": len(text), "lang": language, "stage": "synthesis"})

    if log_chat:
        log_chat("assistant", text, source="tts")

    tmp_dir = os.path.join(os.getcwd(), "audio_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tmp_dir) as tmp:
        wav_path = tmp.name

    try:
        receipt = _synthesis_worker.synthesize(
            text=text,
            wav_path=wav_path,
            language=language,
            timeout_seconds=HEBE_TTS_SYNTHESIS_TIMEOUT_SECONDS,
        )
    except Exception:
        try:
            os.remove(wav_path)
        except OSError:
            pass
        raise

    if emit:
        emit("tts.end", {"stage": "synthesis", "latency_ms": receipt.latency_ms})

    return wav_path, receipt


def warmup(*, timeout_seconds: float | None = None) -> SynthesisReceipt:
    return _synthesis_worker.warmup(
        timeout_seconds=float(timeout_seconds or HEBE_TTS_WARMUP_TIMEOUT_SECONDS)
    )


def cancel() -> None:
    _synthesis_worker.cancel()


def shutdown(*, timeout_seconds: float = 1.0) -> None:
    _synthesis_worker.shutdown(timeout_seconds=timeout_seconds)
