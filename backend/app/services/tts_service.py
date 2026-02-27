from __future__ import annotations

import os
import tempfile
from typing import Callable, Optional

from app.services.tts_piper import piper_to_wav
from app.services.tts_xtts import xtts_to_wav


HEBE_TTS_MODE = os.getenv("HEBE_TTS_MODE", "auto").lower()  # auto | piper | xtts
HEBE_TTS_MIN_VRAM_GB = float(os.getenv("HEBE_TTS_MIN_VRAM_GB", "12"))


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
) -> str:
    """
    Genera un wav en audio_tmp/ y devuelve la ruta.
    El engine decide si reproducirlo o solo emitirlo a la UI.
    """
    if not text:
        return ""

    if emit:
        emit("tts.start", {"text": text, "lang": language})

    if log_chat:
        log_chat("assistant", text, source="tts")

    tmp_dir = os.path.join(os.getcwd(), "audio_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tmp_dir) as tmp:
        wav_path = tmp.name

    backend = pick_tts_backend()
    print(f"🎛️ Backend TTS elegido: {backend}")

    try:
        if backend == "piper":
            piper_to_wav(text=text, wav_path=wav_path, language=language)
        else:
            xtts_to_wav(text=text, wav_path=wav_path, language=language)
    except Exception as e:
        # fallback: si piper falla, intenta xtts
        if backend == "piper":
            print(f"⚠️ Piper falló, fallback a XTTS: {e}")
            xtts_to_wav(text=text, wav_path=wav_path, language=language)
        else:
            raise

    if emit:
        emit("tts.end", {"path": wav_path})

    return wav_path
