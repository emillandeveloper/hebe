from __future__ import annotations

import os
from typing import Optional

_xtts = None  # singleton interno


def _apply_torch_xtts_compat() -> None:
    """
    PyTorch 2.6 puede romper XTTS por el default weights_only=True
    y por safe globals (XttsConfig).
    """
    try:
        import torch

        _orig_load = torch.load

        def _load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _load_compat

        try:
            from torch.serialization import add_safe_globals
            from TTS.tts.configs.xtts_config import XttsConfig

            add_safe_globals([XttsConfig])
        except Exception as e:
            print(f"WARN: XTTS safe_globals not applied: {e}")

    except Exception as e:
        print(f"WARN: torch compat not applied: {e}")


def ensure_xtts_loaded() -> None:
    print("🔥 ensure_xtts_loaded() llamado")
    global _xtts
    if _xtts is not None:
        return

    _apply_torch_xtts_compat()

    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False

    from TTS.api import TTS  # import pesado aquí a propósito

    # Nota: puede tardar bastante si corre en CPU
    _xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
    print(f"✅ XTTS loaded (gpu={use_gpu})")
    print("✅ XTTS inicializado")


def xtts_to_wav(text: str, wav_path: str, language: str = "es") -> str:
    ensure_xtts_loaded()
    assert _xtts is not None

    speaker = os.getenv("HEBE_XTTS_SPEAKER", "Ana Florence").strip()
    speaker_wav = os.getenv("HEBE_XTTS_SPEAKER_WAV", "").strip()

    kwargs = dict(text=text, file_path=wav_path, language=language)

    # Si pasas un wav, XTTS suele aceptar speaker_wav (clonación)
    if speaker_wav:
        kwargs["speaker_wav"] = speaker_wav
    else:
        # Si no hay wav, usa un speaker name
        kwargs["speaker"] = speaker

    _xtts.tts_to_file(**kwargs)
    return wav_path

