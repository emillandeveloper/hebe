from __future__ import annotations

import os
import subprocess


HEBE_PIPER_EXE = os.getenv("HEBE_PIPER_EXE", "")
HEBE_PIPER_MODEL_ES = os.getenv("HEBE_PIPER_MODEL_ES", "")
HEBE_PIPER_MODEL_EN = os.getenv("HEBE_PIPER_MODEL_EN", "")
HEBE_PIPER_LENGTH_SCALE = float(os.getenv("HEBE_PIPER_LENGTH_SCALE", "1.0"))
HEBE_PIPER_SPEAKER = os.getenv("HEBE_PIPER_SPEAKER", "0").strip()


def piper_to_wav(text: str, wav_path: str, language: str = "es") -> str:
    exe = HEBE_PIPER_EXE
    if not exe or not os.path.exists(exe):
        raise FileNotFoundError("Piper no está configurado. Define HEBE_PIPER_EXE.")

    lang = (language or "es").lower()
    model = HEBE_PIPER_MODEL_ES if lang.startswith("es") else HEBE_PIPER_MODEL_EN

    if not model or not os.path.exists(model):
        raise FileNotFoundError("Modelo Piper no encontrado. Define HEBE_PIPER_MODEL_ES / HEBE_PIPER_MODEL_EN.")

    cmd = [exe, "-m", model, "-f", wav_path]
    if HEBE_PIPER_SPEAKER:
        cmd += ["-s", HEBE_PIPER_SPEAKER]


    # opcional: velocidad (si tu build la soporta)
    if HEBE_PIPER_LENGTH_SCALE and float(HEBE_PIPER_LENGTH_SCALE) != 1.0:
        cmd += ["--length_scale", str(HEBE_PIPER_LENGTH_SCALE)]

    subprocess.run(cmd, input=text.encode("utf-8"), check=True)
    return wav_path
