from pathlib import Path
import sys
import os

# Este archivo está en backend/app/scripts/
# Necesitamos añadir backend/ al sys.path para poder importar app.*
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.tts_xtts import xtts_to_wav

speaker = sys.argv[1] if len(sys.argv) > 1 else "Ana Florence"

text = (
    "Hola Leo, soy Hebe. Si esta voz no te convence, "
    "la mandamos al abismo y buscamos otra."
)

out_dir = BACKEND_DIR / "audio_tmp"
out_dir.mkdir(exist_ok=True)

safe_name = (
    speaker.lower()
    .replace(" ", "_")
    .replace("á", "a")
    .replace("é", "e")
    .replace("í", "i")
    .replace("ó", "o")
    .replace("ú", "u")
)

out_path = out_dir / f"test_{safe_name}.wav"

os.environ["HEBE_XTTS_SPEAKER"] = speaker
os.environ["HEBE_XTTS_SPEAKER_WAV"] = ""

xtts_to_wav(
    text=text,
    wav_path=str(out_path),
    language="es",
)

print(f"Generado: {out_path}")