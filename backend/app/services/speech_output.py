from __future__ import annotations

import os
import time
from typing import Callable, Optional

import pygame

from app.services.tts_service import speak as tts_to_wav
from app.services.vts_client import vts_hotkey

HEBE_TTS_VOLUME = float(os.getenv("HEBE_TTS_VOLUME", "0.9"))


def speak(
    text: str,
    language: str = "es",
    emit: Optional[Callable[[str, dict], None]] = None,
    log_chat: Optional[Callable[[str, str, str], None]] = None,
) -> None:
    """
    High-level speech output:
    - Emits chat.assistant
    - Generates WAV via tts_service
    - Triggers VTS talking/idle
    - Plays WAV via pygame
    - Cleans up temp file
    """
    if not text:
        return

    if emit:
        emit("chat.assistant", {"text": text})

    audio_path = ""
    try:
        # Generate wav (this already emits tts.start/tts.end with the path)
        audio_path = tts_to_wav(text=text, language=language, emit=emit, log_chat=log_chat)

        # VTS "talking"
        try:
            vts_hotkey("HebeTalking")
        except Exception:
            pass

        # Playback
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.set_volume(float(HEBE_TTS_VOLUME))
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

    except Exception as e:
        print(f"❌ SpeechOutput error: {e}")

    finally:
        # VTS "idle"
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
        except OSError as e:
            print(f"⚠️ Could not delete temp audio: {e}")