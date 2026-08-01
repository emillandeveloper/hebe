# backend/app/core/input_bus.py
import queue
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class VoiceTranscript:
    text: str
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

_UI_INBOX: "queue.Queue[str]" = queue.Queue()
_VOICE_INBOX: "queue.Queue[str]" = queue.Queue()


def submit_text_from_ui(text: str):
    if text is None:
        return
    _UI_INBOX.put(str(text))


def submit_text_from_voice(text: str, metadata: dict | None = None):
    if text is None:
        return
    _VOICE_INBOX.put(VoiceTranscript(str(text), dict(metadata or {})))


def get_ui_inbox() -> "queue.Queue[str]":
    return _UI_INBOX


def get_voice_inbox() -> "queue.Queue[str]":
    return _VOICE_INBOX
