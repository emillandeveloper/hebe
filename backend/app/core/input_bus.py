# backend/app/core/input_bus.py
import queue
from typing import Optional

_UI_INBOX: "queue.Queue[str]" = queue.Queue()
_VOICE_INBOX: "queue.Queue[str]" = queue.Queue()


def submit_text_from_ui(text: str):
    if text is None:
        return
    _UI_INBOX.put(str(text))


def submit_text_from_voice(text: str):
    if text is None:
        return
    _VOICE_INBOX.put(str(text))


def get_ui_inbox() -> "queue.Queue[str]":
    return _UI_INBOX


def get_voice_inbox() -> "queue.Queue[str]":
    return _VOICE_INBOX