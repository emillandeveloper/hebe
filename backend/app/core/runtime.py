# backend/app/core/runtime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.core.ui_bridge import emit
from app.core.state import HebeState
from app.services.db_sqlite import log_chat
from app.services.interaction_actions import InteractionActions
from app.services.llm_ollama import OllamaLLM
from app.services.speech_output import speak as _speak
from app.services.stt_whisper import STTConfig, STTService
from app.services.tool_system import ToolContext, ToolSystem
from app.services.win_automation import WinAutomationService


def build_speak() -> Callable[[str, str], None]:
    def speak(text: str, language: str = "es") -> None:
        return _speak(
            text=text,
            language=language,
            emit=emit,
            log_chat=log_chat,
        )
    return speak


@dataclass(slots=True)
class HebeRuntime:
    stt: STTService
    llm: OllamaLLM
    win: WinAutomationService
    actions: InteractionActions
    tools: ToolSystem
    speak: Callable[[str, str], None]
    state: HebeState


def build_runtime() -> HebeRuntime:
    speak = build_speak()

    state = HebeState()

    stt = STTService(
        config=STTConfig(),
        emit=emit,
        log_chat=log_chat,
    )

    llm = OllamaLLM(
        model="hebe",
        emit=emit,
        log_chat=log_chat,
    )

    win = WinAutomationService(
        emit=emit,
        speak=speak,
    )

    actions = InteractionActions(
        speak=speak,
        stt=stt,
        win=win,
    )

    tools = ToolSystem(
        ToolContext(
            emit=emit,
            speak=speak,
            win=win,
            open_app_fn=actions.open_app_from_text,
            volume_fn=win.handle_volume_command,
            power_fn=None,
            memory_fn=actions.store_memory_from_text,
        )
    )

    return HebeRuntime(
        stt=stt,
        llm=llm,
        win=win,
        actions=actions,
        tools=tools,
        speak=speak,
        state=state,
    )