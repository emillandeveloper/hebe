# backend/app/core/runtime.py
from __future__ import annotations

from dataclasses import dataclass

from app.core.ui_bridge import emit
from app.services.speech_output import speak as _speak
from app.services.stt_whisper import STTService, STTConfig
from app.services.win_automation import WinAutomationService
from app.services.tool_system import ToolSystem, ToolContext
from app.services.llm_ollama import OllamaLLM
from app.services.intent_resolver import HybridIntentResolver, NLUContext
from app.services.dispatcher import Dispatcher, DispatchContext
from app.services.vts_client import vts_hotkey
from app.services.db_sqlite import log_chat
from app.services.interaction_actions import InteractionActions
from dataclasses import dataclass
from app.core.state import HebeState

def build_speak():
    def speak(text: str, language: str = "es") -> None:
        return _speak(text=text, language=language, emit=emit, log_chat=log_chat)
    return speak
@dataclass
class HebeRuntime:
    stt: STTService
    llm: OllamaLLM
    win: WinAutomationService
    actions: InteractionActions
    tools: ToolSystem
    nlu_ctx: NLUContext
    intent_resolver: HybridIntentResolver
    dispatcher: Dispatcher
    speak: callable
    state: HebeState

def build_runtime() -> HebeRuntime:
    speak = build_speak()

    stt = STTService(
        config=STTConfig(),
        emit=emit,
        log_chat=log_chat,
    )

    llm = OllamaLLM(model="hebe", emit=emit, log_chat=log_chat)

    win = WinAutomationService(emit=emit, speak=speak)

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

    nlu_ctx = NLUContext()

    intent_resolver = HybridIntentResolver(
        llm=llm,
        model_path="models/intent_gate.joblib",
        gate_threshold=0.60,
    )

    dispatcher = Dispatcher(
        DispatchContext(
            speak=speak,
            stt=stt,
            llm=llm,
            tools=tools,
            win=win,
            vts_hotkey=vts_hotkey,
            confirm_action=actions.confirm_action,
            store_memory_from_text=actions.store_memory_from_text,
        ),
        intent_resolver=intent_resolver,
        nlu_ctx=nlu_ctx,
    )
    
    state = HebeState()

    return HebeRuntime(
        stt=stt,
        llm=llm,
        win=win,
        actions=actions,
        tools=tools,
        nlu_ctx=nlu_ctx,
        intent_resolver=intent_resolver,
        dispatcher=dispatcher,
        speak=speak,
        state = state,
    )