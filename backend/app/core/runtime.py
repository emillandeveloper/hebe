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

from app.integrations.twitch.chat_client import TwitchChatClient
from app.integrations.twitch.chat_cache import TwitchChatCache
from app.integrations.twitch.event_memory import TwitchEventMemory
from app.integrations.twitch.target_resolver import TwitchTargetResolver
from app.integrations.twitch.service import TwitchService
from app.integrations.twitch.event_adapter import TwitchEventAdapter


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
    twitch: TwitchService
    twitch_events: TwitchEventAdapter


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

    # =========================
    # Twitch integration
    # =========================

    channel_name = "leonifelheim"
    bot_username = "HebeNifelheim"

    broadcaster_id = "124070929"
    sender_id = "1480877711"
    client_id = "gp762nuuoqcoxypju8c569th9wz7q5"
    oauth_token = "f945r0izxxbt2mrvkoo7zrmpuqv5l3"

    print("[HEBE][TWITCH] creating client...", flush=True)
    print("[HEBE][TWITCH] channel_name =", channel_name, flush=True)
    print("[HEBE][TWITCH] broadcaster_id =", broadcaster_id, flush=True)
    print("[HEBE][TWITCH] sender_id =", sender_id, flush=True)
    print("[HEBE][TWITCH] client_id loaded =", bool(client_id), flush=True)
    print("[HEBE][TWITCH] oauth_token loaded =", bool(oauth_token), flush=True)

    chat_cache = TwitchChatCache()
    event_memory = TwitchEventMemory()

    target_resolver = TwitchTargetResolver(
        chat_cache=chat_cache,
        event_memory=event_memory,
        aliases={},
    )

    chat_client = TwitchChatClient(
        channel_name=channel_name,
        broadcaster_id=broadcaster_id,
        sender_id=sender_id,
        client_id=client_id,
        oauth_token=oauth_token,
        bot_username=bot_username,
        enabled=True,
    )

    twitch = TwitchService(
        chat_client=chat_client,
        target_resolver=target_resolver,
        chat_cache=chat_cache,
        event_memory=event_memory,
        channel_name=channel_name,
        bot_username=bot_username,
    )

    twitch_events = TwitchEventAdapter(
        client_id=client_id,
        user_oauth_token=oauth_token,
        broadcaster_user_id=broadcaster_id,
        bot_user_id=sender_id,
        twitch_service=twitch,
        enabled=True,
    )

    return HebeRuntime(
        stt=stt,
        llm=llm,
        win=win,
        actions=actions,
        tools=tools,
        speak=speak,
        state=state,
        twitch=twitch,
        twitch_events=twitch_events,
    )