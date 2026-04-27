# backend/app/core/runtime.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from app.core.ui_bridge import emit
from app.core.state import HebeState
from app.services.db_sqlite import log_chat
from app.services.interaction_actions import InteractionActions
from app.services.llm_ollama import OllamaLLM
from app.services.speech_output import speak as _speak
from app.services.stt_whisper import STTConfig, STTService
from app.services.tool_system import ToolContext, ToolSystem
from app.services.win_automation import WinAutomationService

from app.llm.ollama_intent_client import OllamaIntentClient

from app.integrations.twitch.chat_bot import TwitchChatBot
from app.integrations.twitch.chat_client import TwitchChatClient
from app.integrations.twitch.chat_cache import TwitchChatCache
from app.integrations.twitch.event_memory import TwitchEventMemory
from app.integrations.twitch.target_resolver import TwitchTargetResolver
from app.integrations.twitch.service import TwitchService
from app.integrations.twitch.event_adapter import TwitchEventAdapter


def build_speak() -> Callable[[str, str], None]:
    tts_enabled = os.getenv("HEBE_TTS_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")

    def speak(text: str, language: str = "es") -> None:
        if not tts_enabled:
            print(f"[HEBE][TTS] disabled, dropped speech: {text!r}", flush=True)
            return

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
    intent_llm: OllamaIntentClient
    win: WinAutomationService
    actions: InteractionActions
    tools: ToolSystem
    speak: Callable[[str, str], None]
    state: HebeState
    twitch: TwitchService
    twitch_events: TwitchEventAdapter
    twitch_chat_bot: TwitchChatBot
    stt_enabled: bool


def build_runtime() -> HebeRuntime:
    speak = build_speak()

    state = HebeState()

    stt = STTService(
        config=STTConfig(),
        emit=emit,
        log_chat=log_chat,
    )

    # Modelo conversacional: habla como Hebe
    llm = OllamaLLM(
        model=os.getenv("HEBE_CHAT_MODEL", "hebe"),
        emit=emit,
        log_chat=log_chat,
    )

    stt_enabled = os.getenv("HEBE_STT_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")

    # Modelo de intent/extracción estructurada: hebe-intent (qwen2.5:3b)
    intent_llm = OllamaIntentClient(
        model=os.getenv("HEBE_INTENT_MODEL", "hebe-intent"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
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
    # IMPORTANTE: las credenciales se leen de variables de entorno.
    # NO hardcodear oauth_token ni client_id en el código fuente.
    # Define en tu .env:
    #   TWITCH_CHANNEL_NAME=leonifelheim
    #   TWITCH_BOT_USERNAME=HebeNifelheim
    #   TWITCH_BROADCASTER_ID=...
    #   TWITCH_SENDER_ID=...
    #   TWITCH_CLIENT_ID=...
    #   TWITCH_OAUTH_TOKEN=...

    channel_name = os.getenv("TWITCH_CHANNEL_NAME", "")
    bot_username = os.getenv("TWITCH_BOT_USERNAME", "")
    broadcaster_id = os.getenv("TWITCH_BROADCASTER_ID", "")
    sender_id = os.getenv("TWITCH_SENDER_ID", "")
    client_id = os.getenv("TWITCH_CLIENT_ID", "")
    oauth_token = os.getenv("TWITCH_OAUTH_TOKEN", "")

    twitch_enabled = all([channel_name, broadcaster_id, sender_id, client_id, oauth_token])

    print("[HEBE][TWITCH] creating client...", flush=True)
    print("[HEBE][TWITCH] channel_name =", channel_name or "(missing)", flush=True)
    print("[HEBE][TWITCH] broadcaster_id =", broadcaster_id or "(missing)", flush=True)
    print("[HEBE][TWITCH] sender_id =", sender_id or "(missing)", flush=True)
    print("[HEBE][TWITCH] client_id loaded =", bool(client_id), flush=True)
    print("[HEBE][TWITCH] oauth_token loaded =", bool(oauth_token), flush=True)
    print("[HEBE][TWITCH] enabled =", twitch_enabled, flush=True)

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
        enabled=twitch_enabled,
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
        enabled=twitch_enabled,
    )

    twitch_chat_bot = TwitchChatBot(
        channel_name=channel_name,
        bot_username=bot_username,
        oauth_token=oauth_token,
        enabled=twitch_enabled,
    )

    return HebeRuntime(
        stt=stt,
        llm=llm,
        intent_llm=intent_llm,
        win=win,
        actions=actions,
        tools=tools,
        speak=speak,
        state=state,
        twitch=twitch,
        twitch_events=twitch_events,
        twitch_chat_bot=twitch_chat_bot,
        stt_enabled=stt_enabled,
    )