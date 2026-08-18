# backend/app/core/runtime.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from app.core.ui_bridge import emit
from app.core.state import HebeState
from app.core.persistent_logs import log_jsonl_event
from app.services.db_sqlite import get_setting, log_chat, set_setting
from app.services.llm_factory import create_conversation_llm
from app.services.speech_output import controller as speech_output_controller
from app.services.speech_output import speak as _speak
from app.services.stt_whisper import STTConfig, STTService
from app.services.win_automation import WinAutomationService

from app.llm.ollama_intent_client import OllamaIntentClient

from app.integrations.twitch.chat_bot import TwitchChatBot
from app.integrations.twitch.chat_client import TwitchChatClient
from app.integrations.twitch.chat_cache import TwitchChatCache
from app.integrations.twitch.event_memory import TwitchEventMemory
from app.integrations.twitch.target_resolver import TwitchTargetResolver
from app.integrations.twitch.service import TwitchService
from app.integrations.twitch.event_adapter import TwitchEventAdapter
from app.integrations.twitch.helix_client import TwitchHelixClient


def build_speak(state: HebeState, stt: STTService | None = None) -> Callable[..., None]:
    def speak(
        text: str,
        language: str = "es",
        *,
        emit_chat: bool = True,
        trace_id: str = "",
    ) -> dict | None:
        if not getattr(state, "tts_enabled", False):
            if emit_chat:
                emit("debug.tts_candidate", {"text": text, "response_stage": "generated"})
            print("[HEBE][TTS] skipped reason=global_disabled", flush=True)
            log_jsonl_event("tts", {
                "output_target": "local_tts",
                "tts_started": False,
                "tts_completed": False,
                "text_length": len(text),
                "reason": "global_disabled",
                "trace_id": trace_id,
            })
            return {"status": "tts_cancelled", "reason": "global_disabled", "trace_id": trace_id}

        print(
            f"[HEBE][TTS] speaking trace_id={trace_id or 'none'} text_length={len(text)}",
            flush=True,
        )
        log_jsonl_event("tts", {
            "output_target": "local_tts",
            "tts_started": True,
            "tts_completed": False,
            "text_length": len(text),
            "trace_id": trace_id,
        })
        def on_playback_state(active: bool) -> None:
            if stt is not None:
                stt.set_tts_playback(active, text)

        try:
            result = _speak(
                text=text,
                language=language,
                emit=emit,
                log_chat=log_chat,
                emit_chat=emit_chat,
                trace_id=trace_id,
                on_playback_state=on_playback_state,
            )
            log_jsonl_event("tts", {
                "output_target": "local_tts",
                "tts_started": True,
                "tts_completed": True,
                "text_length": len(text),
                "trace_id": trace_id,
                "latency_ms": float((result or {}).get("latency_ms") or 0.0),
            })
            return result
        except Exception as exc:
            safe_error = str(exc).replace('"', '\\"')
            print(f"[HEBE][TTS] failed error=\"{safe_error}\"", flush=True)
            log_jsonl_event("tts", {
                "output_target": "local_tts",
                "tts_started": True,
                "tts_completed": False,
                "text_length": len(text),
                "trace_id": trace_id,
                "error": safe_error,
            })
            raise
    return speak


@dataclass(slots=True)
class HebeRuntime:
    stt: STTService
    llm: Any
    intent_llm: OllamaIntentClient
    win: WinAutomationService
    speak: Callable[..., None]
    tts: Any
    state: HebeState
    twitch: TwitchService
    twitch_events: TwitchEventAdapter
    twitch_chat_bot: TwitchChatBot
    stt_enabled: bool


def build_runtime() -> HebeRuntime:
    state = HebeState()
    state.tts_enabled = os.getenv("HEBE_TTS_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")

    stt_config = STTConfig()
    persisted_device = get_setting("stt.input_device_id", os.getenv("HEBE_STT_INPUT_DEVICE", "") or "")
    persisted_device_name = get_setting("stt.input_device_name", os.getenv("HEBE_STT_INPUT_DEVICE_NAME", "") or "")
    persisted_host_api = get_setting("stt.input_device_host_api", os.getenv("HEBE_STT_INPUT_DEVICE_HOST_API", "") or "")
    persisted_sample_rate = get_setting("stt.input_device_sample_rate", os.getenv("HEBE_STT_INPUT_DEVICE_SAMPLE_RATE", "") or "")
    persisted_channels = get_setting("stt.input_device_channels", os.getenv("HEBE_STT_INPUT_DEVICE_CHANNELS", "") or "")
    persisted_signature = get_setting("stt.input_device_signature", os.getenv("HEBE_STT_INPUT_DEVICE_SIGNATURE", "") or "")
    if persisted_device and str(persisted_device).isdigit():
        stt_config.input_device_index = int(str(persisted_device))
    if persisted_device_name:
        stt_config.input_device_name = persisted_device_name
    if persisted_host_api:
        stt_config.input_device_host_api = persisted_host_api
    if str(persisted_sample_rate or "").isdigit():
        stt_config.input_device_sample_rate = int(str(persisted_sample_rate))
    if str(persisted_channels or "").isdigit():
        stt_config.input_device_channels = int(str(persisted_channels))
    if persisted_signature:
        stt_config.input_device_signature = persisted_signature

    stt = STTService(
        config=stt_config,
        emit=emit,
        log_chat=log_chat,
    )
    speak = build_speak(state, stt)
    if persisted_device or persisted_device_name:
        try:
            resolved_device = stt.set_input_device(
                device_id=persisted_device or "",
                device_name=persisted_device_name or "",
                host_api=persisted_host_api or "",
                sample_rate=int(str(persisted_sample_rate)) if str(persisted_sample_rate or "").isdigit() else None,
                channels=int(str(persisted_channels)) if str(persisted_channels or "").isdigit() else None,
                signature=persisted_signature or "",
            )
            for key, value in {
                "stt.input_device_id": resolved_device.get("device_id"),
                "stt.input_device_name": resolved_device.get("device_name"),
                "stt.input_device_host_api": resolved_device.get("host_api"),
                "stt.input_device_sample_rate": resolved_device.get("sample_rate"),
                "stt.input_device_channels": resolved_device.get("channels"),
                "stt.input_device_signature": resolved_device.get("signature"),
            }.items():
                set_setting(key, "" if value is None else str(value))
        except Exception as exc:
            print(f"[HEBE][STT][ERROR] persisted input device invalid: {exc!r}", flush=True)

    # Modelo conversacional: habla como Hebe.
    # Se elige por .env con HEBE_LLM_PROVIDER:
    #   - local / ollama -> OllamaLLM
    #   - openai         -> OpenAILLM, con fallback opcional a Ollama
    llm = create_conversation_llm(
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
    broadcaster_oauth_token = os.getenv("TWITCH_BROADCASTER_OAUTH_TOKEN", oauth_token)
    helix_client = TwitchHelixClient(
        client_id=client_id,
        oauth_token=broadcaster_oauth_token,
        broadcaster_id=broadcaster_id,
        channel_name=channel_name,
    )

    twitch = TwitchService(
        chat_client=chat_client,
        target_resolver=target_resolver,
        chat_cache=chat_cache,
        event_memory=event_memory,
        helix_client=helix_client,
        channel_name=channel_name,
        bot_username=bot_username,
    )
    twitch_events = TwitchEventAdapter(
        client_id=client_id,
        user_oauth_token=broadcaster_oauth_token,
        broadcaster_user_id=broadcaster_id,
        bot_user_id=sender_id,
        twitch_service=twitch,
        enabled=twitch_enabled,
        bot_username=bot_username,
        subscribe_chat_messages=os.getenv("HEBE_TWITCH_EVENTSUB_CHAT_MESSAGES", "false").strip().lower() in ("1", "true", "yes", "on"),
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
        speak=speak,
        tts=speech_output_controller,
        state=state,
        twitch=twitch,
        twitch_events=twitch_events,
        twitch_chat_bot=twitch_chat_bot,
        stt_enabled=stt_enabled,
    )
