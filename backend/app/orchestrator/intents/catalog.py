# backend/app/orchestrator/intents/catalog.py

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class IntentSpec:
    name: str
    required_slots: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


INTENTS: dict[str, IntentSpec] = {
    "open_app": IntentSpec(
        name="open_app",
        required_slots=["app_name"],
        keywords=["abre", "inicia", "ejecuta", "lanza", "open", "start", "run"],
    ),
    "close_window": IntentSpec(
        name="close_window",
        required_slots=[],
        keywords=["cierra", "cerrar", "close"],
    ),
    "set_volume": IntentSpec(
        name="set_volume",
        required_slots=["value"],
        keywords=["volumen", "volume", "sube", "baja", "pon"],
    ),
    "play_music": IntentSpec(
        name="play_music",
        required_slots=[],
        keywords=["pon musica", "pon música", "reproduce", "play music", "music"],
    ),
    "pause_music": IntentSpec(
        name="pause_music",
        required_slots=[],
        keywords=["pausa musica", "pausa música", "pause music", "pausa"],
    ),
    "shutdown_pc": IntentSpec(
        name="shutdown_pc",
        required_slots=[],
        keywords=["apaga el ordenador", "apaga el pc", "shutdown", "turn off pc"],
    ),
    "restart_pc": IntentSpec(
        name="restart_pc",
        required_slots=[],
        keywords=["reinicia el ordenador", "reinicia el pc", "restart"],
    ),
    "sleep_mode": IntentSpec(
        name="sleep_mode",
        required_slots=[],
        keywords=["duerme", "modo reposo", "sleep mode", "vete a dormir"],
    ),
    "chat": IntentSpec(
        name="chat",
        required_slots=[],
        keywords=[],
    ),
}


TOOL_INTENTS: set[str] = {
    "open_app",
    "close_window",
    "set_volume",
    "play_music",
    "pause_music",
    "shutdown_pc",
    "restart_pc",
    "sleep_mode",
}