# app/services/nlu_catalog.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass(frozen=True)
class IntentSpec:
    name: str
    kind: str  # "action" | "chat"
    required_slots: tuple[str, ...] = ()
    dangerous: bool = False

# --- Helpers de normalización simples (sin regex por frase exacta) ---
def _contains_any(text: str, words: list[str]) -> bool:
    t = text.lower()
    return any(w in t for w in words)

def _strip_prefixes(text: str, prefixes: list[str]) -> str:
    t = text.strip()
    tl = t.lower()
    for p in prefixes:
        if tl.startswith(p):
            return t[len(p):].strip()
    return t

# --- Slot extractors (parsing pequeño por intent; NO router de frases) ---
def extract_open_app(text: str) -> dict:
    # "abre obs", "abre la calculadora", etc.
    rest = _strip_prefixes(text, ["abre ", "abre la ", "abre el ", "abrir "])
    rest = rest.strip().strip('"').strip("'")
    return {"app_raw": rest} if rest else {}

def extract_memory_store(text: str) -> dict:
    rest = text
    for p in ["hebe recuerda que ", "eve recuerda que ", "recuerda que "]:
        if rest.lower().startswith(p):
            rest = rest[len(p):]
            break
    rest = rest.strip()
    return {"text": rest} if rest else {}

def extract_volume_control(text: str) -> dict:
    t = text.lower()
    if "silencia" in t or "mute" in t:
        return {"action": "mute"}
    if "sube" in t:
        return {"action": "up"}
    if "baja" in t:
        return {"action": "down"}
    return {}

def extract_power_control(text: str) -> dict:
    t = text.lower()
    if "apaga" in t:
        return {"action": "shutdown"}
    if "reinicia" in t:
        return {"action": "restart"}
    return {}

def extract_ytmusic_control(text: str) -> dict:
    t = text.lower()
    if "pausa" in t:
        return {"action": "pause"}
    if "reproduce" in t or "pon" in t:
        return {"action": "play"}
    if "siguiente" in t:
        return {"action": "next"}
    if "anterior" in t:
        return {"action": "prev"}
    if "silencia" in t:
        return {"action": "mute"}
    return {}

# --- Catálogo v1 (tus intents actuales) ---
INTENTS: dict[str, IntentSpec] = {
    "exit": IntentSpec(name="exit", kind="action"),
    "sleep_mode": IntentSpec(name="sleep_mode", kind="action"),
    "open_app": IntentSpec(name="open_app", kind="action", required_slots=("app_raw",)),
    "close_window": IntentSpec(name="close_window", kind="action"),
    "ytmusic_control": IntentSpec(name="ytmusic_control", kind="action", required_slots=("action",)),
    "volume_control": IntentSpec(name="volume_control", kind="action", required_slots=("action",)),
    "power_control": IntentSpec(name="power_control", kind="action", required_slots=("action",), dangerous=True),
    "memory_store": IntentSpec(name="memory_store", kind="action", required_slots=("text",)),
    "chat": IntentSpec(name="chat", kind="chat"),
}

# --- Gate heurístico (solo para bootstrap). Luego lo cambias por sklearn/fastText ---
# Esto NO es "regex por frase", es detección por señales.
INTENT_KEYWORDS: dict[str, list[str]] = {
    "exit": ["salir", "termina", "cierra hebe"],
    "sleep_mode": ["modo de espera", "descansa", "duerme", "a dormir"],
    "open_app": ["abre", "abrir"],
    "close_window": ["cierra ventana", "cierra la ventana", "cerrar ventana"],
    "ytmusic_control": ["pausa música", "reproduce música", "siguiente canción", "canción anterior", "anterior canción", "silenciar música"],
    "volume_control": ["sube volumen", "baja volumen", "silenciar", "silencia", "mute"],
    "power_control": ["apaga el ordenador", "reinicia el ordenador", "apaga", "reinicia"],
    "memory_store": ["recuerda que", "hebe recuerda que", "eve recuerda que"],
}

SLOT_EXTRACTORS: dict[str, Callable[[str], dict]] = {
    "open_app": extract_open_app,
    "memory_store": extract_memory_store,
    "volume_control": extract_volume_control,
    "power_control": extract_power_control,
    "ytmusic_control": extract_ytmusic_control,
}