import os
import subprocess
import time
import asyncio
import json
import uuid
import sqlite3
from datetime import datetime
import tempfile
import csv
import re

import websockets
import pyautogui
import requests
import pyaudio
import numpy as np
import pygame
import keyboard
import ollama
from TTS.api import TTS
from faster_whisper import WhisperModel
import pygetwindow as gw
from pywinauto.application import Application
from app.tools.registry import find_app_for_command
from app.tools.windows_apps import open_app


# =========================
#  UI / BACKEND BRIDGE
# =========================
import threading
import queue

_UI_INBOX: "queue.Queue[str]" = queue.Queue()
_EMIT = None  # callable(event_type: str, data: dict)

def set_emitter(fn):
    """Inyecta un callback para enviar eventos a la UI (WebSocket)."""
    global _EMIT
    _EMIT = fn

def emit(event_type: str, data: dict | None = None):
    """Envía un evento a la UI si hay emisor configurado."""
    if _EMIT:
        try:
            _EMIT(event_type, data or {})
        except Exception:
            pass

def submit_text_from_ui(text: str):
    """Mete texto escrito en la UI para que Hebe lo procese."""
    if text is None:
        return
    _UI_INBOX.put(str(text))

# =========================
# =========================
#  CONFIG LLM / MODELO
# =========================

OLLAMA_MODEL = "hebe-nsfw"  # nombre del modelo en Ollama

# =========================
#  CONFIG AUDIO / STT
# =========================

RATE = 16000
CHANNELS = 1
CHUNK = 1024
INPUT_DEVICE_INDEX = int(os.getenv("HEBE_INPUT_DEVICE_INDEX", "9"))  # índice del micro (ver listar dispositivos)

SILENCE_THRESHOLD = 0.01
MAX_RECORD_SECONDS = 8.0
MIN_RECORD_SECONDS = 0.5
SILENCE_END_SECONDS = 0.8
SILENCE_FRAMES_NEEDED = int(SILENCE_END_SECONDS / (CHUNK / RATE))

# Modelo STT (Whisper en CPU)
stt_model = None  # se inicializa en init_models()


# =========================
#  MEMORIA EN RAM
# =========================

historial = []

PALABRAS_CLAVE = ["hebe despierta", "eve despierta", "jebe despierta", "asistente despierta"]
MODO_ESPERA = ["a dormir", "modo espera", "descansa"]
SEARCH_KEYWORDS = [
    "busca información sobre", "dime sobre", "explícame", "quiero saber sobre",
    "cuéntame sobre", "investiga sobre", "consulta sobre",
    "encuentra información sobre", "dame detalles de", "resumen sobre"
]

# =========================
#  TTS (XTTS)
# =========================

tts = None  # se inicializa en init_models()

# =========================
#  TTS BACKEND (AUTO / PIPER / XTTS)
# =========================
HEBE_TTS_MODE = os.getenv("HEBE_TTS_MODE", "auto").lower()  # auto | piper | xtts
HEBE_TTS_MIN_VRAM_GB = float(os.getenv("HEBE_TTS_MIN_VRAM_GB", "12"))
HEBE_TTS_VOLUME = float(os.getenv("HEBE_TTS_VOLUME", "0.9"))

# Piper (externo) - si no lo configuras, AUTO caerá a XTTS.
HEBE_PIPER_EXE = os.getenv("HEBE_PIPER_EXE", "")
HEBE_PIPER_MODEL_ES = os.getenv("HEBE_PIPER_MODEL_ES", "")
HEBE_PIPER_MODEL_EN = os.getenv("HEBE_PIPER_MODEL_EN", "")
HEBE_PIPER_LENGTH_SCALE = float(os.getenv("HEBE_PIPER_LENGTH_SCALE", "1.0"))

# Filtrado básico de "alucinaciones" típicas de Whisper en silencio/ruido
STT_BLACKLIST = [
    "subtítulos por la comunidad de amara.org",
    "subtitulos por la comunidad de amara.org",
    "suscríbete",
    "suscribete",
]

def _has_cuda_vram(min_gb: float) -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        return vram_gb >= float(min_gb)
    except Exception:
        return False

def pick_tts_backend() -> str:
    mode = (HEBE_TTS_MODE or "auto").lower()
    if mode in ("piper", "xtts"):
        return mode

    # auto:
    # - Si la GPU tiene VRAM suficiente → XTTS en GPU
    # - Si no → Piper si está configurado; si no, XTTS (CPU)
    if _has_cuda_vram(HEBE_TTS_MIN_VRAM_GB):
        return "xtts"

    if HEBE_PIPER_EXE and os.path.exists(HEBE_PIPER_EXE) and (
        (HEBE_PIPER_MODEL_ES and os.path.exists(HEBE_PIPER_MODEL_ES)) or
        (HEBE_PIPER_MODEL_EN and os.path.exists(HEBE_PIPER_MODEL_EN))
    ):
        return "piper"

    return "xtts"

def _apply_torch_xtts_compat():
    # PyTorch 2.6 cambió el default weights_only=True; XTTS necesita el estado completo.
    try:
        import torch
        _orig_load = torch.load

        def _load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _load_compat

        # Allowlist para el error "Unsupported global: XttsConfig"
        try:
            from torch.serialization import add_safe_globals
            from TTS.tts.configs.xtts_config import XttsConfig
            add_safe_globals([XttsConfig])
        except Exception as e:
            print(f"WARN: safe_globals not applied: {e}")
    except Exception as e:
        print(f"WARN: torch compat not applied: {e}")

def piper_tts_to_wav(text: str, out_wav: str, language: str = "es"):
    exe = HEBE_PIPER_EXE
    if not exe or not os.path.exists(exe):
        raise FileNotFoundError("Piper no está configurado. Define HEBE_PIPER_EXE.")

    lang = (language or "es").lower()
    model = HEBE_PIPER_MODEL_ES if lang.startswith("es") else HEBE_PIPER_MODEL_EN
    if not model or not os.path.exists(model):
        raise FileNotFoundError("Modelo Piper no encontrado. Define HEBE_PIPER_MODEL_ES / HEBE_PIPER_MODEL_EN.")

    # Piper suele leer texto por stdin y generar wav en fichero
    cmd = [exe, "-m", model, "-f", out_wav]
    # Opción de velocidad si tu build de piper la soporta
    if HEBE_PIPER_LENGTH_SCALE and float(HEBE_PIPER_LENGTH_SCALE) != 1.0:
        cmd += ["--length_scale", str(HEBE_PIPER_LENGTH_SCALE)]

    subprocess.run(cmd, input=text.encode("utf-8"), check=True)

def is_blacklisted_stt(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    for bad in STT_BLACKLIST:
        if bad in t:
            return True
    return False


def init_models():
    """Inicializa Whisper/TTS la primera vez (para evitar cargas duplicadas con uvicorn)."""
    global stt_model, tts
    if stt_model is None:
        stt_model = WhisperModel(
            "small",          # puedes probar "tiny" si quieres más velocidad
            device="cpu",
            compute_type="int8",
        )

    # TTS: solo inicializamos XTTS si vamos a usarlo (o como fallback)
    if tts is None and pick_tts_backend() == "xtts":
        _apply_torch_xtts_compat()
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False

        # Nota: XTTS puede tardar bastante si corre en CPU.
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)


# =========================
#  BASE DE DATOS (SQLite)
# =========================

DB_PATH = os.path.join(os.getcwd(), "hebe.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(conn: sqlite3.Connection, table: str, column: str, col_def: str) -> None:
    """Add a column to a table if it doesn't exist (safe migration)."""
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        rows = cur.fetchall()
        existing = set()
        for r in rows:
            # r can be a tuple (sqlite default) or sqlite3.Row
            try:
                existing.add(r[1])
            except Exception:
                existing.add(r["name"])
        if column not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
    except Exception as e:
        # Don't crash the whole engine on a best-effort migration
        print(f"⚠️ No se pudo asegurar la columna {table}.{column}: {e}")

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    # Conversaciones
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            role TEXT NOT NULL,      -- 'user' | 'assistant' | 'system'
            source TEXT NOT NULL,    -- 'voice' | 'tts' | 'llm' | 'wiki'...
            text TEXT NOT NULL
        )
        """
    )

    # Apps que Hebe puede abrir
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            command TEXT NOT NULL,
            description TEXT,
            aliases TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            usage_count INTEGER NOT NULL DEFAULT 0,
            last_used_at TEXT,
            process_name TEXT,
            window_title TEXT
        );
        """
    )

    # Memorias
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT,
            text TEXT NOT NULL,
            category TEXT,
            importance INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            active INTEGER NOT NULL DEFAULT 1
        )
        """
    )

    # Safe migrations (older DBs might miss columns)
    ensure_column(conn, 'app_commands', 'process_name', 'TEXT')
    ensure_column(conn, 'app_commands', 'window_title', 'TEXT')

    conn.commit()
    conn.close()


def log_chat(role: str, text: str, source: str = "voice"):
    """Guarda una línea de conversación en la BD."""
    if not text:
        return
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_log (timestamp, role, source, text)
        VALUES (?, ?, ?, ?)
        """,
        (datetime.utcnow().isoformat(), role, source, text),
    )
    conn.commit()
    conn.close()


def seed_default_apps():
    """Rellena app_commands con algunas apps por defecto si no existen."""
    defaults = [
        ("chrome", "start chrome", "Navegador Chrome", "navegador,google"),
        ("discord", "start discord", "Discord", "dc"),
        ("obs", r"C:\Program Files\obs-studio\bin\64bit\obs64.exe", "OBS Studio", ""),
        ("explorador", "explorer", "Explorador de archivos", "explorer,archivos"),
        ("notas", "notepad", "Bloc de notas", "bloc,notepad"),
        ("calculadora", "calc", "Calculadora", "calc,calculadora"),
        (
            "final",
            r"C:\Program Files (x86)\SquareEnix\FINAL FANTASY XIV - A Realm Reborn\boot\ffxivboot.exe",
            "Final Fantasy XIV",
            "ff14,ffxiv",
        ),
    ]

    conn = get_db_connection()
    cur = conn.cursor()
    for name, cmd, desc, aliases in defaults:
        cur.execute(
            """
            INSERT OR IGNORE INTO app_commands (name, command, description, aliases)
            VALUES (?, ?, ?, ?)
            """,
            (name, cmd, desc, aliases),
        )
    conn.commit()
    conn.close()


def find_app_for_command(command_text: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM app_commands WHERE enabled = 1")
    apps = cur.fetchall()
    conn.close()

    text = (command_text or "").lower()
    # tokens “limpios”
    tokens = set(re.findall(r"[a-z0-9]+", text))

    best = None
    best_score = -1

    for app in apps:
        names = [app["name"]]
        if app["aliases"]:
            names += [a.strip() for a in app["aliases"].split(",") if a.strip()]

        for alias in names:
            a = (alias or "").strip().lower()
            if not a:
                continue

            a_tokens = set(re.findall(r"[a-z0-9]+", a))
            if not a_tokens:
                continue

            # score base: cuántos tokens del alias aparecen en el texto
            hit = len(a_tokens & tokens)
            if hit == 0:
                continue

            # bonus: alias completo como palabra/segmento
            bonus = 0
            if re.search(rf"\b{re.escape(a)}\b", text):
                bonus += 3

            # bonus: cuanto más largo el alias, mejor (evita “calc” vs “calculator pro”)
            bonus += min(len(a), 30) / 10.0

            # bonus: prioriza lo más usado
            usage = int(app["usage_count"] or 0)
            bonus += min(usage, 50) / 25.0

            score = hit * 5 + bonus

            if score > best_score:
                best_score = score
                best = app

    return best


def register_app_usage(app_id: int):
    """Actualiza estadísticas de uso de una app."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE app_commands
        SET usage_count = usage_count + 1,
            last_used_at = ?
        WHERE id = ?
        """,
        (datetime.utcnow().isoformat(), app_id),
    )
    conn.commit()
    conn.close()


# ==== Helpers de MEMORIA y APPS (BD) ====

def add_memory(text: str, key: str | None = None, category: str | None = None, importance: int = 1):
    """Añade una memoria a la tabla memories."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO memories (key, text, category, importance, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (key, text, category, importance, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_active_memories(limit: int = 5):
    """Devuelve las memorias activas más importantes/recientes."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, key, text, category, importance, created_at, last_used_at
        FROM memories
        WHERE active = 1
        ORDER BY importance DESC, created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def save_app_command(name: str, command: str, description: str = "", aliases: str = ""):
    """Inserta o reutiliza una app en app_commands y devuelve su id."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO app_commands (name, command, description, aliases)
        VALUES (?, ?, ?, ?)
        """,
        (name, command, description, aliases),
    )
    conn.commit()
    cur.execute("SELECT id FROM app_commands WHERE name = ?", (name,))
    row = cur.fetchone()
    conn.close()
    return row["id"] if row else None

# =========================
#  VTS (VTube Studio API)
# =========================

VTS_HOST = "127.0.0.1"
VTS_PORT = 8001

VTS_PLUGIN_NAME = "HebeAssistant"
VTS_PLUGIN_AUTHOR = "Leo"
VTS_PLUGIN_ICON = None
VTS_TOKEN_FILE = "vts_auth_token.txt"


class VTSClient:
    def __init__(self, host=VTS_HOST, port=VTS_PORT):
        self.host = host
        self.port = port
        self.ws = None
        self.authenticated = False
        self.auth_token = None

        if os.path.exists(VTS_TOKEN_FILE):
            try:
                with open(VTS_TOKEN_FILE, "r", encoding="utf-8") as f:
                    self.auth_token = f.read().strip() or None
            except Exception:
                self.auth_token = None

    async def connect(self):
        uri = f"ws://{self.host}:{self.port}"
        print(f"🔌 Conectando a VTube Studio en {uri}...")
        self.ws = await websockets.connect(uri)
        await self.authenticate()

    async def request_auth_token(self):
        if self.ws is None:
            raise RuntimeError("WebSocket no inicializado")

        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "token-1",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": VTS_PLUGIN_NAME,
                "pluginDeveloper": VTS_PLUGIN_AUTHOR,
                "pluginIcon": VTS_PLUGIN_ICON,
            },
        }

        await self.ws.send(json.dumps(msg))
        print("📨 Enviado AuthenticationTokenRequest (mira VTS y acepta el plugin)")

        resp_raw = await self.ws.recv()
        resp = json.loads(resp_raw)
        print("📥 Respuesta token:", resp.get("messageType"), resp.get("data"))

        if resp.get("messageType") == "AuthenticationTokenResponse":
            token = resp.get("data", {}).get("authenticationToken")
            if token:
                self.auth_token = token
                with open(VTS_TOKEN_FILE, "w", encoding="utf-8") as f:
                    f.write(token)
                print("✅ Token de VTS guardado en", VTS_TOKEN_FILE)
            else:
                raise RuntimeError("VTS no devolvió token de autenticación.")
        else:
            raise RuntimeError(f"Respuesta inesperada al pedir token: {resp}")

    async def authenticate(self):
        if self.ws is None:
            raise RuntimeError("WebSocket no inicializado")

        if self.auth_token is None:
            await self.request_auth_token()

        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth-1",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": VTS_PLUGIN_NAME,
                "pluginDeveloper": VTS_PLUGIN_AUTHOR,
                "pluginIcon": VTS_PLUGIN_ICON,
                "authenticationToken": self.auth_token,
            },
        }

        await self.ws.send(json.dumps(msg))
        print("📨 Enviado AuthenticationRequest")

        resp_raw = await self.ws.recv()
        resp = json.loads(resp_raw)
        print("📥 Respuesta de auth:", resp.get("messageType"), resp.get("data"))

        if resp.get("messageType") == "AuthenticationResponse" and resp.get("data", {}).get("authenticated"):
            self.authenticated = True
            print("✅ Autenticado con VTube Studio.")
        else:
            print("❌ Fallo autenticando con VTS:", resp)
            raise RuntimeError("No se pudo autenticar con VTS.")

    async def _send_request(self, message_type: str, data: dict, request_id: str = "req-1"):
        if self.ws is None:
            raise RuntimeError("WebSocket no inicializado")

        msg = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": request_id,
            "messageType": message_type,
            "data": data,
        }
        await self.ws.send(json.dumps(msg))
        resp_raw = await self.ws.recv()
        return json.loads(resp_raw)

    async def list_hotkeys(self):
        resp = await self._send_request(
            "HotkeysInCurrentModelRequest",
            {},
            request_id="hotkeys-current-model-1",
        )

        print("🔎 Respuesta cruda de HotkeysInCurrentModelRequest:")
        print(resp)

        data = resp.get("data", {}) or {}
        hotkeys = data.get("availableHotkeys") or data.get("hotkeys") or []

        if not hotkeys:
            print("⚠️ No se han recibido hotkeys en la respuesta.")
        else:
            print("🔎 Hotkeys disponibles en el modelo actual:")
            for hk in hotkeys:
                print(" -", hk)

        return resp

    async def get_hotkey_id_by_name(self, hotkey_name: str) -> str | None:
        resp = await self._send_request(
            "HotkeysInCurrentModelRequest",
            {},
            request_id="hotkeys-lookup",
        )

        data = resp.get("data", {}) or {}
        hotkeys = data.get("availableHotkeys") or data.get("hotkeys") or []

        for hk in hotkeys:
            if hk.get("name") == hotkey_name:
                return hk.get("hotkeyID")

        print(f"⚠️ No encontré ningún hotkey con nombre '{hotkey_name}' en la lista.")
        print("   Nombres encontrados:", [hk.get("name") for hk in hotkeys])
        return None

    async def trigger_hotkey(self, hotkey_name: str):
        print(f"🔥 Disparando hotkey '{hotkey_name}'")

        hotkey_id = await self.get_hotkey_id_by_name(hotkey_name)
        if not hotkey_id:
            print(f"❌ No se pudo obtener hotkeyID para '{hotkey_name}'.")
            return

        resp = await self._send_request(
            "HotkeyTriggerRequest",
            {
                "hotkeyID": hotkey_id,
            },
            request_id=f"hotkey-{hotkey_name}",
        )

        if resp.get("messageType") == "APIError":
            print("⚠️ Error al ejecutar hotkey por ID:", resp)
        else:
            print("📥 Respuesta hotkey:", resp)

    async def close(self):
        if self.ws:
            await self.ws.close()
            self.ws = None
            self.authenticated = False
            print("🔌 Desconectado de VTS.")


async def _vts_hotkey_async(nombre_hotkey: str):
    client = VTSClient()
    try:
        await client.connect()
        await client.trigger_hotkey(nombre_hotkey)
    finally:
        await client.close()


def vts_hotkey(nombre_hotkey: str):
    try:
        asyncio.run(_vts_hotkey_async(nombre_hotkey))
    except Exception as e:
        print(f"❌ Error al disparar hotkey VTS '{nombre_hotkey}': {e}")


# =========================
#  TTS + VTS
# =========================

def speak(text: str, language: str = "es"):
    init_models()
    emit("chat.assistant", {"text": text})
    emit("tts.start", {"text": text, "lang": language})

    # Guardamos lo que dice Hebe
    log_chat("assistant", text, source="tts")

    print(f"🗣️ Generando voz: {text}")

    tmp_dir = os.path.join(os.getcwd(), "audio_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tmp_dir) as tmp_file:
        audio_path = tmp_file.name

    backend = pick_tts_backend()

    try:
        if backend == "piper":
            # Piper (rápido en CPU)
            piper_tts_to_wav(text=text, out_wav=audio_path, language=language)
        else:
            # XTTS (más natural, más pesado). Si falla, hacemos fallback a Piper si existe.
            if tts is None:
                # Por si el modo cambió en caliente
                _apply_torch_xtts_compat()
                try:
                    import torch
                    use_gpu = torch.cuda.is_available()
                except Exception:
                    use_gpu = False
                # noqa: F811
                globals()["tts"] = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)

            try:
                tts.tts_to_file(
                    text=text,
                    file_path=audio_path,
                    language=language,
                    speaker="Ana Florence",
                    split_sentences=True,
                )
            except Exception as e:
                print(f"⚠️ XTTS falló, intento Piper: {e}")
                # Fallback a Piper si está configurado
                piper_tts_to_wav(text=text, out_wav=audio_path, language=language)

        # Cuando el audio está listo, activamos expresión de hablar
        vts_hotkey("HebeTalking")

        # Reproducir
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.set_volume(float(HEBE_TTS_VOLUME))
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

    except Exception as e:
        print(f"❌ Error en speak(): {e}")
    finally:
        emit("tts.end", {})
        # Volver a Idle y limpiar
        vts_hotkey("HebeIdle")
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        try:
            os.remove(audio_path)
        except OSError as e:
            print(f"⚠️ No se pudo borrar el audio temporal: {e}")

# =========================
#  STT / AUDIO INPUT
# =========================

def grabar_audio(segundos: float = 4.0) -> np.ndarray:
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=INPUT_DEVICE_INDEX,
        frames_per_buffer=CHUNK,
    )

    frames = []
    n_chunks = int(RATE / CHUNK * segundos)

    print(f"🎤 Escuchando {segundos:.1f}s...")
    for _ in range(n_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_bytes = b"".join(frames)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    audio_np /= 32768.0

    max_abs = float(np.max(np.abs(audio_np))) if len(audio_np) > 0 else 0.0
    print(f"📈 Samples grabados: {len(audio_np)}, nivel máx: {max_abs:.3f}")

    return audio_np


def listen() -> str:
    # NO init_models() aquí. Hazlo una vez al arrancar el motor.

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=INPUT_DEVICE_INDEX,
        frames_per_buffer=CHUNK,
    )

    emit("status", {"stt": "listening"})
    frames = []
    recording = False
    silence_frames = 0
    start_time = time.time()
    tick = 0

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            level = float(np.max(np.abs(audio_chunk))) if len(audio_chunk) > 0 else 0.0
            tick += 1

            # ✅ cada ~10 frames manda “live” (puede ser el nivel)
            if tick % 10 == 0:
                emit("stt.partial", {"text": f"lvl {level:.3f}"})

            if not recording:
                if level > SILENCE_THRESHOLD:
                    recording = True
                    frames.append(data)
                    start_time = time.time()
                    silence_frames = 0
                    emit("status", {"stt": "recording"})
            else:
                frames.append(data)
                # ✅ mientras grabas, sigues en recording
                # (si quieres, aquí no emitas nada salvo cada X frames)

                if level < SILENCE_THRESHOLD:
                    silence_frames += 1
                else:
                    silence_frames = 0

                elapsed = len(frames) * (CHUNK / RATE)

                if (elapsed >= MIN_RECORD_SECONDS and silence_frames >= SILENCE_FRAMES_NEEDED) or elapsed >= MAX_RECORD_SECONDS:
                    break

                if time.time() - start_time > MAX_RECORD_SECONDS + 2:
                    break

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    if not frames:
        emit("status", {"stt": "listening"})
        emit("stt.partial", {"text": ""})
        return ""

    emit("status", {"stt": "transcribing"})

    audio_bytes = b"".join(frames)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    max_abs = float(np.max(np.abs(audio_np))) if len(audio_np) > 0 else 0.0

    if max_abs < SILENCE_THRESHOLD:
        emit("status", {"stt": "listening"})
        emit("stt.partial", {"text": ""})
        return ""

    segments, info = stt_model.transcribe(
        audio_np,
        language=None,
        beam_size=5,
        vad_filter=True,
    )

    texto = "".join(seg.text for seg in segments).strip()

    if is_blacklisted_stt(texto.lower()):
        emit("status", {"stt": "listening"})
        emit("stt.partial", {"text": ""})
        return ""

    if texto:
        emit("stt.final", {"text": texto})
        emit("chat.user", {"text": texto})
        log_chat("user", texto, source="voice")

    emit("status", {"stt": "listening"})
    emit("stt.partial", {"text": ""})
    return texto


# =========================
#  LÓGICA LLM / WIKIPEDIA
# =========================

def buscar_en_wikipedia(consulta):
    try:
        url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{consulta.replace(' ', '_')}"
        response = requests.get(url)

        print(f"[DEBUG] URL de Wikipedia: {url}")

        if response.status_code == 200:
            data = response.json()
            print(f"[DEBUG] Respuesta de Wikipedia: {data}")

            if "extract" in data:
                extracto = data["extract"]
                primer_parrafo = (
                    extracto.split(". ")[0] + "."
                    if "." in extracto
                    else extracto
                )
                return primer_parrafo

            return "No encontré información relevante en Wikipedia."

        return f"Error en Wikipedia: Código {response.status_code}"

    except Exception as e:
        return f"Error al buscar en Wikipedia: {e}"


def obtener_respuesta_gpt(pregunta: str) -> str:
    try:
        pregunta = (pregunta or "").strip()
        if not pregunta:
            return ""

        # 1) añade user al historial
        historial.append({"role": "user", "content": pregunta})

        # 2) NO mandes system_prompt: ya está embebido en el modelo "hebe"
        response = ollama.chat(
            model=OLLAMA_MODEL,   # "hebe"
            messages=historial,
            options={
                "temperature": 0.7,
                "repeat_penalty": 1.2,
                "top_p": 0.9,
                "num_predict": 1200,   # 5000 es una locura para chat normal
                "num_ctx": 2048,
            },
        )

        texto_generado = (response.get("message", {}) or {}).get("content", "").strip()

        # 3) guarda respuesta
        if texto_generado:
            historial.append({"role": "assistant", "content": texto_generado})
            log_chat("assistant", texto_generado, source="llm")
            emit("llm.final", {"text": texto_generado})

        return texto_generado or "…"

    except Exception as e:
        print("❌ Error al generar respuesta con Ollama:", e)
        emit("error", {"where": "ollama", "error": str(e)})
        return "Lo siento, no puedo generar una respuesta en este momento."



def obtener_respuesta(pregunta):
    return obtener_respuesta_gpt(pregunta)

# =========================
#  ACCIONES / COMANDOS DEL PC
# =========================

# --- App launching helpers ----------------------------------------------------

_APP_OPEN_DEBOUNCE = {}
_APP_OPEN_DEBOUNCE_SECONDS = 2.0

def _win_creationflags() -> int:
    """Creation flags to detach GUI apps from the current console (prevents log spam)."""
    if os.name != "nt":
        return 0
    flags = 0
    # These constants exist only on Windows
    flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return flags

def _guess_exe_from_command(cmd: str):
    """Best-effort guess of the process image name (e.g. 'obs64.exe') from a command."""
    if not cmd:
        return None
    c = cmd.strip().strip('"').strip()
    low = c.lower()

    # 'start chrome' -> 'chrome.exe'
    if low.startswith("start "):
        rest = c[6:].strip().strip('"')
        if not rest:
            return None
        token = rest.split()[0].strip('"')
        if not token:
            return None
        if not token.lower().endswith(".exe"):
            token += ".exe"
        return token.lower()

    # Absolute exe path
    if ".exe" in low:
        # Take last path segment ending with .exe
        base = os.path.basename(c)
        if base.lower().endswith(".exe"):
            return base.lower()

    # Common built-ins
    token = c.split()[0].strip('"')
    if not token:
        return None
    if not token.lower().endswith(".exe"):
        token += ".exe"
    return token.lower()

def is_process_running(process_name: str) -> bool:
    if not process_name:
        return False
    pn = process_name.strip().strip('"')
    if os.name == "nt":
        if not pn.lower().endswith(".exe"):
            pn += ".exe"
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"IMAGENAME eq {pn}", "/FO", "CSV", "/NH"],
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            return pn.lower() in out.lower()
        except Exception:
            return False

    # Non-Windows fallback
    try:
        out = subprocess.check_output(["ps", "ax"], text=True, errors="ignore")
        return pn.lower() in out.lower()
    except Exception:
        return False

def _list_process_names_win() -> set:
    """Return a set of running process image names on Windows (lowercase)."""
    if os.name != "nt":
        return set()
    try:
        out = subprocess.check_output(
            ["tasklist", "/FO", "CSV", "/NH"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        names = set()
        for row in csv.reader(out.splitlines()):
            if not row:
                continue
            names.add(row[0].strip().strip('"').lower())
        return names
    except Exception:
        return set()

def _spawn_detached(exe_path: str, cwd: str | None = None) -> None:
    """Launch an executable detached from this console (prevents stdout spam)."""
    exe = exe_path.strip().strip('"')
    creationflags = _win_creationflags()
    subprocess.Popen(
        [exe],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )

def _run_cmd_windows(cmd: str) -> None:
    cmd_str = (cmd or "").strip()
    creationflags = _win_creationflags()
    if os.name == "nt" and cmd_str.lower().startswith("start "):
        subprocess.Popen(
            ["cmd", "/c", cmd_str],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
    else:
        subprocess.Popen(
            cmd_str,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )

def try_focus_app_window(app: dict) -> bool:
    """Try to bring an app window to front using pygetwindow (best-effort)."""
    try:
        candidates = []
        wt = (app.get("window_title") or "").strip()
        if wt:
            candidates.append(wt)
        nm = (app.get("name") or "").strip()
        if nm:
            candidates.append(nm)
        als = (app.get("aliases") or "").strip()
        if als:
            candidates.extend([a.strip() for a in als.split(",") if a.strip()])

        # Fast path: direct title search
        for cand in candidates:
            wins = gw.getWindowsWithTitle(cand)
            if wins:
                w = wins[0]
                try:
                    w.restore()
                except Exception:
                    pass
                w.activate()
                return True

        # Fallback: scan all titles and match substring
        titles = [t for t in gw.getAllTitles() if t]
        for cand in candidates:
            cl = cand.lower()
            for t in titles:
                if cl in t.lower():
                    wins = gw.getWindowsWithTitle(t)
                    if wins:
                        w = wins[0]
                        try:
                            w.restore()
                        except Exception:
                            pass
                        w.activate()
                        return True

        return False
    except Exception:
        return False

def _update_app_process_name(app_id: int, process_name: str) -> None:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE app_commands SET process_name = ? WHERE id = ?",
            (process_name, app_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ No se pudo guardar process_name en DB: {e}")

def learn_process_name_after_launch(app: dict, before: set, cmd: str) -> str | None:
    """Best-effort learning of process_name by diffing tasklist before/after launch."""
    if os.name != "nt":
        return None

    time.sleep(1.2)
    after = _list_process_names_win()
    new = {p for p in (after - (before or set())) if p}

    # Filter common helper processes
    noisy = {"cmd.exe", "conhost.exe", "powershell.exe", "python.exe"}
    new = {p for p in new if p not in noisy}

    if not new:
        return None

    guessed = _guess_exe_from_command(cmd) or ""
    if guessed and guessed in new:
        return guessed

    if len(new) == 1:
        return next(iter(new))

    # Try match by app name
    name = (app.get("name") or "").lower()
    for p in sorted(new):
        if name and name in p.lower():
            return p

    # Fallback: just pick the first (stable sort)
    return sorted(new)[0]

def abrir_aplicacion(command_text: str):
    app = find_app_for_command(command_text)
    if not app:
        speak("No conozco esa aplicación todavía.")
        return
    open_app(app, speak=speak)

def escribir_texto(command):
    if "escribe" in command:
        texto = command.replace("escribe", "").strip()
        speak(f"Escribiendo: {texto}")
        pyautogui.write(texto, interval=0.1)


def cerrar_ventana():
    speak("Cerrando la ventana actual.")
    pyautogui.hotkey("alt", "f4")


def controlar_volumen(command):
    if "sube volumen" in command:
        speak("Subiendo volumen.")
        for _ in range(5):
            pyautogui.press("volumeup")
    elif "baja volumen" in command:
        speak("Bajando volumen.")
        for _ in range(5):
            pyautogui.press("volumedown")
    elif "silenciar" in command:
        speak("Silenciando.")
        pyautogui.press("volumemute")


def enfocar_opera_youtube():
    ventanas = gw.getAllTitles()

    for ventana in ventanas:
        if "YouTube Music" in ventana or "music.youtube.com" in ventana.lower() or "Opera GX" in ventana:
            try:
                print(f"✅ Encontrado: {ventana}")
                app = Application().connect(title=ventana)
                app.top_window().set_focus()
                return True
            except Exception as e:
                print(f"❌ No se pudo enfocar YouTube Music en Opera GX: {e}")
                return False

    print("❌ No se encontró YouTube Music en Opera GX.")
    return False


def controlar_youtube_music(command: str) -> bool:
    if "pausa música" in command or "reproduce música" in command:
        keyboard.send("play/pause media")
        print("🎵 Música pausada o reanudada.")
        return True

    elif "siguiente canción" in command:
        keyboard.send("next track")
        print("⏭️ Siguiente canción.")
        return True

    elif "canción anterior" in command:
        keyboard.send("previous track")
        print("⏮️ Canción anterior.")
        return True

    elif "sube volumen" in command:
        for _ in range(5):
            keyboard.send("volume up")
        print("🔊 Subiendo volumen.")
        return True

    elif "baja volumen" in command:
        for _ in range(5):
            keyboard.send("volume down")
        print("🔉 Bajando volumen.")
        return True

    elif "silenciar música" in command:
        keyboard.send("volume mute")
        print("🔇 Música silenciada o activada.")
        return True
    return False

def buscar_y_reproducir_cancion(cancion):
    print(f"🔎 Buscando '{cancion}' en YouTube Music...")

    keyboard.send("/")
    time.sleep(0.5)

    keyboard.write(cancion)
    time.sleep(0.5)

    keyboard.send("enter")
    time.sleep(2)

    keyboard.send("tab")
    time.sleep(0.3)
    keyboard.send("enter")

    print("🎵 Reproduciendo la primera canción encontrada.")


def confirmar_accion(accion):
    speak(f"¿Seguro que quieres {accion}? Di sí o no.")
    respuesta = listen()
    if not respuesta:
        return False
    respuesta = respuesta.strip().lower()
    return (
        "sí" in respuesta
        or respuesta.startswith("si ")
        or respuesta == "si"
    )


def controlar_pc(command):
    if "apaga el ordenador" in command:
        if confirmar_accion("apagar el ordenador"):
            speak("Apagando el ordenador en 5 segundos.")
            time.sleep(5)
            os.system("shutdown /s /t 1")
    elif "reinicia el ordenador" in command:
        if confirmar_accion("reiniciar el ordenador"):
            speak("Reiniciando el ordenador en 5 segundos.")
            time.sleep(5)
            os.system("shutdown /r /t 1")

# =========================
#  MEMORIA Y APPS POR VOZ
# =========================

def guardar_memoria_desde_comando(command: str):
    """Procesa frases como 'recuerda que...' y guarda en memories."""
    texto = command
    for pref in ["hebe recuerda que", "eve recuerda que", "recuerda que"]:
        texto = texto.replace(pref, "")
    texto = texto.strip()

    if texto:
        add_memory(texto, category="usuario", importance=2)
        speak("De acuerdo, lo recordaré.")
    else:
        speak("¿Qué quieres que recuerde exactamente?")
        resp = listen()
        if resp:
            add_memory(resp, category="usuario", importance=2)
            speak("Lo recordaré.")
        else:
            speak("No he entendido nada, lo dejamos para más tarde.")


def responder_que_recuerdas():
    """Lee algunas memorias de la BD y las dice en voz alta."""
    mems = get_active_memories(limit=5)
    if not mems:
        speak("De momento no recuerdo nada especial que me hayas dicho.")
        return
    frases = [m["text"] for m in mems]
    respuesta = "Recuerdo algunas cosas: " + "; ".join(frases)
    speak(respuesta)


def aprender_nueva_app():
    """Diálogo por voz para registrar una nueva aplicación en app_commands."""
    speak("Vale, vamos a aprender una nueva aplicación. ¿Cómo quieres llamarla?")
    nombre = listen()
    if not nombre:
        speak("No he entendido el nombre. Lo dejamos para otro momento.")
        return
    nombre = nombre.strip().lower()

    speak(f"De acuerdo, la llamaré {nombre}. Ahora dime el comando o la ruta para abrirla.")
    comando = listen()
    if not comando:
        speak("No he entendido el comando. Cancelamos el registro.")
        return
    comando = comando.strip()

    speak("¿Quieres añadir alias para esta aplicación? Por ejemplo, otras formas de llamarla. Si no, di 'no'.")
    alias_text = listen()
    if alias_text:
        alias_text = alias_text.strip().lower()
        if alias_text in ("no", "no gracias", "nah", "nop"):
            alias_text = ""
    else:
        alias_text = ""

    app_id = save_app_command(nombre, comando, description="", aliases=alias_text)
    if app_id:
        speak(f"He guardado la aplicación {nombre}. Intentaré abrirla cuando me la pidas.")
    else:
        speak(f"No he podido guardar la aplicación {nombre}. Puede que ya exista otra con ese nombre.")

# =========================
#  LOOP DE COMANDOS
# =========================
# =========================
#  TOOL SYSTEM (Paso 1)
# =========================

def _tool_open_app(app: str):
    # Reusa la lógica actual (comando de voz) para abrir apps.
    return abrir_aplicacion(f"abre {app}")

def _tool_type_text(text: str, interval: float = 0.03):
    pyautogui.write(text, interval=interval)
    return {"chars": len(text)}

def _tool_press_keys(keys: list[str]):
    # keys: ["ctrl","l"] o ["CTRL","L"]
    norm = []
    for k in keys:
        kk = str(k).strip().lower()
        if kk in ("control", "ctl"):
            kk = "ctrl"
        if kk == "escape":
            kk = "esc"
        norm.append(kk)
    if not norm:
        return {"pressed": []}
    if len(norm) == 1:
        pyautogui.press(norm[0])
    else:
        pyautogui.hotkey(*norm)
    return {"pressed": norm}

def _tool_open_url(url: str):
    os.startfile(url)
    return {"url": url}

TOOLS = {
    "open_app": _tool_open_app,
    "type_text": _tool_type_text,
    "press_keys": _tool_press_keys,
    "open_url": _tool_open_url,
}

def exec_tool(name: str, args: dict):
    if name not in TOOLS:
        raise ValueError(f"Tool desconocida: {name}")
    return TOOLS[name](**(args or {}))



def handle_command(command: str, source: str = "voice") -> str:
    """Ejecuta un comando ya transcrito. Devuelve: 'continue' | 'sleep' | 'stop'."""
    if command == "":
        return "continue"

    raw = (command or "").strip()
    # /tool <name> <json|texto>
    # Ej: /tool open_url {"url":"https://youtube.com"}
    if raw.startswith("/tool "):
        rest = raw[len("/tool "):].strip()
        name, _, argstr = rest.partition(" ")
        name = name.strip()
        args = {}
        argstr = argstr.strip()

        if argstr:
            try:
                args = json.loads(argstr)
            except Exception:
                # fallback: argumentos simples (sin JSON)
                fallback_key = {
                    "open_app": "app",
                    "type_text": "text",
                    "open_url": "url",
                    "press_keys": "keys",
                }.get(name)
                if name == "press_keys":
                    args = {"keys": [k for k in argstr.replace(" ", "").split("+") if k]}
                elif fallback_key:
                    args = {fallback_key: argstr}

        try:
            tool_call(name, args, lambda: exec_tool(name, args))
            speak("Hecho.")
        except Exception as e:
            speak(f"No pude ejecutar la tool: {e}")
        return "continue"

    # Normaliza por seguridad (en UI puede venir con mayúsculas)
    command = command.strip().lower()

    if "salir" in command:
        speak("Hasta luego.")
        return "stop"

    if "hola" in command:
        speak("¡Hola! ¿Cómo puedo ayudarte?")
        return "continue"

    if any(keyword in command for keyword in MODO_ESPERA):
        speak("Entrando en modo de espera...")
        vts_hotkey("HebeSleep")
        return "sleep"

    if "abre" in command:
        tool_call("open_app", {"command": command}, lambda: abrir_aplicacion(command))
        return "continue"

    if "escribe" in command:
        tool_call("type_text", {"command": command}, lambda: escribir_texto(command))
        return "continue"

    if "cierra ventana" in command:
        tool_call("close_window", {}, lambda: cerrar_ventana())
        return "continue"

    if "sube volumen" in command or "baja volumen" in command or "silenciar" in command:
        tool_call("volume", {"command": command}, lambda: controlar_volumen(command))
        return "continue"

    if "apaga el ordenador" in command or "reinicia el ordenador" in command:
        controlar_pc(command)
        return "continue"

    if "recuerda que" in command or "hebe recuerda que" in command or "eve recuerda que" in command:
        guardar_memoria_desde_comando(command)
        return "continue"

    if "qué recuerdas de mí" in command or "que recuerdas de mi" in command or "qué recuerdas" in command:
        responder_que_recuerdas()
        return "continue"

    if "aprende una nueva aplicación" in command or "registra una aplicación" in command or "añade una aplicación" in command:
        aprender_nueva_app()
        return "continue"

    if any(keyword in command for keyword in SEARCH_KEYWORDS):
        tema = (
            command.replace("busca información sobre", "")
            .replace("dime sobre", "")
            .replace("explícame", "")
            .replace("quiero saber sobre", "")
            .replace("cuéntame sobre", "")
            .replace("investiga sobre", "")
            .replace("consulta sobre", "")
            .replace("encuentra información sobre", "")
            .replace("dame detalles de", "")
            .replace("resumen sobre", "")
            .strip()
        )
        respuesta = buscar_en_wikipedia(tema)
        log_chat("assistant", respuesta, source="wiki")
        speak(respuesta)
        print(f"Hebe: {respuesta}")
        return "continue"

    if "pon la canción" in command or "quiero escuchar" in command:
        cancion = (
            command.replace("pon la canción", "")
            .replace("quiero escuchar", "")
            .strip()
        )
        buscar_y_reproducir_cancion(cancion)
        return "continue"

    music_cmd = any(k in command for k in (
        "pausa", "reproduce", "siguiente", "anterior", "sube el volumen", "baja el volumen"
    ))
    if ("música" in command or "canción" in command or "volume" in command) and music_cmd:
        handled = tool_call("media_control", {"command": command}, lambda: controlar_youtube_music(command))  # <- haz que devuelva True/False
        if handled:
            return "continue"


    # Default: LLM
    respuesta = obtener_respuesta(command)
    speak(respuesta)
    print(f"Hebe: {respuesta}")
    return "continue"


def procesar_comando(stop_event: threading.Event | None = None) -> str:
    """Modo activo: procesa comandos de UI y/o voz. Devuelve 'sleep' o 'stop' cuando toque."""
    while True:
        if stop_event and stop_event.is_set():
            return "stop"

        from_ui = False
        try:
            command = _UI_INBOX.get_nowait()
            from_ui = True
            command = str(command).strip().lower()
        except queue.Empty:
            command = listen()

        if command == "":
            continue

        if from_ui:
            # Para UI, registramos y emitimos aquí (en voz ya lo hace listen()).
            log_chat("user", command, source="ui")
            emit("chat.user", {"text": command})

        res = handle_command(command, source="ui" if from_ui else "voice")
        if res in ("sleep", "stop"):
            return res

t0 = time.time()
def mark(stage):
    emit("status", {"engine":"starting","stage":stage,"t_ms": int((time.time()-t0)*1000)})

def activar_hebe(stop_event: threading.Event | None = None, say_hello: bool = False) -> str:
    """Modo wakeword. En paralelo acepta texto de UI sin necesidad de wakeword."""
    if say_hello:
        speak("¡Hola! ¿Cómo puedo ayudarte?")

    while True:
        if stop_event and stop_event.is_set():
            return "stop"

        # 1) UI siempre puede enviar texto (sin wakeword)
        try:
            cmd = _UI_INBOX.get_nowait()
            cmd = str(cmd).strip().lower()
            if cmd:
                log_chat("user", cmd, source="ui")
                emit("chat.user", {"text": cmd})
                res = handle_command(cmd, source="ui")
                if res == "stop":
                    return "stop"
            continue
        except queue.Empty:
            pass

        # 2) Voz: wakeword
        command = listen()
        if not command:
            continue

        cmd_norm = command.strip().lower()
        if any(keyword in cmd_norm for keyword in PALABRAS_CLAVE):
            vts_hotkey("HebeIdle")
            speak("Te escucho.")
            res = procesar_comando(stop_event=stop_event)
            if res == "stop":
                return "stop"
            # si res == "sleep", volvemos a esperar wakeword

def listar_dispositivos_audio():
    p = pyaudio.PyAudio()
    print("\n=== Dispositivos de audio detectados ===")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"{i}: {info['name']}  | inputs: {info['maxInputChannels']}")
    p.terminate()
    print("========================================\n")

# =========================
#  MAIN
# =========================

class HebeEngine:
    """Motor de Hebe ejecutándose en un hilo, controlable desde el backend/UI."""
    def __init__(self, use_wakeword: bool = True, say_hello: bool = False):
        self.use_wakeword = use_wakeword
        self.say_hello = say_hello
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

    

    def start(self):
        if self._started:
            return
        self._started = True

        def boot():
            try:
                emit("status", {"engine": "starting", "stage": "db"})
                init_db()

                emit("status", {"engine": "starting", "stage": "apps"})
                seed_default_apps()

                emit("status", {"engine": "starting", "stage": "models"})
                init_models()  # <-- tarda, pero ya NO bloquea el WS

                emit("status", {"engine": "ready", "stage": "ready"})

                target = activar_hebe if self.use_wakeword else procesar_comando
                kwargs = {"stop_event": self._stop_event}
                if target is activar_hebe:
                    kwargs["say_hello"] = self.say_hello

                self._thread = threading.Thread(target=target, kwargs=kwargs, daemon=True)
                self._thread.start()

            except Exception as e:
                emit("status", {"engine": "error", "stage": "boot", "error": str(e)})

        threading.Thread(target=boot, daemon=True).start()

    def stop(self):
        self._stop_event.set()

    def submit_text(self, text: str):
        submit_text_from_ui(text)


def tool_call(name: str, args: dict | None, fn):
    """Ejecuta una acción como 'tool' y emite tool.start/tool.end/tool.error a la UI."""
    tool_id = str(uuid.uuid4())
    emit("tool.start", {"id": tool_id, "name": name, "args": args or {}})
    t0 = time.time()
    try:
        result = fn()
        emit("tool.end", {
            "id": tool_id,
            "name": name,
            "ok": True,
            "ms": int((time.time() - t0) * 1000),
            "result": result,
        })
        return result
    except Exception as e:
        emit("tool.error", {
            "id": tool_id,
            "name": name,
            "ok": False,
            "ms": int((time.time() - t0) * 1000),
            "error": str(e),
        })
        raise

if __name__ == "__main__":
    # Modo standalone (sin backend): arranca Hebe y mantén vivo el proceso.
    listar_dispositivos_audio()
    engine = HebeEngine(use_wakeword=True, say_hello=True)
    engine.start()
    while True:
        time.sleep(1)
