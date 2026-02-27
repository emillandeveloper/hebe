import os
import subprocess
import time
import json
import uuid
from datetime import datetime
import tempfile
import csv
import re

import pyautogui
import requests
import pygame
import keyboard
import ollama
import pygetwindow as gw
from pywinauto.application import Application
from app.tools.windows_apps import open_app

from app.services.db_sqlite import (
    init_db,
    log_chat,
    seed_default_apps,
    find_app_for_command as db_find_app_for_command,
    register_app_usage,
    add_memory,
    get_active_memories,
    save_app_command,
    update_app_process_name,
)

from app.services.vts_client import vts_hotkey

from app.services.tts_service import speak as tts_speak
from app.services.stt_whisper import STTService, STTConfig


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

OLLAMA_MODEL = "hebe"  # nombre del modelo en Ollama modelos-> hebe / hebe-nsfw


# =========================
#  MEMORIA EN RAM
# =========================

historial = []

stt = STTService(
    config=STTConfig(),
    emit=emit,
    log_chat=log_chat,
)


PALABRAS_CLAVE = ["hebe despierta", "eve despierta", "jebe despierta", "asistente despierta"]
MODO_ESPERA = ["a dormir", "modo espera", "descansa"]
SEARCH_KEYWORDS = [
    "busca información sobre", "dime sobre", "explícame", "quiero saber sobre",
    "cuéntame sobre", "investiga sobre", "consulta sobre",
    "encuentra información sobre", "dame detalles de", "resumen sobre"
]

# =========================
#  TTS BACKEND (AUTO / PIPER / XTTS)
# =========================

HEBE_TTS_MODE = os.getenv("HEBE_TTS_MODE", "auto").lower()  # auto | piper | xtts
HEBE_TTS_MIN_VRAM_GB = float(os.getenv("HEBE_TTS_MIN_VRAM_GB", "12"))
HEBE_TTS_VOLUME = float(os.getenv("HEBE_TTS_VOLUME", "0.9"))

# =========================
#  TTS + VTS
# =========================
def speak(text: str, language: str = "es"):
    # Mantén init_models() por Whisper si te da tranquilidad (no carga TTS ya)
    # init_models()

    emit("chat.assistant", {"text": text})
    print(f"🗣️ (TTS service) Generando voz: {text}")

    audio_path = ""
    emit("status", {"tts_debug": "before_tts_speak"})

    try:
        try:
            audio_path = tts_speak(text=text, language=language, emit=emit, log_chat=log_chat)
            print("✅ tts_speak devolvió:", audio_path)
        except Exception as e:
            print("❌ tts_speak CRASHEÓ:", repr(e))
            emit("status", {"tts_debug": "tts_speak_error", "error": repr(e)})

            raise

        emit("status", {"tts_debug": "after_tts_speak", "audio_path": audio_path})

        try:
            import os
            size = os.path.getsize(audio_path) if audio_path else -1
        except Exception as e:
            size = -2

        emit("status", {"tts_debug": "wav_size", "bytes": size})

        # VTS talking + playback se mantiene aquí (por ahora) para no romper tu flow
        vts_hotkey("HebeTalking")

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
        vts_hotkey("HebeIdle")
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        try:
            if audio_path:
                os.remove(audio_path)
        except OSError as e:
            print(f"⚠️ No se pudo borrar el audio temporal: {e}")


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
    app = db_find_app_for_command(command_text)
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
    respuesta = stt.listen()
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
        resp = stt.listen()
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
    nombre = stt.listen()
    if not nombre:
        speak("No he entendido el nombre. Lo dejamos para otro momento.")
        return
    nombre = nombre.strip().lower()

    speak(f"De acuerdo, la llamaré {nombre}. Ahora dime el comando o la ruta para abrirla.")
    comando = stt.listen()
    if not comando:
        speak("No he entendido el comando. Cancelamos el registro.")
        return
    comando = comando.strip()

    speak("¿Quieres añadir alias para esta aplicación? Por ejemplo, otras formas de llamarla. Si no, di 'no'.")
    alias_text = stt.listen()
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
            command = stt.listen()

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
        command = stt.listen()
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
                stt.init()  # <-- tarda, pero ya NO bloquea el WS

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
    stt.list_audio_devices()
    engine = HebeEngine(use_wakeword=True, say_hello=True)
    engine.start()
    while True:
        time.sleep(1)
