import os
import time
import uuid
import pyautogui
import requests

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
from app.services.speech_output import speak as _speak
from app.services.stt_whisper import STTService, STTConfig
from app.services.win_automation import WinAutomationService
from app.services.command_router import CommandRouter
from app.services.tool_system import ToolSystem, ToolContext

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

def speak(text: str, language: str = "es") -> None:
    # Always notify UI + persist chat log when Hebe speaks
    return _speak(text=text, language=language, emit=emit, log_chat=log_chat)

win = WinAutomationService(emit=emit, speak=speak)
router = CommandRouter()

router.add(
    "exit",
    r"\bsalir\b",
    lambda t: (speak("Hasta luego."), "stop")[1]
)
router.add(
    "hello",
    r"\bhola\b",
    lambda t: (speak("¡Hola! ¿Cómo puedo ayudarte?"), "continue")[1]
)
sleep_regex = r"(modo de espera|entra en modo de espera|descansa|duerme)"

router.add(
    "sleep_mode",
    sleep_regex,
    lambda t: (
        speak("Entrando en modo de espera..."),
        vts_hotkey("HebeSleep"),
        "sleep"
    )[2]
)
router.add(
    "open_app",
    r"\babre\b",
    lambda t: (
        tools.call(
            "open_app",
            {"command": t},
            lambda: abrir_aplicacion(t)
        ),
        "continue"
    )[1]
)
router.add(
    "close_window",
    r"cierra ventana",
    lambda t: (
        tools.call(
            "close_window",
            {},
            lambda: win.close_active_window()
        ),
        "continue"
    )[1]
)
router.add(
    "volume_control",
    r"(sube volumen|baja volumen|silenciar)",
    lambda t: (
        tools.call(
            "volume",
            {"command": t},
            lambda: controlar_volumen(t)
        ),
        "continue"
    )[1]
)
router.add(
    "power_control",
    r"(apaga el ordenador|reinicia el ordenador)",
    lambda t: (
        controlar_pc(t),
        "continue"
    )[1]
)
router.add(
    "memory_store",
    r"(recuerda que|hebe recuerda que|eve recuerda que)",
    lambda t: (
        guardar_memoria_desde_comando(t),
        "continue"
    )[1]
)
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

def abrir_aplicacion(command_text: str):
    app = db_find_app_for_command(command_text)
    if not app:
        speak("No conozco esa aplicación todavía.")
        return
    win.open_app(app, speak=speak)

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

tools = ToolSystem(
    ToolContext(
        emit=emit,
        speak=speak,
        win=win,
        open_app_fn=abrir_aplicacion,
        volume_fn=controlar_volumen,
        power_fn=controlar_pc,
        memory_fn=guardar_memoria_desde_comando,
    )
)
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

def handle_command(command: str, source: str = "voice") -> str:
    text = (command or "").strip()
    if not text:
        return "continue"

    decision = router.route(text.lower())

    # Si alguna regla lo manejó:
    if decision in ("stop", "sleep", "continue"):
        return decision

    # ✅ Fallback: cualquier cosa fuera de comandos => LLM
    reply = obtener_respuesta_gpt(text)  # o tu ask_llm(...)
    speak(reply)
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

if __name__ == "__main__":
    # Modo standalone (sin backend): arranca Hebe y mantén vivo el proceso.
    stt.list_audio_devices()
    engine = HebeEngine(use_wakeword=True, say_hello=True)
    engine.start()
    while True:
        time.sleep(1)
