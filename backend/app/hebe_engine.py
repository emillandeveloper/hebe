import time
from app.services.db_sqlite import (
    init_db,
    log_chat,
    seed_default_apps,
    find_app_for_command as db_find_app_for_command,
    add_memory,
    get_active_memories,
    save_app_command,
)
from app.services.vts_client import vts_hotkey
from app.services.speech_output import speak as _speak
from app.services.stt_whisper import STTService, STTConfig
from app.services.win_automation import WinAutomationService
from app.services.command_router import CommandRouter
from app.services.tool_system import ToolSystem, ToolContext
from app.services.llm_ollama import OllamaLLM
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

stt = STTService(
    config=STTConfig(),
    emit=emit,
    log_chat=log_chat,
)

llm = OllamaLLM(model="hebe", emit=emit, log_chat=log_chat)
wiki = WikiES(emit=emit)

WAKE_WORDS = ["hebe despierta", "eve despierta", "jebe despierta", "asistente despierta"]
SLEEP_PHRASES = ["a dormir", "modo espera", "descansa"]

def speak(text: str, language: str = "es") -> None:
    # Always notify UI + persist chat log when Hebe speaks
    return _speak(text=text, language=language, emit=emit, log_chat=log_chat)

win = WinAutomationService(emit=emit, speak=speak)
router = CommandRouter()

# =========================
#  ACCIONES / COMANDOS DEL PC
# =========================

def open_app_from_text(command_text: str):
    command_text = (command_text or "").strip().lower()
    app = db_find_app_for_command(command_text)
    if not app:
        speak("No conozco esa aplicación todavía.")
        return
    win.open_app(app)

# =========================
#  MEMORIA Y APPS POR VOZ
# =========================

def store_memory_from_text(command: str):
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

def respond_memory_recall():
    """Lee algunas memorias de la BD y las dice en voz alta."""
    mems = get_active_memories(limit=5)
    if not mems:
        speak("De momento no recuerdo nada especial que me hayas dicho.")
        return
    frases = [m["text"] for m in mems]
    respuesta = "Recuerdo algunas cosas: " + "; ".join(frases)
    speak(respuesta)


def learn_new_app():
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
def confirm_action(action: str) -> bool:
    speak(f"¿Seguro que quieres {action}? Di sí o no.")
    resp = stt.listen()
    if not resp:
        return False
    r = resp.strip().lower()
    return r in ("si", "sí") or r.startswith("si ") or r.startswith("sí ")

tools = ToolSystem(
    ToolContext(
        emit=emit,
        speak=speak,
        win=win,
        open_app_fn=open_app_from_text,
        volume_fn=win.handle_volume_command,
        power_fn=None,
        memory_fn=store_memory_from_text,
    )
)
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
        tools.call("open_app", {"command": t}),
        "continue"
    )[1]
)
router.add(
    "close_window",
    r"cierra ventana",
    lambda t: (
        tools.call("close_window", {}),
        "continue"
    )[1]
)
router.add(
    "ytmusic_controls",
    r"(pausa música|reproduce música|siguiente canción|canción anterior|anterior canción|silenciar música)",
    lambda t: (win.handle_youtube_music_command(t), "continue")[1]
)
router.add(
    "volume_control",
    r"(sube volumen|baja volumen|\bsilenciar\b)",
    lambda t: (
        tools.call("volume", {"command": t}),
        "continue"
    )[1]
)
router.add(
    "power_control",
    r"(apaga el ordenador|reinicia el ordenador)",
    lambda t: (win.handle_power_command(t, confirm_fn=confirm_action), "continue")[1]
)
router.add(
    "memory_store",
    r"(recuerda que|hebe recuerda que|eve recuerda que)",
    lambda t: (
        store_memory_from_text(t),
        "continue"
    )[1]
)

def handle_command(command: str, source: str = "voice") -> str:
    text = (command or "").strip()
    if not text:
        return "continue"

    decision = router.route(text.lower())

    # Si alguna regla lo manejó:
    if decision in ("stop", "sleep", "continue"):
        return decision

    # ✅ Fallback: cualquier cosa fuera de comandos => LLM
    reply = llm.ask(text)
    speak(reply)
    return "continue"

def command_loop(stop_event: threading.Event | None = None) -> str:
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

def wakeword_loop(stop_event: threading.Event | None = None, say_hello: bool = False) -> str:
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
        if any(keyword in cmd_norm for keyword in WAKE_WORDS):
            vts_hotkey("HebeIdle")
            speak("Te escucho.")
            res = command_loop(stop_event=stop_event)
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

                target = wakeword_loop if self.use_wakeword else command_loop
                kwargs = {"stop_event": self._stop_event}
                if target is wakeword_loop:
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