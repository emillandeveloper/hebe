import time
from app.services.db_sqlite import (
    init_db,
    log_chat,
    seed_default_apps,
)
from app.services.vts_client import vts_hotkey
from app.core.ui_bridge import emit
from app.core.input_bus import submit_text_from_ui, get_ui_inbox, get_voice_inbox
from app.core.stt_worker import STTWorker
from app.core.runtime import build_runtime, HebeRuntime
from queue import Empty
import threading

WAKE_WORDS = ["hebe despierta", "eve despierta", "jebe despierta", "asistente despierta"]

t0 = time.time()
def mark(stage):
    emit("status", {"engine":"starting","stage":stage,"t_ms": int((time.time()-t0)*1000)})

# =========================
#  MAIN
# =========================
class HebeEngine:
    """Motor de Hebe ejecutándose en un hilo, controlable desde el backend/UI."""
    def __init__(self, runtime: HebeRuntime, use_wakeword: bool = True, say_hello: bool = False):
        self.runtime = runtime
        self._stt_worker: STTWorker | None = None
        self.say_hello = say_hello
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self.use_wakeword = use_wakeword

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
                self.runtime.stt.init()
                self._stt_worker = STTWorker(stt=self.runtime.stt, stop_event=self._stop_event)
                self._stt_worker.start()
                emit("status", {"engine": "ready", "stage": "ready"})

                target = self.wakeword_loop if self.use_wakeword else self.engine_loop
                kwargs = {"say_hello": self.say_hello}

                self._thread = threading.Thread(target=target, kwargs=kwargs, daemon=True)
                self._thread.start()
                self.runtime.state.is_running = True
                self.runtime.state.mode = "wakeword" if self.use_wakeword else "active"
            except Exception as e:
                emit("status", {"engine": "error", "stage": "boot", "error": str(e)})

        threading.Thread(target=boot, daemon=True).start()

    def stop(self):
        self._stop_event.set()
        self.runtime.state.is_running = False
        self.runtime.state.is_processing = False
        self.runtime.state.mode = "stopped"

    def submit_text(self, text: str):
        print(f"[HEBE] submit_text: {text!r}", flush=True)
        submit_text_from_ui(text)

    def handle_command(self, command: str, source: str = "voice") -> str:
        print(f"[HEBE] handle_command source={source} text={command!r}", flush=True)
        text = (command or "").strip()
        if not text:
            return "continue"

        self.runtime.state.is_processing = True
        self.runtime.state.last_input_text = text
        self.runtime.state.last_input_source = source

        try:
            frame = self.runtime.intent_resolver.resolve(text, ctx=self.runtime.nlu_ctx, source=source)
            self.runtime.state.last_intent = frame.intent
            print(f"[HEBE] resolved frame={frame!r}", flush=True)
            result = self.runtime.dispatcher.dispatch(frame, source=source)
            print(f"[HEBE] dispatch result={result!r}", flush=True)
            return result
        finally:
            self.runtime.state.is_processing = False

    def command_loop(self) -> str:
        while True:
            if self._stop_event.is_set():
                return "stop"

            command = None
            source = None

            try:
                ui_inbox = get_ui_inbox()
                command = ui_inbox.get_nowait()
                print(f"[HEBE] UI inbox -> {command!r}", flush=True)
                source = "ui"
                command = str(command).strip().lower()
            except Empty:
                pass

            if not command:
                try:
                    voice_inbox = get_voice_inbox()
                    command = voice_inbox.get_nowait()
                    print(f"[HEBE] VOICE inbox -> {command!r}", flush=True)
                    source = "voice"
                    command = str(command).strip().lower()
                except Empty:
                    pass

            if not command:
                time.sleep(0.02)
                continue

            if source == "ui":
                log_chat("user", command, source="ui")
                emit("chat.user", {"text": command})

            res = self.handle_command(command, source=source)

            if res in ("sleep", "stop"):
                return res

    def wakeword_loop(self, say_hello: bool = True) -> str:
        self.runtime.state.mode = "sleep"
        if say_hello:
            self.runtime.speak("¡Hola! ¿Cómo puedo ayudarte?")

        while True:
            if self._stop_event.is_set():
                return "stop"

            try:
                ui_inbox = get_ui_inbox()
                cmd = ui_inbox.get_nowait()
                cmd = str(cmd).strip().lower()
                if cmd:
                    log_chat("user", cmd, source="ui")
                    emit("chat.user", {"text": cmd})
                    res = self.handle_command(cmd, source="ui")
                    if res == "stop":
                        return "stop"
                continue
            except Empty:
                pass

            try:
                voice_inbox = get_voice_inbox()
                command = voice_inbox.get_nowait()
                command = str(command).strip().lower()
            except Empty:
                time.sleep(0.02)
                continue

            if any(keyword in command for keyword in WAKE_WORDS):
                self.runtime.state.mode = "active"
                vts_hotkey("HebeIdle")
                self.runtime.speak("Te escucho.")
                res = self.command_loop()
                if res == "stop":
                    return "stop"

    def engine_loop(self, say_hello: bool = True) -> str:
        self.runtime.state.mode = "active"
        if say_hello:
            self.runtime.speak("¡Hola! ¿Cómo puedo ayudarte?")

        while True:
            if self._stop_event.is_set():
                return "stop"

            res = self.command_loop()
            if res == "stop":
                return "stop"
            if res == "sleep":
                self.runtime.state.mode = "sleep"
                res2 = self.wakeword_loop(say_hello=False)
                if res2 == "stop":
                    return "stop"
if __name__ == "__main__":
    runtime = build_runtime()
    engine = HebeEngine(runtime=runtime, use_wakeword=True, say_hello=True)
    engine.start()
    while True:
        time.sleep(1)