import time
import threading
from queue import Empty

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

from app.orchestrator.orchestrator import Orchestrator
from app.orchestrator.executor import OrchestratorExecutor
from app.orchestrator.policy import OrchestratorPolicy
from app.orchestrator.gates import OrchestratorGates
from app.orchestrator.intents.resolver import IntentResolver
from app.orchestrator.dispatcher import OrchestratorDispatcher
from app.orchestrator.tool_handlers import build_tool_handlers


WAKE_WORDS = ["hebe despierta", "eve despierta", "jebe despierta", "asistente despierta"]

t0 = time.time()


def mark(stage):
    emit("status", {"engine": "starting", "stage": stage, "t_ms": int((time.time() - t0) * 1000)})


class HebeEngine:
    """Motor principal de Hebe ejecutándose en un hilo."""

    def __init__(self, runtime: HebeRuntime, use_wakeword: bool = True, say_hello: bool = False):
        self.runtime = runtime
        self._stt_worker: STTWorker | None = None
        self.say_hello = say_hello
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self.use_wakeword = use_wakeword

        chat_runtime = getattr(self.runtime, "llm", None)

        dispatcher = OrchestratorDispatcher(
            runtime=self.runtime,
            tools=build_tool_handlers(self.runtime),
        )

        self.orchestrator = Orchestrator(
            state=self.runtime.state,
            intent_resolver=IntentResolver(
                llm=getattr(self.runtime, "llm", None),
            ),
            executor=OrchestratorExecutor(
                chat_runtime=chat_runtime,
                dispatcher=dispatcher,
            ),
            policy=OrchestratorPolicy(),
            gates=OrchestratorGates(),
        )

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
                self._stt_worker = STTWorker(
                    stt=self.runtime.stt,
                    stop_event=self._stop_event,
                )
                self._stt_worker.start()

                emit("status", {"engine": "ready", "stage": "ready"})

                target = self.wakeword_loop if self.use_wakeword else self.engine_loop
                kwargs = {"say_hello": self.say_hello}

                self._thread = threading.Thread(
                    target=target,
                    kwargs=kwargs,
                    daemon=True,
                )
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

        result = self.orchestrator.handle(
            text=text,
            source=source,
        )

        print(
            "[HEBE] orchestrator result "
            f"status={result.status.value!r} "
            f"success={result.success!r} "
            f"intent={result.intent!r} "
            f"text={result.output_text!r} "
            f"error={result.error!r}",
            flush=True,
        )

        spoken_text = (result.output_text or "").strip()

        # No verbalizar señales internas ni respuestas vacías
        if spoken_text and spoken_text.lower() not in {"continue", "stop", "sleep"}:
            try:
                self.runtime.speak(spoken_text)
            except Exception as e:
                print(f"[HEBE] speak failed: {e!r}", flush=True)

        if result.intent == "sleep_mode" and result.success:
            return "sleep"

        if result.intent == "stop_engine" and result.success:
            return "stop"

        return "continue"

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