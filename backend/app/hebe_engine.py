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

from app.cognitive import MemoryStore, SchedulerService
from app.cognitive.context_builder import ContextBuilder
from app.cognitive.deliberation_service import DeliberationService
from app.cognitive.plan_executor import PlanExecutor
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.cognitive.action_runtime import ActionRuntime

WAKE_WORDS = ["hebe despierta", "eve despierta", "jebe despierta", "asistente despierta"]
STREAM_WAKE_ALIASES = {"hebe", "ebe", "eve", "heve", "jebe"}

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
        # -------------------------
        # Cognitive flow (Hebe v1)
        # -------------------------
        self.memory_store = MemoryStore()
        self.scheduler = SchedulerService(self.memory_store)
        self.context_builder = ContextBuilder(self.memory_store)

        # Modelos:
        # - intent: ya lo usas en legacy
        # - deliberation/summary: de momento reutilizo llm hasta separar runtime
        # - conversation/persona: de momento reutilizo llm hasta separar runtime
        self.deliberation_service = DeliberationService(
            intent_model=getattr(self.runtime, "intent_llm", None),
            reasoning_model=getattr(self.runtime, "llm", None),
        )

        self.action_runtime = ActionRuntime(self.runtime)

        self.plan_executor = PlanExecutor(
            memory_store=self.memory_store,
            action_runtime=self.action_runtime,
        )

        self.response_synthesizer = ResponseSynthesizer(
            conversation_model=getattr(self.runtime, "llm", None),
        )

        # Feature flag inicial
        self.use_cognitive_flow = True
        self.scheduler_poll_interval_sec = 1.0
        self._last_scheduler_poll_ts = 0.0

    def legacy_flow(self, command: str, source: str = "voice") -> str:
        result = self.orchestrator.handle(
            text=command,
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

        if self._should_speak_result(result):
            try:
                self.runtime.speak(spoken_text)
            except Exception as e:
                print(f"[HEBE] speak failed: {e!r}", flush=True)

        if result.intent == "sleep_mode" and result.success:
            return "sleep"

        if result.intent == "stop_engine" and result.success:
            return "stop"

        return "continue"    
    
    def cognitive_flow(self, command: str, source: str = "voice") -> str:
        context = self.context_builder.build(
            state=self.runtime.state,
            input_text=command,
            internal_event=None,
        )

        deliberation = self.deliberation_service.deliberate(context)
        execution = self.plan_executor.execute(deliberation.plan)
        reply_text = self.response_synthesizer.synthesize(
            context=context,
            deliberation=deliberation,
            execution=execution,
        )

        reply_step = execution.first_result_of_type("reply")
        if reply_step:
            mode = reply_step.data.get("mode")

            if mode == "clarify_appointment_datetime":
                self.runtime.state.pending_clarification = {
                    "kind": "appointment_datetime",
                    "draft": reply_step.data.get("draft", {}),
                }
            elif mode == "confirm_appointment":
                self.runtime.state.pending_clarification = None
        print(
            "[HEBE][COG] "
            f"reasoning={deliberation.plan.reasoning!r} "
            f"steps={[step.type for step in deliberation.plan.steps]!r} "
            f"reply={reply_text!r}",
            flush=True,
        )
        
        if reply_text:
            try:
                self.runtime.speak(reply_text)
            except Exception as e:
                print(f"[HEBE][COG] speak failed: {e!r}", flush=True)

        normalized = self._normalize_text(command)
        if normalized in {"duerme", "modo espera", "modo de espera"}:
            return "sleep"

        if normalized in {"apaga hebe", "detente", "stop engine"}:
            return "stop"

        return "continue"
    
    def process_internal_event(self, event) -> None:
        context = self.context_builder.build(
            state=self.runtime.state,
            input_text=None,
            internal_event=event,
        )

        deliberation = self.deliberation_service.deliberate(context)
        execution = self.plan_executor.execute(deliberation.plan)
        reply_text = self.response_synthesizer.synthesize(
            context=context,
            deliberation=deliberation,
            execution=execution,
        )

        print(
            "[HEBE][EVENT] "
            f"type={event.event_type!r} "
            f"reply={reply_text!r}",
            flush=True,
        )

        if reply_text:
            try:
                self.runtime.speak(reply_text)
            except Exception as e:
                print(f"[HEBE][EVENT] speak failed: {e!r}", flush=True)
    def poll_internal_events(self) -> None:
        now = time.time()
        if now - self._last_scheduler_poll_ts < self.scheduler_poll_interval_sec:
            return

        self._last_scheduler_poll_ts = now

        try:
            events = self.scheduler.poll_due_events(limit=10)
            for event in events:
                self.process_internal_event(event)
        except Exception as e:
            print(f"[HEBE][SCHEDULER] poll failed: {e!r}", flush=True)

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

                # Arranque de Twitch EventSub / lectura de chat y eventos
                if hasattr(self.runtime, "twitch_events") and self.runtime.twitch_events:
                    try:
                        self.runtime.twitch_events.start()
                    except Exception as e:
                        print(f"[HEBE][TWITCH][EVENTSUB] start failed: {e!r}", flush=True)

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

        if hasattr(self.runtime, "twitch_events") and self.runtime.twitch_events:
            try:
                self.runtime.twitch_events.stop()
            except Exception:
                pass

        self.runtime.state.is_running = False
        self.runtime.state.is_processing = False
        self.runtime.state.mode = "stopped"

    def submit_text(self, text: str):
        print(f"[HEBE] submit_text: {text!r}", flush=True)
        submit_text_from_ui(text)

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _get_stream_state(self):
        return getattr(self.runtime.state, "stream", None)

    def _is_stream_enabled(self) -> bool:
        stream = self._get_stream_state()
        return bool(stream and getattr(stream, "enabled", False))

    def _arm_stream(self) -> None:
        stream = self._get_stream_state()
        if not stream:
            return
        stream.armed = True
        timeout_sec = float(getattr(stream, "arm_timeout_sec", 8.0) or 8.0)
        stream.armed_until_ts = time.time() + timeout_sec

    def _disarm_stream(self) -> None:
        stream = self._get_stream_state()
        if not stream:
            return
        stream.armed = False
        stream.armed_until_ts = 0.0

    def _stream_is_armed(self) -> bool:
        stream = self._get_stream_state()
        if not stream:
            return False

        armed = bool(getattr(stream, "armed", False))
        if not armed:
            return False

        armed_until_ts = float(getattr(stream, "armed_until_ts", 0.0) or 0.0)
        if time.time() > armed_until_ts:
            self._disarm_stream()
            return False

        return True

    def _extract_stream_command(self, text: str) -> tuple[bool, str | None]:
        """
        Devuelve:
          (handled, command_to_execute)

        handled=True y command_to_execute=None:
          el input ya ha sido consumido por la lógica de stream
          y no se debe ejecutar nada ahora.

        handled=True y command_to_execute="...":
          ejecutar ese comando.

        handled=False:
          no era un caso de stream, continuar con flujo normal.
        """
        if not self._is_stream_enabled():
            return False, None

        normalized = self._normalize_text(text)
        if not normalized:
            return True, None

        parts = normalized.split(" ", 1)
        first_word = parts[0]
        rest = parts[1].strip() if len(parts) > 1 else ""

        # Caso: "hebe" / "eve" / etc. => armar ventana corta
        if first_word in STREAM_WAKE_ALIASES:
            if not rest:
                self._arm_stream()
                try:
                    vts_hotkey("HebeIdle")
                except Exception as e:
                    print(f"[HEBE] vts_hotkey failed while arming stream: {e!r}", flush=True)
                return True, None

            # Caso: "hebe haz shoutout a pepito"
            self._disarm_stream()
            return True, rest

        # Caso: ya está armada por haber dicho antes "Hebe"
        if self._stream_is_armed():
            self._disarm_stream()
            return True, normalized

        # Stream activo pero sin wakeword ni ventana armada => ignorar
        return True, None

    def _should_speak_result(self, result) -> bool:
        spoken_text = (result.output_text or "").strip()
        if not spoken_text:
            return False

        if spoken_text.lower() in {"continue", "stop", "sleep"}:
            return False

        if self._is_stream_enabled():
            if result.intent in {
                "stream_chat_message",
                "stream_shoutout",
                "stream_thank_raid",
            }:
                stream = self._get_stream_state()
                policies = getattr(stream, "policies", None) if stream else None
                return bool(getattr(policies, "allow_tts_replies", False))

        return True

    def handle_command(self, command: str, source: str = "voice") -> str:
        print(f"[HEBE] handle_command source={source} text={command!r}", flush=True)

        text = (command or "").strip()
        if not text:
            return "continue"

        return self.cognitive_flow(text, source=source)

    def command_loop(self) -> str:
        while True:
            if self._stop_event.is_set():
                return "stop"

            self.poll_internal_events()

            command = None
            source = None

            try:
                ui_inbox = get_ui_inbox()
                command = ui_inbox.get_nowait()
                print(f"[HEBE] UI inbox -> {command!r}", flush=True)
                source = "ui"
                command = self._normalize_text(str(command))
            except Empty:
                pass

            if not command:
                try:
                    voice_inbox = get_voice_inbox()
                    command = voice_inbox.get_nowait()
                    print(f"[HEBE] VOICE inbox -> {command!r}", flush=True)
                    source = "voice"
                    command = self._normalize_text(str(command))
                except Empty:
                    pass

            if not command:
                time.sleep(0.02)
                continue

            if source in {"voice", "ui"}:
                handled, stream_command = self._extract_stream_command(command)
                if handled:
                    if not stream_command:
                        continue
                    command = stream_command

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
            self.poll_internal_events()
            try:
                ui_inbox = get_ui_inbox()
                cmd = ui_inbox.get_nowait()
                cmd = self._normalize_text(str(cmd))

                if cmd:
                    handled, stream_command = self._extract_stream_command(cmd)
                    if handled:
                        if not stream_command:
                            continue
                        cmd = stream_command

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
                command = self._normalize_text(str(command))
            except Empty:
                time.sleep(0.02)
                continue

            # Si stream mode está activo, dejamos que command_loop gestione
            # el gate fino con wakeword corto tipo "hebe"/"eve".
            if self._is_stream_enabled():
                res = self.command_loop()
                if res == "stop":
                    return "stop"
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