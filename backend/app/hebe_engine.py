import time
import threading
from queue import Empty
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.services.db_sqlite import (
    DB_PATH,
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
from app.cognitive.memory.memory_extractor import MemoryExtractor

WAKE_WORDS = ["hebe despierta", "eve despierta", "jebe despierta"]
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

        # Conectar Twitch events al scheduler
        if hasattr(self.runtime, 'twitch_events') and self.runtime.twitch_events:
            self.runtime.twitch_events.push_event_callback = lambda event_type, payload: self.scheduler.push_event(event_type, payload)

        if hasattr(self.runtime, 'twitch_chat_bot') and self.runtime.twitch_chat_bot:
            def _twitch_chat_callback(username, display_name, text, channel):
                print(
                    f"[HEBE][TWITCH][CHATBOT] dispatching event twitch_chat_react user={username!r} channel={channel!r} message={text!r}",
                    flush=True,
                )
                self.scheduler.push_event(
                    "twitch_chat_react",
                    {
                        "display_name": display_name,
                        "user_login": username,
                        "message_text": text,
                        "channel": channel,
                        "recent_chat": [],
                    },
                )
                stream = self._get_stream_state()
                if stream is not None:
                    stream.last_chat_activity_ts = time.time()

            self.runtime.twitch_chat_bot.message_callback = _twitch_chat_callback

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
        self.memory_extractor = MemoryExtractor(
            intent_model=getattr(self.runtime, "intent_llm", None),
        )

        # Feature flag inicial
        self.use_cognitive_flow = True
        self.scheduler_poll_interval_sec = 1.0
        self._last_scheduler_poll_ts = 0.0
        self.presence_poll_interval_sec = 30.0
        self._last_presence_poll_ts = 0.0
        self.routine_poll_interval_sec = 30.0
        self._last_routine_poll_ts = 0.0

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
                self._deliver_voice_reply(spoken_text)
            except Exception as e:
                print(f"[HEBE] speak failed: {e!r}", flush=True)

        if result.intent == "sleep_mode" and result.success:
            return "sleep"

        if result.intent == "stop_engine" and result.success:
            return "stop"

        return "continue"    
    
    def cognitive_flow(self, command: str, source: str = "voice") -> str:
        print(
            "[HEBE][COG] incoming "
            f"source={source!r} "
            f"command={command!r} "
            f"current_pending={getattr(self.runtime.state, 'pending_clarification', None)!r}",
            flush=True,
        )

        manual = self._handle_tts_manual_command(command)
        if manual is None:
            manual = self._handle_stream_manual_command(command)
        if manual is not None:
            self._deliver_manual_reply(manual, source=source)
            return "continue"

        context = self.context_builder.build(
            state=self.runtime.state,
            input_text=command,
            internal_event=None,
        )

        print(
            "[HEBE][COG] context pending="
            f"{context.state_snapshot.get('pending_clarification')!r}",
            flush=True,
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

            print(
                "[HEBE][COG] reply_step "
                f"mode={mode!r} "
                f"data={reply_step.data!r}",
                flush=True,
            )

            if mode == "clarify_appointment_datetime":
                self.runtime.state.pending_clarification = {
                    "kind": "appointment_datetime",
                    "draft": reply_step.data.get("draft", {}),
                }

                print(
                    "[HEBE][STATE] saved pending_clarification="
                    f"{self.runtime.state.pending_clarification!r}",
                    flush=True,
                )

            elif mode == "confirm_appointment":
                self.runtime.state.pending_clarification = None

                print(
                    "[HEBE][STATE] cleared pending_clarification",
                    flush=True,
                )

        print(
            "[HEBE][COG] "
            f"reasoning={deliberation.plan.reasoning!r} "
            f"steps={[step.type for step in deliberation.plan.steps]!r} "
            f"reply={reply_text!r}",
            flush=True,
        )

        if source == "ui" and reply_text:
            log_chat("assistant", reply_text, source="ui")
            emit("chat.assistant", {"text": reply_text})

        if reply_text and self._should_extract_memory(source=source, execution=execution):
            try:
                self.memory_extractor.extract_and_store(
                    user_text=command,
                    assistant_reply=reply_text,
                    source=source,
                )
            except Exception as exc:
                print(f"[HEBE][MEMORY_EXTRACT] failed: {exc!r}", flush=True)

        if reply_text:
            try:
                if source != "ui":
                    self._deliver_voice_reply(reply_text)
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

        if not reply_text:
            return

        # Routing del reply según tipo de evento
        if event.event_type.startswith("twitch_"):
            self._deliver_twitch_reply(reply_text)
        else:
            self._deliver_voice_reply(reply_text)

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

    def poll_stream_presence(self) -> None:
        now = time.time()
        if now - self._last_presence_poll_ts < self.presence_poll_interval_sec:
            return
        self._last_presence_poll_ts = now

        stream = self._get_stream_state()
        if not stream or not getattr(stream, "enabled", False):
            return

        mode = getattr(stream, "presence_mode", "reactive") or "reactive"
        if mode not in {"companion", "show"}:
            print(f"[HEBE][PRESENCE] skipped reason=mode_{mode}", flush=True)
            return

        silence_minutes = 4.0 if mode == "companion" else 2.0
        cooldown_minutes = 12.0 if mode == "companion" else 6.0
        voice_quiet_sec = 20.0

        last_chat = float(getattr(stream, "last_chat_activity_ts", 0.0) or 0.0)
        if not last_chat:
            stream.last_chat_activity_ts = now
            print("[HEBE][PRESENCE] skipped reason=no_chat_baseline", flush=True)
            return
        if last_chat and now - last_chat < silence_minutes * 60:
            print("[HEBE][PRESENCE] skipped reason=chat_recently_active", flush=True)
            return

        last_spoken = float(getattr(stream, "last_hebe_stream_speak_ts", 0.0) or 0.0)
        if last_spoken and now - last_spoken < cooldown_minutes * 60:
            print("[HEBE][PRESENCE] skipped reason=cooldown", flush=True)
            return

        last_voice_ts = float(getattr(stream, "last_voice_event_ts", 0.0) or 0.0)
        if last_voice_ts and now - last_voice_ts < voice_quiet_sec:
            print("[HEBE][PRESENCE] skipped reason=leo_recently_spoke", flush=True)
            return

        print(f"[HEBE][PRESENCE] speaking reason=chat_silence mode={mode}", flush=True)
        reply = self.response_synthesizer.generate_stream_presence(
            reason="chat_silence",
            presence_mode=mode,
            last_voice_event=getattr(stream, "last_voice_event", None),
            leo_mood_hint=getattr(stream, "leo_mood_hint", None),
        )
        if reply:
            stream.last_hebe_stream_speak_ts = now
            self._deliver_twitch_reply(reply)

    def poll_stream_routine(self) -> None:
        now_ts = time.time()
        if now_ts - self._last_routine_poll_ts < self.routine_poll_interval_sec:
            return
        self._last_routine_poll_ts = now_ts

        stream = self._get_stream_state()
        if not stream:
            return

        now = datetime.now(ZoneInfo("Europe/Madrid"))
        today = now.date().isoformat()
        if getattr(stream, "no_stream_today_date", None) == today:
            return

        delay = int(getattr(stream, "stream_delay_minutes", 0) or 0)
        schedule = {
            "18:30": "Leo, en media hora tocaría preparar stream.",
            "18:50": "Si hoy hay directo, es buen momento para abrir OBS y el juego.",
            "19:00": "¿Activo modo stream?",
        }

        for base_time, message in schedule.items():
            due = self._today_at(base_time) + timedelta(minutes=delay)
            key = f"{today}:{base_time}:{delay}"
            sent = getattr(stream, "routine_sent_keys", set())
            if key in sent:
                continue
            if 0 <= (now - due).total_seconds() < self.routine_poll_interval_sec + 5:
                sent.add(key)
                stream.routine_sent_keys = sent
                self._deliver_voice_reply(message)

    def start(self):
        if self._started:
            return
        self._started = True

        def boot():
            try:
                emit("status", {"engine": "starting", "stage": "db"})
                print(f"[HEBE][DB] startup path={DB_PATH}", flush=True)
                init_db()
                # Tabla memory_chunks vive en su propio módulo para evitar
                # import circular (memory_store importa db_sqlite). Se inicializa
                # aquí, justo después de init_db(), en lugar de dentro de init_db().
                try:
                    from app.cognitive.memory.memory_store import init_memory_chunks_schema
                    init_memory_chunks_schema()
                except Exception as _e:
                    print(f"[HEBE][MEMORY] init_memory_chunks_schema failed: {_e!r}", flush=True)

                emit("status", {"engine": "starting", "stage": "apps"})
                seed_default_apps()

                emit("status", {"engine": "starting", "stage": "models"})
                if getattr(self.runtime, "stt_enabled", True):
                    self.runtime.stt.init()
                    self._stt_worker = STTWorker(
                        stt=self.runtime.stt,
                        stop_event=self._stop_event,
                    )
                    self._stt_worker.start()
                else:
                    print("[HEBE][STT] disabled by HEBE_STT_ENABLED", flush=True)

                # Arranque de Twitch EventSub / lectura de chat y eventos
                if hasattr(self.runtime, "twitch_events") and self.runtime.twitch_events:
                    try:
                        self.runtime.twitch_events.start()
                    except Exception as e:
                        print(f"[HEBE][TWITCH][EVENTSUB] start failed: {e!r}", flush=True)

                if hasattr(self.runtime, "twitch_chat_bot") and self.runtime.twitch_chat_bot:
                    try:
                        self.runtime.twitch_chat_bot.start()
                    except Exception as e:
                        print(f"[HEBE][TWITCH][CHATBOT] start failed: {e!r}", flush=True)

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

        if hasattr(self.runtime, "twitch_chat_bot") and self.runtime.twitch_chat_bot:
            try:
                self.runtime.twitch_chat_bot.stop()
            except Exception:
                pass

        self.runtime.state.is_running = False
        self.runtime.state.is_processing = False
        self.runtime.state.mode = "stopped"

    def submit_text(self, text: str):
        print(f"[HEBE] submit_text: {text!r}", flush=True)
        submit_text_from_ui(text)

    def _normalize_text(self, text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in (text or "").strip().lower())
        return " ".join(cleaned.split())

    def _today_at(self, hhmm: str) -> datetime:
        hour, minute = [int(part) for part in hhmm.split(":", 1)]
        now = datetime.now(ZoneInfo("Europe/Madrid"))
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

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

    def _classify_voice_event(self, text: str) -> tuple[str, str | None]:
        normalized = self._normalize_text(text)
        if not normalized:
            return "unknown", None
        if "hebe" in normalized or normalized.startswith(("prepara stream", "activa modo stream", "desactiva modo stream")):
            return "direct_command_to_hebe", None
        if any(marker in normalized for marker in ("me han matado", "he muerto", "otra vez", "wipe", "game over")):
            return "gameplay_failure", "frustrated"
        if any(marker in normalized for marker in ("jaj", "lol", "me parto")):
            return "laughter/joke", "playful"
        if any(marker in normalized for marker in ("joder", "mierda", "que sueño", "qué sueño", "estoy muerto", "cansado")):
            return "frustration", "tired"
        if len(normalized.split()) <= 10:
            return "casual_comment", None
        return "unknown", None

    def _record_voice_event(self, text: str, event_type: str, mood_hint: str | None) -> None:
        stream = self._get_stream_state()
        if not stream:
            return
        stream.last_voice_event = event_type
        stream.last_voice_event_ts = time.time()
        if mood_hint:
            stream.leo_mood_hint = mood_hint

    def _handle_stream_manual_command(self, text: str) -> str | None:
        stream = self._get_stream_state()
        if not stream:
            return None

        normalized = self._normalize_text(text)
        for prefix in ("hebe ", "ebe ", "eve ", "jebe "):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()

        presence_modes = {
            "modo silencioso": ("silent", "Modo silencioso, Leo."),
            "silent mode": ("silent", "Silent mode, Leo."),
            "modo reactiva": ("reactive", "Modo reactiva, Leo."),
            "modo reactivo": ("reactive", "Modo reactiva, Leo."),
            "reactive mode": ("reactive", "Reactive mode, Leo."),
            "modo compañera": ("companion", "Modo compañera, Leo."),
            "modo companera": ("companion", "Modo compañera, Leo."),
            "companion mode": ("companion", "Companion mode, Leo."),
            "modo show": ("show", "Modo show, Leo. Con correa, pero show."),
            "show mode": ("show", "Show mode, Leo."),
        }
        if normalized in presence_modes:
            mode, reply = presence_modes[normalized]
            stream.presence_mode = mode
            return reply

        if normalized in {"prepara stream", "prepare stream"}:
            stream.enabled = True
            stream.last_chat_activity_ts = time.time()
            stream.presence_mode = "reactive"
            stream.no_stream_today_date = None
            return "Dejo el stream preparado en modo reactiva. OBS y el juego solo si me lo confirmas."

        if normalized in {"activa modo stream", "activa stream", "stream on", "enable stream"}:
            stream.enabled = True
            stream.last_chat_activity_ts = time.time()
            return "Modo stream activado."

        if normalized in {"desactiva modo stream", "desactiva stream", "stream off", "disable stream"}:
            stream.enabled = False
            return "Modo stream desactivado."

        if normalized in {"hoy no hay stream", "no hay stream hoy", "today no stream"}:
            stream.no_stream_today_date = datetime.now(ZoneInfo("Europe/Madrid")).date().isoformat()
            stream.enabled = False
            return "Vale, hoy no insisto con el stream."

        if normalized in {"retrasa stream media hora", "retrasa el stream media hora", "delay stream half an hour"}:
            stream.stream_delay_minutes = int(getattr(stream, "stream_delay_minutes", 0) or 0) + 30
            return "Retraso el plan de stream media hora."

        return None

    def _handle_tts_manual_command(self, text: str) -> str | None:
        normalized = self._normalize_text(text)
        for prefix in ("hebe ", "ebe ", "eve ", "jebe "):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()

        global_off = {
            "desactiva tu voz",
            "apaga tu voz",
            "desactiva el tts",
            "solo texto",
            "callate la voz",
            "cállate la voz",
            "disable your voice",
            "text only",
            "disable tts",
        }
        if normalized in global_off:
            self.runtime.state.tts_enabled = False
            print("[HEBE][TTS] global=false", flush=True)
            return "Vale, Leo. Me quedo en texto."

        global_on = {
            "activa tu voz",
            "vuelve a hablar",
            "activa el tts",
            "modo voz",
            "enable your voice",
            "enable tts",
            "voice mode",
        }
        if normalized in global_on:
            self.runtime.state.tts_enabled = True
            print("[HEBE][TTS] global=true", flush=True)
            return "Lista, Leo. Vuelvo a hablar."

        stream = self._get_stream_state()
        policies = getattr(stream, "policies", None) if stream else None
        if policies is None:
            return None

        stream_off = {
            "silencia tu voz en stream",
            "desactiva tts en stream",
            "desactiva el tts en stream",
            "responde solo por chat",
            "disable stream tts",
            "text only on stream",
        }
        if normalized in stream_off:
            policies.allow_tts_replies = False
            print("[HEBE][TTS] stream=false", flush=True)
            return "Entendido. En stream responderé solo por chat."

        stream_on = {
            "puedes hablar en stream",
            "activa tts en stream",
            "activa el tts en stream",
            "vuelve a hablar en directo",
            "enable stream tts",
            "enable tts on stream",
        }
        if normalized in stream_on:
            policies.allow_tts_replies = True
            print("[HEBE][TTS] stream=true", flush=True)
            return "Vale. Si toca, también hablaré en stream."

        return None

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

    def _should_extract_memory(self, *, source: str, execution) -> bool:
        if source not in {"ui", "voice"}:
            return False

        reply_step = execution.first_result_of_type("reply") if execution else None
        if not reply_step:
            return False

        return reply_step.data.get("mode") == "chat"

    def _deliver_twitch_reply(self, text: str) -> None:
        """
        Entrega un reply al chat de Twitch.
        Si las policies del stream lo permiten, también lo hablamos por TTS.
        """
        twitch = getattr(self.runtime, "twitch", None)
        stream = getattr(self.runtime.state, "stream", None)
        if stream is not None:
            stream.last_hebe_stream_speak_ts = time.time()
        if twitch is not None and twitch.is_available():
            try:
                twitch.send_message(text)
            except Exception as e:
                print(f"[HEBE][EVENT][TWITCH] send_message failed: {e!r}", flush=True)
        else:
            print("[HEBE][EVENT][TWITCH] service not available, dropping chat reply", flush=True)

        policies = getattr(stream, "policies", None) if stream else None
        if not getattr(self.runtime.state, "tts_enabled", False):
            print("[HEBE][TTS] skipped reason=global_disabled", flush=True)
            return
        if not (policies and getattr(policies, "allow_tts_replies", False)):
            print("[HEBE][TTS] skipped reason=stream_tts_disabled", flush=True)
            return
        self._deliver_voice_reply(text)


    def _deliver_voice_reply(self, text: str) -> None:
        if not text:
            return
        if not getattr(self.runtime.state, "tts_enabled", False):
            emit("chat.assistant", {"text": text})
            print("[HEBE][TTS] skipped reason=global_disabled", flush=True)
            return
        try:
            self.runtime.speak(text)
        except Exception as e:
            print(f"[HEBE][EVENT] speak failed: {e!r}", flush=True)

    def _deliver_manual_reply(self, text: str, *, source: str) -> None:
        if source == "ui":
            log_chat("assistant", text, source="ui")
            emit("chat.assistant", {"text": text})
            return

        self._deliver_voice_reply(text)

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
            self.poll_stream_routine()
            self.poll_stream_presence()

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
                if source == "voice" and self._is_stream_enabled():
                    voice_type, mood_hint = self._classify_voice_event(command)
                    self._record_voice_event(command, voice_type, mood_hint)
                    print(
                        f"[HEBE][VOICE] type={voice_type} mood={mood_hint!r} text={command!r}",
                        flush=True,
                    )
                    if voice_type != "direct_command_to_hebe" and not self._stream_is_armed():
                        continue

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
            self._deliver_voice_reply("Ya estoy aquí, Leo.")

        while True:
            if self._stop_event.is_set():
                return "stop"
            self.poll_internal_events()
            self.poll_stream_routine()
            self.poll_stream_presence()
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
                self._deliver_voice_reply("Dime, Leo.")
                res = self.command_loop()
                if res == "stop":
                    return "stop"

    def engine_loop(self, say_hello: bool = True) -> str:
        self.runtime.state.mode = "active"

        if say_hello:
            self._deliver_voice_reply("Lista, Leo.")

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
