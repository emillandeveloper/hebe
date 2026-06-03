import os
import re
import time
import threading
from queue import Empty
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.services.db_sqlite import (
    DB_PATH,
    init_db,
    log_chat,
    seed_default_apps,
)
from app.services.vts_client import vts_hotkey
from app.services.voice_command_recovery import TranscriptNormalizationResult, normalize_stt_transcript
from app.core.ui_bridge import emit
from app.core.input_bus import submit_text_from_ui, submit_text_from_voice, get_ui_inbox, get_voice_inbox
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
from app.cognitive.scheduler import InternalEvent
from app.cognitive.command_result import CommandResult
from app.cognitive.input_event import InputEvent
from app.cognitive.action_plan import ActionPlan
from app.cognitive.context_builder import ContextBuilder
from app.cognitive.deliberation_service import DeliberationService
from app.cognitive.plan_executor import PlanExecutor
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.cognitive.action_runtime import ActionRuntime
from app.cognitive.memory.memory_extractor import MemoryExtractor
from app.stream.context_sync import StreamContextSyncService
from app.stream.game_profiles import GameProfileStore
from app.stream.game_research import GameKnowledgeResearchConfig, GameKnowledgeResearchService
from app.stream import memory as stream_memory
from app.stream.action_planner import StreamActionPlanner
from app.stream.spontaneity import StreamSpontaneityConfig, StreamSpontaneityService

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
            def _twitch_ambient_callback(username, display_name, text, channel):
                self.observe_twitch_chat_message(username, display_name, text, channel)

            def _twitch_chat_callback(username, display_name, text, channel):
                self.observe_twitch_chat_message(username, display_name, text, channel)
                stream = self._get_stream_state()
                recent_chat = list(getattr(stream, "recent_chat_messages", []) or [])[-10:] if stream is not None else []
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
                        "recent_chat": recent_chat,
                    },
                )
                if stream is not None:
                    stream.last_chat_activity_ts = time.time()

            self.runtime.twitch_chat_bot.ambient_message_callback = _twitch_ambient_callback
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
        self.game_profiles = GameProfileStore()
        self.game_research = GameKnowledgeResearchService(
            store=self.game_profiles,
            config=GameKnowledgeResearchConfig.from_env(),
        )
        self._last_game_research_category = None
        self.stream_spontaneity = StreamSpontaneityService(
            game_profiles=self.game_profiles,
            config=StreamSpontaneityConfig(
                companion_silence_sec=float(os.getenv("HEBE_COMPANION_IDLE_MIN_MINUTES", "20")) * 60,
                show_silence_sec=float(os.getenv("HEBE_SHOW_IDLE_MIN_MINUTES", "9")) * 60,
                companion_max_per_hour=int(os.getenv("HEBE_MAX_IDLE_PROMPTS_PER_HOUR_COMPANION", "2")),
                show_max_per_hour=int(os.getenv("HEBE_MAX_IDLE_PROMPTS_PER_HOUR_SHOW", "5")),
                max_per_stream=int(os.getenv("HEBE_MAX_IDLE_PROMPTS_PER_STREAM", "6")),
                save_equip_topic_cooldown_sec=float(os.getenv("HEBE_SAVE_EQUIP_TOPIC_COOLDOWN_MINUTES", "60")) * 60,
                chat_activity_window_sec=float(os.getenv("HEBE_CHAT_ACTIVITY_WINDOW_SECONDS", "180")),
                chat_active_message_threshold=int(os.getenv("HEBE_CHAT_ACTIVE_MESSAGE_THRESHOLD", "3")),
                chat_active_user_threshold=int(os.getenv("HEBE_CHAT_ACTIVE_USER_THRESHOLD", "1")),
                suppress_when_chat_active=os.getenv("HEBE_IDLE_SUPPRESS_WHEN_CHAT_ACTIVE", "true").strip().lower() in ("1", "true", "yes", "on"),
            ),
        )
        self.stream_spontaneity.start_grace_period(getattr(self.runtime.state, "stream", None))
        self.stream_context_sync = StreamContextSyncService(
            twitch_api=getattr(self.runtime, "twitch", None),
        )
        self.memory_extractor = MemoryExtractor(
            intent_model=getattr(self.runtime, "intent_llm", None),
        )
        self.stream_ambient_stt_enabled = os.getenv(
            "HEBE_STREAM_AMBIENT_STT_ENABLED",
            "false",
        ).strip().lower() in ("1", "true", "yes", "on")
        self.stream_ambient_stt_reply_immediately = os.getenv(
            "HEBE_STREAM_AMBIENT_STT_REPLY_IMMEDIATELY",
            "false",
        ).strip().lower() in ("1", "true", "yes", "on")
        self.stream_observe_chat = os.getenv(
            "HEBE_STREAM_OBSERVE_CHAT",
            "true",
        ).strip().lower() in ("1", "true", "yes", "on")
        self.chat_activity_window_sec = float(os.getenv("HEBE_CHAT_ACTIVITY_WINDOW_SECONDS", "180"))
        self.chat_active_message_threshold = int(os.getenv("HEBE_CHAT_ACTIVE_MESSAGE_THRESHOLD", "3"))
        self.chat_active_user_threshold = int(os.getenv("HEBE_CHAT_ACTIVE_USER_THRESHOLD", "1"))
        self.idle_suppress_when_chat_active = os.getenv(
            "HEBE_IDLE_SUPPRESS_WHEN_CHAT_ACTIVE",
            "true",
        ).strip().lower() in ("1", "true", "yes", "on")

        # Feature flag inicial
        self.use_cognitive_flow = True
        self.scheduler_poll_interval_sec = 1.0
        self._last_scheduler_poll_ts = 0.0
        self.presence_poll_interval_sec = 30.0
        self._last_presence_poll_ts = 0.0
        self.stream_context_poll_interval_sec = float(os.getenv("HEBE_STREAM_CONTEXT_SYNC_SEC", "90"))
        self._last_stream_context_poll_ts = 0.0
        self.routine_poll_interval_sec = 30.0
        self._last_routine_poll_ts = 0.0
        self._manual_reply_ui_only = False
        self.auto_enable_stream_when_live = os.getenv(
            "HEBE_AUTO_ENABLE_STREAM_WHEN_LIVE",
            "true",
        ).strip().lower() in ("1", "true", "yes", "on")
        self.default_live_presence_mode = os.getenv(
            "HEBE_DEFAULT_LIVE_PRESENCE_MODE",
            "companion",
        ).strip().lower() or "companion"
        self.auto_shoutout_raiders = os.getenv(
            "HEBE_AUTO_SHOUTOUT_RAIDERS",
            "true",
        ).strip().lower() in ("1", "true", "yes", "on")
        self.shoutout_cooldown_seconds = float(os.getenv("HEBE_SHOUTOUT_COOLDOWN_SECONDS", "120") or 120)
        self.shoutout_allow_bots = os.getenv(
            "HEBE_SHOUTOUT_ALLOW_BOTS",
            "false",
        ).strip().lower() in ("1", "true", "yes", "on")
        self.shoutout_blocked_users = self._load_shoutout_blocked_users()
        self.voice_command_confirm_ambiguous = os.getenv(
            "HEBE_VOICE_COMMAND_CONFIRM_AMBIGUOUS",
            "true",
        ).strip().lower() in ("1", "true", "yes", "on")
        self.stream_action_planner = self._build_stream_action_planner()
        self._current_input_event: InputEvent | None = None

    def _build_stream_action_planner(self) -> StreamActionPlanner:
        return StreamActionPlanner(
            known_targets_provider=self._known_voice_command_targets,
            normalize_target=self._normalize_shoutout_target,
            build_shoutout_command=self._build_shoutout_command_preview,
            stream_state_provider=self._get_stream_state,
        )

    def observe_twitch_chat_message(self, username: str, display_name: str, text: str, channel: str = "") -> None:
        if not getattr(self, "stream_observe_chat", True):
            return
        stream = self._get_stream_state()
        if not stream or self._is_chat_bot_user(username):
            return

        message = str(text or "").strip()
        if not message:
            return

        now = time.time()
        recent_existing = list(getattr(stream, "recent_chat_messages", []) or [])
        if recent_existing:
            last = recent_existing[-1]
            if (
                str(last.get("username") or "").lower() == str(username or "").lower()
                and str(last.get("text") or "") == message[:180]
                and now - float(last.get("ts", 0.0) or 0.0) < 1.0
            ):
                return
        stream.last_chat_activity_ts = now
        session_id = self._ensure_stream_memory_session_if_live(stream)
        stream_memory.record_chat_message(
            username=username,
            display_name=display_name,
            message_text=message,
            stream_session_id=session_id,
            is_mention_to_hebe=self._message_mentions_hebe(message),
            is_direct_reply_to_hebe=False,
            is_bot=False,
            source="twitch_irc",
            topic_hint=self._classify_chat_topic(message),
        )
        entry = {
            "username": str(username or "").strip(),
            "display_name": str(display_name or username or "").strip(),
            "text": message[:180],
            "ts": now,
            "channel": channel,
            "topic": self._classify_chat_topic(message),
        }
        messages = recent_existing
        messages.append(entry)
        stream.recent_chat_messages = messages[-50:]

        users = []
        for item in stream.recent_chat_messages:
            user = str(item.get("username") or "").strip()
            if user and user.lower() not in {u.lower() for u in users}:
                users.append(user)
        stream.recent_active_users = users[-20:]

        topics = [item.get("topic") for item in stream.recent_chat_messages if item.get("topic")]
        stream.recent_chat_topics = topics[-12:]
        if topics:
            stream.recent_chat_summary = ", ".join(dict.fromkeys(topics[-5:]))

    def _is_chat_bot_user(self, username: str) -> bool:
        user = (username or "").strip().lower().lstrip("@")
        if not user:
            return True
        stream = self._get_stream_state()
        bot_names = {
            "hebenifelheim",
            "jotunbot",
            "streamelements",
            "nightbot",
            "moobot",
            "fossabot",
            "streamlabs",
            (getattr(stream, "bot_username", "") or "").lower(),
            (getattr(getattr(self.runtime, "twitch", None), "bot_username", "") or "").lower(),
            (getattr(getattr(self.runtime, "twitch_chat_bot", None), "bot_username", "") or "").lower(),
        }
        configured = os.getenv("HEBE_TWITCH_BOT_USERNAMES", "")
        bot_names.update(part.strip().lower().lstrip("@") for part in configured.split(",") if part.strip())
        return user in bot_names

    def _load_shoutout_blocked_users(self) -> set[str]:
        configured = os.getenv(
            "HEBE_SHOUTOUT_BLOCKED_USERS",
            "hebenifelheim,jotunbot,streamelements,nightbot",
        )
        blocked = {
            "hebenifelheim",
            "jotunbot",
            "streamelements",
            "nightbot",
            "moobot",
            "fossabot",
            "streamlabs",
        }
        blocked.update(part.strip().lower().lstrip("@") for part in configured.split(",") if part.strip())
        stream = self._get_stream_state()
        twitch = getattr(self.runtime, "twitch", None)
        for value in (
            getattr(stream, "bot_username", "") if stream else "",
            getattr(twitch, "bot_username", "") if twitch else "",
            getattr(getattr(self.runtime, "twitch_chat_bot", None), "bot_username", ""),
        ):
            if value:
                blocked.add(str(value).strip().lower().lstrip("@"))
        return blocked

    def _normalize_shoutout_target(self, target: str) -> str:
        twitch = getattr(self.runtime, "twitch", None)
        normalize = getattr(twitch, "normalize_twitch_username", None)
        if callable(normalize):
            return normalize(target)
        value = re.sub(r"\s+", "", str(target or "").strip().lstrip("@"))
        return value if re.fullmatch(r"[A-Za-z0-9_]{3,25}", value) else ""

    def _resolve_shoutout_target(self, raw_target: str | None, *, allow_last_raider: bool = True) -> tuple[str | None, str | None]:
        raw = str(raw_target or "").strip()
        normalized_raw = self._normalize_text(raw)
        stream = self._get_stream_state()

        last_raider_markers = {
            "",
            "al ultimo raider",
            "al ultimo raider",
            "ultimo raider",
            "último raider",
            "a quien nos ha raideado",
            "quien nos ha raideado",
            "al que nos ha raideado",
            "last raider",
            "the last raider",
        }
        if allow_last_raider and normalized_raw in last_raider_markers:
            raid = getattr(stream, "last_raid_event", None) if stream else None
            target = (raid or {}).get("user_login") or (raid or {}).get("display_name")
            if target:
                return self._normalize_shoutout_target(target), None
            return None, "missing_target"

        target = raw.strip().lstrip("@").strip()
        twitch = getattr(self.runtime, "twitch", None)
        resolve_user = getattr(twitch, "resolve_user", None)
        if callable(resolve_user) and target:
            try:
                resolved = resolve_user(target)
                if resolved:
                    target = resolved
            except Exception as exc:
                print(f"[HEBE][TWITCH][SO] target resolver failed target={target!r} error={exc!r}", flush=True)

        if stream is not None and target:
            target_norm = self._normalize_text(target)
            for item in reversed(list(getattr(stream, "recent_chat_messages", []) or [])):
                for key in ("username", "display_name"):
                    candidate = str(item.get(key) or "").strip()
                    if candidate and self._normalize_text(candidate) == target_norm:
                        target = candidate
                        break

        normalized = self._normalize_shoutout_target(target)
        if not normalized:
            return None, "invalid_target"
        return normalized, None

    def _shoutout_block_reason(self, target: str, *, explicit_self: bool = False) -> str | None:
        normalized = self._normalize_shoutout_target(target)
        if not normalized:
            return "invalid_target"
        lowered = normalized.lower()
        blocked = set(getattr(self, "shoutout_blocked_users", set()) or set()) | self._load_shoutout_blocked_users()
        if not getattr(self, "shoutout_allow_bots", False) and lowered in blocked:
            return "blocked_bot_user"
        twitch = getattr(self.runtime, "twitch", None)
        channel = str(getattr(twitch, "channel_name", "") or "").strip().lower()
        if channel and lowered == channel and not explicit_self:
            return "own_channel"
        return None

    def _send_shoutout(self, target: str, *, source: str, force: bool = False, explicit_self: bool = False) -> tuple[bool, str, str]:
        stream = self._get_stream_state()
        normalized = self._normalize_shoutout_target(target)
        reason = self._shoutout_block_reason(normalized, explicit_self=explicit_self)
        if reason:
            if stream is not None:
                stream.last_shoutout_error = reason
            print(f"[HEBE][TWITCH][SO] blocked reason={reason} target={target}", flush=True)
            return False, normalized, reason

        now = time.time()
        cooldowns = getattr(stream, "shoutout_cooldowns", {}) if stream else {}
        last_ts = float((cooldowns or {}).get(normalized.lower(), 0.0) or 0.0)
        if not force and last_ts and now - last_ts < float(getattr(self, "shoutout_cooldown_seconds", 120) or 120):
            reason = "cooldown_active"
            if stream is not None:
                stream.last_shoutout_error = reason
            print(f"[HEBE][TWITCH][SO] blocked reason={reason} target={normalized}", flush=True)
            return False, normalized, reason

        twitch = getattr(self.runtime, "twitch", None)
        shoutout = getattr(twitch, "shoutout", None)
        try:
            if callable(shoutout):
                ok = bool(shoutout(normalized))
                command = getattr(twitch, "build_shoutout_command", lambda user: f"!so {user}")(normalized)
            else:
                template = os.getenv("HEBE_SHOUTOUT_COMMAND_TEMPLATE", "!so {username}") or "!so {username}"
                command = template.format(username=normalized)
                ok = bool(twitch and twitch.send_message(command))
            if not ok:
                raise RuntimeError("Twitch shoutout command returned false")
            if stream is not None:
                stream.last_shoutout_target = normalized
                stream.last_shoutout_ts = now
                stream.last_shoutout_error = None
                if not isinstance(getattr(stream, "shoutout_cooldowns", None), dict):
                    stream.shoutout_cooldowns = {}
                stream.shoutout_cooldowns[normalized.lower()] = now
            print(f"[HEBE][TWITCH][SO] sent command={command!r}", flush=True)
            return True, normalized, "sent"
        except Exception as exc:
            reason = f"send_failed: {type(exc).__name__}: {exc}"
            if stream is not None:
                stream.last_shoutout_error = reason
            print(f"[HEBE][TWITCH][SO] blocked reason={reason} target={normalized}", flush=True)
            return False, normalized, reason

    def _maybe_auto_shoutout_raider(self, target: str, *, force: bool = False) -> None:
        stream = self._get_stream_state()
        if not bool(getattr(self, "auto_shoutout_raiders", True)) and not force:
            print(f"[HEBE][TWITCH][SO] blocked reason=auto_disabled target={target}", flush=True)
            return
        if not force and not (stream and getattr(stream, "is_live", False)):
            print(f"[HEBE][TWITCH][SO] blocked reason=stream_offline target={target}", flush=True)
            return
        print(f"[HEBE][TWITCH][SO] auto shoutout planned target={target}", flush=True)
        ok, normalized, reason = self._send_shoutout(target, source="raid", force=force)
        if not ok:
            print(f"[HEBE][TWITCH][SO] auto shoutout failed reason={reason} target={normalized or target}", flush=True)

    def _classify_chat_topic(self, text: str) -> str:
        normalized = self._normalize_text(text)
        if any(word in normalized for word in ("linux", "ram", "servidor", "server", "pc", "windows", "obs")):
            return "tech_pc"
        if any(word in normalized for word in ("ff9", "final fantasy", "level 1", "boss", "jefe", "lindblum", "ramuh")):
            return "game"
        if any(word in normalized for word in ("hola", "buenas", "hello")):
            return "greeting"
        return "general_chat"

    def _message_mentions_hebe(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if re.search(r"\b(?:hebe|ebe|eve|heve|jebe)\b", normalized):
            return True
        stream = self._get_stream_state()
        candidates = [
            getattr(stream, "bot_username", "") if stream else "",
            getattr(getattr(self.runtime, "twitch", None), "bot_username", ""),
            getattr(getattr(self.runtime, "twitch_chat_bot", None), "bot_username", ""),
        ]
        return any(candidate and re.search(rf"(?<![a-z0-9_])@?{re.escape(candidate.lower().lstrip('@'))}(?![a-z0-9_])", normalized) for candidate in candidates)

    def _ensure_stream_memory_session_if_live(self, stream=None) -> int | None:
        stream = stream or self._get_stream_state()
        if not stream or not (getattr(stream, "is_live", False) or getattr(stream, "enabled", False)):
            return getattr(stream, "active_stream_session_id", None) if stream else None
        try:
            return stream_memory.ensure_active_stream_session(stream, source="engine")
        except Exception as exc:
            print(f"[HEBE][STREAM_MEMORY] ensure session failed: {exc!r}", flush=True)
            return None

    def _chat_activity_snapshot(self, stream=None, *, now: float | None = None) -> dict:
        stream = stream or self._get_stream_state()
        now = time.time() if now is None else float(now)
        window = float(getattr(self, "chat_activity_window_sec", 180) or 180)
        messages = [
            item for item in list(getattr(stream, "recent_chat_messages", []) or [])
            if now - float(item.get("ts", 0.0) or 0.0) <= window
        ] if stream else []
        users = {
            str(item.get("username") or "").strip().lower()
            for item in messages
            if str(item.get("username") or "").strip()
        }
        topics = [item.get("topic") for item in messages if item.get("topic")]
        active = (
            len(messages) >= int(getattr(self, "chat_active_message_threshold", 3) or 3)
            and len(users) >= int(getattr(self, "chat_active_user_threshold", 1) or 1)
        )
        return {
            "active": active,
            "count": len(messages),
            "users": sorted(users),
            "topics": topics[-8:],
            "summary": ", ".join(dict.fromkeys(topics[-5:])) if topics else "sin tema reciente",
            "window_sec": window,
        }

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

        manual = self._handle_pending_manual_intent(command)
        if manual is None:
            manual = self._handle_tts_manual_command(command)
        if manual is None:
            manual = self._handle_stream_manual_command(command)
        if manual is not None:
            force_ui = bool(getattr(self, "_manual_reply_ui_only", False))
            self._manual_reply_ui_only = False
            if isinstance(manual, CommandResult):
                manual_text = self._synthesize_command_result(manual, input_text=command)
            else:
                manual_text = str(manual)
            self._deliver_manual_reply(manual_text, source="ui" if force_ui else source)
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
                self.runtime.state.pending_reminder = self.runtime.state.pending_clarification

                print(
                    "[HEBE][STATE] saved pending_clarification="
                    f"{self.runtime.state.pending_clarification!r}",
                    flush=True,
                )

            elif mode == "confirm_appointment":
                self.runtime.state.pending_clarification = None
                self.runtime.state.pending_reminder = None

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
        if getattr(event, "event_type", None) in {"stream_online", "stream_offline"}:
            self._handle_stream_lifecycle_event(event)
            return
        if getattr(event, "event_type", None) == "twitch_raid":
            self._handle_twitch_raid_event(event)
            return

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
        if event.event_type == "twitch_idle_prompt":
            stream = self._get_stream_state()
            service = getattr(self, "stream_spontaneity", None)
            if service is not None and service.is_too_similar_to_recent(stream, reply_text):
                reply_text = self._idle_fallback_for_topic((getattr(event, "payload", {}) or {}).get("idle_topic"))

        if event.event_type.startswith("twitch_"):
            self._deliver_twitch_reply(reply_text, event_type=event.event_type, payload=getattr(event, "payload", {}) or {})
        else:
            self._deliver_voice_reply(reply_text)

    def _handle_stream_lifecycle_event(self, event) -> None:
        stream = self._get_stream_state()
        if not stream:
            return
        payload = getattr(event, "payload", {}) or {}
        now = time.time()
        if event.event_type == "stream_online":
            stream.is_live = True
            stream.live_status_known = True
            stream.last_stream_live_transition = "online"
            stream.last_stream_live_transition_ts = now
            stream.stream_started_at = payload.get("started_at") or payload.get("started_at_ts") or stream.stream_started_at
            stream.idle_prompts_sent_stream = 0
            stream.recent_idle_messages = []
            self._auto_enable_stream_if_live(stream, source="stream_online_event")
            self._ensure_stream_memory_session_if_live(stream)
            stream_memory.record_stream_event("stream_online", payload, stream=stream)
            self.poll_stream_context(force=True, require_enabled=False)
            print("[HEBE][STREAM_CONTEXT] stream_online event handled", flush=True)
            return
        if event.event_type == "stream_offline":
            stream.is_live = False
            stream.live_status_known = True
            stream.last_stream_live_transition = "offline"
            stream.last_stream_live_transition_ts = now
            stream.stream_started_at = None
            stream.stream_spontaneity_grace_until_ts = 0.0
            stream.idle_prompts_sent_stream = 0
            if isinstance(getattr(stream, "cooldowns", None), dict):
                stream.cooldowns.pop(getattr(self.stream_spontaneity.config, "cooldown_key", "stream_idle_prompt_next_ts"), None)
            stream_memory.record_stream_event("stream_offline", payload, stream=stream)
            stream_memory.close_active_stream_session(stream, reason="stream_offline_event")
            print("[HEBE][STREAM_CONTEXT] stream_offline event handled", flush=True)

    def _auto_enable_stream_if_live(self, stream, *, source: str) -> bool:
        if not stream or not getattr(stream, "is_live", False):
            return False
        if not bool(getattr(self, "auto_enable_stream_when_live", True)):
            return False

        changed = False
        if not getattr(stream, "enabled", False):
            stream.enabled = True
            changed = True

        mode = (getattr(stream, "presence_mode", "reactive") or "reactive").strip().lower()
        explicit = bool(getattr(stream, "presence_mode_explicit", False))
        default_mode = getattr(self, "default_live_presence_mode", "companion") or "companion"
        if mode == "reactive" and not explicit:
            stream.presence_mode = default_mode
            changed = True

        if changed or source == "stream_online_event":
            self.stream_spontaneity.start_grace_period(stream)
        if changed:
            print("[HEBE][STREAM] auto-enabled stream mode because Twitch is live", flush=True)
        else:
            print(f"[HEBE][STREAM] Twitch live confirmed; stream mode already enabled source={source}", flush=True)
        return changed

    def _handle_twitch_raid_event(self, event) -> None:
        stream = self._get_stream_state()
        payload = getattr(event, "payload", {}) or {}
        username = payload.get("display_name") or payload.get("user_login") or "alguien"
        viewers = int(payload.get("viewer_count") or 0)
        print(f"[HEBE][TWITCH][RAID] received from={username} viewers={viewers}", flush=True)
        if stream is not None:
            self._ensure_stream_memory_session_if_live(stream)
            stream.last_raid_event = {
                "display_name": username,
                "user_login": payload.get("user_login") or username,
                "viewer_count": viewers,
                "ts": time.time(),
            }
            stream_memory.record_stream_event("twitch_raid", payload, stream=stream)
            stream_memory.observe_presence(
                payload.get("user_login") or username,
                username,
                stream_session_id=getattr(stream, "active_stream_session_id", None),
                source="raid",
            )

        if not stream:
            print("[HEBE][TWITCH][RAID] blocked reason=no_stream_state", flush=True)
            return
        if not (getattr(stream, "enabled", False) or getattr(stream, "is_live", False)):
            print("[HEBE][TWITCH][RAID] blocked reason=stream_not_enabled_and_not_live", flush=True)
            return

        print("[HEBE][TWITCH][RAID] planned thank-you", flush=True)
        reply_text = self._synthesize_internal_event_reply(event)
        if not reply_text:
            print("[HEBE][TWITCH][RAID] blocked reason=empty_reply", flush=True)
            return
        self._deliver_twitch_reply(reply_text, event_type="twitch_raid", payload=payload)
        print("[HEBE][TWITCH][RAID] sent thank-you", flush=True)
        self._maybe_auto_shoutout_raider(payload.get("user_login") or username, force=bool(payload.get("_force_shoutout")))

    def _synthesize_internal_event_reply(self, event) -> str:
        context = self.context_builder.build(
            state=self.runtime.state,
            input_text=None,
            internal_event=event,
        )

        deliberation = self.deliberation_service.deliberate(context)
        execution = self.plan_executor.execute(deliberation.plan)
        return self.response_synthesizer.synthesize(
            context=context,
            deliberation=deliberation,
            execution=execution,
        )

    def _idle_fallback_for_topic(self, topic: str | None) -> str:
        fallbacks = {
            "challenge_comment": "En una run de desafio, la paciencia tambien cuenta como recurso.",
            "jrpg_trope": "Esto es muy JRPG: una puerta normal puede esconder una decision absurda.",
            "game_vibe": "La energia aqui pide prudencia elegante y una pizca de caos controlado.",
            "light_roast": "Leo, esa confianza tuya tiene pinta de necesitar supervision adulta.",
            "exploration_comment": "Si hay un camino secundario, el cofre imaginario ya esta gritando.",
            "strategy_without_spoilers": "Mi voto: piensa un turno mas antes de hacer algo heroicamente estupido.",
            "streamer_reaction_hook": "Esto tiene pinta de momento para mirar dos veces antes de tocar nada.",
            "hydration_or_break": "Trago de agua, postura decente, y luego ya seguimos tentando al destino.",
            "save_reminder": "Si puedes guardar, guarda. La epica no paga facturas.",
            "equipment_check": "Antes de avanzar, una mirada al equipo nunca arruina una leyenda.",
            "resource_management": "Recursos primero, valentia despues. Ese orden salva runs.",
        }
        return fallbacks.get(topic or "", "Esto pide calma, mala idea medida y cero spoilers.")

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

    def poll_stream_context(self, *, force: bool = False, require_enabled: bool = True) -> bool:
        now = time.time()
        if not force and now - self._last_stream_context_poll_ts < self.stream_context_poll_interval_sec:
            return False
        self._last_stream_context_poll_ts = now

        stream = self._get_stream_state()
        if not stream:
            print("[HEBE][STREAM_CONTEXT] refresh skipped reason=no_stream_state", flush=True)
            return False
        if require_enabled and not getattr(stream, "enabled", False):
            print("[HEBE][STREAM_CONTEXT] refresh skipped reason=stream_mode_disabled", flush=True)
            return False

        service = getattr(self, "stream_context_sync", None)
        print(
            "[HEBE][STREAM_CONTEXT] poll "
            f"force={force} require_enabled={require_enabled} "
            f"service_exists={service is not None}",
            flush=True,
        )
        if service is None:
            service = StreamContextSyncService(twitch_api=getattr(self.runtime, "twitch", None))
            self.stream_context_sync = service
            print("[HEBE][STREAM_CONTEXT] sync service created from runtime.twitch", flush=True)

        print("[HEBE][STREAM_CONTEXT] about to call Helix via context sync", flush=True)
        was_known = bool(getattr(stream, "live_status_known", False))
        was_live = bool(getattr(stream, "is_live", False))
        ok = bool(service.sync(stream))
        if ok:
            is_live = bool(getattr(stream, "is_live", False))
            if is_live and (not was_known or not was_live):
                stream.last_stream_live_transition = "online"
                stream.last_stream_live_transition_ts = now
            elif was_live and not is_live:
                stream.last_stream_live_transition = "offline"
                stream.last_stream_live_transition_ts = now
            if is_live:
                self._auto_enable_stream_if_live(stream, source="context_sync")
                self._ensure_stream_memory_session_if_live(stream)
            elif was_live:
                stream_memory.close_active_stream_session(stream, reason="context_sync_offline")
            self._maybe_research_game_after_context_sync(stream)
        print(f"[HEBE][STREAM_CONTEXT] refresh result success={ok}", flush=True)
        return ok

    def _maybe_research_game_after_context_sync(self, stream) -> None:
        service = getattr(self, "game_research", None)
        if service is None or not getattr(service.config, "enabled", False):
            return
        category = getattr(stream, "current_category", None) or getattr(stream, "current_game", None)
        if not category or category == getattr(self, "_last_game_research_category", None):
            return
        self._last_game_research_category = category
        ok, profile, reason = service.maybe_research_on_category_change(
            current_category=getattr(stream, "current_category", None),
            current_title=getattr(stream, "current_stream_title", None),
            current_game=getattr(stream, "current_game", None),
        )
        print(
            f"[HEBE][GAME_RESEARCH] category_check ok={ok} reason={reason} profile={profile.game_slug}",
            flush=True,
        )

    def poll_stream_presence(self) -> None:
        now = time.time()
        if now - self._last_presence_poll_ts < self.presence_poll_interval_sec:
            return
        self._last_presence_poll_ts = now

        if bool(getattr(self.runtime.state, "is_processing", False)):
            return

        stream = self._get_stream_state()
        if stream is not None:
            context_updated_ts = float(getattr(stream, "stream_context_updated_ts", 0.0) or 0.0)
            if not context_updated_ts or now - context_updated_ts > 120:
                self.poll_stream_context(force=True, require_enabled=False)

        service = getattr(self, "stream_spontaneity", None)
        if service is None:
            service = StreamSpontaneityService()
            self.stream_spontaneity = service

        event = service.build_due_event(stream)
        if event is None:
            return

        print(
            "[HEBE][PRESENCE] enqueue "
            f"type={event.event_type!r} mode={event.payload.get('presence_mode')!r}",
            flush=True,
        )
        self.process_internal_event(event)

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

                stream = getattr(self.runtime.state, "stream", None)
                policies = getattr(stream, "policies", None) if stream else None
                emit(
                    "status",
                    {
                        "engine": "ready",
                        "stage": "ready",
                        "tts_enabled": bool(getattr(self.runtime.state, "tts_enabled", False)),
                        "stream_tts_enabled": bool(getattr(policies, "allow_tts_replies", False)),
                        "stt_enabled": bool(getattr(self.runtime, "stt_enabled", False)),
                    },
                )

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

    def _known_voice_command_targets(self) -> list[str]:
        stream = self._get_stream_state()
        values: list[str] = []

        def add(value) -> None:
            text = str(value or "").strip().lstrip("@")
            if text and text.lower() not in {item.lower() for item in values}:
                values.append(text)

        if stream is not None:
            for item in list(getattr(stream, "recent_chat_messages", []) or []):
                add(item.get("username"))
                add(item.get("display_name"))
            for user in list(getattr(stream, "recent_active_users", []) or []):
                add(user)
            raid = getattr(stream, "last_raid_event", None) or {}
            add(raid.get("user_login"))
            add(raid.get("display_name"))
            add(getattr(stream, "last_shoutout_target", None))

        twitch = getattr(self.runtime, "twitch", None)
        cache = getattr(twitch, "target_cache", None)
        if isinstance(cache, dict):
            for key, value in cache.items():
                add(key)
                add(value)

        try:
            for name in stream_memory.list_recent_chatter_names(limit=80):
                add(name)
        except Exception:
            pass

        return values[-120:]

    def _normalize_stt_input(self, raw_text: str) -> TranscriptNormalizationResult:
        result = normalize_stt_transcript(raw_text, known_targets=self._known_voice_command_targets())
        self._record_stt_normalization(result)
        return result

    def _record_stt_normalization(self, result: TranscriptNormalizationResult) -> None:
        print(f"[HEBE][STT][RAW] text={result.raw_text!r}", flush=True)
        print(
            "[HEBE][STT][NORMALIZED] "
            f"raw={result.raw_text!r} "
            f"normalized_candidates={result.normalized_candidates!r}",
            flush=True,
        )
        stream = self._get_stream_state()
        if stream is not None:
            stream.last_voice_raw_transcript = result.raw_text
            stream.last_voice_normalized_command = result.normalized_text
            stream.last_voice_command_intent = None
            stream.last_voice_command_target = None
            stream.last_voice_command_status = "normalized"
            stream.last_voice_command_confidence = float(result.confidence)
        emit("voice.command", result.as_event())

    def _build_input_event(
        self,
        *,
        source: str,
        raw_text: str,
        normalized_text: str,
        stt_metadata: dict | None = None,
    ) -> InputEvent:
        event = InputEvent(
            source=source,
            raw_text=str(raw_text or ""),
            normalized_text=str(normalized_text or ""),
            is_voice=source == "stt_voice",
            is_stream_context=self._is_stream_enabled(),
            stt_metadata=stt_metadata or {},
        )
        print(
            "[HEBE][INPUT] "
            f"source={event.source} raw={event.raw_text!r} normalized={event.normalized_text!r}",
            flush=True,
        )
        return event

    def _input_event_has_action_intent(self, event: InputEvent | None) -> bool:
        if event is None:
            return False
        try:
            plan = self._get_stream_action_planner().plan(event)
            return plan is not None
        except Exception as exc:
            print(f"[HEBE][ACTION_PLAN] probe failed: {exc!r}", flush=True)
            return False

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
        if normalized.startswith(("ya hemos pasado ", "hemos pasado ")):
            return "completed_marker", None
        if any(marker in normalized for marker in ("ahora toca", "toca salir", "objetivo", "vamos a ", "hay que ")):
            return "objective_update", None
        if any(marker in normalized for marker in ("estamos en", "estoy en", "hemos llegado a", "salir de")):
            return "location_update", None
        if any(marker in normalized for marker in ("me han matado", "he muerto", "otra vez", "wipe", "game over")):
            return "gameplay_failure", "frustrated"
        if any(marker in normalized for marker in ("bien", "toma", "vamos", "victoria", "victory", "por fin", "ha caido")):
            return "victory", "excited"
        if any(marker in normalized for marker in ("boss", "jefe", "intento", "try", "pull")):
            return "boss_attempt", "focused"
        if any(marker in normalized for marker in ("farm", "farme", "grind", "grinde", "subir nivel", "farmear")):
            return "grinding", None
        if any(marker in normalized for marker in ("exploro", "explorar", "por aqui", "por aquí", "mapa", "cofre")):
            return "exploration", None
        if any(marker in normalized for marker in ("equipo", "menu", "menú", "inventario", "stats", "guardar")):
            return "menu/equipment", None
        if any(marker in normalized for marker in ("donde voy", "dónde voy", "perdido", "no entiendo", "que hago", "qué hago")):
            return "confusion/lost", "confused"
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
        stream.last_voice_summary = self._summarize_voice_event(text, event_type)
        if mood_hint:
            stream.leo_mood_hint = mood_hint
        self._apply_ambient_voice_to_run_context(stream, text, event_type)

    def _apply_ambient_voice_to_run_context(self, stream, text: str, event_type: str) -> None:
        if event_type not in {"completed_marker", "objective_update", "location_update"}:
            return
        raw = str(text or "").strip()
        normalized = self._normalize_text(raw)
        now = time.time()
        if event_type == "completed_marker":
            for prefix in ("ya hemos pasado ", "hemos pasado "):
                if normalized.startswith(prefix):
                    marker = raw.split(" ", len(prefix.split()))[-1].strip()
                    if marker:
                        self._add_completed_marker(stream, marker)
                        stream.run_context_updated_ts = now
                        stream.run_context_source = "ambient_stt"
                    return
        if event_type == "objective_update":
            stream.current_run_objective = raw[:120]
            stream.run_context_updated_ts = now
            stream.run_context_source = "ambient_stt"
            return
        if event_type == "location_update":
            match = re.search(r"(?:estamos en|estoy en|hemos llegado a|salir de)\s+(.+)", raw, flags=re.IGNORECASE)
            if match:
                stream.current_run_location = match.group(1).strip()[:80]
            else:
                stream.current_run_phase = raw[:120]
            stream.run_context_updated_ts = now
            stream.run_context_source = "ambient_stt"

    def _summarize_voice_event(self, text: str, event_type: str) -> str | None:
        if event_type in {"unknown", "direct_command_to_hebe"}:
            return None
        normalized = self._normalize_text(text)
        words = normalized.split()
        return " ".join(words[:8]) if words else None

    def _handle_stream_manual_command(self, text: str) -> str | CommandResult | None:
        stream = self._get_stream_state()
        if not stream:
            return None

        normalized = self._normalize_text(text)
        for prefix in ("hebe ", "ebe ", "eve ", "jebe "):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        raw_command = re.sub(
            r"^\s*(?:hebe|ebe|eve|jebe)[\s,;:.-]+",
            "",
            str(text or "").strip(),
            flags=re.IGNORECASE,
        ).strip()
        raw_lower = raw_command.lower()

        def command_result(
            action_type: str,
            fallback_text: str,
            *,
            state_changes: dict | None = None,
            message_goal: str | None = None,
            constraints: list[str] | None = None,
            success: bool = True,
        ) -> CommandResult:
            return CommandResult(
                action_type=action_type,
                success=success,
                user_visible_summary=message_goal or fallback_text,
                state_changes=state_changes or {},
                constraints=constraints or ["Be concise.", "Do not ask for clarification."],
                suggested_tone="short Hebe stream-control reply",
                fallback_text=fallback_text,
                requires_model_response=success,
                metadata={"message_goal": message_goal or fallback_text},
            )

        action_result = self._plan_and_execute_stream_action(raw_command, normalized, stream)
        if action_result is not None:
            return action_result

        run_reply = self._handle_run_context_command(raw_command, normalized, stream)
        if run_reply is not None:
            return run_reply

        shoutout_reply = self._handle_shoutout_manual_command(raw_command, normalized, stream)
        if shoutout_reply is not None:
            return shoutout_reply

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
            stream.presence_mode_explicit = True
            return command_result(
                "stream_presence_mode_changed",
                reply,
                state_changes={"presence_mode": mode, "presence_mode_explicit": True},
                message_goal=f"Confirm stream presence mode changed to {mode}.",
            )

        if normalized in {"prepara stream", "prepare stream"}:
            stream.enabled = True
            stream.last_chat_activity_ts = time.time()
            stream.presence_mode = "reactive"
            stream.no_stream_today_date = None
            self.stream_spontaneity.start_grace_period(stream)
            return command_result(
                "stream_mode_prepared",
                "Dejo el stream preparado en modo reactiva. OBS y el juego solo si me lo confirmas.",
                state_changes={"stream_enabled": True, "presence_mode": "reactive", "grace_period_started": True},
                message_goal="Confirm stream mode is armed in reactive mode, without implying OBS or the game were opened.",
            )

        if normalized in {"activa modo stream", "activa stream", "stream on", "enable stream"}:
            stream.enabled = True
            stream.last_chat_activity_ts = time.time()
            self.stream_spontaneity.start_grace_period(stream)
            return command_result(
                "stream_mode_enabled",
                "Modo stream activado.",
                state_changes={"stream_enabled": True, "grace_period_started": True},
                message_goal="Confirm stream mode is enabled and a grace period is active.",
            )

        if normalized in {"desactiva modo stream", "desactiva stream", "stream off", "disable stream"}:
            stream.enabled = False
            return command_result(
                "stream_mode_disabled",
                "Modo stream desactivado.",
                state_changes={"stream_enabled": False},
                message_goal="Confirm stream mode is disabled.",
            )

        if normalized in {"inicia memoria de stream", "inicia la memoria de stream"}:
            session_id = stream_memory.ensure_active_stream_session(stream, source="manual")
            return f"Memoria de stream iniciada. Sesion activa: {session_id}."

        if normalized in {"finaliza stream", "termina stream", "cierra stream"}:
            summary = stream_memory.close_active_stream_session(stream, reason="manual_command")
            stream.enabled = False
            return self._format_stream_summary_reply(summary) if summary else "No habia una sesion de stream activa que finalizar."

        if normalized in {"resume este stream", "resumen de este stream"}:
            session_id = self._ensure_stream_memory_session_if_live(stream) or getattr(stream, "active_stream_session_id", None)
            if not session_id:
                return "No tengo una sesion de stream activa para resumir."
            summary = stream_memory.summarize_stream_session(int(session_id), reason="manual_summary")
            return self._format_stream_summary_reply(summary)

        if normalized in {"que paso en el ultimo stream", "qué pasó en el último stream"}:
            summary = stream_memory.get_latest_stream_summary()
            return self._format_latest_stream_summary_reply(summary)

        chatter_match = re.match(r"^(?:que dijo|qué dijo)\s+(.+?)\s+en el ultimo stream$", normalized)
        if chatter_match:
            target = chatter_match.group(1).strip()
            summary = stream_memory.get_last_chatter_summary(target)
            if not summary:
                return f"No tengo resumen del ultimo stream para {target}."
            return f"En el ultimo stream, {target}: {summary.get('summary_text') or 'sin resumen suficiente'}"

        chatter_match = re.match(r"^(?:que sabes de|qué sabes de)\s+(.+)$", normalized)
        if chatter_match and "este juego" not in normalized:
            return stream_memory.format_chatter_profile_reply(chatter_match.group(1).strip())

        chatter_match = re.match(r"^(?:cuando fue la ultima vez que hablo|cuándo fue la última vez que habló)\s+(.+)$", normalized)
        if chatter_match:
            return stream_memory.format_last_seen_reply(chatter_match.group(1).strip(), kind="message")

        chatter_match = re.match(r"^(?:cuando fue la ultima vez que vimos a|cuándo fue la última vez que vimos a)\s+(.+)$", normalized)
        if chatter_match:
            return stream_memory.format_last_seen_reply(chatter_match.group(1).strip(), kind="seen")

        if normalized in {"hoy no hay stream", "no hay stream hoy", "today no stream"}:
            stream.no_stream_today_date = datetime.now(ZoneInfo("Europe/Madrid")).date().isoformat()
            stream.enabled = False
            return "Vale, hoy no insisto con el stream."

        if normalized in {"retrasa stream media hora", "retrasa el stream media hora", "delay stream half an hour"}:
            stream.stream_delay_minutes = int(getattr(stream, "stream_delay_minutes", 0) or 0) + 30
            return "Retraso el plan de stream media hora."

        if normalized in {"actualiza contexto de stream", "actualiza contexto stream", "refresh stream context"}:
            print("[HEBE][STREAM_CONTEXT] manual refresh requested", flush=True)
            service = getattr(self, "stream_context_sync", None)
            print(
                "[HEBE][STREAM_CONTEXT] manual refresh "
                f"service_exists={service is not None} runtime_twitch_exists={getattr(self.runtime, 'twitch', None) is not None}",
                flush=True,
            )
            ok = self.poll_stream_context(force=True, require_enabled=False)
            print(f"[HEBE][STREAM_CONTEXT] manual refresh completed success={ok}", flush=True)
            if ok:
                return command_result(
                    "stream_context_refreshed",
                    "Contexto de stream actualizado.",
                    state_changes={
                        "stream_context_updated": True,
                        "is_live": getattr(stream, "is_live", False),
                        "current_category": getattr(stream, "current_category", None),
                        "current_stream_title": getattr(stream, "current_stream_title", None),
                    },
                    message_goal="Confirm the Twitch stream context was refreshed.",
                )
            return "No he podido actualizar el contexto de stream ahora."

        if normalized in {"que contexto de stream tienes", "qué contexto de stream tienes", "stream context"}:
            return self._build_stream_context_reply(stream)

        if normalized in {"prueba espontaneidad", "previsualiza espontaneidad", "genera una espontanea de prueba", "genera una espontánea de prueba"}:
            self._manual_reply_ui_only = True
            return self._build_spontaneity_preview_reply(stream)

        if normalized in {"comprueba espontaneidad", "estado de espontaneidad"}:
            return self._build_spontaneity_readiness_reply(stream)

        if normalized in {"prueba raid"}:
            self._manual_reply_ui_only = True
            return self._build_raid_preview_reply("tester", viewer_count=1)

        if normalized.startswith("simula raid de "):
            target = normalized[len("simula raid de "):].strip()
            send = False
            if target.endswith(" y envialo"):
                send = True
                target = target[: -len(" y envialo")].strip()
            if target.endswith(" y envíalo"):
                send = True
                target = target[: -len(" y envíalo")].strip()
            if not target:
                target = "tester"
            self._handle_twitch_raid_event(self._build_local_internal_event("twitch_raid", {
                "display_name": target,
                "user_login": target,
                "viewer_count": 1,
                "_force_shoutout": True,
            }))
            return f"Raid simulado de {target} enviado." if send else f"Raid simulado de {target}: thank-you y SO probados."

        if normalized in {"activa simulacion de directo", "activa simulación de directo"}:
            stream.live_test_override = True
            return "Simulacion de directo: activada."

        if normalized in {"desactiva simulacion de directo", "desactiva simulación de directo"}:
            stream.live_test_override = False
            return "Simulacion de directo: desactivada."

        if normalized in {"resetea cooldowns de espontaneidad", "resetea cooldown espontaneidad"}:
            cleared = self.stream_spontaneity.reset_spontaneity_cooldowns(stream)
            return f"Cooldowns de espontaneidad reseteados: {cleared}."

        if normalized in {"espontaneidad en texto"}:
            stream.policies.allow_tts_idle_prompts = False
            return command_result(
                "idle_tts_disabled",
                "Espontaneidad en texto. Si comento por mi cuenta, no lo leo en voz.",
                state_changes={"allow_tts_idle_prompts": False},
                message_goal="Confirm idle spontaneous messages are text-only.",
            )

        if normalized in {"espontaneidad con voz"}:
            stream.policies.allow_tts_idle_prompts = True
            return command_result(
                "idle_tts_enabled",
                "Espontaneidad con voz activada. Con moderacion.",
                state_changes={"allow_tts_idle_prompts": True},
                message_goal="Confirm idle spontaneous messages may use voice, while keeping it restrained.",
            )

        if normalized in {"activa stt ambiental", "activa stt ambiental de stream"}:
            self.stream_ambient_stt_enabled = True
            stt = getattr(self.runtime, "stt", None)
            if stt is not None and hasattr(stt, "clear_device_error"):
                stt.clear_device_error()
            self._ensure_stt_worker_running()
            return command_result(
                "stream_ambient_stt_enabled",
                "STT ambiental de stream activado. Escucho contexto, no respondo a todo.",
                state_changes={"stream_ambient_stt_enabled": True},
                message_goal="Confirm ambient stream STT is enabled only for context, not immediate replies.",
            )

        if normalized in {"desactiva stt ambiental", "desactiva stt ambiental de stream"}:
            self.stream_ambient_stt_enabled = False
            return command_result(
                "stream_ambient_stt_disabled",
                "STT ambiental de stream desactivado.",
                state_changes={"stream_ambient_stt_enabled": False},
                message_goal="Confirm ambient stream STT is disabled.",
            )

        if normalized in {"limpia oido del stream", "limpia oído del stream", "limpia contexto oido", "limpia contexto oído"}:
            stream.last_voice_event = None
            stream.last_voice_event_ts = 0.0
            stream.last_voice_summary = None
            stream.leo_mood_hint = None
            return "Oido ambiental limpiado."

        if normalized in {"pausa espontaneidad"}:
            stream.idle_spontaneity_enabled = False
            return command_result(
                "idle_spontaneity_paused",
                "Pauso la espontaneidad idle. Raids, subs, follows y menciones siguen funcionando.",
                state_changes={"idle_spontaneity_enabled": False},
                message_goal="Confirm idle spontaneity is paused, while events and direct mentions still work.",
            )

        if normalized in {"activa espontaneidad"}:
            stream.idle_spontaneity_enabled = True
            self.stream_spontaneity.start_grace_period(stream)
            return command_result(
                "idle_spontaneity_enabled",
                "Espontaneidad idle activada, con periodo de gracia.",
                state_changes={"idle_spontaneity_enabled": True, "grace_period_started": True},
                message_goal="Confirm idle spontaneity is enabled and protected by a grace period.",
            )

        if normalized in {"habla menos"}:
            stream.presence_mode = "companion"
            stream.presence_mode_explicit = True
            stream.cooldowns["companion_idle_silence_sec"] = max(
                float(stream.cooldowns.get("companion_idle_silence_sec", 20 * 60) or 20 * 60),
                25 * 60,
            )
            return "Bajo intensidad. Menos comentarios, mas aire."

        if normalized in {"habla mas", "habla más"}:
            stream.presence_mode = "show"
            stream.presence_mode_explicit = True
            stream.cooldowns["show_idle_silence_sec"] = min(
                float(stream.cooldowns.get("show_idle_silence_sec", 9 * 60) or 9 * 60),
                8 * 60,
            )
            return "Subo a modo show, pero sin ponerme pesada."

        if normalized in {"que sabes de este juego", "qué sabes de este juego", "que perfil de juego estas usando", "qué perfil de juego estás usando"}:
            return self._build_game_profile_reply(stream)

        if normalized in {"investiga este juego sin spoilers", "actualiza conocimiento de este juego"}:
            return self._research_current_game_reply(stream, force=normalized.startswith("actualiza"))

        if normalized in {"olvida perfil de este juego"}:
            profile = self.game_profiles.forget_profile(
                current_category=getattr(stream, "current_category", None),
                current_title=getattr(stream, "current_stream_title", None),
                current_game=getattr(stream, "current_game", None),
            )
            return f"Perfil olvidado para esta sesion. Usare fallback: {profile.canonical_title}."

        if normalized in {"desactiva investigacion de juegos", "desactiva investigación de juegos"}:
            self.game_research = GameKnowledgeResearchService(
                store=self.game_profiles,
                config=GameKnowledgeResearchConfig(
                    enabled=False,
                    provider=getattr(getattr(self, "game_research", None), "config", GameKnowledgeResearchConfig()).provider,
                    api_key=getattr(getattr(self, "game_research", None), "config", GameKnowledgeResearchConfig()).api_key,
                    cache_days=getattr(getattr(self, "game_research", None), "config", GameKnowledgeResearchConfig()).cache_days,
                ),
            )
            return "Investigacion de juegos desactivada. Usare perfiles locales/cacheados."

        if normalized in {"activa investigacion de juegos", "activa investigación de juegos"}:
            previous = getattr(getattr(self, "game_research", None), "config", GameKnowledgeResearchConfig.from_env())
            self.game_research = GameKnowledgeResearchService(
                store=self.game_profiles,
                config=GameKnowledgeResearchConfig(
                    enabled=True,
                    provider=previous.provider,
                    api_key=previous.api_key,
                    cache_days=previous.cache_days,
                ),
                search_provider=getattr(getattr(self, "game_research", None), "search_provider", None),
            )
            return "Investigacion de juegos activada. Solo la usare en comandos o preparacion, no en cada mensaje."

        if normalized in {"recarga perfiles de juegos", "recarga perfiles de juego"}:
            count = self.game_profiles.reload()
            return f"Perfiles de juegos recargados: {count}."

        if normalized in {"diagnostica twitch", "diagnostico twitch", "twitch diagnostic"}:
            return self._build_twitch_diagnostic_reply()

        if normalized in {"estado de shoutouts", "estado shoutouts", "estado de so"}:
            return self._build_shoutout_status_reply(stream)

        return None

    def _get_stream_action_planner(self) -> StreamActionPlanner:
        planner = getattr(self, "stream_action_planner", None)
        if planner is None:
            planner = self._build_stream_action_planner()
            self.stream_action_planner = planner
        return planner

    def _plan_and_execute_stream_action(self, raw_command: str, normalized: str, stream) -> CommandResult | None:
        input_event = getattr(self, "_current_input_event", None) or self._build_input_event(
            source="typed_ui",
            raw_text=raw_command,
            normalized_text=normalized,
        )
        planner = self._get_stream_action_planner()
        plan = planner.plan(InputEvent(
            source=input_event.source,
            raw_text=input_event.raw_text,
            normalized_text=raw_command or normalized,
            user_id=input_event.user_id,
            username=input_event.username,
            is_voice=input_event.is_voice,
            is_stream_context=input_event.is_stream_context,
            timestamp=input_event.timestamp,
            stt_metadata=input_event.stt_metadata,
        ))
        if plan is None:
            return None
        print(f"[HEBE][COGNITION] intent_candidates={[plan.action_type]!r}", flush=True)
        print(
            "[HEBE][ACTION_PLAN] "
            f"action_type={plan.action_type} target={plan.target} confidence={plan.confidence:.3f} status={plan.status} reason={plan.reason}",
            flush=True,
        )
        emit("voice.command", {
            "raw_text": input_event.raw_text,
            "normalized_text": raw_command or normalized,
            "intent": plan.action_type,
            "target": plan.target,
            "confidence": round(float(plan.confidence), 3),
            "status": plan.status,
            "reason": plan.reason,
            "candidates": plan.candidates,
        })
        if plan.action_type == "twitch_shoutout":
            return self._execute_twitch_shoutout_plan(plan, stream)
        if plan.action_type in {"stream_ambient_stt_enabled", "stream_ambient_stt_disabled"}:
            return self._execute_stream_ambient_stt_plan(plan)
        return None

    def _execute_stream_ambient_stt_plan(self, plan: ActionPlan) -> CommandResult:
        enabled = plan.action_type == "stream_ambient_stt_enabled"
        print(f"[HEBE][ACTION_EXECUTOR] executing action_type={plan.action_type}", flush=True)
        self.stream_ambient_stt_enabled = enabled
        if enabled:
            stt = getattr(self.runtime, "stt", None)
            if stt is not None and hasattr(stt, "clear_device_error"):
                try:
                    stt.clear_device_error()
                except Exception as exc:
                    print(f"[HEBE][STT][ERROR] clear_device_error failed: {exc!r}", flush=True)
            try:
                self._ensure_stt_worker_running()
            except Exception as exc:
                print(f"[HEBE][STT][ERROR] ensure worker failed: {exc!r}", flush=True)
        emit(
            "audio.status",
            {
                "stream_ambient_stt_enabled": enabled,
                "stt_ambient_enabled": enabled,
            },
        )
        print(
            f"[HEBE][ACTION_EXECUTOR] success=true action_type={plan.action_type} enabled={enabled}",
            flush=True,
        )
        return CommandResult(
            action_type=plan.action_type,
            success=True,
            user_visible_summary=(
                "Ambient stream STT was enabled." if enabled else "Ambient stream STT was disabled."
            ),
            state_changes={"stream_ambient_stt_enabled": enabled},
            constraints=[
                "Do not claim any other STT setting changed.",
                "Do not ask for clarification.",
            ],
            suggested_tone="short Hebe stream-control reply",
            fallback_text=(
                "STT ambiental activado." if enabled else "STT ambiental desactivado."
            ),
            requires_model_response=True,
            metadata={
                "action_plan": plan.as_log_dict(),
                "message_goal": (
                    "Confirm to Leo that ambient stream STT is enabled."
                    if enabled
                    else "Confirm to Leo that ambient stream STT is disabled."
                ),
            },
        )

    def _execute_twitch_shoutout_plan(self, plan: ActionPlan, stream) -> CommandResult:
        if plan.status == "needs_confirmation":
            print(
                "[HEBE][ACTION_EXECUTOR] success=false reason=needs_confirmation "
                f"action_type={plan.action_type}",
                flush=True,
            )
            if "target" in plan.missing_slots or plan.reason in {"missing_target", "target_unclear", "invalid_target"}:
                fallback = "¿A quién le hago el SO, Leo?"
                goal = "Ask Leo which Twitch user should receive the shoutout."
            elif plan.reason == "ambiguous_target":
                fallback = "He pillado varios nombres parecidos. Dime el usuario exacto para el SO."
                goal = f"Ask Leo to clarify the shoutout target. Candidates: {', '.join(plan.candidates)}."
            else:
                fallback = "Creo que me has pedido un SO, pero necesito confirmación."
                goal = "Ask Leo to confirm the shoutout target before sending it."
            return CommandResult(
                action_type="twitch_shoutout_clarify",
                success=False,
                user_visible_summary=goal,
                state_changes={},
                constraints=["Ask one concise follow-up question.", "Do not claim the shoutout was sent."],
                suggested_tone="short Hebe clarification",
                fallback_text=fallback,
                requires_model_response=False,
                metadata={"action_plan": plan.as_log_dict(), "message_goal": goal},
            )

        if plan.status != "complete" or not plan.target:
            print(
                "[HEBE][ACTION_EXECUTOR] success=false reason=rejected "
                f"action_type={plan.action_type}",
                flush=True,
            )
            return CommandResult(
                action_type="twitch_shoutout_rejected",
                success=False,
                user_visible_summary="The shoutout request was not clear enough to execute.",
                state_changes={},
                constraints=["Do not claim the shoutout was sent."],
                fallback_text="No lo ejecuto, Leo. No lo he entendido con suficiente seguridad.",
                requires_model_response=False,
                metadata={"action_plan": plan.as_log_dict(), "message_goal": "Tell Leo the command was not clear enough to execute."},
            )

        print(f"[HEBE][ACTION_EXECUTOR] executing action_type={plan.action_type} target={plan.target}", flush=True)
        ok, normalized_target, send_reason = self._send_shoutout(plan.target, source="manual", force=False)
        print(
            "[HEBE][ACTION_EXECUTOR] "
            f"success={ok} action_type={plan.action_type} target={normalized_target} reason={send_reason}",
            flush=True,
        )
        if ok:
            return CommandResult(
                action_type="twitch_shoutout",
                success=True,
                user_visible_summary=f"Shoutout sent to {normalized_target}.",
                state_changes={
                    "action_type": "twitch_shoutout",
                    "target": normalized_target,
                    "command_sent": self._build_shoutout_command_preview(normalized_target),
                    "confidence": plan.confidence,
                },
                constraints=[
                    "Do not claim anything beyond the shoutout command being sent.",
                    "Do not ask for clarification.",
                ],
                suggested_tone="short Hebe stream-control reply",
                fallback_text=f"SO enviado a {normalized_target}.",
                requires_model_response=True,
                metadata={
                    "action_plan": plan.as_log_dict(),
                    "message_goal": f"Tell Leo that the promo/shoutout for {normalized_target} was sent.",
                },
            )
        if send_reason in {"blocked_bot_user", "own_channel", "invalid_target"}:
            fallback = "No le hago SO a ese usuario, Leo. Huele a bot o a bucle infernal."
        elif send_reason == "cooldown_active":
            fallback = f"Ya hice SO a {normalized_target} hace nada, Leo. Evito el spam."
        else:
            fallback = f"No he podido hacer el SO a {normalized_target or plan.target}."
        return CommandResult(
            action_type="twitch_shoutout",
            success=False,
            user_visible_summary=f"Shoutout failed: {send_reason}",
            state_changes={"target": normalized_target, "send_reason": send_reason},
            constraints=["Do not claim the shoutout was sent."],
            fallback_text=fallback,
            requires_model_response=False,
            metadata={"action_plan": plan.as_log_dict(), "message_goal": "Tell Leo the shoutout could not be sent."},
        )

    def _handle_shoutout_manual_command(self, raw_command: str, normalized: str, stream) -> str | CommandResult | None:
        preview = False
        force = False
        raw = raw_command.strip()
        norm = normalized.strip()

        if norm.startswith("previsualiza shoutout a ") or norm.startswith("previsualiza so a "):
            preview = True
            raw = re.sub(r"^\s*previsualiza\s+(?:shoutout|so)\s+a\s+", "", raw, flags=re.IGNORECASE).strip()
            norm = self._normalize_text(raw)
        elif norm.startswith("prueba shoutout a ") or norm.startswith("prueba so a "):
            force = True
            raw = re.sub(r"^\s*prueba\s+(?:shoutout|so)\s+a\s+", "", raw, flags=re.IGNORECASE).strip()
            norm = self._normalize_text(raw)
        elif norm in {"prueba so", "prueba shoutout"}:
            preview = True
            last_raider = getattr(stream, "last_raider_username", None) or getattr(stream, "last_raider_display_name", None)
            raw = str(last_raider or "tester")
            norm = self._normalize_text(raw)
        else:
            patterns = [
                r"^(?:haz\s+un\s+so\s+a|haz\s+so\s+a|hazle\s+so\s+a|dale\s+un\s+so\s+a)\s+(.+)$",
                r"^(?:haz\s+un\s+so\s+al|haz\s+so\s+al|hazle\s+so\s+al|dale\s+un\s+so\s+al)\s+(.+)$",
                r"^(?:shoutout\s+a|haz\s+shoutout\s+a|shoutout)\s+(.+)$",
                r"^(?:promociona\s+a|haz\s+promo\s+a|hazle\s+promo\s+a|dale\s+promo\s+a|recomienda\s+a)\s+(.+)$",
                r"^(?:give\s+a\s+shoutout\s+to|shoutout|promote|so)\s+(.+)$",
                r"^give\s+(.+)\s+a\s+promo$",
                r"^(?:haz\s+so|haz\s+un\s+so|hazle\s+so|dale\s+un\s+so)$",
            ]
            match = None
            for pattern in patterns:
                match = re.match(pattern, norm, flags=re.IGNORECASE)
                if match:
                    break
            if not match:
                return None
            if match.lastindex:
                raw = raw.split()[-len(match.group(1).split()):]
                raw = " ".join(raw)
                norm = match.group(1).strip()
            else:
                raw = ""
                norm = ""

        target, reason = self._resolve_shoutout_target(raw or norm)
        if reason == "missing_target":
            return "¿A quién le hago el SO, Leo?"
        if not target:
            return "No encuentro un usuario válido para el SO, Leo."

        command = self._build_shoutout_command_preview(target)
        if preview:
            self._manual_reply_ui_only = True
            return f"Previsualizacion de shoutout: {command}"

        ok, normalized_target, send_reason = self._send_shoutout(target, source="manual", force=force)
        if ok:
            return CommandResult(
                action_type="shoutout_sent",
                success=True,
                user_visible_summary=f"SO sent to {normalized_target}.",
                state_changes={"shoutout_sent": True, "target": normalized_target},
                constraints=["Do not claim anything beyond the shoutout command being sent.", "Do not ask for clarification."],
                suggested_tone="short Hebe stream-control reply",
                fallback_text=f"SO enviado a {normalized_target}.",
                requires_model_response=True,
                metadata={"message_goal": f"Confirm the SO command was sent to {normalized_target}."},
            )
        if send_reason in {"blocked_bot_user", "own_channel", "invalid_target"}:
            return "No le hago SO a ese usuario, Leo. Huele a bot o a bucle infernal."
        if send_reason == "cooldown_active":
            return f"Ya hice SO a {normalized_target} hace nada, Leo. Evito el spam."
        return f"No he podido hacer el SO a {normalized_target or target}."

    def _build_shoutout_command_preview(self, target: str) -> str:
        twitch = getattr(self.runtime, "twitch", None)
        build = getattr(twitch, "build_shoutout_command", None)
        normalized = self._normalize_shoutout_target(target)
        if callable(build):
            return build(normalized)
        template = os.getenv("HEBE_SHOUTOUT_COMMAND_TEMPLATE", "!so {username}") or "!so {username}"
        return template.format(username=normalized)

    def _handle_run_context_command(self, raw_command: str, normalized: str, stream) -> str | None:
        now = time.time()

        def set_updated(source: str) -> None:
            stream.run_context_updated_ts = now
            stream.run_context_source = source

        lower = raw_command.lower()
        objective_prefixes = ("objetivo actual:", "objetivo actual ")
        for prefix in objective_prefixes:
            if lower.startswith(prefix):
                value = raw_command[len(prefix):].strip()
                if value:
                    stream.current_run_objective = value
                    set_updated("manual")
                    return f"Objetivo actual guardado: {value}."

        progress_prefixes = ("progreso actual:", "progreso actual ")
        for prefix in progress_prefixes:
            if lower.startswith(prefix):
                value = raw_command[len(prefix):].strip()
                if value:
                    stream.current_run_phase = value
                    set_updated("manual")
                    return f"Progreso actual guardado: {value}."

        location_prefixes = ("estamos en ",)
        for prefix in location_prefixes:
            if lower.startswith(prefix):
                value = raw_command[len(prefix):].strip()
                if value:
                    stream.current_run_location = value
                    set_updated("manual")
                    return f"Ubicacion actual guardada: {value}."

        passed_prefixes = ("ya hemos pasado ", "hemos pasado ")
        for prefix in passed_prefixes:
            if lower.startswith(prefix):
                marker = raw_command[len(prefix):].strip()
                if marker:
                    self._add_completed_marker(stream, marker)
                    set_updated("manual")
                    return f"Marcador completado guardado: {marker}."

        forget_prefixes = ("olvida ",)
        if lower.startswith("olvida ") and lower.endswith(" como objetivo actual"):
            marker = raw_command[len("olvida "): -len(" como objetivo actual")].strip()
            if marker:
                self._add_completed_marker(stream, marker)
                if self._same_marker(getattr(stream, "current_run_objective", ""), marker):
                    stream.current_run_objective = None
                set_updated("manual")
                return f"Dejo de tratar {marker} como objetivo actual."

        if normalized in {"limpia contexto de partida"}:
            stream.current_run_objective = None
            stream.current_run_location = None
            stream.current_run_phase = None
            stream.completed_run_markers = []
            stream.run_context_updated_ts = now
            stream.run_context_source = "manual"
            return "Contexto de partida limpiado."

        if normalized in {"que contexto de partida tienes", "qué contexto de partida tienes"}:
            return self._build_run_context_reply(stream)

        if normalized in {"que ha oido del stream", "qué ha oido del stream", "qué ha oído del stream"}:
            return self._build_stream_heard_reply(stream)

        if normalized in {"que esta pasando en chat", "qué está pasando en chat", "que está pasando en chat"}:
            return self._build_chat_context_reply(stream)

        return None

    def _build_raid_preview_reply(self, username: str, *, viewer_count: int = 1) -> str:
        event = self._build_local_internal_event("twitch_raid", {
            "display_name": username,
            "user_login": username,
            "viewer_count": viewer_count,
        })
        message = self._synthesize_internal_event_reply(event)
        return f"Prueba de raid: '{message}'"

    def _build_local_internal_event(self, event_type: str, payload: dict) -> InternalEvent:
        return InternalEvent(
            event_type=event_type,
            payload=payload,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def _lookup_current_game_profile(self, stream):
        store = getattr(self, "game_profiles", None)
        if store is None:
            store = GameProfileStore()
            self.game_profiles = store
        return store.lookup(
            current_category=getattr(stream, "current_category", None),
            current_game=getattr(stream, "current_game", None),
            current_title=getattr(stream, "current_stream_title", None),
        )

    def _build_game_profile_reply(self, stream) -> str:
        profile = self._lookup_current_game_profile(stream)

        def joined(items, fallback="ninguno") -> str:
            return ", ".join(items) if items else fallback

        return (
            "Perfil spoiler-safe de juego:\n\n"
            f"* Juego: {profile.canonical_title}\n"
            f"* Slug: {profile.game_slug}\n"
            f"* Fuente/categoria: {profile.source_category_name or 'perfil local'}\n"
            f"* Generos: {joined(profile.genres)}\n"
            f"* Tono/vibe: {profile.tone_vibe or 'no detectado'}\n"
            f"* Resumen sin spoilers: {profile.general_non_spoiler_summary or profile.channel_context or 'no detectado'}\n"
            f"* Sistemas no-spoiler: {joined(profile.gameplay_systems_non_spoiler)}\n"
            f"* Spoilers: {profile.spoiler_policy}\n"
            f"* Temas seguros: {joined(profile.safe_comment_topics)}\n"
            f"* Hooks de challenge: {joined(profile.challenge_hooks or profile.challenge_notes)}\n"
            f"* Temas prohibidos: {joined(profile.unsafe_comment_topics)}\n"
            f"* Hooks de stream: {joined(profile.stream_hooks)}\n"
            f"* Fuentes: {joined(profile.sources_used, 'perfil local')}\n"
            f"* Actualizado: {profile.updated_at or self._format_stream_context_age(profile.last_updated_ts)}"
        )

    def _format_stream_summary_reply(self, summary: dict | None) -> str:
        if not summary:
            return "No he podido generar resumen del stream."
        topics = []
        try:
            import json

            topics = list((json.loads(summary.get("chat_topics_json") or "{}") or {}).keys())
        except Exception:
            topics = []
        return (
            "Resumen guardado del stream:\n\n"
            f"* Sesion: {summary.get('stream_session_id')}.\n"
            f"* Resumen: {summary.get('summary_text') or 'sin texto'}\n"
            f"* Temas de chat: {', '.join(topics) if topics else 'sin temas suficientes'}."
        )

    def _format_latest_stream_summary_reply(self, summary: dict | None) -> str:
        if not summary:
            return "Todavia no tengo resumen de ningun stream."
        return (
            "Esto tengo del ultimo stream:\n\n"
            f"* Juego/categoria: {summary.get('game') or summary.get('category') or 'sin categoria'}.\n"
            f"* Titulo: {summary.get('title') or 'sin titulo'}.\n"
            f"* Inicio: {summary.get('started_at') or 'desconocido'}.\n"
            f"* Resumen: {summary.get('summary_text') or 'sin resumen'}"
        )

    def _research_current_game_reply(self, stream, *, force: bool = False) -> str:
        service = getattr(self, "game_research", None)
        if service is None:
            service = GameKnowledgeResearchService(store=self.game_profiles)
            self.game_research = service
        ok, profile, reason = service.research_current_game(
            current_category=getattr(stream, "current_category", None),
            current_title=getattr(stream, "current_stream_title", None),
            current_game=getattr(stream, "current_game", None),
            force=force,
        )
        if ok:
            if reason == "cached_profile":
                return f"Conocimiento de juego ya cacheado: {profile.canonical_title}. Spoilers: {profile.spoiler_policy}."
            return f"Conocimiento spoiler-safe actualizado: {profile.canonical_title}. Fuentes: {', '.join(profile.sources_used) or 'proveedor configurado'}."
        if reason == "research_disabled":
            return f"Investigacion de juegos desactivada. Uso perfil local: {profile.canonical_title}."
        if reason == "research_provider_missing":
            return f"Investigacion activada, pero falta proveedor/API configurado. Uso perfil local: {profile.canonical_title}."
        return f"No he podido investigar ahora ({reason}). Uso perfil local: {profile.canonical_title}."

    def _add_completed_marker(self, stream, marker: str) -> None:
        value = str(marker or "").strip()
        if not value:
            return
        existing = list(getattr(stream, "completed_run_markers", []) or [])
        if not any(self._same_marker(item, value) for item in existing):
            existing.append(value)
        stream.completed_run_markers = existing[-30:]

    def _same_marker(self, left: str, right: str) -> bool:
        return self._normalize_text(left) == self._normalize_text(right)

    def _build_run_context_reply(self, stream) -> str:
        now = time.time()
        title_updated = float(getattr(stream, "title_context_updated_ts", 0.0) or 0.0)
        title_age = now - title_updated if title_updated else 999999
        title_ttl = getattr(getattr(self, "stream_spontaneity", None), "config", None)
        ttl = float(getattr(title_ttl, "title_marker_ttl_sec", 55 * 60) or 55 * 60)
        title_status = "fresco" if title_updated and title_age <= ttl else "stale"
        run_updated = float(getattr(stream, "run_context_updated_ts", 0.0) or 0.0)
        run_status = self._format_stream_context_age(run_updated) if run_updated else "nunca"

        def joined(items, fallback="ninguno") -> str:
            return ", ".join(items) if items else fallback

        return (
            "Contexto de partida:\n\n"
            f"* Juego/categoria: {getattr(stream, 'current_category', None) or getattr(stream, 'current_game', None) or 'sin categoria'}\n"
            f"* Tipo de run: {getattr(stream, 'current_playthrough_type', None) or 'no detectado'}\n"
            f"* Challenge: {getattr(stream, 'current_challenge', None) or 'ninguno'}\n"
            f"* Objetivo actual: {getattr(stream, 'current_run_objective', None) or 'no detectado'}\n"
            f"* Ubicacion actual: {getattr(stream, 'current_run_location', None) or 'no detectada'}\n"
            f"* Progreso/fase: {getattr(stream, 'current_run_phase', None) or 'no detectado'}\n"
            f"* Marcadores del titulo: {joined(getattr(stream, 'title_context_markers', []) or [])} ({title_status})\n"
            f"* Marcadores completados: {joined(getattr(stream, 'completed_run_markers', []) or [])}\n"
            f"* Contexto de partida actualizado: {run_status}\n"
            f"* Fuente: {getattr(stream, 'run_context_source', None) or 'unknown'}\n"
            f"* Spoilers: {getattr(stream, 'spoiler_policy', None) or 'no_spoilers'}"
        )

    def _build_stream_heard_reply(self, stream) -> str:
        updated = self._format_stream_context_age(getattr(stream, "last_voice_event_ts", 0.0) or 0.0)
        return (
            "Esto he oido del stream:\n\n"
            f"* Ultimo evento de voz: {getattr(stream, 'last_voice_event', None) or 'ninguno'}\n"
            f"* Mood: {getattr(stream, 'leo_mood_hint', None) or 'no detectado'}\n"
            f"* Resumen: {getattr(stream, 'last_voice_summary', None) or 'ninguno'}\n"
            f"* Actualizado: {updated}"
        )

    def _build_chat_context_reply(self, stream) -> str:
        snapshot = self._chat_activity_snapshot(stream)
        return (
            "Contexto de chat:\n\n"
            f"* Chat activo: {'yes' if snapshot['active'] else 'no'}\n"
            f"* Mensajes recientes: {snapshot['count']} en {int(snapshot['window_sec'])}s\n"
            f"* Usuarios recientes: {', '.join(snapshot['users']) if snapshot['users'] else 'ninguno'}\n"
            f"* Temas recientes: {snapshot['summary']}"
        )

    def _build_spontaneity_preview_reply(self, stream) -> str:
        event = self.stream_spontaneity.build_preview_event(stream)
        if event is None:
            return "Prueba de espontaneidad: no tengo suficiente contexto para generar una prueba."
        message = self.response_synthesizer.generate_twitch_idle_prompt_preview(event.payload)
        stream.last_stream_spontaneity_preview_ts = time.time()
        return f"Prueba de espontaneidad: '{message}'"

    def _build_spontaneity_readiness_reply(self, stream) -> str:
        live_override = bool(getattr(stream, "live_test_override", False))
        readiness = self.stream_spontaneity.evaluate(stream, live_override=live_override)

        def yes_no(value) -> str:
            return "yes" if bool(value) else "no"

        twitch_live = readiness.get("twitch_live")
        if twitch_live == "unknown":
            live_label = "unknown"
        else:
            live_label = yes_no(twitch_live)

        category = getattr(stream, "current_category", None) or getattr(stream, "current_game", None) or "sin categoria"
        playthrough = getattr(stream, "current_playthrough_type", None) or "no detectado"
        challenge = getattr(stream, "current_challenge", None) or "ninguno"
        spoilers = getattr(stream, "spoiler_policy", None) or "no detectado"
        reason = readiness.get("blocked_reason") or "unknown"
        updated = self._format_stream_context_age(getattr(stream, "stream_context_updated_ts", 0.0) or 0.0)
        transition = getattr(stream, "last_stream_live_transition", None) or "ninguna"
        transition_ts = getattr(stream, "last_stream_live_transition_ts", 0.0) or 0.0
        transition_age = self._format_stream_context_age(transition_ts) if transition_ts else "nunca"
        raid = getattr(stream, "last_raid_event", None) or {}
        if raid:
            raid_label = f"{raid.get('display_name')} ({raid.get('viewer_count', 0)})"
        else:
            raid_label = "ninguno"
        is_processing = bool(getattr(self.runtime.state, "is_processing", False))
        if is_processing:
            reason = "command currently processing"
        chat_snapshot = self._chat_activity_snapshot(stream)
        recent_idle = list(getattr(stream, "recent_idle_messages", []) or [])
        next_ts = readiness.get("next_possible_idle_prompt_ts") or (getattr(stream, "cooldowns", {}) or {}).get(
            getattr(self.stream_spontaneity.config, "cooldown_key", "stream_idle_prompt_next_ts"),
            0.0,
        )
        next_label = self._format_stream_context_age(float(next_ts)) if next_ts and float(next_ts) <= time.time() else (
            datetime.fromtimestamp(float(next_ts), ZoneInfo("Europe/Madrid")).strftime("%H:%M:%S") if next_ts else "cuando se cumplan las condiciones"
        )

        return (
            "Estado de espontaneidad:\n\n"
            f"* Stream mode enabled: {yes_no(readiness.get('stream_enabled'))}\n"
            f"* Twitch live: {live_label}\n"
            f"* Auto-enable when live: {yes_no(getattr(self, 'auto_enable_stream_when_live', True))}\n"
            f"* Default live presence mode: {getattr(self, 'default_live_presence_mode', 'companion')}\n"
            f"* Simulacion de directo: {'activada' if live_override else 'desactivada'}\n"
            f"* Presence mode: {readiness.get('presence_mode')}\n"
            f"* Idle enabled/paused: {'enabled' if getattr(stream, 'idle_spontaneity_enabled', True) else 'paused'}\n"
            f"* Idle TTS enabled: {yes_no(getattr(stream.policies, 'allow_tts_idle_prompts', False))}\n"
            f"* Chat active: {yes_no(chat_snapshot['active'])}\n"
            f"* Recent chat messages/window: {chat_snapshot['count']} / {int(chat_snapshot['window_sec'])}s\n"
            f"* Context fresh: {yes_no(readiness.get('context_fresh'))}\n"
            f"* Command currently processing: {yes_no(is_processing)}\n"
            f"* Last stream context update: {updated}\n"
            f"* Last live transition: {transition} ({transition_age})\n"
            f"* Last raid event: {raid_label}\n"
            f"* Game/category: {category}\n"
            f"* Current run objective: {getattr(stream, 'current_run_objective', None) or 'no detectado'}\n"
            f"* Current run location: {getattr(stream, 'current_run_location', None) or 'no detectada'}\n"
            f"* Title markers: {', '.join(getattr(stream, 'title_context_markers', []) or []) or 'ninguno'}\n"
            f"* Stale title markers: {', '.join(readiness.get('title_markers_stale') or []) or 'ninguno'}\n"
            f"* Completed markers: {', '.join(getattr(stream, 'completed_run_markers', []) or []) or 'ninguno'}\n"
            f"* Playthrough: {playthrough}\n"
            f"* Challenge: {challenge}\n"
            f"* Spoilers: {spoilers}\n"
            f"* Last idle topic: {(recent_idle[-1].get('topic') if recent_idle else None) or 'ninguno'}\n"
            f"* Recent idle topics: {', '.join([item.get('topic') for item in recent_idle[-6:] if item.get('topic')]) or 'ninguno'}\n"
            f"* Prompts sent this hour: {readiness.get('prompts_sent_hour', 0)}\n"
            f"* Prompts sent this stream: {getattr(stream, 'idle_prompts_sent_stream', 0)}\n"
            f"* Recent chat block: {yes_no(readiness.get('recent_chat_block'))}\n"
            f"* Recent Hebe message block: {yes_no(readiness.get('recent_hebe_block'))}\n"
            f"* Cooldown ready: {yes_no(readiness.get('cooldown_ready'))}\n"
            f"* Would send now: {yes_no(readiness.get('would_send'))}\n"
            f"* Last spontaneous blocked reason: {getattr(stream, 'last_stream_spontaneity_blocked_reason', None) or 'ninguno'}\n"
            f"* Reason if blocked: {reason}\n"
            f"* Next possible idle prompt time: {next_label}"
        )

    def _build_stream_context_reply(self, stream) -> str:
        if getattr(stream, "live_status_known", False):
            if getattr(stream, "is_live", False):
                status = "online, comprobado con Twitch."
            else:
                status = "offline, comprobado con Twitch."
        else:
            status = "desconocido. No he podido confirmar si el stream esta online."

        title = getattr(stream, "current_stream_title", None) or "sin titulo"
        category = getattr(stream, "current_category", None) or getattr(stream, "current_game", None) or "sin categoria"
        playthrough = getattr(stream, "current_playthrough_type", None) or "no detectado"
        slot = getattr(stream, "current_stream_slot", None) or "no detectado"
        challenge = getattr(stream, "current_challenge", None) or "ninguno"
        language = getattr(stream, "language_mode", None)
        if not language:
            language = "ENG/ESP" if getattr(stream, "bilingual_mode", False) else "no detectado"
        spoiler_policy = getattr(stream, "spoiler_policy", None) or "no detectado"
        updated = self._format_stream_context_age(getattr(stream, "stream_context_updated_ts", 0.0) or 0.0)
        error = getattr(stream, "last_stream_context_error", None) or "ninguno"

        return (
            "Esto tengo ahora mismo:\n\n"
            f"* Estado: {status}\n"
            f"* Juego/categoria: {category}.\n"
            f"* Titulo: {title}\n"
            f"* Tipo de directo detectado: {playthrough}.\n"
            f"* Slot/tema detectado: {slot}.\n"
            f"* Challenge detectado: {challenge}.\n"
            f"* Idioma/modo: {language}.\n"
            f"* Spoilers: {spoiler_policy}.\n"
            f"* Contexto actualizado: {updated}.\n"
            f"* Ultimo error: {error}."
        )

    def _format_stream_context_age(self, updated_ts: float) -> str:
        if not updated_ts:
            return "nunca"
        elapsed = max(0, int(time.time() - float(updated_ts)))
        if elapsed < 60:
            return f"hace {elapsed} segundos"
        minutes = elapsed // 60
        if minutes < 60:
            return f"hace {minutes} minutos"
        hours = minutes // 60
        return f"hace {hours} horas"

    def _build_twitch_diagnostic_reply(self) -> str:
        stream = self._get_stream_state()
        twitch = getattr(self.runtime, "twitch", None)
        helix = getattr(twitch, "helix_client", None) if twitch is not None else None
        chat_bot = getattr(self.runtime, "twitch_chat_bot", None)
        eventsub = getattr(self.runtime, "twitch_events", None)

        channel_name = getattr(twitch, "channel_name", "") or getattr(helix, "channel_name", "")
        broadcaster_id = getattr(helix, "broadcaster_id", "")
        client_id = getattr(helix, "client_id", "")
        sender_id = getattr(getattr(twitch, "chat_client", None), "sender_id", "")
        oauth_token = os.getenv("TWITCH_OAUTH_TOKEN", "")
        broadcaster_token = os.getenv("TWITCH_BROADCASTER_OAUTH_TOKEN", "")

        def yes_no(value) -> str:
            return "yes" if bool(value) else "no"

        chat_connected = getattr(chat_bot, "is_connected", None)
        if chat_connected is None:
            chat_connected = bool(twitch and twitch.is_available())
        eventsub_connected = getattr(eventsub, "is_connected", None)

        updated = getattr(stream, "stream_context_updated_ts", 0.0) if stream else 0.0
        error = getattr(stream, "last_stream_context_error", None) if stream else None

        return (
            "Twitch diag: "
            f"channel_name loaded: {yes_no(channel_name)}; "
            f"broadcaster_id loaded: {yes_no(broadcaster_id)}; "
            f"sender_id loaded: {yes_no(sender_id)}; "
            f"client_id loaded: {yes_no(client_id)}; "
            f"TWITCH_OAUTH_TOKEN loaded: {yes_no(oauth_token)}; "
            f"TWITCH_BROADCASTER_OAUTH_TOKEN loaded: {yes_no(broadcaster_token)}; "
            f"IRC chat bot connected/available: {yes_no(chat_connected)}; "
            f"EventSub connected: {yes_no(eventsub_connected)}; "
            f"last stream context update timestamp: {updated or 'never'}; "
            f"last stream context error: {error or 'none'}."
        )

    def _build_shoutout_status_reply(self, stream) -> str:
        twitch = getattr(self.runtime, "twitch", None)
        template = getattr(twitch, "shoutout_command_template", None) or os.getenv("HEBE_SHOUTOUT_COMMAND_TEMPLATE", "!so {username}")
        raid = getattr(stream, "last_raid_event", None) or {}
        last_raider = raid.get("user_login") or raid.get("display_name") or "ninguno"
        last_so_ts = float(getattr(stream, "last_shoutout_ts", 0.0) or 0.0)
        last_so_age = self._format_stream_context_age(last_so_ts) if last_so_ts else "nunca"
        blocked = sorted(set(getattr(self, "shoutout_blocked_users", set()) or set()) | self._load_shoutout_blocked_users())

        return (
            "Estado de shoutouts:\n\n"
            f"* Auto shoutout raiders enabled: {'yes' if getattr(self, 'auto_shoutout_raiders', True) else 'no'}\n"
            f"* Shoutout command template: {template}\n"
            f"* Last raider: {last_raider}\n"
            f"* Last SO target: {getattr(stream, 'last_shoutout_target', None) or 'ninguno'}\n"
            f"* Last SO timestamp: {last_so_age}\n"
            f"* Blocked users: {', '.join(blocked) if blocked else 'ninguno'}\n"
            f"* Last SO error: {getattr(stream, 'last_shoutout_error', None) or 'ninguno'}"
        )

    def _handle_tts_manual_command(self, text: str) -> str | None:
        priority = self._handle_priority_tts_command(text)
        if priority is not None:
            return priority

        normalized = self._normalize_text(text)
        for prefix in ("hebe ", "ebe ", "eve ", "jebe "):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()

        if normalized in {"que microfono estas usando", "qué micrófono estás usando", "que micro estas usando", "lista microfonos", "lista micrófonos"}:
            return self._build_stt_microphone_reply(include_list=normalized.startswith("lista"))

        if normalized in {"reinicia stt", "reinicia el stt", "limpia error de stt", "limpia error stt"}:
            stt = getattr(self.runtime, "stt", None)
            if stt is not None and hasattr(stt, "clear_device_error"):
                stt.clear_device_error()
                self._ensure_stt_worker_running()
                self._emit_audio_status()
                return "Error de STT limpiado. Puedes probar otro micro o activar STT otra vez."
            return "No tengo servicio STT disponible para reiniciar."

        if normalized in {"prueba micro", "probar micro", "test microfono", "test micrófono"}:
            stt = getattr(self.runtime, "stt", None)
            if stt is None or not hasattr(stt, "test_input_device"):
                return "No tengo servicio STT disponible para probar el micro."
            try:
                result = stt.test_input_device(seconds=4.0)
                status = "Micro OK: entra señal." if result.get("signal_detected") else "No entra señal en este dispositivo. Prueba otro Yeti GX / host API."
                device = result.get("device") or {}
                return (
                    f"{status}\n"
                    f"* Dispositivo: {device.get('display_label') or device.get('name') or 'desconocido'}\n"
                    f"* RMS: {float(result.get('rms') or 0.0):.5f}\n"
                    f"* Peak: {float(result.get('peak') or 0.0):.5f}\n"
                    f"* Sample rate/channels: {result.get('sample_rate')}Hz / {result.get('channels')}ch"
                )
            except Exception as exc:
                return f"No he podido probar el micro: {type(exc).__name__}: {exc}"

        mic_match = re.match(r"^usa (?:el )?microfono\s+(.+)$", normalized)
        if not mic_match:
            mic_match = re.match(r"^usa (?:el )?micr[oó]fono\s+(.+)$", normalized)
        if mic_match:
            target = mic_match.group(1).strip()
            try:
                from app.services.stt_whisper import list_audio_devices

                devices = list_audio_devices()
                best = next((d for d in devices if target.lower() in str(d.get("name") or "").lower()), None)
                if not best:
                    return f"No encuentro un micrófono que coincida con {target}."
                self.apply_stt_input_device(
                    device_id=str(best.get("id") or ""),
                    device_name=str(best.get("name") or ""),
                    host_api=str(best.get("host_api") or ""),
                    sample_rate=int(best.get("default_sample_rate") or best.get("sample_rate") or 0) or None,
                    channels=int(best.get("max_input_channels") or best.get("channels") or 0) or None,
                    signature=str(best.get("signature") or ""),
                )
                return f"Micrófono STT seleccionado: {best.get('name')}."
            except Exception as exc:
                return f"No he podido cambiar el micrófono STT: {type(exc).__name__}: {exc}"

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
            print("[HEBE][TTS] global enabled=false source=command", flush=True)
            self._emit_audio_status()
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
            print("[HEBE][TTS] global enabled=true source=command", flush=True)
            self._emit_audio_status()
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
            print("[HEBE][TTS] stream enabled=false source=command", flush=True)
            self._emit_audio_status()
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
            print("[HEBE][TTS] stream enabled=true source=command", flush=True)
            self._emit_audio_status()
            return "Vale. Si toca, también hablaré en stream."

        return None

    def _normalize_voice_command_text(self, text: str) -> str:
        normalized = self._normalize_text(text)
        aliases = ("hebe", "ebe", "eve", "jebe")
        changed = True
        while changed:
            changed = False
            for alias in aliases:
                if normalized.startswith(alias + " "):
                    normalized = normalized[len(alias):].strip()
                    changed = True
                if normalized.endswith(" " + alias):
                    normalized = normalized[: -len(alias)].strip()
                    changed = True
        return normalized

    def _handle_priority_tts_command(self, text: str) -> str | None:
        normalized = self._normalize_voice_command_text(text)
        global_on = {
            "activa la voz",
            "activa tu voz",
            "activa voz",
            "activa tts",
            "activa el tts",
            "quiero escucharte",
            "habla con voz",
            "usa voz",
            "pon voz",
            "enable voice",
            "enable tts",
            "turn on voice",
            "turn on tts",
        }
        global_off = {
            "desactiva la voz",
            "desactiva tu voz",
            "desactiva voz",
            "desactiva tts",
            "desactiva el tts",
            "solo texto",
            "responde solo por texto",
            "callate la voz",
            "cállate la voz",
            "sin voz",
            "disable voice",
            "disable tts",
            "text only",
            "voice off",
        }
        if normalized in global_on:
            self.runtime.state.tts_enabled = True
            self.runtime.state.pending_tts_scope = {
                "kind": "tts_scope",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            print("[HEBE][INTENT] voice command handled before reminder parser", flush=True)
            print("[HEBE][INTENT] pending_tts_scope set", flush=True)
            self._emit_audio_status()
            return CommandResult(
                action_type="tts_enabled",
                success=True,
                user_visible_summary="Global/local TTS enabled; asking Leo whether voice should stay local or also apply to stream.",
                state_changes={"tts_enabled": True, "pending_tts_scope": True},
                constraints=["Ask only whether scope is local or stream.", "Do not imply stream TTS is enabled yet."],
                suggested_tone="short Hebe voice, useful and warm",
                fallback_text="Voz activada. ¿La quieres solo aquí/local o también para el stream?",
                requires_model_response=True,
                metadata={"message_goal": "Confirm voice is enabled and ask whether Leo wants local only or also stream scope."},
            )
        if normalized in global_off:
            self.runtime.state.tts_enabled = False
            self.runtime.state.pending_tts_scope = None
            print("[HEBE][INTENT] voice command handled before reminder parser", flush=True)
            self._emit_audio_status()
            return CommandResult(
                action_type="tts_disabled",
                success=True,
                user_visible_summary="Global TTS disabled; Hebe will answer in text.",
                state_changes={"tts_enabled": False, "pending_tts_scope": False},
                constraints=["Do not ask a follow-up question."],
                fallback_text="Vale, Leo. Me quedo en texto.",
                requires_model_response=True,
                metadata={"message_goal": "Confirm voice/TTS is disabled and Hebe will stay in text."},
            )
        if normalized in {"estado de voz", "estado de tts", "voice status", "tts status"}:
            return self._build_voice_status_reply()
        return None

    def _handle_pending_manual_intent(self, text: str) -> str | None:
        normalized = self._normalize_voice_command_text(text)
        cancel_phrases = {
            "no quiero que guardes nada",
            "no guardes nada",
            "cancela",
            "cancela eso",
            "olvida eso",
        }
        if normalized in cancel_phrases:
            if getattr(self.runtime.state, "pending_clarification", None) or getattr(self.runtime.state, "pending_reminder", None):
                self.runtime.state.pending_clarification = None
                self.runtime.state.pending_reminder = None
                print("[HEBE][INTENT] reminder pending cancelled", flush=True)
                return "Vale, no guardo nada."

        if not getattr(self.runtime.state, "pending_tts_scope", None):
            return None

        pending_tts = getattr(self.runtime.state, "pending_tts_scope", {}) or {}
        print("[HEBE][PENDING] active=pending_tts_scope", flush=True)
        print("[HEBE][INTENT] pending_tts_scope active", flush=True)
        local_phrases = {
            "local",
            "aqui",
            "aquÃ­",
            "aquí",
            "solo aqui",
            "solo aquÃ­",
            "solo aquí",
            "solo local",
            "solo conmigo",
            "solo por ahora",
            "solo para escucharte",
            "solo por ahora para poder escucharte",
            "no en stream",
            "en stream no",
            "en directo no",
            "no en directo",
            "solo quiero escucharte",
        }
        stream_phrases = {
            "stream",
            "en stream",
            "tambien stream",
            "tambien en stream",
            "tambiÃ©n en stream",
            "también en stream",
            "para stream",
            "tambien en directo",
            "tambiÃ©n en directo",
            "también en directo",
            "para el stream",
            "en directo",
            "en stream si",
            "en directo si",
            "tambien para el stream",
        }
        stream = self._get_stream_state()
        policies = getattr(stream, "policies", None) if stream else None
        if normalized not in local_phrases and normalized not in stream_phrases:
            if self._is_explicit_command_while_pending(normalized):
                self.runtime.state.pending_tts_scope = None
                print("[HEBE][PENDING] new explicit command detected; clearing pending_tts_scope", flush=True)
                print("[HEBE][INTENT] cleared pending_tts_scope", flush=True)
                return None

        if normalized in local_phrases:
            self.runtime.state.tts_enabled = True
            if policies is not None:
                policies.allow_tts_idle_prompts = False
            self.runtime.state.pending_tts_scope = None
            print("[HEBE][INTENT] resolved pending_tts_scope=local", flush=True)
            print("[HEBE][INTENT] cleared pending_tts_scope", flush=True)
            self._emit_audio_status()
            return CommandResult(
                action_type="tts_scope_resolved",
                success=True,
                user_visible_summary="Voice is enabled locally only; stream remains text-only unless Leo asks otherwise.",
                state_changes={"tts_enabled": True, "stream_idle_tts": False, "pending_tts_scope": False},
                constraints=[
                    "Do not ask for more clarification.",
                    "Do not claim stream voice is enabled.",
                    "Keep stream text-only unless Leo asks otherwise.",
                ],
                fallback_text="Perfecto, voz activada solo aquí. En stream seguiré en texto salvo que me digas lo contrario.",
                requires_model_response=True,
                metadata={
                    "scope": "local",
                    "message_goal": "Confirm to Leo that voice is enabled locally only, and stream will remain text-only unless he asks otherwise.",
                },
            )
        if normalized in stream_phrases:
            self.runtime.state.tts_enabled = True
            if policies is not None:
                policies.allow_tts_replies = True
                policies.allow_tts_event_replies = True
                policies.allow_tts_raid_thanks = True
            self.runtime.state.pending_tts_scope = None
            print("[HEBE][INTENT] resolved pending_tts_scope=stream", flush=True)
            print("[HEBE][INTENT] cleared pending_tts_scope", flush=True)
            self._emit_audio_status()
            return CommandResult(
                action_type="tts_scope_resolved",
                success=True,
                user_visible_summary="Voice is enabled locally and for stream event replies; idle spontaneity remains text-only unless Leo asks otherwise.",
                state_changes={
                    "tts_enabled": True,
                    "stream_replies_tts": True,
                    "stream_event_tts": True,
                    "stream_raid_tts": True,
                    "pending_tts_scope": False,
                },
                constraints=[
                    "Do not ask for more clarification.",
                    "Do not claim idle spontaneous voice is enabled.",
                ],
                fallback_text="Perfecto, voz activada aquí y también para eventos del stream. La espontaneidad idle sigue en texto salvo que me digas lo contrario.",
                requires_model_response=True,
                metadata={
                    "scope": "stream",
                    "message_goal": "Confirm voice is enabled locally and for stream event replies, while idle spontaneity remains text-only.",
                },
            )
        if not pending_tts.get("unclear_asked"):
            pending_tts["unclear_asked"] = True
            self.runtime.state.pending_tts_scope = pending_tts
            return "No te he entendido, Leo. ¿Local o también para stream?"

        self.runtime.state.tts_enabled = True
        if policies is not None:
            policies.allow_tts_idle_prompts = False
        self.runtime.state.pending_tts_scope = None
        print("[HEBE][INTENT] resolved pending_tts_scope=local", flush=True)
        print("[HEBE][INTENT] cleared pending_tts_scope", flush=True)
        self._emit_audio_status()
        return CommandResult(
            action_type="tts_scope_resolved",
            success=True,
            user_visible_summary="Ambiguous scope defaulted to local for safety; stream remains text-only.",
            state_changes={"tts_enabled": True, "stream_idle_tts": False, "pending_tts_scope": False},
            constraints=["Do not ask for more clarification.", "Do not claim stream voice is enabled."],
            fallback_text="Lo dejo en local por seguridad. En stream seguiré en texto salvo que me digas lo contrario.",
            requires_model_response=True,
            metadata={
                "scope": "local",
                "message_goal": "Tell Leo Hebe defaults voice scope to local for safety and stream remains text-only.",
            },
        )

    def _is_explicit_command_while_pending(self, normalized: str) -> bool:
        text = str(normalized or "").strip()
        if not text:
            return False

        try:
            plan = self._get_stream_action_planner().plan(
                InputEvent(source="typed_ui", raw_text=text, normalized_text=text)
            )
            if plan is not None:
                return True
        except Exception:
            pass

        command_prefixes = (
            "activa ",
            "desactiva ",
            "enciende ",
            "apaga ",
            "pausa ",
            "reanuda ",
            "resume ",
            "modo ",
            "actualiza ",
            "comprueba ",
            "estado ",
            "que contexto ",
            "qué contexto ",
            "guarda ",
            "finaliza ",
            "haz ",
            "hazle ",
            "dale ",
            "manda ",
            "pon ",
            "promociona ",
            "recomienda ",
            "shoutout ",
            "so ",
            "solo texto",
            "responde solo ",
            "para de hablar",
            "stop speaking",
        )
        return text.startswith(command_prefixes) or text in {
            "texto",
            "sin voz",
            "voice off",
            "text only",
        }

    def _build_voice_status_reply(self) -> str:
        stream = self._get_stream_state()
        policies = getattr(stream, "policies", None) if stream else None
        tts_backend = getattr(getattr(getattr(self.runtime, "tts", None), "__class__", None), "__name__", "unknown")
        speaking = bool(getattr(getattr(self.runtime, "tts", None), "is_speaking", False))

        def yes_no(value) -> str:
            return "yes" if bool(value) else "no"

        return (
            "Estado de voz/TTS:\n\n"
            f"* Global TTS enabled: {yes_no(getattr(self.runtime.state, 'tts_enabled', False))}\n"
            f"* TTS backend: {tts_backend}\n"
            f"* Stream idle TTS: {yes_no(getattr(policies, 'allow_tts_idle_prompts', False))}\n"
            f"* Event TTS: {yes_no(getattr(policies, 'allow_tts_event_replies', False))}\n"
            f"* Raid TTS: {yes_no(getattr(policies, 'allow_tts_raid_thanks', False))}\n"
            f"* Currently speaking: {yes_no(speaking)}"
        )

    def _build_stt_microphone_reply(self, *, include_list: bool = False) -> str:
        stt = getattr(self.runtime, "stt", None)
        selected = stt.get_selected_input_device() if stt is not None and hasattr(stt, "get_selected_input_device") else {}
        name = selected.get("device_name") or "dispositivo por defecto"
        device_id = selected.get("device_id") or "default"
        host_api = selected.get("host_api") or "desconocido"
        sample_rate = selected.get("sample_rate") or "?"
        channels = selected.get("channels") or "?"
        rms = float(selected.get("last_rms") or 0.0)
        peak = float(selected.get("last_peak") or selected.get("last_level") or 0.0)
        error = selected.get("error") or "ninguno"
        lines = [
            "Micrófono STT:",
            f"* Seleccionado: {name}",
            f"* ID: {device_id}",
            f"* Host API: {host_api}",
            f"* Sample rate/channels: {sample_rate}Hz / {channels}ch",
            f"* RMS actual: {rms:.5f}",
            f"* Peak actual: {peak:.5f}",
            f"* Último error: {error}",
        ]
        if include_list:
            try:
                from app.services.stt_whisper import list_audio_devices

                devices = list_audio_devices()
                if devices:
                    lines.append("* Disponibles:")
                    for device in devices[:12]:
                        marker = " (default)" if device.get("is_default_input") else ""
                        lines.append(f"  - {device.get('display_label') or device.get('name')}{marker}")
                else:
                    lines.append("* Disponibles: ninguno")
            except Exception as exc:
                lines.append(f"* No he podido listar micrófonos: {type(exc).__name__}: {exc}")
        return "\n".join(lines)

    def apply_stt_input_device(
        self,
        *,
        device_id: str = "",
        device_name: str = "",
        host_api: str = "",
        sample_rate: int | None = None,
        channels: int | None = None,
        signature: str = "",
    ) -> dict:
        stt = getattr(self.runtime, "stt", None)
        if stt is None or not hasattr(stt, "set_input_device"):
            raise RuntimeError("STT service is not available")
        selected = stt.set_input_device(
            device_id=device_id,
            device_name=device_name,
            host_api=host_api,
            sample_rate=sample_rate,
            channels=channels,
            signature=signature,
        )
        print("[HEBE][STT][DEVICE] restarted with selected input", flush=True)
        self._ensure_stt_worker_running()
        self._emit_audio_status()
        return selected

    def _ensure_stt_worker_running(self) -> bool:
        if not getattr(self.runtime, "stt_enabled", False):
            return False
        if self._stop_event.is_set():
            return False
        if self._stt_worker is not None and self._stt_worker.is_running():
            return True
        self._stt_worker = STTWorker(
            stt=self.runtime.stt,
            stop_event=self._stop_event,
        )
        self._stt_worker.start()
        print("[HEBE][STT] worker started after user action", flush=True)
        return True

    def _synthesize_command_result(self, result: CommandResult, *, input_text: str | None = None) -> str:
        synthesizer = getattr(self, "response_synthesizer", None)
        if synthesizer is None:
            print("[HEBE][RESPONSE_SYNTH] model_response_used=false fallback_used=true reason=no_synthesizer", flush=True)
            return result.fallback_text or result.user_visible_summary
        try:
            text = synthesizer.synthesize_command_result(
                result,
                input_text=input_text,
                state=getattr(self.runtime, "state", None),
            )
            fallback = result.fallback_text or result.user_visible_summary
            used_fallback = bool(fallback and text == fallback)
            print(
                "[HEBE][RESPONSE_SYNTH] "
                f"model_response_used={not used_fallback} fallback_used={used_fallback}",
                flush=True,
            )
            return text
        except Exception as exc:
            print(f"[HEBE][COMMAND_RESULT] synth failed: {exc!r}", flush=True)
            print("[HEBE][RESPONSE_SYNTH] model_response_used=false fallback_used=true", flush=True)
            return result.fallback_text or result.user_visible_summary

    def _emit_audio_status(self) -> None:
        try:
            stream = getattr(self.runtime.state, "stream", None)
            policies = getattr(stream, "policies", None) if stream else None
            stt = getattr(self.runtime, "stt", None)
            stt_device = stt.get_selected_input_device() if stt is not None and hasattr(stt, "get_selected_input_device") else None
            emit(
                "status",
                {
                    "tts_enabled": bool(getattr(self.runtime.state, "tts_enabled", False)),
                    "stream_tts_enabled": bool(getattr(policies, "allow_tts_replies", False)),
                    "stt_enabled": bool(getattr(self.runtime, "stt_enabled", False)),
                    "stt": getattr(stt, "status", "off") if stt is not None else "off",
                    "last_stt_error": getattr(stt, "last_input_device_error", None) if stt is not None else None,
                    "stt_input_device": stt_device,
                },
            )
        except Exception:
            pass

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
        if source not in {"ui", "voice", "stt_voice"}:
            return False

        reply_step = execution.first_result_of_type("reply") if execution else None
        if not reply_step:
            return False

        return reply_step.data.get("mode") == "chat"

    def _deliver_twitch_reply(self, text: str, *, event_type: str | None = None, payload: dict | None = None) -> None:
        """
        Entrega un reply al chat de Twitch.
        Si las policies del stream lo permiten, también lo hablamos por TTS.
        """
        twitch = getattr(self.runtime, "twitch", None)
        stream = getattr(self.runtime.state, "stream", None)
        if stream is not None:
            stream.last_hebe_stream_speak_ts = time.time()
            if event_type == "twitch_idle_prompt":
                topic = (payload or {}).get("idle_topic")
                service = getattr(self, "stream_spontaneity", None)
                if service is not None:
                    service.record_idle_message(stream, text, topic=topic)
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
        allow_tts = bool(policies and getattr(policies, "allow_tts_replies", False))
        if event_type == "twitch_idle_prompt":
            allow_tts = bool(policies and getattr(policies, "allow_tts_idle_prompts", False))
        elif event_type == "twitch_raid":
            allow_tts = bool(policies and getattr(policies, "allow_tts_raid_thanks", True))
        elif event_type and event_type.startswith("twitch_") and event_type != "twitch_chat_react":
            allow_tts = bool(policies and getattr(policies, "allow_tts_event_replies", True))
        if not allow_tts:
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
            safe_text = str(text or "").replace('"', '\\"')
            print(f"[HEBE][TTS] speaking text=\"{safe_text}\"", flush=True)
            self.runtime.speak(text)
        except Exception as e:
            safe_error = str(e).replace('"', '\\"')
            print(f"[HEBE][TTS] failed error=\"{safe_error}\"", flush=True)

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
            self.poll_stream_context(require_enabled=False)
            self.poll_stream_presence()

            command = None
            source = None

            try:
                ui_inbox = get_ui_inbox()
                raw_ui_command = ui_inbox.get_nowait()
                print(f"[HEBE] UI inbox -> {raw_ui_command!r}", flush=True)
                source = "ui"
                command = self._normalize_text(str(raw_ui_command))
                self._current_input_event = self._build_input_event(
                    source="typed_ui",
                    raw_text=str(raw_ui_command),
                    normalized_text=command,
                )
            except Empty:
                pass

            if not command:
                try:
                    voice_inbox = get_voice_inbox()
                    raw_voice_command = voice_inbox.get_nowait()
                    print(f"[HEBE] VOICE inbox -> {raw_voice_command!r}", flush=True)
                    source = "stt_voice"
                    normalization = self._normalize_stt_input(str(raw_voice_command))
                    command = normalization.normalized_text
                    self._current_input_event = self._build_input_event(
                        source="stt_voice",
                        raw_text=str(raw_voice_command),
                        normalized_text=command,
                        stt_metadata=normalization.as_event(),
                    )
                except Empty:
                    pass

            if not command:
                time.sleep(0.02)
                continue

            if source in {"voice", "stt_voice", "ui"}:
                if source in {"voice", "stt_voice"} and self._is_stream_enabled():
                    voice_type, mood_hint = self._classify_voice_event(command)
                    has_action_intent = self._input_event_has_action_intent(getattr(self, "_current_input_event", None))
                    ambient_enabled = bool(getattr(self, "stream_ambient_stt_enabled", False))
                    if voice_type == "direct_command_to_hebe" or ambient_enabled:
                        self._record_voice_event(command, voice_type, mood_hint)
                    logged_text = command if voice_type == "direct_command_to_hebe" else "(ambient)"
                    print(
                        f"[HEBE][VOICE] type={voice_type} mood={mood_hint!r} text={logged_text!r}",
                        flush=True,
                    )
                    if voice_type != "direct_command_to_hebe" and not self._stream_is_armed() and not has_action_intent:
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
            self._current_input_event = None

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
            self.poll_stream_context(require_enabled=False)
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
                raw_voice_command = voice_inbox.get_nowait()
            except Empty:
                time.sleep(0.02)
                continue

            # Si stream mode está activo, dejamos que command_loop gestione
            # el gate fino con wakeword corto tipo "hebe"/"eve".
            if self._is_stream_enabled():
                submit_text_from_voice(str(raw_voice_command))
                res = self.command_loop()
                if res == "stop":
                    return "stop"
                continue

            command = self._normalize_text(str(raw_voice_command))

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
