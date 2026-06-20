import os
import re
import time
import threading
import unicodedata
import hashlib
import uuid
from dataclasses import replace
from types import SimpleNamespace
from queue import Empty
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.services.db_sqlite import (
    DB_PATH,
    cleanup_stt_prompt_injection_rows,
    init_db,
    log_chat,
    seed_default_apps,
)
from app.services.vts_client import vts_hotkey
from app.services.voice_command_recovery import TranscriptNormalizationResult, normalize_stt_transcript
from app.services.stt_whisper import is_stt_prompt_injection
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
from app.cognitive.input_event import InputEnvelope, InputEvent
from app.cognitive.action_plan import ActionPlan
from app.cognitive.stream_companion_flow import (
    ConversationState,
    ContextRelevance,
    ConversationStateResolver,
    InputClassifier,
    KnowledgePolicyResolver,
    ResponseDecisionResolver,
    ResponseFrame,
)
from app.cognitive.wake_name_resolver import WakeNameResolver
from app.cognitive.context_builder import ContextBuilder
from app.cognitive.cognitive_router import CognitiveRouter
from app.cognitive.deliberation_service import DeliberationService
from app.cognitive.local_app_planner import LocalAppActionPlanner
from app.cognitive.plan_executor import PlanExecutor
from app.cognitive.response_synthesizer import ResponseSynthesizer
from app.cognitive.action_runtime import ActionRuntime
from app.cognitive.memory.memory_extractor import MemoryExtractor
from app.stream.context_sync import StreamContextSyncService
from app.stream.game_knowledge import GameKnowledgeConfig, GameKnowledgeResolver
from app.stream.game_profiles import GameProfileStore
from app.stream.game_research import GameKnowledgeResearchConfig, GameKnowledgeResearchService
from app.stream import memory as stream_memory
from app.stream.live_session import LiveSessionBrain, init_live_session_schema
from app.stream.ambient_context import AmbientContextExtractor
from app.stream.action_planner import StreamActionPlanner
from app.stream.policy import (
    active_behavior_blocks,
    PolicyDecision,
    ViewerIntentPolicy,
    apply_owner_game_activity_correction,
    filter_ambient_facts_for_activity,
    owner_behavior_decision,
    policy_trace,
)
from app.stream.input_firewall import (
    ACTION_PROMOTION_SHOUTOUT,
    ACTION_TWITCH_ACTION,
    ACTION_TWITCH_REPLY,
    InputAuthorityFirewall,
    InputFirewallDecision,
    is_known_bot_username,
    looks_like_media_or_singing,
)
from app.stream import session_primer
from app.stream.spontaneity import StreamSpontaneityConfig, StreamSpontaneityService

WAKE_WORDS = ["hebe despierta", "eve despierta", "jebe despierta"]
STREAM_WAKE_ALIASES = {"hebe", "ebe", "eve", "ehbe", "heve", "ebi", "heb", "jebe"}
STREAM_WAKE_MULTI_ALIASES = ("eh ve", "e ve", "e be", "hey be", "he ve")
OUTPUT_TARGET_LOCAL_UI = "local_ui"
OUTPUT_TARGET_LOCAL_TTS = "local_tts"
OUTPUT_TARGET_STREAM_TTS = "stream_tts"
OUTPUT_TARGET_TWITCH_CHAT = "twitch_chat"
OUTPUT_TARGET_TWITCH_COMMAND = "twitch_command"
OUTPUT_TARGET_SILENT_CONTEXT_UPDATE = "silent_context_update"

t0 = time.time()


def mark(stage):
    emit("status", {"engine": "starting", "stage": stage, "t_ms": int((time.time() - t0) * 1000)})


class HebeEngine:
    """Motor principal de Hebe ejecutándose en un hilo."""

    def __init__(self, runtime: HebeRuntime, use_wakeword: bool = True, say_hello: bool = False):
        self.runtime = runtime
        self._stt_worker: STTWorker | None = None
        self._wake_loop_alive = False
        self._wake_loop_last_error = ""
        self.say_hello = say_hello
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self.use_wakeword = use_wakeword
        self.start_awake = os.getenv("HEBE_START_AWAKE", "true").strip().lower() in ("1", "true", "yes", "on")
        if self.start_awake:
            self.runtime.state.hebe_sleeping = False
            self.runtime.state.mode = "active"
            print("[HEBE][WAKE] startup hebe_sleeping=false reason=start_awake_default", flush=True)
        else:
            self.runtime.state.hebe_sleeping = bool(getattr(self.runtime.state, "hebe_sleeping", False))
            print(
                f"[HEBE][WAKE] startup hebe_sleeping={bool(self.runtime.state.hebe_sleeping)} reason=config",
                flush=True,
            )

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
        self.cognitive_router = CognitiveRouter()
        self.wake_name_resolver = WakeNameResolver()

        # Conectar Twitch events al scheduler
        if hasattr(self.runtime, 'twitch_events') and self.runtime.twitch_events:
            self.runtime.twitch_events.push_event_callback = lambda event_type, payload: self.scheduler.push_event(event_type, payload)

        if hasattr(self.runtime, 'twitch_chat_bot') and self.runtime.twitch_chat_bot:
            def _twitch_ambient_callback(username, display_name, text, channel):
                self.observe_twitch_chat_message(username, display_name, text, channel)

            def _twitch_chat_callback(username, display_name, text, channel):
                firewall = self._input_firewall_decision(
                    source="twitch_viewer",
                    text=text,
                    username=username,
                    event_type="twitch_chat_react",
                    addressed_to_hebe=self._message_mentions_hebe(text),
                )
                if not self._firewall_allows_pipeline(firewall):
                    return
                self.observe_twitch_chat_message(username, display_name, text, channel, firewall_decision=firewall)
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
        self.local_app_planner = LocalAppActionPlanner(self.wake_name_resolver)

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
        self.game_knowledge = GameKnowledgeResolver(
            profile_store=self.game_profiles,
            research_service=self.game_research,
            config=GameKnowledgeConfig.from_env(),
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
                require_specific_context=os.getenv("HEBE_SPONTANEITY_REQUIRE_SPECIFIC_CONTEXT", "true").strip().lower() in ("1", "true", "yes", "on"),
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
        self.live_session_brain = LiveSessionBrain(getattr(self.runtime.state, "stream", None))
        initial_output_mode = os.getenv("HEBE_STREAM_OUTPUT_MODE", "").strip()
        if initial_output_mode in {"ui_only", "tts_enabled", "twitch_chat_only", "silent"}:
            stream_state = getattr(self.runtime.state, "stream", None)
            if stream_state is not None:
                stream_state.stream_output_mode = initial_output_mode
                print(f"[HEBE][OUTPUT_MODE] mode={initial_output_mode} reason=config", flush=True)
        self._apply_stream_performance_profile()
        self.ambient_context_extractor = AmbientContextExtractor()
        self.memory_extractor = MemoryExtractor(
            intent_model=getattr(self.runtime, "intent_llm", None),
        )
        self.input_classifier = InputClassifier()
        self.conversation_state_resolver = ConversationStateResolver()
        self.knowledge_policy_resolver = KnowledgePolicyResolver()
        self.response_decision_resolver = ResponseDecisionResolver()
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
        self.spontaneous_twitch_chat_enabled = os.getenv(
            "HEBE_SPONTANEOUS_TWITCH_CHAT_ENABLED",
            os.getenv("HEBE_TWITCH_SPONTANEOUS_ENABLED", "false"),
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
        self.stt_ignore_while_tts_speaking = os.getenv(
            "HEBE_STT_IGNORE_WHILE_TTS_SPEAKING",
            "true",
        ).strip().lower() in ("1", "true", "yes", "on")
        self.stt_tts_echo_window_seconds = float(os.getenv("HEBE_STT_TTS_ECHO_WINDOW_SECONDS", "10") or 10)
        self.stt_tts_echo_similarity_threshold = float(os.getenv("HEBE_STT_TTS_ECHO_SIMILARITY_THRESHOLD", "0.82") or 0.82)
        self.stt_tts_echo_grace_seconds = float(os.getenv("HEBE_STT_TTS_ECHO_GRACE_SECONDS", "2.5") or 2.5)
        self.stt_log_rejected_raw = os.getenv("HEBE_STT_LOG_REJECTED_RAW", "false").strip().lower() in ("1", "true", "yes", "on")
        self.stt_auto_disable_prompt_on_echo = os.getenv("HEBE_STT_AUTO_DISABLE_PROMPT_ON_ECHO", "true").strip().lower() in ("1", "true", "yes", "on")
        self.stt_prompt_echo_window_seconds = float(os.getenv("HEBE_STT_PROMPT_ECHO_WINDOW_SECONDS", "300") or 300)
        self.stt_prompt_echo_disable_threshold = int(os.getenv("HEBE_STT_PROMPT_ECHO_DISABLE_THRESHOLD", "2") or 2)
        self.pending_conversation_ttl_seconds = float(os.getenv("HEBE_PENDING_CONVERSATION_TTL_SECONDS", "120") or 120)
        self.pending_conversation_max_followups = int(os.getenv("HEBE_PENDING_CONVERSATION_MAX_FOLLOWUPS", "1") or 1)
        self.stt_duplicate_window_seconds = float(os.getenv("HEBE_STT_DUPLICATE_WINDOW_SECONDS", "8") or 8)
        self.stt_duplicate_similarity_threshold = float(os.getenv("HEBE_STT_DUPLICATE_SIMILARITY_THRESHOLD", "0.92") or 0.92)
        self._recent_tts_texts: list[dict] = []
        self._last_tts_text = ""
        self._last_tts_normalized = ""
        self._last_tts_message_id = ""
        self._tts_started_at = 0.0
        self._tts_until = 0.0
        self._tts_active = False
        self._recent_stt_transcripts: list[dict] = []
        self._stt_prompt_echo_rejection_ts: list[float] = []
        self._stt_visible_transcripts: set[str] = set()
        self.stream_action_planner = self._build_stream_action_planner()
        self.viewer_intent_policy = ViewerIntentPolicy()
        self.input_authority_firewall = self._build_input_firewall()
        self._last_input_firewall: dict = {}
        self._last_policy_trace: dict = {}
        self._last_cognitive_trace: dict = {}
        self._current_input_event: InputEvent | None = None

    def _build_input_firewall(self) -> InputAuthorityFirewall:
        return InputAuthorityFirewall(extra_bot_usernames=self._input_firewall_bot_usernames())

    def _input_firewall_bot_usernames(self) -> set[str]:
        stream = self._get_stream_state()
        twitch = getattr(self.runtime, "twitch", None)
        configured = os.getenv("HEBE_TWITCH_BOT_USERNAMES", "")
        names = {
            "hebenifelheim",
            getattr(stream, "bot_username", "") if stream else "",
            getattr(twitch, "bot_username", "") if twitch else "",
            getattr(getattr(self.runtime, "twitch_chat_bot", None), "bot_username", ""),
        }
        names.update(part.strip() for part in configured.split(",") if part.strip())
        return {str(name or "").strip().lower().lstrip("@") for name in names if str(name or "").strip()}

    def _get_input_firewall(self) -> InputAuthorityFirewall:
        firewall = getattr(self, "input_authority_firewall", None)
        if firewall is None:
            firewall = self._build_input_firewall()
            self.input_authority_firewall = firewall
        return firewall

    def _current_stream_is_live(self) -> bool:
        stream = self._get_stream_state()
        return bool(stream and getattr(stream, "is_live", False))

    def _input_firewall_decision(
        self,
        *,
        source: str,
        text: str | None = "",
        username: str | None = "",
        event_type: str | None = "",
        addressed_to_hebe: bool = False,
        pending_followup: bool = False,
        has_action_intent: bool = False,
        record: bool = True,
    ) -> InputFirewallDecision:
        decision = self._get_input_firewall().decide(
            source=source,
            text=text,
            username=username,
            stream_is_live=self._current_stream_is_live(),
            is_simulation=False,
            addressed_to_hebe=addressed_to_hebe,
            pending_followup=pending_followup,
            has_action_intent=has_action_intent,
            event_type=event_type,
        )
        if record:
            self._record_input_firewall(decision, text=text)
        return decision

    def _record_input_firewall(self, decision: InputFirewallDecision, *, text: str | None = "") -> None:
        payload = decision.as_dict()
        self._last_input_firewall = payload
        print(
            "[HEBE][INPUT_FIREWALL] "
            f"source={decision.source} authority={decision.authority} "
            f"trust={decision.input_trust} decision={decision.firewall_decision} "
            f"reason={decision.reason}",
            flush=True,
        )
        if decision.media_or_singing_detected:
            print(
                f"[HEBE][MEDIA_GATE] detected=true reason={decision.media_reason or 'singing_or_lyrics'}",
                flush=True,
            )
        if decision.bot_detected:
            print(
                f"[HEBE][BOT_FILTER] ignored username={decision.username} reason=known_bot",
                flush=True,
            )
        if decision.reason == "offline_stream":
            if decision.firewall_decision.startswith("block") or decision.firewall_decision == "ignore":
                print(
                    f"[HEBE][STREAM_GATE] blocked reason=offline_stream source={decision.source}",
                    flush=True,
                )
            elif decision.authority == "owner":
                print("[HEBE][STREAM_GATE] allowed reason=owner_local_command", flush=True)
        elif decision.authority == "owner" and not decision.stream_is_live:
            print("[HEBE][STREAM_GATE] allowed reason=owner_local_command", flush=True)
        for action in (ACTION_TWITCH_REPLY, ACTION_TWITCH_ACTION, ACTION_PROMOTION_SHOUTOUT):
            if decision.blocks_action(action):
                print(
                    f"[HEBE][ACTION_PERMISSIONS] action={action} allowed=false reason={decision.reason}",
                    flush=True,
                )

    def _firewall_allows_pipeline(self, decision: InputFirewallDecision | None) -> bool:
        if decision is None:
            return True
        return decision.firewall_decision in {"allow", "allow_context_only"}

    def _firewall_payload(self) -> dict:
        return dict(getattr(self, "_last_input_firewall", {}) or {})

    def _build_stream_action_planner(self) -> StreamActionPlanner:
        return StreamActionPlanner(
            known_targets_provider=self._known_voice_command_targets,
            normalize_target=self._normalize_shoutout_target,
            build_shoutout_command=self._build_shoutout_command_preview,
            stream_state_provider=self._get_stream_state,
            target_resolver=self._resolve_twitch_target_details,
        )

    def _get_viewer_intent_policy(self) -> ViewerIntentPolicy:
        policy = getattr(self, "viewer_intent_policy", None)
        if policy is None:
            policy = ViewerIntentPolicy()
            self.viewer_intent_policy = policy
        return policy

    def _get_live_session_brain(self) -> LiveSessionBrain:
        brain = getattr(self, "live_session_brain", None)
        stream = self._get_stream_state()
        if brain is None:
            brain = LiveSessionBrain(stream)
            self.live_session_brain = brain
        else:
            try:
                brain.sync_stream_metadata(stream)
            except Exception as exc:
                print(f"[HEBE][LIVE_SESSION] sync failed: {exc!r}", flush=True)
        if stream is not None:
            try:
                stream.live_session_context = brain.state.as_dict()
            except Exception:
                pass
        return brain

    def _record_policy_trace(self, trace: dict | None) -> dict:
        clean = dict(trace or {})
        if not clean:
            return clean
        clean.setdefault("event_id", f"policy_{uuid.uuid4().hex}")
        clean.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        firewall = self._firewall_payload()
        if firewall:
            clean.setdefault("input_firewall", firewall)
        self._last_policy_trace = clean
        stream = self._get_stream_state()
        if stream is not None:
            try:
                stream.last_policy_trace = clean
            except Exception:
                pass
        print(
            f"[HEBE][AUTHORITY] speaker={clean.get('speaker')} authority={clean.get('authority')}",
            flush=True,
        )
        print(
            "[HEBE][INTENT] "
            f"intent={clean.get('intent')} "
            f"requested_behavior={clean.get('requested_behavior')} "
            f"matched_by={clean.get('matched_by')}",
            flush=True,
        )
        print(
            "[HEBE][POLICY] "
            f"decision={clean.get('policy_decision')} "
            f"reason={clean.get('reason')} "
            f"allow_free_llm={clean.get('allow_free_llm')} "
            f"response_intent={clean.get('response_intent')}",
            flush=True,
        )
        blocks = active_behavior_blocks(stream) if stream is not None else []
        print(f"[HEBE][BEHAVIOR_BLOCK] active={blocks}", flush=True)
        return clean

    def _update_policy_trace_response(
        self,
        reply_text: str,
        *,
        response_mode: str = "llm",
        response_source: str = "hybrid",
        style_guard_triggered: bool = False,
        was_generic_refusal_rewritten: bool = False,
    ) -> None:
        trace = self.get_last_policy_trace()
        if not trace:
            return
        updated = dict(trace)
        updated["hebe_response"] = str(reply_text or "")
        updated["final_response"] = str(reply_text or "")
        updated["response_mode"] = response_mode
        updated["response_source"] = response_source
        updated["style_guard_triggered"] = bool(style_guard_triggered)
        updated["was_generic_refusal_rewritten"] = bool(was_generic_refusal_rewritten)
        self._last_policy_trace = updated
        stream = self._get_stream_state()
        if stream is not None:
            try:
                stream.last_policy_trace = updated
            except Exception:
                pass
        print(f"[HEBE][RESPONSE_SOURCE] source={response_source}", flush=True)

    def _synthesize_policy_reply(
        self,
        decision: PolicyDecision,
        *,
        input_text: str,
        source: str,
        speaker: str,
    ) -> dict:
        if not decision.allow_reply:
            return {"text": "", "response_source": "silent"}
        directive = str(getattr(decision, "response_directive", "") or "").strip()
        if not directive:
            return {"text": "", "response_source": "silent"}
        policy_payload = {
            "policy_decision": "blocked",
            "reason": decision.reason,
            "intent": decision.intent,
            "requested_behavior": decision.requested_behavior,
            "behavior_family": decision.behavior_family,
            "target": decision.target,
            "response_intent": decision.response_intent or "hebe_playful_boundary",
            "response_tone": decision.response_tone or "sarcastic_playful_stream_safe",
            "must_include": list(getattr(decision, "must_include", []) or []),
            "must_not_include": list(getattr(decision, "must_not_include", []) or []),
        }
        synthesizer = getattr(self, "response_synthesizer", None)
        if synthesizer is not None and hasattr(synthesizer, "synthesize_policy_boundary_response"):
            return synthesizer.synthesize_policy_boundary_response(
                policy=policy_payload,
                input_text=input_text,
                speaker=speaker,
                source=source,
            )
        print(
            f"[HEBE][POLICY] reply_generation_failed reason={decision.reason} source=no_persona_response_layer",
            flush=True,
        )
        return {"text": "", "response_source": "silent"}

    def get_last_policy_trace(self) -> dict:
        stream = self._get_stream_state()
        stream_trace = getattr(stream, "last_policy_trace", None) if stream is not None else None
        return dict(stream_trace or getattr(self, "_last_policy_trace", {}) or {})

    def get_active_behavior_blocks(self) -> list[dict]:
        stream = self._get_stream_state()
        if stream is None:
            return []
        return list(active_behavior_blocks(stream))

    def clear_active_behavior_blocks(self) -> list[dict]:
        stream = self._get_stream_state()
        if stream is None:
            return []
        stream.active_behavior_blocks = []
        self._record_policy_trace({
            "source": "dev",
            "speaker": "dev",
            "authority": "system",
            "addressed_to_hebe": False,
            "intent": "clear_behavior_blocks",
            "requested_behavior": "all",
            "policy_decision": "allowed",
            "reason": "dev_clear",
            "response_mode": "silent",
        })
        return []

    def simulate_twitch_message(self, payload: dict) -> dict:
        viewer_name = str((payload or {}).get("viewer_name") or (payload or {}).get("user_login") or (payload or {}).get("username") or "viewer").strip()
        display_name = str((payload or {}).get("display_name") or viewer_name).strip()
        text = str((payload or {}).get("text") or (payload or {}).get("message_text") or "").strip()
        channel = str((payload or {}).get("channel") or "").strip()
        event_payload = {
            **(payload or {}),
            "display_name": display_name,
            "user_login": viewer_name,
            "username": viewer_name,
            "message_text": text,
            "channel": channel,
            "_simulated": True,
        }
        firewall = self._input_firewall_decision(
            source="twitch_viewer",
            text=text,
            username=viewer_name,
            event_type="twitch_chat_react",
            addressed_to_hebe=self._message_mentions_hebe(text),
        )
        event_payload["input_firewall"] = firewall.as_dict()
        self.process_internal_event(InternalEvent(
            event_type="twitch_chat_react",
            payload=event_payload,
            created_at=datetime.now(timezone.utc).isoformat(),
        ))
        return self._simulation_debug_payload()

    def simulate_internal_twitch_event(self, *, event_type: str = "twitch_raid", stream_live: bool = False) -> dict:
        stream = self._get_stream_state()
        old_live = getattr(stream, "is_live", False) if stream is not None else False
        old_known = getattr(stream, "live_status_known", False) if stream is not None else False
        if stream is not None:
            stream.is_live = bool(stream_live)
            stream.live_status_known = True
        try:
            self.process_internal_event(InternalEvent(
                event_type=event_type,
                payload={"display_name": "SimulatedRaider", "user_login": "simulatedraider", "viewer_count": 12, "_simulated": True},
                created_at=datetime.now(timezone.utc).isoformat(),
            ))
            return self._simulation_debug_payload(extra={"simulated_stream_live": bool(stream_live)})
        finally:
            if stream is not None:
                stream.is_live = old_live
                stream.live_status_known = old_known

    def simulate_leo_message(self, text: str, *, source: str = "ui", pending_kind: str | None = None) -> dict:
        clean_source = source if source in {"ui", "stt_voice"} else "ui"
        if pending_kind == "appointment_datetime":
            now_ts = time.time()
            self.runtime.state.pending_clarification = {
                "id": f"simulation_pending_{uuid.uuid4().hex}",
                "kind": "appointment_datetime",
                "authority": "owner",
                "created_at": now_ts,
                "expires_at": now_ts + 300,
                "draft": {"title": "Consulta", "source_text": "simulated appointment request"},
            }
        before_event_id = self.get_last_policy_trace().get("event_id")
        previous_simulation_mode = bool(getattr(self, "_manual_simulation_mode", False))
        self._manual_simulation_mode = True
        try:
            self.cognitive_flow(str(text or "").strip(), source=clean_source)
        finally:
            self._manual_simulation_mode = previous_simulation_mode
        after_event_id = self.get_last_policy_trace().get("event_id")
        if before_event_id == after_event_id:
            self._record_policy_trace(policy_trace(
                source=clean_source,
                speaker="Leo",
                text=str(text or "").strip(),
                decision=PolicyDecision(
                    allow_reply=True,
                    allow_llm=True,
                    reason="owner_allowed",
                    intent="owner_message",
                ),
                addressed_to_hebe=True,
                authority="owner",
            ))
        return self._simulation_debug_payload()

    def simulate_ambient_stt(self, text: str) -> dict:
        clean_text = str(text or "").strip()
        voice_type, mood_hint = self._classify_voice_event(clean_text)
        relevance = ContextRelevance(useful=False, category="none", reason="not_stream_enabled")
        firewall = self._input_firewall_decision(
            source="ambient_stt",
            text=clean_text,
            event_type=voice_type,
            addressed_to_hebe=False,
        )
        if firewall.firewall_decision == "allow_context_only" and self._is_stream_enabled():
            relevance = self._record_voice_event(clean_text, voice_type, mood_hint)
        decision = PolicyDecision(
            allow_reply=False,
            allow_llm=False,
            reason=firewall.reason if firewall.firewall_decision != "allow_context_only" else (
                "ambient_context_updated" if getattr(relevance, "useful", False) else "ambient_stt_observed"
            ),
            intent=voice_type or "ambient_stt",
            requested_behavior="ambient_context",
        )
        self._record_policy_trace(policy_trace(
            source="ambient_stt",
            speaker="ambient_stt",
            text=clean_text,
            decision=decision,
            addressed_to_hebe=False,
            authority="ambient",
            requested_behavior="ambient_context",
        ))
        return self._simulation_debug_payload(extra={"relevance": getattr(relevance, "__dict__", {})})

    def _simulation_debug_payload(self, *, extra: dict | None = None) -> dict:
        stream = self._get_stream_state()
        trace = self.get_last_policy_trace()
        cognitive = dict(getattr(self, "_last_cognitive_trace", {}) or {})
        game_state = {
            "game": getattr(stream, "current_game", None) or getattr(stream, "current_category", None) if stream is not None else None,
            "current_activity": getattr(stream, "current_activity", None) if stream is not None else None,
            "combat_state": getattr(stream, "combat_state", None) if stream is not None else None,
            "last_owner_correction": getattr(stream, "last_owner_correction", None) if stream is not None else None,
            "blocked_comment_categories": list(getattr(stream, "blocked_comment_categories", []) or []) if stream is not None else [],
        }
        payload = {
            "ok": True,
            "event_id": trace.get("event_id"),
            "source": trace.get("source"),
            "speaker": trace.get("speaker"),
            "display_name": trace.get("speaker"),
            "authority": trace.get("authority"),
            "addressed_to_hebe": trace.get("addressed_to_hebe"),
            "intent": cognitive.get("intent") or trace.get("intent"),
            "requested_behavior": trace.get("requested_behavior"),
            "behavior_family": trace.get("behavior_family"),
            "target": trace.get("target"),
            "matched_by": trace.get("matched_by"),
            "policy_decision": trace.get("policy_decision"),
            "reason": cognitive.get("reason") or trace.get("reason"),
            "response_mode": cognitive.get("response_mode") or trace.get("response_mode"),
            "response_source": trace.get("response_source"),
            "allow_free_llm": trace.get("allow_free_llm"),
            "execute_as_command": trace.get("execute_as_command"),
            "style_guard_triggered": trace.get("style_guard_triggered"),
            "was_generic_refusal_rewritten": trace.get("was_generic_refusal_rewritten"),
            "hebe_response": trace.get("hebe_response") or "",
            "final_response": cognitive.get("final_response") or trace.get("final_response") or trace.get("hebe_response") or "",
            "last_policy_decision": trace,
            "cognitive_route": cognitive,
            "raw_input": cognitive.get("raw_text") or cognitive.get("input_text"),
            "normalized_input": cognitive.get("normalized_text"),
            "active_pending_task": cognitive.get("active_pending_task") or cognitive.get("pending_task_id"),
            "pending_compatibility": cognitive.get("pending_compatible"),
            "is_new_request": cognitive.get("is_new_request"),
            "uses_pending_task": cognitive.get("uses_pending_task"),
            "allowed_capabilities": cognitive.get("allowed_capabilities") or cognitive.get("required_capability_ids") or [],
            "blocked_capabilities": cognitive.get("blocked_capabilities") or cognitive.get("blocked_capability_ids") or [],
            "selected_route": cognitive.get("selected_route") or cognitive.get("intent"),
            "should_reply": cognitive.get("should_reply"),
            "final_plan_steps": cognitive.get("final_plan_steps") or [],
            "should_stop_pipeline": cognitive.get("should_stop_pipeline"),
            "plan_executor_guard": cognitive.get("plan_executor_guard") or [],
            "input_firewall": self._firewall_payload(),
            "is_simulation": True,
            "behavior_blocks": self.get_active_behavior_blocks(),
            "game_state": game_state,
            "cooldowns": dict(getattr(stream, "viewer_policy_cooldowns", {}) or {}) if stream is not None else {},
            "timeline": self._policy_timeline(trace),
            "stream_output_mode": self._stream_output_mode(),
            "current_activity": game_state["current_activity"],
            "combat_state": game_state["combat_state"],
        }
        if extra:
            payload.update(extra)
        return payload

    def _policy_timeline(self, trace: dict) -> list[str]:
        if not trace:
            return []
        text = str(trace.get("text") or "").replace('"', '\\"')
        response = str(trace.get("hebe_response") or "").replace('"', '\\"')
        source = trace.get("source") or "unknown"
        speaker = trace.get("speaker") or "unknown"
        lines = [
            f"[SIM] source={source} speaker={speaker} text=\"{text}\"",
            f"[AUTHORITY] authority={trace.get('authority') or 'unknown'}",
            f"[INTENT] intent={trace.get('intent') or 'unknown'} requested_behavior={trace.get('requested_behavior') or 'unknown'} matched_by={trace.get('matched_by') or []}",
            f"[POLICY] decision={trace.get('policy_decision') or 'unknown'} reason={trace.get('reason') or 'unknown'} allow_free_llm={trace.get('allow_free_llm')}",
        ]
        if response:
            lines.append(f"[OUTPUT] {trace.get('response_mode') or 'unknown'} text=\"{response}\"")
        else:
            lines.append(f"[OUTPUT] {trace.get('response_mode') or 'unknown'}")
        return lines

    def _owner_policy_decision(self, command: str, *, source: str = "") -> PolicyDecision | None:
        if source not in {"ui", "typed_ui", "stt_voice", "voice"}:
            return None
        stream = self._get_stream_state()
        if stream is None:
            return None
        activity_decision = apply_owner_game_activity_correction(stream, command)
        if not activity_decision.allow_llm:
            self._record_policy_trace(policy_trace(
                source=source,
                speaker="Leo",
                text=command,
                decision=activity_decision,
                addressed_to_hebe=True,
                authority="owner",
            ))
            try:
                brain = self._get_live_session_brain()
                brain.sync_stream_metadata(stream)
                brain.apply_correction(command, self._normalize_text(command))
            except Exception as exc:
                print(f"[HEBE][LIVE_SESSION] owner policy correction failed: {exc!r}", flush=True)
            return activity_decision
        behavior_decision = owner_behavior_decision(stream, command)
        if not behavior_decision.allow_llm:
            self._record_policy_trace(policy_trace(
                source=source,
                speaker="Leo",
                text=command,
                decision=behavior_decision,
                addressed_to_hebe=True,
                authority="owner",
            ))
            return behavior_decision
        return None

    def _viewer_policy_decision(self, payload: dict) -> PolicyDecision | None:
        stream = self._get_stream_state()
        if stream is None:
            return None
        username = str((payload or {}).get("user_login") or (payload or {}).get("username") or "")
        display_name = str((payload or {}).get("display_name") or "")
        text = str((payload or {}).get("message_text") or (payload or {}).get("text") or "")
        decision = self._get_viewer_intent_policy().decide(
            stream,
            username=username,
            display_name=display_name,
            text=text,
        )
        self._record_policy_trace(policy_trace(
            source="twitch_chat",
            speaker=display_name or username or "viewer",
            text=text,
            decision=decision,
            addressed_to_hebe=True,
            authority="viewer",
        ))
        return decision

    def _live_session_debug_snapshot(self) -> dict:
        try:
            return self._get_live_session_brain().as_debug_dict()
        except Exception as exc:
            print(f"[HEBE][LIVE_SESSION] debug snapshot failed: {exc!r}", flush=True)
            return {}

    def _set_wake_loop_alive(self, alive: bool, *, error: str = "") -> None:
        self._wake_loop_alive = bool(alive)
        self._wake_loop_last_error = str(error or "")
        print(
            f"[HEBE][WAKE_LOOP] alive={str(bool(alive)).lower()}"
            + (f" error={self._wake_loop_last_error!r}" if self._wake_loop_last_error else ""),
            flush=True,
        )
        try:
            emit(
                "status",
                {
                    "wake_loop_alive": bool(alive),
                    "wake_loop_error": self._wake_loop_last_error,
                    "wake_loop_status": "alive" if alive else "crashed" if self._wake_loop_last_error else "stopped",
                },
            )
        except Exception:
            pass

    def wake_loop_health(self) -> dict:
        return {
            "alive": bool(getattr(self, "_wake_loop_alive", False)),
            "last_error": str(getattr(self, "_wake_loop_last_error", "") or ""),
            "thread_alive": bool(getattr(getattr(self, "_thread", None), "is_alive", lambda: False)()),
        }

    def _apply_stream_performance_profile(self) -> None:
        try:
            stream = self._get_stream_state()
            policies = getattr(stream, "policies", None) if stream else None
            if policies is None:
                return
            profile = os.getenv("HEBE_GAME_PERFORMANCE_PROFILE", "").strip().lower()
            game = str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or "").strip().lower()
            if not profile:
                if "baldur" in game or "bg3" in game:
                    profile = "bg3"
                elif "persona" in game:
                    profile = "light"
            if not profile:
                return
            print(f"[HEBE][PERFORMANCE_PROFILE] applying profile={profile}", flush=True)
            if profile in {"light", "persona", "persona5", "persona_5_royal"}:
                policies.allow_tts_replies = True
                policies.allow_tts_idle_prompts = os.getenv("HEBE_SPONTANEOUS_TTS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
                self._set_stream_spontaneity_cooldown_floor(5 * 60)
            elif profile in {"bg3", "heavy", "baldurs_gate_3"}:
                policies.allow_tts_replies = True
                policies.allow_tts_idle_prompts = os.getenv("HEBE_SPONTANEOUS_TTS_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
                self._set_stream_spontaneity_cooldown_floor(10 * 60)
        except Exception as exc:
            print(f"[HEBE][PERFORMANCE_PROFILE][ERROR] failed but continuing error={exc!r}", flush=True)

    def _set_stream_spontaneity_cooldown_floor(self, minimum_seconds: float) -> None:
        service = getattr(self, "stream_spontaneity", None)
        config = getattr(service, "config", None) if service is not None else None
        if config is None:
            return
        current = float(getattr(config, "global_stream_cooldown_sec", 0.0) or 0.0)
        effective = max(current, float(minimum_seconds or 0.0))
        try:
            service.config = replace(config, global_stream_cooldown_sec=effective)
            print(f"[HEBE][PERFORMANCE_PROFILE] applied global_stream_cooldown_sec={effective:g}", flush=True)
        except Exception as exc:
            self._stream_spontaneity_global_cooldown_override_sec = effective
            print(
                "[HEBE][PERFORMANCE_PROFILE][ERROR] failed but continuing "
                f"error={exc!r} override_global_stream_cooldown_sec={effective:g}",
                flush=True,
            )

    def _resolve_twitch_target_details(self, raw_target: str):
        twitch = getattr(self.runtime, "twitch", None)
        resolver = getattr(twitch, "resolve_user_details", None)
        if callable(resolver):
            try:
                return resolver(raw_target, intent="twitch_shoutout")
            except Exception as exc:
                print(f"[HEBE][TWITCH][TARGET] resolver failed target={raw_target!r} error={exc!r}", flush=True)
        resolve_user = getattr(twitch, "resolve_user", None)
        if callable(resolve_user):
            try:
                username = resolve_user(raw_target)
                if username:
                    return {"username": username, "confidence": 0.82, "candidates": [username], "reason": "legacy_resolver"}
            except Exception as exc:
                print(f"[HEBE][TWITCH][TARGET] legacy resolver failed target={raw_target!r} error={exc!r}", flush=True)
        return None

    def observe_twitch_chat_message(
        self,
        username: str,
        display_name: str,
        text: str,
        channel: str = "",
        *,
        firewall_decision: InputFirewallDecision | None = None,
    ) -> None:
        if not getattr(self, "stream_observe_chat", True):
            return
        stream = self._get_stream_state()
        if not stream:
            return

        message = str(text or "").strip()
        if not message:
            return

        firewall = firewall_decision or self._input_firewall_decision(
            source="twitch_viewer",
            text=message,
            username=username,
            event_type="twitch_chat_observe",
            addressed_to_hebe=self._message_mentions_hebe(message),
        )
        if not self._firewall_allows_pipeline(firewall):
            return

        print(f"[HEBE][TWITCH][CHAT] observed username={username} message={message!r}", flush=True)
        chat_event = InputEvent(
            source="twitch_chat",
            raw_text=message,
            normalized_text=self._normalize_text(message),
            username=username,
            is_stream_context=True,
        )
        chat_classification = self._get_input_classifier().classify(
            chat_event,
            addressed_to_hebe=self._message_mentions_hebe(message),
            valid=True,
        )
        self._log_input_classification(chat_classification)
        self._declare_output_route(
            input_type="twitch_chat_observe",
            targets=[OUTPUT_TARGET_SILENT_CONTEXT_UPDATE],
            reason="chat_context_update",
        )
        twitch = getattr(self.runtime, "twitch", None)
        remember = getattr(twitch, "remember_chat_message", None)
        if callable(remember):
            try:
                remember(username=username, display_name=display_name or username, text=message)
            except Exception as exc:
                print(f"[HEBE][TWITCH][CHAT] cache_update_failed username={username} error={exc!r}", flush=True)

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
        topic = self._classify_chat_topic(message)
        linked_context = self._linked_run_context_for_chat_topic(stream, topic)
        try:
            stream_memory.record_chat_message(
                username=username,
                display_name=display_name,
                message_text=message,
                stream_session_id=session_id,
                is_mention_to_hebe=self._message_mentions_hebe(message),
                is_direct_reply_to_hebe=False,
                is_bot=False,
                source="twitch_irc",
                topic_hint=topic,
            )
        except Exception as exc:
            print(f"[HEBE][STREAM_MEMORY] chat_message failed: {exc!r}", flush=True)
        entry = {
            "username": str(username or "").strip(),
            "display_name": str(display_name or username or "").strip(),
            "text": message[:180],
            "ts": now,
            "channel": channel,
            "topic": topic,
            "category": "chat_topic",
            "summary": self._summarize_chat_topic(username, display_name, message, topic),
            "linked_to_recent_run_context": linked_context,
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
        print(f"[HEBE][CHATTER] recent_chatter updated username={username} display={display_name or username}", flush=True)

        topics = [item.get("topic") for item in stream.recent_chat_messages if item.get("topic")]
        stream.recent_chat_topics = topics[-12:]
        if topics:
            stream.recent_chat_summary = ", ".join(dict.fromkeys(topics[-5:]))
        try:
            self._get_live_session_brain().observe_chat_message(
                username,
                display_name or username,
                message,
                topic=topic,
                mention=self._message_mentions_hebe(message),
            )
        except Exception as exc:
            print(f"[HEBE][LIVE_SESSION] chat observe failed: {exc!r}", flush=True)
        if topic != "general_chat":
            users = sorted({
                str(item.get("username") or "").strip()
                for item in stream.recent_chat_messages
                if item.get("topic") == topic and str(item.get("username") or "").strip()
            })
            print(
                f"[HEBE][CHAT_CONTEXT] topic={topic} users={users!r} "
                f"summary={entry['summary']!r}",
                flush=True,
            )

    def _is_chat_bot_user(self, username: str) -> bool:
        user = (username or "").strip().lower().lstrip("@")
        if not user:
            return True
        return is_known_bot_username(user, self._input_firewall_bot_usernames())

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
            "wizebot",
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
        stream_live = bool(stream and getattr(stream, "is_live", False))
        if not stream_live:
            reason = "offline_stream"
            if stream is not None:
                stream.last_shoutout_error = reason
            print(f"[HEBE][PROMOTION_GATE] blocked reason={reason} target={target}", flush=True)
            print(f"[HEBE][ACTION_PERMISSIONS] action={ACTION_PROMOTION_SHOUTOUT} allowed=false reason={reason}", flush=True)
            return False, normalized, reason
        if source not in {"manual", "raid"}:
            reason = "untrusted_source"
            if source in {"ambient_stt", "media_or_music"}:
                reason = "ambient_stt_not_allowed"
            if stream is not None:
                stream.last_shoutout_error = reason
            print(f"[HEBE][PROMOTION_GATE] blocked reason={reason} target={target}", flush=True)
            print(f"[HEBE][ACTION_PERMISSIONS] action={ACTION_PROMOTION_SHOUTOUT} allowed=false reason={reason}", flush=True)
            return False, normalized, reason
        print(
            "[HEBE][PROMOTION_GATE] "
            f"allowed reason={'approved_stream_event' if source == 'raid' else 'owner_direct_command'} target={normalized}",
            flush=True,
        )
        reason = self._shoutout_block_reason(normalized, explicit_self=explicit_self)
        if reason:
            if stream is not None:
                stream.last_shoutout_error = reason
            print(f"[HEBE][TWITCH][SO] blocked reason={reason} target={target}", flush=True)
            if reason in {"blocked_bot_user", "invalid_target"}:
                print(f"[HEBE][PROMOTION_GATE] blocked reason=untrusted_target target={target}", flush=True)
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
            self._declare_output_route(
                input_type="direct_stt" if source == "stt_voice" else source,
                targets=[OUTPUT_TARGET_TWITCH_COMMAND],
                reason="action_plan_twitch_shoutout",
            )
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
        if not (stream and getattr(stream, "is_live", False)):
            print(f"[HEBE][PROMOTION_GATE] blocked reason=offline_stream target={target}", flush=True)
            print(f"[HEBE][TWITCH][SO] blocked reason=stream_offline target={target}", flush=True)
            return
        print(f"[HEBE][TWITCH][SO] auto shoutout planned target={target}", flush=True)
        ok, normalized, reason = self._send_shoutout(target, source="raid", force=force)
        if not ok:
            print(f"[HEBE][TWITCH][SO] auto shoutout failed reason={reason} target={normalized or target}", flush=True)

    def _classify_chat_topic(self, text: str) -> str:
        normalized = self._normalize_text(text)
        if any(word in normalized for word in ("rng", "suerte", "azar", "random", "dados", "dado", "parchis")):
            return "rng_dependency"
        if any(word in normalized for word in ("linux", "ram", "servidor", "server", "pc", "windows", "obs")):
            return "tech_pc"
        if any(word in normalized for word in ("ff9", "final fantasy", "level 1", "boss", "jefe", "lindblum", "ramuh")):
            return "game"
        if any(word in normalized for word in ("hola", "buenas", "hello")):
            return "greeting"
        return "general_chat"

    def _linked_run_context_for_chat_topic(self, stream, topic: str) -> str | None:
        if topic == "general_chat":
            return None
        for fact in reversed(list(getattr(stream, "recent_run_context_facts", []) or [])):
            category = str(fact.get("category") or fact.get("kind") or "")
            if topic == category:
                return category
            if topic == "rng_dependency" and category in {"rng_dependency", "challenge_constraint"}:
                return category
        return None

    def _summarize_chat_topic(self, username: str, display_name: str, message: str, topic: str) -> str:
        name = str(display_name or username or "chat").strip()
        if topic == "rng_dependency":
            return f"{name} joked or commented about RNG, luck, or dice."
        if topic == "tech_pc":
            return f"{name} discussed PC/stream tech."
        if topic == "game":
            return f"{name} discussed the current game or run."
        if topic == "greeting":
            return f"{name} greeted the stream."
        return str(message or "").strip()[:140]

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
        if not stream or not getattr(stream, "is_live", False):
            return getattr(stream, "active_stream_session_id", None) if stream else None
        try:
            return stream_memory.ensure_active_stream_session(stream, source="engine")
        except Exception as exc:
            print(f"[HEBE][STREAM_MEMORY] ensure session failed: {exc!r}", flush=True)
            return None

    def _record_stream_event_safe(self, event_type: str, payload: dict | None = None, *, stream=None) -> None:
        try:
            stream_memory.record_stream_event(event_type, payload or {}, stream=stream)
        except Exception as exc:
            print(f"[HEBE][STREAM_MEMORY] record event failed event_type={event_type!r}: {exc!r}", flush=True)

    def _observe_stream_presence_safe(
        self,
        username: str,
        display_name: str,
        *,
        stream_session_id: int | None = None,
        source: str = "event",
    ) -> None:
        try:
            stream_memory.observe_presence(
                username,
                display_name,
                stream_session_id=stream_session_id,
                source=source,
            )
        except Exception as exc:
            print(f"[HEBE][STREAM_MEMORY] observe presence failed source={source!r}: {exc!r}", flush=True)

    def _close_stream_memory_session_safe(self, stream, *, reason: str) -> object | None:
        try:
            return stream_memory.close_active_stream_session(stream, reason=reason)
        except Exception as exc:
            print(f"[HEBE][STREAM_MEMORY] close session failed reason={reason!r}: {exc!r}", flush=True)
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
        print("[HEBE][LEGACY_FLOW] delegated_to=cognitive_flow", flush=True)
        return self.cognitive_flow(command, source=source)

    def _manual_handler_guard(
        self, *, handler: str, cognitive_decision, capabilities: set[str],
        source: str | None = None, require_live: bool = False,
    ) -> bool:
        decision = cognitive_decision or getattr(self, "_active_cognitive_decision", None)
        reason = "authorized"
        allowed = True
        if decision is None:
            allowed, reason = False, "missing_cognitive_decision"
        elif bool(getattr(decision, "should_stop_pipeline", False)):
            allowed, reason = False, "pipeline_stopped"
        elif str(getattr(decision, "authority", "")) != "owner":
            allowed, reason = False, "owner_authority_required"
        elif source and str(getattr(decision, "source", "")) not in {
            source, "ui" if source == "typed_ui" else source,
        }:
            allowed, reason = False, "source_mismatch"
        elif not any(decision.allows_capability(capability) for capability in capabilities):
            allowed, reason = False, "capability_not_authorized"
        elif not set(getattr(decision, "allowed_step_types", []) or []) & {"state_update", "action", "reply"}:
            allowed, reason = False, "step_type_not_authorized"
        elif require_live:
            permissions = getattr(decision, "action_permission_summary", {}) or {}
            if not bool(permissions.get("stream_live")) and not bool(permissions.get("is_simulation")):
                allowed, reason = False, "stream_offline"
        print(
            f"[HEBE][MANUAL_HANDLER_GUARD] {'allowed' if allowed else 'blocked'} "
            f"handler={handler} capabilities={sorted(capabilities)!r} reason={reason}",
            flush=True,
        )
        return allowed

    def _handle_wake_sleep_command(self, text: str, *, cognitive_decision=None, source: str | None = None) -> CommandResult | None:
        if not self._manual_handler_guard(
            handler="wake_sleep", cognitive_decision=cognitive_decision,
            capabilities={"hebe.wake_control"}, source=source,
        ):
            return None
        normalized = self._normalize_text(text)
        resolver = getattr(self, "wake_name_resolver", None) or WakeNameResolver()
        self.wake_name_resolver = resolver
        resolution = resolver.resolve(
            raw_text=text,
            normalized_text=normalized,
            source="command",
            is_sleeping=bool(getattr(self.runtime.state, "hebe_sleeping", False)),
            command_markers={"despierta", "levanta", "duerme", "descansa", "sleep", "wake", "modo"},
        )
        print(
            "[HEBE][WAKE_RESOLVER] "
            f"addressed_to_hebe={resolution.addressed_to_hebe} "
            f"wake_command={resolution.wake_command} "
            f"sleep_command={resolution.sleep_command} "
            f"matched_name={resolution.matched_name!r} "
            f"canonical={getattr(resolution, 'canonical', None)!r} "
            f"confidence={resolution.confidence:.3f} "
            f"reason={resolution.reason}",
            flush=True,
        )
        if resolution.addressed_to_hebe and resolution.matched_name and getattr(resolution, "canonical", None):
            print(
                "[HEBE][WAKE_RESOLVER] "
                f"matched_alias={resolution.matched_name!r} "
                f"canonical={resolution.canonical!r} "
                "addressed_to_hebe=true "
                f"reason={resolution.reason}",
                flush=True,
            )

        if resolution.wake_command or self._is_wake_concept_only(normalized):
            was_sleeping = bool(getattr(self.runtime.state, "hebe_sleeping", False))
            self.runtime.state.hebe_sleeping = False
            self.runtime.state.mode = "active"
            fallback = "Despierta y contigo, Leo." if was_sleeping else "Ya estaba despierta, mi señor."
            return CommandResult(
                action_type="wake_from_sleep" if was_sleeping else "already_awake",
                success=True,
                user_visible_summary=fallback,
                state_changes={"hebe_sleeping": False, "mode": "active"},
                constraints=["Do not claim the app restarted.", "Do not ask for clarification."],
                fallback_text=fallback,
                requires_model_response=True,
                metadata={"message_goal": fallback},
            )

        if resolution.sleep_command or self._is_sleep_concept_only(normalized):
            self.runtime.state.hebe_sleeping = True
            self.runtime.state.mode = "sleep"
            stream = self._get_stream_state()
            if stream is not None:
                stream.idle_spontaneity_enabled = False
            return CommandResult(
                action_type="sleep_mode",
                success=True,
                user_visible_summary="Hebe entered sleep mode.",
                state_changes={"hebe_sleeping": True, "mode": "sleep", "idle_spontaneity_enabled": False},
                constraints=["Do not ask for clarification."],
                fallback_text="Me quedo en espera, Leo.",
                requires_model_response=True,
                metadata={"message_goal": "Tell Leo Hebe is going into sleep mode."},
            )

        return None

    def _is_wake_concept_only(self, normalized: str) -> bool:
        tokens = set(str(normalized or "").split())
        return bool(tokens & {"despierta", "levanta", "wake", "awake"}) and len(tokens) <= 3

    def _is_sleep_concept_only(self, normalized: str) -> bool:
        tokens = set(str(normalized or "").split())
        return bool(tokens & {"duerme", "descansa", "dormir", "sleep", "espera"}) and len(tokens) <= 4
    
    def cognitive_flow(self, command: str, source: str = "voice") -> str:
        print(
            "[HEBE][COG] incoming "
            f"source={source!r} "
            f"command={command!r} "
            f"current_pending={getattr(self.runtime.state, 'pending_clarification', None)!r}",
            flush=True,
        )
        firewall = None
        if source in {"ui", "typed_ui"}:
            firewall = self._input_firewall_decision(
                source="owner_ui",
                text=command,
                addressed_to_hebe=True,
                has_action_intent=False,
            )

        stt_firewall_payload = {}
        current_event = getattr(self, "_current_input_event", None)
        current_metadata = getattr(current_event, "stt_metadata", None)
        if source == "stt_voice" and isinstance(current_metadata, dict):
            stt_firewall_payload = current_metadata.get("input_firewall") or {}
        route_source = str(stt_firewall_payload.get("source") or ("ui" if source in {"ui", "typed_ui"} else source))
        route_authority = str(stt_firewall_payload.get("authority") or (
            "owner" if route_source in {"ui", "stt_voice", "owner_stt_direct", "owner_stt_followup", "voice"} else "system"
        ))
        route_addressed = bool(
            route_source == "owner_stt_direct"
            or route_authority == "owner" and route_source in {"ui", "voice", "stt_voice"}
        )
        if not hasattr(self, "context_builder"):
            context = SimpleNamespace(
                input_text=command, internal_event=None,
                state_snapshot={"pending_clarification": getattr(self.runtime.state, "pending_clarification", None)},
                source=route_source, authority=route_authority,
                addressed_to_hebe=route_addressed,
            )
        else:
            try:
                context = self.context_builder.build(
                    state=self.runtime.state,
                    input_text=command,
                    internal_event=None,
                    source=route_source,
                    authority=route_authority,
                    addressed_to_hebe=route_addressed,
                )
            except TypeError:
                context = self.context_builder.build(
                    state=self.runtime.state, input_text=command, internal_event=None
                )
                context.source = route_source
                context.authority = route_authority
                context.addressed_to_hebe = route_addressed
        router = getattr(self, "cognitive_router", None) or CognitiveRouter()
        context.firewall_decision = str(
            stt_firewall_payload.get("firewall_decision")
            or getattr(firewall, "firewall_decision", "")
            or ""
        )
        stream = self._get_stream_state()
        context.stream_is_live = bool(
            getattr(stream, "is_live", False)
            or (getattr(stream, "enabled", False) and not getattr(stream, "live_status_known", False))
        )
        hints = []
        normalized_route = self._normalize_text(command)
        if hasattr(self, "_parse_tts_control_intent") and self._parse_tts_control_intent(normalized_route) is not None:
            hints.append("tts_control")
        if (
            getattr(self.runtime.state, "pending_tts_scope", None)
            and self._parse_tts_scope_followup(normalized_route) is not None
        ):
            hints.append("pending_tts_reply")
        route_tokens = set(normalized_route.split())
        stream_domain = route_tokens & {"stream", "directo", "twitch", "shoutout", "promo", "raid", "chat", "stt", "ambiental"}
        stream_control = route_tokens & {
            "activa", "activar", "desactiva", "desactivar", "abre", "cierra", "inicia", "para",
            "haz", "hazle", "manda", "envia", "dile", "di", "pausa", "reanuda", "cambia", "pon", "quita",
            "enable", "disable", "start", "stop", "send", "pause", "resume",
        }
        if stream_domain and stream_control:
            hints.append("stream_manual")
        if route_tokens & {"chat"} and stream_control:
            hints.append("stream_action")
        if route_tokens & {"shoutout", "promo", "raid"} and stream_control:
            hints.append("stream_action")
            hints.append("stream_manual")
        if re.search(r"\b(?:que|cual)\s+(?:toca|juego|directo|stream)\b", normalized_route):
            hints.append("stream_manual")
        context.route_hints = hints
        context.cognitive_decision = router.route(context)
        if bool(getattr(self, "_manual_simulation_mode", False)):
            context.cognitive_decision.action_permission_summary["is_simulation"] = True
        self._last_cognitive_trace = context.cognitive_decision.to_dict()
        decision = context.cognitive_decision

        if decision.intent in {"wake_control", "sleep_control"} and decision.allows_capability("hebe.wake_control"):
            wake_result = self._handle_wake_sleep_command(command, cognitive_decision=decision, source=route_source)
            if wake_result is not None:
                text = self._synthesize_command_result(wake_result, input_text=command)
                self._deliver_manual_reply(text, source=source)
                print("[HEBE][COG] decision=authorized_wake_control", flush=True)
                return "continue"

        if bool(getattr(self.runtime.state, "hebe_sleeping", False)):
            print("[HEBE][WAKE] sleeping; ignored input reason=router_did_not_authorize_wake", flush=True)
            return "continue"

        # Legacy harnesses do not construct the deliberation stack. Their
        # command planner remains a compatibility endpoint, but only after the
        # central route has classified the input.
        if not hasattr(self, "context_builder"):
            local_app = self._plan_and_execute_local_app_action(command, source) if decision.allows_capability("pc.open_application") else None
            if local_app is not None:
                text = self._synthesize_command_result(local_app, input_text=command)
                self._deliver_manual_reply(text, source=source)
                return "continue"

        owner_decision = self._owner_policy_decision(command, source=source)
        if owner_decision is not None and not owner_decision.allow_llm:
            if owner_decision.allow_reply:
                policy_reply_result = self._synthesize_policy_reply(
                    owner_decision,
                    input_text=command,
                    source=source,
                    speaker="Leo",
                )
                policy_reply = str(policy_reply_result.get("text") or "")
                if policy_reply:
                    self._update_policy_trace_response(
                        policy_reply,
                        response_source=str(policy_reply_result.get("response_source") or "hybrid"),
                        style_guard_triggered=bool(policy_reply_result.get("style_guard_triggered")),
                        was_generic_refusal_rewritten=bool(policy_reply_result.get("was_generic_refusal_rewritten")),
                    )
                    self._deliver_manual_reply(policy_reply, source=source)
            print(f"[HEBE][COG] decision=owner_policy reason={owner_decision.reason}", flush=True)
            return "continue"

        manual = self._handle_pending_manual_intent(
            command, cognitive_decision=decision, source=route_source,
        ) if (decision.allows_capability("pending.cancel") or decision.allows_capability("audio.tts_control")) else None
        if manual is None and decision.allows_capability("audio.tts_control"):
            manual = self._handle_tts_manual_command(command, cognitive_decision=decision, source=route_source)
        if manual is None and (
            decision.allows_capability("stream.local_state_control")
            or decision.allows_capability("twitch_action")
        ):
            manual = self._handle_stream_manual_command(command, cognitive_decision=decision, source=route_source)
        if manual is not None:
            force_ui = bool(getattr(self, "_manual_reply_ui_only", False))
            self._manual_reply_ui_only = False
            if isinstance(manual, CommandResult):
                manual_text = self._synthesize_command_result(manual, input_text=command)
            else:
                manual_text = str(manual)
            self._deliver_manual_reply(manual_text, source="ui" if force_ui else source)
            return "continue"

        if source == "stt_voice":
            event = getattr(self, "_current_input_event", None)
            metadata = getattr(event, "stt_metadata", None)
            if not isinstance(metadata, dict) or not metadata.get("jarvis_allowed"):
                print("[HEBE][JARVIS][BLOCKED] reason=stt_not_direct", flush=True)
                return "continue"

        cognitive_followup = False
        if source in {"ui", "stt_voice"} and self._pending_conversation_matches(source=source, text=command):
            self._consume_pending_conversation_turn()
            cognitive_followup = True
            print("[HEBE][COG] decision=conversation_followup", flush=True)

        input_event = getattr(self, "_current_input_event", None)
        if input_event is None:
            input_event = self._build_input_event(
                source="ui" if source == "ui" else source,
                raw_text=command,
                normalized_text=self._normalize_text(command),
            )
            self._current_input_event = input_event
        metadata = getattr(input_event, "stt_metadata", None)
        frame_payload = (metadata or {}).get("response_frame") if isinstance(metadata, dict) else None
        if isinstance(frame_payload, dict):
            response_frame_payload = frame_payload
        else:
            classification = self._get_input_classifier().classify(
                input_event,
                addressed_to_hebe=source in {"ui", "typed_ui"},
                has_action_intent=False,
                pending_followup=cognitive_followup,
                valid=True,
            )
            self._log_input_classification(classification)
            conversation_state = self._get_conversation_state_resolver().from_pending_turn(
                self._get_pending_conversation_turn(),
                matched=cognitive_followup,
                reason="cognitive_flow_followup" if cognitive_followup else "direct_or_typed_input",
            )
            self._log_conversation_state(conversation_state)
            response_decision = self._get_response_decision_resolver().decide(
                classification=classification,
                conversation_state=conversation_state,
                relevance=ContextRelevance(useful=False, category="none", reason="not_ambient"),
                output_targets=self._output_targets_for_input_type(classification.input_type),
            )
            self._log_knowledge_resolution()
            self._log_response_decision(response_decision)
            response_frame_payload = self._build_response_frame(
                event=input_event,
                classification=classification,
                conversation_state=conversation_state,
                response_decision=response_decision,
            ).as_dict()

        context.response_frame = response_frame_payload

        print(
            "[HEBE][COG] context pending="
            f"{context.state_snapshot.get('pending_clarification')!r}",
            flush=True,
        )

        deliberation = self.deliberation_service.deliberate(context)
        self._last_cognitive_trace.update({
            "selected_route": context.cognitive_decision.intent,
            "final_plan_steps": [step.type for step in deliberation.plan.steps],
        })
        execution = self.plan_executor.execute(deliberation.plan)
        self._last_cognitive_trace["plan_executor_guard"] = list(getattr(self.plan_executor, "last_guard_results", []) or [])
        reply_text = self.response_synthesizer.synthesize(
            context=context,
            deliberation=deliberation,
            execution=execution,
        )
        self._last_cognitive_trace["final_response"] = reply_text

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
                now_ts = time.time()
                self.runtime.state.pending_clarification = {
                    "id": f"pending_{uuid.uuid4().hex}",
                    "kind": "appointment_datetime",
                    "draft": reply_step.data.get("draft", {}),
                    "authority": "owner",
                    "created_at": now_ts,
                    "expires_at": now_ts + float(os.getenv("HEBE_PENDING_TASK_TTL_SECONDS", "900") or 900),
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
            self._declare_output_route(
                input_type="ui_typed_input",
                targets=[OUTPUT_TARGET_LOCAL_UI],
                reason="typed_input_reply",
            )
            log_chat("assistant", reply_text, source="ui")
            emit("chat.assistant", {"text": reply_text, "source": "ui", "output_target": OUTPUT_TARGET_LOCAL_UI})
            self._record_assistant_reply_for_conversation(reply_text, source=source, synthesizer=getattr(self, "response_synthesizer", None))

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
                    self._record_assistant_reply_for_conversation(reply_text, source=source, synthesizer=getattr(self, "response_synthesizer", None))
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
        event_type = str(getattr(event, "event_type", "") or "")
        payload = getattr(event, "payload", {}) or {}
        event_decision = None
        if event_type.startswith("twitch_"):
            raw_text = str((payload or {}).get("message_text") or (payload or {}).get("text") or "")
            username = str((payload or {}).get("user_login") or (payload or {}).get("username") or "")
            source = "twitch_viewer" if event_type == "twitch_chat_react" else "twitch_system"
            firewall = self._input_firewall_decision(
                source=source,
                text=raw_text,
                username=username,
                event_type=event_type,
                addressed_to_hebe=self._message_mentions_hebe(raw_text) if raw_text else False,
            )
            if isinstance(payload, dict):
                payload["input_firewall"] = firewall.as_dict()
                event.payload = payload
            if not self._firewall_allows_pipeline(firewall):
                self._record_policy_trace(policy_trace(
                    source=firewall.source,
                    speaker=str((payload or {}).get("display_name") or username or firewall.source),
                    text=raw_text,
                    decision=PolicyDecision(
                        allow_reply=False,
                        allow_llm=False,
                        allow_free_llm=False,
                        reason=firewall.reason,
                        intent="input_firewall",
                        requested_behavior="blocked_input",
                    ),
                    addressed_to_hebe=False,
                    authority=firewall.authority,
                ))
                return
            stream = self._get_stream_state()
            route_context = SimpleNamespace(
                input_text=raw_text,
                internal_event=event,
                state_snapshot={"pending_clarification": None},
                source=source,
                authority=firewall.authority,
                addressed_to_hebe=event_type == "twitch_chat_react" or self._message_mentions_hebe(raw_text),
                firewall_decision=firewall.firewall_decision,
                stream_is_live=bool(
                    getattr(stream, "is_live", False)
                    or (getattr(stream, "enabled", False) and not getattr(stream, "live_status_known", False))
                ),
                route_hints=[],
            )
            event_decision = (getattr(self, "cognitive_router", None) or CognitiveRouter()).route(route_context)
            self._last_cognitive_trace = event_decision.to_dict()
            if event_decision.should_stop_pipeline:
                print(f"[HEBE][EVENT_ROUTER] blocked type={event_type} reason={event_decision.reason}", flush=True)
                return
        if event_type == "twitch_raid":
            self._handle_twitch_raid_event(event, cognitive_decision=event_decision)
            return

        if event_type == "twitch_chat_react":
            policy_decision = self._viewer_policy_decision(payload)
            if policy_decision is not None and not policy_decision.allow_llm:
                if policy_decision.allow_reply:
                    speaker = str((payload or {}).get("display_name") or (payload or {}).get("user_login") or "viewer")
                    raw_text = str((payload or {}).get("message_text") or (payload or {}).get("text") or "")
                    policy_reply_result = self._synthesize_policy_reply(
                        policy_decision,
                        input_text=raw_text,
                        source="twitch_chat",
                        speaker=speaker,
                    )
                    policy_reply = str(policy_reply_result.get("text") or "")
                    if policy_reply:
                        self._update_policy_trace_response(
                            policy_reply,
                            response_source=str(policy_reply_result.get("response_source") or "hybrid"),
                            style_guard_triggered=bool(policy_reply_result.get("style_guard_triggered")),
                            was_generic_refusal_rewritten=bool(policy_reply_result.get("was_generic_refusal_rewritten")),
                        )
                        self._deliver_twitch_reply(
                            policy_reply,
                            event_type=event_type,
                            payload=payload,
                        )
                    else:
                        print(
                            f"[HEBE][VIEWER_POLICY] decision=ignored reason={policy_decision.reason}",
                            flush=True,
                        )
                else:
                    print(
                        f"[HEBE][VIEWER_POLICY] decision=ignored reason={policy_decision.reason}",
                        flush=True,
                    )
                return

        context = self.context_builder.build(
            state=self.runtime.state,
            input_text=None,
            internal_event=event,
        )
        stream = self._get_stream_state()
        context.stream_is_live = bool(
            getattr(stream, "is_live", False)
            or (getattr(stream, "enabled", False) and not getattr(stream, "live_status_known", False))
        )
        context.cognitive_decision = event_decision
        try:
            live_context = self._get_live_session_brain().retrieve_context(str(payload), limit_events=12, limit_summaries=3)
            if isinstance(payload, dict):
                payload.setdefault("live_session_context", live_context)
                event.payload = payload
        except Exception as exc:
            print(f"[HEBE][LIVE_SESSION] internal event context failed: {exc!r}", flush=True)
        if event_type == "twitch_chat_react":
            raw_text = str((payload or {}).get("message_text") or "")
            event_source = "twitch_chat"
            addressed = True
        elif event_type == "twitch_idle_prompt":
            raw_text = ""
            event_source = "scheduler/spontaneity"
            addressed = False
        elif event_type.startswith("twitch_"):
            raw_text = ""
            event_source = "twitch_event"
            addressed = False
        else:
            raw_text = ""
            event_source = "system/tool_result"
            addressed = False
        input_event = InputEvent(
            source=event_source,
            raw_text=raw_text,
            normalized_text=self._normalize_text(raw_text),
            is_stream_context=event_type.startswith("twitch_"),
        )
        classification = self._get_input_classifier().classify(
            input_event,
            addressed_to_hebe=addressed,
            valid=True,
        )
        self._log_input_classification(classification)
        conversation_state = self._get_conversation_state_resolver().from_pending_turn(
            None,
            matched=False,
            reason="stream_event_no_private_pending",
        )
        self._log_conversation_state(conversation_state)
        response_decision = self._get_response_decision_resolver().decide(
            classification=classification,
            conversation_state=conversation_state,
            relevance=ContextRelevance(useful=False, category="none", reason="event"),
            output_targets=self._output_targets_for_input_type(classification.input_type, event_type=event_type),
        )
        self._log_knowledge_resolution()
        self._log_response_decision(response_decision)
        context.response_frame = self._build_response_frame(
            event=input_event,
            classification=classification,
            conversation_state=conversation_state,
            response_decision=response_decision,
        ).as_dict()

        deliberation = self.deliberation_service.deliberate(context)
        execution = self.plan_executor.execute(deliberation.plan)
        self._last_cognitive_trace["plan_executor_guard"] = list(getattr(self.plan_executor, "last_guard_results", []) or [])
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
            print(
                "[HEBE][SPONTANEITY] "
                f"candidate=true anchor={(getattr(event, 'payload', {}) or {}).get('used_fact_id') or (getattr(event, 'payload', {}) or {}).get('idle_topic') or (getattr(event, 'payload', {}) or {}).get('specific_context_anchors')}",
                flush=True,
            )
            if service is not None:
                if service.is_too_similar_to_recent(stream, reply_text):
                    print("[HEBE][SPONTANEITY] skipped reason=too_similar_to_recent", flush=True)
                    return
                motif = service.motif_on_cooldown(stream, reply_text)
                if motif:
                    print(f"[HEBE][SPONTANEITY] skipped reason=motif_cooldown motif={motif}", flush=True)
                    return

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
            self._record_stream_event_safe("stream_online", payload, stream=stream)
            try:
                self._get_live_session_brain().observe_stream_metadata(stream, source="stream_online")
            except Exception as exc:
                print(f"[HEBE][LIVE_SESSION] stream_online failed: {exc!r}", flush=True)
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
            self._record_stream_event_safe("stream_offline", payload, stream=stream)
            try:
                brain = self._get_live_session_brain()
                brain.observe_stream_metadata(stream, source="stream_offline")
                brain.retrieve_context("stream ended", limit_events=20, limit_summaries=5)
            except Exception as exc:
                print(f"[HEBE][LIVE_SESSION] stream_offline failed: {exc!r}", flush=True)
            self._close_stream_memory_session_safe(stream, reason="stream_offline_event")
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

    def _handle_twitch_raid_event(self, event, cognitive_decision=None) -> None:
        stream = self._get_stream_state()
        payload = getattr(event, "payload", {}) or {}
        username = payload.get("display_name") or payload.get("user_login") or "alguien"
        viewers = int(payload.get("viewer_count") or 0)
        firewall = self._input_firewall_decision(
            source="twitch_system",
            text="",
            username=payload.get("user_login") or username,
            event_type="twitch_raid",
            addressed_to_hebe=False,
        )
        if isinstance(payload, dict):
            payload["input_firewall"] = firewall.as_dict()
            event.payload = payload
        if not self._firewall_allows_pipeline(firewall):
            print(f"[HEBE][TWITCH][RAID] blocked reason={firewall.reason}", flush=True)
            return
        if cognitive_decision is None:
            cognitive_decision = (getattr(self, "cognitive_router", None) or CognitiveRouter()).route(SimpleNamespace(
                input_text="", internal_event=event, state_snapshot={}, source="twitch_system",
                authority=firewall.authority, addressed_to_hebe=False,
                firewall_decision=firewall.firewall_decision,
                stream_is_live=bool(
                    getattr(stream, "is_live", False)
                    or (getattr(stream, "enabled", False) and not getattr(stream, "live_status_known", False))
                ),
                route_hints=[],
            ))
        print(f"[HEBE][TWITCH][RAID] received from={username} viewers={viewers}", flush=True)
        is_simulated = bool(payload.get("_simulated"))
        if stream is not None:
            stream.last_raid_event = {
                "display_name": username,
                "user_login": payload.get("user_login") or username,
                "viewer_count": viewers,
                "ts": time.time(),
            }
            if is_simulated:
                print("[HEBE][STREAM_SESSION] skipped reason=simulation", flush=True)
            else:
                self._ensure_stream_memory_session_if_live(stream)
                self._record_stream_event_safe("twitch_raid", payload, stream=stream)
                self._observe_stream_presence_safe(
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
        if cognitive_decision is None or not cognitive_decision.allows_capability("twitch.reply"):
            print("[HEBE][TWITCH][RAID] blocked reason=cognitive_reply_not_authorized", flush=True)
            return
        reply_text = self._synthesize_internal_event_reply(event, cognitive_decision=cognitive_decision)
        if not reply_text:
            print("[HEBE][TWITCH][RAID] blocked reason=empty_reply", flush=True)
            return
        self._deliver_twitch_reply(reply_text, event_type="twitch_raid", payload=payload)
        print("[HEBE][TWITCH][RAID] sent thank-you", flush=True)
        if is_simulated:
            print("[HEBE][PROMOTION_GATE] blocked reason=simulation_mode target={}".format(payload.get("user_login") or username), flush=True)
            return
        if not cognitive_decision.allows_capability("twitch.promotion"):
            print("[HEBE][PROMOTION_GATE] blocked reason=cognitive_promotion_not_authorized", flush=True)
            return
        self._maybe_auto_shoutout_raider(payload.get("user_login") or username, force=bool(payload.get("_force_shoutout")))

    def _synthesize_internal_event_reply(self, event, cognitive_decision=None) -> str:
        context = self.context_builder.build(
            state=self.runtime.state,
            input_text=None,
            internal_event=event,
        )
        stream = self._get_stream_state()
        context.stream_is_live = bool(
            getattr(stream, "is_live", False)
            or (getattr(stream, "enabled", False) and not getattr(stream, "live_status_known", False))
        )
        context.cognitive_decision = cognitive_decision

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
            if not is_live:
                self._restore_user_today_game_override(stream)
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
                self._close_stream_memory_session_safe(stream, reason="context_sync_offline")
            self._maybe_research_game_after_context_sync(stream)
            self._apply_stream_performance_profile()
            try:
                self._get_live_session_brain().observe_stream_metadata(stream, source="context_sync")
            except Exception as exc:
                print(f"[HEBE][LIVE_SESSION] context sync observe failed: {exc!r}", flush=True)
        print(f"[HEBE][STREAM_CONTEXT] refresh result success={ok}", flush=True)
        return ok

    def _mark_today_game_override(self, stream, game: str) -> None:
        stream.user_today_game_override = str(game or "").strip()
        stream.user_today_game_override_ts = time.time()
        stream.stream_context_override_reason = "user_today_game_override"
        stream.stream_context_overridden = True
        print(
            "[HEBE][STREAM_CONTEXT] stale_or_overridden=true reason=user_today_game_override "
            f"game={stream.user_today_game_override!r}",
            flush=True,
        )
    def _restore_user_today_game_override(self, stream) -> None:
        game = str(getattr(stream, "user_today_game_override", "") or "").strip()
        if not game:
            return
        changed = (getattr(stream, "current_game", None) != game) or (getattr(stream, "current_category", None) != game)
        stream.current_game = game
        stream.current_category = game
        if changed:
            print(
                "[HEBE][STREAM_CONTEXT] stale_or_overridden=true reason=user_today_game_override "
                f"restored_game={game!r}",
                flush=True,
            )

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
                    init_live_session_schema()
                    cleanup_stt_prompt_injection_rows()
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
                        "stream_output_mode": str(getattr(stream, "stream_output_mode", "tts_enabled") if stream else "tts_enabled"),
                        "stt_enabled": bool(getattr(self.runtime, "stt_enabled", False)),
                        "wake_loop_alive": False,
                        "wake_loop_status": "starting",
                    },
                )

                target = self.wakeword_loop if self.use_wakeword else self.engine_loop
                kwargs = {"say_hello": self.say_hello}

                def run_loop():
                    if self.use_wakeword:
                        self._set_wake_loop_alive(True)
                    try:
                        return target(**kwargs)
                    except Exception as exc:
                        if self.use_wakeword:
                            self._set_wake_loop_alive(False, error=str(exc))
                        print(f"[HEBE][WAKE_LOOP][ERROR] crashed error={exc!r}", flush=True)
                        return "error"
                    finally:
                        if self.use_wakeword and self._stop_event.is_set():
                            self._set_wake_loop_alive(False)

                self._thread = threading.Thread(
                    target=run_loop,
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
        normalized = " ".join(cleaned.split())
        print(f"[HEBE][NORMALIZE] raw={text!r} normalized={normalized!r}", flush=True)
        return normalized

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

    def _normalize_stt_input(self, raw_text: str, *, debug_metadata: dict | None = None) -> TranscriptNormalizationResult:
        result = normalize_stt_transcript(raw_text, known_targets=self._known_voice_command_targets())
        self._record_stt_normalization(result, debug_metadata=debug_metadata)
        return result

    def _record_stt_normalization(self, result: TranscriptNormalizationResult, *, debug_metadata: dict | None = None) -> None:
        print(f"[HEBE][STT][RAW] text={result.raw_text!r}", flush=True)
        print(
            "[HEBE][STT][NORMALIZED] "
            f"raw={result.raw_text!r} "
            f"normalized={result.normalized_text!r} "
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
        event = result.as_event()
        if debug_metadata:
            event.update(debug_metadata)
        emit("voice.command", event)

    def _unsupported_stt_script(self, text: str) -> str | None:
        allowed = {
            part.strip().lower()
            for part in os.getenv("HEBE_STT_ALLOWED_LANGUAGES", "es,en").split(",")
            if part.strip()
        }
        if not allowed.issubset({"es", "en"}):
            return None

        value = str(text or "")
        if "à¤" in value or "à¥" in value:
            return "devanagari"
        checks = [
            ("japanese", r"[\u3040-\u30ff]"),
            ("chinese", r"[\u3400-\u9fff]"),
            ("cyrillic", r"[\u0400-\u04ff]"),
            ("greek", r"[\u0370-\u03ff]"),
            ("tamil", r"[\u0b80-\u0bff]"),
            ("devanagari", r"[\u0900-\u097f]"),
            ("sinhala", r"[\u0d80-\u0dff]"),
            ("korean", r"[\uac00-\ud7af]"),
            ("thai", r"[\u0e00-\u0e7f]"),
            ("arabic", r"[\u0600-\u06ff]"),
            ("hebrew", r"[\u0590-\u05ff]"),
        ]
        for name, pattern in checks:
            if re.search(pattern, value):
                return name
        return None

    def _emit_stt_rejection(
        self,
        raw_text: str,
        *,
        script: str,
        reason: str,
        retry_attempted: bool = False,
        retry_transcript: str = "",
        details: dict | None = None,
    ) -> None:
        safe_raw = "" if reason == "stt_prompt_injection" else str(raw_text or "")
        log_raw = bool(getattr(self, "stt_log_rejected_raw", False))
        log_suffix = f" raw={ascii(safe_raw)}" if log_raw and safe_raw else ""
        detail_text = ""
        if details:
            detail_text = " " + " ".join(f"{key}={value}" for key, value in details.items())
        print(f"[HEBE][STT][REJECTED] reason={reason} script={script}{log_suffix}{detail_text}", flush=True)
        if reason == "stt_prompt_echo_or_hotword_list":
            self._record_stt_prompt_echo_rejection()
        stream = self._get_stream_state()
        if stream is not None:
            stream.last_voice_raw_transcript = safe_raw
            stream.last_voice_normalized_command = ""
            stream.last_voice_command_status = "rejected"
        event = {
            "raw_text": safe_raw,
            "normalized_text": "",
            "status": "rejected",
            "reason": reason,
            "script": script,
            "detected_script": script,
            "retry_attempted": bool(retry_attempted),
            "retry_transcript": str(retry_transcript or ""),
            "final_decision": "rejected",
            "message": (
                "Raw STT rejected: unsupported language/script."
                if reason.startswith("unsupported_script")
                else "Raw STT rejected before cognition."
            ),
        }
        if details:
            event.update(details)
        emit("voice.command", event)
        emit("status", {"last_rejected_stt": event})

    def _record_stt_prompt_echo_rejection(self) -> None:
        if not bool(getattr(self, "stt_auto_disable_prompt_on_echo", True)):
            return
        now = time.time()
        window = max(1.0, float(getattr(self, "stt_prompt_echo_window_seconds", 300) or 300))
        recent = [
            ts for ts in list(getattr(self, "_stt_prompt_echo_rejection_ts", []) or [])
            if now - float(ts or 0.0) <= window
        ]
        recent.append(now)
        self._stt_prompt_echo_rejection_ts = recent
        threshold = max(1, int(getattr(self, "stt_prompt_echo_disable_threshold", 2) or 2))
        if len(recent) < threshold:
            return
        stt = getattr(getattr(self, "runtime", None), "stt", None)
        disabled = False
        service_logs_disable = bool(stt is not None and hasattr(stt, "cfg"))
        disable = getattr(stt, "disable_command_prompt_for_session", None)
        if callable(disable):
            disabled = bool(disable())
        elif stt is not None and hasattr(stt, "cfg"):
            cfg = getattr(stt, "cfg")
            if getattr(cfg, "command_prompt_enabled", True):
                cfg.command_prompt_enabled = False
                disabled = True
        if disabled and not service_logs_disable:
            print("[HEBE][STT][PROMPT] auto_disabled reason=repeated_prompt_echo", flush=True)

    def _stt_prompt_echo_metrics(self, text: str) -> dict:
        normalized = self._normalize_stt_metric_text(text)
        tokens = re.findall(r"[a-z0-9]+", normalized)
        hotwords = {
            "hebe", "ebe", "eve", "heve", "jebe", "leo", "obs", "twitch", "stream",
            "chat", "promo", "shoutout", "so", "zwei", "persona", "final", "fantasy",
        }
        known_target_tokens = set()
        try:
            for target in self._known_voice_command_targets():
                target_norm = re.sub(r"[^a-z0-9]+", " ", str(target or "").lower())
                known_target_tokens.update(re.findall(r"[a-z0-9]+", target_norm))
        except Exception:
            known_target_tokens = set()
        list_tokens = hotwords | known_target_tokens
        action_tokens = {
            "abre", "abrir", "inicia", "pon", "haz", "dale", "manda", "activa",
            "desactiva", "apaga", "enciende", "guarda", "dime", "cuentame",
            "explica", "como", "cual", "que", "donde", "cuando", "por", "quiero",
            "puedes", "ayuda", "necesito", "despierta", "duerme",
        }
        hotword_hits = [token for token in tokens if token in list_tokens]
        content_hits = [token for token in tokens if token not in list_tokens]
        comma_parts = [
            self._normalize_stt_metric_text(part)
            for part in re.split(r"[,;\n]+", str(text or ""))
            if self._normalize_stt_metric_text(part)
        ]
        comma_hotword_parts = [
            part for part in comma_parts
            if all(token in list_tokens for token in re.findall(r"[a-z0-9]+", part))
        ]
        hotword_ratio = len(hotword_hits) / max(1, len(tokens))
        comma_hotword_ratio = len(comma_hotword_parts) / max(1, len(comma_parts))
        has_action_or_question = bool(set(tokens) & action_tokens) or "?" in str(text or "")
        repeated_vocab = len(tokens) >= 3 and len(set(tokens)) <= max(2, len(tokens) - 2)
        looks_like_list = len(comma_parts) >= 3 and comma_hotword_ratio >= 0.75
        rejected = (
            bool(tokens)
            and len(hotword_hits) >= 3
            and not has_action_or_question
            and (
                looks_like_list
                or hotword_ratio >= 0.75
                or (repeated_vocab and hotword_ratio >= 0.6)
                or (len(content_hits) <= 1 and len(hotword_hits) >= 4)
            )
        )
        return {
            "rejected": rejected,
            "overlap": round(hotword_ratio, 3),
            "hotword_ratio": round(hotword_ratio, 3),
            "hotword_count": len(hotword_hits),
            "token_count": len(tokens),
            "comma_hotword_ratio": round(comma_hotword_ratio, 3),
        }

    def _normalize_stt_metric_text(self, text: str) -> str:
        value = str(text or "").lower()
        value = (
            value.replace("Ã¡", "a").replace("Ã©", "e").replace("Ã­", "i")
            .replace("Ã³", "o").replace("Ãº", "u").replace("Ã±", "n")
        )
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return " ".join(value.split())

    def _reject_stt_prompt_echo_if_needed(self, raw_text: str, *, retry_attempted: bool = False, retry_transcript: str = "") -> bool:
        metrics = self._stt_prompt_echo_metrics(raw_text)
        if not metrics.get("rejected"):
            return False
        self._emit_stt_rejection(
            raw_text,
            script=self._unsupported_stt_script(raw_text) or "latin",
            reason="stt_prompt_echo_or_hotword_list",
            retry_attempted=retry_attempted,
            retry_transcript=retry_transcript,
            details={key: value for key, value in metrics.items() if key != "rejected"},
        )
        return True

    def _normalize_for_echo_match(self, text: str) -> str:
        raw = (text or "").strip().lower()
        without_accents = "".join(
            char for char in unicodedata.normalize("NFKD", raw)
            if not unicodedata.combining(char)
        )
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in without_accents)
        return " ".join(cleaned.split())

    def _tts_echo_duration_estimate(self, text: str) -> float:
        normalized = self._normalize_for_echo_match(text)
        words = len(normalized.split())
        chars = len(normalized)
        return max(2.5, min(30.0, max(words * 0.42, chars / 13.0)))

    def _echo_similarity(self, left: str, right: str) -> float:
        a = self._normalize_for_echo_match(left)
        b = self._normalize_for_echo_match(right)
        if not a or not b:
            return 0.0
        direct = SequenceMatcher(None, a, b).ratio()
        sorted_a = " ".join(sorted(a.split()))
        sorted_b = " ".join(sorted(b.split()))
        sorted_ratio = SequenceMatcher(None, sorted_a, sorted_b).ratio()
        a_tokens = set(a.split())
        b_tokens = set(b.split())
        overlap = 0.0
        if min(len(a_tokens), len(b_tokens)) >= 4:
            overlap = len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))
        return max(direct, sorted_ratio, overlap)

    def _remember_tts_text(self, text: str, message_id: str | None = None) -> None:
        value = str(text or "").strip()
        if not value:
            return
        recent = list(getattr(self, "_recent_tts_texts", []) or [])
        now = time.time()
        window = float(getattr(self, "stt_tts_echo_window_seconds", 10) or 10)
        grace = float(getattr(self, "stt_tts_echo_grace_seconds", 2.5) or 2.5)
        duration = self._tts_echo_duration_estimate(value)
        until = now + duration + grace
        normalized = self._normalize_for_echo_match(value)
        tts_message_id = str(message_id or "").strip()
        if not tts_message_id:
            digest = hashlib.sha1(f"{normalized}:{int(now * 1000)}".encode("utf-8")).hexdigest()[:16]
            tts_message_id = f"tts_{digest}"
        self._last_tts_text = value
        self._last_tts_normalized = normalized
        self._last_tts_message_id = tts_message_id
        self._tts_started_at = now
        self._tts_until = until
        self._tts_active = True
        recent = [item for item in recent if now - float(item.get("ts", 0.0) or 0.0) <= window]
        recent.append({
            "text": value,
            "normalized": normalized,
            "message_id": tts_message_id,
            "ts": now,
            "until": until,
        })
        self._recent_tts_texts = recent[-8:]

    def _stt_self_tts_echo_metrics(self, text: str) -> dict:
        now = time.time()
        window = float(getattr(self, "stt_tts_echo_window_seconds", 10) or 10)
        threshold = float(getattr(self, "stt_tts_echo_similarity_threshold", 0.82) or 0.82)
        normalized = self._normalize_for_echo_match(text)
        best = 0.0
        best_text = ""
        best_message_id = ""
        for item in list(getattr(self, "_recent_tts_texts", []) or []):
            if now - float(item.get("ts", 0.0) or 0.0) > window:
                continue
            candidate = str(item.get("normalized") or self._normalize_for_echo_match(item.get("text") or ""))
            if not normalized or not candidate:
                continue
            score = self._echo_similarity(normalized, candidate)
            if score > best:
                best = score
                best_text = str(item.get("text") or "")
                best_message_id = str(item.get("message_id") or "")
        tts = getattr(getattr(self, "runtime", None), "tts", None)
        speaking = bool(getattr(tts, "is_speaking", False))
        ignore_while_speaking = bool(getattr(self, "stt_ignore_while_tts_speaking", True))
        active_until = float(getattr(self, "_tts_until", 0.0) or 0.0)
        active_window = bool(speaking or now <= active_until)
        self._tts_active = active_window
        rejected = best >= threshold
        if ignore_while_speaking and speaking and not best_text and not normalized:
            rejected = True
        return {
            "rejected": rejected,
            "similarity": round(best, 3),
            "threshold": threshold,
            "tts_speaking": speaking,
            "tts_active": active_window,
            "tts_until": active_until,
            "matched_tts_text": best_text,
            "matched_tts_message_id": best_message_id,
        }

    def _reject_self_tts_echo_if_needed(self, raw_text: str) -> bool:
        metrics = self._stt_self_tts_echo_metrics(raw_text)
        if not metrics.get("rejected"):
            if metrics.get("tts_active") or float(metrics.get("similarity") or 0.0) > 0.35:
                safe_text = str(raw_text or "").replace('"', '\\"')
                print(
                    "[HEBE][ECHO_SUPPRESSION] "
                    f"allowed user_speech similarity={metrics.get('similarity')} "
                    f"reason=not_tts_echo text=\"{safe_text}\"",
                    flush=True,
                )
            return False
        safe_text = str(raw_text or "").replace('"', '\\"')
        print(
            "[HEBE][ECHO_SUPPRESSION] "
            f"ignored self_tts_echo similarity={metrics.get('similarity')} text=\"{safe_text}\"",
            flush=True,
        )
        self._emit_stt_rejection(
            raw_text,
            script=self._unsupported_stt_script(raw_text) or "latin",
            reason="self_tts_echo",
            details={key: value for key, value in metrics.items() if key != "rejected"},
        )
        return True

    def _stt_has_meaningful_conversation_content(self, raw_text: str, normalized_text: str) -> bool:
        raw = str(raw_text or "")
        normalized = self._normalize_text(normalized_text)
        tokens = normalized.split()
        if not tokens or normalized in {".", "eh", "mmm", "um", "hebe", "ebe", "eve", "jebe", "heve"}:
            return False
        if self._stt_prompt_echo_metrics(raw_text).get("rejected"):
            return False
        without_names = [token for token in tokens if token not in {"hebe", "ebe", "eve", "jebe", "heve"}]
        if not without_names:
            return False
        meaningful_markers = {
            "como", "estas", "estás", "que", "qué", "cual", "cuál", "donde", "dónde",
            "cuando", "cuándo", "puedes", "quiero", "necesito", "ayuda", "dime",
            "cuenta", "explica", "abre", "haz", "pon", "activa", "desactiva",
            "despierta", "duerme",
        }
        return "?" in raw or bool(set(tokens) & meaningful_markers) or len(without_names) >= 3

    def _publish_accepted_stt_user_input(self, text: str) -> None:
        value = str(text or "").strip()
        if not value:
            return
        seen = getattr(self, "_stt_visible_transcripts", None)
        if seen is None:
            seen = set()
            self._stt_visible_transcripts = seen
        if value in seen:
            return
        seen.add(value)
        if len(seen) > 20:
            self._stt_visible_transcripts = set(list(seen)[-10:])
        try:
            log_chat("user", value, source="stt_voice")
        except Exception as exc:
            print(f"[HEBE][CHAT_LOG] stt user log failed: {exc!r}", flush=True)
        emit("chat.user", {"text": value, "source": "stt_voice"})

    def _assistant_reply_opens_conversation_turn(self, text: str) -> tuple[bool, str]:
        value = str(text or "").strip()
        if not value or "?" not in value and "¿" not in value:
            return False, ""
        normalized = self._normalize_for_echo_match(value)
        casual_markers = {
            "tu que tal", "tú qué tal", "como estas", "como estás", "que tal",
            "todo bien", "como vas", "como lo llevas",
        }
        clarification_markers = {
            "a quien", "a quién", "te refieres", "local o stream", "lo hago",
            "quieres que", "confirmas", "cual", "cuál",
        }
        if any(marker in normalized for marker in clarification_markers):
            return True, "clarification"
        if any(marker in normalized for marker in casual_markers):
            return True, "casual_answer"
        if re.search(r"\b(?:quieres|puedes|dime|confirmas|hago|refieres)\b", normalized):
            return True, "clarification"
        return True, "casual_answer"

    def _record_assistant_reply_for_conversation(self, text: str, *, source: str = "assistant", synthesizer=None) -> None:
        local_sources = {"ui", "voice", "stt_voice"}
        if source not in local_sources:
            print("[HEBE][CONVERSATION] pending_turn_not_created reason=stream_event_or_spontaneous", flush=True)
            return
        opens = bool(getattr(synthesizer, "last_opens_conversation_turn", False)) if synthesizer is not None else False
        expected_type = str(getattr(synthesizer, "last_expected_reply_type", "") or "") if synthesizer is not None else ""
        if not opens:
            print("[HEBE][CONVERSATION] pending_turn_not_created reason=no_synthesizer_marker", flush=True)
            return
        if not expected_type:
            _, expected_type = self._assistant_reply_opens_conversation_turn(text)
        if not opens:
            return
        now = time.time()
        legacy_ttl = float(getattr(self, "pending_conversation_ttl_seconds", 45) or 45)
        has_legacy_override = abs(legacy_ttl - 45.0) > 0.001
        ttl_by_type = {
            "casual_answer": float(os.getenv("HEBE_PENDING_CASUAL_TTL_SECONDS", str(legacy_ttl if has_legacy_override else 40)) or 40),
            "clarification": float(os.getenv("HEBE_PENDING_CLARIFICATION_TTL_SECONDS", str(legacy_ttl if has_legacy_override else 55)) or 55),
            "action_confirmation": float(os.getenv("HEBE_PENDING_ACTION_CONFIRMATION_TTL_SECONDS", str(legacy_ttl if has_legacy_override else 60)) or 60),
        }
        ttl = ttl_by_type.get(expected_type, legacy_ttl)
        turn = {
            "expected_type": expected_type,
            "previous_assistant_message_id": f"assistant-{int(now * 1000)}",
            "previous_assistant_message": str(text or "").strip(),
            "created_at": now,
            "expires_at": now + ttl,
            "source": "assistant_question",
            "allowed_sources": ["stt_voice", "ui"],
            "allow_without_wakeword": True,
            "status": "pending",
            "followups_used": 0,
            "max_followups": int(getattr(self, "pending_conversation_max_followups", 1) or 1),
            "reply_source": source,
        }
        setattr(self.runtime.state, "pending_conversation_turn", turn)
        print(
            "[HEBE][CONVERSATION] pending_turn_created reason=direct_question source=local "
            f"expected_type={expected_type} ttl={int(ttl)}s",
            flush=True,
        )

    def _get_pending_conversation_turn(self) -> dict | None:
        turn = getattr(self.runtime.state, "pending_conversation_turn", None)
        if not isinstance(turn, dict) or turn.get("status") != "pending":
            return None
        now = time.time()
        if now > float(turn.get("expires_at", 0.0) or 0.0):
            turn["status"] = "expired"
            setattr(self.runtime.state, "pending_conversation_turn", turn)
            print("[HEBE][CONVERSATION] pending_turn expired", flush=True)
            return None
        return turn

    def _pending_conversation_matches(self, *, source: str, text: str | None = None, event_type: str | None = None) -> bool:
        turn = self._get_pending_conversation_turn()
        if not turn:
            return False
        allowed = set(turn.get("allowed_sources") or ["stt_voice", "ui"])
        if source not in allowed:
            print(f"[HEBE][FOLLOWUP_GATE] rejected reason=source_not_allowed source={source}", flush=True)
            return False
        if source == "stt_voice":
            if self._stt_self_tts_echo_metrics(text or "").get("rejected"):
                print("[HEBE][CONVERSATION] pending_turn_not_matched reason=self_tts_echo", flush=True)
                print("[HEBE][FOLLOWUP_GATE] rejected reason=self_tts_echo", flush=True)
                return False
            normalized = self._normalize_stt_metric_text(text or "")
            if not normalized or normalized in {"mmm", "um", "eh", "vale", "ok"}:
                print("[HEBE][FOLLOWUP_GATE] rejected reason=empty_or_backchannel", flush=True)
                return False
            media_detected, _media_reason = looks_like_media_or_singing(normalized)
            if media_detected:
                print("[HEBE][FOLLOWUP_GATE] rejected reason=media_or_singing", flush=True)
                return False
            ambient_types = {
                "victory", "objective_update", "location_update",
                "gameplay_failure", "boss_attempt", "grinding", "exploration",
                "menu/equipment", "frustration", "laughter/joke",
            }
            expected = str(turn.get("expected_type") or "")
            if expected != "casual_answer" and event_type in ambient_types:
                print(f"[HEBE][FOLLOWUP_GATE] rejected reason=ambient_stt event_type={event_type}", flush=True)
                return False
            if self._looks_like_stream_ambient_comment(normalized):
                print("[HEBE][FOLLOWUP_GATE] rejected reason=semantic_unrelated", flush=True)
                return False
        print(f"[HEBE][CONVERSATION] pending_turn matched source={source}", flush=True)
        print("[HEBE][FOLLOWUP_GATE] accepted reason=owner_related_followup", flush=True)
        return True

    def _looks_like_stream_ambient_comment(self, normalized: str) -> bool:
        text = str(normalized or "")
        if not text:
            return True
        if re.search(r"\b(?:jaja+|lol|xd)\b", text):
            return True
        ambient_markers = (
            "me han pillado", "no es nada personal", "esta contestando a cosas",
            "está contestando a cosas", "no no no", "donde estan", "donde están",
            "que hago en", "qué hago en", "esto sigue peor", "no es nada",
        )
        return any(marker in text for marker in ambient_markers)

    def _consume_pending_conversation_turn(self) -> None:
        turn = self._get_pending_conversation_turn()
        if not turn:
            return
        used = int(turn.get("followups_used", 0) or 0) + 1
        turn["followups_used"] = used
        if used >= int(turn.get("max_followups", 1) or 1):
            turn["status"] = "consumed"
        setattr(self.runtime.state, "pending_conversation_turn", turn)

    def _is_duplicate_recent_stt(self, raw_text: str) -> tuple[bool, float]:
        normalized = self._normalize_for_echo_match(raw_text)
        if not normalized:
            return False, 0.0
        now = time.time()
        window = float(getattr(self, "stt_duplicate_window_seconds", 8) or 8)
        threshold = float(getattr(self, "stt_duplicate_similarity_threshold", 0.92) or 0.92)
        recent = [
            item for item in list(getattr(self, "_recent_stt_transcripts", []) or [])
            if now - float(item.get("ts", 0.0) or 0.0) <= window
        ]
        self._recent_stt_transcripts = recent
        best = 0.0
        for item in recent:
            candidate = str(item.get("normalized") or "")
            best = max(best, SequenceMatcher(None, normalized, candidate).ratio())
        if best >= threshold:
            return True, best
        recent.append({"raw_text": str(raw_text or ""), "normalized": normalized, "ts": now})
        self._recent_stt_transcripts = recent[-10:]
        return False, best

    def _reject_unsupported_stt_if_needed(self, raw_text: str) -> bool:
        script = self._unsupported_stt_script(raw_text)
        if not script:
            return False
        self._emit_stt_rejection(raw_text, script=script, reason="unsupported_script")
        return True

    def _retry_unsupported_stt_transcript(self, raw_text: str, *, script: str) -> dict:
        stt = getattr(self.runtime, "stt", None)
        if stt is None or not hasattr(stt, "retry_last_command_transcript"):
            return {"attempted": False, "text": "", "speech_detected": False, "reason": "retry_unavailable"}
        speech_detected = bool(getattr(stt, "last_speech_detected", False))
        if not speech_detected:
            return {"attempted": False, "text": "", "speech_detected": False, "reason": "no_speech_detected"}
        language = os.getenv("HEBE_STT_COMMAND_LANGUAGE", "es").strip().lower() or "es"
        print(f"[HEBE][STT][RETRY] reason=unsupported_script forcing_language={language}", flush=True)
        try:
            retry = stt.retry_last_command_transcript(language=language)
        except Exception as exc:
            print(f"[HEBE][STT][RETRY_RESULT] raw='' accepted=false error={exc!r}", flush=True)
            return {"attempted": True, "text": "", "speech_detected": True, "error": repr(exc)}
        if isinstance(retry, str):
            retry = {"text": retry, "attempted": True, "speech_detected": True}
        retry_text = str((retry or {}).get("text") or "").strip()
        if is_stt_prompt_injection(retry_text):
            print("[HEBE][STT][RETRY_RESULT] raw='<suppressed>' accepted=false reason=stt_prompt_injection", flush=True)
            return {
                **(retry or {}),
                "attempted": bool((retry or {}).get("attempted", True)),
                "text": "",
                "accepted": False,
                "prompt_injection": True,
                "original_script": script,
                "forcing_language": language,
            }
        accepted = bool(retry_text and not self._unsupported_stt_script(retry_text))
        print(f"[HEBE][STT][RETRY_RESULT] raw={ascii(retry_text)} accepted={str(accepted).lower()}", flush=True)
        return {
            **(retry or {}),
            "attempted": bool((retry or {}).get("attempted", True)),
            "text": retry_text,
            "accepted": accepted,
            "original_script": script,
            "forcing_language": language,
        }

    def _process_stt_voice_transcript(self, raw_voice_command: str, *, allow_wakeword_prompt: bool = False) -> str:
        original_raw_text = str(raw_voice_command)
        transcript_for_cognition = original_raw_text
        retry_debug: dict = {
            "detected_script": self._unsupported_stt_script(original_raw_text) or "latin",
            "retry_attempted": False,
            "retry_transcript": "",
            "status": "accepted",
            "final_decision": "accepted",
        }
        script = self._unsupported_stt_script(original_raw_text)
        if not script:
            if self._reject_self_tts_echo_if_needed(original_raw_text):
                return "continue"
            if self._reject_stt_prompt_echo_if_needed(original_raw_text):
                return "continue"
            if is_stt_prompt_injection(original_raw_text):
                self._emit_stt_rejection(
                    original_raw_text,
                    script="latin",
                    reason="stt_prompt_injection",
                )
                return "continue"
            duplicate, similarity = self._is_duplicate_recent_stt(original_raw_text)
            if duplicate:
                self._emit_stt_rejection(
                    original_raw_text,
                    script="latin",
                    reason="duplicate_recent_transcript",
                    details={"similarity": round(similarity, 3)},
                )
                return "continue"
        else:
            retry = self._retry_unsupported_stt_transcript(original_raw_text, script=script)
            retry_attempted = bool(retry.get("attempted"))
            retry_text = str(retry.get("text") or "").strip()
            retry_debug.update(
                {
                    "detected_script": script,
                    "retry_attempted": retry_attempted,
                    "retry_transcript": retry_text,
                    "forcing_language": retry.get("forcing_language") or os.getenv("HEBE_STT_COMMAND_LANGUAGE", "es"),
                }
            )
            if bool(retry.get("prompt_injection")):
                self._emit_stt_rejection(
                    original_raw_text,
                    script=script,
                    reason="stt_prompt_echo_or_hotword_list",
                    retry_attempted=retry_attempted,
                    retry_transcript="",
                )
                return "continue"
            if bool(retry.get("accepted")):
                transcript_for_cognition = retry_text
                retry_debug["final_decision"] = "accepted"
                if self._reject_self_tts_echo_if_needed(transcript_for_cognition):
                    return "continue"
                if self._reject_stt_prompt_echo_if_needed(
                    transcript_for_cognition,
                    retry_attempted=retry_attempted,
                    retry_transcript=retry_text,
                ):
                    return "continue"
                duplicate, similarity = self._is_duplicate_recent_stt(transcript_for_cognition)
                if duplicate:
                    self._emit_stt_rejection(
                        original_raw_text,
                        script=script,
                        reason="duplicate_recent_transcript",
                        retry_attempted=retry_attempted,
                        retry_transcript=retry_text,
                        details={"similarity": round(similarity, 3)},
                    )
                    return "continue"
            else:
                reason = "unsupported_script_after_retry" if retry_attempted else "unsupported_script"
                self._emit_stt_rejection(
                    original_raw_text,
                    script=script,
                    reason=reason,
                    retry_attempted=retry_attempted,
                    retry_transcript=retry_text,
                )
                return "continue"

        normalization = self._normalize_stt_input(transcript_for_cognition, debug_metadata=retry_debug)
        command = normalization.normalized_text
        if not command:
            return "continue"
        self._current_input_event = self._build_input_event(
            source="stt_voice",
            raw_text=original_raw_text,
            normalized_text=command,
            stt_metadata={**normalization.as_event(), **retry_debug, "accepted_transcript": transcript_for_cognition},
        )
        voice_type, mood_hint = self._classify_voice_event(command)
        has_action_intent = self._input_event_has_action_intent(getattr(self, "_current_input_event", None))
        media_detected, _media_reason = looks_like_media_or_singing(command)
        try:
            possible_reply_to_hebe = self._get_live_session_brain().is_possible_reply_to_hebe(command)
        except Exception:
            possible_reply_to_hebe = False
        pending_match = self._pending_conversation_matches(source="stt_voice", text=command, event_type=voice_type)
        conversation_followup = (
            not has_action_intent
            and voice_type != "direct_command_to_hebe"
            and self._stt_has_meaningful_conversation_content(transcript_for_cognition, command)
            and (
                pending_match
                or possible_reply_to_hebe
            )
        )
        envelope = self._build_stt_input_envelope(
            self._current_input_event,
            voice_type=voice_type,
            conversation_followup=conversation_followup,
        )
        pending_followup = envelope.is_followup_candidate
        is_direct_command = envelope.source in {"owner_stt_direct", "owner_stt_command"}
        if media_detected and envelope.source == "ambient_stt":
            firewall = self._input_firewall_decision(
                source=envelope.source,
                text=command,
                event_type=voice_type,
                addressed_to_hebe=envelope.addressed_to_hebe,
                has_action_intent=has_action_intent,
            )
            if self._current_input_event is not None:
                self._current_input_event.stt_metadata["input_firewall"] = firewall.as_dict()
            self._current_input_event = None
            return "continue"
        firewall = self._input_firewall_decision(
            source=envelope.source,
            text=command,
            event_type=voice_type,
            addressed_to_hebe=envelope.addressed_to_hebe,
            pending_followup=pending_followup,
            has_action_intent=has_action_intent,
        )
        if self._current_input_event is not None:
            self._current_input_event.stt_metadata["input_firewall"] = firewall.as_dict()
        relevance = ContextRelevance(useful=False, category="none", confidence=0.0, reason="not_evaluated")
        if not is_direct_command and not pending_followup:
            if firewall.firewall_decision == "allow_context_only" and self._is_stream_enabled():
                relevance = self._record_voice_event(command, voice_type, mood_hint)
            classification = self._get_input_classifier().classify(
                self._current_input_event,
                voice_event_type=voice_type,
                addressed_to_hebe=False,
                has_action_intent=False,
                pending_followup=False,
                valid=True,
            )
            self._log_input_classification(classification)
            conversation_state = self._get_conversation_state_resolver().from_pending_turn(
                self._get_pending_conversation_turn(),
                matched=False,
                reason="no_matching_active_conversation",
            )
            self._log_conversation_state(conversation_state)
            output_targets = self._output_targets_for_input_type(classification.input_type)
            response_decision = self._get_response_decision_resolver().decide(
                classification=classification,
                conversation_state=conversation_state,
                relevance=relevance,
                output_targets=output_targets,
            )
            self._log_knowledge_resolution()
            self._log_response_decision(response_decision)
            self._build_response_frame(
                event=self._current_input_event,
                classification=classification,
                conversation_state=conversation_state,
                response_decision=response_decision,
            )
            print("[HEBE][STT_GATE] ambient_only reason=no_wake_no_valid_pending", flush=True)
            self._declare_output_route(
                input_type="ambient_stt",
                targets=response_decision.output_target,
                reason=response_decision.reason,
            )
            if firewall.firewall_decision == "allow_context_only":
                self._log_stt_non_command_decision(
                    command,
                    "ambient_context_updated" if relevance.useful else "ambient_ignored_low_value",
                    reason=firewall.reason,
                )
            else:
                self._log_stt_non_command_decision(command, "ambient_ignored_low_value", reason=firewall.reason)
            self._current_input_event = None
            return "continue"
        pending_turn_for_frame = self._get_pending_conversation_turn()
        if envelope.pending_compatible:
            conversation_state = ConversationState(
                active=True,
                topic=str((envelope.active_pending or {}).get("kind") or "pending_task"),
                source="cognitive_pending_task",
                expected_reply_type=envelope.expected_reply_type,
                allow_no_wakeword=True,
                output_target=[OUTPUT_TARGET_LOCAL_UI, self._direct_voice_tts_target()],
                confidence=0.95,
                matched=True,
                reason="pending_compatible_input_envelope",
            )
        else:
            conversation_state = self._get_conversation_state_resolver().from_pending_turn(
                pending_turn_for_frame,
                matched=bool(pending_followup),
                reason="active_conversation_state" if pending_followup else "no_matching_active_conversation",
            )
        if possible_reply_to_hebe and pending_followup and not conversation_state.active:
            last_utterance = getattr(self._get_live_session_brain().state, "last_hebe_utterance", {}) or {}
            conversation_state = ConversationState(
                active=True,
                topic=str(last_utterance.get("topic") or "reply_to_hebe"),
                source="hebe_utterance_window",
                last_assistant_reply=str(last_utterance.get("text") or ""),
                expected_reply_type="correction_or_ack",
                allow_no_wakeword=True,
                output_target=[OUTPUT_TARGET_LOCAL_UI, self._direct_voice_tts_target()],
                confidence=0.82,
                matched=True,
                reason="last_hebe_utterance_window",
            )
        self._log_conversation_state(conversation_state)
        classification = self._get_input_classifier().classify(
            self._current_input_event,
            voice_event_type=voice_type,
            addressed_to_hebe=is_direct_command,
            has_action_intent=has_action_intent,
            pending_followup=pending_followup,
            valid=True,
        )
        self._log_input_classification(classification)
        try:
            self._get_live_session_brain().observe_leo_stt(
                original_raw_text,
                command,
                addressed_to_hebe=is_direct_command,
                voice_event_type=voice_type,
                topic=classification.input_type,
                confidence=float(getattr(classification, "confidence", 0.72) or 0.72),
            )
        except Exception as exc:
            print(f"[HEBE][LIVE_SESSION] stt observe failed: {exc!r}", flush=True)
        if is_direct_command and self._firewall_allows_pipeline(firewall):
            gate_reason = (
                "owner_direct_addressed_to_hebe"
                if envelope.addressed_to_hebe
                else "high_confidence_local_command"
            )
            print(f"[HEBE][STT_GATE] passed reason={gate_reason}", flush=True)
        stream_enabled = self._is_stream_enabled()
        if pending_followup:
            self._current_input_event.stt_metadata["message_type"] = (
                "pending_reply" if envelope.pending_compatible else "conversation_followup"
            )
            self._current_input_event.stt_metadata["conversation_followup"] = not envelope.pending_compatible
            self._current_input_event.stt_metadata["jarvis_allowed"] = True
            if envelope.pending_compatible:
                print("[HEBE][COG] decision=pending_datetime_followup", flush=True)
            else:
                self._consume_pending_conversation_turn()
                print("[HEBE][COG] decision=conversation_followup", flush=True)
        elif stream_enabled:
            if is_direct_command or not pending_followup:
                relevance = self._record_voice_event(command, voice_type, mood_hint)
            if not is_direct_command and not self._stream_is_armed() and not has_action_intent:
                print("[HEBE][STT_GATE] ambient_only reason=no_wake_no_valid_pending", flush=True)
                output_targets = self._output_targets_for_input_type(classification.input_type)
                response_decision = self._get_response_decision_resolver().decide(
                    classification=classification,
                    conversation_state=conversation_state,
                    relevance=relevance,
                    output_targets=output_targets,
                )
                self._log_knowledge_resolution()
                self._log_response_decision(response_decision)
                self._build_response_frame(
                    event=self._current_input_event,
                    classification=classification,
                    conversation_state=conversation_state,
                    response_decision=response_decision,
                )
                self._declare_output_route(
                    input_type="ambient_stt",
                    targets=response_decision.output_target,
                    reason=response_decision.reason,
                )
                decision_name = "ambient_context_updated" if relevance.useful else "ambient_ignored_low_value"
                self._log_stt_non_command_decision(command, decision_name, reason=voice_type)
                self._current_input_event = None
                return "continue"
        elif not is_direct_command and not has_action_intent:
            print("[HEBE][JARVIS][BLOCKED] reason=stt_not_direct", flush=True)
            output_targets = self._output_targets_for_input_type(classification.input_type)
            response_decision = self._get_response_decision_resolver().decide(
                classification=classification,
                conversation_state=conversation_state,
                relevance=relevance,
                output_targets=output_targets,
            )
            self._log_response_decision(response_decision)
            self._declare_output_route(
                input_type="ambient_stt",
                targets=response_decision.output_target,
                reason=response_decision.reason,
            )
            self._log_stt_non_command_decision(command, "ambient_ignored_low_value", reason="not_direct_command")
            self._current_input_event = None
            return "continue"
        if envelope.input_type == "local_app_command" and envelope.addressed_to_hebe:
            command = str(envelope.wake_evidence.get("stripped_text") or command).strip()
        elif not pending_followup:
            handled, stream_command = self._extract_stream_command(command)
            if handled:
                if not stream_command:
                    self._current_input_event = None
                    return "continue"
                command = stream_command
        if self._current_input_event is not None:
            self._current_input_event.stt_metadata["jarvis_allowed"] = bool(is_direct_command or pending_followup or has_action_intent)
        targets = [OUTPUT_TARGET_LOCAL_UI]
        if self._local_tts_output_enabled():
            targets.append(self._direct_voice_tts_target())
        response_decision = self._get_response_decision_resolver().decide(
            classification=classification,
            conversation_state=conversation_state,
            relevance=relevance,
            output_targets=targets,
        )
        self._log_knowledge_resolution()
        self._log_response_decision(response_decision)
        self._build_response_frame(
            event=self._current_input_event,
            classification=classification,
            conversation_state=conversation_state,
            response_decision=response_decision,
        )
        self._declare_output_route(
            input_type="direct_stt" if not pending_followup else "direct_stt_followup",
            targets=response_decision.output_target,
            reason=response_decision.reason,
        )
        self._publish_accepted_stt_user_input(transcript_for_cognition)
        res = self.handle_command(command, source="stt_voice")
        print("[HEBE][COG] decision=conversation_followup" if pending_followup else "[HEBE][COG] decision=command", flush=True)
        self._current_input_event = None
        return res

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
            f"source={event.source} raw={ascii(event.raw_text)} normalized={ascii(event.normalized_text)}",
            flush=True,
        )
        print(
            "[HEBE][NORMALIZE] "
            f"normalized={ascii(event.normalized_text)}",
            flush=True,
        )
        return event

    def _build_stt_input_envelope(
        self, event: InputEvent, *, voice_type: str, conversation_followup: bool,
    ) -> InputEnvelope:
        normalized = self._normalize_text(event.normalized_text)
        resolver = getattr(self, "wake_name_resolver", None) or WakeNameResolver()
        self.wake_name_resolver = resolver
        wake = resolver.resolve(
            raw_text=event.raw_text,
            normalized_text=normalized,
            source="stt_voice",
            is_sleeping=bool(getattr(self.runtime.state, "hebe_sleeping", False)),
            command_markers=set(self._get_local_app_planner().command_markers()),
        )
        # A trusted command marker is command evidence, not wake-name evidence.
        # WakeNameResolver intentionally accepts that context, but the unified
        # envelope keeps the two facts separate so no-wake commands are routed
        # as owner_stt_command rather than pretending a name was spoken.
        addressed = bool(wake.matched_name or voice_type == "direct_command_to_hebe")
        command_mode = bool((event.stt_metadata or {}).get("command_mode", True))

        local_plan = self._get_local_app_planner().plan(
            event,
            is_awake=not bool(getattr(self.runtime.state, "hebe_sleeping", False)),
        )
        app_result: dict = {}
        intent_candidates: list[str] = []
        app_target = None
        if local_plan is not None and local_plan.action_type == "open_application":
            app_target = local_plan.target or (local_plan.slots or {}).get("application_target")
            intent_candidates.append("open_application")
            app_result = {
                "status": local_plan.status,
                "confidence": float(local_plan.confidence or 0.0),
                "reason": local_plan.reason,
                "target": app_target,
                "whitelisted": bool((local_plan.context_checks or {}).get("whitelisted")),
            }

        pending = getattr(self.runtime.state, "pending_clarification", None)
        active_pending = pending if isinstance(pending, dict) else None
        if active_pending:
            try:
                if float(active_pending.get("expires_at") or 0) and float(active_pending["expires_at"]) <= time.time():
                    active_pending = None
            except (TypeError, ValueError):
                active_pending = None
        pending_kind = str((active_pending or {}).get("kind") or "")
        expected_reply_type = "datetime" if pending_kind == "appointment_datetime" else ""
        router = getattr(self, "cognitive_router", None) or CognitiveRouter()
        stronger_request = bool(
            router._is_current_time_query(normalized)
            or router._is_current_date_query(normalized)
            or router._open_app_target(normalized)
            or router._is_reminder_request(normalized)
            or router._personal_state(normalized)
        )
        pending_compatible = bool(
            active_pending
            and str(active_pending.get("authority") or "owner") == "owner"
            and pending_kind == "appointment_datetime"
            and router._is_datetime_answer(normalized)
            and not stronger_request
        )

        high_confidence_local_app = bool(
            command_mode
            and app_result.get("whitelisted")
            and float(app_result.get("confidence") or 0.0) >= 0.8
        )
        if pending_compatible:
            source, authority, trust = "owner_stt_followup", "owner", "trusted_followup"
            input_type, reason = "pending_reply", "datetime_answer"
        elif addressed:
            source, authority, trust = "owner_stt_direct", "owner", "trusted_direct"
            input_type = (
                "local_app_command" if app_target
                else "explicit_question" if router._looks_like_question(normalized)
                else "direct_to_hebe"
            )
            reason = "wake_or_addressing_evidence"
        elif high_confidence_local_app:
            source, authority, trust = "owner_stt_command", "owner", "trusted_direct"
            input_type, reason = "local_app_command", "high_confidence_local_command"
        elif conversation_followup:
            source, authority, trust = "owner_stt_followup", "owner", "trusted_followup"
            input_type, reason = "active_conversation_followup", "active_conversation_state"
        else:
            source, authority, trust = "ambient_stt", "ambient", "untrusted_ambient"
            input_type, reason = "ambient_stream_context", "no_wake_no_pending_no_command"

        envelope = InputEnvelope(
            raw_text=event.raw_text,
            normalized_text=normalized,
            source=source,
            authority=authority,
            trust=trust,
            addressed_to_hebe=addressed,
            matched_wake_name=getattr(wake, "matched_name", None),
            wake_evidence={
                "reason": getattr(wake, "reason", ""),
                "confidence": float(getattr(wake, "confidence", 0.0) or 0.0),
                "canonical": getattr(wake, "canonical", None),
                "stripped_text": getattr(wake, "stripped_text", ""),
            },
            command_mode=command_mode,
            intent_candidates=intent_candidates,
            app_target=app_target,
            app_resolver_result=app_result,
            active_pending=active_pending,
            pending_compatible=pending_compatible,
            expected_reply_type=expected_reply_type,
            is_followup_candidate=bool(pending_compatible or conversation_followup),
            input_type=input_type,
            reason=reason,
        )
        event.envelope = envelope
        self._last_input_envelope = envelope
        event.stt_metadata["input_envelope"] = envelope.as_dict()
        print(
            "[HEBE][INPUT_ENVELOPE] "
            f"source={source} authority={authority} trust={trust} "
            f"addressed={str(addressed).lower()} pending_compatible={str(pending_compatible).lower()} "
            f"intent_candidates={intent_candidates!r}",
            flush=True,
        )
        if active_pending:
            print(
                "[HEBE][PENDING_FOLLOWUP_GATE] active=true "
                f"compatible={str(pending_compatible).lower()} "
                f"source={source if pending_compatible else 'none'} "
                f"reason={'datetime_answer' if pending_compatible else 'not_datetime_or_new_request'}",
                flush=True,
            )
        print(
            f"[HEBE][SOURCE_CLASSIFY] source={source} reason={reason}"
            + (f" app={app_target}" if app_target else ""),
            flush=True,
        )
        return envelope

    def _input_event_has_action_intent(self, event: InputEvent | None) -> bool:
        if event is None:
            return False
        try:
            local_plan = self._get_local_app_planner().plan(
                event,
                is_awake=not bool(getattr(self.runtime.state, "hebe_sleeping", False)),
            )
            if local_plan is not None and local_plan.status in {"complete", "needs_confirmation"}:
                return True
            plan = self._get_stream_action_planner().plan(event)
            return plan is not None
        except Exception as exc:
            print(f"[HEBE][ACTION_PLAN] probe failed: {exc!r}", flush=True)
            return False

    def _get_local_app_planner(self) -> LocalAppActionPlanner:
        planner = getattr(self, "local_app_planner", None)
        if planner is None:
            planner = LocalAppActionPlanner(getattr(self, "wake_name_resolver", None) or WakeNameResolver())
            self.local_app_planner = planner
        return planner

    def _plan_and_execute_local_app_action(self, command: str, source: str) -> CommandResult | None:
        normalized = self._normalize_text(command)
        input_event = getattr(self, "_current_input_event", None) or self._build_input_event(
            source="ui" if source in {"ui", "typed_ui"} else source,
            raw_text=command,
            normalized_text=normalized,
        )
        planner = self._get_local_app_planner()
        plan = planner.plan(
            input_event,
            is_awake=not bool(getattr(self.runtime.state, "hebe_sleeping", False)),
        )
        if plan is None:
            return None
        if plan.status == "rejected" and plan.reason == "app_not_whitelisted":
            print(
                "[HEBE][ACTION_PLAN] "
                f"action_type={plan.action_type} target={plan.target} status={plan.status} reason={plan.reason}",
                flush=True,
            )
            return None

        print(
            "[HEBE][ACTION_PLAN] "
            f"action_type={plan.action_type} target={plan.target} status={plan.status}",
            flush=True,
        )
        emit("voice.command", {
            "raw_text": input_event.raw_text,
            "normalized_text": input_event.normalized_text,
            "intent": plan.action_type,
            "target": plan.target,
            "confidence": round(float(plan.confidence), 3),
            "status": plan.status,
            "reason": plan.reason,
            "source": input_event.source,
            "final_decision": "accepted" if plan.status == "complete" else "rejected",
        })

        if plan.reason == "app_path_missing":
            print(
                "[HEBE][ACTION_EXECUTOR] "
                "action_type=open_application success=false error_code=app_path_missing",
                flush=True,
            )
            app_name = plan.slots.get("display_name") or plan.target or "la aplicacion"
            return CommandResult(
                action_type="open_application",
                success=False,
                user_visible_summary=(
                    f"{app_name} is recognized but its executable path is not configured."
                ),
                state_changes={
                    "app_id": plan.slots.get("app_id") or plan.target,
                    "app_name": app_name,
                    "error_code": "app_path_missing",
                },
                constraints=[
                    "Do not ask for remote access.",
                    "Do not give manual app-opening instructions.",
                    "Ask Leo to configure HEBE_APP_OBS_PATH or the app registry path.",
                ],
                fallback_text=(
                    f"Reconozco {app_name}, pero no tengo configurada su ruta ejecutable. "
                    "Configura HEBE_APP_OBS_PATH o la ruta en el registro de apps."
                ),
                requires_model_response=True,
                metadata={
                    "action_plan": plan.as_log_dict(),
                    "error_code": "app_path_missing",
                    "app_id": plan.slots.get("app_id") or plan.target,
                    "message_goal": (
                        "Tell Leo that OBS is recognized but the executable path is not configured, "
                        "and ask him to configure HEBE_APP_OBS_PATH or app registry path."
                    ),
                },
            )

        print(
            "[HEBE][ACTION_EXECUTOR] "
            f"executing action_type={plan.action_type}",
            flush=True,
        )
        action_result = self.action_runtime.execute(
            "open_application",
            {
                "app_id": plan.slots.get("app_id") or plan.target,
                "app_record": plan.slots.get("app_record"),
            },
        )
        success = bool(getattr(action_result, "success", False))
        payload = getattr(action_result, "data", {}) or {}
        error_code = payload.get("error_code") or getattr(action_result, "error", None)
        print(
            "[HEBE][ACTION_EXECUTOR] "
            f"action_type=open_application success={str(success).lower()}"
            + (f" error_code={error_code}" if error_code else ""),
            flush=True,
        )
        app_name = payload.get("app_name") or plan.slots.get("display_name") or plan.target or "la aplicacion"
        fallback = f"Abriendo {app_name}." if success else f"Reconozco {app_name}, pero no he podido abrirla."
        return CommandResult(
            action_type="open_application",
            success=success,
            user_visible_summary=fallback,
            state_changes={
                "app_id": payload.get("app_id") or plan.slots.get("app_id") or plan.target,
                "app_name": app_name,
                "error_code": error_code,
            },
            constraints=[
                "Do not ask for remote access.",
                "Do not give manual app-opening instructions unless action_unavailable/manual_help_requested is present.",
                "Do not ask whether to open it.",
            ],
            fallback_text=fallback,
            requires_model_response=True,
            metadata={
                "action_plan": plan.as_log_dict(),
                "error_code": error_code,
                "message_goal": (
                    f"Confirm that {app_name} is opening locally."
                    if success
                    else f"Tell Leo that {app_name} was recognized but could not be opened locally."
                ),
            },
        )

    def _today_at(self, hhmm: str) -> datetime:
        hour, minute = [int(part) for part in hhmm.split(":", 1)]
        now = datetime.now(ZoneInfo("Europe/Madrid"))
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def _get_stream_state(self):
        return getattr(self.runtime.state, "stream", None)

    def _declare_output_route(
        self,
        *,
        input_type: str,
        targets: list[str] | tuple[str, ...] | set[str],
        reason: str = "",
        event_type: str | None = None,
    ) -> None:
        clean_targets = [str(target) for target in targets if str(target or "").strip()]
        target_text = "+".join(clean_targets) if clean_targets else "none"
        suffix = ""
        if event_type:
            suffix += f" event_type={event_type}"
        if reason:
            suffix += f" reason={reason}"
        print(
            f"[HEBE][ROUTING] input_type={input_type} output_target={target_text}{suffix}",
            flush=True,
        )

    def _get_input_classifier(self) -> InputClassifier:
        classifier = getattr(self, "input_classifier", None)
        if classifier is None:
            classifier = InputClassifier()
            self.input_classifier = classifier
        return classifier

    def _get_conversation_state_resolver(self) -> ConversationStateResolver:
        resolver = getattr(self, "conversation_state_resolver", None)
        if resolver is None:
            resolver = ConversationStateResolver()
            self.conversation_state_resolver = resolver
        return resolver

    def _get_knowledge_policy_resolver(self) -> KnowledgePolicyResolver:
        resolver = getattr(self, "knowledge_policy_resolver", None)
        if resolver is None:
            resolver = KnowledgePolicyResolver()
            self.knowledge_policy_resolver = resolver
        return resolver

    def _get_response_decision_resolver(self) -> ResponseDecisionResolver:
        resolver = getattr(self, "response_decision_resolver", None)
        if resolver is None:
            resolver = ResponseDecisionResolver()
            self.response_decision_resolver = resolver
        return resolver

    def _output_targets_for_input_type(self, input_type: str, *, event_type: str | None = None) -> list[str]:
        if input_type in {"ambient_stream_context", "twitch_chat_observed"}:
            return [OUTPUT_TARGET_SILENT_CONTEXT_UPDATE]
        if input_type in {"direct_to_hebe", "explicit_command", "explicit_question", "active_conversation_followup"}:
            targets = [OUTPUT_TARGET_LOCAL_UI]
            if self._local_tts_output_enabled():
                targets.append(self._direct_voice_tts_target())
            return targets
        if input_type == "twitch_chat_mention":
            targets = [OUTPUT_TARGET_TWITCH_CHAT]
            if self._stream_tts_output_enabled_for_event(event_type or "twitch_chat_react"):
                targets.append(OUTPUT_TARGET_STREAM_TTS)
            return targets
        if input_type == "system_event":
            if event_type == "twitch_idle_prompt":
                if self._spontaneous_twitch_chat_enabled():
                    return [OUTPUT_TARGET_TWITCH_CHAT]
                targets = [OUTPUT_TARGET_STREAM_TTS] if self._stream_tts_output_enabled_for_event(event_type) else [OUTPUT_TARGET_LOCAL_UI]
                return targets
            return [OUTPUT_TARGET_LOCAL_UI]
        return ["none"]

    def _build_response_frame(
        self,
        *,
        event: InputEvent | None,
        classification,
        conversation_state,
        response_decision,
        action_plan: ActionPlan | None = None,
    ) -> ResponseFrame:
        stream = self._get_stream_state()
        session_context = {}
        if stream is not None:
            session_context = {
                "current_objective": getattr(stream, "current_run_objective", None),
                "current_location": getattr(stream, "current_run_location", None),
                "current_phase": getattr(stream, "current_run_phase", None),
                "recent_facts_count": len(getattr(stream, "recent_run_context_facts", []) or []),
            }
        live_context = {}
        try:
            live_context = self._get_live_session_brain().retrieve_context(
                getattr(event, "raw_text", "") if event is not None else "",
                limit_events=12,
                limit_summaries=3,
            )
        except Exception as exc:
            print(f"[HEBE][LIVE_SESSION] response context retrieval failed: {exc!r}", flush=True)
        if live_context:
            session_context["live_session"] = live_context.get("live_state")
            session_context["recent_timeline_events"] = live_context.get("recent_events")
            session_context["rolling_summaries"] = live_context.get("rolling_summaries")
        frame = ResponseFrame(
            input_type=classification.input_type,
            source=classification.source,
            current_game=str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or ""),
            current_session_context={key: value for key, value in session_context.items() if value},
            conversation_state=conversation_state,
            intent=getattr(action_plan, "action_type", "") if action_plan is not None else "",
            action_plan=action_plan.as_log_dict() if action_plan is not None and hasattr(action_plan, "as_log_dict") else None,
            output_target=list(response_decision.output_target),
            allow_question=bool(response_decision.allow_question),
            max_questions=int(response_decision.max_questions),
            max_sentences=int(response_decision.max_sentences),
            should_reply=bool(response_decision.should_reply),
            forbidden_patterns=[
                "assistant_action_offer_without_action_plan",
                "engagement_bait_question",
                "customer_support_closing",
            ],
        )
        if event is not None:
            event.stt_metadata["input_type"] = classification.input_type
            event.stt_metadata["response_frame"] = frame.as_dict()
        return frame

    def _log_input_classification(self, classification) -> None:
        print(
            "[HEBE][INPUT_CLASSIFY] "
            f"source={classification.source} input_type={classification.input_type} "
            f"confidence={classification.confidence:.2f} reason={classification.reason}",
            flush=True,
        )

    def _log_conversation_state(self, state) -> None:
        print(
            "[HEBE][CONVERSATION_STATE] "
            f"active={str(bool(state.active)).lower()} matched={str(bool(state.matched)).lower()} "
            f"expected_reply_type={state.expected_reply_type!r} reason={state.reason}",
            flush=True,
        )

    def _log_knowledge_resolution(self) -> None:
        knowledge = self._get_knowledge_policy_resolver().resolve(
            stream=self._get_stream_state(),
            profile_store=getattr(self, "game_profiles", None),
        )
        print(
            "[HEBE][KNOWLEDGE] "
            f"game={knowledge.game!r} profile_found={str(knowledge.profile_found).lower()} "
            f"lookup_used={str(knowledge.lookup_used).lower()} confidence={knowledge.confidence} "
            f"provenance={knowledge.provenance}",
            flush=True,
        )

    def _log_response_decision(self, decision) -> None:
        print(
            "[HEBE][RESPONSE_DECISION] "
            f"should_reply={str(bool(decision.should_reply)).lower()} reason={decision.reason}",
            flush=True,
        )

    def _stream_tts_output_enabled_for_event(self, event_type: str | None = None) -> bool:
        if not bool(getattr(self.runtime.state, "tts_enabled", False)):
            return False
        if self._stream_output_mode() in {"ui_only", "twitch_chat_only", "silent"}:
            return False
        stream = self._get_stream_state()
        policies = getattr(stream, "policies", None) if stream else None
        if event_type == "twitch_idle_prompt":
            config_enabled = os.getenv("HEBE_SPONTANEOUS_TTS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")
            return bool(config_enabled and policies and getattr(policies, "allow_tts_idle_prompts", False))
        if event_type == "twitch_raid":
            return bool(policies and getattr(policies, "allow_tts_raid_thanks", True))
        if event_type and event_type.startswith("twitch_") and event_type != "twitch_chat_react":
            return bool(policies and getattr(policies, "allow_tts_event_replies", True))
        return bool(policies and getattr(policies, "allow_tts_replies", False))

    def _spontaneous_twitch_chat_enabled(self) -> bool:
        configured = getattr(self, "spontaneous_twitch_chat_enabled", None)
        if configured is None:
            configured = os.getenv(
                "HEBE_SPONTANEOUS_TWITCH_CHAT_ENABLED",
                os.getenv("HEBE_TWITCH_SPONTANEOUS_ENABLED", "false"),
            ).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            self.spontaneous_twitch_chat_enabled = configured
        return bool(configured)

    def _twitch_chatbot_connected(self) -> bool:
        bot = getattr(self.runtime, "twitch_chat_bot", None)
        connected = getattr(bot, "is_connected", False)
        if callable(connected):
            try:
                return bool(connected())
            except Exception:
                return False
        return bool(connected)

    def _spontaneous_twitch_chat_delivery_allowed(self, text: str, payload: dict | None = None) -> tuple[bool, str]:
        if not self._spontaneous_twitch_chat_enabled():
            return False, "twitch_spontaneous_disabled"
        stream = self._get_stream_state()
        if not stream or not getattr(stream, "is_live", False):
            return False, "stream_not_live"
        twitch = getattr(self.runtime, "twitch", None)
        is_available = getattr(twitch, "is_available", None)
        if twitch is None or (callable(is_available) and not is_available()):
            return False, "twitch_unavailable"
        if not self._twitch_chatbot_connected():
            return False, "twitch_chatbot_disconnected"

        payload = payload or {}
        anchors = list(payload.get("specific_context_anchors") or [])
        if not anchors:
            return False, "no_high_quality_anchor"

        now = time.time()
        chat_snapshot = self._chat_activity_snapshot(stream, now=now)
        min_cooldown = 10 * 60 if chat_snapshot.get("active") else 5 * 60
        last_sent = float(getattr(stream, "last_spontaneous_twitch_chat_ts", 0.0) or 0.0)
        if last_sent and now - last_sent < min_cooldown:
            return False, "spontaneous_twitch_cooldown"

        anchor_key = str(payload.get("used_fact_id") or payload.get("idle_topic") or "|".join(anchors)).strip()
        try:
            if self._get_live_session_brain().is_anchor_consumed_or_invalidated(anchor_key):
                return False, "anchor_already_used"
        except Exception:
            pass
        used_anchors = set(getattr(stream, "spontaneous_twitch_used_anchor_keys", set()) or set())
        if anchor_key and anchor_key in used_anchors:
            return False, "anchor_already_used"

        clean_text = str(text or "").strip()
        if not clean_text:
            return False, "empty_message"
        if "\n" in clean_text:
            return False, "multi_line_message"
        return True, "spontaneous_twitch_enabled"

    def _record_spontaneous_twitch_chat_sent(self, text: str, payload: dict | None = None) -> None:
        stream = self._get_stream_state()
        if not stream:
            return
        payload = payload or {}
        stream.last_spontaneous_twitch_chat_ts = time.time()
        anchor_key = str(payload.get("used_fact_id") or payload.get("idle_topic") or "|".join(payload.get("specific_context_anchors") or [])).strip()
        if anchor_key:
            used_anchors = set(getattr(stream, "spontaneous_twitch_used_anchor_keys", set()) or set())
            used_anchors.add(anchor_key)
            stream.spontaneous_twitch_used_anchor_keys = used_anchors

    def _local_tts_output_enabled(self) -> bool:
        if not bool(getattr(self.runtime.state, "tts_enabled", False)):
            return False
        stream = self._get_stream_state()
        stream_active = bool(
            stream
            and (
                getattr(stream, "enabled", False)
                or getattr(stream, "is_live", False)
                or getattr(stream, "live_test_override", False)
            )
        )
        if stream_active and self._stream_output_mode() in {"ui_only", "twitch_chat_only", "silent"}:
            print("[HEBE][ROUTING] output_target=local_ui reason=tts_disabled", flush=True)
            return False
        return True

    def _stream_output_mode(self) -> str:
        stream = self._get_stream_state()
        mode = str(getattr(stream, "stream_output_mode", "tts_enabled") if stream is not None else "tts_enabled").strip()
        if mode not in {"ui_only", "tts_enabled", "twitch_chat_only", "silent"}:
            mode = "tts_enabled"
        return mode

    def set_stream_output_mode(self, mode: str, *, reason: str = "user_setting") -> dict:
        stream = self._get_stream_state()
        clean_mode = str(mode or "").strip()
        if clean_mode not in {"ui_only", "tts_enabled", "twitch_chat_only", "silent"}:
            raise ValueError("invalid stream output mode")
        if stream is not None:
            stream.stream_output_mode = clean_mode
            policies = getattr(stream, "policies", None)
            if policies is not None:
                stream_tts = clean_mode == "tts_enabled"
                policies.allow_tts_replies = stream_tts
                policies.allow_tts_event_replies = stream_tts
                policies.allow_tts_raid_thanks = stream_tts
        print(f"[HEBE][OUTPUT_MODE] mode={clean_mode} reason={reason}", flush=True)
        self._emit_audio_status()
        return {
            "stream_output_mode": clean_mode,
            "stream_tts_enabled": clean_mode == "tts_enabled",
            "reason": reason,
        }

    def _is_stream_enabled(self) -> bool:
        stream = self._get_stream_state()
        return bool(stream and getattr(stream, "enabled", False))

    def _direct_voice_tts_target(self) -> str:
        stream = self._get_stream_state()
        stream_voice = bool(
            stream
            and (
                getattr(stream, "enabled", False)
                or getattr(stream, "is_live", False)
                or getattr(stream, "live_test_override", False)
            )
        )
        return OUTPUT_TARGET_STREAM_TTS if stream_voice else OUTPUT_TARGET_LOCAL_TTS

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
        for alias in STREAM_WAKE_MULTI_ALIASES:
            if normalized == alias:
                self._arm_stream()
                try:
                    vts_hotkey("HebeIdle")
                except Exception as e:
                    print(f"[HEBE] vts_hotkey failed while arming stream: {e!r}", flush=True)
                return True, None
            if normalized.startswith(alias + " "):
                self._disarm_stream()
                return True, normalized[len(alias):].strip()

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
        words = set(normalized.split())
        if (
            words.intersection(STREAM_WAKE_ALIASES)
            or any(normalized == alias or normalized.startswith(alias + " ") for alias in STREAM_WAKE_MULTI_ALIASES)
            or normalized.startswith(("prepara stream", "activa modo stream", "desactiva modo stream"))
        ):
            return "direct_command_to_hebe", None
        if normalized.startswith(("ya hemos pasado ", "hemos pasado ")):
            return "completed_marker", None
        if any(marker in normalized for marker in ("terminando la", "terminando el", "tercera dungeon", "tercera mazmorra", "segunda dungeon", "segunda mazmorra")):
            return "progress_update", None
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

    def _log_stt_non_command_decision(self, command: str, decision: str, *, reason: str | None = None) -> None:
        print(
            "[HEBE][COG] incoming "
            f"source='stt_voice' command={command!r} "
            f"current_pending={getattr(self.runtime.state, 'pending_clarification', None)!r}",
            flush=True,
        )
        suffix = f" reason={reason}" if reason else ""
        print(f"[HEBE][COG] decision={decision}{suffix}", flush=True)

    def _record_voice_event(self, text: str, event_type: str, mood_hint: str | None) -> ContextRelevance:
        stream = self._get_stream_state()
        if not stream:
            relevance = ContextRelevance(useful=False, category="none", confidence=0.0, reason="no_stream_state")
            self._log_context_relevance(relevance)
            return relevance
        stream.last_voice_event = event_type
        stream.last_voice_event_ts = time.time()
        stream.last_voice_summary = self._summarize_voice_event(text, event_type)
        if mood_hint:
            stream.leo_mood_hint = mood_hint
        self._apply_ambient_voice_to_run_context(stream, text, event_type)
        relevance = self._extract_and_store_ambient_context(stream, text, event_type)
        try:
            self._get_live_session_brain().update_from_voice_relevance(
                text,
                event_type,
                relevance,
                facts=list(getattr(relevance, "facts", []) or []),
            )
        except Exception as exc:
            print(f"[HEBE][LIVE_SESSION] voice context update failed: {exc!r}", flush=True)
        return relevance

    def _extract_and_store_ambient_context(self, stream, text: str, event_type: str) -> ContextRelevance:
        extractor = getattr(self, "ambient_context_extractor", None)
        if extractor is None:
            extractor = AmbientContextExtractor()
            self.ambient_context_extractor = extractor
        extraction = extractor.extract(text, event_type=event_type)
        if not extraction.useful:
            stream.last_ambient_context_ignored_reason = extraction.reason
            stream.last_ambient_context_ignored_ts = time.time()
            relevance = ContextRelevance(useful=False, category="none", confidence=0.0, reason=extraction.reason)
            self._log_context_relevance(relevance)
            print(f"[HEBE][AMBIENT_CONTEXT] ignored reason={extraction.reason}", flush=True)
            return relevance
        filtered_facts = filter_ambient_facts_for_activity(stream, list(extraction.facts))
        if not filtered_facts:
            stream.last_ambient_context_ignored_reason = "owner_confirmed_activity"
            stream.last_ambient_context_ignored_ts = time.time()
            relevance = ContextRelevance(useful=False, category="none", confidence=0.0, reason="owner_confirmed_activity")
            self._log_context_relevance(relevance)
            print("[HEBE][AMBIENT_CONTEXT] ignored reason=owner_confirmed_activity", flush=True)
            return relevance
        extraction = type(extraction)(
            useful=extraction.useful,
            facts=filtered_facts,
            mood=extraction.mood,
            reason=extraction.reason,
        )
        facts = list(getattr(stream, "recent_run_context_facts", []) or [])
        now = time.time()
        facts = [
            fact for fact in facts
            if float(fact.get("expires_at", 0.0) or 0.0) > now
        ]
        facts.extend(extraction.facts)
        stream.recent_run_context_facts = facts[-20:]
        stream.run_context_updated_ts = now
        stream.run_context_source = "stt_voice"
        if extraction.mood:
            stream.leo_mood_hint = extraction.mood
        self._apply_extracted_facts_to_stream(stream, extraction.facts)
        top_fact = max(extraction.facts, key=lambda fact: float(fact.get("confidence", 0.0) or 0.0), default={})
        relevance = ContextRelevance(
            useful=True,
            category=str(top_fact.get("category") or top_fact.get("kind") or "ambient_note"),
            confidence=float(top_fact.get("confidence", 0.0) or 0.0),
            reason=extraction.reason,
            facts=list(extraction.facts),
        )
        self._log_context_relevance(relevance)
        print(
            "[HEBE][SESSION_CONTEXT] "
            f"updated=true category={relevance.category} source=leo_stt confidence={relevance.confidence:.2f}",
            flush=True,
        )
        for fact in extraction.facts:
            category = fact.get("category") or fact.get("kind") or "unknown"
            summary = fact.get("summary") or fact.get("text") or ""
            confidence = float(fact.get("confidence", 0.0) or 0.0)
            print(
                f"[HEBE][AMBIENT_CONTEXT] extracted category={category} "
                f"summary={summary!r} confidence={confidence:.2f}",
                flush=True,
            )
            print(f"[HEBE][RUN_CONTEXT] updated source=stt_voice category={category}", flush=True)
        return relevance

    def _log_context_relevance(self, relevance: ContextRelevance) -> None:
        print(
            "[HEBE][CONTEXT_RELEVANCE] "
            f"useful={str(bool(relevance.useful)).lower()} category={relevance.category} "
            f"confidence={float(relevance.confidence):.2f} reason={relevance.reason}",
            flush=True,
        )

    def _apply_extracted_facts_to_stream(self, stream, facts: list[dict]) -> None:
        facts = filter_ambient_facts_for_activity(stream, facts)
        for fact in facts:
            kind = str(fact.get("kind") or "")
            text = str(fact.get("text") or "").strip()
            if not text:
                continue
            if kind == "objective":
                stream.current_run_objective = text[:120]
            elif kind == "location":
                stream.current_run_location = text[:80]
            elif kind in {
                "level_gap",
                "phase",
                "ambient_note",
                "game_relation",
                "healing_item_effectiveness",
                "healing_or_recovery",
                "unexpected_attack",
                "guide_strategy",
                "enemy_mechanic",
                "low_hp",
                "combat_risk",
                "rng_dependency",
                "challenge_constraint",
                "failure_or_death",
                "resource_management",
                "boss_or_area_difficulty",
                "navigation_confusion",
                "progress_marker",
                "repeated_failure",
            }:
                stream.current_run_phase = text[:160]

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

    def _handle_live_session_manual_command(self, raw_command: str, normalized: str, stream) -> str | None:
        brain = self._get_live_session_brain()
        if normalized in {"estado de sesion", "estado de sesiÃ³n", "estado sesion", "estado sesiÃ³n"}:
            return self._format_live_session_state_reply(brain.as_debug_dict())

        if normalized in {"que recuerdas de este directo", "quÃ© recuerdas de este directo", "que recuerdas del directo", "memoria de este directo"}:
            context = brain.retrieve_context("current stream memory", limit_events=18, limit_summaries=5)
            return self._format_live_session_memory_reply(context)

        if normalized in {"que ha dicho el chat", "quÃ© ha dicho el chat", "que dice el chat", "resumen del chat"}:
            context = brain.retrieve_context("recent chat topics", limit_events=20, limit_summaries=3)
            return self._format_live_session_chat_reply(context)

        correction_match = re.match(r"^corrige contexto[:\s]+(.+)$", raw_command.strip(), flags=re.IGNORECASE)
        if correction_match:
            correction = correction_match.group(1).strip()
            brain.apply_correction(correction, self._normalize_text(correction))
            return "Contexto corregido. No vuelvo a tirar de ese ancla como si siguiera viva."

        progress_match = re.match(r"^apunta avance[:\s]+(.+)$", raw_command.strip(), flags=re.IGNORECASE)
        if progress_match:
            progress = progress_match.group(1).strip()
            if progress:
                if progress not in getattr(stream, "completed_run_markers", []):
                    self._add_completed_marker(stream, progress)
                brain.update_from_voice_relevance(progress, "progress_update", ContextRelevance(useful=True, category="progress_marker", confidence=0.9, reason="manual_progress"))
                return f"Avance apuntado: {progress}."

        if normalized in {"olvida ese ancla", "olvida el ancla", "invalida ese ancla"}:
            brain.invalidate_anchor(reason="manual_forget_anchor")
            return "Ancla invalidada. No la reutilizo."

        return None

    def _format_live_session_state_reply(self, debug: dict) -> str:
        live = debug.get("live_session") or {}
        meta = debug.get("stream_metadata") or {}
        return (
            "Estado de sesiÃ³n:\n\n"
            f"* Stream: {meta.get('stream_status') or meta.get('live_status') or 'desconocido'}.\n"
            f"* Juego/categoria: {meta.get('game') or meta.get('category') or 'sin detectar'}.\n"
            f"* Titulo: {meta.get('title') or 'sin titulo'}.\n"
            f"* Fase: {live.get('current_phase') or 'sin fase clara'}.\n"
            f"* Objetivo: {live.get('current_objective') or 'sin objetivo claro'}.\n"
            f"* Ubicacion/actividad: {live.get('current_location') or 'sin ubicacion clara'}.\n"
            f"* Ultima correccion: {live.get('latest_correction_from_leo') or 'ninguna'}.\n"
            f"* Tema de chat: {live.get('current_chat_topic') or 'sin tema reciente'}.\n"
            f"* Ultimo mensaje mio: {(live.get('last_hebe_utterance') or {}).get('text') or 'ninguno'}.\n"
            f"* Actualizado: {live.get('last_updated_at') or 'nunca'}."
        )

    def _format_live_session_memory_reply(self, context: dict) -> str:
        live = context.get("live_state") or {}
        events = context.get("recent_events") or []
        summaries = context.get("rolling_summaries") or []
        important = [
            f"{item.get('event_type')}: {item.get('raw_text') or item.get('topic') or ''}".strip()
            for item in events[-8:]
            if item.get("event_type") in {"leo_direct_to_hebe", "leo_reply_to_hebe", "correction", "session_context_update", "twitch_chat_mention"}
        ]
        return (
            "De este directo tengo esto:\n\n"
            f"* Juego/categoria: {live.get('current_game') or live.get('current_category') or 'sin detectar'}.\n"
            f"* Progreso/fase: {live.get('current_phase') or 'sin fase clara'}.\n"
            f"* Objetivo: {live.get('current_objective') or 'sin objetivo claro'}.\n"
            f"* Chatters recientes: {', '.join((item.get('display_name') or item.get('username') or '') for item in (live.get('recent_chatters') or [])[:8]) or 'ninguno'}.\n"
            f"* Eventos importantes: {' | '.join(important) if important else 'sin eventos destacados'}.\n"
            f"* Ultimo resumen: {(summaries[0] or {}).get('summary_text') if summaries else 'aun no he generado resumen rolling'}."
        )

    def _format_live_session_chat_reply(self, context: dict) -> str:
        live = context.get("live_state") or {}
        chatters = live.get("recent_chatters") or []
        names = [item.get("display_name") or item.get("username") for item in chatters[:10] if item.get("display_name") or item.get("username")]
        topics = []
        for item in chatters:
            for topic in item.get("recent_topics") or []:
                if topic and topic not in topics:
                    topics.append(topic)
        return (
            "Chat reciente:\n\n"
            f"* Tema actual: {live.get('current_chat_topic') or 'sin tema claro'}.\n"
            f"* Participantes recientes: {', '.join(names) if names else 'nadie registrado'}.\n"
            f"* Temas: {', '.join(topics[:8]) if topics else 'sin temas clasificados'}."
        )

    def _handle_stream_manual_command(self, text: str, *, cognitive_decision=None, source: str | None = None) -> str | CommandResult | None:
        if not self._manual_handler_guard(
            handler="stream", cognitive_decision=cognitive_decision,
            capabilities={"stream.local_state_control", "twitch_action"},
            source=source, require_live=True,
        ):
            return None
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

        alias_result = self._handle_chatter_alias_command(raw_command, normalized)
        if alias_result is not None:
            return alias_result

        preview_result = self._handle_shoutout_preview_intent(raw_command, normalized)
        if preview_result is not None:
            return preview_result

        invalidation_result = self._handle_game_fact_invalidation(raw_command, normalized, stream)
        if invalidation_result is not None:
            return invalidation_result

        knowledge_result = self._handle_game_knowledge_query(raw_command, normalized, stream)
        if knowledge_result is not None:
            return knowledge_result

        action_result = self._plan_and_execute_stream_action(raw_command, normalized, stream)
        if action_result is not None:
            return action_result

        primer_result = self._handle_stream_session_primer_command(raw_command, normalized, stream, command_result)
        if primer_result is not None:
            return primer_result

        run_reply = self._handle_run_context_command(raw_command, normalized, stream)
        if run_reply is not None:
            return run_reply

        live_reply = self._handle_live_session_manual_command(raw_command, normalized, stream)
        if live_reply is not None:
            return live_reply

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
        if chatter_match and "este juego" not in normalized and not self._looks_like_game_knowledge_target(chatter_match.group(1)):
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
                "_simulated": True,
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

    def _handle_game_knowledge_query(self, raw_command: str, normalized: str, stream) -> CommandResult | None:
        match = re.match(
            r"^(?:que sabes de|quÃ© sabes de|que sabes sobre|quÃ© sabes sobre|dime que sabes de|dime quÃ© sabes de)\s+(.+)$",
            normalized,
        )
        if not match:
            return None
        target = match.group(1).strip()
        if target and not self._looks_like_game_knowledge_target(target):
            return None
        raw_target = re.sub(
            r"^\s*(?:que sabes de|quÃ© sabes de|que sabes sobre|quÃ© sabes sobre|dime que sabes de|dime quÃ© sabes de)\s+",
            "",
            str(raw_command or ""),
            flags=re.IGNORECASE,
        ).strip()
        resolver = getattr(self, "game_knowledge", None)
        if resolver is None:
            resolver = GameKnowledgeResolver(
                profile_store=getattr(self, "game_profiles", None),
                research_service=getattr(self, "game_research", None),
                config=GameKnowledgeConfig.from_env(),
            )
            self.game_knowledge = resolver
        result = resolver.resolve(game=raw_target or target, stream=stream)
        print(
            "[HEBE][GAME_KNOWLEDGE] "
            f"game={result.game_title!r} mode={result.response_mode} "
            f"profile_source={result.profile_source} web_reason={result.web_lookup_reason}",
            flush=True,
        )
        event = getattr(self, "_current_input_event", None)
        if event is not None:
            metadata = getattr(event, "stt_metadata", None)
            if isinstance(metadata, dict):
                metadata["block_memory_extraction"] = True
                metadata["block_memory_extraction_reason"] = "game_knowledge_query"
        return CommandResult(
            action_type="game_knowledge_query",
            success=True,
            user_visible_summary=result.fallback_text,
            state_changes=result.to_state_changes(),
            constraints=[
                "Answer in Spanish.",
                "Separate personal stream memory from public spoiler-safe game knowledge.",
                "Do not invent session progress, locations, characters, bosses, or future story facts.",
                "Mention when personal run memory is missing.",
                "Do not ask for clarification.",
            ],
            suggested_tone="concise Hebe game-knowledge answer",
            fallback_text=result.fallback_text,
            requires_model_response=True,
            metadata={
                "message_goal": (
                    f"Answer what Hebe knows about {result.game_title}. "
                    f"Use response_mode={result.response_mode}; include spoiler-safe public profile if present, "
                    "and clearly say when personal stream memory is missing."
                ),
                "game_knowledge": result.to_state_changes(),
            },
        )

    def _handle_game_fact_invalidation(self, raw_command: str, normalized: str, stream) -> CommandResult | None:
        match = re.match(
            r"^no\s+(?:hay|existe)\s+(?:ningun|ningÃºn|ninguna|un|una)?\s*([a-z0-9][a-z0-9'\- ]{1,40}?)\s+en\s+(.+)$",
            normalized,
        )
        if not match:
            return None
        term = self._title_case_game(match.group(1).strip())
        game = self._title_case_game(self._strip_stream_primer_filler(match.group(2).strip()))
        if game in {"Este Juego", "El Juego"} and stream is not None:
            game = getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or game
        changed = session_primer.invalidate_game_session_term(game, term, source="user_correction")
        event = getattr(self, "_current_input_event", None)
        if event is not None and isinstance(getattr(event, "stt_metadata", None), dict):
            event.stt_metadata["block_memory_extraction"] = True
            event.stt_metadata["block_memory_extraction_reason"] = "user_correction_invalidated_wrong_fact"
        return CommandResult(
            action_type="game_fact_invalidated",
            success=True,
            user_visible_summary=f"Invalidated stored references to {term} for {game}.",
            state_changes={"game": game, "term": term, "invalidated_rows": changed},
            constraints=[
                "Answer in Spanish.",
                "Confirm the correction without claiming any new game facts.",
                "Do not ask for clarification.",
            ],
            suggested_tone="short Hebe correction acknowledgement",
            fallback_text=(
                f"Vale, borro esa pista: {term} no queda como referencia de {game}."
                if changed
                else f"Vale, tomo la correccion: no tratare {term} como referencia de {game}."
            ),
            requires_model_response=True,
            metadata={"message_goal": f"Confirm Leo's correction that {term} is not a valid stored fact for {game}."},
        )

    def _looks_like_game_knowledge_target(self, target: str | None) -> bool:
        text = str(target or "").strip()
        if not text:
            return True
        if self._normalize_text(text) in {"este juego", "el juego"}:
            return True
        store = getattr(self, "game_profiles", None)
        if store is not None:
            try:
                return store.has_specific_profile(current_category=text, current_game=text, current_title=text)
            except Exception:
                pass
        words = set(self._normalize_text(text).split())
        game_terms = {
            "game", "juego", "persona", "royal", "final", "fantasy", "ff9", "ffix",
            "baldur", "gate", "retro", "elden", "souls", "zelda", "mario", "metroid",
            "yakuza", "dragon", "quest",
        }
        if words & game_terms:
            return True
        stream = self._get_stream_state()
        if stream is not None:
            current = self._normalize_text(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or "")
            return bool(current and self._normalize_text(text) == current)
        return False

    def _handle_chatter_alias_command(self, raw_command: str, normalized: str) -> CommandResult | None:
        match = re.match(r"^(?:recuerda que\s+)?(.+?)\s+es\s+(@?[A-Za-z0-9_]{3,25})$", str(raw_command or "").strip(), flags=re.IGNORECASE)
        if not match:
            match = re.match(r"^(?:recuerda que\s+)?(.+?)\s+es\s+(@?[A-Za-z0-9_]{3,25})$", str(normalized or "").strip(), flags=re.IGNORECASE)
        if not match:
            return None
        alias = match.group(1).strip()
        username = match.group(2).strip().lstrip("@")
        if not alias or not username:
            return None
        twitch = getattr(self.runtime, "twitch", None)
        remember = getattr(twitch, "remember_user_alias", None)
        ok = bool(remember(alias, username)) if callable(remember) else False
        if not ok:
            resolver = getattr(twitch, "target_resolver", None)
            remember_resolver = getattr(resolver, "remember_alias", None)
            ok = bool(remember_resolver(alias, username)) if callable(remember_resolver) else False
        print(f"[HEBE][CHATTER] alias stored alias={alias!r} username={username!r} success={ok}", flush=True)
        return CommandResult(
            action_type="chatter_alias_stored",
            success=ok,
            user_visible_summary=f"Alias {alias} stored for {username}." if ok else "Could not store chatter alias.",
            state_changes={"alias": alias, "username": username},
            constraints=["Do not ask for clarification.", "Do not claim a shoutout was sent."],
            suggested_tone="short Hebe stream-control reply",
            fallback_text=(f"Vale, recordaré que {alias} es {username}." if ok else "No he podido guardar ese alias."),
            requires_model_response=ok,
            metadata={"message_goal": f"Confirm that alias {alias} now resolves to {username}."},
        )

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
        if plan.action_type == "stream_chat_message":
            return self._execute_stream_chat_message_plan(plan)
        if plan.action_type in {"stream_ambient_stt_enabled", "stream_ambient_stt_disabled"}:
            return self._execute_stream_ambient_stt_plan(plan)
        return None

    def _execute_stream_chat_message_plan(self, plan: ActionPlan) -> CommandResult:
        message = str((plan.slots or {}).get("message") or "").strip()
        if plan.status != "complete" or not message:
            return CommandResult(
                action_type="stream_chat_message_clarify",
                success=False,
                user_visible_summary="Missing Twitch chat message.",
                state_changes={},
                constraints=["Ask one concise follow-up question.", "Do not claim a Twitch message was sent."],
                fallback_text="¿Qué digo en el chat?",
                requires_model_response=False,
                metadata={"action_plan": plan.as_log_dict(), "message_goal": "Ask Leo what message should be sent to Twitch chat."},
            )
        twitch = getattr(self.runtime, "twitch", None)
        send = getattr(twitch, "send_message", None)
        ok = False
        if callable(send):
            try:
                ok = bool(send(message))
            except Exception as exc:
                print(f"[HEBE][ACTION_EXECUTOR] action_type=stream_chat_message success=false error={exc!r}", flush=True)
        self._declare_output_route(
            input_type="direct_stt",
            targets=[OUTPUT_TARGET_TWITCH_CHAT],
            reason="action_plan_twitch_chat_message",
        )
        if ok:
            print("[HEBE][ACTION_EXECUTOR] success=true action_type=stream_chat_message", flush=True)
            return CommandResult(
                action_type="stream_chat_message",
                success=True,
                user_visible_summary="Twitch chat message sent.",
                state_changes={"action_type": "stream_chat_message", "target": "twitch_chat", "message": message},
                constraints=["Do not ask for clarification.", "Do not add extra content beyond confirming the chat message was sent."],
                suggested_tone="short Hebe stream-control reply",
                fallback_text="Enviado al chat.",
                requires_model_response=True,
                metadata={"action_plan": plan.as_log_dict(), "message_goal": "Confirm to Leo that the Twitch chat message was sent."},
            )
        return CommandResult(
            action_type="stream_chat_message",
            success=False,
            user_visible_summary="Twitch chat message failed.",
            state_changes={"action_type": "stream_chat_message", "target": "twitch_chat", "message": message},
            constraints=["Do not claim the chat message was sent."],
            fallback_text="No he podido escribirlo en el chat.",
            requires_model_response=False,
            metadata={"action_plan": plan.as_log_dict(), "message_goal": "Tell Leo the Twitch chat message could not be sent."},
        )

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

    def _handle_shoutout_preview_intent(self, raw_command: str, normalized: str) -> str | None:
        tokens = set(str(normalized or "").split())
        if not (tokens & {"previsualiza", "preview"}):
            return None

        planner = self._get_stream_action_planner()
        plan = planner.plan(InputEvent(
            source="typed_ui",
            raw_text=raw_command,
            normalized_text=raw_command,
        ))
        if plan is None or plan.action_type != "twitch_shoutout":
            return None
        self._manual_reply_ui_only = True
        if plan.status == "needs_confirmation":
            return "¿A quién le hago el SO, Leo?"
        return f"Previsualizacion de shoutout: {plan.command or self._build_shoutout_command_preview(plan.target or '')}"

    def _handle_shoutout_manual_command(self, raw_command: str, normalized: str, stream) -> str | CommandResult | None:
        return None

    def _build_shoutout_command_preview(self, target: str) -> str:
        twitch = getattr(self.runtime, "twitch", None)
        build = getattr(twitch, "build_shoutout_command", None)
        normalized = self._normalize_shoutout_target(target)
        if callable(build):
            return build(normalized)
        template = os.getenv("HEBE_SHOUTOUT_COMMAND_TEMPLATE", "!so {username}") or "!so {username}"
        return template.format(username=normalized)

    def _handle_stream_session_primer_command(self, raw_command: str, normalized: str, stream, command_result) -> CommandResult | None:
        now_dt = datetime.now(ZoneInfo(os.getenv("HEBE_STREAM_TIMEZONE", "Europe/Madrid")))
        today_weekday = session_primer.weekday_key_for(now_dt)

        schedule_query = normalized in {
            "que toca hoy", "qué toca hoy", "que hay hoy", "qué hay hoy", "stream de hoy",
        }
        prepare_today = normalized in {
            "prepara el stream de hoy", "prepara stream de hoy", "preparar stream",
            "preparar hoy", "prepara hoy",
        }
        title_today = normalized in {
            "sugiere titulo para hoy", "sugiere título para hoy", "titulo para hoy",
            "título para hoy", "dame titulo para hoy", "dame título para hoy",
            "sugiere otro titulo", "sugiere otro título",
        }

        prepare_match = re.match(r"^prepara\s+(.+)$", normalized)
        explicit_prepare_game = ""
        if prepare_match and not prepare_today and "stream" not in normalized:
            explicit_prepare_game = self._strip_stream_primer_filler(prepare_match.group(1))

        title_match = re.match(r"^(?:dame\s+)?(?:\d+\s+)?(?:titulos?|títulos?)\s+para\s+(.+)$", normalized)
        explicit_title_game = self._strip_stream_primer_filler(title_match.group(1)) if title_match else ""

        schedule_game_match = re.match(r"^(?:no\s+)?(?:hoy toca|cambia el juego de hoy a)\s+(.+?)(?:\s+(?:hebe|ebe|eve|jebe))?$", normalized)
        if schedule_game_match:
            game = self._title_case_game(self._strip_stream_primer_filler(schedule_game_match.group(1)))
            schedule = session_primer.update_schedule_for_weekday(today_weekday, game)
            primer = session_primer.build_stream_session_primer(game=game, dt=now_dt)
            session_primer.apply_primer_to_stream(stream, primer)
            self._mark_today_game_override(stream, game)
            return command_result(
                "set_today_stream_game",
                self._format_stream_schedule_reply(primer),
                state_changes={
                    "schedule": schedule,
                    "primer": primer.to_dict(),
                    "current_game": game,
                    "source": "stt_voice",
                    "confidence": "high",
                },
                message_goal="Tell Leo today's scheduled stream game was updated and summarize the prepared primer.",
            )

        title_set_match = re.match(r"^(?:el titulo de hoy sera|el título de hoy será|titulo de hoy sera|título de hoy será)\s+(.+)$", normalized)
        if title_set_match:
            title = str(raw_command or "").strip()
            title = re.sub(
                r"^\s*(?:el titulo de hoy sera|el título de hoy será|titulo de hoy sera|título de hoy será)\s+",
                "",
                title,
                flags=re.IGNORECASE,
            ).strip()
            stream.current_stream_title = title
            stream.title_context_markers = [title]
            stream.title_context_updated_ts = time.time()
            return command_result(
                "stream_title_saved",
                f"Guardado como título de hoy: {title}",
                state_changes={"stream_title": title},
                message_goal="Confirm today's stream title was saved.",
            )

        note_result = self._handle_game_session_note_command(raw_command, normalized, now_dt)
        if note_result is not None:
            primer = session_primer.build_stream_session_primer(game=note_result.get("game"), dt=now_dt)
            session_primer.apply_primer_to_stream(stream, primer)
            return command_result(
                "game_session_note_saved",
                self._format_stream_primer_reply(primer),
                state_changes={"game_session": note_result, "primer": primer.to_dict()},
                message_goal="Confirm the game session note was saved and summarize the next stream primer.",
            )

        if schedule_query:
            primer = session_primer.build_stream_session_primer(dt=now_dt)
            return command_result(
                "stream_schedule_lookup",
                self._format_stream_schedule_reply(primer),
                state_changes={"primer": primer.to_dict()},
                message_goal="Answer what is scheduled today using the stream schedule.",
            )

        if prepare_today or explicit_prepare_game:
            primer = session_primer.build_stream_session_primer(
                game=self._title_case_game(explicit_prepare_game) if explicit_prepare_game else None,
                dt=now_dt,
            )
            session_primer.apply_primer_to_stream(stream, primer)
            return command_result(
                "stream_session_primer_created",
                self._format_stream_primer_reply(primer),
                state_changes={"primer": primer.to_dict()},
                message_goal="Summarize today's stream session primer, including schedule, last session, starting point, safe context, title suggestion, and missing info.",
            )

        if title_today or explicit_title_game:
            primer = session_primer.build_stream_session_primer(
                game=self._title_case_game(explicit_title_game) if explicit_title_game else None,
                dt=now_dt,
            )
            return command_result(
                "stream_title_suggestions",
                self._format_title_suggestions_reply(primer),
                state_changes={"primer": primer.to_dict(), "title_suggestions": primer.title_suggestions},
                message_goal="Suggest English stream titles in Leo's standard [ENG/ESP] format.",
            )

        if normalized in {"no uses spoilers", "sin spoilers", "no spoilers"}:
            setattr(stream, "spoiler_policy", "no_spoilers")
            return command_result(
                "stream_spoiler_policy_updated",
                "Vale, hoy sin spoilers. Usaré solo contexto guardado y confirmado.",
                state_changes={"spoiler_policy": "no_spoilers"},
                message_goal="Confirm no-spoiler policy is active for stream context.",
            )

        return None

    def _strip_stream_primer_filler(self, value: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"^(?:el|la|los|las|juego|stream)\s+", "", text).strip()
        return text

    def _title_case_game(self, value: str) -> str:
        raw = str(value or "").strip()
        key = re.sub(r"[^a-z0-9]+", " ", raw.casefold()).strip()
        known = {
            "persona 5": "Persona 5 Royal",
            "persona 5 royal": "Persona 5 Royal",
            "p5r": "Persona 5 Royal",
            "final fantasy ix": "FINAL FANTASY IX",
            "ff9": "FINAL FANTASY IX",
            "baldur s gate 3": "Baldur's Gate 3",
            "baldurs gate 3": "Baldur's Gate 3",
        }
        return known.get(key, raw.title())

    def _handle_game_session_note_command(self, raw_command: str, normalized: str, now_dt: datetime) -> dict | None:
        game = ""
        text = ""
        start_summary = ""
        end_summary = ""
        next_time_plan = ""
        current_location = ""
        current_objective = ""

        match = re.match(r"^guarda que en\s+(.+?)\s+terminamos\s+(.+)$", normalized)
        if match:
            game = self._title_case_game(match.group(1))
            text = match.group(2).strip()
            end_summary = text
            current_location = text
        if not match:
            match = re.match(r"^(?:proxima|próxima) vez en\s+(.+?)\s+toca\s+(.+)$", normalized)
            if match:
                game = self._title_case_game(match.group(1))
                text = match.group(2).strip()
                next_time_plan = text
                current_objective = text
        if not match:
            generic = re.match(r"^guarda que empezamos\s+(.+)$", normalized)
            if generic:
                stream = self._get_stream_state()
                game = getattr(stream, "current_game", None) or (session_primer.get_schedule_for_date(now_dt) or {}).get("game") or ""
                text = generic.group(1).strip()
                start_summary = text
                current_location = text
            else:
                generic = re.match(r"^guarda que terminamos\s+(.+)$", normalized)
                if generic:
                    stream = self._get_stream_state()
                    game = getattr(stream, "current_game", None) or (session_primer.get_schedule_for_date(now_dt) or {}).get("game") or ""
                    text = generic.group(1).strip()
                    end_summary = text
                    current_location = text
        if not game or not text:
            return None
        return session_primer.save_game_session_note(
            game,
            stream_date=now_dt.date().isoformat(),
            start_summary=start_summary,
            end_summary=end_summary,
            current_location=current_location,
            current_objective=current_objective,
            next_time_plan=next_time_plan,
            spoiler_policy="no_spoilers",
            source="manual_command",
        )

    def _format_stream_schedule_reply(self, primer: session_primer.StreamSessionPrimer) -> str:
        return (
            f"Hoy ({primer.weekday}, {primer.local_now[:10]}) toca {primer.game}"
            + (f" — {primer.playthrough_type}" if primer.playthrough_type else "")
            + (f". Slot: {primer.slot_name}." if primer.slot_name else ".")
        )

    def _format_title_suggestions_reply(self, primer: session_primer.StreamSessionPrimer) -> str:
        lines = [f"Títulos sugeridos para {primer.game}:"]
        lines.extend(f"* {title}" for title in primer.title_suggestions)
        return "\n".join(lines)

    def _format_stream_primer_reply(self, primer: session_primer.StreamSessionPrimer) -> str:
        missing = ", ".join(primer.missing_info) if primer.missing_info else "ninguna"
        title = primer.title_suggestions[0] if primer.title_suggestions else "sin título sugerido"
        return (
            f"Primer de stream para {primer.game} ({primer.playthrough_type}).\n"
            f"* Hoy: {primer.weekday}, {primer.local_now[:10]} ({primer.timezone})\n"
            f"* Slot: {primer.slot_name or 'sin slot'}\n"
            f"* Última sesión: {primer.last_session_summary or 'memoria incompleta'}\n"
            f"* Inicio/objetivo: {primer.starting_point or primer.likely_objective or 'por confirmar'}\n"
            f"* Contexto seguro: {'; '.join(primer.safe_context_for_spontaneity) or 'por confirmar'}\n"
            f"* Título: {title}\n"
            f"* Falta: {missing}"
        )

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

    def _handle_tts_manual_command(self, text: str, *, cognitive_decision=None, source: str | None = None) -> str | CommandResult | None:
        if not self._manual_handler_guard(
            handler="tts", cognitive_decision=cognitive_decision,
            capabilities={"audio.tts_control"}, source=source,
        ):
            return None
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

        stream = self._get_stream_state()
        policies = getattr(stream, "policies", None) if stream else None
        if policies is None:
            return None
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

    def _handle_priority_tts_command(self, text: str) -> CommandResult | str | None:
        normalized = self._normalize_voice_command_text(text)
        intent = self._parse_tts_control_intent(normalized)
        if intent == "global_on":
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
        if intent == "global_off":
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
        if intent in {"stream_on", "stream_off"}:
            stream = self._get_stream_state()
            policies = getattr(stream, "policies", None) if stream else None
            if policies is None:
                return None
            enabled = intent == "stream_on"
            policies.allow_tts_replies = enabled
            stream.stream_output_mode = "tts_enabled" if enabled else "twitch_chat_only"
            print(f"[HEBE][TTS] stream enabled={str(enabled).lower()} source=command", flush=True)
            print(
                f"[HEBE][OUTPUT_MODE] mode={stream.stream_output_mode} reason=user_setting",
                flush=True,
            )
            self._emit_audio_status()
            return CommandResult(
                action_type="stream_tts_enabled" if enabled else "stream_tts_disabled",
                success=True,
                user_visible_summary=(
                    "Stream TTS replies enabled by policy."
                    if enabled
                    else "Stream TTS replies disabled; Twitch replies remain text chat only."
                ),
                state_changes={"stream_tts": enabled, "stream_output_mode": stream.stream_output_mode},
                constraints=["Do not ask for clarification.", "Do not claim idle spontaneous TTS changed."],
                fallback_text=(
                    "Vale. Si toca, también hablaré en stream."
                    if enabled
                    else "Entendido. En stream responderé solo por chat."
                ),
                requires_model_response=True,
                metadata={
                    "message_goal": (
                        "Tell Leo that stream voice replies are enabled when policy allows."
                        if enabled
                        else "Tell Leo that stream voice replies are disabled and stream replies remain text-only."
                    )
                },
            )
        if intent == "status":
            return self._build_voice_status_reply()
        return None

    def _parse_tts_control_intent(self, normalized: str) -> str | None:
        tokens = set(str(normalized or "").split())
        if not tokens:
            return None
        voice = {"voz", "tts", "voice", "hablar"}
        enable = {"activa", "enciende", "enable", "on", "habla", "usa", "pon", "quiero", "puedes", "vuelve"}
        disable = {"desactiva", "apaga", "disable", "off", "callate", "sin", "solo", "silencia", "text", "texto"}
        stream = {"stream", "directo", "chat"}
        status = {"estado", "status"}
        if tokens & status and (tokens & voice or "tts" in tokens):
            return "status"
        if tokens & stream and tokens & disable:
            return "stream_off"
        if tokens & stream and (tokens & enable or "hablar" in tokens):
            return "stream_on"
        if tokens & disable and (tokens & voice or "texto" in tokens or "text" in tokens):
            return "global_off"
        if tokens & enable and (tokens & voice or "escucharte" in tokens):
            return "global_on"
        return None

    def _handle_pending_manual_intent(self, text: str, *, cognitive_decision=None, source: str | None = None) -> CommandResult | str | None:
        normalized = self._normalize_voice_command_text(text)
        capability = "pending.cancel" if self._is_cancel_pending_reminder(normalized) else "audio.tts_control"
        if not self._manual_handler_guard(
            handler="pending", cognitive_decision=cognitive_decision,
            capabilities={capability}, source=source,
        ):
            return None
        if self._is_cancel_pending_reminder(normalized):
            if getattr(self.runtime.state, "pending_clarification", None) or getattr(self.runtime.state, "pending_reminder", None):
                self.runtime.state.pending_clarification = None
                self.runtime.state.pending_reminder = None
                print("[HEBE][INTENT] reminder pending cancelled", flush=True)
                return CommandResult(
                    action_type="pending_reminder_cancelled",
                    success=True,
                    user_visible_summary="Pending reminder or appointment clarification was cancelled.",
                    state_changes={"pending_clarification": None, "pending_reminder": None},
                    constraints=["Do not ask for clarification."],
                    fallback_text="Vale, no guardo nada.",
                    requires_model_response=True,
                    metadata={"message_goal": "Tell Leo the pending reminder clarification was cancelled."},
                )

        if not getattr(self.runtime.state, "pending_tts_scope", None):
            return None

        pending_tts = getattr(self.runtime.state, "pending_tts_scope", {}) or {}
        print("[HEBE][PENDING] active=pending_tts_scope", flush=True)
        print("[HEBE][INTENT] pending_tts_scope active", flush=True)
        if self._is_explicit_command_while_pending(normalized):
            self.runtime.state.pending_tts_scope = None
            print("[HEBE][PENDING] new explicit command detected; clearing pending_tts_scope", flush=True)
            print("[HEBE][INTENT] cleared pending_tts_scope", flush=True)
            return None

        scope = self._parse_tts_scope_followup(normalized)
        if scope == "local":
            return self._resolve_pending_tts_scope_local()
        if scope == "stream":
            return self._resolve_pending_tts_scope_stream()

        if not pending_tts.get("unclear_asked"):
            pending_tts["unclear_asked"] = True
            self.runtime.state.pending_tts_scope = pending_tts
            return CommandResult(
                action_type="tts_scope_clarify",
                success=False,
                user_visible_summary="TTS scope follow-up was unclear; ask whether local or stream.",
                state_changes={"pending_tts_scope": True},
                constraints=["Ask one concise clarification question."],
                fallback_text="No te he entendido, Leo. ¿Local o también para stream?",
                requires_model_response=False,
                metadata={"message_goal": "Ask Leo whether voice should be local only or also for stream."},
            )
        return self._resolve_pending_tts_scope_local(defaulted=True)

    def _is_cancel_pending_reminder(self, normalized: str) -> bool:
        tokens = set(str(normalized or "").split())
        return bool(tokens & {"cancela", "olvida"}) or ({"no", "guardes"}.issubset(tokens))

    def _parse_tts_scope_followup(self, normalized: str) -> str | None:
        tokens = set(str(normalized or "").split())
        if not tokens:
            return None
        stream_tokens = {"stream", "directo"}
        local_tokens = {"local", "aqui", "aquí", "conmigo", "ahora", "escucharte"}
        negation_tokens = {"no", "solo"}
        if tokens & stream_tokens and not (tokens & negation_tokens):
            return "stream"
        if tokens & local_tokens:
            return "local"
        if "solo" in tokens and not (tokens & stream_tokens):
            return "local"
        if tokens & stream_tokens and tokens & negation_tokens:
            return "local"
        return None

    def _resolve_pending_tts_scope_local(self, *, defaulted: bool = False) -> CommandResult:
        stream = self._get_stream_state()
        policies = getattr(stream, "policies", None) if stream else None
        self.runtime.state.tts_enabled = True
        if policies is not None:
            policies.allow_tts_idle_prompts = False
        if stream is not None:
            stream.stream_output_mode = "ui_only"
        self.runtime.state.pending_tts_scope = None
        print("[HEBE][INTENT] resolved pending_tts_scope=local", flush=True)
        print("[HEBE][INTENT] cleared pending_tts_scope", flush=True)
        return CommandResult(
            action_type="tts_scope_resolved",
            success=True,
            user_visible_summary="Voice is enabled locally only; stream remains text-only unless Leo asks otherwise.",
            state_changes={"tts_enabled": True, "stream_idle_tts": False, "stream_output_mode": "ui_only", "pending_tts_scope": False, "defaulted": defaulted},
            constraints=["Do not ask for more clarification.", "Do not claim stream voice is enabled."],
            fallback_text="Perfecto, voz activada solo aquí. En stream seguiré en texto salvo que me digas lo contrario.",
            requires_model_response=True,
            metadata={"scope": "local", "message_goal": "Confirm voice is enabled locally only, and stream remains text-only unless Leo asks otherwise."},
        )

    def _resolve_pending_tts_scope_stream(self) -> CommandResult:
        stream = self._get_stream_state()
        policies = getattr(stream, "policies", None) if stream else None
        self.runtime.state.tts_enabled = True
        if policies is not None:
            policies.allow_tts_replies = True
            policies.allow_tts_event_replies = True
            policies.allow_tts_raid_thanks = True
        if stream is not None:
            stream.stream_output_mode = "tts_enabled"
        self.runtime.state.pending_tts_scope = None
        print("[HEBE][INTENT] resolved pending_tts_scope=stream", flush=True)
        print("[HEBE][INTENT] cleared pending_tts_scope", flush=True)
        return CommandResult(
            action_type="tts_scope_resolved",
            success=True,
            user_visible_summary="Voice is enabled locally and for stream event replies; idle spontaneity remains text-only unless Leo asks otherwise.",
            state_changes={"tts_enabled": True, "stream_replies_tts": True, "stream_event_tts": True, "stream_raid_tts": True, "stream_output_mode": "tts_enabled", "pending_tts_scope": False},
            constraints=["Do not ask for more clarification.", "Do not claim idle spontaneous voice is enabled."],
            fallback_text="Perfecto, voz activada aquí y también para eventos del stream. La espontaneidad idle sigue en texto salvo que me digas lo contrario.",
            requires_model_response=True,
            metadata={"scope": "stream", "message_goal": "Confirm voice is enabled locally and for stream event replies, while idle spontaneity remains text-only."},
        )

    def _is_explicit_command_while_pending(self, normalized: str) -> bool:
        text = str(normalized or "").strip()
        if not text:
            return False

        if self._parse_tts_control_intent(text) is not None:
            return True

        try:
            plan = self._get_stream_action_planner().plan(
                InputEvent(source="typed_ui", raw_text=text, normalized_text=text)
            )
            if plan is not None:
                return True
        except Exception:
            pass

        tokens = set(text.split())
        action_tokens = {
            "activa", "desactiva", "enciende", "apaga", "pausa", "reanuda", "resume",
            "actualiza", "comprueba", "guarda", "finaliza", "haz", "hazle", "dale",
            "manda", "pon", "promociona", "recomienda", "shoutout", "so", "stop",
        }
        object_tokens = {"stt", "tts", "voz", "stream", "contexto", "modo", "promo", "shoutout", "texto"}
        return bool(tokens & action_tokens and tokens & object_tokens)

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
            f"* Stream output mode: {self._stream_output_mode()}\n"
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
                    "stream_output_mode": str(getattr(stream, "stream_output_mode", "tts_enabled") if stream else "tts_enabled"),
                    "stt_enabled": bool(getattr(self.runtime, "stt_enabled", False)),
                    "stt": getattr(stt, "status", "off") if stt is not None else "off",
                    "last_stt_error": getattr(stt, "last_input_device_error", None) if stt is not None else None,
                    "stt_input_device": stt_device,
                    "hebe_sleeping": bool(getattr(self.runtime.state, "hebe_sleeping", False)),
                    "wake_required": bool(getattr(self.runtime.state, "hebe_sleeping", False)),
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

        event = getattr(self, "_current_input_event", None)
        metadata = getattr(event, "stt_metadata", None)
        normalized = self._normalize_text(getattr(event, "normalized_text", "") or "")
        if source == "stt_voice" and (not isinstance(metadata, dict) or not metadata.get("jarvis_allowed")):
            print("[HEBE][MEMORY_EXTRACT] blocked reason=uncertain_stt", flush=True)
            return False
        if isinstance(metadata, dict) and metadata.get("block_memory_extraction"):
            print(
                "[HEBE][MEMORY_EXTRACT] blocked "
                f"reason={metadata.get('block_memory_extraction_reason') or 'model_invented_or_unconfirmed'}",
                flush=True,
            )
            return False
        if re.match(r"^(?:que sabes de|quÃ© sabes de|que sabes sobre|quÃ© sabes sobre)\s+", normalized):
            print("[HEBE][MEMORY_EXTRACT] blocked reason=model_invented_or_unconfirmed", flush=True)
            return False

        reply_step = execution.first_result_of_type("reply") if execution else None
        if not reply_step:
            return False

        if reply_step.data.get("mode") != "chat":
            return False
        if not re.match(r"^(?:recuerda|guarda|acuerdate|acuérdate)\b", normalized):
            print("[HEBE][MEMORY_EXTRACT] blocked reason=no_explicit_memory_request", flush=True)
            return False
        return True

    def _deliver_twitch_reply(self, text: str, *, event_type: str | None = None, payload: dict | None = None) -> None:
        """
        Entrega un reply al chat de Twitch.
        Si las policies del stream lo permiten, también lo hablamos por TTS.
        """
        twitch = getattr(self.runtime, "twitch", None)
        stream = getattr(self.runtime.state, "stream", None)
        is_spontaneous = event_type == "twitch_idle_prompt"
        is_simulated = bool((payload or {}).get("_simulated"))
        if event_type and event_type.startswith("twitch_"):
            raw_text = str((payload or {}).get("message_text") or (payload or {}).get("text") or "")
            username = str((payload or {}).get("user_login") or (payload or {}).get("username") or "")
            source = "twitch_viewer" if event_type == "twitch_chat_react" else "twitch_system"
            firewall = self._input_firewall_decision(
                source=source,
                text=raw_text or text,
                username=username,
                event_type=event_type,
                addressed_to_hebe=self._message_mentions_hebe(raw_text) if raw_text else False,
            )
            if not self._firewall_allows_pipeline(firewall) or firewall.blocks_action(ACTION_TWITCH_REPLY):
                print(
                    f"[HEBE][EVENT][TWITCH] blocked reason={firewall.reason} event_type={event_type}",
                    flush=True,
                )
                return
            if is_simulated:
                print("[HEBE][STREAM_GATE] allowed reason=simulation_mode", flush=True)
        if not is_simulated and not (stream and getattr(stream, "is_live", False)):
            print("[HEBE][STREAM_GATE] blocked reason=offline_stream source=twitch_delivery", flush=True)
            print(f"[HEBE][ACTION_PERMISSIONS] action={ACTION_TWITCH_REPLY} allowed=false reason=offline_stream", flush=True)
            return
        output_mode = self._stream_output_mode()
        tts_allowed = self._stream_tts_output_enabled_for_event(event_type)
        spontaneous_chat_allowed = False
        spontaneous_chat_reason = ""
        if is_spontaneous:
            if output_mode == "ui_only":
                spontaneous_chat_allowed, spontaneous_chat_reason = False, "stream_output_mode_ui_only"
                targets = [OUTPUT_TARGET_LOCAL_UI]
            elif output_mode == "silent":
                spontaneous_chat_allowed, spontaneous_chat_reason = False, "stream_output_mode_silent"
                targets = [OUTPUT_TARGET_SILENT_CONTEXT_UPDATE]
            else:
                spontaneous_chat_allowed, spontaneous_chat_reason = self._spontaneous_twitch_chat_delivery_allowed(text, payload)
                targets = []
            if spontaneous_chat_allowed:
                targets = [OUTPUT_TARGET_TWITCH_CHAT]
                if tts_allowed:
                    targets.append(OUTPUT_TARGET_STREAM_TTS)
            elif not targets:
                if spontaneous_chat_reason == "twitch_spontaneous_disabled":
                    print("[HEBE][SPONTANEITY] skipped reason=twitch_spontaneous_disabled", flush=True)
                else:
                    print(f"[HEBE][SPONTANEITY] skipped reason={spontaneous_chat_reason}", flush=True)
                targets = [OUTPUT_TARGET_STREAM_TTS if tts_allowed else OUTPUT_TARGET_LOCAL_UI]
        else:
            if output_mode == "ui_only" or is_simulated:
                targets = [OUTPUT_TARGET_LOCAL_UI]
            elif output_mode == "silent":
                targets = [OUTPUT_TARGET_SILENT_CONTEXT_UPDATE]
            else:
                targets = [OUTPUT_TARGET_TWITCH_CHAT]
        if not is_spontaneous and tts_allowed and not is_simulated:
            targets.append(OUTPUT_TARGET_STREAM_TTS)
        self._declare_output_route(
            input_type="spontaneity" if is_spontaneous else "twitch_mention_or_event",
            targets=targets,
            event_type=event_type,
            reason=spontaneous_chat_reason or "stream_event_reply",
        )
        if stream is not None:
            stream.last_hebe_stream_speak_ts = time.time()
            if event_type == "twitch_idle_prompt":
                topic = (payload or {}).get("idle_topic")
                service = getattr(self, "stream_spontaneity", None)
                if service is not None:
                    service.record_idle_message(stream, text, topic=topic)
                anchor_id = str((payload or {}).get("used_fact_id") or (payload or {}).get("anchor_id") or topic or "").strip() or None
                try:
                    if anchor_id:
                        self._get_live_session_brain().create_spontaneity_anchor(
                            anchor_id=anchor_id,
                            anchor_type="spontaneity",
                            topic=topic or "unknown",
                            payload=payload or {},
                        )
                except Exception as exc:
                    print(f"[HEBE][LIVE_SESSION] anchor create failed: {exc!r}", flush=True)
        if is_spontaneous:
            if spontaneous_chat_allowed and twitch is not None and twitch.is_available():
                try:
                    print("[HEBE][TWITCH][CHATBOT] send_message reason=spontaneity", flush=True)
                    twitch.send_message(str(text or "").strip())
                    self._record_spontaneous_twitch_chat_sent(text, payload)
                    try:
                        anchor_id = str((payload or {}).get("used_fact_id") or (payload or {}).get("anchor_id") or (payload or {}).get("idle_topic") or "").strip() or None
                        self._get_live_session_brain().observe_hebe_utterance(
                            text,
                            output_target=targets,
                            input_type="spontaneity",
                            anchor_id=anchor_id,
                            topic=(payload or {}).get("idle_topic"),
                        )
                        self._get_live_session_brain().consume_anchor(anchor_id)
                    except Exception as exc:
                        print(f"[HEBE][LIVE_SESSION] hebe spontaneity record failed: {exc!r}", flush=True)
                except Exception as e:
                    print(f"[HEBE][EVENT][TWITCH] send_message failed: {e!r}", flush=True)
            else:
                if output_mode == "silent":
                    print("[HEBE][SPONTANEITY] skipped reason=stream_output_mode_silent", flush=True)
                    return
                emit("chat.assistant", {"text": text, "source": "spontaneity", "output_target": OUTPUT_TARGET_LOCAL_UI})
                try:
                    self._get_live_session_brain().observe_hebe_utterance(
                        text,
                        output_target=targets,
                        input_type="spontaneity",
                        anchor_id=str((payload or {}).get("used_fact_id") or (payload or {}).get("idle_topic") or "").strip() or None,
                        topic=(payload or {}).get("idle_topic"),
                    )
                except Exception as exc:
                    print(f"[HEBE][LIVE_SESSION] hebe spontaneity record failed: {exc!r}", flush=True)
        elif is_simulated or output_mode == "ui_only":
            emit("chat.assistant", {"text": text, "source": "simulation" if is_simulated else "twitch", "output_target": OUTPUT_TARGET_LOCAL_UI})
            try:
                self._get_live_session_brain().observe_hebe_utterance(
                    text,
                    output_target=targets,
                    input_type="twitch_mention_or_event",
                    topic=event_type or "twitch_event",
                )
            except Exception as exc:
                print(f"[HEBE][LIVE_SESSION] hebe twitch record failed: {exc!r}", flush=True)
        elif output_mode == "silent":
            print("[HEBE][EVENT][TWITCH] output_mode=silent dropping chat reply", flush=True)
            return
        elif twitch is not None and twitch.is_available():
            try:
                twitch.send_message(text)
                try:
                    self._get_live_session_brain().observe_hebe_utterance(
                        text,
                        output_target=targets,
                        input_type="twitch_mention_or_event",
                        topic=event_type or "twitch_event",
                    )
                except Exception as exc:
                    print(f"[HEBE][LIVE_SESSION] hebe twitch record failed: {exc!r}", flush=True)
            except Exception as e:
                print(f"[HEBE][EVENT][TWITCH] send_message failed: {e!r}", flush=True)
        else:
            print("[HEBE][EVENT][TWITCH] service not available, dropping chat reply", flush=True)

        if not getattr(self.runtime.state, "tts_enabled", False):
            print("[HEBE][TTS] skipped reason=global_disabled", flush=True)
            return
        if is_simulated:
            print("[HEBE][TTS] skipped reason=dev_simulation", flush=True)
            return
        if is_spontaneous and spontaneous_chat_allowed and not tts_allowed:
            print("[HEBE][TTS] skipped reason=spontaneous_twitch_chat_text_only", flush=True)
            return
        if not tts_allowed:
            print("[HEBE][TTS] skipped reason=stream_tts_disabled", flush=True)
            return
        self._deliver_voice_reply(text, output_target=OUTPUT_TARGET_STREAM_TTS, emit_ui=False, declare_route=False)


    def _deliver_voice_reply(
        self,
        text: str,
        *,
        output_target: str = OUTPUT_TARGET_LOCAL_TTS,
        emit_ui: bool = True,
        input_type: str = "direct_stt",
        declare_route: bool = True,
    ) -> None:
        if not text:
            return
        if declare_route:
            targets = [OUTPUT_TARGET_LOCAL_UI] if emit_ui else []
            if self._local_tts_output_enabled():
                targets.append(output_target)
            self._declare_output_route(
                input_type=input_type,
                targets=targets or [OUTPUT_TARGET_LOCAL_UI],
                reason="voice_reply",
            )
        if emit_ui:
            emit("chat.assistant", {"text": text, "source": input_type, "output_target": OUTPUT_TARGET_LOCAL_UI})
        try:
            targets_for_brain = [OUTPUT_TARGET_LOCAL_UI] if emit_ui else []
            if self._local_tts_output_enabled():
                targets_for_brain.append(output_target)
            self._get_live_session_brain().observe_hebe_utterance(
                text,
                output_target=targets_for_brain or [OUTPUT_TARGET_LOCAL_UI],
                input_type=input_type,
                expects_possible_reply_from_leo=True,
            )
        except Exception as exc:
            print(f"[HEBE][LIVE_SESSION] hebe voice record failed: {exc!r}", flush=True)
        if not getattr(self.runtime.state, "tts_enabled", False):
            print("[HEBE][TTS] skipped reason=global_disabled", flush=True)
            return
        if not self._local_tts_output_enabled():
            print("[HEBE][TTS] skipped reason=stream_output_mode", flush=True)
            return
        try:
            safe_text = str(text or "").replace('"', '\\"')
            print(f"[HEBE][TTS] speaking output_target={output_target} text=\"{safe_text}\"", flush=True)
            self._remember_tts_text(text)
            self.runtime.speak(text, emit_chat=False)
        except Exception as e:
            safe_error = str(e).replace('"', '\\"')
            print(f"[HEBE][TTS] failed error=\"{safe_error}\"", flush=True)

    def _deliver_manual_reply(self, text: str, *, source: str) -> None:
        if source == "ui":
            self._declare_output_route(
                input_type="ui_typed_input",
                targets=[OUTPUT_TARGET_LOCAL_UI],
                reason="typed_input_reply",
            )
            log_chat("assistant", text, source="ui")
            emit("chat.assistant", {"text": text, "source": "ui", "output_target": OUTPUT_TARGET_LOCAL_UI})
            try:
                self._get_live_session_brain().observe_hebe_utterance(
                    text,
                    output_target=[OUTPUT_TARGET_LOCAL_UI],
                    input_type="ui_typed_input",
                    expects_possible_reply_from_leo=True,
                )
            except Exception as exc:
                print(f"[HEBE][LIVE_SESSION] hebe ui record failed: {exc!r}", flush=True)
            self._record_assistant_reply_for_conversation(text, source=source)
            return

        targets = [OUTPUT_TARGET_LOCAL_UI]
        voice_target = self._direct_voice_tts_target()
        if self._local_tts_output_enabled():
            targets.append(voice_target)
        self._declare_output_route(
            input_type="direct_stt" if source == "stt_voice" else source,
            targets=targets,
            reason="direct_reply",
        )
        self._deliver_voice_reply(
            text,
            output_target=voice_target,
            input_type="direct_stt" if source == "stt_voice" else source,
            declare_route=False,
        )
        self._record_assistant_reply_for_conversation(text, source=source)

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
                    source="ui",
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
                    res = self._process_stt_voice_transcript(str(raw_voice_command), allow_wakeword_prompt=True)
                    self._current_input_event = None
                    if res in ("sleep", "stop"):
                        return res
                    continue
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
                        self._declare_output_route(
                            input_type="ambient_stt",
                            targets=[OUTPUT_TARGET_SILENT_CONTEXT_UPDATE],
                            reason=voice_type,
                        )
                        self._log_stt_non_command_decision(
                            command,
                            "ambient_context_only",
                            reason=voice_type,
                        )
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
            try:
                self.poll_internal_events()
                self.poll_stream_routine()
                self.poll_stream_context(require_enabled=False)
                self.poll_stream_presence()
            except Exception as exc:
                print(f"[HEBE][WAKE_LOOP][ERROR] poll failed but continuing error={exc!r}", flush=True)
                self._wake_loop_last_error = str(exc)
                try:
                    emit("status", {"wake_loop_alive": True, "wake_loop_error": str(exc), "wake_loop_status": "alive"})
                except Exception:
                    pass
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
            # Route every accepted STT transcript through Hebe cognition.
            res = self._process_stt_voice_transcript(str(raw_voice_command), allow_wakeword_prompt=True)
            if res == "stop":
                return "stop"
            continue

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
                self.runtime.state.hebe_sleeping = True
                self.runtime.state.mode = "sleep"
                self._emit_audio_status()
                continue


if __name__ == "__main__":
    runtime = build_runtime()
    engine = HebeEngine(runtime=runtime, use_wakeword=True, say_hello=True)
    engine.start()

    while True:
        time.sleep(1)

