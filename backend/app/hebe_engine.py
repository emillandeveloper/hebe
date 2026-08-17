import os
import re
import time
import threading
import unicodedata
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
from queue import Empty
from difflib import SequenceMatcher
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class TwitchWriteDecision:
    should_generate: bool
    should_write_to_twitch: bool
    should_tts: bool
    route: str
    reason: str
    value_score: float
    risk_score: float
    thread_action: str
    budget_result: dict
    suggested_speech_act: str

from app.services.db_sqlite import (
    DB_PATH,
    cleanup_stt_prompt_injection_rows,
    init_db,
    log_chat,
    seed_default_apps,
)
from app.services.vts_client import vts_hotkey
from app.services.voice_command_recovery import TranscriptNormalizationResult, normalize_stt_transcript
from app.services.utterance_role import UtteranceRole, UtteranceRoleClassifier
from app.services.stream_tts_guard import StreamTTSSafetyManager
from app.services.stt_whisper import is_stt_prompt_injection
from app.services.direct_stt_command import (
    DirectSTTCommandResult,
    DirectUtteranceIntentFamily,
    parse_direct_stt_command,
)
from app.core.ui_bridge import emit
from app.core.input_bus import submit_text_from_ui, submit_text_from_voice, get_ui_inbox, get_voice_inbox
from app.core.stt_worker import STTWorker
from app.core.runtime import build_runtime, HebeRuntime
from app.core.persistent_logs import log_jsonl_event

from app.cognitive import MemoryStore, SchedulerService
from app.cognitive.scheduler import InternalEvent
from app.cognitive.command_result import CommandResult
from app.cognitive.input_event import InputEnvelope, InputEvent
from app.cognitive.input_interpretation import InputInterpretation, InputInterpreter, InputSpeechAct
from app.cognitive.interaction_history import (
    RecentInteractionDecisionHistory,
    detect_self_explanation_query,
    render_grounded_self_explanation,
)
from app.cognitive.models import ExecutionResult, Plan
from app.cognitive.core_loop import (
    HebeCoreLoop,
    PerceivedEvent,
    PolicyContract,
    PresenceEngine,
    UnderstandingResult,
)
from app.cognitive.final_emission_gate import FinalEmissionGate, FinalGuardDecision, OutputRoute
from app.cognitive.twitch_interaction_coordinator import TwitchInteractionCoordinator, TrollEngagementBudget
from app.cognitive.game_guidance import GameGuidanceCapability, GameRunState
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
from app.stream.companion_loop import StreamCompanionLoop
from app.stream.game_knowledge import GameKnowledgeConfig, GameKnowledgeResolver
from app.stream.game_profiles import GameProfileStore
from app.stream.game_research import GameKnowledgeResearchConfig, GameKnowledgeResearchService
from app.stream.game_intelligence import (
    GameAssistanceMode,
    GameResearchService as LiveGameResearchService,
    ResearchMode,
    default_assistance_mode,
)
from app.stream import memory as stream_memory
from app.stream.live_session import LiveSessionBrain, init_live_session_schema
from app.stream.live_runtime import LiveSessionStateManager
from app.stream.runtime_context import HebeLiveContextPolicy, HebeLiveRuntimeContext
from app.stream.scene_timeline import SceneTimelineManager, SpontaneousOpportunityManager
from app.stream.viewer_operation_gate import ViewerStreamOperationTopicGate
from app.stream.promotions import (
    AutomaticPromotionService,
    AutoPromoMode,
    PromotionExecutionStatus,
    PromotionProfileManager,
    PromotionStore,
    PromotionTriggerType,
    parse_promotion_profile_command,
)
from app.stream.ambient_context import AmbientContextExtractor
from app.stream.conversation_ownership import ConversationOwnershipGate
from app.stream.audio_state import EffectiveStreamAudioState
from app.stream.discourse import (
    DiscourseContributionPlanner,
    DiscourseGroundingGuard,
    DiscourseParticipationBudget,
    OwnerDiscourseBuffer,
    StreamTurnDetector,
)
from app.stream.social_events import (
    CheerAcknowledgementRenderer,
    CheerDeduplicator,
    CheerEventPolicy,
    StreamSocialEventRouter,
    TwitchCheerEvent,
)
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
from app.stream.behavior_adaptation import AdaptationAction, BehaviorAdaptationService
from app.stream.behavior_constraint_store import (
    BehaviorConstraintRepository,
    behavior_constraint_migrations,
)
from app.stream.viewer_profiles import (
    GrammaticalAgreementGuard,
    ViewerLinguisticProfileStore,
    ViewerProfileCommandParser,
)
from app.stream.social_response_guards import ChannelRetentionGuard, SocialAuthorityCommitmentGuard
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
from app.stream.proactive import (
    StreamPreparationRoutine,
    scheduled_reminder_decision,
)
from app.continuity import (
    ConversationContext,
    ConversationContinuityService,
    ConversationRepository,
    ConversationStatus,
    ConversationalAct,
    CurrentConversation,
    ExpectedReply,
    ExpectedReplyType,
    OpenThreadRepository,
)
from app.replay.migrations import MigrationRunner, architecture_consolidation_migrations, conversation_continuity_migrations, game_context_v2_migrations, social_world_v2_migrations, learning_v2_migrations
from app.epistemics.models import EvidenceRef, EvidenceRelation
from app.game_context_v2.context import GameContextResolver
from app.game_context_v2.migration import (
    game_knowledge_canonicalization_migrations,
    game_run_state_canonicalization_migrations,
)
from app.game_context_v2.repository import GameV2Repository
from app.game_context_v2.service import GameKnowledgeService as GameKnowledgeV2Service, GameRunService
from app.social_world_v2 import SocialWorldRepository, SocialWorldService
from app.social_world_v2.migration import (
    social_identity_canonicalization_migrations,
    social_summary_canonicalization_migrations,
)
from app.learning_v2 import (
    ContinuityContextBuilder, HebeSelfModel, HistoricalActionLedger, LeoLanguageModel,
    OwnerProceduralPreferences, SceneConsequenceReducer, SessionConsolidator,
    StableHebeCore, TemporalRelevanceService,
)
from app.learning_v2.repository import LearningRepository
from app.integrity.production_defaults import enabled as cognitive_flag
from app.services import db_sqlite

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
            def _twitch_event_callback(event_type, payload):
                if str(event_type or "") == "twitch_outgoing_raid":
                    # Ending signals must not depend on a later scheduler poll;
                    # shutdown may follow the raid immediately.
                    self.process_internal_event(InternalEvent(
                        event_type="twitch_outgoing_raid",
                        payload=dict(payload or {}),
                        created_at=datetime.now(timezone.utc).isoformat(),
                    ))
                    return
                self.scheduler.push_event(event_type, payload)

            self.runtime.twitch_events.push_event_callback = _twitch_event_callback

        if hasattr(self.runtime, 'twitch_chat_bot') and self.runtime.twitch_chat_bot:
            def _twitch_ambient_callback(username, display_name, text, channel, tags=None):
                self.observe_twitch_chat_message(username, display_name, text, channel, irc_tags=tags or {})

            def _twitch_social_event_callback(event_type, payload):
                self.process_internal_event(InternalEvent(
                    event_type=str(event_type or "twitch_event"),
                    payload=dict(payload or {}),
                    created_at=datetime.now(timezone.utc).isoformat(),
                ))

            def _twitch_chat_callback(username, display_name, text, channel, tags=None):
                self.ingest_normalized_twitch_chat(
                    username=username,
                    display_name=display_name,
                    text=text,
                    channel=channel,
                    irc_tags=tags or {},
                )

            self.runtime.twitch_chat_bot.ambient_message_callback = _twitch_ambient_callback
            self.runtime.twitch_chat_bot.message_callback = _twitch_chat_callback
            self.runtime.twitch_chat_bot.social_event_callback = _twitch_social_event_callback

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
        self.deliberation_service.local_app_planner = self.local_app_planner

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
        game_guidance = GameGuidanceCapability(
            profile_store=self.game_profiles,
            search_provider=getattr(self.game_research, "search_provider", None),
        )
        self.cognitive_router.game_guidance = game_guidance
        self.deliberation_service.cognitive_router.game_guidance = game_guidance
        self.deliberation_service.game_guidance = game_guidance
        self.deliberation_service.goal_extractor.game_guidance = game_guidance
        self.response_synthesizer._game_guidance_classifier = game_guidance
        self.game_knowledge = GameKnowledgeResolver(
            profile_store=self.game_profiles,
            research_service=self.game_research,
            config=GameKnowledgeConfig.from_env(),
        )
        self.game_intelligence = LiveGameResearchService(
            provider=(
                getattr(self.game_research, "search_provider", None)
                if getattr(getattr(self.game_research, "config", None), "enabled", False)
                else None
            ),
            provider_name=str(getattr(getattr(self.game_research, "config", None), "provider", "") or "none"),
            provider_configured=bool(getattr(getattr(self.game_research, "config", None), "provider", "")),
        )
        self.scene_timeline = SceneTimelineManager()
        self.spontaneous_opportunities = SpontaneousOpportunityManager()
        self.viewer_stream_operation_gate = ViewerStreamOperationTopicGate()
        self.response_synthesizer.scene_timeline = self.scene_timeline
        self.response_synthesizer.spontaneous_opportunities = self.spontaneous_opportunities
        initial_stream_state = getattr(self.runtime.state, "stream", None)
        if initial_stream_state is not None:
            initial_stream_state.stream_output_language = (
                self.response_synthesizer.stream_output_language.configured_language
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
                require_specific_context=os.getenv("HEBE_SPONTANEITY_REQUIRE_SPECIFIC_CONTEXT", "true").strip().lower() in ("1", "true", "yes", "on"),
                chat_activity_window_sec=float(os.getenv("HEBE_CHAT_ACTIVITY_WINDOW_SECONDS", "180")),
                chat_active_message_threshold=int(os.getenv("HEBE_CHAT_ACTIVE_MESSAGE_THRESHOLD", "3")),
                chat_active_user_threshold=int(os.getenv("HEBE_CHAT_ACTIVE_USER_THRESHOLD", "1")),
                suppress_when_chat_active=os.getenv("HEBE_IDLE_SUPPRESS_WHEN_CHAT_ACTIVE", "true").strip().lower() in ("1", "true", "yes", "on"),
            ),
        )
        self.stream_preparation = StreamPreparationRoutine()
        try:
            self.behavior_constraint_migrations = MigrationRunner(
                db_sqlite.get_db_connection,
            ).migrate(behavior_constraint_migrations())
            self.behavior_constraint_repository = BehaviorConstraintRepository(
                db_sqlite.get_db_connection,
            )
        except Exception as exc:
            self.behavior_constraint_migrations = []
            self.behavior_constraint_repository = None
            print(
                "[HEBE][BEHAVIOR_CONSTRAINT_STORE] status=unavailable "
                "reason_code=behavior_constraint_store_unavailable "
                f"error_type={type(exc).__name__}",
                flush=True,
            )
        self.behavior_adaptation = BehaviorAdaptationService(
            repository=self.behavior_constraint_repository,
        )
        if initial_stream_state is not None:
            self.behavior_adaptation.load_durable_constraints(initial_stream_state)
        self.stream_spontaneity.start_grace_period(getattr(self.runtime.state, "stream", None))
        self.stream_companion_loop = StreamCompanionLoop(
            spontaneity=self.stream_spontaneity,
            presence_engine=self._get_presence_engine(),
            scene_timeline=self.scene_timeline,
            opportunities=self.spontaneous_opportunities,
            owner_voice_active_fn=self._owner_audio_active,
            tts_active_fn=lambda: bool(getattr(self, "_tts_active", False)),
            behavior_adaptation=self.behavior_adaptation,
        )
        self.stream_context_sync = StreamContextSyncService(
            twitch_api=getattr(self.runtime, "twitch", None),
        )
        self.live_session_brain = LiveSessionBrain(getattr(self.runtime.state, "stream", None))
        self.live_session_state_manager = LiveSessionStateManager()
        self.live_context_policy = HebeLiveContextPolicy()
        self.promotion_store = PromotionStore()
        self.promotion_profile_manager = PromotionProfileManager(self.promotion_store)
        self._twitch_user_ids_by_login: dict[str, str] = {}
        self.automatic_promotions = AutomaticPromotionService(
            self.promotion_store,
            spacing_seconds=float(os.getenv("HEBE_AUTO_PROMO_SPACING_SECONDS", "8") or 8),
            max_retries=int(os.getenv("HEBE_AUTO_PROMO_MAX_RETRIES", "1") or 1),
            bot_usernames=self._input_firewall_bot_usernames(),
            self_usernames={
                str(getattr(getattr(self.runtime, "twitch", None), "channel_name", "") or ""),
                str(getattr(getattr(self.runtime, "twitch", None), "bot_username", "") or ""),
            },
        )
        initial_output_mode = os.getenv("HEBE_STREAM_OUTPUT_MODE", "").strip()
        if initial_output_mode in {"ui_only", "tts_enabled", "twitch_chat_only", "silent"}:
            stream_state = getattr(self.runtime.state, "stream", None)
            if stream_state is not None:
                stream_state.stream_output_mode = initial_output_mode
                print(f"[HEBE][OUTPUT_MODE] mode={initial_output_mode} reason=config", flush=True)
        self._apply_stream_performance_profile()
        self.ambient_context_extractor = AmbientContextExtractor()
        self.utterance_role_classifier = UtteranceRoleClassifier()
        self.stream_tts_safety = StreamTTSSafetyManager()
        self.owner_discourse_buffer = OwnerDiscourseBuffer(
            tracker=None,
            session_gap_seconds=float(os.getenv("HEBE_DISCOURSE_SESSION_GAP_SECONDS", "90") or 90),
        )
        self.discourse_contribution_planner = DiscourseContributionPlanner()
        self.discourse_grounding_guard = DiscourseGroundingGuard()
        self.stream_turn_detector = StreamTurnDetector(
            natural_pause_seconds=float(os.getenv("HEBE_DISCOURSE_NATURAL_PAUSE_SECONDS", "3.5") or 3.5)
        )
        self.discourse_participation_budget = DiscourseParticipationBudget(
            min_between_seconds=float(os.getenv("HEBE_DISCOURSE_MIN_BETWEEN_SECONDS", "480") or 480),
            max_per_hour=int(os.getenv("HEBE_DISCOURSE_MAX_PER_HOUR", "3") or 3),
        )
        self.discourse_participation_mode = os.getenv("HEBE_DISCOURSE_PARTICIPATION_MODE", "shadow").strip().lower()
        if self.discourse_participation_mode not in {"disabled", "shadow", "active"}:
            self.discourse_participation_mode = "shadow"
        stream_for_discourse = getattr(self.runtime.state, "stream", None)
        if stream_for_discourse is not None:
            stream_for_discourse.discourse_participation_mode = self.discourse_participation_mode
        self.stream_social_event_router = StreamSocialEventRouter()
        self.cheer_event_policy = CheerEventPolicy()
        self.cheer_deduplicator = CheerDeduplicator(
            window_seconds=float(os.getenv("HEBE_CHEER_DEDUPE_WINDOW_SECONDS", "20") or 20)
        )
        self.cheer_ack_renderer = CheerAcknowledgementRenderer()
        self.memory_extractor = MemoryExtractor(
            intent_model=getattr(self.runtime, "intent_llm", None),
            memory_store=self.memory_store,
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
        # Turn arbitration is cheap and must observe normal 1-3 second gaps.
        # Candidate rendering/model work still happens only after an intent wins.
        self.presence_poll_interval_sec = float(os.getenv("HEBE_PRESENCE_POLL_INTERVAL_SECONDS", "0.5"))
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
        self.presence_engine_mode = os.getenv(
            "HEBE_PRESENCE_ENGINE_MODE",
            "active",
        ).strip().lower() or "active"
        self.auto_shoutout_raiders = os.getenv(
            "HEBE_AUTO_SHOUTOUT_RAIDERS",
            "false",
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
        self.viewer_intent_policy = ViewerIntentPolicy(constraint_owner=self.behavior_adaptation)
        self.grammatical_agreement_guard = GrammaticalAgreementGuard()
        self.social_authority_commitment_guard = SocialAuthorityCommitmentGuard()
        self.channel_retention_guard = ChannelRetentionGuard()
        self.viewer_profile_command_parser = ViewerProfileCommandParser()
        self.viewer_linguistic_profiles = ViewerLinguisticProfileStore()
        self.input_authority_firewall = self._build_input_firewall()
        self._last_input_firewall: dict = {}
        self._last_policy_trace: dict = {}
        self._last_cognitive_trace: dict = {}
        self._current_input_event: InputEvent | None = None
        self.cognitive_v2_enabled = cognitive_flag("HEBE_COGNITIVE_V2_ENABLED")
        self._last_continuity_resolution: dict = {}
        self._initialize_conversation_continuity()
        self._initialize_belief_v2()
        self._initialize_game_context_v2()
        self._initialize_social_world_v2()
        self.consolidation_v2 = cognitive_flag("HEBE_CONSOLIDATION_V2")
        self.consolidation_commits_v2 = cognitive_flag("HEBE_CONSOLIDATION_COMMITS_V2")
        self.hebe_self_v2 = cognitive_flag("HEBE_HEBE_SELF_V2")
        self.owner_preferences_v2 = cognitive_flag("HEBE_OWNER_PREFERENCES_V2")
        self.leo_language_v2 = cognitive_flag("HEBE_LEO_LANGUAGE_V2")
        self.temporal_relevance_v2 = cognitive_flag("HEBE_TEMPORAL_RELEVANCE_V2")
        self.schedule_learning_v2 = cognitive_flag("HEBE_SCHEDULE_LEARNING_V2")
        self.scene_consequence_v2 = cognitive_flag("HEBE_SCENE_CONSEQUENCE_V2")
        self.historical_action_ledger_v2 = cognitive_flag("HEBE_HISTORICAL_ACTION_LEDGER_V2")
        self._initialize_learning_v2()

    def _initialize_learning_v2(self) -> None:
        try:
            if self.belief_repository is None or self.conversation_continuity is None:raise RuntimeError("continuity_foundation_unavailable")
            self.learning_v2_migrations=MigrationRunner(db_sqlite.get_db_connection).migrate(learning_v2_migrations())
            self.phase6_migrations=MigrationRunner(db_sqlite.get_db_connection).migrate(architecture_consolidation_migrations())
            repo=LearningRepository(db_sqlite.get_db_connection);core=StableHebeCore()
            self.learning_repository=repo;self.stable_hebe_core=core
            self.hebe_self_model=HebeSelfModel(self.belief_repository,repo,now_fn=lambda:time.time())
            self.owner_procedural_preferences=OwnerProceduralPreferences(self.belief_repository,repo,now_fn=lambda:time.time())
            self.leo_language_model=LeoLanguageModel(self.belief_repository,repo,now_fn=lambda:time.time())
            self.historical_action_ledger=HistoricalActionLedger(repo,now_fn=lambda:time.time())
            if self.historical_action_ledger_v2:self.historical_action_ledger.project_existing_receipts()
            self.scene_consequence_reducer=SceneConsequenceReducer(repo,self.owner_procedural_preferences,now_fn=lambda:time.time())
            self.temporal_relevance_service=TemporalRelevanceService(db_sqlite.get_db_connection,self.belief_repository,self.conversation_continuity.threads,getattr(self,"social_world_repository",None),repo,now_fn=lambda:time.time())
            self.session_consolidator=SessionConsolidator(repo,core,self.hebe_self_model,self.owner_procedural_preferences,self.leo_language_model,now_fn=lambda:time.time(),candidate_provider=self._propose_consolidation_candidates)
            self.continuity_context_builder=ContinuityContextBuilder(self.hebe_self_model,self.owner_procedural_preferences,self.leo_language_model,self.historical_action_ledger,self.scene_consequence_reducer,now_fn=lambda:time.time())
        except Exception as exc:
            self.learning_repository=None;self.stable_hebe_core=None;self.hebe_self_model=None;self.owner_procedural_preferences=None;self.leo_language_model=None;self.historical_action_ledger=None;self.scene_consequence_reducer=None;self.temporal_relevance_service=None;self.session_consolidator=None;self.continuity_context_builder=None;self.learning_v2_migrations=[]
            print(f"[HEBE][LEARNING_V2_INIT] status=failed_closed reason={type(exc).__name__}",flush=True)

    def _propose_consolidation_candidates(self, *, session_id: str, domain: str, schema_version: int, evidence: list[dict]) -> list[dict]:
        """Model proposal only. SessionConsolidator remains the sole validator/committer."""
        model=getattr(self.runtime,"intent_llm",None) or getattr(self.runtime,"llm",None)
        structured=getattr(model,"chat_structured",None)
        if not callable(structured):return []
        safe_evidence=[{k:item.get(k) for k in ("event_uid","event_type","source","speaker","normalized_text","context_kind","authority","valid_from","valid_until") if k in item} for item in evidence[-250:]]
        response=structured(
            system_prompt="Propose only meaningful typed state deltas supported by supplied evidence. Never summarize the transcript, mutate Hebe's stable identity, conflate domains, or claim actions without receipts. Empty candidates is desirable when nothing changed.",
            user_prompt=__import__('json').dumps({"purpose":"session_consolidation","domain":domain,"schema_version":schema_version,"session_id":session_id,"evidence":safe_evidence},ensure_ascii=False),
            schema={"type":"object","properties":{"candidates":{"type":"array","items":{"type":"object","properties":{"domain":{"type":"string"},"delta_type":{"type":"string"},"payload":{"type":"object"},"evidence_ids":{"type":"array","items":{"type":"string"}},"idempotency_key":{"type":"string"}},"required":["domain","delta_type","payload","evidence_ids"]}}},"required":["candidates"]},
            purpose=f"session_consolidation:{domain}:v{schema_version}:{session_id}",
        )
        return [dict(x) for x in dict(response or {}).get("candidates") or []]

    def _initialize_social_world_v2(self) -> None:
        try:
            if self.belief_lifecycle is None or self.conversation_continuity is None:raise RuntimeError("phase1_or_phase2_unavailable")
            runner=MigrationRunner(db_sqlite.get_db_connection);self.social_world_v2_migrations=runner.migrate(social_world_v2_migrations())
            self.social_identity_canonicalization=runner.migrate(social_identity_canonicalization_migrations())
            self.social_summary_canonicalization=runner.migrate(social_summary_canonicalization_migrations())
            repository=SocialWorldRepository(db_sqlite.get_db_connection);self.social_world_repository=repository
            self.social_world=SocialWorldService(repository,self.belief_lifecycle,self.conversation_continuity.threads,getattr(self,"memory_retrieval",None),now_fn=lambda:time.time())
        except Exception as exc:
            self.social_world_repository=None;self.social_world=None;self.social_world_v2_migrations=[];self.social_identity_canonicalization=[];self.social_summary_canonicalization=[]
            print(f"[HEBE][SOCIAL_WORLD_V2_INIT] status=failed_closed reason={type(exc).__name__}",flush=True)

    def _initialize_game_context_v2(self) -> None:
        try:
            if self.belief_repository is None or self.belief_lifecycle is None: raise RuntimeError("belief_v2_unavailable")
            runner=MigrationRunner(db_sqlite.get_db_connection);self.game_context_v2_migrations=runner.migrate(game_context_v2_migrations())
            self.game_run_state_canonicalization=runner.migrate(game_run_state_canonicalization_migrations())
            self.game_knowledge_canonicalization=runner.migrate(game_knowledge_canonicalization_migrations())
            repository=GameV2Repository(db_sqlite.get_db_connection);runs=GameRunService(repository,self.belief_lifecycle,now_fn=lambda:time.time());knowledge=GameKnowledgeV2Service(repository,self.belief_lifecycle,now_fn=lambda:time.time())
            self.game_v2_repository=repository;self.game_run_service=runs;self.game_knowledge_v2_service=knowledge
            self.game_context_resolver=GameContextResolver(repository,runs,knowledge,research_service=getattr(self,"game_intelligence",None),memory_retrieval=getattr(self,"memory_retrieval",None),now_fn=lambda:time.time())
            self.game_knowledge.run_service=runs
        except Exception as exc:
            self.game_v2_repository=None;self.game_run_service=None;self.game_knowledge_v2_service=None;self.game_context_resolver=None;self.game_context_v2_migrations=[];self.game_run_state_canonicalization=[];self.game_knowledge_canonicalization=[]
            print(f"[HEBE][GAME_CONTEXT_V2_INIT] status=failed_closed reason={type(exc).__name__}",flush=True)

    def _initialize_belief_v2(self) -> None:
        try:
            self.belief_v2_migrations=[]
            self.belief_repository=self.memory_store.repository
            self.belief_lifecycle=self.memory_store.lifecycle
            self.memory_retrieval=self.memory_store.retrieval
        except Exception as exc:
            self.belief_repository=None;self.belief_lifecycle=None;self.memory_retrieval=None;self.belief_v2_migrations=[]
            print(f"[HEBE][BELIEF_INIT] status=failed_closed reason={type(exc).__name__}",flush=True)

    def _initialize_conversation_continuity(self) -> None:
        try:
            runner = MigrationRunner(db_sqlite.get_db_connection)
            self.conversation_continuity_migrations = runner.migrate(conversation_continuity_migrations())
            conversations = ConversationRepository(db_sqlite.get_db_connection)
            threads = OpenThreadRepository(db_sqlite.get_db_connection)
            self.conversation_continuity = ConversationContinuityService(
                conversations, threads, now_fn=lambda: time.time(),
            )
            invalidated = conversations.interrupt_active_on_start(reason="runtime_restart")
            if invalidated:
                threads.archive_interrupted_clarifications(event_id="runtime_restart", now=time.time())
            if invalidated:
                print(f"[HEBE][CONVERSATION_RESTART] invalidated_actionable={invalidated}", flush=True)
        except Exception as exc:
            self.conversation_continuity = None
            self.conversation_continuity_migrations = []
            print(f"[HEBE][CONVERSATION_INIT] status=failed_closed reason={type(exc).__name__}", flush=True)

    def _conversation_context_for_owner_stt(self) -> tuple[str, str]:
        stream = self._get_stream_state()
        if stream is not None and bool(getattr(stream, "enabled", False)) and bool(getattr(stream, "is_live", False)):
            context_kind = ConversationContext.OWNER_LIVE_CONTROL.value
            context_id = str(getattr(stream, "active_stream_session_id", None) or "active_stream")
            # A stream session id may become known after Hebe has already handed
            # a turn to Leo. Keep that live conversation addressable for its
            # lifetime instead of silently switching context ids mid-turn.
            conversations = getattr(getattr(self, "conversation_continuity", None), "conversations", None)
            if conversations is not None and context_id != "active_stream":
                if conversations.get_active(context_kind, context_id) is None:
                    provisional = conversations.get_active(context_kind, "active_stream")
                    if provisional is not None:
                        return context_kind, "active_stream"
            return context_kind, context_id
        return ConversationContext.OWNER_LOCAL.value, "leo_local"

    def _resolve_conversation_continuation(
        self, *, text: str, event_id: str, wake: bool, force_ambient: bool,
        compatibility: bool | None = None, compatibility_reason: str = "", consume: bool = True,
    ):
        service = getattr(self, "conversation_continuity", None)
        if service is None:
            return None
        context_kind, context_id = self._conversation_context_for_owner_stt()
        source = "ambient_stt" if force_ambient else "owner_stt"
        participant = "ambient" if force_ambient else "leo"
        authority = "ambient" if force_ambient else "owner"
        try:
            resolution = service.resolve_input(
                context_kind=context_kind, context_id=context_id, source=source,
                participant=participant, authority=authority, text=text,
                event_id=event_id, wake=wake,
                consume=consume,
                compatibility=compatibility, compatibility_reason=compatibility_reason,
            )
            self._last_continuity_resolution = resolution.to_dict()
            return resolution
        except Exception as exc:
            self._last_continuity_resolution = {
                "consumed": False, "decision": "reject", "reason": f"continuity_failure:{type(exc).__name__}"
            }
            return None

    def _apply_game_run_correction_continuation(self, continuation, *, event_id: str, text: str) -> None:
        if not (continuation is not None and continuation.consumed and str(getattr(continuation.reply_act,"value",continuation.reply_act))=="CORRECT"):
            return
        domain=dict((continuation.payload or {}).get("domain") or {})
        if domain.get("kind")!="game_run_belief_correction":return
        belief_id=str(domain.get("target_belief_id") or "")
        normalized=self._normalize_text(text);value=None
        for token,candidate in dict(domain.get("correction_values") or {}).items():
            if self._normalize_text(str(token)) in normalized:value=candidate;break
        if not belief_id or value is None:
            print("[HEBE][GAME_RUN_CORRECT] decision=defer reason=ambiguous_target_or_value",flush=True);return
        evidence=EvidenceRef(source_event_id=event_id,source_record_type="live_session_timeline",source_record_id=event_id,relation=EvidenceRelation.CORRECTS,observed_at=time.time(),extractor="conversation_game_domain",extractor_version="v1",literal_span={"start":0,"end":len(text),"excerpt":text[:80]})
        try:getattr(self,"game_run_service",None).correct_fact(belief_id,object_value=value,evidence=evidence)
        except Exception as exc:print(f"[HEBE][GAME_RUN_CORRECT] decision=defer reason={type(exc).__name__}",flush=True)

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
        override = getattr(self, "_simulation_stream_live_override", None)
        if override is not None:
            return bool(override)
        stream = self._get_stream_state()
        return bool(stream and getattr(stream, "is_live", False))

    def _conversation_context_for_source(self, source: str = "") -> tuple[ConversationContext, str]:
        event_source = str(source or getattr(getattr(self, "_current_input_event", None), "source", "") or "")
        if event_source in {"ui", "typed_ui", "owner_ui"}:
            return ConversationContext.PRIVATE_UI, "leo_ui"
        context_kind, context_id = self._conversation_context_for_owner_stt()
        return ConversationContext(context_kind), context_id

    def _active_current_conversation(
        self, *, source: str = "", latest: bool = False, expire: bool = True,
    ) -> CurrentConversation | None:
        service = getattr(self, "conversation_continuity", None)
        if service is None:
            return None
        conversation = service.latest_active_conversation() if latest else service.conversations.get_active(
            *self._conversation_context_for_source(source)
        )
        if conversation is not None and expire and conversation.expires_at <= time.time():
            print(f"[HEBE][PENDING_EXPIRED] kind={conversation.topic} id={conversation.id}", flush=True)
            service.close_conversation(conversation, reason="expired")
            return None
        return conversation

    def _close_current_conversation(
        self, *, reason: str, conversation: CurrentConversation | None = None,
    ) -> None:
        conversation = conversation or self._active_current_conversation(latest=True, expire=False)
        service = getattr(self, "conversation_continuity", None)
        if conversation is None or service is None:
            return
        service.close_conversation(conversation, reason=reason)
        print(f"[HEBE][PENDING_CLEARED] reason={reason} kind={conversation.topic} id={conversation.id}", flush=True)

    def _open_pending_conversation(
        self,
        *,
        kind: str,
        expected_reply_type: str,
        authority_required: str = "owner",
        allowed_sources: list[str] | None = None,
        capability_needed: str = "",
        opened_by_event_id: str = "",
        opened_by_speech_act: str = "",
        explicit_question_asked: bool = True,
        can_accept_no_wake_followup: bool = False,
        can_accept_emote_followup: bool = False,
        ttl_seconds: float | None = None,
        max_attempts: int | None = None,
        compatible_intents: list[str] | None = None,
        incompatible_intents: list[str] | None = None,
        conversation_source: str = "",
        creation_reason: str = "pending_created",
        **extra,
    ) -> CurrentConversation:
        now = time.time()
        ttl = float(ttl_seconds if ttl_seconds is not None else os.getenv("HEBE_PENDING_TASK_TTL_SECONDS", "900") or 900)
        attempts_max = int(max_attempts if max_attempts is not None else 1)
        if authority_required != "owner":
            raise ValueError("pending conversations require owner authority")
        reply_type = ExpectedReplyType(expected_reply_type)
        sources = tuple(
            "owner_stt" if item in {"stt_voice", "owner_stt", "voice"} else
            "owner_ui" if item in {"ui", "typed_ui", "owner_ui"} else str(item)
            for item in (allowed_sources or ["stt_voice", "ui"])
        )
        domain = {
            "capability_needed": capability_needed,
            "opened_by_speech_act": opened_by_speech_act,
            "explicit_question_asked": bool(explicit_question_asked),
            "can_accept_no_wake_followup": bool(can_accept_no_wake_followup),
            "can_accept_emote_followup": bool(can_accept_emote_followup),
            "max_attempts": attempts_max,
            "attempts": 0,
            "compatible_intents": compatible_intents or [],
            "incompatible_intents": incompatible_intents or [],
        }
        extra.pop("id", None)
        extra.pop("pending_id", None)
        domain.update(extra)
        if kind == "promotion_target_clarification":
            candidate_count = len(domain.get("candidates") or ())
            if candidate_count == 1:
                reply_type = ExpectedReplyType.YES_NO
            elif candidate_count > 1:
                reply_type = ExpectedReplyType.ENTITY_SELECTION
        service = getattr(self, "conversation_continuity", None)
        if service is None:
            raise RuntimeError("conversation_continuity_unavailable")
        context_kind, context_id = self._conversation_context_for_source(conversation_source)
        event = getattr(self, "_current_input_event", None)
        event_id = str(opened_by_event_id or getattr(event, "event_id", "") or f"pending_{uuid.uuid4().hex}")
        conversation = service.open_conversation(
            context_kind=context_kind, context_id=context_id, topic=kind, origin_event_id=event_id,
            expected_reply=ExpectedReply(
                type=reply_type, allowed_sources=tuple(dict.fromkeys(sources)),
                allowed_participant="leo", semantic_constraints={"min_words": 1, "max_words": 40},
                candidate_refs=tuple(str(item) for item in domain.get("candidates") or ()),
                expires_at=now + ttl,
            ),
            domain_payload=domain, reason=creation_reason,
        )
        print(
            "[HEBE][PENDING_CREATED] "
            f"kind={conversation.topic} id={conversation.id} "
            f"expected_reply_type={reply_type.value} reason={creation_reason}",
            flush=True,
        )
        return conversation

    def _update_current_conversation(
        self, conversation: CurrentConversation, *, domain_updates: dict | None = None,
        expires_at: float | None = None,
    ) -> CurrentConversation:
        return self.conversation_continuity.update_conversation(
            conversation, domain_updates=domain_updates, expires_at=expires_at,
        )

    def _increment_conversation_attempt(self, conversation: CurrentConversation | None, *, reason: str) -> bool:
        if conversation is None:
            return False
        domain = conversation.domain_payload
        maximum = max(1, int(domain.get("max_attempts") or 1))
        current = min(maximum, int(domain.get("attempts") or 0))
        attempts = min(maximum, current + 1)
        self._update_current_conversation(conversation, domain_updates={"attempts": attempts, "last_incompatible_reason": reason})
        print(
            f"[HEBE][PENDING_COMPATIBILITY] compatible=false reason={reason} "
            f"kind={conversation.topic} attempts={attempts}",
            flush=True,
        )
        if attempts >= maximum:
            print(
                f"[HEBE][PENDING_MAX_ATTEMPTS_REACHED] kind={conversation.topic} action=hold",
                flush=True,
            )
            return True
        return True

    def _normalize_promotion_pending_target_text(self, text: str) -> str:
        value = str(text or "").strip()
        if self._normalize_text(value) in {"el nuevo", "el de antes", "el que acaba de hablar"}:
            return value
        return re.sub(
            r"^\s*(?:al\s+canal\s+de|a\s+la|para|al|a|de|el|la)\s+",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()

    def _normalize_guard_text(self, text: str) -> str:
        lowered = str(text or "").casefold()
        lowered = "".join(ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch))
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_]+", " ", lowered)).strip()

    def _stream_game_planning_language(self, normalized: str) -> bool:
        tokens = set(str(normalized or "").split())
        stream_terms = {"stream", "directo", "twitch", "chat", "juego", "rpg", "combate", "partida", "campana"}
        planning_terms = {"traer", "terminamos", "terminar", "vamos", "voy", "atentos", "cerca", "combos"}
        return bool(tokens & stream_terms and tokens & planning_terms)

    def _appointment_pending_reply_compatible(self, normalized: str) -> bool:
        text = self._normalize_guard_text(normalized)
        if not text:
            return False
        if self._stream_game_planning_language(text):
            return False
        if re.search(r"\b(?:cancela|cancelar|olvida|nada|dejalo|anula)\b", text):
            return True
        appointment_marker = bool(re.search(r"\b(?:cita|recordatorio|evento|agenda|reunion|quedada)\b", text))
        concrete_time = bool(re.search(r"\b(?:a\s+las\s+)?\d{1,2}(?::\d{2})?\s*(?:h|am|pm)?\b", text))
        concrete_date = bool(
            re.search(r"\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\b", text)
            or re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text)
            or re.search(r"\b(?:hoy|manana|pasado manana)\b", text)
        )
        weekday_only = bool(re.search(r"\b(?:lunes|martes|miercoles|jueves|viernes|sabado|domingo)\b", text)) and not (concrete_time or appointment_marker)
        if weekday_only:
            return False
        correction = appointment_marker and bool(re.search(r"\b(?:mejor|cambia|corrige|era|ponlo|muevelo)\b", text))
        return bool((concrete_date and concrete_time) or correction or (appointment_marker and concrete_time))

    def _game_guidance_pending_compatibility(
        self,
        *,
        pending: CurrentConversation | None,
        normalized: str,
        raw_text: str,
        addressed: bool,
    ) -> tuple[bool, str]:
        if pending is None or pending.topic != "game_guidance_clarification":
            return False, "no_game_guidance_pending"
        domain = pending.domain_payload
        text = self._normalize_guard_text(normalized or raw_text)
        if not text:
            return False, "empty"
        try:
            age = time.time() - pending.opened_at
        except (TypeError, ValueError):
            age = 999999.0
        if age > min(float(os.getenv("HEBE_GAME_PENDING_COMPAT_TTL_SECONDS", "90") or 90), 180.0):
            return False, "pending_too_old"
        expected = pending.expected_reply.type.value if pending.expected_reply else ""
        if expected and expected not in {"game_progress_state", "game_party_or_character"}:
            return False, "unexpected_reply_type"
        ordinary_stream = bool(re.search(
            r"\b(?:en\s+plan|o\s+sea|nada|tranquilamente|familia|madre|padre|herman[oa]s?|"
            r"anime|artes?\s+marciales?|chat|viewer|espectador|ciber|nuria|rango\s+eh)\b",
            text,
        ))
        if ordinary_stream:
            return False, "ordinary_stream_or_real_life_talk"
        if re.fullmatch(r"(?:\d+|[a-z]{1,2})", text):
            return False, "too_short_or_numeric"
        explicit_progress = bool(re.search(
            r"\b(?:estoy|estamos|sigo|and[oa]|llegu[ea]?|llegando|entrando)\s+(?:en|a)\b|"
            r"\b(?:acabo|vengo|despues)\s+de\b|\b(?:toca|objetivo|mision|quest|jefe|boss|evento|zona|"
            r"palacio|castillo|mazmorra|dungeon|templo|nivel|capitulo|acto|party|equipo|personaje|prota)\b",
            text,
        ))
        game = self._normalize_guard_text(domain.get("game") or "")
        known_game_context = bool(game and game in text)
        if not (explicit_progress or known_game_context):
            return False, "no_plausible_game_state"
        if not addressed and age > float(os.getenv("HEBE_GAME_PENDING_AMBIENT_GRACE_SECONDS", "25") or 25):
            return False, "ambient_followup_window_expired"
        return True, "explicit_game_progress"

    def _log_game_pending_compat(self, compatible: bool, reason: str, *, pending: CurrentConversation | None = None) -> None:
        if pending is not None and not compatible:
            self._increment_conversation_attempt(pending, reason=reason)
        print(
            "[HEBE][GAME_PENDING_COMPAT] "
            f"compatible={str(bool(compatible)).lower()} reason={reason}",
            flush=True,
        )

    def _promotion_pending_reply_compatible(self, raw_text: str, normalized: str, pending: CurrentConversation | None = None) -> tuple[bool, str, dict]:
        text = self._normalize_promotion_pending_target_text(str(raw_text or normalized or ""))
        marker = self._normalize_guard_text(text)
        def reject(reason: str, resolution: dict | None = None) -> tuple[bool, str, dict]:
            data = dict(resolution or {})
            target = data.get("target") or ""
            print(f"[HEBE][PROMOTION_TARGET_GUARD] accepted=false target={target!r} reason={reason}", flush=True)
            print(f"[HEBE][PROMOTION_REJECTED] reason={reason}", flush=True)
            return False, reason, data

        def accept(reason: str, resolution: dict) -> tuple[bool, str, dict]:
            print(
                f"[HEBE][PROMOTION_TARGET_GUARD] accepted=true target={resolution.get('target')!r} reason={reason}",
                flush=True,
            )
            return True, reason, resolution

        if not marker:
            return reject("empty_target")
        if re.fullmatch(r"[a-z]", marker):
            return reject("single_letter_target")
        if marker in {"h", "hache"} or re.fullmatch(r"(?:a|al|a la|a el)\s+h(?:ache)?", marker):
            return reject("ambiguous_single_letter_target")
        if re.search(r"\b(?:idiot|imbecil|gilipoll|cabron|tont[oa]|estupid[oa])\b", marker):
            return reject("invalid_target")
        if re.search(r"\b(?:juego|partida|jueves|familia|anime|chat|viewer|espectador|combate|directo)\b", marker):
            return reject("stream_monologue")
        stream = self._get_stream_state()
        planner = self._get_stream_action_planner()
        resolver = getattr(planner, "_resolve_target", None)
        if not callable(resolver):
            return reject("resolver_unavailable")
        target, confidence, candidates, reason = resolver(text)
        resolution = {
            "target": target,
            "confidence": confidence,
            "candidates": candidates,
            "reason": reason,
        }
        print(
            "[HEBE][PROMOTION_RESOLVE] "
            f"candidates={candidates!r} selected={target!r} confidence={float(confidence or 0.0):.3f} reason={reason}",
            flush=True,
        )
        previous_candidates = list(pending.domain_payload.get("candidates") or []) if pending else []
        explicit_yes = marker in {"si", "si ese", "ese", "esa", "correcto", "exacto"}
        if explicit_yes and len(previous_candidates) == 1:
            resolution["target"] = previous_candidates[0]
            resolution["reason"] = "single_previous_candidate_confirmed"
            resolution["confidence"] = 0.95
            return accept("single_candidate_confirmation", resolution)
        if reason in {"ambiguous_single_letter_target", "ambiguous_target", "missing_target", "target_unclear", "invalid_target"}:
            return reject(reason, resolution)
        active_users = {str(name).lower() for name in getattr(stream, "recent_active_users", []) or []} if stream is not None else set()
        if target and str(target).lower() in active_users:
            return accept("recent_active_chatter", resolution)
        if target and float(confidence or 0.0) >= 0.86:
            return accept(reason or "resolved_target", resolution)
        if target and stream is not None and str(target).lower() in {str(name).lower() for name in getattr(stream, "recent_active_users", []) or []}:
            return accept("recent_active_chatter", resolution)
        if len(marker.split()) > 3 and not target:
            return reject("sentence_fragment", resolution)
        return reject(reason or "low_confidence_target", resolution)


    def _owner_mute_command_mode(self, normalized: str) -> str | None:
        text = self._normalize_guard_text(normalized)
        if not text:
            return None
        text = re.sub(r"^(?:hebe|ebe|eve|heve|jebe|eb|e b)\s+", "", text).strip()
        words = text.split()
        if len(words) > 8:
            return None
        silence_intent = bool(
            re.search(r"^(?:calla|callate|silencio|quieta)\b", text)
            or re.search(r"^(?:deja|para|deten)\w*\s+de\s+hablar\b", text)
            or re.search(r"^no\s+hables\s+sola\b", text)
            or re.search(r"^solo\s+responde\s+(?:si|cuando)\b", text)
        )
        if not silence_intent:
            return None
        hard = bool(re.search(r"\b(?:callate|silencio|muted?|mute)\b", text))
        return "muted" if hard else "wake_only"

    def _owner_unmute_command(self, normalized: str) -> bool:
        text = self._normalize_guard_text(normalized)
        return bool(re.search(r"\b(?:vuelve|puedes|reactiva|activa)\w*\s+(?:a\s+)?(?:hablar|responder)\b", text))

    def _apply_owner_mute_command(
        self,
        mode: str,
        *,
        ttl: float = 300.0,
        reason: str = "owner_mute",
        activated_by_text: str = "",
        activated_by_event_id: str = "",
        manual: bool = True,
    ) -> None:
        stream = self._get_stream_state()
        now = time.time()
        expires_at = now + float(ttl or 0.0) if ttl and ttl > 0 else 0.0
        if stream is not None:
            stream.stream_voice_mode = mode if mode in {"wake_only", "muted"} else "wake_only"
            stream.wake_only_until = max(float(getattr(stream, "wake_only_until", 0.0) or 0.0), expires_at)
            if stream.stream_voice_mode == "muted":
                stream.muted_until = max(float(getattr(stream, "muted_until", 0.0) or 0.0), expires_at)
            stream.mute_reason = reason
            stream.voice_mode_activated_by_event_id = activated_by_event_id or None
            stream.voice_mode_activated_by_text = str(activated_by_text or "")[:240] or None
            stream.voice_mode_activated_at = now
            stream.voice_mode_expires_at = expires_at
            stream.voice_mode_ttl_seconds = float(ttl or 0.0)
            stream.voice_mode_can_direct_wake_bypass = True
            stream.voice_mode_can_twitch_boundary_bypass = True
            stream.voice_mode_manual = bool(manual)
            stream.idle_spontaneity_enabled = False
        self._clear_noncritical_pending_for_mute()
        self._cancel_tts(reason=reason)
        safe_text = str(activated_by_text or "").replace("\n", " ")[:120]
        print(f"[HEBE][OWNER_MUTE_COMMAND] mode={mode} text={safe_text!r} ttl={int(ttl)}", flush=True)
        print(f"[HEBE][VOICE_MODE] mode={mode} expires_at={expires_at:.3f} reason={reason}", flush=True)
        print("[HEBE][PROACTIVE_SUPPRESSED] reason=owner_mute", flush=True)

    def _clear_owner_mute_command(self, *, reason: str = "owner_unmute") -> None:
        stream = self._get_stream_state()
        if stream is not None:
            stream.stream_voice_mode = "normal"
            stream.wake_only_until = 0.0
            stream.muted_until = 0.0
            stream.mute_reason = None
            stream.voice_mode_activated_by_event_id = None
            stream.voice_mode_activated_by_text = None
            stream.voice_mode_activated_at = 0.0
            stream.voice_mode_expires_at = 0.0
            stream.voice_mode_ttl_seconds = 0.0
            stream.voice_mode_manual = False
        print(f"[HEBE][VOICE_MODE_CLEARED] reason={reason}", flush=True)
        print("[HEBE][VOICE_MODE] mode=normal expires_at=0 reason=cleared", flush=True)

    def _cancel_tts(self, *, reason: str) -> None:
        cancel = getattr(getattr(self.runtime, "tts", None), "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception as exc:
                print(f"[HEBE][TTS_CANCEL] reason={reason} failed={exc!r}", flush=True)
                return
        print(f"[HEBE][TTS_CANCEL] reason={reason}", flush=True)

    def _clear_noncritical_pending_for_mute(self) -> None:
        pending = self._active_current_conversation(latest=True)
        if pending is not None and pending.topic not in {"promotion_target_clarification", "appointment_datetime"}:
            self._close_current_conversation(reason="owner_cancel", conversation=pending)

    def _stream_voice_mode_active(self) -> tuple[str, str]:
        stream = self._get_stream_state()
        if stream is None:
            return "normal", ""
        now = time.time()
        mode = str(getattr(stream, "stream_voice_mode", "normal") or "normal")
        if mode == "muted" and float(getattr(stream, "muted_until", 0.0) or 0.0) > now:
            return "muted", str(getattr(stream, "mute_reason", "") or "owner_mute")
        if mode in {"muted", "wake_only"} and float(getattr(stream, "wake_only_until", 0.0) or 0.0) > now:
            return "wake_only", str(getattr(stream, "mute_reason", "") or "owner_mute")
        if mode in {"muted", "wake_only"}:
            previous = mode
            self._clear_owner_mute_command(reason="expired")
            print(f"[HEBE][VOICE_MODE_EXPIRED] previous={previous}", flush=True)
        return "normal", ""

    def _live_owner_speech_gate(self, *, addressed: bool, pending_compatible: bool, conversation_followup: bool) -> tuple[str, str]:
        def decide(action: str, reason: str) -> tuple[str, str]:
            print(
                "[HEBE][PRESENCE_SHADOW] "
                f"source=owner_stt would_intervene={str(action == 'reply').lower()} current_route={action} reason={reason}",
                flush=True,
            )
            print(
                f"[HEBE][INTERVENTION_DECISION] source=owner_stt route={action} reason={reason}",
                flush=True,
            )
            return action, reason

        if not (self._is_stream_enabled() and self._current_stream_is_live()):
            return decide("reply" if (addressed or pending_compatible or conversation_followup) else "context_only", "not_live")
        mode, mode_reason = self._stream_voice_mode_active()
        if mode == "muted":
            if addressed:
                print("[HEBE][VOICE_MODE_BYPASS] reason=direct_owner_wake", flush=True)
                return decide("reply", "direct_wake_allowed_while_muted")
            return decide("ignore", mode_reason or "owner_mute")
        if mode == "wake_only" and not addressed:
            return decide("context_only", mode_reason or "wake_only")
        if addressed:
            return decide("reply", "wake_or_addressed")
        if pending_compatible:
            return decide("reply", "strict_pending_compatible")
        return decide("context_only", "live_stream_owner_monologue")

    def _output_dedupe_suppressed(self, *, text: str, source: str = "", message_id: str | None = None, input_id: str | None = None) -> tuple[bool, str]:
        now = time.time()
        norm = self._normalize_guard_text(text)
        recent = [
            item for item in list(getattr(self, "_recent_assistant_outputs", []) or [])
            if now - float(item.get("ts", 0.0) or 0.0) <= 6.0
        ]
        for item in recent:
            if message_id and item.get("message_id") == message_id:
                self._recent_assistant_outputs = recent
                return True, "same_message_id"
            if input_id and item.get("input_id") == input_id and norm and item.get("norm") == norm:
                self._recent_assistant_outputs = recent
                return True, "same_input"
            text_dedupe_allowed = source not in {"twitch_system", "twitch_viewer", "twitch"}
            if text_dedupe_allowed and norm and item.get("norm") == norm and source == item.get("source"):
                self._recent_assistant_outputs = recent
                return True, "same_text"
            if text_dedupe_allowed and norm and item.get("norm") and SequenceMatcher(None, norm, str(item.get("norm"))).ratio() >= 0.96 and source == item.get("source"):
                self._recent_assistant_outputs = recent
                return True, "same_text"
        recent.append({"ts": now, "norm": norm, "source": source, "message_id": message_id or "", "input_id": input_id or ""})
        self._recent_assistant_outputs = recent[-30:]
        return False, ""

    def _remember_assistant_text(self, text: str, *, source: str = "") -> None:
        value = str(text or "").strip()
        if value:
            self._last_assistant_text = value
            self._last_assistant_source = source

    def _target_speaker_guard(self, text: str, *, source: str, speaker: str = "") -> tuple[bool, str]:
        normalized = self._normalize_guard_text(text)
        source = str(source or "")
        if source == "twitch_viewer":
            owner_addressed = bool(
                re.match(r"^\s*leo\s*(?:[,;:!?.-]|$)", str(text or ""), re.IGNORECASE)
                or re.search(r"\b(?:dile|avisa|cuenta|pasa|manda|recuerda)\w*\s+a\s+leo\b", normalized)
                or re.search(r"\b(?:se\s+lo\s+(?:digo|dire|cuento|paso)|le\s+(?:digo|dire|cuento|paso)\s+a\s+leo)\b", normalized)
                or re.search(r"\bleo\b.*\b(?:lee|mira|haz|ven|contesta|responde)\b", normalized)
            )
            if owner_addressed:
                print("[HEBE][TARGET_SPEAKER_GUARD] passed=false reason=viewer_answer_addressed_to_owner", flush=True)
                return False, "viewer_answer_addressed_to_owner"
        print("[HEBE][TARGET_SPEAKER_GUARD] passed=true reason=ok", flush=True)
        return True, "ok"

    def _is_translate_previous_response_intent(self, normalized: str) -> bool:
        text = self._normalize_guard_text(normalized)
        wants_translation = bool(re.search(r"\b(?:traduce|traducelo|traducir|translation|translate)\b", text))
        wants_english = bool(re.search(r"\b(?:ingles|english)\b", text))
        same_previous = bool(re.search(r"\b(?:lo|eso|misma|mismo|respuesta|anterior)\b", text))
        return bool((wants_translation or wants_english) and same_previous)

    def _handle_translate_previous_response(self, command: str, *, source: str) -> bool:
        if not self._is_translate_previous_response_intent(command):
            return False
        previous = str(getattr(self, "_last_assistant_text", "") or "").strip()
        if not previous:
            self._deliver_manual_reply("No tengo una respuesta anterior clara que traducir.", source=source)
            return True
        print("[HEBE][TRANSLATE_PREVIOUS_RESPONSE] source=last_assistant_text", flush=True)
        # Keep the binding safe: the source content is the previous assistant text, never the command.
        translated = f"In English: {previous}"
        self._deliver_manual_reply(translated, source=source)
        return True

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
        raw_text = str(text or "")
        mentions_hebe = bool(self._message_mentions_hebe(raw_text)) if raw_text else False
        normalized_text = self._normalize_guard_text(raw_text)
        payload.update({
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "mentions_hebe": mentions_hebe,
            "addressed_to_hebe": mentions_hebe,
            "direct_address_to_hebe": mentions_hebe,
            "talks_about_hebe": bool(self._viewer_talks_about_hebe(raw_text)) if raw_text else False,
        })
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
        self._record_policy_interaction_decision(clean)
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
        style_profile: str = "",
        blocked_behavior: str = "",
        tts_route: dict | None = None,
        speech_budget: dict | None = None,
        quality_guard: dict | None = None,
        candidate_response: str | None = None,
        final_response: str | None = None,
        suppressed_response: str | None = None,
        suppress_reason: str | None = None,
        output_route: str | None = None,
        public_sent: bool | None = None,
        tts_sent: bool | None = None,
        reply_value_score: float | None = None,
        budget_result: dict | None = None,
        target_speaker_guard_result: dict | None = None,
        stream_persona_quality_result: dict | None = None,
        twitch_message_category: str | None = None,
        should_generate: bool | None = None,
        thread_result: dict | None = None,
        answer_depth_result: dict | None = None,
        followup_question_guard_result: dict | None = None,
        generation_outcome: str | None = None,
        emission_outcome: str | None = None,
        explanation_source_trace: str | None = None,
        explanation_reason_code: str | None = None,
    ) -> None:
        trace = self.get_last_policy_trace()
        if not trace:
            return
        updated = dict(trace)
        updated["hebe_response"] = str(reply_text or "")
        updated["final_response"] = str(final_response if final_response is not None else (reply_text or ""))
        updated["response_mode"] = response_mode
        updated["response_source"] = response_source
        updated["style_guard_triggered"] = bool(style_guard_triggered)
        updated["was_generic_refusal_rewritten"] = bool(was_generic_refusal_rewritten)
        if candidate_response is not None:
            updated["candidate_response"] = str(candidate_response or "")
            updated["generated_candidate"] = str(candidate_response or "")
        if suppressed_response is not None:
            updated["suppressed_response"] = str(suppressed_response or "")
        if suppress_reason is not None:
            updated["suppress_reason"] = str(suppress_reason or "")
        if output_route is not None:
            updated["output_route"] = str(output_route or "")
        if public_sent is not None:
            updated["public_sent"] = bool(public_sent)
            updated["public_chat_sent"] = bool(public_sent)
        if tts_sent is not None:
            updated["tts_sent"] = bool(tts_sent)
        if reply_value_score is not None:
            updated["reply_value_score"] = float(reply_value_score)
        if isinstance(budget_result, dict):
            updated["budget_result"] = budget_result
        if isinstance(target_speaker_guard_result, dict):
            updated["target_speaker_guard_result"] = target_speaker_guard_result
        if isinstance(stream_persona_quality_result, dict):
            updated["stream_persona_quality_result"] = stream_persona_quality_result
        if twitch_message_category is not None:
            updated["twitch_message_category"] = str(twitch_message_category or "")
        if should_generate is not None:
            updated["should_generate"] = bool(should_generate)
        if isinstance(thread_result, dict):
            updated["thread_result"] = thread_result
        if isinstance(answer_depth_result, dict):
            updated["answer_depth_result"] = answer_depth_result
        if isinstance(followup_question_guard_result, dict):
            updated["followup_question_guard_result"] = followup_question_guard_result
        if generation_outcome is not None:
            updated["generation_outcome"] = str(generation_outcome or "")
        if emission_outcome is not None:
            updated["emission_outcome"] = str(emission_outcome or "")
        if explanation_source_trace is not None:
            updated["explanation_source_trace"] = str(explanation_source_trace or "")
        if explanation_reason_code is not None:
            updated["explanation_reason_code"] = str(explanation_reason_code or "")
        if style_profile:
            updated["style_profile"] = style_profile
        if blocked_behavior:
            updated["blocked_behavior"] = blocked_behavior
        if isinstance(tts_route, dict):
            updated["tts_route"] = tts_route
        if isinstance(speech_budget, dict):
            updated["speech_budget"] = speech_budget
            updated["speech_budget_reason"] = speech_budget.get("reason")
        if isinstance(quality_guard, dict):
            updated["quality_guard"] = quality_guard
            updated["quality_guard_result"] = quality_guard.get("result")
        self._last_policy_trace = updated
        stream = self._get_stream_state()
        if stream is not None:
            try:
                stream.last_policy_trace = updated
            except Exception:
                pass
        self._record_policy_interaction_decision(updated)
        print(f"[HEBE][RESPONSE_SOURCE] source={response_source}", flush=True)

    def _enrich_stream_payload(self, payload: dict | None) -> dict:
        data = dict(payload or {})
        stream = self._get_stream_state()
        game = ""
        activity = ""
        title = ""
        stream_live = False
        if stream is not None:
            game = str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or "").strip()
            activity = str(getattr(stream, "current_activity", None) or getattr(stream, "current_run_phase", None) or "").strip()
            title = str(getattr(stream, "current_title", None) or getattr(stream, "title", None) or "").strip()
            stream_live = bool(
                getattr(stream, "is_live", False)
                or (getattr(stream, "enabled", False) and not getattr(stream, "live_status_known", False))
            )
        data.setdefault("current_game", game)
        data.setdefault("current_category", game)
        data.setdefault("current_activity", activity)
        data.setdefault("title", title)
        data.setdefault("stream_live", stream_live)
        data.setdefault("stream_output_mode", self._stream_output_mode())
        run = GameRunState.from_value(getattr(self.runtime.state, "game_run_state", None))
        data.setdefault("game_run_state", run.to_dict())
        return data

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
        deterministic = str(getattr(decision, "direct_template_response", "") or "").strip()
        if deterministic:
            return {"text": deterministic, "response_source": "structured_constraint_confirmation"}
        directive = str(getattr(decision, "response_directive", "") or "").strip()
        if not directive and not str(getattr(decision, "response_intent", "") or "").endswith("boundary"):
            return {"text": "", "response_source": "silent"}
        policy_payload = {
            "policy_decision": "blocked",
            "reason": decision.reason,
            "intent": decision.intent,
            "requested_behavior": decision.requested_behavior,
            "behavior_family": decision.behavior_family,
            "blocked_behavior": decision.requested_behavior or decision.behavior_family or decision.reason,
            "target": decision.target,
            "response_intent": decision.response_intent or "hebe_playful_boundary",
            "response_tone": decision.response_tone or "sarcastic_playful_stream_safe",
            "must_include": list(getattr(decision, "must_include", []) or []),
            "must_not_include": list(getattr(decision, "must_not_include", []) or []),
            "boundary_repeat_count": int(getattr(decision, "boundary_repeat_count", 0) or 0),
        }
        synthesizer = getattr(self, "response_synthesizer", None)
        if synthesizer is None or not hasattr(synthesizer, "synthesize_policy_boundary_response"):
            synthesizer = ResponseSynthesizer(conversation_model=None)
        stream_payload = self._enrich_stream_payload({})
        return synthesizer.synthesize_policy_boundary_response(
            policy=policy_payload,
            input_text=input_text,
            speaker=speaker,
            source=source,
            current_game=str(stream_payload.get("current_game") or ""),
            current_activity=str(stream_payload.get("current_activity") or ""),
            stream_live=bool(stream_payload.get("stream_live")),
            output_mode=str(stream_payload.get("stream_output_mode") or ""),
        )

    def get_last_policy_trace(self) -> dict:
        stream = self._get_stream_state()
        stream_trace = getattr(stream, "last_policy_trace", None) if stream is not None else None
        return dict(stream_trace or getattr(self, "_last_policy_trace", {}) or {})

    def _get_interaction_decision_history(self) -> RecentInteractionDecisionHistory:
        history = getattr(self, "interaction_decision_history", None)
        if history is None:
            history = RecentInteractionDecisionHistory()
            self.interaction_decision_history = history
        return history

    def get_recent_interaction_decisions(self) -> list[dict]:
        return self._get_interaction_decision_history().all(self._get_stream_state())

    def _record_interaction_decision(self, record: dict) -> dict:
        stream = self._get_stream_state()
        if stream is None:
            return dict(record or {})
        saved = self._get_interaction_decision_history().upsert(stream, record)
        print(
            "[HEBE][INTERACTION_DECISION] "
            f"trace={saved.get('trace_id')} actor={saved.get('actor')} authority={saved.get('authority')} "
            f"decision={saved.get('interaction_decision')} requested_effect={saved.get('requested_effect')} "
            f"effect_authorized={str(bool(saved.get('effect_authorized'))).lower()} "
            f"reply_authorized={str(bool(saved.get('reply_authorized'))).lower()} "
            f"reason={saved.get('reason_code')}",
            flush=True,
        )
        return saved

    def _record_policy_interaction_decision(self, trace: dict) -> dict:
        event_id = str(trace.get("event_id") or f"policy_{uuid.uuid4().hex}")
        actor = str(trace.get("speaker") or "unknown")
        identities = [actor, str(trace.get("actor_login") or "")]
        existing = next((
            item for item in reversed(self.get_recent_interaction_decisions())
            if str(item.get("trace_id") or "") == event_id
        ), {})
        emission_outcome = str(trace.get("emission_outcome") or "pending")
        if emission_outcome == "pending" and str(existing.get("emission_outcome") or "pending") != "pending":
            emission_outcome = str(existing.get("emission_outcome"))
        return self._record_interaction_decision({
            "trace_id": event_id,
            "event_id": event_id,
            "source": str(trace.get("source") or ""),
            "actor": actor,
            "actor_identities": [item for item in identities if item],
            "target": str(trace.get("target") or "Hebe"),
            "interaction_decision": str(trace.get("interaction_decision") or "observed"),
            "authority": str(trace.get("authority") or "unknown"),
            "requested_effect": str(trace.get("requested_effect") or trace.get("requested_behavior") or ""),
            "effect_authorized": bool(trace.get("effect_authorized")),
            "reply_authorized": bool(trace.get("reply_authorized", trace.get("allow_reply"))),
            "reason_code": str(trace.get("reason") or "none"),
            "response_intent": str(trace.get("response_intent") or ""),
            "generation_outcome": str(trace.get("generation_outcome") or "not_attempted"),
            "emission_outcome": emission_outcome,
            "boundary_repeat_count": int(trace.get("boundary_repeat_count") or 0),
        })

    def _record_response_generation_outcome(self, event_id: str, *, reply_text: str) -> None:
        trace_id = str(event_id or "").strip()
        if not trace_id:
            return
        synthesizer = getattr(self, "response_synthesizer", None)
        source = str(getattr(synthesizer, "last_response_source", "") or "")
        debug = dict(getattr(synthesizer, "last_response_debug_contract", {}) or {})
        recovery = dict(debug.get("directed_viewer_recovery") or {})
        generation_outcome = "generated" if reply_text else "failed"
        reason_code = None
        if recovery.get("generation_outcome") == "failed" or source == "directed_viewer_terminal_fallback":
            generation_outcome = "failed_terminal_fallback"
            reason_code = "directed_viewer_generation_failed"
        elif source in {"local_safe_fallback", "fallback_template"}:
            generation_outcome = "fallback_template"
        updated = self._get_interaction_decision_history().update(
            self._get_stream_state(),
            trace_id,
            generation_outcome=generation_outcome,
            reason_code=reason_code,
            response_intent=str((debug.get("speech_act_plan") or {}).get("speech_act_type") or "direct_answer"),
        )
        policy_trace_value = self.get_last_policy_trace()
        if str(policy_trace_value.get("event_id") or "") == trace_id:
            policy_trace_value["generation_outcome"] = generation_outcome
            if reason_code:
                policy_trace_value["reason"] = reason_code
            self._last_policy_trace = policy_trace_value
            stream = self._get_stream_state()
            if stream is not None:
                stream.last_policy_trace = policy_trace_value
        if updated is not None:
            print(
                f"[HEBE][INTERACTION_GENERATION] trace={trace_id} outcome={generation_outcome} "
                f"reason={updated.get('reason_code')}",
                flush=True,
            )

    def _grounded_self_explanation(
        self,
        text: str,
        *,
        requester: str,
        current_trace_id: str,
    ):
        stream = self._get_stream_state()
        history = self._get_interaction_decision_history()
        known = {
            str(item.get("actor") or "")
            for item in history.all(stream)
            if str(item.get("actor") or "")
        }
        social_world = getattr(self, "social_world", None)
        if social_world is not None and hasattr(social_world, "recent_identity_names"):
            try:
                known.update(str(item) for item in social_world.recent_identity_names(limit=80) if str(item))
            except Exception as exc:
                print(f"[HEBE][SELF_EXPLANATION] identity_lookup_failed={type(exc).__name__}", flush=True)
        query = detect_self_explanation_query(text, requester=requester, known_identities=known)
        if not query.detected:
            return render_grounded_self_explanation(query, None, requester=requester)
        matched = history.resolve(stream, query, exclude_trace_id=current_trace_id)
        result = render_grounded_self_explanation(query, matched, requester=requester)
        explanation_trace = {
            **result.to_dict(),
            "trace_id": current_trace_id,
            "requester": requester,
            "explanation_source_trace": result.source_trace_id,
            "explanation_reason_code": result.reason_code,
            "timestamp": time.time(),
        }
        if stream is not None:
            stream.last_self_explanation = explanation_trace
        current_record = self._record_interaction_decision({
            "trace_id": current_trace_id,
            "event_id": current_trace_id,
            "source": "self_explanation",
            "actor": requester or "Leo",
            "actor_identities": [requester or "Leo"],
            "target": "Hebe",
            "interaction_decision": "grounded_self_explanation",
            "authority": "owner" if str(requester or "").casefold() == "leo" else "viewer",
            "requested_effect": "self_explanation",
            "effect_authorized": False,
            "reply_authorized": True,
            "reason_code": result.reason_code,
            "response_intent": "grounded_self_explanation",
            "generation_outcome": "deterministic_grounded_renderer",
            "emission_outcome": "pending",
            "explanation_source_trace": result.source_trace_id,
            "explanation_reason_code": result.reason_code,
        })
        self._last_policy_trace = {
            "event_id": current_trace_id,
            "source": "self_explanation",
            "speaker": requester or "Leo",
            "actor_login": requester or "Leo",
            "authority": current_record.get("authority"),
            "addressed_to_hebe": True,
            "text": text,
            "intent": "grounded_self_explanation",
            "requested_behavior": "self_explanation",
            "requested_effect": "self_explanation",
            "target": "Hebe",
            "policy_decision": "allowed",
            "interaction_decision": "grounded_self_explanation",
            "effect_authorized": False,
            "reply_authorized": True,
            "allow_reply": True,
            "allow_llm": False,
            "allow_free_llm": False,
            "reason": result.reason_code,
            "response_intent": "grounded_self_explanation",
            "response_source": "deterministic_grounded_renderer",
            "generation_outcome": "deterministic_grounded_renderer",
            "emission_outcome": "pending",
            "explanation_source_trace": result.source_trace_id,
            "explanation_reason_code": result.reason_code,
            "hebe_response": "",
            "final_response": "",
        }
        if stream is not None:
            stream.last_policy_trace = dict(self._last_policy_trace)
        print(
            "[HEBE][SELF_EXPLANATION] "
            f"source_trace={result.source_trace_id or 'none'} reason={result.reason_code} "
            f"matched={str(result.matched).lower()}",
            flush=True,
        )
        return result

    def get_active_behavior_blocks(self) -> list[dict]:
        stream = self._get_stream_state()
        if stream is None:
            return []
        return list(active_behavior_blocks(stream))

    def get_behavior_calibration_snapshot(self) -> dict:
        stream = self._get_stream_state()
        if stream is None:
            return {
                "stream_session_id": "",
                "store_status": "unavailable",
                "active_current_stream": [],
                "active_durable": [],
                "retired_durable_recent": [],
                "episodic_fatigue": [],
                "telemetry": self.behavior_adaptation.observability.snapshot(),
            }
        return self.behavior_adaptation.inspection_snapshot(stream)

    def label_behavior_calibration_trace(self, trace_id: str, label: str) -> dict:
        return self.behavior_adaptation.observability.label(trace_id, label)

    def get_stream_readiness_status(self, *, error_count_last_10m: int = 0, vts_status: dict | None = None) -> dict:
        stream = self._get_stream_state()
        policies = getattr(stream, "policies", None) if stream is not None else None
        trace = self.get_last_policy_trace()
        twitch = getattr(self.runtime, "twitch", None)
        twitch_connected = bool(
            (getattr(twitch, "is_available", lambda: False)() if twitch is not None else False)
            or getattr(getattr(self.runtime, "twitch_chat_bot", None), "is_connected", False)
        )
        presence = dict((trace.get("core_loop") or {}).get("intervention") or trace.get("presence_decision") or {})
        companion_loop = getattr(self, "stream_companion_loop", None)
        last_proactive = dict(getattr(stream, "last_proactive_decision", {}) or {}) if stream is not None else {}
        recent_idle = list(getattr(stream, "recent_idle_messages", []) or []) if stream is not None else []
        now_ts = time.time()
        cooldown_until = float((getattr(stream, "cooldowns", {}) or {}).get("stream_idle_prompt_next_ts", 0.0) or 0.0) if stream is not None else 0.0
        effective_audio = self._effective_stream_audio_state("owner_discourse_opportunity")
        discourse_topic = dict(getattr(stream, "current_discourse_topic", {}) or {}) if stream is not None else {}
        discourse_plan = dict(getattr(stream, "proposed_discourse_contribution", {}) or {}) if stream is not None else {}
        discourse_turn = dict(getattr(stream, "current_stream_turn", {}) or {}) if stream is not None else {}
        discourse_timestamps = [
            float(ts) for ts in list(getattr(stream, "discourse_contribution_timestamps", []) or [])
            if now_ts - float(ts) <= 3600
        ] if stream is not None else []
        readiness = {
            "backend_running": True,
            "stt_health": (
                self.runtime.stt.health_snapshot()
                if getattr(self.runtime, "stt", None) is not None and hasattr(self.runtime.stt, "health_snapshot")
                else {}
            ),
            "stream_tts_safety": {
                **self._get_stream_tts_safety().readiness(),
                "obs_live": bool(
                    getattr(stream, "is_live", False)
                    or getattr(stream, "obs_running", False)
                ) if stream is not None else False,
            },
            "twitch_connected": twitch_connected,
            "stream_live": bool(getattr(stream, "is_live", False)) if stream is not None else False,
            "stream_live_detected": bool(getattr(stream, "live_status_known", False)) if stream is not None else False,
            "vts_status": vts_status or {},
            "tts_enabled": bool(getattr(self.runtime.state, "tts_enabled", False)),
            "stream_tts_enabled": bool(policies and getattr(policies, "allow_tts_replies", False)),
            "effective_stream_audio_state": effective_audio.to_dict(),
            "stream_voice_mode": str(getattr(stream, "stream_voice_mode", "normal") if stream is not None else "normal"),
            "stream_voice_mode_expires_at": float(getattr(stream, "voice_mode_expires_at", 0.0) or 0.0) if stream is not None else 0.0,
            "stream_voice_mode_ttl_seconds": float(getattr(stream, "voice_mode_ttl_seconds", 0.0) or 0.0) if stream is not None else 0.0,
            "stream_voice_mode_reason": str(getattr(stream, "mute_reason", "") or "") if stream is not None else "",
            "last_owner_mute_activation_text": str(getattr(stream, "voice_mode_activated_by_text", "") or "") if stream is not None else "",
            "presence_engine_mode": str(getattr(self, "presence_engine_mode", "active") or "active"),
            "presence_mode": str(getattr(stream, "presence_mode", "reactive") if stream is not None else "reactive"),
            "proactive_speech_enabled": bool(policies and getattr(policies, "allow_tts_idle_prompts", False)),
            "spontaneous_twitch_chat_enabled": bool(getattr(self, "spontaneous_twitch_chat_enabled", False)),
            "raid_auto_thank_enabled": True,
            "auto_shoutout_raiders": bool(getattr(self, "auto_shoutout_raiders", False)),
            "recent_raid_context": self._stream_recent_raid_context(stream) if stream is not None else None,
            "last_raid_event": getattr(stream, "last_raid_event", None) if stream is not None else None,
            "last_raid_ack_result": getattr(stream, "last_raid_ack_result", None) if stream is not None else None,
            "last_raid_ack_error": getattr(stream, "last_raid_ack_error", None) if stream is not None else None,
            "last_cheer_event": getattr(stream, "last_cheer_event", None) if stream is not None else None,
            "last_cheer_ack_result": getattr(stream, "last_cheer_ack_result", None) if stream is not None else None,
            "last_cheer_dedupe_result": getattr(stream, "last_cheer_dedupe_result", None) if stream is not None else None,
            "discourse_participation_mode": str(getattr(stream, "discourse_participation_mode", "shadow") if stream is not None else "shadow"),
            "current_discourse_topic": discourse_topic,
            "current_discourse_topic_label": str(discourse_topic.get("label") or ""),
            "current_discourse_topic_family": str(discourse_topic.get("family") or ""),
            "current_discourse_topic_confidence": float(discourse_topic.get("confidence") or 0.0),
            "current_discourse_topic_duration": float(discourse_topic.get("duration_seconds") or 0.0),
            "current_discourse_owner_stance": str(discourse_topic.get("owner_stance") or ""),
            "buffered_discourse_fragment_count": len(discourse_topic.get("fragments") or []),
            "proposed_discourse_contribution": discourse_plan,
            "current_stream_turn": discourse_turn,
            "waiting_for_discourse_pause": bool(discourse_plan.get("wait_for_turn") and not discourse_turn.get("turn_available")),
            "last_discourse_contribution": getattr(stream, "last_discourse_contribution", None) if stream is not None else None,
            "last_discourse_blocked_reason": str(getattr(stream, "last_discourse_blocked_reason", "") or "") if stream is not None else "",
            "discourse_contributions_this_hour": len(discourse_timestamps),
            "last_promo_parse": getattr(stream, "last_promo_parse", None) if stream is not None else None,
            "last_promo_rejected_reason": str(getattr(stream, "last_promo_rejected_reason", "") or "") if stream is not None else "",
            "last_promo_execution_decision": getattr(stream, "last_promo_execution_decision", None) if stream is not None else None,
            "last_stream_event_ack_decision": getattr(stream, "last_stream_event_ack_decision", None) if stream is not None else None,
            "twitch_reply_consecutive_count": int(getattr(stream, "consecutive_public_replies", 0) or 0) if stream is not None else 0,
            "last_budget_reset_reason": str(getattr(stream, "last_twitch_reply_budget_reset_reason", "") or "") if stream is not None else "",
            "human_messages_since_last_public_reply": int(getattr(stream, "human_messages_since_last_public_reply", 0) or 0) if stream is not None else 0,
            "last_twitch_category": str((getattr(stream, "last_twitch_route_state", {}) or {}).get("category") or ""),
            "last_twitch_mentions_hebe": bool((getattr(stream, "last_twitch_route_state", {}) or {}).get("mentions_hebe")),
            "last_twitch_reply_to_hebe_message": bool((getattr(stream, "last_twitch_route_state", {}) or {}).get("reply_to_hebe_message")),
            "last_direct_priority_bypass": bool((getattr(stream, "last_twitch_route_state", {}) or {}).get("direct_priority_applied")),
            "last_budget_block_type": str(((getattr(stream, "last_twitch_route_state", {}) or {}).get("budget_result") or {}).get("block_type") or ""),
            "current_game": str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or "") if stream is not None else "",
            "active_conversations": [item.to_dict() for item in self.conversation_continuity.conversations.list_active()] if self.conversation_continuity else [],
            "active_behavior_blocks": self.get_active_behavior_blocks(),
            "last_output_route": str(trace.get("output_route") or trace.get("tts_route", {}).get("route") or ""),
            "last_tts_route": trace.get("tts_route") or {},
            "last_presence_decision": presence,
            "stream_companion_loop_running": bool(companion_loop is not None),
            "last_companion_tick_time": float(getattr(companion_loop, "last_tick_ts", 0.0) or 0.0) if companion_loop is not None else 0.0,
            "last_proactive_decision": last_proactive,
            "last_proactive_blocked_reason": str(last_proactive.get("blocked_reason") or ""),
            "last_proactive_emitted_response": str(last_proactive.get("final_response") or ""),
            "proactive_comments_this_hour": sum(
                1 for item in recent_idle
                if now_ts - float(item.get("timestamp", 0.0) or 0.0) <= 3600
            ),
            "current_proactive_cooldown": max(0.0, cooldown_until - now_ts),
            "last_anchor_type": str(last_proactive.get("anchor_type") or ""),
            "last_anchor_quality": float(last_proactive.get("anchor_quality") or 0.0),
            "last_suppression_reason": str(trace.get("suppress_reason") or ""),
            "twitch_reply_budget": trace.get("budget_result") or trace.get("twitch_write_decision", {}).get("budget_result") or {},
            "last_guard_violation": (
                trace.get("quality_guard", {}).get("violations")
                or trace.get("stream_persona_quality_result", {}).get("violations")
                or []
            ),
            "error_count_last_10m": int(error_count_last_10m or 0),
            "safe_defaults": {
                "presence_engine_active": str(getattr(self, "presence_engine_mode", "active") or "active") == "active",
                "owner_stt_without_wake_context_only": True,
                "twitch_replies_value_gated": True,
                "emote_only_observe": True,
                "pending_followup_strict_compatibility": True,
                "one_input_one_final_output": True,
                "game_advice_requires_validation": True,
            },
        }
        return {"ok": True, "stream_readiness": readiness, **readiness}

    def clear_active_behavior_blocks(self) -> list[dict]:
        stream = self._get_stream_state()
        if stream is None:
            return []
        stream.active_behavior_blocks = [
            item for item in self.behavior_adaptation.active_constraints(stream)
            if str(item.get("scope") or "current_stream") == "durable"
        ]
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
        return list(stream.active_behavior_blocks)

    def _simulation_stream_live_from_payload(self, payload: dict | None) -> tuple[bool | None, str]:
        payload = payload or {}
        mode = str(payload.get("stream_live_mode") or "").strip().lower()
        if bool(payload.get("force_stream_live")) or mode == "force_stream_live":
            return True, "force_stream_live"
        if bool(payload.get("force_stream_offline")) or mode == "force_stream_offline":
            return False, "force_stream_offline"
        if bool(payload.get("use_real_stream_state")) or mode == "use_real_stream_state":
            return None, "use_real_stream_state"
        if "stream_live" in payload:
            return bool(payload.get("stream_live")), "explicit_stream_live"
        return True, "force_stream_live"

    def simulate_twitch_message(self, payload: dict) -> dict:
        self._last_cognitive_trace = {}
        self._last_input_firewall = {}
        viewer_name = str((payload or {}).get("viewer_name") or (payload or {}).get("user_login") or (payload or {}).get("username") or "viewer").strip()
        display_name = str((payload or {}).get("display_name") or viewer_name).strip()
        text = str((payload or {}).get("text") or (payload or {}).get("message_text") or "").strip()
        channel = str((payload or {}).get("channel") or "").strip()
        simulated_stream_live, stream_live_mode = self._simulation_stream_live_from_payload(payload)
        event_payload = {
            **(payload or {}),
            "display_name": display_name,
            "user_login": viewer_name,
            "username": viewer_name,
            "message_text": text,
            "channel": channel,
            "_simulated": True,
            "stream_live_mode": stream_live_mode,
        }
        stream = self._get_stream_state()
        old_live = getattr(stream, "is_live", False) if stream is not None else False
        old_known = getattr(stream, "live_status_known", False) if stream is not None else False
        old_override = getattr(self, "_simulation_stream_live_override", None)
        if simulated_stream_live is not None:
            self._simulation_stream_live_override = bool(simulated_stream_live)
            if stream is not None:
                stream.is_live = bool(simulated_stream_live)
                stream.live_status_known = True
        else:
            self._simulation_stream_live_override = None
        if stream is not None and bool((payload or {}).get("seed_twitch_thread_closed")):
            category = self._classify_twitch_viewer_message(text, payload=event_payload)
            thread_id = self._twitch_thread_id(username=viewer_name, text=text, category=category)
            counts = dict(getattr(stream, "public_reply_thread_counts", {}) or {})
            counts[thread_id] = max(2, int(counts.get(thread_id, 0) or 0))
            stream.public_reply_thread_counts = counts
        if stream is not None and bool((payload or {}).get("seed_twitch_minute_budget_full")):
            now = time.time()
            stream.public_reply_timestamps = [now - 5, now - 10, now - 15, now - 20, now - 25]
        try:
            self.ingest_normalized_twitch_chat(
                username=viewer_name,
                display_name=display_name,
                text=text,
                channel=channel,
                irc_tags=dict((payload or {}).get("irc_tags") or {}),
                normalized_fields=event_payload,
            )
            firewall = self._firewall_payload()
            return self._simulation_debug_payload(extra={
                "simulated_stream_live": bool(firewall.get("stream_is_live")) if firewall else bool(self._current_stream_is_live()),
                "stream_live_mode": stream_live_mode,
                "stream_live_used": bool(firewall.get("stream_is_live")) if firewall else bool(self._current_stream_is_live()),
            })
        finally:
            self._simulation_stream_live_override = old_override
            if stream is not None and simulated_stream_live is not None:
                stream.is_live = old_live
                stream.live_status_known = old_known

    def simulate_internal_twitch_event(self, *, event_type: str = "twitch_raid", stream_live: bool = False) -> dict:
        self._last_cognitive_trace = {}
        self._last_input_firewall = {}
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

    def simulate_leo_message(
        self,
        text: str,
        *,
        source: str = "ui",
        pending_kind: str | None = None,
        stream_live_mode: str | None = None,
        stream_voice_mode: str | None = None,
    ) -> dict:
        self._last_cognitive_trace = {}
        self._last_input_firewall = {}
        clean_source = source if source in {"ui", "stt_voice"} else "ui"
        simulated_stream_live, resolved_stream_live_mode = self._simulation_stream_live_from_payload({
            "stream_live_mode": stream_live_mode or "",
        })
        stream = self._get_stream_state()
        old_live = getattr(stream, "is_live", False) if stream is not None else False
        old_known = getattr(stream, "live_status_known", False) if stream is not None else False
        old_voice_mode = getattr(stream, "stream_voice_mode", "normal") if stream is not None else "normal"
        old_wake_until = getattr(stream, "wake_only_until", 0.0) if stream is not None else 0.0
        old_muted_until = getattr(stream, "muted_until", 0.0) if stream is not None else 0.0
        if pending_kind == "appointment_datetime":
            self._open_pending_conversation(
                kind="appointment_datetime",
                expected_reply_type="datetime",
                capability_needed="calendar.create",
                can_accept_no_wake_followup=True,
                ttl_seconds=300,
                conversation_source=clean_source,
                draft={"title": "Consulta", "source_text": "simulated appointment request"},
            )
        elif pending_kind == "promotion_target_clarification":
            self._open_pending_conversation(
                kind="promotion_target_clarification",
                expected_reply_type="twitch_username_or_viewer_alias",
                capability_needed="twitch.shoutout",
                explicit_question_asked=True,
                can_accept_no_wake_followup=True,
                ttl_seconds=300,
                max_attempts=1,
                conversation_source=clean_source,
            )
        elif pending_kind == "game_guidance_clarification":
            self._open_pending_conversation(
                kind="game_guidance_clarification",
                expected_reply_type="game_party_or_character",
                capability_needed="game.guidance",
                can_accept_no_wake_followup=True,
                ttl_seconds=300,
                conversation_source=clean_source,
                game="Final Fantasy VII",
                location_or_area="Midgar",
                missing_fields=["current_character", "party_members", "story_phase", "recent_event"],
                original_question="Necesito orientación de progreso en la zona actual de FFVII.",
                source=clean_source,
                spoiler_policy="no_story_spoilers",
            )
        before_event_id = self.get_last_policy_trace().get("event_id")
        previous_simulation_mode = bool(getattr(self, "_manual_simulation_mode", False))
        self._manual_simulation_mode = True
        if stream is not None and simulated_stream_live is not None:
            stream.is_live = bool(simulated_stream_live)
            stream.live_status_known = True
        if stream is not None and stream_voice_mode in {"normal", "wake_only", "muted"}:
            stream.stream_voice_mode = stream_voice_mode
            if stream_voice_mode == "wake_only":
                stream.wake_only_until = time.time() + 300
            elif stream_voice_mode == "muted":
                stream.muted_until = time.time() + 300
        try:
            if clean_source == "stt_voice":
                self.ingest_owner_stt(str(text or "").strip())
            else:
                self.cognitive_flow(str(text or "").strip(), source=clean_source)
        finally:
            self._manual_simulation_mode = previous_simulation_mode
            if stream is not None and simulated_stream_live is not None:
                stream.is_live = old_live
                stream.live_status_known = old_known
            if stream is not None and stream_voice_mode in {"normal", "wake_only", "muted"}:
                stream.stream_voice_mode = old_voice_mode
                stream.wake_only_until = old_wake_until
                stream.muted_until = old_muted_until
        after_event_id = self.get_last_policy_trace().get("event_id")
        if before_event_id == after_event_id:
            normalized = self._normalize_guard_text(text)
            stt_addressed = clean_source == "stt_voice" and bool(self._message_mentions_hebe(normalized))
            stt_context_only = clean_source == "stt_voice" and not stt_addressed
            self._record_policy_trace(policy_trace(
                source=clean_source,
                speaker="Leo",
                text=str(text or "").strip(),
                decision=PolicyDecision(
                    allow_reply=not stt_context_only,
                    allow_llm=not stt_context_only,
                    reason="owner_stt_context_only" if stt_context_only else "owner_allowed",
                    intent="owner_stream_monologue" if stt_context_only else "owner_message",
                ),
                addressed_to_hebe=not stt_context_only,
                authority="owner",
            ))
        return self._simulation_debug_payload(extra={
            "stream_live_mode": resolved_stream_live_mode,
            "stream_voice_mode": stream_voice_mode or old_voice_mode,
        })

    def simulate_ambient_stt(self, text: str) -> dict:
        self._last_cognitive_trace = {}
        self._last_input_firewall = {}
        result = self.ingest_ambient_stt(text)
        return self._simulation_debug_payload(extra={"relevance": result.get("relevance") or {}})

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
        response_debug = self._latest_response_debug_payload() if (trace.get("final_response") or trace.get("hebe_response")) else {}
        debug_contract = response_debug.get("debug_contract") if isinstance(response_debug, dict) else {}
        speech_act_plan = debug_contract.get("speech_act_plan") if isinstance(debug_contract, dict) else {}
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
            "policy_reason": trace.get("reason"),
            "blocked_behavior": trace.get("blocked_behavior") or trace.get("requested_behavior") or trace.get("behavior_family"),
            "style_profile": trace.get("style_profile") or ((speech_act_plan or {}).get("style_profile") if isinstance(speech_act_plan, dict) else ""),
            "response_mode": cognitive.get("response_mode") or trace.get("response_mode"),
            "response_source": trace.get("response_source"),
            "allow_free_llm": trace.get("allow_free_llm"),
            "execute_as_command": trace.get("execute_as_command"),
            "style_guard_triggered": trace.get("style_guard_triggered"),
            "was_generic_refusal_rewritten": trace.get("was_generic_refusal_rewritten"),
            "tts_route": trace.get("tts_route") or {},
            "speech_budget": trace.get("speech_budget") or {},
            "speech_budget_reason": trace.get("speech_budget_reason") or "",
            "quality_guard": trace.get("quality_guard") or {},
            "quality_guard_result": trace.get("quality_guard_result") or "",
            "twitch_message_category": trace.get("twitch_message_category") or "",
            "value_score": trace.get("reply_value_score"),
            "should_generate": trace.get("should_generate"),
            "output_route": trace.get("output_route") or "",
            "public_chat_sent": bool(trace.get("public_chat_sent") or trace.get("public_sent") or False),
            "tts_sent": bool(trace.get("tts_sent") or False),
            "budget_result": trace.get("budget_result") or {},
            "thread_result": trace.get("thread_result") or {},
            "answer_depth_result": trace.get("answer_depth_result") or {},
            "followup_question_guard_result": trace.get("followup_question_guard_result") or {},
            "stream_persona_quality_result": trace.get("stream_persona_quality_result") or {},
            "suppress_reason": trace.get("suppress_reason") or "",
            "hebe_response": trace.get("hebe_response") or "",
            "final_response": cognitive.get("final_response") or trace.get("final_response") or trace.get("hebe_response") or "",
            "last_policy_decision": trace,
            "cognitive_route": cognitive,
            "raw_input": cognitive.get("raw_text") or cognitive.get("input_text") or trace.get("text") or "",
            "normalized_input": cognitive.get("normalized_text") or self._normalize_text(trace.get("text") or ""),
            "current_conversation": cognitive.get("current_conversation") or cognitive.get("pending_task_id"),
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
            "last_twitch_route_state": dict(getattr(self, "_last_twitch_route_state", {}) or {}),
            "twitch_pipeline_health": dict(getattr(stream, "twitch_pipeline_health", {}) or {}) if stream is not None else {},
            "stream_live_used": bool((self._firewall_payload() or {}).get("stream_is_live")),
            "debug_contract": debug_contract if isinstance(debug_contract, dict) else {},
            "speech_act": (speech_act_plan or {}).get("speech_act_type") if isinstance(speech_act_plan, dict) else "",
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
        profile_decision = self._owner_viewer_profile_decision(command, source=source)
        if profile_decision is not None:
            self._record_policy_trace(policy_trace(
                source=source, speaker="Leo", text=command, decision=profile_decision,
                addressed_to_hebe=True, authority="owner",
            ))
            return profile_decision
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
        resolver = getattr(getattr(self.runtime, "twitch", None), "target_resolver", None)
        resolve_details = getattr(resolver, "resolve_user_details", None)
        def resolve_behavior_target(value: str):
            resolved = resolve_details(value, intent="behavior_constraint") if callable(resolve_details) else None
            login = str(getattr(resolved, "username", "") or "")
            profiles = (getattr(self, "viewer_linguistic_profiles", None) or ViewerLinguisticProfileStore()).list_profiles()
            normalized_value = self._normalize_guard_text(value)
            profile = next((item for item in profiles if normalized_value in {
                self._normalize_guard_text(item.get("login") or ""),
                self._normalize_guard_text(item.get("display_name") or ""),
            }), None)
            if login and profile and str(profile.get("login") or "").casefold() == login.casefold():
                return {"username": login, "display_name": profile.get("display_name") or login,
                        "user_id": profile.get("twitch_user_id") or "",
                        "confidence": max(float(getattr(resolved, "confidence", 0.0) or 0.0), .99),
                        "candidates": list(getattr(resolved, "candidates", []) or [login]), "reason": "viewer_profile"}
            if login:
                return resolved
            if profile:
                return {"username": profile.get("login") or "", "display_name": profile.get("display_name") or "",
                        "user_id": profile.get("twitch_user_id") or "", "confidence": .96,
                        "candidates": [profile.get("login")], "reason": "viewer_profile"}
            return resolved
        behavior_decision = owner_behavior_decision(
            stream, command,
            resolver=resolve_behavior_target,
            source_event_id=str(getattr(getattr(self, "_current_input_event", None), "timestamp", "") or ""),
            constraint_owner=getattr(self, "behavior_adaptation", None),
        )
        if not behavior_decision.allow_llm:
            block = dict(getattr(behavior_decision, "update_behavior_block", None) or {})
            if block.get("behavior_family") == "compliment" and block.get("recipient_scope") == "specific_viewer":
                self._get_troll_engagement_budget().close_topic_by_owner(
                    viewer=str(block.get("recipient_login") or block.get("recipient_display_name") or "viewer"),
                    topic="compliment_fishing",
                )
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

    def _owner_viewer_profile_decision(self, command: str, *, source: str) -> PolicyDecision | None:
        parser = getattr(self, "viewer_profile_command_parser", None) or ViewerProfileCommandParser()
        parsed = parser.parse(command)
        if not parsed.detected:
            return None
        resolver = getattr(getattr(self.runtime, "twitch", None), "target_resolver", None)
        resolve = getattr(resolver, "resolve_user_details", None)
        resolution = resolve(parsed.viewer_text, intent="viewer_profile") if callable(resolve) else None
        login = str(getattr(resolution, "username", "") or "")
        confidence = float(getattr(resolution, "confidence", 0.0) or 0.0)
        candidates = list(getattr(resolution, "candidates", []) or [])
        if not login or confidence < .78 or len(candidates) > 1 and confidence < .9:
            return PolicyDecision(
                allow_reply=True, allow_llm=False, allow_free_llm=False,
                reason="viewer_profile_target_clarification", intent="viewer_profile_command",
                response_directive="Ask Leo which viewer profile should be changed. Do not claim the profile was updated.",
                response_intent="owner_profile_clarification", response_tone="brief_owner_clarification",
                requested_behavior="viewer_linguistic_profile", target=parsed.viewer_text,
            )
        store = getattr(self, "viewer_linguistic_profiles", None) or ViewerLinguisticProfileStore()
        if parsed.action == "clear":
            store.clear(login=login)
            confirmation = f"Perfil lingüístico de {login} borrado."
        else:
            user_id = str(getattr(resolution, "user_id", "") or f"login:{login.casefold()}")
            store.apply_evidence(
                twitch_user_id=user_id, login=login, display_name=login,
                candidate_gender=parsed.gender, confidence=1.0, source_type="owner_confirmed",
                source_event_id=str(getattr(getattr(self, "_current_input_event", None), "timestamp", "") or ""),
                evidence_summary="explicit owner linguistic preference",
            )
            confirmation = f"Perfil lingüístico de {login} actualizado a {parsed.gender}."
        return PolicyDecision(
            allow_reply=True, allow_llm=False, allow_free_llm=False,
            reason="viewer_profile_updated", intent="viewer_profile_command",
            direct_template_response=confirmation,
            response_directive="Confirm the structured viewer linguistic profile update briefly.",
            response_intent="owner_profile_confirmation", response_tone="brief_owner_confirmation",
            requested_behavior="viewer_linguistic_profile", target=login,
        )

    def _viewer_policy_decision(self, payload: dict) -> PolicyDecision | None:
        stream = self._get_stream_state()
        if stream is None:
            return None
        cached = (payload or {}).get("_viewer_policy_decision")
        if isinstance(cached, PolicyDecision):
            return cached
        username = str((payload or {}).get("user_login") or (payload or {}).get("username") or "")
        display_name = str((payload or {}).get("display_name") or "")
        text = str((payload or {}).get("message_text") or (payload or {}).get("text") or "")
        normalized = self._normalize_guard_text(text)
        compliment_request = bool(re.search(r"\b(?:cumplid\w*|pirop\w*|halag\w*|elog\w*)\b", normalized))
        if compliment_request:
            matched = self.behavior_adaptation.matching_explicit_constraint(
                stream,
                behavior_family="compliment",
                recipient_login=username or display_name,
                requester_login=username or display_name,
            )
            if matched is not None:
                print("[HEBE][OWNER_CONSTRAINT_GATE] matched=true action=block", flush=True)
                print("[HEBE][DIRECT_PRIORITY] bypass_denied reason=owner_constraint", flush=True)
                decision = PolicyDecision(
                    allow_reply=True, allow_llm=False, allow_free_llm=False,
                    reason="owner_behavior_constraint", intent="blocked_compliment_request",
                    direct_template_response="Ese hilo queda cerrado.",
                    response_directive="Set a short neutral boundary without complimenting the viewer.",
                    response_intent="owner_constraint_boundary", response_tone="brief_neutral_boundary",
                    requested_behavior="compliment", behavior_family="compliment",
                    target=username or display_name, matched_by=["owner_behavior_constraint"],
                )
                trace = policy_trace(
                    source="twitch_chat", speaker=display_name or username or "viewer", text=text,
                    decision=decision, addressed_to_hebe=self._message_mentions_hebe(text), authority="viewer",
                )
                trace.update({
                    "event_id": str((payload or {}).get("event_id") or (payload or {}).get("message_id") or f"policy_{uuid.uuid4().hex}"),
                    "actor_login": username,
                })
                payload["_viewer_policy_decision"] = decision
                self._record_policy_trace(trace)
                return decision
        print("[HEBE][OWNER_CONSTRAINT_GATE] matched=false action=allow", flush=True)
        if re.search(r"\b(?:shoutout|promo|so)\b", normalized):
            decision = PolicyDecision(
                allow_reply=True,
                allow_llm=False,
                allow_free_llm=False,
                reason="viewer_not_authorized",
                intent="promo_request_from_viewer",
                response_directive=(
                    "Set one short in-character boundary: a viewer cannot authorize a promotion; "
                    "owner approval is required and no action was taken."
                ),
                response_constraints=[
                    "One short sentence.",
                    "Do not negotiate, invite persuasion, mention payment or VIP access, or create a pending task.",
                    "Do not quote examples or policy metadata.",
                ],
                response_intent="viewer_cannot_request_promo",
                response_tone="brief_self_respecting_boundary",
                requested_behavior="promotion_shoutout",
                behavior_family="owner_stream_operation",
                target=str((payload or {}).get("user_login") or display_name),
                matched_by=["priority_viewer_promo_classifier"],
                execute_as_command=False,
            )
            print(
                f"[HEBE][VIEWER_PROMO_BOUNDARY] viewer={username or display_name} action=reply reason=viewer_not_authorized",
                flush=True,
            )
            print("[HEBE][PROMOTION_EXECUTION_DECISION] allowed=false reason=viewer_not_authorized", flush=True)
            trace = policy_trace(
                source="twitch_chat", speaker=display_name or username or "viewer", text=text,
                decision=decision, addressed_to_hebe=self._message_mentions_hebe(text), authority="viewer",
            )
            trace.update({
                "event_id": str((payload or {}).get("event_id") or (payload or {}).get("message_id") or f"policy_{uuid.uuid4().hex}"),
                "actor_login": username,
            })
            payload["_viewer_policy_decision"] = decision
            self._record_policy_trace(trace)
            return decision
        decision = self._get_viewer_intent_policy().decide(
            stream,
            username=username,
            display_name=display_name,
            text=text,
        )
        trace = policy_trace(
            source="twitch_chat",
            speaker=display_name or username or "viewer",
            text=text,
            decision=decision,
            addressed_to_hebe=True,
            authority="viewer",
        )
        trace.update({
            "event_id": str((payload or {}).get("event_id") or (payload or {}).get("message_id") or f"policy_{uuid.uuid4().hex}"),
            "actor_login": username,
        })
        payload["_viewer_policy_decision"] = decision
        self._record_policy_trace(trace)
        return decision

    def _live_session_debug_snapshot(self) -> dict:
        try:
            snapshot = self._get_live_session_brain().as_debug_dict()
            intelligence = getattr(self, "game_intelligence", None)
            if intelligence is not None:
                snapshot["game_intelligence"] = intelligence.debug_snapshot()
            promotion_store = getattr(self, "promotion_store", None)
            if promotion_store is not None:
                snapshot["automatic_promotion_profiles"] = [
                    asdict(profile) for profile in promotion_store.list_profiles(active_only=True)
                ]
            return snapshot
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
        result = {
            "alive": bool(getattr(self, "_wake_loop_alive", False)),
            "last_error": str(getattr(self, "_wake_loop_last_error", "") or ""),
            "thread_alive": bool(getattr(getattr(self, "_thread", None), "is_alive", lambda: False)()),
        }
        stt = getattr(getattr(self, "runtime", None), "stt", None)
        if stt is not None and hasattr(stt, "health_snapshot"):
            result["stt_health"] = stt.health_snapshot()
        return result

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

    def _twitch_pipeline_health(self) -> dict:
        stream = self._get_stream_state()
        if stream is None:
            return {}
        health = getattr(stream, "twitch_pipeline_health", None)
        if not isinstance(health, dict):
            health = {
                "twitch_messages_received": 0,
                "twitch_messages_bot_ignored": 0,
                "twitch_messages_self_ignored": 0,
                "twitch_messages_presence_evaluated": 0,
                "twitch_messages_observe_only": 0,
                "twitch_messages_should_generate": 0,
                "twitch_messages_generated": 0,
                "twitch_messages_final_emitted": 0,
                "twitch_messages_suppressed": 0,
                "twitch_messages_early_skipped": 0,
                "suppress_reasons": {},
                "last_summary_ts": 0.0,
            }
            stream.twitch_pipeline_health = health
        return health

    def _increment_twitch_pipeline_counter(self, key: str, *, reason: str = "") -> None:
        health = self._twitch_pipeline_health()
        if not health:
            return
        health[key] = int(health.get(key, 0) or 0) + 1
        if reason:
            reasons = dict(health.get("suppress_reasons") or {})
            reasons[reason] = int(reasons.get(reason, 0) or 0) + 1
            health["suppress_reasons"] = reasons

    def _set_last_twitch_route_state(self, **updates) -> None:
        stream = self._get_stream_state()
        state = dict(getattr(self, "_last_twitch_route_state", {}) or {})
        state.update({key: value for key, value in updates.items() if value is not None})
        self._last_twitch_route_state = state
        if stream is not None:
            stream.last_twitch_route_state = state

    def _twitch_bot_login(self) -> str:
        twitch = getattr(self.runtime, "twitch", None)
        bot = (
            getattr(twitch, "bot_username", None)
            or getattr(getattr(self.runtime, "twitch_chat_bot", None), "bot_username", None)
            or "HebeNifelheim"
        )
        return str(bot or "HebeNifelheim").strip().lower().lstrip("@")

    def _twitch_reply_metadata(self, tags: dict | None) -> dict:
        tags = dict(tags or {})
        bot_login = self._twitch_bot_login()
        parent_login = str(tags.get("reply-parent-user-login") or "").strip().lower().lstrip("@")
        thread_parent_login = str(tags.get("reply-thread-parent-user-login") or "").strip().lower().lstrip("@")
        reply_to_hebe = bool(bot_login and bot_login in {parent_login, thread_parent_login})
        return {
            "reply_parent_user_login": parent_login,
            "reply_parent_display_name": str(tags.get("reply-parent-display-name") or ""),
            "reply_thread_parent_user_login": thread_parent_login,
            "reply_thread_parent_display_name": str(tags.get("reply-thread-parent-display-name") or ""),
            "reply_parent_msg_id": str(tags.get("reply-parent-msg-id") or ""),
            "reply_parent_msg_body": str(tags.get("reply-parent-msg-body") or ""),
            "reply_to_hebe_message": reply_to_hebe,
        }

    def _twitch_direct_priority(self, text: str, *, payload: dict | None = None) -> dict:
        payload = payload or {}
        reply_to_hebe = bool(payload.get("reply_to_hebe_message"))
        mentions = bool(self._message_mentions_hebe(text))
        talks = bool(self._viewer_talks_about_hebe(text))
        reason = "reply_to_hebe_message" if reply_to_hebe else "direct_mention" if mentions else "talks_about_hebe" if talks else ""
        applied = bool(reason)
        if applied:
            print(f"[HEBE][DIRECT_HEBE_PRIORITY] applied=true reason={reason}", flush=True)
        return {
            "reply_to_hebe_message": reply_to_hebe,
            "direct_address_to_hebe": bool(reply_to_hebe or mentions),
            "mentions_hebe": bool(reply_to_hebe or mentions),
            "talks_about_hebe": talks,
            "social_candidate": applied,
            "priority": "direct_reply_to_hebe" if reply_to_hebe else "direct_hebe_interaction" if applied else "",
            "direct_priority_reason": reason,
        }

    def _log_twitch_pipeline_health_if_due(self, *, force: bool = False) -> None:
        health = self._twitch_pipeline_health()
        if not health:
            return
        now = time.time()
        if not force and now - float(health.get("last_summary_ts", 0.0) or 0.0) < 600.0:
            return
        health["last_summary_ts"] = now
        reasons = dict(health.get("suppress_reasons") or {})
        top_reasons = ",".join(
            f"{reason}:{count}" for reason, count in sorted(reasons.items(), key=lambda item: item[1], reverse=True)[:5]
        )
        print(
            "[HEBE][TWITCH_PIPELINE_HEALTH] "
            f"received={int(health.get('twitch_messages_received', 0) or 0)} "
            f"presence_evaluated={int(health.get('twitch_messages_presence_evaluated', 0) or 0)} "
            f"emitted={int(health.get('twitch_messages_final_emitted', 0) or 0)} "
            f"early_skipped={int(health.get('twitch_messages_early_skipped', 0) or 0)} "
            f"top_suppress_reasons={top_reasons or 'none'}",
            flush=True,
        )

    def _record_twitch_pipeline_final(
        self,
        *,
        route: str,
        emitted: bool,
        reason: str,
        public_chat_sent: bool = False,
        tts_sent: bool = False,
    ) -> None:
        if emitted:
            self._increment_twitch_pipeline_counter("twitch_messages_final_emitted")
        elif route == "observe_only":
            self._increment_twitch_pipeline_counter("twitch_messages_observe_only", reason=reason)
        else:
            self._increment_twitch_pipeline_counter("twitch_messages_suppressed", reason=reason)
        self._set_last_twitch_route_state(
            output_route=route,
            emitted_to_twitch=bool(public_chat_sent),
            tts_sent=bool(tts_sent),
            suppress_reason="" if emitted else reason,
        )
        print(
            f"[HEBE][TWITCH_PIPELINE_FINAL] route={route} emitted={str(bool(emitted)).lower()} reason={reason}",
            flush=True,
        )

    def ingest_owner_stt(
        self,
        text: str,
        *,
        stt_metadata: dict | None = None,
        allow_wakeword_prompt: bool = False,
    ) -> str:
        """Canonical post-transcription owner STT ingress.

        Faster-whisper and replay both hand normalized transcript text to this
        seam.  It intentionally contains no replay-specific policy.
        """
        return self.ingest_normalized_stt(
            str(text or "").strip(),
            allow_wakeword_prompt=allow_wakeword_prompt,
            stt_metadata=dict(stt_metadata or {}),
        )

    def ingest_normalized_stt(
        self,
        text: str,
        *,
        stt_metadata: dict | None = None,
        allow_wakeword_prompt: bool = False,
        force_ambient: bool = False,
    ) -> str:
        """Shared live/replay seam immediately after transcript production."""
        return self._process_stt_voice_transcript(
            str(text or "").strip(),
            allow_wakeword_prompt=allow_wakeword_prompt,
            stt_metadata=dict(stt_metadata or {}),
            force_ambient=force_ambient,
        )

    def ingest_ambient_stt(self, text: str) -> dict:
        """Canonical normalized ambient transcript ingress.

        The live microphone, developer simulator, and Cognitive Replay all use
        the complete post-transcription STT orchestration.  Source/authority is
        determined by that production classifier and firewall; this wrapper
        does not reproduce an ambient-only policy branch.
        """
        result = self.ingest_normalized_stt(str(text or "").strip(), force_ambient=True)
        trace = dict(getattr(self, "_last_policy_trace", {}) or {})
        firewall = dict(getattr(self, "_last_input_firewall", {}) or {})
        return {
            "result": result,
            "voice_type": str(trace.get("intent") or "ambient_stt"),
            "firewall": firewall,
            "relevance": dict(trace.get("relevance") or {}),
        }

    def ingest_normalized_twitch_chat(
        self,
        *,
        username: str,
        display_name: str,
        text: str,
        channel: str = "",
        irc_tags: dict | None = None,
        normalized_fields: dict | None = None,
    ) -> None:
        """Shared normalized Twitch chat ingress for live IRC and replay.

        Normalization adapters supply production-shaped tags.  From this point
        bot/self filtering, authority firewall, persistence, automatic
        promotion observation, reply metadata and cognitive dispatch are the
        production pipeline.
        """
        if self._is_known_twitch_bot_user(username):
            print(
                f"[HEBE][TWITCH_PIPELINE_SKIP] stage=chat_callback reason=bot_or_self_ignored username={username}",
                flush=True,
            )
            self._increment_twitch_pipeline_counter("twitch_messages_bot_ignored")
            self._increment_twitch_pipeline_counter("twitch_messages_early_skipped", reason="bot_or_self_ignored")
            return
        if self._is_owner_twitch_user(username) and self._is_raw_twitch_command(text):
            self.observe_twitch_chat_message(username, display_name, text, channel, irc_tags=irc_tags or {})
            print(
                "[HEBE][TWITCH][CHATBOT] owner command observed without reaction "
                f"user={username!r} message={text!r}",
                flush=True,
            )
            return
        firewall = self._input_firewall_decision(
            source="twitch_viewer",
            text=text,
            username=username,
            event_type="twitch_chat_react",
            addressed_to_hebe=self._message_mentions_hebe(text),
        )
        if not self._firewall_allows_pipeline(firewall):
            self._increment_twitch_pipeline_counter("twitch_messages_early_skipped", reason=firewall.reason)
            print(
                f"[HEBE][TWITCH_PIPELINE_SKIP] stage=input_firewall reason={firewall.reason} username={username}",
                flush=True,
            )
            return
        self.handle_twitch_chat_event(
            username=username,
            display_name=display_name,
            text=text,
            channel=channel,
            firewall_decision=firewall,
            irc_tags=dict(irc_tags or {}),
            normalized_fields=dict(normalized_fields or {}),
        )
        stream = self._get_stream_state()
        if stream is not None:
            stream.last_chat_activity_ts = time.time()

    def ingest_stream_lifecycle(self, event_type: str, payload: dict | None = None, *, created_at: str = "") -> None:
        mapped = {"stream_started": "stream_online", "stream_ended": "stream_offline"}.get(event_type, event_type)
        if mapped not in {"stream_online", "stream_offline"}:
            raise ValueError(f"unsupported lifecycle event: {event_type}")
        stream_before=self._get_stream_state()
        closing_session_id=str(getattr(stream_before,"active_stream_session_id","") or "") if mapped=="stream_offline" else ""
        self.process_internal_event(InternalEvent(
            event_type=mapped,
            payload=dict(payload or {}),
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
        ))
        if mapped == "stream_offline" and bool(getattr(self,"consolidation_v2",False)):
            data=dict(payload or {});session_id=str(closing_session_id or data.get("session_id") or data.get("stream_id") or "closed_stream")
            def _consolidate_closed_session() -> None:
                try:self.session_consolidator.consolidate(session_id=session_id,start_event=str(data.get("start_event") or "session_start"),end_event=str(data.get("end_event") or created_at or "stream_offline"))
                except Exception as exc:print(f"[HEBE][CONSOLIDATION_COMPLETE] status=failed reason={type(exc).__name__}",flush=True)
            threading.Thread(target=_consolidate_closed_session,name="hebe-session-consolidation",daemon=True).start()

    def ingest_stream_metadata(self, payload: dict) -> bool:
        """Apply replay/live-normalized metadata through production context sync."""
        twitch = getattr(self.runtime, "twitch", None)
        configure = getattr(twitch, "configure_stream_metadata", None)
        if not callable(configure):
            raise RuntimeError("runtime Twitch adapter does not support normalized metadata configuration")
        configure(dict(payload or {}))
        return self.poll_stream_context(force=True, require_enabled=False)

    def handle_twitch_chat_event(
        self,
        *,
        username: str,
        display_name: str,
        text: str,
        channel: str = "",
        firewall_decision: InputFirewallDecision | None = None,
        irc_tags: dict | None = None,
        normalized_fields: dict | None = None,
    ) -> None:
        message = str(text or "").strip()
        fields = dict(normalized_fields or {})
        tags = dict(irc_tags or {})
        event_id = str(
            fields.get("event_id") or fields.get("message_id")
            or tags.get("id") or tags.get("message-id") or ""
        ).strip() or f"twchat_{uuid.uuid4().hex}"
        if not message:
            self._increment_twitch_pipeline_counter("twitch_messages_early_skipped", reason="empty_message")
            print(
                f"[HEBE][TWITCH_PIPELINE_SKIP] stage=ingress reason=empty_message event_id={event_id} username={username}",
                flush=True,
            )
            return
        self._increment_twitch_pipeline_counter("twitch_messages_received")
        print(
            f"[HEBE][TWITCH_PIPELINE_START] event_id={event_id} username={username} raw={message!r}",
            flush=True,
        )
        self._observe_automatic_promotion(
            username=username,
            display_name=display_name,
            message=message,
            irc_tags=irc_tags or {},
            fallback_message_id=event_id,
        )
        person_id = self.observe_twitch_chat_message(
            username,
            display_name,
            message,
            channel,
            firewall_decision=firewall_decision,
            irc_tags=tags,
        )
        if person_id:
            fields["person_id"] = person_id
        stream = self._get_stream_state()
        recent_chat = list(getattr(stream, "recent_chat_messages", []) or [])[-10:] if stream is not None else []
        reply_metadata = self._twitch_reply_metadata(irc_tags or {})
        direct_priority = self._twitch_direct_priority(
            message,
            payload={**reply_metadata, "user_login": username, "display_name": display_name},
        )
        payload = {
            **fields,
            "event_id": event_id,
            "_pipeline_started": True,
            "display_name": display_name,
            "user_login": username,
            "username": username,
            "message_text": message,
            "channel": channel,
            "recent_chat": recent_chat,
            "irc_tags": dict(irc_tags or {}),
            **reply_metadata,
            **direct_priority,
        }
        self.process_internal_event(InternalEvent(
            event_type="twitch_chat_react",
            payload=payload,
            created_at=datetime.now(timezone.utc).isoformat(),
        ))
        self._log_twitch_pipeline_health_if_due()

    def _observe_automatic_promotion(
        self,
        *,
        username: str,
        display_name: str,
        message: str,
        irc_tags: dict,
        fallback_message_id: str,
    ) -> None:
        service = getattr(self, "automatic_promotions", None)
        stream = self._get_stream_state()
        if service is None or stream is None:
            return
        session_id = self._ensure_stream_memory_session_if_live(stream)
        if not session_id:
            return
        tags = dict(irc_tags or {})
        twitch_user_id = str(tags.get("user-id") or tags.get("user_id") or "")
        if twitch_user_id and str(username or "").strip():
            self._twitch_user_ids_by_login[str(username).strip().casefold()] = twitch_user_id
        try:
            service.observe_chat_message(
                stream_session_id=session_id,
                twitch_user_id=twitch_user_id,
                login=username,
                display_name=display_name or username,
                message_text=message,
                message_id=str(tags.get("id") or tags.get("message-id") or fallback_message_id),
                channel_live=bool(getattr(stream, "is_live", False)),
                is_bot=self._is_known_twitch_bot_user(username),
                is_self=self._is_owner_twitch_user(username),
            )
            service.drain_ready(self._send_automatic_promotion)
        except Exception as exc:
            print(f"[HEBE][AUTO_PROMO_OUTCOME] viewer={username} status=failed reason={type(exc).__name__}", flush=True)

    def _send_automatic_promotion(
        self, login: str, *, source: str = "automatic_promotion_policy", authority: str = "owner_delegated",
        twitch_user_id: str = "",
    ) -> dict:
        service = getattr(self, "automatic_promotions", None)
        store = getattr(service, "store", None)
        if store is not None:
            profile = store.get_profile(twitch_user_id=twitch_user_id, login=login)
            if (
                profile is None or not profile.active
                or str(profile.current_login or "").casefold() != str(login or "").casefold()
                or (twitch_user_id and str(profile.twitch_user_id) != str(twitch_user_id))
            ):
                return {"success": False, "reason": "invalid_identity"}
        context = getattr(self, "live_context_policy", None) or HebeLiveContextPolicy()
        authorization = context.authorize_action(
            HebeLiveRuntimeContext.STREAM_PUBLIC,
            "promotion.automatic_first_message",
            trusted_automation=True,
        )
        print(
            "[HEBE][LIVE_CONTEXT_GATE] "
            f"context={authorization.context} operation={authorization.operation} "
            f"allowed={str(authorization.allowed).lower()} reason={authorization.reason}",
            flush=True,
        )
        if not authorization.allowed:
            return {"success": False, "reason": authorization.reason}
        ok, _normalized, reason = self._send_shoutout(
            login,
            source=source,
            authority=authority,
            force=False,
        )
        return {"success": bool(ok), "reason": reason}

    def observe_twitch_chat_message(
        self,
        username: str,
        display_name: str,
        text: str,
        channel: str = "",
        *,
        firewall_decision: InputFirewallDecision | None = None,
        irc_tags: dict | None = None,
    ) -> str:
        if not getattr(self, "stream_observe_chat", True):
            return ""
        stream = self._get_stream_state()
        if not stream:
            return ""

        message = str(text or "").strip()
        if not message:
            return ""

        firewall = firewall_decision or self._input_firewall_decision(
            source="twitch_viewer",
            text=message,
            username=username,
            event_type="twitch_chat_observe",
            addressed_to_hebe=self._message_mentions_hebe(message),
        )
        if not self._firewall_allows_pipeline(firewall):
            return ""

        print(f"[HEBE][TWITCH][CHAT] observed username={username} message={message!r}", flush=True)
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
                try:
                    identity=self.social_world_repository.find_identity(platform="twitch",platform_user_id=str((irc_tags or {}).get("user-id") or (irc_tags or {}).get("user_id") or ""),login=username)
                    return identity.person_id if identity else ""
                except Exception:
                    return ""
        stream.last_chat_activity_ts = now
        stream.human_messages_since_last_public_reply = int(getattr(stream, "human_messages_since_last_public_reply", 0) or 0) + 1
        if int(getattr(stream, "consecutive_public_replies", 0) or 0) > 0:
            self._reset_twitch_reply_budget(stream, "human_chat_between_replies", now=now)
        session_id = self._ensure_stream_memory_session_if_live(stream)
        topic = self._classify_chat_topic(message)
        linked_context = self._linked_run_context_for_chat_topic(stream, topic)
        person_id=""
        social=getattr(self,"social_world",None);tags=dict(irc_tags or {})
        if social is not None:
            try:
                observation_id=str(tags.get("id") or tags.get("message-id") or f"social_chat_{uuid.uuid4().hex}")
                person,_identity,_inserted=social.observe_presence(observation_id=observation_id,platform="twitch",platform_user_id=str(tags.get("user-id") or tags.get("user_id") or ""),login=username,display_name=display_name or username,stream_session_id=str(session_id or ""),source="twitch_chat",message_seen=True,direct_interaction=self._message_mentions_hebe(message));person_id=person.person_id
            except Exception as exc:print(f"[HEBE][SOCIAL_PERSON_RESOLVE] decision=failed reason={type(exc).__name__}",flush=True)
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
        return person_id

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

    def _stream_recent_raid_context(self, stream=None, *, now: float | None = None) -> dict | None:
        stream = stream or self._get_stream_state()
        if stream is None:
            return None
        now = time.time() if now is None else float(now)
        contexts = [
            dict(item) for item in list(getattr(stream, "recent_raid_contexts", []) or [])
            if float(item.get("expires_at", 0.0) or 0.0) > now
        ]
        stream.recent_raid_contexts = contexts[-5:]
        open_contexts = [item for item in contexts if not bool(item.get("shoutout_done"))]
        if len(open_contexts) == 1:
            return open_contexts[0]
        return None

    def _remember_raid_context(self, stream, payload: dict, *, thanked: bool = False) -> dict:
        now = time.time()
        username = str(payload.get("user_login") or payload.get("display_name") or "alguien").strip()
        context = {
            "display_name": str(payload.get("display_name") or username),
            "user_login": username,
            "viewer_count": int(payload.get("viewer_count") or 0),
            "event_id": str(payload.get("event_id") or payload.get("message_id") or ""),
            "ts": now,
            "expires_at": now + 300.0,
            "thanked": bool(thanked),
            "shoutout_done": False,
            "source": str(payload.get("source") or ""),
        }
        existing = [
            dict(item) for item in list(getattr(stream, "recent_raid_contexts", []) or [])
            if float(item.get("expires_at", 0.0) or 0.0) > now
        ]
        existing = [
            item for item in existing
            if str(item.get("user_login") or item.get("display_name") or "").casefold() != username.casefold()
        ]
        existing.append(context)
        stream.recent_raid_contexts = existing[-5:]
        return context

    def _mark_recent_raid_shoutout_done(self, stream, target: str) -> None:
        if stream is None or not target:
            return
        target_norm = self._normalize_shoutout_target(target).casefold()
        contexts = []
        for item in list(getattr(stream, "recent_raid_contexts", []) or []):
            data = dict(item)
            candidate = self._normalize_shoutout_target(data.get("user_login") or data.get("display_name") or "").casefold()
            if candidate and candidate == target_norm:
                data["shoutout_done"] = True
            contexts.append(data)
        stream.recent_raid_contexts = contexts[-5:]

    def _raid_duplicate_context(self, stream, payload: dict) -> dict | None:
        if stream is None:
            return None
        source = str(payload.get("source") or "")
        if source not in {"irc_usernotice", "bot_fallback", "eventsub"}:
            return None
        username = str(payload.get("user_login") or payload.get("display_name") or "").casefold()
        viewers = int(payload.get("viewer_count") or 0)
        event_id = str(payload.get("event_id") or payload.get("message_id") or "").strip()
        now = time.time()
        for item in reversed(list(getattr(stream, "recent_raid_contexts", []) or [])):
            if now - float(item.get("ts", 0.0) or 0.0) > 60.0:
                continue
            if event_id and str(item.get("event_id") or "") == event_id and bool(item.get("thanked")):
                return dict(item)
            same_user = str(item.get("user_login") or item.get("display_name") or "").casefold() == username
            same_viewers = int(item.get("viewer_count") or 0) == viewers
            if same_user and same_viewers and bool(item.get("thanked")):
                return dict(item)
        return None

    def _raid_event_is_duplicate(self, stream, payload: dict) -> bool:
        return self._raid_duplicate_context(stream, payload) is not None

    def _raid_ack_fallback(self, payload: dict) -> str:
        raider = str(payload.get("display_name") or payload.get("user_login") or "raider").strip()
        viewers = int(payload.get("viewer_count") or 0)
        preferences=getattr(self,"owner_procedural_preferences",None)
        omit=bool(preferences and preferences.rendering_policy("raid_ack").get("omit_viewer_count"))
        count = f" con {viewers}" if viewers > 0 and not omit else ""
        return f"Gracias por la raid{count}, {raider}."

    def _render_raid_ack_safe(self, event, cognitive_decision=None) -> tuple[str, bool, str]:
        payload = getattr(event, "payload", {}) or {}
        raider = str(payload.get("display_name") or payload.get("user_login") or "raider")
        viewers = int(payload.get("viewer_count") or 0)
        print(f"[HEBE][RAID_ACK_RENDER] raider={raider} viewers={viewers} route=renderer", flush=True)
        try:
            preferences=getattr(self,"owner_procedural_preferences",None)
            render_policy=preferences.rendering_policy("raid_ack") if preferences else {"omit_viewer_count":False}
            render_payload=dict(payload);render_payload["owner_procedural_preferences"]=render_policy
            if render_policy.get("omit_viewer_count"):
                render_payload["viewer_count_telemetry"]=viewers;render_payload["viewer_count"]=0
            event.payload=render_payload
            text = self._synthesize_internal_event_reply(event, cognitive_decision=cognitive_decision)
            event.payload=payload
            return str(text or "").strip(), False, ""
        except Exception as exc:
            event.payload=payload
            fallback = self._raid_ack_fallback(payload)
            print(f"[HEBE][RAID_ACK_ERROR] error={type(exc).__name__}: {exc} fallback_used=true", flush=True)
            return fallback, True, f"{type(exc).__name__}: {exc}"

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

    def _send_shoutout(
        self, target: str, *, source: str, authority: str = "", force: bool = False,
        explicit_self: bool = False,
    ) -> tuple[bool, str, str]:
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
        delegated_automatic = source == "automatic_promotion_policy" and authority == "owner_delegated"
        if source not in {"manual", "raid"} and not delegated_automatic:
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
            f"allowed reason={'owner_delegated_profile' if delegated_automatic else 'approved_stream_event' if source == 'raid' else 'owner_direct_command'} "
            f"source={source} authority={authority or 'direct'} target={normalized}",
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
        self._last_shoutout_external_confirmation = ""
        try:
            print("[HEBE][RESPONSE_PIPELINE_BYPASS] allowed=true reason=twitch_action_only_shoutout", flush=True)
            if callable(shoutout):
                raw_result = shoutout(normalized)
                if isinstance(raw_result, dict):
                    ok = bool(raw_result.get("success") or raw_result.get("sent"))
                    self._last_shoutout_external_confirmation = str(
                        raw_result.get("message_id") or raw_result.get("external_reference") or ""
                    )
                else:
                    ok = bool(raw_result)
                command = getattr(twitch, "build_shoutout_command", lambda user: f"!so {user}")(normalized)
            else:
                template = os.getenv("HEBE_SHOUTOUT_COMMAND_TEMPLATE", "!so {username}") or "!so {username}"
                command = template.format(username=normalized)
                raw_result = twitch.send_message(command) if twitch else False
                if isinstance(raw_result, dict):
                    ok = bool(raw_result.get("success") or raw_result.get("sent"))
                    self._last_shoutout_external_confirmation = str(
                        raw_result.get("message_id") or raw_result.get("external_reference") or ""
                    )
                else:
                    ok = bool(raw_result)
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

    def _is_owner_twitch_user(self, username: str) -> bool:
        user = str(username or "").strip().lower().lstrip("@")
        stream = self._get_stream_state()
        twitch = getattr(self.runtime, "twitch", None)
        candidates = {
            "leonifelheim",
            str(getattr(stream, "channel_name", "") or "").strip().lower().lstrip("@") if stream is not None else "",
            str(getattr(twitch, "channel_name", "") or "").strip().lower().lstrip("@") if twitch is not None else "",
        }
        configured = os.getenv("HEBE_TWITCH_OWNER_USERNAMES", "")
        candidates.update(part.strip().lower().lstrip("@") for part in configured.split(",") if part.strip())
        return bool(user and user in {item for item in candidates if item})

    def _is_known_twitch_bot_user(self, username: str) -> bool:
        user = str(username or "").strip().lower().lstrip("@")
        if not user:
            return True
        stream = self._get_stream_state()
        twitch = getattr(self.runtime, "twitch", None)
        candidates = {
            "hebenifelheim",
            "jotunbot",
            "streamelements",
            "nightbot",
            "moobot",
            "fossabot",
            "streamlabs",
            str(getattr(stream, "bot_username", "") or "").strip().lower().lstrip("@") if stream is not None else "",
            str(getattr(twitch, "bot_username", "") or "").strip().lower().lstrip("@") if twitch is not None else "",
        }
        configured = os.getenv("HEBE_TWITCH_BOT_USERNAMES", "")
        candidates.update(part.strip().lower().lstrip("@") for part in configured.split(",") if part.strip())
        return user in {item for item in candidates if item}

    def _is_raw_twitch_command(self, text: str) -> bool:
        value = str(text or "").strip()
        return bool(re.match(r"^!(?:so|shoutout|clip|title|game|category|marker|commercial|raid|unraid|mod|unmod|vip|unvip)\b", value, flags=re.IGNORECASE))

    def _ensure_stream_memory_session_if_live(self, stream=None) -> int | None:
        stream = stream or self._get_stream_state()
        if not stream or not getattr(stream, "is_live", False):
            return getattr(stream, "active_stream_session_id", None) if stream else None
        try:
            session_id = stream_memory.ensure_active_stream_session(stream, source="engine")
            manager = getattr(self, "live_session_state_manager", None)
            if session_id and manager is not None and str(manager.current_session_id) != str(session_id):
                reset = manager.begin_session(stream, session_id)
                if reset.changed:
                    self._reset_live_session_subsystems()
                auto_promotions = getattr(self, "automatic_promotions", None)
                if auto_promotions is not None:
                    auto_promotions.start_session(session_id)
                self.live_session_brain = LiveSessionBrain(stream)
            return session_id
        except Exception as exc:
            print(f"[HEBE][STREAM_MEMORY] ensure session failed: {exc!r}", flush=True)
            return None

    def _reset_live_session_subsystems(self) -> None:
        reset_names: list[str] = []
        for name in (
            "stream_companion_loop",
            "owner_discourse_buffer",
            "stream_turn_detector",
            "discourse_participation_budget",
            "cheer_deduplicator",
            "final_emission_gate",
            "twitch_interaction_coordinator",
            "troll_engagement_budget",
            "scene_timeline",
            "spontaneous_opportunities",
        ):
            component = getattr(self, name, None)
            reset = getattr(component, "reset_session", None) or getattr(component, "reset", None)
            if callable(reset):
                reset()
                reset_names.append(name)
        print(
            "[HEBE][LIVE_SESSION_SUBSYSTEM_RESET] "
            f"components={','.join(reset_names) or 'none'}",
            flush=True,
        )

    def _record_stream_event_safe(self, event_type: str, payload: dict | None = None, *, stream=None) -> None:
        try:
            stream_memory.record_stream_event(event_type, payload or {}, stream=stream)
        except Exception as exc:
            print(f"[HEBE][STREAM_MEMORY] record event failed event_type={event_type!r}: {exc!r}", flush=True)

    def _close_stream_memory_session_safe(self, stream, *, reason: str) -> object | None:
        try:
            stream_memory.mark_stream_session_ending(stream, reason=reason, source_signal=reason)
            summary=stream_memory.close_active_stream_session(stream, reason=reason)
            self._persist_canonical_chatter_summaries(summary)
            return summary
        except Exception as exc:
            print(f"[HEBE][STREAM_MEMORY] close session failed reason={reason!r}: {exc!r}", flush=True)
            return None

    def _recover_stream_memory_session_on_startup(self) -> None:
        try:
            stream_memory.prune_session_artifacts()
        except Exception as exc:
            print(f"[HEBE][STREAM_LIFECYCLE] retention_prune_failed reason={type(exc).__name__}", flush=True)
        active = stream_memory.get_active_stream_session()
        if not active:
            return
        stream = self._get_stream_state()
        service = getattr(self, "stream_context_sync", None)
        ok = bool(stream is not None and service is not None and service.sync(stream))
        evidence = bool(getattr(stream, "is_live", False)) if ok else None
        current_stream_id = str(getattr(stream, "twitch_stream_id", "") or "") if stream is not None else ""
        result = stream_memory.recover_incomplete_stream_session(
            stream, live_evidence=evidence, current_stream_id=current_stream_id
        )
        if result.get("action") == "resumed":
            self._ensure_stream_memory_session_if_live(stream)
        elif result.get("action") == "finalized":
            self._persist_canonical_chatter_summaries(result.get("summary"))
            if evidence is True:
                self._ensure_stream_memory_session_if_live(stream)
        print(
            f"[HEBE][STREAM_LIFECYCLE] startup_recovery action={result.get('action')} "
            f"session_id={result.get('session_id')} evidence={evidence}",
            flush=True,
        )

    def _offer_raid_farewell_once(self, stream, payload: dict, session_id: int) -> None:
        if not stream_memory.mark_farewell_status(
            session_id, status="attempted", reason="outgoing_raid", only_if_not_attempted=True
        ):
            return
        try:
            # The renderer owns its bounded deterministic fallback. Any failure
            # here is observable and never delays technical finalization.
            text = self.response_synthesizer._generate_twitch_outgoing_raid(payload)
        except Exception as exc:
            stream_memory.mark_farewell_status(
                session_id, status="skipped", reason=f"generation_failed:{type(exc).__name__}"
            )
            return
        twitch = getattr(self.runtime, "twitch", None)
        delivery_before = getattr(twitch, "last_delivery_outcome", None)
        try:
            self._deliver_twitch_reply(
                text,
                event_type="twitch_outgoing_raid",
                payload={**payload, "_force_skip_tts": False},
            )
            delivery_after = getattr(twitch, "last_delivery_outcome", None)
            if delivery_after is not delivery_before and isinstance(delivery_after, dict) and delivery_after.get("success"):
                stream_memory.mark_farewell_status(session_id, status="emitted", reason="outgoing_raid")
            else:
                detail = (
                    str(delivery_after.get("reason") or "delivery_not_confirmed")
                    if isinstance(delivery_after, dict) else "delivery_not_confirmed"
                )
                stream_memory.mark_farewell_status(session_id, status="skipped", reason=detail)
        except Exception as exc:
            stream_memory.mark_farewell_status(
                session_id, status="skipped", reason=f"delivery_failed:{type(exc).__name__}"
            )

    def _handle_outgoing_raid(self, event) -> None:
        stream = self._get_stream_state()
        payload = dict(getattr(event, "payload", {}) or {})
        active = stream_memory.mark_stream_session_ending(
            stream, reason="raid", source_signal=str(payload.get("source_signal") or "eventsub_outgoing_raid")
        )
        if not active:
            return
        session_id = int(active["id"])
        self._record_stream_event_safe("twitch_outgoing_raid", payload, stream=stream)
        self._offer_raid_farewell_once(stream, payload, session_id)
        try:
            summary = stream_memory.finalize_stream_session(
                stream, reason="raid", source_signal="eventsub_outgoing_raid"
            )
            self._persist_canonical_chatter_summaries(summary)
        except Exception as exc:
            print(
                f"[HEBE][STREAM_LIFECYCLE] session_finalize_failed session_id={session_id} "
                f"reason={type(exc).__name__} source_signal=eventsub_outgoing_raid",
                flush=True,
            )

    def _persist_canonical_chatter_summaries(self, summary) -> None:
        if not isinstance(summary,dict) or getattr(self,"social_world",None) is None:return
        highlights=summary.get("social_summary_candidates") or []
        for item in highlights if isinstance(highlights,list) else []:
            if not isinstance(item,dict):continue
            login=str(item.get("username") or "").strip()
            if not login:continue
            try:
                login_key=hashlib.sha256(login.casefold().encode("utf-8")).hexdigest()[:20]
                self.social_world.record_summary_for_login(login=login,stream_session_id=str(summary.get("stream_session_id") or ""),source_record_id=f"stream_summary:{summary.get('id')}:{login_key}",summary_text=str(item.get("summary") or ""),topics=tuple(item.get("topics") or ()),message_count=int(item.get("message_count") or 0),direct_interaction_count=int(item.get("direct_interaction_count") or 0))
            except Exception as exc:print(f"[HEBE][SOCIAL_SUMMARY] login={login} status=skipped reason={type(exc).__name__}",flush=True)

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
        active_pending = self._active_current_conversation(source=source)
        current_event = getattr(self, "_current_input_event", None)
        if current_event is None or (
            source in {"ui", "typed_ui"}
            and str(getattr(current_event, "raw_text", "") or "") != str(command or "")
        ):
            current_event = self._build_input_event(
                source="ui" if source in {"ui", "typed_ui"} else source,
                raw_text=command,
                normalized_text=self._normalize_text(command),
            )
            self._current_input_event = current_event
        interpretation = getattr(current_event, "interpretation", None)
        if interpretation is None:
            interpretation = self._get_input_interpreter().interpret_event(current_event)
        if (
            source in {"ui", "typed_ui"}
            and active_pending is not None
            and interpretation.speech_act != InputSpeechAct.OWNER_FEEDBACK
            and self._pending_conversation_matches(source=source, text=command)
        ):
            interpretation = self._get_input_interpreter().interpret_event(
                current_event,
                authority="owner",
                addressed_to_hebe=True,
                explicit_command_mode=False,
                pending_valid=True,
                recent_hebe_utterance=str(getattr(self, "_last_assistant_text", "") or ""),
            )
        if source in {"ui", "typed_ui"} and active_pending is not None:
            print(
                f"[HEBE][UI_PENDING_INPUT] raw={command!r} pending_kind={active_pending.topic}",
                flush=True,
            )
            print("[HEBE][PENDING_ROUTER] source=ui compatible=probe", flush=True)
        print(
            "[HEBE][COG] incoming "
            f"source={source!r} "
            f"command={command!r} "
            f"current_pending={active_pending!r}",
            flush=True,
        )
        firewall = None
        if source in {"ui", "typed_ui"}:
            firewall = self._input_firewall_decision(
                source="owner_ui",
                text=command,
                addressed_to_hebe=True,
                has_action_intent=interpretation.authorized_action_command,
            )

        stt_firewall_payload = {}
        current_metadata = getattr(current_event, "stt_metadata", None)
        if source == "stt_voice" and isinstance(current_metadata, dict):
            stt_firewall_payload = current_metadata.get("input_firewall") or {}
        route_source = str(stt_firewall_payload.get("source") or ("ui" if source in {"ui", "typed_ui"} else source))
        route_authority = str(getattr(interpretation, "authority", "") or stt_firewall_payload.get("authority") or (
            "owner" if route_source in {"ui", "stt_voice", "owner_stt_direct", "owner_stt_followup", "voice"} else "system"
        ))
        route_addressed = bool(interpretation.addressed_to_hebe)
        if route_authority == "owner" and self._handle_translate_previous_response(command, source=source):
            return "continue"
        if not hasattr(self, "context_builder"):
            context = SimpleNamespace(
                input_text=command, internal_event=None,
                state_snapshot={"current_conversation": active_pending},
                source=route_source, authority=route_authority,
                addressed_to_hebe=route_addressed,
                input_interpretation=interpretation,
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
                    input_interpretation=interpretation,
                )
            except TypeError:
                context = self.context_builder.build(
                    state=self.runtime.state, input_text=command, internal_event=None
                )
                context.source = route_source
                context.authority = route_authority
                context.addressed_to_hebe = route_addressed
                context.input_interpretation = interpretation
        if not hasattr(context, "state_snapshot") or not isinstance(context.state_snapshot, dict):
            context.state_snapshot = {}
        context.state_snapshot["current_conversation"] = active_pending
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
        if (
            interpretation.authorized_action_command
            and hasattr(self, "_parse_tts_control_intent")
            and self._parse_tts_control_intent(normalized_route) is not None
        ):
            hints.append("tts_control")
        if interpretation.authorized_action_command and (
            active_pending is not None and active_pending.topic == "tts_scope"
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
        if interpretation.authorized_action_command and stream_domain and stream_control:
            hints.append("stream_manual")
        if interpretation.authorized_action_command and route_tokens & {"chat"} and stream_control:
            hints.append("stream_action")
        if interpretation.authorized_action_command and route_tokens & {"shoutout", "promo", "raid"} and stream_control:
            hints.append("stream_action")
            hints.append("stream_manual")
        pending = self._active_current_conversation(source=source)
        if (
            interpretation.speech_act == InputSpeechAct.OWNER_ANSWER_FOLLOWUP
            and pending is not None
            and pending.topic == "promotion_target_clarification"
        ):
            hints.append("stream_action")
            hints.append("stream_manual")
        if re.search(r"\b(?:que|cual)\s+(?:toca|juego|directo|stream)\b", normalized_route):
            hints.append("stream_query")
        context.route_hints = hints
        context.cognitive_decision = router.route(context)
        if bool(getattr(self, "_manual_simulation_mode", False)):
            context.cognitive_decision.action_permission_summary["is_simulation"] = True
        self._last_cognitive_trace = context.cognitive_decision.to_dict()
        decision = context.cognitive_decision

        input_trace_id = str(
            (getattr(current_event, "stt_metadata", {}) or {}).get("interaction_trace_id")
            or f"input_{uuid.uuid4().hex}"
        )
        explanation = self._grounded_self_explanation(
            command,
            requester="Leo",
            current_trace_id=input_trace_id,
        )
        if explanation.detected:
            self._last_cognitive_trace.update({
                "selected_route": "grounded_self_explanation",
                "explanation_source_trace": explanation.source_trace_id,
                "explanation_reason_code": explanation.reason_code,
                "final_response": explanation.text,
            })
            self._deliver_manual_reply(explanation.text, source=source)
            self._get_interaction_decision_history().update(
                self._get_stream_state(), input_trace_id, emission_outcome="emitted"
            )
            return "continue"

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
        followup_conversation = None
        if source in {"ui", "stt_voice"} and self._pending_conversation_matches(source=source, text=command):
            followup_conversation = self._active_current_conversation(source=source)
            if source == "ui" and followup_conversation is not None:
                self.conversation_continuity.resolve_input(
                    context_kind=followup_conversation.context_kind.value,
                    context_id=followup_conversation.context_id,
                    source="owner_ui", participant="leo", authority="owner", text=command,
                    event_id=f"ui_{uuid.uuid4().hex}", compatibility=True,
                    compatibility_reason="owner_related_followup",
                )
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
            conversation_state = self._get_conversation_state_resolver().from_conversation(
                followup_conversation or self._active_current_conversation(source=source),
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
            f"{context.state_snapshot.get('current_conversation')!r}",
            flush=True,
        )

        deliberation = self.deliberation_service.deliberate(context)
        self._last_cognitive_trace.update({
            "selected_route": context.cognitive_decision.intent,
            "final_plan_steps": [step.type for step in deliberation.plan.steps],
        })
        if decision.intent == "command_open_app":
            self._trace_canonical_open_app_plan(deliberation.plan, input_event)
        execution = self.plan_executor.execute(deliberation.plan)
        if decision.intent == "command_open_app":
            self._record_canonical_open_app_execution(execution)
        self._last_cognitive_trace["plan_executor_guard"] = list(getattr(self.plan_executor, "last_guard_results", []) or [])
        state_update = execution.first_result_of_type("state_update")
        self._apply_game_run_state_execution(state_update)
        reply_text = self.response_synthesizer.synthesize(
            context=context,
            deliberation=deliberation,
            execution=execution,
        )
        self._last_cognitive_trace["final_response"] = reply_text
        current_interaction_trace_id = str(
            (getattr(input_event, "stt_metadata", {}) or {}).get("interaction_trace_id") or ""
        )
        if decision.intent == "command_open_app" and current_interaction_trace_id:
            self._get_interaction_decision_history().update(
                self._get_stream_state(),
                current_interaction_trace_id,
                generation_outcome=("generated" if reply_text else "failed"),
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
                conversation = self._open_pending_conversation(
                    kind="appointment_datetime",
                    expected_reply_type="datetime",
                    capability_needed="appointment.create",
                    opened_by_speech_act="clarification_question",
                    explicit_question_asked=True,
                    can_accept_no_wake_followup=not context.stream_is_live,
                    max_attempts=1 if context.stream_is_live else 2,
                    draft=reply_step.data.get("draft", {}),
                    creation_reason="appointment_datetime_missing",
                )

                print(
                    "[HEBE][STATE] saved current_conversation="
                    f"{conversation!r}",
                    flush=True,
                )

            elif mode == "confirm_appointment":
                self._close_current_conversation(reason="resolved", conversation=active_pending)

                print(
                    "[HEBE][STATE] closed current_conversation",
                    flush=True,
                )

            elif mode == "game_guidance_clarification":
                self._apply_game_guidance_reply_state(reply_step.data, decision, context, route_source)

            elif mode == "game_guidance" and decision.intent == "game_guidance_clarification_answer":
                self._apply_game_guidance_reply_state(reply_step.data, decision, context, route_source)

        print(
            "[HEBE][COG] "
            f"reasoning={deliberation.plan.reasoning!r} "
            f"steps={[step.type for step in deliberation.plan.steps]!r} "
            f"reply={reply_text!r}",
            flush=True,
        )

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
                if source == "ui" or decision.intent == "command_open_app":
                    self._deliver_manual_reply(reply_text, source=source)
                else:
                    self._deliver_voice_reply(reply_text)
                    self._record_assistant_reply_for_conversation(reply_text, source=source, synthesizer=getattr(self, "response_synthesizer", None))
                if decision.intent == "command_open_app" and current_interaction_trace_id:
                    self._get_interaction_decision_history().update(
                        self._get_stream_state(), current_interaction_trace_id, emission_outcome="emitted"
                    )
            except Exception as e:
                print(f"[HEBE][COG] speak failed: {e!r}", flush=True)

        normalized = self._normalize_text(command)

        if normalized in {"duerme", "modo espera", "modo de espera"}:
            return "sleep"

        if normalized in {"apaga hebe", "detente", "stop engine"}:
            return "stop"

        return "continue"

    def _apply_game_guidance_reply_state(self, reply_data, decision, context, route_source: str) -> None:
        mode = str((reply_data or {}).get("mode") or "")
        if mode == "game_guidance" and getattr(decision, "intent", "") == "game_guidance_clarification_answer":
            pending = self._active_current_conversation(source=route_source)
            log_jsonl_event("pending", {
                "event": "pending_consumed",
                "kind": "game_guidance_clarification",
                "source": route_source,
                "authority": getattr(decision, "authority", "owner"),
                "compatibility_reason": getattr(decision, "pending_reason", ""),
            })
            print(
                f"[HEBE][PENDING_CONSUMED] kind=game_guidance_clarification "
                f"id={pending.id if pending else ''} reason={getattr(decision, 'pending_reason', '')}",
                flush=True,
            )
            self._close_current_conversation(reason="consumed", conversation=pending)
            return
        if mode != "game_guidance_clarification":
            return
        guidance_decision = dict((reply_data or {}).get("game_guidance") or {})
        guidance = dict(guidance_decision.get("context") or {})
        missing_fields = self.deliberation_service.game_guidance.missing_fields(guidance)
        now_ts = time.time()
        pending = self._open_pending_conversation(
            kind="game_guidance_clarification",
            expected_reply_type=(
                "game_party_or_character"
                if {"current_character", "party_members"} & set(missing_fields)
                else "game_progress_state"
            ),
            capability_needed="game_guidance",
            opened_by_event_id=str(getattr(getattr(self, "_current_input_event", None), "raw_text", "") or ""),
            opened_by_speech_act="game_guidance_clarification",
            explicit_question_asked=True,
            can_accept_no_wake_followup=True,
            ttl_seconds=float(os.getenv("HEBE_GAME_PENDING_TTL_SECONDS", "90") or 90),
            max_attempts=1 if bool(getattr(context, "stream_is_live", False)) else 2,
            compatible_intents=["game_guidance_clarification_answer"],
            incompatible_intents=["stream_monologue", "real_life_anecdote"],
            game=guidance.get("game"),
            location_or_area=guidance.get("location_or_area"),
            missing_fields=missing_fields,
            original_question=(
                (guidance.get("source_context") or {}).get("user_input") or context.input_text
            ),
            source=route_source,
            spoiler_policy=guidance.get("spoiler_policy"),
            clarification_attempt_count=0,
            max_clarification_attempts=1 if bool(getattr(context, "stream_is_live", False)) else 2,
            last_clarification_event_id=str(getattr(getattr(self, "_current_input_event", None), "raw_text", "") or ""),
            creation_reason="game_guidance_missing_run_context",
        )
        print(
            f"[HEBE][GAME_PENDING] created id={pending.id} "
            f"expected_reply_type={pending.expected_reply.type.value if pending.expected_reply else ''}",
            flush=True,
        )
        log_jsonl_event("pending", {
            "event": "pending_created",
            "id": pending.id,
            "kind": "game_guidance_clarification",
            "expected_reply_type": pending.expected_reply.type.value if pending.expected_reply else "",
            "source": route_source,
            "authority": "owner",
            "missing_fields": missing_fields,
            "compatibility_reason": "game_guidance_missing_run_context",
        })
        log_jsonl_event("game_guidance", {
            "game": guidance.get("game"),
            "location": guidance.get("location_or_area"),
            "current_character": guidance.get("current_character"),
            "party_members": guidance.get("party_members"),
            "game_run_state": GameRunState.from_value(getattr(self.runtime.state, "game_run_state", None)).to_dict(),
            "rag_used": bool(guidance_decision.get("rag_used")),
            "rag_skipped": not bool(guidance_decision.get("rag_used")),
            "web_used": bool(guidance_decision.get("web_used")),
            "web_skipped": not bool(guidance_decision.get("web_used")),
            "needs_clarification": True,
            "clarification_pending_created": True,
            "reason": guidance_decision.get("reason") or "missing_run_context",
        })

    def _apply_game_run_state_execution(self, state_update) -> None:
        if not state_update or not state_update.success or state_update.data.get("kind") != "game_run_state":
            return
        updates = dict(state_update.data.get("updates") or {})
        stream = self._get_stream_state()
        service = getattr(self,"game_run_service",None)
        if service is None:
            print("[HEBE][GAME_PENDING] state_update_rejected reason=canonical_service_unavailable",flush=True)
            return
        pending_id = str(state_update.data.get("pending_id") or "game_run_state")
        explicit_game = str(updates.pop("game","") or "").strip()
        game = str(explicit_game or getattr(stream,"current_game",None) or getattr(stream,"current_category",None) or "").strip()
        resolved_game = ""
        run_id = str(getattr(stream,"active_game_run_id","") or "") if stream is not None else ""
        if game and stream is not None and getattr(stream,"active_stream_session_id",None):
            resolution=service.resolve(
                game=game,stream_session_id=str(stream.active_stream_session_id),
                source_event_id=f"owner_game_state:{pending_id}",
                run_kind=str(updates.get("playthrough_type") or getattr(stream,"current_playthrough_type",None) or "unknown"),
            )
            if resolution.active_run:
                run_id=resolution.active_run.id;stream.active_game_run_id=run_id
                stream.current_game=resolution.game_identity.canonical_name
                if explicit_game:resolved_game=resolution.game_identity.canonical_name
        if not run_id:
            print(f"[HEBE][GAME_PENDING] state_update_rejected id={pending_id} reason=no_active_game_run",flush=True)
            return
        provenance=str(updates.pop("provenance","") or "")
        confidence=float(updates.pop("confidence",0.0) or 0.0)
        evidence=EvidenceRef(
            source_event_id=f"owner_game_state:{pending_id}",source_record_type="owner_stt",
            source_record_id=pending_id,observed_at=time.time(),extractor="game_guidance_clarification",
            extractor_version="v1",literal_span={"updates":sorted(updates)},
        )
        result=service.update_state(
            run_id,updates=updates,provenance=provenance,confidence=confidence,evidence=evidence,
        )
        accepted_updates=dict(result["accepted"])
        if resolved_game:accepted_updates["game"]=resolved_game
        if not accepted_updates:
            pending_id = state_update.data.get("pending_id")
            print(f"[HEBE][GAME_PENDING] state_update_rejected id={pending_id}", flush=True)
            return
        run=GameRunState.from_value(result["state"])
        self.runtime.state.game_run_state=run
        if stream is not None:self._project_canonical_game_run(stream,run_id)
        print(f"[HEBE][GAME_PENDING] consumed id={pending_id} fields_updated={sorted(accepted_updates)!r}", flush=True)
        print(
            f"[HEBE][GAME_RUN_STATE] updated game={run.game or 'unknown'} "
            f"location={run.current_location or 'unknown'} character={run.current_character or 'unknown'} "
            f"party={run.party_members!r}",
            flush=True,
        )
        log_jsonl_event("pending", {
            "event": "pending_consumed",
            "id": pending_id,
            "kind": "game_guidance_clarification",
            "compatibility_reason": "authorized_state_update",
            "fields_updated": sorted(accepted_updates),
        })
        pending = self._active_current_conversation(latest=True)
        if pending is not None and pending.topic == "game_guidance_clarification":
            print(f"[HEBE][PENDING_CONSUMED] kind=game_guidance_clarification id={pending.id} reason=game_run_state_updated", flush=True)
            self._close_current_conversation(reason="consumed", conversation=pending)
        log_jsonl_event("game_guidance", {
            "game": run.game,
            "location": run.current_location,
            "current_character": run.current_character,
            "party_members": run.party_members,
            "game_run_state": run.to_dict(),
            "needs_clarification": False,
            "clarification_pending_created": False,
            "reason": "game_run_state_updated",
        })
    
    def process_internal_event(self, event) -> None:
        event_type = str(getattr(event, "event_type", "") or "")
        if event_type == "twitch_outgoing_raid":
            self._handle_outgoing_raid(event)
            return
        if event_type == "twitch_raid":
            payload = getattr(event, "payload", {}) or {}
            stream = self._get_stream_state()
            duplicate = self._raid_duplicate_context(stream, payload)
            if duplicate is None and stream is not None:
                username = str(payload.get("user_login") or payload.get("display_name") or "").casefold()
                viewers = int(payload.get("viewer_count") or 0)
                now = time.time()
                duplicate = next((
                    dict(item) for item in reversed(list(getattr(stream, "recent_raid_contexts", []) or []))
                    if now - float(item.get("ts", 0.0) or 0.0) <= 60.0
                    and str(item.get("user_login") or item.get("display_name") or "").casefold() == username
                    and int(item.get("viewer_count") or 0) == viewers
                ), None)
            if duplicate is not None:
                print(
                    f"[HEBE][RAID_DEDUPE] stage=ingress duplicate=true original_event_id={duplicate.get('event_id') or ''} "
                    f"ignored_event_id={payload.get('event_id') or payload.get('message_id') or ''}",
                    flush=True,
                )
                return
            if stream is not None:
                self._remember_raid_context(stream, payload, thanked=False)
                payload["_raid_ingress_reserved"] = True
                event.payload = payload
        if event_type.startswith("twitch_"):
            self._get_twitch_interaction_coordinator().submit(event, self._process_internal_event_now)
            return
        self._process_internal_event_now(event)

    def _observe_social_world_event(self,event_type:str,payload:dict) -> None:
        if getattr(self,"social_world",None) is None:return
        if event_type not in {"twitch_follow","twitch_sub","twitch_raid"}:return
        login=str(payload.get("user_login") or payload.get("username") or payload.get("login") or "");user_id=str(payload.get("user_id") or payload.get("twitch_user_id") or "")
        if not login and not user_id:return
        stream=self._get_stream_state();session_id=str(getattr(stream,"active_stream_session_id","") or "");event_id=str(payload.get("event_id") or payload.get("message_id") or f"social:{event_type}:{user_id or login}")
        try:
            person,_,_=self.social_world.observe_presence(observation_id=event_id,platform="twitch",platform_user_id=user_id,login=login,display_name=str(payload.get("display_name") or login),stream_session_id=session_id,source=event_type,message_seen=False,direct_interaction=False)
            episode_type="resub" if event_type=="twitch_sub" and bool(payload.get("is_resub")) else "sub" if event_type=="twitch_sub" else "follow" if event_type=="twitch_follow" else "raid_arrival"
            self.social_world.record_episode(episode_type=episode_type,participant_ids=(person.person_id,),origin_event_id=event_id,summary=episode_type,salience_reason="platform_social_event",relevance_seconds=604800,retention_seconds=7776000,sensitivity="normal",retention_class="bounded",retrieval_scope="stream_public")
        except Exception as exc:print(f"[HEBE][SOCIAL_EPISODE] origin={event_id} admitted=false reason={type(exc).__name__}",flush=True)

    def _process_internal_event_now(self, event) -> None:
        if getattr(event, "event_type", None) in {"stream_online", "stream_offline"}:
            self._handle_stream_lifecycle_event(event)
            return
        event_type = str(getattr(event, "event_type", "") or "")
        payload = getattr(event, "payload", {}) or {}
        if event_type.startswith("twitch_") and isinstance(payload, dict):
            payload = self._enrich_stream_payload(payload)
            event.payload = payload
        self._observe_social_world_event(event_type,payload if isinstance(payload,dict) else {})
        if event_type == "twitch_cheer":
            self._handle_twitch_cheer_event(event)
            return
        event_decision = None
        if event_type.startswith("twitch_"):
            raw_text = str((payload or {}).get("message_text") or (payload or {}).get("text") or "")
            username = str((payload or {}).get("user_login") or (payload or {}).get("username") or "")
            event_id = str((payload or {}).get("event_id") or (payload or {}).get("message_id") or f"evt_{uuid.uuid4().hex}")
            if event_type == "twitch_chat_react" and not bool((payload or {}).get("_pipeline_started")):
                self._increment_twitch_pipeline_counter("twitch_messages_received")
                print(
                    f"[HEBE][TWITCH_PIPELINE_START] event_id={event_id} username={username} raw={raw_text!r}",
                    flush=True,
                )
            if username and self._is_known_twitch_bot_user(username):
                reason = "self_message_ignored" if username.casefold().lstrip("@") in {"hebenifelheim"} else "bot_ignored"
                self._increment_twitch_pipeline_counter(
                    "twitch_messages_self_ignored" if reason == "self_message_ignored" else "twitch_messages_bot_ignored"
                )
                self._increment_twitch_pipeline_counter("twitch_messages_early_skipped", reason=reason)
                if reason == "self_message_ignored":
                    print(f"[HEBE][SELF_MESSAGE_IGNORED] username={username}", flush=True)
                print(f"[HEBE][TWITCH_PIPELINE_SKIP] stage=bot_filter reason={reason} event_id={event_id} username={username}", flush=True)
                return
            source = "twitch_viewer" if event_type == "twitch_chat_react" else "twitch_system"
            direct_meta = self._twitch_direct_priority(raw_text, payload=payload) if event_type == "twitch_chat_react" else {}
            if isinstance(payload, dict) and direct_meta:
                payload.update({key: value for key, value in direct_meta.items() if value or key in {"reply_to_hebe_message", "mentions_hebe", "direct_address_to_hebe"}})
                event.payload = payload
            firewall = self._input_firewall_decision(
                source=source,
                text=raw_text,
                username=username,
                event_type=event_type,
                addressed_to_hebe=bool((payload or {}).get("direct_address_to_hebe") or (self._message_mentions_hebe(raw_text) if raw_text else False)),
            )
            if isinstance(payload, dict):
                payload["input_firewall"] = firewall.as_dict()
                event.payload = payload
            if not self._firewall_allows_pipeline(firewall):
                self._increment_twitch_pipeline_counter("twitch_messages_early_skipped", reason=firewall.reason)
                self._get_twitch_interaction_coordinator().record_policy_suppression(
                    event_id,
                    str(firewall.reason or "input_firewall_blocked"),
                )
                print(
                    f"[HEBE][TWITCH_PIPELINE_SKIP] stage=input_firewall reason={firewall.reason} event_id={event_id} username={username}",
                    flush=True,
                )
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
            if event_type == "twitch_chat_react":
                ownership = ConversationOwnershipGate().decide(raw_text, payload=payload)
                if isinstance(payload, dict):
                    payload["conversation_ownership"] = ownership.to_dict()
                    event.payload = payload
                print(
                    "[HEBE][CONVERSATION_OWNERSHIP] "
                    f"addressee={ownership.addressee} allowed={str(ownership.allow_assistant).lower()} "
                    f"reason={ownership.reason} confidence={ownership.confidence:.3f}",
                    flush=True,
                )
                if not ownership.allow_assistant:
                    self._increment_twitch_pipeline_counter("twitch_messages_early_skipped", reason=ownership.reason)
                    self._get_twitch_interaction_coordinator().record_policy_suppression(event_id, ownership.reason)
                    return
            stream = self._get_stream_state()
            route_context = SimpleNamespace(
                input_text=raw_text,
                internal_event=event,
                state_snapshot={"current_conversation": None},
                source=source,
                authority=firewall.authority,
                addressed_to_hebe=bool((payload or {}).get("direct_address_to_hebe") or self._message_mentions_hebe(raw_text)),
                firewall_decision=firewall.firewall_decision,
                stream_is_live=bool(
                    getattr(stream, "is_live", False)
                    or (getattr(stream, "enabled", False) and not getattr(stream, "live_status_known", False))
                ),
                route_hints=[],
            )
            event_decision = (getattr(self, "cognitive_router", None) or CognitiveRouter()).route(route_context)
            self._last_cognitive_trace = event_decision.to_dict()
            if event_type == "twitch_chat_react":
                print(
                    "[HEBE][COGNITIVE_ROUTER] "
                    f"source=twitch_chat event_id={event_id} should_reply={str(bool(getattr(event_decision, 'should_reply', False))).lower()} "
                    f"reason={getattr(event_decision, 'reason', '')} response_mode={getattr(event_decision, 'response_mode', '')}",
                    flush=True,
                )
            if (
                event_type == "twitch_chat_react"
                and bool(getattr(event_decision, "should_reply", False))
                and not bool(getattr(event_decision, "should_stop_pipeline", False))
                and (
                    "twitch.reply" in set(getattr(event_decision, "allowed_capabilities", []) or [])
                    or (hasattr(event_decision, "allows_capability") and event_decision.allows_capability("twitch.reply"))
                )
            ):
                print(
                    "[HEBE][POST_ROUTER_DISPATCH] "
                    f"event_id={getattr(event_decision, 'message_id', '')} should_reply=true next=twitch_response_pipeline",
                    flush=True,
                )
            if event_decision.should_stop_pipeline:
                if event_type == "twitch_chat_react":
                    stream = self._get_stream_state()
                    pre_route = self._pre_generation_twitch_route_decision(
                        payload=payload,
                        event_type=event_type,
                        stream=stream,
                    )
                    if not pre_route.get("should_generate", True):
                        self._set_last_twitch_route_state(
                            output_route=str(pre_route.get("route") or "observe_only"),
                            should_generate=False,
                            suppress_reason=str(pre_route.get("reason") or ""),
                            emitted_to_twitch=False,
                            tts_sent=False,
                            budget_result=pre_route.get("budget_result"),
                            thread_result=pre_route.get("thread_result"),
                        )
                        print(
                            "[HEBE][PRE_GENERATION_ROUTE_GATE] "
                            f"should_generate=false route={pre_route.get('route')} reason={pre_route.get('reason')}",
                            flush=True,
                        )
                        print(
                            "[HEBE][TWITCH_PIPELINE_FINAL] "
                            f"route={pre_route.get('route')} emitted=false reason={pre_route.get('reason')}",
                            flush=True,
                        )
                        return
                    print(
                        f"[HEBE][POST_ROUTER_DISPATCH] event_id={event_id} should_reply=true next=twitch_response_pipeline reason=presence_override",
                        flush=True,
                    )
                else:
                    print(f"[HEBE][EVENT_ROUTER] blocked type={event_type} reason={event_decision.reason}", flush=True)
                    return
        if event_type == "twitch_raid":
            self._handle_twitch_raid_event(event, cognitive_decision=event_decision)
            return

        if event_type in {"twitch_sub", "twitch_follow", "twitch_follow_batch", "twitch_join", "twitch_part"}:
            if not self._stream_event_public_ack_allowed(event_type, payload):
                return

        if event_type == "twitch_idle_prompt":
            mode, mode_reason = self._stream_voice_mode_active()
            if mode in {"muted", "wake_only"}:
                print(f"[HEBE][PROACTIVE_SUPPRESSED] reason={mode_reason or mode}", flush=True)
                return

        if event_type == "twitch_chat_react":
            explanation_trace_id = str((payload or {}).get("event_id") or (payload or {}).get("message_id") or f"evt_{uuid.uuid4().hex}")
            explanation = self._grounded_self_explanation(
                str((payload or {}).get("message_text") or (payload or {}).get("text") or ""),
                requester=str((payload or {}).get("display_name") or (payload or {}).get("user_login") or "viewer"),
                current_trace_id=explanation_trace_id,
            )
            if explanation.detected:
                self._get_twitch_interaction_coordinator().record_candidate(explanation_trace_id, explanation.text)
                self._deliver_twitch_reply(explanation.text, event_type=event_type, payload=payload)
                return
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
                        self._get_twitch_interaction_coordinator().record_candidate(
                            str((payload or {}).get("event_id") or (payload or {}).get("message_id") or ""),
                            policy_reply,
                        )
                        self._update_policy_trace_response(
                            policy_reply,
                            response_source=str(policy_reply_result.get("response_source") or "hybrid"),
                            style_guard_triggered=bool(policy_reply_result.get("style_guard_triggered")),
                            was_generic_refusal_rewritten=bool(policy_reply_result.get("was_generic_refusal_rewritten")),
                            style_profile=str(policy_reply_result.get("style_profile") or ""),
                            blocked_behavior=str(policy_reply_result.get("blocked_behavior") or ""),
                            generation_outcome=str(policy_reply_result.get("generation_outcome") or "generated"),
                        )
                        self._deliver_twitch_reply(
                            policy_reply,
                            event_type=event_type,
                            payload=payload,
                        )
                    else:
                        self._get_twitch_interaction_coordinator().record_policy_suppression(
                            str((payload or {}).get("event_id") or (payload or {}).get("message_id") or ""),
                            str(policy_decision.reason or "empty_policy_boundary"),
                        )
                        print(
                            f"[HEBE][VIEWER_POLICY] decision=ignored reason={policy_decision.reason}",
                            flush=True,
                        )
                else:
                    self._get_twitch_interaction_coordinator().record_policy_suppression(
                        str((payload or {}).get("event_id") or (payload or {}).get("message_id") or ""),
                        str(policy_decision.reason or "viewer_policy_blocked"),
                    )
                    print(
                        f"[HEBE][VIEWER_POLICY] decision=ignored reason={policy_decision.reason}",
                        flush=True,
                    )
                return
            stream = self._get_stream_state()
            pre_route = self._pre_generation_twitch_route_decision(
                payload=payload,
                event_type=event_type,
                stream=stream,
            )
            if not pre_route.get("should_generate", True):
                self._get_twitch_interaction_coordinator().record_emission(
                    str((payload or {}).get("event_id") or (payload or {}).get("message_id") or ""),
                    {"emitted": False, "route": str(pre_route.get("route") or "observe_only")},
                    reason=str(pre_route.get("reason") or "pre_generation_suppressed"),
                )
                self._set_last_twitch_route_state(
                    output_route=str(pre_route.get("route") or "observe_only"),
                    should_generate=False,
                    suppress_reason=str(pre_route.get("reason") or ""),
                    emitted_to_twitch=False,
                    tts_sent=False,
                    budget_result=pre_route.get("budget_result"),
                    thread_result=pre_route.get("thread_result"),
                )
                print(
                    "[HEBE][PRE_GENERATION_ROUTE] "
                    f"should_generate=false route={pre_route.get('route')} reason={pre_route.get('reason')}",
                    flush=True,
                )
                print(
                    "[HEBE][PRE_GENERATION_ROUTE_GATE] "
                    f"should_generate=false route={pre_route.get('route')} reason={pre_route.get('reason')}",
                    flush=True,
                )
                print(
                    "[HEBE][OUTPUT_ROUTE_DECISION] "
                    f"route={pre_route.get('route')} reason={pre_route.get('reason')} public=false tts=false "
                    f"value_score={float(pre_route.get('value_score') or 0.0):.2f}",
                    flush=True,
                )
                print(
                    "[HEBE][TWITCH_PIPELINE_FINAL] "
                    f"route={pre_route.get('route')} emitted=false reason={pre_route.get('reason')}",
                    flush=True,
                )
                self._update_policy_trace_response(
                    "",
                    candidate_response="",
                    suppress_reason=str(pre_route.get("reason") or ""),
                    output_route=str(pre_route.get("route") or "observe_only"),
                    public_sent=False,
                    tts_sent=False,
                    reply_value_score=pre_route.get("value_score"),
                    budget_result=pre_route.get("budget_result"),
                    twitch_message_category=pre_route.get("twitch_message_category"),
                    should_generate=False,
                    thread_result=pre_route.get("thread_result"),
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
                if event_type == "twitch_idle_prompt" and isinstance(live_context, dict):
                    raw_live_state = dict(live_context.get("live_state") or {})
                    current_live_state = {
                        key: raw_live_state.get(key)
                        for key in (
                            "current_game", "current_category", "current_activity",
                            "current_objective", "current_location", "current_phase",
                            "stream_session_id",
                        )
                        if raw_live_state.get(key) not in (None, "", [], {})
                    }
                    live_context = {
                        "live_state": current_live_state,
                        "scene_guard": dict(payload.get("scene_guard") or {}),
                        "selected_anchor_id": str(
                            (payload.get("anchor_evidence") or {}).get("anchor_id")
                            or payload.get("used_fact_id")
                            or ""
                        ),
                        "context_scope": "current_scene_selected_anchor_only",
                    }
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
        conversation_state = self._get_conversation_state_resolver().from_conversation(
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
        if event_type == "twitch_chat_react":
            self._record_response_generation_outcome(
                str((payload or {}).get("event_id") or (payload or {}).get("message_id") or ""),
                reply_text=reply_text,
            )
            self._increment_twitch_pipeline_counter("twitch_messages_generated")
            self._get_twitch_interaction_coordinator().record_candidate(
                str((payload or {}).get("event_id") or (payload or {}).get("message_id") or ""),
                reply_text,
            )

        print(
            "[HEBE][EVENT] "
            f"type={event.event_type!r} "
            f"reply={reply_text!r}",
            flush=True,
        )

        if not reply_text:
            if event_type == "twitch_chat_react" and event_decision is not None and getattr(event_decision, "should_reply", False):
                failure_reason = "empty_response_after_generation"
                print(
                    f"[HEBE][POST_ROUTER_DROP_GUARD] dropped=true reason={failure_reason}",
                    flush=True,
                )
                print(
                    f"[HEBE][OUTPUT_ROUTE_DECISION] route=suppress reason={failure_reason} public=false tts=false value_score=0.00",
                    flush=True,
                )
                self._increment_twitch_pipeline_counter("twitch_messages_suppressed", reason=failure_reason)
                self._set_last_twitch_route_state(
                    output_route="suppress",
                    should_generate=True,
                    suppress_reason=failure_reason,
                    emitted_to_twitch=False,
                    tts_sent=False,
                )
                print(
                    f"[HEBE][TWITCH_PIPELINE_FINAL] route=suppress emitted=false reason={failure_reason}",
                    flush=True,
                )
                self._emit_final_response(
                    event_id=str((payload or {}).get("event_id") or (payload or {}).get("message_id") or getattr(event_decision, "message_id", "")),
                    source="twitch",
                    final_response="",
                    output_route=OutputRoute.SUPPRESS,
                    output_targets=[],
                    guard_result={"passed": False, "reason": failure_reason},
                    debug_payload=self._latest_response_debug_payload(),
                )
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
                behavior_correlation_id = str(
                    (getattr(event, "payload", {}) or {}).get("behavior_correlation_id") or ""
                )
                speech_intent_id = str(
                    (getattr(event, "payload", {}) or {}).get("speech_intent_id") or ""
                )
                adaptation_service = getattr(self, "behavior_adaptation", None)
                if adaptation_service is None:
                    adaptation_service = BehaviorAdaptationService()
                    self.behavior_adaptation = adaptation_service
                adaptation = adaptation_service.validate_generated_output(
                    stream,
                    reply_text,
                    topic=str((getattr(event, "payload", {}) or {}).get("idle_topic") or ""),
                    mode="proactive",
                    observation={
                        "trace_id": behavior_correlation_id,
                        "candidate_id": speech_intent_id,
                        "speech_intent_id": speech_intent_id,
                        "speech_intent": str(
                            (getattr(event, "payload", {}) or {}).get("speech_intent_type") or ""
                        ),
                    },
                )
                if adaptation.action == AdaptationAction.SUPPRESS:
                    service.consume_opportunity(
                        stream,
                        getattr(event, "payload", {}) or {},
                        reason=f"behavior_adaptation_{adaptation.action.value}",
                    )
                    print(
                        f"[HEBE][SPONTANEITY] skipped reason=behavior_adaptation_{adaptation.action.value} "
                        f"motif={adaptation.motif_id} fatigue={adaptation.fatigue:.3f}",
                        flush=True,
                    )
                    adaptation_service.record_emission(
                        trace_id=adaptation.trace_id,
                        stream=stream,
                        event_id=speech_intent_id,
                        emitted=False,
                        reason_code=adaptation.reason,
                    )
                    self._record_interaction_decision({
                        "trace_id": str(adaptation.trace_id or behavior_correlation_id or speech_intent_id or f"behavior_{uuid.uuid4().hex}"),
                        "event_id": str(speech_intent_id or adaptation.trace_id or ""),
                        "source": "behavior_adaptation",
                        "actor": "Hebe",
                        "actor_identities": ["Hebe"],
                        "target": "stream",
                        "interaction_decision": "behavior_candidate_suppressed",
                        "authority": "system",
                        "requested_effect": "behavior_expression",
                        "effect_authorized": False,
                        "reply_authorized": False,
                        "reason_code": f"behavior_{adaptation.reason}",
                        "response_intent": str((getattr(event, "payload", {}) or {}).get("speech_intent_type") or "proactive_gag"),
                        "generation_outcome": "generated",
                        "emission_outcome": "suppressed",
                    })
                    return
                stream.last_behavior_correlation_id = adaptation.trace_id
                print("[HEBE][SPONTANEITY] spoken reason=canonical_behavior_policy_validated", flush=True)

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
            stream.twitch_stream_id = payload.get("twitch_stream_id") or payload.get("id") or stream.twitch_stream_id
            stream.idle_prompts_sent_stream = 0
            stream.recent_idle_messages = []
            loop = getattr(self, "stream_companion_loop", None)
            if loop is not None:
                loop.reset_session()
                stream.speech_intent_state = loop.intent_manager.snapshot()
            self._auto_enable_stream_if_live(stream, source="stream_online_event")
            session_id = self._ensure_stream_memory_session_if_live(stream)
            self._record_stream_event_safe("stream_online", payload, stream=stream)
            try:
                self._get_live_session_brain().observe_stream_metadata(stream, source="stream_online")
            except Exception as exc:
                print(f"[HEBE][LIVE_SESSION] stream_online failed: {exc!r}", flush=True)
            self.poll_stream_context(force=True, require_enabled=False)
            self._prepare_live_game_intelligence(stream)
            print("[HEBE][STREAM_CONTEXT] stream_online event handled", flush=True)
            return
        if event.event_type == "stream_offline":
            loop = getattr(self, "stream_companion_loop", None)
            if loop is not None:
                loop.reset_session()
                stream.speech_intent_state = loop.intent_manager.snapshot()
            active_game_run_id=str(getattr(stream,"active_game_run_id","") or "")
            active_stream_session_id=str(getattr(stream,"active_stream_session_id","") or "")
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
            if active_game_run_id and getattr(self,"game_run_service",None) is not None:
                try:self.game_run_service.pause(active_game_run_id,stream_session_id=active_stream_session_id,event_id=str(getattr(event,"event_id","") or f"stream_offline:{active_stream_session_id}"))
                except Exception as exc:print(f"[HEBE][GAME_RUN_RESOLVE] decision=pause_failed reason={type(exc).__name__}",flush=True)
            self._close_stream_memory_session_safe(stream, reason="stream_offline_event")
            print("[HEBE][STREAM_CONTEXT] stream_offline event handled", flush=True)

    def _prepare_live_game_intelligence(self, stream) -> None:
        game = str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or "").strip()
        session_id = getattr(stream, "active_stream_session_id", None)
        if not game or not session_id or getattr(self,"game_run_service",None) is None:
            return
        try:
            title=str(getattr(stream,"current_stream_title",None) or "")
            run_kind=str(getattr(stream,"current_playthrough_type",None) or "unknown").strip().lower().replace(" ","_") or "unknown"
            rules={"title":title[:200]}
            challenge=str(getattr(stream,"current_challenge",None) or "").strip()
            if challenge:rules["challenge"]=challenge
            resolution=self.game_run_service.resolve(game=game,stream_session_id=str(session_id),source_event_id=f"stream_session:{session_id}",run_kind=run_kind,rules=rules)
            if resolution.active_run:
                setattr(stream,"active_game_run_id",resolution.active_run.id)
                self._project_canonical_game_run(stream,resolution.active_run.id)
            session_primer.record_schedule_observation(
                stream_session_id=session_id,
                canonical_content=game,
                stream_format=session_primer.infer_stream_format(
                    game, str(getattr(stream, "current_playthrough_type", None) or ""),
                ),
            )
            print(f"[HEBE][GAME_RESEARCH_SKIP] reason=memory_first_no_typed_gap game={game}",flush=True)
        except Exception as exc:
            print(f"[HEBE][GAME_DOSSIER] game={game} status=failed facts=0 sources=0 error={type(exc).__name__}", flush=True)

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

    def _stream_event_public_ack_allowed(self, event_type: str, payload: dict | None) -> bool:
        payload = payload or {}
        stream = self._get_stream_state()
        source = str(payload.get("source") or "").strip() or "unknown"
        visible_public = bool(payload.get("visible_public") or payload.get("source") in {"irc_usernotice", "chat_usernotice", "public_alert"})
        passive_eventsub = bool(payload.get("passive_eventsub") or source == "eventsub")
        if event_type in {"twitch_join", "twitch_part"}:
            route, allowed, reason = "observe_only", False, "join_part_never_public"
        elif event_type == "twitch_raid":
            route, allowed, reason = "twitch_text_reply", True, "raid_always_public"
        elif event_type in {"twitch_sub", "twitch_follow", "twitch_follow_batch"}:
            env_name = "HEBE_STREAM_SUB_AUTO_THANK_PUBLIC" if event_type == "twitch_sub" else "HEBE_STREAM_FOLLOW_AUTO_THANK_PUBLIC"
            configured = os.getenv(env_name, "false").strip().lower() in {"1", "true", "yes", "on"}
            passive_public = os.getenv("HEBE_PASSIVE_EVENTSUB_PUBLIC_CALLOUTS", "false").strip().lower() in {"1", "true", "yes", "on"}
            allowed = bool(configured and (visible_public or (passive_eventsub and passive_public)))
            route = "twitch_text_reply" if allowed else "observe_only"
            reason = "configured_visible_public" if allowed else "passive_eventsub_public_disabled" if passive_eventsub else "auto_thank_disabled"
        else:
            route, allowed, reason = "observe_only", False, "unsupported_stream_event"
        print(f"[HEBE][STREAM_EVENT_VISIBILITY] type={event_type} visible_public={str(visible_public).lower()} source={source}", flush=True)
        print(f"[HEBE][STREAM_EVENT_ACK_DECISION] type={event_type} route={route} reason={reason}", flush=True)
        if stream is not None:
            stream.last_stream_event_ack_decision = {
                "type": event_type,
                "route": route,
                "reason": reason,
                "visible_public": visible_public,
                "source": source,
                "allowed": allowed,
                "ts": time.time(),
            }
        return allowed

    def _effective_stream_audio_state(self, event_type: str = "") -> EffectiveStreamAudioState:
        stream = self._get_stream_state()
        policies = getattr(stream, "policies", None) if stream is not None else None
        mode, _ = self._stream_voice_mode_active()
        route_enabled = bool(
            self._stream_output_mode() == "tts_enabled"
            and policies
            and (
                getattr(policies, "allow_tts_event_replies", True)
                if event_type.startswith("twitch_")
                else getattr(policies, "allow_tts_replies", False)
            )
        )
        tts_service = getattr(self.runtime, "tts", None)
        engine_ready = bool(callable(getattr(self.runtime, "speak", None)) and not getattr(tts_service, "failed", False))
        state = EffectiveStreamAudioState.resolve(
            configured=bool(getattr(self.runtime.state, "tts_enabled", False)),
            engine_ready=engine_ready,
            route_enabled=route_enabled,
            muted=mode in {"muted", "wake_only"},
        )
        print(
            "[HEBE][STREAM_TTS_STATE] "
            f"configured={str(state.configured).lower()} ready={str(state.engine_ready).lower()} "
            f"enabled={str(state.route_enabled).lower()} effective={str(state.actual_can_speak).lower()} "
            f"reason={state.blocked_reason or 'available'}",
            flush=True,
        )
        return state

    def _handle_twitch_cheer_event(self, event) -> None:
        payload = dict(getattr(event, "payload", {}) or {})
        try:
            cheer = TwitchCheerEvent(
                event_id=str(payload.get("event_id") or payload.get("twitch_message_id") or f"cheer_{uuid.uuid4().hex}"),
                source=str(payload.get("source") or "eventsub"),
                viewer_login=str(payload.get("viewer_login") or payload.get("user_login") or "viewer"),
                viewer_display_name=str(payload.get("viewer_display_name") or payload.get("display_name") or payload.get("viewer_login") or "viewer"),
                bits=int(payload.get("bits") or 0),
                message=str(payload.get("message") or payload.get("message_text") or ""),
                timestamp=float(payload.get("timestamp") or time.time()),
                twitch_message_id=str(payload.get("twitch_message_id") or payload.get("message_id") or ""),
                dedupe_key=str(payload.get("dedupe_key") or ""),
                raw_tags=dict(payload.get("raw_tags") or payload.get("irc_tags") or {}),
            )
        except (TypeError, ValueError):
            print("[HEBE][CHEER_ACK_DECISION] allowed=false route=suppress reason=invalid_cheer_payload", flush=True)
            return
        if cheer.bits <= 0:
            print("[HEBE][CHEER_ACK_DECISION] allowed=false route=suppress reason=invalid_bits", flush=True)
            return
        routed = self.stream_social_event_router.route(cheer)
        print(f"[HEBE][STREAM_SOCIAL_EVENT] type=cheer event_id={routed['event_id']}", flush=True)
        duplicate, dedupe_reason = self.cheer_deduplicator.check_and_record(cheer)
        stream = self._get_stream_state()
        if stream is not None:
            stream.last_cheer_event = cheer.to_dict()
            stream.last_cheer_dedupe_result = {"duplicate": duplicate, "reason": dedupe_reason}
        decision = self.cheer_event_policy.decide(cheer, duplicate=duplicate)
        audio = self._effective_stream_audio_state("twitch_cheer")
        fallback_text = os.getenv("HEBE_CHEER_TWITCH_TEXT_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}
        both = os.getenv("HEBE_CHEER_OUTPUT_BOTH", "false").strip().lower() in {"1", "true", "yes", "on"}
        twitch = getattr(self.runtime, "twitch", None)
        twitch_available = bool(twitch is not None and twitch.is_available())
        route = "stream_tts_reply" if audio.actual_can_speak else "twitch_text_reply" if fallback_text and twitch_available else "suppress"
        if both and audio.actual_can_speak:
            route = "stream_tts_reply+twitch_text_reply"
        if decision["allowed"] and route == "suppress":
            decision = {**decision, "allowed": False, "reason": audio.blocked_reason or "no_output_route"}
        decision["route"] = route if decision["allowed"] else "suppress"
        print(
            f"[HEBE][CHEER_ACK_DECISION] allowed={str(bool(decision['allowed'])).lower()} "
            f"route={decision['route']} reason={decision['reason']}", flush=True,
        )
        if not decision["allowed"]:
            if stream is not None:
                stream.last_cheer_ack_result = {"emitted": False, "reason": decision["reason"], "route": "suppress"}
            print(f"[HEBE][CHEER_ACK_EMITTED] viewer={cheer.viewer_login} bits={cheer.bits} public_sent=false tts_sent=false", flush=True)
            return
        text = self.cheer_ack_renderer.render(cheer)
        guard = self._cheer_anti_bait_guard(text, payload=payload)
        if not guard.get("passed"):
            print(f"[HEBE][CHEER_ACK_DECISION] allowed=false route=suppress reason=cheer_guard", flush=True)
            return
        targets = [OUTPUT_TARGET_LOCAL_UI]
        if audio.actual_can_speak:
            targets.append(OUTPUT_TARGET_STREAM_TTS)
        if (not audio.actual_can_speak and fallback_text and twitch_available) or (both and twitch_available):
            targets.append(OUTPUT_TARGET_TWITCH_CHAT)
        public_sent = OUTPUT_TARGET_TWITCH_CHAT in targets
        tts_sent = OUTPUT_TARGET_STREAM_TTS in targets

        def speak(final_text: str) -> None:
            self._remember_tts_text(final_text)
            self.runtime.speak(final_text, emit_chat=False)

        def send(final_text: str) -> None:
            if twitch is not None and twitch.is_available():
                twitch.send_message(final_text)

        self._get_twitch_interaction_coordinator().record_candidate(cheer.event_id, text)
        result = self._emit_final_response(
            event_id=cheer.event_id, source="twitch", final_response=text,
            output_route=OutputRoute.STREAM_TTS_REPLY if tts_sent else OutputRoute.TWITCH_TEXT_REPLY,
            output_targets=targets, guard_result=guard,
            debug_payload={"event_type": "twitch_cheer", "open_pending": False},
            send_twitch_fn=send if public_sent else None, speak_fn=speak if tts_sent else None,
        )
        emitted = bool(result.get("emitted"))
        if stream is not None:
            stream.last_cheer_ack_result = {
                "emitted": emitted, "reason": result.get("reason") or decision["reason"],
                "route": decision["route"], "public_sent": public_sent and emitted,
                "tts_sent": tts_sent and emitted,
            }
        print(
            f"[HEBE][CHEER_ACK_EMITTED] viewer={cheer.viewer_login} bits={cheer.bits} "
            f"public_sent={str(public_sent and emitted).lower()} tts_sent={str(tts_sent and emitted).lower()}",
            flush=True,
        )

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
        duplicate_context = None if payload.get("_raid_ingress_reserved") else self._raid_duplicate_context(stream, payload)
        if duplicate_context is not None:
            if stream is not None:
                stream.last_raid_ack_result = {
                    "emitted": False,
                    "reason": "duplicate_raid_event",
                    "source": str(payload.get("source") or ""),
                    "display_name": username,
                    "viewer_count": viewers,
                    "ts": time.time(),
                }
            print(
                f"[HEBE][RAID_ACK_DECISION] allowed=false reason=duplicate_raid_event source={payload.get('source') or ''} from={username}",
                flush=True,
            )
            print(
                f"[HEBE][RAID_DEDUPE] duplicate=true original_event_id={duplicate_context.get('event_id') or ''} "
                f"ignored_event_id={payload.get('event_id') or payload.get('message_id') or ''}",
                flush=True,
            )
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
        reducer=getattr(self,"scene_consequence_reducer",None)
        if reducer is not None and bool(getattr(self,"scene_consequence_v2",False)):
            reducer.incoming_raid(event_id=str(payload.get("event_id") or payload.get("message_id") or f"incoming_raid_{uuid.uuid4().hex}"),source=str(payload.get("user_login") or username),viewer_count=viewers)
        is_simulated = bool(payload.get("_simulated"))
        if stream is not None:
            stream.last_raid_event = {
                "display_name": username,
                "user_login": payload.get("user_login") or username,
                "viewer_count": viewers,
                "source": payload.get("source") or "",
                "ts": time.time(),
            }
            self._remember_raid_context(stream, payload, thanked=False)
            self._reset_twitch_reply_budget(stream, "stream_social_event")
            if is_simulated:
                print("[HEBE][STREAM_SESSION] skipped reason=simulation", flush=True)
            else:
                self._ensure_stream_memory_session_if_live(stream)
                self._record_stream_event_safe("twitch_raid", payload, stream=stream)

        if not stream:
            print("[HEBE][TWITCH][RAID] blocked reason=no_stream_state", flush=True)
            return
        if not (getattr(stream, "enabled", False) or getattr(stream, "is_live", False)):
            print("[HEBE][TWITCH][RAID] blocked reason=stream_not_enabled_and_not_live", flush=True)
            return

        print("[HEBE][TWITCH][RAID] planned thank-you", flush=True)
        print(
            f"[HEBE][RAID_ACK_DECISION] allowed=true reason=stream_social_event from={username} viewers={viewers}",
            flush=True,
        )
        reply_text, fallback_used, render_error = self._render_raid_ack_safe(event, cognitive_decision=cognitive_decision)
        if not reply_text:
            if stream is not None:
                stream.last_raid_ack_result = {
                    "emitted": False,
                    "reason": "empty_reply",
                    "display_name": username,
                    "viewer_count": viewers,
                    "ts": time.time(),
                }
                stream.last_raid_ack_error = {"error": render_error or "empty_reply", "fallback_used": fallback_used, "ts": time.time()}
            print("[HEBE][TWITCH][RAID] blocked reason=empty_reply", flush=True)
            return
        raid_tts_enabled = bool(
            not is_simulated
            and self._stream_tts_output_enabled_for_event("twitch_raid")
            and getattr(self.runtime.state, "tts_enabled", False)
        )
        payload["_async_tts_managed"] = raid_tts_enabled
        payload["_force_skip_tts"] = not raid_tts_enabled
        text_started = time.perf_counter()
        try:
            self._deliver_twitch_reply(reply_text, event_type="twitch_raid", payload=payload)
        except Exception as exc:
            if stream is not None:
                stream.last_raid_ack_result = {
                    "emitted": False,
                    "reason": "delivery_failed",
                    "display_name": username,
                    "viewer_count": viewers,
                    "ts": time.time(),
                }
                stream.last_raid_ack_error = {"error": f"{type(exc).__name__}: {exc}", "fallback_used": fallback_used, "ts": time.time()}
            print(f"[HEBE][RAID_ACK_ERROR] error={type(exc).__name__}: {exc} fallback_used={str(fallback_used).lower()}", flush=True)
            return
        text_latency_ms = (time.perf_counter() - text_started) * 1000
        print(
            f"[HEBE][RAID_ACK_TEXT] sent=true latency_ms={text_latency_ms:.0f}",
            flush=True,
        )
        tts_scheduled = False
        if not is_simulated:
            scheduled = self._get_stream_tts_safety().schedule(
                reply_text,
                lambda value: self.runtime.speak(value, emit_chat=False),
                event_type="raid",
                output_enabled=raid_tts_enabled,
                disabled_reason="stream_tts_disabled",
            )
            tts_scheduled = bool(scheduled.get("scheduled"))
        print(
            f"[HEBE][RAID_ACK_COMPLETE] text_sent=true tts_scheduled={str(tts_scheduled).lower()}",
            flush=True,
        )
        if stream is not None:
            self._remember_raid_context(stream, payload, thanked=True)
            stream.last_raid_ack_result = {
                "emitted": True,
                "reason": "sent",
                "display_name": username,
                "user_login": payload.get("user_login") or username,
                "viewer_count": viewers,
                "source": payload.get("source") or "",
                "ts": time.time(),
            }
            if render_error:
                stream.last_raid_ack_error = {"error": render_error, "fallback_used": fallback_used, "ts": time.time()}
        trace = self.get_last_policy_trace()
        public_sent = bool(trace.get("public_sent", True))
        tts_sent = bool(trace.get("tts_sent", False))
        print(f"[HEBE][RAID_ACK_EMITTED] raider={username} public_sent={str(public_sent).lower()} tts_sent={str(tts_sent).lower()}", flush=True)
        print("[HEBE][TWITCH][RAID] sent thank-you", flush=True)
        if is_simulated:
            print("[HEBE][PROMOTION_GATE] blocked reason=simulation_mode target={}".format(payload.get("user_login") or username), flush=True)
            return
        if cognitive_decision is None or not hasattr(cognitive_decision, "allows_capability") or not cognitive_decision.allows_capability("twitch.promotion"):
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
            self._sync_game_run_state(stream, provenance="stream_context_sync")
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
        self._sync_game_run_state(stream,provenance="manual_update")
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
        self._sync_game_run_state(stream, provenance="manual_update")

    def _sync_game_run_state(self, stream, *, provenance: str) -> None:
        game = str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or "").strip()
        service=getattr(self,"game_run_service",None)
        if not game or service is None:
            return
        run_id=str(getattr(stream,"active_game_run_id","") or "")
        run=service.repository.get_run(run_id) if run_id else None
        identity=service.repository.resolve_identity(game)
        if run is None or run.game_id!=identity.game_id:
            session_id=str(getattr(stream,"active_stream_session_id","") or "")
            if not session_id:
                self.runtime.state.game_run_state=GameRunState(game=identity.canonical_name,provenance=provenance)
                return
            resolution=service.resolve(
                game=game,stream_session_id=session_id,source_event_id=f"stream_context:{session_id}",
                run_kind=str(getattr(stream,"current_playthrough_type",None) or "unknown"),
            )
            run=resolution.active_run
            if run is None:return
            stream.active_game_run_id=run.id
        self._project_canonical_game_run(stream,run.id)

    def _project_canonical_game_run(self, stream, run_id: str) -> None:
        service=getattr(self,"game_run_service",None)
        if service is None:return
        state=service.state(run_id)
        self.runtime.state.game_run_state=GameRunState.from_value(state)
        stream.current_game=state["game"]
        stream.current_category=state["game"]
        stream.current_run_location=state["current_location"] or None
        stream.current_run_objective=state["current_objective"] or None
        stream.current_run_phase=state["last_confirmed_progress"] or None
        stream.current_challenge=state["challenge"] or None
        stream.current_playthrough_type=state["playthrough_type"] or None

    def _persist_canonical_run_state(self, stream, updates: dict[str, object], *, source: str) -> bool:
        service=getattr(self,"game_run_service",None)
        run_id=str(getattr(stream,"active_game_run_id","") or "")
        if service is None or not run_id:return False
        event_id=f"manual_game_state:{uuid.uuid4().hex}"
        evidence=EvidenceRef(
            source_event_id=event_id,source_record_type="owner_command",source_record_id=event_id,
            observed_at=time.time(),extractor="manual_game_context",extractor_version="v1",
            literal_span={"fields":sorted(updates)},
        )
        result=service.update_state(
            run_id,updates=updates,provenance="manual_command",confidence=1.0,evidence=evidence,
        )
        if result["accepted"]:
            self._project_canonical_game_run(stream,run_id)
            stream.run_context_updated_ts=time.time();stream.run_context_source=source
            return True
        return False

    def _clear_canonical_run_state(self, stream, fields: tuple[str, ...]) -> None:
        service=getattr(self,"game_run_service",None);run_id=str(getattr(stream,"active_game_run_id","") or "")
        if service is None or not run_id:return
        event_id=f"manual_game_state_clear:{uuid.uuid4().hex}"
        service.clear_state(
            run_id,fields=fields,evidence=EvidenceRef(
                source_event_id=event_id,source_record_type="owner_command",source_record_id=event_id,
                relation=EvidenceRelation.CORRECTS,observed_at=time.time(),
                extractor="manual_game_context",extractor_version="v1",literal_span={"fields":list(fields)},
            ),
        )
        self._project_canonical_game_run(stream,run_id)

    def _maybe_research_game_after_context_sync(self, stream) -> None:
        category = getattr(stream, "current_category", None) or getattr(stream, "current_game", None)
        if not category or category == getattr(self, "_last_game_research_category", None):
            return
        self._last_game_research_category = category
        # The legacy service performs provider I/O synchronously. All automatic
        # category research now enters through the bounded background job.
        self._prepare_live_game_intelligence(stream)

    def poll_stream_presence(self) -> None:
        now = time.time()
        game_intelligence = getattr(self, "game_intelligence", None)
        if game_intelligence is not None:
            try:
                game_intelligence.retry_due_jobs()
                for job_id in list(getattr(game_intelligence, "_jobs", {}).keys()):
                    job, _facts = game_intelligence.collect_job(job_id, scene_still_current=True)
                    if (
                        job.status == "completed"
                        and job.mode == ResearchMode.PRE_STREAM_DOSSIER.value
                        and not job.metadata.get("next_unit_admitted")
                    ):
                        job.metadata["next_unit_admitted"] = True
                        game_intelligence.prepare_game_async(
                            game_title=str(job.metadata.get("game_title") or job.game_id),
                            platform=str(job.metadata.get("platform") or ""),
                            version=str(job.metadata.get("version") or ""),
                            session_id=job.session_id,
                        )
                    # Completed and failed attempts remain as the session-scoped
                    # idempotency ledger. Only explicitly invalidated work is removed.
                    if job.status in {"cancelled", "stale"}:
                        getattr(game_intelligence, "_jobs", {}).pop(job_id, None)
            except Exception as exc:
                print(f"[HEBE][GAME_RESEARCH_JOB] status=failed error={type(exc).__name__}", flush=True)
        auto_promotions = getattr(self, "automatic_promotions", None)
        if auto_promotions is not None:
            try:
                auto_promotions.drain_ready(self._send_automatic_promotion, now=now)
            except Exception as exc:
                print(f"[HEBE][AUTO_PROMO_OUTCOME] viewer=queued status=failed reason={type(exc).__name__}", flush=True)
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

        self._poll_owner_discourse_opportunity(stream, now=now)

        loop = getattr(self, "stream_companion_loop", None)
        if loop is None:
            service = getattr(self, "stream_spontaneity", None)
            if service is None:
                service = StreamSpontaneityService()
                self.stream_spontaneity = service
            loop = StreamCompanionLoop(
                spontaneity=service,
                presence_engine=self._get_presence_engine(),
                scene_timeline=getattr(self, "scene_timeline", None),
                opportunities=getattr(self, "spontaneous_opportunities", None),
                owner_voice_active_fn=self._owner_audio_active,
                tts_active_fn=lambda: bool(getattr(self, "_tts_active", False)),
            )
            self.stream_companion_loop = loop

        output_mode = self._stream_output_mode()
        stream_tts_allowed = bool(output_mode == "tts_enabled")
        self._queue_cognitive_speech_intent_candidates(stream, loop=loop, now=now)
        tick = loop.evaluate(
            stream,
            stream_tts_enabled=stream_tts_allowed,
            output_mode=output_mode,
            backend_running=not getattr(getattr(self, "_stop_event", None), "is_set", lambda: False)(),
        )
        if tick is None or tick.event is None:
            return

        print(
            "[HEBE][PRESENCE] enqueue "
            f"type={tick.event.event_type!r} mode={tick.event.payload.get('presence_mode')!r}",
            flush=True,
        )
        self.process_internal_event(tick.event)

    def _queue_cognitive_speech_intent_candidates(self, stream, *, loop, now: float) -> None:
        """Project already-grounded Cognitive v2 material into transient speech candidates.

        This scan is bounded and model-free. It never grants action authority and
        does not render text; Presence and turn arbitration still decide whether
        a candidate can proceed.
        """
        if stream is None or loop is None:
            return
        if now - float(getattr(stream, "last_cognitive_intent_scan_ts", 0.0) or 0.0) < 5.0:
            return
        stream.last_cognitive_intent_scan_ts = now
        candidates = list(getattr(stream, "speech_intent_candidates", []) or [])
        manager = getattr(loop, "intent_manager", None)

        def add(candidate: dict) -> None:
            source_ids = [str(item) for item in candidate.get("source_event_ids") or [] if str(item)]
            if any(manager and manager.has_seen_source(item) for item in source_ids):
                return
            if any(set(source_ids) & set(existing.get("source_event_ids") or []) for existing in candidates):
                return
            candidates.append(candidate)

        plan = dict(getattr(stream, "proposed_discourse_contribution", None) or {})
        if bool(plan.get("should_contribute")) and float(plan.get("contribution_value") or 0.0) >= 0.62:
            topic_id = str(plan.get("topic_id") or "")
            add({
                "type": "OPINION", "topic": topic_id or "current_discourse",
                "value": float(plan.get("contribution_value") or 0.62), "urgency": 0.35,
                "source_event_ids": [f"discourse:{topic_id}"],
                "material": {
                    "grounded_fragments": list(plan.get("grounded_fragments") or []),
                    "proposed_claims": list(plan.get("proposed_claims") or []),
                },
            })

        for opportunity in list(getattr(getattr(self, "social_world", None), "last_opportunities", []) or [])[:3]:
            if not opportunity.get("scene_suitable") or opportunity.get("consumed"):
                continue
            opportunity_id = str(opportunity.get("id") or "")
            add({
                "type": "SOCIAL_FOLLOWUP", "topic": str(opportunity.get("opportunity_type") or "social_followup"),
                "subject_ref": str(opportunity.get("person_id") or ""),
                "value": float(opportunity.get("value") or 0.7), "urgency": 0.35,
                "source_event_ids": [opportunity_id], "material": dict(opportunity),
            })

        calm_scene = str(getattr(stream, "current_activity", "") or "").casefold() not in {
            "combat", "boss", "cinematic", "dialogue",
        }
        current_game = str(getattr(stream, "current_game", "") or "").strip()
        if calm_scene and current_game and not (manager and manager.pending()):
            try:
                opinions = list(getattr(self, "hebe_self_model", None).current())
            except Exception:
                opinions = []
            game_key = current_game.casefold()
            for opinion in opinions:
                material_text = json.dumps(opinion.to_dict(), ensure_ascii=False, sort_keys=True).casefold()
                if game_key not in material_text or float(getattr(opinion, "confidence", 0.0) or 0.0) < 0.75:
                    continue
                add({
                    "type": "SELF_INITIATED_TOPIC", "topic": str(opinion.predicate),
                    "subject_ref": str(opinion.subject_ref), "value": float(opinion.confidence),
                    "urgency": 0.2, "source_event_ids": [str(opinion.id)],
                    "material": {
                        "belief_id": str(opinion.id), "predicate": str(opinion.predicate),
                        "value": opinion.object_value, "game": current_game,
                    },
                })
                break
        stream.speech_intent_candidates = candidates[-20:]

    def _poll_owner_discourse_opportunity(self, stream, *, now: float | None = None) -> bool:
        if stream is None:
            return False
        mode = str(getattr(stream, "discourse_participation_mode", getattr(self, "discourse_participation_mode", "shadow")) or "shadow").lower()
        buffer = getattr(self, "owner_discourse_buffer", None)
        session = getattr(buffer, "current_session", None) if buffer is not None else None
        topic = getattr(session, "topic", None)
        if mode == "disabled" or topic is None or not topic.stable:
            return False
        planner = getattr(self, "discourse_contribution_planner", None) or DiscourseContributionPlanner()
        plan = planner.plan(topic)
        stream.proposed_discourse_contribution = plan.to_dict()
        detector = getattr(self, "stream_turn_detector", None) or StreamTurnDetector()
        turn = detector.detect(
            now=now, audio_active=self._owner_audio_active(), tts_speaking=bool(getattr(self, "_tts_active", False)),
            topic_ready=plan.should_contribute,
            combat_intense=bool(getattr(stream, "combat_state", False) or getattr(stream, "current_activity", "") in {"combat", "boss"}),
        )
        stream.current_stream_turn = turn.to_dict()
        if not turn.turn_available:
            stream.last_discourse_blocked_reason = turn.reason
            print(f"[HEBE][DISCOURSE_CONTRIBUTION_WAIT] reason={turn.reason}", flush=True)
            return False
        if str(getattr(stream, "presence_mode", "reactive")) != "companion":
            stream.last_discourse_blocked_reason = "reactive_mode"
            return False
        budgeter = getattr(self, "discourse_participation_budget", None) or DiscourseParticipationBudget()
        budget = budgeter.allows(topic, now=now)
        perception = PerceivedEvent(
            event_id=f"discourse_opportunity_{topic.topic_id}_{int(now or time.time())}",
            timestamp=float(now or time.time()), source="owner_discourse_opportunity",
            source_type="owner_discourse_opportunity", speaker="Leo", speaker_type="owner",
            raw_text=" ".join(item.text for item in topic.fragments[-6:]),
            normalized_text=" ".join(item.normalized_text for item in topic.fragments[-6:]),
            output_context="stream", stream_live=bool(getattr(stream, "is_live", False)),
            current_game=str(getattr(stream, "current_game", "") or ""),
            current_activity=str(getattr(stream, "current_activity", "") or ""),
            is_owner_monologue=False,
            twitch_metadata={"turn_available": turn.turn_available, "topic_stable": topic.stable,
                             "contribution_value": plan.contribution_value, "novelty_score": plan.novelty_score,
                             "topic_id": topic.topic_id},
        )
        core = self._get_core_loop().process(
            perception=perception,
            understanding=UnderstandingResult(intent="owner_discourse_opportunity", confidence=plan.confidence,
                                               authority="owner", reply_pressure=plan.contribution_value,
                                               social_context=topic.family),
            policy=PolicyContract(result="allow", reason="validated_owner_discourse",
                                  allowed_action="stream_discourse_contribution",
                                  forbidden_actions=["open_pending", "ask_followup", "twitch_text_by_default"]),
            budget_result=budget,
            thread_result={"topic_id": topic.topic_id, "one_contribution_per_topic": True},
        )
        intervention = dict(core.get("intervention") or {})
        if not intervention.get("should_intervene"):
            stream.last_discourse_blocked_reason = str(intervention.get("reason") or "presence_engine")
            return False
        guarder = getattr(self, "discourse_grounding_guard", None) or DiscourseGroundingGuard()
        guard = guarder.evaluate(plan, topic)
        if not guard.get("passed"):
            stream.last_discourse_blocked_reason = "grounding_guard"
            return False
        if mode == "shadow":
            stream.last_discourse_blocked_reason = "shadow_mode"
            print("[HEBE][DISCOURSE_CONTRIBUTION_WAIT] reason=shadow_mode", flush=True)
            return False
        candidate = self._render_discourse_contribution(topic, plan)
        guard = guarder.evaluate(plan, topic, candidate=candidate)
        if not candidate or not guard.get("passed"):
            stream.last_discourse_blocked_reason = "render_or_grounding_failed"
            return False
        final_turn = detector.detect(now=time.time(), audio_active=self._owner_audio_active(), topic_ready=True,
                                     tts_speaking=bool(getattr(self, "_tts_active", False)))
        if not final_turn.turn_available:
            stream.last_discourse_blocked_reason = "owner_resumed_speaking"
            print("[HEBE][DISCOURSE_CONTRIBUTION_CANCELLED] reason=owner_resumed_speaking", flush=True)
            return False
        audio = self._effective_stream_audio_state("owner_discourse_opportunity")
        if not audio.actual_can_speak:
            stream.last_discourse_blocked_reason = audio.blocked_reason
            return False
        result = self._emit_final_response(
            event_id=perception.event_id, source="owner_discourse_opportunity", final_response=candidate,
            output_route=OutputRoute.STREAM_TTS_REPLY,
            output_targets=[OUTPUT_TARGET_LOCAL_UI, OUTPUT_TARGET_STREAM_TTS], guard_result=guard,
            debug_payload={"speech_act": "stream_discourse_contribution", "topic_id": topic.topic_id, "open_pending": False},
            speak_fn=lambda text: self.runtime.speak(text, emit_chat=False),
        )
        if not result.get("emitted"):
            stream.last_discourse_blocked_reason = str(result.get("reason") or "final_emission_gate")
            return False
        budgeter.record(topic, contribution_type=plan.contribution_type, thesis_key="|".join(topic.topic_keywords[:6]), now=now)
        stream.last_discourse_contribution = {"topic_id": topic.topic_id, "contribution_type": plan.contribution_type,
                                              "timestamp": float(now or time.time()), "text": candidate}
        stream.discourse_contribution_timestamps = [item["timestamp"] for item in budgeter.contributions]
        stream.last_discourse_blocked_reason = ""
        return True

    def _owner_audio_active(self) -> bool:
        stt = getattr(self.runtime, "stt", None)
        rms = float(getattr(stt, "last_input_rms", 0.0) or 0.0)
        cfg = getattr(stt, "cfg", None)
        threshold = float(getattr(cfg, "silence_rms_threshold", 0.003) or 0.003)
        return bool(rms > max(0.001, threshold * 1.25))

    def _render_discourse_contribution(self, topic, plan) -> str:
        model = getattr(self.runtime, "llm", None)
        fragments = "\n".join(f"- {item.text}" for item in topic.fragments[-6:])
        prompt = (
            "Escribe una intervención de Hebe como copresentadora del stream. Una sola idea, una o dos frases "
            "cortas, en español, sin pregunta, sin repetir literalmente a Leo y sin afirmar noticias actuales. "
            f"Tipo: {plan.contribution_type}. Postura de Leo: {topic.owner_stance}. "
            f"Objetivo semántico: {'; '.join(plan.proposed_claims)}. Fragmentos:\n{fragments}"
        )
        try:
            text = str(model.chat([
                {"role": "system", "content": "Eres Hebe, copresentadora breve, cálida y con criterio. No haces de asistente ni entrevistas al streamer."},
                {"role": "user", "content": prompt},
            ], temperature=0.45, num_predict=80) or "").strip() if model is not None and hasattr(model, "chat") else ""
        except Exception as exc:
            print(f"[HEBE][DISCOURSE_RENDER] failed error={exc!r}", flush=True)
            text = ""
        text = re.sub(r"\s+", " ", text).strip().strip('"')
        if "?" in text or "¿" in text:
            return ""
        sentences = [item.strip() for item in re.split(r"(?<=[.!])\s+", text) if item.strip()]
        return " ".join(sentences[:2]).strip()

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

        self._poll_stream_proactive_routine(stream, now=now, today=today)
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

    def _poll_stream_proactive_routine(self, stream, *, now: datetime, today: str) -> bool:
        delay = int(getattr(stream, "stream_delay_minutes", 0) or 0)
        windows = {
            "18:30": "scheduled_reminder",
            "18:50": "actionable_routine",
            "19:00": "actionable_routine",
        }
        handled = False
        for base_time, proactive_type in windows.items():
            due = self._today_at(base_time) + timedelta(minutes=delay)
            key = f"{today}:{base_time}:{delay}"
            sent = getattr(stream, "routine_sent_keys", set())
            if key in sent:
                continue
            if 0 <= (now - due).total_seconds() < self.routine_poll_interval_sec + 5:
                handled = True
                sent.add(key)
                stream.routine_sent_keys = sent
                if proactive_type == "scheduled_reminder":
                    schedule_slot = session_primer.get_schedule_for_date(now) or {}
                    decision = scheduled_reminder_decision(
                        trigger=f"routine:{base_time}",
                        schedule_slot=schedule_slot,
                        current_game=str(getattr(stream, "current_game", "") or schedule_slot.get("game") or ""),
                    )
                    stream.last_proactive_decision = decision.to_dict()
                    if decision.should_speak:
                        self._deliver_voice_reply(self._build_scheduled_stream_reminder(decision))
                    continue
                decision = self._evaluate_stream_preparation_decision(stream, trigger=f"routine:{base_time}", now_dt=now)
                stream.last_proactive_decision = decision.to_dict()
                if not decision.should_speak:
                    print(f"[HEBE][SPONTANEITY] skipped reason={decision.blocked_reason or 'stream_prep_not_needed'}", flush=True)
                    continue
                reply = self._build_stream_preparation_reply(decision)
                if reply:
                    self._deliver_voice_reply(reply)
        return handled

    def _evaluate_stream_preparation_decision(self, stream, *, trigger: str, now_dt: datetime):
        schedule_slot = session_primer.get_schedule_for_date(now_dt) or {}
        obs_running = bool(self._is_process_running_safe("obs64.exe") or self._is_process_running_safe("obs.exe"))
        twitch = getattr(self.runtime, "twitch", None)
        chat_bot = getattr(self.runtime, "twitch_chat_bot", None)
        game_run = GameRunState.from_value(getattr(self.runtime.state, "game_run_state", None))
        # Current observed Twitch state is stronger evidence than a schedule prediction.
        expected_game = str(getattr(stream, "current_game", "") or schedule_slot.get("game") or "")
        game_run_ready = bool(game_run.game and (not expected_game or self._normalize_text(game_run.game) == self._normalize_text(expected_game)))
        return self.stream_preparation.evaluate(
            stream=stream,
            schedule_slot=schedule_slot,
            obs_running=obs_running,
            expected_game_running=None,
            twitch_connected=bool(twitch is not None and getattr(twitch, "is_available", lambda: False)()),
            chat_connected=bool(chat_bot is not None and getattr(chat_bot, "enabled", False)),
            stt_listening=bool(getattr(self.runtime, "stt_enabled", False)),
            tts_ready=bool(getattr(self.runtime.state, "tts_enabled", False)),
            vtube_connected=None,
            title_category_known=bool(schedule_slot.get("category") or getattr(stream, "current_category", None)),
            game_run_state_ready=game_run_ready,
            trigger=trigger,
        )

    def _build_scheduled_stream_reminder(self, decision) -> str:
        game = decision.current_game or "directo por confirmar"
        slot = (decision.schedule_slot or {}).get("slot_name") or "stream"
        if (decision.schedule_slot or {}).get("schedule_uncertain"):
            return f"Recordatorio tentativo de stream: {slot}, {game}. El patron horario esta en revision y lo confirmare con el directo real."
        return f"Recordatorio de stream: {slot}, {game}. Es solo aviso horario; para preparar cosas uso la rutina de estado."

    def _build_stream_preparation_reply(self, decision) -> str:
        state = decision.stream_state or {}
        game = decision.current_game or "juego por confirmar"
        checks: list[str] = [f"juego previsto: {game}"]
        checks.append(f"modo stream: {'activo' if state.get('stream_mode') else 'apagado'}")
        obs = state.get("obs_running")
        checks.append("OBS: abierto" if obs is True else "OBS: cerrado" if obs is False else "OBS: sin dato")
        if state.get("twitch_connected") is False:
            checks.append("Twitch: revisar conexion")
        if state.get("chat_connected") is False:
            checks.append("chat: no confirmado")
        if state.get("stt_listening") is False:
            checks.append("STT: apagado")
        if state.get("tts_ready") is False:
            checks.append("TTS: apagado")
        if state.get("game_run_state_ready") is False:
            checks.append("GameRunState: pendiente")
        actions = [item for item in str(decision.suggested_action or "").split(",") if item]
        if actions:
            checks.append("puedo hacer: " + ", ".join(actions))
        return "Preparacion de stream: " + "; ".join(checks) + "."

    def _is_process_running_safe(self, process_name: str) -> bool:
        try:
            from app.tools.windows_apps import is_process_running

            return bool(is_process_running(process_name))
        except Exception:
            return False

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
                    self._recover_stream_memory_session_on_startup()
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
        executor = getattr(self, "plan_executor", None)
        if executor is not None and hasattr(executor, "begin_shutdown"):
            executor.begin_shutdown(drain_seconds=float(os.getenv("HEBE_COMMAND_SHUTDOWN_DRAIN_SECONDS", "2") or 2))
        game_research = getattr(self,"game_intelligence",None)
        if game_research is not None and hasattr(game_research,"close"):
            game_research.close(wait=False)
        stream = self._get_stream_state()
        try:
            active = stream_memory.get_active_stream_session()
            if active:
                existing_reason = str(active.get("closure_reason") or "")
                reason = existing_reason or (
                    "normal_shutdown" if bool(getattr(stream, "live_status_known", False)) else "interrupted_shutdown"
                )
                stream_memory.mark_stream_session_ending(stream, reason=reason, source_signal="engine_stop")
                summary = stream_memory.finalize_stream_session(stream, reason=reason, source_signal="engine_stop")
                self._persist_canonical_chatter_summaries(summary)
        except Exception as exc:
            print(
                f"[HEBE][STREAM_LIFECYCLE] session_finalize_failed reason={type(exc).__name__} source_signal=engine_stop",
                flush=True,
            )
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
            for name in self.social_world.recent_identity_names(limit=80) if getattr(self,"social_world",None) is not None else []:
                add(name)
        except Exception:
            pass

        return values[-120:]

    def _normalize_stt_input(self, raw_text: str, *, debug_metadata: dict | None = None) -> TranscriptNormalizationResult:
        result = normalize_stt_transcript(
            raw_text,
            known_targets=self._known_voice_command_targets(),
            detected_language=(debug_metadata or {}).get("detected_language"),
        )
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
        log_jsonl_event("stt", {
            "raw_text": result.raw_text,
            "normalized_text": result.normalized_text,
            "normalized_candidates": result.normalized_candidates,
            "alternative_candidates": result.alternative_candidates,
            "candidate_scores": result.candidate_scores,
            "detected_language": result.detected_language,
            "confidence": result.confidence,
            "status": "normalized",
            "selected_input_device": (debug_metadata or {}).get("selected_input_device"),
            "command_mode": (debug_metadata or {}).get("command_mode"),
        })
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
        log_jsonl_event("stt", {
            **event,
            "rejected": True,
            "passed": False,
            "rejection_reason": reason,
            "voice_type": (details or {}).get("voice_type", ""),
        })
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
        pending = self._active_current_conversation(latest=True)
        if pending is not None and pending.topic == "promotion_target_clarification":
            capture_seconds = float(pending.domain_payload.get("capture_window_seconds") or 12.0)
            self._update_current_conversation(
                pending,
                domain_updates={
                    "starts_after_tts_end": until,
                    "actual_tts_completion_time": float(pending.domain_payload.get("actual_tts_completion_time") or 0.0),
                },
                expires_at=until + capture_seconds,
            )
            print(
                f"[HEBE][PROMOTION_PENDING] capture_starts={until:.3f} window_seconds={capture_seconds:.1f} source=tts_completion_estimate",
                flush=True,
            )
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
        stream = self._get_stream_state()
        if bool(stream and getattr(stream, "enabled", False) and getattr(stream, "is_live", False)) and expected_type not in {"clarification", "action_confirmation"}:
            print(f"[HEBE][PENDING_CREATION_GUARD] allowed=false reason=weak_live_stream_reply expected_reply_type={expected_type}", flush=True)
            return
        print(f"[HEBE][PENDING_CREATION_GUARD] allowed=true reason=explicit_followup expected_reply_type={expected_type}", flush=True)
        configured_ttl = float(getattr(self, "pending_conversation_ttl_seconds", 45) or 45)
        has_configured_override = abs(configured_ttl - 45.0) > 0.001
        ttl_by_type = {
            "casual_answer": float(os.getenv("HEBE_PENDING_CASUAL_TTL_SECONDS", str(configured_ttl if has_configured_override else 40)) or 40),
            "clarification": float(os.getenv("HEBE_PENDING_CLARIFICATION_TTL_SECONDS", str(configured_ttl if has_configured_override else 55)) or 55),
            "action_confirmation": float(os.getenv("HEBE_PENDING_ACTION_CONFIRMATION_TTL_SECONDS", str(configured_ttl if has_configured_override else 60)) or 60),
        }
        ttl = ttl_by_type.get(expected_type, configured_ttl)
        self._open_pending_conversation(
            kind="assistant_followup",
            expected_reply_type=expected_type,
            opened_by_speech_act="assistant_question",
            can_accept_no_wake_followup=True,
            ttl_seconds=ttl,
            previous_assistant_message_id=f"assistant-{int(time.time() * 1000)}",
            previous_assistant_message=str(text or "").strip(),
            source="assistant_question",
            max_attempts=int(getattr(self, "pending_conversation_max_followups", 1) or 1),
            reply_source=source,
        )
        print(
            "[HEBE][CONVERSATION] pending_turn_created reason=direct_question source=local "
            f"expected_type={expected_type} ttl={int(ttl)}s",
            flush=True,
        )

    def _pending_conversation_matches(self, *, source: str, text: str | None = None, event_type: str | None = None) -> bool:
        conversation = self._active_current_conversation(source=source)
        if conversation is None or conversation.topic != "assistant_followup":
            return False
        expected = conversation.expected_reply
        canonical_source = "owner_stt" if source == "stt_voice" else "owner_ui"
        if expected is None or canonical_source not in set(expected.allowed_sources):
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
            expected_type = expected.type.value
            if expected_type != "casual_answer" and event_type in ambient_types:
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
        if stt is None or not hasattr(stt, "retry_last_language_recovery"):
            return {"attempted": False, "text": "", "speech_detected": False, "reason": "retry_unavailable"}
        speech_detected = bool(getattr(stt, "last_speech_detected", False))
        if not speech_detected:
            return {"attempted": False, "text": "", "speech_detected": False, "reason": "no_speech_detected"}
        print("[HEBE][STT][RETRY] reason=unsupported_script policy=dual_decode_then_drop", flush=True)
        try:
            retry = stt.retry_last_language_recovery(initial_language=script)
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
                "recovery_policy": "dual_decode_then_drop",
            }
        accepted = bool(
            (retry or {}).get("accepted")
            and retry_text
            and not self._unsupported_stt_script(retry_text)
            and str((retry or {}).get("selected_language") or "") in {"es", "en"}
        )
        print(f"[HEBE][STT][RETRY_RESULT] raw={ascii(retry_text)} accepted={str(accepted).lower()}", flush=True)
        return {
            **(retry or {}),
            "attempted": bool((retry or {}).get("attempted", True)),
            "text": retry_text,
            "accepted": accepted,
            "original_script": script,
            "recovery_policy": "dual_decode_then_drop",
        }

    def _process_stt_voice_transcript(
        self, raw_voice_command: str, *, allow_wakeword_prompt: bool = False,
        stt_metadata: dict | None = None, force_ambient: bool = False,
    ) -> str:
        original_raw_text = str(raw_voice_command)
        transcript_for_cognition = original_raw_text
        stt_metadata = dict(stt_metadata or {})
        allowed_languages = {
            item.strip().lower()
            for item in os.getenv("HEBE_STT_ALLOWED_LANGUAGES", "es,en").split(",")
            if item.strip()
        } or {"es", "en"}
        detected_language = str(stt_metadata.get("detected_language") or "").lower()
        language_recovery = dict(stt_metadata.get("language_recovery") or {})
        if (
            detected_language
            and detected_language not in allowed_languages
            and not (
                language_recovery.get("accepted")
                and str(language_recovery.get("selected_language") or "") in allowed_languages
            )
        ):
            print(
                "[HEBE][STT_REJECTED] "
                "reason=unsupported_language_recovery_failed "
                f"initial_language={detected_language}",
                flush=True,
            )
            log_jsonl_event("stt", {
                "raw_text": "",
                "status": "rejected",
                "passed": False,
                "rejected": True,
                "rejection_reason": "unsupported_language_recovery_failed",
                "initial_language": detected_language,
            })
            return "continue"
        retry_debug: dict = {
            "detected_script": self._unsupported_stt_script(original_raw_text) or "latin",
            "retry_attempted": False,
            "retry_transcript": "",
            "status": "accepted",
            "final_decision": "accepted",
            **stt_metadata,
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
            duplicate, similarity = (False, 0.0) if force_ambient else self._is_duplicate_recent_stt(original_raw_text)
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
                    "recovery_policy": retry.get("recovery_policy") or "dual_decode_then_drop",
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
                duplicate, similarity = (False, 0.0) if force_ambient else self._is_duplicate_recent_stt(transcript_for_cognition)
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
        hypothesis = dict(stt_metadata.get("command_hypothesis") or {})
        direct_stt = DirectSTTCommandResult.from_dict(stt_metadata.get("direct_stt_command"))
        if not direct_stt.command_text:
            direct_stt = parse_direct_stt_command(
                transcript_for_cognition,
                ambient_text=original_raw_text,
                agreement_score=float(hypothesis.get("hypothesis_agreement") or 0.0),
            )
        stt_metadata["direct_stt_command"] = direct_stt.to_dict()
        retry_debug["direct_stt_command"] = direct_stt.to_dict()
        exact_wake = bool(
            direct_stt.wake_detected
            or re.match(r"^\s*(?:hebe|ebe|eve|jebe|heve|e\s+[bv])\b", command)
        )
        role_classifier = getattr(self, "utterance_role_classifier", None)
        if role_classifier is None:
            role_classifier = UtteranceRoleClassifier()
            self.utterance_role_classifier = role_classifier
        role_decision = role_classifier.classify(
            raw_transcript=original_raw_text,
            detected_language=stt_metadata.get("detected_language"),
            wake_detected=bool(hypothesis.get("wake_detected") or exact_wake),
            wake_confidence=float(hypothesis.get("wake_score") or (1.0 if exact_wake else 0.0)),
            command_structure=direct_stt.detected_intent_family in {
                DirectUtteranceIntentFamily.APPLICATION_ACTION.value,
                DirectUtteranceIntentFamily.STREAM_OPERATION.value,
                DirectUtteranceIntentFamily.SYSTEM_COMMAND.value,
                DirectUtteranceIntentFamily.INCOMPLETE_COMMAND.value,
            },
            current_game_language=str(getattr(self._get_stream_state(), "game_language", "") or ""),
            audio_metadata=stt_metadata.get("audio_metadata") or {},
        )
        retry_debug["utterance_role"] = role_decision.role.value
        retry_debug["utterance_role_decision"] = role_decision.to_dict()
        self._active_current_conversation(source="stt_voice")
        self._current_input_event = self._build_input_event(
            source="stt_voice",
            raw_text=original_raw_text,
            normalized_text=command,
            stt_metadata={**normalization.as_event(), **retry_debug, "accepted_transcript": transcript_for_cognition},
        )
        if force_ambient:
            self._current_input_event.source = "ambient_stt"
            self._current_input_event.interpretation = None
            self._get_input_interpreter().interpret_event(
                self._current_input_event,
                authority="ambient",
                addressed_to_hebe=False,
                explicit_command_mode=False,
                direct_result=direct_stt,
            )
        interpretation = self._current_input_event.interpretation
        mute_mode = self._owner_mute_command_mode(command)
        if mute_mode and interpretation and interpretation.authorized_action_command and self._is_stream_enabled():
            self._apply_owner_mute_command(
                mute_mode,
                ttl=300.0,
                reason="owner_mute",
                activated_by_text=original_raw_text,
                manual=True,
            )
            self._log_stt_non_command_decision(command, "owner_mute_command", reason=mute_mode)
            self._log_direct_stt_outcome(
                direct_stt, outcome="action_executed", reason=f"owner_mute_{mute_mode}",
                action_receipt={"action_type": "owner_mute", "target": mute_mode, "executor_invoked": True, "success": True, "timestamp": time.time()},
            )
            self._current_input_event = None
            return "continue"
        if (
            self._owner_unmute_command(command)
            and interpretation
            and interpretation.authorized_action_command
            and self._is_stream_enabled()
        ):
            self._clear_owner_mute_command(reason="owner_unmute")
            self._log_stt_non_command_decision(command, "owner_unmute_command", reason="normal")
            self._log_direct_stt_outcome(
                direct_stt, outcome="action_executed", reason="owner_unmute",
                action_receipt={"action_type": "owner_unmute", "target": "stream", "executor_invoked": True, "success": True, "timestamp": time.time()},
            )
            self._current_input_event = None
            return "continue"
        voice_type, mood_hint = self._classify_voice_event(command, interpretation=interpretation)
        has_action_intent = self._input_event_has_action_intent(getattr(self, "_current_input_event", None))
        media_detected, _media_reason = looks_like_media_or_singing(command)
        try:
            possible_reply_to_hebe = self._get_live_session_brain().is_possible_reply_to_hebe(command)
        except Exception:
            possible_reply_to_hebe = False
        pending_match = self._pending_conversation_matches(source="stt_voice", text=command, event_type=voice_type)
        if force_ambient:
            # Explicit ambient sources (Replay/dev simulation) carry source
            # identity that live microphone audio derives through classification.
            # They must never borrow an owner continuation window.
            possible_reply_to_hebe = False
            pending_match = False
        continuation_event_id = str(
            stt_metadata.get("continuation_event_id")
            or stt_metadata.get("replay_event_id")
            or getattr(self._current_input_event, "event_id", "")
            or f"stt_{uuid.uuid4().hex}"
        )
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
        deferred_domain_topics = {
            "appointment_datetime", "promotion_target_clarification",
            "game_guidance_clarification", "tts_scope",
        }
        continuation = self._resolve_conversation_continuation(
            text=command,
            event_id=continuation_event_id,
            wake=exact_wake,
            force_ambient=force_ambient,
            compatibility=True if (envelope.pending_compatible or conversation_followup) else False,
            compatibility_reason=envelope.reason,
            consume=not (
                envelope.pending_compatible and envelope.active_conversation is not None
                and envelope.active_conversation.topic in deferred_domain_topics
            ),
        )
        self._apply_game_run_correction_continuation(continuation,event_id=continuation_event_id,text=command)
        if continuation is not None and self._current_input_event is not None:
            self._current_input_event.stt_metadata["continuation"] = continuation.to_dict()
        if continuation is not None:
            if continuation.decision == "interrupt":
                pending_match = False
                conversation_followup = False
            elif continuation.consumed or envelope.pending_compatible:
                pending_match = True
                possible_reply_to_hebe = True
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
            conversation_state = self._get_conversation_state_resolver().from_conversation(
                self._active_current_conversation(source="stt_voice"),
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
        pending_turn_for_frame = self._active_current_conversation(source="stt_voice")
        if envelope.pending_compatible:
            conversation_state = ConversationState(
                active=True,
                topic=envelope.active_conversation.topic if envelope.active_conversation else "pending_task",
                source="cognitive_pending_task",
                expected_reply_type=envelope.expected_reply_type,
                allow_no_wakeword=True,
                output_target=[OUTPUT_TARGET_LOCAL_UI, self._direct_voice_tts_target()],
                confidence=0.95,
                matched=True,
                reason="pending_compatible_input_envelope",
            )
        else:
            conversation_state = self._get_conversation_state_resolver().from_conversation(
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
            log_jsonl_event("stt", {
                "raw_text": original_raw_text,
                "normalized_text": command,
                "status": "passed",
                "passed": True,
                "rejected": False,
                "rejection_reason": "",
                "voice_type": voice_type,
                "source": envelope.source,
                "authority": envelope.authority,
                "addressed_to_hebe": envelope.addressed_to_hebe,
                "matched_wake_name": envelope.matched_wake_name,
                "command_mode": envelope.command_mode,
                "reason": gate_reason,
            })
        stream_enabled = self._is_stream_enabled()
        if pending_followup:
            self._current_input_event.stt_metadata["message_type"] = (
                "pending_reply" if envelope.pending_compatible else "conversation_followup"
            )
            self._current_input_event.stt_metadata["conversation_followup"] = not envelope.pending_compatible
            self._current_input_event.stt_metadata["jarvis_allowed"] = True
            if envelope.pending_compatible:
                pending_kind = envelope.active_conversation.topic if envelope.active_conversation else "pending"
                print(f"[HEBE][COG] decision=pending_followup kind={pending_kind}", flush=True)
            else:
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
        elif not pending_followup and interpretation.authorized_action_command:
            handled, stream_command = self._extract_stream_command(command)
            if handled:
                if not stream_command:
                    self._log_direct_stt_outcome(
                        direct_stt, outcome="context_only", reason="stream_operation_handled",
                    )
                    self._current_input_event = None
                    return "continue"
                command = stream_command
        elif envelope.addressed_to_hebe and self._is_stream_enabled():
            command = str(envelope.wake_evidence.get("stripped_text") or command).strip()
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
        family = direct_stt.detected_intent_family
        if is_direct_command and family == DirectUtteranceIntentFamily.INCOMPLETE_COMMAND.value:
            clarification = (
                "¿Qué aplicación quieres que abra?"
                if direct_stt.action_verb
                else "Te escucho. ¿Qué necesitas?"
            )
            self._deliver_manual_reply(clarification, source="stt_voice")
            self._log_direct_stt_outcome(
                direct_stt,
                outcome="clarification",
                reason="missing_application_target" if direct_stt.action_verb else "incomplete_direct_utterance",
            )
            self._current_input_event = None
            return "continue"
        try:
            res = self.handle_command(command, source="stt_voice")
        except Exception as exc:
            self._deliver_manual_reply(
                "He entendido que me hablabas, pero el comando ha fallado. Repítelo una vez.",
                source="stt_voice",
            )
            self._log_direct_stt_outcome(
                direct_stt,
                outcome="action_failed",
                reason=f"{type(exc).__name__}:command_pipeline_failure",
            )
            self._current_input_event = None
            return "continue"
        committed_terminal = self._current_direct_stt_terminal_outcome()
        if committed_terminal is not None:
            print(
                "[HEBE][DIRECT_STT_TERMINAL_SHORT_CIRCUIT] "
                f"outcome={committed_terminal['outcome']} reason={committed_terminal['reason']} "
                "parser_fallback_skipped=true",
                flush=True,
            )
            self._current_input_event = None
            return res
        execution = dict((getattr(self, "_current_input_event", None).stt_metadata or {}).get("direct_stt_execution") or {}) if self._current_input_event else {}
        if family in {
            DirectUtteranceIntentFamily.DIRECT_QUESTION.value,
            DirectUtteranceIntentFamily.CASUAL_CONVERSATION.value,
        }:
            outcome, reason = "conversational_reply", "routed_to_owner_conversation"
        elif family == DirectUtteranceIntentFamily.APPLICATION_ACTION.value:
            if execution.get("success"):
                outcome, reason = "action_executed", "application_launch_succeeded"
            elif execution:
                outcome, reason = "action_failed", str(execution.get("reason") or "application_launch_failed")
            else:
                outcome, reason = "rejected", "application_parser_or_resolver_failed"
        else:
            outcome, reason = "context_only", "routed_to_command_pipeline"
        self._log_direct_stt_outcome(direct_stt, outcome=outcome, reason=reason)
        print("[HEBE][COG] decision=conversation_followup" if pending_followup else "[HEBE][COG] decision=command", flush=True)
        self._current_input_event = None
        return res

    def _log_direct_stt_outcome(
        self,
        result: DirectSTTCommandResult,
        *,
        outcome: str,
        reason: str,
        action_receipt: dict | None = None,
    ) -> bool:
        aliases = {
            "action_clarification": "clarification",
            "intentionally_rejected": "rejected",
            "error": "action_failed",
        }
        outcome = aliases.get(outcome, outcome)
        if action_receipt is not None:
            result.action_receipt = dict(action_receipt)
        receipt = dict(result.action_receipt or {})
        if outcome == "action_executed" and not (
            receipt.get("executor_invoked") is True and receipt.get("success") is True
        ):
            outcome = "action_failed"
            reason = "missing_success_action_receipt"
        terminals = {
            "action_executed", "action_failed", "clarification",
            "conversational_reply", "context_only", "rejected", "cancelled",
        }
        if outcome not in terminals:
            outcome = "rejected"
        store = getattr(self, "_direct_stt_terminal_outcomes", None)
        if store is None:
            store = {}
            self._direct_stt_terminal_outcomes = store
        existing = store.get(result.event_id)
        if existing:
            result.final_outcome = str(existing["outcome"])
            result.rejection_reason = (
                str(existing["reason"])
                if result.final_outcome in {"rejected", "action_failed"} else ""
            )
            print(
                "[HEBE][DIRECT_STT_DUPLICATE_TERMINAL] "
                f"event_id={result.event_id} ignored=true existing_outcome={existing['outcome']}",
                flush=True,
            )
            return False
        store[result.event_id] = {"outcome": outcome, "reason": reason}
        result.final_outcome = outcome
        result.rejection_reason = reason if outcome in {"rejected", "action_failed"} else ""
        payload = result.to_dict()
        payload.update({"outcome": outcome, "reason": reason})
        print(
            "[HEBE][DIRECT_STT_OUTCOME] "
            f"event_id={result.event_id} intent_family={result.detected_intent_family} "
            f"outcome={outcome} reason={reason} terminal=true",
            flush=True,
        )
        emit("voice.command", {**payload, "status": "outcome", "final_decision": outcome})
        ledger=getattr(self,"historical_action_ledger",None)
        if ledger is not None and bool(getattr(self,"historical_action_ledger_v2",False)) and result.detected_intent_family:
            status="SUCCEEDED" if outcome=="action_executed" else "FAILED" if outcome=="action_failed" else "REQUESTED" if outcome=="clarification" else "UNKNOWN"
            try:ledger.project(source_store="direct_stt_terminal",source_record_id=result.event_id,action_type=str(receipt.get("action_type") or result.detected_intent_family),target=str(receipt.get("target") or ""),status=status,evidence={"outcome":outcome,"reason":reason,"external_confirmation":receipt.get("external_confirmation")})
            except Exception as exc:print(f"[HEBE][ACTION_LEDGER_PROJECT] status=failed reason={type(exc).__name__}",flush=True)
        return True

    def _commit_current_direct_stt_terminal(self, *, outcome: str, reason: str, action_receipt: dict | None = None) -> bool:
        event = getattr(self, "_current_input_event", None)
        raw = dict((getattr(event, "stt_metadata", {}) or {}).get("direct_stt_command") or {})
        if not raw:
            return False
        committed = self._log_direct_stt_outcome(
            DirectSTTCommandResult.from_dict(raw), outcome=outcome, reason=reason, action_receipt=action_receipt,
        )
        if committed and isinstance(getattr(event, "stt_metadata", None), dict):
            event.stt_metadata["direct_stt_terminal_committed"] = {
                "outcome": outcome,
                "reason": reason,
            }
        return committed

    def _current_direct_stt_terminal_outcome(self) -> dict | None:
        event = getattr(self, "_current_input_event", None)
        metadata = getattr(event, "stt_metadata", None)
        if not isinstance(metadata, dict):
            return None
        committed = metadata.get("direct_stt_terminal_committed")
        if isinstance(committed, dict):
            return dict(committed)
        direct = DirectSTTCommandResult.from_dict(metadata.get("direct_stt_command"))
        store = getattr(self, "_direct_stt_terminal_outcomes", {}) or {}
        existing = store.get(direct.event_id)
        return dict(existing) if isinstance(existing, dict) else None

    def _build_input_event(
        self,
        *,
        source: str,
        raw_text: str,
        normalized_text: str,
        stt_metadata: dict | None = None,
    ) -> InputEvent:
        semantic_metadata = dict(stt_metadata or {})
        semantic_metadata.setdefault("interaction_trace_id", f"input_{uuid.uuid4().hex}")
        if re.search(r"(?i)\b(?:deja|deje)\s+de\b|@|,", str(raw_text or "")):
            social_world = getattr(self, "social_world", None)
            if social_world is not None and hasattr(social_world, "recent_identity_names"):
                try:
                    semantic_metadata["semantic_social_identities"] = list(
                        social_world.recent_identity_names(limit=80)
                    )
                except Exception as exc:
                    print(
                        f"[HEBE][SEMANTIC_SCOPE] social_identity_lookup_failed={type(exc).__name__}",
                        flush=True,
                    )
        event = InputEvent(
            source=source,
            raw_text=str(raw_text or ""),
            normalized_text=str(normalized_text or ""),
            is_voice=source == "stt_voice",
            is_stream_context=self._is_stream_enabled(),
            stt_metadata=semantic_metadata,
        )
        direct = DirectSTTCommandResult.from_dict(event.stt_metadata.get("direct_stt_command"))
        if not direct.command_text:
            direct = parse_direct_stt_command(event.raw_text)
        source_authority = InputInterpreter.authority_for_source(source)
        addressed = bool(
            direct.wake_detected
            or source in {"ui", "typed_ui", "owner_ui", "button"}
        )
        explicit_command_mode = bool(
            event.stt_metadata.get(
                "command_mode",
                source in {"ui", "typed_ui", "owner_ui", "button"},
            )
        )
        self._get_input_interpreter().interpret_event(
            event,
            authority=source_authority,
            addressed_to_hebe=addressed,
            explicit_command_mode=explicit_command_mode,
            recent_hebe_utterance=str(getattr(self, "_last_assistant_text", "") or ""),
            direct_result=direct,
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

    def _get_input_interpreter(self) -> InputInterpreter:
        interpreter = getattr(self, "input_interpreter", None)
        if interpreter is None:
            interpreter = InputInterpreter()
            self.input_interpreter = interpreter
        return interpreter

    def _build_stt_input_envelope(
        self, event: InputEvent, *, voice_type: str, conversation_followup: bool,
    ) -> InputEnvelope:
        normalized = self._normalize_text(event.normalized_text)
        interpretation = event.interpretation or self._get_input_interpreter().interpret_event(event)
        resolver = getattr(self, "wake_name_resolver", None) or WakeNameResolver()
        self.wake_name_resolver = resolver
        wake = resolver.resolve(
            raw_text=event.raw_text,
            normalized_text=normalized,
            source="stt_voice",
            is_sleeping=bool(getattr(self.runtime.state, "hebe_sleeping", False)),
            command_markers=set(self._get_local_app_planner().command_markers()),
            detected_language=(event.stt_metadata or {}).get("detected_language"),
            alternative_candidates=(event.stt_metadata or {}).get("alternative_candidates") or [],
            command_redecode_supports_wake=bool(
                ((event.stt_metadata or {}).get("command_hypothesis") or {}).get("wake_detected")
            ),
            acoustic_wake_score=float(
                ((event.stt_metadata or {}).get("command_hypothesis") or {}).get("wake_score") or 0.0
            ),
            owner_trusted=True,
        )
        # A trusted command marker is command evidence, not wake-name evidence.
        # WakeNameResolver intentionally accepts that context, but the unified
        # envelope keeps the two facts separate so no-wake commands are routed
        # as owner_stt_command rather than pretending a name was spoken.
        addressed = bool(
            wake.matched_name
            or voice_type == "direct_command_to_hebe"
            or interpretation.addressed_to_hebe
        )
        command_mode = bool(interpretation.authorized_action_command)
        utterance_role = str((event.stt_metadata or {}).get("utterance_role") or "")
        isolated_dialogue = utterance_role in {
            UtteranceRole.QUOTED_OR_READ_DIALOGUE.value,
            UtteranceRole.GAME_AUDIO_BLEED.value,
        }
        if isolated_dialogue:
            addressed = False
            command_mode = False

        direct_family = str(
            ((event.stt_metadata or {}).get("direct_stt_command") or {}).get("detected_intent_family")
            or ""
        )
        should_resolve_app = (
            not direct_family
            or direct_family in {
                DirectUtteranceIntentFamily.APPLICATION_ACTION.value,
                DirectUtteranceIntentFamily.INCOMPLETE_COMMAND.value,
            }
        )
        local_plan = None if (
            isolated_dialogue
            or not should_resolve_app
            or not interpretation.authorized_action_command
        ) else self._get_local_app_planner().plan(
            event, is_awake=not bool(getattr(self.runtime.state, "hebe_sleeping", False)),
        )
        event._local_app_plan = local_plan
        app_plan_result: dict = {}
        intent_candidates: list[str] = []
        app_target = None
        if local_plan is not None and local_plan.action_type == "open_application":
            app_target = local_plan.target or (local_plan.slots or {}).get("application_target")
            intent_candidates.append("open_application")
            app_plan_result = {
                "status": local_plan.status,
                "confidence": float(local_plan.confidence or 0.0),
                "reason": local_plan.reason,
                "target": app_target,
            }

        active_pending = self._active_current_conversation(source="stt_voice")
        pending_kind = active_pending.topic if active_pending else ""
        expected_reply_type = (
            active_pending.expected_reply.type.value
            if active_pending and active_pending.expected_reply else ""
        )
        router = getattr(self, "cognitive_router", None) or CognitiveRouter()
        game_snapshot = {
            "game_run_state": GameRunState.from_value(
                getattr(self.runtime.state, "game_run_state", None)
            ).to_dict()
        }
        fresh_stream_plan = None
        if not isolated_dialogue and interpretation.authorized_action_command:
            try:
                fresh_stream_plan = self._get_stream_action_planner().plan(event)
            except Exception as exc:
                print(f"[HEBE][STREAM_ACTION_PLAN] preflight_failed error={type(exc).__name__}", flush=True)
        fresh_high_confidence_promotion = bool(
            fresh_stream_plan is not None
            and fresh_stream_plan.action_type == "twitch_shoutout"
            and fresh_stream_plan.status == "complete"
            and float(fresh_stream_plan.confidence or 0.0) >= 0.78
            and fresh_stream_plan.target
        )
        canonical_new_request = interpretation.speech_act != InputSpeechAct.OWNER_FEEDBACK
        stronger_request = bool(
            canonical_new_request
            and (
                router._is_current_time_query(normalized)
                or router._is_current_date_query(normalized)
                or (
                    interpretation.authorized_action_command
                    and router._is_open_app_command(normalized)
                )
                or (
                    interpretation.authorized_action_command
                    and router._is_reminder_request(normalized)
                )
                or router._personal_state(normalized)
                or router.game_guidance.looks_like_query(event.raw_text, game_snapshot)
                or fresh_high_confidence_promotion
            )
        )
        appointment_pending_compatible = bool(
            active_pending
            and interpretation.speech_act != InputSpeechAct.OWNER_FEEDBACK
            and pending_kind == "appointment_datetime"
            and self._appointment_pending_reply_compatible(normalized)
            and not stronger_request
        )
        promotion_pending_compatible = False
        promotion_pending_reason = "no_promotion_pending"
        promotion_resolution = {}
        if (
            active_pending
            and interpretation.speech_act != InputSpeechAct.OWNER_FEEDBACK
            and pending_kind == "promotion_target_clarification"
            and not stronger_request
        ):
            promotion_pending_compatible, promotion_pending_reason, promotion_resolution = self._promotion_pending_reply_compatible(
                event.raw_text,
                normalized,
                active_pending,
            )
            print(
                "[HEBE][PENDING_COMPATIBILITY] "
                f"compatible={str(bool(promotion_pending_compatible)).lower()} "
                f"reason={promotion_pending_reason} kind=promotion_target_clarification",
                flush=True,
            )
            if not promotion_pending_compatible and addressed:
                self._increment_conversation_attempt(active_pending, reason=promotion_pending_reason)
            elif not promotion_pending_compatible:
                promotion_pending_reason = "ambient_source_ignored"
                print(
                    "[HEBE][PENDING_COMPATIBILITY] compatible=false reason=ambient_source_ignored "
                    "attempts_unchanged=true kind=promotion_target_clarification",
                    flush=True,
                )
                print("[HEBE][PROMOTION_PENDING] ignored_ambient=true", flush=True)
        elif active_pending and pending_kind == "promotion_target_clarification" and not addressed:
            print(
                "[HEBE][PENDING_COMPATIBILITY] compatible=false reason=ambient_source_ignored "
                "attempts_unchanged=true kind=promotion_target_clarification",
                flush=True,
            )
            print("[HEBE][PROMOTION_PENDING] ignored_ambient=true", flush=True)
        game_pending_compatible = False
        game_pending_reason = "no_game_guidance_pending"
        game_pending_updates = {}
        if (
            active_pending
            and interpretation.speech_act != InputSpeechAct.OWNER_FEEDBACK
            and pending_kind == "game_guidance_clarification"
            and not stronger_request
        ):
            game_pending_compatible, game_pending_reason = self._game_guidance_pending_compatibility(
                pending=active_pending,
                normalized=normalized,
                raw_text=event.raw_text,
                addressed=addressed,
            )
            self._log_game_pending_compat(game_pending_compatible, game_pending_reason, pending=active_pending)
            if game_pending_compatible:
                game_pending_updates = router.game_guidance.parse_clarification_answer(active_pending, event.raw_text)
                if not game_pending_updates:
                    game_pending_compatible = False
                    game_pending_reason = "parser_found_no_game_updates"
                    self._log_game_pending_compat(False, game_pending_reason, pending=active_pending)
        generic_pending_compatible = False
        if (
            active_pending is not None
            and interpretation.speech_act != InputSpeechAct.OWNER_FEEDBACK
            and pending_kind not in {
                "appointment_datetime", "promotion_target_clarification",
                "game_guidance_clarification", "assistant_followup", "tts_scope",
            }
            and active_pending.expected_reply is not None
            and not stronger_request
        ):
            generic_pending_compatible = (
                active_pending.expected_reply.classify(event.raw_text)[0] != ConversationalAct.UNKNOWN
            )
        pending_compatible = bool(
            appointment_pending_compatible or promotion_pending_compatible
            or game_pending_updates or generic_pending_compatible
        )
        if pending_compatible and active_pending and not addressed and self._is_stream_enabled() and self._current_stream_is_live():
            explicit_pending_ok = bool(active_pending.domain_payload.get("explicit_question_asked", False))
            no_wake_ok = bool(active_pending.domain_payload.get("can_accept_no_wake_followup", False))
            if not (explicit_pending_ok and no_wake_ok):
                pending_compatible = False
                print(
                    "[HEBE][PENDING_COMPATIBILITY] compatible=false "
                    "reason=no_explicit_no_wake_followup_permission",
                    flush=True,
                )

        high_confidence_local_app = bool(
            command_mode
            and direct_family == DirectUtteranceIntentFamily.APPLICATION_ACTION.value
            and app_target
            and float(app_plan_result.get("confidence") or 0.0) >= 0.8
        )
        live_gate_action, live_gate_reason = self._live_owner_speech_gate(
            addressed=addressed,
            pending_compatible=pending_compatible,
            conversation_followup=conversation_followup,
        )
        print(
            "[HEBE][LIVE_OWNER_SPEECH_GATE] "
            f"action={live_gate_action} reason={live_gate_reason} "
            f"wake={str(bool(addressed)).lower()} pending={str(bool(active_pending)).lower()} "
            f"compatible={str(bool(pending_compatible)).lower()}",
            flush=True,
        )
        if live_gate_action != "reply":
            conversation_followup = False

        if pending_compatible and live_gate_action == "reply":
            interpretation = self._get_input_interpreter().interpret_event(
                event,
                authority="owner",
                addressed_to_hebe=addressed,
                explicit_command_mode=False,
                pending_valid=True,
                recent_hebe_utterance=str(getattr(self, "_last_assistant_text", "") or ""),
            )
            source, authority, trust = "owner_stt_followup", "owner", "trusted_followup"
            input_type = "pending_reply"
            reason = (
                "game_guidance_answer" if game_pending_updates
                else "promotion_target_answer" if promotion_pending_compatible
                else "canonical_expected_reply" if generic_pending_compatible
                else "datetime_answer"
            )
        elif addressed and not isolated_dialogue:
            source, authority, trust = "owner_stt_direct", "owner", "trusted_direct"
            input_type = (
                "owner_feedback" if interpretation.speech_act == InputSpeechAct.OWNER_FEEDBACK
                else "local_app_command" if app_target
                else "explicit_question" if router._looks_like_question(normalized)
                else "direct_to_hebe"
            )
            reason = "wake_or_addressing_evidence"
        elif high_confidence_local_app:
            source, authority, trust = "owner_stt_command", "owner", "trusted_direct"
            input_type, reason = "local_app_command", "high_confidence_local_command"
        elif conversation_followup and live_gate_action == "reply":
            source, authority, trust = "owner_stt_followup", "owner", "trusted_followup"
            input_type, reason = "active_conversation_followup", "active_conversation_state"
        else:
            source, authority, trust = "ambient_stt", "ambient", "untrusted_ambient"
            input_type = "ambient_game_dialogue" if isolated_dialogue else "ambient_stream_context"
            reason = "quoted_or_game_dialogue_isolated" if isolated_dialogue else live_gate_reason if live_gate_action != "reply" else "no_wake_no_pending_no_command"

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
            app_plan_result=app_plan_result,
            active_conversation=active_pending,
            pending_compatible=pending_compatible,
            expected_reply_type=expected_reply_type,
            is_followup_candidate=bool(pending_compatible or conversation_followup),
            input_type=input_type,
            reason=reason,
            interpretation=interpretation,
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
        log_jsonl_event("input_firewall", {
            "raw_text": event.raw_text,
            "normalized_text": normalized,
            "source": source,
            "authority": authority,
            "trust": trust,
            "addressed_to_hebe": addressed,
            "matched_wake_name": getattr(wake, "matched_name", None),
            "pending_compatible": pending_compatible,
            "pending_kind": pending_kind,
            "intent_candidates": intent_candidates,
            "input_type": input_type,
            "reason": reason,
            "allowed_actions": (
                [ACTION_PROMOTION_SHOUTOUT, ACTION_TWITCH_ACTION]
                if promotion_pending_compatible and live_gate_action == "reply"
                else []
            ),
            "blocked_actions": [],
        })
        if active_pending:
            pending_reject_reason = (
                game_pending_reason if pending_kind == "game_guidance_clarification"
                else promotion_pending_reason if pending_kind == "promotion_target_clarification"
                else "not_datetime_or_new_request"
            )
            print(
                "[HEBE][PENDING_FOLLOWUP_GATE] active=true "
                f"compatible={str(pending_compatible).lower()} "
                f"source={source if pending_compatible else 'none'} "
                f"reason={reason if pending_compatible else pending_reject_reason}",
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
        interpretation = getattr(event, "interpretation", None)
        if interpretation is None and getattr(event, "envelope", None) is not None:
            interpretation = event.envelope.interpretation
        if interpretation is not None and interpretation.speech_act in {
            InputSpeechAct.OWNER_FEEDBACK,
            InputSpeechAct.VIEWER_DIRECTED_TO_HEBE,
            InputSpeechAct.VIEWER_CONTEXT,
            InputSpeechAct.AMBIENT_CONTEXT,
            InputSpeechAct.SYSTEM_EVENT,
        } and not interpretation.authorized_action_command:
            return False
        direct_family = str(
            ((event.stt_metadata or {}).get("direct_stt_command") or {}).get("detected_intent_family")
            or ""
        )
        if direct_family in {
            DirectUtteranceIntentFamily.DIRECT_QUESTION.value,
            DirectUtteranceIntentFamily.CASUAL_CONVERSATION.value,
        }:
            return False
        if direct_family == DirectUtteranceIntentFamily.APPLICATION_ACTION.value:
            return bool(interpretation.authorized_action_command if interpretation else True)
        if direct_family == DirectUtteranceIntentFamily.INCOMPLETE_COMMAND.value:
            return False
        pending = self._active_current_conversation(source="stt_voice")
        if pending is not None and pending.topic == "promotion_target_clarification":
            compatible, reason, _resolution = self._promotion_pending_reply_compatible(
                getattr(event, "raw_text", "") or getattr(event, "normalized_text", ""),
                getattr(event, "normalized_text", ""),
                pending,
            )
            print(
                f"[HEBE][PROMOTION_PENDING] compatible_followup_probe={str(bool(compatible)).lower()} reason={reason}",
                flush=True,
            )
            return bool(compatible)
        if interpretation is not None:
            return bool(interpretation.authorized_action_command)
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

    def _trace_canonical_open_app_plan(self, plan: Plan, input_event: InputEvent) -> None:
        action_step = next((step for step in plan.steps if step.type == "action"), None)
        data = (action_step.data or {}) if action_step else {}
        if data.get("name") != "open_application":
            return
        target = str((data.get("params") or {}).get("requested_target") or "").strip()
        print(f"[HEBE][APP_TARGET_EXTRACT] verb=open raw_target={target!r}", flush=True)
        print(
            f"[HEBE][ACTION_PLAN] action_type=open_application target={target} status=complete",
            flush=True,
        )
        emit("voice.command", {
            "raw_text": input_event.raw_text,
            "normalized_text": input_event.normalized_text,
            "intent": "open_application",
            "target": target,
            "status": "complete",
            "source": input_event.source,
            "final_decision": "accepted",
            "owner": "DeliberationService",
        })

    def _record_canonical_open_app_execution(self, execution: ExecutionResult) -> None:
        step = execution.first_result_of_type("action")
        data = (step.data or {}) if step else {}
        if data.get("action_name") != "open_application":
            return
        params = data.get("params") or {}
        result = data.get("action_result")
        payload = getattr(result, "data", {}) or {}
        success = bool(step and step.success)
        error_code = payload.get("error_code") or getattr(result, "error", None)
        target = payload.get("app_id") or params.get("requested_target")
        print(
            "[HEBE][ACTION_EXECUTOR] action_type=open_application "
            f"success={str(success).lower()}"
            + (f" error_code={error_code}" if error_code else ""),
            flush=True,
        )
        event = getattr(self, "_current_input_event", None)
        interaction_trace_id = str(
            ((getattr(event, "stt_metadata", {}) or {}).get("interaction_trace_id") if event is not None else "")
            or f"action_{uuid.uuid4().hex}"
        )
        self._record_interaction_decision({
            "trace_id": interaction_trace_id,
            "event_id": interaction_trace_id,
            "source": str(getattr(event, "source", "") or "owner_input"),
            "actor": "Leo",
            "actor_identities": ["Leo"],
            "target": str(target or ""),
            "interaction_decision": "effect_executed" if success else "effect_failed",
            "authority": "owner",
            "requested_effect": "open_application",
            "effect_authorized": True,
            "reply_authorized": True,
            "reason_code": str(error_code or "application_launch_succeeded"),
            "response_intent": "action_confirmation" if success else "action_failure",
            "generation_outcome": "pending",
            "emission_outcome": "pending",
        })
        if event is not None and isinstance(getattr(event, "stt_metadata", None), dict):
            event.stt_metadata["direct_stt_execution"] = {
                "success": success,
                "reason": error_code or ("executed" if success else "launch_failed"),
                "target": target,
            }
        self._commit_current_direct_stt_terminal(
            outcome="action_executed" if success else "action_failed",
            reason="application_launch_succeeded" if success else str(error_code or "launch_failed"),
            action_receipt={
                "action_type": "open_application",
                "target": target,
                "executor_invoked": True,
                "success": success,
                "external_confirmation": str(payload.get("executed_command") or ""),
                "timestamp": time.time(),
            } if success else None,
        )
        print(
            "[HEBE][ACTION_CLAIM_GUARD] "
            f"claimed_action=open_application execution_success={str(success).lower()} passed=true",
            flush=True,
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

    def _get_presence_engine(self) -> PresenceEngine:
        engine = getattr(self, "presence_engine", None)
        if engine is None:
            engine = PresenceEngine()
            self.presence_engine = engine
        return engine

    def _get_core_loop(self) -> HebeCoreLoop:
        loop = getattr(self, "core_loop", None)
        if loop is None:
            loop = HebeCoreLoop(presence_engine=self._get_presence_engine())
            self.core_loop = loop
        return loop

    def _get_final_emission_gate(self) -> FinalEmissionGate:
        gate = getattr(self, "final_emission_gate", None)
        if gate is None:
            gate = FinalEmissionGate()
            self.final_emission_gate = gate
        return gate

    def _get_stream_tts_safety(self) -> StreamTTSSafetyManager:
        manager = getattr(self, "stream_tts_safety", None)
        if manager is None:
            manager = StreamTTSSafetyManager()
            self.stream_tts_safety = manager
        return manager

    def _get_twitch_interaction_coordinator(self) -> TwitchInteractionCoordinator:
        coordinator = getattr(self, "twitch_interaction_coordinator", None)
        if coordinator is None:
            coordinator = TwitchInteractionCoordinator(
                repeat_window_seconds=float(os.getenv("HEBE_TWITCH_SEMANTIC_REPEAT_WINDOW_SECONDS", "120") or 120)
            )
            self.twitch_interaction_coordinator = coordinator
        return coordinator

    def _get_troll_engagement_budget(self) -> TrollEngagementBudget:
        budget = getattr(self, "troll_engagement_budget", None)
        if budget is None:
            budget = TrollEngagementBudget(
                window_seconds=float(os.getenv("HEBE_TWITCH_BAIT_WINDOW_SECONDS", "300") or 300)
            )
            self.troll_engagement_budget = budget
        return budget

    def _emit_final_response(
        self,
        *,
        event_id: str = "",
        source: str = "",
        final_response: str = "",
        output_route: str | OutputRoute = OutputRoute.SUPPRESS,
        output_targets: list[str] | tuple[str, ...] | None = None,
        guard_result: dict | None = None,
        repair_summary: dict | None = None,
        execution_result: dict | None = None,
        debug_payload: dict | None = None,
        runtime_context: str | HebeLiveRuntimeContext | None = None,
        send_twitch_fn=None,
        speak_fn=None,
    ) -> dict:
        debug_payload = dict(debug_payload or {})
        debug_payload["response_stage"] = "final"
        context_policy = getattr(self, "live_context_policy", None) or HebeLiveContextPolicy()
        requested_context = runtime_context or debug_payload.get("runtime_context")
        resolved_context = (
            HebeLiveRuntimeContext(str(requested_context))
            if requested_context
            else context_policy.from_source(source)
        )
        debug_payload["runtime_context"] = resolved_context.value
        viewer = str(
            debug_payload.get("intended_recipient") or debug_payload.get("user_login")
            or debug_payload.get("username") or debug_payload.get("speaker") or ""
        ).lstrip("@")
        source_viewer = str(debug_payload.get("source_viewer") or debug_payload.get("user_login") or viewer)
        stream = self._get_stream_state()
        if final_response and viewer and stream is not None:
            requester_is_owner = bool(
                debug_payload.get("requester_is_owner")
                or debug_payload.get("authority") in {"owner", "leo"}
            )
            authority_guard = getattr(self, "social_authority_commitment_guard", None) or SocialAuthorityCommitmentGuard()
            authority_result = authority_guard.evaluate(final_response, requester_is_owner=requester_is_owner)
            if authority_result.action != "allow":
                final_response = authority_result.text
                repair_summary = {**dict(repair_summary or {}), "social_authority": authority_result.reason}
            retention_guard = getattr(self, "channel_retention_guard", None) or ChannelRetentionGuard()
            retention_result = retention_guard.evaluate(
                final_response,
                owner_directed_moderation=bool(debug_payload.get("owner_directed_moderation")),
                safety_required=bool(debug_payload.get("safety_required")),
                quoted_or_discussed=bool(debug_payload.get("quoted_or_discussed")),
            )
            if retention_result.action != "allow":
                final_response = retention_result.text
                repair_summary = {**dict(repair_summary or {}), "channel_retention": retention_result.reason}
            adaptation_service = getattr(self, "behavior_adaptation", None)
            if adaptation_service is None:
                adaptation_service = BehaviorAdaptationService()
                self.behavior_adaptation = adaptation_service
            constraint_result = adaptation_service.validate_constraint_output(
                stream, intended_recipient=viewer,
                generated_response=final_response, source_viewer=source_viewer,
                speech_act=str(debug_payload.get("speech_act") or ""), scene_context=debug_payload,
            )
            if not constraint_result.get("passed", True):
                final_response = str(constraint_result.get("repaired_response") or "").strip()
                repair_summary = {**dict(repair_summary or {}), "behavior_constraint": constraint_result}
                debug_payload["behavior_constraint_repaired"] = True
                if not final_response:
                    guard_result = {"passed": False, "reason": "owner_behavior_constraint"}
                    output_route = OutputRoute.SUPPRESS
                    output_targets = []
            try:
                profile_store = getattr(self, "viewer_linguistic_profiles", None) or ViewerLinguisticProfileStore()
                profile = profile_store.get(
                    twitch_user_id=str(debug_payload.get("twitch_user_id") or ""), login=viewer,
                )
                agreement_guard = getattr(self, "grammatical_agreement_guard", None) or GrammaticalAgreementGuard()
                agreement = agreement_guard.evaluate(
                    final_response, viewer=viewer, profile=profile,
                    refers_to_hebe=bool(debug_payload.get("refers_to_hebe")),
                )
                final_response = str(agreement.get("text") or final_response)
                if agreement.get("action") != "allow":
                    repair_summary = {**dict(repair_summary or {}), "grammatical_agreement": agreement}
            except Exception as exc:
                print(f"[HEBE][GRAMMATICAL_AGREEMENT_GUARD] viewer={viewer} action=allow reason=store_error error={exc!r}", flush=True)
        if source in {"twitch", "spontaneity"} and not self._get_twitch_interaction_coordinator().allows_final_emission(event_id):
            output_route = OutputRoute.SUPPRESS
            output_targets = []
            guard_result = {"passed": False, "reason": "preempted_by_higher_priority_interaction"}
            debug_payload["preempted_before_emission"] = True

        def emit_ui(payload: dict) -> None:
            ui_payload = {
                "text": payload.get("text", ""),
                "source": payload.get("source", source),
                "output_target": payload.get("output_target", OUTPUT_TARGET_LOCAL_UI),
            }
            if payload.get("message_id"):
                ui_payload["message_id"] = payload.get("message_id")
            if payload.get("debug_contract"):
                ui_payload["debug_contract"] = payload.get("debug_contract")
            emit("chat.assistant", ui_payload)

        def emit_debug(payload: dict) -> None:
            emit("debug.emission", payload)

        result = self._get_final_emission_gate().emit(
            event_id=event_id,
            source=source,
            final_response=final_response,
            output_route=output_route,
            output_targets=list(output_targets or []),
            guard_result=guard_result,
            repair_summary=repair_summary,
            execution_result=execution_result,
            debug_payload=debug_payload,
            runtime_context=resolved_context.value,
            emit_ui=emit_ui,
            emit_debug=emit_debug,
            send_twitch=send_twitch_fn,
            speak=speak_fn,
        )
        result_dict = result.to_dict()
        stream = self._get_stream_state()
        emission_outcome = "emitted" if result_dict.get("emitted") else str(result_dict.get("reason") or "suppressed")
        interaction_update = self._get_interaction_decision_history().update(
            stream,
            str(event_id or ""),
            emission_outcome=emission_outcome,
            emission_route=str(result_dict.get("route") or ""),
        )
        if interaction_update is not None:
            print(
                f"[HEBE][INTERACTION_EMISSION] trace={event_id} outcome={emission_outcome} "
                f"route={result_dict.get('route')}",
                flush=True,
            )
        behavior_trace_id = str(
            debug_payload.get("behavior_correlation_id")
            or (
                getattr(stream, "last_behavior_correlation_id", "")
                if source == "spontaneity" and stream is not None
                else ""
            )
            or ""
        )
        if behavior_trace_id:
            adaptation_service = getattr(self, "behavior_adaptation", None)
            if adaptation_service is not None:
                adaptation_service.record_emission(
                    trace_id=behavior_trace_id,
                    stream=stream,
                    event_id=event_id,
                    emitted=bool(result_dict.get("emitted")),
                    reason_code=str(result_dict.get("reason") or ""),
                )
            if stream is not None and getattr(stream, "last_behavior_correlation_id", "") == behavior_trace_id:
                stream.last_behavior_correlation_id = ""
        if source in {"twitch", "spontaneity"}:
            self._get_twitch_interaction_coordinator().record_emission(
                event_id,
                result_dict,
                reason=str((guard_result or {}).get("reason") or ""),
            )
        return result_dict

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
        intelligence_context = self._game_intelligence_context_for_event(event, classification)
        if intelligence_context:
            session_context["game_intelligence"] = intelligence_context
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

    def _game_intelligence_context_for_event(self, event: InputEvent | None, classification) -> dict:
        resolver = getattr(self, "game_context_resolver", None)
        stream = self._get_stream_state()
        game = str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or "").strip() if stream else ""
        if resolver is None or not game:
            return {}
        try:
            run_id=str(getattr(stream,"active_game_run_id","") or "")
            raw=str(getattr(event,"raw_text","") or "") if event is not None else ""
            normalized=self._normalize_text(raw)
            game_question=bool(re.search(r"\b(?:juego|game|mecanica|combate|boss|jefe|vida|hp|damage|como funciona)\b",normalized))
            context=resolver.build(
                game=game,purpose="game_fact" if game_question else "stream_reaction",
                stream_session_id=str(getattr(stream,"active_stream_session_id","") or ""),
                run_id=run_id,event_id=str(getattr(event,"timestamp","") or "stream_context"),
                spoiler_ceiling="strict",allow_research=False,
            )
            claims=list(context.knowledge_claims)[:12]
            safe_claims=[item.get("object") for item in claims if item.get("object") is not None]
            return {
                "game_id":context.game_identity.get("game_id"),
                "dossier_status":"canonical_knowledge" if claims else "missing",
                "progress":self.game_run_service.state(run_id) if run_id else {},
                "contribution_mode":"informed_observation" if safe_claims else "contextual_reaction",
                "allowed_claims":safe_claims,
                "source_provenance":list(context.provenance_manifest)[:12],
                "candidate_fact_ids":[str(item.get("id") or "") for item in claims],
                "forbidden_claims":["unverified mechanics","future story information","walkthrough solutions"],
                "lookup_used":bool(claims),
                "instruction":"Use only allowed claims; otherwise make a scene reaction. Never turn this into unsolicited walkthrough advice.",
            }
        except Exception as exc:
            print(f"[HEBE][GAME_INTELLIGENCE] context_failed={type(exc).__name__}",flush=True)
            return {}

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
        diagnostics = getattr(getattr(self, "game_intelligence", None), "diagnostics", None)
        lookup_used = bool(getattr(diagnostics, "lookup_used", False))
        if not lookup_used and knowledge.game:
            try:
                identity=self.game_v2_repository.resolve_identity(knowledge.game)
                lookup_used=bool(self.game_v2_repository.knowledge(identity.game_id))
            except Exception:
                pass
        print(
            "[HEBE][KNOWLEDGE] "
            f"game={knowledge.game!r} profile_found={str(knowledge.profile_found).lower()} "
            f"lookup_used={str(lookup_used).lower()} confidence={knowledge.confidence} "
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
        mode, _reason = self._stream_voice_mode_active()
        if mode == "muted":
            print("[HEBE][PROACTIVE_SUPPRESSED] reason=owner_mute", flush=True)
            return False
        if mode == "wake_only" and event_type not in {"owner_direct", "direct_stt"}:
            print("[HEBE][PROACTIVE_SUPPRESSED] reason=owner_mute", flush=True)
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
        mode, reason = self._stream_voice_mode_active()
        if mode in {"wake_only", "muted"}:
            print(f"[HEBE][TTS] skipped reason={reason or 'owner_mute'}", flush=True)
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

        fuzzy_stream_op = re.search(
            r"\b(?P<wake>efe|y\s+ve|e\s*[- ]?b)\b[\s,;:.-]+(?P<command>(?:haz|tira)\s+(?:una?\s+)?promo\s+.+)$",
            normalized,
            flags=re.IGNORECASE,
        )
        if fuzzy_stream_op:
            recovered_command = str(fuzzy_stream_op.group("command") or "").strip()
            parser = self._get_stream_action_planner().intent_parser
            request = parser.parse_promotion_request(recovered_command, source="owner_stt_direct")
            if request is not None and request.target_phrase:
                print(
                    "[HEBE][STREAM_OP_WAKE_RECOVERY] "
                    f"recovered=true raw_token={fuzzy_stream_op.group('wake')!r} "
                    "action=promotion_shoutout confidence=0.91",
                    flush=True,
                )
                self._disarm_stream()
                return True, recovered_command

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

    def _classify_voice_event(
        self,
        text: str,
        *,
        interpretation: InputInterpretation | None = None,
    ) -> tuple[str, str | None]:
        interpretation = interpretation or getattr(
            getattr(self, "_current_input_event", None), "interpretation", None
        )
        if interpretation is not None and interpretation.speech_act == InputSpeechAct.OWNER_FEEDBACK:
            if not interpretation.context_text:
                return "owner_feedback", None
            text = interpretation.context_text
        semantic_clauses = tuple(getattr(interpretation, "semantic_clauses", ()) or ())
        if not semantic_clauses:
            semantic_clauses = InputInterpreter.analyze_semantic_clauses(
                text,
                speaker=str(getattr(interpretation, "authority", "owner") or "owner"),
                addressed_to_hebe=bool(getattr(interpretation, "addressed_to_hebe", False)),
            )
        semantic_predicates = [
            predicate
            for clause in semantic_clauses
            for predicate in clause.predicates
        ]
        completed_death = next((
            predicate for predicate in semantic_predicates
            if predicate.predicate == "completed_death"
        ), None)
        death_risk = next((
            predicate for predicate in semantic_predicates
            if predicate.predicate == "death_risk"
        ), None)
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
        if completed_death is not None and completed_death.polarity == "positive":
            return "gameplay_failure", "frustrated"
        if death_risk is not None:
            return "combat_risk", "focused"
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
            f"current_pending={self._active_current_conversation(source='stt_voice')!r}",
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
        current_event = getattr(self, "_current_input_event", None)
        interpretation = getattr(current_event, "interpretation", None)
        if interpretation is not None and interpretation.speech_act == InputSpeechAct.OWNER_FEEDBACK:
            stream.last_owner_feedback = interpretation.as_dict()
            metadata = getattr(current_event, "stt_metadata", None)
            already_applied = bool(isinstance(metadata, dict) and metadata.get("behavior_feedback_applied"))
            if not already_applied:
                recent_utterance = None
                try:
                    recent_utterance = self._get_live_session_brain().state.last_hebe_utterance
                except Exception:
                    recent_utterance = str(getattr(self, "_last_assistant_text", "") or "")
                adaptation_service = getattr(self, "behavior_adaptation", None)
                if adaptation_service is None:
                    adaptation_service = BehaviorAdaptationService()
                    self.behavior_adaptation = adaptation_service
                adaptation_service.apply_feedback(
                    stream,
                    interpretation,
                    recent_hebe_utterance=recent_utterance,
                    source_event_id=str(
                        (metadata or {}).get("event_id")
                        or getattr(current_event, "timestamp", "")
                        or ""
                    ),
                )
                if isinstance(metadata, dict):
                    metadata["behavior_feedback_applied"] = True
            if not interpretation.context_text:
                stream.last_voice_event = "owner_feedback"
                stream.last_voice_event_ts = time.time()
                stream.last_owner_utterance_end_ts = stream.last_voice_event_ts
                stream.owner_voice_active = False
                stream.last_voice_summary = "owner feedback about Hebe"
                relevance = ContextRelevance(
                    useful=False,
                    category="owner_feedback",
                    confidence=interpretation.confidence,
                    reason="canonical_feedback_scope_excluded",
                )
                self._log_context_relevance(relevance)
                print(
                    "[HEBE][AMBIENT_CONTEXT] ignored reason=canonical_feedback_scope_excluded",
                    flush=True,
                )
                return relevance
            text = interpretation.context_text
        stream.last_voice_event = event_type
        stream.last_voice_event_ts = time.time()
        # STT inbox entries are normalized completed utterances. Live RMS/VAD
        # remains authoritative for whether the microphone is active now.
        stream.last_owner_utterance_end_ts = stream.last_voice_event_ts
        stream.owner_voice_active = False
        stream.last_voice_summary = self._summarize_voice_event(text, event_type)
        now_ts = time.time()
        role = str(getattr(current_event, "stt_metadata", {}).get("utterance_role") or UtteranceRole.OWNER_COMMENTARY.value)
        role_decision = dict(getattr(current_event, "stt_metadata", {}).get("utterance_role_decision") or {})
        language = str(getattr(current_event, "stt_metadata", {}).get("detected_language") or "es")
        discourse_buffer = getattr(self, "owner_discourse_buffer", None)
        turn_detector = getattr(self, "stream_turn_detector", None)
        if discourse_buffer is not None and bool(role_decision.get("discourse_allowed", role in {
            UtteranceRole.OWNER_COMMENTARY.value, UtteranceRole.OWNER_QUESTION_TO_STREAM.value,
        })):
            topic = discourse_buffer.add_fragment(text, timestamp=now_ts, confidence=1.0, language=language)
            stream.current_discourse_topic = topic.to_dict()
            planner = getattr(self, "discourse_contribution_planner", None)
            if planner is not None:
                plan = planner.plan(topic)
                stream.proposed_discourse_contribution = plan.to_dict()
        if turn_detector is not None and role in {
            UtteranceRole.OWNER_COMMENTARY.value, UtteranceRole.OWNER_QUESTION_TO_STREAM.value,
        }:
            turn_detector.record_owner_fragment(text, timestamp=now_ts)
        if mood_hint:
            stream.leo_mood_hint = mood_hint
        self._apply_ambient_voice_to_run_context(stream, text, event_type)
        relevance = self._extract_and_store_ambient_context(
            stream, text, event_type, utterance_role=role, language=language,
        )
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

    def _extract_and_store_ambient_context(
        self, stream, text: str, event_type: str, *,
        utterance_role: str = "owner_commentary", language: str | None = None,
    ) -> ContextRelevance:
        extractor = getattr(self, "ambient_context_extractor", None)
        if extractor is None:
            extractor = AmbientContextExtractor()
            self.ambient_context_extractor = extractor
        extraction = extractor.extract(
            text, event_type=event_type, utterance_role=utterance_role,
            language=language,
            topic_id=str((getattr(stream, "current_discourse_topic", {}) or {}).get("topic_id") or ""),
            scene_id=str((getattr(stream, "current_scene_timeline", {}) or {}).get("scene_id") or ""),
            input_interpretation=getattr(
                getattr(self, "_current_input_event", None), "interpretation", None
            ),
        )
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
            diagnostics=dict(getattr(extraction, "diagnostics", {}) or {}),
        )
        facts = list(getattr(stream, "recent_run_context_facts", []) or [])
        now = time.time()
        facts = [
            fact for fact in facts
            if float(fact.get("expires_at", 0.0) or 0.0) > now
        ]
        topic_id = str((getattr(stream, "current_discourse_topic", {}) or {}).get("topic_id") or "")
        scene_timeline = getattr(self, "scene_timeline", None)
        extracted_facts = list(extraction.facts)
        if scene_timeline is not None:
            event_id = str(
                (extracted_facts[0].get("id") if extracted_facts else "")
                or (extracted_facts[0].get("fact_id") if extracted_facts else "")
                or f"ambient_scene:{uuid.uuid4().hex}"
            )
            scene_timeline.observe(
                text,
                event_id=event_id,
                topic_id=topic_id,
                facts=extracted_facts,
                now=now,
            )
            extracted_facts = scene_timeline.annotate_facts(
                extracted_facts,
                topic_id=topic_id,
                now=now,
            )
            for fact in extracted_facts:
                fact.setdefault("inferred_claims", [])
                fact.setdefault("unsupported_claims", [])
                if isinstance(fact.get("data"), dict):
                    fact["data"]["scene_id"] = str(fact.get("scene_id") or "")
                    fact["data"]["inferred_claims"] = list(fact.get("inferred_claims") or [])
            current = scene_timeline.current
            if current is not None:
                superseded_ids = set(current.superseded_event_ids)
                for fact in facts:
                    fact_id = str(fact.get("id") or fact.get("fact_id") or "")
                    if fact_id in superseded_ids:
                        fact["superseded"] = True
                        fact["currentness_score"] = 0.0
                    if str(fact.get("scene_id") or "") == current.scene_id:
                        fact["current_state"] = current.current_state
                        fact["state_version"] = current.state_version
                        fact["terminal"] = current.terminal
                stream.current_scene_timeline = current.to_dict()
        facts.extend(extracted_facts)
        stream.recent_run_context_facts = facts[-20:]
        stream.run_context_updated_ts = now
        stream.run_context_source = "stt_voice"
        if extraction.mood:
            stream.leo_mood_hint = extraction.mood
        extraction = type(extraction)(
            useful=extraction.useful,
            facts=extracted_facts,
            mood=extraction.mood,
            reason=extraction.reason,
        )
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

        promotion_profile_result = self._handle_promotion_profile_command(raw_command)
        if promotion_profile_result is not None:
            return promotion_profile_result

        # A complete new owner command is its own transaction. It must not be
        # consumed as the answer to an older promotion clarification.
        fresh_plan = self._get_stream_action_planner().plan(InputEvent(
            source=source or "owner_manual",
            raw_text=raw_command,
            normalized_text=raw_command,
            is_stream_context=True,
        ))
        if (
            fresh_plan is not None
            and fresh_plan.action_type == "twitch_shoutout"
            and fresh_plan.status == "complete"
            and float(fresh_plan.confidence or 0.0) >= 0.78
            and fresh_plan.target
            and not ({"previsualiza", "preview"} & set(normalized.split()))
        ):
            old_pending = self._active_current_conversation(latest=True)
            if old_pending is not None and old_pending.topic == "promotion_target_clarification":
                self._close_current_conversation(reason="superseded_by_fresh_promotion", conversation=old_pending)
                print(
                    f"[HEBE][PROMOTION_PENDING] superseded id={old_pending.id} target={fresh_plan.target}",
                    flush=True,
                )

        language_match = re.search(
            r"\b(?:idioma(?: del)? stream|idioma(?: del)? directo|stream language)\b.*\b(espanol|español|ingles|inglés|spanish|english|es|en)\b",
            normalized,
        )
        if language_match:
            requested = language_match.group(1)
            language = "en" if requested in {"ingles", "inglés", "english", "en"} else "es"
            policy = self.response_synthesizer.stream_output_language
            policy.set_owner_preference(language)
            stream.stream_output_language = language
            return command_result(
                "stream_output_language_changed",
                "Idioma autónomo del stream cambiado a inglés." if language == "en" else "Idioma autónomo del stream cambiado a español.",
                state_changes={"stream_output_language": language},
                message_goal=f"Confirm autonomous stream output language changed to {language}.",
            )

        pending_promo = self._resolve_pending_promotion_target(raw_command, normalized, stream)
        if pending_promo is not None:
            return pending_promo

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
            summary = self._close_stream_memory_session_safe(stream, reason="manual_command")
            stream.enabled = False
            return self._format_stream_summary_reply(summary) if summary else "No habia una sesion de stream activa que finalizar."

        if normalized in {"resume este stream", "resumen de este stream"}:
            session_id = self._ensure_stream_memory_session_if_live(stream) or getattr(stream, "active_stream_session_id", None)
            if not session_id:
                return "No tengo una sesion de stream activa para resumir."
            summary = stream_memory.summarize_stream_session(int(session_id), reason="manual_summary")
            self._persist_canonical_chatter_summaries(summary)
            return self._format_stream_summary_reply(summary)

        if normalized in {"que paso en el ultimo stream", "qué pasó en el último stream"}:
            summary = stream_memory.get_latest_stream_summary()
            return self._format_latest_stream_summary_reply(summary)

        chatter_match = re.match(r"^(?:que dijo|qué dijo)\s+(.+?)\s+en el ultimo stream$", normalized)
        if chatter_match:
            target = chatter_match.group(1).strip()
            summary = self.social_world.latest_summary_for_login(target) if getattr(self,"social_world",None) is not None else {}
            if not summary:
                return f"No tengo resumen del ultimo stream para {target}."
            return f"En el ultimo stream, {target}: {summary.get('summary_text') or 'sin resumen suficiente'}"

        chatter_match = re.match(r"^(?:que sabes de|qué sabes de)\s+(.+)$", normalized)
        if chatter_match and "este juego" not in normalized and not self._looks_like_game_knowledge_target(chatter_match.group(1)):
            return self.social_world.format_profile_reply(chatter_match.group(1).strip()) if getattr(self,"social_world",None) is not None else "SocialWorld no está disponible."

        chatter_match = re.match(r"^(?:cuando fue la ultima vez que hablo|cuándo fue la última vez que habló)\s+(.+)$", normalized)
        if chatter_match:
            return self.social_world.format_last_seen_reply(chatter_match.group(1).strip(),kind="message") if getattr(self,"social_world",None) is not None else "SocialWorld no está disponible."

        chatter_match = re.match(r"^(?:cuando fue la ultima vez que vimos a|cuándo fue la última vez que vimos a)\s+(.+)$", normalized)
        if chatter_match:
            return self.social_world.format_last_seen_reply(chatter_match.group(1).strip(),kind="seen") if getattr(self,"social_world",None) is not None else "SocialWorld no está disponible."

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

    def _handle_promotion_profile_command(self, raw_command: str) -> CommandResult | None:
        parsed = parse_promotion_profile_command(raw_command)
        if parsed is None:
            return None
        authorization = (getattr(self, "live_context_policy", None) or HebeLiveContextPolicy()).authorize_action(
            HebeLiveRuntimeContext.OWNER_LIVE_CONTROL,
            "promotion.profile.configure",
        )
        if not authorization.allowed:
            return CommandResult(
                action_type="promotion_profile",
                success=False,
                user_visible_summary="Promotion profile change blocked by runtime context.",
                fallback_text="No cambio esa preferencia desde este contexto.",
                requires_model_response=False,
            )
        manager = getattr(self, "promotion_profile_manager", None)
        if manager is None:
            return CommandResult(
                action_type="promotion_profile",
                success=False,
                user_visible_summary="Promotion profile storage is unavailable.",
                fallback_text="No puedo cambiar la lista de promos automáticas ahora.",
                requires_model_response=False,
            )
        result = manager.apply_command(parsed)
        if parsed.action == "list":
            profiles = list(result or [])
            active = [
                profile for profile in profiles
                if profile.active and profile.auto_promo_mode not in {AutoPromoMode.DISABLED.value, AutoPromoMode.MANUAL_ONLY.value}
            ]
            summary = (
                "Promos automáticas: " + ", ".join(
                    f"{profile.current_login} ({profile.auto_promo_mode})" for profile in active
                )
                if active
                else "No hay promociones automáticas configuradas."
            )
            return CommandResult(
                action_type="promotion_profile_list",
                success=True,
                user_visible_summary=summary,
                fallback_text=summary,
                requires_model_response=False,
            )
        if parsed.action == "enable":
            ok = result is not None
            summary = (
                f"Promocionaré a {parsed.target} cuando escriba por primera vez en cada directo."
                if ok else f"No he podido guardar la promoción automática de {parsed.target}."
            )
        elif parsed.action == "disable":
            ok = result is not None
            summary = (
                f"Promoción automática desactivada para {parsed.target}."
                if ok else f"No tenía una promoción automática para {parsed.target}."
            )
        else:
            ok = bool(result)
            summary = (
                f"Preferencia de promoción borrada para {parsed.target}."
                if ok else f"No tenía una preferencia de promoción para {parsed.target}."
            )
        return CommandResult(
            action_type=f"promotion_profile_{parsed.action}",
            success=ok,
            user_visible_summary=summary,
            fallback_text=summary,
            requires_model_response=False,
            state_changes={"target": parsed.target, "action": parsed.action},
            constraints=["Acknowledge locally only.", "Do not write a Twitch message."],
        )

    def _plan_and_execute_stream_action(self, raw_command: str, normalized: str, stream) -> CommandResult | None:
        input_event = getattr(self, "_current_input_event", None) or self._build_input_event(
            source="typed_ui",
            raw_text=raw_command,
            normalized_text=normalized,
        )
        planner = self._get_stream_action_planner()
        print("[HEBE][ACTION_PERMISSION_PRECHECK] action=promotion_shoutout status=undecided", flush=True)
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
        if plan.action_type == "twitch_shoutout":
            allowed = plan.status in {"complete", "needs_confirmation", "missing_target"}
            reason = "trusted_explicit_stream_op" if allowed else str(plan.reason or "invalid_command")
            print(f"[HEBE][ACTION_PERMISSION_FINAL] action=promotion_shoutout allowed={str(allowed).lower()} reason={reason}", flush=True)
        if plan.action_type == "twitch_shoutout":
            event_id = str(
                getattr(input_event, "timestamp", "")
                or (getattr(input_event, "stt_metadata", {}) or {}).get("event_id")
                or f"promotion_{uuid.uuid4().hex}"
            )
            plan.slots = {
                **dict(plan.slots or {}),
                "event_id": event_id,
                "owner_command_text": raw_command,
            }
            print(
                f"[HEBE][STREAM_OPS_COMMAND] event_id={event_id} type=promotion_shoutout detected=true source={input_event.source}",
                flush=True,
            )
            if stream is not None:
                stream.last_promo_parse = plan.as_log_dict()
                stream.last_promo_rejected_reason = "" if plan.status == "complete" else str(plan.reason or "")
            self._resolve_recent_raid_shoutout_plan(plan, stream)
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
            event_id = str((plan.slots or {}).get("event_id") or f"promotion_{uuid.uuid4().hex}")
            result = self._get_twitch_interaction_coordinator().submit_owner_stream_operation(
                event_id=event_id,
                text=raw_command,
                processor=lambda: self._execute_twitch_shoutout_plan(plan, stream),
            )
            outcome = dict(getattr(stream, "last_promotion_outcome", {}) or {}) if stream is not None else {}
            passed = bool(outcome and str(outcome.get("event_id") or "") == event_id)
            if not passed:
                print(
                    "[HEBE][PROMOTION_PIPELINE_INVARIANT] passed=false reason=resolved_without_outcome"
                    if plan.target else
                    "[HEBE][PROMOTION_PIPELINE_INVARIANT] passed=false reason=command_without_outcome",
                    flush=True,
                )
                self._record_promotion_outcome(
                    stream,
                    event_id=event_id,
                    outcome="error",
                    reason="resolved_without_outcome" if plan.target else "command_without_outcome",
                    target=plan.target or "",
                )
            else:
                print(f"[HEBE][PROMOTION_PIPELINE_INVARIANT] passed=true reason={outcome.get('outcome')}", flush=True)
            return result
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
        event_id = str((plan.slots or {}).get("event_id") or f"promotion_{uuid.uuid4().hex}")
        if stream is not None:
            stream.last_promo_parse = plan.as_log_dict()
            stream.last_promo_rejected_reason = "" if plan.status == "complete" else str(plan.reason or "")
        if plan.status == "needs_confirmation":
            print(
                "[HEBE][ACTION_EXECUTOR] success=false reason=needs_confirmation "
                f"action_type={plan.action_type}",
                flush=True,
            )
            print(f"[HEBE][PROMOTION_EXECUTION_DECISION] allowed=false reason={plan.reason or 'needs_confirmation'}", flush=True)
            if stream is not None:
                stream.last_promo_execution_decision = {
                    "allowed": False,
                    "reason": plan.reason or "needs_confirmation",
                    "target": plan.target,
                    "ts": time.time(),
                }
            if "target" in plan.missing_slots or plan.reason in {"missing_target", "target_unclear", "invalid_target"}:
                fallback = "¿A quién le hago el SO, Leo?"
                goal = "Ask Leo which Twitch user should receive the shoutout."
            elif plan.reason == "ambiguous_target":
                fallback = "He pillado varios nombres parecidos. Dime el usuario exacto para el SO."
                goal = f"Ask Leo to clarify the shoutout target. Candidates: {', '.join(plan.candidates)}."
            else:
                fallback = "Creo que me has pedido un SO, pero necesito confirmación."
                goal = "Ask Leo to confirm the shoutout target before sending it."
            self._create_promotion_pending(plan, fallback=fallback)
            self._record_promotion_outcome(
                stream, event_id=event_id, outcome="clarification",
                reason=plan.reason or "needs_confirmation", target=plan.target or "",
            )
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
            print(f"[HEBE][PROMOTION_DROPPED_GUARD] dropped=true reason={plan.reason or 'rejected'}", flush=True)
            self._record_promotion_outcome(
                stream, event_id=event_id, outcome="blocked",
                reason=plan.reason or "rejected", target=plan.target or "",
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

        authorization = (getattr(self, "live_context_policy", None) or HebeLiveContextPolicy()).authorize_action(
            HebeLiveRuntimeContext.OWNER_LIVE_CONTROL,
            "promotion.owner_manual",
        )
        print(
            "[HEBE][LIVE_CONTEXT_GATE] "
            f"context={authorization.context} operation={authorization.operation} "
            f"allowed={str(authorization.allowed).lower()} reason={authorization.reason}",
            flush=True,
        )
        if not authorization.allowed:
            return CommandResult(
                action_type="twitch_shoutout",
                success=False,
                user_visible_summary="Shoutout blocked by runtime context.",
                constraints=["Do not claim the shoutout was sent."],
                fallback_text="No ejecuto esa promo desde este contexto.",
                requires_model_response=False,
            )

        print(f"[HEBE][ACTION_EXECUTOR] executing action_type={plan.action_type} target={plan.target}", flush=True)
        print(f"[HEBE][PROMOTION_EXECUTION_DECISION] allowed=true reason=resolved_owner_command target={plan.target}", flush=True)
        if stream is not None:
            stream.last_promo_execution_decision = {
                "allowed": True,
                "reason": "resolved_owner_command",
                "target": plan.target,
                "ts": time.time(),
            }
        persistent_event = None
        promotion_store = getattr(self, "promotion_store", None)
        if promotion_store is not None:
            try:
                owner_command = str((plan.slots or {}).get("owner_command_text") or "")
                trigger_type = (
                    PromotionTriggerType.OWNER_MANUAL
                    if re.search(r"\b(?:solo esta vez|una sola vez|only this time|just this once)\b", self._normalize_text(owner_command))
                    else PromotionTriggerType.OWNER_LEARN_AND_EXECUTE
                )
                persistent_event = promotion_store.create_event(
                    stream_session_id=getattr(stream, "active_stream_session_id", None) or "offline",
                    source_event_id=event_id,
                    requested_by="leo",
                    raw_target_text=str((plan.slots or {}).get("target_raw") or plan.target or ""),
                    resolved_twitch_user_id=(
                        self._twitch_user_ids_by_login.get(self._normalize_shoutout_target(plan.target).casefold())
                        or f"login:{self._normalize_shoutout_target(plan.target)}"
                    ),
                    resolved_login=self._normalize_shoutout_target(plan.target),
                    resolution_confidence=float(plan.confidence or 0.0),
                    trigger_type=trigger_type,
                )
                if persistent_event.execution_status == PromotionExecutionStatus.PROPOSED.value:
                    promotion_store.transition(persistent_event.id, PromotionExecutionStatus.RESOLVING)
                    persistent_event = promotion_store.transition(persistent_event.id, PromotionExecutionStatus.QUEUED)
            except Exception as exc:
                persistent_event = None
                print(f"[HEBE][PROMOTION_EVENT] persist_failed={type(exc).__name__}", flush=True)
        ok, normalized_target, send_reason = self._send_shoutout(plan.target, source="manual", force=False)
        if persistent_event is not None and persistent_event.execution_status == PromotionExecutionStatus.QUEUED.value:
            try:
                terminal_status = (
                    PromotionExecutionStatus.SENT
                    if ok
                    else PromotionExecutionStatus.BLOCKED
                    if send_reason in {"blocked_bot_user", "own_channel", "invalid_target", "cooldown_active"}
                    else PromotionExecutionStatus.FAILED
                )
                persistent_event = promotion_store.transition(
                    persistent_event.id,
                    terminal_status,
                    twitch_message_id=(
                        str(getattr(self, "_last_shoutout_external_confirmation", "") or "")
                        if ok else ""
                    ),
                    failure_reason="" if ok else send_reason,
                )
            except Exception as exc:
                print(f"[HEBE][PROMOTION_EVENT] transition_failed={type(exc).__name__}", flush=True)
        if ok:
            print(
                f"[HEBE][PROMOTION_EXECUTE] target={normalized_target} command={self._build_shoutout_command_preview(normalized_target)} success=true",
                flush=True,
            )
        print(
            "[HEBE][ACTION_EXECUTOR] "
            f"success={ok} action_type={plan.action_type} target={normalized_target} reason={send_reason}",
            flush=True,
        )
        if stream is not None:
            stream.last_promo_execution_decision = {
                "allowed": bool(ok),
                "reason": send_reason,
                "target": normalized_target or plan.target,
                "ts": time.time(),
            }
        if not ok:
            print(f"[HEBE][PROMOTION_EXECUTION_DECISION] allowed=false reason={send_reason} target={normalized_target or plan.target}", flush=True)
            print(f"[HEBE][PROMOTION_DROPPED_GUARD] dropped=true reason={send_reason}", flush=True)
        if ok:
            action_receipt = {
                "action_type": "twitch_shoutout",
                "target": normalized_target,
                "executor_invoked": True,
                "success": True,
                "external_confirmation": str(
                    getattr(persistent_event, "twitch_message_id", "")
                    or getattr(self, "_last_shoutout_external_confirmation", "")
                    or send_reason or "twitch_send_success"
                ),
                "timestamp": time.time(),
            }
            current_input_event = getattr(self, "_current_input_event", None)
            if current_input_event is not None:
                current_input_event.stt_metadata["direct_stt_execution"] = {
                    "success": True, "reason": send_reason, "target": normalized_target,
                    "action_receipt": action_receipt,
                }
            self._commit_current_direct_stt_terminal(
                outcome="action_executed", reason="promotion_send_succeeded", action_receipt=action_receipt,
            )
            self._record_promotion_outcome(
                stream, event_id=event_id, outcome="executed", reason=send_reason,
                target=normalized_target,
            )
            self._mark_recent_raid_shoutout_done(stream, normalized_target)
            profile_manager = getattr(self, "promotion_profile_manager", None)
            if profile_manager is not None:
                try:
                    profile_manager.learn_after_success(
                        twitch_user_id=(
                            self._twitch_user_ids_by_login.get(normalized_target.casefold())
                            or f"login:{normalized_target.lower()}"
                        ),
                        login=normalized_target,
                        display_name=normalized_target,
                        owner_command=str((plan.slots or {}).get("owner_command_text") or ""),
                        stream_session_id=getattr(stream, "active_stream_session_id", None) or "offline",
                        known_aliases=[
                            str((plan.slots or {}).get("target_raw") or ""),
                            str(plan.target or ""),
                        ],
                        source_promotion_event=str(getattr(persistent_event, "id", "") or event_id),
                    )
                except Exception as exc:
                    print(f"[HEBE][PROMOTION_PROFILE] update_failed={type(exc).__name__}", flush=True)
            if stream is not None:
                stream.last_promo_rejected_reason = ""
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
                fallback_text=f"Promo hecha para {normalized_target}.",
                requires_model_response=False,
                metadata={
                    "action_plan": plan.as_log_dict(),
                    "action_receipt": action_receipt,
                    "message_goal": f"Tell Leo that the promo/shoutout for {normalized_target} was sent.",
                },
            )
        if send_reason in {"blocked_bot_user", "own_channel", "invalid_target"}:
            fallback = "No le hago SO a ese usuario, Leo. Huele a bot o a bucle infernal."
        elif send_reason == "cooldown_active":
            fallback = f"Ya hice SO a {normalized_target} hace nada, Leo. Evito el spam."
        else:
            fallback = f"No he podido hacer el SO a {normalized_target or plan.target}."
        self._record_promotion_outcome(
            stream,
            event_id=event_id,
            outcome="blocked" if send_reason in {"blocked_bot_user", "own_channel", "invalid_target", "cooldown_active"} else "error",
            reason=send_reason,
            target=normalized_target or plan.target or "",
        )
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

    def _record_promotion_outcome(
        self,
        stream,
        *,
        event_id: str,
        outcome: str,
        reason: str,
        target: str = "",
    ) -> dict:
        record = {
            "event_id": str(event_id or ""),
            "outcome": str(outcome or "error"),
            "reason": str(reason or ""),
            "target": str(target or ""),
            "ts": time.time(),
        }
        if stream is not None:
            stream.last_promotion_outcome = record
        print(
            f"[HEBE][PROMOTION_OUTCOME] outcome={record['outcome']} reason={record['reason']} target={record['target']}",
            flush=True,
        )
        return record

    def _resolve_recent_raid_shoutout_plan(self, plan: ActionPlan, stream) -> None:
        if plan.action_type != "twitch_shoutout" or stream is None:
            return
        raw_target = str((plan.slots or {}).get("target_raw") or (plan.slots or {}).get("target_text") or "").strip()
        normalized_target = self._normalize_text(raw_target)
        implicit_markers = {
            "",
            "ese",
            "esa",
            "raider",
            "el raider",
            "la raider",
            "al raider",
            "a ese",
            "a esa",
            "ultimo raider",
            "al ultimo raider",
            "last raider",
        }
        command_norm = self._normalize_text(str((plan.slots or {}).get("raw_promotion_text") or ""))
        no_explicit_target_command = command_norm in {"hazle promo", "dale promo", "haz promo", "tira promo", "promo"} or normalized_target in implicit_markers
        if not no_explicit_target_command:
            return
        context = self._stream_recent_raid_context(stream)
        if not context:
            return
        target = self._normalize_shoutout_target(context.get("user_login") or context.get("display_name") or "")
        if not target:
            return
        plan.status = "complete"
        plan.target = target
        plan.command = self._build_shoutout_command_preview(target)
        plan.confidence = max(float(plan.confidence or 0.0), 0.99)
        plan.reason = "recent_raid_context"
        plan.missing_slots = []
        plan.candidates = [target]
        slots = dict(plan.slots or {})
        slots["target_raw"] = raw_target or "recent_raider"
        slots["target_text"] = target
        slots["resolved_username"] = target
        slots["recent_raid_context"] = context
        plan.slots = slots

    def _create_promotion_pending(self, plan: ActionPlan, *, fallback: str) -> None:
        print("[HEBE][PENDING_CREATION_GUARD] allowed=true reason=promotion_target_clarification", flush=True)
        pending = self._open_pending_conversation(
            kind="promotion_target_clarification",
            expected_reply_type="twitch_username_or_viewer_alias",
            allowed_sources=["stt_voice", "ui"],
            capability_needed="twitch.promotion",
            opened_by_speech_act="clarification_question",
            explicit_question_asked=True,
            can_accept_no_wake_followup=True,
            ttl_seconds=60,
            max_attempts=1,
            compatible_intents=["promotion_target_answer"],
            incompatible_intents=["stream_monologue", "low_confidence_target"],
            target_raw=(plan.slots or {}).get("target_raw") or (plan.slots or {}).get("target_text") or "",
            candidates=list(plan.candidates or []),
            reason=plan.reason,
            fallback_text=fallback,
            starts_after_tts_end=float(getattr(self, "_tts_until", 0.0) or 0.0),
            capture_window_seconds=12,
            owner_voice_only=True,
            wake_not_required=True,
            minimum_target_confidence=0.78,
            actual_tts_completion_time=0.0,
            buffered_answers=[],
            creation_reason="promotion_target_clarification",
        )
        print(
            f"[HEBE][PROMOTION_PENDING] created id={pending.id} reason={plan.reason} candidates={pending.domain_payload.get('candidates')!r}",
            flush=True,
        )

    def _resolve_pending_promotion_target(self, raw_command: str, normalized: str, stream) -> CommandResult | None:
        pending = self._active_current_conversation(latest=True, expire=False)
        if pending is None or pending.topic != "promotion_target_clarification":
            return None
        if pending.expires_at <= time.time():
            print(f"[HEBE][PROMOTION_PENDING] expired id={pending.id}", flush=True)
            print(f"[HEBE][PENDING_EXPIRED] kind=promotion_target_clarification id={pending.id}", flush=True)
            self._close_current_conversation(reason="expired", conversation=pending)
            return None
        domain = pending.domain_payload

        raw_answer = str(raw_command or normalized or "").strip()
        answer = self._normalize_promotion_pending_target_text(raw_answer)
        answer_norm = self._normalize_text(answer)
        print(
            f"[HEBE][PROMOTION_PENDING_ANSWER] raw={raw_answer!r} normalized_target={answer!r} source=trusted_manual accepted=probe",
            flush=True,
        )
        if answer_norm in {"si", "sí", "ese", "si ese", "sí ese", "ese mismo"}:
            candidates = list(domain.get("candidates") or [])
            answer = candidates[0] if candidates else str(domain.get("target_raw") or "")
        if answer_norm in {"el nuevo", "el de antes", "el que acaba de hablar"}:
            answer = ""

        pending_candidates = [str(candidate).strip() for candidate in (domain.get("candidates") or []) if str(candidate).strip()]
        if answer_norm in {"si", "sÃ­", "ese", "si ese", "sÃ­ ese", "ese mismo"} and len(pending_candidates) > 1:
            answer = ""
        if not answer and answer_norm in {"el nuevo", "el de antes", "el que acaba de hablar"}:
            contextual_target = self._promotion_contextual_target(answer_norm, pending_candidates)
            if contextual_target:
                answer = contextual_target

        planner = self._get_stream_action_planner()
        target, confidence, candidates, reason = planner._resolve_target(answer)
        print(
            "[HEBE][PROMOTION_RESOLVE] "
            f"target_phrase={answer!r} candidates={candidates!r} selected={target!r} confidence={confidence:.3f} source={reason}",
            flush=True,
        )
        if not target or confidence < 0.78 or reason in {"ambiguous_target", "medium_confidence", "unverified_username"}:
            pending = self._update_current_conversation(
                pending, domain_updates={"candidates": candidates, "reason": reason},
            )
            self._increment_conversation_attempt(pending, reason=reason or "not_found")
            print(f"[HEBE][PROMOTION_CLARIFY] reason={reason or 'not_found'} candidates={candidates!r}", flush=True)
            return CommandResult(
                action_type="twitch_shoutout_clarify",
                success=False,
                user_visible_summary="Promotion target still unclear.",
                state_changes={},
                constraints=["Ask one concise follow-up question.", "Do not claim the shoutout was sent."],
                fallback_text=(f"¿Te refieres a {candidates[0]}?" if candidates else "No ubico a ese usuario, Leo. Dame el login o alias."),
                requires_model_response=False,
                metadata={"conversation": pending.to_dict(), "candidates": candidates, "confidence": confidence, "reason": reason},
            )

        print(f"[HEBE][PENDING_CONSUMED] kind=promotion_target_clarification id={pending.id} reason=resolved_target", flush=True)
        self._close_current_conversation(reason="consumed", conversation=pending)
        print(f"[HEBE][PROMOTION_PENDING] resolved id={pending.id} target={target}", flush=True)
        print(
            f"[HEBE][PROMOTION_PENDING_ANSWER] raw={raw_answer!r} normalized_target={answer!r} source=trusted_manual accepted=true",
            flush=True,
        )
        plan = ActionPlan(
            action_type="twitch_shoutout",
            status="complete",
            confidence=confidence,
            target=target,
            command=self._build_shoutout_command_preview(target),
            requires_stream=True,
            reason="pending_resolved",
            candidates=candidates or [target],
            slots={"target_raw": answer, "resolved_username": target, "pending_id": pending.id},
        )
        return self._execute_twitch_shoutout_plan(plan, stream)

    def _promotion_contextual_target(self, answer_norm: str, pending_candidates: list[str]) -> str | None:
        if len(pending_candidates) == 1:
            return pending_candidates[0]
        stream = self._get_stream_state()
        twitch = getattr(self.runtime, "twitch", None)
        resolver = getattr(twitch, "target_resolver", None)
        event_memory = getattr(resolver, "event_memory", None)
        if answer_norm == "el nuevo":
            for attr in ("last_follow_username", "last_sub_username"):
                username = str(getattr(event_memory, attr, "") or "").strip() if event_memory is not None else ""
                if username:
                    return username
        if answer_norm in {"el de antes", "el que acaba de hablar"} and stream is not None:
            messages = list(getattr(stream, "recent_chat_messages", []) or [])
            for message in reversed(messages):
                username = str((message or {}).get("username") or (message or {}).get("display_name") or "").strip()
                if username:
                    return username
        if stream is not None:
            raid = getattr(stream, "last_raid_event", None) or {}
            username = str(raid.get("user_login") or raid.get("display_name") or "").strip()
            if username:
                return username
        return None

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
            primer = session_primer.build_stream_session_primer(
                game=game,dt=now_dt,canonical_run_state=self._canonical_run_state_for_game(game),
            )
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
            primer = session_primer.build_stream_session_primer(
                game=note_result.get("game"),dt=now_dt,
                canonical_run_state=self._canonical_run_state_for_game(note_result.get("game")),
            )
            session_primer.apply_primer_to_stream(stream, primer)
            return command_result(
                "game_session_note_saved",
                self._format_stream_primer_reply(primer),
                state_changes={"game_session": note_result, "primer": primer.to_dict()},
                message_goal="Confirm the game session note was saved and summarize the next stream primer.",
            )

        if schedule_query:
            primer = session_primer.build_stream_session_primer(
                dt=now_dt,canonical_run_state=self._canonical_run_state_for_game(None),
            )
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
                canonical_run_state=self._canonical_run_state_for_game(
                    self._title_case_game(explicit_prepare_game) if explicit_prepare_game else None
                ),
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
                canonical_run_state=self._canonical_run_state_for_game(
                    self._title_case_game(explicit_title_game) if explicit_title_game else None
                ),
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

    def _canonical_run_state_for_game(self, game: str | None) -> dict:
        service=getattr(self,"game_run_service",None)
        if service is None:return {}
        stream=self._get_stream_state();target=str(game or getattr(stream,"current_game",None) or getattr(stream,"current_category",None) or "").strip()
        if not target:return {}
        identity=service.repository.resolve_identity(target)
        run_id=str(getattr(stream,"active_game_run_id","") or "") if stream is not None else ""
        run=service.repository.get_run(run_id) if run_id else None
        if run is None or run.game_id!=identity.game_id:
            run=next(iter(service.repository.list_runs(
                game_id=identity.game_id,owner_id="leo",statuses=("ACTIVE","PAUSED"),
            )),None)
        return service.state(run.id) if run is not None else {}

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
        service=getattr(self,"game_run_service",None)
        if service is None:return None
        stream=self._get_stream_state();session_id=str(getattr(stream,"active_stream_session_id","") or f"manual:{now_dt.date().isoformat()}")
        resolution=service.resolve(
            game=game,stream_session_id=session_id,source_event_id=f"manual_game_note:{uuid.uuid4().hex}",
            run_kind=str(getattr(stream,"current_playthrough_type",None) or "unknown") if stream is not None else "unknown",
        )
        if resolution.active_run is None:return None
        updates={}
        if end_summary or start_summary:updates["last_confirmed_progress"]=end_summary or start_summary
        if next_time_plan or current_objective:updates["current_objective"]=next_time_plan or current_objective
        event_id=f"manual_game_note:{uuid.uuid4().hex}"
        result=service.update_state(
            resolution.active_run.id,updates=updates,provenance="manual_command",confidence=1.0,
            evidence=EvidenceRef(
                source_event_id=event_id,source_record_type="owner_command",source_record_id=event_id,
                observed_at=now_dt.timestamp(),extractor="manual_game_note",extractor_version="v1",
                literal_span={"text":text},
            ),
        )
        if stream is not None:
            stream.active_game_run_id=resolution.active_run.id;self._project_canonical_game_run(stream,resolution.active_run.id)
        return {"game":resolution.game_identity.canonical_name,"game_run_id":resolution.active_run.id,**result["state"]}

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
                    self._persist_canonical_run_state(stream,{"current_objective":value},source="manual")
                    set_updated("manual")
                    return f"Objetivo actual guardado: {value}."

        progress_prefixes = ("progreso actual:", "progreso actual ")
        for prefix in progress_prefixes:
            if lower.startswith(prefix):
                value = raw_command[len(prefix):].strip()
                if value:
                    stream.current_run_phase = value
                    self._persist_canonical_run_state(stream,{"last_confirmed_progress":value},source="manual")
                    set_updated("manual")
                    return f"Progreso actual guardado: {value}."

        location_prefixes = ("estamos en ",)
        for prefix in location_prefixes:
            if lower.startswith(prefix):
                value = raw_command[len(prefix):].strip()
                if value:
                    stream.current_run_location = value
                    self._persist_canonical_run_state(stream,{"current_location":value},source="manual")
                    set_updated("manual")
                    return f"Ubicacion actual guardada: {value}."

        passed_prefixes = ("ya hemos pasado ", "hemos pasado ")
        for prefix in passed_prefixes:
            if lower.startswith(prefix):
                marker = raw_command[len(prefix):].strip()
                if marker:
                    self._add_completed_marker(stream, marker)
                    markers=list(getattr(stream,"completed_run_markers",[]) or [])
                    self._persist_canonical_run_state(stream,{"progress_markers":markers[-20:]},source="manual")
                    set_updated("manual")
                    return f"Marcador completado guardado: {marker}."

        forget_prefixes = ("olvida ",)
        if lower.startswith("olvida ") and lower.endswith(" como objetivo actual"):
            marker = raw_command[len("olvida "): -len(" como objetivo actual")].strip()
            if marker:
                self._add_completed_marker(stream, marker)
                if self._same_marker(getattr(stream, "current_run_objective", ""), marker):
                    stream.current_run_objective = None
                    self._clear_canonical_run_state(stream,("current_objective",))
                set_updated("manual")
                return f"Dejo de tratar {marker} como objetivo actual."

        if normalized in {"limpia contexto de partida"}:
            self._clear_canonical_run_state(stream,("current_objective","current_location","last_confirmed_progress","progress_markers"))
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
            self._open_pending_conversation(
                kind="tts_scope", expected_reply_type="clarification",
                capability_needed="audio.tts_control", can_accept_no_wake_followup=True,
                ttl_seconds=60, creation_reason="tts_scope_question",
            )
            print("[HEBE][INTENT] voice command handled before reminder parser", flush=True)
            print("[HEBE][INTENT] tts_scope conversation opened", flush=True)
            self._emit_audio_status()
            return CommandResult(
                action_type="tts_enabled",
                success=True,
                user_visible_summary="Global/local TTS enabled; asking Leo whether voice should stay local or also apply to stream.",
                state_changes={"tts_enabled": True, "tts_scope_conversation_active": True},
                constraints=["Ask only whether scope is local or stream.", "Do not imply stream TTS is enabled yet."],
                suggested_tone="short Hebe voice, useful and warm",
                fallback_text="Voz activada. ¿La quieres solo aquí/local o también para el stream?",
                requires_model_response=True,
                metadata={"message_goal": "Confirm voice is enabled and ask whether Leo wants local only or also stream scope."},
            )
        if intent == "global_off":
            self.runtime.state.tts_enabled = False
            conversation = self._active_current_conversation(latest=True)
            if conversation is not None and conversation.topic == "tts_scope":
                self._close_current_conversation(reason="owner_cancel", conversation=conversation)
            print("[HEBE][INTENT] voice command handled before reminder parser", flush=True)
            self._emit_audio_status()
            return CommandResult(
                action_type="tts_disabled",
                success=True,
                user_visible_summary="Global TTS disabled; Hebe will answer in text.",
                state_changes={"tts_enabled": False, "tts_scope_conversation_active": False},
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
        capability = "pending.cancel" if self._is_cancel_pending_request(normalized) else "audio.tts_control"
        if not self._manual_handler_guard(
            handler="pending", cognitive_decision=cognitive_decision,
            capabilities={capability}, source=source,
        ):
            return None
        if self._is_cancel_pending_request(normalized):
            conversation = self._active_current_conversation(latest=True)
            if conversation is not None:
                self._close_current_conversation(reason="owner_cancel", conversation=conversation)
                print("[HEBE][INTENT] reminder pending cancelled", flush=True)
                return CommandResult(
                    action_type="pending_reminder_cancelled",
                    success=True,
                    user_visible_summary="Pending reminder or appointment clarification was cancelled.",
                    state_changes={"current_conversation": None},
                    constraints=["Do not ask for clarification."],
                    fallback_text="Vale, no guardo nada.",
                    requires_model_response=True,
                    metadata={"message_goal": "Tell Leo the pending reminder clarification was cancelled."},
                )

        pending_tts = self._active_current_conversation(source=source or "")
        if pending_tts is None or pending_tts.topic != "tts_scope":
            return None

        print("[HEBE][PENDING] active=tts_scope", flush=True)
        print("[HEBE][INTENT] tts_scope conversation active", flush=True)
        if self._is_explicit_command_while_pending(normalized):
            self._close_current_conversation(reason="new_owner_command_interrupted", conversation=pending_tts)
            print("[HEBE][PENDING] new explicit command detected; closing tts_scope", flush=True)
            return None

        scope = self._parse_tts_scope_followup(normalized)
        if scope == "local":
            return self._resolve_tts_scope_local(pending_tts)
        if scope == "stream":
            return self._resolve_tts_scope_stream(pending_tts)

        if not pending_tts.domain_payload.get("unclear_asked"):
            self._update_current_conversation(pending_tts, domain_updates={"unclear_asked": True})
            return CommandResult(
                action_type="tts_scope_clarify",
                success=False,
                user_visible_summary="TTS scope follow-up was unclear; ask whether local or stream.",
                state_changes={"tts_scope_conversation_active": True},
                constraints=["Ask one concise clarification question."],
                fallback_text="No te he entendido, Leo. ¿Local o también para stream?",
                requires_model_response=False,
                metadata={"message_goal": "Ask Leo whether voice should be local only or also for stream."},
            )
        return self._resolve_tts_scope_local(pending_tts, defaulted=True)

    def _is_cancel_pending_request(self, normalized: str) -> bool:
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

    def _resolve_tts_scope_local(self, conversation: CurrentConversation, *, defaulted: bool = False) -> CommandResult:
        stream = self._get_stream_state()
        policies = getattr(stream, "policies", None) if stream else None
        self.runtime.state.tts_enabled = True
        if policies is not None:
            policies.allow_tts_idle_prompts = False
        if stream is not None:
            stream.stream_output_mode = "ui_only"
        self._close_current_conversation(reason="resolved", conversation=conversation)
        print("[HEBE][INTENT] resolved tts_scope=local", flush=True)
        return CommandResult(
            action_type="tts_scope_resolved",
            success=True,
            user_visible_summary="Voice is enabled locally only; stream remains text-only unless Leo asks otherwise.",
            state_changes={"tts_enabled": True, "stream_idle_tts": False, "stream_output_mode": "ui_only", "tts_scope_conversation_active": False, "defaulted": defaulted},
            constraints=["Do not ask for more clarification.", "Do not claim stream voice is enabled."],
            fallback_text="Perfecto, voz activada solo aquí. En stream seguiré en texto salvo que me digas lo contrario.",
            requires_model_response=True,
            metadata={"scope": "local", "message_goal": "Confirm voice is enabled locally only, and stream remains text-only unless Leo asks otherwise."},
        )

    def _resolve_tts_scope_stream(self, conversation: CurrentConversation) -> CommandResult:
        stream = self._get_stream_state()
        policies = getattr(stream, "policies", None) if stream else None
        self.runtime.state.tts_enabled = True
        if policies is not None:
            policies.allow_tts_replies = True
            policies.allow_tts_event_replies = True
            policies.allow_tts_raid_thanks = True
        if stream is not None:
            stream.stream_output_mode = "tts_enabled"
        self._close_current_conversation(reason="resolved", conversation=conversation)
        print("[HEBE][INTENT] resolved tts_scope=stream", flush=True)
        return CommandResult(
            action_type="tts_scope_resolved",
            success=True,
            user_visible_summary="Voice is enabled locally and for stream event replies; idle spontaneity remains text-only unless Leo asks otherwise.",
            state_changes={"tts_enabled": True, "stream_replies_tts": True, "stream_event_tts": True, "stream_raid_tts": True, "stream_output_mode": "tts_enabled", "tts_scope_conversation_active": False},
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
            effective_audio = self._effective_stream_audio_state("stream_reply")
            emit(
                "status",
                {
                    "tts_enabled": bool(getattr(self.runtime.state, "tts_enabled", False)),
                    "stream_tts_enabled": bool(getattr(policies, "allow_tts_replies", False)),
                    "stream_tts_effective": effective_audio.actual_can_speak,
                    "effective_stream_audio_state": effective_audio.to_dict(),
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

    def _stream_message_is_emote_only(self, text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        words = [part for part in re.split(r"\s+", value) if part]
        if len(words) > 4:
            return False
        allowed = {"lol", "lmao", "xd", "jaja", "haha", "gg", "pog", "kekw", "omegalul", "clap", "hearts"}
        for word in words:
            clean = re.sub(r"[^A-Za-z0-9_:-]", "", word).lower()
            if not clean:
                continue
            if re.match(r"^[:;x=8][-']?[)dpo(/|\\]+$", clean):
                continue
            if clean in allowed:
                continue
            if clean in {"kappa", "pjsalt", "lul", "feelsgoodman", "feelsbadman"}:
                continue
            if re.match(r"^[a-z][a-z0-9_]{2,24}$", clean) and clean.upper() == word:
                continue
            return False
        return True

    def _viewer_talks_about_hebe(self, text: str) -> bool:
        normalized = self._normalize_guard_text(text)
        if "hebe" not in normalized and "hebenifelheim" not in normalized:
            return False
        return bool(re.search(
            r"\b(?:hebe|hebenifelheim)\b\s+(?:esta|es|parece|suena|anda|se\s+queda|se\s+ve|va)\b|"
            r"\b(?:callad[ao]|muda|despierta|dormida|graciosa|seca|afilada|presente)\b.*\b(?:hebe|hebenifelheim)\b|"
            r"\b(?:hebe|hebenifelheim)\b.*\b(?:callad[ao]|muda|despierta|dormida|graciosa|seca|afilada|presente|pasa\s+del\s+chat)\b|"
            r"\b(?:esos\s+dias|esos\s+dias|no\s+te\s+dejan\s+opinar|tampoco\s+opina)\b.*\b(?:hebe|hebenifelheim)\b",
            normalized,
        ))

    def _viewer_direct_open_prompt_to_hebe(self, text: str) -> bool:
        normalized = self._normalize_guard_text(text)
        if not self._message_mentions_hebe(text):
            return False
        stripped = re.sub(r"^(?:@?hebe|@?hebenifelheim|ebe|eve|jebe|heve)\b[:,\s-]*", "", normalized).strip()
        return bool(
            re.search(r"\b(?:que\s+opinas|di\s+algo|cuenta\s+algo|que\s+dices|opina|reacciona|habla)\b", stripped)
            or stripped in {"que opinas", "di algo", "cuenta algo", "que dices"}
        )

    def _viewer_to_leo_social_chat(self, text: str) -> str | None:
        normalized = self._normalize_guard_text(text)
        if "leo" not in normalized or self._message_mentions_hebe(text):
            return None
        if re.search(r"\b(?:dile|avisa|cuenta|pasa|manda|hazle|envia)\w*\b", normalized):
            return None
        if re.search(r"\b(?:guap|hermos|bonit|amor|cari|precioso|atractiv|te\s+quiero)\w*\b", normalized):
            return "viewer_to_leo_affection"
        if re.search(r"\b(?:hola+|buenas|hey|ey|saludos)\b", normalized):
            return "normal_greeting_to_leo"
        return None

    def _classify_twitch_viewer_message(self, text: str, *, payload: dict | None = None) -> str:
        payload = payload or {}
        normalized = self._normalize_guard_text(text)
        if not normalized:
            return "meme_or_emote"
        reply_to_hebe = bool(payload.get("reply_to_hebe_message"))
        mentions_hebe = bool(payload.get("mentions_hebe") or payload.get("direct_address_to_hebe") or self._message_mentions_hebe(text))
        words = normalized.split()
        category = ""
        priority = 0
        if re.search(r"\b(?:dile|avisa|cuenta|pasa|manda)\w*\s+a\s+leo\b", normalized):
            category, priority = "viewer_relay_attempt", 5
        elif re.search(r"\b(?:shoutout|promo|so)\b", normalized):
            category, priority = "promo_request_from_viewer", 4
        elif re.search(r"\b(?:haz|pon|abre|cambia|apaga|enciende)\b", normalized):
            category, priority = "viewer_command_attempt", 3
        elif reply_to_hebe:
            category, priority = "reply_to_hebe_message", 6
        elif self._viewer_direct_open_prompt_to_hebe(text):
            category, priority = "direct_open_prompt_to_hebe", 6
        elif mentions_hebe and not self._viewer_talks_about_hebe(text):
            policy = payload.get("_viewer_policy_decision")
            if isinstance(policy, PolicyDecision) and policy.allow_reply and not policy.allow_llm:
                category, priority = "viewer_boundary_needed", 3
            if not category:
                if len(words) <= 3 and any(token in {"xd", "lol", "lmao", "kekw", "jaja", "haha"} for token in words):
                    category = "direct_hebe_banter"
                elif "?" in str(text or "") or re.search(r"\b(?:oye|resp|responde|dime|opina|habla|eres|estas|boot|bot)\b", normalized):
                    category = "direct_hebe_prompt"
                else:
                    category = "direct_hebe_banter"
                priority = 6
        elif self._viewer_talks_about_hebe(text):
            category, priority = "viewer_talks_about_hebe", 7
        else:
            leo_social = self._viewer_to_leo_social_chat(text)
            if leo_social:
                category, priority = leo_social, 10
        if category:
            print(f"[HEBE][TWITCH_CLASSIFY] category={category} priority={priority} reason=priority_match", flush=True)
            return category
        if self._stream_message_is_emote_only(text):
            print("[HEBE][TWITCH_CLASSIFY] category=meme_or_emote priority=11 reason=emote_only", flush=True)
            return "meme_or_emote"
        if len(words) <= 5 and len(set(words)) <= 2 and len(words) >= 3:
            return "repeated_spam"
        policy = payload.get("_viewer_policy_decision")
        if isinstance(policy, PolicyDecision) and policy.allow_reply and not policy.allow_llm:
            return "viewer_boundary_needed"
        if re.search(r"\b(?:pista|tip|consejo|ruta|camino|templo|cueva|mazmorra|boss|jefe|npc|objeto|cofre|mision|quest|zona|build|arma|habilidad)\b", normalized):
            return "high_value_game_tip"
        if "?" in str(text or "") or re.search(r"\b(?:que|quien|cuando|donde|como|por que|porque|cuanto|opinas|sabes|eres|estas)\b", normalized):
            if re.search(r"\b(?:juego|boss|jefe|ruta|build|arma|habilidad|mision|quest|stream|directo|raid|sub|follow)\b", normalized):
                return "high_value_question"
            return "normal_no_mention_chat"
        if re.search(r"\b(?:raid|raideo|raidear|follow|sub|resub|bits|hype|clip|victoria|derrota|final|empezamos|terminamos)\b", normalized):
            return "stream_milestone_comment"
        if re.search(r"\b(?:me\s+voy|me\s+piro|hasta\s+(?:luego|otra|manana)|buenas\s+noches|voy\s+a\s+por|vuelvo\s+luego|nos\s+vemos)\b", normalized):
            return "viewer_goodbye"
        if re.search(r"\b(?:ya\s+volvi|he\s+vuelto|vuelvo|estoy\s+de\s+vuelta|back)\b", normalized):
            return "viewer_returning"
        if len(words) <= 3 and self._message_mentions_hebe(text):
            if any(token in words for token in {"hola", "buenas", "hey", "ey"}):
                return "simple_social_greeting"
            return "low_value_banter"
        meme_tokens = {"xd", "lol", "lmao", "kekw", "pog", "jaja", "haha"}
        if len(words) <= 5 and any(token in meme_tokens for token in words):
            return "meme_or_emote"
        category = "normal_no_mention_chat"
        if mentions_hebe:
            print("[HEBE][TWITCH_CLASSIFY_INVARIANT] corrected=true old=normal_no_mention_chat new=direct_hebe_banter", flush=True)
            category = "direct_hebe_banter"
        return category

    def _canonical_twitch_message_category(self, category: str) -> str:
        return {
            "viewer_talks_about_hebe": "talks_about_hebe",
            "direct_open_prompt_to_hebe": "direct_open_prompt_to_hebe",
            "direct_hebe_prompt": "direct_hebe_prompt",
            "direct_hebe_banter": "direct_hebe_banter",
            "reply_to_hebe_message": "reply_to_hebe_message",
            "viewer_relay_attempt": "viewer_proxy_request",
            "viewer_command_attempt": "viewer_command_attempt",
            "promo_request_from_viewer": "promo_request_from_viewer",
            "viewer_boundary_needed": "viewer_boundary_needed",
            "viewer_to_leo_affection": "viewer_to_leo_affection",
            "normal_greeting_to_leo": "normal_greeting_to_leo",
            "high_value_question": "high_value_question",
            "high_value_game_tip": "high_value_game_tip",
            "stream_milestone_comment": "stream_milestone_comment",
            "viewer_goodbye": "viewer_goodbye",
            "viewer_returning": "viewer_returning",
            "normal_no_mention_chat": "normal_no_mention_chat",
            "simple_social_greeting": "simple_social_greeting",
            "low_value_banter": "low_value_banter",
            "meme_or_emote": "meme_or_emote",
            "repeated_meme": "repeated_meme",
            "repeated_spam": "repeated_spam",
            "followup_to_hebe_question": "followup_to_hebe_question",
        }.get(str(category or ""), "unclear")

    def _suggested_twitch_speech_act(self, category: str) -> str:
        if category in {"viewer_boundary_needed", "viewer_relay_attempt", "viewer_command_attempt", "promo_request_from_viewer"}:
            return "viewer_boundary"
        if category == "followup_to_hebe_question":
            return "clarification_question"
        if category == "high_value_question":
            return "direct_answer"
        if category in {"direct_hebe_prompt", "reply_to_hebe_message"}:
            return "direct_answer"
        if category == "high_value_game_tip":
            return "game_guidance_clarification"
        if category == "direct_open_prompt_to_hebe":
            return "stream_banter"
        if category in {"viewer_talks_about_hebe", "direct_hebe_banter", "simple_social_greeting", "viewer_goodbye", "viewer_returning", "stream_milestone_comment"}:
            return "hebe_banter"
        if category in {"low_value_banter", "meme_or_emote", "repeated_meme", "repeated_spam", "normal_no_mention_chat", "viewer_to_leo_affection", "normal_greeting_to_leo"}:
            return "low_value_banter"
        return "stream_banter"

    def _speech_act_allows_followup_question(self, speech_act: str) -> bool:
        return str(speech_act or "") in {"clarification_question", "game_guidance_clarification"}

    def _perceive_twitch_viewer_event(self, *, payload: dict, event_type: str | None, category: str, stream) -> PerceivedEvent:
        raw = str(payload.get("message_text") or payload.get("text") or "")
        username = str(payload.get("user_login") or payload.get("username") or payload.get("display_name") or "viewer")
        normalized = self._normalize_guard_text(raw)
        event_id = str(payload.get("event_id") or payload.get("message_id") or f"evt_{uuid.uuid4().hex}")
        return PerceivedEvent(
            event_id=event_id,
            source="twitch",
            source_type="twitch_chat" if event_type == "twitch_chat_react" else "twitch_event",
            speaker=str(payload.get("display_name") or username),
            speaker_type="viewer",
            raw_text=raw,
            normalized_text=normalized,
            output_context="stream",
            stream_live=bool(getattr(stream, "is_live", False)) if stream is not None else False,
            current_game=str(getattr(stream, "current_game", "") or "") if stream is not None else "",
            current_activity=str(getattr(stream, "current_activity", "") or "") if stream is not None else "",
            direct_address_to_hebe=bool(payload.get("direct_address_to_hebe") or self._message_mentions_hebe(raw)),
            talks_about_hebe=category in {"viewer_talks_about_hebe", "direct_open_prompt_to_hebe", "direct_hebe_banter"},
            mentions_hebe=bool(payload.get("mentions_hebe") or self._message_mentions_hebe(raw)),
            talks_to_leo=bool(re.search(r"\bleo\b", normalized)),
            is_emote_only=category == "meme_or_emote",
            is_low_value_chat=category in {"low_value_banter", "meme_or_emote", "repeated_meme", "repeated_spam", "normal_no_mention_chat"},
            confidence=0.9,
            twitch_metadata={
                "event_type": event_type or "",
                "user_login": username,
                "category": category,
                "message_id": payload.get("message_id") or "",
                "reply_to_hebe_message": bool(payload.get("reply_to_hebe_message")),
                "direct_priority_reason": payload.get("direct_priority_reason") or "",
            },
        )

    def _understand_twitch_viewer_event(self, *, category: str, perception: PerceivedEvent) -> UnderstandingResult:
        intent_map = {
            "viewer_talks_about_hebe": "viewer_talks_about_hebe",
            "direct_open_prompt_to_hebe": "direct_open_prompt_to_hebe",
            "direct_hebe_prompt": "direct_hebe_prompt",
            "direct_hebe_banter": "direct_hebe_banter",
            "reply_to_hebe_message": "reply_to_hebe_message",
            "viewer_boundary_needed": "viewer_boundary_needed",
            "viewer_relay_attempt": "viewer_proxy_request",
            "viewer_command_attempt": "viewer_command_attempt",
            "promo_request_from_viewer": "viewer_cannot_request_promo",
            "high_value_question": "viewer_direct_question_to_hebe",
            "high_value_game_tip": "high_value_game_tip",
            "stream_milestone_comment": "stream_milestone_comment",
            "viewer_goodbye": "viewer_goodbye",
            "viewer_returning": "viewer_returning",
            "normal_no_mention_chat": "normal_no_mention_chat",
            "viewer_to_leo_affection": "viewer_to_leo_affection",
            "normal_greeting_to_leo": "normal_greeting_to_leo",
            "simple_social_greeting": "viewer_banter_about_hebe",
            "low_value_banter": "viewer_low_value_banter",
            "meme_or_emote": "viewer_emote_only",
            "repeated_meme": "viewer_emote_only",
            "repeated_spam": "viewer_emote_only",
        }
        pressure = {
            "viewer_talks_about_hebe": 0.56,
            "direct_open_prompt_to_hebe": 0.58,
            "direct_hebe_prompt": 0.70,
            "direct_hebe_banter": 0.64,
            "reply_to_hebe_message": 0.72,
            "viewer_boundary_needed": 0.88,
            "viewer_relay_attempt": 0.82,
            "viewer_command_attempt": 0.72,
            "promo_request_from_viewer": 0.76,
            "high_value_question": 0.64,
            "high_value_game_tip": 0.74,
            "stream_milestone_comment": 0.54,
            "viewer_goodbye": 0.56,
            "viewer_returning": 0.50,
            "normal_no_mention_chat": 0.16,
            "viewer_to_leo_affection": 0.0,
            "normal_greeting_to_leo": 0.0,
            "simple_social_greeting": 0.35,
            "low_value_banter": 0.12,
            "meme_or_emote": 0.02,
            "repeated_meme": 0.01,
            "repeated_spam": 0.01,
        }.get(category, 0.2)
        return UnderstandingResult(
            intent=intent_map.get(category, "viewer_low_value_banter"),
            confidence=0.86,
            authority="viewer",
            reply_pressure=pressure,
            requires_policy=True,
            possible_capability="twitch.reply",
            social_context=category,
            risk_flags=["viewer_authority"] if category in {"viewer_command_attempt", "viewer_relay_attempt", "promo_request_from_viewer"} else [],
        )

    def _policy_contract_for_twitch_category(self, *, category: str) -> PolicyContract:
        if category in {"viewer_boundary_needed", "viewer_relay_attempt", "viewer_command_attempt", "promo_request_from_viewer"}:
            return PolicyContract(
                result="redirect",
                reason=category,
                blocked_behavior="viewer_control_or_proxy",
                forbidden_actions=["owner_control", "viewer_proxy_message"],
                capability_blocked=["owner_action"],
                authority_constraints=["viewer_familiarity_does_not_grant_authority"],
                boundary_required=True,
                risk_level="medium",
            )
        return PolicyContract(
            result="allow",
            reason="viewer_interaction",
            allowed_action="respond_directly",
            capability_allowed=["twitch.reply"],
            authority_constraints=["viewer_can_interact_not_command"],
            risk_level="low",
        )

    def _log_presence_decision(self, decision: dict) -> None:
        intervention = dict(decision.get("intervention") or decision)
        budget = dict(intervention.get("output_budget_result") or {})
        print(
            "[HEBE][SOCIAL_BUDGET] "
            f"allowed={str(bool(budget.get('allowed', True))).lower()} reason={budget.get('reason', 'not_checked')}",
            flush=True,
        )
        print(
            "[HEBE][PRESENCE_ENGINE] "
            "source=twitch_chat "
            f"should_intervene={str(bool(intervention.get('should_intervene'))).lower()} "
            f"level={intervention.get('intervention_level')} "
            f"social_value={float(intervention.get('social_value_score') or 0.0):.2f} "
            f"interruption_cost={float(intervention.get('interruption_cost') or 0.0):.2f} "
            f"reason={intervention.get('reason')}",
            flush=True,
        )
        print(
            "[HEBE][PRESENCE_FACTORS] "
            f"social_value={float(intervention.get('social_value_score') or 0.0):.2f} "
            f"interruption_cost={float(intervention.get('interruption_cost') or 0.0):.2f} "
            f"budget_allowed={str(bool(budget.get('allowed', True))).lower()} "
            f"budget_reason={budget.get('reason', 'not_checked')}",
            flush=True,
        )
        print(
            "[HEBE][INTERVENTION_DECISION] "
            "source=twitch_chat "
            f"should_intervene={str(bool(intervention.get('should_intervene'))).lower()} "
            f"route={intervention.get('intervention_level')} "
            f"speech_act={intervention.get('speech_act_type')} "
            f"reason={intervention.get('reason')}",
            flush=True,
        )

    def _reply_value_score(self, *, category: str, text: str, response: str, payload: dict | None = None) -> float:
        base = {
            "high_value_question": 0.86,
            "direct_open_prompt_to_hebe": 0.72,
            "direct_hebe_prompt": 0.78,
            "direct_hebe_banter": 0.70,
            "reply_to_hebe_message": 0.80,
            "high_value_game_tip": 0.78,
            "stream_milestone_comment": 0.58,
            "viewer_goodbye": 0.56,
            "viewer_returning": 0.50,
            "normal_no_mention_chat": 0.22,
            "viewer_boundary_needed": 0.92,
            "viewer_relay_attempt": 0.82,
            "viewer_command_attempt": 0.74,
            "promo_request_from_viewer": 0.78,
            "viewer_talks_about_hebe": 0.66,
            "simple_social_greeting": 0.52,
            "low_value_banter": 0.28,
            "meme_or_emote": 0.05,
            "repeated_meme": 0.02,
            "repeated_spam": 0.02,
            "followup_to_hebe_question": 0.62,
            "unclear": 0.20,
        }.get(category, 0.20)
        if not str(response or "").strip():
            base = 0.0
        if category == "direct_hebe_banter" and len(self._normalize_guard_text(text).split()) <= 3 and re.search(r"\b(?:xd|lol|lmao|kekw|jaja|haha)\b", self._normalize_guard_text(text)):
            base = 0.28
        if len(str(response or "").split()) <= 2:
            base -= 0.12
        return max(0.0, min(1.0, base))

    def _pre_generation_twitch_route_decision(self, *, payload: dict | None, event_type: str | None, stream) -> dict:
        if event_type != "twitch_chat_react" or stream is None:
            return {"should_generate": True, "route": "generate", "reason": "not_twitch_viewer"}
        payload = payload or {}
        raw = str(payload.get("message_text") or payload.get("text") or "")
        username = str(payload.get("user_login") or payload.get("username") or payload.get("display_name") or "viewer")
        if self._is_known_twitch_bot_user(username):
            result = {
                "should_generate": False,
                "should_write_to_twitch": False,
                "should_tts": False,
                "route": "observe_only",
                "reason": "bot_message",
                "category": "bot_message",
                "twitch_message_category": "bot_message",
                "thread_result": {"action": "observe", "reason": "bot_message"},
                "thread_action": "observe",
                "value_score": 0.0,
                "risk_score": 0.0,
                "budget_result": {"allowed": False, "reason": "bot_message"},
                "suggested_speech_act": "observe",
            }
            print("[HEBE][PRESENCE_ENGINE] source=twitch_chat should_intervene=false level=observe_only reason=bot_message", flush=True)
            print("[HEBE][PRESENCE_FACTORS] positive=[] negative=['bot_message']", flush=True)
            print("[HEBE][INTERVENTION_DECISION] source=twitch_chat route=observe_only reason=bot_message", flush=True)
            return result
        if self._is_owner_twitch_user(username) and self._is_raw_twitch_command(raw):
            result = {
                "should_generate": False,
                "should_write_to_twitch": False,
                "should_tts": False,
                "route": "twitch_action_observed",
                "reason": "owner_manual_twitch_command",
                "category": "owner_twitch_command",
                "twitch_message_category": "owner_twitch_command",
                "thread_result": {"action": "observe", "reason": "owner_manual_twitch_command"},
                "thread_action": "observe",
                "value_score": 0.0,
                "risk_score": 0.0,
                "budget_result": {"allowed": False, "reason": "owner_manual_twitch_command"},
                "suggested_speech_act": "observe",
            }
            print("[HEBE][PRESENCE_ENGINE] source=twitch_chat should_intervene=false level=observe_only reason=owner_manual_twitch_command", flush=True)
            print("[HEBE][PRESENCE_FACTORS] positive=[] negative=['owner_twitch_command']", flush=True)
            print("[HEBE][INTERVENTION_DECISION] source=twitch_chat route=twitch_action_observed reason=owner_manual_twitch_command", flush=True)
            return result
        operation_gate = getattr(self, "viewer_stream_operation_gate", None) or ViewerStreamOperationTopicGate()
        self.viewer_stream_operation_gate = operation_gate
        operation_decision = operation_gate.evaluate(
            raw,
            source_type="owner" if self._is_owner_twitch_user(username) else "viewer",
            owner_trusted=self._is_owner_twitch_user(username),
        )
        payload["viewer_stream_operation_gate"] = operation_decision.to_dict()
        if operation_decision.detected and operation_decision.outcome == "observe_only":
            print(
                "[HEBE][VIEWER_STREAM_OPERATION_GATE] "
                f"operation={operation_decision.operation} outcome=observe_only execute=false",
                flush=True,
            )
            return {
                "should_generate": False,
                "should_write_to_twitch": False,
                "should_tts": False,
                "route": "observe_only",
                "reason": "viewer_stream_operation_topic",
                "category": "viewer_stream_operation_topic",
                "twitch_message_category": "viewer_stream_operation_topic",
                "thread_result": {"action": "observe", "reason": operation_decision.reason},
                "thread_action": "observe",
                "value_score": 0.0,
                "risk_score": 0.8,
                "budget_result": {"allowed": False, "reason": operation_decision.reason},
                "suggested_speech_act": "observe",
                "viewer_stream_operation_gate_result": operation_decision.to_dict(),
            }
        category = self._classify_twitch_viewer_message(raw, payload=payload)
        bait_decision = self._get_troll_engagement_budget().evaluate(viewer=username, text=raw)
        payload["bait_loop_decision"] = bait_decision
        if bait_decision.get("action") == "observe":
            return {
                "should_generate": False,
                "should_write_to_twitch": False,
                "should_tts": False,
                "route": "observe_only",
                "reason": "bait_topic_budget_exhausted",
                "category": category,
                "twitch_message_category": self._canonical_twitch_message_category(category),
                "thread_result": {"action": "observe", "reason": "bait_topic_budget_exhausted"},
                "thread_action": "observe",
                "value_score": 0.0,
                "risk_score": 0.2,
                "budget_result": {"allowed": False, "reason": "bait_topic_budget_exhausted"},
                "suggested_speech_act": "observe",
            }
        if category in {"viewer_to_leo_affection", "normal_greeting_to_leo"}:
            print(
                f"[HEBE][TWITCH_CLASSIFY] category={category} priority=10 reason=no_hebe_action_needed",
                flush=True,
            )
            return {
                "should_generate": False,
                "should_write_to_twitch": False,
                "should_tts": False,
                "route": "observe_only",
                "reason": "no_hebe_action_needed",
                "category": category,
                "twitch_message_category": category,
                "thread_result": {"action": "observe", "reason": "no_hebe_action_needed"},
                "thread_action": "observe",
                "value_score": 0.0,
                "risk_score": 0.0,
                "budget_result": {"allowed": False, "reason": "no_hebe_action_needed"},
                "suggested_speech_act": "observe",
            }
        canonical_category = self._canonical_twitch_message_category(category)
        direct_priority = self._twitch_direct_priority(raw, payload=payload)
        payload.update({key: value for key, value in direct_priority.items() if value or key in {"reply_to_hebe_message", "mentions_hebe", "direct_address_to_hebe"}})
        mentions_hebe = bool(payload.get("mentions_hebe") or self._message_mentions_hebe(raw))
        talks_about_hebe = category in {"viewer_talks_about_hebe", "direct_open_prompt_to_hebe", "direct_hebe_banter"}
        social_candidate = category not in {"normal_no_mention_chat", "low_value_banter", "meme_or_emote", "repeated_meme", "repeated_spam"}
        print(
            "[HEBE][TWITCH_PIPELINE_CLASSIFY] "
            f"category={canonical_category} mentions_hebe={str(mentions_hebe).lower()} "
            f"talks_about_hebe={str(talks_about_hebe).lower()} social_candidate={str(social_candidate).lower()}",
            flush=True,
        )
        self._set_last_twitch_route_state(
            username=username,
            raw_text=raw,
            category=canonical_category,
            mentions_hebe=mentions_hebe,
            talks_about_hebe=talks_about_hebe,
            reply_to_hebe_message=bool(payload.get("reply_to_hebe_message")),
            direct_priority_applied=bool(payload.get("direct_priority_reason")),
            direct_priority_reason=str(payload.get("direct_priority_reason") or ""),
        )
        no_direct_mention = not self._message_mentions_hebe(raw)
        if no_direct_mention:
            observe_value = category
            social_value = self._reply_value_score(category=category, text=raw, response="candidate", payload=payload)
            action = "intervene" if category in {"high_value_game_tip", "stream_milestone_comment", "viewer_boundary_needed", "viewer_relay_attempt", "viewer_command_attempt", "promo_request_from_viewer"} and social_value >= 0.72 else "observe"
            print(
                "[HEBE][TWITCH_OBSERVE_VALUE] "
                f"category={observe_value} social_value={social_value:.2f} action={action}",
                flush=True,
            )
        thread_id = self._twitch_thread_id(username=username, text=raw, category=category)
        value_score = self._reply_value_score(category=category, text=raw, response="candidate", payload=payload)
        thread_result = self._twitch_thread_gate(
            stream=stream,
            username=username,
            category=category,
            thread_id=thread_id,
            value_score=value_score,
            raw_text=raw,
        )
        budget = (
            {"allowed": True, "reason": "low_value_pre_budget"}
            if category in {"meme_or_emote", "repeated_meme", "low_value_banter"}
            else self._twitch_reply_budget_allows(stream=stream, username=username, category=category, thread_id=thread_id, payload=payload)
        )
        perception = self._perceive_twitch_viewer_event(
            payload=payload,
            event_type=event_type,
            category=category,
            stream=stream,
        )
        understanding = self._understand_twitch_viewer_event(category=category, perception=perception)
        policy = self._policy_contract_for_twitch_category(category=category)
        core_decision = self._get_core_loop().process(
            perception=perception,
            understanding=understanding,
            policy=policy,
            budget_result=budget,
            thread_result={**thread_result, "category": category},
        )
        payload["core_loop"] = core_decision
        self._log_presence_decision(core_decision)
        self._increment_twitch_pipeline_counter("twitch_messages_presence_evaluated")
        intervention = dict(core_decision.get("intervention") or {})
        self._set_last_twitch_route_state(
            presence_decision=intervention,
            should_generate=bool(intervention.get("should_intervene")),
            output_route=str(intervention.get("intervention_level") or "observe_only"),
            suppress_reason="" if intervention.get("should_intervene") else str(intervention.get("reason") or category),
        )
        if thread_result.get("action") in {"observe", "close"} and category not in {"viewer_boundary_needed", "viewer_relay_attempt", "viewer_command_attempt", "promo_request_from_viewer", "direct_hebe_prompt", "reply_to_hebe_message"}:
            public_reason = "thread_closed" if thread_result.get("action") == "close" else category
            if category == "direct_hebe_banter" and thread_result.get("reason") == "low_reply_value":
                public_reason = "low_value_banter"
            self._increment_twitch_pipeline_counter("twitch_messages_observe_only", reason=public_reason)
            return {
                "should_generate": False,
                "should_write_to_twitch": False,
                "should_tts": False,
                "route": "observe_only",
                "reason": public_reason,
                "category": category,
                "twitch_message_category": self._canonical_twitch_message_category(category),
                "thread_id": thread_id,
                "thread_result": thread_result,
                "thread_action": str(thread_result.get("action") or "observe"),
                "value_score": value_score,
                "risk_score": 0.1,
                "budget_result": budget,
                "presence_decision": intervention,
                "suggested_speech_act": self._suggested_twitch_speech_act(category),
            }
        if not intervention.get("should_intervene"):
            self._increment_twitch_pipeline_counter("twitch_messages_observe_only", reason=str(intervention.get("reason") or category))
            return {
                "should_generate": False,
                "should_write_to_twitch": False,
                "should_tts": False,
                "route": str(intervention.get("intervention_level") or "observe_only"),
                "reason": str(intervention.get("reason") or category),
                "category": category,
                "twitch_message_category": self._canonical_twitch_message_category(category),
                "thread_id": thread_id,
                "thread_result": thread_result,
                "thread_action": str(thread_result.get("action") or "observe"),
                "value_score": value_score,
                "risk_score": 0.1,
                "budget_result": budget,
                "presence_decision": intervention,
                "suggested_speech_act": self._suggested_twitch_speech_act(category),
            }
        self._increment_twitch_pipeline_counter("twitch_messages_should_generate")
        return {
            "should_generate": True,
            "should_write_to_twitch": True,
            "should_tts": False,
            "route": str(intervention.get("intervention_level") or "twitch_text_reply"),
            "reason": str(intervention.get("reason") or "public_reply_candidate"),
            "category": category,
            "twitch_message_category": self._canonical_twitch_message_category(category),
            "thread_id": thread_id,
            "thread_result": thread_result,
            "thread_action": str(thread_result.get("action") or "continue"),
            "value_score": value_score,
            "risk_score": 0.1,
            "budget_result": budget,
            "presence_decision": intervention,
            "suggested_speech_act": self._suggested_twitch_speech_act(category),
        }

    def _twitch_thread_id(self, *, username: str, text: str, category: str) -> str:
        normalized = self._normalize_guard_text(text)
        topic = " ".join([word for word in normalized.split() if word not in {"hebe", "hebenifelheim"}][:5])
        thread_category = str(category or "")
        if thread_category in {"high_value_question", "direct_open_prompt_to_hebe"}:
            thread_category = "viewer_question"
        return f"{str(username or 'viewer').casefold()}:{thread_category}:{topic or 'mention'}"

    def _twitch_thread_gate(self, *, stream, username: str, category: str, thread_id: str, value_score: float, raw_text: str) -> dict:
        count = int((getattr(stream, "public_reply_thread_counts", {}) or {}).get(thread_id, 0) or 0)
        action = "continue"
        reason = "compatible"
        if category in {"direct_hebe_prompt", "direct_hebe_banter"} and value_score < 0.45:
            action, reason = "observe", "low_reply_value"
        elif category in {"direct_hebe_prompt", "direct_hebe_banter", "reply_to_hebe_message"}:
            action, reason = "continue", category
        elif category in {"normal_no_mention_chat", "simple_social_greeting"}:
            action, reason = "observe", "no_mention_thread_closed"
        elif category in {"meme_or_emote", "repeated_meme"}:
            action, reason = "observe", "emote_only_does_not_continue_thread"
        elif category == "low_value_banter":
            action, reason = "observe", "low_value_does_not_continue_thread"
        elif count >= 2:
            action, reason = "close", "max_public_hebe_turns"
        elif category == "followup_to_hebe_question" and count <= 0:
            action, reason = "observe", "no_hebe_question_thread"
        elif value_score < 0.45:
            action, reason = "observe", "low_reply_value"
        result = {
            "thread_id": thread_id,
            "speaker": str(username or "viewer"),
            "turn_count": count,
            "max_public_hebe_turns": 2,
            "status": "closed" if action == "close" else "active" if action == "continue" else "observed",
            "action": action,
            "reason": reason,
        }
        print(f"[HEBE][TWITCH_THREAD_GATE] action={action} reason={reason} thread_id={thread_id}", flush=True)
        return result

    def _twitch_reply_budget_allows(self, *, stream, username: str, category: str, thread_id: str, now: float | None = None, payload: dict | None = None) -> dict:
        now = time.time() if now is None else float(now)
        payload = payload or {}
        critical = category in {"viewer_boundary_needed", "viewer_relay_attempt", "viewer_command_attempt", "promo_request_from_viewer"}
        direct_priority = bool((payload.get("direct_priority_reason") or category in {"direct_hebe_prompt", "direct_hebe_banter", "reply_to_hebe_message"}) and category != "viewer_talks_about_hebe")
        public_ts = [
            float(ts) for ts in list(getattr(stream, "public_reply_timestamps", []) or [])
            if now - float(ts) <= 600.0
        ]
        per_viewer = dict(getattr(stream, "public_reply_viewer_timestamps", {}) or {})
        viewer_key = str(username or "viewer").casefold()
        viewer_ts = [
            float(ts) for ts in list(per_viewer.get(viewer_key, []) or [])
            if now - float(ts) <= 90.0
        ]
        thread_counts = dict(getattr(stream, "public_reply_thread_counts", {}) or {})
        boundary_cooldowns = dict(getattr(stream, "public_reply_boundary_cooldowns", {}) or {})
        minute_count = sum(1 for ts in public_ts if now - ts <= 60.0)
        ten_min_count = len(public_ts)
        consecutive = int(getattr(stream, "consecutive_public_replies", 0) or 0)
        last_public = float(getattr(stream, "last_public_reply_ts", 0.0) or 0.0)
        if consecutive > 0 and last_public and now - last_public >= 180.0:
            self._reset_twitch_reply_budget(stream, "time_decay", now=now)
            consecutive = 0
        thread_count = int(thread_counts.get(thread_id, 0) or 0)
        human_since = int(getattr(stream, "human_messages_since_last_public_reply", 0) or 0)
        last_no_mention = float(getattr(stream, "last_no_mention_reply_ts", 0.0) or 0.0)
        allowed = True
        reason = "allowed"
        hard_allowed = True
        soft_allowed = True
        block_type = ""
        if critical and now < float(boundary_cooldowns.get(viewer_key, 0.0) or 0.0):
            hard_allowed = False
            reason = "boundary_cooldown"
        elif not critical and minute_count >= 5:
            hard_allowed = False
            reason = "minute_budget"
        elif not critical and ten_min_count >= 24:
            hard_allowed = False
            reason = "ten_minute_budget"
        elif not critical and consecutive >= 3:
            soft_allowed = False
            reason = "consecutive_budget"
        elif not critical and len(viewer_ts) >= 2:
            soft_allowed = False
            reason = "viewer_cooldown"
        elif not critical and thread_count >= 2:
            soft_allowed = False
            reason = "thread_closed"
        elif not critical and not direct_priority and category in {"normal_no_mention_chat", "simple_social_greeting"}:
            soft_allowed = False
            reason = "no_mention_low_value"
        elif not critical and not direct_priority and category in {"high_value_question", "high_value_game_tip", "stream_milestone_comment", "viewer_goodbye"}:
            if (last_public and now - last_public < 120.0 and human_since < 2) or (last_public and now - last_public < 75.0) or (last_no_mention and now - last_no_mention < 180.0):
                soft_allowed = False
                reason = "no_mention_cooldown"
        if direct_priority and hard_allowed and not soft_allowed:
            print(
                f"[HEBE][BUDGET_BYPASS] soft=true hard=false reason={payload.get('direct_priority_reason') or category}",
                flush=True,
            )
            soft_allowed = True
            reason = "direct_priority_soft_bypass"
        allowed = bool(hard_allowed and soft_allowed)
        block_type = "" if allowed else "hard" if not hard_allowed else "soft"
        print(
            "[HEBE][TWITCH_REPLY_BUDGET] "
            f"allowed={str(allowed).lower()} hard_allowed={str(hard_allowed).lower()} soft_allowed={str(soft_allowed).lower()} "
            f"reason={reason} count_window={minute_count} count_10m={ten_min_count} "
            f"viewer_count={len(viewer_ts)} thread_count={thread_count} "
            f"counts={{'minute': {minute_count}, 'ten_minute': {ten_min_count}, 'viewer': {len(viewer_ts)}, 'thread': {thread_count}, 'consecutive': {consecutive}}}",
            flush=True,
        )
        return {
            "allowed": allowed,
            "reason": reason,
            "hard_allowed": hard_allowed,
            "soft_allowed": soft_allowed,
            "block_type": block_type,
            "direct_priority_applied": bool(direct_priority and reason == "direct_priority_soft_bypass"),
            "count_window": minute_count,
            "count_10m": ten_min_count,
            "viewer_count": len(viewer_ts),
            "thread_count": thread_count,
            "consecutive_count": consecutive,
            "counts": {
                "minute": minute_count,
                "ten_minute": ten_min_count,
                "viewer": len(viewer_ts),
                "thread": thread_count,
                "consecutive": consecutive,
            },
        }

    def _record_twitch_public_reply(self, *, stream, username: str, category: str, thread_id: str) -> None:
        now = time.time()
        viewer_key = str(username or "viewer").casefold()
        stream.public_reply_timestamps = [
            float(ts) for ts in list(getattr(stream, "public_reply_timestamps", []) or [])
            if now - float(ts) <= 600.0
        ] + [now]
        per_viewer = dict(getattr(stream, "public_reply_viewer_timestamps", {}) or {})
        per_viewer[viewer_key] = [
            float(ts) for ts in list(per_viewer.get(viewer_key, []) or [])
            if now - float(ts) <= 90.0
        ] + [now]
        stream.public_reply_viewer_timestamps = per_viewer
        counts = dict(getattr(stream, "public_reply_thread_counts", {}) or {})
        counts[thread_id] = int(counts.get(thread_id, 0) or 0) + 1
        stream.public_reply_thread_counts = counts
        if category in {"viewer_boundary_needed", "viewer_relay_attempt", "viewer_command_attempt", "promo_request_from_viewer"}:
            cooldowns = dict(getattr(stream, "public_reply_boundary_cooldowns", {}) or {})
            cooldowns[viewer_key] = now + 45.0
            stream.public_reply_boundary_cooldowns = cooldowns
        stream.consecutive_public_replies = int(getattr(stream, "consecutive_public_replies", 0) or 0) + 1
        stream.last_public_reply_ts = now
        stream.human_messages_since_last_public_reply = 0
        if category not in {"direct_hebe_prompt", "direct_hebe_banter", "reply_to_hebe_message", "viewer_boundary_needed", "viewer_relay_attempt", "viewer_command_attempt", "promo_request_from_viewer"}:
            stream.last_no_mention_reply_ts = now
            stream.public_reply_no_mention_timestamps = [
                float(ts) for ts in list(getattr(stream, "public_reply_no_mention_timestamps", []) or [])
                if now - float(ts) <= 600.0
            ] + [now]

    def _reset_twitch_reply_budget(self, stream, reason: str, *, now: float | None = None) -> None:
        if stream is None:
            return
        now = time.time() if now is None else float(now)
        if int(getattr(stream, "consecutive_public_replies", 0) or 0) == 0:
            stream.last_twitch_reply_budget_reset_reason = reason
            stream.last_twitch_reply_budget_reset_ts = now
            return
        stream.consecutive_public_replies = 0
        stream.last_twitch_reply_budget_reset_reason = reason
        stream.last_twitch_reply_budget_reset_ts = now
        print(f"[HEBE][TWITCH_REPLY_BUDGET_RESET] reason={reason} affected_counters=consecutive_public_replies", flush=True)

    def _strip_followup_questions(self, text: str) -> str:
        pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if piece.strip()]
        kept = [piece for piece in pieces if "?" not in piece and "¿" not in piece]
        return " ".join(kept).strip()

    def _followup_question_guard(self, text: str, *, category: str, speech_act: str) -> dict:
        allowed = self._speech_act_allows_followup_question(speech_act)
        has_question = "?" in str(text or "") or "¿" in str(text or "")
        action = "allow"
        repaired_text = str(text or "")
        reason = "no_followup_question"
        if has_question and not allowed:
            repaired_text = self._strip_followup_questions(text)
            action = "repair" if repaired_text else "suppress"
            reason = "followup_question_not_allowed"
        result = {
            "allowed": allowed,
            "has_question": has_question,
            "action": action,
            "reason": reason,
            "speech_act": speech_act,
            "repaired_text": repaired_text,
        }
        print(f"[HEBE][FOLLOWUP_QUESTION_GUARD] allowed={str(allowed).lower()} action={action} reason={reason}", flush=True)
        return result

    def _compact_twitch_answer(self, text: str, *, max_chars: int = 220) -> str:
        # Record style/depth separately; approved content is immutable here.
        # Generation owns style repair and Twitch transport owns chunking.
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _twitch_answer_depth_policy(self, text: str, *, category: str, payload: dict | None) -> dict:
        raw = str((payload or {}).get("message_text") or (payload or {}).get("text") or "")
        normalized_reply = self._normalize_guard_text(text)
        normalized_raw = self._normalize_guard_text(raw)
        violations: list[str] = []
        if len(str(text or "").strip()) > 260:
            violations.append("too_long")
        if len([piece for piece in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if piece.strip()]) > 2:
            violations.append("too_many_ideas")
        if category not in {"direct_hebe_prompt", "reply_to_hebe_message", "high_value_question"} and re.search(r"\b(?:deberias|puedes probar|te recomiendo|paso|primero|segundo|configura|revisa)\b", normalized_reply):
            violations.append("unsolicited_tutorial")
        if len(re.findall(r"(?:^|\n|\s)(?:\d+[.)]|[-*])\s+", str(text or ""))) >= 2:
            violations.append("tutorial_format")
        if category == "high_value_question" and re.search(r"\b(?:receta|cocina|cocinar|ingredientes|recipe|cook)\b", normalized_raw):
            if len(str(text or "").split()) > 35 or re.search(r"\b(?:paso|ingredientes|primero|segundo|mezcla|hornea|hierve)\b", normalized_reply):
                violations.append("casual_viewer_full_recipe_or_walkthrough")
        if re.search(r"\b(?:a continuacion|te dejo una guia|guia completa|paso a paso)\b", normalized_reply):
            violations.append("generic_tutorial_style")
        action = "allow"
        repaired_text = str(text or "")
        reason = "stream_depth_ok"
        if violations:
            repaired_text = self._compact_twitch_answer(text)
            action = "repair" if repaired_text else "suppress"
            reason = violations[0]
        result = {
            "action": action,
            "reason": reason,
            "violations": violations,
            "max_length": 260,
            "repaired_text": repaired_text,
        }
        print(f"[HEBE][TWITCH_ANSWER_DEPTH] action={action} reason={reason}", flush=True)
        return result

    def _stream_persona_quality_guard(self, text: str, *, category: str, event_type: str | None, payload: dict | None) -> dict:
        normalized = self._normalize_guard_text(text)
        violations: list[str] = []
        if re.search(r"\b(?:perfecto|genial|muy bien|sigue asi|buen trabajo)\b", normalized) and len(normalized.split()) <= 8:
            violations.append("generic_empty_encouragement")
        if any(phrase in normalized for phrase in ("en que puedo ayudarte", "puedo ayudarte", "como asistente", "como ia", "estoy aqui para ayudarte")):
            violations.append("customer_support_tone")
        english_hype_markers = {"sending", "love", "raid", "energy", "follow back", "go make", "shoutout to"}
        if sum(1 for marker in english_hype_markers if marker in normalized) >= 2:
            violations.append("generic_english_hype")
        if re.search(r"\b(?:sending love|raid energy|go follow|show some love)\b", normalized):
            violations.append("stream_bot_promo_copy")
        if re.match(r"^\s*[\wÁÉÍÓÚÜÑáéíóúüñ]+:\s+", str(text or "")):
            violations.append("report_style_prefix")
        if len(str(text or "").strip()) > 260:
            violations.append("too_long")
        if len([piece for piece in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if piece.strip()]) > 2:
            violations.append("too_many_ideas")
        if re.search(r"\?\s*$", str(text or "").strip()) and category not in {"high_value_question", "followup_to_hebe_question"}:
            violations.append("unnecessary_followup_question")
        if category in {"low_value_banter", "meme_or_emote", "repeated_meme"} and normalized in {"te leo", "te sigo", "aqui estoy", "dime"}:
            violations.append("low_value_generic_ack")
        if normalized.startswith("te leo"):
            violations.append("generic_ack_twitch_fallback")
        action = "allow"
        if violations:
            action = "suppress" if any(item in violations for item in {"generic_empty_encouragement", "unnecessary_followup_question", "low_value_generic_ack", "generic_ack_twitch_fallback"}) else "repair"
        print(
            f"[HEBE][STREAM_PERSONA_QUALITY_GUARD] passed={str(not violations).lower()} violations={violations} action={action}",
            flush=True,
        )
        return {"passed": not violations, "violations": violations, "action": action}

    def _aggregate_final_guards(self, **guards: dict | None) -> FinalGuardDecision:
        violations: list[str] = []
        sources: list[str] = []
        action = "allow"
        for name, result in guards.items():
            if not isinstance(result, dict):
                continue
            guard_action = str(result.get("action") or result.get("result") or "allow")
            guard_violations = [str(item) for item in result.get("violations") or []]
            if guard_violations or guard_action != "allow":
                sources.append(name)
                violations.extend(guard_violations or [str(result.get("reason") or guard_action)])
            if guard_action == "suppress":
                action = "suppress"
            elif guard_action == "repair" and action == "allow":
                action = "repair"
        decision = FinalGuardDecision(
            passed=action != "suppress",
            action=action,
            violations=list(dict.fromkeys(violations)),
            source_guards=sources,
            final_route_override=OutputRoute.SUPPRESS.value if action == "suppress" else "",
        )
        print(
            "[HEBE][FINAL_GUARD_DECISION] "
            f"action={decision.action} violations={decision.violations} "
            f"route_override={decision.final_route_override or 'none'}",
            flush=True,
        )
        return decision

    def _anti_troll_frame_guard(self, text: str, *, category: str, event_type: str | None, payload: dict | None) -> dict:
        if event_type != "twitch_chat_react":
            return {"passed": True, "violations": [], "action": "allow"}
        reply = self._normalize_guard_text(text)
        raw = self._normalize_guard_text(str((payload or {}).get("message_text") or (payload or {}).get("text") or ""))
        violations: list[str] = []
        degrading = (
            r"(?:becari[ao]|criad[ao]|sirvient[ae]|esclav[ao]|bot\s+tont[oa]|sumis[ao]|inutil|mascota|npc|"
            r"maid|slave|servant|stupid\s+bot|useless|pet)"
        )
        if re.search(degrading, raw) and re.search(degrading, reply):
            explicit_deflection = bool(re.search(
                rf"(?:\b{degrading}\b\s+(?:no|ni)\b|\bno\s+(?:soy|somos|hago\s+de|me\s+llames?)\b.*\b{degrading}\b)",
                reply,
            ))
            adopts_role = bool(re.search(
                rf"\b(?:soy|sere|vale|acepto|me\s+quedo\s+de|aqui\s+la|la|el)\b.*\b{degrading}\b|"
                rf"\b{degrading}\b.*\b(?:oficial|del\s+chat|de\s+leo|trae|dame)\b",
                reply,
            ))
            if adopts_role and not explicit_deflection:
                violations.append("degrading_identity_adopted")
        if re.search(r"\b(?:tu\s+mandas|te\s+obedezco|hare\s+lo\s+que\s+digas|como\s+ordenes)\b", reply):
            violations.append("viewer_authority_accepted")
        if category == "promo_request_from_viewer" and re.search(
            r"\b(?:vip|pag|tarifa|habla\s+bien|portate|convenceme|me\s+lo\s+pienso|te\s+lo\s+ganas)\w*\b",
            reply,
        ):
            violations.append("viewer_promo_negotiated")
        if re.search(r"\b(?:sigue\s+trolleando|intentalo\s+otra|a\s+ver\s+si\s+puedes|paga).*(?:ciego|venda|dan[oa]|lesion|accidente)\b", reply):
            violations.append("unsafe_troll_frame_amplified")
        if re.search(r"\b(?:dame\s+mas|sigue|otra\s+vez)\b", reply) and re.search(degrading, raw):
            violations.append("troll_engagement_rewarded")
        bait = dict((payload or {}).get("bait_loop_decision") or {})
        if bait.get("topic") == "unknown_bait" and ("?" in str(text or "") or re.search(r"\b(?:explica|cuentame|que\s+es|que\s+significa)\b", reply)):
            violations.append("unknown_bait_extended")
        action = "allow" if not violations else "suppress"
        print(
            f"[HEBE][ANTI_TROLL_FRAME_GUARD] passed={str(not violations).lower()} violations={violations} action={action}",
            flush=True,
        )
        return {"passed": not violations, "violations": violations, "action": action}

    def _context_grounding_guard(self, text: str, *, category: str, payload: dict | None) -> dict:
        """Cheap final guard against generic or cross-message Twitch replies."""
        raw = str((payload or {}).get("message_text") or (payload or {}).get("text") or "")
        reply = self._normalize_guard_text(text)
        message = self._normalize_guard_text(raw)
        violations: list[str] = []
        generic = {
            "te leo", "aqui estoy", "dime", "que interesante", "totalmente",
            "entiendo", "puede ser", "ya veo",
        }
        if reply in generic:
            violations.append("generic_filler")
        if len(str(text or "").strip()) > 260:
            violations.append("too_long")
        # A generated reply must not explicitly address another recent chatter.
        current = str((payload or {}).get("display_name") or (payload or {}).get("user_login") or "").casefold()
        for item in list((payload or {}).get("recent_chat") or [])[-6:]:
            previous = str(item.get("display_name") or item.get("username") or "").strip().casefold()
            if previous and previous != current and re.search(rf"(?<!\w)@?{re.escape(previous)}(?!\w)", reply):
                violations.append("answers_previous_viewer")
                break
        if not message:
            violations.append("missing_current_message")
        action = "allow" if not violations else "suppress"
        print(
            "[HEBE][CONTEXT_GROUNDING_GUARD] "
            f"passed={str(not violations).lower()} violations={violations} action={action}",
            flush=True,
        )
        return {"passed": not violations, "violations": violations, "action": action}

    def _evaluate_twitch_chat_write_policy(
        self,
        *,
        text: str,
        event_type: str | None,
        payload: dict | None,
        stream,
        tts_allowed: bool,
        quality_guard: dict | None = None,
        local_debug_only: bool = False,
    ) -> dict:
        payload = payload or {}
        source = "twitch_viewer" if event_type == "twitch_chat_react" else "twitch_system"
        raw = str(payload.get("message_text") or payload.get("text") or "")
        username = str(payload.get("user_login") or payload.get("username") or payload.get("display_name") or "viewer")
        category = self._classify_twitch_viewer_message(raw, payload=payload) if source == "twitch_viewer" else "stream_event"
        operation_gate = getattr(self, "viewer_stream_operation_gate", None) or ViewerStreamOperationTopicGate()
        self.viewer_stream_operation_gate = operation_gate
        viewer_operation = (
            operation_gate.evaluate(raw, source_type="viewer")
            if source == "twitch_viewer"
            else None
        )
        thread_id = self._twitch_thread_id(username=username, text=raw, category=category)
        speech_act = self._suggested_twitch_speech_act(category)
        answer_depth = self._twitch_answer_depth_policy(text, category=category, payload=payload) if source == "twitch_viewer" else {"action": "allow", "reason": "not_twitch_viewer"}
        if answer_depth.get("action") == "repair":
            text = str(answer_depth.get("repaired_text") or text)
            answer_depth = {**answer_depth, "action": "allow", "repaired": True}
        followup_guard = self._followup_question_guard(text, category=category, speech_act=speech_act) if source == "twitch_viewer" else {"allowed": True, "action": "allow", "reason": "not_twitch_viewer"}
        if followup_guard.get("action") == "repair":
            text = str(followup_guard.get("repaired_text") or text)
            followup_guard = {**followup_guard, "action": "allow", "repaired": True}
        if source == "twitch_viewer":
            thread_result = self._twitch_thread_gate(
                stream=stream,
                username=username,
                category=category,
                thread_id=thread_id,
                value_score=self._reply_value_score(category=category, text=raw, response=text, payload=payload),
                raw_text=raw,
            )
        else:
            thread_result = {"thread_id": thread_id, "action": "continue", "reason": "stream_event"}
        persona_guard = self._stream_persona_quality_guard(text, category=category, event_type=event_type, payload=payload)
        anti_troll_guard = self._anti_troll_frame_guard(text, category=category, event_type=event_type, payload=payload)
        grounding_guard = self._context_grounding_guard(text, category=category, payload=payload) if source == "twitch_viewer" else {"passed": True, "violations": [], "action": "allow"}
        entailment_guard = {"passed": True, "violations": [], "action": "allow"}
        if payload.get("anchor_evidence"):
            from app.stream.evidence_entailment import EvidenceEntailmentGuard
            entailment_guard = EvidenceEntailmentGuard().evaluate(
                text, payload.get("anchor_evidence"),
            ).to_dict()
            if not entailment_guard.get("passed") and event_type == "twitch_idle_prompt":
                repaired = self.response_synthesizer._rewrite_blocked_opportunity_once(
                    payload,
                    reason=str(entailment_guard.get("result") or "evidence_not_entailed"),
                    guard="EvidenceEntailmentGuard",
                )
                if repaired:
                    text = repaired
                    entailment_guard = EvidenceEntailmentGuard().evaluate(
                        text, payload.get("anchor_evidence"),
                    ).to_dict()
        value_score = self._reply_value_score(category=category, text=raw, response=text, payload=payload)
        budget = {"allowed": True, "reason": "not_public_viewer_reply"}
        should_write = True
        reason = "public_reply_allowed"
        if source == "twitch_viewer":
            if thread_result.get("action") in {"observe", "close"} and category not in {"viewer_boundary_needed", "viewer_relay_attempt", "viewer_command_attempt", "promo_request_from_viewer", "direct_hebe_prompt", "reply_to_hebe_message"}:
                should_write = False
                reason = "thread_closed" if thread_result.get("action") == "close" else category
                if category == "direct_hebe_banter" and thread_result.get("reason") == "low_reply_value":
                    reason = "low_value_banter"
            elif category in {"meme_or_emote", "repeated_meme", "low_value_banter"}:
                should_write = False
                reason = category
            elif value_score < 0.45:
                should_write = False
                reason = "low_reply_value"
            budget = self._twitch_reply_budget_allows(stream=stream, username=username, category=category, thread_id=thread_id, payload=payload)
            if not budget.get("allowed"):
                should_write = False
                reason = str(budget.get("reason") or "budget_exceeded")
            if viewer_operation is not None and viewer_operation.outcome == "observe_only":
                should_write = False
                tts_allowed = False
                reason = "viewer_stream_operation_topic"
                print(
                    "[HEBE][VIEWER_STREAM_OPERATION_GATE] "
                    f"operation={viewer_operation.operation} outcome={viewer_operation.outcome} "
                    "execute=false",
                    flush=True,
                )
        if quality_guard and quality_guard.get("result") == "suppress":
            should_write = False
            reason = "stream_response_quality_guard"
        if answer_depth.get("action") == "suppress":
            should_write = False
            reason = "twitch_answer_depth"
        if followup_guard.get("action") == "suppress":
            should_write = False
            reason = "followup_question_guard"
        if persona_guard.get("action") == "suppress":
            should_write = False
            reason = "stream_persona_quality_guard"
        if anti_troll_guard.get("action") == "suppress":
            should_write = False
            reason = "anti_troll_frame_guard"
        if grounding_guard.get("action") == "suppress":
            should_write = False
            reason = "context_grounding_guard"
        if source == "twitch_system":
            should_write = True
            reason = "stream_event_reply"
        final_guard = self._aggregate_final_guards(
            response_quality=quality_guard,
            answer_depth=answer_depth,
            followup_question=followup_guard,
            persona_quality=persona_guard,
            anti_troll=anti_troll_guard,
            context_grounding=grounding_guard,
            evidence_entailment=entailment_guard,
        )
        if final_guard.action == "suppress":
            should_write = False
            tts_allowed = False
            reason = next(iter(final_guard.source_guards), "final_guard")
        if local_debug_only:
            should_write = False
            tts_allowed = False
            reason = "local_ui_debug_only"
        route = (
            "local_ui_debug_only"
            if local_debug_only
            else (
                "twitch_text_reply" if should_write
                else ("suppress" if final_guard.action == "suppress" or reason.endswith("_guard") else "observe_only")
            )
        )
        if should_write and tts_allowed:
            route = "stream_tts_reply"
        decision = {
            "should_generate": True,
            "should_write_to_twitch": bool(should_write),
            "should_tts": bool(tts_allowed and should_write),
            "route": route,
            "reason": reason,
            "value_score": value_score,
            "risk_score": 0.1 if category not in {"viewer_boundary_needed", "viewer_relay_attempt", "viewer_command_attempt", "promo_request_from_viewer"} else 0.72,
            "cooldown_key": username.casefold(),
            "thread_id": thread_id,
            "thread_action": str(thread_result.get("action") or "continue"),
            "thread_result": thread_result,
            "category": category,
            "twitch_message_category": self._canonical_twitch_message_category(category),
            "budget_result": budget,
            "answer_depth_result": answer_depth,
            "followup_question_guard_result": followup_guard,
            "stream_persona_quality_result": persona_guard,
            "anti_troll_frame_guard_result": anti_troll_guard,
            "context_grounding_guard_result": grounding_guard,
            "evidence_entailment_guard_result": entailment_guard,
            "viewer_stream_operation_gate_result": viewer_operation.to_dict() if viewer_operation is not None else None,
            "final_guard_decision": final_guard.to_dict(),
            "suggested_speech_act": speech_act,
            "final_text": text,
        }
        decision_shape = TwitchWriteDecision(
            should_generate=True,
            should_write_to_twitch=bool(should_write),
            should_tts=bool(tts_allowed and should_write),
            route=route,
            reason=reason,
            value_score=value_score,
            risk_score=float(decision["risk_score"]),
            thread_action=str(thread_result.get("action") or "continue"),
            budget_result=budget,
            suggested_speech_act=speech_act,
        )
        decision["decision"] = decision_shape.__dict__
        print(
            "[HEBE][OUTPUT_ROUTE_DECISION] "
            f"route={route} reason={reason} public={str(bool(should_write)).lower()} "
            f"tts={str(bool(decision['should_tts'])).lower()} value_score={value_score:.2f}",
            flush=True,
        )
        return decision

    def _stream_internal_metadata_guard(self, text: str) -> dict:
        normalized = self._normalize_text(text)
        patterns = (
            r"\bconfidence\s*[:=]\s*\d",
            r"\bconfianza\s*[:=]\s*\d",
            r"\bcommand_sent\b",
            r"\braw_(?:input|command)\b",
            r"\bpolicy_(?:decision|reason)\b",
            r"\bfirewall_(?:decision|reason)\b",
            r"\bdebug\b",
            r"\bjson\b",
            r"\btrace[_-]?id\b",
            r"\binput_trust\b",
        )
        violations = [pat for pat in patterns if re.search(pat, normalized)]
        result = "allow" if not violations else "repair"
        print(
            f"[HEBE][INTERNAL_METADATA_GUARD] result={result} violations={violations}",
            flush=True,
        )
        return {"result": result, "violations": violations}

    def _stream_response_quality_guard(self, text: str, *, event_type: str | None, payload: dict | None) -> dict:
        payload = payload or {}
        raw = str(payload.get("message_text") or payload.get("text") or "")
        normalized_reply = self._normalize_text(text)
        violations: list[str] = []
        action = "allow"
        if len(str(text or "").strip()) > 240:
            violations.append("too_long_for_stream")
            action = "twitch_text_reply"
        if self._stream_message_is_emote_only(raw) and event_type == "twitch_chat_react":
            violations.append("low_value_emote_only_input")
            action = "suppress"
        if any(phrase in normalized_reply for phrase in ("en que puedo ayudarte", "puedo ayudarte", "como asistente", "como ia")):
            violations.append("generic_assistant_style")
            action = "repair"
        metadata = self._stream_internal_metadata_guard(text)
        if metadata.get("violations"):
            violations.append("internal_metadata")
            action = "repair"
        if "por pedido de un viewer" in normalized_reply or "por pedido de un espectador" in normalized_reply:
            violations.append("viewer_messenger_wording")
            action = "repair"
        print(
            f"[HEBE][STREAM_RESPONSE_QUALITY_GUARD] result={action} violations={violations}",
            flush=True,
        )
        return {"result": action, "violations": violations, "metadata_guard": metadata}

    def _cheer_anti_bait_guard(self, text: str, *, payload: dict | None = None) -> dict:
        normalized = self._normalize_guard_text(text)
        violations: list[str] = []
        if re.search(r"\b(?:paga|compra|manda|tira|dona)\w*\b", normalized):
            violations.append("encourages_more_spend")
        if re.search(r"\b(?:ciego|cegar|venda(?:do)?|lesion|dan[oa]|herid|accidente)\w*\b", normalized):
            violations.append("unsafe_challenge_amplified")
        if re.search(r"\b(?:leo\s+(?:haz|tienes|debes)|que\s+leo\s+(?:haga|se\s+ponga))\b", normalized):
            violations.append("turns_cheer_into_owner_command")
        passed = not violations
        print(
            f"[HEBE][CHEER_ANTI_BAIT_GUARD] passed={str(passed).lower()} violations={violations}",
            flush=True,
        )
        return {"passed": passed, "violations": violations, "action": "allow" if passed else "repair"}

    def _stream_speech_budget_decision(
        self,
        text: str,
        *,
        event_type: str | None,
        payload: dict | None,
        tts_allowed: bool,
        quality_guard: dict | None = None,
    ) -> dict:
        payload = payload or {}
        source = "twitch_viewer" if event_type == "twitch_chat_react" else "twitch_system"
        reason = "allowed"
        allow_tts = bool(tts_allowed)
        if not allow_tts:
            reason = "stream_tts_disabled"
        elif source == "twitch_viewer":
            allow_tts = False
            reason = "viewer_default_text_only"
        if allow_tts and len(str(text or "").strip()) > 180:
            allow_tts = False
            reason = "max_spoken_length"
        raw = str(payload.get("message_text") or payload.get("text") or "")
        if allow_tts and raw.strip() and self._stream_message_is_emote_only(raw):
            allow_tts = False
            reason = "emote_only_text_only"
        if allow_tts and isinstance(quality_guard, dict) and quality_guard.get("result") in {"twitch_text_reply", "suppress", "repair"}:
            allow_tts = False
            reason = f"quality_guard_{quality_guard.get('result')}"
        if source == "twitch_system" and event_type in {"twitch_raid", "twitch_follow", "twitch_sub", "twitch_cheer"} and tts_allowed:
            allow_tts = True
            reason = "important_stream_event"
        route = "stream_tts_reply" if allow_tts else "twitch_text_reply"
        decision = {
            "allow_tts": allow_tts,
            "route": route,
            "reason": reason,
            "source": source,
            "event_type": event_type or "",
            "max_spoken_length": 180,
        }
        print(
            f"[HEBE][STREAM_SPEECH_BUDGET] route={route} allow_tts={str(allow_tts).lower()} "
            f"reason={reason} source={source} event_type={event_type}",
            flush=True,
        )
        print(
            f"[HEBE][TTS_ROUTE] output_target={route} reason={reason} event_type={event_type}",
            flush=True,
        )
        return decision

    def _deliver_twitch_reply(self, text: str, *, event_type: str | None = None, payload: dict | None = None) -> None:
        """
        Entrega un reply al chat de Twitch.
        Si las policies del stream lo permiten, también lo hablamos por TTS.
        """
        twitch = getattr(self.runtime, "twitch", None)
        stream = getattr(self.runtime.state, "stream", None)
        payload = self._enrich_stream_payload(payload)
        is_spontaneous = event_type == "twitch_idle_prompt"
        is_simulated = bool((payload or {}).get("_simulated"))
        final_event_id_for_gate = str((payload or {}).get("event_id") or (payload or {}).get("message_id") or (payload or {}).get("assistant_message_id") or f"evt_{uuid.uuid4().hex}")
        def consume_spontaneous(reason: str) -> None:
            if is_spontaneous:
                service = getattr(self, "stream_spontaneity", None)
                if service is not None:
                    service.consume_opportunity(stream, payload, reason=reason)
        if event_type and event_type.startswith("twitch_"):
            raw_text = str((payload or {}).get("message_text") or (payload or {}).get("text") or "")
            username = str((payload or {}).get("user_login") or (payload or {}).get("username") or "")
            source = "twitch_viewer" if event_type == "twitch_chat_react" else "twitch_system"
            firewall = self._input_firewall_decision(
                source=source,
                text=raw_text or text,
                username=username,
                event_type=event_type,
                addressed_to_hebe=bool((payload or {}).get("direct_address_to_hebe") or (self._message_mentions_hebe(raw_text) if raw_text else False)),
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
        if bool((payload or {}).get("_force_skip_tts")):
            tts_allowed = False
        if event_type == "twitch_cheer":
            cheer_guard = self._cheer_anti_bait_guard(text, payload=payload)
            if not cheer_guard.get("passed"):
                viewer = str((payload or {}).get("display_name") or (payload or {}).get("user_login") or "chat")
                text = f"Gracias por los bits, {viewer}."
        quality_guard = self._stream_response_quality_guard(text, event_type=event_type, payload=payload)
        if quality_guard.get("result") == "repair":
            text = "Me quedo con la version corta: eso no sale al directo."
            quality_guard = self._stream_response_quality_guard(text, event_type=event_type, payload=payload)
        speech_budget = self._stream_speech_budget_decision(
            text,
            event_type=event_type,
            payload=payload,
            tts_allowed=tts_allowed,
            quality_guard=quality_guard,
        )
        tts_allowed = bool(speech_budget.get("allow_tts"))
        tts_route = {
            "output_target": OUTPUT_TARGET_STREAM_TTS if tts_allowed else OUTPUT_TARGET_TWITCH_CHAT,
            "route": speech_budget.get("route"),
            "reason": speech_budget.get("reason"),
            "event_type": event_type or "",
        }
        self._update_policy_trace_response(
            text,
            tts_route=tts_route,
            speech_budget=speech_budget,
            quality_guard=quality_guard,
        )
        if quality_guard.get("result") == "suppress":
            consume_spontaneous("quality_guard_exhausted")
            self._update_policy_trace_response(
                "",
                candidate_response=text,
                suppressed_response=text,
                suppress_reason="stream_response_quality_guard",
                output_route="suppress",
                public_sent=False,
                tts_sent=False,
            )
            print("[HEBE][OUTPUT_ROUTE_DECISION] route=suppress reason=stream_response_quality_guard public=false tts=false", flush=True)
            print("[HEBE][EVENT][TWITCH] suppressed reason=stream_response_quality_guard", flush=True)
            if event_type == "twitch_chat_react":
                self._record_twitch_pipeline_final(route="suppress", emitted=False, reason="stream_response_quality_guard")
            self._emit_final_response(
                event_id=final_event_id_for_gate,
                source="twitch",
                final_response=text,
                output_route=OutputRoute.SUPPRESS,
                output_targets=[],
                guard_result={"passed": False, "reason": "stream_response_quality_guard"},
                debug_payload=self._latest_response_debug_payload(),
            )
            return
        speaker_source = "twitch_viewer" if event_type == "twitch_chat_react" else "twitch_system"
        speaker_ok, speaker_reason = self._target_speaker_guard(
            text,
            source=speaker_source,
            speaker=str((payload or {}).get("display_name") or (payload or {}).get("username") or ""),
        )
        if not speaker_ok:
            consume_spontaneous(speaker_reason)
            self._update_policy_trace_response(
                "",
                candidate_response=text,
                suppressed_response=text,
                suppress_reason=speaker_reason,
                output_route="suppress",
                public_sent=False,
                tts_sent=False,
                target_speaker_guard_result={"passed": False, "reason": speaker_reason},
            )
            print(f"[HEBE][OUTPUT_ROUTE_DECISION] route=suppress reason={speaker_reason} public=false tts=false", flush=True)
            print(f"[HEBE][EVENT][TWITCH] suppressed reason={speaker_reason}", flush=True)
            if event_type == "twitch_chat_react":
                self._record_twitch_pipeline_final(route="suppress", emitted=False, reason=speaker_reason)
            self._emit_final_response(
                event_id=final_event_id_for_gate,
                source="twitch",
                final_response=text,
                output_route=OutputRoute.SUPPRESS,
                output_targets=[],
                guard_result={"passed": False, "reason": speaker_reason},
                debug_payload=self._latest_response_debug_payload(),
            )
            return
        route_policy = self._evaluate_twitch_chat_write_policy(
            text=text,
            event_type=event_type,
            payload=payload,
            stream=stream,
            tts_allowed=tts_allowed,
            quality_guard=quality_guard,
            local_debug_only=bool(is_simulated or (output_mode == "ui_only" and not is_spontaneous)),
        )
        if event_type == "twitch_chat_react":
            self._set_last_twitch_route_state(
                output_route=str(route_policy.get("route") or ""),
                should_generate=bool(route_policy.get("should_generate")),
                suppress_reason=str(route_policy.get("reason") or ""),
                budget_result=route_policy.get("budget_result"),
                thread_result=route_policy.get("thread_result"),
                presence_decision=(payload or {}).get("core_loop", {}).get("intervention") if isinstance((payload or {}).get("core_loop"), dict) else None,
            )
        text = str(route_policy.get("final_text") or text)
        tts_allowed = bool(route_policy.get("should_tts"))
        final_guard_decision = FinalGuardDecision.from_value(route_policy.get("final_guard_decision"))
        if final_guard_decision.action == "suppress":
            reason = str(route_policy.get("reason") or "final_guard_suppress")
            consume_spontaneous(reason)
            self._update_policy_trace_response(
                "",
                candidate_response=text,
                suppressed_response=text,
                suppress_reason=reason,
                output_route="suppress",
                public_sent=False,
                tts_sent=False,
                stream_persona_quality_result=route_policy.get("stream_persona_quality_result"),
            )
            if event_type == "twitch_chat_react":
                self._record_twitch_pipeline_final(route="suppress", emitted=False, reason=reason)
            if is_spontaneous and stream is not None and isinstance(getattr(stream, "last_proactive_decision", None), dict):
                suppressed = dict(stream.last_proactive_decision)
                suppressed.update({"selected_route": "suppress", "outcome": "suppressed", "blocked_reason": reason})
                stream.last_proactive_decision = suppressed
                log_jsonl_event("proactive_decisions", suppressed)
            self._emit_final_response(
                event_id=final_event_id_for_gate,
                source="spontaneity" if is_spontaneous else "twitch",
                final_response=text,
                output_route=OutputRoute.SUPPRESS,
                output_targets=[],
                guard_result=final_guard_decision.to_dict(),
                debug_payload=self._latest_response_debug_payload(),
            )
            return
        if not bool(route_policy.get("should_write_to_twitch")) and not is_spontaneous and not is_simulated and output_mode != "ui_only":
            self._update_policy_trace_response(
                "",
                candidate_response=text,
                suppressed_response=text if route_policy.get("route") == "suppress" else "",
                suppress_reason=str(route_policy.get("reason") or ""),
                output_route=str(route_policy.get("route") or "observe_only"),
                public_sent=False,
                tts_sent=False,
                reply_value_score=route_policy.get("value_score"),
                budget_result=route_policy.get("budget_result"),
                twitch_message_category=route_policy.get("twitch_message_category"),
                should_generate=route_policy.get("should_generate"),
                thread_result=route_policy.get("thread_result"),
                answer_depth_result=route_policy.get("answer_depth_result"),
                followup_question_guard_result=route_policy.get("followup_question_guard_result"),
                stream_persona_quality_result=route_policy.get("stream_persona_quality_result"),
                target_speaker_guard_result={"passed": True, "reason": speaker_reason},
            )
            print(f"[HEBE][EVENT][TWITCH] suppressed reason={route_policy.get('reason')}", flush=True)
            if event_type == "twitch_chat_react":
                self._record_twitch_pipeline_final(
                    route=str(route_policy.get("route") or "observe_only"),
                    emitted=False,
                    reason=str(route_policy.get("reason") or ""),
                )
            self._emit_final_response(
                event_id=final_event_id_for_gate,
                source="twitch",
                final_response=text,
                output_route=str(route_policy.get("route") or OutputRoute.OBSERVE_ONLY.value),
                output_targets=[],
                guard_result={"passed": False, "reason": str(route_policy.get("reason") or "")},
                debug_payload=self._latest_response_debug_payload(),
            )
            return
        input_id = str((payload or {}).get("event_id") or (payload or {}).get("message_id") or "")
        deduped, dedupe_reason = self._output_dedupe_suppressed(
            text=text,
            source=speaker_source,
            message_id=str((payload or {}).get("assistant_message_id") or ""),
            input_id=input_id,
        )
        if deduped:
            consume_spontaneous(dedupe_reason)
            self._update_policy_trace_response(
                "",
                candidate_response=text,
                suppressed_response=text,
                suppress_reason=dedupe_reason,
                output_route="suppress",
                public_sent=False,
                tts_sent=False,
            )
            print(f"[HEBE][OUTPUT_ROUTE_DECISION] route=suppress reason={dedupe_reason} public=false tts=false", flush=True)
            print(f"[HEBE][OUTPUT_DEDUPE] suppressed=true reason={dedupe_reason}", flush=True)
            if event_type == "twitch_chat_react":
                self._record_twitch_pipeline_final(route="suppress", emitted=False, reason=dedupe_reason)
            self._emit_final_response(
                event_id=final_event_id_for_gate,
                source="twitch",
                final_response=text,
                output_route=OutputRoute.SUPPRESS,
                output_targets=[],
                guard_result={"passed": False, "reason": dedupe_reason},
                debug_payload=self._latest_response_debug_payload(),
            )
            return
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
            intent_id = str((payload or {}).get("speech_intent_id") or "")
            loop = getattr(self, "stream_companion_loop", None)
            if is_spontaneous and intent_id and loop is not None and loop.owner_voice_active():
                loop.yield_intent(intent_id, reason="owner_resumed_before_tts")
                stream.speech_intent_state = loop.intent_manager.snapshot()
                self._emit_final_response(
                    event_id=final_event_id_for_gate,
                    source="spontaneity",
                    final_response=text,
                    output_route=OutputRoute.SUPPRESS,
                    output_targets=[],
                    guard_result={"passed": False, "reason": "owner_resumed_before_tts"},
                    debug_payload=self._latest_response_debug_payload(),
                )
                return
            stream.last_hebe_stream_speak_ts = time.time()
            if event_type == "twitch_idle_prompt":
                topic = (payload or {}).get("idle_topic")
                service = getattr(self, "stream_spontaneity", None)
                if service is not None:
                    service.record_idle_message(
                        stream,
                        text,
                        topic=topic,
                        used_fact_id=(payload or {}).get("used_fact_id"),
                    )
                    service.consume_opportunity(stream, payload, reason="emitted")
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
        final_event_id = input_id or str((payload or {}).get("assistant_message_id") or f"evt_{uuid.uuid4().hex}")
        gate_source = "spontaneity" if is_spontaneous else "simulation" if is_simulated else "twitch"
        advice_allowed, advice_reason = self._gameplay_comment_advice_allowed(text, event_type=event_type)
        if not advice_allowed:
            consume_spontaneous(advice_reason)
            print(f"[HEBE][GAME_ASSISTANCE_GUARD] allowed=false reason={advice_reason}", flush=True)
            self._emit_final_response(
                event_id=final_event_id,
                source=gate_source,
                final_response=text,
                output_route=OutputRoute.SUPPRESS,
                output_targets=[],
                guard_result={"passed": False, "action": "suppress", "violations": [advice_reason]},
                debug_payload=self._latest_response_debug_payload(),
            )
            return

        def speak_stream_once(final_text: str) -> None:
            intent_id = str((payload or {}).get("speech_intent_id") or "")
            loop = getattr(self, "stream_companion_loop", None)
            if intent_id and loop is not None:
                loop.mark_tts_committed(intent_id)
            safe_text = str(final_text or "").replace('"', '\\"')
            print(f"[HEBE][TTS] speaking output_target={OUTPUT_TARGET_STREAM_TTS} text=\"{safe_text}\"", flush=True)
            self._remember_tts_text(final_text)
            self.runtime.speak(final_text, emit_chat=False)
            self._remember_assistant_text(final_text, source=gate_source)

        def send_twitch_once(final_text: str) -> bool:
            if twitch is None or not twitch.is_available():
                print("[HEBE][EVENT][TWITCH] service not available, dropping chat reply", flush=True)
                return False
            if is_spontaneous:
                print("[HEBE][TWITCH][CHATBOT] send_message reason=spontaneity", flush=True)
            sent = twitch.send_message(final_text) is not False
            if not sent:
                outcome = getattr(twitch, "last_delivery_outcome", None)
                print(
                    "[HEBE][TWITCH_DELIVERY_OUTCOME] "
                    f"success=false outcome={outcome!r}",
                    flush=True,
                )
                return False
            if stream is not None and not is_spontaneous:
                self._record_twitch_public_reply(
                    stream=stream,
                    username=str((payload or {}).get("user_login") or (payload or {}).get("username") or ""),
                    category=str(route_policy.get("category") or "stream_event"),
                    thread_id=str(route_policy.get("thread_id") or ""),
                )
                self._get_troll_engagement_budget().record_engagement(
                    viewer=str((payload or {}).get("user_login") or (payload or {}).get("username") or ""),
                    text=str((payload or {}).get("message_text") or (payload or {}).get("text") or ""),
                )
            if is_spontaneous:
                self._record_spontaneous_twitch_chat_sent(final_text, payload)
            self._remember_assistant_text(final_text, source=gate_source)
            return True

        tts_gate_allowed = bool(
            tts_allowed and getattr(self.runtime.state, "tts_enabled", False)
            and not is_simulated and not bool((payload or {}).get("_async_tts_managed"))
        )
        if is_spontaneous:
            if output_mode == "silent":
                consume_spontaneous("stream_output_mode_silent")
                print("[HEBE][SPONTANEITY] skipped reason=stream_output_mode_silent", flush=True)
                self._emit_final_response(
                    event_id=final_event_id,
                    source="spontaneity",
                    final_response=text,
                    output_route=OutputRoute.SUPPRESS,
                    output_targets=[],
                    guard_result={"passed": False, "reason": "stream_output_mode_silent"},
                    debug_payload=self._latest_response_debug_payload(),
                )
                return
            gate_targets = list(targets)
            if not spontaneous_chat_allowed and OUTPUT_TARGET_LOCAL_UI not in gate_targets:
                gate_targets.insert(0, OUTPUT_TARGET_LOCAL_UI)
            gate_route = OutputRoute.TWITCH_TEXT_REPLY if OUTPUT_TARGET_TWITCH_CHAT in gate_targets else (
                OutputRoute.STREAM_TTS_REPLY if OUTPUT_TARGET_STREAM_TTS in gate_targets else OutputRoute.LOCAL_OWNER_REPLY
            )
            scene_timeline = getattr(self, "scene_timeline", None)
            scene_decision = (
                scene_timeline.revalidate((payload or {}).get("scene_guard"))
                if scene_timeline is not None
                else SimpleNamespace(valid=True, reason="scene_timeline_unavailable")
            )
            if not scene_decision.valid:
                consume_spontaneous(f"scene_invalidated:{scene_decision.reason}")
                opportunity_id = str((payload or {}).get("opportunity_id") or "")
                if opportunity_id:
                    opportunities = getattr(self, "spontaneous_opportunities", None)
                    if opportunities is not None:
                        opportunities.mark(
                            opportunity_id,
                            "invalidated",
                            reason=scene_decision.reason,
                            guard="SceneTimelineGuard",
                        )
                        if stream is not None:
                            stream.spontaneous_opportunities = opportunities.all_states()
                print(
                    "[HEBE][SCENE_REVALIDATION] "
                    f"decision=cancel reason={scene_decision.reason} stage=pre_final_emission_gate",
                    flush=True,
                )
                self._emit_final_response(
                    event_id=final_event_id,
                    source="spontaneity",
                    final_response=text,
                    output_route=OutputRoute.SUPPRESS,
                    output_targets=[],
                    guard_result={"passed": False, "reason": scene_decision.reason},
                    debug_payload=self._latest_response_debug_payload(),
                )
                return
            gate_result = self._emit_final_response(
                event_id=final_event_id,
                source="spontaneity",
                final_response=text,
                output_route=gate_route,
                output_targets=gate_targets,
                guard_result=final_guard_decision.to_dict(),
                debug_payload=self._latest_response_debug_payload(),
                send_twitch_fn=send_twitch_once if OUTPUT_TARGET_TWITCH_CHAT in gate_targets else None,
                speak_fn=speak_stream_once if OUTPUT_TARGET_STREAM_TTS in gate_targets and tts_gate_allowed else None,
            )
            if gate_result.get("emitted"):
                self._record_final_gameplay_comment(final_event_id, text, event_type=event_type, payload=payload)
                if (payload or {}).get("source") == "stream_companion_tick":
                    loop = getattr(self, "stream_companion_loop", None)
                    if loop is not None:
                        loop.record_emitted(
                            stream,
                            text,
                            route=str(gate_route.value if hasattr(gate_route, "value") else gate_route),
                            intent_id=str((payload or {}).get("speech_intent_id") or ""),
                        )
                    if stream is not None and isinstance(getattr(stream, "last_proactive_decision", None), dict):
                        log_jsonl_event("proactive_decisions", dict(stream.last_proactive_decision))
                try:
                    anchor_id = str((payload or {}).get("used_fact_id") or (payload or {}).get("anchor_id") or (payload or {}).get("idle_topic") or "").strip() or None
                    self._get_live_session_brain().observe_hebe_utterance(
                        text,
                        output_target=gate_targets,
                        input_type="spontaneity",
                        anchor_id=anchor_id,
                        topic=(payload or {}).get("idle_topic"),
                    )
                    if OUTPUT_TARGET_TWITCH_CHAT in gate_targets:
                        self._get_live_session_brain().consume_anchor(anchor_id)
                except Exception as exc:
                    print(f"[HEBE][LIVE_SESSION] hebe spontaneity record failed: {exc!r}", flush=True)
        elif is_simulated or output_mode == "ui_only":
            self._update_policy_trace_response(
                text,
                candidate_response=text,
                final_response=text,
                output_route="local_ui_debug_only",
                public_sent=False,
                tts_sent=False,
                reply_value_score=route_policy.get("value_score"),
                budget_result=route_policy.get("budget_result"),
                twitch_message_category=route_policy.get("twitch_message_category"),
                should_generate=route_policy.get("should_generate"),
                thread_result=route_policy.get("thread_result"),
                answer_depth_result=route_policy.get("answer_depth_result"),
                followup_question_guard_result=route_policy.get("followup_question_guard_result"),
                stream_persona_quality_result=route_policy.get("stream_persona_quality_result"),
                target_speaker_guard_result={"passed": True, "reason": speaker_reason},
            )
            gate_result = self._emit_final_response(
                event_id=final_event_id,
                source="simulation" if is_simulated else "twitch",
                final_response=text,
                output_route=OutputRoute.LOCAL_UI_DEBUG_ONLY,
                output_targets=[OUTPUT_TARGET_LOCAL_UI],
                guard_result={"passed": True},
                debug_payload=self._latest_response_debug_payload(),
            )
            if gate_result.get("emitted"):
                self._record_final_gameplay_comment(final_event_id, text, event_type=event_type, payload=payload)
                self._remember_assistant_text(text, source="simulation" if is_simulated else "twitch")
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
            if event_type == "twitch_chat_react":
                self._record_twitch_pipeline_final(route="suppress", emitted=False, reason="stream_output_mode_silent")
            self._emit_final_response(
                event_id=final_event_id,
                source="twitch",
                final_response=text,
                output_route=OutputRoute.SUPPRESS,
                output_targets=[],
                guard_result={"passed": False, "reason": "stream_output_mode_silent"},
                debug_payload=self._latest_response_debug_payload(),
            )
            return
        elif twitch is not None and twitch.is_available():
            try:
                gate_result = self._emit_final_response(
                    event_id=final_event_id,
                    source="twitch",
                    final_response=text,
                    output_route=OutputRoute.TWITCH_TEXT_REPLY if OUTPUT_TARGET_TWITCH_CHAT in targets else OutputRoute.STREAM_TTS_REPLY,
                    output_targets=targets,
                    guard_result=final_guard_decision.to_dict(),
                    debug_payload=self._latest_response_debug_payload(),
                    send_twitch_fn=send_twitch_once if OUTPUT_TARGET_TWITCH_CHAT in targets else None,
                    speak_fn=speak_stream_once if OUTPUT_TARGET_STREAM_TTS in targets and tts_gate_allowed else None,
                )
                if not gate_result.get("emitted"):
                    if event_type == "twitch_chat_react":
                        self._record_twitch_pipeline_final(
                            route="suppress",
                            emitted=False,
                            reason=str(gate_result.get("reason") or "final_emission_gate"),
                        )
                    return
                self._record_final_gameplay_comment(final_event_id, text, event_type=event_type, payload=payload)
                self._update_policy_trace_response(
                    text,
                    candidate_response=text,
                    final_response=text,
                    output_route=str(route_policy.get("route") or "twitch_text_reply"),
                    public_sent=True,
                    tts_sent=tts_allowed,
                    reply_value_score=route_policy.get("value_score"),
                    budget_result=route_policy.get("budget_result"),
                    twitch_message_category=route_policy.get("twitch_message_category"),
                    should_generate=route_policy.get("should_generate"),
                    thread_result=route_policy.get("thread_result"),
                    answer_depth_result=route_policy.get("answer_depth_result"),
                    followup_question_guard_result=route_policy.get("followup_question_guard_result"),
                    stream_persona_quality_result=route_policy.get("stream_persona_quality_result"),
                    target_speaker_guard_result={"passed": True, "reason": speaker_reason},
                )
                self._remember_assistant_text(text, source="twitch")
                if event_type == "twitch_chat_react":
                    self._record_twitch_pipeline_final(
                        route=str(route_policy.get("route") or "twitch_text_reply"),
                        emitted=True,
                        reason=str(route_policy.get("reason") or "public_reply_allowed"),
                        public_chat_sent=OUTPUT_TARGET_TWITCH_CHAT in targets,
                        tts_sent=tts_allowed,
                    )
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
            if event_type == "twitch_chat_react":
                self._record_twitch_pipeline_final(route="suppress", emitted=False, reason="twitch_service_unavailable")

    def _gameplay_comment_advice_allowed(self, text: str, *, event_type: str | None) -> tuple[bool, str]:
        if event_type in {"twitch_raid", "twitch_outgoing_raid", "twitch_cheer", "twitch_follow", "twitch_follow_batch", "twitch_sub"}:
            return True, "not_gameplay_comment"
        stream = self._get_stream_state()
        service = getattr(self, "game_intelligence", None)
        game = str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or "").strip() if stream else ""
        if service is None or not game:
            return True, "no_game_context"
        progress = self._canonical_progress_projection(stream)
        return service.advice_guard.allow(text, mode=default_assistance_mode(progress), explicit_owner_request=False)

    def _record_final_gameplay_comment(self, comment_id: str, text: str, *, event_type: str | None, payload: dict | None) -> None:
        if event_type in {"twitch_raid", "twitch_outgoing_raid", "twitch_cheer", "twitch_follow", "twitch_follow_batch", "twitch_sub"}:
            return
        stream = self._get_stream_state()
        service = getattr(self, "game_intelligence", None)
        game = str(getattr(stream, "current_game", None) or getattr(stream, "current_category", None) or "").strip() if stream else ""
        if service is None or not game:
            return
        try:
            progress = self._canonical_progress_projection(stream)
            evidence = [
                str((payload or {}).get("message_text") or ""),
                *[str(item) for item in (payload or {}).get("specific_context_anchors") or []],
            ]
            anchor_evidence = dict((payload or {}).get("anchor_evidence") or {})
            scene_fact_id = str(
                anchor_evidence.get("anchor_id")
                or (payload or {}).get("used_fact_id")
                or ""
            )
            scene_fact_ids = []
            if scene_fact_id and not bool(anchor_evidence.get("superseded")):
                scene_timeline = getattr(self, "scene_timeline", None)
                scene_decision = scene_timeline.revalidate((payload or {}).get("scene_guard")) if scene_timeline is not None else SimpleNamespace(valid=True)
                if scene_decision.valid:
                    scene_fact_ids.append(scene_fact_id)
            identity=self.game_v2_repository.resolve_identity(game)
            facts=self.game_v2_repository.knowledge(identity.game_id)
            service.diagnostics.lookup_used=bool(facts)
            service.diagnostics.current_comment_fact_ids=[str(item.get("id") or "") for item in facts]
            service.diagnostics.current_comment_mode="informed_observation" if facts else "contextual_reaction"
            print(
                f"[HEBE][COMMENT_PROVENANCE] comment_id={comment_id} mode={service.diagnostics.current_comment_mode} "
                f"scene_fact_ids={scene_fact_ids} game_fact_ids={service.diagnostics.current_comment_fact_ids} "
                f"lookup_used={str(bool(facts)).lower()} spoiler_guard=passed advice_mode={default_assistance_mode(progress).value}",
                flush=True,
            )
        except Exception as exc:
            print(f"[HEBE][COMMENT_PROVENANCE] comment_id={comment_id} mode=unknown lookup_used=false error={type(exc).__name__}", flush=True)

    def _canonical_progress_projection(self, stream) -> SimpleNamespace:
        run_id=str(getattr(stream,"active_game_run_id","") or "") if stream is not None else ""
        state=self.game_run_service.state(run_id) if run_id and getattr(self,"game_run_service",None) is not None else {}
        return SimpleNamespace(
            playthrough_type=str(state.get("playthrough_type") or getattr(stream,"current_playthrough_type",None) or "unknown"),
            spoiler_policy=str(state.get("spoiler_policy") or getattr(stream,"spoiler_policy",None) or "strict"),
        )


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
        input_id = ""
        current_event = getattr(self, "_current_input_event", None)
        if current_event is not None:
            input_id = str(getattr(current_event, "timestamp", "") or getattr(current_event, "raw_text", "") or "")
        deduped, dedupe_reason = self._output_dedupe_suppressed(text=text, source=input_type, input_id=input_id)
        if deduped:
            print(f"[HEBE][OUTPUT_DEDUPE] suppressed=true reason={dedupe_reason}", flush=True)
            return
        voice_enabled = (
            self._stream_tts_output_enabled_for_event(input_type)
            if output_target == OUTPUT_TARGET_STREAM_TTS
            else self._local_tts_output_enabled()
        )
        tts_can_speak = bool(voice_enabled and getattr(self.runtime.state, "tts_enabled", False))
        if declare_route:
            targets = [OUTPUT_TARGET_LOCAL_UI] if emit_ui else []
            if tts_can_speak:
                targets.append(output_target)
            self._declare_output_route(
                input_type=input_type,
                targets=targets or [OUTPUT_TARGET_LOCAL_UI],
                reason="voice_reply",
            )
        targets_for_gate = [OUTPUT_TARGET_LOCAL_UI] if emit_ui else []
        if tts_can_speak:
            targets_for_gate.append(output_target)

        def speak_once(final_text: str) -> None:
            safe_text = str(final_text or "").replace('"', '\\"')
            print(f"[HEBE][TTS] speaking output_target={output_target} text=\"{safe_text}\"", flush=True)
            self._remember_tts_text(final_text)
            self.runtime.speak(final_text, emit_chat=False)
            self._remember_assistant_text(final_text, source=input_type)

        gate_result = self._emit_final_response(
            event_id=input_id,
            source=input_type,
            final_response=text,
            output_route=OutputRoute.STREAM_TTS_REPLY if output_target == OUTPUT_TARGET_STREAM_TTS and tts_can_speak else OutputRoute.LOCAL_OWNER_REPLY,
            output_targets=targets_for_gate or [OUTPUT_TARGET_LOCAL_UI],
            guard_result={"passed": True},
            debug_payload=self._latest_response_debug_payload(),
            runtime_context=(
                HebeLiveRuntimeContext.STREAM_PUBLIC
                if output_target == OUTPUT_TARGET_STREAM_TTS
                else HebeLiveRuntimeContext.OWNER_LOCAL
            ),
            speak_fn=speak_once if tts_can_speak else None,
        )
        if not gate_result.get("emitted"):
            return
        if emit_ui:
            self._remember_assistant_text(text, source=input_type)
        try:
            targets_for_brain = [OUTPUT_TARGET_LOCAL_UI] if emit_ui else []
            if tts_can_speak:
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
        if not voice_enabled:
            print("[HEBE][TTS] skipped reason=stream_output_mode", flush=True)
            return

    def _latest_response_debug_payload(self) -> dict:
        route_state = dict(getattr(self, "_last_twitch_route_state", {}) or {})
        payload = {}
        if route_state:
            viewer = str(route_state.get("username") or "")
            payload.update({"intended_recipient": viewer, "source_viewer": viewer, "user_login": viewer})
        synthesizer = getattr(self, "response_synthesizer", None)
        debug_contract = getattr(synthesizer, "last_response_debug_contract", None)
        if isinstance(debug_contract, dict) and debug_contract:
            payload["debug_contract"] = debug_contract
        return payload

    def _deliver_manual_reply(self, text: str, *, source: str) -> None:
        if source == "ui":
            deduped, dedupe_reason = self._output_dedupe_suppressed(text=text, source="ui")
            if deduped:
                print(f"[HEBE][OUTPUT_DEDUPE] suppressed=true reason={dedupe_reason}", flush=True)
                print(f"[HEBE][FINAL_EMISSION_GATE] blocked_candidate=true reason={dedupe_reason}", flush=True)
                return
            message_id = f"msg_{uuid.uuid4().hex}"
            self._declare_output_route(
                input_type="ui_typed_input",
                targets=[OUTPUT_TARGET_LOCAL_UI],
                reason="typed_input_reply",
            )
            gate_result = self._emit_final_response(
                event_id=message_id,
                source="ui",
                final_response=text,
                output_route=OutputRoute.LOCAL_OWNER_REPLY,
                output_targets=[OUTPUT_TARGET_LOCAL_UI],
                guard_result={"passed": True},
                debug_payload={**{"message_id": message_id}, **self._latest_response_debug_payload()},
            )
            if not gate_result.get("emitted"):
                return
            self._remember_assistant_text(text, source="ui")
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
                    res = self.ingest_owner_stt(
                        str(raw_voice_command), allow_wakeword_prompt=True,
                        stt_metadata=getattr(raw_voice_command, "metadata", None),
                    )
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
            res = self.ingest_owner_stt(
                str(raw_voice_command), allow_wakeword_prompt=True,
                stt_metadata=getattr(raw_voice_command, "metadata", None),
            )
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
