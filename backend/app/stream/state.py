# backend/app/stream/state.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StreamPolicies:
    require_wakeword_for_actions: bool = True
    allow_tts_replies: bool = True
    allow_tts_idle_prompts: bool = False
    allow_tts_event_replies: bool = True
    allow_tts_raid_thanks: bool = True
    allow_chat_commands: bool = True
    allow_auto_raid_messages: bool = True
    allow_auto_shoutouts: bool = True
    allow_auto_follow_messages: bool = False

@dataclass
class StreamSessionState:
    enabled: bool = False
    presence_mode: str = "reactive"  # silent | reactive | companion | show
    presence_mode_explicit: bool = False
    armed: bool = False
    armed_until_ts: float = 0.0
    arm_timeout_sec: float = 8.0

    channel_name: str = ""
    bot_username: str = "JotunBot"

    last_voice_target_user: Optional[str] = None
    last_voice_event: Optional[str] = None
    last_voice_event_ts: float = 0.0
    last_voice_summary: Optional[str] = None
    last_voice_raw_transcript: Optional[str] = None
    last_voice_normalized_command: Optional[str] = None
    last_voice_command_intent: Optional[str] = None
    last_voice_command_target: Optional[str] = None
    last_voice_command_status: Optional[str] = None
    last_voice_command_confidence: float = 0.0
    leo_mood_hint: Optional[str] = None
    last_event_id: Optional[str] = None
    last_chat_activity_ts: float = 0.0
    last_hebe_stream_speak_ts: float = 0.0
    recent_chat_messages: list[dict] = field(default_factory=list)
    recent_active_users: list[str] = field(default_factory=list)
    recent_chat_topics: list[str] = field(default_factory=list)
    recent_chat_summary: Optional[str] = None

    is_live: bool = False
    live_status_known: bool = False
    live_test_override: bool = False
    current_game: Optional[str] = None
    current_category: Optional[str] = None
    current_stream_title: Optional[str] = None
    current_tags: list[str] = field(default_factory=list)
    current_playthrough_type: Optional[str] = None
    current_challenge: Optional[str] = None
    current_stream_slot: Optional[str] = None
    bilingual_mode: bool = False
    language_mode: Optional[str] = None
    spoiler_policy: str = "no_spoilers"
    stream_started_at: Optional[str] = None
    stream_context_updated_ts: float = 0.0
    current_run_objective: Optional[str] = None
    current_run_location: Optional[str] = None
    current_run_phase: Optional[str] = None
    recent_run_context_facts: list[dict] = field(default_factory=list)
    completed_run_markers: list[str] = field(default_factory=list)
    title_context_markers: list[str] = field(default_factory=list)
    title_context_updated_ts: float = 0.0
    run_context_updated_ts: float = 0.0
    run_context_source: Optional[str] = None
    last_stream_live_transition_ts: float = 0.0
    last_stream_live_transition: Optional[str] = None
    stream_spontaneity_grace_until_ts: float = 0.0
    last_stream_spontaneity_preview_ts: float = 0.0
    last_stream_spontaneity_blocked_reason: Optional[str] = None
    idle_spontaneity_enabled: bool = True
    recent_idle_messages: list[dict] = field(default_factory=list)
    recent_style_motifs: list[dict] = field(default_factory=list)
    idle_prompts_sent_stream: int = 0
    last_raid_event: Optional[dict] = None
    last_shoutout_target: Optional[str] = None
    last_shoutout_ts: float = 0.0
    last_shoutout_error: Optional[str] = None
    shoutout_cooldowns: dict[str, float] = field(default_factory=dict)
    last_stream_context_error: Optional[str] = None
    active_stream_session_id: Optional[int] = None

    normal_start_time: str = "19:00"
    pre_stream_reminder_1: str = "18:30"
    pre_stream_reminder_2: str = "18:50"
    no_stream_today_date: Optional[str] = None
    stream_delay_minutes: int = 0
    routine_sent_keys: set[str] = field(default_factory=set)

    cooldowns: dict[str, float] = field(default_factory=dict)
    processed_event_ids: set[str] = field(default_factory=set)

    policies: StreamPolicies = field(default_factory=StreamPolicies)
