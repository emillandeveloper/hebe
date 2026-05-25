# backend/app/stream/state.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StreamPolicies:
    require_wakeword_for_actions: bool = True
    allow_tts_replies: bool = True
    allow_chat_commands: bool = True
    allow_auto_raid_messages: bool = True
    allow_auto_shoutouts: bool = True
    allow_auto_follow_messages: bool = False

@dataclass
class StreamSessionState:
    enabled: bool = False
    presence_mode: str = "reactive"  # silent | reactive | companion | show
    armed: bool = False
    armed_until_ts: float = 0.0
    arm_timeout_sec: float = 8.0

    channel_name: str = ""
    bot_username: str = "JotunBot"

    last_voice_target_user: Optional[str] = None
    last_voice_event: Optional[str] = None
    last_voice_event_ts: float = 0.0
    leo_mood_hint: Optional[str] = None
    last_event_id: Optional[str] = None
    last_chat_activity_ts: float = 0.0
    last_hebe_stream_speak_ts: float = 0.0

    normal_start_time: str = "19:00"
    pre_stream_reminder_1: str = "18:30"
    pre_stream_reminder_2: str = "18:50"
    no_stream_today_date: Optional[str] = None
    stream_delay_minutes: int = 0
    routine_sent_keys: set[str] = field(default_factory=set)

    cooldowns: dict[str, float] = field(default_factory=dict)
    processed_event_ids: set[str] = field(default_factory=set)

    policies: StreamPolicies = field(default_factory=StreamPolicies)
