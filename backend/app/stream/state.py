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
    armed: bool = False
    armed_until_ts: float = 0.0
    arm_timeout_sec: float = 8.0

    channel_name: str = ""
    bot_username: str = "JotunBot"

    last_voice_target_user: Optional[str] = None
    last_event_id: Optional[str] = None

    cooldowns: dict[str, float] = field(default_factory=dict)
    processed_event_ids: set[str] = field(default_factory=set)

    policies: StreamPolicies = field(default_factory=StreamPolicies)