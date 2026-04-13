from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class LastRaid:
    username: str
    viewer_count: int
    ts: float


@dataclass
class TwitchEventMemory:
    last_raid: Optional[LastRaid] = None
    last_follow_username: Optional[str] = None
    last_sub_username: Optional[str] = None