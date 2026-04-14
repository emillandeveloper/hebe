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

    def set_last_raid(self, *, username: str, viewer_count: int = 0) -> None:
        self.last_raid = LastRaid(
            username=username,
            viewer_count=viewer_count,
            ts=time.time(),
        )

    def set_last_follow(self, *, username: str) -> None:
        self.last_follow_username = username

    def set_last_sub(self, *, username: str) -> None:
        self.last_sub_username = username