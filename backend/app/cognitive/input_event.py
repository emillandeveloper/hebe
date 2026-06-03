from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class InputEvent:
    source: str
    raw_text: str
    normalized_text: str
    user_id: str | None = None
    username: str | None = None
    is_voice: bool = False
    is_stream_context: bool = False
    timestamp: float = field(default_factory=time.time)
    stt_metadata: dict[str, Any] = field(default_factory=dict)

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "username": self.username,
            "is_voice": self.is_voice,
            "is_stream_context": self.is_stream_context,
            "timestamp": self.timestamp,
            "stt_metadata": self.stt_metadata,
        }
