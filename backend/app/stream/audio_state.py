from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class EffectiveStreamAudioState:
    configured: bool
    engine_ready: bool
    route_enabled: bool
    muted: bool
    actual_can_speak: bool
    blocked_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def resolve(
        cls,
        *,
        configured: bool,
        engine_ready: bool,
        route_enabled: bool,
        muted: bool = False,
    ) -> "EffectiveStreamAudioState":
        reason = ""
        if not configured:
            reason = "global_tts_disabled"
        elif not engine_ready:
            reason = "tts_engine_not_ready"
        elif not route_enabled:
            reason = "stream_tts_route_disabled"
        elif muted:
            reason = "stream_voice_muted"
        return cls(configured, engine_ready, route_enabled, muted, not bool(reason), reason)
