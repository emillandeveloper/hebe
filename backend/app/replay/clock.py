from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...
    def monotonic(self) -> float: ...


class SystemClock:
    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass(slots=True)
class ScenarioClock:
    _timestamp: float
    _monotonic: float = 0.0

    @classmethod
    def from_value(cls, value: str | int | float) -> "ScenarioClock":
        if isinstance(value, (int, float)):
            return cls(float(value))
        text = str(value or "").strip()
        if not text:
            raise ValueError("initial_time is required")
        return cls(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())

    def now(self) -> float:
        return self._timestamp

    def monotonic(self) -> float:
        return self._monotonic

    def iso(self) -> str:
        return datetime.fromtimestamp(self._timestamp, timezone.utc).isoformat()

    def advance(self, seconds: float) -> float:
        amount = max(0.0, float(seconds))
        self._timestamp += amount
        self._monotonic += amount
        return self._timestamp

    def move_to(self, value: str | int | float) -> float:
        target = self.from_value(value).now()
        if target < self._timestamp:
            raise ValueError("scenario time cannot move backwards")
        return self.advance(target - self._timestamp)


_DURATION = re.compile(r"^\+(?P<amount>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h|d)$", re.I)


def resolve_event_time(value: str | int | float, *, initial: float, previous: float) -> float:
    if isinstance(value, (int, float)):
        target = float(value)
    else:
        text = str(value or "").strip()
        match = _DURATION.match(text)
        if match:
            factor = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}[match.group("unit").lower()]
            # Relative offsets are from scenario start. This keeps fixtures
            # stable when events are inserted between existing events.
            target = initial + float(match.group("amount")) * factor
        else:
            target = ScenarioClock.from_value(text).now()
    if target < previous:
        raise ValueError(f"event time moved backwards: {target} < {previous}")
    return target
