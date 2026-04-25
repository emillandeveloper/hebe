from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from dateparser.search import search_dates

from app.cognitive.temporal.models import TemporalSignals


class FastParser:
    """
    Detector barato de señales temporales.

    No extrae day/month/hour/minute.
    No decide reglas de negocio.
    No usa regex ni listas hardcodeadas como lógica temporal.
    """

    def __init__(self, timezone_name: str = "Europe/Madrid"):
        self.timezone_name = timezone_name
        self.tz = ZoneInfo(timezone_name)

    def parse(self, text: str, now: Optional[datetime] = None) -> TemporalSignals:
        now = now or datetime.now(self.tz)
        raw = (text or "").strip()

        if not raw:
            return TemporalSignals()

        settings = {
            "TIMEZONE": self.timezone_name,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": now,
        }

        try:
            matches = search_dates(raw, languages=["es"], settings=settings) or []
        except Exception as exc:
            return TemporalSignals(
                has_temporal_signal=False,
                notes=[f"dateparser_error:{type(exc).__name__}"],
            )

        return TemporalSignals(
            has_temporal_signal=bool(matches),
            notes=[f"dateparser_matches:{len(matches)}"],
        )
