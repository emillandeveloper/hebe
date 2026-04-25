from __future__ import annotations

from typing import Any, Optional

from app.cognitive.temporal.models import TemporalFacts


class TemporalFactsNormalizer:
    """
    Valida y normaliza hechos temporales ya extraídos.

    No extrae información desde texto libre y no interpreta contexto temporal.
    RulesEngine sigue siendo la capa que decide futuro/pasado/ambigüedad.
    """

    def normalize(
        self,
        data: TemporalFacts | dict[str, Any] | None,
        *,
        source: str,
    ) -> Optional[TemporalFacts]:
        if data is None:
            return None

        read = self._reader(data)

        hour = self._bounded_int(read("hour"), 0, 23)
        minute = self._bounded_int(read("minute"), 0, 59)
        if hour is not None and minute is None:
            minute = 0

        facts = TemporalFacts(
            day=self._bounded_int(read("day"), 1, 31),
            month=self._bounded_int(read("month"), 1, 12),
            year=self._bounded_int(read("year"), 1900, 2200),
            hour=hour,
            minute=minute,
            relative_day_offset=self._bounded_int(read("relative_day_offset"), -7, 365),
            weekday=self._bounded_int(read("weekday"), 0, 6),
            weekday_is_next=bool(read("weekday_is_next") or False),
            title=self._normalize_title(read("title")),
            source=source,
            confidence=self._confidence(read("confidence")),
            notes=list(read("notes") or []),
        )

        if not self._has_temporal_field(facts):
            return None

        return facts

    def _reader(self, data: TemporalFacts | dict[str, Any]):
        if isinstance(data, TemporalFacts):
            return lambda key: getattr(data, key, None)
        return lambda key: data.get(key)

    def _bounded_int(self, value: Any, minimum: int, maximum: int) -> Optional[int]:
        if value is None:
            return None
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        if minimum <= number <= maximum:
            return number
        return None

    def _confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.5
        return max(0.0, min(1.0, confidence))

    def _normalize_title(self, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            return "Cita"

        cleaned = value.strip()
        aliases = {
            "psicologa": "Psicóloga",
            "psicóloga": "Psicóloga",
            "medico": "Médico",
            "médico": "Médico",
            "dentista": "Dentista",
            "cita": "Cita",
        }
        return aliases.get(cleaned.lower(), cleaned[:60])

    def _has_temporal_field(self, facts: TemporalFacts) -> bool:
        return any(
            value is not None
            for value in (
                facts.day,
                facts.month,
                facts.year,
                facts.hour,
                facts.minute,
                facts.relative_day_offset,
                facts.weekday,
            )
        )
