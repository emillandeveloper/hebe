from __future__ import annotations

import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import dateparser

from app.cognitive.temporal.models import TemporalFacts


MONTHS_ES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


class FastParser:
    """
    Capa 1: parser rápido.

    Filosofía:
    - Lo que el usuario dice LITERALMENTE manda siempre.
    - dateparser es un asistente para referencias relativas (mañana,
      pasado mañana, el jueves que viene), no un intérprete libre.
    - Si el usuario da un día fuera de rango (ej: 32), se marca como
      inválido y rules_engine responde sin subir al LLM.
    """

    def __init__(self, timezone_name: str = "Europe/Madrid"):
        self.timezone_name = timezone_name
        self.tz = ZoneInfo(timezone_name)

    def parse(self, text: str, now: Optional[datetime] = None) -> Optional[TemporalFacts]:
        now = now or datetime.now(self.tz)
        raw = (text or "").strip().lower()
        print(f"[HEBE][TEMPORAL][FAST] raw={raw!r}", flush=True)

        if not raw:
            return None

        literals = self._extract_literals(raw)
        print(f"[HEBE][TEMPORAL][FAST] literals={literals!r}", flush=True)

        if not self._has_any_literal(literals):
            print("[HEBE][TEMPORAL][FAST] no literals -> None", flush=True)
            return None

        # Día fuera de rango: facts marcados como inválidos
        if literals["invalid_day"]:
            print("[HEBE][TEMPORAL][FAST] invalid day -> facts with invalid marker", flush=True)
            facts = TemporalFacts(
                title=self._infer_title(raw),
                source="fast_parser",
                confidence=1.0,
                notes=["invalid_day_value"],
            )
            if literals["hour"] is not None:
                facts.hour = literals["hour"]
                facts.minute = literals["minute"] if literals["minute"] is not None else 0
            if literals["month"] is not None:
                facts.month = literals["month"]
            return facts

        # Construir facts a partir de los literals
        facts = TemporalFacts(
            title=self._infer_title(raw),
            source="fast_parser",
            confidence=0.9,
            day=literals["day"],
            month=literals["month"],
            year=literals["year"],
            hour=literals["hour"],
            minute=literals["minute"] if literals["minute"] is not None else (0 if literals["hour"] is not None else None),
            relative_day_offset=literals["relative_day_offset"],
        )

        # Si hay referencia relativa al día, pedir a dateparser que la resuelva
        if facts.relative_day_offset is not None:
            relative_dt = self._resolve_relative_with_dateparser(raw, now)
            if relative_dt is not None:
                facts.day = relative_dt.day
                facts.month = relative_dt.month
                facts.year = relative_dt.year

        if not self._has_any_useful_field(facts):
            return None

        print(f"[HEBE][TEMPORAL][FAST] parsed_facts={facts!r}", flush=True)
        return facts

    # =========================
    # Extracción literal
    # =========================

    def _extract_literals(self, raw: str) -> dict:
        result = {
            "day": None,
            "month": None,
            "year": None,
            "hour": None,
            "minute": None,
            "is_afternoon": False,
            "relative_day_offset": None,
            "invalid_day": False,
        }

        m = re.search(r"\bel\s+(\d{1,2})\b", raw) or re.search(r"\bd[ií]a\s+(\d{1,2})\b", raw)
        if m:
            day_value = int(m.group(1))
            if 1 <= day_value <= 31:
                result["day"] = day_value
            else:
                result["invalid_day"] = True

        for name, value in MONTHS_ES.items():
            if re.search(rf"\b{name}\b", raw):
                result["month"] = value
                break

        m = re.search(r"\b(20\d{2})\b", raw)
        if m:
            result["year"] = int(m.group(1))

        result["is_afternoon"] = any(
            x in raw for x in ["de la tarde", "de la noche", " tarde", " noche", "pm"]
        )

        if "hoy" in raw:
            result["relative_day_offset"] = 0
        elif re.search(r"\bpasado\s+ma[ñn]ana\b", raw):
            result["relative_day_offset"] = 2
        elif re.search(r"\bma[ñn]ana\b", raw) and "de la ma" not in raw and "por la ma" not in raw:
            result["relative_day_offset"] = 1

        hour, minute = self._extract_time_literals(raw, result["is_afternoon"])
        result["hour"] = hour
        result["minute"] = minute

        return result

    def _extract_time_literals(
        self,
        raw: str,
        is_afternoon: bool,
    ) -> tuple[Optional[int], Optional[int]]:
        def _adjust(h: int) -> int:
            if is_afternoon and 1 <= h <= 11:
                return h + 12
            return h

        # HH:MM o HH.MM
        m = re.search(r"\b(\d{1,2})[:.](\d{2})\b", raw)
        if m:
            h = int(m.group(1))
            mm = int(m.group(2))
            if 0 <= h <= 23 and 0 <= mm <= 59:
                return _adjust(h), mm

        # "las X y media"
        m = re.search(r"\b(?:a\s+)?las\s+(\d{1,2})\s+y\s+media\b", raw)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return _adjust(h), 30

        # "las X y Y"
        m = re.search(r"\b(?:a\s+)?las\s+(\d{1,2})\s+y\s+(\d{1,2})\b", raw)
        if m:
            h = int(m.group(1))
            mm = int(m.group(2))
            if 0 <= h <= 23 and 0 <= mm <= 59:
                return _adjust(h), mm

        # "X de la tarde/noche/mañana"
        m = re.search(r"\b(\d{1,2})\s+de\s+la\s+(?:tarde|noche|ma[ñn]ana)\b", raw)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return _adjust(h), 0

        # "las X"
        m = re.search(r"\b(?:a\s+)?las\s+(\d{1,2})\b", raw)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return _adjust(h), 0

        # mediodía / medianoche
        if re.search(r"\bmedi[oa]d[ií]a\b", raw):
            return 12, 0
        if re.search(r"\bmedianoche\b", raw):
            return 0, 0

        return None, None

    def _resolve_relative_with_dateparser(
        self,
        raw: str,
        now: datetime,
    ) -> Optional[datetime]:
        """Solo se usa para resolver 'mañana', 'pasado mañana', 'el jueves que viene'."""
        settings = {
            "TIMEZONE": self.timezone_name,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": now,
        }

        try:
            parsed_dt = dateparser.parse(raw, languages=["es"], settings=settings)
        except Exception as e:
            print(f"[HEBE][TEMPORAL][FAST] dateparser exception: {e!r}", flush=True)
            return None

        return parsed_dt

    # =========================
    # Helpers
    # =========================

    def _infer_title(self, text: str) -> str:
        if "psicóloga" in text or "psicologa" in text:
            return "Psicóloga"
        if "médico" in text or "medico" in text:
            return "Médico"
        if "dentista" in text:
            return "Dentista"
        if "cita" in text:
            return "Cita"
        return "Cita"

    def _has_any_literal(self, literals: dict) -> bool:
        return literals["invalid_day"] or any(
            literals[k] is not None
            for k in ("day", "month", "year", "hour", "relative_day_offset")
        )

    def _has_any_useful_field(self, facts: TemporalFacts) -> bool:
        return any(
            v is not None
            for v in (
                facts.day,
                facts.month,
                facts.year,
                facts.hour,
                facts.minute,
                facts.relative_day_offset,
                facts.weekday,
            )
        )
