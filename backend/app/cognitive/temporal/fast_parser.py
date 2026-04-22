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
    Capa 1: Parser rápido usando dateparser.

    Filosofía:
    - si dateparser lo entiende con confianza, devolvemos TemporalFacts
    - si dateparser no lo entiende o tiene dudas, devolvemos None y que suba a la capa LLM
    - NUNCA inventamos campos que el usuario no dijo

    Importante:
    - dateparser a veces "rellena" campos por defecto (ej: día actual si no lo dice el usuario).
      Eso NO nos sirve directamente: necesitamos saber qué campos venían realmente del usuario.
      Por eso cruzamos el resultado con una extracción de texto para detectar qué mencionó.
    """

    def __init__(self, timezone_name: str = "Europe/Madrid"):
        self.timezone_name = timezone_name
        self.tz = ZoneInfo(timezone_name)

    def parse(self, text: str, now: Optional[datetime] = None) -> Optional[TemporalFacts]:
        """
        Devuelve TemporalFacts si dateparser extrae algo útil.
        None si no hay suficiente información temporal.
        """
        now = now or datetime.now(self.tz)
        raw = (text or "").strip().lower()
        if not raw:
            return None

        # Detectar qué tipo de información temporal menciona realmente el usuario
        mentions = self._detect_mentions(raw)

        # Si el usuario no ha mencionado NADA temporal, dateparser no nos sirve
        if not any(mentions.values()):
            return None

        parsed_dt = dateparser.parse(
            raw,
            languages=["es"],
            settings={
                "TIMEZONE": self.timezone_name,
                "RETURN_AS_TIMEZONE_AWARE": True,
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": now,
            },
        )

        # Si dateparser no ha podido, delegar a LLM
        if parsed_dt is None:
            return None

        # Construir facts SOLO con los campos que realmente mencionó el usuario
        facts = TemporalFacts(
            title=self._infer_title(raw),
            source="fast_parser",
            confidence=0.9,
        )

        if mentions["day"]:
            facts.day = parsed_dt.day
        if mentions["month"]:
            facts.month = parsed_dt.month
        if mentions["year"]:
            facts.year = parsed_dt.year
        if mentions["time"]:
            facts.hour = parsed_dt.hour
            facts.minute = parsed_dt.minute

        # Referencias relativas
        if "hoy" in raw:
            facts.relative_day_offset = 0
        elif re.search(r"\bpasado\s+ma[ñn]ana\b", raw):
            facts.relative_day_offset = 2
        elif re.search(r"\bma[ñn]ana\b", raw) and "de la ma" not in raw and "por la ma" not in raw:
            facts.relative_day_offset = 1

        # Si dateparser detectó pasado mañana o mañana, ya trajo el día correcto
        if facts.relative_day_offset is not None and mentions["time"]:
            facts.day = parsed_dt.day
            facts.month = parsed_dt.month
            facts.year = parsed_dt.year

        # Si no extrajo nada útil, señal de que hay que subir a LLM
        if not self._has_any_useful_field(facts):
            return None

        return facts

    # =========================
    # Helpers
    # =========================

    def _detect_mentions(self, text: str) -> dict[str, bool]:
        """
        Detecta qué tipo de información temporal menciona el texto.
        NO extrae el valor: solo marca si el usuario habló de ello.
        """
        return {
            "day": self._mentions_day(text),
            "month": self._mentions_month(text),
            "year": self._mentions_year(text),
            "time": self._mentions_time(text),
        }

    def _mentions_day(self, text: str) -> bool:
        # "el 22", "día 22", "el lunes", "mañana", "hoy", "pasado mañana"
        if re.search(r"\bel\s+\d{1,2}\b", text):
            return True
        if re.search(r"\bd[ií]a\s+\d{1,2}\b", text):
            return True
        if re.search(r"\b(lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)\b", text):
            return True
        if re.search(r"\b(hoy|ma[ñn]ana|pasado\s+ma[ñn]ana|ayer)\b", text):
            return True
        return False

    def _mentions_month(self, text: str) -> bool:
        for name in MONTHS_ES.keys():
            if re.search(rf"\b{name}\b", text):
                return True
        if re.search(r"\bmes\s+que\s+viene\b", text):
            return True
        if re.search(r"\bpr[oó]ximo\s+mes\b", text):
            return True
        return False

    def _mentions_year(self, text: str) -> bool:
        if re.search(r"\b20\d{2}\b", text):
            return True
        if re.search(r"\ba[ñn]o\s+que\s+viene\b", text):
            return True
        if re.search(r"\bpr[oó]ximo\s+a[ñn]o\b", text):
            return True
        return False

    def _mentions_time(self, text: str) -> bool:
        # "15:30", "15.30", "a las 3", "las 3", "3 de la tarde"
        if re.search(r"\b\d{1,2}[:.]\d{2}\b", text):
            return True
        if re.search(r"\b(?:a\s+)?las\s+\d{1,2}\b", text):
            return True
        if re.search(r"\b\d{1,2}\s+de\s+la\s+(tarde|noche|ma[ñn]ana)\b", text):
            return True
        if re.search(r"\bmedi[oa]d[ií]a\b", text):
            return True
        if re.search(r"\bmedianoche\b", text):
            return True
        return False

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
