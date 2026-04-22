from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from app.cognitive.temporal.models import TemporalFacts


# Schema JSON que el modelo debe devolver.
# Usamos Optional (null) para TODO: es crítico que el modelo NO invente campos.
TEMPORAL_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "day": {
            "type": ["integer", "null"],
            "description": "Día del mes (1-31) si el usuario lo menciona explícitamente. null si no lo dice.",
        },
        "month": {
            "type": ["integer", "null"],
            "description": "Mes (1-12) si el usuario lo menciona por nombre o número. null si no lo dice.",
        },
        "year": {
            "type": ["integer", "null"],
            "description": "Año completo (ej: 2026) si el usuario lo menciona. null si no lo dice.",
        },
        "hour": {
            "type": ["integer", "null"],
            "description": "Hora en formato 24h (0-23) si el usuario la menciona. Ajustar AM/PM según 'de la tarde', 'de la noche', etc. null si no lo dice.",
        },
        "minute": {
            "type": ["integer", "null"],
            "description": "Minutos (0-59) si el usuario los menciona. 0 si dice hora exacta. null si no menciona hora.",
        },
        "relative_day_offset": {
            "type": ["integer", "null"],
            "description": "Referencia relativa al día de hoy. 0=hoy, 1=mañana, 2=pasado mañana, -1=ayer. null si no aplica.",
        },
        "weekday": {
            "type": ["integer", "null"],
            "description": "Día de la semana (0=lunes, ..., 6=domingo) si el usuario lo menciona. null si no aplica.",
        },
        "weekday_is_next": {
            "type": "boolean",
            "description": "Si el usuario dice 'el próximo jueves' o 'el jueves que viene'. False si dice solo 'el jueves'.",
        },
        "title": {
            "type": "string",
            "description": "Título inferido: 'Psicóloga', 'Médico', 'Dentista', o 'Cita' por defecto.",
        },
        "confidence": {
            "type": "number",
            "description": "Confianza de la extracción, 0.0 a 1.0.",
        },
    },
    "required": [
        "day",
        "month",
        "year",
        "hour",
        "minute",
        "relative_day_offset",
        "weekday",
        "weekday_is_next",
        "title",
        "confidence",
    ],
}


class LLMParser:
    """
    Capa 2: Parser con LLM (hebe-intent) para casos que dateparser no resuelve.

    Filosofía:
    - el LLM solo extrae hechos atómicos
    - NO decide si es futuro, pasado, ambiguo, etc. (eso es del rules_engine)
    - si no está seguro, devuelve null en los campos y confidence bajo
    """

    def __init__(self, intent_client: Any):
        """
        intent_client debe tener el método:
          chat_structured(system_prompt, user_prompt, schema, temperature) -> dict
        """
        self.intent_client = intent_client

    def parse(self, text: str, now: datetime) -> Optional[TemporalFacts]:
        raw = (text or "").strip()
        if not raw:
            return None

        if self.intent_client is None:
            return None

        system_prompt = self._build_system_prompt(now)
        user_prompt = f"Texto del usuario: {raw}"

        try:
            data = self.intent_client.chat_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=TEMPORAL_EXTRACTION_SCHEMA,
                temperature=0.0,
            )
        except Exception as e:
            print(f"[HEBE][TEMPORAL][LLM] extraction failed: {e!r}", flush=True)
            return None

        return self._to_facts(data)

    # =========================
    # Helpers
    # =========================

    def _build_system_prompt(self, now: datetime) -> str:
        now_iso = now.strftime("%Y-%m-%d %H:%M")
        weekday_name = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"][now.weekday()]

        return (
            "Eres un extractor de información temporal en español.\n"
            "Tu tarea: leer un texto y extraer SOLO los campos de fecha/hora que el usuario menciona explícita o implícitamente.\n\n"
            f"Fecha y hora actuales: {now_iso} ({weekday_name}).\n\n"
            "Reglas estrictas:\n"
            "- NO inventes campos. Si el usuario no menciona el mes, devuelve null en month.\n"
            "- NO decidas si la fecha es futura o pasada. Solo extrae.\n"
            "- NO calcules el año si el usuario no lo dice (devuelve null).\n"
            "- Para 'mañana' pon relative_day_offset=1 y devuelve null en day/month.\n"
            "- Para 'hoy' pon relative_day_offset=0 y devuelve null en day/month.\n"
            "- Para 'pasado mañana' pon relative_day_offset=2.\n"
            "- Para '5 de la tarde' pon hour=17, minute=0.\n"
            "- Para '9 y media de la mañana' pon hour=9, minute=30.\n"
            "- Para '10:30' pon hour=10, minute=30.\n"
            "- Para 'mediodía' pon hour=12, minute=0.\n"
            "- Para 'medianoche' pon hour=0, minute=0.\n"
            "- Para 'el jueves' pon weekday=3, weekday_is_next=false.\n"
            "- Para 'el jueves que viene' o 'próximo jueves' pon weekday=3, weekday_is_next=true.\n"
            "- Si no estás seguro de un campo, devuelve null.\n"
            "- El título se infiere de palabras como 'psicóloga' (Psicóloga), 'médico' (Médico), 'dentista' (Dentista). Por defecto: 'Cita'.\n"
        )

    def _to_facts(self, data: dict[str, Any]) -> TemporalFacts:
        def _as_int(key: str) -> Optional[int]:
            value = data.get(key)
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def _as_bool(key: str) -> bool:
            return bool(data.get(key, False))

        def _as_float(key: str, default: float = 0.5) -> float:
            value = data.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        day = _as_int("day")
        month = _as_int("month")
        hour = _as_int("hour")
        minute = _as_int("minute")

        # Validaciones defensivas: si el LLM devuelve valores absurdos, los tratamos como None
        if day is not None and not (1 <= day <= 31):
            day = None
        if month is not None and not (1 <= month <= 12):
            month = None
        if hour is not None and not (0 <= hour <= 23):
            hour = None
        if minute is not None and not (0 <= minute <= 59):
            minute = None

        title = data.get("title") or "Cita"
        if not isinstance(title, str):
            title = "Cita"

        return TemporalFacts(
            day=day,
            month=month,
            year=_as_int("year"),
            hour=hour,
            minute=minute,
            relative_day_offset=_as_int("relative_day_offset"),
            weekday=_as_int("weekday"),
            weekday_is_next=_as_bool("weekday_is_next"),
            title=title,
            source="llm_parser",
            confidence=_as_float("confidence", 0.7),
        )
