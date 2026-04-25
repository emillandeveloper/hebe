from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from app.cognitive.temporal.models import TemporalFacts
from app.cognitive.temporal.normalizer import TemporalFactsNormalizer


TEMPORAL_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "day": {
            "type": ["integer", "null"],
            "description": "Dia del mes (1-31) si el usuario lo menciona. null si no lo dice.",
        },
        "month": {
            "type": ["integer", "null"],
            "description": "Mes (1-12) si el usuario lo menciona por nombre o numero. null si no lo dice.",
        },
        "year": {
            "type": ["integer", "null"],
            "description": "Ano completo (ej: 2026) si el usuario lo menciona. null si no lo dice.",
        },
        "hour": {
            "type": ["integer", "null"],
            "description": "Hora en formato 24h (0-23) si el usuario menciona hora. null si no la dice.",
        },
        "minute": {
            "type": ["integer", "null"],
            "description": "Minutos (0-59). 0 si dice una hora exacta. null si no menciona hora.",
        },
        "relative_day_offset": {
            "type": ["integer", "null"],
            "description": "Referencia relativa: 0=hoy, 1=manana, 2=pasado manana, -1=ayer. null si no aplica.",
        },
        "weekday": {
            "type": ["integer", "null"],
            "description": "Dia de la semana: 0=lunes, ..., 6=domingo. null si no aplica.",
        },
        "weekday_is_next": {
            "type": "boolean",
            "description": "True si dice proximo jueves, jueves que viene o equivalente.",
        },
        "title": {
            "type": "string",
            "description": "Titulo inferido: Psicologa, Medico, Dentista o Cita por defecto.",
        },
        "confidence": {
            "type": "number",
            "description": "Confianza de la extraccion, 0.0 a 1.0.",
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
    Extractor LLM de hechos temporales atomicos.

    No decide futuro/pasado/ambiguedad. Tampoco valida rangos finales:
    eso lo hace TemporalFactsNormalizer.
    """

    def __init__(self, intent_client: Any):
        self.intent_client = intent_client
        self.normalizer = TemporalFactsNormalizer()

    def parse(self, text: str, now: datetime) -> Optional[TemporalFacts]:
        raw = (text or "").strip()
        if not raw or self.intent_client is None:
            return None

        system_prompt = self._build_system_prompt(now)
        user_prompt = f"Texto del usuario: {raw}"

        try:
            print(f"[HEBE][TEMPORAL][LLM] system_prompt length={len(system_prompt)}", flush=True)
            print(f"[HEBE][TEMPORAL][LLM] user_prompt={user_prompt!r}", flush=True)
            data = self.intent_client.chat_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=TEMPORAL_EXTRACTION_SCHEMA,
                temperature=0.0,
            )
        except Exception as exc:
            print(f"[HEBE][TEMPORAL][LLM] extraction failed: {exc!r}", flush=True)
            return None

        return self.normalizer.normalize(data, source="llm_parser")

    def _build_system_prompt(self, now: datetime) -> str:
        now_iso = now.strftime("%Y-%m-%d %H:%M")
        weekday_name = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"][now.weekday()]

        return (
            "Eres un extractor de informacion temporal en espanol.\n"
            "Tu tarea es leer un texto y devolver SOLO hechos temporales atomicos en JSON.\n\n"
            f"Fecha y hora actuales: {now_iso} ({weekday_name}).\n\n"
            "Reglas estrictas:\n"
            "- No inventes campos. Si el usuario no menciona un campo, usa null.\n"
            "- No decidas si la fecha es futura o pasada.\n"
            "- No calcules el ano si el usuario no lo dice.\n"
            "- Para 'manana' usa relative_day_offset=1 y deja day/month/year en null.\n"
            "- Para 'hoy' usa relative_day_offset=0 y deja day/month/year en null.\n"
            "- Para 'pasado manana' usa relative_day_offset=2.\n"
            "- Para '5 de la tarde' usa hour=17, minute=0.\n"
            "- Para '9 y media de la manana' usa hour=9, minute=30.\n"
            "- Para '10:30' usa hour=10, minute=30.\n"
            "- Para 'mediodia' usa hour=12, minute=0.\n"
            "- Para 'medianoche' usa hour=0, minute=0.\n"
            "- Para 'el jueves' usa weekday=3, weekday_is_next=false.\n"
            "- Para 'el jueves que viene' o 'proximo jueves' usa weekday=3, weekday_is_next=true.\n"
            "- Si no estas seguro de un campo, usa null.\n"
            "- El titulo se infiere de palabras como psicologa, medico o dentista. Por defecto: Cita.\n"
        )
