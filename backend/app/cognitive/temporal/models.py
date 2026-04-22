from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class TemporalFacts:
    """
    Hechos atómicos extraídos del texto del usuario.

    Esta es la salida de las capas de extracción (FastParser, LLMParser).
    NO contiene decisiones ni inferencias de contexto: solo lo que el usuario
    literalmente ha dicho.

    - Si un campo es None, significa que el usuario no lo mencionó.
    - Si un campo tiene valor, significa que el usuario lo dijo explícita
      o implícitamente (ej: "mañana" → relative_day_offset=1).
    """

    day: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None

    # Referencias relativas al día de hoy
    # 0 = hoy, 1 = mañana, 2 = pasado mañana, -1 = ayer, etc.
    relative_day_offset: Optional[int] = None

    # Día de la semana como referencia ("el jueves" → weekday=3)
    # 0 = lunes, ..., 6 = domingo
    weekday: Optional[int] = None

    # Si la referencia al día de la semana es "el próximo jueves" vs "este jueves"
    weekday_is_next: bool = False

    # Título inferido a partir de palabras clave ("psicóloga", "dentista"...)
    title: str = "Cita"

    # Metadata de la extracción
    source: str = "unknown"  # "fast_parser" | "llm_parser" | "merged"
    confidence: float = 1.0  # 0.0 - 1.0
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TemporalInterpretation:
    """
    Interpretación final después de pasar los facts por las reglas de negocio.

    Esto es lo que consume el deliberation_service para decidir qué plan ejecutar.
    """

    status: str  # resolved | ambiguous_past_date | invalid | no_match
    title: Optional[str]
    candidate_iso: Optional[str]
    clarification_question: Optional[str]
    reason: Optional[str]
    extracted_day: Optional[int]
    extracted_month: Optional[int]
    extracted_hour: Optional[int]
    extracted_minute: Optional[int]
