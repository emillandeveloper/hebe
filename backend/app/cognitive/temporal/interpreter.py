from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from app.cognitive.temporal.fast_parser import FastParser
from app.cognitive.temporal.llm_parser import LLMParser
from app.cognitive.temporal.models import TemporalFacts, TemporalInterpretation
from app.cognitive.temporal.rules_engine import RulesEngine


# Umbral mínimo de confianza del FastParser para no subir a LLM.
FAST_PARSER_CONFIDENCE_THRESHOLD = 0.85


class TemporalInterpreter:
    """
    Orquestador de las capas de interpretación temporal.

    Flujo:
    1. FastParser intenta extraer con dateparser (barato, rápido).
    2. Si FastParser no lo resuelve o tiene baja confianza, LLMParser lo hace.
    3. RulesEngine aplica decisiones deterministas sobre los facts extraídos.

    API compatible con el temporal_interpreter anterior:
    - interpret_appointment(text, now)
    - resolve_clarification(reply_text, draft, now)
    """

    def __init__(
        self,
        timezone_name: str = "Europe/Madrid",
        intent_client: Any | None = None,
    ):
        self.tz = ZoneInfo(timezone_name)
        self.fast_parser = FastParser(timezone_name=timezone_name)
        self.llm_parser = LLMParser(intent_client=intent_client) if intent_client else None
        self.rules = RulesEngine(timezone_name=timezone_name)

    # =========================
    # API pública
    # =========================

    def interpret_appointment(
        self,
        text: str,
        now: Optional[datetime] = None,
    ) -> TemporalInterpretation:
        now = now or datetime.now(self.tz)

        facts = self._extract_facts(text, now)

        if facts is None:
            return TemporalInterpretation(
                status="no_match",
                title="Cita",
                candidate_iso=None,
                clarification_question=None,
                reason="no_temporal_info",
                extracted_day=None,
                extracted_month=None,
                extracted_hour=None,
                extracted_minute=None,
            )

        return self.rules.interpret(facts, now=now)

    def resolve_clarification(
        self,
        reply_text: str,
        draft: dict,
        now: Optional[datetime] = None,
    ) -> TemporalInterpretation:
        now = now or datetime.now(self.tz)

        fresh_facts = self._extract_facts(reply_text, now)

        return self.rules.merge_with_draft(
            draft=draft,
            fresh_facts=fresh_facts,
            reply_text=reply_text,
            now=now,
        )

    # =========================
    # Orquestación de capas
    # =========================

    def _extract_facts(self, text: str, now: datetime) -> Optional[TemporalFacts]:
        """
        Intenta extraer facts con las capas en cascada.
        Devuelve None si ninguna capa extrae nada útil.
        """
        if not (text or "").strip():
            return None

        # Capa 1: FastParser
        fast_facts = self.fast_parser.parse(text, now=now)

        if fast_facts is not None and fast_facts.confidence >= FAST_PARSER_CONFIDENCE_THRESHOLD:
            return fast_facts

        # Capa 2: LLMParser (si está disponible)
        if self.llm_parser is not None:
            llm_facts = self.llm_parser.parse(text, now=now)
            if llm_facts is not None:
                # Si también teníamos fast_facts, preferimos el LLM porque entiende
                # casos más complejos, pero conservamos datos que el LLM no tenga
                if fast_facts is not None:
                    return self._merge_facts(fast_facts, llm_facts)
                return llm_facts

        # Si no hay LLM, devolver lo que hubiera del FastParser aunque sea baja confianza
        return fast_facts

    def _merge_facts(self, base: TemporalFacts, override: TemporalFacts) -> TemporalFacts:
        """
        Combina dos facts: override pisa base, pero campos None de override
        preservan los de base.
        """
        return TemporalFacts(
            day=override.day if override.day is not None else base.day,
            month=override.month if override.month is not None else base.month,
            year=override.year if override.year is not None else base.year,
            hour=override.hour if override.hour is not None else base.hour,
            minute=override.minute if override.minute is not None else base.minute,
            relative_day_offset=(
                override.relative_day_offset
                if override.relative_day_offset is not None
                else base.relative_day_offset
            ),
            weekday=override.weekday if override.weekday is not None else base.weekday,
            weekday_is_next=override.weekday_is_next or base.weekday_is_next,
            title=override.title if override.title != "Cita" else base.title,
            source="merged",
            confidence=max(override.confidence, base.confidence),
        )
