from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from app.cognitive.temporal.fast_parser import FastParser
from app.cognitive.temporal.llm_parser import LLMParser
from app.cognitive.temporal.models import TemporalFacts, TemporalInterpretation
from app.cognitive.temporal.rules_engine import RulesEngine


class TemporalInterpreter:
    """
    Orquestador de las capas de interpretación temporal.

    Flujo:
    1. FastParser intenta extraer literales.
    2. Si los literales bastan para que rules pueda decidir, se usan directos.
    3. Si no, LLMParser completa con extracción cognitiva.
    4. RulesEngine aplica decisiones deterministas.

    API compatible:
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

        print(
            "[HEBE][TEMPORAL][DEBUG] TemporalInterpreter loaded",
            flush=True,
        )
        print(
            "[HEBE][TEMPORAL][DEBUG] RulesEngine class path="
            f"{inspect.getfile(RulesEngine)}",
            flush=True,
        )
        print(
            "[HEBE][TEMPORAL][DEBUG] RulesEngine instance="
            f"{self.rules!r}",
            flush=True,
        )

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

        print(
            "[HEBE][TEMPORAL][DEBUG] resolve_clarification "
            f"reply_text={reply_text!r} "
            f"draft={draft!r} "
            f"fresh_facts={fresh_facts!r}",
            flush=True,
        )

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
        if not (text or "").strip():
            return None

        fast_facts = self.fast_parser.parse(text, now=now)
        print(f"[HEBE][TEMPORAL] fast_facts={fast_facts!r}", flush=True)

        # Si fast detecta día inválido, no subimos al LLM: que rules responda.
        if fast_facts is not None and "invalid_day_value" in (fast_facts.notes or []):
            print("[HEBE][TEMPORAL] invalid_day from fast, skip LLM", flush=True)
            return fast_facts

        # Si fast tiene info suficiente para rules, no subimos al LLM.
        if fast_facts is not None and self._is_fast_parser_sufficient(fast_facts):
            print("[HEBE][TEMPORAL] using fast parser directly", flush=True)
            return fast_facts

        # Subir al LLM si está disponible.
        if self.llm_parser is not None:
            llm_facts = self.llm_parser.parse(text, now=now)
            print(f"[HEBE][TEMPORAL] llm_facts={llm_facts!r}", flush=True)

            if llm_facts is not None:
                if fast_facts is not None:
                    merged = self._merge_conservative(fast_facts, llm_facts)
                    print(f"[HEBE][TEMPORAL] merged_facts={merged!r}", flush=True)
                    return merged

                return llm_facts

        return fast_facts

    def _is_fast_parser_sufficient(self, facts: TemporalFacts) -> bool:
        """
        Determina si los facts del parser rápido bastan para que RulesEngine
        haga su trabajo sin necesidad del LLM.
        """
        has_time = facts.hour is not None and facts.minute is not None

        has_absolute_date = facts.day is not None and (
            facts.month is not None or facts.year is not None
        )

        has_relative_date = facts.relative_day_offset is not None
        has_weekday = facts.weekday is not None

        return has_time and (has_absolute_date or has_relative_date or has_weekday)

    def _merge_conservative(
        self,
        base: TemporalFacts,
        override: TemporalFacts,
    ) -> TemporalFacts:
        """
        Merge conservador:
        - base/FastParser manda en campos explícitos.
        - override/LLM solo rellena huecos.
        - el LLM nunca pisa valores literales del usuario.
        """
        return TemporalFacts(
            day=base.day if base.day is not None else override.day,
            month=base.month if base.month is not None else override.month,
            year=base.year if base.year is not None else override.year,
            hour=base.hour if base.hour is not None else override.hour,
            minute=base.minute if base.minute is not None else override.minute,
            relative_day_offset=(
                base.relative_day_offset
                if base.relative_day_offset is not None
                else override.relative_day_offset
            ),
            weekday=base.weekday if base.weekday is not None else override.weekday,
            weekday_is_next=base.weekday_is_next or override.weekday_is_next,
            title=base.title if base.title and base.title != "Cita" else override.title,
            source="merged",
            confidence=max(base.confidence, override.confidence),
            notes=[*(base.notes or []), *(override.notes or [])],
        )