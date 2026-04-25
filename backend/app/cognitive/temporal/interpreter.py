from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from app.cognitive.temporal.fast_parser import FastParser
from app.cognitive.temporal.llm_parser import LLMParser
from app.cognitive.temporal.models import TemporalFacts, TemporalInterpretation, TemporalSignals
from app.cognitive.temporal.normalizer import TemporalFactsNormalizer
from app.cognitive.temporal.rules_engine import RulesEngine


class TemporalInterpreter:
    """
    Orquestador temporal.

    FastParser solo detecta señales simples.
    LLMParser extrae hechos.
    TemporalFactsNormalizer valida y normaliza.
    RulesEngine interpreta los hechos.
    """

    def __init__(
        self,
        timezone_name: str = "Europe/Madrid",
        intent_client: Any | None = None,
    ):
        self.tz = ZoneInfo(timezone_name)
        self.fast_parser = FastParser(timezone_name=timezone_name)
        self.llm_parser = LLMParser(intent_client=intent_client) if intent_client else None
        self.normalizer = TemporalFactsNormalizer()
        self.rules = RulesEngine(timezone_name=timezone_name)

    def interpret_appointment(
        self,
        text: str,
        now: Optional[datetime] = None,
    ) -> TemporalInterpretation:
        now = now or datetime.now(self.tz)
        signals = self.detect_signals(text, now=now)
        facts = self.extract_with_llm(text, now=now)
        facts = self.fuse_temporal_results(signals, facts)

        if facts is None:
            return self.empty_interpretation(reason="no_temporal_facts")

        return self.interpret_facts(facts, now=now)

    def resolve_clarification(
        self,
        reply_text: str,
        draft: dict,
        now: Optional[datetime] = None,
    ) -> TemporalInterpretation:
        now = now or datetime.now(self.tz)
        signals = self.detect_signals(reply_text, now=now)
        fresh_facts = self.extract_with_llm(reply_text, now=now)
        fresh_facts = self.fuse_temporal_results(signals, fresh_facts)

        return self.rules.merge_with_draft(
            draft=draft,
            fresh_facts=fresh_facts,
            reply_text=reply_text,
            now=now,
        )

    def detect_signals(self, text: str, now: Optional[datetime] = None) -> TemporalSignals:
        signals = self.fast_parser.parse(text, now=now)
        print(f"[HEBE][TEMPORAL] fast_signals={signals!r}", flush=True)
        return signals

    def extract_with_llm(self, text: str, now: datetime) -> Optional[TemporalFacts]:
        if not (text or "").strip() or self.llm_parser is None:
            return None

        facts = self.llm_parser.parse(text, now=now)
        print(f"[HEBE][TEMPORAL] llm_facts={facts!r}", flush=True)
        return facts

    def fuse_temporal_results(
        self,
        signals: TemporalSignals,
        facts: Optional[TemporalFacts],
    ) -> Optional[TemporalFacts]:
        if facts is None:
            return None

        facts.notes.append(
            "fast_parser:temporal_signal"
            if signals.has_temporal_signal
            else "fast_parser:no_temporal_signal"
        )
        facts.notes.extend(signals.notes)

        return self.normalizer.normalize(facts, source=facts.source)

    def interpret_facts(self, facts: TemporalFacts, now: datetime) -> TemporalInterpretation:
        return self.rules.interpret(facts, now=now)

    def empty_interpretation(self, *, reason: str, title: str = "Cita") -> TemporalInterpretation:
        return TemporalInterpretation(
            status="no_match",
            title=title,
            candidate_iso=None,
            clarification_question=None,
            reason=reason,
            extracted_day=None,
            extracted_month=None,
            extracted_hour=None,
            extracted_minute=None,
        )
