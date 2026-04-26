from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from app.cognitive.temporal.models import TemporalFacts, TemporalInterpretation


# Mensajes semánticos según el motivo del fallo.
# No son el texto final al usuario: son hints para el LLM conversacional.
_INVALID_QUESTIONS = {
    "invalid_current_month_date": "Ese día no existe en este mes. ¿Qué fecha es exactamente?",
    "invalid_explicit_month_date": "Ese día no existe en ese mes. ¿Me confirmas la fecha?",
    "invalid_next_month_date": "Ese día no existe el mes que viene. ¿Me confirmas la fecha?",
    "invalid_day_number": "Ese día no existe. ¿Qué día exacto es?",
    "invalid_merged_date": "La fecha no me cuadra. ¿Me la confirmas?",
}


class RulesEngine:
    """
    Aplica reglas de negocio deterministas sobre los TemporalFacts extraídos.

    NO extrae nada (eso es trabajo de fast_parser / llm_parser).
    SOLO decide:
    - si la fecha resultante es futura, pasada, ambigua o inválida
    - qué preguntar al usuario si hay ambigüedad
    - qué candidato proponer si hay que confirmar
    - cómo fusionar campos de un draft pendiente con nuevos hechos
    """

    def __init__(self, timezone_name: str = "Europe/Madrid"):
        self.tz = ZoneInfo(timezone_name)

    # =========================
    # Entry point: primera interpretación
    # =========================

    def interpret(
        self,
        facts: TemporalFacts,
        now: Optional[datetime] = None,
    ) -> TemporalInterpretation:
        now = now or datetime.now(self.tz)

        # Día fuera de rango señalado por el parser
        if "invalid_day_value" in (facts.notes or []):
            return self._invalid(
                facts.title,
                None,
                facts.month,
                facts.hour,
                facts.minute,
                "invalid_day_number",
            )

        # Resolver día/mes/año desde referencias relativas
        resolved_day, resolved_month, resolved_year = self._resolve_reference_fields(facts, now)

        # Si no tenemos hora, no podemos construir nada
        if facts.hour is None or facts.minute is None:
            return TemporalInterpretation(
                status="no_match",
                title=facts.title,
                candidate_iso=None,
                clarification_question=None,
                reason="missing_time" if resolved_day is not None else "missing_day_or_time",
                extracted_day=resolved_day,
                extracted_month=resolved_month,
                extracted_hour=facts.hour,
                extracted_minute=facts.minute,
            )

        # Si tenemos hora pero no día → asumir hoy o proponer mañana
        if resolved_day is None and resolved_month is None:
            return self._handle_time_only(facts, now)

        # Si tenemos mes pero no día → no podemos resolver
        if resolved_day is None:
            return TemporalInterpretation(
                status="no_match",
                title=facts.title,
                candidate_iso=None,
                clarification_question=None,
                reason="missing_day",
                extracted_day=resolved_day,
                extracted_month=resolved_month,
                extracted_hour=facts.hour,
                extracted_minute=facts.minute,
            )

        # Mes implícito si no hay
        month_was_explicit = facts.month is not None or resolved_month != now.month
        if resolved_month is None:
            resolved_month = now.month

        if resolved_year is None:
            resolved_year = now.year

        candidate = self._safe_build(
            resolved_year, resolved_month, resolved_day, facts.hour, facts.minute
        )

        if candidate is None:
            reason = "invalid_explicit_month_date" if month_was_explicit else "invalid_current_month_date"
            return self._invalid(
                facts.title,
                resolved_day,
                resolved_month,
                facts.hour,
                facts.minute,
                reason,
            )

        if candidate >= now:
            return TemporalInterpretation(
                status="resolved",
                title=facts.title,
                candidate_iso=candidate.isoformat(),
                clarification_question=None,
                reason="explicit_month" if month_was_explicit else "current_month_future",
                extracted_day=resolved_day,
                extracted_month=resolved_month,
                extracted_hour=facts.hour,
                extracted_minute=facts.minute,
            )

        return self._handle_past_date(
            facts, resolved_day, resolved_month, month_was_explicit, now
        )

    # =========================
    # Merge: combinar draft previo con nuevos facts
    # =========================

    def merge_with_draft(
        self,
        draft: dict,
        fresh_facts: Optional[TemporalFacts],
        reply_text: str,
        now: Optional[datetime] = None,
    ) -> TemporalInterpretation:
        now = now or datetime.now(self.tz)
        raw = (reply_text or "").strip().lower()

        draft_day = draft.get("day")
        draft_month = draft.get("month")
        draft_hour = draft.get("hour")
        draft_minute = draft.get("minute")
        draft_title = draft.get("title") or "Cita"
        draft_candidate_iso = draft.get("candidate_iso")

        # 1) Confirmación afirmativa sin datos nuevos → usar candidate_iso del draft
        has_new_data = fresh_facts is not None and self._has_any_field(fresh_facts)
        positive = any(x in raw for x in ["sí", "si", "correcto", "exacto", "eso", "yes", "vale"])
        next_month_explicit = any(
            x in raw
            for x in [
                "mes que viene",
                "el mes que viene",
                "próximo mes",
                "proximo mes",
                "mes siguiente",
            ]
        )

        if positive and not next_month_explicit and not has_new_data and draft_candidate_iso:
            return TemporalInterpretation(
                status="resolved",
                title=draft_title,
                candidate_iso=draft_candidate_iso,
                clarification_question=None,
                reason="clarification_confirmed_candidate",
                extracted_day=draft_day,
                extracted_month=draft_month,
                extracted_hour=draft_hour,
                extracted_minute=draft_minute,
            )

        # 2) Merge: combinar draft con lo nuevo
        merged_day = draft_day
        merged_month = draft_month
        merged_hour = draft_hour
        merged_minute = draft_minute

        if fresh_facts is not None:
            fresh_day, fresh_month, fresh_year = self._resolve_reference_fields(fresh_facts, now)

            if fresh_day is not None:
                merged_day = fresh_day
            if fresh_month is not None:
                merged_month = fresh_month
            if fresh_facts.hour is not None:
                merged_hour = fresh_facts.hour
            if fresh_facts.minute is not None:
                merged_minute = fresh_facts.minute

        # 3) Suficiente para resolver
        if merged_day is not None and merged_hour is not None and merged_minute is not None:
            target_month = int(merged_month) if merged_month is not None else now.month
            target_year = now.year

            candidate = self._safe_build(
                target_year, target_month, int(merged_day), int(merged_hour), int(merged_minute)
            )

            if candidate is None:
                return self._invalid(
                    draft_title,
                    int(merged_day),
                    target_month,
                    int(merged_hour),
                    int(merged_minute),
                    "invalid_merged_date",
                )

            if candidate < now:
                if merged_month is not None:
                    candidate = self._safe_build(
                        target_year + 1, target_month, int(merged_day), int(merged_hour), int(merged_minute)
                    )
                else:
                    next_year, next_month = self._next_month(target_year, target_month)
                    candidate = self._safe_build(
                        next_year, next_month, int(merged_day), int(merged_hour), int(merged_minute)
                    )

                if candidate is None:
                    return self._invalid(
                        draft_title,
                        int(merged_day),
                        target_month,
                        int(merged_hour),
                        int(merged_minute),
                        "invalid_merged_date",
                    )

            return TemporalInterpretation(
                status="resolved",
                title=draft_title,
                candidate_iso=candidate.isoformat(),
                clarification_question=None,
                reason="merged_from_clarification",
                extracted_day=int(merged_day),
                extracted_month=int(merged_month) if merged_month is not None else target_month,
                extracted_hour=int(merged_hour),
                extracted_minute=int(merged_minute),
            )

        # 4) Falta info: preguntar específicamente
        missing = []
        if merged_day is None:
            missing.append("el día")
        if merged_hour is None:
            missing.append("la hora")

        question = "¿Me dices " + " y ".join(missing) + "?" if missing else "¿Me dices la fecha completa?"

        return TemporalInterpretation(
            status="invalid",
            title=draft_title,
            candidate_iso=None,
            clarification_question=question,
            reason="clarification_incomplete",
            extracted_day=merged_day,
            extracted_month=merged_month,
            extracted_hour=merged_hour,
            extracted_minute=merged_minute,
        )

    # =========================
    # Sub-casos
    # =========================

    def _handle_time_only(
        self,
        facts: TemporalFacts,
        now: datetime,
    ) -> TemporalInterpretation:
        today_candidate = self._safe_build(
            now.year, now.month, now.day, facts.hour, facts.minute
        )

        if today_candidate is not None and today_candidate > now:
            return TemporalInterpretation(
                status="resolved",
                title=facts.title,
                candidate_iso=today_candidate.isoformat(),
                clarification_question=None,
                reason="today_implicit",
                extracted_day=now.day,
                extracted_month=now.month,
                extracted_hour=facts.hour,
                extracted_minute=facts.minute,
            )

        tomorrow = now + timedelta(days=1)
        tomorrow_candidate = self._safe_build(
            tomorrow.year, tomorrow.month, tomorrow.day, facts.hour, facts.minute
        )
        question = (
            f"Esa hora ya ha pasado hoy. "
            f"¿Te refieres a mañana a las {facts.hour:02d}:{facts.minute:02d}?"
        )
        return TemporalInterpretation(
            status="ambiguous_past_date",
            title=facts.title,
            candidate_iso=tomorrow_candidate.isoformat() if tomorrow_candidate else None,
            clarification_question=question,
            reason="today_time_past",
            extracted_day=tomorrow.day,
            extracted_month=tomorrow.month,
            extracted_hour=facts.hour,
            extracted_minute=facts.minute,
        )

    def _handle_past_date(
        self,
        facts: TemporalFacts,
        day: int,
        month: int,
        month_was_explicit: bool,
        now: datetime,
    ) -> TemporalInterpretation:
        if month_was_explicit:
            next_year_candidate = self._safe_build(
                now.year + 1, month, day, facts.hour, facts.minute
            )
            month_name = self._month_name(month)
            question = (
                f"El {day} de {month_name} ya ha pasado. "
                f"¿Te refieres al {day} de {month_name} del año que viene?"
            )
            return TemporalInterpretation(
                status="ambiguous_past_date",
                title=facts.title,
                candidate_iso=next_year_candidate.isoformat() if next_year_candidate else None,
                clarification_question=question,
                reason="explicit_month_in_past",
                extracted_day=day,
                extracted_month=month,
                extracted_hour=facts.hour,
                extracted_minute=facts.minute,
            )

        next_year, next_month = self._next_month(now.year, now.month)
        next_candidate = self._safe_build(
            next_year, next_month, day, facts.hour, facts.minute
        )
        question = (
            f"El {day} de este mes ya ha pasado. "
            f"¿Te refieres al {day} del mes que viene a las {facts.hour:02d}:{facts.minute:02d}?"
        )
        return TemporalInterpretation(
            status="ambiguous_past_date",
            title=facts.title,
            candidate_iso=next_candidate.isoformat() if next_candidate else None,
            clarification_question=question,
            reason="implicit_month_in_past",
            extracted_day=day,
            extracted_month=next_month,
            extracted_hour=facts.hour,
            extracted_minute=facts.minute,
        )

    # =========================
    # Resolución de referencias relativas
    # =========================

    def _resolve_reference_fields(
        self,
        facts: TemporalFacts,
        now: datetime,
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        day = facts.day
        month = facts.month
        year = facts.year

        if facts.relative_day_offset is not None and day is None:
            target = now + timedelta(days=facts.relative_day_offset)
            day = target.day
            month = month if month is not None else target.month
            year = year if year is not None else target.year

        if facts.weekday is not None and day is None:
            days_ahead = (facts.weekday - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            if facts.weekday_is_next and days_ahead < 7:
                days_ahead += 7
            target = now + timedelta(days=days_ahead)
            day = target.day
            month = month if month is not None else target.month
            year = year if year is not None else target.year

        return day, month, year

    # =========================
    # Helpers
    # =========================

    def _safe_build(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
    ) -> Optional[datetime]:
        try:
            return datetime(year, month, day, hour, minute, tzinfo=self.tz)
        except ValueError:
            return None

    def _next_month(self, year: int, month: int) -> tuple[int, int]:
        if month == 12:
            return year + 1, 1
        return year, month + 1

    def _month_name(self, month: int) -> str:
        names = {
            1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
            5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
            9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
        }
        return names.get(month, str(month))

    def _has_any_field(self, facts: TemporalFacts) -> bool:
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

    def _invalid(
        self,
        title: Optional[str],
        day: Optional[int],
        month: Optional[int],
        hour: Optional[int],
        minute: Optional[int],
        reason: str,
    ) -> TemporalInterpretation:
        question = _INVALID_QUESTIONS.get(
            reason,
            "La fecha no me cuadra. ¿Me dices la fecha completa?",
        )
        return TemporalInterpretation(
            status="invalid",
            title=title,
            candidate_iso=None,
            clarification_question=question,
            reason=reason,
            extracted_day=day,
            extracted_month=month,
            extracted_hour=hour,
            extracted_minute=minute,
        )
