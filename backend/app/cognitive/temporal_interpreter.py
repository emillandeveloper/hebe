from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo


MADRID_TZ = ZoneInfo("Europe/Madrid")


@dataclass(slots=True)
class TemporalInterpretation:
    status: str  # resolved | ambiguous_past_date | invalid | no_match
    title: Optional[str]
    candidate_iso: Optional[str]
    clarification_question: Optional[str]
    reason: Optional[str]
    extracted_day: Optional[int]
    extracted_month: Optional[int]
    extracted_hour: Optional[int]
    extracted_minute: Optional[int]


class TemporalInterpreter:
    MONTHS = {
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

    # Mensajes semánticos según el motivo del fallo.
    # No son el texto final al usuario: son hints que el response_synthesizer
    # pasa al modelo para que los reformule con la personalidad de Hebe.
    _INVALID_QUESTIONS = {
        "invalid_current_month_date": "Ese día no existe en este mes. ¿Qué fecha es exactamente?",
        "invalid_explicit_month_date": "Ese día no existe en ese mes. ¿Me confirmas la fecha?",
        "invalid_next_month_date": "Ese día no existe el mes que viene. ¿Me confirmas la fecha?",
        "invalid_explicit_month_next_year_date": "Esa fecha no existe ni este año ni el siguiente. ¿Me la repites?",
        "invalid_day_number": "Ese día no existe. ¿Qué día exacto es?",
    }

    def __init__(self, timezone_name: str = "Europe/Madrid"):
        self.tz = ZoneInfo(timezone_name)

    def interpret_appointment(
        self,
        text: str,
        now: Optional[datetime] = None,
    ) -> TemporalInterpretation:
        now = now or datetime.now(self.tz)
        raw = (text or "").strip().lower()

        title = self._infer_title(raw)
        day = self._extract_day(raw)
        month = self._extract_month(raw)
        hour, minute = self._extract_time(raw)

        # BUG 1: día extraído pero inválido (ej: 32, 0)
        if day == -1:
            return TemporalInterpretation(
                status="invalid",
                title=title,
                candidate_iso=None,
                clarification_question=self._INVALID_QUESTIONS["invalid_day_number"],
                reason="invalid_day_number",
                extracted_day=None,
                extracted_month=month,
                extracted_hour=hour,
                extracted_minute=minute,
            )

        # Sin hora no podemos inferir nada útil
        if hour is None or minute is None:
            return TemporalInterpretation(
                status="no_match",
                title=title,
                candidate_iso=None,
                clarification_question=None,
                reason="missing_time" if day is not None else "missing_day_or_time",
                extracted_day=day,
                extracted_month=month,
                extracted_hour=hour,
                extracted_minute=minute,
            )

        # NUEVO: si hay hora pero no día (y sin mes), asumir hoy o proponer mañana
        if day is None and month is None:
            today_candidate = self._safe_build(now.year, now.month, now.day, hour, minute)

            if today_candidate is not None and today_candidate > now:
                return TemporalInterpretation(
                    status="resolved",
                    title=title,
                    candidate_iso=today_candidate.isoformat(),
                    clarification_question=None,
                    reason="today_implicit",
                    extracted_day=now.day,
                    extracted_month=now.month,
                    extracted_hour=hour,
                    extracted_minute=minute,
                )

            # Hora ya pasada hoy -> proponer mañana
            tomorrow = now + timedelta(days=1)
            tomorrow_candidate = self._safe_build(
                tomorrow.year, tomorrow.month, tomorrow.day, hour, minute
            )
            question = (
                f"Esa hora ya ha pasado hoy. "
                f"¿Te refieres a mañana a las {hour:02d}:{minute:02d}?"
            )
            return TemporalInterpretation(
                status="ambiguous_past_date",
                title=title,
                candidate_iso=tomorrow_candidate.isoformat() if tomorrow_candidate else None,
                clarification_question=question,
                reason="today_time_past",
                extracted_day=tomorrow.day,
                extracted_month=tomorrow.month,
                extracted_hour=hour,
                extracted_minute=minute,
            )

        # Si hay hora y mes pero no día, seguimos sin suficiente info
        if day is None:
            return TemporalInterpretation(
                status="no_match",
                title=title,
                candidate_iso=None,
                clarification_question=None,
                reason="missing_day",
                extracted_day=day,
                extracted_month=month,
                extracted_hour=hour,
                extracted_minute=minute,
            )

        next_month_explicit = any(
            phrase in raw
            for phrase in [
                "mes que viene",
                "el mes que viene",
                "próximo mes",
                "proximo mes",
                "mes siguiente",
            ]
        )

        this_month_explicit = "este mes" in raw

        # Caso 1: mes explícito por nombre ("abril", "mayo"...)
        if month is not None:
            candidate = self._safe_build(now.year, month, day, hour, minute)
            if candidate is None:
                return self._invalid(title, day, month, hour, minute, "invalid_explicit_month_date")

            # BUG 3: si la fecha con mes explícito ya pasó, preguntar año siguiente
            if candidate < now:
                next_year_candidate = self._safe_build(now.year + 1, month, day, hour, minute)
                month_name = next(
                    (k for k, v in self.MONTHS.items() if v == month), str(month)
                )
                question = (
                    f"El {day} de {month_name} ya ha pasado. "
                    f"¿Te refieres al {day} de {month_name} del año que viene?"
                )
                return TemporalInterpretation(
                    status="ambiguous_past_date",
                    title=title,
                    candidate_iso=next_year_candidate.isoformat() if next_year_candidate else None,
                    clarification_question=question,
                    reason="explicit_month_in_past",
                    extracted_day=day,
                    extracted_month=month,
                    extracted_hour=hour,
                    extracted_minute=minute,
                )

            return TemporalInterpretation(
                status="resolved",
                title=title,
                candidate_iso=candidate.isoformat(),
                clarification_question=None,
                reason="explicit_month",
                extracted_day=day,
                extracted_month=month,
                extracted_hour=hour,
                extracted_minute=minute,
            )

        # Caso 2: "mes que viene"
        if next_month_explicit:
            year, next_month = self._next_month(now.year, now.month)
            candidate = self._safe_build(year, next_month, day, hour, minute)
            if candidate is None:
                return self._invalid(title, day, next_month, hour, minute, "invalid_next_month_date")

            return TemporalInterpretation(
                status="resolved",
                title=title,
                candidate_iso=candidate.isoformat(),
                clarification_question=None,
                reason="next_month_explicit",
                extracted_day=day,
                extracted_month=next_month,
                extracted_hour=hour,
                extracted_minute=minute,
            )

        # Caso 3: sin mes explícito -> mes actual
        candidate = self._safe_build(now.year, now.month, day, hour, minute)
        if candidate is None:
            return self._invalid(title, day, now.month, hour, minute, "invalid_current_month_date")

        if candidate < now:
            # El día de este mes ya pasó: proponer mes siguiente como candidato futuro
            next_year, next_month = self._next_month(now.year, now.month)
            next_candidate = self._safe_build(next_year, next_month, day, hour, minute)

            question = (
                f"El {day} de este mes ya ha pasado. "
                f"¿Te refieres al {day} del mes que viene a las {hour:02d}:{minute:02d}?"
            )

            return TemporalInterpretation(
                status="ambiguous_past_date",
                title=title,
                candidate_iso=next_candidate.isoformat() if next_candidate else None,
                clarification_question=question,
                reason="explicit_this_month_in_past" if this_month_explicit else "implicit_month_in_past",
                extracted_day=day,
                extracted_month=next_month,
                extracted_hour=hour,
                extracted_minute=minute,
            )

        return TemporalInterpretation(
            status="resolved",
            title=title,
            candidate_iso=candidate.isoformat(),
            clarification_question=None,
            reason="current_month_future",
            extracted_day=day,
            extracted_month=now.month,
            extracted_hour=hour,
            extracted_minute=minute,
        )

    def resolve_clarification(
        self,
        reply_text: str,
        draft: dict,
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

        # 1) Reinterpretar como fecha completa nueva
        fresh = self.interpret_appointment(reply_text, now=now)

        # Si fresh se resolvió por "hoy implícito" pero el draft ya tenía día,
        # preferimos el merge (el draft manda para día/mes, fresh aporta lo que falta)
        if fresh.status == "resolved" and fresh.reason == "today_implicit" and draft_day is not None:
            pass  # continuar al merge
        elif fresh.status in {"resolved", "ambiguous_past_date", "invalid"}:
            return fresh

        # 2) Merge: combinar campos del draft con los que aportó fresh
        merged_day = fresh.extracted_day if fresh.extracted_day is not None else draft_day
        merged_month = fresh.extracted_month if fresh.extracted_month is not None else draft_month
        merged_hour = fresh.extracted_hour if fresh.extracted_hour is not None else draft_hour
        merged_minute = fresh.extracted_minute if fresh.extracted_minute is not None else draft_minute

        # Si ya hay suficiente info, resolver
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
                    "invalid_current_month_date",
                )

            if candidate < now:
                # En pasado tras merge: si mes explícito -> año siguiente, si no -> mes siguiente
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
                        "invalid_current_month_date",
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

        # 3) Confirmación afirmativa: usar el candidate_iso ya propuesto
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

        if positive and not next_month_explicit and draft_candidate_iso:
            return TemporalInterpretation(
                status="resolved",
                title=draft_title,
                candidate_iso=draft_candidate_iso,
                clarification_question=None,
                reason="clarification_confirmed_candidate",
                extracted_day=int(draft_day) if draft_day is not None else None,
                extracted_month=int(draft_month) if draft_month is not None else None,
                extracted_hour=int(draft_hour) if draft_hour is not None else None,
                extracted_minute=int(draft_minute) if draft_minute is not None else None,
            )

        if positive or next_month_explicit:
            if draft_day is None or draft_hour is None or draft_minute is None:
                return TemporalInterpretation(
                    status="invalid",
                    title=draft_title,
                    candidate_iso=None,
                    clarification_question="No me ha quedado clara la fecha. ¿Me la dices completa?",
                    reason="draft_incomplete",
                    extracted_day=draft_day,
                    extracted_month=draft_month,
                    extracted_hour=draft_hour,
                    extracted_minute=draft_minute,
                )

            year, month = self._next_month(now.year, now.month)
            candidate = self._safe_build(year, month, int(draft_day), int(draft_hour), int(draft_minute))
            if candidate is None:
                return self._invalid(
                    draft_title,
                    int(draft_day),
                    month,
                    int(draft_hour),
                    int(draft_minute),
                    "invalid_next_month_date",
                )

            return TemporalInterpretation(
                status="resolved",
                title=draft_title,
                candidate_iso=candidate.isoformat(),
                clarification_question=None,
                reason="clarification_confirmed_next_month",
                extracted_day=int(draft_day),
                extracted_month=month,
                extracted_hour=int(draft_hour),
                extracted_minute=int(draft_minute),
            )

        # 4) Seguimos sin info suficiente: pedir lo que falta de forma específica
        missing = []
        if merged_day is None:
            missing.append("el día")
        if merged_hour is None:
            missing.append("la hora")

        if missing:
            question = "¿Me dices " + " y ".join(missing) + "?"
        else:
            question = "No me ha quedado clara la fecha. ¿Me la repites?"

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

    def _extract_day(self, text: str) -> Optional[int]:
        """
        Devuelve:
          - int 1-31 si el día es válido
          - -1 si se encontró un número fuera de rango
          - None si no se encontró ningún número de día
        """
        patterns = [
            r"\bel\s+(\d{1,2})\b",
            r"\bd[ií]a\s+(\d{1,2})\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                day = int(m.group(1))
                if 1 <= day <= 31:
                    return day
                else:
                    return -1
        return None

    def _extract_month(self, text: str) -> Optional[int]:
        for name, value in self.MONTHS.items():
            if re.search(rf"\b{name}\b", text):
                return value
        return None

    def _extract_time(self, text: str) -> tuple[Optional[int], Optional[int]]:
        # BUG 2: detectar tarde/noche para ajustar AM/PM
        is_afternoon = any(
            x in text
            for x in ["de la tarde", "de la noche", " tarde", " noche", "pm"]
        )

        def _adjust_hour(h: int) -> int:
            if is_afternoon and 1 <= h <= 11:
                return h + 12
            return h

        # 15:30 o 15.30
        m = re.search(r"\b(\d{1,2})[:.](\d{2})\b", text)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return _adjust_hour(hour), minute

        # "a las 3 y media" / "las 3 y media"  ("a" opcional)
        m = re.search(r"\b(?:a\s+)?las\s+(\d{1,2})\s+y\s+media\b", text)
        if m:
            hour = int(m.group(1))
            if 0 <= hour <= 23:
                return _adjust_hour(hour), 30

        # "a las 3 y 20" / "las 3 y 20"  ("a" opcional)
        m = re.search(r"\b(?:a\s+)?las\s+(\d{1,2})\s+y\s+(\d{1,2})\b", text)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return _adjust_hour(hour), minute

        # "a las 3" / "las 3"  ("a" opcional)
        m = re.search(r"\b(?:a\s+)?las\s+(\d{1,2})\b", text)
        if m:
            hour = int(m.group(1))
            if 0 <= hour <= 23:
                return _adjust_hour(hour), 0

        # "5 de la tarde" / "5 de la noche" / "5 de la mañana" (sin "las")
        m = re.search(r"\b(\d{1,2})\s+de\s+la\s+(?:tarde|noche|ma[ñn]ana)\b", text)
        if m:
            hour = int(m.group(1))
            if 0 <= hour <= 23:
                return _adjust_hour(hour), 0

        return None, None

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

    def _invalid(
        self,
        title: Optional[str],
        day: Optional[int],
        month: Optional[int],
        hour: Optional[int],
        minute: Optional[int],
        reason: str,
    ) -> TemporalInterpretation:
        question = self._INVALID_QUESTIONS.get(
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