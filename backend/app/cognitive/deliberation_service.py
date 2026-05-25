# backend/app/cognitive/deliberation_service.py
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.cognitive.context_builder import BuiltContext
from app.cognitive.models import PlanStep, Plan, DeliberationResult
from app.cognitive.temporal import TemporalInterpreter


class DeliberationService:
    def __init__(self, intent_model: Any, reasoning_model: Any):
        """
        intent_model: cliente LLM para extracción estructurada (hebe-intent).
                      Debe exponer chat_structured(system_prompt, user_prompt, schema, temperature).
        reasoning_model: reservado para el futuro.
        """
        self.intent_model = intent_model
        self.reasoning_model = reasoning_model
        self.temporal = TemporalInterpreter(
            timezone_name="Europe/Madrid",
            intent_client=intent_model,
        )

    def deliberate(self, context: BuiltContext) -> DeliberationResult:
        if context.internal_event:
            return self._handle_internal_event(context)

        if context.input_text:
            return self._handle_user_input(context)

        return DeliberationResult(plan=Plan(steps=[]))

    def _handle_internal_event(self, context: BuiltContext) -> DeliberationResult:
        event = context.internal_event

        if event.event_type == "reminder_due":
            return DeliberationResult(
                plan=Plan(
                    steps=[
                        PlanStep(
                            type="reply",
                            data={
                                "mode": "reminder",
                                "payload": event.payload,
                            },
                        )
                    ],
                    reasoning="Reminder due -> notify user",
                )
            )

        # Eventos de stream (Twitch) — modo pasivo: solo reply, sin acciones de PC
        if event.event_type.startswith("twitch_"):
            return self._plan_twitch_event(event)

        return DeliberationResult(plan=Plan(steps=[]))

    def _plan_twitch_event(self, event) -> DeliberationResult:
        return DeliberationResult(
            plan=Plan(
                steps=[
                    PlanStep(
                        type="reply",
                        data={
                            "mode": event.event_type,
                            "payload": event.payload,
                        },
                    )
                ],
                reasoning=f"Twitch event {event.event_type} -> stream reply",
            )
        )

    def _handle_user_input(self, context: BuiltContext) -> DeliberationResult:
        text = (context.input_text or "").strip().lower()

        reminder = self._parse_relative_reminder(context.input_text or "")
        if reminder is not None:
            return self._plan_relative_reminder(
                title=reminder["title"],
                message=reminder["message"],
                due_at=reminder["due_at"],
                relative_label=reminder["relative_label"],
                source_text=context.input_text,
            )

        pending = context.state_snapshot.get("pending_clarification")
        if pending and pending.get("kind") == "appointment_datetime":
            return self._resolve_pending_appointment(context, pending)

        if self._looks_like_appointment(text):
            return self._plan_appointment(context)

        app_name = self._extract_open_app_target(text)
        if app_name:
            return self._plan_open_app(app_name)

        return self._plan_with_llm(context)

    def _plan_appointment(self, context: BuiltContext) -> DeliberationResult:
        now = datetime.now(ZoneInfo("Europe/Madrid"))
        interp = self.temporal.interpret_appointment(context.input_text or "", now=now)

        print(
            "[HEBE][APPOINTMENT] first interpretation "
            f"status={interp.status!r} "
            f"reason={interp.reason!r} "
            f"day={interp.extracted_day!r} "
            f"month={interp.extracted_month!r} "
            f"hour={interp.extracted_hour!r} "
            f"minute={interp.extracted_minute!r} "
            f"candidate={interp.candidate_iso!r}",
            flush=True,
        )

        if interp.status == "resolved":
            return self._build_resolved_plan(
                title=interp.title or "Cita",
                candidate_iso=interp.candidate_iso,
                source_text=context.input_text,
                reason=interp.reason,
            )

        if interp.status in {"ambiguous_past_date", "invalid"}:
            draft = self._build_draft(interp, context.input_text)
            question = interp.clarification_question or "Pide al usuario que confirme la fecha exacta."
            return self._build_clarify_plan(question, draft, interp.reason)

        draft = self._build_draft(interp, context.input_text)
        question = interp.clarification_question or self._build_missing_fields_question(interp)
        return self._build_clarify_plan(question, draft, interp.reason or "no_match")

    def _resolve_pending_appointment(self, context: BuiltContext, pending: dict) -> DeliberationResult:
        now = datetime.now(ZoneInfo("Europe/Madrid"))
        reply_text = context.input_text or ""
        pending_draft = pending.get("draft", {})

        print(
            "[HEBE][PENDING] loaded draft="
            f"{pending_draft!r} reply_text={reply_text!r}",
            flush=True,
        )

        interp = self.temporal.resolve_clarification(
            reply_text=reply_text,
            draft=pending_draft,
            now=now,
        )

        print(
            "[HEBE][PENDING] interp="
            f"status={interp.status!r} "
            f"reason={interp.reason!r} "
            f"day={interp.extracted_day!r} "
            f"month={interp.extracted_month!r} "
            f"hour={interp.extracted_hour!r} "
            f"minute={interp.extracted_minute!r} "
            f"candidate={interp.candidate_iso!r}",
            flush=True,
        )

        if interp.status == "resolved":
            return self._build_resolved_plan(
                title=interp.title or "Cita",
                candidate_iso=interp.candidate_iso,
                source_text=pending_draft.get("source_text"),
                reason=interp.reason,
            )

        if interp.status in {"ambiguous_past_date", "invalid"}:
            draft = self._build_draft(
                interp,
                pending_draft.get("source_text"),
            )
            question = interp.clarification_question or "Pide al usuario que confirme la fecha exacta."

            return self._build_clarify_plan(
                question=question,
                draft=draft,
                reason=interp.reason,
            )

        if interp.status == "no_match":
            draft = self._build_draft(
                interp,
                pending_draft.get("source_text"),
            )
            question = interp.clarification_question or self._build_missing_fields_question(interp)

            return self._build_clarify_plan(
                question=question,
                draft=draft,
                reason=interp.reason or "clarification_incomplete",
            )

        return self._build_clarify_plan(
            question="No se ha podido completar la fecha. Pide al usuario la fecha completa.",
            draft=pending_draft,
            reason=interp.reason or "unresolved",
        )

    # =========================
    # Helpers: construcción de planes
    # =========================

    def _build_resolved_plan(
        self,
        *,
        title: str,
        candidate_iso: str | None,
        source_text: str | None,
        reason: str | None,
    ) -> DeliberationResult:
        return DeliberationResult(
            plan=Plan(
                steps=[
                    PlanStep(
                        type="memory",
                        data={
                            "kind": "appointment",
                            "title": title,
                            "due_at": candidate_iso,
                            "source_text": source_text,
                            "payload": {
                                "title": title,
                                "due_at": candidate_iso,
                            },
                        },
                    ),
                    PlanStep(
                        type="reminder",
                        data={
                            "title": title,
                            "due_at": candidate_iso,
                            "kind": "appointment",
                            "message": f"Te recuerdo: {title}",
                            "payload": {
                                "title": title,
                                "due_at": candidate_iso,
                            },
                        },
                    ),
                    PlanStep(
                        type="reply",
                        data={"mode": "confirm_appointment"},
                    ),
                ],
                reasoning=f"Resolved appointment datetime: {reason}",
            )
        )

    def _plan_relative_reminder(
        self,
        *,
        title: str,
        message: str,
        due_at: str,
        relative_label: str,
        source_text: str | None,
    ) -> DeliberationResult:
        return DeliberationResult(
            plan=Plan(
                steps=[
                    PlanStep(
                        type="reminder",
                        data={
                            "title": title,
                            "due_at": due_at,
                            "kind": "generic",
                            "message": message,
                            "timezone_name": "Europe/Madrid",
                            "payload": {
                                "title": title,
                                "message": message,
                                "due_at": due_at,
                                "relative_label": relative_label,
                                "source_text": source_text,
                            },
                        },
                    ),
                    PlanStep(
                        type="reply",
                        data={
                            "mode": "confirm_reminder",
                            "title": title,
                            "message": message,
                            "due_at": due_at,
                            "relative_label": relative_label,
                        },
                    ),
                ],
                reasoning="Resolved relative reminder deterministically",
            )
        )

    def _build_clarify_plan(
        self,
        question: str,
        draft: dict,
        reason: str | None,
    ) -> DeliberationResult:
        print(
            "[HEBE][CLARIFY] building clarify plan "
            f"reason={reason!r} "
            f"question={question!r} "
            f"draft={draft!r}",
            flush=True,
        )

        return DeliberationResult(
            plan=Plan(
                steps=[
                    PlanStep(
                        type="reply",
                        data={
                            "mode": "clarify_appointment_datetime",
                            "question": question,
                            "draft": draft,
                        },
                    )
                ],
                reasoning=f"Appointment clarification: {reason}",
            )
        )

    def _build_draft(self, interp, source_text: str | None) -> dict:
        return {
            "title": interp.title or "Cita",
            "day": interp.extracted_day,
            "month": interp.extracted_month,
            "hour": interp.extracted_hour,
            "minute": interp.extracted_minute,
            "candidate_iso": interp.candidate_iso,
            "source_text": source_text,
            "reason": interp.reason,
        }

    def _build_missing_fields_question(self, interp) -> str:
        missing = []

        if interp.extracted_day is None:
            missing.append("el día")

        if interp.extracted_hour is None:
            missing.append("la hora")

        if missing:
            return "Pide al usuario solo " + " y ".join(missing) + "."

        return "Pide al usuario la fecha completa porque no ha quedado clara."

    # =========================
    # Helpers: detección
    # =========================

    def _looks_like_appointment(self, text: str) -> bool:
        keywords = [
            "psicóloga",
            "psicologa",
            "médico",
            "medico",
            "dentista",
            "cita",
            "hora",
        ]
        return any(k in text for k in keywords)

    def _parse_relative_reminder(self, text: str) -> dict[str, str] | None:
        raw = (text or "").strip()
        normalized = self._normalize_text(raw)
        if not normalized:
            return None

        match = self._match_relative_reminder(normalized)
        if match is None:
            return None

        minutes = match["minutes"]
        message = self._extract_reminder_message(normalized, raw)
        if not message:
            return None

        due = datetime.now(ZoneInfo("Europe/Madrid")) + timedelta(minutes=minutes)
        return {
            "title": message,
            "message": f"Te recuerdo: {message}",
            "due_at": due.isoformat(),
            "relative_label": match["relative_label"],
        }

    def _match_relative_reminder(self, normalized: str) -> dict[str, Any] | None:
        if not any(marker in normalized for marker in ("recuerdame", "avisame", "recordatorio")):
            return None

        patterns = [
            r"\b(?:recuerdame|avisame)\s+en\s+(?P<amount>\d+|un|una|media)\s+(?P<unit>minuto|minutos|hora|horas)\b",
            r"\bdentro\s+de\s+(?P<amount>\d+|un|una|media)\s+(?P<unit>minuto|minutos|hora|horas)\s+(?:recuerdame|avisame)\b",
            r"\bponme\s+un\s+recordatorio\s+en\s+(?P<amount>\d+|un|una|media)\s+(?P<unit>minuto|minutos|hora|horas)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            amount_text = match.group("amount")
            unit = match.group("unit")
            minutes = self._relative_minutes(amount_text, unit)
            if minutes is None:
                continue
            return {
                "minutes": minutes,
                "relative_label": self._relative_label(amount_text, unit, minutes),
            }
        return None

    def _relative_minutes(self, amount_text: str, unit: str) -> int | None:
        if amount_text == "media":
            if unit.startswith("hora"):
                return 30
            return None
        if amount_text in {"un", "una"}:
            amount = 1
        else:
            try:
                amount = int(amount_text)
            except ValueError:
                return None
        if amount <= 0:
            return None
        return amount * 60 if unit.startswith("hora") else amount

    def _relative_label(self, amount_text: str, unit: str, minutes: int) -> str:
        if amount_text == "media" and unit.startswith("hora"):
            return "media hora"
        if minutes == 1:
            return "1 minuto"
        if minutes < 60:
            return f"{minutes} minutos"
        hours = minutes // 60
        if hours == 1:
            return "1 hora"
        return f"{hours} horas"

    def _extract_reminder_message(self, normalized: str, raw: str) -> str:
        raw_patterns = [
            r"\b(?:recu[eé]rdame|av[ií]same)\s+en\s+(?:\d+|un|una|media)\s+(?:minuto|minutos|hora|horas)\s+(?:que|de|para)\s+(.+)$",
            r"\bdentro\s+de\s+(?:\d+|un|una|media)\s+(?:minuto|minutos|hora|horas)\s+(?:recu[eé]rdame|av[ií]same)\s+(?:que|de|para)?\s*(.+)$",
            r"\bponme\s+un\s+recordatorio\s+en\s+(?:\d+|un|una|media)\s+(?:minuto|minutos|hora|horas)\s+(?:que|de|para)\s+(.+)$",
        ]
        for pattern in raw_patterns:
            match = re.search(pattern, raw, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip(" .,:;")
                if candidate:
                    return candidate

        normalized_patterns = [
            r"\b(?:recuerdame|avisame)\s+en\s+(?:\d+|un|una|media)\s+(?:minuto|minutos|hora|horas)\s+(?:que|de|para)\s+(.+)$",
            r"\bdentro\s+de\s+(?:\d+|un|una|media)\s+(?:minuto|minutos|hora|horas)\s+(?:recuerdame|avisame)\s+(?:que|de|para)?\s*(.+)$",
            r"\bponme\s+un\s+recordatorio\s+en\s+(?:\d+|un|una|media)\s+(?:minuto|minutos|hora|horas)\s+(?:que|de|para)\s+(.+)$",
        ]
        for pattern in normalized_patterns:
            match = re.search(pattern, normalized)
            if match:
                candidate = match.group(1).strip(" .,:;")
                if candidate:
                    return candidate

        # Fallback on the original text keeps accents/casing for messages
        # that used an uncommon but still complete phrasing.
        text = (raw or "").strip()
        for marker in (" que ", " para ", " de "):
            idx = text.lower().rfind(marker)
            if idx >= 0:
                return text[idx + len(marker):].strip(" .,:;")
        return ""

    def _normalize_text(self, text: str) -> str:
        raw = (text or "").strip().lower()
        without_accents = "".join(
            ch for ch in unicodedata.normalize("NFKD", raw)
            if not unicodedata.combining(ch)
        )
        cleaned = re.sub(r"[^a-z0-9ñ\s]", " ", without_accents)
        return " ".join(cleaned.split())

    def _extract_open_app_target(self, text: str) -> str | None:
        prefixes = ["abre ", "open "]

        for prefix in prefixes:
            if text.startswith(prefix):
                candidate = text[len(prefix):].strip()

                if candidate:
                    return candidate

        return None

    def _plan_open_app(self, app_name: str) -> DeliberationResult:
        return DeliberationResult(
            plan=Plan(
                steps=[
                    PlanStep(
                        type="action",
                        data={
                            "name": "open_app",
                            "params": {"app_name": app_name},
                        },
                    ),
                    PlanStep(
                        type="reply",
                        data={"mode": "confirm_action"},
                    ),
                ],
                reasoning=f"User requested open_app for {app_name}",
            )
        )

    def _plan_with_llm(self, context: BuiltContext) -> DeliberationResult:
        return DeliberationResult(
            plan=Plan(
                steps=[
                    PlanStep(
                        type="reply",
                        data={"mode": "chat"},
                    )
                ],
                reasoning="Fallback chat",
            )
        )
    def _plan_twitch_event(self, event) -> DeliberationResult:
        """
        Plan común para eventos de stream en modo pasivo.
        El synthesizer decide el texto exacto según event_type.
        """
        return DeliberationResult(
            plan=Plan(
                steps=[
                    PlanStep(
                        type="reply",
                        data={
                            "mode": event.event_type,
                            "payload": event.payload,
                        },
                    )
                ],
                reasoning=f"Stream event: {event.event_type}",
            )
        )
