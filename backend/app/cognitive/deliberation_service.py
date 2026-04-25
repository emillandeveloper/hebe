# backend/app/cognitive/deliberation_service.py
from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.cognitive.context_builder import BuiltContext
from app.cognitive.models import PlanStep, Plan, DeliberationResult
from app.cognitive.temporal import TemporalFacts, TemporalInterpreter, TemporalSignals


class DeliberationService:
    def __init__(self, intent_model: Any, reasoning_model: Any):
        """
        intent_model: cliente LLM para extracción estructurada (hebe-intent).
                      Debe exponer chat_structured(system_prompt, user_prompt, schema, temperature).
        reasoning_model: reservado para el futuro (planificación compleja).
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

        return DeliberationResult(plan=Plan(steps=[]))

    def _handle_user_input(self, context: BuiltContext) -> DeliberationResult:
        text = (context.input_text or "").strip().lower()

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
        interp = self._interpret_temporal_from_deliberation(context.input_text or "", now)

        if interp.status == "resolved":
            return self._build_resolved_plan(
                title=interp.title or "Cita",
                candidate_iso=interp.candidate_iso,
                source_text=context.input_text,
                reason=interp.reason,
            )

        if interp.status in {"ambiguous_past_date", "invalid"}:
            draft = self._build_draft(interp, context.input_text)
            question = interp.clarification_question or "¿Qué fecha exacta quieres decir?"
            return self._build_clarify_plan(question, draft, interp.reason)

        # no_match: guardar campos parciales y preguntar solo lo que falta
        draft = self._build_draft(interp, context.input_text)
        question = self._build_missing_fields_question(interp)
        return self._build_clarify_plan(question, draft, interp.reason or "no_match")

    def _resolve_pending_appointment(self, context: BuiltContext, pending: dict) -> DeliberationResult:
        now = datetime.now(ZoneInfo("Europe/Madrid"))
        reply_text = context.input_text or ""
        signals = self.temporal.detect_signals(reply_text, now=now)
        llm_facts = self.temporal.extract_with_llm(reply_text, now=now)
        fresh_facts = self._fuse_temporal_results(signals, llm_facts)

        interp = self.temporal.rules.merge_with_draft(
            draft=pending.get("draft", {}),
            fresh_facts=fresh_facts,
            reply_text=reply_text,
            now=now,
        )

        if interp.status == "resolved":
            return self._build_resolved_plan(
                title=interp.title or "Cita",
                candidate_iso=interp.candidate_iso,
                source_text=pending.get("draft", {}).get("source_text"),
                reason=interp.reason,
            )

        if interp.status in {"ambiguous_past_date", "invalid"}:
            draft = self._build_draft(interp, context.input_text)
            question = interp.clarification_question or "¿Qué fecha exacta quieres decir?"
            return self._build_clarify_plan(question, draft, interp.reason)

        # Fallback
        return self._build_clarify_plan(
            question="No me ha quedado claro. ¿Me dices la fecha completa?",
            draft=pending.get("draft", {}),
            reason=interp.reason or "unresolved",
        )

    def _interpret_temporal_from_deliberation(self, text: str, now: datetime):
        signals = self.temporal.detect_signals(text, now=now)

        # FastParser solo detecta señales. Aunque no detecte nada, en flujos
        # de cita el extractor LLM sigue siendo la fuente de hechos temporales.
        llm_facts = self.temporal.extract_with_llm(text, now=now)
        facts = self._fuse_temporal_results(signals, llm_facts)

        if facts is None:
            return self.temporal.empty_interpretation(reason="no_temporal_facts")

        return self.temporal.interpret_facts(facts, now=now)

    def _fuse_temporal_results(
        self,
        signals: TemporalSignals,
        llm_facts: TemporalFacts | None,
    ) -> TemporalFacts | None:
        """
        Deliberation fusiona detección barata y extracción LLM.
        No interpreta fechas; solo conserva trazas de evidencia.
        """
        return self.temporal.fuse_temporal_results(signals, llm_facts)

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

    def _build_clarify_plan(
        self,
        question: str,
        draft: dict,
        reason: str | None,
    ) -> DeliberationResult:
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
            return "¿Me dices " + " y ".join(missing) + "?"
        return "No me ha quedado clara la fecha. ¿Me la dices completa?"

    # =========================
    # Helpers: detección
    # =========================

    def _looks_like_appointment(self, text: str) -> bool:
        keywords = ["psicóloga", "psicologa", "médico", "medico", "dentista", "cita", "hora"]
        return any(k in text for k in keywords)

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
