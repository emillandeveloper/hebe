# backend/app/cognitive/deliberation_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, List

from app.cognitive.context_builder import BuiltContext
from app.cognitive.models import PlanStep, Plan, DeliberationResult

# =========================
# Service
# =========================

class DeliberationService:
    """
    Cerebro de Hebe v1.

    Responsabilidades:
    - interpretar el contexto
    - decidir qué hacer
    - generar un plan de ejecución

    Importante:
    - NO ejecuta nada
    - NO genera texto final
    """

    def __init__(self, intent_model, reasoning_model):
        self.intent_model = intent_model
        self.reasoning_model = reasoning_model

    # =========================
    # Entry point
    # =========================

    def deliberate(self, context: BuiltContext) -> DeliberationResult:
        # Caso 1: evento interno (ej: reminder_due)
        if context.internal_event:
            return self._handle_internal_event(context)

        # Caso 2: input de usuario
        if context.input_text:
            return self._handle_user_input(context)

        # fallback
        return DeliberationResult(plan=Plan(steps=[]))

    # =========================
    # Internal events
    # =========================

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
                                "payload": event.payload
                            }
                        )
                    ],
                    reasoning="Reminder due → notify user"
                )
            )

        # default fallback
        return DeliberationResult(plan=Plan(steps=[]))

    # =========================
    # User input
    # =========================

    def _handle_user_input(self, context: BuiltContext) -> DeliberationResult:
        text = (context.input_text or "").strip().lower()

        if self._looks_like_appointment(text):
            return self._plan_appointment(context)

        app_name = self._extract_open_app_target(text)
        if app_name:
            return self._plan_open_app(app_name)

        return self._plan_with_llm(context)

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
                            "params": {
                                "app_name": app_name,
                            },
                        },
                    ),
                    PlanStep(
                        type="reply",
                        data={
                            "mode": "confirm_action",
                        },
                    ),
                ],
                reasoning=f"User requested open_app for {app_name}",
            )
        )

    # =========================
    # Heuristics (v1)
    # =========================

    def _looks_like_appointment(self, text: str) -> bool:
        keywords = ["psicóloga", "médico", "dentista", "cita", "hora"]
        return any(k in text for k in keywords)

    def _plan_appointment(self, context: BuiltContext) -> DeliberationResult:
        text = context.input_text

        # ⚠️ v1: simplificado (luego irá a parsing real con LLM)
        fake_due = "2026-04-12T15:30:00+02:00"

        return DeliberationResult(
            plan=Plan(
                steps=[
                    PlanStep(
                        type="memory",
                        data={
                            "kind": "appointment",
                            "title": "Cita",
                            "due_at": fake_due,
                            "source_text": text,
                        }
                    ),
                    PlanStep(
                        type="reminder",
                        data={
                            "title": "Cita",
                            "due_at": fake_due,
                            "message": "Tienes una cita programada"
                        }
                    ),
                    PlanStep(
                        type="reply",
                        data={
                            "mode": "confirm_appointment"
                        }
                    )
                ],
                reasoning="Detected appointment from text"
            )
        )

    # =========================
    # LLM fallback
    # =========================

    def _plan_with_llm(self, context: BuiltContext) -> DeliberationResult:
        """
        Aquí irá el verdadero poder en v2.
        De momento fallback básico.
        """

        # Placeholder (puedes conectar tu modelo luego)
        return DeliberationResult(
            plan=Plan(
                steps=[
                    PlanStep(
                        type="reply",
                        data={
                            "mode": "chat"
                        }
                    )
                ],
                reasoning="Fallback chat"
            )
        )