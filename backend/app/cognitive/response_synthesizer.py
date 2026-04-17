# backend/app/cognitive/response_synthesizer.py
from __future__ import annotations

from typing import Optional, Any

from app.cognitive.context_builder import BuiltContext
from app.cognitive.models import DeliberationResult, ExecutionResult

class ResponseSynthesizer:
    """
    Convierte:
    - contexto
    - resultado de deliberation
    - resultado de ejecución

    en una respuesta natural.

    Aquí vive la personalidad de Hebe (v1 básica).
    """

    def __init__(self, conversation_model: Any | None = None):
        self.conversation_model = conversation_model

    # =========================
    # Entry point
    # =========================

    def synthesize(
        self,
        context: BuiltContext,
        deliberation: DeliberationResult,
        execution: ExecutionResult,
    ) -> str:
        # Caso 1: evento interno (reminder, etc)
        if context.internal_event:
            return self._handle_internal_event(context, execution)

        # Caso 2: respuesta basada en plan
        reply_step = execution.first_result_of_type("reply")

        if reply_step:
            mode = reply_step.data.get("mode")

            if mode == "confirm_appointment":
                return self._confirm_appointment(context, execution)

            if mode == "confirm_action":
                return self._confirm_action(context, execution)

            if mode == "chat":
                return self._chat_fallback(context)

        # fallback absoluto
        return self._default_response()

    # =========================
    # Internal events
    # =========================

    def _handle_internal_event(
        self,
        context: BuiltContext,
        execution: ExecutionResult,
    ) -> str:
        event = context.internal_event

        if event.event_type == "reminder_due":
            payload = event.payload

            title = payload.get("title") or "algo pendiente"
            message = payload.get("message")

            if message:
                return message

            return f"Oye, te recuerdo: {title}"

        return "Ha ocurrido algo, pero no tengo claro qué."

    # =========================
    # Appointment
    # =========================

    def _confirm_appointment(
        self,
        context: BuiltContext,
        execution: ExecutionResult,
    ) -> str:
        memory_result = execution.first_result_of_type("memory")
        reminder_result = execution.first_result_of_type("reminder")

        title = "la cita"
        due_at = None

        if memory_result:
            fact = memory_result.data.get("fact")
            if fact and fact.payload:
                title = fact.payload.get("title", title)
                due_at = fact.payload.get("due_at")

        if due_at:
            return f"Vale, te lo guardo: {title} el {self._format_datetime(due_at)}. Te avisaré cuando toque."
        else:
            return f"Vale, te lo guardo: {title}. Te avisaré cuando toque."

    # =========================
    # Action
    # =========================

    def _confirm_action(
        self,
        context: BuiltContext,
        execution: ExecutionResult,
    ) -> str:
        action_result = execution.first_result_of_type("action")

        if action_result and action_result.success:
            data = action_result.data or {}
            action_name = data.get("action_name")
            result_obj = data.get("action_result")

            if action_name == "open_app" and result_obj:
                opened = getattr(result_obj, "data", {}).get("app_name")
                if opened:
                    return f"Abriendo {opened}."

            return "Hecho."

        return "Lo he intentado, pero algo no ha ido bien."

    # =========================
    # Chat fallback
    # =========================

    def _chat_fallback(self, context: BuiltContext) -> str:
        if self.conversation_model is None:
            return "No estoy segura de qué hacer con eso todavía."

        prompt = self._build_prompt(context)

        try:
            return self.conversation_model.generate(prompt)
        except Exception as e:
            print(f"⚠️ Error en modelo conversacional: {e}")
            return "Hmm... no estoy muy fina ahora mismo."

    def _build_prompt(self, context: BuiltContext) -> str:
        parts = []

        if context.input_text:
            parts.append(f"Usuario: {context.input_text}")

        if context.relevant_facts:
            parts.append("Memoria relevante:")
            for fact in context.relevant_facts:
                parts.append(f"- {fact.subject}: {fact.payload}")

        parts.append("Responde de forma natural y breve.")

        return "\n".join(parts)

    # =========================
    # Helpers
    # =========================

    def _format_datetime(self, iso_str: str) -> str:
        """
        Formateo simple v1 (luego se puede mejorar mucho)
        """
        try:
            # ejemplo: 2026-04-12T15:30:00+02:00
            date_part = iso_str.split("T")[0]
            time_part = iso_str.split("T")[1][:5]

            return f"{date_part} a las {time_part}"
        except Exception:
            return iso_str

    def _default_response(self) -> str:
        return "Vale."