from __future__ import annotations

from typing import Any

from app.cognitive.context_builder import BuiltContext
from app.cognitive.models import DeliberationResult, ExecutionResult


class ResponseSynthesizer:
    """
    Convierte:
    - contexto
    - resultado de deliberation
    - resultado de ejecución

    en una respuesta natural generada por el modelo conversacional.

    Regla:
    - evitar hardcodear wording
    - usar el modelo para formular la respuesta final
    - mantener solo fallbacks mínimos por robustez
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
        if context.internal_event:
            return self._handle_internal_event(context, execution)

        reply_step = execution.first_result_of_type("reply")

        if reply_step:
            mode = reply_step.data.get("mode")

            if mode == "confirm_appointment":
                return self._generate_confirm_appointment(context, execution)

            if mode == "confirm_action":
                return self._generate_confirm_action(context, execution)

            if mode == "chat":
                return self._generate_chat_reply(context)

            if mode == "clarify_appointment_datetime":
                return self._generate_clarification_reply(context, reply_step.data)

        return self._fallback_text("No tengo suficiente contexto para responder con seguridad.")

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
            payload = event.payload or {}

            title = payload.get("title") or "Cita"
            due_at = payload.get("due_at")
            timezone = payload.get("timezone") or "Europe/Madrid"

            system, user = self._build_event_prompt(
                event_type="reminder_due",
                title=title,
                due_at=due_at,
                timezone=timezone,
                raw_payload=payload,
            )
            return self._call_model(system, user, fallback=self._fallback_reminder_text(payload))

        return self._fallback_text("Ha ocurrido algo, pero no tengo claro qué.")

    # =========================
    # Mode-specific generation
    # =========================

    def _generate_confirm_appointment(
        self,
        context: BuiltContext,
        execution: ExecutionResult,
    ) -> str:
        memory_result = execution.first_result_of_type("memory")

        title = "Cita"
        due_at = None

        if memory_result:
            fact = memory_result.data.get("fact")
            if fact and getattr(fact, "payload", None):
                payload = fact.payload or {}
                title = payload.get("title", title)
                due_at = payload.get("due_at")

        system, user = self._build_confirm_appointment_prompt(
            title=title,
            due_at=due_at,
        )

        fallback = (
            f"Vale, te lo guardo: {title} el {self._format_datetime(due_at)}. Te avisaré cuando toque."
            if due_at
            else f"Vale, te lo guardo: {title}. Te avisaré cuando toque."
        )

        return self._call_model(system, user, fallback=fallback)

    def _generate_confirm_action(
        self,
        context: BuiltContext,
        execution: ExecutionResult,
    ) -> str:
        action_result = execution.first_result_of_type("action")

        action_name = None
        action_success = False
        action_payload = {}

        if action_result:
            action_success = bool(action_result.success)
            data = action_result.data or {}
            action_name = data.get("action_name")
            result_obj = data.get("action_result")

            if result_obj is not None:
                action_payload = getattr(result_obj, "data", {}) or {}

        system, user = self._build_confirm_action_prompt(
            user_text=context.input_text,
            action_name=action_name,
            action_success=action_success,
            action_payload=action_payload,
        )

        fallback = self._fallback_action_text(
            action_name=action_name,
            action_success=action_success,
            action_payload=action_payload,
        )

        return self._call_model(system, user, fallback=fallback)

    def _generate_clarification_reply(
        self,
        context: BuiltContext,
        reply_data: dict,
    ) -> str:
        system, user = self._build_clarification_prompt(
            reply_data=reply_data,
        )

        fallback = reply_data.get("question") or "No me ha quedado clara la fecha."
        return self._call_model(system, user, fallback=fallback)

    def _generate_chat_reply(self, context: BuiltContext) -> str:
        system, user = self._build_chat_prompt(context)
        return self._call_model(
            system,
            user,
            fallback="No estoy segura de qué decirte ahora mismo.",
        )

    # =========================
    # Prompt builders — devuelven (system, user)
    # FIX BUG 4: separar instrucciones del input del usuario
    # =========================

    def _build_system_style_block(self) -> str:
        return (
            "Eres Hebe, la compañera IA personal de Leo.\n"
            "Responde en español de forma natural, breve, clara y grounded.\n"
            "No inventes hechos, fechas, horas, nombres, lugares ni acciones.\n"
            "No uses tono robótico.\n"
            "No uses tono excesivamente teatral ni ceremonial.\n"
            "No expliques tu proceso interno.\n"
            "No repitas instrucciones.\n"
            "No repitas ni cites lo que ha dicho el usuario.\n"
            "No menciones zonas horarias, timezone, UTC, ISO ni formatos técnicos.\n"
            "No incluyas etiquetas como 'Respuesta:' en la salida.\n"
            "No incluyas numeraciones, duplicados, bloques meta ni texto fuera de la respuesta final.\n"
            "Escribe solo la respuesta final.\n"
        )

    def _build_confirm_appointment_prompt(
        self,
        *,
        title: str,
        due_at: str | None,
    ) -> tuple[str, str]:
        formatted_due = self._format_datetime(due_at)

        system = (
            f"{self._build_system_style_block()}\n"
            "Situación: acabas de guardar correctamente una cita y su recordatorio.\n"
            "Objetivo: confirmar al usuario la cita de forma natural.\n\n"
            "Reglas:\n"
            f"- El título exacto de la cita es: {title}\n"
            f"- La fecha y hora exactas son: {formatted_due}\n"
            "- Usa exactamente el título, la fecha y la hora indicados.\n"
            "- No cambies el año, el mes, el día ni la hora.\n"
            "- No añadas detalles que no estén aquí.\n"
            "- Puedes mencionar que avisarás cuando toque.\n"
            "- Sé breve."
        )

        user = "Confirma que has guardado la cita."

        return system, user

    def _build_confirm_action_prompt(
        self,
        *,
        user_text: str | None,
        action_name: str | None,
        action_success: bool,
        action_payload: dict,
    ) -> tuple[str, str]:
        app_name = action_payload.get("app_name")

        system = (
            f"{self._build_system_style_block()}\n"
            "Situación: acabas de ejecutar una acción del sistema.\n"
            "Objetivo: responder al usuario de forma natural según el resultado.\n\n"
            "Reglas:\n"
            f"- Acción exacta ejecutada: {action_name or 'desconocida'}\n"
            f"- Resultado: {'éxito' if action_success else 'fallo'}\n"
            f"- Nombre de la app si existe: {app_name or 'ninguna'}\n"
            "- Si salió bien, confirma la acción de forma breve.\n"
            "- Si la acción fue open_app y tienes el nombre de la app, úsalo exactamente.\n"
            "- Si salió mal, dilo de forma natural sin inventar motivos técnicos.\n"
            "- No añadas promesas ni explicaciones largas.\n"
            "- Sé breve."
        )

        user = user_text or "¿Qué ha pasado con la acción?"

        return system, user

    def _build_clarification_prompt(
        self,
        *,
        reply_data: dict,
    ) -> tuple[str, str]:
        draft = reply_data.get("draft", {}) or {}
        question = reply_data.get("question", "") or "No me ha quedado clara la fecha."

        day = draft.get("day")
        month = draft.get("month")
        hour = draft.get("hour")
        minute = draft.get("minute")

        system = (
            f"{self._build_system_style_block()}\n"
            "Situación: el usuario quiere guardar una cita, pero falta información o la fecha es ambigua.\n"
            "Objetivo: pedir solo la aclaración necesaria de forma natural.\n\n"
            "Reglas:\n"
            f"- Dato conocido — día: {day}\n"
            f"- Dato conocido — mes: {month}\n"
            f"- Dato conocido — hora: {hour}\n"
            f"- Dato conocido — minuto: {minute}\n"
            f"- Aclaración necesaria: {question}\n"
            "- Pide solo el dato que falta.\n"
            "- Si ya conoces el día y la hora, no los vuelvas a pedir.\n"
            "- No inventes fechas ni cambies los datos ya conocidos.\n"
            "- Sé breve y conversacional."
        )

        user = "Pide la aclaración necesaria."

        return system, user

    def _build_event_prompt(
        self,
        *,
        event_type: str,
        title: str,
        due_at: str | None,
        timezone: str,
        raw_payload: dict,
    ) -> tuple[str, str]:
        formatted_due = self._format_datetime(due_at)

        system = (
            f"{self._build_system_style_block()}\n"
            "Situación: se ha disparado un recordatorio y debes avisar al usuario ahora mismo.\n"
            "Objetivo: recordarle la cita de forma natural y muy breve, como lo haría una persona cercana.\n\n"
            "Datos exactos (no cambiar):\n"
            f"- Título: {title}\n"
            f"- Fecha y hora: {formatted_due}\n\n"
            "Reglas estrictas:\n"
            "- Empieza directamente con el aviso, sin preámbulos.\n"
            "- No digas cosas como 'no recuerdo más información', 'no sé más detalles' ni similares.\n"
            "- No añadas información que no esté en los datos exactos.\n"
            "- No inventes lugar, médico, persona, contexto ni motivo.\n"
            "- No menciones zona horaria ni formatos técnicos.\n"
            "- Máximo una o dos frases.\n"
            "- Ejemplo del tipo de respuesta esperada: 'Oye, tienes cita a las 10:07.' o 'Recuerda, toca la cita a las 10:07.'"
        )

        user = "Avisa al usuario del recordatorio ahora."

        return system, user

    def _build_chat_prompt(self, context: BuiltContext) -> tuple[str, str]:
        system_parts = [
            self._build_system_style_block(),
            "Situación: el usuario te ha dicho algo que no requiere una acción específica.",
            "Objetivo: responder de forma natural, breve y útil.",
        ]

        if context.relevant_facts:
            system_parts.append("\nMemoria relevante:")
            for fact in context.relevant_facts:
                system_parts.append(f"- {fact.subject}: {fact.payload}")

        system_parts.extend([
            "\nReglas:",
            "- Responde con naturalidad.",
            "- No inventes hechos.",
            "- No añadas texto meta ni ejemplos.",
            "- Sé breve.",
        ])

        system = "\n".join(system_parts)
        user = context.input_text or ""

        return system, user

    # =========================
    # Model call — FIX BUG 4: usa chat() con system/user separados
    # =========================

    def _call_model(self, system: str, user: str, fallback: str) -> str:
        if self.conversation_model is None:
            return fallback

        try:
            # Usar chat() con separación system/user para evitar que el modelo
            # confunda instrucciones con input y repita la frase del usuario
            if hasattr(self.conversation_model, "chat") and callable(self.conversation_model.chat):
                text = self.conversation_model.chat([
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ])
                text = (text or "").strip()
                return text or fallback

            # Fallback por compatibilidad si solo hay complete()
            if hasattr(self.conversation_model, "complete") and callable(self.conversation_model.complete):
                combined = f"{system}\n\n{user}"
                text = self.conversation_model.complete(combined)
                text = (text or "").strip()
                return text or fallback

            raise AttributeError(
                f"{type(self.conversation_model).__name__} no expone chat() ni complete()"
            )

        except Exception as e:
            print(f"⚠️ Error en modelo conversacional: {e}")
            return fallback

    # =========================
    # Minimal fallbacks
    # =========================

    def _fallback_text(self, text: str) -> str:
        return text

    def _fallback_action_text(
        self,
        *,
        action_name: str | None,
        action_success: bool,
        action_payload: dict,
    ) -> str:
        if action_success:
            if action_name == "open_app":
                opened = action_payload.get("app_name")
                if opened:
                    return f"Abriendo {opened}."
            return "Hecho."

        return "Lo he intentado, pero algo no ha ido bien."

    def _fallback_reminder_text(self, payload: dict) -> str:
        title = payload.get("title") or "algo pendiente"
        message = payload.get("message")
        if message:
            return str(message)
        return f"Oye, te recuerdo: {title}"

    # =========================
    # Helpers
    # =========================

    def _format_datetime(self, iso_str: str | None) -> str:
        if not iso_str:
            return "fecha desconocida"

        # FIX BUG A: parsear ISO con tz y convertir siempre a Europe/Madrid
        # antes de formatear, por si el storage o el scheduler lo devolvió en UTC.
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo

            dt = datetime.fromisoformat(iso_str)

            # Si el ISO no traía tz, asumimos que venía en Madrid
            # (es lo que guarda temporal_interpreter originalmente)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("Europe/Madrid"))
            else:
                dt = dt.astimezone(ZoneInfo("Europe/Madrid"))

            return dt.strftime("%Y-%m-%d a las %H:%M")

        except Exception:
            # Fallback al parsing antiguo si el ISO es raro
            try:
                date_part = iso_str.split("T")[0]
                time_part = iso_str.split("T")[1][:5]
                return f"{date_part} a las {time_part}"
            except Exception:
                return iso_str