from __future__ import annotations

import os
import random
import uuid
from typing import Any

from app.cognitive.context_builder import BuiltContext
from app.cognitive.command_result import CommandResult
from app.cognitive.entity_resolver import entity_prompt_lines
from app.cognitive.models import DeliberationResult, ExecutionResult
from app.cognitive.persona.chatter_names import normalize_chatter_name
from app.cognitive.persona.hebe_voice import (
    build_chat_react_examples,
    build_stream_style_block as build_hebe_stream_style_block,
)
from app.cognitive.persona.hebe_identity import (
    build_hebe_core_identity,
    build_private_mode_style,
)
from app.cognitive.persona.reply_cleaner import (
    clean_jarvis_reply,
    clean_twitch_reply,
    detect_helper_pattern,
)
from app.cognitive.persona.stream_metrics import StreamReplyStats
from app.cognitive.persona.stream_dataset_logger import StreamDatasetLogger
from app.core.ui_bridge import emit


# Cuántas veces reintentamos la generación si detectamos un patrón helper.
# 1 retry suele bastar: con seed distinto, qwen 2.5:3b suele recuperarse.
# Subirlo aumenta latencia por mensaje y puede no aportar nada.
MAX_HELPER_RETRIES = int(os.getenv("HEBE_MAX_HELPER_RETRIES", "1"))


class ResponseSynthesizer:
    """
    Convierte:
    - contexto
    - resultado de deliberation
    - resultado de ejecución

    en una respuesta natural generada por el modelo conversacional.
    """

    def __init__(self, conversation_model: Any | None = None):
        self.conversation_model = conversation_model
        # Métricas acumuladas del stream (en memoria, se pierden al reiniciar).
        self._stream_stats = StreamReplyStats()
        self._dataset_logger = StreamDatasetLogger()

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

            if mode == "confirm_reminder":
                return self._generate_confirm_reminder(context, execution, reply_step.data)

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

    def _handle_internal_event(self, context: BuiltContext, execution: ExecutionResult) -> str:
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
            fallback = self._fallback_reminder_text(payload)
            return clean_jarvis_reply(self._call_model(system, user, fallback=fallback)) or fallback

        if event.event_type.startswith("twitch_"):
            return self._generate_twitch_reply(event, context)

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

        return clean_jarvis_reply(self._call_model(system, user, fallback=fallback)) or fallback

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

        return clean_jarvis_reply(self._call_model(system, user, fallback=fallback)) or fallback

    def _generate_confirm_reminder(
        self,
        context: BuiltContext,
        execution: ExecutionResult,
        reply_data: dict,
    ) -> str:
        reminder_result = execution.first_result_of_type("reminder")
        due_at = reply_data.get("due_at")
        relative_label = (reply_data.get("relative_label") or "").strip()
        message = reply_data.get("message") or reply_data.get("title") or "eso"
        if reminder_result:
            reminder = reminder_result.data.get("reminder")
            if reminder is not None:
                due_at = getattr(reminder, "due_at", due_at)
                message = getattr(reminder, "message", None) or getattr(reminder, "title", message)

        if relative_label:
            return f"Vale, Leo. Te aviso en {relative_label}."

        minutes = self._minutes_until(due_at)
        if minutes is not None and minutes <= 1:
            return "Vale, Leo. Te aviso en 1 minuto."
        if minutes is not None:
            return f"Vale, Leo. Te aviso en {minutes} minutos."
        return f"Vale, Leo. Te aviso: {message}."

    def _generate_clarification_reply(
        self,
        context: BuiltContext,
        reply_data: dict,
    ) -> str:
        system, user = self._build_clarification_prompt(
            reply_data=reply_data,
        )

        fallback = reply_data.get("question") or "No me ha quedado clara la fecha."
        return clean_jarvis_reply(self._call_model(system, user, fallback=fallback)) or fallback

    def _generate_chat_reply(self, context: BuiltContext) -> str:
        """
        Respuesta de Hebe en modo JARVIS (conversación directa con Leo desde la UI).

        Usa la identidad central de Hebe + estilo privado. No reutiliza el
        bloque Twitch, porque private/JARVIS tiene otro formato y longitud.

        Estructura del prompt:
          system  : voz + few-shots (siempre idéntico → cacheable)
          messages: [turn1_user, turn1_assistant, ..., current_user]
          current_user: mensaje de Leo PRIMERO, bloque de memoria al FINAL
        """
        msg = (context.input_text or "").strip()

        system = (
            f"{build_hebe_core_identity()}\n\n"
            f"{build_private_mode_style()}"
        )

        # Construcción del user actual: mensaje PRIMERO, memoria al FINAL.
        # El mensaje va primero para que el prefijo del último user sea relativamente
        # estable; la memoria (variable por similitud semántica) va al final.
        user_parts: list[str] = [
            "Speaker: Leo, your companion and broadcaster. "
            "Do not treat him like a random viewer. You can tease him with trust.\n\n"
            f"Message type: {getattr(context, 'message_type', 'unknown')}.\n"
            f"Context policy: {getattr(context, 'context_policy', {})}.\n"
            + (
                "This is casual small talk or banter. Answer directly in character, maximum two short sentences. "
                "Respond to Leo's mood first. Do not recap previous conversation, do not mention memory, "
                "do not mention calendar or stream schedule, and do not ask planning questions.\n\n"
                if getattr(context, "message_type", "unknown") in {"small_talk", "banter"}
                else ""
            )
            +
            f"Leo: {msg}"
        ]

        entity_lines = entity_prompt_lines(getattr(context, "resolved_entities", []) or [])
        if entity_lines:
            user_parts.append("Entity resolution:\n" + "\n".join(f"- {line}" for line in entity_lines))

        memory_lines: list[str] = []
        if context.relevant_facts:
            for fact in context.relevant_facts:
                memory_lines.append(f"- (about '{fact.subject}') {fact.payload}")
        if context.relevant_chunks:
            for ch in context.relevant_chunks:
                subj = ch.get("subject") or "general"
                text = ch.get("text", "")
                if text:
                    memory_lines.append(f"- (about '{subj}') {text}")
        memory_mode = (getattr(context, "context_policy", {}) or {}).get("memory")
        if memory_lines and memory_mode in {"full", "relevant"}:
            user_parts.append(
                "Relevant memory (each item is about a specific entity; "
                "do not merge details across unrelated items):\n"
                + "\n".join(memory_lines)
            )

        current_user_content = "\n\n".join(user_parts)

        # Construcción del array messages: historial + turno actual.
        # Los turnos históricos van limpios (sin bloque de memoria).
        messages: list[dict] = []
        for turn in context.conversation_history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": current_user_content})

        print(
            f"[HEBE][JARVIS][CHAT] msg={msg!r} "
            f"facts={len(context.relevant_facts)} "
            f"chunks={len(context.relevant_chunks)} "
            f"history={len(context.conversation_history)}",
            flush=True,
        )

        raw = self._call_model(system, messages=messages, fallback="")
        reply = clean_jarvis_reply(raw)

        print(
            f"[HEBE][JARVIS][REPLY] raw={raw!r} cleaned={reply!r}",
            flush=True,
        )

        return reply or self._fallback_text("No tengo una respuesta útil ahora mismo.")

    # =========================
    # Prompt builders — devuelven (system, user)
    # =========================

    def _build_system_style_block(self) -> str:
        return (
            f"{build_hebe_core_identity()}\n"
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



    def synthesize_command_result(self, result: CommandResult, *, input_text: str | None = None, state: Any | None = None) -> str:
        fallback = result.fallback_text or result.user_visible_summary or "Hecho."
        if not result.requires_model_response or not result.success:
            return fallback

        system = (
            f"{build_hebe_core_identity()}\n\n"
            "You are writing Hebe's final reply after deterministic code already updated state.\n"
            "Rules:\n"
            "- Speak as Hebe, warm, concise, slightly divine/playful if natural.\n"
            "- One short reply only, no markdown.\n"
            "- Do not undo, reinterpret, or add actions beyond the state_changes.\n"
            "- Do not ask for clarification if the action is already resolved.\n"
            "- Keep the exact meaning of the message_goal and state_changes.\n"
            "- If constraints forbid a topic, avoid it."
        )
        user = (
            "Manual command result:\n"
            f"action_type: {result.action_type}\n"
            f"success: {result.success}\n"
            f"user_input: {input_text or ''}\n"
            f"user_visible_summary: {result.user_visible_summary}\n"
            f"state_changes: {result.state_changes}\n"
            f"constraints: {result.constraints}\n"
            f"suggested_tone: {result.suggested_tone}\n"
            f"message_goal: {result.metadata.get('message_goal') or result.user_visible_summary}\n\n"
            "Write Hebe's reply now."
        )
        raw = self._call_model(system, user, fallback=fallback)
        reply = clean_jarvis_reply(raw).strip()
        if not self._valid_command_reply(reply, result):
            return fallback
        return reply

    def _valid_command_reply(self, reply: str, result: CommandResult) -> bool:
        if not reply:
            return False
        lowered = reply.lower()
        if result.action_type in {"tts_scope_resolved", "tts_disabled"} and "?" in reply:
            return False
        if result.action_type == "tts_scope_resolved" and result.metadata.get("scope") == "local":
            forbidden = ("tambien para el stream", "también para el stream", "en stream activ")
            if any(item in lowered for item in forbidden):
                return False
        return True

    # =========================
    # Model call con system/user separados
    # =========================

    def _call_model(
        self,
        system: str,
        user: str | None = None,
        fallback: str = "",
        *,
        messages: list[dict] | None = None,
        seed: int | None = None,
    ) -> str:
        """
        Llama al modelo conversacional con system/user separados.

        Acepta dos modos:
        - (system, user): single-turn, compatibilidad con todos los generators.
        - (system, messages=[...]): multi-turn, usado por _generate_chat_reply.

        seed: útil para retry en Twitch — cambia la generación entre intentos.
        """
        if self.conversation_model is None:
            return fallback

        try:
            num_predict = int(os.getenv("HEBE_REPLY_NUM_PREDICT", "120"))
            kwargs: dict[str, Any] = {"num_predict": num_predict}
            if seed is not None:
                kwargs["seed"] = seed

            if hasattr(self.conversation_model, "chat") and callable(self.conversation_model.chat):
                if messages is None:
                    messages = [{"role": "user", "content": user or ""}]
                full_messages = [{"role": "system", "content": system}] + messages
                text = self.conversation_model.chat(full_messages, **kwargs)
                text = (text or "").strip()
                # OllamaLLM.chat devuelve "…" si el modelo no produjo texto.
                # Lo tratamos como vacío para que el fallback funcione.
                if text in ("", "…"):
                    return fallback
                return text

            if hasattr(self.conversation_model, "complete") and callable(self.conversation_model.complete):
                if messages is None:
                    combined = f"{system}\n\n{user or ''}"
                else:
                    body = "\n\n".join(
                        f"{m['role']}: {m['content']}" for m in messages
                    )
                    combined = f"{system}\n\n{body}"
                text = self.conversation_model.complete(combined, **kwargs)
                text = (text or "").strip()
                if text in ("", "…"):
                    return fallback
                return text

            raise AttributeError(
                f"{type(self.conversation_model).__name__} no expone chat() ni complete()"
            )

        except Exception as e:
            print(f"⚠️ Error en modelo conversacional: {e}", flush=True)
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

        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo

            dt = datetime.fromisoformat(iso_str)

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("Europe/Madrid"))
            else:
                dt = dt.astimezone(ZoneInfo("Europe/Madrid"))

            return dt.strftime("%Y-%m-%d a las %H:%M")

        except Exception:
            try:
                date_part = iso_str.split("T")[0]
                time_part = iso_str.split("T")[1][:5]
                return f"{date_part} a las {time_part}"
            except Exception:
                return iso_str

    def _minutes_until(self, iso_str: str | None) -> int | None:
        if not iso_str:
            return None
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo

            due = datetime.fromisoformat(str(iso_str))
            if due.tzinfo is None:
                due = due.replace(tzinfo=ZoneInfo("Europe/Madrid"))
            now = datetime.now(ZoneInfo("Europe/Madrid"))
            seconds = max(0.0, (due.astimezone(ZoneInfo("Europe/Madrid")) - now).total_seconds())
            return max(1, int(round(seconds / 60.0)))
        except Exception:
            return None

    # =========================
    # Twitch / stream generation
    # =========================

    def _generate_twitch_reply(self, event, context: BuiltContext | None = None) -> str:
        payload = event.payload or {}

        if event.event_type == "twitch_sub":
            return self._generate_twitch_sub(payload)
        if event.event_type == "twitch_raid":
            return self._generate_twitch_raid(payload)
        if event.event_type == "twitch_follow_batch":
            return self._generate_twitch_follow_batch(payload)
        if event.event_type == "twitch_chat_react":
            return self._generate_twitch_chat_react(payload, context=context)
        if event.event_type == "twitch_idle_prompt":
            return self._generate_twitch_idle_prompt(payload)

        return self._fallback_text("")

    def _generate_twitch_idle_prompt(self, payload: dict) -> str:
        presence_mode = payload.get("presence_mode") or "companion"
        title = payload.get("title") or "unknown"
        category = payload.get("current_category") or payload.get("current_game") or "unknown"
        playthrough_type = payload.get("playthrough_type") or "unknown"
        challenge = payload.get("challenge") or "none"
        stream_slot = payload.get("stream_slot") or "unknown"
        language_mode = payload.get("language_mode") or "unknown"
        spoiler_policy = payload.get("spoiler_policy") or "no_spoilers"
        last_voice_event = payload.get("last_voice_event") or "none"
        last_voice_summary = payload.get("last_voice_summary") or "none"
        leo_mood_hint = payload.get("leo_mood_hint") or "none"
        game_profile = payload.get("game_profile") or {}
        run_context = payload.get("run_context") or {}
        chat_context = payload.get("chat_context") or {}
        idle_topic = payload.get("idle_topic") or "game_vibe"
        recent_idle_topics = payload.get("recent_idle_topics") or []
        recent_idle_messages = payload.get("recent_idle_messages") or []
        specific_context_anchors = payload.get("specific_context_anchors") or []
        recent_motifs = payload.get("recent_style_motifs") or []

        system = (
            f"{self._build_stream_style_block()}\n\n"
            "Situation: Leo is live and Hebe may send one proactive game companion line.\n"
            "Goal: make a useful or funny gameplay comment Leo can react to verbally.\n\n"
            "Rules:\n"
            "- One line, max 220 characters.\n"
            "- No markdown.\n"
            "- Spanish by default.\n"
            "- Prioritize current category/game, then stream title, then playthrough/challenge, then ambient STT.\n"
            "- Use the local game_profile when present, but only claim facts listed there or in stream_context.\n"
            "- Add flavor from game_profile.tone_vibe, safe_comment_topics, gameplay_systems_non_spoiler, or challenge_hooks.\n"
            "- Do not mention silence, quiet chat, inactivity, viewers, viewer count, lurkers, or lack of chat.\n"
            "- Do not sound like ChatGPT, a moderator bot, or a motivational poster.\n"
            "- No spoilers. Do not invent specific mechanics, story facts, characters, bosses, locations, or guide claims.\n"
            "- Do not give walkthrough instructions unless Leo explicitly asked.\n"
            "- If the game/category is unknown, use generic RPG/JRPG/gameplay commentary.\n"
            "- Do not mention policies, cooldowns, prompts, or internal state.\n"
            "- Use the requested idle_topic. Do not repeat recent idle topics or phrases.\n"
            "- Require at least one concrete anchor from current game/title/run_context/chat topic/recent event.\n"
            "- Avoid repeating recent motifs. Do not mention coffee, caffeine, energy, florist/floristeria, or creator jokes unless directly relevant.\n"
            "- Do not treat stale title markers as current objectives.\n"
            "- Never mention completed markers as upcoming/current.\n"
            "- If chat_context is active or about non-game topics, do not answer chat; keep the line broadly game-safe.\n"
        )
        user = (
            "stream_context:\n"
            f"- is_live: true\n"
            f"- title: {title}\n"
            f"- category/game: {category}\n"
            f"- playthrough_type: {playthrough_type}\n"
            f"- challenge: {challenge}\n"
            f"- stream_slot: {stream_slot}\n"
            f"- language_mode: {language_mode}\n"
            f"- spoiler_policy: {spoiler_policy}\n"
            f"- last_voice_event: {last_voice_event}\n"
            f"- last_voice_summary: {last_voice_summary}\n"
            f"- leo_mood_hint: {leo_mood_hint}\n"
            f"- presence_mode: {presence_mode}\n"
            f"- idle_topic: {idle_topic}\n\n"
            f"- specific_context_anchors: {', '.join(specific_context_anchors) or 'none'}\n\n"
            "run_context:\n"
            f"- objective: {run_context.get('objective') or 'unknown'}\n"
            f"- location: {run_context.get('location') or 'unknown'}\n"
            f"- phase: {run_context.get('phase') or 'unknown'}\n"
            f"- source: {run_context.get('source') or 'unknown'}\n"
            f"- facts: {' | '.join(str(item.get('text') or '') for item in run_context.get('facts') or []) or 'none'}\n"
            f"- completed_markers: {', '.join(run_context.get('completed_markers') or []) or 'none'}\n"
            f"- title_markers_fresh: {', '.join(run_context.get('title_markers_fresh') or []) or 'none'}\n"
            f"- title_markers_stale: {', '.join(run_context.get('title_markers_stale') or []) or 'none'}\n\n"
            "chat_context:\n"
            f"- active: {bool(chat_context.get('active'))}\n"
            f"- recent_count: {chat_context.get('recent_count') or 0}\n"
            f"- recent_topics: {', '.join(chat_context.get('recent_topics') or []) or 'none'}\n"
            f"- summary: {chat_context.get('summary') or 'none'}\n\n"
            "recent_idle:\n"
            f"- topics: {', '.join(recent_idle_topics) or 'none'}\n"
            f"- messages: {' | '.join(str(item) for item in recent_idle_messages) or 'none'}\n\n"
            f"- recent_motifs: {', '.join(str(item) for item in recent_motifs) or 'none'}\n\n"
            "game_profile:\n"
            f"- title: {game_profile.get('title') or 'unknown'}\n"
            f"- source_category_name: {game_profile.get('source_category_name') or 'unknown'}\n"
            f"- genres: {', '.join(game_profile.get('genres') or []) or 'unknown'}\n"
            f"- tone_vibe: {game_profile.get('tone_vibe') or 'unknown'}\n"
            f"- general_non_spoiler_summary: {game_profile.get('general_non_spoiler_summary') or 'unknown'}\n"
            f"- gameplay_systems_non_spoiler: {', '.join(game_profile.get('gameplay_systems_non_spoiler') or []) or 'none'}\n"
            f"- channel_context: {game_profile.get('channel_context') or 'unknown'}\n"
            f"- leo_relationship: {game_profile.get('leo_relationship') or 'unknown'}\n"
            f"- safe_comment_topics: {', '.join(game_profile.get('safe_comment_topics') or []) or 'none'}\n"
            f"- spoiler_policy: {game_profile.get('spoiler_policy') or 'no_spoilers'}\n"
            f"- unsafe_comment_topics: {', '.join(game_profile.get('unsafe_comment_topics') or []) or 'none'}\n"
            f"- stream_hooks: {', '.join(game_profile.get('stream_hooks') or []) or 'none'}\n"
            f"- challenge_notes: {', '.join(game_profile.get('challenge_notes') or []) or 'none'}\n"
            f"- challenge_hooks: {', '.join(game_profile.get('challenge_hooks') or []) or 'none'}\n\n"
            "Generate only Hebe's Twitch chat message."
        )
        fallback = random.choice(
            [
                "Mi senor, antes de lanzarte al caos, revisa vida, recursos y si puedes guardar.",
                "Esto huele a zona donde el juego espera que vayas confiado. Sospechoso.",
                "En una run de desafio, cada menu cuenta. El verdadero boss es la preparacion.",
                "Esto es muy JRPG: abres una puerta y acabas resolviendo un problema geopolitico.",
                "Mi voto divino: guardar partida antes de hacer cualquier cosa remotamente estupida.",
            ]
        )
        reply = clean_twitch_reply(self._call_model(system, user, fallback=fallback))[:220]
        return self._safe_spontaneous_stream_reply(reply, fallback, payload=payload)

    def generate_twitch_idle_prompt_preview(self, payload: dict) -> str:
        return self._generate_twitch_idle_prompt(payload)

    def _safe_spontaneous_stream_reply(self, reply: str, fallback: str, payload: dict | None = None) -> str:
        text = (reply or "").strip()
        lowered = text.lower()
        payload = payload or {}
        if not self._has_specific_anchor(payload):
            return ""
        if lowered in {".", "..", "...", "....", ".....", "......", "…"}:
            return ""
        forbidden = (
            "silencio",
            "silencio en la sala",
            "quieto",
            "tranquilo",
            "está esto tranquilo",
            "esta esto tranquilo",
            "nadie",
            "sin chat",
            "chat esta",
            "chat está",
            "chat está muerto",
            "chat esta muerto",
            "no habla",
            "nadie habla",
            "nadie está hablando",
            "nadie esta hablando",
            "inactivo",
            "muerto",
            "no viewers",
            "viewer",
            "viewers",
            "espectador",
            "espectadores",
            "lurking",
            "lurker",
            "lurkers",
            "si alguien está",
            "si alguien esta",
            "aunque no haya",
            "aunque no haya nadie",
        )
        game_profile = payload.get("game_profile") or {}
        game_title = " ".join(
            str(value or "").lower()
            for value in (
                payload.get("current_game"),
                payload.get("current_category"),
                game_profile.get("title"),
            )
        )
        if "final fantasy ix" in game_title or "ffix" in game_title or "ff9" in game_title:
            if "esfera" in lowered or "esferas" in lowered:
                return fallback

        run_context = payload.get("run_context") or {}
        completed = [str(item).lower() for item in run_context.get("completed_markers") or []]
        stale = [str(item).lower() for item in run_context.get("title_markers_stale") or []]
        if any(marker and marker in lowered for marker in completed + stale):
            return fallback

        if self._response_repeats_motif(text, payload):
            return fallback

        if not text or any(marker in lowered for marker in forbidden):
            return fallback
        return text

    def _has_specific_anchor(self, payload: dict) -> bool:
        if "specific_context_anchors" not in payload:
            return True
        anchors = payload.get("specific_context_anchors") or []
        if anchors:
            return True
        if payload.get("current_game") or payload.get("current_category") or payload.get("title"):
            return True
        run_context = payload.get("run_context") or {}
        if any(run_context.get(key) for key in ("objective", "location", "phase")):
            return True
        chat_context = payload.get("chat_context") or {}
        if chat_context.get("recent_topics"):
            return True
        return False

    def _response_repeats_motif(self, text: str, payload: dict) -> bool:
        motifs = self._detect_motifs(text)
        if not motifs:
            return False
        recent = {
            str(item or "").lower()
            for item in (payload.get("recent_style_motifs") or [])
        }
        aliases = {"coffee": "cafe"}
        normalized_recent = {aliases.get(item, item) for item in recent}
        overused = {
            item.strip().lower()
            for item in os.getenv("HEBE_STYLE_OVERUSED_MOTIFS", "cafe,coffee,energy,florist,creator").split(",")
            if item.strip()
        }
        normalized_overused = {aliases.get(item, item) for item in overused}
        return any(aliases.get(motif, motif) in normalized_recent | normalized_overused for motif in motifs)

    def _detect_motifs(self, text: str) -> list[str]:
        lowered = str(text or "").lower()
        terms = {
            "cafe": ("cafe", "café", "coffee", "cafeina", "cafeína"),
            "energy": ("energia", "energía", "pilas"),
            "florist": ("florist", "florister", "flores"),
            "creator": ("creador", "creadores"),
            "chaos": ("caos", "caotico", "caótico"),
        }
        return [motif for motif, values in terms.items() if any(value in lowered for value in values)]

    def generate_stream_presence(
        self,
        *,
        reason: str,
        presence_mode: str,
        last_voice_event: str | None = None,
        leo_mood_hint: str | None = None,
    ) -> str:
        return self._generate_twitch_idle_prompt(
            {
                "reason": reason,
                "presence_mode": presence_mode,
                "last_voice_event": last_voice_event,
                "leo_mood_hint": leo_mood_hint,
            }
        )

    def _build_stream_style_block(self) -> str:
        return build_hebe_stream_style_block()

    def _generate_twitch_sub(self, payload: dict) -> str:
        display_name = payload.get("display_name") or payload.get("user_login") or "alguien"
        cumulative_months = int(payload.get("cumulative_months") or 1)
        is_resub = cumulative_months > 1
        is_gift = bool(payload.get("is_gift"))
        gifter_name = payload.get("gifter_display_name")

        if is_gift and gifter_name:
            situation = f"{gifter_name} acaba de regalar una sub a {display_name}."
        elif is_resub:
            situation = f"{display_name} se ha re-suscrito ({cumulative_months} meses seguidos)."
        else:
            situation = f"{display_name} acaba de hacerse sub por primera vez."

        system = (
            f"{self._build_stream_style_block()}\n\n"
            f"Situación: {situation}\n"
            "Objetivo: agradecer de forma natural, breve y con energía.\n\n"
            "Reglas:\n"
            f"- El nombre exacto a usar es: {display_name}\n"
            "- Una sola frase. Máximo 15 palabras.\n"
            "- Usa el nombre exactamente como aparece, respetando mayúsculas.\n"
            "- No generes diálogos ni turnos."
        )
        user = "Genera SOLO el mensaje final de Hebe para enviar al chat de Twitch."
        fallback = f"Gracias por la sub, {display_name}."
        reply = self._call_model(system, user, fallback=fallback)
        return clean_twitch_reply(reply)

    def _generate_twitch_raid(self, payload: dict) -> str:
        display_name = payload.get("display_name") or payload.get("user_login") or "alguien"
        viewer_count = int(payload.get("viewer_count") or 0)

        system = (
            f"{self._build_stream_style_block()}\n\n"
            f"Situación: {display_name} acaba de hacer raid al canal con {viewer_count} viewers.\n"
            "Objetivo: dar la bienvenida al raid de forma natural y con calor.\n\n"
            "Reglas:\n"
            f"- Nombre exacto: {display_name}\n"
            f"- Número de viewers exacto: {viewer_count}. No inventes otro número.\n"
            "- Una o dos frases. Máximo 25 palabras.\n"
            "- No generes diálogos ni turnos."
        )
        user = "Genera SOLO el mensaje final de Hebe para enviar al chat de Twitch."
        fallback = f"Bienvenidos los del raid de {display_name}."
        reply = self._call_model(system, user, fallback=fallback)
        return clean_twitch_reply(reply)

    def _generate_twitch_follow_batch(self, payload: dict) -> str:
        names = payload.get("display_names") or []
        count = int(payload.get("count") or len(names))

        if not names:
            return self._fallback_text("")

        if len(names) == 1:
            situation = f"{names[0]} acaba de seguir el canal."
        else:
            joined = ", ".join(names[:-1]) + f" y {names[-1]}"
            situation = f"Han seguido el canal {joined} ({count} en total)."

        system = (
            f"{self._build_stream_style_block()}\n\n"
            f"Situación: {situation}\n"
            "Objetivo: dar la bienvenida muy breve.\n\n"
            "Reglas:\n"
            f"- Nombres exactos: {names}\n"
            "- Una frase. Máximo 15 palabras.\n"
            "- Si hay varios, agrúpalos sin enumerar mucho.\n"
            "- No generes diálogos ni turnos."
        )
        user = "Genera SOLO el mensaje final de Hebe para enviar al chat de Twitch."
        fallback = f"Gracias por el follow, {names[0]}."
        reply = self._call_model(system, user, fallback=fallback)
        return clean_twitch_reply(reply)

    def _generate_twitch_chat_react(self, payload: dict, context: BuiltContext | None = None) -> str:
        """
        Reacción a un mensaje de chat clasificado como digno de respuesta.

        Usa formato de continuación [chatter Nombre]: ... \\n[tú]: para que
        el modelo complete según los few-shots de hebe_voice.

        Flujo (28/04 — añadido retry + filtro helper):
          1. Normaliza el display_name del chatter ('nuriiia___' -> 'Nuria').
          2. Detecta is_broadcaster con la lógica rica habitual.
          3. Construye el prompt con [chatter Nombre]: msg\\n[tú]:.
          4. Llama al modelo. Aplica clean_twitch_reply al resultado.
          5. Si el resultado engancha algún patrón helper, retry con seed
             distinto (hasta MAX_HELPER_RETRIES). Si retry vuelve a fallar,
             publica la respuesta igualmente — mejor algo imperfecto que
             silencio en chat.
          6. Registra métricas en self._stream_stats. Cada 50 mensajes
             vuelca un STREAM_SUMMARY para datos parciales aunque caiga
             el backend.

        Logs emitidos (todos con tag [HEBE][REPLY] para grepear):
          - BEGIN  : entrada del flujo, datos crudos.
          - RAW    : la respuesta del modelo, cruda y limpiada (por intento).
          - HELPER_DETECTED   : si engancha un patrón helper.
          - HELPER_PUBLISHED  : si tras todos los retries seguía enganchando.
          - END    : resumen del flujo de esta respuesta.
        """
        user_login = (payload.get("user_login") or "").strip()
        display_name_raw = payload.get("display_name") or user_login or ""
        chatter_clean = normalize_chatter_name(display_name_raw)
        message = (payload.get("message_text") or "").strip()
        recent = payload.get("recent_chat") or []

        is_broadcaster = self._is_broadcaster(payload)

        # trace_id corto para correlacionar todas las líneas de log de
        # esta generación. Si el caller ya pasa uno (porque viene del
        # scheduler con un trace de mayor nivel), úsalo.
        trace_id = payload.get("trace_id") or uuid.uuid4().hex[:8]

        # Bloque de contexto reciente, si lo hay.
        recent_block = ""
        if recent:
            lines = [
                f"- {m.get('display_name', '?')}: {m.get('text', '')}"
                for m in recent[-6:]
            ]
            recent_block = "Contexto del chat reciente:\n" + "\n".join(lines)

        # Aviso sobre quién está hablando. Va al user, no al system,
        # para no romper el caché del system prompt.
        speaker_block = (
            "IMPORTANTE: quien escribe este mensaje ES Leo, tu compañero y broadcaster. "
            "No lo trates como un viewer cualquiera. Puedes vacilarle con confianza."
            if is_broadcaster
            else ""
        )

        # System: SOLO identidad + few-shots (siempre idéntico → cacheable).
        system = (
            f"{self._build_stream_style_block()}\n\n"
            f"{build_chat_react_examples()}"
        )

        # Si hay viewer_facts para este chatter, añadir perfil compacto al system.
        # Nota: esto rompe el caché de OpenAI solo cuando existen facts del viewer.
        # Para la mayoría de viewers (sin facts) el system sigue siendo idéntico.
        if context is not None and context.relevant_chunks:
            profile_bits: list[str] = []
            for ch in context.relevant_chunks:
                text = ch.get("text", "")
                if text:
                    profile_bits.append(text)
            if profile_bits:
                profile_line = "; ".join(profile_bits)
                system = system + f"\n\nSobre este viewer: {profile_line}"

        # User: parte variable (contexto + mensaje). El modelo completa
        # después de [tú]:. Siempre usamos el nombre limpio del chatter.
        chatter_tag = (
            "[chatter Leo]:"
            if is_broadcaster
            else f"[chatter {chatter_clean}]:"
        )
        user_parts: list[str] = []
        if speaker_block:
            user_parts.append(speaker_block)
        if recent_block:
            user_parts.append(recent_block)
        user_parts.append(f"{chatter_tag} {message}\n[tú]:")
        user = "\n\n".join(user_parts)

        print(
            f"[HEBE][REPLY][BEGIN] trace={trace_id} "
            f"chatter_raw={display_name_raw!r} chatter_clean={chatter_clean!r} "
            f"is_broadcaster={is_broadcaster} msg={message!r}",
            flush=True,
        )

        helper_hits: list[str] = []
        final_reply = ""
        final_raw = ""
        final_helper: str | None = None

        for attempt in range(MAX_HELPER_RETRIES + 1):
            # Seed distinto en cada intento para que el retry no regenere
            # exactamente lo mismo. Si OllamaLLM no acepta seed como kwarg,
            # se pasa al options de Ollama y se ignora silenciosamente si
            # no lo soporta tu versión del wrapper.
            seed = random.randint(0, 1_000_000)
            if os.getenv("HEBE_PROMPT_DEBUG", "false").strip().lower() in ("1", "true", "yes", "on"):
                print("[HEBE][REPLY][PROMPT_DEBUG]", system, user, flush=True)
            raw = self._call_model(system, user, fallback="", seed=seed)
            cleaned = clean_twitch_reply(raw, source_message=message)

            print(
                f"[HEBE][REPLY][RAW] trace={trace_id} attempt={attempt} "
                f"seed={seed} raw={raw!r} cleaned={cleaned!r}",
                flush=True,
            )

            helper = detect_helper_pattern(cleaned)
            if helper is None:
                # Respuesta limpia (o vacía, que también está bien — el
                # caller publicará "" como silencio).
                final_reply = cleaned
                final_raw = raw
                final_helper = None
                break

            # Engancha un patrón helper.
            helper_hits.append(helper)
            print(
                f"[HEBE][REPLY][HELPER_DETECTED] trace={trace_id} "
                f"attempt={attempt} pattern={helper} text={cleaned!r}",
                flush=True,
            )

            if attempt == MAX_HELPER_RETRIES:
                # Se acabaron los reintentos. No publicamos una fuga helper:
                # en directo es mejor un fallback seco que mandar al chat
                # "¿en qué puedo ayudarte?" o un roleplay roto.
                final_reply = self._fallback_twitch_chat_react(
                    chatter=chatter_clean,
                    message=message,
                    is_broadcaster=is_broadcaster,
                )
                final_raw = raw
                final_helper = helper
                print(
                    f"[HEBE][REPLY][HELPER_BLOCKED] trace={trace_id} "
                    f"patterns={helper_hits} fallback={final_reply!r}",
                    flush=True,
                )

        retried = len(helper_hits) > 0
        salvaged = retried and final_helper is None

        print(
            f"[HEBE][REPLY][END] trace={trace_id} helper_hits={helper_hits} "
            f"retried={retried} salvaged={salvaged} final={final_reply!r}",
            flush=True,
        )

        # Registrar en métricas acumuladas del stream.
        self._stream_stats.record(
            chatter=chatter_clean,
            helper_hits=helper_hits,
            retried=retried,
            salvaged=salvaged,
        )

        # Volcar resumen cada 50 mensajes para tener datos parciales aunque
        # se caiga el backend antes de que termine el stream.
        if self._stream_stats.total > 0 and self._stream_stats.total % 50 == 0:
            self._stream_stats.log_summary()

        model_meta = self._model_meta()

        self._dataset_logger.log_twitch_chat_react(
            trace_id=trace_id,
            payload=payload,
            chatter_clean=chatter_clean,
            is_broadcaster=is_broadcaster,
            raw_response=final_raw,
            cleaned_response=final_reply,
            helper_hits=helper_hits,
            retried=retried,
            salvaged=salvaged,
            model_meta=model_meta,
            full_prompt={"system": system, "user": user},
        )

        # Avisar a la UI de que esta respuesta tiene ejemplo de dataset
        # para poder mostrar el mensaje original y botones de curación.
        try:
            emit(
                "dataset.example",
                {
                    "trace_id": trace_id,
                    "event_type": "twitch_chat_react",
                    "user_login": user_login,
                    "display_name": display_name_raw,
                    "chatter_clean": chatter_clean,
                    "is_broadcaster": is_broadcaster,
                    "message": message,
                    "response": final_reply,
                    "model": model_meta,
                    "curation": {
                        "status": None,
                        "approved": None,
                        "corrected_response": None,
                        "notes": None,
                        "tags": [],
                    },
                },
            )
        except Exception as exc:
            print(f"[HEBE][DATASET][UI_EVENT_ERROR] {exc!r}", flush=True)

        return final_reply

    def _model_meta(self) -> dict[str, Any]:
        model = self.conversation_model
        if model is None:
            return {"provider": "none"}

        # Si usamos FallbackConversationLLM, el objeto externo es el wrapper.
        # Para dataset nos interesa el provider real que respondió en la última llamada.
        actual = getattr(model, "last_used", None) or getattr(model, "primary", None) or model
        class_name = type(actual).__name__
        provider_attr = getattr(actual, "provider", None)
        provider = str(provider_attr or "").strip().lower()
        if not provider:
            provider = "openai" if class_name.lower().startswith("openai") else "local"

        meta: dict[str, Any] = {
            "provider": provider,
            "class": class_name,
            "name": getattr(actual, "model", None),
        }

        usage = getattr(actual, "last_usage", None)
        if isinstance(usage, dict):
            meta["usage"] = usage

        elapsed_ms = getattr(actual, "last_elapsed_ms", None)
        if elapsed_ms is not None:
            meta["elapsed_ms"] = elapsed_ms

        wrapper_name = type(model).__name__
        if actual is not model:
            meta["wrapper"] = wrapper_name

        return meta

    def _fallback_twitch_chat_react(
        self,
        *,
        chatter: str,
        message: str,
        is_broadcaster: bool,
    ) -> str:
        """
        Fallback determinista para Twitch.

        No intenta ser brillante; intenta no romper personaje ni publicar
        basura si el modelo devuelve una fuga helper/roleplay.
        """
        msg = (message or "").lower().strip()
        name = "Leo" if is_broadcaster else (chatter or "chat")

        if any(word in msg for word in ("hola", "buenas", "ey", "hey")):
            return f"hola, {name}. ¿qué cuentas?" if not is_broadcaster else "hola, Leo. te leo."

        if "quien eres" in msg or "quién eres" in msg:
            return (
                "soy Hebe, Leo. tu compañera de chat, no una etiqueta rara."
                if is_broadcaster
                else f"soy Hebe, {name}. intento poner algo de criterio por aquí."
            )

        if "mal hebe" in msg or "eso ha salido" in msg or "qué estás diciendo" in msg or "que estas diciendo" in msg:
            return "sí, esa ha salido torcida. recalibro." if is_broadcaster else f"he derrapado un poco, {name}. recalibro."

        if "relajate" in msg or "relájate" in msg:
            return "vale, Leo. bajo dos tonos." if is_broadcaster else f"voy bajando el filo, {name}."

        if "vete a la mierda" in msg:
            return (
                "vale, Leo. bajo el filo, pero no me entierres todavía."
                if is_broadcaster
                else f"con cariño, {name}: primero aprende a saludar."
            )

        if "personalidad" in msg:
            return (
                "la tengo, Leo. solo estoy aprendiendo a no atropellarte con ella."
                if is_broadcaster
                else f"personalidad tengo, {name}. paciencia con el despliegue."
            )

        if is_broadcaster:
            return "te leo, Leo. sigo calibrando."

        return f"te leo, {name}."

    def _is_broadcaster(self, payload: dict) -> bool:
        if bool(payload.get("is_broadcaster")):
            return True

        candidates = {
            str(payload.get("user_login") or "").lower().strip(),
            str(payload.get("display_name") or "").lower().strip(),
            str(payload.get("chatter_user_login") or "").lower().strip(),
            str(payload.get("chatter_user_name") or "").lower().strip(),
        }

        candidates.discard("")

        broadcaster_aliases = {
            "leonifelheim",
            "leo_nifelheim",
            "leo nifelheim",
            "leo",
        }

        return bool(candidates & broadcaster_aliases)
