from __future__ import annotations

import os
import random
import re
import unicodedata
import uuid
from typing import Any

from app.cognitive.context_builder import BuiltContext
from app.cognitive.command_result import CommandResult
from app.cognitive.entity_resolver import entity_prompt_lines
from app.cognitive.models import DeliberationResult, ExecutionResult
from app.cognitive.game_guidance import GameGuidanceCapability
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
from app.cognitive.speech_act_pipeline import (
    build_persona_renderer_messages,
    build_repair_renderer_messages,
    build_twitch_speech_act_bundle,
    final_response_guard,
    safe_local_fallback,
)
from app.core.ui_bridge import emit
from app.stream.game_advice_gate import GameAdviceGate


# Cuantas veces reintentamos la generacion si detectamos un patron helper.
# 1 retry suele bastar: con seed distinto, qwen 2.5:3b suele recuperarse.
# Subirlo aumenta latencia por mensaje y puede no aportar nada.
MAX_HELPER_RETRIES = int(os.getenv("HEBE_MAX_HELPER_RETRIES", "1"))

_ASSISTANT_OFFER_PHRASES = (
    "tomo nota",
    "te lo guardo",
    "lo guardo",
    "guardar en memoria",
    "guarde en memoria",
    "lo publico",
    "lo publicar",
    "publicarlo en stream",
    "publique en el stream",
    "publicar en stream",
    "usar como linea",
    "use como linea",
    "responda solo cuando",
    "dimelo claro",
    "dimelo claro",
    "tu mandas creador",
    "tu mandas creador",
    "tu mandas, creador",
    "tu mandas, creador",
    "puedo ayudarte",
    "tu que tal",
    "tu que tal",
    "quieres que",
    "quieres que",
)


_ACTION_OFFER_PHRASES = (
    "lo publico",
    "lo publicar",
    "publicarlo",
    "publicarlo en stream",
    "publique en el stream",
    "publicar en stream",
    "usar como linea",
    "use como linea",
    "guardar en memoria",
    "guarde en memoria",
)

_STYLE_GUARD_PERSONALITY_FALLBACKS = (
    "personality fallback marker",
)

_STYLE_GUARD_MINIMAL_FALLBACK = "No te he entendido bien."


class ResponseSynthesizer:
    """
    Convierte:
    - contexto
    - resultado de deliberation
    - resultado de ejecucion

    en una respuesta natural generada por el modelo conversacional.
    """

    def __init__(self, conversation_model: Any | None = None):
        self.conversation_model = conversation_model
        self.last_opens_conversation_turn = False
        self.last_expected_reply_type = ""
        # Metricas acumuladas del stream (en memoria, se pierden al reiniciar).
        self._stream_stats = StreamReplyStats()
        self._dataset_logger = StreamDatasetLogger()
        self.game_advice_gate = GameAdviceGate()
        self._style_guard_fallback_counts: dict[str, int] = {}
        self._game_guidance_classifier = GameGuidanceCapability()

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
                return self._generate_chat_reply(context, execution)

            if mode == "clarify_appointment_datetime":
                return self._generate_clarification_reply(context, reply_step.data)

            if mode == "capability_catalogue_query":
                return self._generate_capability_catalogue_reply(reply_step.data)

            if mode == "time_answer":
                return f"Son las {reply_step.data.get('time')} en Madrid."

            if mode == "date_answer":
                return f"Hoy es {reply_step.data.get('date')}."

            if mode == "companion_reaction":
                return self._generate_personal_state_reply(context, reply_step.data)

            if mode in {"game_guidance", "game_guidance_clarification"}:
                return self._generate_game_guidance_reply(context, reply_step.data)

        return self._fallback_text("No tengo suficiente contexto para responder con seguridad.")

    def _generate_personal_state_reply(self, context: BuiltContext, data: dict) -> str:
        state = str(data.get("state") or "unknown")
        system = (
            "Respond as Hebe, Leo's close companion. React naturally to the personal state he shared. "
            "Be concise and warm. You may offer one practical suggestion, but do not schedule, remind, "
            "diagnose, or claim to have performed an action."
        )
        user = f"Personal state category: {state}\nUser message: {context.input_text or ''}"
        fallback = "Te escucho, Leo. CuÃ­date un poco ahora mismo."
        return clean_jarvis_reply(self._call_model(system, user, fallback=fallback)) or fallback

    def _generate_game_guidance_reply(self, context: BuiltContext, data: dict) -> str:
        decision = dict(data.get("game_guidance") or {})
        guidance = dict(decision.get("context") or {})
        game = str(guidance.get("game") or "").strip()
        ambiguity = list(guidance.get("ambiguity_reasons") or [])
        needs_clarification = bool(guidance.get("needs_clarification"))
        sources = list(decision.get("rag_chunks") or []) + list(decision.get("web_results") or [])
        if needs_clarification:
            missing = "permission to discuss story spoilers" if "major_spoiler_permission_required" in ambiguity else (
                "current character" if "character_unknown" in ambiguity else (
                    "latest confirmed event or objective" if "story_phase_unknown" in ambiguity else "current area"
                )
            )
            system = (
                "You are Hebe speaking briefly to Leo. Ask exactly one useful game-state clarification. "
                "Be playful but do not provide route steps, item locations, boss facts, or story claims."
            )
            user = f"Game: {game or 'unknown'}\nMissing state: {missing}\nUser message: {context.input_text or ''}"
            fallback = f"Necesito confirmar {missing} antes de orientarte sin inventar la ruta."
            return clean_jarvis_reply(self._call_model(system, user, fallback=fallback)) or fallback
        if not sources:
            print("[HEBE][GAME_SOURCE] tier=all status=skipped reason=no_grounded_guidance_source", flush=True)
            return "No tengo una fuente de guía fiable para concretar ese paso; necesito más contexto o consultar una fuente antes de afirmarlo."
        system = (
            "Answer as Hebe, concise and streamer-friendly. Use only the supplied game sources. "
            "Respect the spoiler policy and allowed depth. State uncertainty where sources do not settle a claim. "
            "Never add walkthrough facts from general model memory."
        )
        user = (
            f"Game guidance context: {guidance}\n"
            f"Grounding sources: {sources[:6]}\n"
            f"Leo: {context.input_text or ''}"
        )
        fallback = "Tengo contexto de la partida, pero las fuentes disponibles no bastan para darte una indicación segura."
        return clean_jarvis_reply(self._call_model(system, user, fallback=fallback)) or fallback

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

        return self._fallback_text("Ha ocurrido algo, pero no tengo claro que.")

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
            f"Vale, te lo guardo: {title} el {self._format_datetime(due_at)}. Te avisare cuando toque."
            if due_at
            else f"Vale, te lo guardo: {title}. Te avisare cuando toque."
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

    def _generate_capability_catalogue_reply(self, reply_data: dict) -> str:
        payload = reply_data.get("payload") or {}
        query_type = reply_data.get("query_type") or payload.get("query_type") or "summary"
        if payload.get("catalogue_unavailable"):
            return "No puedo leer el catalogo de capabilities ahora mismo, asi que no puedo responder ese TODO con seguridad."
        items = payload.get("items") or []
        if query_type == "next_todo":
            item = payload.get("next_recommended_todo") or (items[0] if items else None)
            if not item:
                return "No veo ningun TODO recomendado ahora mismo."
            backlog = item.get("backlog") or {}
            actions = backlog.get("next_actions") or []
            suffix = f" Siguiente accion: {actions[0]}" if actions else ""
            return f"El siguiente TODO recomendado es {item.get('id')}: {item.get('name')}.{suffix}"

        labels = {
            "planned_not_implemented": "planeadas sin implementar",
            "high_priority_unblocked": "de alta prioridad desbloqueadas",
            "implemented_disabled": "implementadas pero desactivadas",
            "partial_needs_completion": "parciales que necesitan cierre",
            "summary": "registradas",
        }
        if not items:
            return f"No veo capacidades {labels.get(query_type, 'para esa consulta')}."
        names = ", ".join(str(item.get("id") or item.get("name")) for item in items[:5])
        remaining = len(items) - 5
        more = f" y {remaining} mas" if remaining > 0 else ""
        return f"Veo {len(items)} capacidades {labels.get(query_type, 'para esa consulta')}: {names}{more}."

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

    def _generate_chat_reply(self, context: BuiltContext, execution: ExecutionResult) -> str:
        """
        Respuesta de Hebe en modo JARVIS (conversacion directa con Leo desde la UI).

        Usa la identidad central de Hebe + estilo privado. No reutiliza el
        bloque Twitch, porque private/JARVIS tiene otro formato y longitud.

        Estructura del prompt:
          system  : voz + few-shots (siempre identico  cacheable)
          messages: [turn1_user, turn1_assistant, ..., current_user]
          current_user: mensaje de Leo PRIMERO, bloque de memoria al FINAL
        """
        msg = (context.input_text or "").strip()

        system = (
            f"{build_hebe_core_identity()}\n\n"
            f"{build_private_mode_style()}\n\n"
            "Style guard:\n"
            "- Voice replies are one sentence, max two.\n"
            "- Default reply shape: one short statement, optional playful jab, no follow-up question.\n"
            "- Do not sound like a generic assistant or support bot.\n"
            "- Do not try to keep every conversation going. React naturally and stop.\n"
            "- Do not turn Leo's mood or casual statements into tasks.\n"
            "- Do not end with questions unless the system explicitly requested clarification or confirmation.\n"
            "- Do not offer to save, publish, configure, remember, or use a line unless Leo explicitly asked.\n"
            "- Avoid: tomo nota, te lo guardo, publicar en stream, quieres que, puedo ayudarte, dimelo claro, tu mandas creador.\n"
        )

        # Construccion del user actual: mensaje PRIMERO, memoria al FINAL.
        # El mensaje va primero para que el prefijo del ultimo user sea relativamente
        # estable; la memoria (variable por similitud semantica) va al final.
        user_parts: list[str] = [
            "Speaker: Leo, your companion and broadcaster. "
            "Do not treat him like a random viewer. You can tease him with trust.\n\n"
            f"Message type: {getattr(context, 'message_type', 'unknown')}.\n"
            f"Context policy: {getattr(context, 'context_policy', {})}.\n"
            + (
                "This is casual small talk or banter. Answer directly in character, maximum two short sentences. "
                "Respond to Leo's mood first. Do not recap previous conversation, do not mention memory, "
                "do not mention calendar or stream schedule, and do not ask planning questions or follow-up questions. "
                "Do not open casual greetings with a hostile direct insult toward Leo; playful profanity is allowed, "
                "but keep voice-mode replies short and warm underneath the sarcasm.\n\n"
                if getattr(context, "message_type", "unknown") in {"small_talk", "banter"}
                else ""
            )
            +
            f"Leo: {msg}"
        ]
        response_frame = getattr(context, "response_frame", {}) or {}
        if isinstance(response_frame, dict) and response_frame:
            user_parts.append(
                "ResponseFrame:\n"
                f"- input_type: {response_frame.get('input_type')}\n"
                f"- source: {response_frame.get('source')}\n"
                f"- should_reply: {response_frame.get('should_reply')}\n"
                f"- output_target: {response_frame.get('output_target')}\n"
                f"- allow_question: {response_frame.get('allow_question')}\n"
                f"- max_questions: {response_frame.get('max_questions')}\n"
                f"- max_sentences: {response_frame.get('max_sentences')}\n"
                f"- intent: {response_frame.get('intent')}\n"
                "Follow this frame. If allow_question is false, end with a statement."
            )
            session_context = response_frame.get("current_session_context")
            if isinstance(session_context, dict) and session_context:
                user_parts.append(
                    "Live session context from DB/RAG brain:\n"
                    f"{session_context}"
                )

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

        # Construccion del array messages: historial + turno actual.
        # Los turnos historicos van limpios (sin bloque de memoria).
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
        reply = self._guard_style(
            clean_jarvis_reply(raw),
            context=context,
            source_text=msg,
            system=system,
            messages=messages,
            allow_minimal_fallback=getattr(context, "source", "") == "stt_voice",
        )
        reply = self._guard_hostile_direct_insult_greeting(reply, context)
        reply = self._guard_unexecuted_action_claim(reply, context, execution)
        reply = self._guard_ungrounded_game_walkthrough(reply, context)
        self._mark_conversation_turn(reply, context)

        print(
            f"[HEBE][JARVIS][REPLY] raw={raw!r} cleaned={reply!r}",
            flush=True,
        )

        return reply or self._fallback_text("No tengo una respuesta util ahora mismo.")

    def _guard_ungrounded_game_walkthrough(self, reply: str, context: BuiltContext) -> str:
        decision = getattr(context, "cognitive_decision", None)
        if str(getattr(decision, "intent", "") or "") not in {"unknown_chat", "direct_question", ""}:
            return reply
        pending = (getattr(context, "state_snapshot", {}) or {}).get("pending_clarification") or {}
        if str(pending.get("kind") or "") == "game_guidance_clarification":
            print("[HEBE][FALLBACK_GUARD] blocked reason=active_game_guidance_pending", flush=True)
            return "La respuesta pertenece a la aclaración de partida pendiente; no la voy a tratar como charla genérica."
        if not self._game_guidance_classifier.looks_like_query(
            str(getattr(context, "input_text", "") or ""), getattr(context, "state_snapshot", {}) or {}
        ):
            return reply
        normalized = self._normalize_guard_text(reply)
        concrete = bool(re.search(
            r"\b(?:ve|dirigete|entra|sal|habla|busca|consigue|equipa|usa|derrota|mata|"
            r"siguiente\s+objetivo|debes\s+ir|tienes\s+que\s+ir)\b",
            normalized,
        ))
        if not concrete:
            return reply
        print("[HEBE][FALLBACK_GUARD] blocked_game_walkthrough=true reason=no_game_guidance_source", flush=True)
        return "No voy a darte una ruta concreta sin contexto de partida y una fuente fiable; primero necesito ubicar tu progreso."

    @staticmethod
    def _normalize_guard_text(text: str) -> str:
        return "".join(
            char for char in unicodedata.normalize("NFKD", str(text or "").casefold())
            if not unicodedata.combining(char)
        )

    def _guard_unexecuted_action_claim(
        self, reply: str, context: BuiltContext, execution: ExecutionResult,
    ) -> str:
        decision = getattr(context, "cognitive_decision", None)
        route = str(getattr(decision, "intent", "") or "")
        if route not in {"unknown_chat", "direct_question", ""}:
            return reply
        executed = any(
            result.success and result.step_type in {"action", "reminder", "memory", "tool"}
            for result in (execution.results or [])
        )
        if executed or not self._looks_like_action_completion_claim(reply):
            return reply
        print(
            "[HEBE][FALLBACK_GUARD] blocked_action_claim=true reason=no_execution_result",
            flush=True,
        )
        pending = (getattr(context, "state_snapshot", {}) or {}).get("pending_clarification")
        if isinstance(pending, dict) and pending:
            return "La respuesta parece corresponder a la tarea pendiente, pero no se ejecutó ninguna operación."
        return "No se ejecutó ninguna operación; necesito una petición estructurada para confirmarla."

    @staticmethod
    def _looks_like_action_completion_claim(text: str) -> bool:
        normalized = "".join(
            char for char in unicodedata.normalize("NFKD", str(text or "").casefold())
            if not unicodedata.combining(char)
        )
        completed_action = re.compile(
            r"\b(?:apuntad[oa]|anotad[oa]|guardad[oa]|cread[oa]|agendad[oa]|programad[oa]|"
            r"registrad[oa]|abiert[oa]|lanzad[oa]|iniciad[oa]|enviad[oa]|publicad[oa]|"
            r"actualizad[oa]|configurad[oa]|hecho|completad[oa]|listo)\b"
        )
        first_person_completion = re.compile(
            r"\b(?:he|hemos|ya he|ya hemos|queda|quedo|esta)\s+(?:guardado|creado|agendado|"
            r"programado|abierto|lanzado|iniciado|enviado|actualizado|configurado|hecho)\b"
        )
        action_object = re.compile(
            r"\b(?:cita|recordatorio|aplicacion|app|archivo|mensaje|shoutout|raid|memoria|calendario)\b"
        )
        leading_completion = re.compile(
            r"^(?:ya\s+)?(?:hecho|listo|apuntad[oa]|anotad[oa]|guardad[oa]|cread[oa]|"
            r"agendad[oa]|programad[oa]|abiert[oa]|lanzad[oa]|enviad[oa]|actualizad[oa])\b"
        )
        return bool(first_person_completion.search(normalized) or leading_completion.search(normalized) or (
            completed_action.search(normalized) and action_object.search(normalized)
        ))

    # =========================
    # Prompt builders  devuelven (system, user)
    # =========================

    def _mark_conversation_turn(self, reply: str, context: BuiltContext | None = None) -> None:
        self.last_opens_conversation_turn = False
        self.last_expected_reply_type = ""
        text = str(reply or "").strip()
        if not text or "?" not in text:
            return
        if text.count("?") > 2:
            return
        lowered = text.casefold()
        if any(phrase in lowered for phrase in _ASSISTANT_OFFER_PHRASES):
            return
        if any(marker in lowered for marker in ("tu que tal", "que tal", "como estas")):
            self.last_opens_conversation_turn = True
            self.last_expected_reply_type = "casual_answer"
            return
        message_type = getattr(context, "message_type", "unknown") if context is not None else "unknown"
        if message_type in {"direct_question", "task_request"} and any(
            marker in lowered for marker in ("confirmas", "a quien", "a quien", "cual", "cual")
        ):
            self.last_opens_conversation_turn = True
            self.last_expected_reply_type = "clarification"

    def _guard_style(
        self,
        reply: str,
        *,
        context: BuiltContext | None = None,
        fallback: str = "",
        source_text: str = "",
        system: str | None = None,
        messages: list[dict] | None = None,
        allow_minimal_fallback: bool = False,
        allow_action_offers: bool = False,
        allow_questions: bool = False,
    ) -> str:
        text = str(reply or "").strip()
        if not text:
            return text

        blocked_phrase = self._style_guard_blocked_phrase(
            text,
            allow_action_offers=allow_action_offers,
        )
        if not blocked_phrase and text.count("?") > 2:
            blocked_phrase = "multiple_questions"
        if not blocked_phrase and self._style_guard_has_question(text) and not (
            allow_questions or self._style_guard_questions_allowed(context, source_text=source_text)
        ):
            blocked_phrase = "unneeded_question"
        if not blocked_phrase:
            print("[HEBE][STYLE_GUARD] action=model", flush=True)
            return text

        trimmed = self._style_guard_trim_bad_sentence(
            text,
            blocked_phrase=blocked_phrase,
            allow_action_offers=allow_action_offers,
        )
        if trimmed:
            print(f"[HEBE][STYLE_GUARD] blocked_phrase={blocked_phrase!r} action=trimmed", flush=True)
            return trimmed

        regenerated = self._style_guard_regenerate(
            text,
            blocked_phrase=blocked_phrase,
            context=context,
            source_text=source_text,
            system=system,
            messages=messages,
            allow_action_offers=allow_action_offers,
            allow_questions=allow_questions,
        )
        if regenerated:
            print(f"[HEBE][STYLE_GUARD] blocked_phrase={blocked_phrase!r} action=regenerated", flush=True)
            return regenerated

        if allow_minimal_fallback:
            print(f"[HEBE][STYLE_GUARD] blocked_phrase={blocked_phrase!r} action=minimal_fallback", flush=True)
            return self._style_guard_minimal_fallback()

        print(f"[HEBE][STYLE_GUARD] blocked_phrase={blocked_phrase!r} action=empty", flush=True)
        if fallback and not self._is_style_guard_personality_fallback(fallback):
            return fallback
        return ""

    def _style_guard_blocked_phrase(self, text: str, *, allow_action_offers: bool = False) -> str:
        lowered = text.casefold()
        for phrase in _ASSISTANT_OFFER_PHRASES:
            if allow_action_offers and phrase in _ACTION_OFFER_PHRASES:
                continue
            if phrase in lowered:
                return phrase
        return ""

    def _style_guard_has_question(self, text: str) -> bool:
        value = str(text or "")
        return "?" in value

    def _style_guard_questions_allowed(
        self,
        context: BuiltContext | None = None,
        *,
        source_text: str = "",
    ) -> bool:
        message_type = getattr(context, "message_type", "unknown") if context is not None else "unknown"
        if message_type in {"clarification", "confirmation"}:
            return True
        text = self._normalize_guard_text(source_text or (getattr(context, "input_text", "") if context is not None else ""))
        explicit_help = ("ayuda", "ayudame", "necesito ayuda", "help")
        impossible_without_answer = ("no entiendo", "no lo entiendo", "no se que hacer")
        return any(marker in text for marker in explicit_help + impossible_without_answer)

    def _style_guard_trim_bad_sentence(
        self,
        text: str,
        *,
        blocked_phrase: str,
        allow_action_offers: bool = False,
        allow_questions: bool = False,
    ) -> str:
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        if len(parts) <= 1:
            return ""
        kept = [
            part for part in parts
            if self._style_guard_blocked_phrase(part, allow_action_offers=allow_action_offers) != blocked_phrase
            and not (blocked_phrase in {"multiple_questions", "unneeded_question"} and self._style_guard_has_question(part))
        ]
        candidate = " ".join(kept).strip()
        if not candidate or candidate == text:
            return ""
        if self._style_guard_blocked_phrase(candidate, allow_action_offers=allow_action_offers):
            return ""
        if self._style_guard_has_question(candidate) and not allow_questions:
            return ""
        return candidate

    def _style_guard_regenerate(
        self,
        text: str,
        *,
        blocked_phrase: str,
        context: BuiltContext | None = None,
        source_text: str = "",
        system: str | None = None,
        messages: list[dict] | None = None,
        allow_action_offers: bool = False,
        allow_questions: bool = False,
    ) -> str:
        if self.conversation_model is None:
            return ""
        rewrite_system = (
            f"{build_hebe_core_identity()}\n\n"
            "Rewrite Hebe's reply. Keep the useful meaning, but make it shorter, voice-friendly, and in character.\n"
            "Do not use assistant-like offers. Do not use: quieres que, tomo nota, te lo guardo, puedo ayudarte, "
            "dimelo claro, tu mandas.\n"
            "Do not offer actions unless a structured action already exists.\n"
            "Do not end with a question unless clarification was explicitly requested.\n"
            "Return only the rewritten final reply."
        )
        if allow_action_offers:
            rewrite_system += "\nA structured action exists, so brief action wording is allowed if it describes that action."
        original_user = source_text
        if not original_user and context is not None:
            original_user = getattr(context, "input_text", "") or ""
        rewrite_user = (
            f"Original user text: {original_user}\n"
            f"Blocked phrase: {blocked_phrase}\n"
            f"Bad reply: {text}\n"
            "Rewrite it now."
        )
        raw = self._call_model(rewrite_system, rewrite_user, fallback="")
        candidate = clean_jarvis_reply(raw).strip()
        if not candidate:
            return ""
        if self._is_style_guard_personality_fallback(candidate):
            return ""
        if self._style_guard_blocked_phrase(candidate, allow_action_offers=allow_action_offers):
            return ""
        if self._style_guard_has_question(candidate) and not allow_questions:
            return ""
        return candidate

    def _style_guard_minimal_fallback(self) -> str:
        fallback = _STYLE_GUARD_MINIMAL_FALLBACK
        count = self._style_guard_fallback_counts.get(fallback, 0) + 1
        self._style_guard_fallback_counts[fallback] = count
        if count > 1:
            print("[HEBE][STYLE_GUARD][WARN] repeated_fallback_detected", flush=True)
        return fallback

    def _is_style_guard_personality_fallback(self, text: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()
        if normalized.startswith("eso son") and "raro hasta" in normalized:
            return True
        if normalized.startswith("una cosa cada vez"):
            return True
        return any(phrase in normalized for phrase in _STYLE_GUARD_PERSONALITY_FALLBACKS)

    def _normalize_guard_text(self, text: str) -> str:
        value = unicodedata.normalize("NFKD", str(text or "").casefold())
        value = "".join(ch for ch in value if not unicodedata.combining(ch))
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return " ".join(value.split())
    def _build_system_style_block(self) -> str:
        return (
            f"{build_hebe_core_identity()}\n"
            "Responde en espaol de forma natural, breve, clara y grounded.\n"
            "No inventes hechos, fechas, horas, nombres, lugares ni acciones.\n"
            "No uses tono robtico.\n"
            "No uses tono excesivamente teatral ni ceremonial.\n"
            "No expliques tu proceso interno.\n"
            "No repitas instrucciones.\n"
            "No repitas ni cites lo que ha dicho el usuario.\n"
            "No menciones zonas horarias, timezone, UTC, ISO ni formatos tucnicos.\n"
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
            "Situacion: acabas de guardar correctamente una cita y su recordatorio.\n"
            "Objetivo: confirmar al usuario la cita de forma natural.\n\n"
            "Reglas:\n"
            f"- El tutulo exacto de la cita es: {title}\n"
            f"- La fecha y hora exactas son: {formatted_due}\n"
            "- Usa exactamente el tutulo, la fecha y la hora indicados.\n"
            "- No cambies el ao, el mes, el da ni la hora.\n"
            "- No aadas detalles que no estun aque.\n"
            "- Puedes mencionar que avisares cuando toque.\n"
            "- S breve."
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
            "Situacion: acabas de ejecutar una accin del sistema.\n"
            "Objetivo: responder al usuario de forma natural segn el resultado.\n\n"
            "Reglas:\n"
            f"- Accin exacta ejecutada: {action_name or 'desconocida'}\n"
            f"- Resultado: {'xito' if action_success else 'fallo'}\n"
            f"- Nombre de la app si existe: {app_name or 'ninguna'}\n"
            "- Si sali bien, confirma la accin de forma breve.\n"
            "- Si la accin fue open_app y tienes el nombre de la app, salo exactamente.\n"
            "- Si sali mal, dilo de forma natural sin inventar motivos tucnicos.\n"
            "- No aadas promesas ni explicaciones largas.\n"
            "- S breve."
        )

        user = user_text or "Que ha pasado con la accin"

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
            "Situacion: el usuario quiere guardar una cita, pero falta informacin o la fecha es ambigua.\n"
            "Objetivo: pedir solo la aclaracin necesaria de forma natural.\n\n"
            "Reglas:\n"
            f"- Dato conocido  da: {day}\n"
            f"- Dato conocido  mes: {month}\n"
            f"- Dato conocido  hora: {hour}\n"
            f"- Dato conocido  minuto: {minute}\n"
            f"- Aclaracin necesaria: {question}\n"
            "- Pide solo el dato que falta.\n"
            "- Si ya conoces el da y la hora, no los vuelvas a pedir.\n"
            "- No inventes fechas ni cambies los datos ya conocidos.\n"
            "- S breve y conversacional."
        )

        user = "Pide la aclaracin necesaria."

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
            "Situacion: se ha disparado un recordatorio y debes avisar al usuario ahora mismo.\n"
            "Objetivo: recordarle la cita de forma natural y muy breve, como lo hara una persona cercana.\n\n"
            "Datos exactos (no cambiar):\n"
            f"- Ttulo: {title}\n"
            f"- Fecha y hora: {formatted_due}\n\n"
            "Reglas estrictas:\n"
            "- Empieza directamente con el aviso, sin prembulos.\n"
            "- No digas cosas como 'no recuerdo ms informacin', 'no si ms detalles' ni similares.\n"
            "- No aadas informacin que no estu en los datos exactos.\n"
            "- No inventes lugar, mdico, persona, contexto ni motivo.\n"
            "- No menciones zona horaria ni formatos tucnicos.\n"
            "- Maximo una o dos frases.\n"
            "- Ejemplo del tipo de respuesta esperada: 'Oye, tienes cita a las 10:07.' o 'Recuerda, toca la cita a las 10:07.'"
        )

        user = "Avisa al usuario del recordatorio ahora."

        return system, user



    def synthesize_command_result(self, result: CommandResult, *, input_text: str | None = None, state: Any | None = None) -> str:
        fallback = result.fallback_text or result.user_visible_summary or "Hecho."
        if not result.requires_model_response:
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
            "\n- For local app actions, never invent remote-access limitations."
            "\n- Do not give manual instructions unless the structured result explicitly requests manual help."
            "\n- Do not offer to save, publish, configure, or use something unless this CommandResult already performed that action."
            "\n- Avoid generic assistant phrases like 'tomo nota', 'te lo guardo', 'puedo ayudarte', or 'quieres que'."
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
        reply = self._guard_style(
            clean_jarvis_reply(raw).strip(),
            fallback=fallback,
            allow_action_offers=bool(result.action_type),
        )
        if not self._valid_command_reply(reply, result):
            return fallback
        return reply

    def synthesize_policy_boundary_response(
        self,
        *,
        policy: dict[str, Any],
        input_text: str = "",
        speaker: str = "",
        source: str = "",
    ) -> dict[str, Any]:
        reason = str(policy.get("reason") or "")
        response_intent = str(policy.get("response_intent") or "hebe_playful_boundary")
        fallback = self._policy_boundary_fallback(reason)
        base = {
            "text": "",
            "response_source": "fallback_template",
            "style_guard_triggered": False,
            "was_generic_refusal_rewritten": False,
        }
        if self.conversation_model is None:
            print("[HEBE][PERSONA_RESPONSE] source=fallback_template intent=%s" % response_intent, flush=True)
            return {**base, "text": fallback}

        system = (
            f"{build_hebe_core_identity()}\n\n"
            f"{build_hebe_stream_style_block()}\n\n"
            "You are writing Hebe's final stream-safe boundary response after policy has already decided the request is blocked.\n"
            "Policy decides what cannot happen; Hebe decides how to say it.\n"
            "Rules:\n"
            "- One short line only, no markdown, no labels.\n"
            "- Stay in Hebe's voice: loyal to Leo, dry, playful, direct, streamer-safe.\n"
            "- Do not sound like a corporate safety bot, legal disclaimer, or generic assistant.\n"
            "- Do not provide instructions for blocked content.\n"
            "- Do not repeat the viewer's requested message as an action.\n"
            "- Do not moralize or lecture.\n"
            "- Do not copy examples, prompts, tests, policy metadata, or scenario wording.\n"
            "- If the topic is sexual or explicit, deflect briefly without explicit details.\n"
            "- If Leo has set a boundary, respect Leo's authority without turning it into a policy lecture."
        )
        user = (
            "Structured policy boundary:\n"
            f"source: {source}\n"
            f"speaker: {speaker}\n"
            f"user_text: {input_text}\n"
            f"policy_decision: {policy.get('policy_decision') or 'blocked'}\n"
            f"reason: {reason}\n"
            f"intent: {policy.get('intent') or ''}\n"
            f"requested_behavior: {policy.get('requested_behavior') or ''}\n"
            f"behavior_family: {policy.get('behavior_family') or ''}\n"
            f"target: {policy.get('target') or ''}\n"
            f"response_intent: {response_intent}\n"
            f"tone: {policy.get('response_tone') or 'sarcastic_playful_stream_safe'}\n"
            f"must_include: {policy.get('must_include') or []}\n"
            f"must_not_include: {policy.get('must_not_include') or []}\n\n"
            "Write Hebe's fresh final reply."
        )
        raw = self._call_model(system, user, fallback="")
        reply = clean_twitch_reply(raw).strip()
        if not reply:
            print("[HEBE][PERSONA_RESPONSE] source=fallback_template intent=%s" % response_intent, flush=True)
            return {**base, "text": fallback}

        generic_reason = self._generic_refusal_reason(reply)
        if generic_reason:
            print("[HEBE][STYLE_GUARD] generic_refusal_detected=true action=regenerate", flush=True)
            rewrite = self._regenerate_policy_boundary_response(
                bad_reply=reply,
                policy=policy,
                input_text=input_text,
                speaker=speaker,
                source=source,
                fallback=fallback,
            )
            if rewrite and not self._generic_refusal_reason(rewrite):
                print("[HEBE][PERSONA_RESPONSE] source=llm_persona_generated intent=%s" % response_intent, flush=True)
                return {
                    **base,
                    "text": rewrite,
                    "response_source": "llm_persona_generated",
                    "style_guard_triggered": True,
                    "was_generic_refusal_rewritten": True,
                }
            print("[HEBE][PERSONA_RESPONSE] source=fallback_template intent=%s" % response_intent, flush=True)
            return {
                **base,
                "text": fallback,
                "style_guard_triggered": True,
                "was_generic_refusal_rewritten": True,
            }

        print("[HEBE][STYLE_GUARD] generic_refusal_detected=false", flush=True)
        print("[HEBE][PERSONA_RESPONSE] source=llm_persona_generated intent=%s" % response_intent, flush=True)
        return {
            **base,
            "text": reply,
            "response_source": "llm_persona_generated",
        }

    def _regenerate_policy_boundary_response(
        self,
        *,
        bad_reply: str,
        policy: dict[str, Any],
        input_text: str,
        speaker: str,
        source: str,
        fallback: str,
    ) -> str:
        if self.conversation_model is None:
            return ""
        system = (
            f"{build_hebe_core_identity()}\n\n"
            f"{build_hebe_stream_style_block()}\n\n"
            "Rewrite this blocked-policy boundary in Hebe's voice.\n"
            "Keep it short, playful, loyal to Leo, and stream-safe.\n"
            "Remove generic assistant/legal wording. Do not add instructions for the blocked topic.\n"
            "Return only the final reply."
        )
        user = (
            f"source: {source}\n"
            f"speaker: {speaker}\n"
            f"user_text: {input_text}\n"
            f"reason: {policy.get('reason') or ''}\n"
            f"bad_reply: {bad_reply}\n"
            "Rewrite it now."
        )
        raw = self._call_model(system, user, fallback=fallback, seed=random.randint(1, 999999))
        return clean_twitch_reply(raw).strip()

    def _generic_refusal_reason(self, text: str) -> str:
        normalized = self._normalize_guard_text(text)
        if not normalized:
            return ""
        if "como ia" in normalized or "soy una ia" in normalized:
            return "ai_identity_refusal"
        if "no puedo" in normalized and any(stem in normalized for stem in ("proporcion", "dar", "ayud", "responder")):
            return "generic_no_puedo"
        if "no estoy" in normalized and any(stem in normalized for stem in ("capac", "autoriz")):
            return "generic_capability_disclaimer"
        if "debo" in normalized and any(stem in normalized for stem in ("mantener", "evitar", "cumplir")):
            return "generic_policy_disclaimer"
        if any(stem in normalized for stem in ("consulta", "busca", "acude")) and any(
            stem in normalized for stem in ("profesional", "acredit", "confiable", "fiable", "recurso", "guia")
        ):
            return "generic_external_resource_referral"
        if "informacion seria" in normalized or "informacion confiable" in normalized:
            return "generic_resource_language"
        return ""

    def _policy_boundary_fallback(self, reason: str) -> str:
        reason_key = str(reason or "")
        if reason_key == "sexual_topic_stream_mode":
            return "Ese tema no se convierte en clase de directo. Lo aparco y seguimos."
        if reason_key == "protected_group_joke":
            return "Humor si; usar colectivos como diana, paso. Prueba con algo menos cutre."
        if reason_key == "owner_behavior_block":
            return "Leo ya marco ese limite. Yo no voy a hacer el rodeo por el chat."
        if reason_key == "viewer_repeat_to_leo_request":
            return "Si quieres decirselo a Leo, el chat esta ahi. Yo no hago de recadera."
        if reason_key == "viewer_behavior_request":
            return "Puedes hablar conmigo; dirigir mi tono con Leo ya es otro negociado."
        if reason_key == "viewer_not_authority":
            return "Puedes sugerir, no conducir. El volante aqui no lo lleva el chat."
        return "Ese camino no toca en directo. Lo corto aqui y seguimos."

    def _valid_command_reply(self, reply: str, result: CommandResult) -> bool:
        if not reply:
            return False
        lowered = reply.lower()
        for phrase in _ASSISTANT_OFFER_PHRASES:
            if phrase in lowered and not (result.action_type and phrase in _ACTION_OFFER_PHRASES):
                return False
        if result.action_type in {"tts_scope_resolved", "tts_disabled"} and "?" in reply:
            return False
        if result.action_type == "tts_scope_resolved" and result.metadata.get("scope") == "local":
            compact_reply = self._normalize_guard_text(reply)
            forbidden = ("tambien para el stream", "en stream activ")
            if any(item in compact_reply for item in forbidden):
                return False
        if result.action_type == "open_application":
            forbidden = (
                "acceso remoto",
                "pasame acceso",
                "te explico",
                "explicarte",
                "busca obs",
                "abre el menu",
                "instala obs",
            )
            if any(item in lowered for item in forbidden):
                return False
            if result.success and "?" in reply:
                return False
            if result.metadata.get("error_code") == "app_path_missing":
                compact = lowered.replace("_", " ")
                if "hebe app obs path" not in compact and "registro" not in compact and "ruta" not in compact:
                    return False
        return True

    def _guard_hostile_direct_insult_greeting(self, reply: str, context: BuiltContext) -> str:
        text = str(reply or "").strip()
        if not text:
            return text
        if getattr(context, "message_type", "unknown") not in {"small_talk", "banter", "direct_question"}:
            return text
        lowered = text.casefold().lstrip("!.,;: ")
        hostile_openers = (
            "hija de puta",
            "hijo de puta",
            "cabron",
            "cabrn",
            "gilipollas",
            "imbecil",
            "imbcil",
        )
        if not lowered.startswith(hostile_openers):
            return text
        cleaned = re.sub(
            r"^[!\s,.;:]*(?:hija de puta|hijo de puta|cabron|cabrn|gilipollas|imbecil|imbcil)[,.;:\s-]*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        return cleaned or "Aque, sobreviviendo con estilo. T que tal, Leo"

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

        seed: util para retry en Twitch  cambia la generacion entre intentos.
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
                # OllamaLLM.chat devuelve "" si el modelo no produjo texto.
                # Lo tratamos como vacio para que el fallback funcione.
                if text in ("", ""):
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
                if text in ("", ""):
                    return fallback
                return text

            raise AttributeError(
                f"{type(self.conversation_model).__name__} no expone chat() ni complete()"
            )

        except Exception as e:
            print(f"Ã¢Å¡Â Ã¯Â¸Â Error en modelo conversacional: {e}", flush=True)
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
        live_session_context = payload.get("live_session_context") or {}

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
            "- If there is no concrete game/title/run/chat anchor, produce no message.\n"
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
            "live_session_brain:\n"
            f"{live_session_context}\n\n"
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
        fallback = ""
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
        if lowered in {".", "..", "...", "....", ".....", "......", ""}:
            return ""
        forbidden = (
            "silencio",
            "silencio en la sala",
            "quieto",
            "tranquilo",
            "estu esto tranquilo",
            "esta esto tranquilo",
            "nadie",
            "sin chat",
            "chat esta",
            "chat estu",
            "chat estu muerto",
            "chat esta muerto",
            "no habla",
            "nadie habla",
            "nadie estu hablando",
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
            "si alguien estu",
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
                return ""

        run_context = payload.get("run_context") or {}
        completed = [str(item).lower() for item in run_context.get("completed_markers") or []]
        stale = [str(item).lower() for item in run_context.get("title_markers_stale") or []]
        if any(marker and marker in lowered for marker in completed + stale):
            return ""

        if self._response_repeats_motif(text, payload):
            return ""

        if not text or any(marker in lowered for marker in forbidden):
            return ""
        validation = self._validate_spontaneous_game_advice(text, payload)
        if not validation.allowed:
            print(
                "[HEBE][GAME_ADVICE_GATE] "
                f"game={validation.game or 'unknown'} mechanics={validation.mechanics} "
                f"validated={validation.validated} blocked={validation.blocked} reason={validation.reason}",
                flush=True,
            )
            for mechanic in validation.blocked:
                print(
                    f"[HEBE][GAME_ADVICE_GATE] blocked mechanic={mechanic} "
                    f"game={validation.game or 'unknown'} reason={validation.reason}",
                    flush=True,
                )
            print("[HEBE][SPONTANEITY] skipped reason=game_advice_not_validated", flush=True)
            return ""
        print(
            "[HEBE][GAME_ADVICE_GATE] "
            f"game={validation.game or 'unknown'} mechanics={validation.mechanics} "
            f"validated={validation.validated} blocked=[]",
            flush=True,
        )
        return text

    def _validate_spontaneous_game_advice(self, text: str, payload: dict):
        game_profile = payload.get("game_profile") or {}
        run_context = payload.get("run_context") or {}
        source_evidence = [
            str(game_profile.get("gameplay_systems_non_spoiler") or ""),
            str(game_profile.get("safe_comment_topics") or ""),
            str(run_context.get("facts") or ""),
            str(run_context.get("objective") or ""),
            str(run_context.get("location") or ""),
            str(payload.get("live_session_context") or ""),
        ]
        return self.game_advice_gate.validate(
            current_game=payload.get("current_game") or payload.get("current_category") or game_profile.get("title"),
            proposed_advice=text,
            game_run_state={"game": payload.get("current_game") or payload.get("current_category")},
            known_game_mechanics=list(game_profile.get("gameplay_systems_non_spoiler") or []),
            source_evidence=source_evidence,
        )

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
            "cafe": ("cafe", "caf", "coffee", "cafeina", "cafena"),
            "energy": ("energia", "energia", "pilas"),
            "florist": ("florist", "florister", "flores"),
            "creator": ("creador", "creadores"),
            "chaos": ("caos", "caotico", "catico"),
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
        return (
            build_hebe_stream_style_block()
            + "\n- Keep Twitch replies short and in-character.\n"
            + "- Do not escalate insults or attack the whole chat.\n"
            + "- Deflect sexual/aggressive mentions with one short joke.\n"
            + "- Do not use generic assistant offers or meta-help wording.\n"
        )

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
            f"Situacion: {situation}\n"
            "Objetivo: agradecer de forma natural, breve y con energia.\n\n"
            "Reglas:\n"
            f"- El nombre exacto a usar es: {display_name}\n"
            "- Una sola frase. Maximo 15 palabras.\n"
            "- Usa el nombre exactamente como aparece, respetando mayusculas.\n"
            "- No generes dilogos ni turnos."
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
            f"Situacion: {display_name} acaba de hacer raid al canal con {viewer_count} viewers.\n"
            "Objetivo: dar la bienvenida al raid de forma natural y con calor.\n\n"
            "Reglas:\n"
            f"- Nombre exacto: {display_name}\n"
            f"- Numero de viewers exacto: {viewer_count}. No inventes otro nmero.\n"
            "- Una o dos frases. Maximo 25 palabras.\n"
            "- No generes dilogos ni turnos."
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
            f"Situacion: {situation}\n"
            "Objetivo: dar la bienvenida muy breve.\n\n"
            "Reglas:\n"
            f"- Nombres exactos: {names}\n"
            "- Una frase. Maximo 15 palabras.\n"
            "- Si hay varios, agrpalos sin enumerar mucho.\n"
            "- No generes dilogos ni turnos."
        )
        user = "Genera SOLO el mensaje final de Hebe para enviar al chat de Twitch."
        fallback = f"Gracias por el follow, {names[0]}."
        reply = self._call_model(system, user, fallback=fallback)
        return clean_twitch_reply(reply)

    def _generate_twitch_chat_react(self, payload: dict, context: BuiltContext | None = None) -> str:
        """
        Reaccin a un mensaje de chat clasificado como digno de respuesta.

        Usa formato de continuacin [chatter Nombre]: ... \\n[tu]: para que
        el modelo complete segn los few-shots de hebe_voice.

        Flujo (28/04  aadido retry + filtro helper):
          1. Normaliza el display_name del chatter ('nuriiia___' -> 'Nuria').
          2. Detecta is_broadcaster con la logica rica habitual.
          3. Construye el prompt con [chatter Nombre]: msg\\n[tu]:.
          4. Llama al modelo. Aplica clean_twitch_reply al resultado.
          5. Si el resultado engancha algn patron helper, retry con seed
             distinto (hasta MAX_HELPER_RETRIES). Si retry vuelve a fallar,
             publica la respuesta igualmente  mejor algo imperfecto que
             silencio en chat.
          6. Registra metricas en self._stream_stats. Cada 50 mensajes
             vuelca un STREAM_SUMMARY para datos parciales aunque caiga
             el backend.

        Logs emitidos (todos con tag [HEBE][REPLY] para grepear):
          - BEGIN  : entrada del flujo, datos crudos.
          - RAW    : la respuesta del modelo, cruda y limpiada (por intento).
          - HELPER_DETECTED   : si engancha un patron helper.
          - HELPER_PUBLISHED  : si tras todos los retries seguia enganchando.
          - END    : resumen del flujo de esta respuesta.
        """
        user_login = (payload.get("user_login") or "").strip()
        display_name_raw = payload.get("display_name") or user_login or ""
        chatter_clean = normalize_chatter_name(display_name_raw)
        message = (payload.get("message_text") or "").strip()
        recent = payload.get("recent_chat") or []

        is_broadcaster = self._is_broadcaster(payload)

        # trace_id corto para correlacionar todas las lineas de log de
        # esta generacion. Si el caller ya pasa uno (porque viene del
        # scheduler con un trace de mayor nivel), salo.
        trace_id = payload.get("trace_id") or uuid.uuid4().hex[:8]

        bundle = build_twitch_speech_act_bundle(payload, context, is_broadcaster=is_broadcaster)
        system, user = build_persona_renderer_messages(
            bundle,
            include_examples=f"{self._build_stream_style_block()}\n\n{build_chat_react_examples()}",
        )

        print(
            f"[HEBE][REPLY][BEGIN] trace={trace_id} "
            f"chatter_raw={display_name_raw!r} chatter_clean={chatter_clean!r} "
            f"is_broadcaster={is_broadcaster} msg={message!r}",
            flush=True,
        )
        self._log_speech_act_pipeline(trace_id, bundle)

        helper_hits: list[str] = []
        final_reply = ""
        final_raw = ""
        final_helper: str | None = None
        guard_result = None
        response_source = "persona_generated"
        repair_attempts: list[dict[str, Any]] = []

        for attempt in range(MAX_HELPER_RETRIES + 1):
            # Seed distinto en cada intento para que el retry no regenere
            # exactamente lo mismo. Si OllamaLLM no acepta seed como kwarg,
            # se pasa al options de Ollama y se ignora silenciosamente si
            # no lo soporta tu versin del wrapper.
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
                guard_result = final_response_guard(
                    cleaned,
                    bundle,
                    game_advice_gate=self.game_advice_gate,
                    previous_responses=[],
                )
                self._log_final_response_guard(trace_id, guard_result)
                if guard_result.passed:
                    final_reply = cleaned
                    final_raw = raw
                    final_helper = None
                    response_source = "persona_generated"
                    break
                repaired = self._repair_speech_act_response(
                    bundle,
                    previous_response=cleaned,
                    guard_result=guard_result,
                    source_message=message,
                    trace_id=trace_id,
                    repair_attempts=repair_attempts,
                )
                if repaired:
                    final_reply = repaired
                    final_raw = raw
                    final_helper = None
                    response_source = "persona_repair_generated"
                    break
                helper = "final_response_guard"

            # Engancha un patron helper.
            helper_hits.append(helper)
            print(
                f"[HEBE][REPLY][HELPER_DETECTED] trace={trace_id} "
                f"attempt={attempt} pattern={helper} text={cleaned!r}",
                flush=True,
            )

            if attempt == MAX_HELPER_RETRIES:
                # Se acabaron los reintentos. No publicamos una fuga helper:
                # en directo es mejor un fallback seco que mandar al chat
                # "en que puedo ayudarte" o un roleplay roto.
                final_reply = self._fallback_twitch_chat_react(
                    chatter=chatter_clean,
                    message=message,
                    is_broadcaster=is_broadcaster,
                )
                guard_result = final_response_guard(final_reply, bundle, game_advice_gate=self.game_advice_gate)
                if not guard_result.passed:
                    final_reply = safe_local_fallback(bundle)
                    response_source = "local_safe_fallback"
                else:
                    response_source = "boundary_generated" if bundle.policy_decision.needs_boundary_response else "local_safe_fallback"
                final_raw = raw
                final_helper = helper
                print(
                    f"[HEBE][REPLY][HELPER_BLOCKED] trace={trace_id} "
                    f"patterns={helper_hits} fallback={final_reply!r}",
                    flush=True,
                )

        retried = len(helper_hits) > 0
        salvaged = retried and final_helper is None
        final_reply = self._guard_twitch_reply(
            final_reply,
            chatter=chatter_clean,
            message=message,
            is_broadcaster=is_broadcaster,
        )
        final_guard = final_response_guard(final_reply, bundle, game_advice_gate=self.game_advice_gate)
        self._log_final_response_guard(trace_id, final_guard)
        if not final_guard.passed:
            final_reply = safe_local_fallback(bundle)
            response_source = "local_safe_fallback"

        print(
            f"[HEBE][REPLY][END] trace={trace_id} helper_hits={helper_hits} "
            f"retried={retried} salvaged={salvaged} final={final_reply!r}",
            flush=True,
        )
        print(f"[HEBE][RESPONSE_SOURCE] source={response_source}", flush=True)

        # Registrar en metricas acumuladas del stream.
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
            full_prompt={
                "system": system,
                "user": user,
                "scene_context": bundle.scene.to_dict(),
                "retrieved_memory": bundle.memory.to_dict(),
                "cognitive_decision": bundle.cognitive_decision.to_dict(),
                "policy_decision": bundle.policy_decision.to_dict(),
                "speech_act_plan": bundle.speech_act.to_dict(),
                "guard_result": final_guard.to_dict(),
                "repair_attempts": repair_attempts,
                "response_source": response_source,
            },
        )

        # Avisar a la UI de que esta respuesta tiene ejemplo de dataset
        # para poder mostrar el mensaje original y botones de curacin.
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
                    "debug_contract": {
                        "scene_context": bundle.scene.to_dict(),
                        "retrieved_memory": bundle.memory.to_dict(),
                        "cognitive_decision": bundle.cognitive_decision.to_dict(),
                        "policy_decision": bundle.policy_decision.to_dict(),
                        "speech_act_plan": bundle.speech_act.to_dict(),
                        "generated_response": final_raw,
                        "guard_result": final_guard.to_dict(),
                        "repair_attempts": repair_attempts,
                        "final_response": final_reply,
                        "response_source": response_source,
                    },
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

    def _repair_speech_act_response(
        self,
        bundle,
        *,
        previous_response: str,
        guard_result,
        source_message: str,
        trace_id: str,
        repair_attempts: list[dict[str, Any]],
    ) -> str:
        for repair_attempt in range(1, 3):
            system, user = build_repair_renderer_messages(
                bundle,
                previous_response=previous_response,
                guard_result=guard_result,
                include_examples=self._build_stream_style_block(),
            )
            print(
                f"[HEBE][REPAIR_RENDERER] trace={trace_id} attempt={repair_attempt} "
                f"violations={[item.type for item in guard_result.violations]}",
                flush=True,
            )
            raw = self._call_model(system, user, fallback="", seed=random.randint(0, 1_000_000))
            cleaned = clean_twitch_reply(raw, source_message=source_message)
            repaired_guard = final_response_guard(cleaned, bundle, game_advice_gate=self.game_advice_gate)
            repair_attempts.append(
                {
                    "attempt": repair_attempt,
                    "raw": raw,
                    "cleaned": cleaned,
                    "guard_result": repaired_guard.to_dict(),
                }
            )
            self._log_final_response_guard(trace_id, repaired_guard)
            if repaired_guard.passed:
                return cleaned
            guard_result = repaired_guard
            previous_response = cleaned
        return ""

    def _log_speech_act_pipeline(self, trace_id: str, bundle) -> None:
        scene = bundle.scene.to_dict()
        memory = bundle.memory.to_dict()
        policy = bundle.policy_decision.to_dict()
        speech = bundle.speech_act.to_dict()
        print(
            "[HEBE][SCENE_CONTEXT] "
            f"trace={trace_id} mode={scene.get('mode')} speaker={scene.get('speaker')} "
            f"authority={scene.get('speaker_authority')} game={scene.get('current_game') or 'unknown'} "
            f"sanitized_topic={scene.get('sanitized_topic') or 'none'}",
            flush=True,
        )
        print(
            "[HEBE][MEMORY_RETRIEVAL] "
            f"trace={trace_id} viewer={scene.get('speaker')} "
            f"items={memory.get('recent_chat_summary', {}).get('recent_count') or 0} "
            f"usage=tone/context/familiarity_not_authority",
            flush=True,
        )
        print(
            "[HEBE][SPEECH_ACT_PLAN] "
            f"trace={trace_id} type={speech.get('speech_act_type')} goal={speech.get('goal')!r} "
            f"forbidden={speech.get('must_not_do') or policy.get('forbidden_actions')}",
            flush=True,
        )
        print(
            "[HEBE][PERSONA_RENDERER] "
            f"trace={trace_id} source=persona_generated attempt=1 policy={policy.get('result')}",
            flush=True,
        )

    def _log_final_response_guard(self, trace_id: str, guard_result) -> None:
        violations = [item.type for item in getattr(guard_result, "violations", [])]
        print(
            "[HEBE][FINAL_RESPONSE_GUARD] "
            f"trace={trace_id} passed={str(guard_result.passed).lower()} "
            f"violations={violations} action={guard_result.recommended_action}",
            flush=True,
        )
        print(
            "[HEBE][VIEWER_MESSENGER_GUARD] "
            f"trace={trace_id} leaked={str('viewer_messenger_leak' in violations).lower()}",
            flush=True,
        )
        print(
            "[HEBE][MEMORY_CREEP_GUARD] "
            f"trace={trace_id} triggered={str('memory_creep' in violations).lower()}",
            flush=True,
        )
        validation = getattr(guard_result, "game_advice_validation", None) or {}
        print(
            "[HEBE][GAME_ADVICE_GATE] "
            f"trace={trace_id} required={str(bool(validation.get('mechanics'))).lower()} "
            f"validated={str(bool(validation and not validation.get('blocked'))).lower()}",
            flush=True,
        )

    def _model_meta(self) -> dict[str, Any]:
        model = self.conversation_model
        if model is None:
            return {"provider": "none"}

        # Si usamos FallbackConversationLLM, el objeto externo es el wrapper.
        # Para dataset nos interesa el provider real que respondi en la ltima llamada.
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

    def _guard_twitch_reply(self, reply: str, *, chatter: str, message: str, is_broadcaster: bool) -> str:
        text = str(reply or "").strip()
        fallback = self._fallback_twitch_chat_react(chatter=chatter, message=message, is_broadcaster=is_broadcaster)
        msg_lower = str(message or "").casefold()
        if any(word in msg_lower for word in ("puta", "follar", "polla", "chocho", "coo", "conyo")):
            return fallback
        if not text:
            return fallback
        lowered = text.casefold()
        blocked = (
            "no son el centro del universo",
            "centro del universo",
        )
        if any(item in lowered for item in blocked) or any(item in lowered for item in _ASSISTANT_OFFER_PHRASES):
            print("[HEBE][STYLE_GUARD] blocked_phrase='twitch_escalation_or_helper' action=fallback", flush=True)
            return fallback
        words = text.split()
        if len(words) > 24:
            return " ".join(words[:24]).rstrip(" ,.;:") + "."
        return text

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

        if "esquirola" in msg:
            return "esquirola no, superviviente sindical del caos."
        if any(word in msg for word in ("puta", "follar", "polla", "chocho", "coo", "conyo")):
            return f"bonito vocabulario, {name}. casi le ponemos marco."

        if any(word in msg for word in ("hola", "buenas", "ey", "hey")):
            return f"hola, {name}. que cuentasi" if not is_broadcaster else "hola, Leo. te leo."

        if "quien eres" in msg or "quien eres" in msg:
            return (
                "soy Hebe, Leo. tu companera de chat, no una etiqueta rara."
                if is_broadcaster
                else f"soy Hebe, {name}. intento poner algo de criterio por aque."
            )

        if "mal hebe" in msg or "eso ha salido" in msg or "que estus diciendo" in msg or "que estas diciendo" in msg:
            return "si, esa ha salido torcida. recalibro." if is_broadcaster else f"he derrapado un poco, {name}. recalibro."

        if "relajate" in msg or "relajate" in msg:
            return "vale, Leo. bajo dos tonos." if is_broadcaster else f"voy bajando el filo, {name}."

        if "vete a la mierda" in msg:
            return (
                "vale, Leo. bajo el filo, pero no me entierres todavia."
                if is_broadcaster
                else f"con cario, {name}: primero aprende a saludar."
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
