from __future__ import annotations

import re
import time
import unicodedata
import uuid
from dataclasses import replace
from typing import Any

from app.cognitive.cognitive_decision import CognitiveDecision
from app.cognitive.game_guidance import CAP_GAME_GUIDANCE, GameGuidanceCapability
from app.core.persistent_logs import log_jsonl_event
from app.continuity.models import CurrentConversation, ConversationStatus


CAP_TIME = "time.get_current_time"
CAP_DATE = "time.get_current_date"
CAP_CHAT = "hebe.chat_reply"
CAP_APPOINTMENT = "appointment.create"
CAP_REMINDER = "reminder.create"
CAP_SCHEDULER = "scheduler.create"
CAP_OPEN_APP = "pc.open_application"
CAP_TWITCH_ACTION = "twitch_action"
CAP_TWITCH_REPLY = "twitch.reply"
CAP_TWITCH_PROMOTION = "twitch.promotion"
CAP_WAKE_CONTROL = "hebe.wake_control"
CAP_TTS_CONTROL = "audio.tts_control"
CAP_STREAM_STATE = "stream.local_state_control"
CAP_PENDING_CANCEL = "pending.cancel"
CAP_FALLBACK_CHAT = CAP_CHAT

CREATE_CAPABILITIES = [CAP_APPOINTMENT, CAP_REMINDER, CAP_SCHEDULER]


class CognitiveRouter:
    """Owns intent priority and grants downstream subsystems permission to run."""

    def __init__(self, game_guidance: GameGuidanceCapability | None = None):
        self.game_guidance = game_guidance or GameGuidanceCapability()

    def route(self, context: Any) -> CognitiveDecision:
        raw = str(getattr(context, "input_text", "") or "")
        normalized = self.normalize(raw)
        source = str(getattr(context, "source", "") or self._source(context))
        authority = str(getattr(context, "authority", "") or self._authority(source))
        addressed = bool(getattr(context, "addressed_to_hebe", authority == "owner"))
        message_id = str(getattr(context, "message_id", "") or f"msg_{uuid.uuid4().hex}")
        pending = (getattr(context, "state_snapshot", {}) or {}).get("current_conversation")
        firewall_decision = str(getattr(context, "firewall_decision", "") or "")
        event_type = str(getattr(getattr(context, "internal_event", None), "event_type", "") or "")
        stream_is_live = bool(getattr(context, "stream_is_live", False))
        route_hints = list(getattr(context, "route_hints", []) or [])

        decision = self._classify(
            message_id=message_id,
            source=source,
            authority=authority,
            addressed=addressed,
            raw=raw,
            normalized=normalized,
            firewall_decision=firewall_decision,
            event_type=event_type,
            stream_is_live=stream_is_live,
            route_hints=route_hints,
            state_snapshot=getattr(context, "state_snapshot", {}) or {},
        )
        decision = self._apply_pending_contract(decision, pending)
        decision.debug_trace.extend([
            f"intent:{decision.intent}",
            f"priority_reason:{decision.reason}",
            f"pending:{decision.pending_reason}",
        ])
        self._log(decision, pending)
        print(
            f"[HEBE][COGNITIVE_TRACE] message_id={decision.message_id} route={decision.intent}",
            flush=True,
        )
        return decision

    @staticmethod
    def normalize(text: str) -> str:
        lowered = str(text or "").strip().lower()
        unaccented = "".join(
            char for char in unicodedata.normalize("NFKD", lowered)
            if not unicodedata.combining(char)
        )
        return " ".join(re.sub(r"[^a-z0-9\s:/_-]", " ", unaccented).split())

    def _classify(
        self, *, message_id: str, source: str, authority: str,
        addressed: bool, raw: str, normalized: str, firewall_decision: str,
        event_type: str, stream_is_live: bool, route_hints: list[str], state_snapshot: dict[str, Any],
    ) -> CognitiveDecision:
        base = dict(
            message_id=message_id, source=source, authority=authority,
            addressed_to_hebe=addressed, raw_text=raw, normalized_text=normalized,
            intent="unknown_chat", intent_confidence=0.45, is_new_request=True,
            uses_pending_task=False, goal_type="answer_question",
            allowed_capabilities=[CAP_CHAT], blocked_capabilities=[],
            allowed_step_types=["reply"], should_reply=True, response_mode="chat",
            response_intent="answer_user", reason="fallback_chat",
            action_permission_summary={"reply": True, "external_actions": False, "stream_live": bool(stream_is_live)},
        )

        if authority == "bot" or firewall_decision in {"ignore", "block_reply", "block_action"}:
            reason = "bot_input_blocked" if authority == "bot" else f"input_firewall:{firewall_decision}"
            return CognitiveDecision(**{**base, "intent": "ambient_noise", "intent_confidence": 1.0,
                "is_new_request": False, "allowed_capabilities": [],
                "blocked_capabilities": CREATE_CAPABILITIES + [CAP_OPEN_APP, CAP_TWITCH_REPLY, CAP_TWITCH_ACTION, CAP_TWITCH_PROMOTION],
                "allowed_step_types": ["noop"], "blocked_step_types": ["action", "memory", "reminder", "reply", "tool"],
                "should_reply": False, "should_stop_pipeline": True, "response_mode": "silent", "reason": reason,
                "action_permission_summary": {"reply": False, "external_actions": False}})

        if event_type.startswith("twitch_"):
            if not stream_is_live:
                return CognitiveDecision(**{**base, "intent": "stream_context_update", "intent_confidence": .99,
                    "is_new_request": False, "allowed_capabilities": [],
                    "blocked_capabilities": [CAP_TWITCH_REPLY, CAP_TWITCH_ACTION, CAP_TWITCH_PROMOTION],
                    "allowed_step_types": ["noop"], "blocked_step_types": ["reply", "action", "tool"],
                    "should_reply": False, "should_stop_pipeline": True, "response_mode": "silent",
                    "reason": "offline_stream", "action_permission_summary": {"stream_live": False, "reply": False}})
            if authority == "viewer" and not addressed:
                return CognitiveDecision(**{**base, "intent": "stream_context_update", "intent_confidence": .95,
                    "is_new_request": False, "allowed_capabilities": [],
                    "blocked_capabilities": [CAP_TWITCH_REPLY, CAP_TWITCH_ACTION, CAP_TWITCH_PROMOTION, CAP_OPEN_APP],
                    "allowed_step_types": ["state_update"], "should_reply": False, "should_stop_pipeline": True,
                    "response_mode": "silent", "reason": "viewer_context_only"})
            allowed = [CAP_TWITCH_REPLY]
            if authority == "system" and event_type == "twitch_raid":
                allowed.append(CAP_TWITCH_PROMOTION)
            return CognitiveDecision(**{**base, "intent": "twitch_internal_event" if authority == "system" else "twitch_viewer_message",
                "intent_confidence": .96, "is_new_request": True, "allowed_capabilities": allowed,
                "blocked_capabilities": CREATE_CAPABILITIES + [CAP_OPEN_APP, CAP_TWITCH_ACTION],
                "allowed_step_types": ["reply"], "response_mode": event_type,
                "response_intent": "stream_event_reply", "reason": "authorized_live_twitch_event",
                "action_permission_summary": {"stream_live": True, "reply": True, "promotion": CAP_TWITCH_PROMOTION in allowed}})

        if event_type == "reminder_due":
            return CognitiveDecision(**{**base, "intent": "reminder_due", "intent_confidence": 1.0,
                "is_new_request": False, "allowed_capabilities": ["reminder.notify"],
                "blocked_capabilities": CREATE_CAPABILITIES + [CAP_OPEN_APP, CAP_CHAT],
                "allowed_step_types": ["reply"], "response_mode": "reminder",
                "reason": "authorized_system_reminder", "action_permission_summary": {"reply": True}})

        if not normalized:
            return CognitiveDecision(**{**base, "intent": "ambient_noise", "intent_confidence": 1.0,
                "is_new_request": False, "allowed_capabilities": [], "should_reply": False,
                "should_stop_pipeline": True, "response_mode": "silent", "reason": "empty_input"})

        if authority == "ambient" and not addressed:
            return CognitiveDecision(**{**base, "intent": "stream_context_update", "intent_confidence": .9,
                "is_new_request": False, "goal_type": "update_session_state",
                "allowed_capabilities": [], "blocked_capabilities": CREATE_CAPABILITIES + [CAP_TWITCH_ACTION],
                "allowed_step_types": ["state_update"], "should_reply": False,
                "should_stop_pipeline": True, "response_mode": "silent", "reason": "ambient_context_only"})

        if self._is_pending_cancel(normalized):
            return CognitiveDecision(**{**base, "intent": "cancel_pending", "intent_confidence": .95,
                "goal_type": "cancel_pending_task", "allowed_capabilities": [CAP_PENDING_CANCEL],
                "blocked_capabilities": CREATE_CAPABILITIES + [CAP_OPEN_APP, CAP_CHAT],
                "allowed_step_types": ["state_update", "reply"], "response_mode": "command_result",
                "reason": "explicit_pending_cancellation"})

        wake_intent = self._wake_control_intent(normalized)
        if wake_intent:
            return CognitiveDecision(**{**base, "intent": wake_intent, "intent_confidence": .94,
                "goal_type": "control_wake_state", "allowed_capabilities": [CAP_WAKE_CONTROL],
                "blocked_capabilities": CREATE_CAPABILITIES + [CAP_OPEN_APP, CAP_CHAT],
                "allowed_step_types": ["state_update", "reply"], "response_mode": "command_result",
                "reason": "explicit_wake_state_command"})

        if authority == "owner" and self._is_promotion_command(normalized):
            return CognitiveDecision(**{**base, "intent": "promotion_shoutout", "intent_confidence": .96,
                "goal_type": "stream_promotion", "allowed_capabilities": [CAP_TWITCH_PROMOTION, CAP_TWITCH_ACTION],
                "blocked_capabilities": CREATE_CAPABILITIES + [CAP_OPEN_APP],
                "allowed_step_types": ["action", "reply"], "response_mode": "command_result",
                "response_intent": "confirm_stream_promotion", "reason": "deterministic_promotion_command",
                "action_permission_summary": {"stream_live": stream_is_live, "promotion": True}})

        hinted = self._hinted_owner_route(authority, addressed, route_hints, stream_is_live)
        if hinted:
            return CognitiveDecision(**{**base, **hinted})

        if self._is_current_time_query(normalized):
            return CognitiveDecision(**{**base, "intent": "current_time_query", "intent_confidence": .97,
                "goal_type": "answer_current_time", "allowed_capabilities": [CAP_TIME],
                "blocked_capabilities": CREATE_CAPABILITIES + [CAP_FALLBACK_CHAT], "response_mode": "time_answer",
                "response_intent": "report_current_time", "reason": "explicit_time_question"})

        if self._is_current_date_query(normalized):
            return CognitiveDecision(**{**base, "intent": "current_date_query", "intent_confidence": .96,
                "goal_type": "answer_current_date", "allowed_capabilities": [CAP_DATE],
                "blocked_capabilities": CREATE_CAPABILITIES + [CAP_FALLBACK_CHAT], "response_mode": "date_answer",
                "response_intent": "report_current_date", "reason": "explicit_date_question"})

        state = self._personal_state(normalized)
        if authority == "owner" and addressed and state:
            return CognitiveDecision(**{**base, "intent": "owner_personal_state", "intent_confidence": .94,
                "goal_type": "respond_to_personal_state", "allowed_capabilities": [CAP_CHAT],
                "blocked_capabilities": CREATE_CAPABILITIES + [CAP_TWITCH_ACTION],
                "response_mode": "companion_reaction", "response_intent": "react_to_personal_state",
                "reason": f"owner_state_signal:{state}", "personal_state": state})

        if self.game_guidance.looks_like_query(raw, state_snapshot):
            return CognitiveDecision(**{**base, "intent": "game_guidance_query", "intent_confidence": .91,
                "goal_type": "research_game_strategy", "allowed_capabilities": [CAP_GAME_GUIDANCE],
                "blocked_capabilities": CREATE_CAPABILITIES + [CAP_FALLBACK_CHAT],
                "allowed_step_types": ["reply"], "response_mode": "game_guidance",
                "response_intent": "provide_grounded_game_guidance", "reason": "structured_game_guidance_request"})

        if self._is_open_app_command(normalized):
            return CognitiveDecision(**{**base, "intent": "command_open_app", "intent_confidence": .95,
                "goal_type": "control_pc", "allowed_capabilities": [CAP_OPEN_APP],
                "blocked_capabilities": CREATE_CAPABILITIES, "allowed_step_types": ["action", "reply"],
                "response_mode": "confirm_action", "response_intent": "confirm_pc_action",
                "reason": "explicit_open_application_command"})

        if self._is_reminder_request(normalized):
            return CognitiveDecision(**{**base, "intent": "reminder_create_request", "intent_confidence": .92,
                "goal_type": "create_reminder", "allowed_capabilities": [CAP_REMINDER],
                "blocked_capabilities": [CAP_APPOINTMENT], "allowed_step_types": ["reminder", "reply"],
                "response_mode": "reminder", "reason": "explicit_reminder_marker"})

        if self._is_appointment_request(normalized):
            return CognitiveDecision(**{**base, "intent": "appointment_create_request", "intent_confidence": .93,
                "goal_type": "create_appointment", "allowed_capabilities": [CAP_APPOINTMENT],
                "blocked_capabilities": [], "allowed_step_types": ["memory", "reminder", "reply"],
                "response_mode": "appointment", "reason": "strong_appointment_marker"})

        if self._is_catalogue_query(normalized):
            return CognitiveDecision(**{**base, "intent": "capability_catalogue_query", "intent_confidence": .88,
                "goal_type": "analyze_data", "allowed_capabilities": ["hebe.capability_backlog_query"],
                "blocked_capabilities": CREATE_CAPABILITIES, "response_mode": "capability_catalogue_query",
                "reason": "capability_catalogue_question"})

        if self._looks_like_question(normalized):
            return CognitiveDecision(**{**base, "intent": "direct_question", "intent_confidence": .78,
                "blocked_capabilities": CREATE_CAPABILITIES, "reason": "independent_question"})

        return CognitiveDecision(**base)

    def _apply_pending_contract(self, decision: CognitiveDecision, pending: Any) -> CognitiveDecision:
        if not isinstance(pending, CurrentConversation):
            return decision
        kind = pending.topic
        common = {"current_conversation": pending, "pending_task_kind": kind}
        if self._pending_expired(pending):
            return replace(decision, **common, pending_reason="expired")
        if not self._authority_may_answer(decision, pending):
            return replace(decision, **common, pending_reason="authority_mismatch")

        if kind == "game_guidance_clarification" and decision.intent != "game_guidance_query":
            updates = self.game_guidance.parse_clarification_answer(pending, decision.raw_text)
            if updates:
                return replace(
                    decision, **common,
                    intent="game_guidance_clarification_answer", intent_confidence=.96,
                    is_new_request=False, uses_pending_task=True,
                    pending_resolution_allowed=True, pending_compatible=True,
                    pending_reason="compatible_game_guidance_answer",
                    goal_type="research_game_strategy",
                    allowed_capabilities=[CAP_GAME_GUIDANCE],
                    blocked_capabilities=CREATE_CAPABILITIES + [CAP_FALLBACK_CHAT],
                    allowed_step_types=["state_update", "reply"],
                    response_mode="game_guidance",
                    response_intent="continue_game_guidance",
                    reason="compatible_game_guidance_clarification_answer",
                )

        high_priority = decision.intent in {
            "current_time_query", "current_date_query", "owner_personal_state",
            "direct_question", "command_open_app", "reminder_create_request",
            "appointment_create_request", "capability_catalogue_query",
            "game_guidance_query",
            "cancel_pending", "wake_control", "sleep_control", "owner_manual_command",
        }
        if high_priority:
            return replace(decision, **common, pending_reason="new_request_override")

        if kind == "promotion_target_clarification":
            return replace(
                decision, **common,
                intent="promotion_shoutout", intent_confidence=.92,
                is_new_request=False, uses_pending_task=True,
                pending_resolution_allowed=True, pending_compatible=True,
                pending_reason="promotion_target_answer",
                goal_type="stream_action",
                allowed_capabilities=[CAP_TWITCH_PROMOTION, CAP_TWITCH_ACTION],
                blocked_capabilities=CREATE_CAPABILITIES + [CAP_FALLBACK_CHAT],
                allowed_step_types=["action", "reply"],
                response_mode="command_result",
                response_intent="resolve_promotion_target",
                should_reply=True,
                reason="compatible_promotion_target_clarification",
                action_permission_summary={
                    **(decision.action_permission_summary or {}),
                    "stream_live": bool((decision.action_permission_summary or {}).get("stream_live")),
                    "promotion": True,
                },
            )

        compatible = kind == "appointment_datetime" and self._is_datetime_answer(decision.normalized_text)
        if compatible:
            return replace(
                decision, **common, intent="pending_datetime_answer", intent_confidence=.95,
                is_new_request=False, uses_pending_task=True, pending_resolution_allowed=True,
                pending_compatible=True, pending_reason="datetime_answer", goal_type="create_appointment",
                allowed_capabilities=[CAP_APPOINTMENT], blocked_capabilities=[],
                allowed_step_types=["memory", "reminder", "reply"], response_mode="appointment",
                response_intent="resolve_pending_datetime", reason="compatible_pending_datetime_answer",
            )
        return replace(decision, **common, pending_reason="incompatible_reply_type")

    @staticmethod
    def _pending_expired(pending: CurrentConversation) -> bool:
        return pending.status == ConversationStatus.EXPIRED or pending.expires_at <= time.time()

    @staticmethod
    def _authority_may_answer(decision: CognitiveDecision, pending: CurrentConversation) -> bool:
        expected = pending.expected_reply
        return decision.authority == "owner" and expected is not None and expected.allowed_participant == "leo"

    @staticmethod
    def _source(context: Any) -> str:
        event = getattr(context, "internal_event", None)
        return "twitch" if event is not None and str(getattr(event, "event_type", "")).startswith("twitch_") else "ui"

    @staticmethod
    def _authority(source: str) -> str:
        if source in {"ui", "typed_ui", "voice", "stt_voice", "owner_ui", "owner_stt_direct", "owner_stt_command", "owner_stt_followup"}:
            return "owner"
        if source in {"ambient", "ambient_stt"}:
            return "ambient"
        if source in {"twitch", "twitch_viewer"}:
            return "viewer"
        return "system"

    @staticmethod
    def _is_promotion_command(text: str) -> bool:
        value = f" {text or ''} "
        patterns = (
            r"\b(?:haz(?:le)?|dale|tira)\s+(?:una?\s+)?promo(?:\s+a\b|\b)",
            r"\bpromociona\s+a\b",
            r"\bshoutout\s+(?:a|to)\b",
            r"\b(?:haz(?:le)?|dale|manda|give)\s+(?:un\s+)?shoutout\s+(?:a|to)\b",
            r"\b(?:dale|haz)\s+so\s+a\b",
            r"\b(?:haz(?:le)?|dale|tira)\s+(?:una?\s+)?(?:prom[a-z0-9_]{3,}|so[a-z0-9_]{3,}|shoutout[a-z0-9_]{3,})\b",
        )
        return any(re.search(pattern, value) for pattern in patterns)

    @staticmethod
    def _is_current_time_query(text: str) -> bool:
        has_time = bool(re.search(r"\b(time|hora)\b", text))
        current = bool(re.search(r"\b(current|actual|ahora)\b", text))
        question = bool(re.search(r"\b(what|que|cual|dime|tell|sabes)\b", text))
        copular_question = bool(re.search(r"\b(?:hora|time)\s+(?:es|is)\b", text))
        return has_time and (current or question or copular_question) and not CognitiveRouter._is_appointment_request(text)

    @staticmethod
    def _is_current_date_query(text: str) -> bool:
        has_date = bool(re.search(r"\b(date|fecha|dia)\b", text))
        current = bool(re.search(r"\b(today|hoy|current|actual)\b", text))
        question = bool(re.search(r"\b(what|que|cual|dime|tell)\b", text))
        return has_date and current and question

    @staticmethod
    def _is_appointment_request(text: str) -> bool:
        markers = (
            "cita", "appointment", "medico", "psicolog", "dentista", "reunion",
            "consulta", "reserva", "agendame", "agenda una", "apuntame",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_reminder_request(text: str) -> bool:
        return bool(re.search(r"\b(recuerdame|avisame|recordatorio|remind|reminder)\b", text))

    @staticmethod
    def _is_pending_cancel(text: str) -> bool:
        return bool(re.search(r"\b(?:cancela|cancelar|descarta|olvida|anula|cancel|discard)\b", text)) and bool(
            re.search(r"\b(?:eso|esto|pendiente|cita|appointment|task|tarea)\b", text)
        )

    @staticmethod
    def _wake_control_intent(text: str) -> str | None:
        if re.search(r"\b(?:despierta|despiertate|wake|awake)\b", text):
            return "wake_control"
        if re.search(r"\b(?:duerme|duermete|descansa|sleep)\b", text):
            return "sleep_control"
        return None

    @staticmethod
    def _hinted_owner_route(authority: str, addressed: bool, hints: list[str], stream_is_live: bool) -> dict[str, Any] | None:
        if authority != "owner" or not addressed:
            return None
        capabilities: list[str] = []
        if any(hint in {"tts_control", "pending_tts_reply"} for hint in hints):
            capabilities.append(CAP_TTS_CONTROL)
        if "stream_manual" in hints:
            capabilities.append(CAP_STREAM_STATE)
        if "stream_action" in hints and stream_is_live:
            capabilities.append(CAP_TWITCH_ACTION)
        if not capabilities:
            return None
        return {
            "intent": "owner_manual_command", "intent_confidence": .9,
            "goal_type": "owner_control", "allowed_capabilities": capabilities,
            "blocked_capabilities": CREATE_CAPABILITIES + [CAP_CHAT],
            "allowed_step_types": ["state_update", "action", "reply"],
            "response_mode": "command_result", "reason": "authorized_owner_route_hint",
            "action_permission_summary": {"stream_live": stream_is_live, "external_actions": CAP_TWITCH_ACTION in capabilities},
        }

    @staticmethod
    def _is_open_app_command(text: str) -> bool:
        return bool(re.match(
            r"^(?:(?:hebe|ebe|eve)\s+)?(?:(?:oye|mira|vale|ok|okay|por favor|puedes|podrias)\s+)*"
            r"(?:abre|abrir|inicia|iniciar|arranca|arrancar|lanza|lanzar|ejecuta|ejecutar|open|start|launch|run)\s+(.+)$",
            text,
        ))

    @staticmethod
    def _personal_state(text: str) -> str | None:
        rules = (
            ("hunger", r"\b(?:(?:tengo|con)\s+(?:mucha\s+)?hambre|me (?:ha entrado|dio|da) hambre)\b"),
            ("sleep", r"\b(?:tengo|con)\s+(?:much[oa]\s+)?sueno\b"),
            ("fatigue", r"\b(?:estoy|me siento)\s+(?:muy\s+)?(?:cansad[oa]|agotad[oa]|reventad[oa])\b"),
            ("pain", r"\b(?:me duele|tengo dolor)\b"),
            ("boredom", r"\b(?:estoy|me siento)\s+(?:muy\s+)?aburrid[oa]\b"),
            ("frustration", r"\b(?:estoy|me siento)\s+(?:harto|harta|frustrad[oa]|hasta)\b"),
        )
        for state, pattern in rules:
            if re.search(pattern, text):
                return state
        return None

    @staticmethod
    def _is_datetime_answer(text: str) -> bool:
        number_word = r"(?:una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce)"
        time_value = rf"(?:(?:[01]?\d|2[0-3])(?::[0-5]\d)?|{number_word})"
        clock = bool(re.search(rf"\b(?:a\s+las?|sobre\s+las?|at)\s+{time_value}\b", text))
        date = bool(re.search(
            r"\b(?:hoy|manana|pasado manana|lunes|martes|miercoles|jueves|viernes|sabado|domingo|"
            r"today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"el\s+dia\s+\d{1,2}|el\s+\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|"
            r"julio|agosto|septiembre|octubre|noviembre|diciembre)|"
            r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", text
        ))
        meridiem = bool(re.search(r"\b(?:por la|de la)\s+(?:manana|tarde|noche)\b", text))
        return clock or date or meridiem

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        return bool(re.search(r"\b(?:que|cual|como|cuando|donde|por que|quien|what|which|how|when|where|why|who)\b", text))

    @staticmethod
    def _is_catalogue_query(text: str) -> bool:
        explicit_catalogue = bool(re.search(r"\b(?:capabilit|capacidad|catalog|backlog|todo)\w*\b", text))
        implementation_status = bool(re.search(
            r"\b(?:falta|pendiente|planned|partial|implementar|implementado|implemented)\b", text
        ))
        return explicit_catalogue or implementation_status

    @staticmethod
    def _log(decision: CognitiveDecision, pending: Any) -> None:
        log_jsonl_event("cognitive_router", {
            "message_id": decision.message_id,
            "raw_text": decision.raw_text,
            "normalized_text": decision.normalized_text,
            "source": decision.source,
            "authority": decision.authority,
            "addressed_to_hebe": decision.addressed_to_hebe,
            "intent": decision.intent,
            "confidence": decision.intent_confidence,
            "is_new_request": decision.is_new_request,
            "uses_pending_task": decision.uses_pending_task,
            "pending_kind": decision.pending_task_kind,
            "pending_compatible": decision.pending_compatible,
            "should_reply": decision.should_reply,
            "should_stop_pipeline": decision.should_stop_pipeline,
            "response_mode": decision.response_mode,
            "allowed_capabilities": decision.allowed_capabilities,
            "blocked_capabilities": decision.blocked_capabilities,
            "allowed_step_types": decision.allowed_step_types,
            "blocked_step_types": decision.blocked_step_types,
            "personal_state": decision.personal_state,
            "reason": decision.reason,
        })
        if isinstance(pending, CurrentConversation):
            event_name = "pending_consumed" if decision.uses_pending_task else "pending_rejected"
            if decision.pending_reason == "expired":
                event_name = "pending_expired"
            log_jsonl_event("pending", {
                "event": event_name,
                "message_id": decision.message_id,
                "kind": decision.pending_task_kind or pending.topic,
                "expected_reply_type": pending.expected_reply.type.value if pending.expected_reply else "",
                "source": decision.source,
                "authority": decision.authority,
                "pending_compatible": decision.pending_compatible,
                "compatibility_reason": decision.pending_reason,
            })
        print(
            "[HEBE][COGNITIVE_ROUTER] "
            f"intent={decision.intent} confidence={decision.intent_confidence:.2f} "
            f"new_request={str(decision.is_new_request).lower()} "
            f"uses_pending={str(decision.uses_pending_task).lower()} "
            f"should_reply={str(decision.should_reply).lower()} "
            f"stop={str(decision.should_stop_pipeline).lower()} "
            f"state={decision.personal_state or 'none'} reason={decision.reason}", flush=True,
        )
        print(
            "[HEBE][PENDING_ROUTER] "
            f"active={str(bool(pending)).lower()} kind={decision.pending_task_kind or 'none'} "
            f"compatible={str(decision.pending_compatible).lower()} reason={decision.pending_reason}", flush=True,
        )
