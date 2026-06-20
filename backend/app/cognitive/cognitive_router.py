from __future__ import annotations

import re
import time
import unicodedata
import uuid
from dataclasses import replace
from typing import Any

from app.cognitive.cognitive_decision import CognitiveDecision


CAP_TIME = "time.get_current_time"
CAP_DATE = "time.get_current_date"
CAP_CHAT = "hebe.chat_reply"
CAP_APPOINTMENT = "appointment.create"
CAP_REMINDER = "reminder.create"
CAP_SCHEDULER = "scheduler.create"
CAP_OPEN_APP = "pc.open_application"
CAP_TWITCH_ACTION = "twitch_action"

CREATE_CAPABILITIES = [CAP_APPOINTMENT, CAP_REMINDER, CAP_SCHEDULER]


class CognitiveRouter:
    """Owns intent priority and grants downstream subsystems permission to run."""

    def route(self, context: Any) -> CognitiveDecision:
        raw = str(getattr(context, "input_text", "") or "")
        normalized = self.normalize(raw)
        source = str(getattr(context, "source", "") or self._source(context))
        authority = str(getattr(context, "authority", "") or self._authority(source))
        addressed = bool(getattr(context, "addressed_to_hebe", authority == "owner"))
        message_id = str(getattr(context, "message_id", "") or f"msg_{uuid.uuid4().hex}")
        pending = (getattr(context, "state_snapshot", {}) or {}).get("pending_clarification")

        decision = self._classify(
            message_id=message_id,
            source=source,
            authority=authority,
            addressed=addressed,
            raw=raw,
            normalized=normalized,
        )
        decision = self._apply_pending_contract(decision, pending)
        decision.debug_trace.extend([
            f"intent:{decision.intent}",
            f"priority_reason:{decision.reason}",
            f"pending:{decision.pending_reason}",
        ])
        self._log(decision, pending)
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
        addressed: bool, raw: str, normalized: str,
    ) -> CognitiveDecision:
        base = dict(
            message_id=message_id, source=source, authority=authority,
            addressed_to_hebe=addressed, input_text=raw, normalized_text=normalized,
            intent="unknown_chat", intent_confidence=0.45, is_new_request=True,
            uses_pending_task=False, goal_type="answer_question",
            required_capability_ids=[CAP_CHAT], blocked_capability_ids=[],
            allowed_step_types=["reply"], should_reply=True, response_mode="chat",
            response_intent="answer_user", reason="fallback_chat",
        )

        if not normalized:
            return CognitiveDecision(**{**base, "intent": "ambient_noise", "intent_confidence": 1.0,
                "is_new_request": False, "required_capability_ids": [], "should_reply": False,
                "should_stop_pipeline": True, "response_mode": "silent", "reason": "empty_input"})

        if authority == "ambient" and not addressed:
            return CognitiveDecision(**{**base, "intent": "stream_context_update", "intent_confidence": .9,
                "is_new_request": False, "goal_type": "update_session_state",
                "required_capability_ids": [], "blocked_capability_ids": CREATE_CAPABILITIES + [CAP_TWITCH_ACTION],
                "allowed_step_types": ["state_update"], "should_reply": False,
                "should_stop_pipeline": True, "response_mode": "silent", "reason": "ambient_context_only"})

        if self._is_current_time_query(normalized):
            return CognitiveDecision(**{**base, "intent": "current_time_query", "intent_confidence": .97,
                "goal_type": "answer_current_time", "required_capability_ids": [CAP_TIME],
                "blocked_capability_ids": CREATE_CAPABILITIES, "response_mode": "time_answer",
                "response_intent": "report_current_time", "reason": "explicit_time_question"})

        if self._is_current_date_query(normalized):
            return CognitiveDecision(**{**base, "intent": "current_date_query", "intent_confidence": .96,
                "goal_type": "answer_current_date", "required_capability_ids": [CAP_DATE],
                "blocked_capability_ids": CREATE_CAPABILITIES, "response_mode": "date_answer",
                "response_intent": "report_current_date", "reason": "explicit_date_question"})

        state = self._personal_state(normalized)
        if authority == "owner" and addressed and state:
            return CognitiveDecision(**{**base, "intent": "owner_personal_state", "intent_confidence": .94,
                "goal_type": "respond_to_personal_state", "required_capability_ids": [],
                "blocked_capability_ids": CREATE_CAPABILITIES + [CAP_TWITCH_ACTION],
                "response_mode": "companion_reaction", "response_intent": "react_to_personal_state",
                "reason": f"owner_state_signal:{state}", "personal_state": state})

        app_target = self._open_app_target(normalized)
        if app_target:
            return CognitiveDecision(**{**base, "intent": "command_open_app", "intent_confidence": .95,
                "goal_type": "control_pc", "required_capability_ids": [CAP_OPEN_APP],
                "blocked_capability_ids": CREATE_CAPABILITIES, "allowed_step_types": ["action", "reply"],
                "response_mode": "confirm_action", "response_intent": "confirm_pc_action",
                "reason": "explicit_open_application_command"})

        if self._is_reminder_request(normalized):
            return CognitiveDecision(**{**base, "intent": "reminder_create_request", "intent_confidence": .92,
                "goal_type": "create_reminder", "required_capability_ids": [CAP_REMINDER],
                "blocked_capability_ids": [CAP_APPOINTMENT], "allowed_step_types": ["reminder", "reply"],
                "response_mode": "reminder", "reason": "explicit_reminder_marker"})

        if self._is_appointment_request(normalized):
            return CognitiveDecision(**{**base, "intent": "appointment_create_request", "intent_confidence": .93,
                "goal_type": "create_appointment", "required_capability_ids": [CAP_APPOINTMENT],
                "blocked_capability_ids": [], "allowed_step_types": ["memory", "reminder", "reply"],
                "response_mode": "appointment", "reason": "strong_appointment_marker"})

        if self._is_catalogue_query(normalized):
            return CognitiveDecision(**{**base, "intent": "capability_catalogue_query", "intent_confidence": .88,
                "goal_type": "analyze_data", "required_capability_ids": ["hebe.capability_backlog_query"],
                "blocked_capability_ids": CREATE_CAPABILITIES, "response_mode": "capability_catalogue_query",
                "reason": "capability_catalogue_question"})

        if self._looks_like_question(normalized):
            return CognitiveDecision(**{**base, "intent": "direct_question", "intent_confidence": .78,
                "blocked_capability_ids": CREATE_CAPABILITIES, "reason": "independent_question"})

        return CognitiveDecision(**base)

    def _apply_pending_contract(self, decision: CognitiveDecision, pending: Any) -> CognitiveDecision:
        if not isinstance(pending, dict) or not pending:
            return decision
        pending_id = str(pending.get("id") or pending.get("task_id") or "pending_clarification")
        kind = str(pending.get("kind") or "unknown")
        common = {"pending_task_id": pending_id, "pending_task_kind": kind}
        if self._pending_expired(pending):
            return replace(decision, **common, pending_reason="expired")
        if not self._authority_may_answer(decision, pending):
            return replace(decision, **common, pending_reason="authority_mismatch")

        high_priority = decision.intent in {
            "current_time_query", "current_date_query", "owner_personal_state",
            "direct_question", "command_open_app", "reminder_create_request",
            "appointment_create_request", "capability_catalogue_query",
        }
        if high_priority:
            return replace(decision, **common, pending_reason="new_request_override")

        compatible = kind == "appointment_datetime" and self._is_datetime_answer(decision.normalized_text)
        if compatible:
            return replace(
                decision, **common, intent="pending_datetime_answer", intent_confidence=.95,
                is_new_request=False, uses_pending_task=True, pending_resolution_allowed=True,
                pending_compatible=True, pending_reason="datetime_answer", goal_type="create_appointment",
                required_capability_ids=[CAP_APPOINTMENT], blocked_capability_ids=[],
                allowed_step_types=["memory", "reminder", "reply"], response_mode="appointment",
                response_intent="resolve_pending_datetime", reason="compatible_pending_datetime_answer",
            )
        return replace(decision, **common, pending_reason="incompatible_reply_type")

    @staticmethod
    def _pending_expired(pending: dict) -> bool:
        expires_at = pending.get("expires_at")
        if expires_at is None:
            return False
        try:
            return float(expires_at) <= time.time()
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _authority_may_answer(decision: CognitiveDecision, pending: dict) -> bool:
        expected = str(pending.get("authority") or pending.get("expected_authority") or "owner")
        return decision.authority == expected

    @staticmethod
    def _source(context: Any) -> str:
        event = getattr(context, "internal_event", None)
        return "twitch" if event is not None and str(getattr(event, "event_type", "")).startswith("twitch_") else "ui"

    @staticmethod
    def _authority(source: str) -> str:
        if source in {"ui", "typed_ui", "voice", "stt_voice", "owner_ui"}:
            return "owner"
        if source in {"ambient", "ambient_stt"}:
            return "ambient"
        if source in {"twitch", "twitch_viewer"}:
            return "viewer"
        return "system"

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
    def _open_app_target(text: str) -> str:
        match = re.search(r"(?:^|\s)(?:abre|abrir|inicia|iniciar|arranca|lanza|ejecuta|open|start|launch|run)\s+(.+)$", text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _personal_state(text: str) -> str | None:
        rules = (
            ("hunger", r"\b(?:(?:tengo|con)\s+(?:mucha\s+)?hambre|me (?:ha entrado|dio|da) hambre)\b"),
            ("sleepy", r"\b(?:tengo|con)\s+(?:much[oa]\s+)?sueno\b"),
            ("fatigue", r"\b(?:estoy|me siento)\s+(?:muy\s+)?(?:cansad[oa]|agotad[oa]|reventad[oa])\b"),
            ("pain", r"\b(?:me duele|tengo dolor)\b"),
            ("bored", r"\b(?:estoy|me siento)\s+(?:muy\s+)?aburrid[oa]\b"),
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
            r"el\s+dia\s+\d{1,2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", text
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
        print(
            "[HEBE][COGNITIVE_ROUTER] "
            f"intent={decision.intent} confidence={decision.intent_confidence:.2f} "
            f"new_request={str(decision.is_new_request).lower()} "
            f"uses_pending={str(decision.uses_pending_task).lower()} "
            f"should_reply={str(decision.should_reply).lower()} reason={decision.reason}", flush=True,
        )
        print(
            "[HEBE][PENDING_ROUTER] "
            f"active={str(bool(pending)).lower()} kind={decision.pending_task_kind or 'none'} "
            f"compatible={str(decision.pending_compatible).lower()} reason={decision.pending_reason}", flush=True,
        )
