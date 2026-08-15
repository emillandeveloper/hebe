from __future__ import annotations

import re
import unicodedata
import uuid
from typing import Any

from app.cognitive.capabilities.goal import Goal, GOAL_TYPES
from app.cognitive.game_guidance import GameGuidanceCapability


class GoalExtractor:
    def __init__(self, game_guidance: GameGuidanceCapability | None = None):
        self.game_guidance = game_guidance or GameGuidanceCapability()

    def extract(self, context: Any) -> Goal:
        raw_text = str(getattr(context, "input_text", "") or "")
        normalized = self._normalize(raw_text)
        source = self._source_from_context(context)
        entities = list(getattr(context, "resolved_entities", []) or [])
        slots: dict[str, Any] = {}
        missing_slots: list[str] = []
        goal_type = "answer_question"
        confidence = 0.45
        reasoning = "default conversational question"
        risk_level = "low"
        requires_confirmation = False
        spoiler_sensitivity = "normal"

        decision = getattr(context, "cognitive_decision", None)
        if decision is not None:
            goal_type = decision.goal_type
            confidence = decision.intent_confidence
            reasoning = f"CognitiveRouter decision: {decision.reason}"
            if decision.personal_state:
                slots["personal_state"] = decision.personal_state
            if decision.intent == "game_guidance_query":
                guidance_decision = self.game_guidance.evaluate(context)
                context.game_guidance_decision = guidance_decision
                guidance = guidance_decision.context
                slots.update({
                    "game": guidance.game,
                    "spoiler_policy": guidance.spoiler_policy,
                    "run_state": guidance.source_context.get("GameRunState") or {},
                })
                spoiler_sensitivity = guidance.spoiler_policy

        catalogue_query = self._detect_catalogue_query(normalized)
        if decision is not None:
            if catalogue_query:
                slots["catalogue_query"] = catalogue_query
        elif catalogue_query:
            goal_type = "analyze_data"
            confidence = 0.9
            reasoning = "capability backlog/catalogue query detected"
            slots["catalogue_query"] = catalogue_query
        elif self._looks_like_chat_activity(normalized):
            goal_type = "analyze_chat_activity"
            confidence = 0.82
            reasoning = "chat activity analysis request detected"
            slots.update({
                "time_range": "all_recorded_history",
                "metric": "message_count",
                "exclude_bots": True,
            })
        elif self._looks_like_game_strategy(normalized):
            guidance_decision = self.game_guidance.evaluate(context)
            context.game_guidance_decision = guidance_decision
            goal_type = "research_game_strategy"
            confidence = 0.84
            reasoning = "game strategy research request detected"
            slots.update({
                "game": guidance_decision.context.normalized_game,
                "strategy_mode": guidance_decision.context.playthrough_type,
            })
            if not slots["game"]:
                missing_slots.append("game")
            spoiler_sensitivity = "mechanics_ok_story_avoid"
        elif self._looks_like_session_update(normalized):
            goal_type = "correct_assumption" if self._looks_like_correction(normalized) else "update_session_state"
            confidence = 0.78
            reasoning = "live session state update detected"
            slots.update({
                "event": "progress_update",
                "state_change": raw_text.strip(),
            })
        elif self._looks_like_schedule(normalized):
            goal_type = "schedule_task"
            confidence = 0.78
            reasoning = "schedule/reminder request detected"
            slots["source_text"] = raw_text
        elif self._looks_like_diagnostic(normalized):
            goal_type = "diagnose_problem"
            confidence = 0.72
            reasoning = "diagnostic request detected"

        if goal_type not in GOAL_TYPES:
            goal_type = "unknown"

        goal_id = str(uuid.uuid4())
        message_id = self._message_id_from_context(context) or f"msg_{goal_id}"
        goal = Goal(
            goal_id=goal_id,
            message_id=message_id,
            goal_type=goal_type,
            raw_text=raw_text,
            normalized_text=normalized,
            source=source,
            target_audience="stream" if source == "twitch" else "leo",
            entities=entities,
            slots=slots,
            missing_slots=missing_slots,
            confidence=confidence,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            spoiler_sensitivity=spoiler_sensitivity,
            memory_relevance="relevant",
            reasoning_summary=reasoning,
            should_reply_candidate=True,
        )
        print(
            "[HEBE][GOAL] extracted "
            f"goal_type={goal.goal_type} confidence={goal.confidence:.2f} "
            f"slots={list(goal.slots.keys())!r}",
            flush=True,
        )
        return goal

    def _message_id_from_context(self, context: Any) -> str:
        for attr in ("message_id", "event_id", "input_message_id"):
            value = str(getattr(context, attr, "") or "").strip()
            if value:
                return value
        input_event = getattr(context, "input_event", None)
        metadata = getattr(input_event, "stt_metadata", None) if input_event is not None else None
        if isinstance(metadata, dict):
            value = str(metadata.get("message_id") or "").strip()
            if value:
                return value
        return ""

    def _source_from_context(self, context: Any) -> str:
        internal_event = getattr(context, "internal_event", None)
        if internal_event is not None and getattr(internal_event, "event_type", "").startswith("twitch_"):
            return "twitch"
        state_snapshot = getattr(context, "state_snapshot", {}) or {}
        if state_snapshot.get("live_mode") or state_snapshot.get("stream_mode"):
            return "stream"
        return "ui"

    def _normalize(self, text: str) -> str:
        raw = (text or "").strip().lower()
        without_accents = "".join(
            char for char in unicodedata.normalize("NFKD", raw)
            if not unicodedata.combining(char)
        )
        cleaned = re.sub(r"[^a-z0-9ñ\s/_-]", " ", without_accents)
        return " ".join(cleaned.split())

    def _detect_catalogue_query(self, normalized: str) -> str | None:
        if any(term in normalized for term in ("falta por implementar", "por implementar", "sin implementar")):
            return "planned_not_implemented"
        if not any(token in normalized for token in ("capability", "capabilities", "capacidades", "catalog", "catalogo", "backlog", "todo")):
            return None
        if any(token in normalized for token in ("planned", "planead", "not implemented", "sin implementar", "pendiente")):
            return "planned_not_implemented"
        if any(token in normalized for token in ("high priority", "alta prioridad", "p0", "p1", "unblocked", "desbloquead")):
            return "high_priority_unblocked"
        if any(token in normalized for token in ("next", "siguiente", "recomendad", "todo")):
            return "next_todo"
        if any(token in normalized for token in ("disabled", "desactivad", "apagada")):
            return "implemented_disabled"
        if any(token in normalized for token in ("partial", "parcial", "incomplete", "completion", "completar")):
            return "partial_needs_completion"
        return "summary"

    def _looks_like_chat_activity(self, normalized: str) -> bool:
        return (
            any(token in normalized for token in ("chat", "viewer", "viewers", "twitch"))
            and any(token in normalized for token in ("actividad", "activity", "quien", "who", "mensajes", "message", "habla", "talk"))
        )

    def _looks_like_game_strategy(self, normalized: str) -> bool:
        return (
            any(token in normalized for token in ("romper", "break", "optimizar", "optimize", "estrategia", "strategy", "build"))
            and any(token in normalized for token in ("juego", "game", "kingdom", "hearts", "boss", "nivel", "level", "deck"))
        )

    def _looks_like_session_update(self, normalized: str) -> bool:
        progress_terms = ("vencimos", "derrotamos", "defeated", "matamos", "pasamos", "ya pase", "ya hemos", "boss")
        return any(term in normalized for term in progress_terms)

    def _looks_like_correction(self, normalized: str) -> bool:
        return any(token in normalized for token in ("pero", "no ", "actually", "en realidad", "ya "))

    def _looks_like_schedule(self, normalized: str) -> bool:
        return any(token in normalized for token in ("recuerdame", "avisame", "recordatorio", "cita", "agenda"))

    def _looks_like_diagnostic(self, normalized: str) -> bool:
        return any(token in normalized for token in ("diagnostica", "debug", "por que no", "no responde", "fallo", "error"))

    def _extract_game_name(self, raw_text: str, normalized: str) -> str:
        match = re.search(r"(?:juego|game)\s+(.+)$", raw_text, flags=re.IGNORECASE)
        return match.group(1).strip(" .,:;") if match else ""
