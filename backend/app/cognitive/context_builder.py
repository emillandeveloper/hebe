# backend/app/cognitive/context_builder.py
from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Any

from app.cognitive.memory_store import MemoryStore, MemoryFact, Reminder
from app.cognitive.scheduler import InternalEvent
from app.core.state import HebeState


@dataclass(slots=True)
class BuiltContext:
    """
    Contexto mínimo estructurado para Hebe v1.
    Esto es lo que consumirá deliberation y response synthesis.
    """
    input_text: Optional[str]
    internal_event: Optional[InternalEvent]

    # Memoria relevante
    relevant_facts: list[MemoryFact]
    recent_appointments: list[MemoryFact]

    # Estado de reminders
    pending_reminders: list[Reminder]

    # Estado del sistema
    state_snapshot: dict[str, Any]


class ContextBuilder:
    """
    Construye el contexto cognitivo para cada interacción.

    NO decide nada.
    NO ejecuta nada.
    SOLO recopila información relevante.
    """

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store

    # =========================
    # Entry point
    # =========================

    def build(
        self,
        state: HebeState,
        input_text: Optional[str] = None,
        internal_event: Optional[InternalEvent] = None,
    ) -> BuiltContext:
        """
        Construye contexto tanto para:
        - input del usuario
        - eventos internos (ej: reminder_due)
        """

        relevant_facts = self._get_relevant_facts(input_text)
        recent_appointments = self.memory_store.get_recent_appointments(limit=3)
        pending_reminders = self.memory_store.list_pending_reminders(limit=5)

        state_snapshot = self._build_state_snapshot(state)

        return BuiltContext(
            input_text=input_text,
            internal_event=internal_event,
            relevant_facts=relevant_facts,
            recent_appointments=recent_appointments,
            pending_reminders=pending_reminders,
            state_snapshot=state_snapshot,
        )

    # =========================
    # Memory gathering
    # =========================

    def _get_relevant_facts(
        self,
        input_text: Optional[str],
        limit: int = 5,
    ) -> list[MemoryFact]:
        """
        Búsqueda básica de memoria relevante.
        En v1 es simple: LIKE sobre texto.
        """
        if not input_text:
            return []

        facts = self.memory_store.search_facts(
            query_text=input_text,
            active_only=True,
            limit=limit,
            touch=True,  # importante para ranking futuro
        )

        return facts

    # =========================
    # State snapshot
    # =========================

    def _build_state_snapshot(self, state: HebeState) -> dict[str, Any]:
        stream = getattr(state, "stream", None)

        return {
            "now_iso": __import__("datetime").datetime.now().astimezone().isoformat(),
            "mode": getattr(state, "mode", None),
            "is_processing": getattr(state, "is_processing", None),
            "last_intent": getattr(state, "last_intent", None),
            "current_task": getattr(state, "current_task", None),
            "pending_clarification": getattr(state, "pending_clarification", None),
            "stream_enabled": getattr(stream, "enabled", False) if stream else False,
            "stream_armed": getattr(stream, "armed", False) if stream else False,
        }