# backend/app/cognitive/scheduler.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.cognitive.memory_store import MemoryStore, Reminder


@dataclass(slots=True)
class InternalEvent:
    event_type: str
    payload: dict
    created_at: str


class SchedulerService:
    """
    Scheduler mínimo para Hebe v1.

    Responsabilidades:
    - consultar reminders vencidos
    - marcarlos como fired
    - emitir eventos internos reminder_due
    - dejar trazabilidad en DB

    Importante:
    - no ejecuta respuestas por sí mismo
    - no habla por sí mismo
    - solo convierte reminders -> eventos internos
    """

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self._pending: deque[InternalEvent] = deque()

    # Reemplazar poll_due_events por:
    def poll_due_events(self, limit: int = 20) -> list[InternalEvent]:
        events: list[InternalEvent] = []

        # 1. Eventos pusheados manualmente (Twitch y otros conectores)
        while self._pending and len(events) < limit:
            events.append(self._pending.popleft())

        # 2. Reminders vencidos (lógica existente)
        remaining = limit - len(events)
        if remaining > 0:
            due_reminders = self.memory_store.list_due_reminders(limit=remaining)
            for reminder in due_reminders:
                event = self._fire_reminder(reminder)
                if event is not None:
                    events.append(event)

        return events

    def _fire_reminder(self, reminder: Reminder) -> Optional[InternalEvent]:
        """
        Convierte un reminder vencido en un InternalEvent.
        Marca el reminder como fired para evitar re-disparos.
        """
        try:
            self.memory_store.mark_reminder_fired(reminder.id)

            payload = {
                "reminder_id": reminder.id,
                "kind": reminder.kind,
                "title": reminder.title,
                "message": reminder.message,
                "due_at": reminder.due_at,
                "timezone": reminder.timezone,
                "status": "fired",
                "source_memory_id": reminder.source_memory_id,
                "payload": reminder.payload,
            }

            created_at = self._utc_now_iso()

            self.memory_store.log_internal_event(
                event_type="reminder_due",
                payload=payload,
            )

            return InternalEvent(
                event_type="reminder_due",
                payload=payload,
                created_at=created_at,
            )
        except Exception as e:
            print(f"⚠️ Scheduler: no se pudo disparar reminder {reminder.id}: {e}")
            return None

    def build_manual_event(
        self,
        event_type: str,
        payload: Optional[dict] = None,
    ) -> InternalEvent:
        """
        Helper útil para tests o futuros eventos internos.
        """
        safe_payload = payload or {}
        created_at = self._utc_now_iso()

        self.memory_store.log_internal_event(
            event_type=event_type,
            payload=safe_payload,
        )

        return InternalEvent(
            event_type=event_type,
            payload=safe_payload,
            created_at=created_at,
        )

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
    
    def push_event(self, event_type: str, payload: Optional[dict] = None) -> InternalEvent:
        """
        Encola un evento manual para que poll_due_events lo entregue
        al orquestador en su siguiente tick.

        Útil para eventos externos (Twitch, futuros conectores) que deben
        atravesar el mismo pipeline cognitivo que reminder_due.
        """
        event = self.build_manual_event(event_type=event_type, payload=payload)
        self._pending.append(event)
        return event
