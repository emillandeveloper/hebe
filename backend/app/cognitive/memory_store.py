# backend/app/cognitive/memory_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

from app.services import db_sqlite


@dataclass(slots=True)
class MemoryFact:
    id: int
    kind: str
    subject: Optional[str]
    payload: Optional[dict]
    source_text: Optional[str]
    confidence: float
    created_at: str
    updated_at: str
    last_used_at: Optional[str]
    active: bool


@dataclass(slots=True)
class Reminder:
    id: int
    kind: str
    title: str
    message: Optional[str]
    due_at: str
    timezone: str
    status: str
    source_memory_id: Optional[int]
    payload: Optional[dict]
    created_at: str
    fired_at: Optional[str]


class MemoryStore:
    """
    Capa de acceso a memoria persistente de Hebe v1.

    Responsabilidades:
    - guardar facts estructurados
    - buscarlos por tipo o texto
    - crear reminders ligados a un fact
    - recuperar reminders pendientes / vencidos
    - marcar uso de memorias
    """

    # =========================
    # Memory facts
    # =========================

    def create_fact(
        self,
        kind: str,
        subject: Optional[str] = None,
        payload: Optional[dict] = None,
        source_text: Optional[str] = None,
        confidence: float = 1.0,
        active: bool = True,
    ) -> MemoryFact:
        memory_id = db_sqlite.insert_memory_fact(
            kind=kind,
            subject=subject,
            payload=payload,
            source_text=source_text,
            confidence=confidence,
            active=active,
        )
        fact = self.get_fact(memory_id)
        if fact is None:
            raise RuntimeError(f"No se pudo recuperar memory_fact recién creada: {memory_id}")
        return fact

    def get_fact(self, memory_id: int) -> Optional[MemoryFact]:
        row = db_sqlite.get_memory_fact(memory_id)
        if not row:
            return None
        return self._to_memory_fact(row)

    def search_facts(
        self,
        query_text: Optional[str] = None,
        kind: Optional[str] = None,
        active_only: bool = True,
        limit: int = 10,
        touch: bool = False,
    ) -> list[MemoryFact]:
        rows = db_sqlite.search_memory_facts(
            query_text=query_text,
            kind=kind,
            active_only=active_only,
            limit=limit,
        )
        facts = [self._to_memory_fact(row) for row in rows]

        if touch:
            for fact in facts:
                db_sqlite.touch_memory_fact(fact.id)

        return facts

    def deactivate_fact(self, memory_id: int) -> None:
        db_sqlite.deactivate_memory_fact(memory_id)

    def touch_fact(self, memory_id: int) -> None:
        db_sqlite.touch_memory_fact(memory_id)

    # =========================
    # Higher-level helpers
    # =========================

    def create_appointment(
        self,
        title: str,
        due_at_iso: str,
        source_text: Optional[str] = None,
        notes: Optional[str] = None,
        timezone_name: str = "Europe/Madrid",
        reminder_message: Optional[str] = None,
        create_reminder: bool = True,
    ) -> tuple[MemoryFact, Optional[Reminder]]:
        """
        Helper específico para citas tipo:
        'me han puesto la psicóloga el 12 a las 3 y media'
        """
        payload = {
            "title": title,
            "due_at": due_at_iso,
            "timezone": timezone_name,
            "notes": notes,
        }

        fact = self.create_fact(
            kind="appointment",
            subject=title,
            payload=payload,
            source_text=source_text,
            confidence=1.0,
            active=True,
        )

        reminder = None
        if create_reminder:
            reminder = self.create_reminder(
                title=title,
                due_at=due_at_iso,
                message=reminder_message or f"Te recuerdo: {title}",
                kind="appointment",
                timezone_name=timezone_name,
                source_memory_id=fact.id,
                payload=payload,
            )

        return fact, reminder

    def get_recent_appointments(self, limit: int = 5) -> list[MemoryFact]:
        return self.search_facts(kind="appointment", active_only=True, limit=limit)

    # =========================
    # Reminders
    # =========================

    def create_reminder(
        self,
        title: str,
        due_at: str,
        message: Optional[str] = None,
        kind: str = "generic",
        timezone_name: str = "Europe/Madrid",
        source_memory_id: Optional[int] = None,
        payload: Optional[dict] = None,
    ) -> Reminder:
        reminder_id = db_sqlite.create_reminder(
            title=title,
            due_at=due_at,
            message=message,
            kind=kind,
            timezone_name=timezone_name,
            source_memory_id=source_memory_id,
            payload=payload,
        )
        reminder = self.get_reminder(reminder_id)
        if reminder is None:
            raise RuntimeError(f"No se pudo recuperar reminder recién creado: {reminder_id}")
        return reminder

    def get_reminder(self, reminder_id: int) -> Optional[Reminder]:
        row = db_sqlite.get_reminder(reminder_id)
        if not row:
            return None
        return self._to_reminder(row)

    def list_due_reminders(self, limit: int = 20) -> list[Reminder]:
        rows = db_sqlite.list_due_reminders(limit=limit)
        return [self._to_reminder(row) for row in rows]

    def list_pending_reminders(self, limit: int = 20) -> list[Reminder]:
        rows = db_sqlite.list_pending_reminders(limit=limit)
        return [self._to_reminder(row) for row in rows]

    def mark_reminder_fired(self, reminder_id: int) -> None:
        db_sqlite.mark_reminder_fired(reminder_id)

    def mark_reminder_done(self, reminder_id: int) -> None:
        db_sqlite.mark_reminder_done(reminder_id)

    def cancel_reminder(self, reminder_id: int) -> None:
        db_sqlite.cancel_reminder(reminder_id)

    # =========================
    # Event logging
    # =========================

    def log_internal_event(self, event_type: str, payload: Optional[dict] = None) -> int:
        return db_sqlite.log_internal_event(event_type=event_type, payload=payload)

    # =========================
    # Mapping helpers
    # =========================

    def _to_memory_fact(self, row: dict[str, Any]) -> MemoryFact:
        return MemoryFact(
            id=int(row["id"]),
            kind=str(row["kind"]),
            subject=row.get("subject"),
            payload=row.get("payload"),
            source_text=row.get("source_text"),
            confidence=float(row.get("confidence", 1.0)),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_used_at=row.get("last_used_at"),
            active=bool(row.get("active", 1)),
        )

    def _to_reminder(self, row: dict[str, Any]) -> Reminder:
        return Reminder(
            id=int(row["id"]),
            kind=str(row["kind"]),
            title=str(row["title"]),
            message=row.get("message"),
            due_at=str(row["due_at"]),
            timezone=str(row.get("timezone") or "UTC"),
            status=str(row["status"]),
            source_memory_id=row.get("source_memory_id"),
            payload=row.get("payload"),
            created_at=str(row["created_at"]),
            fired_at=row.get("fired_at"),
        )