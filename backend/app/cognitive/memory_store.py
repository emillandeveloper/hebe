from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from app.epistemics.memory_migration import legacy_memory_fact_migrations, map_memory_fact
from app.epistemics.models import Belief, BeliefStatus, EvidenceRef, RetrievalRequest
from app.epistemics.repository import BeliefRepository
from app.epistemics.retrieval import MemoryRetrievalCoordinator
from app.epistemics.service import BeliefLifecycleService
from app.replay.migrations import MigrationRunner, belief_v2_migrations
from app.services import db_sqlite


@dataclass(slots=True)
class MemoryFact:
    """Presentation DTO for a canonical structured-memory belief."""

    id: str
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
    """Structured-memory facade backed exclusively by canonical beliefs.

    Reminders retain their scheduler table. The deprecated ``memory_facts``
    table is absent from this runtime API and is read only by the versioned
    startup migration.
    """

    def __init__(
        self,
        repository: BeliefRepository | None = None,
        lifecycle: BeliefLifecycleService | None = None,
        retrieval: MemoryRetrievalCoordinator | None = None,
    ) -> None:
        if repository is None:
            runner = MigrationRunner(db_sqlite.get_db_connection)
            runner.migrate(belief_v2_migrations())
            runner.migrate(legacy_memory_fact_migrations())
            repository = BeliefRepository(db_sqlite.get_db_connection)
        self.repository = repository
        self.lifecycle = lifecycle or BeliefLifecycleService(repository)
        self.retrieval = retrieval or MemoryRetrievalCoordinator(repository)

    @classmethod
    def from_connection_factory(
        cls,
        connection_factory,
        *,
        run_legacy_migration: bool = True,
    ) -> "MemoryStore":
        runner = MigrationRunner(connection_factory)
        runner.migrate(belief_v2_migrations())
        if run_legacy_migration:
            runner.migrate(legacy_memory_fact_migrations())
        repository = BeliefRepository(connection_factory)
        return cls(
            repository=repository,
            lifecycle=BeliefLifecycleService(repository),
            retrieval=MemoryRetrievalCoordinator(repository),
        )

    def create_fact(
        self,
        kind: str,
        subject: Optional[str] = None,
        payload: Optional[dict] = None,
        source_text: Optional[str] = None,
        confidence: float = 1.0,
        active: bool = True,
    ) -> MemoryFact:
        claim = map_memory_fact(
            kind=kind, subject=subject, payload=payload or {}, source_text=source_text,
        )
        literal_text = str(source_text or (payload or {}).get("text") or "").strip()
        fingerprint = json.dumps(
            {
                "namespace": claim.namespace,
                "scope_kind": claim.scope_kind,
                "scope_id": claim.scope_id,
                "subject": claim.subject_ref,
                "predicate": claim.predicate,
                "object": claim.object_value,
                "source": literal_text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        source_event_id = "memory_write:" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        belief = self.lifecycle.propose(
            namespace=claim.namespace,
            scope_kind=claim.scope_kind,
            scope_id=claim.scope_id,
            subject_ref=claim.subject_ref,
            predicate=claim.predicate,
            object_value=claim.object_value,
            confidence=confidence,
            authority_class="owner",
            status=BeliefStatus.INFERRED if active else BeliefStatus.HISTORICAL,
            evidence=EvidenceRef(
                source_event_id=source_event_id,
                source_record_type="memory_write",
                source_record_id=source_event_id,
                observed_at=time.time(),
                extractor="canonical_memory_store",
                extractor_version="v1",
                literal_span={"text": literal_text, "payload": payload or {}},
            ),
        )
        if belief is None:
            raise RuntimeError("canonical memory write rejected by belief lifecycle")
        return self._to_memory_fact(belief)

    def upsert_fact(
        self,
        kind: str,
        subject: Optional[str] = None,
        payload: Optional[dict] = None,
        source_text: Optional[str] = None,
        confidence: float = 1.0,
    ) -> tuple[MemoryFact, bool]:
        claim = map_memory_fact(kind=kind, subject=subject, payload=payload or {}, source_text=source_text)
        previous = self.repository.active_for_identity(
            namespace=claim.namespace,
            scope_kind=claim.scope_kind,
            scope_id=claim.scope_id,
            subject_ref=claim.subject_ref,
            predicate=claim.predicate,
        )
        fact = self.create_fact(
            kind=kind, subject=subject, payload=payload, source_text=source_text,
            confidence=confidence, active=True,
        )
        created = all(item.id != fact.id for item in previous)
        if created:
            for old in previous:
                if old.id != fact.id and old.object_value != claim.object_value and not old.owner_confirmed:
                    self.lifecycle.supersede(old.id, superseded_by=fact.id)
        return fact, created

    def get_fact(self, memory_id: str) -> Optional[MemoryFact]:
        belief = self.repository.get(str(memory_id))
        if belief is None or not belief.namespace.startswith("memory."):
            return None
        return self._to_memory_fact(belief)

    def search_facts(
        self,
        query_text: Optional[str] = None,
        kind: Optional[str] = None,
        active_only: bool = True,
        limit: int = 10,
        touch: bool = False,
    ) -> list[MemoryFact]:
        del touch  # Beliefs do not mutate merely because retrieval selected them.
        if active_only:
            result = self.retrieval.retrieve(RetrievalRequest(
                context_kind="owner_local",
                purpose="structured_memory",
                max_results=max(limit * 8, 50),
                provenance_required=True,
            ))
            beliefs = [self.repository.get(str(item["id"])) for item in result.selected_claims]
            beliefs = [item for item in beliefs if item is not None]
        else:
            beliefs = self.repository.list()
        facts = [self._to_memory_fact(item) for item in beliefs if item.namespace.startswith("memory.")]
        if kind:
            facts = [fact for fact in facts if fact.kind == kind]
        if query_text:
            query = str(query_text).casefold().strip()
            terms = [term for term in query.split() if len(term) > 2] or [query]
            facts = [fact for fact in facts if any(term in self._searchable_text(fact) for term in terms)]
        return facts[: max(0, limit)]

    def deactivate_fact(self, memory_id: str) -> None:
        self.lifecycle.mark_historical(str(memory_id))

    def count_facts(self, *, active_only: bool = True) -> int:
        return len(self.search_facts(active_only=active_only, limit=100000))

    def recent_facts(self, *, limit: int = 10, active_only: bool = True) -> list[MemoryFact]:
        return self.search_facts(active_only=active_only, limit=limit)

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
        payload = {"title": title, "due_at": due_at_iso, "timezone": timezone_name, "notes": notes}
        fact = self.create_fact(
            kind="appointment", subject=title, payload=payload,
            source_text=source_text, confidence=1.0, active=True,
        )
        reminder = None
        if create_reminder:
            reminder = self.create_reminder(
                title=title, due_at=due_at_iso, message=reminder_message or f"Te recuerdo: {title}",
                kind="appointment", timezone_name=timezone_name,
                source_memory_id=fact.id, payload=payload,
            )
        return fact, reminder

    def get_recent_appointments(self, limit: int = 5) -> list[MemoryFact]:
        return self.search_facts(kind="appointment", active_only=True, limit=limit)

    def create_reminder(
        self,
        title: str,
        due_at: str,
        message: Optional[str] = None,
        kind: str = "generic",
        timezone_name: str = "Europe/Madrid",
        source_memory_id: Any = None,
        payload: Optional[dict] = None,
    ) -> Reminder:
        reminder_payload = dict(payload or {})
        if source_memory_id:
            reminder_payload["source_belief_id"] = str(source_memory_id)
        reminder_id = db_sqlite.create_reminder(
            title=title, due_at=due_at, message=message, kind=kind,
            timezone_name=timezone_name, source_memory_id=None, payload=reminder_payload,
        )
        reminder = self.get_reminder(reminder_id)
        if reminder is None:
            raise RuntimeError(f"Could not retrieve newly created reminder: {reminder_id}")
        return reminder

    def get_reminder(self, reminder_id: int) -> Optional[Reminder]:
        row = db_sqlite.get_reminder(reminder_id)
        return self._to_reminder(row) if row else None

    def list_due_reminders(self, limit: int = 20) -> list[Reminder]:
        return [self._to_reminder(row) for row in db_sqlite.list_due_reminders(limit=limit)]

    def list_pending_reminders(self, limit: int = 20) -> list[Reminder]:
        return [self._to_reminder(row) for row in db_sqlite.list_pending_reminders(limit=limit)]

    def mark_reminder_fired(self, reminder_id: int) -> None:
        db_sqlite.mark_reminder_fired(reminder_id)

    def mark_reminder_done(self, reminder_id: int) -> None:
        db_sqlite.mark_reminder_done(reminder_id)

    def cancel_reminder(self, reminder_id: int) -> None:
        db_sqlite.cancel_reminder(reminder_id)

    def log_internal_event(self, event_type: str, payload: Optional[dict] = None) -> int:
        return db_sqlite.log_internal_event(event_type=event_type, payload=payload)

    def _to_memory_fact(self, belief: Belief) -> MemoryFact:
        payload = belief.object_value if isinstance(belief.object_value, dict) else {
            "predicate": belief.predicate, "value": belief.object_value,
        }
        evidence = self.repository.evidence_for(belief.id)
        literal = json.loads(evidence[-1]["literal_span_json"] or "{}") if evidence else {}
        return MemoryFact(
            id=belief.id, kind=self._kind_for_namespace(belief.namespace), subject=belief.subject_ref,
            payload=payload, source_text=str(literal.get("text") or "") or None,
            confidence=belief.confidence,
            created_at=datetime.fromtimestamp(belief.created_at, tz=timezone.utc).isoformat(),
            updated_at=datetime.fromtimestamp(belief.last_confirmed_at, tz=timezone.utc).isoformat(),
            last_used_at=None,
            active=(belief.epistemic_status in {BeliefStatus.KNOWN, BeliefStatus.INFERRED, BeliefStatus.SUSPECTED}
                    and not bool(belief.superseded_by)),
        )

    @staticmethod
    def _kind_for_namespace(namespace: str) -> str:
        return {
            "memory.preference": "preference", "memory.owner_fact": "leo_fact",
            "memory.hebe_identity": "hebe_identity", "memory.project": "project_fact",
            "memory.stream": "stream_fact", "memory.habit": "habit",
            "memory.viewer": "viewer_fact", "memory.appointment": "appointment",
            "memory.fact": "fact", "memory.person": "person",
        }.get(namespace, namespace.removeprefix("memory."))

    @staticmethod
    def _searchable_text(fact: MemoryFact) -> str:
        return " ".join((
            str(fact.subject or ""), str(fact.source_text or ""),
            json.dumps(fact.payload or {}, ensure_ascii=False),
        )).casefold()

    @staticmethod
    def _to_reminder(row: dict[str, Any]) -> Reminder:
        return Reminder(
            id=int(row["id"]), kind=str(row["kind"]), title=str(row["title"]),
            message=row.get("message"), due_at=str(row["due_at"]),
            timezone=str(row.get("timezone") or "UTC"), status=str(row["status"]),
            source_memory_id=row.get("source_memory_id"), payload=row.get("payload"),
            created_at=str(row["created_at"]), fired_at=row.get("fired_at"),
        )
