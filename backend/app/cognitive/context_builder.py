# backend/app/cognitive/context_builder.py
from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, field
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

    # Memoria relevante (estructurada, MemoryFact)
    relevant_facts: list[MemoryFact]
    recent_appointments: list[MemoryFact]

    # Estado de reminders
    pending_reminders: list[Reminder]

    # Estado del sistema
    state_snapshot: dict[str, Any]

    # Chunks RAG (texto libre + embeddings). Campo hermano de relevant_facts;
    # no mezclar — facts son dataclasses estructurados, chunks son dicts con score.
    # Poblado por _retrieve_memory_for_jarvis (JARVIS) o _retrieve_memory_for_twitch
    # (Twitch chat react). Vacío en todos los demás eventos.
    relevant_chunks: list[dict] = field(default_factory=list)

    # Historial conversacional corto (short-term memory). Solo path JARVIS.
    # Formato OpenAI: [{"role": "user"|"assistant", "content": str}, ...]
    # ordenado cronológicamente ASC (antiguo → reciente).
    conversation_history: list[dict] = field(default_factory=list)


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
        - input del usuario (JARVIS)
        - eventos internos (reminder_due, twitch_chat_react, …)
        """

        relevant_facts = self._get_relevant_facts(input_text)
        recent_appointments = self.memory_store.get_recent_appointments(limit=3)
        pending_reminders = self.memory_store.list_pending_reminders(limit=5)
        state_snapshot = self._build_state_snapshot(state)

        # RAG retrieval — solo donde aporta valor, para no añadir latencia/coste
        # en eventos donde no se usa (reminders, subs, raids, follows…).
        relevant_chunks: list[dict] = []
        conversation_history: list[dict] = []
        if internal_event is None and input_text:
            # Path JARVIS: búsqueda semántica + historial conversacional.
            relevant_chunks = self._retrieve_memory_for_jarvis(input_text)
            from app.services.db_sqlite import get_recent_chat_turns
            conversation_history = get_recent_chat_turns(source="ui", limit=10)
        elif (
            internal_event is not None
            and internal_event.event_type == "twitch_chat_react"
        ):
            # Path Twitch chat: lookup estructurado del viewer (sin embeddings).
            # Twitch sigue single-turn por diseño — sin historial conversacional.
            user_login = str((internal_event.payload or {}).get("user_login") or "")
            relevant_chunks = self._retrieve_memory_for_twitch(user_login)

        return BuiltContext(
            input_text=input_text,
            internal_event=internal_event,
            relevant_facts=relevant_facts,
            recent_appointments=recent_appointments,
            pending_reminders=pending_reminders,
            state_snapshot=state_snapshot,
            relevant_chunks=relevant_chunks,
            conversation_history=conversation_history,
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
    # RAG chunk retrieval
    # =========================

    def _retrieve_memory_for_jarvis(self, user_message: str) -> list[dict]:
        """
        Recall semántico + resúmenes recientes de stream para JARVIS.

        Combina:
        - Búsqueda vectorial top-4 sobre la query del usuario (min_similarity=0.3
          filtra correlación espuria; un "hola" no trae todos los chunks).
        - Los 2 stream_summary más recientes (sin filtro semántico, siempre útiles).

        Deduplicamos por id para que un chunk que aparezca en ambas listas
        no se inyecte dos veces en el prompt.

        Retorna lista vacía en cualquier error (sentence-transformers no instalado,
        tabla vacía, conexión fallida…). El synthesizer lo trata como ausencia de
        memoria, no como fallo.
        """
        if not user_message:
            return []
        try:
            from app.cognitive.memory.memory_store import search_chunks, get_recent_chunks

            chunks = search_chunks(
                query=user_message,
                top_k=4,
                min_similarity=0.3,
            )
            recent_streams = get_recent_chunks(kind="stream_summary", limit=2)

            seen: set[int] = set()
            combined: list[dict] = []
            for item in chunks + recent_streams:
                item_id = item.get("id")
                if item_id in seen:
                    continue
                seen.add(item_id)
                combined.append(item)
            return combined
        except Exception as exc:
            print(f"[HEBE][MEMORY][JARVIS] retrieval error: {exc!r}", flush=True)
            return []

    def _retrieve_memory_for_twitch(self, user_login: str) -> list[dict]:
        """
        Lookup estructurado del viewer activo. Sin embeddings → coste constante
        (no añade latencia por embedding en cada mensaje de chat).

        Usa LIKE sobre el campo subject para encontrar viewer_facts del usuario.
        Devuelve los chunks en el mismo shape que _retrieve_memory_for_jarvis
        (dict con clave 'text') para que el synthesizer no los distinga.
        """
        if not user_login:
            return []
        try:
            from app.services.db_sqlite import search_memory_facts

            rows = search_memory_facts(
                query_text=user_login,
                kind="viewer_fact",
                active_only=True,
                limit=3,
            )
            result: list[dict] = []
            for r in rows:
                # source_text es la representación legible; payload como fallback.
                text = r.get("source_text") or str(r.get("payload") or "")
                if not text:
                    continue
                result.append(
                    {
                        "id": r.get("id"),
                        "text": text,
                        "kind": "viewer_fact",
                        "subject": r.get("subject"),
                    }
                )
            return result
        except Exception as exc:
            print(f"[HEBE][MEMORY][TWITCH] retrieval error: {exc!r}", flush=True)
            return []

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