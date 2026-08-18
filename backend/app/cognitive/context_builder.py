# backend/app/cognitive/context_builder.py
from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any

from app.cognitive.entity_resolver import EntityResolver
from app.cognitive.memory_store import MemoryStore, MemoryFact, Reminder
from app.cognitive.scheduler import InternalEvent
from app.core.state import HebeState
from app.cognitive.cognitive_decision import CognitiveDecision
from app.cognitive.game_guidance import GameRunState
from app.cognitive.game_guidance import GameGuidanceDecision
from app.cognitive.input_interpretation import InputInterpretation, InputSpeechAct


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

    # small_talk | banter | planning_request | memory_query | direct_question | command | stream_event | unknown
    message_type: str = "unknown"
    inject_memory: bool = True
    context_policy: dict[str, Any] = field(default_factory=dict)
    resolved_entities: list[dict[str, Any]] = field(default_factory=list)
    response_frame: dict[str, Any] = field(default_factory=dict)
    source: str = "ui"
    authority: str = "owner"
    addressed_to_hebe: bool = True
    message_id: str = ""
    cognitive_decision: CognitiveDecision | None = None
    game_guidance_decision: GameGuidanceDecision | None = None
    firewall_decision: str = ""
    stream_is_live: bool = False
    route_hints: list[str] = field(default_factory=list)
    input_interpretation: InputInterpretation | None = None


class ContextBuilder:
    """
    Construye el contexto cognitivo para cada interacción.

    NO decide nada.
    NO ejecuta nada.
    SOLO recopila información relevante.
    """

    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
        self.entity_resolver = EntityResolver()

    # =========================
    # Entry point
    # =========================

    def build(
        self,
        state: HebeState,
        input_text: Optional[str] = None,
        internal_event: Optional[InternalEvent] = None,
        source: str = "ui",
        authority: str = "owner",
        addressed_to_hebe: bool = True,
        message_id: str = "",
        input_interpretation: InputInterpretation | None = None,
    ) -> BuiltContext:
        """
        Construye contexto tanto para:
        - input del usuario (JARVIS)
        - eventos internos (reminder_due, twitch_chat_react, …)
        """

        message_type = self._classify_message_type(
            input_text=input_text,
            internal_event=internal_event,
            input_interpretation=input_interpretation,
        )
        context_policy = self._build_context_policy(message_type)
        inject_memory = context_policy["memory"] in {"full", "relevant"}
        source_context = (
            "stream"
            if internal_event is not None and internal_event.event_type.startswith("twitch_")
            else "private"
        )
        resolved_entities = [
            resolution.to_dict()
            for resolution in self.entity_resolver.resolve(
                input_text,
                source_context=source_context,
            )
        ]

        relevant_facts = (
            self._get_relevant_facts(input_text)
            if context_policy["memory"] in {"full", "relevant"}
            else []
        )
        relevant_facts = self._filter_facts_by_entity(
            relevant_facts,
            resolved_entities=resolved_entities,
        )
        recent_appointments = self.memory_store.get_recent_appointments(limit=3)
        pending_reminders = self.memory_store.list_pending_reminders(limit=5)
        state_snapshot = self._build_state_snapshot(state)

        # RAG retrieval — solo donde aporta valor, para no añadir latencia/coste
        # en eventos donde no se usa (reminders, subs, raids, follows…).
        relevant_chunks: list[dict] = []
        conversation_history: list[dict] = []
        if internal_event is None and input_text:
            # Path JARVIS: búsqueda semántica + historial conversacional.
            if context_policy["memory"] in {"full", "relevant"}:
                relevant_chunks = self._retrieve_memory_for_jarvis(
                    input_text,
                    include_recent_streams=bool(context_policy["schedule"]),
                )
                relevant_chunks = self._filter_chunks_by_entity(
                    relevant_chunks,
                    resolved_entities=resolved_entities,
                )
            from app.services.db_sqlite import get_recent_chat_turns
            conversation_history = get_recent_chat_turns(
                source="ui",
                limit=int(context_policy["history_turns"]),
            )
            # UI turns are logged before cognitive_flow builds context. Drop the
            # current user turn if it is already in chat_log, otherwise the model
            # sees Leo's latest message twice.
            if conversation_history:
                last = conversation_history[-1]
                if (
                    last.get("role") == "user"
                    and self._normalize_for_compare(last.get("content")) == self._normalize_for_compare(input_text)
                ):
                    conversation_history = conversation_history[:-1]
        elif (
            internal_event is not None
            and internal_event.event_type == "twitch_chat_react"
        ):
            # Path Twitch chat: lookup estructurado del viewer (sin embeddings).
            # Twitch sigue single-turn por diseño — sin historial conversacional.
            user_login = str((internal_event.payload or {}).get("user_login") or "")
            relevant_chunks = self._retrieve_memory_for_twitch(user_login)

        print(
            f"[HEBE][CONTEXT] message_type={message_type} inject_memory={inject_memory}",
            flush=True,
        )
        print(
            "[HEBE][CONTEXT_POLICY] "
            f"type={message_type} "
            f"memory={context_policy['memory']} "
            f"schedule={str(context_policy['schedule']).lower()} "
            f"history_turns={context_policy['history_turns']}",
            flush=True,
        )
        for entity in resolved_entities:
            print(
                "[HEBE][ENTITY] "
                f"mention={entity.get('mention')!r} "
                f"candidates={list(entity.get('candidates') or [])!r} "
                f"selected={entity.get('selected')!r} "
                f"reason={entity.get('reason')!r}",
                flush=True,
            )

        return BuiltContext(
            input_text=input_text,
            internal_event=internal_event,
            relevant_facts=relevant_facts,
            recent_appointments=recent_appointments,
            pending_reminders=pending_reminders,
            state_snapshot=state_snapshot,
            relevant_chunks=relevant_chunks,
            conversation_history=conversation_history,
            message_type=message_type,
            inject_memory=inject_memory,
            context_policy=context_policy,
            resolved_entities=resolved_entities,
            source=source,
            authority=authority,
            addressed_to_hebe=addressed_to_hebe,
            message_id=message_id,
            input_interpretation=input_interpretation,
        )

    # =========================
    # Message classification
    # =========================

    def _classify_message_type(
        self,
        *,
        input_text: Optional[str],
        internal_event: Optional[InternalEvent],
        input_interpretation: InputInterpretation | None = None,
    ) -> str:
        if internal_event is not None:
            if internal_event.event_type.startswith("twitch_"):
                return "stream_event"
            return "unknown"

        if input_interpretation is not None:
            if input_interpretation.speech_act == InputSpeechAct.OWNER_FEEDBACK:
                return "owner_feedback"
            if input_interpretation.speech_act == InputSpeechAct.OWNER_COMMAND:
                return "command"
            if input_interpretation.speech_act == InputSpeechAct.OWNER_ANSWER_FOLLOWUP:
                return "followup"

        normalized = self._normalize_for_compare(input_text)
        if not normalized:
            return "unknown"

        memory_markers = (
            "que sabes de mi",
            "que sabes sobre mi",
            "que recuerdas",
            "recuerdas de mi",
            "de que hemos hablado",
            "que hemos hablado",
            "resumeme",
            "resume",
            "recap",
            "que dije antes",
            "que te dije antes",
            "hemos dicho",
            "what do you remember",
            "what do you know about me",
            "what have we talked about",
            "summarize what we said",
            "what did i say before",
        )
        if any(marker in normalized for marker in memory_markers):
            return "memory_query"

        if normalized.startswith(("recuerdas que", "remember that", "do you remember")):
            return "memory_query"

        planning_markers = (
            "que toca hoy",
            "que deberia jugar",
            "seguimos con",
            "que tenia planeado",
            "que hago hoy en directo",
            "que hago en directo",
            "plan de stream",
            "stream plan",
            "what should i play",
            "what is planned for stream",
            "what are we doing on stream",
        )
        if any(marker in normalized for marker in planning_markers):
            return "planning_request"

        command_markers = (
            "abre ",
            "open ",
            "guarda ",
            "recuerda que",
            "remember ",
            "apunta ",
            "crea ",
            "pon ",
            "avisame",
            "mandame",
            "envia ",
        )
        if normalized.startswith(command_markers):
            return "command"

        task_markers = (
            "abre ",
            "open ",
            "guarda ",
            "recuerda que",
            "remember ",
            "apunta ",
            "crea ",
            "pon ",
            "avisame",
            "avísame",
            "mandame",
            "mándame",
            "envia ",
            "envía ",
        )
        if normalized.startswith(task_markers):
            return "command"

        small_talk_exact = {
            "hola",
            "hola hebe",
            "buenas",
            "buenas hebe",
            "hey",
            "hey hebe",
            "ey",
            "ey hebe",
            "hebe",
            "que tal",
            "que tal hebe",
            "como vas",
            "como vas hebe",
            "como lo llevas",
            "como lo llevas hebe",
            "como estas",
            "como estas hebe",
            "estas ahi",
            "estas ahi hebe",
            "sigues ahi",
            "hello",
            "hello hebe",
            "hi",
            "hi hebe",
            "how are you",
            "how are you hebe",
            "how is it going",
            "you there",
        }
        if normalized in small_talk_exact:
            return "small_talk"

        small_talk_patterns = (
            "como lo llevas",
            "como vas",
            "que tal",
            "como estas",
            "estas ahi",
            "sigues ahi",
            "how are you",
            "how's it going",
        )
        if len(normalized.split()) <= 7 and any(pattern in normalized for pattern in small_talk_patterns):
            return "small_talk"

        banter_markers = (
            "jaja",
            "jajaja",
            "lol",
            "modo zombie",
            "estoy muerto",
            "estoy muerta",
            "vaya dia",
            "menudo dia",
            "zombie",
            "reventado",
            "reventada",
            "hecho polvo",
            "hecha polvo",
            "no puedo con mi alma",
        )
        if any(marker in normalized for marker in banter_markers):
            return "banter"

        raw = input_text or ""
        if "?" in raw or normalized.startswith(
            ("que ", "como ", "cuando ", "donde ", "por que ", "why ", "what ", "how ")
        ):
            return "direct_question"

        return "unknown"

    def _build_context_policy(self, message_type: str) -> dict[str, Any]:
        if message_type in {"small_talk", "banter"}:
            return {
                "memory": "limited",
                "schedule": False,
                "history_turns": 2,
                "max_sentences": 2,
            }

        if message_type in {"planning_request", "memory_query"}:
            return {
                "memory": "full",
                "schedule": True,
                "history_turns": 10,
                "max_sentences": None,
            }

        if message_type in {"direct_question", "command"}:
            return {
                "memory": "relevant",
                "schedule": False,
                "history_turns": 6 if message_type == "direct_question" else 4,
                "max_sentences": None,
            }

        return {
            "memory": "relevant",
            "schedule": False,
            "history_turns": 4,
            "max_sentences": None,
        }

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
        )

        low = input_text.lower()
        if any(token in low for token in ("hablar", "idioma", "femenin", "deberias", "deberías", "speak", "language")):
            facts.extend(
                self.memory_store.search_facts(
                    kind="hebe_identity",
                    active_only=True,
                    limit=3,
                )
            )
            facts.extend(
                self.memory_store.search_facts(
                    kind="preference",
                    active_only=True,
                    limit=3,
                )
            )

        seen: set[str] = set()
        deduped: list[MemoryFact] = []
        for fact in facts:
            if fact.id in seen:
                continue
            seen.add(fact.id)
            deduped.append(fact)

        return deduped[:limit]

    # =========================
    # Entity-aware filtering
    # =========================

    def _filter_facts_by_entity(
        self,
        facts: list[MemoryFact],
        *,
        resolved_entities: list[dict[str, Any]],
    ) -> list[MemoryFact]:
        if not facts or not resolved_entities:
            return facts

        resolution = resolved_entities[0]
        selected = str(resolution.get("selected") or "")
        candidates = set(resolution.get("candidates") or [])
        broad = bool(resolution.get("broad_query"))

        primary: list[MemoryFact] = []
        secondary: list[MemoryFact] = []
        unknown: list[MemoryFact] = []
        for fact in facts:
            entity_id = self._entity_id_for_fact(fact)
            if not entity_id:
                unknown.append(fact)
            elif entity_id == selected:
                primary.append(fact)
            elif broad and entity_id in candidates:
                secondary.append(fact)

        return primary + secondary + unknown if primary or secondary else unknown

    def _filter_chunks_by_entity(
        self,
        chunks: list[dict],
        *,
        resolved_entities: list[dict[str, Any]],
    ) -> list[dict]:
        if not chunks or not resolved_entities:
            return chunks

        resolution = resolved_entities[0]
        selected = str(resolution.get("selected") or "")
        candidates = set(resolution.get("candidates") or [])
        broad = bool(resolution.get("broad_query"))

        primary: list[dict] = []
        secondary: list[dict] = []
        unknown: list[dict] = []
        for chunk in chunks:
            entity_id = self._entity_id_for_chunk(chunk)
            if not entity_id:
                unknown.append(chunk)
            elif entity_id == selected:
                primary.append(chunk)
            elif broad and entity_id in candidates:
                secondary.append(chunk)

        return primary + secondary + unknown if primary or secondary else unknown

    def _entity_id_for_fact(self, fact: MemoryFact) -> str | None:
        payload = fact.payload or {}
        entity_id = payload.get("entity_id")
        if isinstance(entity_id, str) and entity_id.strip():
            return entity_id.strip()
        return self._infer_entity_id(
            " ".join(
                str(part or "")
                for part in (
                    fact.subject,
                    fact.source_text,
                    payload.get("text"),
                    payload.get("subject"),
                )
            )
        )

    def _entity_id_for_chunk(self, chunk: dict) -> str | None:
        tags = chunk.get("tags") or {}
        entity_id = None
        if isinstance(tags, dict):
            entity_id = tags.get("entity_id")
        if isinstance(entity_id, str) and entity_id.strip():
            return entity_id.strip()
        return self._infer_entity_id(
            " ".join(
                str(part or "")
                for part in (
                    chunk.get("subject"),
                    chunk.get("text"),
                    chunk.get("kind"),
                )
            )
        )

    def _infer_entity_id(self, text: str) -> str | None:
        normalized = self._normalize_for_compare(text)
        if not normalized:
            return None

        if "jotunbot" in normalized or "jotun bot" in normalized:
            return "jotun_bot"
        if "jotun" in normalized and any(
            marker in normalized
            for marker in ("bot", "comando", "comandos", "follow", "twitch", "chat")
        ):
            return "jotun_bot"
        if "jotun" in normalized and any(
            marker in normalized
            for marker in ("perro", "dog", "mascota", "fisico", "físico")
        ):
            return "jotun_dog"
        if "hebe" in normalized:
            return "hebe_ai"
        if re.search(r"(^|\s)leo($|\s)", normalized):
            return "leo"
        return None

    # =========================
    # RAG chunk retrieval
    # =========================

    def _retrieve_memory_for_jarvis(
        self,
        user_message: str,
        *,
        include_recent_streams: bool = True,
    ) -> list[dict]:
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
            recent_streams = (
                get_recent_chunks(kind="stream_summary", limit=2)
                if include_recent_streams
                else []
            )

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
        """Retrieve viewer facts from canonical beliefs without embeddings."""
        if not user_login:
            return []
        try:
            facts = self.memory_store.search_facts(
                query_text=user_login,
                kind="viewer_fact",
                active_only=True,
                limit=3,
            )
            result: list[dict] = []
            for fact in facts:
                text = fact.source_text or str(fact.payload or "")
                if not text:
                    continue
                result.append(
                    {
                        "id": fact.id,
                        "text": text,
                        "kind": "viewer_fact",
                        "subject": fact.subject,
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
        game_run_state = GameRunState.from_value(getattr(state, "game_run_state", None))
        current_game_context: dict[str, Any] = {}
        if game_run_state.game:
            current_game_context = {
                "game": game_run_state.game,
                "source": game_run_state.provenance or "game_run",
                "confidence": max(0.65, float(game_run_state.confidence or 0.0)),
            }
        if not game_run_state.game and stream is not None:
            live = dict(getattr(stream, "live_session_context", None) or {})
            title_game = str(getattr(stream, "current_stream_title", None) or "").split("|", 1)[0].strip()
            candidates = (
                (live.get("current_game") or live.get("game"), "stream_session", 0.92),
                (getattr(stream, "current_game", None), "stream_session", 0.9),
                (getattr(stream, "current_category", None), "twitch_category", 0.88),
                (title_game, "stream_title", 0.62),
            )
            selected = next(
                ((str(value).strip(), source, confidence) for value, source, confidence in candidates if str(value or "").strip()),
                ("", "", 0.0),
            )
            game_run_state.game = selected[0]
            game_run_state.current_location = str(getattr(stream, "current_location", None) or "")
            game_run_state.current_objective = str(getattr(stream, "current_objective", None) or "")
            game_run_state.last_confirmed_progress = str(
                (getattr(stream, "recent_progress_markers", None) or [""])[-1] or ""
            )
            game_run_state.spoiler_policy = str(getattr(stream, "spoiler_policy", None) or "spoiler_safe_hints")
            game_run_state.provenance = selected[1] or "current_live_session"
            game_run_state.confidence = selected[2] if game_run_state.game else 0.0
            if game_run_state.game:
                current_game_context = {
                    "game": game_run_state.game,
                    "source": game_run_state.provenance,
                    "confidence": game_run_state.confidence,
                }

        return {
            "now_iso": __import__("datetime").datetime.now().astimezone().isoformat(),
            "mode": getattr(state, "mode", None),
            "is_processing": getattr(state, "is_processing", None),
            "last_intent": getattr(state, "last_intent", None),
            "current_task": getattr(state, "current_task", None),
            "stream_enabled": getattr(stream, "enabled", False) if stream else False,
            "stream_armed": getattr(stream, "armed", False) if stream else False,
            "game_run_state": game_run_state.to_dict(),
            "current_game_context": current_game_context,
        }

    def _normalize_for_compare(self, text: Optional[str]) -> str:
        raw = (text or "").strip().lower()
        without_accents = "".join(
            ch for ch in unicodedata.normalize("NFKD", raw)
            if not unicodedata.combining(ch)
        )
        cleaned = re.sub(r"[^a-z0-9ñ\s']", " ", without_accents)
        return " ".join(cleaned.split())
