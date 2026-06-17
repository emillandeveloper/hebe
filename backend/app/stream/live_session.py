from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.services import db_sqlite


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _loads(value: str | None, fallback: Any = None) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _norm(text: str | None) -> str:
    value = str(text or "").casefold()
    value = value.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    value = value.replace("ü", "u").replace("ñ", "n")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9_ ]+", " ", value)
    return " ".join(value.split())


def init_live_session_schema(conn: sqlite3.Connection | None = None) -> None:
    own_conn = conn is None
    conn = conn or db_sqlite.get_db_connection()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS live_session_state (
            session_id TEXT PRIMARY KEY,
            stream_session_id INTEGER,
            stream_status TEXT,
            current_game TEXT,
            current_category TEXT,
            current_title TEXT,
            language_mode TEXT,
            spoiler_policy TEXT,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS live_session_timeline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            stream_session_id INTEGER,
            event_uid TEXT NOT NULL UNIQUE,
            event_type TEXT NOT NULL,
            event_ts TEXT NOT NULL,
            source TEXT,
            raw_text TEXT,
            normalized_text TEXT,
            speaker TEXT,
            target TEXT,
            topic TEXT,
            category TEXT,
            confidence REAL NOT NULL DEFAULT 0.0,
            provenance TEXT,
            related_event_id INTEGER,
            output_target TEXT,
            index_for_rag INTEGER NOT NULL DEFAULT 0,
            payload_json TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS live_session_spontaneity_anchors (
            anchor_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            anchor_type TEXT,
            source_event_ids_json TEXT,
            topic TEXT,
            category TEXT,
            payload_json TEXT,
            created_at TEXT NOT NULL,
            consumed_at TEXT,
            invalidated_at TEXT,
            cooldown_key TEXT
        );

        CREATE TABLE IF NOT EXISTS live_session_rolling_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            stream_session_id INTEGER,
            summary_text TEXT NOT NULL,
            event_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            tags_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_live_timeline_session_ts ON live_session_timeline(session_id, event_ts);
        CREATE INDEX IF NOT EXISTS idx_live_timeline_type ON live_session_timeline(event_type);
        CREATE INDEX IF NOT EXISTS idx_live_timeline_topic ON live_session_timeline(topic);
        CREATE INDEX IF NOT EXISTS idx_live_summaries_session ON live_session_rolling_summaries(session_id, created_at);
        """
    )
    if own_conn:
        conn.commit()
        conn.close()


@dataclass
class ParticipantState:
    username: str
    display_name: str = ""
    first_seen_at: str = ""
    last_seen_at: str = ""
    last_message: str = ""
    recent_topics: list[str] = field(default_factory=list)
    interaction_count: int = 0
    last_mentioned_by_leo: str | None = None
    last_replied_by_hebe: str | None = None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["likely_still_around"] = True
        return data


@dataclass
class LastHebeUtterance:
    event_id: int | None = None
    text: str = ""
    source: str = "hebe"
    output_target: list[str] = field(default_factory=list)
    anchor_id: str | None = None
    topic: str | None = None
    created_at: str = ""
    expires_at: float = 0.0
    expects_possible_reply_from_leo: bool = False


@dataclass
class LiveSessionState:
    session_id: str
    stream_status: str = "unknown"
    stream_session_id: int | None = None
    current_game: str | None = None
    current_category: str | None = None
    current_title: str | None = None
    language_mode: str | None = None
    spoiler_policy: str | None = None
    current_phase: str | None = None
    current_activity: str | None = None
    combat_state: bool | None = None
    current_activity_provenance: str | None = None
    blocked_comment_categories: list[str] = field(default_factory=list)
    current_location: str | None = None
    current_objective: str | None = None
    recent_progress_markers: list[str] = field(default_factory=list)
    latest_boss_state: str | None = None
    latest_failure_or_success: str | None = None
    latest_resource_state: str | None = None
    latest_strategy_topic: str | None = None
    latest_confusion: str | None = None
    latest_correction_from_leo: str | None = None
    current_chat_topic: str | None = None
    recent_chatters: list[dict] = field(default_factory=list)
    active_chatters: list[dict] = field(default_factory=list)
    possible_lurkers_from_recent_chat: list[dict] = field(default_factory=list)
    last_hebe_utterance: dict | None = None
    last_hebe_anchor: str | None = None
    last_hebe_output_target: list[str] = field(default_factory=list)
    last_direct_interaction_with_leo: dict | None = None
    last_spontaneous_message: dict | None = None
    consumed_spontaneity_anchors: list[str] = field(default_factory=list)
    invalidated_anchors: list[str] = field(default_factory=list)
    facts_provenance: dict[str, dict] = field(default_factory=dict)
    meaningful_event_count: int = 0
    session_context_update_count: int = 0
    latest_rolling_summary_time: str | None = None
    last_retrieved_context_used: dict | None = None
    last_updated_at: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class LiveSessionBrain:
    """Fast current-session brain backed by a persistent event timeline."""

    def __init__(self, stream: Any | None = None, *, session_id: str | None = None):
        init_live_session_schema()
        self.state = LiveSessionState(session_id=session_id or str(uuid.uuid4()))
        self.participants: dict[str, ParticipantState] = {}
        self._last_summary_ts = time.time()
        self._last_summary_event_count = 0
        self.sync_stream_metadata(stream)
        self._persist_state()

    def sync_stream_metadata(self, stream: Any | None) -> None:
        if stream is None:
            return
        self.state.stream_session_id = getattr(stream, "active_stream_session_id", None)
        self.state.stream_status = "live" if getattr(stream, "is_live", False) else "offline"
        if not getattr(stream, "live_status_known", False):
            self.state.stream_status = "unknown"
        self._set("current_game", getattr(stream, "current_game", None), source="twitch_title")
        self._set("current_category", getattr(stream, "current_category", None), source="twitch_title")
        self._set("current_title", getattr(stream, "current_stream_title", None), source="twitch_title")
        self._set("language_mode", getattr(stream, "language_mode", None), source="twitch_title")
        self._set("spoiler_policy", getattr(stream, "spoiler_policy", None), source="twitch_title")
        self._set("current_phase", getattr(stream, "current_run_phase", None), source=getattr(stream, "run_context_source", None))
        self._set("current_activity", getattr(stream, "current_activity", None), source=getattr(stream, "current_game_activity_provenance", None))
        self.state.combat_state = getattr(stream, "combat_state", None)
        self.state.current_activity_provenance = getattr(stream, "current_game_activity_provenance", None)
        self.state.blocked_comment_categories = list(getattr(stream, "blocked_comment_categories", []) or [])
        self._set("current_location", getattr(stream, "current_run_location", None), source=getattr(stream, "run_context_source", None))
        self._set("current_objective", getattr(stream, "current_run_objective", None), source=getattr(stream, "run_context_source", None))
        markers = list(getattr(stream, "completed_run_markers", []) or [])[-8:]
        if markers:
            self.state.recent_progress_markers = markers
        self._touch()

    def record_event(
        self,
        event_type: str,
        *,
        source: str = "",
        raw_text: str = "",
        normalized_text: str = "",
        speaker: str = "",
        target: str = "",
        topic: str = "",
        category: str = "",
        confidence: float = 0.0,
        provenance: str = "",
        related_event_id: int | None = None,
        output_target: list[str] | str | None = None,
        index_for_rag: bool = False,
        payload: dict | None = None,
    ) -> int:
        now = _now_iso()
        targets = output_target if isinstance(output_target, str) else "+".join(output_target or [])
        payload = payload or {}
        conn = db_sqlite.get_db_connection()
        cur = conn.execute(
            """
            INSERT INTO live_session_timeline (
                session_id, stream_session_id, event_uid, event_type, event_ts,
                source, raw_text, normalized_text, speaker, target, topic, category,
                confidence, provenance, related_event_id, output_target,
                index_for_rag, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.state.session_id,
                self.state.stream_session_id,
                str(uuid.uuid4()),
                event_type,
                now,
                source,
                raw_text,
                normalized_text,
                speaker,
                target,
                topic,
                category,
                float(confidence or 0.0),
                provenance,
                related_event_id,
                targets,
                1 if index_for_rag else 0,
                _json(payload),
                now,
            ),
        )
        conn.commit()
        event_id = int(cur.lastrowid)
        conn.close()
        self.state.meaningful_event_count += 1
        self._touch()
        self._maybe_roll_summary()
        return event_id

    def observe_stream_metadata(self, stream: Any | None, *, source: str = "stream_metadata_update") -> None:
        self.sync_stream_metadata(stream)
        self.record_event(
            "stream_metadata_update",
            source=source,
            topic="stream_meta",
            category="stream_metadata_update",
            confidence=0.9,
            provenance="twitch_title",
            index_for_rag=True,
            payload=self.metadata_snapshot(),
        )

    def observe_chat_message(self, username: str, display_name: str, text: str, *, topic: str, mention: bool) -> None:
        user = _norm(username) or str(username or "").strip().lower()
        now = _now_iso()
        participant = self.participants.get(user)
        if participant is None:
            participant = ParticipantState(username=user, display_name=display_name or username, first_seen_at=now)
            self.participants[user] = participant
        participant.display_name = display_name or participant.display_name or username
        participant.last_seen_at = now
        participant.last_message = str(text or "").strip()[:220]
        if topic and topic not in participant.recent_topics:
            participant.recent_topics.append(topic)
            participant.recent_topics = participant.recent_topics[-6:]
        participant.interaction_count += 1
        self.state.current_chat_topic = topic or self.state.current_chat_topic
        self._refresh_participants()
        self.record_event(
            "twitch_chat_mention" if mention else "twitch_chat_message",
            source="twitch_chat",
            raw_text=text,
            normalized_text=_norm(text),
            speaker=user,
            target="hebe" if mention else "",
            topic=topic,
            category=topic,
            confidence=0.72,
            provenance="twitch_chat",
            index_for_rag=bool(topic and topic != "general_chat"),
        )

    def observe_leo_stt(
        self,
        raw_text: str,
        normalized_text: str,
        *,
        addressed_to_hebe: bool,
        voice_event_type: str,
        topic: str = "",
        confidence: float = 0.72,
    ) -> int:
        event_type = "leo_direct_to_hebe" if addressed_to_hebe else "leo_stt"
        if self.is_possible_reply_to_hebe(normalized_text) and not addressed_to_hebe:
            event_type = "leo_reply_to_hebe"
        category = topic or self.classify_topic(normalized_text, voice_event_type=voice_event_type)
        if category == "correction_to_hebe":
            event_type = "correction"
            event_id = self.apply_correction(raw_text, normalized_text)
            self.state.last_direct_interaction_with_leo = {
                "event_type": event_type,
                "text": raw_text,
                "topic": category,
                "at": _now_iso(),
            }
            return event_id
        elif event_type == "leo_reply_to_hebe" and self._looks_like_correction(normalized_text):
            event_type = "correction"
            event_id = self.apply_correction(raw_text, normalized_text)
            self.state.last_direct_interaction_with_leo = {
                "event_type": event_type,
                "text": raw_text,
                "topic": category,
                "at": _now_iso(),
            }
            return event_id
        self.state.last_direct_interaction_with_leo = {
            "event_type": event_type,
            "text": raw_text,
            "topic": category,
            "at": _now_iso(),
        }
        return self.record_event(
            event_type,
            source="leo_stt",
            raw_text=raw_text,
            normalized_text=normalized_text,
            speaker="leo",
            target="hebe" if addressed_to_hebe or event_type in {"leo_reply_to_hebe", "correction"} else "",
            topic=category,
            category=category,
            confidence=confidence,
            provenance="leo_stt",
            index_for_rag=event_type != "leo_stt" or category not in {"casual_comment", "unknown"},
        )

    def update_from_voice_relevance(self, text: str, event_type: str, relevance: Any | None, *, facts: list[dict] | None = None) -> None:
        facts = facts or list(getattr(relevance, "facts", []) or [])
        category = str(getattr(relevance, "category", "") or event_type or "")
        if event_type in {"completed_marker", "progress_update"}:
            marker = str(text or "").strip()
            if marker and marker not in self.state.recent_progress_markers:
                self.state.recent_progress_markers.append(marker)
                self.state.recent_progress_markers = self.state.recent_progress_markers[-8:]
            if event_type == "progress_update":
                self._set("current_phase", text, source="leo_stt", confidence=0.8)
        if event_type == "gameplay_failure":
            self._set("latest_failure_or_success", text, source="leo_stt", confidence=0.82)
        if event_type == "victory":
            self._set("latest_failure_or_success", text, source="leo_stt", confidence=0.82)
        if event_type == "confusion/lost":
            self._set("latest_confusion", text, source="leo_stt", confidence=0.78)
        if event_type == "objective_update":
            self._set("current_objective", text, source="leo_stt", confidence=0.8)
        if event_type == "location_update":
            self._set("current_location", text, source="leo_stt", confidence=0.78)
        for fact in facts:
            self._apply_fact(fact)
        if category and category != "none":
            self.record_event(
                "session_context_update",
                source="leo_stt",
                raw_text=text,
                normalized_text=_norm(text),
                speaker="leo",
                topic=category,
                category=category,
                confidence=float(getattr(relevance, "confidence", 0.0) or 0.0),
                provenance="leo_stt",
                index_for_rag=True,
                payload={"facts": facts},
            )
            self.state.session_context_update_count += 1
        self._touch()

    def observe_hebe_utterance(
        self,
        text: str,
        *,
        output_target: list[str] | str,
        input_type: str = "",
        anchor_id: str | None = None,
        topic: str | None = None,
        expects_possible_reply_from_leo: bool = True,
    ) -> int:
        targets = output_target if isinstance(output_target, list) else str(output_target or "").split("+")
        event_type = "hebe_twitch_message" if "twitch_chat" in targets else "hebe_tts_message" if any("tts" in t for t in targets) else "hebe_ui_message"
        event_id = self.record_event(
            event_type,
            source="hebe",
            raw_text=text,
            normalized_text=_norm(text),
            speaker="hebe",
            target="twitch_chat" if "twitch_chat" in targets else "leo",
            topic=topic or input_type,
            category=input_type or topic or "hebe_reply",
            confidence=1.0,
            provenance="hebe_output",
            output_target=targets,
            index_for_rag=True,
            payload={"anchor_id": anchor_id},
        )
        now = time.time()
        utterance = LastHebeUtterance(
            event_id=event_id,
            text=str(text or "").strip(),
            output_target=[t for t in targets if t],
            anchor_id=anchor_id,
            topic=topic or input_type,
            created_at=_now_iso(),
            expires_at=now + 90,
            expects_possible_reply_from_leo=expects_possible_reply_from_leo,
        )
        self.state.last_hebe_utterance = asdict(utterance)
        self.state.last_hebe_anchor = anchor_id
        self.state.last_hebe_output_target = utterance.output_target
        if input_type == "spontaneity":
            self.state.last_spontaneous_message = asdict(utterance)
        self._touch()
        return event_id

    def create_spontaneity_anchor(self, *, anchor_id: str | None = None, anchor_type: str = "", topic: str = "", source_event_ids: list[int] | None = None, payload: dict | None = None) -> str:
        anchor_id = anchor_id or str(uuid.uuid4())
        now = _now_iso()
        conn = db_sqlite.get_db_connection()
        conn.execute(
            """
            INSERT OR IGNORE INTO live_session_spontaneity_anchors (
                anchor_id, session_id, anchor_type, source_event_ids_json,
                topic, category, payload_json, created_at, cooldown_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (anchor_id, self.state.session_id, anchor_type, _json(source_event_ids or []), topic, anchor_type, _json(payload or {}), now, f"{anchor_type}:{topic}"),
        )
        conn.commit()
        conn.close()
        self.record_event(
            "spontaneity_anchor_created",
            source="spontaneity",
            topic=topic,
            category=anchor_type,
            confidence=0.8,
            provenance="live_session",
            payload={"anchor_id": anchor_id, **(payload or {})},
        )
        return anchor_id

    def consume_anchor(self, anchor_id: str | None) -> None:
        if not anchor_id:
            return
        now = _now_iso()
        conn = db_sqlite.get_db_connection()
        conn.execute("UPDATE live_session_spontaneity_anchors SET consumed_at = COALESCE(consumed_at, ?) WHERE anchor_id = ?", (now, anchor_id))
        conn.commit()
        conn.close()
        if anchor_id not in self.state.consumed_spontaneity_anchors:
            self.state.consumed_spontaneity_anchors.append(anchor_id)
            self.state.consumed_spontaneity_anchors = self.state.consumed_spontaneity_anchors[-20:]
        self.record_event("spontaneity_anchor_consumed", source="spontaneity", topic=anchor_id, payload={"anchor_id": anchor_id})

    def invalidate_anchor(self, anchor_id: str | None = None, *, reason: str = "correction") -> None:
        anchor_id = anchor_id or self.state.last_hebe_anchor
        if not anchor_id:
            return
        now = _now_iso()
        conn = db_sqlite.get_db_connection()
        conn.execute("UPDATE live_session_spontaneity_anchors SET invalidated_at = COALESCE(invalidated_at, ?) WHERE anchor_id = ?", (now, anchor_id))
        conn.commit()
        conn.close()
        if anchor_id not in self.state.invalidated_anchors:
            self.state.invalidated_anchors.append(anchor_id)
            self.state.invalidated_anchors = self.state.invalidated_anchors[-20:]
        self.record_event("spontaneity_anchor_invalidated", source="leo_stt", topic=anchor_id, category=reason, payload={"anchor_id": anchor_id, "reason": reason})

    def is_anchor_consumed_or_invalidated(self, anchor_id: str | None) -> bool:
        if not anchor_id:
            return False
        return anchor_id in set(self.state.consumed_spontaneity_anchors) or anchor_id in set(self.state.invalidated_anchors)

    def is_possible_reply_to_hebe(self, normalized_text: str) -> bool:
        utterance = self.state.last_hebe_utterance or {}
        if not utterance or float(utterance.get("expires_at", 0.0) or 0.0) < time.time():
            return False
        text = _norm(normalized_text)
        if not text:
            return False
        reply_markers = (
            "gracias hebe", "gracias eve", "pero", "ya esta", "ya vencimos",
            "ya lo hicimos", "eso ya paso", "no eso", "no hebe", "no eve",
            "vencimos al boss", "hemos ganado", "ya paso",
        )
        return any(marker in text for marker in reply_markers)

    def apply_correction(self, raw_text: str, normalized_text: str | None = None) -> int:
        normalized = _norm(normalized_text or raw_text)
        self._set("latest_correction_from_leo", raw_text, source="manual_correction", confidence=1.0)
        if any(marker in normalized for marker in ("no estoy peleando", "no estoy en combate", "fuera de combate", "vinculos sociales", "social links", "confidant", "confidants")):
            self._set("current_activity", "confidant_event" if "confidant" in normalized else "social_links", source="manual_correction", confidence=1.0)
            self.state.combat_state = False
            self.state.current_activity_provenance = "owner_correction"
            self.state.blocked_comment_categories = [
                "combat_advice",
                "healing_advice",
                "boss_strategy",
                "wipe_comment",
                "dungeon_resource_management",
                "SP_management",
            ]
            self.invalidate_anchor(reason="activity_corrected")
        if any(marker in normalized for marker in ("vencimos al boss", "vencimos el boss", "derrotamos al boss", "boss derrotado", "jefe derrotado")):
            self._set("latest_boss_state", "defeated", source="manual_correction", confidence=1.0)
            self.invalidate_anchor(reason="boss_state_corrected")
        if any(marker in normalized for marker in ("eso ya paso", "ya lo hicimos", "ya esta")):
            self.invalidate_anchor(reason="stale_assumption_corrected")
        return self.record_event(
            "correction",
            source="leo_stt",
            raw_text=raw_text,
            normalized_text=normalized,
            speaker="leo",
            target="hebe",
            topic="correction_to_hebe",
            category="correction_to_hebe",
            confidence=1.0,
            provenance="manual_correction",
            index_for_rag=True,
        )

    def retrieve_context(self, query: str = "", *, limit_events: int = 12, limit_summaries: int = 3) -> dict:
        conn = db_sqlite.get_db_connection()
        event_rows = conn.execute(
            """
            SELECT id, event_type, event_ts, source, raw_text, speaker, topic, category, output_target
            FROM live_session_timeline
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (self.state.session_id, limit_events),
        ).fetchall()
        summary_rows = conn.execute(
            """
            SELECT id, summary_text, event_count, created_at
            FROM live_session_rolling_summaries
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (self.state.session_id, limit_summaries),
        ).fetchall()
        conn.close()
        payload = {
            "live_state": self.state.as_dict(),
            "recent_events": [dict(row) for row in reversed(event_rows)],
            "rolling_summaries": [dict(row) for row in summary_rows],
            "query": query,
        }
        self.state.last_retrieved_context_used = {
            "at": _now_iso(),
            "query": query,
            "events": len(payload["recent_events"]),
            "summaries": len(payload["rolling_summaries"]),
        }
        self._persist_state()
        return payload

    def metadata_snapshot(self) -> dict:
        return {
            "stream_status": self.state.stream_status,
            "title": self.state.current_title,
            "game": self.state.current_game,
            "category": self.state.current_category,
            "language_mode": self.state.language_mode,
            "spoiler_policy": self.state.spoiler_policy,
        }

    def as_debug_dict(self) -> dict:
        return {
            "stream_metadata": self.metadata_snapshot(),
            "live_session": self.state.as_dict(),
            "memory_rag": {
                "latest_rolling_summary_time": self.state.latest_rolling_summary_time,
                "meaningful_events": self.state.meaningful_event_count,
                "session_context_updates": self.state.session_context_update_count,
                "last_retrieved_context_used": self.state.last_retrieved_context_used,
            },
        }

    def classify_topic(self, text: str, *, voice_event_type: str = "") -> str:
        normalized = _norm(text)
        if self._looks_like_correction(normalized):
            return "correction_to_hebe"
        if voice_event_type == "confusion/lost" or any(marker in normalized for marker in ("donde voy", "que hago", "no entiendo")):
            return "latest_confusion"
        if any(marker in normalized for marker in ("boss", "jefe", "pull", "intento")):
            return "current_game_combat"
        if any(marker in normalized for marker in ("vida", "hp", "cura", "pocion", "recurso", "mana", "sp")):
            return "current_game_resource"
        if any(marker in normalized for marker in ("objetivo", "vamos a", "hay que", "ascensor", "lleva")):
            return "current_game_objective"
        if any(marker in normalized for marker in ("dungeon", "mazmorra", "pasado", "terminando", "avanzar")):
            return "current_game_progress"
        if any(marker in normalized for marker in ("anime", "manga", "serie", "pelicula")):
            return "anime_topic"
        if any(marker in normalized for marker in ("obs", "micro", "camara", "audio", "se ve", "lag")):
            return "technical_issue"
        if any(marker in normalized for marker in ("jaja", "lol", "broma")):
            return "joke/banter"
        return voice_event_type or "casual_comment"

    def _looks_like_correction(self, normalized: str) -> bool:
        return any(marker in normalized for marker in ("pero ya", "no eso", "eso ya paso", "ya vencimos", "ya lo hicimos", "corrige contexto"))

    def _apply_fact(self, fact: dict) -> None:
        kind = str(fact.get("kind") or fact.get("category") or "")
        text = str(fact.get("text") or fact.get("summary") or "").strip()
        if not text:
            return
        confidence = float(fact.get("confidence", 0.0) or 0.0)
        if kind == "objective":
            self._set("current_objective", text, source="leo_stt", confidence=confidence)
        elif kind == "location":
            self._set("current_location", text, source="leo_stt", confidence=confidence)
        elif kind in {"progress_marker", "phase", "level_gap"}:
            self._set("current_phase", text, source="leo_stt", confidence=confidence)
        elif kind in {"combat_risk", "boss_or_area_difficulty", "enemy_mechanic", "boss_attempt"}:
            self._set("latest_boss_state", text, source="leo_stt", confidence=confidence)
        elif kind in {"failure_or_death", "repeated_failure"}:
            self._set("latest_failure_or_success", text, source="leo_stt", confidence=confidence)
        elif kind in {"resource_management", "healing_or_recovery", "low_hp"}:
            self._set("latest_resource_state", text, source="leo_stt", confidence=confidence)
        elif kind in {"guide_strategy", "challenge_constraint"}:
            self._set("latest_strategy_topic", text, source="leo_stt", confidence=confidence)
        elif kind == "navigation_confusion":
            self._set("latest_confusion", text, source="leo_stt", confidence=confidence)

    def _set(self, field_name: str, value: Any, *, source: str | None = None, confidence: float = 0.7) -> None:
        if value in (None, "", []):
            return
        setattr(self.state, field_name, value)
        self.state.facts_provenance[field_name] = {
            "source": source or "unknown",
            "confidence": confidence,
            "updated_at": _now_iso(),
        }

    def _refresh_participants(self) -> None:
        values = [p.as_dict() for p in self.participants.values()]
        values.sort(key=lambda p: p.get("last_seen_at") or "", reverse=True)
        self.state.recent_chatters = values[:20]
        self.state.active_chatters = values[:8]
        self.state.possible_lurkers_from_recent_chat = values[8:20]

    def _maybe_roll_summary(self) -> None:
        now = time.time()
        event_delta = self.state.meaningful_event_count - self._last_summary_event_count
        if event_delta < 25 and now - self._last_summary_ts < 10 * 60:
            return
        text = self._build_rolling_summary_text()
        if not text:
            return
        created_at = _now_iso()
        conn = db_sqlite.get_db_connection()
        conn.execute(
            """
            INSERT INTO live_session_rolling_summaries (
                session_id, stream_session_id, summary_text, event_count, created_at, tags_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (self.state.session_id, self.state.stream_session_id, text, self.state.meaningful_event_count, created_at, _json({"kind": "rolling_session_summary"})),
        )
        conn.commit()
        conn.close()
        self.state.latest_rolling_summary_time = created_at
        self._last_summary_ts = now
        self._last_summary_event_count = self.state.meaningful_event_count
        try:
            conn = db_sqlite.get_db_connection()
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_chunks'"
            ).fetchone()
            conn.close()
            if not exists:
                return
            from app.cognitive.memory.memory_store import add_chunk_if_new

            add_chunk_if_new(
                text,
                "stream_summary",
                subject="live_session",
                source_session=self.state.session_id,
                importance=0.72,
                tags={"session_id": self.state.session_id, "rolling": True},
            )
        except Exception as exc:
            print(f"[HEBE][LIVE_SESSION] rolling summary chunk skipped error={exc!r}", flush=True)

    def _build_rolling_summary_text(self) -> str:
        parts = [
            f"Live session {self.state.session_id}.",
            f"Game/category: {self.state.current_game or self.state.current_category or 'unknown'}.",
        ]
        if self.state.current_objective:
            parts.append(f"Objective: {self.state.current_objective}.")
        if self.state.current_phase:
            parts.append(f"Phase: {self.state.current_phase}.")
        if self.state.latest_correction_from_leo:
            parts.append(f"Latest correction from Leo: {self.state.latest_correction_from_leo}.")
        if self.state.current_chat_topic:
            parts.append(f"Current chat topic: {self.state.current_chat_topic}.")
        if self.state.recent_chatters:
            names = [item.get("display_name") or item.get("username") for item in self.state.recent_chatters[:8]]
            parts.append("Recent chatters: " + ", ".join(name for name in names if name) + ".")
        if self.state.last_hebe_utterance:
            parts.append(f"Last Hebe utterance: {self.state.last_hebe_utterance.get('text')}.")
        return " ".join(part for part in parts if part)

    def _touch(self) -> None:
        self.state.last_updated_at = _now_iso()
        self._persist_state()

    def _persist_state(self) -> None:
        now = self.state.last_updated_at or _now_iso()
        conn = db_sqlite.get_db_connection()
        conn.execute(
            """
            INSERT INTO live_session_state (
                session_id, stream_session_id, stream_status, current_game,
                current_category, current_title, language_mode, spoiler_policy,
                state_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                stream_session_id=excluded.stream_session_id,
                stream_status=excluded.stream_status,
                current_game=excluded.current_game,
                current_category=excluded.current_category,
                current_title=excluded.current_title,
                language_mode=excluded.language_mode,
                spoiler_policy=excluded.spoiler_policy,
                state_json=excluded.state_json,
                updated_at=excluded.updated_at
            """,
            (
                self.state.session_id,
                self.state.stream_session_id,
                self.state.stream_status,
                self.state.current_game,
                self.state.current_category,
                self.state.current_title,
                self.state.language_mode,
                self.state.spoiler_policy,
                _json(self.state.as_dict()),
                now,
                now,
            ),
        )
        conn.commit()
        conn.close()


def latest_live_session_debug() -> dict | None:
    try:
        init_live_session_schema()
        conn = db_sqlite.get_db_connection()
        row = conn.execute(
            "SELECT * FROM live_session_state ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            conn.close()
            return None
        state = _loads(row["state_json"], {})
        events = [
            dict(event)
            for event in conn.execute(
                """
                SELECT id, event_type, event_ts, source, speaker, topic, raw_text, output_target
                FROM live_session_timeline
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 20
                """,
                (row["session_id"],),
            ).fetchall()
        ]
        summaries = [
            dict(summary)
            for summary in conn.execute(
                """
                SELECT id, summary_text, event_count, created_at
                FROM live_session_rolling_summaries
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 5
                """,
                (row["session_id"],),
            ).fetchall()
        ]
        conn.close()
        return {
            "stream_metadata": {
                "live_status": row["stream_status"],
                "title": row["current_title"],
                "game": row["current_game"],
                "category": row["current_category"],
                "language": row["language_mode"],
            },
            "live_session": state,
            "memory_rag": {
                "latest_rolling_summary_time": state.get("latest_rolling_summary_time"),
                "meaningful_events": state.get("meaningful_event_count", 0),
                "session_context_updates": state.get("session_context_update_count", 0),
                "last_retrieved_context_used": state.get("last_retrieved_context_used"),
                "recent_timeline_events": events,
                "rolling_summaries": summaries,
            },
        }
    except Exception as exc:
        print(f"[HEBE][LIVE_SESSION] debug read failed: {exc!r}", flush=True)
        return None
