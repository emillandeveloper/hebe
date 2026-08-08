from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable, Protocol

from app.services import db_sqlite


def _now_iso(now: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if now is None else now, timezone.utc).isoformat()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_") or "unknown_game"


def _norm(value: str) -> str:
    text = str(value or "").casefold().translate(str.maketrans("áéíóúüñ", "aeiouun"))
    return " ".join(re.sub(r"[^a-z0-9_ ]+", " ", text).split())


class ResearchMode(StrEnum):
    PRE_STREAM_DOSSIER = "pre_stream_dossier"
    CONTEXTUAL_LOOKUP = "contextual_lookup"
    KNOWLEDGE_GAP_FILL = "knowledge_gap_fill"
    OWNER_EXPLICIT_QUESTION = "owner_explicit_question"
    OFFLINE_REFRESH = "offline_refresh"
    REPLAY_FIXTURE = "replay_fixture"


class SpoilerClassification(StrEnum):
    SAFE_GENERAL_MECHANIC = "safe_general_mechanic"
    SAFE_CURRENT_PROGRESS = "safe_current_progress"
    UNCERTAIN_PROGRESS = "uncertain_progress"
    FUTURE_MECHANIC = "future_mechanic"
    STORY_SPOILER = "story_spoiler"
    IDENTITY_SPOILER = "identity_spoiler"
    LOCATION_SPOILER = "location_spoiler"
    ENDING_SPOILER = "ending_spoiler"


class GameAssistanceMode(StrEnum):
    REACTIONS_ONLY = "reactions_only"
    MECHANICS_WITHOUT_SOLUTIONS = "mechanics_without_solutions"
    HINTS_ON_REQUEST = "hints_on_request"
    FULL_HELP_ON_REQUEST = "full_help_on_request"


@dataclass(slots=True)
class GameDossier:
    game_id: str
    canonical_title: str
    aliases: list[str] = field(default_factory=list)
    release: str = ""
    version: str = ""
    platform: str = ""
    core_genre: list[str] = field(default_factory=list)
    premise_without_spoilers: str = ""
    main_known_characters: list[str] = field(default_factory=list)
    core_combat_systems: list[str] = field(default_factory=list)
    safe_gameplay_vocabulary: list[str] = field(default_factory=list)
    common_ui_terminology: list[str] = field(default_factory=list)
    confirmed_general_mechanics: list[str] = field(default_factory=list)
    unsafe_story_topics: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    dossier_version: int = 1


@dataclass(slots=True)
class GameProgressState:
    game_id: str
    stream_session_id: str
    playthrough_type: str = "unknown"
    spoiler_policy: str = "strict"
    current_chapter: str = ""
    current_area: str = ""
    known_party_members: list[str] = field(default_factory=list)
    encountered_characters: list[str] = field(default_factory=list)
    encountered_bosses: list[str] = field(default_factory=list)
    unlocked_mechanics: list[str] = field(default_factory=list)
    recent_progress_markers: list[str] = field(default_factory=list)
    confidence: float = 0.0
    last_updated_at: str = ""


@dataclass(frozen=True, slots=True)
class SpoilerGuardResult:
    allowed: bool
    classification: str
    progress_basis: str
    hidden_claims: list[str]
    reason: str


@dataclass(frozen=True, slots=True)
class ResearchTriggerDecision:
    should_research: bool
    mode: str
    query_scope: str
    evidence: list[str]
    urgency: str
    cache_key: str
    reason: str


@dataclass(frozen=True, slots=True)
class GameSearchPlan:
    query: str
    game_id: str
    entity: str
    question_type: str
    spoiler_limit: str
    expected_fact_type: str
    cache_key: str


@dataclass(slots=True)
class RetrievedGameFact:
    fact_id: str
    claim: str
    source_title: str
    source_location: str
    retrieved_at: str
    confidence: float
    corroboration_count: int
    spoiler_classification: str
    progress_compatibility: str
    exact_supporting_excerpt_internal: str
    usable_for_comment: bool
    usable_for_advice: bool
    source_type: str = "unknown"
    progress_requirements: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeGap:
    term: str
    game_id: str
    raw_evidence: list[str]
    first_seen: str
    occurrence_count: int
    priority: str
    status: str
    resolved_fact_ids: list[str]


@dataclass(slots=True)
class CommentKnowledgeContract:
    scene_evidence: list[str]
    game_facts: list[RetrievedGameFact]
    source_provenance: list[dict[str, str]]
    progress_state: GameProgressState
    spoiler_constraints: list[str]
    allowed_claims: list[str]
    forbidden_claims: list[str]
    contribution_mode: str
    scene_id: str = ""
    topic_id: str = ""
    scene_fact_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GameResearchJob:
    job_id: str
    plan: GameSearchPlan
    scene_id: str
    session_id: str = ""
    game_id: str = ""
    mode: str = ResearchMode.CONTEXTUAL_LOOKUP.value
    attempt: int = 1
    status: str = "queued"
    queued_at: float = 0.0
    started_at: float = 0.0
    timeout_seconds: float = 0.0
    expires_at: float = 0.0
    completed_at: float = 0.0
    failure_reason: str = ""
    next_retry_at: float = 0.0
    fact_ids: list[str] = field(default_factory=list)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GameIntelligenceDiagnostics:
    current_game: str = ""
    dossier_status: str = "missing"
    progress_state: dict[str, Any] = field(default_factory=dict)
    spoiler_mode: str = "strict"
    last_research_trigger: dict[str, Any] = field(default_factory=dict)
    active_research_job: dict[str, Any] = field(default_factory=dict)
    last_query: str = ""
    sources_found: int = 0
    facts_accepted: list[str] = field(default_factory=list)
    facts_rejected: list[str] = field(default_factory=list)
    last_spoiler_block: dict[str, Any] = field(default_factory=dict)
    unresolved_knowledge_gaps: list[str] = field(default_factory=list)
    cache_hits: int = 0
    lookup_used: bool = False
    current_comment_fact_ids: list[str] = field(default_factory=list)
    current_comment_mode: str = "contextual_reaction"
    current_comment_provenance: list[dict[str, Any]] = field(default_factory=list)
    research_provider: str = "none"
    research_provider_configured: bool = False
    research_provider_available: bool = False
    research_provider_reason: str = "provider_missing"


class GameResearchProvider(Protocol):
    provider_name: str
    available: bool

    def search(
        self,
        query: str,
        constraints: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        cancellation: threading.Event | None = None,
    ) -> list[dict[str, Any]]:
        ...


class GameIntelligenceStore:
    def __init__(self, *, connection: sqlite3.Connection | None = None) -> None:
        self._connection = connection
        if connection is not None:
            connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.init_schema()

    def _connect(self) -> tuple[sqlite3.Connection, bool]:
        if self._connection is not None:
            return self._connection, False
        return db_sqlite.get_db_connection(), True

    def init_schema(self) -> None:
        conn, close = self._connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS game_dossiers (
                game_id TEXT PRIMARY KEY,
                canonical_title TEXT NOT NULL,
                dossier_json TEXT NOT NULL,
                dossier_version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS game_research_cache (
                cache_key TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                query TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                facts_json TEXT NOT NULL,
                spoiler_classification_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS game_progress_states (
                game_id TEXT NOT NULL,
                stream_session_id TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(game_id, stream_session_id)
            );
            CREATE TABLE IF NOT EXISTS game_knowledge_gaps (
                game_id TEXT NOT NULL,
                term TEXT NOT NULL,
                gap_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(game_id, term)
            );
            CREATE INDEX IF NOT EXISTS idx_game_cache_game ON game_research_cache(game_id);
            CREATE INDEX IF NOT EXISTS idx_game_gap_status ON game_knowledge_gaps(game_id, updated_at);
            """
        )
        conn.commit()
        if close:
            conn.close()

    def save_dossier(self, dossier: GameDossier) -> GameDossier:
        now = dossier.updated_at or _now_iso()
        dossier.created_at = dossier.created_at or now
        dossier.updated_at = now
        conn, close = self._connect()
        conn.execute(
            """
            INSERT INTO game_dossiers(game_id, canonical_title, dossier_json, dossier_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET canonical_title=excluded.canonical_title,
                dossier_json=excluded.dossier_json, dossier_version=excluded.dossier_version,
                updated_at=excluded.updated_at
            """,
            (
                dossier.game_id,
                dossier.canonical_title,
                json.dumps(asdict(dossier), ensure_ascii=False),
                dossier.dossier_version,
                dossier.created_at,
                dossier.updated_at,
            ),
        )
        conn.commit()
        if close:
            conn.close()
        return dossier

    def get_dossier(self, game_id_or_title: str) -> GameDossier | None:
        key = _slug(game_id_or_title)
        conn, close = self._connect()
        row = conn.execute(
            """
            SELECT dossier_json FROM game_dossiers
            WHERE game_id = ? OR lower(canonical_title) = lower(?)
               OR EXISTS (
                   SELECT 1 FROM json_each(json_extract(dossier_json, '$.aliases'))
                   WHERE lower(json_each.value) = lower(?)
               )
            LIMIT 1
            """,
            (key, str(game_id_or_title or ""), str(game_id_or_title or "")),
        ).fetchone()
        if close:
            conn.close()
        return GameDossier(**json.loads(row["dossier_json"])) if row is not None else None

    def save_progress(self, progress: GameProgressState) -> GameProgressState:
        progress.last_updated_at = progress.last_updated_at or _now_iso()
        conn, close = self._connect()
        conn.execute(
            """
            INSERT INTO game_progress_states(game_id, stream_session_id, state_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(game_id, stream_session_id) DO UPDATE SET
                state_json=excluded.state_json, updated_at=excluded.updated_at
            """,
            (progress.game_id, progress.stream_session_id, json.dumps(asdict(progress), ensure_ascii=False), progress.last_updated_at),
        )
        conn.commit()
        if close:
            conn.close()
        return progress

    def get_progress(self, game_id: str, session_id: str | int) -> GameProgressState | None:
        conn, close = self._connect()
        row = conn.execute(
            "SELECT state_json FROM game_progress_states WHERE game_id = ? AND stream_session_id = ?",
            (_slug(game_id), str(session_id or "")),
        ).fetchone()
        if close:
            conn.close()
        return GameProgressState(**json.loads(row["state_json"])) if row is not None else None

    def save_cache(
        self,
        plan: GameSearchPlan,
        sources: list[dict[str, Any]],
        facts: list[RetrievedGameFact],
        *,
        ttl_seconds: float,
        now: float | None = None,
    ) -> None:
        ts = time.time() if now is None else float(now)
        conn, close = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO game_research_cache(
                cache_key, game_id, query, sources_json, facts_json,
                spoiler_classification_json, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan.cache_key,
                plan.game_id,
                plan.query,
                json.dumps(sources, ensure_ascii=False),
                json.dumps([asdict(fact) for fact in facts], ensure_ascii=False),
                json.dumps([fact.spoiler_classification for fact in facts], ensure_ascii=False),
                _now_iso(ts),
                ts + ttl_seconds,
            ),
        )
        conn.commit()
        if close:
            conn.close()

    def get_cached_facts(self, cache_key: str, *, now: float | None = None) -> list[RetrievedGameFact] | None:
        ts = time.time() if now is None else float(now)
        conn, close = self._connect()
        row = conn.execute(
            "SELECT facts_json FROM game_research_cache WHERE cache_key = ? AND expires_at >= ?",
            (cache_key, ts),
        ).fetchone()
        if close:
            conn.close()
        if row is None:
            return None
        return [RetrievedGameFact(**item) for item in json.loads(row["facts_json"] or "[]")]

    def save_gap(self, gap: KnowledgeGap) -> KnowledgeGap:
        conn, close = self._connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO game_knowledge_gaps(game_id, term, gap_json, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (gap.game_id, _norm(gap.term), json.dumps(asdict(gap), ensure_ascii=False), _now_iso()),
        )
        conn.commit()
        if close:
            conn.close()
        return gap

    def get_gap(self, game_id: str, term: str) -> KnowledgeGap | None:
        conn, close = self._connect()
        row = conn.execute(
            "SELECT gap_json FROM game_knowledge_gaps WHERE game_id = ? AND term = ?",
            (_slug(game_id), _norm(term)),
        ).fetchone()
        if close:
            conn.close()
        return KnowledgeGap(**json.loads(row["gap_json"])) if row is not None else None

    def unresolved_gaps(self, game_id: str) -> list[KnowledgeGap]:
        conn, close = self._connect()
        rows = conn.execute("SELECT gap_json FROM game_knowledge_gaps WHERE game_id = ?", (_slug(game_id),)).fetchall()
        if close:
            conn.close()
        gaps = [KnowledgeGap(**json.loads(row["gap_json"])) for row in rows]
        return [gap for gap in gaps if gap.status in {"open", "researching", "uncertain", "failed"}]


class GameProgressTracker:
    def __init__(self, store: GameIntelligenceStore) -> None:
        self.store = store

    def start(self, game: str, session_id: str | int, *, title: str = "") -> GameProgressState:
        game_id = _slug(game)
        existing = self.store.get_progress(game_id, session_id)
        if existing:
            return existing
        normalized_title = _norm(title)
        first = bool(re.search(r"\b(?:first playthrough|blind|primera partida|primera vez|sin spoilers|no spoilers)\b", normalized_title))
        state = GameProgressState(
            game_id=game_id,
            stream_session_id=str(session_id or ""),
            playthrough_type="first_playthrough" if first else "unknown",
            spoiler_policy="strict" if first else "strict",
            confidence=0.9 if first else 0.2,
            last_updated_at=_now_iso(),
        )
        return self.store.save_progress(state)

    def apply_owner_progress(
        self,
        state: GameProgressState,
        statement: str,
        *,
        explicit_owner_statement: bool,
        quoted_dialogue: bool = False,
        chapter: str = "",
        area: str = "",
        party_member: str = "",
        encountered_character: str = "",
        encountered_boss: str = "",
        unlocked_mechanic: str = "",
        confidence: float = 0.9,
    ) -> GameProgressState:
        if not explicit_owner_statement or quoted_dialogue:
            return state
        raw = str(statement or "").strip()
        if chapter:
            state.current_chapter = chapter
        if area:
            state.current_area = area
        for value, collection in (
            (party_member, state.known_party_members),
            (encountered_character, state.encountered_characters),
            (encountered_boss, state.encountered_bosses),
            (unlocked_mechanic, state.unlocked_mechanics),
        ):
            if value and value not in collection:
                collection.append(value)
        if raw and raw not in state.recent_progress_markers:
            state.recent_progress_markers.append(raw[:240])
            state.recent_progress_markers = state.recent_progress_markers[-12:]
        state.confidence = max(state.confidence, min(1.0, confidence))
        state.last_updated_at = _now_iso()
        return self.store.save_progress(state)


class SpoilerFirewall:
    _STRICT_ALLOWED = {
        SpoilerClassification.SAFE_GENERAL_MECHANIC.value,
        SpoilerClassification.SAFE_CURRENT_PROGRESS.value,
    }

    def evaluate(self, fact: RetrievedGameFact, progress: GameProgressState) -> SpoilerGuardResult:
        classification = str(fact.spoiler_classification or SpoilerClassification.UNCERTAIN_PROGRESS.value)
        progress_basis = self._progress_basis(progress)
        strict = progress.spoiler_policy in {"strict", "mechanics_only", "no_spoilers", "owner_defined"}
        allowed = classification in self._STRICT_ALLOWED if strict else classification not in {
            SpoilerClassification.ENDING_SPOILER.value,
            SpoilerClassification.STORY_SPOILER.value,
            SpoilerClassification.IDENTITY_SPOILER.value,
        }
        if classification == SpoilerClassification.SAFE_CURRENT_PROGRESS.value and not fact.progress_compatibility.startswith("compatible"):
            allowed = False
        if progress.confidence < 0.5 and classification != SpoilerClassification.SAFE_GENERAL_MECHANIC.value:
            allowed = False
        reason = "spoiler_safe" if allowed else "classification_or_progress_blocked"
        return SpoilerGuardResult(
            allowed,
            classification,
            progress_basis,
            [] if allowed else [fact.claim],
            reason,
        )

    @staticmethod
    def _progress_basis(progress: GameProgressState) -> str:
        bits = [progress.current_chapter, progress.current_area, *progress.encountered_bosses[-3:], *progress.unlocked_mechanics[-3:]]
        return " | ".join(bit for bit in bits if bit) or "unknown_progress"


class ResearchTriggerEngine:
    def __init__(self, store: GameIntelligenceStore) -> None:
        self.store = store
        self._counts: dict[tuple[str, str], int] = {}

    def decide(
        self,
        *,
        game_id: str,
        text: str,
        entity: str = "",
        explicit_direct_question: bool = False,
        owner_uncertainty: bool = False,
        unknown_mechanic: bool = False,
        high_value_topic: bool = False,
        scheduled_pre_stream: bool = False,
        quoted_dialogue: bool = False,
        filler: bool = False,
        confidence: float = 1.0,
    ) -> ResearchTriggerDecision:
        scope = _norm(entity or text)[:120]
        cache_key = _cache_key(game_id, scope, "contextual")
        if self.store.get_cached_facts(cache_key) is not None:
            return ResearchTriggerDecision(False, "", scope, [], "none", cache_key, "cached_query")
        if filler or quoted_dialogue:
            return ResearchTriggerDecision(False, "", scope, [], "none", cache_key, "filler_or_quoted_dialogue")
        if confidence < 0.65 and not explicit_direct_question:
            return ResearchTriggerDecision(False, "", scope, [text], "none", cache_key, "low_confidence_fragment")
        key = (_slug(game_id), scope)
        if owner_uncertainty or unknown_mechanic:
            self._counts[key] = self._counts.get(key, 0) + 1
        count = self._counts.get(key, 0)
        if scheduled_pre_stream:
            mode, reason, urgency = ResearchMode.PRE_STREAM_DOSSIER.value, "scheduled_pre_stream", "normal"
        elif explicit_direct_question:
            mode, reason, urgency = ResearchMode.OWNER_EXPLICIT_QUESTION.value, "owner_direct_game_question", "high"
        elif (owner_uncertainty or unknown_mechanic) and count >= 2:
            mode, reason, urgency = ResearchMode.CONTEXTUAL_LOOKUP.value, "repeated_stable_unknown", "normal"
        elif high_value_topic:
            mode, reason, urgency = ResearchMode.CONTEXTUAL_LOOKUP.value, "high_value_discourse_topic", "normal"
        else:
            return ResearchTriggerDecision(False, "", scope, [text] if text else [], "none", cache_key, "insufficient_research_evidence")
        return ResearchTriggerDecision(True, mode, scope, [text] if text else [], urgency, cache_key, reason)


class GameAssistanceGuard:
    _SOLUTION_PATTERNS = (
        r"\b(?:solucion|solution|respuesta del puzzle|puzzle answer)\b",
        r"\b(?:ruta exacta|exact route|ve primero a|gira a la izquierda|gira a la derecha)\b",
        r"\b(?:debilidad del jefe|boss weakness|es debil a|weak to)\b",
        r"\b(?:build exacta|equipa exactamente|equip exactly|best build)\b",
    )

    def allow(
        self,
        text: str,
        *,
        mode: GameAssistanceMode | str,
        explicit_owner_request: bool = False,
        accessibility_or_safety: bool = False,
    ) -> tuple[bool, str]:
        resolved = GameAssistanceMode(str(mode))
        prescriptive = any(re.search(pattern, _norm(text)) for pattern in self._SOLUTION_PATTERNS)
        if not prescriptive:
            return True, "general_mechanic_or_reaction"
        if accessibility_or_safety:
            return True, "accessibility_or_safety_exception"
        if resolved is GameAssistanceMode.FULL_HELP_ON_REQUEST and explicit_owner_request:
            return True, "explicit_full_help_request"
        return False, "walkthrough_behavior_suppressed"


class CommentKnowledgePolicy:
    def __init__(self, firewall: SpoilerFirewall | None = None) -> None:
        self.firewall = firewall or SpoilerFirewall()

    def build_contract(
        self,
        *,
        scene_evidence: list[str],
        facts: list[RetrievedGameFact],
        progress: GameProgressState,
        requested_tip: bool = False,
        high_social_value: bool = False,
        low_interruption_cost: bool = False,
        scene_facts: list[dict[str, Any]] | None = None,
        current_scene_id: str = "",
        current_topic_id: str = "",
        now: float | None = None,
    ) -> CommentKnowledgeContract:
        filtered_scene_facts = self.filter_scene_facts(
            scene_facts or [],
            current_scene_id=current_scene_id,
            current_topic_id=current_topic_id,
            now=now,
        )
        if scene_facts is not None:
            scene_evidence = [
                str(item.get("raw_evidence") or item.get("raw_text") or item.get("text") or item.get("summary") or "")
                for item in filtered_scene_facts
                if str(item.get("raw_evidence") or item.get("raw_text") or item.get("text") or item.get("summary") or "").strip()
            ]
        allowed_facts: list[RetrievedGameFact] = []
        forbidden: list[str] = []
        provenance: list[dict[str, str]] = []
        for fact in facts:
            guard = self.firewall.evaluate(fact, progress)
            if guard.allowed and fact.usable_for_comment:
                allowed_facts.append(fact)
                provenance.append({"fact_id": fact.fact_id, "source": fact.source_location})
            else:
                forbidden.extend(guard.hidden_claims or [fact.claim])
        validated_tip = (
            requested_tip
            and high_social_value
            and low_interruption_cost
            and any(fact.usable_for_advice for fact in allowed_facts)
        )
        if validated_tip:
            mode = "validated_tip"
        elif allowed_facts:
            mode = "informed_observation"
        elif scene_evidence:
            mode = "contextual_reaction"
        else:
            mode = "no_output"
        return CommentKnowledgeContract(
            list(scene_evidence),
            allowed_facts,
            provenance,
            progress,
            ["strict_progress_filter", "remove_forbidden_before_render"],
            [fact.claim for fact in allowed_facts],
            list(dict.fromkeys(forbidden)),
            mode,
            str(current_scene_id or ""),
            str(current_topic_id or ""),
            [
                str(item.get("id") or item.get("fact_id") or "")
                for item in filtered_scene_facts
                if str(item.get("id") or item.get("fact_id") or "")
            ],
        )

    @staticmethod
    def filter_scene_facts(
        facts: list[dict[str, Any]],
        *,
        current_scene_id: str,
        current_topic_id: str,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        now = time.time() if now is None else float(now)
        selected: list[dict[str, Any]] = []
        for source in facts:
            fact = dict(source)
            timestamp = float(fact.get("timestamp", now) or now)
            ttl = max(1.0, float(fact.get("ttl_sec", 60.0) or 60.0))
            age = max(0.0, now - timestamp)
            fact["age_seconds"] = age
            fact["currentness_score"] = 0.0 if fact.get("superseded") else max(0.0, 1.0 - age / ttl)
            if bool(fact.get("superseded")) or float(fact.get("expires_at", now + 1.0) or 0.0) <= now:
                continue
            if current_scene_id and str(fact.get("scene_id") or "") != str(current_scene_id):
                continue
            if current_topic_id and str(fact.get("topic_id") or "") != str(current_topic_id):
                continue
            selected.append(fact)
        return selected


class KnowledgeGapTracker:
    def __init__(self, store: GameIntelligenceStore, *, persist_after: int = 2) -> None:
        self.store = store
        self.persist_after = max(2, int(persist_after))
        self._observations: dict[tuple[str, str], list[str]] = {}

    def observe(self, *, game_id: str, term: str, raw_evidence: str, priority: str = "normal") -> KnowledgeGap | None:
        game_key, term_key = _slug(game_id), _norm(term)
        if not term_key or len(term_key) < 3:
            return None
        key = (game_key, term_key)
        evidence = self._observations.setdefault(key, [])
        evidence.append(str(raw_evidence or "")[:300])
        existing = self.store.get_gap(game_key, term_key)
        count = (existing.occurrence_count if existing else 0) + 1
        if existing is None and len(evidence) < self.persist_after:
            return None
        gap = existing or KnowledgeGap(term_key, game_key, [], _now_iso(), 0, priority, "open", [])
        gap.occurrence_count = max(count, len(evidence))
        gap.raw_evidence = list(dict.fromkeys([*gap.raw_evidence, *evidence]))[-8:]
        gap.priority = priority
        if gap.status not in {"resolved", "researching"}:
            gap.status = "open"
        return self.store.save_gap(gap)

    def resolve(self, gap: KnowledgeGap, facts: list[RetrievedGameFact], dossier: GameDossier | None = None) -> KnowledgeGap:
        safe = [fact for fact in facts if fact.usable_for_comment and fact.spoiler_classification == SpoilerClassification.SAFE_GENERAL_MECHANIC.value]
        if not safe:
            gap.status = "failed" if not facts else "uncertain"
            gap.resolved_fact_ids = []
            return self.store.save_gap(gap)
        gap.status = "resolved"
        gap.resolved_fact_ids = [fact.fact_id for fact in safe]
        self.store.save_gap(gap)
        if dossier is not None:
            dossier.confirmed_general_mechanics = list(dict.fromkeys([
                *dossier.confirmed_general_mechanics,
                *(fact.claim for fact in safe),
            ]))
            dossier.dossier_version += 1
            dossier.updated_at = _now_iso()
            self.store.save_dossier(dossier)
        return gap


class GameResearchService:
    """Provider-neutral, cached, spoiler-aware game research for Hebe Live."""

    def __init__(
        self,
        *,
        store: GameIntelligenceStore | None = None,
        provider: GameResearchProvider | None = None,
        cache_ttl_seconds: float = 30 * 86400,
        max_workers: int = 2,
        now_fn: Callable[[], float] = time.time,
        provider_name: str = "",
        provider_configured: bool | None = None,
        contextual_timeout_seconds: float = 10.0,
        dossier_timeout_seconds: float = 15.0,
        max_attempts: int = 3,
        retry_base_seconds: float = 2.0,
    ) -> None:
        self.store = store or GameIntelligenceStore()
        self.provider = provider
        self.provider_name = str(provider_name or getattr(provider, "provider_name", "") or (type(provider).__name__ if provider is not None else "none"))
        self.provider_configured = bool(provider is not None if provider_configured is None else provider_configured)
        self.cache_ttl_seconds = max(60.0, float(cache_ttl_seconds))
        self.now_fn = now_fn
        self.contextual_timeout_seconds = max(1.0, float(contextual_timeout_seconds))
        self.dossier_timeout_seconds = max(self.contextual_timeout_seconds, float(dossier_timeout_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_seconds = max(0.1, float(retry_base_seconds))
        self.progress = GameProgressTracker(self.store)
        self.spoiler_firewall = SpoilerFirewall()
        self.trigger_engine = ResearchTriggerEngine(self.store)
        self.gaps = KnowledgeGapTracker(self.store)
        self.comment_policy = CommentKnowledgePolicy(self.spoiler_firewall)
        self.advice_guard = GameAssistanceGuard()
        self.diagnostics = GameIntelligenceDiagnostics()
        self.diagnostics.research_provider = self.provider_name
        self.diagnostics.research_provider_configured = self.provider_configured
        provider_available = bool(provider is not None and getattr(provider, "available", True))
        self.diagnostics.research_provider_available = provider_available
        self.diagnostics.research_provider_reason = (
            "ready" if provider_available else "configured_provider_unavailable" if self.provider_configured else "provider_not_configured"
        )
        self._log_provider_availability()
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="hebe-game-research")
        self._jobs: dict[str, tuple[GameResearchJob, Future[list[RetrievedGameFact]]]] = {}
        self._job_lock = threading.RLock()

    def canonical_game(self, title: str, *, aliases: list[str] | None = None) -> tuple[str, str, list[str]]:
        clean = str(title or "").strip()
        known = self.store.get_dossier(clean)
        if known:
            return known.game_id, known.canonical_title, known.aliases
        return _slug(clean), clean, sorted(set([clean, *(aliases or [])]))

    def get_or_build_dossier(
        self,
        *,
        game_title: str,
        platform: str = "",
        version: str = "",
        force_refresh: bool = False,
    ) -> tuple[GameDossier | None, str]:
        game_id, canonical, aliases = self.canonical_game(game_title)
        self.diagnostics.current_game = canonical
        existing = self.store.get_dossier(game_id)
        if existing and not force_refresh and self._dossier_sufficient(existing):
            self.diagnostics.dossier_status = "loaded"
            self._log_dossier(existing, "loaded")
            return existing, "loaded"
        if self.provider is None:
            status = "insufficient" if existing else "failed"
            self.diagnostics.dossier_status = status
            self._log_dossier(existing, status, game=canonical)
            return existing, status
        plan = self.plan_search(
            game_title=canonical,
            game_id=game_id,
            platform=platform,
            entity="core systems and spoiler-free premise",
            question_type="dossier",
            expected_fact_type="general_mechanics",
        )
        try:
            facts, sources, _ = self._retrieve(plan, progress=None, allow_cache=not force_refresh)
        except Exception as exc:
            self.diagnostics.dossier_status = "failed"
            self._log_dossier(existing, "failed", game=canonical)
            print(f"[HEBE][GAME_RESEARCH_JOB] status=failed mode=pre_stream_dossier error={type(exc).__name__}", flush=True)
            return existing, "failed"
        dossier = self._build_dossier(canonical, game_id, aliases, platform, version, facts, sources, existing)
        if not self._dossier_sufficient(dossier):
            self.diagnostics.dossier_status = "insufficient"
            self._log_dossier(existing, "insufficient", game=canonical)
            return existing, "insufficient"
        self.store.save_dossier(dossier)
        status = "created" if existing is None else "loaded"
        self.diagnostics.dossier_status = status
        self._log_dossier(dossier, status)
        return dossier, status

    def prepare_game_async(
        self, *, game_title: str, platform: str = "", version: str = "", session_id: str | int = "",
    ) -> GameResearchJob:
        game_id, canonical, _ = self.canonical_game(game_title)
        plan = self.plan_search(
            game_title=canonical,
            game_id=game_id,
            platform=platform,
            entity="core systems and spoiler-free premise",
            question_type="dossier",
            expected_fact_type="general_mechanics",
        )
        canonical_session = str(session_id or "pre_stream")
        with self._job_lock:
            matching = [
                job for job, _future in self._jobs.values()
                if job.game_id == game_id and job.session_id == canonical_session
                and job.mode == ResearchMode.PRE_STREAM_DOSSIER.value
            ]
        if matching:
            latest = max(matching, key=lambda item: (item.attempt, item.queued_at))
            if latest.status in {"queued", "running", "completed"}:
                return latest
            if (
                latest.attempt >= self.max_attempts
                or not latest.next_retry_at
                or self.now_fn() < latest.next_retry_at
            ):
                return latest
            attempt = latest.attempt + 1
        else:
            attempt = 1
        job = self.queue_research(
            plan,
            progress=None,
            scene_id="pre_stream",
            ttl_seconds=300,
            mode=ResearchMode.PRE_STREAM_DOSSIER.value,
            session_id=canonical_session,
            attempt=attempt,
            timeout_seconds=self.dossier_timeout_seconds,
            metadata={"game_title": canonical, "platform": platform, "version": version, "build_dossier": True},
        )
        return job

    def plan_search(
        self,
        *,
        game_title: str,
        game_id: str,
        entity: str,
        question_type: str,
        expected_fact_type: str,
        platform: str = "",
        version: str = "",
        owner_uncertainty: str = "",
        spoiler_limit: str = "strict",
    ) -> GameSearchPlan:
        pieces = [str(game_title or "").strip()]
        if platform:
            pieces.append(str(platform).strip())
        if version:
            pieces.append(str(version).strip())
        if entity:
            pieces.append(str(entity).strip()[:120])
        if owner_uncertainty:
            pieces.append(str(owner_uncertainty).strip()[:160])
        pieces.append("spoiler-safe no future story information")
        query = " ".join(piece for piece in pieces if piece)
        key = _cache_key(game_id, query, expected_fact_type)
        plan = GameSearchPlan(query, _slug(game_id), str(entity or ""), question_type, spoiler_limit, expected_fact_type, key)
        print(
            "[HEBE][GAME_SEARCH_PLAN] "
            f"game={plan.game_id} mode={question_type} entity={plan.entity!r} "
            f"spoiler_limit={plan.spoiler_limit} cache_key={plan.cache_key}",
            flush=True,
        )
        return plan

    def research(
        self,
        plan: GameSearchPlan,
        *,
        progress: GameProgressState | None,
        allow_cache: bool = True,
        timeout_seconds: float | None = None,
    ) -> list[RetrievedGameFact]:
        facts, _sources, cache_hit = self._retrieve(
            plan, progress=progress, allow_cache=allow_cache,
            timeout_seconds=timeout_seconds or self.contextual_timeout_seconds,
        )
        self.diagnostics.lookup_used = True
        if cache_hit:
            self.diagnostics.cache_hits += 1
        return facts

    def queue_research(
        self,
        plan: GameSearchPlan,
        *,
        progress: GameProgressState | None,
        scene_id: str,
        ttl_seconds: float = 20.0,
        mode: ResearchMode | str = ResearchMode.CONTEXTUAL_LOOKUP.value,
        session_id: str | int = "",
        attempt: int = 1,
        timeout_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GameResearchJob:
        with self._job_lock:
            for existing_job, existing_future in self._jobs.values():
                if (
                    existing_job.plan.cache_key == plan.cache_key
                    and existing_job.session_id == str(session_id or "")
                    and existing_job.mode == str(mode)
                    and not existing_future.done() and existing_job.status in {"queued", "running"}
                ):
                    return existing_job
        now = self.now_fn()
        job = GameResearchJob(
            job_id=f"research_{uuid.uuid4().hex}",
            plan=plan,
            scene_id=scene_id,
            session_id=str(session_id or ""),
            game_id=plan.game_id,
            mode=str(mode),
            attempt=max(1, int(attempt)),
            queued_at=now,
            timeout_seconds=float(timeout_seconds or (
                self.dossier_timeout_seconds if str(mode) == ResearchMode.PRE_STREAM_DOSSIER.value
                else self.contextual_timeout_seconds
            )),
            expires_at=now + max(0.1, ttl_seconds),
            metadata=dict(metadata or {}),
        )

        def work() -> list[RetrievedGameFact]:
            job.status = "running"
            job.started_at = self.now_fn()
            print(f"[HEBE][GAME_RESEARCH_JOB] status=running mode={job.mode} job_id={job.job_id} cache_key={plan.cache_key}", flush=True)
            return self.research(plan, progress=progress, timeout_seconds=job.timeout_seconds)

        future = self._executor.submit(work)
        with self._job_lock:
            self._jobs[job.job_id] = (job, future)
        self.diagnostics.active_research_job = asdict(job)
        print(f"[HEBE][GAME_RESEARCH_JOB] status=queued mode={job.mode} job_id={job.job_id} cache_key={plan.cache_key}", flush=True)
        return job

    def collect_job(self, job_id: str, *, scene_still_current: bool = True) -> tuple[GameResearchJob, list[RetrievedGameFact]]:
        with self._job_lock:
            pair = self._jobs.get(job_id)
        if pair is None:
            raise KeyError(job_id)
        job, future = pair
        if job.status in {"completed", "failed", "cancelled", "stale"}:
            return job, []
        if not future.done():
            now = self.now_fn()
            if job.status == "running" and now - job.started_at >= job.timeout_seconds:
                job.status = "failed"
                job.completed_at = now
                job.failure_reason = f"provider_timeout: exceeded {job.timeout_seconds:.1f}s"
                job.error = job.failure_reason
                if job.attempt < self.max_attempts:
                    job.next_retry_at = now + self.retry_base_seconds * (2 ** (job.attempt - 1))
                future.cancel()
                if job.mode == ResearchMode.PRE_STREAM_DOSSIER.value:
                    self.diagnostics.dossier_status = "failed"
                    self.diagnostics.current_comment_mode = "contextual_reaction"
                self.diagnostics.active_research_job = asdict(job)
                print(
                    f"[HEBE][GAME_RESEARCH_JOB] status=failed mode={job.mode} job_id={job.job_id} "
                    f"error=TimeoutError reason={job.failure_reason}",
                    flush=True,
                )
            return job, []
        if self.now_fn() > job.expires_at or not scene_still_current:
            job.status = "stale"
            future.cancel()
            print(f"[HEBE][GAME_RESEARCH_JOB] status=stale mode={job.mode} job_id={job.job_id}", flush=True)
            return job, []
        try:
            facts = future.result()
        except Exception as exc:
            job.status = "failed"
            job.completed_at = self.now_fn()
            job.failure_reason = f"{type(exc).__name__}: {exc}"
            job.error = job.failure_reason
            if job.attempt < self.max_attempts and _is_transient_research_failure(job.failure_reason):
                job.next_retry_at = job.completed_at + self.retry_base_seconds * (2 ** (job.attempt - 1))
            if job.mode == ResearchMode.PRE_STREAM_DOSSIER.value:
                self.diagnostics.dossier_status = "failed"
                self.diagnostics.current_comment_mode = "contextual_reaction"
            print(
                f"[HEBE][GAME_RESEARCH_JOB] status=failed mode={job.mode} job_id={job.job_id} "
                f"error={type(exc).__name__} reason={str(exc) or type(exc).__name__}",
                flush=True,
            )
            return job, []
        job.status = "completed"
        job.completed_at = self.now_fn()
        job.fact_ids = [fact.fact_id for fact in facts]
        if job.metadata.get("build_dossier"):
            title = str(job.metadata.get("game_title") or job.plan.game_id)
            existing = self.store.get_dossier(job.plan.game_id)
            dossier = self._build_dossier(
                title,
                job.plan.game_id,
                [title],
                str(job.metadata.get("platform") or ""),
                str(job.metadata.get("version") or ""),
                facts,
                [],
                existing,
            )
            if self._dossier_sufficient(dossier):
                self.store.save_dossier(dossier)
                self.diagnostics.dossier_status = "ready" if dossier.confirmed_general_mechanics else "partial"
                self._log_dossier(dossier, self.diagnostics.dossier_status)
            else:
                self.diagnostics.dossier_status = "insufficient"
                self.diagnostics.current_comment_mode = "contextual_reaction"
                self._log_dossier(existing, "insufficient", game=title)
        self.diagnostics.active_research_job = asdict(job)
        print(
            f"[HEBE][GAME_RESEARCH_JOB] status=completed mode={job.mode} job_id={job.job_id} "
            f"accepted_fact_count={len([fact for fact in facts if fact.usable_for_comment])} "
            f"source_count={len({fact.source_location for fact in facts if fact.source_location})}",
            flush=True,
        )
        return job, facts

    def retry_due_jobs(self) -> list[GameResearchJob]:
        """Queue due retries without ever waiting for provider I/O on the cognition thread."""
        now = self.now_fn()
        with self._job_lock:
            due = [
                job for job, _future in self._jobs.values()
                if job.status == "failed" and job.next_retry_at and job.next_retry_at <= now
                and job.attempt < self.max_attempts
            ]
        queued: list[GameResearchJob] = []
        for job in due:
            # Mark the failed attempt consumed before queueing to make repeated ticks idempotent.
            job.next_retry_at = 0.0
            queued.append(self.queue_research(
                job.plan,
                progress=None,
                scene_id=job.scene_id,
                ttl_seconds=max(0.1, job.expires_at - job.queued_at),
                mode=job.mode,
                session_id=job.session_id,
                attempt=job.attempt + 1,
                timeout_seconds=job.timeout_seconds,
                metadata=job.metadata,
            ))
        return queued

    def cancel_job(self, job_id: str) -> bool:
        with self._job_lock:
            pair = self._jobs.get(job_id)
        if pair is None:
            return False
        job, future = pair
        cancelled = future.cancel()
        job.status = "cancelled"
        print(f"[HEBE][GAME_RESEARCH_JOB] status=cancelled job_id={job.job_id}", flush=True)
        return cancelled

    def record_comment_provenance(
        self,
        *,
        comment_id: str,
        contract: CommentKnowledgeContract,
        advice_mode: GameAssistanceMode | str,
    ) -> dict[str, Any]:
        fact_ids = [fact.fact_id for fact in contract.game_facts]
        self.diagnostics.lookup_used = bool(fact_ids)
        self.diagnostics.current_comment_fact_ids = fact_ids
        self.diagnostics.current_comment_mode = contract.contribution_mode
        self.diagnostics.current_comment_provenance = list(contract.source_provenance)
        payload = {
            "comment_id": comment_id,
            "mode": contract.contribution_mode,
            "scene_fact_ids": list(contract.scene_fact_ids) or [hashlib.sha256(item.encode("utf-8")).hexdigest()[:10] for item in contract.scene_evidence],
            "game_fact_ids": fact_ids,
            "lookup_used": bool(fact_ids),
            "spoiler_guard": "passed" if not contract.forbidden_claims else "filtered",
            "advice_mode": GameAssistanceMode(str(advice_mode)).value,
        }
        print(
            "[HEBE][COMMENT_PROVENANCE] "
            + " ".join(f"{key}={value}" for key, value in payload.items()),
            flush=True,
        )
        return payload

    def record_final_comment(
        self,
        *,
        comment_id: str,
        text: str,
        game: str,
        scene_evidence: list[str] | None = None,
        scene_fact_ids: list[str] | None = None,
        advice_mode: GameAssistanceMode | str = GameAssistanceMode.MECHANICS_WITHOUT_SOLUTIONS,
    ) -> dict[str, Any]:
        dossier = self.store.get_dossier(game)
        normalized_text_tokens = set(_norm(text).split())
        fact_ids: list[str] = []
        if dossier is not None:
            for source in dossier.sources:
                meaningful = {
                    token for token in _norm(str(source.get("claim") or "")).split()
                    if len(token) >= 4
                }
                overlap_required = min(2, len(meaningful))
                if meaningful and len(meaningful & normalized_text_tokens) >= overlap_required:
                    fact_id = str(source.get("fact_id") or "")
                    if fact_id:
                        fact_ids.append(fact_id)
        allowed_advice, advice_reason = self.advice_guard.allow(text, mode=advice_mode)
        payload = {
            "comment_id": comment_id,
            "mode": "informed_observation" if fact_ids else "contextual_reaction",
            "scene_fact_ids": list(dict.fromkeys(scene_fact_ids or [])) or [
                hashlib.sha256(item.encode("utf-8")).hexdigest()[:10]
                for item in (scene_evidence or []) if item
            ],
            "game_fact_ids": list(dict.fromkeys(fact_ids)),
            "lookup_used": bool(fact_ids),
            "spoiler_guard": "passed",
            "advice_mode": GameAssistanceMode(str(advice_mode)).value,
            "advice_allowed": allowed_advice,
            "advice_reason": advice_reason,
        }
        self.diagnostics.lookup_used = bool(fact_ids)
        self.diagnostics.current_comment_fact_ids = payload["game_fact_ids"]
        self.diagnostics.current_comment_mode = payload["mode"]
        selected_fact_ids = set(payload["game_fact_ids"])
        self.diagnostics.current_comment_provenance = [
            dict(source)
            for source in (dossier.sources if dossier is not None else [])
            if str(source.get("fact_id") or "") in selected_fact_ids
        ]
        print(
            "[HEBE][COMMENT_PROVENANCE] "
            f"comment_id={comment_id} mode={payload['mode']} "
            f"scene_fact_ids={payload['scene_fact_ids']} game_fact_ids={payload['game_fact_ids']} "
            f"lookup_used={str(payload['lookup_used']).lower()} spoiler_guard={payload['spoiler_guard']} "
            f"advice_mode={payload['advice_mode']}",
            flush=True,
        )
        return payload

    def debug_snapshot(self) -> dict[str, Any]:
        current = self.diagnostics.current_game
        self.diagnostics.unresolved_knowledge_gaps = [gap.term for gap in self.store.unresolved_gaps(current)] if current else []
        return asdict(self.diagnostics)

    def _retrieve(
        self,
        plan: GameSearchPlan,
        *,
        progress: GameProgressState | None,
        allow_cache: bool,
        timeout_seconds: float | None = None,
    ) -> tuple[list[RetrievedGameFact], list[dict[str, Any]], bool]:
        self.diagnostics.last_query = plan.query
        if allow_cache:
            cached = self.store.get_cached_facts(plan.cache_key, now=self.now_fn())
            if cached is not None:
                checked = self._recheck_progress(cached, progress)
                return checked, [], True
        if self.provider is None or not getattr(self.provider, "available", True):
            raise RuntimeError("research_provider_missing")
        constraints = {
            "spoiler_limit": plan.spoiler_limit,
            "expected_fact_type": plan.expected_fact_type,
            "entity": plan.entity,
            "strict_first_playthrough": progress is None or progress.spoiler_policy == "strict",
        }
        try:
            rows = self.provider.search(
                plan.query, constraints,
                timeout=float(timeout_seconds or self.contextual_timeout_seconds),
            )
        except TypeError:
            rows = self.provider.search(plan.query)
        if not isinstance(rows, list):
            rows = []
        sources = [dict(row) for row in rows if isinstance(row, dict)]
        facts = self._normalize_facts(plan, sources, progress)
        for fact in facts:
            print(
                "[HEBE][RETRIEVED_GAME_FACT] "
                f"fact_id={fact.fact_id} confidence={fact.confidence:.3f} "
                f"usable_for_comment={str(fact.usable_for_comment).lower()} "
                f"source={fact.source_location or 'none'}",
                flush=True,
            )
        self.store.save_cache(plan, sources, facts, ttl_seconds=self.cache_ttl_seconds, now=self.now_fn())
        self.diagnostics.sources_found = len(sources)
        self.diagnostics.facts_accepted = [fact.fact_id for fact in facts if fact.usable_for_comment]
        self.diagnostics.facts_rejected = [fact.fact_id for fact in facts if not fact.usable_for_comment]
        return facts, sources, False

    def _normalize_facts(
        self,
        plan: GameSearchPlan,
        rows: list[dict[str, Any]],
        progress: GameProgressState | None,
    ) -> list[RetrievedGameFact]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            claim = str(row.get("claim") or row.get("fact") or row.get("summary") or "").strip()
            if not claim:
                continue
            grouped.setdefault(_norm(claim), []).append(row)
        facts: list[RetrievedGameFact] = []
        conflict_groups = {
            str(row.get("conflict_group"))
            for row in rows
            if row.get("conflict_group") and (row.get("conflicting") or row.get("conflict"))
        }
        for claim_key, evidence in grouped.items():
            row = evidence[0]
            claim = str(row.get("claim") or row.get("fact") or row.get("summary") or "").strip()
            source_locations = {
                str(item.get("url") or item.get("source_location") or item.get("source") or "").strip()
                for item in evidence
                if str(item.get("url") or item.get("source_location") or item.get("source") or "").strip()
            }
            source_location = sorted(source_locations)[0] if source_locations else ""
            excerpt = str(row.get("exact_supporting_excerpt_internal") or row.get("excerpt") or row.get("content") or row.get("snippet") or "").strip()
            snippet_only = bool(row.get("snippet")) and not any(row.get(key) for key in ("excerpt", "content", "exact_supporting_excerpt_internal"))
            confidence = min(1.0, float(row.get("confidence") or 0.65))
            if snippet_only:
                confidence = min(confidence, 0.45)
            classification = self._classify_fact(claim, row)
            conflict = bool(row.get("conflicting") or row.get("conflict") or (row.get("conflict_group") in conflict_groups))
            if conflict:
                classification = SpoilerClassification.UNCERTAIN_PROGRESS.value
                confidence = min(confidence, 0.35)
            compatibility = self._progress_compatibility(row, progress)
            corroboration = len(source_locations) if source_locations else len(evidence)
            fact_id = f"fact_{hashlib.sha256((plan.game_id + claim_key).encode('utf-8')).hexdigest()[:16]}"
            provisional = RetrievedGameFact(
                fact_id,
                claim,
                str(row.get("source_title") or row.get("title") or "unknown source"),
                source_location,
                _now_iso(self.now_fn()),
                confidence,
                max(1, corroboration),
                classification,
                compatibility,
                excerpt[:1200],
                False,
                False,
                str(row.get("source_type") or "unknown"),
                {
                    key: str(row.get(key) or "")
                    for key in ("required_area", "required_boss", "required_mechanic")
                    if str(row.get(key) or "").strip()
                },
            )
            if progress is None:
                spoiler_allowed = classification == SpoilerClassification.SAFE_GENERAL_MECHANIC.value
            else:
                spoiler_allowed = self.spoiler_firewall.evaluate(provisional, progress).allowed
            has_support = bool(source_location and excerpt and not snippet_only)
            provisional.usable_for_comment = bool(spoiler_allowed and has_support and confidence >= 0.6 and not conflict)
            provisional.usable_for_advice = bool(
                provisional.usable_for_comment
                and confidence >= 0.75
                and corroboration >= 2
                and classification in {
                    SpoilerClassification.SAFE_GENERAL_MECHANIC.value,
                    SpoilerClassification.SAFE_CURRENT_PROGRESS.value,
                }
            )
            facts.append(provisional)
        return facts

    def _recheck_progress(self, facts: list[RetrievedGameFact], progress: GameProgressState | None) -> list[RetrievedGameFact]:
        checked = []
        for original in facts:
            fact = RetrievedGameFact(**asdict(original))
            if progress is not None:
                if fact.progress_requirements:
                    fact.progress_compatibility = self._requirements_compatibility(fact.progress_requirements, progress)
                guard = self.spoiler_firewall.evaluate(fact, progress)
                fact.usable_for_comment = bool(fact.usable_for_comment and guard.allowed)
                fact.usable_for_advice = bool(fact.usable_for_advice and guard.allowed)
                if not guard.allowed:
                    self.diagnostics.last_spoiler_block = asdict(guard)
            checked.append(fact)
        return checked

    @staticmethod
    def _classify_fact(claim: str, row: dict[str, Any]) -> str:
        explicit = str(row.get("spoiler_classification") or "").strip()
        if explicit in {item.value for item in SpoilerClassification}:
            return explicit
        text = _norm(claim)
        if re.search(r"\b(?:ending|final scene|end boss|final boss)\b", text):
            return SpoilerClassification.ENDING_SPOILER.value
        if re.search(r"\b(?:secret identity|really is|traitor|dies|death|plot twist)\b", text):
            return SpoilerClassification.IDENTITY_SPOILER.value
        if re.search(r"\b(?:later chapter|future area|next location)\b", text):
            return SpoilerClassification.LOCATION_SPOILER.value
        if re.search(r"\b(?:later unlocks|future mechanic|eventually gains)\b", text):
            return SpoilerClassification.FUTURE_MECHANIC.value
        if row.get("current_progress"):
            return SpoilerClassification.SAFE_CURRENT_PROGRESS.value
        if row.get("general_mechanic") or re.search(r"\b(?:combat|menu|ui|resource|turn based|damage|guard|dodge|stamina|mana)\b", text):
            return SpoilerClassification.SAFE_GENERAL_MECHANIC.value
        return SpoilerClassification.UNCERTAIN_PROGRESS.value

    @staticmethod
    def _progress_compatibility(row: dict[str, Any], progress: GameProgressState | None) -> str:
        explicit = str(row.get("progress_compatibility") or "").strip()
        if explicit:
            return explicit
        if progress is None:
            return "unknown"
        required_area = _norm(str(row.get("required_area") or ""))
        required_boss = _norm(str(row.get("required_boss") or ""))
        required_mechanic = _norm(str(row.get("required_mechanic") or ""))
        if required_area and required_area != _norm(progress.current_area):
            return "future_or_unknown"
        if required_boss and required_boss not in {_norm(item) for item in progress.encountered_bosses}:
            return "future_or_unknown"
        if required_mechanic and required_mechanic not in {_norm(item) for item in progress.unlocked_mechanics}:
            return "future_or_unknown"
        return "compatible"

    @staticmethod
    def _requirements_compatibility(requirements: dict[str, str], progress: GameProgressState) -> str:
        required_area = _norm(requirements.get("required_area", ""))
        required_boss = _norm(requirements.get("required_boss", ""))
        required_mechanic = _norm(requirements.get("required_mechanic", ""))
        if required_area and required_area != _norm(progress.current_area):
            return "future_or_unknown"
        if required_boss and required_boss not in {_norm(item) for item in progress.encountered_bosses}:
            return "future_or_unknown"
        if required_mechanic and required_mechanic not in {_norm(item) for item in progress.unlocked_mechanics}:
            return "future_or_unknown"
        return "compatible"

    def _build_dossier(
        self,
        title: str,
        game_id: str,
        aliases: list[str],
        platform: str,
        version: str,
        facts: list[RetrievedGameFact],
        sources: list[dict[str, Any]],
        existing: GameDossier | None,
    ) -> GameDossier:
        safe = [fact for fact in facts if fact.spoiler_classification == SpoilerClassification.SAFE_GENERAL_MECHANIC.value and fact.usable_for_comment]
        source_rows = []
        for fact in safe:
            source_rows.append({
                "fact_id": fact.fact_id,
                "claim": fact.claim,
                "title": fact.source_title,
                "location": fact.source_location,
                "retrieved_at": fact.retrieved_at,
            })
        now = _now_iso(self.now_fn())
        dossier = existing or GameDossier(game_id=game_id, canonical_title=title, created_at=now)
        dossier.aliases = sorted(set([*dossier.aliases, *aliases, title]))
        dossier.platform = platform or dossier.platform
        dossier.version = version or dossier.version
        dossier.confirmed_general_mechanics = list(dict.fromkeys([
            *dossier.confirmed_general_mechanics,
            *(fact.claim for fact in safe),
        ]))
        dossier.core_combat_systems = list(dict.fromkeys([
            *dossier.core_combat_systems,
            *(fact.claim for fact in safe if re.search(r"\b(?:combat|damage|turn|guard|dodge)\b", _norm(fact.claim))),
        ]))
        dossier.sources = list({item["location"]: item for item in [*dossier.sources, *source_rows] if item.get("location")}.values())
        dossier.unsafe_story_topics = sorted(set([
            *dossier.unsafe_story_topics,
            "future story beats",
            "future party members",
            "future bosses and weaknesses",
            "secret identities",
            "future locations",
            "endings",
        ]))
        dossier.updated_at = now
        dossier.dossier_version = max(1, dossier.dossier_version + (1 if existing else 0))
        return dossier

    @staticmethod
    def _dossier_sufficient(dossier: GameDossier) -> bool:
        return bool(dossier.sources and (dossier.confirmed_general_mechanics or dossier.core_combat_systems or dossier.safe_gameplay_vocabulary))

    @staticmethod
    def _log_dossier(dossier: GameDossier | None, status: str, *, game: str = "") -> None:
        print(
            "[HEBE][GAME_DOSSIER] "
            f"game={game or (dossier.canonical_title if dossier else 'unknown')} status={status} "
            f"facts={len(dossier.confirmed_general_mechanics) if dossier else 0} "
            f"sources={len(dossier.sources) if dossier else 0}",
            flush=True,
        )

    def _log_provider_availability(self) -> None:
        print(
            "[HEBE][GAME_RESEARCH_PROVIDER] "
            f"provider={self.provider_name} "
            f"configured={str(self.provider_configured).lower()} "
            f"available={str(self.diagnostics.research_provider_available).lower()} "
            f"reason={self.diagnostics.research_provider_reason}",
            flush=True,
        )


def default_assistance_mode(progress: GameProgressState) -> GameAssistanceMode:
    if progress.playthrough_type == "first_playthrough":
        return GameAssistanceMode.MECHANICS_WITHOUT_SOLUTIONS
    return GameAssistanceMode.HINTS_ON_REQUEST


def _cache_key(game_id: str, scope: str, expected: str) -> str:
    normalized = f"{_slug(game_id)}|{_norm(scope)}|{_norm(expected)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _is_transient_research_failure(reason: str) -> bool:
    normalized = _norm(reason)
    permanent = {
        "auth", "unauthorized", "forbidden", "invalid api key", "api key missing",
        "research provider missing", "policy", "invalid profile", "bad request",
    }
    if any(marker in normalized for marker in permanent):
        return False
    transient = {
        "timeout", "timed out", "connection", "network", "temporar", "rate limit",
        "provider down", "429", "502", "503", "504", "service unavailable",
    }
    return any(marker in normalized for marker in transient)


__all__ = [
    "CommentKnowledgeContract",
    "CommentKnowledgePolicy",
    "GameAssistanceGuard",
    "GameAssistanceMode",
    "GameDossier",
    "GameIntelligenceDiagnostics",
    "GameIntelligenceStore",
    "GameProgressState",
    "GameProgressTracker",
    "GameResearchJob",
    "GameResearchProvider",
    "GameResearchService",
    "GameSearchPlan",
    "KnowledgeGap",
    "KnowledgeGapTracker",
    "ResearchMode",
    "ResearchTriggerDecision",
    "ResearchTriggerEngine",
    "RetrievedGameFact",
    "SpoilerClassification",
    "SpoilerFirewall",
    "SpoilerGuardResult",
    "default_assistance_mode",
]
