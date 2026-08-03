from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
import time
import unicodedata
from typing import Any


TOPIC_FAMILIES = {
    "industry_opinion", "personal_story", "media_discussion", "technology_discussion",
    "ethical_discussion", "game_design_opinion", "stream_meta_discussion",
    "emotional_reflection", "humorous_story", "gameplay_commentary", "unknown",
}

_STOPWORDS = {
    "a", "al", "algo", "como", "con", "de", "del", "el", "en", "es", "esa", "ese", "esto",
    "hay", "la", "las", "lo", "los", "mas", "me", "mi", "no", "para", "pero", "por", "que",
    "se", "si", "sin", "su", "un", "una", "y", "ya", "the", "and", "of", "to", "is", "it",
    "that", "this", "with", "for", "you", "we", "they", "are", "be", "on", "in",
}

_FAMILY_TERMS = {
    "industry_opinion": {"industria", "publisher", "publishers", "editora", "editoras", "sony", "nintendo", "xbox", "mercado", "lanzamiento", "release", "ventas", "consumidor", "consumidores"},
    "technology_discussion": {"tecnologia", "digital", "plataforma", "plataformas", "servidor", "software", "hardware", "internet", "ia", "algoritmo"},
    "ethical_discussion": {"etico", "etica", "derecho", "derechos", "justo", "injusto", "eleccion", "libertad", "control", "propiedad"},
    "game_design_opinion": {"diseno", "mecanica", "mecanicas", "balance", "dificultad", "nivel", "combate", "gameplay"},
    "stream_meta_discussion": {"stream", "directo", "chat", "canal", "audiencia", "microfono"},
    "media_discussion": {"pelicula", "serie", "libro", "musica", "videojuego", "videojuegos", "edicion", "ediciones", "fisico", "digital"},
    "personal_story": {"recuerdo", "cuando", "historia", "paso", "ocurrio", "antes", "familia", "amigo"},
    "emotional_reflection": {"siento", "sentir", "preocupa", "miedo", "triste", "alegra", "frustra", "emocion"},
    "humorous_story": {"gracia", "risa", "jaja", "absurdo", "ridiculo", "comico"},
    "gameplay_commentary": {"boss", "jefe", "ruta", "mapa", "hp", "vida", "cura", "arma", "inventario", "rng", "nivel", "combate"},
}

_ARGUMENT_TERMS = {"porque", "ya que", "significa", "implica", "problema", "creo", "pienso", "opino", "deberia", "pierde", "perder", "depende", "permite", "evita", "afecta"}


def normalize_discourse_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "").casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9_ ]+", " ", value).split())


def keywords_for(text: str) -> set[str]:
    words = {word for word in normalize_discourse_text(text).split() if len(word) > 2 and word not in _STOPWORDS}
    concepts = {
        "ownership_choice": {"propiedad", "poseer", "dueno", "reventa", "revender", "eleccion", "control", "consumidor", "consumidores"},
        "digital_distribution": {"digital", "digitales", "fisico", "fisica", "formato", "edicion", "ediciones", "plataforma", "plataformas", "lanzamiento", "lanzamientos", "editoras", "publisher", "mercado", "industria", "transicion"},
        "gameplay_state": {"boss", "jefe", "ruta", "mapa", "hp", "cura", "combate", "inventario", "rng"},
    }
    for concept, aliases in concepts.items():
        if words & aliases:
            words.add(concept)
    return words


@dataclass(slots=True)
class OwnerDiscourseFragment:
    text: str
    normalized_text: str
    timestamp: float
    confidence: float = 1.0
    language: str = "es"
    sentence_complete: bool = False
    keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DiscourseTopic:
    topic_id: str
    discourse_session_id: str
    label: str
    family: str
    fragments: list[OwnerDiscourseFragment]
    start_time: float
    last_fragment_time: float
    confidence: float
    topic_keywords: list[str]
    owner_stance: str
    supporting_points: list[str]
    rhetorical_questions: list[str]
    emotional_tone: str
    game_related: bool
    non_game_discussion: bool
    topic_stability: float
    novelty: float
    possible_contribution_value: float
    stable: bool

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.last_fragment_time - self.start_time)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["duration_seconds"] = self.duration_seconds
        return result


@dataclass(slots=True)
class OwnerDiscourseSession:
    discourse_session_id: str
    fragments: list[OwnerDiscourseFragment]
    start_time: float
    last_fragment_time: float
    language: str = "es"
    topic: DiscourseTopic | None = None


class DiscourseTopicTracker:
    def __init__(self, *, min_fragments: int = 3, min_duration_seconds: float = 25.0, topic_change_similarity: float = 0.16) -> None:
        self.min_fragments = int(min_fragments)
        self.min_duration_seconds = float(min_duration_seconds)
        self.topic_change_similarity = float(topic_change_similarity)

    def similarity(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(1, min(len(left), len(right)))

    def belongs_to_session(self, fragments: list[OwnerDiscourseFragment], candidate: OwnerDiscourseFragment) -> bool:
        existing = set().union(*(set(item.keywords) for item in fragments[-6:])) if fragments else set()
        candidate_words = set(candidate.keywords)
        if self.similarity(existing, candidate_words) >= self.topic_change_similarity:
            return True
        existing_family = self._family(existing)
        candidate_family = self._family(candidate_words)
        return existing_family != "unknown" and existing_family == candidate_family

    def build_topic(self, session: OwnerDiscourseSession) -> DiscourseTopic:
        fragments = list(session.fragments)
        all_words = [word for fragment in fragments for word in fragment.keywords]
        counts = {word: all_words.count(word) for word in set(all_words)}
        keywords = sorted(counts, key=lambda word: (-counts[word], word))[:30]
        family = self._family(set(keywords))
        coherent_pairs = 0
        for left, right in zip(fragments, fragments[1:]):
            if self.similarity(set(left.keywords), set(right.keywords)) >= 0.1 or self._family(set(left.keywords)) == self._family(set(right.keywords)) != "unknown":
                coherent_pairs += 1
        coherence = coherent_pairs / max(1, len(fragments) - 1)
        confidence = min(0.98, sum(item.confidence for item in fragments) / max(1, len(fragments)) * (0.65 + 0.35 * coherence))
        duration = max(0.0, session.last_fragment_time - session.start_time)
        has_argument = any(set(item.normalized_text.split()) & _ARGUMENT_TERMS for item in fragments)
        stable = bool(
            confidence >= 0.55
            and has_argument
            and ((len(fragments) >= self.min_fragments and coherence >= 0.45) or duration >= self.min_duration_seconds)
        )
        stance = self._stance(fragments)
        points = [item.text.strip() for item in fragments if set(item.normalized_text.split()) & _ARGUMENT_TERMS][:4]
        questions = [item.text.strip() for item in fragments if "?" in item.text or "¿" in item.text]
        emotional_tone = self._tone(" ".join(item.normalized_text for item in fragments))
        game_related = family in {"game_design_opinion", "gameplay_commentary"}
        # A topic segment keeps one identity while its vocabulary becomes richer.
        topic_hash = session.discourse_session_id.removeprefix("discourse_")
        label = self._label(family, keywords)
        stability = min(1.0, 0.3 * min(1.0, len(fragments) / self.min_fragments) + 0.35 * coherence + 0.35 * min(1.0, duration / max(1.0, self.min_duration_seconds)))
        return DiscourseTopic(
            topic_id=f"topic_{topic_hash}", discourse_session_id=session.discourse_session_id,
            label=label, family=family, fragments=fragments, start_time=session.start_time,
            last_fragment_time=session.last_fragment_time, confidence=round(confidence, 3),
            topic_keywords=keywords, owner_stance=stance, supporting_points=points,
            rhetorical_questions=questions, emotional_tone=emotional_tone,
            game_related=game_related, non_game_discussion=not game_related and family != "unknown",
            topic_stability=round(stability, 3), novelty=min(1.0, 0.45 + len(set(keywords)) / 30),
            possible_contribution_value=round(min(1.0, 0.25 + stability * 0.5 + (0.2 if has_argument else 0)), 3),
            stable=stable,
        )

    def _family(self, words: set[str]) -> str:
        scores = {family: len(words & terms) for family, terms in _FAMILY_TERMS.items()}
        if not any(scores.values()):
            return "unknown"
        family = max(scores, key=lambda item: (scores[item], item == "industry_opinion"))
        if scores.get("industry_opinion", 0) and scores.get("media_discussion", 0):
            return "industry_opinion"
        if scores.get("ethical_discussion", 0) >= 2 and scores.get("industry_opinion", 0) == 0:
            return "ethical_discussion"
        return family

    @staticmethod
    def _stance(fragments: list[OwnerDiscourseFragment]) -> str:
        text = " ".join(item.normalized_text for item in fragments)
        if re.search(r"\b(?:preocupa|problema|perder|pierde|malo|critico|contra|dependencia|obliga|danino)\b", text):
            return "critical_or_concerned"
        if re.search(r"\b(?:bien|mejor|bueno|favor|gusta|beneficia)\b", text):
            return "supportive_or_positive"
        return "analytical_or_mixed"

    @staticmethod
    def _tone(text: str) -> str:
        if re.search(r"\b(?:preocupa|miedo|triste|problema|pierde|perder)\b", text):
            return "concerned"
        if re.search(r"\b(?:jaja|gracia|absurdo|ridiculo)\b", text):
            return "playful"
        return "reflective"

    @staticmethod
    def _label(family: str, keywords: list[str]) -> str:
        return f"{family}: {' / '.join(keywords[:5])}" if keywords else family


class OwnerDiscourseBuffer:
    def __init__(self, *, tracker: DiscourseTopicTracker | None = None, session_gap_seconds: float = 90.0) -> None:
        self.tracker = tracker or DiscourseTopicTracker()
        self.session_gap_seconds = float(session_gap_seconds)
        self.sessions: list[OwnerDiscourseSession] = []
        self.current_session: OwnerDiscourseSession | None = None

    def reset_session(self) -> None:
        self.sessions.clear()
        self.current_session = None

    def add_fragment(self, text: str, *, timestamp: float | None = None, confidence: float = 1.0, language: str = "es") -> DiscourseTopic:
        ts = float(timestamp if timestamp is not None else time.time())
        normalized = normalize_discourse_text(text)
        fragment = OwnerDiscourseFragment(
            text=str(text or "").strip(), normalized_text=normalized, timestamp=ts,
            confidence=float(confidence), language=language,
            sentence_complete=bool(re.search(r"[.!?…]\s*$", str(text or "").strip())),
            keywords=sorted(keywords_for(normalized)),
        )
        session = self.current_session
        topic_changed = False
        if session is None or ts - session.last_fragment_time > self.session_gap_seconds:
            topic_changed = session is not None
            session = self._new_session(fragment)
        elif len(session.fragments) >= 2 and not self.tracker.belongs_to_session(session.fragments, fragment):
            topic_changed = True
            session = self._new_session(fragment)
        else:
            session.fragments.append(fragment)
            session.last_fragment_time = ts
        session.topic = self.tracker.build_topic(session)
        if topic_changed:
            previous = self.sessions[-2].topic.topic_id if len(self.sessions) > 1 and self.sessions[-2].topic else "none"
            print(f"[HEBE][DISCOURSE_TOPIC_CHANGE] previous={previous} current={session.topic.topic_id}", flush=True)
        print(f"[HEBE][DISCOURSE_TOPIC] topic_id={session.topic.topic_id} family={session.topic.family} stable={str(session.topic.stable).lower()} confidence={session.topic.confidence:.2f}", flush=True)
        print(f"[HEBE][DISCOURSE_STANCE] owner_stance={session.topic.owner_stance} supporting_points={session.topic.supporting_points!r}", flush=True)
        return session.topic

    def _new_session(self, fragment: OwnerDiscourseFragment) -> OwnerDiscourseSession:
        session = OwnerDiscourseSession(
            discourse_session_id=f"discourse_{hashlib.sha1(f'{fragment.timestamp}:{fragment.normalized_text}'.encode()).hexdigest()[:12]}",
            fragments=[fragment], start_time=fragment.timestamp, last_fragment_time=fragment.timestamp,
            language=fragment.language,
        )
        self.sessions.append(session)
        self.sessions = self.sessions[-12:]
        self.current_session = session
        return session

    def snapshot(self) -> dict[str, Any]:
        topic = self.current_session.topic if self.current_session else None
        return topic.to_dict() if topic else {}


@dataclass(slots=True)
class DiscourseContributionPlan:
    topic_id: str
    should_contribute: bool
    contribution_type: str
    contribution_value: float
    novelty_score: float
    confidence: float
    grounded_fragments: list[str]
    owner_stance: str
    proposed_claims: list[str]
    forbidden_claims: list[str]
    preferred_route: str
    wait_for_turn: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiscourseContributionPlanner:
    def plan(self, topic: DiscourseTopic) -> DiscourseContributionPlan:
        should = bool(topic.stable and topic.possible_contribution_value >= 0.62 and topic.novelty >= 0.5)
        contribution_type = "no_contribution"
        claims: list[str] = []
        if should:
            if topic.owner_stance == "critical_or_concerned":
                contribution_type = "consequence_or_implication"
                claims = ["connect the owner's concern to practical agency, access, or long-term dependence"]
            elif topic.emotional_tone == "playful":
                contribution_type = "playful_observation"
                claims = ["add one playful but relevant observation"]
            else:
                contribution_type = "synthesis"
                claims = ["join two grounded supporting points into one useful implication"]
        return DiscourseContributionPlan(
            topic_id=topic.topic_id, should_contribute=should, contribution_type=contribution_type,
            contribution_value=topic.possible_contribution_value, novelty_score=topic.novelty,
            confidence=topic.confidence, grounded_fragments=[item.text for item in topic.fragments[-6:]],
            owner_stance=topic.owner_stance, proposed_claims=claims,
            forbidden_claims=["unverified current news", "stale gameplay anchor", "unrelated factual assertion", "request for a follow-up"],
            preferred_route="stream_tts_reply", wait_for_turn=should,
            reason="stable_topic_adds_value" if should else "topic_not_ready_or_low_value",
        )


class DiscourseGroundingGuard:
    def evaluate(self, plan: DiscourseContributionPlan, topic: DiscourseTopic, *, candidate: str = "", introduced_current_facts: bool = False) -> dict[str, Any]:
        violations: list[str] = []
        if plan.topic_id != topic.topic_id:
            violations.append("stale_topic")
        if not plan.grounded_fragments:
            violations.append("missing_grounded_fragments")
        if introduced_current_facts:
            violations.append("unverified_current_fact")
        normalized = normalize_discourse_text(candidate)
        if candidate and topic.non_game_discussion and set(normalized.split()) & _FAMILY_TERMS["gameplay_commentary"]:
            violations.append("stale_gameplay_anchor")
        fragment_norms = {normalize_discourse_text(item) for item in plan.grounded_fragments}
        if normalized and normalized in fragment_norms:
            violations.append("paraphrase_only")
        passed = not violations and plan.should_contribute and bool(plan.proposed_claims)
        result = {"passed": passed, "violations": violations, "action": "allow" if passed else "suppress"}
        print(f"[HEBE][DISCOURSE_GROUNDING_GUARD] passed={str(passed).lower()} violations={violations!r} action={result['action']}", flush=True)
        return result


@dataclass(slots=True)
class StreamTurnState:
    owner_speaking: bool
    pause_seconds: float
    turn_available: bool
    reason: str
    sentence_complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StreamTurnDetector:
    def __init__(self, *, natural_pause_seconds: float = 3.5) -> None:
        self.natural_pause_seconds = float(natural_pause_seconds)
        self.last_owner_speech_at = 0.0
        self.last_sentence_complete = False
        self.owner_speaking = False

    def reset_session(self) -> None:
        self.last_owner_speech_at = 0.0
        self.last_sentence_complete = False
        self.owner_speaking = False

    def record_owner_fragment(self, text: str, *, timestamp: float | None = None) -> None:
        self.last_owner_speech_at = float(timestamp if timestamp is not None else time.time())
        self.last_sentence_complete = bool(re.search(r"[.!?…]\s*$", str(text or "").strip()))
        self.owner_speaking = True

    def detect(self, *, now: float | None = None, audio_active: bool = False, tts_speaking: bool = False, topic_ready: bool = True, combat_intense: bool = False) -> StreamTurnState:
        now = float(now if now is not None else time.time())
        pause = max(0.0, now - self.last_owner_speech_at) if self.last_owner_speech_at else 0.0
        owner_speaking = bool(audio_active or (self.owner_speaking and pause < self.natural_pause_seconds))
        available = bool(topic_ready and not owner_speaking and pause >= self.natural_pause_seconds and not tts_speaking and not combat_intense)
        reason = "natural_pause" if available else "owner_still_speaking" if owner_speaking else "tts_speaking" if tts_speaking else "intense_gameplay" if combat_intense else "topic_not_ready" if not topic_ready else "pause_too_short"
        self.owner_speaking = owner_speaking
        result = StreamTurnState(owner_speaking, round(pause, 3), available, reason, self.last_sentence_complete)
        print(f"[HEBE][STREAM_TURN] owner_speaking={str(owner_speaking).lower()} pause_seconds={pause:.2f} turn_available={str(available).lower()}", flush=True)
        return result


class DiscourseParticipationBudget:
    def __init__(self, *, min_between_seconds: float = 480.0, max_per_hour: int = 3, max_per_topic: int = 1) -> None:
        self.min_between_seconds = float(min_between_seconds)
        self.max_per_hour = int(max_per_hour)
        self.max_per_topic = int(max_per_topic)
        self.contributions: list[dict[str, Any]] = []

    def reset_session(self) -> None:
        self.contributions.clear()

    def allows(self, topic: DiscourseTopic, *, now: float | None = None, direct_priority: bool = False, event_type: str = "") -> dict[str, Any]:
        now = float(now if now is not None else time.time())
        if direct_priority or event_type in {"direct_mention", "cheer", "raid", "owner_command"}:
            return self._result(True, "priority_path_bypass", topic, now)
        recent = [item for item in self.contributions if now - float(item["timestamp"]) <= 3600]
        topic_count = sum(1 for item in recent if item["topic_id"] == topic.topic_id)
        if topic_count >= self.max_per_topic:
            return self._result(False, "one_contribution_per_topic", topic, now, recent)
        current_words = set(topic.topic_keywords)
        for item in recent:
            previous_words = set(item.get("topic_keywords") or [])
            overlap = len(current_words & previous_words) / max(1, min(len(current_words), len(previous_words)))
            if overlap >= 0.5:
                return self._result(False, "repeated_same_thesis", topic, now, recent)
        if len(recent) >= self.max_per_hour:
            return self._result(False, "hourly_discourse_limit", topic, now, recent)
        if recent and now - float(recent[-1]["timestamp"]) < self.min_between_seconds:
            return self._result(False, "discourse_soft_cooldown", topic, now, recent)
        return self._result(True, "allowed", topic, now, recent)

    def record(self, topic: DiscourseTopic, *, contribution_type: str, thesis_key: str = "", now: float | None = None) -> None:
        self.contributions.append({"topic_id": topic.topic_id, "timestamp": float(now if now is not None else time.time()),
                                   "contribution_type": contribution_type, "thesis_key": thesis_key,
                                   "topic_keywords": list(topic.topic_keywords)})
        self.contributions = self.contributions[-50:]

    def _result(self, allowed: bool, reason: str, topic: DiscourseTopic, now: float, recent: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        recent = recent if recent is not None else [item for item in self.contributions if now - float(item["timestamp"]) <= 3600]
        topic_count = sum(1 for item in recent if item["topic_id"] == topic.topic_id)
        result = {"allowed": allowed, "reason": reason, "topic_count": topic_count, "hourly_count": len(recent)}
        print(f"[HEBE][DISCOURSE_BUDGET] allowed={str(allowed).lower()} reason={reason} topic_count={topic_count} hourly_count={len(recent)}", flush=True)
        return result
