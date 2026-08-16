from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import hashlib
import math
import re
import time
import unicodedata
import uuid
from typing import Any

from app.cognitive.input_interpretation import InputInterpretation, InputSpeechAct
from app.core.persistent_logs import log_jsonl_event
from app.stream.behavior_constraints import (
    BehaviorConstraint,
    BehaviorConstraintOutputGuard,
    constraint_matches,
    persist_constraint,
)
from app.stream.behavior_constraint_store import BehaviorConstraintRepository


class FeedbackKind(StrEnum):
    EPISODIC_NEGATIVE = "episodic_negative"
    EPISODIC_POSITIVE = "episodic_positive"
    EXPLICIT_TEMPORARY_INSTRUCTION = "explicit_temporary_instruction"
    EXPLICIT_DURABLE_PREFERENCE = "explicit_durable_preference"
    CORRECTION_REVERSAL = "correction_reversal"


class ReferentProvenance(StrEnum):
    EXPLICIT_TEXT = "explicit_text"
    RECENT_HEBE_UTTERANCE = "recent_hebe_utterance"
    RECENT_TOPIC = "recent_topic"
    UNRESOLVED = "unresolved"


class AdaptationAction(StrEnum):
    ALLOW = "allow"
    DOWNRANK = "downrank"
    COOLDOWN = "cooldown"
    SUPPRESS = "suppress"


_GENERIC_REFERENTS = {
    "", "eso", "esa", "esto", "esta", "aquello", "response", "respuesta",
    "previous", "previous_hebe_utterance", "previous hebe utterance", "lo", "la", "broma", "comentario",
    "hacerla", "hacerlas", "hacerlo", "hacerlos", "seguirla", "seguirlas", "seguirlo", "seguirlos",
}
_STOP_WORDS = {
    "abre", "abrir", "dale", "dile", "decir", "diciendo", "diciendome", "hace",
    "hacer", "hagas", "mira", "mirar", "otra", "otra vez", "puto", "puta", "que",
    "con", "como", "para", "por", "una", "uno", "unos", "unas", "esa", "ese", "eso",
    "esta", "este", "esto", "del", "las", "los", "hay", "ninguna", "ningun", "todo",
    "rato", "stream", "lleva", "llevas", "quieres", "callar", "callate", "vuelvas",
    "volver", "tipo", "tema", "broma", "bromas", "comentario", "comentarios", "hoy",
    "ahora", "nuevo", "nueva", "again", "this", "that", "the", "about", "with",
}


def _normalize(value: str) -> str:
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", str(value or "").casefold())
        if not unicodedata.combining(ch)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def motif_terms(value: str) -> tuple[str, ...]:
    terms: list[str] = []
    for raw in _normalize(value).split():
        if raw in _STOP_WORDS or len(raw) < 3:
            continue
        term = raw[:-2] if len(raw) > 6 and raw.endswith(("es", "os", "as")) else raw[:-1] if len(raw) > 5 and raw.endswith("s") else raw
        if term not in terms:
            terms.append(term)
    return tuple(terms[:16])


def semantic_similarity(left: str | tuple[str, ...], right: str | tuple[str, ...]) -> float:
    a = set(left if isinstance(left, tuple) else motif_terms(left))
    b = set(right if isinstance(right, tuple) else motif_terms(right))
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if not overlap:
        return 0.0
    containment = overlap / max(1, min(len(a), len(b)))
    jaccard = overlap / max(1, len(a | b))
    return min(1.0, 0.7 * containment + 0.3 * jaccard)


def motif_id(terms: tuple[str, ...]) -> str:
    stable = " ".join(sorted(set(terms)))
    return "motif_" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class ResolvedReferent:
    text: str
    terms: tuple[str, ...]
    provenance: ReferentProvenance
    confidence: float
    topic: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.terms and self.provenance != ReferentProvenance.UNRESOLVED)


@dataclass(frozen=True, slots=True)
class FeedbackApplication:
    applied: bool
    kind: FeedbackKind | None
    referent: ResolvedReferent
    constraint_id: str = ""
    reason: str = ""
    fatigue: float = 0.0


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    action: AdaptationAction
    fatigue: float
    reason: str
    motif_id: str
    recent_uses: int
    negative_weight: float
    positive_weight: float
    constraint_id: str = ""
    score_multiplier: float = 1.0
    stage: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = self.action.value
        return value


class BehaviorAdaptationService:
    """Owns volatile owner-feedback adaptation for optional Hebe behavior.

    Recent utterances remain owned by StreamSpontaneityService. This service only
    stores feedback effects and consults BehaviorConstraint for explicit scopes.
    """

    NEGATIVE_HALF_LIFE_SEC = 30 * 60
    POSITIVE_HALF_LIFE_SEC = 20 * 60
    USE_WINDOW_SEC = 45 * 60

    def __init__(self, repository: BehaviorConstraintRepository | None = None) -> None:
        self.repository = repository
        self.output_guard = BehaviorConstraintOutputGuard()

    def load_durable_constraints(self, stream: Any) -> list[dict[str, Any]]:
        if self.repository is None:
            return []
        existing = [
            BehaviorConstraint.from_value(item)
            for item in list(getattr(stream, "active_behavior_blocks", []) or [])
            if isinstance(item, (dict, BehaviorConstraint))
        ]
        durable = self.repository.list_active()
        merged = [item for item in existing if item.scope != "durable"]
        merged.extend(durable)
        stream.active_behavior_blocks = [item.to_dict() for item in merged if item.active]
        return [item.to_dict() for item in durable]

    def active_constraints(self, stream: Any, *, now: float | None = None) -> list[dict[str, Any]]:
        now = time.time() if now is None else float(now)
        active = [
            item for item in (
                BehaviorConstraint.from_value(raw)
                for raw in list(getattr(stream, "active_behavior_blocks", []) or [])
                if isinstance(raw, (dict, BehaviorConstraint))
            )
            if item.active
            and item.status == "ACTIVE"
            and (not item.expires_at or item.expires_at > now)
        ]
        stream.active_behavior_blocks = [item.to_dict() for item in active]
        return list(stream.active_behavior_blocks)

    def register_explicit_constraint(
        self,
        stream: Any,
        constraint: BehaviorConstraint,
    ) -> dict[str, Any]:
        if constraint.authority != "owner" or constraint.explicitness != "explicit":
            raise ValueError("behavior_constraint_requires_explicit_owner_authority")
        if constraint.scope == "durable":
            if self.repository is None:
                raise RuntimeError("durable_constraint_store_unavailable")
            self.repository.save_durable(constraint)
        return persist_constraint(stream, constraint)

    def matching_explicit_constraint(
        self,
        stream: Any,
        *,
        behavior_family: str,
        recipient_login: str = "",
        requester_login: str = "",
        now: float | None = None,
    ) -> dict[str, Any] | None:
        if not behavior_family:
            return None
        return next((
            item for item in self.active_constraints(stream, now=now)
            if constraint_matches(
                item,
                behavior_family=behavior_family,
                recipient_login=recipient_login,
                requester_login=requester_login,
            )
        ), None)

    def apply_feedback(
        self,
        stream: Any,
        interpretation: InputInterpretation,
        *,
        now: float | None = None,
        recent_hebe_utterance: dict[str, Any] | str | None = None,
        source_event_id: str = "",
    ) -> FeedbackApplication:
        now = time.time() if now is None else float(now)
        unresolved = ResolvedReferent("", (), ReferentProvenance.UNRESOLVED, 0.0)
        if (
            interpretation.speech_act != InputSpeechAct.OWNER_FEEDBACK
            or interpretation.authority != "owner"
            or interpretation.feedback is None
        ):
            return FeedbackApplication(False, None, unresolved, reason="not_authoritative_owner_feedback")
        kind = self.classify(interpretation.feedback_text or interpretation.context_text, interpretation)
        self._log("feedback_received", {
            "kind": kind.value,
            "polarity": interpretation.feedback.polarity,
            "strength": interpretation.feedback.strength,
            "authority": interpretation.authority,
        })
        referent = self.resolve_referent(
            stream,
            interpretation,
            recent_hebe_utterance=recent_hebe_utterance,
            now=now,
        )
        self._log("referent_resolved", {
            "resolved": referent.resolved,
            "referent_provenance": referent.provenance.value,
            "referent_confidence": referent.confidence,
            "motif_id": motif_id(referent.terms) if referent.terms else "",
        })
        if not referent.resolved:
            result = FeedbackApplication(False, kind, referent, reason="referent_unresolved")
            self._remember_last(stream, result)
            return result
        if kind == FeedbackKind.EXPLICIT_DURABLE_PREFERENCE and (
            interpretation.confidence < 0.9
            or interpretation.feedback.explicitness != "explicit"
            or referent.confidence < 0.8
            or self._looks_ambiguous(interpretation.feedback_text)
        ):
            result = FeedbackApplication(False, kind, referent, reason="durable_evidence_insufficient")
            self._remember_last(stream, result)
            return result
        if kind == FeedbackKind.EXPLICIT_DURABLE_PREFERENCE and self.repository is None:
            result = FeedbackApplication(False, kind, referent, reason="durable_constraint_store_unavailable")
            self._remember_last(stream, result)
            return result

        constraint_id = ""
        if kind in {
            FeedbackKind.EXPLICIT_TEMPORARY_INSTRUCTION,
            FeedbackKind.EXPLICIT_DURABLE_PREFERENCE,
        }:
            constraint = self._create_constraint(
                interpretation,
                referent,
                scope="durable" if kind == FeedbackKind.EXPLICIT_DURABLE_PREFERENCE else "current_stream",
                now=now,
                source_event_id=source_event_id,
            )
            self.register_explicit_constraint(stream, constraint)
            constraint_id = constraint.id
            self._log("constraint_applied", {
                "constraint_id": constraint.id,
                "scope": constraint.scope,
                "motif_id": motif_id(referent.terms),
            })
        elif kind == FeedbackKind.CORRECTION_REVERSAL:
            constraint_id = self._reverse_constraints(
                stream, referent, source_event_id=source_event_id, now=now,
            )
            self._log("constraint_reversed", {
                "constraint_ids": constraint_id,
                "motif_id": motif_id(referent.terms),
            })

        state = self._state(stream)
        entries = list(state.get("entries") or [])
        matching = self._best_entry(entries, referent.terms)
        entry = matching
        if entry is None:
            entry = {
                "motif_id": motif_id(referent.terms),
                "referent_text": referent.text,
                "motif_terms": list(referent.terms),
                "negative_weight": 0.0,
                "positive_weight": 0.0,
                "negative_applications": 0,
                "positive_applications": 0,
                "created_at": now,
            }
            entries.append(entry)
        self._decay_entry(entry, now)
        strength = float(interpretation.feedback.strength or 0.0)
        if kind in {FeedbackKind.EPISODIC_NEGATIVE, FeedbackKind.EXPLICIT_TEMPORARY_INSTRUCTION, FeedbackKind.EXPLICIT_DURABLE_PREFERENCE}:
            entry["negative_applications"] = int(entry.get("negative_applications") or 0) + 1
            repeated_multiplier = min(1.6, 1.0 + 0.2 * (entry["negative_applications"] - 1))
            entry["negative_weight"] = min(2.0, float(entry.get("negative_weight") or 0.0) + strength * repeated_multiplier)
            entry["suppress_until"] = now + min(90 * 60, 8 * 60 * entry["negative_applications"] * max(0.7, strength))
        elif kind == FeedbackKind.EPISODIC_POSITIVE:
            entry["positive_applications"] = int(entry.get("positive_applications") or 0) + 1
            entry["positive_weight"] = min(0.8, float(entry.get("positive_weight") or 0.0) + strength * 0.35)
            entry["negative_weight"] = max(0.0, float(entry.get("negative_weight") or 0.0) - strength * 0.18)
        else:
            entry["negative_weight"] = max(0.0, float(entry.get("negative_weight") or 0.0) - max(0.45, strength * 0.7))
            entry["positive_weight"] = min(0.8, float(entry.get("positive_weight") or 0.0) + strength * 0.2)
            entry["suppress_until"] = 0.0
        entry.update({
            "updated_at": now,
            "provenance": referent.provenance.value,
            "last_kind": kind.value,
            "topic": referent.topic,
        })
        state["entries"] = entries[-50:]
        result = FeedbackApplication(
            True,
            kind,
            referent,
            constraint_id=constraint_id,
            reason="feedback_state_updated",
            fatigue=min(1.0, float(entry.get("negative_weight") or 0.0)),
        )
        self._remember_last(stream, result)
        self._log("feedback_applied", {
            "kind": kind.value,
            "motif_id": entry["motif_id"],
            "fatigue": result.fatigue,
            "constraint_id": constraint_id,
            "reason": result.reason,
        })
        return result

    def evaluate_candidate(
        self,
        stream: Any,
        text: str,
        *,
        topic: str = "",
        mode: str = "proactive",
        now: float | None = None,
    ) -> CandidateEvaluation:
        now = time.time() if now is None else float(now)
        terms = motif_terms(f"{text} {topic}")
        candidate_id = motif_id(terms) if terms else "motif_unresolved"
        if mode == "direct_response":
            return CandidateEvaluation(
                AdaptationAction.ALLOW, 0.0, "direct_required_response",
                candidate_id, 0, 0.0, 0.0, score_multiplier=1.0,
            )
        constraint = self._matching_constraint(stream, terms)
        if constraint is not None:
            result = CandidateEvaluation(
                AdaptationAction.SUPPRESS, 1.0, "explicit_behavior_constraint",
                candidate_id, 0, 1.0, 0.0, constraint.id, 0.0,
            )
            self._record_candidate(stream, result)
            return result

        recent_uses = 0
        use_score = 0.0
        for item in list(getattr(stream, "recent_idle_messages", []) or [])[-30:]:
            age = max(0.0, now - float(item.get("timestamp", 0.0) or 0.0))
            if age > self.USE_WINDOW_SEC:
                continue
            similarity = semantic_similarity(terms, motif_terms(f"{item.get('text', '')} {item.get('topic', '')}"))
            if similarity < 0.25:
                continue
            recent_uses += 1
            use_score += similarity * math.exp(-age / 1200.0)

        negative = 0.0
        positive = 0.0
        suppression_active = False
        state = self._state(stream)
        for entry in list(state.get("entries") or []):
            similarity = semantic_similarity(terms, tuple(entry.get("motif_terms") or ()))
            if similarity < 0.25:
                continue
            self._decay_entry(entry, now)
            negative += float(entry.get("negative_weight") or 0.0) * similarity
            positive += float(entry.get("positive_weight") or 0.0) * similarity
            suppression_active = suppression_active or now < float(entry.get("suppress_until") or 0.0)
        fatigue = min(1.0, use_score * 0.22 + negative * 0.72 - positive * 0.10)
        if suppression_active and negative >= 0.25 or fatigue >= 0.86:
            action = AdaptationAction.SUPPRESS
            reason = "negative_feedback_and_recent_repetition" if recent_uses else "negative_owner_feedback"
        elif fatigue >= 0.62:
            action = AdaptationAction.COOLDOWN
            reason = "motif_fatigue_cooldown"
        elif fatigue >= 0.30 or recent_uses and positive > 0.0:
            action = AdaptationAction.DOWNRANK
            reason = "motif_repetition_downrank"
        else:
            action = AdaptationAction.ALLOW
            reason = "no_material_motif_fatigue"
        multiplier = {
            AdaptationAction.ALLOW: 1.0,
            AdaptationAction.DOWNRANK: 0.45,
            AdaptationAction.COOLDOWN: 0.0,
            AdaptationAction.SUPPRESS: 0.0,
        }[action]
        result = CandidateEvaluation(
            action, fatigue, reason, candidate_id, recent_uses, negative, positive,
            score_multiplier=multiplier,
        )
        self._record_candidate(stream, result)
        return result

    def validate_generated_output(
        self,
        stream: Any,
        text: str,
        *,
        topic: str = "",
        mode: str = "proactive",
        now: float | None = None,
    ) -> CandidateEvaluation:
        """Validate final text against canonical active suppression only.

        Candidate ranking owns fatigue and recent-use scoring. This safety check
        does not rank again; it catches a generated text that drifted into an
        explicitly constrained or currently suppressed motif.
        """
        now = time.time() if now is None else float(now)
        terms = motif_terms(f"{text} {topic}")
        candidate_id = motif_id(terms) if terms else "motif_unresolved"
        if mode == "direct_response":
            return CandidateEvaluation(
                AdaptationAction.ALLOW, 0.0, "direct_required_response",
                candidate_id, 0, 0.0, 0.0, score_multiplier=1.0,
                stage="generated_output",
            )
        constraint = self._matching_constraint(stream, terms)
        if constraint is not None:
            result = CandidateEvaluation(
                AdaptationAction.SUPPRESS, 1.0, "generated_output_matches_constraint",
                candidate_id, 0, 1.0, 0.0, constraint.id, 0.0,
                "generated_output",
            )
            self._record_candidate(stream, result)
            return result
        for entry in list(self._state(stream).get("entries") or []):
            similarity = semantic_similarity(terms, tuple(entry.get("motif_terms") or ()))
            if similarity < 0.25:
                continue
            self._decay_entry(entry, now)
            negative = float(entry.get("negative_weight") or 0.0) * similarity
            if now < float(entry.get("suppress_until") or 0.0) and negative >= 0.25:
                result = CandidateEvaluation(
                    AdaptationAction.SUPPRESS, min(1.0, negative),
                    "generated_output_reincides_in_suppressed_motif",
                    candidate_id, 0, negative, 0.0, score_multiplier=0.0,
                    stage="generated_output",
                )
                self._record_candidate(stream, result)
                return result
        result = CandidateEvaluation(
            AdaptationAction.ALLOW, 0.0, "generated_output_matches_selected_policy",
            candidate_id, 0, 0.0, 0.0, score_multiplier=1.0,
            stage="generated_output",
        )
        self._record_candidate(stream, result)
        return result

    def validate_constraint_output(
        self,
        stream: Any,
        *,
        intended_recipient: str,
        generated_response: str,
        source_viewer: str = "",
        speech_act: str = "",
        scene_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.output_guard.evaluate(
            list(getattr(stream, "active_behavior_blocks", []) or []),
            intended_recipient=intended_recipient,
            generated_response=generated_response,
            source_viewer=source_viewer,
            speech_act=speech_act,
            scene_context=scene_context,
        )

    def resolve_referent(
        self,
        stream: Any,
        interpretation: InputInterpretation,
        *,
        recent_hebe_utterance: dict[str, Any] | str | None = None,
        now: float | None = None,
    ) -> ResolvedReferent:
        now = time.time() if now is None else float(now)
        feedback = interpretation.feedback
        if feedback is None:
            return ResolvedReferent("", (), ReferentProvenance.UNRESOLVED, 0.0)
        explicit = str(feedback.referent or "").strip()
        explicit_terms = motif_terms(explicit)
        explicit_named = _normalize(explicit) not in _GENERIC_REFERENTS and bool(explicit_terms)

        recent: list[tuple[str, str, float, str]] = []
        for item in list(getattr(stream, "recent_idle_messages", []) or [])[-12:]:
            age = max(0.0, now - float(item.get("timestamp", now) or now))
            if age <= self.USE_WINDOW_SEC and item.get("text"):
                recent.append((str(item.get("text")), ReferentProvenance.RECENT_HEBE_UTTERANCE.value, age, str(item.get("topic") or "")))
        utterance = recent_hebe_utterance or getattr(stream, "last_hebe_utterance", None)
        if isinstance(utterance, dict) and utterance.get("text"):
            recent.append((str(utterance.get("text")), ReferentProvenance.RECENT_HEBE_UTTERANCE.value, 0.0, str(utterance.get("topic") or "")))
        elif isinstance(utterance, str) and utterance.strip():
            recent.append((utterance, ReferentProvenance.RECENT_HEBE_UTTERANCE.value, 0.0, ""))
        query = explicit if explicit_named else interpretation.feedback_text
        best: tuple[float, str, str, str] | None = None
        for text, provenance, age, topic in recent:
            similarity = semantic_similarity(query, text)
            recency = math.exp(-age / 1200.0)
            score = similarity * 0.85 + recency * 0.15
            if best is None or score > best[0]:
                best = (score, text, provenance, topic)
        if best and (best[0] >= 0.30 or not explicit_named):
            terms = motif_terms(best[1])
            provenance = ReferentProvenance(best[2])
            return ResolvedReferent(best[1], terms, provenance, min(0.98, best[0]), best[3])
        if explicit_named:
            return ResolvedReferent(explicit, explicit_terms, ReferentProvenance.EXPLICIT_TEXT, 0.9)
        topic = str(getattr(stream, "current_discourse_topic", "") or "")
        if topic and motif_terms(topic):
            return ResolvedReferent(topic, motif_terms(topic), ReferentProvenance.RECENT_TOPIC, 0.62, topic)
        return ResolvedReferent("", (), ReferentProvenance.UNRESOLVED, 0.0)

    @staticmethod
    def classify(text: str, interpretation: InputInterpretation) -> FeedbackKind:
        normalized = _normalize(text)
        if re.search(r"\b(?:puedes|puede)\s+(?:volver|seguir)|\bya\s+puedes|\ben realidad\b.*\b(?:seguir|volver)\b", normalized):
            return FeedbackKind.CORRECTION_REVERSAL
        if re.search(r"\b(?:hoy|por hoy|este stream|esta sesion)\b", normalized) and re.search(r"\b(?:no|deja|evita)\b", normalized):
            return FeedbackKind.EXPLICIT_TEMPORARY_INSTRUCTION
        if re.search(r"\b(?:nunca mas|no quiero que vuelvas|no vuelvas a hacer ese tipo|no me recomiendes nunca)\b", normalized):
            return FeedbackKind.EXPLICIT_DURABLE_PREFERENCE
        if interpretation.feedback and interpretation.feedback.polarity == "positive":
            return FeedbackKind.EPISODIC_POSITIVE
        return FeedbackKind.EPISODIC_NEGATIVE

    def _create_constraint(
        self,
        interpretation: InputInterpretation,
        referent: ResolvedReferent,
        *,
        scope: str,
        now: float,
        source_event_id: str,
    ) -> BehaviorConstraint:
        key = motif_id(referent.terms)
        return BehaviorConstraint(
            id=f"constraint_{uuid.uuid4().hex[:12]}",
            actor="Hebe",
            behavior_family="semantic_motif",
            behavior_variants=[f"motif:{key}", *referent.terms],
            recipient_scope="everyone",
            source_event_id=source_event_id,
            source_text=interpretation.feedback_text if scope != "durable" else "",
            created_by="owner",
            authority=interpretation.authority,
            priority="owner_absolute",
            scope=scope,
            explicitness=interpretation.feedback.explicitness if interpretation.feedback else "",
            confidence=interpretation.confidence,
            created_at=now,
            reason=f"explicit owner {scope} motif constraint",
        )

    @staticmethod
    def _state(stream: Any) -> dict[str, Any]:
        state = getattr(stream, "behavior_adaptation_state", None)
        if not isinstance(state, dict):
            state = {"entries": []}
            setattr(stream, "behavior_adaptation_state", state)
        state.setdefault("entries", [])
        return state

    @staticmethod
    def _decay_entry(entry: dict[str, Any], now: float) -> None:
        updated = float(entry.get("updated_at") or entry.get("created_at") or now)
        elapsed = max(0.0, now - updated)
        if elapsed:
            entry["negative_weight"] = float(entry.get("negative_weight") or 0.0) * math.pow(0.5, elapsed / BehaviorAdaptationService.NEGATIVE_HALF_LIFE_SEC)
            entry["positive_weight"] = float(entry.get("positive_weight") or 0.0) * math.pow(0.5, elapsed / BehaviorAdaptationService.POSITIVE_HALF_LIFE_SEC)
            entry["updated_at"] = now

    @staticmethod
    def _best_entry(entries: list[dict[str, Any]], terms: tuple[str, ...]) -> dict[str, Any] | None:
        matches = [(semantic_similarity(terms, tuple(item.get("motif_terms") or ())), item) for item in entries]
        matches = [item for item in matches if item[0] >= 0.25]
        return max(matches, default=(0.0, None), key=lambda item: item[0])[1]

    @staticmethod
    def _matching_constraint(stream: Any, terms: tuple[str, ...]) -> BehaviorConstraint | None:
        for raw in list(getattr(stream, "active_behavior_blocks", []) or []):
            if not isinstance(raw, (dict, BehaviorConstraint)):
                continue
            item = BehaviorConstraint.from_value(raw)
            if not item.active or item.behavior_family != "semantic_motif":
                continue
            variants = tuple(value for value in item.behavior_variants if not value.startswith("motif:"))
            if semantic_similarity(terms, variants) >= 0.25:
                return item
        return None

    def _reverse_constraints(
        self,
        stream: Any,
        referent: ResolvedReferent,
        *,
        source_event_id: str,
        now: float,
    ) -> str:
        reversed_ids: list[str] = []
        kept: list[dict[str, Any]] = []
        for raw in list(getattr(stream, "active_behavior_blocks", []) or []):
            if not isinstance(raw, (dict, BehaviorConstraint)):
                continue
            item = BehaviorConstraint.from_value(raw)
            variants = tuple(value for value in item.behavior_variants if not value.startswith("motif:"))
            if item.behavior_family == "semantic_motif" and semantic_similarity(referent.terms, variants) >= 0.25:
                reversed_ids.append(item.id)
                if item.scope == "durable" and self.repository is not None:
                    self.repository.retire(
                        item.id,
                        reason="explicit_owner_reversal",
                        source_event_id=source_event_id,
                        authority="owner",
                        now=now,
                    )
                continue
            kept.append(item.to_dict())
        stream.active_behavior_blocks = kept
        return ",".join(reversed_ids)

    @staticmethod
    def _looks_ambiguous(text: str) -> bool:
        normalized = _normalize(text)
        return bool(re.search(r"\b(?:si claro|seguro|supongo|quizas|a lo mejor)\b", normalized))

    def _record_candidate(self, stream: Any, result: CandidateEvaluation) -> None:
        stream.last_behavior_adaptation_decision = result.to_dict()
        self._log("candidate_evaluated", result.to_dict())

    @staticmethod
    def _remember_last(stream: Any, result: FeedbackApplication) -> None:
        stream.last_feedback_application = {
            "applied": result.applied,
            "kind": result.kind.value if result.kind else "",
            "referent": result.referent.text,
            "referent_provenance": result.referent.provenance.value,
            "referent_confidence": result.referent.confidence,
            "constraint_id": result.constraint_id,
            "reason": result.reason,
            "fatigue": result.fatigue,
        }

    @staticmethod
    def _log(event: str, payload: dict[str, Any]) -> None:
        value = {"event": event, **payload}
        print("[HEBE][BEHAVIOR_ADAPTATION] " + " ".join(f"{key}={item!r}" for key, item in value.items()), flush=True)
        log_jsonl_event("behavior_adaptation", value)
__all__ = [
    "AdaptationAction",
    "BehaviorAdaptationService",
    "CandidateEvaluation",
    "FeedbackApplication",
    "FeedbackKind",
    "ReferentProvenance",
    "ResolvedReferent",
    "motif_id",
    "motif_terms",
    "semantic_similarity",
]
