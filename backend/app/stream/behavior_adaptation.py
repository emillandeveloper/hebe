from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
from app.stream.behavior_observability import (
    BehaviorObservability,
    GLOBAL_BEHAVIOR_OBSERVABILITY,
)


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


def semantic_similarity_evidence(
    left: str | tuple[str, ...],
    right: str | tuple[str, ...],
) -> dict[str, Any]:
    """Explain the existing lexical score without participating in policy."""
    left_terms = tuple(left if isinstance(left, tuple) else motif_terms(left))
    right_terms = tuple(right if isinstance(right, tuple) else motif_terms(right))
    a = set(left_terms)
    b = set(right_terms)
    shared = sorted(a & b)
    overlap = len(shared)
    containment = overlap / max(1, min(len(a), len(b))) if a and b else 0.0
    jaccard = overlap / max(1, len(a | b)) if a and b else 0.0
    score = semantic_similarity(left_terms, right_terms)
    return {
        "left_terms": list(left_terms),
        "right_terms": list(right_terms),
        "shared_terms": shared,
        "containment": round(containment, 6),
        "jaccard": round(jaccard, 6),
        "similarity": round(score, 6),
        "matched": score >= 0.25,
    }


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
    trace_id: str = ""
    candidate_id: str = ""

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

    def __init__(
        self,
        repository: BehaviorConstraintRepository | None = None,
        *,
        observability: BehaviorObservability | None = None,
    ) -> None:
        self.repository = repository
        self.output_guard = BehaviorConstraintOutputGuard()
        self.observability = observability or GLOBAL_BEHAVIOR_OBSERVABILITY

    def load_durable_constraints(self, stream: Any) -> list[dict[str, Any]]:
        if self.repository is None:
            self.observability.record(
                "store_failure",
                trace_id="behavior_constraint_store",
                reason_code="behavior_constraint_store_unavailable",
                operation="load",
                stream_session_id=self._session_id(stream),
            )
            return []
        existing = [
            BehaviorConstraint.from_value(item)
            for item in list(getattr(stream, "active_behavior_blocks", []) or [])
            if isinstance(item, (dict, BehaviorConstraint))
        ]
        try:
            durable = self.repository.list_active()
        except Exception as exc:
            self.observability.record(
                "store_failure",
                trace_id="behavior_constraint_store",
                reason_code="durable_constraint_load_failed",
                operation="load",
                error_type=type(exc).__name__,
                stream_session_id=self._session_id(stream),
            )
            raise
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
                self.observability.record(
                    "store_failure",
                    trace_id=constraint.source_event_id or constraint.id,
                    reason_code="behavior_constraint_store_unavailable",
                    operation="write",
                    constraint_id=constraint.id,
                )
                raise RuntimeError("durable_constraint_store_unavailable")
            try:
                self.repository.save_durable(constraint)
            except Exception as exc:
                self.observability.record(
                    "store_failure",
                    trace_id=constraint.source_event_id or constraint.id,
                    reason_code="durable_constraint_write_failed",
                    operation="write",
                    constraint_id=constraint.id,
                    error_type=type(exc).__name__,
                )
                raise
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
        feedback_trace_id = str(source_event_id or f"behavior_feedback_{uuid.uuid4().hex}")
        unresolved = ResolvedReferent("", (), ReferentProvenance.UNRESOLVED, 0.0)

        def finish(result: FeedbackApplication) -> FeedbackApplication:
            feedback = interpretation.feedback
            self.observability.record(
                "feedback",
                trace_id=feedback_trace_id,
                timestamp=now,
                stream_session_id=self._session_id(stream),
                source_event_id=str(source_event_id or ""),
                feedback_type=result.kind.value if result.kind else "",
                polarity=str(getattr(feedback, "polarity", "") or ""),
                explicitness=str(getattr(feedback, "explicitness", "") or ""),
                referent_provenance=result.referent.provenance.value,
                referent_resolved=result.referent.resolved,
                normalized_motif_identity=(
                    motif_id(result.referent.terms) if result.referent.terms else ""
                ),
                semantic_terms=list(result.referent.terms),
                effect_applied=bool(result.applied),
                constraint_id=result.constraint_id,
                fatigue=result.fatigue,
                reason_code=result.reason,
            )
            return result

        if (
            interpretation.speech_act != InputSpeechAct.OWNER_FEEDBACK
            or interpretation.authority != "owner"
            or interpretation.feedback is None
        ):
            return finish(FeedbackApplication(False, None, unresolved, reason="not_authoritative_owner_feedback"))
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
            return finish(result)
        if kind == FeedbackKind.EXPLICIT_DURABLE_PREFERENCE and (
            interpretation.confidence < 0.9
            or interpretation.feedback.explicitness != "explicit"
            or referent.confidence < 0.8
            or self._looks_ambiguous(interpretation.feedback_text)
        ):
            result = FeedbackApplication(False, kind, referent, reason="durable_evidence_insufficient")
            self._remember_last(stream, result)
            return finish(result)
        if kind == FeedbackKind.EXPLICIT_DURABLE_PREFERENCE and self.repository is None:
            result = FeedbackApplication(False, kind, referent, reason="durable_constraint_store_unavailable")
            self._remember_last(stream, result)
            return finish(result)

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
            self.observability.record(
                "constraint_created",
                trace_id=feedback_trace_id,
                timestamp=now,
                stream_session_id=self._session_id(stream),
                constraint_id=constraint.id,
                scope=constraint.scope,
                normalized_motif_identity=motif_id(referent.terms),
                semantic_terms=list(referent.terms),
                source_event_id=str(source_event_id or ""),
                reason_code=constraint.reason,
            )
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
            "last_source_event_id": str(source_event_id or ""),
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
        return finish(result)

    def evaluate_candidate(
        self,
        stream: Any,
        text: str,
        *,
        topic: str = "",
        mode: str = "proactive",
        now: float | None = None,
        observation: dict[str, Any] | None = None,
    ) -> CandidateEvaluation:
        now = time.time() if now is None else float(now)
        terms = motif_terms(f"{text} {topic}")
        candidate_id = motif_id(terms) if terms else "motif_unresolved"

        def finish(result: CandidateEvaluation, *, record_runtime: bool = True) -> CandidateEvaluation:
            context = dict(observation or {})
            enriched = replace(
                result,
                trace_id=str(context.get("trace_id") or f"behavior_candidate_{uuid.uuid4().hex}"),
                candidate_id=str(context.get("candidate_id") or candidate_id),
            )
            if record_runtime:
                self._record_candidate(stream, enriched)
            self._observe_candidate_safely(stream, enriched, terms=terms, topic=topic, now=now, context=context)
            return enriched

        if mode == "direct_response":
            return finish(CandidateEvaluation(
                AdaptationAction.ALLOW, 0.0, "direct_required_response",
                candidate_id, 0, 0.0, 0.0, score_multiplier=1.0,
            ), record_runtime=False)
        constraint = self._matching_constraint(stream, terms)
        if constraint is not None:
            result = CandidateEvaluation(
                AdaptationAction.SUPPRESS, 1.0, "explicit_behavior_constraint",
                candidate_id, 0, 1.0, 0.0, constraint.id, 0.0,
            )
            return finish(result)

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
        return finish(result)

    def validate_generated_output(
        self,
        stream: Any,
        text: str,
        *,
        topic: str = "",
        mode: str = "proactive",
        now: float | None = None,
        observation: dict[str, Any] | None = None,
    ) -> CandidateEvaluation:
        """Validate final text against canonical active suppression only.

        Candidate ranking owns fatigue and recent-use scoring. This safety check
        does not rank again; it catches a generated text that drifted into an
        explicitly constrained or currently suppressed motif.
        """
        now = time.time() if now is None else float(now)
        terms = motif_terms(f"{text} {topic}")
        candidate_id = motif_id(terms) if terms else "motif_unresolved"

        def finish(result: CandidateEvaluation, *, record_runtime: bool = True) -> CandidateEvaluation:
            context = dict(observation or {})
            enriched = replace(
                result,
                trace_id=str(context.get("trace_id") or f"behavior_output_{uuid.uuid4().hex}"),
                candidate_id=str(context.get("candidate_id") or candidate_id),
            )
            if record_runtime:
                self._record_candidate(stream, enriched)
            self._observe_candidate_safely(stream, enriched, terms=terms, topic=topic, now=now, context=context)
            return enriched

        if mode == "direct_response":
            return finish(CandidateEvaluation(
                AdaptationAction.ALLOW, 0.0, "direct_required_response",
                candidate_id, 0, 0.0, 0.0, score_multiplier=1.0,
                stage="generated_output",
            ), record_runtime=False)
        constraint = self._matching_constraint(stream, terms)
        if constraint is not None:
            result = CandidateEvaluation(
                AdaptationAction.SUPPRESS, 1.0, "generated_output_matches_constraint",
                candidate_id, 0, 1.0, 0.0, constraint.id, 0.0,
                "generated_output",
            )
            return finish(result)
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
                return finish(result)
        result = CandidateEvaluation(
            AdaptationAction.ALLOW, 0.0, "generated_output_matches_selected_policy",
            candidate_id, 0, 0.0, 0.0, score_multiplier=1.0,
            stage="generated_output",
        )
        return finish(result)

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

    def record_ranking(
        self,
        stream: Any,
        ranked_candidates: list[dict[str, Any]],
        *,
        selected_intent_id: str = "",
        generation_attempted: bool = False,
        timestamp: float | None = None,
    ) -> None:
        now = time.time() if timestamp is None else float(timestamp)
        for item in list(ranked_candidates or []):
            policy = dict(item.get("policy") or {})
            intent_id = str(item.get("intent_id") or "")
            selected = bool(intent_id and intent_id == selected_intent_id)
            self.observability.record(
                "candidate_ranking",
                trace_id=str(policy.get("trace_id") or intent_id),
                timestamp=now,
                stream_session_id=self._session_id(stream),
                candidate_id=str(policy.get("candidate_id") or intent_id),
                speech_intent_id=intent_id,
                topic=str(item.get("topic") or ""),
                policy_decision=str(policy.get("action") or "").upper(),
                base_score=item.get("base_score"),
                adjusted_score=item.get("adjusted_score"),
                candidate_selected=selected,
                generation_attempted=bool(selected and generation_attempted),
                reason_code=str(policy.get("reason") or ""),
            )

    def record_emission(
        self,
        *,
        trace_id: str,
        stream: Any = None,
        event_id: str = "",
        emitted: bool,
        reason_code: str = "",
        timestamp: float | None = None,
    ) -> None:
        self.observability.record(
            "emission",
            trace_id=trace_id,
            timestamp=timestamp,
            stream_session_id=self._session_id(stream),
            event_id=str(event_id or ""),
            emitted=bool(emitted),
            reason_code=str(reason_code or ""),
        )

    def inspection_snapshot(self, stream: Any, *, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else float(now)
        constraints = [
            BehaviorConstraint.from_value(raw)
            for raw in list(getattr(stream, "active_behavior_blocks", []) or [])
            if isinstance(raw, (dict, BehaviorConstraint))
        ]
        active = [
            item for item in constraints
            if item.active and item.status == "ACTIVE" and (not item.expires_at or item.expires_at > now)
        ]
        retired: list[BehaviorConstraint] = []
        store_status = "available" if self.repository is not None else "unavailable"
        if self.repository is not None:
            try:
                retired = [item for item in self.repository.list_all() if item.status == "RETIRED"][-50:]
            except Exception as exc:
                store_status = "durable_constraint_load_failed"
                self.observability.record(
                    "store_failure",
                    trace_id="behavior_constraint_inspector",
                    reason_code="durable_constraint_load_failed",
                    operation="inspect",
                    error_type=type(exc).__name__,
                    stream_session_id=self._session_id(stream),
                )
        episodic = []
        state = getattr(stream, "behavior_adaptation_state", None)
        for raw in list((state if isinstance(state, dict) else {}).get("entries") or []):
            item = dict(raw)
            updated = float(item.get("updated_at") or item.get("created_at") or now)
            elapsed = max(0.0, now - updated)
            negative = float(item.get("negative_weight") or 0.0) * math.pow(0.5, elapsed / self.NEGATIVE_HALF_LIFE_SEC)
            positive = float(item.get("positive_weight") or 0.0) * math.pow(0.5, elapsed / self.POSITIVE_HALF_LIFE_SEC)
            if negative < 0.001 and positive < 0.001 and now >= float(item.get("suppress_until") or 0.0):
                continue
            episodic.append({
                "motif_id": str(item.get("motif_id") or ""),
                "motif_terms": list(item.get("motif_terms") or []),
                "negative_weight": round(negative, 6),
                "positive_weight": round(positive, 6),
                "negative_applications": int(item.get("negative_applications") or 0),
                "positive_applications": int(item.get("positive_applications") or 0),
                "provenance": str(item.get("provenance") or ""),
                "created_at": float(item.get("created_at") or 0.0),
                "updated_at": updated,
                "suppress_until": float(item.get("suppress_until") or 0.0),
                "last_kind": str(item.get("last_kind") or ""),
                "source_event_id": str(item.get("last_source_event_id") or ""),
                "status": "ACTIVE" if negative >= 0.001 or positive >= 0.001 else "DECAYED",
            })
        return {
            "stream_session_id": self._session_id(stream),
            "store_status": store_status,
            "active_current_stream": [self._constraint_view(item) for item in active if item.scope == "current_stream"],
            "active_durable": [self._constraint_view(item) for item in active if item.scope == "durable"],
            "retired_durable_recent": [self._constraint_view(item) for item in retired],
            "episodic_fatigue": episodic,
            "telemetry": self.observability.snapshot(),
        }

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
                if item.scope == "durable":
                    if self.repository is None:
                        self.observability.record(
                            "store_failure",
                            trace_id=source_event_id or item.id,
                            timestamp=now,
                            reason_code="behavior_constraint_store_unavailable",
                            operation="retire",
                            constraint_id=item.id,
                            stream_session_id=self._session_id(stream),
                        )
                    else:
                        try:
                            self.repository.retire(
                                item.id,
                                reason="explicit_owner_reversal",
                                source_event_id=source_event_id,
                                authority="owner",
                                now=now,
                            )
                        except Exception as exc:
                            self.observability.record(
                                "store_failure",
                                trace_id=source_event_id or item.id,
                                timestamp=now,
                                reason_code="durable_constraint_write_failed",
                                operation="retire",
                                constraint_id=item.id,
                                error_type=type(exc).__name__,
                                stream_session_id=self._session_id(stream),
                            )
                            raise
                self.observability.record(
                    "constraint_reverted",
                    trace_id=source_event_id or item.id,
                    timestamp=now,
                    stream_session_id=self._session_id(stream),
                    constraint_id=item.id,
                    scope=item.scope,
                    normalized_motif_identity=motif_id(referent.terms),
                    semantic_terms=list(referent.terms),
                    source_event_id=str(source_event_id or ""),
                    reason_code="explicit_owner_reversal",
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

    def _observe_candidate(
        self,
        stream: Any,
        result: CandidateEvaluation,
        *,
        terms: tuple[str, ...],
        topic: str,
        now: float,
        context: dict[str, Any],
    ) -> None:
        comparisons = self._comparison_rows(stream, terms, now=now)
        max_similarity = max((float(item.get("similarity") or 0.0) for item in comparisons), default=0.0)
        event = "post_generation" if result.stage == "generated_output" else "candidate_policy"
        payload = {
            "stream_session_id": str(context.get("stream_session_id") or self._session_id(stream)),
            "candidate_id": result.candidate_id,
            "speech_intent_id": str(context.get("speech_intent_id") or ""),
            "speech_intent": str(context.get("speech_intent") or ""),
            "topic": str(topic or context.get("topic") or ""),
            "normalized_motif_identity": result.motif_id,
            "semantic_terms": list(terms),
            "recent_comparable_motifs": comparisons,
            "similarity_score": round(max_similarity, 6),
            "usage_count": result.recent_uses,
            "fatigue": result.fatigue,
            "positive_weight": result.positive_weight,
            "negative_weight": result.negative_weight,
            "active_constraint": result.constraint_id,
            "reason_code": result.reason,
        }
        if event == "post_generation":
            payload["post_generation_decision"] = result.action.value.upper()
        else:
            payload["policy_decision"] = result.action.value.upper()
        self.observability.record(event, trace_id=result.trace_id, timestamp=now, **payload)

    def _observe_candidate_safely(
        self,
        stream: Any,
        result: CandidateEvaluation,
        *,
        terms: tuple[str, ...],
        topic: str,
        now: float,
        context: dict[str, Any],
    ) -> None:
        try:
            self._observe_candidate(
                stream, result, terms=terms, topic=topic, now=now, context=context,
            )
        except Exception as exc:
            self.observability.record(
                "telemetry_failure",
                trace_id=result.trace_id,
                timestamp=now,
                stream_session_id=self._session_id(stream),
                candidate_id=result.candidate_id,
                error_type=type(exc).__name__,
                reason_code="behavior_telemetry_observation_failed",
            )

    def _comparison_rows(self, stream: Any, terms: tuple[str, ...], *, now: float) -> list[dict[str, Any]]:
        comparable: list[tuple[str, tuple[str, ...], str, float, str]] = []
        for item in list(getattr(stream, "recent_idle_messages", []) or [])[-30:]:
            other = motif_terms(f"{item.get('text', '')} {item.get('topic', '')}")
            if other:
                comparable.append((
                    motif_id(other), other, "recent_usage",
                    max(0.0, now - float(item.get("timestamp", now) or now)),
                    str(item.get("speech_intent_id") or item.get("used_fact_id") or ""),
                ))
        state = getattr(stream, "behavior_adaptation_state", None)
        for item in list((state if isinstance(state, dict) else {}).get("entries") or []):
            other = tuple(item.get("motif_terms") or ())
            if other:
                comparable.append((
                    str(item.get("motif_id") or motif_id(other)), other,
                    "episodic_feedback", 0.0, str(item.get("last_source_event_id") or ""),
                ))
        for raw in list(getattr(stream, "active_behavior_blocks", []) or []):
            if not isinstance(raw, (dict, BehaviorConstraint)):
                continue
            constraint = BehaviorConstraint.from_value(raw)
            other = tuple(value for value in constraint.behavior_variants if not value.startswith("motif:"))
            if other:
                comparable.append((
                    constraint.id, other, "active_constraint", 0.0,
                    constraint.source_event_id,
                ))
        rows = []
        for identity, other, source, age, related_event_id in comparable:
            evidence = semantic_similarity_evidence(terms, other)
            rows.append({
                "motif_identity": identity,
                "source": source,
                "semantic_terms": list(other),
                "shared_terms": evidence["shared_terms"],
                "containment": evidence["containment"],
                "jaccard": evidence["jaccard"],
                "similarity": evidence["similarity"],
                "matched": evidence["matched"],
                "age_seconds": round(age, 3),
                "related_event_id": related_event_id,
            })
        rows.sort(key=lambda item: (float(item["similarity"]), -float(item["age_seconds"])), reverse=True)
        return rows[:12]

    @staticmethod
    def _constraint_view(item: BehaviorConstraint) -> dict[str, Any]:
        return {
            "id": item.id,
            "behavior_family": item.behavior_family,
            "target": [value for value in item.behavior_variants if not value.startswith("motif:")],
            "recipient_scope": item.recipient_scope,
            "scope": item.scope,
            "authority": item.authority,
            "provenance": item.source_event_id,
            "explicitness": item.explicitness,
            "confidence": item.confidence,
            "created_at": item.created_at,
            "expires_at": item.expires_at,
            "status": item.status,
            "reason": item.reason,
            "retired_at": item.retired_at,
            "retirement_reason": item.retirement_reason,
            "version": item.version,
        }

    @staticmethod
    def _session_id(stream: Any) -> str:
        if stream is None:
            return ""
        return str(
            getattr(stream, "active_stream_session_id", "")
            or getattr(stream, "stream_session_id", "")
            or ""
        )

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
    "semantic_similarity_evidence",
]
