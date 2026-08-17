from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
import re
import unicodedata
from typing import Any

from app.core.persistent_logs import log_jsonl_event
from app.services.direct_stt_command import (
    DirectSTTCommandResult,
    DirectUtteranceIntentFamily,
    parse_direct_stt_command,
)


class InputSpeechAct(StrEnum):
    OWNER_COMMAND = "owner_command"
    OWNER_FEEDBACK = "owner_feedback"
    OWNER_COMMENTARY = "owner_commentary"
    OWNER_ANSWER_FOLLOWUP = "owner_answer_followup"
    VIEWER_DIRECTED_TO_HEBE = "viewer_directed_to_hebe"
    VIEWER_CONTEXT = "viewer_context"
    AMBIENT_CONTEXT = "ambient_context"
    SYSTEM_EVENT = "system_event"


@dataclass(frozen=True, slots=True)
class FeedbackSignal:
    target: str
    polarity: str
    strength: float
    explicitness: str
    referent: str = ""
    meta_about_hebe: bool = True
    recipient: str = "Hebe"
    target_provenance: str = ""


@dataclass(frozen=True, slots=True)
class SemanticPredicate:
    predicate: str
    polarity: str
    evidence_span: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticClause:
    text: str
    speaker: str
    resolved_addressee: str
    addressee_provenance: str
    subject: str
    subject_provenance: str
    predicates: tuple[SemanticPredicate, ...]
    semantic_role: str
    eligible_domains: tuple[str, ...]
    excluded_domains: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["predicates"] = [item.as_dict() for item in self.predicates]
        value["eligible_domains"] = list(self.eligible_domains)
        value["excluded_domains"] = list(self.excluded_domains)
        return value


@dataclass(frozen=True, slots=True)
class InputInterpretation:
    """Canonical communicative meaning consumed by routing and domain gates."""

    speech_act: InputSpeechAct
    source: str
    authority: str
    addressed_to_hebe: bool
    confidence: float
    possible_command_syntax: bool
    authorized_action_command: bool
    context_eligible: bool
    feedback: FeedbackSignal | None = None
    feedback_text: str = ""
    context_text: str = ""
    excluded_context_spans: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    semantic_clauses: tuple[SemanticClause, ...] = field(default_factory=tuple)
    negated_predicates: tuple[str, ...] = field(default_factory=tuple)
    resolved_addressee: str = ""
    feedback_target: str = ""
    feedback_target_provenance: str = ""
    excluded_domain_scopes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def action_eligible(self) -> bool:
        return self.authorized_action_command

    @property
    def meta_about_hebe(self) -> bool:
        return bool(self.feedback and self.feedback.meta_about_hebe)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["speech_act"] = self.speech_act.value
        value["excluded_context_spans"] = list(self.excluded_context_spans)
        value["semantic_clauses"] = [item.as_dict() for item in self.semantic_clauses]
        value["negated_predicates"] = list(self.negated_predicates)
        value["excluded_domain_scopes"] = list(self.excluded_domain_scopes)
        return value


_OWNER_SOURCES = {
    "ui", "typed_ui", "owner_ui", "voice", "stt_voice", "button",
    "owner_stt_direct", "owner_stt_command", "owner_stt_followup",
}
_VIEWER_SOURCES = {"twitch", "twitch_chat", "twitch_viewer"}
_AMBIENT_SOURCES = {"ambient", "ambient_stt", "ambient_context"}
_SYSTEM_SOURCES = {
    "system", "twitch_system", "twitch_event", "scheduler/spontaneity",
    "system/tool_result", "scheduler", "internal_event",
}

_NEGATIVE_FEEDBACK = (
    r"\bdeja(?:\s+ya)?\s+de\b",
    r"\bno\s+me\s+vuelvas\s+a\b",
    r"\bno\s+hace\s+falta\s+que\s+me\b",
    r"\bte\s+quieres\s+callar\b",
    r"\botra\s+vez\s+con\s+(?:lo\s+de(?:l)?|la|el)\b",
    r"\b(?:todo\s+el\s+rato|todo\s+el\s+(?:puto\s+)?stream)\b.*\b(?:diciendome|recordandome|recomendandome)\b",
    r"\bno\s+vuelvas\s+a\b",
    r"\bno\s+quiero\s+que\s+vuelvas\s+a\b",
    r"\bdeja\s+(?:ese|esa)\s+(?:tema|broma)\b",
)
_POSITIVE_FEEDBACK = (
    r"\beso\s+si\s+me\s+ha\s+hecho\s+gracia\b",
    r"\b(?:esa|esta|tu)\s+respuesta\s+(?:estuvo|ha\s+estado|esta)\s+bien\b",
    r"\bme\s+ha\s+gustado\s+(?:eso|esa|tu\s+respuesta)\b",
    r"\b(?:puedes|puede)\s+(?:volver|seguir)\b",
    r"\bya\s+puedes\b.*\b(?:seguir|volver|hacer)\b",
    r"\b(?:esa|eso)\s+vuelve\s+a\s+tener\s+gracia\b",
)
_GENERIC_COMMAND_PREFIX = re.compile(
    r"^(?:(?:hebe|ebe|eve|jebe|heve)\s*[,;:!?-]*\s*)?"
    r"(?:(?:oye|mira|vale|ok(?:ay)?|por\s+favor|puedes|podrias)\s+)*"
    r"(?:abre|abrir|inicia|iniciar|arranca|arrancar|lanza|lanzar|ejecuta|ejecutar|"
    r"pon|ponme|cierra|cerrar|haz|hazle|dale|manda|envia|dile|di|activa|desactiva|"
    r"pausa|reanuda|cambia|quita|deja|recuerdame|avisame|agenda|agendame|apuntame|apaontame|"
    r"cancela|descarta|anula|despierta|duerme|calla|callate|silencio|"
    r"open|start|launch|run|close|send|enable|disable|stop|pause|resume|cancel)\b"
)
_TEMPORAL_REMINDER_COMMAND = re.compile(
    r"^(?:(?:dentro\s+de|en)\s+(?:\d+|un[ao]?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\s+"
    r"(?:minut[oa]s?|horas?|dias?)\s+)(?:recuerdame|avisame)\b"
)


class InputInterpreter:
    """Single owner for primary input meaning and semantic scope."""

    def interpret_event(
        self,
        event: Any,
        *,
        authority: str | None = None,
        addressed_to_hebe: bool | None = None,
        explicit_command_mode: bool | None = None,
        pending_valid: bool = False,
        recent_hebe_utterance: str = "",
        social_identities: list[str] | tuple[str, ...] | None = None,
        direct_result: DirectSTTCommandResult | None = None,
    ) -> InputInterpretation:
        existing = getattr(event, "interpretation", None)
        if isinstance(existing, InputInterpretation) and not pending_valid:
            return existing
        metadata = dict(getattr(event, "stt_metadata", {}) or {})
        if social_identities is None:
            social_identities = list(metadata.get("semantic_social_identities") or [])
        direct = direct_result or DirectSTTCommandResult.from_dict(metadata.get("direct_stt_command"))
        if not direct.command_text:
            direct = parse_direct_stt_command(
                getattr(event, "raw_text", "") or getattr(event, "normalized_text", "")
            )
        source = str(getattr(event, "source", "") or "ambient")
        resolved_authority = authority or self.authority_for_source(source)
        addressed = (
            bool(addressed_to_hebe)
            if addressed_to_hebe is not None
            else bool(direct.wake_detected or source in {"ui", "typed_ui", "owner_ui", "button"})
        )
        command_mode = (
            bool(explicit_command_mode)
            if explicit_command_mode is not None
            else bool(metadata.get("command_mode", source in {"ui", "typed_ui", "owner_ui", "button"}))
        )
        result = self.interpret(
            raw_text=getattr(event, "raw_text", "") or getattr(event, "normalized_text", ""),
            source=source,
            authority=resolved_authority,
            addressed_to_hebe=addressed,
            explicit_command_mode=command_mode,
            pending_valid=pending_valid,
            recent_hebe_utterance=recent_hebe_utterance,
            social_identities=social_identities,
            direct_result=direct,
        )
        event.interpretation = result
        return result

    def interpret(
        self,
        *,
        raw_text: str,
        source: str,
        authority: str,
        addressed_to_hebe: bool,
        explicit_command_mode: bool = False,
        pending_valid: bool = False,
        recent_hebe_utterance: str = "",
        social_identities: list[str] | tuple[str, ...] | None = None,
        direct_result: DirectSTTCommandResult | None = None,
    ) -> InputInterpretation:
        raw = str(raw_text or "").strip()
        direct = direct_result or parse_direct_stt_command(raw)
        interpreted_text = str(direct.command_text or raw).strip()
        normalized = self.normalize(interpreted_text)
        clauses = self.analyze_semantic_clauses(
            interpreted_text,
            speaker=authority,
            addressed_to_hebe=addressed_to_hebe,
            recent_hebe_utterance=recent_hebe_utterance,
            social_identities=social_identities,
        )

        def finish(result: InputInterpretation) -> InputInterpretation:
            feedback = result.feedback
            negated = tuple(dict.fromkeys(
                predicate.predicate
                for clause in clauses
                for predicate in clause.predicates
                if predicate.polarity == "negative"
            ))
            resolved_addressee = next((
                clause.resolved_addressee for clause in clauses
                if clause.resolved_addressee
            ), "")
            excluded = tuple(
                f"clause_{index}:{domain}:{clause.reason}"
                for index, clause in enumerate(clauses)
                for domain in clause.excluded_domains
            )
            return self._log(replace(
                result,
                semantic_clauses=clauses,
                negated_predicates=negated,
                resolved_addressee=resolved_addressee,
                feedback_target=feedback.recipient if feedback else "",
                feedback_target_provenance=feedback.target_provenance if feedback else "",
                excluded_domain_scopes=excluded,
            ))
        possible_command = bool(
            direct.detected_intent_family in {
                DirectUtteranceIntentFamily.APPLICATION_ACTION.value,
            }
            or (
                direct.detected_intent_family == DirectUtteranceIntentFamily.INCOMPLETE_COMMAND.value
                and direct.wake_detected
            )
            or _GENERIC_COMMAND_PREFIX.match(normalized)
            or _TEMPORAL_REMINDER_COMMAND.match(normalized)
        )
        silence_control = bool(re.match(
            r"^(?:(?:hebe|ebe|eve|jebe|heve)\s+)?(?:deja(?: ya)? de hablar|"
            r"no hables sola|calla|callate|silencio|quieta)\s*$",
            normalized,
        ))
        if not silence_control and clauses and all(
            "assistant_command" in clause.excluded_domains
            for clause in clauses
            if any(item.predicate in {"stop_request", "aspectual_continuation"} for item in clause.predicates)
        ) and any(
            any(item.predicate in {"stop_request", "aspectual_continuation"} for item in clause.predicates)
            for clause in clauses
        ):
            possible_command = False

        feedback = None if silence_control else self._feedback_signal(
            normalized,
            recent_hebe_utterance=recent_hebe_utterance,
            clauses=clauses,
            addressed_to_hebe=addressed_to_hebe,
        )
        if feedback is not None and authority == "owner":
            feedback_text, context_text = self._semantic_scopes(interpreted_text, clauses)
            result = InputInterpretation(
                speech_act=InputSpeechAct.OWNER_FEEDBACK,
                source=source,
                authority=authority,
                addressed_to_hebe=bool(addressed_to_hebe or self._feedback_is_directed(normalized)),
                confidence=0.97 if feedback.polarity == "negative" else 0.9,
                possible_command_syntax=possible_command,
                authorized_action_command=False,
                context_eligible=bool(context_text),
                feedback=feedback,
                feedback_text=feedback_text,
                context_text=context_text,
                excluded_context_spans=(feedback_text,) if feedback_text else (),
                reason="explicit_owner_feedback",
            )
            return finish(result)

        if authority == "system" or source in _SYSTEM_SOURCES:
            return finish(InputInterpretation(
                InputSpeechAct.SYSTEM_EVENT, source, "system", addressed_to_hebe,
                1.0, possible_command, False, False, reason="system_source",
            ))
        if authority == "viewer" or source in _VIEWER_SOURCES:
            act = (
                InputSpeechAct.VIEWER_DIRECTED_TO_HEBE
                if addressed_to_hebe or self._mentions_hebe(normalized)
                else InputSpeechAct.VIEWER_CONTEXT
            )
            return finish(InputInterpretation(
                act, source, "viewer", act == InputSpeechAct.VIEWER_DIRECTED_TO_HEBE,
                0.94 if act == InputSpeechAct.VIEWER_DIRECTED_TO_HEBE else 0.84,
                possible_command, False, act == InputSpeechAct.VIEWER_CONTEXT,
                context_text=interpreted_text if act == InputSpeechAct.VIEWER_CONTEXT else "",
                reason="viewer_addressing" if act == InputSpeechAct.VIEWER_DIRECTED_TO_HEBE else "viewer_context",
            ))
        if authority == "ambient" or source in _AMBIENT_SOURCES:
            return finish(InputInterpretation(
                InputSpeechAct.AMBIENT_CONTEXT, source, "ambient", False, 0.9,
                possible_command, False, True, context_text=interpreted_text, reason="ambient_source",
            ))
        if pending_valid and authority == "owner":
            return finish(InputInterpretation(
                InputSpeechAct.OWNER_ANSWER_FOLLOWUP, source, authority, addressed_to_hebe,
                0.96, possible_command, False, False, reason="valid_pending_followup",
            ))

        authorized = bool(
            authority == "owner"
            and possible_command
            and (
                addressed_to_hebe
                or explicit_command_mode
                or source in {
                    "ui", "typed_ui", "owner_ui", "button", "stt_voice", "voice",
                    "owner_stt_direct", "owner_stt_command",
                }
            )
        )
        if authorized:
            return finish(InputInterpretation(
                InputSpeechAct.OWNER_COMMAND, source, authority, addressed_to_hebe,
                0.98 if addressed_to_hebe else 0.88, possible_command, True, False,
                reason="authorized_owner_command",
            ))
        return finish(InputInterpretation(
            InputSpeechAct.OWNER_COMMENTARY, source, authority, addressed_to_hebe,
            0.78, possible_command, False, True, context_text=interpreted_text,
            reason="owner_non_action_input",
        ))

    @staticmethod
    def authority_for_source(source: str) -> str:
        source = str(source or "").lower()
        if source in _OWNER_SOURCES:
            return "owner"
        if source in _VIEWER_SOURCES:
            return "viewer"
        if source in _AMBIENT_SOURCES:
            return "ambient"
        return "system"

    @staticmethod
    def normalize(text: str) -> str:
        value = str(text or "").casefold()
        value = "".join(
            character for character in unicodedata.normalize("NFKD", value)
            if not unicodedata.combining(character)
        )
        value = re.sub(r"[^a-z0-9\s;:'\"-]", " ", value)
        return " ".join(value.split())

    def _feedback_signal(
        self,
        normalized: str,
        *,
        recent_hebe_utterance: str,
        clauses: tuple[SemanticClause, ...],
        addressed_to_hebe: bool,
    ) -> FeedbackSignal | None:
        negative_clause = next((
            clause for clause in clauses
            if "behavior_feedback" in clause.eligible_domains
            and any(re.search(pattern, self.normalize(clause.text)) for pattern in _NEGATIVE_FEEDBACK)
        ), None)
        negative = negative_clause is not None
        positive_pattern = any(re.search(pattern, normalized) for pattern in _POSITIVE_FEEDBACK)
        positive = bool(
            positive_pattern
            and (
                addressed_to_hebe
                or self._mentions_hebe(normalized)
                or bool(recent_hebe_utterance.strip())
                or "respuesta" in normalized
            )
        )
        contextual_positive = bool(
            recent_hebe_utterance.strip()
            and re.search(r"\bbuenisima\s+esa\b", normalized)
        )
        if not (negative or positive or contextual_positive):
            return None
        polarity = "negative" if negative else "positive"
        repeated = bool(re.search(r"\b(?:otra vez|repetir|insistir|todo el (?:puto )?(?:rato|stream))\b", normalized))
        target = (
            "repeated_hebe_behavior" if repeated
            else "hebe_response" if positive or contextual_positive
            else "hebe_behavior"
        )
        referent = self._extract_referent(
            self.normalize(negative_clause.text) if negative_clause is not None else normalized
        )
        if contextual_positive and not referent:
            referent = "previous_hebe_utterance"
        strong_language = bool(re.search(r"\b(?:put[oa]|joder|mierda)\b", normalized))
        strength = 0.98 if strong_language else 0.9 if negative else 0.78
        return FeedbackSignal(
            target=target,
            polarity=polarity,
            strength=strength,
            explicitness="explicit" if not contextual_positive else "contextual",
            referent=referent,
            recipient="Hebe",
            target_provenance=(
                negative_clause.addressee_provenance
                if negative_clause is not None
                else "recent_hebe_utterance" if contextual_positive or recent_hebe_utterance.strip()
                else "explicit_hebe_reference"
            ),
        )

    @classmethod
    def analyze_semantic_clauses(
        cls,
        raw: str,
        *,
        speaker: str,
        addressed_to_hebe: bool = False,
        recent_hebe_utterance: str = "",
        social_identities: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[SemanticClause, ...]:
        parts = [
            part.strip() for part in re.split(r"\s*;\s*|(?<=[.!?])\s+", str(raw or ""))
            if part.strip()
        ]
        identity_names = [str(item or "").strip() for item in (social_identities or ()) if str(item or "").strip()]
        normalized_identities = {cls.normalize(item): item for item in identity_names}
        clauses: list[SemanticClause] = []
        for part in parts:
            normalized = cls.normalize(part)
            addressee = ""
            addressee_provenance = ""
            mentioned_identity = ""
            for identity_norm, identity in normalized_identities.items():
                if identity_norm and re.search(rf"\b{re.escape(identity_norm)}\b", normalized):
                    mentioned_identity = identity
                    if re.search(rf"(?i)@{re.escape(identity)}\b", part):
                        addressee = identity
                        addressee_provenance = "resolved_social_identity_mention"
                        break
                    if re.search(rf"(?i)(?:^|[,;]\s*)@?{re.escape(identity)}\s*,", part):
                        addressee = identity
                        addressee_provenance = "resolved_social_identity_vocative"
                        break
            if cls._mentions_hebe(normalized):
                hebe_vocative = bool(re.search(
                    r"(?i)(?:^|[,]\s*)(?:hebe|ebe|eve|jebe|heve)\s*(?:,|$)",
                    part,
                ))
                if hebe_vocative or addressed_to_hebe:
                    addressee = "Hebe"
                    addressee_provenance = "explicit_hebe_vocative"
            if not addressee and addressed_to_hebe:
                addressee = "Hebe"
                addressee_provenance = "canonical_addressing_context"
            explicit_other_mention = next((
                identity for identity_norm, identity in normalized_identities.items()
                if identity_norm and re.search(rf"\b{re.escape(identity_norm)}\b", normalized)
            ), "")

            subject = ""
            subject_provenance = ""
            if addressee and addressee != "Hebe":
                subject, subject_provenance = addressee, "addressee_is_imperative_subject"
            elif re.search(r"\bel\s+(?:jefe|boss|enemigo)\b", normalized):
                subject, subject_provenance = "game_entity", "explicit_game_subject"
            elif re.search(r"\b(?:llover|llueve|lloviendo)\b", normalized):
                subject, subject_provenance = "weather", "weather_predicate"
            elif re.search(r"\b(?:me\s+han\s+matado|me\s+ha\s+matado|me\s+he\s+muerto|he\s+muerto|(?:me\s+)?habia\s+muerto|casi\s+me\s+muero|estoy\s+a\s+punto\s+de\s+morir|\d+\s*hp)\b", normalized):
                subject, subject_provenance = "owner_player", "explicit_first_person"
            elif explicit_other_mention and re.search(r"\b(?:hace|hacer|dile\s+a)\b", normalized):
                subject, subject_provenance = explicit_other_mention, "resolved_social_identity_subject"
            elif addressee == "Hebe":
                subject, subject_provenance = "Hebe", addressee_provenance

            predicates: list[SemanticPredicate] = []
            aspectual = re.search(r"\bno\s+deja\s+de\s+([a-z0-9 ]+)", normalized)
            if aspectual:
                predicates.append(SemanticPredicate(
                    "aspectual_continuation", "positive", aspectual.group(0)[:120],
                    "negated_aspectual_auxiliary_not_stop_imperative",
                ))
            stop = re.search(r"\bdeja(?:\s+ya)?\s+de\s+([a-z0-9 ]+)", normalized)
            if stop and not aspectual:
                predicates.append(SemanticPredicate(
                    "stop_request", "positive", stop.group(0)[:120], "surface_stop_imperative",
                ))
            subordinate_stop = re.search(r"\b(?:que\s+)?deje\s+de\s+([a-z0-9 ]+)", normalized)
            if subordinate_stop:
                predicates.append(SemanticPredicate(
                    "third_party_stop_request", "positive", subordinate_stop.group(0)[:120],
                    "subordinate_or_third_party_request",
                ))

            risk = re.search(r"\b(?:casi\s+(?:me\s+)?muero|(?:estoy\s+)?a\s+punto\s+de\s+morir)\b", normalized)
            if risk:
                predicates.append(SemanticPredicate(
                    "death_risk", "uncertain", risk.group(0), "near_death_not_completed",
                ))
            reported = re.search(r"\b(?:pense|crei)\s+que\s+(?:habia|me\s+habia)\s+muerto\b", normalized)
            death = re.search(
                r"\b(?:me\s+han\s+matado|me\s+ha\s+matado|me\s+mataron|me\s+he\s+muerto|"
                r"he\s+muerto|(?:me\s+)?habia\s+muerto|estoy\s+muert[oa]|mori|game\s+over|wipe)\b",
                normalized,
            )
            if death and not risk:
                prefix = normalized[:death.start()].split()[-4:]
                locally_negated = "no" in prefix
                polarity = "uncertain" if reported else "negative" if locally_negated else "positive"
                predicates.append(SemanticPredicate(
                    "completed_death", polarity, death.group(0),
                    "reported_or_thought_event" if reported else "clause_local_negation" if locally_negated else "explicit_completed_event",
                ))
            hp = re.search(r"\b(?:\d+\s*hp|hp|poca\s+vida|sin\s+vida)\b", normalized)
            if hp:
                predicates.append(SemanticPredicate(
                    "owner_health_state", "positive", hp.group(0), "explicit_health_evidence",
                ))
            ongoing_attack = re.search(r"\bno\s+deja\s+de\s+atacar\b", normalized)
            if ongoing_attack:
                predicates.append(SemanticPredicate(
                    "ongoing_enemy_attack", "positive", ongoing_attack.group(0),
                    "aspectual_continuation_asserts_ongoing_attack",
                ))

            negative_feedback_surface = any(re.search(pattern, normalized) for pattern in _NEGATIVE_FEEDBACK)
            strong_hebe_behavior = bool(re.search(
                r"\b(?:otra\s+vez\s+con\s+lo\s+de|repetir|insistir|diciendome|recordandome|recomendandome|"
                r"decirme|te\s+quieres\s+callar|tu\s+respuesta|hacerlo|tema|broma)\b",
                normalized,
            ))
            explicit_other = bool(addressee and addressee != "Hebe")
            feedback_evidence = bool(
                negative_feedback_surface
                and not aspectual
                and not explicit_other
                and subject not in {"weather", "game_entity"}
                and (
                    addressee == "Hebe"
                    or addressed_to_hebe
                    or strong_hebe_behavior
                    or bool(recent_hebe_utterance.strip())
                )
            )
            proxy_command = bool(re.search(r"\b(?:dile|di)\s+a\b.*\bque\s+deje\s+de\b", normalized))
            has_gameplay = any(
                item.predicate in {"completed_death", "death_risk", "owner_health_state", "ongoing_enemy_attack"}
                for item in predicates
            )
            if proxy_command:
                role, eligible, excluded, reason = (
                    "owner_command", ("assistant_command",), ("behavior_feedback", "gameplay_context"),
                    "third_party_request_not_feedback_about_hebe",
                )
            elif feedback_evidence:
                role, eligible, excluded, reason = (
                    "feedback_clause", ("behavior_feedback",), ("gameplay_context",),
                    "hebe_feedback_target_resolved",
                )
                if not addressee:
                    addressee = "Hebe"
                    addressee_provenance = (
                        "recent_hebe_utterance" if recent_hebe_utterance.strip()
                        else "strong_second_person_behavior_context"
                    )
            elif subordinate_stop and subject and subject != "Hebe":
                role, eligible, excluded, reason = (
                    "social_addressed_clause", ("social_context",),
                    ("behavior_feedback", "assistant_command"),
                    "resolved_third_party_subject",
                )
            elif explicit_other:
                role, eligible, excluded, reason = (
                    "social_addressed_clause", ("social_context",), ("behavior_feedback", "assistant_command"),
                    "explicit_other_addressee",
                )
            elif aspectual:
                role = "gameplay_observation" if subject == "game_entity" else "descriptive_clause"
                eligible = ("gameplay_context",) if has_gameplay or subject == "game_entity" else ("general_context",)
                excluded = ("behavior_feedback", "assistant_command")
                reason = "aspectual_construction_not_imperative"
            elif has_gameplay:
                role, eligible, excluded, reason = (
                    "gameplay_observation", ("gameplay_context",), ("behavior_feedback",),
                    "gameplay_predicate_detected",
                )
            elif stop:
                role, eligible, excluded, reason = (
                    "non_assistant_stop_clause", ("general_context",), ("behavior_feedback", "assistant_command"),
                    "stop_request_without_hebe_target_evidence",
                )
            else:
                role, eligible, excluded, reason = (
                    "commentary_clause", ("general_context",), (), "no_special_semantic_scope",
                )
            clauses.append(SemanticClause(
                text=part[:240], speaker=str(speaker or ""), resolved_addressee=addressee,
                addressee_provenance=addressee_provenance, subject=subject,
                subject_provenance=subject_provenance, predicates=tuple(predicates),
                semantic_role=role, eligible_domains=tuple(eligible),
                excluded_domains=tuple(excluded), reason=reason,
            ))
        return tuple(clauses)

    @staticmethod
    def _extract_referent(normalized: str) -> str:
        patterns = (
            r"\botra vez con lo de(?:l)?\s+(.+?)(?:[;.]|$)",
            r"\bdeja(?: ya)? de\s+(.+?)(?:[;.]|$)",
            r"\bno me vuelvas a\s+(.+?)(?:[;.]|$)",
            r"\bno vuelvas a\s+(.+?)(?:[;.]|$)",
            r"\bno quiero que vuelvas a\s+(.+?)(?:[;.]|$)",
            r"\bdeja\s+(?:ese|esa)\s+(?:tema|broma)\s*(.*?)(?:[;.]|$)",
            r"\b(?:puedes|puede)\s+(?:volver|seguir)(?:\s+a)?\s*(.*?)(?:[;.]|$)",
            r"\b(?:esa|esta|tu) respuesta\b",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            if match.lastindex:
                value = match.group(1).strip(" ,:-")
                return value[:160]
            return "response"
        return ""

    @staticmethod
    def _semantic_scopes(raw: str, clauses: tuple[SemanticClause, ...]) -> tuple[str, str]:
        feedback_clauses = [item.text for item in clauses if "behavior_feedback" in item.eligible_domains]
        context_clauses = [item.text for item in clauses if "gameplay_context" in item.eligible_domains]
        feedback_text = "; ".join(feedback_clauses) or raw
        context_text = "; ".join(context_clauses)
        return feedback_text, context_text

    @staticmethod
    def _feedback_is_directed(normalized: str) -> bool:
        return bool(
            InputInterpreter._mentions_hebe(normalized)
            or re.search(r"\b(?:te quieres callar|deja(?: ya)? de|no me vuelvas a|no hace falta que me)\b", normalized)
        )

    @staticmethod
    def _mentions_hebe(normalized: str) -> bool:
        return bool(re.search(r"\b(?:hebe|ebe|eve|jebe|heve)\b", normalized))

    @staticmethod
    def _log(result: InputInterpretation) -> InputInterpretation:
        feedback = result.feedback
        payload = {
            "source": result.source,
            "authority": result.authority,
            "speech_act": result.speech_act.value,
            "addressed_to_hebe": result.addressed_to_hebe,
            "command_eligible": result.authorized_action_command,
            "possible_command_syntax": result.possible_command_syntax,
            "context_eligible": result.context_eligible,
            "feedback_polarity": feedback.polarity if feedback else "",
            "feedback_target": feedback.target if feedback else "",
            "feedback_referent": feedback.referent if feedback else "",
            "resolved_addressee": result.resolved_addressee,
            "feedback_recipient": result.feedback_target,
            "feedback_target_provenance": result.feedback_target_provenance,
            "negated_predicates": list(result.negated_predicates),
            "excluded_domain_scopes": list(result.excluded_domain_scopes),
            "semantic_clauses": [item.as_dict() for item in result.semantic_clauses],
            "confidence": result.confidence,
        }
        print(
            "[HEBE][INPUT_INTERPRETATION] "
            + " ".join(f"{key}={value!r}" for key, value in payload.items()),
            flush=True,
        )
        log_jsonl_event("input_interpretation", payload)
        return result
