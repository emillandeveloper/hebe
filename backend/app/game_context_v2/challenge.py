from __future__ import annotations

import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable

from app.core.persistent_logs import log_jsonl_event
from app.game_context_v2.models import ChallengeDefinition
from app.game_context_v2.repository import GameV2Repository


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


@dataclass(slots=True)
class ChallengeCaptureContext:
    challenge_id: str
    run_id: str
    game_id: str
    started_at: float
    last_activity_at: float
    expires_at: float


class ChallengeContextService:
    """Learns owner-explicit challenge definitions without mixing them into run progress."""

    def __init__(
        self, repository: GameV2Repository, *, now_fn: Callable[[], float] = time.time,
        capture_ttl_seconds: float = 180.0,
    ) -> None:
        self.repository = repository
        self.now_fn = now_fn
        self.capture_ttl_seconds = max(30.0, float(capture_ttl_seconds))
        self.capture: ChallengeCaptureContext | None = None
        self.last_event: dict[str, Any] = {}

    def observe_owner_utterance(
        self, text: str, *, game: str, run_id: str = "", source_event_id: str = "owner_utterance",
    ) -> dict[str, Any]:
        raw = str(text or "").strip()
        normalized = _normalize(raw)
        now = self.now_fn()
        result: dict[str, Any] = {
            "challenge_rule_detected": False,
            "challenge_rule_added": False,
            "challenge_rule_corrected": False,
            "challenge_capture_started": False,
            "challenge_capture_closed": False,
            "playthrough_type": "",
            "challenge_id": "",
            "challenge_name": "",
            "run_override": False,
            "source_event_id": source_event_id,
        }
        if not raw:
            return result
        if self.capture and now >= self.capture.expires_at:
            self._close_capture("timeout", result)

        identity = self.repository.resolve_identity(game) if game else None
        game_id = identity.game_id if identity else ""
        challenge_mentioned = self._challenge_mentioned(normalized)
        challenge_name = self._challenge_name(raw)
        definition: ChallengeDefinition | None = None
        if challenge_name:
            definition = self.get_or_create_definition(
                challenge_name, game_id=game_id, provenance="owner_explicit", confidence=1.0,
            )
            result.update({"challenge_id": definition.challenge_id, "challenge_name": definition.name})
        elif run_id:
            definition = self.definition_for_run(run_id)

        if challenge_mentioned:
            result["playthrough_type"] = "challenge"
            if run_id:
                rules = {"challenge": definition.name} if definition else {}
                if definition:
                    rules["challenge_definition_id"] = definition.challenge_id
                    self.repository.link_run_challenge(run_id, definition.challenge_id, at=now)
                self.repository.update_run_contract(run_id, run_kind="challenge", rules=rules, at=now)

        if self._starts_rule_capture(normalized):
            if definition is None and run_id:
                definition = self.definition_for_run(run_id)
            if definition is not None:
                self.capture = ChallengeCaptureContext(
                    definition.challenge_id, run_id, game_id, now, now, now + self.capture_ttl_seconds,
                )
                result.update({
                    "challenge_capture_started": True,
                    "challenge_id": definition.challenge_id,
                    "challenge_name": definition.name,
                })
                self._emit("challenge_capture_started", result)
            else:
                result["capture_start_rejected"] = "challenge_definition_unknown"
            self.last_event = result
            return result

        capture = self.capture
        if capture is None or (capture.run_id and run_id and capture.run_id != run_id):
            self.last_event = result
            return result
        if self._closes_rule_capture(normalized):
            self._close_capture("explicit_close", result)
            self.last_event = result
            return result
        if self._clear_topic_shift(normalized):
            self._close_capture("topic_shift", result)
            self.last_event = result
            return result

        definition = self.repository.get_challenge_definition(capture.challenge_id)
        if definition is None:
            self._close_capture("definition_missing", result)
            self.last_event = result
            return result
        capture.last_activity_at = now
        capture.expires_at = now + self.capture_ttl_seconds
        result.update({"challenge_id": definition.challenge_id, "challenge_name": definition.name})
        if self._is_run_override(normalized):
            override = self._new_rule(raw, provenance="owner_explicit", confidence=1.0, rule_type="RUN_OVERRIDE")
            linked = self.repository.run_challenge(run_id) or {}
            overrides = list(linked.get("overrides") or [])
            overrides.append(override)
            self.repository.link_run_challenge(run_id, definition.challenge_id, overrides=overrides, at=now)
            result.update({"challenge_rule_detected": True, "challenge_rule_added": True, "run_override": True})
            self._emit("challenge_rule_added", result)
        elif self._is_correction(normalized):
            corrected = self.correct_rule(definition.challenge_id, raw)
            result.update({"challenge_rule_detected": True, "challenge_rule_corrected": bool(corrected)})
            self._emit("challenge_rule_corrected", result)
        elif self._looks_like_rule(normalized):
            self.add_rule(definition.challenge_id, raw, provenance="owner_explicit", confidence=1.0)
            result.update({"challenge_rule_detected": True, "challenge_rule_added": True})
            self._emit("challenge_rule_added", result)
        self.last_event = result
        return result

    def get_or_create_definition(
        self, name: str, *, game_id: str = "", game_family: str = "",
        provenance: str = "owner_explicit", confidence: float = 1.0,
    ) -> ChallengeDefinition:
        existing = self.repository.find_challenge_definition(name, game_id=game_id, game_family=game_family)
        if existing is not None:
            return existing
        now = self.now_fn()
        definition = ChallengeDefinition(
            challenge_id=f"challenge_{uuid.uuid4().hex}", name=" ".join(str(name).split()),
            game_id=game_id, game_family=game_family, provenance=provenance,
            confidence=float(confidence), created_at=now, updated_at=now,
        )
        return self.repository.save_challenge_definition(definition)

    def definition_for_run(self, run_id: str) -> ChallengeDefinition | None:
        link = self.repository.run_challenge(run_id)
        return self.repository.get_challenge_definition(str((link or {}).get("challenge_definition_id") or ""))

    def apply_known_definition_from_metadata(
        self, *, title: str, game: str, run_id: str,
    ) -> ChallengeDefinition | None:
        if not title or not game or not run_id:return None
        identity=self.repository.resolve_identity(game)
        definition=self.repository.match_challenge_definition(title,game_id=identity.game_id)
        if definition is None:return None
        existing=self.repository.run_challenge(run_id) or {}
        run=self.repository.get_run(run_id)
        if (
            str(existing.get("challenge_definition_id") or "")==definition.challenge_id
            and run is not None and run.run_kind=="challenge"
        ):
            return definition
        now=self.now_fn()
        self.repository.link_run_challenge(run_id,definition.challenge_id,at=now)
        self.repository.update_run_contract(
            run_id,run_kind="challenge",
            rules={"challenge":definition.name,"challenge_definition_id":definition.challenge_id},at=now,
        )
        payload={
            "challenge_id":definition.challenge_id,"challenge_name":definition.name,
            "current_game_source":"stream_title","playthrough_type":"challenge",
        }
        self._emit("challenge_definition_reused",payload)
        return definition

    def add_rule(
        self, challenge_id: str, text: str, *, provenance: str, confidence: float,
        rule_type: str = "CHALLENGE_RULE",
    ) -> ChallengeDefinition:
        definition = self.repository.get_challenge_definition(challenge_id)
        if definition is None: raise KeyError(challenge_id)
        normalized = _normalize(text)
        active = [rule for rule in definition.rules if rule.get("status", "ACTIVE") == "ACTIVE"]
        duplicate = next((rule for rule in active if rule.get("normalized_key") == normalized), None)
        if duplicate is not None:
            return definition
        now = self.now_fn()
        rules = [dict(rule) for rule in definition.rules]
        rules.append(self._new_rule(text, provenance=provenance, confidence=confidence, rule_type=rule_type))
        updated = ChallengeDefinition(
            **{**definition.to_dict(), "rules": tuple(rules), "updated_at": now, "version": definition.version + 1}
        )
        return self.repository.save_challenge_definition(updated)

    def correct_rule(self, challenge_id: str, correction_text: str) -> ChallengeDefinition | None:
        definition = self.repository.get_challenge_definition(challenge_id)
        if definition is None: return None
        active = [dict(rule) for rule in definition.rules if rule.get("status", "ACTIVE") == "ACTIVE"]
        target = max(active, key=lambda rule: self._rule_similarity(str(rule.get("text") or ""), correction_text), default=None)
        if target is None or self._rule_similarity(str(target.get("text") or ""), correction_text) < 0.28:
            return self.add_rule(challenge_id, correction_text, provenance="owner_explicit", confidence=1.0, rule_type="CHALLENGE_CORRECTION")
        now = self.now_fn()
        rules = []
        for rule in definition.rules:
            item = dict(rule)
            if item.get("rule_id") == target.get("rule_id"):
                item.update({"status": "SUPERSEDED", "updated_at": now, "version": int(item.get("version") or 1) + 1})
            rules.append(item)
        replacement = self._new_rule(
            correction_text, provenance="owner_explicit", confidence=1.0, rule_type="CHALLENGE_CORRECTION",
        )
        replacement["supersedes_rule_id"] = target.get("rule_id")
        rules.append(replacement)
        updated = ChallengeDefinition(
            **{**definition.to_dict(), "rules": tuple(rules), "updated_at": now, "version": definition.version + 1}
        )
        return self.repository.save_challenge_definition(updated)

    def context_for_run(self, run_id: str) -> dict[str, Any]:
        link = self.repository.run_challenge(run_id) or {}
        definition = self.repository.get_challenge_definition(str(link.get("challenge_definition_id") or ""))
        return {
            "challenge_definition": definition.to_dict() if definition else {},
            "run_challenge_state": dict(link.get("state") or {}),
            "run_overrides": list(link.get("overrides") or []),
        }

    def _close_capture(self, reason: str, result: dict[str, Any]) -> None:
        result["challenge_capture_closed"] = True
        result["capture_close_reason"] = reason
        self.capture = None
        self._emit("challenge_capture_closed", result)

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        clean = {"event": event, **payload}
        print(
            f"[HEBE][CHALLENGE_CONTEXT] event={event} challenge_id={payload.get('challenge_id') or ''} "
            f"run_id={getattr(self.capture, 'run_id', '')}", flush=True,
        )
        log_jsonl_event("challenge_context", clean)

    def _new_rule(self, text: str, *, provenance: str, confidence: float, rule_type: str) -> dict[str, Any]:
        now = self.now_fn()
        return {
            "rule_id": f"challenge_rule_{uuid.uuid4().hex}", "text": " ".join(str(text).split()),
            "normalized_key": _normalize(text), "rule_type": rule_type, "status": "ACTIVE",
            "provenance": provenance, "confidence": float(confidence), "created_at": now,
            "updated_at": now, "version": 1, "supersedes_rule_id": "",
        }

    @staticmethod
    def _challenge_mentioned(text: str) -> bool:
        return bool(re.search(r"\b(?:desafio|challenge|reto|run\s+de\s+reto)\b", text))

    @staticmethod
    def _challenge_name(text: str) -> str:
        match = re.search(
            r"\b(?:con|haciendo|es)\s+(?:un|una|el|la)?\s*(?:desaf(?:i|\u00ed)o|challenge|reto)"
            r"(?:\s+(?:llamad[oa]|de\s+nombre))?\s+([^,.;!?]+)",
            text, flags=re.IGNORECASE,
        )
        if not match: return ""
        name = " ".join(match.group(1).strip().split())
        if _normalize(name) in {"", "este", "esta", "run", "partida", "y", "con reglas"}:
            return ""
        return name[:120]

    @staticmethod
    def _starts_rule_capture(text: str) -> bool:
        return bool(
            re.search(r"\b(?:repaso|repasar|explico|explicar|cuento|contar|recordamos|revisamos)\b", text)
            and re.search(r"\b(?:reglas|normas|reto|challenge|desafio)\b", text)
        )

    @staticmethod
    def _closes_rule_capture(text: str) -> bool:
        return bool(re.search(r"\b(?:fin\s+de\s+las\s+reglas|eso\s+es\s+todo|ya\s+estan\s+todas|terminamos\s+el\s+repaso)\b", text))

    @staticmethod
    def _clear_topic_shift(text: str) -> bool:
        return bool(re.search(r"\b(?:cambiando\s+de\s+tema|dejando\s+el\s+reto|otra\s+cosa)\b", text))

    @staticmethod
    def _looks_like_rule(text: str) -> bool:
        if len(text.split()) < 3: return False
        return bool(re.search(
            r"\b(?:no\s+(?:se\s+)?puede|hay\s+que|tenemos\s+que|solo\s+se|esta\s+permitido|"
            r"esta\s+prohibido|obligatorio|regla|decide|sirve\s+para|cada\s+vez|cuando)\b",
            text,
        ))

    @staticmethod
    def _is_correction(text: str) -> bool:
        return bool(re.search(r"\b(?:corrijo|rectifico|en\s+realidad|no\s+es\s+verdad|no\s+sirve|no\s+decide|no\s+controla)\b", text))

    @staticmethod
    def _is_run_override(text: str) -> bool:
        return bool(re.search(r"\b(?:en\s+esta\s+(?:run|partida)|solo\s+para\s+esta\s+(?:run|partida)|esta\s+vez)\b", text))

    @staticmethod
    def _rule_similarity(left: str, right: str) -> float:
        a, b = _normalize(left), _normalize(right)
        stop = {"la", "el", "un", "una", "de", "del", "para", "no", "si", "se", "que", "es", "sirve"}
        at = {token for token in a.split() if token not in stop}
        bt = {token for token in b.split() if token not in stop}
        overlap = len(at & bt) / max(1, min(len(at), len(bt)))
        return max(overlap, SequenceMatcher(None, a, b).ratio())
