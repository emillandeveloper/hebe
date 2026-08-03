from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from app.stream.game_advice_gate import GameAdviceGate


def _normalize(value: str) -> str:
    raw = "".join(
        char for char in unicodedata.normalize("NFKD", str(value or "").casefold())
        if not unicodedata.combining(char)
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", raw).split())


@dataclass(frozen=True)
class EvidenceEntailmentDecision:
    passed: bool
    action: str
    claims: list[str] = field(default_factory=list)
    entailed: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    contradicted: list[str] = field(default_factory=list)
    wrong_referent: list[str] = field(default_factory=list)
    result: str = "reasonable_low-risk_reaction"
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceEntailmentGuard:
    """Checks candidate gameplay claims against the exact selected anchor."""

    _MECHANICS = {
        "autopotion": ("autopotion", "autopocion", "auto potion", "auto pocion"),
        "healing": ("se cura", "regenera", "heals", "healing", "pocion", "potion"),
        "player_low_level": (
            "vas bajo de nivel", "estas bajo de nivel", "eres nivel bajo",
            "leo esta bajo de nivel", "tu nivel es bajo", "player is low level",
        ),
        "enemy_low_level": (
            "enemigos de nivel bajo", "enemigos son nivel bajo", "son de nivel bajo",
            "low level enemies", "they are low level",
        ),
        "enemy_health_not_decreasing": (
            "no baja su vida", "no le baja la vida", "su vida no baja",
            "enemy health is not decreasing",
        ),
    }

    _PURE_EMOTIONAL = re.compile(
        r"^(?:vaya|vaya momento|uf+|uff|uf+ que tension|madre mia|bien|brutal|vamos|que tension|menudo momento|"
        r"eso tiene pinta de ser (?:durisimo|dificil|intenso)|"
        r"wow|oof|nice|damn|lets go|that was close|close one|that looks rough)[!. ]*$"
    )

    def __init__(self, advice_gate: GameAdviceGate | None = None) -> None:
        self.advice_gate = advice_gate or GameAdviceGate()

    def evaluate(self, candidate: str, anchor_evidence: dict | None) -> EvidenceEntailmentDecision:
        evidence = dict(anchor_evidence or {})
        candidate_norm = _normalize(candidate)
        exact = [
            str(item) for item in (
                list(evidence.get("exact_supported_claims") or [])
                + list(evidence.get("raw_owner_fragments") or [])
                + list(evidence.get("supported_claims") or [])
            ) if str(item).strip()
        ]
        evidence_norm = " ".join(_normalize(item) for item in exact)
        subject = str(evidence.get("extracted_subject") or "").casefold()
        unsupported_evidence = " ".join(
            _normalize(item) for item in evidence.get("unsupported_claims") or []
        )
        claims = [
            claim for claim, aliases in self._MECHANICS.items()
            if any(_normalize(alias) in candidate_norm for alias in aliases)
        ]
        claims = list(dict.fromkeys([
            *claims,
            *self.advice_gate.detect_mechanics(candidate),
            *self.advice_gate.extract_substantive_claims(candidate),
        ]))
        entailed: list[str] = []
        unsupported: list[str] = []
        wrong_referent: list[str] = []
        contradicted: list[str] = []
        terminal = bool(evidence.get("terminal"))
        current_state = _normalize(str(evidence.get("current_state") or ""))
        for claim in claims:
            aliases = self._MECHANICS.get(claim, (claim.replace("_", " "),))
            exact_match = any(_normalize(alias) in evidence_norm for alias in aliases)
            if not exact_match:
                exact_match = claim in set(self.advice_gate.extract_substantive_claims(evidence_norm))
            if claim == "enemy_alive_assumption" and (
                terminal or current_state in {"enemy dead", "battle ended", "puzzle completed"}
            ):
                contradicted.append(claim)
            if claim == "player_low_level" and subject in {"enemies", "unknown_plural_entity"}:
                wrong_referent.append(claim)
            elif claim == "enemy_low_level" and subject == "owner_player":
                wrong_referent.append(claim)
            elif claim in unsupported_evidence:
                unsupported.append(claim)
            elif claim in contradicted:
                continue
            elif exact_match:
                entailed.append(claim)
            else:
                unsupported.append(claim)
        pure_emotional = bool(self._PURE_EMOTIONAL.fullmatch(candidate_norm))
        extraction_failure = bool(candidate_norm and not claims and not pure_emotional)
        violations = [
            *(f"unsupported:{item}" for item in unsupported),
            *(f"contradicted:{item}" for item in contradicted),
            *(f"wrong_referent:{item}" for item in wrong_referent),
            *(["extraction_failure"] if extraction_failure else []),
        ]
        action = "suppress" if wrong_referent or contradicted else "repair" if unsupported or extraction_failure else "allow"
        result = (
            "wrong_referent" if wrong_referent else "contradicted" if contradicted
            else "unsupported" if unsupported else "extraction_failure" if extraction_failure else "entailed" if claims
            else "reasonable_low-risk_reaction"
        )
        decision = EvidenceEntailmentDecision(
            passed=not violations,
            action=action,
            claims=claims,
            entailed=entailed,
            unsupported=unsupported,
            contradicted=contradicted,
            wrong_referent=wrong_referent,
            result=result,
            violations=violations,
        )
        print(
            "[HEBE][EVIDENCE_ENTAILMENT_GUARD] "
            f"claims={claims!r} entailed={entailed!r} unsupported={unsupported!r} "
            f"wrong_referent={wrong_referent!r} action={action}",
            flush=True,
        )
        return decision
