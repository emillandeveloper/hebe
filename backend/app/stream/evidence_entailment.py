from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any


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
        entailed: list[str] = []
        unsupported: list[str] = []
        wrong_referent: list[str] = []
        contradicted: list[str] = []
        for claim in claims:
            aliases = self._MECHANICS[claim]
            exact_match = any(_normalize(alias) in evidence_norm for alias in aliases)
            if claim == "player_low_level" and subject in {"enemies", "unknown_plural_entity"}:
                wrong_referent.append(claim)
            elif claim == "enemy_low_level" and subject == "owner_player":
                wrong_referent.append(claim)
            elif claim in unsupported_evidence:
                unsupported.append(claim)
            elif exact_match:
                entailed.append(claim)
            else:
                unsupported.append(claim)
        violations = [
            *(f"unsupported:{item}" for item in unsupported),
            *(f"contradicted:{item}" for item in contradicted),
            *(f"wrong_referent:{item}" for item in wrong_referent),
        ]
        action = "suppress" if wrong_referent or contradicted else "repair" if unsupported else "allow"
        result = (
            "wrong_referent" if wrong_referent else "contradicted" if contradicted
            else "unsupported" if unsupported else "entailed" if claims
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
