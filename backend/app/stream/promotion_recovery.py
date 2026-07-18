from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import re
import time
import unicodedata
from typing import Any, Callable


def _norm(value: str) -> str:
    text = "".join(ch for ch in unicodedata.normalize("NFKD", str(value or "").casefold()) if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9_]+", " ", text).split())


@dataclass(slots=True)
class PromotionRecoveryResult:
    command_candidate: bool
    recovered: bool = False
    action_type: str = "promotion_shoutout"
    raw_token: str = ""
    action_token: str = ""
    target_suffix: str = ""
    resolved_target: str = ""
    confidence: float = 0.0
    reason: str = ""
    candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromotionSTTRecovery:
    ACTION_PREFIXES = ("shoutout", "promo", "prom", "pro", "so", "soa")

    def __init__(self, resolver: Callable[[str], Any] | None = None, known_targets_provider: Callable[[], list[str]] | None = None) -> None:
        self.resolver = resolver
        self.known_targets_provider = known_targets_provider or (lambda: [])

    def recover(self, text: str, *, trusted_owner: bool, addressed_to_hebe: bool, imperative: bool | None = None) -> PromotionRecoveryResult:
        normalized = _norm(text)
        imperative = bool(imperative if imperative is not None else re.search(r"\b(?:haz|hazle|dale|tira|pon)\b", normalized))
        if not trusted_owner or not addressed_to_hebe or not imperative:
            return PromotionRecoveryResult(False, reason="untrusted_or_not_direct_imperative")
        tokens = normalized.split()
        raw_token = next((token for token in tokens if self._looks_fused(token)), "")
        if not raw_token:
            return PromotionRecoveryResult(False, reason="no_approximate_promotion_token")
        action_token, suffix, action_conf = self._split(raw_token)
        result = PromotionRecoveryResult(True, raw_token=raw_token, action_token=action_token, target_suffix=suffix, confidence=action_conf, reason="missing_fused_suffix" if not suffix else "target_unresolved")
        print(f"[HEBE][PROMOTION_STT_RECOVERY] raw_token={raw_token} action_token={action_token} target_suffix={suffix} confidence={action_conf:.3f}", flush=True)
        if not suffix:
            return result
        target, confidence, candidates, reason = self._resolve(suffix)
        result.resolved_target = target; result.candidates = candidates
        result.confidence = min(1.0, action_conf * .45 + confidence * .55)
        result.recovered = bool(target and result.confidence >= .78 and reason != "ambiguous_target")
        result.reason = "recovered" if result.recovered else reason or "target_unclear"
        print(f"[HEBE][PROMOTION_STT_RECOVERY_RESOLVE] selected={target or ''} confidence={result.confidence:.3f}", flush=True)
        return result

    def _looks_fused(self, token: str) -> bool:
        if len(token) < 6: return False
        return any(token.startswith(prefix) and len(token) > len(prefix) + 2 for prefix in self.ACTION_PREFIXES)

    def _split(self, token: str) -> tuple[str, str, float]:
        # Common Spanish STT substitution: the final /o/ in "promo" is heard as
        # /a/ immediately before a viewer name ("promanuria").
        if token.startswith("proma") and len(token) > 7:
            return "proma", token[5:], .88
        best = ("", token, 0.0)
        for index in range(2, min(10, len(token) - 2) + 1):
            prefix, suffix = token[:index], token[index:]
            score = self._approx_action(prefix)
            if score > best[2]: best = (prefix, suffix, score)
        return best

    @staticmethod
    def _approx_action(token: str) -> float:
        return max(SequenceMatcher(None, token, action).ratio() for action in ("promo", "promocion", "shoutout", "so"))

    def _resolve(self, suffix: str) -> tuple[str, float, list[str], str]:
        if self.resolver is not None:
            value = self.resolver(suffix)
            get = value.get if isinstance(value, dict) else lambda key, default=None: getattr(value, key, default) if value is not None else default
            target = str(get("username", "") or get("login", "") or "")
            if target:
                return target, float(get("confidence", 0.0) or 0.0), list(get("candidates", []) or [target]), str(get("reason", "resolved") or "resolved")
        scored = []
        for target in self.known_targets_provider():
            compact = _norm(target).replace(" ", "")
            score = SequenceMatcher(None, suffix, compact).ratio()
            if suffix in compact or compact in suffix: score = max(score, .9)
            if score >= .7: scored.append((score, target))
        scored.sort(reverse=True)
        if not scored: return "", 0.0, [], "target_unclear"
        close = [target for score, target in scored if scored[0][0] - score <= .04]
        return str(scored[0][1]), float(scored[0][0]), close, "ambiguous_target" if len(close) > 1 else "known_target"


@dataclass(slots=True)
class PendingAnswerCapture:
    pending_id: str
    starts_after_tts_end: float
    capture_window_seconds: float = 12.0
    expected_answer_type: str = "twitch_username_or_viewer_alias"
    owner_voice_only: bool = True
    wake_not_required: bool = True
    minimum_target_confidence: float = .78
    actual_tts_completion_time: float = 0.0
    buffered_answers: list[dict[str, Any]] = field(default_factory=list)

    def active(self, *, now: float | None = None) -> bool:
        now = float(now if now is not None else time.time())
        start = self.actual_tts_completion_time or self.starts_after_tts_end
        return bool(start and start <= now <= start + self.capture_window_seconds)

    def buffer(self, text: str, *, source: str, timestamp: float | None = None) -> bool:
        if self.owner_voice_only and source not in {"stt_voice", "owner_stt_direct", "owner_stt_followup"}: return False
        value = str(text or "").strip()
        if not value or len(value.split()) > 5: return False
        self.buffered_answers.append({"text": value, "source": source, "timestamp": float(timestamp if timestamp is not None else time.time())})
        return True

    def mark_tts_completed(self, timestamp: float | None = None) -> None:
        self.actual_tts_completion_time = float(timestamp if timestamp is not None else time.time())

    def next_answer(self, *, now: float | None = None) -> str:
        if not self.active(now=now) or not self.buffered_answers: return ""
        return str(self.buffered_answers.pop(0)["text"])


def stream_ops_no_generic_fallback(result: PromotionRecoveryResult | None, *, routed_to_generic: bool) -> dict[str, Any]:
    candidate = bool(result and result.command_candidate)
    passed = not (candidate and routed_to_generic)
    print(f"[HEBE][STREAM_OPS_NO_GENERIC_FALLBACK] passed={str(passed).lower()}", flush=True)
    return {"passed": passed, "operation_candidate": candidate, "outcome": "parser_error" if candidate and not result.recovered else "recovered" if candidate else "not_operation"}
