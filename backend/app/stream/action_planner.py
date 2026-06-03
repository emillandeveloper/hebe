from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Callable

from app.cognitive.action_plan import ActionPlan
from app.cognitive.input_event import InputEvent


def normalize_command_text(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() or ch == "_" else " " for ch in str(text or "").strip().lower())
    return " ".join(cleaned.split())


class StreamActionPlanner:
    def __init__(
        self,
        *,
        known_targets_provider: Callable[[], list[str]],
        normalize_target: Callable[[str], str],
        build_shoutout_command: Callable[[str], str],
        stream_state_provider: Callable[[], object | None],
    ):
        self.known_targets_provider = known_targets_provider
        self.normalize_target = normalize_target
        self.build_shoutout_command = build_shoutout_command
        self.stream_state_provider = stream_state_provider

    def plan(self, input_event: InputEvent) -> ActionPlan | None:
        raw_text = self._strip_wakeword_preserve(input_event.normalized_text)
        text = normalize_command_text(raw_text)
        stt_ambient = self._plan_stt_ambient(text)
        if stt_ambient is not None:
            return stt_ambient
        shoutout = self._plan_shoutout(text, raw_text, input_event)
        if shoutout is not None:
            return shoutout
        return None

    def _plan_stt_ambient(self, text: str) -> ActionPlan | None:
        if re.match(r"^(?:desactiva|apaga|pausa|quita)\s+(?:el\s+)?stt\s+ambiental(?:\s+(?:de|del|para)\s+stream)?$", text):
            return ActionPlan(
                action_type="stream_ambient_stt_disabled",
                status="complete",
                confidence=0.98,
                reason="ok",
                context_checks=self._stream_context_checks(),
            )
        if re.match(r"^(?:activa|enciende|reanuda|pon)\s+(?:el\s+)?stt\s+ambiental(?:\s+(?:de|del|para)\s+stream)?$", text):
            return ActionPlan(
                action_type="stream_ambient_stt_enabled",
                status="complete",
                confidence=0.98,
                reason="ok",
                context_checks=self._stream_context_checks(),
            )
        return None

    def _plan_shoutout(self, text: str, raw_text: str, input_event: InputEvent) -> ActionPlan | None:
        intent_confidence, target_raw = self._match_shoutout_intent(text, raw_text)
        if intent_confidence <= 0:
            return None

        checks = self._stream_context_checks()
        if target_raw is None:
            return ActionPlan(
                action_type="twitch_shoutout",
                status="needs_confirmation",
                confidence=intent_confidence,
                requires_stream=True,
                reason="missing_target",
                missing_slots=["target"],
                context_checks=checks,
                slots={"target_raw": ""},
            )

        target, target_confidence, candidates, reason = self._resolve_target(target_raw)
        confidence = min(1.0, (intent_confidence * 0.55) + (target_confidence * 0.45))
        if reason == "ambiguous_target":
            return ActionPlan(
                action_type="twitch_shoutout",
                status="needs_confirmation",
                confidence=confidence,
                target=target,
                requires_stream=True,
                reason=reason,
                candidates=candidates,
                context_checks=checks,
                slots={"target_raw": target_raw},
            )
        if not target:
            return ActionPlan(
                action_type="twitch_shoutout",
                status="needs_confirmation",
                confidence=confidence,
                requires_stream=True,
                reason=reason or "target_unclear",
                candidates=candidates,
                context_checks=checks,
                slots={"target_raw": target_raw},
                missing_slots=["target"],
            )

        command = self.build_shoutout_command(target)
        status = "complete" if confidence >= 0.78 else "needs_confirmation"
        return ActionPlan(
            action_type="twitch_shoutout",
            status=status,
            confidence=confidence,
            target=target,
            command=command,
            requires_stream=True,
            reason="ok" if status == "complete" else "medium_confidence",
            candidates=candidates,
            context_checks=checks,
            slots={"target_raw": target_raw},
        )

    def _match_shoutout_intent(self, text: str, raw_text: str) -> tuple[float, str | None]:
        patterns = [
            r"^(?:haz|hazle|dale|manda|pon)\s+(?:una?\s+)?(?:promo|promocion|so|shoutout)\s+(?:a\s+|al\s+)?(.+)$",
            r"^(?:haz|hazle|dale)\s+(?:una?\s+)?(?:promo|so)\s*$",
            r"^promociona\s+(?:a\s+)?(.+)$",
            r"^recomienda\s+(?:a\s+)?(.+)$",
            r"^shoutout\s+(?:to\s+|a\s+)?(.+)$",
            r"^give\s+a\s+shoutout\s+to\s+(.+)$",
            r"^so\s+(.+)$",
        ]
        raw_patterns = [
            r"^(?:haz|hazle|dale|manda|pon)\s+(?:una?\s+)?(?:promo|promocion|so|shoutout)\s+(?:a\s+|al\s+)?(.+)$",
            r"^(?:haz|hazle|dale)\s+(?:una?\s+)?(?:promo|so)\s*$",
            r"^promociona\s+(?:a\s+)?(.+)$",
            r"^recomienda\s+(?:a\s+)?(.+)$",
            r"^shoutout\s+(?:to\s+|a\s+)?(.+)$",
            r"^give\s+a\s+shoutout\s+to\s+(.+)$",
            r"^so\s+(.+)$",
        ]
        for pattern, raw_pattern in zip(patterns, raw_patterns):
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            raw_match = re.match(raw_pattern, raw_text.strip(), flags=re.IGNORECASE)
            if raw_match and raw_match.lastindex:
                target = raw_match.group(1).strip()
            else:
                target = match.group(1).strip() if match.lastindex else None
            return 0.95, target
        return 0.0, None

    def _resolve_target(self, raw_target: str) -> tuple[str | None, float, list[str], str]:
        raw = str(raw_target or "").strip().lstrip("@")
        marker = normalize_command_text(raw)
        if marker in {
            "ultimo raider",
            "último raider",
            "al ultimo raider",
            "al último raider",
            "quien nos ha raideado",
            "a quien nos ha raideado",
            "last raider",
            "the last raider",
        }:
            stream = self.stream_state_provider()
            raid = getattr(stream, "last_raid_event", None) if stream is not None else None
            target = (raid or {}).get("user_login") or (raid or {}).get("display_name")
            if target:
                return self.normalize_target(target), 1.0, [target], "last_raider"
            return None, 0.0, [], "missing_target"
        normalized = self.normalize_target(raw)
        known = self._known_pairs()
        raw_key = _compact(raw)

        for original, key in known:
            if raw_key == key:
                return self.normalize_target(raw), 1.0, [original], "exact_target"
            if raw_key == f"a{key}":
                return self.normalize_target(original), 1.0, [original], "exact_target"

        scored: list[tuple[float, str]] = []
        variants = {raw_key}
        if raw_key.startswith("a") and len(raw_key) > 4:
            variants.add(raw_key[1:])
        for original, key in known:
            score = max(_similar(variant, key) for variant in variants)
            if score >= 0.78:
                scored.append((score, original))
        scored.sort(reverse=True, key=lambda item: item[0])
        if scored:
            best = scored[0][0]
            matches = [name for score, name in scored if best - score <= 0.04]
            if len(matches) > 1:
                return self.normalize_target(matches[0]), best, matches[:4], "ambiguous_target"
            return self.normalize_target(matches[0]), best, matches, "fuzzy_known_target"

        if normalized and re.fullmatch(r"[A-Za-z0-9_]{3,25}", normalized):
            if raw_key.startswith("a") and len(raw_key) > 4:
                return None, 0.25, [], "target_unclear"
            return normalized, 0.78, [normalized], "valid_username"
        return None, 0.0, [], "invalid_target"

    def _known_pairs(self) -> list[tuple[str, str]]:
        values: list[tuple[str, str]] = []
        seen: set[str] = set()
        for target in self.known_targets_provider():
            raw = str(target or "").strip().lstrip("@")
            key = _compact(raw)
            if key and key not in seen:
                values.append((raw, key))
                seen.add(key)
        return values

    def _stream_context_checks(self) -> dict:
        stream = self.stream_state_provider()
        return {
            "stream_enabled": bool(getattr(stream, "enabled", False)) if stream is not None else False,
            "is_live": bool(getattr(stream, "is_live", False)) if stream is not None else False,
            "live_status_known": bool(getattr(stream, "live_status_known", False)) if stream is not None else False,
            "presence_mode": getattr(stream, "presence_mode", None) if stream is not None else None,
        }

    def _strip_wakeword_preserve(self, text: str) -> str:
        return re.sub(r"^\s*(?:hebe|ebe|eve|jebe|heve)[\s,;:.-]+", "", str(text or "").strip(), flags=re.IGNORECASE).strip()


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", normalize_command_text(value).lstrip("@"))


def _similar(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()
