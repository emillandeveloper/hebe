from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Callable

from app.cognitive.action_plan import ActionPlan
from app.cognitive.input_event import InputEvent
from app.cognitive.wake_name_resolver import WakeNameResolver
from app.stream.intent_parser import StreamIntentParser


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
        target_resolver: Callable[[str], object | None] | None = None,
    ):
        self.known_targets_provider = known_targets_provider
        self.normalize_target = normalize_target
        self.build_shoutout_command = build_shoutout_command
        self.stream_state_provider = stream_state_provider
        self.target_resolver = target_resolver
        self.intent_parser = StreamIntentParser()
        self.wake_resolver = WakeNameResolver()

    def plan(self, input_event: InputEvent) -> ActionPlan | None:
        raw_text = self._strip_wakeword_preserve(input_event.normalized_text)
        candidates = self.intent_parser.parse(raw_text, raw_text=raw_text)
        print(f"[HEBE][COG] intent_candidates={[candidate.intent for candidate in candidates]!r}", flush=True)
        for candidate in candidates:
            if candidate.intent in {"stream_ambient_stt_enabled", "stream_ambient_stt_disabled"}:
                return self._plan_stt_ambient(candidate)
            if candidate.intent == "twitch_shoutout":
                return self._plan_shoutout(candidate, input_event)
            if candidate.intent == "stream_chat_message":
                return self._plan_chat_message(candidate)
        return None

    def _plan_stt_ambient(self, candidate) -> ActionPlan:
        return ActionPlan(
            action_type=candidate.intent,
            status="complete",
            confidence=candidate.confidence,
            reason=candidate.reason or "ok",
            context_checks=self._stream_context_checks(),
        )

    def _plan_shoutout(self, candidate, input_event: InputEvent) -> ActionPlan | None:
        intent_confidence = float(candidate.confidence)
        target_raw = (candidate.entities or {}).get("target_text") or None
        checks = self._stream_context_checks()
        if target_raw is None:
            print("[HEBE][ENTITY] extracted={'target_text': None}", flush=True)
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

        print(f"[HEBE][ENTITY] extracted={{'target_text': {target_raw!r}}}", flush=True)
        guard_ok, guard_reason, guarded_raw = self._promotion_target_guard(target_raw, resolved=False)
        if not guard_ok:
            print(f"[HEBE][PROMOTION_CLARIFY] reason={guard_reason} candidates=[]", flush=True)
            return ActionPlan(
                action_type="twitch_shoutout",
                status="needs_confirmation",
                confidence=min(intent_confidence, 0.45),
                requires_stream=True,
                reason=guard_reason,
                candidates=[],
                context_checks=checks,
                slots={"target_raw": target_raw, "target_text": guarded_raw, "resolved_username": None},
                missing_slots=["target"],
            )
        target_raw = guarded_raw
        target, target_confidence, candidates, reason = self._resolve_target(target_raw)
        print(
            "[HEBE][PROMOTION_RESOLVE] "
            f"target_text={target_raw!r} resolved={target!r} "
            f"confidence={target_confidence:.3f} reason={reason} candidates={candidates!r}",
            flush=True,
        )
        confidence = min(1.0, (intent_confidence * 0.55) + (target_confidence * 0.45))
        if reason == "ambiguous_target":
            print(f"[HEBE][PROMOTION_CLARIFY] reason=ambiguous candidates={candidates!r}", flush=True)
            return ActionPlan(
                action_type="twitch_shoutout",
                status="needs_confirmation",
                confidence=confidence,
                target=target,
                requires_stream=True,
                reason=reason,
                candidates=candidates,
                context_checks=checks,
                slots={"target_raw": target_raw, "target_text": target_raw, "resolved_username": target},
            )
        if not target:
            print(f"[HEBE][PROMOTION_CLARIFY] reason={reason or 'not_found'} candidates={candidates!r}", flush=True)
            return ActionPlan(
                action_type="twitch_shoutout",
                status="needs_confirmation",
                confidence=confidence,
                requires_stream=True,
                reason=reason or "target_unclear",
                candidates=candidates,
                context_checks=checks,
                slots={"target_raw": target_raw, "target_text": target_raw, "resolved_username": target},
                missing_slots=["target"],
            )

        final_ok, final_reason, guarded_target = self._promotion_target_guard(target, resolved=True)
        if not final_ok:
            print(f"[HEBE][PROMOTION_CLARIFY] reason={final_reason} candidates={candidates!r}", flush=True)
            return ActionPlan(
                action_type="twitch_shoutout",
                status="needs_confirmation",
                confidence=min(confidence, 0.45),
                target=None,
                requires_stream=True,
                reason=final_reason,
                candidates=candidates,
                context_checks=checks,
                slots={"target_raw": target_raw, "target_text": target_raw, "resolved_username": target},
                missing_slots=["target"],
            )
        target = guarded_target
        command = self.build_shoutout_command(target)
        status = "complete" if confidence >= 0.78 else "needs_confirmation"
        if reason in {"medium_confidence", "unverified_username"}:
            status = "needs_confirmation"
        if status != "complete":
            print(f"[HEBE][PROMOTION_CLARIFY] reason=medium_confidence candidates={candidates!r}", flush=True)
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
            slots={
                "target_raw": target_raw,
                "target_text": target_raw,
                "resolved_username": target,
                "requires_confirmation": status != "complete",
            },
        )

    def _promotion_target_guard(self, target: str, *, resolved: bool) -> tuple[bool, str, str]:
        raw = str(target or "").strip().lstrip("@")
        normalized = normalize_command_text(raw)
        compact = _compact(raw)
        reason = "accepted"
        accepted = True
        guarded = raw
        command_words = {"haz", "hazle", "dale", "tira", "promo", "promocion", "promociona", "shoutout", "so"}
        sentence_markers = {
            "juego", "partida", "jueves", "familia", "anime", "chat", "viewer", "espectador",
            "combate", "directo", "stream", "vamos", "estoy", "esta", "donde", "cuando",
        }
        insult_markers = {"idiot", "imbecil", "gilipoll", "cabron", "tonto", "tonta", "estupido", "estupida"}
        tokens = set(normalized.split())
        if not normalized:
            accepted, reason = False, "missing_target"
        elif len(compact) == 1:
            accepted, reason = False, "ambiguous_single_letter_target"
        elif normalized in {"h", "hache"} or re.fullmatch(r"(?:a|al|a la|a el)\s+h(?:ache)?", normalized):
            accepted, reason = False, "ambiguous_single_letter_target"
        elif tokens & insult_markers:
            accepted, reason = False, "invalid_target"
        elif tokens & command_words or re.search(r"(?:haz|promo|shoutout|so)", compact):
            accepted, reason = False, "command_words_in_target"
        elif tokens & sentence_markers and not resolved:
            accepted, reason = False, "sentence_fragment"
        elif resolved and not re.fullmatch(r"[A-Za-z0-9_]{3,25}", raw):
            accepted, reason = False, "invalid_twitch_username"
        elif not resolved and re.fullmatch(r"[A-Za-z0-9_]{3,25}", raw):
            guarded = raw
        print(
            f"[HEBE][PROMOTION_TARGET_GUARD] accepted={str(accepted).lower()} target={guarded!r} reason={reason}",
            flush=True,
        )
        return accepted, reason, guarded

    def _plan_chat_message(self, candidate) -> ActionPlan:
        message = str((candidate.entities or {}).get("message") or "").strip()
        checks = self._stream_context_checks()
        if not message:
            return ActionPlan(
                action_type="stream_chat_message",
                status="needs_confirmation",
                confidence=float(candidate.confidence),
                requires_stream=True,
                reason="missing_message",
                missing_slots=["message"],
                context_checks=checks,
                slots={"message": ""},
            )
        return ActionPlan(
            action_type="stream_chat_message",
            status="complete",
            confidence=float(candidate.confidence),
            target="twitch_chat",
            requires_stream=True,
            reason="ok",
            context_checks=checks,
            slots={"message": message},
        )

    def _resolve_target(self, raw_target: str) -> tuple[str | None, float, list[str], str]:
        raw = str(raw_target or "").strip().lstrip("@")
        marker = normalize_command_text(raw)
        raw_key = _compact(raw)
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

        resolver = getattr(self, "target_resolver", None)
        if len(raw_key) == 1:
            aliases = getattr(resolver, "aliases", {}) if resolver is not None else {}
            alias_target = aliases.get(raw_key) if isinstance(aliases, dict) else None
            if alias_target:
                target = self.normalize_target(str(alias_target))
                return target, 0.99, [target], "alias"
            known_matches = [original for original, key in self._known_pairs() if raw_key in key][:4]
            return None, 0.0, known_matches, "ambiguous_single_letter_target"
        if callable(resolver):
            resolved = resolver(raw)
            if resolved is not None:
                username = _get_resolution_value(resolved, "username")
                confidence = float(_get_resolution_value(resolved, "confidence") or 0.0)
                candidates = list(_get_resolution_value(resolved, "candidates") or [])
                reason = str(_get_resolution_value(resolved, "reason") or "target_unclear")
                source = str(_get_resolution_value(resolved, "source") or reason)
                if username:
                    target = self.normalize_target(str(username))
                    if reason == "ambiguous_target":
                        return target, confidence, candidates or [target], "ambiguous_target"
                    if confidence >= 0.82:
                        return target, confidence, candidates or [target], reason
                    if confidence >= 0.68:
                        return target, confidence, candidates or [target], "medium_confidence"
                elif reason == "missing_target":
                    return None, confidence, candidates, reason

        normalized = self.normalize_target(raw)
        known = self._known_pairs()

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
            if "_" in normalized:
                return normalized, 0.62, [normalized], "unverified_username"
            return normalized, 0.82, [normalized], "valid_username"
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
        original = str(text or "").strip()
        resolution = self.wake_resolver.resolve(
            raw_text=original,
            normalized_text=original,
            source="planner",
            command_markers=self.intent_parser.shoutout_concepts | self.intent_parser.enable_concepts | self.intent_parser.disable_concepts,
        )
        if not resolution.matched_name:
            return original
        stripped = re.sub(
            r"^\s*(?:hebe|ebe|eve|jebe|heve|e\.?\s*b\.?)[\s,;:.-]+",
            "",
            original,
            flags=re.IGNORECASE,
        ).strip()
        if stripped == original:
            stripped = re.sub(
                r"[\s,;:.-]+(?:hebe|ebe|eve|jebe|heve|e\.?\s*b\.?)\s*$",
                "",
                original,
                flags=re.IGNORECASE,
            ).strip()
        return stripped or original


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", normalize_command_text(value).lstrip("@"))


def _similar(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _get_resolution_value(resolved: object, key: str):
    if resolved is None:
        return None
    if isinstance(resolved, dict):
        return resolved.get(key)
    return getattr(resolved, key, None)
