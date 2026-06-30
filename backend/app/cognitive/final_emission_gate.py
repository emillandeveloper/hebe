from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable
import re


class OutputRoute(StrEnum):
    OBSERVE_ONLY = "observe_only"
    LOCAL_UI_DEBUG_ONLY = "local_ui_debug_only"
    LOCAL_OWNER_REPLY = "local_owner_reply"
    TWITCH_TEXT_REPLY = "twitch_text_reply"
    STREAM_TTS_REPLY = "stream_tts_reply"
    TWITCH_ACTION_ONLY = "twitch_action_only"
    SUPPRESS = "suppress"


TEXT_ROUTES = {
    OutputRoute.LOCAL_UI_DEBUG_ONLY,
    OutputRoute.LOCAL_OWNER_REPLY,
    OutputRoute.TWITCH_TEXT_REPLY,
    OutputRoute.STREAM_TTS_REPLY,
}

PUBLIC_TEXT_ROUTES = {
    OutputRoute.LOCAL_OWNER_REPLY,
    OutputRoute.TWITCH_TEXT_REPLY,
    OutputRoute.STREAM_TTS_REPLY,
}


def normalize_output_route(route: str | OutputRoute | None) -> OutputRoute:
    try:
        return OutputRoute(str(route or OutputRoute.SUPPRESS.value))
    except ValueError:
        if str(route or "").strip() == "text_only":
            return OutputRoute.TWITCH_TEXT_REPLY
        return OutputRoute.SUPPRESS


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


@dataclass(slots=True)
class FinalEmissionResult:
    emitted: bool
    route: str
    targets: list[str] = field(default_factory=list)
    event_id: str = ""
    suppressed: bool = False
    deduped: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "emitted": self.emitted,
            "route": self.route,
            "targets": list(self.targets),
            "event_id": self.event_id,
            "suppressed": self.suppressed,
            "deduped": self.deduped,
            "reason": self.reason,
        }


class FinalEmissionGate:
    """Single boundary for normal Hebe speech/text emission."""

    def __init__(self) -> None:
        self._seen_event_ids: set[str] = set()
        self._seen_message_keys: set[tuple[str, str, str]] = set()

    def emit(
        self,
        *,
        event_id: str = "",
        source: str = "",
        final_response: str = "",
        output_route: str | OutputRoute | None = None,
        output_targets: list[str] | tuple[str, ...] | None = None,
        guard_result: dict[str, Any] | None = None,
        repair_summary: dict[str, Any] | None = None,
        execution_result: dict[str, Any] | None = None,
        debug_payload: dict[str, Any] | None = None,
        emit_ui: Callable[[dict[str, Any]], None] | None = None,
        emit_debug: Callable[[dict[str, Any]], None] | None = None,
        send_twitch: Callable[[str], Any] | None = None,
        speak: Callable[[str], Any] | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> FinalEmissionResult:
        route = normalize_output_route(output_route)
        targets = [str(target) for target in (output_targets or []) if str(target or "").strip()]
        text = str(final_response or "").strip()
        event_key = str(event_id or "").strip()
        debug = dict(debug_payload or {})
        debug.update({
            "final_response": text,
            "output_route": route.value,
            "emitted_targets": list(targets),
            "guard_result": guard_result or {},
            "repair_summary": repair_summary or {},
            "execution_result": execution_result or {},
            "event_id": event_key,
        })

        def log(message: str) -> None:
            if logger is not None:
                logger(message)
            else:
                print(message, flush=True)

        if route in {OutputRoute.OBSERVE_ONLY, OutputRoute.SUPPRESS, OutputRoute.TWITCH_ACTION_ONLY}:
            reason = "observe_only" if route == OutputRoute.OBSERVE_ONLY else "suppressed_route"
            if route == OutputRoute.TWITCH_ACTION_ONLY:
                reason = "action_only"
            if emit_debug is not None:
                emit_debug({**debug, "suppress_reason": reason})
            log(f"[HEBE][FINAL_EMISSION_GATE] suppressed=true reason={reason} route={route.value} event_id={event_key}")
            return FinalEmissionResult(False, route.value, targets, event_key, suppressed=True, reason=reason)

        if route in TEXT_ROUTES and not text:
            reason = "missing_final_response"
            if emit_debug is not None:
                emit_debug({**debug, "suppress_reason": reason})
            log(f"[HEBE][FINAL_EMISSION_GATE] suppressed=true reason={reason} route={route.value} event_id={event_key}")
            return FinalEmissionResult(False, route.value, targets, event_key, suppressed=True, reason=reason)

        if route in PUBLIC_TEXT_ROUTES:
            guard_passed = True
            if isinstance(guard_result, dict) and "passed" in guard_result:
                guard_passed = bool(guard_result.get("passed"))
            if not guard_passed:
                reason = "guard_failed"
                if emit_debug is not None:
                    emit_debug({**debug, "failed_guard_response": text, "suppress_reason": reason})
                log(f"[HEBE][FINAL_EMISSION_GATE] suppressed=true reason={reason} route={route.value} event_id={event_key}")
                return FinalEmissionResult(False, route.value, targets, event_key, suppressed=True, reason=reason)

        dedupe_key = (str(source or ""), route.value, _normalize_text(text))
        if event_key and event_key in self._seen_event_ids:
            reason = "duplicate_event_id"
            log(f"[HEBE][FINAL_EMISSION_GATE] deduped=true reason={reason} route={route.value} event_id={event_key}")
            return FinalEmissionResult(False, route.value, targets, event_key, suppressed=True, deduped=True, reason=reason)
        if not event_key and text and dedupe_key in self._seen_message_keys:
            reason = "duplicate_normalized_text"
            log(f"[HEBE][FINAL_EMISSION_GATE] deduped=true reason={reason} route={route.value} event_id={event_key}")
            return FinalEmissionResult(False, route.value, targets, event_key, suppressed=True, deduped=True, reason=reason)

        if route == OutputRoute.LOCAL_UI_DEBUG_ONLY:
            if emit_debug is not None:
                emit_debug(debug)
            if event_key:
                self._seen_event_ids.add(event_key)
            elif text:
                self._seen_message_keys.add(dedupe_key)
            log(f"[HEBE][FINAL_EMISSION_GATE] emitted=true route={route.value} targets={targets} event_id={event_key}")
            return FinalEmissionResult(True, route.value, targets, event_key)

        if emit_ui is not None and "local_ui" in targets:
            emit_ui({"text": text, "source": source, "output_target": "local_ui", **debug})
        if send_twitch is not None and "twitch_chat" in targets:
            send_twitch(text)
        if speak is not None and any(target in targets for target in ("local_tts", "stream_tts")):
            speak(text)

        if event_key:
            self._seen_event_ids.add(event_key)
        elif text:
            self._seen_message_keys.add(dedupe_key)
        log(f"[HEBE][FINAL_EMISSION_GATE] emitted=true route={route.value} targets={targets} event_id={event_key}")
        return FinalEmissionResult(True, route.value, targets, event_key)
