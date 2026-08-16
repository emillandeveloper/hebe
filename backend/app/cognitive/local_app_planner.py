from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from app.cognitive.action_plan import ActionPlan
from app.cognitive.input_event import InputEvent
from app.cognitive.input_interpretation import InputInterpreter
from app.cognitive.wake_name_resolver import WakeNameResolver
from app.services.direct_stt_command import (
    DirectUtteranceIntentFamily,
    parse_direct_stt_command,
)


OPEN_APP_MARKERS = {
    "abre",
    "abrir",
    "inicia",
    "iniciar",
    "arranca",
    "arrancar",
    "lanza",
    "lanzar",
    "ejecuta",
    "ejecutar",
    "open",
    "start",
    "launch",
    "run",
}

class LocalAppActionPlanner:
    def __init__(self, wake_resolver: WakeNameResolver | None = None):
        self.wake_resolver = wake_resolver or WakeNameResolver()

    def plan(self, input_event: InputEvent, *, is_awake: bool = True) -> ActionPlan | None:
        direct_metadata = dict((input_event.stt_metadata or {}).get("direct_stt_command") or {})
        canonical_text = str(
            direct_metadata.get("command_text")
            or input_event.raw_text
            or input_event.normalized_text
        )
        parsed = parse_direct_stt_command(
            canonical_text,
            ambient_text=input_event.raw_text,
            event_id=direct_metadata.get("event_id"),
        )
        normalized = self._normalize(canonical_text)
        if not normalized:
            return None

        resolution = self.wake_resolver.resolve(
            raw_text=input_event.raw_text,
            normalized_text=normalized,
            source=input_event.source,
            is_sleeping=not is_awake,
            command_markers=OPEN_APP_MARKERS,
        )
        print(
            "[HEBE][WAKE_RESOLVER] "
            f"addressed_to_hebe={str(resolution.addressed_to_hebe).lower()} "
            f"matched_name={resolution.matched_name}",
            flush=True,
        )

        interpretation = getattr(input_event, "interpretation", None)
        if interpretation is None and getattr(input_event, "envelope", None) is not None:
            interpretation = input_event.envelope.interpretation
        if interpretation is None:
            interpretation = InputInterpreter().interpret_event(
                input_event,
                addressed_to_hebe=bool(resolution.addressed_to_hebe),
                explicit_command_mode=input_event.source in {
                    "ui", "typed_ui", "owner_ui", "button", "stt_voice",
                    "owner_stt_direct", "owner_stt_command",
                },
                direct_result=parsed,
            )
        if not interpretation.authorized_action_command:
            print(
                "[HEBE][APP_PLAN_SKIPPED] "
                f"speech_act={interpretation.speech_act.value} reason=canonical_command_not_authorized",
                flush=True,
            )
            return None

        if not is_awake and not resolution.wake_command:
            return None

        target = parsed.raw_target if (
            parsed.detected_intent_family == DirectUtteranceIntentFamily.APPLICATION_ACTION.value
        ) else None
        candidates = ["open_application"] if target else []
        print(f"[HEBE][COG] intent_candidates={candidates!r}", flush=True)
        if not target:
            return None

        addressed_or_trusted = resolution.addressed_to_hebe or input_event.source in {
            "ui", "typed_ui", "stt_voice", "voice", "button",
            "owner_stt_direct", "owner_stt_command", "owner_stt_followup",
        }
        if not addressed_or_trusted:
            return None

        print(f"[HEBE][ENTITY] application_target={target}", flush=True)
        confidence = 1.0 if resolution.addressed_to_hebe else 0.84

        return ActionPlan(
            action_type="open_application",
            status="complete",
            confidence=confidence,
            target=target,
            command=None,
            reason="target_extracted",
            slots={"application_target": target},
            context_checks={
                "source": input_event.source,
                "awake": is_awake,
            },
            missing_slots=[],
        )

    def command_markers(self) -> Iterable[str]:
        return OPEN_APP_MARKERS

    def _normalize(self, text: str) -> str:
        value = str(text or "").strip().lower()
        value = "".join(
            ch for ch in unicodedata.normalize("NFKD", value)
            if not unicodedata.combining(ch)
        )
        value = re.sub(r"[^a-z0-9_ ]+", " ", value)
        return " ".join(value.split())
