from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from app.cognitive.action_plan import ActionPlan
from app.cognitive.input_event import InputEvent
from app.cognitive.wake_name_resolver import WakeNameResolver
from app.services.app_registry import resolve_whitelisted_app


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

_FILLER_TOKENS = {"si", "sí", "vale", "ok", "okay", "por", "favor"}


class LocalAppActionPlanner:
    def __init__(self, wake_resolver: WakeNameResolver | None = None):
        self.wake_resolver = wake_resolver or WakeNameResolver()

    def plan(self, input_event: InputEvent, *, is_awake: bool = True) -> ActionPlan | None:
        normalized = self._normalize(input_event.normalized_text or input_event.raw_text)
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

        if not is_awake and not resolution.wake_command:
            return None

        command_text = resolution.stripped_text or normalized
        target = self._extract_target(command_text)
        candidates = ["open_application"] if target else []
        print(f"[HEBE][COG] intent_candidates={candidates!r}", flush=True)
        if not target:
            return None

        addressed_or_trusted = resolution.addressed_to_hebe or input_event.source in {"ui", "typed_ui", "stt_voice", "voice", "button"}
        if not addressed_or_trusted:
            return None

        print(f"[HEBE][ENTITY] application_target={target}", flush=True)
        app = resolve_whitelisted_app(target)
        if app is None:
            return ActionPlan(
                action_type="open_application",
                status="rejected",
                confidence=0.45,
                target=target,
                reason="app_not_whitelisted",
                slots={"application_target": target},
                context_checks={"source": input_event.source, "awake": is_awake, "whitelisted": False},
            )

        executable_path = str(app.get("executable_path") or app.get("command") or "").strip()
        confidence = 1.0 if resolution.addressed_to_hebe else 0.84
        status = "complete"
        reason = "ok"
        missing_slots: list[str] = []
        if not executable_path:
            status = "rejected"
            reason = "app_path_missing"
            missing_slots = ["executable_path"]

        return ActionPlan(
            action_type="open_application",
            status=status,
            confidence=confidence,
            target=app.get("app_id") or target,
            command=executable_path or None,
            reason=reason,
            slots={
                "application_target": target,
                "app_id": app.get("app_id"),
                "display_name": app.get("display_name") or app.get("name"),
                "app_record": app,
                "requires_confirmation": bool(app.get("requires_confirmation")),
            },
            context_checks={
                "source": input_event.source,
                "awake": is_awake,
                "whitelisted": True,
                "path_configured": bool(executable_path),
            },
            missing_slots=missing_slots,
        )

    def command_markers(self) -> Iterable[str]:
        return OPEN_APP_MARKERS

    def _extract_target(self, text: str) -> str | None:
        tokens = [token for token in self._normalize(text).split() if token not in _FILLER_TOKENS]
        if not tokens:
            return None
        for index, token in enumerate(tokens):
            if token in OPEN_APP_MARKERS:
                target = " ".join(tokens[index + 1:]).strip()
                return target or None
        return None

    def _normalize(self, text: str) -> str:
        value = str(text or "").strip().lower()
        value = "".join(
            ch for ch in unicodedata.normalize("NFKD", value)
            if not unicodedata.combining(ch)
        )
        value = re.sub(r"[^a-z0-9_ ]+", " ", value)
        return " ".join(value.split())
