# backend/app/orchestrator/intents/resolver.py

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Optional

from app.orchestrator.models import IntentResult, OrchestratorInput

from .catalog import INTENTS, TOOL_INTENTS


class IntentResolver:
    """
    Resolver nuevo para el orquestador.

    Pipeline:
    1. Normalización
    2. Reglas rápidas y extracción local de slots
    3. Fallback LLM estructurado
    4. Resultado final en IntentResult

    No depende del código legacy.
    """

    def __init__(
        self,
        *,
        llm: Any | None = None,
        low_confidence_threshold: float = 0.55,
    ) -> None:
        self.llm = llm
        self.low_confidence_threshold = low_confidence_threshold

    def resolve(
        self,
        user_input: OrchestratorInput,
        state: Any = None,
    ) -> IntentResult:
        raw = (user_input.text or "").strip()
        if not raw:
            return IntentResult()

        normalized = self._normalize(raw)

        # 1. reglas rápidas
        rule_result = self._resolve_by_rules(raw=raw, normalized=normalized)
        if rule_result is not None:
            return rule_result

        # 2. fallback LLM
        if self.llm is not None:
            llm_result = self._resolve_by_llm(
                raw=raw,
                normalized=normalized,
                state=state,
            )
            if llm_result is not None:
                return llm_result

        # 3. nada claro -> chat
        return IntentResult(
            intent="chat",
            confidence=0.30,
            slots={},
            source="fallback_chat",
            raw={
                "raw_text": raw,
                "normalized_text": normalized,
            },
        )

    # =========================
    # Normalización
    # =========================

    def _normalize(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # =========================
    # Reglas rápidas
    # =========================

    def _resolve_by_rules(self, *, raw: str, normalized: str) -> IntentResult | None:
        # saludo / small talk muy básico -> chat
        if normalized in {"hola", "buenas", "hello", "hi", "que tal", "qué tal"}:
            return IntentResult(
                intent="chat",
                confidence=0.90,
                slots={},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        # open_app
        open_app = self._match_open_app(raw, normalized)
        if open_app is not None:
            return open_app

        # close_window
        close_window = self._match_close_window(raw, normalized)
        if close_window is not None:
            return close_window

        # set_volume
        set_volume = self._match_set_volume(raw, normalized)
        if set_volume is not None:
            return set_volume

        # play_music
        play_music = self._match_play_music(raw, normalized)
        if play_music is not None:
            return play_music

        # pause_music
        pause_music = self._match_pause_music(raw, normalized)
        if pause_music is not None:
            return pause_music

        # shutdown / restart / sleep_mode
        power = self._match_power_commands(raw, normalized)
        if power is not None:
            return power

        return None

    def _match_open_app(self, raw: str, normalized: str) -> IntentResult | None:
        if not normalized.startswith(("abre ", "inicia ", "ejecuta ", "lanza ", "open ", "start ", "run ")):
            return None

        parts = normalized.split(" ", 1)
        if len(parts) < 2:
            return IntentResult(
                intent="open_app",
                confidence=0.88,
                slots={},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        app_name = parts[1].strip()

        if not app_name:
            return IntentResult(
                intent="open_app",
                confidence=0.88,
                slots={},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        print(f"[HEBE][INTENT] open_app matched app_name={app_name!r}", flush=True)

        return IntentResult(
            intent="open_app",
            confidence=0.95,
            slots={"app_name": app_name},
            source="rules",
            raw={"raw_text": raw, "normalized_text": normalized},
        )

    def _match_close_window(self, raw: str, normalized: str) -> IntentResult | None:
        if not any(k in normalized for k in ["cierra", "cerrar", "close"]):
            return None

        if "ventana activa" in normalized or normalized in {"cierra la ventana", "cerrar ventana", "close window"}:
            return IntentResult(
                intent="close_window",
                confidence=0.95,
                slots={"target": "active"},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        pattern = r"^(cierra|cerrar|close)\s+(.+)$"
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if match:
            target = match.group(2).strip()
            generic_targets = {
                "la ventana",
                "ventana",
                "window",
            }
            if target and target not in generic_targets:
                return IntentResult(
                    intent="close_window",
                    confidence=0.90,
                    slots={"target": target},
                    source="rules",
                    raw={"raw_text": raw, "normalized_text": normalized},
                )

        return IntentResult(
            intent="close_window",
            confidence=0.82,
            slots={},
            source="rules",
            raw={"raw_text": raw, "normalized_text": normalized},
        )

    def _match_set_volume(self, raw: str, normalized: str) -> IntentResult | None:
        if "volumen" not in normalized and "volume" not in normalized:
            return None

        value_match = re.search(r"\b(100|[1-9]?\d)\b", normalized)
        if value_match:
            value = int(value_match.group(1))
            value = max(0, min(100, value))
            return IntentResult(
                intent="set_volume",
                confidence=0.94,
                slots={"value": value},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        # casos como "sube el volumen" / "baja el volumen"
        if "sube" in normalized:
            return IntentResult(
                intent="set_volume",
                confidence=0.70,
                slots={"direction": "up"},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        if "baja" in normalized:
            return IntentResult(
                intent="set_volume",
                confidence=0.70,
                slots={"direction": "down"},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        return IntentResult(
            intent="set_volume",
            confidence=0.85,
            slots={},
            source="rules",
            raw={"raw_text": raw, "normalized_text": normalized},
        )

    def _match_play_music(self, raw: str, normalized: str) -> IntentResult | None:
        triggers = [
            "pon musica",
            "pon música",
            "reproduce",
            "play music",
            "pon ",
        ]
        if not any(t in normalized for t in ["pon", "reproduce", "play"]):
            return None

        if "musica" in normalized or "music" in normalized or normalized.startswith("pon "):
            query = self._extract_music_query(normalized)
            slots = {"query": query} if query else {}
            return IntentResult(
                intent="play_music",
                confidence=0.88 if query else 0.78,
                slots=slots,
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        return None

    def _extract_music_query(self, normalized: str) -> str | None:
        patterns = [
            r"^pon\s+musica\s+de\s+(.+)$",
            r"^pon\s+música\s+de\s+(.+)$",
            r"^reproduce\s+(.+)$",
            r"^play\s+music\s+(.+)$",
            r"^pon\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, normalized, flags=re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                if query and query not in {"musica", "música", "music"}:
                    return query
        return None

    def _match_pause_music(self, raw: str, normalized: str) -> IntentResult | None:
        if any(k in normalized for k in ["pausa la musica", "pausa la música", "pausa musica", "pausa música", "pause music"]):
            return IntentResult(
                intent="pause_music",
                confidence=0.95,
                slots={},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )
        return None

    def _match_power_commands(self, raw: str, normalized: str) -> IntentResult | None:
        shutdown_terms = {
            "apaga el ordenador",
            "apaga el pc",
            "apagar el ordenador",
            "apagar el pc",
            "shutdown",
            "turn off pc",
        }
        if normalized in shutdown_terms:
            return IntentResult(
                intent="shutdown_pc",
                confidence=0.98,
                slots={},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        restart_terms = {
            "reinicia el ordenador",
            "reinicia el pc",
            "reiniciar el ordenador",
            "reiniciar el pc",
            "restart",
        }
        if normalized in restart_terms:
            return IntentResult(
                intent="restart_pc",
                confidence=0.98,
                slots={},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        sleep_terms = {
            "duerme",
            "vete a dormir",
            "modo reposo",
            "sleep mode",
        }
        if normalized in sleep_terms:
            return IntentResult(
                intent="sleep_mode",
                confidence=0.95,
                slots={},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        return None

    # =========================
    # Fallback LLM
    # =========================

    def _resolve_by_llm(
        self,
        *,
        raw: str,
        normalized: str,
        state: Any = None,
    ) -> IntentResult | None:
        prompt = self._build_llm_prompt(raw=raw, normalized=normalized, state=state)

        try:
            # interfaz esperada: ask_stateless(prompt, temperature=0.0)
            output = self.llm.ask_stateless(prompt, temperature=0.0)
        except Exception:
            return None

        try:
            data = json.loads(output)
        except Exception:
            return None

        intent = str(data.get("intent", "chat")).strip()
        if intent not in INTENTS:
            intent = "chat"

        confidence = float(data.get("confidence", 0.40))
        confidence = max(0.0, min(confidence, 1.0))

        slots = data.get("slots") or {}
        if not isinstance(slots, dict):
            slots = {}

        slots = self._normalize_slots(intent=intent, slots=slots)

        return IntentResult(
            intent=intent,
            confidence=confidence,
            slots=slots,
            source="llm",
            raw={
                "raw_text": raw,
                "normalized_text": normalized,
                "llm_output": data,
            },
        )

    def _build_llm_prompt(self, *, raw: str, normalized: str, state: Any = None) -> str:
        allowed_intents = list(INTENTS.keys())

        state_mode = getattr(state, "mode", "active") if state is not None else "active"
        last_intent = getattr(state, "last_intent", None) if state is not None else None

        return f"""
Eres un clasificador de intención para un asistente personal modular llamado Hebe.

Devuelve SOLO JSON válido.
No añadas explicación.
No añadas markdown.
No añadas texto antes ni después.

Intents permitidos:
{allowed_intents}

Reglas:
- Si el usuario quiere conversar o no está pidiendo una acción concreta, usa intent="chat".
- Si detectas una acción, devuelve el intent más adecuado.
- Para abrir aplicaciones usa EXACTAMENTE el slot "app_name".
- Para cambiar el volumen usa EXACTAMENTE el slot "value" cuando haya número explícito.
- Para poner música usa el slot "query" si el usuario especifica qué quiere escuchar.
- No inventes valores.
- Usa confidence entre 0.0 y 1.0.

JSON schema:
{{
  "intent": "chat|open_app|close_window|set_volume|play_music|pause_music|shutdown_pc|restart_pc|sleep_mode",
  "confidence": 0.0,
  "slots": {{}}
}}

State:
- mode: {state_mode}
- last_intent: {last_intent or "none"}

User raw text:
{raw}

User normalized text:
{normalized}
""".strip()

    # =========================
    # Slot normalization
    # =========================

    def _normalize_slots(self, *, intent: str, slots: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(slots or {})

        if intent == "open_app":
            if "app_raw" in normalized and "app_name" not in normalized:
                normalized["app_name"] = normalized.pop("app_raw")

            if "name" in normalized and "app_name" not in normalized:
                normalized["app_name"] = normalized.pop("name")

            app_name = normalized.get("app_name")
            if isinstance(app_name, str):
                normalized["app_name"] = self._normalize(app_name)

        if intent == "set_volume":
            value = normalized.get("value")
            if isinstance(value, str) and value.isdigit():
                normalized["value"] = max(0, min(100, int(value)))
            elif isinstance(value, (int, float)):
                normalized["value"] = max(0, min(100, int(value)))

        if intent == "play_music":
            query = normalized.get("query")
            if isinstance(query, str):
                normalized["query"] = query.strip()

        if intent == "close_window":
            target = normalized.get("target")
            if isinstance(target, str):
                normalized["target"] = target.strip()

        return normalized