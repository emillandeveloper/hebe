from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from app.orchestrator.models import IntentResult, OrchestratorInput

from .catalog import INTENTS


class IntentResolver:
    """
    Resolver nuevo para el orquestador.

    Pipeline:
    1. Normalización
    2. Reglas rápidas y extracción local de slots
    3. Fallback LLM estructurado
    4. Resultado final en IntentResult
    """

    INTENT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(INTENTS.keys()),
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "slots": {
                "type": "object",
                "additionalProperties": True,
            },
        },
        "required": ["intent", "confidence", "slots"],
    }

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

        rule_result = self._resolve_by_rules(raw=raw, normalized=normalized)
        if rule_result is not None:
            return rule_result

        if self.llm is not None:
            llm_result = self._resolve_by_llm(
                raw=raw,
                normalized=normalized,
                state=state,
            )
            if llm_result is not None:
                return llm_result

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

    def _clean_app_name(self, text: str) -> str:
        value = self._normalize(text)

        prefixes = [
            "el ",
            "la ",
            "los ",
            "las ",
            "por favor ",
            "porfa ",
            "please ",
        ]

        suffixes = [
            " por favor",
            " porfa",
            " please",
        ]

        changed = True
        while changed:
            changed = False

            for prefix in prefixes:
                if value.startswith(prefix):
                    value = value[len(prefix):].strip()
                    changed = True

            for suffix in suffixes:
                if value.endswith(suffix):
                    value = value[: -len(suffix)].strip()
                    changed = True

        return value

    def _looks_like_non_imperative(self, normalized: str) -> bool:
        markers = [
            "quiero ",
            "me apetece ",
            "deberia ",
            "debería ",
            "crees que ",
            "podria ",
            "podría ",
            "me gustaria ",
            "me gustaría ",
            "tengo ganas de ",
            "me vendria bien ",
            "me vendría bien ",
        ]
        return any(marker in normalized for marker in markers)

    # =========================
    # Reglas rápidas
    # =========================

    def _resolve_by_rules(self, *, raw: str, normalized: str) -> IntentResult | None:
        stream_enable = self._match_stream_enable(raw, normalized)
        if stream_enable is not None:
            return stream_enable

        stream_disable = self._match_stream_disable(raw, normalized)
        if stream_disable is not None:
            return stream_disable

        stream_shoutout = self._match_stream_shoutout(raw, normalized)
        if stream_shoutout is not None:
            return stream_shoutout

        stream_chat = self._match_stream_chat_message(raw, normalized)
        if stream_chat is not None:
            return stream_chat
        
        if normalized in {
            "hola",
            "buenas",
            "hello",
            "hi",
            "que tal",
            "qué tal",
        }:
            return IntentResult(
                intent="chat",
                confidence=0.90,
                slots={},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        if normalized in {"abrelo otra vez", "ábrelo otra vez"}:
            return IntentResult(
                intent="chat",
                confidence=0.45,
                slots={},
                source="rules_ambiguous",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        if self._looks_like_non_imperative(normalized):
            return IntentResult(
                intent="chat",
                confidence=0.70,
                slots={},
                source="rules_non_imperative",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        open_app = self._match_open_app(raw, normalized)
        if open_app is not None:
            return open_app

        close_window = self._match_close_window(raw, normalized)
        if close_window is not None:
            return close_window

        set_volume = self._match_set_volume(raw, normalized)
        if set_volume is not None:
            return set_volume

        play_music = self._match_play_music(raw, normalized)
        if play_music is not None:
            return play_music

        pause_music = self._match_pause_music(raw, normalized)
        if pause_music is not None:
            return pause_music

        power = self._match_power_commands(raw, normalized)
        if power is not None:
            return power

        return None

    def _match_open_app(self, raw: str, normalized: str) -> IntentResult | None:
        if not normalized.startswith(
            ("abre ", "inicia ", "ejecuta ", "lanza ", "open ", "start ", "run ")
        ):
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

        app_name = self._clean_app_name(app_name)

        if " y luego " in app_name:
            app_name = app_name.split(" y luego ")[0].strip()
        if " y " in app_name:
            app_name = app_name.split(" y ")[0].strip()

        vague_targets = {
            "eso",
            "esto",
            "lo",
            "la",
            "algo",
            "otra vez",
        }
        if app_name in vague_targets:
            return IntentResult(
                intent="chat",
                confidence=0.40,
                slots={},
                source="rules_ambiguous",
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
        lowered_raw = raw.lower()

        if normalized in {"cierralo", "ciérralo", "cierrame eso", "ciérrame eso"}:
            return IntentResult(
                intent="chat",
                confidence=0.40,
                slots={},
                source="rules_ambiguous",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        if not any(k in normalized for k in ["cierra", "cerrar", "close"]):
            return None

        if "la ventana" in normalized and "close" in normalized:
            return IntentResult(
                intent="close_window",
                confidence=0.90,
                slots={"target": "active"},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        if "ventana activa" in normalized or normalized in {
            "cierra la ventana",
            "cerrar ventana",
            "close window",
            "cierra ventana",
        }:
            return IntentResult(
                intent="close_window",
                confidence=0.95,
                slots={"target": "active"},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        if "lo que este abierto" in normalized or "lo que esté abierto" in lowered_raw:
            return IntentResult(
                intent="close_window",
                confidence=0.92,
                slots={"target": "active"},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        pattern = r"^(cierra|cerrar|close)\s+(.+)$"
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if match:
            target = match.group(2).strip()
            generic_targets = {"la ventana", "ventana", "window"}
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
        lowered_raw = raw.lower()

        volume_words = {"volumen", "volume", "bolumen"}
        directional_words = {
            "sube",
            "baja",
            "mute",
            "silencia",
            "silence",
            "subelo",
            "súbelo",
            "bajalo",
            "bájalo",
        }

        if not any(w in normalized for w in volume_words | directional_words):
            return None

        if ("maximo" in normalized or "máximo" in lowered_raw) and any(
            w in normalized for w in volume_words
        ):
            return IntentResult(
                intent="set_volume",
                confidence=0.92,
                slots={"value": 100},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        if ("minimo" in normalized or "mínimo" in lowered_raw) and any(
            w in normalized for w in volume_words
        ):
            return IntentResult(
                intent="set_volume",
                confidence=0.92,
                slots={"value": 0},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        if "bajalo" in normalized or "bájalo" in lowered_raw:
            return IntentResult(
                intent="set_volume",
                confidence=0.75,
                slots={"direction": "down"},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        if "subelo" in normalized or "súbelo" in lowered_raw:
            return IntentResult(
                intent="set_volume",
                confidence=0.75,
                slots={"direction": "up"},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        value_match = re.search(r"\b(100|[1-9]?\d)\b", normalized)
        if value_match and any(w in normalized for w in volume_words):
            value = int(value_match.group(1))
            value = max(0, min(100, value))
            return IntentResult(
                intent="set_volume",
                confidence=0.94,
                slots={"value": value},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        if "sube" in normalized and any(w in normalized for w in volume_words):
            return IntentResult(
                intent="set_volume",
                confidence=0.74,
                slots={"direction": "up"},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        if "baja" in normalized and any(w in normalized for w in volume_words):
            return IntentResult(
                intent="set_volume",
                confidence=0.74,
                slots={"direction": "down"},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        if "mute" in normalized or "silencia" in normalized or "silence" in normalized:
            return IntentResult(
                intent="set_volume",
                confidence=0.82,
                slots={"value": 0},
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
        if not any(t in normalized for t in ["pon", "ponme", "reproduce", "play"]):
            return None

        if "musica" in normalized or "music" in normalized or normalized.startswith(
            ("pon ", "ponme ")
        ):
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
            r"^pon\s+musica\s+(.+)$",
            r"^pon\s+música\s+(.+)$",
            r"^ponme\s+(.+)$",
            r"^reproduce\s+(.+)$",
            r"^play\s+music\s+de\s+(.+)$",
            r"^play\s+music\s+(.+)$",
            r"^play\s+(.+)$",
            r"^pon\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, normalized, flags=re.IGNORECASE)
            if match:
                query = match.group(1).strip()

                query = query.replace("algo de ", "").strip()
                query = query.replace("musica de ", "").strip()
                query = query.replace("música de ", "").strip()
                query = query.replace("musica ", "").strip()
                query = query.replace("música ", "").strip()

                if query and query not in {"musica", "música", "music"}:
                    return query
        return None

    def _match_pause_music(self, raw: str, normalized: str) -> IntentResult | None:
        if any(
            k in normalized
            for k in [
                "pausa la musica",
                "pausa la música",
                "pausa musica",
                "pausa música",
                "pause music",
                "para la musica",
                "para la música",
                "para musica",
                "para música",
                "quita la musica",
                "quita la música",
                "quita musica",
                "quita música",
            ]
        ):
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
        system_prompt = self._build_llm_system_prompt()
        user_prompt = self._build_llm_user_prompt(
            raw=raw,
            normalized=normalized,
            state=state,
        )

        data = self._call_llm_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        if not isinstance(data, dict):
            return None

        intent = str(data.get("intent", "chat")).strip()
        if intent not in INTENTS:
            intent = "chat"

        confidence = data.get("confidence", 0.40)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.40
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

    def _call_llm_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any] | None:
        if self.llm is None:
            return None

        model_name = getattr(self.llm, "model", "unknown")

        if hasattr(self.llm, "ask_structured"):
            print(f"[HEBE][OLLAMA_INTENT] model={model_name}", flush=True)
            try:
                data = self.llm.ask_structured(
                    prompt=f"{system_prompt}\n\n{user_prompt}",
                    schema=self.INTENT_SCHEMA,
                    temperature=0.0,
                )
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                print(f"[HEBE][OLLAMA_INTENT] ask_structured failed: {exc}", flush=True)

        if hasattr(self.llm, "chat_structured"):
            print(f"[HEBE][OLLAMA_INTENT] model={model_name}", flush=True)
            try:
                data = self.llm.chat_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=self.INTENT_SCHEMA,
                    temperature=0.0,
                )
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                print(f"[HEBE][OLLAMA_INTENT] chat_structured failed: {exc}", flush=True)

        if hasattr(self.llm, "ask_stateless"):
            print(f"[HEBE][OLLAMA_INTENT] model={model_name}", flush=True)
            strict_prompt = (
                f"{system_prompt}\n\n"
                f"{user_prompt}\n\n"
                "Return ONLY one valid JSON object. No markdown. No prose."
            )

            try:
                output = self.llm.ask_stateless(strict_prompt, temperature=0.0)
            except Exception as exc:
                print(f"[HEBE][OLLAMA_INTENT] ask_stateless failed: {exc}", flush=True)
                return None

            return self._parse_json_maybe_dirty(output)

        return None

    def _build_llm_system_prompt(self) -> str:
        allowed_intents = ", ".join(INTENTS.keys())
        return (
            "You are a strict intent classifier and slot extractor for a local desktop assistant named Hebe.\n"
            "You must output ONLY valid JSON.\n"
            "Do not explain.\n"
            "Do not add markdown.\n"
            "Do not add prose.\n"
            "Do not continue the user's text.\n"
            f"Allowed intents: {allowed_intents}.\n"
            "Use the most specific allowed intent.\n"
            "If the user is chatting or no concrete action is requested, use intent='chat'.\n"
            "If the user is expressing desire, preference, doubt, or asking for advice rather than issuing a direct command, use intent='chat'.\n"
            "Examples of chat:\n"
            '- "quiero escuchar música"\n'
            '- "me apetece algo de música"\n'
            '- "debería subir el volumen"\n'
            '- "crees que debería cerrar chrome"\n'
            "For open_app use slot 'app_name'.\n"
            "For set_volume use slot 'value' when the number is explicit.\n"
            "For set_volume use slot 'direction' with values 'up' or 'down' when the user asks to raise or lower volume without a number.\n"
            "For play_music use slot 'query' only if the user specifies what to play.\n"
            "For close_window use slot 'target' only if the user specifies what to close.\n"
            "Never invent slot values.\n"
            "For stream_chat_message use slot 'message'.\n"
            "For stream_shoutout use slot 'target_raw'.\n"
            "For stream_enable and stream_disable do not use slots.\n"
            "Confidence must be a number between 0.0 and 1.0." 
        )

    def _build_llm_user_prompt(
        self,
        *,
        raw: str,
        normalized: str,
        state: Any = None,
    ) -> str:
        state_mode = getattr(state, "mode", "active") if state is not None else "active"
        last_intent = getattr(state, "last_intent", None) if state is not None else None

        return f"""
Classify the following request and extract slots.

Return JSON with this exact shape:
{{
  "intent": "chat|open_app|close_window|set_volume|play_music|pause_music|shutdown_pc|restart_pc|sleep_mode|stream_enable|stream_disable|stream_chat_message|stream_shoutout",
  "confidence": 0.0,
  "slots": {{}}
}}

Examples:
User request: "abre obs"
JSON: {{"intent":"open_app","confidence":0.98,"slots":{{"app_name":"obs"}}}}

User request: "sube el volumen"
JSON: {{"intent":"set_volume","confidence":0.82,"slots":{{"direction":"up"}}}}

User request: "pon el volumen al 30"
JSON: {{"intent":"set_volume","confidence":0.97,"slots":{{"value":30}}}}

User request: "cierra la ventana"
JSON: {{"intent":"close_window","confidence":0.95,"slots":{{"target":"active"}}}}

User request: "quiero escuchar música"
JSON: {{"intent":"chat","confidence":0.90,"slots":{{}}}}

User request: "debería subir el volumen"
JSON: {{"intent":"chat","confidence":0.90,"slots":{{}}}}

User request: "hola"
JSON: {{"intent":"chat","confidence":0.95,"slots":{{}}}}

User request: "activa modo stream"
JSON: {{"intent":"stream_enable","confidence":0.98,"slots":{{}}}}

User request: "desactiva modo stream"
JSON: {{"intent":"stream_disable","confidence":0.98,"slots":{{}}}}

User request: "escribe en el chat hola gente"
JSON: {{"intent":"stream_chat_message","confidence":0.97,"slots":{{"message":"hola gente"}}}}

User request: "haz shoutout a tito charly"
JSON: {{"intent":"stream_shoutout","confidence":0.96,"slots":{{"target_raw":"tito charly"}}}}

State:
- mode: {state_mode}
- last_intent: {last_intent or "none"}

User raw text:
{raw}

User normalized text:
{normalized}
""".strip()

    def _parse_json_maybe_dirty(self, output: Any) -> dict[str, Any] | None:
        if isinstance(output, dict):
            return output

        if not isinstance(output, str):
            return None

        text = output.strip()
        if not text:
            return None

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL,
        )
        if fenced_match:
            candidate = fenced_match.group(1).strip()
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        candidate = self._extract_first_json_object(text)
        if candidate:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        return None

    def _extract_first_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for idx in range(start, len(text)):
            ch = text[idx]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

        return None

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

            if "app" in normalized and "app_name" not in normalized:
                normalized["app_name"] = normalized.pop("app")

            target = normalized.get("target")
            if "app_name" not in normalized and isinstance(target, str):
                normalized["app_name"] = target
                normalized.pop("target", None)

            app_name = normalized.get("app_name")
            if isinstance(app_name, str):
                normalized["app_name"] = self._clean_app_name(app_name)

        if intent == "set_volume":
            value = normalized.get("value")
            if isinstance(value, str):
                value = value.strip()
                if value.isdigit():
                    normalized["value"] = max(0, min(100, int(value)))
                else:
                    normalized.pop("value", None)
            elif isinstance(value, (int, float)):
                normalized["value"] = max(0, min(100, int(value)))

            direction = normalized.get("direction")
            if isinstance(direction, str):
                direction = direction.strip().lower()
                if direction in {"up", "down"}:
                    normalized["direction"] = direction
                else:
                    normalized.pop("direction", None)

        if intent == "play_music":
            query = normalized.get("query")
            if isinstance(query, str):
                query = query.strip()
                if query:
                    normalized["query"] = query
                else:
                    normalized.pop("query", None)

        if intent == "close_window":
            target = normalized.get("target")
            if isinstance(target, str):
                target = target.strip()
                if target:
                    normalized["target"] = target
                else:
                    normalized.pop("target", None)

        if intent == "stream_chat_message":
            message = normalized.get("message")
            if isinstance(message, str):
                message = message.strip()
                if message:
                    normalized["message"] = message
                else:
                    normalized.pop("message", None)

        if intent == "stream_shoutout":
            target_raw = normalized.get("target_raw")
            if isinstance(target_raw, str):
                target_raw = target_raw.strip()
                if target_raw:
                    normalized["target_raw"] = target_raw
                else:
                    normalized.pop("target_raw", None)

        return normalized
    
    def _match_stream_enable(self, raw: str, normalized: str) -> IntentResult | None:
        patterns = [
            r"^(activa|pon|entra en|enable)\s+(el\s+)?modo\s+stream$",
            r"^(activa|pon|enable)\s+stream$",
            r"^stream\s+on$",
        ]
        for pattern in patterns:
            if re.match(pattern, normalized, flags=re.IGNORECASE):
                return IntentResult(
                    intent="stream_enable",
                    confidence=0.96,
                    slots={},
                    source="rules",
                    raw={"raw_text": raw, "normalized_text": normalized},
                )
        return None

    def _match_stream_disable(self, raw: str, normalized: str) -> IntentResult | None:
        patterns = [
            r"^(desactiva|quita|sal del|disable)\s+(el\s+)?modo\s+stream$",
            r"^(desactiva|quita|disable)\s+stream$",
            r"^stream\s+off$",
        ]
        for pattern in patterns:
            if re.match(pattern, normalized, flags=re.IGNORECASE):
                return IntentResult(
                    intent="stream_disable",
                    confidence=0.96,
                    slots={},
                    source="rules",
                    raw={"raw_text": raw, "normalized_text": normalized},
                )
        return None

    def _match_stream_shoutout(self, raw: str, normalized: str) -> IntentResult | None:
        patterns = [
            r"^(haz|dale|manda)?\s*(un\s+)?shoutout\s+a\s+(.+)$",
            r"^(haz|dale|manda)?\s*(un\s+)?so\s+a\s+(.+)$",
            r"^shoutout\s+a\s+(.+)$",
            r"^so\s+a\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue

            target_raw = match.group(match.lastindex).strip()
            if not target_raw:
                return IntentResult(
                    intent="stream_shoutout",
                    confidence=0.70,
                    slots={},
                    source="rules",
                    raw={"raw_text": raw, "normalized_text": normalized},
                )

            return IntentResult(
                intent="stream_shoutout",
                confidence=0.95,
                slots={"target_raw": target_raw},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        return None

    def _match_stream_chat_message(self, raw: str, normalized: str) -> IntentResult | None:
        patterns = [
            r"^(escribe|di|manda|pon)\s+en\s+el\s+chat\s+(.+)$",
            r"^(escribe|di|manda|pon)\s+al\s+chat\s+(.+)$",
            r"^(say)\s+in\s+chat\s+(.+)$",
            r"^(write)\s+in\s+chat\s+(.+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue

            message = match.group(match.lastindex).strip()
            if not message:
                return IntentResult(
                    intent="stream_chat_message",
                    confidence=0.72,
                    slots={},
                    source="rules",
                    raw={"raw_text": raw, "normalized_text": normalized},
                )

            return IntentResult(
                intent="stream_chat_message",
                confidence=0.95,
                slots={"message": message},
                source="rules",
                raw={"raw_text": raw, "normalized_text": normalized},
            )

        return None