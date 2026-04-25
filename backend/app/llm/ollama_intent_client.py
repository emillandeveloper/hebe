from __future__ import annotations

import json
import re
from typing import Any

import requests


class OllamaIntentClient:
    """
    Cliente Ollama para clasificación de intent/slots con salida estructurada.

    Usa /api/chat con:
    - model fijo (ej. "hebe-intent")
    - temperature 0
    - format schema JSON cuando se pide structured output
    """

    def __init__(
        self,
        *,
        model: str = "hebe-intent",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def ask_structured(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Modo simple: un prompt combinado, salida JSON estructurada.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
            "format": schema,
            "options": {
                "temperature": temperature,
                "num_predict": 512,
                "num_ctx": 4096,
            },
        }

        data = self._post_chat(payload)
        content = self._extract_message_content(data)
        parsed = self._loads_json_object(content)
        if not isinstance(parsed, dict):
            raise ValueError("Ollama structured response is not a JSON object")
        return parsed

    def chat_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Modo recomendado para el IntentResolver:
        system + user + schema.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "format": schema,
            "options": {
                "temperature": temperature,
                "num_predict": 512,
                "num_ctx": 4096,
            },
        }

        data = self._post_chat(payload)
        content = self._extract_message_content(data)
        parsed = self._loads_json_object(content)
        if not isinstance(parsed, dict):
            raise ValueError("Ollama structured response is not a JSON object")
        return parsed

    def ask_stateless(
        self,
        prompt: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Fallback no estructurado.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        text = data.get("response", "")
        if not isinstance(text, str):
            raise ValueError("Invalid Ollama generate response")
        return text

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid Ollama chat response")
        return data

    def _extract_message_content(self, data: dict[str, Any]) -> str:
        message = data.get("message")
        if not isinstance(message, dict):
            raise ValueError("Missing message in Ollama response")

        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("Invalid message content in Ollama response")
        return content

    def _loads_json_object(self, text: str) -> dict[str, Any]:
        """
        Parser JSON tolerante:
        - quita espacios/saltos de línea
        - intenta parse directo
        - si falla, busca el primer objeto { ... } dentro del texto
        - limpia trailing commas típicos de modelos pequeños
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("Empty Ollama response")

        # 1) Intento directo
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # 2) Extraer el primer bloque { ... } balanceado
        extracted = self._extract_first_json_object(text)
        if extracted:
            try:
                data = json.loads(extracted)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                # 3) Limpieza de trailing commas
                cleaned = re.sub(r",(\s*[}\]])", r"\1", extracted)
                try:
                    data = json.loads(cleaned)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Invalid JSON from Ollama: {text}")

    def _extract_first_json_object(self, text: str) -> str | None:
        """
        Encuentra el primer objeto JSON balanceado dentro del texto.
        Respeta strings con llaves escapadas.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape:
                escape = False
                continue

            if ch == "\\":
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None