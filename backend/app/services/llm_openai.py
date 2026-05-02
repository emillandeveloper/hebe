from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional


EmitFn = Callable[[str, dict], None]


class OpenAILLM:
    """
    Cliente conversacional para OpenAI usando Responses API.

    Mantiene la misma interfaz mínima que OllamaLLM:
    - chat(messages, temperature=..., num_predict=...)
    - complete(prompt, temperature=..., num_predict=...)

    Así ResponseSynthesizer no necesita saber si habla con Ollama o con OpenAI.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        emit: Optional[EmitFn] = None,
        log_chat: Optional[Callable[[str, str, str], None]] = None,
    ):
        self.model = model or os.getenv("HEBE_OPENAI_MODEL", "gpt-5-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.emit = emit
        self.log_chat = log_chat

        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.timeout_seconds = float(os.getenv("HEBE_OPENAI_TIMEOUT_SECONDS", "20"))
        self.max_output_tokens = int(os.getenv("HEBE_OPENAI_MAX_OUTPUT_TOKENS", "120"))

        # En modelos GPT-5, es más seguro no mandar temperature salvo que lo actives.
        # Algunos modelos/API rechazan parámetros no soportados.
        self.temperature = float(os.getenv("HEBE_OPENAI_TEMPERATURE", "0.7"))
        self.send_temperature = os.getenv("HEBE_OPENAI_SEND_TEMPERATURE", "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # Controles útiles para modelos GPT-5. Si OpenAI cambia soporte por modelo,
        # puedes desactivarlos dejando las variables vacías.
        self.reasoning_effort = os.getenv("HEBE_OPENAI_REASONING_EFFORT", "minimal").strip()
        self.verbosity = os.getenv("HEBE_OPENAI_VERBOSITY", "low").strip()

        self.log_usage = os.getenv("HEBE_OPENAI_LOG_USAGE", "true").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # Última llamada, para que ResponseSynthesizer pueda volcar
        # usage/latencia en el dataset JSONL.
        self.last_usage: dict[str, Any] | None = None
        self.last_elapsed_ms: int | None = None

    # =========================
    # Emisión / errores
    # =========================

    def _emit_final(self, text: str) -> None:
        if text and self.emit:
            self.emit("llm.final", {"text": text})

    def _emit_error(self, where: str, error: str) -> None:
        if self.emit:
            self.emit("error", {"where": where, "error": error})

    def _emit_usage(self, usage: dict[str, Any]) -> None:
        if not self.log_usage:
            return

        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")

        print(
            "[HEBE][OPENAI][USAGE] "
            f"model={self.model!r} input={input_tokens} output={output_tokens} total={total_tokens}",
            flush=True,
        )

        if self.emit:
            self.emit(
                "llm.usage",
                {
                    "provider": "openai",
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            )

    # =========================
    # API pública compatible
    # =========================

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        num_predict: Optional[int] = None,
        seed: Optional[int] = None,
        **_: Any,
    ) -> str:
        """
        Inferencia stateless sobre un prompt único.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            return ""

        return self._responses_create(
            instructions="Devuelve únicamente la respuesta final.",
            user_input=prompt,
            temperature=temperature,
            max_output_tokens=num_predict,
            seed=seed,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        num_predict: Optional[int] = None,
        seed: Optional[int] = None,
        **_: Any,
    ) -> str:
        """
        Chat stateless con mensajes explícitos system/user.

        ResponseSynthesizer suele pasar:
        [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."}
        ]
        """
        if not messages:
            return ""

        system_parts: list[str] = []
        user_parts: list[str] = []

        for message in messages:
            role = (message.get("role") or "").strip().lower()
            content = (message.get("content") or "").strip()
            if not content:
                continue

            if role in ("system", "developer"):
                system_parts.append(content)
            elif role == "assistant":
                # No usamos conversation state para evitar arrastrar contexto raro.
                # Si se necesita más adelante, se añadirá de forma explícita.
                user_parts.append(f"[respuesta previa de Hebe]\n{content}")
            else:
                user_parts.append(content)

        instructions = "\n\n".join(system_parts).strip() or "Eres Hebe."
        user_input = "\n\n".join(user_parts).strip()

        if not user_input:
            return ""

        return self._responses_create(
            instructions=instructions,
            user_input=user_input,
            temperature=temperature,
            max_output_tokens=num_predict,
            seed=seed,
        )

    # =========================
    # Responses API
    # =========================

    def _responses_create(
        self,
        *,
        instructions: str,
        user_input: str,
        temperature: float,
        max_output_tokens: Optional[int],
        seed: Optional[int],
    ) -> str:
        self.last_usage = None
        self.last_elapsed_ms = None

        if not self.api_key:
            self._emit_error("openai.responses", "OPENAI_API_KEY no está configurada")
            return ""

        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": user_input,
            "max_output_tokens": int(max_output_tokens or self.max_output_tokens),
        }

        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        if self.verbosity:
            payload["text"] = {"verbosity": self.verbosity}

        if self.send_temperature:
            payload["temperature"] = temperature if temperature is not None else self.temperature

        # Seed no está garantizado en Responses para todos los modelos. Lo dejamos
        # fuera para evitar errores de API. El retry seguirá cambiando por muestreo.
        _ = seed

        url = f"{self.base_url}/responses"
        body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        started = time.time()

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")

            data = json.loads(raw)
            text = self._extract_text(data).strip()

            usage = data.get("usage")
            if isinstance(usage, dict):
                self.last_usage = dict(usage)
                self._emit_usage(usage)

            elapsed_ms = int((time.time() - started) * 1000)
            self.last_elapsed_ms = elapsed_ms
            print(
                f"[HEBE][OPENAI] model={self.model!r} elapsed_ms={elapsed_ms} "
                f"chars={len(text)}",
                flush=True,
            )

            if text:
                self._emit_final(text)
                if self.log_chat:
                    self.log_chat("assistant", text, source="openai")

            return text

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            self._emit_error("openai.responses.http", f"{e.code}: {error_body}")
            print(f"[HEBE][OPENAI][ERROR] HTTP {e.code}: {error_body}", flush=True)
            return ""

        except Exception as e:
            self._emit_error("openai.responses", str(e))
            print(f"[HEBE][OPENAI][ERROR] {e}", flush=True)
            return ""

    def _extract_text(self, data: dict[str, Any]) -> str:
        """
        Extrae texto de Responses API de forma tolerante a cambios menores
        del formato de respuesta.
        """
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        chunks: list[str] = []

        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue

                content = item.get("content")
                if not isinstance(content, list):
                    continue

                for part in content:
                    if not isinstance(part, dict):
                        continue

                    text = part.get("text")
                    if isinstance(text, str) and text:
                        chunks.append(text)

        if chunks:
            return "".join(chunks).strip()

        # Fallback defensivo por si la API devuelve otro wrapper.
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()

        return ""
