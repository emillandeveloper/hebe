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
    Cliente conversacional para OpenAI usando Chat Completions API.

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

        self.temperature = float(os.getenv("HEBE_OPENAI_TEMPERATURE", "0.7"))
        self.send_temperature = os.getenv("HEBE_OPENAI_SEND_TEMPERATURE", "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

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

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        cached_tokens = usage.get("cached_tokens", 0)
        cost_usd = usage.get("cost_usd", 0.0)

        print(
            "[HEBE][OPENAI][USAGE] "
            f"model={self.model!r} "
            f"input={input_tokens} "
            f"output={output_tokens} "
            f"total={total_tokens} "
            f"cached={cached_tokens} "
            f"cost_usd={cost_usd:.4f}",
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
                    "cached_tokens": cached_tokens,
                    "cost_usd": cost_usd,
                },
            )

    def _normalize_usage(self, raw_usage: dict[str, Any]) -> dict[str, Any]:
        """Normaliza la respuesta de usage de Chat Completions al formato interno."""
        input_tokens = int(raw_usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(raw_usage.get("completion_tokens", 0) or 0)
        total_tokens = int(raw_usage.get("total_tokens", 0) or 0)

        details = raw_usage.get("prompt_tokens_details") or {}
        cached_tokens = int(details.get("cached_tokens", 0) or 0)

        non_cached = max(0, input_tokens - cached_tokens)
        cost_usd = round(
            (non_cached * 0.25 + cached_tokens * 0.025) / 1_000_000
            + output_tokens * 2.00 / 1_000_000,
            6,
        )

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": cost_usd,
        }

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
        """Inferencia stateless sobre un prompt único."""
        prompt = (prompt or "").strip()
        if not prompt:
            return ""

        return self._chat_completions_create(
            messages=[
                {"role": "system", "content": "Devuelve únicamente la respuesta final."},
                {"role": "user", "content": prompt},
            ],
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
        Chat stateless con mensajes explícitos system/user/assistant.

        ResponseSynthesizer suele pasar:
        [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."}
        ]
        """
        if not messages:
            return ""

        normalized: list[dict[str, str]] = []
        for m in messages:
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "developer":
                role = "system"
            normalized.append({"role": role, "content": content})

        if not normalized:
            return ""

        if not any(m["role"] == "user" for m in normalized):
            return ""

        return self._chat_completions_create(
            messages=normalized,
            temperature=temperature,
            max_output_tokens=num_predict,
            seed=seed,
        )

    # =========================
    # Chat Completions API
    # =========================

    def _chat_completions_create(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_output_tokens: Optional[int],
        seed: Optional[int],
    ) -> str:
        self.last_usage = None
        self.last_elapsed_ms = None

        if not self.api_key:
            self._emit_error("openai.chat", "OPENAI_API_KEY no está configurada")
            return ""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": int(max_output_tokens or self.max_output_tokens),
        }

        if self.send_temperature:
            payload["temperature"] = temperature if temperature is not None else self.temperature

        if seed is not None:
            payload["seed"] = seed

        url = f"{self.base_url}/chat/completions"
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

            raw_usage = data.get("usage")
            if isinstance(raw_usage, dict):
                self.last_usage = self._normalize_usage(raw_usage)
                self._emit_usage(self.last_usage)

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
            self._emit_error("openai.chat.http", f"{e.code}: {error_body}")
            print(f"[HEBE][OPENAI][ERROR] HTTP {e.code}: {error_body}", flush=True)
            return ""

        except Exception as e:
            self._emit_error("openai.chat", str(e))
            print(f"[HEBE][OPENAI][ERROR] {e}", flush=True)
            return ""

    def _extract_text(self, data: dict[str, Any]) -> str:
        """Extrae texto de Chat Completions API."""
        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content.strip()
        except (KeyError, IndexError, TypeError):
            pass

        # Fallback defensivo
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()

        return ""
