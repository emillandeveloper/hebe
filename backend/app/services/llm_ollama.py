from __future__ import annotations

import os
from typing import Callable, Optional, Any

import ollama


EmitFn = Callable[[str, dict], None]


class OllamaLLM:
    """
    Cliente de modelo para Ollama.

    Filosofía actual:
    - Hebe controla el contexto y la memoria
    - este wrapper NO debe convertirse en una mente paralela
    - `complete()` es la API recomendada para el flujo cognitivo
    - `ask_stateless()` queda útil para NLU / clasificación
    - `ask()` se mantiene temporalmente para compatibilidad con legacy

    CAMBIO 28/04: añadido soporte de `seed` opcional en chat()/complete()
    para permitir retry con generación distinta tras detectar patrones
    helper en respuestas de Twitch.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        emit: Optional[EmitFn] = None,
        log_chat: Optional[Callable[[str, str, str], None]] = None,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "hebe")
        self.emit = emit
        self.log_chat = log_chat

        # Legacy only: historial interno temporal mientras exista legacy_flow
        self.history: list[dict[str, str]] = []

        # opciones por defecto
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
        self.repeat_penalty = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.2"))
        self.top_p = float(os.getenv("OLLAMA_TOP_P", "0.9"))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "1200"))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))

    # =========================
    # Helpers
    # =========================

    def reset(self) -> None:
        """
        Limpia el historial legacy.
        No afecta al flujo cognitivo stateless.
        """
        self.history.clear()

    def _emit_final(self, text: str) -> None:
        if text and self.emit:
            self.emit("llm.final", {"text": text})

    def _emit_error(self, where: str, error: str) -> None:
        if self.emit:
            self.emit("error", {"where": where, "error": error})

    def _extract_text(self, resp: Any) -> str:
        if resp is None:
            return ""

        if isinstance(resp, str):
            return resp.strip()

        # Caso dict
        if isinstance(resp, dict):
            msg = (resp.get("message", {}) or {})
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()

            for key in ("content", "text", "response", "output"):
                value = resp.get(key)
                if isinstance(value, str):
                    return value.strip()

        # Caso objeto con .message.content  <-- ESTE ES EL IMPORTANTE
        message = getattr(resp, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()

        # Caso objeto con .content
        content = getattr(resp, "content", None)
        if isinstance(content, str):
            return content.strip()

        # Caso objeto con .text
        text = getattr(resp, "text", None)
        if isinstance(text, str):
            return text.strip()

        return str(resp).strip()

    def _ollama_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        top_p: Optional[float] = None,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        # Construimos options dinámicamente: seed solo se incluye si se ha
        # pasado explícitamente, para no cambiar el comportamiento de las
        # llamadas existentes que NO usan seed (que deben seguir teniendo
        # generación libre y no determinista de Ollama por defecto).
        options: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "repeat_penalty": self.repeat_penalty if repeat_penalty is None else repeat_penalty,
            "top_p": self.top_p if top_p is None else top_p,
            "num_predict": self.num_predict if num_predict is None else num_predict,
            "num_ctx": self.num_ctx if num_ctx is None else num_ctx,
        }
        if seed is not None:
            options["seed"] = seed

        resp = ollama.chat(
            model=self.model,
            messages=messages,
            options=options,
        )

        return self._extract_text(resp)

    # =========================
    # API recomendada para Hebe cognitiva
    # =========================

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        num_predict: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """
        Inferencia stateless sobre un prompt único.
        Esta es la API recomendada para ResponseSynthesizer.

        seed: si se pasa, se fija el seed de Ollama para que la
              generación sea reproducible. Úsalo para retry con
              variación controlada.
        """
        try:
            prompt = (prompt or "").strip()
            if not prompt:
                return ""

            text = self._ollama_chat(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                num_predict=num_predict,
                seed=seed,
            )

            if text:
                self._emit_final(text)

            return text or "…"

        except Exception as e:
            self._emit_error("ollama.complete", str(e))
            return "Lo siento, no puedo generar una respuesta en este momento."

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        num_predict: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """
        Chat stateless con mensajes explícitos.
        Útil si más adelante quieres prompts tipo system/user/assistant
        controlados por Hebe, no por el wrapper.

        seed: si se pasa, se fija el seed de Ollama. Útil para retry
              con variación tras detectar patrones helper.
        """
        try:
            if not messages:
                return ""

            text = self._ollama_chat(
                messages,
                temperature=temperature,
                num_predict=num_predict,
                seed=seed,
            )

            if text:
                self._emit_final(text)

            return text or "…"

        except Exception as e:
            self._emit_error("ollama.chat", str(e))
            return "Lo siento, no puedo generar una respuesta en este momento."

    # =========================
    # API útil para NLU / clasificación
    # =========================

    def ask_stateless(self, user_text: str, temperature: float = 0.0) -> str:
        """
        Llamada sin historial. No modifica self.history.
        Ideal para clasificación NLU / extracción controlada.
        """
        try:
            user_text = (user_text or "").strip()
            if not user_text:
                return ""

            text = self._ollama_chat(
                [{"role": "user", "content": user_text}],
                temperature=temperature,
                repeat_penalty=1.0,
                top_p=1.0,
                num_predict=400,
            )

            return text or ""

        except Exception as e:
            self._emit_error("ollama.ask_stateless", str(e))
            return ""

    # =========================
    # Compat temporal con legacy
    # =========================

    def ask(self, user_text: str) -> str:
        """
        Compatibilidad temporal con el flujo legacy.
        Mantiene historial interno.
        Cuando desconectes legacy, este método debería desaparecer.
        """
        try:
            user_text = (user_text or "").strip()
            if not user_text:
                return ""

            self.history.append({"role": "user", "content": user_text})

            text = self._ollama_chat(self.history)

            if text:
                self.history.append({"role": "assistant", "content": text})

                if self.log_chat:
                    self.log_chat("assistant", text, source="llm")

                self._emit_final(text)

            return text or "…"

        except Exception as e:
            self._emit_error("ollama.ask", str(e))
            return "Lo siento, no puedo generar una respuesta en este momento."