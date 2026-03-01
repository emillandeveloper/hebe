from __future__ import annotations

import os
from typing import Callable, Optional

import ollama


EmitFn = Callable[[str, dict], None]
LogChatFn = Callable[[str, str], None]  # (role, text)
HistoryItem = dict  # {"role": "...", "content": "..."}


class OllamaLLM:
    """
    Conversational LLM wrapper (Ollama chat) with internal history.
    Engine calls llm.ask(text) and that's it.
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
        self.history: list[HistoryItem] = []

        # options
        self.temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
        self.repeat_penalty = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.2"))
        self.top_p = float(os.getenv("OLLAMA_TOP_P", "0.9"))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "1200"))
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))

    def reset(self) -> None:
        self.history.clear()

    def ask(self, user_text: str) -> str:
        try:
            user_text = (user_text or "").strip()
            if not user_text:
                return ""

            # add user message
            self.history.append({"role": "user", "content": user_text})

            resp = ollama.chat(
                model=self.model,
                messages=self.history,
                options={
                    "temperature": self.temperature,
                    "repeat_penalty": self.repeat_penalty,
                    "top_p": self.top_p,
                    "num_predict": self.num_predict,
                    "num_ctx": self.num_ctx,
                },
            )

            text = (resp.get("message", {}) or {}).get("content", "").strip()

            if text:
                self.history.append({"role": "assistant", "content": text})
                if self.log_chat:
                    self.log_chat("assistant", text, source="llm")
                if self.emit:
                    self.emit("llm.final", {"text": text})

            return text or "…"

        except Exception as e:
            if self.emit:
                self.emit("error", {"where": "ollama", "error": str(e)})
            return "Lo siento, no puedo generar una respuesta en este momento."