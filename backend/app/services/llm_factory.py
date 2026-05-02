from __future__ import annotations

import os
from typing import Any, Callable, Optional

from app.services.llm_ollama import OllamaLLM
from app.services.llm_openai import OpenAILLM


EmitFn = Callable[[str, dict], None]


class FallbackConversationLLM:
    """
    Wrapper simple: intenta primero el provider principal y, si falla o
    devuelve vacío, usa el provider local de fallback.

    Mantiene la interfaz chat()/complete() que espera ResponseSynthesizer.
    Además expone last_used para que el dataset sepa qué provider respondió.
    """

    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback
        self.last_used: Any | None = None

    @property
    def model(self) -> Any:
        return getattr(self.last_used or self.primary, "model", None)

    @property
    def last_usage(self) -> Any:
        return getattr(self.last_used or self.primary, "last_usage", None)

    @property
    def last_elapsed_ms(self) -> Any:
        return getattr(self.last_used or self.primary, "last_elapsed_ms", None)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.last_used = None
        text = ""
        try:
            text = self.primary.chat(messages, **kwargs)
        except Exception as e:
            print(f"[HEBE][LLM][FALLBACK] primary chat failed: {e}", flush=True)

        text = (text or "").strip()
        if text and text != "…":
            self.last_used = self.primary
            return text

        print("[HEBE][LLM][FALLBACK] using local chat fallback", flush=True)
        self.last_used = self.fallback
        try:
            return (self.fallback.chat(messages, **kwargs) or "").strip()
        except TypeError:
            safe_kwargs = {k: v for k, v in kwargs.items() if k in {"temperature", "num_predict"}}
            return (self.fallback.chat(messages, **safe_kwargs) or "").strip()

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self.last_used = None
        text = ""
        try:
            text = self.primary.complete(prompt, **kwargs)
        except Exception as e:
            print(f"[HEBE][LLM][FALLBACK] primary complete failed: {e}", flush=True)

        text = (text or "").strip()
        if text and text != "…":
            self.last_used = self.primary
            return text

        print("[HEBE][LLM][FALLBACK] using local complete fallback", flush=True)
        self.last_used = self.fallback
        try:
            return (self.fallback.complete(prompt, **kwargs) or "").strip()
        except TypeError:
            safe_kwargs = {k: v for k, v in kwargs.items() if k in {"temperature", "num_predict"}}
            return (self.fallback.complete(prompt, **safe_kwargs) or "").strip()


def _build_local_llm(
    *,
    emit: Optional[EmitFn] = None,
    log_chat: Optional[Callable[[str, str, str], None]] = None,
) -> OllamaLLM:
    return OllamaLLM(
        model=os.getenv("HEBE_CHAT_MODEL", "hebe"),
        emit=emit,
        log_chat=log_chat,
    )


def create_conversation_llm(
    *,
    emit: Optional[EmitFn] = None,
    log_chat: Optional[Callable[[str, str, str], None]] = None,
) -> Any:
    """
    Crea el modelo conversacional principal de Hebe.

    Variables relevantes:
      HEBE_LLM_PROVIDER=local | ollama | openai
      HEBE_API_FALLBACK_TO_LOCAL=true | false
    """
    provider = os.getenv("HEBE_LLM_PROVIDER", "local").strip().lower()
    fallback_to_local = os.getenv("HEBE_API_FALLBACK_TO_LOCAL", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if provider in ("local", "ollama", ""):
        print("[HEBE][LLM] provider=local/ollama", flush=True)
        return _build_local_llm(emit=emit, log_chat=log_chat)

    if provider == "openai":
        print(
            f"[HEBE][LLM] provider=openai model={os.getenv('HEBE_OPENAI_MODEL', 'gpt-5-mini')!r}",
            flush=True,
        )

        openai_llm = OpenAILLM(
            model=os.getenv("HEBE_OPENAI_MODEL", "gpt-5-mini"),
            emit=emit,
            log_chat=log_chat,
        )

        if not os.getenv("OPENAI_API_KEY", "").strip():
            print("[HEBE][LLM][WARN] OPENAI_API_KEY no está configurada", flush=True)
            if fallback_to_local:
                print("[HEBE][LLM] falling back to local/ollama", flush=True)
                return _build_local_llm(emit=emit, log_chat=log_chat)
            return openai_llm

        if fallback_to_local:
            return FallbackConversationLLM(
                primary=openai_llm,
                fallback=_build_local_llm(emit=emit, log_chat=log_chat),
            )

        return openai_llm

    print(
        f"[HEBE][LLM][WARN] HEBE_LLM_PROVIDER={provider!r} no reconocido. Usando local/ollama.",
        flush=True,
    )
    return _build_local_llm(emit=emit, log_chat=log_chat)
