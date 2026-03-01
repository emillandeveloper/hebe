# app/services/dispatcher.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.intent_resolver import IntentFrame, HybridIntentResolver, NLUContext
from app.services.nlu_catalog import INTENTS

@dataclass
class DispatchContext:
    speak: callable
    stt: Any
    llm: Any
    tools: Any
    win: Any
    vts_hotkey: callable
    confirm_action: callable
    store_memory_from_text: callable

class Dispatcher:
    def __init__(self, ctx: DispatchContext, intent_resolver: HybridIntentResolver, nlu_ctx: NLUContext):
        self.ctx = ctx
        self.intent_resolver = intent_resolver
        self.nlu_ctx = nlu_ctx

    def dispatch(self, frame: IntentFrame, source: str = "voice") -> str:
        # 1) Conversación
        if frame.type == "chat" or frame.intent == "chat":
            reply = self.ctx.llm.ask(frame.raw_text)
            self.ctx.speak(reply)
            return "continue"

        # 2) Clarify (preguntar y reintentar en voz)
        if frame.type == "clarify":
            question = self._build_clarify_question(frame)
            self.ctx.speak(question)

            # Si viene de UI, no bloqueamos esperando STT.
            if source == "ui":
                return "continue"

            answer = self.ctx.stt.listen()
            if not answer:
                self.ctx.speak("No te he entendido. Dímelo otra vez cuando quieras.")
                return "continue"

            # Re-resolver con la respuesta como nuevo input
            new_frame = self.intent_resolver.resolve(answer, ctx=self.nlu_ctx, source=source)
            return self.dispatch(new_frame, source=source)

        # 3) Acciones
        intent = frame.intent

        # EXIT
        if intent == "exit":
            self.ctx.speak("Hasta luego.")
            return "stop"

        # SLEEP
        if intent == "sleep_mode":
            self.ctx.speak("Entrando en modo de espera...")
            try:
                self.ctx.vts_hotkey("HebeSleep")
            except Exception:
                pass
            return "sleep"

        # CLOSE WINDOW
        if intent == "close_window":
            self.ctx.tools.call("close_window", {})
            return "continue"

        # OPEN APP
        if intent == "open_app":
            app_raw = str(frame.slots.get("app_raw", "")).strip()
            if not app_raw:
                self.ctx.speak("¿Qué aplicación quieres que abra?")
                return "continue"
            # Mantengo tu tool actual, pero pásale lo mínimo limpio
            self.ctx.tools.call("open_app", {"command": app_raw})
            return "continue"

        # VOLUME
        if intent == "volume_control":
            action = frame.slots.get("action")
            # Por compatibilidad con tu handler actual, puedes traducir a texto simple
            cmd = self._volume_action_to_text(action)
            self.ctx.tools.call("volume", {"command": cmd})
            return "continue"

        # YTMUSIC
        if intent == "ytmusic_control":
            action = frame.slots.get("action")
            cmd = self._music_action_to_text(action)
            # ahora mismo lo gestionas por win directamente; lo dejamos igual
            self.ctx.win.handle_youtube_music_command(cmd)
            return "continue"

        # POWER (confirmación obligatoria)
        if intent == "power_control":
            action = frame.slots.get("action")
            human = "apagar el ordenador" if action == "shutdown" else "reiniciar el ordenador"
            if not self.ctx.confirm_action(human):
                self.ctx.speak("Cancelado.")
                return "continue"
            # tu método actual acepta texto comando, mantenemos compatibilidad
            cmd = "apaga el ordenador" if action == "shutdown" else "reinicia el ordenador"
            self.ctx.win.handle_power_command(cmd, confirm_fn=self.ctx.confirm_action)
            return "continue"

        # MEMORY STORE
        if intent == "memory_store":
            text = str(frame.slots.get("text", "")).strip()
            if not text:
                self.ctx.speak("¿Qué quieres que recuerde exactamente?")
                return "continue"
            self.ctx.store_memory_from_text("recuerda que " + text)
            return "continue"

        # default: si llega algo raro, chat
        reply = self.ctx.llm.ask(frame.raw_text)
        self.ctx.speak(reply)
        return "continue"

    def _build_clarify_question(self, frame: IntentFrame) -> str:
        # Preguntas mínimas por slot faltante
        missing = set(frame.missing or [])
        if frame.intent == "open_app" and "app_raw" in missing:
            return "¿Qué aplicación quieres que abra?"
        if frame.intent == "memory_store" and "text" in missing:
            return "¿Qué quieres que recuerde exactamente?"
        if frame.intent == "volume_control" and "action" in missing:
            return "¿Quieres que suba, baje o silencie el volumen?"
        if frame.intent == "ytmusic_control" and "action" in missing:
            return "¿Quieres pausar, reproducir, siguiente o anterior?"
        if frame.intent == "power_control" and "action" in missing:
            return "¿Quieres apagar o reiniciar el ordenador?"
        return "¿Puedes aclararlo un poco?"

    def _volume_action_to_text(self, action: Any) -> str:
        if action == "up":
            return "sube volumen"
        if action == "down":
            return "baja volumen"
        if action == "mute":
            return "silenciar"
        # fallback
        return "sube volumen"

    def _music_action_to_text(self, action: Any) -> str:
        if action == "pause":
            return "pausa música"
        if action == "play":
            return "reproduce música"
        if action == "next":
            return "siguiente canción"
        if action == "prev":
            return "canción anterior"
        if action == "mute":
            return "silenciar música"
        return "pausa música"