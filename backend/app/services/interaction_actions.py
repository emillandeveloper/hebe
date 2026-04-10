from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.services.db_sqlite import (
    find_app_for_command as db_find_app_for_command,
    add_memory,
    get_active_memories,
    save_app_command,
)


@dataclass
class InteractionActions:
    speak: Callable[[str], None]
    stt: Any
    win: Any

    def open_app_from_text(self, command_text: str):
        command_text = (command_text or "").strip().lower()
        app = db_find_app_for_command(command_text)
        if not app:
            self.speak("No conozco esa aplicación todavía.")
            return
        self.win.open_app(app)

    def store_memory_from_text(self, command: str):
        texto = command
        for pref in ["hebe recuerda que", "eve recuerda que", "recuerda que"]:
            texto = texto.replace(pref, "")
        texto = texto.strip()

        if texto:
            add_memory(texto, category="usuario", importance=2)
            self.speak("De acuerdo, lo recordaré.")
        else:
            self.speak("¿Qué quieres que recuerde exactamente?")
            resp = self.stt.listen()
            if resp:
                add_memory(resp, category="usuario", importance=2)
                self.speak("Lo recordaré.")
            else:
                self.speak("No he entendido nada, lo dejamos para más tarde.")

    def respond_memory_recall(self):
        mems = get_active_memories(limit=5)
        if not mems:
            self.speak("De momento no recuerdo nada especial que me hayas dicho.")
            return
        frases = [m["text"] for m in mems]
        respuesta = "Recuerdo algunas cosas: " + "; ".join(frases)
        self.speak(respuesta)

    def learn_new_app(self):
        self.speak("Vale, vamos a aprender una nueva aplicación. ¿Cómo quieres llamarla?")
        nombre = self.stt.listen()
        if not nombre:
            self.speak("No he entendido el nombre. Lo dejamos para otro momento.")
            return
        nombre = nombre.strip().lower()

        self.speak(f"De acuerdo, la llamaré {nombre}. Ahora dime el comando o la ruta para abrirla.")
        comando = self.stt.listen()
        if not comando:
            self.speak("No he entendido el comando. Cancelamos el registro.")
            return
        comando = comando.strip()

        self.speak("¿Quieres añadir alias para esta aplicación? Por ejemplo, otras formas de llamarla. Si no, di 'no'.")
        alias_text = self.stt.listen()
        if alias_text:
            alias_text = alias_text.strip().lower()
            if alias_text in ("no", "no gracias", "nah", "nop"):
                alias_text = ""
        else:
            alias_text = ""

        app_id = save_app_command(nombre, comando, description="", aliases=alias_text)
        if app_id:
            self.speak(f"He guardado la aplicación {nombre}. Intentaré abrirla cuando me la pidas.")
        else:
            self.speak(f"No he podido guardar la aplicación {nombre}. Puede que ya exista otra con ese nombre.")

    def confirm_action(self, action: str) -> bool:
        self.speak(f"¿Seguro que quieres {action}? Di sí o no.")
        resp = self.stt.listen()
        if not resp:
            return False
        r = resp.strip().lower()
        return r in ("si", "sí") or r.startswith("si ") or r.startswith("sí ")