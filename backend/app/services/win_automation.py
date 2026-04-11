from __future__ import annotations

import re
import time
import os
from dataclasses import dataclass
from typing import Callable, Optional

# Reuse existing low-level tools (do NOT reimplement)
from app.tools.windows_apps import open_app
from app.tools.windows_input import hotkey, type_text
from pywinauto import Desktop

@dataclass
class WinAutomationConfig:
    """High-level configuration for Windows automation."""
    default_typing_delay_s: float = 0.0


class WinAutomationService:
    """
    High-level Windows automation.

    - Uses app.tools.* for low-level actions.
    - Keeps engine clean: engine decides WHEN, service executes HOW.
    """

    def __init__(
        self,
        config: WinAutomationConfig | None = None,
        emit: Optional[Callable[[str, dict], None]] = None,
        speak: Optional[Callable[[str, str], None]] = None,
    ):
        self.cfg = config or WinAutomationConfig()
        self.emit = emit
        self.speak = speak

    def _emit(self, event: str, data: dict | None = None) -> None:
        if not self.emit:
            return
        try:
            self.emit(event, data or {})
        except Exception:
            pass

    # ----------------------------
    # Core actions
    # ----------------------------

    def open_app(self, app, speak=None, **_ignored) -> bool:
        # sqlite3.Row -> dict
        if app is None:
            self._emit("status", {"win_automation": "open_app_error", "error": "app is None"})
            return False

        if not isinstance(app, dict):
            try:
                app = dict(app)
            except Exception:
                # fallback mínimo: que al menos no pete el log
                app = {"name": str(app), "command": ""}

        self._emit("status", {"win_automation": "open_app", "app": app.get("name")})
        try:
            from app.tools.windows_apps import open_app as tool_open_app
            tool_open_app(app, speak=None)
            return True
        except Exception as e:
            self._emit("status", {"win_automation": "open_app_error", "error": repr(e)})
            return False

    def type_text(self, text: str) -> None:
        """Types text into the currently focused window."""
        self._emit("status", {"win_automation": "type_text"})
        # tools.windows_input.type_text already handles chunking
        type_text(text)

    def press_hotkey(self, keys: list[str]) -> None:
        """Presses a hotkey combination, e.g. ['alt', 'f4']."""
        self._emit("status", {"win_automation": "hotkey", "keys": keys})
        hotkey(keys)

    def close_active_window(self) -> None:
        """Closes the currently active window (Alt+F4)."""
        self.press_hotkey(["alt", "f4"])


    def close_app_by_process_name(self, process_name: str) -> bool:
        pn = (process_name or "").strip().lower()
        if not pn:
            return False

        # Resolve process PIDs
        pids = []
        import psutil
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if (p.info.get("name") or "").lower() == pn:
                    pids.append(p.info["pid"])
            except Exception:
                continue

        if not pids:
            return False

        closed_any = False
        desktop = Desktop(backend="uia")

        # Close top-level windows belonging to those PIDs
        for w in desktop.windows():
            try:
                if w.process_id() in pids:
                    # This sends a close request (WM_CLOSE style), not a kill
                    w.close()
                    closed_any = True
            except Exception:
                continue

        return closed_any
    
    # ----------------------------
    # Helpers for text commands (optional)
    # ----------------------------

    def extract_text_after_keyword(self, command: str, keyword: str) -> str:
        """
        Extracts everything after the first occurrence of keyword.
        Example: 'escribe hola mundo' -> 'hola mundo'
        """
        cmd = (command or "").strip()
        if not cmd:
            return ""
        idx = cmd.lower().find(keyword.lower())
        if idx == -1:
            return ""
        return cmd[idx + len(keyword):].strip()

    def handle_type_command(self, command: str, keyword: str = "escribe") -> bool:
        """
        Handles Spanish-style 'escribe ...' commands:
        - Extracts the text after keyword and types it.
        """
        text = self.extract_text_after_keyword(command, keyword)
        if not text:
            return False
        self.type_text(text)
        return True

    def handle_close_command(self, command: str) -> bool:
        """
        Handles typical close commands in Spanish/English.
        """
        t = (command or "").strip().lower()
        if any(x in t for x in ["cierra ventana", "close window", "cerrar ventana", "alt f4"]):
            self.close_active_window()
            return True
        return False
    
    # ----------------------------
    # Volume (system)
    # ----------------------------
    def handle_volume_command(self, command: str) -> bool:
        """
        System volume via media keys (organic, no window focus required).
        Mirrors controlar_volumen() from engine.
        """
        t = (command or "").lower()

        # Use low-level hotkey tool (pyautogui in tools/windows_input)
        # If you prefer to keep it strictly via tools: create a press_key helper there.
        import pyautogui

        if "sube volumen" in t:
            if self.speak:
                self.speak("Subiendo volumen.", "es")
            for _ in range(5):
                pyautogui.press("volumeup")
            return True

        if "baja volumen" in t:
            if self.speak:
                self.speak("Bajando volumen.", "es")
            for _ in range(5):
                pyautogui.press("volumedown")
            return True

        if "silenciar" in t:
            if self.speak:
                self.speak("Silenciando.", "es")
            pyautogui.press("volumemute")
            return True

        return False

    # ----------------------------
    # YouTube Music (Opera focus + media keys)
    # ----------------------------
    def focus_youtube_music_opera(self) -> bool:
        """
        Focus Opera GX window that contains YouTube Music.
        Mirrors enfocar_opera_youtube() from engine.
        """
        try:
            import pygetwindow as gw
            from pywinauto.application import Application
        except Exception as e:
            self._emit("status", {"win_automation": "focus_youtube_import_error", "error": repr(e)})
            return False

        ventanas = gw.getAllTitles()
        for ventana in ventanas:
            v = ventana or ""
            if ("YouTube Music" in v) or ("music.youtube.com" in v.lower()) or ("Opera GX" in v):
                try:
                    self._emit("status", {"win_automation": "focus_youtube", "title": v})
                    app = Application().connect(title=v)
                    app.top_window().set_focus()
                    return True
                except Exception as e:
                    self._emit("status", {"win_automation": "focus_youtube_error", "error": repr(e)})
                    return False

        self._emit("status", {"win_automation": "focus_youtube_not_found"})
        return False

    def handle_youtube_music_command(self, command: str) -> bool:
        """
        Mirrors controlar_youtube_music() from engine.
        Uses global media keys, no need to focus Opera for play/pause/next/prev/volume.
        """
        t = (command or "").lower()
        try:
            import keyboard
        except Exception as e:
            self._emit("status", {"win_automation": "keyboard_import_error", "error": repr(e)})
            return False

        if "pausa música" in t or "reproduce música" in t:
            keyboard.send("play/pause media")
            return True

        if "siguiente canción" in t:
            keyboard.send("next track")
            return True

        if "canción anterior" in t:
            keyboard.send("previous track")
            return True

        if "sube volumen" in t:
            for _ in range(5):
                keyboard.send("volume up")
            return True

        if "baja volumen" in t:
            for _ in range(5):
                keyboard.send("volume down")
            return True

        if "silenciar música" in t:
            keyboard.send("volume mute")
            return True

        return False

    def play_song_on_youtube_music(self, song: str) -> bool:
        """
        Mirrors buscar_y_reproducir_cancion() from engine.
        This DOES require focusing Opera with YT Music.
        """
        if not song:
            return False

        if not self.focus_youtube_music_opera():
            if self.speak:
                self.speak("No encuentro YouTube Music ahora mismo.", "es")
            return False

        try:
            import keyboard
        except Exception as e:
            self._emit("status", {"win_automation": "keyboard_import_error", "error": repr(e)})
            return False

        self._emit("status", {"win_automation": "play_song", "song": song})
        keyboard.send("/")
        time.sleep(0.5)

        keyboard.write(song)
        time.sleep(0.5)

        keyboard.send("enter")
        time.sleep(2)

        keyboard.send("tab")
        time.sleep(0.3)
        keyboard.send("enter")
        return True

    # ----------------------------
    # Power commands (with confirm callback)
    # ----------------------------
    def handle_power_command(self, command: str, confirm_fn: Callable[[str], bool]) -> bool:
        """
        Mirrors controlar_pc() + confirmar_accion() from engine.
        confirm_fn(action_text)->bool is provided by engine (uses STT).
        """
        t = (command or "").lower()

        if "apaga el ordenador" in t:
            if confirm_fn("apagar el ordenador"):
                if self.speak:
                    self.speak("Apagando el ordenador en 5 segundos.", "es")
                os.system("shutdown /s /t 5")
            return True

        if "reinicia el ordenador" in t:
            if confirm_fn("reiniciar el ordenador"):
                if self.speak:
                    self.speak("Reiniciando el ordenador en 5 segundos.", "es")
                os.system("shutdown /r /t 5")
            return True

        return False
    