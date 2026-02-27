from __future__ import annotations

import re
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
            tool_open_app(app, speak=speak or self.speak)
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