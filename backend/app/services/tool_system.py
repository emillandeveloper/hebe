from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pyautogui


EmitFn = Callable[[str, Dict[str, Any]], None]
SpeakFn = Callable[[str, str], None]


@dataclass
class ToolContext:
    emit: Optional[EmitFn] = None
    speak: Optional[Callable[[str], None]] = None
    win: Any = None  # WinAutomationService
    open_app_fn: Optional[Callable[[str], Any]] = None
    volume_fn: Optional[Callable[[str], Any]] = None
    power_fn: Optional[Callable[[str], Any]] = None
    memory_fn: Optional[Callable[[str], Any]] = None


class ToolSystem:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.tools: Dict[str, Callable[..., Any]] = {}
        self.register_defaults()

    # -------------------------
    # Event wrapper (tool.start/end/error)
    # -------------------------
    def call(self, name: str, args: Optional[dict] = None, fn: Optional[Callable[[], Any]] = None) -> Any:
        """
        Call a tool by name. If `fn` is provided, wraps that callable as a tool call
        (useful to keep compatibility while migrating).
        """
        tool_id = str(uuid.uuid4())
        payload_args = args or {}

        if self.ctx.emit:
            self.ctx.emit("tool.start", {"id": tool_id, "name": name, "args": payload_args})

        t0 = time.time()
        try:
            if fn is not None:
                result = fn()
            else:
                result = self.exec(name, payload_args)

            if self.ctx.emit:
                self.ctx.emit(
                    "tool.end",
                    {
                        "id": tool_id,
                        "name": name,
                        "ok": True,
                        "ms": int((time.time() - t0) * 1000),
                        "result": result,
                    },
                )
            return result
        except Exception as e:
            if self.ctx.emit:
                self.ctx.emit(
                    "tool.error",
                    {
                        "id": tool_id,
                        "name": name,
                        "ok": False,
                        "ms": int((time.time() - t0) * 1000),
                        "error": str(e),
                    },
                )
            raise

    # -------------------------
    # Registry + execution
    # -------------------------
    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self.tools[name] = fn

    def exec(self, name: str, args: Optional[dict] = None) -> Any:
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        return self.tools[name](**(args or {}))

    # -------------------------
    # Default tools
    # -------------------------
    def register_defaults(self) -> None:
        self.register("open_app", self._open_app)
        self.register("type_text", self._type_text)
        self.register("press_keys", self._press_keys)
        self.register("open_url", self._open_url)
        self.register("close_window", self._close_window)
        self.register("volume", self._volume)

    # ---- tool impls ----
    def _open_app(self, command: str) -> Any:
        """
        command: e.g. 'abre obs'
        Delegates to engine's open_app_fn (which already resolves DB + opens).
        """
        if not self.ctx.open_app_fn:
            raise RuntimeError("open_app_fn not provided in ToolContext")
        return self.ctx.open_app_fn(command)

    def _type_text(self, command: str, interval: float = 0.03) -> dict:
        """
        Backward-compatible with your current router that passes {'command': t}.
        It will type everything after 'escribe' if present; otherwise types raw command.
        """
        text = (command or "").strip()
        low = text.lower()
        if "escribe" in low:
            # Extract after first 'escribe'
            idx = low.find("escribe")
            text = text[idx + len("escribe") :].strip()

        pyautogui.write(text, interval=interval)
        return {"chars": len(text)}

    def _press_keys(self, keys: list[str]) -> dict:
        norm = []
        for k in keys or []:
            kk = str(k).strip().lower()
            if kk in ("control", "ctl"):
                kk = "ctrl"
            if kk == "escape":
                kk = "esc"
            norm.append(kk)

        if not norm:
            return {"pressed": []}

        if len(norm) == 1:
            pyautogui.press(norm[0])
        else:
            pyautogui.hotkey(*norm)

        return {"pressed": norm}

    def _open_url(self, url: str) -> dict:
        os.startfile(url)
        return {"url": url}

    def _close_window(self) -> dict:
        if not self.ctx.win:
            raise RuntimeError("win automation service not provided")
        self.ctx.win.close_active_window()
        return {"ok": True}

    def _volume(self, command: str) -> Any:
        if not self.ctx.volume_fn:
            raise RuntimeError("volume_fn not provided in ToolContext")
        return self.ctx.volume_fn(command)