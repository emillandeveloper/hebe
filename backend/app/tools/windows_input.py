from __future__ import annotations
from typing import Sequence

def hotkey(keys: Sequence[str]) -> None:
    try:
        import pyautogui  # type: ignore
    except Exception:
        raise RuntimeError("pyautogui no está instalado")

    pyautogui.hotkey(*keys)

def type_text(text: str) -> None:
    try:
        import pyautogui  # type: ignore
    except Exception:
        raise RuntimeError("pyautogui no está instalado")

    pyautogui.write(text or "", interval=0.01)
