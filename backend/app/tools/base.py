from __future__ import annotations
import os
import subprocess
from typing import Optional, Sequence

def is_windows() -> bool:
    return os.name == "nt"

def normalize_exe_path(p: str) -> str:
    p = (p or "").strip().strip('"')
    return os.path.normpath(p)

def guess_exe_from_command(cmd: str) -> Optional[str]:
    """
    Intenta deducir un exe (nombre de proceso) desde un 'command' guardado.
    Ej:
      - C:\\...\\obs64.exe -> obs64.exe
      - start chrome       -> chrome.exe (best-effort)
      - notepad            -> notepad.exe
    """
    cmd = (cmd or "").strip()
    if not cmd:
        return None

    raw = cmd.strip().strip('"')

    # Ruta directa a exe
    if raw.lower().endswith(".exe") and (":" in raw or raw.startswith("\\\\")):
        return os.path.basename(raw)

    low = raw.lower()

    # "start xxx"
    if low.startswith("start "):
        token = raw[6:].strip().strip('"').split()[0]
        if not token:
            return None
        if not token.lower().endswith(".exe"):
            token += ".exe"
        return token

    # Comandos comunes
    known = {
        "explorer": "explorer.exe",
        "notepad": "notepad.exe",
        "calc": "calc.exe",
        "cmd": "cmd.exe",
        "powershell": "powershell.exe",
    }
    if low in known:
        return known[low]

    # Si parece un "chrome" suelto
    token = raw.split()[0]
    if token and token.isascii():
        if not token.lower().endswith(".exe") and token.isalpha():
            return token + ".exe"

    return None

def run_cmd_windows(cmd: str, cwd: Optional[str] = None) -> None:
    """
    Ejecuta comando en Windows sin bloquear.
    """
    cmd = (cmd or "").strip()
    if not cmd:
        return
    subprocess.Popen(cmd, cwd=cwd, shell=True)

def spawn_detached(exe_path: str, args: Optional[Sequence[str]] = None, cwd: Optional[str] = None) -> None:
    """
    Lanza un exe como proceso independiente (sin bloquear).
    """
    exe_path = normalize_exe_path(exe_path)
    if not exe_path:
        return

    argv = [exe_path] + list(args or [])
    if is_windows():
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        subprocess.Popen(
            argv,
            cwd=cwd,
            shell=False,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        )
    else:
        subprocess.Popen(argv, cwd=cwd, shell=False)
