from __future__ import annotations
import base64
import json
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

def _external_launcher(payload: dict) -> int:
    """Create a user application through Windows' CIM process broker.

    ``Win32_Process.Create`` runs in WmiPrvSE, so the new process is parented by
    that Windows service rather than Uvicorn. ``taskkill /T`` can consequently
    keep removing all Hebe descendants without reaching user applications.
    """
    payload_encoded = base64.b64encode(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    script = r"""
$ErrorActionPreference = 'Stop'
$raw = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:HEBE_EXTERNAL_LAUNCH))
$request = $raw | ConvertFrom-Json
$arguments = @{ CommandLine = [string]$request.command_line }
if ([string]$request.cwd) { $arguments.CurrentDirectory = [string]$request.cwd }
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments $arguments
if ([int]$result.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($result.ReturnValue)" }
[Console]::Out.Write((@{ pid = [int]$result.ProcessId } | ConvertTo-Json -Compress))
""".strip()
    script_encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    CREATE_NO_WINDOW = 0x08000000
    environment = dict(os.environ)
    environment["HEBE_EXTERNAL_LAUNCH"] = payload_encoded
    completed = subprocess.run(
        [
            "powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass", "-EncodedCommand", script_encoded,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=CREATE_NO_WINDOW,
        env=environment,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "external launcher failed").strip()
        raise OSError(detail)
    try:
        receipt = json.loads((completed.stdout or "").strip())
        return int(receipt["pid"])
    except Exception as exc:
        raise OSError("external launcher returned no valid process receipt") from exc


def run_cmd_windows(cmd: str, cwd: Optional[str] = None) -> int | None:
    """
    Ejecuta comando en Windows sin bloquear.
    """
    cmd = (cmd or "").strip()
    if not cmd:
        return None
    if is_windows():
        comspec = os.environ.get("COMSPEC") or r"C:\Windows\System32\cmd.exe"
        command_line = subprocess.list2cmdline([comspec, "/d", "/s", "/c", cmd])
        return _external_launcher({"command_line": command_line, "cwd": cwd or ""})
    return int(subprocess.Popen(cmd, cwd=cwd, shell=True).pid)

def spawn_detached(exe_path: str, args: Optional[Sequence[str]] = None, cwd: Optional[str] = None) -> int | None:
    """
    Lanza un exe como proceso independiente (sin bloquear).
    """
    exe_path = normalize_exe_path(exe_path)
    if not exe_path:
        return None

    argv = [exe_path] + list(args or [])
    if is_windows():
        # CreateProcess does not resolve Windows shell shortcuts itself. Route
        # .lnk registrations through Explorer, still created by the CIM broker,
        # so learned/registered shortcuts keep their shell semantics.
        if exe_path.lower().endswith(".lnk"):
            windows_dir = os.environ.get("WINDIR") or r"C:\Windows"
            argv = [os.path.join(windows_dir, "explorer.exe"), exe_path] + list(args or [])
        return _external_launcher({"command_line": subprocess.list2cmdline(argv), "cwd": cwd or ""})
    return int(subprocess.Popen(argv, cwd=cwd, shell=False).pid)
