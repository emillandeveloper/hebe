from __future__ import annotations
import os
import time
from typing import Any, Dict, Optional, Callable

import psutil

from .base import spawn_detached, run_cmd_windows, normalize_exe_path, guess_exe_from_command
from .registry import register_app_usage, update_app_fields

SpeakFn = Callable[[str], None]

def is_process_running(process_name: str) -> bool:
    pn = (process_name or "").strip().lower()
    if not pn:
        return False

    for p in psutil.process_iter(["name"]):
        try:
            name = (p.info.get("name") or "").lower()
            if name == pn:
                return True
        except Exception:
            continue
    return False

def try_focus_app_window(app: Dict[str, Any]) -> bool:
    """
    Best-effort: si tienes pygetwindow instalado, intenta traer la ventana al frente.
    Si no, devuelve False sin petar.
    """
    try:
        import pygetwindow as gw  # type: ignore
    except Exception:
        return False

    title = (app.get("window_title") or "").strip()
    candidates = []
    if title:
        candidates = gw.getWindowsWithTitle(title)
    if not candidates:
        # fallback por name
        name = (app.get("name") or "").strip()
        if name:
            candidates = gw.getWindowsWithTitle(name)

    if not candidates:
        return False

    w = candidates[0]
    try:
        w.restore()
    except Exception:
        pass
    try:
        w.activate()
        return True
    except Exception:
        return False

def learn_process_name_after_launch(app_id: int, expected_exe: Optional[str], timeout: float = 8.0) -> Optional[str]:
    """
    Heurística:
    - Guarda snapshot de procesos (pid, create_time)
    - Espera a que aparezcan procesos nuevos
    - Si expected_exe está, prioriza match por nombre o ruta
    """
    expected_exe = (expected_exe or "").strip().lower()
    t0 = time.time()

    before = {}
    for p in psutil.process_iter(["pid", "name", "create_time", "exe"]):
        try:
            before[p.info["pid"]] = p.info.get("create_time")
        except Exception:
            continue

    while time.time() - t0 < timeout:
        time.sleep(0.25)

        new_candidates = []
        for p in psutil.process_iter(["pid", "name", "create_time", "exe"]):
            try:
                pid = p.info["pid"]
                ct = p.info.get("create_time")
                if pid in before:
                    continue
                name = (p.info.get("name") or "").lower()
                exe = (p.info.get("exe") or "").lower()
                new_candidates.append((name, exe))
            except Exception:
                continue

        if not new_candidates:
            continue

        # 1) match por exe esperado
        if expected_exe:
            for name, exe in new_candidates:
                if name == expected_exe:
                    update_app_fields(app_id, process_name=name)
                    return name
                if exe and expected_exe in exe:
                    update_app_fields(app_id, process_name=name)
                    return name

        # 2) fallback: primer proceso nuevo “con pinta”
        name, _exe = new_candidates[0]
        if name:
            update_app_fields(app_id, process_name=name)
            return name

    return None

def open_app(app: Dict[str, Any], speak: SpeakFn) -> None:
    """
    Abre una app evitando duplicar instancias si ya está abierta.
    """
    name = app["name"]
    cmd = app["command"]

    process_name = (app.get("process_name") or "").strip()
    if not process_name:
        process_name = guess_exe_from_command(cmd) or ""

    # Ya está abierta -> focus
    if process_name and is_process_running(process_name):
        focused = try_focus_app_window(app)
        if focused:
            speak(f"{name} ya estaba abierto. Lo he puesto en primer plano.")
        else:
            speak(f"{name} ya está abierto.")
        try:
            register_app_usage(app["id"])
        except Exception:
            pass
        return

    speak(f"Abriendo {name}.")

    cmd_str = (cmd or "").strip()

    # Ruta directa
    if cmd_str.lower().endswith(".exe") and (":" in cmd_str or cmd_str.startswith("\\\\")):
        exe_path = normalize_exe_path(cmd_str)
        exe_dir = os.path.dirname(exe_path) or None
        spawn_detached(exe_path, cwd=exe_dir)
    else:
        # start xxx / notepad / explorer etc
        run_cmd_windows(cmd_str)

    register_app_usage(app["id"])

    # Aprendizaje post-lanzado
    if not (app.get("process_name") or "").strip():
        expected = guess_exe_from_command(cmd_str)
        learned = learn_process_name_after_launch(app["id"], expected_exe=expected)
        if learned:
            print(f"🧠 Aprendido process_name para {name}: {learned}")
