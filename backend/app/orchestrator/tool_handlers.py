# backend/app/orchestrator/tool_handlers.py

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from .models import make_error, make_success
from app.services.app_registry import resolve_candidates, register_app
from .models import make_error, make_success
import re

def build_tool_handlers(runtime: Any) -> dict[str, Any]:
    return {
        "open_app": lambda args, source="voice", metadata=None: handle_open_app(runtime, args, source, metadata),
        "close_window": lambda args, source="voice", metadata=None: handle_close_window(runtime, args, source, metadata),
        "set_volume": lambda args, source="voice", metadata=None: handle_set_volume(runtime, args, source, metadata),
        "play_music": lambda args, source="voice", metadata=None: handle_play_music(runtime, args, source, metadata),
        "pause_music": lambda args, source="voice", metadata=None: handle_pause_music(runtime, args, source, metadata),
        "shutdown_pc": lambda args, source="voice", metadata=None: handle_shutdown_pc(runtime, args, source, metadata),
        "restart_pc": lambda args, source="voice", metadata=None: handle_restart_pc(runtime, args, source, metadata),
        "sleep_mode": lambda args, source="voice", metadata=None: handle_sleep_mode(runtime, args, source, metadata),
    }


def handle_open_app(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    app_name = str(args.get("app_name", "")).strip()
    if not app_name:
        return make_error(
            error="Missing app_name",
            output_text="No me has dicho qué aplicación abrir.",
        )

    candidates = resolve_candidates(app_name)
    if not candidates:
        return make_error(
            error=f"App not found: {app_name}",
            output_text=f"No conozco esa aplicación todavía: {app_name}.",
        )

    for candidate in candidates:
        print(f"[HEBE][OPEN_APP] trying candidate={candidate!r}", flush=True)

        if hasattr(runtime, "win") and hasattr(runtime.win, "open_app"):
            ok = runtime.win.open_app(candidate)
            if ok:
                # si no viene de BD, la registramos automáticamente
                if candidate.get("source") != "db":
                    saved = register_app(candidate)
                    if saved:
                        candidate = saved

                spoken = f"Abriendo {candidate.get('name', app_name)}."
                return make_success(
                    output_text=spoken,
                    data={
                        "opened_app": candidate.get("name", app_name),
                        "app_record": candidate,
                    },
                )

    return make_error(
        error=f"Failed opening app: {app_name}",
        output_text=f"No he podido abrir {app_name}.",
    )


def handle_close_window(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    target = args.get("target", "active")

    if hasattr(runtime, "win"):
        if target == "active" and hasattr(runtime.win, "close_active_window"):
            runtime.win.close_active_window()
            return make_success(output_text="Cerrando la ventana activa.", data={"closed_target": "active"})

        if isinstance(target, str) and hasattr(runtime.win, "close_app_by_process_name"):
            ok = runtime.win.close_app_by_process_name(target)
            if ok:
                return make_success(output_text=f"Cerrando {target}.", data={"closed_target": target})

    return make_error(
        error=f"Could not close window: {target}",
        output_text="No he podido cerrar esa ventana.",
    )

import re

def handle_set_volume(
    runtime: Any,
    args: dict[str, Any],
    source: str = "voice",
    metadata: dict[str, Any] | None = None,
):
    direction = args.get("direction")
    value = args.get("value")
    print(f"[HEBE][VOLUME] handle_set_volume args={args!r}", flush=True)
    # Si value viene como texto largo, sacar el número
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            value = int(match.group())
        else:
            value = None

    # Volumen exacto
    if value is not None:
        value = max(0, min(100, int(value)))

        if hasattr(runtime, "win") and hasattr(runtime.win, "set_volume"):
            ok = runtime.win.set_volume(value=value)
            if ok:
                return make_success(
                    output_text=f"Poniendo volumen al {value}%.",
                    data={"volume": value},
                )

        return make_error(
            error=f"Could not set volume to {value}",
            output_text="No he podido cambiar el volumen.",
        )

    # Subir volumen
    if direction == "up":
        if hasattr(runtime, "win") and hasattr(runtime.win, "handle_volume_command"):
            print("[HEBE][VOLUME] calling handle_volume_command('sube volumen')", flush=True)
            ok = runtime.win.handle_volume_command("sube volumen")
            if ok:
                return make_success(
                    output_text="Subiendo el volumen.",
                    data={"direction": "up"},
                )

        return make_error(
            error="Could not increase volume",
            output_text="No he podido subir el volumen.",
        )

    # Bajar volumen
    if direction == "down":
        if hasattr(runtime, "win") and hasattr(runtime.win, "handle_volume_command"):
            ok = runtime.win.handle_volume_command("baja volumen")
            print("[HEBE][VOLUME] calling handle_volume_command('baja volumen')", flush=True)
            if ok:
                return make_success(
                    output_text="Bajando el volumen.",
                    data={"direction": "down"},
                )

        return make_error(
            error="Could not decrease volume",
            output_text="No he podido bajar el volumen.",
        )

    return make_error(
        error="Missing volume info",
        output_text="¿Qué volumen quieres que ponga?",
    )

def handle_play_music(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    query = str(args.get("query", "")).strip()

    if not hasattr(runtime, "win"):
        return make_error(
            error="No play_music backend available",
            output_text="No tengo backend para reproducir música.",
        )

    if query:
        if hasattr(runtime.win, "play_song_on_youtube_music"):
            ok = runtime.win.play_song_on_youtube_music(query)
            if ok:
                return make_success(output_text=f"Reproduciendo {query}.", data={"query": query})

    if hasattr(runtime.win, "handle_youtube_music_command"):
        ok = runtime.win.handle_youtube_music_command("reproduce música")
        if ok:
            return make_success(output_text="Reproduciendo música.", data={"query": query})

    return make_error(
        error=f"Could not play music: {query}",
        output_text="No he podido poner esa música.",
    )


def handle_pause_music(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    if hasattr(runtime, "win") and hasattr(runtime.win, "handle_youtube_music_command"):
        ok = runtime.win.handle_youtube_music_command("pausa música")
        if ok:
            return make_success(output_text="Pausando música.")

    return make_error(
        error="Could not pause music",
        output_text="No he podido pausar la música.",
    )


def handle_shutdown_pc(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    if hasattr(runtime, "win") and hasattr(runtime.win, "handle_power_command"):
        # De momento no llamamos aquí a la confirmación legacy; el orquestador ya confirma antes.
        ok = runtime.win.handle_power_command(
            "apaga el ordenador",
            confirm_fn=lambda _: True,
        )
        if ok:
            return make_success(output_text="Apagando el ordenador.")

    return make_error(
        error="Could not shutdown PC",
        output_text="No he podido apagar el ordenador.",
    )


def handle_restart_pc(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    if hasattr(runtime, "win") and hasattr(runtime.win, "handle_power_command"):
        ok = runtime.win.handle_power_command(
            "reinicia el ordenador",
            confirm_fn=lambda _: True,
        )
        if ok:
            return make_success(output_text="Reiniciando el ordenador.")

    return make_error(
        error="Could not restart PC",
        output_text="No he podido reiniciar el ordenador.",
    )


def handle_sleep_mode(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    return make_success(
        output_text="Vale, me quedo en espera.",
        data={"mode": "sleep"},
    )


# =========================
# Helpers
# =========================

def _find_app_record(app_name: str) -> dict[str, Any] | None:
    """
    Busca una app por nombre o alias en la base SQLite.
    Adapta el path de la BD a backend/data/hebe.db si existe.
    """
    normalized = app_name.strip().lower()
    if not normalized:
        return None

    db_path = _resolve_db_path()
    if db_path is None:
        return _fallback_app_alias(normalized)

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Busca primero por name exacto o alias exacto
        cur.execute(
            """
            SELECT *
            FROM apps
            WHERE lower(name) = ?
               OR lower(alias) = ?
               OR lower(command) LIKE ?
            LIMIT 1
            """,
            (normalized, normalized, f"%{normalized}%"),
        )
        row = cur.fetchone()
        conn.close()

        if row is not None:
            return dict(row)

    except Exception:
        pass

    return _fallback_app_alias(normalized)


def _resolve_db_path() -> Path | None:
    here = Path(__file__).resolve()
    backend_root = here.parents[2]

    candidates = [
        backend_root / "data" / "hebe.db",
        backend_root / "hebe.db",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def _fallback_app_alias(name: str) -> dict[str, Any] | None:
    """
    Fallback mínimo para aliases comunes mientras la BD no resuelva bien.
    """
    aliases = {
        "obs": {
            "id": -1,
            "name": "OBS",
            "alias": "obs",
            "command": "C:\\Program Files\\obs-studio\\bin\\64bit\\obs64.exe",
        },
        "paint": {
            "id": -1,
            "name": "Paint",
            "alias": "paint",
            "command": "mspaint.exe",
        },
        "chrome": {
            "id": -1,
            "name": "Chrome",
            "alias": "chrome",
            "command": "chrome.exe",
        },
        "discord": {
            "id": -1,
            "name": "Discord",
            "alias": "discord",
            "command": "discord.exe",
        },
        "opera": {
            "id": -1,
            "name": "Opera GX",
            "alias": "opera",
            "command": "opera.exe",
        },
    }

    return aliases.get(name)


def _set_volume_via_steps(runtime: Any, value: int):
    """
    Tu WinAutomation actual no tiene set_volume(value) directo.
    Para no romper nada, dejamos una respuesta honesta.
    """
    return make_error(
        error=f"Absolute volume not supported yet: {value}",
        output_text="Todavía no sé poner un volumen exacto directamente.",
    )