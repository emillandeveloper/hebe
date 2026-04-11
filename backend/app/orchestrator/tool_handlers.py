# backend/app/orchestrator/tool_handlers.py

from __future__ import annotations

from typing import Any

from .models import make_error, make_success


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

    if hasattr(runtime, "win") and hasattr(runtime.win, "open_app"):
        ok = runtime.win.open_app(app_name)
        if ok:
            return make_success(
                output_text=None,
                data={"opened_app": app_name},
            )
        return make_error(
            error=f"App not found: {app_name}",
            output_text=f"No conozco esa aplicación todavía: {app_name}.",
        )

    if hasattr(runtime, "tools") and hasattr(runtime.tools, "open_app"):
        result = runtime.tools.open_app(app_name)
        return _normalize_tool_result(
            result=result,
            success_text=None,
            fail_text=f"No conozco esa aplicación todavía: {app_name}.",
            data={"opened_app": app_name},
        )

    return make_error(
        error="No open_app backend available",
        output_text="No tengo backend para abrir aplicaciones.",
    )


def handle_close_window(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    target = args.get("target", "active")

    if hasattr(runtime, "win") and hasattr(runtime.win, "close_window"):
        ok = runtime.win.close_window(target)
        if ok:
            return make_success(output_text=None, data={"closed_target": target})
        return make_error(
            error=f"Could not close window: {target}",
            output_text="No he podido cerrar esa ventana.",
        )

    return make_error(
        error="No close_window backend available",
        output_text="No tengo backend para cerrar ventanas.",
    )


def handle_set_volume(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    if "value" not in args:
        return make_error(
            error="Missing value",
            output_text="No me has dicho qué volumen poner.",
        )

    value = int(args["value"])

    if hasattr(runtime, "win") and hasattr(runtime.win, "set_volume"):
        ok = runtime.win.set_volume(value)
        if ok:
            return make_success(output_text=None, data={"volume": value})
        return make_error(
            error=f"Could not set volume: {value}",
            output_text="No he podido cambiar el volumen.",
        )

    return make_error(
        error="No set_volume backend available",
        output_text="No tengo backend para cambiar el volumen.",
    )


def handle_play_music(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    query = str(args.get("query", "")).strip()

    if hasattr(runtime, "win") and hasattr(runtime.win, "play_music"):
        ok = runtime.win.play_music(query=query)
        if ok:
            return make_success(output_text=None, data={"query": query})
        return make_error(
            error=f"Could not play music: {query}",
            output_text="No he podido poner esa música.",
        )

    return make_error(
        error="No play_music backend available",
        output_text="No tengo backend para reproducir música.",
    )


def handle_pause_music(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    if hasattr(runtime, "win") and hasattr(runtime.win, "pause_music"):
        ok = runtime.win.pause_music()
        if ok:
            return make_success(output_text=None)
        return make_error(
            error="Could not pause music",
            output_text="No he podido pausar la música.",
        )

    return make_error(
        error="No pause_music backend available",
        output_text="No tengo backend para pausar música.",
    )


def handle_shutdown_pc(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    if hasattr(runtime, "win") and hasattr(runtime.win, "shutdown_pc"):
        ok = runtime.win.shutdown_pc()
        if ok:
            return make_success(output_text="Apagando el ordenador.")
        return make_error(
            error="Could not shutdown PC",
            output_text="No he podido apagar el ordenador.",
        )

    return make_error(
        error="No shutdown backend available",
        output_text="No tengo backend para apagar el ordenador.",
    )


def handle_restart_pc(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    if hasattr(runtime, "win") and hasattr(runtime.win, "restart_pc"):
        ok = runtime.win.restart_pc()
        if ok:
            return make_success(output_text="Reiniciando el ordenador.")
        return make_error(
            error="Could not restart PC",
            output_text="No he podido reiniciar el ordenador.",
        )

    return make_error(
        error="No restart backend available",
        output_text="No tengo backend para reiniciar el ordenador.",
    )


def handle_sleep_mode(runtime: Any, args: dict[str, Any], source: str = "voice", metadata: dict[str, Any] | None = None):
    return make_success(
        output_text="Vale, me quedo en espera.",
        data={"mode": "sleep"},
    )


def _normalize_tool_result(result: Any, success_text: str | None, fail_text: str, data: dict[str, Any] | None = None):
    data = data or {}

    if isinstance(result, bool):
        if result:
            return make_success(output_text=success_text, data=data)
        return make_error(error=fail_text, output_text=fail_text, data=data)

    if isinstance(result, str):
        if result.strip().lower() in {"ok", "success", "done"}:
            return make_success(output_text=success_text, data=data)
        return make_success(output_text=result, data=data)

    if isinstance(result, dict):
        success = bool(result.get("success", True))
        if success:
            return make_success(
                output_text=result.get("output_text", success_text),
                data={**data, **dict(result.get("data", {}))},
            )
        return make_error(
            error=result.get("error", fail_text),
            output_text=result.get("output_text", fail_text),
            data={**data, **dict(result.get("data", {}))},
        )

    return make_success(output_text=success_text, data=data)