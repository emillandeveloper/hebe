from app.services.app_registry import resolve_candidates, register_app, resolve_whitelisted_app
from app.services.local_capability import LocalCapabilityResolver
from app.cognitive.models import ActionResult

class ActionRuntime:
    def __init__(self, runtime):
        self.runtime = runtime
        self.local_capability = LocalCapabilityResolver()

    def execute(self, action_name: str, params: dict) -> ActionResult:
        print(f"[HEBE][ACTION] execute name={action_name!r} params={params!r}", flush=True)

        if action_name == "open_app":
            return self._open_app(params)
        if action_name == "open_application":
            return self._open_application(params)

        if action_name == "close_window":
            return self._close_window(params)

        return ActionResult(success=False, error=f"Unknown action: {action_name}")

    def _open_app(self, params: dict) -> ActionResult:
        app_name = str(params.get("app_name", "")).strip()
        if not app_name:
            return ActionResult(success=False, error="Missing app_name")

        candidates = resolve_candidates(app_name)
        if not candidates:
            return ActionResult(
                success=False,
                error=f"App not found: {app_name}",
                data={"app_name": app_name},
            )

        if not hasattr(self.runtime, "win") or not hasattr(self.runtime.win, "open_app"):
            return ActionResult(
                success=False,
                error="runtime.win.open_app no implementado",
                data={"app_name": app_name},
            )

        for candidate in candidates:
            print(f"[HEBE][ACTION][OPEN_APP] trying candidate={candidate!r}", flush=True)
            ok = self.runtime.win.open_app(candidate)
            if ok:
                if candidate.get("source") != "db":
                    saved = register_app(candidate)
                    if saved:
                        candidate = saved

                return ActionResult(
                    success=True,
                    data={
                        "app_name": candidate.get("name", app_name),
                        "app_record": candidate,
                    },
                )

        return ActionResult(
            success=False,
            error=f"Failed opening app: {app_name}",
            data={"app_name": app_name},
        )

    def _open_application(self, params: dict) -> ActionResult:
        app_id = str(params.get("app_id") or params.get("app_name") or "").strip()
        app_record = params.get("app_record")
        if not isinstance(app_record, dict):
            app_record = resolve_whitelisted_app(app_id) if app_id else None
        if not app_record:
            return ActionResult(
                success=False,
                error="app_not_whitelisted",
                data={"error_code": "app_not_whitelisted", "app_id": app_id},
            )

        app_id = str(app_record.get("app_id") or app_id).strip()
        display_name = str(app_record.get("display_name") or app_record.get("name") or app_id).strip()

        resolution = self.local_capability.resolve_open_application(app_record, requested_target=params.get("requested_target") or display_name)
        implementation = resolution.implementation

        if resolution.status == "not_found":
            return ActionResult(
                success=False,
                error="app_not_found",
                data={
                    "error_code": "app_not_found",
                    "app_id": app_id,
                    "app_name": display_name,
                    "app_record": app_record,
                    "clarification_question": resolution.clarification_question,
                    "diagnostics": resolution.diagnostics,
                },
            )

        if resolution.status == "ambiguous" or implementation is None:
            return ActionResult(
                success=False,
                error="ambiguous_app_selection",
                data={
                    "error_code": "ambiguous_app_selection",
                    "app_id": app_id,
                    "app_name": display_name,
                    "app_record": app_record,
                    "clarification_question": resolution.clarification_question,
                    "candidate_count": resolution.candidate_count,
                    "diagnostics": resolution.diagnostics,
                },
            )

        if not hasattr(self.runtime, "win") or not hasattr(self.runtime.win, "open_app"):
            return ActionResult(
                success=False,
                error="runtime_win_open_app_missing",
                data={
                    "error_code": "action_unavailable",
                    "app_id": app_id,
                    "app_name": display_name,
                    "app_record": app_record,
                },
            )

        executable_path = implementation.executable_path
        print(
            "[HEBE][ACTION_EXECUTOR] "
            f"action_type=open_application launching app_id={app_id} path={executable_path!r} source={implementation.source_type}",
            flush=True,
        )

        app_record = dict(app_record)
        app_record["executable_path"] = executable_path
        app_record["command"] = implementation.command

        ok = bool(self.runtime.win.open_app(app_record))
        return ActionResult(
            success=ok,
            error=None if ok else "launch_failed",
            data={
                "error_code": None if ok else "launch_failed",
                "app_id": app_id,
                "app_name": display_name,
                "app_record": app_record,
                "executed_command": implementation.command,
                "launch_source": implementation.source_type,
                "persisted": resolution.persisted,
            },
        )

    def _close_window(self, params: dict) -> ActionResult:
        target = params.get("target", "active")

        if not hasattr(self.runtime, "win"):
            return ActionResult(success=False, error="runtime.win no disponible")

        if target == "active" and hasattr(self.runtime.win, "close_active_window"):
            self.runtime.win.close_active_window()
            return ActionResult(success=True, data={"closed_target": "active"})

        if isinstance(target, str) and hasattr(self.runtime.win, "close_app_by_process_name"):
            ok = self.runtime.win.close_app_by_process_name(target)
            if ok:
                return ActionResult(success=True, data={"closed_target": target})

        return ActionResult(success=False, error=f"Could not close window: {target}")
