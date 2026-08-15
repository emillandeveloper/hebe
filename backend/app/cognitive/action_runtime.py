from app.services.local_capability import LocalCapabilityResolver
from app.cognitive.models import ActionResult

class ActionRuntime:
    def __init__(self, runtime):
        self.runtime = runtime
        self.local_capability = LocalCapabilityResolver()

    def execute(self, action_name: str, params: dict) -> ActionResult:
        print(f"[HEBE][ACTION] execute name={action_name!r} params={params!r}", flush=True)

        if action_name == "open_application":
            return self._open_application(params)

        if action_name == "close_window":
            return self._close_window(params)

        return ActionResult(success=False, error=f"Unknown action: {action_name}")

    def _open_application(self, params: dict) -> ActionResult:
        requested_target = str(params.get("requested_target") or "").strip()
        if not requested_target:
            return ActionResult(
                success=False,
                error="application_target_missing",
                data={"error_code": "application_target_missing"},
            )

        resolution = self.local_capability.resolve_open_application(requested_target)
        implementation = resolution.implementation
        app_record = dict(resolution.app_record or {})
        app_id = str(app_record.get("app_id") or getattr(implementation, "canonical_name", "") or requested_target).strip()
        display_name = str(app_record.get("display_name") or app_record.get("name") or resolution.canonical_target or requested_target).strip()

        if resolution.status == "not_found":
            error_code = "app_path_missing" if resolution.diagnostics.get("registered") else "app_not_found"
            return ActionResult(
                success=False,
                error=error_code,
                data={
                    "error_code": error_code,
                    "app_id": app_id,
                    "app_name": display_name,
                    "app_record": app_record or None,
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
