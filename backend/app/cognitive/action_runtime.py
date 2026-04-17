from app.services.app_registry import resolve_candidates, register_app
from app.cognitive.models import ActionResult

class ActionRuntime:
    def __init__(self, runtime):
        self.runtime = runtime

    def execute(self, action_name: str, params: dict) -> ActionResult:
        print(f"[HEBE][ACTION] execute name={action_name!r} params={params!r}", flush=True)

        if action_name == "open_app":
            return self._open_app(params)

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