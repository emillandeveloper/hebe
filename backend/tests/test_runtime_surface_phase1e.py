from __future__ import annotations

import inspect
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.cognitive.action_runtime import ActionRuntime
from app.cognitive.models import ActionResult, Plan, PlanStep
from app.cognitive.plan_executor import PlanExecutor
from app.core import runtime as runtime_module
from app.core.runtime import HebeRuntime
from app.scripts.eval_cognitive_router import evaluate
from app.services.local_capability import ApplicationCandidate, CapabilityResolution


class RuntimeSurfacePhase1ETests(unittest.TestCase):
    def setUp(self) -> None:
        self.win = SimpleNamespace(open_app=Mock(return_value=True))
        self.runtime = SimpleNamespace(win=self.win)
        self.action_runtime = ActionRuntime(self.runtime)

    @staticmethod
    def _candidate() -> ApplicationCandidate:
        return ApplicationCandidate(
            canonical_name="obs",
            display_name="OBS Studio",
            executable_path=r"C:\Fixture\obs64.exe",
            source_type="fixture",
            source_location=r"C:\Fixture\obs64.exe",
            confidence=1.0,
            exists=True,
            executable=True,
        )

    def test_b_known_application_resolves_and_executes_once(self) -> None:
        candidate = self._candidate()
        self.action_runtime.local_capability = SimpleNamespace(
            resolve_open_application=Mock(return_value=CapabilityResolution(
                capability="open_application",
                requested_target="OBS",
                canonical_target="obs",
                status="known",
                implementation=candidate,
                candidate_count=1,
                confidence=1.0,
                provenance="fixture",
                app_record={"app_id": "obs", "display_name": "OBS Studio"},
            )),
        )

        result = self.action_runtime.execute("open_application", {"requested_target": "OBS"})

        self.assertTrue(result.success)
        self.action_runtime.local_capability.resolve_open_application.assert_called_once_with("OBS")
        self.win.open_app.assert_called_once()

    def test_c_ambiguous_application_never_executes(self) -> None:
        self.action_runtime.local_capability = SimpleNamespace(
            resolve_open_application=Mock(return_value=CapabilityResolution(
                capability="open_application",
                requested_target="Studio",
                canonical_target="studio",
                status="ambiguous",
                candidate_count=2,
                clarification_question="¿Qué Studio quieres abrir?",
            )),
        )

        result = self.action_runtime.execute("open_application", {"requested_target": "Studio"})

        self.assertFalse(result.success)
        self.assertEqual(result.error, "ambiguous_app_selection")
        self.win.open_app.assert_not_called()

    def test_d_unknown_application_never_executes(self) -> None:
        self.action_runtime.local_capability = SimpleNamespace(
            resolve_open_application=Mock(return_value=CapabilityResolution(
                capability="open_application",
                requested_target="Missing",
                canonical_target="missing",
                status="not_found",
            )),
        )

        result = self.action_runtime.execute("open_application", {"requested_target": "Missing"})

        self.assertFalse(result.success)
        self.assertEqual(result.error, "app_not_found")
        self.win.open_app.assert_not_called()

    def test_e_f_hebe_runtime_exposes_no_parallel_tool_or_action_surface(self) -> None:
        names = {field.name for field in fields(HebeRuntime)}
        self.assertNotIn("tools", names)
        self.assertNotIn("actions", names)

    def test_g_plan_executor_executes_authorized_action(self) -> None:
        action_runtime = Mock()
        action_runtime.execute.return_value = ActionResult(success=True, data={"app_id": "obs"})
        executor = PlanExecutor(Mock(), action_runtime)
        decision = {
            "intent": "command_open_app",
            "source": "ui",
            "authority": "owner",
            "allowed_capabilities": ["pc.open_application"],
            "blocked_capabilities": [],
            "allowed_step_types": ["action"],
            "blocked_step_types": [],
            "should_stop_pipeline": False,
        }
        plan = Plan(
            steps=[PlanStep(
                type="action",
                capability_id="pc.open_application",
                data={"name": "open_application", "params": {"requested_target": "OBS"}},
            )],
            metadata={"cognitive_decision": decision},
        )

        result = executor.execute(plan)

        self.assertTrue(result.ok)
        action_runtime.execute.assert_called_once_with("open_application", {"requested_target": "OBS"})
        self.assertEqual(executor.last_guard_results[0]["allowed"], True)

    def test_h_plan_executor_guard_blocks_unauthorized_action(self) -> None:
        action_runtime = Mock()
        executor = PlanExecutor(Mock(), action_runtime)
        plan = Plan(
            steps=[PlanStep(
                type="action",
                capability_id="pc.open_application",
                data={"name": "open_application", "params": {"requested_target": "OBS"}},
            )],
            metadata={"cognitive_decision": {
                "intent": "command_open_app",
                "source": "twitch_chat",
                "authority": "viewer",
                "allowed_capabilities": ["pc.open_application"],
                "blocked_capabilities": [],
                "allowed_step_types": ["action"],
                "blocked_step_types": [],
                "should_stop_pipeline": False,
            }},
        )

        result = executor.execute(plan)

        self.assertFalse(result.ok)
        self.assertEqual(result.results[0].error, "authority_not_authorized")
        action_runtime.execute.assert_not_called()

    def test_i_build_runtime_constructs_core_contract_without_legacy_surfaces(self) -> None:
        stt = Mock()
        state = SimpleNamespace(tts_enabled=False)
        with patch.multiple(
            runtime_module,
            HebeState=Mock(return_value=state),
            STTService=Mock(return_value=stt),
            build_speak=Mock(return_value=Mock()),
            get_setting=Mock(return_value=""),
            create_conversation_llm=Mock(return_value=Mock()),
            OllamaIntentClient=Mock(return_value=Mock()),
            WinAutomationService=Mock(return_value=Mock()),
            TwitchChatCache=Mock(return_value=Mock()),
            TwitchEventMemory=Mock(return_value=Mock()),
            TwitchTargetResolver=Mock(return_value=Mock()),
            TwitchChatClient=Mock(return_value=Mock()),
            TwitchHelixClient=Mock(return_value=Mock()),
            TwitchService=Mock(return_value=Mock()),
            TwitchEventAdapter=Mock(return_value=Mock()),
            TwitchChatBot=Mock(return_value=Mock()),
        ), patch.dict("os.environ", {}, clear=False):
            runtime = runtime_module.build_runtime()

        self.assertIs(runtime.stt, stt)
        self.assertFalse(hasattr(runtime, "tools"))
        self.assertFalse(hasattr(runtime, "actions"))

    def test_j_production_source_has_no_legacy_execution_surface(self) -> None:
        app_root = Path(__file__).resolve().parents[1] / "app"
        forbidden = ("ToolSystem", "InteractionActions", "runtime.tools", "runtime.actions", "open_app_from_text")
        hits = []
        for path in app_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in forbidden):
                hits.append(str(path.relative_to(app_root)))
        self.assertEqual(hits, [])

    def test_k_canonical_router_dev_evaluation_imports_and_passes(self) -> None:
        self.assertEqual(evaluate(), [])
        self.assertNotIn("app.orchestrator", inspect.getsource(evaluate))


if __name__ == "__main__":
    unittest.main()
