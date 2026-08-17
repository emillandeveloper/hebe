from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from app.cognitive.action_runtime import ActionRuntime
from app.core import runtime as runtime_module
from app.services.local_capability import ApplicationDiscoveryService, LocalCapabilityResolver


class WindowsAppDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.user_data = self.root / "user-data"
        self.program_data = self.root / "program-data"
        self.user_profile = self.root / "user-profile"
        self.public_profile = self.root / "public-profile"
        self.user_start = self.user_data / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        self.global_start = self.program_data / "Microsoft" / "Windows" / "Start Menu" / "Programs"
        self.user_start.mkdir(parents=True)
        self.global_start.mkdir(parents=True)
        self.env = patch.dict(os.environ, {
            "APPDATA": str(self.user_data),
            "PROGRAMDATA": str(self.program_data),
            "USERPROFILE": str(self.user_profile),
            "PUBLIC": str(self.public_profile),
            "PROGRAMFILES": str(self.root / "empty-program-files"),
            "PROGRAMFILES(X86)": str(self.root / "empty-program-files-x86"),
            "LOCALAPPDATA": str(self.root / "empty-local-app-data"),
        })
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.temp_dir.cleanup()

    def _exe(self, relative_path: str) -> Path:
        executable = self.root / relative_path
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
        return executable

    @staticmethod
    def _shortcut(root: Path, relative_path: str) -> Path:
        shortcut = root / relative_path
        shortcut.parent.mkdir(parents=True, exist_ok=True)
        shortcut.touch()
        return shortcut

    def _shortcut_discovery(self, resolutions: dict[Path, tuple[str, str] | None]) -> ApplicationDiscoveryService:
        discovery = ApplicationDiscoveryService()
        discovery._resolve_windows_shortcut = Mock(
            side_effect=lambda path: resolutions.get(path)
        )
        discovery._search_registry_app_paths = Mock(return_value=[])
        discovery._search_installed_registry = Mock(return_value=[])
        discovery._search_windows_index = Mock(return_value=[])
        discovery._search_executables = Mock(return_value=[])
        discovery._search_persisted_db_entries = Mock(return_value=[])
        return discovery

    def test_user_start_menu_shortcut_resolves_to_valid_executable(self):
        executable = self._exe("apps/Fixture Tool/tool.exe")
        shortcut = self._shortcut(self.user_start, "Fixture Tool.lnk")
        discovery = self._shortcut_discovery({shortcut: (str(executable), "")})

        candidates = discovery.search("Fixture Tool")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].executable_path, str(executable))
        self.assertEqual(candidates[0].source_location, str(shortcut))
        self.assertTrue(candidates[0].executable)

    def test_global_start_menu_shortcut_resolves_to_valid_executable(self):
        executable = self._exe("apps/Global Tool/global.exe")
        shortcut = self._shortcut(self.global_start, "Global Tool.lnk")
        discovery = self._shortcut_discovery({shortcut: (str(executable), "")})

        candidates = discovery.search("Global Tool")

        self.assertEqual([candidate.executable_path for candidate in candidates], [str(executable)])

    def test_start_menu_subfolder_is_scanned_recursively(self):
        executable = self._exe("apps/Nested Tool/nested.exe")
        shortcut = self._shortcut(self.user_start, "Vendor/Suite/Nested Tool.lnk")
        discovery = self._shortcut_discovery({shortcut: (str(executable), "--profile default")})

        candidates = discovery.search("Nested Tool")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].arguments, "--profile default")
        self.assertEqual(candidates[0].executable_path, str(executable))

    def test_shortcut_display_name_can_differ_from_executable_name(self):
        executable = self._exe("vendor/bin/vendor-host.exe")
        shortcut = self._shortcut(self.user_start, "Photo Studio.lnk")
        discovery = self._shortcut_discovery({shortcut: (str(executable), "")})

        candidates = discovery.search("Photo Studio")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].display_name, "Photo Studio")
        self.assertEqual(candidates[0].executable_name_match, "vendor-host.exe")
        self.assertGreater(candidates[0].confidence, 0.0)

    def test_broken_shortcut_is_diagnostic_but_never_executable(self):
        shortcut = self._shortcut(self.user_start, "Broken Tool.lnk")
        missing = self.root / "missing" / "broken.exe"
        discovery = self._shortcut_discovery({shortcut: (str(missing), "")})

        candidates = discovery.search("Broken Tool")
        resolution = self._resolve(discovery, "Broken Tool")

        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0].executable)
        self.assertIn("path_missing_or_invalid", candidates[0].validation_notes)
        self.assertEqual(resolution.status, "not_found")
        self.assertTrue(discovery.last_diagnostics["discarded"])
        self.assertTrue(discovery.last_diagnostics["rejected_candidates"])

    def test_two_distinct_shortcuts_with_equal_evidence_are_ambiguous(self):
        first_exe = self._exe("one/Ambiguous Tool.exe")
        second_exe = self._exe("two/Ambiguous Tool.exe")
        first_link = self._shortcut(self.user_start, "Vendor One/Ambiguous Tool.lnk")
        second_link = self._shortcut(self.global_start, "Vendor Two/Ambiguous Tool.lnk")
        discovery = self._shortcut_discovery({
            first_link: (str(first_exe), ""),
            second_link: (str(second_exe), ""),
        })

        resolution = self._resolve(discovery, "Ambiguous Tool")

        self.assertEqual(resolution.status, "ambiguous")
        self.assertEqual(resolution.candidate_count, 2)
        self.assertTrue(resolution.clarification_question)

    def test_discovered_app_is_resolved_and_executed_once(self):
        executable = self._exe("apps/Once Tool/once.exe")
        shortcut = self._shortcut(self.user_start, "Once Tool.lnk")
        discovery = self._shortcut_discovery({shortcut: (str(executable), "")})
        runtime = SimpleNamespace(win=SimpleNamespace(open_app=Mock(return_value=True)))
        action_runtime = ActionRuntime(runtime)
        resolver = LocalCapabilityResolver(discovery)
        resolver.resolve_open_application = Mock(wraps=resolver.resolve_open_application)
        action_runtime.local_capability = resolver

        with patch("app.services.local_capability.resolve_whitelisted_app", return_value=None), \
             patch("app.services.local_capability.persist_learned_app", return_value=None):
            result = action_runtime.execute("open_application", {"requested_target": "Once Tool"})

        self.assertTrue(result.success)
        resolver.resolve_open_application.assert_called_once_with("Once Tool")
        runtime.win.open_app.assert_called_once()
        self.assertEqual(runtime.win.open_app.call_args.args[0]["executable_path"], str(executable))

    def test_spaced_display_name_finds_compact_indexed_portable_executable(self):
        executable = self._exe("Users/Public/Documents/WinDS PRO/emu/melonds/melonDS.exe")
        item_url = "file:" + str(executable).replace(os.sep, "/")
        discovery = ApplicationDiscoveryService()
        discovery._iter_windows_index_rows = Mock(side_effect=lambda term: iter([
            ("melonDS.exe", item_url, "melonDS.exe"),
        ]) if term == "melonds" else iter([]))
        discovery._search_registry_app_paths = Mock(return_value=[])
        discovery._search_installed_registry = Mock(return_value=[])
        discovery._search_shortcuts = Mock(return_value=[])
        discovery._search_executables = Mock(return_value=[])
        discovery._search_persisted_db_entries = Mock(return_value=[])

        candidates = discovery.search("Melon DS")
        resolution = self._resolve(discovery, "Melon DS")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source_type, "windows_search_index")
        self.assertEqual(candidates[0].executable_path, str(executable))
        self.assertEqual(resolution.status, "discovered")
        self.assertEqual(
            [call.args[0] for call in discovery._iter_windows_index_rows.call_args_list],
            ["Melon DS", "melonds"],
        )
        self.assertEqual(
            discovery.last_diagnostics["checked_sources"],
            [
                "learned_db",
                "app_paths_registry",
                "installed_registry",
                "start_menu_shortcuts",
                "windows_search_index",
                "exe_scan",
            ],
        )
        self.assertEqual(len(discovery.last_diagnostics["discovered_candidates"]), 1)
        self.assertEqual(discovery.last_diagnostics["rejected_candidates"], [])

    def test_learned_schema_executable_path_survives_restart(self):
        executable = self._exe("portable-suite/bin/PortablePlayer.exe")
        discovery = ApplicationDiscoveryService()
        discovery._search_registry_app_paths = Mock(return_value=[])
        discovery._search_installed_registry = Mock(return_value=[])
        discovery._search_shortcuts = Mock(return_value=[])
        discovery._search_windows_index = Mock(return_value=[])
        discovery._search_executables = Mock(return_value=[])
        learned_record = {
            "app_id": "portableplayer",
            "canonical_name": "Portable Player",
            "executable_path": str(executable),
            "launch_arguments": "--profile default",
            "source": "windows_search_index",
        }

        with patch(
            "app.services.local_capability.app_registry.lookup_learned_app_record",
            return_value=learned_record,
        ):
            resolution = self._resolve(discovery, "Portable Player")

        self.assertEqual(resolution.status, "discovered")
        self.assertEqual(resolution.provenance, "learned_db")
        self.assertEqual(resolution.implementation.executable_path, str(executable))
        self.assertEqual(resolution.implementation.arguments, "--profile default")

    def test_build_runtime_restart_discovers_and_executes_portable_app_once(self):
        executable = self._exe("Users/Public/Documents/Portable Suite/PortablePlayer.exe")
        item_url = "file:" + str(executable).replace(os.sep, "/")
        win = Mock()
        win.open_app.return_value = True
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
            WinAutomationService=Mock(return_value=win),
            TwitchChatCache=Mock(return_value=Mock()),
            TwitchEventMemory=Mock(return_value=Mock()),
            TwitchTargetResolver=Mock(return_value=Mock()),
            TwitchChatClient=Mock(return_value=Mock()),
            TwitchHelixClient=Mock(return_value=Mock()),
            TwitchService=Mock(return_value=Mock()),
            TwitchEventAdapter=Mock(return_value=Mock()),
            TwitchChatBot=Mock(return_value=Mock()),
        ):
            runtime = runtime_module.build_runtime()

        discovery = ApplicationDiscoveryService()
        discovery._iter_windows_index_rows = Mock(side_effect=lambda term: iter([
            ("PortablePlayer.exe", item_url, "PortablePlayer.exe"),
        ]) if term == "portableplayer" else iter([]))
        discovery._search_registry_app_paths = Mock(return_value=[])
        discovery._search_installed_registry = Mock(return_value=[])
        discovery._search_shortcuts = Mock(return_value=[])
        discovery._search_executables = Mock(return_value=[])
        discovery._search_persisted_db_entries = Mock(return_value=[])
        action_runtime = ActionRuntime(runtime)
        resolver = LocalCapabilityResolver(discovery)
        resolver.resolve_open_application = Mock(wraps=resolver.resolve_open_application)
        action_runtime.local_capability = resolver

        with patch("app.services.local_capability.resolve_whitelisted_app", return_value=None), \
             patch("app.services.local_capability.persist_learned_app", return_value=None):
            result = action_runtime.execute(
                "open_application", {"requested_target": "Portable Player"}
            )

        self.assertTrue(result.success)
        resolver.resolve_open_application.assert_called_once_with("Portable Player")
        win.open_app.assert_called_once()
        self.assertEqual(
            win.open_app.call_args.args[0]["executable_path"], str(executable)
        )

    @staticmethod
    def _resolve(discovery: ApplicationDiscoveryService, target: str):
        with patch("app.services.local_capability.resolve_whitelisted_app", return_value=None), \
             patch("app.services.local_capability.persist_learned_app", return_value=None):
            return LocalCapabilityResolver(discovery).resolve_open_application(target)


if __name__ == "__main__":
    unittest.main()
