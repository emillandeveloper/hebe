from __future__ import annotations

import os
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.cognitive.cognitive_router import CognitiveRouter
from app.cognitive.input_event import InputEvent
from app.cognitive.local_app_planner import LocalAppActionPlanner
from app.cognitive import speech_act_pipeline
from app.services.app_registry import AppRegistryEntry, resolve_whitelisted_app
from app.services.local_capability import (
    ApplicationCandidate,
    LocalCapabilityResolver,
)
from tests.test_voice_command_pipeline import make_engine, wire_canonical_app_pipeline


def _candidate(name: str, path: str, *, confidence: float = 0.9) -> ApplicationCandidate:
    return ApplicationCandidate(
        canonical_name=name.casefold().replace(" ", ""),
        display_name=name,
        executable_path=path,
        source_type="discovered_fixture",
        source_location=path,
        executable_name_match=os.path.basename(path),
        confidence=confidence,
        exists=True,
        executable=True,
    )


def _route(text: str, *, source: str = "ui", authority: str = "owner", addressed: bool = True):
    return CognitiveRouter().route(SimpleNamespace(
        input_text=text,
        source=source,
        authority=authority,
        addressed_to_hebe=addressed,
        state_snapshot={},
        firewall_decision="allow",
        stream_is_live=True,
        route_hints=[],
        internal_event=None,
    ))


class AppOpenArchitectureTests(unittest.TestCase):
    def test_steam_command_extracts_then_resolves_and_executes_once(self):
        engine = wire_canonical_app_pipeline(make_engine())
        engine._deliver_manual_reply = lambda _text, *, source: None
        discovery = Mock()
        discovery.search.return_value = [_candidate("Steam", r"C:\Fixture\steam.exe")]
        resolver = LocalCapabilityResolver(discovery)
        resolver.resolve_open_application = Mock(wraps=resolver.resolve_open_application)
        engine.action_runtime.local_capability = resolver
        planner = LocalAppActionPlanner()
        planner.plan = Mock(wraps=planner.plan)
        engine.local_app_planner = planner
        engine.deliberation_service.local_app_planner = planner
        engine.plan_executor.execute = Mock(wraps=engine.plan_executor.execute)
        engine.action_runtime.execute = Mock(wraps=engine.action_runtime.execute)
        registry_lookup = Mock(wraps=resolve_whitelisted_app)

        with patch("app.services.local_capability.resolve_whitelisted_app", registry_lookup), \
             patch("app.services.local_capability.persist_learned_app", return_value=None):
            result = engine.cognitive_flow("Hebe, abre Steam", source="ui")

        self.assertEqual(result, "continue")
        engine.plan_executor.execute.assert_called_once()
        engine.action_runtime.execute.assert_called_once_with(
            "open_application", {"requested_target": "Steam"}
        )
        planner.plan.assert_called_once()
        resolver.resolve_open_application.assert_called_once_with("Steam")
        registry_lookup.assert_called_once_with("Steam")
        discovery.search.assert_called_once()
        self.assertEqual(len(engine.runtime.win.opened), 1)
        self.assertEqual(engine.runtime.win.opened[0]["app_id"], "steam")

    def test_reported_command_is_not_interpreted_as_open_application(self):
        text = "He dicho 'abre Steam' tres veces, deja de insistir."
        decision = _route(text)
        plan = LocalAppActionPlanner().plan(
            InputEvent(source="ui", raw_text=text, normalized_text=text),
        )
        self.assertNotEqual(decision.intent, "command_open_app")
        self.assertIsNone(plan)

    def test_door_feedback_is_not_interpreted_as_open_application(self):
        text = "Mira, otra vez con lo de abre la puerta esa..."
        decision = _route(text)
        plan = LocalAppActionPlanner().plan(
            InputEvent(source="ui", raw_text=text, normalized_text=text),
        )
        self.assertNotEqual(decision.intent, "command_open_app")
        self.assertIsNone(plan)

    def test_ambient_door_narration_cannot_open_an_application(self):
        text = "y entonces abre la puerta y aparece el jefe"
        decision = _route(text, source="ambient_stt", authority="ambient", addressed=False)
        plan = LocalAppActionPlanner().plan(
            InputEvent(source="ambient_stt", raw_text=text, normalized_text=text),
        )
        self.assertEqual(decision.intent, "stream_context_update")
        self.assertNotIn("pc.open_application", decision.allowed_capabilities)
        self.assertIsNone(plan)

    def test_known_builtin_resolves_normally(self):
        record = AppRegistryEntry(
            app_id="fixture",
            display_name="Fixture App",
            aliases=("fixture",),
            executable_path=r"C:\Fixture\fixture.exe",
            source="builtin",
        ).as_dict()
        with patch("app.services.local_capability.resolve_whitelisted_app", return_value=record), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.is_file", return_value=True):
            resolution = LocalCapabilityResolver(Mock()).resolve_open_application("fixture")
        self.assertEqual(resolution.status, "known")
        self.assertEqual(resolution.app_record["app_id"], "fixture")

    def test_env_configured_app_resolves_normally(self):
        with patch.dict(os.environ, {"HEBE_APP_OBS_PATH": sys.executable}):
            resolution = LocalCapabilityResolver(Mock()).resolve_open_application("obs")
        self.assertEqual(resolution.status, "known")
        self.assertEqual(resolution.implementation.source_type, "env")

    def test_discovered_app_resolves_without_builtin_registration(self):
        discovery = Mock()
        discovery.search.return_value = [_candidate("Fixture Tool", r"C:\Fixture\tool.exe")]
        with patch("app.services.local_capability.resolve_whitelisted_app", return_value=None), \
             patch("app.services.local_capability.persist_learned_app", return_value=None):
            resolution = LocalCapabilityResolver(discovery).resolve_open_application("Fixture Tool")
        self.assertEqual(resolution.status, "discovered")
        self.assertEqual(resolution.app_record["app_id"], "fixturetool")

    def test_learned_app_resolves_through_the_same_capability_owner(self):
        learned = _candidate("Learned Tool", r"C:\Fixture\learned.exe")
        learned.source_type = "learned_db"
        discovery = Mock()
        discovery.search.return_value = [learned]
        with patch("app.services.local_capability.resolve_whitelisted_app", return_value=None), \
             patch("app.services.local_capability.persist_learned_app", return_value=None):
            resolution = LocalCapabilityResolver(discovery).resolve_open_application("Learned Tool")
        self.assertEqual(resolution.status, "discovered")
        self.assertEqual(resolution.provenance, "learned_db")

    def test_unknown_app_is_not_found(self):
        discovery = Mock()
        discovery.search.return_value = []
        with patch("app.services.local_capability.resolve_whitelisted_app", return_value=None):
            resolution = LocalCapabilityResolver(discovery).resolve_open_application("Definitely Missing")
        self.assertEqual(resolution.status, "not_found")
        self.assertIsNone(resolution.implementation)

    def test_genuinely_ambiguous_candidates_request_clarification(self):
        discovery = Mock()
        discovery.search.return_value = [
            _candidate("Fixture", r"C:\One\fixture.exe", confidence=0.75),
            _candidate("Fixture", r"D:\Two\fixture.exe", confidence=0.74),
        ]
        with patch("app.services.local_capability.resolve_whitelisted_app", return_value=None):
            resolution = LocalCapabilityResolver(discovery).resolve_open_application("Fixture")
        self.assertEqual(resolution.status, "ambiguous")
        self.assertEqual(resolution.candidate_count, 2)
        self.assertTrue(resolution.clarification_question)

    def test_steam_is_not_a_production_builtin(self):
        self.assertIsNone(resolve_whitelisted_app("steam"))

    def test_speech_act_envelope_has_no_runtime_compatibility_alias(self):
        self.assertTrue(hasattr(speech_act_pipeline, "SpeechActInputEnvelope"))
        self.assertFalse(hasattr(speech_act_pipeline, "InputEnvelope"))


if __name__ == "__main__":
    unittest.main()
