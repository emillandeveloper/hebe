from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.cognitive.input_event import InputEvent
from app.cognitive.local_app_planner import LocalAppActionPlanner
from app.services.app_registry import resolve_whitelisted_app
from app.services.direct_stt_command import (
    DirectUtteranceIntentFamily,
    parse_direct_stt_command,
)
from app.services.stt_whisper import (
    STTConfig,
    STTService,
    build_stt_prompt_profile,
)


class _FakeModel:
    def transcribe(self, *_args, **_kwargs):
        return [], SimpleNamespace(language="es", language_probability=1.0)


class DirectSTTRoutingTests(unittest.TestCase):
    def test_extract_melon_ds_target(self):
        result = parse_direct_stt_command("Hebe, abre Melon DS.")
        self.assertEqual(result.action_verb, "open")
        self.assertEqual(result.raw_target, "Melon DS")
        self.assertEqual(result.detected_intent_family, "application_action")

    def test_extract_target_with_eb_wake(self):
        result = parse_direct_stt_command("E.B. Abre Melon DS")
        self.assertTrue(result.wake_detected)
        self.assertEqual(result.raw_target, "Melon DS")

    def test_extract_obs_target_and_punctuation(self):
        result = parse_direct_stt_command("E.V. Abre O.B.S.")
        self.assertEqual(result.raw_target, "OBS")
        self.assertEqual(result.action_verb, "open")

    def test_direct_questions_are_not_application_actions(self):
        for text in ("Ebe, ¿estás ahí?", "Hebe, ¿cómo estás?"):
            result = parse_direct_stt_command(text)
            self.assertEqual(result.detected_intent_family, DirectUtteranceIntentFamily.DIRECT_QUESTION.value)
            self.assertFalse(result.action_verb)
            self.assertFalse(result.raw_target)

    def test_target_not_lost_between_redecode_and_resolver(self):
        service = STTService(STTConfig(model_size="medium", device="cpu", compute_type="int8"))
        hypothesis = service._build_command_hypothesis(
            "Ebe, abre Melón de Ese.",
            "Hebe, abre Melon DS.",
            {"detected_language": "es"},
            {"detected_language": "es", "language_allowed": True},
        )
        self.assertEqual(hypothesis.final_command_text, "Hebe, abre Melon DS.")
        self.assertEqual(service.last_direct_stt_result["raw_target"], "Melon DS")
        self.assertEqual(hypothesis.decision, "execute")

    def test_canonical_second_pass_reaches_local_app_planner(self):
        event = InputEvent(
            source="stt_voice",
            raw_text="Ebe, abre Melón de Ese.",
            normalized_text="hebe abre melon ds",
            stt_metadata={
                "direct_stt_command": parse_direct_stt_command(
                    "Hebe, abre Melon DS.",
                    ambient_text="Ebe, abre Melón de Ese.",
                ).to_dict()
            },
        )
        plan = LocalAppActionPlanner().plan(event)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.slots["application_target"], "Melon DS")
        self.assertEqual(set(plan.slots), {"application_target"})

    def test_portable_app_regression_is_not_a_builtin_or_special_alias(self):
        for target in ("melonds", "melon ds", "melón de ese"):
            self.assertIsNone(resolve_whitelisted_app(target))

    def test_app_prompt_is_focused(self):
        prompt = build_stt_prompt_profile("app_command", max_chars=300)
        self.assertIn("OBS", prompt)
        self.assertIn("abre", prompt)
        for excluded in ("Nuria", "Totodile", "Persona", "Final Fantasy", "Twitch"):
            self.assertNotIn(excluded, prompt)

    def test_promotion_terms_only_load_for_promotion_profile(self):
        app_prompt = build_stt_prompt_profile("app_command", max_chars=300)
        promo_prompt = build_stt_prompt_profile("promotion_command", max_chars=300)
        self.assertNotIn("shoutout", app_prompt)
        self.assertIn("shoutout", promo_prompt)

    def test_prompt_echo_retries_without_prompt_and_preserves_ambient_obs(self):
        service = STTService(STTConfig(model_size="medium", device="cpu", compute_type="int8"))
        service._model = _FakeModel()
        service.engine_status = "ready"
        calls = []

        def retry(*_args, **kwargs):
            calls.append(kwargs)
            return (
                "Hebe, Ebe, Eve, OBS, Twitch, stream, chat, promo, shoutout",
                {"language_allowed": True, "options": {}},
            )

        service._transcribe_audio = retry
        logs = io.StringIO()
        with redirect_stdout(logs):
            selected, _, debug = service._recover_prompt_echo_command(
                audio_np=np.zeros(1600, dtype=np.float32),
                ambient_text="E.V. Abre O.B.S.",
                ambient_meta={"language_allowed": True},
                command_text="Hebe, Ebe, Eve, OBS, Twitch, stream, chat, promo, shoutout",
                command_meta={"language_allowed": True, "options": {"initial_prompt": "Hebe, OBS."}},
                prompt_profile="app_command",
            )
        self.assertEqual(selected, "E.V. Abre O.B.S.")
        self.assertTrue(debug["ambient_fallback_used"])
        self.assertFalse(calls[0]["force_prompt"])
        self.assertIn("[HEBE][STT_COMMAND_REDECODE_FALLBACK]", logs.getvalue())


if __name__ == "__main__":
    unittest.main()
