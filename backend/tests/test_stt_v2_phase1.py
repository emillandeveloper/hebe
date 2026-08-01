from __future__ import annotations

import io
import os
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.services.stt_whisper import (
    STTConfig,
    STTMode,
    STTService,
    is_stt_prompt_hotword_list,
)


class FakeModel:
    def __init__(self, text="Abre OBS", language="es"):
        self.text = text
        self.language = language
        self.calls = []

    def transcribe(self, audio, language=None, task=None, beam_size=None, vad_filter=None,
                   temperature=None, initial_prompt=None, hotwords=None):
        self.calls.append({
            "language": language, "initial_prompt": initial_prompt, "hotwords": hotwords,
        })
        return [SimpleNamespace(text=self.text)], SimpleNamespace(language=self.language, language_probability=0.97)


class STTV2Phase1Tests(unittest.TestCase):
    def service(self, text="Abre OBS", language="es"):
        service = STTService(STTConfig(model_size="medium", device="cpu", compute_type="int8"))
        service._model = FakeModel(text, language)
        service.engine_status = "ready"
        service.effective_device = "cpu"
        service.effective_compute_type = "int8"
        return service

    def test_default_model_is_medium(self):
        with patch.dict(os.environ, {"HEBE_WHISPER_MODEL": "medium"}):
            self.assertEqual(STTConfig().model_size, "medium")

    def test_configured_cuda_profile_is_used(self):
        with patch.dict(os.environ, {"HEBE_WHISPER_DEVICE": "cuda", "HEBE_WHISPER_COMPUTE": "float16"}):
            cfg = STTConfig()
        self.assertEqual((cfg.device, cfg.compute_type), ("cuda", "float16"))

    def test_cuda_failure_is_visible_and_not_silent(self):
        service = STTService(STTConfig(model_size="medium", device="cuda", compute_type="int8_float16", explicit_cpu_fallback=False))
        with patch("app.services.stt_whisper.WhisperModel", side_effect=RuntimeError("CUDA unavailable")):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                service.init()
        health = service.health_snapshot()
        self.assertEqual(health["engine_status"], "error")
        self.assertEqual(health["device"], "cuda")
        self.assertIn("CUDA unavailable", health["engine_error"])

    def test_explicit_cpu_fallback_is_logged(self):
        service = STTService(STTConfig(model_size="medium", device="cuda", compute_type="int8_float16", explicit_cpu_fallback=True))
        calls = []
        def factory(model, device, compute_type):
            calls.append((model, device, compute_type))
            if device == "cuda":
                raise RuntimeError("CUDA unavailable")
            return FakeModel()
        out = io.StringIO()
        with patch("app.services.stt_whisper.WhisperModel", side_effect=factory), redirect_stdout(out):
            service.init()
        self.assertEqual(calls, [("medium", "cuda", "int8_float16"), ("medium", "cpu", "int8")])
        self.assertEqual(service.engine_status, "fallback")
        self.assertIn("status=fallback", out.getvalue())

    def test_ambient_mode_has_no_prompt_and_auto_language(self):
        service = self.service(text="This is English", language="en")
        _, metadata = service._transcribe_audio(np.zeros(1600, dtype=np.float32), mode=STTMode.AMBIENT_DISCOURSE)
        call = service._model.calls[-1]
        self.assertIsNone(call["language"])
        self.assertIsNone(call["initial_prompt"])
        self.assertEqual(metadata["detected_language"], "en")
        self.assertFalse(metadata["action_eligible"])

    def test_direct_command_uses_bounded_prompt(self):
        service = self.service()
        service._transcribe_audio(np.zeros(1600, dtype=np.float32), mode=STTMode.DIRECT_COMMAND)
        call = service._model.calls[-1]
        self.assertTrue(call["initial_prompt"])
        self.assertLessEqual(len(call["initial_prompt"]), service.cfg.command_prompt_max_chars)

    def test_pending_answer_uses_expected_candidates_only(self):
        service = self.service(text="Nuria")
        service._transcribe_audio(
            np.zeros(1600, dtype=np.float32), mode=STTMode.PENDING_SHORT_ANSWER,
            expected_vocabulary=["Nuria", "Muria"],
        )
        self.assertEqual(service._model.calls[-1]["initial_prompt"], "Nuria, Muria.")

    def test_performance_and_rolling_metrics(self):
        service = self.service()
        out = io.StringIO()
        with redirect_stdout(out):
            service._transcribe_audio(np.zeros(16000, dtype=np.float32), mode=STTMode.DIAGNOSTIC)
        self.assertIn("[HEBE][STT_PERF]", out.getvalue())
        health = service.health_snapshot()
        self.assertGreaterEqual(health["p95_latency_seconds"], 0)
        self.assertEqual(health["transcriptions_per_minute"], 1)

    def test_partial_hebe_obs_clarifies_not_prompt_echo(self):
        service = self.service()
        hypothesis = service._build_command_hypothesis(
            "Hebe, Ebe, OBS", "Hebe, OBS", {"detected_language": "es"},
            {"detected_language": "es", "language_allowed": True},
        )
        self.assertEqual(hypothesis.decision, "clarify")
        self.assertFalse(hypothesis.action_eligible)
        self.assertFalse(is_stt_prompt_hotword_list("Hebe, Ebe, OBS"))
        self.assertTrue(is_stt_prompt_hotword_list("Hebe, OBS, Twitch, stream, promo"))

    def test_clear_second_pass_can_execute(self):
        service = self.service()
        hypothesis = service._build_command_hypothesis(
            "Y bien, abre OBS", "Abre OBS", {"detected_language": "es"},
            {"detected_language": "es", "language_allowed": True},
        )
        self.assertEqual(hypothesis.decision, "execute")
        self.assertTrue(hypothesis.action_eligible)

    def test_tts_echo_and_tail(self):
        service = self.service()
        service.set_tts_playback(True, "Ya estoy aqui Leo")
        self.assertTrue(service._tts_echo_decision("Ya estoy aqui Leo")[0])
        service.set_tts_playback(False, "Ya estoy aqui Leo")
        service._tts_ended_at = time.time() - 2
        self.assertFalse(service._tts_echo_decision("Leo abre OBS ahora")[0])

    def test_hallucination_loops_rejected_without_short_overblock(self):
        service = self.service()
        self.assertTrue(service._hallucination_reason("abril " * 30))
        self.assertTrue(service._hallucination_reason("volver a " * 20))
        self.assertTrue(service._hallucination_reason("ja " * 40))
        self.assertEqual(service._hallucination_reason("no no espera"), "")


if __name__ == "__main__":
    unittest.main()
