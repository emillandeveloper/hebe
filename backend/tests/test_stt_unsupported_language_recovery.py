from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.hebe_engine import HebeEngine
from app.core.stt_worker import STTWorker
from app.services.stt_whisper import STTConfig, STTMode, STTService


class _Segment:
    def __init__(
        self, text: str, *, avg_logprob: float = -0.15,
        no_speech_prob: float = 0.02, compression_ratio: float = 1.05,
    ):
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.compression_ratio = compression_ratio


class _LanguageModel:
    def __init__(self, results: dict[str | None, tuple[str, str, float, float]]):
        self.results = results
        self.calls: list[dict] = []

    def transcribe(
        self, _audio, *, language=None, task=None, beam_size=None,
        vad_filter=None, temperature=None, initial_prompt=None, hotwords=None,
    ):
        self.calls.append({
            "language": language,
            "initial_prompt": initial_prompt,
            "hotwords": hotwords,
        })
        text, detected, avg_logprob, no_speech = self.results[language]
        segments = [] if not text else [
            _Segment(text, avg_logprob=avg_logprob, no_speech_prob=no_speech),
        ]
        return segments, SimpleNamespace(
            language=detected,
            language_probability=0.98,
        )


def _service(results) -> STTService:
    cfg = STTConfig(model_size="medium", device="cpu", compute_type="int8")
    cfg.language_recovery_min_score = 0.62
    cfg.language_recovery_min_margin = 0.06
    service = STTService(cfg)
    service._model = _LanguageModel(results)
    service.engine_status = "ready"
    service.effective_device = "cpu"
    service.effective_compute_type = "int8"
    return service


class UnsupportedLanguageRecoveryTests(unittest.TestCase):
    def _unsupported_engine(self, language: str = "el"):
        engine = object.__new__(HebeEngine)
        engine._normalize_stt_input = Mock(side_effect=AssertionError("wake/normalization reached"))
        engine._record_voice_event = Mock(side_effect=AssertionError("memory reached"))
        engine.handle_command = Mock(side_effect=AssertionError("action reached"))
        with patch("app.hebe_engine.log_jsonl_event"):
            result = engine._process_stt_voice_transcript(
                "Hebe abre OBS",
                stt_metadata={
                    "detected_language": language,
                    "language_allowed": False,
                    "language_recovery": {
                        "accepted": False,
                        "initial_language": language,
                        "selected_language": "",
                    },
                },
            )
        self.assertEqual(result, "continue")
        return engine

    def test_russian_detection_does_not_trigger_command_prompt(self):
        service = _service({
            None: ("Привет мир", "ru", -0.1, 0.02),
            "es": ("Hebe abre OBS", "es", -1.6, 0.65),
            "en": ("Hello world", "en", -1.5, 0.6),
        })
        audio = np.ones(48000, dtype=np.float32)
        initial, metadata = service._transcribe_audio(
            audio, mode=STTMode.AMBIENT_DISCOURSE, force_prompt=False,
        )
        recovery = service._dual_decode_language_recovery(
            audio, initial_language=metadata["detected_language"], initial_text=initial,
        )
        self.assertFalse(recovery["accepted"])
        self.assertTrue(all(call["initial_prompt"] is None for call in service._model.calls))
        self.assertTrue(all(call["hotwords"] is None for call in service._model.calls))

    def test_greek_detection_does_not_trigger_command_prompt(self):
        service = _service({
            None: ("Για στο γιακίλι λέω.", "el", -0.1, 0.01),
            "es": ("Ya estoy aquí, Leo.", "es", -1.7, 0.7),
            "en": ("I am here.", "en", -1.65, 0.68),
        })
        audio = np.ones(48000, dtype=np.float32)
        _, initial_meta = service._transcribe_audio(audio, mode=STTMode.AMBIENT_DISCOURSE)
        recovery = service._dual_decode_language_recovery(
            audio, initial_language=initial_meta["detected_language"],
        )
        self.assertFalse(recovery["accepted"])
        self.assertNotIn(STTMode.DIRECT_COMMAND.value, [
            service.last_transcription_options.get("mode"),
        ])
        self.assertTrue(all(call["initial_prompt"] is None for call in service._model.calls))

    def test_prompt_vocabulary_does_not_win_unsupported_recovery(self):
        service = _service({
            "es": ("Hebe abre OBS", "es", -0.1, 0.01),
            "en": ("Hebe abre OBS", "en", -0.1, 0.01),
        })
        result = service._dual_decode_language_recovery(
            np.ones(24000, dtype=np.float32), initial_language="ja",
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["selected_text"], "")

    def test_unsupported_language_is_dropped_before_engine_routing(self):
        for language in ("ru", "el", "ja", "hi", "it"):
            with self.subTest(language=language):
                engine = self._unsupported_engine(language)
                engine._normalize_stt_input.assert_not_called()
                engine._record_voice_event.assert_not_called()
                engine.handle_command.assert_not_called()

    def test_unsupported_language_cannot_create_wake(self):
        self._unsupported_engine("ja")._normalize_stt_input.assert_not_called()

    def test_unsupported_language_cannot_execute_action(self):
        self._unsupported_engine("ru").handle_command.assert_not_called()

    def test_unsupported_language_cannot_update_memory(self):
        self._unsupported_engine("hi")._record_voice_event.assert_not_called()

    def test_short_spanish_command_selected_as_spanish(self):
        service = _service({
            "es": ("abre OBS", "es", -0.08, 0.01),
            "en": ("open obvious", "en", -1.45, 0.55),
        })
        result = service._dual_decode_language_recovery(
            np.ones(16000, dtype=np.float32),
            initial_language="es",
            short_audio=True,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["selected_language"], "es")
        self.assertEqual(result["selected_text"], "abre OBS")

    def test_short_english_phrase_selected_as_english(self):
        service = _service({
            "es": ("sí", "es", -1.5, 0.6),
            "en": ("yes", "en", -0.05, 0.01),
        })
        result = service._dual_decode_language_recovery(
            np.ones(16000, dtype=np.float32),
            initial_language="en",
            short_audio=True,
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["selected_language"], "en")
        self.assertEqual(result["selected_text"], "yes")

    def test_short_noise_not_interpreted_as_foreign_command(self):
        service = _service({
            "es": ("", "es", -2.0, 0.98),
            "en": ("Hebe", "en", -1.8, 0.9),
        })
        result = service._dual_decode_language_recovery(
            np.zeros(8000, dtype=np.float32),
            initial_language="hi",
            short_audio=True,
        )
        self.assertFalse(result["accepted"])

    def test_failed_recovery_never_publishes_stt_final(self):
        emitted = []
        service = STTService(
            STTConfig(),
            emit=lambda event_type, data=None: emitted.append((event_type, data or {})),
        )
        result = service._publish_transcript_or_reject("", {
            "detected_language": "el",
            "language_allowed": False,
            "rejection_reason": "unsupported_language_recovery_failed",
            "language_recovery": {
                "accepted": False,
                "initial_language": "el",
                "selected_language": "",
            },
        })
        self.assertEqual(result, "")
        self.assertFalse(any(event == "stt.final" for event, _ in emitted))
        self.assertTrue(any(
            data.get("reason") == "unsupported_language_recovery_failed"
            for event, data in emitted if event == "voice.command"
        ))

    def test_worker_refuses_unsupported_metadata_even_with_text(self):
        self.assertFalse(STTWorker._language_metadata_allows_submission({
            "detected_language": "ru",
            "language_allowed": False,
            "language_recovery": {"accepted": False},
        }))
        self.assertTrue(STTWorker._language_metadata_allows_submission({
            "detected_language": "en",
            "language_allowed": True,
        }))


if __name__ == "__main__":
    unittest.main()
