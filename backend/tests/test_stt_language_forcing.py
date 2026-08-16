import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.services import db_sqlite
from app.services.stt_whisper import (
    STTConfig,
    STTService,
    build_stt_command_prompt,
    is_stt_prompt_injection,
)


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeWhisperModel:
    def __init__(self, text="Hebe abre OBS"):
        self.text = text
        self.calls = []

    def transcribe(
        self,
        audio,
        *,
        language=None,
        task=None,
        beam_size=None,
        vad_filter=None,
        temperature=None,
        initial_prompt=None,
        hotwords=None,
    ):
        self.calls.append(
            {
                "language": language,
                "task": task,
                "beam_size": beam_size,
                "vad_filter": vad_filter,
                "temperature": temperature,
                "initial_prompt": initial_prompt,
                "hotwords": hotwords,
            }
        )
        return [FakeSegment(self.text)], SimpleNamespace(language=language)


class STTLanguageForcingTests(unittest.TestCase):
    def test_stt_prompt_builder_deduplicates_and_limits_vocabulary(self):
        prompt = build_stt_command_prompt(
            "Hebe, Ebe, OBS, Twitch, chat, promo, shoutout, OBS, stream, chat, promo, shoutout, Zwei",
            max_chars=120,
            log=False,
        )

        self.assertEqual(prompt.count("OBS"), 1)
        self.assertEqual(prompt.count("chat"), 1)
        self.assertLessEqual(len(prompt), 120)
        self.assertIn("Hebe", prompt)
        self.assertIn("Zwei", prompt)

    def test_command_mode_stt_uses_spanish_and_transcribe_task(self):
        service = STTService(config=STTConfig())
        model = FakeWhisperModel()
        service._model = model

        text, metadata = service._transcribe_audio(np.zeros(1600, dtype=np.float32), command_mode=True)

        self.assertEqual(text, "Hebe abre OBS")
        self.assertEqual(model.calls[-1]["language"], "es")
        self.assertEqual(model.calls[-1]["task"], "transcribe")
        self.assertNotEqual(model.calls[-1]["task"], "translate")
        self.assertEqual(model.calls[-1]["temperature"], 0)
        self.assertGreater(model.calls[-1]["beam_size"], 1)
        self.assertIn("Hebe", model.calls[-1]["initial_prompt"])
        self.assertIn("OBS", model.calls[-1]["initial_prompt"])
        self.assertIn("abre", model.calls[-1]["initial_prompt"])
        self.assertNotIn("Persona", model.calls[-1]["initial_prompt"])
        self.assertEqual(metadata["task"], "transcribe")

    def test_stt_prompt_is_only_transcriber_config_not_published_as_user_text(self):
        emitted = []
        logged = []
        service = STTService(
            config=STTConfig(),
            emit=lambda event_type, data=None: emitted.append((event_type, data or {})),
            log_chat=lambda role, text, source="voice": logged.append((role, text, source)),
        )

        result = service._publish_transcript_or_reject(service.cfg.command_prompt, {"language": "es"})

        self.assertEqual(result, "")
        self.assertEqual(logged, [])
        self.assertFalse(any(event_type in {"stt.final", "chat.user"} for event_type, _ in emitted))
        self.assertTrue(any(data.get("reason") == "stt_prompt_echo_or_hotword_list" for event_type, data in emitted if event_type == "voice.command"))

    def test_rejected_prompt_echo_raw_is_hidden_from_normal_logs_by_default(self):
        emitted = []
        logs = []
        service = STTService(
            config=STTConfig(),
            emit=lambda event_type, data=None: emitted.append((event_type, data or {})),
        )

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            result = service._publish_transcript_or_reject("Hebe, Ebe, Zwei, Persona, Final Fantasy.", {"language": "es"})

        joined = "\n".join(logs)
        self.assertEqual(result, "")
        self.assertIn("[HEBE][STT][REJECTED] reason=stt_prompt_echo_or_hotword_list", joined)
        self.assertNotIn("[HEBE][STT][RAW]", joined)
        self.assertTrue(any(data.get("raw_text") for event_type, data in emitted if event_type == "voice.command"))
        self.assertFalse(any(event_type == "stt.final" for event_type, _ in emitted))

    def test_repeated_prompt_echo_disables_initial_prompt_for_session(self):
        emitted = []
        logs = []
        cfg = STTConfig()
        cfg.prompt_echo_window_seconds = 300
        cfg.prompt_echo_disable_threshold = 2
        cfg.auto_disable_prompt_on_echo = True
        service = STTService(
            config=cfg,
            emit=lambda event_type, data=None: emitted.append((event_type, data or {})),
        )
        model = FakeWhisperModel("Hebe abre OBS")
        service._model = model

        with patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            service._publish_transcript_or_reject("Hebe, Ebe, Zwei, Persona, Final Fantasy.", {"language": "es"})
            service._publish_transcript_or_reject("Xarly, Xarly, Zwei, Totodile.", {"language": "es"})
            service._transcribe_audio(np.zeros(1600, dtype=np.float32), command_mode=True)

        self.assertFalse(service.cfg.command_prompt_enabled)
        self.assertIn("[HEBE][STT][PROMPT] auto_disabled reason=repeated_prompt_echo", "\n".join(logs))
        self.assertIsNone(model.calls[-1]["initial_prompt"])
        self.assertIsNone(model.calls[-1]["hotwords"])
        self.assertTrue(any(data.get("last_rejected_stt") for event_type, data in emitted if event_type == "status"))

    def test_restricted_auto_language_does_not_leave_language_unset(self):
        cfg = STTConfig()
        cfg.restrict_auto_language = True
        cfg.force_language_for_commands = False
        cfg.default_language = "es"
        service = STTService(config=cfg)
        model = FakeWhisperModel()
        service._model = model

        service._transcribe_audio(np.zeros(1600, dtype=np.float32), command_mode=True)

        self.assertEqual(model.calls[-1]["language"], "es")

    def test_retry_last_command_transcript_forces_spanish(self):
        service = STTService(config=STTConfig())
        model = FakeWhisperModel("Hebe abre OBS")
        service._model = model
        service.last_audio_np = np.zeros(1600, dtype=np.float32)
        service.last_speech_detected = True

        retry = service.retry_last_command_transcript(language="es")

        self.assertTrue(retry["attempted"])
        self.assertEqual(retry["text"], "Hebe abre OBS")
        self.assertEqual(model.calls[-1]["language"], "es")
        self.assertEqual(model.calls[-1]["task"], "transcribe")

    def test_env_defaults_are_command_focused(self):
        with patch.dict(
            "os.environ",
            {
                "HEBE_STT_ALLOWED_LANGUAGES": "es,en",
                "HEBE_STT_DEFAULT_LANGUAGE": "es",
                "HEBE_STT_COMMAND_LANGUAGE": "es",
                "HEBE_STT_RESTRICT_AUTO_LANGUAGE": "true",
                "HEBE_STT_FORCE_LANGUAGE_FOR_COMMANDS": "true",
            },
        ):
            cfg = STTConfig()

        self.assertEqual(cfg.allowed_languages, ("es", "en"))
        self.assertEqual(cfg.default_language, "es")
        self.assertEqual(cfg.command_language, "es")
        self.assertTrue(cfg.restrict_auto_language)
        self.assertTrue(cfg.force_language_for_commands)
        self.assertEqual(cfg.task, "transcribe")

    def test_repeated_hotword_loop_is_detected_as_prompt_injection(self):
        text = "Hebe, Ebe, OBS, Twitch, chat, promo, shoutout, OBS, stream, chat, promo, shoutout"

        self.assertTrue(is_stt_prompt_injection(text))
        self.assertTrue(is_stt_prompt_injection(build_stt_command_prompt(log=False)))

    def test_valid_transcript_is_not_prompt_injection(self):
        self.assertFalse(is_stt_prompt_injection("Hebe abre OBS"))

    def test_prompt_injection_is_not_stored_in_chat_log_cleanup_removes_existing_rows(self):
        original = db_sqlite.DB_PATH
        with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as tmp:
            db_sqlite.DB_PATH = tmp.name
        try:
            db_sqlite.init_db()
            bad = "Hebe, Ebe, OBS, Twitch, chat, promo, shoutout, OBS, stream, chat, promo, shoutout"
            db_sqlite.log_chat("user", bad, source="voice")
            db_sqlite.log_chat("user", "Hebe abre OBS", source="voice")

            deleted = db_sqlite.cleanup_stt_prompt_injection_rows()
            rows = db_sqlite.get_recent_chat_log(source="voice", limit=10)

            self.assertGreaterEqual(deleted.get("chat_log", 0), 1)
            self.assertTrue(any(row["text"] == "Hebe abre OBS" for row in rows))
            self.assertFalse(any("OBS, Twitch, chat, promo, shoutout, OBS" in row["text"] for row in rows))
        finally:
            db_sqlite.DB_PATH = original


if __name__ == "__main__":
    unittest.main()
