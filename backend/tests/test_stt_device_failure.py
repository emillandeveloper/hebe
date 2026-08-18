import threading
import time
import unittest
from unittest.mock import Mock, patch

from app.core.stt_worker import STTWorker
from app.services.stt_whisper import (
    STTConfig,
    STTDeviceOpenFailure,
    STTService,
    _device_signature,
    _input_health_window_is_silent,
    _select_audio_input_device,
    _sort_by_host_api_preference,
)


class FailingSTT:
    def __init__(self):
        self.calls = 0

    def listen(self):
        self.calls += 1
        raise STTDeviceOpenFailure("OSError(-9999, 'Unanticipated host error')")


class STTDeviceFailureTests(unittest.TestCase):
    @staticmethod
    def _device(
        *, index=9, name="Micrófono (Studio USB)", host="MME",
        endpoint="{11111111-2222-3333-4444-555555555555}", rate=44100,
    ):
        return {
            "id": str(index), "index": index, "name": name, "host_api": host,
            "endpoint_id": endpoint, "default_sample_rate": rate,
            "max_input_channels": 1, "is_default_input": False,
            "signature": _device_signature(name, host, rate, 1, endpoint_id=endpoint),
        }

    def test_unicode_identity_matches_enumerated_device(self):
        device = self._device()

        selected, reason = _select_audio_input_device(
            [device], device_name="Micrófono (Studio USB)", host_api="MME",
        )

        self.assertIs(selected, device)
        self.assertEqual(reason, "stable_name_host_api")

    def test_unique_legacy_mojibake_identity_is_repaired_against_real_device(self):
        device = self._device()

        selected, reason = _select_audio_input_device(
            [device], device_name="MicrÃ³fono (Studio USB)", host_api="MME",
            signature="micrã³fono (studio usb)|mme|44100|1",
        )

        self.assertEqual(selected["name"], "Micrófono (Studio USB)")
        self.assertTrue(selected["identity_repaired"])
        self.assertEqual(selected["identity_repair_reason"], "legacy_encoding_repair")
        self.assertEqual(reason, "legacy_encoding_repair")

    def test_ambiguous_legacy_mojibake_identity_is_not_guessed(self):
        devices = [
            self._device(index=9, endpoint="{aaaaaaaa-2222-3333-4444-555555555555}"),
            self._device(index=10, endpoint="{bbbbbbbb-2222-3333-4444-555555555555}"),
        ]

        selected, reason = _select_audio_input_device(
            devices, device_name="MicrÃ³fono (Studio USB)", host_api="MME",
        )

        self.assertIsNone(selected)
        self.assertEqual(reason, "configured_identity_ambiguous")

    def test_stable_endpoint_survives_default_sample_rate_change(self):
        before = self._device(index=9, rate=44100)
        after = self._device(index=37, rate=48000)
        self.assertEqual(before["signature"], after["signature"])

        selected, reason = _select_audio_input_device(
            [after], device_index=9, device_name=before["name"], host_api="MME",
            signature=before["signature"],
        )

        self.assertEqual(selected["index"], 37)
        self.assertEqual(selected["default_sample_rate"], 48000)
        self.assertEqual(reason, "stable_endpoint")

    def test_repaired_identity_persists_as_v2_and_resolves_after_restart(self):
        initial = self._device(index=9, rate=44100)
        service = STTService(config=STTConfig())
        port_audio = Mock()
        with (
            patch("app.services.stt_whisper.pyaudio.PyAudio", return_value=port_audio),
            patch("app.services.stt_whisper._list_audio_devices_with_instance", return_value=[initial]),
        ):
            persisted = service.set_input_device(
                device_id="8", device_name="MicrÃ³fono (Studio USB)", host_api="MME",
                signature="micrã³fono (studio usb)|mme|44100|1",
            )

        restarted_device = self._device(index=41, rate=48000)
        restarted = STTService(config=STTConfig())
        restarted_port_audio = Mock()
        with (
            patch("app.services.stt_whisper.pyaudio.PyAudio", return_value=restarted_port_audio),
            patch("app.services.stt_whisper._list_audio_devices_with_instance", return_value=[restarted_device]),
        ):
            selected = restarted.set_input_device(
                device_id=persisted["device_id"], device_name=persisted["device_name"],
                host_api=persisted["host_api"], sample_rate=persisted["sample_rate"],
                channels=persisted["channels"], signature=persisted["signature"],
            )

        self.assertTrue(persisted["signature"].startswith("audio-input:v2|windows-endpoint|"))
        self.assertEqual(persisted["identity_repair"]["reason"], "legacy_encoding_repair")
        self.assertEqual(selected["device_id"], "41")
        self.assertEqual(selected["resolution_reason"], "stable_endpoint")

    def test_disappeared_stable_device_is_explicitly_unavailable(self):
        selected, reason = _select_audio_input_device(
            [], device_index=9, device_name="Micrófono (Studio USB)", host_api="MME",
            signature=self._device()["signature"],
        )

        self.assertIsNone(selected)
        self.assertEqual(reason, "configured_identity_unavailable")

    def test_health_window_requires_sustained_zero_not_one_quiet_chunk(self):
        self.assertTrue(_input_health_window_is_silent(0.0, 0.0))
        self.assertFalse(_input_health_window_is_silent(0.0001, 0.0002))

    def test_stable_identity_rebinds_after_numeric_index_changes(self):
        devices = [
            {
                "id": "8", "index": 8, "name": "Voicemeeter Out A4", "host_api": "MME",
                "signature": "voicemeeter out a4|mme|44100|8", "is_default_input": True,
            },
            {
                "id": "9", "index": 9, "name": "Micrófono (Yeti GX)", "host_api": "MME",
                "endpoint_id": "{11111111-2222-3333-4444-555555555555}",
                "signature": "audio-input:v2|windows-endpoint|{11111111-2222-3333-4444-555555555555}",
                "is_default_input": False,
            },
        ]

        selected, reason = _select_audio_input_device(
            devices,
            device_index=8,
            device_name="Micrófono (Yeti GX)",
            host_api="MME",
            signature="audio-input:v2|windows-endpoint|{11111111-2222-3333-4444-555555555555}",
        )

        self.assertEqual(selected["index"], 9)
        self.assertEqual(reason, "stable_endpoint")

    def test_applying_stable_identity_updates_current_index_and_stream_generation(self):
        devices = [{
            "id": "9", "index": 9, "name": "Micrófono (Yeti GX)", "host_api": "MME",
            "endpoint_id": "{11111111-2222-3333-4444-555555555555}",
            "signature": "audio-input:v2|windows-endpoint|{11111111-2222-3333-4444-555555555555}",
            "is_default_input": False,
            "default_sample_rate": 44100, "max_input_channels": 1,
        }]
        service = STTService(config=STTConfig())
        port_audio = Mock()
        with (
            patch("app.services.stt_whisper.pyaudio.PyAudio", return_value=port_audio),
            patch("app.services.stt_whisper._list_audio_devices_with_instance", return_value=devices),
        ):
            selected = service.set_input_device(
                device_id="8",
                device_name="Micrófono (Yeti GX)",
                host_api="MME",
                signature="audio-input:v2|windows-endpoint|{11111111-2222-3333-4444-555555555555}",
            )

        self.assertEqual(selected["device_id"], "9")
        self.assertEqual(selected["resolution_reason"], "stable_endpoint")
        self.assertEqual(service._input_device_generation, 1)
        port_audio.terminate.assert_called_once()

    def test_numeric_index_alone_is_not_trusted_or_silently_defaulted(self):
        devices = [{
            "id": "8", "index": 8, "name": "Different Device", "host_api": "MME",
            "signature": "different|mme|44100|2", "is_default_input": True,
        }]

        selected, reason = _select_audio_input_device(devices, device_index=8)

        self.assertIsNone(selected)
        self.assertEqual(reason, "configured_identity_unavailable")

    def test_missing_stable_device_uses_default_only_when_explicitly_authorized(self):
        default = {
            "id": "1", "index": 1, "name": "Default", "host_api": "MME",
            "signature": "default|mme|44100|2", "is_default_input": True,
        }
        unavailable, _ = _select_audio_input_device(
            [default], device_name="Missing Mic", host_api="MME", allow_default_fallback=False,
        )
        selected, reason = _select_audio_input_device(
            [default], device_name="Missing Mic", host_api="MME", allow_default_fallback=True,
        )

        self.assertIsNone(unavailable)
        self.assertIs(selected, default)
        self.assertEqual(reason, "authorized_default_fallback")

    def test_device_open_failure_stops_worker_loop(self):
        stt = FailingSTT()
        worker = STTWorker(stt=stt, stop_event=threading.Event())

        worker.start()
        time.sleep(0.1)

        self.assertEqual(stt.calls, 1)
        self.assertFalse(worker.is_running())

    def test_failed_device_is_not_retried_until_cleared(self):
        service = STTService(config=STTConfig())
        service.status = "error"
        service.failed_input_error = "OSError(-9999, 'Unanticipated host error')"

        with self.assertRaises(STTDeviceOpenFailure):
            service.listen()

        service.clear_device_error()
        self.assertEqual(service.status, "idle")
        self.assertIsNone(service.last_input_device_error)

    def test_wdm_ks_is_not_preferred_over_wasapi_for_duplicate_name(self):
        devices = [
            {"index": 89, "name": "Micrófono (Yeti GX)", "host_api": "Windows WDM-KS"},
            {"index": 8, "name": "Micrófono (Yeti GX)", "host_api": "Windows WASAPI"},
            {"index": 30, "name": "Micrófono (Yeti GX)", "host_api": "MME"},
        ]

        ordered = _sort_by_host_api_preference(devices)

        self.assertEqual(ordered[0]["host_api"], "Windows WASAPI")
        self.assertEqual(ordered[-1]["host_api"], "Windows WDM-KS")

    def test_stt_device_diagnostic_warns_for_output_mix(self):
        service = STTService(config=STTConfig())
        logs = []

        with unittest.mock.patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            service._log_input_device_diagnostic({"name": "Voicemeeter Out A2"})

        joined = "\n".join(logs)
        self.assertIn("[HEBE][STT_DEVICE_DIAGNOSTIC]", joined)
        self.assertIn("warning=possible_output_mix", joined)

    def test_stt_active_device_logged(self):
        service = STTService(config=STTConfig())
        logs = []

        with unittest.mock.patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            service._log_input_device_diagnostic(
                {"name": "MicrÃ³fono (Yeti GX)", "display_label": "MicrÃ³fono (Yeti GX) - WASAPI - id 8"},
                default_device={"name": "Voicemeeter Out A2", "display_label": "Voicemeeter Out A2 - WASAPI - id 3"},
            )

        joined = "\n".join(logs)
        self.assertIn("[HEBE][STT_DEVICE_ACTIVE]", joined)
        self.assertIn("actual_capture=MicrÃ³fono (Yeti GX)", joined)
        self.assertIn("default=Voicemeeter Out A2", joined)
        self.assertIn("warning=none", joined)

    def test_output_bus_warns_only_if_actual_capture(self):
        service = STTService(config=STTConfig())
        logs = []

        with unittest.mock.patch("builtins.print", lambda *args, **kwargs: logs.append(" ".join(str(arg) for arg in args))):
            service._log_input_device_diagnostic(
                {"name": "MicrÃ³fono (Yeti GX)"},
                default_device={"name": "Voicemeeter Out A2"},
            )
            service._log_input_device_diagnostic(
                {"name": "Voicemeeter Out A2"},
                default_device={"name": "MicrÃ³fono (Yeti GX)"},
            )

        active_lines = [line for line in logs if "[HEBE][STT_DEVICE_ACTIVE]" in line]
        self.assertIn("warning=none", active_lines[0])
        self.assertIn("warning=possible_output_mix", active_lines[1])


if __name__ == "__main__":
    unittest.main()
