import threading
import time
import unittest

from app.core.stt_worker import STTWorker
from app.services.stt_whisper import STTConfig, STTDeviceOpenFailure, STTService, _sort_by_host_api_preference


class FailingSTT:
    def __init__(self):
        self.calls = 0

    def listen(self):
        self.calls += 1
        raise STTDeviceOpenFailure("OSError(-9999, 'Unanticipated host error')")


class STTDeviceFailureTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
