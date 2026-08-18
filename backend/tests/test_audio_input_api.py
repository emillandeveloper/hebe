import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.audio import router
from app.services import db_sqlite


class FakeAdapter:
    def __init__(self):
        self.calls = []

    async def set_audio_input_device(self, **kwargs) -> bool:
        self.calls.append(kwargs)
        return True


class AudioInputApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "hebe.db"
        sqlite3.connect(self.db_path).close()
        self.adapter = FakeAdapter()
        app = FastAPI()
        app.state.adapter = self.adapter
        app.include_router(router)
        self.client = TestClient(app)
        self.patch_db = patch.object(db_sqlite, "DB_PATH", str(self.db_path))
        self.patch_db.start()

    def tearDown(self):
        self.patch_db.stop()
        self.tmp.cleanup()

    def test_lists_audio_input_devices(self):
        devices = [{
            "id": "2",
            "index": 2,
            "name": "Mic Test",
            "host_api": "WASAPI",
            "is_default": True,
            "is_default_input": True,
            "is_loopback": False,
            "channels": 2,
            "sample_rate": 48000,
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_sample_rate": 48000,
            "signature": "mic test|wasapi|48000|2",
            "display_label": "Mic Test — WASAPI — id 2 — 48000Hz — 2ch",
        }]
        with patch("app.api.audio.list_audio_devices", return_value=devices):
            res = self.client.get("/audio/input-devices")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["devices"], devices)
        self.assertIn("WASAPI", res.json()["devices"][0]["display_label"])

    def test_selected_microphone_is_persisted_and_applied(self):
        device = {
            "id": "3", "index": 3, "name": "GoXLR Mic", "host_api": "WASAPI",
            "default_sample_rate": 48000, "max_input_channels": 1,
            "signature": "goxlr mic|wasapi|48000|1", "is_default_input": False,
        }
        with patch("app.api.audio.list_audio_devices", return_value=[device]):
            res = self.client.post("/audio/input-device", json={
                "device_id": "3",
                "device_name": "GoXLR Mic",
                "host_api": "WASAPI",
                "sample_rate": 48000,
                "channels": 1,
                "signature": "goxlr mic|wasapi|48000|1",
            })

        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["ok"])
        self.assertEqual(self.adapter.calls, [{
            "device_id": "3",
            "device_name": "GoXLR Mic",
            "host_api": "WASAPI",
            "sample_rate": 48000,
            "channels": 1,
            "signature": "goxlr mic|wasapi|48000|1",
        }])

        current = self.client.get("/audio/input-device")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["device_id"], "3")
        self.assertEqual(current.json()["device_name"], "GoXLR Mic")
        self.assertEqual(current.json()["host_api"], "WASAPI")
        self.assertEqual(current.json()["sample_rate"], 48000)
        self.assertEqual(current.json()["channels"], 1)

    def test_reused_numeric_id_is_rebound_by_stable_identity_before_persisting(self):
        devices = [
            {
                "id": "8", "index": 8, "name": "Voicemeeter Out A4", "host_api": "MME",
                "default_sample_rate": 44100, "max_input_channels": 8,
                "signature": "voicemeeter out a4|mme|44100|8", "is_default_input": True,
            },
            {
                "id": "9", "index": 9, "name": "Yeti GX", "host_api": "MME",
                "default_sample_rate": 44100, "max_input_channels": 1,
                "signature": "yeti gx|mme|44100|1", "is_default_input": False,
            },
        ]
        with patch("app.api.audio.list_audio_devices", return_value=devices):
            response = self.client.post("/audio/input-device", json={
                "device_id": "8", "device_name": "Yeti GX", "host_api": "MME",
                "sample_rate": 44100, "channels": 1, "signature": "yeti gx|mme|44100|1",
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["device_id"], "9")
        self.assertEqual(response.json()["resolution_reason"], "stable_signature")
        self.assertEqual(self.client.get("/audio/input-device").json()["device_id"], "9")

    def test_microphone_test_returns_rms_and_peak(self):
        result = {
            "ok": True,
            "signal_detected": True,
            "rms": 0.012,
            "peak": 0.08,
            "sample_rate": 48000,
            "channels": 1,
            "device": {"display_label": "Mic Test — WASAPI — id 2 — 48000Hz — 1ch"},
        }
        with patch("app.api.audio.test_audio_input_device", return_value=result) as test_fn:
            res = self.client.post("/audio/input-device/test", json={
                "device_id": "2",
                "device_name": "Mic Test",
                "host_api": "WASAPI",
                "seconds": 3,
            })

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["rms"], 0.012)
        self.assertEqual(res.json()["peak"], 0.08)
        test_fn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
