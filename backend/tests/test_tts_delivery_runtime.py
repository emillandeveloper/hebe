import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from app.cognitive.final_emission_gate import FinalEmissionGate, OutputRoute
from app.core import runtime as runtime_module
from app.hebe_engine import HebeEngine
from app.services import speech_output
from app.services import tts_piper, tts_service
from app.services.speech_output import SpeechOutputController, TTSCancelled, TTSPlaybackTimeout
from app.services.stream_tts_guard import StreamTTSSafetyManager
from app.services.tts_worker import (
    SynthesisReceipt,
    TTSSynthesisTimeout,
    TTSSynthesisWorker,
    TTSWarmupInProgress,
)
from app.stream.runtime_context import HebeLiveContextPolicy


class MultiTargetDeliveryTests(unittest.TestCase):
    def emit(self, *, twitch=True, tts_status="tts_delivered"):
        gate = FinalEmissionGate()
        return gate.emit(
            event_id="delivery",
            source="twitch",
            final_response="respuesta",
            output_route=OutputRoute.TWITCH_TEXT_REPLY,
            output_targets=["twitch_chat", "stream_tts"],
            guard_result={"passed": True},
            debug_payload={"response_stage": "final"},
            send_twitch=lambda _text: twitch,
            speak=lambda _text: {"status": tts_status, "trace_id": "delivery"},
            logger=lambda _line: None,
        )

    def test_twitch_and_tts_delivered(self):
        result = self.emit()
        self.assertTrue(result.emitted)
        self.assertEqual(result.outcome, "delivered")
        self.assertEqual(result.target_receipts["twitch_chat"]["status"], "delivered")
        self.assertEqual(result.target_receipts["stream_tts"]["status"], "tts_delivered")

    def test_twitch_delivered_and_tts_timeout_is_partial(self):
        result = self.emit(tts_status="tts_timed_out")
        self.assertTrue(result.emitted)
        self.assertEqual(result.outcome, "partial_delivery")
        self.assertEqual(result.target_receipts["stream_tts"]["status"], "tts_timed_out")

    def test_twitch_delivered_and_tts_failure_is_partial(self):
        result = self.emit(tts_status="tts_failed")
        self.assertTrue(result.emitted)
        self.assertEqual(result.outcome, "partial_delivery")

    def test_twitch_failure_and_tts_delivery_is_partial(self):
        result = self.emit(twitch=False)
        self.assertTrue(result.emitted)
        self.assertEqual(result.outcome, "partial_delivery")
        self.assertEqual(result.target_receipts["twitch_chat"]["status"], "failed")

    def test_both_target_failures_are_response_failure(self):
        result = self.emit(twitch=False, tts_status="tts_failed")
        self.assertFalse(result.emitted)
        self.assertEqual(result.outcome, "response_failed")


class TTSQueueTests(unittest.TestCase):
    def manager(self, **kwargs):
        manager = StreamTTSSafetyManager(**kwargs)
        manager.min_free_vram_mb = 0
        manager.warn_seconds = 100
        return manager

    def test_normal_tts_is_delivered(self):
        manager = self.manager()
        result = manager.schedule("hola", lambda _text: {"status": "tts_delivered"}, trace_id="normal")
        receipt = manager.wait("normal", timeout_seconds=1)
        self.assertTrue(result["scheduled"])
        self.assertEqual(receipt["status"], "tts_delivered")
        manager.shutdown()

    def test_synthesis_timeout_is_observable_and_job_finishes(self):
        manager = self.manager()

        def timeout(_text):
            raise TTSSynthesisTimeout("slow provider")

        manager.schedule("hola", timeout, trace_id="timeout")
        receipt = manager.wait("timeout", timeout_seconds=1)
        self.assertEqual(receipt["status"], "tts_timed_out")
        self.assertEqual(receipt["stage"], "synthesis")
        self.assertEqual(manager.current_gpu_task, "")
        self.assertEqual(manager.queue_depth, 0)
        self.assertTrue(manager.shutdown()["stopped"])

    def test_provider_failure_isolated(self):
        manager = self.manager()
        manager.schedule("hola", lambda _text: (_ for _ in ()).throw(RuntimeError("provider")), trace_id="failed")
        self.assertEqual(manager.wait("failed", timeout_seconds=1)["status"], "tts_failed")
        manager.shutdown()

    def test_request_during_warmup_is_cancelled_without_breaking_warmup(self):
        manager = self.manager()
        manager.schedule(
            "hola",
            lambda _text: (_ for _ in ()).throw(TTSWarmupInProgress("warming")),
            trace_id="warming",
            optional=False,
        )
        receipt = manager.wait("warming", timeout_seconds=1)
        self.assertEqual(receipt["status"], "tts_cancelled")
        self.assertEqual(receipt["stage"], "warmup")
        self.assertEqual(receipt["reason"], "warmup_in_progress")
        manager.shutdown()

    def test_playback_latency_does_not_open_synthesis_circuit(self):
        manager = self.manager()
        manager.warn_seconds = 0.001
        manager.slow_limit = 1
        manager.schedule(
            "hola",
            lambda _text: {
                "status": "tts_delivered",
                "synthesis_ms": 0.1,
                "playback_ms": 20_000,
            },
            trace_id="long-playback",
        )
        self.assertEqual(manager.wait("long-playback", timeout_seconds=1)["status"], "tts_delivered")
        self.assertEqual(manager.readiness()["circuit_state"], "closed")
        manager.shutdown()

    def test_successful_warmup_rearms_circuit(self):
        manager = self.manager()
        manager._slow_count = manager.slow_limit
        manager._open_until = time.time() + 300
        manager.warmup(lambda: {"status": "ready"})
        self.assertEqual(manager.readiness()["circuit_state"], "closed")
        self.assertEqual(manager._slow_count, 0)

    def test_queue_is_bounded_and_supersedes_optional_pending_speech(self):
        started = threading.Event()
        release = threading.Event()
        manager = self.manager()
        manager.max_queue_size = 2

        def blocking(_text):
            started.set()
            release.wait(1)
            return {"status": "tts_delivered"}

        manager.schedule("active", blocking, trace_id="active")
        self.assertTrue(started.wait(0.5))
        manager.schedule("old-1", lambda _text: None, trace_id="old-1")
        manager.schedule("old-2", lambda _text: None, trace_id="old-2")
        manager.schedule("new", lambda _text: None, trace_id="new")
        self.assertLessEqual(manager.queue_depth, 2)
        self.assertEqual(manager.wait("old-1", timeout_seconds=0.1)["status"], "tts_dropped_stale")
        release.set()
        manager.shutdown(timeout_seconds=1)

    def test_stale_optional_speech_is_dropped_before_playback(self):
        started = threading.Event()
        release = threading.Event()
        manager = self.manager()

        def blocking(_text):
            started.set()
            release.wait(1)

        manager.schedule("active", blocking, trace_id="active")
        self.assertTrue(started.wait(0.5))
        manager.schedule("stale", lambda _text: None, trace_id="stale", stale_after_seconds=0.01)
        time.sleep(0.02)
        manager.schedule("current", lambda _text: None, trace_id="current")
        self.assertEqual(manager.wait("stale", timeout_seconds=0.1)["status"], "tts_dropped_stale")
        release.set()
        manager.shutdown(timeout_seconds=1)

    def test_direct_speech_preempts_active_optional_speech(self):
        cancelled = threading.Event()
        active_started = threading.Event()

        def cancel_active():
            cancelled.set()

        manager = self.manager(cancel_active=cancel_active)

        def optional(_text):
            active_started.set()
            self.assertTrue(cancelled.wait(1))
            raise TTSCancelled("superseded")

        manager.schedule("optional", optional, trace_id="optional")
        self.assertTrue(active_started.wait(0.5))
        manager.schedule(
            "direct",
            lambda _text: {"status": "tts_delivered"},
            trace_id="direct",
            priority="direct",
            optional=False,
        )
        self.assertEqual(manager.wait("direct", timeout_seconds=1)["status"], "tts_delivered")
        self.assertEqual(manager.wait("optional", timeout_seconds=1)["status"], "tts_cancelled")
        manager.shutdown()

    def test_shutdown_cancels_active_tts_with_bounded_wait(self):
        cancelled = threading.Event()
        started = threading.Event()
        manager = self.manager(cancel_active=cancelled.set)

        def active(_text):
            started.set()
            cancelled.wait(1)
            raise TTSCancelled("shutdown")

        manager.schedule("active", active, trace_id="shutdown")
        self.assertTrue(started.wait(0.5))
        before = time.perf_counter()
        result = manager.shutdown(timeout_seconds=0.5)
        self.assertLess(time.perf_counter() - before, 0.6)
        self.assertTrue(result["stopped"])
        self.assertEqual(manager.wait("shutdown", timeout_seconds=0.1)["status"], "tts_cancelled")

    def test_disabled_tts_creates_no_worker(self):
        manager = self.manager()
        result = manager.schedule("disabled", Mock(), output_enabled=False, trace_id="disabled")
        self.assertFalse(result["scheduled"])
        self.assertFalse(manager.worker_alive)
        self.assertEqual(manager.warmup_status, "not_run")

    def test_engine_emits_ui_and_returns_without_waiting_for_tts(self):
        release = threading.Event()
        started = threading.Event()
        manager = self.manager(cancel_active=release.set)
        engine = HebeEngine.__new__(HebeEngine)
        engine.runtime = SimpleNamespace(
            state=SimpleNamespace(tts_enabled=True, stream=None),
            tts=SimpleNamespace(cancel=release.set),
        )
        engine.stream_tts_safety = manager
        engine.final_emission_gate = FinalEmissionGate()
        engine.live_context_policy = HebeLiveContextPolicy()
        engine.interaction_decision_history = SimpleNamespace(update=lambda *_args, **_kwargs: None)

        def slow_speak(_text):
            started.set()
            release.wait(1)
            return {"status": "tts_delivered"}

        visible = []
        with patch("app.hebe_engine.emit", lambda event_type, payload=None: visible.append((event_type, payload or {}))):
            before = time.perf_counter()
            result = engine._emit_final_response(
                event_id="direct-nonblocking",
                source="direct_stt",
                final_response="respuesta",
                output_route=OutputRoute.LOCAL_OWNER_REPLY,
                output_targets=["local_ui", "local_tts"],
                guard_result={"passed": True},
                speak_fn=slow_speak,
            )
            elapsed = time.perf_counter() - before
        self.assertLess(elapsed, 0.2)
        self.assertEqual([item[0] for item in visible].count("chat.assistant"), 1)
        self.assertTrue(result["emitted"])
        self.assertIn(result["target_receipts"]["local_tts"]["status"], {"tts_queued", "tts_started"})
        self.assertTrue(started.wait(0.5))
        release.set()
        self.assertEqual(manager.wait("direct-nonblocking", timeout_seconds=1)["status"], "tts_delivered")
        self.assertTrue(manager.shutdown()["stopped"])


class TTSDeadlineTests(unittest.TestCase):
    def test_stt_echo_guard_tracks_playback_not_synthesis(self):
        state = SimpleNamespace(tts_enabled=True)
        stt = Mock()

        def fake_speak(**kwargs):
            stt.set_tts_playback.assert_not_called()
            kwargs["on_playback_state"](True)
            kwargs["on_playback_state"](False)
            return {"status": "tts_delivered"}

        with (
            patch.object(runtime_module, "_speak", side_effect=fake_speak),
            patch.object(runtime_module, "log_jsonl_event"),
        ):
            result = runtime_module.build_speak(state, stt)("Hola.")

        self.assertEqual(result["status"], "tts_delivered")
        self.assertEqual(stt.set_tts_playback.call_args_list, [
            call(True, "Hola."),
            call(False, "Hola."),
        ])

    def test_cold_background_warmup_is_not_active_speech_and_returns_idle(self):
        controller = SpeechOutputController()
        started = threading.Event()
        release = threading.Event()

        def blocked_warmup(timeout_seconds=None):
            started.set()
            release.wait(1)
            return SynthesisReceipt("fake", 1)

        engine = HebeEngine.__new__(HebeEngine)
        engine.runtime = SimpleNamespace(tts=controller)
        engine._tts_active = True
        engine._last_tts_activity_state = "TTS_SYNTHESIZING"
        with patch.object(speech_output, "warmup_synthesis", side_effect=blocked_warmup):
            thread = threading.Thread(target=controller.warmup)
            thread.start()
            self.assertTrue(started.wait(0.5))
            self.assertEqual(controller.activity_state, "TTS_WARMING")
            self.assertFalse(controller.is_speaking)
            self.assertFalse(engine._is_tts_active())
            release.set()
            thread.join(1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(controller.activity_state, "TTS_IDLE")
        self.assertFalse(engine._is_tts_active())

    def test_synthesis_and_playback_publish_real_activity_then_return_idle(self):
        controller = SpeechOutputController()
        synthesis_started = threading.Event()
        release_synthesis = threading.Event()
        playback_started = threading.Event()
        release_playback = threading.Event()
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp.close()

        def synthesize(*_args, **_kwargs):
            synthesis_started.set()
            release_synthesis.wait(1)
            return temp.name, SynthesisReceipt("fake", 1)

        music = Mock()
        music.get_busy.side_effect = lambda: not release_playback.is_set()
        mixer = Mock()
        mixer.get_init.return_value = True
        mixer.music = music
        playback_states = []
        result = {}

        def record_playback_state(active):
            playback_states.append(active)
            if active:
                playback_started.set()

        with (
            patch.object(speech_output, "tts_to_wav", side_effect=synthesize),
            patch.object(speech_output, "pygame", SimplePygame(mixer)),
            patch.object(controller, "_wav_duration", return_value=1.0),
        ):
            thread = threading.Thread(
                target=lambda: result.update(controller.speak("Hola.", on_playback_state=record_playback_state))
            )
            thread.start()
            self.assertTrue(synthesis_started.wait(0.5))
            self.assertEqual(controller.activity_state, "TTS_SYNTHESIZING")
            self.assertTrue(controller.is_speaking)
            release_synthesis.set()
            self.assertTrue(playback_started.wait(0.5))
            self.assertEqual(controller.activity_state, "TTS_PLAYING")
            self.assertTrue(controller.is_playing)
            release_playback.set()
            thread.join(1)

        self.assertEqual(result["status"], "tts_delivered")
        self.assertEqual(playback_states, [True, False])
        self.assertEqual(controller.activity_state, "TTS_IDLE")
        self.assertFalse(controller.is_speaking)

    def test_failed_or_cancelled_synthesis_cannot_leave_tts_active(self):
        for error in (TTSSynthesisTimeout("timeout"), TTSCancelled("cancelled"), RuntimeError("failed")):
            with self.subTest(error=type(error).__name__):
                controller = SpeechOutputController()
                with patch.object(speech_output, "tts_to_wav", side_effect=error):
                    with self.assertRaises(type(error)):
                        controller.speak("Hola.")
                self.assertEqual(controller.activity_state, "TTS_IDLE")
                self.assertFalse(controller.is_speaking)

    def test_tts_delivery_is_independent_of_vts_hotkeys_and_connection_lifecycle(self):
        controller = SpeechOutputController()
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp.close()
        music = Mock()
        music.get_busy.return_value = False
        mixer = Mock()
        mixer.get_init.return_value = True
        mixer.music = music
        with (
            patch.object(speech_output, "tts_to_wav", return_value=(temp.name, SynthesisReceipt("fake", 1))),
            patch.object(speech_output, "pygame", SimplePygame(mixer)),
            patch("app.services.vts_client.vts_hotkey", side_effect=RuntimeError("hotkey_not_found")) as hotkey,
            patch("app.services.vts_client.start_vts", side_effect=AssertionError("TTS must not start VTS")) as start,
        ):
            receipt = controller.speak("Vale, eso ha estado mejor.")

        self.assertEqual(receipt["status"], "tts_delivered")
        hotkey.assert_not_called()
        start.assert_not_called()
        self.assertFalse(Path(temp.name).exists())

    def test_supported_text_lengths_use_benchmarked_fixed_deadline(self):
        samples = {
            "short": "Vale, eso ha estado bastante mejor.",
            "medium": "Ahora podemos avanzar con calma, guardar los recursos importantes y revisar la siguiente sala antes de decidir.",
            "long": (
                "Esta parte parece tranquila, pero todavía quedan varias puertas cerradas y muy poca munición. "
                "Revisaría primero la sala del fondo, guardaría el recurso especial y dejaría preparada una salida "
                "antes de que vuelva la patrulla.")
        }
        generated_paths = []
        try:
            with patch.object(tts_service, "_synthesis_worker") as worker:
                worker.synthesize.return_value = SynthesisReceipt("xtts", 8_100)
                for bucket, text in samples.items():
                    with self.subTest(bucket=bucket):
                        wav_path, receipt = tts_service.speak(text)
                        generated_paths.append(Path(wav_path))
                        self.assertEqual(receipt.backend, "xtts")
                        self.assertEqual(worker.synthesize.call_args.kwargs["timeout_seconds"], 15.0)
                        worker.synthesize.reset_mock()
        finally:
            for path in generated_paths:
                path.unlink(missing_ok=True)

    def test_synthesis_request_does_not_terminate_active_warmup(self):
        worker = TTSSynthesisWorker()
        worker._current_kind = "warmup"
        with self.assertRaises(TTSWarmupInProgress):
            worker.synthesize(text="hola", wav_path="unused.wav", timeout_seconds=10)
        self.assertEqual(worker._current_kind, "warmup")

    def test_piper_subprocess_has_its_own_deadline(self):
        with (
            patch.object(tts_piper, "HEBE_PIPER_EXE", "piper.exe"),
            patch.object(tts_piper, "HEBE_PIPER_MODEL_ES", "voice.onnx"),
            patch.object(tts_piper, "HEBE_PIPER_TIMEOUT_SECONDS", 4.5),
            patch.object(tts_piper.os.path, "exists", return_value=True),
            patch.object(tts_piper.subprocess, "run") as run,
        ):
            tts_piper.piper_to_wav("hola", "out.wav")
        self.assertEqual(run.call_args.kwargs["timeout"], 4.5)

    def test_playback_deadline_stops_audio(self):
        controller = SpeechOutputController()
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp.close()
        music = Mock()
        music.get_busy.return_value = True
        mixer = Mock()
        mixer.get_init.return_value = True
        mixer.music = music
        with (
            patch.object(speech_output, "tts_to_wav", return_value=(temp.name, SynthesisReceipt("fake", 1))),
            patch.object(speech_output, "pygame", SimplePygame(mixer)),
            patch.object(speech_output, "HEBE_TTS_PLAYBACK_GRACE_SECONDS", 0.01),
            patch.object(speech_output, "HEBE_TTS_PLAYBACK_MAX_SECONDS", 0.02),
            patch.object(controller, "_wav_duration", return_value=0.0),
        ):
            with self.assertRaises(TTSPlaybackTimeout):
                controller.speak("hola")
        music.stop.assert_called_once()
        self.assertFalse(Path(temp.name).exists())

    def test_synthesis_timeout_terminates_worker_process(self):
        context = FakeContext()
        worker = TTSSynthesisWorker()
        worker._context = context
        with self.assertRaises(TTSSynthesisTimeout):
            worker.warmup(timeout_seconds=0.05)
        self.assertTrue(context.process.terminated)
        self.assertFalse(worker.is_alive)

    def test_hung_synthesis_times_out_and_terminates_worker_process(self):
        context = FakeContext()
        worker = TTSSynthesisWorker()
        worker._context = context
        with self.assertRaises(TTSSynthesisTimeout):
            worker.synthesize(
                text="Esta síntesis no responde.",
                wav_path="unused.wav",
                timeout_seconds=0.05,
            )
        self.assertTrue(context.process.terminated)
        self.assertFalse(worker.is_alive)


class SimplePygame:
    def __init__(self, mixer):
        self.mixer = mixer


class FakeQueue:
    def put(self, _value, timeout=None):
        return None

    def put_nowait(self, _value):
        return None

    def get(self, timeout=None):
        raise queue.Empty

    def close(self):
        return None

    def cancel_join_thread(self):
        return None


class FakeProcess:
    def __init__(self):
        self.alive = False
        self.terminated = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True
        self.alive = False

    def join(self, timeout=None):
        return None


class FakeContext:
    def __init__(self):
        self.process = FakeProcess()

    def Queue(self, maxsize=0):
        return FakeQueue()

    def Process(self, **_kwargs):
        return self.process


if __name__ == "__main__":
    unittest.main()
