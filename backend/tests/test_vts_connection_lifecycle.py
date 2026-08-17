import asyncio
import threading
import time
import unittest

from app.services.vts_client import (
    VTSAuthError,
    VTSConnectionManager,
    VTSConnectionState,
    VTSProtocolError,
)


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class FakeVTSClient:
    def __init__(self, factory, connect_result=None):
        self.factory = factory
        self.connect_result = connect_result
        self.closed = threading.Event()
        self.triggered = []
        self.close_count = 0

    async def connect(self):
        self.factory.attempt_times.append(time.monotonic())
        if isinstance(self.connect_result, BaseException):
            raise self.connect_result
        if self.connect_result == "wait":
            while not self.closed.is_set():
                await asyncio.sleep(0.005)

    async def wait_closed(self):
        while not self.closed.is_set():
            await asyncio.sleep(0.005)

    async def trigger_hotkey(self, hotkey_name):
        self.triggered.append(hotkey_name)
        return True

    async def close(self):
        self.close_count += 1
        self.closed.set()

    def drop(self):
        self.closed.set()


class FakeVTSFactory:
    def __init__(self, results=None, default=None):
        self.results = list(results or [])
        self.default = default
        self.clients = []
        self.attempt_times = []

    def __call__(self):
        result = self.results.pop(0) if self.results else self.default
        client = FakeVTSClient(self, result)
        self.clients.append(client)
        return client


class VTSConnectionLifecycleTests(unittest.TestCase):
    def manager(self, factory, *, logs=None, enabled=True, backoff_min=0.02, backoff_max=0.08):
        return VTSConnectionManager(
            enabled=enabled,
            client_factory=factory,
            backoff_min_seconds=backoff_min,
            backoff_max_seconds=backoff_max,
            action_ttl_seconds=0.05,
            logger=(logs.append if logs is not None else lambda _line: None),
        )

    def test_available_at_startup_connects_in_background(self):
        factory = FakeVTSFactory()
        manager = self.manager(factory)
        started = time.perf_counter()
        manager.start()
        self.assertLess(time.perf_counter() - started, 0.05)
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "CONNECTED"))
        self.assertEqual(manager.status()["attempt_count"], 0)
        self.assertTrue(manager.shutdown()["stopped"])

    def test_connected_action_uses_existing_connection(self):
        factory = FakeVTSFactory()
        manager = self.manager(factory)
        manager.start()
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "CONNECTED"))
        attempts = manager.status()["total_attempt_count"]
        self.assertTrue(manager.trigger_hotkey("ConfiguredExpression"))
        self.assertTrue(wait_until(lambda: factory.clients[0].triggered == ["ConfiguredExpression"]))
        self.assertEqual(manager.status()["total_attempt_count"], attempts)
        self.assertEqual(manager.status()["action_delivered_count"], 1)
        manager.shutdown()

    def test_closed_at_startup_enters_backoff_without_blocking(self):
        factory = FakeVTSFactory(default=ConnectionRefusedError("closed"))
        manager = self.manager(factory, backoff_min=0.2, backoff_max=0.2)
        started = time.perf_counter()
        manager.start()
        self.assertLess(time.perf_counter() - started, 0.05)
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "BACKOFF"))
        self.assertEqual(manager.status()["last_error_kind"], "connection_refused")
        self.assertTrue(manager.shutdown()["stopped"])

    def test_repeated_refusals_use_exponential_backoff_and_deduplicate_error(self):
        logs = []
        factory = FakeVTSFactory(default=ConnectionRefusedError("same refusal"))
        manager = self.manager(factory, logs=logs, backoff_min=0.02, backoff_max=0.08)
        manager.start()
        self.assertTrue(wait_until(lambda: len(factory.attempt_times) >= 4))
        manager.shutdown()
        gaps = [
            factory.attempt_times[index + 1] - factory.attempt_times[index]
            for index in range(3)
        ]
        self.assertGreaterEqual(gaps[0], 0.015)
        self.assertGreaterEqual(gaps[1], 0.035)
        self.assertGreaterEqual(gaps[2], 0.07)
        self.assertEqual(sum("event=vts_unavailable" in line for line in logs), 1)
        self.assertGreaterEqual(sum("event=vts_backoff" in line for line in logs), 3)

    def test_vts_appearing_later_connects_and_resets_backoff(self):
        factory = FakeVTSFactory(
            results=[ConnectionRefusedError("closed"), ConnectionRefusedError("closed"), None],
            default=None,
        )
        manager = self.manager(factory)
        manager.start()
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "CONNECTED"))
        status = manager.status()
        self.assertEqual(status["total_attempt_count"], 3)
        self.assertEqual(status["attempt_count"], 0)
        self.assertEqual(status["next_retry_at"], 0.0)
        manager.shutdown()

    def test_connection_drop_is_observable_and_reconnects(self):
        logs = []
        factory = FakeVTSFactory(default=None)
        manager = self.manager(factory, logs=logs)
        manager.start()
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "CONNECTED"))
        factory.clients[0].drop()
        self.assertTrue(wait_until(lambda: len(factory.clients) >= 2))
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "CONNECTED"))
        self.assertTrue(any("event=vts_disconnected" in line for line in logs))
        self.assertGreaterEqual(manager.status()["total_attempt_count"], 2)
        manager.shutdown()

    def test_actions_while_unavailable_do_not_create_attempts_or_log_storm(self):
        logs = []
        factory = FakeVTSFactory(default=ConnectionRefusedError("closed"))
        manager = self.manager(factory, logs=logs, backoff_min=0.5, backoff_max=0.5)
        manager.start()
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "BACKOFF"))
        attempts = manager.status()["total_attempt_count"]
        self.assertEqual([manager.trigger_hotkey("ConfiguredExpression") for _ in range(10)], [False] * 10)
        self.assertEqual(manager.status()["total_attempt_count"], attempts)
        self.assertEqual(manager.status()["action_dropped_count"], 10)
        self.assertEqual(sum("event=vts_action_dropped" in line for line in logs), 1)
        manager.shutdown()

    def test_actions_dropped_while_unavailable_are_not_replayed_after_reconnect(self):
        factory = FakeVTSFactory(results=[ConnectionRefusedError("closed"), None], default=None)
        manager = self.manager(factory, backoff_min=0.03, backoff_max=0.03)
        manager.start()
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "BACKOFF"))
        for _ in range(10):
            self.assertFalse(manager.trigger_hotkey("ConfiguredExpression"))
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "CONNECTED"))
        time.sleep(0.03)
        self.assertEqual(factory.clients[-1].triggered, [])
        manager.shutdown()

    def test_auth_failure_is_hard_and_does_not_retry(self):
        logs = []
        factory = FakeVTSFactory(default=VTSAuthError("invalid token"))
        manager = self.manager(factory, logs=logs)
        manager.start()
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "UNAVAILABLE"))
        time.sleep(0.1)
        self.assertEqual(manager.status()["total_attempt_count"], 1)
        self.assertTrue(any("event=vts_auth_failed" in line for line in logs))
        manager.shutdown()

    def test_protocol_failure_is_hard_and_observable(self):
        logs = []
        factory = FakeVTSFactory(default=VTSProtocolError("version mismatch"))
        manager = self.manager(factory, logs=logs)
        manager.start()
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "UNAVAILABLE"))
        time.sleep(0.1)
        self.assertEqual(manager.status()["total_attempt_count"], 1)
        self.assertEqual(manager.status()["last_error_kind"], "protocol_incompatible")
        self.assertTrue(any("reason=protocol_incompatible" in line for line in logs))
        manager.shutdown()

    def test_shutdown_during_backoff_prevents_later_retry(self):
        factory = FakeVTSFactory(default=ConnectionRefusedError("closed"))
        manager = self.manager(factory, backoff_min=0.2, backoff_max=0.2)
        manager.start()
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "BACKOFF"))
        attempts = manager.status()["total_attempt_count"]
        result = manager.shutdown(timeout_seconds=0.2)
        time.sleep(0.25)
        self.assertTrue(result["stopped"])
        self.assertEqual(manager.status()["total_attempt_count"], attempts)
        self.assertFalse(manager.worker_alive)

    def test_shutdown_while_connecting_is_bounded_and_cancels_connect(self):
        factory = FakeVTSFactory(default="wait")
        manager = self.manager(factory)
        manager.start()
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "CONNECTING"))
        started = time.perf_counter()
        result = manager.shutdown(timeout_seconds=0.2)
        self.assertLess(time.perf_counter() - started, 0.2)
        self.assertTrue(result["stopped"])
        self.assertFalse(manager.worker_alive)

    def test_shutdown_connected_closes_client_and_worker(self):
        factory = FakeVTSFactory()
        manager = self.manager(factory)
        manager.start()
        self.assertTrue(wait_until(lambda: manager.status()["status"] == "CONNECTED"))
        result = manager.shutdown(timeout_seconds=0.2)
        self.assertTrue(result["stopped"])
        self.assertGreaterEqual(factory.clients[0].close_count, 1)
        self.assertFalse(manager.worker_alive)

    def test_disabled_vts_creates_no_client_or_retry_thread(self):
        factory = FakeVTSFactory()
        manager = self.manager(factory, enabled=False)
        status = manager.start()
        self.assertEqual(status["status"], VTSConnectionState.DISABLED.value)
        self.assertEqual(factory.clients, [])
        self.assertFalse(manager.trigger_hotkey("ConfiguredExpression"))
        self.assertFalse(manager.worker_alive)
        self.assertTrue(manager.shutdown()["stopped"])


if __name__ == "__main__":
    unittest.main()
