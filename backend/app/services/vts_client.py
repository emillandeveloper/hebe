from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import os
import threading
import time
from typing import Any, Callable

import websockets


VTS_HOST = os.getenv("HEBE_VTS_HOST", "127.0.0.1")
VTS_PORT = int(os.getenv("HEBE_VTS_PORT", "8001"))
VTS_PLUGIN_NAME = os.getenv("HEBE_VTS_PLUGIN_NAME", "HebeAssistant")
VTS_PLUGIN_AUTHOR = os.getenv("HEBE_VTS_PLUGIN_AUTHOR", "Leo")
VTS_PLUGIN_ICON = None
VTS_TOKEN_FILE = os.getenv("HEBE_VTS_TOKEN_FILE", "vts_auth_token.txt")
VTS_CONNECT_TIMEOUT_SECONDS = float(os.getenv("HEBE_VTS_CONNECT_TIMEOUT_SECONDS", "5") or 5)
VTS_REQUEST_TIMEOUT_SECONDS = float(os.getenv("HEBE_VTS_REQUEST_TIMEOUT_SECONDS", "15") or 15)
VTS_BACKOFF_MIN_SECONDS = float(os.getenv("HEBE_VTS_BACKOFF_MIN_SECONDS", "5") or 5)
VTS_BACKOFF_MAX_SECONDS = float(os.getenv("HEBE_VTS_BACKOFF_MAX_SECONDS", "300") or 300)
VTS_ACTION_TTL_SECONDS = float(os.getenv("HEBE_VTS_ACTION_TTL_SECONDS", "2") or 2)
VTS_SHUTDOWN_TIMEOUT_SECONDS = float(os.getenv("HEBE_VTS_SHUTDOWN_TIMEOUT_SECONDS", "2") or 2)


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class VTSConnectionState(str, Enum):
    DISABLED = "DISABLED"
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    BACKOFF = "BACKOFF"
    UNAVAILABLE = "UNAVAILABLE"


class VTSAuthError(RuntimeError):
    pass


class VTSProtocolError(RuntimeError):
    pass


class VTSConnectionDropped(ConnectionError):
    pass


class VTSClient:
    def __init__(self, host: str | None = None, port: int | None = None):
        self.host = host or os.getenv("HEBE_VTS_HOST", VTS_HOST)
        self.port = int(port if port is not None else os.getenv("HEBE_VTS_PORT", str(VTS_PORT)))
        self.token_file = os.getenv("HEBE_VTS_TOKEN_FILE", VTS_TOKEN_FILE)
        self.plugin_name = os.getenv("HEBE_VTS_PLUGIN_NAME", VTS_PLUGIN_NAME)
        self.plugin_author = os.getenv("HEBE_VTS_PLUGIN_AUTHOR", VTS_PLUGIN_AUTHOR)
        self.ws = None
        self.authenticated = False
        self.auth_token = None
        self._hotkey_ids: dict[str, str] = {}

        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, "r", encoding="utf-8") as token_file:
                    self.auth_token = token_file.read().strip() or None
            except OSError:
                self.auth_token = None

    async def connect(self) -> None:
        uri = f"ws://{self.host}:{self.port}"
        self.ws = await websockets.connect(
            uri,
            open_timeout=VTS_CONNECT_TIMEOUT_SECONDS,
            close_timeout=1,
            ping_interval=20,
            ping_timeout=10,
        )
        await self.authenticate()

    async def _receive_json(self) -> dict:
        if self.ws is None:
            raise VTSConnectionDropped("VTS websocket is not initialized")
        try:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=VTS_REQUEST_TIMEOUT_SECONDS)
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise VTSProtocolError("VTS returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise VTSProtocolError("VTS returned a non-object response")
        return response

    async def request_auth_token(self) -> None:
        if self.ws is None:
            raise VTSConnectionDropped("VTS websocket is not initialized")
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "token-1",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_author,
                "pluginIcon": VTS_PLUGIN_ICON,
            },
        }))
        response = await self._receive_json()
        if response.get("messageType") != "AuthenticationTokenResponse":
            raise VTSAuthError("VTS rejected the authentication token request")
        token = response.get("data", {}).get("authenticationToken")
        if not token:
            raise VTSAuthError("VTS returned an empty authentication token")
        self.auth_token = str(token)
        try:
            with open(self.token_file, "w", encoding="utf-8") as token_file:
                token_file.write(self.auth_token)
        except OSError as exc:
            raise VTSAuthError("VTS authentication token could not be saved") from exc

    async def authenticate(self) -> None:
        if self.ws is None:
            raise VTSConnectionDropped("VTS websocket is not initialized")
        if self.auth_token is None:
            await self.request_auth_token()
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth-1",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.plugin_author,
                "pluginIcon": VTS_PLUGIN_ICON,
                "authenticationToken": self.auth_token,
            },
        }))
        response = await self._receive_json()
        if response.get("messageType") != "AuthenticationResponse":
            raise VTSProtocolError("VTS returned an incompatible authentication response")
        if not response.get("data", {}).get("authenticated"):
            raise VTSAuthError("VTS authentication was rejected")
        self.authenticated = True

    async def _send_request(self, message_type: str, data: dict, request_id: str) -> dict:
        if self.ws is None or not self.authenticated:
            raise VTSConnectionDropped("VTS is not connected and authenticated")
        await self.ws.send(json.dumps({
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": request_id,
            "messageType": message_type,
            "data": data,
        }))
        response = await self._receive_json()
        if response.get("messageType") == "APIError":
            detail = str(response.get("data", {}).get("message") or "VTS API error")
            raise VTSProtocolError(detail)
        return response

    async def get_hotkey_id_by_name(self, hotkey_name: str) -> str | None:
        cached = self._hotkey_ids.get(hotkey_name)
        if cached:
            return cached
        response = await self._send_request(
            "HotkeysInCurrentModelRequest", {}, request_id="hotkeys-lookup"
        )
        data = response.get("data", {}) or {}
        hotkeys = data.get("availableHotkeys") or data.get("hotkeys") or []
        for hotkey in hotkeys:
            if hotkey.get("name") == hotkey_name and hotkey.get("hotkeyID"):
                hotkey_id = str(hotkey["hotkeyID"])
                self._hotkey_ids[hotkey_name] = hotkey_id
                return hotkey_id
        return None

    async def trigger_hotkey(self, hotkey_name: str) -> bool:
        hotkey_id = await self.get_hotkey_id_by_name(hotkey_name)
        if not hotkey_id:
            return False
        await self._send_request(
            "HotkeyTriggerRequest",
            {"hotkeyID": hotkey_id},
            request_id=f"hotkey-{hotkey_name}",
        )
        return True

    async def wait_closed(self) -> None:
        if self.ws is not None:
            await self.ws.wait_closed()

    async def close(self) -> None:
        websocket = self.ws
        self.ws = None
        self.authenticated = False
        self._hotkey_ids.clear()
        if websocket is not None:
            await websocket.close()


@dataclass(slots=True)
class _VTSAction:
    hotkey_name: str
    created_at: float


class VTSConnectionManager:
    """Single background lifecycle for the optional VTube Studio bridge."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        client_factory: Callable[[], Any] | None = None,
        backoff_min_seconds: float | None = None,
        backoff_max_seconds: float | None = None,
        action_ttl_seconds: float | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.enabled = _env_enabled("HEBE_VTS_ENABLED", True) if enabled is None else bool(enabled)
        self.client_factory = client_factory or VTSClient
        configured_min = float(
            backoff_min_seconds
            if backoff_min_seconds is not None
            else os.getenv("HEBE_VTS_BACKOFF_MIN_SECONDS", str(VTS_BACKOFF_MIN_SECONDS))
        )
        configured_max = float(
            backoff_max_seconds
            if backoff_max_seconds is not None
            else os.getenv("HEBE_VTS_BACKOFF_MAX_SECONDS", str(VTS_BACKOFF_MAX_SECONDS))
        )
        configured_ttl = float(
            action_ttl_seconds
            if action_ttl_seconds is not None
            else os.getenv("HEBE_VTS_ACTION_TTL_SECONDS", str(VTS_ACTION_TTL_SECONDS))
        )
        self.backoff_min_seconds = max(0.01, configured_min)
        self.backoff_max_seconds = max(self.backoff_min_seconds, configured_max)
        self.action_ttl_seconds = max(0.01, configured_ttl)
        self._logger = logger or (lambda line: print(line, flush=True))
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_stop: asyncio.Event | None = None
        self._action_queue: asyncio.Queue[_VTSAction] | None = None
        self._client = None
        self._stopping = False
        self._state = VTSConnectionState.DISABLED if not self.enabled else VTSConnectionState.DISCONNECTED
        self._attempt_count = 0
        self._total_attempt_count = 0
        self._consecutive_failures = 0
        self._last_error = ""
        self._last_error_kind = ""
        self._last_logged_error_signature = ""
        self._last_attempt_at = 0.0
        self._next_retry_at = 0.0
        self._connected_at = 0.0
        self._action_dropped_count = 0
        self._action_delivered_count = 0
        self._last_action_outcome = ""
        self._last_drop_signature = ""

    @property
    def worker_alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def _emit(self, event: str, *, reason: str = "", **details: Any) -> None:
        fields = [f"event={event}", f"state={self._state.value}"]
        if reason:
            fields.append(f"reason={reason}")
        fields.extend(f"{key}={value}" for key, value in details.items() if value not in {None, ""})
        self._logger("[HEBE][VTS_LIFECYCLE] " + " ".join(fields))

    def _transition(self, state: VTSConnectionState, event: str, *, reason: str = "", **details: Any) -> None:
        with self._lock:
            previous = self._state
            self._state = state
        self._emit(event, reason=reason, previous_state=previous.value, **details)

    def status(self) -> dict:
        with self._lock:
            now = time.time()
            return {
                "status": self._state.value,
                "enabled": self.enabled,
                "attempt_count": self._attempt_count,
                "total_attempt_count": self._total_attempt_count,
                "last_error": self._last_error,
                "last_error_kind": self._last_error_kind,
                "last_attempt_at": self._last_attempt_at,
                "next_retry_at": self._next_retry_at,
                "connection_age": max(0.0, now - self._connected_at) if self._connected_at else 0.0,
                "action_dropped_count": self._action_dropped_count,
                "action_delivered_count": self._action_delivered_count,
                "last_action_outcome": self._last_action_outcome,
                "worker_alive": self.worker_alive,
            }

    def start(self) -> dict:
        with self._lock:
            if not self.enabled:
                self._state = VTSConnectionState.DISABLED
                self._emit("vts_disabled", reason="config_disabled")
                return self.status()
            if self.worker_alive:
                return self.status()
            self._stopping = False
            self._attempt_count = 0
            self._consecutive_failures = 0
            self._last_error = ""
            self._last_error_kind = ""
            self._last_logged_error_signature = ""
            self._next_retry_at = 0.0
            self._state = VTSConnectionState.DISCONNECTED
            self._thread = threading.Thread(
                target=self._thread_main,
                name="hebe-vts-lifecycle",
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            if not self._stopping:
                kind, _retryable = self._classify_error(exc)
                with self._lock:
                    self._last_error = self._safe_error(exc)
                    self._last_error_kind = kind
                self._transition(VTSConnectionState.UNAVAILABLE, "vts_unavailable", reason=kind)
        finally:
            with self._lock:
                self._loop = None
                self._async_stop = None
                self._action_queue = None
                self._client = None

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._async_stop = asyncio.Event()
        self._action_queue = asyncio.Queue(maxsize=2)
        while not self._stopping:
            client = None
            was_connected = False
            try:
                with self._lock:
                    self._attempt_count = self._consecutive_failures + 1
                    self._total_attempt_count += 1
                    self._last_attempt_at = time.time()
                    attempt = self._attempt_count
                self._transition(VTSConnectionState.CONNECTING, "vts_connecting", attempt_count=attempt)
                client = self.client_factory()
                with self._lock:
                    self._client = client
                if not await self._connect_or_stop(client):
                    break
                was_connected = True
                with self._lock:
                    self._attempt_count = 0
                    self._consecutive_failures = 0
                    self._last_error = ""
                    self._last_error_kind = ""
                    self._last_logged_error_signature = ""
                    self._next_retry_at = 0.0
                    self._connected_at = time.time()
                self._transition(VTSConnectionState.CONNECTED, "vts_connected")
                await self._serve_connected(client)
                if not self._stopping:
                    raise VTSConnectionDropped("VTS websocket closed")
            except Exception as exc:
                if self._stopping:
                    break
                kind, retryable = self._classify_error(exc)
                safe_error = self._safe_error(exc)
                with self._lock:
                    self._last_error = safe_error
                    self._last_error_kind = kind
                    self._connected_at = 0.0
                if was_connected:
                    self._transition(VTSConnectionState.DISCONNECTED, "vts_disconnected", reason=kind)
                    self._drop_pending_actions("connection_lost")
                if client is not None:
                    try:
                        await client.close()
                    except Exception:
                        pass
                    with self._lock:
                        if self._client is client:
                            self._client = None
                    client = None
                if not retryable:
                    event = "vts_auth_failed" if kind == "auth_failed" else "vts_unavailable"
                    self._transition(VTSConnectionState.UNAVAILABLE, event, reason=kind, error=safe_error)
                    await self._wait_for_shutdown()
                    break
                with self._lock:
                    self._consecutive_failures += 1
                    failures = self._consecutive_failures
                signature = f"{kind}:{safe_error}"
                if signature != self._last_logged_error_signature:
                    self._last_logged_error_signature = signature
                    self._emit("vts_unavailable", reason=kind, error=safe_error)
                delay = min(
                    self.backoff_max_seconds,
                    self.backoff_min_seconds * (2 ** max(0, failures - 1)),
                )
                with self._lock:
                    self._next_retry_at = time.time() + delay
                self._transition(
                    VTSConnectionState.BACKOFF,
                    "vts_backoff",
                    reason=kind,
                    attempt_count=failures,
                    retry_in_seconds=f"{delay:.2f}",
                    next_retry_at=f"{self._next_retry_at:.3f}",
                )
                if await self._wait_or_stop(delay):
                    break
            finally:
                if client is not None:
                    try:
                        await client.close()
                    except Exception:
                        pass
                with self._lock:
                    if self._client is client:
                        self._client = None

    async def _connect_or_stop(self, client: Any) -> bool:
        assert self._async_stop is not None
        connect_task = asyncio.create_task(client.connect())
        stop_task = asyncio.create_task(self._async_stop.wait())
        try:
            done, _pending = await asyncio.wait(
                {connect_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done:
                connect_task.cancel()
                await asyncio.gather(connect_task, return_exceptions=True)
                return False
            await connect_task
            return True
        finally:
            if not stop_task.done():
                stop_task.cancel()

    async def _serve_connected(self, client: Any) -> None:
        assert self._async_stop is not None
        assert self._action_queue is not None
        closed_task = asyncio.create_task(client.wait_closed())
        stop_task = asyncio.create_task(self._async_stop.wait())
        try:
            while not self._stopping:
                action_task = asyncio.create_task(self._action_queue.get())
                done, _pending = await asyncio.wait(
                    {closed_task, stop_task, action_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_task in done:
                    action_task.cancel()
                    return
                if closed_task in done:
                    action_task.cancel()
                    exception = closed_task.exception()
                    if exception is not None:
                        raise exception
                    return
                action = action_task.result()
                if time.monotonic() - action.created_at > self.action_ttl_seconds:
                    self._record_action_drop(action.hotkey_name, "stale")
                    continue
                try:
                    delivered = bool(await client.trigger_hotkey(action.hotkey_name))
                except Exception:
                    self._record_action_drop(action.hotkey_name, "connection_lost")
                    raise
                with self._lock:
                    if delivered:
                        self._action_delivered_count += 1
                        self._last_action_outcome = "delivered"
                    else:
                        self._last_action_outcome = "hotkey_not_found"
                if not delivered:
                    self._record_action_drop(action.hotkey_name, "hotkey_not_found")
        finally:
            for task in (closed_task, stop_task):
                if not task.done():
                    task.cancel()

    async def _wait_or_stop(self, delay: float) -> bool:
        assert self._async_stop is not None
        try:
            await asyncio.wait_for(self._async_stop.wait(), timeout=max(0.0, delay))
            return True
        except asyncio.TimeoutError:
            return False

    async def _wait_for_shutdown(self) -> None:
        assert self._async_stop is not None
        await self._async_stop.wait()

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        return " ".join(str(exc).split())[:180] or type(exc).__name__

    @staticmethod
    def _classify_error(exc: Exception) -> tuple[str, bool]:
        if isinstance(exc, VTSAuthError):
            return "auth_failed", False
        if isinstance(exc, VTSProtocolError):
            return "protocol_incompatible", False
        if isinstance(exc, ConnectionRefusedError):
            return "connection_refused", True
        if isinstance(exc, OSError) and getattr(exc, "winerror", None) in {10061, 1225}:
            return "connection_refused", True
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return "connection_timeout", True
        connection_closed = getattr(websockets, "ConnectionClosed", ())
        invalid_handshake = getattr(websockets, "InvalidHandshake", ())
        if connection_closed and isinstance(exc, connection_closed):
            return "connection_dropped", True
        if invalid_handshake and isinstance(exc, invalid_handshake):
            return "protocol_incompatible", False
        if isinstance(exc, VTSConnectionDropped):
            return "connection_dropped", True
        return "connection_error", True

    def trigger_hotkey(self, hotkey_name: str) -> bool:
        name = str(hotkey_name or "").strip()
        with self._lock:
            state = self._state
            loop = self._loop
            if (
                not name
                or self._stopping
                or state is not VTSConnectionState.CONNECTED
                or loop is None
                or not loop.is_running()
            ):
                reason = "disabled" if state is VTSConnectionState.DISABLED else "not_connected"
                self._record_action_drop(name or "unknown", reason)
                return False
        action = _VTSAction(name, time.monotonic())
        try:
            loop.call_soon_threadsafe(self._enqueue_action, action)
        except RuntimeError:
            self._record_action_drop(name, "lifecycle_stopping")
            return False
        return True

    def _enqueue_action(self, action: _VTSAction) -> None:
        queue = self._action_queue
        if self._stopping or self._state is not VTSConnectionState.CONNECTED or queue is None:
            self._record_action_drop(action.hotkey_name, "not_connected")
            return
        if queue.full():
            try:
                dropped = queue.get_nowait()
                self._record_action_drop(dropped.hotkey_name, "superseded")
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(action)

    def _drop_pending_actions(self, reason: str) -> None:
        queue = self._action_queue
        if queue is None:
            return
        while True:
            try:
                action = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._record_action_drop(action.hotkey_name, reason)

    def _record_action_drop(self, hotkey_name: str, reason: str) -> None:
        with self._lock:
            self._action_dropped_count += 1
            self._last_action_outcome = f"dropped:{reason}"
            count = self._action_dropped_count
            signature = f"{reason}:{self._state.value}"
            should_log = signature != self._last_drop_signature or count % 25 == 0
            if should_log:
                self._last_drop_signature = signature
        if should_log:
            self._emit(
                "vts_action_dropped",
                reason=reason,
                hotkey=hotkey_name,
                dropped_count=count,
            )

    def shutdown(self, *, timeout_seconds: float = VTS_SHUTDOWN_TIMEOUT_SECONDS) -> dict:
        with self._lock:
            self._stopping = True
            loop = self._loop
            async_stop = self._async_stop
            thread = self._thread
        if loop is not None and loop.is_running():
            if async_stop is not None:
                loop.call_soon_threadsafe(async_stop.set)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout_seconds)))
        with self._lock:
            stopped = not self.worker_alive
            self._next_retry_at = 0.0
            self._connected_at = 0.0
            self._state = VTSConnectionState.DISABLED if not self.enabled else VTSConnectionState.DISCONNECTED
            self._thread = None if stopped else thread
        self._emit("vts_shutdown", reason="engine_stop", stopped=str(stopped).lower())
        return {"status": self._state.value, "stopped": stopped, "worker_alive": self.worker_alive}


_manager: VTSConnectionManager | None = None
_manager_lock = threading.Lock()


def _get_manager() -> VTSConnectionManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = VTSConnectionManager()
        return _manager


def start_vts() -> dict:
    return _get_manager().start()


def shutdown_vts(*, timeout_seconds: float = VTS_SHUTDOWN_TIMEOUT_SECONDS) -> dict:
    return _get_manager().shutdown(timeout_seconds=timeout_seconds)


def get_vts_status() -> dict:
    return _get_manager().status()


def vts_hotkey(nombre_hotkey: str) -> bool:
    return _get_manager().trigger_hotkey(nombre_hotkey)
