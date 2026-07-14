from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional, Callable

import requests

try:
    import websocket  # websocket-client
except ImportError:
    websocket = None


class TwitchEventAdapter:
    EVENTSUB_WS_URL = "wss://eventsub.wss.twitch.tv/ws"
    CREATE_SUB_URL = "https://api.twitch.tv/helix/eventsub/subscriptions"

    def __init__(
        self,
        *,
        client_id: str,
        user_oauth_token: str,
        broadcaster_user_id: str,
        bot_user_id: str,
        twitch_service: Any,
        enabled: bool = True,
        keepalive_timeout_seconds: int = 30,
        session: Optional[requests.Session] = None,
        push_event_callback: Optional[Callable[[str, dict], None]] = None,
        bot_username: str = "",
        subscribe_chat_messages: bool = False,
    ) -> None:
        self.client_id = str(client_id or "").strip()
        self.user_oauth_token = str(user_oauth_token or "").strip()
        self.broadcaster_user_id = str(broadcaster_user_id or "").strip()
        self.bot_user_id = str(bot_user_id or "").strip()
        self.bot_username = str(bot_username or "").strip().lower()
        self.twitch_service = twitch_service
        self.enabled = enabled
        self.keepalive_timeout_seconds = int(keepalive_timeout_seconds)
        self._session = session or requests.Session()
        self.push_event_callback = push_event_callback
        self.subscribe_chat_messages = bool(subscribe_chat_messages)

        self._ws_app = None
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        self._session_id: Optional[str] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self) -> bool:
        if not self.enabled:
            print("[HEBE][TWITCH][EVENTSUB] disabled", flush=True)
            return False

        if websocket is None:
            print("[HEBE][TWITCH][EVENTSUB] websocket-client not installed", flush=True)
            return False

        if self._thread and self._thread.is_alive():
            return True

        self._stop = False
        url = f"{self.EVENTSUB_WS_URL}?keepalive_timeout_seconds={self.keepalive_timeout_seconds}"

        self._ws_app = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop = True
        self._connected = False
        try:
            if self._ws_app is not None:
                self._ws_app.close()
        except Exception:
            pass

    def _run_forever(self) -> None:
        try:
            assert self._ws_app is not None
            self._ws_app.run_forever()
        except Exception as exc:
            print(f"[HEBE][TWITCH][EVENTSUB] run_forever failed: {exc!r}", flush=True)

    def _on_open(self, ws) -> None:
        self._connected = True
        print("[HEBE][TWITCH][EVENTSUB] websocket opened", flush=True)

    def _on_close(self, ws, status_code, msg) -> None:
        self._connected = False
        print(
            f"[HEBE][TWITCH][EVENTSUB] websocket closed "
            f"status={status_code} msg={msg}",
            flush=True,
        )

        if not self._stop:
            time.sleep(2.0)
            self.start()

    def _on_error(self, ws, error) -> None:
        print(f"[HEBE][TWITCH][EVENTSUB] websocket error: {error!r}", flush=True)

    def _on_message(self, ws, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
        except Exception as exc:
            print(f"[HEBE][TWITCH][EVENTSUB] invalid json: {exc!r}", flush=True)
            return

        metadata = payload.get("metadata", {})
        msg_type = metadata.get("message_type")

        if msg_type == "session_welcome":
            session = payload.get("payload", {}).get("session", {})
            self._session_id = session.get("id")
            print(
                f"[HEBE][TWITCH][EVENTSUB] session_welcome session_id={self._session_id}",
                flush=True,
            )
            self._subscribe_defaults()
            return

        if msg_type == "session_keepalive":
            print("[HEBE][TWITCH][EVENTSUB] keepalive", flush=True)
            return

        if msg_type == "notification":
            subscription = payload.get("payload", {}).get("subscription", {})
            event = payload.get("payload", {}).get("event", {})
            sub_type = subscription.get("type", "")
            self._handle_event(sub_type, event)
            return

        if msg_type == "session_reconnect":
            reconnect_url = payload.get("payload", {}).get("session", {}).get("reconnect_url")
            print(
                f"[HEBE][TWITCH][EVENTSUB] reconnect requested url={reconnect_url!r}",
                flush=True,
            )
            return

        if msg_type == "revocation":
            print(f"[HEBE][TWITCH][EVENTSUB] subscription revoked: {payload}", flush=True)
            return

        print(f"[HEBE][TWITCH][EVENTSUB] unhandled message: {payload}", flush=True)

    def _subscribe_defaults(self) -> None:
        if not self._session_id:
            return

        if self.subscribe_chat_messages:
            self._create_subscription(
                sub_type="channel.chat.message",
                version="1",
                condition={
                    "broadcaster_user_id": self.broadcaster_user_id,
                    "user_id": self.bot_user_id,
                },
            )
        else:
            print(
                "[HEBE][TWITCH][EVENTSUB] channel.chat.message subscription disabled; IRC chat bot handles chat",
                flush=True,
            )

        # Follow
        self._create_subscription(
            sub_type="channel.follow",
            version="2",
            condition={
                "broadcaster_user_id": self.broadcaster_user_id,
                "moderator_user_id": self.broadcaster_user_id,
            },
        )

        # Subscribe
        self._create_subscription(
            sub_type="channel.subscribe",
            version="1",
            condition={
                "broadcaster_user_id": self.broadcaster_user_id,
            },
        )

        # Raid
        self._create_subscription(
            sub_type="channel.raid",
            version="1",
            condition={
                "to_broadcaster_user_id": self.broadcaster_user_id,
            },
        )

        self._create_subscription(
            sub_type="stream.online",
            version="1",
            condition={
                "broadcaster_user_id": self.broadcaster_user_id,
            },
        )

        self._create_subscription(
            sub_type="stream.offline",
            version="1",
            condition={
                "broadcaster_user_id": self.broadcaster_user_id,
            },
        )

    def _create_subscription(
        self,
        *,
        sub_type: str,
        version: str,
        condition: dict[str, str],
    ) -> bool:
        if not self._session_id:
            return False

        headers = self._build_headers()
        body = {
            "type": sub_type,
            "version": version,
            "condition": condition,
            "transport": {
                "method": "websocket",
                "session_id": self._session_id,
            },
        }

        try:
            response = self._session.post(
                self.CREATE_SUB_URL,
                headers=headers,
                json=body,
                timeout=10,
            )
        except requests.RequestException as exc:
            print(
                f"[HEBE][TWITCH][EVENTSUB] create subscription failed "
                f"type={sub_type} exc={exc!r}",
                flush=True,
            )
            return False

        if not response.ok:
            if sub_type == "channel.chat.message" and response.status_code == 403:
                print(
                    "[HEBE][TWITCH][EVENTSUB][WARN] optional channel.chat.message "
                    "subscription missing proper authorization; continuing because IRC chat bot handles chat",
                    flush=True,
                )
                return False
            print(
                f"[HEBE][TWITCH][EVENTSUB] create subscription failed "
                f"type={sub_type} status={response.status_code} body={response.text}",
                flush=True,
            )
            return False

        print(
            f"[HEBE][TWITCH][EVENTSUB] subscribed type={sub_type} version={version}",
            flush=True,
        )
        return True

    def _handle_event(self, sub_type: str, event: dict[str, Any]) -> None:
        print(f"[HEBE][EVENTSUB_NOTIFICATION] type={sub_type}", flush=True)
        print(f"[HEBE][TWITCH][EVENTSUB] event type={sub_type} event={event}", flush=True)

        if sub_type == "channel.chat.message":
            # Solo alimentamos chat_cache para que target_resolver funcione.
            # NO disparamos twitch_chat_react desde aquí — lo hace TwitchChatBot
            # vía IRC con su propio filtro de mention.
            chatter_user_name = str(event.get("chatter_user_name") or "").strip()
            chatter_user_login = str(event.get("chatter_user_login") or "").strip()
            message_text = str((event.get("message") or {}).get("text") or "").strip()

            username = chatter_user_login or chatter_user_name
            display_name = chatter_user_name or chatter_user_login

            if username and message_text:
                self.twitch_service.remember_chat_message(
                    username=username,
                    display_name=display_name,
                    text=message_text,
                )
            return

        if sub_type == "channel.follow":
            username = str(event.get("user_login") or event.get("user_name") or "").strip()
            if username:
                self.twitch_service.remember_follow(username=username)
                if self.push_event_callback:
                    self.push_event_callback("twitch_follow_batch", {
                        "display_names": [username],
                        "count": 1,
                        "source": "eventsub",
                        "visible_public": False,
                        "passive_eventsub": True,
                    })
            return

        if sub_type == "channel.subscribe":
            username = str(event.get("user_login") or event.get("user_name") or "").strip()
            if username:
                self.twitch_service.remember_sub(username=username)
                if self.push_event_callback:
                    self.push_event_callback("twitch_sub", {
                        "display_name": username,
                        "user_login": username,
                        "cumulative_months": 1,  # TODO: obtener de event si disponible
                        "is_gift": False,
                        "is_resub": False,
                        "source": "eventsub",
                        "visible_public": False,
                        "passive_eventsub": True,
                    })
            return

        if sub_type == "channel.raid":
            username = str(event.get("from_broadcaster_user_login") or "").strip()
            viewers = int(event.get("viewers") or 0)
            if username:
                print(
                    f"[HEBE][TWITCH_RAID_EVENT] source=eventsub raider={username!r} viewers={viewers}",
                    flush=True,
                )
                self.twitch_service.remember_raid(username=username, viewer_count=viewers)
                if self.push_event_callback:
                    self.push_event_callback("twitch_raid", {
                        "display_name": username,
                        "user_login": username,
                        "viewer_count": viewers,
                        "event_id": event.get("id") or event.get("event_id") or "",
                        "source": "eventsub",
                        "visible_public": True,
                    })
            return

        if sub_type == "stream.online":
            if self.push_event_callback:
                self.push_event_callback("stream_online", {
                    "started_at": event.get("started_at"),
                    "broadcaster_user_id": event.get("broadcaster_user_id") or self.broadcaster_user_id,
                })
            return

        if sub_type == "stream.offline":
            if self.push_event_callback:
                self.push_event_callback("stream_offline", {
                    "broadcaster_user_id": event.get("broadcaster_user_id") or self.broadcaster_user_id,
                })
            return

        # TODO (Fase 1.5 — memory/stream_end_hook):
        # Cuando se suscribas a "stream.offline" añadir aquí:
        #
        # if sub_type == "stream.offline":
        #     if self.push_event_callback:
        #         self.push_event_callback("stream_offline", {})
        #     return
        #
        # Y en hebe_engine.process_internal_event / scheduler, manejar
        # event_type="stream_offline" para:
        #   1. Leer los últimos ~50 mensajes de chat_log (source='twitch').
        #   2. Llamar a GPT-5-mini con un prompt de resumen.
        #   3. Guardar con add_chunk(text=summary, kind="stream_summary", ...).
        # Ver: backend/app/cognitive/memory/memory_store.add_chunk

    def _build_headers(self) -> dict[str, str]:
        token = self.user_oauth_token
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"

        return {
            "Authorization": token,
            "Client-Id": self.client_id,
            "Content-Type": "application/json",
        }
