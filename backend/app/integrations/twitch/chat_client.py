from __future__ import annotations

import requests
import os
import threading
import uuid
from typing import Optional


class TwitchChatClient:

    CHAT_MESSAGE_URL = "https://api.twitch.tv/helix/chat/messages"
    SHOUTOUT_URL = "https://api.twitch.tv/helix/chat/shoutouts"

    def __init__(
        self,
        *,
        channel_name: str = "leonifelheim",
        broadcaster_id: str = "124070929",
        sender_id: str = "1480877711",
        client_id: str = "gp762nuuoqcoxypju8c569th9wz7q5",
        oauth_token: str = "f945r0izxxbt2mrvkoo7zrmpuqv5l3",
        bot_username: str = "HebeNifelheim",
        enabled: bool = True,
        session: Optional[requests.Session] = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self.channel_name = str(channel_name or "").strip()
        self.broadcaster_id = str(broadcaster_id or "").strip()
        self.sender_id = str(sender_id or "").strip()
        self.client_id = str(client_id or "").strip()
        self.oauth_token = str(oauth_token or "").strip()
        self.bot_username = str(bot_username or "").strip() or "HebeNifelheim"
        self.enabled = enabled
        self.timeout_sec = timeout_sec

        self._session = session or requests.Session()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        if not self.enabled:
            print("[HEBE][TWITCH][CHAT] client disabled", flush=True)
            self._connected = False
            return False

        missing = []
        if not self.channel_name:
            missing.append("channel_name")
        if not self.broadcaster_id:
            missing.append("broadcaster_id")
        if not self.sender_id:
            missing.append("sender_id")
        if not self.client_id:
            missing.append("client_id")
        if not self.oauth_token:
            missing.append("oauth_token")

        if missing:
            print(
                f"[HEBE][TWITCH][CHAT] missing config: {', '.join(missing)}",
                flush=True,
            )
            self._connected = False
            return False

        self._connected = True
        print(
            f"[HEBE][TWITCH][CHAT] ready "
            f"bot={self.bot_username!r} channel={self.channel_name!r}",
            flush=True,
        )
        return True

    def ensure_connected(self) -> bool:
        if self._connected:
            return True
        return self.connect()

    def send_message(self, text: str) -> bool:
        if not self.enabled:
            print("[HEBE][TWITCH][CHAT] send blocked: client disabled", flush=True)
            return False

        message = str(text or "").strip()
        if not message:
            print("[HEBE][TWITCH][CHAT] send blocked: empty message", flush=True)
            return False

        if not self.ensure_connected():
            print("[HEBE][TWITCH][CHAT] send blocked: not ready", flush=True)
            return False

        # 🔥 TRACE ID
        trace_id = uuid.uuid4().hex[:8]

        print(
            f"[TRACE {trace_id}] BEFORE POST | "
            f"pid={os.getpid()} thread={threading.get_ident()} | "
            f"channel={self.channel_name} sender_id={self.sender_id} | "
            f"message={message!r}",
            flush=True,
        )

        headers = self._build_headers()
        payload = {
            "broadcaster_id": self.broadcaster_id,
            "sender_id": self.sender_id,
            "message": message,
        }

        try:
            response = self._session.post(
                self.CHAT_MESSAGE_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout_sec,
            )
        except requests.RequestException as exc:
            print(f"[TRACE {trace_id}] EXCEPTION: {exc!r}", flush=True)
            return False

        print(
            f"[TRACE {trace_id}] AFTER POST | "
            f"status={response.status_code} body={response.text}",
            flush=True,
        )

        if not response.ok:
            print(
                f"[HEBE][TWITCH][CHAT] send failed "
                f"status={response.status_code} body={response.text}",
                flush=True,
            )
            return False

        print(
            f"[HEBE][TWITCH][CHAT][SEND] "
            f"#{self.channel_name} <{self.bot_username}>: {message}",
            flush=True,
        )
        return True

    def _build_headers(self) -> dict[str, str]:
        token = self.oauth_token
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"

        return {
            "Authorization": token,
            "Client-Id": self.client_id,
            "Content-Type": "application/json",
        }