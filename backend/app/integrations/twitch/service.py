# backend/app/integrations/twitch/service.py

from __future__ import annotations

from typing import Any


class TwitchService:
    def __init__(
        self,
        *,
        chat_client: Any | None = None,
        target_resolver: Any | None = None,
        chat_cache: Any | None = None,
        event_memory: Any | None = None,
        channel_name: str = "",
        bot_username: str = "JotunBot",
    ) -> None:
        self.chat_client = chat_client
        self.target_resolver = target_resolver
        self.chat_cache = chat_cache
        self.event_memory = event_memory
        self.channel_name = channel_name
        self.bot_username = bot_username

    def is_available(self) -> bool:
        return self.chat_client is not None

    def send_message(self, text: str) -> bool:
        message = str(text or "").strip()
        if not message:
            return False

        if self.chat_client is None:
            raise RuntimeError("Twitch chat client is not configured")

        send_fn = getattr(self.chat_client, "send_message", None)
        if not callable(send_fn):
            raise RuntimeError("Twitch chat client does not implement send_message")

        send_fn(message)
        return True

    def shoutout(self, username: str) -> bool:
        target = str(username or "").strip()
        if not target:
            return False

        return self.send_message(f"/shoutout {target}")

    def resolve_user(self, raw_target: str) -> str | None:
        if self.target_resolver is None:
            return None

        resolve_fn = getattr(self.target_resolver, "resolve_user", None)
        if not callable(resolve_fn):
            return None

        return resolve_fn(raw_target)

    def remember_chat_message(
        self,
        *,
        username: str,
        display_name: str = "",
        text: str = "",
    ) -> None:
        if self.chat_cache is None:
            return

        add_fn = getattr(self.chat_cache, "add_message", None)
        if not callable(add_fn):
            return

        add_fn(username=username, display_name=display_name, text=text)

    def remember_raid(
        self,
        *,
        username: str,
        viewer_count: int = 0,
    ) -> None:
        if self.event_memory is None:
            return

        set_raid_fn = getattr(self.event_memory, "set_last_raid", None)
        if callable(set_raid_fn):
            set_raid_fn(username=username, viewer_count=viewer_count)
            return

    def remember_follow(self, *, username: str) -> None:
        if self.event_memory is None:
            return

        set_follow_fn = getattr(self.event_memory, "set_last_follow", None)
        if callable(set_follow_fn):
            set_follow_fn(username=username)
            return

        if hasattr(self.event_memory, "last_follow_username"):
            self.event_memory.last_follow_username = username

    def remember_sub(self, *, username: str) -> None:
        if self.event_memory is None:
            return

        set_sub_fn = getattr(self.event_memory, "set_last_sub", None)
        if callable(set_sub_fn):
            set_sub_fn(username=username)
            return

        if hasattr(self.event_memory, "last_sub_username"):
            self.event_memory.last_sub_username = username