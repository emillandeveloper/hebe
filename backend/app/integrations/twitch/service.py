# backend/app/integrations/twitch/service.py

from __future__ import annotations

import os
import re
from typing import Any


class TwitchService:
    def __init__(
        self,
        *,
        chat_client: Any | None = None,
        target_resolver: Any | None = None,
        chat_cache: Any | None = None,
        event_memory: Any | None = None,
        helix_client: Any | None = None,
        channel_name: str = "",
        bot_username: str = "JotunBot",
        shoutout_command_template: str | None = None,
    ) -> None:
        self.chat_client = chat_client
        self.target_resolver = target_resolver
        self.chat_cache = chat_cache
        self.event_memory = event_memory
        self.helix_client = helix_client
        self.channel_name = channel_name
        self.bot_username = bot_username
        self.shoutout_command_template = (
            shoutout_command_template
            or os.getenv("HEBE_SHOUTOUT_COMMAND_TEMPLATE", "!so {username}")
            or "!so {username}"
        )

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

    def get_current_stream(self) -> dict | None:
        if self.helix_client is None:
            raise RuntimeError("Twitch Helix client is not configured")
        return self.helix_client.get_stream()

    def get_channel_info(self) -> dict | None:
        if self.helix_client is None:
            raise RuntimeError("Twitch Helix client is not configured")
        return self.helix_client.get_channel_info()

    def shoutout(self, username: str) -> bool:
        target = self.normalize_twitch_username(username)
        if not target:
            return False

        command = self.shoutout_command_template.format(username=target)
        return self.send_message(command)

    def build_shoutout_command(self, username: str) -> str:
        target = self.normalize_twitch_username(username)
        if not target:
            return ""
        return self.shoutout_command_template.format(username=target)

    @staticmethod
    def normalize_twitch_username(username: str) -> str:
        target = str(username or "").strip().lstrip("@").strip()
        target = re.sub(r"\s+", "", target)
        if not re.fullmatch(r"[A-Za-z0-9_]{3,25}", target):
            return ""
        return target

    def resolve_user(self, raw_target: str) -> str | None:
        if self.target_resolver is None:
            return None

        resolve_fn = getattr(self.target_resolver, "resolve_user", None)
        if not callable(resolve_fn):
            return None

        return resolve_fn(raw_target)

    def resolve_user_details(self, raw_target: str, intent: str = ""):
        if self.target_resolver is None:
            return None

        resolve_fn = getattr(self.target_resolver, "resolve_user_details", None)
        if callable(resolve_fn):
            return resolve_fn(raw_target, intent=intent)

        username = self.resolve_user(raw_target)
        if not username:
            return None
        return {"username": username, "confidence": 0.82, "candidates": [username], "reason": "legacy_resolver"}

    def remember_user_alias(self, alias: str, username: str) -> bool:
        if self.target_resolver is None:
            return False
        remember_fn = getattr(self.target_resolver, "remember_alias", None)
        if not callable(remember_fn):
            return False
        return bool(remember_fn(alias, username))

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
