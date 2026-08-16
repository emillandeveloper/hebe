# backend/app/integrations/twitch/service.py

from __future__ import annotations

import os
import re
from typing import Any

from app.integrations.twitch.message_transport import (
    TWITCH_CHAT_MESSAGE_LIMIT,
    TwitchDeliveryOutcome,
    split_twitch_message,
)


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
        message_max_chars: int | None = None,
    ) -> None:
        self.chat_client = chat_client
        self.target_resolver = target_resolver
        self.chat_cache = chat_cache
        self.event_memory = event_memory
        self.helix_client = helix_client
        self.channel_name = channel_name
        self.bot_username = bot_username
        configured_limit = message_max_chars or int(
            os.getenv("HEBE_TWITCH_MESSAGE_MAX_CHARS", str(TWITCH_CHAT_MESSAGE_LIMIT))
        )
        self.message_max_chars = max(1, min(int(configured_limit), TWITCH_CHAT_MESSAGE_LIMIT))
        self.last_delivery_outcome: dict[str, Any] | None = None
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

        plan = split_twitch_message(message, max_chars=self.message_max_chars)
        for index, chunk in enumerate(plan.chunks, start=1):
            try:
                result = send_fn(chunk)
            except Exception as exc:
                outcome = TwitchDeliveryOutcome(
                    success=False,
                    total_chunks=len(plan.chunks),
                    sent_chunks=index - 1,
                    failed_chunk=index,
                    reason=f"send_exception:{type(exc).__name__}",
                    chunks=plan.chunks,
                    separators=plan.separators,
                    max_chars=plan.max_chars,
                )
                self.last_delivery_outcome = outcome.to_dict()
                print(
                    "[HEBE][TWITCH_DELIVERY] "
                    f"success=false sent_chunks={index - 1} total_chunks={len(plan.chunks)} "
                    f"failed_chunk={index} reason={outcome.reason}",
                    flush=True,
                )
                return False
            if result is False:
                outcome = TwitchDeliveryOutcome(
                    success=False,
                    total_chunks=len(plan.chunks),
                    sent_chunks=index - 1,
                    failed_chunk=index,
                    reason="chunk_send_failed",
                    chunks=plan.chunks,
                    separators=plan.separators,
                    max_chars=plan.max_chars,
                )
                self.last_delivery_outcome = outcome.to_dict()
                print(
                    "[HEBE][TWITCH_DELIVERY] "
                    f"success=false sent_chunks={index - 1} total_chunks={len(plan.chunks)} "
                    f"failed_chunk={index} reason=chunk_send_failed",
                    flush=True,
                )
                return False

        outcome = TwitchDeliveryOutcome(
            success=True,
            total_chunks=len(plan.chunks),
            sent_chunks=len(plan.chunks),
            chunks=plan.chunks,
            separators=plan.separators,
            max_chars=plan.max_chars,
        )
        self.last_delivery_outcome = outcome.to_dict()
        print(
            "[HEBE][TWITCH_DELIVERY] "
            f"success=true sent_chunks={len(plan.chunks)} total_chunks={len(plan.chunks)}",
            flush=True,
        )
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
