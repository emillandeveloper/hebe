from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
import re

from app.stream.state import StreamSessionState
from app.stream.title_parser import parse_stream_title


@dataclass(frozen=True)
class StreamContextSyncConfig:
    max_safe_context_age_sec: float = 5 * 60


class StreamContextSyncService:
    def __init__(
        self,
        *,
        twitch_api: Any | None,
        config: StreamContextSyncConfig | None = None,
        now_fn=None,
    ) -> None:
        self.twitch_api = twitch_api
        self.config = config or StreamContextSyncConfig()
        self._now_fn = now_fn

    def sync(self, stream: StreamSessionState | None) -> bool:
        if stream is None:
            print("[HEBE][STREAM_CONTEXT] refresh skipped reason=no_stream_state", flush=True)
            return False

        now = self._now()
        print("[HEBE][STREAM_CONTEXT] refresh started", flush=True)
        if self.twitch_api is None:
            self._mark_error(stream, "Context sync service not initialized", now)
            return False

        try:
            previous_live_known = bool(getattr(stream, "live_status_known", False))
            previous_is_live = bool(getattr(stream, "is_live", False))
            print("[HEBE][STREAM_CONTEXT] calling Helix Get Streams", flush=True)
            live_stream = self._get_stream()
            self._log_endpoint_status("get_streams")
            if live_stream:
                stream.is_live = True
                stream.live_status_known = True
                self._apply_live_stream(stream, live_stream)
                if not previous_live_known or not previous_is_live:
                    stream.stream_spontaneity_grace_until_ts = max(
                        float(getattr(stream, "stream_spontaneity_grace_until_ts", 0.0) or 0.0),
                        now + 4 * 60,
                    )
            else:
                stream.is_live = False
                stream.live_status_known = True
                stream.stream_started_at = None

            print("[HEBE][STREAM_CONTEXT] calling Helix Get Channel Information", flush=True)
            channel = self._get_channel_info()
            self._log_endpoint_status("get_channel_information")
            if channel:
                self._apply_channel_info(stream, channel)

            stream.stream_context_updated_ts = now
            stream.last_stream_context_error = None
            self._apply_title_context(stream)
            print(
                "[HEBE][STREAM_CONTEXT] state updated "
                f"is_live={stream.is_live} "
                f"title={stream.current_stream_title!r} "
                f"category={stream.current_category!r} "
                f"game={stream.current_game!r}",
                flush=True,
            )
            return True
        except Exception as exc:
            message = str(exc) or repr(exc)
            if message.startswith("Helix ") or message.startswith("Missing Twitch config:"):
                error = message
            else:
                error = f"Unexpected exception in context sync: {type(exc).__name__}: {message}"
            self._mark_error(stream, error, now)
            return False

    def _get_stream(self) -> dict | None:
        if hasattr(self.twitch_api, "get_stream"):
            return self.twitch_api.get_stream()
        if hasattr(self.twitch_api, "get_current_stream"):
            return self.twitch_api.get_current_stream()
        raise RuntimeError("Context sync service not initialized")

    def _get_channel_info(self) -> dict | None:
        if hasattr(self.twitch_api, "get_channel_info"):
            return self.twitch_api.get_channel_info()
        raise RuntimeError("Context sync service not initialized")

    def _log_endpoint_status(self, endpoint_name: str) -> None:
        helix = getattr(self.twitch_api, "helix_client", self.twitch_api)
        statuses = getattr(helix, "last_status_by_endpoint", None)
        if isinstance(statuses, dict) and endpoint_name in statuses:
            print(
                f"[HEBE][STREAM_CONTEXT] Helix {endpoint_name} status={statuses[endpoint_name]}",
                flush=True,
            )

    def _apply_live_stream(self, stream: StreamSessionState, data: dict) -> None:
        title = data.get("title")
        category = data.get("game_name")
        tags = data.get("tags")

        if title is not None:
            stream.current_stream_title = str(title or "").strip() or None
        if category is not None:
            stream.current_category = str(category or "").strip() or None
            stream.current_game = stream.current_category
        if isinstance(tags, list):
            stream.current_tags = [str(tag) for tag in tags if str(tag or "").strip()]
        stream.stream_started_at = data.get("started_at") or stream.stream_started_at

    def _apply_channel_info(self, stream: StreamSessionState, data: dict) -> None:
        title = data.get("title")
        category = data.get("game_name")
        tags = data.get("tags")

        if title is not None and not getattr(stream, "is_live", False):
            stream.current_stream_title = str(title or "").strip() or None
        if category is not None and not getattr(stream, "is_live", False):
            stream.current_category = str(category or "").strip() or None
            stream.current_game = stream.current_category
        if isinstance(tags, list) and (tags or not getattr(stream, "is_live", False)):
            stream.current_tags = [str(tag) for tag in tags if str(tag or "").strip()]

    def _apply_title_context(self, stream: StreamSessionState) -> None:
        parsed = parse_stream_title(stream.current_stream_title)
        stream.current_playthrough_type = parsed.playthrough_type
        stream.current_challenge = parsed.challenge_value
        stream.current_stream_slot = parsed.stream_slot
        stream.spoiler_policy = parsed.spoiler_policy
        stream.bilingual_mode = parsed.bilingual_mode
        stream.language_mode = parsed.language_mode
        markers = self._extract_title_markers(stream.current_stream_title, stream.current_category)
        if markers != list(getattr(stream, "title_context_markers", []) or []):
            stream.title_context_markers = markers
            stream.title_context_updated_ts = self._now()
            if not getattr(stream, "run_context_source", None):
                stream.run_context_source = "title"

    def _extract_title_markers(self, title: str | None, category: str | None = None) -> list[str]:
        text = str(title or "")
        if not text.strip():
            return []
        category_words = {part.lower() for part in re.findall(r"[A-Za-z0-9]+", str(category or "")) if len(part) > 2}
        ignored = {
            "eng", "esp", "retro", "weekend", "challenge", "playthrough", "first",
            "level", "final", "fantasy", "episode", "stream", "streaming", "then",
            "know", "bye", "soooo", "food", "leveling", "the", "and", "for", "you",
        }
        markers: list[str] = []
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9'!-]{2,}(?:\s+[A-Z][A-Za-z0-9'!-]{2,}){0,2}\b", text):
            value = match.group(0).strip(" -|:.,")
            words = [w.lower().strip("'!-") for w in value.split()]
            if not words or all(w in ignored or w in category_words for w in words):
                continue
            if value not in markers:
                markers.append(value)
        return markers[:8]

    def _mark_error(self, stream: StreamSessionState, error: str, now: float) -> None:
        stream.last_stream_context_error = error
        if now - float(getattr(stream, "stream_context_updated_ts", 0.0) or 0.0) > self.config.max_safe_context_age_sec:
            stream.live_status_known = False
            stream.is_live = False
        print(f"[HEBE][STREAM_CONTEXT] sync failed: {error}", flush=True)

    def _now(self) -> float:
        if self._now_fn is not None:
            return float(self._now_fn())
        return time.time()
