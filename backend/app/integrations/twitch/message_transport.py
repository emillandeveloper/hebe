from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass, field


TWITCH_CHAT_MESSAGE_LIMIT = 500
_SENTENCE_ENDINGS = frozenset(".!?…")
_PUNCTUATION_ENDINGS = frozenset(",;:—–-")


@dataclass(frozen=True)
class TwitchMessagePlan:
    original: str
    chunks: list[str] = field(default_factory=list)
    separators: list[str] = field(default_factory=list)
    max_chars: int = TWITCH_CHAT_MESSAGE_LIMIT

    def reconstruct(self) -> str:
        if not self.chunks:
            return ""
        rebuilt = self.chunks[0]
        for separator, chunk in zip(self.separators, self.chunks[1:]):
            rebuilt += separator + chunk
        return rebuilt


@dataclass(frozen=True)
class TwitchDeliveryOutcome:
    success: bool
    total_chunks: int
    sent_chunks: int
    failed_chunk: int | None = None
    reason: str = ""
    chunks: list[str] = field(default_factory=list)
    separators: list[str] = field(default_factory=list)
    max_chars: int = TWITCH_CHAT_MESSAGE_LIMIT

    def to_dict(self) -> dict:
        return asdict(self)


def split_twitch_message(text: str, *, max_chars: int = TWITCH_CHAT_MESSAGE_LIMIT) -> TwitchMessagePlan:
    message = str(text or "").strip()
    limit = max(1, min(int(max_chars), TWITCH_CHAT_MESSAGE_LIMIT))
    if not message:
        return TwitchMessagePlan(original="", max_chars=limit)
    if len(message) <= limit:
        return TwitchMessagePlan(original=message, chunks=[message], max_chars=limit)

    chunks: list[str] = []
    separators: list[str] = []
    remaining = message
    while len(remaining) > limit:
        split_at = _choose_split(remaining, limit)
        split_at = _safe_unicode_boundary(remaining, split_at)
        if split_at <= 0:
            split_at = min(limit, len(remaining))
        chunk = remaining[:split_at].rstrip()
        separator_start = len(chunk)
        separator_end = split_at
        while separator_end < len(remaining) and remaining[separator_end].isspace():
            separator_end += 1
        separator = remaining[separator_start:separator_end]
        if not chunk:
            chunk = remaining[:split_at]
            separator_start = split_at
            separator = remaining[separator_start:separator_end]
        chunks.append(chunk)
        separators.append(separator)
        remaining = remaining[separator_end:]

    if remaining:
        chunks.append(remaining)
    if len(separators) >= len(chunks):
        separators = separators[: max(0, len(chunks) - 1)]
    plan = TwitchMessagePlan(
        original=message,
        chunks=chunks,
        separators=separators,
        max_chars=limit,
    )
    if plan.reconstruct() != message:
        raise ValueError("Twitch message splitter did not preserve the original content")
    return plan


def _choose_split(text: str, limit: int) -> int:
    whitespace = [index for index in range(1, min(limit, len(text) - 1) + 1) if text[index].isspace()]
    if not whitespace:
        return limit

    sentence = [index for index in whitespace if text[index - 1] in _SENTENCE_ENDINGS]
    punctuation = [index for index in whitespace if text[index - 1] in _PUNCTUATION_ENDINGS]
    for candidates in (sentence, punctuation, whitespace):
        selected = _select_without_tiny_tail(text, candidates, limit)
        if selected is not None:
            return selected
    return whitespace[-1]


def _select_without_tiny_tail(text: str, candidates: list[int], limit: int) -> int | None:
    for index in reversed(candidates):
        tail = text[index:].strip()
        if len(tail) > limit or len(tail.split()) >= 3:
            return index
    return None


def _safe_unicode_boundary(text: str, index: int) -> int:
    index = min(max(0, index), len(text))
    while 0 < index < len(text):
        current = text[index]
        previous = text[index - 1]
        if (
            unicodedata.combining(current)
            or current in {"\ufe0e", "\ufe0f", "\u200d"}
            or previous == "\u200d"
            or "\U0001f3fb" <= current <= "\U0001f3ff"
        ):
            index -= 1
            continue
        break
    return index
