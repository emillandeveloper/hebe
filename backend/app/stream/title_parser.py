from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StreamTitleContext:
    playthrough_type: str | None = None
    challenges: list[str] = field(default_factory=list)
    stream_slot: str | None = None
    spoiler_policy: str = "no_spoilers"
    bilingual_mode: bool = False
    language_mode: str | None = None

    @property
    def challenge_value(self) -> str | None:
        return ",".join(self.challenges) if self.challenges else None


def parse_stream_title(title: str | None) -> StreamTitleContext:
    text = str(title or "")
    normalized = _normalize(text)
    challenges: list[str] = []
    playthrough_type: str | None = None
    stream_slot: str | None = None
    spoiler_policy = "no_spoilers"

    if "first playthrough" in normalized:
        playthrough_type = "first_playthrough"
        spoiler_policy = "no_spoilers"
    elif "chat playthrough" in normalized:
        playthrough_type = "chat_playthrough"
        spoiler_policy = "no_spoilers"
    elif "challenge playthrough" in normalized:
        playthrough_type = "challenge"

    challenge_rules = (
        ("level_1", r"\blevel\s*1\b"),
        ("no_sphere_grid", r"\bno\s+sphere\s+grid\b"),
        ("no_shops", r"\bno\s+shops?\b"),
    )
    for value, pattern in challenge_rules:
        if re.search(pattern, normalized) and value not in challenges:
            challenges.append(value)

    slot_text = _normalize_repeated_letters(normalized)
    if "retro weekend" in slot_text:
        stream_slot = "retro_weekend"
    elif "challenge monday" in slot_text:
        stream_slot = "challenge_monday"

    language_mode = None
    if "[eng/esp]" in normalized or "eng/esp" in normalized:
        language_mode = "ENG/ESP"
    bilingual_mode = language_mode == "ENG/ESP"

    return StreamTitleContext(
        playthrough_type=playthrough_type,
        challenges=challenges,
        stream_slot=stream_slot,
        spoiler_policy=spoiler_policy,
        bilingual_mode=bilingual_mode,
        language_mode=language_mode,
    )


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _normalize_repeated_letters(text: str) -> str:
    # Slot names are stable enough that collapsing long repeated-letter typos is safe.
    return re.sub(r"([a-z])\1{2,}", r"\1\1", text)
