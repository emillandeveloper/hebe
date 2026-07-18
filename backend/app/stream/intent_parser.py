from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata


@dataclass(frozen=True)
class StreamIntentCandidate:
    intent: str
    confidence: float
    entities: dict[str, str] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class PromotionRequest:
    raw_text: str
    target_phrase: str
    stripped_trailing_text: str = ""
    requested_by: str = "owner"
    source: str = "owner_stt_direct"
    confidence: float = 0.0
    command_detected: bool = True
    target_detected: bool = False
    reason: str = ""


class StreamIntentParser:
    """Small semantic parser for stream commands using concepts, not sentence lists."""

    shoutout_concepts = {"promo", "promocion", "promociona", "promocionar", "shoutout", "so", "recomienda"}
    chat_message_concepts = {"di", "dile", "avisa", "cuenta", "escribe", "manda", "send", "tell", "say"}
    chat_targets = {"chat", "directo", "stream"}
    ambient_concepts = {"stt", "ambiental", "ambiente"}
    enable_concepts = {"activa", "enciende", "reanuda", "pon", "enable", "resume", "on"}
    disable_concepts = {"desactiva", "apaga", "pausa", "quita", "disable", "pause", "off"}
    target_prepositions = {"a", "al", "para", "to"}
    filler = {"haz", "hazle", "dale", "manda", "pon", "un", "una", "el", "la", "de", "del", "give"}
    wake_prefix_re = re.compile(r"^\s*(?:hebe|eve|ebe|e\s*[-.]?\s*b|eb|jebe|heve)[\s,;:.-]+", re.IGNORECASE)
    promotion_patterns = (
        ("haz_promo", re.compile(r"\b(?:haz|tira)\s+(?:una?\s+)?promo\s+(?:a|al|a\s+la|al\s+canal\s+de|a\s+el\s+canal\s+de)\s+(.+)$", re.IGNORECASE)),
        ("haz_promo_compact", re.compile(r"\b(?:haz|tira)\s+(?:una?\s+)?promo\s+(@?[A-Za-z0-9_]{2,25})(?:\b|$)", re.IGNORECASE)),
        ("hazle_promo", re.compile(r"\bhazle\s+promo\s+(?:a|al|a\s+la|al\s+canal\s+de)\s+(.+)$", re.IGNORECASE)),
        ("dale_promo", re.compile(r"\bdale\s+promo\s+(?:a|al|a\s+la|al\s+canal\s+de)\s+(.+)$", re.IGNORECASE)),
        ("promociona", re.compile(r"\bpromociona\s+(?:a|al|a\s+la|al\s+canal\s+de)\s+(.+)$", re.IGNORECASE)),
        ("shoutout", re.compile(r"\bshoutout\s+(?:a|to)\s+(.+)$", re.IGNORECASE)),
        ("give_shoutout", re.compile(r"\b(?:haz(?:le)?|dale|manda|give)\s+(?:un\s+)?shoutout\s+(?:a|to)\s+(.+)$", re.IGNORECASE)),
        ("so", re.compile(r"\b(?:so|s\s*o|dale\s+so|haz\s+so)\s+(?:a|al|to)\s+(.+)$", re.IGNORECASE)),
        ("bare_promo", re.compile(r"\bpromo\s+(?:a|al|para)\s+(.+)$", re.IGNORECASE)),
    )
    trailing_banter_patterns = (
        r"\ba\s+ver\s+si\b.*$",
        r"\bsi\s+ahora\s+lo\s+hace\b.*$",
        r"\bque\s+lo\s+haga\b.*$",
        r"\bpor\s*fa(?:vor)?\b.*$",
        r"\bvenga\b.*$",
        r"\bdale\b.*$",
        r"\bque\s+esta\s+en\s+el\s+chat\b.*$",
        r"\bque\s+acaba\s+de\s+(?:seguir|hablar)\b.*$",
        r"\bel\s+que\s+acaba\s+de\s+hablar\b.*$",
        r"\bel\s+nuevo\b.*$",
        r"\bel\s+de\s+antes\b.*$",
    )

    def parse(self, text: str, *, raw_text: str | None = None) -> list[StreamIntentCandidate]:
        normalized = self.normalize(text)
        raw = str(raw_text if raw_text is not None else text or "").strip()
        candidates: list[StreamIntentCandidate] = []
        ambient = self._parse_ambient_stt(normalized)
        if ambient:
            candidates.append(ambient)
        shoutout = self._parse_shoutout(normalized, raw)
        if shoutout:
            candidates.append(shoutout)
        chat_message = self._parse_chat_message(normalized, raw)
        if chat_message:
            candidates.append(chat_message)
        return candidates

    def _parse_ambient_stt(self, normalized: str) -> StreamIntentCandidate | None:
        tokens = normalized.split()
        token_set = set(tokens)
        if "stt" not in token_set or not (token_set & self.ambient_concepts):
            return None
        if token_set & self.disable_concepts:
            return StreamIntentCandidate("stream_ambient_stt_disabled", 0.93, reason="disable_stt_ambient")
        if token_set & self.enable_concepts:
            return StreamIntentCandidate("stream_ambient_stt_enabled", 0.93, reason="enable_stt_ambient")
        return None

    def _parse_shoutout(self, normalized: str, raw_text: str) -> StreamIntentCandidate | None:
        request = self.parse_promotion_request(raw_text)
        if request is not None:
            return StreamIntentCandidate(
                "twitch_shoutout",
                request.confidence,
                entities={
                    "target_text": request.target_phrase,
                    "stripped_trailing_text": request.stripped_trailing_text,
                    "raw_promotion_text": request.raw_text,
                },
                reason="promotion_command_parser",
            )
        if self._looks_like_missing_target_promotion_command(raw_text):
            return StreamIntentCandidate(
                "twitch_shoutout",
                0.88,
                entities={"target_text": ""},
                reason="missing_promotion_target",
            )
        if set(normalized.split()) & self.shoutout_concepts:
            # Promotion resolution is executable only after the canonical
            # parser has recognized an explicit command shape.
            return None
        tokens = normalized.split()
        if not tokens:
            return None
        concept_indexes = [idx for idx, token in enumerate(tokens) if token in self.shoutout_concepts]
        if not concept_indexes:
            return None
        concept_index = concept_indexes[0]
        concept = tokens[concept_index]
        if concept == "so" and not re.search(r"\b(?:so|s\s*o|dale\s+so|haz\s+so)\s+(?:a|al|to)\b", raw_text, flags=re.IGNORECASE):
            return None
        target_text = self._extract_target(tokens, concept_index, raw_text)
        confidence = 0.92 if target_text else 0.88
        return StreamIntentCandidate(
            "twitch_shoutout",
            confidence,
            entities={"target_text": target_text or ""},
            reason="shoutout_concept",
        )

    def parse_promotion_request(self, text: str, *, source: str = "owner_stt_direct") -> PromotionRequest | None:
        raw = str(text or "").strip()
        print(f"[HEBE][PROMOTION_PARSE_ATTEMPT] raw={raw!r} source={source}", flush=True)
        if not raw:
            print("[HEBE][PROMOTION_PARSE_RESULT] command_detected=false target_phrase='' reason=empty", flush=True)
            return None
        command = self.wake_prefix_re.sub("", raw).strip()
        normalized_command = self.normalize(command)
        for pattern_name, pattern in self.promotion_patterns:
            match = pattern.search(command) or pattern.search(normalized_command)
            if not match:
                continue
            target_raw = str(match.group(1) or "").strip(" ,.;:")
            target, trailing = self._strip_promotion_trailing_banter(target_raw)
            target = self._strip_promotion_target_filler(target)
            print(
                "[HEBE][PROMOTION_PARSE] "
                f"raw={raw!r} command_pattern={pattern_name} target_phrase={target!r} "
                f"stripped_prefix={command[:match.start()].strip()!r} stripped_suffix={trailing!r}",
                flush=True,
            )
            print(
                "[HEBE][PROMOTION_PARSE_RESULT] "
                f"command_detected=true target_phrase={target!r} reason={'parsed' if target else 'no_target'}",
                flush=True,
            )
            return PromotionRequest(
                raw_text=raw,
                target_phrase=target,
                stripped_trailing_text=trailing,
                requested_by="owner",
                source=source,
                confidence=0.96 if target else 0.88,
                target_detected=bool(target),
                reason="parsed" if target else "missing_target",
            )
        if self._looks_like_missing_target_promotion_command(raw):
            print("[HEBE][PROMOTION_PARSE_RESULT] command_detected=true target_detected=false target_phrase='' reason=missing_target", flush=True)
            return PromotionRequest(
                raw_text=raw, target_phrase="", requested_by="owner", source=source, confidence=0.88,
                target_detected=False, reason="missing_target",
            )
        promo_language = bool(set(normalized_command.split()) & (self.shoutout_concepts | {"promos"}))
        reason = "meta_or_no_explicit_command" if promo_language else "not_promotion_language"
        print(
            f"[HEBE][PROMOTION_PARSE_RESULT] command_detected=false target_phrase='' reason={reason}",
            flush=True,
        )
        return None

    def _looks_like_broken_promotion_command(self, normalized: str) -> bool:
        tokens = set(str(normalized or "").split())
        return bool(tokens & self.shoutout_concepts) and bool(tokens & {"haz", "hazle", "dale", "tira", "shoutout", "so"})

    def _looks_like_missing_target_promotion_command(self, text: str) -> bool:
        command = self.wake_prefix_re.sub("", str(text or "").strip()).strip()
        return bool(re.search(
            r"^(?:(?:haz(?:le)?|dale|tira)\s+(?:una?\s+)?(?:promo|shoutout|so)|(?:promo|shoutout))\s*$",
            command,
            flags=re.IGNORECASE,
        ))

    def _strip_promotion_target_filler(self, target: str) -> str:
        value = str(target or "").strip(" @,.;:")
        value = re.sub(r"^(?:a|al|a\s+la|al\s+canal\s+de|canal\s+de)\s+", "", value, flags=re.IGNORECASE).strip(" @,.;:")
        value = re.sub(r"\b(?:haz|hazle|dale|tira|promo|promocion|shoutout|so)\b", " ", value, flags=re.IGNORECASE)
        value = re.sub(r"\s+", " ", value).strip(" @,.;:")
        return value

    def _strip_promotion_trailing_banter(self, target_raw: str) -> tuple[str, str]:
        original = str(target_raw or "").strip(" ,.;:")
        if not original:
            return "", ""
        normalized = self.normalize(original)
        cut_words: int | None = None
        for pattern in self.trailing_banter_patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            prefix = normalized[: match.start()].strip()
            cut_words = len(prefix.split()) if prefix else 0
            break
        words = original.split()
        if cut_words is not None:
            target = " ".join(words[:cut_words]).strip(" ,.;:")
            trailing = " ".join(words[cut_words:]).strip(" ,.;:")
            return target, trailing
        for separator in (",", ";", " - ", " -- "):
            if separator in original:
                target, trailing = original.split(separator, 1)
                return target.strip(" ,.;:"), trailing.strip(" ,.;:")
        return original, ""

    def _extract_target(self, tokens: list[str], concept_index: int, raw_text: str) -> str:
        tail = tokens[concept_index + 1 :]
        for idx, token in enumerate(tail):
            if token in self.target_prepositions:
                return self._raw_tail_after_token(raw_text, tail[idx + 1 :]) or " ".join(tail[idx + 1 :]).strip()
        if tail and tail[0] not in self.filler:
            return self._raw_tail_after_token(raw_text, tail) or " ".join(tail).strip()
        if len(tail) > 1:
            cleaned = [token for token in tail if token not in self.filler]
            return self._raw_tail_after_token(raw_text, cleaned) or " ".join(cleaned).strip()
        return ""

    def _parse_chat_message(self, normalized: str, raw_text: str) -> StreamIntentCandidate | None:
        tokens = normalized.split()
        token_set = set(tokens)
        if not tokens or not (token_set & self.chat_targets) or not (token_set & self.chat_message_concepts):
            return None
        message = self._extract_chat_message(tokens, raw_text)
        return StreamIntentCandidate(
            "stream_chat_message",
            0.9 if message else 0.78,
            entities={"message": message},
            reason="chat_message_concept",
        )

    def _extract_chat_message(self, tokens: list[str], raw_text: str) -> str:
        normalized = " ".join(tokens)
        split_markers = (" que ", ":", " - ")
        raw = str(raw_text or "").strip()
        raw_lower = self.normalize(raw)
        for marker in split_markers:
            if marker.strip() in {":", "-"} and marker in raw:
                tail = raw.split(marker, 1)[1].strip()
                if tail:
                    return tail
            marker_norm = marker.strip()
            if marker_norm and f" {marker_norm} " in f" {normalized} ":
                tail_norm = normalized.split(f" {marker_norm} ", 1)[1].strip()
                if raw and tail_norm:
                    raw_tokens = raw.split()
                    for idx, token in enumerate(raw_tokens):
                        if self.normalize(" ".join(raw_tokens[idx:])).startswith(tail_norm):
                            return " ".join(raw_tokens[idx:]).strip(" ,.;:")
                    return tail_norm
        if "chat" in tokens:
            idx = tokens.index("chat")
            tail = tokens[idx + 1 :]
            while tail and tail[0] in {"que", "de", "al", "a"}:
                tail = tail[1:]
            if tail:
                return self._raw_tail_after_token(raw, tail) or " ".join(tail).strip()
        return ""

    def _raw_tail_after_token(self, raw_text: str, normalized_tail: list[str]) -> str:
        if not raw_text or not normalized_tail:
            return ""
        first = normalized_tail[0]
        raw_tokens = str(raw_text or "").strip().split()
        for index, token in enumerate(raw_tokens):
            if self.normalize(token) == first:
                return " ".join(raw_tokens[index:]).strip(" ,.;:")
        return ""

    def normalize(self, text: str) -> str:
        lowered = str(text or "").strip().lower()
        lowered = "".join(ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch))
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() or ch == "_" else " " for ch in lowered)
        cleaned = cleaned.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        cleaned = cleaned.replace("ü", "u").replace("ñ", "n")
        return re.sub(r"\s+", " ", cleaned).strip()
