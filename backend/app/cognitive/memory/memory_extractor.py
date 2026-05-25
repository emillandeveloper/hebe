from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.services import db_sqlite


ALLOWED_KINDS = {
    "preference",
    "leo_fact",
    "hebe_identity",
    "project_fact",
    "stream_fact",
    "habit",
    "misc",
}
ALLOWED_STORE_AS = {"fact", "chunk", "both"}


@dataclass(slots=True)
class StoredMemoryResult:
    fact_ids: list[int]
    chunk_ids: list[int]
    skipped: int = 0


class MemoryExtractor:
    """
    Extracts stable long-term memories from a normal private conversation turn.

    This service is intentionally conservative. It prefers storing nothing over
    turning casual chat into permanent memory.
    """

    def __init__(self, intent_model: Any | None = None):
        self.intent_model = intent_model

    def extract(
        self,
        *,
        user_text: str,
        assistant_reply: str,
    ) -> dict[str, Any]:
        user_text = (user_text or "").strip()
        assistant_reply = (assistant_reply or "").strip()
        if not user_text or not assistant_reply:
            return {"memories": []}

        if self.intent_model is not None and hasattr(self.intent_model, "chat_structured"):
            try:
                return self._extract_with_model(user_text=user_text, assistant_reply=assistant_reply)
            except Exception as exc:
                print(f"[HEBE][MEMORY_EXTRACT] structured extraction failed: {exc!r}", flush=True)

        return self._extract_with_rules(user_text=user_text)

    def extract_and_store(
        self,
        *,
        user_text: str,
        assistant_reply: str,
        source: str = "ui",
    ) -> StoredMemoryResult:
        extracted = self.extract(user_text=user_text, assistant_reply=assistant_reply)
        memories = extracted.get("memories") or []

        fact_ids: list[int] = []
        chunk_ids: list[int] = []
        skipped = 0

        for raw in memories:
            item = self._normalize_memory(raw)
            if item is None:
                skipped += 1
                continue

            store_as = item["store_as"]
            kind = item["kind"]
            subject = item["subject"]
            text = item["text"]
            payload = {
                "kind": kind,
                "subject": subject,
                "text": text,
                "tags": item["tags"],
                "source": source,
                "entity_id": item.get("entity_id"),
                "entity_type": item.get("entity_type"),
                "aliases": item.get("aliases"),
                "source_context": item.get("source_context") or source,
            }

            if store_as in {"fact", "both"}:
                fact_id, created = db_sqlite.upsert_memory_fact(
                    kind=kind,
                    subject=subject,
                    payload=payload,
                    source_text=text,
                    confidence=item["confidence"],
                    active=True,
                )
                fact_ids.append(fact_id)
                print(
                    f"[HEBE][MEMORY_EXTRACT] {'inserted' if created else 'updated'} "
                    f"fact id={fact_id} kind={kind!r} subject={subject!r}",
                    flush=True,
                )

            if store_as in {"chunk", "both"}:
                try:
                    from app.cognitive.memory.memory_store import add_chunk_if_new

                    chunk_id, created = add_chunk_if_new(
                        text=text,
                        kind=kind,
                        subject=subject,
                        source_session=source,
                        importance=item["importance"],
                        tags={
                            "source": source,
                            "tags": item["tags"],
                            "entity_id": item.get("entity_id"),
                            "entity_type": item.get("entity_type"),
                            "aliases": item.get("aliases"),
                            "source_context": item.get("source_context") or source,
                        },
                    )
                    if chunk_id is not None:
                        chunk_ids.append(chunk_id)
                    print(
                        f"[HEBE][MEMORY_EXTRACT] {'inserted' if created else 'skipped duplicate'} "
                        f"chunk id={chunk_id} kind={kind!r} subject={subject!r}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[HEBE][MEMORY_EXTRACT] chunk write failed: {exc!r}", flush=True)

        if not fact_ids and not chunk_ids:
            print(f"[HEBE][MEMORY_EXTRACT] stored nothing skipped={skipped}", flush=True)

        return StoredMemoryResult(fact_ids=fact_ids, chunk_ids=chunk_ids, skipped=skipped)

    def _extract_with_model(self, *, user_text: str, assistant_reply: str) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": sorted(ALLOWED_KINDS),
                            },
                            "subject": {"type": "string"},
                            "text": {"type": "string"},
                            "confidence": {"type": "number"},
                            "importance": {"type": "number"},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "store_as": {
                                "type": "string",
                                "enum": sorted(ALLOWED_STORE_AS),
                            },
                            "entity_id": {"type": "string"},
                            "entity_type": {"type": "string"},
                            "aliases": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "source_context": {"type": "string"},
                        },
                        "required": [
                            "kind",
                            "subject",
                            "text",
                            "confidence",
                            "importance",
                            "tags",
                            "store_as",
                        ],
                    },
                }
            },
            "required": ["memories"],
        }

        system = (
            "You extract long-term memory for Hebe, Leo's local AI companion.\n"
            "Return only stable, useful memories. Do not store casual one-off chat, "
            "temporary mood, noisy Twitch messages, or jokes unless they define channel lore.\n"
            "Store corrections from Leo about Hebe's identity or voice.\n"
            "Prefer 0 memories when unsure."
        )
        user = (
            "Latest private conversation turn:\n\n"
            f"Leo/user: {user_text}\n\n"
            f"Hebe: {assistant_reply}\n\n"
            "Extract stable memories using the requested schema."
        )
        result = self.intent_model.chat_structured(
            system_prompt=system,
            user_prompt=user,
            schema=schema,
            temperature=0.0,
        )
        if not isinstance(result.get("memories"), list):
            return {"memories": []}
        return result

    def _extract_with_rules(self, *, user_text: str) -> dict[str, Any]:
        text = " ".join(user_text.strip().split())
        low = text.lower()
        memories: list[dict[str, Any]] = []

        if not re.search(r"\b(recuerda|remember|prefiero|i prefer|quiero que|should|deberias|deberías)\b", low):
            return {"memories": []}

        if "femenin" in low or "female" in low:
            memories.append(
                {
                    "kind": "hebe_identity",
                    "subject": "hebe_voice",
                    "text": "Hebe should speak about herself using feminine grammatical form.",
                    "confidence": 0.95,
                    "importance": 0.9,
                    "tags": ["identity", "voice", "feminine"],
                    "store_as": "both",
                }
            )

        if "español de españa" in low or "espana" in low or "peninsular" in low:
            memories.append(
                {
                    "kind": "preference",
                    "subject": "leo.language.spanish",
                    "text": "Leo prefers Hebe to use Spanish from Spain / peninsular Spanish when speaking Spanish.",
                    "confidence": 0.95,
                    "importance": 0.85,
                    "tags": ["language", "spanish", "spain"],
                    "store_as": "both",
                }
            )

        if "english" in low and ("natural" in low or "inglés" in low or "ingles" in low):
            memories.append(
                {
                    "kind": "preference",
                    "subject": "leo.language.english",
                    "text": "Leo prefers Hebe to use natural English when speaking English.",
                    "confidence": 0.9,
                    "importance": 0.75,
                    "tags": ["language", "english"],
                    "store_as": "both",
                }
            )

        return {"memories": memories}

    def _normalize_memory(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None

        kind = str(raw.get("kind") or "").strip()
        subject = str(raw.get("subject") or "").strip()
        text = " ".join(str(raw.get("text") or "").strip().split())
        if kind not in ALLOWED_KINDS or not subject or not text:
            return None

        confidence = self._clamp_float(raw.get("confidence"), 0.0, 1.0, default=0.5)
        importance = self._clamp_float(raw.get("importance"), 0.0, 1.0, default=0.5)
        if confidence < 0.65 or importance < 0.45:
            return None

        tags_raw = raw.get("tags")
        tags = [str(t).strip() for t in tags_raw if str(t).strip()] if isinstance(tags_raw, list) else []
        store_as = str(raw.get("store_as") or "fact").strip()
        if store_as not in ALLOWED_STORE_AS:
            store_as = "fact"
        entity_id = self._clean_optional_string(raw.get("entity_id")) or self._infer_entity_id(subject, text)
        entity_type = self._clean_optional_string(raw.get("entity_type"))
        source_context = self._clean_optional_string(raw.get("source_context"))
        aliases_raw = raw.get("aliases")
        aliases = (
            [str(alias).strip() for alias in aliases_raw if str(alias).strip()]
            if isinstance(aliases_raw, list)
            else []
        )

        return {
            "kind": kind,
            "subject": subject[:120],
            "text": text[:1000],
            "confidence": confidence,
            "importance": importance,
            "tags": tags[:12],
            "store_as": store_as,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "aliases": aliases[:12],
            "source_context": source_context,
        }

    def _clean_optional_string(self, value: Any) -> str | None:
        text = str(value or "").strip()
        return text or None

    def _infer_entity_id(self, subject: str, text: str) -> str | None:
        low = f"{subject} {text}".lower()
        if "jotunbot" in low or "jotun bot" in low:
            return "jotun_bot"
        if "jotun" in low and any(word in low for word in ("bot", "comando", "comandos", "twitch", "chat")):
            return "jotun_bot"
        if "jotun" in low and any(word in low for word in ("perro", "dog", "mascota")):
            return "jotun_dog"
        if "hebe" in low:
            return "hebe_ai"
        if "leo" in low:
            return "leo"
        return None

    def _clamp_float(self, value: Any, low: float, high: float, *, default: float) -> float:
        try:
            number = float(value)
        except Exception:
            return default
        return max(low, min(high, number))
