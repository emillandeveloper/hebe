from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


class StreamDatasetLogger:
    """
    Logger JSONL para construir dataset de Hebe a partir de actividad real.

    Guarda una línea por ejemplo generado. No intenta decidir si el ejemplo es
    bueno o malo: deja campos de curación (`approved`, `corrected_response`) para
    que luego puedas revisar/editar sin perder el dato original.

    Ruta por defecto, relativa al cwd del backend:
      data/dataset/hebe_stream_replies.jsonl
    """

    def __init__(self) -> None:
        self.enabled = _env_bool("HEBE_DATASET_LOG_ENABLED", True)
        self.path = Path(
            os.getenv(
                "HEBE_DATASET_LOG_PATH",
                "data/dataset/hebe_stream_replies.jsonl",
            )
        )
        self.max_message_chars = int(os.getenv("HEBE_DATASET_MAX_MESSAGE_CHARS", "500"))
        self.max_response_chars = int(os.getenv("HEBE_DATASET_MAX_RESPONSE_CHARS", "500"))
        self.max_recent_items = int(os.getenv("HEBE_DATASET_MAX_RECENT_ITEMS", "8"))

    def log_twitch_chat_react(
        self,
        *,
        trace_id: str,
        payload: dict[str, Any],
        chatter_clean: str,
        is_broadcaster: bool,
        raw_response: str,
        cleaned_response: str,
        helper_hits: list[str],
        retried: bool,
        salvaged: bool,
        model_meta: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        message = str(payload.get("message_text") or "")
        display_name = str(payload.get("display_name") or payload.get("user_login") or "")
        user_login = str(payload.get("user_login") or "")

        row: dict[str, Any] = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": "twitch",
            "event_type": "twitch_chat_react",
            "trace_id": trace_id,
            "user_login": user_login,
            "display_name": display_name,
            "chatter_clean": chatter_clean,
            "is_broadcaster": bool(is_broadcaster),
            "message": self._clip(message, self.max_message_chars),
            "response": self._clip(cleaned_response, self.max_response_chars),
            "raw_response": self._clip(raw_response, self.max_response_chars),
            "helper_hits": helper_hits,
            "retried": bool(retried),
            "salvaged": bool(salvaged),
            "model": model_meta or {},
            "recent_chat": self._clean_recent(payload.get("recent_chat") or []),
            "curation": {
                "approved": None,
                "corrected_response": None,
                "notes": None,
                "tags": [],
            },
        }

        self._append(row)


    def update_curation(
        self,
        *,
        trace_id: str,
        status: str,
        corrected_response: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        """
        Actualiza la curación de una fila JSONL por trace_id.

        status:
          - ok                -> approved=True
          - no_ok             -> approved=False
          - needs_enhancement -> approved=False, marcado para mejorar
        """
        trace_id = str(trace_id or "").strip()
        status = str(status or "").strip().lower()
        if not trace_id:
            print("[HEBE][DATASET][CURATE] missing trace_id", flush=True)
            return False

        allowed = {"ok", "no_ok", "needs_enhancement"}
        if status not in allowed:
            print(f"[HEBE][DATASET][CURATE] invalid status={status!r}", flush=True)
            return False

        if not self.path.exists():
            print(f"[HEBE][DATASET][CURATE] file not found path={str(self.path)!r}", flush=True)
            return False

        updated = False
        rows: list[str] = []

        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    raw_line = line.rstrip("\n")
                    if not raw_line.strip():
                        continue

                    try:
                        row = json.loads(raw_line)
                    except Exception:
                        rows.append(raw_line)
                        continue

                    if str(row.get("trace_id") or "") == trace_id:
                        curation = row.get("curation")
                        if not isinstance(curation, dict):
                            curation = {}

                        existing_tags = curation.get("tags")
                        if not isinstance(existing_tags, list):
                            existing_tags = []

                        merged_tags = list(dict.fromkeys([str(t) for t in existing_tags if t] + (tags or [])))
                        if status == "needs_enhancement" and "needs_enhancement" not in merged_tags:
                            merged_tags.append("needs_enhancement")

                        curation.update(
                            {
                                "status": status,
                                "approved": True if status == "ok" else False,
                                "corrected_response": corrected_response if corrected_response is not None else curation.get("corrected_response"),
                                "notes": notes if notes is not None else curation.get("notes"),
                                "tags": merged_tags,
                                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        row["curation"] = curation
                        updated = True
                        raw_line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))

                    rows.append(raw_line)

            if not updated:
                print(f"[HEBE][DATASET][CURATE] trace_id not found trace={trace_id!r}", flush=True)
                return False

            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for row_line in rows:
                    f.write(row_line)
                    f.write("\n")
            tmp.replace(self.path)

            print(f"[HEBE][DATASET][CURATE] updated trace={trace_id!r} status={status!r}", flush=True)
            return True

        except Exception as exc:
            print(f"[HEBE][DATASET][CURATE][ERROR] {exc!r}", flush=True)
            return False

    def _append(self, row: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                f.write("\n")
        except Exception as exc:
            print(f"[HEBE][DATASET][ERROR] could not write dataset row: {exc!r}", flush=True)
            return

        print(f"[HEBE][DATASET] wrote twitch_chat_react path={str(self.path)!r}", flush=True)

    def _clean_recent(self, recent: Any) -> list[dict[str, str]]:
        if not isinstance(recent, list):
            return []

        cleaned: list[dict[str, str]] = []
        for item in recent[-self.max_recent_items :]:
            if not isinstance(item, dict):
                continue
            cleaned.append(
                {
                    "display_name": self._clip(str(item.get("display_name") or ""), 80),
                    "text": self._clip(str(item.get("text") or ""), self.max_message_chars),
                }
            )
        return cleaned

    def _clip(self, text: str, max_chars: int) -> str:
        text = str(text or "")
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"
