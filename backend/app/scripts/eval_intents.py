from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.llm.ollama_intent_client import OllamaIntentClient
from app.orchestrator.intents.resolver import IntentResolver


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "app" / "data" / "intent_eval.jsonl"


@dataclass(slots=True)
class EvalRow:
    text: str
    intent: str
    slots: dict[str, Any]


class DummyInput:
    def __init__(self, text: str) -> None:
        self.text = text


SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "chat",
                "open_app",
                "close_window",
                "set_volume",
                "play_music",
                "pause_music",
                "shutdown_pc",
                "restart_pc",
                "sleep_mode",
            ],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "slots": {
            "type": "object",
            "additionalProperties": True,
        },
    },
    "required": ["intent", "confidence", "slots"],
}


def load_dataset(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}") from exc

            text = data.get("text")
            intent = data.get("intent")
            slots = data.get("slots", {})

            if not isinstance(text, str) or not isinstance(intent, str) or not isinstance(slots, dict):
                raise ValueError(f"Invalid row at line {line_number}: {data}")

            rows.append(EvalRow(text=text, intent=intent, slots=slots))

    return rows


def compare_slots(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return expected == actual


def main() -> int:
    rows = load_dataset(DATASET_PATH)

    client = OllamaIntentClient(
        model="hebe-intent",
        base_url="http://127.0.0.1:11434",
        timeout=20.0,
    )

    resolver = IntentResolver(llm=client)

    total = 0
    intent_ok = 0
    full_ok = 0
    rule_hits = 0
    llm_hits = 0
    failures: list[dict[str, Any]] = []

    for row in rows:
        total += 1
        result = resolver.resolve(DummyInput(row.text))

        got_intent = result.intent or ""
        got_slots = result.slots or {}
        source = getattr(result, "source", None) or "unknown"

        if source == "rules":
            rule_hits += 1
        elif source == "llm":
            llm_hits += 1

        is_intent_ok = got_intent == row.intent
        is_full_ok = is_intent_ok and compare_slots(row.slots, got_slots)

        if is_intent_ok:
            intent_ok += 1

        if is_full_ok:
            full_ok += 1
        else:
            failures.append(
                {
                    "text": row.text,
                    "expected_intent": row.intent,
                    "got_intent": got_intent,
                    "expected_slots": row.slots,
                    "got_slots": got_slots,
                    "source": source,
                    "confidence": getattr(result, "confidence", None),
                }
            )

    print("=" * 72)
    print("HEBE INTENT EVAL")
    print("=" * 72)
    print(f"dataset:         {DATASET_PATH}")
    print(f"total:           {total}")
    print(f"intent accuracy: {intent_ok}/{total} = {intent_ok / total:.2%}")
    print(f"full accuracy:   {full_ok}/{total} = {full_ok / total:.2%}")
    print(f"rules used:      {rule_hits}")
    print(f"llm used:        {llm_hits}")
    print()

    if failures:
        print("FAILURES")
        print("-" * 72)
        for failure in failures[:25]:
            print(f"TEXT:      {failure['text']}")
            print(f"EXPECTED:  intent={failure['expected_intent']} slots={failure['expected_slots']}")
            print(f"GOT:       intent={failure['got_intent']} slots={failure['got_slots']}")
            print(f"SOURCE:    {failure['source']} confidence={failure['confidence']}")
            print("-" * 72)

        if len(failures) > 25:
            print(f"... and {len(failures) - 25} more failures")
            print("-" * 72)
    else:
        print("No failures.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())