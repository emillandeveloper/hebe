from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from app.cognitive.cognitive_router import CognitiveRouter


@dataclass(frozen=True, slots=True)
class RouterCase:
    text: str
    expected_intent: str
    source: str = "ui"
    authority: str = "owner"
    addressed_to_hebe: bool = True


CASES = (
    RouterCase("Hebe, abre OBS", "command_open_app"),
    RouterCase("abre una aplicación que no existe", "command_open_app"),
    RouterCase("He dicho 'abre Steam' tres veces", "unknown_chat"),
    RouterCase("y entonces abre la puerta y aparece el jefe", "stream_context_update", "ambient_stt", "ambient", False),
)


def evaluate(cases: tuple[RouterCase, ...] = CASES) -> list[dict[str, str]]:
    router = CognitiveRouter()
    failures: list[dict[str, str]] = []
    for case in cases:
        decision = router.route(SimpleNamespace(
            input_text=case.text,
            source=case.source,
            authority=case.authority,
            addressed_to_hebe=case.addressed_to_hebe,
            state_snapshot={},
            firewall_decision="allow",
            stream_is_live=True,
            route_hints=[],
            internal_event=None,
        ))
        if decision.intent != case.expected_intent:
            failures.append({
                "text": case.text,
                "expected": case.expected_intent,
                "actual": decision.intent,
            })
    return failures


def main() -> int:
    failures = evaluate()
    if failures:
        for failure in failures:
            print(
                "FAIL "
                f"expected={failure['expected']} actual={failure['actual']} "
                f"text={failure['text']!r}"
            )
        return 1
    print(f"CognitiveRouter evaluation passed: {len(CASES)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
