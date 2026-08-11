from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from app.replay.scenario import ScenarioAssertion


@dataclass(frozen=True, slots=True)
class AssertionResult:
    passed: bool
    assertion: str
    path: str
    expected: Any
    actual: Any
    description: str = ""
    future_phase: str = ""
    skipped: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TOKEN = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")


def resolve_path(root: Any, path: str) -> tuple[bool, Any]:
    if not path:
        return True, root
    current = root
    for match in _TOKEN.finditer(path):
        key, index = match.groups()
        try:
            if index is not None:
                current = current[int(index)]
            elif isinstance(current, dict):
                if key not in current:
                    return False, None
                current = current[key]
            else:
                current = getattr(current, key)
        except (IndexError, KeyError, TypeError, AttributeError):
            return False, None
    return True, current


def evaluate(assertion: ScenarioAssertion, state: dict[str, Any]) -> AssertionResult:
    if assertion.future_phase:
        return AssertionResult(
            passed=True,
            assertion=assertion.assertion,
            path=assertion.path,
            expected=assertion.expected,
            actual=None,
            description=assertion.description,
            future_phase=assertion.future_phase,
            skipped=True,
            reason=f"pending_future_phase:{assertion.future_phase}",
        )
    exists, actual = resolve_path(state, assertion.path)
    kind = assertion.assertion
    passed = False
    reason = ""
    if kind in {"equals", "path_equals"}:
        passed = exists and actual == assertion.expected
    elif kind == "exists":
        passed = exists and actual is not None
    elif kind == "absent":
        passed = not exists or actual is None
    elif kind in {"count", "collection_count"}:
        passed = exists and hasattr(actual, "__len__") and len(actual) == int(assertion.count or 0)
    elif kind in {"contains", "contains_matching"}:
        passed = exists and isinstance(actual, list) and any(_matches(item, assertion.matching) for item in actual)
    elif kind in {"no_match", "no_item_matches"}:
        passed = exists and isinstance(actual, list) and not any(_matches(item, assertion.matching) for item in actual)
    elif kind == "exactly_once":
        passed = exists and isinstance(actual, list) and sum(1 for item in actual if _matches(item, assertion.matching)) == 1
    elif kind == "zero_external_calls":
        passed = exists and isinstance(actual, list) and len(actual) == 0
    else:
        reason = f"unsupported_assertion:{kind}"
    if not passed and not reason:
        reason = "assertion_mismatch" if exists else "path_missing"
    return AssertionResult(
        passed=passed,
        assertion=kind,
        path=assertion.path,
        expected=assertion.expected if assertion.expected is not None else assertion.matching if assertion.matching else assertion.count,
        actual=actual,
        description=assertion.description,
        reason=reason,
    )


def _matches(value: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    for path, wanted in expected.items():
        exists, actual = resolve_path(value, path)
        if not exists or actual != wanted:
            return False
    return True
